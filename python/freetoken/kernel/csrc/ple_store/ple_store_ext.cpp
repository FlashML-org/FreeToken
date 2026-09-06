// Disk-backed PLE row store: rows read straight from the checkpoint's fp8 shard tensors
// through an extent table (PleRowSource in ple_ssd.py). Engine-thread only, no locks.
// Duplicate rows in one fill dedup into ONE batched read round; no RAM cache and no
// per-sequence state. Hash reference: tests/models/qwen4_exp/test_ple_disk.py.
// Platform seams: TableFile (O_DIRECT+pread; Win: NO_BUFFERING), BatchReader (io_uring,
// pread-pool fallback = the portable shape), cumemop_* (dlopen libcuda; Win: nvcuda).

#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__linux__) && __has_include(<linux/io_uring.h>)
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#define PLE_HAS_IO_URING 1
#else
#define PLE_HAS_IO_URING 0
#endif

#include <torch/extension.h>

namespace py = pybind11;

namespace {

constexpr int64_t kPage = 4096;
constexpr int64_t kSpanMax = 2 * kPage;  // a row is <= one page, so it spans at most two
constexpr unsigned kBatchEntries = 64;
// fio on this class of disk: pread saturates at ~16 threads; more only adds latency
constexpr unsigned kReaderThreads = 16;

// ---- portability shims ----

void release_store_i64(int64_t *ptr, int64_t value) {
  __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
}

uint8_t *page_aligned_alloc(size_t bytes) {
  void *p = nullptr;
  if (posix_memalign(&p, kPage, bytes) != 0) throw std::bad_alloc();
  return static_cast<uint8_t *>(p);
}

// Stream memops for the flag-sync fast path, resolved from the driver at runtime.
using CuMemOp64Fn = int (*)(void *stream, unsigned long long addr, unsigned long long value,
                            unsigned int flags);
CuMemOp64Fn g_cu_write64 = nullptr;
CuMemOp64Fn g_cu_wait64 = nullptr;
constexpr unsigned kCuWaitValueGeq = 0x0;
constexpr unsigned kCuWriteDefault = 0x0;

bool cumemop_resolve() {
  static bool resolved = [] {
    void *h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (h == nullptr) h = dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
    if (h == nullptr) return false;
    g_cu_write64 = reinterpret_cast<CuMemOp64Fn>(dlsym(h, "cuStreamWriteValue64_v2"));
    if (g_cu_write64 == nullptr)
      g_cu_write64 = reinterpret_cast<CuMemOp64Fn>(dlsym(h, "cuStreamWriteValue64"));
    g_cu_wait64 = reinterpret_cast<CuMemOp64Fn>(dlsym(h, "cuStreamWaitValue64_v2"));
    if (g_cu_wait64 == nullptr)
      g_cu_wait64 = reinterpret_cast<CuMemOp64Fn>(dlsym(h, "cuStreamWaitValue64"));
    return g_cu_write64 != nullptr && g_cu_wait64 != nullptr;
  }();
  return resolved;
}

int memop_write(uintptr_t stream, uintptr_t addr, uint64_t value) {
  if (!cumemop_resolve()) return -1;
  return g_cu_write64(reinterpret_cast<void *>(stream), addr, value, kCuWriteDefault);
}

int memop_wait_geq(uintptr_t stream, uintptr_t addr, uint64_t value) {
  if (!cumemop_resolve()) return -1;
  return g_cu_wait64(reinterpret_cast<void *>(stream), addr, value, kCuWaitValueGeq);
}

// WAIT(>=1) then RESET: resetting first would race a fast host signal and deadlock the stream.
void memop_wait_reset(uintptr_t stream, uintptr_t flag_addr) {
  if (memop_wait_geq(stream, flag_addr, 1) != 0 || memop_write(stream, flag_addr, 0) != 0)
    throw std::runtime_error("stream memops rejected in capture; set FREETOKEN_PLE_SYNC=gate");
}

void signal_flag(uintptr_t flag_addr) {
  release_store_i64(reinterpret_cast<int64_t *>(flag_addr), 1);
}

// ---- TableFile: platform seam for on-disk files ----

// Read at least need bytes; len is the larger aligned span the request must keep.
// Resuming past need is not safe: a read that crossed EOF ends at an unaligned offset.
void pread_min(int fd, uint8_t *buf, int64_t len, int64_t need, int64_t off) {
  int64_t done = 0;
  while (done < need) {
    ssize_t got = ::pread(fd, buf + done, len - done, off + done);
    if (got < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(std::string("pread: ") + std::strerror(errno));
    }
    if (got == 0) break;
    done += got;
  }
  if (done < need)
    throw std::runtime_error("short read at offset " + std::to_string(off) + ": got " +
                             std::to_string(done) + " of " + std::to_string(need));
}

class TableFile {
 public:
  explicit TableFile(const std::string &path) {
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    direct_ = fd_ >= 0;
    if (fd_ < 0) {
      fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
      direct_ = false;
    }
    if (fd_ < 0) throw std::runtime_error(path + ": " + std::strerror(errno));
    struct stat st{};
    if (fstat(fd_, &st) != 0) {
      ::close(fd_);
      throw std::runtime_error(path + ": fstat: " + std::strerror(errno));
    }
    size_ = st.st_size;
    if (!direct_) posix_fadvise(fd_, 0, 0, POSIX_FADV_RANDOM);
  }

  ~TableFile() {
    if (fd_ >= 0) ::close(fd_);
  }

  TableFile(const TableFile &) = delete;
  TableFile &operator=(const TableFile &) = delete;

  bool direct_io() const { return direct_; }
  int native_fd() const { return fd_; }
  int64_t size() const { return size_; }

  // keep buffered fallback reads out of the page cache; a no-op under direct I/O
  void discard_cache(int64_t off, int64_t len) const {
    if (!direct_) posix_fadvise(fd_, off, len, POSIX_FADV_DONTNEED);
  }

 private:
  int fd_ = -1;
  bool direct_ = false;
  int64_t size_ = 0;
};

// ---- BatchReader: platform seam for parallel positioned reads ----

// Pipelined: at most capacity() reads in flight; wait_one() returns a finished tag to refill.
class BatchReader {
 public:
  virtual ~BatchReader() = default;
  virtual std::string name() const = 0;
  virtual unsigned capacity() const = 0;
  virtual void submit(unsigned tag, int fd, uint8_t *buf, int64_t len, int64_t need,
                      int64_t off) = 0;
  virtual unsigned wait_one() = 0;
  // reap every in-flight read so stale completions cannot leak into the next fill
  virtual void drain() noexcept = 0;
};

class ThreadPoolBatchReader final : public BatchReader {
 public:
  ThreadPoolBatchReader() {
    // these threads block on I/O, not compute, so the core count is only a default
    unsigned n = std::max(1u, std::min(kReaderThreads, std::thread::hardware_concurrency()));
    if (const char *env = std::getenv("FREETOKEN_PLE_READER_THREADS")) {
      // kBatchEntries is the submit depth, so threads past it never get a read
      const int v = std::atoi(env);
      if (v > 0) n = std::min((unsigned)v, kBatchEntries);
    }
    for (unsigned i = 0; i < n; i++) workers_.emplace_back([this] { work(); });
  }

  ~ThreadPoolBatchReader() override {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    work_cv_.notify_all();
    for (auto &w : workers_) w.join();
  }

  std::string name() const override { return "pread-pool x" + std::to_string(workers_.size()); }
  unsigned capacity() const override { return kBatchEntries; }

  void submit(unsigned tag, int fd, uint8_t *buf, int64_t len, int64_t need,
              int64_t off) override {
    {
      std::lock_guard<std::mutex> lock(mu_);
      queue_.push_back(Req{tag, fd, buf, len, need, off});
      in_flight_++;
    }
    work_cv_.notify_one();
  }

  unsigned wait_one() override {
    std::unique_lock<std::mutex> lock(mu_);
    done_cv_.wait(lock, [this] { return !done_.empty(); });
    Done d = std::move(done_.front());
    done_.pop_front();
    in_flight_--;
    if (!d.error.empty()) throw std::runtime_error(d.error);
    return d.tag;
  }

  void drain() noexcept override {
    // wait out all in-flight reads: a late worker write must not race the slot's reuse
    std::unique_lock<std::mutex> lock(mu_);
    done_cv_.wait(lock, [this] { return done_.size() == in_flight_; });
    in_flight_ = 0;
    done_.clear();
  }

 private:
  struct Req {
    unsigned tag;
    int fd;
    uint8_t *buf;
    int64_t len;
    int64_t need;
    int64_t off;
  };
  struct Done {
    unsigned tag;
    std::string error;
  };

  void work() {
    for (;;) {
      Req r;
      {
        std::unique_lock<std::mutex> lock(mu_);
        work_cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
        if (stop_) return;
        r = queue_.front();
        queue_.pop_front();
      }
      Done d{r.tag, {}};
      try {
        pread_min(r.fd, r.buf, r.len, r.need, r.off);
      } catch (const std::exception &e) {
        d.error = e.what();
      }
      {
        std::lock_guard<std::mutex> lock(mu_);
        done_.push_back(std::move(d));
      }
      done_cv_.notify_one();
    }
  }

  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable work_cv_, done_cv_;
  std::deque<Req> queue_;
  std::deque<Done> done_;
  size_t in_flight_ = 0;
  bool stop_ = false;
};

#if PLE_HAS_IO_URING

// Minimal single-issuer io_uring: submit up to `entries` reads, wait for all.
class IoUringBatchReader final : public BatchReader {
 public:
  IoUringBatchReader() = default;

  bool init(unsigned entries) {
    struct io_uring_params p{};
    fd_ = (int)syscall(__NR_io_uring_setup, entries, &p);
    if (fd_ < 0) return false;
    sq_size_ = p.sq_off.array + p.sq_entries * sizeof(uint32_t);
    cq_size_ = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
    if (p.features & IORING_FEAT_SINGLE_MMAP) sq_size_ = cq_size_ = std::max(sq_size_, cq_size_);
    sq_ptr_ = mmap(nullptr, sq_size_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd_,
                   IORING_OFF_SQ_RING);
    if (sq_ptr_ == MAP_FAILED) return false;
    cq_ptr_ = (p.features & IORING_FEAT_SINGLE_MMAP)
                  ? sq_ptr_
                  : mmap(nullptr, cq_size_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                         fd_, IORING_OFF_CQ_RING);
    if (cq_ptr_ == MAP_FAILED) return false;
    sqes_size_ = p.sq_entries * sizeof(io_uring_sqe);
    sqes_ = (io_uring_sqe *)mmap(nullptr, sqes_size_, PROT_READ | PROT_WRITE,
                                 MAP_SHARED | MAP_POPULATE, fd_, IORING_OFF_SQES);
    if (sqes_ == MAP_FAILED) return false;

    auto at = [&](void *base, uint32_t off) { return (uint8_t *)base + off; };
    sq_tail_ = (uint32_t *)at(sq_ptr_, p.sq_off.tail);
    sq_mask_ = (uint32_t *)at(sq_ptr_, p.sq_off.ring_mask);
    sq_array_ = (uint32_t *)at(sq_ptr_, p.sq_off.array);
    cq_head_ = (uint32_t *)at(cq_ptr_, p.cq_off.head);
    cq_tail_ = (uint32_t *)at(cq_ptr_, p.cq_off.tail);
    cq_mask_ = (uint32_t *)at(cq_ptr_, p.cq_off.ring_mask);
    cqes_ = (io_uring_cqe *)at(cq_ptr_, p.cq_off.cqes);
    entries_ = p.sq_entries;
    lens_.assign(entries_, 0);
    sq_shadow_tail_ = *sq_tail_;
    return true;
  }

  ~IoUringBatchReader() override {
    if (sqes_ && sqes_ != MAP_FAILED) munmap(sqes_, sqes_size_);
    if (cq_ptr_ && cq_ptr_ != MAP_FAILED && cq_ptr_ != sq_ptr_) munmap(cq_ptr_, cq_size_);
    if (sq_ptr_ && sq_ptr_ != MAP_FAILED) munmap(sq_ptr_, sq_size_);
    if (fd_ >= 0) ::close(fd_);
  }

  std::string name() const override { return "io_uring"; }
  unsigned capacity() const override { return entries_; }

  void submit(unsigned tag, int fd, uint8_t *buf, int64_t len, int64_t need,
              int64_t off) override {
    io_uring_sqe *sqe = &sqes_[sq_shadow_tail_ & *sq_mask_];
    std::memset(sqe, 0, sizeof(*sqe));
    sqe->opcode = IORING_OP_READ;
    sqe->fd = fd;
    sqe->addr = (uint64_t)(uintptr_t)buf;
    sqe->len = (uint32_t)len;
    sqe->off = (uint64_t)off;
    sqe->user_data = tag;
    sq_array_[sq_shadow_tail_ & *sq_mask_] = sq_shadow_tail_ & *sq_mask_;
    sq_shadow_tail_++;
    __atomic_store_n(sq_tail_, sq_shadow_tail_, __ATOMIC_RELEASE);
    lens_[tag] = need;
    to_submit_++;
    in_flight_++;
  }

  unsigned wait_one() override {
    for (;;) {
      uint32_t head = *cq_head_;
      uint32_t ctail = __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE);
      if (head != ctail) {
        const io_uring_cqe &cqe = cqes_[head & *cq_mask_];
        const unsigned tag = (unsigned)cqe.user_data;
        const int res = cqe.res;
        __atomic_store_n(cq_head_, head + 1, __ATOMIC_RELEASE);
        in_flight_--;
        if (res < 0)
          throw std::runtime_error(std::string("io_uring read: ") + std::strerror(-res));
        if (res < lens_[tag]) throw std::runtime_error("io_uring short read");
        return tag;
      }
      const unsigned to_submit = to_submit_;
      long rc = syscall(__NR_io_uring_enter, fd_, to_submit, 1, IORING_ENTER_GETEVENTS, nullptr, 0);
      if (rc < 0) {
        if (errno == EINTR) continue;
        throw std::runtime_error(std::string("io_uring_enter: ") + std::strerror(errno));
      }
      // partial submission is legal (signal, transient alloc); the rest stay in the ring
      to_submit_ = to_submit - (unsigned)rc;
    }
  }

  void drain() noexcept override {
    while (in_flight_ > 0) {
      uint32_t head = *cq_head_;
      uint32_t ctail = __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE);
      if (head != ctail) {
        __atomic_store_n(cq_head_, head + 1, __ATOMIC_RELEASE);
        in_flight_--;
        continue;
      }
      const unsigned to_submit = to_submit_;
      long rc = syscall(__NR_io_uring_enter, fd_, to_submit, 1, IORING_ENTER_GETEVENTS, nullptr, 0);
      if (rc < 0) {
        if (errno != EINTR) return;
        continue;
      }
      to_submit_ = to_submit - (unsigned)rc;
    }
  }

 private:
  int fd_ = -1;
  void *sq_ptr_ = nullptr, *cq_ptr_ = nullptr;
  io_uring_sqe *sqes_ = nullptr;
  size_t sq_size_ = 0, cq_size_ = 0, sqes_size_ = 0;
  uint32_t *sq_tail_ = nullptr, *sq_mask_ = nullptr, *sq_array_ = nullptr;
  uint32_t *cq_head_ = nullptr, *cq_tail_ = nullptr, *cq_mask_ = nullptr;
  io_uring_cqe *cqes_ = nullptr;
  unsigned entries_ = 0;
  uint32_t sq_shadow_tail_ = 0;
  unsigned to_submit_ = 0;
  unsigned in_flight_ = 0;
  std::vector<int64_t> lens_;
};

#endif  // PLE_HAS_IO_URING

std::unique_ptr<BatchReader> make_batch_reader(bool use_io_uring) {
#if PLE_HAS_IO_URING
  if (use_io_uring) {
    auto ring = std::make_unique<IoUringBatchReader>();
    if (ring->init(kBatchEntries)) return ring;
  }
#else
  (void)use_io_uring;
#endif
  return std::make_unique<ThreadPoolBatchReader>();
}

// ---- row store (platform-free) ----

int64_t wrap_mul(int64_t a, int64_t b) {
  return (int64_t)((uint64_t)a * (uint64_t)b);
}

int64_t pos_mod(int64_t v, int64_t m) {
  int64_t r = v % m;
  return r < 0 ? r + m : r;
}

class PleStore {
  struct Extent {
    const TableFile *file;
    int64_t base;
  };

 public:
  PleStore(std::vector<std::string> paths, std::vector<int64_t> extent_file,
           std::vector<int64_t> extent_base, int64_t rows_per_extent, int64_t row_bytes,
           int64_t row_stride, std::vector<int64_t> multipliers, std::vector<int64_t> head_vocab_sizes,
           std::vector<int64_t> head_offsets, int64_t eos_token_id, bool use_io_uring)
      : row_bytes_(row_bytes),
        row_stride_(row_stride),
        rows_per_extent_(rows_per_extent),
        mult_(std::move(multipliers)),
        sizes_(std::move(head_vocab_sizes)),
        offsets_(std::move(head_offsets)),
        eos_(eos_token_id) {
    if (mult_.size() != 3 || sizes_.size() != offsets_.size() || sizes_.empty())
      throw std::runtime_error("PLE hash geometry: want 3 multipliers and equal-length head tables");
    if (row_bytes_ > kPage)
      throw std::runtime_error("PLE row_bytes " + std::to_string(row_bytes_) +
                               " exceeds a page; bounce slots assume one-page rows");
    for (const std::string &p : paths)
      files_.push_back(std::make_unique<TableFile>(p));
    const int64_t extent_bytes = (rows_per_extent_ - 1) * row_stride_ + row_bytes_;
    for (size_t e = 0; e < extent_file.size(); e++) {
      const size_t fi = (size_t)extent_file.at(e);
      const int64_t base = extent_base.at(e);
      if (base + extent_bytes > files_.at(fi)->size())
        throw std::runtime_error(paths[fi] + ": extent needs " +
                                 std::to_string(base + extent_bytes) + " bytes, file has " +
                                 std::to_string(files_[fi]->size()));
      extents_.push_back(Extent{files_[fi].get(), base});
    }
    reader_ = make_batch_reader(use_io_uring);
    bounce_ = page_aligned_alloc((size_t)reader_->capacity() * kSpanMax);
  }

  // reader first: a still-running read must not land in freed bounce memory
  ~PleStore() {
    reader_.reset();
    free(bounce_);
  }

  PleStore(const PleStore &) = delete;
  PleStore &operator=(const PleStore &) = delete;

  // Row ids for the token at w[2] with context (w[0], w[1]); mirrors NGramEmbedding.row_ids incl. the eos barrier.
  void hash_rows(const int64_t *w, int64_t *rows) {
    const int64_t prev1 = w[1];
    const int64_t prev2 = prev1 == eos_ ? eos_ : w[0];
    const int64_t bigram = wrap_mul(w[2], mult_[0]) ^ wrap_mul(prev1, mult_[1]);
    const int64_t trigram = bigram ^ wrap_mul(prev2, mult_[2]);
    const size_t half = sizes_.size() / 2;
    for (size_t h = 0; h < sizes_.size(); h++)
      rows[h] = pos_mod(h < half ? bigram : trigram, sizes_[h]) + offsets_[h];
  }

  // Hash and queue one run: tokens_addr holds n+2 ids, the leading two are context. No I/O until flush().
  void stage(uintptr_t tokens_addr, int64_t n, uintptr_t staging_addr) {
    const int64_t *tokens = reinterpret_cast<const int64_t *>(tokens_addr);
    uint8_t *staging = reinterpret_cast<uint8_t *>(staging_addr);
    const size_t heads = sizes_.size();
    std::vector<int64_t> rows(heads);
    for (int64_t i = 0; i < n; i++) {
      hash_rows(tokens + i, rows.data());
      for (size_t h = 0; h < heads; h++)
        request_row(rows[h], staging + ((size_t)i * heads + h) * row_bytes_);
    }
  }

  // One batched disk round for everything staged; signals even when nothing was.
  void flush(uintptr_t signal_addr) {
    flush_pending();
    if (signal_addr) signal_flag(signal_addr);
  }

  std::string io_backend() const {
    size_t direct = 0;
    for (const auto &f : files_) direct += f->direct_io() ? 1 : 0;
    std::string s = reader_->name();
    if (direct == files_.size()) return s + ", O_DIRECT";
    return s + ", buffered " + std::to_string(files_.size() - direct) + "/" +
           std::to_string(files_.size()) + " files";
  }

 private:
  struct Pending {
    const TableFile *file;
    int64_t read_off;
    int64_t read_len;
    int64_t row_off;  // row payload start inside the read buffer
    std::vector<uint8_t *> dsts;
  };

  // Queue dst on this fill's pending batch; duplicate rows fan out from one read.
  void request_row(int64_t row_id, uint8_t *dst) {
    auto pit = pending_index_.find(row_id);
    if (pit != pending_index_.end()) {
      pending_[pit->second].dsts.push_back(dst);
      return;
    }
    const Extent &ext = extents_[row_id / rows_per_extent_];
    const int64_t off = ext.base + (row_id % rows_per_extent_) * row_stride_;
    Pending p{ext.file, off, row_bytes_, 0, {dst}};
    if (ext.file->direct_io()) {
      // full aligned span even past EOF; truncating would break direct-I/O alignment
      p.read_off = off & ~(kPage - 1);
      p.row_off = off - p.read_off;
      p.read_len = ((off + row_bytes_ + kPage - 1) & ~(kPage - 1)) - p.read_off;
    }
    pending_index_.emplace(row_id, pending_.size());
    pending_.push_back(std::move(p));
  }

  // Read every pending row in reader-capacity batches and fan out the copies.
  void flush_pending() {
    if (pending_.empty()) return;
    struct Cleanup {
      PleStore *s;
      ~Cleanup() {
        s->pending_.clear();
        s->pending_index_.clear();
      }
    } cleanup{this};

    const unsigned cap = reader_->capacity();
    const size_t total = pending_.size();
    std::vector<size_t> tag_pending(cap);
    size_t next = 0;
    auto submit_slot = [&](unsigned tag) {
      const Pending &p = pending_[next];
      tag_pending[tag] = next++;
      reader_->submit(tag, p.file->native_fd(), bounce_ + (size_t)tag * kSpanMax, p.read_len,
                      p.row_off + row_bytes_, p.read_off);
    };
    try {
      for (unsigned tag = 0; tag < std::min((size_t)cap, total); tag++) submit_slot(tag);
      for (size_t completed = 0; completed < total; completed++) {
        const unsigned tag = reader_->wait_one();
        const Pending &p = pending_[tag_pending[tag]];
        const uint8_t *row = bounce_ + (size_t)tag * kSpanMax + p.row_off;
        for (uint8_t *dst : p.dsts) std::memcpy(dst, row, row_bytes_);
        p.file->discard_cache(p.read_off, p.read_len);
        if (next < total) submit_slot(tag);
      }
    } catch (...) {
      reader_->drain();
      throw;
    }
  }

  int64_t row_bytes_, row_stride_, rows_per_extent_;
  std::vector<int64_t> mult_, sizes_, offsets_;
  int64_t eos_;
  std::vector<std::unique_ptr<TableFile>> files_;
  std::vector<Extent> extents_;
  std::unique_ptr<BatchReader> reader_;
  uint8_t *bounce_ = nullptr;

  std::vector<Pending> pending_;
  std::unordered_map<int64_t, size_t> pending_index_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<PleStore>(m, "PleStore")
      .def(py::init<std::vector<std::string>, std::vector<int64_t>, std::vector<int64_t>,
                    int64_t, int64_t, int64_t, std::vector<int64_t>,
                    std::vector<int64_t>, std::vector<int64_t>, int64_t, bool>(),
           py::arg("paths"), py::arg("extent_file"), py::arg("extent_base"),
           py::arg("rows_per_extent"), py::arg("row_bytes"), py::arg("row_stride"),
           py::arg("multipliers"),
           py::arg("head_vocab_sizes"), py::arg("head_offsets"), py::arg("eos_token_id"),
           py::arg("use_io_uring") = true)
      .def("stage", &PleStore::stage, py::arg("tokens_addr"), py::arg("n"),
           py::arg("staging_addr"), py::call_guard<py::gil_scoped_release>())
      .def("flush", &PleStore::flush, py::arg("signal_addr") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("io_backend", &PleStore::io_backend);
  m.def("memop_write", &memop_write, py::arg("stream"), py::arg("addr"), py::arg("value"));
  m.def("memop_wait_geq", &memop_wait_geq, py::arg("stream"), py::arg("addr"), py::arg("value"));
  m.def("memop_wait_reset", &memop_wait_reset, py::arg("stream"), py::arg("flag_addr"));
  m.def("signal_flag", &signal_flag, py::arg("flag_addr"));
}
