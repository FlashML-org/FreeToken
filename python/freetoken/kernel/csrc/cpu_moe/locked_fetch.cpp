// Locked-layer fetch for hybrid decode (python/freetoken/moe/locked_fetch.py).
//
// A locked layer's host banks have no device address, so the GPU cannot gather its rows. Per decode step a publish kernel
// writes the fetch list (count + layer-local expert rows) into mapped meta and rings a doorbell (GpuSync's words); the
// coordinator and worker threads memcpy each listed row of every bank into pinned staging, then complete the doorbell,
// and fast_index_copy_multi copies staging row i into the i-th evicted slot after a stream memop wait, or with its own
// spin wait where memops do not work. Everything is kernels and memops, so it captures into the decode graph. Row i of a
// bank sits at i * the layer's own row bytes, so layers of different widths share staging.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include <torch/extension.h>

#include <freetoken/hip_compat.h>

#include "gpu_sync.h"
#include "locked_fetch_io.h"
#include "threads.h"

namespace freetoken::cpu_moe {
namespace {

using fetch_clock = std::chrono::steady_clock;
// same idle policy as the MoE coordinator: spin while decode traffic flows, sleep once it stops
constexpr auto kHotWindow = std::chrono::milliseconds(50);
constexpr int64_t kChunkBytes = 256 << 10;

struct CopyJob {
  char* dst;
  const char* src;
  int64_t bytes;
};

class LockedFetch {
 public:
  // src[l][b] / row_bytes[l][b]: host base and row bytes of layer l's bank b (0 for banks or layers it never fetches).
  // memops: wait on the doorbell with a stream memop, else the gather spins on done_word().
  LockedFetch(std::vector<std::vector<uintptr_t>> src, std::vector<std::vector<int64_t>> row_bytes, std::vector<uintptr_t> staging,
              int max_rows, std::vector<int> cpus, bool memops)
      : src_(std::move(src)), row_bytes_(std::move(row_bytes)), staging_(std::move(staging)), max_rows_(max_rows), memops_(memops) {
    sync_ = std::make_unique<GpuSync>(memops);
    void* meta = nullptr;
    void* meta_dev = nullptr;
    const size_t meta_bytes = sizeof(int64_t) * (1 + (max_rows + 1) / 2);
    if (cudaHostAlloc(&meta, meta_bytes, cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer(&meta_dev, meta, 0) != cudaSuccess) {
      (void)cudaGetLastError();
      if (meta) (void)cudaFreeHost(meta);
      throw std::runtime_error("locked fetch: mapped meta allocation failed");
    }
    meta_ = meta;
    count_ = static_cast<volatile int64_t*>(meta);
    rows_ = reinterpret_cast<const int32_t*>(static_cast<int64_t*>(meta) + 1);
    count_dev_ = static_cast<int64_t*>(meta_dev);
    rows_dev_ = reinterpret_cast<int32_t*>(static_cast<int64_t*>(meta_dev) + 1);
    const int coord_cpu = cpus.empty() ? -1 : cpus[0];
    for (size_t i = 1; i < cpus.size(); ++i) workers_.emplace_back([this, cpu = cpus[i]] { worker_loop(cpu); });
    coord_ = std::thread([this, coord_cpu] { coordinator_loop(coord_cpu); });
  }

  ~LockedFetch() {
    stop_.store(true);
    coord_.join();
    for (auto& t : workers_) t.join();
    (void)cudaFreeHost(meta_);
  }

  LockedFetch(const LockedFetch&) = delete;
  LockedFetch& operator=(const LockedFetch&) = delete;

  // Hands one layer step's fetch list (the hybrid ensure's num_indices and src_indices) to the copy threads, then waits for
  // them on `stream` when memops work; otherwise the gather that follows waits on done_word() itself.
  void publish(uintptr_t stream, int layer, uintptr_t num_indices, uintptr_t src_indices) {
    locked_publish(stream, reinterpret_cast<const int64_t*>(num_indices), reinterpret_cast<const int32_t*>(src_indices), max_rows_,
                   count_dev_, rows_dev_, reinterpret_cast<volatile int64_t*>(sync_->ready_word()),
                   reinterpret_cast<volatile int64_t*>(sync_->done_word()), static_cast<int64_t>(layer) + 1);
    if (memops_) sync_->gpu_wait(stream);
  }
  uintptr_t done_word() const { return sync_->done_word(); }

 private:
  void run_jobs() {
    for (;;) {
      const int64_t i = next_.fetch_add(1, std::memory_order_relaxed);
      if (i >= static_cast<int64_t>(jobs_.size())) return;
      std::memcpy(jobs_[i].dst, jobs_[i].src, static_cast<size_t>(jobs_[i].bytes));
    }
  }

  void worker_loop(int cpu) {
    if (cpu >= 0) pin_this_thread(cpu);
    uint64_t seen = 0;
    auto last_active = fetch_clock::now();
    int64_t sleep_us = 20;
    while (!stop_.load(std::memory_order_relaxed)) {
      const uint64_t gen = gen_.load(std::memory_order_acquire);
      if (gen == seen) {
        if (fetch_clock::now() - last_active < kHotWindow) {
          cpu_relax();
        } else {
          std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
          sleep_us = std::min<int64_t>(sleep_us * 2, 2000);
        }
        continue;
      }
      seen = gen;
      run_jobs();
      finished_.fetch_add(1, std::memory_order_acq_rel);
      last_active = fetch_clock::now();
      sleep_us = 20;
    }
  }

  // Splits this step's rows into chunk jobs and runs them with the workers.
  void copy_layer(int layer) {
    int64_t n = *count_;
    n = std::max<int64_t>(0, std::min<int64_t>(n, max_rows_));
    jobs_.clear();
    const auto& src = src_[layer];
    const auto& row_bytes = row_bytes_[layer];
    for (int64_t i = 0; i < n; ++i) {
      const int64_t row = rows_[i];
      for (size_t b = 0; b < staging_.size(); ++b) {
        const int64_t rb = row_bytes[b];
        if (rb == 0) continue;
        char* dst = reinterpret_cast<char*>(staging_[b]) + i * rb;
        const char* s = reinterpret_cast<const char*>(src[b]) + row * rb;
        for (int64_t off = 0; off < rb; off += kChunkBytes) jobs_.push_back({dst + off, s + off, std::min(kChunkBytes, rb - off)});
      }
    }
    if (jobs_.empty()) return;
    next_.store(0, std::memory_order_relaxed);
    finished_.store(0, std::memory_order_relaxed);
    gen_.fetch_add(1, std::memory_order_acq_rel);
    run_jobs();
    // on stop a worker may leave before it joins this generation; the coordinator's run_jobs already took every job
    while (finished_.load(std::memory_order_acquire) < static_cast<int64_t>(workers_.size()) && !stop_.load(std::memory_order_relaxed))
      cpu_relax();
  }

  void coordinator_loop(int cpu) {
    if (cpu >= 0) pin_this_thread(cpu);
    auto last_active = fetch_clock::now();
    int64_t sleep_us = 20;
    while (!stop_.load(std::memory_order_relaxed)) {
      const uint64_t task = sync_->cpu_poll();
      if (task == 0) {
        if (fetch_clock::now() - last_active < kHotWindow) {
          cpu_relax();
        } else {
          std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
          sleep_us = std::min<int64_t>(sleep_us * 2, 2000);
        }
        continue;
      }
      copy_layer(static_cast<int>(task - 1));
      // release: the staging stores are visible before the GPU sees done
      sync_->cpu_done();
      last_active = fetch_clock::now();
      sleep_us = 20;
    }
    sync_->cpu_done();  // a replay caught mid-teardown does not stall
  }

  std::vector<std::vector<uintptr_t>> src_;
  std::vector<std::vector<int64_t>> row_bytes_;
  std::vector<uintptr_t> staging_;
  int64_t max_rows_;
  bool memops_;
  std::unique_ptr<GpuSync> sync_;
  void* meta_ = nullptr;
  const volatile int64_t* count_ = nullptr;
  const int32_t* rows_ = nullptr;
  int64_t* count_dev_ = nullptr;
  int32_t* rows_dev_ = nullptr;
  std::vector<CopyJob> jobs_;
  std::atomic<int64_t> next_{0};
  std::atomic<int64_t> finished_{0};
  std::atomic<uint64_t> gen_{0};
  std::atomic<bool> stop_{false};
  std::vector<std::thread> workers_;
  std::thread coord_;
};

}  // namespace

void bind_locked_fetch(pybind11::module_& m) {
  namespace py = pybind11;
  py::class_<LockedFetch>(m, "LockedFetch")
      .def(py::init<std::vector<std::vector<uintptr_t>>, std::vector<std::vector<int64_t>>, std::vector<uintptr_t>, int, std::vector<int>,
                    bool>(),
           py::arg("src"), py::arg("row_bytes"), py::arg("staging"), py::arg("max_rows"), py::arg("cpus"), py::arg("memops"))
      .def("publish", &LockedFetch::publish, py::arg("stream"), py::arg("layer"), py::arg("num_indices"), py::arg("src_indices"))
      .def("done_word", &LockedFetch::done_word);
}

}  // namespace freetoken::cpu_moe
