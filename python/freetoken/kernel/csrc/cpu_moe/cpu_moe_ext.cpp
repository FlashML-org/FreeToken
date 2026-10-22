// CPU-compute MoE executor for the cpu and hybrid MoE strategies, and its Python binding.
//
// Decode ships activations to the CPU, computes the routed experts here (reading
// the pinned host expert banks at full RAM bandwidth), and ships the results back.
// One task (one MoE layer) is in flight at a time: submit() wakes the persistent worker
// pool, the pool runs pass 1 (gate_up) and pass 2 (down), and sync() waits for it.
// The GPU rings a doorbell to submit and waits on it to sync (GpuSync, gpu_sync.h), so the
// whole decode step stays in one CUDA graph; cudaLaunchHostFunc nodes are the fallback.
//
// Weight formats live in kernels/, one file each behind the MoeFormat interface (format.h).
// Compute is FP32-accumulate; the intermediate is stored bf16 to match the GPU decode path.

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <freetoken/hip_compat.h>
#include <torch/extension.h>

#include "format.h"
#include "gpu_sync.h"
#include "gpu_sync_io.h"
#include "threads.h"

namespace freetoken::cpu_moe {

std::unique_ptr<MoeFormat> make_bf16_format(const MoeParams& p, const FormatArgs& a);
std::unique_ptr<MoeFormat> make_nvfp4_format(const MoeParams& p, const FormatArgs& a);
std::unique_ptr<MoeFormat> make_mxfp4_format(const MoeParams& p, const FormatArgs& a);
std::unique_ptr<MoeFormat> make_ds_fp4_format(const MoeParams& p, const FormatArgs& a);
std::unique_ptr<MoeFormat> make_fp8_block_format(const MoeParams& p, const FormatArgs& a);
std::unique_ptr<MoeFormat> make_gguf_format(const MoeParams& p, const FormatArgs& a);
std::unique_ptr<MoeFormat> make_gguf_scaled_format(const MoeParams& p, const FormatArgs& a);
void set_ggml_cpu(uintptr_t addr);
void bind_locked_fetch(pybind11::module_& m);

namespace {

constexpr int kMaxTopK = 64;

// One MoE layer on the first num_tokens rows of the pinned IO.
// It is also the doorbell word the GPU writes, a constant baked into each captured graph; num_tokens >= 1 keeps it nonzero, since 0 means no doorbell.
struct MoeTask {
  uint64_t layer : 32;
  uint64_t num_tokens : 32;
};
static_assert(sizeof(MoeTask) == sizeof(uint64_t), "MoeTask must pack into one 64-bit word");

inline uint64_t to_word(MoeTask t) {
  uint64_t w;
  std::memcpy(&w, &t, sizeof(w));
  return w;
}

inline MoeTask from_word(uint64_t w) {
  MoeTask t;
  std::memcpy(&t, &w, sizeof(t));
  return t;
}

// Every weight format this module serves, keyed by the name cpu_executor.py passes; weight_formats() exports the names.
const std::pair<const char*, std::unique_ptr<MoeFormat> (*)(const MoeParams&, const FormatArgs&)> kFormats[] = {
    {"bf16", make_bf16_format},     {"nvfp4", make_nvfp4_format},         {"mxfp4_triton", make_mxfp4_format},
    {"ds_fp4", make_ds_fp4_format}, {"fp8_block", make_fp8_block_format}, {"gguf", make_gguf_format},
    {"gguf_scaled", make_gguf_scaled_format},
};

std::unique_ptr<MoeFormat> make_format(const std::string& name, const MoeParams& p, const FormatArgs& a) {
  for (const auto& [n, make] : kFormats)
    if (name == n) return make(p, a);
  throw std::invalid_argument("unknown CPU MoE weight format " + name);
}

// One format instance as cpu_executor.py passes it: name, bank tables by role, integer options.
using FormatSpec = std::tuple<std::string, std::map<std::string, uint64_t>, std::map<std::string, int64_t>>;

struct CpuMoeExecutor {
  // A host-function callback gets one pointer, so it carries the executor with the task.
  struct HostTask {
    CpuMoeExecutor* exec;
    MoeTask task;
  };

  const int num_layers, num_experts;
  const bool apply_on_input;
  MoeParams params;  // the formats keep a reference; declared before fmts
  std::vector<std::unique_ptr<MoeFormat>> fmts;
  std::vector<int> layer_fmt;  // [num_layers], index into fmts
  int n_iblk = 0, n_hblk = 0;  // kRowBlock blocks per intermediate row and per output row

  // Pinned IO shared by every layer and batch size (one task in flight): [max_tokens, H] x and y, [max_tokens, top_k] ids and w.
  const bf16_t* io_x;
  const int32_t* io_ids;
  const float* io_w;
  bf16_t* io_y;
  AlignedVector<bf16_t> g_scratch;   // [max_tokens * top_k * I] intermediate

  // Worker pool.
  int num_threads;
  std::vector<int> core_ids;          // worker tid -> logical CPU to pin to (may be empty)
  std::vector<std::thread> workers;
  std::mutex task_mtx;
  std::condition_variable task_cv;
  std::mutex sync_mtx;
  std::condition_variable sync_cv;

  std::atomic<bool> stop{false};  // spinning workers read it without the lock
  uint64_t cur_gen = 0;
  MoeTask cur_task{};
  // Separate cache lines: idle workers poll `submitted` while busy workers hit p1_next/p2_next.
  alignas(64) std::atomic<uint64_t> submitted{0};
  alignas(64) std::atomic<uint64_t> completed{0};

  // Spin-then-park waits. Off unless the Python wrapper finds spare CPUs for the spinners.
  static constexpr auto kSpinWindow = std::chrono::milliseconds(50);
  std::atomic<bool> spin_wait{false};
  bool coord_pinned = false;

  // Work split of the current task.
  alignas(64) std::atomic<int64_t> p1_next{0};
  std::atomic<int64_t> p2_next{0};
  std::atomic<int64_t> prt_next{0};  // intermediate-row prep phase (prep_inter)
  int64_t p1_total = 0, p2_total = 0, prt_total = 0;
  std::atomic<int> done_count{0};
  std::atomic<int> bar_count{0};
  std::atomic<int> bar_sense{0};

  // ---- Flag-based GPU<->CPU handshake (replaces the per-layer cudaLaunchHostFunc pair) ----
  // The GPU rings the doorbell with a task word at submit; this coordinator thread busy-polls it, runs the task on the worker pool, and completes it, which the GPU waits on at sync.
  // This removes the ~2x30-50us host-func dispatch round trips per MoE layer per decode step that otherwise idle the GPU (~6 ms/step on a 75-layer model).
  std::unique_ptr<GpuSync> gpu_sync;
  std::unique_ptr<KernelIo> kernel_io;  // replaces the copies around the doorbell once enable_kernel_io() runs
  std::thread coord_thread;
  std::atomic<bool> coord_stop{false};
  std::atomic<int64_t> flag_served{0};  // completed doorbells (tests, watchdog)
  std::map<uint64_t, HostTask> host_tasks;  // host-function userData, keyed by doorbell word (graph-stable)

  CpuMoeExecutor(int num_threads_, int num_layers_, int num_experts_, int top_k_,
                 int hidden_size, int inter_size, int max_tokens_, int activation_id,
                 int apply_router_weight_on_input, std::vector<FormatSpec> formats,
                 std::vector<int> layer_formats, double swiglu_alpha_, double swiglu_limit_,
                 std::vector<int> core_ids_, uintptr_t x_ptr, uintptr_t ids_ptr, uintptr_t w_ptr,
                 uintptr_t y_ptr)
      : num_layers(num_layers_),
        num_experts(num_experts_),
        apply_on_input(apply_router_weight_on_input != 0),
        params{top_k_, max_tokens_, hidden_size, inter_size, activation_id, static_cast<float>(swiglu_alpha_),
               static_cast<float>(swiglu_limit_)},
        io_x(reinterpret_cast<const bf16_t*>(x_ptr)),
        io_ids(reinterpret_cast<const int32_t*>(ids_ptr)),
        io_w(reinterpret_cast<const float*>(w_ptr)),
        io_y(reinterpret_cast<bf16_t*>(y_ptr)),
        num_threads(num_threads_ > 0 ? num_threads_ : 1),
        core_ids(std::move(core_ids_)) {
    if (params.top_k > kMaxTopK) throw std::invalid_argument("CPU MoE supports top_k <= " + std::to_string(kMaxTopK));
    for (auto& [name, banks, options] : formats)
      fmts.push_back(make_format(name, params, FormatArgs{name, std::move(banks), std::move(options)}));
    if (static_cast<int>(layer_formats.size()) != num_layers)
      throw std::invalid_argument("CPU MoE needs the format of every layer");
    for (const int f : layer_formats)
      if (f < 0 || f >= static_cast<int>(fmts.size())) throw std::out_of_range("CPU MoE layer format index out of range");
    layer_fmt = std::move(layer_formats);
    n_iblk = (params.I + kRowBlock - 1) / kRowBlock;
    n_hblk = (params.H + kRowBlock - 1) / kRowBlock;
    g_scratch.assign(static_cast<size_t>(params.max_tokens) * params.top_k * params.I, 0);
    for (int t = 0; t < num_threads; ++t)
      workers.emplace_back([this, t] { worker_loop(t); });
  }

  void pin_self(int tid) {
    if (!core_ids.empty()) pin_this_thread(core_ids[tid % static_cast<int>(core_ids.size())]);
  }

  ~CpuMoeExecutor() {
    coord_stop.store(true);
    if (coord_thread.joinable()) coord_thread.join();
    {
      std::lock_guard<std::mutex> lk(task_mtx);
      stop = true;
    }
    task_cv.notify_all();
    for (auto& th : workers)
      if (th.joinable()) th.join();
  }

  MoeTask task(int layer_id, int num_tokens) const {
    if (layer_id < 0 || layer_id >= num_layers || num_tokens < 1 || num_tokens > params.max_tokens)
      throw std::out_of_range("CPU MoE task (layer " + std::to_string(layer_id) + ", " +
                              std::to_string(num_tokens) + " tokens) is outside the executor's layers or IO");
    return MoeTask{static_cast<uint64_t>(layer_id), static_cast<uint64_t>(num_tokens)};
  }

  std::string isa_name() const {
    std::vector<std::string> names;
    for (const auto& f : fmts)
      if (const std::string n = f->isa_name(); std::find(names.begin(), names.end(), n) == names.end()) names.push_back(n);
    std::string out;
    for (const auto& n : names) out += (out.empty() ? "" : ",") + n;
    return out;
  }

  const MoeFormat& format_of(const MoeTask& t) const { return *fmts[layer_fmt[t.layer]]; }

  // Whether any route of the task names an expert; do_pass1 and do_pass2 skip the others.
  bool has_routes(const MoeTask& t) const {
    const int64_t n = static_cast<int64_t>(t.num_tokens) * params.top_k;
    for (int64_t i = 0; i < n; ++i)
      if (io_ids[i] >= 0 && io_ids[i] < num_experts) return true;
    return false;
  }

  void barrier(int& local_sense) {
    local_sense ^= 1;
    if (bar_count.fetch_add(1) + 1 == num_threads) {
      bar_count.store(0);
      bar_sense.store(local_sense);
    } else {
      while (bar_sense.load() != local_sense) cpu_relax();
    }
  }

  // Pass 1 item p = one kRowBlock block of one (token, route)'s intermediate row.
  void do_pass1(const MoeTask& t, int64_t p) {
    const int top_k = params.top_k, H = params.H, I = params.I;
    const int64_t ib = p % n_iblk;
    const int64_t tk = p / n_iblk;
    const int k = static_cast<int>(tk % top_k);
    const int tok = static_cast<int>(tk / top_k);
    const int e = io_ids[static_cast<size_t>(tok) * top_k + k];
    if (e < 0 || e >= num_experts) return;
    const float w_in = apply_on_input ? io_w[static_cast<size_t>(tok) * top_k + k] : 1.0f;
    const int i0 = static_cast<int>(ib) * kRowBlock;
    const int i1 = std::min(I, i0 + kRowBlock);
    format_of(t).gate_up(static_cast<int>(t.layer), e, tok, i0, i1, io_x + (size_t)tok * H, w_in,
                         g_scratch.data() + ((size_t)tok * top_k + k) * I);
  }

  // Pass 2 item p = one kRowBlock block of one token's output row, summed over its routes.
  void do_pass2(const MoeTask& t, int64_t p) {
    const int top_k = params.top_k, H = params.H, I = params.I;
    const int64_t hb = p % n_hblk;
    const int tok = static_cast<int>(p / n_hblk);
    const int h0 = static_cast<int>(hb) * kRowBlock;
    const int h1 = std::min(H, h0 + kRowBlock);
    DownRoute routes[kMaxTopK];
    float w_out[kMaxTopK];
    int n = 0;
    for (int k = 0; k < top_k; ++k) {
      const int e = io_ids[static_cast<size_t>(tok) * top_k + k];
      if (e < 0 || e >= num_experts) continue;
      const size_t r = (size_t)tok * top_k + k;
      routes[n] = {e, static_cast<int64_t>(r), g_scratch.data() + r * I};
      w_out[n] = apply_on_input ? 1.0f : io_w[r];
      ++n;
    }
    float part[kMaxTopK * kRowBlock];
    const MoeFormat& f = format_of(t);
    if (n > 0) f.down(static_cast<int>(t.layer), h0, h1, routes, n, part);
    const bool rounds_routes = f.rounds_routes();
    bf16_t* y_row = io_y + (size_t)tok * H;
    for (int h = h0; h < h1; ++h) {
      float acc = 0.0f;
      for (int j = 0; j < n; ++j) {
        const float v = part[j * kRowBlock + h - h0] * w_out[j];
        acc += rounds_routes ? bf16_to_f32(f32_to_bf16(v)) : v;
      }
      y_row[h] = f32_to_bf16(acc);
    }
  }

  void run_task_body(const MoeTask& t) {
    int local_sense = 0;
    for (;;) {
      int64_t p = p1_next.fetch_add(1, std::memory_order_relaxed);
      if (p >= p1_total) break;
      do_pass1(t, p);
    }
    barrier(local_sense);
    // Prepare the intermediate rows (per token,route) for the down GEMV, see prep_inter.
    // Needs all of pass1 done (a full row spans every iblk).
    if (prt_total > 0) {
      MoeFormat& f = *fmts[layer_fmt[t.layer]];
      for (;;) {
        int64_t r = prt_next.fetch_add(1, std::memory_order_relaxed);
        if (r >= prt_total) break;
        f.prep_inter(r, g_scratch.data() + (size_t)r * params.I);
      }
      barrier(local_sense);
    }
    for (;;) {
      int64_t p = p2_next.fetch_add(1, std::memory_order_relaxed);
      if (p >= p2_total) break;
      do_pass2(t, p);
    }
  }

  // Tasks arrive once per MoE layer, sub-ms apart. Spinning skips a futex wakeup per worker per layer; parking after kSpinWindow keeps an idle server off the CPU.
  void worker_loop(int tid) {
    pin_self(tid);
    uint64_t my_gen = 0;
    using wclk = std::chrono::steady_clock;
    auto last_active = wclk::now();
    for (;;) {
      uint64_t gen = submitted.load(std::memory_order_acquire);
      if (gen == my_gen && spin_wait.load(std::memory_order_relaxed)) {
        unsigned polls = 0;
        while (gen == my_gen) {
          cpu_relax();
          if (stop.load(std::memory_order_relaxed)) return;
          if ((++polls & 1023u) == 0 && wclk::now() - last_active > kSpinWindow) break;
          gen = submitted.load(std::memory_order_acquire);
        }
      }
      if (gen == my_gen) {
        std::unique_lock<std::mutex> lk(task_mtx);
        task_cv.wait(lk, [&] { return stop.load() || cur_gen != my_gen; });
        if (stop.load()) return;
        gen = cur_gen;
      }
      // submit() writes cur_task before the release store to `submitted`.
      my_gen = gen;
      run_task_body(cur_task);
      last_active = wclk::now();
      if (done_count.fetch_add(1) + 1 == num_threads) {
        completed.store(my_gen, std::memory_order_release);
        {
          std::lock_guard<std::mutex> lk(sync_mtx);
        }
        sync_cv.notify_all();
      }
    }
  }

  void submit(MoeTask t) {
    const int num_tokens = static_cast<int>(t.num_tokens);
    MoeFormat& fmt = *fmts[layer_fmt[t.layer]];
    p1_total = static_cast<int64_t>(num_tokens) * params.top_k * n_iblk;
    p2_total = static_cast<int64_t>(num_tokens) * n_hblk;
    prt_total = fmt.preps_inter() ? static_cast<int64_t>(num_tokens) * params.top_k : 0;
    p1_next.store(0, std::memory_order_relaxed);
    p2_next.store(0, std::memory_order_relaxed);
    prt_next.store(0, std::memory_order_relaxed);
    done_count.store(0, std::memory_order_relaxed);
    bar_count.store(0, std::memory_order_relaxed);
    bar_sense.store(0, std::memory_order_relaxed);
    // Prepare the per-token input once, up front.
    // Single-threaded: tiny for decode, and done before the workers are woken below.
    for (int tok = 0; tok < num_tokens; ++tok) fmt.prep_input(tok, io_x + (size_t)tok * params.H);
    {
      std::lock_guard<std::mutex> lk(task_mtx);
      cur_task = t;
      ++cur_gen;
      submitted.store(cur_gen, std::memory_order_release);
    }
    task_cv.notify_all();
  }

  // Only a coordinator on its own core may spin: host-func callbacks and eager calls share cores with the workers.
  void sync(bool spin = false) {
    const uint64_t target = submitted.load(std::memory_order_acquire);
    if (spin) {
      using sclk = std::chrono::steady_clock;
      const auto t0 = sclk::now();
      unsigned polls = 0;
      while (completed.load(std::memory_order_acquire) < target) {
        cpu_relax();
        if ((++polls & 1023u) == 0 && sclk::now() - t0 > kSpinWindow) break;
      }
      if (completed.load(std::memory_order_acquire) >= target) return;
    }
    std::unique_lock<std::mutex> lk(sync_mtx);
    sync_cv.wait(lk, [&] { return completed.load(std::memory_order_acquire) >= target; });
  }

  void submit_with_cuda_stream(uintptr_t stream, int layer_id, int num_tokens) {
    const MoeTask t = task(layer_id, num_tokens);
    HostTask& h = host_tasks.try_emplace(to_word(t), HostTask{this, t}).first->second;
    cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream), &CpuMoeExecutor::submit_cb, &h);
  }

  void sync_with_cuda_stream(uintptr_t stream) {
    cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream), &CpuMoeExecutor::sync_cb, this);
  }

  // Start the busy-poll coordinator over the doorbell. ``pin_core`` >= 0
  // pins the coordinator to that logical CPU (the worker auto-sizing reserves it), so its
  // polling never migrates onto / contends with a GEMV worker's core.
  void start_flag_coordinator(int pin_core) {
    gpu_sync = std::make_unique<GpuSync>();
    coord_stop.store(false);
    coord_pinned = CPU_MOE_HAS_AFFINITY && pin_core >= 0;
    coord_thread = std::thread([this, pin_core] {
      if (pin_core >= 0) pin_this_thread(pin_core);
      coordinator_loop();
    });
  }

  void flag_signal(uintptr_t stream, int layer_id, int num_tokens) {
    gpu_sync->gpu_submit(stream, to_word(task(layer_id, num_tokens)));
  }
  void flag_wait(uintptr_t stream) { gpu_sync->gpu_wait(stream); }
  void enable_kernel_io() {
    kernel_io = std::make_unique<KernelIo>(gpu_sync->ready_word(), gpu_sync->done_word(), io_x, io_ids, io_w, io_y, params.H,
                                           params.top_k);
  }
  void flag_publish(uintptr_t stream, uintptr_t x, uintptr_t ids, uintptr_t w, int layer_id, int num_tokens) {
    kernel_io->publish(stream, reinterpret_cast<const void*>(x), reinterpret_cast<const int32_t*>(ids),
                       reinterpret_cast<const float*>(w), num_tokens, to_word(task(layer_id, num_tokens)));
  }
  void flag_combine(uintptr_t stream, uintptr_t out, int num_tokens, bool accumulate) {
    kernel_io->combine(stream, reinterpret_cast<void*>(out), num_tokens, accumulate);
  }
  bool flag_pending() const { return gpu_sync && gpu_sync->pending(); }
  // Complete a doorbell the coordinator never answered so the stream can make progress.
  void poison_flag() {
    if (gpu_sync) gpu_sync->cpu_done();
  }

  void coordinator_loop() {
    // Idle backoff (edge-device friendliness): spin hot only while decode traffic is
    // flowing, with a TIME-based hot window (an iteration count varies 2-6x with pause
    // cost, so some configs dozed every token). 50 ms since the last
    // flag comfortably covers intra- and inter-token gaps -- including prefill bursts and
    // --moe-cpu-layers gaps -- so steady decode never
    // sleeps and keeps the ~us-level wakeup. Past it, i.e. the engine is actually idle,
    // escalate to timed sleeps capped at 2 ms: a dozing coordinator costs <0.1% of a
    // core instead of 100%, and the only price is a <=2 ms discovery delay on the FIRST
    // MoE layer after an idle period (irrelevant next to prefill). The clock is sampled
    // every 1024 empty polls (~0.1-1 ms) to keep the hot loop cheap.
    using coord_clock = std::chrono::steady_clock;
    constexpr auto kHotWindow = std::chrono::milliseconds(50);
    constexpr int64_t kSleepCapUs = 2000;
    auto last_active = coord_clock::now();
    unsigned empty_polls = 0;  // unsigned: the hot-phase ++ must not overflow into UB
    int64_t sleep_us = 100;
    bool dozing = false;
    while (!coord_stop.load(std::memory_order_relaxed)) {
      const uint64_t word = gpu_sync->cpu_poll();
      if (word != 0) {
        const MoeTask t = from_word(word);
        if (has_routes(t)) {
          submit(t);
          sync(coord_pinned && spin_wait.load(std::memory_order_relaxed));
        } else {
          // What the pool would write for a task without routes, minus its fork and join.
          std::memset(io_y, 0, sizeof(bf16_t) * t.num_tokens * params.H);
        }
        // Release: the workers' y stores are visible before the GPU sees done.
        gpu_sync->cpu_done();
        flag_served.fetch_add(1, std::memory_order_relaxed);
        last_active = coord_clock::now();
        empty_polls = 0;
        sleep_us = 100;
        dozing = false;
        continue;
      }
      // Hot phase: pause-spin, consulting the clock only every 1024 empty polls (the
      // amortization must gate the CLOCK, not the sleep -- gating the sleep decision
      // let 1023/1024 idle iterations hot-poll).
      // Doze phase: sleep on EVERY iteration until traffic returns; one wake-up scan
      // (~0.5 us) per 2 ms sleep is ~0.03% duty.
      if (!dozing) {
        if ((++empty_polls & 1023u) != 0 ||
            coord_clock::now() - last_active < kHotWindow) {
          cpu_relax();
          continue;
        }
        dozing = true;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
      sleep_us = std::min<int64_t>(sleep_us * 2, kSleepCapUs);
    }
    // Teardown: complete any in-flight wait so a replay caught mid-shutdown does not stall
    // until the watchdog fires. Runs before the destructor's join() returns, while the doorbell is still alive.
    gpu_sync->cpu_done();
  }

  // Eager (non-graph) path: run one task to completion on the pool.
  void run(int layer_id, int num_tokens) {
    submit(task(layer_id, num_tokens));
    sync();
  }

  static void CUDART_CB submit_cb(void* ud) {
    HostTask* h = reinterpret_cast<HostTask*>(ud);
    h->exec->submit(h->task);
  }
  static void CUDART_CB sync_cb(void* ud) { reinterpret_cast<CpuMoeExecutor*>(ud)->sync(); }
};

}  // namespace
}  // namespace freetoken::cpu_moe

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  using freetoken::cpu_moe::CpuMoeExecutor;
  using freetoken::cpu_moe::GpuSync;
  py::class_<CpuMoeExecutor>(m, "CpuMoeExecutor")
      .def(py::init<int, int, int, int, int, int, int, int, int, std::vector<freetoken::cpu_moe::FormatSpec>,
                    std::vector<int>, double, double, std::vector<int>, uintptr_t, uintptr_t, uintptr_t,
                    uintptr_t>(),
           py::arg("num_threads"), py::arg("num_layers"), py::arg("num_experts"),
           py::arg("top_k"), py::arg("hidden_size"), py::arg("inter_size"),
           py::arg("max_tokens"), py::arg("activation_id"),
           py::arg("apply_router_weight_on_input"), py::arg("formats"), py::arg("layer_formats"),
           py::arg("swiglu_alpha"), py::arg("swiglu_limit"),
           py::arg("core_ids"), py::arg("x_ptr"), py::arg("ids_ptr"), py::arg("w_ptr"),
           py::arg("y_ptr"))
      .def("submit_with_cuda_stream", &CpuMoeExecutor::submit_with_cuda_stream,
           py::arg("stream"), py::arg("layer_id"), py::arg("num_tokens"),
           py::call_guard<py::gil_scoped_release>())
      .def("sync_with_cuda_stream", &CpuMoeExecutor::sync_with_cuda_stream,
           py::arg("stream"), py::call_guard<py::gil_scoped_release>())
      .def("run", &CpuMoeExecutor::run, py::arg("layer_id"), py::arg("num_tokens"),
           py::call_guard<py::gil_scoped_release>())
      .def("flag_served", [](const CpuMoeExecutor& e) { return e.flag_served.load(); })
      .def("start_flag_coordinator", &CpuMoeExecutor::start_flag_coordinator, py::arg("pin_core"))
      .def("flag_signal", &CpuMoeExecutor::flag_signal, py::arg("stream"), py::arg("layer_id"),
           py::arg("num_tokens"))
      .def("flag_wait", &CpuMoeExecutor::flag_wait, py::arg("stream"))
      .def("flag_pending", &CpuMoeExecutor::flag_pending)
      .def("enable_kernel_io", &CpuMoeExecutor::enable_kernel_io)
      .def("flag_publish", &CpuMoeExecutor::flag_publish, py::arg("stream"), py::arg("x"), py::arg("ids"), py::arg("w"),
           py::arg("layer_id"), py::arg("num_tokens"))
      .def("flag_combine", &CpuMoeExecutor::flag_combine, py::arg("stream"), py::arg("out"), py::arg("num_tokens"),
           py::arg("accumulate"))
      .def("poison_flag", &CpuMoeExecutor::poison_flag)
      .def("isa_name", &CpuMoeExecutor::isa_name)
      .def("set_spin_wait",
           [](CpuMoeExecutor& e, bool v) { e.spin_wait.store(v, std::memory_order_relaxed); },
           py::arg("enabled"))
      .def("get_spin_wait", [](const CpuMoeExecutor& e) { return e.spin_wait.load(); });
  freetoken::cpu_moe::bind_locked_fetch(m);
  m.def("memops_probe", &GpuSync::probe, py::arg("stream"));
  m.def("kernel_io_probe", &freetoken::cpu_moe::KernelIo::probe, py::arg("stream"));
  m.def("host_func_capture_safe", &GpuSync::host_func_capture_safe);
  // ABI capability marker: the highest ActKind this build implements in the
  // GENERIC epilogue. CpuMoeExecutor.__init__ probes it before requesting an act
  // id the epilogue must handle -- a prebuilt .so from before ACT_SWIGLUOAI
  // accepts id 3 without error and silently computes the wrong activation
  // (act_apply falls through to gelu_tanh); the probe turns a stale extension
  // into a loud rebuild instruction instead of wrong model outputs.
  m.def("max_generic_act_id", []() { return static_cast<int>(freetoken::cpu_moe::ACT_SWIGLU_CLAMP); });
  // The ggml-cpu variant the gguf format calls: the kernel table address that freetoken.kernel.ggml_cpu.select() returns.
  m.def("set_ggml_cpu", &freetoken::cpu_moe::set_ggml_cpu, py::arg("api"));
  m.def("weight_formats", []() {
    py::list names;
    for (const auto& f : freetoken::cpu_moe::kFormats) names.append(f.first);
    return names;
  });
}
