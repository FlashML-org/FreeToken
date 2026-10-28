// CPU-compute MoE executor for the cpu and hybrid MoE strategies, and its Python binding.
//
// Decode ships activations to the CPU, computes the routed experts here (reading
// the pinned host expert banks at full RAM bandwidth), and ships the results back.
// One task (one MoE layer) is in flight at a time: submit() wakes the persistent worker
// pool, the pool runs pass 1 (gate_up) and pass 2 (down), and sync() waits for it.
// The GPU rings a doorbell to submit and waits on it to sync (GpuSync, gpu_sync.h), so the
// whole decode step stays in one CUDA graph; cudaLaunchHostFunc nodes are the fallback.
//
// One task is in flight at a time (per MoE layer): submit() wakes the pool,
// sync() blocks the host-func thread until the pool drains. The heavy GEMV runs
// on the persistent worker threads, not the host-func thread.
//
// Weight formats: bf16, NVFP4, MXFP4, ds_fp4, Q4_0, Q8_0 and block-FP8 expert
// banks (see WFmt and the per-format bank schemas). Compute is FP32-accumulate;
// the intermediate is stored bf16 to match the GPU decode path. ISA is chosen
// once at construction (AVX-512-BF16 dpbf16 -> AVX-512F widening -> AVX2+FMA -> scalar).

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

// Output-row tiling. Small enough to give every worker independent work even at
// batch size 1; large enough to amortize the atomic work-grab.
//
// Bandwidth notes (Sapphire Rapids 8480+, 13 cores): the two passes already read
// every expert weight byte exactly once per token (each output row block is owned
// by one worker), and x stays hot in L1 across a (token,expert)'s rows -- so the
// kernel is single-read bandwidth-optimal at bs=1 (~205 GB/s vs ~55 GB/s PCIe).
// One worker per *physical* core, pinned, is the sweet spot; SMT oversubscription
// thrashes the spin-barrier. Deferred (not worth it here / for this workload):
//   - AMX-bf16: a GEMM tile engine; decode is M=1 GEMV so tiles sit idle. It would
//     only pay off in a grouped/batched (dedup) path.
//   - expert dedup for bs>1: read each distinct expert once and GEMM its tokens.
//     Helps locality+bytes when bs is large; decode batches here are tiny (<=4).
//   - NUMA: a single node is assumed. Multi-socket machines would split each
//     expert's K dimension per node (banks are already per-row contiguous).
constexpr int IBLK = 32;
constexpr int HBLK = 32;

// -------------------------------- Q4_0 (W4A8) --------------------------------
// Native GGUF Q4_0 experts (gemma4 GGUF): per-32 block = fp16 scale d + 16 packed
// bytes; byte j holds element j in its low nibble and j+16 in its high nibble, so a
// block's storage order is [lo0..lo15, hi0..hi15] and w = (nibble - 8) * d. Matches
// the reference dequant (models/gguf/dequant.py) and the packed banks the GPU offload
// path streams.
//
// llama.cpp ggml_vec_dot_q4_0_q8_0: W4A8. The activation is pre-quantized to Q8_0
// (per-32-block int8 ``aq`` + fp32 scale ``asb``); each block unpacks its 16 bytes to
// 32 int8 weights in [-8,7] (bytes_from_nibbles_32: low nibbles -> elems 0..15, high
// -> 16..31) and runs an integer block dot -- VPDPBUSD (AVX-VNNI) or VPMADDUBSW+VPMADDWD
// (AVX2) with the ggml sign trick |w|*(sign(w)*a)=w*a, or a scalar int loop -- then
// scales the block sum by wd*xd in fp32. No fp weight dequant / shuffle chain. The GPU
// offload path (ggml_moe_a8_vec / MMVQ) is also W4A8, so cpu and hybrid stay close.
using q4dot_fn = float (*)(const uint8_t*, const int8_t*, const float*, int);
using q8dot_fn = float (*)(const uint8_t*, const bf16_t*, int);

// Native GGUF Q8_0 experts: per-32 block = fp16 scale d + 32 signed int8 values,
// with w = d * q. The CPU path keeps activations in bf16 and accumulates in fp32.
float q8_0_dot_bf16_scalar(const uint8_t* w, const bf16_t* x, int K) {
  float acc = 0.0f;
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* blk = w + (size_t)b * 34;
    uint16_t dh;
    std::memcpy(&dh, blk, sizeof(dh));
    const int8_t* q = reinterpret_cast<const int8_t*>(blk + 2);
    float block = 0.0f;
    for (int j = 0; j < 32; ++j)
      block += static_cast<float>(q[j]) * bf16_to_f32(x[(size_t)b * 32 + j]);
    acc += fp16_to_f32(dh) * block;
  }
  return acc;
}

float q4_0_dot_i8_scalar(const uint8_t* w, const int8_t* aq, const float* asb, int K) {
  float acc = 0.0f;
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* blk = w + (size_t)b * 18;
    uint16_t dh;
    std::memcpy(&dh, blk, sizeof(dh));
    const uint8_t* q = blk + 2;  // 16 nibble bytes
    const int8_t* a = aq + (size_t)b * 32;
    int isum = 0;
    for (int j = 0; j < 16; ++j) {
      isum += ((int)(q[j] & 0x0F) - 8) * (int)a[j];       // elem j
      isum += ((int)(q[j] >> 4) - 8) * (int)a[16 + j];    // elem 16+j
    }
    acc += fp16_to_f32(dh) * asb[b] * (float)isum;
  }
  return acc;
}

#if CPU_MOE_X86
// fp16 block scale -> fp32 via HW F16C (single value in lane 0).
__attribute__((target("f16c")))
static inline float q4_scale(uint16_t h) {
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
}

// Unpack one Q4_0 block's 16 bytes -> 32 int8 weights in [-8,7] (elems 0..15 = low
// nibbles, 16..31 = high nibbles). ``eight`` = _mm256_set1_epi8(8).
__attribute__((target("avx2")))
static inline __m256i q4_unpack32(const uint8_t* blk, __m128i mask, __m256i eight) {
  const __m128i qb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 2));
  const __m128i lo = _mm_and_si128(qb, mask);
  const __m128i hi = _mm_and_si128(_mm_srli_epi16(qb, 4), mask);
  return _mm256_sub_epi8(_mm256_set_m128i(hi, lo), eight);
}

// AVX2 W4A8 (llama.cpp non-VNNI mul_sum_i8_pairs): integer block dot via VPMADDUBSW +
// VPMADDWD (sign trick), scaled by wd*xd. |aw*sa| pair sums <= 8*127*2 < 32767 -> no
// int16 saturation. This is the fast path on AVX2 CPUs without AVX-VNNI (and the
// avx512-tier fallback, since the block dot is 256-bit either way).
__attribute__((target("avx2,fma,f16c")))
float q4_0_dot_i8_avx2(const uint8_t* w, const int8_t* aq, const float* asb, int K) {
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m256i eight = _mm256_set1_epi8(8);
  const __m256i ones16 = _mm256_set1_epi16(1);
  __m256 accF = _mm256_setzero_ps();
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* blk = w + (size_t)b * 18;
    _mm_prefetch(reinterpret_cast<const char*>(blk) + 512, _MM_HINT_T0);
    uint16_t dh;
    std::memcpy(&dh, blk, sizeof(dh));
    __m256i wq = q4_unpack32(blk, mask, eight);
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aq + (size_t)b * 32));
    __m256i aw = _mm256_sign_epi8(wq, wq);              // |wq|          (unsigned operand)
    __m256i sa = _mm256_sign_epi8(a, wq);               // sign(wq) * a  (signed operand)
    __m256i d32 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sa), ones16);  // 8 int32
    accF = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d32), _mm256_set1_ps(q4_scale(dh) * asb[b]), accF);
  }
  return hsum256(accF);
}

// AVX-VNNI W4A8: one VPDPBUSD per block (the fast path on modern CPUs).
__attribute__((target("avx2,avxvnni,fma,f16c")))
float q4_0_dot_i8_vnni(const uint8_t* w, const int8_t* aq, const float* asb, int K) {
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m256i eight = _mm256_set1_epi8(8);
  __m256 accF = _mm256_setzero_ps();
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* blk = w + (size_t)b * 18;
    _mm_prefetch(reinterpret_cast<const char*>(blk) + 512, _MM_HINT_T0);
    uint16_t dh;
    std::memcpy(&dh, blk, sizeof(dh));
    __m256i wq = q4_unpack32(blk, mask, eight);
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aq + (size_t)b * 32));
    __m256i aw = _mm256_sign_epi8(wq, wq);   // |wq|            (unsigned operand)
    __m256i sa = _mm256_sign_epi8(a, wq);    // sign(wq) * a    (signed operand)
    __m256i di = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), aw, sa);
    // All 32 elems of the block share wd*xd; distribute over di's 8 partial sums and
    // reduce at the end (equivalent to scale * block_total).
    accF = _mm256_fmadd_ps(_mm256_cvtepi32_ps(di), _mm256_set1_ps(q4_scale(dh) * asb[b]), accF);
  }
  return hsum256(accF);
}
#endif  // CPU_MOE_X86

// All tiers are W4A8 (int8 activations pre-quantized to Q8_0). AVX-VNNI is orthogonal to
// the ISA tier (gated by cpu_has_avxvnni() / FREETOKEN_CPU_MOE_NO_VNNI), so it wins when
// present; otherwise the 256-bit VPMADDUBSW kernel covers both the avx2 and avx512 tiers.
q4dot_fn select_q4dot() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
  if (cpu_has_avxvnni()) return q4_0_dot_i8_vnni;
  if (t >= ISA_AVX2) return q4_0_dot_i8_avx2;
#endif
  (void)t;
  return q4_0_dot_i8_scalar;
}

enum WFmt {
  WF_BF16 = 0,
  WF_NVFP4 = 1,
  WF_MXFP4 = 2,
  WF_DSFP4 = 3,
  WF_Q4_0 = 4,
  WF_Q8_0 = 5,
  WF_FP8_BLOCK = 6,
};

// Each ctor pointer arg is the address of a CPU int64 array of length
// num_layers (one base address per layer, built by cpu_executor.py's
// _make_table), not a single flat bank. tbl_at resolves
// tbl[layer_id] once per task/pass; a null table (bank unused by this fmt, ptr
// arg 0) resolves to nullptr without dereferencing.
inline const void* tbl_at(const uint64_t* tbl, int layer_id) {
  return tbl ? reinterpret_cast<const void*>(tbl[layer_id]) : nullptr;
}

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
  int num_layers, num_experts, top_k;
  int H, I;
  int act, apply_on_input;
  int fmt;                // WFmt
  bool needs_di = false;  // pre-deinterleave activations to fp32 (nvfp4/ds_fp4/fp8_block)
  // Per-layer pointer tables (one base address per layer, see tbl_at). gate_up_tbl
  // doubles as the bf16 gate_up table and the nvfp4/mxfp4/q4_0/ds_fp4 packed-gate_up
  // table (down_tbl likewise for down); which reinterpretation applies is picked by
  // fmt at each resolve site (see gemm1_dot/gemm2_dot/do_pass1_mxfp4/do_pass1_dsfp4).
  const uint64_t* gate_up_tbl;   // bf16: [E,2I,H] rows; else: packed e2m1/mxfp4-blocks
  const uint64_t* down_tbl;      // bf16: [E,H,I] rows; else: packed e2m1/mxfp4-blocks
  const uint64_t* gu_scale_tbl;  // nvfp4/mxfp4/ds_fp4: [E,2I,*] block scales
  const uint64_t* gu_global_tbl; // nvfp4: [E,2I] fp16 row globals
  const uint64_t* dn_scale_tbl;  // nvfp4/mxfp4/ds_fp4: [E,H,*] block scales
  const uint64_t* dn_global_tbl; // nvfp4: [E,H] fp16 row globals
  const uint64_t* gu_bias_tbl;   // mxfp4: [E,2I] bf16 biases
  const uint64_t* dn_bias_tbl;   // mxfp4: [E,H] bf16 biases
  float swiglu_alpha;
  float swiglu_limit;          // +inf == no clamp
  dot_fn dot;
  nvdot_fn nvdot;
  nvi8dot_fn nvi8dot = nullptr;  // AVX-VNNI W4A8 nvfp4 dot (nullptr -> use fp32 nvdot)
  bool use_vnni = false;         // nvfp4 + AVX-VNNI: decode via int8 VPDPBUSD (W4A8)
  bool use_q4a8 = false;       // q4_0: always W4A8 (llama.cpp Q4_0 x Q8_0); int8 pre-quant
  dsdot_fn dsdot;
  mxgemv_fn mxgemv;
  q4dot_fn q4dot;
  q8dot_fn q8dot;
  bool use_q8 = false;          // q8_0: W8A16 with bf16 activations
  fp8_block_dot_fn fp8_block_dot;
  // fp8_block: bf16 scale per 128x128 weight block. These are the scale tensor's
  // trailing strides, including any padding used by the host bank layout.
  int fp8_block_gu_scale_row = 0, fp8_block_dn_scale_row = 0;
  // ds_fp4: the caller already FP8-round-tripped the input activations on the GPU
  // (same reference grid), so submit() must not repeat it on the host-callback
  // thread. That scalar per-element pass is single-threaded ON THE DECODE CRITICAL
  // PATH (~0.3ms/layer at H=4096, every worker and the GPU waiting on it); moving
  // it to a captured GPU elementwise kernel removes it while keeping the official
  bool input_prequant = false;
  // ds_fp4 activation round-trip block (the checkpoint's fp8 block); set via set_act_block.
  int act_block = 128;
  // Q4_0 packed-row byte strides (H/32*18 for gate_up over K=H, I/32*18 for down over K=I).
  int q4_gu_row_bytes = 0, q4_dn_row_bytes = 0;
  int q8_gu_row_bytes = 0, q8_dn_row_bytes = 0;
  float e2m1_lut[16];
  float e4m3_lut[256];
  float e8m0_lut[256];         // mxfp4 block scale: 2^(s-127), s clamped to [0,254]
  const char* isa;

  std::vector<bf16_t> g_scratch;   // [max_tokens * top_k * I] intermediate
  std::vector<bf16_t> xq_scratch;  // [max_tokens * H] ds_fp4 fp8-roundtripped input
  // nvfp4/ds_fp4/fp8_block activations pre-deinterleaved to fp32 (even/odd K) for the row-major dot.
  std::vector<float> xe_scratch, xo_scratch;  // [max_tokens * H/2]   (input)
  std::vector<float> ge_scratch, go_scratch;  // [max_tokens*top_k*I/2] (intermediate)
  // AVX-VNNI W4A8: per-16-block int8 activations [even(8),odd(8)] + per-block scale.
  std::vector<int8_t> xi8_scratch, gi8_scratch;  // [max_tokens*H], [max_tokens*top_k*I]
  std::vector<float> xas_scratch, gas_scratch;   // [max_tokens*H/16], [..*top_k*I/16]
  std::string isa_str;

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
    DotChoice c = select_dot();
    dot = c.fn;
    nvdot = select_nvdot();
    dsdot = select_dsdot();
    mxgemv = select_mxgemv();
    q4dot = select_q4dot();
    q8dot = q8_0_dot_bf16_scalar;
    fp8_block_dot = select_fp8_block_dot();
    if (weight_format == WF_Q4_0) {
      if (H % 32 != 0 || I % 32 != 0)
        throw std::runtime_error("Q4_0 CPU MoE requires H and I to be multiples of 32");
      q4_gu_row_bytes = (H / 32) * 18;  // K = H (gate_up rows)
      q4_dn_row_bytes = (I / 32) * 18;  // K = I (down rows)
    }
    if (weight_format == WF_Q8_0) {
      if (H % 32 != 0 || I % 32 != 0)
        throw std::runtime_error("Q8_0 CPU MoE requires H and I to be multiples of 32");
      q8_gu_row_bytes = (H / 32) * 34;  // K = H (gate_up rows)
      q8_dn_row_bytes = (I / 32) * 34;  // K = I (down rows)
    }
    isa = c.name;
    // nvfp4 (AVX-VNNI only): W4A8 int8 decode when the CPU supports it. q4_0 is always
    // W4A8 (activations pre-quantized to Q8_0); select_q4dot picks VPDPBUSD / VPMADDUBSW
    // / scalar for the tier, so the tag reflects which of those q4dot resolved to.
    nvi8dot = select_nvi8dot();
    use_vnni = (weight_format == WF_NVFP4) && (nvi8dot != nullptr);
    use_q4a8 = (weight_format == WF_Q4_0);
    use_q8 = (weight_format == WF_Q8_0);
    const char* q4tag = use_q4a8 ? (cpu_has_avxvnni() ? "+vnni(q4_0-w4a8)" : "+q4_0-w4a8") : "";
    const char* q8tag = use_q8 ? "+q8_0-w8a16" : "";
    const char* vnni_tag =
        cpu_has_avx512vnni() ? "+avx512vnni(nvfp4-w4a8)" : "+vnni(nvfp4-w4a8)";
    isa_str = std::string(c.name) + (use_vnni ? vnni_tag : "") + q4tag + q8tag;
    isa = isa_str.c_str();
    for (int i = 0; i < 16; ++i) e2m1_lut[i] = kE2M1[i];
    for (int i = 0; i < 256; ++i) e4m3_lut[i] = e4m3_decode((uint8_t)i);
    // e8m0 (mxfp4 block scale) = 2^(s-127); the GPU GEMV clamps s to [0,254].
    for (int i = 0; i < 256; ++i) e8m0_lut[i] = std::ldexp(1.0f, std::min(i, 254) - 127);
    g_scratch.assign(static_cast<size_t>(max_tokens) * top_k * I, 0);
    // Row-major fp4 (nvfp4/ds_fp4) and fp8_block pre-deinterleave activations to fp32 even/odd.
    needs_di = (fmt == WF_NVFP4 || fmt == WF_DSFP4 || fmt == WF_FP8_BLOCK);
    if (needs_di) {
      if (fmt == WF_DSFP4) xq_scratch.assign(static_cast<size_t>(max_tokens) * H, 0);
      xe_scratch.assign(static_cast<size_t>(max_tokens) * (H / 2), 0);
      xo_scratch.assign(static_cast<size_t>(max_tokens) * (H / 2), 0);
      ge_scratch.assign(static_cast<size_t>(max_tokens) * top_k * (I / 2), 0);
      go_scratch.assign(static_cast<size_t>(max_tokens) * top_k * (I / 2), 0);
      if (use_vnni) {
        xi8_scratch.assign(static_cast<size_t>(max_tokens) * H, 0);
        xas_scratch.assign(static_cast<size_t>(max_tokens) * (H / 16), 0);
        gi8_scratch.assign(static_cast<size_t>(max_tokens) * top_k * I, 0);
        gas_scratch.assign(static_cast<size_t>(max_tokens) * top_k * (I / 16), 0);
      }
    }
    // q4_0 W4A8: per-32-block Q8_0 activations (int8 + fp32 scale) for input + intermediate.
    if (use_q4a8) {
      xi8_scratch.assign(static_cast<size_t>(max_tokens) * H, 0);
      xas_scratch.assign(static_cast<size_t>(max_tokens) * (H / 32), 0);
      gi8_scratch.assign(static_cast<size_t>(max_tokens) * top_k * I, 0);
      gas_scratch.assign(static_cast<size_t>(max_tokens) * top_k * (I / 32), 0);
    }
    for (int t = 0; t < num_threads; ++t)
      workers.emplace_back([this, t] { worker_loop(t); });
  }

  // Quantize a bf16 activation row to Q8_0 (llama.cpp): per-32-block symmetric int8 in
  // natural order + one fp32 scale (amax/127) per block. Done once per token/route,
  // amortized over every output row that the q4_0 W4A8 GEMV reads. K % 32 == 0.
  void quant_q8_0(const bf16_t* x, int K, int8_t* aq, float* asb) {
    const int nb = K / 32;
    for (int b = 0; b < nb; ++b) {
      const bf16_t* xb = x + (size_t)b * 32;
      float xf[32], amax = 0.0f;
      for (int j = 0; j < 32; ++j) {
        xf[j] = bf16_to_f32(xb[j]);
        amax = std::max(amax, std::fabs(xf[j]));
      }
      const float d = amax > 0.0f ? amax / 127.0f : 1.0f;
      asb[b] = d;
      const float inv = amax > 0.0f ? 1.0f / d : 0.0f;
      int8_t* o = aq + (size_t)b * 32;
      for (int j = 0; j < 32; ++j)
        o[j] = (int8_t)std::max(-127, std::min(127, (int)std::lround(xf[j] * inv)));
    }
  }

  // Quantize the pre-deinterleaved fp32 even/odd activations to per-16-block int8 in the
  // [even(8),odd(8)] layout the VNNI dot expects. Done once per token/route (amortized over
  // every output row), so a scalar pass is fine relative to the GEMV.
  void quant_i8_pg16(const float* xe, const float* xo, int K, int8_t* asi8, float* asb) {
    const int nb = K / 16;
    for (int b = 0; b < nb; ++b) {
      const float* xeb = xe + (size_t)b * 8;
      const float* xob = xo + (size_t)b * 8;
      float amax = 0.0f;
      for (int j = 0; j < 8; ++j)
        amax = std::max(amax, std::max(std::fabs(xeb[j]), std::fabs(xob[j])));
      const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
      asb[b] = s;
      const float inv = 1.0f / s;
      int8_t* ae = asi8 + (size_t)b * 16;
      for (int j = 0; j < 8; ++j) {
        int qe = (int)std::lround(xeb[j] * inv), qo = (int)std::lround(xob[j] * inv);
        ae[j] = (int8_t)std::max(-127, std::min(127, qe));
        ae[8 + j] = (int8_t)std::max(-127, std::min(127, qo));
      }
    }
  }

  // gate_up output row `row` (in [0, 2I)) dotted with activation over K = H. ``e`` is
  // the layer-local expert row (0..num_experts); the layer bases (already resolved
  // once per task/pass by the caller via tbl_at) pick the layer's own tensors.
  // bf16 uses the interleaved bf16 row; nvfp4 uses the pre-split fp32 even/odd halves
  // (or, with AVX-VNNI, the per-16-block int8 activations).
  // fp8_block uses the halves for whole 128-blocks and the bf16 row for a ragged tail.
  inline float gemm1_dot(const bf16_t* gate_up_l, const uint8_t* gu_packed_l,
                         const uint8_t* gu_scale_l, const uint16_t* gu_global_l, int e, int row,
                         const bf16_t* x, const float* xe, const float* xo, const int8_t* xi8,
                         const float* xas) {
    if (fmt == WF_BF16) {
      const bf16_t* w = gate_up_l + ((size_t)e * (2 * I) + row) * H;
      return dot(w, x, H);
    }
    if (fmt == WF_Q4_0) {
      const uint8_t* w =
          gu_packed_l + ((size_t)e * (2 * I) + row) * (size_t)q4_gu_row_bytes;
      return q4dot(w, xi8, xas, H);  // W4A8: int8 activations (Q8_0), scale in xas
    }
    if (fmt == WF_Q8_0) {
      const uint8_t* w =
          gu_packed_l + ((size_t)e * (2 * I) + row) * (size_t)q8_gu_row_bytes;
      return q8dot(w, x, H);
    }
    if (fmt == WF_FP8_BLOCK) {
      const uint8_t* w = gu_packed_l + ((size_t)e * (2 * I) + row) * (size_t)H;
      const bf16_t* sc = reinterpret_cast<const bf16_t*>(gu_scale_l) +
                         ((size_t)e * ((2 * I + FP8_BLK - 1) / FP8_BLK) + row / FP8_BLK) *
                             (size_t)fp8_block_gu_scale_row;
      return fp8_block_dot(w, xe, xo, x, sc, H, e4m3_lut);
    }
    const size_t r = (size_t)e * (2 * I) + row;
    if (use_vnni)
      return nvi8dot(gu_packed_l + r * (size_t)(H / 2), gu_scale_l + r * (size_t)(H / 16),
                     fp16_to_f32(gu_global_l[r]), xi8, H, e4m3_lut, xas);
    return nvdot(gu_packed_l + r * (size_t)(H / 2), gu_scale_l + r * (size_t)(H / 16),
                 fp16_to_f32(gu_global_l[r]), xe, xo, H, e2m1_lut, e4m3_lut);
  }

  // down output row `row` (in [0, H)) dotted with the intermediate over K = I. Same
  // layer-local-base convention as gemm1_dot.
  inline float gemm2_dot(const bf16_t* down_l, const uint8_t* dn_packed_l,
                         const uint8_t* dn_scale_l, const uint16_t* dn_global_l, int e, int row,
                         const bf16_t* g, const float* ge, const float* go, const int8_t* gi8,
                         const float* gas) {
    if (fmt == WF_BF16) {
      const bf16_t* w = down_l + ((size_t)e * H + row) * I;
      return dot(w, g, I);
    }
    if (fmt == WF_Q4_0) {
      const uint8_t* w = dn_packed_l + ((size_t)e * H + row) * (size_t)q4_dn_row_bytes;
      return q4dot(w, gi8, gas, I);  // W4A8: int8 activations (Q8_0), scale in gas
    }
    if (fmt == WF_Q8_0) {
      const uint8_t* w = dn_packed_l + ((size_t)e * H + row) * (size_t)q8_dn_row_bytes;
      return q8dot(w, g, I);
    }
    if (fmt == WF_FP8_BLOCK) {
      const uint8_t* w = dn_packed_l + ((size_t)e * H + row) * (size_t)I;
      const bf16_t* sc = reinterpret_cast<const bf16_t*>(dn_scale_l) +
                         ((size_t)e * ((H + FP8_BLK - 1) / FP8_BLK) + row / FP8_BLK) *
                             (size_t)fp8_block_dn_scale_row;
      return fp8_block_dot(w, ge, go, g, sc, I, e4m3_lut);
    }
    const size_t r = (size_t)e * H + row;
    if (use_vnni)
      return nvi8dot(dn_packed_l + r * (size_t)(I / 2), dn_scale_l + r * (size_t)(I / 16),
                     fp16_to_f32(dn_global_l[r]), gi8, I, e4m3_lut, gas);
    return nvdot(dn_packed_l + r * (size_t)(I / 2), dn_scale_l + r * (size_t)(I / 16),
                 fp16_to_f32(dn_global_l[r]), ge, go, I, e2m1_lut, e4m3_lut);
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
