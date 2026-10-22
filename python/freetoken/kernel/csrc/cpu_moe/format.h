// The interface between the CPU MoE executor and one expert weight format.
#pragma once

#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include "common.h"

namespace freetoken::cpu_moe {

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
constexpr int kRowBlock = 32;

struct MoeParams {
  int top_k = 0, max_tokens = 0;
  int H = 0, I = 0;
  int act = ACT_SILU;
  float swiglu_alpha = 1.0f;
  float swiglu_limit = INFINITY;  // +inf == no clamp
};

// What cpu_executor.py passes for one format, by name: banks map canonical roles (gate_up, gate_up_scale, ...) to per-layer tables, options hold small integers.
// A format reads what it needs once, in its constructor.
struct FormatArgs {
  std::string format;
  std::map<std::string, uint64_t> banks;
  std::map<std::string, int64_t> options;

  const uint64_t* bank(const std::string& role) const {
    const auto it = banks.find(role);
    if (it == banks.end() || it->second == 0)
      throw std::invalid_argument("CPU MoE " + format + " experts need the " + role + " bank");
    return reinterpret_cast<const uint64_t*>(it->second);
  }
  int64_t option(const std::string& name) const {
    const auto it = options.find(name);
    if (it == options.end()) throw std::invalid_argument("CPU MoE " + format + " experts need the " + name + " option");
    return it->second;
  }
  int64_t option(const std::string& name, int64_t fallback) const {
    const auto it = options.find(name);
    return it == options.end() ? fallback : it->second;
  }
};

// A bank table is the address of a CPU int64 array of length
// num_layers (one base address per layer, built by cpu_executor.py's
// _make_table), not a single flat bank. tbl_at resolves
// tbl[layer_id] once per task/pass.
inline const void* tbl_at(const uint64_t* tbl, int layer_id) {
  return reinterpret_cast<const void*>(tbl[layer_id]);
}

// One route of a pass 2 item: expert e on intermediate row r = tok * top_k + k, whose bf16 values are g.
struct DownRoute {
  int e;
  int64_t r;
  const bf16_t* g;
};

// g[i] for i in [i0, i1) from the gate row i and the up row I + i; dot(row) is the row's GEMV.
// ACT_SWIGLUOAI and ACT_SWIGLU_CLAMP clamp both inputs; the other kinds go through act_apply.
template <class RowDot>
inline void gated_rows(const MoeParams& p, int i0, int i1, float w_in, bf16_t* g, RowDot dot) {
  const bool clamped = p.act == ACT_SWIGLUOAI || p.act == ACT_SWIGLU_CLAMP;
  const float up_bias = p.act == ACT_SWIGLUOAI ? 1.0f : 0.0f;
  const float lim = p.swiglu_limit, alpha = p.swiglu_alpha;
  for (int i = i0; i < i1; ++i) {
    // gate = row i, up = row I+i
    float gate = dot(i) * w_in;
    float up = dot(p.I + i) * w_in;
    if (clamped) {
      // clamp(gate, max=lim) * sigmoid(alpha * gate) * (clamp(up, +-lim) + up_bias)
      // -- swigluoai carries the +1 up bias (gpt-oss/MiniMax); swiglu_clamp
      // (GLM-5.3) does not. lim == +inf: no clamp.
      if (gate > lim) gate = lim;
      if (up > lim) up = lim;
      else if (up < -lim) up = -lim;
      const float glu = gate / (1.0f + std::exp(-gate * alpha));
      g[i] = f32_to_bf16(glu * (up + up_bias));
    } else {
      g[i] = f32_to_bf16(act_apply(p.act, gate) * up);
    }
  }
}

// One expert weight format: its kernels, its activation layout and the scratch for it.
// The executor owns the threads and the work split.
// Scratch is sized once from max_tokens; gate_up, down and prep_inter run on the worker threads, prep_input before they wake.
class MoeFormat {
 public:
  explicit MoeFormat(const MoeParams& p) : p(p) {}
  virtual ~MoeFormat() = default;

  virtual std::string isa_name() const = 0;
  // Once per input row x[tok] before pass 1.
  virtual void prep_input(int /*tok*/, const bf16_t* /*x*/) {}
  // Whether pass 2 needs prep_inter on every intermediate row.
  virtual bool preps_inter() const { return false; }
  virtual void prep_inter(int64_t /*r*/, bf16_t* /*g*/) {}
  // Pass 1: g[i] for i in [i0, i1) (at most kRowBlock rows), expert e of `layer` on token tok.
  // w_in scales the gate and up dots: the router weight when it applies on the input, else 1.
  virtual void gate_up(int layer, int e, int tok, int i0, int i1, const bf16_t* x, float w_in,
                       bf16_t* g) const = 0;
  // Pass 2: out[j * kRowBlock + h - h0] for h in [h0, h1) (at most kRowBlock rows) and each of the n routes, before the router weight.
  // Visiting the routes' rows h together keeps n weight streams in flight, which the down GEMV needs to reach DRAM bandwidth.
  virtual void down(int layer, int h0, int h1, const DownRoute* routes, int n, float* out) const = 0;
  // Round each router-weighted down output to bf16 before the sum over routes.
  virtual bool rounds_routes() const { return false; }

 protected:
  const MoeParams& p;
};

}  // namespace freetoken::cpu_moe
