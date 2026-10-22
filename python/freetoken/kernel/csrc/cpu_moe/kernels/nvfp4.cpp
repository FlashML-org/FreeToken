#include <vector>

#include "../format.h"

namespace freetoken::cpu_moe {
namespace {

// --------------------------- NVFP4 (W4A16) dequant ---------------------------
// Weights: e2m1 4-bit codes (2/byte, low nibble first), per-16 block scale in
// fp8-e4m3, per-output-row global scale in fp32. Dequant matches the GPU kernels
// (freetoken/kernel/triton/nvfp4_dequant.py): w = E2M1[code] * e4m3(scale) * global.
// Activations stay bf16 (W4A16); the GEMV dequantizes weights inside the K-loop.


// e2m1 * 2 as exact int8 (all e2m1 values are multiples of 0.5). Used by the AVX-VNNI
// W4A8 path: nibble -> int8 weight via PSHUFB LUT, then VPDPBUSD against int8 activations;
// the *2 is undone by a 0.5 folded into the final scale. Mirrors ggml's kvalues_mxfp4.
alignas(16) const int8_t kE2M1x2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// Activations pre-deinterleaved to fp32 (xe[m]=x[2m], xo[m]=x[2m+1]); see the
// ds_fp4 dot in ds_fp4.cpp for why (drops the hot loop to ~1.5 shuffle ops / 16 weights).
using nvdot_fn = float (*)(const uint8_t*, const uint8_t*, float, const float*, const float*,
                           int, const float*, const float*);

float dot_nvfp4_scalar(const uint8_t* packed, const uint8_t* scale, float global,
                       const float* xe, const float* xo, int K, const float* e2m1,
                       const float* e4m3) {
  float acc = 0.0f;
  const int nb = K / 16;
  for (int b = 0; b < nb; ++b) {
    const float bs = e4m3[scale[b]];
    const uint8_t* pk = packed + (size_t)b * 8;
    const float* xeb = xe + (size_t)b * 8;  // 16 K -> 8 even + 8 odd
    const float* xob = xo + (size_t)b * 8;
    float bsum = 0.0f;
    for (int j = 0; j < 8; ++j) {
      const uint8_t byte = pk[j];
      bsum += e2m1[byte & 0xF] * xeb[j];
      bsum += e2m1[byte >> 4] * xob[j];
    }
    acc += bs * bsum;
  }
  return acc * global;
}

// ---- NVFP4 W4A8 (int8 activations) dot: nibble->int8 via LUT, per-16 act scale ----
// asi8: int8 activations laid out per-16 block as [even(8), odd(8)]; asb[b] = per-block
// activation scale (absmax/127). Result folds the e2m1*2 -> *0.5 into the scale.
using nvi8dot_fn = float (*)(const uint8_t*, const uint8_t*, float, const int8_t*, int,
                             const float*, const float*);

[[maybe_unused]] float dot_nvfp4_i8_scalar(const uint8_t* packed, const uint8_t* scale,
                          float global, const int8_t* asi8, int K, const float* e4m3,
                          const float* asb) {
  float acc = 0.0f;
  const int nb = K / 16;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* pk = packed + (size_t)b * 8;
    const int8_t* ae = asi8 + (size_t)b * 16;       // even(8)
    const int8_t* ao = ae + 8;                      // odd(8)
    int isum = 0;
    for (int j = 0; j < 8; ++j) {
      isum += (int)kE2M1x2[pk[j] & 0xF] * (int)ae[j];
      isum += (int)kE2M1x2[pk[j] >> 4] * (int)ao[j];
    }
    acc += (e4m3[scale[b]] * asb[b]) * (float)isum;
  }
  return acc * (0.5f * global);
}

#if CPU_MOE_X86
// Two 16-K blocks (16 packed bytes) per iter: lo nibbles -> even-K, hi -> odd-K,
// gathered via two vpermps. The per-16 e4m3 scale differs across the two blocks, so
// it is applied per lane (low 8 lanes = block b, high 8 = block b+1).
CPU_MOE_TARGET("avx512f")
inline __m512 nvfp4_blk2(const uint8_t* pk, const float* xeb, const float* xob, __m512 lut,
                         __m512i loma, float s0, float s1) {
  __m512i wi = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pk)));
  __m512 vlo = _mm512_permutexvar_ps(_mm512_and_si512(wi, loma), lut);
  __m512 vhi = _mm512_permutexvar_ps(_mm512_and_si512(_mm512_srli_epi32(wi, 4), loma), lut);
  __m512 prod = _mm512_fmadd_ps(vlo, _mm512_loadu_ps(xeb), _mm512_mul_ps(vhi, _mm512_loadu_ps(xob)));
  // lanes 0-7 (block b) -> s0, lanes 8-15 (block b+1) -> s1  (pure AVX512F mask move)
  __m512 scv = _mm512_mask_mov_ps(_mm512_set1_ps(s0), 0xFF00, _mm512_set1_ps(s1));
  return _mm512_mul_ps(prod, scv);
}

CPU_MOE_TARGET("avx512f")
float dot_nvfp4_avx512(const uint8_t* packed, const uint8_t* scale, float global,
                       const float* xe, const float* xo, int K, const float* e2m1,
                       const float* e4m3) {
  const __m512 lut = _mm512_loadu_ps(e2m1);  // 16 e2m1 values
  const __m512i loma = _mm512_set1_epi32(0xF);
  __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
  const int nb = K / 16;  // 8 packed bytes + one e4m3 scale per block
  int b = 0;
  for (; b + 4 <= nb; b += 4) {  // two blk2 calls -> 4 blocks, 2 accumulators
    acc0 = _mm512_add_ps(acc0, nvfp4_blk2(packed + (size_t)b * 8, xe + (size_t)b * 8,
                                          xo + (size_t)b * 8, lut, loma, e4m3[scale[b]], e4m3[scale[b + 1]]));
    acc1 = _mm512_add_ps(acc1, nvfp4_blk2(packed + (size_t)(b + 2) * 8, xe + (size_t)(b + 2) * 8,
                                          xo + (size_t)(b + 2) * 8, lut, loma, e4m3[scale[b + 2]], e4m3[scale[b + 3]]));
  }
  for (; b + 2 <= nb; b += 2)
    acc0 = _mm512_add_ps(acc0, nvfp4_blk2(packed + (size_t)b * 8, xe + (size_t)b * 8,
                                          xo + (size_t)b * 8, lut, loma, e4m3[scale[b]], e4m3[scale[b + 1]]));
  float s = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
  for (; b < nb; ++b) {  // odd final 16-K block
    const uint8_t* pk = packed + (size_t)b * 8;
    const float* xeb = xe + (size_t)b * 8;
    const float* xob = xo + (size_t)b * 8;
    float bsum = 0.0f;
    for (int j = 0; j < 8; ++j) {
      bsum += e2m1[pk[j] & 0xF] * xeb[j] + e2m1[pk[j] >> 4] * xob[j];
    }
    s += e4m3[scale[b]] * bsum;
  }
  return s * global;
}

// AVX2: one 16-K block (8 packed bytes) per call, 8 even + 8 odd lanes.
CPU_MOE_TARGET("avx2,fma")
inline __m256 nvfp4_blk_avx2(const uint8_t* pk, const float* xeb, const float* xob,
                             __m256 mag8, float sc) {
  __m256i wi = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(pk)));
  __m256 vlo = e2m1_decode8(_mm256_and_si256(wi, _mm256_set1_epi32(0xF)), mag8);
  __m256 vhi = e2m1_decode8(_mm256_srli_epi32(wi, 4), mag8);
  __m256 prod = _mm256_fmadd_ps(vlo, _mm256_loadu_ps(xeb), _mm256_mul_ps(vhi, _mm256_loadu_ps(xob)));
  return _mm256_mul_ps(prod, _mm256_set1_ps(sc));
}

CPU_MOE_TARGET("avx2,fma")
float dot_nvfp4_avx2(const uint8_t* packed, const uint8_t* scale, float global,
                     const float* xe, const float* xo, int K, const float* e2m1,
                     const float* e4m3) {
  const __m256 mag8 = _mm256_loadu_ps(e2m1);  // e2m1[0..7] magnitudes
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  const int nb = K / 16;
  int b = 0;
  for (; b + 2 <= nb; b += 2) {
    acc0 = _mm256_add_ps(acc0, nvfp4_blk_avx2(packed + (size_t)b * 8, xe + (size_t)b * 8,
                                              xo + (size_t)b * 8, mag8, e4m3[scale[b]]));
    acc1 = _mm256_add_ps(acc1, nvfp4_blk_avx2(packed + (size_t)(b + 1) * 8, xe + (size_t)(b + 1) * 8,
                                              xo + (size_t)(b + 1) * 8, mag8, e4m3[scale[b + 1]]));
  }
  for (; b < nb; ++b)
    acc0 = _mm256_add_ps(acc0, nvfp4_blk_avx2(packed + (size_t)b * 8, xe + (size_t)b * 8,
                                              xo + (size_t)b * 8, mag8, e4m3[scale[b]]));
  return hsum256(_mm256_add_ps(acc0, acc1)) * global;
}

// AVX-VNNI W4A8: decode 8 packed bytes (16 nibbles) of one 16-block to int8 [lo(8),hi(8)]
// via PSHUFB against the e2m1*2 LUT (replaces the 2 vpermps fp32 expands -- ~4x less
// port-5 traffic). lo=even-K weights, hi=odd-K, matching the [even(8),odd(8)] act layout.
CPU_MOE_TARGET("avx2,avxvnni,fma")
inline __m128i nvfp4_decode_block_i8(const uint8_t* pk, __m128i lut) {
  __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(pk));   // 8 bytes
  __m128i lo = _mm_and_si128(b, _mm_set1_epi8(0x0F));
  __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), _mm_set1_epi8(0x0F));
  return _mm_shuffle_epi8(lut, _mm_unpacklo_epi64(lo, hi));            // [lo(8),hi(8)] -> int8
}

// Two 16-blocks per VPDPBUSD (32 int8). Sign trick (ggml mul_add_epi8): |w|*(sign(w)*a)=w*a,
// so u8*s8 needs no offset/correction term. Per-block scale (e4m3 * act-scale) folded in fp32
// (lanes 0-3 -> block b, 4-7 -> block b+1). Bit-faithful weight; only the int8 activation quant
// (W4A8) differs from the bf16 reference.
CPU_MOE_TARGET("avx2,avxvnni,fma")
float dot_nvfp4_i8_vnni(const uint8_t* packed, const uint8_t* scale, float global,
                        const int8_t* asi8, int K, const float* e4m3, const float* asb) {
  const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kE2M1x2));
  __m256 accF = _mm256_setzero_ps();
  const int nb = K / 16;
  int b = 0;
  for (; b + 2 <= nb; b += 2) {
    __m128i wb = nvfp4_decode_block_i8(packed + (size_t)b * 8, lut);
    __m128i wb1 = nvfp4_decode_block_i8(packed + (size_t)(b + 1) * 8, lut);
    __m256i w = _mm256_set_m128i(wb1, wb);                              // [blk b | blk b+1]
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(asi8 + (size_t)b * 16));
    __m256i aw = _mm256_sign_epi8(w, w);                               // |w| (u8 operand)
    __m256i sa = _mm256_sign_epi8(a, w);                               // sign(w)*a (s8 operand)
    __m256i di = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), aw, sa);
    __m256 scv = _mm256_blend_ps(_mm256_set1_ps(e4m3[scale[b]] * asb[b]),
                                 _mm256_set1_ps(e4m3[scale[b + 1]] * asb[b + 1]), 0xF0);
    accF = _mm256_fmadd_ps(_mm256_cvtepi32_ps(di), scv, accF);
  }
  float s = hsum256(accF);
  for (; b < nb; ++b) {  // tail (odd block count)
    const uint8_t* pk = packed + (size_t)b * 8;
    const int8_t* ae = asi8 + (size_t)b * 16; const int8_t* ao = ae + 8;
    int isum = 0;
    for (int j = 0; j < 8; ++j)
      isum += (int)kE2M1x2[pk[j] & 0xF] * (int)ae[j] + (int)kE2M1x2[pk[j] >> 4] * (int)ao[j];
    s += (e4m3[scale[b]] * asb[b]) * (float)isum;
  }
  return s * (0.5f * global);
}

#ifdef CPU_MOE_HAS_AVX512VNNI

// Software-prefetch distance for the W4A8 weight stream, in 16-K blocks (8 packed
// bytes each). Returns -1 when FREETOKEN_CPU_MOE_PF_BLOCKS is unset: the kernel then
// uses the built-in default min(512 blocks = 4 KB, 2 rows) -- 4 KB is the empirical
// optimum on large-row machines (Emerald Rapids sweep), while the 2-row cap keeps a small-row
// model's overshoot bounded (the executor works in 32-row tiles, so a fixed byte
// distance otherwise prefetches another worker's tile: duplicated DRAM traffic that
// regresses at the bandwidth ceiling). An EXPLICIT env value is honored verbatim
// (no clamp; 0 disables): the per-machine optimum can sit past the safe default (+20%
// at 4 KB on a 24-thread Ice Lake with 256B rows), so the escape hatch must reach it.
// Prefetch never faults, so overshooting a row/bank tail is safe.
static int nvfp4_pf_blocks() {
  static const int v = [] {
    const char* s = getenv("FREETOKEN_CPU_MOE_PF_BLOCKS");
    return (s && s[0]) ? atoi(s) : -1;
  }();
  return v;
}

// AVX-512 VNNI W4A8: FOUR 16-K blocks per VPDPBUSD (64 int8) -- 2x the AVX-VNNI
// (256-bit) path. Decode 32 packed bytes -> 64 int8 with a single _mm512_shuffle_epi8
// (e2m1*2 LUT replicated to all 4 128-bit lanes). AVX-512 has no _mm512_sign_epi8, so
// the u8*s8 sign trick (|w| as u8, sign(w)*a as s8) uses abs_epi8 + a masked negate.
// Bit-faithful weight; only the int8 activation quant (W4A8) differs from bf16.
// One 4-block group (64 int8) -> scaled fp32 partial (16 lanes). Isolated as a helper so
// the caller can run several independent chains into separate accumulators (the decode ->
// dpbusd -> scale chain is long, so a single accumulator leaves the core latency-bound).
CPU_MOE_TARGET("avx512f,avx512bw,avx512vnni,avx2")
static inline __m512 nvfp4_i8_grp4(const uint8_t* packed, const uint8_t* scale,
                                   const int8_t* asi8, const float* e4m3, const float* asb,
                                   int b, __m512i lut, __m512i idx, __m512i mask0F,
                                   __m512i idxsc) {
  const __mmask64 hi_half = 0xFF00FF00FF00FF00ULL;  // bytes 8-15 of each 128b lane -> hi nibbles
  __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed + (size_t)b * 8));
  __m512i src = _mm512_permutexvar_epi64(idx, _mm512_castsi256_si512(raw));
  __m512i lo = _mm512_and_si512(src, mask0F);
  __m512i hi = _mm512_and_si512(_mm512_srli_epi16(src, 4), mask0F);
  __m512i comb = _mm512_mask_blend_epi8(hi_half, lo, hi);
  __m512i w = _mm512_shuffle_epi8(lut, comb);  // 64 int8 weights (e2m1*2)
  // u8*s8 sign trick without _mm512_sign_epi8: aw=|w|, sa = (w<0 ? -a : a) (a moot at w==0)
  __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(asi8 + (size_t)b * 16));
  __m512i aw = _mm512_abs_epi8(w);
  __mmask64 neg = _mm512_movepi8_mask(w);
  __m512i sa = _mm512_mask_sub_epi8(a, neg, _mm512_setzero_si512(), a);
  __m512i di = _mm512_dpbusd_epi32(_mm512_setzero_si512(), aw, sa);  // 16 int32 (groups of 4)
  // int32[0..3]->blk b, [4..7]->b+1, [8..11]->b+2, [12..15]->b+3. The 4 block scales
  // (e4m3 LUT x per-block act scale) are computed vectorized: a 4-byte load + epu8->epi32
  // widen + one 4-lane LUT gather + one mul replaces 8 scalar loads + 4 scalar muls +
  // a set_ps assembly, which otherwise dominates the per-group op count.
  int sc_raw;
  memcpy(&sc_raw, scale + b, 4);
  __m128i sc4 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(sc_raw));
  __m128 s4 = _mm_mul_ps(_mm_i32gather_ps(e4m3, sc4, 4), _mm_loadu_ps(asb + b));
  __m512 scv = _mm512_permutexvar_ps(idxsc, _mm512_castps128_ps512(s4));
  return _mm512_mul_ps(_mm512_cvtepi32_ps(di), scv);
}

CPU_MOE_TARGET("avx512f,avx512bw,avx512vnni,avx2")
float dot_nvfp4_i8_avx512vnni(const uint8_t* packed, const uint8_t* scale, float global,
                              const int8_t* asi8, int K, const float* e4m3, const float* asb) {
  const __m512i lut = _mm512_broadcast_i32x4(
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(kE2M1x2)));
  const __m512i idx = _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0);  // block i -> 128b lane i
  const __m512i mask0F = _mm512_set1_epi8(0x0F);
  const __m512i idxsc = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
  __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
  __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
  const int nb = K / 16;
  // Built-in default: min(4 KB, 2 rows) -- see nvfp4_pf_blocks(). An explicitly set
  // env value is used verbatim so a per-machine sweep can reach operating points past
  // the conservative default.
  const int pfb = nvfp4_pf_blocks();
  const int pf = (pfb < 0) ? std::min(512, 2 * nb) : pfb;
  int b = 0;
  // Four independent 4-block groups per iter -> four accumulator chains hide the ~10-op
  // decode->dpbusd->scale latency (1-2 chains leave the loop latency-bound: the per-core
  // rate sat at ~60% of the core's achievable DRAM stream rate).
  for (; b + 16 <= nb; b += 16) {
    // Prefetch the weight stream ahead: the interleaved decode lowers the L1-miss
    // concurrency the HW prefetcher sustains on its own (134 -> 167 GB/s on Emerald Rapids).
    if (pf > 0) {
      _mm_prefetch(reinterpret_cast<const char*>(packed + ((size_t)b + (size_t)pf) * 8),
                   _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(packed + ((size_t)b + (size_t)pf) * 8 + 64),
                   _MM_HINT_T0);
    }
    acc0 = _mm512_add_ps(acc0, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b, lut, idx,
                                             mask0F, idxsc));
    acc1 = _mm512_add_ps(acc1, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b + 4, lut, idx,
                                             mask0F, idxsc));
    acc2 = _mm512_add_ps(acc2, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b + 8, lut, idx,
                                             mask0F, idxsc));
    acc3 = _mm512_add_ps(acc3, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b + 12, lut, idx,
                                             mask0F, idxsc));
  }
  for (; b + 4 <= nb; b += 4)
    acc0 = _mm512_add_ps(acc0, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b, lut, idx,
                                             mask0F, idxsc));
  float s = _mm512_reduce_add_ps(
      _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3)));
  for (; b < nb; ++b) {  // tail (<4 remaining 16-K blocks)
    const uint8_t* pk = packed + (size_t)b * 8;
    const int8_t* ae = asi8 + (size_t)b * 16; const int8_t* ao = ae + 8;
    int isum = 0;
    for (int j = 0; j < 8; ++j)
      isum += (int)kE2M1x2[pk[j] & 0xF] * (int)ae[j] + (int)kE2M1x2[pk[j] >> 4] * (int)ao[j];
    s += (e4m3[scale[b]] * asb[b]) * (float)isum;
  }
  return s * (0.5f * global);
}
#endif  // avx512vnni available
#endif

nvdot_fn select_nvdot() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
  if (t >= ISA_AVX512) return dot_nvfp4_avx512;
  if (t >= ISA_AVX2) return dot_nvfp4_avx2;
#endif
  (void)t;
  return dot_nvfp4_scalar;
}

// Best W4A8 (int8-activation) nvfp4 dot, or nullptr if no SIMD VNNI (caller keeps the
// faithful fp32 nvdot path). The scalar i8 dot exists only as a correctness reference.
nvi8dot_fn select_nvi8dot() {
#if CPU_MOE_X86
#if defined(CPU_MOE_HAS_AVX512VNNI)
  if (cpu_has_avx512vnni()) return dot_nvfp4_i8_avx512vnni;
#endif
  if (cpu_has_avxvnni()) return dot_nvfp4_i8_vnni;
#endif
  return nullptr;
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

class Nvfp4Format final : public MoeFormat {
 public:
  Nvfp4Format(const MoeParams& p, const FormatArgs& a)
      : MoeFormat(p), nvdot_(select_nvdot()), nvi8dot_(select_nvi8dot()) {
    gate_up_ = a.bank("gate_up");
    gu_scale_ = a.bank("gate_up_scale");
    gu_global_ = a.bank("gate_up_global");
    down_ = a.bank("down");
    dn_scale_ = a.bank("down_scale");
    dn_global_ = a.bank("down_global");
    const size_t tokens = p.max_tokens, routes = (size_t)p.max_tokens * p.top_k;
    xe_.resize(tokens * (p.H / 2));
    xo_.resize(tokens * (p.H / 2));
    ge_.resize(routes * (p.I / 2));
    go_.resize(routes * (p.I / 2));
    if (nvi8dot_ != nullptr) {
      xi8_.resize(tokens * p.H);
      xas_.resize(tokens * (p.H / 16));
      gi8_.resize(routes * p.I);
      gas_.resize(routes * (p.I / 16));
    }
    isa_ = tier_name(pick_isa());
    if (nvi8dot_ != nullptr) isa_ += cpu_has_avx512vnni() ? "+avx512vnni(nvfp4-w4a8)" : "+vnni(nvfp4-w4a8)";
    for (int i = 0; i < 16; ++i) e2m1_lut_[i] = kE2M1[i];
    for (int i = 0; i < 256; ++i) e4m3_lut_[i] = e4m3_decode((uint8_t)i);
  }

  std::string isa_name() const override { return isa_; }


  void prep_input(int tok, const bf16_t* x) override {
    float* xe = xe_.data() + (size_t)tok * (p.H / 2);
    float* xo = xo_.data() + (size_t)tok * (p.H / 2);
    deinterleave_bf16_f32(x, xe, xo, p.H);
    if (nvi8dot_ != nullptr)
      quant_i8_pg16(xe, xo, p.H, xi8_.data() + (size_t)tok * p.H, xas_.data() + (size_t)tok * (p.H / 16));
  }

  bool preps_inter() const override { return true; }

  void prep_inter(int64_t r, bf16_t* g) override {
    float* ge = ge_.data() + (size_t)r * (p.I / 2);
    float* go = go_.data() + (size_t)r * (p.I / 2);
    deinterleave_bf16_f32(g, ge, go, p.I);
    if (nvi8dot_ != nullptr)
      quant_i8_pg16(ge, go, p.I, gi8_.data() + (size_t)r * p.I, gas_.data() + (size_t)r * (p.I / 16));
  }

  void gate_up(int layer, int e, int tok, int i0, int i1, const bf16_t* /*x*/, float w_in,
               bf16_t* g) const override {
    const int H = p.H;
    const uint8_t* packed = static_cast<const uint8_t*>(tbl_at(gate_up_, layer));
    const uint8_t* scale = static_cast<const uint8_t*>(tbl_at(gu_scale_, layer));
    const float* global = static_cast<const float*>(tbl_at(gu_global_, layer));
    const float* xe = xe_.data() + (size_t)tok * (H / 2);
    const float* xo = xo_.data() + (size_t)tok * (H / 2);
    const int8_t* xi8 = nvi8dot_ != nullptr ? xi8_.data() + (size_t)tok * H : nullptr;
    const float* xas = nvi8dot_ != nullptr ? xas_.data() + (size_t)tok * (H / 16) : nullptr;
    gated_rows(p, i0, i1, w_in, g, [&](int row) {
      const size_t r = (size_t)e * (2 * p.I) + row;
      if (nvi8dot_ != nullptr)
        return nvi8dot_(packed + r * (size_t)(H / 2), scale + r * (size_t)(H / 16),
                        global[r], xi8, H, e4m3_lut_, xas);
      return nvdot_(packed + r * (size_t)(H / 2), scale + r * (size_t)(H / 16),
                    global[r], xe, xo, H, e2m1_lut_, e4m3_lut_);
    });
  }

  void down(int layer, int h0, int h1, const DownRoute* routes, int n, float* out) const override {
    const int I = p.I;
    const uint8_t* packed = static_cast<const uint8_t*>(tbl_at(down_, layer));
    const uint8_t* scale = static_cast<const uint8_t*>(tbl_at(dn_scale_, layer));
    const float* global = static_cast<const float*>(tbl_at(dn_global_, layer));
    for (int h = h0; h < h1; ++h) {
      for (int j = 0; j < n; ++j) {
        const size_t r = (size_t)routes[j].r;
        const size_t row = (size_t)routes[j].e * p.H + h;
        out[j * kRowBlock + h - h0] = nvi8dot_ != nullptr
            ? nvi8dot_(packed + row * (size_t)(I / 2), scale + row * (size_t)(I / 16), global[row],
                       gi8_.data() + r * I, I, e4m3_lut_, gas_.data() + r * (I / 16))
            : nvdot_(packed + row * (size_t)(I / 2), scale + row * (size_t)(I / 16), global[row],
                     ge_.data() + r * (I / 2), go_.data() + r * (I / 2), I, e2m1_lut_, e4m3_lut_);
      }
    }
  }

 private:
  nvdot_fn nvdot_;
  nvi8dot_fn nvi8dot_;  // AVX-VNNI W4A8 nvfp4 dot (nullptr -> use fp32 nvdot)
  // ModelOpt rows: packed e2m1 [E,2I,H/2], e4m3 scales per 16 along K [E,2I,H/16], fp32 row globals [E,2I]; down likewise with K = I.
  const uint64_t* gate_up_ = nullptr;
  const uint64_t* gu_scale_ = nullptr;
  const uint64_t* gu_global_ = nullptr;
  const uint64_t* down_ = nullptr;
  const uint64_t* dn_scale_ = nullptr;
  const uint64_t* dn_global_ = nullptr;
  std::string isa_;
  float e2m1_lut_[16];
  float e4m3_lut_[256];
  // Activations pre-deinterleaved to fp32 even/odd for the row-major dot; with VNNI also per-16-block int8 + scale.
  AlignedVector<float> xe_, xo_, ge_, go_;
  AlignedVector<int8_t> xi8_, gi8_;
  AlignedVector<float> xas_, gas_;
};

}  // namespace

std::unique_ptr<MoeFormat> make_nvfp4_format(const MoeParams& p, const FormatArgs& a) {
  return std::make_unique<Nvfp4Format>(p, a);
}

}  // namespace freetoken::cpu_moe
