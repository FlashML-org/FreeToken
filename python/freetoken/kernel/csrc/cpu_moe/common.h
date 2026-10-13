// Numeric helpers and ISA detection shared by the CPU MoE formats.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define CPU_MOE_X86 1
#else
#define CPU_MOE_X86 0
#endif

// Each SIMD kernel is compiled for its own ISA and picked at runtime (pick_isa), so the build sets no global -march.
#define CPU_MOE_TARGET(...) __attribute__((target(__VA_ARGS__)))

#if CPU_MOE_X86 && ((defined(__GNUC__) && __GNUC__ >= 10) || defined(__clang__))
#define CPU_MOE_HAS_AVX512BF16 1
#define CPU_MOE_HAS_AVX512VNNI 1
#endif

namespace freetoken::cpu_moe {

// Scratch rows are read with 64-byte SIMD loads, so their base is cache-line aligned whatever the heap state; std::vector alone guarantees 16 bytes.
template <class T>
struct CacheLineAllocator {
  using value_type = T;
  CacheLineAllocator() = default;
  template <class U>
  CacheLineAllocator(const CacheLineAllocator<U>&) {}
  T* allocate(size_t n) { return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(64))); }
  void deallocate(T* p, size_t) { ::operator delete(p, std::align_val_t(64)); }
  template <class U>
  bool operator==(const CacheLineAllocator<U>&) const { return true; }
};

template <class T>
using AlignedVector = std::vector<T, CacheLineAllocator<T>>;

using bf16_t = uint16_t;

inline float bf16_to_f32(bf16_t v) {
  uint32_t u = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

inline bf16_t f32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  // round-to-nearest-even
  const uint32_t lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return static_cast<bf16_t>(u >> 16);
}

// ACT_SWIGLUOAI is the clamped (up + 1) swiglu (gpt-oss "swigluoai" /
// MiniMax-M3): gate/up are combined jointly with the runtime alpha/limit
// scalars, so it is handled in the gated_rows epilogue (act_apply never sees it;
// the mxfp4 kernel additionally fuses its own copy of the same math).
// ACT_SWIGLU_CLAMP (GLM-5.3 "swiglu_limit") is the same clamped form WITHOUT
// the (up + 1) bias: clamp(gate, max=lim) * sigmoid(alpha*gate) * clamp(up, +-lim).
enum ActKind {
  ACT_SILU = 0,
  ACT_GELU = 1,
  ACT_GELU_TANH = 2,
  ACT_SWIGLUOAI = 3,
  ACT_SWIGLU_CLAMP = 4,
};

inline float act_apply(int act, float x) {
  if (act == ACT_SILU) return x / (1.0f + std::exp(-x));
  if (act == ACT_GELU)
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f));
  // gelu_tanh
  const float k0 = 0.79788456080286535588f;  // sqrt(2/pi)
  const float inner = k0 * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + std::tanh(inner));
}

// Software prefetch distance (bytes) ahead of the current weight row stream. The
// weight stream is the bandwidth bottleneck (read once, never reused); nudging the
// HW prefetcher with a few cache lines of lookahead raises sustained throughput.
constexpr int PF_AHEAD = 512;

const float kE2M1[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                         -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

inline float fp16_to_f32(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t man = h & 0x3FFu;
  uint32_t f;
  if (exp == 0) {
    if (man == 0) {
      f = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((man & 0x400u) == 0) {
        man <<= 1;
        --exp;
      }
      man &= 0x3FFu;
      f = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 0x1Fu) {
    f = sign | 0x7F800000u | (man << 13);
  } else {
    f = sign | ((exp + (127 - 15)) << 23) | (man << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

// e4m3 (OCP "fn": finite, max-normal 448, exp bias 7). Decoded into a 256-entry LUT.
inline float e4m3_decode(uint8_t v) {
  const float sign = (v & 0x80u) ? -1.0f : 1.0f;
  const uint32_t exp = (v >> 3) & 0xFu;
  const uint32_t man = v & 0x7u;
  if (exp == 0) return sign * (man / 8.0f) * 0.015625f;  // 2^(1-7) = 2^-6
  return sign * (1.0f + man / 8.0f) * std::ldexp(1.0f, (int)exp - 7);
}

#if CPU_MOE_X86
CPU_MOE_TARGET("avx2,fma")
inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  lo = _mm_add_ps(lo, _mm256_extractf128_ps(v, 1));
  lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
  lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
  return _mm_cvtss_f32(lo);
}

// AVX2 e2m1 nibble decode: codes (0..15) in 8 int32 lanes -> fp32. AVX2 vpermps is
// only 8-wide, so instead of a 16-entry LUT we use the e2m1 sign/magnitude symmetry:
// value = (code&8 ? - : +) * mag8[code&7], mag8 = e2m1[0..7]. The sign is bit 3 of
// the code shifted into the fp32 sign bit (bit 31). Bit-identical to the e2m1 LUT.
CPU_MOE_TARGET("avx2,fma")
inline __m256 e2m1_decode8(__m256i codes, __m256 mag8) {
  __m256 mag = _mm256_permutevar8x32_ps(mag8, _mm256_and_si256(codes, _mm256_set1_epi32(7)));
  __m256i sgn = _mm256_slli_epi32(_mm256_and_si256(codes, _mm256_set1_epi32(8)), 28);
  return _mm256_xor_ps(mag, _mm256_castsi256_ps(sgn));
}
#endif  // CPU_MOE_X86

// SIMD tiers, ascending. Each format picks the highest tier <= the one chosen by
// pick_isa() that it implements (fp4 and fp8_block formats have no bf16-specific tier, so the
// avx512bf16 tier maps to their avx512 kernel).
enum IsaTier { ISA_SCALAR = 0, ISA_AVX2 = 1, ISA_AVX512 = 2, ISA_AVX512BF16 = 3 };

// Best tier the CPU+build supports, optionally capped DOWN by
// FREETOKEN_CPU_MOE_ISA={scalar,avx2,avx512,avx512bf16} (A/B testing on a machine
// that supports more). FREETOKEN_CPU_MOE_SCALAR=1 forces scalar (legacy alias).
inline IsaTier pick_isa() {
#if CPU_MOE_X86
  if (getenv("FREETOKEN_CPU_MOE_SCALAR")) return ISA_SCALAR;
  IsaTier best = ISA_SCALAR;
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) best = ISA_AVX2;
  if (best >= ISA_AVX2 && __builtin_cpu_supports("avx512f")) best = ISA_AVX512;
#ifdef CPU_MOE_HAS_AVX512BF16
  if (best >= ISA_AVX512 && __builtin_cpu_supports("avx512bf16")) best = ISA_AVX512BF16;
#endif
  if (const char* f = getenv("FREETOKEN_CPU_MOE_ISA")) {
    IsaTier want = best;
    if (!std::strcmp(f, "scalar")) want = ISA_SCALAR;
    else if (!std::strcmp(f, "avx2")) want = ISA_AVX2;
    else if (!std::strcmp(f, "avx512")) want = ISA_AVX512;
    else if (!std::strcmp(f, "avx512bf16")) want = ISA_AVX512BF16;
    if (want < best) best = want;  // cap downward; never force above hw/build support
  }
  return best;
#else
  return ISA_SCALAR;
#endif
}

inline const char* tier_name(IsaTier t) {
#if CPU_MOE_X86
#ifdef CPU_MOE_HAS_AVX512BF16
  if (t >= ISA_AVX512BF16) return "avx512bf16";
#endif
  if (t >= ISA_AVX512) return "avx512f";
  if (t >= ISA_AVX2) return "avx2";
#endif
  (void)t;
  return "scalar";
}

// AVX-VNNI (VEX-256 VPDPBUSD) availability: Alder/Raptor Lake, Sapphire Rapids+, Zen5.
// Distinct from AVX-512 VNNI. Opt out with FREETOKEN_CPU_MOE_NO_VNNI=1 (A/B the W4A8 path).
inline bool cpu_has_avxvnni() {
#if CPU_MOE_X86
  const char* no = getenv("FREETOKEN_CPU_MOE_NO_VNNI");
  if (no && no[0] && no[0] != '0') return false;  // ignore unset/empty/"0"
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("avxvnni");
#else
  return false;
#endif
}

// AVX-512 VNNI (512-bit VPDPBUSD): Cascade Lake+, Ice Lake, Sapphire/Emerald Rapids,
// Zen4+. 2x the 256-bit AVX-VNNI width. FREETOKEN_CPU_MOE_NO_AVX512VNNI=1 forces the
// 256-bit path (A/B the two W4A8 kernels on the same box); FREETOKEN_CPU_MOE_NO_VNNI=1
// still disables the whole W4A8 family (back to the faithful fp32 nvdot), so it is
// honored here too.
inline bool cpu_has_avx512vnni() {
#if CPU_MOE_X86 && defined(CPU_MOE_HAS_AVX512VNNI)
  const char* no = getenv("FREETOKEN_CPU_MOE_NO_AVX512VNNI");
  if (no && no[0] && no[0] != '0') return false;
  const char* no_vnni = getenv("FREETOKEN_CPU_MOE_NO_VNNI");
  if (no_vnni && no_vnni[0] && no_vnni[0] != '0') return false;
  return __builtin_cpu_supports("avx512vnni");
#else
  return false;
#endif
}

// Split an interleaved bf16 row into fp32 even/odd halves (even[m]=src[2m]).
// bf16->fp32 is exact, so this only reorders -- done once per token/route and
// reused across every output row of the GEMV.
inline void deinterleave_bf16_f32(const bf16_t* src, float* even, float* odd, int K) {
  for (int m = 0; m < K / 2; ++m) {
    even[m] = bf16_to_f32(src[2 * m]);
    odd[m] = bf16_to_f32(src[2 * m + 1]);
  }
}

}  // namespace freetoken::cpu_moe
