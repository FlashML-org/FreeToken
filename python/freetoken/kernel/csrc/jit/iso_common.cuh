// IsoQuant (ISO3/ISO4) shared device code: constants, block layouts,
// block quantize/dequantize helpers. Included by iso_store.cu and
// iso_attention.cu (each load_jit TU gets its own copy of the tables).
//
// Math matches the llama-cpp-turbo-planar-iso CPU reference
// (ggml-iso-quant.c / ggml-iso4-quant.c) with the unit-norm quaternion table.
//
// Block layouts (per 128 values of one head vector):
//   iso3: norm fp16 (2B) | qs 2-bit idx (32B) | signs hi-bit (16B)  = 50 B
//   iso4: norm fp16 (2B) | rnorm=0 (2B) | qs 4-bit nibbles (64B)    = 68 B

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace iso {

constexpr int kIsoBlock = 128;
constexpr int kIsoGroups = 32;  // 4D groups per block

__device__ __constant__ float kIsoQuat[kIsoGroups][4] = {
    {0.5765609741f, 0.4450169504f, 0.2695076466f, -0.6300023794f},
    {0.3176580369f, -0.5780548453f, -0.0201656222f, -0.7513582706f},
    {-0.3234235942f, 0.7089627385f, -0.1687686443f, -0.6035611629f},
    {-0.5127438903f, -0.3940812945f, -0.5415957570f, 0.5370919704f},
    {0.9233905673f, -0.0897334740f, -0.2796611190f, 0.2471584976f},
    {-0.3323571086f, 0.4727236331f, 0.3510629535f, 0.7367672324f},
    {0.5468608141f, 0.5542563796f, 0.2609911859f, 0.5706370473f},
    {-0.2500519454f, 0.0450818054f, -0.2715902030f, 0.9282674193f},
    {-0.5812215805f, -0.3657043576f, -0.0937586129f, 0.7208684087f},
    {0.3228830695f, -0.4298477769f, 0.3095585108f, -0.7843156457f},
    {-0.7299832702f, 0.4666220546f, -0.4123268127f, -0.2817355990f},
    {-0.4535493255f, 0.7556306720f, -0.4394895136f, -0.1736787707f},
    {-0.7338157296f, -0.5284956098f, 0.0626545250f, 0.4222335219f},
    {-0.2884652913f, 0.7042509317f, -0.4811822474f, -0.4350655377f},
    {-0.9000198841f, 0.0230921544f, -0.0407132693f, 0.4333281815f},
    {-0.0377033800f, 0.7110687494f, -0.4566248953f, 0.5333415866f},
    {0.5104404092f, 0.3024962246f, 0.7834537029f, 0.1847889870f},
    {0.2033989877f, -0.1157865301f, -0.6187923551f, 0.7498788238f},
    {-0.2462528497f, 0.7490812540f, 0.0809760988f, 0.6096553802f},
    {0.2314069420f, -0.2582575679f, -0.8879503012f, -0.3021556735f},
    {0.0072374810f, -0.2255804837f, -0.8928058147f, -0.3898189068f},
    {0.3923372924f, 0.3838746250f, 0.8350352049f, 0.0377884321f},
    {0.4958070219f, -0.3209520578f, -0.6994170547f, 0.4024685621f},
    {-0.7235037088f, -0.3477301002f, 0.5606835485f, 0.2031257302f},
    {-0.9383618832f, 0.1824720055f, 0.2933705449f, 0.0107116764f},
    {0.4430379272f, 0.4032751918f, 0.7377059460f, -0.3112498820f},
    {-0.2075705230f, 0.8433781862f, 0.4534837306f, 0.1999502629f},
    {0.1983736306f, 0.9533935785f, -0.0009816211f, -0.2273492515f},
    {-0.8834578991f, -0.0620501526f, -0.3632916510f, 0.2892593443f},
    {0.7389573455f, 0.0927560627f, -0.3959124386f, 0.5372074246f},
    {-0.0156172011f, 0.2964956462f, 0.1631654203f, 0.9408631325f},
    {0.7738668919f, 0.2402082384f, 0.5088164806f, 0.2907505929f},
};

__device__ __constant__ float kIsoCent3[8] = {
    -0.1906850000f, -0.1178320000f, -0.0657170000f, -0.0214600000f,
    0.0214600000f, 0.0657170000f, 0.1178320000f, 0.1906850000f,
};

__device__ __constant__ float kIsoCent4[16] = {
    -0.1739260000f, -0.1171950000f, -0.0895270000f, -0.0687560000f,
    -0.0512620000f, -0.0355970000f, -0.0209890000f, -0.0069380000f,
    0.0069380000f, 0.0209890000f, 0.0355970000f, 0.0512620000f,
    0.0687560000f, 0.0895270000f, 0.1171950000f, 0.1739260000f,
};

__device__ __forceinline__ int nearest_centroid3(float v) {
  int best = 0;
  float bd = fabsf(v - kIsoCent3[0]);
#pragma unroll
  for (int i = 1; i < 8; ++i) {
    float d = fabsf(v - kIsoCent3[i]);
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

__device__ __forceinline__ int nearest_centroid4(float v) {
  int best = 0;
  float bd = fabsf(v - kIsoCent4[0]);
#pragma unroll
  for (int i = 1; i < 16; ++i) {
    float d = fabsf(v - kIsoCent4[i]);
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

template <int FMT>
__device__ __forceinline__ int nearest_centroid(float v) {
  if constexpr (FMT == 3) return nearest_centroid3(v);
  else return nearest_centroid4(v);
}

template <int FMT>
__device__ __forceinline__ float centroid(int idx) {
  if constexpr (FMT == 3) return kIsoCent3[idx];
  else return kIsoCent4[idx];
}

template <int FMT>
struct IsoBlockBytes;
template <> struct IsoBlockBytes<3> { static constexpr int value = 50; };
template <> struct IsoBlockBytes<4> { static constexpr int value = 68; };

// Warp-cooperative quantize of one 128-value bf16 block; lane g owns elements
// 4g..4g+3. Writes one packed block to `dst`.
template <int FMT>
__device__ __forceinline__ void iso_quantize_block(const __nv_bfloat16 *src,
                                                   uint8_t *dst, int lane) {
  // load 4 bf16 values (8 bytes)
  const uint2 raw = reinterpret_cast<const uint2 *>(src)[lane];
  __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162 *>(&raw.x);
  __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162 *>(&raw.y);
  float v0 = __bfloat162float(lo.x), v1 = __bfloat162float(lo.y);
  float v2 = __bfloat162float(hi.x), v3 = __bfloat162float(hi.y);

  float part = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    part += __shfl_xor_sync(0xffffffffu, part, off);
  const float grp_norm = sqrtf(part);
  const float inv = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
  v0 *= inv; v1 *= inv; v2 *= inv; v3 *= inv;

  // forward rotation: r = q_lane ⊗ v (left Hamilton product)
  const float qw = kIsoQuat[lane][0], qx = kIsoQuat[lane][1];
  const float qy = kIsoQuat[lane][2], qz = kIsoQuat[lane][3];
  const float rw = qw * v0 - qx * v1 - qy * v2 - qz * v3;
  const float rx = qw * v1 + qx * v0 + qy * v3 - qz * v2;
  const float ry = qw * v2 - qx * v3 + qy * v0 + qz * v1;
  const float rz = qw * v3 + qx * v2 - qy * v1 + qz * v0;

  const int i0 = nearest_centroid<FMT>(rw), i1 = nearest_centroid<FMT>(rx);
  const int i2 = nearest_centroid<FMT>(ry), i3 = nearest_centroid<FMT>(rz);

  const float c0 = centroid<FMT>(i0), c1 = centroid<FMT>(i1);
  const float c2 = centroid<FMT>(i2), c3 = centroid<FMT>(i3);
  float recon = c0 * c0 + c1 * c1 + c2 * c2 + c3 * c3;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    recon += __shfl_xor_sync(0xffffffffu, recon, off);
  const float recon_norm = sqrtf(recon);
  const float corrected = recon_norm > 1e-10f ? grp_norm / recon_norm : grp_norm;

  if constexpr (FMT == 3) {
    if (lane == 0) {
      const __half h = __float2half_rn(corrected);
      *reinterpret_cast<__half *>(dst) = h;
    }
    dst[2 + lane] = static_cast<uint8_t>((i0 & 3) | ((i1 & 3) << 2) |
                                         ((i2 & 3) << 4) | ((i3 & 3) << 6));
    // 3rd index bits: lane g holds bits 4g..4g+3 -> nibble of signs byte g/2
    const uint32_t nib = static_cast<uint32_t>((i0 >> 2) | ((i1 >> 2) << 1) |
                                               ((i2 >> 2) << 2) | ((i3 >> 2) << 3));
    const uint32_t hi_nib = __shfl_down_sync(0xffffffffu, nib, 1);
    if ((lane & 1) == 0)
      dst[2 + 32 + (lane >> 1)] = static_cast<uint8_t>(nib | (hi_nib << 4));
  } else {
    if (lane == 0) {
      const __half h = __float2half_rn(corrected);
      const uint32_t word = static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&h));
      *reinterpret_cast<uint32_t *>(dst) = word;  // norm + rnorm(=0)
    }
    const uint16_t b0 = static_cast<uint16_t>((i0 & 0xF) | ((i1 & 0xF) << 4));
    const uint16_t b1 = static_cast<uint16_t>((i2 & 0xF) | ((i3 & 0xF) << 4));
    dst[4 + 2 * lane] = static_cast<uint8_t>(b0);
    dst[4 + 2 * lane + 1] = static_cast<uint8_t>(b1);
  }
}

// Warp-cooperative dequantize of one packed block back to 128 bf16 values.
template <int FMT>
__device__ __forceinline__ void iso_dequantize_block(const uint8_t *src,
                                                     __nv_bfloat16 *dst,
                                                     int lane) {
  float c0, c1, c2, c3;
  if constexpr (FMT == 3) {
    const uint32_t qs = src[2 + lane];
    const uint32_t sg = src[2 + 32 + (lane >> 1)];
    const int sh = (lane & 1) * 4;
    c0 = kIsoCent3[(qs & 3) | (((sg >> sh) & 1) << 2)];
    c1 = kIsoCent3[((qs >> 2) & 3) | (((sg >> (sh + 1)) & 1) << 2)];
    c2 = kIsoCent3[((qs >> 4) & 3) | (((sg >> (sh + 2)) & 1) << 2)];
    c3 = kIsoCent3[((qs >> 6) & 3) | (((sg >> (sh + 3)) & 1) << 2)];
  } else {
    const uint32_t b0 = src[4 + 2 * lane], b1 = src[4 + 2 * lane + 1];
    c0 = kIsoCent4[b0 & 0xF];
    c1 = kIsoCent4[(b0 >> 4) & 0xF];
    c2 = kIsoCent4[b1 & 0xF];
    c3 = kIsoCent4[(b1 >> 4) & 0xF];
  }
  const float norm = __half2float(*reinterpret_cast<const __half *>(src));
  // inverse rotation: conj(q_lane) ⊗ c
  const float qw = kIsoQuat[lane][0], qx = -kIsoQuat[lane][1];
  const float qy = -kIsoQuat[lane][2], qz = -kIsoQuat[lane][3];
  const float rw = qw * c0 - qx * c1 - qy * c2 - qz * c3;
  const float rx = qw * c1 + qx * c0 + qy * c3 - qz * c2;
  const float ry = qw * c2 - qx * c3 + qy * c0 + qz * c1;
  const float rz = qw * c3 + qx * c2 - qy * c1 + qz * c0;
  __nv_bfloat162 lo = __floats2bfloat162_rn(rw * norm, rx * norm);
  __nv_bfloat162 hi = __floats2bfloat162_rn(ry * norm, rz * norm);
  uint2 out;
  out.x = *reinterpret_cast<const uint32_t *>(&lo);
  out.y = *reinterpret_cast<const uint32_t *>(&hi);
  reinterpret_cast<uint2 *>(dst)[lane] = out;
}

// Dequantize one element pair (elements 2u, 2u+1 within a 128-block) to fp32.
// Used by the attention kernels; each thread redundantly computes the full
// 4D group rotation (cheap, no cross-thread shuffles needed).
template <int FMT>
__device__ __forceinline__ void iso_dequant_pair(const uint8_t *blk, int u,
                                                 float &e0, float &e1) {
  const int g = u >> 1;          // 4D group index 0..31
  const int c = (u & 1) * 2;     // pair offset within the group: 0 or 2
  float cv[4];
  if constexpr (FMT == 3) {
    const uint32_t qs = blk[2 + g];
    const uint32_t sg = blk[2 + 32 + (g >> 1)];
    const int sb = (4 * g) & 7;  // bit offset of this group in the signs byte
#pragma unroll
    for (int i = 0; i < 4; ++i)
      cv[i] = kIsoCent3[((qs >> (2 * i)) & 3) | (((sg >> (sb + i)) & 1) << 2)];
  } else {
    const uint32_t b0 = blk[4 + 2 * g], b1 = blk[4 + 2 * g + 1];
    cv[0] = kIsoCent4[b0 & 0xF];
    cv[1] = kIsoCent4[(b0 >> 4) & 0xF];
    cv[2] = kIsoCent4[b1 & 0xF];
    cv[3] = kIsoCent4[(b1 >> 4) & 0xF];
  }
  const float norm = __half2float(*reinterpret_cast<const __half *>(blk));
  const float qw = kIsoQuat[g][0], qx = -kIsoQuat[g][1];
  const float qy = -kIsoQuat[g][2], qz = -kIsoQuat[g][3];
  const float rw = qw * cv[0] - qx * cv[1] - qy * cv[2] - qz * cv[3];
  const float rx = qw * cv[1] + qx * cv[0] + qy * cv[3] - qz * cv[2];
  const float ry = qw * cv[2] - qx * cv[3] + qy * cv[0] + qz * cv[1];
  const float rz = qw * cv[3] + qx * cv[2] - qy * cv[1] + qz * cv[0];
  const float r[4] = {rw, rx, ry, rz};
  e0 = r[c] * norm;
  e1 = r[c + 1] * norm;
}

} // namespace iso
