"""IsoQuant KV-cache quantization (ISO3/ISO4): torch reference + shared constants.

Ported from the llama.cpp fork ``llama-cpp-turbo-planar-iso`` (ggml-iso-quant.c /
ggml-iso4-quant.c). One block = 128 values of one attention head vector (one token):

- iso3 (GGUF type "iso3", ggml id 46): fp16 norm + 2-bit centroid indices (32 B)
  + 1-bit high index bits (16 B) = 50 B / 128 values (3.125 bpw).
- iso4 (GGUF type "iso4", ggml id 48): fp16 norm + fp16 rnorm (always 0) +
  4-bit nibble-packed centroid indices (64 B) = 68 B / 128 values (4.25 bpw).

Algorithm (per 128-value block): L2-normalize, rotate each 4D group by a fixed
unit quaternion (left Hamilton product), quantize to the nearest Lloyd-Max
centroid, store the "corrected norm" ||x|| / ||centroids|| so that dequant
preserves the vector norm exactly (rotation is norm-preserving). Dequant looks
up centroids and applies the conjugate (inverse) rotation, then scales by norm.

The quaternion table below is the *reference* table from ggml-iso-quant.c /
ggml-iso4-quant.c / ggml-metal.metal (all unit-norm). The fork's CUDA path uses
a different, non-normalized table (planar-iso-constants.cuh) which we do NOT
copy — FreeToken uses this table consistently on CPU and CUDA.
"""

from __future__ import annotations

import functools

import torch

from .utils import load_jit, make_cpp_args

QK_ISO = 128  # values per block
ISO_N_GROUPS = 32  # 4D rotation groups per block

ISO3_BLOCK_BYTES = 50  # norm(2) + qs(32) + signs(16)
ISO4_BLOCK_BYTES = 68  # norm(2) + rnorm(2) + qs(64)

FORMATS = ("iso3", "iso4")

# Unit quaternions, one per 4D group (reference "Set A"; identical in
# ggml-iso-quant.c, ggml-iso4-quant.c and the fork's Metal shaders).
ISO_QW = (
    0.5765609741, 0.3176580369, -0.3234235942, -0.5127438903,
    0.9233905673, -0.3323571086, 0.5468608141, -0.2500519454,
    -0.5812215805, 0.3228830695, -0.7299832702, -0.4535493255,
    -0.7338157296, -0.2884652913, -0.9000198841, -0.0377033800,
    0.5104404092, 0.2033989877, -0.2462528497, 0.2314069420,
    0.0072374810, 0.3923372924, 0.4958070219, -0.7235037088,
    -0.9383618832, 0.4430379272, -0.2075705230, 0.1983736306,
    -0.8834578991, 0.7389573455, -0.0156172011, 0.7738668919,
)
ISO_QX = (
    0.4450169504, -0.5780548453, 0.7089627385, -0.3940812945,
    -0.0897334740, 0.4727236331, 0.5542563796, 0.0450818054,
    -0.3657043576, -0.4298477769, 0.4666220546, 0.7556306720,
    -0.5284956098, 0.7042509317, 0.0230921544, 0.7110687494,
    0.3024962246, -0.1157865301, 0.7490812540, -0.2582575679,
    -0.2255804837, 0.3838746250, -0.3209520578, -0.3477301002,
    0.1824720055, 0.4032751918, 0.8433781862, 0.9533935785,
    -0.0620501526, 0.0927560627, 0.2964956462, 0.2402082384,
)
ISO_QY = (
    0.2695076466, -0.0201656222, -0.1687686443, -0.5415957570,
    -0.2796611190, 0.3510629535, 0.2609911859, -0.2715902030,
    -0.0937586129, 0.3095585108, -0.4123268127, -0.4394895136,
    0.0626545250, -0.4811822474, -0.0407132693, -0.4566248953,
    0.7834537029, -0.6187923551, 0.0809760988, -0.8879503012,
    -0.8928058147, 0.8350352049, -0.6994170547, 0.5606835485,
    0.2933705449, 0.7377059460, 0.4534837306, -0.0009816211,
    -0.3632916510, -0.3959124386, 0.1631654203, 0.5088164806,
)
ISO_QZ = (
    -0.6300023794, -0.7513582706, -0.6035611629, 0.5370919704,
    0.2471584976, 0.7367672324, 0.5706370473, 0.9282674193,
    0.7208684087, -0.7843156457, -0.2817355990, -0.1736787707,
    0.4222335219, -0.4350655377, 0.4333281815, 0.5333415866,
    0.1847889870, 0.7498788238, 0.6096553802, -0.3021556735,
    -0.3898189068, 0.0377884321, 0.4024685621, 0.2031257302,
    0.0107116764, -0.3112498820, 0.1999502629, -0.2273492515,
    0.2892593443, 0.5372074246, 0.9408631325, 0.2907505929,
)

# Lloyd-Max centroids for N(0, 1/128) (post-rotation components of a
# unit 128-vector). ISO3 uses the 8-point table (index = 2 low bits in qs
# + high bit in signs), ISO4 the 16-point table (nibble in qs).
ISO3_CENTROIDS = (
    -0.1906850000, -0.1178320000, -0.0657170000, -0.0214600000,
    0.0214600000, 0.0657170000, 0.1178320000, 0.1906850000,
)
ISO4_CENTROIDS = (
    -0.1739260000, -0.1171950000, -0.0895270000, -0.0687560000,
    -0.0512620000, -0.0355970000, -0.0209890000, -0.0069380000,
    0.0069380000, 0.0209890000, 0.0355970000, 0.0512620000,
    0.0687560000, 0.0895270000, 0.1171950000, 0.1739260000,
)


def check_format(fmt: str) -> None:
    if fmt not in FORMATS:
        raise ValueError(f"unknown iso format {fmt!r} (expected one of {FORMATS})")


def block_bytes(fmt: str) -> int:
    check_format(fmt)
    return ISO3_BLOCK_BYTES if fmt == "iso3" else ISO4_BLOCK_BYTES


def packed_row_bytes(head_dim: int, fmt: str) -> int:
    """Bytes one quantized head vector of ``head_dim`` values occupies."""
    if head_dim % QK_ISO != 0:
        raise ValueError(f"head_dim {head_dim} not divisible by {QK_ISO}")
    return head_dim // QK_ISO * block_bytes(fmt)


def _centroids(fmt: str) -> tuple[float, ...]:
    return ISO3_CENTROIDS if fmt == "iso3" else ISO4_CENTROIDS


@functools.cache
def _tables(device: torch.device, fmt: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(quaternions [32, 4] as (w, x, y, z), centroids [8 or 16]) on ``device``."""
    quat = torch.tensor(
        list(zip(ISO_QW, ISO_QX, ISO_QY, ISO_QZ)), dtype=torch.float32, device=device
    )
    cent = torch.tensor(_centroids(fmt), dtype=torch.float32, device=device)
    return quat, cent


def _hamilton_left(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Left Hamilton product q ⊗ v. q: [32, 4], v: [..., 32, 4] (w, x, y, z)."""
    qw, qx, qy, qz = q.unbind(-1)
    v0, v1, v2, v3 = v.unbind(-1)
    return torch.stack(
        (
            qw * v0 - qx * v1 - qy * v2 - qz * v3,
            qw * v1 + qx * v0 + qy * v3 - qz * v2,
            qw * v2 - qx * v3 + qy * v0 + qz * v1,
            qw * v3 + qx * v2 - qy * v1 + qz * v0,
        ),
        dim=-1,
    )


def quantize_ref(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """Quantize rows of ``x`` (..., D) to packed ISO blocks (..., D//128 * B) uint8.

    Reference implementation mirroring quantize_row_iso3_0_ref /
    quantize_row_iso4_0_ref (brute-force nearest centroid, tie -> lower index,
    corrected norm rounded to fp16).
    """
    check_format(fmt)
    if x.shape[-1] % QK_ISO != 0:
        raise ValueError(f"last dim {x.shape[-1]} not divisible by {QK_ISO}")
    quat, cent = _tables(x.device, fmt)
    nb = x.shape[-1] // QK_ISO
    lead = x.shape[:-1]

    xf = x.float().reshape(*lead, nb, QK_ISO)
    grp_norm = xf.norm(dim=-1)  # [..., nb]
    inv = torch.where(grp_norm > 1e-10, grp_norm.reciprocal(), torch.zeros_like(grp_norm))
    xn = (xf * inv.unsqueeze(-1)).reshape(*lead, nb, ISO_N_GROUPS, 4)

    rotated = _hamilton_left(quat, xn)  # [..., nb, 32, 4]
    dist = (rotated.unsqueeze(-1) - cent).abs()
    idx = dist.argmin(dim=-1)  # first-min wins, matches C strict '<'  [..., nb, 32, 4]
    chosen = cent[idx]

    recon_sq = chosen.square().sum(dim=(-2, -1))  # [..., nb]
    recon_norm = recon_sq.sqrt()
    corrected = torch.where(recon_norm > 1e-10, grp_norm / recon_norm, grp_norm)
    norm_h = corrected.half()  # fp16 rounding at store time

    idx = idx.reshape(*lead, nb, QK_ISO)
    norm_bytes = norm_h.view(torch.uint8).reshape(*lead, nb, 2)
    if fmt == "iso3":
        low = (idx & 0x3).to(torch.uint8).reshape(*lead, nb, QK_ISO // 4, 4)
        qs = (
            low[..., 0]
            | (low[..., 1] << 2)
            | (low[..., 2] << 4)
            | (low[..., 3] << 6)
        )  # [..., nb, 32]
        hi = ((idx >> 2) & 0x1).to(torch.uint8).reshape(*lead, nb, QK_ISO // 8, 8)
        signs = torch.zeros_like(qs[..., : QK_ISO // 8])
        for b in range(8):
            signs |= hi[..., b] << b
        block = torch.cat((norm_bytes, qs, signs), dim=-1)
    else:
        nib = idx.to(torch.uint8).reshape(*lead, nb, QK_ISO // 2, 2)
        qs = nib[..., 0] | (nib[..., 1] << 4)  # [..., nb, 64]
        rnorm = torch.zeros_like(norm_bytes)
        block = torch.cat((norm_bytes, rnorm, qs), dim=-1)
    return block.reshape(*lead, nb * block.shape[-1])


def dequantize_ref(packed: torch.Tensor, fmt: str, head_dim: int) -> torch.Tensor:
    """Inverse of :func:`quantize_ref`; returns float32 (..., head_dim)."""
    check_format(fmt)
    if head_dim % QK_ISO != 0:
        raise ValueError(f"head_dim {head_dim} not divisible by {QK_ISO}")
    quat, cent = _tables(packed.device, fmt)
    nb = head_dim // QK_ISO
    bb = block_bytes(fmt)
    lead = packed.shape[:-1]
    if packed.shape[-1] != nb * bb:
        raise ValueError(f"packed row {packed.shape[-1]} != {nb * bb} for {fmt}")

    blocks = packed.reshape(*lead, nb, bb)
    norm = blocks[..., 0:2].reshape(*lead, nb, 1, 2).contiguous().view(torch.float16)
    norm = norm.reshape(*lead, nb, 1).float()  # [..., nb, 1]
    if fmt == "iso3":
        qs = blocks[..., 2 : 2 + QK_ISO // 4].long()  # [..., nb, 32]
        signs = blocks[..., 2 + QK_ISO // 4 : bb].long()  # [..., nb, 16]
        low = torch.stack(
            [(qs >> s) & 0x3 for s in (0, 2, 4, 6)], dim=-1
        ).reshape(*lead, nb, QK_ISO)
        hi = torch.stack(
            [(signs >> b) & 0x1 for b in range(8)], dim=-1
        ).reshape(*lead, nb, QK_ISO)
        idx = low | (hi << 2)
    else:
        qs = blocks[..., 4:bb].long()  # [..., nb, 64]
        idx = torch.stack((qs & 0xF, (qs >> 4) & 0xF), dim=-1).reshape(*lead, nb, QK_ISO)

    chosen = cent[idx].reshape(*lead, nb, ISO_N_GROUPS, 4)
    conj = quat * torch.tensor([1.0, -1.0, -1.0, -1.0], device=packed.device)
    out = _hamilton_left(conj, chosen)  # [..., nb, 32, 4]
    return (out.reshape(*lead, nb, QK_ISO) * norm).reshape(*lead, head_dim)


# ---------------------------------------------------------------------------
# CUDA kernels (JIT via tvm-ffi, sources in csrc/jit/iso_store.cu)
# ---------------------------------------------------------------------------


@functools.cache
def _iso_module(fmt: str):
    bits = 3 if fmt == "iso3" else 4
    cpp_args = make_cpp_args(bits, 128)  # <fmt_bits, num_threads>
    return load_jit(
        "iso_store",
        *cpp_args,
        cuda_files=["iso_store.cu"],
        cuda_wrappers=[
            ("store", f"IsoStoreKernel<{cpp_args}>::run"),
            ("dequant", f"IsoDequantKernel<{cpp_args}>::run"),
        ],
    )


def iso_store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    nheads: int,
    head_dim: int,
    fmt: str,
) -> None:
    """Quantize k/v rows (N, nheads*head_dim) bf16 into the packed paged caches.

    ``k_cache``/``v_cache`` are (total_slots, nheads * head_dim//128 * B) uint8
    views; ``indices`` are the destination token slots (int32/int64, CUDA).
    """
    check_format(fmt)
    _iso_module(fmt).store(k_cache, v_cache, indices, k, v, nheads, head_dim)


def iso_dequant_rows(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    nheads: int,
    head_dim: int,
    fmt: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize packed rows back to (N, nheads*head_dim) bf16 (debug/tests)."""
    check_format(fmt)
    n = indices.numel()
    k = torch.empty(n, nheads * head_dim, dtype=torch.bfloat16, device=k_cache.device)
    v = torch.empty(n, nheads * head_dim, dtype=torch.bfloat16, device=k_cache.device)
    _iso_module(fmt).dequant(k, v, k_cache, v_cache, indices, nheads, head_dim)
    return k, v


@functools.cache
def _iso_attn_module(fmt: str, nb: int, gt: int):
    bits = 3 if fmt == "iso3" else 4
    cpp_args = make_cpp_args(bits, nb, gt)  # <fmt_bits, nb, gt>
    return load_jit(
        "iso_attention",
        *cpp_args,
        cuda_files=["iso_attention.cu"],
        cuda_wrappers=[
            ("decode", f"IsoAttnDecodeKernel<{cpp_args}>::run"),
            ("extend", f"IsoAttnExtendKernel<{cpp_args}>::run"),
        ],
    )


def _nb_gt(nheads_q: int, nheads_kv: int, head_dim: int) -> tuple[int, int]:
    if head_dim % QK_ISO != 0:
        raise ValueError(f"head_dim {head_dim} not divisible by {QK_ISO}")
    if nheads_q % nheads_kv != 0:
        raise ValueError(f"nheads_q {nheads_q} not divisible by nheads_kv {nheads_kv}")
    nb = head_dim // QK_ISO
    gq = nheads_q // nheads_kv
    gt = 1
    while gt < gq:
        gt <<= 1
    if gt > 16:
        raise ValueError(f"GQA group size {gq} too large (max 16)")
    return nb, gt


def iso_attention_decode(
    q: torch.Tensor,
    out: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    nheads_q: int,
    nheads_kv: int,
    head_dim: int,
    scale: float,
    fmt: str,
) -> None:
    """Decode attention over the packed pool: q (R, nheads_q*head_dim) bf16."""
    check_format(fmt)
    nb, gt = _nb_gt(nheads_q, nheads_kv, head_dim)
    _iso_attn_module(fmt, nb, gt).decode(
        q, out, k_cache, v_cache, kv_indptr, kv_indices, nheads_q, nheads_kv,
        head_dim, scale,
    )


def iso_attention_extend(
    q: torch.Tensor,
    out: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_ext: torch.Tensor,
    v_ext: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    nheads_q: int,
    nheads_kv: int,
    head_dim: int,
    scale: float,
    max_extend: int,
    fmt: str,
) -> None:
    """Extend/prefill attention: packed prefix (kv_indptr/kv_indices) plus
    causal bf16 extend rows k_ext/v_ext (deferred quantization contract)."""
    check_format(fmt)
    nb, gt = _nb_gt(nheads_q, nheads_kv, head_dim)
    _iso_attn_module(fmt, nb, gt).extend(
        q, out, k_cache, v_cache, k_ext, v_ext, qo_indptr, kv_indptr, kv_indices,
        nheads_q, nheads_kv, head_dim, scale, max_extend,
    )
