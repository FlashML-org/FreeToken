"""NVFP4 expert-GEMM backends borrowed from vLLM (Marlin) and flashinfer (b12x).

FreeToken stores MoE experts as ModelOpt NVFP4 (packed e2m1 codes + fp8-e4m3 per-16
block scales + per-tensor global scale) in pinned host banks and gathers the routed
experts into a GPU slot cache. This module picks the fastest fused-MoE kernel for
that format by compute capability and owns the matching bank layout:

==========  ==========================  =============================================
backend     compute capability          kernel / weight layout
==========  ==========================  =============================================
marlin      sm_80 .. sm_99              vLLM ``fused_marlin_moe`` (W4A16
                                        dequant-in-kernel); Marlin-tiled weights
b12x        sm_120+ and CUDA>=13        ``flashinfer`` SM12x CuTe-DSL MoE (W4A16);
                                        b12x-packed weights
triton      anything (fallback)         FreeToken's own Triton kernels; the native
                                        row-major ModelOpt layout (used on sm_120 +
                                        CUDA 12.x, where b12x cannot run)
==========  ==========================  =============================================

The Marlin kernels come from vLLM (not sgl-kernel) deliberately: vLLM ships the
NVFP4 instantiations in its AOT wheel together with the matching host-side layout
transforms, so the pair stays consistent by construction. sgl-kernel's AOT wheel
excludes the NVFP4 MoE Marlin kernels to cut wheel size (its python transforms pair
with sglang's *JIT* kernel tree, whose scale encoding has since diverged).

Hard rules:

* A backend owns ``{load-time pack, forward}`` as a unit. Prefill and decode consume
  the *same* bank layout -- there is never a second copy of the experts.
* Every pack keeps the expert dimension outermost with per-expert blocks contiguous
  and byte-identical in size to the native layout, so the banks are repacked *in
  place* and the offload cache's slot gather (``copy_missing``) works unchanged.
* The per-tensor global scales never enter the offload banks: they are tiny
  ``[L*E]`` vectors kept resident on the GPU and gathered per forward call
  (device-side, CUDA-graph safe).

The small layout-transform helpers are ported from sglang
(``srt/layers/quantization/marlin_utils{,_fp4}.py``, Apache-2.0); the kernels
themselves are imported, not vendored.
"""

from __future__ import annotations

import os

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

# The 6 native ModelOpt NVFP4 source bank names (== _BANK_SCHEMAS["nvfp4"]) the repack
# reads, and the 4 post-repack bank names both marlin and b12x emit (== the "nvfp4_marlin"
# / "nvfp4_b12x" schemas); the two *_global banks fold into per-expert alphas, not banks.
_NATIVE_NVFP4_BANKS = (
    "gate_up_packed", "gate_up_scale", "gate_up_global",
    "down_packed", "down_scale", "down_global",
)
_POST_NVFP4_BANKS = ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale")

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def _donor_symbols_ok(backend: str) -> bool:
    """Probe the exact donor symbols the pack/forward paths below use.

    ``find_spec`` only proves the top-level package exists; an incompatible donor
    version would otherwise crash mid-init, after minutes of bank loading. Probing
    at selection time degrades to triton before any bank is touched.
    """
    try:
        if backend == "marlin":
            from vllm import _custom_ops  # noqa: F401
            from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (  # noqa: F401
                fused_marlin_moe,
            )
            from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # noqa: F401
                marlin_permute_scales,
            )
            from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (  # noqa: F401
                nvfp4_marlin_process_global_scale,
                nvfp4_marlin_process_scales,
            )
            from vllm.scalar_type import scalar_types  # noqa: F401
        else:
            assert backend == "b12x"
            # Probe exactly what the pack/forward use: the row-major->swizzled scale
            # transform, the W4A16 prepare, and the prepared-weights launch (which also
            # transitively imports the CuTe-DSL kernel + cutlass).
            from flashinfer import nvfp4_block_scale_interleave  # noqa: F401
            from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (  # noqa: F401
                _launch_sm120_w4a16_moe,
            )
            from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare import (  # noqa: F401
                W4A16PackedWeights,
                _make_workspace,
                prepare_w4a16_packed_weights,
            )
    except Exception as exc:
        logger.warning(
            f"NVFP4 {backend} backend is installed but unusable ({exc!r}); "
            "falling back to the Triton inline-dequant backend"
        )
        return False
    return True


def _flashinfer_cuda_major() -> int | None:
    """flashinfer's own toolkit-CUDA major; None if it can't be determined. Only a
    fallback proxy when the driver version is unavailable -- the b12x kernel is JIT
    PTX-compiled through the driver, so the toolkit major is *not* what gates it."""
    try:
        from flashinfer.jit.cpp_ext import get_cuda_version

        return int(get_cuda_version().major)
    except Exception:
        return None


def _b12x_unusable_reason(cc: tuple[int, int]) -> str | None:
    """None if the flashinfer b12x decode/prefill kernel can actually run on this device,
    else a human-readable reason. The b12x *pack* works anywhere, but the SM12x CuTe-DSL
    kernel itself requires sm_120+ and a CUDA>=13 *driver* (it JIT-compiles PTX through the
    driver), so selection must check the runtime here -- not just that flashinfer imports --
    or the model loads in the b12x layout and then crashes on the first decode."""
    import importlib.util

    if cc < (12, 0):
        return f"b12x requires sm_120+, got sm_{cc[0]}{cc[1]}"
    if importlib.util.find_spec("flashinfer") is None:
        return "flashinfer is not installed"
    from freetoken.kernel.backend import driver_cuda_version

    drv = driver_cuda_version()
    if drv is not None:
        if drv < 13000:
            return (
                "b12x fused MoE requires a CUDA>=13 driver "
                f"(driver supports CUDA {drv // 1000}.{(drv % 1000) // 10})"
            )
    else:
        # Driver version undetermined: fall back to flashinfer's toolkit major as a
        # conservative proxy rather than risk loading the b12x layout and crashing.
        major = _flashinfer_cuda_major()
        if major is not None and major < 13:
            return (
                "b12x fused MoE requires CUDA>=13 (driver undetermined; "
                f"flashinfer toolkit is CUDA {major}.x)"
            )
    if not _donor_symbols_ok("b12x"):
        return "flashinfer b12x donor symbols are unusable"
    return None


def _b12x_min_intermediate() -> int:
    """Smallest ``moe_intermediate_size`` for which ``auto`` prefers b12x over Triton.

    One bank layout is chosen per process (b12x's prepared/tiled weights and Triton's
    native row-major layout are not byte-compatible, so they can't coexist without a 2x
    expert-memory copy), so this is a *load-time* pick by shape. The default (1024) is the
    M=1 single-stream decode crossover measured on RTX 5090 (CUDA-graph %HBM, top_k=8):

        I=512  (Qwen3.x-35B-A3B):  triton 43%  vs  b12x 27%   -> triton
        I=768  (Qwen3-30B-A3B):    triton 41%  vs  b12x 41%   -> triton (tie; b12x wins batched)
        I=1536 (MiniMax-M2):       triton 65%  vs  b12x 82%   -> b12x

    b12x always wins *batched* decode (M>=~4) thanks to tensor cores, so a throughput
    deployment of a small-I model should force it with ``--nvfp4-backend flashinfer``.
    Override the threshold with ``FREETOKEN_NVFP4_B12X_MIN_I``."""
    raw = os.environ.get("FREETOKEN_NVFP4_B12X_MIN_I")
    if raw is None:
        return 1024
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"bad FREETOKEN_NVFP4_B12X_MIN_I={raw!r}; expected an integer"
        ) from exc


def select_nvfp4_backend(
    device: torch.device,
    intermediate_size: int | None = None,
    requested: str = "auto",
    activation: str = "silu",
) -> str:
    """Pick the NVFP4 expert-GEMM backend for ``device`` (and, in ``auto``, ``intermediate_size``).

    ``requested`` is the ``--nvfp4-backend`` choice; a forced backend fails loudly (never
    silently degrades) when it cannot run on this hardware/toolkit:

    * ``auto`` (default) -- capability pick: ``marlin`` on sm_80-99 (+ vLLM), flashinfer
      ``b12x`` on sm_120+ (CUDA>=13 driver) when the MoE is wide enough that its tensor cores
      beat Triton at single-stream M=1 decode (``intermediate_size >= _b12x_min_intermediate()``),
      else the portable Triton inline-dequant GEMV.
    * ``marlin`` / ``flashinfer`` / ``triton`` -- force that backend (and its bank layout),
      raising if it cannot run rather than degrading.

    ``activation`` is the model's routed-expert activation: the borrowed marlin/b12x
    kernels hard-code silu (their fused epilogue), so any other activation (MiniMax-M3's
    ``swigluoai``) resolves ``auto`` to the Triton kernels -- which dispatch the
    activation as a separate elementwise op -- and rejects a forced marlin/flashinfer.

    Returns the internal backend name (``marlin`` / ``b12x`` / ``triton``); ``flashinfer``
    maps to ``b12x``.
    """
    import importlib.util

    requested = (requested or "auto").strip().lower()
    if requested not in ("auto", "marlin", "flashinfer", "triton"):
        raise ValueError(
            f"bad --nvfp4-backend={requested!r}; expected auto, marlin, flashinfer or triton"
        )
    from freetoken.runtime.gpu import is_hip, nvidia_only_error

    if is_hip():
        if requested in ("marlin", "flashinfer"):
            raise nvidia_only_error(
                f"--nvfp4-backend={requested}",
                hint="Use --nvfp4-backend triton (or auto).",
            )
        return "triton"

    if requested == "triton":
        return "triton"
    if activation != "silu":
        if requested != "auto":
            raise RuntimeError(
                f"--nvfp4-backend={requested} only supports silu routed experts (its fused "
                f"epilogue); this model's experts use {activation!r} -- use "
                "--nvfp4-backend triton (or auto)."
            )
        logger.info(
            f"NVFP4 auto backend: routed experts use {activation!r}, which only the "
            "Triton kernels support; using the Triton inline-dequant kernels"
        )
        return "triton"
    if requested == "marlin":
        if device.type != "cuda":
            raise RuntimeError("--nvfp4-backend=marlin requires a CUDA device")
        if not _donor_symbols_ok("marlin"):
            raise RuntimeError(
                "--nvfp4-backend=marlin is unusable: vLLM Marlin donor symbols are unavailable"
            )
        return "marlin"
    if requested == "flashinfer":
        cc = torch.cuda.get_device_capability(device) if device.type == "cuda" else (0, 0)
        reason = _b12x_unusable_reason(cc)
        if reason is not None:
            raise RuntimeError(f"--nvfp4-backend=flashinfer is unavailable: {reason}")
        return "b12x"

    # auto: capability pick, but only land on marlin/b12x when they can really run.
    if device.type != "cuda":
        return "triton"
    cc = torch.cuda.get_device_capability(device)
    if (
        (8, 0) <= cc < (10, 0)
        and importlib.util.find_spec("vllm") is not None
        and _donor_symbols_ok("marlin")
    ):
        return "marlin"
    if cc >= (12, 0):
        reason = _b12x_unusable_reason(cc)
        if reason is None:
            thr = _b12x_min_intermediate()
            if intermediate_size is None or intermediate_size >= thr:
                return "b12x"
            logger.info(
                f"NVFP4 auto backend: b12x is runnable but moe_intermediate_size="
                f"{intermediate_size} < {thr}; using the Triton decode kernels "
                "(faster at single-stream M=1 decode for small-I MoE; set "
                "--nvfp4-backend flashinfer to force b12x for batched throughput)"
            )
            return "triton"
        logger.info(
            f"NVFP4 auto backend: flashinfer b12x unavailable ({reason}); "
            "using the Triton inline-dequant decode kernels"
        )
    return "triton"  # sm_100/103, b12x unusable, or the donor package is missing


# ---------------------------------------------------------------------------
# Marlin pack (kernels and layout transforms borrowed from vLLM; mirrors its
# prepare_moe_fp4_layer_for_marlin so the host prep always matches the AOT op)
# ---------------------------------------------------------------------------


def _marlin_pack_proj(
    packed: torch.Tensor,
    scale: torch.Tensor,
    row_global: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack one expert projection (on GPU) into the Marlin layout.

    ``packed`` ``[N, K//2]`` uint8 e2m1 pairs, ``scale`` ``[N, K//16]`` e4m3,
    ``row_global`` ``[N]`` fp16 per-row global (rows of a merged gate_up may carry
    w1's and w3's different globals). Returns ``(qweight [K//16, 2N] int32,
    scales [K//16, N] e4m3-coded, global scalar bf16)``.

    Marlin takes one global scalar per projection, so when the row globals differ
    the ratio is folded into the fp16 block scales before they are re-encoded
    (exact when all rows share one global, <=1 scale ulp otherwise).
    """
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        nvfp4_marlin_process_global_scale,
        nvfp4_marlin_process_scales,
    )

    assert size_n % 64 == 0, f"Marlin requires N % 64 == 0, got {size_n}"
    qweight = ops.gptq_marlin_repack(
        b_q_weight=packed.view(torch.int32).T.contiguous(),
        perm=torch.empty(0, dtype=torch.int, device=packed.device),
        size_k=size_k,
        size_n=size_n,
        num_bits=4,
    )

    g = row_global.to(torch.float32)
    g_max = g.max()
    s = scale.to(torch.bfloat16)
    if not torch.all(g == g_max):
        s = (s.float() * (g / g_max).unsqueeze(1)).to(torch.bfloat16)
    s = marlin_permute_scales(s.T, size_k=size_k, size_n=size_n, group_size=16)
    s = nvfp4_marlin_process_scales(s)
    if isinstance(s, tuple):  # newer vLLM returns (scales, scale_factor)
        s, factor = s
        g_max = g_max / factor
    g_out = nvfp4_marlin_process_global_scale(g_max.to(torch.bfloat16).reshape(1))
    return qweight, s, g_out


@torch.no_grad()
def marlin_repack_layer(
    layer_banks: dict[str, torch.Tensor],
    config,
    device: torch.device,
    *,
    chunk: int = 32,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Repack ONE layer's 6 native NVFP4 banks into the Marlin layout, in place.

    ``layer_banks`` are that layer's native ``[E, ...]`` tensors keyed by the 6 native
    NVFP4 bank names (``gate_up_packed/scale/global``, ``down_packed/scale/global``).
    The Marlin blocks are byte-identical in size, so the weight/scale storage is
    reinterpreted rather than reallocated; the two ``*_global`` banks are consumed into
    per-expert alphas and are not part of the output layout.

    Returns ``({post-repack bank name -> reinterpreted tensor}, gate_up_alpha, down_alpha)``
    where the dict is keyed by the 4 ``nvfp4_marlin`` bank names and the two alphas are
    ``[E]`` bf16 on ``device``. Staging ``.to(device, non_blocking=True)`` from the native
    source works whether or not it is pinned (pageable -> synchronous copy)."""
    H = config.hidden_size
    I = config.moe_intermediate_size
    gu_packed_l = layer_banks["gate_up_packed"]
    gu_scale_l = layer_banks["gate_up_scale"]
    gu_global_l = layer_banks["gate_up_global"]
    dn_packed_l = layer_banks["down_packed"]
    dn_scale_l = layer_banks["down_scale"]
    dn_global_l = layer_banks["down_global"]
    E = gu_packed_l.size(0)

    gate_up_q = gu_packed_l.view(torch.int32).view(E, H // 16, 4 * I)
    gate_up_s = gu_scale_l.view(E, H // 16, 2 * I)
    down_q = dn_packed_l.view(torch.int32).view(E, I // 16, 2 * H)
    down_s = dn_scale_l.view(E, I // 16, H)
    gate_up_alpha = torch.empty(E, dtype=torch.bfloat16, device=device)
    down_alpha = torch.empty(E, dtype=torch.bfloat16, device=device)

    for start in range(0, E, chunk):
        end = min(start + chunk, E)
        # Stage the native rows on GPU before their (in-place) storage is overwritten.
        gu_p = gu_packed_l[start:end].to(device, non_blocking=True)
        gu_s = gu_scale_l[start:end].to(device, non_blocking=True)
        gu_g = gu_global_l[start:end].to(device, non_blocking=True)
        dn_p = dn_packed_l[start:end].to(device, non_blocking=True)
        dn_s = dn_scale_l[start:end].to(device, non_blocking=True)
        dn_g = dn_global_l[start:end].to(device, non_blocking=True)
        for i in range(end - start):
            qw, sc, al = _marlin_pack_proj(gu_p[i], gu_s[i], gu_g[i], size_k=H, size_n=2 * I)
            gate_up_q[start + i].copy_(qw, non_blocking=True)
            gate_up_s[start + i].copy_(sc.view(torch.float8_e4m3fn), non_blocking=True)
            gate_up_alpha[start + i] = al[0]
            qw, sc, al = _marlin_pack_proj(dn_p[i], dn_s[i], dn_g[i], size_k=I, size_n=H)
            down_q[start + i].copy_(qw, non_blocking=True)
            down_s[start + i].copy_(sc.view(torch.float8_e4m3fn), non_blocking=True)
            down_alpha[start + i] = al[0]
        torch.cuda.synchronize(device)

    return (
        {
            "gate_up_packed": gate_up_q,
            "gate_up_scale": gate_up_s,
            "down_packed": down_q,
            "down_scale": down_s,
        },
        gate_up_alpha,
        down_alpha,
    )


@torch.no_grad()
def marlin_repack_sources_inplace(
    sources: dict[str, list[torch.Tensor]],
    config,
    device: torch.device,
    *,
    chunk: int = 32,
) -> dict[str, list[torch.Tensor] | torch.Tensor]:
    """Convert the native NVFP4 banks from ``load_nvfp4_expert_sources`` to the
    Marlin layout, in place per layer (thin per-layer loop over
    :func:`marlin_repack_layer`; the Marlin blocks are byte-identical in size, so each
    layer's bank tensor is reinterpreted rather than reallocated).

    Returns the per-layer bank lists for ``OffloadMoeCache.set_bank_sources`` plus
    the processed per-expert global-scale vectors (``gate_up_alpha`` / ``down_alpha``,
    flat ``[L*E]`` bf16 on ``device``, layer-major) that the forward path gathers
    per slot.
    """
    num_layers = len(sources["gate_up_packed"])

    gate_up_q_layers: list[torch.Tensor] = []
    gate_up_s_layers: list[torch.Tensor] = []
    down_q_layers: list[torch.Tensor] = []
    down_s_layers: list[torch.Tensor] = []
    gate_up_alpha_layers: list[torch.Tensor] = []
    down_alpha_layers: list[torch.Tensor] = []

    for layer_id in range(num_layers):
        layer_banks = {name: sources[name][layer_id] for name in _NATIVE_NVFP4_BANKS}
        post, gate_up_alpha, down_alpha = marlin_repack_layer(
            layer_banks, config, device, chunk=chunk
        )
        gate_up_q_layers.append(post["gate_up_packed"])
        gate_up_s_layers.append(post["gate_up_scale"])
        down_q_layers.append(post["down_packed"])
        down_s_layers.append(post["down_scale"])
        gate_up_alpha_layers.append(gate_up_alpha)
        down_alpha_layers.append(down_alpha)

    return {
        "gate_up_packed": gate_up_q_layers,
        "gate_up_scale": gate_up_s_layers,
        "down_packed": down_q_layers,
        "down_scale": down_s_layers,
        "gate_up_alpha": torch.cat(gate_up_alpha_layers),
        "down_alpha": torch.cat(down_alpha_layers),
    }


# ---------------------------------------------------------------------------
# Marlin forward (vLLM's fused_marlin_moe, borrowed wholesale)
# ---------------------------------------------------------------------------


def marlin_fused_experts(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,
    gate_up_s: torch.Tensor,
    gate_up_alpha: torch.Tensor,
    down_q: torch.Tensor,
    down_s: torch.Tensor,
    down_alpha: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    apply_router_weight_on_input: bool,
) -> torch.Tensor:
    """Marlin W4A16 fused MoE (two grouped GEMMs + activation + reduce).

    Two calling regimes share this entry: decode passes the full ``[S]`` slot cache
    with ``topk_ids`` rewritten to slot ids; full-layer prefill passes banks whose
    position == expert id (the materialized ``[:E]`` slot view or the overlap double
    buffer views), so the raw routing ids arrive unmapped. The ``*_alpha`` vectors
    are the matching per-row global scales in both regimes.
    vLLM's implementation is device-side only (no host syncs), so the decode call is
    CUDA-graph capturable.
    """
    from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe
    from vllm.scalar_type import scalar_types

    assert activation == "silu", "Marlin NVFP4 backend supports gated silu only"
    return fused_marlin_moe(
        hidden_states,
        gate_up_q,
        down_q,
        None,  # bias1
        None,  # bias2
        gate_up_s,
        down_s,
        gating_output=None,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_type_id=scalar_types.float4_e2m1f.id,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=gate_up_q.size(0),
        activation=activation,
        global_scale1=gate_up_alpha,
        global_scale2=down_alpha,
    )


# ---------------------------------------------------------------------------
# b12x (flashinfer SM12x CuTe-DSL, W4A16) -- the pack (prepare_w4a16_packed_weights)
# runs on any CUDA build, but the fused-MoE kernel needs sm_120/121 AND a CUDA>=13
# *driver* (it JIT-compiles PTX at runtime). select_nvfp4_backend gates on that, so
# this path is only reached on hardware that can actually run it.
#
# Layout contract (verified against flashinfer 0.6.12 prepare_w4a16_packed_weights):
#   * prepare() takes the block scales already *swizzled* (it calls unswizzle_expert_
#     scales internally), so the native row-major fp8 scales are run through
#     nvfp4_block_scale_interleave first.
#   * prepare() reorders the w13 rows with reorder_w13_to_gate_up == cat([second, first]),
#     i.e. it expects the merged proj as [up, gate] and emits [gate, up]. FreeToken stores
#     [gate, up], so the halves are swapped before prepare() to come out [gate, up] again
#     (the kernel computes silu(first) * second == silu(gate) * up).
#   * modelopt globals are passed straight through (no 1/g inversion).
# The fused forward reconstructs the W4A16PackedWeights from the banks and uses the
# prepared-weights launch (_launch_sm120_w4a16_moe) directly, so weights are prepared
# exactly once at load time (b12x_fused_moe would re-prepare + ptr-cache every call).
# ---------------------------------------------------------------------------


def _b12x_swizzle_block_scales(scale: torch.Tensor) -> torch.Tensor:
    """Row-major per-16 fp8 block scales ``[n, N, K//16]`` -> flashinfer's expert-leading
    *swizzled* storage, which is what :func:`prepare_w4a16_packed_weights` expects on input
    (it unswizzles internally). The 128x4 interleave is per-expert; ``N`` is a multiple of
    128 and ``K//16`` a multiple of 4 for the supported models, so the shape is preserved."""
    from flashinfer import nvfp4_block_scale_interleave

    pieces = [
        nvfp4_block_scale_interleave(scale[e].view(torch.uint8).contiguous())
        for e in range(scale.shape[0])
    ]
    return torch.stack(pieces, dim=0).view(torch.float8_e4m3fn)


@torch.no_grad()
def b12x_repack_layer(
    layer_banks: dict[str, torch.Tensor],
    config,
    device: torch.device,
    *,
    chunk: int = 32,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Repack ONE layer's 6 native NVFP4 banks into flashinfer's SM12x W4A16 layout, in place.

    Mirrors :func:`marlin_repack_layer`; the b12x packed blocks are also byte-identical
    per expert (4-bit weights + e4m3-coded scales), so the weight/scale storage is
    reinterpreted and the two ``*_global`` banks fold into ``[E]`` fp32 alphas. Uses
    flashinfer's own prepare helpers so the layout always matches the installed kernel.

    Returns ``({post-repack bank name -> reinterpreted tensor}, gate_up_alpha, down_alpha)``
    keyed by the 4 ``nvfp4_b12x`` bank names. Staging from the native source works whether
    or not it is pinned (pageable ``.to(device)`` is a synchronous copy)."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare import (
        prepare_w4a16_packed_weights,
    )

    I = config.moe_intermediate_size
    gu_packed_l = layer_banks["gate_up_packed"]
    gu_scale_l = layer_banks["gate_up_scale"]
    gu_global_l = layer_banks["gate_up_global"]
    dn_packed_l = layer_banks["down_packed"]
    dn_scale_l = layer_banks["down_scale"]
    dn_global_l = layer_banks["down_global"]
    E = gu_packed_l.size(0)

    gate_up_alpha = torch.empty(E, dtype=torch.float32, device=device)
    down_alpha = torch.empty(E, dtype=torch.float32, device=device)
    gate_up_q = gate_up_s = down_q = down_s = None

    for start in range(0, E, chunk):
        end = min(start + chunk, E)
        n = end - start
        gu_g = gu_global_l[start:end].to(device)
        # The merged gate_up rows carry w1/w3 globals; b12x also takes one alpha per
        # expert, so fold any ratio into the block scales like the Marlin pack does.
        g = gu_g.float()
        g_max = g.max(dim=1, keepdim=True).values
        gu_s_native = gu_scale_l[start:end].to(device).to(torch.float16)
        ratio = g / g_max
        if not torch.all(ratio == 1.0):
            gu_s_native = (gu_s_native.float() * ratio.unsqueeze(-1)).to(torch.float16)
        # The down bank stores one w2 global broadcast per output row; b12x takes a
        # single alpha per expert, so a row-varying global cannot be represented (the
        # marlin pack folds the ratio into the block scales instead -- do not let such
        # a checkpoint be silently truncated to row 0 here).
        dn_g = dn_global_l[start:end].to(device)
        if not torch.all(dn_g == dn_g[:, :1]):
            raise ValueError("b12x pack requires a row-constant per-expert down_proj global scale")
        # prepare() emits [gate, up] from a [up, gate] input (reorder_w13_to_gate_up),
        # so swap FreeToken's native [gate, up] halves here; swap the (ratio-folded)
        # block scales identically to keep weight/scale rows aligned.
        gu_p = gu_packed_l[start:end].to(device)
        gu_p = torch.cat([gu_p[:, I:], gu_p[:, :I]], dim=1).contiguous()
        gu_s_native = torch.cat(
            [gu_s_native[:, I:], gu_s_native[:, :I]], dim=1
        ).contiguous()
        prepared = prepare_w4a16_packed_weights(
            gu_p,
            _b12x_swizzle_block_scales(gu_s_native.to(torch.float8_e4m3fn)),
            g_max.squeeze(1),
            dn_packed_l[start:end].to(device),
            _b12x_swizzle_block_scales(dn_scale_l[start:end].to(device)),
            dn_g[:, 0].float(),
            activation="silu",
            params_dtype=torch.bfloat16,
            source_format="modelopt",
        )
        if gate_up_q is None:
            def _as_bank(bank: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                assert t[0].nbytes * E == bank.nbytes, (
                    f"b12x block size mismatch: {t[0].nbytes} * {E} != {bank.nbytes}"
                )
                return bank.view(t.dtype).view(E, *t.shape[1:])

            gate_up_q = _as_bank(gu_packed_l, prepared.w13)
            gate_up_s = _as_bank(gu_scale_l, prepared.w13_scale)
            down_q = _as_bank(dn_packed_l, prepared.w2)
            down_s = _as_bank(dn_scale_l, prepared.w2_scale)
        gate_up_q[start:end].copy_(prepared.w13[:n])
        gate_up_s[start:end].copy_(prepared.w13_scale[:n])
        down_q[start:end].copy_(prepared.w2[:n])
        down_s[start:end].copy_(prepared.w2_scale[:n])
        gate_up_alpha[start:end] = prepared.w13_global_scale[:n].float()
        down_alpha[start:end] = prepared.w2_global_scale[:n].float()

    return (
        {
            "gate_up_packed": gate_up_q,
            "gate_up_scale": gate_up_s,
            "down_packed": down_q,
            "down_scale": down_s,
        },
        gate_up_alpha,
        down_alpha,
    )


@torch.no_grad()
def b12x_repack_sources_inplace(
    sources: dict[str, list[torch.Tensor]],
    config,
    device: torch.device,
    *,
    chunk: int = 32,
) -> dict[str, list[torch.Tensor] | torch.Tensor]:
    """Convert the native NVFP4 banks to flashinfer's SM12x W4A16 layout in place, per
    layer (thin per-layer loop over :func:`b12x_repack_layer`).

    Mirrors :func:`marlin_repack_sources_inplace`; the b12x packed blocks are also
    byte-identical per expert (4-bit weights + e4m3-coded scales), so each layer's
    bank tensor is reinterpreted.
    """
    num_layers = len(sources["gate_up_packed"])

    gate_up_q_layers: list[torch.Tensor] = []
    gate_up_s_layers: list[torch.Tensor] = []
    down_q_layers: list[torch.Tensor] = []
    down_s_layers: list[torch.Tensor] = []
    gate_up_alpha_layers: list[torch.Tensor] = []
    down_alpha_layers: list[torch.Tensor] = []

    for layer_id in range(num_layers):
        layer_banks = {name: sources[name][layer_id] for name in _NATIVE_NVFP4_BANKS}
        post, gate_up_alpha, down_alpha = b12x_repack_layer(
            layer_banks, config, device, chunk=chunk
        )
        gate_up_q_layers.append(post["gate_up_packed"])
        gate_up_s_layers.append(post["gate_up_scale"])
        down_q_layers.append(post["down_packed"])
        down_s_layers.append(post["down_scale"])
        gate_up_alpha_layers.append(gate_up_alpha)
        down_alpha_layers.append(down_alpha)

    return {
        "gate_up_packed": gate_up_q_layers,
        "gate_up_scale": gate_up_s_layers,
        "down_packed": down_q_layers,
        "down_scale": down_s_layers,
        "gate_up_alpha": torch.cat(gate_up_alpha_layers),
        "down_alpha": torch.cat(down_alpha_layers),
    }


# Per-device scratch (sized sms*4+2) that run_w4a16_moe reads off the prepared object;
# tiny and shape-independent, so one per device is reused across all layers/decodes.
_B12X_WORKSPACE: dict[torch.device, torch.Tensor] = {}


def _b12x_small_workspace(device: torch.device) -> torch.Tensor:
    ws = _B12X_WORKSPACE.get(device)
    if ws is None:
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare import (
            _make_workspace,
        )

        ws = _make_workspace(device, max_blocks_per_sm=4)
        _B12X_WORKSPACE[device] = ws
    return ws


def b12x_fused_experts(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,
    gate_up_s: torch.Tensor,
    gate_up_alpha: torch.Tensor,
    down_q: torch.Tensor,
    down_s: torch.Tensor,
    down_alpha: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    apply_router_weight_on_input: bool,
) -> torch.Tensor:
    """SM12x W4A16 fused MoE over cache slots (same calling convention as
    :func:`marlin_fused_experts`). Requires sm_120/121 + a CUDA>=13 driver; reached only
    when :func:`select_nvfp4_backend` has confirmed the runtime supports it.

    The banks already hold flashinfer's prepared (tiled) layout from
    :func:`b12x_repack_sources_inplace`, so this wraps them back into a
    ``W4A16PackedWeights`` and calls the prepared-weights launch directly -- the public
    ``b12x_fused_moe`` would re-prepare the raw modelopt weights (and ptr-cache them) on
    every call, which both costs a prepare per step and breaks for the offload cache,
    whose slot contents move between calls.

    CUDA-graph note: the launch resolves a (shape-keyed, module-cached) scratch workspace
    on first use and flashinfer raises if that happens *during* capture, so the decode
    path must be warmed once eagerly before graph capture (FreeToken already does)."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
        _launch_sm120_w4a16_moe,
    )
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare import (
        W4A16PackedWeights,
    )

    assert activation == "silu", "b12x backend supports gated silu only"
    assert not apply_router_weight_on_input
    num_experts = gate_up_q.size(0)
    hidden_size = hidden_states.size(-1)
    # down bank is the prepared w2 == [E, K_tiles, ...] with K_tiles == intermediate//16.
    intermediate_size = down_q.size(1) * 16
    prepared = W4A16PackedWeights(
        w13=gate_up_q,
        w13_scale=gate_up_s,
        w13_global_scale=gate_up_alpha,
        w2=down_q,
        w2_scale=down_s,
        w2_global_scale=down_alpha,
        workspace=_b12x_small_workspace(hidden_states.device),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        is_gated=True,
        params_dtype=hidden_states.dtype,
        source_format="modelopt",
    )
    out = torch.empty(
        hidden_states.size(0), hidden_size, dtype=hidden_states.dtype, device=hidden_states.device
    )
    _launch_sm120_w4a16_moe(
        a=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w1_weight=gate_up_q,
        w1_weight_sf=gate_up_s,
        w1_alpha=gate_up_alpha,
        w2_weight=down_q,
        w2_weight_sf=down_s,
        w2_alpha=down_alpha,
        num_experts=num_experts,
        top_k=topk_ids.size(1),
        num_local_experts=num_experts,
        scatter_output=out,
        activation="silu",
        source_format="modelopt",
        _prepared_weights=prepared,
    )
    return out


__all__ = [
    "select_nvfp4_backend",
    "marlin_repack_layer",
    "marlin_repack_sources_inplace",
    "marlin_fused_experts",
    "b12x_repack_layer",
    "b12x_repack_sources_inplace",
    "b12x_fused_experts",
]
