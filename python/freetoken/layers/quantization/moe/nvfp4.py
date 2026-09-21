"""NVFP4 experts: served from the offload cache through the Triton inline-dequant kernels, vLLM's Marlin or flashinfer's b12x.

FreeToken stores MoE experts as ModelOpt NVFP4 (packed e2m1 codes + fp8-e4m3 per-16
block scales + per-tensor global scale) in pinned host banks and gathers the routed
experts into a GPU slot cache. Rules every kernel here follows:

* A kernel owns ``{load-time pack, forward}`` as a unit. Prefill and decode consume
  the *same* bank layout -- there is never a second copy of the experts.
* Every pack keeps the expert dimension outermost with per-expert blocks contiguous
  and byte-identical in size to the native layout, so the banks are repacked *in
  place* and the offload cache's slot gather (``copy_missing``) works unchanged.
* The per-tensor global scales never enter the offload banks: they are tiny
  ``[L*E]`` vectors kept resident on the GPU and gathered per forward call
  (device-side, CUDA-graph safe).
"""

from __future__ import annotations

import math

import torch

from freetoken.kernel import backend
from freetoken.utils import init_logger

from ..registry import LayerKind, register_method
from ..scheme import NVFP4_GROUP as GROUP, QuantKind
from .base import BankSpec, ExpertView, fused_global, fused_piece, gated_epilogue_reason, global_rows, limit_or_inf, MoEConfig, MoEKernel, MoEMethod

logger = init_logger(__name__)

FP8 = torch.float8_e4m3fn
MARLIN_MAX_SLOTS = 992
B12X_MIN_INTERMEDIATE = 1024


class TritonNvfp4MoEKernel(MoEKernel):
    """FreeToken's inline-dequant kernels over the native ModelOpt rows."""

    name = "triton"
    cpu_format = "nvfp4"

    def unusable_reason(self, cfg: MoEConfig) -> str | None:
        reason = self._common_reject(cfg, resident_ok=False, tp_ok=False, cpu_ok=True, plain_silu_only=False)
        if reason:
            return reason
        reason = gated_epilogue_reason(cfg)
        return f"triton nvfp4 MoE kernel: {reason}" if reason else None

    def layout(self, cfg: MoEConfig) -> dict[str, BankSpec]:
        i, h = cfg.intermediate, cfg.hidden
        return {
            "gate_up": BankSpec((2 * i, h // 2), torch.uint8),
            "gate_up_scale": BankSpec((2 * i, h // GROUP), FP8),
            "gate_up_global": BankSpec((2 * i,), torch.float16),
            "down": BankSpec((h, i // 2), torch.uint8),
            "down_scale": BankSpec((h, i // GROUP), FP8),
            "down_global": BankSpec((h,), torch.float16),
        }

    def pack(self, pieces, cfg: MoEConfig, out):
        out["gate_up"].copy_(fused_piece(pieces, "gate_up"))
        out["gate_up_scale"].copy_(fused_piece(pieces, "gate_up_scale"))
        out["gate_up_global"].copy_(fused_global(pieces, cfg.intermediate))
        out["down"].copy_(pieces["down"])
        out["down_scale"].copy_(pieces["down_scale"])
        out["down_global"].copy_(global_rows(pieces["down_global"], cfg.hidden))
        return {}

    def apply(self, layer, x, topk_weights, topk_ids, view: ExpertView, *, is_prefill: bool):
        from freetoken.moe.fused_nvfp4 import fused_experts_decode_nvfp4_marlin, fused_experts_nvfp4

        t = view.tensors
        banks = (t["gate_up"], t["gate_up_scale"], t["gate_up_global"], t["down"], t["down_scale"], t["down_global"])
        alpha, limit = float(layer.alpha), limit_or_inf(layer)
        if is_prefill:
            return fused_experts_nvfp4(x, *banks, topk_weights, topk_ids, view.n, layer.activation, layer.apply_router_weight_on_input, alpha, limit)
        return fused_experts_decode_nvfp4_marlin(x, *banks, topk_weights, topk_ids, layer.activation, layer.apply_router_weight_on_input, alpha, limit)


# ---------------------------------------------------------------------------
# Marlin: vLLM's fused Marlin MoE (W4A16, dequant in kernel) over Marlin-tiled banks.
# The Marlin kernels come from vLLM (not sgl-kernel) deliberately: vLLM ships the
# NVFP4 instantiations in its AOT wheel together with the matching host-side layout
# transforms, so the pair stays consistent by construction. sgl-kernel's AOT wheel
# excludes the NVFP4 MoE Marlin kernels to cut wheel size (its python transforms pair
# with sglang's *JIT* kernel tree, whose scale encoding has since diverged).
#
# The small layout-transform helpers are ported from sglang
# (``srt/layers/quantization/marlin_utils{,_fp4}.py``, Apache-2.0); the kernels
# themselves are imported, not vendored.
# ---------------------------------------------------------------------------


def _marlin_symbols_ok() -> bool:
    """Probe the exact vLLM symbols the pack/forward paths below use.

    ``find_spec`` only proves the top-level package exists; an incompatible donor
    version would otherwise crash mid-init, after minutes of bank loading. Probing
    at selection time degrades to triton before any bank is touched.
    """
    try:
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
    except Exception as exc:
        logger.warning(
            f"NVFP4 marlin backend is installed but unusable ({exc!r}); "
            "falling back to the Triton inline-dequant backend"
        )
        return False
    return True


# Marlin pack: mirrors vLLM's prepare_moe_fp4_layer_for_marlin so the host prep always matches the AOT op.
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


class MarlinNvfp4MoEKernel(MoEKernel):
    """vLLM's fused Marlin MoE over pre-tiled banks; the globals fold into GPU-resident alphas."""

    name = "marlin"
    max_slots = MARLIN_MAX_SLOTS

    def unusable_reason(self, cfg: MoEConfig) -> str | None:
        if not backend.is_vllm_installed():
            return "vLLM is not installed"
        reason = self._common_reject(cfg, resident_ok=False, tp_ok=False, cpu_ok=False, plain_silu_only=True)
        if reason:
            return reason
        if not _marlin_symbols_ok():
            return "vLLM Marlin donor symbols are unusable"
        return None

    def worth_it(self, cfg: MoEConfig) -> bool:
        return (8, 0) <= backend.device_capability() < (10, 0)

    def layout(self, cfg: MoEConfig) -> dict[str, BankSpec]:
        i, h = cfg.intermediate, cfg.hidden
        return {
            "gate_up": BankSpec((h // GROUP, 4 * i), torch.int32),
            "gate_up_scale": BankSpec((h // GROUP, 2 * i), FP8),
            "down": BankSpec((i // GROUP, 2 * h), torch.int32),
            "down_scale": BankSpec((i // GROUP, h), FP8),
            "gate_up_alpha": BankSpec((), torch.bfloat16, resident=True),
            "down_alpha": BankSpec((), torch.bfloat16, resident=True),
        }

    def pack(self, pieces, cfg: MoEConfig, out):
        i, h = cfg.intermediate, cfg.hidden
        device = torch.device("cuda")
        gu, gus, gug = fused_piece(pieces, "gate_up"), fused_piece(pieces, "gate_up_scale"), fused_global(pieces, i)
        dn, dns, dng = pieces["down"], pieces["down_scale"], global_rows(pieces["down_global"], h)
        e = gu.shape[0]
        gate_up_alpha = torch.empty(e, dtype=torch.bfloat16, device=device)
        down_alpha = torch.empty(e, dtype=torch.bfloat16, device=device)
        for k in range(e):
            qw, sc, al = _marlin_pack_proj(gu[k].to(device), gus[k].to(device), gug[k].to(device), size_k=h, size_n=2 * i)
            out["gate_up"][k].copy_(qw)
            out["gate_up_scale"][k].copy_(sc.view(FP8))
            gate_up_alpha[k] = al[0]
            qw, sc, al = _marlin_pack_proj(dn[k].to(device), dns[k].to(device), dng[k].to(device), size_k=i, size_n=h)
            out["down"][k].copy_(qw)
            out["down_scale"][k].copy_(sc.view(FP8))
            down_alpha[k] = al[0]
        torch.cuda.synchronize(device)
        return {"gate_up_alpha": gate_up_alpha, "down_alpha": down_alpha}

    def apply(self, layer, x, topk_weights, topk_ids, view: ExpertView, *, is_prefill: bool):
        t = view.tensors
        assert view.alphas is not None
        return marlin_fused_experts(x, t["gate_up"], t["gate_up_scale"], view.alphas[0], t["down"], t["down_scale"], view.alphas[1], topk_weights, topk_ids, layer.activation, layer.apply_router_weight_on_input)


# ---------------------------------------------------------------------------
# b12x: flashinfer's SM12x CuTe-DSL fused MoE (W4A16) over b12x-packed banks.
# The pack (``prepare_w4a16_packed_weights``) runs on any CUDA build, but the fused-MoE
# kernel needs sm_120/121 AND a CUDA>=13 *driver* (it JIT-compiles PTX at runtime). The
# kernel object gates on that, so the forward is only reached on hardware that can run it.
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


def _b12x_symbols_ok() -> bool:
    """Probe the exact flashinfer symbols the pack/forward paths below use.

    ``find_spec`` only proves the top-level package exists; an incompatible donor
    version would otherwise crash mid-init, after minutes of bank loading. Probing
    at selection time degrades to triton before any bank is touched.
    """
    try:
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
            f"NVFP4 b12x backend is installed but unusable ({exc!r}); "
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
    if not _b12x_symbols_ok():
        return "flashinfer b12x donor symbols are unusable"
    return None


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


# Per-device scratch (sized sms*4+2) that run_w4a16_moe reads off the prepared object;
# tiny and shape-independent, so one per device is reused across all layers/decodes.
_B12X_WORKSPACE: dict[torch.device, torch.Tensor] = {}


@torch.no_grad()
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
    when the b12x kernel object has confirmed the runtime supports it.

    The banks already hold flashinfer's prepared (tiled) layout from
    the b12x kernel object's pack, so this wraps them back into a
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


class B12xNvfp4MoEKernel(MoEKernel):
    """flashinfer's SM12x CuTe-DSL W4A16 MoE; the packed blocks reuse the native bank bytes."""

    name = "b12x"

    def unusable_reason(self, cfg: MoEConfig) -> str | None:
        cc = backend.device_capability()
        if cc < (12, 0):
            return f"b12x requires sm_120+, got sm_{cc[0]}{cc[1]}"
        if not backend.is_flashinfer_installed():
            return "flashinfer is not installed"
        reason = self._common_reject(cfg, resident_ok=False, tp_ok=False, cpu_ok=False, plain_silu_only=True)
        if reason:
            return reason
        return _b12x_unusable_reason(cc)

    def worth_it(self, cfg: MoEConfig) -> bool:
        # NOTE: never auto-selected. flashinfer's cute launcher indexes each bank with int32 element offsets, so the GPU slot cache is capped at (2^31 - 1) / elements-per-slot (about 1000 slots for GLM-5.3-Flash's 12960 experts); until flashinfer lifts that, b12x stays behind triton in the table; the selection does not weigh the cap, cache-auto clamps to slot_limit() and OffloadMoeCache refuses a larger cache.
        return cfg.intermediate >= B12X_MIN_INTERMEDIATE

    def slot_limit(self, cfg: MoEConfig) -> int | None:
        # the cute launcher indexes each bank with int32 element offsets
        per_slot = max(math.prod(spec.shape) for spec in self.layout(cfg).values() if not spec.resident)
        return (2**31 - 1) // per_slot

    def layout(self, cfg: MoEConfig) -> dict[str, BankSpec]:
        # flashinfer's prepared tiles: the Marlin shapes, byte-identical to the native rows
        i, h = cfg.intermediate, cfg.hidden
        return {
            "gate_up": BankSpec((h // GROUP, 4 * i), torch.int32),
            "gate_up_scale": BankSpec((h // GROUP, 2 * i), FP8),
            "down": BankSpec((i // GROUP, 2 * h), torch.int32),
            "down_scale": BankSpec((i // GROUP, h), FP8),
            "gate_up_alpha": BankSpec((), torch.float32, resident=True),
            "down_alpha": BankSpec((), torch.float32, resident=True),
        }

    def pack(self, pieces, cfg: MoEConfig, out):
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare import prepare_w4a16_packed_weights

        i, h = cfg.intermediate, cfg.hidden
        device = torch.device("cuda")
        gu = fused_piece(pieces, "gate_up").to(device)
        gug = fused_global(pieces, i).to(device).float()
        g_max = gug.max(dim=1, keepdim=True).values
        gus = fused_piece(pieces, "gate_up_scale").to(device).to(torch.float16)
        ratio = gug / g_max
        if not torch.all(ratio == 1.0):
            gus = (gus.float() * ratio.unsqueeze(-1)).to(torch.float16)
        dng = global_rows(pieces["down_global"], h).to(device)
        if not torch.all(dng == dng[:, :1]):
            raise ValueError("b12x pack requires a row-constant per-expert down_proj global scale")
        # b12x wants up|gate row order
        gu = torch.cat([gu[:, i:], gu[:, :i]], dim=1).contiguous()
        gus = torch.cat([gus[:, i:], gus[:, :i]], dim=1).contiguous()
        prepared = prepare_w4a16_packed_weights(
            gu,
            _b12x_swizzle_block_scales(gus.to(FP8)),
            g_max.squeeze(1),
            pieces["down"].to(device),
            _b12x_swizzle_block_scales(pieces["down_scale"].to(device)),
            dng[:, 0].float(),
            activation="silu",
            params_dtype=torch.bfloat16,
            source_format="modelopt",
        )
        e = gu.shape[0]
        for role, t in (("gate_up", prepared.w13), ("gate_up_scale", prepared.w13_scale), ("down", prepared.w2), ("down_scale", prepared.w2_scale)):
            bank = out[role]
            assert t.shape[1:] == bank.shape[1:] and t.dtype == bank.dtype, (role, t.shape, t.dtype, bank.shape, bank.dtype)
            bank.copy_(t[:e])
        return {"gate_up_alpha": prepared.w13_global_scale[:e].float(), "down_alpha": prepared.w2_global_scale[:e].float()}

    def apply(self, layer, x, topk_weights, topk_ids, view: ExpertView, *, is_prefill: bool):
        t = view.tensors
        assert view.alphas is not None
        return b12x_fused_experts(x, t["gate_up"], t["gate_up_scale"], view.alphas[0], t["down"], t["down_scale"], view.alphas[1], topk_weights, topk_ids, layer.activation, layer.apply_router_weight_on_input)


@register_method(QuantKind.NVFP4, LayerKind.MOE)
class Nvfp4MoEMethod(MoEMethod):
    candidates = (TritonNvfp4MoEKernel, MarlinNvfp4MoEKernel, B12xNvfp4MoEKernel)

    def create_weights(self, layer) -> None:
        raise NotImplementedError("NVFP4 experts are served from the offload cache, not resident")

    def resident_view(self, layer) -> ExpertView:
        raise NotImplementedError("NVFP4 experts are not resident")
