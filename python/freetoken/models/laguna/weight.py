"""Laguna weight loading.

Dense tensors (attention q/k/v/o/g, QK norms, router gate, shared experts, layer 0's
dense MLP, embeddings, lm_head, norms) are all bf16 and stream through ``iter_weights``.

Routed experts go to the offload cache as native NVFP4 banks. Two checkpoint quirks
drive ``setup_offload_expert_banks``:

* **compressed-tensors naming.** Laguna ships ``weight_packed`` / ``weight_scale`` /
  ``weight_global_scale`` (W4A16, group 16, ``tensor_group``), not ModelOpt's
  ``weight`` / ``weight_scale`` / ``weight_scale_2``. The shared loader keys off the
  ModelOpt names, so the regex maps ``kind`` onto them.
* **The global scale is quant-side.** ``weight_global_scale`` is ``448*6/amax``; the
  dequant kernel wants its reciprocal (``fp4 * block_scale * global``), so it is
  inverted on the way into the bank. ModelOpt's ``weight_scale_2`` is already
  dequant-side and is stored verbatim -- cf. ``models/loader.py`` and the comment in
  ``models/nvfp4_banks.py``.

* **Mixed precision.** Only layers 1-39 are quantized; layers 40-47 ship bf16 experts
  (the checkpoint's ``ignore`` list). Those are quantized to real NVFP4 here so all 47
  MoE layers share one bank layout and one kernel path.
"""

from __future__ import annotations

import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import iter_weight_files, shard_tensor
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
    load_nvfp4_expert_source_banks_parallel,
)
from freetoken.utils import cached_load_hf_config
from tqdm import tqdm

from .config import parse_config

_EXPERT_RE = re.compile(r"^model\.layers\.\d+\.mlp\.experts\.\d+\.")

# compressed-tensors NVFP4 expert keys. The named groups layer/expert/proj/kind are a
# hard contract with models/nvfp4_banks.py, which indexes matches by group name; ``kind``
# is mapped onto the ModelOpt names the shared loader dispatches on.
_EXPERT_KEY_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\."
    r"(?P<kind>weight_packed|weight_scale|weight_global_scale)$"
)

_E2M1_MAX = 6.0  # largest finite E2M1 magnitude
_FP8_E4M3_MAX = 448.0  # largest finite e4m3 magnitude (block-scale dtype)
# E2M1 magnitudes by code (codes 8-15 are these negated; bit 3 is the sign).
# Mirrors kernel/triton/nvfp4_dequant.py's LUT.
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _is_expert(name: str) -> bool:
    return _EXPERT_RE.search(name) is not None


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Dense (non-expert) weights. Routed experts always go through the offload banks.

    ``include_moe_experts`` is accepted for interface parity but never yields experts:
    Laguna's are NVFP4 (and partly bf16), which the generic stacked-bf16 path cannot
    represent. ``setup_offload_expert_banks`` owns them.
    """
    config = parse_config(cached_load_hf_config(model_path))
    tp = get_tp_info()

    shared_buf: dict[str, dict[str, torch.Tensor]] = {}
    dense_buf: dict[str, dict[str, torch.Tensor]] = {}

    for file in tqdm(iter_weight_files(model_path), desc="Loading weights", disable=not tp.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                # e_score_correction_bias sits under mlp.experts.* in the checkpoint but
                # belongs to the router, so it is a dense tensor despite the prefix.
                is_bias = raw_name.endswith("e_score_correction_bias")
                if _is_expert(raw_name) and not is_bias:
                    continue
                if not include_non_moe:
                    continue

                # HF-faithful key: modeling_laguna.py's _checkpoint_conversion_mapping
                # moves the bias onto the router.
                name = raw_name.replace(
                    "mlp.experts.e_score_correction_bias", "mlp.gate.e_score_correction_bias"
                )

                raw = f.get_tensor(raw_name)
                tensor = shard_tensor(name, raw, rank=tp.rank, world_size=tp.size, num_kv_heads=config.num_kv_heads)
                del raw

                # Fuse gate+up on the output-row axis (gate first), matching
                # LinearColParallelMerged's [gate, up] layout.
                if "shared_expert.gate_proj" in name or "shared_expert.up_proj" in name:
                    prefix = name.split(".shared_expert.")[0] + ".shared_expert"
                    slot = "gate" if "gate_proj" in name else "up"
                    slots = shared_buf.setdefault(prefix, {})
                    slots[slot] = tensor
                    if "gate" in slots and "up" in slots:
                        merged = torch.cat([slots["gate"], slots["up"]], dim=0)
                        del shared_buf[prefix]
                        yield f"{prefix}.gate_up_proj.weight", merged
                    continue
                if name.endswith(("mlp.gate_proj.weight", "mlp.up_proj.weight")):
                    # Dense MLP (layer 0). "mlp.gate.weight" (the router) does not match.
                    mlp_prefix = name[: name.index(".mlp.") + 4]
                    slot = "gate" if "gate_proj" in name else "up"
                    slots = dense_buf.setdefault(mlp_prefix, {})
                    slots[slot] = tensor
                    if "gate" in slots and "up" in slots:
                        merged = torch.cat([slots["gate"], slots["up"]], dim=0)
                        del dense_buf[mlp_prefix]
                        yield f"{mlp_prefix}.gate_up_proj.weight", merged
                    continue

                yield name, tensor

    assert not shared_buf, f"Laguna: incomplete shared_expert merges: {list(shared_buf)}"
    assert not dense_buf, f"Laguna: incomplete dense mlp merges: {list(dense_buf)}"


def _quant_bf16_to_nvfp4(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BF16 ``[O, K]`` -> NVFP4 ``(packed [O, K//2], block_scale [O, K//16], global [O])``.

    Exact inverse of ``kernel/triton/nvfp4_dequant.py``: ``w = E2M1[code] * block_scale
    * global``. Used only for Laguna's bf16 expert layers (40-47) so every MoE layer
    ends up in one bank layout.

    Two-level scaling, matching the packed layers byte-for-byte in convention: the
    global is **per-tensor** (verified against the checkpoint -- for layer 1 expert 0
    ``448*6/amax`` reproduces the stored ``weight_global_scale`` exactly), and each
    16-wide group's scale is stored fp8-e4m3. A per-row global would inflate the scale
    on low-magnitude rows and push the block scales off the e4m3 grid (~2.4% error vs
    ~0.2%). The returned global is already **inverted** (dequant-side), like the packed
    path, and is broadcast to one entry per output row because that is the bank layout.
    """
    O, K = weight.shape
    assert K % 16 == 0, f"NVFP4 group size 16 does not divide K={K}"
    w = weight.float().reshape(O, K // 16, 16)

    # Per-TENSOR global: maps the tensor's amax onto E2M1_MAX * FP8_MAX, so the largest
    # weight is the largest code under the largest block scale.
    amax = w.abs().amax().clamp(min=1e-12).item()
    g_quant = (_FP8_E4M3_MAX * _E2M1_MAX) / amax  # quant-side, as stored in the checkpoint

    # Per-group block scale, quantized through e4m3 and read back: the codes must be
    # chosen against the scale the kernel will actually see, not the ideal one.
    group_amax = w.abs().amax(dim=-1)  # [O, groups]
    block_fp8 = (group_amax * g_quant / _E2M1_MAX).clamp(max=_FP8_E4M3_MAX).to(torch.float8_e4m3fn)

    # Effective per-element step, guarding all-zero groups (scale 0 -> codes 0).
    step = (block_fp8.float() / g_quant).clamp(min=1e-30)
    normalized = w / step[:, :, None]

    # Round the magnitude to the nearest E2M1 value: a non-uniform table, so this is a
    # nearest-neighbour search over 8 magnitudes, not a linear round.
    mags = torch.tensor(_E2M1_MAGNITUDES, dtype=torch.float32, device=weight.device)
    codes = (normalized.abs().unsqueeze(-1) - mags).abs().argmin(dim=-1).to(torch.uint8)
    codes |= (normalized < 0).to(torch.uint8) << 3  # bit 3 is the sign

    # Pack two codes per byte, low nibble = lower K index (nvfp4_dequant.py stores
    # 2*byte_off from the low nibble).
    pairs = codes.reshape(O, K // 2, 2)
    packed = pairs[..., 0] | (pairs[..., 1] << 4)

    g_dequant = torch.full((O,), 1.0 / g_quant, dtype=torch.float16, device=weight.device)
    # The bank dtype is fp16, so a very small amax pushes 1/g into the subnormal range
    # and eventually to zero (which would silently blank the layer). Real Laguna expert
    # tensors sit at amax ~0.09-0.24 -> 1/g ~3e-5..9e-5, subnormal but exact to ~0.02%.
    # Fail loudly rather than emit zeros if a future checkpoint is far smaller.
    assert g_dequant[0].item() > 0.0, (
        f"NVFP4 global scale underflowed fp16 (amax={amax:.3e}); the expert bank dtype "
        "cannot represent this tensor's scale"
    )
    return packed.contiguous(), block_fp8.contiguous(), g_dequant


def _synthesize_bf16_layer(
    reader, moe_layer_ids: list[int], E: int, I: int, bank_layer: int, banks: dict
) -> None:
    """Quantize one bf16 expert layer straight into its banks (layers 40-47).

    Called by the shared loader for bank layers absent from the checkpoint. Reads and
    quantizes one expert at a time, so peak extra memory is a single projection.
    """
    lid = moe_layer_ids[bank_layer]
    for eid in range(E):
        for proj, row_off in (("gate_proj", 0), ("up_proj", I)):
            base = f"model.layers.{lid}.mlp.experts.{eid}.{proj}"
            packed, scale, glob = _quant_bf16_to_nvfp4(reader.get_tensor(f"{base}.weight"))
            banks["gate_up_packed"][eid, row_off : row_off + I] = packed
            banks["gate_up_scale"][eid, row_off : row_off + I] = scale
            banks["gate_up_global"][eid, row_off : row_off + I] = glob
        base = f"model.layers.{lid}.mlp.experts.{eid}.down_proj"
        packed, scale, glob = _quant_bf16_to_nvfp4(reader.get_tensor(f"{base}.weight"))
        banks["down_packed"][eid] = packed
        banks["down_scale"][eid] = scale
        banks["down_global"][eid] = glob


def _laguna_nvfp4_spec(model_path: str, model_config) -> Nvfp4ExpertSourceSpec:
    from freetoken.models.loader import ShardReader

    first_dense = int(model_config.first_k_dense_replace)
    num_layers = int(model_config.num_layers)
    moe_layer_ids = list(range(first_dense, num_layers))
    assert len(moe_layer_ids) == model_config.num_moe_layers, (
        f"Laguna MoE layer count {len(moe_layer_ids)} != {model_config.num_moe_layers}"
    )
    E = int(model_config.num_experts)
    I = int(model_config.moe_intermediate_size)

    # Opened lazily: only touched if the checkpoint actually has bf16 expert layers.
    reader = ShardReader(model_path, torch.device("cpu"))

    return Nvfp4ExpertSourceSpec(
        key_pattern=_EXPERT_KEY_RE,
        proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
        # Experts exist for layers [first_k_dense_replace, num_layers); banks pack by
        # MoE-layer index so the leading dense layer leaves no hole.
        layer_to_bank=lambda layer, config: (
            None
            if layer < config.first_k_dense_replace or layer >= config.num_layers
            else layer - config.first_k_dense_replace
        ),
        desc="Laguna NVFP4 experts",
        kind_map={
            "weight_packed": "weight",
            "weight_scale": "weight_scale",
            "weight_global_scale": "weight_scale_2",
        },
        # compressed-tensors stores the quant-side scale; the kernel wants 1/g.
        global_transform=lambda t: 1.0 / t.float(),
        synthesize_layer=lambda bank_layer, banks: _synthesize_bf16_layer(
            reader, moe_layer_ids, E, I, bank_layer, banks
        ),
    )


def setup_offload_expert_banks(
    model_path: str,
    model_config,
    *,
    device: torch.device,
    dtype: torch.dtype,
    dummy: bool = False,
    parallel: bool = False,
    workers: int = 8,
    chunk: int = 8 << 20,
    decode_target: str = "gpu",
    layer_sink=None,
):
    """Build Laguna's native NVFP4 expert banks (all 47 MoE layers).

    Overrides the generic ``nvfp4`` provider because the checkpoint is mixed precision:
    layers 1-39 ship packed experts, 40-47 ship bf16 ones that are quantized here so a
    single bank layout and kernel path covers every layer. Bank layout, layer-completion
    tracking, pin-after-fill and FTW streaming are the shared machinery.

    The native (6-bank) layout is returned unconditionally: Laguna's ``I`` is 1024, and
    ``select_nvfp4_backend`` keeps narrow-MoE models on the Triton inline-dequant
    kernels, which read exactly this layout.
    """
    from freetoken.models.loader import drop_page_cache
    from freetoken.moe.expert_banks import ExpertBanks

    if dummy:
        from freetoken.models.weight import dummy_nvfp4_expert_sources

        return ExpertBanks("nvfp4", dummy_nvfp4_expert_sources(model_config), streamed=False)

    spec = _laguna_nvfp4_spec(model_path, model_config)
    primary = get_tp_info().is_primary()
    loader = (
        load_nvfp4_expert_source_banks_parallel if parallel else load_nvfp4_expert_source_banks
    )
    kwargs = {"workers": workers, "chunk": chunk} if parallel else {}
    sources = loader(
        model_path, model_config, spec,
        drop_page_cache=drop_page_cache, primary=primary, layer_sink=layer_sink, **kwargs,
    )
    return ExpertBanks("nvfp4", sources, streamed=layer_sink is not None)


__all__ = ["iter_weights", "setup_offload_expert_banks"]
