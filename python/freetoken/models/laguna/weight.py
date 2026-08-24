from __future__ import annotations

import re
from collections.abc import Iterator

import torch

from freetoken.distributed import get_tp_info
from freetoken.models.gguf.dequant import GGML_BF16, GGML_Q4_0, row_bytes
from freetoken.models.loader import ShardReader, drop_page_cache, iter_weight_files

_EXPERT_RE = re.compile(r"^model\.layers\.\d+\.mlp\.experts\.\d+\.(?:gate|up|down)_proj\.")
_MERGE_PARTS = {
    ".mlp.gate_proj.weight": (".mlp.gate_up_proj.weight", "gate"),
    ".mlp.up_proj.weight": (".mlp.gate_up_proj.weight", "up"),
    ".mlp.shared_expert.gate_proj.weight": (".mlp.shared_experts.gate_up_proj.weight", "gate"),
    ".mlp.shared_expert.up_proj.weight": (".mlp.shared_experts.gate_up_proj.weight", "up"),
}


def _rename(name: str) -> str:
    name = name.replace(".self_attn.g_proj.", ".self_attn.gate_proj.")
    name = name.replace(".post_attention_layernorm.", ".ffn_norm.")
    name = name.replace(".mlp.shared_expert.", ".mlp.shared_experts.")
    name = name.replace(".mlp.experts.e_score_correction_bias", ".mlp.e_score_correction_bias")
    return name


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    if get_tp_info().size > 1:
        raise NotImplementedError("Laguna safetensors currently supports TP=1 only")
    if include_moe_experts:
        raise NotImplementedError(
            "Laguna compressed INT4 experts require --moe-backend offload; "
            "resident/fused expert loading is not supported"
        )
    if not include_non_moe:
        return

    import safetensors

    merge: dict[str, dict[str, torch.Tensor]] = {}
    for file in iter_weight_files(model_path):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                if _EXPERT_RE.match(raw_name):
                    continue
                # Scale/shape tensors occur only under routed experts in this artifact,
                # but keep this guard explicit for a useful failure on future variants.
                if raw_name.endswith((".weight_packed", ".weight_scale", ".weight_shape")):
                    raise ValueError(f"unsupported quantized non-expert Laguna tensor: {raw_name}")

                merged = None
                for suffix, (target, slot) in _MERGE_PARTS.items():
                    if raw_name.endswith(suffix):
                        merged = (raw_name[: -len(suffix)] + target, slot)
                        break
                tensor = f.get_tensor(raw_name)
                if merged is None:
                    name = _rename(raw_name)
                    # Laguna routes in fp32: the model intentionally allocates this
                    # parameter as fp32 even though the checkpoint stores it in bf16.
                    if name.endswith(".mlp.gate.weight"):
                        tensor = tensor.float()
                    yield name, tensor
                    continue
                key, slot = merged
                slots = merge.setdefault(_rename(key), {})
                slots[slot] = tensor
                if len(slots) == 2:
                    yield _rename(key), torch.cat([slots["gate"], slots["up"]], dim=0)
                    del merge[_rename(key)]
        drop_page_cache(file)
    assert not merge, f"incomplete Laguna gate/up groups: {sorted(merge)}"


def _ct_int4_to_q4_0(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Repack compressed-tensors symmetric INT4 into GGML Q4_0 row blocks.

    Both encode signed values as an offset nibble (q + 8).  Only nibble ordering
    differs.  Q4_0 stores the scale as fp16, so the checkpoint's bf16 group scale is
    rounded once while the 4-bit values remain bit-exact.
    """
    if packed.dtype is not torch.int32 or scale.dtype is not torch.bfloat16:
        raise ValueError(f"unexpected Laguna INT4 dtypes: {packed.dtype}/{scale.dtype}")
    rows, words = packed.shape
    groups = scale.shape[1]
    if words != groups * 4 or scale.shape[0] != rows:
        raise ValueError(f"unexpected Laguna INT4 geometry: {packed.shape}/{scale.shape}")
    src = packed.contiguous().view(torch.uint8).reshape(rows, groups, 16)
    lo_half, hi_half = src[..., :8], src[..., 8:]
    q0 = torch.stack((lo_half & 0x0F, lo_half >> 4), dim=-1).flatten(-2)
    q1 = torch.stack((hi_half & 0x0F, hi_half >> 4), dim=-1).flatten(-2)
    out = torch.empty((rows, groups, 18), dtype=torch.uint8)
    out[..., :2].copy_(scale.to(torch.float16).contiguous().view(torch.uint8).reshape(rows, groups, 2))
    out[..., 2:].copy_(q0 | (q1 << 4))
    return out.reshape(rows, groups * 18)


def _bank_payloads(config, qtype: int) -> tuple[int, int]:
    H, I = config.hidden_size, config.moe_intermediate_size
    if qtype == GGML_Q4_0:
        return 2 * I * row_bytes(H, qtype), H * row_bytes(I, qtype)
    if qtype == GGML_BF16:
        return 2 * I * H * 2, H * I * 2
    raise ValueError(f"unsupported Laguna safetensors expert type {qtype}")


def load_int4_expert_sources(
    model_path: str, config, *, layer_sink=None
) -> dict[str, list[torch.Tensor]]:
    """Load exact mixed INT4/BF16 expert layers into variable-size flat banks."""
    if get_tp_info().size > 1:
        raise NotImplementedError("Laguna safetensors expert banks currently support TP=1 only")
    from freetoken.moe.host_banks import HostBank, PinPipeline

    types = config.gguf_expert_types
    assert types and len(types) == config.num_moe_layers
    E, H, I = config.num_experts, config.hidden_size, config.moe_intermediate_size
    host = {"gate_up": [], "down": []}
    for gu_t, dn_t in types:
        gu_pay, _ = _bank_payloads(config, gu_t)
        _, dn_pay = _bank_payloads(config, dn_t)
        host["gate_up"].append(HostBank((E, gu_pay), torch.uint8))
        host["down"].append(HostBank((E, dn_pay), torch.uint8))
    banks = {name: [bank.tensor for bank in per_layer] for name, per_layer in host.items()}

    reader = ShardReader(model_path, torch.device("cpu"))
    pins = PinPipeline() if layer_sink is None and torch.cuda.is_available() else None
    try:
        for bank_id, (gu_t, dn_t) in enumerate(types):
            layer = bank_id + config.first_k_dense_replace
            prefix = f"model.layers.{layer}.mlp.experts"
            gu_dst, dn_dst = banks["gate_up"][bank_id], banks["down"][bank_id]
            for expert in range(E):
                ep = f"{prefix}.{expert}"
                if gu_t == GGML_Q4_0:
                    gate = _ct_int4_to_q4_0(
                        reader.get_tensor(ep + ".gate_proj.weight_packed"),
                        reader.get_tensor(ep + ".gate_proj.weight_scale"),
                    )
                    up = _ct_int4_to_q4_0(
                        reader.get_tensor(ep + ".up_proj.weight_packed"),
                        reader.get_tensor(ep + ".up_proj.weight_scale"),
                    )
                    down = _ct_int4_to_q4_0(
                        reader.get_tensor(ep + ".down_proj.weight_packed"),
                        reader.get_tensor(ep + ".down_proj.weight_scale"),
                    )
                    half = I * row_bytes(H, GGML_Q4_0)
                    gu_dst[expert, :half].copy_(gate.reshape(-1))
                    gu_dst[expert, half:].copy_(up.reshape(-1))
                    dn_dst[expert].copy_(down.reshape(-1))
                elif gu_t == dn_t == GGML_BF16:
                    gate = reader.get_tensor(ep + ".gate_proj.weight")
                    up = reader.get_tensor(ep + ".up_proj.weight")
                    down = reader.get_tensor(ep + ".down_proj.weight")
                    if not (gate.dtype is up.dtype is down.dtype is torch.bfloat16):
                        raise ValueError(f"unexpected BF16 Laguna expert dtype at {ep}")
                    gu_dst[expert].copy_(
                        torch.cat((gate, up), dim=0).contiguous().view(torch.uint8).reshape(-1)
                    )
                    dn_dst[expert].copy_(down.contiguous().view(torch.uint8).reshape(-1))
                else:
                    raise ValueError(f"gate/up and down storage differ at Laguna layer {layer}")
            layer_banks = {name: host[name][bank_id] for name in host}
            if layer_sink is not None:
                layer_sink(bank_id, layer_banks)
            elif pins is not None:
                pins(bank_id, layer_banks)
    finally:
        reader.close()
        if pins is not None:
            pins.wait()
        for file in iter_weight_files(model_path):
            drop_page_cache(file)
    return banks


def dummy_int4_expert_sources(config) -> dict[str, list[torch.Tensor]]:
    from freetoken.moe.host_banks import HostBank, pin_banks

    host = {"gate_up": [], "down": []}
    for gu_t, dn_t in config.gguf_expert_types or ():
        gu_pay, _ = _bank_payloads(config, gu_t)
        _, dn_pay = _bank_payloads(config, dn_t)
        host["gate_up"].append(HostBank((config.num_experts, gu_pay), torch.uint8))
        host["down"].append(HostBank((config.num_experts, dn_pay), torch.uint8))
    banks = {name: [bank.tensor for bank in per_layer] for name, per_layer in host.items()}
    for tensor in banks["gate_up"] + banks["down"]:
        tensor.random_(0, 256)
    if torch.cuda.is_available():
        pin_banks(host)
    return banks


__all__ = [
    "_ct_int4_to_q4_0",
    "dummy_int4_expert_sources",
    "iter_weights",
    "load_int4_expert_sources",
]
