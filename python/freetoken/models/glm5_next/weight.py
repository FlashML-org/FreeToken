"""GLM-5.3-Flash ModelOpt checkpoint reader."""

from __future__ import annotations

import re
from collections.abc import Iterator

import safetensors
import torch
from tqdm import tqdm

from freetoken.distributed import get_tp_info
from freetoken.models.loader import drop_page_cache, iter_weight_files
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
    load_nvfp4_expert_source_banks_parallel,
)

_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
_BASE_LAYER_RE = re.compile(r"^(?:model\.)?language_model\.layers\.(?P<layer>\d+)\.")
# The released checkpoint appends one next-token-prediction block as language
# layer 45 instead of placing it below ``mtp.*``.  The serving graph contains
# decoder layers 0..44 only.
_NUM_BASE_LAYERS = 45
_EXPERT_KEY_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: (
        layer - config.first_k_dense_replace
        if config.first_k_dense_replace <= layer < config.num_layers
        else None
    ),
    desc="GLM-5.3-Flash NVFP4 experts",
)

# The release stores one depthwise short-convolution kernel beside each of the
# q/k/v projections.  The runtime executes a single concatenated q|k|v
# convolution, so combine those three small tensors while streaming the shards.
_KDA_CONV_PARTS = (
    ".self_attn.q_conv1d.weight",
    ".self_attn.k_conv1d.weight",
    ".self_attn.v_conv1d.weight",
)
_KDA_CONV_FUSED = ".self_attn.conv1d.weight"


def _rename(raw: str) -> str | None:
    layer_match = _BASE_LAYER_RE.match(raw)
    if (
        raw.startswith(("model.visual.", "visual.", "mtp."))
        or _EXPERT_RE.search(raw)
        or (
            layer_match is not None
            and int(layer_match.group("layer")) >= _NUM_BASE_LAYERS
        )
    ):
        return None
    if raw.endswith((".weight_scale", ".weight_scale_2", ".input_scale")):
        return None
    if raw.startswith("model.language_model."):
        name = "model." + raw[len("model.language_model.") :]
    elif raw.startswith("language_model."):
        name = "model." + raw[len("language_model.") :]
    else:
        name = raw
    for checkpoint, runtime in (
        (".hc_attn_fn", ".attn_hc.fn"),
        (".hc_attn_base", ".attn_hc.base"),
        (".hc_attn_scale", ".attn_hc.scale"),
        (".hc_ffn_fn", ".ffn_hc.fn"),
        (".hc_ffn_base", ".ffn_hc.base"),
        (".hc_ffn_scale", ".ffn_hc.scale"),
        (".mlp.gate.e_score_correction_bias", ".mlp.e_score_correction_bias"),
    ):
        if checkpoint in name:
            name = name.replace(checkpoint, runtime)
    return name


def _try_fuse_kda_conv(
    name: str,
    tensor: torch.Tensor,
    buf: dict[str, dict[int, torch.Tensor]],
) -> tuple[str, torch.Tensor] | tuple[()] | None:
    """Fuse checkpoint q/k/v depthwise kernels in runtime channel order.

    Returns ``None`` for an unrelated tensor, ``()`` while a layer is still
    incomplete, and ``(runtime_name, concatenated_tensor)`` after all three
    pieces have arrived.  The buffer intentionally lives across shard files.
    """
    for idx, suffix in enumerate(_KDA_CONV_PARTS):
        if not name.endswith(suffix):
            continue
        key = name[: -len(suffix)] + _KDA_CONV_FUSED
        slots = buf.setdefault(key, {})
        if idx in slots:
            raise ValueError(f"duplicate GLM-5.3 KDA convolution part: {name}")
        slots[idx] = tensor
        if len(slots) < len(_KDA_CONV_PARTS):
            return ()
        del buf[key]
        return key, torch.cat([slots[i] for i in range(len(_KDA_CONV_PARTS))], dim=0)
    return None


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    if get_tp_info().size > 1:
        raise NotImplementedError("GLM-5.3 weight loading supports TP=1 only")
    if include_moe_experts:
        raise ValueError("GLM-5.3 NVFP4 experts must use the offload source banks")
    if not include_non_moe:
        return
    conv_buf: dict[str, dict[int, torch.Tensor]] = {}
    for file in tqdm(
        iter_weight_files(model_path), desc="Loading GLM-5.3 dense weights"
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            # ``safe_open`` exposes keys but is not itself iterable.
            for raw in f.keys():  # noqa: SIM118
                name = _rename(raw)
                if name is not None:
                    tensor = f.get_tensor(raw)
                    fused = _try_fuse_kda_conv(name, tensor, conv_buf)
                    if fused is not None:
                        if fused != ():
                            yield fused
                        continue
                    yield name, tensor
        drop_page_cache(file)
    if conv_buf:
        missing = {
            key: [
                _KDA_CONV_PARTS[i]
                for i in range(len(_KDA_CONV_PARTS))
                if i not in parts
            ]
            for key, parts in conv_buf.items()
        }
        raise ValueError(f"incomplete GLM-5.3 KDA convolution fusions: {missing}")


def load_nvfp4_expert_sources(model_path: str, config, *, layer_sink=None):
    return load_nvfp4_expert_source_banks(
        model_path,
        config,
        _SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        layer_sink=layer_sink,
    )


def load_nvfp4_expert_sources_parallel(
    model_path: str, config, *, workers: int = 8, chunk: int = 8 << 20, layer_sink=None
):
    return load_nvfp4_expert_source_banks_parallel(
        model_path,
        config,
        _SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        workers=workers,
        chunk=chunk,
        layer_sink=layer_sink,
    )


__all__ = [
    "iter_weights",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
]
