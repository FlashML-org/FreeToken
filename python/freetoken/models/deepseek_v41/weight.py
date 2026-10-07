"""Weight loading for DeepSeek-V4.1-Flash.

* :func:`iter_weights` streams the resident (non-expert) tensors keyed by the model's attribute
  paths: checkpoint ``layers.N.*`` -> ``model.layers.N.*``, ``embed`` / ``norm`` -> ``model.*``,
  ``head`` -> ``head``. fp8 linears keep their e4m3 payload and e8m0 ``scale`` (declared by the quant
  method as ``weight_scale_inv``); ``wo_a`` keeps its e4m3 payload and ``wo_a_scale`` too (the layer
  dequantizes in-kernel for the reference's bf16 grouped einsum). The engine casts everything else to
  the declared parameter dtype (fp32 compressor weights, hc gates, engram gate weights).
* :func:`iter_expert_pieces` streams the routed MXFP4 experts (e2m1 pairs + e8m0 per-32 scales) per
  expert for the offload banks.

Dropped: the DSpark ``mtp.*`` stage; vision weights and ``gate.bias_vl`` when serving text-only.
The Engram tables (``layers.{1,14}.engram.embed.*``) are
not state-dict tensors: the disk row store maps them in place (``engram_table.py``); an FTW conversion
carries them as side files (:func:`ftw_side_files`).
"""

from __future__ import annotations

from contextlib import ExitStack
import json
import os
import re

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.layers.quantization import QuantKind
from freetoken.models.loader import drop_page_cache
from tqdm import tqdm

from .args import DeepseekV41Args, Mode


class _ShardReader:
    def __init__(self, folder: str, weight_map: dict, device):
        self._folder = folder
        self._weight_map = weight_map
        self._device = str(device)
        self._handles: dict[str, object] = {}
        self._stack = ExitStack()

    def has(self, name: str) -> bool:
        return name in self._weight_map

    def get(self, name: str) -> torch.Tensor:
        shard = self._weight_map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = self._stack.enter_context(safetensors.safe_open(os.path.join(self._folder, shard), framework="pt", device=self._device))
            self._handles[shard] = handle
        return handle.get_tensor(name)

    def close(self) -> None:
        try:
            self._stack.close()
        finally:
            for shard in self._handles:
                drop_page_cache(os.path.join(self._folder, shard))
            self._handles.clear()


def weight_map(model_path: str) -> dict:
    with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def load_args(model_path: str) -> DeepseekV41Args:
    from freetoken.utils.hf import cached_load_hf_config

    return DeepseekV41Args.from_hf(cached_load_hf_config(model_path))


def dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Square block-scaled e4m3 -> bf16; ``scale`` holds e8m0 codes (``2^(code-127)``), the block
    edge follows from the shapes."""
    n, k = weight.shape
    block = n // scale.shape[0]
    assert (n // block, k // block) == tuple(scale.shape), (weight.shape, scale.shape)
    s = torch.exp2(scale.view(torch.uint8).to(torch.float32) - 127.0)
    s = s.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    return (weight.to(torch.float32) * s).to(torch.bfloat16)


def iter_weights(model_path: str, device, *, include_moe_experts: bool = True, include_non_moe: bool = True, include_vision: bool = True):
    args = load_args(model_path)
    reader = _ShardReader(model_path, weight_map(model_path), device)
    get = reader.get
    def linear(src: str, dst: str):
        yield f"{dst}.weight", get(f"{src}.weight")
        if reader.has(f"{src}.scale"):
            yield f"{dst}.weight_scale_inv", get(f"{src}.scale")

    try:
        if include_moe_experts:
            # Resident experts support unquantized checkpoints; FP4 uses the offload cache.
            if reader.has("layers.0.ffn.experts.0.w1.scale"):
                raise ValueError(
                    "DeepSeek-V4.1's fp4 routed experts are served from the offload cache; run with "
                    "--moe-strategy offload (include_moe_experts must be False)."
                )
            for L in range(args.n_layers):
                p = f"layers.{L}.ffn.experts"
                gate_up = torch.stack([torch.cat([get(f"{p}.{e}.w1.weight"), get(f"{p}.{e}.w3.weight")]) for e in range(args.n_routed_experts)])
                down = torch.stack([get(f"{p}.{e}.w2.weight") for e in range(args.n_routed_experts)])
                yield f"model.layers.{L}.ffn.experts.gate_up_proj", gate_up
                yield f"model.layers.{L}.ffn.experts.down_proj", down
        if not include_non_moe:
            return
        if include_vision and reader.has("vision.patch_embed.proj.weight"):
            yield from iter_vision_weights(model_path, device)
        yield "model.embed.weight", get("embed.weight")
        yield "model.norm.weight", get("norm.weight")
        yield "head.weight", get("head.weight")
        for L in range(args.n_layers):
            role = args.roles[L]
            a, m = f"layers.{L}.attn", f"model.layers.{L}.attn"
            yield from linear(f"{a}.wq_a", f"{m}.wq_a")
            yield f"{m}.q_norm.weight", get(f"{a}.q_norm.weight")
            yield from linear(f"{a}.wq_b", f"{m}.wq_b")
            yield from linear(f"{a}.wkv", f"{m}.wkv")
            yield f"{m}.kv_norm.weight", get(f"{a}.kv_norm.weight")
            yield f"{m}.wo_a", get(f"{a}.wo_a.weight")
            if reader.has(f"{a}.wo_a.scale"):  # the fp8 payload + scale; the layer dequantizes in-kernel
                yield f"{m}.wo_a_scale", get(f"{a}.wo_a.scale")
            yield from linear(f"{a}.wo_b", f"{m}.wo_b")
            yield f"{m}.attn_sink", get(f"{a}.attn_sink")
            if role.mode is Mode.FULL:
                c = f"{a}.compressor"
                yield f"{m}.compressor.norm.weight", get(f"{c}.norm.weight")
                if role.ratio > 1:
                    yield f"{m}.compressor.wkv", get(f"{c}.wkv.weight")
                    yield f"{m}.compressor.wgate", get(f"{c}.wgate.weight")
                else:
                    yield from linear(f"{c}.wkv", f"{m}.compressor.wkv")
                yield from linear(f"{a}.indexer.wk", f"{m}.indexer.wk")
                yield f"{m}.indexer.k_norm.weight", get(f"{a}.indexer.k_norm.weight")
            if role.has_indexer:
                yield from linear(f"{a}.indexer.wq_b", f"{m}.indexer.wq_b")
                yield from linear(f"{a}.indexer.weights_proj", f"{m}.indexer.weights_proj")

            p = f"layers.{L}"
            yield f"model.{p}.attn_norm.weight", get(f"{p}.attn_norm.weight")
            yield f"model.{p}.ffn_norm.weight", get(f"{p}.ffn_norm.weight")
            yield f"model.{p}.ffn.gate.weight", get(f"{p}.ffn.gate.weight")
            yield f"model.{p}.ffn.gate.bias", get(f"{p}.ffn.gate.bias")
            for proj in ("w1", "w2", "w3"):
                yield from linear(f"{p}.ffn.shared_experts.{proj}", f"model.{p}.ffn.shared_experts.{proj}")
            for nm in ("hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base", "hc_attn_scale", "hc_ffn_scale"):
                yield f"model.{p}.{nm}", get(f"{p}.{nm}")
            if L in args.engram_layer_ids:
                e = f"{p}.engram"
                yield from linear(f"{e}.wkv", f"model.{e}.wkv")
                yield f"model.{e}.q_weight", get(f"{e}.q_weight")
                yield f"model.{e}.k_weight", get(f"{e}.k_weight")
    finally:
        reader.close()


def iter_vision_weights(model_path: str, device):
    names = weight_map(model_path)
    reader = _ShardReader(model_path, names, device)
    try:
        for name in names:
            if name.startswith(("vision.", "aligner.")) or name in ("image_start", "image_end", "image_newline"):
                yield f"visual.{name}", reader.get(name)
        args = load_args(model_path)
        yield "visual.routing_bias", torch.stack([reader.get(f"layers.{i}.ffn.gate.bias_vl") for i in range(args.n_layers)])
    finally:
        reader.close()


# ----- FTW side files: the Engram tables ----------------------------------------------------------
def engram_table_names(layer_id: int) -> tuple[str, str]:
    return f"layers.{layer_id}.engram.embed.weight", f"layers.{layer_id}.engram.embed.scale"


def ftw_side_files(model_path: str, out_dir: str) -> list[str]:
    """Copy the checkpoint shards holding the Engram tables (fp8 rows + ue8m0 scales) next to an FTW
    checkpoint. The tables are never FTW entries: ``load_host_tables`` maps them in place from whatever
    safetensors file holds them (``engram_table.engram_row_source`` reads the headers), and the FTW
    loader never reads ``.safetensors`` as weights, so the shard's other tensors ride along unused."""
    import shutil

    from freetoken.models.loader import safetensors_weight_map
    from freetoken.utils import download_hf_weight

    folder = download_hf_weight(model_path)
    args = load_args(folder)
    shard_of = safetensors_weight_map(folder)
    shards = sorted({shard_of[n] for layer_id in args.engram_layer_ids if layer_id < args.n_layers for n in engram_table_names(layer_id)})
    for shard in shards:
        shutil.copyfile(os.path.join(folder, shard), os.path.join(out_dir, shard))
    return shards


# ----- routed MXFP4 expert pieces ---------------------------------------------------------------
_EXPERT_RE = re.compile(r"^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\.(?P<proj>w1|w2|w3)\.(?P<kind>weight|scale)$")
_PROJ_ROLE = {"w1": "gate", "w3": "up", "w2": "down"}
_KIND_SUFFIX = {"weight": "", "scale": "_scale"}


def iter_expert_pieces(model_path: str, config, kind: QuantKind, *, parallel: bool | None = False, workers: int = 8, chunk: int = 8 << 20):
    """One piece per expert: ``{gate, up, down}`` e2m1 pairs and their e8m0 ``_scale`` companions.
    The DSpark stage's experts (``mtp.*``) never match."""
    if kind is not QuantKind.MXFP4:
        return None
    if get_tp_info().size > 1:
        raise NotImplementedError("DeepSeek-V4.1 expert banks support TP=1 only")
    from freetoken.models.weight import iter_expert_tensors_parallel
    from freetoken.moe.expert_pieces import per_expert_pieces

    args = load_args(model_path)
    L, E = args.n_layers, args.n_routed_experts

    def locate(raw_name: str):
        m = _EXPERT_RE.match(raw_name)
        if m is None or int(m["layer"]) >= L:
            return None
        return int(m["layer"]), int(m["expert"]), _PROJ_ROLE[m["proj"]] + _KIND_SUFFIX[m["kind"]]

    if parallel:
        tensors = iter_expert_tensors_parallel(model_path, lambda n: locate(n) is not None, workers=workers, chunk=chunk)
        return per_expert_pieces(tensors, locate, tensors_per_expert=6)

    def _serial():
        reader = _ShardReader(model_path, weight_map(model_path), torch.device("cpu"))
        try:
            for li in tqdm(range(L), desc="Loading DeepSeek-V4.1 experts (serial)", disable=not get_tp_info().is_primary()):
                for e in range(E):
                    base = f"layers.{li}.ffn.experts.{e}"
                    for proj in ("w1", "w3", "w2"):
                        for kind_ in ("weight", "scale"):
                            name = f"{base}.{proj}.{kind_}"
                            yield name, reader.get(name)
        finally:
            reader.close()

    return per_expert_pieces(_serial(), locate, tensors_per_expert=6)


__all__ = ["iter_weights", "iter_vision_weights", "iter_expert_pieces", "ftw_side_files", "load_args", "weight_map", "dequant_fp8_block"]
