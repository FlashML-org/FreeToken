"""Weight loading for DeepSeek-V4-Flash (engine path).

  - :func:`iter_weights` streams resident (non-expert) tensors keyed by the model's
    attribute paths (``model.`` + the checkpoint name). ``wo_a`` dequantized to bf16 to match
    the reference bf16 einsum.
  - :func:`iter_expert_pieces` streams the routed MXFP4 experts (e2m1 pairs + e8m0 per-32
    scales, no global) as per-expert pieces for the expert quant method's banks.
"""

from __future__ import annotations

import json
import os
import re

import safetensors
import torch
from tqdm import tqdm

from freetoken.layers.quantization import QuantKind
from freetoken.distributed import get_tp_info
from freetoken.models.loader import drop_page_cache

from .args import DeepseekV4Args, load_args
from .parallel import shard, shard_vocab


class _ShardReader:
    def __init__(self, folder: str, weight_map: dict, device):
        self._folder = folder
        self._weight_map = weight_map
        self._device = str(device)
        self._handles: dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self._weight_map

    def get(self, name: str) -> torch.Tensor:
        shard = self._weight_map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safetensors.safe_open(
                os.path.join(self._folder, shard), framework="pt", device=self._device
            ).__enter__()
            self._handles[shard] = handle
        return handle.get_tensor(name)

    def close(self) -> None:
        for shard, handle in self._handles.items():
            try:
                handle.__exit__(None, None, None)
            except Exception:
                pass
            drop_page_cache(os.path.join(self._folder, shard))
        self._handles.clear()


def _weight_map(model_path: str) -> dict:
    with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def _dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor, block: int = 128) -> torch.Tensor:
    """Dequantize 128x128 block-scaled FP8 (e4m3) to bf16.

    scale is e8m0 exponent codes, ``value = 2^(code-127)`` (Triton FP8 GEMM convention).
    Used for ``wo_a`` to match the reference's bf16 einsum.
    """
    n, k = weight.shape
    codes = scale.view(torch.uint8).to(torch.float32)
    s = torch.exp2(codes - 127.0)
    s = s.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)[:n, :k]
    return (weight.to(torch.float32) * s).to(torch.bfloat16)


def iter_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool = True,
    include_non_moe: bool = True,
):
    """Stream resident (non-expert) weights as ``(name, tensor)`` keyed to engine params.

    Routed MXFP4 experts come from the offload cache, so ``include_moe_experts`` must be
    False (DeepSeek-V4 only runs ``--moe-strategy offload``). Tensors yielded in checkpoint
    dtype (fp8 + e8m0 preserved); ``wo_a`` dequantized to bf16 to match the reference einsum.
    """
    if include_moe_experts:
        raise ValueError(
            "DeepSeek-V4 routed experts are served from the offload cache; "
            "run with --moe-strategy offload (include_moe_experts must be False)."
        )
    if not include_non_moe:
        return

    args = load_args(model_path, max_batch_size=1)
    reader = _ShardReader(model_path, _weight_map(model_path), device)

    def get(name: str) -> torch.Tensor:
        return reader.get(name)

    def linear(src: str, dst: str, split: int | None = None):
        """Yield one linear's tensors, cut for this rank.

        ``split=0`` is column-parallel (the output rows), ``split=1`` row-parallel (the
        input columns), ``None`` replicates. The TP-aware layer class declares the
        matching local shape, so the reader is the ONLY place the checkpoint is cut -- the
        layer is always handed the FULL logical size, or it would shard twice. The 128x128
        FP8 grid splits on the SAME axis as its weight, so the two stay aligned.
        """
        w = get(f"{src}.weight")
        yield f"{dst}.weight", w if split is None else shard(w, split)
        # fp8 linears declare the e8m0 block scale under the quant method's role name
        if reader.has(f"{src}.scale"):
            s = get(f"{src}.scale")
            yield f"{dst}.weight_scale_inv", s if split is None else shard(s, split)

    try:
        # Vocabulary-parallel embed + head: one contiguous block of rows per rank, cut the
        # way VocabParallelEmbedding sizes its own weight.
        yield "model.embed.weight", shard_vocab(get("embed.weight"))
        yield "model.norm.weight", get("norm.weight")
        yield "model.head.weight", shard_vocab(get("head.weight"))
        for nm in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            yield f"model.{nm}", get(nm)

        for L in range(args.n_layers):
            a = f"layers.{L}.attn"
            m = f"model.{a}"
            # wq_a / wkv stay replicated: MLA keeps ONE latent KV per token that every
            # head reads, so there is nothing to split on that path.
            yield from linear(f"{a}.wq_a", f"{m}.wq_a")
            yield f"{m}.q_norm.weight", get(f"{a}.q_norm.weight")
            yield from linear(f"{a}.wq_b", f"{m}.wq_b", split=0)  # column-parallel over heads
            yield from linear(f"{a}.wkv", f"{m}.wkv")
            yield f"{m}.kv_norm.weight", get(f"{a}.kv_norm.weight")
            # wo_a: FP8 in the checkpoint, dequantized to bf16 (reference bf16 einsum).
            # Its rows are o_groups blocks of o_lora_rank, so a dim-0 cut hands each rank
            # whole groups -- matching the heads its wq_b shard produced. The per-group
            # width is a property of one group and does NOT shard.
            yield f"{m}.wo_a", shard(_dequant_fp8_block(
                get(f"{a}.wo_a.weight"), get(f"{a}.wo_a.scale")
            ), 0)
            # Row-parallel over the input columns; wo_b all-reduces the partial sum.
            yield from linear(f"{a}.wo_b", f"{m}.wo_b", split=1)
            yield f"{m}.attn_sink", shard(get(f"{a}.attn_sink"), 0)

            ratio = args.compress_ratios[L]
            if ratio:
                c = f"{a}.compressor"
                for nm in ("ape", "wkv.weight", "wgate.weight", "norm.weight"):
                    yield f"model.{c}.{nm}", get(f"{c}.{nm}")
                if ratio == 4:
                    idx = f"{a}.indexer"
                    yield from linear(f"{idx}.wq_b", f"model.{idx}.wq_b")
                    yield f"model.{idx}.weights_proj.weight", get(f"{idx}.weights_proj.weight")
                    ic = f"{idx}.compressor"
                    for nm in ("ape", "wkv.weight", "wgate.weight", "norm.weight"):
                        yield f"model.{ic}.{nm}", get(f"{ic}.{nm}")

            yield f"model.layers.{L}.attn_norm.weight", get(f"layers.{L}.attn_norm.weight")
            yield f"model.layers.{L}.ffn_norm.weight", get(f"layers.{L}.ffn_norm.weight")

            g = f"layers.{L}.ffn.gate"
            yield f"model.{g}.weight", get(f"{g}.weight")
            if L < args.n_hash_layers:
                yield f"model.{g}.tid2eid", get(f"{g}.tid2eid")
            else:
                yield f"model.{g}.bias", get(f"{g}.bias")
            # Shared expert: the intermediate dim splits (w1/w3 column, w2 row) and w2
            # all-reduces, so the shared expert comes out complete on its own.
            for proj, split in (("w1", 0), ("w2", 1), ("w3", 0)):
                src = f"layers.{L}.ffn.shared_experts.{proj}"
                yield from linear(src, f"model.{src}", split=split)

            for nm in (
                "hc_attn_fn", "hc_ffn_fn", "hc_attn_base",
                "hc_ffn_base", "hc_attn_scale", "hc_ffn_scale",
            ):
                yield f"model.layers.{L}.{nm}", get(f"layers.{L}.{nm}")
    finally:
        reader.close()


# --------------------------------------------------------------------------------------
# Routed MXFP4 expert pieces.
# --------------------------------------------------------------------------------------
_EXPERT_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>w1|w2|w3)\.(?P<kind>weight|scale)$"
)
_PROJ_ROLE = {"w1": "gate", "w3": "up", "w2": "down"}
_KIND_SUFFIX = {"weight": "", "scale": "_scale"}
# Which axis of each per-expert piece carries the intermediate dim. A scale companion's
# axis is a fixed subdivision of I (the e8m0 grid, or 2 e2m1 codes per byte), so dividing
# THAT axis by the TP size is exactly the i_lo//2 and i_lo//32 arithmetic -- no per-role
# divisor to keep in sync.
_I_AXIS = {"gate": 0, "up": 0, "gate_scale": 0, "up_scale": 0, "down": 1, "down_scale": 1}


def shard_expert_piece(role: str, t: torch.Tensor, *, rank: int, tp_size: int) -> torch.Tensor:
    """This rank's block of the intermediate dim in one routed-expert piece.

    ``gate``/``up`` (checkpoint ``w1``/``w3``) and their e8m0 companions carry I on the
    ROW axis -- the scale grid blocks along H, so a scale splits on the same axis as its
    weight, undivided. ``down`` and its scale carry I on the COLUMN axis.

    The copy is the point: a ``narrow`` view keeps the whole parent tensor alive behind a
    1/N slice, so every rank would pay for the experts it just threw away -- measured at
    5.1 GiB per GPU on DeepSeek-V4-Flash at TP=4, silently charged to the model and out
    of the KV / expert-cache budget.
    """
    if tp_size == 1:
        return t
    axis = _I_AXIS[role]
    lo, step = _i_block(role, t.shape[axis], rank=rank, tp_size=tp_size)
    return t.narrow(axis, lo, step).clone(memory_format=torch.contiguous_format)


def _i_block(role: str, total: int, *, rank: int, tp_size: int) -> tuple[int, int]:
    """``(start, length)`` of this rank's block along a piece's I axis of ``total`` entries."""
    if total % tp_size != 0:
        raise ValueError(
            f"DeepSeek-V4 expert piece {role!r} has {total} entries on axis {_I_AXIS[role]}, "
            f"which does not divide over {tp_size} ranks"
        )
    step = total // tp_size
    return rank * step, step


def iter_expert_pieces(
    model_path: str, config, kind: QuantKind, *, parallel: bool | None = False, workers: int = 8, chunk: int = 8 << 20,
    prefetch: int = 2,
):
    """Routed experts, one piece per expert: ``{gate, up, down}`` e2m1 pairs and their e8m0
    ``_scale`` companions (``w1`` / ``w3`` / ``w2``). The MTP layer's experts are skipped."""
    if kind is not QuantKind.MXFP4:
        return None
    from freetoken.models.weight import iter_expert_tensors_parallel
    from freetoken.moe.expert_pieces import per_expert_pieces

    args = load_args(model_path, max_batch_size=1)
    L, E = args.n_layers, args.n_routed_experts

    def locate(raw_name: str):
        m = _EXPERT_RE.match(raw_name)
        if m is None or int(m["layer"]) >= L:
            return None
        return int(m["layer"]), int(m["expert"]), _PROJ_ROLE[m["proj"]] + _KIND_SUFFIX[m["kind"]]

    # TP: the expert kernel sizes its banks from MoEConfig.local_intermediate (= I // tp),
    # so a rank must yield only its own block of the intermediate dim. That is the memory
    # win that makes the model fit -- the host expert banks divide by the TP size.
    tp = get_tp_info()

    def _cut(stream):
        if tp.size == 1:
            return stream

        def sharded():
            for name, t in stream:
                loc = locate(name)
                if loc is None:
                    yield name, t
                    continue
                if t.ndim != 2:
                    # Never pass an expert piece through unsharded: the rank would build a
                    # full-width bank inside a layout sized for I/tp, and the loader's shape
                    # assert is the only thing that would notice -- long after the read.
                    raise ValueError(
                        f"DeepSeek-V4 expert piece {name!r} is {t.ndim}-D; the reader yields "
                        f"per-expert 2-D pieces and only their I axis is sharded"
                    )
                yield name, shard_expert_piece(loc[2], t, rank=tp.rank, tp_size=tp.size)

        return sharded()

    if parallel:
        tensors = iter_expert_tensors_parallel(
            model_path, lambda n: locate(n) is not None, workers=workers, chunk=chunk, prefetch=prefetch
        )
        return per_expert_pieces(_cut(tensors), locate, tensors_per_expert=6)

    def _serial():
        # One shard at a time, dropping its page cache before the next: the banks and a whole
        # checkpoint's cache do not both fit in host RAM when every TP rank reads at once.
        by_shard: dict[str, list[str]] = {}
        for name, shard_file in _weight_map(model_path).items():
            if locate(name) is not None:
                by_shard.setdefault(shard_file, []).append(name)
        for shard_file in by_shard:
            drop_page_cache(os.path.join(model_path, shard_file))
        for shard_file in tqdm(sorted(by_shard), desc="Loading DSV4 experts (serial)", disable=not tp.is_primary()):
            path = os.path.join(model_path, shard_file)
            try:
                with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                    for name in by_shard[shard_file]:
                        role = locate(name)[2]
                        if tp.size > 1 and _I_AXIS[role] == 0:
                            # gate/up carry I on the contiguous row axis, so read only this rank's rows
                            rows = f.get_slice(name)
                            lo, step = _i_block(role, rows.get_shape()[0], rank=tp.rank, tp_size=tp.size)
                            piece = rows[lo:lo + step].clone(memory_format=torch.contiguous_format)
                            del rows  # a live slice handle keeps the shard mapped, and mapped pages survive the drop
                        else:
                            piece = shard_expert_piece(role, f.get_tensor(name), rank=tp.rank, tp_size=tp.size)
                            if tp.size == 1:
                                piece = piece.clone()  # get_tensor maps the file; a held view would pin the shard's cache
                        yield name, piece
                        del piece
            finally:
                drop_page_cache(path)

    return per_expert_pieces(_serial(), locate, tensors_per_expert=6)


__all__ = ["iter_weights", "iter_expert_pieces"]
