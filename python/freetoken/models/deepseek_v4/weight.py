"""Weight loading for DeepSeek-V4-Flash (engine path).

  - :func:`iter_weights` streams resident (non-expert) tensors keyed to engine param
    names; model's ``load_state_dict`` casts each. ``wo_a`` dequantized to bf16 to match
    the reference bf16 einsum.
  - :func:`load_dsfp4_expert_sources` packs routed FP4 experts into pinned CPU banks for
    the offload cache. DeepSeek FP4: e8m0 per-32 block scale, no global scale.
"""

from __future__ import annotations

import collections
import json
import os
import re
from typing import Iterator

import safetensors
import torch
from tqdm import tqdm

from freetoken.models.loader import drop_page_cache

from .args import DeepseekV4Args, load_args
from .parallel import div_tp, shard, tp_info


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

    Routed FP4 experts come from the offload cache, so ``include_moe_experts`` must be
    False (DeepSeek-V4 only runs ``--moe-backend offload``). Tensors yielded in checkpoint
    dtype (fp8 + e8m0 preserved); ``wo_a`` dequantized to bf16 to match the reference einsum.
    """
    if include_moe_experts:
        raise ValueError(
            "DeepSeek-V4 routed experts are served from the offload cache; "
            "run with --moe-backend offload (include_moe_experts must be False)."
        )
    if not include_non_moe:
        return

    args = load_args(model_path, max_batch_size=1)
    reader = _ShardReader(model_path, _weight_map(model_path), device)

    def get(name: str) -> torch.Tensor:
        return reader.get(name)

    def linear(prefix: str, split: int | None = None):
        """Yield one linear's tensors, sharded for this rank.

        ``split=0`` is column-parallel (split the output rows), ``split=1`` is
        row-parallel (split the input columns), ``None`` replicates. The 128x128 FP8
        ``scale`` grid splits on the SAME axis as its weight, so the two stay aligned.
        """
        w = get(f"{prefix}.weight")
        yield f"{prefix}.weight", w if split is None else shard(w, split)
        if reader.has(f"{prefix}.scale"):
            s = get(f"{prefix}.scale")
            yield f"{prefix}.scale", s if split is None else shard(s, split)

    try:
        # Vocabulary-parallel: each rank keeps one contiguous block of rows.
        yield "embed.weight", shard(get("embed.weight"), 0)
        yield "norm.weight", get("norm.weight")
        yield "head", shard(get("head.weight"), 0)
        for nm in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            yield nm, get(nm)

        for L in range(args.n_layers):
            a = f"layers.{L}.attn"
            # wq_a / wkv / the norms stay replicated: MLA keeps ONE latent KV per token
            # that every head reads, so there is nothing to split on that path.
            yield from linear(f"{a}.wq_a")
            yield f"{a}.q_norm.weight", get(f"{a}.q_norm.weight")
            yield from linear(f"{a}.wq_b", split=0)  # column-parallel over heads
            yield from linear(f"{a}.wkv")
            yield f"{a}.kv_norm.weight", get(f"{a}.kv_norm.weight")
            # wo_a: FP8 in the checkpoint, dequantized to bf16 (reference bf16 einsum).
            # Rows are o_groups blocks of o_lora_rank, so a dim-0 split hands each rank
            # whole groups -- matching the heads its wq_b shard produced.
            yield f"{a}.wo_a", shard(_dequant_fp8_block(
                get(f"{a}.wo_a.weight"), get(f"{a}.wo_a.scale")
            ), 0)
            yield from linear(f"{a}.wo_b", split=1)  # row-parallel; all-reduced in _wo
            yield f"{a}.attn_sink", shard(get(f"{a}.attn_sink"), 0)

            ratio = args.compress_ratios[L]
            if ratio:
                c = f"{a}.compressor"
                yield f"{c}.ape", get(f"{c}.ape")
                yield f"{c}.wkv.weight", get(f"{c}.wkv.weight")
                yield f"{c}.wgate.weight", get(f"{c}.wgate.weight")
                yield f"{c}.norm.weight", get(f"{c}.norm.weight")
                if ratio == 4:
                    idx = f"{a}.indexer"
                    yield from linear(f"{idx}.wq_b")
                    yield f"{idx}.weights_proj.weight", get(f"{idx}.weights_proj.weight")
                    ic = f"{idx}.compressor"
                    yield f"{ic}.ape", get(f"{ic}.ape")
                    yield f"{ic}.wkv.weight", get(f"{ic}.wkv.weight")
                    yield f"{ic}.wgate.weight", get(f"{ic}.wgate.weight")
                    yield f"{ic}.norm.weight", get(f"{ic}.norm.weight")

            yield f"layers.{L}.attn_norm.weight", get(f"layers.{L}.attn_norm.weight")
            yield f"layers.{L}.ffn_norm.weight", get(f"layers.{L}.ffn_norm.weight")

            g = f"layers.{L}.ffn.gate"
            yield f"{g}.weight", get(f"{g}.weight")
            if L < args.n_hash_layers:
                yield f"{g}.tid2eid", get(f"{g}.tid2eid")
            else:
                yield f"{g}.bias", get(f"{g}.bias")
            # Shared expert: the intermediate dim splits (w1/w3 column, w2 row); the
            # partial sum is all-reduced together with the routed half in MoE.forward.
            for proj, split in (("w1", 0), ("w2", 1), ("w3", 0)):
                yield from linear(f"layers.{L}.ffn.shared_experts.{proj}", split=split)

            for nm in (
                "hc_attn_fn", "hc_ffn_fn", "hc_attn_base",
                "hc_ffn_base", "hc_attn_scale", "hc_ffn_scale",
            ):
                yield f"layers.{L}.{nm}", get(f"layers.{L}.{nm}")
    finally:
        reader.close()


# --------------------------------------------------------------------------------------
# Routed FP4 expert pinned banks.
# --------------------------------------------------------------------------------------
_EXPERT_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>w1|w2|w3)\.(?P<kind>weight|scale)$"
)

# The dSpark drafter's own routed experts. Its layers are appended after the target's,
# so ``mtp.k`` becomes bank layer ``n_layers + k`` and every layer-indexed structure
# (host banks, GPU slot cache) addresses draft and target layers identically.
_MTP_EXPERT_RE = re.compile(
    r"^mtp\.(?P<mtp>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>w1|w2|w3)\.(?P<kind>weight|scale)$"
)


def _expert_key(name: str, args: DeepseekV4Args):
    """``(bank_layer, expert, proj, kind)`` for a routed-expert tensor, else None.

    Returns None for anything this run does not serve: a target layer past ``n_layers``
    and, when dSpark is off, every ``mtp.*`` expert.
    """
    m = _EXPERT_RE.match(name)
    if m is not None:
        layer = int(m.group("layer"))
        if layer >= args.n_layers:  # a trailing MTP layer in the target namespace
            return None
        return layer, int(m.group("expert")), m.group("proj"), m.group("kind")
    m = _MTP_EXPERT_RE.match(name)
    if m is None:
        return None
    k = int(m.group("mtp"))
    if k >= args.n_draft_layers:  # dSpark off, or a layer beyond n_mtp_layers
        return None
    return args.n_layers + k, int(m.group("expert")), m.group("proj"), m.group("kind")


def load_dsfp4_expert_sources(
    model_path: str, args: DeepseekV4Args, *, layer_sink=None
) -> dict[str, list[torch.Tensor]]:
    """Build pinned CPU DeepSeek-FP4 banks for the routed experts.

    4 banks, one tensor per layer (independent allocations): ``gate_up_packed/scale``
    ``[E, 2I, H//2]`` uint8 / ``[..., H//32]`` e8m0, ``down_packed/scale`` ``[E, H, I//2]``
    / ``[..., I//32]``.

    ``layer_sink=None`` (serving): pin each layer as its writes complete, via an
    internally-owned :class:`PinPipeline`. ``layer_sink`` given (converter): the
    completion tracker fires into it instead -- nothing here is pinned, and the sink
    may release banks it has written out, so the returned tensors are only valid
    until then (the caller owns that tradeoff).
    """
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_layer_banks

    folder = model_path
    weight_map = _weight_map(folder)
    # L covers the target layers plus, when dSpark is enabled, the drafter's own.
    L, E = args.n_moe_layers, args.n_routed_experts

    for shard_file in sorted(set(weight_map.values())):
        drop_page_cache(os.path.join(folder, shard_file))

    shards: dict[str, list[tuple[str, tuple]]] = collections.defaultdict(list)
    for name, shard_file in weight_map.items():
        key = _expert_key(name, args)
        if key is not None:
            shards[shard_file].append((name, key))

    specs, I, i_lo = _expert_bank_specs(args)
    hb = alloc_layer_banks(specs, L)
    banks = {name: [b.tensor for b in hb[name]] for name in specs}

    # Under TP a rank keeps only rows [i_lo, i_lo+I) of w1/w3, and those rows are
    # CONTIGUOUS in the file -- so read just them instead of reading the whole tensor and
    # throwing (N-1)/N of it away. w2 needs a COLUMN slice, whose rows are strided, so it
    # is still read whole. w1+w3 are two thirds of the expert bytes, so at TP=4 this cuts
    # a rank's read (and its copy work) roughly in half.
    sliced_read = I != args.moe_inter_dim

    def _load(sink) -> int:
        tracker = LayerCompletionTracker(E * 6, hb, sink)  # {w1,w2,w3} x {weight,scale} x experts
        placed = 0
        for shard_file in tqdm(sorted(shards), desc="Loading DSV4 FP4 experts"):
            path = os.path.join(folder, shard_file)
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                for name, key in shards[shard_file]:
                    rows_ready = sliced_read and key[2] in ("w1", "w3")
                    t = f.get_slice(name)[i_lo:i_lo + I] if rows_ready else f.get_tensor(name)
                    layer = _place_dsfp4(banks, key, t, I, i_lo, rows_ready)
                    tracker.note(layer)
                    placed += 1
            drop_page_cache(path)
        return placed

    if layer_sink is not None:
        placed = _load(layer_sink)
    else:
        with PinPipeline() as pins:
            placed = _load(pins)

    expected = L * E * 6  # {w1,w2,w3} x {weight, scale}
    assert placed == expected, f"loaded {placed} expert tensors, expected {expected}"
    return banks


def dummy_dsfp4_expert_sources(args: DeepseekV4Args) -> dict[str, list[torch.Tensor]]:
    """Fabricate the 4 ds_fp4 banks for --dummy-weight (no checkpoint on disk)."""
    from freetoken.moe.host_banks import alloc_layer_banks, pin_banks

    L = args.n_moe_layers
    specs, _I, _i_lo = _expert_bank_specs(args)
    hb = alloc_layer_banks(specs, L)
    banks = {name: [b.tensor for b in hb[name]] for name in specs}
    for t in banks["gate_up_packed"]:  # packed e2m1; scales stay 0 (valid e8m0)
        t.random_(0, 256)
    for t in banks["down_packed"]:
        t.random_(0, 256)
    pin_banks(hb)
    return banks


def is_expert_tensor(name: str) -> bool:
    """Predicate for the common parallel reader: is this a routed-expert tensor?"""
    return _EXPERT_RE.match(name) is not None or _MTP_EXPERT_RE.match(name) is not None


def _expert_bank_specs(args: DeepseekV4Args) -> tuple[dict, int, int]:
    """The 4 routed-expert bank specs for THIS rank, plus its slice of ``moe_inter_dim``.

    Tensor parallelism splits the intermediate dim, so each rank allocates and reads
    only its own ``I // tp`` rows. This is what divides the 143 GB of host expert banks
    across the ranks instead of replicating them.
    """
    E, H = args.n_routed_experts, args.dim
    # FP4 packs 2 values per byte and carries one e8m0 scale per 32 values, so a rank's
    # slice must stay a multiple of 32 on the intermediate axis.
    i_local = div_tp(args.moe_inter_dim, "moe_inter_dim", multiple_of=32)
    i_lo = tp_info()[0] * i_local
    e8m0 = torch.float8_e8m0fnu
    specs = {  # alloc UNPINNED, fill, then pin-after-fill (skips slow cudaHostAlloc zero-fill)
        "gate_up_packed": ((E, 2 * i_local, H // 2), torch.uint8),
        "gate_up_scale": ((E, 2 * i_local, H // 32), e8m0),
        "down_packed": ((E, H, i_local // 2), torch.uint8),
        "down_scale": ((E, H, i_local // 32), e8m0),
    }
    return specs, i_local, i_lo


def _place_dsfp4(
    banks: dict, key: tuple, t: torch.Tensor, I: int, i_lo: int = 0,
    rows_ready: bool = False,
) -> int:
    """Copy one expert tensor into its layer/expert slot (shared by serial + parallel
    readers): w1->gate_up[:,:I], w3->gate_up[:,I:], w2->down. Returns the layer index.

    ``key`` is ``_expert_key``'s ``(bank_layer, expert, proj, kind)``, so a drafter
    expert lands in bank layer ``n_layers + k`` with no special case here.

    ``I`` is this rank's intermediate width and ``i_lo`` its offset into the checkpoint's
    full width, so each rank keeps only its own rows of w1/w3 and its own columns of w2.
    ``rows_ready`` says the caller already read JUST those rows (the serial reader slices
    w1/w3 at the file), so they must not be sliced a second time.
    """
    layer, expert, proj, kind = key

    def rows(x: torch.Tensor) -> torch.Tensor:
        return x if rows_ready else x[i_lo:i_lo + I]

    if kind == "weight":
        t = t.view(torch.uint8)
        if proj == "w1":
            banks["gate_up_packed"][layer][expert, :I] = rows(t)
        elif proj == "w3":
            banks["gate_up_packed"][layer][expert, I:] = rows(t)
        else:  # w2 -> down; the intermediate axis is packed 2-per-byte
            banks["down_packed"][layer][expert] = t[:, i_lo // 2:(i_lo + I) // 2]
    else:  # scale (e8m0), one per 32 values on the intermediate axis
        if proj == "w1":
            banks["gate_up_scale"][layer][expert, :I] = rows(t)
        elif proj == "w3":
            banks["gate_up_scale"][layer][expert, I:] = rows(t)
        else:
            banks["down_scale"][layer][expert] = t[:, i_lo // 32:(i_lo + I) // 32]
    return layer


def load_dsfp4_expert_sources_parallel(
    model_path: str, args: DeepseekV4Args, *, workers: int = 8, chunk: int = 8 << 20,
    layer_sink=None,
) -> dict[str, list[torch.Tensor]]:
    """parallel path: same banks as load_dsfp4_expert_sources, filled from the common
    chunked multi-threaded O_DIRECT reader instead of serial per-shard safe_open.
    ``layer_sink``: see :func:`load_dsfp4_expert_sources`."""
    from freetoken.models.weight import iter_expert_tensors_parallel
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_layer_banks

    # L covers the target layers plus, when dSpark is enabled, the drafter's own.
    L, E = args.n_moe_layers, args.n_routed_experts
    specs, I, i_lo = _expert_bank_specs(args)
    hb = alloc_layer_banks(specs, L)  # lazy host banks (unpinned)
    banks = {name: [b.tensor for b in hb[name]] for name in specs}

    def _is_expert(name: str) -> bool:
        return _expert_key(name, args) is not None

    def _load(sink) -> int:
        tracker = LayerCompletionTracker(E * 6, hb, sink)
        placed = 0
        for name, t in iter_expert_tensors_parallel(model_path, _is_expert, workers=workers, chunk=chunk):
            layer = _place_dsfp4(banks, _expert_key(name, args), t, I, i_lo)
            tracker.note(layer)
            placed += 1
        return placed

    if layer_sink is not None:
        placed = _load(layer_sink)
    else:
        with PinPipeline() as pins:
            placed = _load(pins)

    expected = L * E * 6
    assert placed == expected, f"loaded {placed} expert tensors, expected {expected}"
    return banks


__all__ = [
    "iter_weights",
    "load_dsfp4_expert_sources",
    "load_dsfp4_expert_sources_parallel",
    "is_expert_tensor",
]
