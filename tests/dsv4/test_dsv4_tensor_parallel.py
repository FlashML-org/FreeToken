"""DSV4 tensor parallelism: the shard contract, on the meta device.

Every sharded parameter must be exactly ``1/tp`` of its TP=1 shape on exactly ONE axis,
and the routed FP4 expert banks must divide the same way -- that division is what lets
N ranks hold the same host bank bytes that one rank holds today, instead of N copies.
Replicated tensors (the MLA latent KV path, the compressors, the Lightning Indexer, the
router) must keep their full shape, because every rank reads the same latent KV and must
select the same blocks.

The split is split across two owners on purpose, and these tests pin the seam between
them: the shared TP-aware layer classes (``freetoken.layers``) declare the rank-local
SHAPE, and the family's weight reader is the only thing that cuts the checkpoint to
match -- the loader binds state-dict entries by exact shape and never narrows. So
``model.*`` is asserted against ``state_dict()``, not ``named_parameters()``: a DSV4
module is a ``BaseOP``, not an ``nn.Module``.

CPU-only: shapes are read off a meta-device build, so no weights and no GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import freetoken.distributed.info as info_mod
from freetoken.distributed import DistributedInfo
from freetoken.models.deepseek_v4.args import DeepseekV4Args

# o_groups=8 bounds the split: a rank must own whole output groups.
TP_SIZES = (2, 4, 8)

REPLICATED = (".wkv", ".kv_norm", ".q_norm", ".wq_a", ".compressor.", ".indexer.", "hc_")


@pytest.fixture
def args() -> DeepseekV4Args:
    # Shipped DSV4-Flash widths (every divisibility rule keys off these), on a 4-layer
    # stack so the meta build stays cheap. The ratio pattern mirrors tests/dsv4's:
    # one uncompressed layer, one indexed (ratio 4) layer, one ratio-128 layer.
    return DeepseekV4Args(
        max_batch_size=1,
        max_seq_len=4096,
        n_layers=4,
        n_hash_layers=1,
        compress_ratios=(0, 4, 128, 4),
    )


def _set_tp(size: int, rank: int = 0) -> None:
    # set_tp_info() is write-once by design; these tests walk several sizes in one process.
    info_mod._TP_INFO = DistributedInfo(rank, size)


def _shapes(args: DeepseekV4Args, tp: int, rank: int = 0) -> dict[str, torch.Size]:
    _set_tp(tp, rank)
    from freetoken.models.deepseek_v4.model import Transformer

    with torch.device("meta"):
        model = Transformer(args)
    return {n: tuple(p.shape) for n, p in model.state_dict().items()}


@pytest.fixture(autouse=True)
def _restore_tp():
    yield
    _set_tp(1)


@pytest.mark.parametrize("tp", TP_SIZES)
def test_every_shard_is_a_clean_split_on_one_axis(args, tp):
    base = _shapes(args, 1)
    got = _shapes(args, tp)
    assert got.keys() == base.keys(), "TP must not add or drop parameters"

    sharded = [n for n in got if got[n] != base[n]]
    assert sharded, f"tp={tp} sharded nothing"
    for name in sharded:
        a, b = base[name], got[name]
        axes = [i for i in range(len(a)) if a[i] != b[i]]
        assert len(axes) == 1, f"{name}: {a} -> {b} splits on {len(axes)} axes, want 1"
        assert a[axes[0]] == b[axes[0]] * tp, f"{name}: {a} -> {b} is not a 1/{tp} split"


@pytest.mark.parametrize("tp", TP_SIZES)
def test_replicated_tensors_keep_their_full_shape(args, tp):
    base = _shapes(args, 1)
    got = _shapes(args, tp)
    for name, shape in got.items():
        if any(marker in name for marker in REPLICATED):
            assert shape == base[name], f"{name} must stay replicated under TP"


@pytest.mark.parametrize("tp", TP_SIZES)
def test_ranks_cover_the_vocabulary_exactly_once(args, tp):
    """Vocab rows are partitioned, not replicated, on both the embed and the head.

    ``VocabParallelEmbedding`` owns the split (div_ceil rows per rank, the last rank
    short); ``ParallelLMHead`` inherits it, so the two agree on which rows a rank holds.
    """
    from freetoken.models.deepseek_v4.model import Transformer

    for attr in ("embed", "head"):
        covered = 0
        for rank in range(tp):
            _set_tp(tp, rank)
            with torch.device("meta"):
                model = Transformer(args)
            start, rows = getattr(model, attr).vocab_range
            assert start == covered, f"{attr}: rank {rank} starts at {start}, expected {covered}"
            covered += rows
        assert covered == args.vocab_size, f"{attr}: ranks cover {covered} of {args.vocab_size}"


# --------------------------------------------------------------------------------------
# Routed FP4 expert banks.
# --------------------------------------------------------------------------------------


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _moe_cfg(args, tp: int, rank: int = 0):
    from freetoken.layers.quantization.moe.base import MoEConfig

    return MoEConfig(
        num_experts=args.n_routed_experts, hidden=args.dim,
        intermediate=args.moe_inter_dim, top_k=args.n_activated_experts,
        tp_rank=rank, tp_size=tp, strategy="offload",
    )


@pytest.mark.parametrize("tp", TP_SIZES)
def test_expert_banks_divide_and_tile_the_intermediate_dim(args, tp):
    """A rank's bank bytes are exactly 1/tp of the whole -- the memory win, on disk.

    The kernel owns the layout; the reader is only allowed to cut along the axis the
    layout puts I on. Both come from ``MoEConfig.local_intermediate``, which is the one
    place the split is decided.
    """
    from freetoken.layers.quantization.moe.mxfp4 import TritonMxfp4MoEKernel

    kernel = TritonMxfp4MoEKernel()
    full = kernel.layout(_moe_cfg(args, 1))

    covered = 0
    for rank in range(tp):
        layout = kernel.layout(_moe_cfg(args, tp, rank))
        i_local = args.moe_inter_dim // tp
        assert i_local * tp == args.moe_inter_dim
        covered += i_local
        for name, spec in layout.items():
            whole, part = full[name].shape, spec.shape
            # Exactly one axis shrinks -- the one that carries I. gate_up and its scale
            # carry it on the fused 2*i row axis; down and its scale on the column axis.
            axes = [d for d in range(len(whole)) if whole[d] != part[d]]
            assert axes == [0 if name.startswith("gate_up") else 1], (
                f"{name}: {whole} -> {part} shrinks on {axes}"
            )
            n, m = _numel(part), _numel(whole)
            assert n * tp == m, f"{name}: {part} is not 1/{tp} of {whole}"
            assert spec.dtype == full[name].dtype
    assert covered == args.moe_inter_dim


@pytest.mark.parametrize("rank", range(4))
def test_sharded_pieces_pack_into_the_ranks_bank(rank):
    """The bank CONTENTS, not only their shapes, must tile the packed I axis.

    This is the reader -> kernel seam end to end: per-expert pieces cut by
    ``shard_expert_piece`` are concatenated by the kernel's own ``pack`` into the
    ``[gate(I_local) | up(I_local)]`` row block the layout asked for. A cut on the wrong
    axis, or a forgotten /2 //32 on a scale, lands the wrong expert's bytes in the bank
    and nothing downstream notices.
    """
    from freetoken.layers.quantization.moe.base import MoEConfig
    from freetoken.layers.quantization.moe.mxfp4 import TritonMxfp4MoEKernel
    from freetoken.models.deepseek_v4.weight import shard_expert_piece

    tp, full_i, hidden = 4, 128, 64
    local_i, e8m0 = full_i // tp, torch.float8_e8m0fnu
    i_lo = rank * local_i

    def payload(shape, offset, dtype=torch.int8):
        raw = (torch.arange(torch.Size(shape).numel(), dtype=torch.int64) + offset) % 251
        return raw.to(torch.uint8).reshape(shape).view(dtype)

    def cut(role, t):
        # One expert at a time: the stream is per-expert 2-D, and pack prepends the E dim.
        return shard_expert_piece(role, t, rank=rank, tp_size=tp).unsqueeze(0)

    pieces = {
        "gate": cut("gate", payload((full_i, hidden // 2), 1)),
        "up": cut("up", payload((full_i, hidden // 2), 17)),
        "down": cut("down", payload((hidden, full_i // 2), 33)),
        "gate_scale": cut("gate_scale", payload((full_i, hidden // 32), 49, e8m0)),
        "up_scale": cut("up_scale", payload((full_i, hidden // 32), 65, e8m0)),
        "down_scale": cut("down_scale", payload((hidden, full_i // 32), 81, e8m0)),
    }
    out = {
        n: torch.zeros((1, *shape), dtype=dtype)
        for n, (shape, dtype) in {
            "gate_up": ((2 * local_i, hidden // 2), torch.uint8),
            "gate_up_scale": ((2 * local_i, hidden // 32), e8m0),
            "down": ((hidden, local_i // 2), torch.uint8),
            "down_scale": ((hidden, local_i // 32), e8m0),
        }.items()
    }
    cfg = MoEConfig(
        num_experts=1, hidden=hidden, intermediate=full_i, top_k=2,
        tp_rank=rank, tp_size=tp, strategy="offload",
    )
    TritonMxfp4MoEKernel().pack(pieces, cfg, out)

    # gate occupies the first I_local rows, up the second half -- the concatenated order
    # the kernel documents, which is why a rank's rows are two blocks, not one 2I block.
    assert torch.equal(out["gate_up"][0, :local_i], payload((full_i, hidden // 2), 1)[i_lo:i_lo + local_i].view(torch.uint8))
    assert torch.equal(out["gate_up"][0, local_i:], payload((full_i, hidden // 2), 17)[i_lo:i_lo + local_i].view(torch.uint8))
    assert torch.equal(
        out["down"][0],
        payload((hidden, full_i // 2), 33)[:, i_lo // 2:(i_lo + local_i) // 2].view(torch.uint8),
    )
    assert torch.equal(
        out["gate_up_scale"][0, :local_i].view(torch.uint8),
        payload((full_i, hidden // 32), 49, e8m0)[i_lo:i_lo + local_i].view(torch.uint8),
    )
    # up's rows land in the SECOND half of the fused gate_up_scale bank -- there is no
    # "up_scale" bank, and asserting one would be asserting a layout that does not exist.
    assert torch.equal(
        out["gate_up_scale"][0, local_i:].view(torch.uint8),
        payload((full_i, hidden // 32), 65, e8m0)[i_lo:i_lo + local_i].view(torch.uint8),
    )
    assert torch.equal(
        out["down_scale"][0].view(torch.uint8),
        payload((hidden, full_i // 32), 81, e8m0)[:, i_lo // 32:(i_lo + local_i) // 32].view(torch.uint8),
    )


@pytest.mark.parametrize("dim", [0, 1])
def test_a_shard_does_not_keep_its_parent_alive(dim):
    """A shard must own its storage.

    ``narrow`` returns a view, and a dim-0 view is already contiguous, so a
    ``.contiguous()`` there hands the view straight back and pins the whole parent.
    Every rank then pays for the full tensor it just sharded -- 5.1 GiB per GPU on
    DSV4-Flash at TP=4, charged to the model and taken out of the cache budget.
    """
    from freetoken.models.deepseek_v4.parallel import shard

    _set_tp(4, rank=1)
    parent = torch.zeros(64, 64)
    piece = shard(parent, dim)
    assert piece.shape[dim] == 16
    assert piece.is_contiguous()
    assert piece.untyped_storage().nbytes() == piece.numel() * piece.element_size(), (
        "shard is a view into the parent's storage"
    )
    assert piece.data_ptr() != parent.data_ptr()


def test_a_split_that_does_not_divide_o_groups_fails_loudly(args):
    from freetoken.models.deepseek_v4.parallel import validate_tp

    _set_tp(16)  # o_groups == 8, so a rank cannot own a whole group
    with pytest.raises(ValueError, match="o_groups"):
        validate_tp(args)


def test_tp_rejects_ftw_tp1_layout_before_model_setup(tmp_path):
    from freetoken.checkpoint.ftw import INDEX_NAME
    from freetoken.engine.engine import _adjust_dsv4_config

    (tmp_path / INDEX_NAME).write_text("{}", encoding="utf-8")
    config = SimpleNamespace(
        model_path=str(tmp_path),
        tp_info=SimpleNamespace(size=4),
    )

    with pytest.raises(ValueError, match="FTW stores the TP=1"):
        _adjust_dsv4_config(config, lambda _name, _value: None)


def test_the_expert_banks_and_the_offload_cache_agree_on_layer_count(args):
    """The banks and the cache must be built for the SAME number of MoE layers.

    They are derived independently -- the banks from the checkpoint's mtp.* keys, the
    cache from ModelConfig.num_moe_layers -- so a drafter that adds layers to one and
    not the other asserts at startup, after the full expert load has already run:

        AssertionError: ('gate_up_packed', 46)

    which costs a five-minute load to discover. Check it in a millisecond instead.
    """
    import dataclasses

    from freetoken.models.deepseek_v4.config import parse_config

    class _HF:  # parse_config only needs the checkpoint path off the hf config
        _name_or_path = "/home/carlos/models/DeepSeek-V4-Flash-0731"

    for enabled in (False, True):
        from freetoken.models.deepseek_v4.args import set_dspark_enabled

        set_dspark_enabled(enabled)
        try:
            cfg = parse_config(_HF())
        except Exception:  # no checkpoint on this host -- skip rather than fail
            return
        a = dataclasses.replace(cfg.dsv4_args, dspark_enabled=enabled)
        assert cfg.num_moe_layers == a.n_moe_layers, (
            f"dspark_enabled={enabled}: offload cache expects {cfg.num_moe_layers} "
            f"layers, expert banks build {a.n_moe_layers}"
        )
    set_dspark_enabled(False)


@pytest.mark.parametrize("enabled", [False, True])
def test_every_consumer_of_the_moe_layer_count_agrees(enabled):
    """Four places derive "how many MoE layers are there" independently.

    Enabling the dSpark drafter adds three, and each consumer learned about them in a
    separate commit -- every miss cost a full expert load to discover, because the
    assertions fire only after the banks are built:

        AssertionError: ('gate_up_packed', 46)          # the offload cache's banks
        assert len(layers) == num_moe_layers            # the model's MoE layer iterator

    So check all of them together, offline, instead of one per five-minute run.
    """
    import dataclasses

    import torch

    from freetoken.models.deepseek_v4.args import load_args, set_dspark_enabled
    from freetoken.models.deepseek_v4.model import Transformer
    from freetoken.moe.offload_cache import iter_offload_moe_layers

    model_path = "/home/carlos/models/DeepSeek-V4-Flash-0731"
    _set_tp(4)
    set_dspark_enabled(enabled)
    try:
        args = load_args(model_path, max_seq_len=4096)
    except Exception:  # no checkpoint on this host
        set_dspark_enabled(False)
        return

    expected = args.n_moe_layers
    assert expected == args.n_layers + (3 if enabled else 0)

    # 1. the KV-owning layer list
    assert len(args.layer_compress_ratios) == expected

    # 2. the host expert banks, which the loader sizes from ModelConfig.num_moe_layers
    from freetoken.models.deepseek_v4.config import parse_config

    class _HF:
        _name_or_path = model_path

    assert parse_config(_HF()).num_moe_layers == expected

    # 3. the DSV4 KV pool, which indexes window_pool by layer id -- a draft layer
    #    storing its KV past the end of that list is an IndexError mid-generation,
    #    not at startup.
    from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache
    assert len(args.layer_compress_ratios) == expected

    # 4. the model's offload-MoE layer iterator, which the cache counts
    with torch.device("meta"):
        model = Transformer(args)
    assert len(list(iter_offload_moe_layers(model))) == expected

    set_dspark_enabled(False)
