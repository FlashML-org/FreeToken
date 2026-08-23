"""DSV4 tensor parallelism: the shard contract, on the meta device.

Every sharded parameter must be exactly ``1/tp`` of its TP=1 shape on exactly ONE axis,
and the routed FP4 expert banks must divide the same way -- that division is what lets
N ranks hold the same host bank bytes that one rank holds today, instead of N copies.
Replicated tensors (the MLA latent KV path, the compressors, the Lightning Indexer, the
router) must keep their full shape, because every rank reads the same latent KV and must
select the same blocks.

CPU-only: shapes are read off a meta-device build, so no weights and no GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import freetoken.distributed.info as info_mod
from freetoken.distributed import DistributedInfo
from freetoken.models.deepseek_v4.args import DeepseekV4Args
from freetoken.models.deepseek_v4.weight import _expert_bank_specs, _place_dsfp4

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


def _shapes(args: DeepseekV4Args, tp: int, rank: int = 0) -> dict[str, tuple[int, ...]]:
    _set_tp(tp, rank)
    from freetoken.models.deepseek_v4.model import Transformer

    with torch.device("meta"):
        model = Transformer(args)
    return {n: tuple(p.shape) for n, p in model.named_parameters()}


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
    _set_tp(tp)
    from freetoken.models.deepseek_v4.model import Transformer

    covered = 0
    for rank in range(tp):
        _set_tp(tp, rank)
        with torch.device("meta"):
            model = Transformer(args)
        assert model.vocab_start == covered
        covered += model.vocab_local
    assert covered == args.vocab_size


@pytest.mark.parametrize("tp", TP_SIZES)
def test_expert_banks_divide_and_tile_the_intermediate_dim(args, tp):
    _set_tp(1)
    full, full_i, _ = _expert_bank_specs(args)

    covered = 0
    for rank in range(tp):
        _set_tp(tp, rank)
        specs, i_local, i_lo = _expert_bank_specs(args)
        assert i_lo == covered, "rank slices must tile the intermediate dim with no gap"
        covered += i_local
        assert i_local * tp == full_i
        # A rank's bank bytes are exactly 1/tp of the whole -- the memory win.
        for name, (shape, dtype) in specs.items():
            whole = torch.Size(full[name][0]).numel()
            part = torch.Size(shape).numel()
            assert part * tp == whole, f"{name}: {shape} is not 1/{tp} of {full[name][0]}"
            assert dtype == full[name][1]
    assert covered == full_i


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


@pytest.mark.parametrize("rank", range(4))
def test_expert_loader_places_exact_rank_slice(rank):
    """The bank contents, not only their shapes, must tile the packed I axis."""

    full_i, local_i, hidden = 128, 32, 64
    i_lo = rank * local_i
    e8m0 = torch.float8_e8m0fnu

    def payload(shape, offset, dtype=torch.int8):
        raw = (torch.arange(torch.Size(shape).numel(), dtype=torch.int64) + offset) % 251
        return raw.to(torch.uint8).reshape(shape).view(dtype)

    w1 = payload((full_i, hidden // 2), 1)
    w3 = payload((full_i, hidden // 2), 17)
    w2 = payload((hidden, full_i // 2), 33)
    s1 = payload((full_i, hidden // 32), 49, e8m0)
    s3 = payload((full_i, hidden // 32), 65, e8m0)
    s2 = payload((hidden, full_i // 32), 81, e8m0)
    banks = {
        "gate_up_packed": [torch.zeros(1, 2 * local_i, hidden // 2, dtype=torch.uint8)],
        "gate_up_scale": [torch.zeros(1, 2 * local_i, hidden // 32, dtype=e8m0)],
        "down_packed": [torch.zeros(1, hidden, local_i // 2, dtype=torch.uint8)],
        "down_scale": [torch.zeros(1, hidden, local_i // 32, dtype=e8m0)],
    }

    for proj, kind, tensor in (
        ("w1", "weight", w1),
        ("w3", "weight", w3),
        ("w2", "weight", w2),
        ("w1", "scale", s1),
        ("w3", "scale", s3),
        ("w2", "scale", s2),
    ):
        _place_dsfp4(banks, (0, 0, proj, kind), tensor, local_i, i_lo)

    assert torch.equal(
        banks["gate_up_packed"][0][0, :local_i],
        w1.view(torch.uint8)[i_lo:i_lo + local_i],
    )
    assert torch.equal(
        banks["gate_up_packed"][0][0, local_i:],
        w3.view(torch.uint8)[i_lo:i_lo + local_i],
    )
    assert torch.equal(
        banks["down_packed"][0][0],
        w2.view(torch.uint8)[:, i_lo // 2:(i_lo + local_i) // 2],
    )
    assert torch.equal(
        banks["gate_up_scale"][0][0, :local_i].view(torch.uint8),
        s1[i_lo:i_lo + local_i].view(torch.uint8),
    )
    assert torch.equal(
        banks["gate_up_scale"][0][0, local_i:].view(torch.uint8),
        s3[i_lo:i_lo + local_i].view(torch.uint8),
    )
    assert torch.equal(
        banks["down_scale"][0][0].view(torch.uint8),
        s2[:, i_lo // 32:(i_lo + local_i) // 32].view(torch.uint8),
    )


def test_serial_expert_loader_does_not_slice_pre_sliced_rows_twice():
    full_i, local_i, hidden, i_lo = 128, 32, 64, 64
    full = torch.arange(full_i * (hidden // 2), dtype=torch.int64)
    full = (full % 251).to(torch.uint8).reshape(full_i, hidden // 2).view(torch.int8)
    rank_rows = full[i_lo:i_lo + local_i]
    bank = torch.zeros(1, 2 * local_i, hidden // 2, dtype=torch.uint8)
    banks = {"gate_up_packed": [bank]}

    _place_dsfp4(
        banks,
        (0, 0, "w1", "weight"),
        rank_rows,
        local_i,
        i_lo,
        rows_ready=True,
    )

    assert torch.equal(bank[0, :local_i], rank_rows.view(torch.uint8))


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
    from freetoken.models.deepseek_v4.weight import _expert_bank_specs

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
        _specs, _i, _lo = _expert_bank_specs(a)
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
    from freetoken.models.deepseek_v4.model import DeepseekV4ForCausalLM
    from freetoken.models.deepseek_v4.weight import _expert_bank_specs

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

    # 2. the host expert banks (one entry per layer, per bank)
    specs, _i, _lo = _expert_bank_specs(args)
    assert all(len(s) == 3 for s, _d in specs.values()), "bank specs are per-layer shapes"

    # 3. the DSV4 KV pool, which indexes window_pool by layer id -- a draft layer
    #    storing its KV past the end of that list is an IndexError mid-generation,
    #    not at startup.
    from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache
    assert len(args.layer_compress_ratios) == expected

    # 4. the model's offload-MoE layer iterator, which the cache counts
    class _Cfg:
        dsv4_args = args
    with torch.device("meta"):
        model = DeepseekV4ForCausalLM(_Cfg())
    assert len(list(model._iter_offload_moe_layers())) == expected

    set_dspark_enabled(False)
