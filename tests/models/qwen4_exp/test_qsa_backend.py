"""The QSA backend behind the real Qwen4ExpAttention layer.

(a) dense-oracle equivalence -- while a request sees at most ``index_budget + index_ratio - 1``
    tokens every complete block is selected, so QSA IS dense attention: the selection must be
    exactly the causal prefix and the layer output must match ``TorchDenseQSAReference`` (fp32)
    and a flashinfer dense run over the same pool;
(b) chunked prefill at unaligned cut points equals one-shot prefill (the dual-source compress);
(c) a captured decode replay equals the eager decode step.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from .common import Fixture, requires_cuda, parsed_config, selection_spy

QSA_LAYER = 3


def _inputs(fixture: Fixture, lengths, extra: int = 0, seed: int = 11):
    generator = torch.Generator(device=fixture.device).manual_seed(seed)
    return [
        torch.randn(
            n + extra, fixture.config.hidden_size, device=fixture.device,
            dtype=fixture.dtype, generator=generator,
        )
        * 0.5
        for n in lengths
    ]


def _assert_selection_is_causal_prefix(indices: torch.Tensor, positions: torch.Tensor) -> None:
    for row, position in enumerate(positions.tolist()):
        selected = indices[row][indices[row] >= 0]
        assert torch.equal(
            selected.sort().values,
            torch.arange(position + 1, dtype=selected.dtype, device=selected.device),
        ), f"row {row} (position {position}) did not select its whole causal prefix"


@requires_cuda
def test_prefill_is_dense_below_the_budget(monkeypatch):
    """bs=3 ragged prefill, longest request exactly at budget + ratio - 1."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=128)
    attn = fixture.layer(QSA_LAYER)
    lengths = [2051, 1000, 137]
    inputs = _inputs(fixture, lengths)
    x = torch.cat([row[:n] for row, n in zip(inputs, lengths)])
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]

    seen = selection_spy(monkeypatch, fixture.backend)
    batch = fixture.batch(reqs, "prefill")
    got = attn.forward(x, batch)
    _assert_selection_is_causal_prefix(seen["indices"], batch.positions)

    fixture.ctx.attn_backend = _dense_oracle(fixture)
    reference = attn.forward(x, batch)
    torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)


def _dense_oracle(fixture: Fixture):
    from freetoken.models.qwen4_exp.attention import TorchDenseQSAReference

    return TorchDenseQSAReference(
        fixture.config,
        num_slots=fixture.num_req_slots,
        max_len=4096,
        device=fixture.device,
        dtype=fixture.dtype,
    )


@requires_cuda
def test_decode_is_dense_below_the_budget(monkeypatch):
    """Prefill then five decode steps, sparse path vs the fp32 dense oracle."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=128)
    attn = fixture.layer(QSA_LAYER)
    lengths, steps = [300, 411, 64], 5
    inputs = _inputs(fixture, lengths, extra=steps)
    oracle = _dense_oracle(fixture)

    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]
    seen = selection_spy(monkeypatch, fixture.backend)

    steps_x = [torch.cat([row[:n] for row, n in zip(inputs, lengths)])]
    steps_x += [
        torch.stack([row[n + step] for row, n in zip(inputs, lengths)]) for step in range(steps)
    ]
    for step, x in enumerate(steps_x):
        if step:
            for req in reqs:
                fixture.step(req)
        batch = fixture.batch(reqs, "prefill" if step == 0 else "decode")
        fixture.ctx.attn_backend = fixture.backend
        got = attn.forward(x, batch)
        _assert_selection_is_causal_prefix(seen["indices"], batch.positions)
        fixture.ctx.attn_backend = oracle
        reference = attn.forward(x, batch)
        torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)


@requires_cuda
def test_flashinfer_dense_matches_the_sparse_path():
    """The engine's dense FULL backend over the same pool, as an independent oracle."""
    pytest.importorskip("flashinfer")
    from freetoken.attention.fi import FlashInferBackend

    config = parsed_config()
    fixture = Fixture(config, num_pages=64)
    attn = fixture.layer(QSA_LAYER)
    length = 500
    x = _inputs(fixture, [length])[0]
    req = fixture.req(0, 0, length)
    got = attn.forward(x, fixture.batch([req], "prefill"))

    dense = FlashInferBackend(config)
    fixture.ctx.attn_backend = SimpleNamespace(
        qsa_forward=lambda q, k, v, index, layer_id, batch: dense.forward(
            q, k, v, layer_id, batch
        )
    )
    batch = fixture.batch([req], "prefill")
    dense.prepare_metadata(batch)
    reference = attn.forward(x, batch)
    torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)


@requires_cuda
@pytest.mark.parametrize("cut", [1001, 4096, 4097], ids=["unaligned", "page-boundary", "boundary+1"])
def test_chunked_prefill_matches_one_shot(cut: int):
    """Cut points that are not multiples of index_ratio exercise the dual-source compress."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=512)
    attn = fixture.layer(QSA_LAYER)
    length = 5000
    x = _inputs(fixture, [length])[0]

    one_shot = attn.forward(x, fixture.batch([fixture.req(0, 0, length)], "prefill"))
    head = fixture.req(1, 0, cut)
    attn.forward(x[:cut], fixture.batch([head], "prefill"))
    tail = fixture.req(1, cut, length)
    got = attn.forward(x[cut:], fixture.batch([tail], "prefill"))
    assert torch.equal(got, one_shot[cut:])


@requires_cuda
def test_decode_graph_replay_matches_eager():
    config = parsed_config()
    fixture = Fixture(config, num_pages=256)
    attn = fixture.layer(QSA_LAYER)
    lengths, steps = [300, 411], 4
    bs = len(lengths)
    inputs = _inputs(fixture, lengths, extra=steps)
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]
    attn.forward(
        torch.cat([row[:n] for row, n in zip(inputs, lengths)]),
        fixture.batch(reqs, "prefill"),
    )

    fixture.backend.init_capture_graph(max_seq_len=fixture.page_table.shape[1], bs_list=[bs])
    dummy = SimpleNamespace(
        table_idx=fixture.num_req_slots - 1, cached_len=1, device_len=2, extend_len=1
    )
    static = {
        "x": torch.zeros(bs, config.hidden_size, device=fixture.device, dtype=fixture.dtype),
        "positions": torch.zeros(bs, dtype=torch.int32, device=fixture.device),
        "out_loc": torch.zeros(bs, dtype=torch.int32, device=fixture.device),
    }
    capture_batch = SimpleNamespace(
        padded_reqs=[dummy] * bs, reqs=[dummy] * bs, phase="decode", size=bs, padded_size=bs,
        is_prefill=False, is_decode=True, positions=static["positions"],
        out_loc=static["out_loc"], attn_metadata=None, active_table_idx=None,
    )
    fixture.backend.prepare_for_capture(capture_batch)
    attn.forward(static["x"], capture_batch)  # warmup, same metadata object as the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = attn.forward(static["x"], capture_batch)
    torch.cuda.synchronize()

    for step in range(steps):
        for req in reqs:
            fixture.step(req)
        x = torch.stack([row[n + step] for row, n in zip(inputs, lengths)])
        batch = fixture.batch(reqs, "decode")
        static["x"].copy_(x)
        static["positions"].copy_(batch.positions)
        static["out_loc"].copy_(batch.out_loc)
        fixture.backend.prepare_for_replay(batch)
        # replay must stage into the captured buffers, never reallocate them
        md = batch.attn_metadata
        assert md.block_table.data_ptr() == fixture.backend._graph["block_table"].data_ptr()
        graph.replay()
        replayed = captured_out.clone()
        eager = attn.forward(x, fixture.batch(reqs, "decode"))
        assert torch.equal(replayed, eager), f"graph replay diverged at decode step {step}"


@requires_cuda
def test_row_chunked_scoring_matches_one_chunk(monkeypatch):
    """The scoring workspace bound splits long prefills into row chunks."""
    import freetoken.attention.qsa_sparse as qsa_sparse

    config = parsed_config()
    fixture = Fixture(config, num_pages=64)
    attn = fixture.layer(QSA_LAYER)
    length = 600
    x = _inputs(fixture, [length])[0]
    whole = attn.forward(x, fixture.batch([fixture.req(0, 0, length)], "prefill"))

    columns = fixture.page_table.shape[1] // config.qwen4_args.index_ratio
    monkeypatch.setattr(qsa_sparse, "_LOGITS_WORKSPACE_BYTES", 64 * columns * 4)
    chunked = attn.forward(x, fixture.batch([fixture.req(1, 0, length)], "prefill"))
    assert torch.equal(chunked, whole)


@requires_cuda
def test_two_qsa_layers_keep_separate_slab_slots(monkeypatch):
    """Both QSA layers of one forward must hit their own slab slot and ring slice."""
    config = parsed_config(num_layers=8)
    assert config.attention_groups[1].layer_ids == (3, 7)
    fixture = Fixture(config, num_pages=64)
    layers = [fixture.layer(layer_id, seed=layer_id) for layer_id in (3, 7)]
    oracle = _dense_oracle(fixture)
    lengths, steps = [200, 71], 3
    inputs = _inputs(fixture, lengths, extra=steps)
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]

    xs = [torch.cat([row[:n] for row, n in zip(inputs, lengths)])]
    xs += [torch.stack([row[n + step] for row, n in zip(inputs, lengths)]) for step in range(steps)]
    for step, x in enumerate(xs):
        if step:
            for req in reqs:
                fixture.step(req)
        batch = fixture.batch(reqs, "prefill" if step == 0 else "decode")
        for attn in layers:
            fixture.ctx.attn_backend = fixture.backend
            got = attn.forward(x, batch)
            fixture.ctx.attn_backend = oracle
            reference = attn.forward(x, batch)
            torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)

    slab = fixture.pool.cmp_k_cache
    assert not torch.equal(slab(0), slab(1))


def _unfuse_qkv(fused: torch.Tensor, config) -> dict[str, torch.Tensor]:
    """Split the model's fused ``qkv_proj`` buffer back into the checkpoint's raw q/k/v keys.

    The fused order is ``[2*qo*head_dim | kv*head_dim | kv*head_dim]`` (q carries the gate),
    so the split is exact for both TP1 and a rank-local buffer.
    """
    qo = config.num_qo_heads * config.head_dim
    kv = config.num_kv_heads * config.head_dim
    q, k, v = fused.split([2 * qo, kv, kv], dim=0)
    return {
        f"layers.{QSA_LAYER}.self_attn.q_proj.weight": q,
        f"layers.{QSA_LAYER}.self_attn.k_proj.weight": k,
        f"layers.{QSA_LAYER}.self_attn.v_proj.weight": v,
    }


@requires_cuda
def test_tp2_rank_partials_sum_to_the_tp1_output(monkeypatch):
    """P2 numeric gate for QSA: two TP2 ranks' local layer outputs sum to the TP1 output,
    and both ranks must select the SAME blocks (the indexer and its compressed slab are
    replicated, so a split selection would corrupt the all-reduced result).

    Each rank runs the real backend over its own rank-local K/V slab; only the row-parallel
    ``o_proj`` all-reduce is replaced by an identity so the unreduced partials can be summed
    here, which is exactly the TP contract.
    """
    import freetoken.distributed.info as info
    from freetoken.distributed import DistributedCommunicator
    from freetoken.models.qwen4_exp.weight import shard_qwen4_exp_dense_tensor

    lengths = [2051, 1000, 137]  # dense regime: every complete block is selected
    req_meta = [(i, 0, n) for i, n in enumerate(lengths)]

    # ---- TP1 reference -------------------------------------------------------------
    monkeypatch.setattr(info, "_TP_INFO", info.DistributedInfo(rank=0, size=1))
    config1 = parsed_config()
    fix1 = Fixture(config1, num_pages=128)
    attn1 = fix1.layer(QSA_LAYER)
    x = torch.cat(_inputs(fix1, lengths))
    seen1 = selection_spy(monkeypatch, fix1.backend)
    batch1 = fix1.batch([fix1.req(*meta) for meta in req_meta], "prefill")
    want = attn1.forward(x, batch1).clone()
    want_idx = seen1["indices"].clone()
    full_state = {k: v.detach().clone() for k, v in attn1.state_dict().items()}
    assert config1.num_qo_heads == 4 and config1.num_kv_heads == 2  # global stays global

    # ---- TP2: two rank-local runs ---------------------------------------------------
    monkeypatch.setattr(DistributedCommunicator, "all_reduce", lambda self, x: x)
    partials, selections, slab_kv_heads = [], [], []
    for rank in range(2):
        monkeypatch.setattr(info, "_TP_INFO", info.DistributedInfo(rank=rank, size=2))
        config2 = parsed_config()  # geometry resolves rank-local via get_tp_info()
        fix2 = Fixture(config2, num_pages=128)
        attn2 = fix2.layer(QSA_LAYER)
        assert attn2.num_q == 2 and attn2.num_kv == 1, "rank-local heads not applied"

        # Rank-local weights, built the way the real loader does it: shard the RAW q/k/v
        # separately (so head groups stay intact), then re-fuse them into `qkv_proj`.
        q, k, v = _unfuse_qkv(full_state["qkv_proj.weight"], config1).values()
        parts = [
            shard_qwen4_exp_dense_tensor(
                name, tensor, config=config1, rank=rank, world_size=2
            )
            for name, tensor in (
                (f"layers.{QSA_LAYER}.self_attn.q_proj.weight", q),
                (f"layers.{QSA_LAYER}.self_attn.k_proj.weight", k),
                (f"layers.{QSA_LAYER}.self_attn.v_proj.weight", v),
            )
        ]
        raw = {
            "qkv_proj.weight": torch.cat(parts, dim=0),
            "o_proj.weight": shard_qwen4_exp_dense_tensor(
                f"layers.{QSA_LAYER}.self_attn.o_proj.weight", full_state["o_proj.weight"],
                config=config1, rank=rank, world_size=2,
            ),
            "q_norm.weight": full_state["q_norm.weight"].clone(),
            "k_norm.weight": full_state["k_norm.weight"].clone(),
        }
        # The indexer is replicated: copy it verbatim (no sharding) and prove it is identical.
        for leaf in (
            "indexer.index_qk_proj.weight", "indexer.q_layernorm.weight",
            "indexer.k_layernorm.weight",
        ):
            raw[leaf] = full_state[leaf].clone()
            assert torch.equal(raw[leaf], full_state[leaf])
        attn2.load_state_dict(raw)

        seen2 = selection_spy(monkeypatch, fix2.backend)
        batch2 = fix2.batch([fix2.req(*meta) for meta in req_meta], "prefill")
        partials.append(attn2.forward(x, batch2).clone())
        selections.append(seen2["indices"].clone())
        slab_kv_heads.append(fix2.pool.k_cache(QSA_LAYER).shape[2])

    # (a) replicated indexer => both ranks select exactly the same tokens as TP1
    assert torch.equal(selections[0], selections[1]), "TP2 ranks selected different blocks"
    assert torch.equal(selections[0], want_idx), "TP2 selection diverged from TP1"
    # (b) the K/V slab really is rank-local (TP1 has 2 kv heads, each rank has 1)
    assert slab_kv_heads == [1, 1]
    # (c) numeric gate: unreduced row-parallel partials sum to the TP1 output
    merged = partials[0] + partials[1]
    torch.testing.assert_close(merged.float(), want.float(), rtol=2e-2, atol=2e-2)

