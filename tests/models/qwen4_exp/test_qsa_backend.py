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
        get_attn_positions=lambda: static["positions"],
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


# ------------------------------------------------------------------ K/V in host memory


def test_staging_windows_follow_each_request_in_logical_order():
    from freetoken.attention.qsa_sparse import plan_staging_windows

    # two requests: 3 cached pages at physical 7, 2, 9 and 2 cached pages at 4, 5
    block_table = torch.tensor([[7, 2, 9, 0], [4, 5, 0, 0]], dtype=torch.int32)
    whole = plan_staging_windows(block_table, [3, 2], capacity=8)
    assert len(whole) == 1
    assert whole[0].source_pages.tolist() == [7, 2, 9, 4, 5]
    assert whole[0].block_table.tolist() == [[0, 1, 2, -1], [3, 4, -1, -1]]

    split = plan_staging_windows(block_table, [3, 2], capacity=2)
    assert [w.source_pages.tolist() for w in split] == [[7, 2], [9, 4], [5]]
    assert [w.block_table.tolist() for w in split] == [
        [[0, 1, -1, -1], [-1, -1, -1, -1]],
        [[-1, -1, 0, -1], [1, -1, -1, -1]],
        [[-1, -1, -1, -1], [-1, 0, -1, -1]],
    ]
    # every cached page is staged exactly once across the windows
    staged = sorted(p for w in split for p in w.source_pages.tolist())
    assert staged == sorted([7, 2, 9, 4, 5])


def test_a_prefill_stages_only_when_its_rows_select_more_than_is_cached():
    from freetoken.attention.qsa_sparse import staging_moves_less

    select_width = 2051
    assert staging_moves_less(rows=8192, select_width=select_width, cached_tokens=1 << 20)
    assert not staging_moves_less(rows=20, select_width=select_width, cached_tokens=1 << 16)
    assert not staging_moves_less(rows=32, select_width=64, cached_tokens=32 * 64)
    assert staging_moves_less(rows=33, select_width=64, cached_tokens=32 * 64)


@requires_cuda
def test_a_prefill_folded_over_windows_allocates_nothing_one_window_does_not(monkeypatch):
    """The startup fit measures one staging window; a prefill folding several must not reach
    past that peak, or the first long prompt takes the memory the fit kept free."""
    import freetoken.kvcache.qsa_pool as qsa_pool
    from freetoken.attention.qsa_sparse import QSASparseAttnBackend

    original = QSASparseAttnBackend._attend
    peaks: dict[int, int] = {}

    def measured(self, q, layer_id, indices, md):
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = original(self, q, layer_id, indices, md)
        torch.cuda.synchronize()
        peaks[len(md.staging)] = torch.cuda.max_memory_allocated() - before
        return out

    monkeypatch.setattr(QSASparseAttnBackend, "_attend", measured)
    config = parsed_config()
    length = 5000
    for window_tokens in (1 << 17, 128):
        monkeypatch.setattr(qsa_pool, "STAGING_TOKENS", window_tokens)
        fixture = Fixture(config, num_pages=128, placement="host")
        attn = fixture.layer(QSA_LAYER)
        x = _inputs(fixture, [length])[0]
        attn.forward(x, fixture.batch([fixture.req(0, 0, length)], "prefill"))
        del fixture, attn, x
    assert sorted(peaks) == [1, 40]
    output_bytes = length * config.num_qo_heads * config.head_dim * 2
    assert peaks[1] >= output_bytes, "the measured step must include the attend's own output"
    assert peaks[40] <= peaks[1] + (1 << 20), (
        f"folding 40 windows peaked at {peaks[40] >> 20} MiB, one window at {peaks[1] >> 20} MiB"
    )


@requires_cuda
@pytest.mark.parametrize("window_tokens", [1 << 17, 128], ids=["one-window", "folded-windows"])
def test_host_placement_matches_device_placement(monkeypatch, window_tokens: int):
    """K/V in host memory: a prefill staged into device windows -- folded by log-sum-exp when the
    context spans several -- and a decode read in place both equal the device-placed pool."""
    import freetoken.kvcache.qsa_pool as qsa_pool

    monkeypatch.setattr(qsa_pool, "STAGING_TOKENS", window_tokens)
    config = parsed_config()
    length, steps = 5000, 3
    outputs = {}
    for placement in ("device", "host"):
        fixture = Fixture(config, num_pages=128, placement=placement)
        attn = fixture.layer(QSA_LAYER)
        x = _inputs(fixture, [length], extra=steps)[0]
        req = fixture.req(0, 0, length)
        prefill = fixture.batch([req], "prefill")
        got = [attn.forward(x[:length], prefill)]
        windows = prefill.attn_metadata.staging
        if placement == "host":
            pages, window_pages = -(-length // 64), window_tokens // 64
            assert windows is not None and len(windows) == -(-pages // window_pages)
        else:
            assert windows is None
        for step in range(steps):
            fixture.step(req)
            got.append(attn.forward(x[length + step : length + step + 1], fixture.batch([req], "decode")))
        outputs[placement] = got
    for step, (device_out, host_out) in enumerate(zip(outputs["device"], outputs["host"])):
        if window_tokens >= length or step > 0:
            assert torch.equal(host_out, device_out), f"output {step} differs"
        else:
            torch.testing.assert_close(host_out.float(), device_out.float(), rtol=1e-2, atol=1e-2)
