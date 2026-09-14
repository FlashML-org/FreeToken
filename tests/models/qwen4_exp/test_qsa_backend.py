"""The QSA backend behind the real Qwen4ExpAttention layer.

(a) dense-oracle equivalence -- while a request sees at most ``index_budget + index_ratio - 1``
    tokens every complete block is selected, so QSA IS dense attention: the selection must be
    exactly the causal prefix and the layer output must match ``TorchDenseQSAReference`` (fp32)
    and a flashinfer dense run over the same pool;
(b) chunked prefill at unaligned cut points equals one-shot prefill (the dual-source compress);
(c) a captured decode replay equals the eager decode step;
(d) an fp8 KV pool (``--kv-cache-dtype fp8``) keeps block selection bit-identical to the
    16-bit run -- only the selected K/V rows are read back as e4m3 codes -- and the layer
    output stays within quantization error of it.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from .common import Fixture, hf_config, requires_cuda, parsed_config, selection_spy

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


def _prefill_under_kv(monkeypatch, config, kv_quant: str, lengths):
    """One prefill of the QSA layer under a given KV store.

    Each call builds its own Fixture on purpose: a Fixture owns the global ctx (pool,
    page table, backend), so two KV stores cannot share one scenario. The weight seed
    (``Fixture.layer``) and the input seed (``_inputs``) are fixed, so the two runs differ
    ONLY in how the K/V rows are stored.
    """
    fixture = Fixture(config, num_pages=128, kv_quant=kv_quant)
    attn = fixture.layer(QSA_LAYER)
    seen = selection_spy(monkeypatch, fixture.backend)
    inputs = _inputs(fixture, lengths)
    x = torch.cat([row[:n] for row, n in zip(inputs, lengths)])
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]
    batch = fixture.batch(reqs, "prefill")
    out = attn.forward(x, batch)
    # the selection lives in a scratch buffer the next forward overwrites
    return fixture, out.clone(), seen["indices"].clone(), batch.positions.clone()


@requires_cuda
def test_fp8_kv_pool_keeps_selection_and_output(monkeypatch):
    """--kv-cache-dtype fp8 through the real layer: e4m3 codes + per-row scales in, same
    answer out to within quantization error -- and, because block selection scores 16-bit
    compressed index keys that fp8 never touches, the SAME selection bit for bit."""
    config = parsed_config()
    lengths = [2051, 1000, 137]  # every complete block is selected here

    plain, plain_out, plain_idx, _ = _prefill_under_kv(monkeypatch, config, "none", lengths)
    quant, quant_out, quant_idx, positions = _prefill_under_kv(
        monkeypatch, config, "fp8", lengths
    )

    # The tripwire for the field failure: the backend sizes its indexer scratch with
    # pool.dtype, which must stay the COMPUTE dtype even when store_dtype is e4m3. An
    # fp8 q_index compiles into qsa_mqa_paged's dot and dies at graph capture.
    assert quant.backend.dtype is torch.bfloat16
    assert plain.backend.dtype is torch.bfloat16
    assert quant.pool.store_dtype != torch.bfloat16
    assert quant.pool.kv_quant == "fp8" and plain.pool.kv_quant == "none"
    assert quant.pool.k_cache(QSA_LAYER).element_size() == 1
    assert quant.pool.v_cache(QSA_LAYER).element_size() == 1
    assert plain.pool.k_scale(QSA_LAYER) is None and plain.pool.v_scale(QSA_LAYER) is None
    pages, page_size, kv_heads = quant.pool.k_cache(QSA_LAYER).shape[:3]
    assert quant.pool.k_scale(QSA_LAYER).shape == (pages * page_size, kv_heads)
    assert quant.pool.k_scale(QSA_LAYER).dtype is torch.float32

    for pool in (plain.pool, quant.pool):
        assert pool.cmp_k_cache(0).dtype is torch.bfloat16
    assert torch.equal(quant_idx, plain_idx), (
        "quantizing the KV rows changed which blocks the indexer selected -- the index "
        "tier is supposed to be 16-bit in both runs"
    )
    _assert_selection_is_causal_prefix(quant_idx, positions)

    # Looser than the 2e-2 the 16-bit run needs against the same reference: e4m3 carries
    # four significant bits, so ~1e-2 relative per stored element is the floor here.
    torch.testing.assert_close(quant_out.float(), plain_out.float(), rtol=4e-2, atol=4e-2)


def _mrope_config(rope_type):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.models.qwen4_exp.config import parse_config

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    hf = hf_config(budget=32, max_position=512)
    hf.text_config.rope_parameters.update(mrope_section=[11, 11, 10], mrope_interleaved=True)
    if rope_type == "yarn":
        hf.text_config.rope_parameters.update(rope_type="yarn", factor=4.0,
                                             original_max_position_embeddings=262144)
    hf.vision_config = SimpleNamespace(
        hidden_size=32, depth=1, num_heads=4, intermediate_size=64, patch_size=2,
        temporal_patch_size=2, spatial_merge_size=2, num_position_embeddings=64,
        out_hidden_size=hf.text_config.hidden_size, in_channels=3,
    )
    config = parse_config(hf)
    assert config.model_is_mrope and config.rotary_config.mrope_layout == "interleaved"
    return config


def _image_positions(length, start, height, width, device):
    positions = torch.arange(length, dtype=torch.int32, device=device).repeat(3, 1)
    pixels = torch.arange(height * width, dtype=torch.int32, device=device)
    end = start + pixels.numel()
    positions[0, start:end] = start
    positions[1, start:end] = start + pixels // width
    positions[2, start:end] = start + pixels % width
    positions[:, end:] += max(height, width) - pixels.numel()
    return positions


def _index_mrope_reference(positions, rope_type):
    # Independently spell out Qwen's [11, 11, 10] interleaving and partial-rope frequencies.
    axes = torch.tensor([1 if i % 3 == 1 else 2 if i % 3 == 2 and i < 30 else 0
                         for i in range(32)], device=positions.device)
    inv = 1.0 / (1e7 ** (torch.arange(0, 64, 2, device=positions.device).float() / 64))
    amplitude = 1.0
    if rope_type == "yarn":
        low = math.floor(32 * math.log(262144 / (64 * math.pi)) / math.log(1e7))
        high = math.ceil(32 * math.log(262144 / (2 * math.pi)) / math.log(1e7))
        blend = ((torch.arange(32, device=positions.device).float() - low) / (high - low)).clamp(0, 1)
        inv = inv * (1 - 0.75 * blend)
        amplitude = 1 + 0.1 * math.log(4)
    phases = positions[axes].t().float() * inv
    return torch.cat((phases.cos(), phases.sin()), dim=-1) * amplitude


def _mrope_under_kv(monkeypatch, kv_quant, chunked, rope_type):
    fixture = Fixture(_mrope_config(rope_type), num_pages=16, max_running_req=2, kv_quant=kv_quant)
    attn = fixture.layer(QSA_LAYER)
    lengths, steps = [83, 59], 3
    inputs = _inputs(fixture, lengths, extra=steps)
    full_positions = [_image_positions(n + steps, start, h, w, fixture.device)
                      for n, start, h, w in ((83, 29, 3, 7), (59, 13, 4, 5))]
    cuts = [37, 19] if chunked else lengths
    reqs = [fixture.req(i, 0, cut) for i, cut in enumerate(cuts)]
    snapshots = []
    with monkeypatch.context() as patch:
        seen = selection_spy(patch, fixture.backend)

        def forward(phase):
            batch = fixture.batch(reqs, phase)
            batch.mrope_positions = torch.cat([p[:, r.cached_len:r.device_len]
                                                for p, r in zip(full_positions, reqs)], dim=1)
            batch.get_attn_positions = lambda: batch.mrope_positions
            x = torch.cat([row[r.cached_len:r.device_len] for row, r in zip(inputs, reqs)])
            out = attn.forward(x, batch)
            md = batch.attn_metadata
            group_positions = torch.cat([p[:, torch.arange(r.cached_len, r.device_len, device=fixture.device) // 4 * 4]
                                          for p, r in zip(full_positions, reqs)], dim=1)
            torch.testing.assert_close(md.q_rope_cache, _index_mrope_reference(batch.mrope_positions, rope_type),
                                       rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(md.k_rope_cache, _index_mrope_reference(group_positions, rope_type),
                                       rtol=1e-6, atol=1e-6)
            for r, positions in zip(reqs, full_positions):
                slots = fixture.page_table[r.table_idx, :r.device_len].long()
                assert torch.equal(fixture.pool.rope_positions[slots], positions[:, :r.device_len].t())
            snapshots.append(dict(
                out=out.float().cpu(), indices=seen["indices"].cpu(),
                # Non-closing groups collide in unread scratch rows; only the persistent slab is defined.
                compressed=fixture.pool.cmp_k_cache(0)[:fixture.pool.cmp_scratch_base].cpu().clone(),
                ring=fixture.pool.pending_ring(0).cpu().clone(),
                rope=fixture.pool.rope_positions.cpu().clone(),
            ))

        forward("prefill")
        if chunked:
            reqs = [fixture.req(i, cut, n) for i, (cut, n) in enumerate(zip(cuts, lengths))]
            forward("prefill")
        for _ in range(steps):
            for req in reqs:
                fixture.step(req)
            forward("decode")
    assert fixture.pool.kv_quant == kv_quant
    assert fixture.pool.k_cache(QSA_LAYER).dtype == (torch.uint8 if kv_quant == "nvfp4" else fixture.dtype)
    return snapshots


@requires_cuda
@pytest.mark.parametrize("rope_type", ["default", "yarn"])
@pytest.mark.parametrize("chunked", [False, True], ids=["prefill-decode", "image-cut-decode"])
def test_mrope_nvfp4_kv_keeps_index_and_rope_state(monkeypatch, chunked, rope_type):
    """Image cuts split ratio-4 groups; NVFP4 changes only K/V, including later decode writes."""
    plain = _mrope_under_kv(monkeypatch, "none", chunked, rope_type)
    quant = _mrope_under_kv(monkeypatch, "nvfp4", chunked, rope_type)
    assert len(plain) == len(quant)
    for step, (expected, actual) in enumerate(zip(plain, quant)):
        for key in ("indices", "compressed", "ring", "rope"):
            assert torch.equal(actual[key], expected[key]), f"{key} changed at step {step}"
        assert torch.isfinite(actual["out"]).all()
        relative_rmse = ((actual["out"] - expected["out"]).square().mean()
                         / expected["out"].square().mean()).sqrt()
        cosine = torch.nn.functional.cosine_similarity(actual["out"].flatten(), expected["out"].flatten(), dim=0)
        assert relative_rmse < 0.18, f"NVFP4 relative RMSE {relative_rmse.item():.4f} at step {step}"
        assert cosine > 0.98, f"NVFP4 cosine {cosine.item():.4f} at step {step}"

