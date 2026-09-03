"""PLE layer acceptance: hash vs HF, pinned-host table vs the GPU oracle, conv state
advancement (prefill / chunked / stepwise decode), CUDA-graph decode, and prefetch overlap.

The HF ground truth comes from ``ple_hf_ref.py`` run under a transformers build that ships
qwen4_exp (``FREETOKEN_QWEN4_HF_PYTHON``); those tests skip when it is unset.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from freetoken.models.config import ModelConfig
from freetoken.models.qwen4_exp.config import parse_config
from freetoken.models.qwen4_exp.ple import (
    GpuResidentTable,
    PinnedUVATable,
    PLELayer,
    PLEMetadata,
    build_ple_metadata,
    commit_ngram_context,
    short_conv_reference,
)

from .common import EOS, VOCAB, hash_constants, requires_cuda, toy_hf_config

_HF_REF_PYTHON = os.environ.get("FREETOKEN_QWEN4_HF_PYTHON", "")
_HF_REF_SCRIPT = Path(__file__).with_name("ple_hf_ref.py")

requires_hf_ref = pytest.mark.skipif(
    not (_HF_REF_PYTHON and Path(_HF_REF_PYTHON).exists()),
    reason="set FREETOKEN_QWEN4_HF_PYTHON to a transformers build that ships qwen4_exp",
)


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------

def _config() -> ModelConfig:
    return parse_config(toy_hf_config())


def _padded_vocab(args) -> int:
    """HF pads the concatenated per-head vocabs up to make_ngram_vocab_size_divisible_by."""
    _, sizes, _ = hash_constants(args)
    div = args.make_ngram_vocab_size_divisible_by
    return -(-int(sizes.sum()) // div) * div


def _make_layer(config, *, device="cpu", dtype=torch.float32, rows=None, seed=3, table=None):
    from freetoken.utils.torch_utils import torch_dtype

    args = config.qwen4_args
    rows = _padded_vocab(args) if rows is None else rows
    device = torch.device(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    with torch.device(device), torch_dtype(dtype):
        layer = PLELayer(config, args.ple_layer_ids[0])
    for tensor in layer.state_dict().values():
        if tensor.is_floating_point():
            tensor.normal_(0.0, 0.05, generator=gen)
    multipliers, sizes, offsets = hash_constants(args)
    layer.ple_embedding.layer_multipliers.copy_(multipliers)
    layer.ple_embedding.ngram_heads_vocab_sizes.copy_(sizes)
    layer.ple_embedding.ngram_heads_offsets.copy_(offsets)
    if table is None:
        weight = torch.randn(rows, args.ngram_head_dim, generator=gen, device=device, dtype=dtype)
        table = GpuResidentTable(weight * 0.05, dtype=dtype)
    layer.ple_embedding.attach_table(table)
    return layer


def _meta(sequences, contexts, *, device="cpu", slots=None, fresh=None, decode=False):
    lens = [len(s) for s in sequences]
    to = lambda xs, dtype: torch.tensor(xs, dtype=dtype, device=device)
    cu = torch.tensor([0, *lens], dtype=torch.int64).cumsum(0).to(device)
    return PLEMetadata(
        input_ids=to([t for s in sequences for t in s], torch.int64),
        cu_seqlens=cu,
        seq_lens=tuple(lens),
        ngram_context=to(contexts, torch.int64),
        state_slots=(
            torch.arange(len(sequences), dtype=torch.int64, device=device)
            if slots is None
            else to(slots, torch.int64)
        ),
        fresh_slots=None if fresh is None else to(fresh, torch.bool),
        is_decode=decode,
    )


def _run_hf_reference(tmp_path, data: dict, layer_idx=2, ple_layer_index=0) -> dict:
    spec = {"config": vars(toy_hf_config().text_config), "layer_idx": layer_idx, "ple_layer_index": ple_layer_index}
    (tmp_path / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    np.savez(tmp_path / "in.npz", **data)
    subprocess.run(
        [_HF_REF_PYTHON, str(_HF_REF_SCRIPT), str(tmp_path / "spec.json"),
         str(tmp_path / "in.npz"), str(tmp_path / "out.npz")],
        check=True,
        capture_output=True,
    )
    return dict(np.load(tmp_path / "out.npz"))


# --------------------------------------------------------------------------------------
# hash
# --------------------------------------------------------------------------------------

# sequence start (all-eos context), eos inside the chunk, eos as the newest context token
_HASH_CASES = [
    ([EOS, EOS], [3, 4, EOS, 5, 6, 8]),
    ([21, 22], [2, EOS, 11, 12, 13, 14]),
    ([EOS, 31], [9, 10, 11, 12, 13, 14]),
    ([31, EOS], [9, 10, 11, 12, 13, 14]),
]


@requires_hf_ref
def test_hash_ids_match_hf(tmp_path):
    """row_ids equals HF Qwen4ExpTextNGramEmbedding per id, over eos boundaries and at sequence start."""
    config = _config()
    layer = _make_layer(config)
    # HF pads its own all-eos context, so feeding [context | tokens] reproduces a resumed request
    tokens = np.array([c + s for c, s in _HASH_CASES], dtype=np.int64)
    ref = _run_hf_reference(tmp_path, {"hash_tokens": tokens, **_layer_ref_inputs(config, layer)})
    hf_ids = torch.as_tensor(ref["hash_ids"])[:, len(_HASH_CASES[0][0]) :]

    contexts = [c for c, _ in _HASH_CASES]
    sequences = [s for _, s in _HASH_CASES]
    got = layer.ple_embedding.row_ids(_meta(sequences, contexts))
    offset = 0
    for i, seq in enumerate(sequences):
        assert torch.equal(got[offset : offset + len(seq)], hf_ids[i]), f"request {i}"
        offset += len(seq)


@requires_hf_ref
def test_hash_constants_match_hf(tmp_path):
    """derive_ngram_hash_constants reproduces the multipliers/vocab sizes/offsets HF builds at init."""
    config = _config()
    layer = _make_layer(config)
    ref = _run_hf_reference(
        tmp_path,
        {"hash_tokens": np.array([[EOS, EOS, 3, 4]], dtype=np.int64), **_layer_ref_inputs(config, layer)},
    )
    multipliers, sizes, offsets = hash_constants(config.qwen4_args)
    assert torch.equal(multipliers, torch.as_tensor(ref["layer_multipliers"]))
    assert torch.equal(sizes, torch.as_tensor(ref["ngram_heads_vocab_sizes"]))
    assert torch.equal(offsets, torch.as_tensor(ref["ngram_heads_offsets"]))


def test_decode_hash_matches_prefill_hash():
    """The decode window (context + one token) hashes to the same ids as the same token in a prefill."""
    config = _config()
    layer = _make_layer(config)
    sequences = [[3, 4, EOS, 5, 6, 8], [2, EOS, 11, 12, 13, 14]]
    contexts = [[EOS, EOS], [21, 22]]
    prefill = layer.ple_embedding.row_ids(_meta(sequences, contexts))
    for step in range(len(sequences[0])):
        window = [(contexts[i] + s)[step : step + 2] for i, s in enumerate(sequences)]
        got = layer.ple_embedding.row_ids(
            _meta([[s[step]] for s in sequences], window, decode=True)
        )
        for i in range(len(sequences)):
            assert torch.equal(got[i], prefill[i * len(sequences[0]) + step])


# --------------------------------------------------------------------------------------
# table backends
# --------------------------------------------------------------------------------------


def _pinned_bank(rows: int, dim: int, dtype: torch.dtype, seed: int = 11):
    from freetoken.moe.host_banks import HostBank

    gen = torch.Generator().manual_seed(seed)
    bank = HostBank((rows, dim), dtype)
    bank.tensor.copy_((torch.randn(rows, dim, generator=gen) * 0.4).to(dtype))
    bank.pin()
    return bank


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
def test_pinned_uva_matches_gpu_resident(dtype):
    """PinnedUVATable is bitwise equal to the GPU-resident oracle, through lookup and prefetch."""
    rows, dim, scale = 8192, 160, 0.0234375
    bank = _pinned_bank(rows, dim, dtype)
    oracle = GpuResidentTable(bank.tensor.cuda(), scale, dtype=torch.bfloat16)
    pinned = PinnedUVATable(bank.tensor, scale)

    ids = torch.randint(0, rows, (37, 16), device="cuda")
    want = oracle.lookup(ids)
    assert torch.equal(pinned.lookup(ids), want)

    pinned.prefetch(ids)
    assert torch.equal(pinned.lookup(ids), want)

    # a stale prefetch must still be joined before its staging buffer is reused
    pinned.prefetch(ids)
    other = torch.randint(0, rows, (37, 16), device="cuda")
    assert torch.equal(pinned.lookup(other), oracle.lookup(other))

    out = torch.empty(37, 16 * dim, dtype=torch.bfloat16, device="cuda")
    assert pinned.lookup(ids, out) is out
    assert torch.equal(out, want)


@requires_cuda
def test_pinned_uva_zeroes_out_of_range_ids():
    bank = _pinned_bank(64, 160, torch.float8_e4m3fn)
    pinned = PinnedUVATable(bank.tensor, 1.0)
    ids = torch.tensor([[0, 64, 1, -1]], device="cuda")
    rows = pinned.lookup(ids).view(4, 160)
    assert rows[1].abs().sum() == 0 and rows[3].abs().sum() == 0
    assert torch.equal(rows[0], bank.tensor[0].cuda().to(torch.bfloat16))


@pytest.mark.skipif(
    not os.environ.get("FREETOKEN_QWEN4EXP_MODEL"), reason="needs FREETOKEN_QWEN4EXP_MODEL"
)
@requires_cuda
def test_pinned_uva_real_table():
    """The real 47.7 GiB FP8 table: sampled rows equal the checkpoint bytes dequantized on CPU."""
    import safetensors
    from freetoken.models.qwen4_exp.weight import _PLE_SHARD_RE, _ple_table_files, load_ple_table

    path = os.environ["FREETOKEN_QWEN4EXP_MODEL"]
    with open(os.path.join(path, "config.json"), encoding="utf-8") as fh:
        text = json.load(fh)["text_config"]
    heads = (text["ngram_size"] - 1) * text["heads_per_ngram"]
    args = SimpleNamespace(
        split_ngram_parts=text["split_ngram_parts"],
        ngram_head_dim=text["ple_embed_dim"] // heads,
    )
    table = load_ple_table(path, args)
    scale = float(table.weight_scale)
    rows_per_shard = table.tensor.shape[0] // args.split_ngram_parts
    backend = PinnedUVATable(table.tensor, scale)

    gen = torch.Generator().manual_seed(5)
    sample = torch.randint(0, table.tensor.shape[0], (1000,), generator=gen)
    got = backend.lookup(sample.view(-1, 1).cuda()).cpu()

    shard_key = {}
    for file in _ple_table_files(path):
        with safetensors.safe_open(file, framework="pt", device="cpu") as fh:
            for key in fh.keys():
                match = _PLE_SHARD_RE.search(key)
                if match is not None:
                    shard_key[int(match.group("shard"))] = (file, key)

    by_file = {}
    for i, row in enumerate(sample.tolist()):
        file, key = shard_key[row // rows_per_shard]
        by_file.setdefault(file, []).append((i, key, row % rows_per_shard))
    for file, items in by_file.items():
        with safetensors.safe_open(file, framework="pt", device="cpu") as fh:
            for i, key, offset in items:
                raw = fh.get_slice(key)[offset : offset + 1]
                want = (raw.float() * scale).to(torch.bfloat16).reshape(-1)
                assert torch.equal(got[i], want), f"sample {i}"


# --------------------------------------------------------------------------------------
# conv state
# --------------------------------------------------------------------------------------


def _forward(layer, R, meta, states):
    return layer.forward(R, batch=None, meta=meta, conv_states=states)


def test_prefill_conv_matches_reference():
    """The packed single-conv prefill equals the per-request reference conv, chunks shorter than the state included."""
    torch.manual_seed(12)
    config = _config()
    args = config.qwen4_args
    layer = _make_layer(config)
    sequences = [[3, 4, EOS, 5, 6, 8, 9, 2, 4, 5, 6], [2, EOS, 11], [9]]
    contexts = [[EOS, EOS], [21, 22], [EOS, 31]]
    meta = _meta(sequences, contexts)
    total = sum(len(s) for s in sequences)
    x = torch.randn(total, args.ple_state_width)
    states = torch.randn(len(sequences), args.ple_state_width, args.ple_conv_state_len) * 0.1

    got_states = states.clone()
    got = layer._short_conv(x, meta, got_states)
    ref_states = states.clone()
    ref = short_conv_reference(x, meta, ref_states, layer.conv1d.weight, args.ple_conv_dilation)
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-6)
    assert torch.allclose(got_states, ref_states, rtol=1e-5, atol=1e-6)


def test_fresh_slots_read_a_zero_state():
    """A request marked fresh ignores whatever the pool slot still holds."""
    torch.manual_seed(13)
    config = _config()
    args = config.qwen4_args
    layer = _make_layer(config)
    meta = _meta([[3, 4, 5], [6, 7, 8]], [[EOS, EOS]] * 2, fresh=[True, False])
    x = torch.randn(6, args.ple_state_width)
    dirty = torch.randn(2, args.ple_state_width, args.ple_conv_state_len)
    clean = dirty.clone()
    clean[0] = 0
    got = layer._short_conv(x, meta, dirty.clone())
    want = layer._short_conv(x, _meta([[3, 4, 5], [6, 7, 8]], [[EOS, EOS]] * 2), clean.clone())
    assert torch.equal(got, want)


@pytest.mark.parametrize("cuts", [[1], [2, 3, 4], [9]], ids=["first-token", "uneven-mix", "penultimate"])
def test_chunked_prefill_matches_one_shot(cuts):
    """Chunked prefill at arbitrary cut points (including chunks shorter than the conv state) matches one shot."""
    torch.manual_seed(14)
    config = _config()
    args = config.qwen4_args
    layer = _make_layer(config)
    sequences = [[3, 4, EOS, 5, 6, 8, 9, 2, 4, 5, 6, 7], [2, EOS, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
    length = len(sequences[0])
    contexts = [[EOS, EOS], [21, 22]]
    x = torch.randn(len(sequences) * length, args.ple_state_width)
    per_req = [x[i * length : (i + 1) * length] for i in range(len(sequences))]
    zeros = torch.zeros(len(sequences), args.ple_state_width, args.ple_conv_state_len)

    full_states = zeros.clone()
    full = _forward(layer, x, _meta(sequences, contexts), full_states)

    chunk_states = zeros.clone()
    pieces, start = [], 0
    for size in [*cuts, length]:
        end = min(start + size, length)
        if end == start:
            continue
        window = [(c + s)[start : start + 2] for c, s in zip(contexts, sequences)]
        pieces.append(
            _forward(
                layer,
                torch.cat([r[start:end] for r in per_req]),
                _meta([s[start:end] for s in sequences], window),
                chunk_states,
            )
        )
        start = end

    for i, seq in enumerate(sequences):
        rebuilt = torch.cat(
            [p.chunk(len(sequences))[i] for p in pieces]
        )
        assert torch.allclose(rebuilt, full[i * length : (i + 1) * length], rtol=1e-4, atol=1e-5)
    assert torch.allclose(chunk_states, full_states, rtol=1e-4, atol=1e-5)


def _state_pool(config, num_slots=8):
    from freetoken.kvcache.linear_state_pool import LinearStatePool

    return LinearStatePool(
        config.linear_attention_group(), num_slots, torch.float32,
        torch.device("cpu"), tp_size=1, slot_states=config.slot_states,
    )


def _track_batch(req, tokens, pool):
    """Prefill batch whose FLAMetadata carries the hybrid-radix track indices for ``req``."""
    import freetoken.core as core
    from freetoken.attention.linear import build_fla_metadata
    from freetoken.core import Context, set_global_ctx

    core._GLOBAL_CTX = None  # test-only: build_fla_metadata reads the state pool off the ctx
    set_global_ctx(Context(page_size=64, linear_state_pool=pool))
    batch = _fake_batch([req], decode=False, input_ids=tokens)
    batch.fla_metadata = build_fla_metadata(batch, torch.device("cpu"))
    return batch


def _tracked_req(table_idx, cached_len, tokens, *, live, ping_pong):
    req = _req(table_idx, cached_len, tokens, extend_len=len(tokens))
    req.linear_slot_idx = live
    req.mamba_ping_pong = ping_pong
    req.mamba_next_track_idx = 0
    return req


def _no_eos_tokens(n, start=0):
    return [(t + start) * 13 % (VOCAB - 8) + 8 for t in range(n)]


def test_track_snapshot_equals_a_prefill_stopped_at_the_boundary():
    """The snapshot in the donated slot equals the state a prefill truncated at the boundary leaves."""
    from freetoken.kernel.fla.chunk import CHUNK_SIZE

    torch.manual_seed(17)
    config = _config()
    args = config.qwen4_args
    layer = _make_layer(config)
    pool = _state_pool(config)
    live, dst = 1, 5
    tokens = _no_eos_tokens(CHUNK_SIZE + 6)
    req = _tracked_req(0, 0, tokens, live=live, ping_pong=(dst, 6))
    batch = _track_batch(req, tokens, pool)

    fla = batch.fla_metadata
    assert fla.track_dst.tolist() == [dst]
    assert req.mamba_last_track_seqlen == CHUNK_SIZE
    assert fla.track_boundary_row.tolist() == [CHUNK_SIZE]

    R = torch.randn(len(tokens), args.ple_state_width)
    slab = pool.slot_state("ple_conv", args.ple_layer_ids[0])
    layer.forward(R, batch, meta=_meta([tokens], [[EOS, EOS]], slots=[live]), conv_states=slab)
    got = pool.slot_state("ple_conv", args.ple_layer_ids[0])[dst].clone()

    stopped = torch.zeros_like(slab)
    _forward(layer, R[:CHUNK_SIZE], _meta([tokens[:CHUNK_SIZE]], [[EOS, EOS]], slots=[live]), stopped)
    assert torch.equal(got, stopped[live])


def test_prefix_hit_matches_the_uncached_run():
    """A prefix hit that COW-restores the donated snapshot reproduces the tail of an uncached prefill."""
    from freetoken.kernel.fla.chunk import CHUNK_SIZE

    torch.manual_seed(18)
    config = _config()
    args = config.qwen4_args
    layer = _make_layer(config)
    pool = _state_pool(config)
    tokens = _no_eos_tokens(CHUNK_SIZE + 6)
    context = [[EOS, EOS]]
    R = torch.randn(len(tokens), args.ple_state_width)

    uncached = _forward(
        layer, R, _meta([tokens], context, slots=[1]), torch.zeros_like(pool.slot_state("ple_conv", args.ple_layer_ids[0]))
    )

    live, dst = 1, 5
    req = _tracked_req(0, 0, tokens, live=live, ping_pong=(dst, 6))
    batch = _track_batch(req, tokens, pool)
    layer.forward(R, batch, meta=_meta([tokens], context, slots=[live]), conv_states=pool.slot_state("ple_conv", args.ple_layer_ids[0]))

    resumed_slot = 3
    pool.copy_from(dst, resumed_slot)
    tail = tokens[CHUNK_SIZE:]
    got = _forward(
        layer,
        R[CHUNK_SIZE:],
        _meta([tail], [tokens[CHUNK_SIZE - 2 : CHUNK_SIZE]], slots=[resumed_slot]),
        pool.slot_state("ple_conv", args.ple_layer_ids[0]),
    )
    assert torch.allclose(got, uncached[CHUNK_SIZE:], rtol=1e-5, atol=1e-6)


def test_prefill_matches_stepwise_decode():
    """bs=3 ragged prefill equals feeding the same tokens one decode step at a time."""
    torch.manual_seed(15)
    config = _config()
    args = config.qwen4_args
    layer = _make_layer(config)
    sequences = [[3, 4, EOS, 5, 6, 8, 9, 2, 4, 5, 6, 12], [2, EOS, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                 [9, 10, 11, 12, 13, 14, EOS, 16, 17, 18, 19, 20]]
    length = len(sequences[0])
    contexts = [[EOS, EOS], [21, 22], [EOS, 31]]
    x = torch.randn(len(sequences) * length, args.ple_state_width)
    per_req = [x[i * length : (i + 1) * length] for i in range(len(sequences))]
    zeros = torch.zeros(len(sequences), args.ple_state_width, args.ple_conv_state_len)

    full_states = zeros.clone()
    full = _forward(layer, x, _meta(sequences, contexts), full_states)

    step_states = zeros.clone()
    steps = []
    for t in range(length):
        window = [(c + s)[t : t + 2] for c, s in zip(contexts, sequences)]
        steps.append(
            _forward(
                layer,
                torch.stack([r[t] for r in per_req]),
                _meta([[s[t]] for s in sequences], window, decode=True),
                step_states,
            )
        )
    for i in range(len(sequences)):
        got = torch.stack([step[i] for step in steps])
        assert torch.allclose(got, full[i * length : (i + 1) * length], rtol=1e-4, atol=1e-5)
    assert torch.allclose(step_states, full_states, rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------------------
# full layer vs HF
# --------------------------------------------------------------------------------------


def _layer_ref_inputs(config, layer, tokens=None, hidden=None):
    args = config.qwen4_args
    data = {
        "key_proj": layer.key_proj.weight.float().cpu().numpy(),
        "value_proj": layer.value_proj.weight.float().cpu().numpy(),
        "norm_key": layer.norm_key.weight.float().cpu().numpy(),
        "norm_query": layer.norm_query.weight.float().cpu().numpy(),
        "norm_conv": layer.norm_conv.weight.float().cpu().numpy(),
        "conv1d": layer.conv1d.weight.float().cpu().numpy(),
        "table": layer.ple_embedding.table.weight.float().cpu().numpy(),
    }
    if tokens is None:
        tokens = np.array([[3, 4]], dtype=np.int64)
    if hidden is None:
        hidden = np.zeros((1, tokens.shape[1], args.ple_state_width), dtype=np.float32)
    data["layer_tokens"] = tokens
    data["hidden"] = hidden
    return data


@requires_cuda
@requires_hf_ref
def test_layer_matches_hf(tmp_path):
    """bf16 PLELayer output matches the fp32 HF Qwen4ExpTextPLELayer within 2e-2."""
    torch.manual_seed(16)
    config = _config()
    args = config.qwen4_args
    layer = _make_layer(config)
    tokens = [3, 4, EOS, 5, 6, 8, 9, 2, 4, 5, 6, 12, 13, 14]
    hidden = (torch.randn(1, len(tokens), args.ple_state_width) * 0.5).numpy()
    ref = _run_hf_reference(
        tmp_path,
        {
            "hash_tokens": np.array([[EOS, EOS, 3]], dtype=np.int64),
            **_layer_ref_inputs(config, layer, np.array([tokens], dtype=np.int64), hidden),
        },
    )
    want = torch.as_tensor(ref["layer_out"])[0]
    assert int(ref["padded_vocab_size"]) == layer.ple_embedding.table.num_rows

    gpu = _make_layer(config, device="cuda", dtype=torch.bfloat16)
    for name in ("key_proj", "value_proj", "norm_key", "norm_query", "norm_conv"):
        getattr(gpu, name).weight.copy_(getattr(layer, name).weight)
    gpu.conv1d.weight.copy_(layer.conv1d.weight)
    gpu.ple_embedding.attach_table(
        GpuResidentTable(layer.ple_embedding.table.weight.to("cuda", torch.bfloat16), dtype=torch.bfloat16)
    )
    R = torch.as_tensor(hidden)[0].to("cuda", torch.bfloat16)
    states = torch.zeros(1, args.ple_state_width, args.ple_conv_state_len, device="cuda", dtype=torch.bfloat16)
    got = _forward(gpu, R, _meta([tokens], [[EOS, EOS]], device="cuda"), states)
    assert torch.allclose(got.float().cpu(), want, rtol=2e-2, atol=2e-2)


# --------------------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------------------


def _fake_batch(reqs, *, decode, input_ids, positions=None, table_idx=None, device="cpu"):
    return SimpleNamespace(
        padded_reqs=reqs,
        reqs=reqs,
        is_decode=decode,
        is_prefill=not decode,
        input_ids=torch.tensor(input_ids, dtype=torch.int64, device=device),
        positions=None if positions is None else torch.tensor(positions, dtype=torch.int32, device=device),
        linear_table_idx=(
            None if table_idx is None else torch.tensor(table_idx, dtype=torch.int32, device=device)
        ),
    )


def _req(table_idx, cached_len, host_ids, extend_len=1):
    return SimpleNamespace(
        table_idx=table_idx,
        cached_len=cached_len,
        extend_len=extend_len,
        linear_slot_idx=None,
        input_ids=torch.tensor(host_ids, dtype=torch.int64),
    )


def test_commit_writes_the_track_slot_at_the_boundary():
    """The donated snapshot must carry the context AT the xCHUNK boundary, not the chunk end."""
    from freetoken.kernel.fla.chunk import CHUNK_SIZE

    args = _config().qwen4_args
    eos = args.ngram_boundary_token_id
    ctxp = torch.full((8, 2), eos, dtype=torch.int32)
    tokens = _no_eos_tokens(CHUNK_SIZE + 6)
    batch = _fake_batch([_req(1, 0, tokens, extend_len=len(tokens))], decode=False, input_ids=tokens)
    meta = build_ple_metadata(batch, args, torch.device("cpu"), context_pool=ctxp)
    fla = SimpleNamespace(
        track_boundary_row=torch.tensor([CHUNK_SIZE]), track_dst=torch.tensor([5])
    )
    commit_ngram_context(meta, fla, ctxp)
    assert ctxp[1].tolist() == tokens[-2:]
    assert ctxp[5].tolist() == tokens[CHUNK_SIZE - 2 : CHUNK_SIZE]


def test_context_matches_the_token_history_across_chunks_and_decode():
    """Rolling the slot state chunk by chunk reproduces the last-2-tokens oracle exactly."""
    args = _config().qwen4_args
    eos = args.ngram_boundary_token_id
    ctxp = torch.full((3, 2), 99, dtype=torch.int32)  # stale tenant garbage; fresh rows must mask to eos
    history = _no_eos_tokens(11, start=3)
    cached = 0
    for chunk in (3, 1, 2, 5):
        ids = history[cached : cached + chunk]
        batch = _fake_batch(
            [_req(1, cached, history[: cached + chunk], extend_len=chunk)],
            decode=False, input_ids=ids,
        )
        meta = build_ple_metadata(batch, args, torch.device("cpu"), context_pool=ctxp)
        assert meta.ngram_context.tolist() == [([eos, eos] + history[:cached])[-2:]]
        commit_ngram_context(meta, None, ctxp)
        cached += chunk
    for step in range(3):
        tok = 200 + step
        batch = _fake_batch(
            [_req(1, cached, history + [tok], extend_len=1)],
            decode=True, input_ids=[tok], positions=[cached], table_idx=[1],
        )
        meta = build_ple_metadata(batch, args, torch.device("cpu"), context_pool=ctxp)
        assert meta.ngram_context.tolist() == [history[-2:]]
        commit_ngram_context(meta, None, ctxp)
        history.append(tok)
        cached += 1


# --------------------------------------------------------------------------------------
# CUDA graph + prefetch overlap
# --------------------------------------------------------------------------------------


@requires_cuda
def test_decode_graph_replay_matches_eager():
    """A captured decode PLE forward replays to the eager result, table gather included."""
    torch.manual_seed(17)
    config = _config()
    args = config.qwen4_args
    rows, bs = 4096, 4
    layer = _make_layer(config, device="cuda", dtype=torch.bfloat16, rows=rows)
    bank = _pinned_bank(rows, args.ngram_head_dim, torch.float8_e4m3fn)
    layer.ple_embedding.attach_table(PinnedUVATable(bank.tensor, 0.05))

    ctxp = torch.full((bs + 1, 2), EOS, dtype=torch.int32, device="cuda")
    ctxp[1:] = torch.randint(0, VOCAB, (bs, 2), device="cuda", dtype=torch.int32)
    positions = torch.full((bs,), 8, dtype=torch.int32, device="cuda")
    slots = torch.arange(1, bs + 1, dtype=torch.int32, device="cuda")
    batch = SimpleNamespace(
        padded_reqs=[None] * bs, is_decode=True, is_prefill=False,
        input_ids=torch.randint(0, VOCAB, (bs,), device="cuda", dtype=torch.int32),
        positions=positions, linear_table_idx=slots,
    )
    R = torch.randn(bs, args.ple_state_width, device="cuda", dtype=torch.bfloat16)
    states0 = torch.randn(bs + 1, args.ple_state_width, args.ple_conv_state_len,
                          device="cuda", dtype=torch.bfloat16) * 0.1
    states = states0.clone()

    def step():
        layer.start_prefetch(batch, build_ple_metadata(batch, args, R.device, context_pool=ctxp))
        return layer.forward(R, batch, conv_states=states)

    eager = step().clone()
    eager_states = states.clone()

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            states.copy_(states0)
            step()
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    states.copy_(states0)
    with torch.cuda.graph(graph):
        static_out = step()
    states.copy_(states0)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(static_out, eager)
    assert torch.equal(states, eager_states)

    # new inputs in the same buffers must flow through the replay
    ctxp[1:] = torch.randint(0, VOCAB, (bs, 2), device="cuda", dtype=torch.int32)
    batch.input_ids.copy_(torch.randint(0, VOCAB, (bs,), device="cuda", dtype=torch.int32))
    states.copy_(states0)
    graph.replay()
    replayed = static_out.clone()
    states.copy_(states0)
    assert torch.equal(step(), replayed)

    # a bigger eager gather (prefill) must not move the buffer the graph writes into
    layer.ple_embedding.table.lookup(torch.randint(0, rows, (4096, 16), device="cuda"))
    states.copy_(states0)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(static_out, replayed)
