"""Logits parity against the vendored reference implementation on the tiny synthetic checkpoint.

The reference runs bf16 or FP8/FP4 weights with its own torch quantizers, sparse attention, engram
hash and single-pass mHC; FreeToken runs the same checkpoint through its kernels and packed pools.
Compared: exact-mode prefill, greedy decode steps, a two-request batch, and the bounded-replay path
against the reference's ``forward_bounded`` oracle.
"""

from __future__ import annotations

from dataclasses import asdict, fields

import pytest
import torch

from .common import VOCAB, requires_cuda, tiny_text_config, write_tiny_checkpoint

pytestmark = requires_cuda

# The sparse-attention implementations accumulate in different orders.
ATOL = RTOL = 1e-2
# The fp4 experts also round the routed sum to bf16 before the shared-expert add and weight each
# expert's down output; the reference keeps that sum fp32 and weights the intermediate before its fp8
# round-trip (10 seeds x 16 checks: max |err| 0.019, against 0.007 in the reference's order).
QUANT_TOL = 2e-2


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    folder = tmp_path_factory.mktemp("dsv41-parity")
    tensors = write_tiny_checkpoint(str(folder), seed=11)
    return str(folder), tensors


def _tokens(n: int, seed: int) -> torch.Tensor:
    return torch.randint(3, VOCAB, (n,), generator=torch.Generator().manual_seed(seed))


class Reference:
    """The vendored ``Transformer`` on CUDA, loaded from the synthetic checkpoint (identity token map)."""

    def __init__(self, tensors: dict, text: dict, *, max_seq_len: int, max_batch_size: int, quantized: bool = False, vision_config: dict | None = None):
        from freetoken.models.deepseek_v41.args import DeepseekV41Args
        from types import SimpleNamespace

        from . import reference
        from .reference import engram as ref_engram
        from .reference import model as ref_model

        args = DeepseekV41Args.from_hf(SimpleNamespace(text_config=text))
        names = {f.name for f in fields(ref_model.ModelArgs)}
        kwargs = {k: v for k, v in asdict(args).items() if k in names}
        kwargs.update(dtype="fp8" if quantized else "bf16", expert_dtype="fp4" if quantized else None,
                      max_seq_len=max_seq_len, max_batch_size=max_batch_size, vision_n_layers=0, temperature=0.0)
        if vision_config:
            kwargs.update(vision_n_layers=vision_config["num_hidden_layers"], vision_dim=vision_config["hidden_size"],
                          vision_n_heads=vision_config["num_attention_heads"], vision_inter_dim=vision_config["intermediate_size"],
                          vision_patch_size=vision_config["patch_size"], vision_downsample_ratio=vision_config["downsample_ratio"])
        self.margs = ref_model.ModelArgs(**kwargs)
        # the synthetic checkpoint's tokenizer is the identity over VOCAB ids
        ref_engram.build_compressed_token_map = lambda tokenizer: (list(range(VOCAB)), VOCAB)
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("cuda"):
                self.model = ref_model.Transformer(self.margs, tokenizer=object())
        finally:
            torch.set_default_dtype(torch.float32)
        state = {k: v for k, v in tensors.items() if not k.startswith("mtp.")
                 and (vision_config or (not k.startswith("vision.") and not k.endswith("bias_vl")))}
        if quantized:
            state = {k: v.view(torch.float4_e2m1fn_x2) if ".ffn.experts." in k and k.endswith(".weight") else v
                     for k, v in state.items()}
            # The reference's grouped output projection holds dequantized bf16 weights.
            for name in list(state):
                if name.endswith(".wo_a.scale"):
                    scale = state.pop(name).float().repeat_interleave(32, 0).repeat_interleave(32, 1)
                    weight_name = name.removesuffix(".scale") + ".weight"
                    state[weight_name] = (state[weight_name].float() * scale).bfloat16()
        missing, unexpected = self.model.load_state_dict({k: v.cuda() for k, v in state.items()}, strict=False)
        assert not unexpected, unexpected
        assert not [m for m in missing if "freqs_cis" not in m], missing
        self.reference = reference

    # the reference builds its index tensors on the default device (generate.py sets it to cuda)
    def prefill(self, ids: torch.Tensor) -> torch.Tensor:
        with torch.device("cuda"):
            return self.model(ids.view(1, -1).cuda(), 0)[1].float()

    def prefill_batch(self, ids: torch.Tensor) -> torch.Tensor:
        with torch.device("cuda"):
            return self.model(ids.cuda(), 0)[1].float()

    def prefill_bounded(self, ids: torch.Tensor) -> torch.Tensor:
        with torch.device("cuda"):
            return self.model.forward_bounded(ids.reshape(-1, ids.shape[-1]).cuda(), self.margs.window_size).float()

    def prefill_replay(self, prefix: torch.Tensor, suffix: torch.Tensor, n_win: int,
                       exact_decoder: bool = False) -> torch.Tensor:
        with torch.device("cuda"):
            return self.model.forward_replay(prefix.view(1, -1).cuda(), suffix.view(1, -1).cuda(),
                                             n_win, exact_decoder).float()

    def decode(self, tokens: torch.Tensor, pos: int) -> torch.Tensor:
        with torch.device("cuda"):
            return self.model(tokens.view(-1, 1).cuda(), pos)[1].float()


def _engram_table_for(engine, tensors, text):
    """Our engine reads the Engram rows straight from the synthetic shard (the disk table), hashed
    with the same identity token map the reference oracle uses."""
    from freetoken.models.deepseek_v41.engram import EngramHash
    from freetoken.models.deepseek_v41.engram_table import EngramDiskTable, EngramHost, engram_row_source

    hash = EngramHash(engine.args, list(range(VOCAB)), VOCAB)
    tables = [
        EngramDiskTable(engram_row_source(engine.checkpoint, layer.layer_id), hash.n_cols, engine.device, max_graph_rows=8, max_extend_tokens=64)
        for layer in engine.model.engram_layers()
    ]
    for layer, table in zip(engine.model.engram_layers(), tables):
        layer.attach_table(table)
    host = EngramHost(hash, tables, engine.device)
    engine.model.forward_host_ctx = host.forward_host_ctx
    engine.engram_host = host


def _compare(name, got, want, *, tol=ATOL):
    err = (got - want).abs().max().item()
    assert torch.equal(got.argmax(-1), want.argmax(-1)), f"{name}: argmax differs (max abs err {err:.4f})"
    torch.testing.assert_close(got, want, atol=tol, rtol=tol, msg=lambda m: f"{name}: {m}")


@pytest.mark.parametrize("mode", ["exact", "bounded"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_quantized_offload_prefill_and_decode_match_reference(tmp_path, mode, batch_size):
    from .harness import TinyEngine

    tensors = write_tiny_checkpoint(str(tmp_path), seed=11, quantized=True)
    text = tiny_text_config(moe_intermediate_size=256)
    ref = Reference(tensors, text, max_seq_len=1024, max_batch_size=3, quantized=True)
    eng = TinyEngine(str(tmp_path), swa_decoder_replay=mode, quantized=True)
    _engram_table_for(eng, tensors, text)
    ids = torch.stack([_tokens(300, 1 + i) for i in range(batch_size)])
    want = ref.prefill_batch(ids) if mode == "exact" else ref.prefill_bounded(ids)
    reqs = [eng.new_request(i, row.tolist()) for i, row in enumerate(ids)]
    got = eng.prefill(reqs)
    _compare(f"quantized {mode} prefill", got, want, tol=QUANT_TOL)
    eng.finish_prefill(reqs)
    for step in range(3):
        token = want.argmax(-1)
        want = ref.decode(token, ids.shape[-1] + step)
        got = eng.decode(reqs, token.tolist())
        _compare(f"quantized {mode} decode {step}", got, want, tol=QUANT_TOL)


def test_exact_prefill_and_greedy_decode_match_the_reference(checkpoint):
    from .harness import TinyEngine

    folder, tensors = checkpoint
    text = tiny_text_config()
    ref = Reference(tensors, text, max_seq_len=1024, max_batch_size=3)
    eng = TinyEngine(folder, max_seq_len=1024, max_running_req=2, swa_decoder_replay="exact")
    _engram_table_for(eng, tensors, text)

    ids = _tokens(300, 1)
    want = ref.prefill(ids)
    req = eng.new_request(0, ids.tolist())
    got = eng.prefill([req])
    _compare("prefill", got, want)
    eng.finish_prefill([req])
    # greedy decode from the reference's own picks keeps both sides on one trajectory
    pos = 300
    for step in range(12):
        nxt = int(want.argmax(-1).item())
        want = ref.decode(torch.tensor([nxt]), pos)
        got = eng.decode([req], [nxt])
        _compare(f"decode step {step}", got, want)
        pos += 1


def test_batched_prefill_and_decode_match_the_reference(checkpoint):
    from .harness import TinyEngine

    folder, tensors = checkpoint
    text = tiny_text_config()
    ref = Reference(tensors, text, max_seq_len=1024, max_batch_size=3)
    eng = TinyEngine(folder, max_seq_len=1024, max_running_req=2, swa_decoder_replay="exact")
    _engram_table_for(eng, tensors, text)

    ids = torch.stack([_tokens(200, 2), _tokens(200, 3)])
    want = ref.prefill_batch(ids)
    reqs = [eng.new_request(i, ids[i].tolist()) for i in range(2)]
    got = eng.prefill(reqs)
    _compare("batched prefill", got, want)
    eng.finish_prefill(reqs)
    pos = 200
    for step in range(6):
        nxt = want.argmax(-1).cpu()
        want = ref.decode(nxt, pos)
        got = eng.decode(reqs, nxt.tolist())
        _compare(f"batched decode step {step}", got, want)
        pos += 1


def test_bounded_replay_matches_the_reference_oracle(checkpoint):
    from .harness import TinyEngine

    folder, tensors = checkpoint
    text = tiny_text_config()
    ref = Reference(tensors, text, max_seq_len=1024, max_batch_size=3)
    eng = TinyEngine(folder, max_seq_len=1024, max_running_req=2, swa_decoder_replay="bounded")
    _engram_table_for(eng, tensors, text)

    ids = _tokens(300, 4)
    want = ref.prefill_bounded(ids)
    req = eng.new_request(0, ids.tolist())
    got = eng.prefill([req])
    _compare("bounded prefill", got, want)
    eng.finish_prefill([req])
    pos = 300
    for step in range(6):
        nxt = int(want.argmax(-1).item())
        want = ref.decode(torch.tensor([nxt]), pos)
        got = eng.decode([req], [nxt])
        _compare(f"decode after bounded prefill, step {step}", got, want)
        pos += 1


# ---- Encoder SWA Bounded Replay: a prefix hit keeps the FULL global-KV match and replays the
# hit's last window through the encoder (the radix fallback when that window KV was evicted).
# The logits-level drift of a P-token replay is below the parity tolerance, so the oracle check
# is paired with structural pins: the evicted page must be regenerated, the global tiers below
# the hit must be untouched, and the regenerated rows must show the floored-window semantics.

P = 128


def _window_rows(eng, req, lo, hi, layer):
    ws = eng.pool.translate_full_to_window(eng.page_table[req.table_idx, lo:hi].long())
    return eng.pool.read_window(layer, ws)


def _attach_replay(eng, req):
    """What the scheduler does for a bounded-encoder-replay hit: the radix evicted the hit's top
    window page (a later alloc would hand it to someone else, so its rows are zeroed here), the
    rebind gives it a fresh window slot, and the request carries the replay start."""
    F, P = req.cached_len, eng.args.window_size
    locs = eng.page_table[req.table_idx, F - P : F]
    ws = eng.pool.translate_full_to_window(locs.long())
    assert bool((ws >= 0).all()), "the hit's top window page must be live before the eviction"
    eng.pool.free_swa(locs)
    for l in range(eng.args.n_layers):  # a reused page's content is arbitrary; zero it
        if not eng.pool.is_private_window(l):
            eng.pool.window_pool[l][ws] = 0
    eng.pool.alloc_swa(locs)
    req.enc_replay_lo = F - P


def _global_tiers(eng, table_idx: int, upto: int):
    """Dequantized global-tier rows of [0, upto) per kv source, for the bit-exact no-overwrite check."""
    out = []
    for src in eng.args.backbone_kv_sources:
        ratio = eng.args.compress_ratios[src]
        starts = torch.arange(0, upto // ratio, device=eng.device) * ratio
        rows = torch.div(eng.page_table[table_idx, starts].long(), ratio)
        out.append((eng.pool.read_main(src, rows).clone(), eng.pool.read_index(src, rows).clone()))
    return out


def _replay_setup(checkpoint, decoder_replay: str = "bounded"):
    from .harness import TinyEngine

    folder, tensors = checkpoint
    text = tiny_text_config()
    ref = Reference(tensors, text, max_seq_len=2048, max_batch_size=3)
    eng = TinyEngine(folder, max_seq_len=2048, max_running_req=2, swa_decoder_replay=decoder_replay)
    _engram_table_for(eng, tensors, text)
    prefix, suffix = _tokens(640, 31), _tokens(148, 32)
    donor = eng.new_request(0, prefix.tolist() + suffix.tolist())
    eng.prefill([donor])
    eng.finish_prefill([donor])
    full = prefix.tolist() + suffix.tolist()
    # the oracle's prefix is the depth the scheduler hits (new_request_on_prefix's rule): capped a
    # window below the end under bounded replay, the full page-aligned prompt under exact
    limit = len(full) - 1
    if eng.pool.prefix_replay_tokens:
        limit = min(limit, max(0, len(full) - eng.pool.prefix_replay_tokens))
    F = min(donor.device_len, limit) // P * P
    want = ref.prefill_replay(torch.tensor(full[:F]), torch.tensor(full[F:]), P,
                              exact_decoder=decoder_replay == "exact")
    return eng, ref, donor, full, want


def test_encoder_bounded_replay_matches_the_reference_oracle(checkpoint):
    eng, ref, donor, toks, want = _replay_setup(checkpoint)

    hit = eng.new_request_on_prefix(1, donor, toks)
    assert hit.cached_len == 640 and hit.extend_len == 148
    before_g = _global_tiers(eng, 1, 640)
    before_w = {l: _window_rows(eng, hit, 640 - P, 640, l) for l in (0, 2)}  # a window and a compressing layer
    _attach_replay(eng, hit)
    assert hit.extend_len == 148 + P
    got = eng.prefill([hit])
    _compare("encoder bounded replay prefill", got, want)
    for (m0, i0), (m1, i1) in zip(before_g, _global_tiers(eng, 1, 640)):
        assert torch.equal(m0, m1) and torch.equal(i0, i1), "the replay must not rewrite global rows [0, F)"
    now0, now2 = _window_rows(eng, hit, 640 - P, 640, 0), _window_rows(eng, hit, 640 - P, 640, 2)
    assert bool((now0 != 0).any()) and bool((now2 != 0).any()), "the replayed page was not regenerated"
    # layer 0's window KV is a pure projection of the embedding: the recompute is bit-exact
    assert torch.equal(now0, before_w[0])
    # deeper rows show the floor: position 639's window ([512, 639]) is unchanged by the replay,
    # position 512's ([512, 512]) lost the 127 keys below the floor the original pass had
    d = (now2 - before_w[2]).abs().mean(-1)
    assert d[0].item() > 5 * max(d[-1].item(), 1e-6), f"floored-window drift {d[0]:.4f} vs full-window {d[-1]:.4f}"
    eng.finish_prefill([hit])
    pos = 788
    for step in range(6):
        nxt = int(want.argmax(-1).item())
        want = ref.decode(torch.tensor([nxt]), pos)
        got = eng.decode([hit], [nxt])
        _compare(f"decode after encoder replay, step {step}", got, want)
        pos += 1


def test_encoder_bounded_replay_under_exact_decoder_matches_the_reference_oracle(checkpoint):
    """Exact decoder + bounded encoder (the corner the backend metadata test pins structurally):
    the uncapped hit is the tree's full page-aligned depth, the decoder runs every replay-chunk
    row with the window still floored at the replay start, and the CED publishes only [hit, N)."""
    eng, ref, donor, toks, want = _replay_setup(checkpoint, decoder_replay="exact")

    hit = eng.new_request_on_prefix(1, donor, toks)
    assert hit.cached_len == 6 * P and hit.extend_len == 20    # no replay cap: the full committed depth
    _attach_replay(eng, hit)
    assert hit.extend_len == 20 + P
    got = eng.prefill([hit])
    _compare("exact-decoder encoder bounded replay prefill", got, want)
    assert bool((_window_rows(eng, hit, 6 * P - P, 6 * P, 2) != 0).any()), "the replayed page was not regenerated"
    eng.finish_prefill([hit])
    pos = 788
    for step in range(4):
        nxt = int(want.argmax(-1).item())
        want = ref.decode(torch.tensor([nxt]), pos)
        got = eng.decode([hit], [nxt])
        _compare(f"decode after exact-decoder encoder replay, step {step}", got, want)
        pos += 1


def test_encoder_bounded_replay_attaches_to_the_first_chunk_only(checkpoint):
    """The replay rides chunk 1 (which must reach past the hit F); the continuation is an ordinary
    chunk whose cached watermark already covers the replayed page."""
    eng, ref, donor, toks, want = _replay_setup(checkpoint)

    hit = eng.new_request_on_prefix(1, donor, toks)
    _attach_replay(eng, hit)
    hit.device_len = 768  # chunk 1 = [512, 768): the replay page plus [640, 768) of new tokens
    first = eng.prefill([hit])
    assert first.shape == (1, VOCAB) and torch.isfinite(first).all()
    assert bool((_window_rows(eng, hit, 640 - P, 640, 2) != 0).any()), "chunk 1 did not replay the dead page"
    hit.cached_len, hit.device_len = 768, 788
    hit.enc_replay_lo = -1  # complete_one clears the attach for the continuation
    _compare("encoder replay chunked prefill", eng.prefill([hit]), want)
