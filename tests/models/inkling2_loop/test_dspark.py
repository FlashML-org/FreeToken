"""Inkling 2 DSpark draft against an fp32 torch reference of the vLLM / sglang definition, and DSparkProposer's bookkeeping against a from-scratch draft.

The reference runs one request at a time: standard pre-norm layers over ``[anchor, mask x 7]``, dense attention with explicit window masks, the
Markov head with ``hidden_out`` unfolded. A tiny synthetic draft covers every path; ``FREETOKEN_INKLING2_MINI_MODEL`` repeats the parity at the
checkpoint's own weights and dimensions. The proposer test drives a toy looped target through chunked prefill and verify steps with every
accept count on a scattered page table; after each step its drafts must equal a draft recomputed from the committed tokens alone, with the
features captured the way vLLM and sglang capture them (at the inputs of the tapped visits).
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REAL_MODEL = os.environ.get("FREETOKEN_INKLING2_MINI_MODEL", "")

# a looped toy target: 4 layers, core [1, 3) run 3 times -> 8 visits; tap 5 restarts the core, tap 8 is the final norm
TOY = dict(hidden=256, layers=3, heads=4, kv_heads=2, head_dim=64, inter=512, vocab=512, padded_vocab=528, window=16, block=8,
           rank=32, local=(0, 2), taps=(2, 5, 7, 8), target_layers=4, core=(1, 3), passes=3)


def _hf_config(t: dict) -> dict:
    """The target checkpoint's config.json as far as the draft reads it."""
    return {
        "architectures": ["Inkling2LoopForConditionalGeneration"], "model_type": "inkling2_loop",
        "text_config": {"hidden_size": t["hidden"], "num_hidden_layers": t["target_layers"], "vocab_size": t["padded_vocab"],
                        "unpadded_vocab_size": t["vocab"], "final_logit_softcapping": 30.0, "rms_norm_eps": 1e-6,
                        "loop_config": {"core_start": t["core"][0], "core_end": t["core"][1], "num_passes": t["passes"],
                                        "gate_rank": 8, "is_varloop": False}},
        "mtp_config": {"hidden_size": t["hidden"], "num_hidden_layers": t["layers"], "num_attention_heads": t["heads"],
                       "num_key_value_heads": t["kv_heads"], "head_dim": t["head_dim"], "intermediate_size": t["inter"],
                       "max_position_embeddings": 1 << 20, "torch_dtype": "bfloat16", "q_bias": False, "o_bias": False,
                       "rms_norm_eps": 1e-6, "rope_theta": 1e6, "local_layer_ids": list(t["local"]), "sliding_window_size": t["window"],
                       "block_size": t["block"], "markov_rank": t["rank"], "markov_head_type": "rnn_hidden",
                       "target_layer_ids": list(t["taps"])},
    }


def _toy_weights(t: dict, seed: int = 0) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Fan-in scaled bf16 draft tensors under the checkpoint's names, and the target's embedding, embedding norm and head."""
    g = torch.Generator().manual_seed(seed)
    H, q, kv, r = t["hidden"], t["heads"] * t["head_dim"], t["kv_heads"] * t["head_dim"], t["rank"]

    def mat(*shape):
        return (torch.randn(*shape, generator=g) * shape[-1] ** -0.5).to(torch.bfloat16)

    def gain(n):
        return (1 + 0.1 * torch.randn(n, generator=g)).to(torch.bfloat16)

    w = {"fuse_weight": torch.randn(t["layers"] + 1, len(t["taps"]), generator=g).to(torch.bfloat16),
         "mask_embed": torch.randn(H, generator=g).to(torch.bfloat16), "norm.weight": gain(H),
         "markov_head.dflare_norm.weight": gain(H), "markov_head.markov_w1.weight": mat(t["padded_vocab"], r),
         "markov_head.rnn_proj.weight": mat(3 * r, 2 * r + 2 * H), "markov_head.rnn_proj.bias": (0.1 * torch.randn(3 * r, generator=g)).to(torch.bfloat16),
         "markov_head.hidden_out.weight": mat(H, r), "markov_head.hidden_out.bias": (0.1 * torch.randn(H, generator=g)).to(torch.bfloat16)}
    for i in range(t["layers"]):
        p = f"layers.{i}."
        w.update({p + "dflare_norm.weight": gain(H), p + "input_layernorm.weight": gain(H), p + "post_attention_layernorm.weight": gain(H),
                  p + "self_attn.q_proj.weight": mat(q, H), p + "self_attn.k_proj.weight": mat(kv, H), p + "self_attn.v_proj.weight": mat(kv, H),
                  p + "self_attn.o_proj.weight": mat(H, q), p + "self_attn.q_norm.weight": gain(t["head_dim"]),
                  p + "self_attn.k_norm.weight": gain(t["head_dim"]), p + "self_attn.k_norm_ctx.weight": gain(t["head_dim"]),
                  p + "self_attn.k_proj_ctx.weight": mat(kv, H), p + "self_attn.v_proj_ctx.weight": mat(kv, H),
                  p + "mlp.gate_proj.weight": mat(t["inter"], H), p + "mlp.up_proj.weight": mat(t["inter"], H), p + "mlp.down_proj.weight": mat(H, t["inter"])})
    target = {"embed": torch.randn(t["padded_vocab"], H, generator=g).to(torch.bfloat16), "embed_norm": gain(H),
              "head": (torch.randn(t["padded_vocab"], H, generator=g) * 0.3).to(torch.bfloat16)}
    return w, target


def _target(parts: dict[str, torch.Tensor]):
    """The modules the draft shares with the target, as FreeToken's Inkling target names them: the token embedding, its fp32 norm and the bf16 head."""
    embed, norm, head = (parts[n].to(DEV) for n in ("embed", "embed_norm", "head"))
    embed_tokens = SimpleNamespace(forward=lambda ids: embed[ids.long()])
    return SimpleNamespace(model=SimpleNamespace(embed_tokens=embed_tokens, embed_norm=norm.float()), lm_head=SimpleNamespace(weight=head))


def _toy_checkpoint(tmp_path, t: dict = TOY, seed: int = 0):
    path = tmp_path / "ckpt"
    path.mkdir(parents=True, exist_ok=True)
    w, target = _toy_weights(t, seed)
    (path / "config.json").write_text(json.dumps(_hf_config(t)))
    save_file(w, str(path / "mtp.safetensors"))
    return str(path), w, target


def _load(path: str, target_parts: dict[str, torch.Tensor], dtype=torch.bfloat16):
    from freetoken.models.inkling2_loop.dspark import Inkling2DSparkDraft, parse_dspark_config
    from freetoken.utils import cached_load_hf_config

    config = parse_dspark_config(cached_load_hf_config(path))
    return Inkling2DSparkDraft.load(config, _target(target_parts), path, DEV, dtype)


class _Reference:
    """The draft in fp32 for one request; ``features [n, taps*H]`` are the target's at positions ``0 .. n-1``."""

    def __init__(self, w: dict[str, torch.Tensor], target: dict[str, torch.Tensor], c) -> None:
        self.w = {name: t.to(DEV, torch.float32) for name, t in w.items()}
        self.t = {name: t.to(DEV, torch.float32) for name, t in target.items()}
        self.c = c
        self.alpha = torch.softmax(self.w["fuse_weight"], -1)

    def rms(self, x, weight):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * weight

    def rope(self, x, pos):
        d = self.c.head_dim
        inv = 1.0 / (self.c.rope_theta ** (torch.arange(0, d, 2, device=DEV, dtype=torch.float64) / d))
        freqs = pos.double()[:, None] * inv[None, :]
        cos, sin = freqs.cos().float()[:, None], freqs.sin().float()[:, None]
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)

    def mixed(self, features, row, norm_name):
        taps = features.float().view(features.shape[0], self.c.num_taps, -1)
        return self.rms((taps * self.alpha[row][None, :, None]).sum(1), self.w[norm_name])

    def context_kv(self, layer, features):
        c, p = self.c, f"layers.{layer}."
        ctx = self.mixed(features, layer, p + "dflare_norm.weight")
        k = self.rms((ctx @ self.w[p + "self_attn.k_proj_ctx.weight"].T).view(-1, c.num_kv_heads, c.head_dim), self.w[p + "self_attn.k_norm_ctx.weight"])
        v = (ctx @ self.w[p + "self_attn.v_proj_ctx.weight"].T).view(-1, c.num_kv_heads, c.head_dim)
        return self.rope(k, torch.arange(features.shape[0], device=DEV)), v

    def block(self, features, anchor: int):
        """Final hidden ``[block, H]`` of ``[anchor, mask x 7]`` at positions ``n .. n+7`` after the ``n`` context rows."""
        c, w = self.c, self.w
        n, B = features.shape[0], c.block_size
        # the target hands over its first layer's input, a bf16 row
        e = self.rms(self.t["embed"][anchor], self.t["embed_norm"]).to(torch.bfloat16).float()
        x = torch.cat([e[None], w["mask_embed"][None].expand(B - 1, -1)])
        pos = n + torch.arange(B, device=DEV)
        for i in range(c.num_layers):
            p = f"layers.{i}.self_attn."
            h = self.rms(x, w[f"layers.{i}.input_layernorm.weight"])
            q = self.rope(self.rms((h @ w[p + "q_proj.weight"].T).view(B, c.num_heads, c.head_dim), w[p + "q_norm.weight"]), pos)
            k = self.rope(self.rms((h @ w[p + "k_proj.weight"].T).view(B, c.num_kv_heads, c.head_dim), w[p + "k_norm.weight"]), pos)
            v = (h @ w[p + "v_proj.weight"].T).view(B, c.num_kv_heads, c.head_dim)
            ck, cv = self.context_kv(i, features)
            # a local layer reads the window of positions before the block start; every layer reads the whole block
            lo = max(0, n - c.window) if i in c.local_layer_ids else 0
            keys, values = torch.cat([ck[lo:], k]), torch.cat([cv[lo:], v])
            group = c.num_heads // c.num_kv_heads
            out = torch.empty(B, c.num_heads, c.head_dim, device=DEV)
            for head in range(c.num_heads):
                scores = q[:, head] @ keys[:, head // group].T * c.head_dim**-0.5
                out[:, head] = torch.softmax(scores, -1) @ values[:, head // group]
            x = x + out.reshape(B, -1) @ w[p + "o_proj.weight"].T
            h = self.rms(x, w[f"layers.{i}.post_attention_layernorm.weight"])
            m = F.silu(h @ w[f"layers.{i}.mlp.gate_proj.weight"].T) * (h @ w[f"layers.{i}.mlp.up_proj.weight"].T)
            x = x + m @ w[f"layers.{i}.mlp.down_proj.weight"].T
        return self.rms(x, w["norm.weight"])

    def path_logits(self, hidden, last_features, anchor: int, tokens):
        """Each mask row's logits ``[steps, vocab]`` with the Markov state advanced over ``tokens`` (teacher forced): ``softcap(head @ (h + hidden_out(latent)))``."""
        w, r, V = self.w, self.c.markov_rank, self.c.vocab_size
        f_rnn = self.mixed(last_features[None], -1, "markov_head.dflare_norm.weight")[0]
        head = self.t["head"][:V]
        state, prev, out = torch.zeros(r, device=DEV), anchor, []
        for j in range(hidden.shape[0]):
            z = torch.cat([state, w["markov_head.markov_w1.weight"][prev], hidden[j], f_rnn])
            gate, cand, o = (z @ w["markov_head.rnn_proj.weight"].T + w["markov_head.rnn_proj.bias"]).split(r)
            gate = torch.sigmoid(gate)
            state = gate * state + (1 - gate) * torch.tanh(cand)
            bias = torch.tanh(o) @ w["markov_head.hidden_out.weight"].T + w["markov_head.hidden_out.bias"]
            logits = head @ hidden[j] + head @ bias
            out.append(torch.tanh(logits / 30.0) * 30.0)
            prev = int(tokens[j])
        return torch.stack(out)


def _check_draft(draft, ref, features_by_req, anchors, kv_tol, hidden_tol, logit_tol):
    """Context K/V, block hidden and each drafted token of a batch of requests against the reference; a draft may differ from the reference's argmax only within ``logit_tol``."""
    c = draft.config
    starts = torch.tensor([f.shape[0] for f in features_by_req], device=DEV)
    kvs = [draft.context_kv(f, torch.arange(f.shape[0], device=DEV)) for f in features_by_req]
    for f, (k, v) in zip(features_by_req, kvs):
        for i in range(c.num_layers):
            # bf16 rounds the mixed context and the projection, about 2^-8 of each head row's norm
            for got, want in zip((k[i], v[i]), ref.context_kv(i, f)):
                err = (got.float() - want).norm(dim=-1) / want.norm(dim=-1)
                assert err.max() < kv_tol, f"layer {i} context K/V off by {err.max():.3g} of a head row"
    width = int(starts.max())

    def context(layer):
        # right-aligned dense context; a local layer keeps only the window before each start
        k = torch.zeros(len(kvs), width, c.num_kv_heads, c.head_dim, dtype=draft.dtype, device=DEV)
        v, valid = torch.zeros_like(k), torch.zeros(len(kvs), width, dtype=torch.bool, device=DEV)
        for b, (kk, vv) in enumerate(kvs):
            n = kk.shape[1]
            lo = max(0, n - c.window) if layer in c.local_layer_ids else 0
            k[b, width - n + lo :], v[b, width - n + lo :] = kk[layer, lo:], vv[layer, lo:]
            valid[b, width - n + lo :] = True
        return [(k, v, valid)]

    anchor = torch.tensor(anchors, device=DEV)
    hidden = draft.forward_block(anchor, starts, context)
    f_rnn = draft.f_rnn(torch.stack([f[-1] for f in features_by_req]))
    steps = []

    def pick(j, logits):
        steps.append(logits.float())
        return logits.argmax(-1)

    tokens = draft.walk(hidden[:, 1:], anchor, f_rnn, pick)
    for b, f in enumerate(features_by_req):
        want = ref.block(f, anchors[b])
        err = (hidden[b].float() - want).norm(dim=-1) / want.norm(dim=-1)
        assert err.max() < hidden_tol, f"request {b}: block hidden off by {err.max():.3g} (relative)"
        # the Markov head alone: the reference walks the draft's own hidden and tokens
        logits = ref.path_logits(hidden[b, 1:].float(), f[-1], anchors[b], tokens[b])
        got = torch.stack([s[b] for s in steps])
        assert (got - logits).abs().max() < logit_tol, f"request {b}: step logits off by {(got - logits).abs().max():.3g}"
        picked = logits.gather(1, tokens[b, :, None].long())[:, 0]
        # a draft may differ from the fp32 argmax only at a near tie
        assert (picked >= logits.max(-1).values - logit_tol).all(), f"request {b}: drafts {tokens[b].tolist()} miss the reference argmax {logits.argmax(-1).tolist()}"


# (dtype, context K/V, block hidden, logit) tolerances: fp32 pins the math (its rope runs in fp32, the reference's in fp64);
# bf16, the serving dtype, is held to bf16 noise: 2^-8 of a row per rounding, through three layers for the hidden
TOLERANCES = [(torch.float32, 1e-4, 2e-4, 1e-3), (torch.bfloat16, 1e-2, 5e-2, 0.5)]


@pytest.mark.parametrize("dtype, kv_tol, hidden_tol, logit_tol", TOLERANCES)
def test_toy_draft_matches_reference(tmp_path, dtype, kv_tol, hidden_tol, logit_tol):
    """A short context (no window cut) and a long one (local layers cut at the window) in one batch, on the tiny draft."""
    path, w, target = _toy_checkpoint(tmp_path)
    draft = _load(path, target, dtype)
    ref = _Reference(w, target, draft.config)
    g = torch.Generator().manual_seed(1)
    feats = [torch.randn(n, draft.config.num_taps * draft.config.hidden_size, generator=g).to(DEV, torch.bfloat16) for n in (5, 40)]
    _check_draft(draft, ref, feats, [7, 300], kv_tol, hidden_tol, logit_tol)


@pytest.mark.needs_weights
@pytest.mark.skipif(not os.path.isfile(os.path.join(REAL_MODEL, "config.json")), reason="FREETOKEN_INKLING2_MINI_MODEL not set")
@pytest.mark.parametrize("dtype, kv_tol, hidden_tol, logit_tol", TOLERANCES)
def test_checkpoint_draft_matches_reference(dtype, kv_tol, hidden_tol, logit_tol):
    """The checkpoint's draft at its own dimensions, with the target's real embedding and head; contexts below and past the 1024 window."""
    from freetoken.models.loader import safetensors_weight_map

    names = {"embed": "model.llm.embed.weight", "embed_norm": "model.llm.embed_norm.weight", "head": "model.llm.unembed.weight"}
    shards = safetensors_weight_map(REAL_MODEL)
    target = {}
    for key, name in names.items():
        with safe_open(os.path.join(REAL_MODEL, shards[name]), framework="pt") as f:
            target[key] = f.get_tensor(name)
    draft = _load(REAL_MODEL, target, dtype)
    with safe_open(os.path.join(REAL_MODEL, "mtp.safetensors"), framework="pt") as f:
        w = {name: f.get_tensor(name) for name in f.keys()}
    ref = _Reference(w, target, draft.config)
    g = torch.Generator().manual_seed(1)
    # target features are O(1) residual-stream rows; the final-norm column is too
    feats = [torch.randn(n, draft.config.num_taps * draft.config.hidden_size, generator=g).to(DEV, torch.bfloat16) for n in (37, 1100)]
    _check_draft(draft, ref, feats, [1000, 4242], kv_tol, hidden_tol, logit_tol)


# ---------------------------------------------------------------------- proposer


class _ToyLoopTarget:
    """Per-row stand-in for the target's residual stream: visit v maps its input through layer ``layer_of[v]``; a core pass after the first restarts from the prelude's output."""

    def __init__(self, t: dict, H: int, seed: int = 3) -> None:
        g = torch.Generator().manual_seed(seed)
        start, end = t["core"]
        self.visits = list(range(start)) + [i for _ in range(t["passes"]) for i in range(start, end)] + list(range(end, t["target_layers"]))
        self.restarts = {start + p * (end - start) for p in range(1, t["passes"])}
        self.core_start = start
        # elementwise, so a row's features do not depend on which rows share its forward
        self.layers = [(torch.randn(H, generator=g), torch.randn(H, generator=g)) for _ in range(t["target_layers"])]
        self.table = torch.randn(t["padded_vocab"], H, generator=g)
        self.taps = t["taps"]

    def run(self, tokens: torch.Tensor, positions: torch.Tensor, sink=None) -> torch.Tensor:
        """Features ``[n, taps*H]`` captured at the tapped visits' inputs (vLLM / sglang); hands every visit output to ``sink`` the way a FreeToken target does."""
        x = self.table[tokens.long().cpu()] + 0.01 * positions.float().cpu()[:, None]
        captured, prelude = {}, None
        for v, layer in enumerate(self.visits):
            if v == self.core_start:
                prelude = x
            if v in self.restarts:
                x = prelude
            if v in self.taps:
                captured[v] = x
            a, b = self.layers[layer]
            x = torch.tanh(x * a + b) + x
            if sink is not None and v in sink.wanted_layers:
                sink.layer_output(v, x.to(DEV, torch.bfloat16))
        final = torch.tanh(x)
        if sink is not None:
            sink.final_hidden(final.to(DEV, torch.bfloat16))
        cols = [captured[v] for v in self.taps[:-1]] + [final]
        return torch.cat(cols, dim=-1).to(DEV, torch.bfloat16)


def _proposer(path: str, target_parts, k: int, page_table, num_pages: int, page_size: int, max_running_req: int, dtype=torch.bfloat16):
    from freetoken.core import Req, SamplingParams
    from freetoken.spec.dspark.proposer import DSparkProposer
    from freetoken.spec.proposer import DraftContext
    from freetoken.utils import cached_load_hf_config

    config = SimpleNamespace(
        speculative_num_steps=k, speculative_draft_model_path=None, model_path=path, hf_config=cached_load_hf_config(path),
        model_spec=SimpleNamespace(spec_drafts=(("dspark_mtp", "freetoken.models.inkling2_loop.dspark:Inkling2DSparkDraft"),)),
        dtype=dtype, max_running_req=max_running_req, page_size=page_size,
    )
    proposer = DSparkProposer(config, _target(target_parts))
    dummy = Req(input_ids=torch.zeros(1, dtype=torch.int32), table_idx=max_running_req, cached_len=0, output_len=1, uid=-1,
                sampling_params=SamplingParams(), cache_handle=None)
    proposer.attach(DraftContext(page_table=page_table, num_pages=num_pages, device=DEV, stream=None, dummy_req=dummy,
                                 slot_states={}, kv_cache=None))
    return proposer


def _from_scratch(draft, target: _ToyLoopTarget, seq: list[int], k: int, pick=None):
    """The draft for ``seq`` (its last token pending) from features recomputed over the committed tokens alone."""
    c = draft.config
    p = len(seq) - 1
    feats = target.run(torch.tensor(seq[:p]), torch.arange(p))
    ck, cv = draft.context_kv(feats, torch.arange(p, device=DEV))

    def context(layer):
        lo = max(0, p - c.window) if layer in c.local_layer_ids else 0
        valid = torch.zeros(1, p, dtype=torch.bool, device=DEV)
        valid[:, lo:] = True
        return [(ck[layer][None], cv[layer][None], valid)]

    anchor = torch.tensor([seq[-1]], device=DEV)
    hidden = draft.forward_block(anchor, torch.tensor([p], device=DEV), context)
    return draft.walk(hidden[:, 1 : k + 1], anchor, draft.f_rnn(feats[-1:]), pick or (lambda j, logits: logits.argmax(-1)))[0]


# the second case reads the full layer's context 7 positions at a time, so the softmax merges many chunks, some masked out for one request
@pytest.mark.parametrize("k, chunk_rows", [(7, 16384), (3, 14)])
def test_proposer_tracks_committed_tokens(tmp_path, monkeypatch, k, chunk_rows):
    """Chunked prefill, then verify steps with every accept count: the pool rows past the accepted ones and the Markov context's row must never show in the next drafts."""
    from freetoken.spec.dspark.proposer import DSparkProposer
    from freetoken.core import Batch, Req, SamplingParams

    monkeypatch.setattr(DSparkProposer, "CONTEXT_CHUNK_ROWS", chunk_rows)
    path, _, target_parts = _toy_checkpoint(tmp_path)
    H, page_size, num_pages, mr = TOY["hidden"], 4, 64, 2
    g = torch.Generator().manual_seed(5)
    # scattered pages: positions of one request land on unrelated slots
    pages = torch.randperm(num_pages, generator=g).view(mr, -1)
    cols = torch.arange(pages.shape[1] * page_size)
    page_table = torch.full((mr + 1, cols.numel()), num_pages * page_size, dtype=torch.int32)
    page_table[:mr] = (pages[:, cols // page_size] * page_size + cols % page_size).int()
    page_table = page_table.to(DEV)
    # fp32: the from-scratch draft runs other batch shapes, whose GEMMs round differently; bf16 would flip near ties
    proposer = _proposer(path, target_parts, k, page_table, num_pages, page_size, mr, torch.float32)
    toy = _ToyLoopTarget(TOY, H)
    greedy = SamplingParams()
    width = k + 1

    def forward(reqs, phase, tokens, positions, rows_per_req=None):
        batch = Batch(reqs=reqs, phase=phase, rows_per_req=rows_per_req)
        batch.padded_reqs = reqs
        batch.positions = positions.to(DEV)
        rows = torch.cat([torch.full((len(p),), r.table_idx) for r, p in zip(reqs, positions.split([len(x) for x in tokens]))])
        batch.out_loc = page_table[rows.to(DEV), batch.positions.long()]
        sink = proposer.feature_sink(batch)
        toy.run(torch.cat(tokens), positions, sink)
        return batch

    # request 0 prefills in two chunks past the window, request 1 in one chunk inside it
    seqs = [torch.randint(0, TOY["vocab"], (37,), generator=g).tolist(), torch.randint(0, TOY["vocab"], (9,), generator=g).tolist()]
    chunks = [[(0, 20), (20, 37)], [(0, 9)]]
    for step in range(2):
        reqs, toks, pos = [], [], []
        for row, seq in enumerate(seqs):
            if step < len(chunks[row]):
                s, e = chunks[row][step]
                reqs.append(Req(input_ids=torch.tensor(seq[:e], dtype=torch.int32), table_idx=row, cached_len=s, output_len=64, uid=row,
                                sampling_params=greedy, cache_handle=None))
                toks.append(torch.tensor(seq[s:e]))
                pos.append(torch.arange(s, e))
        batch = forward(reqs, "prefill", toks, torch.cat(pos))
        proposer.on_prefill(batch, None)
    for seq in seqs:
        seq.append(int(torch.randint(0, TOY["vocab"], (1,), generator=g)))

    seen = set()
    for step in range(10):
        reqs = [Req(input_ids=torch.tensor(seq, dtype=torch.int32), table_idx=row, cached_len=len(seq) - 1, output_len=64, uid=row,
                    sampling_params=greedy, cache_handle=None) for row, seq in enumerate(seqs)]
        rows = torch.tensor([0, 1], device=DEV)
        pending = torch.tensor([seq[-1] for seq in seqs], device=DEV)
        drafts = proposer.propose(reqs, rows, pending).draft_token_ids
        for row, seq in enumerate(seqs):
            steps = []
            want = _from_scratch(proposer.draft, toy, seq, k, pick=lambda j, logits: (steps.append(logits[0]), logits.argmax(-1))[1]).tolist()
            got = drafts[row].tolist()
            # the two may part only where the from-scratch step has a near tie, and nothing after it is comparable
            j = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
            assert j is None or steps[j][want[j]] - steps[j][got[j]] < 1e-4, f"step {step}, request {row}: drafts {got}, from scratch {want}"
        # the target takes n drafts, then a token of its own
        emitted = torch.tensor([(step + 3 * row) % width for row in range(2)])
        seen.update(emitted.tolist())
        toks = [torch.tensor([seq[-1]] + drafts[row].tolist()) for row, seq in enumerate(seqs)]
        pos = torch.cat([torch.arange(len(seq) - 1, len(seq) - 1 + width) for seq in seqs])
        batch = forward(reqs, "verify", toks, pos, rows_per_req=width)
        out = torch.cat([drafts.cpu(), drafts.cpu()[:, :1]], dim=1)
        proposer.post_verify(batch, out.to(DEV), emitted.to(DEV))
        for row, seq in enumerate(seqs):
            n = int(emitted[row])
            seq.extend(drafts[row, :n].tolist())
            seq.append((int(drafts[row, n]) + 1) % TOY["vocab"] if n < k else int(torch.randint(0, TOY["vocab"], (1,), generator=g)))
    assert seen == set(range(width))


def test_sampled_drafts_report_their_distribution(tmp_path):
    """A greedy and a sampled request in one batch: the greedy one drafts what it drafts alone, the sampled one reports the reference's tempered softmax over its step's top logits along the path it drew. fp32, so q compares tightly."""
    from freetoken.core import Req, SamplingParams

    path, w, target_parts = _toy_checkpoint(tmp_path)
    mr, page_size, num_pages, temperature = 2, 4, 32, 0.8
    page_table = torch.arange((mr + 1) * 40, dtype=torch.int32).view(mr + 1, 40).to(DEV)
    proposer = _proposer(path, target_parts, 7, page_table, num_pages, page_size, mr, torch.float32)
    ref = _Reference(w, target_parts, proposer.draft_config)
    toy = _ToyLoopTarget(TOY, TOY["hidden"])
    g = torch.Generator().manual_seed(9)
    seqs = [torch.randint(0, TOY["vocab"], (12,), generator=g).tolist() for _ in range(2)]
    # the pool as a prefill leaves it
    feats = [toy.run(torch.tensor(seq[:-1]), torch.arange(len(seq) - 1)) for seq in seqs]
    for row, (seq, f) in enumerate(zip(seqs, feats)):
        proposer._store_context(f, torch.arange(len(seq) - 1, device=DEV), page_table[row, : len(seq) - 1].long())
        proposer.f_rnn[row] = proposer.draft.f_rnn(f[-1:])[0]
    params = [SamplingParams(), SamplingParams(temperature=temperature)]
    reqs = [Req(input_ids=torch.tensor(seq, dtype=torch.int32), table_idx=row, cached_len=len(seq) - 1, output_len=8, uid=row,
                sampling_params=params[row], cache_handle=None) for row, seq in enumerate(seqs)]
    torch.manual_seed(0)
    proposal = proposer.propose(reqs, torch.tensor([0, 1], device=DEV), torch.tensor([s[-1] for s in seqs], device=DEV))
    assert proposal.draft_token_ids[0].tolist() == _from_scratch(proposer.draft, toy, seqs[0], 7).tolist()
    tokens, support, q = proposal.draft_token_ids[1], proposal.draft_support_ids[1], proposal.draft_support_probs[1]
    torch.testing.assert_close(q.sum(-1), torch.ones(7, device=DEV))
    hits = (support == tokens[:, None]).float()
    assert (hits.sum(-1) == 1).all() and ((q * hits).sum(-1) > 0).all()
    anchor = seqs[1][-1]
    logits = ref.path_logits(ref.block(feats[1].float(), anchor)[1:], feats[1][-1], anchor, tokens)
    top = logits.topk(proposer.SAMPLE_TOP_K, dim=-1).values
    for j in range(7):
        vals = logits[j].gather(0, support[j])
        # the support is the step's top K (up to a tie at its edge), and q their softmax at the request's temperature
        assert vals.min() >= top[j, -1] - 1e-3
        torch.testing.assert_close(q[j], torch.softmax(vals / temperature, -1), atol=1e-4, rtol=1e-3)
    # the drafts are drawn from that q: many copies of the sampled request, the first draft's frequencies against q at 4 sigma
    n = 4096
    many = proposer.propose([reqs[1]] * n, torch.ones(n, dtype=torch.int64, device=DEV), torch.full((n,), anchor, device=DEV))
    q0, ids0 = many.draft_support_probs[0, 0], many.draft_support_ids[0, 0]
    freq = (many.draft_token_ids[:, :1] == ids0[None]).float().mean(0)
    assert ((freq - q0).abs() <= 4 * (q0 * (1 - q0) / n).sqrt() + 1e-3).all(), (freq - q0).abs().max()
