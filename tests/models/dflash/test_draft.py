"""DFlash2DraftModel against an fp32 torch reference of the sglang DFlash 2 math, on a synthetic bf16 draft checkpoint: context rows into the per-layer K/V ring, the block forward (window mask, grouped convs, residual norms) and the selector walk over the target head's candidates."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV = torch.device("cuda")
H, LAYERS, HEADS, KV_HEADS, HEAD_DIM, INTER = 64, 2, 4, 2, 64, 128
GROUP_SIZE, TAPS, RANK, TOP_K, WINDOW, VOCAB, MASK = 16, (1, 2), 8, 4, 16, 64, 61
RING = WINDOW + 64
THETA, EPS = 10000.0, 1e-6
CONFIG = {
    "architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "is_causal": False, "dtype": "bfloat16",
    "dflash_config": {"block_size": 8, "conv_group_size": GROUP_SIZE, "conv_kernel_size": 2, "mask_token_id": MASK,
                      "selector_rank": RANK, "selector_top_k": TOP_K, "target_layer_ids": list(TAPS)},
    "head_dim": HEAD_DIM, "hidden_act": "silu", "hidden_size": H, "intermediate_size": INTER,
    "layer_types": ["sliding_attention"] * LAYERS, "max_position_embeddings": 4096, "num_attention_heads": HEADS,
    "num_hidden_layers": LAYERS, "num_key_value_heads": KV_HEADS, "num_target_layers": 4, "rms_norm_eps": EPS,
    "rope_parameters": {"rope_theta": THETA, "rope_type": "default"}, "sliding_window": WINDOW, "use_sliding_window": True,
    "tie_word_embeddings": False, "vocab_size": VOCAB,
}


def _weights() -> dict[str, torch.Tensor]:
    """A fan-in scaled bf16 draft plus the target's embedding and head (``target.*``, not part of the checkpoint)."""
    g = torch.Generator().manual_seed(0)

    def mat(*shape):
        return (torch.randn(*shape, generator=g) * shape[-1] ** -0.5).to(torch.bfloat16)

    def gain(n):
        return (1 + 0.1 * torch.randn(n, generator=g)).to(torch.bfloat16)

    w = {"fc.weight": mat(H, len(TAPS) * H), "hidden_norm.weight": gain(H), "norm.weight": gain(H),
         "candidate_selector.hidden_projection.weight": mat(RANK, H),
         "candidate_selector.predecessor_codebook": mat(VOCAB, RANK), "candidate_selector.successor_codebook": mat(VOCAB, RANK),
         "target.embed": mat(VOCAB, H), "target.head": mat(VOCAB, H)}
    for i in range(LAYERS):
        p = f"layers.{i}"
        w.update({f"{p}.self_attn.q_proj.weight": mat(HEADS * HEAD_DIM, H), f"{p}.self_attn.k_proj.weight": mat(KV_HEADS * HEAD_DIM, H),
                  f"{p}.self_attn.v_proj.weight": mat(KV_HEADS * HEAD_DIM, H), f"{p}.self_attn.o_proj.weight": mat(H, HEADS * HEAD_DIM),
                  f"{p}.self_attn.q_norm.weight": gain(HEAD_DIM), f"{p}.self_attn.k_norm.weight": gain(HEAD_DIM),
                  f"{p}.mlp.gate_proj.weight": mat(INTER, H), f"{p}.mlp.up_proj.weight": mat(INTER, H), f"{p}.mlp.down_proj.weight": mat(H, INTER),
                  f"{p}.input_layernorm.weight": gain(H), f"{p}.post_attention_layernorm.weight": gain(H)})
        for conv in ("attention_conv", "mlp_conv"):
            w[f"{p}.{conv}.base_kernel"] = (0.5 * torch.randn(2, 2, H, generator=g)).to(torch.bfloat16)
            w[f"{p}.{conv}.kernel_projection.weight"] = mat(2 * 2 * (H // GROUP_SIZE), H)
    return w


def _load_draft(tmp_path, w: dict[str, torch.Tensor], dtype=torch.bfloat16):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers import ParallelLMHead, VocabParallelEmbedding
    from freetoken.layers.quantization import finalize_quant
    from freetoken.models.dflash import DFlash2DraftModel, parse_config
    from freetoken.utils import cached_load_hf_config, torch_dtype

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    path = tmp_path / "draft"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(CONFIG))
    save_file({name: t for name, t in w.items() if not name.startswith("target.")}, str(path / "model.safetensors"))
    with torch.device("meta"), torch_dtype(dtype):
        embed, head = VocabParallelEmbedding(VOCAB, H), ParallelLMHead(VOCAB, H)
    for op, name in ((embed, "target.embed"), (head, "target.head")):
        op.load_state_dict({"weight": w[name].to(DEV, dtype)})
        finalize_quant(op)
    target = SimpleNamespace(model=SimpleNamespace(embed_tokens=embed), lm_head=head)
    return DFlash2DraftModel.load(parse_config(cached_load_hf_config(str(path))), target, str(path), DEV, dtype)


class _Reference:
    """The draft math in fp32 from the HF definition: pre-norm layers, per-head q/k norm, NeoX rope, grouped dynamic conv, sliding-window block attention."""

    def __init__(self, w: dict[str, torch.Tensor]) -> None:
        self.w = {name: t.to(DEV, torch.float32) for name, t in w.items()}

    @staticmethod
    def rmsnorm(x, weight):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * weight

    @staticmethod
    def rope(x, pos):
        """``x [n, heads, HEAD_DIM]`` at positions ``pos [n]``."""
        inv = 1.0 / (THETA ** (torch.arange(0, HEAD_DIM, 2, device=DEV, dtype=torch.float32) / HEAD_DIM))
        freqs = pos.float()[:, None] * inv[None, :]
        cos, sin = freqs.cos()[:, None], freqs.sin()[:, None]
        x1, x2 = x[..., : HEAD_DIM // 2], x[..., HEAD_DIM // 2 :]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)

    def context_rows(self, features):
        return self.rmsnorm(features.float() @ self.w["fc.weight"].T, self.w["hidden_norm.weight"])

    def kv(self, layer, ctx, pos):
        """K (normed, roped) and V of context rows ``ctx`` at ``pos``, ``[n, KV_HEADS, HEAD_DIM]`` each."""
        p = f"layers.{layer}.self_attn"
        k = self.rmsnorm((ctx @ self.w[f"{p}.k_proj.weight"].T).view(-1, KV_HEADS, HEAD_DIM), self.w[f"{p}.k_norm.weight"])
        return self.rope(k, pos), (ctx @ self.w[f"{p}.v_proj.weight"].T).view(-1, KV_HEADS, HEAD_DIM)

    def conv(self, x, prefix, side, coef=None):
        """``out[j] = sum_t (base[side, t] + coef[j, side, t, group]) * x[j - t]`` over one block of rows; the first row has no previous row."""
        if coef is None:
            coef = (x @ self.w[f"{prefix}.kernel_projection.weight"].T).view(x.shape[0], 2, 2, H // GROUP_SIZE)
        base = self.w[f"{prefix}.base_kernel"][side]
        c = base.view(1, 2, H // GROUP_SIZE, GROUP_SIZE) + coef[:, side].unsqueeze(-1)
        blocks = x.view(x.shape[0], H // GROUP_SIZE, GROUP_SIZE)
        prev = F.pad(blocks[:-1], (0, 0, 0, 0, 1, 0))
        return (c[:, 0] * blocks + c[:, 1] * prev).flatten(-2), coef

    def block(self, bonus, start, ctx, block):
        """Final hidden ``[block, H]`` of ``[bonus, mask x (block-1)]`` at ``start``, over the context rows ``ctx`` at positions ``0 .. len(ctx)-1``."""
        w = self.w
        x = torch.cat([w["target.embed"][bonus][None], w["target.embed"][MASK][None].expand(block - 1, -1)])
        pos = start + torch.arange(block, device=DEV)
        hist = torch.arange(ctx.shape[0], device=DEV)
        residual = None
        for i in range(LAYERS):
            p = f"layers.{i}"
            residual = x if residual is None else residual + x
            h = self.rmsnorm(residual, w[f"{p}.input_layernorm.weight"])
            h, coef = self.conv(h, f"{p}.attention_conv", 0)
            q = self.rope(self.rmsnorm((h @ w[f"{p}.self_attn.q_proj.weight"].T).view(block, HEADS, HEAD_DIM), w[f"{p}.self_attn.q_norm.weight"]), pos)
            k_hist, v_hist = self.kv(i, ctx, hist)
            k_block, v_block = self.kv(i, h, pos)
            k = torch.cat([k_hist, k_block]).repeat_interleave(HEADS // KV_HEADS, dim=1)
            v = torch.cat([v_hist, v_block]).repeat_interleave(HEADS // KV_HEADS, dim=1)
            key_pos = torch.cat([hist, pos])
            # a query at start+j sees the window of positions > start+j-WINDOW and the whole block
            keep = (key_pos[None, :] >= (pos - WINDOW + 1)[:, None]) | (key_pos >= start)[None, :]
            scores = torch.einsum("jhd,nhd->jhn", q, k) * HEAD_DIM**-0.5
            probs = torch.softmax(scores.masked_fill(~keep[:, None, :], float("-inf")), dim=-1)
            o = torch.einsum("jhn,nhd->jhd", probs, v).reshape(block, -1) @ w[f"{p}.self_attn.o_proj.weight"].T
            o, _ = self.conv(o, f"{p}.attention_conv", 1, coef)
            residual = residual + o
            h = self.rmsnorm(residual, w[f"{p}.post_attention_layernorm.weight"])
            h, coef = self.conv(h, f"{p}.mlp_conv", 0)
            m = (F.silu(h @ w[f"{p}.mlp.gate_proj.weight"].T) * (h @ w[f"{p}.mlp.up_proj.weight"].T)) @ w[f"{p}.mlp.down_proj.weight"].T
            x, _ = self.conv(m, f"{p}.mlp_conv", 1, coef)
        return self.rmsnorm(residual + x, w["norm.weight"])

    def walk(self, hidden, cand, anchor, temperature, greedy, uniforms):
        """One path over the given candidates of slots 1..: ``unary + <pre[prev] * state, suc[cand]>`` per slot, argmax or a draw with ``uniforms``; ``cand`` must be a best-first top-K of the head within the real head's bf16 noise."""
        w = self.w
        bs = anchor.shape[0]
        pred = hidden.float().view(bs, -1, H)[:, 1:]
        logits = pred @ w["target.head"].T
        unary = logits.gather(-1, cand)
        noise = 2**-7 * logits.abs().max()
        runner_up = logits.topk(TOP_K + 1, dim=-1).values[..., -1:]
        assert (unary >= runner_up - noise).all() and (unary[..., :-1] >= unary[..., 1:] - noise).all()
        state = pred @ w["candidate_selector.hidden_projection.weight"].T
        prev, tokens, qs = anchor, [], []
        for s in range(pred.shape[1]):
            scale = w["candidate_selector.predecessor_codebook"][prev] * state[:, s]
            scores = unary[:, s] + (w["candidate_selector.successor_codebook"][cand[:, s]] * scale[:, None]).sum(-1)
            probs = torch.softmax(scores / temperature[:, None], -1)
            top2 = scores.topk(2, dim=-1).values
            # the argmax and the draw must be clear of that noise too, or the comparison means nothing
            assert (top2[:, 0] - top2[:, 1])[greedy].min() > 5e-2
            assert (uniforms[:, s : s + 1] - probs.cumsum(-1)).abs()[~greedy].min() > 2e-2
            index = torch.where(greedy, scores.argmax(-1), (uniforms[:, s : s + 1] >= probs.cumsum(-1)).sum(-1).clamp(max=TOP_K - 1))
            qs.append(torch.where(greedy[:, None], F.one_hot(index, TOP_K).float(), probs))
            prev = cand[:, s].gather(1, index[:, None])[:, 0]
            tokens.append(prev)
        return torch.stack(tokens, 1), torch.stack(qs, 1)


@pytest.mark.parametrize("dtype, tol", [(torch.float32, 2**-6), (torch.bfloat16, 2**-2)])
@pytest.mark.parametrize("block", [4, 8])
def test_block_forward_matches_reference(tmp_path, block, dtype, tol):
    """Two requests in one block forward: one with a context shorter than the window (positions before 0 masked), one longer than the ring (wrapped slots). The fp32 draft pins the math (tf32 dots); the bf16 draft, the serving dtype, is held to bf16 noise. A ring rebuilt from only the window rows gives the same hidden, bitwise."""
    w = _weights()
    draft, ref = _load_draft(tmp_path, w, dtype), _Reference(w)
    dflash_ring = draft.make_dflash_ring(max_rows=3, block=block, ring_len=RING, max_position=4096, device=DEV)
    torch.manual_seed(1)
    lengths, rows = (5, 100), (2, 1)
    contexts, draft_contexts = [], []
    for n, row in zip(lengths, rows):
        features = torch.randn(n, len(TAPS) * H, device=DEV, dtype=torch.bfloat16).to(dtype)
        ctx = draft.context_rows(features)
        torch.testing.assert_close(ctx.float(), ref.context_rows(features), atol=0, rtol=2**-6)
        draft_contexts.append(ctx)
        # as the proposer stores a chunk: its last RING rows at most
        tail = torch.arange(max(0, n - RING), n, device=DEV)
        draft.store_context(ctx[tail], tail, torch.full((len(tail),), row, device=DEV), dflash_ring)
        contexts.append(ref.context_rows(features))
        for i in range(LAYERS):
            k, v = ref.kv(i, contexts[-1][tail], tail)
            torch.testing.assert_close(dflash_ring.k[i, row, tail % RING].float(), k, atol=2**-6, rtol=2**-6)
            torch.testing.assert_close(dflash_ring.v[i, row, tail % RING].float(), v, atol=2**-6, rtol=2**-6)
    bonus = torch.tensor([3, 40], device=DEV)
    starts, rows_t = torch.tensor(lengths, device=DEV), torch.tensor(rows, device=DEV)

    hidden = draft.forward_block(bonus, starts, rows_t, dflash_ring)

    assert hidden.shape == (2 * block, H) and hidden.dtype == dtype
    for b in range(2):
        want = ref.block(int(bonus[b]), lengths[b], contexts[b], block)
        torch.testing.assert_close(hidden[b * block : (b + 1) * block].float(), want, atol=tol, rtol=tol)

    # a restore from the window loses nothing, and the block reads nothing older
    dflash_ring.k.zero_()
    dflash_ring.v.zero_()
    for n, row, ctx in zip(lengths, rows, draft_contexts):
        window = torch.arange(max(0, n - WINDOW + 1), n, device=DEV)
        draft.store_context(ctx[window], window, torch.full((len(window),), row, device=DEV), dflash_ring)
    assert torch.equal(draft.forward_block(bonus, starts, rows_t, dflash_ring), hidden)


def test_sample_path_walks_the_selector_lattice(tmp_path):
    """Greedy and sampled requests follow the fp32 walk (same uniforms); a request whose scores all tie takes the first candidate; a NaN hidden yields in-range ids and a q that is no distribution, without disturbing the other requests."""
    w = _weights()
    draft, ref = _load_draft(tmp_path, w), _Reference(w)
    block, bs = 4, 4
    torch.manual_seed(3)
    hidden = torch.randn(bs * block, H, device=DEV, dtype=torch.bfloat16)
    hidden.view(bs, block, H)[2] = 0
    hidden.view(bs, block, H)[3] = float("nan")
    anchor = torch.tensor([5, 17, 30, 2], device=DEV)
    temperature = torch.tensor([1.0, 0.7, 1.0, 0.7], device=DEV)
    greedy = torch.tensor([True, False, True, False], device=DEV)

    pred = hidden.view(bs, block, H)[:, 1:]
    cand, unary = draft.compute_candidates(pred)
    uniforms = torch.rand(bs, block - 1, device=DEV)
    tokens, q = draft.candidate_selector.sample_path(cand, unary, pred, anchor, temperature, greedy, uniforms)
    want_tokens, want_q = ref.walk(hidden[: 2 * block], cand[:2], anchor[:2], temperature[:2], greedy[:2], uniforms[:2])

    assert tokens.shape == (bs, block - 1) and cand.shape == q.shape == (bs, block - 1, TOP_K)
    assert torch.equal(tokens[:2], want_tokens)
    torch.testing.assert_close(q[:2], want_q, atol=1e-2, rtol=0)
    assert torch.equal(tokens[2], cand[2, :, 0]) and torch.equal(q[2], F.one_hot(torch.zeros(block - 1, dtype=torch.int64, device=DEV), TOP_K).float())
    assert ((tokens >= 0) & (tokens < VOCAB)).all() and ((cand >= 0) & (cand < VOCAB)).all()
    assert not torch.isclose(q[3].sum(-1), torch.ones(block - 1, device=DEV)).any()
