"""DFlash2DraftModel against an fp32 torch reference of the sglang DFlash 2 math, on a synthetic bf16 draft checkpoint (safetensors, a float GGUF, or a Q8_0/Q4_0 GGUF read through the ggml kernels): context rows into the per-layer K/V ring, the block forward (window mask, grouped convs, residual norms) and the selector walk over the target head's candidates."""

from __future__ import annotations

import importlib.util
import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

# tests/ is not a package, so a plain `pytest` run cannot import a sibling test by name
_spec = importlib.util.spec_from_file_location(
    "_qwen_gguf_writer", os.path.join(os.path.dirname(__file__), "..", "test_qwen3_5_moe_weight.py"))
_writer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_writer)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV = torch.device("cuda")
H, LAYERS, HEADS, KV_HEADS, HEAD_DIM, INTER = 64, 2, 4, 2, 64, 128
GROUP_SIZE, TAPS, RANK, TOP_K, WINDOW, VOCAB, MASK = 16, (1, 2), 8, 4, 16, 64, 61
# Q8_0 blocks are 32 values, so only the quantized fixture uses a rank its codebook rows divide
Q8_RANK = 32
# the q8 fixture's own draw: its wider rank already moves the stream, and this seed keeps the fixed bf16 tolerances
Q8_SEED = 4
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


def _weights(rank: int = RANK, seed: int = 0) -> dict[str, torch.Tensor]:
    """A fan-in scaled bf16 draft plus the target's embedding and head (``target.*``, not part of the checkpoint)."""
    g = torch.Generator().manual_seed(seed)

    def mat(*shape):
        return (torch.randn(*shape, generator=g) * shape[-1] ** -0.5).to(torch.bfloat16)

    def gain(n):
        return (1 + 0.1 * torch.randn(n, generator=g)).to(torch.bfloat16)

    w = {"fc.weight": mat(H, len(TAPS) * H), "hidden_norm.weight": gain(H), "norm.weight": gain(H),
         "candidate_selector.hidden_projection.weight": mat(rank, H),
         "candidate_selector.predecessor_codebook": mat(VOCAB, rank), "candidate_selector.successor_codebook": mat(VOCAB, rank),
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


def _draft_tensor_names() -> dict[str, str]:
    """GGUF tensor name -> the state-dict name the tiny draft stores it under (the real file's names)."""
    names = {
        "fc.weight": "fc.weight", "enc.output_norm.weight": "hidden_norm.weight",
        "output_norm.weight": "norm.weight",
        "selector_hidden.weight": "candidate_selector.hidden_projection.weight",
        "selector_predecessor.weight": "candidate_selector.predecessor_codebook",
        "selector_successor.weight": "candidate_selector.successor_codebook",
    }
    for i in range(LAYERS):
        p, b = f"layers.{i}", f"blk.{i}"
        names.update({
            f"{b}.attn_norm.weight": f"{p}.input_layernorm.weight",
            f"{b}.ffn_norm.weight": f"{p}.post_attention_layernorm.weight",
            f"{b}.attn_q.weight": f"{p}.self_attn.q_proj.weight",
            f"{b}.attn_k.weight": f"{p}.self_attn.k_proj.weight",
            f"{b}.attn_v.weight": f"{p}.self_attn.v_proj.weight",
            f"{b}.attn_output.weight": f"{p}.self_attn.o_proj.weight",
            f"{b}.attn_q_norm.weight": f"{p}.self_attn.q_norm.weight",
            f"{b}.attn_k_norm.weight": f"{p}.self_attn.k_norm.weight",
            f"{b}.attn_conv_proj.weight": f"{p}.attention_conv.kernel_projection.weight",
            f"{b}.attn_conv_base": f"{p}.attention_conv.base_kernel",
            f"{b}.ffn_conv_proj.weight": f"{p}.mlp_conv.kernel_projection.weight",
            f"{b}.ffn_conv_base": f"{p}.mlp_conv.base_kernel",
            f"{b}.ffn_gate.weight": f"{p}.mlp.gate_proj.weight",
            f"{b}.ffn_up.weight": f"{p}.mlp.up_proj.weight",
            f"{b}.ffn_down.weight": f"{p}.mlp.down_proj.weight",
        })
    return names


_DRAFT_TENSORS = _draft_tensor_names()


def _tiny_gguf(directory, plain: dict[str, torch.Tensor], *, q8: bool, fc_q8: bool = False, rope_sections=None):
    """Write the tiny draft as a GGUF under the real file's names; returns its path and the tensor-name map.

    ``q8`` quantizes every matrix (the codebooks too) and leaves norms and conv bases F32. The context
    projection stays BF16 unless ``fc_q8``, so the block test's exact context-row assertion compares like with
    like while the oracle test can run it through ggml. Only the q and gate parts are Q4_0 (k, v and up stay
    Q8_0), so those fused modules split into ``weight`` / ``weight_1``."""
    directory.mkdir(parents=True, exist_ok=True)
    model_path = str(directory / "draft.gguf")
    tensors = {gguf: plain[hf] for gguf, hf in _DRAFT_TENSORS.items()}
    rank = plain["candidate_selector.predecessor_codebook"].shape[1]
    if q8:
        types = {name: _writer.F32 if name.endswith(("norm.weight", "conv_base")) else _writer.Q8_0 for name in tensors}
        if not fc_q8:
            types["fc.weight"] = _writer.BF16
        types["blk.0.attn_q.weight"] = _writer.Q4_0
        types["blk.1.ffn_gate.weight"] = _writer.Q4_0
    else:
        types = {name: _writer.F32 if name.endswith(("norm.weight", "conv_base")) else _writer.BF16 for name in tensors}
    metadata = {
        "block_count": LAYERS, "embedding_length": H, "feed_forward_length": INTER,
        "attention.head_count": HEADS, "attention.head_count_kv": KV_HEADS,
        "attention.causal": False, "attention.key_length": HEAD_DIM, "attention.value_length": HEAD_DIM,
        "attention.sliding_window": WINDOW, "attention.sliding_window_pattern": [True] * LAYERS,
        "attention.layer_norm_rms_epsilon": EPS, "rope.freq_base": THETA,
        "block_size": 8, "conv_kernel_size": 2, "conv_group_size": GROUP_SIZE,
        "selector_rank": rank, "selector_top_k": TOP_K,
        # non-monotonic 1-based, so a sort or a dropped -1 fails the parsed-order assertion below
        "target_layers": [3, 2], "tokenizer.ggml.tokens": [f"t{i}" for i in range(VOCAB)],
        "tokenizer.ggml.mask_token_id": MASK,
    }
    if rope_sections is not None:
        metadata["rope.dimension_sections"] = rope_sections
    _writer._write_gguf(directory / "draft.gguf", "dflash", metadata, {
        name: (types[name], tuple(reversed(t.shape)), _writer._gguf_bytes(t, types[name])) for name, t in tensors.items()})
    return model_path, _DRAFT_TENSORS


def _quantized_weights(directory, w: dict[str, torch.Tensor], *, fc_q8: bool = False) -> dict[str, torch.Tensor]:
    """The tiny Q8_0 GGUF's own values as a float state dict, for an oracle that holds no pre-quantization weights."""
    from freetoken.models.gguf.reader import gguf_tensor_index
    from freetoken.models.gguf.tensors import dequantized_tensor

    path, names = _tiny_gguf(directory, {n: t for n, t in w.items() if not n.startswith("target.")}, q8=True, fc_q8=fc_q8)
    index = gguf_tensor_index(path)
    out = {hf: dequantized_tensor(index[gguf]) for gguf, hf in names.items()}
    out.update({n: t for n, t in w.items() if n.startswith("target.")})
    return out


def _load_draft(tmp_path, w: dict[str, torch.Tensor], dtype=torch.bfloat16, source="safetensors"):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers import ParallelLMHead, VocabParallelEmbedding
    from freetoken.layers.quantization import finalize_quant
    from freetoken.models.dflash import DFlash2DraftModel, parse_config, parse_gguf_config
    from freetoken.utils import cached_load_hf_config, torch_dtype

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    path = tmp_path / "draft"
    path.mkdir(parents=True)
    plain = {name: t for name, t in w.items() if not name.startswith("target.")}
    rank = plain["candidate_selector.predecessor_codebook"].shape[1]
    if source in ("gguf", "gguf-q8", "gguf-q8-fc"):
        model_path, _names = _tiny_gguf(path, plain, q8=source != "gguf", fc_q8=source == "gguf-q8-fc")
        config = parse_gguf_config(model_path)
        assert config.target_layer_ids == (2, 1) and config.mask_token_id == MASK
        assert config.num_target_layers is None and config.vocab_size == VOCAB
    else:
        model_path = str(path)
        config_json = {**CONFIG, "dflash_config": {**CONFIG["dflash_config"], "selector_rank": rank}}
        (path / "config.json").write_text(json.dumps(config_json))
        save_file(plain, str(path / "model.safetensors"))
        config = parse_config(cached_load_hf_config(model_path))
    with torch.device("meta"), torch_dtype(dtype):
        embed, head = VocabParallelEmbedding(VOCAB, H), ParallelLMHead(VOCAB, H)
    for op, name in ((embed, "target.embed"), (head, "target.head")):
        op.load_state_dict({"weight": w[name].to(DEV, dtype)})
        finalize_quant(op)
    target = SimpleNamespace(model=SimpleNamespace(embed_tokens=embed), lm_head=head)
    return DFlash2DraftModel.load(config, target, model_path, DEV, dtype)


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


@pytest.mark.parametrize("sections, accepted", [
    # llama.cpp records an M-RoPE target's text-only layout this way; for text it is plain NeoX rope
    ([HEAD_DIM // 2, 0, 0, 0], True),
    # a split into nonzero parts needs the M-RoPE path the draft does not have
    ([HEAD_DIM // 4, HEAD_DIM // 4, 0, 0], False),
])
def test_gguf_rope_dimension_sections(tmp_path, sections, accepted):
    """The draft's GGUF accepts the degenerate sections llama.cpp writes for an M-RoPE target's text, and still refuses a genuinely split head."""
    from freetoken.models.dflash import parse_gguf_config

    plain = {name: t for name, t in _weights().items() if not name.startswith("target.")}
    model_path, _ = _tiny_gguf(tmp_path, plain, q8=False, rope_sections=sections)
    if accepted:
        assert parse_gguf_config(model_path).head_dim == HEAD_DIM
    else:
        with pytest.raises(ValueError, match="rope dimension sections"):
            parse_gguf_config(model_path)


@pytest.mark.parametrize("dtype, source, tol", [
    (torch.float32, "safetensors", 2**-6),
    (torch.bfloat16, "safetensors", 2**-2),
    (torch.bfloat16, "gguf", 2**-2),
    (torch.bfloat16, "gguf-q8", 2**-2),
])
@pytest.mark.parametrize("block", [4, 8])
def test_block_forward_matches_reference(tmp_path, block, dtype, source, tol):
    """Two requests in one block forward: one with a context shorter than the window (positions before 0 masked), one longer than the ring (wrapped slots). The fp32 safetensors draft pins the math (tf32 dots); every bf16 draft, the serving dtype, is held to bf16 noise. A ring rebuilt from only the window rows gives the same hidden, bitwise. The GGUF fixtures repeat the run from the real file's tensor names, so a wrong leaf, k/v order, gate/up order or conv-base orientation cannot hide behind the safetensors reader; the Q8_0 one reads its layer matrices through the ggml kernels against the file's own dequantized values."""
    w = _weights(Q8_RANK, Q8_SEED) if source == "gguf-q8" else _weights()
    ref_w = _quantized_weights(tmp_path / "oracle", w) if source == "gguf-q8" else w
    draft, ref = _load_draft(tmp_path, w, dtype, source), _Reference(ref_w)
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


@pytest.mark.parametrize("source", ["safetensors", "gguf", "gguf-q8"])
def test_sample_path_walks_the_selector_lattice(tmp_path, source):
    """Greedy and sampled requests follow the fp32 walk (same uniforms); a request whose scores all tie takes the first candidate; a NaN hidden yields in-range ids and a q that is no distribution, without disturbing the other requests. Every checkpoint form is walked, so a predecessor/successor swap in the reader changes the path."""
    w = _weights(Q8_RANK, Q8_SEED) if source == "gguf-q8" else _weights()
    ref_w = _quantized_weights(tmp_path / "oracle", w) if source == "gguf-q8" else w
    draft, ref = _load_draft(tmp_path, w, source=source), _Reference(ref_w)
    block, bs = 4, 4
    # the q8 fixture's wider rank moves the walk's CDF margins, so it draws with its own seed
    torch.manual_seed(0 if source == "gguf-q8" else 3)
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


@pytest.mark.parametrize("fc_q8", [False, True])
def test_a_quantized_gguf_load_matches_the_dequantized_float_draft(tmp_path, fc_q8):
    """The Q8_0/Q4_0 draft against a safetensors draft holding the GGUF's own dequantized tensors, an oracle that never sees the pre-quantization weights: the K/V rows and codebooks must match bitwise, and one block forward must stay within 15% L2, the 8-bit activation blocks (1/254 of each block's max) rounding through the draft's layers. With ``fc_q8`` the context projection is Q8_0 too, so its ggml kernel runs against the oracle's float fc."""
    source = "gguf-q8-fc" if fc_q8 else "gguf-q8"
    w = _weights(Q8_RANK, Q8_SEED)
    quantized = _load_draft(tmp_path, w, torch.bfloat16, source)
    floated = _load_draft(tmp_path / "float", _quantized_weights(tmp_path / "oracle", w, fc_q8=fc_q8), torch.bfloat16, "safetensors")
    assert torch.equal(quantized._kv_all, floated._kv_all)
    for name in ("predecessor_codebook", "successor_codebook"):
        assert torch.equal(getattr(quantized.candidate_selector, name), getattr(floated.candidate_selector, name))
    torch.manual_seed(5)
    features = torch.randn(1, len(TAPS) * H, device=DEV, dtype=torch.bfloat16)
    zero = torch.zeros(1, dtype=torch.int64, device=DEV)

    def run(draft):
        dflash_ring = draft.make_dflash_ring(max_rows=1, block=1, ring_len=RING, max_position=4096, device=DEV)
        draft.store_context(draft.context_rows(features), zero, zero, dflash_ring)
        return draft.forward_block(torch.tensor([3], device=DEV), torch.tensor([1], device=DEV), zero, dflash_ring)

    got, want = run(quantized).float(), run(floated).float()
    assert (got - want).norm() <= 0.15 * want.norm(), ((got - want).norm() / want.norm()).item()
