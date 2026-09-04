"""Skeleton tests: the frozen qwen4_exp interfaces and their torch references.

Everything runs on a scaled-down copy of the real geometry (4 layers, full attention on layer 3,
PLE on layer 1, hc_count 4, 3-gram hash) so the shapes and the layer split are the shipping ones.
The hyper-connection and PLE references transcribed here are HF ``modeling_qwen4_exp.py``
(``Qwen4ExpTextGatedResidual``:941, ``Qwen4ExpTextNGramEmbedding``:1018, ``Qwen4ExpTextPLELayer``
:1117), so the torch implementations are checked against the math, not against themselves.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from freetoken.layers import BaseOP, LinearReplicated
from freetoken.models.config import ModelConfig
from freetoken.models.qwen4_exp.config import parse_config
from freetoken.models.qwen4_exp.hc import GatedResidual
from freetoken.models.qwen4_exp.ple import GpuResidentTable, PLELayer, PLEMetadata

from .common import EOS, hash_constants, requires_cuda, toy_hf_config


def _config(num_layers: int = 4) -> ModelConfig:
    return parse_config(toy_hf_config(num_layers))


def _fill(op, gen: torch.Generator, scale: float = 0.05) -> None:
    """Random floats / zeroed ints for every state-dict tensor of an op tree."""
    for tensor in op.state_dict().values():
        if tensor.is_floating_point():
            tensor.normal_(0.0, scale, generator=gen)
        else:
            tensor.zero_()


def _group_norm(x, weight, eps, groups):
    xf = x.float().reshape(*x.shape[:-1], groups, -1)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf.flatten(-2) * (1.0 + weight.float())).type_as(x)


# --------------------------------------------------------------------------------------
# hyper-connections
# --------------------------------------------------------------------------------------


def _hf_gated_residual(hc, R, w_norm, w_down, w_up, w_inject, hidden, eps):
    xn = _group_norm(R, w_norm, eps, hc)
    mix = F.silu(F.linear(xn, w_down) / hc)
    mix = torch.sigmoid(F.linear(mix, w_up)).unflatten(-1, (hc, hidden))
    mixed = (mix * xn.unflatten(-1, (hc, hidden))).mean(-2)
    if w_inject is None:
        return mixed, None
    return mixed, 2 * torch.sigmoid(F.linear(xn, w_inject) / hc)


@pytest.mark.parametrize("tokens", [1, 7])
def test_hc_mix_and_combine_match_hf(tokens: int):
    torch.manual_seed(0)
    config = _config()
    args = config.qwen4_args
    hc = GatedResidual(config)
    _fill(hc, torch.Generator().manual_seed(1))

    R = torch.randn(tokens, args.ple_state_width)
    y = torch.randn(tokens, args.hidden_size)
    x, s = hc.mix(R)
    got = hc.combine(R, y, s)

    merged = hc.input_mix_weight_down_block_inject.weight
    ref_x, ref_inject = _hf_gated_residual(
        args.hc_count,
        R,
        hc.hc_norm.weight,
        merged[: args.hc_lowrank],
        hc.input_mix_weight_up.weight,
        merged[args.hc_lowrank : args.hc_lowrank + args.hc_count],
        args.hidden_size,
        config.rms_norm_eps,
    )
    ref = R.unflatten(-1, (args.hc_count, args.hidden_size))
    ref = (ref + y.unsqueeze(-2) * ref_inject.unsqueeze(-1)).flatten(-2)

    assert torch.allclose(x, ref_x, rtol=1e-5, atol=1e-6)
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_hc_merged_gemm_layout_and_top_level_mixer():
    config = _config()
    args = config.qwen4_args
    hc = GatedResidual(config)
    # 320 + 4 lowrank/inject rows padded to a multiple of 16 in the real config
    assert hc.pad_size == (-(args.hc_lowrank + args.hc_count)) % 16
    merged = hc.input_mix_weight_down_block_inject.weight
    assert merged.shape == (
        args.hc_lowrank + args.hc_count + hc.pad_size,
        args.ple_state_width,
    )
    assert set(hc.state_dict()) == {
        "hc_norm.weight",
        "input_mix_weight_down_block_inject.weight",
        "input_mix_weight_up.weight",
    }

    mixer = GatedResidual(config, use_combine=False)
    _fill(mixer, torch.Generator().manual_seed(2))
    assert set(mixer.state_dict()) == {
        "hc_norm.weight",
        "input_mix_weight_down.weight",
        "input_mix_weight_up.weight",
    }
    R = torch.randn(5, args.ple_state_width)
    x, s = mixer.mix(R)
    assert s is None
    ref_x, ref_inject = _hf_gated_residual(
        args.hc_count,
        R,
        mixer.hc_norm.weight,
        mixer.input_mix_weight_down.weight,
        mixer.input_mix_weight_up.weight,
        None,
        args.hidden_size,
        config.rms_norm_eps,
    )
    assert ref_inject is None
    assert torch.allclose(x, ref_x, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------------------
# PLE
# --------------------------------------------------------------------------------------


def _hf_shift_right(tokens, shift, eos):
    if shift == 0:
        return tokens
    batch, seq_len = tokens.shape
    positions = torch.arange(seq_len)
    eos_positions = torch.where(tokens == eos, positions, torch.full_like(positions, -1))
    previous = torch.cummax(eos_positions, dim=1).values
    previous = torch.cat([eos_positions.new_full((batch, 1), -1), previous[:, :-1]], dim=1)
    in_segment = positions.unsqueeze(0) - (previous + 1)
    source = positions - shift
    shifted = tokens.gather(1, source.clamp_min(0).unsqueeze(0).expand(batch, -1))
    valid = (in_segment >= shift) & (source.unsqueeze(0) >= 0)
    return torch.where(valid, shifted, tokens.new_full((), eos))


def _hf_ngram_ids(tokens, context, args, multipliers, sizes, offsets):
    """HF Qwen4ExpTextNGramEmbedding id computation over a dense [B, L] batch."""
    history = torch.cat([context, tokens], dim=-1)
    shifted = [_hf_shift_right(history, s, args.ngram_boundary_token_id) for s in range(args.ngram_size)]
    blocks = []
    for ngram in range(2, args.ngram_size + 1):
        start = (ngram - 2) * args.heads_per_ngram
        end = start + args.heads_per_ngram
        mixed = shifted[0] * multipliers[0]
        for position in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted[position] * multipliers[position])
        ids = torch.remainder(mixed.unsqueeze(-1), sizes[start:end].view(1, 1, -1))
        blocks.append(ids + offsets[start:end].view(1, 1, -1))
    return torch.cat(blocks, dim=-1)[:, -tokens.shape[1] :]


def _ragged(sequences, contexts, args, device="cpu"):
    """Ragged PLEMetadata (prefill) for a list of per-request token lists."""
    lens = [len(s) for s in sequences]
    cu = torch.tensor([0, *lens], dtype=torch.int64).cumsum(0)
    return PLEMetadata(
        input_ids=torch.tensor([t for s in sequences for t in s], dtype=torch.int64, device=device),
        cu_seqlens=cu.to(device),
        seq_lens=tuple(lens),
        ngram_context=torch.tensor(contexts, dtype=torch.int64, device=device),
        state_slots=torch.arange(len(sequences), dtype=torch.int64, device=device),
        fresh_slots=None,
        is_decode=False,
    )


def _make_ple(config, seed: int = 3, rows: int = 4096):
    args = config.qwen4_args
    gen = torch.Generator().manual_seed(seed)
    layer = PLELayer(config, args.ple_layer_ids[0])
    _fill(layer, gen)
    multipliers, sizes, offsets = hash_constants(args)
    layer.ple_embedding.layer_multipliers.copy_(multipliers)
    layer.ple_embedding.ngram_heads_vocab_sizes.copy_(sizes)
    layer.ple_embedding.ngram_heads_offsets.copy_(offsets)
    table = torch.randn(rows, args.ngram_head_dim, generator=gen) * 0.05
    layer.ple_embedding.attach_table(GpuResidentTable(table, dtype=torch.float32))
    return layer, (multipliers, sizes, offsets)


def test_ple_hash_matches_hf():
    """Ragged hash ids vs the HF dense reference, including eos inside and at the start of a request."""
    config = _config()
    args = config.qwen4_args
    layer, (multipliers, sizes, offsets) = _make_ple(config)

    sequences = [[3, 4, EOS, 5, 6], [EOS, 11, 12], [9]]
    contexts = [[EOS, EOS], [21, 22], [EOS, 31]]
    meta = _ragged(sequences, contexts, args)
    got = layer.ple_embedding.row_ids(meta)

    offset = 0
    for tokens, context in zip(sequences, contexts):
        ref = _hf_ngram_ids(
            torch.tensor([tokens]),
            torch.tensor([context]),
            args,
            multipliers,
            sizes,
            offsets,
        )[0]
        assert torch.equal(got[offset : offset + len(tokens)], ref)
        offset += len(tokens)
    # sequences[0][3] sits right after the boundary token, so its window is cut to eos padding
    after_eos = layer.ple_embedding.row_ids(_ragged([[5]], [[EOS, EOS]], args))[0]
    with_history = layer.ple_embedding.row_ids(_ragged([[5]], [[3, 4]], args))[0]
    assert torch.equal(got[3], after_eos)
    assert not torch.equal(got[3], with_history)


def test_ple_forward_matches_hf():
    """Full PLE forward (fp32) against the HF gate/norm chain and an explicit conv tap sum."""
    torch.manual_seed(4)
    config = _config()
    args = config.qwen4_args
    layer, (multipliers, sizes, offsets) = _make_ple(config)
    hidden, hc = args.hidden_size, args.hc_count

    sequences = [[3, 4, EOS, 5, 6, 8], [2, EOS, 11, 12, 13, 14]]
    contexts = [[EOS, EOS], [21, 22]]
    meta = _ragged(sequences, contexts, args)
    total = sum(len(s) for s in sequences)
    R = torch.randn(total, args.ple_state_width)
    states = torch.randn(len(sequences), args.ple_state_width, args.ple_conv_state_len) * 0.1
    got = layer.forward(R, batch=None, meta=meta, conv_states=states.clone())

    offset = 0
    for i, (tokens, context) in enumerate(zip(sequences, contexts)):
        ids = _hf_ngram_ids(
            torch.tensor([tokens]), torch.tensor([context]), args, multipliers, sizes, offsets
        )[0]
        embed = layer.ple_embedding.table.weight[ids.reshape(-1)].view(len(tokens), -1)
        key = _group_norm(
            F.linear(embed, layer.key_proj.weight), layer.norm_key.weight, config.rms_norm_eps, hc
        ).unflatten(-1, (hc, hidden))
        value = F.linear(embed, layer.value_proj.weight)
        rows = R[offset : offset + len(tokens)]
        query = _group_norm(
            rows, layer.norm_query.weight, config.rms_norm_eps, hc
        ).unflatten(-1, (hc, hidden))
        gate = (key * query).sum(-1, keepdim=True) / math.sqrt(hidden)
        gate = torch.sigmoid(gate.sign() * gate.abs().clamp_min(1e-6).sqrt())
        gated = (gate * value.unsqueeze(-2)).flatten(-2)
        normed = _group_norm(gated, layer.norm_conv.weight, config.rms_norm_eps, hc)
        history = torch.cat([states[i], normed.transpose(0, 1)], dim=-1)
        taps = sum(
            layer.conv1d.weight[:, 0, k].unsqueeze(0)
            * history.transpose(0, 1)[k * args.ple_conv_dilation :][: len(tokens)]
            for k in range(args.ple_conv_kernel_size)
        )
        ref = gated + F.silu(taps)
        assert torch.allclose(got[offset : offset + len(tokens)], ref, rtol=1e-4, atol=1e-5)
        offset += len(tokens)


def _fresh_ctx(**fields):
    import freetoken.core as core
    from freetoken.core import Context, set_global_ctx

    core._GLOBAL_CTX = None  # test-only: each scenario builds its own ctx
    ctx = Context(page_size=64)
    for name, value in fields.items():
        setattr(ctx, name, value)
    set_global_ctx(ctx)
    return ctx


def _plus_one_rmsnorm(x, weight, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * (1.0 + weight.float())).type_as(x)


def _hf_rope(x, positions, rotary_dim, base):
    """HF apply_rotary_pos_emb on [T, H, D], rotating only the first rotary_dim dims."""
    inv = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32) / rotary_dim)
    )
    freqs = positions.float().unsqueeze(-1) * inv
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).unsqueeze(1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).unsqueeze(1)
    rot = x[..., :rotary_dim].float()
    half = rotary_dim // 2
    rotated = torch.cat([-rot[..., half:], rot[..., :half]], dim=-1)
    out = (rot * cos + rotated * sin).type_as(x)
    return torch.cat([out, x[..., rotary_dim:]], dim=-1)


def _hf_attention(x, attn, config, positions):
    """HF Qwen4ExpTextAttention with a dense causal mask (QSA selects every block at this length)."""
    num_q, num_kv, dim = attn.num_q, attn.num_kv, attn.head_dim
    qkv = F.linear(x, attn.qkv_proj.weight)
    qg, k, v = qkv.split(attn._qkv_split, dim=-1)
    qg = qg.view(-1, num_q, dim * 2)
    q, gate = qg[..., :dim], qg[..., dim:].reshape(-1, num_q * dim)
    q = _hf_rope(_plus_one_rmsnorm(q, attn.q_norm.weight, config.rms_norm_eps), positions,
                 config.rotary_config.rotary_dim, config.rotary_config.base)
    k = _hf_rope(
        _plus_one_rmsnorm(k.view(-1, num_kv, dim), attn.k_norm.weight, config.rms_norm_eps),
        positions, config.rotary_config.rotary_dim, config.rotary_config.base,
    )
    v = v.view(-1, num_kv, dim)
    rep = num_q // num_kv
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.repeat_interleave(rep, 1).float())
    scores = scores * dim**-0.5
    mask = torch.arange(x.shape[0], device=x.device) > positions.unsqueeze(-1)
    out = torch.einsum(
        "hqk,khd->qhd", scores.masked_fill(mask, float("-inf")).softmax(-1),
        v.repeat_interleave(rep, 1).float(),
    ).to(x.dtype)
    return F.linear(out.reshape(-1, num_q * dim) * torch.sigmoid(gate), attn.o_proj.weight)


@requires_cuda
def test_qsa_layer_matches_hf_dense():
    """The QSA layer under the dense oracle backend equals HF attention, and freezes what the indexer hands the backend."""
    from freetoken.models.qwen4_exp.attention import Qwen4ExpAttention, TorchDenseQSAReference
    from freetoken.utils.torch_utils import torch_dtype

    torch.manual_seed(6)
    config = _config()
    device, dtype = torch.device("cuda"), torch.bfloat16
    with torch.device(device), torch_dtype(dtype):
        attn = Qwen4ExpAttention(config, layer_id=3)
    _fill(attn, torch.Generator(device=device).manual_seed(7))

    seq_len = 24
    x = (torch.randn(seq_len, config.hidden_size, device=device, dtype=dtype) * 0.5)
    positions = torch.arange(seq_len, device=device, dtype=torch.int64)
    req = SimpleNamespace(extend_len=seq_len, cached_len=0, table_idx=1)
    batch = SimpleNamespace(padded_reqs=[req], reqs=[req], positions=positions)

    backend = TorchDenseQSAReference(config, num_slots=4, max_len=64, device=device, dtype=dtype)
    _fresh_ctx(attn_backend=backend)
    ref = _hf_attention(x, attn, config, positions)
    got = attn.forward(x, batch)
    assert torch.allclose(got.float(), ref.float(), rtol=2e-2, atol=2e-2)

    index = attn.indexer.forward(x)
    args = config.qwen4_args
    raw = F.linear(x, attn.indexer.index_qk_proj.weight)
    assert index.q.shape == (seq_len, args.index_n_heads, args.index_head_dim)
    assert index.k.shape == (seq_len, args.index_head_dim)
    assert torch.equal(index.q.reshape(seq_len, -1), raw[:, : args.index_n_heads * args.index_head_dim])
    assert torch.equal(index.k, raw[:, args.index_n_heads * args.index_head_dim :])
    assert index.q_norm_weight.data_ptr() == attn.indexer.q_layernorm.weight.data_ptr()


class _StubLinearMixer(BaseOP):
    """Stands in for the GDN layer; same [T, hidden] -> [T, hidden] shape."""

    def __init__(self, config, layer_id):
        self.out_proj = LinearReplicated(config.hidden_size, config.hidden_size, has_bias=False)

    def forward(self, x):
        return self.out_proj.forward(x)


@requires_cuda
def test_shared_expert_gate_fusion_matches_eager():
    """Qwen4ExpMoE only swaps qwen3_5's gemv+sigmoid+mul+add gate chain for two triton kernels."""
    from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE
    from freetoken.models.qwen4_exp.moe import Qwen4ExpMoE
    from freetoken.moe.fused import FusedMoe
    from freetoken.utils.torch_utils import torch_dtype

    config = _config()
    device, dtype = torch.device("cuda"), torch.bfloat16
    with torch.device(device), torch_dtype(dtype):
        moe = Qwen4ExpMoE(config, 0)
    _fill(moe, torch.Generator(device=device).manual_seed(21), scale=0.2)
    _fresh_ctx(moe_backend=FusedMoe())

    x = torch.randn(6, config.hidden_size, device=device, dtype=dtype) * 0.5
    fused = moe.forward(x.clone())
    eager = Qwen3_5MoE.forward(moe, x.clone())

    routed = moe.experts.forward(hidden_states=x.clone(), router_logits=moe.gate.forward(x))
    gate = torch.sigmoid(x.float() @ moe.shared_expert_gate.weight.float().view(-1))
    ref = routed.float() + gate.unsqueeze(1) * moe.shared_expert.forward(x).float()

    assert fused.shape == x.shape and fused.dtype == dtype
    torch.testing.assert_close(fused, eager, rtol=2e-2, atol=2e-2)
    # The fused gate stays in fp32 where the eager chain rounds the scalar to bf16.
    assert (fused.float() - ref).abs().max() <= (eager.float() - ref).abs().max()


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_tokens,hidden", [(1, 2560), (7, 640)])
def test_shared_gate_kernels_match_torch(num_tokens, hidden, dtype):
    """Shipping hidden size for the two shared-gate kernels, against the torch chain they replace."""
    from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add, shared_gate_sigmoid

    gen = torch.Generator(device="cuda").manual_seed(hidden + num_tokens)
    kw = {"generator": gen, "device": "cuda", "dtype": dtype}
    x = torch.randn(num_tokens, hidden, **kw)
    weight = torch.randn(1, hidden, **kw) * 0.05
    shared = torch.randn(num_tokens, hidden, **kw)
    routed = torch.randn(num_tokens, hidden, **kw)

    fused = shared_gate_mul_add(routed, shared, shared_gate_sigmoid(x, weight.view(-1)))
    eager = routed + shared * torch.sigmoid(F.linear(x, weight))
    ref = routed.float() + shared.float() * torch.sigmoid(x.float() @ weight.float().view(-1))[:, None]

    assert fused.dtype == dtype and fused.shape == routed.shape
    torch.testing.assert_close(fused, eager, rtol=2e-2, atol=2e-2)
    assert (fused.float() - ref).abs().max() <= (eager.float() - ref).abs().max() + 1e-6


@requires_cuda
def test_decoder_stack_prefill_and_decode(monkeypatch):
    """Ragged bs=3 prefill then a bs=3 decode step through the whole model with dummy weights."""
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.qwen4_exp import model as model_module
    from freetoken.models.qwen4_exp.attention import TorchDenseQSAReference
    from freetoken.models.qwen4_exp.ple import GpuResidentTable
    from freetoken.moe.fused import FusedMoe
    from freetoken.utils.torch_utils import torch_dtype

    torch.manual_seed(8)
    config = _config()
    args = config.qwen4_args
    device, dtype = torch.device("cuda"), torch.bfloat16
    monkeypatch.setattr(model_module, "build_linear_mixer", _StubLinearMixer)

    with torch.device(device), torch_dtype(dtype):
        model = model_module.Qwen4ExpForCausalLM(config)
    gen = torch.Generator(device=device).manual_seed(9)
    _fill(model, gen)
    multipliers, sizes, offsets = hash_constants(args)
    table = torch.randn(4096, args.ngram_head_dim, generator=gen, device=device, dtype=dtype) * 0.05
    for ple in model.model.ple_layers:
        ple.ple_embedding.layer_multipliers.copy_(multipliers)
        ple.ple_embedding.ngram_heads_vocab_sizes.copy_(sizes)
        ple.ple_embedding.ngram_heads_offsets.copy_(offsets)
        ple.ple_embedding.attach_table(GpuResidentTable(table, dtype=dtype))

    num_slots, max_len = 4, 64
    pool = LinearStatePool(
        config.linear_attention_group(), num_slots, dtype, device,
        slot_states=config.slot_states,
    )
    prompts = [[3, 4, EOS, 5, 6, 8], [2, EOS, 11, 12], [9, 10, 11, 12, 13]]
    ctx = _fresh_ctx(
        attn_backend=TorchDenseQSAReference(config, num_slots, max_len, device, dtype),
        moe_backend=FusedMoe(),
        linear_state_pool=pool,
    )
    reqs = [
        SimpleNamespace(
            extend_len=len(p), cached_len=0, table_idx=i + 1, linear_slot_idx=None,
            input_ids=torch.tensor(p, dtype=torch.int64),
        )
        for i, p in enumerate(prompts)
    ]
    flat = [t for p in prompts for t in p]
    last = torch.tensor(
        [sum(len(p) for p in prompts[: i + 1]) - 1 for i in range(len(prompts))], device=device
    )
    batch = SimpleNamespace(
        padded_reqs=reqs, reqs=reqs, size=len(reqs), is_prefill=True, is_decode=False,
        input_ids=torch.tensor(flat, dtype=torch.int64, device=device),
        positions=torch.cat([torch.arange(len(p)) for p in prompts]).to(device),
        attn_metadata=SimpleNamespace(get_last_indices=lambda bs: last[:bs]),
    )
    with ctx.forward_batch(batch):
        logits = model.forward()
    assert logits.shape == (len(prompts), config.vocab_size)
    assert torch.isfinite(logits.float()).all()

    for r, p in zip(reqs, prompts):
        r.cached_len = len(p)
        r.extend_len = 1
        r.input_ids = torch.cat([r.input_ids, torch.tensor([14], dtype=torch.int64)])
    decode = SimpleNamespace(
        padded_reqs=reqs, reqs=reqs, size=len(reqs), is_prefill=False, is_decode=True,
        input_ids=torch.tensor([14] * len(reqs), dtype=torch.int64, device=device),
        positions=torch.tensor([len(p) for p in prompts], dtype=torch.int64, device=device),
        attn_metadata=None,
    )
    with ctx.forward_batch(decode):
        decode_logits = model.forward()
    assert decode_logits.shape == (len(prompts), config.vocab_size)
    assert torch.isfinite(decode_logits.float()).all()
