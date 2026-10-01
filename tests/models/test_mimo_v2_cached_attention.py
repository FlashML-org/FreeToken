"""Teacher-forced cached decode through production MiMo attention and Triton KV attention.

This test deliberately avoids loading a model checkpoint: tiny deterministic BF16
weights exercise the same MiMoV2Attention.forward, HybridSWAKVCache.store_kv,
TritonAttentionBackend metadata, prefill, and decode kernels used by the engine.
The oracle below is independent Torch math, not another backend/helper invocation.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from freetoken.utils.torch_utils import torch_dtype

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="production Triton cached attention and KV store require CUDA",
)

HIDDEN = 32
Q_HEADS = 2
HEAD_DIM = 192
V_HEAD_DIM = 128
ROTARY_DIM = 64
WINDOW = 128
TOTAL = 139
# Every chunk is a genuine multi-token prefill, including the 127/128 window edge.
PREFILL_CHUNKS = ((0, 127), (127, 129), (129, 131))
DECODE_POSITIONS = tuple(range(131, 139))
# Triton accumulates attention in FP32 and rounds to BF16; subsequent model projections
# are BF16. 0.02 allows a few BF16 ULPs through those stages while remaining much smaller
# than the deliberately mutated sink/window outputs checked below. No argmax is used, so
# exact top-token ties cannot turn a numeric failure into a pass.
ATOL = 0.02
RTOL = 0.02


def _config():
    from freetoken.models.config import (
        FullAttentionGroupConfig,
        ModelConfig,
        RotaryConfig,
        SWAAttentionGroupConfig,
    )

    full_rope = RotaryConfig(HEAD_DIM, ROTARY_DIM, 256, 10_000_000.0, None)
    swa_rope = RotaryConfig(HEAD_DIM, ROTARY_DIM, 256, 10_000.0, None)
    return ModelConfig(
        num_layers=2,
        num_qo_heads=Q_HEADS,
        num_kv_heads=1,
        head_dim=HEAD_DIM,
        hidden_size=HIDDEN,
        vocab_size=97,
        intermediate_size=64,
        rms_norm_eps=1e-6,
        rotary_config=full_rope,
        hidden_act="silu",
        tie_word_embeddings=False,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        norm_topk_prob=False,
        model_type="mimo_v2_cached_attention_test",
        architectures=["MiMoV2ForCausalLM"],
        attention_value_scale=0.707,
        attention_groups=(
            FullAttentionGroupConfig(
                name="full",
                layer_ids=(0,),
                num_kv_heads=1,
                head_dim=HEAD_DIM,
                rotary_config=full_rope,
                v_head_dim=V_HEAD_DIM,
            ),
            SWAAttentionGroupConfig(
                name="swa",
                layer_ids=(1,),
                num_kv_heads=2,
                head_dim=HEAD_DIM,
                rotary_config=swa_rope,
                sliding_window=WINDOW,
                v_head_dim=V_HEAD_DIM,
            ),
        ),
    )


def _load_tiny_attention(module, device: torch.device, seed: int, *, sinks: bool) -> None:
    state = {}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    old_sinks = module.sinks
    if sinks:
        assert old_sinks is not None and old_sinks.device.type == "cpu"
    for name, value in module.state_dict().items():
        if name == "sinks":
            state[name] = torch.tensor([4.0, -4.0], dtype=torch.float32, device=device)
        else:
            scale = 0.16 if name.endswith("qkv_proj.weight") else 0.06
            state[name] = (
                torch.randn(value.shape, generator=generator, dtype=torch.float32)
                .mul_(scale)
                .to(device=device, dtype=value.dtype)
            )
            if name == "qkv_proj.weight":
                q_width = Q_HEADS * HEAD_DIM
                kv_heads = 1 if module.layer_id == 0 else 2
                k_width = kv_heads * HEAD_DIM
                state[name][: q_width + k_width].mul_(0.25)
                state[name][q_width + k_width :].mul_(3.0)
                q = state[name][:q_width].view(Q_HEADS, HEAD_DIM, HIDDEN)
                k = state[name][q_width : q_width + k_width].view(
                    kv_heads, HEAD_DIM, HIDDEN
                )
                for q_head in range(Q_HEADS):
                    q[q_head].copy_(k[q_head // (Q_HEADS // kv_heads)])
            elif name == "o_proj.weight":
                state[name].zero_()
                state[name][:16, :16] = torch.eye(16, device=device, dtype=value.dtype)
                state[name][16:, V_HEAD_DIM : V_HEAD_DIM + 16] = torch.eye(
                    16, device=device, dtype=value.dtype
                )
    module.load_state_dict(state)
    assert not state
    if sinks:
        assert module.sinks is not old_sinks
        assert module.sinks.is_cuda and module.sinks.dtype == torch.float32
        torch.testing.assert_close(
            module.sinks, torch.tensor([4.0, -4.0], device=device), atol=0, rtol=0
        )
    else:
        assert module.sinks is None


def _rotate_reference(x: torch.Tensor, positions: torch.Tensor, heads: int, theta: float):
    x = x.view(x.shape[0], heads, HEAD_DIM)
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, ROTARY_DIM, 2, device=x.device, dtype=torch.float32) / ROTARY_DIM)
    )
    angles = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    cos, sin = angles.cos()[:, None, :], angles.sin()[:, None, :]
    first = x[..., : ROTARY_DIM // 2].float()
    second = x[..., ROTARY_DIM // 2 : ROTARY_DIM].float()
    rotated = x.float().clone()
    rotated[..., : ROTARY_DIM // 2] = first * cos - second * sin
    rotated[..., ROTARY_DIM // 2 : ROTARY_DIM] = second * cos + first * sin
    return rotated.to(x.dtype).reshape(x.shape[0], heads * HEAD_DIM)


def _reference_qkv(module, x: torch.Tensor, positions: torch.Tensor, theta: float):
    qkv = F.linear(x, module.qkv_proj.weight)
    q_width, k_width, v_width = module.qkv_proj.output_sizes
    q, k, v = qkv.split((q_width, k_width, v_width), dim=-1)
    kv_heads = k_width // HEAD_DIM
    q = _rotate_reference(q, positions, Q_HEADS, theta).view(-1, Q_HEADS, HEAD_DIM)
    k = _rotate_reference(k, positions, kv_heads, theta).view(-1, kv_heads, HEAD_DIM)
    v = F.pad(
        v.view(-1, kv_heads, V_HEAD_DIM) * 0.707,
        (0, HEAD_DIM - V_HEAD_DIM),
    ).contiguous()
    return q, k, v


def _reference_output(
    module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pos: int,
    *,
    window: int | None,
    include_sinks: bool,
) -> torch.Tensor:
    lo = 0 if window is None else max(0, pos - window + 1)
    keys = k[lo : pos + 1].float()
    values = v[lo : pos + 1].float()
    kv_heads = keys.shape[1]
    kv_for_q = torch.arange(Q_HEADS, device=q.device) // (Q_HEADS // kv_heads)
    keys = keys.index_select(1, kv_for_q)
    values = values.index_select(1, kv_for_q)
    scores = (keys * q[pos].float().unsqueeze(0)).sum(-1).transpose(0, 1)
    scores *= HEAD_DIM**-0.5
    if include_sinks and module.sinks is not None:
        scores = torch.cat((scores, module.sinks.float()[:, None]), dim=1)
    probs = torch.softmax(scores, dim=-1)
    if include_sinks and module.sinks is not None:
        probs = probs[:, : keys.shape[0]]
    attended = (probs.unsqueeze(-1) * values.permute(1, 0, 2)).sum(dim=1)
    attended = attended.to(dtype=q.dtype)[None, :, :V_HEAD_DIM].reshape(1, Q_HEADS * V_HEAD_DIM)
    return F.linear(attended, module.o_proj.weight)


def _batch(ctx, ids, full_slots, start: int, end: int, phase: str, device):
    from freetoken.core import Batch, Req

    req = Req(
        input_ids=ids[:end].cpu().to(torch.int32),
        table_idx=0,
        cached_len=start,
        output_len=1,
        uid=0,
        sampling_params=None,
        cache_handle=None,
    )
    batch = Batch(reqs=[req], phase=phase)
    batch.padded_reqs = batch.reqs
    batch.input_ids = ids[start:end].to(device=device, dtype=torch.int32)
    batch.positions = torch.arange(start, end, device=device, dtype=torch.int32)
    batch.out_loc = full_slots[start:end]
    return batch


def _assert_mutation_rejected(name: str, mutant: torch.Tensor, baseline: torch.Tensor) -> None:
    normalized_margin = (
        (mutant - baseline).abs() / (ATOL + RTOL * baseline.abs())
    ).max().item()
    print(f"{name}_mutation_normalized_margin={normalized_margin:.4f}")
    assert normalized_margin > 1.0
    assert not torch.allclose(mutant, baseline, atol=ATOL, rtol=RTOL)


def test_mimo_attention_chunked_prefill_and_cached_decode_match_independent_oracle(monkeypatch):
    from freetoken import core
    from freetoken.attention.triton import TritonAttentionBackend
    from freetoken.core import Context
    from freetoken.distributed.info import DistributedInfo
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache
    from freetoken.layers.rotary import set_rope_device
    from freetoken.models.mimo_v2.attention import MiMoV2Attention

    device = torch.device("cuda")
    config = _config()
    monkeypatch.setattr(
        "freetoken.layers.linear.get_tp_info", lambda: DistributedInfo(rank=0, size=1)
    )
    monkeypatch.setattr(
        "freetoken.kvcache.hybrid_swa_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )
    set_rope_device(device)

    ctx = Context(page_size=1)
    ctx.kv_cache = HybridSWAKVCache(
        groups=config.kv_cache_group_specs(),
        num_layers=2,
        num_full_pages=TOTAL,
        page_size=1,
        dtype=torch.bfloat16,
        device=device,
        num_swa_tokens=TOTAL + 1,
    )
    slots = torch.arange(TOTAL, dtype=torch.int32, device=device)
    ctx.page_table = slots[None, :].clone()
    ctx.kv_cache.alloc_swa(slots)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    ctx.attn_backend = TritonAttentionBackend(config)

    with torch.device("cpu"), torch_dtype(torch.bfloat16):
        full_attn = MiMoV2Attention(config, 0, prefix="model.layers.0.self_attn")
        swa_attn = MiMoV2Attention(config, 1, prefix="model.layers.1.self_attn")
    for module in (full_attn, swa_attn):
        module.rotary._cos_sin_cache = module.rotary._cos_sin_cache.to(device)
    _load_tiny_attention(full_attn, device, 701, sinks=False)
    _load_tiny_attention(swa_attn, device, 702, sinks=True)

    tokens = ((torch.arange(TOTAL, device=device, dtype=torch.int32) * 17 + 5) % 97)
    tokens[10] = tokens[-1]
    torch.manual_seed(703)
    embedding = torch.randn(97, HIDDEN, device=device, dtype=torch.bfloat16)
    teacher_forced_x = F.embedding(tokens.long(), embedding)
    positions = torch.arange(TOTAL, device=device, dtype=torch.int32)
    modules = (
        (full_attn, 10_000_000.0, None),
        (swa_attn, 10_000.0, WINDOW),
    )
    references = {}
    for module, theta, window in modules:
        q, k, v = _reference_qkv(module, teacher_forced_x, positions, theta)
        references[module.layer_id] = [
            _reference_output(
                module, q, k, v, pos, window=window, include_sinks=True
            )
            for pos in range(TOTAL)
        ]

    observed = {0: {}, 1: {}}
    final_decode_batch = None
    with torch.inference_mode():
        for start, end in PREFILL_CHUNKS:
            batch = _batch(ctx, tokens, slots, start, end, "prefill", device)
            ctx.attn_backend.prepare_metadata(batch)
            with ctx.forward_batch(batch):
                for module, _, _ in modules:
                    result = module.forward(teacher_forced_x[start:end])
                    for offset, pos in enumerate(range(start, end)):
                        observed[module.layer_id][pos] = result[offset].float()

        for pos in DECODE_POSITIONS:
            batch = _batch(ctx, tokens, slots, pos, pos + 1, "decode", device)
            ctx.attn_backend.prepare_metadata(batch)
            if pos == DECODE_POSITIONS[-1]:
                final_decode_batch = batch
            with ctx.forward_batch(batch):
                for module, _, _ in modules:
                    result = module.forward(teacher_forced_x[pos : pos + 1])
                    observed[module.layer_id][pos] = result[0].float()

        assert final_decode_batch is not None
        for layer_id in (0, 1):
            assert set(observed[layer_id]) == set(range(TOTAL))
            for pos in range(TOTAL):
                torch.testing.assert_close(
                    observed[layer_id][pos],
                    references[layer_id][pos][0].float(),
                    atol=ATOL,
                    rtol=RTOL,
                    msg=f"layer={layer_id} position={pos}",
                )

        # Deliberate AttentionSpec mutations prove the fixture is sensitive to the
        # load-time sink value and exact SWA boundary, not just attribute replacement.
        baseline_final = observed[1][DECODE_POSITIONS[-1]]
        loaded_sinks = swa_attn.sinks
        swa_attn.sinks = None
        try:
            with ctx.forward_batch(final_decode_batch):
                no_sink = swa_attn.forward(teacher_forced_x[-1:]).float()
        finally:
            swa_attn.sinks = loaded_sinks
        _assert_mutation_rejected("sink_removal", no_sink, baseline_final)

        # An off-by-one window includes position 10 at query 138. That teacher-forced
        # input repeats the final token, making the single-key boundary shift observable.
        loaded_window = swa_attn.sliding_window
        swa_attn.sliding_window = WINDOW + 1
        try:
            with ctx.forward_batch(final_decode_batch):
                window_129_output = swa_attn.forward(teacher_forced_x[-1:]).float()
        finally:
            swa_attn.sliding_window = loaded_window
        _assert_mutation_rejected("window_129", window_129_output, baseline_final)

    assert swa_attn.sliding_window == WINDOW
    assert math.isclose(full_attn.sm_scale, HEAD_DIM**-0.5)
