from types import SimpleNamespace

import torch
import pytest

from freetoken.engine.engine import (
    DFlashMetrics,
    _dflash_target_verify_graph_enabled_for_config,
)
from freetoken.engine.sample import BatchSamplingArgs
from freetoken.speculative.utils import (
    AdaptiveGate,
    rejection_sample_chain,
    rejection_step,
    sampling_probs,
    select_output_tokens,
    select_streaming_output_tokens,
)
from freetoken.speculative.dflash.worker import DFlashWorker
from freetoken.speculative.dflash.config import DFlashConfig
from freetoken.speculative.dflash.model import _DFlashAttention


def testselect_output_tokens_accepts_longest_matching_prefix():
    base = torch.tensor([10], dtype=torch.int32)
    candidates = torch.tensor([20, 30, 40], dtype=torch.int32)
    verify = torch.tensor([20, 30, 41, 50], dtype=torch.int32)

    out, accepted = select_output_tokens(base, candidates, verify)

    assert accepted == 2
    assert torch.equal(out, torch.tensor([10, 20, 30, 41], dtype=torch.int32))


def testselect_streaming_output_tokens_waits_until_bonus_after_all_accept():
    base = torch.tensor([10], dtype=torch.int32)
    candidates = torch.tensor([20, 30], dtype=torch.int64)

    out, accepted, done = select_streaming_output_tokens(
        base, candidates, torch.tensor([20, 30], dtype=torch.int32)
    )
    assert done is False
    assert accepted == 2
    assert out.numel() == 0

    out, accepted, done = select_streaming_output_tokens(
        base, candidates, torch.tensor([20, 30, 41], dtype=torch.int32)
    )
    assert done is True
    assert accepted == 2
    assert torch.equal(out, torch.tensor([10, 20, 30, 41], dtype=torch.int32))


def test_dflash_target_verify_graph_disabled_for_offload_moe_backend(monkeypatch):
    import freetoken.engine.engine as engine

    monkeypatch.setattr(engine, "_DFLASH_TARGET_VERIFY_GRAPH", True)
    config = SimpleNamespace(moe_backend="offload")

    assert _dflash_target_verify_graph_enabled_for_config(config) is False


def test_dflash_target_verify_graph_allows_explicit_fused_moe_with_capture_safe_topk(monkeypatch):
    import freetoken.engine.engine as engine

    monkeypatch.setattr(engine, "_DFLASH_TARGET_VERIFY_GRAPH", True)
    config = SimpleNamespace(
        moe_backend="fused",
        model_config=SimpleNamespace(is_moe=True),
    )

    assert _dflash_target_verify_graph_enabled_for_config(config) is True


def test_dflash_worker_stores_hidden_context_in_single_contiguous_buffer():
    worker = DFlashWorker.__new__(DFlashWorker)
    worker._context_buffer = []
    worker._context_len = 0

    worker.store_hidden_states([
        torch.tensor([[1.0], [2.0]]),
        torch.tensor([[10.0], [20.0]]),
    ])
    worker.store_hidden_states([
        torch.tensor([[3.0]]),
        torch.tensor([[30.0]]),
    ])

    context = worker.get_context()

    assert len(worker._context_buffer) == 0
    assert worker.context_length == 3
    assert context.tolist() == [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]


def test_dflash_attention_cached_context_matches_uncached_forward():
    from freetoken.speculative.dflash.model import _DFlashAttention

    if not torch.cuda.is_available():
        pytest.skip("FlashInfer RMSNorm projection path requires CUDA")

    device = torch.device("cuda")
    cfg = DFlashConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=64,
        layer_types=["full_attention"],
    )
    attn = _DFlashAttention(cfg, layer_id=0).to(device)
    attn._apply_rope_inplace = lambda positions, query, key: None
    with torch.no_grad():
        eye = torch.eye(64, dtype=torch.bfloat16, device=device)
        attn.q_proj.weight.copy_(eye)
        attn.k_proj.weight.copy_(eye)
        attn.v_proj.weight.copy_(eye * 0.5)
        attn.o_proj.weight.copy_(eye)
        attn.q_norm.weight.fill_(1)
        attn.k_norm.weight.fill_(1)

    hidden_states = torch.arange(128, dtype=torch.bfloat16, device=device).view(2, 64) / 128
    context = torch.arange(192, dtype=torch.bfloat16, device=device).view(3, 64) / 192
    positions = torch.arange(2, dtype=torch.int32, device=device)
    context_positions = torch.arange(3, dtype=torch.int32, device=device)
    context_kv = attn.project_context_kv(context, context_positions)

    uncached = attn.forward(hidden_states, context, positions)
    cached = attn.forward(hidden_states, context, positions, context_kv=context_kv)

    assert torch.allclose(cached, uncached, atol=1e-3, rtol=1e-3)


def test_dflash_adaptive_gate_disables_when_slower_than_baseline():
    from freetoken.speculative.utils import AdaptiveGate

    gate = AdaptiveGate(min_cycles=4, eval_interval=4, margin=1.05, warmup_cycles=0)
    for _ in range(4):
        gate.record(cycle_ms=29.6, target_ms=6.0, out_tokens=4)
    assert gate.enabled is False
    assert gate.should_run(uid=1) is False


def test_dflash_adaptive_gate_stays_enabled_when_faster_than_baseline():
    from freetoken.speculative.utils import AdaptiveGate

    gate = AdaptiveGate(min_cycles=4, eval_interval=2, margin=1.05, warmup_cycles=0)
    for _ in range(10):
        gate.record(cycle_ms=22.0, target_ms=6.0, out_tokens=4)
    assert gate.should_run(uid=1) is True


def test_dflash_adaptive_gate_resets_on_new_request():
    from freetoken.speculative.utils import AdaptiveGate

    gate = AdaptiveGate(min_cycles=4, eval_interval=4, margin=1.05, warmup_cycles=0)
    assert gate.should_run(uid=1)
    for _ in range(4):
        gate.record(cycle_ms=29.6, target_ms=6.0, out_tokens=4)
    assert gate.should_run(uid=1) is False
    assert gate.should_run(uid=2) is True


def test_dflash_rejection_chain_accepts_all_when_draft_matches_target():
    from freetoken.speculative.utils import rejection_sample_chain

    V = 8
    base = torch.tensor([5], dtype=torch.int32)
    drafts = torch.tensor([1, 2, 3], dtype=torch.int32)
    draft_probs = torch.zeros(3, V)
    draft_probs[0, 1] = draft_probs[1, 2] = draft_probs[2, 3] = 1.0
    target_probs = torch.zeros(4, V)
    target_probs[0, 1] = target_probs[1, 2] = target_probs[2, 3] = 1.0
    target_probs[3, 7] = 1.0  # bonus distribution

    out, accepted = rejection_sample_chain(
        base, drafts, draft_probs, target_probs,
        uniform=torch.full((3,), 0.5),
    )

    assert accepted == 3
    assert torch.equal(out, torch.tensor([5, 1, 2, 3, 7], dtype=torch.int32))


def test_dflash_rejection_chain_rejects_and_samples_residual():
    from freetoken.speculative.utils import rejection_sample_chain

    V = 8
    base = torch.tensor([5], dtype=torch.int32)
    drafts = torch.tensor([1, 2], dtype=torch.int32)
    draft_probs = torch.zeros(2, V)
    draft_probs[0, 1] = draft_probs[1, 2] = 1.0
    target_probs = torch.zeros(3, V)
    target_probs[0, 6] = 1.0  # target puts zero mass on draft token 1
    target_probs[1, 2] = 1.0
    target_probs[2, 4] = 1.0

    out, accepted = rejection_sample_chain(
        base, drafts, draft_probs, target_probs,
        uniform=torch.full((2,), 0.5),
    )

    assert accepted == 0
    # residual = relu(p - q): p[6]=1, q[1]=1 -> residual one-hot on 6
    assert torch.equal(out, torch.tensor([5, 6], dtype=torch.int32))


def test_sample_and_select_greedy_matches_exact_match():
    from freetoken.engine.engine import _dflash_sample_and_select as sample_and_select

    V = 10
    base = torch.tensor([5], dtype=torch.int32)
    drafts = torch.tensor([1, 2, 3], dtype=torch.int32)
    # target argmax at positions 0..3: [1, 2, 9, 4] -> accept 2, bonus 9
    verify_logits = torch.zeros(4, V)
    verify_logits[0, 1] = 10; verify_logits[1, 2] = 10
    verify_logits[2, 9] = 10  # mismatch: draft[2]=3, target argmax=9
    verify_logits[3, 4] = 10
    args = BatchSamplingArgs(temperatures=None)  # greedy
    worker = SimpleNamespace(last_draft_probs=None)

def test_sample_and_select_sampling_uses_rejection_chain():
    from freetoken.engine.engine import _dflash_sample_and_select as sample_and_select

    V = 8
    base = torch.tensor([5], dtype=torch.int32)
    drafts = torch.tensor([1, 2, 3], dtype=torch.int32)
    # draft_probs one-hot on draft tokens; target_probs one-hot on same -> accept all
    draft_probs = torch.zeros(3, V)
    draft_probs[0, 1] = draft_probs[1, 2] = draft_probs[2, 3] = 1.0
    target_probs_logits = torch.zeros(4, V)
    target_probs_logits[0, 1] = 10; target_probs_logits[1, 2] = 10
    target_probs_logits[2, 3] = 10; target_probs_logits[3, 7] = 10
    args = BatchSamplingArgs(temperatures=torch.tensor([1.0]))
    worker = SimpleNamespace(last_draft_probs=draft_probs)



def test_dflash_target_verify_lens_within_budget_filters_by_snapshot_memory():
    from freetoken.engine.graph import _dflash_target_verify_lens_within_budget

    pool = SimpleNamespace(
        conv_states=torch.zeros((2, 4, 3, 2), dtype=torch.float32),
        recurrent_states=torch.zeros((2, 4, 1, 3, 3), dtype=torch.float32),
    )
    assert _dflash_target_verify_lens_within_budget([1, 2, 3, 4], pool, 1200) == [1, 2, 3, 4]
    assert _dflash_target_verify_lens_within_budget([1, 2, 3, 4], pool, 480) == [1, 2]
    assert _dflash_target_verify_lens_within_budget([1, 2, 3, 4], pool, 0) == []
    assert _dflash_target_verify_lens_within_budget([1, 2], None, 0) == [1, 2]


def test_graph_capture_buffer_target_verify_recurrent_snapshots_are_layer_major():
    from freetoken.core import Batch, Req
    from freetoken.engine.graph import GraphCaptureBuffer

    pool = SimpleNamespace(
        conv_states=torch.zeros((4, 2, 3, 2), dtype=torch.float32),
        recurrent_states=torch.zeros((4, 2, 1, 3, 3), dtype=torch.float32),
    )
    buffer = GraphCaptureBuffer.init_dflash_verify(
        2, vocab_size=5, device=torch.device("cpu"),
        hidden_size=3, hidden_dtype=torch.float32,
        num_hidden_layers=1, linear_state_pool=pool,
    )
    req = Req(
        input_ids=torch.tensor([10, 11, 12], dtype=torch.int32),
        table_idx=1, cached_len=1, output_len=1, uid=1,
        sampling_params=None, cache_handle=None,
    )
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = batch.reqs
    buffer.set_dflash_target_verify_batch(batch, verify_len=2, return_linear_snapshots=True)
    assert buffer.dflash_recurrent_states.shape == (4, 2, 1, 3, 3)
    assert torch.equal(
        batch.fla_metadata.dflash_recurrent_state_indices,
        torch.arange(4, dtype=torch.int32),
    )
