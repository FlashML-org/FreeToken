"""LFM2-MoE routing and the dense MLP.

The router is a transcription of HF ``Lfm2MoeSparseMoeBlock.route_tokens_to_experts`` (the
engine's fused routers are softmax-only), so the transcription itself is the reference: sigmoid
scores, selection on ``scores + expert_bias``, weights from the RAW scores, renormalization
over the top-k with a 1e-6 eps, then ``routed_scaling_factor``. The dense MLP is
``w2(silu(w1(x)) * w3(x))``.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV = "cuda"
HIDDEN = 32
EXPERTS = 8
INTERMEDIATE = 16
EPS = 1e-6


@pytest.fixture(autouse=True)
def _single_rank_tp():
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _config(*, use_expert_bias=True, norm_topk_prob=True, scaling=1.0, top_k=2):
    from freetoken.models.config import ModelConfig, RotaryConfig

    return ModelConfig(
        num_layers=1, num_qo_heads=1, num_kv_heads=1, head_dim=64, hidden_size=HIDDEN,
        vocab_size=8, intermediate_size=INTERMEDIATE, rms_norm_eps=1e-5, hidden_act="silu",
        tie_word_embeddings=True, num_experts=EXPERTS, num_experts_per_tok=top_k,
        moe_intermediate_size=INTERMEDIATE, norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=scaling, use_expert_bias=use_expert_bias,
        model_type="lfm2_moe", architectures=["Lfm2MoeForCausalLM"], moe_enabled=True,
        rotary_config=RotaryConfig(head_dim=64, rotary_dim=64, max_position=64, base=1e4, scaling=None),
    )


def _make_block(seed=0, **kwargs):
    from freetoken.models.lfm2_moe.moe import Lfm2MoeSparseBlock

    config = _config(**kwargs)
    block = Lfm2MoeSparseBlock(config, layer_id=config.first_k_dense_replace, prefix="layers.0.feed_forward")
    torch.manual_seed(seed)
    block.gate.weight = torch.randn(EXPERTS, HIDDEN, device=DEV, dtype=torch.bfloat16)
    if block.expert_bias is not None:
        # a bias with a clear spread: it must move the selection, never the weights
        block.expert_bias = torch.linspace(-0.2, 0.2, EXPERTS, device=DEV, dtype=torch.float32)
    return block


def _hf_route(logits, bias, top_k, norm_topk_prob, scaling, eps=EPS):
    """HF ``route_tokens_to_experts`` verbatim."""
    routing_weights = logits.sigmoid()
    if bias is not None:
        _, selected = torch.topk(routing_weights + bias, k=top_k, dim=-1)
        routing_weights = torch.gather(routing_weights, dim=1, index=selected).type_as(logits)
    else:
        routing_weights, selected = torch.topk(routing_weights, k=top_k, dim=-1)
    if norm_topk_prob:
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + eps)
    routing_weights = routing_weights * scaling
    return selected, routing_weights


def _route(block, x):
    return block._route(x)


@pytest.mark.parametrize("norm_topk_prob,scaling", [(True, 1.0), (True, 2.5), (False, 1.0)])
def test_route_matches_hf(norm_topk_prob, scaling):
    block = _make_block(seed=1, norm_topk_prob=norm_topk_prob, scaling=scaling)
    torch.manual_seed(2)
    x = torch.randn(7, HIDDEN, device=DEV, dtype=torch.bfloat16)

    weights, ids = _route(block, x)
    ref_ids, ref_weights = _hf_route(
        block.gate.forward(x), block.expert_bias, block.top_k, norm_topk_prob, scaling
    )
    assert ids.dtype is torch.int32 and weights.dtype is torch.float32
    assert ids.shape == (7, block.top_k) and weights.shape == ids.shape
    assert torch.equal(ids.long(), ref_ids)
    assert torch.equal(weights, ref_weights.to(torch.float32))


def test_route_without_expert_bias_has_no_bias_tensor():
    block = _make_block(seed=3, use_expert_bias=False)
    assert block.expert_bias is None
    torch.manual_seed(4)
    x = torch.randn(5, HIDDEN, device=DEV, dtype=torch.bfloat16)

    weights, ids = _route(block, x)
    ref_ids, ref_weights = _hf_route(block.gate.forward(x), None, block.top_k, True, 1.0)
    assert torch.equal(ids.long(), ref_ids)
    assert torch.equal(weights, ref_weights.to(torch.float32))


def test_the_bias_moves_the_selection_but_not_the_weights():
    block = _make_block(seed=5)
    torch.manual_seed(6)
    x = torch.randn(4, HIDDEN, device=DEV, dtype=torch.bfloat16)
    logits = block.gate.forward(x)
    scores = logits.sigmoid()

    weights, ids = _route(block, x)

    unbiased = torch.topk(scores, block.top_k, dim=-1)[1]
    assert not torch.equal(ids.long(), unbiased), "the fixture must exercise the bias path"
    # weights are the raw scores of the BIASED selection, renormalized (never the biased score);
    # the division runs in the model dtype, as HF does
    raw = torch.gather(scores, 1, ids.long())
    assert torch.equal(weights, (raw / (raw.sum(-1, keepdim=True) + EPS)).to(torch.float32))


def test_the_renormalization_uses_the_1e_minus_6_eps():
    """Tiny sigmoid scores make the eps observable: glm4_moe's 1e-20 would return ~1.0 where
    LFM2 divides by ``sum + 1e-6``."""
    block = _make_block(seed=7)
    # logits ~ -17 -> top-k scores summing well below 1e-6, but still representable in bf16
    block.gate.weight = torch.full((EXPERTS, HIDDEN), -0.53125, device=DEV, dtype=torch.bfloat16)
    x = torch.ones(3, HIDDEN, device=DEV, dtype=torch.bfloat16)

    weights, ids = _route(block, x)
    logits = block.gate.forward(x)
    assert logits.sigmoid().sum(-1).max().item() < 1e-6  # the fixture is in the observable regime
    assert logits.sigmoid().sum(-1).max().item() > 0  # ... and not flushed to zero

    _ids6, ref6 = _hf_route(logits, block.expert_bias, block.top_k, True, 1.0, eps=1e-6)
    _ids20, ref20 = _hf_route(logits, block.expert_bias, block.top_k, True, 1.0, eps=1e-20)
    assert torch.allclose(weights, ref6.to(torch.float32), rtol=5e-2)
    assert not torch.allclose(weights, ref20.to(torch.float32), rtol=1e-2)


def test_dense_mlp_is_w2_silu_w1_times_w3():
    from freetoken.models.lfm2_moe.moe import Lfm2DenseMLP

    mlp = Lfm2DenseMLP(_config(), prefix="layers.0.feed_forward")
    torch.manual_seed(9)
    for proj in (mlp.w1, mlp.w3, mlp.w2):
        proj.weight = torch.randn(proj.weight.shape, device=DEV, dtype=torch.bfloat16) * 0.3
    x = torch.randn(6, HIDDEN, device=DEV, dtype=torch.bfloat16)

    out = mlp.forward(x)
    ref = mlp.w2.forward(torch.nn.functional.silu(mlp.w1.forward(x)) * mlp.w3.forward(x))
    assert torch.equal(out, ref)
