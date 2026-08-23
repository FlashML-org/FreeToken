"""gpt-oss mxfp4 MoE tests: kernel numerics (vs dequant reference) + offload
movement (bit-identical vs running the same kernel directly on the banks).

Lean suite — five gates:
  1. fused routing == softmax-topk-renorm
  2. split-K decode kernel == dequant reference
  3. _t prefill kernel == dequant reference
  4. offload decode == direct split-K, *under real LRU eviction* (ledger transparency)
  5. offload prefill (overlap=True) runs + == direct _t  (begin_prefill crash guard)
"""

import pytest
import torch

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _tiny_config():
    """Minimal gpt-oss-like ModelConfig for the mxfp4 offload path."""
    from freetoken.models.config import ModelConfig, RotaryConfig
    rotary = RotaryConfig(
        head_dim=64,
        rotary_dim=64,
        max_position=2048,
        base=10000.0,
        scaling=None,
    )
    return ModelConfig(
        num_layers=2,
        num_qo_heads=2,
        num_kv_heads=2,
        head_dim=64,
        hidden_size=128,
        vocab_size=256,
        intermediate_size=64,
        rms_norm_eps=1e-6,
        rotary_config=rotary,
        hidden_act="silu",
        tie_word_embeddings=False,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        norm_topk_prob=True,
        model_type="gpt_oss",
        architectures=["GptOssForCausalLM"],
        moe_weight_format="mxfp4",
        hidden_act_alpha=1.702,
        swiglu_limit=7.0,
    )


@pytest.fixture
def tp1(monkeypatch):
    import freetoken.layers.moe as moe_mod
    import freetoken.models.gpt_oss.weight as gpt_weight
    from freetoken.distributed import DistributedInfo

    tp_info = DistributedInfo(rank=0, size=1)
    monkeypatch.setattr(moe_mod, "get_tp_info", lambda: tp_info)
    monkeypatch.setattr(gpt_weight, "get_tp_info", lambda: tp_info)
    return tp_info


def _make_offload_cache(config, device, *, cache_size=None, prefill_overlap=False):
    from freetoken.moe.expert_banks import load_expert_banks
    from freetoken.moe.offload_cache import OffloadMoeCache

    banks = load_expert_banks(
        None,
        config,
        device=device,
        dtype=torch.bfloat16,
        dummy=True,
    )
    cache = OffloadMoeCache(
        num_layers=config.num_layers,
        num_experts=config.num_experts,
        cache_size=cache_size if cache_size is not None else config.num_layers * config.num_experts,
        device=device,
        cache_policy="lru",
        prefill_overlap=prefill_overlap,
        quant_format=banks.quant_format,
    )
    cache.set_bank_sources(banks.sources, layer_residency=banks.layer_residency)
    cache.set_alphas(banks.gate_up_alpha, banks.down_alpha)
    return cache


# ── kernel numerics ────────────────────────────────────────────────────────


@CUDA
@pytest.mark.parametrize("num_experts", [40, 128])  # 40 exercises the non-pow2 mask path
def test_gpt_oss_fused_routing_matches_softmax_topk_renorm(num_experts):
    from freetoken.kernel import gpt_oss_fused_routing

    device = torch.device("cuda")
    torch.manual_seed(0)
    tokens, top_k = 5, 4
    logits = torch.randn(tokens, num_experts, device=device, dtype=torch.bfloat16)

    weights, ids = gpt_oss_fused_routing(logits, top_k)
    assert weights.shape == (tokens, top_k)
    assert ids.shape == (tokens, top_k)
    assert ids.dtype == torch.int32
    assert weights.dtype == torch.float32

    # reference: softmax over all -> top-k -> renormalize (== softmax over top-k)
    probs = torch.softmax(logits.float(), dim=-1)
    ref_w, ref_id = torch.topk(probs, top_k, dim=-1)
    ref_w = ref_w / ref_w.sum(dim=-1, keepdim=True)

    # compare as {expert_id: weight} maps per token (order-independent)
    for t in range(tokens):
        got = {int(ids[t, k]): float(weights[t, k]) for k in range(top_k)}
        want = {int(ref_id[t, k]): float(ref_w[t, k]) for k in range(top_k)}
        assert set(got) == set(want), f"token {t}: {set(got)} != {set(want)}"
        for e in want:
            assert abs(got[e] - want[e]) < 1e-3, f"token {t} expert {e}: {got[e]} vs {want[e]}"


def _mxfp4_dequant_reference(
    run, M, seed, *, activation="gpt_oss_swiglu", alpha=1.702, limit=7.0
):
    """Run `run` (a split-K decode or _t prefill helper) on random mxfp4 experts and
    compare against a full dequant -> linear -> swiglu -> linear reference."""
    from freetoken.moe.fused_mxfp4 import (
        _transpose_mxfp4_for_decode,
        dequant_mxfp4_blocks,
        gpt_oss_swiglu,
        kimi_situ,
    )

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(seed)
    E, H, I = 8, 128, 64
    top_k = 2
    hb, ib = H // 32, I // 32

    gu_b = torch.randint(0, 256, (E, 2 * I, hb, 16), device=device, dtype=torch.uint8, generator=gen)
    gu_s = torch.full((E, 2 * I, hb), 124, device=device, dtype=torch.uint8)
    gu_bias = 0.05 * torch.randn(E, 2 * I, device=device, dtype=torch.bfloat16, generator=gen)
    dn_b = torch.randint(0, 256, (E, H, ib, 16), device=device, dtype=torch.uint8, generator=gen)
    dn_s = torch.full((E, H, ib), 124, device=device, dtype=torch.uint8)
    dn_bias = 0.05 * torch.randn(E, H, device=device, dtype=torch.bfloat16, generator=gen)

    hidden = 0.1 * torch.randn(M, H, device=device, dtype=torch.bfloat16, generator=gen)
    tid = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32, generator=gen)
    tw = torch.rand(M, top_k, device=device, dtype=torch.float32, generator=gen)

    gbt, gst = _transpose_mxfp4_for_decode(gu_b, gu_s)
    dbt, dst = _transpose_mxfp4_for_decode(dn_b, dn_s)
    out = run(
        hidden, tw, tid, gbt, gst, gu_bias, dbt, dst, dn_bias,
        top_k=top_k, hidden_act_alpha=alpha, swiglu_limit=limit,
        activation=activation,
    )

    expected = torch.zeros(M, H, device=device, dtype=torch.float32)
    for t in range(M):
        for r in range(top_k):
            e = int(tid[t, r])
            gw = dequant_mxfp4_blocks(gu_b[e], gu_s[e], out_dtype=torch.float32)
            gate_up = torch.nn.functional.linear(hidden[t].float(), gw, gu_bias[e].float())
            if activation == "situ":
                act = kimi_situ(gate_up, beta=alpha, linear_beta=limit)
            else:
                act = gpt_oss_swiglu(gate_up, alpha=alpha, limit=limit)
            dw = dequant_mxfp4_blocks(dn_b[e], dn_s[e], out_dtype=torch.float32)
            expected[t] += torch.nn.functional.linear(act, dw, dn_bias[e].float()) * float(tw[t, r])
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), expected, rtol=6e-2, atol=6e-2)


@CUDA
@pytest.mark.parametrize("M", [1, 3])  # M=1 broadcast path; M>1 gather path
def test_run_mxfp4_splitk_decode_experts_matches_reference(M):
    from freetoken.moe.fused_mxfp4 import run_mxfp4_splitk_decode_experts

    _mxfp4_dequant_reference(run_mxfp4_splitk_decode_experts, M, seed=21)


@CUDA
@pytest.mark.parametrize("M", [24])  # M > MXFP4_DECODE_MAX_TOKENS (16): prefill path
def test_run_mxfp4_prefill_experts_t_matches_reference(M):
    from freetoken.moe.fused_mxfp4 import run_mxfp4_prefill_experts_t

    _mxfp4_dequant_reference(run_mxfp4_prefill_experts_t, M, seed=31)


@CUDA
@pytest.mark.parametrize("M", [1, 3])
def test_run_mxfp4_splitk_kimi_situ_matches_reference(M):
    """Kimi's interleaved gate/up SiTU epilogue matches full MXFP4 dequantization."""
    from freetoken.moe.fused_mxfp4 import run_mxfp4_splitk_decode_experts

    _mxfp4_dequant_reference(
        run_mxfp4_splitk_decode_experts,
        M,
        seed=41,
        activation="situ",
        alpha=4.0,
        limit=25.0,
    )


@CUDA
def test_run_mxfp4_prefill_kimi_situ_matches_reference():
    from freetoken.moe.fused_mxfp4 import run_mxfp4_prefill_experts_t

    _mxfp4_dequant_reference(
        run_mxfp4_prefill_experts_t,
        24,
        seed=51,
        activation="situ",
        alpha=4.0,
        limit=25.0,
    )


# ── offload movement (bit-identical vs the same kernel on the banks) ─────────


@CUDA
def test_offload_decode_bit_identical_under_eviction(tp1):
    """Offload decode matches direct split-K across a real evict+reload cycle."""
    from freetoken.models.gpt_oss.moe import GptOssMxfp4OffloadMoELayer
    from freetoken.moe.fused_mxfp4 import (
        run_mxfp4_splitk_decode_experts as _run_mxfp4_splitk_decode_experts,
    )

    dev = torch.device("cuda")
    c = _tiny_config()
    E, H, tk = c.num_experts, c.hidden_size, c.num_experts_per_tok

    cache = _make_offload_cache(c, dev, cache_size=E)  # 4 slots, 8 total experts -> eviction

    layer1 = GptOssMxfp4OffloadMoELayer(c, layer_id=1)
    layer1.offload_cache = cache
    layer0 = GptOssMxfp4OffloadMoELayer(c, layer_id=0)
    layer0.offload_cache = cache

    torch.manual_seed(42)
    hidden = 0.1 * torch.randn(1, H, device=dev, dtype=torch.bfloat16)
    tw = torch.tensor([[0.6, 0.4]], dtype=torch.float32, device=dev)

    def ref(layer_id: int, expert_ids: list[int]) -> torch.Tensor:
        g = lambda k: cache.bank_sources[k][layer_id].to(dev)
        tid_ref = torch.tensor([expert_ids], dtype=torch.int32, device=dev)
        return _run_mxfp4_splitk_decode_experts(
            hidden, tw, tid_ref,
            g("gate_up_blocks"), g("gate_up_scales"), g("gate_up_bias"),
            g("down_blocks"), g("down_scales"), g("down_bias"),
            top_k=tk, hidden_act_alpha=c.hidden_act_alpha, swiglu_limit=c.swiglu_limit,
        )

    # A: cold-load layer-1 experts 0,1
    out_a = layer1._decode_routed(hidden, tw, torch.tensor([[0, 1]], dtype=torch.int32, device=dev))
    assert torch.equal(out_a, ref(1, [0, 1])), "Call A (cold load) failed"

    # B: cold-load layer-1 experts 2,3  (cache now full: L1E0..L1E3)
    out_b = layer1._decode_routed(hidden, tw, torch.tensor([[2, 3]], dtype=torch.int32, device=dev))
    assert torch.equal(out_b, ref(1, [2, 3])), "Call B (fill cache) failed"

    # Evictor: layer0 [0,1] claims the two LRU-oldest slots (A's) -> L1E0,L1E1 evicted
    out_ev = layer0._decode_routed(hidden, tw, torch.tensor([[0, 1]], dtype=torch.int32, device=dev))
    assert torch.equal(out_ev, ref(0, [0, 1])), "Evictor call failed"
    assert int(cache.slot_for_id[1, 0].item()) == -1, "Eviction did not happen (L1E0 still resident)"
    assert int(cache.slot_for_id[1, 1].item()) == -1, "Eviction did not happen (L1E1 still resident)"

    # C: L1E0,L1E1 missing -> reload (evicting L1E2,L1E3)
    out_c = layer1._decode_routed(hidden, tw, torch.tensor([[0, 1]], dtype=torch.int32, device=dev))
    assert torch.equal(out_c, ref(1, [0, 1])), "Call C (reload after eviction) failed"


@CUDA
@pytest.mark.parametrize("M", [16])
def test_offload_prefill_overlap_matches_reference(M, tp1):
    """Prefill overlap must not crash and must match the direct _t kernel."""
    from freetoken.models.gpt_oss.moe import GptOssMxfp4OffloadMoELayer
    from freetoken.moe.fused_mxfp4 import (
        run_mxfp4_prefill_experts_t as _run_mxfp4_prefill_experts_t,
    )

    dev = torch.device("cuda")
    c = _tiny_config()
    E, H, tk = c.num_experts, c.hidden_size, c.num_experts_per_tok

    cache = _make_offload_cache(c, dev, prefill_overlap=True)
    # layer_id=0 is mandatory: _wait_prefill_overlap gates begin_prefill() on layer_id==0,
    # so only layer 0 exercises the previously-crashing code path.
    layer = GptOssMxfp4OffloadMoELayer(c, layer_id=0)
    layer.offload_cache = cache

    torch.manual_seed(2)
    hidden = 0.1 * torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    logits = torch.randn(M, E, device=dev, dtype=torch.bfloat16)
    tw, tid = layer._topk(logits.contiguous())

    out_overlap = layer._prefill_routed(hidden, tw, tid.clone())  # must not raise

    g = lambda k: cache.bank_sources[k][0].to(dev)  # layer-0 source rows
    out_ref = _run_mxfp4_prefill_experts_t(
        hidden, tw, tid,
        g("gate_up_blocks"), g("gate_up_scales"), g("gate_up_bias"),
        g("down_blocks"), g("down_scales"), g("down_bias"),
        top_k=tk, hidden_act_alpha=c.hidden_act_alpha, swiglu_limit=c.swiglu_limit,
    )
    max_diff = (out_overlap - out_ref).abs().max().item()
    assert torch.equal(out_overlap, out_ref), f"overlap prefill differs; max_diff={max_diff}"
