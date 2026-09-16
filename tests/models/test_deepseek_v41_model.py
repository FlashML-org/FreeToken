"""V4.1 mHC passes each pre-mix to the following sublayer, including the head."""

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from freetoken.models.deepseek_v41.model import Block, Transformer, make_identity_pre_mix


def _hc_block():
    block = Block.__new__(Block)
    nn.Module.__init__(block)
    block.dim, block.hc_mult = 2, 2
    block.norm_eps, block.hc_eps, block.hc_sinkhorn_iters = 1e-20, 1e-6, 3
    return block


def test_shifted_pre_mix_and_final_collapse():
    block = _hc_block()
    block.attn_norm, block.ffn_norm, block.ffn = nn.Identity(), nn.Identity(), nn.Identity()
    block.ffn = type("ImageAwareIdentity", (nn.Module,), {"forward": lambda self, x, mask: x})()
    for name in ("hc_attn_fn", "hc_attn_scale", "hc_attn_base", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base"):
        setattr(block, name, None)
    mix_iter = iter([
        (torch.tensor([[[0., 1.]]]), torch.tensor([[[1., 2.]]]), torch.eye(2).view(1, 1, 2, 2)),
        (torch.tensor([[[.25, .75]]]), torch.tensor([[[2., 3.]]]), torch.eye(2).view(1, 1, 2, 2)),
    ])
    block.hc_mixes = lambda *args: next(mix_iter)
    h = torch.tensor([[[[1., 2.], [3., 4.]]]])
    seen = []
    result, next_pre = block._forward(h, make_identity_pre_mix(h, 2), None,
                                      lambda x: seen.append(x.clone()) or x)
    torch.testing.assert_close(seen[0], torch.tensor([[[1., 2.]]]))
    torch.testing.assert_close(result, torch.tensor([[[[12., 20.], [20., 32.]]]]))
    torch.testing.assert_close(block.hc_pre(result, next_pre), torch.tensor([[[18., 29.]]]))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_hc_sinkhorn_and_combination_match_float_reference(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(7)
    block = _hc_block()
    x = torch.randn(2, 3, 2, 2, device=device, dtype=torch.bfloat16)
    fn = torch.randn(8, 4, device=device)
    scale = torch.tensor([.3, -.2, .5], device=device)
    base = torch.randn(8, device=device)
    pre, post, comb = block.hc_mixes(x, fn, scale, base)
    normalized = x.flatten(-2).double()
    projected = (normalized @ fn.double().T) / (normalized.square().mean(-1, keepdim=True) + block.norm_eps).sqrt()
    pre_ref = (projected[..., :2] * scale[0].double() + base[:2].double()).sigmoid() + block.hc_eps
    post_ref = 2 * (projected[..., 2:4] * scale[1].double() + base[2:4].double()).sigmoid()
    comb_ref = (projected[..., 4:] * scale[2].double() + base[4:].double()).view(2, 3, 2, 2).softmax(-1) + block.hc_eps
    comb_ref /= comb_ref.sum(-2, keepdim=True) + block.hc_eps
    for _ in range(block.hc_sinkhorn_iters - 1):
        comb_ref /= comb_ref.sum(-1, keepdim=True) + block.hc_eps
        comb_ref /= comb_ref.sum(-2, keepdim=True) + block.hc_eps
    for actual, expected in ((pre, pre_ref), (post, post_ref), (comb, comb_ref)):
        torch.testing.assert_close(actual.double(), expected, atol=1e-6, rtol=1e-5)
    value = torch.randn(2, 3, 2, device=device, dtype=torch.bfloat16)
    reference = post_ref.unsqueeze(-1) * value.double().unsqueeze(-2)
    reference += torch.einsum("...pq,...pd->...qd", comb_ref, x.double())
    torch.testing.assert_close(block.hc_post(value, x, post, comb), reference.bfloat16(), atol=.02, rtol=.008)
    torch.testing.assert_close(block.hc_pre(x, pre), (pre_ref.unsqueeze(-1) * x.double()).sum(-2).bfloat16())


def test_target_builds_metadata_without_allocating_experts_or_engram_tables():
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.quantization import finalize_quant, QuantKind
    from freetoken.models.deepseek_v41.config import parse_config
    from freetoken.models.deepseek_v41.model import DeepseekV41ForCausalLM
    from freetoken.moe.offload_cache import iter_offload_moe_layers

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    raw = json.loads((Path(__file__).parent / "fixtures/deepseek_v41_nvfp4_config.json").read_text())
    with torch.device("meta"):
        adapter = DeepseekV41ForCausalLM(parse_config(raw))
    model = adapter._transformer
    experts = list(iter_offload_moe_layers(adapter))
    assert len(experts) == 40
    assert all(expert is layer.ffn.experts for expert, layer in zip(experts, model.layers))
    assert all(expert.quant_method.kind is QuantKind.NVFP4 for expert in experts)
    assert finalize_quant(adapter) == 40
    params = dict(model.named_parameters())
    assert sum(p.numel() * p.element_size() for p in params.values()) == 12_190_720_448
    assert "head.weight" in params
    assert not any("hc_head" in name or ".ffn.experts." in name or ".engram.embed." in name for name in params)
    assert len(model.layers) == 40 and model.vision is not None
    assert params["layers.0.ffn.gate.bias_vl"].shape == (384,)
    assert params["layers.1.engram.wkv.weight"].shape == (25600, 6144)
    assert params["layers.20.attn.compressor.wkv.weight"].shape == (512, 5120)
    assert "layers.21.attn.compressor.wkv.weight" not in params


def _cuda_stack():
    import numpy as np
    from dataclasses import asdict

    from freetoken.attention.dsv41_sparse import DSV41SparseAttnBackend
    from freetoken.core import Context, get_global_ctx, set_global_ctx
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kvcache.dsv41_cost_model import dsv41_pool_sizes
    from freetoken.kvcache.dsv41_paged_pool import DSV41PagedKVCache
    from freetoken.models.deepseek_v41.args import DeepseekV41Args
    from freetoken.models.deepseek_v41.config import parse_config
    from freetoken.models.deepseek_v41.engram import DiskEngramTable, EngramRuntime
    from freetoken.models.deepseek_v41.model import DeepseekV41ForCausalLM
    from freetoken.moe.offload_cache import OffloadMoeCache

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    args = DeepseekV41Args(
        n_layers=5, n_mtp_layers=0, compress_ratios=(0, 2, 2, 1, 1),
        kv_source_layers=(1, 3), index_source_layers=(1, 3, 4),
        candidate_source_layer=3, candidate_topk_blocks=2, candidate_block_size=2,
        dim=64, n_heads=2, head_dim=32, rope_head_dim=16, q_lora_rank=32,
        o_lora_rank=32, o_groups=2, window_size=8, index_n_heads=2, index_head_dim=32,
        index_topk=3, moe_inter_dim=64, n_routed_experts=4, n_activated_experts=2,
        vocab_size=128, hc_mult=2, engram_layer_ids=(1,), engram_num_embeddings=(17,),
        engram_max_ngram_size=2, engram_vocab_size=17, engram_n_heads=1,
        engram_head_dim=32, engram_compressed_vocab_size=128,
        vision_n_layers=1, vision_dim=32, vision_n_heads=2, vision_inter_dim=32,
        vision_patch_size=2, vision_downsample_ratio=2, image_token_id=127,
    )
    config = parse_config(asdict(args) | {"quantization_config": {"moe_quant_algo": "NVFP4"}})
    with torch.device("cuda"):
        model = DeepseekV41ForCausalLM(config)
    torch.manual_seed(12)
    state = {}
    for name, param in model.state_dict().items():
        if param.dtype == torch.float8_e8m0fnu:
            value = torch.full(param.shape, 1 / 32, device="cuda").to(param.dtype)
        elif "norm" in name and name.endswith("weight") or name.endswith(("q_weight", "k_weight")):
            value = torch.ones(param.shape, dtype=param.dtype, device="cuda")
        else:
            value = (torch.randn(param.shape, device="cuda") * .12).to(param.dtype)
        state[name] = value
    model.load_state_dict(state)
    pool = DSV41PagedKVCache(dsv41_pool_sizes(40, args, 1., P=8), args,
                            torch.device("cuda"), P=8, n_scratch=3)
    page_table = torch.empty(2, 64, dtype=torch.long, device="cuda")
    for row in range(2):
        page_table[row] = torch.arange(row * 64, (row + 1) * 64, device="cuda").view(-1, 8).flip(0).flatten()
    pool.attach_page_table(page_table)
    for base in range(0, 128, 8):
        pool.bind_window_pages(base, base)
    try:
        ctx = get_global_ctx()
    except AssertionError:
        ctx = Context(page_size=8)
        set_global_ctx(ctx)
    ctx.kv_cache = pool
    ctx.attn_backend = backend = DSV41SparseAttnBackend(config)
    cache = OffloadMoeCache(args.n_layers, args.n_routed_experts, args.n_routed_experts,
                           torch.device("cuda"), quant_format="nvfp4")
    sources = {}
    for name, shape, dtype in (
        ("gate_up_packed", (4, 128, 32), torch.uint8),
        ("gate_up_scale", (4, 128, 4), torch.float8_e4m3fn),
        ("gate_up_global", (4, 128), torch.float16),
        ("down_packed", (4, 64, 32), torch.uint8),
        ("down_scale", (4, 64, 4), torch.float8_e4m3fn),
        ("down_global", (4, 64), torch.float16),
    ):
        sources[name] = []
        for _ in range(args.n_layers):
            values = torch.randint(0, 256, shape, dtype=dtype) if dtype == torch.uint8 else torch.full(shape, .0625).to(dtype)
            sources[name].append(values.pin_memory())
    cache.set_bank_sources(sources)
    cache.reset()
    for layer in model._iter_offload_moe_layers():
        layer.offload_cache = cache
    modules = [layer.engram for layer in model._transformer.layers if layer.engram is not None]
    weights = torch.randn(17, 32).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    table = DiskEngramTable(weights, np.full((17, 1), 126, dtype=np.uint8))
    model._engram_runtime = EngramRuntime(args, modules, [table], np.arange(128), 64, "cuda")
    return model, ctx, backend, pool, cache


def _make_batch(ids, *, cached=0, row=0, decode=False, media=None):
    from freetoken.core import Batch, Req, SamplingParams

    req = Req(torch.tensor(ids, dtype=torch.int32), row, cached, 4, row,
              SamplingParams(), None, media=media)
    batch = Batch([req], "decode" if decode else "prefill")
    batch.padded_reqs = batch.reqs
    batch.input_ids = req.input_ids[-1:].cuda() if decode else req.input_ids[cached:].cuda()
    batch.positions = torch.tensor([len(ids) - 1], device="cuda") if decode else torch.arange(cached, len(ids), device="cuda")
    batch.active_table_idx = torch.tensor([row], dtype=torch.long, device="cuda")
    return batch


def _forward_batch(model, ctx, backend, ids, **kwargs):
    batch = _make_batch(ids, **kwargs)
    backend.prepare_metadata(batch)
    with torch.inference_mode(), ctx.forward_batch(batch), model.forward_host_ctx(batch, False):
        result = model.forward()
    torch.cuda.synchronize()
    return result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_whole_model_nvfp4_csa2_engram_prefill_decode_and_images():
    from freetoken.models.deepseek_v41.image_processor import IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START

    model, ctx, backend, pool, cache = _cuda_stack()
    ids = list(range(5, 18))
    full = _forward_batch(model, ctx, backend, ids, row=0)
    _forward_batch(model, ctx, backend, ids[:-1], row=1)
    decode = _forward_batch(model, ctx, backend, ids, cached=len(ids) - 1, row=1, decode=True)
    assert full.dtype == decode.dtype == torch.float32
    assert torch.isfinite(full).all() and torch.isfinite(decode).all()
    torch.testing.assert_close(decode, full, atol=.025, rtol=.025)
    image_ids = ids.copy()
    image_ids[1:5] = [127] * 4
    image = {"start": 1, "types": torch.tensor([IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END]),
             "patches": torch.randn(4, 3, 2, 2), "n_vit_h": 2, "n_vit_w": 2}
    image_result = _forward_batch(model, ctx, backend, image_ids, row=0, media=[image])
    assert torch.isfinite(image_result).all()
    assert "embeddings" in image and image["patches"] is None
    _forward_batch(model, ctx, backend, image_ids[:3], row=1, media=[image])
    chunked = _forward_batch(model, ctx, backend, image_ids, cached=3, row=1, media=[image])
    torch.testing.assert_close(chunked, image_result, atol=.025, rtol=.025)
    assert not torch.allclose(image_result, full, atol=.005, rtol=.005)
    different_image = {"start": 1, "types": image["types"],
                       "patches": torch.randn(4, 3, 2, 2), "n_vit_h": 2, "n_vit_w": 2}
    changed = _forward_batch(model, ctx, backend, image_ids, row=0, media=[different_image])
    assert not torch.equal(changed, image_result)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("target", ["cpu", "hybrid"])
def test_cuda_whole_model_cpu_and_hybrid_nvfp4_decode(target):
    from freetoken.moe.cpu_executor import CpuMoeExecutor, compiled_extension_supports

    if not compiled_extension_supports("swiglu_clamp"):
        pytest.skip("CPU extension needs the clamped SwiGLU epilogue")
    model, ctx, backend, _, cache = _cuda_stack()
    ids = list(range(5, 18))
    reference = _forward_batch(model, ctx, backend, ids, row=0)
    _forward_batch(model, ctx, backend, ids[:-1], row=1)
    cache.decode_target = target
    cache.cpu_layer_ids = frozenset(range(5)) if target == "cpu" else frozenset()
    cache.hybrid_max_fetch = 1
    executor = CpuMoeExecutor(cache, top_k=2, activation="swiglu_clamp",
                              apply_router_weight_on_input=False, num_threads=2,
                              max_tokens=2, device=torch.device("cuda"),
                              swiglu_alpha=1.0, swiglu_limit=10.0)
    cache.set_cpu_executor(executor)
    result = _forward_batch(model, ctx, backend, ids, cached=len(ids)-1, row=1, decode=True)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, reference, atol=.025, rtol=.025)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_whole_model_batched_requests_keep_engram_and_kv_separate():
    from freetoken.core import Batch

    model, ctx, backend, _, _ = _cuda_stack()
    sequences = [list(range(5, 18)), list(range(30, 43))]
    references = [_forward_batch(model, ctx, backend, ids, row=row) for row, ids in enumerate(sequences)]
    requests = [_make_batch(ids[:-1], row=row).reqs[0] for row, ids in enumerate(sequences)]
    batch = Batch(requests, "prefill")
    batch.padded_reqs = requests
    batch.input_ids = torch.cat([req.input_ids for req in requests]).cuda()
    batch.positions = torch.arange(12, device="cuda").repeat(2)
    batch.active_table_idx = torch.arange(2, dtype=torch.long, device="cuda")
    backend.prepare_metadata(batch)
    with torch.inference_mode(), ctx.forward_batch(batch), model.forward_host_ctx(batch, False):
        model.forward()
    requests = [_make_batch(ids, cached=12, row=row, decode=True).reqs[0] for row, ids in enumerate(sequences)]
    batch = Batch(requests, "decode")
    batch.padded_reqs = requests
    batch.input_ids = torch.tensor([ids[-1] for ids in sequences], dtype=torch.int32, device="cuda")
    batch.positions = torch.full((2,), 12, device="cuda")
    batch.active_table_idx = torch.arange(2, dtype=torch.long, device="cuda")
    backend.prepare_metadata(batch)
    with torch.inference_mode(), ctx.forward_batch(batch), model.forward_host_ctx(batch, False):
        result = model.forward()
    torch.cuda.synchronize()
    torch.testing.assert_close(result, torch.cat(references), atol=.025, rtol=.025)
