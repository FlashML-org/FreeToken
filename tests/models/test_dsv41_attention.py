from types import SimpleNamespace

import pytest
import torch

from freetoken.attention.dsv4_sparse import DSV4AttnMetadata
from freetoken.attention.dsv41_sparse import DSV41SparseAttnBackend
from freetoken.core import Context, get_global_ctx, set_global_ctx
from freetoken.kernel.triton.dsv41.quant import fp4_roundtrip, fp8_roundtrip, unpack_fp4, unpack_fp8
from freetoken.kvcache.dsv41_cost_model import dsv41_pool_sizes
from freetoken.kvcache.dsv41_paged_pool import DSV41PagedKVCache
from freetoken.models.deepseek_v41.attention import Attention, apply_rope


def _args():
    return SimpleNamespace(n_layers=5, compress_ratios=(0, 2, 2, 1, 1),
        kv_source_layers=(1, 3), index_source_layers=(1, 3, 4),
        candidate_source_layer=3, candidate_topk_blocks=2, candidate_block_size=2,
        dim=64, n_heads=2, head_dim=32, rope_head_dim=16, q_lora_rank=32,
        o_lora_rank=32, o_groups=2, window_size=8, index_n_heads=2, index_head_dim=32,
        index_topk=3, norm_eps=1e-20, rope_theta=10000., compress_rope_theta=160000.,
        original_seq_len=65536, rope_factor=16., beta_fast=32, beta_slow=1)


def _stack(device="cpu", weights=None, kv_quant="none"):
    args = _args()
    pool = DSV41PagedKVCache(dsv41_pool_sizes(40, args, 1, P=8), args, torch.device(device),
                           P=8, n_scratch=3, kv_quant=kv_quant)
    table = torch.empty(2, 64, dtype=torch.long, device=device)
    for row in range(2):
        table[row] = torch.arange(row * 64, (row + 1) * 64, device=device).view(-1, 8).flip(0).flatten()
    pool.attach_page_table(table)
    for base in range(0, 128, 8):
        pool.bind_window_pages(base, base)
    try:
        ctx = get_global_ctx()
    except AssertionError:
        ctx = Context(page_size=8)
        set_global_ctx(ctx)
    ctx.kv_cache = pool
    ctx.attn_backend = backend = DSV41SparseAttnBackend(SimpleNamespace(dsv41_args=args))
    layers = [Attention(i, args).to(device) for i in range(args.n_layers)]
    if weights is None:
        torch.manual_seed(8)
        for layer in layers:
            for name, p in layer.named_parameters():
                if p.dtype == torch.float8_e8m0fnu:
                    p.data.copy_(torch.full(p.shape, 1 / 16, device=device).to(p.dtype))
                elif "norm.weight" in name:
                    p.data.fill_(1)
                else:
                    p.data.copy_((torch.randn(p.shape, device=device) * .15).to(p.dtype))
    else:
        for layer, state in zip(layers, weights):
            layer.load_state_dict(state)
    for layer in layers:
        layer.bind(pool, torch.device(device))
    return ctx, backend, pool, layers


def _reference(layers, inputs):
    out = []
    length = inputs.shape[1]
    positions = torch.arange(length, device=inputs.device)
    shared_kv = shared_keys = shared_picks = candidates = None
    for layer, x in zip(layers, inputs):
        qr = layer.q_norm(layer.wq_a(x))
        q = apply_rope(layer.wq_b(qr).unflatten(-1, (layer.n_heads, layer.head_dim)), positions, layer.inv_freq)
        kv = fp8_roundtrip(apply_rope(layer.kv_norm(layer.wkv(x)), positions, layer.inv_freq), block_size=32)
        ratio = layer.ratio
        if layer.is_kv_source:
            comp = layer.compressor
            if ratio == 2:
                count = length // 2 * 2
                raw = comp.wkv(x[:count].float()).view(-1, 2, layer.head_dim)
                scores = comp.wgate(x[:count].float()).view_as(raw)
                latent = comp.norm((raw * scores.softmax(1)).sum(1).to(x.dtype))
            else:
                latent = comp.norm(comp.wkv(x))
            compressed_pos = torch.arange(length // ratio, device=x.device) * ratio
            idx = layer.indexer
            shared_keys = fp4_roundtrip(apply_rope(idx.k_norm(idx.wk(latent)), compressed_pos, layer.inv_freq),
                                        block_size=32, scale_format="e8m0")
            shared_kv = fp4_roundtrip(apply_rope(latent, compressed_pos, layer.inv_freq),
                                      block_size=16, scale_format="e4m3")
        if layer.is_index_source:
            idx = layer.indexer
            iq = fp4_roundtrip(apply_rope(idx.wq_b(qr).unflatten(-1, (idx.n_heads, idx.head_dim)),
                                          positions, layer.inv_freq), block_size=32, scale_format="e8m0")
            weights = idx.weights_proj(x) * idx.scale
            dots = torch.einsum("qhd,kd->qhk", iq.float(), shared_keys.float()).to(iq.dtype)
            score = (dots.relu() * weights[..., None]).sum(1)
            visible = (positions + 1) // ratio
            score.masked_fill_(torch.arange(shared_keys.shape[0], device=x.device)[None] >= visible[:, None], -torch.inf)
            if layer.layer_id == layer.args.candidate_source_layer:
                block_size = layer.args.candidate_block_size
                blocks = torch.nn.functional.pad(score, (0, -score.shape[-1] % block_size), value=-torch.inf)
                blocks = blocks.unflatten(-1, (-1, block_size)).amax(-1)
                newest = (visible - 1) // block_size
                blocks.masked_fill_(torch.arange(blocks.shape[-1], device=x.device)[None] == newest[:, None], torch.inf)
                selected = blocks.argsort(dim=-1, descending=True, stable=True)[..., :layer.args.candidate_topk_blocks]
                keep = torch.zeros_like(blocks, dtype=torch.bool).scatter_(-1, selected, blocks.gather(-1, selected) > -torch.inf)
                candidates = keep.repeat_interleave(block_size, -1)[:, :score.shape[-1]]
            elif layer.layer_id > layer.args.candidate_source_layer:
                score.masked_fill_(~candidates, -torch.inf)
            shared_picks = score.argsort(dim=-1, descending=True, stable=True)[..., :layer.args.index_topk].sort(-1).values
            shared_picks = torch.where(shared_picks < visible[:, None], shared_picks, -1)
        token_outputs = []
        for position in range(length):
            selected = kv[max(0, position - layer.window_size + 1):position + 1]
            if ratio:
                indices = shared_picks[position]
                selected = torch.cat((selected, shared_kv[indices[indices >= 0]]), 0)
            logits = q[position].float() @ selected.float().T * layer.softmax_scale
            probs = torch.cat((logits, layer.attn_sink[:, None]), -1).softmax(-1)[:, :-1]
            token_outputs.append((probs @ selected.float()).to(x.dtype))
        value = apply_rope(torch.stack(token_outputs), positions, layer.inv_freq, inverse=True)
        grouped = value.view(length, layer.n_groups, -1)
        projected = torch.einsum("tgd,grd->tgr", grouped, layer.wo_a.view(layer.n_groups, layer.o_lora_rank, -1))
        out.append(layer.wo_b(projected.flatten(1)))
    return torch.stack(out)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))])
@pytest.mark.parametrize("kv_quant", ["none", "fp8-fp4"])
def test_source_sharing_and_two_stage_attention_match_direct_reference(device, kv_quant):
    _, _, _, layers = _stack(device, kv_quant=kv_quant)
    torch.manual_seed(2)
    inputs = torch.randn(5, 17, 64, device=device).bfloat16()
    expected = _reference(layers, inputs)
    actual = torch.stack([layer.forward_ragged(x[None], [(0, 17, 0, 0)], torch.arange(17, device=device))[0]
                          for layer, x in zip(layers, inputs)])
    torch.testing.assert_close(actual, expected, rtol=.02 if device == "cuda" else 0, atol=.002 if device == "cuda" else 0)


@pytest.mark.parametrize("cut", [1, 2, 7, 8, 9, 16])
@pytest.mark.parametrize("kv_quant", ["none", "fp8-fp4"])
def test_prefill_chunks_preserve_partial_pairs_and_shared_sources(cut, kv_quant):
    _, _, _, layers = _stack(kv_quant=kv_quant)
    weights = [layer.state_dict() for layer in layers]
    torch.manual_seed(31)
    inputs = torch.randn(5, 19, 64).bfloat16()
    expected = torch.stack([layer.forward_ragged(x[None], [(0, 19, 0, 0)], torch.arange(19))[0]
                            for layer, x in zip(layers, inputs)])
    _, _, _, layers = _stack(weights=weights, kv_quant=kv_quant)
    first = [layer.forward_ragged(x[None, :cut], [(0, cut, 0, 0)], torch.arange(cut))[0]
             for layer, x in zip(layers, inputs)]
    second = [layer.forward_ragged(x[None, cut:], [(0, 19 - cut, 0, cut)], torch.arange(cut, 19))[0]
              for layer, x in zip(layers, inputs)]
    actual = torch.stack([torch.cat((a, b)) for a, b in zip(first, second)])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kv_quant", ["none", "fp8-fp4"])
def test_batched_decode_is_request_isolated_and_matches_full_prefill(kv_quant):
    _, _, _, layers = _stack(kv_quant=kv_quant)
    weights = [layer.state_dict() for layer in layers]
    torch.manual_seed(34)
    inputs = torch.randn(2, 5, 17, 64).bfloat16()
    expected = [_reference(layers, sample) for sample in inputs]
    ctx, backend, pool, layers = _stack(weights=weights, kv_quant=kv_quant)
    for row in range(2):
        for layer, x in zip(layers, inputs[row]):
            layer.forward_ragged(x[None, :7], [(0, 7, row, 0)], torch.arange(7))
    actual = [[] for _ in range(2)]
    for position in range(7, 17):
        md = DSV4AttnMetadata(last_indices=torch.arange(2), full_snap=pool.full_loc_map.clone(), window_ar=torch.arange(8))
        batch = SimpleNamespace(attn_metadata=md)
        with ctx.forward_batch(batch):
            rows, positions = torch.arange(2), torch.full((2,), position)
            step = [layer.decode_step(inputs[:, i, position:position + 1], positions, rows, position)[:, 0]
                    for i, layer in enumerate(layers)]
        stacked = torch.stack(step)
        for row in range(2):
            actual[row].append(stacked[:, row])
    for row in range(2):
        torch.testing.assert_close(torch.stack(actual[row], 1), expected[row][:, 7:], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_packed_prefill_and_batched_decode_match_bf16_across_boundaries():
    torch.manual_seed(47)
    inputs = torch.randn(2, 5, 18, 64, device="cuda").bfloat16()
    starts, steps = (7, 15), 3

    def run(kv_quant, weights=None):
        ctx, backend, pool, layers = _stack("cuda", weights=weights, kv_quant=kv_quant)
        segments = [(0, starts[0], 0, 0), (starts[0], starts[1], 1, 0)]
        positions = torch.cat([torch.arange(length, device="cuda") for length in starts])
        selected, prefill = [], []

        def record_indices(layer):
            if layer.ratio:
                selected.append({key: value.clone() for key, value in backend.shared_indices.items()})

        for i, layer in enumerate(layers):
            x = torch.cat([inputs[row, i, :length] for row, length in enumerate(starts)])
            prefill.append(layer.forward_ragged(x[None], segments, positions)[0])
            record_indices(layer)
        prefill = torch.stack(prefill)
        outputs = [[prefill[:, offset:offset + length]]
                   for offset, length in ((0, starts[0]), (starts[0], starts[1]))]
        rows = torch.arange(2, device="cuda")
        for step in range(steps):
            positions = torch.tensor([start + step for start in starts], device="cuda")
            md = DSV4AttnMetadata(last_indices=rows, full_snap=pool.full_loc_map.clone(),
                                 window_ar=torch.arange(8, device="cuda"))
            decoded = []
            with ctx.forward_batch(SimpleNamespace(attn_metadata=md)):
                for i, layer in enumerate(layers):
                    x = torch.stack([inputs[row, i, start + step] for row, start in enumerate(starts)])
                    decoded.append(layer.decode_step(x[:, None], positions, rows, max(starts) + step)[:, 0])
                    record_indices(layer)
            decoded = torch.stack(decoded)
            for row in range(2):
                outputs[row].append(decoded[:, row:row + 1])
        return [torch.cat(parts, 1) for parts in outputs], selected, layers

    baseline, baseline_indices, layers = run("none")
    weights = [layer.state_dict() for layer in layers]
    for row, start in enumerate(starts):
        expected = _reference(layers, inputs[row, :, :start + steps])
        torch.testing.assert_close(baseline[row], expected, rtol=.02, atol=.002)
    packed, packed_indices, _ = run("fp8-fp4", weights)
    for actual, expected in zip(packed, baseline):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(packed_indices) == len(baseline_indices)
    for actual, expected in zip(packed_indices, baseline_indices):
        assert actual.keys() == expected.keys()
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_packed_attention_quantizes_raw_values_at_all_three_write_points(monkeypatch):
    import freetoken.models.deepseek_v41.attention as attention_module

    _, backend, pool, layers = _stack(kv_quant="fp8-fp4")
    layer = layers[1]
    torch.manual_seed(71)
    x = torch.randn(3, 64).bfloat16()
    positions = torch.tensor([0, 2, 4])
    raw_window = apply_rope(layer.kv_norm(layer.wkv(x)), positions, layer.inv_freq)
    original_fp8, original_fp4 = attention_module.pack_fp8, attention_module.pack_fp4
    calls = []

    def pack_window(value, block_size):
        torch.testing.assert_close(value, raw_window, rtol=0, atol=0)
        calls.append("window")
        return original_fp8(value, block_size)

    monkeypatch.setattr(attention_module, "pack_fp8", pack_window)
    _, _, window = layer._project(x, positions)
    backend.store_window(window, layer.layer_id, torch.tensor([1, 3, 5]))
    torch.testing.assert_close(unpack_fp8(pool.window_pool[1][[1, 3, 5]]), fp8_roundtrip(raw_window), rtol=0, atol=0)
    latent = torch.randn(3, 32).bfloat16()
    raw_keys = apply_rope(layer.indexer.k_norm(layer.indexer.wk(latent)), positions, layer.inv_freq)
    raw_compressed = apply_rope(latent.clone(), positions, layer.inv_freq)

    def pack_compressed(value, block_size, scale_format):
        expected = raw_keys if scale_format == "e8m0" else raw_compressed
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
        calls.append(scale_format)
        return original_fp4(value, block_size, scale_format)

    monkeypatch.setattr(attention_module, "pack_fp4", pack_compressed)
    layer._publish(latent, positions, torch.tensor([1, 3, 5]), torch.tensor([1, 3, 5]))
    assert calls == ["window", "e8m0", "e4m3"]
    torch.testing.assert_close(unpack_fp4(pool.idx_pool[1][[1, 3, 5]], 32, "e8m0"),
                               fp4_roundtrip(raw_keys, 32, "e8m0"), rtol=0, atol=0)
    torch.testing.assert_close(unpack_fp4(pool.cmp_pool[1][[1, 3, 5]], 16, "e4m3"),
                               fp4_roundtrip(raw_compressed, 16, "e4m3"), rtol=0, atol=0)
