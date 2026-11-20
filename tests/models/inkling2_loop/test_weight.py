"""Checkpoint transforms against the transformers #51050 conversion ops (``core_model_loading.py`` at 6173a2f679:
``Inkling2LoopDequantizeExperts``, ``Interleave(dim=1)``, ``Inkling2LoopMergeMessageQueries``), and the loop's
cache-layer numbering against ``Inkling2LoopTextModel.forward``'s ``cache_offset``."""

from __future__ import annotations

from types import SimpleNamespace

import torch

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _hf_dequant_nvfp4(weight, scale, amax, denominator):
    e2m1 = torch.tensor(E2M1)
    byte = torch.arange(256)
    byte_to_values = torch.stack((e2m1[byte & 0xF], e2m1[byte >> 4]), dim=-1)
    values = byte_to_values[weight.long()].view(*scale.shape, 16) * scale.float()[..., None]
    values = values * (amax.float() / denominator.float())[:, None, None, None]
    return values.reshape(weight.shape[0], weight.shape[1], -1)


def _hf_interleave(t):
    shape = list(t.shape)
    shape[1:2] = [shape[1] // 2, 2]
    return t.reshape(shape).transpose(1, 2).reshape(t.shape)


def test_nvfp4_experts_match_reference():
    from freetoken.models.inkling2_loop.weight import deinterleave_rows, dequant_nvfp4

    g = torch.Generator().manual_seed(0)
    E, rows, cols = 3, 8, 64
    weight = torch.randint(0, 256, (E, rows, cols // 2), generator=g, dtype=torch.uint8)
    scale = (torch.rand(E, rows, cols // 16, generator=g) * 400 + 1).to(torch.float8_e4m3fn)
    amax = torch.rand(E, generator=g) + 0.1
    denominator = torch.tensor([2688.0])
    ref = _hf_interleave(_hf_dequant_nvfp4(weight, scale, amax, denominator))
    # the reader keeps the packed rows: de-interleave codes and scales, then dequantize as the kernel does
    got = dequant_nvfp4(deinterleave_rows(weight), deinterleave_rows(scale), amax / denominator)
    assert torch.equal(got, ref)


def test_mxfp8_experts_match_reference():
    from freetoken.models.inkling2_loop.weight import deinterleave_rows, dequant_mxfp8

    g = torch.Generator().manual_seed(1)
    weight = (torch.randn(2, 6, 64, generator=g) * 100).to(torch.float8_e4m3fn)
    scale = torch.randint(110, 120, (2, 6, 2), generator=g, dtype=torch.uint8)
    ref = (weight.float().view(2, 6, 2, 32) * torch.exp2(scale.float() - 127)[..., None]).reshape(2, 6, 64)
    assert torch.equal(dequant_mxfp8(deinterleave_rows(weight), deinterleave_rows(scale)), _hf_interleave(ref))


def test_message_queries_merge():
    from freetoken.models.inkling2_loop.weight import merge_message_queries

    q = torch.arange(24 * 3, dtype=torch.float32).view(24, 3)
    kv = -q
    merged = merge_message_queries(q, kv)
    # per loop layer l: [query, key, value, mlp] = q[2l], kv[2l], kv[2l + 1], q[2l + 1]
    for layer in range(12):
        assert torch.equal(merged[layer], torch.stack([q[2 * layer], kv[2 * layer], kv[2 * layer + 1], q[2 * layer + 1]]))


def test_conv_taps_reverse_conv1d():
    from freetoken.models.inkling2_loop.weight import rename

    w = torch.randn(5, 1, 4)
    name, taps = rename("model.llm.layers.3.attn_sconv.weight", w)
    assert name == "model.layers.3.attn_conv"
    x = torch.randn(5, 9)
    out = torch.nn.functional.conv1d(x[None], w, padding=3, groups=5)[0, :, :9]
    xp = torch.nn.functional.pad(x, (3, 0))
    by_lag = sum(taps[lag][:, None] * xp[:, 3 - lag : 3 - lag + 9] for lag in range(4))
    assert torch.allclose(out, by_lag, atol=1e-6)


def test_cache_layers_follow_reference_offsets():
    from freetoken.models.inkling2_loop.config import parse_args

    text = SimpleNamespace(
        num_hidden_layers=24, local_layer_ids=[i for i in range(24) if (i + 1) % 6], hidden_size=2304,
        loop_config={"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64, "is_varloop": False},
        num_attention_heads=24, num_key_value_heads=8, head_dim=128, swa_num_attention_heads=24,
        swa_num_key_value_heads=8, swa_head_dim=128, sliding_window_size=1024, d_rel=16, rel_extent=1024,
        log_scaling_n_floor=8192, log_scaling_alpha=0.1, qk_l2_clip_threshold=22.6, rel_l2_clip_threshold=8.0,
        sconv_kernel_size=4, mlp_gate_softcap=4.0, mlp_up_softcap=25.0, route_scale=8.0, unpadded_vocab_size=200058,
        n_layers_per_attn_res_block=6,
    )
    args = parse_args(text)
    # the reference's call order: prelude, pass 1 at the layer's own cache layer, passes 2.. at layer + cache_offset, coda
    calls = [(i, i) for i in range(6)] + [(i, i) for i in range(6, 18)]
    for loop_idx in (1, 2):
        cache_offset = 24 + (loop_idx - 1) * 12 - 6
        calls += [(i, i + cache_offset) for i in range(6, 18)]
    calls += [(i, i) for i in range(18, 24)]
    sched = args.schedule()
    assert [layer for layer, _ in sched] == [layer for layer, _ in calls]
    assert [args.cache_layer(layer, p) for layer, p in sched] == [c for _, c in calls]
    assert [args.access(layer, p) for layer, p in sched] == list(range(48))
    assert sorted(args.cache_layers_of(True) + args.cache_layers_of(False)) == list(range(48))
