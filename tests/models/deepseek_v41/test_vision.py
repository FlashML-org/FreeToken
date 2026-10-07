"""Image layout, official ViT/logits parity and image replay through the shared encoder cache."""

from __future__ import annotations

import io
import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from freetoken.core import Batch
from freetoken.engine.engine import Engine
from freetoken.mm.config import MultimodalConfig
from freetoken.mm.encoder_cache import EncoderCache
from freetoken.mm.processors.deepseek_v41 import DeepseekV41MMProcessor
from freetoken.models.deepseek_v41.config import VisionConfig
from freetoken.scheduler.mm import mm_rows_after, plan_mm_batch
from freetoken.utils.torch_utils import torch_dtype

from .common import DIM, VOCAB, requires_cuda, tiny_text_config, write_tiny_checkpoint
from .reference.vision import Aligner, ViT

VC = asdict(VisionConfig(num_hidden_layers=2, hidden_size=32, num_attention_heads=4, intermediate_size=48,
                         patch_size=2, min_pixels=0, max_image_tokens=64))


def _args():
    return SimpleNamespace(vision_n_layers=2, vision_dim=32, vision_n_heads=4, vision_inter_dim=48,
                           vision_patch_size=2, vision_downsample_ratio=3, vision_rope_theta=10000.0, dim=DIM)


def _checkpoint(path, quantized=False):
    tensors = write_tiny_checkpoint(str(path), seed=11, quantized=quantized)
    torch.manual_seed(12)
    with torch_dtype(torch.bfloat16):
        vit, aligner = ViT(_args()), Aligner(_args())
    # Replace the unused dummy vision weight from the text-only fixture.
    tensors = {k: v for k, v in tensors.items() if not k.startswith("vision.")}
    for prefix, module in (("vision", vit), ("aligner", aligner)):
        tensors.update({f"{prefix}.{k}": v.detach().bfloat16().contiguous() for k, v in module.state_dict().items()})
    tensors.update({k: torch.randn(DIM, dtype=torch.bfloat16) for k in ("image_start", "image_end", "image_newline")})
    save_file(tensors, path / 'model-00001-of-00001.safetensors')
    (path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {k: 'model-00001-of-00001.safetensors' for k in tensors}}))
    config = json.loads((path / 'config.json').read_text())
    config.update(vision_config=VC, image_token_id=VOCAB - 1)
    (path / 'config.json').write_text(json.dumps(config))
    return tensors


def _processor():
    cfg = SimpleNamespace(vision_config=SimpleNamespace(**VC), image_token_id=VOCAB - 1)
    return DeepseekV41MMProcessor(cfg, '.', MultimodalConfig())


def _image(color):
    buf = io.BytesIO()
    Image.new('RGB', (14, 10), color).save(buf, format='PNG')
    return buf.getvalue()


def _inputs(length=151):
    ids = [3] * 17 + [VOCAB - 1] + [4] * (length - 19) + [VOCAB - 1]
    return _processor().apply(torch.tensor(ids, dtype=torch.int32), [_image('red'), _image('blue')])


def _prefill(eng, reqs):
    if not hasattr(eng, 'encoder_cache'):
        eng.encoder_cache = EncoderCache()
        eng.dtype = torch.bfloat16
        eng._mm_registered = set()
    for req in reqs:
        if id(req) not in eng._mm_registered:
            for item in req.mm_items or ():  # what the scheduler claims at admission
                eng.encoder_cache.register(item.hash, req.uid, mm_rows_after(item, req.cached_len))
            eng._mm_registered.add(id(req))
    batch = Batch(reqs=reqs, phase='prefill')
    batch.padded_reqs = reqs
    batch.input_ids = torch.cat([r.input_ids[r.cached_len:r.device_len] for r in reqs]).cuda()
    batch.positions = torch.cat([torch.arange(r.cached_len, r.device_len) for r in reqs]).cuda()
    jobs, plan, rows, _ = plan_mm_batch(reqs, eng.encoder_cache)
    batch.mm_encoder_jobs, batch.mm_gather_plan = jobs, plan
    if plan:
        batch.mm_rows = torch.tensor(rows, dtype=torch.long, device=eng.device)
        Engine._run_mm_encoder(eng, batch)
    eng._bind()
    eng.backend.prepare_metadata(batch)
    with eng.ctx.forward_batch(batch), eng.model.forward_host_ctx(batch, False):
        return eng.model.forward().float()


def test_processor_patch_order_and_all_span_rows_are_embeddings():
    pixels = np.arange(10 * 14 * 3, dtype=np.uint8).reshape(10, 14, 3)
    item = _processor().process([Image.fromarray(pixels)])[0]
    assert item.grid_thw == [1, 5, 7]
    expected = ((torch.tensor(pixels[:2, :2]).permute(2, 0, 1).float() / 255 - .5) / .5).bfloat16()
    assert torch.equal(item.feature[0], expected)
    assert len(_processor().prompt_replacement(item).full) == 10
    result = _inputs()
    assert [i.offsets for i in result.mm_items] == [[[17, 27]], [[159, 169]]]
    assert result.mrope_positions is None
    assert result.mm_items[0].hash != result.mm_items[1].hash


def test_text_only_skips_every_visual_parameter(tmp_path):
    from freetoken.models.deepseek_v41.weight import iter_weights
    from freetoken.mm.processor import get_mm_processor
    from .test_engine_config import _engine_config
    _checkpoint(tmp_path)
    enabled = _engine_config(str(tmp_path), moe_strategy='offload')
    assert enabled.model_config.is_multimodal
    disabled = MultimodalConfig(disabled_encoders=frozenset({'vision', 'audio'}))
    text = _engine_config(str(tmp_path), moe_strategy='offload', mm=disabled)
    assert not text.model_config.is_multimodal and not text.active_encoders
    assert get_mm_processor(str(tmp_path), disabled) is None
    loaded = dict(iter_weights(str(tmp_path), 'cpu', include_moe_experts=False, include_vision=False))
    assert not any(k.startswith('visual.') for k in loaded)
    assert get_mm_processor(str(tmp_path)).placeholder == [VOCAB - 1]


@requires_cuda
@pytest.mark.parametrize('placement', ['gpu', 'host'])
def test_vision_encoder_matches_official_with_padded_grid(tmp_path, placement):
    from .harness import TinyEngine
    tensors = _checkpoint(tmp_path)
    eng = TinyEngine(str(tmp_path))
    eng.model.place_encoder_weights(placement)
    item = _processor().process([Image.new('RGB', (14, 10), 'red')])[0]
    with torch_dtype(torch.bfloat16), torch.device('cuda'):
        vit, aligner = ViT(_args()), Aligner(_args())
    for prefix, module in (('vision', vit), ('aligner', aligner)):
        module.load_state_dict({k.removeprefix(prefix + '.'): v.cuda() for k, v in tensors.items() if k.startswith(prefix + '.')})
    with torch.inference_mode(), torch.device('cuda'):
        x = aligner(vit(item.feature.cuda(), 5, 7), 5, 7)
    got = eng.model.encode(item)
    torch.testing.assert_close(got[1:-1].view(2, 4, DIM)[:, :3].reshape(-1, DIM), x, atol=1e-3, rtol=1e-3)
    assert torch.equal(got[0], tensors['image_start'].cuda())
    assert torch.equal(got[-1], tensors['image_end'].cuda())
    assert torch.equal(got[[4, 8]], tensors['image_newline'].cuda().expand(2, -1))
    eng.model.place_encoder_weights('gpu')


@requires_cuda
@pytest.mark.parametrize('quantized', [False, True])
def test_image_logits_and_decode_match_reference(tmp_path, quantized):
    from .harness import TinyEngine
    from .test_reference_parity import ATOL, QUANT_TOL, Reference, _compare, _engram_table_for
    tol = QUANT_TOL if quantized else ATOL
    tensors = _checkpoint(tmp_path, quantized)
    text = tiny_text_config(moe_intermediate_size=256) if quantized else tiny_text_config()
    ref = Reference(tensors, text, max_seq_len=512, max_batch_size=2, quantized=quantized, vision_config=VC)
    eng = TinyEngine(str(tmp_path), max_seq_len=512, swa_decoder_replay='exact', quantized=quantized)
    _engram_table_for(eng, tensors, text)
    result = _inputs()
    ids = result.input_ids.clone()
    types = torch.full_like(ids, -1)
    images = []
    for item in result.mm_items:
        lo, hi = item.offsets[0]
        ids[lo:hi] = VOCAB - 1
        typ = torch.tensor([0] + ([1] * 3 + [2]) * 2 + [3])
        types[lo:hi] = typ
        images.append(SimpleNamespace(start=lo, patches=item.feature, n_vit_h=5, n_vit_w=7, types=typ))
    with torch.device('cuda'):
        want = ref.model(ids.cuda()[None], images=[images], token_types=types.cuda()[None])[1]
    req = eng.new_request(0, result.input_ids.tolist())
    req.mm_items = result.mm_items
    got = _prefill(eng, [req])
    _compare('image prefill', got, want, tol=tol)
    eng.finish_prefill([req])
    for step in range(3):
        token = want.argmax(-1)
        want = ref.decode(token, len(ids) + step)
        got = eng.decode([req], token.tolist())
        _compare(f'image decode {step}', got, want, tol=tol)


@requires_cuda
@pytest.mark.parametrize('mode', ['exact', 'bounded'])
def test_images_survive_chunking_and_prefix_replay(tmp_path, mode):
    from .harness import TinyEngine
    from .test_reference_parity import _compare, _engram_table_for
    tensors = _checkpoint(tmp_path)
    eng = TinyEngine(str(tmp_path), max_seq_len=2048, swa_decoder_replay=mode)
    _engram_table_for(eng, tensors, tiny_text_config())
    # bounded mode caps a hit a window before the prompt end: trailing text keeps that cap at 128
    tail = 150 if mode == 'bounded' else 0
    def inputs():
        ids = [3] * 80 + [VOCAB - 1] + [4] * 37 + [VOCAB - 1] + [4] * 31 + [VOCAB - 1] + [4] * tail
        return _processor().apply(torch.tensor(ids, dtype=torch.int32), [_image(c) for c in ('red', 'blue', 'green')])
    whole = inputs()
    assert [i.offsets for i in whole.mm_items] == [[[80, 90]], [[127, 137]], [[168, 178]]]
    a = eng.new_request(0, whole.input_ids.tolist())
    a.mm_items = whole.mm_items
    want = _prefill(eng, [a])
    chunk = inputs()
    b = eng.new_request(1, chunk.input_ids.tolist())
    b.mm_items = chunk.mm_items
    b.device_len = 128
    _prefill(eng, [b])
    b.cached_len, b.device_len = 128, len(chunk.input_ids)
    got = _prefill(eng, [b])
    _compare('chunked image prefill', got, want)
    prefix = inputs()
    # The second image straddles the chunk / prefix boundary at 128 (in bounded mode the first chunk
    # also ends before the prompt's last window and runs no decoder row).
    hit = eng.new_request_on_prefix(1, a, prefix.input_ids.tolist())
    assert hit.cached_len == 128
    hit.mm_items = prefix.mm_items
    _compare('image prefix hit', _prefill(eng, [hit]), want)
