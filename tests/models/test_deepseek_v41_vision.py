from types import SimpleNamespace
import base64
import io

import pytest
import torch
from PIL import Image

from freetoken.models.deepseek_v41.image_processor import (
    IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START, image_token_types,
    load_image, load_image_bytes, num_image_tokens, plan_image_grid,
)
from freetoken.models.deepseek_v41.vision import Aligner, ViT, merge_image_embeddings


def _args(**overrides):
    values = dict(vision_patch_size=2, vision_dim=8, vision_n_heads=2,
                  vision_inter_dim=12, vision_n_layers=2, vision_rope_theta=10000.0,
                  vision_downsample_ratio=2, dim=6, vision_min_pixels=16,
                  vision_max_n_token=24, vision_max_wh_ratio=None)
    return SimpleNamespace(**(values | overrides))


@pytest.mark.parametrize("width,height", [(1, 10000), (10000, 1), (1, 1), (37, 53), (2000, 3000)])
def test_image_resize_respects_token_budget(width, height):
    args = _args()
    h, w, pixels_h, pixels_w = plan_image_grid(width, height, args)
    assert h > 0 and w > 0
    assert pixels_h % args.vision_patch_size == pixels_w % args.vision_patch_size == 0
    assert num_image_tokens(h, w) <= args.vision_max_n_token


def test_image_patches_normalization_and_layout():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 4), (255, 0, 0)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    patches, nh, nw, lh, lw = load_image({"url": "data:image/png;base64," + encoded}, _args())
    assert (nh, nw, lh, lw) == (2, 4, 1, 2)
    assert patches.shape == (8, 3, 2, 2)
    assert patches.dtype == torch.float32
    torch.testing.assert_close(patches[:, 0], torch.ones_like(patches[:, 0]))
    torch.testing.assert_close(patches[:, 1:], -torch.ones_like(patches[:, 1:]))
    assert image_token_types(lh, lw).tolist() == [IMAGE_START, IMAGE, IMAGE, IMAGE_NEW_LINE, IMAGE_END]


@pytest.mark.parametrize("url", ["/etc/passwd", "file:///etc/passwd", "http://127.0.0.1/test", "http://[::1]/test"])
def test_api_image_loader_rejects_local_sources(url):
    with pytest.raises(ValueError):
        load_image_bytes({"url": url})


def test_aligner_channel_and_pixel_order():
    torch.manual_seed(12)
    args = _args()
    aligner = Aligner(args).float()
    values = torch.randn(3, 5, args.vision_dim)
    rows = []
    for h in range(0, 3, 2):
        for w in range(0, 5, 2):
            block = torch.zeros(args.vision_dim, 2, 2)
            crop = values[h:h + 2, w:w + 2].permute(2, 0, 1)
            block[:, :crop.shape[1], :crop.shape[2]] = crop
            rows.append(block.flatten())
    reference = aligner.w2(torch.nn.functional.gelu(aligner.w1(torch.stack(rows))))
    torch.testing.assert_close(aligner(values.reshape(-1, args.vision_dim), 3, 5), reference)


def test_vit_attention_matches_explicit_bidirectional_reference():
    torch.manual_seed(20)
    args = _args()
    model = ViT(args).float()
    patches = torch.randn(6, 3, 2, 2)
    from freetoken.models.deepseek_v41.vision import apply_rotary, get_vision_cos_sin

    x = model.patch_embed(patches)
    cos, sin = get_vision_cos_sin(2, 3, model.rope_dim, model.rope_theta)
    for block in model.blocks:
        q, k, v = [t.reshape(6, args.vision_n_heads, -1).transpose(0, 1)
                   for t in block.attn.wqkv(block.norm1(x)).chunk(3, -1)]
        q = apply_rotary(q.transpose(0, 1), cos, sin).transpose(0, 1)
        k = apply_rotary(k.transpose(0, 1), cos, sin).transpose(0, 1)
        scores = q @ k.transpose(-1, -2) / block.attn.head_dim ** 0.5
        attention = (scores.softmax(-1) @ v).transpose(0, 1).reshape(6, -1)
        x = x + block.attn.wo(attention)
        x = x + block.mlp(block.norm2(x))
    torch.testing.assert_close(model(patches, 2, 3), model.norm(x), atol=1e-6, rtol=1e-5)


def test_image_embedding_scatter_across_every_chunk_boundary():
    class Tower:
        def __init__(self):
            self.calls = 0
            self.vision = SimpleNamespace(patch_embed=SimpleNamespace(proj=SimpleNamespace(weight=torch.zeros(1))))
            self.image_start = torch.full((3,), 10.0)
            self.image_newline = torch.full((3,), 20.0)
            self.image_end = torch.full((3,), 30.0)

        def encode_image(self, patches, nh, nw):
            self.calls += 1
            return torch.arange(12, dtype=torch.float32).reshape(4, 3)

    types = image_token_types(2, 2)
    span = torch.stack([torch.full((3,), 10.0), *torch.arange(6.).reshape(2, 3),
                        torch.full((3,), 20.0), *torch.arange(6., 12.).reshape(2, 3),
                        torch.full((3,), 20.0), torch.full((3,), 30.0)])
    expected = torch.cat((torch.zeros(2, 3), span, torch.zeros(2, 3)))
    for boundary in range(1, expected.shape[0]):
        model = Tower()
        media = [dict(start=2, patches=torch.zeros(4, 3, 2, 2), n_vit_h=2, n_vit_w=2, types=types)]
        outputs, masks = [], []
        for start, stop in ((0, boundary), (boundary, expected.shape[0])):
            req = SimpleNamespace(cached_len=start, extend_len=stop-start, media=media)
            batch = SimpleNamespace(is_prefill=True, reqs=[req])
            h, mask = merge_image_embeddings(model, batch, torch.zeros(stop-start, 3))
            outputs.append(h)
            masks.append(mask)
        torch.testing.assert_close(torch.cat(outputs), expected)
        assert torch.cat(masks).tolist() == [False] * 2 + [True] * types.numel() + [False] * 2
        assert model.calls == 1
        assert media[0]["patches"] is None
        assert media[0]["types"] is types


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_vit_bf16_cuda_matches_cpu():
    torch.manual_seed(31)
    cpu = ViT(_args()).float().eval()
    gpu = ViT(_args()).to(device="cuda", dtype=torch.bfloat16).eval()
    gpu.load_state_dict(cpu.state_dict())
    patches = torch.randn(6, 3, 2, 2)
    with torch.no_grad():
        expected = cpu(patches, 2, 3)
        actual = gpu(patches.to(device="cuda", dtype=torch.bfloat16), 2, 3)
    torch.testing.assert_close(actual.float().cpu(), expected, rtol=0.04, atol=0.025)


def test_vision_parameters_preserve_checkpoint_dtypes():
    vision = ViT(_args())
    aligner = Aligner(_args())
    for name, parameter in vision.named_parameters():
        assert parameter.dtype == (torch.float32 if "norm" in name else torch.bfloat16)
    assert all(p.dtype == torch.bfloat16 for p in aligner.parameters())


def _processor():
    from freetoken.mm.config import MultimodalConfig
    from freetoken.models.deepseek_v41.mm_processor import DeepseekV41MMProcessor

    config = {"image_token_id": 99, "vision_config": {
        "num_hidden_layers": 2, "hidden_size": 8, "num_attention_heads": 2,
        "intermediate_size": 12, "patch_size": 2, "downsample_ratio": 2,
        "min_pixels": 16, "max_image_tokens": 24,
    }}
    return DeepseekV41MMProcessor(config, "unused", MultimodalConfig())


def test_native_media_and_shared_processor_produce_identical_content_keys():
    from freetoken.models.deepseek_v41.image_processor import ImageInput

    processor = _processor()
    buffer = io.BytesIO()
    Image.new("RGB", (8, 4), (255, 0, 0)).save(buffer, format="PNG")
    raw = buffer.getvalue()
    result = processor.apply(torch.tensor([11, 99, 12], dtype=torch.int32), [raw])
    patches, nh, nw, lh, lw = load_image({"data": raw}, processor.args)
    types = image_token_types(lh, lw)
    legacy_ids = torch.tensor([11] + [99] * len(types) + [12], dtype=torch.int32)
    converted = processor.from_media(legacy_ids, [ImageInput(1, patches, nh, nw, types)])
    assert torch.equal(converted.input_ids, result.input_ids)
    assert converted.mm_items[0].hash == result.mm_items[0].hash
    assert torch.equal(converted.mm_items[0].feature, result.mm_items[0].feature)
    assert result.mm_items[0].offsets == [[1, 6]]
    assert result.mm_items[0].types == types.tolist()
    assert result.mrope_positions is None and result.mrope_delta == 0
    changed_grid = processor._item(patches, nw, nh, types, 1)
    assert changed_grid.hash != result.mm_items[0].hash
    assert legacy_ids.tolist() == [11, 99, 99, 99, 99, 99, 12]


def test_precomputed_mm_rows_scatter_with_image_router_mask():
    hidden = torch.zeros(5, 3)
    embeddings = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    batch = SimpleNamespace(is_prefill=True, reqs=[SimpleNamespace()],
                            mm_embeds=embeddings, mm_rows=torch.tensor([1, 3]))
    actual, mask = merge_image_embeddings(None, batch, hidden)
    torch.testing.assert_close(actual[[1, 3]], embeddings)
    assert actual[[0, 2, 4]].count_nonzero() == 0
    assert mask.tolist() == [False, True, False, True, False]


def test_streamer_adapter_rebinds_native_parameters_without_changing_keys():
    from freetoken.models.deepseek_v41.vision import _ModuleBlockAdapter
    from freetoken.models.weight_stream import _slots

    block = ViT(_args()).blocks[0]
    names = set(block.state_dict())
    adapter = _ModuleBlockAdapter(block)
    slots, size = _slots(adapter)
    row = torch.zeros(size, dtype=torch.uint8)
    for slot in slots:
        value = slot.view(row)
        value.fill_(.25)
        setattr(slot.owner, slot.attr, value)
    assert set(block.state_dict()) == names
    assert all(torch.all(p == .25) for p in block.parameters())
    assert all(p.untyped_storage().data_ptr() == row.untyped_storage().data_ptr() for p in block.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_native_vision_host_streaming_matches_resident_across_repeated_images(monkeypatch):
    torch.manual_seed(71)
    vision = ViT(_args(vision_n_layers=3)).cuda().eval()
    patches = [torch.randn(6, 3, 2, 2, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    expected = [vision(patch, 2, 3).clone() for patch in patches]
    weights = {name: p.cpu().clone() for name, p in vision.named_parameters()}
    vision.place_weights("host")
    streamer = vision._streamer
    assert streamer.bank.is_pinned() and streamer.staging.shape[0] == 2
    assert all(p.device.type == "cpu" for block in vision.blocks for p in block.parameters())
    assert vision.patch_embed.proj.weight.is_cuda
    with monkeypatch.context() as mp:
        def fail_block(*args):
            raise RuntimeError("interrupted vision block")

        mp.setattr(vision.blocks[1], "forward", fail_block)
        with pytest.raises(RuntimeError, match="interrupted vision block"):
            vision(patches[0], 2, 3)
    assert all(p.device.type == "cpu" for block in vision.blocks for p in block.parameters())
    for _ in range(2):
        for patch, reference in zip(patches, expected):
            torch.testing.assert_close(vision(patch, 2, 3), reference, rtol=0, atol=0)
    assert all(p.device.type == "cpu" for block in vision.blocks for p in block.parameters())
    for name, p in vision.named_parameters():
        torch.testing.assert_close(p.cpu(), weights[name], rtol=0, atol=0)
    vision.place_weights("gpu")
    assert vision._streamer is None and all(p.is_cuda for p in vision.parameters())
    torch.testing.assert_close(vision(patches[0], 2, 3), expected[0], rtol=0, atol=0)
