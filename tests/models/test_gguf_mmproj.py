"""Which llama.cpp mmproj file a text GGUF picks up: --mmproj first, else a fitting file in its own folder, BF16 before F16 before quantized, and none when the folder mixes several models."""
from __future__ import annotations

import struct

import pytest

_STRING, _ARRAY, _UINT32, _BOOL = 8, 9, 4, 7
_F32, _F16, _Q8_0, _BF16 = 0, 1, 8, 30


def _write_gguf(path, kv: dict, tensors: dict[str, int]) -> None:
    """A GGUF with ``kv`` and one 32-element tensor per ``{name: ggml type}``."""

    def s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    def value(v) -> bytes:
        if isinstance(v, bool):
            return struct.pack("<I?", _BOOL, v)
        if isinstance(v, int):
            return struct.pack("<II", _UINT32, v)
        return struct.pack("<I", _STRING) + s(v)

    nbytes = {_F32: 128, _F16: 64, _BF16: 64, _Q8_0: 34}
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    head += b"".join(s(k) + value(v) for k, v in kv.items())
    offset = 0
    for name, ggml_type in tensors.items():
        head += s(name) + struct.pack("<IQIQ", 1, 32, ggml_type, offset)
        offset += 32 * ((nbytes[ggml_type] + 31) // 32)
    head += b"\0" * (-len(head) % 32)
    path.write_bytes(head + b"\0" * offset)


def _text(path, width: int = 64, name: str = "Qwen3.6-35B-A3B"):
    kv = {"general.architecture": "qwen35", "general.basename": name, "qwen35.embedding_length": width}
    _write_gguf(path, kv, {"token_embd.weight": _F32})
    return str(path)


def _mmproj(path, ggml_type: int, width: int = 64, name: str = "Qwen3.6-35B-A3B", tensors: dict[str, int] | None = None):
    kv = {"general.architecture": "clip", "general.type": "mmproj", "general.basename": name, "clip.has_vision_encoder": True,
          "clip.projector_type": "qwen3vl_merger", "clip.vision.projection_dim": width}
    _write_gguf(path, kv, tensors or {"v.blk.0.attn_qkv.weight": ggml_type})
    return str(path)


@pytest.fixture
def find(monkeypatch):
    from freetoken.models.gguf import mmproj

    monkeypatch.delenv(mmproj.MMPROJ_ENV, raising=False)
    mmproj._find.cache_clear()
    yield mmproj.find_mmproj
    mmproj._find.cache_clear()


def test_prefers_bf16_in_the_model_folder(tmp_path, find):
    text = _text(tmp_path / "model-Q4_K_M.gguf")
    _mmproj(tmp_path / "mmproj-Q8_0.gguf", _Q8_0)
    _mmproj(tmp_path / "mmproj-F16.gguf", _F16)
    bf16 = _mmproj(tmp_path / "mmproj-BF16.gguf", _BF16)
    got = find(text)
    assert (got.path, got.source, got.weight_type) == (bf16, "auto-discovered", "BF16")
    assert len(got.skipped) == 2


def test_blockless_mmproj_is_typed_by_its_largest_non_f32_matrix(tmp_path, find):
    text = _text(tmp_path / "model.gguf")
    # the patch embedding stays F32 in every file type, so it must not decide the type
    _mmproj(tmp_path / "mmproj-a-F16.gguf", _F16, tensors={"v.patch_embd.weight": _F32, "mm.input_projection.weight": _F16})
    bf16 = _mmproj(tmp_path / "mmproj-b-BF16.gguf", _BF16, tensors={"v.patch_embd.weight": _F32, "mm.input_projection.weight": _BF16})
    got = find(text)
    assert (got.path, got.weight_type) == (bf16, "BF16")


def test_does_not_look_outside_the_model_folder(tmp_path, find):
    (tmp_path / "UD-Q4_K_XL").mkdir()
    text = _text(tmp_path / "UD-Q4_K_XL" / "model-UD-Q4_K_XL-00001-of-00002.gguf")
    _mmproj(tmp_path / "mmproj-F16.gguf", _F16)
    assert find(text) is None


def test_several_models_in_one_folder(tmp_path, find):
    from freetoken.models.gguf import mmproj

    text = _text(tmp_path / "Qwen3.6-35B-A3B-Q4_K_M.gguf")
    _mmproj(tmp_path / "mmproj-Qwen3.5-35B-A3B-BF16.gguf", _BF16, name="Qwen3.5-35B-A3B")
    own = _mmproj(tmp_path / "mmproj-Qwen3.6-35B-A3B-F16.gguf", _F16)
    assert find(text).path == own
    other = _text(tmp_path / "Unnamed-Q4_K_M.gguf", name="Unnamed")
    mmproj._find.cache_clear()
    assert find(other) is None


def test_skips_an_mmproj_of_another_width(tmp_path, find):
    text = _text(tmp_path / "model.gguf", width=64)
    _mmproj(tmp_path / "mmproj-other-BF16.gguf", _BF16, width=128)
    assert find(text) is None
    fits = _mmproj(tmp_path / "mmproj-F16.gguf", _F16, width=64)
    from freetoken.models.gguf import mmproj

    mmproj._find.cache_clear()
    assert find(text).path == fits


def test_env_override_wins_and_must_be_an_mmproj(tmp_path, find, monkeypatch):
    from freetoken.models.gguf import mmproj

    text = _text(tmp_path / "model.gguf")
    _mmproj(tmp_path / "mmproj-BF16.gguf", _BF16)
    (tmp_path / "other").mkdir()
    chosen = _mmproj(tmp_path / "other" / "vision.gguf", _Q8_0)
    monkeypatch.setenv(mmproj.MMPROJ_ENV, chosen)
    got = find(text)
    assert (got.path, got.source, got.weight_type) == (chosen, "--mmproj", "Q8_0")
    monkeypatch.setenv(mmproj.MMPROJ_ENV, text)
    with pytest.raises(ValueError, match="not a llama.cpp vision mmproj"):
        find(text)
