"""Image input plumbing: content-part rendering, ref collection, fetch, and the gate.

No GPU and no model checkpoint: everything here is pure frontend logic."""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

import pytest

from freetoken.mm.media import _MAX_IMAGE_BYTES as _MAX
from freetoken.mm.media import collect_image_refs, fetch_image_bytes, image_reject_reason
from freetoken.server.generation import GenerationError, render_messages
from freetoken.server.stats import derive_model_card

PNG = base64.b64encode(b"fakepng").decode()


def _config(**overrides):
    fields = dict(
        model_path="/nonexistent", allowed_media_domains="", allowed_local_media_path="",
        served_model_name="unit-model", max_seq_len=8192, model_config=SimpleNamespace(),
    )
    text_model_only = overrides.pop("text_model_only", False)
    serves_images = overrides.pop("vision_enabled", False)
    mm = SimpleNamespace(
        text_model_only=text_model_only,
        disabled_encoders=frozenset({"vision", "audio"}) if text_model_only else frozenset(),
    )
    return SimpleNamespace(mm=mm, served_modalities=frozenset({"image"}) if serves_images else frozenset(), **{**fields, **overrides})


def test_image_url_part_becomes_template_image_part():
    msgs = render_messages(
        [{"role": "user", "content": [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        ]}]
    )
    content = msgs[0]["content"]
    assert isinstance(content, list)
    assert content[1]["type"] == "image"
    refs = collect_image_refs(msgs)
    assert refs == [{"kind": "url", "data": "https://x/y.png"}]
    # refs are popped; the template-facing part stays
    assert msgs[0]["content"][1] == {"type": "image"}
    assert collect_image_refs(msgs) == []


def test_collect_refs_preserves_prompt_order():
    msgs = render_messages(
        [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u1"}}]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "u2"}},
                {"type": "image_url", "image_url": {"url": "u3"}},
            ]},
        ]
    )
    assert [r["data"] for r in collect_image_refs(msgs)] == ["u1", "u2", "u3"]


def test_fetch_decodes_data_uri_and_raw_b64():
    refs = [
        {"kind": "url", "data": f"data:image/png;base64,{PNG}"},
        {"kind": "b64", "data": PNG},
    ]
    assert asyncio.run(fetch_image_bytes(refs, _config())) == [b"fakepng", b"fakepng"]


def test_fetch_failures_surface_as_generation_error(monkeypatch):
    from freetoken.server import generation as gen

    # bypass the capability gate; the fetch failure itself must become a GenerationError
    monkeypatch.setattr(gen, "image_reject_reason", lambda config: None)
    state = SimpleNamespace(config=_config())
    with pytest.raises(GenerationError):
        asyncio.run(gen._resolve_images([{"kind": "url", "data": "ftp://nope"}], state))


def test_image_gate_reasons():
    assert "text-model-only" in image_reject_reason(_config(text_model_only=True))
    assert "vision" in image_reject_reason(_config(model_path="/nonexistent"))


def test_stats_model_card_lists_the_accepted_input_modalities():
    assert derive_model_card(_config())["input_modalities"] == ["text"]
    assert derive_model_card(_config(vision_enabled=True))["input_modalities"] == ["text", "image"]


def test_media_domain_allowlist():
    from freetoken.mm.media import _check_media_domain

    config = _config(allowed_media_domains="cdn.example.com, Other.COM.")
    _check_media_domain("https://cdn.example.com/a.png", config)  # allowed: no raise
    _check_media_domain("https://OTHER.com./b.png", config)  # case/root-dot normalized
    with pytest.raises(ValueError, match="allowed domains"):
        _check_media_domain("https://evil.com/a.png", config)
    # empty allowlist admits any domain
    _check_media_domain("https://evil.com/a.png", _config())

    with pytest.raises(ValueError, match="allowed domains"):
        asyncio.run(
            fetch_image_bytes([{"kind": "url", "data": "https://evil.com/a.png"}], config)
        )


def test_local_media_requires_allowlisted_root(tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"fakepng")
    url = f"file://{img}"

    # gate off (default): rejected
    with pytest.raises(ValueError, match="allowed-local-media-path"):
        asyncio.run(fetch_image_bytes([{"kind": "url", "data": url}], _config()))

    # gate on, file under the root: served
    config = _config(allowed_local_media_path=str(tmp_path))
    assert asyncio.run(fetch_image_bytes([{"kind": "url", "data": url}], config)) == [b"fakepng"]

    # a path outside the root is rejected even with the gate on
    with pytest.raises(ValueError, match="subpath"):
        asyncio.run(fetch_image_bytes([{"kind": "url", "data": "file:///etc/hostname"}], config))


def test_image_token_budget_flags_land_in_the_multimodal_config():
    from unittest.mock import patch

    from freetoken.server.args import parse_args

    hf = SimpleNamespace(to_dict=lambda: {"architectures": ["Qwen3VLForConditionalGeneration"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: hf):
        args, _ = parse_args([
            "--model", "/models/anon", "--image-min-tokens", "64", "--image-max-tokens", "1024",
            "--mm-processor-kwargs", '{"size": {"longest_edge": 4096}}',
        ])
        assert (args.mm.image_min_tokens, args.mm.image_max_tokens) == (64, 1024)
        assert args.mm.processor_kwargs == {"size": {"longest_edge": 4096}}
        assert parse_args(["--model", "/models/anon"])[0].mm.processor_kwargs == {}
        with pytest.raises(SystemExit):  # argparse reports the bad pair and exits
            parse_args(["--model", "/models/anon", "--image-min-tokens", "2048", "--image-max-tokens", "1024"])


# ---- error bodies and redirect policy (no network: httpx.MockTransport stands in for the remote) ----


def _mock_remote(monkeypatch, routes):
    """Serve ``routes`` ({url: (status, headers, body) | Exception}) through the client media.py builds;
    returns the list of URLs the transport actually saw."""
    import httpx

    seen = []

    def handler(request):
        seen.append(str(request.url))
        route = routes[str(request.url)]
        if isinstance(route, Exception):
            raise route
        status, headers, body = route
        return httpx.Response(status, headers=headers, content=body, request=request)

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    return seen


def _fetch(url, config):
    return asyncio.run(fetch_image_bytes([{"kind": "url", "data": url}], config))


def test_local_media_error_bodies_carry_no_path_root_or_errno(tmp_path):
    root = tmp_path / "media"
    root.mkdir()
    config = _config(allowed_local_media_path=str(root))
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"x")
    (root / "dir").mkdir()

    cases = {
        f"file://{outside}": "subpath",  # escapes the root
        f"file://{root / 'missing.png'}": "missing or unreadable",  # ENOENT
        f"file://{root / 'dir'}": "missing or unreadable",  # EISDIR
    }
    for url, expect in cases.items():
        with pytest.raises(ValueError, match=expect) as ei:
            _fetch(url, config)
        body = str(ei.value)
        assert str(tmp_path) not in body and "Errno" not in body, body
        assert "secret.png" not in body and "missing.png" not in body, body


def test_remote_fetch_failures_are_classified_without_library_text(monkeypatch):
    import httpx

    big = _MAX + 1
    seen = _mock_remote(monkeypatch, {
        "https://cdn.example.com/404.png": (404, {}, b""),
        "https://cdn.example.com/down.png": httpx.ConnectError("[Errno -2] Name or service not known"),
        "https://cdn.example.com/slow.png": httpx.ReadTimeout("timed out"),
        "https://cdn.example.com/declared.png": (200, {"content-length": str(big)}, b""),
        "https://cdn.example.com/streamed.png": (200, {}, b"\0" * big),
    })
    config = _config(allowed_media_domains="cdn.example.com")
    for name, expect in [("404", "HTTP 404"), ("down", "unreachable"), ("slow", "timed out"),
                         ("declared", "exceeds"), ("streamed", "exceeds")]:
        with pytest.raises(ValueError, match=expect) as ei:
            _fetch(f"https://cdn.example.com/{name}.png", config)
        assert "Errno" not in str(ei.value)
    assert len(seen) == 5


def test_redirects_are_rechecked_against_the_allowlist(monkeypatch):
    seen = _mock_remote(monkeypatch, {
        "https://cdn.example.com/a.png": (302, {"location": "https://evil.com/a.png"}, b""),
        "https://cdn.example.com/b.png": (302, {"location": "/real.png"}, b""),
        "https://cdn.example.com/real.png": (200, {}, b"fakepng"),
        "https://evil.com/a.png": (200, {}, b"pwned"),
    })
    config = _config(allowed_media_domains="cdn.example.com")
    with pytest.raises(ValueError, match="redirects outside the allowed domains") as ei:
        _fetch("https://cdn.example.com/a.png", config)
    assert "evil.com" not in str(ei.value)
    # a same-host relative redirect is followed and served
    assert _fetch("https://cdn.example.com/b.png", config) == [b"fakepng"]
    # the disallowed hop was never requested
    assert seen == ["https://cdn.example.com/a.png", "https://cdn.example.com/b.png", "https://cdn.example.com/real.png"]


@pytest.mark.parametrize("target", [
    "http://127.0.0.1/x", "http://10.0.0.1/x", "http://169.254.169.254/latest/meta-data",
    "http://[::1]/x", "http://[::ffff:10.0.0.1]/x", "http://localhost/x", "http://0.0.0.0/x",
    "http://100.64.0.1/x",
])
def test_redirects_to_private_or_loopback_addresses_are_refused(monkeypatch, target):
    seen = _mock_remote(monkeypatch, {
        "https://public.example.com/a.png": (302, {"location": target}, b""),
        target: (200, {}, b"internal"),
    })
    # even with an empty allowlist (any domain admitted) a redirect may not reach inside
    with pytest.raises(ValueError, match="private or loopback"):
        _fetch("https://public.example.com/a.png", _config())
    assert seen == ["https://public.example.com/a.png"]


def test_redirect_hop_count_and_scheme_are_bounded(monkeypatch):
    chain = {f"https://h.example.com/{i}.png": (302, {"location": f"https://h.example.com/{i + 1}.png"}, b"")
             for i in range(7)}
    chain["https://h.example.com/7.png"] = (200, {}, b"fakepng")
    chain["https://h.example.com/file.png"] = (302, {"location": "file:///etc/passwd"}, b"")
    seen = _mock_remote(monkeypatch, chain)
    with pytest.raises(ValueError, match="redirects more than"):
        _fetch("https://h.example.com/0.png", _config())
    assert len(seen) == 6  # the original plus five hops; the sixth hop is never requested
    with pytest.raises(ValueError, match="non-http"):
        _fetch("https://h.example.com/file.png", _config())
    # a short chain that stays public is fine
    assert _fetch("https://h.example.com/5.png", _config()) == [b"fakepng"]


def test_bad_base64_is_named_without_decoder_text():
    for ref in ({"kind": "b64", "data": "not base64!!"}, {"kind": "url", "data": "data:image/png;base64"}):
        with pytest.raises(ValueError, match="not valid base64"):
            asyncio.run(fetch_image_bytes([ref], _config()))
