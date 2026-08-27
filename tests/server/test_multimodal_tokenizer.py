from __future__ import annotations

import base64

import pytest

from freetoken.tokenizer.server import _download_image, _message_image_sources


def test_openai_image_data_url_is_extracted_and_decoded() -> None:
    raw = b"small-image-payload"
    source = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": source}},
                {"type": "text", "text": "describe"},
            ],
        }
    ]

    assert _message_image_sources(messages) == [source]
    assert _download_image(source) == raw


def test_local_file_image_url_is_rejected() -> None:
    with pytest.raises(ValueError, match="data, http, or https"):
        _download_image("file:///C:/secret.png")
