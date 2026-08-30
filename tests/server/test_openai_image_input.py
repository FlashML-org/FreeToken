"""OpenAI image_url extraction stays aligned with generation's content markers."""

from freetoken.server.generation import render_messages
from freetoken.server.openai_api import _openai_image_urls


def test_openai_images_extract_in_rendered_marker_order() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                {"type": "text", "text": "second"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,BB=="}},
            ],
        }
    ]
    assert _openai_image_urls(messages) == [
        {"url": "data:image/png;base64,AA=="},
        {"url": "data:image/png;base64,BB=="},
    ]
    assert render_messages(messages)[0]["content"] == (
        "first<|freetoken-image|>second<|freetoken-image|>"
    )
