from __future__ import annotations

import pytest

from freetoken.server.args import parse_args


@pytest.mark.parametrize(
    ("capacity_args", "expected_pages", "expected_tokens"),
    [
        (["--num-pages", "128"], 128, None),
        (["--num-tokens", "4096"], None, 4096),
    ],
)
def test_kv_cache_dtype_can_combine_with_capacity_override(
    capacity_args: list[str],
    expected_pages: int | None,
    expected_tokens: int | None,
) -> None:
    args, _ = parse_args(
        [
            "--model",
            "/models/anonymous",
            "--dtype",
            "bfloat16",
            "--kv-cache-dtype",
            "q8_0",
            *capacity_args,
        ]
    )

    assert args.kv_cache_dtype == "q8_0"
    assert args.num_page_override == expected_pages
    assert args.num_token_override == expected_tokens


def test_num_pages_and_num_tokens_remain_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--model",
                "/models/anonymous",
                "--dtype",
                "bfloat16",
                "--num-pages",
                "128",
                "--num-tokens",
                "4096",
            ]
        )
