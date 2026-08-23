from __future__ import annotations

from freetoken.server.args import parse_args


def test_parse_absolute_swa_token_capacity():
    args, run_shell = parse_args([
        "--model", "/tmp",
        "--dtype", "bfloat16",
        "--tool-call-parser", "llama3",
        "--reasoning-parser", "off",
        "--swa-num-tokens", "19200",
    ])

    assert run_shell is False
    assert args.swa_num_token_override == 19200
    # Conversion waits for model config resolution, where the SWA page unit is known.
    assert args.swa_num_pages_override is None


def test_rebuild_and_rejection_output_include_complete_prefill_geometry():
    from freetoken.control_cli import _decode_error_body, _format_rebuild

    doc = {
        "status": "rejected",
        "error": "requested cache does not fit",
        "requested_prefill_tokens": 24576,
        "pool_prefill_cap_tokens": 19200,
        "effective_prefill_tokens": 19200,
        "swa_capacity_source": "explicit",
        "prefill_limiting_reason": "swa_pool",
    }
    expected = (
        "prefill requested=24576 pool_cap=19200 effective=19200 "
        "source=explicit reason=swa_pool"
    )
    assert expected in _format_rebuild(doc)
    assert expected in _decode_error_body(__import__("json").dumps(doc).encode())
