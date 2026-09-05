from unittest.mock import patch

from freetoken.server.args import parse_args


class _Config:
    def to_dict(self) -> dict:
        return {"architectures": ["LlamaForCausalLM"], "torch_dtype": "bfloat16"}


def test_moe_collect_stats_flag_is_accepted():
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: _Config()):
        args, _ = parse_args(
            ["--model", "/models/anon", "--moe-collect-stats"]
        )
    assert args.moe_collect_stats is True
