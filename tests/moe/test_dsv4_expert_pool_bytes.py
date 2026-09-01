"""DSV4-Flash-0731 expert-bank footprint from the in-tree ds_fp4 formula.

Does not invent tok/s. The ~137 GiB figure is bytes = layers × experts ×
_BANK_BYTES_PER_EXPERT['ds_fp4'](4096, 2048). Host RAM must cover that plus
runtime (user box ~192 GB).
"""

from freetoken.moe.offload_cache import _BANK_BYTES_PER_EXPERT


def test_dsv4_flash_0731_expert_pool_is_about_137_gib():
    # DeepSeek-V4-Flash-0731: dim=4096, moe_inter_dim=2048, 43 layers, 256 experts.
    H, I, layers, experts = 4096, 2048, 43, 256
    per_expert = _BANK_BYTES_PER_EXPERT["ds_fp4"](H, I)
    total = layers * experts * per_expert
    gib = total / (1 << 30)
    assert per_expert == 13_369_344
    assert total == 43 * 256 * 13_369_344
    assert 137.0 <= gib < 137.2
