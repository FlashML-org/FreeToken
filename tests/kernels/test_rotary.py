import torch


def test_yarn_rope_scales_cos_sin_by_attention_factor():
    import math
    from freetoken.layers.rotary import get_rope

    get_rope.cache_clear()
    head_dim = rotary_dim = 64
    base = 150000.0
    factor = 32.0
    beta_fast, beta_slow = 32.0, 1.0
    orig_max_pos = 4096
    max_position = 8  # large enough to inspect a position past 0, where freqs != 0
    rope = get_rope(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=max_position,
        base=base,
        rope_scaling=(
            ("rope_type", "yarn"),
            ("factor", factor),
            ("original_max_position_embeddings", orig_max_pos),
            ("beta_fast", beta_fast),
            ("beta_slow", beta_slow),
            ("truncate", False),
        ),
    )

    attention_factor = 0.1 * math.log(factor) + 1.0
    half = rotary_dim // 2
    cos, sin = rope._cos_sin_cache[:, :half], rope._cos_sin_cache[:, half:]

    # position 0: freqs = 0*inv_freq is all zeros, so cos == attention_factor and sin == 0
    # everywhere. This alone never touches the correction-dim ramp/blend below.
    torch.testing.assert_close(cos[0], torch.full_like(cos[0], attention_factor))
    torch.testing.assert_close(sin[0], torch.zeros_like(sin[0]))

    # Independent YaRN reference at a nonzero position. The blend keeps the low (fast-rotating)
    # dims at the raw inv_freq (extrapolation) and pulls the high (slow-rotating) dims down to
    # inv_freq/factor (interpolation), ramping linearly over [low, high] correction dims. Written
    # from the YaRN definition, not lifted from rotary.py's post_process.
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))

    def correction_dim(num_rotations: float) -> float:
        return (
            rotary_dim
            * math.log(orig_max_pos / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    low = max(correction_dim(beta_fast), 0.0)        # fast beta -> smaller dim (~8.1)
    high = min(correction_dim(beta_slow), half - 1)  # slow beta -> larger dim (~17.4)
    dims = torch.arange(half, dtype=torch.float32)
    ramp = torch.clamp((dims - low) / max(high - low, 1.0), 0.0, 1.0)
    blended = inv_freq * (1.0 - ramp) + (inv_freq / factor) * ramp

    # The ramp must actually span extrapolation (0), a genuine blend, and interpolation (1);
    # otherwise the position assertions below would not exercise the low/high bounds.
    assert (ramp == 0).any() and (ramp == 1).any() and ((ramp > 0) & (ramp < 1)).any()

    pos = 5
    freqs = pos * blended
    torch.testing.assert_close(cos[pos], torch.cos(freqs) * attention_factor, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(sin[pos], torch.sin(freqs) * attention_factor, rtol=1e-5, atol=1e-6)


def test_proportional_rope_zero_pads_non_rotated_pairs_and_is_cached():
    from freetoken.layers.rotary import get_rope

    get_rope.cache_clear()
    rope_scaling = (("rope_type", "proportional"),)
    rope = get_rope(
        head_dim=64,
        rotary_dim=16,
        max_position=4,
        base=10000.0,
        rope_scaling=rope_scaling,
    )
    same_rope = get_rope(
        head_dim=64,
        rotary_dim=16,
        max_position=4,
        base=10000.0,
        rope_scaling=rope_scaling,
    )

    assert same_rope is rope
    cache = rope._cos_sin_cache
    assert cache.shape == (4, 64)

    cos, sin = cache[:, :32], cache[:, 32:]
    torch.testing.assert_close(cos[:, 8:], torch.ones_like(cos[:, 8:]))
    torch.testing.assert_close(sin[:, 8:], torch.zeros_like(sin[:, 8:]))
    assert not torch.allclose(cos[1, :8], torch.ones_like(cos[1, :8]))


def test_yarn_correction_range_matches_hf_where_the_clamp_and_gap_bind():
    """The ramp bounds must follow HF's find_correction_range (dim-1, and a nudge when the
    endpoints coincide). Clamping to rotary_dim//2 - 1 instead silently over-interpolates the
    longest-wavelength dims; flooring the gap at 1 flattens the ramp for a sub-1 gap."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    from freetoken.layers.rotary import _rope_frequencies

    class _Shim:  # duck-typed config for HF's real _compute_yarn_parameters
        def __init__(self, head_dim, max_position, rope_parameters):
            self.head_dim = self.hidden_size = head_dim
            self.num_attention_heads = 1
            self.max_position_embeddings = max_position
            self.rope_parameters = rope_parameters

        def standardize_rope_params(self):
            pass

    max_position = 512
    cases = [  # (rotary_dim, base, scaling) -- each binds a different branch
        (128, 1e4, {"factor": 16.0, "beta_fast": 32.0, "beta_slow": 1.0,
                    "original_max_position_embeddings": 131072, "truncate": True}),
        (128, 1e6, {"factor": 16.0, "beta_fast": 1.0, "beta_slow": 1.0,
                    "original_max_position_embeddings": 32768, "truncate": False}),
        (128, 1e6, {"factor": 16.0, "beta_fast": 1.15, "beta_slow": 1.0,
                    "original_max_position_embeddings": 32768, "truncate": False}),
    ]
    def table(inv_freq: torch.Tensor, attention_factor: float) -> torch.Tensor:
        freqs = torch.outer(torch.arange(max_position, dtype=torch.float), inv_freq.float())
        return torch.cat((freqs.cos() * attention_factor, freqs.sin() * attention_factor), dim=-1)

    # The frequencies, not get_rope's table: YaRN sizes that to original * factor (up to 2M
    # rows here), and the correction range is a property of the frequencies alone.
    for rotary_dim, base, scaling in cases:
        ours = table(*_rope_frequencies(rotary_dim, rotary_dim, base, {"rope_type": "yarn", **scaling}))
        params = {"rope_type": "yarn", "rope_theta": base, "partial_rotary_factor": 1.0, **scaling}
        expected = table(*ROPE_INIT_FUNCTIONS["yarn"](
            _Shim(rotary_dim, max_position, params), device=torch.device("cpu")))
        torch.testing.assert_close(ours, expected, rtol=0, atol=1e-6)


def test_yarn_mscale_all_dim_zero_falls_back_like_hf():
    # HF treats mscale_all_dim: 0 as unset (truthiness check): attention_factor falls
    # back to get_mscale(factor). Keying on presence instead divided by get_mscale(f, 0)
    # == 1.0 and scaled attention by mscale alone (~7% off for DeepSeek-lineage configs).
    import math

    from freetoken.layers.rotary import get_rope

    get_rope.cache_clear()
    factor = 16.0
    rope = get_rope(
        head_dim=64,
        rotary_dim=64,
        max_position=8,
        base=10000.0,
        rope_scaling=(
            ("rope_type", "yarn"),
            ("factor", factor),
            ("original_max_position_embeddings", 4096),
            ("mscale", 0.707),
            ("mscale_all_dim", 0.0),
        ),
    )
    expected = 0.1 * math.log(factor) + 1.0  # 1.277, not 1.196 (= 1 + 0.1*0.707*ln 16)
    cos = rope._cos_sin_cache[:, :32]
    torch.testing.assert_close(cos[0], torch.full_like(cos[0], expected))


def _yarn(factor: float, original: int) -> tuple:
    return (("rope_type", "yarn"), ("factor", factor), ("original_max_position_embeddings", original))


def test_yarn_table_covers_the_extended_context():
    # A checkpoint extended by overriding only its rope parameters (Qwen3.8's 1M recipe) keeps
    # max_position_embeddings at the trained length; YaRN serves original * factor positions.
    from freetoken.layers.rotary import get_rope

    get_rope.cache_clear()
    rope = get_rope(head_dim=64, rotary_dim=64, max_position=64, base=10000.0, rope_scaling=_yarn(4.0, 64))
    assert rope._cos_sin_cache.shape == (256, 64)

    longer = get_rope(head_dim=64, rotary_dim=64, max_position=512, base=10000.0, rope_scaling=_yarn(4.0, 64))
    assert longer._cos_sin_cache.shape == (512, 64)
    torch.testing.assert_close(longer._cos_sin_cache[:256], rope._cos_sin_cache, rtol=0, atol=0)


def test_one_table_serves_every_rope_of_the_same_frequencies():
    # The QSA indexer ropes 128-wide heads with the attention's 64 rotary dims, and mrope reads
    # the same rows per axis: one table, not a second copy of a 1M-row fp32 tensor.
    from freetoken.layers.rotary import MRotaryEmbedding, get_rope

    get_rope.cache_clear()
    attention = get_rope(head_dim=256, rotary_dim=64, max_position=64, base=1e7, rope_scaling=_yarn(4.0, 64))
    indexer = get_rope(head_dim=128, rotary_dim=64, max_position=64, base=1e7, rope_scaling=_yarn(4.0, 64))
    mrope = get_rope(
        head_dim=256, rotary_dim=64, max_position=64, base=1e7, rope_scaling=_yarn(4.0, 64),
        mrope_section=(11, 11, 10), mrope_layout="interleaved",
    )
    assert indexer.head_size == 128 and isinstance(mrope, MRotaryEmbedding)
    assert attention._cos_sin_cache.shape == (256, 64)
    table = attention._cos_sin_cache.data_ptr()
    assert indexer._cos_sin_cache.data_ptr() == table and mrope._cos_sin_cache.data_ptr() == table

    proportional = (("rope_type", "proportional"),)
    wide = get_rope(head_dim=128, rotary_dim=64, max_position=4, base=1e4, rope_scaling=proportional)
    narrow = get_rope(head_dim=64, rotary_dim=64, max_position=4, base=1e4, rope_scaling=proportional)
    assert wide._cos_sin_cache.shape == (4, 128) and narrow._cos_sin_cache.shape == (4, 64)
