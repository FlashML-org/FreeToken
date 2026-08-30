"""V-head order on the GGUF -> FreeToken boundary.

When a Gated DeltaNet has fewer K heads than V heads, llama.cpp's converter reorders every
tensor indexed by the V-head axis so ggml_repeat can stand in for an interleaved repeat:

    HF / FreeToken (grouped by K head):  [G0_v0, G0_v1, G1_v0, G1_v1, ...]
    GGUF           (tiled for ggml):     [G0_v0, G1_v0, ..., G0_v1, G1_v1, ...]

FreeToken's GDN pairs K head ``k`` with V heads ``[k*R, (k+1)*R)`` -- grouped. A GGUF load
that skips the inverse permutation therefore mispairs K and V in every GDN layer.

That failure is invisible to every structural check: names, shapes, dtypes, packed-ness and
activation magnitudes all stay correct, and the model simply emits nonsense. This suite is
positional rather than structural -- each V head is tagged with the index it must end up at,
so a missing or wrong permutation shows up as a permuted identity.

CPU-only, no network; the fixture writes a real .gguf with gguf.GGUFWriter.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

HID = 128
N_HEADS = 4
N_KV = 2
HEAD_DIM = 64  # rotary asserts head_size in {64, 128, 256, 512}
INTERVAL = 2
N_LAYERS = 2  # layer 0 GDN, layer 1 full attention
FFN = 256
VOCAB = 64

K_HEADS = 2  # ssm.group_count
V_HEADS = 6  # ssm.time_step_rank
K_HEAD_DIM = 16  # ssm.state_size
V_HEAD_DIM = 16
VALUE_DIM = V_HEADS * V_HEAD_DIM  # 96 == ssm.inner_size
KEY_DIM = K_HEADS * K_HEAD_DIM  # 32
CONV_DIM = 2 * KEY_DIM + VALUE_DIM  # 160
CONV_K = 4
R = V_HEADS // K_HEADS  # 3 V heads per K head

# K=2, R=3 makes tiled and grouped genuinely different orders (a permutation that is not
# the identity and not a simple reversal), so a half-applied fix cannot pass by accident.
assert K_HEADS != V_HEADS and R > 1


def _grouped_index_of_tiled_slot(t: int) -> int:
    """The grouped position that GGUF slot ``t`` must land on.

    llama.cpp's tiled axis runs r-major: slot ``t`` holds (r, k) = (t // K, t % K), whose
    grouped position is ``k*R + r``.
    """
    r, k = divmod(t, K_HEADS)
    return k * R + r


TILED_HEAD_IDS = np.array(
    [_grouped_index_of_tiled_slot(t) for t in range(V_HEADS)], dtype=np.float32
)
TAG_OFFSET = np.float32(0.1234567)  # deliberately not exactly representable as bf16
TILED_TAGS = TILED_HEAD_IDS + TAG_OFFSET
# Sanity: writing these tags in file order and un-tiling must produce 0..V_HEADS-1.
assert sorted(TILED_HEAD_IDS.tolist()) == list(range(V_HEADS))
assert TILED_HEAD_IDS.tolist() != list(range(V_HEADS)), "tags must not already be the identity"


def _ensure_tp1() -> None:
    from freetoken.distributed import get_tp_info, set_tp_info

    try:
        get_tp_info()
    except RuntimeError:
        set_tp_info(rank=0, size=1)


def _f32(writer, name: str, data: np.ndarray) -> None:
    writer.add_tensor(name, np.ascontiguousarray(data, dtype=np.float32))


def _q8(writer, name: str, data: np.ndarray) -> None:
    import gguf
    from gguf.quants import quantize

    packed = quantize(np.ascontiguousarray(data, dtype=np.float32),
                      gguf.GGMLQuantizationType.Q8_0)
    writer.add_tensor(name, packed, raw_dtype=gguf.GGMLQuantizationType.Q8_0)


def _build(path) -> None:
    """A minimal qwen35 file whose V-indexed tensors are written in llama.cpp tiled order."""
    import gguf

    rng = np.random.default_rng(7)
    w = gguf.GGUFWriter(str(path), "qwen35")

    w.add_uint32("qwen35.block_count", N_LAYERS)
    w.add_uint32("qwen35.full_attention_interval", INTERVAL)
    w.add_uint32("qwen35.embedding_length", HID)
    w.add_uint32("qwen35.feed_forward_length", FFN)
    w.add_uint32("qwen35.attention.head_count", N_HEADS)
    w.add_uint32("qwen35.attention.head_count_kv", N_KV)
    w.add_uint32("qwen35.attention.key_length", HEAD_DIM)
    w.add_uint32("qwen35.attention.value_length", HEAD_DIM)
    w.add_float32("qwen35.attention.layer_norm_rms_epsilon", 1e-6)
    w.add_uint32("qwen35.context_length", 4096)
    w.add_float32("qwen35.rope.freq_base", 10_000_000.0)
    w.add_uint32("qwen35.rope.dimension_count", HEAD_DIM // 4)
    w.add_uint32("qwen35.ssm.group_count", K_HEADS)
    w.add_uint32("qwen35.ssm.time_step_rank", V_HEADS)
    w.add_uint32("qwen35.ssm.state_size", K_HEAD_DIM)
    w.add_uint32("qwen35.ssm.inner_size", VALUE_DIM)
    w.add_uint32("qwen35.ssm.conv_kernel", CONV_K)

    _q8(w, "token_embd.weight", rng.standard_normal((VOCAB, HID)))
    _f32(w, "output_norm.weight", rng.standard_normal((HID,)))
    _q8(w, "output.weight", rng.standard_normal((VOCAB, HID)))

    for lid in range(N_LAYERS):
        p = f"blk.{lid}."
        _f32(w, p + "attn_norm.weight", rng.standard_normal((HID,)))
        _f32(w, p + "post_attention_norm.weight", rng.standard_normal((HID,)))
        _q8(w, p + "ffn_gate.weight", rng.standard_normal((FFN, HID)))
        _q8(w, p + "ffn_up.weight", rng.standard_normal((FFN, HID)))
        _q8(w, p + "ffn_down.weight", rng.standard_normal((HID, FFN)))

        if (lid + 1) % INTERVAL == 0:  # full attention
            _q8(w, p + "attn_q.weight", rng.standard_normal((N_HEADS * HEAD_DIM * 2, HID)))
            _q8(w, p + "attn_k.weight", rng.standard_normal((N_KV * HEAD_DIM, HID)))
            _q8(w, p + "attn_v.weight", rng.standard_normal((N_KV * HEAD_DIM, HID)))
            _q8(w, p + "attn_output.weight", rng.standard_normal((HID, N_HEADS * HEAD_DIM)))
            _f32(w, p + "attn_q_norm.weight", rng.standard_normal((HEAD_DIM,)))
            _f32(w, p + "attn_k_norm.weight", rng.standard_normal((HEAD_DIM,)))
            continue

        # ---- GDN layer: every V-indexed tensor is written TILED ----
        # ssm_a holds -exp(A_log); the loader recovers A_log = log(-a), so tag through exp.
        _f32(w, p + "ssm_a", -np.exp(TILED_TAGS))
        _f32(w, p + "ssm_dt.bias", TILED_TAGS)

        # attn_gate is the z half of in_proj: V_HEADS blocks of V_HEAD_DIM rows.
        gate = np.repeat(TILED_TAGS, V_HEAD_DIM)[:, None] * np.ones((1, HID), dtype=np.float32)
        _q8(w, p + "attn_gate.weight", gate)

        # attn_qkv is q|k|v: only the trailing V rows are reordered. Give q/k rows their
        # own stable tags so the test proves the adapter did not move them accidentally.
        qk_tags = np.arange(1, 2 * KEY_DIM + 1, dtype=np.float32)
        qk = qk_tags[:, None] * np.ones((1, HID), dtype=np.float32)
        vpart = np.repeat(TILED_TAGS, V_HEAD_DIM)[:, None] * np.ones((1, HID), dtype=np.float32)
        _q8(w, p + "attn_qkv.weight", np.concatenate([qk, vpart], axis=0))

        _q8(w, p + "ssm_beta.weight", np.repeat(TILED_TAGS, 1)[:, None] * np.ones((1, HID), dtype=np.float32))
        _q8(w, p + "ssm_alpha.weight", np.repeat(TILED_TAGS, 1)[:, None] * np.ones((1, HID), dtype=np.float32))
        # F32 on purpose: ssm_out needs its COLUMNS un-tiled, which cannot be done on
        # packed blocks, so the quantized path routes through a CUDA-only dense dequant
        # and the load would need a GPU. Unquantized keeps this suite CPU-only.
        out_cols = np.ones((HID, 1), dtype=np.float32) * np.repeat(
            TILED_TAGS, V_HEAD_DIM
        )[None, :]
        _f32(w, p + "ssm_out.weight", out_cols)
        _f32(w, p + "ssm_norm.weight", rng.standard_normal((V_HEAD_DIM,)))
        conv_qk = rng.standard_normal((2 * KEY_DIM, CONV_K))
        conv_v = np.repeat(TILED_TAGS, V_HEAD_DIM)[:, None] * np.ones(
            (1, CONV_K), dtype=np.float32
        )
        _f32(w, p + "ssm_conv1d.weight", np.concatenate([conv_qk, conv_v], axis=0))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    """Weights as the production path yields them, keyed by param name."""
    import importlib

    from freetoken.models.gguf.config import build_gguf_shim
    from freetoken.models.register import get_model_spec

    _ensure_tp1()
    path = tmp_path_factory.mktemp("vuntile") / "tiny-qwen35.gguf"
    _build(path)

    shim = build_gguf_shim(str(path))
    spec = get_model_spec(shim.architectures[0])
    mod = importlib.import_module(spec.module)
    getattr(mod, spec.parse_config)(shim)  # some adapters stash geometry on the config
    return dict(
        getattr(mod, spec.iter_weights)(
            str(path), torch.device("cpu"),
            # A dense Qwen3.5/3.8 load asks for all resident weights. Keep the fixture on
            # that production contract instead of the MoE/offload-only call shape.
            include_moe_experts=True, include_non_moe=True,
        )
    )


def _per_head(t: torch.Tensor, head_dim: int) -> list[float]:
    """Collapse a [V_HEADS*head_dim, ...] tensor to one representative value per V head."""
    flat = t.reshape(V_HEADS, head_dim, -1) if t.dim() > 1 else t.reshape(V_HEADS, head_dim)
    return [float(flat[i].flatten()[0]) for i in range(V_HEADS)]


def test_a_log_is_regrouped(weights):
    """Per-head scalars: the tags must come back as a plain 0..V-1 run."""
    a_log = weights["model.layers.0.linear_attn.A_log"]
    assert a_log.shape == (V_HEADS,)
    got = [round(float(x)) for x in a_log]
    assert got == list(range(V_HEADS)), (
        f"A_log still in file order {got}; expected grouped {list(range(V_HEADS))}"
    )


def test_dt_bias_is_regrouped(weights):
    dt = weights["model.layers.0.linear_attn.dt_bias"]
    assert dt.shape == (V_HEADS,)
    got = [round(float(x)) for x in dt]
    assert got == list(range(V_HEADS)), f"dt_bias still in file order {got}"


def test_gating_params_keep_fp32_precision(weights):
    """A_log/dt_bias must not round through bf16 before reaching their fp32 params."""
    expected = torch.arange(V_HEADS, dtype=torch.float32) + float(TAG_OFFSET)
    a_log = weights["model.layers.0.linear_attn.A_log"]
    dt = weights["model.layers.0.linear_attn.dt_bias"]
    assert a_log.dtype == torch.float32
    assert dt.dtype == torch.float32
    torch.testing.assert_close(a_log, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(dt, expected, rtol=0, atol=0)
    assert not torch.equal(expected.to(torch.bfloat16).float(), expected)


def _rows_to_f32(t: torch.Tensor, lo: int, hi: int, in_features: int) -> torch.Tensor:
    """Rows [lo, hi) of a param as f32, packed or not.

    Q8_0 is decoded here rather than through the repo's ``dequantize``: that helper is a
    reference path whose type coverage differs between branches, and this suite should run
    against any implementation of the adapter.
    """
    rows = t[lo:hi]
    if rows.dtype != torch.uint8:
        return rows.float()

    # Q8_0 block: fp16 scale (2 bytes) + 32 int8 values.
    blocks = in_features // 32
    raw = rows.reshape(hi - lo, blocks, 34)
    scales = raw[:, :, :2].contiguous().view(torch.float16).float()  # [rows, blocks, 1]
    qs = raw[:, :, 2:].view(torch.int8).float()  # [rows, blocks, 32]
    return (qs * scales).reshape(hi - lo, in_features)


def test_in_proj_z_half_is_regrouped(weights):
    """The z (gate) half of in_proj holds V_HEAD_DIM rows per V head."""
    key = "model.layers.0.linear_attn.in_proj"
    name = f"{key}.qweight" if f"{key}.qweight" in weights else f"{key}.weight"
    assert name in weights, f"no fused in_proj; got {sorted(weights)[:12]}"

    z = _rows_to_f32(weights[name], CONV_DIM, CONV_DIM + VALUE_DIM, HID)
    got = [round(float(z[i * V_HEAD_DIM][0])) for i in range(V_HEADS)]
    assert got == list(range(V_HEADS)), f"in_proj z half still tiled: {got}"


def test_in_proj_qkv_v_rows_are_regrouped(weights):
    """attn_qkv contributes q|k|v; only its trailing V-head blocks must move."""
    key = "model.layers.0.linear_attn.in_proj"
    name = f"{key}.qweight" if f"{key}.qweight" in weights else f"{key}.weight"
    v = _rows_to_f32(weights[name], 2 * KEY_DIM, CONV_DIM, HID)
    got = [round(float(v[i * V_HEAD_DIM][0])) for i in range(V_HEADS)]
    assert got == list(range(V_HEADS)), f"in_proj qkv V rows still tiled: {got}"


def test_in_proj_beta_and_alpha_are_regrouped(weights):
    """beta and alpha are one row per V head, immediately after the z half."""
    key = "model.layers.0.linear_attn.in_proj"
    name = f"{key}.qweight" if f"{key}.qweight" in weights else f"{key}.weight"
    base = CONV_DIM + VALUE_DIM

    beta = _rows_to_f32(weights[name], base, base + V_HEADS, HID)
    alpha = _rows_to_f32(weights[name], base + V_HEADS, base + 2 * V_HEADS, HID)
    assert [round(float(beta[i][0])) for i in range(V_HEADS)] == list(range(V_HEADS))
    assert [round(float(alpha[i][0])) for i in range(V_HEADS)] == list(range(V_HEADS))


def test_qk_half_of_in_proj_is_left_alone(weights):
    """Only the V-indexed rows move. Permuting the q/k half too would be just as wrong and
    just as invisible, so pin that it is untouched."""
    key = "model.layers.0.linear_attn.in_proj"
    name = f"{key}.qweight" if f"{key}.qweight" in weights else f"{key}.weight"
    t = weights[name]
    assert t.shape[0] == CONV_DIM + VALUE_DIM + 2 * V_HEADS, (
        f"unexpected in_proj height {t.shape[0]}"
    )
    qk = _rows_to_f32(t, 0, 2 * KEY_DIM, HID)
    got = [round(float(row[0])) for row in qk]
    assert got == list(range(1, 2 * KEY_DIM + 1)), f"q/k rows were moved: {got}"


def test_conv1d_v_channels_are_regrouped(weights):
    """conv1d channels are q|k|v, so only the trailing V channel blocks move."""
    w = weights["model.layers.0.linear_attn.conv1d.weight"]
    assert w.shape == (CONV_DIM, 1, CONV_K)
    v = w[2 * KEY_DIM :, 0, :].float()
    got = [round(float(v[i * V_HEAD_DIM][0])) for i in range(V_HEADS)]
    assert got == list(range(V_HEADS)), f"conv1d V channels still tiled: {got}"


def test_out_proj_v_columns_are_regrouped(weights):
    """ssm_out consumes V heads along columns, unlike every row-wise packed group."""
    key = "model.layers.0.linear_attn.out_proj"
    name = f"{key}.qweight" if f"{key}.qweight" in weights else f"{key}.weight"
    w = _rows_to_f32(weights[name], 0, HID, VALUE_DIM)
    assert w.shape == (HID, VALUE_DIM)
    got = [round(float(w[0, i * V_HEAD_DIM])) for i in range(V_HEADS)]
    assert got == list(range(V_HEADS)), f"out_proj V columns still tiled: {got}"


def test_q8_requantization_is_bounded_and_preserves_values():
    """The memory-safe ssm_out path keeps only Q8_0 and has ordinary Q8 error."""
    from freetoken.models.qwen3_5_moe import gguf as qwen_gguf

    source = torch.linspace(-5.0, 5.0, steps=2 * 64, dtype=torch.float32).reshape(2, 64)
    source[0, :32] = 0  # pin the zero-scale edge case
    packed = qwen_gguf._requant_q8_0(source)
    assert packed.dtype == torch.uint8
    assert packed.shape == (2, 2 * 34)

    decoded = _rows_to_f32(packed, 0, 2, 64)
    torch.testing.assert_close(decoded, source, rtol=0, atol=0.04)


def test_untile_is_not_applied_when_k_equals_v():
    """The permutation is only correct when K != V; applying it to a K == V model would
    corrupt a checkpoint llama.cpp never reordered."""
    from freetoken.models.qwen3_5_moe import gguf as moe_gguf

    ungroup = getattr(moe_gguf, "_ungroup_v", None)
    if ungroup is None:
        pytest.skip("this adapter exposes no _ungroup_v helper")
    t = torch.arange(8, dtype=torch.float32)
    # K == V means R == 1, and the permutation must be the identity.
    torch.testing.assert_close(ungroup(t, 0, 8, 1, 1), t)
