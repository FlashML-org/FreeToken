"""``_compressed_linear_storage`` (weight_map sniffing for mixed compressed-tensors
exports, e.g. unsloth/Qwen3.8-27B-NVFP4) and the per-layer dense-MLP storage gate.

The dense pass keeps each linear in the storage the checkpoint actually uses: native
W4A16 for packed layers, native W8A16 for fp8 attention linears, bf16 dequant for the
rest. When the safetensors index is absent the checkpoint is an FTW dir (``ft
checkpoint`` output) if ``freetoken_weight.json`` is there -- the tuple is then
re-derived from its per-tensor dtypes; the official dense-NVFP4 assumption (all packed,
no overrides) is only the last-resort fallback (hub id, single-file checkpoint)."""

import json
import os
import tempfile
import unittest

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.models.qwen3_5_moe.config import _compressed_linear_storage

if try_get_tp_info() is None:
    set_tp_info(rank=0, size=1)


class _FakeHFConfig:
    def __init__(self, name_or_path=None):
        self.name_or_path = name_or_path


def _write_index(tmpdir, keys):
    with open(os.path.join(tmpdir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {}, "weight_map": {k: "shard.bin" for k in keys}}, f)


def _write_ftw(tmpdir, tensors):
    """Minimal FTW index: one {name, kind, dtype} entry per tensor."""
    with open(os.path.join(tmpdir, "freetoken_weight.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {},
                "tensors": [{"name": n, "kind": "weight", "dtype": d} for n, d in tensors],
                "shards": [],
            },
            f,
        )


def _ftw_mixed_tensors(
    nvfp4_layers, fp8_layers, *, attn="fp8", lm_head="bfloat16", n_layers=64
):
    """FTW tensors of a converted mixed export (model-side names, dtypes as written).
    fp8 attention registers the split ``in_proj_qkvz``; nvfp4 attention registers the
    fused bf16 ``in_proj`` + packed ``out_proj``. Scale siblings are written too --
    the .weight name filter must exclude them."""
    out = []
    for layer in range(n_layers):
        if attn == "fp8":
            out.append((f"model.layers.{layer}.linear_attn.in_proj_qkvz.weight", "float8_e4m3fn"))
            out.append((f"model.layers.{layer}.linear_attn.in_proj_qkvz.weight_scale", "float32"))
        else:
            out.append((f"model.layers.{layer}.linear_attn.in_proj.weight", "bfloat16"))
            out.append((f"model.layers.{layer}.linear_attn.out_proj.weight", "uint8"))
            out.append((f"model.layers.{layer}.linear_attn.out_proj.weight_scale", "float8_e4m3fn"))
        dt = (
            "uint8"
            if layer in nvfp4_layers
            else "float8_e4m3fn" if layer in fp8_layers else "bfloat16"
        )
        out.append((f"model.layers.{layer}.mlp.gate_up_proj.weight", dt))
        out.append((f"model.layers.{layer}.mlp.down_proj.weight", dt))
        if dt == "uint8":
            out.append((f"model.layers.{layer}.mlp.gate_up_proj.weight_scale", "float8_e4m3fn"))
        elif dt == "float8_e4m3fn":
            out.append((f"model.layers.{layer}.mlp.gate_up_proj.weight_scale", "float32"))
    out.append(("lm_head.weight", lm_head))
    return out


class _CompressedLinearStorageTest(unittest.TestCase):
    def test_mixed_unsloth_layout(self):
        # unsloth/Qwen3.8-27B-NVFP4: attention linears FP8 (weight + scale), most mlp
        # NVFP4 (weight_packed), layers 56-63 mlp FP8 (weight + scale), lm_head FP8.
        keys = []
        for layer in range(64):
            keys += [
                f"model.language_model.layers.{layer}.self_attn.q_proj.weight",
                f"model.language_model.layers.{layer}.self_attn.q_proj.weight_scale",
                f"model.language_model.layers.{layer}.linear_attn.out_proj.weight",
                f"model.language_model.layers.{layer}.linear_attn.out_proj.weight_scale",
            ]
            if layer < 56:
                keys.append(f"model.language_model.layers.{layer}.mlp.gate_proj.weight_packed")
            else:
                keys.append(f"model.language_model.layers.{layer}.mlp.gate_proj.weight")
                keys.append(f"model.language_model.layers.{layer}.mlp.gate_proj.weight_scale")
        keys.append("lm_head.weight")
        keys.append("lm_head.weight_scale")
        with tempfile.TemporaryDirectory() as tmp:
            _write_index(tmp, keys)
            attn, dense_fallback, overrides, lmhead_fp8 = _compressed_linear_storage(
                _FakeHFConfig(tmp)
            )
            self.assertEqual(attn, "fp8")
            self.assertEqual(dense_fallback, "nvfp4")
            self.assertEqual(overrides, {l: "fp8" for l in range(56, 64)})
            self.assertTrue(lmhead_fp8)

    def test_official_dense_nvfp4_kept_native(self):
        keys = [
            "model.layers.0.self_attn.q_proj.weight_packed",
            "model.layers.0.self_attn.q_proj.weight_scale",
            "model.layers.0.linear_attn.out_proj.weight_packed",
            "model.layers.0.mlp.gate_proj.weight_packed",
            "model.layers.0.mlp.up_proj.weight_packed",
            "model.layers.0.mlp.down_proj.weight_packed",
            "lm_head.weight",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            _write_index(tmp, keys)
            self.assertEqual(
                _compressed_linear_storage(_FakeHFConfig(tmp)),
                ("nvfp4", "nvfp4", None, False),
            )

    def test_routed_expert_moe_naming_keeps_native(self):
        # MoE checkpoint: dense linears live under mlp.shared_expert / experts -- no
        # mlp.{gate,up,down}_proj keys at all -> no overrides, native assumption.
        keys = [
            "model.layers.0.self_attn.q_proj.weight_packed",
            "model.layers.0.mlp.shared_expert.gate_proj.weight_packed",
            "model.layers.0.mlp.experts.0.gate_proj.weight_packed",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            _write_index(tmp, keys)
            self.assertEqual(
                _compressed_linear_storage(_FakeHFConfig(tmp)),
                ("nvfp4", "nvfp4", None, False),
            )

    def test_index_unavailable_falls_back_to_native(self):
        # Hub id before download (name_or_path not a local dir) and missing index file.
        self.assertEqual(
            _compressed_linear_storage(_FakeHFConfig("unsloth/Qwen3.8-27B-NVFP4")),
            ("nvfp4", "nvfp4", None, False),
        )
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                _compressed_linear_storage(_FakeHFConfig(tmp)),
                ("nvfp4", "nvfp4", None, False),
            )

    # --- FTW fallback: the ft checkpoint -> ft serve round-trip (reported on PR #208) ----
    # An FTW dir carries no safetensors index; the tuple is re-derived from the
    # freetoken_weight.json tensor names + dtypes instead of silently defaulting.

    def _ftw(self, tensors):
        with tempfile.TemporaryDirectory() as tmp:
            _write_ftw(tmp, tensors)
            return _compressed_linear_storage(_FakeHFConfig(tmp))

    def test_ftw_mixed_roundtrip(self):
        # The unsloth model of test_mixed_unsloth_layout, converted to FTW. Before the
        # dtype fallback the silent default misread attention as nvfp4 and the loader
        # asked for the fused bf16 in_proj key the FTW never wrote (KeyError).
        attn, dense, overrides, lmhead = self._ftw(
            _ftw_mixed_tensors(range(0, 56), range(56, 64), attn="fp8")
        )
        self.assertEqual((attn, dense), ("fp8", "nvfp4"))
        self.assertEqual(overrides, {l: "fp8" for l in range(56, 64)})
        self.assertFalse(lmhead)

    def test_ftw_mixed_fp8_lmhead(self):
        result = self._ftw(
            _ftw_mixed_tensors(
                range(0, 56), range(56, 64), attn="fp8", lm_head="float8_e4m3fn"
            )
        )
        self.assertTrue(result[3])

    def test_ftw_three_tier_dense(self):
        attn, dense, overrides, lmhead = self._ftw(
            _ftw_mixed_tensors(
                range(0, 32), range(32, 48), attn="fp8", lm_head="float8_e4m3fn"
            )
        )
        self.assertEqual((attn, dense), ("fp8", "nvfp4"))
        expected = {l: "fp8" for l in range(32, 48)}
        expected |= {l: "bf16" for l in range(48, 64)}
        self.assertEqual(overrides, expected)
        self.assertTrue(lmhead)

    def test_ftw_all_fp8_dense(self):
        attn, dense, overrides, lmhead = self._ftw(
            _ftw_mixed_tensors((), range(64), attn="fp8")
        )
        self.assertEqual((attn, dense), ("fp8", "none"))
        self.assertEqual(overrides, {l: "fp8" for l in range(64)})

    def test_ftw_pure_nvfp4_matches_native(self):
        # Regression guard: official NVFP4 FTWs loaded through the old silent default;
        # the derivation must reproduce that exact tuple.
        self.assertEqual(
            self._ftw(_ftw_mixed_tensors(range(64), (), attn="nvfp4")),
            ("nvfp4", "nvfp4", None, False),
        )

    def test_ftw_routed_moe_names_ignored(self):
        tensors = [
            ("model.layers.0.linear_attn.in_proj_qkvz.weight", "float8_e4m3fn"),
            ("model.layers.0.mlp.shared_expert.gate_up_proj.weight", "uint8"),
            ("model.layers.0.mlp.experts.bank_gate_up_proj.weight", "uint8"),
        ]
        self.assertEqual(self._ftw(tensors), ("fp8", "nvfp4", None, False))

    def test_ftw_broken_index_falls_back_to_native(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "freetoken_weight.json"), "w", encoding="utf-8") as f:
                f.write("{not json")
            self.assertEqual(
                _compressed_linear_storage(_FakeHFConfig(tmp)),
                ("nvfp4", "nvfp4", None, False),
            )


class _FakeConfig:
    def __init__(self, **kw):
        self.expert_quant = kw.get("expert_quant", "none")
        self.dense_quant = kw.get("dense_quant", "none")
        self.dense_mlp_storage = kw.get("dense_mlp_storage", None)


class _SharedExpertStorageTest(unittest.TestCase):
    """The per-layer dense-MLP storage gate: the dense_mlp_storage override wins over
    the dense_quant flag for the layers it covers."""

    def _expert(self, layer_id=None, **kw):
        from freetoken.models.qwen3_5_moe.moe import _SharedExpert

        # in_features must be %16 for the NVFP4 dense kernels
        return _SharedExpert(_FakeConfig(**kw), 16, 32, layer_id)

    def test_per_layer_bf16_override_wins(self):
        from freetoken.layers import LinearColParallelMerged

        e = self._expert(
            layer_id=0, dense_quant="nvfp4", dense_mlp_storage={0: "bf16"}
        )
        self.assertIsInstance(e.gate_up_proj, LinearColParallelMerged)

    def test_per_layer_fp8_override_stays_native(self):
        from freetoken.kernel.triton.fp8_pertensor_linear import (
            Fp8PerTensorColMerged,
            Fp8PerTensorLinear,
        )

        e = self._expert(
            layer_id=60, dense_quant="nvfp4", dense_mlp_storage={60: "fp8"}
        )
        self.assertIsInstance(e.gate_up_proj, Fp8PerTensorColMerged)
        self.assertIsInstance(e.down_proj, Fp8PerTensorLinear)
        # Orientation: gate/up read hidden -> write intermediate; down reads intermediate
        # -> writes hidden (weight rows = output dim).
        self.assertEqual(tuple(e.gate_up_proj.weight.shape), (64, 16))
        self.assertEqual(tuple(e.down_proj.weight.shape), (16, 32))

    def test_packed_layer_stays_native(self):
        from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseColMerged

        e = self._expert(
            layer_id=0, dense_quant="nvfp4", dense_mlp_storage={1: "bf16"}
        )
        self.assertIsInstance(e.gate_up_proj, Nvfp4DenseColMerged)

    def test_flag_fallback_without_map(self):
        from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseColMerged
        from freetoken.layers import LinearColParallelMerged

        self.assertIsInstance(
            self._expert(dense_quant="nvfp4").gate_up_proj, Nvfp4DenseColMerged
        )
        self.assertIsInstance(
            self._expert().gate_up_proj, LinearColParallelMerged
        )


if __name__ == "__main__":
    unittest.main()
