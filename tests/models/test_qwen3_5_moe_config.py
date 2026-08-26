"""``_compressed_linear_storage`` (weight_map sniffing for mixed compressed-tensors
exports, e.g. unsloth/Qwen3.8-27B-NVFP4) and the per-layer dense-MLP storage gate.

The dense pass keeps each linear in the storage the checkpoint actually uses: native
W4A16 for packed layers, native W8A16 for fp8 attention linears, bf16 dequant for the
rest. When the index is unavailable the fallback must preserve the official
dense-NVFP4 assumption (all packed, no overrides)."""

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
