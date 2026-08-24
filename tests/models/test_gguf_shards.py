"""Tests for GGUF multi-shard discovery and aggregation, plus qwen3moe mapping.

This module verifies that multi-shard GGUF files (following llama.cpp's split
convention) are correctly discovered, validated, and aggregated by
freetoken.models.gguf.reader, and that the qwen3moe tensor-name mapping is correct.

Test fixtures build synthetic GGUF files on disk using gguf.GGUFWriter where
possible, or raw GGUF bytes where needed. The tensor data is tiny (a few F32
values) to keep I/O fast; the focus is on metadata, discovery, and naming, not
quantization or size.

llama.cpp's shard layout (ground truth from measured checksums):
  - Shard 1: full metadata (general.architecture, <arch>.*, tokenizer.*) + split.no=0
  - Shards 2..N: exactly 3 keys (split.no, split.count, split.tensors.count)
  - split.tensors.count is the TOTAL across all shards, not per-shard
  - Filenames are 1-based; split.no is 0-based
"""

from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path

import pytest


class TestSingleFileUnchanged:
    """A plain one-file .gguf still reports correct arch, metadata, tensors.

    This is the regression guard: single-file behavior must be untouched by
    multi-shard support.
    """

    def test_single_file_unchanged(self, tmp_path):
        """Verify a single .gguf file works unchanged."""
        from freetoken.models.gguf.reader import (
            gguf_architecture,
            is_gguf_path,
            load_gguf_metadata,
            gguf_tensor_names,
            iter_gguf_tensors,
        )

        # Write a minimal GGUF file (no shards)
        gguf_path = self._write_minimal_gguf(tmp_path, "single.gguf")

        # Check single-file detection
        assert is_gguf_path(str(gguf_path))

        # Check architecture is readable
        arch = gguf_architecture(str(gguf_path))
        assert arch == "test_model"

        # Check metadata
        metadata = load_gguf_metadata(str(gguf_path))
        assert metadata.get("general.architecture") == "test_model"
        assert metadata.get("test_model.hidden_size") == 256

        # Check tensor names
        names = gguf_tensor_names(str(gguf_path))
        assert "model.embed.weight" in names
        assert len(names) == 1

        # Check tensor iteration
        tensors = list(iter_gguf_tensors(str(gguf_path)))
        assert len(tensors) == 1
        assert tensors[0].name == "model.embed.weight"
        assert tensors[0].shape == (256, 64)

    def _write_minimal_gguf(self, tmp_path: Path, filename: str) -> Path:
        """Write a minimal single-file GGUF with one tensor."""
        try:
            import gguf
        except ImportError:
            pytest.skip("gguf package not available")

        gguf_path = tmp_path / filename
        writer = gguf.GGUFWriter(str(gguf_path))

        # Add metadata
        writer.add_string("general.architecture", "test_model")
        writer.add_uint32("test_model.hidden_size", 256)
        writer.add_uint32("test_model.num_layers", 2)

        # Add one tensor (embedding)
        import numpy as np

        emb_data = np.random.randn(256, 64).astype(np.float32)
        writer.add_tensor("model.embed.weight", emb_data)

        writer.write_header_and_data(str(gguf_path))
        return gguf_path


class TestShardDiscovery:
    """Given a 3-shard set, gguf_shards() returns all three in index order."""

    def test_shard_discovery_orders_and_completes(self, tmp_path):
        """Discover all 3 shards when given any shard or the directory."""
        from freetoken.models.gguf.reader import (
            gguf_shards,
            is_gguf_path,
            gguf_tensor_names,
        )

        # Write 3 shards
        shard_paths = self._write_3_shards(tmp_path)

        # Test 1: discover from shard 1
        result = gguf_shards(str(shard_paths[0]))
        assert len(result) == 3
        assert [Path(p).name for p in result] == [
            "model-00001-of-00003.gguf",
            "model-00002-of-00003.gguf",
            "model-00003-of-00003.gguf",
        ]

        # Test 2: discover from shard 2
        result = gguf_shards(str(shard_paths[1]))
        assert len(result) == 3
        assert result[0] == str(shard_paths[0])  # first shard returned

        # Test 3: discover from shard 3
        result = gguf_shards(str(shard_paths[2]))
        assert len(result) == 3

        # Test 4: directory resolution
        assert is_gguf_path(str(tmp_path))

        # Test 5: tensors are aggregated across shards
        names = gguf_tensor_names(str(shard_paths[0]))
        assert names == {
            "token_embd.weight",  # shard 1
            "blk.0.attn_norm.weight",  # shard 2
            "blk.1.attn_norm.weight",  # shard 3
        }

    def _write_3_shards(self, tmp_path: Path) -> list[Path]:
        """Write a 3-shard GGUF set with metadata and tensors distributed."""
        try:
            import gguf
        except ImportError:
            pytest.skip("gguf package not available")

        import numpy as np

        shard_paths = []

        # Shard 1: full metadata + 1 tensor
        shard1 = tmp_path / "model-00001-of-00003.gguf"
        writer = gguf.GGUFWriter(str(shard1))
        writer.add_string("general.architecture", "qwen3moe")
        writer.add_uint32("qwen3moe.block_count", 2)
        writer.add_uint32("qwen3moe.embedding_length", 128)
        writer.add_uint32("qwen3moe.attention.head_count", 4)
        writer.add_uint32("qwen3moe.attention.head_count_kv", 2)
        writer.add_uint32("qwen3moe.attention.key_length", 32)
        writer.add_float32("qwen3moe.attention.layer_norm_rms_epsilon", 1e-6)
        writer.add_float32("qwen3moe.rope.freq_base", 10000.0)
        writer.add_uint32("qwen3moe.context_length", 4096)
        writer.add_uint32("qwen3moe.expert_count", 8)
        writer.add_uint32("qwen3moe.expert_used_count", 2)
        writer.add_uint32("qwen3moe.expert_feed_forward_length", 512)
        writer.add_uint32("qwen3moe.feed_forward_length", 512)
        # Split metadata for shard 1
        writer.add_uint32("split.no", 0)
        writer.add_uint32("split.count", 3)
        writer.add_uint32("split.tensors.count", 3)
        # One tensor in shard 1
        emb = np.random.randn(1024, 128).astype(np.float32)
        writer.add_tensor("token_embd.weight", emb)
        writer.write_header_and_data(str(shard1))
        shard_paths.append(shard1)

        # Shard 2: minimal metadata + 1 tensor
        shard2 = tmp_path / "model-00002-of-00003.gguf"
        writer = gguf.GGUFWriter(str(shard2))
        writer.add_uint32("split.no", 1)
        writer.add_uint32("split.count", 3)
        writer.add_uint32("split.tensors.count", 3)
        # One tensor in shard 2
        norm = np.random.randn(128).astype(np.float32)
        writer.add_tensor("blk.0.attn_norm.weight", norm)
        writer.write_header_and_data(str(shard2))
        shard_paths.append(shard2)

        # Shard 3: minimal metadata + 1 tensor
        shard3 = tmp_path / "model-00003-of-00003.gguf"
        writer = gguf.GGUFWriter(str(shard3))
        writer.add_uint32("split.no", 2)
        writer.add_uint32("split.count", 3)
        writer.add_uint32("split.tensors.count", 3)
        # One tensor in shard 3
        norm = np.random.randn(128).astype(np.float32)
        writer.add_tensor("blk.1.attn_norm.weight", norm)
        writer.write_header_and_data(str(shard3))
        shard_paths.append(shard3)

        return shard_paths


class TestMissingShard:
    """Missing shard must raise an error naming the missing index."""

    def test_missing_shard_raises(self, tmp_path):
        """Delete the middle shard; opening must raise with missing index named."""
        from freetoken.models.gguf.reader import gguf_shards

        # Write 3 shards
        shard1 = self._write_minimal_shards(tmp_path, count=3)[0]

        # Delete shard 2
        shard2 = tmp_path / "model-00002-of-00003.gguf"
        shard2.unlink()

        # Attempt to discover shards from shard 1 should raise
        with pytest.raises(ValueError) as exc_info:
            gguf_shards(str(shard1))

        error_msg = str(exc_info.value)
        assert "missing" in error_msg.lower()
        assert "2" in error_msg

    def _write_minimal_shards(
        self, tmp_path: Path, count: int
    ) -> list[Path]:
        """Write a minimal set of shards with just metadata."""
        try:
            import gguf
        except ImportError:
            pytest.skip("gguf package not available")

        import numpy as np

        shard_paths = []
        for i in range(count):
            shard_num = i + 1  # 1-based
            shard_file = tmp_path / f"model-{shard_num:05d}-of-{count:05d}.gguf"
            writer = gguf.GGUFWriter(str(shard_file))

            if i == 0:
                # Shard 1: full metadata
                writer.add_string("general.architecture", "qwen3moe")
                writer.add_uint32("qwen3moe.block_count", 1)
                writer.add_uint32("qwen3moe.embedding_length", 128)
                writer.add_uint32("qwen3moe.attention.head_count", 4)
                writer.add_uint32("qwen3moe.attention.head_count_kv", 2)
                writer.add_uint32("qwen3moe.attention.key_length", 32)
                writer.add_float32("qwen3moe.attention.layer_norm_rms_epsilon", 1e-6)
                writer.add_float32("qwen3moe.rope.freq_base", 10000.0)
                writer.add_uint32("qwen3moe.context_length", 4096)
                writer.add_uint32("qwen3moe.expert_count", 8)
                writer.add_uint32("qwen3moe.expert_used_count", 2)
                writer.add_uint32("qwen3moe.expert_feed_forward_length", 512)
                writer.add_uint32("qwen3moe.feed_forward_length", 512)

            # Split metadata (all shards)
            writer.add_uint32("split.no", i)
            writer.add_uint32("split.count", count)
            writer.add_uint32("split.tensors.count", count)

            # Add a dummy tensor
            tensor_data = np.random.randn(32).astype(np.float32)
            writer.add_tensor(f"blk.0.data_{i}.weight", tensor_data)
            writer.write_header_and_data(str(shard_file))
            shard_paths.append(shard_file)

        return shard_paths


class TestMetadataFromShardOne:
    """Shard 1 carries general.architecture and arch keys; others don't."""

    def test_metadata_comes_from_shard_one(self, tmp_path):
        """Reading arch/metadata from any shard returns shard 1's values."""
        from freetoken.models.gguf.reader import (
            gguf_architecture,
            load_gguf_metadata,
        )

        shard_paths = self._write_3_shards_with_metadata(tmp_path)

        # Test from shard 1
        arch1 = gguf_architecture(str(shard_paths[0]))
        meta1 = load_gguf_metadata(str(shard_paths[0]))
        assert arch1 == "qwen3moe"
        assert meta1.get("qwen3moe.block_count") == 2

        # Test from shard 2: should get shard 1's arch
        arch2 = gguf_architecture(str(shard_paths[1]))
        meta2 = load_gguf_metadata(str(shard_paths[1]))
        assert arch2 == "qwen3moe"
        assert meta2.get("qwen3moe.block_count") == 2

        # Test from shard 3: should get shard 1's arch
        arch3 = gguf_architecture(str(shard_paths[2]))
        meta3 = load_gguf_metadata(str(shard_paths[2]))
        assert arch3 == "qwen3moe"
        assert meta3.get("qwen3moe.block_count") == 2

    def _write_3_shards_with_metadata(self, tmp_path: Path) -> list[Path]:
        """Write 3 shards where only shard 1 has arch metadata."""
        try:
            import gguf
        except ImportError:
            pytest.skip("gguf package not available")

        import numpy as np

        shard_paths = []

        # Shard 1: full metadata
        shard1 = tmp_path / "model-00001-of-00003.gguf"
        writer = gguf.GGUFWriter(str(shard1))
        writer.add_string("general.architecture", "qwen3moe")
        writer.add_uint32("qwen3moe.block_count", 2)
        writer.add_uint32("qwen3moe.embedding_length", 128)
        writer.add_uint32("qwen3moe.attention.head_count", 4)
        writer.add_uint32("qwen3moe.attention.head_count_kv", 2)
        writer.add_uint32("qwen3moe.attention.key_length", 32)
        writer.add_float32("qwen3moe.attention.layer_norm_rms_epsilon", 1e-6)
        writer.add_float32("qwen3moe.rope.freq_base", 10000.0)
        writer.add_uint32("qwen3moe.context_length", 4096)
        writer.add_uint32("qwen3moe.expert_count", 8)
        writer.add_uint32("qwen3moe.expert_used_count", 2)
        writer.add_uint32("qwen3moe.expert_feed_forward_length", 512)
        writer.add_uint32("qwen3moe.feed_forward_length", 512)
        writer.add_uint32("split.no", 0)
        writer.add_uint32("split.count", 3)
        writer.add_uint32("split.tensors.count", 2)
        # One tensor
        data = np.random.randn(128).astype(np.float32)
        writer.add_tensor("blk.0.attn_norm.weight", data)
        writer.write_header_and_data(str(shard1))
        shard_paths.append(shard1)

        # Shard 2: only split metadata (no arch keys)
        shard2 = tmp_path / "model-00002-of-00003.gguf"
        writer = gguf.GGUFWriter(str(shard2))
        writer.add_uint32("split.no", 1)
        writer.add_uint32("split.count", 3)
        writer.add_uint32("split.tensors.count", 2)
        # One tensor
        data = np.random.randn(128).astype(np.float32)
        writer.add_tensor("blk.1.attn_norm.weight", data)
        writer.write_header_and_data(str(shard2))
        shard_paths.append(shard2)

        # Shard 3: only split metadata
        shard3 = tmp_path / "model-00003-of-00003.gguf"
        writer = gguf.GGUFWriter(str(shard3))
        writer.add_uint32("split.no", 2)
        writer.add_uint32("split.count", 3)
        writer.add_uint32("split.tensors.count", 2)
        # Minimal tensor to pass validation
        data = np.random.randn(1).astype(np.float32)
        writer.add_tensor("placeholder", data)
        writer.write_header_and_data(str(shard3))
        shard_paths.append(shard3)

        return shard_paths


class TestTensorAggregation:
    """iter_gguf_tensors yields union across all shards in shard order."""

    def test_tensors_aggregate_across_shards(self, tmp_path):
        """Tensors from all shards are yielded in shard order, total matches count."""
        from freetoken.models.gguf.reader import (
            iter_gguf_tensors,
            gguf_tensor_names,
        )

        shard_paths = self._write_3_shards_tensors(tmp_path)

        # Check tensor iteration from shard 1
        tensors = list(iter_gguf_tensors(str(shard_paths[0])))
        assert len(tensors) == 3
        assert tensors[0].name == "blk.0.w1"
        assert tensors[1].name == "blk.0.w2"
        assert tensors[2].name == "blk.1.w1"

        # Check union of names
        names = gguf_tensor_names(str(shard_paths[0]))
        assert names == {"blk.0.w1", "blk.0.w2", "blk.1.w1"}
        assert len(names) == 3

        # Check from shard 2: should still get all 3 tensors
        tensors_from_s2 = list(iter_gguf_tensors(str(shard_paths[1])))
        assert len(tensors_from_s2) == 3

    def _write_3_shards_tensors(self, tmp_path: Path) -> list[Path]:
        """Write 3 shards with tensors distributed across them."""
        try:
            import gguf
        except ImportError:
            pytest.skip("gguf package not available")

        import numpy as np

        shard_paths = []

        # Shard 1: 2 tensors
        shard1 = tmp_path / "model-00001-of-00003.gguf"
        writer = gguf.GGUFWriter(str(shard1))
        writer.add_string("general.architecture", "qwen3moe")
        writer.add_uint32("qwen3moe.block_count", 2)
        writer.add_uint32("qwen3moe.embedding_length", 128)
        writer.add_uint32("qwen3moe.attention.head_count", 4)
        writer.add_uint32("qwen3moe.attention.head_count_kv", 2)
        writer.add_uint32("qwen3moe.attention.key_length", 32)
        writer.add_float32("qwen3moe.attention.layer_norm_rms_epsilon", 1e-6)
        writer.add_float32("qwen3moe.rope.freq_base", 10000.0)
        writer.add_uint32("qwen3moe.context_length", 4096)
        writer.add_uint32("qwen3moe.expert_count", 8)
        writer.add_uint32("qwen3moe.expert_used_count", 2)
        writer.add_uint32("qwen3moe.expert_feed_forward_length", 512)
        writer.add_uint32("qwen3moe.feed_forward_length", 512)
        writer.add_uint32("split.no", 0)
        writer.add_uint32("split.count", 3)
        writer.add_uint32("split.tensors.count", 3)
        writer.add_tensor("blk.0.w1", np.ones((8, 4), dtype=np.float32))
        writer.add_tensor("blk.0.w2", np.ones((4, 8), dtype=np.float32))
        writer.write_header_and_data(str(shard1))
        shard_paths.append(shard1)

        # Shard 2: 1 tensor
        shard2 = tmp_path / "model-00002-of-00003.gguf"
        writer = gguf.GGUFWriter(str(shard2))
        writer.add_uint32("split.no", 1)
        writer.add_uint32("split.count", 3)
        writer.add_uint32("split.tensors.count", 3)
        writer.add_tensor("blk.1.w1", np.ones((8, 4), dtype=np.float32))
        writer.write_header_and_data(str(shard2))
        shard_paths.append(shard2)

        # Shard 3: 0 tensors
        shard3 = tmp_path / "model-00003-of-00003.gguf"
        writer = gguf.GGUFWriter(str(shard3))
        writer.add_uint32("split.no", 2)
        writer.add_uint32("split.count", 3)
        writer.add_uint32("split.tensors.count", 3)
        writer.write_header_and_data(str(shard3))
        shard_paths.append(shard3)

        return shard_paths


class TestDeclaredCountMismatch:
    """Shard 1 says split.count=N but only M<N files exist: must raise."""

    def test_declared_count_mismatch_raises(self, tmp_path):
        """If declared split.count doesn't match found shards, raise with counts."""
        from freetoken.models.gguf.reader import gguf_shards

        # Write 2 shards but declare count=3
        shard1 = self._write_shards_with_bad_count(tmp_path)

        # Attempt to discover should raise
        with pytest.raises(ValueError) as exc_info:
            gguf_shards(str(shard1))

        error_msg = str(exc_info.value)
        assert "Incomplete" in error_msg or "missing" in error_msg.lower()

    def _write_shards_with_bad_count(self, tmp_path: Path) -> Path:
        """Write 2 shards but declare split.count=3."""
        try:
            import gguf
        except ImportError:
            pytest.skip("gguf package not available")

        import numpy as np

        # Shard 1: declares count=3 but we'll only write 2
        shard1 = tmp_path / "model-00001-of-00003.gguf"
        writer = gguf.GGUFWriter(str(shard1))
        writer.add_string("general.architecture", "qwen3moe")
        writer.add_uint32("qwen3moe.block_count", 1)
        writer.add_uint32("qwen3moe.embedding_length", 128)
        writer.add_uint32("qwen3moe.attention.head_count", 4)
        writer.add_uint32("qwen3moe.attention.head_count_kv", 2)
        writer.add_uint32("qwen3moe.attention.key_length", 32)
        writer.add_float32("qwen3moe.attention.layer_norm_rms_epsilon", 1e-6)
        writer.add_float32("qwen3moe.rope.freq_base", 10000.0)
        writer.add_uint32("qwen3moe.context_length", 4096)
        writer.add_uint32("qwen3moe.expert_count", 8)
        writer.add_uint32("qwen3moe.expert_used_count", 2)
        writer.add_uint32("qwen3moe.expert_feed_forward_length", 512)
        writer.add_uint32("qwen3moe.feed_forward_length", 512)
        writer.add_uint32("split.no", 0)
        writer.add_uint32("split.count", 3)  # DECLARED 3
        writer.add_uint32("split.tensors.count", 1)
        writer.add_tensor("data", np.ones((1,), dtype=np.float32))
        writer.write_header_and_data(str(shard1))

        # Shard 2: exists
        shard2 = tmp_path / "model-00002-of-00003.gguf"
        writer = gguf.GGUFWriter(str(shard2))
        writer.add_uint32("split.no", 1)
        writer.add_uint32("split.count", 3)
        writer.add_uint32("split.tensors.count", 1)
        writer.add_tensor("data", np.ones((1,), dtype=np.float32))
        writer.write_header_and_data(str(shard2))

        # Shard 3: does NOT exist (this is the error case)

        return shard1


class TestQwen3MoeMappingFFNNorm:
    """Test qwen3moe's tensor-name mapping: ffn_norm -> post_attention_layernorm.

    This is a pure-function test with no files or fixtures: it just verifies
    that the name mapping from llama.cpp's GGUF naming to FreeToken's module
    names is correct.
    """

    def test_qwen3moe_maps_ffn_norm_to_post_attention(self):
        """Verify blk.N.ffn_norm.weight maps to post_attention_layernorm."""
        from freetoken.models.qwen3_moe.gguf import gguf_name_to_freetoken

        # The critical mapping: ffn_norm -> post_attention_layernorm
        mapped = gguf_name_to_freetoken("blk.0.ffn_norm.weight", num_layers=2)
        assert mapped == "model.layers.0.post_attention_layernorm.weight"

        # Verify on multiple layers
        mapped = gguf_name_to_freetoken("blk.1.ffn_norm.weight", num_layers=2)
        assert mapped == "model.layers.1.post_attention_layernorm.weight"

    def test_qwen3moe_merged_projections_handled(self):
        """Verify attn_q/k/v are reported as merged-projection parts (None).

        qwen3moe's qkv_proj is a merged projection, so the individual
        attn_q/attn_k/attn_v parts return None (handled by iter_gguf_weights).
        """
        from freetoken.models.qwen3_moe.gguf import gguf_name_to_freetoken

        # These are parts of the merged qkv_proj, so they return None
        assert gguf_name_to_freetoken("blk.0.attn_q.weight", num_layers=2) is None
        assert gguf_name_to_freetoken("blk.0.attn_k.weight", num_layers=2) is None
        assert gguf_name_to_freetoken("blk.0.attn_v.weight", num_layers=2) is None

    def test_qwen3moe_expert_suffixes_handled(self):
        """Verify expert stacks (ffn_*_exps) return None (handled by expert-bank loader)."""
        from freetoken.models.qwen3_moe.gguf import gguf_name_to_freetoken

        # Expert stacks are handled by the offload expert-bank loader
        assert (
            gguf_name_to_freetoken("blk.0.ffn_gate_exps.weight", num_layers=2) is None
        )
        assert (
            gguf_name_to_freetoken("blk.0.ffn_up_exps.weight", num_layers=2) is None
        )
        assert (
            gguf_name_to_freetoken("blk.0.ffn_down_exps.weight", num_layers=2) is None
        )
