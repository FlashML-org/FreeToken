"""Protect V4.1 startup budgeting and serve an image in a fresh CUDA process."""

import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("active_encoder", [False, True])
def test_finalized_weights_and_engram_staging_precede_cache_budgets(monkeypatch, tmp_path, active_encoder):
    import freetoken.engine.engine as engine_module

    allocations = {}
    baseline_free = 4096
    resident_and_staging_bytes = 48 + 32 * 4 + 16 * 2 + 8 + (12 if active_encoder else 0)
    host_table_bytes = 1024
    cache_bytes = 64

    class QuantMethod:
        calls = 0

        def finalize(self, layer):
            self.calls += 1
            assert layer.weight.shape == (24,)
            layer.weight = layer.weight.repeat_interleave(2)
            allocations["weight"] = layer.weight
            allocations["quant_workspace"] = torch.empty(32, dtype=torch.float32, device="cpu")

    class Model:
        quant_method = QuantMethod()

        def load_state_dict(self, weights):
            self.weight = weights["weight"]
            allocations["weight"] = self.weight
            if active_encoder:
                allocations["encoder_weights"] = torch.empty(256, dtype=torch.uint8, device="cpu")

        def place_encoder_weights(self, mode):
            assert mode == "host" and self.quant_method.calls == 1
            del allocations["encoder_weights"]
            allocations["encoder_staging"] = torch.empty(12, dtype=torch.uint8, device="cpu")

        def encode(self, item):
            raise AssertionError("encoder warmup is stubbed in the memory-budget test")

        def load_host_tables(self, config):
            assert not active_encoder or "encoder_staging" in allocations
            allocations["engram_values"] = torch.empty(16, dtype=torch.bfloat16, device="cpu")
            allocations["engram_mask"] = torch.empty(8, dtype=torch.bool, device="cpu")
            return host_table_bytes

    class ReachedKVPlanning(Exception):
        pass

    class Pool:
        @staticmethod
        def solve_num_pages(config, available_memory):
            assert model.quant_method.calls == 1
            assert available_memory == 3072 - resident_and_staging_bytes - cache_bytes
            raise ReachedKVPlanning

    model = Model()
    config = SimpleNamespace(
        model_path=str(tmp_path), tp_info=SimpleNamespace(rank=0, size=1),
        dtype=torch.bfloat16, quant_backend="moe.nvfp4=triton", moe_strategy="offload",
        model_config=object(), page_size=8, memory_ratio=0.75,
        active_encoders=(SimpleNamespace(kind="vision"),) if active_encoder else (),
        mm=SimpleNamespace(encoder_weights="host", embed_cache_device="cpu"),
        served_modalities={"image"} if active_encoder else set(), hf_config=SimpleNamespace(),
    )

    def free_memory(self):
        free = baseline_free - sum(t.numel() * t.element_size() for t in allocations.values())
        return free, free

    def initialize_offload_cache(self, config):
        assert self.model.quant_method.calls == 1
        assert self._host_tables_bytes == host_table_bytes
        assert self._weights_bytes == resident_and_staging_bytes
        assert self._post_weights_free == baseline_free - resident_and_staging_bytes
        allocations["offload_cache"] = torch.empty(cache_bytes, dtype=torch.uint8, device="cpu")

    # Keep the real Engine startup and finalize traversal; replace only CUDA and its consumers.
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "Stream", lambda: object())
    monkeypatch.setattr(torch.cuda, "set_stream", lambda stream: None)
    monkeypatch.setattr(torch, "manual_seed", lambda seed: None)
    monkeypatch.setattr("freetoken.gpu_select.bind_assigned_gpu", lambda rank: torch.device("cpu"))
    monkeypatch.setattr(engine_module, "set_tp_info", lambda **kwargs: None)
    monkeypatch.setattr(engine_module, "set_quant_backend", lambda backend: None)
    monkeypatch.setattr(engine_module, "_adjust_ftw_quant_backend", lambda path, backend: backend)
    monkeypatch.setattr(engine_module, "_ensure_expandable_segments", lambda: None)
    monkeypatch.setattr(engine_module, "_adjust_config", lambda config: None)
    monkeypatch.setattr(engine_module, "set_global_ctx", lambda ctx: None)
    monkeypatch.setattr(engine_module, "set_rope_device", lambda device: None)
    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine_module, "resolve_pool_class", lambda config: Pool)
    monkeypatch.setattr(engine_module, "create_model", lambda config: model)
    monkeypatch.setattr(engine_module, "state_pool_bytes", lambda config: 0)
    monkeypatch.setattr(engine_module.Engine, "_init_communication", lambda self, config: None)
    monkeypatch.setattr(engine_module.Engine, "_sync_get_memory", free_memory)
    monkeypatch.setattr(engine_module.Engine, "_load_weight_state_dict", lambda self, config: {
        "weight": torch.ones(24, dtype=torch.uint8, device="cpu"),
    })
    monkeypatch.setattr(engine_module.Engine, "_init_offload_moe_cache", initialize_offload_cache)
    monkeypatch.setattr("freetoken.mm.processor.get_mm_processor", lambda *args: object())
    monkeypatch.setattr(engine_module.Engine, "_warmup_encoders", lambda self: None)
    with pytest.raises(ReachedKVPlanning):
        engine_module.Engine(config)


def _run_engine_smoke(folder, port, kv_quant, image_path):
    import json
    from dataclasses import asdict
    from unittest.mock import patch

    import freetoken.engine.engine as engine_module
    from freetoken.core import Batch, Req, SamplingParams
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from freetoken.engine.engine import Engine
    from freetoken.layers.quantization import QuantKind, finalize_quant
    from freetoken.moe.offload_cache import iter_offload_moe_layers
    from freetoken.models.deepseek_v41.args import DeepseekV41Args
    from freetoken.models.deepseek_v41.image_processor import IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START
    from freetoken.scheduler.mm import plan_mm_batch

    args = DeepseekV41Args(
        n_layers=3, n_mtp_layers=0, compress_ratios=(0, 2, 2),
        kv_source_layers=(1,), index_source_layers=(1,), candidate_source_layer=-1,
        dim=64, n_heads=2, head_dim=32, rope_head_dim=16, q_lora_rank=32,
        o_lora_rank=32, o_groups=2, window_size=8, index_n_heads=2, index_head_dim=32,
        index_topk=3, moe_inter_dim=64, n_routed_experts=4, n_activated_experts=2,
        vocab_size=128, hc_mult=2, engram_layer_ids=(1,), engram_num_embeddings=(17,),
        engram_max_ngram_size=2, engram_vocab_size=17, engram_n_heads=1,
        engram_head_dim=32, engram_compressed_vocab_size=128,
        vision_n_layers=1, vision_dim=32, vision_n_heads=2, vision_inter_dim=32,
        vision_patch_size=2, vision_downsample_ratio=2, image_token_id=127,
    )
    raw = asdict(args) | {
        "architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41",
        "quantization_config": {"moe_quant_algo": "NVFP4"},
        "vision_config": {"num_hidden_layers": 1, "hidden_size": 32, "num_attention_heads": 2,
                          "intermediate_size": 32, "patch_size": 2, "downsample_ratio": 2},
    }
    Path(folder, "config.json").write_text(json.dumps(raw))

    class LocalEngineConfig(EngineConfig):
        @property
        def distributed_addr(self):
            return f"tcp://127.0.0.1:{port}"

    config = LocalEngineConfig(
        model_path=folder, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
        max_running_req=1, moe_strategy="offload", quant_backend="moe.nvfp4=triton",
        moe_cpu_layers="", moe_cache_size=8, kv_quant=kv_quant,
        moe_prefill_overlap=True, use_dummy_weight=True, use_pynccl=False,
        max_seq_len_override=64, num_page_override=40, cuda_graph_max_bs=0,
    )
    finalized_counts = []

    def checked_finalize(model):
        count = finalize_quant(model)
        finalized_counts.append(count)
        return count

    with patch.object(engine_module, "finalize_quant", checked_finalize):
        engine = Engine(config)
    assert finalized_counts == [args.n_layers]
    assert engine.model._engram_runtime is not None
    assert config.attention_backend == "dsv41_sparse"
    assert config.model_config.expert_quant == "nvfp4" and config.page_size == 8
    assert config.model_config.is_multimodal
    assert engine.kv_cache.kv_quant == kv_quant
    experts = list(iter_offload_moe_layers(engine.model))
    assert len(experts) == args.n_layers
    for expert in experts:
        method = expert.quant_method
        assert method.kind is QuantKind.NVFP4
        assert method.kernel.name == "triton"
        assert method.cfg.activation == "swiglu_clamp"
        assert method.cfg.alpha == 1.0 and method.cfg.limit == args.swiglu_limit
        assert method.cfg.strategy == "offload" and method.cfg.decode_target == "gpu"
    assert engine.model._transformer.vision.patch_embed.proj.weight.dtype == torch.bfloat16
    runtime = engine.model._engram_runtime
    resident_bytes = sum(p.numel() * p.element_size() for p in engine.model.state_dict().values() if p.is_cuda)
    staged_bytes = sum(module._values.numel() * module._values.element_size() for module in runtime.modules)
    staged_bytes += runtime.device_mask.numel() * runtime.device_mask.element_size()
    vision_streamer = engine.model._transformer.vision._streamer
    assert vision_streamer is not None and vision_streamer.bank.is_pinned()
    staged_bytes += vision_streamer.device_bytes
    assert engine._weights_bytes >= resident_bytes + staged_bytes
    sample = engine.sampler.sample

    def checked_sample(logits, sampling_args):
        assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
        return sample(logits, sampling_args)

    engine.sampler.sample = checked_sample
    engine.page_table[0, :16] = torch.arange(16, device="cuda")
    for start in (0, 8):
        engine.kv_cache.bind_window_pages(start, start)
    media = [{"start": 1, "types": torch.tensor([IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END]),
              "patches": torch.randn(4, 3, 2, 2), "n_vit_h": 2, "n_vit_w": 2}]
    ids = torch.tensor([5, 127, 127, 127, 127, 6, 7, 8], dtype=torch.int32)
    if image_path == "mm_items":
        result = engine.mm_processor.from_media(ids, media)
        req = Req(result.input_ids, 0, 0, 3, 0, SamplingParams(temperature=0), None, mm_items=result.mm_items)
        for item in req.mm_items:
            engine.encoder_cache.register(item.hash, req.uid, item.num_tokens)
    else:
        req = Req(ids, 0, 0, 3, 0, SamplingParams(temperature=0), None, media=media)
    for phase in ("prefill", "decode"):
        batch = Batch([req], phase)
        batch.padded_reqs = batch.reqs
        batch.input_ids = req.input_ids[req.cached_len:].cuda()
        batch.positions = torch.arange(req.cached_len, req.device_len, device="cuda")
        batch.active_table_idx = torch.tensor([0], dtype=torch.long, device="cuda")
        batch.out_loc = engine.page_table[0, req.cached_len:req.device_len]
        if phase == "prefill" and req.mm_items:
            jobs, plan, rows = plan_mm_batch([req], engine.encoder_cache)
            batch.mm_encoder_jobs, batch.mm_gather_plan = jobs, plan
            batch.mm_rows = torch.tensor(rows, device="cuda", dtype=torch.long)
        engine.attn_backend.prepare_metadata(batch)
        with torch.inference_mode():
            output = engine.forward_batch(batch, engine.sampler.prepare(batch))
        output.copy_done_event.synchronize()
        assert output.next_tokens_cpu.shape == (1,)
        assert 0 <= output.next_tokens_cpu.item() < 128
        req.append_host(output.next_tokens_cpu)
    if image_path == "mm_items":
        assert req.mm_items[0].feature is None
        assert not engine.encoder_cache.has(req.mm_items[0].hash)
    else:
        assert "embeddings" in media[0]
    assert req.cached_len == 9 and req.device_len == 10
    torch.cuda.synchronize()
    torch.distributed.destroy_process_group()
    print("V41_ENGINE_PREFILL_DECODE_IMAGE_OK")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kv_quant", ["none", "fp8-fp4"], ids=["bf16", "fp8-fp4"])
@pytest.mark.parametrize("image_path", ["legacy", "mm_items"])
def test_engine_initialization_and_image_generation(tmp_path, kv_quant, image_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(tmp_path), str(port), kv_quant, image_path],
                            capture_output=True, text=True, timeout=180, env=os.environ.copy())
    assert result.returncode == 0, result.stdout + result.stderr
    assert "V41_ENGINE_PREFILL_DECODE_IMAGE_OK" in result.stdout


if __name__ == "__main__":
    _run_engine_smoke(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])
