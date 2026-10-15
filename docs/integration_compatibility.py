"""CPU-only compatibility checks in the existing image, not native-stack validation."""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata as metadata
from pathlib import Path
import sys

import torch


root = Path("/workspace")
assert not torch.cuda.is_available(), "compatibility checks must not allocate a GPU"
print("COMPATIBILITY ONLY: existing Torch stack; no native ABI or GPU validation", flush=True)
print("Python", sys.version, flush=True)
for package in ("torch", "triton", "flashlib", "transformers", "apache-tvm-ffi", "pytest"):
    try:
        print(package, metadata.version(package), flush=True)
    except metadata.PackageNotFoundError:
        print(package, "NOT INSTALLED", flush=True)

for path in sorted((root / "python").rglob("*.py")) + sorted((root / "tests").rglob("*.py")):
    ast.parse(path.read_bytes(), filename=str(path))
print("All production/test Python parses", flush=True)
for name in (
    "freetoken", "freetoken.engine.engine", "freetoken.models.mimo_v2.model",
    "freetoken.server.function_call_parser", "freetoken.server.reasoning_parser",
    "freetoken.checkpoint.convert", "freetoken.moe.cpu_executor",
):
    module = importlib.import_module(name)
    path = Path(module.__file__)
    assert path.is_relative_to(root / "python"), (name, path)
    print(name, path, hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)

if sys.argv[1:] == ["--smoke"]:
    import json
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import patch

    from safetensors.torch import save_file
    from freetoken.checkpoint.convert import convert_checkpoint
    from freetoken.checkpoint.ftw import iter_ftw_weights
    from freetoken.distributed import set_tp_info
    from freetoken.engine.engine import Engine, _cpu_moe_executor_viable
    from freetoken.kernel import backend
    from freetoken.layers import iter_moe_layers, set_rope_device
    from freetoken.layers.rotary import get_rope
    from freetoken.models.mimo_v2.config import parse_config
    from freetoken.models.mimo_v2.model import MiMoV2ForCausalLM
    from freetoken.moe.cpu_executor import CpuMoeExecutor, compiled_extension_current
    from freetoken.server.function_call_parser import FunctionCallParser
    from freetoken.server.reasoning_parser import ReasoningParser
    from freetoken.utils.hf import RawConfigShim
    from freetoken.utils.torch_utils import torch_dtype

    assert Path("/tmp").is_dir()
    artifacts = Path(tempfile.mkdtemp(prefix="integration-smoke-", dir="/tmp"))
    print("Retained synthetic artifacts", artifacts, flush=True)
    set_tp_info(rank=0, size=1)
    set_rope_device(torch.device("cpu"))
    config_data = dict(
        architectures=["MiMoV2ForCausalLM"], model_type="mimo_v2", dtype="bfloat16",
        hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=2,
        num_key_value_heads=1, swa_num_key_value_heads=2, head_dim=192, swa_head_dim=192,
        v_head_dim=128, swa_v_head_dim=128, vocab_size=32, tie_word_embeddings=True,
        hidden_act="silu", max_position_embeddings=256, attention_value_scale=0.707,
        sliding_window=128, hybrid_layer_pattern=[0, 1, 0], moe_layer_freq=[0, 1, 1],
        rope_theta=1e7, swa_rope_theta=1e4, partial_rotary_factor=0.334,
        add_swa_attention_sink_bias=True, n_routed_experts=2, num_experts_per_tok=1,
        moe_intermediate_size=32, norm_topk_prob=True,
    )
    cfg = parse_config(RawConfigShim(config_data))
    with torch_dtype(torch.float32):
        reference_model = MiMoV2ForCausalLM(cfg)
    raw = {}
    for name, tensor in reference_model.state_dict().items():
        source_name = name.replace(".sinks", ".attention_sink_bias")
        if name.endswith(".e_score_correction_bias"):
            source_name = name.removesuffix(".e_score_correction_bias") + ".gate.e_score_correction_bias"
        raw[source_name] = torch.full(tensor.shape, 1.001953125, dtype=torch.float32)
    for layer in (1, 2):
        for expert in range(2):
            for proj, shape in (("gate", (32, 32)), ("up", (32, 32)), ("down", (32, 32))):
                tensor = torch.full(shape, 1.001953125)
                if expert == 0 and proj == "gate":
                    tensor[0, 0] = 65504.0
                else:
                    tensor = tensor.to(torch.bfloat16)
                raw[f"model.layers.{layer}.mlp.experts.{expert}.{proj}_proj.weight"] = tensor
    source = artifacts / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(config_data))
    save_file(raw, str(source / "model.safetensors"))

    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for path in (source, artifacts / str(dtype).removeprefix("torch.")):
            if path != source:
                with patch.object(torch.cuda, "set_device", lambda device: None):
                    convert_checkpoint(str(source), str(path), dtype=dtype, moe_backend="fused", device="cpu")
            mc = parse_config(RawConfigShim(config_data))
            with torch_dtype(dtype):
                model = MiMoV2ForCausalLM(mc)
            settings = SimpleNamespace(model_path=str(path), model_config=mc, use_dummy_weight=False,
                                       active_encoders=(), expert_load="serial")
            engine = SimpleNamespace(model=model, device=torch.device("cpu"), dtype=dtype)
            state = Engine._load_weight_state_dict(engine, settings)
            model.load_state_dict(dict(state))
            Engine._load_resident_experts(engine, settings, list(iter_moe_layers(model)))
            assert not any(".experts." in name for name in state)
            assert model.lm_head.tied_embedding is model.model.embed_tokens
            for name, weight in state.items():
                assert torch.equal(weight, raw[name.replace(".sinks", ".attention_sink_bias").replace(
                    ".mlp.e_score_correction_bias", ".mlp.gate.e_score_correction_bias")].to(weight.dtype)), name
            assert state["model.layers.1.self_attn.sinks"].dtype == torch.float32
            assert state["model.layers.1.mlp.e_score_correction_bias"].dtype == torch.float32
            for layer in (1, 2):
                block = model.model.layers.op_list[layer].mlp
                assert block.experts.layer_id == layer - 1
                view = block.experts.quant_method.resident_view(block.experts)
                assert view.tensors["gate_up"].shape == (2, 64, 32)
                for expert in range(2):
                    prefix = f"model.layers.{layer}.mlp.experts.{expert}"
                    assert torch.equal(view.tensors["gate_up"][expert, :32], raw[f"{prefix}.gate_proj.weight"].to(dtype))
                    assert torch.equal(view.tensors["gate_up"][expert, 32:], raw[f"{prefix}.up_proj.weight"].to(dtype))
                    assert torch.equal(view.tensors["down"][expert], raw[f"{prefix}.down_proj.weight"].to(dtype))
                weights, ids = block._route(torch.zeros(1, 32, dtype=dtype))
                assert weights.dtype == torch.float32 and ids.dtype == torch.int32
                assert weights.item() == 0.5, "top-one sigmoid weights must not renormalize"
            print("PASS resident source/FTW dense, banks, sinks, bias, tied head, routing", dtype, path, flush=True)

    with torch_dtype(torch.float16):
        model = MiMoV2ForCausalLM(parse_config(RawConfigShim(config_data)))
    settings.model_path = str(artifacts / "float32")
    engine = SimpleNamespace(model=model, device=torch.device("cpu"), dtype=torch.float16)
    Engine._load_resident_experts(engine, settings, list(iter_moe_layers(model)))
    gate = model.model.layers.op_list[1].mlp.experts._expert_banks["gate_up"][0, :32]
    assert torch.isfinite(gate).all()
    assert torch.equal(gate, raw["model.layers.1.mlp.experts.0.gate_proj.weight"].to(torch.float16))
    print("PASS FP32 FTW bank -> FP16 direct replay", flush=True)

    for override in (dict(n_routed_experts=2, n_group=2), dict(moe_layer_freq=[1, 0, 1])):
        try:
            parse_config(RawConfigShim({**config_data, **override}))
        except ValueError:
            pass
        else:
            raise AssertionError("unsupported group width or sparse topology was accepted")
    print("PASS grouped routing and sparse topology admission", flush=True)

    from freetoken.layers.quantization import QuantBackend, set_quant_backend

    set_quant_backend(QuantBackend.parse("moe.nvfp4=triton"))
    quantized_layers = {f"model.layers.{layer}.mlp.experts.{e}.{proj}_proj": {
        "quant_algo": "W4A16_NVFP4", "group_size": 16,
    } for layer in (1, 2) for e in range(2) for proj in ("gate", "up", "down")}
    nv_config = {**config_data, "quantization_config": {
        "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "group_size": 16,
        "quantized_layers": quantized_layers,
    }}
    nv_raw = {name: tensor for name, tensor in raw.items() if ".experts." not in name}
    for name in quantized_layers:
        nv_raw[name + ".weight"] = torch.full((32, 16), 0x21, dtype=torch.uint8)
        nv_raw[name + ".weight_scale"] = torch.full((32, 2), 0.5, dtype=torch.float8_e4m3fn)
        nv_raw[name + ".weight_scale_2"] = torch.tensor(1.25)
    nv_path = artifacts / "nvfp4"
    nv_path.mkdir()
    (nv_path / "config.json").write_text(json.dumps(nv_config))
    save_file(nv_raw, str(nv_path / "model.safetensors"))
    from freetoken.distributed import get_tp_info
    from freetoken.engine.config import EngineConfig

    mc = EngineConfig(model_path=str(nv_path), tp_info=get_tp_info(), dtype=torch.bfloat16,
                      moe_strategy="fused", quant_backend="moe.nvfp4=triton").model_config
    with torch_dtype(torch.bfloat16):
        model = MiMoV2ForCausalLM(mc)
    settings = SimpleNamespace(model_path=str(nv_path), model_config=mc, use_dummy_weight=False,
                               active_encoders=(), expert_load="serial")
    engine = SimpleNamespace(model=model, device=torch.device("cpu"), dtype=torch.bfloat16)
    model.load_state_dict(Engine._load_weight_state_dict(engine, settings))
    Engine._load_resident_experts(engine, settings, list(iter_moe_layers(model)))
    for layer in (1, 2):
        experts = model.model.layers.op_list[layer].mlp.experts
        assert experts.layer_id == layer - 1
        bank = experts.quant_method.resident_view(experts).tensors
        assert bank["gate_up"].shape == (2, 64, 16) and bank["down"].shape == (2, 32, 16)
        assert torch.all(bank["gate_up"] == 0x21) and torch.all(bank["down"] == 0x21)
        assert torch.all(bank["gate_up_scale"].float() == 0.5) and torch.all(bank["down_scale"].float() == 0.5)
        assert torch.all(bank["gate_up_global"] == 1.25) and torch.all(bank["down_global"] == 1.25)
    print("PASS resident NVFP4 ModelOpt bank packing/attachment (no dequant kernel execution)", flush=True)

    with patch.object(backend, "is_flashinfer_installed", lambda: True):
        get_rope.cache_clear()
        full, swa = get_rope(192, 64, 32, 1e7), get_rope(192, 64, 32, 1e4)
        for rope, base in ((full, 1e7), (swa, 1e4)):
            assert rope.apply_rope_with_cos_sin_cache_inplace.__module__.startswith("freetoken.kernel.triton")
            frequency = torch.outer(torch.arange(32, dtype=torch.float32), 1.0 / base ** (torch.arange(0, 64, 2) / 64))
            torch.testing.assert_close(rope._cos_sin_cache, torch.cat((frequency.cos(), frequency.sin()), -1), rtol=0, atol=1e-6)
        assert full._cos_sin_cache.data_ptr() != swa._cos_sin_cache.data_ptr()
    print("PASS 192-wide partial/dual RoPE dispatch and frequencies (no kernel execution)", flush=True)

    tools = [{"type": "function", "function": {"name": "echo", "parameters": {
        "type": "object", "properties": {"text": {"type": "string"}}}}}]
    value = '\nquoted </function></tool_call><function=phantom> " slash\\\n'
    wire = "<tool_call><function=echo><parameter=text>" + value + "</parameter></function></tool_call>"
    whole = FunctionCallParser(tools, "mimo_v2").parse_non_stream(wire)
    assert len(whole.calls) == 1 and json.loads(whole.calls[0].parameters) == {"text": value}
    for size in (1, 7, len(wire)):
        parser = FunctionCallParser(tools, "mimo_v2")
        calls = []
        for offset in range(0, len(wire), size):
            calls.extend(parser.parse_stream_chunk(wire[offset:offset + size])[1])
        calls.extend(parser.finalize_stream())
        assert [call.name for call in calls if call.name] == ["echo"]
        assert json.loads("".join(call.parameters for call in calls)) == {"text": value}
    reason_wire = "<tool_call><function=echo><parameter=text><think>literal</think></parameter></function></tool_call>"
    assert ReasoningParser("mimo_v2", force_reasoning=False).parse_non_stream(reason_wire) == ("", reason_wire)
    print("PASS literal parameter closers, fragmented tools, opaque reasoning markers", flush=True)

    from jinja2 import Environment
    from freetoken.server.generation import _make_reasoning_parser

    fixture = (root / "tests/server/mimo_chat_contract.jinja").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(fixture).hexdigest() == "5d1656529ef121e333c5a5c65aa1717f8ac7a73ac48c801968fe9c6f354ca9c4"
    environment = Environment()
    environment.filters["tojson"] = lambda value, ensure_ascii=True: json.dumps(value, ensure_ascii=ensure_ascii)
    template = environment.from_string(fixture.decode())
    for thinking in (None, True, False):
        for with_tools in (False, True):
            kwargs = {} if thinking is None else {"enable_thinking": thinking}
            prompt = template.render(messages=[{"role": "user", "content": "ping"}],
                                     tools=tools if with_tools else None, add_generation_prompt=True, **kwargs)
            assert prompt.rsplit("<|im_start|>assistant\n", 1)[1] == ("<think></think>" if thinking is False else "")
            parser = _make_reasoning_parser(SimpleNamespace(chat_template_kwargs=kwargs),
                                            SimpleNamespace(config=SimpleNamespace(reasoning_parser="mimo_v2")))
            assert parser.detector.force_reasoning is False
            assert parser.parse_non_stream("answer") == ("", "answer")
    print("PASS content-pinned authored template/factory generation matrix (six cases)", flush=True)

    with patch.object(backend, "is_rocm", lambda: True), patch.object(importlib, "import_module", side_effect=AssertionError("CUDA import on ROCm")):
        for check in (backend.is_flashinfer_installed, backend.is_sgl_kernel_installed, backend.is_vllm_installed):
            check.cache_clear()
            assert check() is False
            check.cache_clear()
    assert not compiled_extension_current(), "old image unexpectedly has the current native ABI"
    assert not _cpu_moe_executor_viable(cfg)
    try:
        CpuMoeExecutor(SimpleNamespace(), top_k=1, activation="silu", apply_router_weight_on_input=False,
                       num_threads=1, max_tokens=1, device=torch.device("cpu"))
    except RuntimeError as exc:
        assert "predates the shared-IO executor ABI" in str(exc)
    else:
        raise AssertionError("stale native ABI was accepted")
    print("PASS ROCm gates and stale native ABI rejection (no native execution)", flush=True)

    from freetoken import kernel
    from freetoken.engine import engine as engine_module
    from freetoken.moe.bench_profile import load_backend_recommendation

    profile = artifacts / "benchbw-v5.json"
    profile.write_text(json.dumps({"version": 5, "gpu": {"name": "FAKE GPU"}, "dtypes": {"bf16": "hybrid"}}))
    with patch.dict("os.environ", {"FREETOKEN_BENCHBW_PATH": str(profile)}), \
         patch.object(engine_module, "_profile_gpu", lambda: ("FAKE GPU", None)), \
         patch.object(engine_module, "_is_unified_memory_gpu", lambda index=None: False):
        assert load_backend_recommendation("bf16", gpu_name="FAKE GPU") == "hybrid"
        for state in ("unavailable", "stale", "wrong-format", "wrong-activation", "current"):
            native = SimpleNamespace(
                weight_formats=lambda: ["nvfp4"] if state == "wrong-format" else ["bf16"],
                memops_probe=lambda *_: True, host_func_capture_safe=lambda: True,
                kernel_io_probe=lambda *_: "", max_generic_act_id=lambda: 2 if state == "wrong-activation" else 4,
            )
            if state == "unavailable":
                native = None
            elif state == "stale":
                native = SimpleNamespace(max_generic_act_id=lambda: 4)
            settings = EngineConfig(model_path=str(source), tp_info=get_tp_info(), dtype=torch.bfloat16,
                                    moe_strategy="auto", attention_backend="triton")
            if state == "wrong-activation":
                object.__setattr__(settings.model_config, "hidden_act", "swiglu_clamp")
            with patch.object(kernel, "_cpu_moe", native, create=True):
                engine_module._adjust_config(settings)
            assert settings.moe_strategy == ("hybrid" if state == "current" else "offload")
    print("PASS actual _adjust_config v5 recommending-profile path (five mocked ABI/format/activation cases)", flush=True)

    for strategy in ("cpu", "hybrid", "offload"):
        settings = EngineConfig(model_path=str(source), tp_info=get_tp_info(), dtype=torch.bfloat16,
                                moe_strategy=strategy, attention_backend="triton",
                                moe_cpu_layers="1" if strategy == "offload" else None)
        with patch.object(kernel, "_cpu_moe", None, create=True), \
             patch.object(engine_module, "shared_offload_method", lambda _model: None), \
             patch.object(engine_module, "load_expert_banks", side_effect=AssertionError("bank allocation reached")), \
             patch.object(engine_module, "OffloadMoeCache", side_effect=AssertionError("GPU cache allocation reached")), \
             patch.object(engine_module, "_check_pin_budget", side_effect=AssertionError("pin budget reached")):
            for operation in (
                lambda: engine_module._adjust_config(settings),
                lambda: Engine._init_offload_moe_cache(SimpleNamespace(model=None, _host_tables_bytes=0), settings),
            ):
                try:
                    operation()
                except RuntimeError as exc:
                    assert "current _cpu_moe shared-IO ABI" in str(exc)
                else:
                    raise AssertionError("CPU path was accepted before allocation guard")
    print("PASS explicit CPU/Hybrid/CPU-layer admission before pinning, bank or GPU-cache allocation", flush=True)

    # Run the pure assertion-only oracle test without importing pytest or the GPU cases.
    test_path = root / "tests/moe/test_hybrid_fetch.py"
    oracle = ast.Module(body=[node for node in ast.parse(test_path.read_bytes()).body
                             if isinstance(node, ast.FunctionDef) and node.name in (
                                 "_balanced_fetch", "test_balanced_fetch_tracks_fraction")], type_ignores=[])
    namespace = {"Q": 1 << 16}
    exec(compile(oracle, str(test_path), "exec"), namespace)
    namespace["test_balanced_fetch_tracks_fraction"]()
    print("PASS exhaustive independent fetch-balance oracle self-test (outside pytest)", flush=True)
    raise SystemExit(0)

if importlib.util.find_spec("pytest") is None:
    raise SystemExit("BLOCKED: pytest is absent; no installation authorized")
import pytest

raise SystemExit(pytest.main([
    *sys.argv[1:], "-q", "-o", "addopts=", "-p", "no:cacheprovider",
    "--basetemp=/tmp/integration-pytest", "-m", "not slow and not needs_weights", "-rs",
]))
