"""Paired decode+gate kernel timings, or a warmed request to a running FT server.

Kernel mode: PYTHONPATH=python python benchmarks/bench_decode_output_gate.py --json kernel.json
Server mode: ... --url http://127.0.0.1:8000 --model /path/to/tokenizer --json server.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path


def server(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": "List the positive integers in order, starting from 1, separated by commas."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    ids = tokenizer.encode(chat, add_special_tokens=False)
    filler = tokenizer.encode(" hello", add_special_tokens=False)
    assert len(filler) == 1 and len(ids) <= args.input_tokens
    prompt = tokenizer.decode(filler * (args.input_tokens - len(ids)) + ids, skip_special_tokens=False)
    assert len(tokenizer.encode(prompt, add_special_tokens=False)) == args.input_tokens
    body = dict(model=args.served_model_name, prompt=prompt, max_tokens=args.output_tokens, temperature=0,
                stream=True, ignore_eos=True, stream_options={"include_usage": True})
    rows = []
    for iteration in range(args.repeats + 1):
        request = urllib.request.Request(args.url.rstrip("/") + "/v1/completions",
                                        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        events, text, usage, finished = [], "", None, None
        with urllib.request.urlopen(request, timeout=600) as response:
            for raw in response:
                if not raw.startswith(b"data: "):
                    continue
                payload = raw[6:].strip()
                if payload == b"[DONE]":
                    break
                value = json.loads(payload)
                elapsed = time.perf_counter() - started
                content = "".join(c.get("text", "") for c in value.get("choices", []))
                if content:
                    text += content
                    events.append(elapsed)
                if any(c.get("finish_reason") for c in value.get("choices", [])):
                    finished = elapsed
                if value.get("usage"):
                    usage = value["usage"]
        elapsed = time.perf_counter() - started
        assert usage and usage["completion_tokens"] == args.output_tokens and events and finished
        rows.append(dict(warmup=iteration == 0, usage=usage, ttft_s=events[0], elapsed_s=elapsed,
                         decode_tok_s=(args.output_tokens - 1) / (finished - events[0]),
                         e2e_tok_s=args.output_tokens / elapsed, nonempty_events=len(events),
                         events_s=events, finished_s=finished,
                         output_sha256=hashlib.sha256(text.encode()).hexdigest()))
    return dict(mode="server", request=body, rows=rows)


def kernel(args):
    import torch
    from freetoken.kernel.triton.attention import (
        _decode_gate_sigmoid_table, decode_kv_split_cap, decode_paged_attention,
    )

    reference = decode_paged_attention
    if args.baseline_path:
        spec = importlib.util.spec_from_file_location("gate_bench_upstream", args.baseline_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        reference = module.decode_paged_attention
    torch.manual_seed(20261021)
    torch.set_num_threads(4)
    _decode_gate_sigmoid_table(torch.cuda.current_device())
    rows = []
    for batch in (1, 4):
        lengths = [512 - i * 17 for i in range(batch)]
        total = sum(lengths)
        q = torch.randn(batch, 16, 256, device="cuda", dtype=torch.bfloat16) * 0.2
        k = torch.randn(total + 7, 2, 256, device="cuda", dtype=torch.bfloat16) * 0.2
        v = torch.randn_like(k) * 0.2
        gate = torch.randn_like(q)
        indptr = torch.tensor([0, *lengths], device="cuda", dtype=torch.int32).cumsum(0).to(torch.int32)
        indices = torch.randperm(total + 7, device="cuda").to(torch.int32)[:total]
        positions = torch.tensor([n - 1 for n in lengths], device="cuda", dtype=torch.int32)
        splits = decode_kv_split_cap(batch, 16, 2, torch.cuda.current_device())
        partial = torch.empty(batch, 16, splits, 256, device="cuda")
        lse = torch.empty(batch, 16, splits, device="cuda")
        params = (q, k, v, indptr, indices, positions, partial, lse, 256 ** -0.5)
        functions = [lambda: reference(*params) * gate.sigmoid(),
                     lambda: decode_paged_attention(*params, output_gate=gate)]
        expected, actual = functions[0](), functions[1]()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        graphs = []
        for function in functions:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(5):
                    function()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(4):
                    function()
            graphs.append(graph)
        for graph in graphs:
            for _ in range(1000):
                graph.replay()
        torch.cuda.synchronize()
        samples = [[], []]
        for group in range(7):
            for index in ((0, 1) if group % 2 == 0 else (1, 0)):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iterations):
                    graphs[index].replay()
                end.record()
                end.synchronize()
                samples[index].append(start.elapsed_time(end) * 1000 / (args.iterations * 4))
        baseline_us, candidate_us = [statistics.median(s) for s in samples]
        rows.append(dict(batch=batch, lengths=lengths, head_dim=256, query_heads=16, kv_heads=2,
                         baseline_us=baseline_us, candidate_us=candidate_us,
                         speedup=baseline_us / candidate_us, samples_us=samples, bitwise_equal=True))
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return dict(mode="kernel", torch=str(torch.__version__), hip=torch.version.hip, gpu=props.name,
                baseline_path=args.baseline_path, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="running server; omit for kernel measurements")
    parser.add_argument("--model", help="local tokenizer path, required with --url")
    parser.add_argument("--served-model-name", default="gate-bench")
    parser.add_argument("--input-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=1, help="measured requests after one warmup")
    parser.add_argument("--iterations", type=int, default=50, help="Graph replays per kernel sample")
    parser.add_argument("--baseline-path", help="upstream kernel/triton/attention.py for kernel A/B")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.url and not args.model:
        parser.error("--model is required with --url")
    result = server(args) if args.url else kernel(args)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
