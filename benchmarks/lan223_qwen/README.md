# LAN-223 Qwen API replication harness

`run_api_benchmark.py` measures a running local FreeToken server through its
OpenAI-compatible streaming API. It does not start a service, modify model
files, change llama-swap, or contact another LAN host. The script refuses to
run unless the operating system host name is LAN-223 or an explicitly supplied
test host.

Run the script on LAN-223 from the isolated FreeToken environment after the
server is already warm:

```bash
python benchmarks/lan223_qwen/run_api_benchmark.py \
  --model qwen3.6-35b-a3b-nvfp4 \
  --tokenizer /home/david/freetoken-amd/models/Qwen3.6-35B-A3B-NVFP4 \
  --base-url http://127.0.0.1:1919/v1 \
  --samples 5 \
  --artifact-dir /home/david/freetoken-amd/artifacts/qwen-replication-$(date -u +%Y%m%dT%H%M%SZ)
```

The harness writes one immutable JSON artifact per request plus a manifest and
summary. Decode TPS is based on tokenizer-counted generated text rather than
the count of network chunks. A server that fails to provide content, returns a
malformed SSE sequence, or emits an error is marked failed rather than silently
excluded.

This harness measures a fixed-length greedy request. It is not the authors'
paper replication until the exact published prompt, sampling, cache state, and
statistic are supplied in the protocol artifact.
