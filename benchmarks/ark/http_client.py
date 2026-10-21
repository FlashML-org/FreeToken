"""SSE client and resource sampling, using the same timing definitions as prior reports."""

import hashlib, json, time, traceback, urllib.error, urllib.request
from pathlib import Path

ROOT = Path("/artifacts")
ORIGIN = "http://127.0.0.1:19673"
DEVICE = None
MODEL = "/models/Qwen3-30B-A3B-FP8-FTW"


def save(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def get(route, timeout=3):
    with urllib.request.urlopen(ORIGIN + route, timeout=timeout) as response:
        return json.load(response)


def tree(pid):
    result, pending = set(), [pid]
    while pending:
        p = pending.pop()
        if p in result:
            continue
        result.add(p)
        try:
            pending.extend(int(x) for x in Path(f"/proc/{p}/task/{p}/children").read_text().split())
        except OSError:
            pass
    return result


def snapshot(pid):
    data = {"time": time.time(), "rss_sum_bytes": 0, "processes": []}
    for p in tree(pid):
        try:
            status = dict(line.split(":", 1) for line in Path(f"/proc/{p}/status").read_text().splitlines() if ":" in line)
            rss = int(status.get("VmRSS", "0 kB").split()[0]) * 1024
            data["rss_sum_bytes"] += rss
            data["processes"].append(
                {
                    "pid": p,
                    "name": status["Name"].strip(),
                    "rss_bytes": rss,
                    "cpus": status["Cpus_allowed_list"].strip(),
                    "mems": status["Mems_allowed_list"].strip(),
                }
            )
        except (OSError, KeyError):
            pass
    for key in ["mem_info_vram_used", "gpu_busy_percent"]:
        try:
            data[key] = int((DEVICE / key).read_text())
        except (OSError, ValueError):
            data[key] = None
    return data


def monitor(pid, stop, path):
    with path.open("w") as output:
        while not stop.is_set():
            output.write(json.dumps(snapshot(pid)) + "\n")
            output.flush()
            stop.wait(0.25)


def request(directory, context, phase, output_tokens=256, prompt=None):
    prompt = prompt if prompt is not None else (ROOT / f"prompt-{context}.txt").read_text()
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        ORIGIN + "/v1/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    row = {"context": context, "phase": phase, "expected_output_tokens": output_tokens, "request": payload, "events": []}
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            for line in response:
                now = time.perf_counter()
                if now - started > 180:
                    raise TimeoutError("request exceeded 180s")
                if not line.startswith(b"data: "):
                    continue
                value = line[6:].strip()
                if value == b"[DONE]":
                    row["done"] = True
                    break
                event = json.loads(value)
                row["events"].append({"elapsed_s": now - started, "data": event})
                if "error" in event:
                    raise RuntimeError(event["error"])
        row["e2e_s"] = time.perf_counter() - started
        row["usage"] = next(x["data"]["usage"] for x in reversed(row["events"]) if "usage" in x["data"])
        chunks = [x for x in row["events"] if any(c.get("text") for c in x["data"].get("choices", []))]
        row["text"] = "".join(c.get("text", "") for x in chunks for c in x["data"]["choices"])
        row["text_sha256"] = hashlib.sha256(row["text"].encode()).hexdigest()
        row["token_text_chunks"] = len(chunks)
        row["ttft_s"] = chunks[0]["elapsed_s"]
        row["decode_span_s"] = chunks[-1]["elapsed_s"] - chunks[0]["elapsed_s"]
        row["decode_tok_s"] = (output_tokens - 1) / row["decode_span_s"]
        # The official completion streamer emits at most one nonempty event per generated token.
        row["decode_timing_kind"] = (
            "one-event-per-token verified"
            if len(chunks) == row["usage"]["completion_tokens"]
            else "client legacy estimate; empty-token events omitted"
        )
        row["e2e_output_tok_s"] = row["usage"]["completion_tokens"] / row["e2e_s"]
        row["prefill_effective_tok_s"] = context / row["ttft_s"]
        assert row.get("done"), "missing SSE DONE"
        assert row["usage"]["prompt_tokens"] == context, row["usage"]
        assert row["usage"]["completion_tokens"] == output_tokens, row["usage"]
        details = row["usage"].get("prompt_tokens_details") or {}
        assert details.get("cached_tokens", 0) == 0, row["usage"]
        assert row["text"].strip(), "empty output"
        row["success"] = True
    except Exception:
        row.update(success=False, error=traceback.format_exc(), elapsed_s=time.perf_counter() - started)
        raise
    finally:
        save(directory / f"{phase}-{context}.json", row)
    return row
