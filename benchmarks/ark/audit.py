"""Independently recompute formal SSE timing and paired summaries from raw files."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def audit(root):
    rows = []
    for path in sorted(root.glob("round-*/measured-*.json")):
        row = json.loads(path.read_text())
        assert row["success"] and row["done"]
        assert row["usage"]["prompt_tokens"] == row["context"]
        assert row["usage"]["completion_tokens"] == 256
        assert row["request"]["prompt"] == (root / f"prompt-{row['context']}.txt").read_text()
        assert row["request"]["temperature"] == 0 and row["request"]["ignore_eos"]
        assert (row["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0) == 0
        events = [x for x in row["events"] if any(c.get("text") for c in x["data"].get("choices", []))]
        span = events[-1]["elapsed_s"] - events[0]["elapsed_s"]
        assert abs(255 / span - row["decode_tok_s"]) < 1e-9
        assert events[0]["elapsed_s"] == row["ttft_s"]
        text = "".join(c.get("text", "") for x in events for c in x["data"]["choices"])
        assert hashlib.sha256(text.encode()).hexdigest() == row["text_sha256"]
        command = json.loads((path.parent / "command.json").read_text())
        assert command["env"]["FREETOKEN_USE_ARK"] == str(row["enabled"])
        assert command["env"]["FREETOKEN_ARK_POLICY"] == "validated"
        assert "FREETOKEN_ARK_TRACE" not in command["env"]
        assert "benchmarks.ark.probe" not in command["argv"]
        assert json.loads((path.parent / "complete.json").read_text())["success"]
        assert json.loads((path.parent / "shutdown.json").read_text())["vram"] < 256 * 2**20
        rows.append({k: v for k, v in row.items() if k not in ("events", "text", "request")})
    paired = []
    for context in (512, 32768):
        pairs = []
        for rid in sorted({r["round"] for r in rows}):
            pair = {r["enabled"]: r for r in rows if r["round"] == rid and r["context"] == context}
            assert set(pair) == {0, 1}
            baseline, candidate = pair[0], pair[1]
            pairs.append(
                dict(
                    round=rid,
                    speedup=candidate["decode_tok_s"] / baseline["decode_tok_s"],
                    ft_decode_tok_s=baseline["decode_tok_s"],
                    ark_decode_tok_s=candidate["decode_tok_s"],
                    outputs_equal=baseline["text_sha256"] == candidate["text_sha256"],
                )
            )
        if pairs:
            ft = statistics.median(r["ft_decode_tok_s"] for r in pairs)
            ark = statistics.median(r["ark_decode_tok_s"] for r in pairs)
            paired.append(
                dict(
                    context=context,
                    rounds=pairs,
                    ft_median=ft,
                    ark_median=ark,
                    median_change_percent=100 * (ark / ft - 1),
                    wins=sum(r["speedup"] > 1 for r in pairs),
                    n=len(pairs),
                )
            )
    return dict(passed=True, samples=rows, paired=paired)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    a = p.parse_args()
    result = audit(a.root)
    (a.root / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["paired"], indent=2))


if __name__ == "__main__":
    main()
