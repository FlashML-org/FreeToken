"""Score the first answer in saved non-thinking generation checks."""

import argparse
import json
from pathlib import Path

from .quality_prompts import CASES, first_answer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    a = p.parse_args()
    groups = {}
    for enabled in (0, 1):
        folder = a.root / f"round-1-ark{enabled}"
        rows = []
        for name, _, predicate in CASES:
            paths = list(folder.glob(f"quality-{name}-*.json"))
            assert len(paths) == 1, (name, paths)
            raw = json.loads(paths[0].read_text())
            assert raw["success"] and raw["usage"]["completion_tokens"] == 128
            assert raw["request"]["prompt"].endswith("<think>\n\n</think>\n\n")
            answer = first_answer(raw["text"])
            rows.append(dict(name=name, passed=bool(predicate(answer)), answer=answer))
        groups[str(enabled)] = rows
        (folder / "quality-audited.json").write_text(json.dumps(rows, indent=2) + "\n")
    old = {r["name"] for r in groups["0"] if not r["passed"]}
    new = {r["name"] for r in groups["1"] if not r["passed"]}
    result = dict(
        groups=groups,
        new_failures=sorted(new - old),
        note="8 bounded functional checks, first answer only, non-thinking template, not a standardized accuracy benchmark",
    )
    (a.root / "quality-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print("baseline", len(groups["0"]) - len(old), "ARK", len(groups["1"]) - len(new), "new failures", sorted(new - old))
    assert not new - old, result


if __name__ == "__main__":
    main()
