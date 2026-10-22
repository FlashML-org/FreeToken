"""Level 2 fix: rename tensors of a local HF safetensors checkpoint (a remote repo's stub is refused). Only the shard headers, the index and the quantization skip lists change; tensor data is kept byte for byte.

    python scripts/checkpoint_diagnose/rename_tensors.py <ckpt> --out <new dir> --rule PATTERN REPL [--rule ...]
    python scripts/checkpoint_diagnose/rename_tensors.py <ckpt> --out <new dir> --from-report report.jsonl
    python scripts/checkpoint_diagnose/rename_tensors.py <ckpt> --in-place ... --plan plan.json
    python scripts/checkpoint_diagnose/rename_tensors.py <ckpt> --restore

A rule is ``re.sub(PATTERN, REPL, name)``; rules apply in order. They are also applied to the quantization skip-list entries and compressed-tensors targets that are module names or globs; ``re:`` entries are not rewritten.
Every run prints the plan first: per rule how many tensors it renames with examples, the skip entries that still name the old modules, and what each shard write costs. Then it asks [y/N].
With ``--plan FILE`` it never asks: the first run saves the plan to FILE and changes nothing, the second identical run applies it.
Refused: a rule that renames nothing, a rename that gives two tensors one name, an ``--out`` that is not empty.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ckpt_edit import Checkpoint, check_out_dir, confirm, plan_lines_for_save, refuse_stub, restore  # noqa: E402
from fix_config import report_fixes  # noqa: E402

_DIGITS = re.compile(r"\.\d+(?=\.|$)")


def _pattern(name: str) -> str:
    return _DIGITS.sub(".N", name)


def rule_effects(names: list[str], rules: list[tuple[str, str]]) -> list[dict]:
    """Per rule, applied in order: the names it changes as pattern pairs, and how many of its matches are not at the start of the name."""
    effects, current = [], list(names)
    for pat, repl in rules:
        rx = re.compile(pat)
        pairs, mid = Counter(), 0
        for i, name in enumerate(current):
            new = rx.sub(repl, name)
            if new != name:
                pairs[(_pattern(name), _pattern(new))] += 1
                m = rx.search(name)
                mid += m is not None and m.start() > 0
                current[i] = new
        anchored = pat.startswith("^") or pat.endswith("$")
        effects.append({"rule": [pat, repl], "renamed": sum(pairs.values()), "pairs": pairs, "mid": 0 if anchored else mid})
    return effects


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--out", help="write the renamed checkpoint into this new directory")
    where.add_argument("--in-place", action="store_true", help="edit the checkpoint itself, keeping a backup")
    where.add_argument("--restore", action="store_true", help="undo the latest --in-place edit")
    ap.add_argument("--rule", nargs=2, action="append", default=[], metavar=("PATTERN", "REPL"))
    ap.add_argument("--from-report", metavar="JSONL", help="the automatic rename fixes diagnose.py --out found for this checkpoint")
    ap.add_argument("--plan", metavar="FILE", help="no prompt: the first run saves the plan here, the second identical run applies it")
    args = ap.parse_args()

    refuse_stub(args.ckpt)
    if args.restore:
        for name in restore(args.ckpt):
            print(f"restored {name}")
        return 0
    rules = []
    if args.from_report:
        for fix in report_fixes(args.from_report, args.ckpt, "rename"):
            rules += [tuple(r) for r in fix["rules"]]
        if not rules and not args.rule:
            raise SystemExit(f"the record of {args.ckpt} in {args.from_report} has no automatic rename fix")
    rules += [tuple(r) for r in args.rule]
    if not rules:
        ap.error("give --rule PATTERN REPL or --from-report")
    ck = Checkpoint(args.ckpt)
    check_out_dir(ck, args.out)

    names = list(ck.tensors())
    effects = rule_effects(names, rules)
    lines = [f"plan: rename tensors of {ck.path} ({len(names)} tensors)"]
    for i, e in enumerate(effects, 1):
        pat, repl = e["rule"]
        lines.append(f"  rule {i}: s/{pat}/{repl}/ renames {e['renamed']} tensors")
        for (old, new), n in e["pairs"].most_common(6):
            lines.append(f"      {old}  ->  {new}" + (f"   (x{n})" if n > 1 else ""))
        if len(e["pairs"]) > 6:
            lines.append(f"      ... {len(e['pairs']) - 6} more name patterns")
        if e["mid"]:
            lines.append(f"      warning: {e['mid']} of these matches are inside the name, not at its start; anchor the pattern with ^ or $ if that is not meant")
    dead = [i for i, e in enumerate(effects, 1) if not e["renamed"]]
    if dead:
        print("\n".join(lines))
        raise SystemExit(f"refused: rule {', '.join(map(str, dead))} renames no tensor (check the pattern against the names above)")

    try:
        changes = ck.rename_regex(rules)
    except ValueError as e:
        raise SystemExit(f"refused: {e}")
    for pat, repl in rules:
        ck.quant_rename(pat, repl)
    lines += [f"  {line}" for line in ck.log]
    lines.append(f"  {len(changes)} tensors renamed, {len(names) - len(changes)} keep their names")
    old_modules = {n.rpartition(".")[0] for n in changes}
    new_modules = {n.rpartition(".")[0] for n in ck.tensors()}
    for entry in ck.stale_quant_entries(old_modules, new_modules):
        lines.append(f"  warning: {entry} still names the old modules and none of the new ones; fix it with fix_config.py after this rename")
    lines += plan_lines_for_save(ck, args.out)
    plan = {"tool": "rename_tensors", "ckpt": ck.path, "fingerprint": ck.fingerprint, "out": args.out and os.path.abspath(args.out),
            "rules": [list(r) for r in rules], "renamed": len(changes), "log": ck.log}
    if not confirm(plan, lines, args.plan):
        return 0
    for name in ck.save(out=None if args.in_place else args.out):
        print(f"wrote {name}")
    if args.plan:
        os.remove(args.plan)
    print("done: run diagnose.py on the result to check it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
