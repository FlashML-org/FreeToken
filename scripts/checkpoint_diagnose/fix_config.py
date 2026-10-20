"""Level 1 fix: edit only config.json / hf_quant_config.json of a local HF safetensors checkpoint (a remote repo's stub is refused).

    python scripts/checkpoint_diagnose/fix_config.py <ckpt> --out <new dir> --from-report report.jsonl
    python scripts/checkpoint_diagnose/fix_config.py <ckpt> --out <new dir> --set text_config.tie_word_embeddings false
    python scripts/checkpoint_diagnose/fix_config.py <ckpt> --in-place --skip-add 'model.layers.*.mlp.gate' --plan plan.json
    python scripts/checkpoint_diagnose/fix_config.py <ckpt> --restore

Edits, applied in this order:
    --from-report JSONL     the automatic config fixes diagnose.py --out found for this checkpoint
    --set PATH VALUE        a config.json key; PATH is dotted, VALUE is JSON (false, 4, "text")
    --skip-add ENTRY        add a module to the quantization skip lists (ignore / exclude_modules / modules_to_not_convert)
    --skip-remove ENTRY     remove an entry from those lists

Write a skip entry in the checkpoint dialect's syntax: a glob for ModelOpt (model.layers.*.mlp.gate), a module name or ``re:<regex>`` for compressed-tensors, a module name for fp8.
Every run prints the plan first and asks [y/N]; with ``--plan FILE`` the first run only saves the plan and the second identical run applies it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ckpt_edit import Checkpoint, check_out_dir, confirm, plan_lines_for_save, refuse_stub, restore  # noqa: E402


def report_fixes(report: str, ckpt: str, level: str) -> list[dict]:
    """The automatic ``level`` fixes in the latest diagnose.py record of ``ckpt`` (matched by path or repo id)."""
    with open(report) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    mine = [r for r in recs if os.path.realpath(r.get("path", "")) == os.path.realpath(ckpt) or r.get("repo") == ckpt]
    if not mine:
        raise SystemExit(f"{report} has no record of {ckpt}: run diagnose.py {ckpt} --out {report} first")
    if mine[-1].get("status") != "fail":
        raise SystemExit(f"{ckpt} passed in {report}: its notes are not fixes to apply")
    return [f for f in mine[-1].get("fixes", []) if f["level"] == level and f.get("auto")]


def apply_fix(ck: Checkpoint, fix: dict) -> None:
    op = fix["op"]
    if op == "set":
        ck.set_config(fix["path"], fix["value"])
    elif op == "skip_add":
        ck.quant_skip_add(fix["entries"])
    elif op == "skip_remove":
        ck.quant_skip_remove(fix["entries"])
    elif op == "quant_sub":
        ck.quant_sub(fix["pattern"], fix["repl"])
    else:
        raise ValueError(f"unknown config op {op!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--out", help="write the fixed checkpoint into this new directory")
    where.add_argument("--in-place", action="store_true", help="edit the checkpoint itself, keeping a backup")
    where.add_argument("--restore", action="store_true", help="undo the latest --in-place edit")
    ap.add_argument("--from-report", metavar="JSONL")
    ap.add_argument("--set", nargs=2, action="append", default=[], metavar=("PATH", "VALUE"))
    ap.add_argument("--skip-add", action="append", default=[], metavar="ENTRY")
    ap.add_argument("--skip-remove", action="append", default=[], metavar="ENTRY")
    ap.add_argument("--plan", metavar="FILE", help="no prompt: the first run saves the plan here, the second identical run applies it")
    args = ap.parse_args()

    refuse_stub(args.ckpt)
    if args.restore:
        for name in restore(args.ckpt):
            print(f"restored {name}")
        return 0
    fixes = report_fixes(args.from_report, args.ckpt, "config") if args.from_report else []
    if args.from_report and not fixes and not (args.set or args.skip_add or args.skip_remove):
        raise SystemExit(f"the record of {args.ckpt} in {args.from_report} has no automatic config fix")
    fixes += [{"op": "set", "path": p, "value": json.loads(v)} for p, v in args.set]
    fixes += [{"op": "skip_add", "entries": args.skip_add}] if args.skip_add else []
    fixes += [{"op": "skip_remove", "entries": args.skip_remove}] if args.skip_remove else []
    if not fixes:
        ap.error("give --from-report, --set, --skip-add or --skip-remove")
    ck = Checkpoint(args.ckpt)
    check_out_dir(ck, args.out)
    for fix in fixes:
        apply_fix(ck, fix)
    if not ck.log:
        print("nothing to change: the config already reads this way")
        return 0
    lines = [f"plan: edit the config of {ck.path}"] + [f"  {line}" for line in ck.log] + plan_lines_for_save(ck, args.out)
    plan = {"tool": "fix_config", "ckpt": ck.path, "fingerprint": ck.fingerprint, "out": args.out and os.path.abspath(args.out), "log": ck.log}
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
