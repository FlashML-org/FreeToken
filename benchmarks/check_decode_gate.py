"""Evaluate final FreeToken base-decode promotion evidence without heuristics."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from pathlib import Path
from typing import Iterable


COMPARATOR_CONTEXT = 9216
COMPARATOR_BATCH = 512
COMPARATOR_UBATCH = 512
COMPARATOR_KV_TYPE = "q8_0"

# Disjoint measurement lanes (rocm-ollama-gap Inc 0). A row belongs to exactly
# one lane; mixing lanes in one promotion claim is always a rejection.
LANE_SAMPLED = "sampled_absolute"
LANE_GREEDY = "greedy_correctness"
LANE_REPLAY = "teacher_forced_replay"
LANES = (LANE_SAMPLED, LANE_GREEDY, LANE_REPLAY)


def _row_lane(row: dict) -> str | None:
    """The lane a row belongs to, or None when the row is unclassifiable."""
    lane = row.get("lane")
    if lane in LANES:
        return str(lane)
    metadata = _mapping(row.get("metadata"))
    meta_lane = metadata.get("lane")
    if isinstance(meta_lane, str) and meta_lane in LANES:
        return meta_lane
    # Legacy freetoken-base rows predate explicit lanes: classify by sampling.
    if str(row.get("schema", "")).startswith("freetoken-base-"):
        sampling = _sampling(row)
        if sampling is not None and sampling.get("temperature") == 0:
            return LANE_GREEDY
        return LANE_SAMPLED
    # Older gate fixtures had no schema/lane field but did carry the sampled
    # throughput shape. Preserve their meaning without accepting arbitrary
    # unlabeled rows.
    sampling = _sampling(row)
    if isinstance(row.get("decode_tok_s"), (int, float)) and sampling is not None:
        return LANE_GREEDY if sampling.get("temperature") == 0 else LANE_SAMPLED
    return None


def _lane_reasons(rows: list[dict]) -> list[str]:
    """Reject mixed or unclassifiable lanes before any promotion arithmetic."""
    reasons: list[str] = []
    lanes: set[str] = set()
    for index, row in enumerate(rows):
        lane = _row_lane(row)
        if lane is None:
            reasons.append(f"accepted row {index} has unknown/mismatched lane identity")
    classified = {lane for lane in (_row_lane(row) for row in rows) if lane}
    if len(classified) > 1:
        reasons.append(
            f"rows mix disjoint lanes: {sorted(classified)}; lanes cannot be combined"
        )
    return reasons


def load_jsonl(path: str) -> list[dict]:
    rows = []
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(value)
    return rows


def _accepted(rows: Iterable[dict], *, lane: str | None = None) -> list[dict]:
    return [
        row for row in rows
        if row.get("status") == "accepted"
        and (lane is None or row.get("lane") == lane)
    ]


def _mapping(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _sha256(value: object) -> str | None:
    if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value):
        return value.lower()
    return None


def _model_sha(row: dict) -> str | None:
    metadata = _mapping(row.get("metadata"))
    fingerprint = _mapping(row.get("model_fingerprint"))
    for value in (row.get("model_sha256"), fingerprint.get("sha256"), metadata.get("model_sha256")):
        digest = _sha256(value)
        if digest:
            return digest
    return None


def _prompt_sha(row: dict) -> str | None:
    metadata = _mapping(row.get("metadata"))
    return _sha256(row.get("prompt_sha256")) or _sha256(metadata.get("prompt_sha256"))


def _comparator_value(row: dict, key: str):
    metadata = _mapping(row.get("metadata"))
    return row.get(key, metadata.get(key))


def _comparator_reasons(rows: list[dict], *, label: str) -> tuple[list[str], str | None]:
    """Require one exact benchmark contract before promotion arithmetic."""
    reasons: list[str] = []
    fixture_shas: set[str] = set()
    expected = {
        "context": COMPARATOR_CONTEXT,
        "batch": COMPARATOR_BATCH,
        "ubatch": COMPARATOR_UBATCH,
        "kv_type": COMPARATOR_KV_TYPE,
    }
    for index, row in enumerate(rows):
        for key, wanted in expected.items():
            value = _comparator_value(row, key)
            if value != wanted:
                reasons.append(
                    f"{label} row {index} {key}={value!r} != comparator {wanted!r}"
                )
        fixture = _sha256(_comparator_value(row, "fixture_sha256"))
        if fixture is None:
            reasons.append(f"{label} row {index} missing full fixture SHA-256")
        else:
            fixture_shas.add(fixture)
    if len(fixture_shas) != 1:
        reasons.append(f"{label} rows do not carry one full fixture SHA-256")
    return reasons, next(iter(fixture_shas), None)


def _sampling(row: dict) -> dict | None:
    metadata = _mapping(row.get("metadata"))
    for value in (row.get("sampling"), row.get("options"), metadata.get("sampling")):
        if isinstance(value, dict):
            return value
    return None


def _execution(row: dict) -> dict | None:
    metadata = _mapping(row.get("metadata"))
    value = row.get("execution", metadata.get("execution"))
    return value if isinstance(value, dict) else None


def _execution_reasons(rows: list[dict]) -> list[str]:
    """Require one observed eligible execution class, never requested CLI flags."""
    reasons: list[str] = []
    required = (
        "effective_moe_backend", "expert_storage", "resident_gguf", "expert_fetches",
        "expert_remaps", "attention_backend", "graph_state", "decode_batch_size",
        "mtp", "speculative",
    )
    fingerprints = set()
    for index, row in enumerate(rows):
        execution = _execution(row)
        if execution is None:
            reasons.append(f"accepted row {index} missing execution-mode evidence")
            continue
        missing = [key for key in required if key not in execution]
        if missing:
            reasons.append(f"accepted row {index} execution evidence missing: {','.join(missing)}")
            continue
        execution_class = execution.get("execution_class")
        # Older accepted fixtures predate the explicit class. Their complete
        # resident evidence maps unambiguously to resident_fused.
        if execution_class is None and execution["effective_moe_backend"] == "fused":
            execution_class = "resident_fused"
        if execution_class == "resident_fused":
            if execution["effective_moe_backend"] != "fused":
                reasons.append(f"accepted row {index} resident_fused backend is not fused")
            if execution["expert_storage"] != "resident_gguf" or execution["resident_gguf"] is not True:
                reasons.append(f"accepted row {index} does not prove resident native GGUF experts")
            if execution["expert_fetches"] != 0 or execution["expert_remaps"] != 0:
                reasons.append(f"accepted row {index} reports expert fetch/remap activity")
        elif execution_class == "offload_warm":
            if execution["effective_moe_backend"] not in {"offload", "hybrid"}:
                reasons.append(f"accepted row {index} offload_warm backend is invalid")
            for key in ("expert_fetch_bytes", "expert_copy_bytes"):
                if execution.get(key) != 0:
                    reasons.append(f"accepted row {index} reports nonzero warm {key}")
        else:
            reasons.append(f"accepted row {index} execution_class={execution_class!r} is not eligible")
        if execution["graph_state"] != "replay":
            reasons.append(f"accepted row {index} graph_state={execution['graph_state']!r} != 'replay'")
        if execution["decode_batch_size"] != 1:
            reasons.append(f"accepted row {index} decode_batch_size is not 1")
        if execution["mtp"] != "off" or execution["speculative"] is not False:
            reasons.append(f"accepted row {index} enables MTP/speculative execution")
        fingerprints.add(
            (execution_class, *(execution.get(key) for key in required))
        )
    if len(fingerprints) > 1:
        reasons.append("accepted rows mix execution modes")
    return reasons


def _same_sampling(left: dict, right: dict) -> bool:
    keys = ("temperature", "top_p", "top_k")
    return isinstance(left, dict) and isinstance(right, dict) and all(
        left.get(key) == right.get(key) for key in keys
    )


def _bootstrap_p02_5(values: list[float], *, samples: int = 10_000, seed: int = 0) -> float:
    if not values:
        return float("nan")
    rng = random.Random(seed)
    medians = [statistics.median(rng.choices(values, k=len(values))) for _ in range(samples)]
    medians.sort()
    return medians[min(len(medians) - 1, int(samples * 0.025))]


def _reference(
    rows: list[dict],
    model_sha: str,
    prompt_sha: str | None,
    sampling: dict,
    fixture_sha: str | None,
) -> tuple[dict | None, list[str]]:
    reasons = []
    # Ollama only becomes eligible when both local files were hashed and matched. A
    # manifest digest alone is deliberately insufficient.
    ollama = [
        row for row in _accepted(rows)
        if row.get("schema", "").startswith("ollama-")
        and row.get("mtp") == "off"
        and row.get("speculative") in (False, "off")
        and _mapping(row.get("acceptance")).get("accepted") is True
        and _mapping(_mapping(row.get("acceptance")).get("checks")).get("mtp_off") is True
        and _mapping(row.get("reference_identity")).get("status") == "verified"
        and _mapping(row.get("reference_identity")).get("same_blob") is True
        and row.get("prompt_sha256") == prompt_sha
        and _sha256(_comparator_value(row, "fixture_sha256")) == fixture_sha
        and not _comparator_reasons([row], label="reference")[0]
        and isinstance(row.get("client_arrival_tok_s"), (int, float))
    ]
    if ollama:
        identity = ollama[0]["reference_identity"]
        reference_gguf = _mapping(identity).get("reference_gguf")
        if _sha256(_mapping(reference_gguf).get("sha256")) != model_sha:
            reasons.append("verified Ollama identity does not match FreeToken model SHA-256")
        elif not all(_sampling(row) and _same_sampling(_sampling(row), sampling) for row in ollama):
            reasons.append("Ollama sampling options do not match FreeToken")
        else:
            values = [float(row["client_arrival_tok_s"]) for row in ollama]
            return {
                "kind": "ollama-client-arrival",
                "sha256": model_sha,
                "rows": len(values),
                "median_tok_s": statistics.median(values),
                "identity": identity,
                "by_repeat": {
                    row.get("repeat"): float(row["client_arrival_tok_s"])
                    for row in ollama if row.get("repeat") is not None
                },
            }, reasons

    cli = [
        row for row in _accepted(rows, lane="llama-cli")
        if row.get("model_sha256") == model_sha
        and row.get("prompt_sha256") == prompt_sha
        and _sha256(_comparator_value(row, "fixture_sha256")) == fixture_sha
        and not _comparator_reasons([row], label="reference")[0]
        and row.get("mtp") == "off"
        and row.get("speculative") is False
        and _mapping(row.get("acceptance")).get("accepted") is True
        and _mapping(_mapping(row.get("acceptance")).get("checks")).get("mtp_off") is True
        and row.get("eval_count") == row.get("decode_requested")
        and isinstance(row.get("native_decode_tok_s"), (int, float))
        and _sampling(row) is not None
        and _same_sampling(_sampling(row), sampling)
    ]
    if cli:
        values = [float(row["native_decode_tok_s"]) for row in cli]
        return {
            "kind": "llama-cli-native",
            "sha256": model_sha,
            "rows": len(values),
            "median_tok_s": statistics.median(values),
            "identity": "full-file-sha256",
            "by_repeat": {
                row.get("repeat"): float(row["native_decode_tok_s"])
                for row in cli if row.get("repeat") is not None
            },
        }, reasons
    reasons.append("no matched full-SHA llama-cli or verified same-blob Ollama reference")
    return None, reasons


def _probe_reasons(probe: object, model_sha: str | None, prompt_sha: str | None) -> list[str]:
    """Validate finite-logit and eager/graph evidence from the offline probe."""
    if not isinstance(probe, dict):
        return ["finite-logit/parity probe unavailable"]
    reasons: list[str] = []
    probe_model = _sha256(_mapping(probe.get("model")).get("sha256"))
    if not model_sha or probe_model != model_sha:
        reasons.append("finite-logit probe model SHA-256 does not match FreeToken rows")
    probe_prompt = _sha256(probe.get("prompt_sha256"))
    if not prompt_sha or probe_prompt != prompt_sha:
        reasons.append("finite-logit probe prompt SHA-256 does not match FreeToken rows")
    if probe.get("mtp") != "off" or probe.get("speculative") is not False:
        reasons.append("finite-logit probe did not prove MTP/speculative mode is off")

    lanes = []
    for name in ("eager", "graph"):
        lane = probe.get(name)
        if lane is None:
            continue
        if not isinstance(lane, dict):
            reasons.append(f"{name} finite-logit probe record is malformed")
            continue
        lanes.append(name)
        if lane.get("finite_logits") is not True:
            reasons.append(f"{name} finite-logit probe failed")
        if not isinstance(lane.get("decode_rows"), int) or lane["decode_rows"] < 1:
            reasons.append(f"{name} finite-logit probe has no decode rows")
    if not lanes:
        reasons.append("finite-logit probe has no eager or graph lane")
    comparison = _mapping(probe.get("comparison"))
    if probe.get("graph") is not None and comparison.get("token_ids_equal") is not True:
        reasons.append("eager/graph greedy token parity failed or is unavailable")
    return reasons


def _paired_deltas(rows: list[dict], reference: dict | None) -> list[float]:
    if not reference:
        return []
    by_repeat = reference.get("by_repeat") or {}
    deltas = []
    for row in rows:
        key = row.get("repeat")
        value = row.get("decode_tok_s")
        ref_value = by_repeat.get(key)
        if isinstance(value, (int, float)) and isinstance(ref_value, (int, float)):
            deltas.append(float(value) - float(ref_value))
    return deltas


def _bootstrap_interval(
    values: list[float], *, samples: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    medians = [statistics.median(rng.choices(values, k=len(values))) for _ in range(samples)]
    medians.sort()
    low = medians[min(len(medians) - 1, int(samples * 0.025))]
    high = medians[min(len(medians) - 1, int(samples * 0.975))]
    return low, high


def _evaluate_sampled_gate(
    freetoken_rows: list[dict],
    reference_rows: list[dict] | None = None,
    probe: dict | None = None,
    *,
    min_runs: int = 10,
    directional_threshold: float = 80.225,
) -> dict:
    reference_rows = reference_rows or []
    rejected = [row for row in freetoken_rows if row.get("status") != "accepted"]
    accepted = _accepted(freetoken_rows)
    reasons: list[str] = []
    if rejected:
        reasons.append(f"{len(rejected)} FreeToken rows rejected")
    if len(accepted) < min_runs:
        reasons.append(f"need >= {min_runs} accepted FreeToken rows, got {len(accepted)}")

    model_shas = {_model_sha(row) for row in accepted}
    model_shas.discard(None)
    model_sha = next(iter(model_shas), None) if len(model_shas) == 1 else None
    if len(model_shas) != 1:
        reasons.append("FreeToken rows do not carry one full model SHA-256")
    prompt_shas = {_prompt_sha(row) for row in accepted}
    if len(prompt_shas) != 1 or None in prompt_shas:
        reasons.append("FreeToken rows do not carry one prompt SHA-256")
    sampling_rows = [_sampling(row) for row in accepted]
    sampling = sampling_rows[0] if sampling_rows else None
    if sampling is None or not all(value is not None and _same_sampling(value, sampling) for value in sampling_rows):
        reasons.append("FreeToken sampling options are missing or mixed")
    comparator_reasons, fixture_sha = _comparator_reasons(accepted, label="FreeToken")
    reasons.extend(comparator_reasons)
    exact = all(
        row.get("completion_tokens") == row.get("decode_requested")
        and _mapping(row.get("acceptance")).get("accepted") is True
        for row in accepted
    )
    if not exact:
        reasons.append("accepted rows contain exact-completion or acceptance mismatch")
    reasons.extend(_execution_reasons(accepted))
    reasons.extend(_lane_reasons(accepted))
    for index, row in enumerate(accepted):
        kernel_observed = _mapping(_mapping(row.get("metadata")).get("kernel_observed"))
        fallback_markers = kernel_observed.get("fallback_markers")
        if isinstance(fallback_markers, int) and fallback_markers > 0:
            reasons.append(
                f"accepted row {index} carries candidate-eligible fallback evidence "
                f"(fallback_markers={fallback_markers})"
            )
    if any(_mapping(row.get("metadata")).get("mtp") != "off" for row in accepted):
        reasons.append("MTP/speculative mode is not off")
    values = [float(row["decode_tok_s"]) for row in accepted if isinstance(row.get("decode_tok_s"), (int, float))]
    if len(values) != len(accepted):
        reasons.append("accepted rows missing decode throughput")
    median = statistics.median(values) if values else None
    p02_5 = _bootstrap_p02_5(values) if values else None
    minimum = min(values) if values else None
    if values and minimum < 70:
        reasons.append(f"run below 70 tok/s: {minimum:.3f}")
    if values and not p02_5 > 75:
        reasons.append(f"bootstrap p02.5={p02_5:.3f} is not >75 tok/s")

    prompt_sha = next(iter(prompt_shas), None) if len(prompt_shas) == 1 else None
    probe_reasons = _probe_reasons(probe, model_sha, prompt_sha)
    reasons.extend(probe_reasons)

    reference = None
    if model_sha and prompt_sha and sampling is not None:
        reference, reference_reasons = _reference(
            reference_rows, model_sha, prompt_sha, sampling, fixture_sha
        )
        reasons.extend(reference_reasons)
    else:
        reasons.append("reference matching blocked by FreeToken identity/options failure")
    if reference is None:
        reasons.append("matched reference unavailable; directional Ollama threshold cannot promote parity")
        threshold = directional_threshold
        threshold_source = "directional-ollama-unproven"
    else:
        threshold = float(reference["median_tok_s"])
        threshold_source = reference["kind"]
        if not values or not median > threshold:
            reasons.append(
                f"FreeToken median {median if median is not None else 'n/a'} "
                f"does not beat reference {threshold:.3f}"
            )

    paired_deltas = _paired_deltas(accepted, reference)
    paired_low, paired_high = _bootstrap_interval(paired_deltas, seed=20260831)
    if reference is not None:
        if len(paired_deltas) < min_runs:
            reasons.append(
                f"need >= {min_runs} complete FreeToken/reference pairs, got {len(paired_deltas)}"
            )
        elif not paired_low > 0:
            reasons.append(
                f"paired median delta bootstrap interval [{paired_low:.3f}, {paired_high:.3f}] crosses zero"
            )

    # Greedy rows must not silently vary output, when this gate is run on greedy evidence.
    output_hashes = {row.get("output_sha1") or row.get("output_sha256") for row in accepted}
    if sampling and sampling.get("temperature") == 0 and output_hashes and None not in output_hashes and len(output_hashes) > 1:
        reasons.append("greedy/output hashes are not stable")
    gate = not reasons
    return {
        "reference_identity": reference,
        "probe": {
            "provided": isinstance(probe, dict),
            "valid": not probe_reasons,
            "reasons": probe_reasons,
        },
        "threshold_source": threshold_source,
        "threshold_tok_s": threshold,
        "runs": {
            "accepted": len(accepted),
            "required": min_runs,
            "median_tok_s": median,
            "p02_5_bootstrap_tok_s": p02_5,
            "min_tok_s": minimum,
            "max_tok_s": max(values) if values else None,
            "paired": len(paired_deltas),
            "paired_median_delta_tok_s": statistics.median(paired_deltas) if paired_deltas else None,
            "paired_delta_p02_5_bootstrap_tok_s": paired_low if paired_deltas else None,
            "paired_delta_p97_5_bootstrap_tok_s": paired_high if paired_deltas else None,
        },
        "rejected": len(rejected),
        "gate": gate,
        "reasons": reasons,
    }


def _replay_rate(row: dict) -> float | None:
    """Normalize replay timing to tok/s; missing timing is never zero."""
    for key in ("decode_tok_s", "replay_tok_s", "native_decode_tok_s"):
        value = row.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    steps = _mapping(row.get("steps"))
    milliseconds = steps.get("ms_per_token_median")
    if not isinstance(milliseconds, (int, float)):
        milliseconds = row.get("decode_ms_per_token_median")
    if isinstance(milliseconds, (int, float)) and milliseconds > 0:
        return 1000.0 / float(milliseconds)
    return None


def _replay_identity_reasons(rows: list[dict], label: str) -> list[str]:
    reasons: list[str] = []
    for index, row in enumerate(rows):
        if _row_lane(row) != LANE_REPLAY:
            reasons.append(f"{label} row {index} is not teacher_forced_replay")
        if row.get("forced") is not True:
            reasons.append(f"{label} row {index} is not forced replay")
        if row.get("ids_match") is not True and row.get("prompt_ids_match") is not True:
            reasons.append(f"{label} row {index} input/token IDs are not proven identical")
        if row.get("route_hash_status") != "matched":
            reasons.append(f"{label} row {index} route hashes are not matched")
        for key in ("model_sha256", "fixture_sha256", "tokenizer_sha256", "manifest_ids_sha256"):
            if _sha256(row.get(key)) is None:
                reasons.append(f"{label} row {index} missing full {key}")
        if row.get("mtp") != "off" or row.get("speculative") is not False:
            reasons.append(f"{label} row {index} enables MTP/speculative execution")
        if row.get("decode_batch_size") != 1:
            reasons.append(f"{label} row {index} decode batch size is not 1")
        if _replay_rate(row) is None:
            reasons.append(f"{label} row {index} missing positive replay timing")
    return reasons


def _evaluate_replay_gate(
    rows: list[dict],
    reference_rows: list[dict],
    *,
    min_runs: int,
) -> dict:
    """Gate B: q8/q8 forced replay, independent from sampled absolute speed."""
    accepted = _accepted(rows)
    references = _accepted(reference_rows)
    reasons = _replay_identity_reasons(accepted, "FreeToken replay")
    reasons.extend(_replay_identity_reasons(references, "reference replay"))
    if len(accepted) < min_runs:
        reasons.append(f"Gate B needs >= {min_runs} accepted replay rows, got {len(accepted)}")
    if len(references) < min_runs:
        reasons.append(f"Gate B needs >= {min_runs} accepted reference replay rows, got {len(references)}")
    for label, candidate in (("FreeToken replay", accepted), ("reference replay", references)):
        comparator_reasons, _ = _comparator_reasons(candidate, label=label)
        reasons.extend(comparator_reasons)

    by_repeat = {}
    for index, row in enumerate(references):
        key = row.get("repeat", index)
        if key in by_repeat:
            reasons.append(f"reference replay has duplicate repeat={key!r}")
        by_repeat[key] = row
    candidate_values: list[float] = []
    reference_values: list[float] = []
    deltas: list[float] = []
    for index, row in enumerate(accepted):
        value = _replay_rate(row)
        if value is not None and value < 70:
            reasons.append(f"Gate B replay run below 70 tok/s: {value:.3f}")
        key = row.get("repeat", index)
        ref = by_repeat.get(key)
        if ref is None:
            reasons.append(f"Gate B missing paired reference for repeat={key!r}")
            continue
        if not isinstance(row.get("route_digest"), dict) or row.get("route_digest") != ref.get("route_digest"):
            reasons.append(f"Gate B route hash mismatch at repeat={key!r}")
        for identity in ("model_sha256", "fixture_sha256", "tokenizer_sha256", "manifest_ids_sha256"):
            if row.get(identity) != ref.get(identity):
                reasons.append(f"Gate B {identity} mismatch at repeat={key!r}")
        reference_value = _replay_rate(ref)
        if value is not None and reference_value is not None:
            candidate_values.append(value)
            reference_values.append(reference_value)
            deltas.append(value - reference_value)
    low, high = _bootstrap_interval(deltas, seed=20260831)
    if len(deltas) < min_runs:
        reasons.append(f"Gate B needs >= {min_runs} complete replay pairs, got {len(deltas)}")
    elif not low > 0:
        reasons.append(f"Gate B paired median delta bootstrap interval [{low:.3f}, {high:.3f}] does not exceed zero")
    return {
        "gate": not reasons,
        "lane": LANE_REPLAY,
        "runs": {
            "accepted": len(accepted),
            "reference_accepted": len(references),
            "paired": len(deltas),
            "required": min_runs,
            "median_tok_s": statistics.median(candidate_values) if candidate_values else None,
            "reference_median_tok_s": statistics.median(reference_values) if reference_values else None,
            "paired_median_delta_tok_s": statistics.median(deltas) if deltas else None,
            "paired_delta_p02_5_bootstrap_tok_s": low if deltas else None,
            "paired_delta_p97_5_bootstrap_tok_s": high if deltas else None,
        },
        "reasons": reasons,
    }


def evaluate_gate(
    freetoken_rows: list[dict],
    reference_rows: list[dict] | None = None,
    probe: dict | None = None,
    *,
    min_runs: int = 10,
    directional_threshold: float = 80.225,
) -> dict:
    """Evaluate Gate A sampled and Gate B replay lanes independently."""
    reference_rows = reference_rows or []
    sampled_rows = [row for row in freetoken_rows if _row_lane(row) == LANE_SAMPLED]
    replay_rows = [row for row in freetoken_rows if _row_lane(row) == LANE_REPLAY]
    gate_a = _evaluate_sampled_gate(
        sampled_rows, reference_rows, probe,
        min_runs=min_runs, directional_threshold=directional_threshold,
    )
    gate_b = None
    if replay_rows:
        gate_b = _evaluate_replay_gate(
            replay_rows,
            [row for row in reference_rows if _row_lane(row) == LANE_REPLAY],
            min_runs=min_runs,
        )
    result = dict(gate_a)
    result["gate_a"] = {
        "gate": gate_a["gate"],
        "runs": gate_a["runs"],
        "reasons": gate_a["reasons"],
    }
    result["gate_b"] = gate_b
    result["gates"] = {
        "gate_a": gate_a["gate"],
        "gate_b": None if gate_b is None else gate_b["gate"],
    }
    result["lane_counts"] = {
        LANE_SAMPLED: len(sampled_rows),
        LANE_GREEDY: sum(_row_lane(row) == LANE_GREEDY for row in freetoken_rows),
        LANE_REPLAY: len(replay_rows),
    }
    if gate_b is not None:
        result["gate"] = gate_a["gate"] and gate_b["gate"]
        result["reasons"] = gate_a["reasons"] + [
            f"Gate B: {reason}" for reason in gate_b["reasons"]
        ]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freetoken", required=True)
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("--probe", help="finite-logit/eager-graph parity probe JSON")
    parser.add_argument("--min-runs", type=int, default=10)
    parser.add_argument("--json", dest="json_out")
    args = parser.parse_args(argv)
    try:
        reference_rows = [row for path in args.reference for row in load_jsonl(path)]
        probe = json.loads(Path(args.probe).read_text()) if args.probe else None
        freetoken_rows = load_jsonl(args.freetoken)
        result = evaluate_gate(freetoken_rows, reference_rows, probe, min_runs=args.min_runs)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "reference_identity": None,
            "probe": {"provided": bool(args.probe), "valid": False, "reasons": []},
            "threshold_source": "input-error",
            "threshold_tok_s": None,
            "runs": {
                "accepted": 0,
                "required": args.min_runs,
                "median_tok_s": None,
                "p02_5_bootstrap_tok_s": None,
                "min_tok_s": None,
                "max_tok_s": None,
            },
            "rejected": 0,
            "gate": False,
            "gate_a": {"gate": False, "runs": {}, "reasons": []},
            "gate_b": None,
            "gates": {"gate_a": False, "gate_b": None},
            "lane_counts": {LANE_SAMPLED: 0, LANE_GREEDY: 0, LANE_REPLAY: 0},
            "reasons": [f"invalid gate input: {type(exc).__name__}: {exc}"],
        }
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    print(encoded)
    if args.json_out:
        Path(args.json_out).write_text(encoded + "\n")
    return 0 if result["gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
