"""Generic AIME end-to-end gate through the production engine.

Runs one or more AIME problems (aime24 / aime25 / aime26) against a real local
checkpoint and checks that the reasoning model reaches the expected answer. The
sampling protocol is always taken from the checkpoint's own
``generation_config.json`` -- pass@N at the recommended temperature, or a single
deterministic sample when (and only when) the checkpoint itself recommends greedy.

Gated behind ``needs_weights``: set ``FREETOKEN_TEST_MODEL`` to the model dir and
the series jsonl env var (see below), and run on an idle GPU.

Env knobs:
  FREETOKEN_TEST_MODEL          local model directory (required)
  FREETOKEN_AIME_SERIES         aime24 | aime25 | aime26 (default aime25)
  FREETOKEN_AIME{24,25,26}_JSONL  path to the chosen series' jsonl (required)
  FREETOKEN_AIME_REQ            1-based problem id, comma list, or 'all' (default '1')
  FREETOKEN_AIME_MAX_TOKENS     per-sample token budget (default 16384)
  FREETOKEN_AIME_SAMPLES        pass@N sample count when sampling (default 3)
  FREETOKEN_AIME_MIN_FREE_GIB   free-GPU-memory gate (default 70; raise for a big resident model)
  FREETOKEN_TEST_MOE_CACHE_SIZE >0 switches to the offload MoE backend (fp8/GLM/MiniMax)
  FREETOKEN_TEST_MOE_BACKEND    offload-family backend override (default offload; e.g. hybrid)
  FREETOKEN_TEST_MOE_CPU_THREADS  CPU MoE worker threads (0 = physical cores)
  FREETOKEN_TEST_MOE_CACHE_AUTO 1 sizes the MoE cache from free VRAM instead of MOE_CACHE_SIZE
  FREETOKEN_TEST_KV_RESERVE     KV token floor reserved before auto-cache fills experts
  FREETOKEN_TEST_MEM_RATIO      offload memory ratio (default 0.9)

fp8 / offload recipe: point FREETOKEN_TEST_MODEL at the fp8 checkpoint dir and set
FREETOKEN_TEST_MOE_CACHE_SIZE (e.g. 8192).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from freetoken.core import SamplingParams
from freetoken.llm import LLM

pytestmark = pytest.mark.needs_weights

# Series -> env var holding that series' jsonl path. The aime25 name is unchanged.
AIME_SERIES_JSONL_ENV = {
    "aime24": "FREETOKEN_AIME24_JSONL",
    "aime25": "FREETOKEN_AIME25_JSONL",
    "aime26": "FREETOKEN_AIME26_JSONL",
}


def _optional_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value).expanduser()


def series_jsonl_env(series: str) -> str:
    """Env var holding the jsonl path for an AIME series. Raises ValueError for unknown series."""
    try:
        return AIME_SERIES_JSONL_ENV[series]
    except KeyError:
        raise ValueError(
            f"unknown FREETOKEN_AIME_SERIES {series!r}; want one of {sorted(AIME_SERIES_JSONL_ENV)}"
        )


def parse_req_selection(value: str, n_rows: int) -> list[int]:
    """Parse FREETOKEN_AIME_REQ into 0-based row indices into a jsonl of ``n_rows`` problems.

    Ids are 1-based: ``'1'`` -> ``[0]``, ``'1,3,7'`` -> ``[0, 2, 6]``, ``'all'`` ->
    ``list(range(n_rows))``. Raises ValueError on junk or an out-of-range id.
    """
    value = value.strip()
    if value.lower() == "all":
        if n_rows == 0:
            raise ValueError("'all' selected but the jsonl has no rows")
        return list(range(n_rows))
    indices: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if not re.fullmatch(r"[0-9]+", part):
            raise ValueError(f"invalid req id {part!r} (want a 1-based id, comma list, or 'all')")
        one_based = int(part)
        if not 1 <= one_based <= n_rows:
            raise ValueError(f"req {one_based} out of range 1..{n_rows}")
        indices.append(one_based - 1)
    if not indices:
        raise ValueError(f"empty req selection {value!r}")
    return indices


def max_tokens() -> int:
    return int(os.environ.get("FREETOKEN_AIME_MAX_TOKENS", "16384"))


def extract_answer(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()

    sum_match = re.findall(r"sum\s+is\s+(?:-?\d+\s*\+\s*)*-?\d+\s*=\s*(-?\d+)", text)
    if sum_match:
        return sum_match[-1].strip()

    answer_match = re.findall(r"(?:final answer|answer is|final result)\D{0,40}(-?\d+)", text, re.I)
    if answer_match:
        return answer_match[-1].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        integer_line = re.fullmatch(r"(?:\D*?)(-?\d+)\D*", lines[-1])
        if integer_line:
            return integer_line.group(1)
    return None


def recommended_sampling(model_dir: Path) -> tuple[SamplingParams, int]:
    """Sampling params from the model's recommended ``generation_config.json``, plus how many
    samples to draw for the self-consistency check.

    Reasoning models (GLM-4, Qwen3-thinking, ...) recommend *sampling*, not greedy -- greedy is
    off-distribution and chaotically sensitive to bf16-level kernel differences (a faithful
    kernel swap can derail the single greedy trajectory into a repetition loop). So we decode at
    the model's recommended temperature/top_p and check pass@N (the expected answer appears in
    at least one of N samples), which is robust to both sampling variance and numerically-
    equivalent kernel changes. Only when the checkpoint *itself* recommends greedy
    (``do_sample: false`` or ``temperature: 0``) do we fall back to a single deterministic
    sample -- that greedy choice is the checkpoint's own recommendation.
    """
    cfg: dict = {}
    gc = Path(model_dir) / "generation_config.json"
    if gc.is_file():
        cfg = json.loads(gc.read_text())
    temperature = float(cfg.get("temperature", 1.0))
    do_sample = bool(cfg.get("do_sample", temperature > 0.0))
    if not do_sample or temperature == 0.0:
        # Model recommends greedy -> deterministic, a single sample.
        return SamplingParams(temperature=0.0, max_tokens=max_tokens(), ignore_eos=False), 1
    # generation_config often omits top_p; 0.95 is the documented nucleus for these models.
    top_p = float(cfg.get("top_p", 0.95))
    top_k = int(cfg.get("top_k", -1))
    sp = SamplingParams(
        temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens(), ignore_eos=False
    )
    n_samples = int(os.environ.get("FREETOKEN_AIME_SAMPLES", "3"))
    return sp, n_samples


def build_llm(model_path: Path) -> LLM:
    """Resident by default; set FREETOKEN_TEST_MOE_CACHE_SIZE>0 to run an offload model
    (fp8 Qwen3.5-MoE, GLM-4.7, MiniMax, ... -- routed experts that do not fit in HBM)."""
    kwargs = dict(
        model_path=str(model_path),
        dtype=torch.bfloat16,
        # "auto" is what `ft serve` resolves, so the gate exercises the backend a model
        # actually ships with -- including the ones a hardcoded "fi" cannot reach at all
        # (DSV4 -> dsv4_sparse, SWA models -> triton).
        attention_backend="auto",
        max_running_req=1,
        max_extend_tokens=8192,
    )
    cache_size = int(os.environ.get("FREETOKEN_TEST_MOE_CACHE_SIZE", "0"))
    cache_auto = os.environ.get("FREETOKEN_TEST_MOE_CACHE_AUTO") == "1"
    if cache_size > 0 or cache_auto:
        kwargs.update(
            moe_backend=os.environ.get("FREETOKEN_TEST_MOE_BACKEND", "offload"),
            moe_cpu_threads=int(os.environ.get("FREETOKEN_TEST_MOE_CPU_THREADS", "0")),
            moe_cache_auto=cache_auto,
            moe_cache_size=cache_size,
            kv_reserve_tokens=int(os.environ.get("FREETOKEN_TEST_KV_RESERVE", "8192")),
            moe_cache_policy="lru",
            memory_ratio=float(os.environ.get("FREETOKEN_TEST_MEM_RATIO", "0.9")),
            max_seq_len_override=max_tokens() + 2048,
            cuda_graph_max_bs=1,
        )
    return LLM(**kwargs)


def visible_gpu_index() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return 0
    first = visible.split(",", 1)[0].strip()
    return int(first)


def free_gpu_memory_gib() -> float:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    target = visible_gpu_index()
    for line in result.stdout.splitlines():
        index, free_mib = [part.strip() for part in line.split(",", 1)]
        if int(index) == target:
            return int(free_mib) / 1024
    raise RuntimeError(f"GPU {target} is not visible in nvidia-smi")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AIME e2e needs CUDA")
def test_aime():
    model_path = _optional_path("FREETOKEN_TEST_MODEL")
    if model_path is None:
        pytest.skip("set FREETOKEN_TEST_MODEL to a local model directory")
    if not model_path.is_dir():
        pytest.skip(f"model is not downloaded: {model_path}")

    series = os.environ.get("FREETOKEN_AIME_SERIES", "aime25").strip().lower()
    try:
        jsonl_env = series_jsonl_env(series)
    except ValueError as e:
        pytest.fail(str(e))
    jsonl_path = _optional_path(jsonl_env)
    if jsonl_path is None or not jsonl_path.is_file():
        pytest.skip(f"set {jsonl_env} to the {series} jsonl file")

    # Enough for the documented fp8 + offload recipe on an 80 GiB card; a resident bf16 checkpoint
    # of this size needs more, so raise the knob rather than lowering the default further.
    min_free_gib = int(os.environ.get("FREETOKEN_AIME_MIN_FREE_GIB", "70"))
    free_gib = free_gpu_memory_gib()
    if free_gib < min_free_gib:
        pytest.skip(f"AIME e2e needs ~{min_free_gib} GiB free; only {free_gib:.2f} GiB")

    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    req_value = os.environ.get("FREETOKEN_AIME_REQ", "1")
    try:
        selected = parse_req_selection(req_value, len(rows))
    except ValueError as e:
        pytest.fail(f"FREETOKEN_AIME_REQ={req_value!r}: {e}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    # Build the engine once and reuse it across every selected problem.
    llm = build_llm(model_path)
    sampling, n_samples = recommended_sampling(model_path)

    failures: list[tuple[int, list[str | None], str]] = []
    for idx in selected:
        row = rows[idx]
        prompt, expected = row["prompt"], str(row["answer"])
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        # Decode at the model's recommended params and check pass@N: the expected answer must
        # appear in at least one of N samples (single greedy/sampled runs are brittle -- see
        # recommended_sampling docstring).
        answers: list[str | None] = []
        for _ in range(n_samples):
            outputs = llm.generate([formatted_prompt], sampling)
            answers.append(extract_answer(str(outputs[0]["text"])))
        if expected not in answers:
            failures.append((idx + 1, answers, expected))

    assert not failures, "AIME problems failed pass@N:\n" + "\n".join(
        f"  req {req_id}: answers={answers} expected={expected}"
        for req_id, answers, expected in failures
    )
