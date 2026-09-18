"""Scheduler capability boundary for the structured-output contract.

The API may validate/transport a schema, but only the scheduler/sampler can enforce
it. Keep one explicit gate here so HTTP preflight and direct IPC callers cannot
silently diverge about support.
"""

from __future__ import annotations


def ensure_structured_output_supported(schema: dict | None) -> None:
    """Raise before admission until a real per-request grammar-mask backend exists."""
    if schema is not None:
        # TODO: integrate xgrammar's tokenizer-aware GrammarCompiler, per-request
        # GrammarMatcher state and a token mask BEFORE both greedy argmax and
        # temperature/top-k/top-p sampling. Respect abort and overlapped batches;
        # parsing a generated string afterward does not enforce a JSON Schema.
        raise NotImplementedError(
            "response_format json_schema is not supported by this scheduler yet "
            "(no grammar logits processor; TODO: integrate xgrammar token masks)"
        )
