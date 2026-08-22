"""Pin-exempt layer policy spec parser.

Pure standard library.  No ``torch``, no ``freetoken`` imports.
This module is importable by argument-parsing code and by tests without
pulling in the CUDA stack.
"""

from __future__ import annotations


def parse_layer_subset_spec(
    spec: str, num_moe_layers: int, *, flag: str = "--pin-exempt-layers"
) -> frozenset[int]:
    """Parse a layer subset spec.

    Grammar (mirrors ``_parse_cpu_layers_spec``):

    * Contains a comma -> explicit id list, e.g. ``"3,7,11"``.
    * Contains a dot (no comma) -> fraction in ``[0.0, 1.0]``.
    * Otherwise -> integer count ``k`` with ``0 <= k <= num_moe_layers``.

    Count and fraction resolve to *k* layers evenly strided across depth via
    ``frozenset(round(i * num_moe_layers / k) for i in range(k))``.
    ``k == 0`` yields the empty frozenset.
    """
    s = spec.strip()
    if not s:
        return frozenset()
    if "," in s:
        ids = {int(x) for x in s.split(",") if x.strip()}
        for i in ids:
            if not 0 <= i < num_moe_layers:
                raise ValueError(
                    f"{flag} id {i} out of range [0, {num_moe_layers})"
                )
        return frozenset(ids)
    if "." in s:
        frac = float(s)
        if not 0.0 <= frac <= 1.0:
            raise ValueError(f"{flag} fraction {frac} must be in [0, 1]")
        k = round(frac * num_moe_layers)
    else:
        k = int(s)
        if not 0 <= k <= num_moe_layers:
            raise ValueError(
                f"{flag} count {k} must be in [0, {num_moe_layers}]"
            )
    # k layers spread evenly across depth (frozenset dedups any rounding
    # collisions; k == 0 yields an empty range, hence an empty set).
    return frozenset(round(i * num_moe_layers / k) for i in range(k))


def resolve_pin_exempt_layers(
    spec: str | None, num_moe_layers: int
) -> frozenset[int]:
    """Return the pin-exempt layer ids for *spec* and *num_moe_layers*.

    * ``None`` or empty/whitespace -> empty frozenset.
    * The literal ``"auto"`` (case-insensitive, surrounding whitespace
      ignored) -> raises ``NotImplementedError`` (planned v2: cached
      pin-quota profile).
    * Anything else -> delegate to :func:`parse_layer_subset_spec`.
    """
    if spec is None:
        return frozenset()
    s = spec.strip()
    if not s:
        return frozenset()
    if s.lower() == "auto":
        raise NotImplementedError(
            "Auto pin-exempt fit policy is a planned v2 feature "
            "(cached pin-quota profile). "
            "Please pass an explicit id list, count, or fraction."
        )
    return parse_layer_subset_spec(s, num_moe_layers)
