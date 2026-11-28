"""A JSON Schema as a byte-level DFA over the JSON texts the decoder may emit.

The DFA accepts a subset of the texts that validate: properties in the schema's order and never an extra one, at
most WS_MAX whitespace bytes in a gap, numbers without exponents and with at most 19 integer and 15 fraction digits,
strings without escaped surrogates, literals (enum, const) in json.dumps's spelling. Every text it accepts parses
and validates. A keyword it cannot enforce is refused (SchemaError), never ignored.

Supported keywords:
  type             string, integer, number, boolean, null, object, array, or a list of them
  enum, const      any JSON values; beside them only type (which filters them) and annotations
  string           minLength, maxLength (code points)
  integer          minimum, maximum, exclusiveMinimum, exclusiveMaximum (numbers)
  number           minimum >= 0 and maximum, both integral, or no bounds
  object           properties, required (names in properties), additionalProperties (extras are never emitted)
  array            items (one schema), minItems, maxItems
  anyOf            any branches; oneOf when the branches' JSON types are disjoint; allOf of one schema
  $ref             "#/$defs/<name>" or "#/definitions/<name>", not recursive
  annotations      title, description, default, examples, $schema, $id, $comment, format (not enforced, as JSON
                   Schema 2020-12's default), deprecated, readOnly, writeOnly
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

WS_MAX = 16
INT_DIGITS = 19
FRAC_DIGITS = 15
# subset construction stops here: a schema this large is refused
MAX_DFA_STATES = 50_000
MAX_NFA_STATES = 500_000

_ANNOTATIONS = frozenset({
    "title", "description", "default", "examples", "$schema", "$id", "$comment", "format", "deprecated",
    "readOnly", "writeOnly", "$defs", "definitions",
})
_TYPES = ("string", "integer", "number", "boolean", "null", "object", "array")
_BY_TYPE = {
    "string": {"minLength", "maxLength"},
    "integer": {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"},
    "number": {"minimum", "maximum"},
    "boolean": set(),
    "null": set(),
    "object": {"properties", "required", "additionalProperties"},
    "array": {"items", "minItems", "maxItems"},
}


class SchemaError(ValueError):
    """A schema outside the supported subset, or not a schema."""


def _bits(*ranges: Tuple[int, int]) -> int:
    mask = 0
    for lo, hi in ranges:
        for b in range(lo, hi + 1):
            mask |= 1 << b
    return mask


def _byte(b: int) -> int:
    return 1 << b


_DIGIT = _bits((0x30, 0x39))
_NONZERO = _bits((0x31, 0x39))
_HEX = _bits((0x30, 0x39), (0x41, 0x46), (0x61, 0x66))
_WS = _bits((0x20, 0x20), (0x09, 0x0A), (0x0D, 0x0D))
_CONT = _bits((0x80, 0xBF))
# a string's unescaped ASCII: printable, DEL, not '"' or '\'
_PLAIN = _bits((0x20, 0x21), (0x23, 0x5B), (0x5D, 0x7F))


class _NFA:
    """Thompson NFA built back to front: each builder takes the state its text continues to and returns its entry."""

    def __init__(self) -> None:
        self.eps: List[List[int]] = [[]]
        self.edges: List[List[Tuple[int, int]]] = [[]]
        self.accept = 0

    def new(self) -> int:
        if len(self.eps) >= MAX_NFA_STATES:
            raise SchemaError(f"the schema needs more than {MAX_NFA_STATES} automaton states")
        self.eps.append([])
        self.edges.append([])
        return len(self.eps) - 1

    def edge(self, mask: int, nxt: int) -> int:
        s = self.new()
        self.edges[s].append((mask, nxt))
        return s

    def split(self, *targets: int) -> int:
        s = self.new()
        self.eps[s].extend(targets)
        return s


Builder = Callable[[int], int]


def _lit(nfa: _NFA, data: bytes) -> Builder:
    def build(nxt: int) -> int:
        for b in reversed(data):
            nxt = nfa.edge(_byte(b), nxt)
        return nxt
    return build


def _cls(nfa: _NFA, *masks: int) -> Builder:
    """A run of one byte from each mask in turn."""
    def build(nxt: int) -> int:
        for m in reversed(masks):
            nxt = nfa.edge(m, nxt)
        return nxt
    return build


def _seq(*parts: Builder) -> Builder:
    def build(nxt: int) -> int:
        for part in reversed(parts):
            nxt = part(nxt)
        return nxt
    return build


def _alt(nfa: _NFA, *parts: Builder) -> Builder:
    def build(nxt: int) -> int:
        return nfa.split(*(part(nxt) for part in parts))
    return build


def _rep(nfa: _NFA, part: Builder, lo: int, hi: int | None) -> Builder:
    """``part`` lo to hi times (hi None: no bound)."""
    def build(nxt: int) -> int:
        if hi is None:
            loop = nfa.split(nxt)
            nfa.eps[loop].append(part(loop))
            cur = loop
        else:
            cur = nxt
            for _ in range(hi - lo):
                cur = nfa.split(nxt, part(cur))
        for _ in range(lo):
            cur = part(cur)
        return cur
    return build


def _ws(nfa: _NFA) -> Builder:
    return _rep(nfa, _cls(nfa, _WS), 0, WS_MAX)


def _string_char(nfa: _NFA) -> Builder:
    """One code point of a JSON string's body: unescaped ASCII, an escape, or a well-formed UTF-8 sequence."""
    escape = _seq(
        _lit(nfa, b"\\"),
        _alt(
            nfa,
            _cls(nfa, _bits((0x22, 0x22), (0x2F, 0x2F), (0x5C, 0x5C), (0x62, 0x62), (0x66, 0x66), (0x6E, 0x6E), (0x72, 0x72), (0x74, 0x74))),
            # \uXXXX outside D800-DFFF: a lone surrogate does not parse everywhere
            _seq(_lit(nfa, b"u"), _alt(
                nfa,
                _cls(nfa, _bits((0x30, 0x39), (0x41, 0x43), (0x45, 0x46), (0x61, 0x63), (0x65, 0x66)), _HEX, _HEX, _HEX),
                _cls(nfa, _bits((0x44, 0x44), (0x64, 0x64)), _bits((0x30, 0x37)), _HEX, _HEX),
            )),
        ),
    )
    utf8 = _alt(
        nfa,
        _cls(nfa, _bits((0xC2, 0xDF)), _CONT),
        _cls(nfa, _byte(0xE0), _bits((0xA0, 0xBF)), _CONT),
        _cls(nfa, _bits((0xE1, 0xEC), (0xEE, 0xEF)), _CONT, _CONT),
        _cls(nfa, _byte(0xED), _bits((0x80, 0x9F)), _CONT),
        _cls(nfa, _byte(0xF0), _bits((0x90, 0xBF)), _CONT, _CONT),
        _cls(nfa, _bits((0xF1, 0xF3)), _CONT, _CONT, _CONT),
        _cls(nfa, _byte(0xF4), _bits((0x80, 0x8F)), _CONT, _CONT),
    )
    return _alt(nfa, _cls(nfa, _PLAIN), escape, utf8)


def _digits_between(nfa: _NFA, lo: str, hi: str) -> Builder:
    """Decimal strings of len(lo) digits from lo to hi, leading zeros allowed (the caller fixes the first digit)."""
    if not lo:
        return _seq()
    if lo[0] == hi[0]:
        return _seq(_lit(nfa, lo[0].encode()), _digits_between(nfa, lo[1:], hi[1:]))
    n = len(lo) - 1
    a, b = int(lo[0]), int(hi[0])
    parts = []
    # a full tail joins the first digit to the middle run; only a partial one recurses, so the size stays linear
    if lo[1:] != "0" * n:
        parts.append(_seq(_lit(nfa, lo[0].encode()), _digits_between(nfa, lo[1:], "9" * n)))
        a += 1
    if hi[1:] != "9" * n:
        parts.append(_seq(_lit(nfa, hi[0].encode()), _digits_between(nfa, "0" * n, hi[1:])))
        b -= 1
    if a <= b:
        parts.append(_cls(nfa, _bits((0x30 + a, 0x30 + b)), *([_DIGIT] * n)))
    return _alt(nfa, *parts)


def _naturals(nfa: _NFA, lo: int, hi: int) -> Builder:
    """The canonical decimal spellings of lo..hi, 0 <= lo <= hi."""
    parts = []
    for n in range(len(str(lo)), len(str(hi)) + 1):
        a = max(lo, 10 ** (n - 1) if n > 1 else 0)
        b = min(hi, 10**n - 1)
        if a <= b:
            parts.append(_digits_between(nfa, str(a), str(b)))
    return _alt(nfa, *parts)


def _integers(nfa: _NFA, lo: int, hi: int) -> Builder:
    parts = []
    if hi >= 0:
        parts.append(_naturals(nfa, max(lo, 0), hi))
    if lo < 0:
        parts.append(_seq(_lit(nfa, b"-"), _naturals(nfa, max(-hi, 1), -lo)))
    return _alt(nfa, *parts)


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "integer" if value.is_integer() else "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    raise SchemaError(f"not a JSON value: {value!r}")


def _number(x: Any, key: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise SchemaError(f"{key} must be a number, got {x!r}")
    return x


def _count(x: Any, key: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or x < 0:
        raise SchemaError(f"{key} must be a non-negative integer, got {x!r}")
    return x


class _Compiler:
    def __init__(self, root: Any) -> None:
        self.root = root
        self.nfa = _NFA()
        self._refs: List[str] = []

    def resolve(self, ref: Any) -> Any:
        if not isinstance(ref, str):
            raise SchemaError(f"$ref must be a string, got {ref!r}")
        for prefix in ("#/$defs/", "#/definitions/"):
            if ref.startswith(prefix):
                defs = self.root.get(prefix[2:-1]) if isinstance(self.root, dict) else None
                name = ref[len(prefix):]
                if isinstance(defs, dict) and name in defs and "/" not in name:
                    return defs[name]
        raise SchemaError(f"$ref {ref!r} is not supported (only #/$defs/<name> and #/definitions/<name>)")

    def types(self, schema: Any) -> set:
        """The JSON types a schema's texts can have, for oneOf's disjointness check."""
        if not isinstance(schema, dict):
            raise SchemaError(f"a schema must be an object, got {schema!r}")
        if "$ref" in schema:
            return self.types(self.resolve(schema["$ref"]))
        if "enum" in schema or "const" in schema:
            values = schema["enum"] if "enum" in schema else [schema["const"]]
            return {_json_type(v) for v in values if isinstance(values, list)}
        for key in ("anyOf", "oneOf", "allOf"):
            if key in schema and isinstance(schema[key], list):
                return set().union(*(self.types(s) for s in schema[key]))
        t = schema.get("type", self._implied_type(schema))
        return set(t) if isinstance(t, list) else {t}

    @staticmethod
    def _implied_type(schema: dict) -> str | None:
        found = [t for t, keys in _BY_TYPE.items() if keys and keys & schema.keys() and t != "number"]
        return found[0] if len(found) == 1 else None

    def value(self, schema: Any) -> Builder:
        if not isinstance(schema, dict):
            raise SchemaError(f"a schema must be an object, got {json.dumps(schema)[:60]}")
        if "$ref" in schema:
            ref = schema["$ref"]
            extra = schema.keys() - _ANNOTATIONS - {"$ref"}
            if extra:
                raise SchemaError(f"$ref beside {sorted(extra)} is not supported")
            if ref in self._refs:
                raise SchemaError(f"recursive $ref {ref!r} is not supported")
            self._refs.append(ref)
            try:
                return self.value(self.resolve(ref))
            finally:
                self._refs.pop()
        for key in ("anyOf", "oneOf", "allOf"):
            if key in schema:
                return self.combined(schema, key)
        if "enum" in schema or "const" in schema:
            return self.literals(schema)
        t = schema.get("type", self._implied_type(schema))
        if t is None:
            raise SchemaError("a schema needs a type, enum or const (an unconstrained value is not supported)")
        if isinstance(t, list):
            if not t or len(set(t)) != len(t):
                raise SchemaError(f"type list {t!r} must be non-empty and without repeats")
            if "integer" in t and "number" in t:
                t = [x for x in t if x != "integer"]
            return _alt(self.nfa, *(self.typed(schema, x, set().union(*(_BY_TYPE.get(y, set()) for y in t))) for x in t))
        return self.typed(schema, t, _BY_TYPE.get(t, set()))

    def combined(self, schema: dict, key: str) -> Builder:
        branches = schema[key]
        extra = schema.keys() - _ANNOTATIONS - {key}
        if extra:
            raise SchemaError(f"{key} beside {sorted(extra)} is not supported")
        if not isinstance(branches, list) or not branches:
            raise SchemaError(f"{key} must be a non-empty list")
        if key == "allOf":
            if len(branches) != 1:
                raise SchemaError("allOf of more than one schema is not supported")
            return self.value(branches[0])
        if key == "oneOf":
            seen: set = set()
            for b in branches:
                kinds = self.types(b)
                if "number" in kinds or "integer" in kinds:
                    kinds = kinds | {"number", "integer"}
                if None in kinds or kinds & seen:
                    raise SchemaError("oneOf is supported only when its branches have different JSON types")
                seen |= kinds
        return _alt(self.nfa, *(self.value(b) for b in branches))

    def literals(self, schema: dict) -> Builder:
        extra = schema.keys() - _ANNOTATIONS - {"enum", "const", "type"}
        if extra:
            raise SchemaError(f"enum or const beside {sorted(extra)} is not supported")
        if "enum" in schema and "const" in schema:
            raise SchemaError("enum and const together are not supported")
        values = schema["enum"] if "enum" in schema else [schema["const"]]
        if not isinstance(values, list) or not values:
            raise SchemaError("enum must be a non-empty list")
        if "type" in schema:
            t = schema["type"]
            allowed = set(t) if isinstance(t, list) else {t}
            if "number" in allowed:
                allowed.add("integer")
            values = [v for v in values if _json_type(v) in allowed]
            if not values:
                raise SchemaError("no enum value has the schema's type")
        texts = {json.dumps(v, ensure_ascii=False).encode("utf-8") for v in values}
        return _alt(self.nfa, *(_lit(self.nfa, t) for t in sorted(texts)))

    def typed(self, schema: dict, t: Any, allowed: set) -> Builder:
        if t not in _TYPES:
            raise SchemaError(f"type {t!r} is not supported")
        extra = schema.keys() - _ANNOTATIONS - allowed - {"type"}
        if extra:
            raise SchemaError(f"keyword(s) {sorted(extra)} not supported for type {t!r}")
        nfa = self.nfa
        if t == "null":
            return _lit(nfa, b"null")
        if t == "boolean":
            return _alt(nfa, _lit(nfa, b"true"), _lit(nfa, b"false"))
        if t == "string":
            lo = _count(schema.get("minLength", 0), "minLength")
            hi = schema.get("maxLength")
            hi = None if hi is None else _count(hi, "maxLength")
            if hi is not None and hi < lo:
                raise SchemaError("maxLength is below minLength")
            return _seq(_lit(nfa, b'"'), _rep(nfa, _string_char(nfa), lo, hi), _lit(nfa, b'"'))
        if t == "integer":
            return self.integer(schema)
        if t == "number":
            return self.number(schema)
        if t == "array":
            return self.array(schema)
        return self.object(schema)

    def integer(self, schema: dict) -> Builder:
        top = 10**INT_DIGITS - 1
        lo, hi = -top, top
        if "minimum" in schema:
            lo = max(lo, math.ceil(_number(schema["minimum"], "minimum")))
        if "exclusiveMinimum" in schema:
            lo = max(lo, math.floor(_number(schema["exclusiveMinimum"], "exclusiveMinimum")) + 1)
        if "maximum" in schema:
            hi = min(hi, math.floor(_number(schema["maximum"], "maximum")))
        if "exclusiveMaximum" in schema:
            hi = min(hi, math.ceil(_number(schema["exclusiveMaximum"], "exclusiveMaximum")) - 1)
        if lo > hi:
            raise SchemaError("no integer satisfies the schema's bounds")
        return _integers(self.nfa, lo, hi)

    def number(self, schema: dict) -> Builder:
        nfa = self.nfa
        if "exclusiveMinimum" in schema or "exclusiveMaximum" in schema:
            raise SchemaError("exclusiveMinimum and exclusiveMaximum are supported only for integers")
        frac = _seq(_lit(nfa, b"."), _rep(nfa, _cls(nfa, _DIGIT), 1, FRAC_DIGITS))
        lo, hi = schema.get("minimum"), schema.get("maximum")
        if lo is None and hi is None:
            whole = _alt(nfa, _lit(nfa, b"0"), _seq(_cls(nfa, _NONZERO), _rep(nfa, _cls(nfa, _DIGIT), 0, INT_DIGITS - 1)))
            return _seq(_rep(nfa, _lit(nfa, b"-"), 0, 1), whole, _rep(nfa, frac, 0, 1))
        lo = _number(lo, "minimum") if lo is not None else None
        hi = _number(hi, "maximum") if hi is not None else None
        if lo is None or lo < 0 or not float(lo).is_integer() or (hi is not None and not float(hi).is_integer()):
            raise SchemaError("number bounds are supported only as an integral minimum >= 0 and an integral maximum")
        lo, top = int(lo), 10**INT_DIGITS - 1
        if hi is None:
            return _seq(_naturals(nfa, lo, top), _rep(nfa, frac, 0, 1))
        hi = int(hi)
        if hi < lo:
            raise SchemaError("maximum is below minimum")
        # I.F <= hi: I below hi with any fraction, or I == hi with a zero fraction
        at_top = _seq(_lit(nfa, str(hi).encode()), _rep(nfa, _seq(_lit(nfa, b"."), _rep(nfa, _lit(nfa, b"0"), 1, FRAC_DIGITS)), 0, 1))
        if hi == lo:
            return at_top
        return _alt(nfa, _seq(_naturals(nfa, lo, hi - 1), _rep(nfa, frac, 0, 1)), at_top)

    def array(self, schema: dict) -> Builder:
        nfa = self.nfa
        if "items" not in schema:
            raise SchemaError("an array needs an items schema")
        if not isinstance(schema["items"], dict):
            raise SchemaError("items must be one schema (tuple forms are not supported)")
        lo = _count(schema.get("minItems", 0), "minItems")
        hi = schema.get("maxItems")
        hi = None if hi is None else _count(hi, "maxItems")
        if hi is not None and hi < lo:
            raise SchemaError("maxItems is below minItems")
        item = self.value(schema["items"])
        ws = _ws(nfa)
        if hi == 0:
            return _seq(_lit(nfa, b"["), ws, _lit(nfa, b"]"))
        more = _rep(nfa, _seq(ws, _lit(nfa, b","), ws, item), max(lo - 1, 0), None if hi is None else hi - 1)
        body = _seq(item, more)
        if lo == 0:
            body = _rep(nfa, body, 0, 1)
        return _seq(_lit(nfa, b"["), ws, body, ws, _lit(nfa, b"]"))

    def object(self, schema: dict) -> Builder:
        nfa = self.nfa
        props = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(props, dict) or not isinstance(required, list):
            raise SchemaError("properties must be an object and required a list")
        missing = [r for r in required if r not in props]
        if missing:
            raise SchemaError(f"required names {missing} are not in properties")
        items = [(name, self.value(sub), name in required) for name, sub in props.items()]
        ws = _ws(nfa)

        def kv(i: int) -> Builder:
            name, value, _ = items[i]
            key = json.dumps(name, ensure_ascii=False).encode("utf-8")
            return _seq(_lit(nfa, key), ws, _lit(nfa, b":"), ws, value)

        def build(nxt: int) -> int:
            close = _seq(ws, _lit(nfa, b"}"))(nxt)
            # first[i]: properties i.. with none written yet; rest[i]: with one written, so each starts with a comma
            first, rest = close, close
            for i in range(len(items) - 1, -1, -1):
                req = items[i][2]
                after = rest
                with_comma = _seq(ws, _lit(nfa, b","), ws, kv(i))(after)
                without = kv(i)(after)
                rest = with_comma if req else nfa.split(with_comma, rest)
                first = without if req else nfa.split(without, first)
            return _seq(_lit(nfa, b"{"), ws)(first)

        return build


@dataclass
class SchemaDFA:
    """``moves[s, byte_class[b]]``: the state after byte b from state s; state 0 is dead (it maps to itself), 1 the start."""

    moves: np.ndarray  # [S, byte classes] int32
    byte_class: np.ndarray  # [256] int64
    accepting: np.ndarray  # [S] bool

    @property
    def num_states(self) -> int:
        return self.moves.shape[0]

    def walk(self, state: int, data: bytes) -> int:
        moves, cls = self.moves, self.byte_class
        for b in data:
            if state == 0:
                break
            state = moves.item(state, cls.item(b))
        return state

    def accepts(self, data: bytes) -> bool:
        return bool(self.accepting[self.walk(1, data)])


def check_schema(schema: Any) -> Tuple[_NFA, int]:
    """The NFA of a schema and its entry state; raises SchemaError outside the subset. Cheap: no DFA."""
    compiler = _Compiler(schema)
    nfa = compiler.nfa
    try:
        entry = _seq(_ws(nfa), compiler.value(schema))(nfa.accept)
    except RecursionError:
        raise SchemaError("the schema is nested too deeply") from None
    return nfa, entry


def compile_schema(schema: Any) -> SchemaDFA:
    nfa, entry = check_schema(schema)
    masks = sorted({m for edges in nfa.edges for m, _ in edges})
    # byte classes: bytes no edge tells apart share a class
    keys: Dict[Tuple[bool, ...], int] = {}
    byte_class = np.zeros(256, dtype=np.int64)
    for b in range(256):
        byte_class[b] = keys.setdefault(tuple(bool(m >> b & 1) for m in masks), len(keys))
    reps = [0] * len(keys)
    for b in range(255, -1, -1):
        reps[byte_class[b]] = b
    moves: List[Dict[int, List[int]]] = []
    for edges in nfa.edges:
        by_class: Dict[int, List[int]] = {}
        for m, target in edges:
            for c, rep in enumerate(reps):
                if m >> rep & 1:
                    by_class.setdefault(c, []).append(target)
        moves.append(by_class)

    def closure(states) -> frozenset:
        seen = set(states)
        stack = list(states)
        while stack:
            for t in nfa.eps[stack.pop()]:
                if t not in seen:
                    seen.add(t)
                    stack.append(t)
        # only states with byte edges, and the accept state, tell DFA states apart
        return frozenset(s for s in seen if moves[s] or s == nfa.accept)

    start = closure([entry])
    ids: Dict[frozenset, int] = {frozenset(): 0, start: 1}
    order = [frozenset(), start]
    rows: List[List[int]] = [[0] * len(reps)]
    i = 1
    while i < len(order):
        current = order[i]
        row = []
        for c in range(len(reps)):
            targets = [t for s in current for t in moves[s].get(c, ())]
            nxt = closure(targets) if targets else frozenset()
            if nxt not in ids:
                if len(order) >= MAX_DFA_STATES:
                    raise SchemaError(f"the schema needs more than {MAX_DFA_STATES} automaton states")
                ids[nxt] = len(order)
                order.append(nxt)
            row.append(ids[nxt])
        rows.append(row)
        i += 1
    accepting = np.array([nfa.accept in s for s in order], dtype=bool)
    return SchemaDFA(np.asarray(rows, dtype=np.int32), byte_class, accepting)


def parse_response_format(response_format: Dict[str, Any] | None) -> str | None:
    """The schema text a request's OpenAI response_format asks to enforce, None for plain text; raises SchemaError.

    The text keeps the schema's key order: properties are emitted in it."""
    if response_format is None:
        return None
    kind = response_format.get("type")
    if kind in (None, "text"):
        return None
    if kind != "json_schema":
        raise SchemaError(f"response_format type {kind!r} is not supported (json_schema and text are)")
    spec = response_format.get("json_schema")
    if not isinstance(spec, dict) or not isinstance(spec.get("schema"), dict):
        raise SchemaError("response_format json_schema needs a json_schema object with a schema object")
    schema = spec["schema"]
    check_schema(schema)
    return json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
