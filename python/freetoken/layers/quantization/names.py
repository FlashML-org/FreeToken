"""Module-name matching for the dialect configs and the attribute -> checkpoint name map."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import Callable, Iterator

Matcher = Callable[[str], bool]


def ancestors(name: str) -> Iterator[str]:
    while name:
        yield name
        name, _, _ = name.rpartition(".")


_GLOB_CHARS = re.compile(r"[*?\[]")


def name_set(patterns: tuple[str, ...]) -> Matcher:
    """Membership of a module or any parent in a list of literal names and glob patterns."""
    literal = frozenset(p for p in patterns if not _GLOB_CHARS.search(p))
    globs = [p for p in patterns if _GLOB_CHARS.search(p)]
    glob = re.compile("|".join(fnmatch.translate(p) for p in globs)) if globs else None

    def hit(name: str) -> bool:
        return any(a in literal or (glob is not None and glob.match(a)) for a in ancestors(name))

    return hit


def substr_set(patterns: tuple[str, ...]) -> Matcher:
    """transformers' fp8 integration skips a module when ``entry + "."`` occurs in ``name + "."``."""
    rx = re.compile("|".join(re.escape(p) + r"\." for p in patterns)) if patterns else None
    return lambda name: rx is not None and rx.search(name + ".") is not None


def ct_set(patterns: tuple[str, ...], *, class_names: bool) -> Matcher:
    """compressed-tensors ``targets`` / ``ignore``: module names, ``re:`` regexes, or the class name Linear."""
    names = name_set(tuple(p for p in patterns if not p.startswith("re:") and p != "Linear"))
    regexes = [p[3:] for p in patterns if p.startswith("re:")]
    rx = re.compile("|".join(f"(?:{r})" for r in regexes)) if regexes else None
    any_linear = class_names and "Linear" in patterns
    return lambda name: any_linear or names(name) or (rx is not None and rx.match(name) is not None)


_ROUTED_EXPERT = re.compile(r"\.experts\.\d+(\.|$)")


def is_routed_expert(name: str) -> bool:
    return bool(_ROUTED_EXPERT.search(name)) or name.endswith(".experts")


@dataclass(frozen=True)
class NameMap:
    """FreeToken attribute path -> checkpoint module name(s), applied to every scheme_for query.

    ``roots`` rename a leading path (``model.layers`` -> ``model.language_model.layers``);
    ``segments`` rename an inner path (``feed_forward.shared_mlp`` -> ``mlp``);
    ``packed`` expands a fused leaf into its source leaves (``qkv_proj`` -> q/k/v_proj).
    """

    roots: tuple[tuple[str, str], ...] = ()
    segments: tuple[tuple[str, str], ...] = ()
    packed: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def to_checkpoint(self, prefix: str) -> tuple[str, ...]:
        for attr_root, ckpt_root in sorted(self.roots, key=lambda r: -len(r[0])):
            if prefix == attr_root or prefix.startswith(attr_root + "."):
                prefix = ckpt_root + prefix[len(attr_root):]
                break
        for attr_seg, ckpt_seg in self.segments:
            prefix = _replace_segment(prefix, attr_seg, ckpt_seg)
        head, _, leaf = prefix.rpartition(".")
        sources = dict(self.packed).get(leaf)
        if not sources:
            return (prefix,)
        return tuple(f"{head}.{s}" if head else s for s in sources)


def _replace_segment(prefix: str, attr_seg: str, ckpt_seg: str) -> str:
    parts, seg, out, i = prefix.split("."), attr_seg.split("."), [], 0
    while i < len(parts):
        if parts[i : i + len(seg)] == seg:
            out.extend(ckpt_seg.split(".") if ckpt_seg else [])
            i += len(seg)
        else:
            out.append(parts[i])
            i += 1
    return ".".join(out)
