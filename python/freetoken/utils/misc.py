from __future__ import annotations


def call_if_main(name: str = "__main__", discard: bool | None = None):
    """Decorator to ensure a function will call when the script is run as main."""
    if name != "__main__":
        discard = False if discard is None else discard
        if discard:
            return lambda _: None
        else:
            return lambda f: f
    else:
        discard = True if discard is None else discard
        if discard:
            return lambda f: (f() or True) and None
        else:
            return lambda f: (f() and None) or f


def opt_in_env(value: str | None) -> tuple[str, bool]:
    """An opt-in switch's environment value as (text, on): the value stripped and lowercased, "" when unset, and on
    only for 1, true, yes or on. A caller that logs its decision stays silent for "" (nobody asked)."""
    text = (value or "").strip().lower()
    return text, text in ("1", "true", "yes", "on")


def div_even(a: int, b: int, allow_replicate: bool = False) -> int:
    """Divides two integers. If allow_replicate=True, allows b > a when b % a == 0, returning 1."""
    if allow_replicate and b > a:
        assert b % a == 0, f"{b = } must be divisible by {a = } for KV head replication"
        return 1
    assert a % b == 0, f"{a = } must be divisible by {b = }"
    return a // b


def div_ceil(a: int, b: int) -> int:
    """Divides two integers, rounding up"""
    return (a + b - 1) // b


def align_ceil(a: int, b: int) -> int:
    """Aligns a to the next multiple of b"""
    return div_ceil(a, b) * b


def align_down(a: int, b: int) -> int:
    """Aligns a to the previous multiple of b"""
    return (a // b) * b


def mem_GB(size: int) -> str:
    """Format a byte count as a human-readable GiB string for logs."""
    return f"{size / (1024**3):.2f} GiB"


class Unset:
    pass


UNSET = Unset()
