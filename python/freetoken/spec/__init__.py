"""Speculative decoding: SpecRunner (the target side) and one Proposer (the draft side) per method."""

from freetoken.utils import Registry

# method name -> "module:Class" of its Proposer, loaded on use so this package imports no method's code
PROPOSERS = Registry[str]("Speculative proposer")

__all__ = ["PROPOSERS"]
