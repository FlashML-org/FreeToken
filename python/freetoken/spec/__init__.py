"""Speculative decoding: SpecRunner (the target side) and one Proposer (the draft side) per method."""

from freetoken.utils import Registry

# method name -> "module:Class" of its Proposer, loaded on use so this package imports no method's code
PROPOSERS = Registry[str]("Speculative proposer")
PROPOSERS.register("mtp")("freetoken.spec.mtp.proposer:MTPProposer")
PROPOSERS.register("dflash")("freetoken.spec.dflash.proposer:DFlashProposer")

# draft method name -> "module:Class" of the proposer it selects behind the CLI algorithm. A family
# opts in through its spec_drafts, so these are not --spec-algo names and carry no user-facing validation
DRAFT_PROPOSERS = {"shared_kv_mtp": "freetoken.spec.mtp.shared_proposer:SharedKVMTPProposer"}

__all__ = ["DRAFT_PROPOSERS", "PROPOSERS"]
