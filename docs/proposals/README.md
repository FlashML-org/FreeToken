# FreeToken proposals

This directory records design changes that affect more than one model, backend,
or hardware platform. A proposal documents the compatibility contract before a
temporary implementation detail becomes a permanent public boundary.

## Status

- **Draft**: open for design and implementation review.
- **Accepted**: the maintainers agree on the boundary and rollout direction.
- **Implemented**: the accepted scope is present in the codebase.
- **Superseded**: a later proposal replaces the design.

## Index

| Proposal | Status | Initial platforms |
|---|---|---|
| [0001: Extract Hybrid MoE Decode Orchestration](0001-hybrid-decode-executor.md) | Draft | CUDA, ROCm |
