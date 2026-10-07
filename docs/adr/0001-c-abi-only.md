# ADR 0001: Link only TBCCL's stable C ABI

- Status: accepted
- Date: 2026-10-04

## Context

exo is an MLX-based Python system. TBCCL's C++ API is not an ABI promise, and PyTorch is deliberately not in this data path.

## Decision

exo-tbccl links only the C ABI (`TBCCL::tbccl_c`) of an installed TBCCL of version 0.5 or newer: no TBCCL C++ header, no source-tree include, no private symbol, no private copy of TBCCL, and no PyTorch or MLX C++ internals (arrays enter through the Python DLPack protocol). exo owns discovery, topology, placement, shard assignment, model loading and scheduling; this package never rediscovers interfaces and never derives rank from node ordering. TBCCL is not changed for exo's convenience: a generic defect is fixed in TBCCL as its own change, and no exo-specific API is added there.

## Consequences

- The package depends on the C ABI version only; a TBCCL wire protocol change needs a rebuild against the new prefix, not code changes.
- Scope is pipeline parallelism and text models.
