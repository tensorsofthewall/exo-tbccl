# Changelog

All notable user-facing changes are recorded here. The format follows Keep a Changelog, and the project follows Semantic Versioning once it has releases. exo-tbccl has had no final release. 0.3.0rc1 is a release candidate (pre-release), not production-ready.

## Unreleased (planned 0.3.0)

### Fixed

- The MLX bridge works with MLX builds whose Metal DLPack export carries the `id<MTLBuffer>` handle instead of the CPU-visible contents address (the 0.32.4 development build that upstream exo now pins on macOS). 0.3.0rc1 read the handle as an address, which crashed pipeline ranks on a Mac with that MLX (SIGSEGV/SIGBUS) or would corrupt memory. The convention is now probed once per process with an exact-content check and an unrecognized export is refused. 0.3.0rc1 is unaffected with the MLX it was validated with (0.32.0 development builds). The receive pool's poison control goes through the same resolution.

### Changed

- Validated against the exo integration re-derived on upstream exo `75d8ad2b` (2026-10-12), including exo's new persistent Worker (no rebuild on master election), at TBCCL wire protocol 4: a real two-host exo cluster with Qwen3-0.6B (three cycles) and Qwen3.8-27B-4bit (both pipeline orders).

## 0.3.0rc1 (release candidate)

First release candidate of the first planned release, 0.3.0. Expect an rc2 if a blocker is found.

### Installation

- CI-built wheels (Linux x86-64 manylinux_2_28, macOS arm64 14.0+, CPython 3.13) attached to the GitHub pre-release (not published on PyPI or TestPyPI), with an SPDX SBOM, `SHA256SUMS` and a build-provenance attestation. The wheel links TBCCL 0.6.0rc1 statically; install it into exo's virtual environment, for example `pip install --pre --no-deps -i https://test.pypi.org/simple/ exo-tbccl==0.3.0rc1`.

### Added

- A TBCCL data plane for exo pipeline-parallel text generation (`MlxTbccl` instance type) over a Thunderbolt 4 link or any TCP network between Metal, CUDA and CPU hosts.
- Zero-copy transfer of MLX arrays through DLPack where the platform allows it, with copies counted and visible in trace mode.
- Opt-in fast paths for the decode loop.

### Changed

- None.

### Fixed

- None.

### Compatibility

- Requires an installed TBCCL 0.6.0 or newer (C ABI 1; there is no public 0.5.x) and an exo build that provides the pipeline communication seam and the runner byte exchange; see `compatibility.json`.
- Python 3.13 or newer; validated with MLX 0.32.

### Known limitations

- Pipeline parallelism and text models only; no tensor parallelism and no image models.
- exo owns discovery, topology, placement and shard assignment.
- Experimental: inherits TBCCL's security model (no authentication or encryption).
