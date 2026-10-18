# Changelog

All notable user-facing changes are recorded here. The format follows Keep a Changelog, and the project follows Semantic Versioning once it has releases. The first final release is 0.3.0; the current release is 0.3.1. Both are experimental. 0.3.0rc1 and 0.3.0rc2 were release candidates (pre-releases) and are superseded.

Two final wheels are in use on purpose: 0.3.1 (bundles TBCCL 0.6.1) is the current standalone wheel, and 0.3.0 (bundles TBCCL 0.6.0) is what the `tbccl` extra of the immutable `tbccl-integration-v2` fork tag installs. The two have the same adapter code and the same C ABI 1 / wire protocol 4. The 0.3.0 installation through the extra is intentional, not a stale dependency: a tag cannot be changed.

## 0.3.1

Patch release. It adds the public Exo benchmark provider and its documentation. There is no change to the adapter (`MlxTbccl`), the Exo fork pin (`tbccl-integration-v2`, and `tbccl-integration-v1`), the validated MLX versions, the Python version (CPython 3.13), the TBCCL C ABI (1) or the wire protocol (4), and nothing in this release is a performance improvement. The wheels are built against TBCCL 0.6.1 (a patch release of 0.6.0 without runtime change); the minimum TBCCL package stays 0.6.0.

### Installation

- The same two CI-built wheels as 0.3.0, named `exo_tbccl-0.3.1-...`, attached to the GitHub release with an SPDX SBOM, `SHA256SUMS` and a build-provenance attestation.
- The immutable `tbccl-integration-v2` tag of the Exo fork keeps resolving its `tbccl` extra to the 0.3.0 wheel, which has the same adapter code; install the 0.3.1 wheel by hand to have 0.3.1 in the environment.
- The benchmark provider is **not** in the wheel. It is the `benchmarks/provider` directory of the source of this release (the source archive of the tag `v0.3.1`); the shared runner from the `tbccl` repository runs it with the interpreter of your Exo environment. The provider records the installed adapter version and the hashes of its own scripts, so a 0.3.1 provider running with the 0.3.0 wheel is visible in the result.

### Added

- A benchmark provider (provider protocol 1) for the shared `tbccl` benchmark suite: streamed Exo requests on the `MlxRing` and `MlxTbccl` backends with matched placements, TTFT/TPOT/ITL/end-to-end latency, concurrency scaling, backend and placement verification, strict stream accounting and lifecycle cleanup, with `describe` and `validate` steps. The experimental activity-step policy is configurable and is off by default.
- Benchmark documentation: the workloads, how they are timed, the measured reference results (including a 27B short-context run) with their hardware, software and limits, and a quickstart.

### Compatibility

- CPython 3.13, TBCCL 0.6.0 or newer (C ABI 1, wire protocol 4), Linux x86-64 and macOS arm64, with the Exo fork stack named in the compatibility page.

### Known limitations

- The reference benchmark results were measured on one Linux CUDA / Apple Metal machine pair. The activity-step comparison and the 27B results are exploratory.

## 0.3.0

First release. Promoted from release candidate 0.3.0rc2 without a code change. Wire protocol and C ABI are unchanged (C ABI 1, wire protocol 4).

### Installation

- exo is installed from the maintained fork at an immutable tag: `tensorsofthewall/exo-fork@tbccl-integration-v1` (1416807b, upstream exo `75d8ad2b` plus the integration). The earlier unpublished integration series is gone from the instructions.

### Fixed

- The MLX bridge works with MLX builds whose Metal DLPack export carries the `id<MTLBuffer>` handle instead of the CPU-visible contents address (the 0.32.4 development build that upstream exo now pins on macOS). 0.3.0rc1 read the handle as an address, which crashed pipeline ranks on a Mac with that MLX (SIGSEGV/SIGBUS) or would corrupt memory. The convention is now probed once per process with an exact-content check and an unrecognized export is refused. 0.3.0rc1 is unaffected with the MLX it was validated with (0.32.0 development builds). The receive pool's poison control goes through the same resolution.

### Changed

- Validated against the exo integration re-derived on upstream exo `75d8ad2b` (2026-10-12), including exo's new persistent Worker (no rebuild on master election), at TBCCL wire protocol 4: a real two-host exo cluster with Qwen3-0.6B (three cycles) and Qwen3.8-27B-4bit (both pipeline orders).

## 0.3.0rc2 (release candidate)

Supersedes 0.3.0rc1. Wire protocol and C ABI are unchanged (C ABI 1, wire protocol 4); this release changes the MLX bridge, not the wire.

### Installation

- exo is installed from the maintained fork at an immutable tag: `tensorsofthewall/exo-fork@tbccl-integration-v1` (1416807b, upstream exo `75d8ad2b` plus the integration). The earlier unpublished integration series is gone from the instructions.

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
