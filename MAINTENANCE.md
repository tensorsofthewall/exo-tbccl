# Maintenance

The shared maintenance policy of the TBCCL ecosystem (branches, tag immutability, pinning, release cadence, security and correctness fixes, upstream updates, what needs physical requalification, automation and its failure categories) is [MAINTENANCE.md in tbccl](https://github.com/tensorsofthewall/tbccl/blob/main/MAINTENANCE.md); the versioning rules are in [VERSIONING.md](https://github.com/tensorsofthewall/tbccl/blob/main/VERSIONING.md). This file records only what is specific to exo-tbccl.

## Supported release

exo-tbccl 0.4.0: CPython 3.13 (the wheels are `cp313`), TBCCL 0.7.0 bundled (minimum 0.6.0; C ABI 1; the 0.3.0 wheel of the fork's `tbccl` extra bundles 0.6.0), Linux x86-64 and macOS arm64, with the exo fork stack below. The current ecosystem status (all four projects and their pins) is in the [ecosystem page](https://tbccl.tensorsofthewall.com/en/latest/project/ecosystem.html), built from [`release/ecosystem.json`](https://github.com/tensorsofthewall/tbccl/blob/main/release/ecosystem.json). Documentation built from a release tag describes that release only and is not updated when other projects release.

## Pins

exo is installed from `tensorsofthewall/exo-fork` at the immutable tag `tbccl-integration-v2` (`d1497b28b89618172d273e70e9ac7992811635f5`, upstream exo `bc412e2230f395e652c23426a5eb88dfa515769b` plus the integration; `tbccl-integration-v1` is the legacy tag and is never changed). `python3 tools/check_framework_pin.py` checks the pin offline; `--online` confirms the tag and that it still descends from the recorded upstream base.

## What needs physical requalification here

Runtime sources of this repository: `src/` and `exo_tbccl/`. A change to them, to the framework pin or to the supported TBCCL line needs physical requalification: a real exo pipeline run across the Mac and the Linux CUDA host from the public wheels and the pinned exo fork tag. Documentation, tests, CI, workflows, release metadata and GitHub Release edits do not. The full matrix is in the shared policy.

## Automation in this repository

`docs` workflow (every pull request, offline): release state, release documentation (README, homepage and installation commands against the package version), tooling tests, internal links, framework pin. `maintenance` workflow (weekly and by hand): release state, stable release surface, links, documentation hosts and the documentation build, the online framework pin. `release-preflight` workflow (by hand, with the intended tag; it checks the release documentation against that tag before anything is tagged). The `guard` job of `release-artifacts` refuses a build whose tag, version, manifest, pin or TBCCL line disagree. `tools/check_trust_dependencies.py` (in the `docs`, `maintenance` and `release-preflight` workflows) rejects an affected certifi, uv or container pin, pip used without the hash-pinned bootstrap and any TLS-verification bypass; the build environment policy is in the shared maintenance policy. `tools/verify_public_release.py` verifies a published release from the outside, and `tools/collect_diagnostics.py` prints the environment report requested by the issue template.

## Pin integrity versus current-ecosystem consistency

`tools/check_framework_pin.py --online` reports two separate things. Release pin integrity (the immutable fork tag still resolves to the SHA recorded in `compatibility.json`) fails hard whatever the release state. Current-ecosystem consistency (this repository against `release/ecosystem.json` on tbccl `main`) fails only for the published stable version; for an unreleased candidate it is a warning that the ecosystem file has not yet been updated, and `--ecosystem-file` or `--ecosystem-ref` compares against a candidate ecosystem instead. See the [maintenance policy](https://github.com/tensorsofthewall/tbccl/blob/main/MAINTENANCE.md).
