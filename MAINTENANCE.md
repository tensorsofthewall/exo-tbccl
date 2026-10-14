# Maintenance

The shared maintenance policy of the TBCCL ecosystem (branches, tag immutability, pinning, release cadence, security and correctness fixes, upstream updates, what needs physical requalification, automation and its failure categories) is [MAINTENANCE.md in tbccl](https://github.com/tensorsofthewall/tbccl/blob/main/MAINTENANCE.md); the versioning rules are in [VERSIONING.md](https://github.com/tensorsofthewall/tbccl/blob/main/VERSIONING.md). This file records only what is specific to exo-tbccl.

## Supported release

exo-tbccl 0.3.0: CPython 3.13, TBCCL 0.6.0 (C ABI 1), Linux x86-64 and macOS arm64, with the exo fork stack below. The current ecosystem status (all four projects and their pins) is in the [ecosystem page](https://tbccl.tensorsofthewall.com/en/latest/project/ecosystem.html), built from [`release/ecosystem.json`](https://github.com/tensorsofthewall/tbccl/blob/main/release/ecosystem.json). Documentation built from a release tag describes that release only and is not updated when other projects release.

## Pins

exo is installed from `tensorsofthewall/exo-fork` at the immutable tag `tbccl-integration-v1` (`1416807b46efd87c3601c69ce5784f51c9dd4c75`, upstream exo `75d8ad2bc1eac25aaf7adb39c0784ad0def8c1b6` plus the integration). `python3 tools/check_framework_pin.py` checks the pin offline; `--online` confirms the tag and that it still descends from the recorded upstream base.

## What needs physical requalification here

Runtime sources of this repository: `src/` and `exo_tbccl/`. A change to them, to the framework pin or to the supported TBCCL line needs physical requalification: a real exo pipeline run across the Mac and the Linux CUDA host from the public wheels and the pinned exo fork tag. Documentation, tests, CI, workflows, release metadata and GitHub Release edits do not. The full matrix is in the shared policy.

## Automation in this repository

`docs` workflow (every pull request, offline): release state, tooling tests, internal links, framework pin. `maintenance` workflow (weekly and by hand): release state, stable release surface, links, documentation hosts and the documentation build, the online framework pin. `release-preflight` workflow (by hand, with the intended tag). The `guard` job of `release-artifacts` refuses a build whose tag, version, manifest, pin or TBCCL line disagree. `tools/verify_public_release.py` verifies a published release from the outside, and `tools/collect_diagnostics.py` prints the environment report requested by the issue template.
