# Security policy

## Supported versions

Security fixes are made against the latest release series only (currently 0.5.x). Development versions and unreleased branches are not supported.

## Package-manager and trust-store notice (October 2026)

Private keys of eight eMudhra intermediate certificates were reported compromised in October 2026, and the affected emSign root (emSign ECC Root CA - G3) was removed from the CA bundles that Python tooling uses. This is not a vulnerability of this adapter or of any published release: the wheels contain no CA bundle and perform no TLS. It concerns the tools that fetch packages and the environments that build them. Use at least certifi 2026.10.19, pip 26.2.2 and uv 0.13.4 in every environment where you install the adapter or its framework; upgrading the `certifi` package does not patch an older pip, so upgrade pip itself. The details and the minimums for the other tools are in the [TBCCL security policy](https://tbccl.tensorsofthewall.com/en/latest/project/security-policy.html). The release workflows of this repository install a specific pip (from its hash-pinned wheel) and uv, build in a patched manylinux image and block the revoked emSign root in its system bundle; `tools/check_trust_dependencies.py` rejects affected pins.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability.

Use GitHub private vulnerability reporting: open the repository's Security tab and choose "Report a vulnerability". If that option is not available, open an issue that says only that you have a security report and asks for a private channel; do not put details in it.

## What to include

- The affected version or commit and the platform.
- What you observed and what you expected.
- The smallest reproduction you can give: code, input bytes or steps.
- Whether you believe it is exploitable, and under which deployment assumptions.
- Whether and how you want to be credited.

Please do not include credentials, private keys or details of a private network.

## What to expect

This is a volunteer-maintained project. Reports are acknowledged on a best-effort basis, normally within a few days. A confirmed issue is fixed on the main branch first and released in the next release of the supported series; the advisory credits the reporter unless they ask otherwise. Please allow a reasonable time to fix a confirmed issue before you disclose it publicly.

## Scope

This policy covers the exo pipeline data plane and its compiled bridge to the TBCCL C ABI. Memory-safety or crash bugs in the compiled bridge and in the Python package are in scope. Behavior of exo, MLX or TBCCL themselves should be reported to those projects.

The supported deployment assumptions are described in the security model in the documentation: TBCCL assumes trusted peers on a trusted network.
