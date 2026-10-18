# Documentation versions

The hosted documentation is built by Read the Docs from the configuration in `.readthedocs.yaml`, with the pinned requirements in `docs/requirements.txt`, and warnings fail the build.

| Version | Built from | Meaning |
|---|---|---|
| `latest` | the `main` branch | Development documentation. It is not a release, carries a banner that says so, and may describe behavior that no release contains yet. |
| `stable` | the newest final release tag | The documentation of the latest final release, frozen to its tag. Nothing unreleased is ever published as `stable`. |
| a version such as `v0.3.0` or `v0.3.0rc1` | a release or release-candidate tag | The documentation of that release, frozen to the tag. |

## Release-specific and current information

Documentation built from a tag is historical by design: it records what that release supports, what was validated for it, its dependencies and its limitations, and it never changes. Anything it says about the other projects was true when it was built and is not updated when they release later. The current state of the whole family (versions, tags, framework pins) is on the [ecosystem page](../project/ecosystem.md) of the `latest` TBCCL documentation, built from `release/ecosystem.json`; use it, not a versioned page, to answer "what is current".


## Documentation version policy

Five kinds of information are kept apart, and each has one place where it is allowed to change:

| Kind | Where it lives | May it change after publication? |
|---|---|---|
| Current development documentation | `latest`, built from `main` | Yes. It may describe unreleased behavior and says so in its banner. |
| Stable documentation | `stable`, built from the newest final release tag | No. It is the tag's documentation, frozen. |
| Historical documentation | numbered versions, built from earlier tags | No. |
| The live ecosystem compatibility manifest | `release/ecosystem.json` and the [ecosystem page](https://tbccl.tensorsofthewall.com/en/latest/project/ecosystem.html) on the `latest` TBCCL documentation | Yes, on `main` only. This is the one place that says what is current for the whole family. A manifest inside a tag is a historical snapshot. |
| Historical benchmark results | the benchmark and validation pages | Never rewritten. A result keeps the exact versions that were measured, even when newer packages exist. |

Rules that follow from this:

* Release tags and release artifacts are immutable. A tag is never moved, deleted or re-pointed, and published archives and wheels are never replaced.
* A release page describes the release it was built from: its README, homepage and installation commands name that release's version, artifact names and URLs. This is checked before the tag exists (see below).
* A mistake found in a tagged page is corrected on `main` and, if users could be misled, announced as an erratum in the "Errata" section below and in the release notes. The tag is not rewritten.
* Benchmark and validation pages state the versions measured. Do not update them to the current version to make them look consistent; re-measure instead, and add the new measurement beside the old one.
* A version that is deliberately not the current one (a historical section of the changelog, a comparison between releases, a warning about a superseded or defective release, a dependency that an immutable tag pins) is written as such, in a sentence that says why.

Release-specific statements (the README status line, the documentation homepage, the installation commands, the archive or wheel names and the expected version output) are checked by `tools/check_release_docs.py`. It derives the version from the package source and `compatibility.json`, not from the ecosystem manifest, which is only updated after a release is published. Release preflight runs it against the intended tag before anything is tagged; the documentation workflow runs it against the current branch.


## Building and checking locally

`make docs` builds the documentation, `make docs-linkcheck` checks the local links, and the compatibility manifest check runs in the same CI job. None of them needs the network after the requirements are installed.

Cross-project references (Intersphinx) are off by default so that a build never depends on another site. Set `DOCS_INTERSPHINX=1` to enable them against the other projects' published development inventories (`latest`); set `DOCS_INVENTORY_<PROJECT>` to a local `objects.inv` to use a local copy.
