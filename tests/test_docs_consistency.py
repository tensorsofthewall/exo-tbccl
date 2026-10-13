"""Documentation contract checks: the documented environment variables, versions and requirements match the code and the manifest."""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*parts):
    with open(os.path.join(ROOT, *parts)) as f:
        return f.read()


def _source_env_vars():
    names = set()
    for base in ("exo_tbccl", "src"):
        for dirpath, _, files in os.walk(os.path.join(ROOT, base)):
            for name in files:
                if name.endswith((".py", ".c", ".h")):
                    with open(os.path.join(dirpath, name), errors="ignore") as f:
                        names.update(re.findall(r'"(EXO_TBCCL_[A-Z0-9_]+)"', f.read()))
    return names


def test_environment_variables_read_by_the_code_are_documented():
    page = _read("docs", "reference", "configuration.md")
    documented = set(re.findall(r"`(EXO_TBCCL_[A-Z0-9_]+)`", page))
    missing = {v for v in _source_env_vars() if v not in documented}
    assert not missing, f"environment variables read by the code but not documented: {sorted(missing)}"


def test_documented_environment_variables_exist():
    page = _read("docs", "reference", "configuration.md")
    documented = set(re.findall(r"`(EXO_TBCCL_[A-Z0-9_]+)`", page))
    alias = {"EXO_TBCCL_STEP_ACTIVITY"}  # documented as an alias in the same row as EXO_TBCCL_ACTIVITY_MODE
    stale = {v for v in documented if v not in _source_env_vars() and v not in alias}
    assert not stale, f"documented environment variables that the code does not read: {sorted(stale)}"


def test_versions_in_the_docs_match_the_manifest_and_no_stale_requirements_remain():
    manifest = json.loads(_read("compatibility.json"))
    version = manifest["package"]["development_version"]
    compat = _read("docs", "reference", "compatibility.md")
    install = _read("docs", "getting-started", "install.md")
    assert version in compat and version in install
    minimum = manifest["requires_tbccl"]["minimum_package"]
    for name in ("CONTRIBUTING.md", "AGENTS.md"):
        text = _read(name)
        assert not re.search(r"TBCCL[^\n]{0,40}(>=|>= ?)\s?0\.5\b", text), f"{name} still names the old TBCCL 0.5 requirement"
        assert minimum in text, f"{name} should state the minimum TBCCL package {minimum}"
    assert "there is no public 0.5" not in install  # historical wording: the requirement is simply TBCCL 0.6.0 or newer


def test_docs_do_not_claim_registry_publication_or_editable_only():
    install = _read("docs", "getting-started", "install.md")
    assert "not published on PyPI or TestPyPI" in install
    assert "test.pypi.org" not in install and "pip install --pre" not in install
    assert "does **not** need an editable install" in install


def test_the_discovery_workaround_is_documented():
    two_host = _read("docs", "guides", "two-host.md")
    assert "ssh -tt" in two_host and "52413" in two_host
    assert "not architectural requirements" in two_host


def test_the_exo_base_is_recorded_consistently_and_stale_bases_are_gone():
    manifest = json.loads(_read("compatibility.json"))
    base = manifest["frameworks"]["exo"]["upstream_base"]
    compat = _read("docs", "reference", "compatibility.md")
    assert base in compat and base[:8] in _read("docs", "getting-started", "install.md") and base[:8] in _read("README.md")
    for name in ("README.md", "docs/reference/compatibility.md", "docs/getting-started/install.md", "docs/project/limitations.md"):
        text = _read(*name.split("/"))
        assert "14 commits" not in text, f"{name} still counts the old integration series"
    assert "21a54c5e" in compat and "re-derived" in compat   # named only as the history of the series
    assert not [v for v in manifest["validated_configurations"] if "tbccl_wire_protocol_at_validation" in v]


def test_the_exo_fork_tag_is_the_published_anchor():
    manifest = json.loads(_read("compatibility.json"))
    exo = manifest["frameworks"]["exo"]
    assert exo["fork"] == "tensorsofthewall/exo-fork" and exo["fork_tag"] == "tbccl-integration-v1"
    for name in ("README.md", "docs/reference/compatibility.md", "docs/getting-started/install.md"):
        text = _read(*name.split("/"))
        assert exo["fork_tag"] in text, f"{name} does not name the exo fork tag"
        assert "/mnt/BigChonk" not in text and "has not been published" not in text and "not yet published" not in text
    assert exo["fork_revision"] in _read("docs", "reference", "compatibility.md")
