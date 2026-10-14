#!/usr/bin/env python3
"""Check the exo fork pin of exo-tbccl. Stable releases pin an immutable tag of the maintained exo fork, never a moving branch.

    check_framework_pin.py                  offline: the pin is well formed (repository, tag, full SHAs)
    check_framework_pin.py --online         additionally: the fork tag exists and resolves to the recorded SHA (release pin integrity);
                                            the tagged commit still descends from the recorded upstream exo base (GitHub compare API)

Two different questions, kept apart: RELEASE PIN INTEGRITY (the immutable tag and SHA recorded here; a mismatch is always an INTEGRITY FAILURE) and CURRENT-ECOSYSTEM
CONSISTENCY (this repository against tbccl's current ecosystem file; an INTEGRITY FAILURE only for the published stable version, a WARNING for an unreleased candidate).
A tag that resolves to another commit is an INTEGRITY FAILURE; the script never updates the recorded SHA.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import maint_common as mc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def check_offline(manifest, rep):
    exo = manifest.get("frameworks", {}).get("exo", {})
    for key in ("fork", "fork_tag", "fork_revision", "upstream_base"):
        if not exo.get(key):
            rep.add(mc.INTEGRITY, f"frameworks.exo.{key} is missing")
    if rep.worst() != mc.OK:
        return False
    if not mc.REPO_NAME.match(exo["fork"]):
        rep.add(mc.INTEGRITY, f"fork repository {exo['fork']!r} is not owner/name")
    if re.search(r"(^|/)(main|master|HEAD)$", exo["fork_tag"]):
        rep.add(mc.INTEGRITY, f"fork_tag {exo['fork_tag']!r} is a moving ref")
    for key in ("fork_revision", "upstream_base"):
        if not mc.FULL_SHA.match(exo[key]):
            rep.add(mc.INTEGRITY, f"{key} is not a full 40-character SHA")
    if rep.worst() == mc.OK:
        rep.ok(f"pin {exo['fork']}@{exo['fork_tag']} -> {exo['fork_revision'][:12]} (upstream base {exo['upstream_base'][:12]}) is well formed")
    return rep.worst() == mc.OK


def check_pin_online(manifest, rep):
    """RELEASE PIN INTEGRITY: the immutable fork tag still resolves to the recorded SHA. Independent of any release state and of the current ecosystem record."""
    exo = manifest["frameworks"]["exo"]
    try:
        got = mc.ls_remote_tag(exo["fork"], exo["fork_tag"])
    except mc.Network as e:
        rep.add(mc.WARNING, f"cannot reach GitHub for {exo['fork']}@{exo['fork_tag']} ({e})")
    else:
        if got is None:
            rep.add(mc.INTEGRITY, f"{exo['fork']}@{exo['fork_tag']} no longer exists")
        elif got != exo["fork_revision"]:
            rep.add(mc.INTEGRITY, f"{exo['fork']}@{exo['fork_tag']} resolves to {got}, recorded {exo['fork_revision']}")
        else:
            rep.ok(f"{exo['fork']}@{exo['fork_tag']} resolves to {got}")
    try:
        cmp = mc.github_api(f"repos/{exo['fork']}/compare/{exo['upstream_base']}...{exo['fork_revision']}")
    except mc.Network as e:
        rep.add(mc.WARNING, f"upstream-base ancestry not checked ({e})")
    else:
        if cmp is None or cmp.get("merge_base_commit", {}).get("sha") != exo["upstream_base"]:
            rep.add(mc.INTEGRITY, "the pinned commit does not descend from the recorded upstream exo base")
        else:
            rep.ok(f"the pinned commit is {cmp.get('ahead_by')} commits ahead of upstream base {exo['upstream_base'][:12]}")



def check_current_ecosystem(manifest, rep, ecosystem_file=None, ecosystem_ref="main"):
    """CURRENT-ECOSYSTEM CONSISTENCY: compare with release/ecosystem.json of tbccl (main, or an explicit candidate file or ref). Never a verdict on the pin itself."""
    exo = manifest["frameworks"]["exo"]
    pkg = manifest["package"]
    try:
        eco = mc.load_ecosystem(ecosystem_file, ecosystem_ref)
    except mc.Network as e:
        rep.add(mc.WARNING, f"the current ecosystem record could not be read ({e}); only the release pin was checked")
        return
    mine = {"exo_repo": exo["fork"], "exo_tag": exo["fork_tag"], "exo_sha": exo["fork_revision"], "exo_upstream_base": exo["upstream_base"]}
    mc.compare_current_ecosystem(rep, eco, "exo_tbccl", mine, pkg.get("development_version"), pkg.get("released") is True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--ecosystem-file", help="compare with this candidate ecosystem.json instead of tbccl main")
    ap.add_argument("--ecosystem-ref", default="main", help="compare with release/ecosystem.json at this tbccl git ref (default main)")
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()
    manifest = mc.load_json(Path(args.root) / "compatibility.json")
    rep = mc.Report("exo fork pin")
    if check_offline(manifest, rep) and args.online:
        check_pin_online(manifest, rep)
        check_current_ecosystem(manifest, rep, args.ecosystem_file, args.ecosystem_ref)
    return rep.finish()


if __name__ == "__main__":
    mc.main_guard(main)
