#!/usr/bin/env python3
"""Check the exo fork pin of exo-tbccl. Stable releases pin an immutable tag of the maintained exo fork, never a moving branch.

    check_framework_pin.py                  offline: the pin is well formed (repository, tag, full SHAs)
    check_framework_pin.py --online         additionally: the fork tag exists and resolves to the recorded SHA; the pin agrees with the TBCCL ecosystem file;
                                            the tagged commit still descends from the recorded upstream exo base (GitHub compare API)

A tag that resolves to another commit is an INTEGRITY FAILURE; the script never updates the recorded SHA.
"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import maint_common as mc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ECOSYSTEM_URL = "https://raw.githubusercontent.com/tensorsofthewall/tbccl/main/release/ecosystem.json"


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


def check_online(manifest, rep):
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
    try:
        with urllib.request.urlopen(urllib.request.Request(ECOSYSTEM_URL, headers={"User-Agent": "tbccl-maintenance"}), timeout=30) as r:
            eco = json.load(r)["exo_tbccl"]
    except (OSError, ValueError, KeyError) as e:
        rep.add(mc.WARNING, f"the TBCCL ecosystem file could not be read ({type(e).__name__})")
        return
    mine = {"exo_repo": exo["fork"], "exo_tag": exo["fork_tag"], "exo_sha": exo["fork_revision"], "exo_upstream_base": exo["upstream_base"]}
    for key, value in mine.items():
        if eco.get(key) != value:
            rep.add(mc.INTEGRITY, f"{key}: ecosystem file says {eco.get(key)!r}, this repository records {value!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()
    manifest = mc.load_json(Path(args.root) / "compatibility.json")
    rep = mc.Report("exo fork pin")
    if check_offline(manifest, rep) and args.online:
        check_online(manifest, rep)
    return rep.finish()


if __name__ == "__main__":
    mc.main_guard(main)
