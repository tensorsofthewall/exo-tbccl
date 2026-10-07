"""Inspect a built exo-tbccl wheel WITHOUT importing it.

    python tools/inspect_wheel.py dist/exo_tbccl-*.whl [--json OUT] [--allow-local-platform]

Fails (exit 1) on: a missing module or metadata; a missing licence file; a version that disagrees with exo_tbccl/__init__.py; a wheel tag that is not valid for publication (Linux: manylinux_*, never a bare linux_*; macOS: macosx_*_arm64); development files (tests, tools,
docs, examples, sources) or build-machine paths in any member, an extension that depends on libtbccl or libcudart at run time or has an absolute runtime search path.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import zipfile

MODULE = "exo_tbccl"
REQUIRED = ["exo_tbccl/__init__.py", "exo_tbccl/group.py", "exo_tbccl/bridge.py"]
NATIVE = True
SYSTEM_DIRS = ("/usr/lib", "/usr/lib64", "/lib", "/lib64", "/usr/local/lib")
DEV_PREFIXES = ("tests/", "tools/", "docs/", "examples/", "src/", "benchmarks/", ".github/")


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def parse_tags(filename):
    parts = os.path.basename(filename)[:-4].split("-")
    return parts[-3], parts[-2], parts[-1].split(".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wheel")
    ap.add_argument("--json", default=None)
    ap.add_argument("--allow-local-platform", action="store_true", help="accept a bare linux_* platform tag (a locally built wheel; never for publication)")
    a = ap.parse_args()
    problems, report = [], {"wheel": os.path.basename(a.wheel)}
    z = zipfile.ZipFile(a.wheel)
    names = z.namelist()
    report["members"] = names
    for m in REQUIRED:
        if m not in names:
            problems.append(f"missing module {m}")
    meta = next((n for n in names if n.endswith(".dist-info/METADATA")), None)
    text = z.read(meta).decode() if meta else ""
    if not meta:
        problems.append("missing METADATA")
    report["requires_dist"] = re.findall(r"^Requires-Dist: (.+)$", text, re.M)
    report["requires_python"] = (re.search(r"^Requires-Python: (.+)$", text, re.M) or [None, None])[1]
    py_tag, abi_tag, plats = parse_tags(a.wheel)
    report["tag"] = [py_tag, abi_tag, plats]
    want_py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if py_tag != want_py or abi_tag != want_py:
        problems.append(f"wheel python/abi tag {py_tag}/{abi_tag} != running {want_py}")
    if sys.platform == "darwin":
        if not plats or not all(re.fullmatch(r"macosx_\d+_\d+_arm64", p) for p in plats):
            problems.append(f"macOS wheel platform tag must be macosx_<major>_<minor>_arm64, got {plats}")
    elif not a.allow_local_platform:
        if not plats or not all(p.startswith("manylinux_") for p in plats):
            problems.append(f"Linux wheel platform tag must be manylinux_*, got {plats} (a bare linux_x86_64 wheel cannot be published)")
    if not any(re.fullmatch(r".*\.dist-info/licenses/LICENSE", n) or re.fullmatch(r".*\.dist-info/LICENSE", n) for n in names):
        problems.append("the licence file is not in the wheel")
    ver_in_name = os.path.basename(a.wheel).split("-")[1]
    ver_src = re.search(r'__version__ = "([^"]+)"', z.read("exo_tbccl/__init__.py").decode()) if "exo_tbccl/__init__.py" in names else None
    if not ver_src or ver_src.group(1) != ver_in_name:
        problems.append(f"wheel version {ver_in_name} != exo_tbccl/__init__.py {ver_src.group(1) if ver_src else None}")
    if False:
        meta_ver = re.search(r"^Version: (.+)$", text, re.M)
        if not meta_ver or meta_ver.group(1) != ver_in_name:
            problems.append(f"metadata version {meta_ver.group(1) if meta_ver else None} != file name version {ver_in_name}")
    for n in names:
        if n.startswith(DEV_PREFIXES) or n.endswith((".pyc", ".o", ".a", ".c", ".cpp", ".h")):
            problems.append(f"development file in the wheel: {n}")
        if not n.endswith(".so"):
            private = re.findall(rb"/(?:home|Users|mnt|tmp)/[\w.\-]+", z.read(n))
            if private:
                problems.append(f"{n} contains private or build paths: {sorted(set(x.decode() for x in private))}")
    exts = [n for n in names if re.match(MODULE + r"/_native\..*\.(so|pyd)$", n)]
    if len(exts) != 1:
        problems.append(f"expected exactly one compiled extension {MODULE}/_native.*, found {exts}")
    else:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            z.extract(exts[0], d)
            so = os.path.join(d, exts[0])
            if sys.platform == "darwin":
                out = run(["otool", "-L", so]) or ""
                needed = [l.split()[0] for l in out.splitlines()[1:] if l.strip()]
                lc = run(["otool", "-l", so]) or ""
                rpaths = re.findall(r"cmd LC_RPATH\n\s+cmdsize \d+\n\s+path (\S+)", lc)
                for n in needed:
                    if not n.startswith(("/usr/lib/", "/System/Library/", "@rpath/", "@loader_path/")):
                        problems.append(f"dependency outside the system directories: {n}")
            else:
                out = run(["readelf", "-d", so]) or ""
                needed = re.findall(r"NEEDED\)\s+Shared library: \[([^\]]+)\]", out)
                rpaths = [p for m in re.finditer(r"\((?:RUNPATH|RPATH)\)\s+Library (?:runpath|rpath): \[([^\]]*)\]", out) for p in m.group(1).split(":") if p]
                for n in needed:
                    if "tbccl" in n or "cudart" in n:
                        problems.append(f"the extension depends on {n} at run time (TBCCL and the CUDA runtime must be linked statically)")
            report["needed"], report["search_paths"] = needed, rpaths
            for p in rpaths:
                if not (p.startswith(("$ORIGIN", "@loader_path", "@executable_path")) or p.startswith(SYSTEM_DIRS)):
                    problems.append(f"absolute runtime search path: {p}")
            blob = open(so, "rb").read()
            leaks = sorted({m.decode(errors="replace") for m in re.findall(rb"/(?:home|Users|mnt)/[\w.\-]+", blob)})
            report["embedded_home_paths"] = leaks
            if leaks:
                problems.append(f"the extension embeds build-machine paths: {leaks}")
    report["problems"] = problems
    print(json.dumps(report, indent=1))
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
