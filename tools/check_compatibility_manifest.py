#!/usr/bin/env python3
"""Check that compatibility.json agrees with pyproject.toml and the extension source.

The sources stay the single definition; this check fails when the manifest drifts from them. The minimum TBCCL package is read from CMakeLists.txt and the C ABI version from the documented contract in exo_tbccl/__init__.py;
nothing is imported or built.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
manifest = json.loads((ROOT / "compatibility.json").read_text())
pyproject = (ROOT / "pyproject.toml").read_text()

cmake = (ROOT / "CMakeLists.txt").read_text()
minimum = re.search(r"find_package\(TBCCL\s+([0-9.]+)\s+CONFIG", cmake).group(1)
abi = re.search(r"TBCCL\s*\(C ABI (\d+)\)", (ROOT / "exo_tbccl" / "__init__.py").read_text())

checks = {
    "package.development_version": (manifest["package"]["development_version"], re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M).group(1)),
    "python.requires": (manifest["python"]["requires"], re.search(r'requires-python\s*=\s*"([^"]+)"', pyproject).group(1)),
    "requires_tbccl.minimum_package": (manifest["requires_tbccl"]["minimum_package"], minimum),
    "requires_tbccl.c_abi": (tuple(manifest["requires_tbccl"]["c_abi"]), (int(abi.group(1)),) if abi else None),
}
bad = [k for k, (declared, actual) in checks.items() if declared != actual]
for k in bad:
    print(f"MISMATCH {k}: manifest {checks[k][0]!r}, sources {checks[k][1]!r}")
if bad:
    sys.exit(1)
print("compatibility.json agrees with the sources")
