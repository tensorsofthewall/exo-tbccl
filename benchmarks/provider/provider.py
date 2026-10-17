#!/usr/bin/env python3
"""Benchmark provider entry point for exo-tbccl (TBCCL benchmark provider protocol 1).

    python provider.py describe     one JSON object on stdout: identity, protocol, workloads, requirements and what this environment provides
    python provider.py validate F   F is a JSON file with a workload configuration; prints {"ok": bool, "errors": [...]}

Both operations are lightweight and self-contained: they import no framework, load no model, touch no device, start no server and open no network connection. They look
framework packages up by metadata only. Logs go to stderr; stdout carries the JSON object and nothing else. The workload scripts in this directory are started by the
TBCCL benchmark runner (``python -m benchmarks.suite``), which owns process supervision; run one directly for a standalone measurement (see the README in this directory).
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys

PROTOCOL_VERSION = 1
RESULT_SCHEMAS = ["tbccl-bench-result/1"]
PROVIDER_ID = 'exo-tbccl'
DISTRIBUTION = 'exo-tbccl'
WORKLOADS = ['correctness', 'latency', 'throughput', 'transport_ab', 'stage_profile', 'large_model', 'lifecycle']
PLATFORMS = ['linux', 'darwin']
MEMORY_KINDS = ['host', 'cuda', 'metal']
DEVICE_PAIRINGS = ['cuda-metal', 'metal-metal']
REQUIRED_PACKAGES = [('exo', 'exo'), ('exo_tbccl', 'exo-tbccl'), ('mlx', 'mlx')]
OPTIONAL_PACKAGES = [('psutil', 'psutil')]
REQUIRED_VERSIONS = {'python': '>=3.13',
 'exo-tbccl': '>=0.3,<0.4',
 'exo': 'tensorsofthewall/exo-fork tag tbccl-integration-v2 (revision d1497b28b89618172d273e70e9ac7992811635f5)',
 'mlx': "0.32.0 (Linux CUDA); 0.32.4.dev20261013+0e3ff3643 (macOS, exo's pin)",
 'libtbccl': {'c_abi': [1], 'minimum_package': '0.6.0'}}
RUNTIME_CAPABILITIES = ['exo cluster with the MlxRing and MlxTbccl placements', 'TBCCL communicator (MlxTbccl)', 'MLX CUDA on Linux, MLX Metal on macOS (DLPack compatibility check on Metal)']
MODEL_CONFIGURATION = {'required': True, 'note': 'an MLX model already present on every host; the provider never downloads models. Reference configuration: Qwen3-0.6B-8bit'}
CONFIGURATION_SCHEMA = {'type': 'object',
 'required': ['workload'],
 'properties': {'workload': {'type': 'string'},
                'backend': {'type': 'string', 'enum': ['MlxRing', 'MlxTbccl']},
                'stage_layers': {'type': 'string'},
                'stage_hosts': {'type': 'string'},
                'concurrency': {'type': 'integer', 'minimum': 1},
                'max_tokens': {'type': 'integer', 'minimum': 1},
                'tier': {'type': 'string', 'enum': ['baseline', 'fastpath']},
                'activity_policy': {'type': 'string'}}}
ENTRY_POINTS = {'client': 'client.py', 'sampler': 'sampler.py', 'trace_hook': 'trace_hook/sitecustomize.py'}


def _version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def native_modules():
    """SHA-256 and size of each compiled extension the installed distribution ships (the TBCCL library bundled in an adapter wheel is linked into it), so a result names the exact binary that ran and
    not the TBCCL checkout it was compared with. None when the distribution is not installed; the file contents are only read, never loaded."""
    try:
        dist = importlib.metadata.distribution(DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return None
    found = {}
    for entry in dist.files or []:
        if str(entry).endswith((".so", ".dylib", ".pyd")):
            path = dist.locate_file(entry)
            if path.is_file():
                found[str(entry)] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    return found


def availability():
    """available / environment_missing / unsupported, with reasons. Versions come from package metadata, never from importing the framework."""
    detected = {"python": platform.python_version(), "platform": f"{platform.system().lower()}-{platform.machine().lower()}"}
    reasons = []
    if platform.system().lower() not in PLATFORMS:
        return {"status": "unsupported", "reasons": [f"{platform.system()} is not a supported platform ({', '.join(PLATFORMS)})"], "detected": detected}
    detected["native_modules"] = native_modules()
    for module, distribution in REQUIRED_PACKAGES:
        version = _version(distribution)
        detected[distribution] = version
        if not _has_module(module):
            reasons.append(f"Python package '{distribution}' (module '{module}') is not installed in this interpreter")
    for module, distribution in OPTIONAL_PACKAGES:
        detected[distribution] = _version(distribution) if _has_module(module) else None
    if _has_module("exo_tbccl") and _version("exo-tbccl") and not (_version("exo-tbccl") or "").startswith("0.3."):
        reasons.append(f"exo-tbccl {_version('exo-tbccl')} is outside the supported series 0.3")
    return {"status": "environment_missing" if reasons else "available", "reasons": reasons, "detected": detected}


def entry_point_hashes():
    """SHA-256 of each workload script this provider ships, so a result can name the exact scripts that produced it."""
    from pathlib import Path
    here = Path(__file__).resolve().parent
    return {name: hashlib.sha256((here / relative).read_bytes()).hexdigest() for name, relative in ENTRY_POINTS.items() if (here / relative).is_file()}


def describe():
    return {
        "provider_id": PROVIDER_ID, "provider_protocol_version": PROTOCOL_VERSION, "provider_version": _version(DISTRIBUTION) or "source-checkout",
        "supported_workloads": WORKLOADS, "supported_platforms": PLATFORMS, "supported_memory_kinds": MEMORY_KINDS, "supported_device_pairings": DEVICE_PAIRINGS,
        "required_framework_versions": REQUIRED_VERSIONS, "required_runtime_capabilities": RUNTIME_CAPABILITIES, "optional_dependencies": [d for _, d in OPTIONAL_PACKAGES],
        "required_model_configuration": MODEL_CONFIGURATION, "benchmark_configuration_schema": CONFIGURATION_SCHEMA,
        "supported_result_schema_versions": RESULT_SCHEMAS, "entry_points": ENTRY_POINTS, "entry_point_sha256": entry_point_hashes(), "availability": availability(),
    }


def validate(config):
    errors = []
    if not isinstance(config, dict):
        return ["the configuration must be a JSON object"]
    workload = config.get("workload")
    if workload not in WORKLOADS:
        errors.append(f"workload must be one of {WORKLOADS}")
    for key, spec in CONFIGURATION_SCHEMA.get("properties", {}).items():
        if key not in config:
            continue
        value = config[key]
        kind = spec.get("type")
        if kind == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            errors.append(f"{key} must be an integer")
        elif kind == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            errors.append(f"{key} must be a number")
        elif kind == "string" and not isinstance(value, str):
            errors.append(f"{key} must be a string")
        elif "enum" in spec and value not in spec["enum"]:
            errors.append(f"{key} must be one of {spec['enum']}")
        elif "minimum" in spec and isinstance(value, (int, float)) and value < spec["minimum"]:
            errors.append(f"{key} must be at least {spec['minimum']}")
    for key in CONFIGURATION_SCHEMA.get("required", []):
        if key not in config:
            errors.append(f"missing '{key}'")
    if config.get("activity_policy") and config.get("tier") != "fastpath":
        errors.append("an activity policy is experimental: set tier to 'fastpath' to enable it explicitly; the default benchmark never enables one")
    if config.get("backend") == "MlxRing" and config.get("activity_policy"):
        errors.append("activity policies apply to MlxTbccl only")
    return errors


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["describe"] and len(argv) == 1:
        print(json.dumps(describe(), indent=2))
        return 0
    if argv[:1] == ["validate"] and len(argv) == 2:
        try:
            with open(argv[1]) as handle:
                config = json.load(handle)
        except (OSError, ValueError) as error:
            print(json.dumps({"ok": False, "errors": [f"cannot read the configuration: {error}"]}))
            return 1
        errors = validate(config)
        print(json.dumps({"ok": not errors, "errors": errors}))
        return 0 if not errors else 1
    print(__doc__.strip().splitlines()[2] + "\n" + __doc__.strip().splitlines()[3], file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
