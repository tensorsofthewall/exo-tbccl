"""The exo-tbccl benchmark provider (offline): the description, validation, absence reporting and the independence of this provider from the other adapter packages.

No framework needs to be installed: the provider looks packages up by metadata and these tests substitute the lookups."""
import ast
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

PROVIDER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROVIDER_DIR))
import provider  # noqa: E402

OTHER_ADAPTERS = ['vllm_tbccl', 'torch_tbccl', 'vllm']
FRAMEWORKS = ("torch", "vllm", "exo", "mlx", "torch_tbccl", "vllm_tbccl", "exo_tbccl")
REQUIRED = ("provider_id", "provider_protocol_version", "provider_version", "supported_workloads", "supported_platforms", "supported_memory_kinds", "supported_device_pairings",
            "required_framework_versions", "required_runtime_capabilities", "optional_dependencies", "required_model_configuration", "benchmark_configuration_schema",
            "supported_result_schema_versions", "entry_points", "availability")


def run(*args):
    return subprocess.run([sys.executable, str(PROVIDER_DIR / "provider.py"), *args], capture_output=True, text=True, cwd=str(PROVIDER_DIR), stdin=subprocess.DEVNULL, timeout=60)


class Describe(unittest.TestCase):
    def test_stdout_is_exactly_one_json_object_with_every_required_field(self):
        out = run("describe")
        self.assertEqual(out.returncode, 0, out.stderr)
        d = json.loads(out.stdout)
        for key in REQUIRED:
            self.assertIn(key, d)
        self.assertEqual((d["provider_id"], d["provider_protocol_version"]), ("exo-tbccl", 1))
        self.assertEqual(d["supported_result_schema_versions"], ["tbccl-bench-result/1"])
        self.assertIn(d["availability"]["status"], ("available", "environment_missing", "unsupported"))

    def test_describe_imports_no_framework(self):
        code = ("import sys; sys.path.insert(0, %r); import provider; provider.describe(); print(sorted(m for m in %r if m in sys.modules))" % (str(PROVIDER_DIR), FRAMEWORKS))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(out.stdout.strip(), "[]", out.stderr)

    def test_every_entry_point_exists_and_stays_inside_the_provider_directory(self):
        for name, relative in provider.ENTRY_POINTS.items():
            path = (PROVIDER_DIR / relative).resolve()
            self.assertTrue(path.is_file(), name)
            self.assertTrue(str(path).startswith(str(PROVIDER_DIR.resolve())), name)

    def test_a_missing_framework_is_reported_with_the_package_name(self):
        with mock.patch.object(provider, "_has_module", return_value=False):
            a = provider.availability()
        self.assertEqual(a["status"], "environment_missing")
        for _, distribution in provider.REQUIRED_PACKAGES:
            self.assertTrue(any(distribution in reason for reason in a["reasons"]), distribution)

    def test_an_installed_environment_is_available_and_its_versions_are_read_from_metadata(self):
        versions = {dist: "9.9.9" for _, dist in provider.REQUIRED_PACKAGES + provider.OPTIONAL_PACKAGES}
        with mock.patch.object(provider, "_has_module", return_value=True), mock.patch.object(provider, "_version", side_effect=lambda d: versions.get(d)):
            a = provider.availability()
        self.assertTrue(set(a["detected"]) >= {dist for _, dist in provider.REQUIRED_PACKAGES})
        self.assertEqual(a["detected"][provider.REQUIRED_PACKAGES[0][1]], "9.9.9")

    def test_an_unsupported_platform_is_reported_not_attempted(self):
        with mock.patch.object(provider.platform, "system", return_value="Windows"):
            a = provider.availability()
        self.assertEqual(a["status"], "unsupported")
        self.assertIn("Windows", a["reasons"][0])

    def test_provider_version_is_never_invented(self):
        with mock.patch.object(provider, "_version", return_value=None):
            self.assertEqual(provider.describe()["provider_version"], "source-checkout")


class Validate(unittest.TestCase):
    def test_a_minimal_valid_configuration(self):
        self.assertEqual(provider.validate({"workload": provider.WORKLOADS[0]}), [])

    def test_unknown_workload_and_bad_types_are_rejected(self):
        errors = provider.validate({"workload": "nonsense"})
        self.assertTrue(any("workload must be one of" in e for e in errors))
        self.assertTrue(provider.validate([]))
        for key, spec in provider.CONFIGURATION_SCHEMA["properties"].items():
            if spec.get("type") == "integer":
                self.assertTrue(provider.validate({"workload": provider.WORKLOADS[0], key: "x"}), key)
                break

    def test_cli_exit_codes_follow_the_result(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            good, bad = Path(tmp, "g.json"), Path(tmp, "b.json")
            good.write_text(json.dumps({"workload": provider.WORKLOADS[0]}))
            bad.write_text(json.dumps({"workload": "nope"}))
            ok, no = run("validate", str(good)), run("validate", str(bad))
        self.assertEqual((ok.returncode, json.loads(ok.stdout)["ok"]), (0, True))
        self.assertEqual((no.returncode, json.loads(no.stdout)["ok"]), (1, False))
        self.assertEqual(run("validate", "/nonexistent.json").returncode, 1)
        self.assertEqual(run("bogus").returncode, 2)

    def test_an_activity_policy_is_never_implied(self):
        self.assertTrue(provider.validate({"workload": "transport_ab", "backend": "MlxTbccl", "activity_policy": "step"}))  # needs the explicit experimental tier
        self.assertEqual(provider.validate({"workload": "transport_ab", "backend": "MlxTbccl", "activity_policy": "step", "tier": "fastpath"}), [])
        self.assertTrue(provider.validate({"workload": "transport_ab", "backend": "MlxRing", "activity_policy": "step", "tier": "fastpath"}))
        self.assertEqual(provider.validate({"workload": "transport_ab", "backend": "MlxTbccl"}), [])
        self.assertTrue(provider.validate({"workload": "transport_ab", "tier": "turbo"}))


class NativeModules(unittest.TestCase):
    """The compiled extension an installed wheel ships is identified by its SHA-256 (its contents are read, never loaded)."""

    def fake_distribution(self, root, names):
        class Dist:
            files = [Path(n) for n in names]

            def locate_file(self, entry):
                return Path(root) / entry
        return Dist()

    def test_a_distribution_that_is_not_installed_gives_none_not_a_guess(self):
        with mock.patch.object(provider.importlib.metadata, "distribution", side_effect=provider.importlib.metadata.PackageNotFoundError):
            self.assertIsNone(provider.native_modules())
            self.assertIsNone(provider.availability()["detected"]["native_modules"])

    def test_each_shipped_extension_is_hashed_and_nothing_else(self):
        import hashlib
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "pkg").mkdir()
            (Path(tmp) / "pkg" / "_ext.cpython-313-x86_64-linux-gnu.so").write_bytes(b"native bytes")
            (Path(tmp) / "pkg" / "notes.txt").write_text("not a binary")
            dist = self.fake_distribution(tmp, ["pkg/_ext.cpython-313-x86_64-linux-gnu.so", "pkg/notes.txt", "pkg/missing.so"])
            with mock.patch.object(provider.importlib.metadata, "distribution", return_value=dist):
                found = provider.native_modules()
        self.assertEqual(found, {"pkg/_ext.cpython-313-x86_64-linux-gnu.so": {"sha256": hashlib.sha256(b"native bytes").hexdigest(), "bytes": 12}})

    def test_the_native_module_never_triggers_an_import(self):
        code = ("import sys; sys.path.insert(0, %r); import provider; provider.native_modules(); print(sorted(m for m in %r if m in sys.modules))" % (str(PROVIDER_DIR), FRAMEWORKS))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(out.stdout.strip(), "[]", out.stderr)


class Independence(unittest.TestCase):
    def imports(self):
        found = {}
        for path in PROVIDER_DIR.rglob("*.py"):
            if "tests" in path.relative_to(PROVIDER_DIR).parts:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                for name in names:
                    found.setdefault(name.split(".")[0], set()).add(path.name)
        return found

    def test_no_other_adapter_package_is_imported(self):
        found = self.imports()
        for other in OTHER_ADAPTERS:
            self.assertNotIn(other, found, f"{other} imported by {found.get(other)}")

    def test_the_provider_directory_does_not_reach_into_a_sibling_directory(self):
        for path in PROVIDER_DIR.rglob("*.py"):
            if "tests" in path.relative_to(PROVIDER_DIR).parts:
                continue
            self.assertNotIn("parent.parent", path.read_text().replace(" ", ""), path.name)


if __name__ == "__main__":
    unittest.main()
