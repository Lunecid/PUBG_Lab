"""Structural contracts: no database, model fitting, services or protected source access."""

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest

from pubg_lab.__main__ import command_catalog
from pubg_lab.common.compat import redirect
from pubg_lab.common.paths import REPO_ROOT, repository_root
from pubg_lab.repository.check_structure import check


class CodebaseTests(unittest.TestCase):
    def test_implementation_has_ownership_and_import_boundaries(self):
        result = check()
        self.assertTrue(result["ok"], result["errors"])

    def test_manifested_moves_have_single_implementation(self):
        plan = json.loads(
            (REPO_ROOT / "docs/repository/refactor_2026-10-26/move_plan.json").read_text()
        )
        destinations = []
        for row in plan["files"]:
            source = REPO_ROOT / row["new"]
            shim = REPO_ROOT / row["old"]
            self.assertTrue(source.is_file(), row["new"])
            text = shim.read_text()
            tree = ast.parse(text)
            self.assertFalse(
                any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in ast.walk(tree))
            )
            redirects = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_redirect"
            ]
            self.assertEqual(len(redirects), 1)
            module = ".".join(Path(row["new"]).with_suffix("").parts[1:])
            module = module.removesuffix(".__init__")
            self.assertEqual(redirects[0].args[1].value, module)
            destinations.append(row["new"])
        self.assertEqual(len(destinations), len(set(destinations)))

    def test_private_modules_are_outside_canonical_package(self):
        self.assertFalse(list((REPO_ROOT / "src").rglob("splits.py")))
        self.assertFalse(list((REPO_ROOT / "src").rglob("split_v3.py")))
        self.assertFalse(list((REPO_ROOT / "src").rglob(".env*")))

    def test_command_discovery_does_not_import_applications(self):
        before = set(sys.modules)
        catalog = command_catalog()
        self.assertTrue(catalog)
        self.assertEqual(before, set(sys.modules))
        for record in catalog.values():
            path = REPO_ROOT / "src" / (record["module"].replace(".", "/") + ".py")
            self.assertTrue(path.is_file(), record)

    def test_source_root_is_stable_from_other_working_directory(self):
        output = subprocess.check_output(
            [sys.executable, str(REPO_ROOT / "tools/repository_release.py"), "--help"],
            cwd=REPO_ROOT.parent,
            text=True,
        )
        self.assertIn("snapshot", output)
        self.assertEqual(repository_root(), REPO_ROOT)

    def test_spec_loader_and_normal_import_share_exported_objects(self):
        # Use a stdlib-only migrated application; do not import models or data access.
        name = "compat_quality_fixture"
        spec = importlib.util.spec_from_file_location(
            name, REPO_ROOT / "tools/corpus_quality_index.py"
        )
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        from pubg_lab.data.quality import corpus_quality_index

        self.assertIs(legacy.build, corpus_quality_index.build)
        self.assertIs(sys.modules[name], corpus_quality_index)

    def test_compatibility_forwarding_does_not_copy_implementation(self):
        target_name = "_pubg_target_fixture"
        old_name = "_pubg_old_fixture"
        target = types.ModuleType(target_name)
        target.value = object()
        sys.modules[target_name] = target
        try:
            namespace = {"__name__": old_name}
            redirect(namespace, target_name)
            self.assertIs(sys.modules[old_name], target)
            self.assertIs(namespace["__getattr__"]("value"), target.value)
        finally:
            sys.modules.pop(old_name, None)
            sys.modules.pop(target_name, None)


if __name__ == "__main__":
    unittest.main()
