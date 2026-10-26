"""Physical layout contracts: no DB, services, fitting, or protected source access."""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from pubg_lab.__main__ import command_catalog
from pubg_lab.common.paths import REPO_ROOT, repository_root
from pubg_lab.repository.check_structure import check


class CodebaseTests(unittest.TestCase):
    def test_implementation_has_ownership_and_import_boundaries(self):
        result = check()
        self.assertTrue(result["ok"], result["errors"])

    def test_generated_root_shims_are_physically_removed(self):
        plan = json.loads((REPO_ROOT / "docs/repository/layout_2026-10-26/plan.json").read_text())
        for row in plan["delete_generated_shims"]:
            self.assertFalse((REPO_ROOT / row["path"]).exists(), row["path"])
        self.assertFalse((REPO_ROOT / "src/pubg_lab/common/compat.py").exists())

    def test_private_modules_are_outside_distributable_packages(self):
        for folder in ("src", "projects"):
            self.assertFalse(list((REPO_ROOT / folder).rglob("splits.py")))
            self.assertFalse(list((REPO_ROOT / folder).rglob("split_v3.py")))
            self.assertFalse(list((REPO_ROOT / folder).rglob(".env")))

    def test_command_discovery_does_not_import_applications(self):
        before = set(sys.modules)
        catalog = command_catalog()
        self.assertTrue(catalog)
        self.assertEqual(before, set(sys.modules))
        for record in catalog.values():
            prefix = "src/" if record["module"].startswith("pubg_lab.") else ""
            path = REPO_ROOT / (prefix + record["module"].replace(".", "/") + ".py")
            self.assertTrue(path.is_file(), record)
            self.assertEqual(path.relative_to(REPO_ROOT).as_posix(), record["source_path"])

    def test_cli_resolves_workspace_from_an_unrelated_working_directory(self):
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT)]))
        with tempfile.TemporaryDirectory() as tmp:
            output = subprocess.check_output(
                [sys.executable, "-m", "pubg_lab.repository.repository_release", "--help"],
                cwd=tmp,
                env=env,
                text=True,
            )
        self.assertIn("snapshot", output)
        self.assertEqual(repository_root(), REPO_ROOT)

    def test_survival_source_is_grouped_by_role(self):
        root = REPO_ROOT / "src/pubg_lab/survival"
        self.assertEqual(
            {p.name for p in root.glob("*.py")}, {"__init__.py", "config.py", "provenance.py"}
        )
        for path in (
            "models/stgcn.py",
            "models/snapshot_encoder.py",
            "data/dataset.py",
            "data/phase_v2/assemble.py",
            "training/train.py",
            "evaluation/eval.py",
            "features/snapshot_edge_controls.py",
            "baselines/sagong2025/features.py",
        ):
            self.assertTrue((root / path).is_file(), path)

    def test_shared_resource_helpers_do_not_import_models_or_execution(self):
        path = REPO_ROOT / "projects/world_model/common/resources.py"
        tree = ast.parse(path.read_text())
        names = {
            node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        self.assertEqual(names, {"sha", "tensor_hash", "ResourceSampler"})
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertFalse(
            any(name.startswith(("projects.", "pubg_lab.")) for name in imports if name)
        )

    def test_private_boundary_reuses_public_functions_without_copying_or_calling(self):
        from pubg_lab.survival.data import split_access

        fake = types.ModuleType("model_stgcn.splits")
        fake.split_ids = lambda: self.fail("private operation was executed")
        with patch.dict(sys.modules, {"model_stgcn.splits": fake}):
            self.assertIs(split_access.split_ids, fake.split_ids)
            with self.assertRaises(AttributeError):
                getattr(split_access, "unapproved_private_detail")


if __name__ == "__main__":
    unittest.main()
