"""Check code ownership and import boundaries without executing research code."""

import ast
import json
from pathlib import Path

from pubg_lab.common.paths import REPO_ROOT

PRIVATE_MODULES = {"model_stgcn.splits", "tools.movement_diffusion.split_v3"}
OLD_ROOTS = {
    "world_model",
    "movement_diffusion",
    "model_stgcn",
    "stdb",
    "network_analysis",
    "tools",
    "scripts",
}


def check(root: Path = REPO_ROOT) -> dict:
    source = root / "src/pubg_lab"
    ownership = json.loads((root / "configs/codebase.json").read_text())
    owners = set(ownership["domains"])
    errors = []
    dependency_edges: dict[str, set[str]] = {owner: set() for owner in owners}
    libraries = {"data", "world_model", "movement", "survival", "spatial", "analysis"}
    applications = {"projects", "collectors", "repository"}
    files = sorted(source.rglob("*.py"))
    for path in files:
        relative = path.relative_to(source)
        if len(relative.parts) > 1 and relative.parts[0] not in owners:
            errors.append({"path": str(relative), "reason": "missing owner"})
        tree = ast.parse(path.read_bytes(), filename=str(path))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            for module in modules:
                dependency = module.split(".")[1] if module.startswith("pubg_lab.") else None
                owner = relative.parts[0]
                if owner in owners and dependency in owners and owner != dependency:
                    dependency_edges[owner].add(dependency)
                if owner in libraries and dependency in applications:
                    errors.append(
                        {
                            "path": str(relative),
                            "line": node.lineno,
                            "reason": "library depends on an application",
                            "module": module,
                        }
                    )
                if module.split(".")[0] in OLD_ROOTS and module not in PRIVATE_MODULES:
                    errors.append(
                        {
                            "path": str(relative),
                            "line": node.lineno,
                            "reason": "canonical code imports compatibility layer",
                            "module": module,
                        }
                    )
                if relative.parts[0] == "common" and module.startswith("pubg_lab."):
                    errors.append(
                        {
                            "path": str(relative),
                            "line": node.lineno,
                            "reason": "common infrastructure depends on a domain",
                        }
                    )
    for path in source.rglob("*"):
        if path.is_dir() and path.name != "__pycache__" and not (path / "README.md").is_file():
            errors.append(
                {"path": str(path.relative_to(source)), "reason": "missing package README"}
            )
    visited: set[str] = set()
    active: set[str] = set()

    def visit(owner: str, trail: tuple[str, ...] = ()) -> None:
        if owner in active:
            errors.append({"reason": "domain dependency cycle", "cycle": [*trail, owner]})
            return
        if owner in visited:
            return
        active.add(owner)
        for dependency in sorted(dependency_edges[owner]):
            visit(dependency, (*trail, owner))
        active.remove(owner)
        visited.add(owner)

    for owner in sorted(owners):
        visit(owner)
    return {"checked_files": len(files), "ok": not errors, "errors": errors}


def main() -> int:
    result = check()
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
