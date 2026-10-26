"""Check physical ownership and import boundaries without executing research code."""

import ast
import json
from pathlib import Path

from pubg_lab.common.paths import REPO_ROOT

PRIVATE_MODULES = {"model_stgcn.splits", "tools.movement_diffusion.split_v3"}
OLD_ROOTS = {
    "model_stgcn",
    "world_model",
    "movement_diffusion",
    "stdb",
    "network_analysis",
    "tools",
    "scripts",
}


def code_files(root: Path) -> list[Path]:
    """Only development packages; never traverse evidence or private source roots."""
    return sorted(
        p for folder in ("src/pubg_lab", "projects") for p in (root / folder).rglob("*.py")
    )


def check(root: Path = REPO_ROOT) -> dict:
    ownership = json.loads((root / "configs/codebase.json").read_text())
    owners = set(ownership["domains"])
    project_owners = set(ownership["projects"])
    edges: dict[str, set[str]] = {owner: set() for owner in owners}
    errors = []
    files = code_files(root)
    exported = {}
    for path in files:
        name = path.relative_to(root).with_suffix("").as_posix().removeprefix("src/")
        name = name.replace("/", ".").removesuffix(".__init__")
        tree = ast.parse(path.read_bytes(), filename=str(path))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
        exported[name] = names
    for path in files:
        relative = path.relative_to(root)
        library = relative.parts[0] == "src"
        parts = relative.parts[2:] if library else relative.parts[1:]
        if len(parts) == 1:
            continue
        owner = parts[0]
        if owner not in (owners if library else project_owners):
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
                reason = None
                if library and owner in owners and dependency in owners and owner != dependency:
                    edges[owner].add(dependency)
                if library and module.startswith("projects."):
                    reason = "library depends on a project application"
                elif library and owner == "common" and dependency and dependency != "common":
                    reason = "common infrastructure depends on a domain"
                elif module.split(".")[0] in OLD_ROOTS and module not in PRIVATE_MODULES:
                    reason = "active code imports a removed root package"
                if reason:
                    errors.append(
                        {
                            "path": str(relative),
                            "line": node.lineno,
                            "reason": reason,
                            "module": module,
                        }
                    )
                if module.startswith(("pubg_lab.", "projects.")):
                    if module not in exported:
                        errors.append(
                            {
                                "path": str(relative),
                                "line": node.lineno,
                                "reason": "missing local import target",
                                "module": module,
                            }
                        )
                    elif isinstance(node, ast.ImportFrom):
                        for alias in node.names:
                            if (
                                alias.name != "*"
                                and alias.name not in exported[module]
                                and module + "." + alias.name not in exported
                            ):
                                errors.append(
                                    {
                                        "path": str(relative),
                                        "line": node.lineno,
                                        "reason": "missing local export",
                                        "module": module + "." + alias.name,
                                    }
                                )
        if any(isinstance(n, ast.Name) and n.id == "_redirect" for n in ast.walk(tree)):
            errors.append(
                {"path": str(relative), "reason": "generated forwarding shim in development code"}
            )
    packages = {p.parent for p in files if p.name == "__init__.py"}
    for package in sorted(packages):
        if not (package / "README.md").is_file():
            errors.append(
                {"path": str(package.relative_to(root)), "reason": "missing package README"}
            )
    for folder in ("world_model", "movement_diffusion", "network_analysis", "stdb", "scripts"):
        if (root / folder).exists():
            errors.append({"path": folder, "reason": "obsolete development root remains"})
    survival = root / "src/pubg_lab/survival"
    allowed = {"__init__.py", "config.py", "provenance.py"}
    for path in survival.glob("*.py"):
        if path.name not in allowed:
            errors.append(
                {
                    "path": str(path.relative_to(root)),
                    "reason": "flat survival implementation must have a role folder",
                }
            )
    for folder in (
        "models",
        "data",
        "training",
        "evaluation",
        "features",
        "experiments",
        "baselines",
    ):
        if not (survival / folder / "__init__.py").is_file():
            errors.append(
                {"path": str(survival / folder), "reason": "missing survival role package"}
            )
    for folder in ("model_stgcn", "tools"):
        for path in (root / folder).glob("*.py"):
            if str(path.relative_to(root)) not in ownership["private_source_exceptions"]:
                errors.append(
                    {"path": str(path.relative_to(root)), "reason": "root forwarding code remains"}
                )
    visited, active = set(), set()

    def visit(owner: str, trail: tuple[str, ...] = ()) -> None:
        if owner in active:
            errors.append({"reason": "domain dependency cycle", "cycle": [*trail, owner]})
            return
        if owner in visited:
            return
        active.add(owner)
        for dependency in sorted(edges[owner]):
            visit(dependency, (*trail, owner))
        active.remove(owner)
        visited.add(owner)

    for owner in sorted(owners):
        visit(owner)
    return {
        "checked_files": len(files),
        "dependency_edges": {key: sorted(value) for key, value in sorted(edges.items())},
        "ok": not errors,
        "errors": errors,
    }


def main() -> int:
    result = check()
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
