"""Create and verify immutable, local repository inventories; never run experiments."""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[2]))

from pubg_lab.common.paths import REPO_ROOT

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = REPO_ROOT
RESTRICTED = {"model_stgcn/splits.py", "tools/movement_diffusion/split_v3.py"}
EXCLUDED = (
    "versions/releases/",
    "docs/repository/cleanup_2026-10-26/",
    "docs/repository/refactor_2026-10-26/",
)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def restricted(name: str) -> bool:
    path = Path(name)
    lower = path.name.lower()
    return (
        name in RESTRICTED
        or lower.startswith(".env")
        or any(x in lower for x in ("pgpass", "credential", "private_key"))
        or path.suffix.lower() in {".pem", ".key"}
    )


def file_names() -> list[str]:
    raw = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    )
    names = sorted(set(raw.decode("utf-8").rstrip("\0").split("\0")))
    return [
        name
        for name in names
        if name
        and not name.startswith(EXCLUDED)
        and not restricted(name)
        and (ROOT / name).is_file()
        and not (ROOT / name).is_symlink()
        and (ROOT / name).resolve().is_relative_to(ROOT.resolve())
    ]


def release_dir(release: str) -> Path:
    if not re.fullmatch(r"\d{4}\.\d{2}\.\d{2}-r[1-9]\d*", release):
        raise ValueError("release must have the form YYYY.MM.DD-rN")
    return ROOT / "versions" / "releases" / release


def snapshot(release: str) -> None:
    if (ROOT / "VERSION").read_text().strip() != release:
        raise ValueError("VERSION does not match the requested release")
    target = release_dir(release)
    if target.exists():
        raise FileExistsError("release already exists; create a new release")
    files = [
        {"path": name, "bytes": (ROOT / name).stat().st_size, "sha256": digest(ROOT / name)}
        for name in file_names()
    ]
    record = {
        "schema_version": 1,
        "release": release,
        "kind": "repository organization snapshot, not a new model result",
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "hash_algorithm": "sha256",
        "files": files,
        "access_restricted_paths": sorted(RESTRICTED),
        "policy": "restricted settings and split-module bodies are not read or hashed",
        "excluded_mutable_audit_paths": list(EXCLUDED),
    }
    target.mkdir(parents=True, exist_ok=False)
    (target / "manifest.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"release": release, "files": len(files), "bytes": sum(row["bytes"] for row in files)}
        )
    )


def verify(release: str) -> bool:
    record = json.loads((release_dir(release) / "manifest.json").read_text())
    errors = []
    names = []
    for row in record["files"]:
        name = row["path"]
        path = Path(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or restricted(name)
            or name.startswith(EXCLUDED)
        ):
            errors.append({"path": name, "reason": "forbidden manifest path"})
            continue
        names.append(name)
        item = ROOT / path
        if (
            item.is_symlink()
            or not item.is_file()
            or not item.resolve().is_relative_to(ROOT.resolve())
        ):
            errors.append({"path": name, "reason": "missing or symlink"})
        elif item.stat().st_size != row["bytes"] or digest(item) != row["sha256"]:
            errors.append({"path": name, "reason": "content changed"})
    if len(names) != len(set(names)):
        errors.append({"reason": "duplicate manifest paths"})
    live = set(file_names())
    expected = set(names)
    if live != expected:
        errors.append(
            {
                "reason": "file set changed",
                "added": sorted(live - expected),
                "removed": sorted(expected - live),
            }
        )
    if record.get("release") != release or (ROOT / "VERSION").read_text().strip() != release:
        errors.append({"reason": "release identity mismatch"})
    print(
        json.dumps(
            {"release": release, "verified_files": len(names), "ok": not errors, "errors": errors},
            ensure_ascii=False,
        )
    )
    return not errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("snapshot", "verify"))
    parser.add_argument("--release", required=True)
    args = parser.parse_args()
    try:
        if args.action == "snapshot":
            snapshot(args.release)
            return 0
        return 0 if verify(args.release) else 1
    except (OSError, ValueError, KeyError) as exc:
        print(f"release operation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
