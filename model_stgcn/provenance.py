"""Small provenance helpers for experiment artifacts.

The experiment outputs are often copied or summarized outside the repository.
These helpers stamp artifacts with the code revision and a compact dirty-worktree
summary so a result can be tied back to the exact local code state that made it.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git(args: list[str], *, cwd: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return out.strip()


def _repo_root(cwd: Path) -> Path:
    root = _git(["rev-parse", "--show-toplevel"], cwd=cwd)
    return Path(root) if root else cwd


def git_provenance(cwd: str | Path | None = None, *, max_dirty_files: int = 50) -> dict[str, Any]:
    """Return a JSON-serializable code provenance block."""
    base = Path(cwd or Path.cwd()).resolve()
    root = _repo_root(base)
    status = _git(["status", "--short"], cwd=root) or ""
    dirty_files = [line for line in status.splitlines() if line.strip()]
    commit = _git(["rev-parse", "HEAD"], cwd=root)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(root),
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": bool(dirty_files),
        "git_dirty_count": len(dirty_files),
        "git_dirty_files_sample": dirty_files[:max_dirty_files],
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
    }


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading it all into memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
