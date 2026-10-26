"""Resolve the workspace explicitly without opening data or private settings."""

import os
from pathlib import Path


def repository_root() -> Path:
    """Find a source checkout, or use an explicitly supplied installed workspace."""
    configured = os.environ.get("PUBG_REPOSITORY_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not (root / "VERSION").is_file():
            raise RuntimeError("PUBG_REPOSITORY_ROOT must point to a PUBG workspace")
        return root
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "VERSION").is_file() and (candidate / "docs/guidelines").is_dir():
            return candidate
    raise RuntimeError(
        "An installed package needs PUBG_REPOSITORY_ROOT for local research artifacts"
    )


REPO_ROOT = repository_root()
