"""Shared match-level split helpers for phase-model experiments."""

from __future__ import annotations

import hashlib
import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from model_stgcn.provenance import git_provenance

logger = logging.getLogger("model_stgcn.splits")


def split_ids(
    ids: list[str],
    *,
    seed: int = 42,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> tuple[list[str], list[str], list[str]]:
    """Deterministic train/val/test split at match granularity."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ids))
    rng.shuffle(idx)
    n = len(idx)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_ids = [ids[i] for i in idx[:n_test]]
    val_ids = [ids[i] for i in idx[n_test : n_test + n_val]]
    train_ids = [ids[i] for i in idx[n_test + n_val :]]
    return train_ids, val_ids, test_ids


def _ids_sha1(ids: list[str]) -> str:
    payload = "\n".join(ids).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def write_split_manifest(
    path: str | Path,
    *,
    ids: list[str],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    seed: int,
    val_frac: float,
    test_frac: float,
    metadata: dict[str, Any] | None = None,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "n_matches": len(ids),
        "ids_sha1": _ids_sha1(ids),
        "metadata": metadata or {},
        "provenance": git_provenance(),
        "splits": {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        },
    }
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_split_manifest(path: str | Path, *, available_ids: list[str] | None = None) -> tuple[list[str], list[str], list[str]]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = manifest["splits"]
    train_ids = list(splits["train"])
    val_ids = list(splits["val"])
    test_ids = list(splits["test"])

    all_split_ids = train_ids + val_ids + test_ids
    if len(set(all_split_ids)) != len(all_split_ids):
        raise ValueError(f"split manifest has duplicate match ids: {path}")

    if available_ids is not None:
        available = set(available_ids)
        missing = [mid for mid in all_split_ids if mid not in available]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"split manifest contains {len(missing)} ids outside current candidate set: {preview}"
            )

    return train_ids, val_ids, test_ids


def load_or_create_split(
    ids: list[str],
    *,
    split_json: str | None,
    seed: int = 42,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    metadata: dict[str, Any] | None = None,
    require_existing: bool = False,
) -> tuple[list[str], list[str], list[str]]:
    """Load a split manifest if present; otherwise create it when requested."""
    if split_json:
        path = Path(split_json)
        if path.exists():
            manifest_seed = json.loads(path.read_text(encoding="utf-8")).get("seed")
            if manifest_seed is not None and manifest_seed != seed:
                logger.warning(
                    "split manifest %s was created with seed=%s; --seed %s is IGNORED "
                    "(manifest takes precedence)", path, manifest_seed, seed,
                )
            return load_split_manifest(path, available_ids=ids)
        if require_existing:
            raise FileNotFoundError(
                f"required split manifest does not exist: {path}. "
                "Refusing to silently create a different train/val/test split."
            )
    elif require_existing:
        raise ValueError("--require-existing-split needs --split-json")

    train_ids, val_ids, test_ids = split_ids(
        ids,
        seed=seed,
        val_frac=val_frac,
        test_frac=test_frac,
    )

    if split_json:
        write_split_manifest(
            split_json,
            ids=ids,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            seed=seed,
            val_frac=val_frac,
            test_frac=test_frac,
            metadata=metadata,
        )

    return train_ids, val_ids, test_ids
