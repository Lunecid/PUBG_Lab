"""Small shared helpers: provenance, seeding, strict-JSON dumping."""

from __future__ import annotations

from pubg_lab.common.paths import REPO_ROOT

import hashlib
import json
import math
import random
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = REPO_ROOT
PTR_DIR = REPO_ROOT / "src/pubg_lab/survival/ptr"
# modules whose source can change a result: any edit invalidates cached runs (config hash).  The runner, the
# aggregator, the cohort lister and the synthetic generator do not feed results and are left out on purpose.
RESULT_MODULES: tuple[str, ...] = (
    "arms", "data", "extra", "hgb", "loss", "metrics", "model", "player", "relations", "train", "util",
)  # fmt: skip


# Logical module keys stay fixed; their source locations follow physical role folders.
RESULT_MODULE_PATHS = {
    "arms": "models/arms.py",
    "data": "data/data.py",
    "extra": "data/extra.py",
    "hgb": "models/hgb.py",
    "loss": "training/loss.py",
    "metrics": "evaluation/metrics.py",
    "model": "models/model.py",
    "player": "models/player.py",
    "relations": "models/relations.py",
    "train": "training/train.py",
    "util": "common/util.py",
}


def git_info() -> dict[str, Any]:
    """Commit / branch / tracked-file dirtiness (untracked files are not scanned: fast and enough)."""

    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL, text=True, timeout=20
            ).strip()
        except Exception:
            return None

    status = run("status", "--porcelain", "--untracked-files=no") or ""
    dirty = [line for line in status.splitlines() if line.strip()]
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_tracked": bool(dirty),
        "dirty_tracked_files": dirty[:20],
    }


def code_fingerprint() -> str:
    """SHA-1 over the source of the result-affecting modules (:data:`RESULT_MODULES`)."""
    h = hashlib.sha1()
    for name in sorted(RESULT_MODULES):
        h.update(name.encode() + b"\0" + (PTR_DIR / RESULT_MODULE_PATHS[name]).read_bytes() + b"\0")
    return h.hexdigest()


def dump_fingerprint(
    dump_dir: str | Path, cohort: str, splits: Sequence[str] = ("train", "val", "test")
) -> dict:
    """``{split: [size, mtime_ns]}`` of the dump files (None for a missing file): changes when a cohort is re-dumped."""
    out = {}
    for split in splits:
        path = Path(dump_dir) / cohort / f"{split}.pt"
        try:
            st = path.stat()
            out[split] = [st.st_size, st.st_mtime_ns]
        except OSError:
            out[split] = None
    return out


def config_hash(spec: Mapping[str, Any]) -> str:
    """16-hex digest of a run specification (canonical JSON)."""
    return hashlib.sha1(json.dumps(to_jsonable(dict(spec)), sort_keys=True).encode()).hexdigest()[
        :16
    ]


def run_dir_name(seed: int, subsample: int | None = None, subsample_seed: int | None = None) -> str:
    """Directory of a run below ``<out>/<cohort>/<arm>/``: ``seed<s>`` for a full-size run, and
    ``sub<N>_ss<subsample seed>_seed<s>`` when the training set is a random subsample of N matches."""
    if subsample is None:
        return f"seed{seed}"
    return f"sub{int(subsample)}_ss{seed if subsample_seed is None else int(subsample_seed)}_seed{seed}"


def check_run_dir(out_dir: str | Path, labels: str) -> None:
    """Refuse to overwrite the results of a run that used a different label choice.

    Re-running the same arm / seed / labels into a directory is fine (it replaces the files); writing
    ``--labels pipeline`` results over ``--labels v2`` ones (or the reverse) would silently destroy a
    finished run, so that raises ``FileExistsError`` and the caller must pick another ``--out-dir``.
    """
    path = Path(out_dir) / "metrics.json"
    if not path.is_file():
        return
    try:
        previous = json.loads(path.read_text(encoding="utf-8")).get("labels")
    except (ValueError, OSError):
        return
    if previous is not None and previous != labels:
        raise FileExistsError(
            f"{out_dir} already holds results made with labels={previous!r}; refusing to overwrite them "
            f"with labels={labels!r} (use another --out-dir)"
        )


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU runs are bitwise reproducible; MPS is not guaranteed)."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert to strict JSON types (NaN / inf become null)."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), indent=2, sort_keys=False, allow_nan=False) + "\n")
