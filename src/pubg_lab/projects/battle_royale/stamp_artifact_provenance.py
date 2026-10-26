"""Stamp existing experiment artifact roots with code and file hashes."""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pubg_lab.survival.provenance import file_sha256, git_provenance


DEFAULT_ROOTS = [
    Path("model_stgcn/checkpoints/full_required_core13_v1"),
    Path("model_stgcn/checkpoints/followup_erangel_required_v1"),
    Path("model_stgcn/checkpoints/br_o7_advantage_edge_required_v1"),
    Path("model_stgcn/checkpoints/br_rq4_graph_design_required_v1"),
    Path("model_stgcn/checkpoints/br_rq4_prior_baseline_required_v1"),
    Path("model_stgcn/checkpoints/br_rq4_prior_model_zoo_required_v1"),
    Path("model_stgcn/checkpoints/br_rq4_same_condition_snapshot_required_v2"),
    Path("model_stgcn/checkpoints/br_rq4_sequence_model_zoo_required_v1"),
    Path("reports/rq4_board_baseline_comparison"),
    Path("reports/rq4_dynamic_probability_fluctuation"),
]

EVIDENCE_SUFFIXES = {".csv", ".json", ".log", ".md", ".tsv", ".txt"}
SKIP_NAMES = {"_provenance.json"}


def _evidence_files(root: Path) -> list[Path]:
    files: list[Path] = []
    if not root.exists():
        return files
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in SKIP_NAMES:
            continue
        if path.suffix.lower() not in EVIDENCE_SUFFIXES:
            continue
        files.append(path)
    return files


def stamp_root(root: Path) -> dict[str, Any]:
    files = _evidence_files(root)
    inventory = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in files
    ]
    manifest = {
        "artifact_root": str(root),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "provenance": git_provenance(),
        "evidence_file_count": len(inventory),
        "evidence_files": inventory,
    }
    root.mkdir(parents=True, exist_ok=True)
    out = root / "_provenance.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "artifact_root": str(root),
        "provenance_json": str(out),
        "evidence_file_count": len(inventory),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=DEFAULT_ROOTS,
        help="artifact roots to stamp; defaults to required manuscript evidence roots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [stamp_root(root) for root in args.roots if root.exists()]
    print(json.dumps({"stamped": rows}, indent=2))


if __name__ == "__main__":
    main()
