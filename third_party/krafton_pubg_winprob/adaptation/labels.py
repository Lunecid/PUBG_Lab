"""Compact label tables of the PTR dumps (model_stgcn/dumps/bigcomp2027_phase_v1) for the KRAFTON baseline.

The baseline uses the dumps' cohorts, splits, event based death times (``team_death_s``), winner flags, phase
boundaries and the v2 risk set / hazard labels (``select_labels`` of model_stgcn.ptr.data is called, not
re-implemented, so that the rows are the PTR rows by construction).  Only small per-match arrays are kept; the
big feature tensors of the dump are dropped, so workers and the training process never load them.

    python -m adaptation.labels --dump-dir model_stgcn/dumps/bigcomp2027_phase_v1 --out-dir <out> [--cohorts a,b]
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from model_stgcn.ptr.data import DEFAULT_MATCH_TYPES, check_match_types, select_labels  # noqa: E402  (read-only use)

SPLITS = ("train", "val", "test")


def _np(x, dtype=None) -> np.ndarray:
    a = torch.as_tensor(x).numpy() if not isinstance(x, np.ndarray) else x
    return a if dtype is None else a.astype(dtype)


def compact_match(m: dict) -> dict:
    y, target = select_labels(m, "v2")  # v2 risk set + winner censoring, exactly as the PTR arms
    out = {
        "match_id": str(m["match_id"]),
        "match_type": str(m["match_type"]),
        "team_ids": _np(m["team_ids"], np.int64),
        "team_death_s": _np(m["team_death_s"], np.float64),
        "team_won": _np(m["team_won"], bool),
        "phase_end_s": _np(m["phase_end_s"], np.float64),
        "phase_start_s": _np(m["phase_start_s"], np.float64),
        "phase_mask": _np(m["phase_mask"], bool),
        "target": _np(target, bool),
        "y": _np(y, np.float32),
        "final_rank_norm": _np(m["final_rank_norm"], np.float32),
        "top5": _np(m["top5"], np.float32),
        "last_phase_idx": None if m.get("last_phase_idx") is None else int(m["last_phase_idx"]),
        "clock_delta_s": float(m.get("clock_delta_s", 0.0)),
    }
    return out


def build_cohort(dump_dir: Path, cohort: str, out_dir: Path) -> dict:
    base = dump_dir / cohort
    info = {"cohort": cohort, "splits": {}}
    for split in SPLITS:
        t0 = time.time()
        dump = torch.load(base / f"{split}.pt", map_location="cpu", weights_only=False)
        check_match_types(dump, DEFAULT_MATCH_TYPES, f"{cohort}/{split}")
        meta = dump["meta"]
        mats = [compact_match(m) for m in dump["matches"]]
        rec = {
            "cohort": cohort, "split": split, "map_name": meta.get("map_name"), "game_mode": meta.get("game_mode"),
            "split_ids_sha1": meta.get("split_ids_sha1"), "matches": mats,
        }  # fmt: skip
        out = out_dir / cohort
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"labels_{split}.pkl", "wb") as f:
            pickle.dump(rec, f, protocol=4)
        n_rows = int(sum(m["target"].sum() for m in mats))
        info["splits"][split] = {"n_matches": len(mats), "n_target_rows": n_rows, "sec": round(time.time() - t0, 1)}
        print(f"{cohort}/{split}: {len(mats)} matches, {n_rows} risk-set rows, {time.time() - t0:.1f}s", flush=True)
        del dump
    return info


def load_labels(out_dir: Path | str, cohort: str, split: str) -> dict:
    with open(Path(out_dir) / cohort / f"labels_{split}.pkl", "rb") as f:
        return pickle.load(f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default=str(REPO / "model_stgcn/dumps/bigcomp2027_phase_v1"))
    ap.add_argument("--out-dir", default=str(REPO / "model_stgcn/dumps/krafton_winprob_v1"))
    ap.add_argument("--cohorts", default="")
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    dump_dir, out_dir = Path(a.dump_dir), Path(a.out_dir)
    from model_stgcn.ptr.cohorts import discover_cohorts  # read-only

    cohorts = [c for c in a.cohorts.split(",") if c] or list(discover_cohorts(dump_dir))
    for c in cohorts:
        build_cohort(dump_dir, c, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
