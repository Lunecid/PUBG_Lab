"""Stage ``plan``: PTR dumps -> the exact evaluation rows and phase windows (needs torch, no DB).

For every cohort the stored dumps (``<dump-dir>/<cohort>/{train,val,test}.pt``) are read with the PTR loader and the
SAME label code as the neural arms and the HGB baselines (``data.select_labels`` = labels v2 + winner censoring,
``data.effective_alive`` = event-based alive tokens).  Written to ``<out-dir>/<cohort>/``:

``units.parquet``    one row per (split, match, team, cut phase k = 1..7) that is alive at phase start or at risk:
                     ``alive_start`` (death time > phase_start_s(k): the units "in" phase k), ``target`` (the risk-set
                     row that every arm is scored on), ``y``, ``final_rank_norm``, ``top5``, ``match_type``, and
                     ``S`` / ``E`` = phase_start_s / phase_end_s on the LogMatchStart clock.
``windows.parquet``  (split, match_id, phase, S, E) for the phases a match has (feature windows).
``plan_audit.json``  counts and the row-equality check against ``ptr.hgb.build_rows`` (val and test) and against the
                     ``n_train_rows`` of the existing HGB-cut run.

The row-equality assertion is the guarantee that the SAGONG predictions cover exactly the PTR rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from model_stgcn.ptr import data as D
from model_stgcn.ptr import hgb as H
from model_stgcn.ptr.util import dump_fingerprint

N_CUTS = 7  # cut phases 1..7 (dump slots 0..6)


def _split_frames(dump: dict, split: str, labels: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    event = D.uses_event_alive(dump, labels)
    units, windows = [], []
    for m in dump["matches"]:
        y, target = D.select_labels(m, labels)
        alive = D.effective_alive(m, labels, event).numpy()
        y, target = y.numpy(), target.numpy()
        team_ids = np.asarray(m["team_ids"], dtype=np.int64)
        frn = np.asarray(m["final_rank_norm"], dtype=np.float64)
        top5 = np.asarray(m["top5"], dtype=np.float64)
        start = np.asarray(m["phase_start_s"], dtype=np.float64)
        end = np.asarray(m["phase_end_s"], dtype=np.float64)
        pmask = np.asarray(m["phase_mask"], dtype=bool)
        mid, mtype = str(m["match_id"]), str(m["match_type"])
        for p in range(N_CUTS):
            if not pmask[p]:
                continue
            if np.isfinite(start[p]) and np.isfinite(end[p]):
                windows.append((split, mid, p + 1, float(start[p]), float(end[p])))
            sel = np.flatnonzero(alive[p] | target[p])
            if not len(sel):
                continue
            units.append(
                pd.DataFrame(
                    {
                        "split": split, "match_id": mid, "match_type": mtype, "team_id": team_ids[sel],
                        "phase": p + 1, "alive_start": alive[p, sel], "target": target[p, sel],
                        "y": y[p, sel].astype(np.float64), "final_rank_norm": frn[sel], "top5": top5[sel],
                        "S": float(start[p]), "E": float(end[p]),
                    }
                )
            )
    return pd.concat(units, ignore_index=True), pd.DataFrame(windows, columns=["split", "match_id", "phase", "S", "E"])


def _rows_equal(units: pd.DataFrame, dump: dict, labels: str) -> dict:
    """Assert the plan's risk-set rows equal ``ptr.hgb.build_rows`` (keys, y, final_rank_norm, top5, match_type)."""
    ref = H.build_rows(dump, "HGB-cut", labels)
    mine = units[units["target"]].sort_values(["match_id", "team_id", "phase"], kind="mergesort").reset_index(drop=True)
    rk = ref.keys.assign(y=ref.y, final_rank_norm=ref.final_rank_norm, top5=ref.top5, match_type=ref.match_type)
    rk = rk.sort_values(["match_id", "team_id", "phase"], kind="mergesort").reset_index(drop=True)
    assert len(mine) == len(rk), f"row count {len(mine)} != {len(rk)}"
    for c in ("match_id", "team_id", "phase", "match_type"):
        assert np.array_equal(mine[c].to_numpy().astype(str), rk[c].to_numpy().astype(str)), f"column {c} differs"
    assert np.array_equal(mine["y"].to_numpy(), rk["y"].to_numpy()), "y differs"
    assert np.array_equal(mine["top5"].to_numpy().astype(np.float32), rk["top5"].to_numpy().astype(np.float32)), "top5 differs"
    assert np.allclose(mine["final_rank_norm"], rk["final_rank_norm"], rtol=0, atol=1e-6), "final_rank_norm differs"
    return {"n_rows": int(len(mine)), "n_pos": int(mine["y"].sum())}


def build_plan(dump_dir, cohort: str, out_dir, labels: str = D.DEFAULT_LABELS, ref_runs: str | None = None) -> dict:
    splits = D.load_cohort(dump_dir, cohort, labels=labels)
    out = Path(out_dir) / cohort
    out.mkdir(parents=True, exist_ok=True)
    units_all, windows_all, audit = [], [], {"cohort": cohort, "labels": labels, "splits": {}}
    for s in D.SPLITS:
        units, windows = _split_frames(splits[s], s, labels)
        units_all.append(units)
        windows_all.append(windows)
        info = {"n_matches": int(units["match_id"].nunique()), "n_unit_phase_rows": int(len(units)),
                "n_alive_start": int(units["alive_start"].sum()), "n_target": int(units["target"].sum()),
                "n_pos": int(units.loc[units["target"], "y"].sum())}
        if s in ("val", "test"):
            info["row_check_vs_hgb_build_rows"] = _rows_equal(units, splits[s], labels)
        audit["splits"][s] = info
    if ref_runs:  # the existing HGB-cut run: same train row count
        p = Path(ref_runs) / cohort / "HGB-cut" / "seed42" / "metrics.json"
        if p.is_file():
            n_ref = int(json.loads(p.read_text())["n_train_rows"])
            assert n_ref == audit["splits"]["train"]["n_target"], f"{cohort}: train rows {audit['splits']['train']['n_target']} != HGB-cut {n_ref}"
            audit["train_rows_equal_hgb_cut_run"] = n_ref
    pd.concat(units_all, ignore_index=True).to_parquet(out / "units.parquet", index=False)
    pd.concat(windows_all, ignore_index=True).to_parquet(out / "windows.parquet", index=False)
    audit["meta"] = {k: splits["train"]["meta"].get(k) for k in ("cohort", "design_version", "git_commit", "n_phases")}
    audit["dump_fingerprint"] = dump_fingerprint(dump_dir, cohort)  # {split: [size, mtime_ns]} of the dump files read
    (out / "plan_audit.json").write_text(json.dumps(audit, indent=1, default=str) + "\n")
    return audit
