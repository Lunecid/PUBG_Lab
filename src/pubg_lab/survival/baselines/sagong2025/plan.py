"""Stage ``plan``: PTR dumps -> the exact evaluation rows and phase windows (needs torch, no DB).

For every cohort the stored dumps (``<dump-dir>/<cohort>/{train,val,test}.pt``) are read with the PTR loader and the
SAME label code as the neural arms and the HGB baselines (``data.select_labels``; v1: labels v2 + winner censoring and
``data.effective_alive`` = event-based alive tokens).  Written to ``<out-dir>/<cohort>/``:

``units.parquet``    one row per (split, match, team, cut phase k = 1..7) that is a token of the dump: ``alive_cut`` (the unit is
                     "in" phase k), ``target`` (the risk-set row that every arm is scored on), ``y``, ``final_rank_norm``,
                     ``top5``, ``match_type``, and ``S`` / ``E`` = phase_start_s / phase_end_s on the LogMatchStart clock.
                     v1 dump: ``alive_cut`` = ``team_death_s > phase_start_s(k)`` (also written as ``alive_start``), rows = alive
                     or at risk.  **v2 dump** (``meta['version'] == 'v2'``): ``alive_cut`` = the dump's ``node_mask``, the teams
                     causally alive at the cut ``a_k = b_k - 10 s`` (``phase_v2.alive``), rows = ``query_mask`` (``node_mask |
                     target``: the comeback rows, risk-set teams that are causally dead at the cut, are units too), ``A`` = the
                     exact cut a_k, and only the label-valid cuts (``cut_valid``) are emitted; nothing is derived from death times.
``windows.parquet``  (split, match_id, phase, S, E[, A]) for the phases a match has (feature windows; v2: ``[S, A)`` with the
                     producer's exact ``A``).
``plan_audit.json``  counts, ``version`` and the row-equality check against ``ptr.hgb.build_rows`` (val and test) and against the
                     ``n_train_rows`` of the existing HGB-cut run.

The row-equality assertion is the guarantee that the SAGONG predictions cover exactly the PTR rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import pubg_lab.survival.ptr.data.data as D
import pubg_lab.survival.ptr.models.hgb as H
from pubg_lab.survival.ptr.common.util import dump_fingerprint
from pubg_lab.survival.baselines.sagong2025.features import ALIVE_COLUMNS  # noqa: F401  (v2 / v1 name of the "unit is in phase k" flag)

N_CUTS = 7  # cut phases 1..7 (dump slots 0..6)


def _split_frames(dump: dict, split: str, labels: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    v2 = D.dump_version(dump) == D.DUMP_V2
    event = D.uses_event_alive(dump, labels)
    units, windows = [], []
    for m in dump["matches"]:
        y, target = D.select_labels(m, labels)
        alive = D.key_mask(
            m, labels, event
        ).numpy()  # v1: the alive tokens; v2: node_mask (causally alive at a_k)
        rows = (
            D.effective_alive(m, labels, event).numpy() | target.numpy()
        )  # v1: alive or at risk; v2: query_mask
        y, target = y.numpy(), target.numpy()
        team_ids = np.asarray(m["team_ids"], dtype=np.int64)
        frn = np.asarray(m["final_rank_norm"], dtype=np.float64)
        top5 = np.asarray(m["top5"], dtype=np.float64)
        start = (
            D.boundary_times(m["phase_start_s"])
            if v2
            else np.asarray(m["phase_start_s"], dtype=np.float64)
        )
        end = (
            D.boundary_times(m["phase_end_s"])
            if v2
            else np.asarray(m["phase_end_s"], dtype=np.float64)
        )
        pmask = np.asarray(m["phase_mask"], dtype=bool)
        valid = (
            np.asarray(m["cut_valid"], dtype=bool)
            if (v2 and "cut_valid" in m)
            else target.any(axis=1)
        )
        mid, mtype = str(m["match_id"]), str(m["match_type"])
        for p in range(N_CUTS):
            if not pmask[p]:
                continue
            if (
                v2 and not valid[p]
            ):  # excluded cut (truncated log): no label, no unit, no feature window
                continue
            if np.isfinite(start[p]) and np.isfinite(end[p]):
                row = (split, mid, p + 1, float(start[p]), float(end[p]))
                windows.append(row + (float(end[p]) - D.CUT_LEAD_S,) if v2 else row)
            sel = np.flatnonzero(rows[p])
            if not len(sel):
                continue
            frame = {
                "split": split, "match_id": mid, "match_type": mtype, "team_id": team_ids[sel],
                "phase": p + 1, "alive_cut": alive[p, sel], "target": target[p, sel],
                "y": y[p, sel].astype(np.float64), "final_rank_norm": frn[sel], "top5": top5[sel],
                "S": float(start[p]), "E": float(end[p]),
            }  # fmt: skip
            if not v2:
                frame["alive_start"] = frame[
                    "alive_cut"
                ]  # v1 name (the v1 caches and the code that reads them)
            else:
                frame["A"] = float(end[p]) - D.CUT_LEAD_S
            units.append(pd.DataFrame(frame))
    cols = ["split", "match_id", "phase", "S", "E"] + (["A"] if v2 else [])
    return pd.concat(units, ignore_index=True), pd.DataFrame(windows, columns=cols)


def _rows_equal(units: pd.DataFrame, dump: dict, labels: str) -> dict:
    """Assert the plan's risk-set rows equal ``ptr.hgb.build_rows`` (keys, y, final_rank_norm, top5, match_type)."""
    ref = H.build_rows(dump, "HGB-cut", labels)
    mine = (
        units[units["target"]]
        .sort_values(["match_id", "team_id", "phase"], kind="mergesort")
        .reset_index(drop=True)
    )
    rk = ref.keys.assign(
        y=ref.y, final_rank_norm=ref.final_rank_norm, top5=ref.top5, match_type=ref.match_type
    )
    rk = rk.sort_values(["match_id", "team_id", "phase"], kind="mergesort").reset_index(drop=True)
    assert len(mine) == len(rk), f"row count {len(mine)} != {len(rk)}"
    for c in ("match_id", "team_id", "phase", "match_type"):
        assert np.array_equal(mine[c].to_numpy().astype(str), rk[c].to_numpy().astype(str)), (
            f"column {c} differs"
        )
    assert np.array_equal(mine["y"].to_numpy(), rk["y"].to_numpy()), "y differs"
    assert np.array_equal(
        mine["top5"].to_numpy().astype(np.float32), rk["top5"].to_numpy().astype(np.float32)
    ), "top5 differs"
    assert np.allclose(mine["final_rank_norm"], rk["final_rank_norm"], rtol=0, atol=1e-6), (
        "final_rank_norm differs"
    )
    return {"n_rows": int(len(mine)), "n_pos": int(mine["y"].sum())}


def windows_sha1(windows: pd.DataFrame) -> str:
    """Fingerprint of the feature windows (what ``extract`` read): ``features_meta.json`` repeats it so that ``fit`` can refuse
    features that were extracted for other windows (for example the v1 windows next to a v2 plan)."""
    import hashlib

    w = windows.sort_values(["split", "match_id", "phase"], kind="mergesort").reset_index(drop=True)
    return hashlib.sha1(w.to_csv(index=False, float_format="%.9f").encode("utf-8")).hexdigest()


def build_plan(
    dump_dir, cohort: str, out_dir, labels: str = D.DEFAULT_LABELS, ref_runs: str | None = None
) -> dict:
    splits = D.load_cohort(dump_dir, cohort, labels=labels)
    out = Path(out_dir) / cohort
    out.mkdir(parents=True, exist_ok=True)
    units_all, windows_all, audit = [], [], {"cohort": cohort, "labels": labels, "splits": {}}
    for s in D.SPLITS:
        units, windows = _split_frames(splits[s], s, labels)
        units_all.append(units)
        windows_all.append(windows)
        info = {
            "n_matches": int(units["match_id"].nunique()),
            "n_unit_phase_rows": int(len(units)),
            "n_alive_start": int(units["alive_cut"].sum()),
            "n_alive_cut": int(units["alive_cut"].sum()),
            "n_target": int(units["target"].sum()),
            "n_pos": int(units.loc[units["target"], "y"].sum()),
            "n_comeback": int((units["target"] & ~units["alive_cut"]).sum()),
        }
        if s in ("val", "test"):
            info["row_check_vs_hgb_build_rows"] = _rows_equal(units, splits[s], labels)
        audit["splits"][s] = info
    if ref_runs:  # the existing HGB-cut run: same train row count
        p = Path(ref_runs) / cohort / "HGB-cut" / "seed42" / "metrics.json"
        if p.is_file():
            n_ref = int(json.loads(p.read_text())["n_train_rows"])
            assert n_ref == audit["splits"]["train"]["n_target"], (
                f"{cohort}: train rows {audit['splits']['train']['n_target']} != HGB-cut {n_ref}"
            )
            audit["train_rows_equal_hgb_cut_run"] = n_ref
    pd.concat(units_all, ignore_index=True).to_parquet(out / "units.parquet", index=False)
    pd.concat(windows_all, ignore_index=True).to_parquet(out / "windows.parquet", index=False)
    audit["meta"] = {
        k: splits["train"]["meta"].get(k)
        for k in ("cohort", "design_version", "git_commit", "n_phases", "version")
    }
    audit["version"] = D.dump_version(splits["train"])
    audit["alive_status"] = (
        "causal alive at a_k (the dump's node_mask); rows = query_mask; cut_valid cuts only"
        if audit["version"] == D.DUMP_V2
        else "team_death_s > phase_start_s (alive at the start of the phase)"
    )
    audit["windows_sha1"] = windows_sha1(pd.concat(windows_all, ignore_index=True))
    audit["dump_fingerprint"] = dump_fingerprint(
        dump_dir, cohort
    )  # {split: [size, mtime_ns]} of the dump files read
    (out / "plan_audit.json").write_text(json.dumps(audit, indent=1, default=str) + "\n")
    return audit
