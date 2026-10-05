"""Prefix-replay check of the feature reconstruction on REAL matches (read-only DB access).

For a sample of matches of a cohort split the squad features are computed twice at the evaluation cut times and at
KRAFTON-style random train times: once from the full raw rows of the match and once from the rows that exist at
time t only (every row with event time > t removed).  A causal reconstruction gives identical features; any
difference would mean that a feature reads information from after t.  Also checks that the reconstructed team
death times equal the dump's ``team_death_s`` and prints the share of squads without any position.

    python -m adaptation.validate_causal --cohorts karakin_tpp_squad,taego_tpp_squad --n 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaptation import extract as E  # noqa: E402
from adaptation.features import compute_features  # noqa: E402
from adaptation.labels import load_labels  # noqa: E402


def truncate(raw: dict, t: float) -> dict:
    out = dict(raw)
    for key in ("kills", "groggy", "damage", "pos", "pick", "equip", "use", "zone"):
        sub = raw[key]
        m = sub["t"] <= t
        out[key] = {k: v[m] for k, v in sub.items()}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--data-dir", default=str(E.DEFAULT_OUT))
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    rng = np.random.RandomState(0)
    report = {"cohorts": {}, "max_abs_diff": 0.0, "n_checked": 0, "n_nonidentical": 0}
    for cohort in a.cohorts.split(","):
        labs = load_labels(a.data_dir, cohort, "test")["matches"]
        pick = [labs[i] for i in rng.choice(len(labs), size=min(a.n, len(labs)), replace=False)]
        T = E.fetch_batch([m["match_id"] for m in pick])
        meta = T["meta"].set_index("match_id")
        TT = {k: E._by_match(v) for k, v in T.items() if k != "meta"}
        rec = {"n_matches": 0, "n_times": 0, "max_abs_diff": 0.0, "death_mismatch": 0, "squads_without_position": 0, "squad_rows": 0}
        for lab in pick:
            mid = lab["match_id"]
            row = meta.loc[mid]
            raw, extra = E.build_raw(mid, lab, TT, row, float(row.started_ep))
            rec["death_mismatch"] += extra["flags"]["n_death_mismatch"]
            slots = [p for p in range(len(lab["phase_end_s"])) if lab["target"][p].any()]
            ts = np.sort(np.concatenate([[lab["phase_end_s"][p] - 10.0 for p in slots], E.krafton_times(mid)[::17]]))
            ts = ts[ts >= extra["flags"]["first_state_w"]]
            full = compute_features(raw, ts)
            for k, t in enumerate(ts):
                part = compute_features(truncate(raw, float(t)), np.array([t]))
                d = np.abs(full["F"][:, k, :].astype(np.float64) - part["F"][:, 0, :].astype(np.float64))
                pos_same = np.array_equal(np.isnan(full["POS"][:, k]), np.isnan(part["POS"][:, 0])) and np.allclose(
                    np.nan_to_num(full["POS"][:, k]), np.nan_to_num(part["POS"][:, 0]))
                rec["max_abs_diff"] = max(rec["max_abs_diff"], float(d.max()))
                report["n_checked"] += 1
                report["n_nonidentical"] += int((d.max() > 1e-6) or (not pos_same))
                alive = (np.where(lab["team_won"], np.inf, lab["team_death_s"]) > t)
                rec["squad_rows"] += int(alive.sum())
                rec["squads_without_position"] += int((np.isnan(full["POS"][alive, k, :, 0]).all(axis=1)).sum())
            rec["n_matches"] += 1
            rec["n_times"] += len(ts)
        report["cohorts"][cohort] = rec
        report["max_abs_diff"] = max(report["max_abs_diff"], rec["max_abs_diff"])
        print(cohort, json.dumps(rec), flush=True)
    print("TOTAL", json.dumps({k: v for k, v in report.items() if k != "cohorts"}))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))
    return 0 if report["n_nonidentical"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
