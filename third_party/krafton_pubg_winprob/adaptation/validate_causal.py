"""Prefix-replay check of the feature reconstruction on REAL matches (read-only DB access).

For a sample of matches of a cohort split the squad features are computed twice at the evaluation cut times and at
KRAFTON-style random train times: once from the full raw rows of the match and once from the rows that exist at
time t only (``extract.truncate_raw``: every row with event time >= t removed for v2 labels, > t for v1).  A causal
reconstruction gives identical features (v2: bit for bit); any difference would mean that a feature reads information
from after t.  v1 labels: also checks that the reconstructed team death times equal the dump's ``team_death_s``.  v2 labels: also
checks that the causal squad status at every cut equals the dump's ``node_mask`` and that it is unchanged by the truncation, and
prints the share of squads without any position.

    python -m adaptation.validate_causal --cohorts karakin_tpp_squad,taego_tpp_squad --n 12 [--data-dir <label tables>]
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
from adaptation.features import compute_features, team_alive_causal  # noqa: E402
from adaptation.labels import load_labels  # noqa: E402


def truncate(raw: dict, t: float) -> dict:
    """Kept name: the rows that exist at ``t`` (see ``extract.truncate_raw``: strict ``< t`` for v2, ``<= t`` for v1)."""
    return E.truncate_raw(raw, t)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--data-dir", default=str(E.DEFAULT_OUT))
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    rng = np.random.RandomState(0)
    report = {"cohorts": {}, "max_abs_diff": 0.0, "n_checked": 0, "n_nonidentical": 0, "alive_mismatch": 0}
    for cohort in a.cohorts.split(","):
        labs = load_labels(a.data_dir, cohort, "test")["matches"]
        pick = [labs[i] for i in rng.choice(len(labs), size=min(a.n, len(labs)), replace=False)]
        T = E.fetch_batch([m["match_id"] for m in pick])
        meta = T["meta"].set_index("match_id")
        TT = {k: E._by_match(v) for k, v in T.items() if k != "meta"}
        rec = {"n_matches": 0, "n_times": 0, "max_abs_diff": 0.0, "death_mismatch": 0, "alive_mismatch": 0, "squads_without_position": 0, "squad_rows": 0}
        for lab in pick:
            mid = lab["match_id"]
            row = meta.loc[mid]
            raw, extra = E.build_raw(mid, lab, TT, row, float(row.started_ep))
            v2 = E._is_v2(lab)
            rec["death_mismatch"] += extra["flags"]["n_death_mismatch"]
            slots = [p for p in range(len(lab["phase_end_s"])) if lab["target"][p].any()]
            cuts = lab["cut_t"][slots] if v2 else np.asarray([lab["phase_end_s"][p] - 10.0 for p in slots])
            ts = np.sort(np.concatenate([cuts, E.krafton_times(mid)[::17]]))
            ts = ts[ts >= extra["flags"]["first_state_w"]]
            full = compute_features(raw, ts)
            if v2:  # the causal squad status at every cut is the dump's node_mask
                alive_cut = team_alive_causal(raw, cuts)
                rec["alive_mismatch"] += int((alive_cut != lab["node_mask"][slots].T).sum())
            for k, t in enumerate(ts):
                part = compute_features(truncate(raw, float(t)), np.array([t]))
                d = np.abs(full["F"][:, k, :].astype(np.float64) - part["F"][:, 0, :].astype(np.float64))
                pos_same = np.array_equal(np.isnan(full["POS"][:, k]), np.isnan(part["POS"][:, 0])) and np.allclose(
                    np.nan_to_num(full["POS"][:, k]), np.nan_to_num(part["POS"][:, 0]))
                rec["max_abs_diff"] = max(rec["max_abs_diff"], float(d.max()))
                report["n_checked"] += 1
                if v2:  # bit for bit, and the squad status itself must not change under truncation
                    same = np.array_equal(full["F"][:, k, :], part["F"][:, 0, :], equal_nan=True) and pos_same
                    same &= np.array_equal(team_alive_causal(raw, np.array([t]))[:, 0], team_alive_causal(truncate(raw, float(t)), np.array([t]))[:, 0])
                    report["n_nonidentical"] += int(not same)
                    alive = team_alive_causal(raw, np.array([t]))[:, 0]
                else:
                    report["n_nonidentical"] += int((d.max() > 1e-6) or (not pos_same))
                    alive = (np.where(lab["team_won"], np.inf, lab["team_death_s"]) > t)
                rec["squad_rows"] += int(alive.sum())
                rec["squads_without_position"] += int((np.isnan(full["POS"][alive, k, :, 0]).all(axis=1)).sum())
            rec["n_matches"] += 1
            rec["n_times"] += len(ts)
        report["cohorts"][cohort] = rec
        report["max_abs_diff"] = max(report["max_abs_diff"], rec["max_abs_diff"])
        report["alive_mismatch"] += rec["alive_mismatch"]
        print(cohort, json.dumps(rec), flush=True)
    print("TOTAL", json.dumps({k: v for k, v in report.items() if k != "cohorts"}))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))
    return 0 if (report["n_nonidentical"] == 0 and report["alive_mismatch"] == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
