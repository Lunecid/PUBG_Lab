#!/usr/bin/env python3
"""Match-offset decomposition for the structural contrasts of the main matrix (stored seed-42 test predictions only).

Applies :class:`tools.sac2027_match_offset.Decomposition` to the team-axis main effect of the 2x2 phase/team design
(``team_main`` = (M01 - M00 + M11 - M10) / 2, the primary contrast P2) and to PTR-Former minus the reference GBDT
(M11R - HGB-cut) in every main-analysis cohort, then pools cohorts with DerSimonian-Laird.  If structure adds match-level
context, its within-phase gain should sit in the match-mean between-match AUC and shrink after match-phase centring.
No model is retrained.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from model_stgcn.ptr import metrics as M  # noqa: E402
from model_stgcn.ptr.supplement import RiskSetAUC  # noqa: E402
from tools.sac2027_match_offset import BASE, N_BOOT, Decomposition, load_meta  # noqa: E402

OUT = BASE / "control_evaluation/structure_match_offset"
ARMS = ("M00", "M01", "M10", "M11", "M11R", "HGB-cut")
CONTRASTS = {
    "team_main": {"M01": 0.5, "M00": -0.5, "M11": 0.5, "M10": -0.5},
    "ptr_minus_gbdt": {"M11R": 1.0, "HGB-cut": -1.0},
}


def analyze(cohort: str, population: str, meta: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    frames, provenance = {}, []
    for arm in ARMS:
        path = BASE / "checkpoints/ptr_v2" / cohort / arm / "seed42" / "pred_test.csv"
        frame = M.read_predictions(path)
        joined = frame.merge(meta, on=M.KEY_COLUMNS, validate="one_to_one", how="outer", indicator=True,
                             suffixes=("", "_check"))
        assert bool(joined["_merge"].eq("both").all()) and np.array_equal(joined["y"], joined["y_check"])
        if population == "human":
            frame = joined.loc[joined["ai"] == 0, M.PRED_COLUMNS].reset_index(drop=True)
        frames[arm] = frame
        provenance.append(dict(path=str(path.relative_to(REPO)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    seed = int(hashlib.sha256(f"20261008|structure_offset|{cohort}|{population}".encode()).hexdigest()[:8], 16)
    n_matches = next(iter(frames.values()))["match_id"].nunique()
    counts = M.resample_counts(n_matches, N_BOOT, seed)
    point, draws = {}, {}
    for arm, frame in frames.items():
        dec = Decomposition(frame)
        assert dec.n_matches == n_matches
        point[arm] = {m: float(v[0]) for m, v in dec.evaluate(np.ones(n_matches)).items()}
        draws[arm] = dec.evaluate(counts)
        rs = RiskSetAUC(frame)
        point[arm]["within_match"] = rs.evaluate(np.ones(n_matches))
        draws[arm]["within_match"] = rs.evaluate(counts)
    rows = []
    for name, weights in CONTRASTS.items():
        for m in point[ARMS[0]]:
            est = sum(c * point[a][m] for a, c in weights.items())
            d = sum(c * draws[a][m] for a, c in weights.items())
            d = d[np.isfinite(d)]
            rows.append(dict(cohort=cohort, population=population, contrast=name, metric=m, estimate=float(est),
                             se=float(d.std(ddof=1)), ci_low=float(np.quantile(d, 0.025)), ci_high=float(np.quantile(d, 0.975))))
    return rows, provenance


def pool(per: pd.DataFrame) -> pd.DataFrame:
    """DerSimonian-Laird pooling of the per-cohort contrasts (``k`` and the interval come from the pooling helper)."""
    pooled = []
    for key, g in per.groupby(["population", "contrast", "metric"]):
        population, contrast, metric = key
        if metric == "same_match_pair_share":
            continue
        dl = M.dersimonian_laird(g["estimate"].to_numpy(), g["se"].to_numpy())
        same = int((np.sign(g["estimate"]) == np.sign(dl["theta"])).sum())
        pooled.append(dict(population=population, contrast=contrast, metric=metric, same_sign=same, **dl))
    out = pd.DataFrame(pooled)
    out.to_csv(OUT / "pooled.csv", index=False)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    roles = pd.read_csv(BASE / "cohort_counts.csv")
    cohorts = roles.loc[roles["analysis_role"] == "main", "cohort"].tolist()
    rows, provenance = [], []
    for cohort in cohorts:
        meta = load_meta(cohort)
        for population in ("all", "human"):
            print("START", cohort, population, flush=True)
            r, files = analyze(cohort, population, meta)
            rows += r
            provenance += files
            pd.DataFrame(rows).to_csv(OUT / "contrasts_per_cohort.csv", index=False)
    pool(pd.DataFrame(rows))
    (OUT / "complete.json").write_text(json.dumps(dict(cohorts=cohorts, arms=ARMS, contrasts=CONTRASTS, n_boot=N_BOOT,
                                                       files=provenance), indent=1, default=str))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
