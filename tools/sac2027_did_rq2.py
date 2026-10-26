#!/usr/bin/env python3
"""Phase-minus-match differences for the RQ2 contrasts (review item S2), on the resampled matches of the RQ2 offset analysis.

For each contrast the quantity D = (within-phase AUC difference) - (within-match AUC difference with equal phase weights)
is computed per bootstrap draw, so that "the gain depends on the evaluation unit" has an estimate and an interval instead
of a significant verdict placed next to an equivalent one.  Arms, prediction files, and the resampling seed are those of
``tools/sac2027_match_offset_rq2.py`` (``checkpoints/ptr_v2/<cohort>/<arm>/seed42/pred_test.csv``); nothing is retrained.
Per-cohort rows are pooled with DerSimonian-Laird over the 11 primary cohorts.
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
from tools.sac2027_match_offset import BASE, N_BOOT, Decomposition  # noqa: E402

OUT = BASE / "control_evaluation/did_rq2"
ARMS = ("M00", "M01", "M10", "M11", "M11R", "HGB-cut", "KRAFTON-tf-h0")
CONTRASTS = {
    "ptr_minus_krafton_phase": {"M11R": 1.0, "KRAFTON-tf-h0": -1.0},
    "gbdt_minus_krafton_phase": {"HGB-cut": 1.0, "KRAFTON-tf-h0": -1.0},
    "team_main": {"M01": 0.5, "M00": -0.5, "M11": 0.5, "M10": -0.5},
    "ptr_minus_gbdt": {"M11R": 1.0, "HGB-cut": -1.0},
}
PHASE, MATCH = "original_within_phase", "within_match_phase_mean"


def analyze(cohort: str) -> tuple[list[dict], list[dict]]:
    frames, files = {}, []
    for arm in ARMS:
        path = BASE / "checkpoints/ptr_v2" / cohort / arm / "seed42" / "pred_test.csv"
        frames[arm] = M.read_predictions(path)
        files.append(dict(path=str(path.relative_to(REPO)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    ref = frames["HGB-cut"]
    seed = int(hashlib.sha256(f"20261008|rq2_offset|{cohort}".encode()).hexdigest()[:8], 16)
    n_matches = ref["match_id"].nunique()
    counts = M.resample_counts(n_matches, N_BOOT, seed)
    point, draws = {}, {}
    for arm, frame in frames.items():
        dec = Decomposition(frame)
        point[arm] = {m: float(v[0]) for m, v in dec.evaluate(np.ones(n_matches)).items()}
        draws[arm] = dec.evaluate(counts)
    rows = []
    for name, weights in CONTRASTS.items():
        for metric, est, d in (
            ("phase_minus_match",
             sum(c * (point[a][PHASE] - point[a][MATCH]) for a, c in weights.items()),
             sum(c * (draws[a][PHASE] - draws[a][MATCH]) for a, c in weights.items())),
            (PHASE, sum(c * point[a][PHASE] for a, c in weights.items()), sum(c * draws[a][PHASE] for a, c in weights.items())),
            (MATCH, sum(c * point[a][MATCH] for a, c in weights.items()), sum(c * draws[a][MATCH] for a, c in weights.items())),
        ):
            d = d[np.isfinite(d)]
            rows.append(dict(cohort=cohort, contrast=name, metric=metric, estimate=float(est), se=float(d.std(ddof=1)),
                             ci_low=float(np.quantile(d, 0.025)), ci_high=float(np.quantile(d, 0.975)),
                             ci90_low=float(np.quantile(d, 0.05)), ci90_high=float(np.quantile(d, 0.95))))
    return rows, files


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cohorts = pd.read_csv(BASE / "cohort_counts.csv").query("analysis_role == 'main'")["cohort"].tolist()
    rows, files = [], []
    for c in cohorts:
        print("START", c, flush=True)
        r, f = analyze(c)
        rows += r
        files += f
        pd.DataFrame(rows).to_csv(OUT / "contrasts_per_cohort.csv", index=False)
    per = pd.DataFrame(rows)
    pooled = []
    for (name, m), g in per.groupby(["contrast", "metric"]):
        dl = M.dersimonian_laird(g["estimate"].to_numpy(), g["se"].to_numpy())
        pooled.append(dict(contrast=name, metric=m, same_sign=int((np.sign(g["estimate"]) == np.sign(dl["theta"])).sum()), **dl))
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(OUT / "pooled.csv", index=False)
    (OUT / "complete.json").write_text(json.dumps(dict(cohorts=cohorts, arms=ARMS, contrasts=CONTRASTS, n_boot=N_BOOT, files=files), indent=1))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(pooled[pooled.metric == "phase_minus_match"].to_string(index=False))


if __name__ == "__main__":
    main()
