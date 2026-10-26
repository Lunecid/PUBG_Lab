#!/usr/bin/env python3
"""Review-scale meaning of the SESOI: top-3 capture differences for the primary and design contrasts (stored seed-42 test
predictions only).

For every main-analysis cohort this scores top-3 capture (``tools.sac2027_extra_evaluation.TopK``), within-match AUC, and
overall AUC for the arms of the five primary structure decisions (Table 7: phase axis, team axis, relation alignment, zone
token; the rank-correlation decision uses the team-axis arms) and of the RQ2 design contrasts, with one paired match bootstrap
per cohort, and pools each contrast with DerSimonian-Laird.  Across all (contrast, cohort) estimates it then fits the
top-3 difference on the within-match AUC difference through the origin, which converts the SESOI (0.005) into percentage
points of top-3 capture and into boundaries per additional captured elimination.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT

import hashlib
import json
import sys
from pathlib import Path

REPO = REPO_ROOT
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import pubg_lab.survival.ptr.evaluation.metrics as M  # noqa: E402
from pubg_lab.survival.ptr.evaluation.supplement import RiskSetAUC  # noqa: E402
from projects.sac2027.evaluation.metrics.sac2027_extra_evaluation import TopK  # noqa: E402

BASE = REPO / "reports/sac2027_patch_v14"
OUT = BASE / "control_evaluation/sesoi_top3"
ARMS = (
    "M00",
    "M01",
    "M10",
    "M11",
    "M11R",
    "M11R-shuf",
    "M11RZ-noHab",
    "M11R-noHab",
    "HGB-cut",
    "KRAFTON-tf-h0",
    "HGB-union",
)
H = 0.5
CONTRASTS = {
    "P1_phase_axis": {"M10": H, "M00": -H, "M11": H, "M01": -H},
    "P2_team_axis": {"M01": H, "M00": -H, "M11": H, "M10": -H},
    "P4_relation_alignment": {"M11R": 1.0, "M11R-shuf": -1.0},
    "P5_zone_token": {"M11RZ-noHab": 1.0, "M11R-noHab": -1.0},
    "PTR_minus_KRAFTON_phase": {"M11R": 1.0, "KRAFTON-tf-h0": -1.0},
    "GBDT_minus_KRAFTON_phase": {"HGB-cut": 1.0, "KRAFTON-tf-h0": -1.0},
    "PTR_minus_GBDT": {"M11R": 1.0, "HGB-cut": -1.0},
    "Union_minus_GBDT": {"HGB-union": 1.0, "HGB-cut": -1.0},
    "Logistic_minus_GBDT": {"Logistic": 1.0, "HGB-cut": -1.0},
}
PRIMARY = ("P1_phase_axis", "P2_team_axis", "P4_relation_alignment", "P5_zone_token")
METRICS = ("auc", "auc_within_match", "top3_capture")
SESOI = 0.005
N_BOOT = 2000


def analyze(cohort: str) -> tuple[list[dict], dict]:
    paths = {a: BASE / "checkpoints/ptr_v2" / cohort / a / "seed42" / "pred_test.csv" for a in ARMS}
    paths["Logistic"] = BASE / "baseline_logistic" / cohort / "pred_test.csv"
    frames = {a: M.read_predictions(p) for a, p in paths.items()}
    ref = frames["HGB-cut"]
    for a, f in frames.items():
        assert f[M.KEY_COLUMNS + ["y"]].equals(ref[M.KEY_COLUMNS + ["y"]]), (cohort, a)
    n_matches = ref["match_id"].nunique()
    seed = int(hashlib.sha256(f"20261008|sesoi_top3|{cohort}".encode()).hexdigest()[:8], 16)
    counts = M.resample_counts(n_matches, N_BOOT, seed)
    _, point, draws = M.bootstrap_arms(frames, metrics=("auc",), n_boot=N_BOOT, seed=seed)
    for a, f in frames.items():
        rs, tk = RiskSetAUC(f), TopK(f, f["p_hazard"].to_numpy())
        point[a]["auc_within_match"], draws[a]["auc_within_match"] = (
            rs.evaluate(np.ones(n_matches)),
            rs.evaluate(counts),
        )
        point[a]["top3_capture"], draws[a]["top3_capture"] = (
            tk.evaluate(np.ones(n_matches)),
            tk.evaluate(counts),
        )
    rows = []
    for name, weights in CONTRASTS.items():
        for m in METRICS:
            est = sum(c * point[a][m] for a, c in weights.items())
            d = sum(c * np.asarray(draws[a][m]) for a, c in weights.items())
            d = d[np.isfinite(d)]
            rows.append(
                dict(
                    cohort=cohort,
                    contrast=name,
                    metric=m,
                    estimate=float(est),
                    se=float(d.std(ddof=1)),
                    ci_low=float(np.quantile(d, 0.025)),
                    ci_high=float(np.quantile(d, 0.975)),
                )
            )
    tk = TopK(ref, ref["p_hazard"].to_numpy())
    boundaries = int(ref.groupby(["match_id", "phase"])["y"].sum().gt(0).sum())
    info = dict(
        cohort=cohort,
        n_matches=n_matches,
        boundaries_with_elimination=boundaries,
        eliminations=float(tk.events.sum()),
        eliminations_per_boundary=float(tk.events.sum() / boundaries),
    )
    return rows, info


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    roles = pd.read_csv(BASE / "cohort_counts.csv")
    cohorts = roles.loc[roles["analysis_role"] == "main", "cohort"].tolist()
    rows, infos = [], []
    for c in cohorts:
        print("START", c, flush=True)
        r, info = analyze(c)
        rows += r
        infos.append(info)
        pd.DataFrame(rows).to_csv(OUT / "contrasts_per_cohort.csv", index=False)
    per = pd.DataFrame(rows)
    pd.DataFrame(infos).to_csv(OUT / "boundaries.csv", index=False)
    pooled = []
    for (name, m), g in per.groupby(["contrast", "metric"]):
        dl = M.dersimonian_laird(g["estimate"].to_numpy(), g["se"].to_numpy())
        pooled.append(
            dict(
                contrast=name,
                metric=m,
                same_sign=int((np.sign(g["estimate"]) == np.sign(dl["theta"])).sum()),
                **dl,
            )
        )
    pd.DataFrame(pooled).to_csv(OUT / "pooled.csv", index=False)
    wide = per.pivot_table(
        index=["cohort", "contrast"], columns="metric", values="estimate"
    ).reset_index()
    x, y = wide["auc_within_match"].to_numpy(), wide["top3_capture"].to_numpy()
    beta = float(
        (x @ y) / (x @ x)
    )  # top-3 capture (proportion) per unit of within-match AUC, through the origin
    xa = wide["auc"].to_numpy()
    beta_auc = float((xa @ y) / (xa @ xa))
    rng = np.random.default_rng(20261008)  # cohort-cluster bootstrap of the slope
    cohort_ids = wide["cohort"].unique()
    boot = []
    for _ in range(N_BOOT):
        pick = rng.choice(cohort_ids, size=len(cohort_ids), replace=True)
        sub = pd.concat([wide[wide["cohort"] == c] for c in pick])
        bx, by = sub["auc_within_match"].to_numpy(), sub["top3_capture"].to_numpy()
        boot.append((bx @ by) / (bx @ bx))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    epb = float(pd.DataFrame(infos)["eliminations_per_boundary"].mean())
    pp = beta * SESOI * 100
    summary = dict(
        slope_top3_per_within_match_auc=beta,
        slope_ci95=[float(lo), float(hi)],
        slope_top3_per_overall_auc=beta_auc,
        sesoi=SESOI,
        sesoi_in_top3_pp=pp,
        sesoi_in_top3_pp_ci95=[float(lo * SESOI * 100), float(hi * SESOI * 100)],
        mean_eliminations_per_boundary=epb,
        boundaries_per_additional_capture=float(1 / (beta * SESOI * epb)),
        n_points=int(len(wide)),
        contrasts=list(CONTRASTS),
        primary=list(PRIMARY),
        cohorts=cohorts,
        n_boot=N_BOOT,
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
