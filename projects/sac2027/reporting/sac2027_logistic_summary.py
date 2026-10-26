#!/usr/bin/env python3
"""Table-4 metrics of the full-feature logistic baseline, computed with the code that produced the other rows.

``tools.sac2027_evaluate_controls.analyze`` gives AUC, within-phase and within-match AUC, log loss, Brier, and phase calibration
error for all and for human teams; the native helpers of ``tools.sac2027_native_evaluation`` give winner accuracy and C-index
from ``rank_score``.  The reference GBDT (``HGB-cut``) is recomputed alongside and must reproduce its stored Table-4 inputs.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT

import sys
from pathlib import Path

REPO = REPO_ROOT
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

import pubg_lab.survival.ptr.data.data as D  # noqa: E402
import pubg_lab.survival.ptr.models.hgb as H  # noqa: E402
import pubg_lab.survival.ptr.evaluation.metrics as M  # noqa: E402
from projects.sac2027.evaluation.metrics.sac2027_evaluate_controls import analyze  # noqa: E402
from projects.sac2027.evaluation.metrics.sac2027_native_evaluation import F  # noqa: E402

BASE = REPO / "reports/sac2027_patch_v14"
DUMP = BASE / "dumps/phase_v2"
OUT = BASE / "baseline_logistic"
CHECKED = (
    "auc",
    "auc_within_phase",
    "nll",
    "brier",
    "auc_within_match_phase",
    "phase_calibration_error",
)


def native(frame: pd.DataFrame, mats: dict) -> dict:
    P, Y, V, _, _ = F.padded(frame, frame.rank_score.to_numpy(), mats)
    st = F.per_sample_stats(P, Y, V)
    return dict(
        winner_accuracy=float(st["acc"].mean()), c_index=float(st["conc"].sum() / st["comp"].sum())
    )


def main() -> None:
    roles = pd.read_csv(BASE / "cohort_counts.csv")
    cohorts = roles.loc[roles["analysis_role"] == "main", "cohort"].tolist()
    stored = pd.read_csv(BASE / "control_evaluation/main_population/arm_metrics.csv")
    stored_native = pd.read_csv(BASE / "native_evaluation/per_cohort.csv")
    records = []
    for c in cohorts:
        dump = D.load_dump(DUMP / c / "test.pt")
        rows = H.build_rows(dump, "HGB-cut", "v2")
        meta = rows.keys.copy()
        meta["ai"] = rows.X[:, rows.feature_names.index("team_ai_frac")]
        meta["y"] = rows.y
        paths = {
            "Logistic": [OUT / c / "pred_test.csv"],
            "HGB-cut": [BASE / "checkpoints/ptr_v2" / c / "HGB-cut/seed42/pred_test.csv"],
        }
        mats = {str(m["match_id"]): m for m in dump["matches"]}
        for pop in ("all", "human"):
            arms, _, _, _ = analyze(
                "logistic_baseline", c, pop, paths, [("Logistic", "HGB-cut")], meta
            )
            for a in arms:
                if a["arm"] == "HGB-cut":
                    ref = stored[
                        (stored.arm == "HGB-cut")
                        & (stored.cohort == c)
                        & (stored.population == pop)
                    ]
                    assert len(ref) == 1 and all(
                        abs(ref.iloc[0][m] - a[m]) < 1e-10 for m in CHECKED
                    ), (c, pop)
                records.append(a)
        for arm, files in paths.items():
            nat = native(M.read_predictions(files[0]), mats)
            if arm == "HGB-cut":
                ref = stored_native[(stored_native.arm == "HGB-cut") & (stored_native.cohort == c)]
                assert len(ref) == 1 and all(
                    abs(ref.iloc[0][m] - v) < 1e-10 for m, v in nat.items()
                ), c
            records.append(dict(cohort=c, population="native", arm=arm, **nat))
        print("DONE", c, flush=True)
    df = pd.DataFrame(records)
    df.to_csv(OUT / "table_metrics_per_cohort.csv", index=False)
    means = df.groupby(["arm", "population"]).mean(numeric_only=True)
    means.to_csv(OUT / "table_metrics_means.csv")
    print(means.T.to_string())


if __name__ == "__main__":
    main()
