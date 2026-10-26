#!/usr/bin/env python3
"""Full-feature regularized logistic baseline on the patch split (phase intercepts + the reference GBDT's 151 inputs).

Same rows, inputs, and split as the reference GBDT (``HGB-cut``): trained on patch 40.2, the L2 strength chosen by
validation NLL on patch 41.1 (training-only standardization), and scored on patch 41.2.  Predictions are written in the
common prediction format so the existing evaluation code can score them.  Answers the review's request for a standard
same-target discrete-time baseline (phase intercept + full-feature regularized logistic).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from model_stgcn.ptr import data as D  # noqa: E402
from model_stgcn.ptr import hgb as H  # noqa: E402
from model_stgcn.ptr import metrics as M  # noqa: E402

BASE = REPO / "reports/sac2027_patch_v14"
DUMP = BASE / "dumps/phase_v2"
OUT = BASE / "baseline_logistic"
GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
PHASES = tuple(range(1, 8))
EPS = 1e-12


def design(rows: H.Rows, mean: np.ndarray, sd: np.ndarray, keep: np.ndarray) -> np.ndarray:
    phase = rows.keys["phase"].to_numpy()
    dummies = np.stack([(phase == k).astype(np.float64) for k in PHASES[1:]], axis=1)  # phase 1 = intercept
    z = (rows.X[:, keep] - mean[keep]) / sd[keep]
    return np.hstack([dummies, z])


def nll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def frame(rows: H.Rows, p: np.ndarray, ref: Path) -> pd.DataFrame:
    out = rows.keys.copy()
    out["y"] = rows.y.astype(int)
    out["p_hazard"] = p
    out["rank_score"] = -p  # higher = better survival, as for the reference GBDT
    meta = pd.read_csv(ref)[M.KEY_COLUMNS + ["final_rank_norm", "top5", "match_type", "y"]]
    joined = out.merge(meta, on=M.KEY_COLUMNS, validate="one_to_one", how="outer", indicator=True, suffixes=("", "_ref"))
    assert bool(joined["_merge"].eq("both").all()) and np.array_equal(joined["y"], joined["y_ref"])
    return joined[M.PRED_COLUMNS]


def run(cohort: str) -> dict:
    rows = {s: H.build_rows(D.load_dump(DUMP / cohort / f"{s}.pt"), "HGB-cut", "v2") for s in ("train", "val", "test")}
    Xtr = rows["train"].X
    mean, sd = Xtr.mean(0), Xtr.std(0)
    keep = sd > EPS
    design_tr, design_va, design_te = (design(rows[s], mean, sd, keep) for s in ("train", "val", "test"))
    ytr, yva = rows["train"].y, rows["val"].y
    scores, models = {}, {}
    for c in GRID:
        model = LogisticRegression(C=c, penalty="l2", solver="lbfgs", max_iter=5000)
        model.fit(design_tr, ytr)
        scores[c] = nll(yva, model.predict_proba(design_va)[:, 1])
        models[c] = model
    best = min(scores, key=scores.get)
    model = models[best]
    out = OUT / cohort
    out.mkdir(parents=True, exist_ok=True)
    ref = BASE / "checkpoints/ptr_v2" / cohort / "HGB-cut" / "seed42"
    frame(rows["val"], model.predict_proba(design_va)[:, 1], ref / "pred_val.csv").to_csv(out / "pred_val.csv", index=False)
    frame(rows["test"], model.predict_proba(design_te)[:, 1], ref / "pred_test.csv").to_csv(out / "pred_test.csv", index=False)
    info = dict(cohort=cohort, n_train=int(len(ytr)), n_features=int(keep.sum()), n_phase_dummies=len(PHASES) - 1,
                grid=list(GRID), val_nll={str(k): v for k, v in scores.items()}, selected_C=best,
                dropped_constant=[n for n, k in zip(rows["train"].feature_names, keep) if not k])
    (out / "selection.json").write_text(json.dumps(info, indent=1))
    return info


def main() -> None:
    roles = pd.read_csv(BASE / "cohort_counts.csv")
    for cohort in roles["cohort"]:
        info = run(cohort)
        print(cohort, "C =", info["selected_C"], "features", info["n_features"], flush=True)


if __name__ == "__main__":
    main()
