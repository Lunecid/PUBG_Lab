"""Stage ``fit``: Sagong et al. (2025) types, Eq. 4 survival rate and the three classifiers on the PTR rows.

Inputs (``<plan-dir>/<cohort>/``): ``units.parquet`` (stage ``plan``) and ``features.parquet`` (stage ``extract``).

Pipeline (every statistic is fitted on the TRAIN split of the cohort only):

1. ``FightScore = attack + kill + damage + assist`` and ``ActiveScore = distance + density + item + carepackage`` of a
   unit (solo: the player; duo / squad: the team, see ``features.py``), raw unweighted sums (Eq. 1, 2).
2. Type (Eq. 3, Table 5): Hunter 1 = F+ A+, Sniper 2 = F+ A-, Traveler 3 = F- A+, Camper 4 = F- A-, where + means strictly
   above the phase mean over the TRAIN units that are alive in that phase (v1 plan: alive at the start, ``alive_start``; v2
   plan: causally alive at the cut a_k = the dump's ``node_mask``, ``alive_cut``), - means <=.
3. Survival rate (Eq. 4), a type-level lookup over the PRECEDING transition.  ``sr_mode='paper'`` (default, literal):
   ``SR_k[i]`` = share of the train units of type ``i`` present in phase k-1 (alive at its start; v2: causally alive at the
   cut k-1) that are still alive at the start of phase k (v2: causally alive at the cut k), over the train matches that have
   phase k (the paper's "100 hunters in 2.0, 85 survive to 2.5"); a statistic fitted on the TRAIN split only.
   ``sr_mode='label'`` (sensitivity): ``SR_k[i] = 1 - mean(y_{k-1})`` over the train risk-set rows of cut k-1 of type ``i``
   (survival of phase k among the units at risk at the end of k-1).  A row of cut k looks up the rate of its own type
   at k (the paper does not say whether the current or the previous type indexes the rate).  Cut 1 has no preceding
   phase: rate 1.0 (as in the paper for the first phase).  An empty (k, type) cell falls back to the phase-wide rate.
4. The paper's 12 inputs: the 8 features, FightScore, ActiveScore, PlayerType (1..4) and the survival rate.
5. XGBoost / RandomForest / LogisticRegression with library defaults, ``random_state = seed``.  Pooled over cuts 1..7
   (``scope='pooled'``, the PTR protocol) or one model per cut (``scope='per_phase'``, sensitivity only).
   ``hgb`` is a control for the paper headline (not a Sagong model): ``HistGradientBoostingClassifier`` built by
   ``model_stgcn.ptr.hgb.make_classifier`` (max_iter 200, learning_rate 0.06, l2 0.01, early_stopping False) on the same 12
   inputs; its parameters are asserted equal to the ``classifier`` block of the cohort's HGB-cut ``metrics.json``.
   ``hgb-hab`` is the same classifier on the 12 inputs PLUS the 44 zone / habitat columns of the dump (the columns the
   noHab arms remove; stage ``habcols`` writes them to ``hab44.parquet``), joined by (match_id, team_id, phase).
   LogisticRegression gets a train-fitted StandardScaler (the paper names no preprocessing; raw metres / damage do not
   converge under the default ``max_iter``); the scaler is a deviation and is written to metrics.json.

Outputs: ``<out-dir>/<cohort>/SAGONG-{xgb,rf,lr,hgb,hgb-hab}/seed<seed>/{pred_val.csv, pred_test.csv, metrics.json}`` in the PTR
format.  The rows are the PTR risk-set rows (``plan``); they are asserted equal to the existing HGB-cut and M11R
prediction files of the same cohort when those exist.
"""

from __future__ import annotations

from pubg_lab.common.paths import REPO_ROOT

import hashlib
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import pubg_lab.survival.ptr.evaluation.metrics as M
from pubg_lab.survival.ptr.common.util import config_hash as _hash
from pubg_lab.survival.ptr.common.util import git_info, write_json
from pubg_lab.survival.baselines.sagong2025.features import (
    ACTIVE_FEATURES,
    ALIVE_COLUMNS,
    FEATURES,
    FIGHT_FEATURES,
)

logger = logging.getLogger("pubg_lab.survival.baselines.sagong2025.fit")

MODELS: dict[str, str] = {
    "xgb": "SAGONG-xgb",
    "rf": "SAGONG-rf",
    "lr": "SAGONG-lr",
    "hgb": "SAGONG-hgb",
    "hgb-hab": "SAGONG-hgb-hab",
}
HGB_KINDS: tuple[str, ...] = ("hgb", "hgb-hab")  # both use ptr.hgb.make_classifier
TYPE_CODES: dict[str, int] = {"Hunter": 1, "Sniper": 2, "Traveler": 3, "Camper": 4}
INPUT_NAMES: list[str] = [
    "phase_attack_count", "phase_kill_count", "phase_total_damage", "phase_assist_count", "distance",
    "phase_max_density", "phase_item_count", "carepackage_use", "FightScore", "ActiveScore", "PlayerType",
    "survival_rate",
]  # fmt: skip
N_CUTS = 7
_NOISE_WARNINGS = ("encountered in matmul", "The `disp` and `iprint` options")
REF_ARMS = (
    "HGB-cut",
    "M11R",
)  # existing arms whose prediction rows must be identical (when their files exist)
REQUIRED_REF_ARM = "HGB-cut"  # ... and HGB-cut must exist


# --------------------------------------------------------------------------- pure steps (tested)
def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fight"] = df[list(FIGHT_FEATURES)].sum(axis=1)
    df["active"] = df[list(ACTIVE_FEATURES)].sum(axis=1)
    return df


def fit_phase_means(units: pd.DataFrame) -> pd.DataFrame:
    """Per phase mean FightScore / ActiveScore over the given (train, alive-at-start) units."""
    return (
        units.groupby("phase")[["fight", "active"]]
        .mean()
        .rename(columns={"fight": "mean_fight", "active": "mean_active"})
    )


def assign_types(df: pd.DataFrame, means: pd.DataFrame) -> np.ndarray:
    """Eq. 3: 1 Hunter, 2 Sniper, 3 Traveler, 4 Camper from the phase means (strictly greater = high)."""
    mf = df["phase"].map(means["mean_fight"]).to_numpy()
    ma = df["phase"].map(means["mean_active"]).to_numpy()
    hi_f = df["fight"].to_numpy() > mf
    hi_a = df["active"].to_numpy() > ma
    return np.where(hi_f & hi_a, 1, np.where(hi_f & ~hi_a, 2, np.where(~hi_f & hi_a, 3, 4))).astype(
        np.int64
    )


def fit_survival_table(rows: pd.DataFrame, n_cuts: int = N_CUTS) -> tuple[dict, dict]:
    """``sr_mode='label'``: Eq. 4 as the survival of the previous cut's label.  ``rows``: train risk-set rows with
    ``phase``, ``ptype``, ``y``.  Returns ``(table[(k, type)] -> rate, info)``; cut 1 -> 1.0 for every type."""
    table: dict[tuple[int, int], float] = {(1, t): 1.0 for t in TYPE_CODES.values()}
    info = {"fallback_cells": [], "n_by_cell": {}}
    for k in range(2, n_cuts + 1):
        prev = rows[rows["phase"] == k - 1]
        overall = float(1.0 - prev["y"].mean()) if len(prev) else 1.0
        for t in TYPE_CODES.values():
            sel = prev[prev["ptype"] == t]
            info["n_by_cell"][f"{k}:{t}"] = int(len(sel))
            if len(sel):
                table[(k, t)] = float(1.0 - sel["y"].mean())
            else:
                table[(k, t)] = overall
                info["fallback_cells"].append([k, t])
    return table, info


def alive_column(df: pd.DataFrame) -> str:
    """Name of the "unit is in phase k" flag: ``alive_cut`` (v2 plan) or ``alive_start`` (v1 plan)."""
    for name in ALIVE_COLUMNS:
        if name in df.columns:
            return name
    raise KeyError(f"units carry neither of {ALIVE_COLUMNS}")


def fit_survival_table_paper(
    units: pd.DataFrame, phase_exists: set, n_cuts: int = N_CUTS
) -> tuple[dict, dict]:
    """``sr_mode='paper'``: Eq. 4 literally.  ``units``: train unit-phase rows (``match_id, team_id, phase, alive_start | alive_cut,
    ptype``); ``phase_exists``: set of ``(match_id, phase)`` of the train matches.  ``SR_k[i]`` = (units of type ``i`` alive
    at the start of phase k-1 that are alive at the start of phase k) / (units of type ``i`` alive at the start of k-1),
    over the matches that have phase k.  Cut 1 -> 1.0 for every type.  With a v2 plan "alive" is the causal status at the cut."""
    table: dict[tuple[int, int], float] = {(1, t): 1.0 for t in TYPE_CODES.values()}
    info = {"fallback_cells": [], "n_by_cell": {}}
    alive = units[units[alive_column(units)]]
    for k in range(2, n_cuts + 1):
        prev = alive[alive["phase"] == k - 1]
        keep = np.fromiter(
            ((m, k) in phase_exists for m in prev["match_id"]), dtype=bool, count=len(prev)
        )
        prev = prev[keep]
        now = set(
            zip(
                alive.loc[alive["phase"] == k, "match_id"],
                alive.loc[alive["phase"] == k, "team_id"],
            )
        )
        surv = np.fromiter(
            ((m, t) in now for m, t in zip(prev["match_id"], prev["team_id"])),
            dtype=bool,
            count=len(prev),
        )
        overall = float(surv.mean()) if len(prev) else 1.0
        for t in TYPE_CODES.values():
            sel = (prev["ptype"] == t).to_numpy()
            info["n_by_cell"][f"{k}:{t}"] = int(sel.sum())
            if sel.any():
                table[(k, t)] = float(surv[sel].mean())
            else:
                table[(k, t)] = overall
                info["fallback_cells"].append([k, t])
    return table, info


def model_inputs(df: pd.DataFrame, table: dict) -> np.ndarray:
    """The paper's 12 inputs as ``[n, 12]`` float64 (column order = :data:`INPUT_NAMES`)."""
    sr = np.array(
        [table[(int(k), int(t))] for k, t in zip(df["phase"], df["ptype"])], dtype=np.float64
    )
    cols = [df[f].to_numpy(dtype=np.float64) for f in FEATURES]
    return np.column_stack(
        [
            *cols,
            df["fight"].to_numpy(dtype=np.float64),
            df["active"].to_numpy(dtype=np.float64),
            df["ptype"].to_numpy(dtype=np.float64),
            sr,
        ]
    )


# --------------------------------------------------------------------------- models
def make_model(kind: str, seed: int, n_jobs: int):
    if kind == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(random_state=seed, n_jobs=n_jobs)
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(random_state=seed, n_jobs=n_jobs)
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), LogisticRegression(random_state=seed))
    if (
        kind in HGB_KINDS
    ):  # the HGB-cut classifier, imported so that its settings cannot drift from model_stgcn/ptr/hgb.py
        from pubg_lab.survival.ptr.models.hgb import make_classifier

        return make_classifier(seed)
    raise ValueError(f"unknown model '{kind}', known: {sorted(MODELS)}")


def model_params(model) -> dict:
    if hasattr(model, "steps"):
        return {name: est.get_params() for name, est in model.steps}
    return model.get_params()


def _fit_predict(kind: str, seed: int, n_jobs: int, Xtr, ytr, Xs: dict) -> tuple[dict, dict]:
    model = make_model(kind, seed, n_jobs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(Xtr, ytr)
    # numpy 2.x + macOS Accelerate raise spurious 'divide by zero / overflow / invalid value encountered in matmul' on finite
    # inputs (verified: a @ b on finite random arrays), and scipy 1.16 deprecates an L-BFGS option sklearn 1.7 passes
    noise = [w for w in caught if any(t in str(w.message) for t in _NOISE_WARNINGS)]
    info = {
        "warnings": sorted({str(w.message)[:160] for w in caught if w not in noise}),
        "n_ignored_noise_warnings": len(noise),
    }
    last = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(last, "n_iter_"):
        info["n_iter"] = np.asarray(last.n_iter_).tolist()
    return {s: model.predict_proba(X)[:, 1] for s, X in Xs.items()}, {"model": model, **info}


def _unscaled_lr_diagnostic(seed: int, Xtr, ytr, Xs: dict, ys: dict) -> dict:
    """Plain default ``LogisticRegression`` on the raw inputs (no scaler): did lbfgs converge, and its AUC (diagnostic only)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = LogisticRegression(random_state=seed).fit(Xtr, ytr)
    return {
        "converged": not any("failed to converge" in str(w.message) for w in caught),
        "n_iter": np.asarray(model.n_iter_).tolist(),
        "max_iter": int(model.max_iter),
        **{
            f"{s}_auc": float(roc_auc_score(ys[s], model.predict_proba(X)[:, 1]))
            for s, X in Xs.items()
        },
    }


def _importances(model) -> dict | None:
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    imp = getattr(est, "feature_importances_", None)
    return None if imp is None else {n: float(v) for n, v in zip(INPUT_NAMES, imp)}


def _reference_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    """Accuracy / F1 at ``p >= thr`` for reference only (the paper reports them with SURVIVAL as the positive class)."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    yhat = (p >= thr).astype(int)
    y = y.astype(int)
    kw = {"zero_division": 0}
    return {
        "threshold": thr,
        "accuracy": float(accuracy_score(y, yhat)),
        "elimination_positive": {
            "f1": float(f1_score(y, yhat, **kw)),
            "precision": float(precision_score(y, yhat, **kw)),
            "recall": float(recall_score(y, yhat, **kw)),
        },
        "survival_positive": {
            "f1": float(f1_score(1 - y, 1 - yhat, **kw)),
            "precision": float(precision_score(1 - y, 1 - yhat, **kw)),
            "recall": float(recall_score(1 - y, 1 - yhat, **kw)),
        },
    }


def _pred_frame(rows: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    df = rows[["match_id", "team_id", "phase"]].copy()
    df["y"] = rows["y"].to_numpy()
    df["p_hazard"] = p
    df["rank_score"] = -p
    df["final_rank_norm"] = rows["final_rank_norm"].to_numpy()
    df["top5"] = rows["top5"].to_numpy()
    df["match_type"] = rows["match_type"].to_numpy()
    return df


def _code_fingerprint() -> str:
    h = hashlib.sha1()
    for name in ("features", "fit", "plan", "extract", "db", "hab"):
        h.update(((REPO_ROOT / "model_stgcn/sagong2025") / f"{name}.py").read_bytes())
    return h.hexdigest()


def _file_fp(path: Path) -> list:
    st = path.stat()
    return [st.st_size, st.st_mtime_ns]


# --------------------------------------------------------------------------- one cohort
def _check_cache_versions(base: Path) -> tuple[str, dict]:
    """The plan, the extracted features and (if present) the habitat table must come from ONE dump version; absent markers = v1."""
    from pubg_lab.survival.ptr.data.data import check_same_version
    from pubg_lab.survival.baselines.sagong2025.extract import FEATURES_META

    audit = (
        json.loads((base / "plan_audit.json").read_text())
        if (base / "plan_audit.json").is_file()
        else {}
    )
    version = str(audit.get("version", "v1"))
    fmeta = (
        json.loads((base / FEATURES_META).read_text())
        if (base / FEATURES_META).is_file()
        else {"version": "v1"}
    )
    check_same_version(
        str(fmeta.get("version", "v1")),
        version,
        what=f"the Sagong features under {base}",
        against="the plan (units.parquet)",
    )
    if (
        fmeta.get("windows_sha1") not in (None, audit.get("windows_sha1"))
        and audit.get("windows_sha1") is not None
    ):
        raise ValueError(
            f"{base}: features.parquet was extracted for other windows than the plan's (re-run plan and extract --force)"
        )
    if version == "v2":
        expected = hashlib.sha256(
            REPO_ROOT / "src/pubg_lab/survival/baselines/sagong2025/features.py".read_bytes()
        ).hexdigest()
        if fmeta.get("feature_source_sha256") != expected:
            raise ValueError(
                f"{base}: stale or missing feature source fingerprint (re-run extract --force)"
            )
    return version, audit


def prepare(plan_dir, cohort: str, sr_mode: str = "paper") -> tuple[pd.DataFrame, dict]:
    """Merge plan units with the extracted features; scores, types and the survival table fitted on train."""
    if sr_mode not in ("paper", "label"):
        raise ValueError("sr_mode must be 'paper' or 'label'")
    base = Path(plan_dir) / cohort
    units = pd.read_parquet(base / "units.parquet")
    feats = pd.read_parquet(base / "features.parquet")
    version, plan_info = _check_cache_versions(base)
    df = units.merge(
        feats.drop(columns=["n_members"]),
        on=["match_id", "team_id", "phase"],
        how="left",
        indicator=True,
    )
    n_missing = int((df["_merge"] != "both").sum())
    df = df.drop(columns="_merge")
    df[list(FEATURES)] = df[list(FEATURES)].fillna(0.0)
    df["n_obs"] = df["n_obs"].fillna(0.0)
    df = add_scores(df)
    acol = alive_column(df)
    train_alive = df[(df["split"] == "train") & df[acol]]
    means = fit_phase_means(train_alive)
    df["ptype"] = assign_types(df, means)
    train_alive = df[(df["split"] == "train") & df[acol]]  # re-slice: now carries ptype
    train_rows = df[(df["split"] == "train") & df["target"]]
    if sr_mode == "label":
        table, sr_info = fit_survival_table(train_rows)
    else:
        win = pd.read_parquet(base / "windows.parquet")
        win = win[win["split"] == "train"]
        table, sr_info = fit_survival_table_paper(
            df[df["split"] == "train"], set(zip(win["match_id"], win["phase"]))
        )
    info = {
        "dump_version": version,
        "alive_status": (
            "units causally alive at the cut a_k (the dump's node_mask), tokens = query_mask"
            if version == "v2"
            else "units alive at the start of the phase (team_death_s > phase_start_s)"
        ),
        "alive_column": acol,
        "survival_rate_mode": sr_mode,
        "n_unit_phase_rows_without_features": n_missing,
        "phase_means": {
            int(k): {"fight": float(r.mean_fight), "active": float(r.mean_active)}
            for k, r in means.iterrows()
        },
        "type_share_train_alive": {
            int(k): {name: float((g["ptype"] == code).mean()) for name, code in TYPE_CODES.items()}
            for k, g in train_alive.groupby("phase")
        },
        "survival_rate_table": {f"{k}:{t}": v for (k, t), v in sorted(table.items())},
        "survival_rate_fallbacks": sr_info,
        "target_rows_without_position_sample": int((df["target"] & (df["n_obs"] == 0)).sum()),
        "target_rows": int(df["target"].sum()),
        "plan_fingerprint": {n: _file_fp(base / n) for n in ("units.parquet", "features.parquet")},
    }
    return df, {"table": table, "info": info}


def run_cohort(
    plan_dir,
    cohort: str,
    out_dir,
    seed: int = 42,
    n_jobs: int = 2,
    models=("xgb", "rf", "lr", "hgb"),
    scope: str = "pooled",
    runs_ref=None,
    sr_mode: str = "paper",
) -> dict:
    n_jobs = max(
        1, min(int(n_jobs), 2)
    )  # resource cap agreed with the coordinator (also caps OpenMP / BLAS threads)
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=n_jobs):
        return _run_cohort(
            plan_dir, cohort, out_dir, seed, n_jobs, models, scope, runs_ref, sr_mode
        )


def _run_cohort(
    plan_dir,
    cohort: str,
    out_dir,
    seed: int,
    n_jobs: int,
    models,
    scope: str,
    runs_ref,
    sr_mode: str,
) -> dict:
    if scope not in ("pooled", "per_phase"):
        raise ValueError("scope must be 'pooled' or 'per_phase'")
    if (
        (scope != "pooled" or sr_mode != "paper")
        and runs_ref is not None
        and Path(out_dir).resolve() == Path(runs_ref).resolve()
    ):
        raise ValueError(
            "sensitivity runs (per_phase scope or sr_mode='label') must not be written into the main runs directory"
        )
    df, prep = prepare(plan_dir, cohort, sr_mode)
    table, info = prep["table"], prep["info"]
    rows = {
        s: df[(df["split"] == s) & df["target"]]
        .sort_values(["match_id", "team_id", "phase"], kind="mergesort")
        .reset_index(drop=True)
        for s in ("train", "val", "test")
    }
    X = {s: model_inputs(r, table) for s, r in rows.items()}
    XH, hab_names, hab_info = None, None, None
    if (
        "hgb-hab" in models
    ):  # the 12 inputs + the 44 zone / habitat columns of the dump (stage habcols)
        from pubg_lab.survival.baselines.sagong2025.hab import attach_hab

        XH, hab_names, hab_info = attach_hab(
            plan_dir, cohort, rows, X, version=info["dump_version"]
        )
    y_tr = rows["train"]["y"].to_numpy().astype(int)
    if len(np.unique(y_tr)) < 2:
        raise ValueError(f"{cohort}: the train rows hold a single class")
    results = {}
    for kind in models:
        name = MODELS[kind]
        t0 = time.perf_counter()
        Xk = XH if kind == "hgb-hab" else X
        input_names = INPUT_NAMES + hab_names if kind == "hgb-hab" else INPUT_NAMES
        if kind == "hgb-hab" and scope != "pooled":
            raise ValueError("hgb-hab is defined for the pooled scope only")
        if scope == "pooled":
            preds, extra = _fit_predict(
                kind, seed, n_jobs, Xk["train"], y_tr, {"val": Xk["val"], "test": Xk["test"]}
            )
            model_info = {
                "warnings": extra["warnings"],
                **({"n_iter": extra["n_iter"]} if "n_iter" in extra else {}),
            }
            params = model_params(extra["model"])
            imp = _importances(extra["model"])
            if imp is not None:
                model_info["feature_importance"] = imp
            if kind == "lr":
                model_info["diagnostic_unscaled_default_lr"] = _unscaled_lr_diagnostic(
                    seed,
                    X["train"],
                    y_tr,
                    {"val": X["val"], "test": X["test"]},
                    {
                        "val": rows["val"]["y"].to_numpy().astype(int),
                        "test": rows["test"]["y"].to_numpy().astype(int),
                    },
                )
        else:
            preds = {s: np.full(len(rows[s]), np.nan) for s in ("val", "test")}
            model_info = {"per_phase": {}}
            params = model_params(make_model(kind, seed, n_jobs))
            for k in range(1, N_CUTS + 1):
                tr = (rows["train"]["phase"] == k).to_numpy()
                yk = y_tr[tr]
                ok = tr.sum() >= 50 and len(np.unique(yk)) == 2
                masks = {s: (rows[s]["phase"] == k).to_numpy() for s in ("val", "test")}
                if ok:
                    pk, ex = _fit_predict(
                        kind,
                        seed,
                        n_jobs,
                        X["train"][tr],
                        yk,
                        {s: X[s][m] for s, m in masks.items()},
                    )
                else:
                    pk = {
                        s: np.full(
                            int(m.sum()), float(yk.mean()) if len(yk) else float(y_tr.mean())
                        )
                        for s, m in masks.items()
                    }
                    ex = {"warnings": []}
                for s, m in masks.items():
                    preds[s][m] = pk[s]
                model_info["per_phase"][str(k)] = {
                    "n_train": int(tr.sum()),
                    "fitted": bool(ok),
                    "warnings": ex["warnings"],
                }
        fit_sec = time.perf_counter() - t0
        if kind in HGB_KINDS and runs_ref is not None:
            _assert_hgb_settings(params, Path(runs_ref), cohort, seed)
        out = Path(out_dir) / cohort / name / f"seed{seed}"
        res = {
            "variant": name,
            "cohort": cohort,
            "seed": seed,
            "labels": "v2",
            "dump_version": info["dump_version"],
            "label_fields": {"y": "y_hazard_v2", "target": "target_mask_v2"},
            "scope": scope,
            "sr_mode": sr_mode,
            "config_hash": _hash(
                {
                    "kind": "sagong2025",
                    "variant": name,
                    "cohort": cohort,
                    "seed": seed,
                    "scope": scope,
                    "sr_mode": sr_mode,
                    "inputs": input_names,
                    "plan": info["plan_fingerprint"],
                    "hab": None if kind != "hgb-hab" else hab_info["hab44_file"],
                    "code": _code_fingerprint(),
                }
            ),
            "match_types_allowed": ["official", "competitive"],
            "n_features": len(input_names),
            "input_names": input_names,
            "n_train_rows": int(len(y_tr)),
            "classifier": params,
            **model_info,
            "fit_sec": fit_sec,
            "git": git_info(),
            "sagong": info,
            **(
                {"habitat_columns": hab_info, "n_sagong_inputs": len(INPUT_NAMES)}
                if kind == "hgb-hab"
                else {}
            ),
        }
        frames = {split: _pred_frame(rows[split], preds[split]) for split in ("val", "test")}
        if (
            runs_ref is not None
        ):  # assert BEFORE anything is written: the rows are exactly the PTR rows
            for split, frame in frames.items():
                _assert_rows_match(frame, Path(runs_ref), cohort, split)
        for split, frame in frames.items():
            assert np.isfinite(frame["p_hazard"]).all() and frame["p_hazard"].between(0, 1).all()
            M.write_predictions(frame, out / f"pred_{split}.csv")
            res[split] = M.all_metrics(frame)
            res[split]["reference"] = _reference_metrics(
                frame["y"].to_numpy(), frame["p_hazard"].to_numpy()
            )
        write_json(out / "metrics.json", res)
        results[name] = res
        logger.info(
            "%s %s %s seed=%d test AUC=%.4f (%.1fs)",
            cohort,
            name,
            scope,
            seed,
            res["test"]["hazard"]["auc"],
            fit_sec,
        )
    return results


def _assert_hgb_settings(params: dict, runs_ref: Path, cohort: str, seed: int) -> None:
    """The control must use EXACTLY the classifier settings that produced the cohort's HGB-cut run (its ``metrics.json``)."""
    p = runs_ref / cohort / "HGB-cut" / f"seed{seed}" / "metrics.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"{p} not found: the HGB-cut classifier settings cannot be asserted"
        )
    recorded = json.loads(p.read_text(encoding="utf-8"))["classifier"]
    diff = {
        k: (params.get(k), recorded.get(k))
        for k in sorted(set(params) | set(recorded))
        if params.get(k) != recorded.get(k)
    }
    if diff:
        raise AssertionError(
            f"{cohort}: SAGONG-hgb classifier differs from the HGB-cut run (own, recorded): {diff}"
        )


def _assert_rows_match(frame: pd.DataFrame, runs_ref: Path, cohort: str, split: str) -> None:
    """The prediction rows (keys, y, final_rank_norm, top5, match_type) must equal those of the existing arms."""
    checked = []
    for arm in REF_ARMS:
        p = runs_ref / cohort / arm / "seed42" / f"pred_{split}.csv"
        if p.is_file():
            M.align_predictions(frame, M.read_predictions(p))  # raises ValueError on any difference
            checked.append(arm)
    if (
        REQUIRED_REF_ARM not in checked
    ):  # HGB-cut is the reference of the PTR rows: the assertion needs its file
        raise FileNotFoundError(
            f"{cohort}/{split}: {REQUIRED_REF_ARM} has no prediction file under {runs_ref}; "
            "the PTR row equality cannot be asserted"
        )
    logger.info("%s/%s rows identical to %s", cohort, split, checked)
