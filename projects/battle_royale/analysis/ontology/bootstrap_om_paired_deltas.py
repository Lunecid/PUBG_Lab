"""Paired match-cluster bootstrap for OM ladder prediction CSVs.

Inputs are prediction CSVs produced by `tools/generate_om_prediction_csvs.py`.
The bootstrap samples test matches with replacement inside each cohort, keeping
all team-phase rows from a sampled match together. Deltas are computed as
`to_preset - from_preset` for adjacent OM steps.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DEFAULT_ROOT = Path("model_stgcn/checkpoints/br_om_ladder_required_v1")
DEFAULT_METRICS = DEFAULT_ROOT / "om_ladder_metrics_all_cohorts.csv"
DEFAULT_PRED_DIR = DEFAULT_ROOT / "predictions"

PRESET_ORDER = (
    "om0_context",
    "om1_survival_capital",
    "om2_movement",
    "om3_pressure_habitat",
    "om4_threat_exposure",
    "om5_relation_structure",
    "om6_spatial_proxy",
)

PHASE_GROUPS = {
    "early_1_2": (1, 2),
    "mid_3_4": (3, 4),
    "late_5_6": (5, 6),
}

MetricFn = Callable[[pd.DataFrame, str], float]


@dataclass
class BootstrapRow:
    cohort: str
    cohort_slug: str
    map: str
    mode: str
    phase_scope: str
    phase_group: str
    phases: str
    step: str
    from_preset: str
    to_preset: str
    metric: str
    higher_is_better: bool
    n_rows: int
    n_matches: int
    point_from: float
    point_to: float
    delta: float
    ci_low: float
    ci_high: float
    n_boot_valid: int
    ci_excludes_zero: bool
    supports_to_better: bool


def _load_metrics_index(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        out.setdefault(row["cohort_slug"], row)
    return out


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def _spearman(y: np.ndarray, p: np.ndarray) -> float:
    if y.size < 2:
        return math.nan
    yr = _rankdata(y)
    pr = _rankdata(p)
    if float(np.std(yr)) == 0.0 or float(np.std(pr)) == 0.0:
        return math.nan
    return float(np.corrcoef(yr, pr)[0, 1])


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    if y.size == 0 or np.unique(y).size < 2:
        return math.nan
    return float(roc_auc_score(y, p))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    if y.size == 0:
        return math.nan
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    if y.size == 0:
        return math.nan
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = y.size
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
    return float(ece if total else math.nan)


def _metric_hazard_auc(df: pd.DataFrame, suffix: str) -> float:
    return _auc(df["hazard_true"].to_numpy(), df[f"hazard_prob_{suffix}"].to_numpy())


def _metric_hazard_brier(df: pd.DataFrame, suffix: str) -> float:
    return _brier(df["hazard_true"].to_numpy(), df[f"hazard_prob_{suffix}"].to_numpy())


def _metric_hazard_ece(df: pd.DataFrame, suffix: str) -> float:
    return _ece(df["hazard_true"].to_numpy(), df[f"hazard_prob_{suffix}"].to_numpy())


def _metric_rank_spearman(df: pd.DataFrame, suffix: str) -> float:
    return _spearman(df["final_rank_norm"].to_numpy(), df[f"rank_pred_norm_{suffix}"].to_numpy())


def _metric_top10_auc(df: pd.DataFrame, suffix: str) -> float:
    return _auc(df["final_top10"].to_numpy(), df[f"top10_prob_{suffix}"].to_numpy())


METRICS: dict[str, tuple[MetricFn, bool]] = {
    "hazard_auc": (_metric_hazard_auc, True),
    "hazard_brier": (_metric_hazard_brier, False),
    "hazard_ece": (_metric_hazard_ece, False),
    "rank_spearman": (_metric_rank_spearman, True),
    "top10_auc": (_metric_top10_auc, True),
}


def _read_prediction(path: Path) -> pd.DataFrame:
    cols = [
        "match_id",
        "team_id",
        "phase_slot",
        "cut_phase",
        "hazard_true",
        "hazard_prob",
        "final_rank_norm",
        "rank_pred_norm",
        "final_top10",
        "top10_prob",
    ]
    df = pd.read_csv(path, usecols=cols)
    df["match_id"] = df["match_id"].astype(str)
    return df


def _paired_frame(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    keys = ["match_id", "team_id", "phase_slot", "cut_phase"]
    lcols = keys + [
        "hazard_true",
        "hazard_prob",
        "final_rank_norm",
        "rank_pred_norm",
        "final_top10",
        "top10_prob",
    ]
    rcols = keys + [
        "hazard_true",
        "hazard_prob",
        "final_rank_norm",
        "rank_pred_norm",
        "final_top10",
        "top10_prob",
    ]
    merged = left[lcols].merge(
        right[rcols], on=keys, suffixes=("_from", "_to"), how="inner", validate="one_to_one"
    )
    for col in ("hazard_true", "final_rank_norm", "final_top10"):
        a = merged[f"{col}_from"]
        b = merged[f"{col}_to"]
        if not np.allclose(a.to_numpy(), b.to_numpy()):
            raise ValueError(f"paired prediction labels differ for {col}")
        merged[col] = a
        merged.drop(columns=[f"{col}_from", f"{col}_to"], inplace=True)
    return merged


def _cluster_indices(groups: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    unique = np.unique(groups)
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    starts = np.searchsorted(sorted_groups, unique, side="left")
    ends = np.searchsorted(sorted_groups, unique, side="right")
    return unique, [order[s:e] for s, e in zip(starts, ends)]


def _bootstrap_delta(
    df: pd.DataFrame,
    metric_fn: MetricFn,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float, float, float, int]:
    point_from = metric_fn(df, "from")
    point_to = metric_fn(df, "to")
    point_delta = (
        point_to - point_from if np.isfinite(point_from) and np.isfinite(point_to) else math.nan
    )
    if not np.isfinite(point_delta):
        return point_from, point_to, point_delta, math.nan, math.nan, 0
    groups = df["match_id"].to_numpy(dtype=str)
    unique, rows_by_match = _cluster_indices(groups)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_boot):
        picks = rng.integers(0, len(unique), len(unique))
        idx = np.concatenate([rows_by_match[i] for i in picks])
        sample = df.iloc[idx]
        try:
            b_from = metric_fn(sample, "from")
            b_to = metric_fn(sample, "to")
            delta = b_to - b_from
        except ValueError:
            continue
        if np.isfinite(delta):
            deltas.append(float(delta))
    if not deltas:
        return point_from, point_to, point_delta, math.nan, math.nan, 0
    lo, hi = np.percentile(np.asarray(deltas), [2.5, 97.5])
    return point_from, point_to, point_delta, float(lo), float(hi), len(deltas)


def _phase_scopes(include_phase7: bool, scope: str) -> list[tuple[str, str, tuple[int, ...]]]:
    scopes: list[tuple[str, str, tuple[int, ...]]] = []
    if scope in ("all", "phase_group"):
        for name, phases in PHASE_GROUPS.items():
            scopes.append(("phase_group", name, phases))
    if scope in ("all", "single_phase"):
        max_phase = 7 if include_phase7 else 6
        for phase in range(1, max_phase + 1):
            scopes.append(("single_phase", f"phase_{phase}", (phase,)))
    return scopes


def _cohort_slugs(pred_dir: Path, requested: set[str] | None) -> list[str]:
    slugs = sorted(path.name for path in pred_dir.iterdir() if path.is_dir())
    if requested:
        slugs = [slug for slug in slugs if slug in requested]
    return slugs


def _step_pairs(requested_steps: str | None) -> list[tuple[str, str]]:
    all_steps = list(zip(PRESET_ORDER[:-1], PRESET_ORDER[1:]))
    if not requested_steps:
        return all_steps
    aliases = {f"om{i}->om{i + 1}": pair for i, pair in enumerate(all_steps)}
    aliases.update({f"{left}->{right}": (left, right) for left, right in all_steps})
    out: list[tuple[str, str]] = []
    for raw in requested_steps.split(","):
        key = raw.strip()
        if not key:
            continue
        try:
            out.append(aliases[key])
        except KeyError as exc:
            valid = ", ".join(sorted(aliases))
            raise SystemExit(f"unknown step {key!r}; valid steps: {valid}") from exc
    return out


def _summarize(rows: list[BootstrapRow]) -> dict:
    out: dict[str, dict[str, dict[str, float | int]]] = {}
    buckets: dict[tuple[str, str, str], list[BootstrapRow]] = defaultdict(list)
    for row in rows:
        if row.phase_scope != "phase_group":
            continue
        buckets[(row.step, row.phase_group, row.metric)].append(row)
    for key, vals in sorted(buckets.items()):
        step, phase_group, metric = key
        out_key = f"{step}|{phase_group}|{metric}"
        deltas = np.asarray([row.delta for row in vals if np.isfinite(row.delta)], dtype=float)
        out[out_key] = {
            "n_cohorts": len(vals),
            "median_delta": float(np.median(deltas)) if deltas.size else math.nan,
            "positive_delta_cohorts": int(
                sum(row.delta > 0 for row in vals if np.isfinite(row.delta))
            ),
            "negative_delta_cohorts": int(
                sum(row.delta < 0 for row in vals if np.isfinite(row.delta))
            ),
            "ci_excludes_zero_cohorts": int(sum(row.ci_excludes_zero for row in vals)),
            "supports_to_better_cohorts": int(sum(row.supports_to_better for row in vals)),
        }
    return out


def run(args: argparse.Namespace) -> dict:
    pred_dir = Path(args.pred_dir)
    metrics_index = _load_metrics_index(Path(args.metrics_csv))
    requested_cohorts = (
        {part.strip() for part in args.cohorts.split(",") if part.strip()} if args.cohorts else None
    )
    metric_names = [part.strip() for part in args.metrics.split(",") if part.strip()]
    unknown_metrics = [name for name in metric_names if name not in METRICS]
    if unknown_metrics:
        raise SystemExit(f"unknown metrics: {unknown_metrics}")
    rows: list[BootstrapRow] = []
    scopes = _phase_scopes(args.include_phase7, args.scope)
    steps = _step_pairs(args.steps)
    for slug in _cohort_slugs(pred_dir, requested_cohorts):
        meta = metrics_index.get(slug)
        if not meta:
            print(f"warning: skipping {slug}; not found in metrics index")
            continue
        print(f"bootstrap {meta['cohort']} ({slug})", flush=True)
        preds = {
            preset: _read_prediction(pred_dir / slug / f"{preset}.pred.csv")
            for preset in PRESET_ORDER
        }
        for from_preset, to_preset in steps:
            paired = _paired_frame(preds[from_preset], preds[to_preset])
            step = f"{from_preset}->{to_preset}"
            print(f"  step {step}", flush=True)
            for phase_scope, phase_group, phases in scopes:
                scoped = paired[paired["cut_phase"].isin(phases)].copy()
                if scoped.empty:
                    continue
                n_matches = int(scoped["match_id"].nunique())
                for metric_name in metric_names:
                    metric_fn, higher_is_better = METRICS[metric_name]
                    point_from, point_to, delta, lo, hi, n_valid = _bootstrap_delta(
                        scoped,
                        metric_fn,
                        n_boot=args.n_boot,
                        seed=args.seed,
                    )
                    ci_excludes_zero = bool(
                        np.isfinite(lo) and np.isfinite(hi) and (lo > 0.0 or hi < 0.0)
                    )
                    if higher_is_better:
                        supports = bool(np.isfinite(lo) and lo > 0.0)
                    else:
                        supports = bool(np.isfinite(hi) and hi < 0.0)
                    rows.append(
                        BootstrapRow(
                            cohort=meta["cohort"],
                            cohort_slug=slug,
                            map=meta["map"],
                            mode=meta["mode"],
                            phase_scope=phase_scope,
                            phase_group=phase_group,
                            phases=",".join(str(p) for p in phases),
                            step=step,
                            from_preset=from_preset,
                            to_preset=to_preset,
                            metric=metric_name,
                            higher_is_better=higher_is_better,
                            n_rows=int(len(scoped)),
                            n_matches=n_matches,
                            point_from=float(point_from),
                            point_to=float(point_to),
                            delta=float(delta),
                            ci_low=float(lo),
                            ci_high=float(hi),
                            n_boot_valid=int(n_valid),
                            ci_excludes_zero=ci_excludes_zero,
                            supports_to_better=supports,
                        )
                    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "om_paired_bootstrap_all.csv"
    group_path = out_dir / "om_paired_bootstrap_phase_groups.csv"
    phase_path = out_dir / "om_paired_bootstrap_single_phases.csv"
    fields = list(asdict(rows[0]).keys()) if rows else []
    with all_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    for path, keep_scope in ((group_path, "phase_group"), (phase_path, "single_phase")):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                if row.phase_scope == keep_scope:
                    writer.writerow(asdict(row))
    summary = {
        "pred_dir": str(pred_dir),
        "metrics_csv": args.metrics_csv,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_rows": len(rows),
        "phase7_included": args.include_phase7,
        "metrics": metric_names,
        "summary_by_step_phase_group_metric": _summarize(rows),
        "outputs": {
            "all_csv": str(all_path),
            "phase_group_csv": str(group_path),
            "single_phase_csv": str(phase_path),
            "summary_json": str(out_dir / "om_paired_bootstrap_summary.json"),
        },
    }
    summary_path = out_dir / "om_paired_bootstrap_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", default=str(DEFAULT_PRED_DIR))
    parser.add_argument("--metrics-csv", default=str(DEFAULT_METRICS))
    parser.add_argument("--out-dir", default=str(DEFAULT_ROOT / "bootstrap"))
    parser.add_argument("--cohorts", default=None, help="comma-separated cohort slugs")
    parser.add_argument(
        "--metrics", default="hazard_auc,rank_spearman,top10_auc,hazard_brier,hazard_ece"
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-phase7", action="store_true")
    parser.add_argument("--scope", choices=("all", "phase_group", "single_phase"), default="all")
    parser.add_argument(
        "--steps",
        default=None,
        help="comma-separated adjacent steps, e.g. om0->om1,om2->om3,om3->om4",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["outputs"], indent=2))
