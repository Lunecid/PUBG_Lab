"""Audit O1 survival-capital features on the OM ladder test splits.

This script does not train or evaluate a model. It checks whether the O1
feature block has real variation and direct univariate association with the
next-phase hazard label on the same cohort-separated test splits used by the
BR OM ladder run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from model_stgcn.v3.cohorts import REQUIRED_COHORTS, CohortSpec
from model_stgcn.v3.factorized_data import fetch_factorized_phase_match
from model_stgcn.v3.spec import O1_SURVIVAL_CAPITAL_FEATURES


DEFAULT_ROOT = Path("model_stgcn/checkpoints/br_om_ladder_required_v1")


@dataclass
class FeatureAuditRow:
    cohort: str
    cohort_slug: str
    map: str
    perspective: str
    team_size: str
    split: str
    feature: str
    n: int
    hazard_rate: float
    mean: float
    std: float
    min: float
    p01: float
    p25: float
    median: float
    p75: float
    p99: float
    max: float
    unique_count_rounded_6: int
    zero_frac: float
    one_frac: float
    finite_frac: float
    mean_survived_next: float
    mean_eliminated_next: float
    eliminated_minus_survived: float
    pearson_hazard: float
    degenerate: bool


@dataclass
class CohortAuditRow:
    cohort: str
    cohort_slug: str
    map: str
    perspective: str
    team_size: str
    split: str
    n_matches: int
    n_rows: int
    hazard_rate: float
    degenerate_feature_count: int
    max_abs_pearson_hazard: float
    strongest_feature: str
    strongest_feature_pearson_hazard: float
    alive_players_end_pearson_hazard: float
    alive_players_end_eliminated_minus_survived: float


def _load_split_ids(path: Path, split: str) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    try:
        ids = payload["splits"][split]
    except KeyError as exc:
        raise ValueError(f"{path} does not contain split {split!r}") from exc
    return list(ids)


def _safe_float(value: float) -> float:
    if value is None:
        return math.nan
    if isinstance(value, np.generic):
        value = value.item()
    return float(value)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return math.nan
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 0.0 or y_std <= 0.0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def _feature_stats(
    *,
    spec: CohortSpec | None,
    split: str,
    feature: str,
    values: np.ndarray,
    labels: np.ndarray,
) -> FeatureAuditRow:
    finite = np.isfinite(values)
    finite_values = values[finite].astype(np.float64, copy=False)
    finite_labels = labels[finite].astype(np.float64, copy=False)
    n = int(finite_values.size)
    if n == 0:
        qs = [math.nan] * 7
        survived_mean = math.nan
        eliminated_mean = math.nan
        diff = math.nan
        corr = math.nan
        unique_count = 0
        mean = std = min_v = max_v = math.nan
        zero_frac = one_frac = hazard_rate = math.nan
    else:
        qs = [float(v) for v in np.quantile(finite_values, [0.01, 0.25, 0.5, 0.75, 0.99])]
        survived = finite_values[finite_labels < 0.5]
        eliminated = finite_values[finite_labels >= 0.5]
        survived_mean = float(np.mean(survived)) if survived.size else math.nan
        eliminated_mean = float(np.mean(eliminated)) if eliminated.size else math.nan
        diff = eliminated_mean - survived_mean if np.isfinite(survived_mean) and np.isfinite(eliminated_mean) else math.nan
        corr = _pearson(finite_values, finite_labels)
        unique_count = int(np.unique(np.round(finite_values, 6)).size)
        mean = float(np.mean(finite_values))
        std = float(np.std(finite_values))
        min_v = float(np.min(finite_values))
        max_v = float(np.max(finite_values))
        zero_frac = float(np.mean(np.isclose(finite_values, 0.0, atol=1e-8)))
        one_frac = float(np.mean(np.isclose(finite_values, 1.0, atol=1e-8)))
        hazard_rate = float(np.mean(finite_labels))
    total_n = int(values.size)
    finite_frac = float(n / total_n) if total_n else math.nan
    degenerate = bool(n == 0 or unique_count <= 1 or (np.isfinite(std) and std < 1e-8))
    return FeatureAuditRow(
        cohort=spec.label if spec else "ALL_REQUIRED_COHORTS",
        cohort_slug=spec.slug if spec else "all_required_cohorts",
        map=spec.map_label if spec else "ALL",
        perspective=spec.perspective if spec else "ALL",
        team_size=spec.team_size if spec else "ALL",
        split=split,
        feature=feature,
        n=n,
        hazard_rate=_safe_float(hazard_rate),
        mean=_safe_float(mean),
        std=_safe_float(std),
        min=_safe_float(min_v),
        p01=_safe_float(qs[0]),
        p25=_safe_float(qs[1]),
        median=_safe_float(qs[2]),
        p75=_safe_float(qs[3]),
        p99=_safe_float(qs[4]),
        max=_safe_float(max_v),
        unique_count_rounded_6=unique_count,
        zero_frac=_safe_float(zero_frac),
        one_frac=_safe_float(one_frac),
        finite_frac=_safe_float(finite_frac),
        mean_survived_next=_safe_float(survived_mean),
        mean_eliminated_next=_safe_float(eliminated_mean),
        eliminated_minus_survived=_safe_float(diff),
        pearson_hazard=_safe_float(corr),
        degenerate=degenerate,
    )


def _cohort_row(
    *,
    spec: CohortSpec,
    split: str,
    split_ids: list[str],
    labels: np.ndarray,
    feature_rows: list[FeatureAuditRow],
) -> CohortAuditRow:
    valid_rows = [row for row in feature_rows if not math.isnan(row.pearson_hazard)]
    if valid_rows:
        strongest = max(valid_rows, key=lambda row: abs(row.pearson_hazard))
        max_abs = abs(strongest.pearson_hazard)
    else:
        strongest = None
        max_abs = math.nan
    alive_end = next(row for row in feature_rows if row.feature == "alive_players_end_norm")
    return CohortAuditRow(
        cohort=spec.label,
        cohort_slug=spec.slug,
        map=spec.map_label,
        perspective=spec.perspective,
        team_size=spec.team_size,
        split=split,
        n_matches=len(split_ids),
        n_rows=int(labels.size),
        hazard_rate=float(np.mean(labels)) if labels.size else math.nan,
        degenerate_feature_count=sum(1 for row in feature_rows if row.degenerate),
        max_abs_pearson_hazard=float(max_abs),
        strongest_feature=strongest.feature if strongest else "",
        strongest_feature_pearson_hazard=float(strongest.pearson_hazard) if strongest else math.nan,
        alive_players_end_pearson_hazard=float(alive_end.pearson_hazard),
        alive_players_end_eliminated_minus_survived=float(alive_end.eliminated_minus_survived),
    )


def _write_csv(path: Path, rows: Iterable[object]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def audit_cohort(root: Path, spec: CohortSpec, split: str) -> tuple[CohortAuditRow, list[FeatureAuditRow], dict[str, np.ndarray], np.ndarray]:
    split_path = root / spec.slug / "split.json"
    split_ids = _load_split_ids(split_path, split)
    feature_values: dict[str, list[np.ndarray]] = {name: [] for name in O1_SURVIVAL_CAPITAL_FEATURES}
    labels: list[np.ndarray] = []
    skipped: list[str] = []

    for match_id in split_ids:
        tensors = fetch_factorized_phase_match(match_id, feature_preset="full")
        if tensors is None:
            skipped.append(match_id)
            continue
        missing = [name for name in O1_SURVIVAL_CAPITAL_FEATURES if name not in tensors.node_feature_names]
        if missing:
            raise RuntimeError(f"{match_id} missing O1 features: {missing}")
        name_to_idx = {name: idx for idx, name in enumerate(tensors.node_feature_names)}
        valid = tensors.target_mask.astype(bool, copy=False)
        y = tensors.y_hazard[valid].astype(np.float32, copy=False)
        labels.append(y)
        X = tensors.X_node[:-1]
        for name in O1_SURVIVAL_CAPITAL_FEATURES:
            feature_values[name].append(X[:, :, name_to_idx[name]][valid].astype(np.float32, copy=False))

    if skipped:
        print(f"warning: {spec.label}: skipped {len(skipped)} matches with unavailable tensors")
    y_all = np.concatenate(labels) if labels else np.array([], dtype=np.float32)
    feature_arrays = {
        name: np.concatenate(parts) if parts else np.array([], dtype=np.float32)
        for name, parts in feature_values.items()
    }
    feature_rows = [
        _feature_stats(spec=spec, split=split, feature=name, values=feature_arrays[name], labels=y_all)
        for name in O1_SURVIVAL_CAPITAL_FEATURES
    ]
    cohort_row = _cohort_row(spec=spec, split=split, split_ids=split_ids, labels=y_all, feature_rows=feature_rows)
    return cohort_row, feature_rows, feature_arrays, y_all


def run(args: argparse.Namespace) -> dict:
    root = Path(args.root)
    all_feature_rows: list[FeatureAuditRow] = []
    cohort_rows: list[CohortAuditRow] = []
    pooled_features: dict[str, list[np.ndarray]] = defaultdict(list)
    pooled_labels: list[np.ndarray] = []

    for spec in REQUIRED_COHORTS:
        print(f"auditing {spec.label} ({spec.slug})")
        cohort_row, feature_rows, feature_arrays, labels = audit_cohort(root, spec, args.split)
        cohort_rows.append(cohort_row)
        all_feature_rows.extend(feature_rows)
        pooled_labels.append(labels)
        for name, values in feature_arrays.items():
            pooled_features[name].append(values)

    y_pooled = np.concatenate(pooled_labels) if pooled_labels else np.array([], dtype=np.float32)
    pooled_rows = [
        _feature_stats(
            spec=None,
            split=args.split,
            feature=name,
            values=np.concatenate(pooled_features[name]) if pooled_features[name] else np.array([], dtype=np.float32),
            labels=y_pooled,
        )
        for name in O1_SURVIVAL_CAPITAL_FEATURES
    ]
    all_feature_rows.extend(pooled_rows)

    by_feature_path = root / "o1_survival_capital_integrity_by_feature.csv"
    by_cohort_path = root / "o1_survival_capital_integrity_by_cohort.csv"
    summary_path = root / "o1_survival_capital_integrity_summary.json"
    _write_csv(by_feature_path, all_feature_rows)
    _write_csv(by_cohort_path, cohort_rows)

    alive_end_rows = [row for row in all_feature_rows if row.feature == "alive_players_end_norm" and row.cohort_slug != "all_required_cohorts"]
    summary = {
        "root": str(root),
        "split": args.split,
        "n_cohorts": len(cohort_rows),
        "n_rows_total": int(sum(row.n_rows for row in cohort_rows)),
        "n_matches_total": int(sum(row.n_matches for row in cohort_rows)),
        "any_degenerate_features_by_cohort": [
            row.cohort for row in cohort_rows if row.degenerate_feature_count > 0
        ],
        "pooled_feature_rows": [asdict(row) for row in pooled_rows],
        "alive_players_end_norm": {
            "positive_corr_cohorts": int(sum(row.pearson_hazard > 0 for row in alive_end_rows if not math.isnan(row.pearson_hazard))),
            "negative_corr_cohorts": int(sum(row.pearson_hazard < 0 for row in alive_end_rows if not math.isnan(row.pearson_hazard))),
            "median_pearson_hazard": float(np.median([row.pearson_hazard for row in alive_end_rows if not math.isnan(row.pearson_hazard)])),
            "median_eliminated_minus_survived": float(np.median([row.eliminated_minus_survived for row in alive_end_rows if not math.isnan(row.eliminated_minus_survived)])),
        },
        "strongest_feature_counts": dict(
            sorted(
                {
                    feature: sum(row.strongest_feature == feature for row in cohort_rows)
                    for feature in O1_SURVIVAL_CAPITAL_FEATURES
                }.items()
            )
        ),
        "outputs": {
            "by_feature_csv": str(by_feature_path),
            "by_cohort_csv": str(by_cohort_path),
            "summary_json": str(summary_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="OM ladder required-cohort run root")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"), help="Split to audit")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["outputs"], indent=2))
