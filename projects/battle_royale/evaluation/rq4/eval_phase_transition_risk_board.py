"""Evaluate phase-transition risk and advantage boards from trajectory rows."""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_TRAJ = Path(
    "reports/rq4_dynamic_probability_fluctuation/dynamic_probability_trajectories.csv"
)
DEFAULT_OUT = Path("reports/rq4_dynamic_probability_fluctuation")


def _reciprocal_rank(sorted_labels: np.ndarray) -> float:
    hits = np.flatnonzero(sorted_labels.astype(int) == 1)
    if hits.size == 0:
        return float("nan")
    return float(1.0 / (int(hits[0]) + 1))


def _hit_at(sorted_labels: np.ndarray, k: int) -> float:
    if sorted_labels.size == 0 or int(sorted_labels.sum()) == 0:
        return float("nan")
    return float(sorted_labels[:k].sum() > 0)


def _precision_at(sorted_labels: np.ndarray, k: int) -> float:
    if sorted_labels.size == 0:
        return float("nan")
    top = sorted_labels[: min(k, sorted_labels.size)]
    if top.size == 0:
        return float("nan")
    return float(top.mean())


def _ndcg_at(sorted_relevance: np.ndarray, k: int) -> float:
    rel = sorted_relevance[: min(k, sorted_relevance.size)].astype(float)
    if rel.size == 0:
        return float("nan")
    discount = 1.0 / np.log2(np.arange(2, rel.size + 2))
    dcg = float((rel * discount).sum())
    ideal = np.sort(sorted_relevance.astype(float))[::-1][: rel.size]
    idcg = float((ideal * discount).sum())
    if idcg <= 0.0:
        return float("nan")
    return float(dcg / idcg)


def build_phase_board_rows(traj: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["cohort_slug", "model_variant", "match_id", "phase_slot"]
    for key, phase in traj.groupby(group_cols, sort=False):
        cohort, variant, match_id, phase_slot = key
        observed = phase[
            (phase.get("observed_readout", 1) == 1)
            & phase["hazard_prob"].notna()
            & phase["hazard_true"].notna()
        ].copy()
        advantage = phase[phase["dynamic_win_prob"].notna()].copy()
        if observed.empty or advantage.empty:
            continue

        risk_sorted = observed.sort_values("hazard_prob", ascending=False)
        risk_labels = risk_sorted["hazard_true"].astype(int).to_numpy()
        risk_top = risk_sorted.iloc[0]
        has_elimination = int(risk_labels.sum()) > 0

        adv_sorted = advantage.sort_values("dynamic_win_prob", ascending=False)
        adv_top = adv_sorted.iloc[0]
        topk_adv = adv_sorted.head(top_k)
        topk_rel = adv_sorted["final_top10"].astype(int).to_numpy()

        rows.append(
            {
                "cohort_slug": cohort,
                "model_variant": variant,
                "match_id": match_id,
                "phase_slot": int(phase_slot),
                "at_risk_rows": int(len(observed)),
                "next_phase_eliminations": int(risk_labels.sum()),
                "risk_top_team_id": str(risk_top["team_id"]),
                "risk_top_hazard_prob": float(risk_top["hazard_prob"]),
                "risk_top1_hit": float(int(risk_top["hazard_true"]))
                if has_elimination
                else float("nan"),
                "risk_top3_hit": _hit_at(risk_labels, 3),
                f"risk_top{top_k}_hit": _hit_at(risk_labels, top_k),
                "risk_mrr": _reciprocal_rank(risk_labels),
                "risk_precision_at_3": _precision_at(risk_labels, 3),
                f"risk_precision_at_{top_k}": _precision_at(risk_labels, top_k),
                "adv_top_team_id": str(adv_top["team_id"]),
                "adv_top_survival_share": float(adv_top["dynamic_win_prob"]),
                "adv_top_final_top5": int(adv_top["final_top5"]),
                "adv_top_final_top10": int(adv_top["final_top10"]),
                "adv_top_rank_norm": float(adv_top["final_rank_norm"]),
                f"adv_top{top_k}_contains_top5": float(topk_adv["final_top5"].astype(int).max()),
                f"adv_top{top_k}_contains_top10": float(topk_adv["final_top10"].astype(int).max()),
                f"adv_top{top_k}_top10_precision": float(
                    topk_adv["final_top10"].astype(int).mean()
                ),
                f"adv_ndcg_top{top_k}_top10": _ndcg_at(topk_rel, top_k),
                "leader_changed_from_prev_phase": np.nan,
            }
        )

    board = pd.DataFrame(rows)
    if board.empty:
        return board
    board = board.sort_values(
        ["cohort_slug", "model_variant", "match_id", "phase_slot"]
    ).reset_index(drop=True)
    leader_change = (
        board.groupby(["cohort_slug", "model_variant", "match_id"])["adv_top_team_id"]
        .apply(lambda s: s.astype(str) != s.astype(str).shift())
        .reset_index(level=[0, 1, 2], drop=True)
    )
    board["leader_changed_from_prev_phase"] = leader_change.astype(int)
    first_rows = ~board.duplicated(["cohort_slug", "model_variant", "match_id"])
    board.loc[first_rows, "leader_changed_from_prev_phase"] = 0
    return board


def summarize(board: pd.DataFrame, *, top_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if board.empty:
        return pd.DataFrame(), pd.DataFrame()
    metric_cols = [
        "risk_top1_hit",
        "risk_top3_hit",
        f"risk_top{top_k}_hit",
        "risk_mrr",
        "risk_precision_at_3",
        f"risk_precision_at_{top_k}",
        "adv_top_final_top5",
        "adv_top_final_top10",
        f"adv_top{top_k}_contains_top5",
        f"adv_top{top_k}_contains_top10",
        f"adv_top{top_k}_top10_precision",
        f"adv_ndcg_top{top_k}_top10",
        "leader_changed_from_prev_phase",
    ]
    by_cohort = (
        board.groupby(["cohort_slug", "model_variant"])
        .agg(
            phase_boards=("phase_slot", "count"),
            matches=("match_id", "nunique"),
            mean_at_risk_rows=("at_risk_rows", "mean"),
            mean_next_phase_eliminations=("next_phase_eliminations", "mean"),
            **{col: (col, "mean") for col in metric_cols},
        )
        .reset_index()
    )
    aggregate = (
        by_cohort.groupby("model_variant")
        .agg(
            cohorts=("cohort_slug", "nunique"),
            phase_boards=("phase_boards", "sum"),
            median_risk_top1_hit=("risk_top1_hit", "median"),
            median_risk_top3_hit=("risk_top3_hit", "median"),
            median_risk_mrr=("risk_mrr", "median"),
            median_adv_top_final_top5=("adv_top_final_top5", "median"),
            median_adv_top_final_top10=("adv_top_final_top10", "median"),
            median_adv_top5_contains_top10=(f"adv_top{top_k}_contains_top10", "median"),
            median_adv_ndcg_top5_top10=(f"adv_ndcg_top{top_k}_top10", "median"),
            mean_leader_change_rate=("leader_changed_from_prev_phase", "mean"),
        )
        .reset_index()
    )
    return by_cohort, aggregate


def to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
        else:
            display[col] = display[col].map(lambda v: "" if pd.isna(v) else str(v))
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join("---" for _ in display.columns) + " |",
    ]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_doc(
    path: Path, board: pd.DataFrame, by_cohort: pd.DataFrame, aggregate: pd.DataFrame, *, top_k: int
) -> None:
    lines = [
        "# RQ4 Phase-Transition Risk/Advantage Board Result",
        "",
        "Date: 2026-07-06",
        "Class: `ExperimentResult`",
        "Lifecycle: `analysis_complete_bounded`",
        "",
        "## Purpose",
        "",
        "This artifact evaluates the model as a phase-transition board rather than",
        "only as row-level hazard predictions.",
        "",
        "## Board Outputs",
        "",
        "| Board | Ranking variable | Question |",
        "| --- | --- | --- |",
        "| risk board | `hazard_prob` descending | Which teams are most likely to be eliminated next? |",
        "| advantage board | `dynamic_win_prob` descending | Which teams are in the strongest survival position? |",
        "",
        "## Aggregate",
        "",
        to_markdown_table(aggregate),
        "",
        "## Cohort Summary",
        "",
        to_markdown_table(by_cohort),
        "",
        "## Supported Claim",
        "",
        "```text",
        "The dynamic survival readout can be converted into phase-transition boards",
        "that identify the highest next-phase elimination risk and the strongest",
        "match-local survival advantage at each phase cut.",
        "```",
        "",
        "## Boundary",
        "",
        "This board uses the existing phase-cut trajectory artifact.  It does not",
        "validate a separate 10-second streaming metric.  The risk-board metrics",
        "should be compared against explicit score, health, zone-distance, and",
        "tabular baselines before being treated as a final manuscript result.",
        "",
        "## Metric Notes",
        "",
        "- Risk hit-rate metrics are averaged over boards with at least one next-phase elimination; boards with no next-phase elimination have undefined risk-hit targets.",
        f"- `risk_top{top_k}_hit`: at least one next-phase eliminated team appears in the top-{top_k} risk list.",
        "- `risk_mrr`: reciprocal rank of the first actually eliminated team in the risk board.",
        f"- `adv_top{top_k}_contains_top10`: at least one final top-10 team appears in the top-{top_k} advantage list.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    traj = pd.read_csv(args.trajectories)
    board = build_phase_board_rows(traj, top_k=args.top_k)
    by_cohort, aggregate = summarize(board, top_k=args.top_k)
    board_path = out_dir / "phase_transition_risk_advantage_board.csv"
    by_cohort_path = out_dir / "phase_transition_risk_advantage_by_cohort.csv"
    aggregate_path = out_dir / "phase_transition_risk_advantage_aggregate.csv"
    doc_path = out_dir / "RQ4_PHASE_TRANSITION_RISK_ADVANTAGE_BOARD_RESULT_2026-07-06.md"
    manifest_path = out_dir / "phase_transition_risk_advantage_manifest.json"
    board.to_csv(board_path, index=False)
    by_cohort.to_csv(by_cohort_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    write_doc(doc_path, board, by_cohort, aggregate, top_k=args.top_k)
    manifest = {
        "trajectories": str(args.trajectories.resolve()),
        "phase_boards": int(len(board)),
        "cohorts": int(board["cohort_slug"].nunique()) if not board.empty else 0,
        "matches": int(board["match_id"].nunique()) if not board.empty else 0,
        "top_k": int(args.top_k),
        "outputs": {
            "board": str(board_path),
            "by_cohort": str(by_cohort_path),
            "aggregate": str(aggregate_path),
            "doc": str(doc_path),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, default=DEFAULT_TRAJ)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
