"""Build dynamic survival / win-probability trajectories from hazard predictions.

This is the manuscript-facing bridge from phase-wise hazard prediction to a
"live" probability view:

    lambda_i(k) = predicted next-phase elimination hazard
    S_i(k)      = product_{j<=k} (1 - lambda_i(j))
    pi_i(k)     = match-local normalized survival share at cut k

The model is dynamic at phase-cut resolution.  It does not claim a 10-second
streaming readout unless the input prediction file itself contains such
readouts.
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
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


DEFAULT_ROOT = Path("model_stgcn/checkpoints/br_rq4_sequence_model_zoo_required_v1")
DEFAULT_OUT = Path("reports/rq4_dynamic_probability_fluctuation")


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.size == 0:
        return np.asarray([], dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks_sorted = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks_sorted[start:end] = (start + 1 + end) / 2.0
        start = end
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _auc(y: pd.Series, p: pd.Series) -> float:
    yv = y.astype(int).to_numpy()
    pv = p.astype(float).to_numpy()
    if yv.size == 0 or np.min(yv) == np.max(yv):
        return float("nan")
    return float(roc_auc_score(yv, pv))


def _ap(y: pd.Series, p: pd.Series) -> float:
    yv = y.astype(int).to_numpy()
    pv = p.astype(float).to_numpy()
    if yv.size == 0 or np.min(yv) == np.max(yv):
        return float("nan")
    return float(average_precision_score(yv, pv))


def _brier(y: pd.Series, p: pd.Series) -> float:
    yv = y.astype(int).to_numpy()
    pv = np.clip(p.astype(float).to_numpy(), 0.0, 1.0)
    if yv.size == 0:
        return float("nan")
    return float(brier_score_loss(yv, pv))


def _prediction_paths(root: Path, model_variant: str) -> list[Path]:
    direct = sorted(root.glob(f"*/{model_variant}/survival_bridge_test.pred.csv"))
    if direct:
        return direct
    tabular = sorted(root.glob("*/snapshot_tabular_zoo/snapshot_tabular_predictions.csv"))
    if tabular:
        return tabular
    return []


def load_predictions(paths: list[Path], model_variant: str | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        cohort = path.parts[-3] if path.parent.name != "snapshot_tabular_zoo" else path.parts[-3]
        df = pd.read_csv(path)
        df["cohort_slug"] = cohort
        df["source_prediction_csv"] = str(path)
        if model_variant and "model_variant" in df.columns:
            df = df[df["model_variant"] == model_variant].copy()
        frames.append(df)
    if not frames:
        raise SystemExit("No prediction CSV rows found.")
    df = pd.concat(frames, ignore_index=True)
    if "phase_idx" in df.columns and "phase_slot" not in df.columns:
        df = df.rename(columns={"phase_idx": "phase_slot"})
    if "cut_phase" not in df.columns:
        df["cut_phase"] = df["phase_slot"]
    needed = [
        "cohort_slug",
        "model_variant",
        "match_id",
        "team_id",
        "team_idx",
        "phase_slot",
        "cut_phase",
        "hazard_true",
        "hazard_prob",
        "final_rank_norm",
        "final_top5",
        "final_top10",
    ]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise SystemExit(f"Prediction CSV missing columns: {missing}")
    for col in [
        "team_id",
        "team_idx",
        "phase_slot",
        "cut_phase",
        "hazard_true",
        "final_top5",
        "final_top10",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["hazard_prob", "final_rank_norm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["match_id", "team_id", "phase_slot", "hazard_prob"]).copy()
    df["phase_slot"] = df["phase_slot"].astype(int)
    df["cut_phase"] = df["cut_phase"].astype(int)
    df["team_id"] = df["team_id"].astype(str)
    return df


def build_trajectories(pred: pd.DataFrame) -> pd.DataFrame:
    keys = ["cohort_slug", "model_variant", "match_id", "team_id"]
    pred = pred.sort_values(keys + ["phase_slot"]).copy()
    pred["observed_readout"] = 1
    if "cumulative_survival_score" not in pred.columns:
        pred["cumulative_survival_score"] = (
            (1.0 - pred["hazard_prob"].clip(0.0, 1.0))
            .groupby([pred[col] for col in keys])
            .cumprod()
        )
    else:
        pred["cumulative_survival_score"] = pd.to_numeric(
            pred["cumulative_survival_score"], errors="coerce"
        )
        missing = pred["cumulative_survival_score"].isna()
        if missing.any():
            pred.loc[missing, "cumulative_survival_score"] = (
                (1.0 - pred.loc[missing, "hazard_prob"].clip(0.0, 1.0))
                .groupby([pred.loc[missing, col] for col in keys])
                .cumprod()
            )

    pred = densify_phase_rows(pred, keys)
    pred = pred.sort_values(keys + ["phase_slot"]).copy()

    group = pred.groupby(keys, sort=False)
    pred["hazard_delta"] = group["hazard_prob"].diff().fillna(0.0)
    pred["survival_delta"] = group["cumulative_survival_score"].diff().fillna(0.0)
    pred["hazard_abs_delta"] = pred["hazard_delta"].abs()
    pred["survival_abs_delta"] = pred["survival_delta"].abs()

    match_phase = ["cohort_slug", "model_variant", "match_id", "phase_slot"]
    denom = (
        pred.groupby(match_phase)["cumulative_survival_score"].transform("sum").replace(0.0, np.nan)
    )
    pred["dynamic_win_prob"] = (pred["cumulative_survival_score"] / denom).fillna(0.0)
    pred["dynamic_rank_score"] = 1.0 - pred["final_rank_norm"].astype(float)
    pred["match_local_probability_rank"] = (
        pred.groupby(match_phase)["dynamic_win_prob"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    pred["win_prob_delta"] = pred.groupby(keys, sort=False)["dynamic_win_prob"].diff().fillna(0.0)
    pred["win_prob_abs_delta"] = pred["win_prob_delta"].abs()

    observed_hazard_delta = pred.loc[pred["observed_readout"] == 1, "hazard_abs_delta"]
    q = observed_hazard_delta.quantile(0.90) if len(observed_hazard_delta) else 0.0
    pred["risk_spike"] = (
        (pred["observed_readout"] == 1) & (pred["hazard_delta"] >= max(0.05, float(q)))
    ).astype(int)
    pred["probability_shock"] = (
        pred["win_prob_abs_delta"] >= pred["win_prob_abs_delta"].quantile(0.90)
    ).astype(int)
    return pred


def densify_phase_rows(pred: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Carry survival forward through missing phase slots inside each match.

    Prediction CSVs only contain scored hazard readouts.  For a dynamic
    match-local probability display, the denominator should not change merely
    because a still-relevant team lacks an explicit readout at one cut.  We
    therefore carry the most recent survival score forward until the team is
    known to be eliminated or the match window ends.
    """

    rows: list[dict[str, Any]] = []
    static_cols = [
        "split",
        "cohort_slug",
        "model_variant",
        "match_id",
        "team_idx",
        "team_id",
        "final_rank_norm",
        "final_top5",
        "final_top10",
        "source_prediction_csv",
    ]
    for _, match in pred.groupby(["cohort_slug", "model_variant", "match_id"], sort=False):
        max_phase = int(match["phase_slot"].max())
        for _, team in match.groupby("team_id", sort=False):
            team = team.sort_values("phase_slot")
            observed = {int(row.phase_slot): row._asdict() for row in team.itertuples(index=False)}
            first_phase = int(team["phase_slot"].min())
            last_phase = int(team["phase_slot"].max())
            last_observed_row: dict[str, Any] | None = None
            for phase in range(first_phase, max_phase + 1):
                if phase in observed:
                    row = dict(observed[phase])
                    row["observed_readout"] = 1
                    rows.append(row)
                    last_observed_row = row
                    continue
                if last_observed_row is None:
                    continue
                if (
                    phase > last_phase
                    and int(float(last_observed_row.get("hazard_true", 0) or 0)) == 1
                ):
                    continue
                row = {col: last_observed_row.get(col) for col in static_cols}
                row.update(
                    {
                        "phase_slot": phase,
                        "cut_phase": phase,
                        "hazard_true": np.nan,
                        "hazard_prob": np.nan,
                        "cumulative_survival_score": float(
                            last_observed_row["cumulative_survival_score"]
                        ),
                        "rank_pred_norm": np.nan,
                        "top5_prob": np.nan,
                        "top10_prob": np.nan,
                        "observed_readout": 0,
                    }
                )
                rows.append(row)
    out = pd.DataFrame(rows)
    for col in pred.columns:
        if col not in out.columns:
            out[col] = np.nan
    return out[pred.columns]


def match_summaries(traj: pd.DataFrame) -> pd.DataFrame:
    keys = ["cohort_slug", "model_variant", "match_id", "team_id"]
    per_team = (
        traj.groupby(keys)
        .agg(
            phases_observed=("phase_slot", "nunique"),
            first_phase=("phase_slot", "min"),
            last_phase=("phase_slot", "max"),
            final_rank_norm=("final_rank_norm", "first"),
            final_rank_score=("dynamic_rank_score", "first"),
            final_top5=("final_top5", "first"),
            final_top10=("final_top10", "first"),
            max_hazard=("hazard_prob", "max"),
            mean_hazard=("hazard_prob", "mean"),
            latest_hazard=("hazard_prob", "last"),
            latest_survival=("cumulative_survival_score", "last"),
            latest_win_prob=("dynamic_win_prob", "last"),
            max_win_prob=("dynamic_win_prob", "max"),
            min_win_prob=("dynamic_win_prob", "min"),
            hazard_volatility=("hazard_abs_delta", "sum"),
            win_prob_volatility=("win_prob_abs_delta", "sum"),
            risk_spikes=("risk_spike", "sum"),
            probability_shocks=("probability_shock", "sum"),
            best_probability_rank=("match_local_probability_rank", "min"),
            latest_probability_rank=("match_local_probability_rank", "last"),
        )
        .reset_index()
    )

    phase_leaders = traj.loc[
        traj.groupby(["cohort_slug", "model_variant", "match_id", "phase_slot"])[
            "dynamic_win_prob"
        ].idxmax(),
        ["cohort_slug", "model_variant", "match_id", "phase_slot", "team_id"],
    ].sort_values(["cohort_slug", "model_variant", "match_id", "phase_slot"])
    leader_changes = (
        phase_leaders.groupby(["cohort_slug", "model_variant", "match_id"])["team_id"]
        .apply(lambda s: int((s.astype(str) != s.astype(str).shift()).sum() - 1))
        .reset_index(name="leader_changes")
    )
    match_level = (
        per_team.groupby(["cohort_slug", "model_variant", "match_id"])
        .agg(
            teams=("team_id", "nunique"),
            mean_team_hazard_volatility=("hazard_volatility", "mean"),
            mean_team_win_prob_volatility=("win_prob_volatility", "mean"),
            max_team_win_prob_volatility=("win_prob_volatility", "max"),
            total_risk_spikes=("risk_spikes", "sum"),
            total_probability_shocks=("probability_shocks", "sum"),
        )
        .reset_index()
        .merge(leader_changes, on=["cohort_slug", "model_variant", "match_id"], how="left")
    )
    match_level["leader_changes"] = match_level["leader_changes"].fillna(0).astype(int)
    return per_team, match_level


def model_metrics(
    traj: pd.DataFrame, per_team: pd.DataFrame, match_level: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["cohort_slug", "model_variant"]
    for key, sub in traj.groupby(group_cols):
        cohort, variant = key
        observed = sub[sub["observed_readout"] == 1].copy()
        latest = (
            sub.sort_values("phase_slot").groupby(["match_id", "team_id"], as_index=False).tail(1)
        )
        team_sub = per_team[
            (per_team["cohort_slug"] == cohort) & (per_team["model_variant"] == variant)
        ]
        match_sub = match_level[
            (match_level["cohort_slug"] == cohort) & (match_level["model_variant"] == variant)
        ]
        rows.append(
            {
                "cohort_slug": cohort,
                "model_variant": variant,
                "trajectory_rows": int(len(sub)),
                "observed_prediction_rows": int(len(observed)),
                "matches": int(sub["match_id"].nunique()),
                "teams": int(sub[["match_id", "team_id"]].drop_duplicates().shape[0]),
                "hazard_auc_all_cuts": _auc(observed["hazard_true"], observed["hazard_prob"]),
                "hazard_ap_all_cuts": _ap(observed["hazard_true"], observed["hazard_prob"]),
                "hazard_brier_all_cuts": _brier(observed["hazard_true"], observed["hazard_prob"]),
                "latest_top5_auc_from_win_prob": _auc(
                    latest["final_top5"],
                    latest["latest_win_prob"]
                    if "latest_win_prob" in latest
                    else latest["dynamic_win_prob"],
                ),
                "latest_top10_auc_from_win_prob": _auc(
                    latest["final_top10"], latest["dynamic_win_prob"]
                ),
                "latest_rank_spearman_score": _spearman(
                    latest["dynamic_rank_score"].to_numpy(), latest["dynamic_win_prob"].to_numpy()
                ),
                "median_team_hazard_volatility": float(team_sub["hazard_volatility"].median())
                if len(team_sub)
                else float("nan"),
                "median_team_win_prob_volatility": float(team_sub["win_prob_volatility"].median())
                if len(team_sub)
                else float("nan"),
                "mean_leader_changes_per_match": float(match_sub["leader_changes"].mean())
                if len(match_sub)
                else float("nan"),
                "median_leader_changes_per_match": float(match_sub["leader_changes"].median())
                if len(match_sub)
                else float("nan"),
                "mean_probability_shocks_per_match": float(
                    match_sub["total_probability_shocks"].mean()
                )
                if len(match_sub)
                else float("nan"),
                "mean_risk_spikes_per_match": float(match_sub["total_risk_spikes"].mean())
                if len(match_sub)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def choose_example_match(match_level: pd.DataFrame, cohort: str | None) -> tuple[str, str, str]:
    candidates = match_level.copy()
    if cohort:
        candidates = candidates[candidates["cohort_slug"] == cohort].copy()
    if candidates.empty:
        candidates = match_level.copy()
    candidates["score"] = (
        candidates["leader_changes"].astype(float) * 0.50
        + candidates["max_team_win_prob_volatility"].astype(float) * 4.00
        + candidates["total_probability_shocks"].astype(float) * 0.05
    )
    row = candidates.sort_values("score", ascending=False).iloc[0]
    return str(row["cohort_slug"]), str(row["model_variant"]), str(row["match_id"])


def plot_match(
    traj: pd.DataFrame, cohort: str, variant: str, match_id: str, out: Path, top_n: int
) -> None:
    sub = traj[
        (traj["cohort_slug"] == cohort)
        & (traj["model_variant"] == variant)
        & (traj["match_id"] == match_id)
    ].copy()
    if sub.empty:
        return
    team_order = (
        sub.sort_values("phase_slot")
        .groupby("team_id")
        .tail(1)
        .sort_values(["dynamic_win_prob", "dynamic_rank_score"], ascending=False)
        .head(top_n)["team_id"]
        .astype(str)
        .tolist()
    )
    plot_df = sub[sub["team_id"].astype(str).isin(team_order)].copy()
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(team_order))))
    for color, team_id in zip(colors, team_order):
        team = plot_df[plot_df["team_id"].astype(str) == team_id].sort_values("phase_slot")
        label = f"Team {team_id}"
        axes[0].plot(
            team["cut_phase"],
            team["dynamic_win_prob"],
            marker="o",
            linewidth=2.0,
            color=color,
            label=label,
        )
        axes[1].plot(
            team["cut_phase"],
            team["hazard_prob"],
            marker="o",
            linewidth=1.8,
            color=color,
            label=label,
        )
        shocks = team[team["probability_shock"] == 1]
        if not shocks.empty:
            axes[0].scatter(
                shocks["cut_phase"],
                shocks["dynamic_win_prob"],
                s=60,
                color=color,
                edgecolor="black",
                zorder=5,
            )
        spikes = team[team["risk_spike"] == 1]
        if not spikes.empty:
            axes[1].scatter(
                spikes["cut_phase"],
                spikes["hazard_prob"],
                s=60,
                color=color,
                edgecolor="black",
                zorder=5,
            )

    axes[0].set_ylabel("Match-local survival share")
    axes[1].set_ylabel("Next-phase elimination hazard")
    axes[1].set_xlabel("Phase cut")
    readable_cohort = cohort.replace("_", " ")
    axes[0].set_title(f"Dynamic survival-score trajectory in {readable_cohort}")
    for ax in axes:
        ax.grid(True, alpha=0.22)
        ax.set_ylim(bottom=0.0)
    axes[0].legend(loc="upper right", ncols=2, fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
        else:
            display[col] = display[col].map(lambda v: "" if pd.isna(v) else str(v))
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_summary_doc(
    out: Path,
    root: Path,
    model_variant: str,
    metrics: pd.DataFrame,
    example: tuple[str, str, str],
    figure_path: Path,
) -> None:
    aggregate = (
        metrics.groupby("model_variant")
        .agg(
            cohorts=("cohort_slug", "nunique"),
            median_hazard_auc=("hazard_auc_all_cuts", "median"),
            mean_hazard_auc=("hazard_auc_all_cuts", "mean"),
            median_latest_top5_auc=("latest_top5_auc_from_win_prob", "median"),
            median_latest_top10_auc=("latest_top10_auc_from_win_prob", "median"),
            median_latest_rank_spearman=("latest_rank_spearman_score", "median"),
            median_team_win_prob_volatility=("median_team_win_prob_volatility", "median"),
            mean_leader_changes_per_match=("mean_leader_changes_per_match", "mean"),
        )
        .reset_index()
    )
    lines = [
        "# Battle Royale RQ4 Dynamic Probability Fluctuation Result",
        "",
        "Date: 2026-07-06",
        "Class: `ExperimentResult`",
        "Lifecycle: `analysis_complete_bounded`",
        "",
        "## Purpose",
        "",
        "This artifact reframes the existing phase-wise hazard predictions as a",
        "dynamic probability model.  It is intended to show what Sagong-style",
        "phase score prediction does not directly show: the probability assigned",
        "to each team changes as the match progresses.",
        "",
        "## Model Definition",
        "",
        "```text",
        "lambda_i(k) = predicted next-phase elimination hazard",
        "S_i(k)      = product_{j <= k} (1 - lambda_i(j))",
        "pi_i(k)     = S_i(k) / sum_j S_j(k) within the same match and cut phase",
        "```",
        "",
        "`pi_i(k)` is a match-local dynamic win-probability proxy, not a fully",
        "calibrated bookmaker-style win probability.",
        "",
        "## Source",
        "",
        f"- run root: `{root}`",
        f"- model variant requested: `{model_variant}`",
        "",
        "## Aggregate",
        "",
        to_markdown_table(aggregate),
        "",
        "## Example Figure",
        "",
        f"- `{figure_path}`",
        f"- selected example: cohort `{example[0]}`, model `{example[1]}`, match `{example[2]}`",
        "",
        "## Supported Claim",
        "",
        "```text",
        "The proposed survival formulation can be rendered as a dynamic probability",
        "trajectory: at each observed phase cut, the model updates next-phase",
        "elimination hazard, cumulative survival, and match-local win-probability",
        "share for every still-at-risk team.",
        "```",
        "",
        "This directly separates the thesis from a static phase-score baseline: the",
        "output is not only an eventual survival/win label, but a changing",
        "probability curve over the course of the match.",
        "",
        "## Boundary",
        "",
        "Do not claim strict 10-second real-time deployment yet.  The current",
        "artifact is dynamic at phase-cut resolution because the stored prediction",
        "rows are phase-cut readouts.  A 10-second streaming version requires an",
        "additional prefix-readout evaluation pass over snapshot sequences.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = _prediction_paths(root, args.model_variant)
    if not paths:
        raise SystemExit(f"No prediction files found for {args.model_variant} under {root}")
    pred = load_predictions(paths, args.model_variant)
    traj = build_trajectories(pred)
    per_team, match_level = match_summaries(traj)
    metrics = model_metrics(traj, per_team, match_level)

    traj_path = out_dir / "dynamic_probability_trajectories.csv"
    team_path = out_dir / "dynamic_probability_team_summary.csv"
    match_path = out_dir / "dynamic_probability_match_summary.csv"
    metrics_path = out_dir / "dynamic_probability_metrics_by_cohort.csv"
    traj.to_csv(traj_path, index=False)
    per_team.to_csv(team_path, index=False)
    match_level.to_csv(match_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    example = choose_example_match(match_level, args.example_cohort)
    fig_path = out_dir / "dynamic_probability_example_match.png"
    plot_match(traj, *example, out=fig_path, top_n=args.top_n)
    doc_path = out_dir / "BATTLE_ROYALE_RQ4_DYNAMIC_PROBABILITY_FLUCTUATION_RESULT_2026-07-06.md"
    write_summary_doc(doc_path, root, args.model_variant, metrics, example, fig_path)

    manifest = {
        "root": str(root),
        "model_variant": args.model_variant,
        "prediction_files": len(paths),
        "trajectory_rows": int(len(traj)),
        "observed_prediction_rows": int(traj["observed_readout"].sum()),
        "cohorts": int(traj["cohort_slug"].nunique()),
        "matches": int(traj["match_id"].nunique()),
        "teams": int(traj[["match_id", "team_id"]].drop_duplicates().shape[0]),
        "outputs": {
            "trajectories": str(traj_path),
            "team_summary": str(team_path),
            "match_summary": str(match_path),
            "metrics_by_cohort": str(metrics_path),
            "example_figure": str(fig_path),
            "result_doc": str(doc_path),
        },
        "example": {
            "cohort_slug": example[0],
            "model_variant": example[1],
            "match_id": example[2],
        },
    }
    manifest_path = out_dir / "dynamic_probability_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--model-variant", default="snapshot_lstm_no_graph")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--example-cohort", default="erangel_tpp_squad")
    parser.add_argument("--top-n", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
