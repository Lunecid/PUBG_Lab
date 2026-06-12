"""
Basic graph-network analysis for the intermediate PUBG team network dataset.

This script reads the CSV files produced by build_analysis_dataset.py and writes
reproducible summary artifacts:
  - basic_graph_report.md
  - analysis_summary.json
  - degree_distribution.csv
  - centrality_correlations.csv
  - metric_quartiles.csv
  - position_rank_summary.csv
  - figures/*.png

Example:
  python3 -m network_analysis.basic_graph_analysis \
      --dataset network_analysis/output/analysis_datasets/Baltic_Main_squad_n100
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


DEFAULT_DATASET = os.path.join(
    "network_analysis", "output", "analysis_datasets", "Baltic_Main_squad_n100"
)

CENTRALITY_METRICS = [
    ("degree", "Degree"),
    ("weighted_deg", "Weighted degree"),
    ("c_closeness", "Closeness"),
    ("c_betweenness", "Betweenness"),
    ("c_eigenvector", "Eigenvector"),
    ("c_pagerank", "PageRank"),
]

EARLY_METRICS = [
    ("early_degree", "Early degree"),
    ("early_weighted_deg", "Early weighted degree"),
    ("early_c_betweenness", "Early betweenness"),
    ("early_c_pagerank", "Early PageRank"),
]


def _as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [_clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return _clean_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _summary(values: pd.Series) -> dict[str, float | int | None]:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return {
            "n": 0, "mean": None, "median": None, "std": None,
            "min": None, "q25": None, "q75": None, "max": None,
        }
    return {
        "n": int(vals.shape[0]),
        "mean": float(vals.mean()),
        "median": float(vals.median()),
        "std": float(vals.std(ddof=0)),
        "min": float(vals.min()),
        "q25": float(vals.quantile(0.25)),
        "q75": float(vals.quantile(0.75)),
        "max": float(vals.max()),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(v):
        return "-"
    return f"{v:.{digits}f}"


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float, int]:
    xy = pd.DataFrame({"x": _as_float(x), "y": _as_float(y)}).dropna()
    n = int(xy.shape[0])
    if n < 4 or xy["x"].std(ddof=0) < 1e-12 or xy["y"].std(ddof=0) < 1e-12:
        return float("nan"), float("nan"), n
    if method == "pearson":
        r, p = pearsonr(xy["x"], xy["y"])
    elif method == "spearman":
        res = spearmanr(xy["x"], xy["y"])
        r, p = res.correlation, res.pvalue
    else:
        raise ValueError(method)
    return float(r), float(p), n


def _partial_corr(x: pd.Series, y: pd.Series, z: pd.Series) -> tuple[float, int]:
    xyz = pd.DataFrame({
        "x": _as_float(x),
        "y": _as_float(y),
        "z": _as_float(z),
    }).dropna()
    n = int(xyz.shape[0])
    if n < 5:
        return float("nan"), n
    if xyz["x"].std(ddof=0) < 1e-12 or xyz["y"].std(ddof=0) < 1e-12:
        return float("nan"), n
    z_arr = xyz["z"].to_numpy(dtype=float)
    design = np.column_stack([np.ones_like(z_arr), z_arr])
    try:
        bx = np.linalg.lstsq(design, xyz["x"].to_numpy(dtype=float), rcond=None)[0]
        by = np.linalg.lstsq(design, xyz["y"].to_numpy(dtype=float), rcond=None)[0]
    except np.linalg.LinAlgError:
        return float("nan"), n
    rx = xyz["x"].to_numpy(dtype=float) - design @ bx
    ry = xyz["y"].to_numpy(dtype=float) - design @ by
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan"), n
    return float(np.corrcoef(rx, ry)[0, 1]), n


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean_json(payload), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def _load_dataset(dataset_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    matches = pd.read_csv(os.path.join(dataset_dir, "matches.csv"))
    teams = pd.read_csv(os.path.join(dataset_dir, "teams.csv"))
    edges = pd.read_csv(os.path.join(dataset_dir, "team_edges.csv"))
    meta_path = os.path.join(dataset_dir, "metadata.json")
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
    return matches, teams, edges, metadata


def _degree_distribution(teams: pd.DataFrame) -> pd.DataFrame:
    deg = _as_float(teams["degree"]).fillna(0).astype(int)
    counts = deg.value_counts().sort_index()
    out = pd.DataFrame({
        "degree": counts.index.astype(int),
        "count": counts.values.astype(int),
    })
    out["probability"] = out["count"] / out["count"].sum()
    return out


def _centrality_correlations(teams: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = CENTRALITY_METRICS + EARLY_METRICS + [
        ("combat_events", "Combat events"),
        ("dmg_dealt", "Damage dealt"),
        ("dmg_taken", "Damage taken"),
        ("displacement_per_min", "Displacement/min"),
        ("battle_center_score", "Battle center score"),
        ("combat_participation", "Combat participation"),
    ]
    for col, label in metrics:
        if col not in teams.columns:
            continue
        pearson_r, pearson_p, n = _safe_corr(teams[col], teams["final_rank"], "pearson")
        spearman_r, spearman_p, _ = _safe_corr(teams[col], teams["final_rank"], "spearman")
        partial_r, partial_n = _partial_corr(
            teams[col], teams["final_rank"], teams["survival_seconds"]
        )
        rows.append({
            "metric": col,
            "label": label,
            "n": n,
            "pearson_r_vs_rank": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r_vs_rank": spearman_r,
            "spearman_p": spearman_p,
            "partial_r_vs_rank_given_survival": partial_r,
            "partial_n": partial_n,
            "direction_note": "negative means higher metric is associated with better rank",
        })
    return pd.DataFrame(rows)


def _metric_quartiles(teams: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = CENTRALITY_METRICS + [
        ("battle_center_score", "Battle center score"),
        ("combat_participation", "Combat participation"),
        ("displacement_per_min", "Displacement/min"),
    ]
    for col, label in metrics:
        if col not in teams.columns:
            continue
        vals = _as_float(teams[col])
        valid = teams.loc[vals.notna()].copy()
        valid["_metric"] = vals.loc[vals.notna()]
        if valid.shape[0] < 8 or valid["_metric"].std(ddof=0) < 1e-12:
            continue
        q25 = float(valid["_metric"].quantile(0.25))
        q75 = float(valid["_metric"].quantile(0.75))
        low = valid[valid["_metric"] <= q25]
        high = valid[valid["_metric"] >= q75]
        rows.append({
            "metric": col,
            "label": label,
            "q25": q25,
            "q75": q75,
            "low_n": int(low.shape[0]),
            "high_n": int(high.shape[0]),
            "low_mean_rank": float(low["final_rank"].mean()),
            "high_mean_rank": float(high["final_rank"].mean()),
            "delta_high_minus_low": float(high["final_rank"].mean() - low["final_rank"].mean()),
            "high_mean_survival_s": float(high["survival_seconds"].mean()),
            "low_mean_survival_s": float(low["survival_seconds"].mean()),
        })
    return pd.DataFrame(rows)


def _position_rank_summary(teams: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if "position_label" not in teams.columns:
        return pd.DataFrame(rows)
    for label, group in teams.groupby("position_label", dropna=False):
        rows.append({
            "position_label": str(label),
            "n": int(group.shape[0]),
            "mean_rank": float(group["final_rank"].mean()),
            "median_rank": float(group["final_rank"].median()),
            "mean_degree": float(group["degree"].mean()),
            "mean_betweenness": float(group["c_betweenness"].mean()),
            "mean_weighted_degree": float(group["weighted_deg"].mean()),
            "mean_survival_s": float(group["survival_seconds"].mean()),
        })
    return pd.DataFrame(rows).sort_values("mean_rank")


def _match_summaries(matches: pd.DataFrame) -> dict[str, Any]:
    keys = [
        "full_n_edges", "full_density", "full_avg_degree",
        "full_avg_clustering", "full_n_components", "full_giant_size",
        "full_diameter", "full_avg_path_length", "damage_events",
        "damage_total", "early_n_edges", "early_damage_total",
    ]
    out = {k: _summary(matches[k]) for k in keys if k in matches.columns}
    if {"early_n_edges", "full_n_edges"}.issubset(matches.columns):
        ratio = _as_float(matches["early_n_edges"]) / _as_float(matches["full_n_edges"]).replace(0, np.nan)
        out["early_edge_ratio"] = _summary(ratio)
    if {"early_damage_total", "damage_total"}.issubset(matches.columns):
        ratio = _as_float(matches["early_damage_total"]) / _as_float(matches["damage_total"]).replace(0, np.nan)
        out["early_damage_ratio"] = _summary(ratio)
    if {"full_giant_size", "n_teams"}.issubset(matches.columns):
        ratio = _as_float(matches["full_giant_size"]) / _as_float(matches["n_teams"]).replace(0, np.nan)
        out["giant_team_ratio"] = _summary(ratio)
    return out


def _community_summary(teams: pd.DataFrame) -> dict[str, Any]:
    by_match = (
        teams.drop_duplicates("match_id")
        if "match_id" in teams.columns else pd.DataFrame()
    )
    out: dict[str, Any] = {}
    if "match_modularity_q" in by_match.columns:
        out["match_modularity_q"] = _summary(by_match["match_modularity_q"])
    if "community_size" in teams.columns:
        out["community_size"] = _summary(teams["community_size"])
    if "community_density" in teams.columns:
        out["community_density"] = _summary(teams["community_density"])
    if "position_label" in teams.columns:
        out["position_counts"] = teams["position_label"].fillna("unknown").value_counts().to_dict()
    if "archetype" in teams.columns:
        out["archetype_counts"] = teams["archetype"].fillna("unknown").value_counts().to_dict()
    if "movement_label" in teams.columns:
        out["movement_counts"] = teams["movement_label"].fillna("unknown").value_counts().to_dict()
    return out


def _top_rows(teams: pd.DataFrame, metric: str, n: int = 10) -> list[dict[str, Any]]:
    cols = [
        "match_id", "team_idx", "team_id", "final_rank", "degree",
        "weighted_deg", "c_betweenness", "c_pagerank", "community_id",
        "position_label", metric,
    ]
    cols = list(dict.fromkeys([c for c in cols if c in teams.columns]))
    out = teams.sort_values(metric, ascending=False).head(n)[cols].copy()
    if "match_id" in out.columns:
        out["match_id"] = out["match_id"].astype(str).str.slice(0, 8)
    return out.to_dict(orient="records")


def _plot_degree_distribution(df: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    ax.bar(df["degree"], df["probability"], color="#3b82f6", width=0.82)
    ax.set_title("Team Degree Distribution")
    ax.set_xlabel("degree k")
    ax.set_ylabel("P(k)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_match_properties(matches: pd.DataFrame, path: str) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), dpi=160)
    specs = [
        ("full_density", "Density"),
        ("full_avg_clustering", "Avg clustering"),
        ("full_avg_path_length", "Avg path length"),
        ("full_n_edges", "Edges"),
    ]
    for ax, (col, title) in zip(axs.flat, specs):
        ax.hist(_as_float(matches[col]).dropna(), bins=18, color="#0f766e", alpha=0.85)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Match-Level Network Properties", y=1.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_centrality_rank(teams: pd.DataFrame, path: str) -> None:
    fig, axs = plt.subplots(2, 3, figsize=(13, 7), dpi=160)
    for ax, (col, label) in zip(axs.flat, CENTRALITY_METRICS):
        sample = teams[[col, "final_rank"]].dropna()
        if sample.shape[0] > 1200:
            sample = sample.sample(1200, random_state=0)
        ax.scatter(sample[col], sample["final_rank"], s=8, alpha=0.35, color="#475569")
        ax.set_title(label)
        ax.set_xlabel(col)
        ax.set_ylabel("rank (lower is better)")
        ax.invert_yaxis()
        ax.grid(alpha=0.2)
    fig.suptitle("Centrality vs Final Rank", y=1.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_early_full(matches: pd.DataFrame, path: str) -> None:
    edge_ratio = _as_float(matches["early_n_edges"]) / _as_float(matches["full_n_edges"]).replace(0, np.nan)
    dmg_ratio = _as_float(matches["early_damage_total"]) / _as_float(matches["damage_total"]).replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    ax.scatter(edge_ratio, dmg_ratio, s=28, alpha=0.72, color="#b45309")
    ax.set_title("Early 300s Share of Full-Match Combat")
    ax.set_xlabel("early edges / full edges")
    ax.set_ylabel("early damage / full damage")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _report_markdown(
    *,
    dataset_dir: str,
    out_dir: str,
    metadata: dict,
    match_summary: dict,
    community_summary: dict,
    degree_dist: pd.DataFrame,
    correlations: pd.DataFrame,
    quartiles: pd.DataFrame,
    position_summary: pd.DataFrame,
    top_tables: dict[str, list[dict[str, Any]]],
) -> str:
    n_matches = int(metadata.get("n_matches") or match_summary["full_n_edges"]["n"])
    n_teams = int(metadata.get("n_teams") or degree_dist["count"].sum())
    lines: list[str] = []
    lines.append("# Basic Graph Network Analysis")
    lines.append("")
    lines.append(f"- Dataset: `{dataset_dir}`")
    lines.append(f"- Output: `{out_dir}`")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Scope: {n_matches} matches / {n_teams} teams")
    lines.append("- Graph: team-level weighted undirected combat network")
    lines.append("- Node = team, edge = damage interaction, weight = cumulative damage")
    lines.append("")
    lines.append("## 1. Network Properties")
    props = [
        ("full_n_edges", "edges"),
        ("full_density", "density"),
        ("full_avg_degree", "avg degree"),
        ("full_avg_clustering", "avg clustering"),
        ("full_giant_size", "giant size"),
        ("giant_team_ratio", "giant/team ratio"),
        ("full_diameter", "diameter"),
        ("full_avg_path_length", "avg path length"),
    ]
    lines.append("| metric | mean | median | q25 | q75 | min | max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for key, label in props:
        s = match_summary.get(key, {})
        lines.append(
            f"| {label} | {_fmt(s.get('mean'))} | {_fmt(s.get('median'))} | "
            f"{_fmt(s.get('q25'))} | {_fmt(s.get('q75'))} | "
            f"{_fmt(s.get('min'))} | {_fmt(s.get('max'))} |"
        )
    lines.append("")
    lines.append("Interpretation: density is low, but the giant component usually covers most teams. "
                 "This means PUBG squad combat is sparse at pair level, yet most active teams become connected through a few steps.")
    lines.append("")
    lines.append("## 2. Degree Distribution")
    top_degrees = degree_dist.sort_values("count", ascending=False).head(8)
    lines.append("| degree k | count | P(k) |")
    lines.append("|---:|---:|---:|")
    for _, row in top_degrees.iterrows():
        lines.append(f"| {int(row['degree'])} | {int(row['count'])} | {_fmt(row['probability'])} |")
    lines.append("")
    lines.append("## 3. Centrality vs Rank")
    show = correlations[
        correlations["metric"].isin([m[0] for m in CENTRALITY_METRICS] + ["battle_center_score"])
    ].copy()
    lines.append("| metric | Pearson r | Spearman rho | partial r | n |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, row in show.iterrows():
        lines.append(
            f"| {row['label']} | {_fmt(row['pearson_r_vs_rank'])} | "
            f"{_fmt(row['spearman_r_vs_rank'])} | "
            f"{_fmt(row['partial_r_vs_rank_given_survival'])} | {int(row['n'])} |"
        )
    lines.append("")
    lines.append("Negative correlation means a higher network score is associated with a better final rank, because rank 1 is best.")
    lines.append("")
    lines.append("Quartile comparison:")
    lines.append("| metric | low mean rank | high mean rank | high-low delta |")
    lines.append("|---|---:|---:|---:|")
    for _, row in quartiles.iterrows():
        if row["metric"] in [m[0] for m in CENTRALITY_METRICS] + ["battle_center_score"]:
            lines.append(
                f"| {row['label']} | {_fmt(row['low_mean_rank'])} | "
                f"{_fmt(row['high_mean_rank'])} | {_fmt(row['delta_high_minus_low'])} |"
            )
    lines.append("")
    lines.append("## 4. Community Structure")
    q = community_summary.get("match_modularity_q", {})
    lines.append(f"- Modularity Q mean: {_fmt(q.get('mean'))}, median: {_fmt(q.get('median'))}")
    lines.append(f"- Position counts: `{community_summary.get('position_counts', {})}`")
    lines.append("")
    lines.append("| position | n | mean rank | median rank | mean degree | mean betweenness |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in position_summary.iterrows():
        lines.append(
            f"| {row['position_label']} | {int(row['n'])} | {_fmt(row['mean_rank'])} | "
            f"{_fmt(row['median_rank'])} | {_fmt(row['mean_degree'])} | "
            f"{_fmt(row['mean_betweenness'])} |"
        )
    lines.append("")
    lines.append("## 5. Early 300s vs Full Match")
    er = match_summary.get("early_edge_ratio", {})
    dr = match_summary.get("early_damage_ratio", {})
    lines.append(f"- Early edge share mean: {_fmt(er.get('mean'))}, median: {_fmt(er.get('median'))}")
    lines.append(f"- Early damage share mean: {_fmt(dr.get('mean'))}, median: {_fmt(dr.get('median'))}")
    lines.append("")
    lines.append("This separates early voluntary engagement from full-match survival exposure. "
                 "It should be kept as a standard control axis in later hypothesis tests.")
    lines.append("")
    lines.append("## 6. Top Teams by Network Role")
    for name, rows in top_tables.items():
        lines.append(f"### {name}")
        lines.append("| match | team | rank | degree | weighted_deg | betweenness | pagerank | position |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for row in rows[:8]:
            lines.append(
                f"| {row.get('match_id')} | {row.get('team_id')} | {row.get('final_rank')} | "
                f"{_fmt(row.get('degree'))} | {_fmt(row.get('weighted_deg'))} | "
                f"{_fmt(row.get('c_betweenness'))} | {_fmt(row.get('c_pagerank'))} | "
                f"{row.get('position_label')} |"
            )
        lines.append("")
    lines.append("## Recommended Next Step")
    lines.append("Proceed with three formal analysis tracks: degree/engagement intensity, bridge/battle-center roles, and community structure. "
                 "For each track, compare full-match metrics with early-300s metrics to separate survival-time exposure from early strategic tendency.")
    lines.append("")
    return "\n".join(lines)


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = os.path.abspath(args.dataset)
    out_dir = os.path.abspath(args.out or os.path.join(dataset_dir, "basic_graph_analysis"))
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    matches, teams, edges, metadata = _load_dataset(dataset_dir)
    degree_dist = _degree_distribution(teams)
    correlations = _centrality_correlations(teams)
    quartiles = _metric_quartiles(teams)
    position_summary = _position_rank_summary(teams)
    match_summary = _match_summaries(matches)
    community_summary = _community_summary(teams)
    top_tables = {
        "Top Degree": _top_rows(teams, "degree"),
        "Top Weighted Degree": _top_rows(teams, "weighted_deg"),
        "Top Betweenness": _top_rows(teams, "c_betweenness"),
        "Top PageRank": _top_rows(teams, "c_pagerank"),
    }

    degree_dist.to_csv(os.path.join(out_dir, "degree_distribution.csv"), index=False)
    correlations.to_csv(os.path.join(out_dir, "centrality_correlations.csv"), index=False)
    quartiles.to_csv(os.path.join(out_dir, "metric_quartiles.csv"), index=False)
    position_summary.to_csv(os.path.join(out_dir, "position_rank_summary.csv"), index=False)

    enriched_matches = matches.copy()
    enriched_matches["early_edge_ratio"] = (
        _as_float(enriched_matches["early_n_edges"])
        / _as_float(enriched_matches["full_n_edges"]).replace(0, np.nan)
    )
    enriched_matches["early_damage_ratio"] = (
        _as_float(enriched_matches["early_damage_total"])
        / _as_float(enriched_matches["damage_total"]).replace(0, np.nan)
    )
    enriched_matches["giant_team_ratio"] = (
        _as_float(enriched_matches["full_giant_size"])
        / _as_float(enriched_matches["n_teams"]).replace(0, np.nan)
    )
    enriched_matches.to_csv(os.path.join(out_dir, "match_network_properties.csv"), index=False)

    _plot_degree_distribution(degree_dist, os.path.join(fig_dir, "degree_distribution.png"))
    _plot_match_properties(matches, os.path.join(fig_dir, "match_network_properties.png"))
    _plot_centrality_rank(teams, os.path.join(fig_dir, "centrality_vs_rank.png"))
    _plot_early_full(matches, os.path.join(fig_dir, "early_vs_full.png"))

    summary = {
        "dataset_dir": dataset_dir,
        "out_dir": out_dir,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_matches": int(matches.shape[0]),
        "n_teams": int(teams.shape[0]),
        "n_edges_rows": int(edges.shape[0]),
        "match_summary": match_summary,
        "community_summary": community_summary,
        "top_tables": top_tables,
        "artifacts": {
            "report": os.path.join(out_dir, "basic_graph_report.md"),
            "summary_json": os.path.join(out_dir, "analysis_summary.json"),
            "degree_distribution": os.path.join(out_dir, "degree_distribution.csv"),
            "centrality_correlations": os.path.join(out_dir, "centrality_correlations.csv"),
            "metric_quartiles": os.path.join(out_dir, "metric_quartiles.csv"),
            "position_rank_summary": os.path.join(out_dir, "position_rank_summary.csv"),
            "figures": fig_dir,
        },
    }

    report = _report_markdown(
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        metadata=metadata,
        match_summary=match_summary,
        community_summary=community_summary,
        degree_dist=degree_dist,
        correlations=correlations,
        quartiles=quartiles,
        position_summary=position_summary,
        top_tables=top_tables,
    )
    with open(os.path.join(out_dir, "basic_graph_report.md"), "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n")

    _write_json(os.path.join(out_dir, "analysis_summary.json"), summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run basic graph analysis on PUBG team-network CSVs.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="Dataset directory produced by build_analysis_dataset.py")
    parser.add_argument("--out", default=None,
                        help="Output directory; default is <dataset>/basic_graph_analysis")
    return parser.parse_args()


def main() -> None:
    summary = run_analysis(parse_args())
    print(json.dumps({
        "out_dir": summary["out_dir"],
        "n_matches": summary["n_matches"],
        "n_teams": summary["n_teams"],
        "report": summary["artifacts"]["report"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
