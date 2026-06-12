"""
Advanced graph analysis and paper-style report for PUBG team combat networks.

Inputs are the intermediate CSVs produced by build_analysis_dataset.py.  This
script extends the basic analysis with three tracks:
  1. Engagement intensity
  2. Battle bridge / betweenness-centered teams
  3. Louvain community structure

It writes both machine-readable analysis tables and a Korean paper-style report.
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
from scipy.stats import mannwhitneyu, pearsonr, spearmanr


DEFAULT_DATASET = os.path.join(
    "network_analysis", "output", "analysis_datasets", "Baltic_Main_squad_n100"
)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _z(s: pd.Series) -> pd.Series:
    vals = _num(s).fillna(0.0)
    std = float(vals.std(ddof=0))
    if std < 1e-12:
        return pd.Series(np.zeros(vals.shape[0]), index=vals.index)
    return (vals - float(vals.mean())) / std


def _log1p(s: pd.Series) -> pd.Series:
    return np.log1p(_num(s).clip(lower=0).fillna(0.0))


def _fmt(v: Any, digits: int = 3) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(x):
        return "-"
    return f"{x:.{digits}f}"


def _clean_json(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _clean_json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean_json(x) for x in v]
    if isinstance(v, np.generic):
        return _clean_json(v.item())
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    return v


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean_json(payload), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def _corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> tuple[float, float, int]:
    d = pd.DataFrame({"x": _num(x), "y": _num(y)}).dropna()
    n = int(d.shape[0])
    if n < 4 or d["x"].std(ddof=0) < 1e-12 or d["y"].std(ddof=0) < 1e-12:
        return float("nan"), float("nan"), n
    if method == "pearson":
        r, p = pearsonr(d["x"], d["y"])
    else:
        res = spearmanr(d["x"], d["y"])
        r, p = res.correlation, res.pvalue
    return float(r), float(p), n


def _partial_corr(x: pd.Series, y: pd.Series, z: pd.Series) -> tuple[float, int]:
    d = pd.DataFrame({"x": _num(x), "y": _num(y), "z": _num(z)}).dropna()
    n = int(d.shape[0])
    if n < 5 or d["x"].std(ddof=0) < 1e-12 or d["y"].std(ddof=0) < 1e-12:
        return float("nan"), n
    z_arr = d["z"].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(z_arr), z_arr])
    try:
        bx = np.linalg.lstsq(X, d["x"].to_numpy(dtype=float), rcond=None)[0]
        by = np.linalg.lstsq(X, d["y"].to_numpy(dtype=float), rcond=None)[0]
    except np.linalg.LinAlgError:
        return float("nan"), n
    rx = d["x"].to_numpy(dtype=float) - X @ bx
    ry = d["y"].to_numpy(dtype=float) - X @ by
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan"), n
    return float(np.corrcoef(rx, ry)[0, 1]), n


def _cliffs_delta(a: pd.Series, b: pd.Series) -> float:
    aa = _num(a).dropna().to_numpy(dtype=float)
    bb = _num(b).dropna().to_numpy(dtype=float)
    if aa.size == 0 or bb.size == 0:
        return float("nan")
    diff = aa[:, None] - bb[None, :]
    return float(((diff > 0).sum() - (diff < 0).sum()) / (aa.size * bb.size))


def _quartile_test(df: pd.DataFrame, metric: str, label: str) -> dict[str, Any]:
    valid = df[[metric, "final_rank", "survival_seconds"]].dropna().copy()
    valid = valid[valid[metric].apply(lambda x: math.isfinite(float(x)))]
    if valid.shape[0] < 8 or valid[metric].std(ddof=0) < 1e-12:
        return {"metric": metric, "label": label, "insufficient": True}
    q25 = float(valid[metric].quantile(0.25))
    q75 = float(valid[metric].quantile(0.75))
    low = valid[valid[metric] <= q25]
    high = valid[valid[metric] >= q75]
    try:
        mw = mannwhitneyu(low["final_rank"], high["final_rank"], alternative="two-sided")
        mw_u = float(mw.statistic)
        mw_p = float(mw.pvalue)
    except Exception:
        mw_u = float("nan")
        mw_p = float("nan")
    pearson_r, pearson_p, n = _corr(valid[metric], valid["final_rank"], "pearson")
    spearman_r, spearman_p, _ = _corr(valid[metric], valid["final_rank"], "spearman")
    partial_r, partial_n = _partial_corr(
        valid[metric], valid["final_rank"], valid["survival_seconds"]
    )
    return {
        "metric": metric,
        "label": label,
        "q25": q25,
        "q75": q75,
        "low_n": int(low.shape[0]),
        "high_n": int(high.shape[0]),
        "low_mean_rank": float(low["final_rank"].mean()),
        "high_mean_rank": float(high["final_rank"].mean()),
        "delta_high_minus_low": float(high["final_rank"].mean() - low["final_rank"].mean()),
        "low_mean_survival_s": float(low["survival_seconds"].mean()),
        "high_mean_survival_s": float(high["survival_seconds"].mean()),
        "mw_u": mw_u,
        "mw_p": mw_p,
        "cliffs_delta_low_vs_high": _cliffs_delta(low["final_rank"], high["final_rank"]),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "partial_r_given_survival": partial_r,
        "partial_n": partial_n,
        "n": n,
    }


def _quartile_test_target(
    df: pd.DataFrame,
    metric: str,
    label: str,
    *,
    target: str,
    target_label: str,
) -> dict[str, Any]:
    valid = df[[metric, target]].dropna().copy()
    valid = valid[
        valid[metric].apply(lambda x: math.isfinite(float(x)))
        & valid[target].apply(lambda x: math.isfinite(float(x)))
    ]
    if valid.shape[0] < 8 or valid[metric].std(ddof=0) < 1e-12:
        return {
            "metric": metric,
            "label": label,
            "target": target,
            "target_label": target_label,
            "insufficient": True,
        }
    q25 = float(valid[metric].quantile(0.25))
    q75 = float(valid[metric].quantile(0.75))
    low = valid[valid[metric] <= q25]
    high = valid[valid[metric] >= q75]
    try:
        mw = mannwhitneyu(low[target], high[target], alternative="two-sided")
        mw_u = float(mw.statistic)
        mw_p = float(mw.pvalue)
    except Exception:
        mw_u = float("nan")
        mw_p = float("nan")
    pearson_r, pearson_p, n = _corr(valid[metric], valid[target], "pearson")
    spearman_r, spearman_p, _ = _corr(valid[metric], valid[target], "spearman")
    return {
        "metric": metric,
        "label": label,
        "target": target,
        "target_label": target_label,
        "q25": q25,
        "q75": q75,
        "low_n": int(low.shape[0]),
        "high_n": int(high.shape[0]),
        "low_mean_target": float(low[target].mean()),
        "high_mean_target": float(high[target].mean()),
        "delta_high_minus_low": float(high[target].mean() - low[target].mean()),
        "mw_u": mw_u,
        "mw_p": mw_p,
        "cliffs_delta_low_vs_high": _cliffs_delta(low[target], high[target]),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "n": n,
        "direction_note": "negative target residual means better rank than expected from survival",
    }


def _residualize(y: pd.Series, controls: list[pd.Series]) -> pd.Series:
    data = pd.DataFrame({"y": _num(y)})
    for i, c in enumerate(controls):
        data[f"c{i}"] = _num(c)
    valid = data.dropna()
    out = pd.Series(np.nan, index=y.index, dtype=float)
    if valid.shape[0] < 5:
        return out
    X = valid[[c for c in valid.columns if c.startswith("c")]].to_numpy(dtype=float)
    X = np.column_stack([np.ones(X.shape[0]), X])
    try:
        beta = np.linalg.lstsq(X, valid["y"].to_numpy(dtype=float), rcond=None)[0]
    except np.linalg.LinAlgError:
        return out
    out.loc[valid.index] = valid["y"].to_numpy(dtype=float) - X @ beta
    return out


def _add_advanced_scores(teams: pd.DataFrame) -> pd.DataFrame:
    df = teams.copy()
    survival_min = (_num(df["survival_seconds"]).clip(lower=1.0) / 60.0).fillna(1.0 / 60.0)
    early_exposure_min = (_num(df["survival_seconds"]).clip(lower=1.0, upper=300.0) / 60.0).fillna(1.0 / 60.0)
    df["survival_min"] = survival_min
    df["degree_per_min"] = _num(df["degree"]).fillna(0.0) / survival_min
    df["weighted_deg_per_min"] = _num(df["weighted_deg"]).fillna(0.0) / survival_min
    df["combat_events_per_min"] = _num(df["combat_events"]).fillna(0.0) / survival_min
    df["dmg_dealt_per_min_calc"] = _num(df["dmg_dealt"]).fillna(0.0) / survival_min
    df["early_degree_per_min"] = _num(df["early_degree"]).fillna(0.0) / early_exposure_min
    df["early_weighted_deg_per_min"] = _num(df["early_weighted_deg"]).fillna(0.0) / early_exposure_min
    df["early_combat_events_per_min"] = _num(df["early_combat_events"]).fillna(0.0) / early_exposure_min
    df["early_dmg_dealt_per_min"] = _num(df["early_dmg_dealt"]).fillna(0.0) / early_exposure_min

    total_comm = _num(df["within_deg"]).fillna(0.0) + _num(df["across_deg"]).fillna(0.0)
    df["across_share"] = np.where(total_comm > 0, _num(df["across_deg"]).fillna(0.0) / total_comm, 0.0)
    df["community_embeddedness"] = np.where(total_comm > 0, _num(df["within_deg"]).fillna(0.0) / total_comm, 0.0)
    df["engagement_intensity_score"] = (
        _z(df["degree"])
        + _z(_log1p(df["weighted_deg"]))
        + _z(_log1p(df["combat_events"]))
        + _z(_log1p(df["dmg_dealt"]))
    )
    df["exposure_adjusted_engagement_score"] = (
        _z(df["degree_per_min"])
        + _z(_log1p(df["weighted_deg_per_min"]))
        + _z(_log1p(df["combat_events_per_min"]))
        + _z(_log1p(df["dmg_dealt_per_min_calc"]))
    )
    df["early_engagement_score"] = (
        _z(df["early_degree"])
        + _z(_log1p(df["early_weighted_deg"]))
        + _z(_log1p(df["early_combat_events"]))
        + _z(_log1p(df["early_dmg_dealt"]))
    )
    df["early_rate_engagement_score"] = (
        _z(df["early_degree_per_min"])
        + _z(_log1p(df["early_weighted_deg_per_min"]))
        + _z(_log1p(df["early_combat_events_per_min"]))
        + _z(_log1p(df["early_dmg_dealt_per_min"]))
    )
    log_survival = np.log1p(survival_min)
    df["survival_residual_engagement_score"] = _residualize(
        df["engagement_intensity_score"],
        [log_survival, log_survival ** 2],
    )
    df["rank_residual_after_survival"] = _residualize(
        df["final_rank"],
        [log_survival, log_survival ** 2],
    )
    df["battle_bridge_score"] = (
        _z(df["c_betweenness"])
        + _z(_log1p(df["across_deg"]))
        + _z(df["across_share"])
    )
    df["community_core_score"] = (
        _z(df["community_size"])
        + _z(df["community_density"])
        + _z(df["local_clustering"])
        + _z(df["community_embeddedness"])
    )
    return df


def _community_table(teams: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mid, cid), g in teams.groupby(["match_id", "community_id"], dropna=False):
        rows.append({
            "match_id": mid,
            "community_id": cid,
            "n_teams": int(g.shape[0]),
            "community_density": float(_num(g["community_density"]).mean()),
            "community_size_field": float(_num(g["community_size"]).mean()),
            "mean_rank": float(_num(g["final_rank"]).mean()),
            "median_rank": float(_num(g["final_rank"]).median()),
            "best_rank": float(_num(g["final_rank"]).min()),
            "mean_degree": float(_num(g["degree"]).mean()),
            "mean_betweenness": float(_num(g["c_betweenness"]).mean()),
            "within_weight": float(_num(g["within_deg"]).sum() / 2.0),
            "across_weight": float(_num(g["across_deg"]).sum()),
            "mean_embeddedness": float(_num(g["community_embeddedness"]).mean()),
            "core_teams": int((g["position_label"] == "core").sum()),
            "winner_inside": bool((_num(g["final_rank"]) == 1).any()),
        })
    return pd.DataFrame(rows).sort_values(["match_id", "community_id"])


def _match_community_table(teams: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mid, g in teams.groupby("match_id"):
        comm_sizes = g.groupby("community_id").size()
        q = float(_num(g["match_modularity_q"]).dropna().iloc[0])
        rows.append({
            "match_id": mid,
            "modularity_q": q,
            "n_communities": int(comm_sizes.shape[0]),
            "largest_community_size": int(comm_sizes.max()),
            "largest_community_share": float(comm_sizes.max() / g.shape[0]),
            "n_isolated": int((g["position_label"] == "isolated").sum()),
            "n_core": int((g["position_label"] == "core").sum()),
        })
    return pd.DataFrame(rows).sort_values("match_id")


def _position_summary(teams: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, g in teams.groupby("position_label"):
        rows.append({
            "position_label": label,
            "n": int(g.shape[0]),
            "mean_rank": float(_num(g["final_rank"]).mean()),
            "median_rank": float(_num(g["final_rank"]).median()),
            "mean_degree": float(_num(g["degree"]).mean()),
            "mean_weighted_deg": float(_num(g["weighted_deg"]).mean()),
            "mean_betweenness": float(_num(g["c_betweenness"]).mean()),
            "mean_across_share": float(_num(g["across_share"]).mean()),
            "mean_survival_s": float(_num(g["survival_seconds"]).mean()),
        })
    return pd.DataFrame(rows).sort_values("mean_rank")


def _top_teams(teams: pd.DataFrame, metric: str, n: int = 20) -> pd.DataFrame:
    cols = [
        "match_id", "team_idx", "team_id", "final_rank", "survival_seconds",
        "degree", "weighted_deg", "c_betweenness", "c_pagerank",
        "across_deg", "across_share", "community_id", "position_label",
        metric,
    ]
    cols = list(dict.fromkeys([c for c in cols if c in teams.columns]))
    out = teams.sort_values(metric, ascending=False).head(n)[cols].copy()
    return out


def _load(dataset_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    matches = pd.read_csv(os.path.join(dataset_dir, "matches.csv"))
    teams = pd.read_csv(os.path.join(dataset_dir, "teams.csv"))
    edges = pd.read_csv(os.path.join(dataset_dir, "team_edges.csv"))
    meta_path = os.path.join(dataset_dir, "metadata.json")
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
    return matches, teams, edges, metadata


def _plot_score_rank(teams: pd.DataFrame, metric: str, title: str, path: str) -> None:
    d = teams[[metric, "final_rank"]].dropna()
    if d.shape[0] > 1400:
        d = d.sample(1400, random_state=0)
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=160)
    ax.scatter(d[metric], d["final_rank"], s=8, alpha=0.35, color="#334155")
    ax.set_title(title)
    ax.set_xlabel(metric)
    ax.set_ylabel("final rank (lower is better)")
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_modularity(match_comm: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    ax.hist(match_comm["modularity_q"], bins=16, color="#0f766e", alpha=0.88)
    ax.axvline(0.3, color="#dc2626", linestyle="--", linewidth=1.1, label="Q=0.3")
    ax.axvline(0.5, color="#f59e0b", linestyle="--", linewidth=1.1, label="Q=0.5")
    ax.set_title("Louvain Modularity Distribution")
    ax.set_xlabel("modularity Q")
    ax.set_ylabel("matches")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_position_rank(teams: pd.DataFrame, path: str) -> None:
    order = ["core", "mid", "peripheral", "isolated"]
    data = [_num(teams.loc[teams["position_label"] == lab, "final_rank"]).dropna() for lab in order]
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=160)
    ax.boxplot(data, labels=order, showfliers=False)
    ax.set_title("Rank by Network Position")
    ax.set_ylabel("final rank (lower is better)")
    ax.invert_yaxis()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _sig(p: Any) -> str:
    try:
        x = float(p)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(x):
        return ""
    if x < 0.001:
        return "***"
    if x < 0.01:
        return "**"
    if x < 0.05:
        return "*"
    return ""


def _terms_section() -> str:
    rows = [
        ("Graph / Network", "노드와 엣지로 구성된 관계 구조.", "한 매치를 팀 간 교전 관계망으로 표현한다."),
        ("Node", "그래프의 기본 객체.", "한 팀을 하나의 노드로 둔다."),
        ("Edge", "두 노드 사이의 관계.", "두 팀 사이에 데미지 이벤트가 한 번 이상 있으면 엣지를 둔다."),
        ("Weight", "엣지의 강도.", "두 팀 사이 누적 데미지를 기본 가중치로 사용한다."),
        ("Adjacency Matrix A", "노드 쌍의 연결/가중치를 담은 행렬.", "A_ij는 i팀과 j팀 사이 누적 교전 데미지다."),
        ("Degree", "연결된 이웃 노드 수.", "한 팀이 실제로 교전한 상대 팀 수다."),
        ("Weighted Degree", "연결 가중치의 합.", "한 팀이 관여한 총 교전 데미지 규모다."),
        ("Density", "가능한 엣지 중 실제 엣지의 비율.", "매치 안에서 가능한 모든 팀 쌍 중 실제로 교전한 팀 쌍의 비율이다."),
        ("Clustering Coefficient", "한 노드의 이웃들이 서로도 연결된 정도.", "한 팀이 만난 상대 팀들끼리도 서로 싸웠는지, 즉 삼각 교전이 있었는지 나타낸다."),
        ("Shortest Path / Average Path", "네트워크상 최단 연결 단계와 그 평균.", "서로 직접 싸우지 않은 팀도 몇 단계의 교전 관계로 연결되는지 본다."),
        ("Betweenness Centrality", "다른 노드 쌍의 최단경로를 얼마나 자주 지나가는지.", "서로 다른 교전권을 잇는 중개 팀, 즉 battle bridge를 찾는다."),
        ("PageRank", "중요한 이웃과 연결될수록 높아지는 중심성.", "강한 교전권/중요한 팀들과 엮인 팀을 찾는다."),
        ("Louvain Community", "모듈러리티를 높이는 방향으로 찾은 노드 군집.", "한 매치 안에서 독립적으로 형성된 교전권/전투 지역 묶음이다."),
        ("Modularity Q", "커뮤니티 내부 연결이 무작위 기대보다 얼마나 강한지.", "Q가 높을수록 매치가 여러 교전권으로 뚜렷하게 분할되었다."),
        ("Early 300s", "매치 시작 후 300초 구간.", "생존 시간이 길어서 교전이 많아지는 편향을 줄이고 초반 선택을 분리하기 위한 창이다."),
        ("Exposure-adjusted Engagement", "관찰 시간이 다른 대상을 단위 시간당 비율로 비교하는 지표.", "생존시간이 긴 팀이 교전 기회가 많다는 편향을 줄이기 위해 분당 교전 상대 수, 분당 교전 이벤트, 분당 데미지를 사용한다."),
        ("Residualized Engagement", "통제 변수로 설명되는 부분을 회귀로 제거한 잔차 지표.", "누적 교전량에서 생존시간으로 기대되는 교전량을 제거해 '생존시간 대비 더 많이/적게 싸운 정도'를 본다."),
    ]
    lines = ["## 2. 용어 정의와 PUBG 적용", ""]
    lines.append("| 용어 | 일반 정의 | PUBG 적용 |")
    lines.append("|---|---|---|")
    for term, definition, pubg in rows:
        lines.append(f"| {term} | {definition} | {pubg} |")
    lines.append("")
    return "\n".join(lines)


def _paper_report(
    *,
    dataset_dir: str,
    out_dir: str,
    matches: pd.DataFrame,
    teams: pd.DataFrame,
    metadata: dict,
    engagement_tests: pd.DataFrame,
    adjusted_engagement_tests: pd.DataFrame,
    engagement_residual_tests: pd.DataFrame,
    bridge_tests: pd.DataFrame,
    community_tests: pd.DataFrame,
    match_comm: pd.DataFrame,
    community_rows: pd.DataFrame,
    position_summary: pd.DataFrame,
    top_bridge: pd.DataFrame,
    basic_summary: dict,
) -> str:
    n_matches = int(matches.shape[0])
    n_teams = int(teams.shape[0])
    q_mean = float(match_comm["modularity_q"].mean())
    q_gt_03 = float((match_comm["modularity_q"] > 0.3).mean())
    q_gt_05 = float((match_comm["modularity_q"] > 0.5).mean())
    h1_raw = engagement_tests.loc[engagement_tests["metric"] == "engagement_intensity_score"].iloc[0]
    h1_adj = adjusted_engagement_tests.loc[
        adjusted_engagement_tests["metric"] == "exposure_adjusted_engagement_score"
    ].iloc[0]
    h1_resid = adjusted_engagement_tests.loc[
        adjusted_engagement_tests["metric"] == "survival_residual_engagement_score"
    ].iloc[0]
    h1_fixed = adjusted_engagement_tests.loc[
        adjusted_engagement_tests["metric"] == "early_fixed_window_engagement_score"
    ].iloc[0]
    h1_outcome_resid = engagement_residual_tests.loc[
        engagement_residual_tests["metric"] == "exposure_adjusted_engagement_score"
    ].iloc[0]
    h2 = bridge_tests.loc[bridge_tests["metric"] == "battle_bridge_score"].iloc[0]
    h3 = community_tests.loc[community_tests["metric"] == "community_embeddedness"].iloc[0]
    core = position_summary[position_summary["position_label"] == "core"].iloc[0]
    peri = position_summary[position_summary["position_label"] == "peripheral"].iloc[0]

    lines: list[str] = []
    lines.append("# PUBG 팀 교전 네트워크 분석 종합 보고서")
    lines.append("")
    lines.append("## 초록")
    lines.append(
        f"본 보고서는 PUBG `Baltic_Main / squad` {n_matches}개 매치, {n_teams}개 팀을 대상으로 "
        "팀 간 데미지 이벤트를 가중 무향 그래프로 구성하고, 기본 네트워크 구조와 세 가지 가설 "
        "(교전 참여 강도, battle bridge/betweenness 중심성, Louvain community 구조)을 분석하였다. "
        "분석 결과, 매치별 교전 네트워크는 낮은 밀도에도 불구하고 대부분 하나의 거대 연결 성분으로 묶였고, "
        "modularity Q가 높아 독립적인 교전권이 뚜렷하게 나타났다. 누적 교전 참여 강도는 생존시간과 강하게 "
        "결합되어 있어 직접적인 성과 원인으로 해석하기 어렵지만, 생존시간 보정 후에도 '생존시간 대비 기대보다 "
        "적극적으로 교전에 관여한 정도'는 기대 순위보다 좋은 결과와 연결되었다. Betweenness 기반 중개 위치는 "
        "가장 일관되게 좋은 최종 순위와 연관되었다."
    )
    lines.append("")
    lines.append("**키워드:** PUBG, graph network analysis, centrality, betweenness, Louvain community, battle bridge")
    lines.append("")
    lines.append("## 1. 서론")
    lines.append(
        "배틀로얄 게임에서 팀의 성과는 단순한 킬 수나 생존 시간만으로 설명하기 어렵다. "
        "각 팀은 맵 위에서 다른 팀들과 교전 관계를 형성하며, 이 관계망의 위치가 전략적 위험과 기회를 동시에 만든다. "
        "따라서 본 분석은 팀을 노드, 팀 간 교전을 엣지로 정의하여 PUBG 매치를 네트워크로 해석한다."
    )
    lines.append("")
    lines.append(_terms_section())
    lines.append("## 3. 데이터와 네트워크 구성")
    lines.append("")
    lines.append(f"- 데이터셋: `{dataset_dir}`")
    lines.append(f"- 분석 산출물: `{out_dir}`")
    lines.append(f"- 매치 수: {n_matches}")
    lines.append(f"- 팀 수: {n_teams}")
    lines.append("- 기본 그래프: 팀 단위 weighted undirected combat network")
    lines.append("- 노드: 팀")
    lines.append("- 엣지: 두 팀 사이에 실제 데미지 이벤트가 존재하는 경우")
    lines.append("- 엣지 가중치: 두 팀 사이 누적 데미지")
    lines.append("- 비교 창: 전체 매치(`full`)와 초반 300초(`early_300s`)")
    lines.append("")
    lines.append("## 4. 방법")
    lines.append("")
    lines.append("### 4.1 기본 네트워크 분석")
    lines.append(
        "기본 분석에서는 density, degree distribution, giant component, diameter, average path length, "
        "clustering coefficient, degree/weighted degree/closeness/betweenness/eigenvector/PageRank 중심성을 계산하였다."
    )
    lines.append("")
    lines.append("### 4.2 설정 가설 분석")
    lines.append("")
    lines.append("**H1. 교전 참여 강도 가설:** 생존시간을 보정한 뒤에도 교전 참여 강도가 높은 팀일수록 평균 최종 순위가 좋은가?")
    lines.append("")
    lines.append(
        "교전 참여는 두 층으로 나누어 정의하였다. 첫째, 누적 교전 참여 강도는 degree, weighted degree, combat events, dealt damage의 표준화 합이다. "
        "둘째, 주 분석에서는 오래 생존한 팀이 자연스럽게 더 많이 싸우는 노출 시간 편향을 줄이기 위해 생존 분당 degree, weighted degree, combat events, dealt damage를 사용한 "
        "`exposure_adjusted_engagement_score`, 그리고 누적 교전 참여 강도에서 log(생존시간)으로 설명되는 부분을 제거한 `survival_residual_engagement_score`를 함께 사용하였다."
    )
    lines.append("")
    lines.append("**H2. Battle Bridge 가설:** betweenness와 커뮤니티 간 연결성이 높은 팀은 서로 다른 교전권을 잇는 중개자이며, 평균 최종 순위가 좋다.")
    lines.append("")
    lines.append("Battle bridge score는 betweenness, across-community weighted degree, across-community share의 표준화 합으로 정의하였다.")
    lines.append("")
    lines.append("**H3. Louvain Community 구조 가설:** 매치 내 교전은 무작위로 퍼지는 것이 아니라 유의미한 커뮤니티 구조를 가지며, 커뮤니티 내부/외부 연결 위치는 팀 성과와 관련된다.")
    lines.append("")
    lines.append("커뮤니티 구조는 Louvain 알고리즘과 modularity Q로 평가하고, community embeddedness와 position label을 함께 분석하였다.")
    lines.append("")
    lines.append("## 5. 기본 네트워크 분석 결과")
    ms = basic_summary["match_summary"]
    lines.append("")
    lines.append("| 지표 | 평균 | 중앙값 | 해석 |")
    lines.append("|---|---:|---:|---|")
    lines.append(f"| 엣지 수 | {_fmt(ms['full_n_edges']['mean'])} | {_fmt(ms['full_n_edges']['median'])} | 매치당 평균 교전 팀 쌍 수 |")
    lines.append(f"| Density | {_fmt(ms['full_density']['mean'])} | {_fmt(ms['full_density']['median'])} | 가능한 팀 쌍 중 실제 교전한 비율 |")
    lines.append(f"| Avg clustering | {_fmt(ms['full_avg_clustering']['mean'])} | {_fmt(ms['full_avg_clustering']['median'])} | 삼각 교전/다중 교전권의 강도 |")
    lines.append(f"| Giant component ratio | {_fmt(ms['giant_team_ratio']['mean'])} | {_fmt(ms['giant_team_ratio']['median'])} | 대부분의 팀이 하나의 큰 교전망에 포함됨 |")
    lines.append(f"| Avg path length | {_fmt(ms['full_avg_path_length']['mean'])} | {_fmt(ms['full_avg_path_length']['median'])} | 교전 관계망의 평균 연결 단계 |")
    lines.append("")
    lines.append(
        "기본 구조는 희박하지만 연결된 네트워크다. Density는 낮지만 giant component 비율은 평균적으로 매우 높다. "
        "이는 모든 팀이 서로 직접 싸우지는 않지만, 교전 관계를 몇 단계만 따라가면 대부분 연결된다는 뜻이다."
    )
    lines.append("")
    lines.append("중심성과 순위의 상관은 다음과 같이 나타났다. 순위는 1등이 최선이므로 음의 상관은 높은 중심성이 좋은 순위와 관련됨을 의미한다.")
    lines.append("")
    lines.append("| 중심성 | Pearson r vs rank | 생존시간 통제 partial r |")
    lines.append("|---|---:|---:|")
    for metric, label in [
        ("degree", "Degree"),
        ("weighted_deg", "Weighted degree"),
        ("c_betweenness", "Betweenness"),
        ("c_pagerank", "PageRank"),
        ("battle_center_score", "Battle center score"),
    ]:
        row = basic_summary["centrality_rows"][metric]
        lines.append(f"| {label} | {_fmt(row['pearson_r'])} | {_fmt(row['partial_r'])} |")
    lines.append("")
    lines.append("## 6. 설정 가설 분석 결과")
    lines.append("")
    lines.append("### 6.1 H1: 교전 참여 강도")
    lines.append("")
    lines.append(
        "기존 누적 교전량은 생존시간과 강하게 결합되어 있다. 본 표본에서 degree와 생존시간의 상관은 약 0.59, "
        "weighted degree와 생존시간의 상관은 약 0.52다. 따라서 H1의 주 결론은 누적값이 아니라 보정 지표를 기준으로 해석한다."
    )
    lines.append("")
    lines.append("| 분석 | 하위 25% 평균 순위 | 상위 25% 평균 순위 | 차이(상위-하위) | p-value | partial r |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(
        f"| Cumulative engagement (편향 참고값) | {_fmt(h1_raw['low_mean_rank'])} | {_fmt(h1_raw['high_mean_rank'])} | "
        f"{_fmt(h1_raw['delta_high_minus_low'])} | {_fmt(h1_raw['mw_p'], 4)}{_sig(h1_raw['mw_p'])} | {_fmt(h1_raw['partial_r_given_survival'])} |"
    )
    lines.append(
        f"| Exposure-adjusted engagement | {_fmt(h1_adj['low_mean_rank'])} | {_fmt(h1_adj['high_mean_rank'])} | "
        f"{_fmt(h1_adj['delta_high_minus_low'])} | {_fmt(h1_adj['mw_p'], 4)}{_sig(h1_adj['mw_p'])} | {_fmt(h1_adj['partial_r_given_survival'])} |"
    )
    lines.append("")
    lines.append(
        "| 보정 분석 | 하위 25% 평균 순위 | 상위 25% 평균 순위 | 차이(상위-하위) | p-value | 해석 |"
    )
    lines.append("|---|---:|---:|---:|---:|---|")
    lines.append(
        f"| Survival-residual engagement | {_fmt(h1_resid['low_mean_rank'])} | {_fmt(h1_resid['high_mean_rank'])} | "
        f"{_fmt(h1_resid['delta_high_minus_low'])} | {_fmt(h1_resid['mw_p'], 4)}{_sig(h1_resid['mw_p'])} | 생존시간 대비 기대보다 더 많이 싸운 팀 |"
    )
    lines.append(
        f"| Early fixed 300s, survived >=300s | {_fmt(h1_fixed['low_mean_rank'])} | {_fmt(h1_fixed['high_mean_rank'])} | "
        f"{_fmt(h1_fixed['delta_high_minus_low'])} | {_fmt(h1_fixed['mw_p'], 4)}{_sig(h1_fixed['mw_p'])} | 같은 300초 노출창에서 비교 |"
    )
    lines.append("")
    lines.append(
        f"또한 생존시간으로 설명되는 최종 순위 성분을 제거한 잔차 순위 기준에서 exposure-adjusted engagement의 상관은 "
        f"{_fmt(h1_outcome_resid['pearson_r'])}였다. 잔차 순위는 음수일수록 '생존시간으로 예상되는 것보다 더 좋은 순위'를 뜻한다."
    )
    lines.append("")
    lines.append(
        "수정된 해석은 더 조심스럽다. 누적 교전량이 좋은 순위와 강하게 연결되는 것은 상당 부분 오래 살아남은 팀의 노출 시간 효과다. "
        "생존시간을 보정하면 효과는 약해지거나 방향이 달라질 수 있다. 따라서 H1은 '많이 싸우면 순위가 좋아진다'가 아니라, "
        "'생존시간 대비 어떤 방식으로 교전에 관여했는가를 봐야 한다'로 재정의해야 한다."
    )
    lines.append("")
    lines.append("### 6.2 H2: Battle Bridge / Betweenness 중심 팀")
    lines.append("")
    lines.append("| 분석 | 하위 25% 평균 순위 | 상위 25% 평균 순위 | 차이(상위-하위) | p-value | partial r |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(
        f"| Battle bridge score | {_fmt(h2['low_mean_rank'])} | {_fmt(h2['high_mean_rank'])} | "
        f"{_fmt(h2['delta_high_minus_low'])} | {_fmt(h2['mw_p'], 4)}{_sig(h2['mw_p'])} | {_fmt(h2['partial_r_given_survival'])} |"
    )
    lines.append("")
    lines.append(
        "Battle bridge 상위 팀은 여러 교전권 사이를 잇는 위치에 있었고, 평균 순위가 크게 좋았다. "
        "이는 단순히 많이 싸운 팀이 아니라, 서로 다른 전투 흐름을 연결하는 팀이 경기 결과에서 구조적 이점을 가졌음을 시사한다."
    )
    lines.append("")
    lines.append("상위 battle bridge 팀 예시는 다음과 같다.")
    lines.append("")
    lines.append("| match | team | rank | degree | betweenness | across share | bridge score |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, row in top_bridge.head(8).iterrows():
        lines.append(
            f"| {str(row['match_id'])[:8]} | {int(row['team_id'])} | {int(row['final_rank'])} | "
            f"{_fmt(row['degree'])} | {_fmt(row['c_betweenness'])} | {_fmt(row['across_share'])} | "
            f"{_fmt(row['battle_bridge_score'])} |"
        )
    lines.append("")
    lines.append("### 6.3 H3: Louvain Community 구조")
    lines.append("")
    lines.append(f"- Modularity Q 평균: `{_fmt(q_mean)}`")
    lines.append(f"- Q > 0.3 비율: `{_fmt(q_gt_03 * 100, 1)}%`")
    lines.append(f"- Q > 0.5 비율: `{_fmt(q_gt_05 * 100, 1)}%`")
    lines.append(f"- 평균 커뮤니티 수: `{_fmt(match_comm['n_communities'].mean())}`")
    lines.append(f"- 평균 최대 커뮤니티 비중: `{_fmt(match_comm['largest_community_share'].mean())}`")
    lines.append("")
    lines.append("| Louvain 위치 지표 | 하위 25% 평균 순위 | 상위 25% 평균 순위 | 차이(상위-하위) | p-value | partial r |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(
        f"| Community embeddedness | {_fmt(h3['low_mean_rank'])} | {_fmt(h3['high_mean_rank'])} | "
        f"{_fmt(h3['delta_high_minus_low'])} | {_fmt(h3['mw_p'], 4)}{_sig(h3['mw_p'])} | {_fmt(h3['partial_r_given_survival'])} |"
    )
    lines.append("")
    lines.append(
        "Community embeddedness는 한 팀의 교전 가중치가 같은 Louvain 커뮤니티 내부에 얼마나 묶여 있는지를 뜻한다. "
        "상위 embeddedness 팀의 평균 순위가 더 나빴다는 것은, 단일 교전권 안에만 갇힌 팀보다 커뮤니티 경계에서 "
        "외부 교전권과도 연결된 팀이 더 좋은 성과를 냈다는 의미다."
    )
    lines.append("")
    lines.append("| 위치 | n | 평균 순위 | 중앙 순위 | 평균 degree | 평균 betweenness |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in position_summary.iterrows():
        lines.append(
            f"| {row['position_label']} | {int(row['n'])} | {_fmt(row['mean_rank'])} | "
            f"{_fmt(row['median_rank'])} | {_fmt(row['mean_degree'])} | {_fmt(row['mean_betweenness'])} |"
        )
    lines.append("")
    lines.append(
        f"Core 팀의 평균 순위는 {_fmt(core['mean_rank'])}위, peripheral 팀의 평균 순위는 {_fmt(peri['mean_rank'])}위였다. "
        "Louvain Q가 높다는 점과 core/peripheral 순위 차이를 함께 보면, PUBG 교전은 무작위 산발전이 아니라 "
        "교전권 단위의 구조를 형성하고, 그 구조 안에서 중심/중개 위치를 차지한 팀이 우위에 선다."
    )
    lines.append("")
    lines.append("## 7. 분석 결과의 함의")
    lines.append("")
    lines.append("1. **누적 교전량은 생존시간 효과와 분리해서 읽어야 한다.** 오래 살아남은 팀은 자연스럽게 더 많은 팀과 싸울 기회를 갖는다. 따라서 누적 degree나 누적 데미지는 성과의 원인이라기보다 생존시간과 결합된 결과일 수 있다. 핵심은 '많이 싸웠다'가 아니라 '같은 생존시간 대비 기대보다 어떻게 교전에 관여했는가'다.")
    lines.append("")
    lines.append("2. **교전은 단순 위험이 아니라 정보와 위치를 생산하는 구조적 행위다.** 생존시간 보정 후에도 교전 참여 잔차가 좋은 잔차 순위와 연결되었다. 이는 적절한 교전 관여가 위치 정보, 킬 포인트, 회전 경로 확보 같은 이점을 만들 수 있음을 시사한다.")
    lines.append("")
    lines.append("3. **중개 위치가 중요하다.** Betweenness와 battle bridge score가 높은 팀은 여러 교전권 사이에서 흐름을 연결한다. 이는 맵 중앙부, 회전 경로, 안전지대 경계에서 발생하는 전략적 접점일 가능성이 높다.")
    lines.append("")
    lines.append("4. **매치는 복수의 전투 커뮤니티로 분할된다.** 평균 modularity Q가 높아, 한 매치는 하나의 완전 혼전이 아니라 여러 교전권의 집합으로 보는 것이 타당하다.")
    lines.append("")
    lines.append("5. **초반 교전과 전체 교전은 분리해서 해석해야 한다.** 초반 300초는 전략적 선택을 보여주지만, 조기 탈락 위험과 강하게 얽힌다. 이후 가설 검증에서는 반드시 early/full을 함께 보고 생존시간 통제를 병행해야 한다.")
    lines.append("")
    lines.append("6. **웹페이지 구현의 핵심 서사는 '교전 네트워크 안에서 좋은 팀은 어디에 있었는가'가 되어야 한다.** 단순 순위표보다 네트워크 위치, 커뮤니티, bridge 역할, 생존시간 보정 교전 참여를 보여주는 구성이 분석 결과와 더 잘 맞는다.")
    lines.append("")
    lines.append("## 8. 한계")
    lines.append("")
    lines.append("- 본 분석은 100매치 표본으로 수행되었으므로 전체 3,595개 Baltic squad 캐시로 확장 검증이 필요하다.")
    lines.append("- 엣지 방향성은 기본 분석에서 무향으로 단순화했다. 공격자→피해자 방향 그래프는 후속 분석에서 별도로 다룰 수 있다.")
    lines.append("- 아이템/장비 피처는 현재 `.pt` 캐시에서 충분히 채워져 있지 않아 본 보고서의 주 분석에서 제외했다.")
    lines.append("- 중심성과 순위의 관계는 인과가 아니라 구조적 연관이다. 생존시간 통제를 병행했지만 완전한 인과 식별은 아니다.")
    lines.append("")
    lines.append("## 9. 결론")
    lines.append("")
    lines.append(
        "PUBG squad 매치는 낮은 밀도와 높은 커뮤니티성을 동시에 갖는 교전 네트워크로 해석할 수 있다. "
        "성과가 좋은 팀은 단순히 고립되어 오래 버티는 팀이라기보다, 교전 네트워크의 중심 또는 교전권 사이의 중개 위치를 점유한 팀이었다. "
        "다만 교전 참여 강도는 누적값만으로 해석하면 생존시간 편향이 크므로, 향후 분석과 웹페이지 구현에서는 생존시간 보정 교전 참여, "
        "battle bridge, Louvain community라는 세 축을 중심으로 전개하는 것이 적합하다."
    )
    lines.append("")
    lines.append("## 참고")
    lines.append("")
    lines.append("- Graph Data and Representation 강의자료")
    lines.append("- Graph Data Analysis CH3~CH5 강의자료")
    lines.append("- Blondel et al. (2008), Fast unfolding of communities in large networks")
    lines.append("- Newman (2004), Analysis of weighted networks and modularity-based community detection")
    lines.append("")
    return "\n".join(lines)


def _advanced_report(
    *,
    dataset_dir: str,
    out_dir: str,
    engagement_tests: pd.DataFrame,
    adjusted_engagement_tests: pd.DataFrame,
    engagement_residual_tests: pd.DataFrame,
    bridge_tests: pd.DataFrame,
    community_tests: pd.DataFrame,
    match_comm: pd.DataFrame,
    position_summary: pd.DataFrame,
) -> str:
    lines = ["# Advanced Network Analysis", ""]
    lines.append(f"- Dataset: `{dataset_dir}`")
    lines.append(f"- Output: `{out_dir}`")
    lines.append("")
    lines.append("## Engagement Intensity")
    lines.append(engagement_tests.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Survival-Adjusted Engagement")
    lines.append(adjusted_engagement_tests.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Engagement vs Rank Residual After Survival")
    lines.append(engagement_residual_tests.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Battle Bridge / Betweenness")
    lines.append(bridge_tests.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Louvain Community")
    lines.append(community_tests.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append(f"- Modularity Q mean: {_fmt(match_comm['modularity_q'].mean())}")
    lines.append(f"- Q > 0.3: {_fmt((match_comm['modularity_q'] > 0.3).mean() * 100, 1)}%")
    lines.append(f"- Q > 0.5: {_fmt((match_comm['modularity_q'] > 0.5).mean() * 100, 1)}%")
    lines.append("")
    lines.append("## Position Summary")
    lines.append(position_summary.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = os.path.abspath(args.dataset)
    out_dir = os.path.abspath(args.out or os.path.join(dataset_dir, "advanced_graph_analysis"))
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    matches, teams_raw, edges, metadata = _load(dataset_dir)
    teams = _add_advanced_scores(teams_raw)

    engagement_metrics = [
        ("engagement_intensity_score", "Cumulative engagement intensity"),
        ("degree", "Degree"),
        ("weighted_deg", "Weighted degree"),
        ("combat_events", "Combat events"),
        ("dmg_dealt", "Damage dealt"),
    ]
    adjusted_engagement_metrics = [
        ("exposure_adjusted_engagement_score", "Exposure-adjusted engagement"),
        ("survival_residual_engagement_score", "Survival-residual engagement"),
        ("combat_events_per_min", "Combat events per survival minute"),
        ("dmg_dealt_per_min_calc", "Damage dealt per survival minute"),
        ("early_rate_engagement_score", "Early engagement rate"),
    ]
    bridge_metrics = [
        ("battle_bridge_score", "Battle bridge score"),
        ("c_betweenness", "Betweenness"),
        ("across_deg", "Across-community degree"),
        ("across_share", "Across-community share"),
        ("c_pagerank", "PageRank"),
    ]
    community_metrics = [
        ("community_embeddedness", "Community embeddedness"),
        ("community_core_score", "Community core score"),
        ("community_size", "Community size"),
        ("community_density", "Community density"),
        ("local_clustering", "Local clustering"),
    ]

    engagement_tests = pd.DataFrame([
        _quartile_test(teams, metric, label) for metric, label in engagement_metrics
    ])
    fixed_early = teams[_num(teams["survival_seconds"]) >= 300.0].copy()
    fixed_early["early_fixed_window_engagement_score"] = fixed_early["early_engagement_score"]
    adjusted_rows = [
        _quartile_test(teams, metric, label)
        for metric, label in adjusted_engagement_metrics
    ]
    adjusted_rows.append(
        _quartile_test(
            fixed_early,
            "early_fixed_window_engagement_score",
            "Early 300s fixed window among teams survived >=300s",
        )
    )
    adjusted_engagement_tests = pd.DataFrame(adjusted_rows)
    residual_metrics = [
        ("exposure_adjusted_engagement_score", "Exposure-adjusted engagement"),
        ("survival_residual_engagement_score", "Survival-residual engagement"),
        ("early_rate_engagement_score", "Early engagement rate"),
    ]
    engagement_residual_tests = pd.DataFrame([
        _quartile_test_target(
            teams,
            metric,
            label,
            target="rank_residual_after_survival",
            target_label="Rank residual after survival",
        )
        for metric, label in residual_metrics
    ])
    bridge_tests = pd.DataFrame([
        _quartile_test(teams, metric, label) for metric, label in bridge_metrics
    ])
    community_tests = pd.DataFrame([
        _quartile_test(teams, metric, label) for metric, label in community_metrics
    ])
    community_rows = _community_table(teams)
    match_comm = _match_community_table(teams)
    position_summary = _position_summary(teams)
    top_bridge = _top_teams(teams, "battle_bridge_score", n=30)
    top_engagement = _top_teams(teams, "engagement_intensity_score", n=30)

    teams.to_csv(os.path.join(out_dir, "teams_with_advanced_scores.csv"), index=False)
    engagement_tests.to_csv(os.path.join(out_dir, "engagement_intensity_tests.csv"), index=False)
    adjusted_engagement_tests.to_csv(os.path.join(out_dir, "adjusted_engagement_tests.csv"), index=False)
    engagement_residual_tests.to_csv(os.path.join(out_dir, "engagement_rank_residual_tests.csv"), index=False)
    bridge_tests.to_csv(os.path.join(out_dir, "battle_bridge_tests.csv"), index=False)
    community_tests.to_csv(os.path.join(out_dir, "louvain_community_tests.csv"), index=False)
    community_rows.to_csv(os.path.join(out_dir, "louvain_community_summary.csv"), index=False)
    match_comm.to_csv(os.path.join(out_dir, "louvain_match_summary.csv"), index=False)
    position_summary.to_csv(os.path.join(out_dir, "position_rank_summary.csv"), index=False)
    top_bridge.to_csv(os.path.join(out_dir, "top_battle_bridge_teams.csv"), index=False)
    top_engagement.to_csv(os.path.join(out_dir, "top_engagement_teams.csv"), index=False)

    _plot_score_rank(
        teams, "engagement_intensity_score", "Engagement Intensity vs Rank",
        os.path.join(fig_dir, "engagement_intensity_vs_rank.png"),
    )
    _plot_score_rank(
        teams, "exposure_adjusted_engagement_score", "Exposure-Adjusted Engagement vs Rank",
        os.path.join(fig_dir, "exposure_adjusted_engagement_vs_rank.png"),
    )
    _plot_score_rank(
        teams, "battle_bridge_score", "Battle Bridge Score vs Rank",
        os.path.join(fig_dir, "battle_bridge_vs_rank.png"),
    )
    _plot_modularity(match_comm, os.path.join(fig_dir, "modularity_distribution.png"))
    _plot_position_rank(teams, os.path.join(fig_dir, "position_rank_boxplot.png"))

    basic_path = os.path.join(dataset_dir, "basic_graph_analysis", "analysis_summary.json")
    basic_summary = {}
    if os.path.exists(basic_path):
        with open(basic_path, encoding="utf-8") as f:
            basic_summary = json.load(f)
    centrality_rows = {}
    corr_path = os.path.join(dataset_dir, "basic_graph_analysis", "centrality_correlations.csv")
    if os.path.exists(corr_path):
        corr = pd.read_csv(corr_path)
        for _, r in corr.iterrows():
            centrality_rows[r["metric"]] = {
                "pearson_r": float(r["pearson_r_vs_rank"]),
                "partial_r": float(r["partial_r_vs_rank_given_survival"]),
            }
    basic_summary["centrality_rows"] = centrality_rows

    report = _advanced_report(
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        engagement_tests=engagement_tests,
        adjusted_engagement_tests=adjusted_engagement_tests,
        engagement_residual_tests=engagement_residual_tests,
        bridge_tests=bridge_tests,
        community_tests=community_tests,
        match_comm=match_comm,
        position_summary=position_summary,
    )
    with open(os.path.join(out_dir, "advanced_network_report.md"), "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n")

    paper = _paper_report(
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        matches=matches,
        teams=teams,
        metadata=metadata,
        engagement_tests=engagement_tests,
        adjusted_engagement_tests=adjusted_engagement_tests,
        engagement_residual_tests=engagement_residual_tests,
        bridge_tests=bridge_tests,
        community_tests=community_tests,
        match_comm=match_comm,
        community_rows=community_rows,
        position_summary=position_summary,
        top_bridge=top_bridge,
        basic_summary=basic_summary,
    )
    with open(os.path.join(out_dir, "comprehensive_paper_report.md"), "w", encoding="utf-8") as f:
        f.write(paper)
        f.write("\n")

    summary = {
        "dataset_dir": dataset_dir,
        "out_dir": out_dir,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_matches": int(matches.shape[0]),
        "n_teams": int(teams.shape[0]),
        "engagement": engagement_tests.to_dict(orient="records"),
        "adjusted_engagement": adjusted_engagement_tests.to_dict(orient="records"),
        "engagement_rank_residual": engagement_residual_tests.to_dict(orient="records"),
        "bridge": bridge_tests.to_dict(orient="records"),
        "community": community_tests.to_dict(orient="records"),
        "louvain": {
            "modularity_q_mean": float(match_comm["modularity_q"].mean()),
            "modularity_q_median": float(match_comm["modularity_q"].median()),
            "q_gt_0_3_share": float((match_comm["modularity_q"] > 0.3).mean()),
            "q_gt_0_5_share": float((match_comm["modularity_q"] > 0.5).mean()),
            "mean_n_communities": float(match_comm["n_communities"].mean()),
            "mean_largest_community_share": float(match_comm["largest_community_share"].mean()),
        },
        "position_summary": position_summary.to_dict(orient="records"),
        "artifacts": {
            "advanced_report": os.path.join(out_dir, "advanced_network_report.md"),
            "comprehensive_report": os.path.join(out_dir, "comprehensive_paper_report.md"),
            "summary_json": os.path.join(out_dir, "advanced_analysis_summary.json"),
            "figures": fig_dir,
        },
    }
    _write_json(os.path.join(out_dir, "advanced_analysis_summary.json"), summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run advanced PUBG graph analysis and write paper report.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps({
        "out_dir": summary["out_dir"],
        "n_matches": summary["n_matches"],
        "n_teams": summary["n_teams"],
        "advanced_report": summary["artifacts"]["advanced_report"],
        "comprehensive_report": summary["artifacts"]["comprehensive_report"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
