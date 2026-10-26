"""
Network-based Hunter/Sniper/Traveler/Camper strategy-type analysis.

This script adapts the four strategy types from the PUBG paper to our
team-level ST combat network data.  The types are not copied verbatim:

  - FightScore is defined from team combat-network exposure and centrality.
  - ActiveScore is defined from post-landing movement, spatial spread, and
    dense-area activity.

Kills and assists are intentionally kept out of the type score because they are
part of the ranked RP proxy outcome.  They are analyzed as outcomes by type.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

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
from scipy.stats import kruskal, mannwhitneyu


DEFAULT_DATASET = os.path.join(
    "projects/network_analysis", "outputs", "analysis_datasets", "Baltic_Main_squad_n100"
)

TYPE_ORDER = ["Hunter", "Sniper", "Traveler", "Camper"]

FIGHT_COMPONENTS = [
    ("combat_st_participation_rate", "ST combat frame participation"),
    ("combat_event_rate_per_min", "Damage event involvement/min"),
    ("degree_per_min", "Combat-network opponent degree/min"),
    ("weighted_deg_per_min", "Weighted combat-network degree/min"),
    ("c_pagerank", "Combat-network PageRank"),
    ("battle_bridge_score", "Battle bridge score"),
]

ACTIVE_COMPONENTS = [
    ("movement_per_min", "Post-landing movement/min"),
    ("movement_active_frame_rate", "Active movement frame rate"),
    ("position_std_m", "Spatial spread"),
    ("landing_density_score", "Landing density activity"),
    ("landing_neighbor_count", "Nearby landing teams"),
]

GRAPH_SUMMARY_METRICS = [
    "degree",
    "weighted_deg",
    "degree_per_min",
    "weighted_deg_per_min",
    "c_betweenness",
    "c_pagerank",
    "battle_bridge_score",
    "community_embeddedness",
    "local_clustering",
]

OUTCOME_METRICS = [
    "final_rank",
    "ranked_rp_proxy",
    "ranked_placement_percentile",
    "team_kills",
    "team_assists",
    "survival_seconds",
]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(x):
        return "-"
    return f"{x:.{digits}f}"


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return _clean_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean_json(payload), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def _within_match_z(df: pd.DataFrame, col: str) -> pd.Series:
    out = pd.Series(0.0, index=df.index, dtype=float)
    vals_all = _num(df[col]).fillna(0.0)
    for _match_id, idx in df.groupby("match_id").groups.items():
        vals = vals_all.loc[idx]
        std = float(vals.std(ddof=0))
        if std < 1e-12:
            out.loc[idx] = 0.0
        else:
            out.loc[idx] = (vals - float(vals.mean())) / std
    return out


def _mean_and_share(series: pd.Series) -> tuple[float, int]:
    vals = _num(series).dropna()
    if vals.empty:
        return float("nan"), 0
    return float(vals.mean()), int(vals.shape[0])


def load_features(dataset_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    st_path = os.path.join(dataset_dir, "st_hypothesis_analysis", "st_team_features.csv")
    adv_path = os.path.join(
        dataset_dir, "advanced_graph_analysis", "teams_with_advanced_scores.csv"
    )
    teams_path = os.path.join(dataset_dir, "teams.csv")
    edges_path = os.path.join(dataset_dir, "team_edges.csv")

    st = pd.read_csv(st_path)
    graph_src = pd.read_csv(adv_path if os.path.exists(adv_path) else teams_path)
    edges = pd.read_csv(edges_path)

    graph_cols = [
        "match_id",
        "team_idx",
        "degree",
        "weighted_deg",
        "c_degree",
        "c_closeness",
        "c_betweenness",
        "c_eigenvector",
        "c_pagerank",
        "community_size",
        "community_density",
        "within_deg",
        "across_deg",
        "local_clustering",
        "match_modularity_q",
        "degree_per_min",
        "weighted_deg_per_min",
        "combat_events_per_min",
        "dmg_dealt_per_min_calc",
        "across_share",
        "community_embeddedness",
        "battle_bridge_score",
        "community_core_score",
    ]
    graph_cols = [c for c in graph_cols if c in graph_src.columns]
    df = st.merge(
        graph_src[graph_cols],
        on=["match_id", "team_idx"],
        how="left",
        validate="one_to_one",
    )
    return df, edges


def add_strategy_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    used_fight: list[str] = []
    used_active: list[str] = []

    for col, _label in FIGHT_COMPONENTS:
        if col in out.columns:
            out[f"z_{col}"] = _within_match_z(out, col)
            used_fight.append(f"z_{col}")
    for col, _label in ACTIVE_COMPONENTS:
        if col in out.columns:
            out[f"z_{col}"] = _within_match_z(out, col)
            used_active.append(f"z_{col}")

    out["network_fight_score"] = out[used_fight].mean(axis=1) if used_fight else 0.0
    out["network_active_score"] = out[used_active].mean(axis=1) if used_active else 0.0
    out["fight_high"] = False
    out["active_high"] = False

    for _match_id, idx in out.groupby("match_id").groups.items():
        fight_med = float(out.loc[idx, "network_fight_score"].median())
        active_med = float(out.loc[idx, "network_active_score"].median())
        out.loc[idx, "fight_high"] = out.loc[idx, "network_fight_score"] > fight_med
        out.loc[idx, "active_high"] = out.loc[idx, "network_active_score"] > active_med

    conditions = [
        out["fight_high"] & out["active_high"],
        out["fight_high"] & ~out["active_high"],
        ~out["fight_high"] & out["active_high"],
        ~out["fight_high"] & ~out["active_high"],
    ]
    out["strategy_type"] = np.select(
        conditions,
        TYPE_ORDER,
        default="Unclassified",
    )
    return out


def summarize_by_type(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for t in TYPE_ORDER:
        g = df[df["strategy_type"] == t]
        row: dict[str, Any] = {
            "strategy_type": t,
            "n": int(g.shape[0]),
            "share": float(g.shape[0] / df.shape[0]) if df.shape[0] else 0.0,
            "fight_score_mean": float(g["network_fight_score"].mean()),
            "active_score_mean": float(g["network_active_score"].mean()),
            "top10_rate": float((_num(g["final_rank"]) <= 10).mean()),
            "top5_rate": float((_num(g["final_rank"]) <= 5).mean()),
        }
        for col in (
            OUTCOME_METRICS
            + GRAPH_SUMMARY_METRICS
            + [
                "movement_per_min",
                "movement_active_frame_rate",
                "landing_density_score",
                "distance_to_hotzone_m",
                "combat_st_participation_rate",
                "combat_event_rate_per_min",
            ]
        ):
            if col in g.columns:
                row[f"{col}_mean"] = float(_num(g[col]).mean())
                row[f"{col}_median"] = float(_num(g[col]).median())
        rows.append(row)
    return pd.DataFrame(rows)


def type_metric_tests(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    metrics = (
        OUTCOME_METRICS
        + GRAPH_SUMMARY_METRICS
        + [
            "network_fight_score",
            "network_active_score",
            "movement_per_min",
            "combat_st_participation_rate",
        ]
    )
    for metric in metrics:
        if metric not in df.columns:
            continue
        groups = [
            _num(df.loc[df["strategy_type"] == t, metric]).dropna().to_numpy(dtype=float)
            for t in TYPE_ORDER
        ]
        valid_groups = [g for g in groups if g.size > 0]
        if len(valid_groups) >= 2:
            try:
                stat, p = kruskal(*valid_groups)
            except Exception:
                stat, p = float("nan"), float("nan")
        else:
            stat, p = float("nan"), float("nan")
        rows.append(
            {
                "metric": metric,
                "test": "kruskal",
                "statistic": float(stat),
                "p_value": float(p),
            }
        )

    for metric in ["final_rank", "ranked_rp_proxy", "team_kills", "survival_seconds"]:
        if metric not in df.columns:
            continue
        for i, a in enumerate(TYPE_ORDER):
            for b in TYPE_ORDER[i + 1 :]:
                va = _num(df.loc[df["strategy_type"] == a, metric]).dropna()
                vb = _num(df.loc[df["strategy_type"] == b, metric]).dropna()
                if va.empty or vb.empty:
                    continue
                try:
                    res = mannwhitneyu(va, vb, alternative="two-sided")
                    u = float(res.statistic)
                    p = float(res.pvalue)
                except Exception:
                    u = float("nan")
                    p = float("nan")
                pair_rows.append(
                    {
                        "metric": metric,
                        "type_a": a,
                        "type_b": b,
                        "type_a_mean": float(va.mean()),
                        "type_b_mean": float(vb.mean()),
                        "delta_b_minus_a": float(vb.mean() - va.mean()),
                        "mw_u": u,
                        "mw_p": p,
                        "direction_note": (
                            "for final_rank lower is better; for ranked_rp_proxy/kills/survival higher is better"
                        ),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def edge_mixing(df: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    full_edges = edges[edges["window"].fillna("full") == "full"].copy()
    node_type = df[["match_id", "team_idx", "strategy_type"]].copy()
    src = node_type.rename(
        columns={
            "team_idx": "source_team_idx",
            "strategy_type": "source_type",
        }
    )
    dst = node_type.rename(
        columns={
            "team_idx": "target_team_idx",
            "strategy_type": "target_type",
        }
    )
    e = full_edges.merge(src, on=["match_id", "source_team_idx"], how="left")
    e = e.merge(dst, on=["match_id", "target_team_idx"], how="left")
    e = e.dropna(subset=["source_type", "target_type"])
    e["type_pair"] = [
        " - ".join(sorted([str(a), str(b)])) for a, b in zip(e["source_type"], e["target_type"])
    ]
    out = e.groupby("type_pair", as_index=False).agg(
        n_edges=("type_pair", "size"),
        total_events=("events", "sum"),
        total_damage=("damage", "sum"),
        mean_weight=("weight", "mean"),
    )
    total_edges = float(out["n_edges"].sum()) if not out.empty else 0.0
    total_events = float(out["total_events"].sum()) if not out.empty else 0.0
    out["edge_share"] = out["n_edges"] / total_edges if total_edges else 0.0
    out["event_share"] = out["total_events"] / total_events if total_events else 0.0
    out = out.sort_values(["n_edges", "total_events"], ascending=False)
    return out


def _write_figures(df: pd.DataFrame, summary: pd.DataFrame, out_dir: str) -> dict[str, str]:
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    paths: dict[str, str] = {}

    colors = {
        "Hunter": "#d62728",
        "Sniper": "#9467bd",
        "Traveler": "#2ca02c",
        "Camper": "#7f7f7f",
    }
    plt.figure(figsize=(7.2, 5.2))
    for t in TYPE_ORDER:
        g = df[df["strategy_type"] == t]
        plt.scatter(
            g["network_active_score"],
            g["network_fight_score"],
            s=14,
            alpha=0.45,
            label=t,
            color=colors[t],
        )
    plt.axhline(0, color="#222222", lw=0.8, alpha=0.4)
    plt.axvline(0, color="#222222", lw=0.8, alpha=0.4)
    plt.xlabel("Network ActiveScore")
    plt.ylabel("Network FightScore")
    plt.title("Strategy types in FightScore x ActiveScore space")
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    path = os.path.join(fig_dir, "strategy_type_scatter.png")
    plt.savefig(path, dpi=160)
    plt.close()
    paths["strategy_type_scatter.png"] = path

    for metric, filename, ylabel, invert in [
        ("final_rank", "strategy_type_rank_boxplot.png", "Final rank", True),
        ("ranked_rp_proxy", "strategy_type_rp_proxy_boxplot.png", "Ranked RP proxy", False),
        ("team_kills", "strategy_type_kills_boxplot.png", "Team kills", False),
    ]:
        data = [
            _num(df.loc[df["strategy_type"] == t, metric]).dropna().to_numpy(dtype=float)
            for t in TYPE_ORDER
        ]
        plt.figure(figsize=(7.0, 4.4))
        plt.boxplot(data, labels=TYPE_ORDER, showfliers=False)
        if invert:
            plt.gca().invert_yaxis()
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} by strategy type")
        plt.tight_layout()
        path = os.path.join(fig_dir, filename)
        plt.savefig(path, dpi=160)
        plt.close()
        paths[filename] = path

    metric_cols = [
        "degree_per_min_mean",
        "battle_bridge_score_mean",
        "community_embeddedness_mean",
        "movement_per_min_mean",
    ]
    available = [c for c in metric_cols if c in summary.columns]
    if available:
        plot_df = summary.set_index("strategy_type").loc[TYPE_ORDER, available]
        ax = plot_df.plot(kind="bar", figsize=(8.2, 4.8))
        ax.set_ylabel("Mean value")
        ax.set_title("Network and movement profile by strategy type")
        ax.legend(frameon=False, fontsize=8)
        plt.xticks(rotation=0)
        plt.tight_layout()
        path = os.path.join(fig_dir, "strategy_type_network_profile.png")
        plt.savefig(path, dpi=160)
        plt.close()
        paths["strategy_type_network_profile.png"] = path

    return paths


def _best_type(summary: pd.DataFrame, metric: str, *, higher: bool) -> str:
    col = f"{metric}_mean"
    if col not in summary.columns:
        return "-"
    idx = summary[col].idxmax() if higher else summary[col].idxmin()
    return str(summary.loc[idx, "strategy_type"])


def make_report(
    *,
    dataset_dir: str,
    out_dir: str,
    df: pd.DataFrame,
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    pairwise: pd.DataFrame,
    mixing: pd.DataFrame,
    figures: dict[str, str],
) -> str:
    by_type = summary.set_index("strategy_type")
    lines = [
        "# PUBG 네트워크 기반 4유형 전략 분석",
        "",
        "## 1. 분석 목적",
        "",
        (
            "본 분석은 Hunter/Sniper/Traveler/Camper 4유형을 우리 팀 단위 ST 전투 네트워크에 맞게 "
            "재정의하고, 유형별 그래프 위치와 성과 차이를 비교한다. 성과는 최종 순위와 경쟁전 RP proxy를 "
            "함께 사용한다."
        ),
        "",
        "## 2. 유형 정의",
        "",
        (
            "- Network FightScore: 10초 교전 프레임 참여율, damage event 관여율, 전투 네트워크 degree/weighted degree, "
            "PageRank, battle bridge score를 매치 내 표준화해 평균낸 값이다."
        ),
        (
            "- Network ActiveScore: 착륙 후 이동률, 활발 이동 프레임 비율, 공간 확산도, 착륙 밀도, 주변 착륙 팀 수를 "
            "매치 내 표준화해 평균낸 값이다."
        ),
        (
            "- 분류 기준: 각 매치 안에서 FightScore와 ActiveScore가 중앙값보다 높은지 낮은지로 네 유형을 나눈다."
        ),
        "",
        "| 유형 | FightScore | ActiveScore | 해석 |",
        "|---|---:|---:|---|",
        "| Hunter | 높음 | 높음 | 전투 네트워크 중심 + 활발 이동 |",
        "| Sniper | 높음 | 낮음 | 전투성은 높지만 공간 활동성은 낮음 |",
        "| Traveler | 낮음 | 높음 | 교전성은 낮고 이동/탐색성은 높음 |",
        "| Camper | 낮음 | 낮음 | 교전성도 활동성도 낮음 |",
        "",
        "## 3. 유형별 성과 요약",
        "",
        "| 유형 | n | 평균 순위 | RP proxy | 평균 킬 | Top10 | 생존초 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for t in TYPE_ORDER:
        r = by_type.loc[t]
        lines.append(
            f"| {t} | {int(r['n'])} | {_fmt(r.get('final_rank_mean'))} | "
            f"{_fmt(r.get('ranked_rp_proxy_mean'))} | {_fmt(r.get('team_kills_mean'))} | "
            f"{_fmt(r.get('top10_rate'))} | {_fmt(r.get('survival_seconds_mean'))} |"
        )

    best_rank = _best_type(summary, "final_rank", higher=False)
    best_rp = _best_type(summary, "ranked_rp_proxy", higher=True)
    best_kill = _best_type(summary, "team_kills", higher=True)

    lines.extend(
        [
            "",
            "## 4. 핵심 해석",
            "",
            (
                f"- 평균 순위가 가장 좋은 유형은 **{best_rank}**이다. "
                "즉 순위 성과는 전투성/활동성의 단순 합보다 어떤 조합을 취했는지에 따라 달라진다."
            ),
            (
                f"- 경쟁전 RP proxy가 가장 높은 유형은 **{best_rp}**이다. "
                "RP proxy는 팀 순위, 킬, 어시스트를 함께 반영하므로 순위만 볼 때와 다를 수 있다."
            ),
            (
                f"- 평균 킬이 가장 높은 유형은 **{best_kill}**이다. "
                "이 유형은 점수 보상 측면에서는 강하지만, 순위 리스크와 함께 해석해야 한다."
            ),
            "",
            "### 유형별 한줄 해석",
            "",
            (
                f"- **Hunter**: 평균 순위 {_fmt(by_type.loc['Hunter'].get('final_rank_mean'))}, "
                f"RP proxy {_fmt(by_type.loc['Hunter'].get('ranked_rp_proxy_mean'))}, "
                f"평균 킬 {_fmt(by_type.loc['Hunter'].get('team_kills_mean'))}. "
                "활발히 움직이며 전투 네트워크 중심에 들어가는 점수형 강자다."
            ),
            (
                f"- **Traveler**: 평균 순위 {_fmt(by_type.loc['Traveler'].get('final_rank_mean'))}, "
                f"RP proxy {_fmt(by_type.loc['Traveler'].get('ranked_rp_proxy_mean'))}, "
                f"평균 킬 {_fmt(by_type.loc['Traveler'].get('team_kills_mean'))}. "
                "순위는 Hunter와 비슷하게 좋지만, 킬 보상이 낮아 RP proxy는 크게 밀린다."
            ),
            (
                f"- **Camper**: 평균 순위 {_fmt(by_type.loc['Camper'].get('final_rank_mean'))}, "
                f"RP proxy {_fmt(by_type.loc['Camper'].get('ranked_rp_proxy_mean'))}. "
                "단순 회피/저활동 전략은 어느 정도 생존에는 도움이 되지만 상위권과 점수 확보에는 한계가 있다."
            ),
            (
                f"- **Sniper**: 평균 순위 {_fmt(by_type.loc['Sniper'].get('final_rank_mean'))}, "
                f"RP proxy {_fmt(by_type.loc['Sniper'].get('ranked_rp_proxy_mean'))}. "
                "교전성은 높지만 활동성이 낮은 팀은 네트워크상 위험 노출이 커지고, 성과가 가장 나빴다."
            ),
            "",
            "## 5. 네트워크 차이",
            "",
            "| 유형 | degree/min | PageRank | betweenness | battle bridge | embeddedness | 이동 m/min |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for t in TYPE_ORDER:
        r = by_type.loc[t]
        lines.append(
            f"| {t} | {_fmt(r.get('degree_per_min_mean'))} | {_fmt(r.get('c_pagerank_mean'))} | "
            f"{_fmt(r.get('c_betweenness_mean'))} | {_fmt(r.get('battle_bridge_score_mean'))} | "
            f"{_fmt(r.get('community_embeddedness_mean'))} | {_fmt(r.get('movement_per_min_mean'))} |"
        )

    if not mixing.empty:
        lines.extend(
            [
                "",
                "## 6. 유형 간 교전 엣지",
                "",
                "| 유형 쌍 | 엣지 수 | 이벤트 수 | 엣지 비중 |",
                "|---|---:|---:|---:|",
            ]
        )
        for r in mixing.head(8).itertuples(index=False):
            lines.append(
                f"| {r.type_pair} | {int(r.n_edges)} | {int(r.total_events)} | {_fmt(r.edge_share)} |"
            )

    lines.extend(
        [
            "",
            "## 7. 산출 파일",
            "",
            "- `strategy_type_team_features.csv`: 팀별 4유형, 점수, 네트워크/성과 지표",
            "- `strategy_type_summary.csv`: 유형별 요약",
            "- `strategy_type_metric_tests.csv`: 유형 간 전체 차이 검정",
            "- `strategy_type_pairwise_tests.csv`: 주요 성과 pairwise 검정",
            "- `strategy_type_edge_mixing.csv`: 유형 간 교전 엣지 혼합",
        ]
    )

    if figures:
        lines.extend(["", "## 8. 그림", ""])
        for name, path in figures.items():
            lines.append(f"- `{name}`: `{path}`")

    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = args.dataset
    out_dir = args.out or os.path.join(dataset_dir, "strategy_type_network_analysis")
    os.makedirs(out_dir, exist_ok=True)

    df, edges = load_features(dataset_dir)
    df = add_strategy_scores(df)
    summary = summarize_by_type(df)
    tests, pairwise = type_metric_tests(df)
    mixing = edge_mixing(df, edges)
    figures = _write_figures(df, summary, out_dir)

    feature_path = os.path.join(out_dir, "strategy_type_team_features.csv")
    summary_path = os.path.join(out_dir, "strategy_type_summary.csv")
    tests_path = os.path.join(out_dir, "strategy_type_metric_tests.csv")
    pairwise_path = os.path.join(out_dir, "strategy_type_pairwise_tests.csv")
    mixing_path = os.path.join(out_dir, "strategy_type_edge_mixing.csv")
    report_path = os.path.join(out_dir, "strategy_type_network_report.md")

    df.to_csv(feature_path, index=False)
    summary.to_csv(summary_path, index=False)
    tests.to_csv(tests_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)
    mixing.to_csv(mixing_path, index=False)

    report = make_report(
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        df=df,
        summary=summary,
        tests=tests,
        pairwise=pairwise,
        mixing=mixing,
        figures=figures,
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n")

    payload = {
        "dataset_dir": dataset_dir,
        "out_dir": out_dir,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_teams": int(df.shape[0]),
        "n_matches": int(df["match_id"].nunique()),
        "type_counts": df["strategy_type"].value_counts().to_dict(),
        "artifacts": {
            "team_features": feature_path,
            "summary": summary_path,
            "metric_tests": tests_path,
            "pairwise_tests": pairwise_path,
            "edge_mixing": mixing_path,
            "report": report_path,
            "figures": figures,
        },
    }
    _write_json(os.path.join(out_dir, "strategy_type_network_summary.json"), payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run network-based Hunter/Sniper/Traveler/Camper analysis."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(
        json.dumps(
            {
                "out_dir": summary["out_dir"],
                "n_matches": summary["n_matches"],
                "n_teams": summary["n_teams"],
                "report": summary["artifacts"]["report"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
