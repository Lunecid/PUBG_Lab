"""
Phase-wise Hunter/Sniper/Traveler/Camper strategy analysis.

The match-level strategy analysis assigns one static type to each team.  This
script instead builds team-phase rows so a team can change type as the match
progresses.
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
from collections import defaultdict
from datetime import datetime
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from pubg_lab.analysis.network.config.config import LANDING_OFFSET_S, SNAPSHOT_INTERVAL_SEC
from pubg_lab.analysis.network.data.data_loader import build_match_data
from pubg_lab.analysis.network.data.db_loader import load_match_damage, load_phase_intervals


DEFAULT_DATASET = os.path.join(
    "projects/network_analysis", "outputs", "analysis_datasets", "Baltic_Main_squad_n1000"
)
TYPE_ORDER = ["Hunter", "Sniper", "Traveler", "Camper"]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    try:
        if den and math.isfinite(float(den)):
            return float(num) / float(den)
    except (TypeError, ValueError):
        pass
    return default


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _z_within(df: pd.DataFrame, group_cols: list[str], col: str, out_col: str) -> pd.DataFrame:
    out = df.copy()
    out[out_col] = 0.0
    for _, idx in out.groupby(group_cols).groups.items():
        vals = _num(out.loc[idx, col]).fillna(0.0)
        std = float(vals.std(ddof=0))
        if std < 1e-12:
            out.loc[idx, out_col] = 0.0
        else:
            out.loc[idx, out_col] = (vals - float(vals.mean())) / std
    return out


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [_clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: str, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean_json(data), f, ensure_ascii=False, indent=2)


def _phase_for_elapsed(intervals: list[dict[str, Any]], elapsed: float) -> dict[str, Any] | None:
    for phase in intervals:
        if float(phase["phase_start_s"]) <= elapsed < float(phase["phase_end_s"]):
            return phase
    return None


def _movement_stats(
    points: pd.DataFrame,
    *,
    active_threshold_m: float,
) -> dict[str, Any]:
    if points.empty:
        return {
            "phase_trajectory_m": 0.0,
            "phase_active_steps": 0,
            "phase_step_count": 0,
            "phase_active_frame_rate": 0.0,
            "phase_position_std_m": 0.0,
        }
    pts = points.sort_values("t")
    coords = list(zip(_num(pts["centroid_x"]), _num(pts["centroid_y"])))
    trajectory = 0.0
    active = 0
    steps = 0
    for prev, cur in zip(coords, coords[1:]):
        if any(pd.isna(v) for v in (*prev, *cur)):
            continue
        d = _dist(prev, cur)
        trajectory += d
        steps += 1
        if d >= active_threshold_m:
            active += 1
    xs = _num(pts["centroid_x"]).dropna().to_numpy(dtype=float)
    ys = _num(pts["centroid_y"]).dropna().to_numpy(dtype=float)
    if len(xs) and len(ys):
        pos_std = float(np.sqrt(xs.var(ddof=0) + ys.var(ddof=0)))
    else:
        pos_std = 0.0
    return {
        "phase_trajectory_m": float(trajectory),
        "phase_active_steps": int(active),
        "phase_step_count": int(steps),
        "phase_active_frame_rate": _safe_div(active, steps),
        "phase_position_std_m": pos_std,
    }


def _combat_rows_for_match(
    *,
    static: dict[str, Any],
    damage_events: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
    observed_min_by_phase_team: dict[tuple[int, int], float],
) -> tuple[
    dict[tuple[int, int], dict[str, Any]], dict[int, dict[tuple[int, int], dict[str, float]]]
]:
    account_to_team = {p["account_id"]: int(p["team_idx"]) for p in static.get("players", [])}
    combat = defaultdict(
        lambda: {
            "phase_damage_dealt": 0.0,
            "phase_damage_taken": 0.0,
            "phase_damage_event_dealt_count": 0,
            "phase_damage_event_taken_count": 0,
            "phase_combat_bins": set(),
            "phase_enemy_teams": set(),
        }
    )
    edge_by_phase: dict[int, dict[tuple[int, int], dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"damage": 0.0, "events": 0.0})
    )

    for event in damage_events:
        elapsed = float(event.get("elapsed") or 0.0)
        phase = _phase_for_elapsed(intervals, elapsed)
        if not phase:
            continue
        phase_i = int(phase["phase"])
        attacker_team = account_to_team.get(event.get("attacker"))
        victim_team = account_to_team.get(event.get("victim"))
        if attacker_team is None or victim_team is None or attacker_team == victim_team:
            continue
        damage = float(event.get("damage") or 0.0)
        key_a = (phase_i, int(attacker_team))
        key_v = (phase_i, int(victim_team))
        combat[key_a]["phase_damage_dealt"] += damage
        combat[key_a]["phase_damage_event_dealt_count"] += 1
        combat[key_a]["phase_enemy_teams"].add(int(victim_team))
        combat[key_v]["phase_damage_taken"] += damage
        combat[key_v]["phase_damage_event_taken_count"] += 1
        combat[key_v]["phase_enemy_teams"].add(int(attacker_team))
        bin_i = int(math.floor((elapsed - float(phase["phase_start_s"])) / SNAPSHOT_INTERVAL_SEC))
        combat[key_a]["phase_combat_bins"].add(bin_i)
        combat[key_v]["phase_combat_bins"].add(bin_i)
        edge_key = tuple(sorted((int(attacker_team), int(victim_team))))
        edge_by_phase[phase_i][edge_key]["damage"] += damage
        edge_by_phase[phase_i][edge_key]["events"] += 1.0

    out: dict[tuple[int, int], dict[str, Any]] = {}
    for key, vals in combat.items():
        observed_min = max(float(observed_min_by_phase_team.get(key, 0.0)), 1.0 / 60.0)
        event_count = int(
            vals["phase_damage_event_dealt_count"] + vals["phase_damage_event_taken_count"]
        )
        out[key] = {
            "phase_damage_dealt": float(vals["phase_damage_dealt"]),
            "phase_damage_taken": float(vals["phase_damage_taken"]),
            "phase_damage_event_dealt_count": int(vals["phase_damage_event_dealt_count"]),
            "phase_damage_event_taken_count": int(vals["phase_damage_event_taken_count"]),
            "phase_damage_event_involved_count": event_count,
            "phase_combat_event_rate_per_min": _safe_div(event_count, observed_min),
            "phase_combat_active_frames": len(vals["phase_combat_bins"]),
            "phase_unique_enemy_teams": len(vals["phase_enemy_teams"]),
        }
    return out, edge_by_phase


def _add_graph_metrics(
    rows: list[dict[str, Any]],
    *,
    edge_by_phase: dict[int, dict[tuple[int, int], dict[str, float]]],
) -> None:
    rows_by_phase: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_phase[int(row["phase"])].append(row)

    for phase_i, phase_rows in rows_by_phase.items():
        node_ids = [int(r["team_idx"]) for r in phase_rows]
        node_set = set(node_ids)
        graph = nx.Graph()
        graph.add_nodes_from(node_ids)
        for (u, v), vals in edge_by_phase.get(phase_i, {}).items():
            if u in node_set and v in node_set:
                graph.add_edge(
                    int(u), int(v), weight=float(vals["damage"]), events=float(vals["events"])
                )

        degree = dict(graph.degree())
        weighted = dict(graph.degree(weight="weight"))
        try:
            pagerank = (
                nx.pagerank(graph, weight="weight")
                if graph.number_of_edges()
                else {n: 0.0 for n in graph.nodes}
            )
        except Exception:
            pagerank = {n: 0.0 for n in graph.nodes}
        try:
            betweenness = nx.betweenness_centrality(graph, weight=None, normalized=True)
        except Exception:
            betweenness = {n: 0.0 for n in graph.nodes}

        communities: list[set[int]]
        if graph.number_of_edges():
            try:
                communities = [
                    set(c)
                    for c in nx.community.louvain_communities(graph, weight="weight", seed=13)
                ]
            except Exception:
                communities = [set(c) for c in nx.connected_components(graph)]
        else:
            communities = [{n} for n in graph.nodes]
        comm_id = {}
        for i, comm in enumerate(communities):
            for node in comm:
                comm_id[int(node)] = i

        within = defaultdict(float)
        across = defaultdict(float)
        for u, v, data in graph.edges(data=True):
            w = float(data.get("weight") or 0.0)
            if comm_id.get(int(u)) == comm_id.get(int(v)):
                within[int(u)] += w
                within[int(v)] += w
            else:
                across[int(u)] += w
                across[int(v)] += w

        for row in phase_rows:
            ti = int(row["team_idx"])
            observed_min = max(float(row.get("phase_observed_s") or 0.0) / 60.0, 1.0 / 60.0)
            total_comm = within[ti] + across[ti]
            row.update(
                {
                    "phase_degree": int(degree.get(ti, 0)),
                    "phase_weighted_deg": float(weighted.get(ti, 0.0)),
                    "phase_degree_per_min": _safe_div(degree.get(ti, 0), observed_min),
                    "phase_weighted_deg_per_min": _safe_div(weighted.get(ti, 0.0), observed_min),
                    "phase_c_pagerank": float(pagerank.get(ti, 0.0)),
                    "phase_c_betweenness": float(betweenness.get(ti, 0.0)),
                    "phase_community_id": int(comm_id.get(ti, -1)),
                    "phase_within_deg": float(within[ti]),
                    "phase_across_deg": float(across[ti]),
                    "phase_across_share": _safe_div(across[ti], total_comm),
                    "phase_community_embeddedness": _safe_div(within[ti], total_comm),
                }
            )


def build_phase_features(
    dataset_dir: str,
    *,
    max_phase: int,
    min_phase_observed_s: float,
    movement_active_threshold_m: float,
    landing_offset_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = pd.read_csv(os.path.join(dataset_dir, "matches.csv"))
    centroids = pd.read_csv(os.path.join(dataset_dir, "team_centroids.csv"))
    st_path = os.path.join(dataset_dir, "st_hypothesis_analysis", "st_team_features.csv")
    if os.path.exists(st_path):
        outcomes = pd.read_csv(st_path)
    else:
        outcomes = pd.read_csv(os.path.join(dataset_dir, "teams.csv"))

    outcome_cols = [
        "match_id",
        "team_idx",
        "team_id",
        "final_rank",
        "n_members",
        "survival_seconds",
        "ranked_rp_proxy",
        "ranked_placement_percentile",
        "team_kills",
        "team_assists",
    ]
    outcomes = outcomes[[c for c in outcome_cols if c in outcomes.columns]].copy()
    outcomes["match_id"] = outcomes["match_id"].astype(str)
    outcome_lookup = {
        (str(r.match_id), int(r.team_idx)): r for r in outcomes.itertuples(index=False)
    }
    centroid_groups = {str(k): g.copy() for k, g in centroids.groupby("match_id")}

    rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    total = int(matches.shape[0])

    for i, match in enumerate(matches.itertuples(index=False), start=1):
        match_id = str(match.match_id)
        if i == 1 or i % 10 == 0 or i == total:
            print(f"[{i}/{total}] {match_id}")

        intervals = load_phase_intervals(
            match_id,
            match_duration_s=float(getattr(match, "duration_s", 0.0) or 0.0),
            landing_offset_s=landing_offset_s,
        )
        intervals = [p for p in intervals if int(p["phase"]) <= int(max_phase)]
        if not intervals:
            continue

        md = build_match_data(str(match.pt_path), fetch_damage=False, stride=1)
        static = md["static"]
        damage_pack = load_match_damage(match_id)
        damage_events = damage_pack.get("events") or []
        match_centroids = centroid_groups.get(match_id, pd.DataFrame())

        observed_min_by_key: dict[tuple[int, int], float] = {}
        match_rows: list[dict[str, Any]] = []

        for phase in intervals:
            phase_i = int(phase["phase"])
            start_s = float(phase["phase_start_s"])
            end_s = float(phase["phase_end_s"])
            phase_duration_s = end_s - start_s
            for team_idx in range(int(getattr(match, "n_teams", static.get("n_teams", 0)) or 0)):
                base = outcome_lookup.get((match_id, int(team_idx)))
                if base is None:
                    continue
                survival_s = float(getattr(base, "survival_seconds", 0.0) or 0.0)
                observed_s = max(0.0, min(survival_s, end_s) - start_s)
                if observed_s < float(min_phase_observed_s):
                    continue
                pts = match_centroids[
                    (match_centroids["team_idx"] == int(team_idx))
                    & (_num(match_centroids["t"]) >= start_s)
                    & (_num(match_centroids["t"]) < min(end_s, survival_s + 1e-9))
                ]
                mv = _movement_stats(pts, active_threshold_m=movement_active_threshold_m)
                observed_min = observed_s / 60.0
                key = (phase_i, int(team_idx))
                observed_min_by_key[key] = observed_min
                row = {
                    "match_id": match_id,
                    "map_name": getattr(match, "map_name", None),
                    "game_mode": getattr(match, "game_mode", None),
                    "phase": phase_i,
                    "phase_start_s": start_s,
                    "phase_end_s": end_s,
                    "phase_duration_s": phase_duration_s,
                    "team_idx": int(team_idx),
                    "team_id": int(getattr(base, "team_id", team_idx) or team_idx),
                    "final_rank": int(getattr(base, "final_rank", 0) or 0),
                    "n_members": int(getattr(base, "n_members", 0) or 0),
                    "survival_seconds": survival_s,
                    "phase_observed_s": observed_s,
                    "phase_observed_min": observed_min,
                    "survived_phase": bool(survival_s >= end_s),
                    "ranked_rp_proxy": float(getattr(base, "ranked_rp_proxy", np.nan)),
                    "ranked_placement_percentile": float(
                        getattr(base, "ranked_placement_percentile", np.nan)
                    ),
                    "team_kills": float(getattr(base, "team_kills", np.nan)),
                    "team_assists": float(getattr(base, "team_assists", np.nan)),
                    **mv,
                }
                row["phase_movement_per_min"] = _safe_div(row["phase_trajectory_m"], observed_min)
                match_rows.append(row)

        combat, edge_by_phase = _combat_rows_for_match(
            static=static,
            damage_events=damage_events,
            intervals=intervals,
            observed_min_by_phase_team=observed_min_by_key,
        )
        _add_graph_metrics(match_rows, edge_by_phase=edge_by_phase)

        for row in match_rows:
            key = (int(row["phase"]), int(row["team_idx"]))
            c = combat.get(key, {})
            event_count = int(c.get("phase_damage_event_involved_count", 0))
            expected_frames = max(
                1, int(math.ceil(float(row["phase_observed_s"]) / SNAPSHOT_INTERVAL_SEC))
            )
            active_frames = int(c.get("phase_combat_active_frames", 0))
            row.update(
                {
                    "phase_damage_dealt": float(c.get("phase_damage_dealt", 0.0)),
                    "phase_damage_taken": float(c.get("phase_damage_taken", 0.0)),
                    "phase_damage_event_dealt_count": int(
                        c.get("phase_damage_event_dealt_count", 0)
                    ),
                    "phase_damage_event_taken_count": int(
                        c.get("phase_damage_event_taken_count", 0)
                    ),
                    "phase_damage_event_involved_count": event_count,
                    "phase_combat_event_rate_per_min": _safe_div(
                        event_count, row["phase_observed_min"]
                    ),
                    "phase_combat_active_frames": active_frames,
                    "phase_combat_exposure_frames": expected_frames,
                    "phase_combat_st_participation_rate": _safe_div(active_frames, expected_frames),
                    "phase_unique_enemy_teams": int(c.get("phase_unique_enemy_teams", 0)),
                }
            )
            rows.append(row)

        for phase_i, edges in edge_by_phase.items():
            for (u, v), vals in edges.items():
                edge_rows.append(
                    {
                        "match_id": match_id,
                        "phase": int(phase_i),
                        "source_team_idx": int(u),
                        "target_team_idx": int(v),
                        "damage": float(vals["damage"]),
                        "events": int(vals["events"]),
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(edge_rows)


def add_phase_strategy_types(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    if df.empty:
        return df

    df["phase_battle_bridge_score"] = 0.0
    for _, idx in df.groupby(["match_id", "phase"]).groups.items():
        b = _num(df.loc[idx, "phase_c_betweenness"]).fillna(0.0)
        across = np.log1p(_num(df.loc[idx, "phase_across_deg"]).fillna(0.0))
        share = _num(df.loc[idx, "phase_across_share"]).fillna(0.0)
        parts = []
        for vals in (b, across, share):
            std = float(vals.std(ddof=0))
            parts.append((vals - float(vals.mean())) / std if std >= 1e-12 else vals * 0.0)
        df.loc[idx, "phase_battle_bridge_score"] = parts[0] + parts[1] + parts[2]

    fight_components = [
        "phase_combat_st_participation_rate",
        "phase_combat_event_rate_per_min",
        "phase_degree_per_min",
        "phase_weighted_deg_per_min",
        "phase_c_pagerank",
        "phase_battle_bridge_score",
    ]
    active_components = [
        "phase_movement_per_min",
        "phase_active_frame_rate",
        "phase_position_std_m",
    ]
    for col in fight_components + active_components:
        df = _z_within(df, ["match_id", "phase"], col, f"z_{col}")

    df["phase_fight_score"] = df[[f"z_{c}" for c in fight_components]].mean(axis=1)
    df["phase_active_score"] = df[[f"z_{c}" for c in active_components]].mean(axis=1)
    df["phase_fight_high"] = False
    df["phase_active_high"] = False
    for _, idx in df.groupby(["match_id", "phase"]).groups.items():
        fight_med = float(_num(df.loc[idx, "phase_fight_score"]).median())
        active_med = float(_num(df.loc[idx, "phase_active_score"]).median())
        df.loc[idx, "phase_fight_high"] = _num(df.loc[idx, "phase_fight_score"]) >= fight_med
        df.loc[idx, "phase_active_high"] = _num(df.loc[idx, "phase_active_score"]) >= active_med

    conditions = [
        df["phase_fight_high"] & df["phase_active_high"],
        df["phase_fight_high"] & ~df["phase_active_high"],
        ~df["phase_fight_high"] & df["phase_active_high"],
        ~df["phase_fight_high"] & ~df["phase_active_high"],
    ]
    df["phase_strategy_type"] = np.select(
        conditions,
        ["Hunter", "Sniper", "Traveler", "Camper"],
        default="Unknown",
    )
    return df


def summarize(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if features.empty:
        return pd.DataFrame(), pd.DataFrame()
    metric_cols = [
        "final_rank",
        "ranked_rp_proxy",
        "team_kills",
        "team_assists",
        "survival_seconds",
        "phase_observed_s",
        "survived_phase",
        "phase_fight_score",
        "phase_active_score",
        "phase_degree",
        "phase_c_betweenness",
        "phase_c_pagerank",
        "phase_battle_bridge_score",
        "phase_movement_per_min",
        "phase_combat_st_participation_rate",
    ]
    rows = []
    for (phase, typ), g in features.groupby(["phase", "phase_strategy_type"], dropna=False):
        row = {"phase": int(phase), "phase_strategy_type": typ, "n": int(g.shape[0])}
        for col in metric_cols:
            if col not in g.columns:
                continue
            vals = _num(g[col]).dropna()
            row[f"{col}_mean"] = float(vals.mean()) if not vals.empty else np.nan
        row["top10_rate"] = float((_num(g["final_rank"]) <= 10).mean())
        rows.append(row)
    phase_type_summary = pd.DataFrame(rows).sort_values(["phase", "phase_strategy_type"])

    trans_rows = []
    ordered = features.sort_values(["match_id", "team_idx", "phase"])
    for (_, _), g in ordered.groupby(["match_id", "team_idx"]):
        prev = None
        for row in g.itertuples(index=False):
            cur = (int(row.phase), str(row.phase_strategy_type))
            if prev is not None:
                trans_rows.append(
                    {
                        "from_phase": prev[0],
                        "to_phase": cur[0],
                        "from_type": prev[1],
                        "to_type": cur[1],
                        "transition": f"{prev[1]} -> {cur[1]}",
                    }
                )
            prev = cur
    transitions = pd.DataFrame(trans_rows)
    if not transitions.empty:
        transitions = (
            transitions.groupby(["from_phase", "to_phase", "from_type", "to_type", "transition"])
            .size()
            .reset_index(name="n")
            .sort_values(["from_phase", "to_phase", "n"], ascending=[True, True, False])
        )
    return phase_type_summary, transitions


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(x):
        return "-"
    return f"{x:.{digits}f}"


def make_report(
    *,
    features: pd.DataFrame,
    phase_type_summary: pd.DataFrame,
    transitions: pd.DataFrame,
    dataset_dir: str,
    out_dir: str,
) -> str:
    lines = [
        "# Phase-wise PUBG Strategy Type Analysis",
        "",
        "## 1. 분석 단위",
        "",
        (
            "본 분석은 정적 team 단위가 아니라 `team × phase` 단위로 Hunter/Sniper/"
            "Traveler/Camper 유형을 다시 계산한다. 따라서 같은 팀도 phase가 바뀌면 "
            "다른 전략 유형으로 전환될 수 있다."
        ),
        "",
        (
            "여기서 `n`은 독립적인 새 팀 수가 아니라 team-phase 관측치 수다. "
            "같은 팀이 여러 팀으로 분화되는 것이 아니라, 같은 `match_id × team_idx`가 "
            "phase별 상태 기록으로 반복 관측되는 long-format 구조다. 따라서 여러 phase의 "
            "`n`을 합산한 값은 고유 팀 수가 아니다."
        ),
        "",
        (
            "`final_rank`는 phase별 순위가 아니라 매치 종료 후 결정되는 최종 순위다. "
            "따라서 아래의 mean rank는 해당 phase에서 특정 유형으로 관측된 팀들의 "
            "최종 평균 순위이며, phase 내부 실시간 순위가 아니다."
        ),
        "",
        f"- Dataset: `{dataset_dir}`",
        f"- Output: `{out_dir}`",
        f"- team-phase rows: {features.shape[0]:,}",
        f"- matches: {features['match_id'].nunique() if not features.empty else 0:,}",
        "",
        "## 2. 유형 정의",
        "",
        (
            "각 `match_id × phase` 안에서 FightScore와 ActiveScore를 표준화한 뒤, "
            "해당 phase의 중앙값 기준으로 high/low를 나눴다."
        ),
        "",
        "| 유형 | FightScore | ActiveScore | 의미 |",
        "|---|---:|---:|---|",
        "| Hunter | 높음 | 높음 | 해당 phase에서 교전성과 이동/공간 활동성이 모두 높은 팀 |",
        "| Sniper | 높음 | 낮음 | 해당 phase에서 교전성은 높지만 활동성은 낮은 팀 |",
        "| Traveler | 낮음 | 높음 | 해당 phase에서 교전성은 낮고 이동/공간 활동성이 높은 팀 |",
        "| Camper | 낮음 | 낮음 | 해당 phase에서 교전성과 활동성이 모두 낮은 팀 |",
        "",
        "## 3. Phase별 유형 성과 요약",
        "",
        "| phase | type | team-phase n | final mean rank | RP proxy | kills | survived phase | fight | active |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in phase_type_summary.itertuples(index=False):
        lines.append(
            f"| {int(row.phase)} | {row.phase_strategy_type} | {int(row.n):,} | "
            f"{_fmt(getattr(row, 'final_rank_mean', np.nan))} | "
            f"{_fmt(getattr(row, 'ranked_rp_proxy_mean', np.nan))} | "
            f"{_fmt(getattr(row, 'team_kills_mean', np.nan))} | "
            f"{_fmt(getattr(row, 'survived_phase_mean', np.nan))} | "
            f"{_fmt(getattr(row, 'phase_fight_score_mean', np.nan))} | "
            f"{_fmt(getattr(row, 'phase_active_score_mean', np.nan))} |"
        )

    lines.extend(
        [
            "",
            "## 4. 주요 전환",
            "",
            "| from phase | to phase | transition | n |",
            "|---:|---:|---|---:|",
        ]
    )
    if transitions.empty:
        lines.append("| - | - | - | 0 |")
    else:
        for row in transitions.head(30).itertuples(index=False):
            lines.append(
                f"| {int(row.from_phase)} | {int(row.to_phase)} | {row.transition} | {int(row.n):,} |"
            )

    lines.extend(
        [
            "",
            "## 5. 해석",
            "",
            (
                "이 결과는 기존 정적 4유형 분석과 달리, 전략 유형을 고정된 팀 성향으로 보지 않고 "
                "phase별 상태로 해석한다. 즉 Hunter, Sniper, Traveler, Camper는 한 팀의 "
                "영구적 라벨이 아니라 특정 phase에서의 전투-이동 조합이다."
            ),
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = args.dataset
    out_dir = args.out or os.path.join(dataset_dir, "phase_strategy_type_analysis")
    os.makedirs(out_dir, exist_ok=True)

    features, edges = build_phase_features(
        dataset_dir,
        max_phase=args.max_phase,
        min_phase_observed_s=args.min_phase_observed_s,
        movement_active_threshold_m=args.movement_active_threshold_m,
        landing_offset_s=args.landing_offset_s,
    )
    features = add_phase_strategy_types(features)
    phase_type_summary, transitions = summarize(features)

    features_path = os.path.join(out_dir, "phase_team_features.csv")
    edges_path = os.path.join(out_dir, "phase_team_edges.csv")
    summary_path = os.path.join(out_dir, "phase_type_summary.csv")
    transitions_path = os.path.join(out_dir, "phase_type_transitions.csv")
    report_path = os.path.join(out_dir, "phase_strategy_type_report.md")

    features.to_csv(features_path, index=False)
    edges.to_csv(edges_path, index=False)
    phase_type_summary.to_csv(summary_path, index=False)
    transitions.to_csv(transitions_path, index=False)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(
            make_report(
                features=features,
                phase_type_summary=phase_type_summary,
                transitions=transitions,
                dataset_dir=dataset_dir,
                out_dir=out_dir,
            )
        )
        f.write("\n")

    summary = {
        "dataset_dir": dataset_dir,
        "out_dir": out_dir,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_matches": int(features["match_id"].nunique()) if not features.empty else 0,
        "n_team_phase_rows": int(features.shape[0]),
        "n_phase_edges": int(edges.shape[0]),
        "artifacts": {
            "features": features_path,
            "edges": edges_path,
            "summary": summary_path,
            "transitions": transitions_path,
            "report": report_path,
        },
    }
    _write_json(os.path.join(out_dir, "phase_strategy_type_summary.json"), summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase-wise PUBG strategy type analysis.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-phase", type=int, default=6)
    parser.add_argument("--min-phase-observed-s", type=float, default=30.0)
    parser.add_argument("--movement-active-threshold-m", type=float, default=25.0)
    parser.add_argument("--landing-offset-s", type=float, default=LANDING_OFFSET_S)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(
        json.dumps(
            {
                "out_dir": summary["out_dir"],
                "n_matches": summary["n_matches"],
                "n_team_phase_rows": summary["n_team_phase_rows"],
                "n_phase_edges": summary["n_phase_edges"],
                "report": summary["artifacts"]["report"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
