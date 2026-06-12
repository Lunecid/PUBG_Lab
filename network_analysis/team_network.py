"""
Team-level Network Builder
==========================
매치 단위로 팀을 노드, 팀간 교전을 엣지로 하는 weighted undirected graph 구성.

- Node V  = 팀 집합 (team_idx)
- Edge E  = 두 팀 사이에 damage event 가 한 번이라도 있으면 연결
- Weight  = 누적 damage / 교전 pair 개수 / 1 (binary)

핵심 엔트리:
    build_team_graph(match_data, ...)         - 매치 전체 기간
    build_early_team_graph(match_data, sec)   - 매치 시작 후 sec 초 구간만
각 행 (row) 에 survival_seconds 가 포함되며, h1_analysis 에서 rate/partial
control 분석에 사용된다.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Optional

import numpy as np
import networkx as nx

from network_analysis.db_loader import load_match_damage


# ============================================================
# 내부 헬퍼
# ============================================================

def _compute_team_survival(static: dict) -> dict[int, float]:
    """팀별 survival_seconds = 팀의 마지막 멤버 사망 시각.
    모두 끝까지 생존한 팀은 match duration 으로 대체.
    """
    duration = float(static.get("duration_s") or 0.0)
    death_times: dict = static.get("death_times", {}) or {}
    players = static.get("players", [])
    aid_to_idx = {p["account_id"]: p["idx"] for p in players}
    idx_to_aid = {p["idx"]: p["account_id"] for p in players}

    out: dict[int, float] = {}
    for t in static.get("teams", []):
        ti = t["idx"]
        members = t.get("members", []) or []
        if not members:
            out[ti] = 0.0
            continue
        times: list[float] = []
        for m in members:
            aid = idx_to_aid.get(m)
            dt = death_times.get(aid)
            if dt is None:
                # 끝까지 생존한 플레이어 → match duration 까지 생존
                times.append(duration)
            else:
                times.append(float(dt))
        out[ti] = max(times) if times else 0.0
    return out


def _events_to_team_graph(
    static: dict,
    events: list[dict],
    weight_mode: str,
    exclude_self_team: bool,
) -> tuple[nx.Graph, list[dict]]:
    """damage events 목록을 받아 팀 그래프 G + 팀별 row 리스트를 구성.

    공통 코어. build_team_graph / build_early_team_graph 에서 사용.
    """
    teams = static["teams"]
    players = static["players"]
    account_to_team = {p["account_id"]: p["team_idx"] for p in players}

    pair_dmg: dict[tuple[int, int], float] = defaultdict(float)
    pair_events: dict[tuple[int, int], int] = defaultdict(int)
    dmg_dealt: dict[int, float] = defaultdict(float)
    dmg_taken: dict[int, float] = defaultdict(float)
    team_combat_events: dict[int, int] = defaultdict(int)

    for e in events:
        a_team = account_to_team.get(e["attacker"])
        v_team = account_to_team.get(e["victim"])
        if a_team is None or v_team is None:
            continue
        if exclude_self_team and a_team == v_team:
            continue
        dmg = float(e["damage"])
        key = (min(a_team, v_team), max(a_team, v_team))
        pair_dmg[key] += dmg
        pair_events[key] += 1
        dmg_dealt[a_team] += dmg
        dmg_taken[v_team] += dmg
        team_combat_events[a_team] += 1
        team_combat_events[v_team] += 1

    # 그래프
    G = nx.Graph()
    for t in teams:
        G.add_node(t["idx"])

    for (ta, tb), dmg in pair_dmg.items():
        if weight_mode == "dmg":
            w = dmg
        elif weight_mode == "count":
            w = pair_events[(ta, tb)]
        elif weight_mode == "binary":
            w = 1.0
        else:
            raise ValueError(f"unknown weight_mode: {weight_mode}")
        G.add_edge(
            ta, tb,
            weight=float(w),
            dmg=float(dmg),
            events=int(pair_events[(ta, tb)]),
        )

    # 팀별 row
    survival_by_team = _compute_team_survival(static)

    rows: list[dict] = []
    for t in teams:
        ti = t["idx"]
        rows.append({
            "match_id":         static["match_id"],
            "map_name":         static.get("map_name"),
            "game_mode":        static.get("game_mode"),
            "team_idx":         ti,
            "team_id":          t["team_id"],
            "final_rank":       t.get("rank"),
            "n_members":        len(t.get("members", [])),
            "survival_seconds": float(survival_by_team.get(ti, 0.0)),
            "degree":           int(G.degree(ti)),
            "weighted_deg":     float(sum(
                d.get("weight", 0.0) for _, _, d in G.edges(ti, data=True)
            )),
            "combat_events":    int(team_combat_events.get(ti, 0)),
            "dmg_dealt":        float(dmg_dealt.get(ti, 0.0)),
            "dmg_taken":        float(dmg_taken.get(ti, 0.0)),
        })

    return G, rows


# ============================================================
# Public builders
# ============================================================

def build_team_graph(
    match_data: dict,
    weight_mode: str = "dmg",
    exclude_self_team: bool = True,
    preloaded_events: Optional[list[dict]] = None,
) -> tuple[nx.Graph, list[dict]]:
    """매치 한 건 전체 기간의 팀 네트워크 + 팀 row.

    preloaded_events 를 전달하면 DB 재조회를 스킵 (early_game 과 pair 로 호출 시).
    """
    static = match_data["static"]
    if preloaded_events is None:
        damage_pack = load_match_damage(static["match_id"])
        events = damage_pack.get("events", [])
    else:
        events = preloaded_events
    return _events_to_team_graph(static, events, weight_mode, exclude_self_team)


def build_early_team_graph(
    match_data: dict,
    window_sec: float,
    weight_mode: str = "dmg",
    exclude_self_team: bool = True,
    preloaded_events: Optional[list[dict]] = None,
) -> tuple[nx.Graph, list[dict]]:
    """매치 시작 후 window_sec 초 구간 damage events 만으로 팀 그래프 빌드.

    survival bias 제거용. 이 구간에는 대부분의 팀이 아직 생존 중이므로
    교전 참여도 측정이 생존 시간에 덜 의존한다.
    """
    static = match_data["static"]
    if preloaded_events is None:
        damage_pack = load_match_damage(static["match_id"])
        events = damage_pack.get("events", [])
    else:
        events = preloaded_events
    filtered = [e for e in events
                if float(e.get("elapsed", 0.0)) <= float(window_sec)]
    return _events_to_team_graph(static, filtered, weight_mode, exclude_self_team)


# ============================================================
# Adjacency / Degree 행렬
# ============================================================

def adjacency_matrix(G: nx.Graph, weight: str = "weight") -> tuple[np.ndarray, list[int]]:
    """A = [a_ij] 의 n × n 대칭 행렬과 노드 순서."""
    nodes = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes, weight=weight)
    return A, nodes


def degree_matrix(G: nx.Graph, weight: str = "weight") -> tuple[np.ndarray, list[int]]:
    """D = diag(d_i), d_i = Σ_j w_ij."""
    A, nodes = adjacency_matrix(G, weight=weight)
    D = np.diag(A.sum(axis=1))
    return D, nodes


# ============================================================
# Global 지표
# ============================================================

def graph_globals(G: nx.Graph) -> dict:
    """매치 1건의 팀 그래프에 대한 global 특성."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    out = {
        "n_nodes": int(n),
        "n_edges": int(m),
        "density": float(nx.density(G)) if n > 1 else 0.0,
        "avg_degree": float(2 * m / n) if n > 0 else 0.0,
        "avg_clustering": float(nx.average_clustering(G)) if n > 0 else 0.0,
    }
    comps = list(nx.connected_components(G))
    if comps:
        out["n_components"] = len(comps)
        giant_nodes = max(comps, key=len)
        out["giant_size"] = len(giant_nodes)
        H = G.subgraph(giant_nodes)
        try:
            out["diameter"] = int(nx.diameter(H))
        except Exception:
            out["diameter"] = 0
        try:
            out["avg_path_length"] = float(nx.average_shortest_path_length(H))
        except Exception:
            out["avg_path_length"] = 0.0
    else:
        out.update({"n_components": 0, "giant_size": 0,
                    "diameter": 0, "avg_path_length": 0.0})
    return out


def degree_distribution(G: nx.Graph) -> tuple[list[int], list[float]]:
    """P(k) — 노드 중 degree = k 일 확률."""
    if G.number_of_nodes() == 0:
        return [], []
    from collections import Counter
    degs = [d for _, d in G.degree()]
    cnt = Counter(degs)
    ks = sorted(cnt.keys())
    n = len(degs)
    probs = [cnt[k] / n for k in ks]
    return ks, probs


if __name__ == "__main__":
    import sys
    from network_analysis.data_loader import list_matches, build_match_data

    pts = list_matches("Baltic_Main", "squad", limit=1)
    if not pts:
        print("no match data"); sys.exit(1)

    md = build_match_data(pts[0], fetch_damage=True, stride=1)
    G_full, rows_full = build_team_graph(md)
    G_early, rows_early = build_early_team_graph(md, window_sec=300)

    print(f"Full:  |V|={G_full.number_of_nodes()}  |E|={G_full.number_of_edges()}")
    print(f"Early: |V|={G_early.number_of_nodes()}  |E|={G_early.number_of_edges()}")
    print("\nTop-3 by rank (full):")
    rs = sorted(rows_full, key=lambda r: r["final_rank"] or 99)
    for r in rs[:3]:
        print(f"  rank={r['final_rank']:>2} T{r['team_id']} "
              f"deg={r['degree']} surv={r['survival_seconds']:.0f}s "
              f"combat_events={r['combat_events']}")
