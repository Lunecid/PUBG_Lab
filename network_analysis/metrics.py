"""
프레임별 네트워크 지표 계산 (v2)
================================
네트워크 구성 (player-player):
  - ally edges   (weight = 1.0)
  - enemy edges  (weight = 0.5)   — encounter proximity
  - combat edges (weight = dmg/50, clipped) — 실제 교전

무향 가중 그래프로 결합 (중복 엣지는 가중치 합산).

Chapter 3 지표:
  전역: n_nodes, n_edges, density, avg_clustering, avg_path_length, diameter
  중심성: degree, closeness, betweenness, eigenvector, pagerank
  추가:  combat_in_degree (피격 가해자 수), combat_out_degree (공격 대상 수),
         dmg_dealt_window, dmg_taken_window

팀 노드는 시각화용으로만 쓰고 centrality 계산에서는 제외 (해석 단순화).
"""

import numpy as np
import networkx as nx
from collections import Counter, defaultdict

from network_analysis.community_detect import detect_for_frame
from network_analysis.narrative import describe_frame


_ALLY_W = 1.0
_ENEMY_W = 0.5
_DMG_NORM = 50.0
_DMG_MAX = 3.0


def build_nx_graph(frame):
    """
    프레임 → 무향 가중 networkx.Graph (alive 플레이어만).

    return: (G, alive_indices)
    """
    alive = [i for i, a in enumerate(frame["alive_mask"]) if a]
    alive_set = set(alive)

    G = nx.Graph()
    for i in alive:
        G.add_node(i)

    # ally
    for s, d in frame["edges"]["ally"]:
        if s in alive_set and d in alive_set:
            if G.has_edge(s, d):
                G[s][d]["weight"] += _ALLY_W
            else:
                G.add_edge(s, d, weight=_ALLY_W)

    # enemy
    for s, d in frame["edges"]["enemy"]:
        if s in alive_set and d in alive_set:
            if G.has_edge(s, d):
                G[s][d]["weight"] += _ENEMY_W
            else:
                G.add_edge(s, d, weight=_ENEMY_W)

    # combat (undirected로 합산)
    for s, d, dmg in frame["edges"]["combat"]:
        if s in alive_set and d in alive_set:
            w = min(dmg / _DMG_NORM, _DMG_MAX)
            if G.has_edge(s, d):
                G[s][d]["weight"] += w
            else:
                G.add_edge(s, d, weight=w)

    return G, alive


def compute_frame_metrics(frame, top_k=10):
    """
    프레임에 'metrics' 필드를 채워 반환.
    """
    G, alive = build_nx_graph(frame)
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if n == 0:
        frame["metrics"] = _empty_metrics()
        return frame

    # ── 전역 지표 ──
    global_ = {
        "n_nodes": int(n),
        "n_edges": int(m),
        "density": float(nx.density(G)) if n > 1 else 0.0,
        "avg_degree": float(2 * m / n),
        "avg_clustering": float(nx.average_clustering(G)),
    }
    # path stats on giant component (가중치 무시하고 hop distance)
    comps = list(nx.connected_components(G))
    if comps:
        giant = max(comps, key=len)
        H = G.subgraph(giant)
        try:
            global_["diameter"] = int(nx.diameter(H))
        except Exception:
            global_["diameter"] = 0
        try:
            global_["avg_path_length"] = float(nx.average_shortest_path_length(H))
        except Exception:
            global_["avg_path_length"] = 0.0
        global_["giant_size"] = len(giant)
        global_["n_components"] = len(comps)
    else:
        global_.update({"diameter": 0, "avg_path_length": 0.0,
                        "giant_size": 0, "n_components": 0})

    # ── degree distribution (alive 기준) ──
    degs = [d for _, d in G.degree()]
    cnt = Counter(degs)
    ks = sorted(cnt.keys())
    degree_dist = {
        "ks": ks,
        "probs": [cnt[k] / len(degs) for k in ks] if degs else [],
    }

    # ── Centrality (5종) ──
    # 거리 그래프 (weight 역수)
    G_dist = G.copy()
    for u, v, d in G_dist.edges(data=True):
        d["distance"] = 1.0 / max(d.get("weight", 1.0), 1e-6)

    centr = {}
    denom = max(n - 1, 1)
    centr["degree"] = {v: G.degree(v) / denom for v in G.nodes()}
    try:
        centr["closeness"] = nx.closeness_centrality(G_dist, distance="distance")
    except Exception:
        centr["closeness"] = {v: 0.0 for v in G.nodes()}
    try:
        centr["betweenness"] = nx.betweenness_centrality(
            G_dist, weight="distance", normalized=True
        )
    except Exception:
        centr["betweenness"] = {v: 0.0 for v in G.nodes()}
    try:
        centr["eigenvector"] = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        try:
            centr["eigenvector"] = nx.eigenvector_centrality(
                G, weight="weight", max_iter=500, tol=1e-6
            )
        except Exception:
            centr["eigenvector"] = {v: 0.0 for v in G.nodes()}
    try:
        centr["pagerank"] = nx.pagerank(G, weight="weight", alpha=0.85)
    except Exception:
        centr["pagerank"] = {v: 1.0 / n for v in G.nodes()}

    # dict → per-player list (길이 = n_players 전체, 미생존은 None)
    n_all = len(frame["alive_mask"])
    centrality_arrays = {}
    for k, d in centr.items():
        arr = [None] * n_all
        for v, val in d.items():
            arr[v] = float(val)
        centrality_arrays[k] = arr

    # ── Combat-specific 지표 ──
    in_deg = defaultdict(int)
    out_deg = defaultdict(int)
    dmg_dealt = defaultdict(float)
    dmg_taken = defaultdict(float)
    for s, d, dmg in frame["edges"]["combat"]:
        out_deg[s] += 1
        in_deg[d] += 1
        dmg_dealt[s] += dmg
        dmg_taken[d] += dmg
    combat_arrays = {
        "combat_in_degree":  [in_deg.get(i, 0) for i in range(n_all)],
        "combat_out_degree": [out_deg.get(i, 0) for i in range(n_all)],
        "dmg_dealt_window":  [float(dmg_dealt.get(i, 0.0)) for i in range(n_all)],
        "dmg_taken_window":  [float(dmg_taken.get(i, 0.0)) for i in range(n_all)],
    }

    # ── Top-K 랭킹 (player idx, value) 각 지표별 ──
    top_rank = {}
    for k in centr.keys():
        items = [(i, v) for i, v in enumerate(centrality_arrays[k]) if v is not None]
        items.sort(key=lambda x: x[1], reverse=True)
        top_rank[k] = items[:top_k]

    # ── Community Detection (CH5) ──
    n_teams_alive = sum(
        1 for c in frame.get("team_alive_count", []) if c and c > 0
    )
    community_data = detect_for_frame(
        G, n_all,
        n_teams_alive=n_teams_alive if n_teams_alive >= 2 else None,
        seed=0,
    )

    frame["metrics"] = {
        "global": global_,
        "degree_dist": degree_dist,
        "centrality": centrality_arrays,
        "combat": combat_arrays,
        "top_rank": top_rank,
        "community": community_data,
    }
    return frame


def _empty_metrics():
    return {
        "global": {"n_nodes": 0, "n_edges": 0, "density": 0.0,
                   "avg_degree": 0.0, "avg_clustering": 0.0,
                   "diameter": 0, "avg_path_length": 0.0,
                   "giant_size": 0, "n_components": 0},
        "degree_dist": {"ks": [], "probs": []},
        "centrality": {k: [] for k in
                       ["degree", "closeness", "betweenness",
                        "eigenvector", "pagerank"]},
        "combat": {"combat_in_degree": [], "combat_out_degree": [],
                   "dmg_dealt_window": [], "dmg_taken_window": []},
        "top_rank": {k: [] for k in
                     ["degree", "closeness", "betweenness",
                      "eigenvector", "pagerank"]},
        "community": {
            "louvain":  {"assignments": [], "sizes": [],
                         "n_communities": 0, "modularity_q": 0.0},
            "spectral": {"assignments": [], "sizes": [],
                         "n_communities": 0, "modularity_q": 0.0,
                         "k_used": 0, "k_source": "empty"},
            "seed": 0,
        },
    }


def compute_all_frames(match_data, top_k=10,
                       narrative_enabled=True, trend_window=5):
    """전체 프레임에 metrics + timeseries + narrative 를 채운다.

    Pipeline:
      1) 프레임마다 compute_frame_metrics (global/centrality/community)
      2) extract_timeseries 로 매치 레벨 시계열 생성 (narrative trend 재료)
      3) 각 프레임에 describe_frame 호출해 narrative 부여
    """
    # 1-pass: per-frame metrics (community 포함)
    for fr in match_data["frames"]:
        compute_frame_metrics(fr, top_k=top_k)

    # 2-pass: match-level timeseries
    ts = extract_timeseries(match_data)
    match_data["timeseries"] = ts

    # 3-pass: per-frame narrative (trend 에 ts 사용)
    if narrative_enabled:
        static = match_data.get("static", {})
        for i, fr in enumerate(match_data["frames"]):
            try:
                fr["narrative"] = describe_frame(
                    fr, ts, static,
                    frame_idx=i, window=trend_window,
                )
            except Exception as e:  # noqa: BLE001
                fr["narrative"] = {
                    "summary": f"T={fr.get('t', 0):.0f}",
                    "global": [f"[해설 생성 실패: {e}]"],
                    "centrality": [], "community": [], "trend": [],
                }
    return match_data


# ============================================================
# 시계열 추출 (요약/시각화용)
# ============================================================

def extract_timeseries(match_data):
    """매치-레벨 시계열. narrative.describe_trend / HTML 시계열 차트가 사용."""
    ts = {
        "t": [], "alive": [], "n_edges": [], "avg_clustering": [],
        "density": [], "avg_path_length": [], "diameter": [],
        "combat_edges": [], "combat_dmg_total": [],
        # Community (CH5)
        "modularity_q_louvain": [], "modularity_q_spectral": [],
        "n_communities_louvain": [], "n_communities_spectral": [],
    }
    for fr in match_data["frames"]:
        g = fr["metrics"]["global"]
        com = fr["metrics"].get("community", {})
        lv = com.get("louvain", {})
        sp = com.get("spectral", {})
        ts["t"].append(fr["t"])
        ts["alive"].append(g["n_nodes"])
        ts["n_edges"].append(g["n_edges"])
        ts["avg_clustering"].append(g["avg_clustering"])
        ts["density"].append(g["density"])
        ts["avg_path_length"].append(g["avg_path_length"])
        ts["diameter"].append(g["diameter"])
        ts["combat_edges"].append(len(fr["edges"]["combat"]))
        ts["combat_dmg_total"].append(
            float(sum(e[2] for e in fr["edges"]["combat"]))
        )
        ts["modularity_q_louvain"].append(float(lv.get("modularity_q", 0.0)))
        ts["modularity_q_spectral"].append(float(sp.get("modularity_q", 0.0)))
        ts["n_communities_louvain"].append(int(lv.get("n_communities", 0)))
        ts["n_communities_spectral"].append(int(sp.get("n_communities", 0)))
    return ts
