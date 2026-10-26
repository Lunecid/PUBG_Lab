"""
Graph Community Detection (Chapter 5)
=====================================
프레임별 weighted undirected 그래프에서 Louvain 및 Spectral Clustering 을
돌려 node-level community 할당과 modularity Q 를 계산한다.

- Louvain: modularity Q 를 greedy 로 최대화 (NetworkX 내장)
- Spectral: 연결 성분별로 Laplacian 기반 clustering (sklearn)
- Modularity Q: Newman 2004. Q > 0.3 이면 유의미한 community 구조.

metrics.py::compute_frame_metrics 에서 호출되며, 이미 만들어진
weighted nx.Graph 를 그대로 입력으로 받는다 (그래프 재구성 없음).
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[5]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))
from typing import Optional

import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering


# ============================================================
# Louvain
# ============================================================


def louvain_partition(G: nx.Graph, seed: int = 0) -> list[set[int]]:
    """NetworkX 내장 Louvain. 빈 그래프/싱글턴/무엣지 safe."""
    n = G.number_of_nodes()
    if n == 0:
        return []
    if n == 1:
        return [set(G.nodes())]
    if G.number_of_edges() == 0:
        return [{v} for v in G.nodes()]
    try:
        comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
        return [set(c) for c in comms if c]
    except Exception:
        return [set(c) for c in nx.connected_components(G)]


# ============================================================
# Spectral Clustering
# ============================================================


def _spectral_one_component(H: nx.Graph, k: int, seed: int) -> list[set[int]]:
    """단일 연결 성분에 대해 SpectralClustering(affinity='precomputed')."""
    nodes = list(H.nodes())
    n = len(nodes)
    if n == 0:
        return []
    if n == 1:
        return [{nodes[0]}]
    k = max(1, min(int(k), n))
    if k == 1:
        return [set(nodes)]

    A = nx.to_numpy_array(H, nodelist=nodes, weight="weight")
    try:
        sc = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=seed,
        )
        labels = sc.fit_predict(A)
    except Exception:
        return [{v} for v in nodes]

    buckets: list[set[int]] = [set() for _ in range(k)]
    for v, lab in zip(nodes, labels):
        buckets[int(lab)].add(v)
    return [b for b in buckets if b]


def spectral_partition(
    G: nx.Graph,
    k: Optional[int] = None,
    n_teams_alive: Optional[int] = None,
    min_component_size: int = 3,
    seed: int = 0,
) -> list[set[int]]:
    """연결 성분별 spectral clustering. k 없으면 n_teams_alive → fallback."""
    n = G.number_of_nodes()
    if n == 0:
        return []
    if G.number_of_edges() == 0:
        return [{v} for v in G.nodes()]

    comps = list(nx.connected_components(G))

    if k is not None:
        k_total = max(2, int(k))
    elif n_teams_alive and n_teams_alive >= 2:
        k_total = int(n_teams_alive)
    else:
        k_total = max(2, int(np.sqrt(n / 2.0)))

    big = [c for c in comps if len(c) >= min_component_size]
    small = [c for c in comps if len(c) < min_component_size]

    result: list[set[int]] = [set(c) for c in small]

    if not big:
        return result

    total_big = sum(len(c) for c in big)
    for c in big:
        k_c = max(1, int(round(k_total * len(c) / max(total_big, 1))))
        H = G.subgraph(c)
        result.extend(_spectral_one_component(H, k_c, seed))

    return [p for p in result if p]


# ============================================================
# Modularity Q
# ============================================================


def modularity_q(G: nx.Graph, partition: list[set[int]], weight: str = "weight") -> float:
    """nx.community.modularity wrapper. 에러/빈값 → 0.0."""
    if not partition or G.number_of_edges() == 0:
        return 0.0
    try:
        return float(nx.community.modularity(G, partition, weight=weight))
    except Exception:
        return 0.0


# ============================================================
# Assignments array (color-stable)
# ============================================================


def partition_to_array(
    partition: list[set[int]], n_players: int
) -> tuple[list[Optional[int]], list[int]]:
    """커뮤니티 id 를 크기 내림차순으로 재할당 (큰 그룹=0번).

    return:
        assignments: 길이 n_players, 소속 안 된 노드는 None
        sizes:       커뮤니티별 크기 (내림차순)
    """
    sorted_parts = sorted(partition, key=lambda s: (-len(s), min(s) if s else 0))
    assignments: list[Optional[int]] = [None] * n_players
    sizes: list[int] = []
    for cid, comm in enumerate(sorted_parts):
        sizes.append(len(comm))
        for node in comm:
            if 0 <= node < n_players:
                assignments[node] = cid
    return assignments, sizes


# ============================================================
# Top-level entry
# ============================================================


def detect_for_frame(
    G: nx.Graph,
    n_players_total: int,
    n_teams_alive: Optional[int] = None,
    seed: int = 0,
) -> dict:
    """frame["metrics"]["community"] 에 들어갈 dict 반환.

    n_players_total: assignments 배열 길이 (dead 플레이어 포함 전체 플레이어 수)
    n_teams_alive:   spectral 의 k 기본값. None 이면 louvain 결과 개수 사용.
    """
    empty_one = {
        "assignments": [None] * n_players_total,
        "sizes": [],
        "n_communities": 0,
        "modularity_q": 0.0,
    }

    if G.number_of_nodes() < 2:
        return {
            "louvain": dict(empty_one),
            "spectral": {**empty_one, "k_used": 0, "k_source": "empty"},
            "seed": int(seed),
        }

    # Louvain
    l_part = louvain_partition(G, seed=seed)
    l_q = modularity_q(G, l_part)
    l_assign, l_sizes = partition_to_array(l_part, n_players_total)

    # Spectral: k 우선순위 — Louvain 결과를 기본 (modularity 최대화된 자연 분할)
    # → n_teams_alive fallback → 최종 fallback. 과분할 방지.
    if len(l_part) >= 2:
        k_used = len(l_part)
        k_source = "louvain_based"
    elif n_teams_alive and n_teams_alive >= 2:
        k_used = min(int(n_teams_alive), G.number_of_nodes())
        k_source = "n_teams_alive"
    else:
        k_used = 2
        k_source = "fallback"

    s_part = spectral_partition(G, k=k_used, seed=seed)
    s_q = modularity_q(G, s_part)
    s_assign, s_sizes = partition_to_array(s_part, n_players_total)

    return {
        "louvain": {
            "assignments": l_assign,
            "sizes": l_sizes,
            "n_communities": len(l_sizes),
            "modularity_q": round(l_q, 4),
        },
        "spectral": {
            "assignments": s_assign,
            "sizes": s_sizes,
            "n_communities": len(s_sizes),
            "modularity_q": round(s_q, 4),
            "k_used": int(k_used),
            "k_source": k_source,
        },
        "seed": int(seed),
    }


# ============================================================
# Karate Club smoke test (CH5 교재 기본 예시)
# ============================================================


def _karate_smoke() -> None:
    import random

    random.seed(0)
    np.random.seed(0)

    G = nx.karate_club_graph()
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0

    data = detect_for_frame(G, G.number_of_nodes(), n_teams_alive=2, seed=0)

    l = data["louvain"]
    s = data["spectral"]
    print(
        f"[Karate] Louvain:  n={l['n_communities']}  Q={l['modularity_q']:.3f}  sizes={l['sizes']}"
    )
    print(
        f"[Karate] Spectral: n={s['n_communities']}  Q={s['modularity_q']:.3f}  k={s['k_used']}  sizes={s['sizes']}"
    )

    assert l["modularity_q"] > 0.3, (
        f"Louvain Q={l['modularity_q']} ≤ 0.3 — community 구조 감지 실패"
    )
    assert 2 <= l["n_communities"] <= 5, f"n_communities={l['n_communities']} out of expected range"
    print("PASS: Q > 0.3 (Newman 2004 threshold)")


if __name__ == "__main__":
    _karate_smoke()
