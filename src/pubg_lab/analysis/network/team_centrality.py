"""
Team-level Centrality Measures (CH3 + CH4 반영)
================================================
팀 네트워크 위에서 교재의 5개 중심성 지표를 모두 계산.

수식 (교재):
  Degree      : C_D(i) = k(i) / (|N| − 1)
  Closeness   : C_C(i) = (|N| − 1) / Σ_j d(i, j)
  Betweenness : C_B(i) = Σ_{j,k} p_{jk}(i) / p_{jk}
  Eigenvector : λ c = A c  (Perron-Frobenius, power iteration)
  PageRank    : r = β M r + (1 − β) / N   (random teleport)

distance 기반 지표(Closeness, Betweenness)에서는 weight 가 클수록 "가깝다" 고
해석하므로, distance = 1 / weight 로 변환하여 사용한다.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))
from typing import Optional

import networkx as nx


MEASURES = ("degree", "closeness", "betweenness", "eigenvector", "pagerank")


def _safe_distance_view(G: nx.Graph) -> nx.Graph:
    """weight → distance (= 1/weight) 변환된 복사본."""
    H = G.copy()
    for u, v, d in H.edges(data=True):
        w = float(d.get("weight", 1.0))
        d["distance"] = 1.0 / max(w, 1e-9)
    return H


def compute_team_centralities(
    G: nx.Graph,
    pagerank_alpha: float = 0.85,
) -> dict[str, dict[int, float]]:
    """G 의 모든 노드에 대해 5종 centrality 계산.

    빈 그래프, 싱글 노드, 비연결 그래프 모두 안전.
    """
    n = G.number_of_nodes()
    out: dict[str, dict[int, float]] = {m: {} for m in MEASURES}
    if n == 0:
        return out

    nodes = list(G.nodes())
    denom = max(n - 1, 1)

    # 1) Degree Centrality
    out["degree"] = {v: G.degree(v) / denom for v in nodes}

    # 2) Closeness — distance 기반
    H = _safe_distance_view(G)
    try:
        cc = nx.closeness_centrality(H, distance="distance")
    except Exception:
        cc = {v: 0.0 for v in nodes}
    out["closeness"] = {v: float(cc.get(v, 0.0)) for v in nodes}

    # 3) Betweenness — distance 기반 weighted
    try:
        bc = nx.betweenness_centrality(H, weight="distance", normalized=True)
    except Exception:
        bc = {v: 0.0 for v in nodes}
    out["betweenness"] = {v: float(bc.get(v, 0.0)) for v in nodes}

    # 4) Eigenvector Centrality — weighted (numpy 폴백 포함)
    try:
        ec = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        try:
            ec = nx.eigenvector_centrality(G, weight="weight", max_iter=1000, tol=1e-6)
        except Exception:
            ec = {v: 0.0 for v in nodes}
    out["eigenvector"] = {v: float(ec.get(v, 0.0)) for v in nodes}

    # 5) PageRank — weighted
    if G.number_of_edges() == 0:
        pr = {v: 1.0 / n for v in nodes}
    else:
        try:
            pr = nx.pagerank(G, weight="weight", alpha=pagerank_alpha)
        except Exception:
            pr = {v: 1.0 / n for v in nodes}
    out["pagerank"] = {v: float(pr.get(v, 0.0)) for v in nodes}

    return out


def attach_centralities_to_rows(
    rows: list[dict],
    centr: dict[str, dict[int, float]],
    prefix: str = "c_",
) -> list[dict]:
    """팀 row 에 centrality 값들을 'c_degree', 'c_pagerank' 등으로 덧붙인다."""
    for r in rows:
        ti = r["team_idx"]
        for m in MEASURES:
            r[f"{prefix}{m}"] = float(centr[m].get(ti, 0.0))
    return rows


def rank_teams_by_measure(
    centr: dict[str, dict[int, float]], measure: str, top_k: int = 5
) -> list[tuple[int, float]]:
    """한 measure 기준 top-k (team_idx, value) 리스트."""
    items = list(centr.get(measure, {}).items())
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:top_k]


if __name__ == "__main__":
    # Smoke on 1 match + karate + comparison
    import sys
    import numpy as np
    import networkx as nx
    from pubg_lab.analysis.network.data_loader import list_matches, build_match_data
    from pubg_lab.analysis.network.team_network import build_team_graph

    # 1) Karate club — 교재 기본 예시로 sanity 확인
    print("=== Karate club (교재 예시) ===")
    K = nx.karate_club_graph()
    for u, v in K.edges():
        K[u][v]["weight"] = 1.0
    kc = compute_team_centralities(K)
    for m in MEASURES:
        topv = max(kc[m].values())
        print(f"  {m:12s} max = {topv:.4f}")

    # 2) 실제 PUBG 매치
    pts = list_matches("Baltic_Main", "squad", limit=1)
    if not pts:
        print("no PUBG data")
        sys.exit(0)
    md = build_match_data(pts[0], fetch_damage=True, stride=1)
    G, rows = build_team_graph(md)
    centr = compute_team_centralities(G)
    rows = attach_centralities_to_rows(rows, centr)

    print("\n=== PUBG 매치: Centrality Top-3 per measure ===")
    teams = {r["team_idx"]: r["team_id"] for r in rows}
    for m in MEASURES:
        top = rank_teams_by_measure(centr, m, top_k=3)
        tops = ", ".join(f"T{teams[ti]}={v:.3f}" for ti, v in top)
        print(f"  {m:12s} {tops}")

    # 3) 간단 "가설 검증 preview" — 각 centrality 와 final_rank 의 상관
    print("\n=== H1 preview: centrality vs final_rank correlation ===")
    from scipy.stats import pearsonr, spearmanr

    ranks = [r["final_rank"] for r in rows if r["final_rank"] is not None]
    idxs_with_rank = [i for i, r in enumerate(rows) if r["final_rank"] is not None]
    for m in MEASURES:
        xs = [rows[i][f"c_{m}"] for i in idxs_with_rank]
        if len(xs) >= 4 and np.std(xs) > 1e-9:
            pr_r, pr_p = pearsonr(xs, ranks)
            sp_r, sp_p = spearmanr(xs, ranks)
            print(
                f"  {m:12s} Pearson r={pr_r:+.3f} (p={pr_p:.3f})   "
                f"Spearman ρ={sp_r:+.3f} (p={sp_p:.3f})"
            )
