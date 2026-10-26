"""
가설 검증용 공용 feature 엔진
===============================
팀 그래프 / centrality / community 위에 세 가설(H1·H2·H3)이 공유하는
팀 수준 feature 를 파생시킨다.

핵심 파생 지표
--------------
community_id       : Louvain 커뮤니티 번호 (크기 내림차순)
community_size     : 소속 커뮤니티의 팀 수
community_density  : 소속 커뮤니티 서브그래프의 density
within_deg         : 같은 커뮤니티 팀과의 weighted degree
across_deg         : 다른 커뮤니티 팀과의 weighted degree
local_clustering   : networkx local clustering coefficient
position_label     : "core" | "mid" | "peripheral" | "isolated"
                     - 고립(degree=0) 은 isolated
                     - betweenness 상위 25% 는 core
                     - 하위 25% 는 peripheral, 나머지 mid
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))
from collections import defaultdict

import numpy as np
import networkx as nx

from pubg_lab.analysis.network.community_detect import louvain_partition, modularity_q


# ============================================================
# Community feature attachment
# ============================================================


def attach_community_features(
    rows: list[dict],
    G: nx.Graph,
    seed: int = 0,
) -> None:
    """in-place 로 각 row 에 community_* feature 를 부착."""
    # Louvain partition (size-sorted community id)
    parts = louvain_partition(G, seed=seed)
    parts_sorted = sorted(parts, key=lambda s: (-len(s), min(s) if s else 0))
    team_to_cid: dict[int, int] = {}
    cid_to_size: dict[int, int] = {}
    for cid, comm in enumerate(parts_sorted):
        cid_to_size[cid] = len(comm)
        for n in comm:
            team_to_cid[n] = cid

    # community-subgraph density 계산 (cid 단위로 한 번만)
    cid_to_density: dict[int, float] = {}
    cid_to_modQ_contrib: dict[int, float] = {}
    for cid, comm in enumerate(parts_sorted):
        if len(comm) < 2:
            cid_to_density[cid] = 0.0
            continue
        H = G.subgraph(comm)
        cid_to_density[cid] = float(nx.density(H))

    # 전체 modularity Q (팀 그래프 기준)
    full_Q = modularity_q(G, parts_sorted)

    # local clustering (전체 그래프)
    lc_all = (
        nx.clustering(G, weight="weight")
        if G.number_of_edges() > 0
        else {v: 0.0 for v in G.nodes()}
    )

    # within / across degree
    within = defaultdict(float)
    across = defaultdict(float)
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        cu = team_to_cid.get(u)
        cv = team_to_cid.get(v)
        if cu is None or cv is None:
            continue
        if cu == cv:
            within[u] += w
            within[v] += w
        else:
            across[u] += w
            across[v] += w

    # betweenness 기준 position label
    try:
        bet = nx.betweenness_centrality(G, weight="weight", normalized=True)
    except Exception:
        bet = {v: 0.0 for v in G.nodes()}

    bet_values = sorted(bet.values(), reverse=True)
    n = len(bet_values)
    hi_cut = bet_values[max(0, n // 4 - 1)] if n >= 4 else float("inf")
    lo_cut = bet_values[max(0, (3 * n) // 4 - 1)] if n >= 4 else -float("inf")

    for r in rows:
        ti = r["team_idx"]
        cid = team_to_cid.get(ti)
        r["community_id"] = cid if cid is not None else -1
        r["community_size"] = cid_to_size.get(cid, 0) if cid is not None else 0
        r["community_density"] = cid_to_density.get(cid, 0.0) if cid is not None else 0.0
        r["within_deg"] = float(within.get(ti, 0.0))
        r["across_deg"] = float(across.get(ti, 0.0))
        r["local_clustering"] = float(lc_all.get(ti, 0.0))
        r["match_modularity_q"] = float(full_Q)

        # position label
        deg = int(r.get("degree", 0))
        b = float(bet.get(ti, 0.0))
        if deg == 0:
            label = "isolated"
        elif b >= hi_cut and n >= 4:
            label = "core"
        elif b <= lo_cut and n >= 4:
            label = "peripheral"
        else:
            label = "mid"
        r["position_label"] = label
        r["betweenness_raw"] = b


# ============================================================
# 지표 → PUBG 도메인 해석 문장
# ============================================================


def _fmt(v, d=3):
    try:
        if v is None or not np.isfinite(float(v)):
            return "—"
        return f"{float(v):.{d}f}"
    except Exception:
        return "—"


def density_interpretation(density: float, avg_n_teams: float) -> str:
    """density 값을 평균 참가 팀 수 기준으로 해석."""
    n = max(int(round(avg_n_teams)), 2)
    max_pairs = n * (n - 1) / 2
    actual = density * max_pairs
    level = "매우 희박" if density < 0.10 else "중간" if density < 0.25 else "조밀"
    return (
        f"평균 {n}개 팀 중 이론상 가능한 교전 쌍은 {int(max_pairs)}개. "
        f"실제로는 그 중 약 {actual:.1f}쌍만 맞붙음 ({level}). "
        "대부분의 팀 쌍은 서로 만나지 않고 매치가 끝난다는 뜻."
    )


def clustering_interpretation(cc: float) -> str:
    level = "낮음" if cc < 0.25 else "중간" if cc < 0.50 else "높음"
    return (
        f"한 팀이 교전한 두 팀이 서로도 교전했을 확률 약 {cc * 100:.0f}% ({level}). "
        "값이 높을수록 '같은 존에 세 팀 이상이 뭉쳐 삼각 교전을 벌인' 상황이 "
        "흔했다는 신호."
    )


def diameter_interpretation(diam: float, apl: float) -> str:
    return (
        f"가장 먼 두 팀도 평균 {diam:.1f}단계면 이어짐 · 평균 {apl:.2f}단계. "
        "모든 팀이 교전 그래프 상으로는 '몇 다리만 건너면 연결' 되는 "
        "작은 세상 구조."
    )


def modularity_interpretation(q: float, n_comm: float) -> str:
    level = "거의 없음" if q < 0.3 else "뚜렷함" if q < 0.5 else "매우 강함"
    return (
        f"Q = {_fmt(q, 3)} · 평균 {n_comm:.1f}개 커뮤니티. "
        f"팀 연합 구조가 {level}. "
        "한 매치 안에서 교전이 소수 지역에 집중되어 '지리적 진영' 이 "
        "나뉘는 경향을 수치로 보여준다."
    )


def pagerank_interpretation() -> str:
    return (
        "PageRank 상위 팀은 다른 강팀들과도 서로 교전한 팀. "
        "즉 맵의 '메인 교전지' 에 있던 팀. 단순히 많이 싸운 것(degree)이 "
        "아니라 '중요한 상대와 싸웠다'는 가중치가 반영된다."
    )


def betweenness_interpretation() -> str:
    return (
        "Betweenness 상위 팀은 서로 다른 교전 클러스터를 잇는 '중개자'. "
        "맵 중앙부 또는 경계 지역에서 여러 방향의 팀들과 엮인 전투를 "
        "벌였을 가능성이 크다."
    )


def frame_community_map(G: nx.Graph, seed: int = 0) -> dict[int, int]:
    """프레임별 community 색 안정화용 매핑.

    community id 로 커뮤니티 내 최소 team_idx 를 사용 → 같은 팀 집합이
    유지되면 id 가 불변, 팀 이탈/합병이 있을 때만 id 변경되어 의미 있는
    구조 변화를 색으로 드러냄.
    """
    parts = louvain_partition(G, seed=seed)
    out: dict[int, int] = {}
    for comm in parts:
        if not comm:
            continue
        anchor = min(comm)
        for n in comm:
            out[int(n)] = int(anchor)
    return out


def frame_position_map(G: nx.Graph) -> dict[int, str]:
    """프레임별 position 라벨 맵 (betweenness 기반)."""
    if G.number_of_nodes() == 0:
        return {}
    try:
        bet = nx.betweenness_centrality(G, weight="weight", normalized=True)
    except Exception:
        bet = {v: 0.0 for v in G.nodes()}
    values = sorted(bet.values(), reverse=True)
    n = len(values)
    hi = values[max(0, n // 4 - 1)] if n >= 4 else float("inf")
    lo = values[max(0, (3 * n) // 4 - 1)] if n >= 4 else -float("inf")
    out: dict[int, str] = {}
    for v in G.nodes():
        deg = G.degree(v)
        b = float(bet.get(v, 0.0))
        if deg == 0:
            label = "isolated"
        elif b >= hi and n >= 4:
            label = "core"
        elif b <= lo and n >= 4:
            label = "peripheral"
        else:
            label = "mid"
        out[int(v)] = label
    return out


def position_label_kr(label: str) -> str:
    return {
        "core": "중심 (core)",
        "mid": "중간 (mid)",
        "peripheral": "외곽 (peripheral)",
        "isolated": "고립 (isolated)",
    }.get(label, label)


if __name__ == "__main__":
    import sys
    from pubg_lab.analysis.network.data_loader import list_matches, build_match_data
    from pubg_lab.analysis.network.team_network import build_team_graph

    pts = list_matches("Baltic_Main", "squad", limit=1)
    if not pts:
        print("no data")
        sys.exit(1)

    md = build_match_data(pts[0], fetch_damage=True, stride=1)
    G, rows = build_team_graph(md)
    attach_community_features(rows, G)

    rs = sorted(rows, key=lambda r: r.get("final_rank") or 99)
    print(f"Full modularity Q = {rs[0]['match_modularity_q']:.3f}")
    print(
        f"\n{'rank':>4} {'team':>5} {'deg':>4} {'comm':>4} {'csz':>4} "
        f"{'cdens':>6} {'within':>7} {'across':>7} {'pos':<12}"
    )
    for r in rs[:8]:
        print(
            f"{r['final_rank']:>4} T{r['team_id']:>3} "
            f"{r['degree']:>4} {r['community_id']:>4} "
            f"{r['community_size']:>4} "
            f"{r['community_density']:>6.3f} "
            f"{r['within_deg']:>7.0f} {r['across_deg']:>7.0f} "
            f"{r['position_label']:<12}"
        )
