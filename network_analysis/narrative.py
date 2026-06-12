"""
Per-frame narrative generation (프레임별 자연어 해설)
======================================================
LLM 없이 템플릿 + 값 버킷팅으로 재현 가능한 한국어 해설을 만든다.
교재(CH2~5) 용어를 그대로 사용하여 학습 연결성을 유지.

frame["narrative"] 에 다음 구조로 채운다:
    {
      "summary":    "T=04:20 · 생존 18명 · 교전 2건",
      "global":     [str, ...],   # 전역 구조 (density, clustering, diameter...)
      "centrality": [str, ...],   # Top-K 해석
      "community":  [str, ...],   # Louvain/Spectral Q 와 크기
      "trend":      [str, ...],   # 최근 N 프레임 대비 추세
    }
"""

from __future__ import annotations
from typing import Optional


# ============================================================
# 기본 버킷 (config 에서 override 가능)
# ============================================================

DEFAULT_BUCKETS = {
    "density":        [(0.1, "희박"), (0.3, "중간"), (1.0, "조밀")],
    "avg_clustering": [(0.2, "낮음"), (0.5, "중간"), (1.0, "높음")],
    "modularity_q":   [(0.3, "약함"), (0.5, "유의"), (1.0, "강함")],
}


def bucket(value: float, thresholds: list[tuple[float, str]]) -> str:
    """value 이상인 첫 upper 의 label 반환. thresholds 는 오름차순 upper."""
    for upper, label in thresholds:
        if value <= upper:
            return label
    return thresholds[-1][1] if thresholds else ""


# ============================================================
# 보조
# ============================================================

def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _player_label(players: list[dict], idx: int) -> str:
    if idx is None or idx < 0 or idx >= len(players):
        return f"#{idx}"
    p = players[idx]
    short = p.get("short") or (str(p.get("account_id", ""))[:8])
    team = p.get("team_idx", "?")
    return f"P{idx} · T{team}"


# ============================================================
# 섹션별 해설
# ============================================================

def describe_global(g: dict, buckets: dict = DEFAULT_BUCKETS) -> list[str]:
    """전역 지표 해설."""
    out: list[str] = []
    n_nodes = g.get("n_nodes", 0)
    if n_nodes == 0:
        return ["생존자 없음 — 네트워크가 비어있음"]

    density = g.get("density", 0.0)
    cc = g.get("avg_clustering", 0.0)
    diam = g.get("diameter", 0)
    apl = g.get("avg_path_length", 0.0)
    avg_deg = g.get("avg_degree", 0.0)
    giant = g.get("giant_size", 0)
    ncomp = g.get("n_components", 0)

    d_lbl = bucket(density, buckets["density"])
    cc_lbl = bucket(cc, buckets["avg_clustering"])

    out.append(
        f"density = {density:.3f} ({d_lbl}) — 가능한 엣지 중 "
        f"{_fmt_pct(density)}가 실제로 연결됨"
    )
    out.append(
        f"평균 클러스터링 C̄ = {cc:.3f} ({cc_lbl}) — "
        "이웃 간 삼각형 밀도 (높을수록 팀/그룹 단위 응집)"
    )
    out.append(f"평균 degree = {avg_deg:.2f} — 한 플레이어 평균 연결 수")

    if diam > 0 and apl > 0:
        sw = "small-world" if apl < 3.0 else "long-path"
        out.append(
            f"diameter = {diam}, avg path = {apl:.2f} — {sw} 경향"
        )

    if ncomp > 0:
        if giant == n_nodes:
            out.append(f"giant component = {giant}/{n_nodes} — 한 덩어리로 연결")
        else:
            out.append(
                f"컴포넌트 {ncomp}개, giant = {giant}/{n_nodes} — 네트워크 분열"
            )
    return out


_CENTR_LABELS = {
    "pagerank":    ("PageRank",    "이웃의 중요도까지 반영한 재귀적 영향력"),
    "betweenness": ("Betweenness", "서로 다른 그룹을 잇는 중개자 역할"),
    "closeness":   ("Closeness",   "모든 플레이어와 가까운 중심 위치"),
    "eigenvector": ("Eigenvector", "중요한 이웃과 연결된 정도"),
    "degree":      ("Degree",      "직접 연결된 이웃 수"),
}


def describe_top_centrality(
    top_rank: dict,
    players: list[dict],
    measures: tuple = ("pagerank", "betweenness", "degree"),
) -> list[str]:
    """각 measure 의 Top-1 을 한 문장씩."""
    out: list[str] = []
    for m in measures:
        items = top_rank.get(m, [])
        if not items:
            continue
        top_i, top_v = items[0]
        name, desc = _CENTR_LABELS.get(m, (m, ""))
        out.append(
            f"{name} 1위: {_player_label(players, int(top_i))} "
            f"({top_v:.4f}) — {desc}"
        )
    return out


def describe_community(
    community: dict,
    players: list[dict],
    buckets: dict = DEFAULT_BUCKETS,
) -> list[str]:
    """Louvain + Spectral 해설."""
    out: list[str] = []
    if not community:
        return out

    lv = community.get("louvain", {})
    sp = community.get("spectral", {})

    lv_n = lv.get("n_communities", 0)
    lv_q = lv.get("modularity_q", 0.0)
    lv_sizes = lv.get("sizes", [])
    if lv_n >= 1:
        q_lbl = bucket(lv_q, buckets["modularity_q"])
        sizes_str = str(lv_sizes) if lv_sizes else "[]"
        out.append(
            f"Louvain: Q = {lv_q:.3f} ({q_lbl}) · {lv_n}개 커뮤니티 · "
            f"크기 {sizes_str}"
        )
        if lv_q > 0.3:
            out.append("  → Q > 0.3 (Newman 2004) · 유의미한 community 구조")

    sp_n = sp.get("n_communities", 0)
    sp_q = sp.get("modularity_q", 0.0)
    sp_k = sp.get("k_used", 0)
    sp_sizes = sp.get("sizes", [])
    if sp_n >= 1:
        sizes_str = str(sp_sizes) if sp_sizes else "[]"
        out.append(
            f"Spectral (k={sp_k}): Q = {sp_q:.3f} · {sp_n}개 커뮤니티 · "
            f"크기 {sizes_str}"
        )
    return out


# ============================================================
# Trend (최근 N 프레임 대비 방향)
# ============================================================

def _trend_arrow(prev_avg: float, curr: float) -> tuple[str, str]:
    if curr is None or prev_avg is None:
        return "→", "변동 없음"
    if abs(prev_avg) < 1e-9:
        if curr > 1e-6:
            return "↑", "상승 (0에서 증가)"
        return "→", "안정 (0 부근)"
    delta = (curr - prev_avg) / abs(prev_avg)
    if delta > 0.10:
        return "↑", "상승"
    if delta < -0.10:
        return "↓", "하락"
    return "→", "안정"


_TREND_KEYS = [
    ("density",              "density"),
    ("avg_clustering",       "avg_clustering"),
    ("avg_path_length",      "avg_path"),
    ("modularity_q_louvain", "modularity Q"),
    ("n_communities_louvain", "n_communities"),
]


def describe_trend(
    ts: Optional[dict],
    current_idx: int,
    window: int = 5,
) -> list[str]:
    """매치 레벨 timeseries 기반 최근 window 프레임 평균 대비 현재 추세."""
    if not ts or current_idx < 1:
        return []

    out: list[str] = []
    start = max(0, current_idx - window)

    for key, label in _TREND_KEYS:
        arr = ts.get(key)
        if arr is None or current_idx >= len(arr):
            continue
        prev = [x for x in arr[start:current_idx] if x is not None]
        if not prev:
            continue
        prev_avg = sum(prev) / len(prev)
        curr = arr[current_idx]
        if curr is None:
            continue
        arrow, verb = _trend_arrow(prev_avg, curr)
        if arrow == "→" and verb.startswith("안정"):
            out.append(f"{label} {arrow} {verb} (최근 평균 {prev_avg:.3f} · 현재 {curr:.3f})")
        else:
            out.append(
                f"{label} {arrow} {verb} "
                f"(최근 {len(prev)}프레임 평균 {prev_avg:.3f} → 현재 {curr:.3f})"
            )
    return out


# ============================================================
# Top-level
# ============================================================

def describe_frame(
    frame: dict,
    ts: Optional[dict],
    static: dict,
    frame_idx: int,
    window: int = 5,
    buckets: dict = DEFAULT_BUCKETS,
) -> dict:
    """frame["narrative"] 에 저장할 dict 반환.

    Args:
        frame: 한 스냅샷 dict (metrics 포함 필수)
        ts: 매치 전체 시계열 dict (없으면 trend 스킵)
        static: 매치 메타 (players, teams 등)
        frame_idx: 프레임 인덱스
        window: 추세 판단용 lookback
    """
    metr = frame.get("metrics", {})
    g = metr.get("global", {})
    top_rank = metr.get("top_rank", {})
    community = metr.get("community", {})
    players = static.get("players", [])

    t = frame.get("t", 0.0)
    alive = g.get("n_nodes", 0)
    combat_edges = len(frame.get("edges", {}).get("combat", []))
    mm, ss = divmod(int(t), 60)
    summary = f"T={mm:02d}:{ss:02d} · 생존 {alive}명 · 교전 {combat_edges}건"

    return {
        "summary":    summary,
        "global":     describe_global(g, buckets=buckets),
        "centrality": describe_top_centrality(top_rank, players),
        "community":  describe_community(community, players, buckets=buckets),
        "trend":      describe_trend(ts, frame_idx, window=window),
    }


# ============================================================
# Smoke
# ============================================================

def _smoke() -> None:
    fake_frame = {
        "t": 260.0,
        "edges": {"combat": [[0, 1, 15.0], [2, 3, 40.0]]},
        "metrics": {
            "global": {
                "n_nodes": 18, "n_edges": 24, "density": 0.157,
                "avg_degree": 2.67, "avg_clustering": 0.42,
                "diameter": 4, "avg_path_length": 1.95,
                "giant_size": 18, "n_components": 1,
            },
            "top_rank": {
                "pagerank":    [[3, 0.082], [7, 0.074]],
                "betweenness": [[7, 0.31],  [3, 0.22]],
                "degree":      [[3, 0.45],  [4, 0.40]],
            },
            "community": {
                "louvain": {
                    "assignments": [None] * 20, "sizes": [8, 6, 3, 1],
                    "n_communities": 4, "modularity_q": 0.56,
                },
                "spectral": {
                    "assignments": [None] * 20, "sizes": [9, 5, 4],
                    "n_communities": 3, "modularity_q": 0.48,
                    "k_used": 3, "k_source": "n_teams_alive",
                },
            },
        },
    }
    fake_ts = {
        "density":              [0.10, 0.12, 0.13, 0.14, 0.16, 0.157],
        "avg_clustering":       [0.30, 0.35, 0.38, 0.40, 0.41, 0.42],
        "avg_path_length":      [2.50, 2.30, 2.10, 2.05, 2.00, 1.95],
        "modularity_q_louvain": [0.30, 0.40, 0.48, 0.52, 0.55, 0.56],
        "n_communities_louvain": [6,    5,    4,    4,    4,    4],
    }
    fake_static = {
        "players": [{"idx": i, "short": f"p{i:02d}", "team_idx": i // 4}
                    for i in range(20)],
    }
    nr = describe_frame(fake_frame, fake_ts, fake_static, frame_idx=5)

    print("=== summary ===")
    print(nr["summary"])
    for section in ("global", "centrality", "community", "trend"):
        print(f"\n=== {section} ===")
        for line in nr[section]:
            print(f"  • {line}")

    assert nr["summary"].startswith("T=04:20")
    assert len(nr["global"]) >= 3
    assert len(nr["centrality"]) >= 3
    assert len(nr["community"]) >= 2
    assert len(nr["trend"]) >= 3
    print("\nPASS: describe_frame produced all 4 sections.")


if __name__ == "__main__":
    _smoke()
