"""
H1 · Bias-controlled combat engagement vs placement
====================================================
팀 교전 참여도와 최종 순위의 상관을 네 가지 독립적인 관점에서 동시 검증:

    1. raw       — 기존 기본 상관 (degree / combat_events 등)
    2. intensity — 단위 시간당 정규화 (per-minute)
    3. early     — 매치 시작 후 window_sec 구간만
    4. partial   — survival_seconds 를 통제한 편상관

네 축 결과가 일치하면 결론 "confirmed", partial/intensity/early 가 0 부근이면
"rejected (bias-driven)", 혼재면 "inconclusive".
"""

from __future__ import annotations
from typing import Callable, Optional

import numpy as np
import networkx as nx
from scipy.stats import pearsonr, spearmanr, linregress, ttest_ind, mannwhitneyu

from pubg_lab.analysis.network.team_network import (
    build_team_graph,
    build_early_team_graph,
    graph_globals,
    degree_distribution,
    adjacency_matrix,
)
from pubg_lab.analysis.network.team_centrality import (
    compute_team_centralities,
    attach_centralities_to_rows,
    MEASURES,
)
from pubg_lab.analysis.network.community_detect import detect_for_frame
from pubg_lab.analysis.network.db_loader import load_match_damage


# ============================================================
# Feature mapping for four-way comparison
# ============================================================

# (raw_key, intensity_key, early_key, label)
FOURWAY_FEATURES = [
    ("degree", "degree_per_min", "early_degree", "degree"),
    ("combat_events", "combat_per_min", "early_combat_events", "combat events"),
    ("dmg_dealt", "dmg_per_min", "early_dmg_dealt", "dmg dealt"),
    ("c_degree", None, "early_c_degree", "centrality · degree"),
    ("c_closeness", None, "early_c_closeness", "centrality · closeness"),
    ("c_betweenness", None, "early_c_betweenness", "centrality · betweenness"),
    ("c_eigenvector", None, "early_c_eigenvector", "centrality · eigenvector"),
    ("c_pagerank", None, "early_c_pagerank", "centrality · pagerank"),
]


# ============================================================
# Per-match collection
# ============================================================


def _attach_rate_features(rows: list[dict]) -> None:
    """in-place: row 에 *_per_min feature 파생."""
    for r in rows:
        surv = float(r.get("survival_seconds") or 0.0)
        minutes = max(surv / 60.0, 1.0 / 60.0)  # 최소 1초 방어
        r["combat_per_min"] = float(r.get("combat_events", 0)) / minutes
        r["dmg_per_min"] = float(r.get("dmg_dealt", 0.0)) / minutes
        r["dmg_taken_per_min"] = float(r.get("dmg_taken", 0.0)) / minutes
        r["degree_per_min"] = float(r.get("degree", 0)) / minutes


def _attach_early_features(
    rows: list[dict],
    rows_early: list[dict],
    centr_early: dict[str, dict[int, float]],
) -> None:
    """in-place: row 에 early_* feature 부착 (team_idx 매칭)."""
    by_ti = {r["team_idx"]: r for r in rows_early}
    for r in rows:
        ti = r["team_idx"]
        er = by_ti.get(ti, {})
        r["early_degree"] = int(er.get("degree", 0))
        r["early_combat_events"] = int(er.get("combat_events", 0))
        r["early_dmg_dealt"] = float(er.get("dmg_dealt", 0.0))
        r["early_dmg_taken"] = float(er.get("dmg_taken", 0.0))
        for m in MEASURES:
            r[f"early_c_{m}"] = float(centr_early.get(m, {}).get(ti, 0.0))


def collect_all_matches(
    match_pts: list[str],
    build_match_data_fn: Callable,
    weight_mode: str = "dmg",
    early_window_sec: Optional[float] = 300.0,
    verbose: bool = True,
) -> dict:
    """매치 목록 순회하며 팀 row + full graph + early graph + community 를 집계.

    early_window_sec 이 None 이 아니면 각 row 에 early_* feature 를 함께 부착.
    한 매치에서 damage events 는 한 번만 로드해 full / early 양쪽에 재사용.

    Returns dict with keys:
        rows        — list[dict]  (row 당 full + rate + early feature 모두 포함)
        graphs      — list[nx.Graph]      (full)
        globals     — list[dict]          (full)
        adjs        — list[(A, nodes)]    (full)
        pks         — list[(ks, probs)]   (full)
        communities — list[dict]          (Louvain per full-match)
        match_ids   — list[str]
        early_window_sec
    """
    rows_all: list[dict] = []
    graphs: list[nx.Graph] = []
    globals_: list[dict] = []
    adjs = []
    pks = []
    communities = []
    match_ids = []

    for i, pt in enumerate(match_pts, 1):
        if verbose:
            print(f"[{i}/{len(match_pts)}] {pt.split('/')[-1]} ...", end=" ")

        md = build_match_data_fn(pt, fetch_damage=True, stride=1)
        match_id = md["static"]["match_id"]
        events = load_match_damage(match_id).get("events", [])

        # Full graph
        G, rows = build_team_graph(
            md,
            weight_mode=weight_mode,
            preloaded_events=events,
        )
        centr = compute_team_centralities(G)
        rows = attach_centralities_to_rows(rows, centr)

        # Rate-normalized features (생존시간 기반)
        _attach_rate_features(rows)

        # Early-game features
        if early_window_sec is not None:
            G_e, rows_e = build_early_team_graph(
                md,
                window_sec=early_window_sec,
                weight_mode=weight_mode,
                preloaded_events=events,
            )
            centr_e = compute_team_centralities(G_e)
            _attach_early_features(rows, rows_e, centr_e)

        glb = graph_globals(G)
        glb["match_id"] = match_id
        A, nodes = adjacency_matrix(G)
        ks, probs = degree_distribution(G)

        com = detect_for_frame(
            G,
            n_players_total=G.number_of_nodes(),
            n_teams_alive=glb["n_nodes"],
            seed=0,
        )

        rows_all.extend(rows)
        graphs.append(G)
        globals_.append(glb)
        adjs.append((A, nodes))
        pks.append((ks, probs))
        communities.append(com)
        match_ids.append(match_id)

        if verbose:
            print(
                f"|V|={glb['n_nodes']} |E|={glb['n_edges']} Q={com['louvain']['modularity_q']:.2f}"
            )

    return {
        "rows": rows_all,
        "graphs": graphs,
        "globals": globals_,
        "adjs": adjs,
        "pks": pks,
        "communities": communities,
        "match_ids": match_ids,
        "early_window_sec": early_window_sec,
    }


# ============================================================
# Statistical helpers
# ============================================================


def _clean_triplets(xs, ys, zs=None):
    out_x, out_y, out_z = [], [], []
    zs_iter = zs if zs is not None else [0.0] * len(xs)
    for x, y, z in zip(xs, ys, zs_iter):
        if x is None or y is None or z is None:
            continue
        try:
            xv = float(x)
            yv = float(y)
            zv = float(z)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(xv) and np.isfinite(yv) and np.isfinite(zv)):
            continue
        out_x.append(xv)
        out_y.append(yv)
        out_z.append(zv)
    return out_x, out_y, out_z


def _pearson_safe(xs, ys) -> tuple[float, float, int]:
    xs_c, ys_c, _ = _clean_triplets(xs, ys, None)
    n = len(xs_c)
    if n < 4 or np.std(xs_c) < 1e-12 or np.std(ys_c) < 1e-12:
        return float("nan"), float("nan"), n
    r, p = pearsonr(xs_c, ys_c)
    return float(r), float(p), n


def partial_correlation(
    xs: list,
    ys: list,
    zs: list,
) -> tuple[float, int]:
    """r_{xy·z} = (r_xy − r_xz·r_yz) / √((1 − r_xz²)(1 − r_yz²))"""
    xs_c, ys_c, zs_c = _clean_triplets(xs, ys, zs)
    n = len(xs_c)
    if n < 4:
        return float("nan"), n
    if min(np.std(xs_c), np.std(ys_c), np.std(zs_c)) < 1e-12:
        return float("nan"), n
    r_xy, _ = pearsonr(xs_c, ys_c)
    r_xz, _ = pearsonr(xs_c, zs_c)
    r_yz, _ = pearsonr(ys_c, zs_c)
    denom = (1.0 - r_xz**2) * (1.0 - r_yz**2)
    if denom <= 1e-12:
        return float("nan"), n
    r = (r_xy - r_xz * r_yz) / (denom**0.5)
    return float(r), n


def correlations_for_scatter(
    rows: list[dict],
    feature_keys: list[str],
) -> list[dict]:
    """각 feature 에 대해 scatter 차트 + regression 에 필요한 통계."""
    ranks = [r.get("final_rank") for r in rows]
    out: list[dict] = []
    for feat in feature_keys:
        xs_raw = [r.get(feat) for r in rows]
        xs, ys, _ = _clean_triplets(xs_raw, ranks)
        n = len(xs)
        if n < 4 or np.std(xs) < 1e-12 or np.std(ys) < 1e-12:
            out.append(
                {
                    "feature": feat,
                    "n": n,
                    "pearson_r": float("nan"),
                    "pearson_p": float("nan"),
                    "lin_slope": float("nan"),
                    "lin_intercept": float("nan"),
                    "lin_r2": float("nan"),
                }
            )
            continue
        pr_r, pr_p = pearsonr(xs, ys)
        lin = linregress(xs, ys)
        out.append(
            {
                "feature": feat,
                "n": n,
                "pearson_r": float(pr_r),
                "pearson_p": float(pr_p),
                "lin_slope": float(lin.slope),
                "lin_intercept": float(lin.intercept),
                "lin_r2": float(lin.rvalue**2),
            }
        )
    return out


def fourway_correlation_table(rows: list[dict]) -> list[dict]:
    """각 feature 에 대해 raw / intensity / early / partial(r|survival) 계산."""
    ranks = [r.get("final_rank") for r in rows]
    survs = [r.get("survival_seconds") for r in rows]
    table: list[dict] = []
    for raw_k, int_k, early_k, label in FOURWAY_FEATURES:
        xs_raw = [r.get(raw_k) for r in rows]
        raw_r, raw_p, raw_n = _pearson_safe(xs_raw, ranks)

        if int_k:
            xs_int = [r.get(int_k) for r in rows]
            int_r, int_p, int_n = _pearson_safe(xs_int, ranks)
        else:
            int_r, int_p, int_n = float("nan"), float("nan"), 0

        if early_k:
            xs_e = [r.get(early_k) for r in rows]
            early_r, early_p, early_n = _pearson_safe(xs_e, ranks)
        else:
            early_r, early_p, early_n = float("nan"), float("nan"), 0

        part_r, part_n = partial_correlation(xs_raw, ranks, survs)

        table.append(
            {
                "label": label,
                "raw_key": raw_k,
                "raw_r": raw_r,
                "raw_p": raw_p,
                "raw_n": raw_n,
                "int_r": int_r,
                "int_p": int_p,
                "int_n": int_n,
                "early_r": early_r,
                "early_p": early_p,
                "early_n": early_n,
                "partial_r": part_r,
                "partial_n": part_n,
            }
        )
    return table


# ============================================================
# Quartile group comparison
# ============================================================


def quartile_group_compare(
    rows: list[dict],
    feature: str,
    target: str = "final_rank",
) -> dict:
    pairs = [
        (r[feature], r[target])
        for r in rows
        if r.get(feature) is not None and r.get(target) is not None
    ]
    if len(pairs) < 8:
        return {"feature": feature, "insufficient": True, "n_top": 0, "n_bot": 0}
    pairs.sort(key=lambda p: p[0], reverse=True)
    q = max(1, len(pairs) // 4)
    top = [p[1] for p in pairs[:q]]
    bot = [p[1] for p in pairs[-q:]]
    t_res = ttest_ind(top, bot, equal_var=False)
    mw_res = mannwhitneyu(top, bot, alternative="two-sided")
    return {
        "feature": feature,
        "insufficient": False,
        "n_top": len(top),
        "n_bot": len(bot),
        "top_mean_rank": float(np.mean(top)),
        "bot_mean_rank": float(np.mean(bot)),
        "top_median_rank": float(np.median(top)),
        "bot_median_rank": float(np.median(bot)),
        "welch_t": float(t_res.statistic),
        "welch_p": float(t_res.pvalue),
        "mannwhitney_U": float(mw_res.statistic),
        "mannwhitney_p": float(mw_res.pvalue),
    }


# ============================================================
# Conclusion engine
# ============================================================


def _sig(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def classify_verdict(degree_row: dict) -> str:
    """degree 행의 네 축 r 값을 기준으로 confirmed / rejected / inconclusive.

    규칙:
      raw_r 이 충분히 음수 (<= -0.2) 이고:
        - int/early/partial 이 모두 |r| < 0.15         → rejected (bias)
        - int/early/partial 의 중앙값이 -0.2 이하      → confirmed
        - 나머지                                        → inconclusive
      raw_r 이 음수가 아니면 inconclusive.
    """
    raw_r = degree_row.get("raw_r")
    if raw_r is None or not np.isfinite(raw_r) or raw_r > -0.2:
        return "inconclusive"

    controls = [
        degree_row.get("int_r"),
        degree_row.get("early_r"),
        degree_row.get("partial_r"),
    ]
    ctrl_valid = [float(r) for r in controls if r is not None and np.isfinite(r)]
    if not ctrl_valid:
        return "inconclusive"

    if all(abs(r) < 0.15 for r in ctrl_valid):
        return "rejected"

    # "지지하는" = 같은 부호 & 절대값 >= 0.15
    supporting = [r for r in ctrl_valid if r < -0.15]
    if len(supporting) >= 2 and float(np.median(ctrl_valid)) <= -0.2:
        return "confirmed"

    return "inconclusive"


def generate_h1_conclusion(
    fourway_table: list[dict],
    quartile_results: list[dict],
    n_matches: int,
    n_teams_total: int,
    early_window_sec: Optional[float] = None,
) -> tuple[str, list[str], dict]:
    """4-way 판정 후 결론 문장 + hero 용 데이터 반환.

    returns:
        verdict  — "confirmed" | "rejected" | "inconclusive"
        lines    — 번호 매긴 결론 문장 리스트 (한국어)
        hero     — hero 섹션 숫자 (raw/int/early/partial/rank_delta/verdict)
    """
    deg_row = next((r for r in fourway_table if r["raw_key"] == "degree"), None)
    verdict = classify_verdict(deg_row) if deg_row else "inconclusive"

    lines: list[str] = []
    lines.append(
        f"분석 대상 · {n_matches}개 매치, {n_teams_total}개 팀 데이터포인트"
        f"{' · 초반 윈도우 ' + str(int(early_window_sec)) + '초' if early_window_sec else ''}"
    )

    if not deg_row:
        lines.append("데이터 부족 — 결론 도출 불가")
        return verdict, lines, {"verdict": verdict}

    raw_r = deg_row["raw_r"]
    int_r = deg_row["int_r"]
    early_r = deg_row["early_r"]
    part_r = deg_row["partial_r"]

    # 원시 상관 보고
    lines.append(
        f"원시 Pearson r(degree, 순위) = {raw_r:+.3f} "
        f"(n = {deg_row['raw_n']}, p = {deg_row['raw_p']:.4f}{_sig(deg_row['raw_p'])})"
    )

    # 편향 통제 세 축
    lines.append(
        f"강도 r(분당 degree, 순위) = {int_r:+.3f}"
        f"{_sig(deg_row['int_p'])}  ·  "
        f"초반 r(초반 degree, 순위) = {early_r:+.3f}"
        f"{_sig(deg_row['early_p'])}  ·  "
        f"편상관 r(degree, 순위 | 생존시간) = {part_r:+.3f}"
    )

    # 판정 해석
    if verdict == "confirmed":
        lines.append(
            "판정 · **가설 지지**. 생존 시간을 통제한 세 축에서도 음의 상관이 "
            "유지되어, 교전 참여도 자체가 순위 향상에 기여하는 것으로 보인다 "
            "(단순 생존 편향의 결과가 아님)."
        )
    elif verdict == "rejected":
        lines.append(
            "판정 · **기각 — 편향 기인**. 통제 후 상관이 사라짐. "
            "원시 상관은 오래 살아남아 자연스럽게 교전이 늘어난 데서 비롯된 "
            "생존 편향의 산물이며, 원 가설은 기각된다."
        )
    else:
        lines.append(
            "판정 · **결론 유보**. 네 축 사이에 부호와 강도가 엇갈려 단일한 "
            "결론을 내기 어렵다. 매치 수를 늘리거나 맵·모드별로 나눠 재분석할 "
            "것을 권장한다."
        )

    # Quartile 정보 (참고)
    q_deg = next((q for q in quartile_results if q.get("feature") == "degree"), None)
    if q_deg and not q_deg.get("insufficient"):
        diff = q_deg["top_mean_rank"] - q_deg["bot_mean_rank"]
        lines.append(
            f"참고 · 상위 25% degree 팀 평균순위 {q_deg['top_mean_rank']:.1f}, "
            f"하위 25% {q_deg['bot_mean_rank']:.1f} (Δ {diff:+.1f}, "
            f"Welch p = {q_deg['welch_p']:.4f}{_sig(q_deg['welch_p'])})"
        )

    # rank_delta from quartile (for hero)
    rank_delta = None
    if q_deg and not q_deg.get("insufficient"):
        rank_delta = abs(q_deg["top_mean_rank"] - q_deg["bot_mean_rank"])

    hero = {
        "verdict": verdict,
        "raw_r": raw_r if np.isfinite(raw_r) else None,
        "int_r": int_r if np.isfinite(int_r) else None,
        "early_r": early_r if np.isfinite(early_r) else None,
        "partial_r": part_r if np.isfinite(part_r) else None,
        "rank_delta": rank_delta,
        "n_matches": n_matches,
        "n_teams": n_teams_total,
    }
    return verdict, lines, hero
