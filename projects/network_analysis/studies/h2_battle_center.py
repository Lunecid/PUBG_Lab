"""
H2 (v2) · 다중 교전 중심 (Battle Center "C") vs 평균 순위
=============================================================
여러 교전이 동시 다발적으로 벌어지는 팀들의 중심에 위치한 팀(= 다중 교전
중심)의 평균 순위가 그렇지 않은 팀에 비해 어떻게 되는가?

"중심성"의 정의 — battle_center_score:
    z(c_betweenness) + z(c_degree) + z(local_clustering)
    · betweenness: 다른 교전 클러스터를 잇는 중개자
    · degree:      다양한 상대팀과 직접 교전
    · local_clustering: 자신의 교전 이웃들이 서로도 교전 (삼각 교전 다수)

검증 그룹:
    상위 25% battle_center_score (battle_center) vs 하위 75% (rest)
    단, **degree == 0 (네트워크 비참여) 팀은 제외** — Louvain 단일 노드
    오염 문제(plan A.0)를 회피.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from pubg_lab.analysis.network.legacy.h1_analysis import _pearson_safe, partial_correlation


# ============================================================
# Helpers
# ============================================================


def _sig(p) -> str:
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _cliffs_delta(a: list[float], b: list[float]) -> float:
    a = [x for x in a if x is not None and np.isfinite(x)]
    b = [x for x in b if x is not None and np.isfinite(x)]
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return float("nan")
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    diff = a_arr[:, None] - b_arr[None, :]
    return float(((diff > 0).sum() - (diff < 0).sum()) / (n_a * n_b))


# ============================================================
# 메인
# ============================================================


def compute_h2_battle_center(rows: list[dict]) -> dict:
    """H2 (v2) — degree>0 팀 중 battle_center_score 상·하위 비교.

    Pre-condition: archetype.assign_archetype_labels() 가 battle_center_score 를
    이미 모든 row 에 부착해놓아야 함.
    """
    eligible = [
        r
        for r in rows
        if int(r.get("degree") or 0) > 0
        and r.get("final_rank") is not None
        and r.get("battle_center_score") is not None
        and np.isfinite(float(r["battle_center_score"]))
    ]
    n_eligible = len(eligible)
    n_excluded_degzero = sum(1 for r in rows if int(r.get("degree") or 0) == 0)

    if n_eligible < 8:
        return {
            "verdict": "inconclusive",
            "n_eligible": n_eligible,
            "n_excluded_degzero": n_excluded_degzero,
            "conclusion_lines": [f"네트워크 참여 팀이 {n_eligible} 개로 적어 검증 불가."],
        }

    scores = np.array([float(r["battle_center_score"]) for r in eligible])
    q_high = float(np.percentile(scores, 75))

    centers = [
        float(r["final_rank"]) for r in eligible if float(r["battle_center_score"]) >= q_high
    ]
    rest = [float(r["final_rank"]) for r in eligible if float(r["battle_center_score"]) < q_high]

    n_c, n_r = len(centers), len(rest)
    mw_U = float("nan")
    mw_p = float("nan")
    if n_c >= 3 and n_r >= 3:
        try:
            res = mannwhitneyu(centers, rest, alternative="two-sided")
            mw_U = float(res.statistic)
            mw_p = float(res.pvalue)
        except Exception:
            pass

    delta = _cliffs_delta(centers, rest) if n_c and n_r else float("nan")

    # 연속형: battle_center_score ↔ final_rank (eligible 만)
    bc_xs = [float(r["battle_center_score"]) for r in eligible]
    rk_ys = [float(r["final_rank"]) for r in eligible]
    sv_zs = [r.get("survival_seconds") for r in eligible]

    raw_r, raw_p, raw_n = _pearson_safe(bc_xs, rk_ys)
    part_r, part_n = partial_correlation(bc_xs, rk_ys, sv_zs)

    sp_r = float("nan")
    sp_p = float("nan")
    try:
        if len(bc_xs) >= 4 and np.std(bc_xs) > 1e-12:
            sp = spearmanr(bc_xs, rk_ys)
            sp_r = float(sp.correlation)
            sp_p = float(sp.pvalue)
    except Exception:
        pass

    # position_label 분포 (참고용 — eligible 만)
    pos_groups: dict[str, list[float]] = {}
    for r in eligible:
        lab = r.get("position_label", "mid")
        pos_groups.setdefault(lab, []).append(float(r["final_rank"]))
    position_stats = [
        {
            "position": lab,
            "n": len(vs),
            "mean_rank": float(np.mean(vs)) if vs else float("nan"),
            "median_rank": float(np.median(vs)) if vs else float("nan"),
        }
        for lab, vs in pos_groups.items()
    ]

    mean_c = float(np.mean(centers)) if centers else float("nan")
    mean_r = float(np.mean(rest)) if rest else float("nan")
    delta_rank = mean_c - mean_r if (centers and rest) else float("nan")

    verdict, lines = _verdict_h2(
        n_eligible,
        n_excluded_degzero,
        n_c,
        n_r,
        mean_c,
        mean_r,
        mw_p,
        delta,
        raw_r,
        raw_p,
        part_r,
    )

    return {
        "verdict": verdict,
        "conclusion_lines": lines,
        "n_eligible": n_eligible,
        "n_excluded_degzero": n_excluded_degzero,
        "groups": {
            "battle_center": {
                "label_kr": "다중 교전 중심 (상위 25%)",
                "n": n_c,
                "mean_rank": mean_c,
                "median_rank": float(np.median(centers)) if centers else float("nan"),
            },
            "rest": {
                "label_kr": "그 외 (하위 75%)",
                "n": n_r,
                "mean_rank": mean_r,
                "median_rank": float(np.median(rest)) if rest else float("nan"),
            },
        },
        "test": {
            "mw_U": mw_U,
            "mw_p": mw_p,
            "cliffs_delta": delta,
            "delta_rank": delta_rank,
            "score_q_high": q_high,
        },
        "continuous": {
            "pearson_r": raw_r,
            "pearson_p": raw_p,
            "pearson_n": raw_n,
            "partial_r": part_r,
            "partial_n": part_n,
            "spearman_r": sp_r,
            "spearman_p": sp_p,
        },
        "position_stats": position_stats,
    }


# ============================================================
# 결론 생성
# ============================================================


def _verdict_h2(
    n_eligible,
    n_excluded,
    n_c,
    n_r,
    mean_c,
    mean_r,
    mw_p,
    delta,
    raw_r,
    raw_p,
    part_r,
) -> tuple[str, list[str]]:
    lines: list[str] = []
    lines.append(
        f"분석 대상 · degree>0 팀 {n_eligible} (degree=0 비참여 팀 "
        f"{n_excluded} 제외; plan A.0 버킷 오염 회피)"
    )

    if not np.isfinite(mw_p):
        return "inconclusive", lines + ["Mann-Whitney 검정 불가 — 표본 부족."]

    diff = mean_c - mean_r
    lines.append(f"평균 순위 · 다중 교전 중심 {mean_c:.2f} vs 그 외 {mean_r:.2f} (Δ = {diff:+.2f})")
    lines.append(f"Mann-Whitney U p = {mw_p:.4f}{_sig(mw_p)} · Cliff's δ = {delta:+.3f}")

    if abs(diff) >= 1.5 and mw_p < 0.05 and abs(delta) >= 0.15:
        if diff < 0:
            verdict = "battle_center_better"
            lines.append(
                "판정 · **다중 교전 중심에 위치한 팀이 평균적으로 더 좋은 순위**. "
                '여러 교전 클러스터를 가로지르는 위치가 단순히 "많이 싸우는 것" '
                "이상의 결과 향상에 기여."
            )
        else:
            verdict = "battle_center_worse"
            lines.append(
                "판정 · **다중 교전 중심에 위치한 팀이 더 나쁜 순위**. "
                "여러 방향에서 압박이 쏟아지는 노출 효과가 더 컸다."
            )
    else:
        verdict = "no_difference"
        lines.append("판정 · **다중 교전 중심 여부와 평균 순위는 유의미한 차이 없음**.")

    if np.isfinite(raw_r):
        lines.append(
            f"참고 · battle_center_score ↔ 순위 Pearson r = {raw_r:+.3f} "
            f"(p = {raw_p:.4f}{_sig(raw_p)})"
            + (f" / 생존시간 통제 편상관 r = {part_r:+.3f}" if np.isfinite(part_r) else "")
        )

    return verdict, lines
