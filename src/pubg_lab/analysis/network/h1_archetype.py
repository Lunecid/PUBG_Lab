"""
H1 (v2) · 외곽 파밍형 vs 교전 참여형
========================================
"외곽에서 파밍에 집중한 팀과 적극적으로 교전한 팀, 어느 쪽이 평균 순위가
더 좋은가?" 를 archetype 기준 두 극단 그룹으로 검증.

archetype 라벨은 archetype.assign_archetype_labels() 가 미리 부착해야 함.

검증:
    1. Mann-Whitney U  : final_rank (외곽 파밍 vs 교전 참여)
    2. Cliff's delta   : 비모수 효과 크기
    3. 부분 상관       : combat_participation ↔ final_rank | survival_seconds
    4. Spearman ρ      : combat_participation ↔ final_rank
"""

from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from pubg_lab.analysis.network.legacy.h1_analysis import _pearson_safe, partial_correlation
from pubg_lab.analysis.network.archetype import archetype_label_kr


# ============================================================
# 효과 크기
# ============================================================


def cliffs_delta(a: list[float], b: list[float]) -> float:
    """δ = (#(a > b) - #(a < b)) / (n_a * n_b).
    +1: a 가 항상 b 보다 큼 / -1: a 가 항상 b 보다 작음.
    PUBG rank 는 작을수록 좋으므로 δ < 0 이면 a 가 더 좋은 순위.
    """
    a = [x for x in a if x is not None and np.isfinite(x)]
    b = [x for x in b if x is not None and np.isfinite(x)]
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return float("nan")
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    diff = a_arr[:, None] - b_arr[None, :]
    pos = int((diff > 0).sum())
    neg = int((diff < 0).sum())
    return float((pos - neg) / (n_a * n_b))


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


# ============================================================
# 메인
# ============================================================


def compute_h1_archetype(rows: list[dict]) -> dict:
    """H1 (v2) 검증 결과 dict.

    Pre-condition: rows 에 archetype, combat_participation 가 부착되어 있어야 함.
    """
    outer = [
        float(r["final_rank"])
        for r in rows
        if r.get("archetype") == "outer_farmer" and r.get("final_rank") is not None
    ]
    combat = [
        float(r["final_rank"])
        for r in rows
        if r.get("archetype") == "combat_participant" and r.get("final_rank") is not None
    ]
    neutral = [
        float(r["final_rank"])
        for r in rows
        if r.get("archetype") == "neutral" and r.get("final_rank") is not None
    ]

    n_o, n_c, n_n = len(outer), len(combat), len(neutral)

    mw_U = float("nan")
    mw_p = float("nan")
    if n_o >= 3 and n_c >= 3:
        try:
            res = mannwhitneyu(outer, combat, alternative="two-sided")
            mw_U = float(res.statistic)
            mw_p = float(res.pvalue)
        except Exception:
            pass

    delta = cliffs_delta(outer, combat) if n_o and n_c else float("nan")

    # 연속형: combat_participation ↔ final_rank
    cp_xs = [r.get("combat_participation") for r in rows]
    rk_ys = [r.get("final_rank") for r in rows]
    sv_zs = [r.get("survival_seconds") for r in rows]

    raw_r, raw_p, raw_n = _pearson_safe(cp_xs, rk_ys)
    part_r, part_n = partial_correlation(cp_xs, rk_ys, sv_zs)

    sp_r = float("nan")
    sp_p = float("nan")
    try:
        cp_clean, rk_clean = [], []
        for x, y in zip(cp_xs, rk_ys):
            if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
                cp_clean.append(float(x))
                rk_clean.append(float(y))
        if len(cp_clean) >= 4 and np.std(cp_clean) > 1e-12:
            sp = spearmanr(cp_clean, rk_clean)
            sp_r = float(sp.correlation)
            sp_p = float(sp.pvalue)
    except Exception:
        pass

    median_outer = float(np.median(outer)) if outer else float("nan")
    median_combat = float(np.median(combat)) if combat else float("nan")
    mean_outer = float(np.mean(outer)) if outer else float("nan")
    mean_combat = float(np.mean(combat)) if combat else float("nan")
    delta_rank = (mean_outer - mean_combat) if (outer and combat) else float("nan")

    verdict, lines = _verdict_h1(
        n_o,
        n_c,
        mean_outer,
        mean_combat,
        mw_p,
        delta,
        raw_r,
        raw_p,
        part_r,
    )

    return {
        "verdict": verdict,
        "conclusion_lines": lines,
        "groups": {
            "outer_farmer": {
                "label_kr": archetype_label_kr("outer_farmer"),
                "n": n_o,
                "mean_rank": mean_outer,
                "median_rank": median_outer,
            },
            "combat_participant": {
                "label_kr": archetype_label_kr("combat_participant"),
                "n": n_c,
                "mean_rank": mean_combat,
                "median_rank": median_combat,
            },
            "neutral": {
                "label_kr": archetype_label_kr("neutral"),
                "n": n_n,
                "mean_rank": float(np.mean(neutral)) if neutral else float("nan"),
                "median_rank": float(np.median(neutral)) if neutral else float("nan"),
            },
        },
        "test": {
            "mw_U": mw_U,
            "mw_p": mw_p,
            "cliffs_delta": delta,
            "delta_rank": delta_rank,
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
    }


# ============================================================
# 결론 생성
# ============================================================


def _verdict_h1(
    n_o,
    n_c,
    mean_outer,
    mean_combat,
    mw_p,
    delta,
    raw_r,
    raw_p,
    part_r,
) -> tuple[str, list[str]]:
    lines: list[str] = []
    lines.append(f"분석 대상 · 외곽 파밍형 {n_o}팀 vs 교전 참여형 {n_c}팀")

    if n_o < 3 or n_c < 3 or not np.isfinite(mw_p):
        verdict = "inconclusive"
        lines.append("표본이 부족해 두 archetype 의 평균 순위 차이를 판정할 수 없음.")
        return verdict, lines

    diff = mean_outer - mean_combat
    direction = (
        "외곽 파밍형이 더 좋은 순위 (평균 순위 작음)"
        if diff < 0
        else "교전 참여형이 더 좋은 순위 (평균 순위 작음)"
        if diff > 0
        else "차이 없음"
    )

    lines.append(
        f"평균 순위 · 외곽 파밍형 {mean_outer:.2f} vs 교전 참여형 "
        f"{mean_combat:.2f} (Δ = {diff:+.2f}; {direction})"
    )
    lines.append(f"Mann-Whitney U p = {mw_p:.4f}{_sig(mw_p)} · Cliff's δ = {delta:+.3f}")

    if abs(diff) >= 1.5 and mw_p < 0.05 and abs(delta) >= 0.15:
        if diff < 0:
            verdict = "outer_farmer_better"
            lines.append(
                "판정 · **외곽 파밍형이 유의미하게 더 좋은 순위**. "
                "평균적으로 약 "
                f"{abs(diff):.1f}등 더 높은 위치에서 매치를 마친다."
            )
        else:
            verdict = "combat_participant_better"
            lines.append(
                "판정 · **교전 참여형이 유의미하게 더 좋은 순위**. "
                f"평균 약 {abs(diff):.1f}등 더 좋다."
            )
    else:
        verdict = "no_difference"
        lines.append(
            "판정 · **두 archetype 간 유의미한 순위 차이 없음**. "
            "외곽 파밍과 교전 참여 모두 비슷한 평균 결과로 마무리된다."
        )

    if np.isfinite(raw_r):
        lines.append(
            f"참고 · 교전 참여 점수 ↔ 순위 Pearson r = {raw_r:+.3f} "
            f"(p = {raw_p:.4f}{_sig(raw_p)})"
            + (f" / 생존시간 통제 후 편상관 r = {part_r:+.3f}" if np.isfinite(part_r) else "")
        )

    return verdict, lines
