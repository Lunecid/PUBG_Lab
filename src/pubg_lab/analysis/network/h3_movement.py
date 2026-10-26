"""
H3 (v2) · 존버형 vs 활발 이동형 평균 순위 비교
==================================================
프레임이 지나도 좌표 변화가 적은 "존버형(sedentary)" 팀과 끊임없이 이동하며
교전을 활발히 하는 "활발 이동형(mobile)" 팀의 평균 순위는 어떻게 다른가?

라벨 (movement.assign_movement_labels 가 부착):
    sedentary : displacement_per_min Q1 (하위 25%)
    mobile    : displacement_per_min Q4 (상위 25%)
    mid       : 그 외

검증:
    1. Mann-Whitney U   : final_rank (sedentary vs mobile)
    2. Cliff's δ        : 비모수 효과 크기
    3. Spearman ρ       : displacement_per_min ↔ final_rank
    4. 부분 상관        : displacement_per_min ↔ final_rank | survival_seconds
    5. 보조 비교        : Mann-Whitney on survival_seconds
       (혹시 단순히 일찍 죽어서 안 움직인 것이 아닌지 확인)
    6. archetype 교차표 : archetype × movement_label (4분면)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from pubg_lab.analysis.network.legacy.h1_analysis import _pearson_safe, partial_correlation
from pubg_lab.analysis.network.movement import movement_label_kr


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


def compute_h3_movement(rows: list[dict]) -> dict:
    """H3 (v2) 검증 결과 dict.

    Pre-condition: rows 에 movement_label, displacement_per_min, archetype 가
    이미 부착되어 있어야 함.
    """
    sed = [
        float(r["final_rank"])
        for r in rows
        if r.get("movement_label") == "sedentary" and r.get("final_rank") is not None
    ]
    mob = [
        float(r["final_rank"])
        for r in rows
        if r.get("movement_label") == "mobile" and r.get("final_rank") is not None
    ]
    mid = [
        float(r["final_rank"])
        for r in rows
        if r.get("movement_label") == "mid" and r.get("final_rank") is not None
    ]

    n_s, n_m, n_mi = len(sed), len(mob), len(mid)

    mw_U = float("nan")
    mw_p = float("nan")
    if n_s >= 3 and n_m >= 3:
        try:
            res = mannwhitneyu(sed, mob, alternative="two-sided")
            mw_U = float(res.statistic)
            mw_p = float(res.pvalue)
        except Exception:
            pass
    delta = _cliffs_delta(sed, mob) if n_s and n_m else float("nan")

    # 보조: survival_seconds 비교 (단순 빨리 죽어서 안 움직인 것은 아닌지)
    sed_sv = [
        float(r["survival_seconds"])
        for r in rows
        if r.get("movement_label") == "sedentary" and r.get("survival_seconds") is not None
    ]
    mob_sv = [
        float(r["survival_seconds"])
        for r in rows
        if r.get("movement_label") == "mobile" and r.get("survival_seconds") is not None
    ]
    surv_mw_p = float("nan")
    surv_delta = float("nan")
    if len(sed_sv) >= 3 and len(mob_sv) >= 3:
        try:
            res2 = mannwhitneyu(sed_sv, mob_sv, alternative="two-sided")
            surv_mw_p = float(res2.pvalue)
            surv_delta = float(np.mean(sed_sv) - np.mean(mob_sv))
        except Exception:
            pass

    # 연속형: displacement_per_min ↔ final_rank
    dpm_xs = [r.get("displacement_per_min") for r in rows]
    rk_ys = [r.get("final_rank") for r in rows]
    sv_zs = [r.get("survival_seconds") for r in rows]

    raw_r, raw_p, raw_n = _pearson_safe(dpm_xs, rk_ys)
    part_r, part_n = partial_correlation(dpm_xs, rk_ys, sv_zs)

    sp_r = float("nan")
    sp_p = float("nan")
    try:
        clean_x, clean_y = [], []
        for x, y in zip(dpm_xs, rk_ys):
            if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
                clean_x.append(float(x))
                clean_y.append(float(y))
        if len(clean_x) >= 4 and np.std(clean_x) > 1e-12:
            sp = spearmanr(clean_x, clean_y)
            sp_r = float(sp.correlation)
            sp_p = float(sp.pvalue)
    except Exception:
        pass

    # archetype × movement 교차표
    cross_keys = [
        ("outer_farmer", "sedentary"),
        ("outer_farmer", "mid"),
        ("outer_farmer", "mobile"),
        ("neutral", "sedentary"),
        ("neutral", "mid"),
        ("neutral", "mobile"),
        ("combat_participant", "sedentary"),
        ("combat_participant", "mid"),
        ("combat_participant", "mobile"),
    ]
    cross: dict[str, dict] = {}
    for arc, mvm in cross_keys:
        sub = [
            float(r["final_rank"])
            for r in rows
            if r.get("archetype") == arc
            and r.get("movement_label") == mvm
            and r.get("final_rank") is not None
        ]
        cross[f"{arc}__{mvm}"] = {
            "archetype": arc,
            "movement": mvm,
            "n": len(sub),
            "mean_rank": float(np.mean(sub)) if sub else float("nan"),
            "median_rank": float(np.median(sub)) if sub else float("nan"),
        }

    mean_sed = float(np.mean(sed)) if sed else float("nan")
    mean_mob = float(np.mean(mob)) if mob else float("nan")
    delta_rank = mean_sed - mean_mob if (sed and mob) else float("nan")

    verdict, lines = _verdict_h3(
        n_s,
        n_m,
        mean_sed,
        mean_mob,
        mw_p,
        delta,
        surv_mw_p,
        surv_delta,
        raw_r,
        raw_p,
        part_r,
    )

    return {
        "verdict": verdict,
        "conclusion_lines": lines,
        "groups": {
            "sedentary": {
                "label_kr": movement_label_kr("sedentary"),
                "n": n_s,
                "mean_rank": mean_sed,
                "median_rank": float(np.median(sed)) if sed else float("nan"),
                "mean_surv_s": float(np.mean(sed_sv)) if sed_sv else float("nan"),
            },
            "mid": {
                "label_kr": movement_label_kr("mid"),
                "n": n_mi,
                "mean_rank": float(np.mean(mid)) if mid else float("nan"),
                "median_rank": float(np.median(mid)) if mid else float("nan"),
            },
            "mobile": {
                "label_kr": movement_label_kr("mobile"),
                "n": n_m,
                "mean_rank": mean_mob,
                "median_rank": float(np.median(mob)) if mob else float("nan"),
                "mean_surv_s": float(np.mean(mob_sv)) if mob_sv else float("nan"),
            },
        },
        "test": {
            "mw_U": mw_U,
            "mw_p": mw_p,
            "cliffs_delta": delta,
            "delta_rank": delta_rank,
            "survival_mw_p": surv_mw_p,
            "survival_delta_s": surv_delta,
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
        "cross_archetype_movement": cross,
    }


def _verdict_h3(
    n_s,
    n_m,
    mean_sed,
    mean_mob,
    mw_p,
    delta,
    surv_mw_p,
    surv_delta,
    raw_r,
    raw_p,
    part_r,
) -> tuple[str, list[str]]:
    lines: list[str] = []
    lines.append(f"분석 대상 · 존버형 {n_s} 팀 vs 활발 이동형 {n_m} 팀")

    if not np.isfinite(mw_p):
        return "inconclusive", lines + ["표본이 부족해 두 그룹 비교 불가."]

    diff = mean_sed - mean_mob
    lines.append(
        f"평균 순위 · 존버형 {mean_sed:.2f} vs 활발 이동형 {mean_mob:.2f} (Δ = {diff:+.2f})"
    )
    lines.append(f"Mann-Whitney U p = {mw_p:.4f}{_sig(mw_p)} · Cliff's δ = {delta:+.3f}")
    if np.isfinite(surv_mw_p):
        lines.append(
            f"보조 검정 · 생존시간 차이 Δ = {surv_delta:+.0f}s "
            f"(Mann-Whitney p = {surv_mw_p:.4f}{_sig(surv_mw_p)})"
            + (
                ' — 두 그룹 생존시간이 비슷할수록 "단순히 일찍 죽어서 안 움직인 게 아님" 신호.'
                if abs(surv_delta) < 60
                else ""
            )
        )

    if abs(diff) >= 1.5 and mw_p < 0.05 and abs(delta) >= 0.15:
        if diff < 0:
            verdict = "sedentary_better"
            lines.append(
                "판정 · **존버형이 평균적으로 더 좋은 순위**. "
                f"평균 약 {abs(diff):.1f}등 더 좋은 결과로 마무리."
            )
        else:
            verdict = "mobile_better"
            lines.append(
                "판정 · **활발 이동형이 평균적으로 더 좋은 순위**. "
                f"평균 약 {abs(diff):.1f}등 더 좋다."
            )
    else:
        verdict = "no_difference"
        lines.append("판정 · **존버 vs 활발 이동 사이에 유의미한 평균 순위 차이 없음**.")

    if np.isfinite(raw_r):
        lines.append(
            f"참고 · 분당 이동거리 ↔ 순위 Pearson r = {raw_r:+.3f} "
            f"(p = {raw_p:.4f}{_sig(raw_p)})"
            + (f" / 생존시간 통제 편상관 r = {part_r:+.3f}" if np.isfinite(part_r) else "")
        )

    return verdict, lines
