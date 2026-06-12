"""
H3 · 네트워크 위치(외곽 vs 중심)와 생존의 관계
================================================
팀을 교전 네트워크 상 위치로 분류하여 평균 생존시간과 순위를 비교.

포지션 레이블 (hypotheses.py::attach_community_features 에서 부여):
  isolated   — degree = 0, 아예 교전이 없던 팀
  peripheral — betweenness 하위 25%, 연결은 있지만 주변부
  mid        — 중간
  core       — betweenness 상위 25%, 네트워크 중심

검증:
  - 그룹별 평균 survival_seconds, 평균 final_rank
  - "외곽 · 고립 팀이 오래 사는가?" → survival 비교
  - "중심 팀이 순위가 좋은가?" → rank 비교
  - Kruskal–Wallis + 쌍별 Mann–Whitney
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kruskal, mannwhitneyu, pearsonr


POSITION_ORDER = ["isolated", "peripheral", "mid", "core"]

POSITION_LABEL_KR = {
    "isolated":   "고립 (교전 0회)",
    "peripheral": "외곽 (betweenness 하위 25%)",
    "mid":        "중간",
    "core":       "중심 (betweenness 상위 25%)",
}


def _sig(p):
    if p is None or not np.isfinite(p): return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def compute_h3(rows: list[dict]) -> dict:
    groups_rank: dict[str, list[float]] = {g: [] for g in POSITION_ORDER}
    groups_surv: dict[str, list[float]] = {g: [] for g in POSITION_ORDER}
    for r in rows:
        pos = r.get("position_label")
        rk = r.get("final_rank")
        sv = r.get("survival_seconds")
        if pos not in groups_rank: continue
        if rk is not None:
            groups_rank[pos].append(float(rk))
        if sv is not None:
            groups_surv[pos].append(float(sv))

    group_stats = []
    for pos in POSITION_ORDER:
        ranks = groups_rank[pos]
        survs = groups_surv[pos]
        group_stats.append({
            "pos": pos,
            "label_kr": POSITION_LABEL_KR[pos],
            "n": len(ranks),
            "mean_rank":   float(np.mean(ranks)) if ranks else float("nan"),
            "median_rank": float(np.median(ranks)) if ranks else float("nan"),
            "mean_surv_s": float(np.mean(survs)) if survs else float("nan"),
            "median_surv_s": float(np.median(survs)) if survs else float("nan"),
        })

    # Kruskal on rank
    kw_rank_stat = kw_rank_p = float("nan")
    nonempty_r = [v for v in groups_rank.values() if len(v) >= 3]
    if len(nonempty_r) >= 2:
        try:
            kw = kruskal(*nonempty_r)
            kw_rank_stat, kw_rank_p = float(kw.statistic), float(kw.pvalue)
        except Exception:
            pass

    # Kruskal on survival
    kw_surv_stat = kw_surv_p = float("nan")
    nonempty_s = [v for v in groups_surv.values() if len(v) >= 3]
    if len(nonempty_s) >= 2:
        try:
            kw = kruskal(*nonempty_s)
            kw_surv_stat, kw_surv_p = float(kw.statistic), float(kw.pvalue)
        except Exception:
            pass

    # 쌍별 비교: peripheral vs core, isolated vs core
    pairwise = {}
    for a, b in [("peripheral", "core"),
                 ("isolated", "core"),
                 ("isolated", "peripheral")]:
        rw_a, rw_b = groups_rank[a], groups_rank[b]
        sv_a, sv_b = groups_surv[a], groups_surv[b]
        entry = {"n_a": len(rw_a), "n_b": len(rw_b)}
        if len(rw_a) >= 3 and len(rw_b) >= 3:
            try:
                mw = mannwhitneyu(rw_a, rw_b, alternative="two-sided")
                entry["rank_mean_a"]  = float(np.mean(rw_a))
                entry["rank_mean_b"]  = float(np.mean(rw_b))
                entry["rank_mw_p"]    = float(mw.pvalue)
            except Exception:
                pass
        if len(sv_a) >= 3 and len(sv_b) >= 3:
            try:
                mw = mannwhitneyu(sv_a, sv_b, alternative="two-sided")
                entry["surv_mean_a"]  = float(np.mean(sv_a))
                entry["surv_mean_b"]  = float(np.mean(sv_b))
                entry["surv_mw_p"]    = float(mw.pvalue)
            except Exception:
                pass
        pairwise[f"{a}_vs_{b}"] = entry

    verdict, lines = _verdict_h3(group_stats, pairwise)

    return {
        "group_stats": group_stats,
        "kruskal_rank": {"statistic": kw_rank_stat, "p": kw_rank_p},
        "kruskal_surv": {"statistic": kw_surv_stat, "p": kw_surv_p},
        "pairwise": pairwise,
        "verdict": verdict,
        "conclusion_lines": lines,
    }


def _verdict_h3(group_stats, pairwise) -> tuple[str, list[str]]:
    lines = []
    by_pos = {g["pos"]: g for g in group_stats}

    # 중심 vs 외곽 순위 비교
    pw = pairwise.get("peripheral_vs_core") or {}
    rank_p = pw.get("rank_mw_p")
    core_mean = by_pos.get("core", {}).get("mean_rank")
    peri_mean = by_pos.get("peripheral", {}).get("mean_rank")

    verdict = "inconclusive"
    if core_mean is not None and peri_mean is not None \
       and np.isfinite(core_mean) and np.isfinite(peri_mean):
        diff = core_mean - peri_mean  # 음수면 core 가 더 좋은 순위
        if rank_p is not None and np.isfinite(rank_p):
            if diff < -1.0 and rank_p < 0.05:
                verdict = "core_better"
                lines.append(
                    f"중심(core) 팀 평균순위 {core_mean:.1f}, "
                    f"외곽(peripheral) 팀 평균순위 {peri_mean:.1f} — "
                    f"중심 팀이 {abs(diff):.1f}등 우위 "
                    f"(Mann–Whitney p = {rank_p:.4f}{_sig(rank_p)})."
                )
            elif diff > 1.0 and rank_p < 0.05:
                verdict = "periphery_better"
                lines.append(
                    f"외곽(peripheral) 팀 평균순위 {peri_mean:.1f}, "
                    f"중심(core) 팀 평균순위 {core_mean:.1f} — "
                    f"외곽이 {diff:.1f}등 우위 "
                    f"(Mann–Whitney p = {rank_p:.4f}{_sig(rank_p)}). "
                    "중심 교전지에 오래 머문 팀들이 오히려 불리."
                )
            else:
                lines.append(
                    f"중심(core) 팀 평균순위 {core_mean:.1f} vs 외곽 {peri_mean:.1f} — "
                    f"유의한 차이 없음 (p = {rank_p:.4f})."
                )

    # 고립 팀의 생존시간 비교 (파밍형 시그니처)
    pw2 = pairwise.get("isolated_vs_core") or {}
    iso_surv = by_pos.get("isolated", {}).get("mean_surv_s")
    core_surv = by_pos.get("core", {}).get("mean_surv_s")
    surv_p = pw2.get("surv_mw_p")
    if iso_surv is not None and core_surv is not None \
       and np.isfinite(iso_surv) and np.isfinite(core_surv):
        diff_min = (iso_surv - core_surv) / 60.0
        if surv_p is not None and np.isfinite(surv_p):
            if diff_min < -3.0 and surv_p < 0.05:
                lines.append(
                    f"고립 팀 평균 생존 {iso_surv/60:.1f}분, "
                    f"중심 팀 {core_surv/60:.1f}분 — "
                    f"고립 팀이 {abs(diff_min):.1f}분 빨리 탈락 "
                    f"(p = {surv_p:.4f}{_sig(surv_p)}). "
                    "파밍만 하다 존에 밀려 빨리 죽는 패턴."
                )
            elif diff_min > 3.0 and surv_p < 0.05:
                lines.append(
                    f"고립 팀 평균 생존 {iso_surv/60:.1f}분, "
                    f"중심 팀 {core_surv/60:.1f}분 — "
                    f"고립 팀이 {diff_min:.1f}분 더 오래 생존 "
                    f"(p = {surv_p:.4f}{_sig(surv_p)}). "
                    "파밍형 플레이가 생존 자체는 길게 유지."
                )
            else:
                lines.append(
                    f"고립 팀 평균 생존 {iso_surv/60:.1f}분, "
                    f"중심 팀 {core_surv/60:.1f}분 — 큰 차이 없음."
                )

    if not lines:
        lines.append("유의한 그룹 차이를 검출하지 못함 (표본 부족 가능).")

    return verdict, lines
