"""
H2 · 교전 밀집도와 최종 순위의 관계
====================================
"팀이 속한 교전 클러스터가 조밀할수록 순위가 높은가, 아니면 파밍형(고립)
팀이 유리한가?" 를 검증한다.

측정 축:
  1. community_size  — 소속 커뮤니티의 팀 수 (1 = 고립/파밍, 5+ = 대규모 교전지)
  2. community_density — 소속 커뮤니티 서브그래프의 density
  3. local_clustering — 이 팀 주변의 삼각 교전 밀도

검증:
  - 연속형 상관: community_density ↔ final_rank, local_clustering ↔ final_rank
  - 카테고리 비교: 고립(size=1) vs 소규모(2-3) vs 중간(4-5) vs 대규모(6+) 평균 순위
  - ANOVA + 쌍별 Mann-Whitney
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr, kruskal, mannwhitneyu

from network_analysis.legacy.h1_analysis import _pearson_safe, partial_correlation


def _bucket_size(s: int) -> str:
    if s <= 1: return "isolated"
    if s <= 3: return "small"
    if s <= 5: return "medium"
    return "large"


SIZE_BUCKET_LABEL_KR = {
    "isolated": "고립 (크기 1, 파밍형)",
    "small":    "소규모 (크기 2–3)",
    "medium":   "중간 (크기 4–5)",
    "large":    "대규모 (크기 6+)",
}


def compute_h2(rows: list[dict]) -> dict:
    """H2 통계 결과 dict 를 반환."""
    # 1) 연속형 상관 (density/local_clustering ↔ rank)
    ranks = [r.get("final_rank") for r in rows]
    survs = [r.get("survival_seconds") for r in rows]

    feats = {
        "community_density": "커뮤니티 내부 밀도",
        "local_clustering":  "팀 주변 삼각 교전 밀도",
        "community_size":    "소속 커뮤니티 크기",
        "within_deg":        "같은 커뮤니티 내 누적 데미지",
        "across_deg":        "다른 커뮤니티와의 누적 데미지",
    }
    corr_rows = []
    for key, label in feats.items():
        xs = [r.get(key) for r in rows]
        raw_r, raw_p, raw_n = _pearson_safe(xs, ranks)
        part_r, part_n = partial_correlation(xs, ranks, survs)
        corr_rows.append({
            "feature": key, "label": label,
            "raw_r": raw_r, "raw_p": raw_p, "raw_n": raw_n,
            "partial_r": part_r, "partial_n": part_n,
        })

    # 2) 크기 버킷 비교
    groups: dict[str, list[float]] = {
        "isolated": [], "small": [], "medium": [], "large": [],
    }
    groups_surv: dict[str, list[float]] = {
        "isolated": [], "small": [], "medium": [], "large": [],
    }
    for r in rows:
        sz = int(r.get("community_size") or 0)
        rk = r.get("final_rank")
        sv = r.get("survival_seconds")
        if rk is None: continue
        g = _bucket_size(sz)
        groups[g].append(float(rk))
        if sv is not None:
            groups_surv[g].append(float(sv))

    group_stats = []
    for g, vals in groups.items():
        group_stats.append({
            "bucket": g,
            "label_kr": SIZE_BUCKET_LABEL_KR[g],
            "n": len(vals),
            "mean_rank":   float(np.mean(vals)) if vals else float("nan"),
            "median_rank": float(np.median(vals)) if vals else float("nan"),
            "mean_surv_s": float(np.mean(groups_surv[g])) if groups_surv[g] else float("nan"),
        })

    # Kruskal–Wallis (정규성 가정 안 함, 세 개 이상 그룹 비교에 적합)
    nonempty = [vals for vals in groups.values() if len(vals) >= 3]
    kw_stat = kw_p = float("nan")
    if len(nonempty) >= 2:
        try:
            kw_res = kruskal(*nonempty)
            kw_stat = float(kw_res.statistic)
            kw_p = float(kw_res.pvalue)
        except Exception:
            pass

    # 쌍별: isolated vs large — 가장 극적인 비교
    pairwise = {}
    for a, b in [("isolated", "large"), ("isolated", "medium"),
                 ("small", "large")]:
        va, vb = groups[a], groups[b]
        if len(va) >= 3 and len(vb) >= 3:
            try:
                mw = mannwhitneyu(va, vb, alternative="two-sided")
                pairwise[f"{a}_vs_{b}"] = {
                    "n_a": len(va), "n_b": len(vb),
                    "mean_a": float(np.mean(va)),
                    "mean_b": float(np.mean(vb)),
                    "diff":   float(np.mean(va) - np.mean(vb)),
                    "mw_U":   float(mw.statistic),
                    "mw_p":   float(mw.pvalue),
                }
            except Exception:
                pass

    # 결론 생성
    verdict, lines = _verdict_h2(corr_rows, group_stats, pairwise)

    return {
        "corr_rows": corr_rows,
        "group_stats": group_stats,
        "kruskal": {"statistic": kw_stat, "p": kw_p},
        "pairwise": pairwise,
        "verdict": verdict,
        "conclusion_lines": lines,
    }


def _sig(p):
    if p is None or not np.isfinite(p): return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def _verdict_h2(corr_rows, group_stats, pairwise) -> tuple[str, list[str]]:
    lines = []

    # isolated vs large 비교
    pw = pairwise.get("isolated_vs_large") or pairwise.get("isolated_vs_medium")
    gs = {g["bucket"]: g for g in group_stats}

    if pw and np.isfinite(pw["mw_p"]):
        diff = pw["diff"]
        if diff < -1.5 and pw["mw_p"] < 0.05:
            verdict = "isolated_better"
            lines.append(
                "파밍형(고립 팀) 이 대규모 교전지 팀보다 평균 순위가 "
                f"{abs(diff):.1f}등 더 좋음 (Mann–Whitney p = "
                f"{pw['mw_p']:.4f}{_sig(pw['mw_p'])}). 외곽에서 안전하게 "
                "파밍한 전략이 유의미하게 유리."
            )
        elif diff > 1.5 and pw["mw_p"] < 0.05:
            verdict = "dense_better"
            lines.append(
                "대규모 교전지 소속 팀이 고립 팀보다 평균 순위가 "
                f"{diff:.1f}등 더 좋음 (Mann–Whitney p = "
                f"{pw['mw_p']:.4f}{_sig(pw['mw_p'])}). 파밍보다는 "
                "교전 진영에 섞이는 쪽이 유리."
            )
        else:
            verdict = "no_difference"
            lines.append(
                f"고립 팀과 대규모 교전지 팀의 평균 순위 차이는 "
                f"{diff:+.1f}등으로 통계적으로 유의하지 않음 "
                f"(Mann–Whitney p = {pw['mw_p']:.4f})."
            )
    else:
        verdict = "inconclusive"
        lines.append("표본이 부족하여 그룹 비교를 판정할 수 없음.")

    # density 상관
    d_row = next((c for c in corr_rows
                  if c["feature"] == "community_density"), None)
    if d_row and np.isfinite(d_row["raw_r"]):
        r = d_row["raw_r"]; p = d_row["raw_p"]; pr = d_row["partial_r"]
        if abs(r) > 0.15 and p < 0.05:
            dir_txt = "높을수록 순위가 좋음" if r < 0 else "높을수록 순위가 나쁨"
            lines.append(
                f"커뮤니티 밀도 ↔ 순위 상관 r = {r:+.3f} "
                f"(p = {p:.4f}{_sig(p)}). 밀도가 {dir_txt}."
                + (f" 생존시간 통제 후 편상관 r = {pr:+.3f}." if np.isfinite(pr) else "")
            )
        else:
            lines.append(
                f"커뮤니티 밀도 자체는 순위와 유의한 상관이 없음 "
                f"(r = {r:+.3f}, p = {p:.4f})."
            )

    return verdict, lines
