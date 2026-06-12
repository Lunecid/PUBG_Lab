"""
Team archetype labeler (v2)
============================
세 신규 가설(H1·H2·H3) 이 공유하는 팀 단위 파생 지표 + archetype 라벨.

Stage A — 매치 단위 (compute_per_match_archetype_features):
    각 row 에 in-place 부착:
        dist_from_center_m   : 첫 alive frame 의 팀 centroid ↔ map 중앙 거리
        centroid_first_xy    : 첫 alive frame centroid [x, y]
        combat_per_min       : combat_events / minutes_alive
        dmg_per_min          : dmg_dealt   / minutes_alive

Stage B — 전체 dataset 단위 (assign_archetype_labels):
    모든 매치 row 가 모인 시점에서 dataset 내 quartile 로 archetype 결정.
        combat_participation : z(combat_per_min) + z(dmg_per_min) + z(degree)
        battle_center_score  : z(c_betweenness) + z(c_degree) + z(local_clustering)
        archetype            : "combat_participant" | "outer_farmer" | "neutral"

archetype 규칙:
    - 상위 25% combat_participation                                  → combat_participant
    - 하위 25% combat_participation AND degree <= 1
      AND dist_from_center_m > dataset median                         → outer_farmer
    - 그 외                                                            → neutral
"""

from __future__ import annotations
import math
import numpy as np


# ============================================================
# Stage A — 매치 단위 파생
# ============================================================

def compute_per_match_archetype_features(rows: list[dict], match_data: dict) -> None:
    """단일 매치 내에서 결정 가능한 archetype 보조 feature 부착 (in-place).

    rows 는 build_team_graph 가 반환한 팀 row 목록.
    match_data 는 data_loader.build_match_data() 결과.
    """
    static = match_data["static"]
    frames = match_data.get("frames", [])
    arena_m = float(static.get("arena_size_m") or 8160.0)
    cx_map, cy_map = arena_m / 2.0, arena_m / 2.0

    # team_idx → 첫 non-None centroid (= 매치 시작 시 alive 위치)
    first_centroid: dict[int, tuple[float, float]] = {}
    for f in frames:
        for ti, c in enumerate(f.get("team_centroid") or []):
            if c is None:
                continue
            if ti in first_centroid:
                continue
            first_centroid[ti] = (float(c[0]), float(c[1]))

    for r in rows:
        ti = r["team_idx"]
        c = first_centroid.get(ti)
        if c is not None:
            x, y = c
            r["dist_from_center_m"] = float(math.hypot(x - cx_map, y - cy_map))
            r["centroid_first_xy"] = [round(x, 1), round(y, 1)]
        else:
            r["dist_from_center_m"] = float("nan")
            r["centroid_first_xy"] = None

        surv_min = max(float(r.get("survival_seconds") or 0.0) / 60.0, 1.0 / 60.0)
        r["combat_per_min"] = float(r.get("combat_events", 0)) / surv_min
        r["dmg_per_min"] = float(r.get("dmg_dealt", 0.0)) / surv_min


# ============================================================
# Stage B — 데이터셋 단위 archetype 라벨
# ============================================================

def _z(arr: np.ndarray) -> np.ndarray:
    """안전 z-score. std=0 이면 zero 반환."""
    mu = float(np.nanmean(arr)) if arr.size else 0.0
    sd = float(np.nanstd(arr)) if arr.size else 0.0
    if sd < 1e-12:
        return np.zeros_like(arr, dtype=float)
    out = (arr - mu) / sd
    return np.where(np.isfinite(out), out, 0.0)


def assign_archetype_labels(rows: list[dict]) -> dict:
    """전체 dataset row 에 대해 archetype + composite score 부착 (in-place).

    Returns:
        meta dict: 컷오프, 분포, 메타 통계, 그리고 시간 축 동적 archetype 재계산을
        위한 means/stds.
    """
    if not rows:
        return {"archetype_counts": {}, "n_rows": 0}

    cp = np.array([float(r.get("combat_per_min") or 0.0) for r in rows], dtype=float)
    dp = np.array([float(r.get("dmg_per_min") or 0.0) for r in rows], dtype=float)
    dg = np.array([float(r.get("degree") or 0) for r in rows], dtype=float)

    cp_mean, cp_std = float(np.mean(cp)), float(np.std(cp))
    dp_mean, dp_std = float(np.mean(dp)), float(np.std(dp))
    dg_mean, dg_std = float(np.mean(dg)), float(np.std(dg))

    combat_participation = _z(cp) + _z(dp) + _z(dg)
    for i, r in enumerate(rows):
        r["combat_participation"] = float(combat_participation[i])

    cb = np.array([float(r.get("c_betweenness") or 0.0) for r in rows], dtype=float)
    cd = np.array([float(r.get("c_degree") or 0.0) for r in rows], dtype=float)
    lc = np.array([float(r.get("local_clustering") or 0.0) for r in rows], dtype=float)
    battle_center_score = _z(cb) + _z(cd) + _z(lc)
    for i, r in enumerate(rows):
        r["battle_center_score"] = float(battle_center_score[i])

    # quartile cutoffs (dataset-wide)
    q_high = float(np.percentile(combat_participation, 75))
    q_low = float(np.percentile(combat_participation, 25))

    d_arr = np.array(
        [float(r.get("dist_from_center_m") or float("nan")) for r in rows],
        dtype=float,
    )
    d_finite = d_arr[np.isfinite(d_arr)]
    d_median = float(np.median(d_finite)) if d_finite.size else 0.0

    counts = {"outer_farmer": 0, "combat_participant": 0, "neutral": 0}
    for r in rows:
        cp_v = float(r["combat_participation"])
        d_v = float(r.get("dist_from_center_m") or float("nan"))
        deg = int(r.get("degree") or 0)
        if cp_v >= q_high:
            label = "combat_participant"
        elif (cp_v <= q_low
              and np.isfinite(d_v) and d_v > d_median
              and deg <= 1):
            label = "outer_farmer"
        else:
            label = "neutral"
        r["archetype"] = label
        counts[label] += 1

    return {
        "n_rows": len(rows),
        "q_high_combat_participation": q_high,
        "q_low_combat_participation": q_low,
        "median_dist_from_center_m": d_median,
        "archetype_counts": counts,
        # 동적 (시간 t) archetype 재계산용
        "combat_per_min_mean": cp_mean, "combat_per_min_std": cp_std,
        "dmg_per_min_mean":    dp_mean, "dmg_per_min_std":    dp_std,
        "degree_mean":         dg_mean, "degree_std":         dg_std,
    }


def archetype_label_kr(label: str) -> str:
    return {
        "outer_farmer": "외곽 파밍형",
        "combat_participant": "교전 참여형",
        "neutral": "중간",
    }.get(label, label)
