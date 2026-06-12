"""
Team movement / trajectory features (v2)
=========================================
H3 (존버형 vs 활발 이동형) 가설을 위한 팀 단위 이동 지표.

**주의** — trajectory 누적은 LANDING_OFFSET_S (기본 90초) 이후 frame 만 사용.
PUBG 비행기·낙하 구간의 직선 이동은 모든 팀에 공통적인 큰 노이즈를 만들어
지상 이동 차이 검증을 무의미하게 만든다.

Stage A — 매치 단위 (compute_movement_features):
    각 row 에 in-place 부착:
        trajectory_length_m   : 착륙 후 alive frame centroid 의 연속 거리 합
        position_std_m        : 착륙 후 centroid x, y std 의 quadrature 합
        centroid_last_xy      : 마지막 alive frame centroid
        displacement_per_min  : trajectory_length_m / 착륙 후 minutes_alive
        net_displacement_m    : 착륙 후 첫 alive ↔ 마지막 alive 직선 거리
        n_alive_frames        : 착륙 후 alive centroid 가 있는 frame 수
        landing_offset_s      : 사용된 착륙 컷 (메타)

Stage B — 전체 dataset 단위 (assign_movement_labels):
    quartile cutoff 으로 sedentary / mobile / mid 라벨.
        Q1 (하위 25%) → sedentary
        Q4 (상위 25%) → mobile
        그 외          → mid
"""

from __future__ import annotations
import math
import numpy as np

from network_analysis.config import LANDING_OFFSET_S, MOVEMENT_WINDOW_S


# ============================================================
# Stage A — 매치 단위 trajectory feature
# ============================================================

def compute_movement_features(
    rows: list[dict],
    match_data: dict,
    landing_offset_s: float = LANDING_OFFSET_S,
    window_s: float = MOVEMENT_WINDOW_S,
) -> None:
    """단일 매치 frame centroid 시퀀스로부터 팀별 이동 지표 부착 (in-place).

    측정 구간: [landing_offset_s, landing_offset_s + window_s].
    - 비행기·낙하 직선 이동은 모든 팀 공통 노이즈 → 제외
    - 후반 zone 강제 이동은 생존 시간과 강하게 결합된 confounder → 윈도우로 제거
    분당 이동거리는 "그 팀이 윈도우 안에 살아있었던 시간"으로 정규화.
    """
    static = match_data["static"]
    frames = match_data.get("frames", [])
    n_teams = int(static.get("n_teams") or 0)
    window_end = landing_offset_s + window_s

    # team_idx → list[(x,y)] (alive 인 프레임 + 윈도우 [landing, window_end] 안)
    traj: dict[int, list[tuple[float, float]]] = {ti: [] for ti in range(n_teams)}
    for f in frames:
        ft = float(f.get("t", 0.0))
        if ft < landing_offset_s or ft > window_end:
            continue
        for ti, c in enumerate(f.get("team_centroid") or []):
            if c is None:
                continue
            traj[ti].append((float(c[0]), float(c[1])))

    by_idx = {r["team_idx"]: r for r in rows}
    for ti in range(n_teams):
        r = by_idx.get(ti)
        if r is None:
            continue
        pts = traj.get(ti, [])
        n = len(pts)
        r["n_alive_frames"] = int(n)

        if n == 0:
            r["trajectory_length_m"] = 0.0
            r["position_std_m"] = 0.0
            r["net_displacement_m"] = 0.0
            r["centroid_last_xy"] = None
        elif n == 1:
            r["trajectory_length_m"] = 0.0
            r["position_std_m"] = 0.0
            r["net_displacement_m"] = 0.0
            r["centroid_last_xy"] = [round(pts[0][0], 1), round(pts[0][1], 1)]
        else:
            total = 0.0
            for i in range(1, n):
                dx = pts[i][0] - pts[i - 1][0]
                dy = pts[i][1] - pts[i - 1][1]
                total += math.hypot(dx, dy)
            xs = np.array([p[0] for p in pts], dtype=float)
            ys = np.array([p[1] for p in pts], dtype=float)
            r["trajectory_length_m"] = float(total)
            r["position_std_m"] = float(math.hypot(np.std(xs), np.std(ys)))
            r["net_displacement_m"] = float(math.hypot(
                pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]
            ))
            r["centroid_last_xy"] = [round(pts[-1][0], 1), round(pts[-1][1], 1)]

        surv_s = float(r.get("survival_seconds") or 0.0)
        # 윈도우 안에서 alive 였던 시간 (초): max(0, min(surv, window_end) - landing)
        in_window_s = max(min(surv_s, window_end) - landing_offset_s, 0.0)
        denom_min = max(in_window_s / 60.0, 1.0 / 60.0)
        r["displacement_per_min"] = float(r["trajectory_length_m"]) / denom_min
        r["landing_offset_s"] = float(landing_offset_s)
        r["movement_window_s"] = float(window_s)
        r["in_window_alive_s"] = float(in_window_s)


# ============================================================
# Stage B — 데이터셋 단위 movement 라벨
# ============================================================

def assign_movement_labels(rows: list[dict]) -> dict:
    """전체 dataset 에 대해 displacement_per_min quartile 로 라벨 부착 (in-place).

    Returns:
        meta dict: cutoff 값, 분포 카운트
    """
    if not rows:
        return {"movement_counts": {}, "n_rows": 0}

    vals = np.array(
        [float(r.get("displacement_per_min") or 0.0) for r in rows],
        dtype=float,
    )
    finite = vals[np.isfinite(vals)]
    if finite.size < 4:
        for r in rows:
            r["movement_label"] = "mid"
        return {
            "n_rows": len(rows),
            "movement_counts": {"sedentary": 0, "mid": len(rows), "mobile": 0},
            "q1_displacement_per_min": float("nan"),
            "q3_displacement_per_min": float("nan"),
        }

    q1 = float(np.percentile(finite, 25))
    q3 = float(np.percentile(finite, 75))
    dpm_mean = float(np.mean(finite))
    dpm_std = float(np.std(finite))

    counts = {"sedentary": 0, "mid": 0, "mobile": 0}
    for r in rows:
        v = float(r.get("displacement_per_min") or 0.0)
        if not np.isfinite(v):
            label = "mid"
        elif v <= q1:
            label = "sedentary"
        elif v >= q3:
            label = "mobile"
        else:
            label = "mid"
        r["movement_label"] = label
        counts[label] += 1

    return {
        "n_rows": len(rows),
        "movement_counts": counts,
        "q1_displacement_per_min": q1,
        "q3_displacement_per_min": q3,
        "displacement_per_min_mean": dpm_mean,
        "displacement_per_min_std":  dpm_std,
    }


def movement_label_kr(label: str) -> str:
    return {
        "sedentary": "존버형 (저 이동)",
        "mid":       "중간",
        "mobile":    "활발 이동형",
    }.get(label, label)
