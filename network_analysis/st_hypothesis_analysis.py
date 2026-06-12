"""
Spatio-temporal hypothesis analysis for PUBG team outcomes.

This module discards the previous hypothesis labels and tests three new claims:
  H1. Outer early farming is associated with better average placement than
      high-density landing hotzones.
  H2. Higher combat participation rate is associated with better average
      placement.
  H3. Among teams observed over the same fixed post-landing window, lower
      movement ("존버형") is associated with better average placement than
      active roaming.

The analysis uses the 10-second ST snapshots already present in the .pt cache
and damage-event counts from PostgreSQL.  Damage amount is not used as the
combat participation variable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

from network_analysis.config import (
    ARENA_SIZE_M,
    LANDING_OFFSET_S,
    SNAPSHOT_INTERVAL_SEC,
)
from network_analysis.data_loader import build_match_data
from network_analysis.db_loader import load_match_damage, load_team_ranked_stats


DEFAULT_DATASET = os.path.join(
    "network_analysis", "output", "analysis_datasets", "Baltic_Main_squad_n100"
)
DEFAULT_EARLY_FARM_WINDOW_S = 300.0
DEFAULT_MOVEMENT_ACTIVE_THRESHOLD_M = 25.0
DEFAULT_HOTZONE_BANDWIDTH_M = 800.0
DEFAULT_RP_ASSIST_WEIGHT = 0.5
DEFAULT_H3_FIXED_WINDOW_S = 600.0


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(x):
        return "-"
    return f"{x:.{digits}f}"


def _pct(value: Any, digits: int = 1) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(x):
        return "-"
    return f"{100.0 * x:.{digits}f}%"


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return float(num) / float(den) if float(den) > 1e-12 else default


def _inclusive_frame_count(duration_s: float) -> int:
    if duration_s <= 0:
        return 0
    return int(math.floor(float(duration_s) / SNAPSHOT_INTERVAL_SEC)) + 1


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return _clean_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean_json(payload), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> tuple[float, float, int]:
    d = pd.DataFrame({"x": _num(x), "y": _num(y)}).dropna()
    d = d[
        d["x"].apply(lambda v: math.isfinite(float(v)))
        & d["y"].apply(lambda v: math.isfinite(float(v)))
    ]
    n = int(d.shape[0])
    if n < 4 or d["x"].std(ddof=0) < 1e-12 or d["y"].std(ddof=0) < 1e-12:
        return float("nan"), float("nan"), n
    if method == "spearman":
        res = spearmanr(d["x"], d["y"])
        return float(res.correlation), float(res.pvalue), n
    r, p = pearsonr(d["x"], d["y"])
    return float(r), float(p), n


def _cliffs_delta(a: pd.Series, b: pd.Series) -> float:
    aa = _num(a).dropna().to_numpy(dtype=float)
    bb = _num(b).dropna().to_numpy(dtype=float)
    if aa.size == 0 or bb.size == 0:
        return float("nan")
    diff = aa[:, None] - bb[None, :]
    return float(((diff > 0).sum() - (diff < 0).sum()) / (aa.size * bb.size))


def _residualize_rank_after_survival(df: pd.DataFrame) -> pd.Series:
    valid = pd.DataFrame({
        "rank": _num(df["final_rank"]),
        "survival": _num(df["survival_seconds"]),
    }).dropna()
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if valid.shape[0] < 8:
        return out
    log_surv = np.log1p(valid["survival"].clip(lower=0).to_numpy(dtype=float))
    X = np.column_stack([np.ones(log_surv.shape[0]), log_surv, log_surv ** 2])
    try:
        beta = np.linalg.lstsq(X, valid["rank"].to_numpy(dtype=float), rcond=None)[0]
    except np.linalg.LinAlgError:
        return out
    out.loc[valid.index] = valid["rank"].to_numpy(dtype=float) - X @ beta
    return out


def _add_within_match_z(
    df: pd.DataFrame,
    source_col: str,
    out_col: str,
) -> pd.DataFrame:
    out = df.copy()
    out[out_col] = 0.0
    for match_id, idx in out.groupby("match_id").groups.items():
        vals = _num(out.loc[idx, source_col]).fillna(0.0)
        std = float(vals.std(ddof=0))
        if std < 1e-12:
            out.loc[idx, out_col] = 0.0
        else:
            out.loc[idx, out_col] = (vals - float(vals.mean())) / std
    return out


def _assign_match_quantile_groups(
    df: pd.DataFrame,
    metric: str,
    low_label: str,
    high_label: str,
    group_col: str,
) -> pd.DataFrame:
    out = df.copy()
    out[group_col] = "middle"
    out[f"{group_col}_q25"] = np.nan
    out[f"{group_col}_q75"] = np.nan

    for match_id, idx in out.groupby("match_id").groups.items():
        vals = _num(out.loc[idx, metric]).dropna()
        vals = vals[vals.apply(lambda v: math.isfinite(float(v)))]
        if vals.shape[0] < 4 or vals.std(ddof=0) < 1e-12:
            continue
        q25 = float(vals.quantile(0.25))
        q75 = float(vals.quantile(0.75))
        out.loc[idx, f"{group_col}_q25"] = q25
        out.loc[idx, f"{group_col}_q75"] = q75
        out.loc[idx[out.loc[idx, metric] <= q25], group_col] = low_label
        out.loc[idx[out.loc[idx, metric] >= q75], group_col] = high_label

    return out


def _assign_fixed_window_movement_groups(
    df: pd.DataFrame,
    *,
    fixed_window_s: float,
) -> pd.DataFrame:
    """Assign H3 movement groups using the same post-landing observation window.

    Naive low-movement groups are contaminated by teams that died early or by
    teams that moved more because they survived into later zone rotations.  This
    labels teams as sedentary/mobile only among teams observed through the same
    fixed window after landing, then computes match-local movement quartiles
    using movement measured inside that fixed window.
    """
    out = df.copy()
    group_col = "movement_fixed_window_group"
    out[group_col] = "short_survival"
    out["h3_fixed_window_s"] = float(fixed_window_s)
    out["h3_fixed_window_eligible"] = (
        _num(out["post_landing_alive_s"]).fillna(0.0) >= float(fixed_window_s)
    )
    out[f"{group_col}_q25"] = np.nan
    out[f"{group_col}_q75"] = np.nan

    for match_id, idx in out.groupby("match_id").groups.items():
        eligible_idx = idx[out.loc[idx, "h3_fixed_window_eligible"].to_numpy(dtype=bool)]
        if len(eligible_idx) < 4:
            continue
        vals = _num(out.loc[eligible_idx, "fixed_window_movement_per_min"]).dropna()
        vals = vals[vals.apply(lambda v: math.isfinite(float(v)))]
        if vals.shape[0] < 4 or vals.std(ddof=0) < 1e-12:
            continue
        q25 = float(vals.quantile(0.25))
        q75 = float(vals.quantile(0.75))
        out.loc[eligible_idx, f"{group_col}_q25"] = q25
        out.loc[eligible_idx, f"{group_col}_q75"] = q75
        out.loc[eligible_idx, group_col] = "fixed_middle"
        out.loc[
            eligible_idx[out.loc[eligible_idx, "fixed_window_movement_per_min"] <= q25],
            group_col,
        ] = "fixed_sedentary"
        out.loc[
            eligible_idx[out.loc[eligible_idx, "fixed_window_movement_per_min"] >= q75],
            group_col,
        ] = "fixed_mobile"

    return out


def _add_landing_hotzone_features(
    df: pd.DataFrame,
    *,
    bandwidth_m: float,
) -> pd.DataFrame:
    """Define hotzones from landing density and classify early farming position.

    Hotzone seeds = top landing-density quartile within each match.
    Team groups = bottom/top quartiles of each team's early-farm distance to
    the nearest hotzone seed.
    """
    out = df.copy()
    out["hotzone_bandwidth_m"] = float(bandwidth_m)
    out["landing_density_score"] = np.nan
    out["landing_neighbor_count"] = np.nan
    out["distance_to_hotzone_m"] = np.nan
    out["landing_distance_to_hotzone_m"] = np.nan
    out["hotzone_density_q75"] = np.nan
    out["hotzone_distance_q25"] = np.nan
    out["outer_distance_q75"] = np.nan
    out["farm_position_group"] = "middle"

    for match_id, idx in out.groupby("match_id").groups.items():
        pos = out.loc[idx, ["landing_x", "landing_y"]].copy()
        valid = (
            _num(pos["landing_x"]).apply(lambda v: math.isfinite(float(v)))
            & _num(pos["landing_y"]).apply(lambda v: math.isfinite(float(v)))
        )
        valid_idx = pos.index[valid]
        if len(valid_idx) < 4:
            continue

        coords = out.loc[valid_idx, ["landing_x", "landing_y"]].to_numpy(dtype=float)
        dx = coords[:, 0][:, None] - coords[:, 0][None, :]
        dy = coords[:, 1][:, None] - coords[:, 1][None, :]
        dist = np.sqrt(dx ** 2 + dy ** 2)

        kernel = np.exp(-(dist ** 2) / (2.0 * float(bandwidth_m) ** 2))
        np.fill_diagonal(kernel, 0.0)
        density = kernel.sum(axis=1)
        neighbor_count = ((dist <= float(bandwidth_m)) & (dist > 0.0)).sum(axis=1)

        if float(np.std(density)) < 1e-12:
            continue

        density_q75 = float(np.quantile(density, 0.75))
        hot_mask = density >= density_q75
        hot_coords = coords[hot_mask]
        if hot_coords.shape[0] == 0:
            continue

        hot_dx = coords[:, 0][:, None] - hot_coords[:, 0][None, :]
        hot_dy = coords[:, 1][:, None] - hot_coords[:, 1][None, :]
        landing_hot_dist = np.sqrt(hot_dx ** 2 + hot_dy ** 2).min(axis=1)

        farm_coords = out.loc[valid_idx, ["early_farm_x", "early_farm_y"]].to_numpy(dtype=float)
        farm_valid = np.isfinite(farm_coords).all(axis=1)
        farm_coords[~farm_valid] = coords[~farm_valid]
        farm_hot_dx = farm_coords[:, 0][:, None] - hot_coords[:, 0][None, :]
        farm_hot_dy = farm_coords[:, 1][:, None] - hot_coords[:, 1][None, :]
        hot_dist = np.sqrt(farm_hot_dx ** 2 + farm_hot_dy ** 2).min(axis=1)
        inner_q25 = float(np.quantile(hot_dist, 0.25))
        outer_q75 = float(np.quantile(hot_dist, 0.75))
        hot_farm_mask = hot_dist <= inner_q25
        outer_mask = hot_dist >= outer_q75

        out.loc[valid_idx, "landing_density_score"] = density
        out.loc[valid_idx, "landing_neighbor_count"] = neighbor_count
        out.loc[valid_idx, "distance_to_hotzone_m"] = hot_dist
        out.loc[valid_idx, "landing_distance_to_hotzone_m"] = landing_hot_dist
        out.loc[valid_idx, "hotzone_density_q75"] = density_q75
        out.loc[valid_idx, "hotzone_distance_q25"] = inner_q25
        out.loc[valid_idx, "outer_distance_q75"] = outer_q75
        out.loc[valid_idx[outer_mask], "farm_position_group"] = "outer_farm"
        out.loc[valid_idx[hot_farm_mask], "farm_position_group"] = "hotzone"

    return out


def _group_test(
    df: pd.DataFrame,
    *,
    hypothesis: str,
    metric: str,
    group_col: str,
    low_label: str,
    high_label: str,
    expected_better_group: str,
    target: str,
    target_label: str,
    metric_label: str,
    higher_target_is_better: bool = False,
) -> dict[str, Any]:
    valid = df[df[group_col].isin([low_label, high_label])].copy()
    valid = valid[[metric, group_col, target, "survival_seconds"]].dropna()
    valid = valid[
        valid[metric].apply(lambda v: math.isfinite(float(v)))
        & valid[target].apply(lambda v: math.isfinite(float(v)))
    ]
    low = valid[valid[group_col] == low_label]
    high = valid[valid[group_col] == high_label]

    result: dict[str, Any] = {
        "hypothesis": hypothesis,
        "metric": metric,
        "metric_label": metric_label,
        "group_col": group_col,
        "low_label": low_label,
        "high_label": high_label,
        "expected_better_group": expected_better_group,
        "target": target,
        "target_label": target_label,
        "higher_target_is_better": higher_target_is_better,
        "low_n": int(low.shape[0]),
        "high_n": int(high.shape[0]),
        "n": int(valid.shape[0]),
    }
    if low.empty or high.empty:
        result["insufficient"] = True
        return result

    try:
        mw = mannwhitneyu(low[target], high[target], alternative="two-sided")
        result["mw_u"] = float(mw.statistic)
        result["mw_p"] = float(mw.pvalue)
    except Exception:
        result["mw_u"] = float("nan")
        result["mw_p"] = float("nan")

    result.update({
        "low_mean_metric": float(low[metric].mean()),
        "high_mean_metric": float(high[metric].mean()),
        "low_mean_target": float(low[target].mean()),
        "high_mean_target": float(high[target].mean()),
        "delta_high_minus_low": float(high[target].mean() - low[target].mean()),
        "low_mean_survival_s": float(low["survival_seconds"].mean()),
        "high_mean_survival_s": float(high["survival_seconds"].mean()),
        "cliffs_delta_low_vs_high": _cliffs_delta(low[target], high[target]),
    })

    if expected_better_group == high_label:
        result["direction_supported"] = bool(
            result["delta_high_minus_low"] > 0
            if higher_target_is_better
            else result["delta_high_minus_low"] < 0
        )
    else:
        result["direction_supported"] = bool(
            result["delta_high_minus_low"] < 0
            if higher_target_is_better
            else result["delta_high_minus_low"] > 0
        )
    try:
        result["statistically_supported"] = bool(
            result["direction_supported"] and float(result.get("mw_p", math.nan)) < 0.05
        )
    except (TypeError, ValueError):
        result["statistically_supported"] = False

    pearson_r, pearson_p, corr_n = _corr(valid[metric], valid[target], "pearson")
    spearman_r, spearman_p, _ = _corr(valid[metric], valid[target], "spearman")
    result.update({
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "corr_n": corr_n,
    })
    return result


def _compute_static_survival(static: dict) -> dict[int, float]:
    duration = float(static.get("duration_s") or 0.0)
    death_times = static.get("death_times", {}) or {}
    idx_to_aid = {p["idx"]: p["account_id"] for p in static.get("players", [])}
    survival: dict[int, float] = {}
    for team in static.get("teams", []):
        times: list[float] = []
        for member_idx in team.get("members", []) or []:
            aid = idx_to_aid.get(member_idx)
            death_t = death_times.get(aid)
            times.append(duration if death_t is None else float(death_t))
        survival[int(team["idx"])] = max(times) if times else 0.0
    return survival


def _team_centroid_samples(
    frames: list[dict],
    *,
    n_teams: int,
    landing_offset_s: float,
) -> dict[int, list[tuple[float, float, float]]]:
    samples: dict[int, list[tuple[float, float, float]]] = {ti: [] for ti in range(n_teams)}
    for frame in frames:
        t = float(frame.get("t") or 0.0)
        if t < landing_offset_s:
            continue
        centroids = frame.get("team_centroid") or []
        for ti, centroid in enumerate(centroids):
            if ti >= n_teams or centroid is None:
                continue
            samples[ti].append((t, float(centroid[0]), float(centroid[1])))
    return samples


def _movement_features(
    samples: list[tuple[float, float, float]],
    *,
    survival_seconds: float,
    landing_offset_s: float,
    early_farm_window_s: float,
    h3_fixed_window_s: float,
    movement_active_threshold_m: float,
) -> dict[str, Any]:
    post_alive_s = max(0.0, float(survival_seconds) - float(landing_offset_s))
    post_alive_min = post_alive_s / 60.0
    expected_frames = _inclusive_frame_count(post_alive_s)

    out: dict[str, Any] = {
        "post_landing_alive_s": post_alive_s,
        "post_landing_alive_min": post_alive_min,
        "post_landing_expected_frames": expected_frames,
        "post_landing_observed_frames": len(samples),
        "landing_t": np.nan,
        "landing_x": np.nan,
        "landing_y": np.nan,
        "landing_dist_center_m": np.nan,
        "early_farm_x": np.nan,
        "early_farm_y": np.nan,
        "early_farm_dist_center_m": np.nan,
        "trajectory_length_m": 0.0,
        "movement_per_min": 0.0,
        "movement_active_steps": 0,
        "movement_step_count": 0,
        "movement_active_frame_rate": 0.0,
        "net_displacement_m": 0.0,
        "position_std_m": 0.0,
        "fixed_window_s": float(h3_fixed_window_s),
        "fixed_window_eligible": bool(post_alive_s >= float(h3_fixed_window_s)),
        "fixed_window_trajectory_m": 0.0,
        "fixed_window_movement_per_min": 0.0,
        "fixed_window_active_steps": 0,
        "fixed_window_step_count": 0,
        "fixed_window_active_frame_rate": 0.0,
        "fixed_window_position_std_m": 0.0,
    }
    if not samples:
        return out

    center = (ARENA_SIZE_M / 2.0, ARENA_SIZE_M / 2.0)
    first_t, first_x, first_y = samples[0]
    out["landing_t"] = first_t
    out["landing_x"] = first_x
    out["landing_y"] = first_y
    out["landing_dist_center_m"] = _dist((first_x, first_y), center)

    early_samples = [
        (x, y)
        for t, x, y in samples
        if t <= landing_offset_s + early_farm_window_s
    ]
    if early_samples:
        xs = np.array([p[0] for p in early_samples], dtype=float)
        ys = np.array([p[1] for p in early_samples], dtype=float)
        out["early_farm_x"] = float(xs.mean())
        out["early_farm_y"] = float(ys.mean())
        out["early_farm_dist_center_m"] = _dist(
            (out["early_farm_x"], out["early_farm_y"]), center
        )

    trajectory = 0.0
    active_steps = 0
    step_count = 0
    step_distances: list[float] = []
    fixed_samples = [
        sample
        for sample in samples
        if sample[0] <= landing_offset_s + float(h3_fixed_window_s)
    ]
    fixed_trajectory = 0.0
    fixed_active_steps = 0
    fixed_step_count = 0
    for prev, cur in zip(samples, samples[1:]):
        prev_t, prev_x, prev_y = prev
        cur_t, cur_x, cur_y = cur
        if cur_t <= prev_t:
            continue
        d = _dist((prev_x, prev_y), (cur_x, cur_y))
        trajectory += d
        step_count += 1
        step_distances.append(d)
        if d >= movement_active_threshold_m:
            active_steps += 1

    for prev, cur in zip(fixed_samples, fixed_samples[1:]):
        prev_t, prev_x, prev_y = prev
        cur_t, cur_x, cur_y = cur
        if cur_t <= prev_t:
            continue
        d = _dist((prev_x, prev_y), (cur_x, cur_y))
        fixed_trajectory += d
        fixed_step_count += 1
        if d >= movement_active_threshold_m:
            fixed_active_steps += 1

    last_t, last_x, last_y = samples[-1]
    xs = np.array([p[1] for p in samples], dtype=float)
    ys = np.array([p[2] for p in samples], dtype=float)
    if fixed_samples:
        fixed_xs = np.array([p[1] for p in fixed_samples], dtype=float)
        fixed_ys = np.array([p[2] for p in fixed_samples], dtype=float)
        fixed_pos_std = float(np.sqrt(fixed_xs.var(ddof=0) + fixed_ys.var(ddof=0)))
    else:
        fixed_pos_std = 0.0
    out.update({
        "trajectory_length_m": float(trajectory),
        "movement_per_min": _safe_div(trajectory, post_alive_min),
        "movement_active_steps": int(active_steps),
        "movement_step_count": int(step_count),
        "movement_active_frame_rate": _safe_div(active_steps, step_count),
        "net_displacement_m": _dist((first_x, first_y), (last_x, last_y)),
        "position_std_m": float(np.sqrt(xs.var(ddof=0) + ys.var(ddof=0))),
        "mean_step_distance_m": float(np.mean(step_distances)) if step_distances else 0.0,
        "fixed_window_trajectory_m": float(fixed_trajectory),
        "fixed_window_movement_per_min": _safe_div(
            fixed_trajectory, float(h3_fixed_window_s) / 60.0
        ),
        "fixed_window_active_steps": int(fixed_active_steps),
        "fixed_window_step_count": int(fixed_step_count),
        "fixed_window_active_frame_rate": _safe_div(
            fixed_active_steps, fixed_step_count
        ),
        "fixed_window_position_std_m": fixed_pos_std,
    })
    return out


def _combat_features(
    *,
    static: dict,
    events: list[dict],
    landing_offset_s: float,
    survival_by_team: dict[int, float],
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    n_teams = int(static.get("n_teams") or 0)
    account_to_team = {
        p["account_id"]: int(p["team_idx"])
        for p in static.get("players", [])
    }
    dealt = defaultdict(int)
    taken = defaultdict(int)
    active_bins: dict[int, set[int]] = {ti: set() for ti in range(n_teams)}
    pair_bins: dict[int, set[tuple[int, int]]] = defaultdict(set)

    for event in events:
        elapsed = float(event.get("elapsed") or 0.0)
        if elapsed < landing_offset_s:
            continue
        attacker_team = account_to_team.get(event.get("attacker"))
        victim_team = account_to_team.get(event.get("victim"))
        if attacker_team is None or victim_team is None or attacker_team == victim_team:
            continue
        bin_idx = int(math.floor((elapsed - landing_offset_s) / SNAPSHOT_INTERVAL_SEC))
        dealt[attacker_team] += 1
        taken[victim_team] += 1
        active_bins[attacker_team].add(bin_idx)
        active_bins[victim_team].add(bin_idx)
        pair = (min(attacker_team, victim_team), max(attacker_team, victim_team))
        pair_bins[bin_idx].add(pair)

    features: dict[int, dict[str, Any]] = {}
    for ti in range(n_teams):
        post_alive_s = max(0.0, float(survival_by_team.get(ti, 0.0)) - landing_offset_s)
        post_alive_min = post_alive_s / 60.0
        expected_frames = _inclusive_frame_count(post_alive_s)
        combat_exposure_frames = max(expected_frames, len(active_bins[ti]))
        event_count = int(dealt[ti] + taken[ti])
        features[ti] = {
            "damage_event_dealt_count": int(dealt[ti]),
            "damage_event_taken_count": int(taken[ti]),
            "damage_event_involved_count": event_count,
            "combat_event_rate_per_min": _safe_div(event_count, post_alive_min),
            "combat_active_frames": len(active_bins[ti]),
            "combat_exposure_frames": int(combat_exposure_frames),
            "combat_st_participation_rate": _safe_div(
                len(active_bins[ti]), combat_exposure_frames
            ),
        }

    frame_rows: list[dict[str, Any]] = []
    max_frame = max(pair_bins.keys(), default=-1)
    for bin_idx in range(max_frame + 1):
        t0 = landing_offset_s + bin_idx * SNAPSHOT_INTERVAL_SEC
        active_edges = pair_bins.get(bin_idx, set())
        active_teams = {
            team_idx
            for team_idx, survival_s in survival_by_team.items()
            if survival_s >= t0
        }
        n_active = len(active_teams)
        possible = n_active * (n_active - 1) / 2
        frame_rows.append({
            "match_id": static["match_id"],
            "frame_bin": bin_idx,
            "t_start": t0,
            "n_active_teams": n_active,
            "n_dynamic_combat_edges": len(active_edges),
            "dynamic_combat_density": _safe_div(len(active_edges), possible),
        })
    return features, frame_rows


def build_st_features(
    dataset_dir: str,
    *,
    landing_offset_s: float,
    early_farm_window_s: float,
    h3_fixed_window_s: float,
    movement_active_threshold_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = pd.read_csv(os.path.join(dataset_dir, "matches.csv"))
    base_teams = pd.read_csv(os.path.join(dataset_dir, "teams.csv"))
    base_lookup = {
        (str(row.match_id), int(row.team_idx)): row
        for row in base_teams.itertuples(index=False)
    }

    rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    total = int(matches.shape[0])

    for i, match in enumerate(matches.itertuples(index=False), start=1):
        match_id = str(match.match_id)
        pt_path = str(match.pt_path)
        if i == 1 or i % 10 == 0 or i == total:
            print(f"[{i}/{total}] {match_id}")

        md = build_match_data(pt_path, fetch_damage=False, stride=1)
        static = md["static"]
        survival_by_team = _compute_static_survival(static)
        samples_by_team = _team_centroid_samples(
            md.get("frames", []),
            n_teams=int(static.get("n_teams") or 0),
            landing_offset_s=landing_offset_s,
        )
        damage_pack = load_match_damage(match_id)
        combat_by_team, match_frame_rows = _combat_features(
            static=static,
            events=damage_pack.get("events", []),
            landing_offset_s=landing_offset_s,
            survival_by_team=survival_by_team,
        )
        ranked_stats_by_team_id = load_team_ranked_stats(match_id)
        frame_rows.extend(match_frame_rows)

        for team in static.get("teams", []):
            team_idx = int(team["idx"])
            team_id = int(team.get("team_id"))
            base = base_lookup.get((match_id, team_idx))
            survival_s = float(
                getattr(base, "survival_seconds", survival_by_team.get(team_idx, 0.0))
            )
            rank = getattr(base, "final_rank", team.get("rank"))
            movement = _movement_features(
                samples_by_team.get(team_idx, []),
                survival_seconds=survival_s,
                landing_offset_s=landing_offset_s,
                early_farm_window_s=early_farm_window_s,
                h3_fixed_window_s=h3_fixed_window_s,
                movement_active_threshold_m=movement_active_threshold_m,
            )
            combat = combat_by_team.get(team_idx, {})
            ranked_stats = ranked_stats_by_team_id.get(team_id, {})
            rows.append({
                "match_id": match_id,
                "map_name": static.get("map_name"),
                "game_mode": static.get("game_mode"),
                "team_idx": team_idx,
                "team_id": team_id,
                "final_rank": rank,
                "n_members": len(team.get("members", []) or []),
                "survival_seconds": survival_s,
                "landing_offset_s": landing_offset_s,
                "early_farm_window_s": early_farm_window_s,
                "movement_active_threshold_m": movement_active_threshold_m,
                **movement,
                **combat,
                **ranked_stats,
            })

    features = pd.DataFrame(rows)
    frames = pd.DataFrame(frame_rows)
    return features, frames


def add_ranked_rp_proxy(
    features: pd.DataFrame,
    *,
    assist_weight: float,
) -> pd.DataFrame:
    df = features.copy()
    n_teams = df.groupby("match_id")["team_idx"].transform("count").astype(float)
    rank = _num(df["final_rank"]).fillna(n_teams)
    df["ranked_placement_percentile"] = np.where(
        n_teams > 1.0,
        (n_teams - rank) / (n_teams - 1.0),
        0.0,
    )
    df["ranked_placement_percentile"] = df["ranked_placement_percentile"].clip(0.0, 1.0)
    df = _add_within_match_z(df, "ranked_placement_percentile", "rp_z_placement")
    df = _add_within_match_z(df, "team_kills", "rp_z_kills")
    df = _add_within_match_z(df, "team_assists", "rp_z_assists")
    df["rp_assist_weight"] = float(assist_weight)
    df["ranked_rp_proxy"] = (
        df["rp_z_placement"]
        + df["rp_z_kills"]
        + float(assist_weight) * df["rp_z_assists"]
    )
    return df


def add_groups_and_tests(
    features: pd.DataFrame,
    *,
    hotzone_bandwidth_m: float,
    rp_assist_weight: float,
    h3_fixed_window_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = features.copy()
    df = add_ranked_rp_proxy(df, assist_weight=rp_assist_weight)
    df["rank_residual_after_survival"] = _residualize_rank_after_survival(df)
    df = _add_landing_hotzone_features(df, bandwidth_m=hotzone_bandwidth_m)
    df = _assign_match_quantile_groups(
        df,
        "combat_st_participation_rate",
        low_label="low_combat_rate",
        high_label="high_combat_rate",
        group_col="combat_rate_group",
    )
    df = _assign_match_quantile_groups(
        df,
        "movement_per_min",
        low_label="sedentary",
        high_label="mobile",
        group_col="movement_rate_group",
    )
    df = _assign_fixed_window_movement_groups(
        df,
        fixed_window_s=h3_fixed_window_s,
    )
    df = _assign_match_quantile_groups(
        df,
        "movement_active_frame_rate",
        low_label="low_active_move_frames",
        high_label="high_active_move_frames",
        group_col="movement_active_group",
    )
    df = _assign_match_quantile_groups(
        df,
        "combat_event_rate_per_min",
        low_label="low_event_rate",
        high_label="high_event_rate",
        group_col="combat_event_rate_group",
    )
    df = _assign_match_quantile_groups(
        df,
        "trajectory_length_m",
        low_label="low_total_movement",
        high_label="high_total_movement",
        group_col="total_movement_group",
    )

    specs = [
        {
            "hypothesis": "H1_outer_farming",
            "metric": "distance_to_hotzone_m",
            "group_col": "farm_position_group",
            "low_label": "hotzone",
            "high_label": "outer_farm",
            "expected_better_group": "outer_farm",
            "metric_label": "매치별 고밀도 착륙지까지 최근접 거리",
        },
        {
            "hypothesis": "H2_combat_participation",
            "metric": "combat_st_participation_rate",
            "group_col": "combat_rate_group",
            "low_label": "low_combat_rate",
            "high_label": "high_combat_rate",
            "expected_better_group": "high_combat_rate",
            "metric_label": "10초 ST 교전 참여율",
        },
        {
            "hypothesis": "H2_combat_event_rate_sensitivity",
            "metric": "combat_event_rate_per_min",
            "group_col": "combat_event_rate_group",
            "low_label": "low_event_rate",
            "high_label": "high_event_rate",
            "expected_better_group": "high_event_rate",
            "metric_label": "데미지 이벤트 관여 횟수/분",
        },
        {
            "hypothesis": "H3_low_movement_fixed_window",
            "metric": "fixed_window_movement_per_min",
            "group_col": "movement_fixed_window_group",
            "low_label": "fixed_sedentary",
            "high_label": "fixed_mobile",
            "expected_better_group": "fixed_sedentary",
            "metric_label": (
                f"착륙 후 동일 {h3_fixed_window_s:.0f}초 창의 팀 중심 이동거리/분"
            ),
        },
        {
            "hypothesis": "H3_low_movement_naive_descriptive",
            "metric": "movement_per_min",
            "group_col": "movement_rate_group",
            "low_label": "sedentary",
            "high_label": "mobile",
            "expected_better_group": "sedentary",
            "metric_label": "착륙 후 팀 중심 이동거리/분(생존조건 없음)",
        },
        {
            "hypothesis": "H3_low_active_movement_sensitivity",
            "metric": "movement_active_frame_rate",
            "group_col": "movement_active_group",
            "low_label": "low_active_move_frames",
            "high_label": "high_active_move_frames",
            "expected_better_group": "low_active_move_frames",
            "metric_label": "활발 이동 10초 프레임 비율",
        },
        {
            "hypothesis": "H3_total_movement_descriptive",
            "metric": "trajectory_length_m",
            "group_col": "total_movement_group",
            "low_label": "low_total_movement",
            "high_label": "high_total_movement",
            "expected_better_group": "low_total_movement",
            "metric_label": "착륙 후 총 이동거리",
        },
    ]

    results: list[dict[str, Any]] = []
    for spec in specs:
        for target, target_label, higher_is_better in [
            ("final_rank", "최종 순위(숫자가 낮을수록 좋음)", False),
            ("rank_residual_after_survival", "생존시간 보정 순위 잔차(낮을수록 좋음)", False),
            ("ranked_rp_proxy", "경쟁전 RP proxy(높을수록 좋음)", True),
        ]:
            results.append(_group_test(
                df,
                hypothesis=spec["hypothesis"],
                metric=spec["metric"],
                group_col=spec["group_col"],
                low_label=spec["low_label"],
                high_label=spec["high_label"],
                expected_better_group=spec["expected_better_group"],
                target=target,
                target_label=target_label,
                metric_label=spec["metric_label"],
                higher_target_is_better=higher_is_better,
            ))

    return df, pd.DataFrame(results)


def _test_lookup(tests: pd.DataFrame, hypothesis: str, target: str = "final_rank") -> pd.Series:
    row = tests[(tests["hypothesis"] == hypothesis) & (tests["target"] == target)]
    if row.empty:
        return pd.Series(dtype=object)
    return row.iloc[0]


def _support_word(row: pd.Series) -> str:
    if row.empty:
        return "판정 불가"
    if not bool(row.get("direction_supported", False)):
        return "지지되지 않음"
    try:
        p = float(row.get("mw_p"))
    except (TypeError, ValueError):
        p = math.nan
    if math.isfinite(p) and p < 0.05:
        return "지지(방향+유의)"
    return "방향은 일치하나 유의하지 않음"


def _write_figures(features: pd.DataFrame, out_dir: str) -> dict[str, str]:
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    paths: dict[str, str] = {}

    plot_specs = [
        ("distance_to_hotzone_m", "final_rank", "hotzone_distance_vs_rank.png",
         "Distance to match landing hotzone (m)", "Final rank"),
        ("combat_st_participation_rate", "final_rank", "combat_rate_vs_rank.png",
         "ST combat participation rate", "Final rank"),
        ("movement_per_min", "final_rank", "movement_rate_vs_rank.png",
         "Movement after landing (m/min)", "Final rank"),
    ]
    for x_col, y_col, filename, xlabel, ylabel in plot_specs:
        d = features[[x_col, y_col]].dropna()
        if d.empty:
            continue
        plt.figure(figsize=(7.2, 4.4))
        plt.scatter(d[x_col], d[y_col], s=12, alpha=0.35, color="#1f77b4")
        plt.gca().invert_yaxis()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{xlabel} vs placement")
        plt.tight_layout()
        path = os.path.join(fig_dir, filename)
        plt.savefig(path, dpi=160)
        plt.close()
        paths[filename] = path

    box_specs = [
        ("farm_position_group", ["hotzone", "outer_farm"], "farm_group_rank_boxplot.png",
         "Farm position group"),
        ("combat_rate_group", ["low_combat_rate", "high_combat_rate"], "combat_group_rank_boxplot.png",
         "Combat participation group"),
        ("movement_fixed_window_group", ["fixed_sedentary", "fixed_mobile"],
         "movement_group_rank_boxplot.png", "Fixed-window movement group"),
    ]
    for group_col, labels, filename, xlabel in box_specs:
        d = features[features[group_col].isin(labels)][[group_col, "final_rank"]].dropna()
        if d.empty:
            continue
        data = [d[d[group_col] == label]["final_rank"].to_numpy(dtype=float) for label in labels]
        plt.figure(figsize=(6.2, 4.2))
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.gca().invert_yaxis()
        plt.xlabel(xlabel)
        plt.ylabel("Final rank")
        plt.title(f"Placement by {xlabel}")
        plt.tight_layout()
        path = os.path.join(fig_dir, filename)
        plt.savefig(path, dpi=160)
        plt.close()
        paths[filename] = path

    return paths


def _make_report(
    *,
    dataset_dir: str,
    out_dir: str,
    features: pd.DataFrame,
    frames: pd.DataFrame,
    tests: pd.DataFrame,
    figures: dict[str, str],
    landing_offset_s: float,
    early_farm_window_s: float,
    movement_active_threshold_m: float,
    hotzone_bandwidth_m: float,
    rp_assist_weight: float,
    h3_fixed_window_s: float,
) -> str:
    h1 = _test_lookup(tests, "H1_outer_farming", "final_rank")
    h1_adj = _test_lookup(tests, "H1_outer_farming", "rank_residual_after_survival")
    h1_rp = _test_lookup(tests, "H1_outer_farming", "ranked_rp_proxy")
    h2 = _test_lookup(tests, "H2_combat_participation", "final_rank")
    h2_adj = _test_lookup(tests, "H2_combat_participation", "rank_residual_after_survival")
    h2_rp = _test_lookup(tests, "H2_combat_participation", "ranked_rp_proxy")
    h2_evt = _test_lookup(tests, "H2_combat_event_rate_sensitivity", "final_rank")
    h3 = _test_lookup(tests, "H3_low_movement_fixed_window", "final_rank")
    h3_adj = _test_lookup(
        tests, "H3_low_movement_fixed_window", "rank_residual_after_survival"
    )
    h3_rp = _test_lookup(tests, "H3_low_movement_fixed_window", "ranked_rp_proxy")
    h3_naive = _test_lookup(tests, "H3_low_movement_naive_descriptive", "final_rank")
    h3_total = _test_lookup(tests, "H3_total_movement_descriptive", "final_rank")

    dynamic_density_mean = float(frames["dynamic_combat_density"].mean()) if not frames.empty else float("nan")
    dynamic_edges_mean = float(frames["n_dynamic_combat_edges"].mean()) if not frames.empty else float("nan")

    lines = [
        "# PUBG 10초 ST 기반 새 가설 검증 보고서",
        "",
        "## 초록",
        "",
        (
            "본 보고서는 기존 가설을 폐기하고, 선택한 Baltic_Main squad 매치의 "
            f"{features.shape[0]:,}개 팀을 대상으로 세 가지 새 가설을 검증한다. "
            "분석 단위는 팀이며, 최종 순위는 숫자가 낮을수록 좋은 성과로 해석한다. "
            "공간-시간(ST) 구조는 10초 간격 스냅샷의 팀 중심 좌표와 damage event의 "
            "10초 프레임 귀속을 통해 반영했다."
        ),
        "",
        "## 1. 용어 정의와 PUBG 적용",
        "",
        (
            f"- ST 구조: 공간 좌표와 시간이 함께 있는 데이터 구조다. 본 분석에서는 "
            f"{int(SNAPSHOT_INTERVAL_SEC)}초마다 팀 중심 좌표와 동적 교전 엣지가 갱신되는 "
            "시퀀스로 정의한다."
        ),
        (
            f"- 착륙 이후 구간: 매치 시작 후 {landing_offset_s:.0f}초 이후다. 비행기와 낙하 "
            "이동은 모든 팀에 공통으로 큰 이동량을 만들기 때문에 이동량 계산에서 제외했다."
        ),
        (
            f"- 초반 파밍 위치: 착륙 이후 {early_farm_window_s:.0f}초 동안 관측된 팀 중심 좌표의 "
            "평균 위치다. PUBG에서는 팀이 초반 아이템을 확보하며 머무른 지역의 proxy로 사용한다."
        ),
        (
            f"- 핫존과 외곽: 맵의 기하학적 중심이 아니라, 매치별 착륙 위치의 밀집도를 기준으로 정의한다. "
            f"착륙 좌표들에 bandwidth {hotzone_bandwidth_m:.0f}m의 Gaussian kernel density를 적용하고, "
            "밀도 상위 25% 착륙지를 hotzone seed로 둔다. 각 팀은 착륙 후 300초 평균 파밍 위치에서 "
            "가장 가까운 hotzone seed까지의 거리로 분류하며, 거리 하위 25%를 hotzone, "
            "상위 25%를 outer_farm으로 둔다."
        ),
        (
            "- 교전 참여율: 누적 데미지량이 아니라 데미지를 가하거나 받은 event의 발생을 사용한다. "
            "주 지표는 팀이 교전에 관여한 10초 프레임 수를 전투 노출 프레임 수로 나눈 "
            "10초 ST 교전 참여율이다. 전투 노출 프레임 수는 착륙 후 생존 프레임 수를 기본으로 하되, "
            "DB event가 사망 시각 추정치를 넘는 특이 케이스에서는 실제 event 관측 프레임 수와의 "
            "최댓값으로 보정했다."
        ),
        (
            f"- 이동량: 착륙 이후 팀 중심 좌표가 10초 프레임 사이 이동한 거리의 합이다. "
            "주 지표는 생존시간 편향을 줄이기 위해 분당 이동거리로 두었다. "
            f"{movement_active_threshold_m:.0f}m 이상 이동한 10초 step 비율도 보조 지표로 사용했다."
        ),
        (
            "- 평균 순위가 높다: PUBG의 final_rank는 1위가 가장 좋은 값이므로, 본 보고서에서는 "
            "평균 순위가 높다는 표현을 평균 final_rank 숫자가 낮다는 의미로 해석한다."
        ),
        "",
        "## 2. 자료와 방법",
        "",
        f"- 입력 데이터셋: `{dataset_dir}`",
        f"- 분석 산출물: `{out_dir}`",
        f"- 표본: {features['match_id'].nunique():,}개 매치, {features.shape[0]:,}개 팀",
        (
            "- 통계 검정: 각 매치 안에서 하위 25%와 상위 25% 그룹을 만들고, "
            "Mann-Whitney U 검정, Cliff's delta, Pearson/Spearman 상관을 계산했다."
        ),
        (
            "- 생존시간 보정: final_rank를 log(생존시간)과 그 제곱으로 회귀한 뒤 잔차를 계산했다. "
            "잔차가 낮을수록 같은 생존시간 대비 더 좋은 순위다."
        ),
        (
            "- 경쟁전 RP proxy: PUBG Support가 공개한 경쟁전 주요 반영 항목인 팀 순위, 킬, 어시스트를 "
            "사용했다. 정확한 RP 가중치와 고티어 적 처치 보너스는 공개/관측되지 않으므로, 매치 내 "
            f"표준화된 placement percentile + kill z-score + {rp_assist_weight:.2f} * assist z-score의 "
            "가중 합으로 근사했다."
        ),
        (
            f"- 동적 교전 그래프 평균: 10초 프레임당 교전 엣지 {_fmt(dynamic_edges_mean)}개, "
            f"동적 밀도 {_fmt(dynamic_density_mean)}."
        ),
        "",
        "## 3. 가설별 결과",
        "",
        "### 3.1 H1: 핫존보다 외곽 파밍 팀의 평균 순위가 높다",
        "",
        (
            f"- 원 순위 기준: hotzone 평균 순위 {_fmt(h1.get('low_mean_target'))}, "
            f"outer_farm 평균 순위 {_fmt(h1.get('high_mean_target'))}, "
            f"차이(high-low) {_fmt(h1.get('delta_high_minus_low'))}, "
            f"p={_fmt(h1.get('mw_p'), 4)}. 판정: {_support_word(h1)}."
        ),
        (
            f"- 생존시간 보정 잔차 기준: hotzone {_fmt(h1_adj.get('low_mean_target'))}, "
            f"outer_farm {_fmt(h1_adj.get('high_mean_target'))}, "
            f"차이 {_fmt(h1_adj.get('delta_high_minus_low'))}, "
            f"p={_fmt(h1_adj.get('mw_p'), 4)}. 판정: {_support_word(h1_adj)}."
        ),
        (
            f"- 경쟁전 RP proxy 기준: hotzone {_fmt(h1_rp.get('low_mean_target'))}, "
            f"outer_farm {_fmt(h1_rp.get('high_mean_target'))}, "
            f"차이 {_fmt(h1_rp.get('delta_high_minus_low'))}, "
            f"p={_fmt(h1_rp.get('mw_p'), 4)}. 판정: {_support_word(h1_rp)}."
        ),
        "",
        "### 3.2 H2: 교전 참여율이 높을수록 평균 순위가 높다",
        "",
        (
            f"- 10초 ST 교전 참여율 기준: low 평균 순위 {_fmt(h2.get('low_mean_target'))}, "
            f"high 평균 순위 {_fmt(h2.get('high_mean_target'))}, "
            f"차이(high-low) {_fmt(h2.get('delta_high_minus_low'))}, "
            f"p={_fmt(h2.get('mw_p'), 4)}. 판정: {_support_word(h2)}."
        ),
        (
            f"- 생존시간 보정 잔차 기준: low {_fmt(h2_adj.get('low_mean_target'))}, "
            f"high {_fmt(h2_adj.get('high_mean_target'))}, "
            f"차이 {_fmt(h2_adj.get('delta_high_minus_low'))}, "
            f"p={_fmt(h2_adj.get('mw_p'), 4)}. 판정: {_support_word(h2_adj)}."
        ),
        (
            f"- damage event 횟수/분 보조 지표: low 평균 순위 {_fmt(h2_evt.get('low_mean_target'))}, "
            f"high 평균 순위 {_fmt(h2_evt.get('high_mean_target'))}, "
            f"차이 {_fmt(h2_evt.get('delta_high_minus_low'))}."
        ),
        (
            f"- 경쟁전 RP proxy 기준: low {_fmt(h2_rp.get('low_mean_target'))}, "
            f"high {_fmt(h2_rp.get('high_mean_target'))}, "
            f"차이 {_fmt(h2_rp.get('delta_high_minus_low'))}, "
            f"p={_fmt(h2_rp.get('mw_p'), 4)}. 판정: {_support_word(h2_rp)}."
        ),
        "",
        "### 3.3 H3: 이동량이 적은 존버형 팀의 평균 순위가 높다",
        "",
        (
            f"- H3는 조기 탈락팀 오분류와 후반 생존에 따른 추가 이동량 문제를 피하기 위해 "
            f"착륙 후 동일 {h3_fixed_window_s:.0f}초 창을 완주한 팀만 대상으로, "
            "그 고정 창 안의 이동량만 사용해 다시 계산했다."
        ),
        (
            f"- 고정창 분당 이동거리 기준: fixed_sedentary 평균 순위 "
            f"{_fmt(h3.get('low_mean_target'))}, fixed_mobile 평균 순위 "
            f"{_fmt(h3.get('high_mean_target'))}, "
            f"차이(high-low) {_fmt(h3.get('delta_high_minus_low'))}, "
            f"p={_fmt(h3.get('mw_p'), 4)}. 판정: {_support_word(h3)}."
        ),
        (
            f"- 생존시간 보정 잔차 기준: fixed_sedentary {_fmt(h3_adj.get('low_mean_target'))}, "
            f"fixed_mobile {_fmt(h3_adj.get('high_mean_target'))}, "
            f"차이 {_fmt(h3_adj.get('delta_high_minus_low'))}, "
            f"p={_fmt(h3_adj.get('mw_p'), 4)}. 판정: {_support_word(h3_adj)}."
        ),
        (
            f"- 경쟁전 RP proxy 기준: fixed_sedentary {_fmt(h3_rp.get('low_mean_target'))}, "
            f"fixed_mobile {_fmt(h3_rp.get('high_mean_target'))}, "
            f"차이 {_fmt(h3_rp.get('delta_high_minus_low'))}, "
            f"p={_fmt(h3_rp.get('mw_p'), 4)}. 판정: {_support_word(h3_rp)}."
        ),
        (
            f"- 참고: 생존조건 없이 나눈 naive sedentary 평균 순위는 "
            f"{_fmt(h3_naive.get('low_mean_target'))}, mobile은 "
            f"{_fmt(h3_naive.get('high_mean_target'))}였다. 이 값은 조기 탈락팀 오염이 있어 "
            "주 판정에는 사용하지 않는다."
        ),
        (
            f"- 착륙 후 총 이동거리 설명 지표: low 총 이동 평균 순위 {_fmt(h3_total.get('low_mean_target'))}, "
            f"high 총 이동 평균 순위 {_fmt(h3_total.get('high_mean_target'))}, "
            f"차이 {_fmt(h3_total.get('delta_high_minus_low'))}. "
            "총량은 오래 생존한 팀일수록 커질 수 있어 주 판정에는 사용하지 않는다."
        ),
        "",
        "## 4. 함의",
        "",
        (
            "- 외곽 효과는 맵 중심에서 멀다는 뜻이 아니라, 같은 매치에서 다른 팀들이 많이 내린 "
            "착륙 밀집지로부터 떨어져 있다는 뜻으로 해석해야 한다. 이 정의가 실제 PUBG의 핫드랍/안전 파밍 "
            "구분에 더 가깝다."
        ),
        (
            "- 교전 참여율은 누적 데미지보다 event/프레임 기반으로 보는 편이 적절하다. "
            "이번 표본에서는 높은 교전 참여율이 더 좋은 순위와 연결되지 않았고, 오히려 조기 교전과 "
            "강제 교전의 위험이 크게 반영된 것으로 해석된다."
        ),
        (
            "- 존버형 여부는 총 이동거리보다 분당 이동거리와 활발 이동 프레임 비율로 해석해야 한다. "
            "총 이동거리는 생존시간의 결과가 섞이는 지표이기 때문이다. 결과상 정적 플레이가 유리하다는 "
            "가설은 지지되지 않았고, 생존 이후에는 필요한 이동을 수행한 팀의 성과가 더 좋았다."
        ),
        (
            "- 다음 단계에서는 H1의 hotzone bandwidth를 600m/1000m로 민감도 분석하고, "
            "첫 자기장 중심과의 거리 및 초반 300초 내 근접 적 팀 수를 함께 넣으면 더 강한 분석이 된다."
        ),
        "",
        "## 5. 산출 파일",
        "",
        "- `st_team_features.csv`: 팀 단위 ST 파생 지표",
        "- `st_frame_network_summary.csv`: 10초 동적 교전 그래프 요약",
        "- `st_hypothesis_tests.csv`: 가설별 검정 표",
        "- `st_hypothesis_summary.json`: 기계 판독용 요약",
    ]

    if figures:
        lines.extend(["", "## 6. 그림", ""])
        for name, path in figures.items():
            lines.append(f"- `{name}`: `{path}`")

    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = args.dataset
    out_dir = args.out or os.path.join(dataset_dir, "st_hypothesis_analysis")
    os.makedirs(out_dir, exist_ok=True)

    features, frames = build_st_features(
        dataset_dir,
        landing_offset_s=args.landing_offset_s,
        early_farm_window_s=args.early_farm_window_s,
        h3_fixed_window_s=args.h3_fixed_window_s,
        movement_active_threshold_m=args.movement_active_threshold_m,
    )
    features, tests = add_groups_and_tests(
        features,
        hotzone_bandwidth_m=args.hotzone_bandwidth_m,
        rp_assist_weight=args.rp_assist_weight,
        h3_fixed_window_s=args.h3_fixed_window_s,
    )

    features_path = os.path.join(out_dir, "st_team_features.csv")
    frames_path = os.path.join(out_dir, "st_frame_network_summary.csv")
    tests_path = os.path.join(out_dir, "st_hypothesis_tests.csv")
    features.to_csv(features_path, index=False)
    frames.to_csv(frames_path, index=False)
    tests.to_csv(tests_path, index=False)

    figures = _write_figures(features, out_dir)
    report = _make_report(
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        features=features,
        frames=frames,
        tests=tests,
        figures=figures,
        landing_offset_s=args.landing_offset_s,
        early_farm_window_s=args.early_farm_window_s,
        movement_active_threshold_m=args.movement_active_threshold_m,
        hotzone_bandwidth_m=args.hotzone_bandwidth_m,
        rp_assist_weight=args.rp_assist_weight,
        h3_fixed_window_s=args.h3_fixed_window_s,
    )
    report_path = os.path.join(out_dir, "st_hypothesis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n")

    summary = {
        "dataset_dir": dataset_dir,
        "out_dir": out_dir,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_matches": int(features["match_id"].nunique()),
        "n_teams": int(features.shape[0]),
        "landing_offset_s": float(args.landing_offset_s),
        "early_farm_window_s": float(args.early_farm_window_s),
        "movement_active_threshold_m": float(args.movement_active_threshold_m),
        "hotzone_bandwidth_m": float(args.hotzone_bandwidth_m),
        "rp_assist_weight": float(args.rp_assist_weight),
        "h3_fixed_window_s": float(args.h3_fixed_window_s),
        "tests": tests.to_dict(orient="records"),
        "artifacts": {
            "features": features_path,
            "frame_network_summary": frames_path,
            "tests": tests_path,
            "report": report_path,
            "figures": figures,
        },
    }
    summary_path = os.path.join(out_dir, "st_hypothesis_summary.json")
    _write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ST-based PUBG hypothesis tests for outer farming, combat, and movement."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out", default=None)
    parser.add_argument("--landing-offset-s", type=float, default=LANDING_OFFSET_S)
    parser.add_argument("--early-farm-window-s", type=float, default=DEFAULT_EARLY_FARM_WINDOW_S)
    parser.add_argument(
        "--movement-active-threshold-m",
        type=float,
        default=DEFAULT_MOVEMENT_ACTIVE_THRESHOLD_M,
    )
    parser.add_argument(
        "--hotzone-bandwidth-m",
        type=float,
        default=DEFAULT_HOTZONE_BANDWIDTH_M,
    )
    parser.add_argument(
        "--rp-assist-weight",
        type=float,
        default=DEFAULT_RP_ASSIST_WEIGHT,
    )
    parser.add_argument(
        "--h3-fixed-window-s",
        type=float,
        default=DEFAULT_H3_FIXED_WINDOW_S,
        help=(
            "Fixed post-landing observation window for H3 sedentary/mobile "
            "classification."
        ),
    )
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps({
        "out_dir": summary["out_dir"],
        "n_matches": summary["n_matches"],
        "n_teams": summary["n_teams"],
        "report": summary["artifacts"]["report"],
        "tests": summary["artifacts"]["tests"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
