"""
stdb/queries.py
===============
Query functions over the STDB layer that produce learning-ready tensors for
the ST-GCN model.

The main entry point is :func:`fetch_match_tensors` which returns

    X     : float32  (T, N, F)         — node features
    A_t   : float32  (T, N, N)         — dynamic adjacency (row-normalized)
    y_reg : float32  (N,)              — final rank per team (1 = winner)
    y_cls : float32  (N, 2)            — [top5, top10] binary labels
    mask  : bool     (N,)              — team-was-present-in-match mask
    meta  : dict                       — match_id, team_ids, snapshot_ts, …

`F` is 8 (see FEATURE_NAMES below).  `N_MAX` is 25 for the legacy snapshot
model.  `T_WIN` is 20 snapshots (~200 seconds at 10s grid) sampled evenly
from match start to match end.

PostGIS functions used: ST_X, ST_Y, ST_Z, ST_Distance, ST_Boundary, ST_Force2D,
ST_Contains.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from stdb.conn import connect

logger = logging.getLogger("stdb.queries")

T_WIN: int = 20
N_MAX: int = 25
FEATURE_DIM: int = 8
SNAPSHOT_INTERVAL: int = 10
K_NEAREST: int = 5
ENEMY_DENSITY_RADIUS_M: float = 100.0
COMBAT_LOOKBACK_S: float = 30.0
ARENA_SIZE_M: float = 8160.0
ALTITUDE_MAX_M: float = 400.0
DAMAGE_NORM: float = 200.0  # 30s window of 200dmg ≈ saturation

FEATURE_NAMES = [
    "x_norm",                  # 0  centroid X / arena
    "y_norm",                  # 1  centroid Y / arena
    "z_norm",                  # 2  centroid Z / altitude_max
    "dist_to_boundary_norm",   # 3  signed dist to poison ring, normalized
    "enemy_density_100m",      # 4  num enemy teams within 100m (normalized by N)
    "recent_combat_exposure",  # 5  Σ damage_taken in last COMBAT_LOOKBACK_S, normalized
    "is_alive",                # 6  binary
    "elapsed_norm",            # 7  snapshot_t / match.duration_s
]


PHASE_T_MAX: int = 8
# PUBG can have up to 100 solo teams and some telemetry rows contain more than
# 32 squad roster ids.  Keep a cushion so phase tensors never need final-rank
# based truncation, which would leak the label into the input.
PHASE_N_MAX: int = 128
PHASE_EDGE_TYPES = ["proximity", "combat", "copressure"]


PHASE_NODE_FEATURE_SPECS = [
    # global phase context, broadcast to every team node
    ("phase_idx_norm", "phase_idx", 8.0, "linear"),
    ("phase_duration_10min", "phase_duration_s", 600.0, "linear"),
    ("avg_alive_teams_norm", "avg_alive_teams", 100.0, "linear"),
    ("avg_alive_players_norm", "avg_alive_players", 100.0, "linear"),
    ("safe_radius_start_norm", "safe_radius_start_m", ARENA_SIZE_M, "linear"),
    ("safe_radius_end_norm", "safe_radius_end_m", ARENA_SIZE_M, "linear"),
    ("safe_radius_delta_norm", "safe_radius_delta_m", ARENA_SIZE_M, "linear"),
    ("safe_center_shift_norm", "safe_center_shift_m", ARENA_SIZE_M, "linear"),
    ("poison_radius_start_norm", "poison_radius_start_m", ARENA_SIZE_M, "linear"),
    ("poison_radius_end_norm", "poison_radius_end_m", ARENA_SIZE_M, "linear"),
    ("zone_gap_start_norm", "zone_gap_start_m", ARENA_SIZE_M, "linear"),
    ("zone_gap_end_norm", "zone_gap_end_m", ARENA_SIZE_M, "linear"),
    # team state
    ("team_size_norm", "team_size", 4.0, "linear"),
    ("alive_players_start_norm", "alive_players_start", 4.0, "linear"),
    ("alive_players_end_norm", "alive_players_end", 4.0, "linear"),
    ("alive_players_min_norm", "alive_players_min", 4.0, "linear"),
    ("alive_players_avg_norm", "alive_players_avg", 4.0, "linear"),
    ("mean_health_avg_norm", "mean_health_avg", 100.0, "linear"),
    ("min_health_min_norm", "min_health_min", 100.0, "linear"),
    ("health_slope_norm", "health_slope", 10.0, "linear"),
    ("team_spread_avg_norm", "team_spread_avg_m", 1000.0, "linear"),
    ("team_spread_max_norm", "team_spread_max_m", 1000.0, "linear"),
    ("vehicle_ratio", "vehicle_ratio", 1.0, "linear"),
    ("vehicle_speed_mean_norm", "vehicle_speed_mean", 40.0, "linear"),
    ("vehicle_speed_max_norm", "vehicle_speed_max", 40.0, "linear"),
    # trajectory
    ("path_length_norm", "path_length_m", 5000.0, "linear"),
    ("displacement_norm", "displacement_m", 5000.0, "linear"),
    ("log1p_sinuosity", "sinuosity", 1.0, "log1p"),
    ("avg_speed_norm", "avg_speed_mps", 40.0, "linear"),
    ("rotation_timing", "rotation_timing", 1.0, "linear"),
    ("safe_alignment", "safe_alignment", 1.0, "linear"),
    ("zone_entry_time", "zone_entry_time", 1.0, "linear"),
    ("outside_safe_duration_10min", "outside_safe_duration_s", 600.0, "linear"),
    ("boundary_distance_slope_norm", "boundary_distance_slope", 100.0, "linear"),
    # exposure
    ("mean_dist_to_boundary_norm", "mean_dist_to_boundary_m", ARENA_SIZE_M, "linear"),
    ("min_dist_to_boundary_norm", "min_dist_to_boundary_m", ARENA_SIZE_M, "linear"),
    ("outside_safe_ratio", "outside_safe_ratio", 1.0, "linear"),
    ("avg_enemy_density_100m_norm", "avg_enemy_density_100m", 10.0, "linear"),
    ("min_nearest_enemy_norm", "min_nearest_enemy_m", ARENA_SIZE_M, "linear"),
    ("path_near_enemy_ratio", "path_near_enemy_ratio", 1.0, "linear"),
    ("recent_damage_taken_norm", "recent_damage_taken", 1000.0, "linear"),
    ("combat_zone_pass_count_norm", "combat_zone_pass_count", 50.0, "linear"),
    # directed/aggregate combat, kept as node attributes separate from A_combat
    ("damage_dealt_norm", "damage_dealt", 1000.0, "linear"),
    ("damage_taken_norm", "damage_taken", 1000.0, "linear"),
    ("damage_events_dealt_norm", "damage_events_dealt", 50.0, "linear"),
    ("damage_events_taken_norm", "damage_events_taken", 50.0, "linear"),
    ("groggy_dealt_norm", "groggy_dealt", 10.0, "linear"),
    ("groggy_taken_norm", "groggy_taken", 10.0, "linear"),
    ("kills_dealt_norm", "kills_dealt", 10.0, "linear"),
    ("kills_taken_norm", "kills_taken", 10.0, "linear"),
    ("distinct_opponents_damaged_norm", "distinct_opponents_damaged", 10.0, "linear"),
    ("distinct_opponents_damaged_by_norm", "distinct_opponents_damaged_by", 10.0, "linear"),
    ("combat_initiated", "combat_initiated", 1.0, "bool"),
    # loot/readiness
    ("pickup_count_norm", "pickup_count", 100.0, "linear"),
    ("equip_count_norm", "equip_count", 50.0, "linear"),
    ("use_count_norm", "use_count", 50.0, "linear"),
    ("unique_item_count_norm", "unique_item_count", 100.0, "linear"),
    ("weapon_pickup_count_norm", "weapon_pickup_count", 20.0, "linear"),
    ("weapon_equip_count_norm", "weapon_equip_count", 20.0, "linear"),
    ("ammo_pickup_count_norm", "ammo_pickup_count", 50.0, "linear"),
    ("armor_pickup_count_norm", "armor_pickup_count", 20.0, "linear"),
    ("throwable_pickup_count_norm", "throwable_pickup_count", 20.0, "linear"),
    ("heal_use_count_norm", "heal_use_count", 20.0, "linear"),
    ("boost_use_count_norm", "boost_use_count", 20.0, "linear"),
    ("carepackage_pickup_count_norm", "carepackage_pickup_count", 5.0, "linear"),
    ("lootbox_pickup_count_norm", "lootbox_pickup_count", 20.0, "linear"),
]

PHASE_NODE_FEATURE_NAMES = [spec[0] for spec in PHASE_NODE_FEATURE_SPECS]
PHASE_FEATURE_DIM = len(PHASE_NODE_FEATURE_NAMES)


@dataclass
class MatchTensors:
    match_id: str
    X: np.ndarray             # (T, N, F)
    A_t: np.ndarray           # (T, N, N)
    y_reg: np.ndarray         # (N,)
    y_cls: np.ndarray         # (N, 2)
    mask: np.ndarray          # (N,) bool
    team_ids: list[int]       # length N (zero-padded at the tail)
    snapshot_ts: list[int]    # length T


@dataclass
class PhaseMatchTensors:
    match_id: str
    X: np.ndarray              # (T, N, F)
    A: np.ndarray              # (T, E, N, N)
    y_hazard: np.ndarray       # (T-1, N)
    target_mask: np.ndarray    # (T-1, N) bool
    y_rank: np.ndarray         # (N,) normalized final rank
    y_topk: np.ndarray         # (N, 2) [top5, top10]
    team_mask: np.ndarray      # (N,) bool, team exists in match
    node_mask: np.ndarray      # (T, N) bool, alive at phase start
    phase_mask: np.ndarray     # (T,) bool, phase exists in match
    team_ids: list[int]        # length N, zero-padded
    phase_indices: list[int]   # length T, zero-padded
    feature_names: list[str]
    edge_types: list[str]


# ── Internal helpers ─────────────────────────────────────────────────────────


def _pick_snapshot_grid(duration_s: int) -> list[int]:
    """Pick T_WIN snapshot times evenly spaced across the match.

    Snapshots are aligned to the SNAPSHOT_INTERVAL grid (rounded).
    """
    if duration_s <= SNAPSHOT_INTERVAL:
        return [0]
    raw = np.linspace(0, duration_s, T_WIN, dtype=float)
    grid = sorted({int(round(r / SNAPSHOT_INTERVAL)) * SNAPSHOT_INTERVAL for r in raw})
    # In rare cases the unique-rounded set is shorter than T_WIN
    while len(grid) < T_WIN:
        grid.append(grid[-1] + SNAPSHOT_INTERVAL)
    return grid[:T_WIN]


def _team_order(cur, match_id: str) -> list[int]:
    """Return team_ids ordered by final_rank ASC (best first).  Clipped to N_MAX."""
    cur.execute(
        """
        SELECT team_id, final_rank
          FROM stdb.teams
         WHERE match_id = %s
         ORDER BY coalesce(final_rank, 9999), team_id
         LIMIT %s
        """,
        (match_id, N_MAX),
    )
    return [r[0] for r in cur.fetchall()]


def _fetch_team_features(
    cur, match_id: str, team_ids: list[int], snapshot_ts: list[int], duration_s: int
) -> np.ndarray:
    """Build (T, N, F) feature tensor."""
    team_index = {tid: i for i, tid in enumerate(team_ids)}
    n = len(team_ids)
    X = np.zeros((T_WIN, N_MAX, FEATURE_DIM), dtype=np.float32)

    # 1. team centroid + alive
    cur.execute(
        """
        SELECT mo.team_id, mo.snapshot_t,
               ST_X(mo.pos) AS x, ST_Y(mo.pos) AS y, ST_Z(mo.pos) AS z,
               mo.is_alive::int
          FROM stdb.moving_objects mo
         WHERE mo.match_id = %s
           AND mo.agent_kind = 'team'
           AND mo.team_id = ANY(%s)
           AND mo.snapshot_t = ANY(%s)
        """,
        (match_id, team_ids, snapshot_ts),
    )
    centroid: dict[tuple[int, int], tuple[float, float, float, int]] = {}
    for tid, st, x, y, z, alive in cur.fetchall():
        centroid[(tid, st)] = (float(x), float(y), float(z), int(alive))

    # 2. poison boundary distance + safe-zone inside check
    cur.execute(
        """
        SELECT snapshot_t, ST_X(center_pos), ST_Y(center_pos), radius_m
          FROM stdb.dynamic_regions
         WHERE match_id = %s
           AND region_kind = 'poison'
           AND snapshot_t = ANY(%s)
        """,
        (match_id, snapshot_ts),
    )
    poison: dict[int, tuple[float, float, float]] = {
        int(t): (float(cx), float(cy), float(r)) for (t, cx, cy, r) in cur.fetchall()
    }

    # 3. enemy density (count of enemy teams within ENEMY_DENSITY_RADIUS_M)
    cur.execute(
        """
        SELECT snapshot_t, team_a, team_b
          FROM stdb.team_proximity
         WHERE match_id = %s
           AND snapshot_t = ANY(%s)
           AND distance_m < %s
        """,
        (match_id, snapshot_ts, ENEMY_DENSITY_RADIUS_M),
    )
    density: dict[tuple[int, int], int] = {}
    for st, a, b in cur.fetchall():
        density[(a, st)] = density.get((a, st), 0) + 1
        density[(b, st)] = density.get((b, st), 0) + 1

    # 4. recent combat exposure (Σ damage_taken in past COMBAT_LOOKBACK_S)
    cur.execute(
        """
        SELECT victim_team, event_t, coalesce(damage, 0)::real AS dmg
          FROM stdb.spatial_events
         WHERE match_id = %s
           AND event_kind = 'damage'
           AND victim_team IS NOT NULL
        """,
        (match_id,),
    )
    dmg_events: list[tuple[int, float, float]] = list(cur.fetchall())

    for ti, st in enumerate(snapshot_ts):
        for tid in team_ids:
            j = team_index[tid]
            cx_team = cy_team = cz_team = 0.0
            alive = 0
            cent = centroid.get((tid, st))
            if cent is not None:
                cx_team, cy_team, cz_team, alive = cent
            X[ti, j, 0] = float(np.clip(cx_team / ARENA_SIZE_M, 0, 1))
            X[ti, j, 1] = float(np.clip(cy_team / ARENA_SIZE_M, 0, 1))
            X[ti, j, 2] = float(np.clip(cz_team / ALTITUDE_MAX_M, 0, 1))

            poison_t = poison.get(st)
            if poison_t is not None and cent is not None:
                px, py, r = poison_t
                dxy = float(np.hypot(cx_team - px, cy_team - py))
                # negative if outside the safe area (i.e. in poison)
                signed = (r - dxy) / max(r, 1.0)
                X[ti, j, 3] = float(np.clip(signed, -1, 1))

            X[ti, j, 4] = float(density.get((tid, st), 0)) / max(N_MAX - 1, 1)
            X[ti, j, 6] = float(alive)
            X[ti, j, 7] = float(st) / max(duration_s, 1)

        # combat exposure: accumulate damage events in the lookback window
        for victim_team, et, dmg in dmg_events:
            if victim_team not in team_index:
                continue
            if et > st or et < st - COMBAT_LOOKBACK_S:
                continue
            j = team_index[victim_team]
            X[ti, j, 5] += float(dmg) / DAMAGE_NORM

    X[:, :n, 5] = np.clip(X[:, :n, 5], 0, 1)
    return X


def _fetch_dynamic_adjacency(
    cur, match_id: str, team_ids: list[int], snapshot_ts: list[int]
) -> np.ndarray:
    """Build (T, N, N) row-normalized adjacency.  Symmetric, with self-loops."""
    team_index = {tid: i for i, tid in enumerate(team_ids)}
    A = np.zeros((T_WIN, N_MAX, N_MAX), dtype=np.float32)
    n = len(team_ids)

    cur.execute(
        """
        SELECT snapshot_t, team_a, team_b, distance_m, encounter_recently
          FROM stdb.team_proximity
         WHERE match_id = %s AND snapshot_t = ANY(%s)
        """,
        (match_id, snapshot_ts),
    )
    by_snap: dict[int, list[tuple[int, int, float, bool]]] = {}
    for st, a, b, d, enc in cur.fetchall():
        by_snap.setdefault(int(st), []).append((int(a), int(b), float(d), bool(enc)))

    for ti, st in enumerate(snapshot_ts):
        # encounter edges: weight = 1
        for a, b, d, enc in by_snap.get(st, []):
            if a not in team_index or b not in team_index:
                continue
            ia, ib = team_index[a], team_index[b]
            if enc:
                A[ti, ia, ib] = 1.0
                A[ti, ib, ia] = 1.0
        # k-NN by distance per team
        per_team: dict[int, list[tuple[float, int]]] = {tid: [] for tid in team_ids}
        for a, b, d, enc in by_snap.get(st, []):
            if a in per_team and b in team_index:
                per_team[a].append((d, b))
            if b in per_team and a in team_index:
                per_team[b].append((d, a))
        for tid, neigh in per_team.items():
            neigh.sort()
            for d, other in neigh[:K_NEAREST]:
                ia = team_index[tid]
                ib = team_index[other]
                # weight decays with distance: w = exp(-d / 200m), clipped
                w = float(np.exp(-d / 200.0))
                if A[ti, ia, ib] < w:
                    A[ti, ia, ib] = w
                    A[ti, ib, ia] = w
        # self-loops + row-normalize (only over the present nodes)
        for i in range(n):
            A[ti, i, i] = 1.0
        rs = A[ti, :n, :n].sum(axis=1, keepdims=True)
        rs = np.where(rs > 0, rs, 1.0)
        A[ti, :n, :n] = A[ti, :n, :n] / rs

    return A


def _as_float(value, *, scale: float = 1.0, transform: str = "linear") -> float:
    if value is None:
        return 0.0
    if transform == "bool":
        return 1.0 if bool(value) else 0.0
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(x):
        return 0.0
    if transform == "log1p":
        return float(np.log1p(max(x, 0.0)))
    return x / scale if scale else x


def _fetch_phase_indices(cur, match_id: str) -> list[int]:
    cur.execute(
        """
        SELECT phase_idx
          FROM stdb.v_match_phase_bounds
         WHERE match_id = %s
           AND phase_idx BETWEEN 1 AND %s
         ORDER BY phase_idx
        """,
        (match_id, PHASE_T_MAX),
    )
    return [int(r[0]) for r in cur.fetchall()]


def _phase_team_order(cur, match_id: str) -> list[int]:
    """Return phase tensor team order without using outcome labels.

    The old implementation sorted by final_rank and clipped at PHASE_N_MAX.
    That leaks the target, especially for solo/duo where N can exceed 32.  Team
    id is a stable roster identifier available before the match outcome.
    """
    cur.execute(
        """
        SELECT team_id
          FROM stdb.teams
         WHERE match_id = %s
         ORDER BY team_id
         LIMIT %s
        """,
        (match_id, PHASE_N_MAX),
    )
    return [int(r[0]) for r in cur.fetchall()]


def _fetch_phase_node_features(
    cur,
    match_id: str,
    team_ids: list[int],
    phase_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return X plus labels/masks for phase-level modeling."""
    team_index = {tid: i for i, tid in enumerate(team_ids)}
    phase_index = {p: i for i, p in enumerate(phase_indices)}
    X = np.zeros((PHASE_T_MAX, PHASE_N_MAX, PHASE_FEATURE_DIM), dtype=np.float32)
    y_rank = np.zeros(PHASE_N_MAX, dtype=np.float32)
    y_topk = np.zeros((PHASE_N_MAX, 2), dtype=np.float32)
    team_mask = np.zeros(PHASE_N_MAX, dtype=bool)
    node_mask = np.zeros((PHASE_T_MAX, PHASE_N_MAX), dtype=bool)
    phase_mask = np.zeros(PHASE_T_MAX, dtype=bool)

    for p in phase_indices:
        phase_mask[phase_index[p]] = True

    if not team_ids or not phase_indices:
        return X, y_rank, y_topk, team_mask, node_mask, phase_mask

    cur.execute(
        """
        SELECT *
          FROM stdb.v_team_phase_feature_long
         WHERE match_id = %s
           AND team_id = ANY(%s)
           AND phase_idx = ANY(%s)
         ORDER BY phase_idx, team_id
        """,
        (match_id, team_ids, phase_indices),
    )
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    seen_rank: set[int] = set()
    for row in rows:
        tid = int(row["team_id"])
        p = int(row["phase_idx"])
        if tid not in team_index or p not in phase_index:
            continue
        i = team_index[tid]
        t = phase_index[p]
        team_mask[i] = True

        for f, (_name, col, scale, transform) in enumerate(PHASE_NODE_FEATURE_SPECS):
            X[t, i, f] = _as_float(row.get(col), scale=scale, transform=transform)

        if tid not in seen_rank:
            rank = row.get("final_rank")
            if rank is not None:
                y_rank[i] = _as_float(row.get("final_rank_norm"), scale=1.0)
                if y_rank[i] == 0.0:
                    # Fallback for older rows where final_rank_norm is absent.
                    y_rank[i] = float(rank) / max(len(team_ids), 1)
                y_topk[i, 0] = 1.0 if int(rank) <= 5 else 0.0
                y_topk[i, 1] = 1.0 if int(rank) <= 10 else 0.0
            seen_rank.add(tid)

    # Use leakage-safe labels for alive-at-phase-start mask. This comes from the
    # label table rather than inferred feature nullness.
    cur.execute(
        """
        SELECT team_id, cut_phase, alive_at_phase_start
          FROM stdb.team_phase_labels
         WHERE match_id = %s
           AND team_id = ANY(%s)
           AND cut_phase = ANY(%s)
        """,
        (match_id, team_ids, phase_indices),
    )
    for tid, cut_phase, alive_at_phase_start in cur.fetchall():
        tid = int(tid)
        p = int(cut_phase)
        if tid in team_index and p in phase_index:
            node_mask[phase_index[p], team_index[tid]] = bool(alive_at_phase_start)

    return X, y_rank, y_topk, team_mask, node_mask, phase_mask


def _fetch_phase_targets(
    cur,
    match_id: str,
    team_ids: list[int],
    phase_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    y_hazard = np.zeros((PHASE_T_MAX - 1, PHASE_N_MAX), dtype=np.float32)
    target_mask = np.zeros((PHASE_T_MAX - 1, PHASE_N_MAX), dtype=bool)
    team_index = {tid: i for i, tid in enumerate(team_ids)}

    # Hazard target for output index t is cut_phase = t + 1.
    valid_cut_phases = [p for p in phase_indices if 1 <= p < PHASE_T_MAX]
    if not team_ids or not valid_cut_phases:
        return y_hazard, target_mask

    cur.execute(
        """
        SELECT team_id, cut_phase, target_defined, hazard_next
          FROM stdb.team_phase_labels
         WHERE match_id = %s
           AND team_id = ANY(%s)
           AND cut_phase = ANY(%s)
        """,
        (match_id, team_ids, valid_cut_phases),
    )
    for tid, cut_phase, target_defined, hazard_next in cur.fetchall():
        tid = int(tid)
        p = int(cut_phase)
        if tid not in team_index or not (1 <= p < PHASE_T_MAX):
            continue
        out_t = p - 1
        i = team_index[tid]
        target_mask[out_t, i] = bool(target_defined)
        if hazard_next is not None:
            y_hazard[out_t, i] = float(hazard_next)

    return y_hazard, target_mask


def _fetch_phase_edges(
    cur,
    match_id: str,
    team_ids: list[int],
    phase_indices: list[int],
    node_mask: np.ndarray,
) -> np.ndarray:
    A = np.zeros(
        (PHASE_T_MAX, len(PHASE_EDGE_TYPES), PHASE_N_MAX, PHASE_N_MAX),
        dtype=np.float32,
    )
    team_index = {tid: i for i, tid in enumerate(team_ids)}
    phase_index = {p: i for i, p in enumerate(phase_indices)}
    edge_index = {e: i for i, e in enumerate(PHASE_EDGE_TYPES)}

    if team_ids and phase_indices:
        cur.execute(
            """
            SELECT phase_idx, edge_type, src_team_id, dst_team_id, weight
              FROM stdb.team_phase_edge_cache
             WHERE match_id = %s
               AND phase_idx = ANY(%s)
               AND src_team_id = ANY(%s)
               AND dst_team_id = ANY(%s)
            """,
            (match_id, phase_indices, team_ids, team_ids),
        )
        for phase_idx, edge_type, src, dst, weight in cur.fetchall():
            p = int(phase_idx)
            src = int(src)
            dst = int(dst)
            if p not in phase_index or edge_type not in edge_index:
                continue
            if src not in team_index or dst not in team_index:
                continue
            t = phase_index[p]
            e = edge_index[edge_type]
            i = team_index[src]
            j = team_index[dst]
            A[t, e, i, j] = max(A[t, e, i, j], float(weight))

    # Proximity channel gets self-loops for alive nodes. Combat/copressure remain
    # interaction-only; the model's residual path carries each node's own state.
    prox_e = edge_index["proximity"]
    for t in range(PHASE_T_MAX):
        alive_idx = np.where(node_mask[t])[0]
        A[t, prox_e, alive_idx, alive_idx] = 1.0

    # Row-normalize each edge type independently. Rows with no outgoing edge stay
    # zero, which is intentional for combat/copressure.
    for t in range(PHASE_T_MAX):
        for e in range(len(PHASE_EDGE_TYPES)):
            rs = A[t, e].sum(axis=1, keepdims=True)
            np.divide(A[t, e], rs, out=A[t, e], where=rs > 0)

    return A


# ── Public API ───────────────────────────────────────────────────────────────


def fetch_match_tensors(match_id: str) -> MatchTensors | None:
    """Build learning tensors for a single match.  Returns None if not in stdb."""
    conn = connect()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT duration_s FROM stdb.matches WHERE match_id = %s", (match_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        duration_s = int(row[0])

        team_ids = _team_order(cur, match_id)
        n = len(team_ids)
        if n == 0:
            return None

        snapshot_ts = _pick_snapshot_grid(duration_s)
        X = _fetch_team_features(cur, match_id, team_ids, snapshot_ts, duration_s)
        A_t = _fetch_dynamic_adjacency(cur, match_id, team_ids, snapshot_ts)

        # labels
        cur.execute(
            "SELECT team_id, final_rank FROM stdb.teams WHERE match_id = %s",
            (match_id,),
        )
        rank_map = {tid: rk for (tid, rk) in cur.fetchall()}
        y_reg = np.zeros(N_MAX, dtype=np.float32)
        y_cls = np.zeros((N_MAX, 2), dtype=np.float32)
        mask = np.zeros(N_MAX, dtype=bool)
        for j, tid in enumerate(team_ids):
            r = rank_map.get(tid)
            if r is None:
                continue
            y_reg[j] = float(r)
            y_cls[j, 0] = 1.0 if r <= 5 else 0.0
            y_cls[j, 1] = 1.0 if r <= 10 else 0.0
            mask[j] = True

        padded_team_ids = team_ids + [0] * (N_MAX - n)

        return MatchTensors(
            match_id=match_id,
            X=X,
            A_t=A_t,
            y_reg=y_reg,
            y_cls=y_cls,
            mask=mask,
            team_ids=padded_team_ids,
            snapshot_ts=snapshot_ts,
        )
    finally:
        cur.close(); conn.close()


def fetch_phase_match_tensors(match_id: str) -> PhaseMatchTensors | None:
    """Build phase-level tensors for DynamicSTGNN.

    Shapes:
        X           (8, PHASE_N_MAX, F)
        A           (8, 3, PHASE_N_MAX, PHASE_N_MAX), edge types = proximity/combat/copressure
        y_hazard    (7, PHASE_N_MAX), next-phase hazard label
        target_mask (7, PHASE_N_MAX), True where hazard label is defined

    Returns None if the match lacks teams, phase bounds, or edge cache.
    """
    conn = connect()
    cur = conn.cursor()
    try:
        team_ids = _phase_team_order(cur, match_id)
        if not team_ids:
            return None
        phase_indices = _fetch_phase_indices(cur, match_id)
        if not phase_indices:
            return None

        cur.execute(
            "SELECT 1 FROM stdb.team_phase_edge_cache WHERE match_id = %s LIMIT 1",
            (match_id,),
        )
        if cur.fetchone() is None:
            return None

        (
            X,
            y_rank,
            y_topk,
            team_mask,
            node_mask,
            phase_mask,
        ) = _fetch_phase_node_features(cur, match_id, team_ids, phase_indices)
        y_hazard, target_mask = _fetch_phase_targets(
            cur, match_id, team_ids, phase_indices
        )
        A = _fetch_phase_edges(cur, match_id, team_ids, phase_indices, node_mask)

        padded_team_ids = team_ids + [0] * (PHASE_N_MAX - len(team_ids))
        padded_phase_indices = phase_indices + [0] * (PHASE_T_MAX - len(phase_indices))

        return PhaseMatchTensors(
            match_id=match_id,
            X=X,
            A=A,
            y_hazard=y_hazard,
            target_mask=target_mask,
            y_rank=y_rank,
            y_topk=y_topk,
            team_mask=team_mask,
            node_mask=node_mask,
            phase_mask=phase_mask,
            team_ids=padded_team_ids,
            phase_indices=padded_phase_indices,
            feature_names=PHASE_NODE_FEATURE_NAMES.copy(),
            edge_types=PHASE_EDGE_TYPES.copy(),
        )
    finally:
        cur.close()
        conn.close()


def list_eligible_match_ids(
    map_name: str | None = "Baltic_Main",
    mode: str | None = None,
    limit: int | None = None,
) -> list[str]:
    """List stdb.matches that have at least one snapshot and at least 4 teams."""
    conn = connect()
    cur = conn.cursor()
    sql = """
        SELECT m.match_id
          FROM stdb.matches m
         WHERE m.num_teams >= 4
           AND EXISTS (SELECT 1 FROM stdb.moving_objects mo
                        WHERE mo.match_id = m.match_id
                          AND mo.agent_kind = 'team'
                        LIMIT 1)
    """
    params: list = []
    if map_name:
        sql += " AND m.map_name = %s"
        params.append(map_name)
    if mode:
        sql += " AND m.game_mode = %s"
        params.append(mode)
    sql += " ORDER BY m.started_at"
    if limit:
        sql += " LIMIT %s"
        params.append(limit)
    cur.execute(sql, params)
    ids = [r[0] for r in cur.fetchall()]
    cur.close(); conn.close()
    return ids


def list_phase_eligible_match_ids(
    map_name: str | None = None,
    mode: str | None = None,
    limit: int | None = None,
) -> list[str]:
    """List matches ready for phase-level node/edge/label tensor extraction."""
    conn = connect()
    cur = conn.cursor()
    sql = """
        SELECT m.match_id
         FROM stdb.matches m
         WHERE m.num_teams >= 4
           AND m.num_teams <= %s
           AND EXISTS (
                SELECT 1 FROM stdb.team_phase_labels l
                 WHERE l.match_id = m.match_id AND l.target_defined
           )
           AND EXISTS (
                SELECT 1 FROM stdb.team_phase_edge_cache e
                 WHERE e.match_id = m.match_id
           )
    """
    params: list = [PHASE_N_MAX]
    if map_name:
        sql += " AND m.map_name = %s"
        params.append(map_name)
    if mode:
        sql += " AND m.game_mode = %s"
        params.append(mode)
    sql += " ORDER BY m.started_at, m.match_id"
    if limit:
        sql += " LIMIT %s"
        params.append(limit)
    cur.execute(sql, params)
    ids = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return ids


def list_phase_feature_match_ids(
    map_name: str | None = None,
    mode: str | None = None,
    limit: int | None = None,
) -> list[str]:
    """List matches ready for row-level phase feature experiments.

    This intentionally does not require the graph edge cache. Tabular feature
    ablations only need the long phase feature view and should include domains
    whose STDB/node features are available even when dense graph tensors have
    not been materialized.
    """
    conn = connect()
    cur = conn.cursor()
    sql = """
        SELECT DISTINCT match_id
          FROM stdb.v_team_phase_feature_long
         WHERE target_defined
    """
    params: list = []
    if map_name:
        sql += " AND map_name = %s"
        params.append(map_name)
    if mode:
        sql += " AND game_mode = %s"
        params.append(mode)
    sql += " ORDER BY match_id"
    if limit:
        sql += " LIMIT %s"
        params.append(limit)
    cur.execute(sql, params)
    ids = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return ids


def fetch_batch(match_ids: Iterable[str]) -> list[MatchTensors]:
    """Convenience: build tensors for a list of matches."""
    out: list[MatchTensors] = []
    for mid in match_ids:
        t = fetch_match_tensors(mid)
        if t is None:
            logger.warning("skipping %s — not in stdb or no teams", mid)
            continue
        out.append(t)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--match-id")
    ap.add_argument("--phase", action="store_true", help="Use phase-level tensor API")
    args = ap.parse_args()

    if args.phase:
        ids = [args.match_id] if args.match_id else list_phase_eligible_match_ids(limit=1)
        if not ids:
            print("no phase-eligible matches")
        else:
            mtp = fetch_phase_match_tensors(ids[0])
            if mtp is None:
                print("not found")
            else:
                print("match_id       :", mtp.match_id)
                print("X shape        :", mtp.X.shape)
                print("A shape        :", mtp.A.shape)
                print("y_hazard shape :", mtp.y_hazard.shape)
                print("target count   :", int(mtp.target_mask.sum()))
                print("team count     :", int(mtp.team_mask.sum()))
                print("phase count    :", int(mtp.phase_mask.sum()))
                print("node alive     :", int(mtp.node_mask.sum()))
                print("feature dim    :", len(mtp.feature_names))
                print("edge types     :", mtp.edge_types)
    elif args.match_id:
        mt = fetch_match_tensors(args.match_id)
        if mt is None:
            print("not found")
        else:
            print("match_id  :", mt.match_id)
            print("X shape   :", mt.X.shape)
            print("A_t shape :", mt.A_t.shape)
            print("y_reg     :", mt.y_reg[:10])
            print("mask sum  :", mt.mask.sum())
    else:
        ids = list_eligible_match_ids(limit=5)
        print("eligible (first 5):", ids)
