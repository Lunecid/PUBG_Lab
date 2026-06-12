"""
Snapshot-resolution (10s) tensor fetch for the MR-STGNN Tier-S.

Builds, per match, the full snapshot sequence (team nodes + dynamic graph)
that the snapshot encoder consumes. The cumulative cut (use only snapshots
with phase_of_snap <= k when predicting phase k) is applied downstream via
`phase_of_snap`, mirroring how the phase tier feeds a full causal sequence
and masks per cut. So fetch returns the WHOLE match; the model/loss restrict
to <= cut.

Leakage rules baked in here:
  * node features come from stdb.team_snapshot_features (instantaneous/past
    only, see snapshot_feature_etl).
  * proximity edges use team_proximity.snapshot_t (distance AT t).
  * combat edges attribute each event to the snapshot at ceil(event_t/10)*10,
    i.e. the event falls in the PAST window (t-10, t] of that snapshot — never
    the future [t, t+10). This is the dual of the old combat_zone_pass leak.
  * team order = _phase_team_order (team_id sort, outcome-free).

Team indexing (N) is shared with the phase tier (PHASE_N_MAX) so a team keeps
one index across both resolutions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stdb.conn import connect
from stdb.queries import PHASE_N_MAX, _phase_team_order

SNAP_N_MAX: int = PHASE_N_MAX          # 128, shared team index space
SNAP_T_MAX: int = 200                  # max 10s snapshots per match (observed ~196)
SNAP_EDGE_TYPES = ["proximity", "combat"]

# (name, column, scale) — fixed-constant scaling, NULL/absent -> 0.0.
SNAP_FEATURE_SPECS = [
    ("pos_x", "pos_x", 8160.0),
    ("pos_y", "pos_y", 8160.0),
    ("pos_z", "pos_z", 400.0),
    ("rel_zone_x", "rel_zone_x", 1.0),
    ("rel_zone_y", "rel_zone_y", 1.0),
    ("speed_mps", "speed_mps", 40.0),
    ("dist_blue_boundary_m", "dist_blue_boundary_m", 8160.0),
    ("inside_blue", "inside_blue", 1.0),
    ("dist_target_center_m", "dist_target_center_m", 8160.0),
    ("mean_health", "mean_health", 100.0),
    ("min_health", "min_health", 100.0),
    ("alive_n", "alive_n", 4.0),
    ("vehicle_ratio", "vehicle_ratio", 1.0),
]
SNAP_FEATURE_NAMES = [s[0] for s in SNAP_FEATURE_SPECS]
SNAP_FEATURE_DIM = len(SNAP_FEATURE_SPECS)


@dataclass
class SnapshotTensors:
    match_id: str
    X_s: np.ndarray          # (T_s, N, F)
    A_s: np.ndarray          # (T_s, E, N, N) row-normalized
    phase_of_snap: np.ndarray  # (T_s,) int, phase_idx of each snapshot (0 if none)
    node_mask: np.ndarray    # (T_s, N) bool, team alive at snapshot
    death_snap_idx: np.ndarray  # (N,) int, sequence index of team's last-alive snapshot (-1 = censored/absent)
    team_ids: list[int]      # length N (zero-padded tail)
    snapshot_ts: np.ndarray  # (T_s,) int, real snapshot_t (-1 padded)


def _as_float(value, *, scale: float) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(x):
        return 0.0
    return x / scale if scale else x


def _row_normalize(A: np.ndarray) -> np.ndarray:
    """Row-stochastic per (t, e); zero rows stay zero (edge-less node)."""
    rs = A.sum(axis=-1, keepdims=True)
    return np.where(rs > 0, A / np.clip(rs, 1e-9, None), A)


def fetch_snapshot_match_tensors(match_id: str) -> SnapshotTensors | None:
    conn = connect()
    cur = conn.cursor()
    try:
        team_ids = _phase_team_order(cur, match_id)
        if not team_ids:
            return None
        team_index = {tid: i for i, tid in enumerate(team_ids)}

        # snapshot grid = distinct snapshot_t present in the node-feature table
        cur.execute(
            """
            SELECT DISTINCT snapshot_t
              FROM stdb.team_snapshot_features
             WHERE match_id = %s
             ORDER BY snapshot_t
             LIMIT %s
            """,
            (match_id, SNAP_T_MAX),
        )
        snaps = [int(r[0]) for r in cur.fetchall()]
        if not snaps:
            return None
        snap_index = {s: i for i, s in enumerate(snaps)}
        T_s = len(snaps)
        N = SNAP_N_MAX
        F = SNAP_FEATURE_DIM

        X_s = np.zeros((T_s, N, F), dtype=np.float32)
        phase_of_snap = np.zeros(T_s, dtype=np.int64)
        node_mask = np.zeros((T_s, N), dtype=bool)
        last_alive_idx = np.full(N, -1, dtype=np.int64)

        # ---- node features ----
        cols = ["team_id", "snapshot_t", "phase_idx"] + [s[1] for s in SNAP_FEATURE_SPECS]
        cur.execute(
            f"""
            SELECT {', '.join(cols)}
              FROM stdb.team_snapshot_features
             WHERE match_id = %s AND snapshot_t = ANY(%s) AND team_id = ANY(%s)
            """,
            (match_id, snaps, team_ids),
        )
        coldesc = [d[0] for d in cur.description]
        for row in cur.fetchall():
            rd = dict(zip(coldesc, row))
            tid = int(rd["team_id"])
            st = int(rd["snapshot_t"])
            if tid not in team_index or st not in snap_index:
                continue
            i = team_index[tid]
            t = snap_index[st]
            node_mask[t, i] = True
            if rd.get("phase_idx") is not None:
                phase_of_snap[t] = int(rd["phase_idx"])
            for f, (_name, col, scale) in enumerate(SNAP_FEATURE_SPECS):
                X_s[t, i, f] = _as_float(rd.get(col), scale=scale)
            if t > last_alive_idx[i]:
                last_alive_idx[i] = t

        # death = last-alive snapshot index, but only if the team actually
        # disappears before the match's final snapshot (else censored -> -1).
        match_last_t = T_s - 1
        death_snap_idx = np.where(last_alive_idx < match_last_t, last_alive_idx, -1)

        # ---- edges ----
        A_s = np.zeros((T_s, len(SNAP_EDGE_TYPES), N, N), dtype=np.float32)
        PROX, COMBAT = 0, 1

        # proximity: distance AT snapshot_t, symmetric, exp(-d/200); + self-loop on alive
        cur.execute(
            """
            SELECT snapshot_t, team_a, team_b, distance_m
              FROM stdb.team_proximity
             WHERE match_id = %s AND snapshot_t = ANY(%s)
               AND team_a = ANY(%s) AND team_b = ANY(%s)
            """,
            (match_id, snaps, team_ids, team_ids),
        )
        for st, a, b, d in cur.fetchall():
            st = int(st)
            if st not in snap_index or a not in team_index or b not in team_index or d is None:
                continue
            t = snap_index[st]
            ia, ib = team_index[int(a)], team_index[int(b)]
            w = float(np.exp(-float(d) / 200.0))
            A_s[t, PROX, ia, ib] = w
            A_s[t, PROX, ib, ia] = w
        for t in range(T_s):
            for i in range(N):
                if node_mask[t, i]:
                    A_s[t, PROX, i, i] = 1.0  # self-loop on alive node

        # combat: event attributed to snapshot at ceil(event_t/10)*10 -> PAST
        # window (t-10, t]. directed attacker -> victim, log-damped intensity.
        cur.execute(
            """
            SELECT (ceil(event_t / 10.0) * 10)::int AS snap_t,
                   attacker_team, victim_team,
                   sum(CASE WHEN event_kind = 'damage' THEN coalesce(damage, 0) ELSE 0 END)::real AS dmg,
                   count(*) FILTER (WHERE event_kind = 'kill')::int  AS kills,
                   count(*) FILTER (WHERE event_kind = 'groggy')::int AS groggies
              FROM stdb.spatial_events
             WHERE match_id = %s
               AND event_kind IN ('damage', 'kill', 'groggy')
               AND attacker_team IS NOT NULL AND victim_team IS NOT NULL
               AND attacker_team <> victim_team
             GROUP BY 1, 2, 3
            """,
            (match_id,),
        )
        for st, atk, vic, dmg, kills, groggies in cur.fetchall():
            st = int(st)
            if st not in snap_index or atk not in team_index or vic not in team_index:
                continue
            t = snap_index[st]
            ia, iv = team_index[int(atk)], team_index[int(vic)]
            intensity = max(0.0, float(dmg or 0)) + 100.0 * int(kills) + 50.0 * int(groggies)
            if intensity > 0:
                A_s[t, COMBAT, ia, iv] = float(np.log1p(intensity))

        A_s = _row_normalize(A_s)

        # pad snapshot axis to SNAP_T_MAX for batching
        snapshot_ts = np.full(SNAP_T_MAX, -1, dtype=np.int64)
        snapshot_ts[:T_s] = np.array(snaps, dtype=np.int64)
        Xp = np.zeros((SNAP_T_MAX, N, F), dtype=np.float32); Xp[:T_s] = X_s
        Ap = np.zeros((SNAP_T_MAX, len(SNAP_EDGE_TYPES), N, N), dtype=np.float32); Ap[:T_s] = A_s
        pp = np.zeros(SNAP_T_MAX, dtype=np.int64); pp[:T_s] = phase_of_snap
        nm = np.zeros((SNAP_T_MAX, N), dtype=bool); nm[:T_s] = node_mask

        return SnapshotTensors(
            match_id=match_id,
            X_s=Xp,
            A_s=Ap,
            phase_of_snap=pp,
            node_mask=nm,
            death_snap_idx=death_snap_idx,
            team_ids=team_ids + [0] * (N - len(team_ids)),
            snapshot_ts=snapshot_ts,
        )
    finally:
        cur.close()
        conn.close()
