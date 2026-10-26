"""All 150 phase-tier features at ONE landmark cut, from records strictly before the cut (spec rule 2).

``compute_cut_features(src, k, a_k, ...)`` is a pure function of the source records with ``w < a_k`` plus the
explicit arguments ``(k, a_k)`` and the static fields of the source (teams, roster, zone schedule): the very
first thing it does is ``truncate(src, a_k)`` and it never sees the untruncated source again.  The prefix-replay
test therefore compares it with a source that the DATABASE truncated (``fetch_match_source(cut_w=a_k)``).

Definitions follow the v1 ETL SQL (stdb/rq1_feature_etl.py, phase_spatial_feature_etl.py, phase_edge_etl.py,
network_feature_etl.py, habitat_feature_etl.py, z_affordance_feature_etl.py); what changed is listed in
``REPLACED_FEATURES`` / ``CHANGED_FEATURES``.  Phase k window = ``[s_k, a_k)`` with ``s_k`` the first
LogPhaseChange of phase k and ``a_k = b_k - 10 s``; the grid points of the window are the admissible multiples of
10 s (``t >= s_k`` and ``t + 5 < a_k``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pubg_lab.survival.phase_v2 import alive as AL
from pubg_lab.survival.phase_v2 import events as EV
from pubg_lab.survival.phase_v2 import grid as GR
from pubg_lab.survival.phase_v2.phases import CUT_LEAD_S, phases_of
from pubg_lab.survival.phase_v2.source import MatchSource, truncate
from pubg_lab.data.stdb.queries import PHASE_NODE_FEATURE_SPECS, PHASE_NODE_FEATURE_SPECS_V2

SPECS = list(PHASE_NODE_FEATURE_SPECS) + list(PHASE_NODE_FEATURE_SPECS_V2)
FEATURE_NAMES: list[str] = [s[0] for s in SPECS]
N_FEATURES = len(FEATURE_NAMES)
ALIVE_SCALE = 4.0
PHASE_NORM = 8.0
EDGE_PRESSURE_M = 150.0  # habitat / co-pressure threshold
CONTACT_M = 200.0
HEIGHT_SCALE_M = 50.0
Z_MAX_PHASE = 7  # the z-affordance builder only exists for phases 1..7
EDGE_KEYS = ("prox", "combat", "copress")

# v1 columns whose VALUES are recomputed in v2 with the same names (leak fixes L1-L3 / D3)
REPLACED_FEATURES: tuple[str, ...] = (
    "alive_players_start_norm", "alive_players_end_norm", "alive_players_min_norm", "alive_players_avg_norm",
)  # fmt: skip
# every other column is recomputed under rule 2 (cut a_k = b_k - 10 s, W clock, W grid); these additionally change
# their DEFINITION (scheduled instead of observed phase duration, D3) or their input set (L2: uncapped events)
DURATION_FEATURES: tuple[str, ...] = (
    "phase_duration_10min", "avg_speed_norm", "rotation_timing", "zone_entry_time",
    "net_combat_out_strength_per_min", "net_combat_in_strength_per_min",
    "hab_shrink_rate_m2ps", "hab_required_speed_to_safe_mps", "hab_observed_speed_mps",
    "hab_migration_deficit_mps", "hab_rotation_delay_ratio",
)  # fmt: skip
EVENT_FEATURES_L2: tuple[str, ...] = tuple(
    s[0]
    for s in SPECS
    if s[1]
    in (
        "recent_damage_taken", "combat_zone_pass_count", "damage_dealt", "damage_taken", "damage_events_dealt",
        "damage_events_taken", "groggy_dealt", "groggy_taken", "kills_dealt", "kills_taken",
        "distinct_opponents_damaged", "distinct_opponents_damaged_by", "combat_initiated",
        "pickup_count", "equip_count", "use_count", "unique_item_count", "weapon_pickup_count",
        "weapon_equip_count", "ammo_pickup_count", "armor_pickup_count", "throwable_pickup_count",
        "heal_use_count", "boost_use_count", "carepackage_pickup_count", "lootbox_pickup_count",
    )
    or s[1].startswith(("net_combat", "cum_"))
)  # fmt: skip


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def scale_columns(raw: dict[str, np.ndarray], n_teams: int) -> np.ndarray:
    """``[N, F]`` float32 exactly like ``stdb.queries._as_float``: value / scale, NULL / NaN / inf -> 0.0,
    bool -> {0, 1}, log1p for the sinuosity."""
    X = np.zeros((n_teams, N_FEATURES), dtype=np.float32)
    for f, (_name, col, scale, transform) in enumerate(SPECS):
        v = raw.get(col)
        if v is None:
            continue
        v = np.asarray(v, dtype=np.float64)
        ok = np.isfinite(v)
        x = np.zeros(n_teams, dtype=np.float64)
        if transform == "bool":
            x[ok] = (v[ok] != 0).astype(np.float64)
        elif transform == "log1p":
            x[ok] = np.log1p(np.maximum(v[ok], 0.0))
        else:
            x[ok] = v[ok] / scale if scale else v[ok]
        X[:, f] = x.astype(np.float32)
    return X


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """PostgreSQL ``regr_slope(y, x)`` (NaN when fewer than 2 points or constant x)."""
    if len(x) < 2:
        return float("nan")
    xm, ym = float(np.mean(x)), float(np.mean(y))
    sxx = float(np.sum((x - xm) ** 2))
    if sxx <= 0.0:
        return float("nan")
    return float(np.sum((x - xm) * (y - ym)) / sxx)


def _nan_full(n: int) -> np.ndarray:
    return np.full(n, np.nan, dtype=np.float64)


def _first_last(mask_idx: np.ndarray) -> tuple[int | None, int | None]:
    return (None, None) if len(mask_idx) == 0 else (int(mask_idx[0]), int(mask_idx[-1]))


# --------------------------------------------------------------------------------------
# phase-global block (match_phase_global): zone geometry and population from the game states
# --------------------------------------------------------------------------------------
@dataclass
class GlobalBlock:
    avg_alive_teams: float = float("nan")
    avg_alive_players: float = float("nan")
    end_alive_teams: float = float("nan")
    safe_start: tuple[float, float, float] | None = None  # (x, y, r)
    safe_end: tuple[float, float, float] | None = None
    poison_start: tuple[float, float, float] | None = None
    poison_end: tuple[float, float, float] | None = None


def global_block(gs: dict[str, np.ndarray], lo: float, hi: float) -> GlobalBlock:
    m = (gs["w"] >= lo) & (gs["w"] < hi)
    out = GlobalBlock()
    idx = np.nonzero(m)[0]
    if len(idx) == 0:
        return out
    for field_name, col in (("avg_alive_teams", "n_teams"), ("avg_alive_players", "n_players")):
        v = gs[col][idx]
        v = v[np.isfinite(v)]
        if len(v):
            setattr(out, field_name, float(np.mean(v)))
    out.end_alive_teams = float(gs["n_teams"][idx[-1]])
    for kind in ("safe", "poison"):
        ok = idx[np.isfinite(gs[f"{kind}_r"][idx])]
        first, last = _first_last(ok)
        if first is not None:
            setattr(
                out,
                f"{kind}_start",
                tuple(float(gs[f"{kind}_{c}"][first]) for c in ("x", "y", "r")),
            )
            setattr(
                out, f"{kind}_end", tuple(float(gs[f"{kind}_{c}"][last]) for c in ("x", "y", "r"))
            )
    return out


# --------------------------------------------------------------------------------------
# team state from raw positions (team_phase_state without the four alive columns)
# --------------------------------------------------------------------------------------
STATE_COLUMNS = (
    "mean_health_avg", "min_health_min", "health_slope", "team_spread_avg_m", "team_spread_max_m",
    "vehicle_ratio", "vehicle_speed_mean", "vehicle_speed_max",
)  # fmt: skip


def state_features(
    pos: dict[str, np.ndarray], lo: float, hi: float, team_index: dict[int, int]
) -> dict[str, np.ndarray]:
    """Per (team, integer elapsed second) snapshots of the raw positions inside ``[lo, hi)``, then per-team
    aggregates exactly as the SQL (snap -> spread -> agg)."""
    N = len(team_index)
    out = {c: _nan_full(N) for c in STATE_COLUMNS}
    m = (pos["w"] >= lo) & (pos["w"] < hi)
    if not m.any():
        return out
    tmap = np.fromiter(
        (team_index.get(int(t), -1) for t in pos["team"][m]), dtype=np.int64, count=int(m.sum())
    )
    df = pd.DataFrame(
        {
            "team": tmap,
            "g": pos["g"][m],
            "health": pos["health"][m],
            "x": pos["x"][m],
            "y": pos["y"][m],
            "veh": pos["veh"][m].astype(np.float64),
            "vspeed": np.where(pos["veh"][m], pos["vspeed"][m], np.nan),
        }
    )
    df = df[df["team"] >= 0]
    if df.empty:
        return out
    h = df["health"].to_numpy()
    df["healthy"] = np.isfinite(h) & (h > 0)
    hd = df[df["healthy"]].copy()
    grp = ["team", "g"]
    snap = df.groupby(grp, sort=True).agg(
        veh_ratio=("veh", "mean"), vs_mean=("vspeed", "mean"), vs_max=("vspeed", "max")
    )
    if not hd.empty:
        hs = hd.groupby(grp, sort=True).agg(mean_health=("health", "mean"), min_health=("health", "min"),
                                            cx=("x", "mean"), cy=("y", "mean"))  # fmt: skip
        hd = hd.join(hs[["cx", "cy"]], on=grp)
        hd["dist"] = np.hypot(hd["x"] - hd["cx"], hd["y"] - hd["cy"])
        sp = hd.groupby(grp, sort=True).agg(spread_avg=("dist", "mean"), spread_max=("dist", "max"))
        snap = snap.join(hs[["mean_health", "min_health"]], how="left").join(sp, how="left")
    else:
        snap["mean_health"] = np.nan
        snap["min_health"] = np.nan
        snap["spread_avg"] = np.nan
        snap["spread_max"] = np.nan
    snap = snap.reset_index()
    for team, grp_df in snap.groupby("team", sort=True):
        i = int(team)
        mh = grp_df["mean_health"].to_numpy()
        okm = np.isfinite(mh)
        if okm.any():
            out["mean_health_avg"][i] = float(np.mean(mh[okm]))
            out["min_health_min"][i] = float(np.nanmin(grp_df["min_health"].to_numpy()))
            out["health_slope"][i] = ols_slope(grp_df["g"].to_numpy(dtype=np.float64)[okm], mh[okm])
        sa = grp_df["spread_avg"].to_numpy()
        oks = np.isfinite(sa)
        if oks.any():
            out["team_spread_avg_m"][i] = float(np.mean(sa[oks]))
            out["team_spread_max_m"][i] = float(np.nanmax(grp_df["spread_max"].to_numpy()))
        out["vehicle_ratio"][i] = float(np.mean(grp_df["veh_ratio"].to_numpy()))
        vm = grp_df["vs_mean"].to_numpy()
        okv = np.isfinite(vm)
        if okv.any():
            out["vehicle_speed_mean"][i] = float(np.mean(vm[okv]))
            out["vehicle_speed_max"][i] = float(np.nanmax(grp_df["vs_max"].to_numpy()))
    return out


# --------------------------------------------------------------------------------------
# per-team grid features: trajectory, exposure, habitat movement/local blocks, z-affordance
# --------------------------------------------------------------------------------------
TRAJ_COLUMNS = (
    "path_length_m", "displacement_m", "sinuosity", "avg_speed_mps", "rotation_timing", "safe_alignment",
    "zone_entry_time", "outside_safe_duration_s", "boundary_distance_slope",
)  # fmt: skip
EXPO_COLUMNS = (
    "mean_dist_to_boundary_m", "min_dist_to_boundary_m", "outside_safe_ratio", "avg_enemy_density_100m",
    "min_nearest_enemy_m", "path_near_enemy_ratio", "recent_damage_taken", "combat_zone_pass_count",
)  # fmt: skip
HAB_TEAM_COLUMNS = (
    "hab_dist_to_safe_boundary_start_m", "hab_dist_to_safe_boundary_end_m", "hab_signed_boundary_distance_start_m",
    "hab_signed_boundary_distance_end_m", "hab_distance_to_target_safe_m", "hab_required_speed_to_safe_mps",
    "hab_observed_speed_mps", "hab_migration_deficit_mps", "hab_outside_safe_ratio", "hab_edge_pressure_ratio",
    "hab_rotation_delay_ratio", "hab_safe_alignment_to_target", "hab_local_enemy_count_100m",
    "hab_local_enemy_count_200m", "hab_local_enemy_count_500m", "hab_local_enemy_density_200m",
    "hab_safe_entry_enemy_count_200m", "hab_safe_entry_congestion", "hab_refuge_scarcity_simple",
)  # fmt: skip
Z_COLUMNS = (
    "team_z_mean_m", "team_z_range_m", "team_z_percentile_alive", "contact_delta_z_mean_m",
    "contact_delta_z_positive_ratio", "high_ground_contact_advantage", "high_ground_x_boundary_pressure",
    "z_percentile_x_hab_crowding_pressure", "contact_delta_z_x_outside_safe_ratio",
)  # fmt: skip


def _cos(ax: float, ay: float, bx: float, by: float, min_norm: float = 1.0) -> float:
    na, nb = math.hypot(ax, ay), math.hypot(bx, by)
    if na > min_norm and nb > min_norm:
        return (ax * bx + ay * by) / (na * nb)
    return float("nan")


def team_grid_features(
    grid: GR.TeamGrid,
    gi: np.ndarray,
    safe: tuple[np.ndarray, np.ndarray, np.ndarray],
    D: np.ndarray,
    duration_s: float | None,
    start_w: float,
    ev: EV.TeamEvents,
    ev_lo: float,
    ev_hi: float,
    target: tuple[float, float, float] | None,
    blue_end_r: float | None,
    z_phase: bool,
    team_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Trajectory + exposure (phase_spatial_feature_etl), habitat team/local blocks (habitat_feature_etl) and the
    z-affordance block for the grid points ``gi`` of ONE phase window.  Returns ``{column: [N] float64}`` (NaN =
    SQL NULL).  ``duration_s`` is the SCHEDULED phase duration (D3); ``target`` the white circle visible at the cut
    (x, y, r) or None."""
    N = grid.alive.shape[1]
    out: dict[str, np.ndarray] = {
        c: _nan_full(N) for c in (*TRAJ_COLUMNS, *EXPO_COLUMNS, *HAB_TEAM_COLUMNS, *Z_COLUMNS)
    }
    if len(gi) == 0:
        out["_has_rows"] = np.zeros(N)
        return out
    dur = float(duration_s) if duration_s is not None and np.isfinite(duration_s) else float("nan")
    A = grid.alive[gi]
    cx, cy, cz = grid.cx[gi], grid.cy[gi], grid.cz[gi]
    gt = grid.g[gi]
    sx, sy, sr = safe[0][gi], safe[1][gi], safe[2][gi]
    region_ok = np.isfinite(sr)
    d_ctr = np.hypot(cx - sx[:, None], cy - sy[:, None])
    in_safe = np.where(region_ok[:, None], (d_ctr < sr[:, None]).astype(np.float64), np.nan)
    d_bound = np.where(region_ok[:, None], np.abs(sr[:, None] - d_ctr), np.nan)
    Dw = D[gi]
    # events that sit on a grid bucket [g, g + 10): kills / damage / groggy with a position
    evm = (ev.w >= ev_lo) & (ev.w < ev_hi) & np.isfinite(ev.px) & np.isfinite(ev.py)
    e_w, e_x, e_y = ev.w[evm], ev.px[evm], ev.py[evm]
    # recent damage taken: every damage event of the window whose victim has a team
    dm = (ev.w >= ev_lo) & (ev.w < ev_hi) & (ev.kind == EV.KIND_DAMAGE) & (ev.vic >= 0)
    pass_cnt = np.zeros(N)
    for r in range(len(gi)):
        lo_i = int(np.searchsorted(e_w, gt[r], side="left"))
        hi_i = int(np.searchsorted(e_w, gt[r] + GR.INTERVAL, side="left"))
        if hi_i > lo_i:
            al = np.nonzero(A[r])[0]
            if len(al):
                d = np.hypot(
                    cx[r, al][:, None] - e_x[lo_i:hi_i][None, :],
                    cy[r, al][:, None] - e_y[lo_i:hi_i][None, :],
                )
                pass_cnt[al] += (d <= 200.0).sum(axis=1)
    dmg_by_team: dict[int, float] = {}
    for vt, dmg in zip(ev.vic[dm], ev.damage[dm]):
        dmg_by_team[int(vt)] = dmg_by_team.get(int(vt), 0.0) + float(dmg)

    tx = ty = tr = None
    if target is not None:
        tx, ty, tr = target
    z_means = _nan_full(N)
    rows_of: list[np.ndarray] = []
    for i in range(N):
        rows = np.nonzero(A[:, i])[0]
        rows_of.append(rows)
        n = len(rows)
        if n == 0:
            continue
        x, y, z, g = cx[rows, i], cy[rows, i], cz[rows, i], gt[rows]
        # ---- trajectory --------------------------------------------------------------------------
        path = float(np.sum(np.hypot(np.diff(x), np.diff(y)))) if n >= 2 else 0.0
        disp = float(math.hypot(x[-1] - x[0], y[-1] - y[0]))
        out["path_length_m"][i] = path
        out["displacement_m"][i] = disp
        out["sinuosity"][i] = path / disp if disp > 1.0 else np.nan
        out["avg_speed_mps"][i] = path / dur if dur > 0 else np.nan
        if disp > 1.0 and dur > 0:
            far = np.nonzero(np.hypot(x - x[0], y - y[0]) >= 0.3 * disp)[0]
            if len(far):
                out["rotation_timing"][i] = min(max((g[far[0]] - start_w) / dur, 0.0), 1.0)
        has_c = np.nonzero(region_ok[rows])[0]
        if len(has_c):
            lr = rows[has_c[-1]]  # last row with a visible blue zone
            out["safe_alignment"][i] = _cos(
                x[-1] - x[0], y[-1] - y[0], sx[lr] - x[0], sy[lr] - y[0]
            )
        ins = in_safe[rows, i]
        first_in = np.nonzero(ins == 1.0)[0]
        if len(first_in) and dur > 0:
            out["zone_entry_time"][i] = min(max((g[first_in[0]] - start_w) / dur, 0.0), 1.0)
        out["outside_safe_duration_s"][i] = GR.INTERVAL * float(np.sum(ins == 0.0))
        db = d_bound[rows, i]
        okd = np.isfinite(db)
        out["boundary_distance_slope"][i] = ols_slope(g[okd], db[okd]) if okd.any() else np.nan
        # ---- exposure ----------------------------------------------------------------------------
        if okd.any():
            out["mean_dist_to_boundary_m"][i] = float(np.mean(db[okd]))
            out["min_dist_to_boundary_m"][i] = float(np.min(db[okd]))
        out["outside_safe_ratio"][i] = float(np.mean((ins == 0.0).astype(np.float64)))
        Di = Dw[rows, i, :]
        fin = np.isfinite(Di)
        n100 = np.sum(fin & (Di < 100.0), axis=1)
        out["avg_enemy_density_100m"][i] = float(np.mean(n100))
        out["path_near_enemy_ratio"][i] = float(np.mean((n100 > 0).astype(np.float64)))
        if fin.any():
            out["min_nearest_enemy_m"][i] = float(np.min(np.where(fin, Di, np.inf)))
        out["recent_damage_taken"][i] = dmg_by_team.get(int(team_ids[i]), 0.0)
        out["combat_zone_pass_count"][i] = float(pass_cnt[i])
        # ---- habitat: team movement-pressure block -----------------------------------------------
        okr = np.nonzero(np.isfinite(ins))[0]
        if len(okr):
            d0, d1 = db[okr[0]], db[okr[-1]]
            out["hab_dist_to_safe_boundary_start_m"][i] = d0
            out["hab_dist_to_safe_boundary_end_m"][i] = d1
            out["hab_signed_boundary_distance_start_m"][i] = d0 if ins[okr[0]] == 1.0 else -d0
            out["hab_signed_boundary_distance_end_m"][i] = d1 if ins[okr[-1]] == 1.0 else -d1
            pressure = (ins[okr] == 0.0) | ((ins[okr] == 1.0) & (db[okr] <= EDGE_PRESSURE_M))
            out["hab_edge_pressure_ratio"][i] = float(np.mean(pressure.astype(np.float64)))
        out["hab_outside_safe_ratio"][i] = out["outside_safe_ratio"][i]
        out["hab_observed_speed_mps"][i] = out["avg_speed_mps"][i]
        if target is not None:
            dt = max(math.hypot(x[-1] - tx, y[-1] - ty) - tr, 0.0)
            out["hab_distance_to_target_safe_m"][i] = dt
            if dur > 0:
                req = dt / dur
                out["hab_required_speed_to_safe_mps"][i] = req
                out["hab_migration_deficit_mps"][i] = req - out["avg_speed_mps"][i]
            inside = np.hypot(x - tx, y - ty) < tr
            ent = np.nonzero(inside)[0]
            if len(ent) == 0:
                out["hab_rotation_delay_ratio"][i] = 1.0
            elif dur > 0:
                out["hab_rotation_delay_ratio"][i] = min(max((g[ent[0]] - start_w) / dur, 0.0), 1.0)
            if disp > 1.0 and math.hypot(tx - x[0], ty - y[0]) > 1.0:
                out["hab_safe_alignment_to_target"][i] = _cos(
                    x[-1] - x[0], y[-1] - y[0], tx - x[0], ty - y[0]
                )
            if len(ent):
                re_ = rows[ent[0]]
                De = Dw[re_, i, :]
                fe = np.isfinite(De)
                cnt200 = int(np.sum(fe & (De < 200.0)))
                out["hab_safe_entry_enemy_count_200m"][i] = cnt200
                al = np.nonzero(A[re_])[0]
                al = al[al != i]
                out["hab_safe_entry_congestion"][i] = int(
                    np.sum(np.hypot(cx[re_, al] - tx, cy[re_, al] - ty) < tr)
                )
                if blue_end_r is not None:
                    out["hab_refuge_scarcity_simple"][i] = cnt200 / (math.pi * blue_end_r**2 + 1.0)
        # local competition block at the team's last alive grid point
        Dl = Dw[rows[-1], i, :]
        fl = np.isfinite(Dl)
        out["hab_local_enemy_count_100m"][i] = int(np.sum(fl & (Dl < 100.0)))
        out["hab_local_enemy_count_200m"][i] = int(np.sum(fl & (Dl < 200.0)))
        out["hab_local_enemy_count_500m"][i] = int(np.sum(fl & (Dl < 500.0)))
        out["hab_local_enemy_density_200m"][i] = out["hab_local_enemy_count_200m"][i] / (
            math.pi * 0.2**2
        )
        # ---- z-affordance ------------------------------------------------------------------------
        if z_phase:
            out["team_z_mean_m"][i] = float(np.mean(z))
            out["team_z_range_m"][i] = float(np.max(z) - np.min(z))
            z_means[i] = out["team_z_mean_m"][i]
            rr, jj = np.nonzero(fin & (Di <= CONTACT_M))
            if len(rr):
                delta = z[rr] - cz[rows[rr], jj]
                out["contact_delta_z_mean_m"][i] = float(np.mean(delta))
                out["contact_delta_z_positive_ratio"][i] = float(
                    np.mean((delta > 0).astype(np.float64))
                )
                out["high_ground_contact_advantage"][i] = float(
                    np.mean(np.clip(delta / HEIGHT_SCALE_M, -1.0, 1.0))
                )
    # percent_rank of the team z mean among the teams that have one (PARTITION BY phase)
    if z_phase:
        have = np.nonzero(np.isfinite(z_means))[0]
        if len(have):
            zs = z_means[have]
            nh = len(have)
            for i in have:
                out["team_z_percentile_alive"][i] = (
                    (float(np.sum(zs < z_means[i])) / (nh - 1)) if nh > 1 else 0.0
                )
    out["_has_rows"] = np.asarray([len(r) > 0 for r in rows_of], dtype=np.float64)
    return out


# --------------------------------------------------------------------------------------
# edges of one phase window (phase_edge_etl) and the network features built from them
# --------------------------------------------------------------------------------------
@dataclass
class PhaseEdges:
    """Directed team edges of one phase window (indices are team rows).  ``prox`` / ``copress`` are stored in both
    directions like the edge cache; ``combat`` is attacker -> victim."""

    prox_i: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    prox_j: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    prox_w: np.ndarray = field(default_factory=lambda: np.zeros(0))
    prox_d: np.ndarray = field(
        default_factory=lambda: np.zeros(0)
    )  # mean grid distance of the pair
    comb_i: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    comb_j: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    comb_w: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cop_i: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    cop_j: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    cop_w: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cop_ratio: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cop_support: np.ndarray = field(
        default_factory=lambda: np.zeros(0)
    )  # copress_count of the pair

    def as_triples(self) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return {
            "prox": (self.prox_i, self.prox_j, self.prox_w),
            "combat": (self.comb_i, self.comb_j, self.comb_w),
            "copress": (self.cop_i, self.cop_j, self.cop_w),
        }


def phase_edges(
    grid: GR.TeamGrid,
    gi: np.ndarray,
    safe: tuple[np.ndarray, np.ndarray, np.ndarray],
    D: np.ndarray,
    ev_win: EV.TeamEvents,
    team_index: dict[int, int],
) -> PhaseEdges:
    out = PhaseEdges()
    comb = EV.combat_edges(ev_win, team_index)
    if comb:
        out.comb_i = np.asarray([c[0] for c in comb], dtype=np.int64)
        out.comb_j = np.asarray([c[1] for c in comb], dtype=np.int64)
        out.comb_w = np.asarray([c[2] for c in comb], dtype=np.float64)
    if len(gi) == 0:
        return out
    Dw = D[gi]
    valid = np.isfinite(Dw)
    C = valid.sum(axis=0)
    S = np.where(valid, Dw, 0.0).sum(axis=0)
    ii, jj = np.nonzero(C > 0)  # row-major: ascending (i, j)
    avg = S[ii, jj] / C[ii, jj]
    out.prox_i, out.prox_j, out.prox_d = ii.astype(np.int64), jj.astype(np.int64), avg
    out.prox_w = np.exp(-avg / 200.0)
    # co-pressure: both teams outside the blue zone or within 150 m of its boundary, and within 200 m of each other
    A = grid.alive[gi]
    sx, sy, sr = safe[0][gi], safe[1][gi], safe[2][gi]
    region_ok = np.isfinite(sr)
    d_ctr = np.hypot(grid.cx[gi] - sx[:, None], grid.cy[gi] - sy[:, None])
    under = (
        A
        & region_ok[:, None]
        & ((d_ctr >= sr[:, None]) | (np.abs(sr[:, None] - d_ctr) <= EDGE_PRESSURE_M))
    )
    # NB: a point exactly on the boundary is not contained in the polygon -> counted as outside (d >= r)
    both = under[:, :, None] & under[:, None, :] & valid
    cop = both & (Dw <= CONTACT_M)
    n_cop = cop.sum(axis=0)
    ci, cj = np.nonzero(n_cop > 0)
    if len(ci):
        s_cd = np.where(cop, Dw, 0.0).sum(axis=0)[ci, cj]
        avg_cd = s_cd / n_cop[ci, cj]
        ratio = n_cop[ci, cj] / C[ci, cj]
        out.cop_i, out.cop_j = ci.astype(np.int64), cj.astype(np.int64)
        out.cop_w = ratio * np.exp(-avg_cd / 200.0)
        out.cop_ratio = ratio
        out.cop_support = n_cop[ci, cj].astype(np.float64)
    return out


NET_LOCAL_COLUMNS = (
    "net_combat_out_degree", "net_combat_in_degree", "net_combat_total_degree", "net_combat_out_strength",
    "net_combat_in_strength", "net_combat_total_strength", "net_combat_balance", "net_combat_unique_targets",
    "net_combat_unique_attackers", "net_combat_reciprocal_pairs", "net_combat_reciprocity_ratio",
    "net_combat_out_strength_per_min", "net_combat_in_strength_per_min", "net_prox_degree_100m",
    "net_prox_degree_200m", "net_prox_degree_500m", "net_prox_strength_decay", "net_prox_mean_distance_m",
    "net_prox_min_distance_m", "net_prox_top1_weight", "net_prox_top3_strength", "net_prox_isolation_m",
    "net_prox_neighbor_entropy", "net_copress_degree", "net_copress_strength", "net_copress_max_weight",
    "net_copress_mean_ratio", "net_copress_shared_pressure_count", "net_copress_top1_weight",
    "cum_combat_out_degree", "cum_combat_in_degree", "cum_combat_out_strength", "cum_combat_in_strength",
    "cum_combat_total_strength", "cum_combat_reciprocal_pairs", "cum_unique_targets", "cum_unique_attackers",
    "cum_copress_strength", "cum_prox_close_exposure_200m", "net_combat_out_strength_per_alive_team",
    "net_combat_in_strength_per_alive_team", "net_prox_degree_200m_per_alive_team",
    "net_copress_degree_per_alive_team",
)  # fmt: skip


def network_features(
    edges: dict[int, PhaseEdges],
    k: int,
    n_teams: int,
    avg_alive_teams: float,
    duration_s: float | None,
) -> dict[str, np.ndarray]:
    """``net_*`` (phase k edges) and ``cum_*`` (phases 1..k) of network_feature_etl, from the raw edge weights."""
    N = n_teams
    out = {c: np.zeros(N) for c in NET_LOCAL_COLUMNS}
    nan_cols = (
        "net_combat_out_strength_per_min", "net_combat_in_strength_per_min", "net_prox_mean_distance_m",
        "net_prox_min_distance_m", "net_prox_top1_weight", "net_prox_top3_strength", "net_prox_isolation_m",
        "net_prox_neighbor_entropy", "net_combat_out_strength_per_alive_team",
        "net_combat_in_strength_per_alive_team", "net_prox_degree_200m_per_alive_team",
        "net_copress_degree_per_alive_team",
    )  # fmt: skip
    for c in nan_cols:
        out[c] = _nan_full(N)
    e = edges.get(k, PhaseEdges())
    # ---- combat local
    deg_o = np.bincount(e.comb_i, minlength=N).astype(np.float64)
    deg_i = np.bincount(e.comb_j, minlength=N).astype(np.float64)
    str_o = np.bincount(e.comb_i, weights=e.comb_w, minlength=N)
    str_i = np.bincount(e.comb_j, weights=e.comb_w, minlength=N)
    pairs = set(zip(e.comb_i.tolist(), e.comb_j.tolist()))
    recip = np.zeros(N)
    for a, b in pairs:
        if (b, a) in pairs:
            recip[a] += 1
    out["net_combat_out_degree"], out["net_combat_in_degree"] = deg_o, deg_i
    out["net_combat_total_degree"] = deg_o + deg_i
    out["net_combat_out_strength"], out["net_combat_in_strength"] = str_o, str_i
    out["net_combat_total_strength"] = str_o + str_i
    out["net_combat_balance"] = (str_o - str_i) / (str_o + str_i + 1e-6)
    out["net_combat_unique_targets"], out["net_combat_unique_attackers"] = (
        deg_o.copy(),
        deg_i.copy(),
    )
    out["net_combat_reciprocal_pairs"] = recip
    denom = deg_o + deg_i - recip
    out["net_combat_reciprocity_ratio"] = np.where(
        denom > 0, recip / np.where(denom > 0, denom, 1.0), 0.0
    )
    if duration_s is not None and np.isfinite(duration_s) and duration_s > 0:
        out["net_combat_out_strength_per_min"] = str_o * 60.0 / duration_s
        out["net_combat_in_strength_per_min"] = str_i * 60.0 / duration_s
    # ---- proximity local
    pi_, pd_, pw = e.prox_i, e.prox_d, e.prox_w
    out["net_prox_degree_100m"] = np.bincount(pi_[pd_ < 100], minlength=N).astype(np.float64)
    out["net_prox_degree_200m"] = np.bincount(pi_[pd_ < 200], minlength=N).astype(np.float64)
    out["net_prox_degree_500m"] = np.bincount(pi_[pd_ < 500], minlength=N).astype(np.float64)
    strength = np.bincount(pi_, weights=pw, minlength=N)
    out["net_prox_strength_decay"] = strength
    cnt = np.bincount(pi_, minlength=N)
    for i in np.nonzero(cnt)[0]:
        sel = pi_ == i
        w_i, d_i = pw[sel], pd_[sel]
        out["net_prox_mean_distance_m"][i] = float(np.mean(d_i))
        out["net_prox_min_distance_m"][i] = float(np.min(d_i))
        out["net_prox_isolation_m"][i] = float(np.min(d_i))
        out["net_prox_top1_weight"][i] = float(np.max(w_i))
        out["net_prox_top3_strength"][i] = float(np.sum(np.sort(w_i)[::-1][:3]))
        if strength[i] > 0:
            wl = float(np.sum(w_i[w_i > 0] * np.log(w_i[w_i > 0])))
            out["net_prox_neighbor_entropy"][i] = float(np.log(strength[i]) - wl / strength[i])
    # ---- co-pressure local
    ci = e.cop_i
    out["net_copress_degree"] = np.bincount(ci, minlength=N).astype(np.float64)
    out["net_copress_strength"] = np.bincount(ci, weights=e.cop_w, minlength=N)
    out["net_copress_shared_pressure_count"] = np.bincount(ci, weights=e.cop_support, minlength=N)
    ccnt = np.bincount(ci, minlength=N)
    sum_ratio = np.bincount(ci, weights=e.cop_ratio, minlength=N)
    out["net_copress_mean_ratio"] = np.where(ccnt > 0, sum_ratio / np.maximum(ccnt, 1), 0.0)
    for i in np.nonzero(ccnt)[0]:
        mx = float(np.max(e.cop_w[ci == i]))
        out["net_copress_max_weight"][i] = mx
        out["net_copress_top1_weight"][i] = mx
    # ---- cumulative over phases 1..k
    cum_out_deg = np.zeros(N)
    cum_in_deg = np.zeros(N)
    cum_out_str = np.zeros(N)
    cum_in_str = np.zeros(N)
    cum_cop = np.zeros(N)
    cum_prox = np.zeros(N)
    union: set[tuple[int, int]] = set()
    for j, ej in edges.items():
        if j > k:
            continue
        cum_out_deg += np.bincount(ej.comb_i, minlength=N)
        cum_in_deg += np.bincount(ej.comb_j, minlength=N)
        cum_out_str += np.bincount(ej.comb_i, weights=ej.comb_w, minlength=N)
        cum_in_str += np.bincount(ej.comb_j, weights=ej.comb_w, minlength=N)
        cum_cop += np.bincount(ej.cop_i, weights=ej.cop_w, minlength=N)
        cum_prox += np.bincount(ej.prox_i[ej.prox_d < 200], minlength=N)
        union.update(zip(ej.comb_i.tolist(), ej.comb_j.tolist()))
    targets = np.zeros(N)
    attackers = np.zeros(N)
    cum_recip = np.zeros(N)
    seen_t = [set() for _ in range(N)]
    seen_a = [set() for _ in range(N)]
    for a, b in union:
        seen_t[a].add(b)
        seen_a[b].add(a)
        if (b, a) in union:
            cum_recip[a] += 1
    targets = np.asarray([len(s) for s in seen_t], dtype=np.float64)
    attackers = np.asarray([len(s) for s in seen_a], dtype=np.float64)
    out["cum_combat_out_degree"], out["cum_combat_in_degree"] = cum_out_deg, cum_in_deg
    out["cum_combat_out_strength"], out["cum_combat_in_strength"] = cum_out_str, cum_in_str
    out["cum_combat_total_strength"] = cum_out_str + cum_in_str
    out["cum_combat_reciprocal_pairs"] = cum_recip
    out["cum_unique_targets"], out["cum_unique_attackers"] = targets, attackers
    out["cum_copress_strength"], out["cum_prox_close_exposure_200m"] = cum_cop, cum_prox
    # ---- phase-normalised (nullif(avg_alive_teams, 0))
    if np.isfinite(avg_alive_teams) and avg_alive_teams != 0:
        out["net_combat_out_strength_per_alive_team"] = str_o / avg_alive_teams
        out["net_combat_in_strength_per_alive_team"] = str_i / avg_alive_teams
        out["net_prox_degree_200m_per_alive_team"] = out["net_prox_degree_200m"] / avg_alive_teams
        out["net_copress_degree_per_alive_team"] = out["net_copress_degree"] / avg_alive_teams
    return out


# --------------------------------------------------------------------------------------
# the cut
# --------------------------------------------------------------------------------------
ZONE_COLUMNS = (
    "cur_cx",
    "cur_cy",
    "cur_r",
    "next_cx",
    "next_cy",
    "next_r",
    "phase_frac",
    "phase_no",
)
RAW_GLOBAL = (
    "phase_idx", "phase_duration_s", "avg_alive_teams", "avg_alive_players", "safe_radius_start_m",
    "safe_radius_end_m", "safe_radius_delta_m", "safe_center_shift_m", "poison_radius_start_m",
    "poison_radius_end_m", "zone_gap_start_m", "zone_gap_end_m", "team_size",
    "alive_players_start", "alive_players_end", "alive_players_min", "alive_players_avg",
)  # fmt: skip
HAB_GLOBAL = (
    "hab_safe_area_start_m2", "hab_safe_area_end_m2", "hab_safe_area_delta_m2", "hab_loss_ratio",
    "hab_shrink_rate_m2ps", "hab_alive_teams", "hab_safe_area_per_alive_team", "hab_crowding_pressure",
    "hab_center_shift_m", "hab_center_shift_per_radius", "hab_zone_gap_start_m", "hab_zone_gap_end_m",
    "hab_zone_gap_pressure",
)  # fmt: skip


def _div(a: float, b: float) -> float:
    return a / b if (np.isfinite(a) and np.isfinite(b) and b != 0) else float("nan")


@dataclass
class CutFeatures:
    k: int
    a_k: float
    s_k: float | None  # first LogPhaseChange of phase k (None: not yet visible at the cut)
    X: np.ndarray  # [N, F] float32
    node_alive: np.ndarray  # [N] bool: team causally alive at a_k (token / attention key mask)
    alive_counts: np.ndarray  # [N, 4] float64 (start, end, min, avg) member counts (before /4)
    edges: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]  # phase-k edges (i, j, w)
    team_pos: np.ndarray  # [N, 2] float32, NaN if none
    zone: np.ndarray  # [8] float32
    zone_valid: bool
    team_ai_frac: np.ndarray  # [N] float32
    diag: dict = field(default_factory=dict)


def team_ai_share(src: MatchSource) -> np.ndarray:
    """Share of the roster's participants whose account id starts with ``ai.`` (NaN without participants)."""
    out = np.full(src.n_teams, np.nan, dtype=np.float32)
    by: dict[int, list[str]] = {}
    for t, pid, _dt, _ts in src.roster:
        by.setdefault(int(t), []).append(str(pid))
    for i, tid in enumerate(src.team_ids):
        ids = by.get(int(tid))
        if ids:
            out[i] = sum(p.startswith("ai.") for p in ids) / len(ids)
    return out


def latest_zone(gs: dict[str, np.ndarray], kind: str) -> tuple[float, float, float] | None:
    """Latest game state with a valid zone of ``kind`` (the zone visible at the cut of a truncated source)."""
    idx = np.nonzero(np.isfinite(gs[f"{kind}_r"]))[0]
    if len(idx) == 0:
        return None
    j = int(idx[-1])
    return float(gs[f"{kind}_x"][j]), float(gs[f"{kind}_y"][j]), float(gs[f"{kind}_r"][j])


def compute_cut_features(src: MatchSource, k: int, a_k: float, *, map_size: float) -> CutFeatures:
    """Everything the model may see at the cut ``a_k`` of phase ``k`` (see the module docstring)."""
    src = truncate(src, a_k)  # from here on only records with w < a_k exist
    N = src.n_teams
    team_index = {int(t): i for i, t in enumerate(src.team_ids)}
    ph = phases_of(src)
    assert not any(j > k for j in ph.indices), "a later phase is visible before the cut"
    wins: dict[int, tuple[float, float]] = {}
    for j in ph.indices:
        end = (
            a_k if j == k else (ph.boundary(j) - CUT_LEAD_S if ph.boundary(j) is not None else a_k)
        )
        wins[j] = (ph.start(j), min(end, a_k))
    cur = wins.get(
        k
    )  # None: phase k has not started before its own cut (a phase shorter than the 10 s lead)
    s_k = None if cur is None else cur[0]
    sched = src.schedule.get(k)
    dur = float(sched) if sched is not None else None
    acct_idx = src.account_index()

    # ---- causal alive (rule 3) ----------------------------------------------------------------------
    tl = AL.build_timelines(
        src.kills["w"], src.kills["victim"], src.pos["w"], src.pos["acct"], src.pos["health"]
    )
    members = AL.team_members(src.roster, acct_idx)
    counts = np.zeros((N, 4))
    for i, tid in enumerate(src.team_ids):
        mem = members.get(int(tid), [])
        lo_c = cur[0] if cur is not None else a_k
        counts[i] = AL.alive_count_features(mem, tl, lo_c, a_k)
    node_alive = counts[:, 1] > 0

    # ---- grid / zone / distances for every visible phase ---------------------------------------------
    acct_team = EV.account_team_table(src, src.pos)
    tev = EV.build_team_events(src, acct_team)
    first_start = min((w[0] for w in wins.values()), default=None)
    if first_start is not None:
        grid = GR.build_team_grid(
            src.pos, team_index, GR.first_grid_time(first_start), GR.last_grid_time(a_k)
        )
    else:
        grid = GR.build_team_grid(src.pos, team_index, 1.0, 0.0)
    safe = GR.zone_at_grid(src.gs, grid.g, "safe")
    D = GR.distance_cube(grid)

    def grid_idx(lo: float, hi: float) -> np.ndarray:
        return np.nonzero((grid.g >= lo) & (grid.g + GR.HALF_WIN < hi))[0]

    edges_by_phase: dict[int, PhaseEdges] = {}
    for j, (lo, hi) in wins.items():
        edges_by_phase[j] = phase_edges(
            grid, grid_idx(lo, hi), safe, D, EV.window(tev, lo, hi), team_index
        )

    # ---- raw columns ---------------------------------------------------------------------------------
    raw: dict[str, np.ndarray] = {}
    raw["phase_idx"] = np.full(N, float(k))
    raw["phase_duration_s"] = np.full(N, np.nan if dur is None else dur)
    raw["team_size"] = src.team_num_players.astype(np.float64)
    raw["alive_players_start"], raw["alive_players_end"] = counts[:, 0], counts[:, 1]
    raw["alive_players_min"], raw["alive_players_avg"] = counts[:, 2], counts[:, 3]

    gb = global_block(src.gs, *cur) if cur is not None else GlobalBlock()
    ss, se, ps, pe = gb.safe_start, gb.safe_end, gb.poison_start, gb.poison_end
    nan = float("nan")
    r_ss = ss[2] if ss else nan
    r_se = se[2] if se else nan
    r_ps = ps[2] if ps else nan
    r_pe = pe[2] if pe else nan
    shift = float(math.hypot(ss[0] - se[0], ss[1] - se[1])) if (ss and se) else nan

    def bc(v: float) -> np.ndarray:
        return np.full(N, v)

    raw["avg_alive_teams"], raw["avg_alive_players"] = (
        bc(gb.avg_alive_teams),
        bc(gb.avg_alive_players),
    )
    raw["safe_radius_start_m"], raw["safe_radius_end_m"] = bc(r_ss), bc(r_se)
    raw["safe_radius_delta_m"] = bc(r_se - r_ss)
    raw["safe_center_shift_m"] = bc(shift)
    raw["poison_radius_start_m"], raw["poison_radius_end_m"] = bc(r_ps), bc(r_pe)
    raw["zone_gap_start_m"], raw["zone_gap_end_m"] = bc(r_ps - r_ss), bc(r_pe - r_se)

    # team state from raw positions
    if cur is not None:
        raw.update(state_features(src.pos, cur[0], cur[1], team_index))
    # combat / loot / trajectory / exposure / habitat team blocks / z-affordance
    target = latest_zone(src.gs, "poison")
    if cur is not None:
        lo, hi = cur
        raw.update(EV.combat_features(EV.window(tev, lo, hi), team_index))
        raw.update(EV.loot_features(src.items, lo, hi, team_index))
        tg = team_grid_features(
            grid,
            grid_idx(lo, hi),
            safe,
            D,
            dur,
            lo,
            tev,
            lo,
            hi,
            target,
            r_se if np.isfinite(r_se) else None,
            z_phase=(k <= Z_MAX_PHASE),
            team_ids=src.team_ids,
        )
        tg.pop("_has_rows", None)
        raw.update(tg)
    # network features
    raw.update(network_features(edges_by_phase, k, N, gb.avg_alive_teams, dur))
    # habitat global block
    pi = math.pi
    a_s, a_e = pi * r_ss**2, pi * r_se**2
    end_teams = gb.end_alive_teams
    raw["hab_safe_area_start_m2"], raw["hab_safe_area_end_m2"] = bc(a_s), bc(a_e)
    raw["hab_safe_area_delta_m2"] = bc(pi * (r_se**2 - r_ss**2))
    raw["hab_loss_ratio"] = bc(1.0 - _div(r_se**2, r_ss**2))
    raw["hab_shrink_rate_m2ps"] = bc(_div(pi * (r_ss**2 - r_se**2), dur if dur else nan))
    raw["hab_alive_teams"] = bc(end_teams)
    raw["hab_safe_area_per_alive_team"] = bc(
        a_e / max(end_teams, 1.0) if np.isfinite(end_teams) else a_e
    )
    raw["hab_crowding_pressure"] = bc(_div(end_teams, a_e))
    raw["hab_center_shift_m"] = bc(shift)
    raw["hab_center_shift_per_radius"] = bc(_div(shift, r_ss))
    raw["hab_zone_gap_start_m"], raw["hab_zone_gap_end_m"] = bc(r_ps - r_ss), bc(r_pe - r_se)
    raw["hab_zone_gap_pressure"] = bc(_div(r_pe - r_se, r_pe))
    if cur is not None:
        # z-affordance interactions need the habitat / exposure columns of the same cut
        edge_p = raw["hab_edge_pressure_ratio"]
        out_ratio = raw["outside_safe_ratio"]
        coalesce = np.where(
            np.isfinite(edge_p), edge_p, np.where(np.isfinite(out_ratio), out_ratio, 0.0)
        )
        raw["high_ground_x_boundary_pressure"] = raw["high_ground_contact_advantage"] * coalesce
        raw["z_percentile_x_hab_crowding_pressure"] = (
            raw["team_z_percentile_alive"] * raw["hab_crowding_pressure"]
        )
        co = np.where(
            np.isfinite(out_ratio),
            out_ratio,
            np.where(
                np.isfinite(raw["hab_outside_safe_ratio"]), raw["hab_outside_safe_ratio"], 0.0
            ),
        )
        raw["contact_delta_z_x_outside_safe_ratio"] = raw["contact_delta_z_mean_m"] * co
    X = scale_columns(raw, N)

    # ---- team position and zone at the cut ------------------------------------------------------------
    team_pos = np.full((N, 2), np.nan, dtype=np.float32)
    if cur is not None:
        gi = grid_idx(*cur)
        for i in range(N):
            rows = gi[grid.alive[gi, i]]
            if len(rows):
                r = int(rows[-1])
                team_pos[i] = (grid.cx[r, i] / map_size, grid.cy[r, i] / map_size)
    zone = np.full(len(ZONE_COLUMNS), np.nan, dtype=np.float32)
    blue, white = latest_zone(src.gs, "safe"), latest_zone(src.gs, "poison")
    if blue:
        zone[0:3] = np.asarray(blue, dtype=np.float64) / map_size
    if white:
        zone[3:6] = np.asarray(white, dtype=np.float64) / map_size
    if cur is not None and dur and dur > 0:
        zone[6] = float(np.clip((a_k - cur[0]) / dur, 0.0, 1.0))
    zone[7] = float(k)
    zone_valid = bool(np.isfinite(zone[:6]).all())

    return CutFeatures(
        k=k,
        a_k=float(a_k),
        s_k=s_k,
        X=X,
        node_alive=node_alive,
        alive_counts=counts,
        edges=edges_by_phase[k].as_triples() if k in edges_by_phase else PhaseEdges().as_triples(),
        team_pos=team_pos,
        zone=zone,
        zone_valid=zone_valid,
        team_ai_frac=team_ai_share(src),
        diag={
            "n_grid": int(grid.n_grid),
            "window": cur,
            "duration_s": dur,
            "schedule_source": src.schedule_source,
        },
    )
