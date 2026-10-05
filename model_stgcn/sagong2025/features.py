"""The 8 in-game features of Sagong et al. (2025, Table 4) per unit (team) and phase from raw telemetry (DB-free core).

Everything is computed on ONE clock, the LogMatchStart clock W (``event_time - stdb.matches.started_at``), the clock
of ``phase_start_s`` / ``phase_end_s`` of the PTR dumps.  Telemetry rows carry their own ``event_time``, so position
samples are placed on W exactly (no game-clock ``delta`` is needed; ``stdb/causal_cut.py`` needs it only for the
resampled 10 s grid).  Phase ``k`` window = ``[phase_start_s(k), phase_end_s(k) - GUARD_S]`` (``GUARD_S`` = the
``stdb.causal_cut.GUARD_S`` = 10 s): a feature of phase ``k`` never reads a row later than ``phase_end_s(k) - 10 s``.

Features and the proxies used (our tables have no LogPlayerAttack events and no assist identities):

attack        number of damage events dealt to an enemy player (``telem_damage`` rows with attacker and victim
              both participants of different teams, damage > 0).  Proxy for the LogPlayerAttack count.
kill          kills credited to the player (``telem_kills.killer_id``) on an enemy victim, not a suicide.
              Over a whole match this reproduces ``participants.kills`` exactly (all 12 cohorts, every match).
damage        sum of those damage events (reproduces ``participants.damage_dealt`` to rounding, same check).
assist        proxy: for every enemy kill by someone else inside the phase window, each distinct other player who
              damaged that victim during the preceding ASSIST_LOOKBACK_S seconds (inside the window) gets +1.
              (The game's own ``assists`` is 0 in every solo match.  In squads ``telem_kills.assists_cnt`` agrees
              best with the killer's teammates that damaged the victim in the previous ~60 s: 81 % of enemy kills at
              60 s vs 69 % at 5 s, 30 Taego squad matches.)
distance      path length in metres: sum of the straight steps between consecutive position samples of the
              player, a step belonging to the window of its END sample (any movement: foot, vehicle, parachute).
density       max over the 10 s grid points of the window of the number of OTHER-TEAM players within
              DENSITY_RADIUS_M metres of the player (all participants incl. AI, knocked players included).
              A 10 s grid point takes each player's sample nearest to it, restricted to samples inside the window.
item          number of item pick-up events (``telem_item_pickup`` event types starting with ``pickup``: ground,
              care package, vehicle trunk; ``drop`` excluded).
carepackage   1 if the player picked up at least one item from a care package (``pickup_carepackage``) in the window.

Team adaptation (duo / squad; solo: unit = player, a one-member team): counts (attack, kill, damage, assist, item,
carepackage) are SUMMED over the roster members, density is the MAX over members, distance is the MEAN over the
members that have at least one position sample in the window (0 if none).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

FEATURES: tuple[str, ...] = ("attack", "kill", "damage", "assist", "distance", "density", "item", "carepackage")
FIGHT_FEATURES: tuple[str, ...] = ("attack", "kill", "damage", "assist")  # Eq. 1: FightScore = sum
ACTIVE_FEATURES: tuple[str, ...] = ("distance", "density", "item", "carepackage")  # Eq. 2: ActiveScore = sum
GUARD_S = 10.0  # == stdb.causal_cut.GUARD_S (checked in tests/test_sagong2025.py)
ASSIST_LOOKBACK_S = 60.0
DENSITY_RADIUS_M = 100.0
DENSITY_GRID_S = 10.0
PICKUP_PREFIX = "pickup"
CAREPACKAGE_EVENT = "pickup_carepackage"
OUT_COLUMNS: list[str] = ["team_id", "phase", *FEATURES, "n_members", "n_obs"]


@dataclass
class MatchRaw:
    """Raw telemetry of one match (plain DataFrames; epochs in seconds, positions in metres)."""

    match_id: str
    start_epoch: float  # stdb.matches.started_at (the LogMatchStart clock origin)
    team_of: dict  # account / player id -> roster team id (participants JOIN rosters)
    pos: pd.DataFrame  # acc, ep, x, y
    dmg: pd.DataFrame  # attacker, victim, damage, ep
    kill: pd.DataFrame  # killer, victim, is_suicide, ep
    pick: pd.DataFrame  # acc, event_type, ep
    official: dict = field(default_factory=dict)  # participants sums for QC: kills, damage_dealt


def _empty(cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: [] for c in cols})


def max_density_by_team(
    w: np.ndarray, x: np.ndarray, y: np.ndarray, acc: np.ndarray, team_j: np.ndarray, n_teams: int,
    *, grid_s: float = DENSITY_GRID_S, radius_m: float = DENSITY_RADIUS_M,
) -> np.ndarray:
    """``[n_teams]`` max over grid points and members of the number of other-team samples within ``radius_m``.

    ``w, x, y, acc, team_j`` describe the position samples of one window.  Grid point ``g = round(w / grid_s)``;
    a player contributes its sample nearest to ``g * grid_s`` (ties by order); players are neighbours only if they
    belong to different teams.
    """
    out = np.zeros(n_teams, dtype=np.float64)
    if len(w) < 2:
        return out
    b = np.rint(w / grid_s).astype(np.int64)
    dist = np.abs(w - b * grid_s)
    order = np.lexsort((dist, acc, b))
    b, acc, x, y, team_j = b[order], acc[order], x[order], y[order], team_j[order]
    first = np.ones(len(b), dtype=bool)
    first[1:] = (b[1:] != b[:-1]) | (acc[1:] != acc[:-1])
    b, x, y, team_j = b[first], x[first], y[first], team_j[first]
    cuts = np.flatnonzero(np.r_[True, b[1:] != b[:-1], True])
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if hi - lo < 2:
            continue
        xy = np.column_stack([x[lo:hi], y[lo:hi]])
        tj = team_j[lo:hi]
        pairs = cKDTree(xy).query_pairs(radius_m, output_type="ndarray")
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            diff = tj[i] != tj[j]
            i, j = i[diff], j[diff]
            cnt = np.bincount(np.r_[i, j], minlength=hi - lo).astype(np.float64)
            np.maximum.at(out, tj, cnt)
    return out


def compute_match_features(
    raw: MatchRaw,
    windows: Mapping[int, tuple[float, float]],
    *,
    assist_lookback_s: float = ASSIST_LOOKBACK_S,
    radius_m: float = DENSITY_RADIUS_M,
) -> pd.DataFrame:
    """One row per (team, phase) with the 8 team-level features; ``windows[k] = (phase_start_s, phase_end_s)`` on W.

    Rows exist for every roster team of the match (dead teams get zeros); ``n_members`` = roster size,
    ``n_obs`` = members with at least one position sample inside the window.
    """
    accs = sorted(raw.team_of)
    a_idx = {a: i for i, a in enumerate(accs)}
    n_acc = len(accs)
    a_team = np.array([raw.team_of[a] for a in accs], dtype=np.int64)
    teams = np.unique(a_team)
    team_pos = {int(t): j for j, t in enumerate(teams)}
    a_tj = np.array([team_pos[int(t)] for t in a_team], dtype=np.int64)
    n_teams = len(teams)
    n_members = np.bincount(a_tj, minlength=n_teams).astype(np.float64)

    # ---- positions (W clock, metres), sorted per player ----
    pos = raw.pos
    pa = pos["acc"].map(a_idx)
    ok = pa.notna().to_numpy()
    pa = pa.to_numpy()[ok].astype(np.int64)
    pw = pos["ep"].to_numpy(dtype=np.float64)[ok] - raw.start_epoch
    px = pos["x"].to_numpy(dtype=np.float64)[ok]
    py = pos["y"].to_numpy(dtype=np.float64)[ok]
    o = np.lexsort((pw, pa))
    pa, pw, px, py = pa[o], pw[o], px[o], py[o]
    step = np.zeros(len(pa))
    if len(pa) > 1:
        same = pa[1:] == pa[:-1]
        step[1:] = np.where(same, np.hypot(px[1:] - px[:-1], py[1:] - py[:-1]), 0.0)
    ptj = a_tj[pa] if len(pa) else np.zeros(0, dtype=np.int64)

    # ---- enemy damage events ----
    dm = raw.dmg
    da, dv = dm["attacker"].map(a_idx), dm["victim"].map(a_idx)
    ok = (da.notna() & dv.notna()).to_numpy()
    da = da.to_numpy()[ok].astype(np.int64)
    dv = dv.to_numpy()[ok].astype(np.int64)
    dw = dm["ep"].to_numpy(dtype=np.float64)[ok] - raw.start_epoch
    dd = dm["damage"].to_numpy(dtype=np.float64)[ok]
    keep = (a_tj[da] != a_tj[dv]) & (dd > 0) & (da != dv)
    da, dv, dw, dd = da[keep], dv[keep], dw[keep], dd[keep]
    o = np.lexsort((dw, dv))  # sorted by (victim, time) for the assist look-back
    sdw, sdv, sda = dw[o], dv[o], da[o]
    v_lo = np.searchsorted(sdv, np.arange(n_acc), side="left")
    v_hi = np.searchsorted(sdv, np.arange(n_acc), side="right")

    # ---- enemy kills ----
    kl = raw.kill
    ka, kv = kl["killer"].map(a_idx), kl["victim"].map(a_idx)
    ok = (ka.notna() & kv.notna()).to_numpy() & ~kl["is_suicide"].fillna(False).to_numpy(dtype=bool)
    ka = ka.to_numpy()[ok].astype(np.int64)
    kv = kv.to_numpy()[ok].astype(np.int64)
    kw = kl["ep"].to_numpy(dtype=np.float64)[ok] - raw.start_epoch
    keep = a_tj[ka] != a_tj[kv]
    ka, kv, kw = ka[keep], kv[keep], kw[keep]

    # ---- item pick-ups ----
    pk = raw.pick
    pi = pk["acc"].map(a_idx)
    ok = pi.notna().to_numpy() & pk["event_type"].astype(str).str.startswith(PICKUP_PREFIX).to_numpy()
    pi = pi.to_numpy()[ok].astype(np.int64)
    pkw = pk["ep"].to_numpy(dtype=np.float64)[ok] - raw.start_epoch
    pk_cp = (pk["event_type"].to_numpy()[ok] == CAREPACKAGE_EVENT)

    frames = []
    for k in sorted(windows):
        s, e = windows[k]
        if not (np.isfinite(s) and np.isfinite(e)):
            continue
        c = float(e) - GUARD_S  # causal cut of the phase
        zero = np.zeros(n_acc)
        att, kil, dam, ast, dis, itm, cpk = (zero.copy() for _ in range(7))
        den_team = np.zeros(n_teams)
        obs = np.zeros(n_acc)
        if c > s:
            mk = (dw >= s) & (dw <= c)
            att = np.bincount(da[mk], minlength=n_acc).astype(np.float64)
            dam = np.bincount(da[mk], weights=dd[mk], minlength=n_acc)
            mk = (kw >= s) & (kw <= c)
            kil = np.bincount(ka[mk], minlength=n_acc).astype(np.float64)
            for killer, victim, t in zip(ka[mk], kv[mk], kw[mk]):  # assist proxy
                lo, hi = v_lo[victim], v_hi[victim]
                if lo == hi:
                    continue
                seg_t = sdw[lo:hi]
                a0 = lo + np.searchsorted(seg_t, max(s, t - assist_lookback_s), side="left")
                a1 = lo + np.searchsorted(seg_t, t, side="right")
                if a1 > a0:
                    helpers = np.unique(sda[a0:a1])
                    helpers = helpers[helpers != killer]
                    ast[helpers] += 1.0
            mk = (pkw >= s) & (pkw <= c)
            itm = np.bincount(pi[mk], minlength=n_acc).astype(np.float64)
            cpk = (np.bincount(pi[mk & pk_cp], minlength=n_acc) > 0).astype(np.float64)
            mk = (pw >= s) & (pw <= c)
            if mk.any():
                obs = np.bincount(pa[mk], minlength=n_acc).astype(np.float64)
                dis = np.bincount(pa[mk], weights=step[mk], minlength=n_acc)
                den_team = max_density_by_team(pw[mk], px[mk], py[mk], pa[mk], ptj[mk], n_teams, radius_m=radius_m)
        seen = obs > 0
        n_obs = np.bincount(a_tj[seen], minlength=n_teams).astype(np.float64)
        dist_sum = np.bincount(a_tj[seen], weights=dis[seen], minlength=n_teams)
        frames.append(
            pd.DataFrame(
                {
                    "team_id": teams,
                    "phase": int(k),
                    "attack": np.bincount(a_tj, weights=att, minlength=n_teams),
                    "kill": np.bincount(a_tj, weights=kil, minlength=n_teams),
                    "damage": np.bincount(a_tj, weights=dam, minlength=n_teams),
                    "assist": np.bincount(a_tj, weights=ast, minlength=n_teams),
                    "distance": np.divide(dist_sum, n_obs, out=np.zeros(n_teams), where=n_obs > 0),
                    "density": den_team,
                    "item": np.bincount(a_tj, weights=itm, minlength=n_teams),
                    "carepackage": np.bincount(a_tj, weights=cpk, minlength=n_teams),
                    "n_members": n_members,
                    "n_obs": n_obs,
                }
            )
        )
    if not frames:
        return _empty(OUT_COLUMNS)
    return pd.concat(frames, ignore_index=True)[OUT_COLUMNS]


def whole_match_totals(raw: MatchRaw) -> dict:
    """QC: enemy attack / damage / kill totals over the whole match (all time) next to the ``participants`` sums.

    The damage and kill rules are the ones of :func:`compute_match_features`; over a whole match they should
    reproduce ``participants.damage_dealt`` and ``participants.kills`` (checked on the extraction output).
    """
    known = set(raw.team_of)
    dm = raw.dmg
    m = dm["attacker"].isin(known) & dm["victim"].isin(known) & (dm["damage"] > 0) & (dm["attacker"] != dm["victim"])
    d = dm[m]
    d = d[d["attacker"].map(raw.team_of).to_numpy() != d["victim"].map(raw.team_of).to_numpy()]
    kl = raw.kill
    m = kl["killer"].isin(known) & kl["victim"].isin(known) & ~kl["is_suicide"].fillna(False).astype(bool)
    k = kl[m]
    k = k[k["killer"].map(raw.team_of).to_numpy() != k["victim"].map(raw.team_of).to_numpy()]
    return {
        "events_attacks": int(len(d)),
        "events_damage": float(d["damage"].sum()),
        "events_kills": int(len(k)),
        "official_kills": float(raw.official.get("kills", np.nan)),
        "official_damage": float(raw.official.get("damage_dealt", np.nan)),
    }
