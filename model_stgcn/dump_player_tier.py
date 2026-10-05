"""Per-cohort PLAYER-level tensor dumps (design stage 2, section 3) next to the phase dumps.

Writes ``<out-dir>/<cohort>/{train,val,test}.pt`` (+ ``dump_manifest.json``): the same 12 cohorts, splits and
matches (same order) as ``model_stgcn/dumps/bigcomp2027_phase_v1``.  The phase dump is the source of the
timing: team order, phase bounds, cut times, clock delta and zone circles are READ FROM IT, never recomputed, so
the two dumps always describe the same instants.  Everything else comes from the raw ``pubg.*`` telemetry
(read-only).  Training reads these files and never touches the DB.

    python -m model_stgcn.dump_player_tier --all --workers 2
    python -m model_stgcn.dump_player_tier --cohort erangel_tpp_squad --limit 20 --out-dir /tmp/player_smoke

Schema (one ``torch.save`` dict per split, loadable with ``weights_only=True``)::

    meta     cohort, split, created_at, git_commit, design_version, player_feature_names (Fp = 13),
             player_feature_groups / scales / notes, edge_types, clock, population, phase_dump (provenance),
             coverage (alive players with a position sample within 15 s of the cut), stats, skipped_matches
    matches  list of dicts (P = players of the match, N = teams of the phase dump, 8 phase slots like the phase dump)
             match_id, match_type (str, as in the phase dump), player_ids[P] (list of str),
             player_team_idx[P] int64 (index into the PHASE DUMP team_ids order),
             player_is_ai[P] bool, Xp[8,P,Fp] float32, player_node_mask[8,P] bool,
             edges_player {teammate | enemy_prox | combat: (idx[E,3] int64 = (p, i, j), w[E] float32)}
             extras: team_ids[N], player_alive_cut[8,P] / player_alive_cut_strict[8,P] / player_pos_valid[8,P] bool,
             player_death_s[P] (last death, +inf = alive at the end) / player_first_death_s[P] float32,
             cum_use_boost[8,P] float32 (Boost item uses since the match start), zone_cur_valid[8] / zone_next_valid[8] bool,
             phase_mask[8], phase_start_s[8], phase_end_s[8], cut_t[8], cut_wall_s[8], clock_delta_s, map_size

Conventions (also in ``meta['player_feature_notes']``):

* Phase axis ``p = phase_idx - 1``, exactly as in the phase dump.  The CUT of phase p is the phase dump's ``cut_t[p]``
  (a 10 s grid time on the game clock G, <= phase_end_s - delta - 10); ``cut_wall_s[p] = cut_t[p] + delta`` is the
  same instant on the LogMatchStart clock W (<= phase_end_s - 10).  A phase without a cut (too short) has
  ``Xp = 0``, no edges and ``player_pos_valid = False``; its node mask is still defined.
* ONE CLOCK.  Raw position rows are on G: the sample of a player at a cut is the LAST row with
  ``elapsed_time <= cut_t`` and ``cut_t - elapsed_time <= 15 s`` (``player_pos_valid`` False otherwise; samples
  come every ~10 s, so a missing sample means a gap or a player that stopped reporting).  Events (kills, knocks,
  damage, item use) are on W and are used up to ``cut_wall_s`` only; "in phase" = ``phase_start_s <= w <=
  cut_wall_s``, "cumulative" = everything up to ``cut_wall_s``.  Nothing after the cut is read.  delta is the
  drift-aware per-match value of stdb/causal_cut.py (``clock_delta_s`` of the phase dump).
* ``player_node_mask[p, i]`` = ``player_death_s[i] > phase_start_s[p]`` (alive at the phase start, event based):
  the label audit's per-player death time ``d`` = +inf if ``death_type = 'alive'``, else the LAST kill-event time
  (W), else ``time_survived`` + the match offset (model_stgcn.dump_phase_tier.member_death_times).  The max of
  ``d`` over a team is the phase dump's ``team_death_s``.  A revived / redeployed player (7 % of the alive
  player-cuts) therefore stays in the node mask between his first and his last death.
* ``player_alive_cut[p, i]`` (alive AT the cut, strictly causal, used for centroids and edges) = no kill event of
  the player at or before ``cut_wall_s[p]`` (``player_first_death_s > cut_wall_s``; players without a kill event
  use the ``time_survived`` death time), OR revived: his latest position sample (<= cut_t, fresh) has health > 0 and
  is newer than his last kill event before the cut.  ``player_alive_cut_strict`` is the first part alone (no death
  event yet).  Neither looks past the cut.
* Players = ``pubg.participants`` with a roster team that is in the phase dump's ``team_ids`` (AI accounts
  ``ai.*`` included, flagged by ``player_is_ai``).  Accounts that are not participants (``Monster.*``, ...) never
  become nodes, and events with such an account on the other side are ignored for the "dealt" / "kills" /
  "knocks" / combat features; damage TAKEN from them (zone has no attacker at all) still counts.
* ``Xp`` columns (``meta['player_feature_names']``):
    0 rel_x_hm, 1 rel_y_hm        position minus the team centroid, units of 100 m.  Centroid = mean of the
                                  event-alive team members that have a sample (all members with a sample if none
                                  is alive); a lone player has (0, 0).
    2 pos_x_norm, 3 pos_y_norm    position / map size (the phase dump's ``map_size``)
    4 health_norm                 sample health / 100 (0 = knocked down)
    5 dmg_dealt_hp100             damage to players of OTHER teams in the phase / 100 (matches the participants'
                                  ``damage_dealt`` stat: friendly and self damage are excluded)
    6 dmg_taken_hp100             damage taken from ANY source in the phase / 100 (enemies, blue zone, falls,
                                  vehicles, teammates; self-inflicted and zero-damage rows excluded)
    7 knocks_dealt                ``pubg.telem_groggy`` rows in the phase, attacker = player, victim of another team
    8 kills_phase                 ``pubg.telem_kills`` rows in the phase, killer = player, victim of another team
    9 kills_cum                   the same since the match start
   10 heals_cum                   ``pubg.telem_item_use`` rows since the match start with category 'Use' and
                                  sub_category 'Heal' (bandage, first aid, med kit): at the end of the match this
                                  is ``participants.heals``.  Boost items (energy drink, painkiller, adrenaline =
                                  ``participants.boosts``) are in the extra ``cum_use_boost``
   11 zone_cur_signed_dist_r      (r - distance to the centre) / r of the CURRENT blue zone (phase dump zone cols
                                  0-2): +1 at the centre, 0 on the boundary, < 0 outside; clipped to [-10, 1]
   12 zone_next_signed_dist_r     the same for the announced next circle (zone cols 3-5)
  Missing values (no sample, no zone circle) are 0.0; ``player_pos_valid`` / ``zone_cur_valid`` /
  ``zone_next_valid`` say which zeros are real.
* Edges (all at the cut; players need a valid sample and ``player_alive_cut``; indices (p, i, j) refer to the
  player axis; every pair is stored in both directions except ``combat``):
    teammate    same team, ``w = exp(-d / 200 m)``
    enemy_prox  different teams, ``d <= 300 m``, ``w = exp(-d / 200 m)``
    combat      attacker -> victim of ANOTHER team with damage, a kill or a knock in the phase window,
                ``w = log1p(damage + 100 * kills + 50 * knocks)``  (pairs need no position / alive condition)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model_stgcn import dump_phase_tier as D
from model_stgcn.provenance import git_provenance

logger = logging.getLogger("model_stgcn.dump_player_tier")

DEFAULT_PHASE_DIR = Path("model_stgcn/dumps/bigcomp2027_phase_v1")
DEFAULT_OUT_DIR = Path("model_stgcn/dumps/bigcomp2027_player_v1")
SPLITS = D.SPLITS
N_PHASES = D.N_PHASES  # 8

PLAYER_FEATURE_NAMES: list[str] = [
    "rel_x_hm", "rel_y_hm", "pos_x_norm", "pos_y_norm", "health_norm",
    "dmg_dealt_hp100", "dmg_taken_hp100", "knocks_dealt", "kills_phase", "kills_cum", "heals_cum",
    "zone_cur_signed_dist_r", "zone_next_signed_dist_r",
]  # fmt: skip
N_PLAYER_FEATURES = len(PLAYER_FEATURE_NAMES)  # 13
PLAYER_FEATURE_GROUPS: dict[str, list[int]] = {
    "position": [0, 1, 2, 3],
    "health": [4],
    "phase_combat": [5, 6, 7, 8],
    "cumulative": [9, 10],
    "zone": [11, 12],
}
PLAYER_FEATURE_SCALES: dict[str, str] = {
    "rel_x_hm": "metres / 100", "rel_y_hm": "metres / 100",
    "pos_x_norm": "metres / map_size", "pos_y_norm": "metres / map_size",
    "health_norm": "health / 100",
    "dmg_dealt_hp100": "HP / 100", "dmg_taken_hp100": "HP / 100",
    "knocks_dealt": "count", "kills_phase": "count", "kills_cum": "count", "heals_cum": "count",
    "zone_cur_signed_dist_r": "(r - d) / r, clipped to [-10, 1]", "zone_next_signed_dist_r": "(r - d) / r, clipped to [-10, 1]",
}  # fmt: skip
EDGE_TYPES = ("teammate", "enemy_prox", "combat")

POS_STALE_S = 15.0  # a position sample older than this (game clock) at the cut does not count
ENEMY_PROX_M = 300.0
EDGE_SCALE_M = 200.0
ZONE_CLIP = (-10.0, 1.0)
MIN_ZONE_RADIUS_M = 1.0
HEAL_SUB_CATEGORIES = ("Heal", "Boost")

NOTES = {
    "cut": "phase dump cut_t[p] (G grid time <= phase_end_s - delta - 10); cut_wall_s = cut_t + delta (W)",
    "positions": "last raw telem_positions row with cut_t - 15 s <= elapsed_time <= cut_t (game clock), cm / 100",
    "events": "W clock (event_time - started_at); in phase = phase_start_s <= w <= cut_wall_s; cumulative = w <= cut_wall_s",
    "node_mask": "event based: last kill-event death time (label audit rule) > phase_start_s",
    "alive_cut": "no kill event up to cut_wall_s, or revived (fresh health > 0 sample newer than the last kill event before the cut); "
                 "strict variant = no kill event yet",
    "dmg_dealt": "enemy-team victims only (= participants.damage_dealt); dmg_taken: all sources except self-inflicted",
    "knocks_kills": "attacker/killer = player, victim of another team; telem_groggy / telem_kills",
    "heals_cum": "telem_item_use category 'Use', sub_category Heal (= participants.heals at the match end); Boost uses in cum_use_boost",
    "zone": "phase dump zone circles at cut_t: current (kind safe) and announced next (kind poison)",
    "edges": "teammate / enemy_prox: alive + valid sample, both directions, exp(-d/200); enemy_prox d <= 300 m; "
             "combat: other-team attacker -> victim in the phase window, log1p(dmg + 100 kills + 50 knocks)",
    "exclusions": "Monster.* and every account that is not a participant of the match are never player nodes",
}


# --------------------------------------------------------------------------------------
# raw input of one match (pure containers; no DB handles)
# --------------------------------------------------------------------------------------
@dataclass
class PhaseInfo:
    """What the player dump takes from the phase dump record of the same match."""

    match_id: str
    team_ids: list[int]
    phase_mask: np.ndarray  # bool [8]
    phase_start_s: np.ndarray  # float64 [8] (W clock, NaN for absent phases)
    phase_end_s: np.ndarray  # float64 [8]
    cut_t: np.ndarray  # float64 [8] (G grid time, NaN if the phase has no cut)
    delta_s: float
    zone: np.ndarray  # float64 [8, 8] normalised by map_size (phase dump layout), NaN = missing
    map_size: float
    match_type: str = ""


def phase_info_from_record(m: dict) -> PhaseInfo:
    """Phase-dump match record -> :class:`PhaseInfo` (float64 copies)."""
    return PhaseInfo(
        match_id=str(m["match_id"]),
        team_ids=[int(t) for t in m["team_ids"].tolist()],
        phase_mask=m["phase_mask"].numpy().astype(bool).copy(),
        phase_start_s=m["phase_start_s"].numpy().astype(np.float64),
        phase_end_s=m["phase_end_s"].numpy().astype(np.float64),
        cut_t=m["cut_t"].numpy().astype(np.float64),
        delta_s=float(m["clock_delta_s"]),
        zone=m["zone"].numpy().astype(np.float64),
        map_size=float(m["map_size"]),
        match_type=str(m.get("match_type", "")),
    )


@dataclass
class RawPlayerMatch:
    """Raw telemetry of one match (plain python / numpy)."""

    info: PhaseInfo
    # (team_id, player_id, death_type, time_survived, last_kill_w) -- model_stgcn.dump_phase_tier.fetch_label_audit_inputs
    members: list[tuple]
    # account -> (G elapsed_time [n] ascending, health [n], x_m [n], y_m [n], event clock W of the row [n]) rows near the cuts
    positions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=dict)
    kills: list[tuple] = field(default_factory=list)  # (killer_id, victim_id, w)
    knocks: list[tuple] = field(default_factory=list)  # (attacker_id, victim_id, w)
    damage: list[tuple] = field(default_factory=list)  # (attacker_id | None, victim_id, damage, w)
    items: list[tuple] = field(default_factory=list)  # (account_id, sub_category, w)


# --------------------------------------------------------------------------------------
# pure assembly
# --------------------------------------------------------------------------------------
def _f32(x) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def _pair_edges(
    p: int, i_idx: np.ndarray, j_idx: np.ndarray, w: np.ndarray
) -> list[tuple[int, int, int, float]]:
    return [(p, int(i), int(j), float(x)) for i, j, x in zip(i_idx, j_idx, w)]


def _edges_to_tensors(rows: list[tuple[int, int, int, float]]) -> tuple[torch.Tensor, torch.Tensor]:
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    if not rows:
        return torch.zeros((0, 3), dtype=torch.int64), torch.zeros((0,), dtype=torch.float32)
    idx = torch.tensor([r[:3] for r in rows], dtype=torch.int64)
    w = torch.tensor([r[3] for r in rows], dtype=torch.float32)
    return idx, w


def _zone_feature(px: np.ndarray, py: np.ndarray, circle_m: np.ndarray) -> tuple[np.ndarray, bool]:
    """(r - d) / r for every player (clipped) and whether the circle exists."""
    cx, cy, r = (float(v) for v in circle_m)
    if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(r)) or r < MIN_ZONE_RADIUS_M:
        return np.zeros(len(px)), False
    dist = np.hypot(px - cx, py - cy)
    return np.clip((r - dist) / r, ZONE_CLIP[0], ZONE_CLIP[1]), True


def assemble_player_match(raw: RawPlayerMatch) -> dict[str, Any]:
    """RawPlayerMatch -> the per-match dict of the player dump (tensors).  Pure and DB-free."""
    info = raw.info
    N = len(info.team_ids)
    team_index = {tid: i for i, tid in enumerate(info.team_ids)}
    size = float(info.map_size)
    delta = float(info.delta_s)
    PH = N_PHASES

    # ---- players: participants whose roster team is in the phase dump, ordered by (team index, id) ----
    seen: set[str] = set()
    kept: list[tuple[int, str]] = []
    for tid, pid, _dt, _ts, _lk in raw.members:
        pid = str(pid)
        if tid in team_index and pid not in seen:
            seen.add(pid)
            kept.append((team_index[tid], pid))
    kept.sort()
    player_ids = [pid for _t, pid in kept]
    P = len(player_ids)
    pidx = {pid: i for i, pid in enumerate(player_ids)}
    team_of = np.array([t for t, _pid in kept], dtype=np.int64)
    is_ai = np.array([pid.startswith("ai.") for pid in player_ids], dtype=bool)

    # ---- event-based death times (same rule and member list as the phase dump's team_death_s) ----
    d_all, _offset = D.member_death_times(list(raw.members))
    d_by_pid: dict[str, float] = {}
    for (tid, pid, *_rest), d in zip(raw.members, d_all):
        if str(pid) in pidx and str(pid) not in d_by_pid:
            d_by_pid[str(pid)] = d
    death = np.array([d_by_pid[pid] for pid in player_ids], dtype=np.float64)

    node = np.zeros((PH, P), dtype=bool)
    for k in range(PH):
        if info.phase_mask[k]:
            node[k] = death > info.phase_start_s[k]

    # ---- events as index arrays (-1 = not a player node: Monster.*, non-participants, no attacker) ----
    def index_of(account) -> int:
        return pidx.get(str(account), -1) if account is not None else -1

    k_att = np.array([index_of(a) for a, _v, _w in raw.kills], dtype=np.int64)
    k_vic = np.array([index_of(v) for _a, v, _w in raw.kills], dtype=np.int64)
    k_w = np.array([w for _a, _v, w in raw.kills], dtype=np.float64)
    g_att = np.array([index_of(a) for a, _v, _w in raw.knocks], dtype=np.int64)
    g_vic = np.array([index_of(v) for _a, v, _w in raw.knocks], dtype=np.int64)
    g_w = np.array([w for _a, _v, w in raw.knocks], dtype=np.float64)
    m_att = np.array([index_of(a) for a, _v, _d, _w in raw.damage], dtype=np.int64)
    m_vic = np.array([index_of(v) for _a, v, _d, _w in raw.damage], dtype=np.int64)
    m_dmg = np.array([d for _a, _v, d, _w in raw.damage], dtype=np.float64)
    m_w = np.array([w for _a, _v, _d, w in raw.damage], dtype=np.float64)
    i_acc = np.array([index_of(a) for a, _s, _w in raw.items], dtype=np.int64)
    i_heal = np.array([s == "Heal" for _a, s, _w in raw.items], dtype=bool)
    i_boost = np.array([s == "Boost" for _a, s, _w in raw.items], dtype=bool)
    i_w = np.array([w for _a, _s, w in raw.items], dtype=np.float64)

    def enemy(a: np.ndarray, v: np.ndarray) -> np.ndarray:
        ok = (a >= 0) & (v >= 0)
        out = np.zeros(len(a), dtype=bool)
        out[ok] = team_of[a[ok]] != team_of[v[ok]]
        return out

    k_enemy, g_enemy, m_enemy = enemy(k_att, k_vic), enemy(g_att, g_vic), enemy(m_att, m_vic)
    m_self = (m_att == m_vic) & (m_att >= 0)

    # first death: the first kill event of the player as victim (any killer, also none); a player without a kill
    # event uses the fallback death time (time_survived based, +inf if alive) -- the same one as ``death``
    first_death = death.copy()
    has_kill_event = np.zeros(P, dtype=bool)
    ok_v = k_vic >= 0
    if ok_v.any():
        first = np.full(P, np.inf)
        np.minimum.at(first, k_vic[ok_v], k_w[ok_v])
        has_kill_event[np.unique(k_vic[ok_v])] = True
        first_death[has_kill_event] = first[has_kill_event]

    # ---- per phase ----
    Xp = np.zeros((PH, P, N_PLAYER_FEATURES), dtype=np.float32)
    alive_cut = np.zeros((PH, P), dtype=bool)
    alive_cut_strict = np.zeros((PH, P), dtype=bool)
    pos_valid = np.zeros((PH, P), dtype=bool)
    cum_boost = np.zeros((PH, P), dtype=np.float32)
    zone_cur_valid = np.zeros(PH, dtype=bool)
    zone_next_valid = np.zeros(PH, dtype=bool)
    cut_wall = np.full(PH, np.nan, dtype=np.float64)
    edge_rows: dict[str, list[tuple[int, int, int, float]]] = {name: [] for name in EDGE_TYPES}
    cov_alive = np.zeros(PH, dtype=np.int64)  # strict alive (no kill event yet) at the cut
    cov_valid = np.zeros(PH, dtype=np.int64)
    cov_alive_last = np.zeros(PH, dtype=np.int64)  # alive by the last-death rule (death_s > cut_wall_s)
    cov_valid_last = np.zeros(PH, dtype=np.int64)

    for k in range(PH):
        if not info.phase_mask[k] or not math.isfinite(info.cut_t[k]):
            continue
        T = float(info.cut_t[k])
        Tw = T + delta
        start = float(info.phase_start_s[k])
        cut_wall[k] = Tw

        # position sample at the cut: last row <= T within POS_STALE_S
        valid = np.zeros(P, dtype=bool)
        px = np.zeros(P)
        py = np.zeros(P)
        health = np.zeros(P)
        w_samp = np.full(P, -np.inf)
        for i, pid in enumerate(player_ids):
            rows = raw.positions.get(pid)
            if rows is None:
                continue
            t_arr = rows[0]
            j = int(np.searchsorted(t_arr, T, side="right")) - 1
            if j >= 0 and T - t_arr[j] <= POS_STALE_S:
                valid[i] = True
                health[i] = 0.0 if not math.isfinite(rows[1][j]) else rows[1][j]
                px[i], py[i] = rows[2][j], rows[3][j]
                w_samp[i] = rows[4][j]
        pos_valid[k] = valid

        # alive AT the cut (strictly causal): no kill event up to the cut, or revived (a fresh sample with health > 0
        # that is newer than the last kill event before the cut)
        strict = first_death > Tw
        last_kill = np.full(P, -np.inf)
        sel = ok_v & (k_w <= Tw)
        if sel.any():
            np.maximum.at(last_kill, k_vic[sel], k_w[sel])
        no_event = ~has_kill_event & (first_death <= Tw)  # dead by the time_survived fallback
        last_kill[no_event] = first_death[no_event]
        revived = valid & (health > 0) & (w_samp > last_kill)
        alive = strict | revived
        alive_cut[k], alive_cut_strict[k] = alive, strict
        last_rule = death > Tw
        cov_alive[k], cov_valid[k] = int(strict.sum()), int((strict & valid).sum())
        cov_alive_last[k], cov_valid_last[k] = int(last_rule.sum()), int((last_rule & valid).sum())

        # team centroid: alive members with a sample, else every member with a sample
        def centroid(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            n = np.bincount(team_of[mask], minlength=N).astype(np.float64)
            sx = np.bincount(team_of[mask], weights=px[mask], minlength=N)
            sy = np.bincount(team_of[mask], weights=py[mask], minlength=N)
            return n, sx, sy

        n_a, sx_a, sy_a = centroid(valid & alive)
        n_v, sx_v, sy_v = centroid(valid)
        use_alive = n_a > 0
        n_c = np.where(use_alive, n_a, n_v)
        cx = np.where(use_alive, sx_a, sx_v) / np.maximum(n_c, 1.0)
        cy = np.where(use_alive, sy_a, sy_v) / np.maximum(n_c, 1.0)

        cur_f, cur_ok = _zone_feature(px, py, info.zone[k, 0:3] * size)
        nxt_f, nxt_ok = _zone_feature(px, py, info.zone[k, 3:6] * size)
        zone_cur_valid[k], zone_next_valid[k] = cur_ok, nxt_ok

        in_win = lambda w: (w >= start) & (w <= Tw)  # noqa: E731 - tiny local helper
        # damage: dealt (other-team victims) and taken (any source except self-inflicted), both in the phase window
        mw = in_win(m_w) & (m_dmg > 0)
        dealt = np.bincount(m_att[mw & m_enemy], weights=m_dmg[mw & m_enemy], minlength=P)
        taken_sel = mw & (m_vic >= 0) & ~m_self
        taken = np.bincount(m_vic[taken_sel], weights=m_dmg[taken_sel], minlength=P)
        # knocks / kills (other-team victims)
        knocks = np.bincount(g_att[in_win(g_w) & g_enemy], minlength=P)
        kills_ph = np.bincount(k_att[in_win(k_w) & k_enemy], minlength=P)
        kills_cum = np.bincount(k_att[(k_w <= Tw) & k_enemy], minlength=P)
        heal_cum = np.bincount(i_acc[(i_w <= Tw) & (i_acc >= 0) & i_heal], minlength=P)
        boost_cum = np.bincount(i_acc[(i_w <= Tw) & (i_acc >= 0) & i_boost], minlength=P)
        cum_boost[k] = boost_cum

        feat = np.zeros((P, N_PLAYER_FEATURES))
        feat[valid, 0] = (px[valid] - cx[team_of[valid]]) / 100.0
        feat[valid, 1] = (py[valid] - cy[team_of[valid]]) / 100.0
        feat[valid, 2] = px[valid] / size
        feat[valid, 3] = py[valid] / size
        feat[valid, 4] = health[valid] / 100.0
        feat[:, 5] = dealt / 100.0
        feat[:, 6] = taken / 100.0
        feat[:, 7] = knocks
        feat[:, 8] = kills_ph
        feat[:, 9] = kills_cum
        feat[:, 10] = heal_cum
        feat[valid, 11] = cur_f[valid]
        feat[valid, 12] = nxt_f[valid]
        Xp[k] = feat.astype(np.float32)

        # ---- edges at the cut ----
        ids = np.nonzero(valid & alive)[0]
        if len(ids) >= 2:
            ii, jj = np.triu_indices(len(ids), k=1)
            a, b = ids[ii], ids[jj]
            d = np.hypot(px[a] - px[b], py[a] - py[b])
            w = np.exp(-d / EDGE_SCALE_M)
            same = team_of[a] == team_of[b]
            for name, sel in (("teammate", same), ("enemy_prox", ~same & (d <= ENEMY_PROX_M))):
                if sel.any():
                    edge_rows[name] += _pair_edges(k, a[sel], b[sel], w[sel])
                    edge_rows[name] += _pair_edges(k, b[sel], a[sel], w[sel])
        # combat: other-team attacker -> victim pairs of the phase window, [damage, kills, knocks] per pair
        pair: dict[tuple[int, int], list[float]] = {}
        for slot, rows_sel, att, vic, amount in (
            (0, mw & m_enemy, m_att, m_vic, m_dmg),
            (1, in_win(k_w) & k_enemy, k_att, k_vic, None),
            (2, in_win(g_w) & g_enemy, g_att, g_vic, None),
        ):
            for r in np.nonzero(rows_sel)[0]:
                rec = pair.setdefault((int(att[r]), int(vic[r])), [0.0, 0.0, 0.0])
                rec[slot] += 1.0 if amount is None else float(amount[r])
        edge_rows["combat"] += [
            (k, a_, v_, float(np.log1p(dmg + 100.0 * kl + 50.0 * kn))) for (a_, v_), (dmg, kl, kn) in pair.items()
        ]

    edges = {name: _edges_to_tensors(rows) for name, rows in edge_rows.items()}
    team_ids_t = torch.tensor(info.team_ids, dtype=torch.int64)
    return {
        "match_id": info.match_id,
        "match_type": info.match_type,
        "player_ids": player_ids,
        "player_team_idx": torch.from_numpy(team_of),
        "player_is_ai": torch.from_numpy(is_ai),
        "Xp": torch.from_numpy(Xp),
        "player_node_mask": torch.from_numpy(node),
        "edges_player": edges,
        # extras
        "team_ids": team_ids_t,
        "player_alive_cut": torch.from_numpy(alive_cut),
        "player_alive_cut_strict": torch.from_numpy(alive_cut_strict),
        "player_pos_valid": torch.from_numpy(pos_valid),
        "player_death_s": _f32(death),
        "player_first_death_s": _f32(first_death),
        "cum_use_boost": torch.from_numpy(cum_boost),
        "zone_cur_valid": torch.from_numpy(zone_cur_valid),
        "zone_next_valid": torch.from_numpy(zone_next_valid),
        "phase_mask": torch.from_numpy(info.phase_mask.copy()),
        "phase_start_s": _f32(info.phase_start_s),
        "phase_end_s": _f32(info.phase_end_s),
        "cut_t": _f32(info.cut_t),
        "cut_wall_s": _f32(cut_wall),
        "clock_delta_s": delta,
        "map_size": size,
        "_stats": {
            "n_players": P,
            "n_players_dropped": len({str(m[1]) for m in raw.members}) - P,
            "coverage_alive": cov_alive.tolist(),
            "coverage_valid": cov_valid.tolist(),
            "coverage_alive_last": cov_alive_last.tolist(),
            "coverage_valid_last": cov_valid_last.tolist(),
        },
    }


# --------------------------------------------------------------------------------------
# DB side (read-only)
# --------------------------------------------------------------------------------------
def fetch_raw_player_match(cur, info: PhaseInfo) -> RawPlayerMatch:
    """Raw telemetry of one match; reads ``pubg.*`` and ``stdb.matches`` only."""
    mid = info.match_id
    audit = D.fetch_label_audit_inputs(cur, mid)
    members = audit["members"]
    accounts = sorted({str(m[1]) for m in members})

    cuts = sorted({int(t) for t in info.cut_t[np.isfinite(info.cut_t)]})
    positions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    if cuts and accounts:
        windows = " OR ".join("(elapsed_time BETWEEN %s AND %s)" for _ in cuts)
        params: list[Any] = [mid, accounts]
        for t in cuts:
            params += [int(t - POS_STALE_S) - 1, int(t)]
        cur.execute(
            f"""
            SELECT account_id, elapsed_time, health::float8, pos_x::float8, pos_y::float8, extract(epoch FROM event_time)::float8
              FROM pubg.telem_positions
             WHERE match_id = %s AND account_id = ANY(%s)
               AND is_game >= 0.1 AND pos_x IS NOT NULL AND pos_y IS NOT NULL AND elapsed_time IS NOT NULL
               AND ({windows})
             ORDER BY account_id, elapsed_time, id
            """,
            params,
        )
        rows = cur.fetchall()
        cur.execute("SELECT extract(epoch FROM started_at)::float8 FROM stdb.matches WHERE match_id = %s", (mid,))
        s_ep = float(cur.fetchone()[0])
        buckets: dict[str, list] = {}
        for acc, t, h, x, y, ep in rows:
            buckets.setdefault(str(acc), []).append(
                (float(t), float("nan") if h is None else float(h), float(x) / 100.0, float(y) / 100.0, float(ep) - s_ep)
            )
        for acc, lst in buckets.items():
            arr = np.array(lst, dtype=np.float64)
            positions[acc] = (arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4])

    w_expr = "extract(epoch FROM (e.event_time - m.started_at))::float8"
    cur.execute(
        f"SELECT e.killer_id, e.victim_id, {w_expr} FROM pubg.telem_kills e JOIN stdb.matches m ON m.match_id = e.match_id "
        "WHERE e.match_id = %s AND e.victim_id IS NOT NULL",
        (mid,),
    )
    kills = [(a, v, float(w)) for a, v, w in cur.fetchall()]
    cur.execute(
        f"SELECT e.attacker_id, e.victim_id, {w_expr} FROM pubg.telem_groggy e JOIN stdb.matches m ON m.match_id = e.match_id "
        "WHERE e.match_id = %s AND e.victim_id IS NOT NULL",
        (mid,),
    )
    knocks = [(a, v, float(w)) for a, v, w in cur.fetchall()]
    cur.execute(
        f"SELECT e.attacker_id, e.victim_id, e.damage::float8, {w_expr} FROM pubg.telem_damage e JOIN stdb.matches m ON m.match_id = e.match_id "
        "WHERE e.match_id = %s AND e.victim_id IS NOT NULL AND e.damage > 0",
        (mid,),
    )
    damage = [(a, v, float(d), float(w)) for a, v, d, w in cur.fetchall()]
    cur.execute(
        f"SELECT e.account_id, e.sub_category, {w_expr} FROM pubg.telem_item_use e JOIN stdb.matches m ON m.match_id = e.match_id "
        "WHERE e.match_id = %s AND e.category = 'Use' AND e.sub_category = ANY(%s) AND e.account_id IS NOT NULL",
        (mid, list(HEAL_SUB_CATEGORIES)),
    )
    items = [(a, s, float(w)) for a, s, w in cur.fetchall()]
    return RawPlayerMatch(info=info, members=members, positions=positions, kills=kills, knocks=knocks, damage=damage, items=items)


# --------------------------------------------------------------------------------------
# validation against the phase dump (DB-free)
# --------------------------------------------------------------------------------------
def validate_player_match(pm: dict, phase_m: dict | None = None) -> None:
    """Raise ``ValueError`` if a player-dump match violates the schema; with the phase-dump record also checks
    the alignment (team order, event-based alive counts, team death times)."""
    where = f"match {pm.get('match_id')}"
    P = len(pm["player_ids"])
    PH, F = N_PHASES, N_PLAYER_FEATURES
    expect = {
        "player_team_idx": (P,), "player_is_ai": (P,), "Xp": (PH, P, F), "player_node_mask": (PH, P),
        "player_alive_cut": (PH, P), "player_alive_cut_strict": (PH, P), "player_pos_valid": (PH, P),
        "player_death_s": (P,), "player_first_death_s": (P,), "cum_use_boost": (PH, P),
    }  # fmt: skip
    for key, shape in expect.items():
        if tuple(pm[key].shape) != shape:
            raise ValueError(f"{where}: {key} must have shape {shape}, got {tuple(pm[key].shape)}")
    if not torch.isfinite(pm["Xp"]).all():
        raise ValueError(f"{where}: non-finite Xp")
    if P and (pm["player_team_idx"].min() < 0 or pm["player_team_idx"].max() >= len(pm["team_ids"])):
        raise ValueError(f"{where}: player_team_idx outside the team axis")
    if set(pm["edges_player"]) != set(EDGE_TYPES):
        raise ValueError(f"{where}: edge types {sorted(pm['edges_player'])}")
    for name, (idx, w) in pm["edges_player"].items():
        if idx.numel() == 0 and w.numel() == 0:
            continue
        if idx.dim() != 2 or idx.shape[1] != 3 or w.shape != (idx.shape[0],) or idx.dtype != torch.int64:
            raise ValueError(f"{where}: edges_player['{name}'] must be (idx[E,3] int64, w[E])")
        if (idx < 0).any() or (idx[:, 0] >= PH).any() or (idx[:, 1:] >= P).any():
            raise ValueError(f"{where}: edges_player['{name}'] index out of range")
        if not torch.isfinite(w).all() or (w < 0).any():
            raise ValueError(f"{where}: edges_player['{name}'] has a bad weight")
        if name != "combat" and ((idx[:, 1] == idx[:, 2]).any()):
            raise ValueError(f"{where}: self loop in '{name}'")
    for name in ("teammate", "enemy_prox"):  # symmetric: both directions stored
        idx, w = pm["edges_player"][name]
        fwd = {tuple(r) for r in idx.tolist()}
        if any((p, j, i) not in fwd for p, i, j in fwd):
            raise ValueError(f"{where}: '{name}' is not stored in both directions")
    if phase_m is None:
        return
    if pm["match_id"] != phase_m["match_id"] or not torch.equal(pm["team_ids"], phase_m["team_ids"]):
        raise ValueError(f"{where}: match / team order differs from the phase dump")
    if not torch.equal(pm["phase_mask"], phase_m["phase_mask"]):
        raise ValueError(f"{where}: phase_mask differs from the phase dump")
    for key in ("cut_t", "phase_start_s", "phase_end_s"):
        if not torch.equal(torch.nan_to_num(pm[key], nan=-1.0), torch.nan_to_num(phase_m[key], nan=-1.0)):
            raise ValueError(f"{where}: {key} differs from the phase dump")
    if abs(pm["clock_delta_s"] - float(phase_m["clock_delta_s"])) > 0.0 or abs(pm["map_size"] - float(phase_m["map_size"])) > 0.0:
        raise ValueError(f"{where}: clock_delta_s / map_size differ from the phase dump")
    # event-based alive counts and team death times must agree with the phase dump
    N = len(pm["team_ids"])
    team_idx = pm["player_team_idx"]
    d_team = torch.full((N,), float("nan"))
    for i in range(N):
        sel = team_idx == i
        if sel.any():
            d_team[i] = pm["player_death_s"][sel].max()
    ref = phase_m["team_death_s"]
    ok = torch.isnan(d_team) | (torch.isinf(d_team) & torch.isinf(ref)) | ((d_team - ref).abs() < 1e-2)
    if not bool(ok.all()):
        raise ValueError(f"{where}: max of the player death times differs from the phase dump's team_death_s")
    f_start = D.FEATURE_NAMES.index("alive_players_start_norm")
    X = phase_m["X"]
    for k in range(N_PHASES):
        if not bool(pm["phase_mask"][k]):
            continue
        counts = torch.bincount(team_idx[pm["player_node_mask"][k]], minlength=N).float() / D.ALIVE_SCALE
        have = torch.bincount(team_idx, minlength=N) > 0
        if not torch.allclose(counts[have], X[k, :, f_start][have], atol=1e-5):
            raise ValueError(f"{where}: phase {k + 1}: node-mask counts differ from alive_players_start_norm")


# --------------------------------------------------------------------------------------
# worker pool (<= 2 processes: the machine is shared)
# --------------------------------------------------------------------------------------
MAX_WORKERS = 2
_CONN = None


def _init_worker() -> None:
    global _CONN
    from stdb.conn import connect

    _CONN = connect()
    _CONN.set_session(readonly=True, autocommit=True)


def _work(info: PhaseInfo) -> tuple[str, dict | None, str | None]:
    try:
        with _CONN.cursor() as cur:
            raw = fetch_raw_player_match(cur, info)
        return info.match_id, assemble_player_match(raw), None
    except Exception as exc:  # pragma: no cover - reported per match
        return info.match_id, None, f"error: {exc!r}"


# --------------------------------------------------------------------------------------
# cohort driver
# --------------------------------------------------------------------------------------
def coverage_summary(items: list[dict]) -> dict[str, Any]:
    """Share of alive players at the cut that have a position sample within POS_STALE_S, per phase.

    ``strict`` (headline, the data profile's definition): players with no kill event up to the cut.
    ``last_death``: players whose LAST death is after the cut -- it also holds revived players that are dead at the
    cut, which have no samples, so it is lower."""
    out: dict[str, Any] = {
        "definition": f"alive players at the cut with a position sample <= {POS_STALE_S:.0f} s before it (game clock) / alive players at the cut",
    }
    for name, ka, kv in (("strict", "coverage_alive", "coverage_valid"), ("last_death", "coverage_alive_last", "coverage_valid_last")):
        alive = np.zeros(N_PHASES, dtype=np.int64)
        valid = np.zeros(N_PHASES, dtype=np.int64)
        for s_ in items:
            alive += np.array(s_[ka], dtype=np.int64)
            valid += np.array(s_[kv], dtype=np.int64)
        out[name] = {
            "alive_at_cut": alive.tolist(),
            "with_sample": valid.tolist(),
            "share_by_phase": [float(v / a) if a else None for a, v in zip(alive, valid)],
            "share_phases_1_7": float(valid[:7].sum() / max(alive[:7].sum(), 1)),
            "share_all_phases": float(valid.sum() / max(alive.sum(), 1)),
        }
    out["share_phases_1_7"] = out["strict"]["share_phases_1_7"]
    out["share_all_phases"] = out["strict"]["share_all_phases"]
    return out


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def dump_cohort(
    cohort: str,
    phase_dir: Path,
    out_dir: Path,
    *,
    workers: int,
    limit: int | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    workers = max(1, min(int(workers), MAX_WORKERS))
    cohort_dir = out_dir / cohort
    cohort_dir.mkdir(parents=True, exist_ok=True)
    prov = git_provenance()
    manifest: dict[str, Any] = {"cohort": cohort, "git_commit": prov.get("git_commit"), "splits": {}}
    ctx = mp.get_context("spawn")  # fork + libpq is unsafe on macOS
    with ctx.Pool(processes=workers, initializer=_init_worker) as pool:
        for split in SPLITS:
            t_split = time.time()
            phase_path = phase_dir / cohort / f"{split}.pt"
            phase = torch.load(phase_path, map_location="cpu", weights_only=True)
            pmeta = phase["meta"]
            matches_in = phase["matches"][:limit] if limit else phase["matches"]
            infos = [phase_info_from_record(m) for m in matches_in]
            by_id = {m["match_id"]: m for m in matches_in}
            results = pool.imap(_work, infos, chunksize=4)
            matches, problems, stat_items = [], {}, []
            n_dropped = 0
            for mid, item, err in results:
                if item is None:
                    problems[mid] = err
                    continue
                stats = item.pop("_stats")
                stat_items.append(stats)
                n_dropped += stats["n_players_dropped"]
                validate_player_match(item, by_id[mid])
                matches.append(item)
            if problems:
                logger.error("%s/%s: %d of %d matches failed, e.g. %s", cohort, split, len(problems), len(infos), list(problems.items())[:3])
            coverage = coverage_summary(stat_items)
            n_players = int(sum(s["n_players"] for s in stat_items))
            meta = {
                "cohort": cohort,
                "cohort_label": pmeta.get("cohort_label"),
                "map_name": pmeta.get("map_name"),
                "game_mode": pmeta.get("game_mode"),
                "split": split,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": prov.get("git_commit"),
                "git_dirty": bool(prov.get("git_dirty")),
                "design_version": "v2.3-stage2",
                "player_feature_names": list(PLAYER_FEATURE_NAMES),
                "player_feature_groups": {k: list(v) for k, v in PLAYER_FEATURE_GROUPS.items()},
                "player_feature_scales": dict(PLAYER_FEATURE_SCALES),
                "player_feature_notes": dict(NOTES),
                "n_player_features": N_PLAYER_FEATURES,
                "n_phases": N_PHASES,
                "edge_types": list(EDGE_TYPES),
                "pos_stale_s": POS_STALE_S,
                "enemy_prox_m": ENEMY_PROX_M,
                "edge_scale_m": EDGE_SCALE_M,
                "clock": {
                    **D.CLOCK_DOC,
                    "player_cut": "the phase dump's cut_t (G) / cut_wall_s = cut_t + clock_delta_s (W); positions are the last raw row "
                                  f"<= cut_t within {POS_STALE_S:.0f} s, events are used up to cut_wall_s",
                },
                "population": pmeta.get("population"),
                "phase_dump": {
                    "path": str(phase_path),
                    "created_at": pmeta.get("created_at"),
                    "git_commit": pmeta.get("git_commit"),
                    "split_ids_sha1": pmeta.get("split_ids_sha1"),
                    "n_matches": len(phase["matches"]),
                    "match_ids_sha1": hashlib.sha1("\n".join(m["match_id"] for m in phase["matches"]).encode("utf-8")).hexdigest(),
                },
                "n_matches": len(matches),
                "n_players": n_players,
                "n_players_without_phase_team": int(n_dropped),
                "coverage": coverage,
                "skipped_matches": problems,
            }
            path = cohort_dir / f"{split}.pt"
            tmp = cohort_dir / f".{split}.pt.tmp"
            torch.save({"meta": meta, "matches": matches}, tmp)
            os.replace(tmp, path)
            manifest["splits"][split] = {
                "file": path.name,
                "bytes": path.stat().st_size,
                "n_matches": len(matches),
                "n_skipped": len(problems),
                "n_players": n_players,
                "coverage_share_phases_1_7": coverage["share_phases_1_7"],
                "coverage": coverage,
                "elapsed_s": round(time.time() - t_split, 1),
            }
            logger.info(
                "%s/%s: %d/%d matches, %d players -> %s (%.1f MB), coverage(1-7) %.4f, %.0f s",
                cohort, split, len(matches), len(infos), n_players, path, path.stat().st_size / 1e6,
                coverage["share_phases_1_7"], time.time() - t_split,
            )
            del phase, matches, matches_in
    manifest["elapsed_s"] = round(time.time() - t0, 1)
    manifest["total_bytes"] = sum(s["bytes"] for s in manifest["splits"].values())
    _write_json(cohort_dir / "dump_manifest.json", manifest)
    return manifest


def validate_dump_dir(player_dir: Path, phase_dir: Path, cohorts: list[str] | None = None) -> dict[str, Any]:
    """Re-read the written player dumps and check every match against the phase dump of the same cohort / split:
    same matches in the same order, schema, index ranges, event-based alive counts and team death times."""
    summary: dict[str, Any] = {}
    for cohort in cohorts or phase_cohorts(player_dir):
        rec = {"matches": 0, "players": 0, "edges": {k: 0 for k in EDGE_TYPES}}
        for split in SPLITS:
            player = torch.load(player_dir / cohort / f"{split}.pt", map_location="cpu", weights_only=True)
            phase = torch.load(phase_dir / cohort / f"{split}.pt", map_location="cpu", weights_only=True)
            ids_p = [m["match_id"] for m in player["matches"]]
            ids_h = [m["match_id"] for m in phase["matches"]]
            if ids_p != ids_h:
                raise ValueError(f"{cohort}/{split}: the player dump has {len(ids_p)} matches, the phase dump {len(ids_h)}, or a different order")
            if player["meta"]["player_feature_names"] != PLAYER_FEATURE_NAMES:
                raise ValueError(f"{cohort}/{split}: feature names differ")
            for pm, hm in zip(player["matches"], phase["matches"]):
                validate_player_match(pm, hm)
                if pm["match_type"] != hm["match_type"]:
                    raise ValueError(f"{cohort}/{split}/{pm['match_id']}: match_type differs from the phase dump")
                rec["matches"] += 1
                rec["players"] += len(pm["player_ids"])
                for k in EDGE_TYPES:
                    rec["edges"][k] += int(pm["edges_player"][k][0].shape[0])
        summary[cohort] = rec
    return summary


def phase_cohorts(phase_dir: Path) -> list[str]:
    """Cohorts with a complete phase dump (the study population after the v2.3 exclusions)."""
    return sorted(
        d.name for d in phase_dir.iterdir() if d.is_dir() and all((d / f"{s}.pt").exists() for s in SPLITS)
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cohort", action="append", help="cohort slug (repeatable)")
    g.add_argument("--all", action="store_true", help="every cohort of the phase dump directory")
    ap.add_argument("--phase-dir", type=Path, default=DEFAULT_PHASE_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--workers", type=int, default=2, help=f"1..{MAX_WORKERS} (the Mac is shared)")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N matches per split")
    ap.add_argument("--validate-only", action="store_true", help="re-read the written dumps and check them against the phase dumps")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    torch.set_num_threads(1)  # the Mac is shared
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    if args.workers > MAX_WORKERS:
        logger.warning("capping --workers %d to %d", args.workers, MAX_WORKERS)
    cohorts = phase_cohorts(args.phase_dir) if args.all else list(args.cohort)
    if args.validate_only:
        summary = validate_dump_dir(args.out_dir, args.phase_dir, cohorts)
        for cohort, rec in summary.items():
            logger.info("%s: %d matches, %d players, edges %s -- valid", cohort, rec["matches"], rec["players"], rec["edges"])
        return 0
    missing = [c for c in cohorts if not all((args.phase_dir / c / f"{s}.pt").exists() for s in SPLITS)]
    if missing:
        raise SystemExit(f"no complete phase dump for {missing} in {args.phase_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pop = args.phase_dir / "population.json"
    if pop.exists():
        _write_json(args.out_dir / "population.json", json.loads(pop.read_text(encoding="utf-8")))
    failed = 0
    summary: dict[str, Any] = {}
    for cohort in cohorts:
        manifest = dump_cohort(cohort, args.phase_dir, args.out_dir, workers=args.workers, limit=args.limit)
        summary[cohort] = {
            "bytes": manifest["total_bytes"],
            "elapsed_s": manifest["elapsed_s"],
            "n_matches": {s: v["n_matches"] for s, v in manifest["splits"].items()},
        }
        failed += int(any(s["n_skipped"] for s in manifest["splits"].values()))
    _write_json(args.out_dir / "dump_summary.json", summary)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
