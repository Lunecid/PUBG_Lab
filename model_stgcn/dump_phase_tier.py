"""Per-cohort tensor dumps of the phase tier (design v2.3: one clock + Fix B cap) for the PTR study.

Writes ``<out-dir>/<cohort>/{train,val,test}.pt`` (+ ``dump_manifest.json``), one
``torch.save`` dict per split.  Training reads these files and never touches the DB.

    python -m model_stgcn.dump_phase_tier --cohort erangel_tpp_squad
    python -m model_stgcn.dump_phase_tier --all --out-dir model_stgcn/dumps/bigcomp2027_phase_v1

Population (design v2.3 section 4.2): matches with ``pubg.matches.match_type = 'airoyale'`` and the
Vikendi cohort are excluded, the stored split assignment is kept for the remaining matches, and a
cohort left with fewer than 200 matches is dropped.  ``meta['population']`` carries the per-cohort
before / after counts.

Schema (consumed by ``model_stgcn/ptr/data.py``)::

    meta     cohort, split, created_at, git_commit, fixb=True, feature_names (F=150, column order
             of X), feature_groups {name: [col]} (incl. 'zone' and 'habitat'), zone_columns,
             replaced_features, population, label_agreement, clock, conventions, ...
    matches  list of per-match dicts (P = 8 phases, N = N_m teams in ascending team_id order)
             match_id, team_ids[N] int64, X[P,N,F] float32, node_mask[P,N] bool,
             target_mask[P,N] bool, y_hazard[P,N] float32 (the pipeline's labels, unchanged),
             final_rank_norm[N] / top5[N] / top10[N] float32,
             edges {prox|combat|copress: (idx[E,3] int64 (p,i,j), w[E] float32)},
             team_pos[P,N,2] float32, zone[P,8] float32, zone_valid[P] bool,
             audit extras: map_size, phase_mask[P], phase_start_s[P], phase_end_s[P], cut_t[P],
             cut_wall_s[P], clock_delta_s, clock_delta_median_s, node_mask_label[P,N] bool, last_phase_idx,
             and the labels v2 / audit fields:
             y_hazard_v2[P,N] float32, target_mask_v2[P,N] bool, node_mask_v2[P,N] bool,
             team_death_s[N] float32 (+inf = alive at match end, NaN = unknown), team_won[N] bool,
             team_ai_frac[N] float32, team_has_monster_dmg[N] float32, match_type (str).

Conventions (all also recorded in ``meta['conventions']`` / ``meta['clock']``):

* Phase axis ``p = phase_idx - 1`` for phases 1..8.  Phases a match does not have stay zero /
  False (X = 0, node_mask False, zone NaN, zone_valid False).
* ``X`` = ``stdb.queries._fetch_phase_node_features(include_v2_features=True)``, i.e. exactly the
  loader used by every existing model: per-column fixed-constant scaling, ``log1p`` for
  sinuosity, bool -> {0,1}, and NULL / NaN / inf -> 0.0 (the loader's only missing-value rule;
  there is no missing-indicator).  Column order = ``PHASE_NODE_FEATURE_NAMES +
  PHASE_NODE_FEATURE_NAMES_V2`` (66 + 84).  EXCEPT the four features derived from
  ``team_phase_state.alive_players_*`` (``meta['replaced_features']``): they are recomputed from
  kill events (members whose last death time is after the boundary, see ``alive_counts``), same names,
  same /4 scaling.
* ONE CLOCK.  ``phase_start_s`` / ``phase_end_s`` are numbers on the LogMatchStart clock W
  (``event_time - started_at``) because ``telem_phase_changes.elapsed_time`` is NULL; the 10 s grid, the
  raw positions and the game states are on the game clock G (``elapsed_time``), ``W = G + delta`` with a
  per-match ``delta``.  ``clock_delta_median_s`` is the audit's estimate (median of
  ``event_time - started_at - elapsed_time`` over the game states); ``clock_delta_s`` is the value the
  ETLs and the dump use: the median, RAISED (in 0.25 s steps) in the ~6 % of the matches whose clock drifts
  (offset W - G growing up to ~45 s over the match) until no phase window reads past the first
  LogPhaseChange of the next phase (stdb/causal_cut.py).  Every feature window of the ETL was moved onto
  one clock before the cap: ``[phase_start_s - delta, phase_end_s - delta - 10]`` on G.  Kill / damage
  events, ``team_death_s`` and ``phase_*_s`` are on W.  ``cut_t[p]`` is the last G grid point inside that
  window and ``cut_wall_s[p] = cut_t[p] + delta`` its W time (<= phase_end_s - 10).
* ``node_mask[p, i]`` = label-table ``alive_at_phase_start``; ``target_mask[p, i]`` =
  ``target_defined`` for cut phases 1..7 (phase 8 is never a target); ``y_hazard`` =
  ``hazard_next`` (0 outside the target mask): the pipeline's labels, NOT changed.  A team that is
  alive at the END of phase p is by definition alive at its start: the rows where the label table
  disagrees (alive_at_end but not alive_at_start) get ``node_mask`` raised so that ``target_mask``
  stays a subset of ``node_mask``; ``node_mask_label`` keeps the unraised flag.
* LABELS v2 (label audit 2026-10-05).  Per member the death time ``d`` on W is +inf if
  ``death_type = 'alive'``, else the member's LAST kill event time ``max(event_time) - started_at``,
  else ``time_survived`` + the match's median (kill event - time_survived) over human members
  (0.70 s if there is none).  ``team_death_s = max d`` over the members (+inf if any member is alive at
  the end = winning team).  ``target_mask_v2[k] = team_death_s > phase_end_s[k]`` and phase k+1 exists;
  ``y_hazard_v2[k] = isfinite(team_death_s) and team_death_s <= phase_end_s[k+1]``, with
  ``phase_end_s[k+1] := +inf`` (match end) when k+1 is the match's last phase (``last_phase_idx``);
  ``node_mask_v2[k] = team_death_s > phase_start_s[k]``.  The v2 arrays use the dump's phase axis (cuts
  1..7 only).
* ``final_rank_norm = final_rank / n_teams`` (rank 1 = winner, so SMALL is a GOOD placement);
  ``top5`` / ``top10`` = 1.0 iff final_rank <= 5 / 10.
* ``edges`` are the raw ``stdb.team_phase_edge_cache`` rows of the (one-clock, capped) cache before any
  row-normalisation or self-loop: ``prox`` (``exp(-mean distance/200 m)``, stored in both
  directions), ``combat`` (DIRECTED attacker -> victim, ``ln(1 + damage + 100*kills +
  50*groggy)``, exact event timestamps), ``copress`` (both directions).
* ``team_pos[p, i]`` = team centroid (x, y) / map size at the latest alive grid snapshot
  ``t <= cut_t[p]`` inside the window of phase p (``t >= phase_start_s - delta``).  Normally that is the
  grid point ``cut_t[p]`` itself; a team with no row there (dead, or a telemetry gap) falls back to its
  last earlier alive snapshot of the phase; NaN if it has none.  Source ``stdb.team_snapshot_features``.
* ``team_won`` = ``stdb.teams.won``; ``team_ai_frac`` = share of the roster's participants whose
  ``player_id`` starts with ``ai.`` (NaN without participants); ``team_has_monster_dmg`` = total damage
  (not a flag) the roster's members took from attackers whose id starts with ``Monster.`` (0.0 if
  none).
* ``zone[p] = [cur_cx, cur_cy, cur_r, next_cx, next_cy, next_r, phase_frac, phase_no]``: current blue
  zone (``dynamic_regions`` kind ``safe``) and the announced white circle (kind ``poison``) at
  ``cut_t[p]``, centres / radii divided by the map size; ``phase_frac = (cut_wall_s - phase_start_s) /
  (phase_end_s - phase_start_s)`` clipped to [0, 1] (both on W); ``phase_no = phase_idx`` (1-based, NOT
  normalised).  Missing numbers are NaN; ``zone_valid[p]`` is True iff all six geometry values exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from model_stgcn.provenance import git_provenance
from model_stgcn.splits import load_or_create_split
from model_stgcn.v3 import spec as feature_spec
from model_stgcn.v3.cohorts import REQUIRED_COHORTS, CohortSpec, resolve_cohort
from stdb import causal_cut
from stdb.queries import (
    PHASE_NODE_FEATURE_NAMES,
    PHASE_NODE_FEATURE_NAMES_V2,
    PHASE_T_MAX,
)

logger = logging.getLogger("model_stgcn.dump_phase_tier")

DEFAULT_OUT_DIR = Path("model_stgcn/dumps/bigcomp2027_phase_v1")
SPLIT_ROOT = Path("model_stgcn/checkpoints/br_om_ladder_required_v1")
SPLITS = ("train", "val", "test")
SEED = 42

# design v2.3 population rules
EXCLUDED_MATCH_TYPES: tuple[str, ...] = ("airoyale",)
EXCLUDED_COHORTS: tuple[str, ...] = ("vikendi_tpp_squad",)
MIN_COHORT_MATCHES = 200

N_PHASES = PHASE_T_MAX  # 8
FEATURE_NAMES: list[str] = list(PHASE_NODE_FEATURE_NAMES) + list(PHASE_NODE_FEATURE_NAMES_V2)
N_FEATURES = len(FEATURE_NAMES)  # 150

# The four features derived from team_phase_state.alive_players_* (position-based, single-snapshot rule
# that the label audit shows to be wrong for ~6 % of rows); recomputed from kill events.
REPLACED_FEATURES: tuple[str, ...] = (
    "alive_players_start_norm",
    "alive_players_end_norm",
    "alive_players_min_norm",
    "alive_players_avg_norm",
)
ALIVE_SCALE = 4.0  # the loader's fixed constant for every alive_players_*_norm column
FALLBACK_OFFSET_S = 0.70  # label audit: median (kill event - time_survived) of human players

# Same names / order as model_stgcn.ptr.data.ZONE_COLUMNS
ZONE_COLUMNS: tuple[str, ...] = (
    "cur_cx", "cur_cy", "cur_r", "next_cx", "next_cy", "next_r", "phase_frac", "phase_no",
)  # fmt: skip

# edge_cache edge_type -> dump key
EDGE_KEYS = {"proximity": "prox", "combat": "combat", "copressure": "copress"}

# Square side in metres (telemetry cm / 100), nominal values.  Verified against the data by
# ``verify_map_sizes`` (the initial safe circle is centred at ~size / 2; Sanhok and Karakin are
# 0.8 % below nominal, everything else exact; in-play team centroids stay inside [0, size]).
MAP_SIZE_M: dict[str, float] = {
    "Baltic_Main": 8160.0,      # Erangel
    "Tiger_Main": 8160.0,       # Taego
    "Desert_Main": 8160.0,      # Miramar / Desert
    "Neon_Main": 8160.0,        # Rondo
    "DihorOtok_Main": 8160.0,   # Vikendi (the data says 8 km, not the classic 6 km)
    "Savage_Main": 4080.0,      # Sanhok
    "Chimera_Main": 3060.0,     # Paramo
    "Summerland_Main": 2040.0,  # Karakin
}

CLOCK_DOC: dict[str, str] = {
    "W": "LogMatchStart clock = event_time - started_at (seconds).  phase_start_s / phase_end_s, kill and "
         "damage events, item events and team_death_s are on W.  phase_end_s is the stored "
         "v_match_phase_bounds number: the last 10 s grid point strictly before the first phase-(k+1) "
         "LogPhaseChange, i.e. 0-10 s before the true boundary.",
    "G": "game clock = elapsed_time of LogPlayerPosition / LogGameStatePeriodic.  The 10 s grid (snapshot_t), "
         "raw position windows and game states are on G.  W = G + delta, delta = per-match median of "
         "event_time - started_at - elapsed_time over the game states (clock_delta_median_s; median 7-8 s, p95 "
         "20 s, max 42-75 s).  In matches whose clock drifts (offset W - G growing steadily) the median would end "
         "the late windows after the true phase boundary; clock_delta_s = the median raised in 0.25 s steps until "
         "every phase window ends before the next phase starts (grid horizon >= 1 s, latest game state > 0.05 s).",
    "why": "telem_phase_changes.elapsed_time is NULL, so stdb.etl tagged the G grid with W boundary numbers.",
    "feature_windows": "every grid / position window of the phase-tier ETLs is [phase_start_s - delta, "
                       "phase_end_s - delta - 10] on G (one clock, then the 10 s cap); game-state aggregates "
                       "and event windows are shifted onto the same clock but not capped (exact timestamps).  "
                       "In matches whose delta was raised the window also starts earlier by the raise "
                       "(older data of the same match, never look-ahead).",
    "cut_t": "G grid time of the cut: last grid point <= phase_end_s - delta - 10.  cut_wall_s = cut_t + delta "
             "(<= phase_end_s - 10) is the same instant on W.",
    "team_death_s": "W clock, directly comparable with phase_end_s (the labels v2 rule).",
}


# --------------------------------------------------------------------------------------
# feature groups (pure)
# --------------------------------------------------------------------------------------
def build_feature_groups(names: list[str] | None = None) -> dict[str, list[int]]:
    """``{group: [column index]}`` from model_stgcn/v3/spec.py (groups may overlap).

    ``zone`` + ``habitat`` are the "hand-crafted zone and habitat families" of the design (what the
    *-noHab arms remove): ``habitat`` = the ``hab_*`` columns of spec.py, ``zone`` = the O2 pressure
    blocks of spec.py (OM3 increment).  Design v2.2: "noHab removes spec's zone and habitat groups but
    keeps the alive-count variables, which are not spatial": the three alive-count columns that spec.py
    files under those groups (``avg_alive_teams_norm``, ``avg_alive_players_norm`` in O2 and
    ``hab_alive_teams`` in habitat) are therefore moved to the separate ``alive_counts`` group together
    with the four ``alive_players_*_norm`` columns, so ``zone`` has 13 and ``habitat`` 31 columns.
    Finer cuts are exposed as separate groups.
    """
    names = list(FEATURE_NAMES if names is None else names)
    index = {n: i for i, n in enumerate(names)}
    S = feature_spec

    def cols(seq: Iterable[str]) -> list[int]:
        missing = [n for n in seq if n not in index]
        if missing:
            raise KeyError(f"feature names missing from the dump layout: {missing}")
        return sorted(index[n] for n in seq)

    alive_counts = (
        "alive_players_start_norm", "alive_players_end_norm", "alive_players_min_norm",
        "alive_players_avg_norm", "avg_alive_teams_norm", "avg_alive_players_norm", "hab_alive_teams",
    )  # fmt: skip
    zone_geometry = (
        "safe_radius_start_norm", "safe_radius_end_norm", "safe_radius_delta_norm",
        "safe_center_shift_norm", "poison_radius_start_norm", "poison_radius_end_norm",
        "zone_gap_start_norm", "zone_gap_end_norm",
    )  # fmt: skip
    zone_movement = ("safe_alignment", "zone_entry_time")
    zone_interaction = (
        "high_ground_x_boundary_pressure",
        "contact_delta_z_x_outside_safe_ratio",
        "z_percentile_x_hab_crowding_pressure",
    )
    spatial_only = set(names) - set(alive_counts)
    zone_global = [n for n in S.O2_PRESSURE_GLOBAL_FEATURES if n in spatial_only]
    zone_local = list(S.O2_LOCAL_PRESSURE_FEATURES)
    habitat = [n for n in S.HABITAT_FEATURES if n in spatial_only]

    covered = set()
    for g in (
        S.OM0_CONTEXT_FEATURES, S.O1_SURVIVAL_CAPITAL_FEATURES, S.O3_MOVEMENT_FEATURES,
        S.O2_PRESSURE_GLOBAL_FEATURES, S.O2_LOCAL_PRESSURE_FEATURES, S.O4_THREAT_EXPOSURE_FEATURES,
        S.O6_SPATIAL_PROXY_FEATURES, S.O6_Z_AFFORDANCE_FEATURES, S.NETWORK_NODE_FEATURES,
        S.HABITAT_FEATURES,
    ):  # fmt: skip
        covered |= set(g)
    loot_vehicle = [n for n in names if n not in covered]

    groups = {
        # headline families (what the noHab arms remove: zone + habitat)
        "zone": cols(zone_global + zone_local),
        "habitat": cols(habitat),
        "network": cols(S.NETWORK_NODE_FEATURES),
        "z_affordance": cols(S.O6_Z_AFFORDANCE_FEATURES),
        "alive_counts": cols(alive_counts),
        # spec.py blocks
        "context": cols(S.OM0_CONTEXT_FEATURES),
        "survival_capital": cols(S.O1_SURVIVAL_CAPITAL_FEATURES),
        "movement": cols(S.O3_MOVEMENT_FEATURES),
        "threat_exposure": cols(S.O4_THREAT_EXPOSURE_FEATURES),
        "spatial_proxy": cols(S.O6_SPATIAL_PROXY_FEATURES),
        "stdb_spatial": cols(S.STDB_SPATIAL_NODE_FEATURES),
        "global": cols(S.GLOBAL_FEATURES),
        "conventional": cols(S.CONVENTIONAL_NODE_FEATURES),
        "loot_vehicle": cols(loot_vehicle),
        "base66": cols(S.BASE66_FEATURES),
        "v2": cols(PHASE_NODE_FEATURE_NAMES_V2),
        # finer / alternative cuts of the zone family
        "zone_global": cols(zone_global),
        "zone_local": cols(zone_local),
        "zone_geometry": cols(zone_geometry),
        "zone_movement": cols(zone_movement),
        "zone_extended": cols(zone_global + zone_local + list(zone_movement)),
        "zone_interaction_z": cols(zone_interaction),
        # the features whose values were recomputed from kill events (meta['replaced_features'])
        "replaced": cols(REPLACED_FEATURES),
    }
    assert {"zone", "habitat"} <= set(groups)
    assert not set(groups["zone"]) & set(groups["alive_counts"])
    assert not set(groups["habitat"]) & set(groups["alive_counts"])
    return groups


# --------------------------------------------------------------------------------------
# event-based death model (labels v2, alive counts) -- pure
# --------------------------------------------------------------------------------------
Member = tuple  # (team_id, player_id, death_type, time_survived, last_kill_w)


def member_death_times(
    members: list[Member], *, fallback_offset_s: float = FALLBACK_OFFSET_S
) -> tuple[list[float], float]:
    """Per-member last death time on W (+inf = alive at match end) and the match offset used.

    Same rule as the label audit (reports/bigcomp2027_data_profile/scripts/label_audit_lib.py):
    alive -> +inf; else the last kill event; else ``time_survived + off``; else +inf, where ``off`` is
    the median over human members that have both a kill event and a time_survived of
    ``kill event - time_survived`` (``fallback_offset_s`` if there is none).
    """
    human = [
        lk - ts
        for (_t, pid, _dt, ts, lk) in members
        if not str(pid).startswith("ai.") and lk is not None and ts is not None
    ]
    off = float(np.median(human)) if human else float(fallback_offset_s)
    out: list[float] = []
    for (_t, _pid, dt, ts, lk) in members:
        if dt == "alive":
            d = float("inf")
        elif lk is not None:
            d = float(lk)
        elif ts is not None:
            d = float(ts) + off
        else:
            d = float("inf")
        out.append(d)
    return out, off


def team_death_times(
    members: list[Member], team_ids: list[int]
) -> tuple[np.ndarray, dict[int, list[float]], float]:
    """``team_death_s`` per team (max over members; +inf if any member is alive; NaN without members)
    plus the per-team member death lists and the match offset."""
    d, off = member_death_times(members)
    per_team: dict[int, list[float]] = {}
    for (tid, *_rest), dd in zip(members, d):
        per_team.setdefault(int(tid), []).append(dd)
    t = np.full(len(team_ids), np.nan, dtype=np.float64)
    for i, tid in enumerate(team_ids):
        vals = per_team.get(int(tid))
        if vals:
            t[i] = max(vals)
    return t, per_team, off


def alive_counts(
    member_deaths: list[float], start: float, end: float
) -> tuple[float, float, float, float]:
    """Event-based (start, end, min, avg) alive member counts of a phase ``[start, end]`` on W.

    A member is alive at time ``b`` iff its last death time is after ``b``.  With last-death
    semantics the count never increases, so ``min`` equals the count at the end; ``avg`` is the time
    average of the count over the phase.
    """
    n_start = float(sum(d > start for d in member_deaths))
    n_end = float(sum(d > end for d in member_deaths))
    if end > start:
        avg = float(
            sum(min(max((min(d, end) - start) / (end - start), 0.0), 1.0) for d in member_deaths)
        )
    else:
        avg = n_start
    return n_start, n_end, n_end, avg


def labels_v2(
    team_death: np.ndarray,
    bounds: dict[int, tuple[float, float]],
    last_phase_idx: int | None,
    n_phases: int = N_PHASES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(target_mask_v2, y_hazard_v2, node_mask_v2)`` on the dump's phase axis ([P, N])."""
    N = len(team_death)
    target = np.zeros((n_phases, N), dtype=bool)
    y = np.zeros((n_phases, N), dtype=np.float32)
    node = np.zeros((n_phases, N), dtype=bool)
    known = ~np.isnan(team_death)
    for ph, (start, _end) in bounds.items():
        if 1 <= ph <= n_phases:
            node[ph - 1] = known & (team_death > start)
    for k in range(1, n_phases):  # cuts 1..7 (the dump never has a cut at phase 8)
        if k not in bounds or (k + 1) not in bounds:
            continue
        e_k = bounds[k][1]
        e_next = float("inf") if last_phase_idx is not None and (k + 1) == last_phase_idx else bounds[k + 1][1]
        tgt = known & (team_death > e_k)
        target[k - 1] = tgt
        y[k - 1] = (tgt & np.isfinite(team_death) & (team_death <= e_next)).astype(np.float32)
    return target, y, node


# --------------------------------------------------------------------------------------
# pure assembly of one match
# --------------------------------------------------------------------------------------
@dataclass
class RawMatch:
    """Everything the DB returns for one match (plain python / numpy, no DB handles)."""

    match_id: str
    map_name: str
    team_ids: list[int]  # ascending, length N
    phase_indices: list[int]  # phases present in match_phase_global, within 1..8
    X_loader: np.ndarray  # (8, 128, F) loader layout (position-indexed)
    node_mask_loader: np.ndarray  # (8, 128) bool
    y_hazard_loader: np.ndarray  # (7, 128) loader layout (phase-1 indexed)
    target_mask_loader: np.ndarray  # (7, 128) bool
    y_rank: np.ndarray  # (128,) final_rank_norm
    y_topk: np.ndarray  # (128, 2) top5, top10
    edge_rows: list[tuple[int, str, int, int, float]]  # (phase_idx, edge_type, src, dst, weight)
    bounds: dict[int, tuple[float, float]]  # phase_idx -> (phase_start_s, phase_end_s), W numbers
    zone_rows: dict[tuple[str, int], tuple[float, float, float]]  # (kind, snapshot_t) -> (cx, cy, r)
    pos_rows: dict[int, list[tuple[float, float, float]]]  # team_id -> [(t, x_m, y_m)] sorted by t
    stats: dict[str, int] = field(default_factory=dict)
    clock_delta_s: float = 0.0  # W = G + delta used by the ETL windows (audit median, raised for drifting clocks)
    clock_delta_median_s: float = 0.0  # the audit's plain median
    last_phase_idx: int | None = None  # the match's last phase index (any index, may exceed 8)
    # --- label / audit inputs ---
    match_type: str | None = None
    team_won: dict[int, bool] = field(default_factory=dict)
    # (team_id, player_id, death_type, time_survived, last victim kill event on W or None)
    members: list[Member] = field(default_factory=list)
    monster_damage: dict[str, float] = field(default_factory=dict)  # victim account id -> Monster.* damage


def causal_cut_times(
    bounds: dict[int, tuple[float, float]], delta_s: float = 0.0
) -> dict[int, float | None]:
    """phase_idx -> last G grid point <= phase_end_s - delta - SNAPSHOT_INTERVAL inside the window
    (None when the phase is too short for one)."""
    return {
        ph: causal_cut.last_allowed_grid_point(start, end, delta_s) for ph, (start, end) in bounds.items()
    }


def _f32(x) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def assemble_match(raw: RawMatch, *, map_size_m: float) -> dict[str, Any]:
    """RawMatch -> the per-match dict of the dump schema (tensors).  Pure and DB-free."""
    N = len(raw.team_ids)
    P = N_PHASES
    F = raw.X_loader.shape[-1]
    team_index = {tid: i for i, tid in enumerate(raw.team_ids)}
    present = {ph for ph in raw.phase_indices if 1 <= ph <= P}
    delta = float(raw.clock_delta_s)

    X = np.zeros((P, N, F), dtype=np.float32)
    node = np.zeros((P, N), dtype=bool)
    phase_mask = np.zeros(P, dtype=bool)
    for pos, ph in enumerate(raw.phase_indices):
        if not 1 <= ph <= P:
            continue
        X[ph - 1] = raw.X_loader[pos, :N]
        node[ph - 1] = raw.node_mask_loader[pos, :N]
        phase_mask[ph - 1] = True
    # the loader fills NaN/inf with 0.0; assert it so the dump never carries non-finite inputs
    if not np.isfinite(X).all():
        raise ValueError(f"{raw.match_id}: non-finite value in X (loader contract broken)")

    target = np.zeros((P, N), dtype=bool)
    y = np.zeros((P, N), dtype=np.float32)
    target[: P - 1] = raw.target_mask_loader[:, :N]
    y[: P - 1] = raw.y_hazard_loader[:, :N]
    target &= phase_mask[:, None]
    y[~target] = 0.0

    # alive at the end of phase p  =>  alive at its start.  ``node_mask_label`` keeps the label
    # table's own alive_at_phase_start untouched for the label audit.
    node_label = node.copy()
    raised = target & ~node
    n_raised = int(raised.sum())
    node |= raised

    # --- edges (raw cache weights) ---
    buckets: dict[str, list[tuple[int, int, int, float]]] = {k: [] for k in EDGE_KEYS.values()}
    for ph, etype, src, dst, w in raw.edge_rows:
        key = EDGE_KEYS.get(etype)
        if key is None or ph not in present:
            continue
        i, j = team_index.get(int(src)), team_index.get(int(dst))
        if i is None or j is None or i == j:
            continue
        buckets[key].append((ph - 1, i, j, float(w)))
    edges: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for key, rows in buckets.items():
        rows.sort(key=lambda r: (r[0], r[1], r[2]))
        if rows:
            idx = torch.tensor([r[:3] for r in rows], dtype=torch.int64)
            w = torch.tensor([r[3] for r in rows], dtype=torch.float32)
        else:
            idx = torch.zeros((0, 3), dtype=torch.int64)
            w = torch.zeros((0,), dtype=torch.float32)
        edges[key] = (idx, w)

    # --- causal geometry: cut_t (G grid), team_pos, zone ---
    cuts = causal_cut_times({ph: raw.bounds[ph] for ph in present if ph in raw.bounds}, delta)
    cut_t = np.full(P, np.nan, dtype=np.float32)
    cut_wall = np.full(P, np.nan, dtype=np.float32)
    phase_start = np.full(P, np.nan, dtype=np.float32)
    phase_end = np.full(P, np.nan, dtype=np.float32)
    team_pos = np.full((P, N, 2), np.nan, dtype=np.float32)
    zone = np.full((P, len(ZONE_COLUMNS)), np.nan, dtype=np.float32)
    zone_valid = np.zeros(P, dtype=bool)
    size = float(map_size_m)

    for ph in sorted(present):
        if ph not in raw.bounds:
            continue
        p = ph - 1
        start, end = raw.bounds[ph]
        phase_start[p], phase_end[p] = start, end
        zone[p, 7] = float(ph)
        t_c = cuts.get(ph)
        if t_c is None:
            continue
        if not t_c + delta <= end - causal_cut.GUARD_S + 1e-6:  # pragma: no cover - by construction
            raise AssertionError(f"{raw.match_id} phase {ph}: cut {t_c} + delta beyond phase_end - guard")
        cut_t[p] = t_c
        cut_wall[p] = t_c + delta
        duration = max(end - start, 1.0)
        zone[p, 6] = float(np.clip((t_c + delta - start) / duration, 0.0, 1.0))
        safe = raw.zone_rows.get(("safe", int(t_c)))
        poison = raw.zone_rows.get(("poison", int(t_c)))
        if safe is not None:
            zone[p, 0:3] = np.asarray(safe, dtype=np.float64) / size
        if poison is not None:
            zone[p, 3:6] = np.asarray(poison, dtype=np.float64) / size
        zone_valid[p] = safe is not None and poison is not None
        t_lo = start - delta
        for tid, rows in raw.pos_rows.items():
            i = team_index.get(tid)
            if i is None:
                continue
            best = None
            for t, x, yy in rows:  # sorted by t
                if t < t_lo:
                    continue
                if t > t_c:
                    break
                best = (x, yy)
            if best is not None:
                team_pos[p, i] = (best[0] / size, best[1] / size)

    # --- labels v2 + event-based alive counts (W clock) ---
    team_death, per_team_deaths, _off = team_death_times(raw.members, raw.team_ids)
    tgt_v2, y_v2, node_v2 = labels_v2(
        team_death, {ph: raw.bounds[ph] for ph in present if ph in raw.bounds}, raw.last_phase_idx
    )
    if raw.members:
        fi = {name: FEATURE_NAMES.index(name) for name in REPLACED_FEATURES}
        for ph in sorted(present):
            if ph not in raw.bounds:
                continue
            start, end = raw.bounds[ph]
            for tid, i in team_index.items():
                n_s, n_e, n_min, n_avg = alive_counts(per_team_deaths.get(tid, []), start, end)
                X[ph - 1, i, fi["alive_players_start_norm"]] = n_s / ALIVE_SCALE
                X[ph - 1, i, fi["alive_players_end_norm"]] = n_e / ALIVE_SCALE
                X[ph - 1, i, fi["alive_players_min_norm"]] = n_min / ALIVE_SCALE
                X[ph - 1, i, fi["alive_players_avg_norm"]] = n_avg / ALIVE_SCALE

    # --- audit fields ---
    team_won = np.zeros(N, dtype=bool)
    team_ai = np.full(N, np.nan, dtype=np.float32)
    team_monster = np.zeros(N, dtype=np.float32)
    by_team: dict[int, list[str]] = {}
    account_team: dict[str, int] = {}
    for tid, pid, _dt, _ts, _lk in raw.members:
        by_team.setdefault(int(tid), []).append(str(pid))
        account_team[str(pid)] = int(tid)
    for tid, i in team_index.items():
        team_won[i] = bool(raw.team_won.get(tid, False))
        ids = by_team.get(tid, [])
        if ids:
            team_ai[i] = sum(pid.startswith("ai.") for pid in ids) / len(ids)
    for victim, dmg in raw.monster_damage.items():
        tid = account_team.get(str(victim))
        if tid in team_index:
            team_monster[team_index[tid]] += float(dmg)

    return {
        "match_id": raw.match_id,
        "team_ids": torch.tensor(raw.team_ids, dtype=torch.int64),
        "X": torch.from_numpy(X),
        "node_mask": torch.from_numpy(node),
        "target_mask": torch.from_numpy(target),
        "y_hazard": torch.from_numpy(y),
        "final_rank_norm": _f32(raw.y_rank[:N]),
        "top5": _f32(raw.y_topk[:N, 0]),
        "top10": _f32(raw.y_topk[:N, 1]),
        "edges": edges,
        "team_pos": torch.from_numpy(team_pos),
        "zone": torch.from_numpy(zone),
        "zone_valid": torch.from_numpy(zone_valid),
        # audit extras
        "map_size": size,
        "phase_mask": torch.from_numpy(phase_mask),
        "phase_start_s": torch.from_numpy(phase_start),
        "phase_end_s": torch.from_numpy(phase_end),
        "cut_t": torch.from_numpy(cut_t),
        "cut_wall_s": torch.from_numpy(cut_wall),
        "clock_delta_s": delta,
        "clock_delta_median_s": float(raw.clock_delta_median_s),
        "node_mask_label": torch.from_numpy(node_label),
        "last_phase_idx": -1 if raw.last_phase_idx is None else int(raw.last_phase_idx),
        # labels v2 (the pipeline labels above are untouched)
        "y_hazard_v2": torch.from_numpy(y_v2),
        "target_mask_v2": torch.from_numpy(tgt_v2),
        "node_mask_v2": torch.from_numpy(node_v2),
        "team_death_s": _f32(team_death),
        "team_won": torch.from_numpy(team_won),
        "team_ai_frac": _f32(team_ai),
        "team_has_monster_dmg": _f32(team_monster),
        "match_type": raw.match_type,
        "_stats": {"node_mask_raised": n_raised, "n_targets": int(target.sum())},
    }


# --------------------------------------------------------------------------------------
# DB side (read-only)
# --------------------------------------------------------------------------------------
def fetch_raw_match(cur, match_id: str, map_name: str) -> RawMatch | None:
    """Collect one match with the stdb.queries building blocks + the extra causal reads."""
    from stdb.queries import (
        _fetch_phase_indices,
        _fetch_phase_node_features,
        _fetch_phase_targets,
        _phase_team_order,
    )

    team_ids = _phase_team_order(cur, match_id)
    if not team_ids:
        return None
    phase_indices = _fetch_phase_indices(cur, match_id)
    if not phase_indices:
        return None
    cur.execute(
        "SELECT 1 FROM stdb.team_phase_edge_cache WHERE match_id = %s LIMIT 1", (match_id,)
    )
    if cur.fetchone() is None:
        return None

    X, y_rank, y_topk, _team_mask, node_mask, _phase_mask = _fetch_phase_node_features(
        cur, match_id, team_ids, phase_indices, include_v2_features=True
    )
    y_hazard, target_mask = _fetch_phase_targets(cur, match_id, team_ids, phase_indices)

    cur.execute(
        """
        SELECT phase_idx, edge_type, src_team_id, dst_team_id, weight
          FROM stdb.team_phase_edge_cache
         WHERE match_id = %s AND phase_idx = ANY(%s)
           AND src_team_id = ANY(%s) AND dst_team_id = ANY(%s)
        """,
        (match_id, phase_indices, team_ids, team_ids),
    )
    edge_rows = [(int(p), str(e), int(s), int(d), float(w)) for p, e, s, d, w in cur.fetchall()]

    cur.execute(
        """
        SELECT phase_idx, phase_start_s, phase_end_s
          FROM stdb.v_match_phase_bounds
         WHERE match_id = %s AND phase_idx = ANY(%s)
        """,
        (match_id, phase_indices),
    )
    bounds = {int(p): (float(a), float(b)) for p, a, b in cur.fetchall()}
    cur.execute(
        "SELECT max(phase_idx) FROM stdb.v_match_phase_bounds WHERE match_id = %s AND phase_idx > 0",
        (match_id,),
    )
    row = cur.fetchone()
    last_phase_idx = int(row[0]) if row and row[0] is not None else None

    clock = causal_cut.fetch_clock_delta_detail(cur, match_id)
    delta = clock.safe_s
    cuts = causal_cut_times(bounds, delta)
    cut_points = sorted({int(t) for t in cuts.values() if t is not None})
    zone_rows: dict[tuple[str, int], tuple[float, float, float]] = {}
    pos_rows: dict[int, list[tuple[float, float, float]]] = {}
    if cut_points:
        cur.execute(
            """
            SELECT region_kind, snapshot_t, ST_X(center_pos), ST_Y(center_pos), radius_m
              FROM stdb.dynamic_regions
             WHERE match_id = %s AND region_kind IN ('safe', 'poison') AND snapshot_t = ANY(%s)
            """,
            (match_id, cut_points),
        )
        for kind, t, cx, cy, r in cur.fetchall():
            zone_rows[(str(kind), int(t))] = (float(cx), float(cy), float(r))
        lo = min(a for a, _ in bounds.values()) - delta
        cur.execute(
            """
            SELECT team_id, snapshot_t, pos_x, pos_y
              FROM stdb.team_snapshot_features
             WHERE match_id = %s AND snapshot_t BETWEEN %s AND %s
               AND pos_x IS NOT NULL AND pos_y IS NOT NULL
             ORDER BY team_id, snapshot_t
            """,
            (match_id, int(lo) - 1, int(max(cut_points))),
        )
        for tid, t, x, y in cur.fetchall():
            pos_rows.setdefault(int(tid), []).append((float(t), float(x), float(y)))

    audit = fetch_label_audit_inputs(cur, match_id)

    return RawMatch(
        match_id=match_id,
        map_name=map_name,
        team_ids=list(team_ids),
        phase_indices=list(phase_indices),
        X_loader=X,
        node_mask_loader=node_mask,
        y_hazard_loader=y_hazard,
        target_mask_loader=target_mask,
        y_rank=y_rank,
        y_topk=y_topk,
        edge_rows=edge_rows,
        bounds=bounds,
        zone_rows=zone_rows,
        pos_rows=pos_rows,
        clock_delta_s=delta,
        clock_delta_median_s=clock.median_s,
        last_phase_idx=last_phase_idx,
        **audit,
    )


def fetch_label_audit_inputs(cur, match_id: str) -> dict[str, Any]:
    """Raw inputs of the labels v2 / audit fields (read-only; per-match, index-friendly queries)."""
    cur.execute("SELECT match_type FROM pubg.matches WHERE match_id = %s", (match_id,))
    row = cur.fetchone()
    match_type = row[0] if row else None

    cur.execute("SELECT team_id, won FROM stdb.teams WHERE match_id = %s", (match_id,))
    team_won = {int(t): bool(w) for t, w in cur.fetchall()}

    # last victim kill event per account, on the LogMatchStart clock (stdb.matches.started_at)
    cur.execute(
        """
        SELECT r.team_id, p.player_id, p.death_type, p.time_survived::float8, k.last_w
          FROM pubg.participants p
          JOIN pubg.rosters r ON r.match_id = p.match_id AND r.roster_id = p.roster_id
          LEFT JOIN (
                SELECT kk.victim_id,
                       max(extract(epoch FROM (kk.event_time - m.started_at)))::float8 AS last_w
                  FROM pubg.telem_kills kk
                  JOIN stdb.matches m ON m.match_id = kk.match_id
                 WHERE kk.match_id = %s AND kk.victim_id IS NOT NULL
                 GROUP BY kk.victim_id
          ) k ON k.victim_id = p.player_id
         WHERE p.match_id = %s AND r.team_id IS NOT NULL
         ORDER BY r.team_id, p.player_id
        """,
        (match_id, match_id),
    )
    members = [
        (
            int(t),
            str(pid),
            None if dt is None else str(dt),
            None if ts is None else float(ts),
            None if lw is None else float(lw),
        )
        for t, pid, dt, ts, lw in cur.fetchall()
    ]

    cur.execute(
        """
        SELECT victim_id, sum(damage)::float8
          FROM pubg.telem_damage
         WHERE match_id = %s AND attacker_id LIKE 'Monster.%%'
         GROUP BY victim_id
        """,
        (match_id,),
    )
    monster_damage = {str(v): float(d) for v, d in cur.fetchall() if v is not None and d is not None}
    return {
        "match_type": match_type,
        "team_won": team_won,
        "members": members,
        "monster_damage": monster_damage,
    }


def verify_map_sizes(cur, maps: Iterable[str], tol: float = 0.01) -> dict[str, dict[str, float]]:
    """Cross-check MAP_SIZE_M against the data: the first safe circle is centred on the map."""
    out: dict[str, dict[str, float]] = {}
    for name in sorted(set(maps)):
        cur.execute(
            """
            SELECT m.match_id FROM stdb.matches m
             WHERE m.map_name = %s ORDER BY m.match_id LIMIT 5
            """,
            (name,),
        )
        sizes = []
        for (mid,) in cur.fetchall():
            cur.execute(
                """
                SELECT ST_X(center_pos), ST_Y(center_pos)
                  FROM stdb.dynamic_regions
                 WHERE match_id = %s AND region_kind = 'safe'
                 ORDER BY snapshot_t LIMIT 1
                """,
                (mid,),
            )
            row = cur.fetchone()
            if row:
                sizes.append(float(row[0]) + float(row[1]))
        implied = float(np.median(sizes)) if sizes else float("nan")
        expected = MAP_SIZE_M[name]
        out[name] = {"expected_m": expected, "implied_m": implied, "n_matches": len(sizes)}
        if sizes and abs(implied - expected) / expected > tol:
            raise ValueError(
                f"{name}: map size table says {expected} m but the initial safe circle implies "
                f"{implied:.0f} m"
            )
    return out


# --------------------------------------------------------------------------------------
# population (design v2.3)
# --------------------------------------------------------------------------------------
def compute_population(
    cur, specs: Iterable[CohortSpec], split_root: Path = SPLIT_ROOT
) -> tuple[dict[str, Any], dict[str, dict[str, list[str]]]]:
    """Per-cohort train / val / test counts before and after the v2.3 exclusions.

    Returns ``(population_meta, kept_ids)`` where ``kept_ids[cohort][split]`` are the surviving
    match ids in the stored split order (the split assignment itself is never changed).
    """
    manifests: dict[str, dict[str, list[str]]] = {}
    for spec in specs:
        path = split_root / spec.slug / "split.json"
        manifests[spec.slug] = json.loads(path.read_text(encoding="utf-8"))["splits"]
    all_ids = sorted({m for sp in manifests.values() for ids in sp.values() for m in ids})
    cur.execute("SELECT match_id, match_type FROM pubg.matches WHERE match_id = ANY(%s)", (all_ids,))
    mtype = {m: t for m, t in cur.fetchall()}

    cohorts: dict[str, Any] = {}
    kept_ids: dict[str, dict[str, list[str]]] = {}
    for spec in specs:
        sp = manifests[spec.slug]
        before = {s: len(sp[s]) for s in SPLITS}
        by_type: dict[str, int] = {}
        for s in SPLITS:
            for m in sp[s]:
                by_type[str(mtype.get(m))] = by_type.get(str(mtype.get(m)), 0) + 1
        cohort_excluded = spec.slug in EXCLUDED_COHORTS
        kept = {
            s: [m for m in sp[s] if (not cohort_excluded) and mtype.get(m) not in EXCLUDED_MATCH_TYPES]
            for s in SPLITS
        }
        after = {s: len(kept[s]) for s in SPLITS}
        total_after = sum(after.values())
        excluded_by_type = {
            t: sum(1 for s in SPLITS for m in sp[s] if mtype.get(m) == t) for t in EXCLUDED_MATCH_TYPES
        }
        if cohort_excluded:
            status, reason = "excluded", "cohort rule (Vikendi)"
        elif total_after < MIN_COHORT_MATCHES:
            status, reason = "dropped", f"fewer than {MIN_COHORT_MATCHES} matches after exclusion"
        else:
            status, reason = "kept", ""
        by_type_after: dict[str, int] = {}
        for s in SPLITS:
            for m in kept[s]:
                by_type_after[str(mtype.get(m))] = by_type_after.get(str(mtype.get(m)), 0) + 1
        cohorts[spec.slug] = {
            "label": spec.label,
            "before": {**before, "total": sum(before.values())},
            "after": {**after, "total": total_after},
            "match_type_before": by_type,
            "match_type_after": by_type_after,
            "excluded_by_match_type": excluded_by_type,
            "status": status,
            "reason": reason,
        }
        kept_ids[spec.slug] = kept
    population = {
        "rules": {
            "exclude_match_types": list(EXCLUDED_MATCH_TYPES),
            "exclude_cohorts": list(EXCLUDED_COHORTS),
            "min_matches_per_cohort": MIN_COHORT_MATCHES,
            "split_assignment": "stored split.json kept for the surviving matches (no re-split)",
        },
        "cohorts": cohorts,
    }
    return population, kept_ids


# --------------------------------------------------------------------------------------
# worker pool
# --------------------------------------------------------------------------------------
_CONN = None


def _init_worker() -> None:
    global _CONN
    from stdb.conn import connect

    _CONN = connect()
    _CONN.set_session(readonly=True, autocommit=True)


def required_since(since_iso: str | None, bump_since_iso: str | None, raised: bool) -> str | None:
    """Earliest ``created_at`` the derived rows of a match may have.

    Matches whose clock delta was raised (drifting clock) must have been rebuilt AFTER the drift-aware rule
    went live (``bump_since_iso``); without that marker they count as stale (``datetime.max``)."""
    if not since_iso:
        return None
    if not raised:
        return since_iso
    if not bump_since_iso:
        return datetime.max.replace(tzinfo=timezone.utc).isoformat()
    later = max(datetime.fromisoformat(since_iso), datetime.fromisoformat(bump_since_iso))
    return later.isoformat()


def _work(job: tuple[str, str, str, str]) -> tuple[str, dict | None, str | None]:
    match_id, map_name, since_iso, bump_since_iso = job
    try:
        with _CONN.cursor() as cur:
            if since_iso:
                raised = causal_cut.fetch_clock_delta_detail(cur, match_id).raised
                need = required_since(since_iso, bump_since_iso, raised)
                stale = stale_fixb_tables(cur, match_id, need)
                if stale:
                    why = "raised clock delta (drifting clock) but not rebuilt after the drift-aware rule" if raised else "pre Fix B"
                    return match_id, None, f"stale ({why}) derived rows: {stale}"
            raw = fetch_raw_match(cur, match_id, map_name)
            if raw is None:
                return match_id, None, "not phase-ready (no teams / phases / edge cache)"
            return match_id, assemble_match(raw, map_size_m=MAP_SIZE_M[map_name]), None
    except Exception as exc:  # pragma: no cover - reported per match
        return match_id, None, f"error: {exc!r}"


FIXB_CREATED_AT_TABLES = (
    "match_phase_global",
    "team_phase_state",
    "team_phase_edge_cache",
    "team_phase_habitat_features",
    "team_phase_network_features",
    "team_phase_z_affordance_features",
)


def stale_fixb_tables(cur, match_id: str, since_iso: str) -> list[str]:
    """Derived tables whose rows for this match predate the Fix B rerun (created_at < since)."""
    stale = []
    for table in FIXB_CREATED_AT_TABLES:
        cur.execute(
            f"SELECT min(created_at) < %s::timestamptz FROM stdb.{table} WHERE match_id = %s",
            (since_iso, match_id),
        )
        row = cur.fetchone()
        if row is None or row[0] is None or row[0]:
            # None => no rows at all: z-affordance may legitimately be absent only if never built
            if row is not None and row[0] is None and table == "team_phase_z_affordance_features":
                continue
            stale.append(table)
    return stale


# --------------------------------------------------------------------------------------
# label agreement (labels v2 vs the pipeline) -- pure
# --------------------------------------------------------------------------------------
def three_state(target: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """0 = outside the risk set, 1 = in the risk set and negative, 2 = in the risk set and positive."""
    return torch.where(target, torch.where(y > 0.5, 2, 1), 0)


def label_agreement(matches: list[dict]) -> dict[str, Any]:
    """3x3 confusion (rows: pipeline, cols: v2) over every (match, team, cut k = 1..7) whose phases k
    and k+1 both exist, plus the per-cut hazard of both label sets."""
    conf = np.zeros((3, 3), dtype=np.int64)
    pos_p = np.zeros(7)
    risk_p = np.zeros(7)
    pos_v = np.zeros(7)
    risk_v = np.zeros(7)
    won_teams = won_finite = won_finite_positives = 0
    for m in matches:
        # winners are censored (+inf) only if a member is alive at the end; a winner whose members all died
        # keeps a finite team_death_s and can carry v2 positives (reported, not overridden: team_won is in the dump)
        won = m["team_won"]
        finite_winner = won & torch.isfinite(m["team_death_s"])
        won_teams += int(won.sum())
        won_finite += int(finite_winner.sum())
        won_finite_positives += int(((m["y_hazard_v2"] > 0.5) & m["target_mask_v2"] & finite_winner[None, :]).sum())
        mask = m["phase_mask"]
        valid = torch.zeros_like(m["target_mask"])
        for k in range(1, N_PHASES):
            if bool(mask[k - 1]) and bool(mask[k]):
                valid[k - 1] = True
        sp = three_state(m["target_mask"], m["y_hazard"])
        sv = three_state(m["target_mask_v2"], m["y_hazard_v2"])
        for a in range(3):
            for b in range(3):
                conf[a, b] += int(((sp == a) & (sv == b) & valid).sum())
        for k in range(7):
            vk = valid[k]
            risk_p[k] += float((m["target_mask"][k] & vk).sum())
            pos_p[k] += float(((m["y_hazard"][k] > 0.5) & m["target_mask"][k] & vk).sum())
            risk_v[k] += float((m["target_mask_v2"][k] & vk).sum())
            pos_v[k] += float(((m["y_hazard_v2"][k] > 0.5) & m["target_mask_v2"][k] & vk).sum())
    n = int(conf.sum())
    both_pos = int(conf[2, 2])
    return {
        "rows": n,
        "confusion_rows_pipeline_cols_v2": conf.tolist(),
        "states": ["outside_risk_set", "negative", "positive"],
        "agree": float(np.trace(conf) / max(n, 1)),
        "risk_set_pipeline": int(conf[1:, :].sum()),
        "risk_set_v2": int(conf[:, 1:].sum()),
        "positives_pipeline": int(conf[2, :].sum()),
        "positives_v2": int(conf[:, 2].sum()),
        "positive_precision_of_pipeline": float(both_pos / max(conf[2, :].sum(), 1)),
        "positive_recall_of_pipeline": float(both_pos / max(conf[:, 2].sum(), 1)),
        "winner_teams": won_teams,
        "winner_teams_with_finite_death_s": won_finite,
        "v2_positives_on_those_winners": won_finite_positives,
        "hazard_by_cut_pipeline": [float(a / b) if b else None for a, b in zip(pos_p, risk_p)],
        "hazard_by_cut_v2": [float(a / b) if b else None for a, b in zip(pos_v, risk_v)],
    }


# --------------------------------------------------------------------------------------
# cohort driver
# --------------------------------------------------------------------------------------
def build_meta(
    spec: CohortSpec, split: str, *, split_path: Path, split_ids: list[str], extra: dict
) -> dict[str, Any]:
    prov = git_provenance()
    groups = build_feature_groups()
    return {
        "cohort": spec.slug,
        "cohort_label": spec.label,
        "map_name": spec.map_name,
        "game_mode": spec.game_mode,
        "split": split,
        "split_json": str(split_path),
        "split_ids_sha1": hashlib.sha1("\n".join(split_ids).encode("utf-8")).hexdigest(),
        "split_seed": SEED,
        "n_matches": len(split_ids),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": prov.get("git_commit"),
        "git_dirty": bool(prov.get("git_dirty")),
        "fixb": True,
        "fixb_cut_guard_s": causal_cut.GUARD_S,
        "single_clock": True,
        "design_version": "v2.3",
        "feature_names": list(FEATURE_NAMES),
        "feature_groups": {k: [int(c) for c in v] for k, v in groups.items()},
        "replaced_features": list(REPLACED_FEATURES),
        "zone_columns": list(ZONE_COLUMNS),
        "map_size_m": {k: float(v) for k, v in MAP_SIZE_M.items()},
        "n_phases": N_PHASES,
        "n_features": N_FEATURES,
        "edge_keys": dict(EDGE_KEYS),
        "conventions": {
            "phase_axis": "p = phase_idx - 1 for phases 1..8; absent phases are zero/False/NaN",
            "X": "stdb.queries._fetch_phase_node_features(include_v2_features=True); fixed-constant "
                 "scaling; NULL/NaN/inf -> 0.0 (loader rule, no missing indicator); the 4 "
                 "meta['replaced_features'] columns are event-based alive member counts / 4",
            "alive_counts": "start = #members with last death time > phase_start_s, end = min = # > "
                            "phase_end_s, avg = time average over [phase_start_s, phase_end_s] (W clock)",
            "node_mask": "label alive_at_phase_start, raised where alive_at_phase_end disagrees "
                         "(node_mask_label keeps the unraised flag; the count is in stats)",
            "target_mask / y_hazard": "the pipeline's labels (team_phase_labels), unchanged",
            "labels_v2": "team_death_s = max over members of the last kill event - started_at (+inf if a member "
                         "is alive at the end; fallback time_survived + match offset when a dead member has no "
                         "kill event); target_mask_v2[k] = team_death_s > phase_end_s[k] and phase k+1 exists; "
                         "y_hazard_v2[k] = team_death_s <= phase_end_s[k+1] (+inf when k+1 is last_phase_idx); "
                         "node_mask_v2[k] = team_death_s > phase_start_s[k]",
            "final_rank_norm": "final_rank / n_teams; rank 1 = winner, so smaller is better",
            "edges": "raw stdb.team_phase_edge_cache weights; prox/copress stored both directions; "
                     "combat attacker -> victim; no self loops, no row-normalisation",
            "team_pos": "centroid / map size at latest alive grid snapshot <= cut_t within the phase "
                        "window; NaN if none",
            "zone": "[cur_cx, cur_cy, cur_r, next_cx, next_cy, next_r, phase_frac, phase_no] at cut_t; "
                    "geometry / map size, phase_frac = (cut_wall_s - start)/(end - start), phase_no raw",
            "cut_t": "G grid time: last grid point <= phase_end_s - clock_delta_s - 10 (one clock, then cap)",
            "team_has_monster_dmg": "damage taken from attackers like 'Monster.%' (0.0 if none), not a flag",
            "team_ai_frac": "share of the roster with player_id like 'ai.%'",
        },
        "clock": CLOCK_DOC,
        **extra,
    }


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def dump_cohort(
    spec: CohortSpec,
    out_dir: Path,
    *,
    workers: int,
    population: dict[str, Any],
    kept_ids: dict[str, list[str]],
    split_root: Path = SPLIT_ROOT,
    limit: int | None = None,
    fixb_since: str | None = None,
    bump_since: str | None = None,
) -> dict[str, Any]:
    from stdb.queries import list_phase_eligible_match_ids

    t0 = time.time()
    split_path = split_root / spec.slug / "split.json"
    ids = list_phase_eligible_match_ids(spec.map_name, spec.game_mode, None)
    # validates that the stored manifest (the full, pre-exclusion one) is still a subset of the eligible set
    load_or_create_split(ids, split_json=str(split_path), seed=SEED, require_existing=True)
    parts = {s: list(kept_ids[s]) for s in SPLITS}
    if limit:
        parts = {k: v[:limit] for k, v in parts.items()}

    cohort_dir = out_dir / spec.slug
    cohort_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")  # fork + libpq is unsafe on macOS
    manifest: dict[str, Any] = {
        "cohort": spec.slug,
        "git_commit": git_provenance().get("git_commit"),
        "fixb_since": fixb_since,
        "bump_since": bump_since,
        "population": population["cohorts"][spec.slug],
        "splits": {},
    }
    with ctx.Pool(processes=max(1, workers), initializer=_init_worker) as pool:
        for split, split_ids in parts.items():
            jobs = [(mid, spec.map_name, fixb_since or "", bump_since or "") for mid in split_ids]
            results = pool.map(_work, jobs, chunksize=4)
            matches, problems = [], {}
            stats = {"node_mask_raised": 0, "n_targets": 0, "zone_invalid_phases": 0, "clock_delta_raised": 0}
            for mid, item, err in results:
                if item is None:
                    problems[mid] = err
                    continue
                s = item.pop("_stats")
                stats["node_mask_raised"] += s["node_mask_raised"]
                stats["n_targets"] += s["n_targets"]
                stats["zone_invalid_phases"] += int(
                    (item["phase_mask"] & ~item["zone_valid"]).sum()
                )
                stats["clock_delta_raised"] += int(item["clock_delta_s"] > item["clock_delta_median_s"] + 1e-9)
                matches.append(item)
            if problems:
                logger.error(
                    "%s/%s: %d of %d matches could not be dumped, e.g. %s",
                    spec.slug, split, len(problems), len(split_ids), list(problems.items())[:3],
                )
            agreement = label_agreement(matches)
            meta = build_meta(
                spec, split, split_path=split_path, split_ids=split_ids,
                extra={
                    "n_dumped": len(matches),
                    "skipped_matches": problems,
                    "stats": stats,
                    "label_agreement": agreement,
                    "population": {**population, "this_cohort": spec.slug},
                    "fixb_since": fixb_since,
                    "bump_since": bump_since,
                },
            )
            path = cohort_dir / f"{split}.pt"
            tmp = cohort_dir / f".{split}.pt.tmp"
            torch.save({"meta": meta, "matches": matches}, tmp)
            os.replace(tmp, path)  # atomic: a concurrent reader never sees a half-written file
            manifest["splits"][split] = {
                "file": path.name,
                "bytes": path.stat().st_size,
                "n_split_ids": len(split_ids),
                "n_dumped": len(matches),
                "n_skipped": len(problems),
                "stats": stats,
                "label_agreement": agreement,
            }
            logger.info(
                "%s/%s: %d/%d matches -> %s (%.1f MB) stats=%s agree=%.3f",
                spec.slug, split, len(matches), len(split_ids), path, path.stat().st_size / 1e6, stats,
                agreement["agree"],
            )
    manifest["elapsed_s"] = round(time.time() - t0, 1)
    manifest["total_bytes"] = sum(s["bytes"] for s in manifest["splits"].values())
    _write_json(cohort_dir / "dump_manifest.json", manifest)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cohort", action="append", help="cohort slug / label / alias (repeatable)")
    g.add_argument("--all", action="store_true", help="all required cohorts that survive the population rules")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--split-root", type=Path, default=SPLIT_ROOT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N matches per split")
    ap.add_argument(
        "--fixb-since",
        default=None,
        help="ISO timestamp; every dumped match must have derived rows created at/after it "
        "(default: model_stgcn/etl_runs/fixb_2026-10-05/fixb_since.txt)",
    )
    ap.add_argument(
        "--bump-since",
        default=None,
        help="ISO timestamp from which the drift-aware clock delta is live; matches with a raised delta must be "
        "rebuilt after it (default: model_stgcn/etl_runs/fixb_2026-10-05/fixb_bump_since.txt if present)",
    )
    ap.add_argument("--no-fixb-check", action="store_true")
    ap.add_argument("--ignore-population-rules", action="store_true",
                    help="dump an excluded / dropped cohort anyway (never for the study)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    all_specs = list(REQUIRED_COHORTS)
    requested = all_specs if args.all else [resolve_cohort(c) for c in args.cohort]
    since = None
    if not args.no_fixb_check:
        since = args.fixb_since
        if since is None:
            marker = Path("model_stgcn/etl_runs/fixb_2026-10-05/fixb_since.txt")
            if not marker.exists():
                raise SystemExit(
                    f"{marker} is missing: run tools/run_fixb_phase_etl.py first, or pass "
                    "--fixb-since / --no-fixb-check"
                )
            since = marker.read_text(encoding="utf-8").strip()

    bump = None
    if since is not None:
        bump = args.bump_since
        if bump is None:
            bump_marker = Path("model_stgcn/etl_runs/fixb_2026-10-05/fixb_bump_since.txt")
            bump = bump_marker.read_text(encoding="utf-8").strip() if bump_marker.exists() else None

    from stdb.conn import connect

    conn = connect()
    try:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor() as cur:
            population, kept = compute_population(cur, all_specs, args.split_root)
            sizes = verify_map_sizes(cur, {s.map_name for s in requested})
    finally:
        conn.close()
    logger.info("map sizes verified against the data: %s", sizes)
    for slug, rec in population["cohorts"].items():
        logger.info("population %-26s %-8s before=%s after=%s %s", slug, rec["status"], rec["before"],
                    rec["after"], rec["reason"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_dir / "population.json", population)

    failed = 0
    for spec in requested:
        status = population["cohorts"][spec.slug]["status"]
        if status != "kept" and not args.ignore_population_rules:
            logger.warning("skipping %s: %s (%s)", spec.slug, status, population["cohorts"][spec.slug]["reason"])
            continue
        manifest = dump_cohort(
            spec, args.out_dir, workers=args.workers, population=population, kept_ids=kept[spec.slug],
            split_root=args.split_root, limit=args.limit, fixb_since=since, bump_since=bump,
        )
        if any(s["n_skipped"] for s in manifest["splits"].values()):
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
