"""Stage-2 player layer: the player dump, its join to the phase dump, and the player relation tensors.

Design doc section 3 (stage 2) and 4.1.  The player dump (written by another job, one directory per cohort next to the
phase dump: ``model_stgcn/dumps/bigcomp2027_player_v1/<cohort>/{train,val,test}.pt``) has the layout of the phase dump
(``{"meta": {...}, "matches": [...]}``) with, per match::

    match_id           str, the join key to the phase dump
    player_ids         [P]            participant ids (list of account-id strings or a tensor)
    player_team_idx    [P] int        index into the PHASE dump's ``team_ids`` of the match (-1 = no team: never alive)
    player_is_ai       [P]            AI flag
    Xp                 f32 [8, P, Fp] player features per phase slot (NaN = missing)
    player_alive_cut   bool [8, P]    the CAUSAL alive flag at the cut: no kill event up to the cut, or revived (a fresh
                                      health > 0 sample after the last kill event).  The only player alive mask used.
    edges_player       {type: (idx int64 [E, 3] = (p, i, j), w f32 [E])}   types 'teammate', 'enemy_prox', 'combat'

and ``meta["player_feature_names"]`` (``Fp`` names).  ``player_node_mask`` (event based, "alive at the phase start") is
NEVER read: it follows the last-kill-event rule, so it keeps a player that was killed and later recalled or redeployed in
the dead gap, which leaks the recall outcome; ``player_alive_cut_strict`` is not used either.  The alive player tokens are
``player_alive_cut`` AND the team's alive flag of the phase dump (team-level masks stay as they are).  Slot ``p`` is the
phase slot of the phase dump, ``i`` / ``j`` index
the players of the match.  Other edge types are ignored (and reported); everything else that is missing or malformed fails
loudly with the match id.

Relation channels of the player attention, indexed ``[query i, key j]`` (weights used as dumped, no row normalisation,
diagonal 0 as on the team axis)::

    teammate    w   symmetric (stored once or in both directions)
    enemy_prox  w   symmetric; the dump holds only pairs of different teams within 300 m (not re-checked here)
    combat_out  log1p(intensity[i -> j])  directed attacker -> victim, ``raw`` or already ``log`` weights (see relations)
    combat_in   log1p(intensity[j -> i])  its transpose

Escalation (as for teams): when a combat edge exists in either direction for a pair, the ``enemy_prox`` weight of that
pair is zeroed, provided a combat channel is requested.  The ``-tm`` arm simply has no ``teammate`` channel.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[5]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

import pubg_lab.survival.ptr.models.relations as R

PLAYER_DUMP_NAME = "bigcomp2027_player_v1"
PLAYER_EDGE_TYPES: tuple[str, ...] = ("teammate", "enemy_prox", "combat")
PLAYER_SYMMETRIC_EDGE_TYPES: tuple[str, ...] = ("teammate", "enemy_prox")
PLAYER_RELATION_TYPES: tuple[str, ...] = ("teammate", "enemy_prox", "combat_out", "combat_in")
PLAYER_ALIVE_FIELD = (
    "player_alive_cut"  # the causal per-cut alive flag; player_node_mask is deliberately never read
)
PLAYER_FIELDS: tuple[str, ...] = (
    "match_id", "player_ids", "player_team_idx", "player_is_ai", "Xp", PLAYER_ALIVE_FIELD, "edges_player",
)  # fmt: skip
ENDPOINT_STRICT_TYPES: tuple[str, ...] = (
    "teammate",
    "enemy_prox",
)  # built from alive tokens only: every endpoint must be live
PLAYER_COMBAT_LOG_MAX = (
    30.0  # same rule as the team edges: every train weight <= 30 means ln(1 + intensity) is stored
)


def check_player_relation_types(types: Sequence[str]) -> tuple[str, ...]:
    types = tuple(types)
    unknown = [t for t in types if t not in PLAYER_RELATION_TYPES]
    if unknown or len(set(types)) != len(types):
        raise ValueError(
            f"player_relation_types must be distinct members of {PLAYER_RELATION_TYPES}, got {types}"
        )
    return types


def default_player_dump_dir(phase_dump_dir: str | Path) -> Path:
    """The player dump directory next to a phase dump directory."""
    return Path(phase_dump_dir).parent / PLAYER_DUMP_NAME


# --------------------------------------------------------------------------- loading and validation
def load_player_dump(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"player dump file not found: {path}")
    dump = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(dump, Mapping) or "meta" not in dump or "matches" not in dump:
        raise ValueError(f"{path}: a player dump is a dict with 'meta' and 'matches'")
    names = dump["meta"].get("player_feature_names")
    if not names:
        raise ValueError(f"{path}: meta['player_feature_names'] is missing or empty")
    return dump


def validate_player_match(pm: Mapping, n_teams: int, n_phases: int, n_features: int) -> list[str]:
    """Check one player match against its phase match (``n_teams`` teams, ``n_phases`` slots); returns the ignored edge types."""
    mid = pm.get("match_id", "?")
    missing = [k for k in PLAYER_FIELDS if k not in pm]
    if PLAYER_ALIVE_FIELD in missing:
        raise ValueError(
            f"player match {mid}: the dump has no '{PLAYER_ALIVE_FIELD}' (the causal per-cut alive flag).  Stage 2 needs it; "
            "'player_node_mask' is never used (it follows the last-kill-event rule and leaks the recall outcome)"
        )
    if missing:
        raise ValueError(f"player match {mid}: missing fields {missing}")
    xp = torch.as_tensor(pm["Xp"])
    if xp.ndim != 3 or xp.shape[0] != n_phases or xp.shape[2] != n_features:
        raise ValueError(
            f"player match {mid}: Xp must be [{n_phases}, P, {n_features}], got {tuple(xp.shape)}"
        )
    n_players = int(xp.shape[1])
    for key in ("player_ids", "player_team_idx", "player_is_ai"):
        got = len(
            pm[key]
        )  # player_ids is a list of account ids (strings) in the real dump, a tensor elsewhere
        if got != n_players:
            raise ValueError(f"player match {mid}: {key} must have {n_players} entries, got {got}")
    mask = torch.as_tensor(pm[PLAYER_ALIVE_FIELD])
    if tuple(mask.shape) != (n_phases, n_players):
        raise ValueError(
            f"player match {mid}: {PLAYER_ALIVE_FIELD} must be [{n_phases}, {n_players}], got {tuple(mask.shape)}"
        )
    team = torch.as_tensor(pm["player_team_idx"]).to(torch.int64)
    if n_players and (int(team.min()) < -1 or int(team.max()) >= n_teams):
        raise ValueError(
            f"player match {mid}: player_team_idx must index the {n_teams} teams of the phase dump (or be -1), "
            f"got range [{int(team.min())}, {int(team.max())}]"
        )
    edges = pm["edges_player"]
    if not isinstance(edges, Mapping):
        raise ValueError(f"player match {mid}: edges_player must be a dict type -> (idx, w)")
    ignored = []
    for name, item in edges.items():
        if name not in PLAYER_EDGE_TYPES:
            ignored.append(str(name))
            continue
        idx, w = torch.as_tensor(item[0]), torch.as_tensor(item[1])
        if idx.ndim != 2 or idx.shape[1] != 3 or w.ndim != 1 or idx.shape[0] != w.shape[0]:
            raise ValueError(
                f"player match {mid}: edges '{name}' must be idx [E, 3] and w [E], got {tuple(idx.shape)} / {tuple(w.shape)}"
            )
        if idx.shape[0]:
            idx = idx.to(torch.int64)
            if (
                int(idx[:, 0].min()) < 0
                or int(idx[:, 0].max()) >= n_phases
                or int(idx[:, 1:].min()) < 0
                or int(idx[:, 1:].max()) >= n_players
            ):
                raise ValueError(
                    f"player match {mid}: edges '{name}' index outside [phase < {n_phases}, player < {n_players}]"
                )
    return ignored


def load_player_cohort(
    player_dir: str | Path,
    cohort: str,
    phase_splits: Mapping[str, Mapping],
    splits: Iterable[str] = ("train", "val", "test"),
    labels: str = "v2",
) -> tuple[dict[str, dict[str, Mapping]], dict]:
    """Load ``<player_dir>/<cohort>/<split>.pt`` and join it to the phase dumps by ``match_id``.

    ``phase_splits``: ``{split: phase dump}`` (as returned by ``data.load_cohort``).  Every phase match needs a player
    match (an incomplete player dump fails with the first missing ids); player matches that are not in the phase dump
    (population filter, subsample) are ignored.  Returns ``({split: {match_id: player match}}, info)`` where ``info``
    holds the feature names, the number of matches joined / ignored per split, the ignored edge types and, per split, the
    edge endpoint report.  Every ``teammate`` / ``enemy_prox`` endpoint must be a live token (``alive_players`` under the
    team alive rule of ``labels``), else ``ValueError``; ``combat`` endpoints are only reported."""
    import pubg_lab.survival.ptr.data.data as D  # lazy: data imports this module

    base = Path(player_dir) / cohort
    if not base.is_dir():
        raise FileNotFoundError(
            f"player dump for cohort {cohort} not found in {player_dir} (stage 2 needs {base})"
        )
    out: dict[str, dict[str, Mapping]] = {}
    names: list[str] | None = None
    info: dict = {"per_split": {}, "ignored_edge_types": []}
    ignored: set[str] = set()
    for split in splits:
        dump = load_player_dump(base / f"{split}.pt")
        these = list(dump["meta"]["player_feature_names"])
        if names is None:
            names = these
        elif these != names:
            raise ValueError(
                f"{cohort}: player_feature_names of {split} differ from the other splits"
            )
        by_id = {str(m["match_id"]): m for m in dump["matches"]}
        if len(by_id) != len(dump["matches"]):
            raise ValueError(f"{cohort}/{split}: duplicated match ids in the player dump")
        wanted = [str(m["match_id"]) for m in phase_splits[split]["matches"]]
        absent = [i for i in wanted if i not in by_id]
        if absent:
            raise ValueError(
                f"{cohort}/{split}: {len(absent)} of {len(wanted)} matches of the phase dump have no player match "
                f"(first: {absent[:3]}); the player dump is incomplete"
            )
        joined = {}
        endpoints: dict[str, list[int]] = {t: [0, 0] for t in PLAYER_EDGE_TYPES}
        event = D.uses_event_alive(phase_splits[split], labels)
        for m in phase_splits[split]["matches"]:
            pm = by_id[str(m["match_id"])]
            n_phases = int(torch.as_tensor(m["X"]).shape[0])
            ignored.update(validate_player_match(pm, len(m["team_ids"]), n_phases, len(names)))
            live = alive_players(D.effective_alive(m, labels, event), pm)
            for t, (n_e, n_bad) in assert_edges_on_live_tokens(pm, live).items():
                endpoints[t][0] += n_e
                endpoints[t][1] += n_bad
            joined[str(m["match_id"])] = pm
        out[split] = joined
        info["per_split"][split] = {
            "joined": len(joined), "player_matches_in_dump": len(by_id), "ignored": len(by_id) - len(joined),
            "edge_endpoints_not_live": endpoints,
        }  # fmt: skip
    info["feature_names"] = names
    info["ignored_edge_types"] = sorted(ignored)
    return out, info


# --------------------------------------------------------------------------- alive players and relation tensors
def alive_players(team_alive: torch.Tensor, pm: Mapping) -> torch.Tensor:
    """``[P_slots, P]`` bool: the causal ``player_alive_cut`` flag AND the player's team is alive at that slot.

    ``player_node_mask`` is never used (see the module docstring).  A player of a dead team (a ghost, or a team the phase
    dump does not carry: ``player_team_idx`` = -1) is never alive, so dead teams cannot reach the alive ones through
    the player layer.  A team that is alive but has no alive player at the cut simply has no player tokens (its team
    token is the stage-1 embedding alone)."""
    mask = torch.as_tensor(pm[PLAYER_ALIVE_FIELD]).bool()
    team = torch.as_tensor(pm["player_team_idx"]).to(torch.int64)
    valid = team >= 0
    gathered = team_alive[:, team.clamp(min=0)] & valid[None, :]
    return mask & gathered


def edge_endpoint_report(pm: Mapping, live: torch.Tensor) -> dict[str, list[int]]:
    """``{edge type: [edges, edges with an endpoint that is not a live token]}`` (``live [P_slots, P]``)."""
    out: dict[str, list[int]] = {}
    for name, item in pm["edges_player"].items():
        if name not in PLAYER_EDGE_TYPES:
            continue
        idx = torch.as_tensor(item[0]).to(torch.int64).reshape(-1, 3)
        if idx.shape[0] == 0:
            out[name] = [0, 0]
            continue
        ok = live[idx[:, 0], idx[:, 1]] & live[idx[:, 0], idx[:, 2]]
        out[name] = [int(idx.shape[0]), int((~ok).sum())]
    return out


def assert_edges_on_live_tokens(
    pm: Mapping, live: torch.Tensor, strict_types: Sequence[str] = ENDPOINT_STRICT_TYPES
) -> dict[str, list[int]]:
    """Every endpoint of a ``strict_types`` edge (teammate, enemy_prox: built from alive tokens only) must be a live token;
    raises ``ValueError`` otherwise.  Returns :func:`edge_endpoint_report` (combat edges are reported, not asserted: their
    victim is typically dead at the cut by construction; the pair mask of the attention drops those edges)."""
    report = edge_endpoint_report(pm, live)
    bad = {t: report[t] for t in strict_types if t in report and report[t][1]}
    if bad:
        raise ValueError(
            f"player match {pm.get('match_id', '?')}: edges with an endpoint that is not a live token (alive at the cut and in an "
            f"alive team): {{type: [edges, not live]}} = {bad}"
        )
    return report


def infer_player_combat_weights(player_matches: Iterable[Mapping]) -> tuple[str, float]:
    """``("log", max_w)`` when every player combat weight is <= :data:`PLAYER_COMBAT_LOG_MAX`, else ``("raw", max_w)``."""
    top = 0.0
    for pm in player_matches:
        item = pm["edges_player"].get("combat")
        if item is None:
            continue
        w = torch.as_tensor(item[1], dtype=torch.float32)
        w = w[torch.isfinite(w)]
        if w.numel():
            top = max(top, float(w.max()))
    return ("log" if top <= PLAYER_COMBAT_LOG_MAX else "raw"), top


def densify_player_edges_batch(
    edges_list: Sequence[Mapping | None], n_phases: int, n_max: int
) -> dict[str, torch.Tensor]:
    """Sparse player edges -> dense ``[B, P, N, N]`` (zero padded to ``n_max`` players); see
    :func:`model_stgcn.ptr.relations.densify_edges_batch` (same rules: duplicates collapse with ``max``, self loops
    dropped, non-finite / negative weights 0, symmetric types symmetrised with ``max``)."""
    return R.densify_edges_batch(
        edges_list,
        n_phases,
        n_max,
        edge_types=PLAYER_EDGE_TYPES,
        symmetric=PLAYER_SYMMETRIC_EDGE_TYPES,
    )


def build_player_relation_tensor(
    dense: Mapping[str, torch.Tensor],
    relation_types: Sequence[str] = PLAYER_RELATION_TYPES,
    combat_weights: str = "raw",
) -> torch.Tensor:
    """Dense matrices -> ``[..., R, N, N]`` player relation weights (escalation applied, no row normalisation)."""
    types = check_player_relation_types(relation_types)
    if combat_weights not in R.COMBAT_WEIGHT_MODES:
        raise ValueError(
            f"combat_weights must be one of {R.COMBAT_WEIGHT_MODES}, got '{combat_weights}'"
        )
    exists = dense["combat_exists"]
    prox = dense["enemy_prox"]
    if "combat_out" in types or "combat_in" in types:
        prox = prox.masked_fill(exists | exists.transpose(-1, -2), 0.0)
    combat = dense["combat"].clamp(min=0.0)
    combat_out = torch.log1p(combat) if combat_weights == "raw" else combat
    mats = {
        "teammate": dense["teammate"],
        "enemy_prox": prox,
        "combat_out": combat_out,
        "combat_in": combat_out.transpose(-1, -2),
    }
    return torch.stack([mats[t] for t in types], dim=-3)


def player_summary(
    player_matches: Iterable[Mapping], team_alive_by_match: Iterable[torch.Tensor]
) -> dict:
    """Counts of what the player layer sees (alive player tokens, edges per type) for ``metrics.json``."""
    n_matches = n_alive = n_players = 0
    edges = {t: 0 for t in PLAYER_EDGE_TYPES}
    for pm, team_alive in zip(player_matches, team_alive_by_match):
        n_matches += 1
        n_players += int(torch.as_tensor(pm["Xp"]).shape[1])
        n_alive += int(alive_players(team_alive, pm).sum())
        for t in PLAYER_EDGE_TYPES:
            item = pm["edges_player"].get(t)
            if item is not None:
                edges[t] += int(torch.as_tensor(item[1]).shape[0])
    return {
        "n_matches": n_matches,
        "n_players": n_players,
        "n_alive_player_tokens": n_alive,
        "n_edges": edges,
    }


# --------------------------------------------------------------------------- command line: check a player dump
def check_cohort(
    player_dir: str | Path, phase_dir: str | Path, cohort: str, labels: str = "v2"
) -> dict:
    """Join a cohort's player dump to its phase dump and report what the player layer would see.

    Loads every split (``data.load_cohort`` for the phase side).  Reports: matches joined, feature names, ignored edge
    types, alive team-slots without an alive player (``teams_without_players``, mostly winners with a finite death
    time or teams whose players are all dead at the cut), alive_cut players of dead teams (``players_in_dead_teams``), NaN
    share of the alive player features, the weight range per edge type and the edge endpoints that are not live tokens."""
    import pubg_lab.survival.ptr.data.data as D

    splits = D.load_cohort(phase_dir, cohort, labels=labels)
    players, info = load_player_cohort(player_dir, cohort, splits, labels=labels)
    rep: dict = {
        "cohort": cohort,
        "feature_names": info["feature_names"],
        "ignored_edge_types": info["ignored_edge_types"],
        "splits": {},
    }
    for split, dump in splits.items():
        event = D.uses_event_alive(dump, labels)
        stat = {"matches": len(dump["matches"]), "alive_team_slots": 0, "teams_without_players": 0, "players_in_dead_teams": 0,
                "alive_player_tokens": 0, "nan_cells": 0, "cells": 0, "n_players": 0, "weights": {}}  # fmt: skip
        for m in dump["matches"]:
            pm = players[split][str(m["match_id"])]
            alive = D.effective_alive(m, labels, event)
            pa = alive_players(alive, pm)
            raw = torch.as_tensor(pm[PLAYER_ALIVE_FIELD]).bool()
            team = torch.as_tensor(pm["player_team_idx"]).long()
            stat["players_in_dead_teams"] += int((raw & ~pa).sum())
            if "player_node_mask" in pm:  # information only: the leak the causal flag avoids
                stat["node_mask_only_tokens"] = stat.get("node_mask_only_tokens", 0) + int(
                    (torch.as_tensor(pm["player_node_mask"]).bool() & ~raw).sum()
                )
            stat["alive_player_tokens"] += int(pa.sum())
            stat["n_players"] += int(raw.shape[1])
            xp = torch.as_tensor(pm["Xp"])[pa]
            stat["nan_cells"] += int((~torch.isfinite(xp)).sum())
            stat["cells"] += int(xp.numel())
            has = torch.zeros_like(alive)
            for i in range(alive.shape[1]):
                has[:, i] = pa[:, team == i].any(dim=1) if bool((team == i).any()) else False
            stat["alive_team_slots"] += int(alive.sum())
            stat["teams_without_players"] += int((alive & ~has).sum())
            for name, item in pm["edges_player"].items():
                w = torch.as_tensor(item[1], dtype=torch.float32)
                if w.numel() and name in PLAYER_EDGE_TYPES:
                    lo, hi = stat["weights"].get(name, (float("inf"), float("-inf")))
                    stat["weights"][name] = (min(lo, float(w.min())), max(hi, float(w.max())))
        stat["edge_endpoints_not_live"] = info["per_split"][split]["edge_endpoints_not_live"]
        rep["splits"][split] = stat
    return rep


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(
        prog="python -m model_stgcn.ptr.player",
        description="Check a player dump against the phase dump.",
    )
    ap.add_argument("--player-dir", required=True)
    ap.add_argument("--dump-dir", required=True, help="the phase dump directory")
    ap.add_argument(
        "--cohorts",
        default=None,
        help="comma separated; default: every cohort directory of the phase dump",
    )
    ap.add_argument("--labels", default="v2")
    args = ap.parse_args(argv)
    from pubg_lab.survival.ptr.data.cohorts import discover_cohorts

    cohorts = (
        [c for c in args.cohorts.split(",") if c]
        if args.cohorts
        else discover_cohorts(args.dump_dir)
    )
    failed = 0
    for c in cohorts:
        try:
            rep = check_cohort(args.player_dir, args.dump_dir, c, args.labels)
        except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
            failed += 1
            print(f"{c}: FAILED: {exc}", file=sys.stderr)
            continue
        tr = rep["splits"]["train"]
        print(
            f"{c}: ok, {tr['matches']} train matches, {len(rep['feature_names'])} features, ignored edge types {rep['ignored_edge_types']}; "
            f"train alive team-slots {tr['alive_team_slots']} ({tr['teams_without_players']} without an alive player), "
            f"alive_cut players of dead teams {tr['players_in_dead_teams']}, NaN share {tr['nan_cells'] / max(tr['cells'], 1):.4f}, "
            f"weights {json.dumps({k: [round(v[0], 4), round(v[1], 4)] for k, v in tr['weights'].items()})}, "
            f"edges with a non-live endpoint {json.dumps(tr['edge_endpoints_not_live'])}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
