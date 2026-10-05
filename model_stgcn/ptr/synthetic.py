"""Synthetic PTR dumps (DB-free) following the cohort dump schema (design doc v2.3).

One dump is a ``torch.save``-able dict::

    {"meta":    {"feature_names": [F names], "feature_groups": {name: [cols]},
                 "cohort": str, "split": str, "zone_columns": [8 names]},
     "matches": [ {match_id, match_type, team_ids[N], X[P,N,F], node_mask[P,N],
                   target_mask[P,N], y_hazard[P,N],            # pipeline labels (team_phase_labels)
                   target_mask_v2[P,N], y_hazard_v2[P,N],      # event-based labels v2
                   final_rank_norm[N], top5[N], top10[N], team_ai_frac[N],
                   team_won[N], team_death_s[N], phase_start_s[P], phase_end_s[P],
                   edges{"prox"|"combat"|"copress": (idx[E,3]=(p,i,j), w[E])},
                   team_pos[P,N,2], zone[P,8], zone_valid[P]} ... ]}

The generator plants a learnable signal so that tiny trainings are meaningful:

* column 0 of ``X`` is lowered when the team dies in the next phase (current-state signal),
* column 1 of ``X`` at slot ``q`` is lowered when the team dies in phase ``q + 2``
  (visible one slot early, i.e. only a phase-axis model or a lag feature can use it),
* column 2 of ``X`` is raised for teams that finish better (``rank_signal``): a rank-direction probe
  (survival-like score: higher = better placement = smaller ``final_rank_norm``).

Combat edge weights follow the real dump by default (``combat_weights="log"``: ``ln(1 + intensity)`` as
``stdb/phase_edge_etl.py`` stores it); ``"raw"`` writes the raw intensity ``damage + 100*kills + 50*groggy``.

Elimination convention: ``elim[i]`` is the phase slot in which team ``i`` is eliminated; the winner (rank 1,
``team_won``) never is.  Alive at start of phase ``p`` iff ``p <= elim``; alive at end iff ``p < elim``; target
at cut ``p`` iff alive at end and ``p`` is not the last slot; ``hazard = (elim == p + 1)``.  These are the
**v2 (truth) labels** (``target_mask_v2`` / ``y_hazard_v2``): with ``team_death_s`` a death time on the game
clock inside the elimination phase (NaN for the winner) and ``phase_end_s`` they read exactly
``T > phase_end_s[p]`` and ``y = T <= phase_end_s[p + 1]`` of design doc section 4.2.  ``final_rank`` 1 is the
winner, ``final_rank_norm = rank / N``.

The **pipeline labels** (``target_mask`` / ``y_hazard``, what ``team_phase_labels`` holds) are a corrupted copy
controlled by ``pipeline_noise`` (0 = identical to v2): v2 targets the pipeline misses (for half of them the
token is missing from ``node_mask`` too), "ghost" targets (the team died during the phase but the pipeline still
reads it alive; random label), flipped labels, and **ghost tokens** (teams that are already dead at the phase
start but that the pipeline ``node_mask`` still reads as alive).  As in the real dump ``node_mask`` is raised
over every pipeline target, so ``target_mask <= node_mask`` always holds for the pipeline labels, whereas
``target_mask_v2`` can sit on a token that ``node_mask`` does not cover.

Event-based death times (``event_death=True``, the default, like the new dump): ``team_death_s`` is the last-kill
time on the clock of ``phase_start_s`` / ``phase_end_s`` (+inf for the winner; ``phase_mask`` is all True) and, as
the dump script does, each match carries its own ``node_mask_v2 = known & (team_death_s > phase_start_s)``
(``event_marker='field'``; ``'flag'`` marks the dump with ``meta['team_death_source'] = 'last_kill_event'``
instead, ``'both'`` does both).  About 5 % of the deaths fall into the 10 s gap between ``phase_end_s[k-1]`` and
``phase_start_s[k]``, where that mask and the loader's ``team_death_s > phase_end_s[k-1]`` rule differ.
``event_death=False`` writes a legacy dump (NaN for the winner, no marker, no ``node_mask_v2``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from model_stgcn.ptr.data import ZONE_COLUMNS

GROUP_FRACTIONS = (("base", 0.60), ("network", 0.15), ("zone", 0.125), ("habitat", 0.125))


def group_columns(n_features: int) -> dict[str, list[int]]:
    """Contiguous column blocks; every group gets at least one column."""
    if n_features < len(GROUP_FRACTIONS):
        raise ValueError(f"n_features must be >= {len(GROUP_FRACTIONS)}")
    sizes = [max(1, int(round(frac * n_features))) for _, frac in GROUP_FRACTIONS]
    sizes[0] += n_features - sum(sizes)  # absorb rounding in the base block
    if sizes[0] < 1:
        raise ValueError(f"cannot split {n_features} features into {len(sizes)} groups")
    out: dict[str, list[int]] = {}
    start = 0
    for (name, _), size in zip(GROUP_FRACTIONS, sizes):
        out[name] = list(range(start, start + size))
        start += size
    return out


def _pair_edges(
    rng: np.random.Generator,
    alive: np.ndarray,
    p: int,
    prob: float,
    *,
    symmetric: bool,
    both_directions: bool,
    weight_fn,
) -> tuple[list[tuple[int, int, int]], list[float]]:
    idx, w = [], []
    ids = np.flatnonzero(alive)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            if rng.random() >= prob:
                continue
            i, j = int(ids[a]), int(ids[b])
            wt = float(weight_fn())
            if symmetric:
                idx.append((p, i, j))
                w.append(wt)
                if both_directions:
                    idx.append((p, j, i))
                    w.append(wt)
            else:  # directed: random orientation attacker -> victim
                if rng.random() < 0.5:
                    i, j = j, i
                idx.append((p, i, j))
                w.append(wt)
    return idx, w


def _edge_tensors(idx: list, w: list) -> tuple[torch.Tensor, torch.Tensor]:
    if not idx:
        return torch.zeros((0, 3), dtype=torch.int64), torch.zeros((0,), dtype=torch.float32)
    return torch.tensor(idx, dtype=torch.int64), torch.tensor(w, dtype=torch.float32)


def _pipeline_labels(
    rng: np.random.Generator,
    alive_start: np.ndarray,
    target: np.ndarray,
    y: np.ndarray,
    elim: np.ndarray,
    ph: np.ndarray,
    n_phases: int,
    noise: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Corrupted copy of the (v2) truth: ``(node_mask, target_mask, y_hazard)`` as the pipeline made them.

    ``node_mask`` also holds ghost tokens: teams already dead at the phase start that stale snapshots keep alive."""
    if noise <= 0:
        return alive_start.copy(), target.copy(), y.copy()
    shape = target.shape
    missing = target & (rng.random(shape) < noise)  # v2 targets the pipeline does not have
    token_missing = missing & (rng.random(shape) < 0.5)  # ... half of them lose their token as well
    ghost = (elim[None, :] == ph) & (ph <= n_phases - 2) & (rng.random(shape) < noise)  # died this phase
    target_p = (target & ~missing) | ghost
    flip = target & ~missing & (rng.random(shape) < noise)
    y_p = np.where(flip, 1.0 - y, y)
    y_p = np.where(ghost, (rng.random(shape) < 0.5).astype(np.float32), y_p)
    y_p = np.where(target_p, y_p, 0.0).astype(np.float32)
    ghost_token = ~alive_start & (rng.random(shape) < noise)  # dead before the phase, still "alive" in the table
    node_p = (alive_start & ~token_missing) | ghost_token | target_p  # the dump raises node_mask over targets
    return node_p, target_p, y_p


def _make_match(
    rng: np.random.Generator,
    match_id: str,
    *,
    n_teams: int,
    n_phases: int,
    n_features: int,
    signal: float,
    lag_signal: float,
    rank_signal: float,
    combat_weights: str,
    nan_frac: float,
    dead_garbage: bool,
    both_directions: bool,
    zone_missing_prob: float,
    edge_prob: float,
    match_type: str,
    pipeline_noise: float,
    v2_fields: bool,
    ai_nan_frac: float,
    event_death: bool,
    event_marker: str,
) -> dict:
    P, N, F = n_phases, n_teams, n_features
    team_ids = np.sort(rng.choice(100_000, size=N, replace=False)).astype(np.int64)
    elim = rng.integers(0, P + 3, size=N)
    order = np.lexsort((rng.random(N), -elim))  # elim descending, random tie break
    rank = np.empty(N, dtype=np.int64)
    rank[order] = np.arange(1, N + 1)
    frn = (rank / N).astype(np.float32)
    winner = int(order[0])
    death_phase = elim.copy()  # for the death time of the others (before the winner is made immortal)
    elim[winner] = 10**6  # the winner never dies (censored)

    ph = np.arange(P)[:, None]
    alive_start = ph <= elim[None, :]
    alive_end = ph < elim[None, :]
    target = alive_end.copy()
    target[P - 1] = False
    y = ((elim[None, :] == ph + 1) & target).astype(np.float32)

    X = rng.standard_normal((P, N, F)).astype(np.float32)
    X[..., 0] -= signal * (elim[None, :] - ph == 1)
    # lag signal: slot q announces death in phase q + 2
    X[..., 1] -= lag_signal * (elim[None, :] - ph == 2)
    X[..., 2] += rank_signal * (0.5 - frn)[None, :]  # better placement (small frn) -> larger value
    if nan_frac > 0:
        X[rng.random(X.shape) < nan_frac] = np.nan
    if dead_garbage:
        dead = ~alive_start
        X[dead] = (50.0 * rng.standard_normal((int(dead.sum()), F))).astype(np.float32)

    pos = rng.random((P, N, 2)).astype(np.float32)
    zone = np.zeros((P, len(ZONE_COLUMNS)), dtype=np.float32)
    centers = 0.5 + 0.1 * rng.standard_normal((P + 1, 2))
    radii = 0.45 * 0.65 ** np.arange(P + 1)
    for p in range(P):
        zone[p] = [
            centers[p, 0], centers[p, 1], radii[p],
            centers[p + 1, 0], centers[p + 1, 1], radii[p + 1],
            (p + 0.5) / P, p + 1,
        ]  # fmt: skip
    zone_valid = rng.random(P) >= zone_missing_prob

    prox_i, prox_w, com_i, com_w, cop_i, cop_w = [], [], [], [], [], []
    for p in range(P):
        alive = alive_start[p]
        a, b = _pair_edges(
            rng, alive, p, edge_prob * 2, symmetric=True, both_directions=both_directions,
            weight_fn=lambda: np.exp(-rng.uniform(0, 800) / 200.0),
        )  # fmt: skip
        prox_i += a
        prox_w += b
        intensity = lambda: rng.exponential(120.0) + 100.0 * rng.integers(0, 2)  # noqa: E731
        a, b = _pair_edges(
            rng, alive, p, edge_prob, symmetric=False, both_directions=False,
            weight_fn=(lambda: np.log1p(intensity())) if combat_weights == "log" else intensity,
        )  # fmt: skip
        com_i += a
        com_w += b
        a, b = _pair_edges(
            rng, alive, p, edge_prob, symmetric=True, both_directions=both_directions,
            weight_fn=lambda: rng.uniform(0.05, 1.0),
        )  # fmt: skip
        cop_i += a
        cop_w += b

    node_p, target_p, y_p = _pipeline_labels(rng, alive_start, target, y, elim, ph, P, pipeline_noise)

    phase_start = (90.0 + 200.0 * np.arange(P)).astype(np.float32)
    phase_end = phase_start + 190.0  # last grid point of the phase
    # death time inside the elimination phase; ~5 % fall into the 10 s gap before its first grid point
    death = 90.0 + 200.0 * death_phase + rng.uniform(-9.5, 190.0, size=N)
    death[winner] = np.inf if event_death else np.nan  # censored: the winning team never dies
    known = ~np.isnan(death)
    node_v2 = known[None, :] & (death[None, :] > phase_start[:, None].astype(np.float64))  # the dump's own rule
    team_ai = rng.integers(0, 5, size=N) / 4.0
    team_ai[rng.random(N) < ai_nan_frac] = np.nan  # no participants -> NaN

    match = {
        "match_id": match_id,
        "match_type": match_type,
        "team_ids": torch.from_numpy(team_ids),
        "X": torch.from_numpy(X),
        "node_mask": torch.from_numpy(node_p),
        "target_mask": torch.from_numpy(target_p),
        "y_hazard": torch.from_numpy(y_p),
        "final_rank_norm": torch.from_numpy(frn),
        "top5": torch.from_numpy((rank <= 5).astype(np.float32)),
        "top10": torch.from_numpy((rank <= 10).astype(np.float32)),
        "team_ai_frac": torch.from_numpy(team_ai.astype(np.float32)),
        "team_won": torch.from_numpy(rank == 1),
        "team_death_s": torch.from_numpy(death.astype(np.float32)),
        "phase_start_s": torch.from_numpy(phase_start),
        "phase_end_s": torch.from_numpy(phase_end),
        "phase_mask": torch.ones(P, dtype=torch.bool),
        "edges": {
            "prox": _edge_tensors(prox_i, prox_w),
            "combat": _edge_tensors(com_i, com_w),
            "copress": _edge_tensors(cop_i, cop_w),
        },
        "team_pos": torch.from_numpy(pos),
        "zone": torch.from_numpy(zone),
        "zone_valid": torch.from_numpy(zone_valid),
    }
    if v2_fields:
        match["target_mask_v2"] = torch.from_numpy(target)
        match["y_hazard_v2"] = torch.from_numpy(y)
        if event_death and event_marker in ("field", "both"):
            match["node_mask_v2"] = torch.from_numpy(node_v2)
    return match


def make_dump(
    n_matches: int = 8,
    *,
    n_phases: int = 8,
    n_features: int = 24,
    team_range: tuple[int, int] = (6, 12),
    seed: int = 0,
    cohort: str = "synthetic",
    split: str = "train",
    signal: float = 1.5,
    lag_signal: float = 0.0,
    rank_signal: float = 0.0,
    combat_weights: str = "log",
    nan_frac: float = 0.0,
    dead_garbage: bool = True,
    both_directions: bool = True,
    zone_missing_prob: float = 0.0,
    edge_prob: float = 0.15,
    alive_count_names: bool = False,
    pipeline_noise: float = 0.15,
    v2_fields: bool = True,
    ai_nan_frac: float = 0.0,
    match_types: tuple[str, ...] = ("official", "competitive"),
    event_death: bool = True,
    event_marker: str = "field",
) -> dict:
    """Build one in-memory dump (``meta`` + ``matches``) that follows the schema.

    ``pipeline_noise`` corrupts the pipeline labels relative to the v2 truth (0 = identical);
    ``v2_fields=False`` writes a legacy dump without ``y_hazard_v2`` / ``target_mask_v2``;
    ``ai_nan_frac`` makes that share of ``team_ai_frac`` NaN (a team without participants);
    match ``m`` gets ``match_types[m % len(match_types)]``; ``event_death`` / ``event_marker`` select the
    event-based (new) or legacy death-time format, see the module docstring.

    ``alive_count_names=True`` renames the first three ``zone`` and ``habitat`` columns like the real feature
    layout, so that the noHab alive-count exception has something to act on: zone = [avg_alive_teams_norm,
    avg_alive_players_norm, safe_radius_start_norm], habitat = [hab_alive_teams,
    hab_safe_area_per_alive_team, hab_loss_ratio].
    """
    if combat_weights not in ("raw", "log"):
        raise ValueError("combat_weights must be 'raw' or 'log'")
    if event_marker not in ("field", "flag", "both"):
        raise ValueError("event_marker must be 'field', 'flag' or 'both'")
    rng = np.random.default_rng(seed)
    groups = group_columns(n_features)
    names = [""] * n_features
    for g, cols in groups.items():
        for k, c in enumerate(cols):
            names[c] = f"{g}_{k}"
    if alive_count_names:
        zone, hab = groups["zone"], groups["habitat"]
        if len(zone) < 3 or len(hab) < 3:
            raise ValueError("alive_count_names needs >= 3 zone and habitat columns (use n_features >= 24)")
        for col, name in zip(
            (*zone[:3], *hab[:3]),
            ("avg_alive_teams_norm", "avg_alive_players_norm", "safe_radius_start_norm",
             "hab_alive_teams", "hab_safe_area_per_alive_team", "hab_loss_ratio"),
        ):  # fmt: skip
            names[col] = name
    matches = []
    for m in range(n_matches):
        n_teams = int(rng.integers(team_range[0], team_range[1] + 1))
        matches.append(
            _make_match(
                rng,
                f"{cohort}-{split}-{m:04d}",
                n_teams=n_teams,
                n_phases=n_phases,
                n_features=n_features,
                signal=signal,
                lag_signal=lag_signal,
                rank_signal=rank_signal,
                combat_weights=combat_weights,
                nan_frac=nan_frac,
                dead_garbage=dead_garbage,
                both_directions=both_directions,
                zone_missing_prob=zone_missing_prob,
                edge_prob=edge_prob,
                match_type=match_types[m % len(match_types)],
                pipeline_noise=pipeline_noise,
                v2_fields=v2_fields,
                ai_nan_frac=ai_nan_frac,
                event_death=event_death,
                event_marker=event_marker,
            )
        )
    meta = {
        "feature_names": names,
        "feature_groups": groups,
        "cohort": cohort,
        "split": split,
        "zone_columns": list(ZONE_COLUMNS),
    }
    if event_death and event_marker in ("flag", "both"):
        meta["team_death_source"] = "last_kill_event"
    return {"meta": meta, "matches": matches}


def write_cohort_dumps(
    root: str | Path,
    cohort: str = "synthetic_cohort",
    *,
    sizes: dict[str, int] | None = None,
    seed: int = 0,
    **kwargs,
) -> Path:
    """Write ``root/<cohort>/{train,val,test}.pt`` (+ a ``dump_manifest.json`` like the dump script's) and
    return the cohort directory."""
    sizes = sizes or {"train": 16, "val": 6, "test": 6}
    out = Path(root) / cohort
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"cohort": cohort, "splits": {}}
    for k, (split, n) in enumerate(sizes.items()):
        dump = make_dump(n, seed=seed * 1000 + k, cohort=cohort, split=split, **kwargs)
        path = out / f"{split}.pt"
        torch.save(dump, path)
        manifest["splits"][split] = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "n_split_ids": n,
            "n_dumped": n,
            "n_skipped": 0,
        }
    (out / "dump_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


# =========================================================================== synthetic *player* dumps (stage 2)
def make_player_dump(
    phase_dump: dict,
    *,
    seed: int = 0,
    n_player_features: int = 6,
    team_size: tuple[int, int] = (1, 4),
    signal: float = 0.0,
    combat_weights: str = "log",
    edge_prob: float = 0.06,
    nan_frac: float = 0.0,
    dead_garbage: bool = True,
    both_directions: bool = True,
    ai_frac: float = 0.2,
    gap_prob: float = 0.15,
    revive_prob: float = 0.03,
) -> dict:
    """A player dump (see :mod:`model_stgcn.ptr.player`) for every match of a synthetic phase dump.

    Every team gets ``team_size`` players (ids ``acct.<k>`` strings, as in the real dump).  The player alive flags mimic
    the real dump: ``player_node_mask`` (event based, alive at the phase start: a team that is alive at a slot has
    at least one such player, a player stays dead once dead) and the causal ``player_alive_cut`` (what the model uses): a
    node-alive player is dead at the cut with probability ``gap_prob`` (killed inside the phase, later "recalled": the
    node mask keeps it, the cut flag does not), and a node-dead player is alive at the cut with probability
    ``revive_prob``.  Dead players carry garbage (``dead_garbage``: huge values and NaN) at every cell where the cut flag
    is False, so that any leak shows.  ``signal`` plants ``Xp[..., 0] = signal * y_hazard_v2[p, team] + noise`` for
    cut-alive players, a hazard cue that exists *only* at the player level.  Edges: teammate and enemy-proximity pairs
    among live tokens (cut-alive players of alive teams) and directed ``combat`` edges whose victim may be dead at the cut
    (as in the real dump, where about 70 % of the combat edges end on a player that was killed in the window)."""
    if combat_weights not in ("raw", "log"):
        raise ValueError("combat_weights must be 'raw' or 'log'")
    rng = np.random.default_rng(seed)
    meta = {
        "player_feature_names": [f"player_{k}" for k in range(n_player_features)],
        "cohort": phase_dump["meta"].get("cohort", ""),
        "split": phase_dump["meta"].get("split", ""),
    }
    matches = []
    for m in phase_dump["matches"]:
        n_teams = len(m["team_ids"])
        team_node = (m["node_mask_v2"] if "node_mask_v2" in m else m["node_mask"]).numpy().astype(bool)  # [P, N]
        n_phases = team_node.shape[0]
        sizes = rng.integers(team_size[0], team_size[1] + 1, size=n_teams)
        team_idx = np.repeat(np.arange(n_teams), sizes)
        n_players = int(sizes.sum())
        node = np.zeros((n_phases, n_players), dtype=bool)  # event based, alive at the phase start
        start = 0
        for i in range(n_teams):
            slots = np.flatnonzero(team_node[:, i])
            for k in range(int(sizes[i])):
                if slots.size:
                    last = int(slots.max()) if k == 0 else int(rng.integers(int(slots.min()), int(slots.max()) + 1))
                    node[: last + 1, start + k] = team_node[: last + 1, i]
            start += int(sizes[i])
        gap = rng.random(node.shape) < gap_prob
        revive = rng.random(node.shape) < revive_prob
        cut = (node & ~gap) | (~node & revive)  # the causal flag: differs from the node mask in both directions
        live = cut & team_node[:, team_idx]  # live token = alive at the cut and in an alive team
        xp = rng.normal(size=(n_phases, n_players, n_player_features)).astype(np.float32)
        if signal:
            y = m["y_hazard_v2"].numpy() if "y_hazard_v2" in m else m["y_hazard"].numpy()
            xp[..., 0] = signal * y[:, team_idx] + 0.3 * rng.normal(size=(n_phases, n_players))
        if nan_frac:
            xp[rng.random(xp.shape) < nan_frac] = np.nan
        if dead_garbage:
            dead = ~cut
            xp[dead] = np.where(rng.random(xp[dead].shape) < 0.5, 1e6, np.nan)
        idx = {"teammate": ([], []), "enemy_prox": ([], []), "combat": ([], [])}
        for p in range(n_phases):
            ids = np.flatnonzero(live[p])
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    i, j = int(ids[a]), int(ids[b])
                    if team_idx[i] == team_idx[j]:
                        wt = float(rng.uniform(0.3, 1.0))
                        idx["teammate"][0].append((p, i, j))
                        idx["teammate"][1].append(wt)
                        if both_directions:
                            idx["teammate"][0].append((p, j, i))
                            idx["teammate"][1].append(wt)
                    elif rng.random() < edge_prob:
                        wt = float(rng.uniform(0.1, 1.0))
                        idx["enemy_prox"][0].append((p, i, j))
                        idx["enemy_prox"][1].append(wt)
                        if both_directions:
                            idx["enemy_prox"][0].append((p, j, i))
                            idx["enemy_prox"][1].append(wt)
            starters = np.flatnonzero(node[p])  # combat: the victim may have been killed inside the window (dead at the cut)
            for a in ids:
                for v in starters:
                    if team_idx[a] != team_idx[v] and rng.random() < edge_prob / 3:
                        intensity = float(rng.uniform(20.0, 400.0))
                        idx["combat"][0].append((p, int(a), int(v)))
                        idx["combat"][1].append(float(np.log1p(intensity)) if combat_weights == "log" else intensity)
        matches.append({
            "match_id": m["match_id"],
            "player_ids": [f"acct.{k}" for k in range(n_players)],
            "player_team_idx": torch.from_numpy(team_idx.astype(np.int64)),
            "player_is_ai": torch.from_numpy(rng.random(n_players) < ai_frac),
            "Xp": torch.from_numpy(xp),
            "player_node_mask": torch.from_numpy(node),
            "player_alive_cut": torch.from_numpy(cut),
            "edges_player": {k: _edge_tensors(*v) for k, v in idx.items()},
        })  # fmt: skip
    return {"meta": meta, "matches": matches}


def write_player_dumps(
    phase_root: str | Path, cohort: str, player_root: str | Path, *, seed: int = 0, splits=("train", "val", "test"), **kwargs
) -> Path:
    """Write ``player_root/<cohort>/{split}.pt`` for the phase dumps ``phase_root/<cohort>/{split}.pt``; returns the directory."""
    out = Path(player_root) / cohort
    out.mkdir(parents=True, exist_ok=True)
    for k, split in enumerate(splits):
        phase = torch.load(Path(phase_root) / cohort / f"{split}.pt", map_location="cpu", weights_only=False)
        torch.save(make_player_dump(phase, seed=seed * 1000 + k, **kwargs), out / f"{split}.pt")
    return out


# =========================================================================== synthetic extra (kf_*) feature tables (union arms)
def make_extra_table(
    phase_dump: dict,
    *,
    seed: int = 0,
    n_features: int = 4,
    signal: float = 0.0,
    nan_frac: float = 0.0,
    junk_rows: bool = True,
    drop=None,
    labels: str = "v2",
    shuffle_columns: bool = False,
    label_columns: bool = True,
):
    """The table the KRAFTON-feature dump holds, for the matches of a synthetic phase dump: one row per (match_id,
    team_id, phase) for every alive token of a cut slot (the alive rule of ``labels``; cut phase ``k = p + 1`` for the
    slots ``p = 0 .. P - 2``), columns ``kf_signal, kf_1, ...``, the flag / label columns of the real table
    (``label_columns``: ``cut_t``, ``in_risk_set``, ``y_hazard_v2`` = the label, ``alive_at_cut``; none of them is a feature) and a
    column ``note`` that says why the row exists.

    ``signal``: ``kf_signal = signal * y_hazard + noise`` (a cue that only these features carry);  ``nan_frac`` makes that
    share of the feature cells NaN;  ``junk_rows`` adds rows nobody needs (dead teams, a cut after the last slot, an
    unknown team, an unknown match);  ``drop(match_id, team_id, phase) -> bool`` removes rows (to make gaps)."""
    import pandas as pd

    from model_stgcn.ptr import data as D

    rng = np.random.default_rng(seed)
    names = ["kf_signal", *[f"kf_{k}" for k in range(1, n_features)]]
    event = D.uses_event_alive(phase_dump, labels)
    keys, feats, risk, haz, notes, cuts = [], [], [], [], [], []

    def add(match_id, team_id, phase, values, in_risk, hazard, note, cut_t):
        keys.append((match_id, int(team_id), int(phase)))
        feats.append(values)
        risk.append(int(in_risk))
        haz.append(hazard)
        notes.append(note)
        cuts.append(float(cut_t))

    for m in phase_dump["matches"]:
        alive = D.effective_alive(m, labels, event).numpy()
        y, target = (a.numpy() for a in D.select_labels(m, labels))
        n_phases = alive.shape[0]
        team_ids = [int(t) for t in m["team_ids"]]
        mid = str(m["match_id"])
        ends = np.asarray(m["phase_end_s"], dtype=np.float64) if "phase_end_s" in m else np.full(n_phases, np.nan)

        def cut_t(p):  # the producer's cut time: ten seconds before the end of phase slot p (arbitrary when the dump has none)
            return ends[p] - 10.0 if np.isfinite(ends[p]) else 100.0 * (p + 1)

        for p in range(n_phases - 1):
            for i, t in enumerate(team_ids):
                if not alive[p, i]:
                    continue
                values = rng.normal(size=n_features)
                if signal:
                    values[0] = signal * float(y[p, i]) + 0.3 * rng.normal()  # y is 0 outside the risk set
                add(mid, t, p + 1, values, target[p, i], float(y[p, i]) if target[p, i] else np.nan, "ok", cut_t(p))
        if junk_rows:
            for p, i in np.argwhere(~alive[: n_phases - 1])[:3]:
                add(mid, team_ids[i], int(p) + 1, rng.normal(size=n_features), 0, np.nan, "dead team", cut_t(int(p)))
            add(mid, team_ids[0], n_phases, rng.normal(size=n_features), 0, np.nan, "after the last slot", cut_t(n_phases - 1))
            add(mid, 10_000_001, 1, rng.normal(size=n_features), 0, np.nan, "unknown team", cut_t(0))
    if junk_rows:
        add("no-such-match", 1, 1, rng.normal(size=n_features), 0, np.nan, "unknown match", 100.0)
    df = pd.DataFrame(keys, columns=["match_id", "team_id", "phase"])
    if label_columns:
        df["cut_t"] = cuts
        df["in_risk_set"] = risk
        df["y_hazard_v2"] = haz
        df["alive_at_cut"] = risk  # a token at risk is alive at the cut (the real table also has alive tokens that died before it)
    values = np.array(feats, dtype=np.float64).reshape(len(keys), n_features)
    if nan_frac:
        values[rng.random(values.shape) < nan_frac] = np.nan
    for j, name in enumerate(names):
        df[name] = values[:, j]
    df["note"] = notes
    if drop is not None:
        keep = [not drop(r.match_id, int(r.team_id), int(r.phase)) for r in df.itertuples(index=False)]
        df = df[keep].reset_index(drop=True)
    if shuffle_columns:
        df = df[["note", *reversed(names), *[c for c in df.columns if c not in names and c != "note"][::-1]]]
    return df


def write_extra_tables(
    phase_root, cohort: str, extra_root, *, fmt: str = "csv.gz", seed: int = 0, splits=("train", "val", "test"), transform=None, **kwargs
):
    """Write ``extra_root/<cohort>/<split>.<fmt>`` (``csv.gz`` | ``csv`` | ``parquet``) for the phase dumps of ``phase_root``
    (``**kwargs`` go to :func:`make_extra_table`, ``transform(df) -> df`` may rewrite a table before it is written); returns the
    cohort directory."""
    out = Path(extra_root) / cohort
    out.mkdir(parents=True, exist_ok=True)
    for k, split in enumerate(splits):
        phase = torch.load(Path(phase_root) / cohort / f"{split}.pt", map_location="cpu", weights_only=False)
        df = make_extra_table(phase, seed=seed * 1000 + k, **kwargs)
        if transform is not None:
            df = transform(df)
        path = out / f"{split}.{fmt}"
        if fmt == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
    return out


# =========================================================================== synthetic *prediction* runs (aggregator)
def write_prediction_runs(
    runs_dir: str | Path,
    cohorts: list[str],
    quality: dict[str, float],
    rank_quality: dict[str, float] | None = None,
    *,
    n_matches: int = 50,
    seed: int = 0,
    cohort_scale: dict[str, float] | None = None,
    base_quality: float = 0.9,
    seeds: dict[str, tuple[int, ...]] | None = None,
    near_copy: dict[str, str] | None = None,
    skip: set[tuple[str, str]] | None = None,
    labels: str = "v2",
    match_types: tuple[str, ...] = ("official", "official", "competitive"),
    n_train: dict[str, int] | None = None,
    phase_rate: tuple[float, float] | None = None,
    phase_slope: dict[str, float] | None = None,
) -> Path:
    """Write ``<runs_dir>/<cohort>/<arm>/seed<s>/{pred_test.csv, metrics.json}`` with *known* effects.

    No model is trained: per cohort one set of risk-set rows is drawn (matches of 8-13 teams, cut phases 1..6,
    ``y ~ Bernoulli(0.3)``, a final placement per team, one ``match_type`` per match), and every arm gets
    ``score = d' * (y - 0.5) + N(0, 1)`` so that its hazard AUC is exactly ``Phi(d' / sqrt(2))`` in the large-sample
    limit, with ``p_hazard = sigmoid(score)``.  The rank head is ``rank_score = q * (0.5 - final_rank_norm) + N(0, 1)``
    (survival-like: better placement, larger score).

    ``quality[arm]`` is d' of the arm; ``cohort_scale[cohort]`` multiplies the *effect* ``d' - base_quality`` in that
    cohort (heterogeneity between cohorts).  ``seeds[cohort]`` lists extra seeds (e.g. ``(13, 2026)``) that are
    written next to seed 42 for the arms M00, M11, M11R, M11R-shuf and M11RZ only, like the real matrix.
    ``near_copy`` maps an arm to the arm whose *hazard* scores it copies up to 1e-3 noise (a null hazard contrast
    with a tiny bootstrap standard error; the rank head still follows ``rank_quality``).  ``skip`` removes ``(cohort, arm)`` pairs.  Every ``metrics.json`` carries ``labels``
    and a placeholder config hash; ``n_train[cohort]`` adds ``n_matches.train`` to the ``metrics.json`` of the neural arms
    (the training size the meta-regression of the aggregator regresses on).

    ``phase_rate = (r1, r6)``: the hazard rate rises linearly with the cut phase from ``r1`` (phase 1) to ``r6`` (phase 6) instead of
    being 0.3 everywhere; ``phase_slope[arm] = g`` adds ``g * (phase - 3.5)`` to the arm's hazard score, i.e. the arm knows the phase
    baseline: it gains pooled AUC (it ranks late-phase rows above early-phase ones, which is right on average) but nothing within a phase.
    """
    import pandas as pd  # local: pandas is only needed by the tests / aggregator

    from model_stgcn.ptr import metrics as M

    rank_quality = rank_quality or quality
    cohort_scale = cohort_scale or {}
    seeds = seeds or {}
    near_copy = near_copy or {}
    skip = skip or set()
    phase_slope = phase_slope or {}
    seed_arms = {"M00", "M11", "M11R", "M11R-shuf", "M11RZ"}
    root = Path(runs_dir)
    for ci, cohort in enumerate(cohorts):
        rng = np.random.default_rng([seed, ci])
        rows = []
        for m in range(n_matches):
            n = int(rng.integers(8, 14))
            frn = (rng.permutation(n) + 1) / n
            mt = match_types[m % len(match_types)]
            for p in range(1, 7):
                teams = rng.choice(n, size=int(rng.integers(4, n + 1)), replace=False)
                rate = 0.3 if phase_rate is None else phase_rate[0] + (phase_rate[1] - phase_rate[0]) * (p - 1) / 5
                for t in teams:
                    rows.append((f"{cohort}-m{m:03d}", 100 + int(t), p, float(rng.random() < rate), float(frn[t]),
                                 float(frn[t] <= 5 / n), mt))  # fmt: skip
        base = pd.DataFrame(rows, columns=["match_id", "team_id", "phase", "y", "final_rank_norm", "top5", "match_type"])
        scale = cohort_scale.get(cohort, 1.0)

        def hazard_scores(arm: str, run_seed: int, _scale=scale, _base=base, _ci=ci):
            if arm in near_copy and near_copy[arm] != arm:  # same hazard scores up to 1e-3 noise
                src = hazard_scores(near_copy[arm], run_seed)
                g = np.random.default_rng([seed, _ci, run_seed, 77, sum(map(ord, arm))])  # deterministic (no hash())
                return src + 1e-3 * g.standard_normal(len(_base))
            g = np.random.default_rng([seed, _ci, run_seed, sum(map(ord, arm))])
            d = base_quality + _scale * (quality[arm] - base_quality)
            score = d * (_base["y"].to_numpy() - 0.5) + g.standard_normal(len(_base))
            if arm in phase_slope:  # the arm knows the phase baseline
                score = score + phase_slope[arm] * (_base["phase"].to_numpy() - 3.5)
            return score

        def rank_scores(arm: str, run_seed: int, _scale=scale, _base=base, _ci=ci):
            g = np.random.default_rng([seed, _ci, run_seed, 31, sum(map(ord, arm))])
            q = base_quality + _scale * (rank_quality[arm] - base_quality)
            return q * (0.5 - _base["final_rank_norm"].to_numpy()) + g.standard_normal(len(_base))

        for arm in quality:
            if (cohort, arm) in skip:
                continue
            run_seeds = [42] + (list(seeds.get(cohort, ())) if arm in seed_arms else [])
            for run_seed in run_seeds:
                df = base.copy()
                df["p_hazard"] = 1.0 / (1.0 + np.exp(-hazard_scores(arm, run_seed)))
                if arm.startswith(("HGB", "SAGONG")):  # a tree model has no rank head: -p_hazard is the survival-like score
                    df["rank_score"] = -df["p_hazard"]
                else:
                    df["rank_score"] = rank_scores(arm, run_seed)
                out = root / cohort / arm / f"seed{run_seed}"
                M.write_predictions(df, out / "pred_test.csv")
                meta = {"labels": labels, "config_hash": f"synthetic-{cohort}-{arm}-{run_seed}",
                        "arm": arm, "cohort": cohort, "seed": run_seed}  # fmt: skip
                if n_train and cohort in n_train and not arm.startswith(("HGB", "SAGONG")):
                    meta["n_matches"] = {"train": int(n_train[cohort]), "val": 10, "test": 10}
                (out / "metrics.json").write_text(json.dumps(meta))
    return root
