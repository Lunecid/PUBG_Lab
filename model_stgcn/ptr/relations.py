"""Relation matrices for the team-axis attention bias, and the within-match edge shuffle.

Edge storage (dump schema): ``{"prox"|"combat"|"copress": (idx int64 [E, 3] = (p, i, j), w f32 [E])}``.
``combat`` is directed attacker -> victim (``i`` hit ``j``); its weight is the combat intensity
``damage + 100 * kills + 50 * groggy`` either raw or already log-compressed, see
:data:`COMBAT_WEIGHT_MODES` (the real dump copies ``ln(1 + intensity)`` from the edge cache).  ``prox`` /
``copress`` are symmetric; they may be stored once or in both directions (both are accepted: the dense
matrix is symmetrised with ``max``).

Relation channels, always indexed ``[query i, key j]``; weights are used as dumped (no row normalisation),
the diagonal is 0 (self loops are dropped; self-attention itself is allowed):

``prox``        w                                   symmetric
``combat_out``  log1p(intensity[i -> j])            i attacked j
``combat_in``   log1p(intensity[j -> i])            j attacked i  (transpose of ``combat_out``)
``copress``     w                                   symmetric

Escalation (design doc section 3): when a combat edge exists in *either* direction for the pair
``(p, i, j)`` the ``prox`` weight of that pair is zeroed, so the pair is carried by the combat
channels only.  Existence means "listed in the edge set", independent of the weight value.
Escalation is only applied when at least one combat channel is requested, otherwise the missing
``prox`` weights would leak the combat edge set into a model that was meant not to see it.
"""

from __future__ import annotations

import zlib
from typing import Mapping, Sequence

import numpy as np
import torch

RELATION_TYPES: tuple[str, ...] = ("prox", "combat_out", "combat_in", "copress")
# How the dump's combat weights relate to the model channel ``log1p(intensity)``:
#   "raw"  the dump holds the raw intensity damage + 100*kills + 50*groggy  -> channel = log1p(w)
#   "log"  the dump already holds ln(1 + intensity) (what stdb/phase_edge_etl.py:166 writes into
#          team_phase_edge_cache, hence what dump_phase_tier.py copies)       -> channel = w
COMBAT_WEIGHT_MODES: tuple[str, ...] = ("raw", "log")
EDGE_TYPES: tuple[str, ...] = ("prox", "combat", "copress")
SYMMETRIC_EDGE_TYPES: tuple[str, ...] = ("prox", "copress")


def check_relation_types(types: Sequence[str]) -> tuple[str, ...]:
    types = tuple(types)
    unknown = [t for t in types if t not in RELATION_TYPES]
    if unknown or len(set(types)) != len(types) or not types:
        raise ValueError(f"relation_types must be a non-empty subset of {RELATION_TYPES}, got {types}")
    return types


def densify_edges_batch(
    edges_list: Sequence[Mapping | None],
    n_phases: int,
    n_max: int,
    *,
    edge_types: Sequence[str] = EDGE_TYPES,
    symmetric: Sequence[str] = SYMMETRIC_EDGE_TYPES,
    directed: str = "combat",
) -> dict[str, torch.Tensor]:
    """Sparse edge dicts -> dense ``[B, P, N, N]`` tensors (zero padded to ``n_max`` nodes).

    Returns one tensor per ``edge_types`` entry (team layer: ``prox``, ``combat`` = raw directed weights,
    ``copress``; the player layer passes its own types), ``combat_exists`` (bool, directed: which pairs have an edge of
    the ``directed`` type) and symmetrises the ``symmetric`` types.  Duplicated entries collapse with ``max``; self
    loops are dropped; non-finite or negative weights become 0.
    """
    batch = len(edges_list)
    shape = (batch, n_phases, n_max, n_max)
    dense = {name: torch.zeros(shape) for name in edge_types}
    exists = torch.zeros(shape, dtype=torch.bool)
    for b, edges in enumerate(edges_list):
        if not edges:
            continue
        for name in edge_types:
            item = edges.get(name)
            if item is None:
                continue
            idx = torch.as_tensor(item[0], dtype=torch.long).reshape(-1, 3)
            w = torch.as_tensor(item[1], dtype=torch.float32).reshape(-1)
            if idx.shape[0] == 0:
                continue
            keep = idx[:, 1] != idx[:, 2]
            idx, w = idx[keep], w[keep]
            w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
            flat = (idx[:, 0] * n_max + idx[:, 1]) * n_max + idx[:, 2]
            dense[name][b].view(-1).scatter_reduce_(0, flat, w, reduce="amax", include_self=True)
            if name == directed:
                exists[b].view(-1)[flat] = True
    for name in symmetric:
        dense[name] = torch.maximum(dense[name], dense[name].transpose(-1, -2))
    dense["combat_exists"] = exists
    return dense


def build_relation_tensor(
    dense: Mapping[str, torch.Tensor],
    relation_types: Sequence[str] = RELATION_TYPES,
    combat_weights: str = "raw",
) -> torch.Tensor:
    """Dense matrices -> ``[..., R, N, N]`` relation weights (escalation applied, no row normalisation).

    ``combat_weights`` says what the dumped combat weights are (see :data:`COMBAT_WEIGHT_MODES`); the
    combat channels are ``log1p(damage + 100*kills + 50*groggy)`` either way.
    """
    types = check_relation_types(relation_types)
    if combat_weights not in COMBAT_WEIGHT_MODES:
        raise ValueError(f"combat_weights must be one of {COMBAT_WEIGHT_MODES}, got '{combat_weights}'")
    exists = dense["combat_exists"]
    prox = dense["prox"]
    if "combat_out" in types or "combat_in" in types:
        prox = prox.masked_fill(exists | exists.transpose(-1, -2), 0.0)
    combat = dense["combat"].clamp(min=0.0)
    combat_out = torch.log1p(combat) if combat_weights == "raw" else combat
    mats = {
        "prox": prox,
        "combat_out": combat_out,
        "combat_in": combat_out.transpose(-1, -2),
        "copress": dense["copress"],
    }
    return torch.stack([mats[t] for t in types], dim=-3)


def shuffle_permutations(match_id, seed: int, node_mask) -> np.ndarray:
    """Fixed permutation of the alive teams per (match, phase, seed): ``perm[p, i]`` = new label of ``i``.

    Dead teams keep their label.  The generator is keyed by ``(seed, crc32(match_id), phase)`` so it
    does not depend on iteration order, epoch or process (``hash()`` is salted, ``crc32`` is not).
    """
    alive = np.asarray(node_mask, dtype=bool)
    n_phases, n_teams = alive.shape
    perms = np.tile(np.arange(n_teams, dtype=np.int64), (n_phases, 1))
    key = zlib.crc32(str(match_id).encode("utf-8"))
    for p in range(n_phases):
        ids = np.flatnonzero(alive[p])
        if ids.size > 1:
            rng = np.random.default_rng([int(seed), key, p])
            perms[p, ids] = rng.permutation(ids)
    return perms


def shuffle_edges(edges: Mapping, perms: np.ndarray) -> dict:
    """Relabel both endpoints of every edge with the phase's permutation (same one for all types).

    Edges stay in their phase and match; weights and their order are untouched, so the degree
    multiset (per type and jointly across types) is preserved.
    """
    table = torch.as_tensor(perms, dtype=torch.long)
    out = {}
    for name, (idx, w) in edges.items():
        idx = torch.as_tensor(idx, dtype=torch.long).reshape(-1, 3)
        new = idx.clone()
        if idx.shape[0]:
            p = idx[:, 0]
            new[:, 1] = table[p, idx[:, 1]]
            new[:, 2] = table[p, idx[:, 2]]
        out[name] = (new, w)
    return out
