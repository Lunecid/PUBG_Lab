"""G1 causal input boundary for the frozen V3 loader.

Design: docs/experiment_design/WORLD_MODEL_G1_IMPLEMENTATION_2026-10-20.md.
No rank, human/bot label, future mask, bank episode or terminal time is accepted
by the encoder. IDs are alignment metadata, never learned identity embeddings.
"""
from dataclasses import dataclass
from collections import Counter
import sqlite3

import numpy as np
import torch

CONDITIONS = ("map", "perspective", "nominal_team_size", "game_mode", "patch")


@dataclass(frozen=True)
class ConditionVocabulary:
    tokens: dict

    @classmethod
    def from_dataset(cls, dataset):
        return cls({key: {value: j+2 for j, value in enumerate(sorted({
            str(dataset.metadata[mid][key]) for mid in dataset.fit_ids}))}
                    for key in CONDITIONS})

    @property
    def sizes(self):
        return [len(self.tokens[k])+2 for k in CONDITIONS]

    def encode(self, conditions):
        return np.asarray([0 if k not in conditions else
                           self.tokens[k].get(str(conditions[k]), 1)
                           for k in CONDITIONS], np.int64)


@dataclass(frozen=True)
class EncoderInput:
    x: torch.Tensor                 # L,P,17; SI-derived fixed scales
    mask: torch.Tensor              # L,P,17
    absolute_xy: torch.Tensor       # L,P,2; metres / 10000
    history_valid: torch.Tensor     # L,P; latest observable segment only
    mobility: torch.Tensor          # L,P; 0 unknown
    mobility_mask: torch.Tensor
    confidence: torch.Tensor        # L,P,2: value, known
    items: torch.Tensor             # N,3: local frame, local player, fit token
    game: torch.Tensor              # L,9
    game_mask: torch.Tensor
    zone: torch.Tensor              # 2,3: dx,dy,radius in 100m
    zone_mask: torch.Tensor         # 2,3
    conditions: torch.Tensor        # 5; 0 absent, 1 UNK
    cells: torch.Tensor             # C,8; relative xy, support, absolute xy
    observed: torch.Tensor          # E,5: src,dst,mode,count,match count
    inferred: torch.Tensor          # E,2; hypotheses, not passability labels
    located: torch.Tensor           # E,2: local player, cell
    members: torch.Tensor           # E,2: local player, team
    team_count: int
    ego_local_index: int
    target_player_indices: torch.Tensor  # current-known/current-valid only
    actor_ids: torch.Tensor         # alignment only
    origin_m: torch.Tensor          # inverse target transform only


@dataclass
class ModelSample:
    inputs: EncoderInput
    targets: dict
    provenance: dict


@dataclass
class ModelBatch:
    inputs: list
    targets: list
    provenance: list


def collate_samples(samples):
    """Ragged list batching; retains every player/team, with no four-player cap."""
    return ModelBatch([s.inputs for s in samples], [s.targets for s in samples],
                      [s.provenance for s in samples])


def _clean(values, mask):
    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if not np.isfinite(values[mask]).all():
        raise ValueError("Non-finite observed value")
    return np.where(mask, values, 0).astype(np.float32)


def _tensor(value, integer=False):
    return torch.from_numpy(np.array(value, copy=True)).to(
        torch.long if integer else torch.float32)


def adapt_graph(graph, *, ego_id, frame_end, t0, view, vocabulary, focal_ids=None):
    """Only allowlisted view fields enter EncoderInput; targets are not accepted.

    Visibility is enforced by WorldDataset.view_at, then reapplied before all
    transforms. This is a HUD/telemetry proxy, not reconstruction of visual FOV.
    """
    if view not in ("observer", "server"):
        raise ValueError("Unknown information view")
    ids = np.asarray(graph["player_ids"])
    matches = np.flatnonzero(ids == ego_id)
    if len(matches) != 1:
        raise ValueError("Exactly one ego is required")
    ego = int(matches[0])
    mask = np.array(graph["mask"], bool, copy=True)
    if view == "observer":
        mask[:, ids != ego_id, 8:] = False
    raw = _clean(graph["x"], mask)
    L, P, F = raw.shape
    if F != 17 or L < 1:
        raise ValueError("Expected L,P,17 input")
    times = t0 - np.arange(L-1, -1, -1)*10
    pos_valid = mask[:, :, :2].all(-1)
    source = np.asarray(graph["source_time"])
    if np.any((source > times[:, None]+1e-4) & pos_valid):
        raise ValueError("future position source")
    if not pos_valid[-1, ego]:
        raise ValueError("Ego requires current position")
    origin = raw[-1, ego, :2].copy()

    # Derive resets only from permitted evidence. Never read hidden bank episodes.
    starts = np.zeros(P, np.int64)
    for p in range(P):
        for j in range(1, L):
            if not pos_valid[j, p]:
                continue
            gap = (not pos_valid[j-1, p] or
                   source[j, p]-source[j-1, p] > 15)
            dead_to_alive = (mask[j-1, p, 5] and mask[j, p, 5] and
                             raw[j-1, p, 5] <= 0 < raw[j, p, 5])
            episode_reset = (mask[j-1, p, 16] and mask[j, p, 16] and
                             raw[j, p, 16] < raw[j-1, p, 16])
            if gap or dead_to_alive or episode_reset:
                starts[p] = j
    keep = np.arange(L)[:, None] >= starts[None, :]
    mask &= keep[:, :, None]
    for p, start in enumerate(starts):
        if start:
            mask[start, p, 3:5] = False  # velocity cannot bridge a reset
            elapsed = (np.arange(L)-start)*10
            for feature, window in ((9, 10), (10, 30), (11, 30), (12, 30), (13, 30)):
                mask[elapsed < window, p, feature] = False
            stale = raw[:, p, 14] > elapsed
            mask[stale, p, 14:16] = False
    raw = _clean(raw, mask)
    valid = mask[:, :, :2].all(-1)
    absolute = np.where(mask[:, :, :2], raw[:, :, :2]/10000., 0)
    x = raw.copy()
    x[:, :, :2] = (x[:, :, :2]-origin)/100.
    x[:, :, 2] /= 100.
    x[:, :, 3:5] /= 10.
    x[:, :, 5] /= 100.
    x[:, :, [6, 14, 16]] /= 120.
    x[:, :, 9:14] = np.log1p(np.maximum(x[:, :, 9:14], 0))
    x = _clean(x, mask)
    mob_mask = np.asarray(graph["mobility_mask"], bool).copy() & keep & valid
    conf_mask = np.asarray(graph["reentry_confidence_mask"], bool).copy() & keep
    if view == "observer":
        mob_mask[:, ids != ego_id] = False
        conf_mask[:, ids != ego_id] = False
    mob = np.where(mob_mask, graph["mobility"], 0)
    conf = _clean(graph["reentry_confidence"], conf_mask)
    items = []
    lookup = {int(v): j for j, v in enumerate(ids)}
    for frame, actor, token in graph["items_sparse"]:
        j = int(frame)-(frame_end-L+1)
        if j >= L:
            raise ValueError("future equipment frame")
        p = lookup.get(int(actor))
        if p is None or j < 0 or (view == "observer" and p != ego):
            continue
        if keep[j, p] and valid[j, p] and mask[j, p, 14]:
            items.append((j, p, int(token)))

    game_mask = np.array(graph["game_mask"], bool, copy=True)
    if view == "observer":
        game_mask[:, 2] = False  # exact surviving teams is not verified public
    if np.any((graph["game_source_time"] > times+1e-4) & game_mask.any(-1)):
        raise ValueError("future game source")
    game = _clean(graph["game"], game_mask)
    zone = game[-1, 3:9].reshape(2, 3).copy()
    zone_mask = game_mask[-1, 3:9].reshape(2, 3)
    zone[:, :2] -= origin
    zone = _clean(zone/100., zone_mask)
    game[:, 0] /= 10.
    game[:, 1:3] /= 100.
    for a, b in ((3, 4), (6, 7)):
        game[:, a:b+1] = (game[:, a:b+1]-origin)/100.
    game[:, [5, 8]] /= 100.
    game = _clean(game, game_mask)

    cells = np.array(graph["cell_features"], np.float32, copy=True)
    if not np.isfinite(cells).all():
        raise ValueError("Non-finite spatial feature")
    cells_abs = cells[:, :2]/10000.
    cells[:, :2] = (cells[:, :2]-origin)/100.
    cells[:, 2:5] = np.log1p(np.maximum(cells[:, 2:5], 0))
    cells = np.concatenate([cells, cells_abs], axis=-1)
    members = np.asarray(graph["member_edges"], np.int64).reshape(-1, 2)
    if focal_ids is None:
        own_team = members[members[:, 0] == ego, 1]
        focal_ids = (ids[members[members[:, 1] == own_team[0], 0]]
                     if len(own_team) else np.array([ego_id]))
    target_players = np.flatnonzero(np.isin(ids, focal_ids) & valid[-1])
    return EncoderInput(
        _tensor(x), torch.from_numpy(mask), _tensor(absolute), torch.from_numpy(valid),
        _tensor(mob, True), torch.from_numpy(mob_mask),
        _tensor(np.stack([conf, conf_mask], -1)),
        _tensor(np.asarray(items).reshape(-1, 3), True),
        _tensor(game), torch.from_numpy(game_mask), _tensor(zone),
        torch.from_numpy(zone_mask), _tensor(vocabulary.encode(graph["conditions"]), True),
        _tensor(cells), _tensor(graph["observed_connections"], True),
        _tensor(graph["inferred_adjacency"], True), _tensor(graph["located_in"], True),
        _tensor(members, True), len(graph["team_ids"]), ego,
        _tensor(target_players, True), _tensor(ids, True), _tensor(origin))


def exclude_current_match_spatial(dataset, match_id, graph):
    """Remove a training match's entire contribution, including its future, from prior.

    Frozen banks/catalog and WorldDataset are unchanged. Other fit matches remain
    a population prior, not an online map acquired before the absolute wall-clock t0.
    """
    if match_id not in dataset.fit_ids or not len(graph["cell_coordinates"]):
        return graph
    geometry = dataset.metadata[match_id]["geometry_key"]
    cache_key = (match_id, geometry)
    cache = getattr(dataset, "_g1_exclusion_cache", {})
    if cache_key not in cache:
        counts, rows_per_key = Counter(), Counter()
        with sqlite3.connect(f"file:{dataset.root/'catalog.sqlite'}?mode=ro", uri=True) as db:
            rows = db.execute(
                "SELECT sx,sy,dx,dy,mode,n FROM spatial_support WHERE match_id=? AND geometry_key=?",
                (match_id, geometry))
            for sx, sy, dx, dy, mode, n in rows:
                key = (sx, sy, dx, dy, mode)
                counts[key] += n
                rows_per_key[key] += 1
        cache = {cache_key: (counts, rows_per_key)}  # one match, bounded
        dataset._g1_exclusion_cache = cache
    counts, rows_per_key = cache[cache_key]
    cells = graph["cell_coordinates"]
    retained = []
    for src, dst, mode, n, nmatches in graph["observed_connections"]:
        key = (*map(int, cells[src]), *map(int, cells[dst]), int(mode))
        n_other, m_other = int(n)-counts[key], int(nmatches)-rows_per_key[key]
        if n_other < 0 or m_other < 0:
            raise ValueError("Inconsistent spatial support subtraction")
        if n_other > 0 and m_other > 0:
            retained.append((src, dst, mode, n_other, m_other))
    graph = dict(graph)
    graph["observed_connections"] = np.array(retained, np.int64).reshape(-1, 5)
    features = graph["cell_features"].copy()
    features[:, 2:] = 0
    for src, dst, mode, n, _ in retained:
        features[src, 2] += n
        features[dst, 3] += n
        features[src, 4] += 1
        features[dst, 4] += 1
        features[src, 5] = features[dst, 5] = 1
    graph["cell_features"] = features
    lookup = {tuple(cell): j for j, cell in enumerate(cells)}
    connected = {(row[0], row[1]) for row in retained}
    inferred = []
    if dataset.ablation != "observed_edges_only":
        for (x, y), src in lookup.items():
            for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                dst = lookup.get((x+dx, y+dy))
                if dst is not None and (src, dst) not in connected:
                    inferred.append((src, dst))
    graph["inferred_adjacency"] = np.array(inferred, np.int64).reshape(-1, 2)
    return graph


def load_sample(dataset, match_id, anchor_index, vocabulary):
    graph, targets = dataset.sample(match_id, anchor_index)
    graph = exclude_current_match_spatial(dataset, match_id, graph)
    bank = dataset.bank(match_id)
    k, ego = map(int, bank["anchors"][anchor_index])
    focal_ids = np.flatnonzero((bank["team"] == bank["team"][ego]) & bank["known"][k])
    inputs = adapt_graph(graph, ego_id=ego, frame_end=k, t0=float(bank["times"][k]),
                         view=dataset.view, vocabulary=vocabulary, focal_ids=focal_ids)
    # Alignment is computed after input construction. Future masks never select inputs.
    target_lookup = {int(v): j for j, v in enumerate(targets["members"])}
    rows = np.array([target_lookup[int(inputs.actor_ids[p])]
                     for p in inputs.target_player_indices], np.int64)
    targets = dict(targets)
    targets["selected_rows"] = rows
    targets["joint_complete_60s"] = bool(len(rows) and targets["mask"][rows].all())
    return ModelSample(inputs, targets, {
        "match_id": match_id, "anchor_index": anchor_index, "frame": k,
        "t0": float(bank["times"][k]), "view": dataset.view,
        "spatial_prior_policy": "fit_minus_current_match" if match_id in dataset.fit_ids
                                else "fit_only_exact_patch",
        "patch": graph["conditions"].get("patch"),
        "known_focal_members": len(focal_ids),
        "current_valid_focal_members": len(rows)})
