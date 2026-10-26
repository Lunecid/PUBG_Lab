"""Lazy graph/target loader with ablation-specific train-only spatial support.

No training code. Unobserved adjacent cells are hypotheses, never negative
traversability labels. No fitted normalizer is shared across ablation folds.
"""

import csv
import json
from pathlib import Path
import sqlite3
from collections import Counter

import numpy as np
from .preprocess import view_at, targets_at


class WorldDataset:
    def __init__(self, root, ablation="full", role="train", view="observer"):
        self.root = Path(root)
        if role not in ("train", "validation"):
            raise PermissionError("No test or protected-match loader exists.")
        self.plan = json.loads((self.root / "ablations" / f"{ablation}.json").read_text())
        self.ablation = ablation
        self.view = ablation if ablation in ("observer", "server") else view
        self.match_ids = self.plan[role + "_match_ids"]
        self.fit_ids = set(self.plan["train_match_ids"])
        with (self.root / "matches.csv").open() as handle:
            self.metadata = {r["match_id"]: r for r in csv.DictReader(handle)}
        assert not set(self.match_ids) & {
            k for k, v in self.metadata.items() if v["role"] == "protected"
        }
        self.spatial_cache = {}
        self.bank_cache = {}
        tokens = set()
        for mid in sorted(self.fit_ids):
            tokens.update(
                json.loads((self.root / "meta" / f"{mid}.json").read_text())[
                    "item_vocabulary_local"
                ]
            )
        self.item_vocabulary = {v: i + 2 for i, v in enumerate(sorted(tokens))}

    def bank(self, mid):
        if mid not in self.match_ids:
            raise PermissionError("Match is outside the selected role/ablation.")
        if mid not in self.bank_cache:
            self.bank_cache.clear()  # bounded to one match
            with np.load(self.root / "banks" / f"{mid}.npz", allow_pickle=False) as f:
                self.bank_cache[mid] = dict(f)
        return self.bank_cache[mid]

    def support(self, geometry):
        if geometry not in self.spatial_cache:
            db = sqlite3.connect(f"file:{self.root / 'catalog.sqlite'}?mode=ro", uri=True)
            rows = db.execute(
                "SELECT match_id,sx,sy,dx,dy,mode,n FROM spatial_support WHERE geometry_key=?",
                (geometry,),
            )
            counts, matches = Counter(), Counter()
            for mid, sx, sy, dx, dy, mode, n in rows:
                if mid in self.fit_ids:
                    key = (sx, sy, dx, dy, mode)
                    counts[key] += n
                    matches[key] += 1
            db.close()
            self.spatial_cache.clear()  # one map/version at a time
            self.spatial_cache[geometry] = (counts, matches)
        return self.spatial_cache[geometry]

    def sample(self, mid, anchor_index):
        bank = self.bank(mid)
        k, ego = map(int, bank["anchors"][anchor_index])
        graph = view_at(bank, k, ego, self.view)
        targets = targets_at(bank, k, ego)
        meta = self.metadata[mid]
        local_meta = json.loads((self.root / "meta" / f"{mid}.json").read_text())
        item_rows = graph["items_sparse"].copy()
        if len(item_rows):
            local_tokens = local_meta["item_vocabulary_local"]
            item_rows[:, 2] = [
                self.item_vocabulary.get(local_tokens[int(v)], 1) for v in item_rows[:, 2]
            ]
        graph["items_sparse"] = item_rows
        players = graph["player_ids"]
        confidence = bank["confidence"][max(0, k - 12) : k + 1, players].copy()
        confidence_mask = bank["known"][max(0, k - 12) : k + 1, players].copy()
        if self.view == "observer":
            confidence_mask[:, players != ego] = False
        confidence[~confidence_mask] = 0
        graph.update(reentry_confidence=confidence, reentry_confidence_mask=confidence_mask)
        graph["conditions"] = {
            f: meta[f]
            for f in (
                "map",
                "perspective",
                "nominal_team_size",
                "game_mode",
                "patch",
                "geometry_key",
            )
        }
        if self.ablation.startswith("no_") and self.ablation.endswith("_condition"):
            field = self.ablation[3:-10]
            field = {"team_size": "nominal_team_size"}.get(field, field)
            graph["conditions"].pop(field, None)
        cells = set()
        located = []
        # Use only own-team observed positions to define the common paired crop.
        own = (bank["team"] == bank["team"][ego]) & bank["known"][k] & bank["valid"][k]
        for p in np.flatnonzero(own):
            ix, iy = np.floor(bank["x"][k, p, :2] / 100).astype(int)
            cells.update((ix + dx, iy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        keys = sorted(cells)
        lookup = {v: i for i, v in enumerate(keys)}
        for j, p in enumerate(graph["player_ids"]):
            if graph["mask"][-1, j, 0] and graph["mask"][-1, j, 1]:
                cell = tuple(np.floor(graph["x"][-1, j, :2] / 100).astype(int))
                if cell in lookup:
                    located.append((j, lookup[cell]))
        counts, matches = self.support(meta["geometry_key"])
        observed, inferred = [], []
        for (sx, sy, dx, dy, mode), n in counts.items():
            if (sx, sy) in lookup and (dx, dy) in lookup:
                observed.append(
                    [lookup[sx, sy], lookup[dx, dy], mode, n, matches[sx, sy, dx, dy, mode]]
                )
        connected = {(r[0], r[1]) for r in observed}
        for cell, i in lookup.items():
            for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                dst = (cell[0] + dx, cell[1] + dy)
                if dst in lookup and (i, lookup[dst]) not in connected:
                    inferred.append((i, lookup[dst]))
        cell_features = np.zeros((len(keys), 6), np.float32)
        if len(keys):
            cell_features[:, :2] = (np.asarray(keys) * 100 + 50).astype(np.float32)
        for src, dst, mode, n, nmatch in observed:
            cell_features[src, 2] += n
            cell_features[dst, 3] += n
            cell_features[src, 4] += 1
            cell_features[dst, 4] += 1
            cell_features[src, 5] = cell_features[dst, 5] = 1
        graph.update(
            cell_coordinates=np.asarray(keys, dtype=np.int32).reshape(-1, 2),
            cell_features=cell_features,
            located_in=np.asarray(located, dtype=np.int32).reshape(-1, 2),
            observed_connections=np.asarray(observed, dtype=np.int64).reshape(-1, 5),
            inferred_adjacency=np.asarray(inferred, dtype=np.int32).reshape(-1, 2),
            spatial_uncertainty="All geometry unverified; observed edges carry count/match support; adjacency is hypothetical",
        )
        if self.ablation == "no_spatial":
            for key in (
                "cell_coordinates",
                "cell_features",
                "located_in",
                "observed_connections",
                "inferred_adjacency",
            ):
                graph[key] = graph[key][:0]
        if self.ablation == "observed_edges_only":
            graph["inferred_adjacency"] = graph["inferred_adjacency"][:0]
        if self.ablation == "no_team_node":
            for key in ("team_ids", "team_x", "member_edges"):
                graph[key] = graph[key][:0]
        if self.ablation == "short_history_30s":
            for key in (
                "x",
                "mask",
                "game",
                "game_mask",
                "game_source_time",
                "source_time",
                "mobility_ego",
                "mobility",
                "mobility_mask",
                "reentry_confidence",
                "reentry_confidence_mask",
            ):
                graph[key] = graph[key][-4:]
            graph["items_sparse"] = graph["items_sparse"][graph["items_sparse"][:, 0] >= k - 3]
        # Only focal-team final outcome is a supervised target for this sample.
        focal_team = int(bank["team"][ego])
        targets["outcomes"] = [v for v in local_meta["outcomes"] if v["team"] == focal_team]
        if self.ablation == "no_outcome_loss":
            for row in targets["outcomes"]:
                row["rank_mask"] = row["winner_mask"] = False
        return graph, targets
