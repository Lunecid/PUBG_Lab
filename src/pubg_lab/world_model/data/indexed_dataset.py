"""Spatial query acceleration with identical frozen-loader output.

Indexes only the fit support already authorized by WorldDataset, never test.
Insertion-order ranks preserve floating-point aggregation order of G1.
"""

from collections import defaultdict, Counter
import numpy as np

from pubg_lab.world_model.data.dataset import WorldDataset


class IndexedWorldDataset(WorldDataset):
    def sample(self, mid, anchor_index):
        bank = self.bank(mid)
        k, ego = map(int, bank["anchors"][anchor_index])
        own = (bank["team"] == bank["team"][ego]) & bank["known"][k] & bank["valid"][k]
        self._crop = set()
        for ix, iy in np.floor(bank["x"][k, own, :2] / 100).astype(int):
            self._crop.update((ix + dx, iy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        # Bank spatial_edges is the exact per-match catalog source. Used only
        # for current-match subtraction, never as an input prior itself.
        geometry = self.metadata[mid]["geometry_key"]
        if mid in self.fit_ids:
            counts, nr = Counter(), Counter()
            for sx, sy, dx, dy, mode, n in bank["spatial_edges"]:
                key = tuple(map(int, (sx, sy, dx, dy, mode)))
                counts[key] += int(n)
                nr[key] += 1
            self._g1_exclusion_cache = {(mid, geometry): (counts, nr)}
        return super().sample(mid, anchor_index)

    def support(self, geometry):
        if getattr(self, "_indexed_geometry", None) != geometry:
            counts, matches = super().support(geometry)
            by_source = defaultdict(list)
            for order, (key, count) in enumerate(counts.items()):
                by_source[key[:2]].append((order, key, count, matches[key]))
            self._indexed_geometry = geometry
            self._by_source = by_source
        selected = sorted(
            (
                row
                for cell in self._crop
                for row in self._by_source.get(cell, [])
                if row[1][2:4] in self._crop
            ),
            key=lambda row: row[0],
        )
        return ({k: n for _, k, n, m in selected}, {k: m for _, k, n, m in selected})
