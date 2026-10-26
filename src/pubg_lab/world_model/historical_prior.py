"""Past-only behavioral edge evidence, NEVER a current passability label.

41.1 known geometry changes: https://www.pubg.com/en-asia/news/9926 (World).
G3 contract: same map, allowed actual fit IDs, donor end < target start,
exclude current match, keep historical edges separate from observed geometry.
"""

import sqlite3
from datetime import datetime
from collections import defaultdict
from .cohorts import PRIMARY

CHANGED_41_1 = {"Baltic_Main", "Desert_Main", "Tiger_Main", "Neon_Main"}


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


class HistoricalPrior:
    def __init__(self, path, actual_fit_ids):
        self.db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        donors = self.db.execute("SELECT match_id,role,cohort FROM donors").fetchall()
        if any(role != "train" or cohort not in PRIMARY for _, role, cohort in donors):
            self.db.close()
            raise PermissionError("Nonfit or nonprimary donor table")
        self.fit_ids = set(actual_fit_ids)
        if not self.fit_ids.issubset({m for m, _, _ in donors}):
            self.db.close()
            raise PermissionError("Requested donor IDs outside approved prior")

    def query(self, *, map_name, target_mid, target_start, target_patch, cells, source_patch=None):
        totals = defaultdict(lambda: [0, set(), -float("inf"), set()])
        cells = {tuple(map(int, cell)) for cell in cells}
        for sx, sy in sorted(cells):
            rows = self.db.execute(
                """
                SELECT e.match_id,e.dx,e.dy,e.mode,e.n,d.end_ts,d.patch
                FROM contributions e JOIN donors d USING(match_id)
                WHERE e.map=? AND e.sx=? AND e.sy=? AND d.end_ts<? AND e.match_id!=?
                  AND (? IS NULL OR d.patch=?)
                ORDER BY e.match_id,e.dx,e.dy,e.mode""",
                (
                    map_name,
                    int(sx),
                    int(sy),
                    float(target_start),
                    target_mid,
                    source_patch,
                    source_patch,
                ),
            )
            for mid, dx, dy, mode, n, end, patch in rows:
                if mid not in self.fit_ids or (dx, dy) not in cells:
                    continue
                if mode not in (1, 2, 3):
                    raise ValueError("Air/special mode cannot enter ground historical prior")
                value = totals[(sx, sy, dx, dy, mode)]
                value[0] += n
                value[1].add(mid)
                value[2] = max(value[2], end)
                value[3].add(patch)
        return [
            dict(
                sx=k[0],
                sy=k[1],
                dx=k[2],
                dy=k[3],
                mode=k[4],
                count=v[0],
                donor_matches=len(v[1]),
                newest_donor_end=v[2],
                age_days=(target_start - v[2]) / 86400,
                source_patches=sorted(v[3]),
                known_change=int(target_patch == "41.1" and map_name in CHANGED_41_1),
                geometry_verified=False,
                relation="historical_behavior",
            )
            for k, v in sorted(totals.items())
        ]

    def close(self):
        self.db.close()
