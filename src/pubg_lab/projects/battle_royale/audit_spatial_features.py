"""Audit whether elevation(z)/terrain/high-ground/LOS spatial variables exist
anywhere in the STDB — in the V3 feature view, the underlying tables, and the
raw telemetry. Answers: is position-z captured (node feature) or missing?
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))
from pubg_lab.data.stdb.conn import connect

# terms that would indicate elevation / terrain / high-ground / visibility
ELEV = ["_z", "elev", "alt", "height", "terrain", "ground", "slope", "vis", "los", "cover", "high"]
# general position terms (to see what spatial info IS stored)
POS = [
    "_x",
    "_y",
    "_z",
    "posit",
    "coord",
    "location",
    "loc_",
    "center",
    "dist",
    "radius",
    "spatial",
]


def main():
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema='stdb'
            ORDER BY table_name, ordinal_position
        """)
        rows = cur.fetchall()
        cur.close()

    by_table = {}
    for t, c, d in rows:
        by_table.setdefault(t, []).append((c, d))

    print(f"=== stdb schema: {len(by_table)} tables/views, {len(rows)} columns total ===\n")

    def hits(col, terms):
        cl = col.lower()
        return any(term in cl for term in terms)

    # 1) elevation/terrain columns anywhere
    print("### (1) ELEVATION / TERRAIN / HIGH-GROUND / LOS columns (anywhere in stdb):")
    found = False
    for t in sorted(by_table):
        e = [c for c, d in by_table[t] if hits(c, ELEV)]
        # filter false positives: 'vis' matches 'division'? unlikely here. keep raw.
        if e:
            found = True
            print(f"  {t}: {e}")
    if not found:
        print("  >>> NONE FOUND <<<")

    # 2) the V3 feature view spatial columns
    print("\n### (2) v_team_phase_feature_v2 — all position/spatial-ish columns:")
    v2 = by_table.get("v_team_phase_feature_v2", [])
    print(f"  (view has {len(v2)} columns total)")
    sp = [c for c, d in v2 if hits(c, POS)]
    for c in sp:
        print(f"    {c}")

    # 3) raw position/telemetry tables: do they store z?
    print("\n### (3) raw position/telemetry tables — z/elevation presence:")
    raw_like = [
        t
        for t in by_table
        if any(
            k in t.lower()
            for k in (
                "moving",
                "position",
                "telem",
                "snapshot",
                "proximity",
                "spatial_event",
                "location",
                "object",
            )
        )
    ]
    for t in sorted(raw_like):
        cols = [c for c, d in by_table[t]]
        zc = [c for c in cols if hits(c, ["_z", "elev", "alt", "height"])]
        xyc = [
            c
            for c in cols
            if c.lower()
            in (
                "x",
                "y",
                "z",
                "loc_x",
                "loc_y",
                "loc_z",
                "location_x",
                "location_y",
                "location_z",
                "pos_x",
                "pos_y",
                "pos_z",
            )
        ]
        print(f"  {t}: z/elev cols={zc or '—'} | xyz-like={xyc or '—'}")
        # show all columns for the most relevant raw tables
        if any(k in t.lower() for k in ("moving", "snapshot", "proximity")):
            print(f"      all cols: {cols}")


if __name__ == "__main__":
    main()
