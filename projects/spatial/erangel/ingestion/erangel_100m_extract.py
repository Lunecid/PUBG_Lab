"""Read-only, resumable full 40.2 inferred Erangel extraction.

All selected matches are processed. Export only match-local spatial aggregates,
never account IDs. The graph builder consumes training aggregates exclusively.
"""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT
import csv
import getpass
import hashlib
import json
import time
from collections import Counter, defaultdict
from datetime import timezone
from pathlib import Path

import numpy as np
import psycopg2
from PIL import Image
from scipy.ndimage import map_coordinates

ROOT = REPO_ROOT
OUT = ROOT / "reports/erangel_100m_40_2_2026-10-15"
EXTENT, CELL, WIDTH = 8160.0, 100.0, 82
PATCH_START = "2026-03-11T08:30:00+00:00"
PATCH_END = "2026-04-08T00:00:00+00:00"
CONTRACT = dict(
    version=1,
    patch="40.2 inferred; exact build unknown",
    start_utc=PATCH_START,
    end_utc=PATCH_END,
    cell_m=CELL,
    grid_width=WIDTH,
    split="80/20 within match_type and game_mode; SHA256 erangel100-v1 + match ID",
    point_filter="participant membership; unique account/timestamp; landed; before first recorded death; health>0; finite XYZ; XY inside 8160m",
    pair_filter="consecutive retained raw rows within same account; 5<=dt<=15 seconds; planar speed <=12m/s nonvehicle or <=90m/s vehicle; same transport at endpoints",
    player_kind={"0": "unknown", "1": "human account. prefix", "2": "bot ai. prefix"},
    motion={"0": "no vehicle recorded, not certified on foot", "1": "vehicle recorded"},
    height_prior="2019 raw height, nominal XY, fixed proxy scale 1/128 m per raw; not certified ground height",
    node_columns=[
        "cell",
        "kind",
        "motion",
        "count",
        "z_median",
        "z_q10",
        "z_q90",
        "height_residual_median",
        "height_residual_q10",
        "height_residual_q90",
    ],
    edge_columns=["source", "target", "kind", "motion", "count", "dt_sum_s", "distance_sum_m"],
)


def export_match(cur, row, height):
    mid = row["match_id"]
    counts = Counter()
    cur.execute("SELECT player_id FROM pubg.participants WHERE match_id=%s", (mid,))
    members = {r[0] for r in cur.fetchall() if r[0]}
    cur.execute(
        "SELECT account_id,min(event_time) FROM pubg.telem_parachute_landing WHERE match_id=%s GROUP BY 1",
        (mid,),
    )
    landing = dict(cur.fetchall())
    cur.execute(
        "SELECT victim_id,min(event_time) FROM pubg.telem_kills WHERE match_id=%s GROUP BY 1",
        (mid,),
    )
    deaths = dict(cur.fetchall())
    cur.execute(
        """SELECT account_id,event_time,pos_x::float8,pos_y::float8,pos_z::float8,
        health::float8,vehicle_type,vehicle_id FROM pubg.telem_positions
        WHERE match_id=%s ORDER BY account_id,event_time,id""",
        (mid,),
    )
    raw = cur.fetchall()
    counts["raw_positions"] = len(raw)
    duplicates = Counter((r[0], r[1]) for r in raw)
    kept = []
    edges = defaultdict(lambda: np.zeros(3))
    prev = None
    for account, t, x, y, z, hp, vt, vi in raw:
        reason = None
        if account not in members:
            reason = "not_participant"
        elif t is None or any(v is None or not np.isfinite(v) for v in (x, y, z)):
            reason = "invalid_xyz_time"
        elif duplicates[account, t] != 1:
            reason = "duplicate_timestamp"
        elif account not in landing or landing[account] is None:
            reason = "no_landing"
        elif t < landing[account]:
            reason = "before_landing"
        elif account in deaths and deaths[account] is not None and t >= deaths[account]:
            reason = "after_first_death"
        elif hp is None or not np.isfinite(hp) or hp <= 0:
            reason = "not_alive"
        elif not (0 <= x < 816000 and 0 <= y < 816000):
            reason = "out_of_bounds"
        if reason:
            counts["excluded_" + reason] += 1
            prev = None
            continue
        kind = 1 if account.startswith("account.") else 2 if account.startswith("ai.") else 0
        motion = int(bool(vt or vi))
        x, y, z = x / 100, y / 100, z / 100
        cell = int(y // CELL) * WIDTH + int(x // CELL)
        kept.append((cell, kind, motion, x, y, z))
        counts["retained_positions"] += 1
        counts[f"kind_{kind}_motion_{motion}"] += 1
        if prev is not None and prev[0] == account and prev[5] == motion:
            dt = (t - prev[1]).total_seconds()
            dist = float(np.hypot(x - prev[2], y - prev[3]))
            if 5 <= dt <= 15 and dist / dt <= (90 if motion else 12):
                counts["accepted_pairs"] += 1
                if prev[4] != cell:
                    edges[prev[4], cell, kind, motion] += np.array([1, dt, dist])
                    counts["nonself_pairs"] += 1
            else:
                counts["rejected_time_or_speed_pairs"] += 1
        prev = (account, t, x, y, cell, motion)
    nodes = []
    if kept:
        p = np.asarray(kept)
        raw_h = map_coordinates(
            height,
            [p[:, 4] * 8192 / EXTENT, p[:, 3] * 8192 / EXTENT],
            order=1,
            mode="nearest",
            output=np.float64,
        )
        residual = p[:, 5] - (raw_h - 32768) / 128
        key = p[:, 0].astype(int) * 6 + p[:, 1].astype(int) * 2 + p[:, 2].astype(int)
        order = np.argsort(key, kind="stable")
        bounds = np.r_[0, np.where(np.diff(key[order]))[0] + 1, len(order)]
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            ids = order[lo:hi]
            first = p[ids[0]]
            zz = np.quantile(p[ids, 5], [0.5, 0.1, 0.9])
            rr = np.quantile(residual[ids], [0.5, 0.1, 0.9])
            nodes.append([*first[:3], len(ids), *zz, *rr])
    edge_rows = [[*key, *value] for key, value in sorted(edges.items())]
    return (
        np.asarray(nodes, dtype=float).reshape(-1, 10),
        np.asarray(edge_rows, dtype=float).reshape(-1, 7),
        dict(counts),
    )


def main():
    OUT.mkdir(exist_ok=True)
    cache = OUT / "match_cache"
    cache.mkdir(exist_ok=True)
    contract_hash = hashlib.sha256(json.dumps(CONTRACT, sort_keys=True).encode()).hexdigest()
    contract_file = OUT / "contract.json"
    if contract_file.exists():
        assert json.loads(contract_file.read_text()) == CONTRACT, (
            "Contract changed; use a new output directory"
        )
    contract_file.write_text(json.dumps(CONTRACT, indent=2) + "\n")
    password = getpass.getpass("DB password: ")
    conn = psycopg2.connect(
        host="localhost",
        dbname="pubg_survival",
        user="postgres",
        password=password,
        connect_timeout=5,
        options="-c default_transaction_read_only=on -c statement_timeout=60000",
    )
    del password
    height = np.asarray(Image.open(ROOT / "data/spatial/erangel/source/height_2019.png")).astype(
        np.float32
    )
    with conn.cursor() as cur:
        cur.execute("SHOW transaction_read_only")
        assert cur.fetchone()[0] == "on"
        cur.execute(
            """SELECT m.match_id,m.map_name,m.game_mode,m.match_type,m.shard_id,
            s.event_time,s.weather_id,s.is_event_mode FROM pubg.matches m
            JOIN pubg.telem_match_start s USING(match_id)
            WHERE m.map_name IN ('Baltic_Main','Erangel_Main') AND m.shard_id='steam'
            AND s.event_time >= %s AND s.event_time < %s ORDER BY m.match_id""",
            (PATCH_START, PATCH_END),
        )
        rows = [
            dict(
                zip(
                    [
                        "match_id",
                        "map",
                        "mode",
                        "match_type",
                        "shard",
                        "start_at",
                        "weather",
                        "event_mode",
                    ],
                    r,
                )
            )
            for r in cur.fetchall()
        ]
        assert len({r["match_id"] for r in rows}) == len(rows)
        groups = defaultdict(list)
        for r in rows:
            r["start_at"] = r["start_at"].astimezone(timezone.utc).isoformat()
            groups[r["match_type"], r["mode"]].append(r)
        for group in groups.values():
            group.sort(
                key=lambda r: hashlib.sha256(
                    ("erangel100-v1:" + r["match_id"]).encode()
                ).hexdigest()
            )
            nval = max(1, round(len(group) * 0.2))
            for i, r in enumerate(group):
                r["split"] = "validation" if i < nval else "train"
        rows.sort(key=lambda r: r["match_id"])
        for i, r in enumerate(rows):
            r["match_index"] = i
        inventory = OUT / "match_split.csv"
        if inventory.exists():
            old = list(csv.DictReader(inventory.open()))
            assert [(r["match_id"], r["split"]) for r in old] == [
                (r["match_id"], r["split"]) for r in rows
            ], "Inventory drift; start a new run"
        with inventory.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0])
            w.writeheader()
            w.writerows(rows)
        conn.commit()
        start = time.monotonic()
        summary = []
        for i, r in enumerate(rows):
            dest = cache / f"{r['match_index']:04d}.npz"
            if dest.exists():
                with np.load(dest) as saved:
                    assert (
                        str(saved["contract_hash"]) == contract_hash
                        and str(saved["match_id"]) == r["match_id"]
                    )
                    counts = json.loads(str(saved["counts"]))
            else:
                nodes, edges, counts = export_match(cur, r, height)
                temp = dest.with_suffix(".tmp.npz")
                np.savez_compressed(
                    temp,
                    nodes=nodes,
                    edges=edges,
                    counts=json.dumps(counts),
                    match_id=r["match_id"],
                    contract_hash=contract_hash,
                )
                temp.replace(dest)
                conn.commit()
            summary.append({**r, **counts})
            if (i + 1) % 50 == 0 or i + 1 == len(rows):
                progress = dict(
                    completed=i + 1,
                    total=len(rows),
                    elapsed_seconds=round(time.monotonic() - start, 1),
                    last_match_index=r["match_index"],
                )
                (OUT / "progress.json").write_text(json.dumps(progress))
                print(json.dumps(progress), flush=True)
        (OUT / "extraction.json").write_text(
            json.dumps(
                dict(
                    status="complete",
                    matches=summary,
                    totals=dict(
                        sum(
                            (
                                Counter(
                                    {
                                        k: v
                                        for k, v in r.items()
                                        if isinstance(v, int)
                                        and not isinstance(v, bool)
                                        and k != "match_index"
                                    }
                                )
                                for r in summary
                            ),
                            Counter(),
                        )
                    ),
                ),
                indent=2,
            )
        )
    conn.close()
    print("Complete: all selected matches processed", flush=True)


if __name__ == "__main__":
    main()
