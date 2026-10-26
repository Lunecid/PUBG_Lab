"""Export complete official TPP squad cohort, readonly and resumable."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import argparse
import csv
import gzip
import getpass
import hashlib
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import psycopg2

ROOT = REPO_ROOT
OUT = ROOT / "data/processed/heterograph_diffusion_v1"
TABLES = {
    "positions": (
        "account_id,event_time,pos_x::float8,pos_y::float8,pos_z::float8,health::float8,vehicle_type,vehicle_id",
        ["actor", "t", "x", "y", "z", "hp", "vehicle_type", "vehicle_class"],
    ),
    "item_equip": (
        "account_id,event_time,event_type,item_id,category,sub_category",
        ["actor", "t", "action", "item", "category", "subcategory"],
    ),
    "item_pickup": (
        "account_id,event_time,event_type,item_id,category,stack_count",
        ["actor", "t", "action", "item", "category", "count"],
    ),
    "item_use": (
        "account_id,event_time,item_id,category,stack_count",
        ["actor", "t", "item", "category", "count"],
    ),
    "damage": (
        "victim_id,event_time,attacker_id,damage::float8,damage_type,attacker_x::float8,attacker_y::float8",
        ["actor", "t", "attacker", "damage", "type", "attacker_x", "attacker_y"],
    ),
    "groggy": ("victim_id,event_time", ["actor", "t"]),
    "kills": ("victim_id,event_time", ["actor", "t"]),
    "parachute_landing": ("account_id,event_time", ["actor", "t"]),
    "game_states": (
        "event_time,is_game::float8,num_alive_players,num_alive_teams,safe_zone_x::float8,safe_zone_y::float8,safe_zone_radius::float8,poison_zone_x::float8,poison_zone_y::float8,poison_zone_radius::float8",
        [
            "t",
            "phase",
            "alive_players",
            "alive_teams",
            "safe_x",
            "safe_y",
            "safe_r",
            "poison_x",
            "poison_y",
            "poison_r",
        ],
    ),
}


def export(row, pw):
    path = OUT / "raw" / f"{row['match_id']}.json.gz"
    if path.exists():
        with gzip.open(path, "rt") as f:
            x = json.load(f)
        assert x["meta"]["match_id"] == row["match_id"]
        return {"match_id": row["match_id"], "cached": True, "counts": x["counts"]}
    conn = psycopg2.connect(
        host="localhost",
        dbname="pubg_survival",
        user="postgres",
        password=pw,
        connect_timeout=10,
        options="-c default_transaction_read_only=on -c statement_timeout=120000",
    )
    start = datetime.fromisoformat(row["start_at"])
    tables = {}
    counts = {}
    dups = {}
    unknown = {}
    with conn, conn.cursor() as cur:
        cur.execute("SHOW transaction_read_only")
        assert cur.fetchone()[0] == "on"
        cur.execute(
            "SELECT p.player_id,r.team_id FROM pubg.participants p JOIN pubg.rosters r ON r.roster_id=p.roster_id WHERE p.match_id=%s",
            (row["match_id"],),
        )
        roster = sorted(set(cur.fetchall()))
        assert len({p for p, t in roster}) == len(roster)
        roster = [(p, t) for p, t in roster if p and t is not None]
        ids = {p: i for i, (p, t) in enumerate(roster)}
        teamids = {t: i for i, t in enumerate(sorted({t for p, t in roster}))}
        actors = [
            {
                "actor": ids[p],
                "team": teamids[t],
                "kind": 1 if p.startswith("account.") else 2 if p.startswith("ai.") else 0,
            }
            for p, t in roster
        ]
        for table, (cols, keys) in TABLES.items():
            cur.execute(
                f"SELECT {cols} FROM pubg.telem_{table} WHERE match_id=%s ORDER BY event_time,id",
                (row["match_id"],),
            )
            raw = cur.fetchall()
            kept = []
            seen = set()
            dropped = 0
            for values in raw:
                r = dict(zip(keys, values))
                r["t"] = round((r["t"] - start).total_seconds(), 3)
                if "actor" in r:
                    if r["actor"] not in ids:
                        dropped += 1
                        continue
                    r["actor"] = ids[r["actor"]]
                if "attacker" in r:
                    r["attacker"] = ids.get(r["attacker"], -1)
                for k in (
                    "x",
                    "y",
                    "z",
                    "attacker_x",
                    "attacker_y",
                    "safe_x",
                    "safe_y",
                    "safe_r",
                    "poison_x",
                    "poison_y",
                    "poison_r",
                ):
                    if k in r and r[k] is not None:
                        r[k] = round(r[k] / 100.0, 4)
                key = tuple(r.get(k) for k in keys)
                if key in seen:
                    continue
                seen.add(key)
                kept.append(r)
            tables[table] = kept
            counts[table] = len(raw)
            dups[table] = len(raw) - dropped - len(kept)
            unknown[table] = dropped
    conn.close()
    result = {
        "format": 1,
        "meta": row,
        "actors": actors,
        "tables": tables,
        "counts": counts,
        "duplicates_removed": dups,
        "unknown_actor_rows": unknown,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "scope": "server recorded evidence, no direct account IDs; not observer inputs",
    }
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", compresslevel=5) as f:
        json.dump(result, f, separators=(",", ":"))
    tmp.replace(path)
    return {"match_id": row["match_id"], "cached": False, "counts": counts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    rows = [
        r
        for r in csv.DictReader(
            (ROOT / "reports/erangel_100m_40_2_2026-10-15/match_split.csv").open()
        )
        if r["match_type"] == "official" and r["mode"] == "squad"
    ]
    if args.limit:
        rows = rows[: args.limit]
    OUT.mkdir(exist_ok=True)
    (OUT / "raw").mkdir(exist_ok=True)
    with (OUT / "matches.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "source_queries.json").write_text(
        json.dumps(
            {
                k: {
                    "sql": f"SELECT {v[0]} FROM pubg.telem_{k} WHERE match_id=%s ORDER BY event_time,id",
                    "columns": v[1],
                }
                for k, v in TABLES.items()
            },
            indent=2,
        )
    )
    pw = getpass.getpass("DB password: ")
    done = []
    errors = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(export, r, pw): r for r in rows}
        for f in as_completed(futures):
            try:
                done.append(f.result())
            except Exception as e:
                errors.append({"match_id": futures[f]["match_id"], "error": str(e)})
                print("ERROR", errors[-1], flush=True)
            if (len(done) + len(errors)) % 25 == 0:
                print(
                    json.dumps(
                        {
                            "completed": len(done),
                            "failed": len(errors),
                            "total": len(rows),
                            "seconds": round(time.time() - started),
                        }
                    ),
                    flush=True,
                )
    del pw
    (OUT / "extraction_report.json").write_text(
        json.dumps(
            {
                "completed": len(done),
                "expected": len(rows),
                "errors": errors,
                "total_raw_rows": dict(sum((Counter(d["counts"]) for d in done), Counter())),
                "elapsed_s": time.time() - started,
            },
            indent=2,
        )
    )
    print("FINISHED", len(done), len(errors), flush=True)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
