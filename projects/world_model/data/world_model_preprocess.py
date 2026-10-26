"""Versioned full-corpus preprocessing; no optimizer, training, or test loader.

Run prepare first, then run with a hidden password prompt, then finalize.
Only SELECT/SHOW statements run against PostgreSQL; SQLite is local output.
"""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import getpass
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import sqlite3
import sys
import time

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from pubg_lab.world_model.data.preprocess import (
    build_bank,
    FEATURES,
    GAME_FEATURES,
    MODES,
    view_at,
    targets_at,
)
from pubg_lab.world_model.data.patches import infer_patch, PATCH_WINDOWS

DEFAULT = Path("/Users/lunecid/PUBG_Data/runs/20261020_world_model_patch_preprocess_v3")
QUALITY = ROOT / "reports/corpus_quality_2026-10-20"
TABLES = {
    "positions": (
        "telem_positions",
        "id,account_id,event_time,pos_x::float8,pos_y::float8,pos_z::float8,"
        "health::float8,vehicle_type,vehicle_id,is_game::float8",
        ["row_id", "actor", "t", "x", "y", "z", "hp", "vehicle_type", "vehicle_class", "phase"],
    ),
    "equip": (
        "telem_item_equip",
        "id,account_id,event_time,event_type,item_id,category,sub_category",
        ["row_id", "actor", "t", "action", "item", "category", "subcategory"],
    ),
    "pickup": (
        "telem_item_pickup",
        "id,account_id,event_time,event_type,item_id,category,stack_count",
        ["row_id", "actor", "t", "action", "item", "category", "count"],
    ),
    "use": (
        "telem_item_use",
        "id,account_id,event_time,item_id,category,stack_count",
        ["row_id", "actor", "t", "item", "category", "count"],
    ),
    "damage": (
        "telem_damage",
        "id,victim_id,event_time,attacker_id,damage::float8,damage_type",
        ["row_id", "actor", "t", "attacker", "damage", "type"],
    ),
    "kills": (
        "telem_kills",
        "id,victim_id,event_time,victim_rank",
        ["row_id", "actor", "t", "rank"],
    ),
    "landing": ("telem_parachute_landing", "id,account_id,event_time", ["row_id", "actor", "t"]),
    "game": (
        "telem_game_states",
        "id,event_time,is_game::float8,num_alive_players,num_alive_teams,"
        "safe_zone_x::float8,safe_zone_y::float8,safe_zone_radius::float8,"
        "poison_zone_x::float8,poison_zone_y::float8,poison_zone_radius::float8",
        [
            "row_id",
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


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n")
    tmp.replace(path)


def csv_write(path, rows, fields=None):
    with Path(path).open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def frozen(out):
    if (out / "COMPLETE.json").exists():
        raise RuntimeError("Completed run is immutable. Use a new run directory.")


def exposure_registry():
    roles, sources = defaultdict(set), []
    for folder in ["model_stgcn/checkpoints", "model_stgcn/experiment_datasets"]:
        for p in sorted((ROOT / folder).rglob("*split*.json")):
            try:
                d = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            splits = d.get("splits") if isinstance(d, dict) else None
            if not isinstance(splits, dict):
                continue
            valid = {
                k: v
                for k, v in splits.items()
                if k in ("train", "val", "validation", "test")
                and isinstance(v, list)
                and all(isinstance(x, str) for x in v)
            }
            if not valid:
                continue
            sources.append({"path": str(p.relative_to(ROOT)), "sha256": sha(p)})
            for role, ids in valid.items():
                for mid in ids:
                    roles[mid].add("validation" if role == "val" else role)
    for rel in [
        "reports/erangel_100m_40_2_2026-10-15/match_split.csv",
        "data/processed/heterograph_diffusion_v1/matches.csv",
        "data/processed/heterograph_diffusion_v3/matches.csv",
    ]:
        p = ROOT / rel
        sources.append({"path": rel, "sha256": sha(p)})
        for r in csv.DictReader(p.open()):
            roles[r["match_id"]].add(r["split"])
    return roles, sources


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    frozen(out)
    if (out / "matches.csv").exists():
        raise RuntimeError("Preparation already exists; do not reassign an active run.")
    quality = list(csv.DictReader((QUALITY / "match_quality.csv").open()))
    roles, sources = exposure_registry()
    inventory, exposures = [], []
    for r in quality:
        mid = r["match_id"]
        historic = roles[mid]
        patch, patch_evidence = infer_patch(r)
        if "test" in historic or r["locked_not_inspected"] == "True":
            role, assignment = "protected", "historical_test_priority"
        elif patch == "unknown":
            role, assignment = "quarantine", patch_evidence
        elif patch == "41.2":
            role, assignment = "reserved_patch", "later_patch_no_detail_read_not_pristine"
        elif patch == "41.1":
            role, assignment = "validation", "next_patch_development"
        elif patch == "40.2" and "validation" in historic:
            role, assignment = (
                "excluded_prior_validation",
                "never_promote_old_validation_to_training",
            )
        elif patch == "40.2":
            role, assignment = "train", "earlier_patch_training"
        else:
            raise ValueError("Patch requires an explicitly approved fold")
        eligible = (
            role in ("train", "validation")
            and float(r.get("position_rows") or 0) > 0
            and bool(r.get("start_time"))
            and float(r.get("participant_accounts") or 0) > 0
            and not float(r.get("participant_missing_id") or 0)
            and not float(r.get("participant_orphan_roster") or 0)
            and not float(r.get("participant_cross_match_roster") or 0)
            and not float(r.get("roster_team_missing") or 0)
        )
        mode = r["game_mode"]
        perspective = (
            "FPP" if "fpp" in mode else "TPP" if mode in ("solo", "duo", "squad") else "unknown"
        )
        size = {"solo": 1, "duo": 2, "squad": 4}.get(mode.replace("-fpp", ""), 0)
        inventory.append(
            {
                "match_id": mid,
                "role": role,
                "eligible": int(eligible),
                "map": r["map_name"],
                "perspective": perspective,
                "nominal_team_size": size,
                "game_mode": mode,
                "match_type": r["match_type"],
                "created_at": r["created_at"],
                "geometry_key": r["map_name"] + "__patch_" + patch + "__geometry_unverified",
                "patch": patch,
                "patch_evidence": patch_evidence,
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "br_rule_candidate": int(r["br_rule_candidate"] == "True"),
                "quality_codes": r["issue_codes"],
                "anomaly_class": r["anomaly_class"],
                "assignment": assignment,
                "historical_roles": "|".join(sorted(historic)),
            }
        )
        exposures.append(
            {
                "match_id": mid,
                "historical_roles": "|".join(sorted(historic)),
                "new_role": role,
                "role_conflict": int(len(historic) > 1),
                "pristine_test_claim": False,
            }
        )
    for directory in ("raw", "banks", "meta", "errors", "ablations", "priors"):
        (out / directory).mkdir(exist_ok=True)
    csv_write(out / "matches.csv", inventory)
    csv_write(out / "exposure_registry.csv", exposures)
    save(out / "exposure_sources.json", sources)
    spec = {
        "status": "preprocessing_only",
        "training_started": False,
        "quality_sha256": sha(QUALITY / "match_quality.csv"),
        "match_manifest_sha256": sha(out / "matches.csv"),
        "code_sha256": {
            p: sha(ROOT / p)
            for p in [
                "src/pubg_lab/world_model/data/preprocess.py",
                "src/pubg_lab/world_model/data/dataset.py",
                "src/pubg_lab/world_model/data/patches.py",
                "projects/world_model/data/world_model_preprocess.py",
            ]
        },
        "patch_policy": {
            "train": ["40.2"],
            "validation": ["41.1"],
            "reserved": ["41.2"],
            "assignment": "official PC schedule inference; executable/hotfix build unverified",
            "windows": PATCH_WINDOWS,
            "next_release_source": "https://www.pubg.com/en/news/10179",
            "author_confirmation": "2026-10-20: previous patch train, next patch validation",
        },
        "features": FEATURES,
        "game_features": GAME_FEATURES,
        "modes": MODES,
        "step_s": 10,
        "history_s": 120,
        "horizon_s": [10, 20, 30, 40, 50, 60],
        "anchor_stride_s": 30,
        "coordinate_unit": "metres",
        "clock": "event_time minus actual match_start",
        "target_interpolation": "label_only; <=15 s; same episode, segment and mode",
        "match_end_censoring": "exclude all events after a uniquely recorded match_end before state/target/prior construction; keep source raw and exclusion counts",
        "observer": "telemetry proxy; team HUD kinematics/health, ego private state; no enemy exact positions",
        "game_observability": "server zone record as-of proxy; announcement latency unverified",
        "team_slots": "variable; never truncate to 4",
        "outcomes": "label-only cross-field candidates; terminal_time remains masked without finality proof",
        "terrain": "observed transitions, not surveyed geometry or collision",
        "normalization": "not fitted; physical units retained",
        "pristine_test_available": False,
        "protected_policy": "union of historical test assignments plus reserved later patch; no detail loading",
        "exposure_scope": f"{len(sources)} split manifests; not proof all historical experiments discovered",
        "unavailable_fields": [
            "true_visibility",
            "complete_inventory_ammunition",
            "vehicle_unique_instance",
            "verified_patch_geometry",
            "terminal_elimination_time",
        ],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "queries": {
            k: f"SELECT {v[1]} FROM pubg.{v[0]} WHERE match_id=%s ORDER BY event_time,id"
            for k, v in TABLES.items()
        },
    }
    save(out / "preprocess_spec.json", spec)
    summary = {
        "inventory": len(inventory),
        "roles": dict(Counter(r["role"] for r in inventory)),
        "eligible_by_role": dict(Counter(r["role"] for r in inventory if r["eligible"])),
        "patch_counts": dict(Counter(r["patch"] for r in inventory)),
        "historical_conflicts": sum(r["role_conflict"] for r in exposures),
        "exposure_sources": len(sources),
    }
    save(out / "preparation_summary.json", summary)
    print(json.dumps(summary), flush=True)


def extract(conn, meta):
    mid = meta["match_id"]
    assert meta["role"] in ("train", "validation")
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT event_time,map_name,team_size,camera_view_type,is_event_mode "
            "FROM pubg.telem_match_start WHERE match_id=%s ORDER BY event_time,id",
            (mid,),
        )
        starts = cur.fetchall()
        if len(starts) != 1:
            raise ValueError("ambiguous_match_start")
        start = starts[0]["event_time"]
        cur.execute(
            "SELECT event_time FROM pubg.telem_match_end WHERE match_id=%s ORDER BY event_time,id",
            (mid,),
        )
        ends = cur.fetchall()
        cur.execute(
            "SELECT p.player_id,r.team_id,p.win_place,r.rank,r.won "
            "FROM pubg.participants p JOIN pubg.rosters r "
            "ON r.roster_id=p.roster_id AND r.match_id=p.match_id "
            "WHERE p.match_id=%s ORDER BY p.player_id",
            (mid,),
        )
        participants = cur.fetchall()
        ids = [r["player_id"] for r in participants]
        if len(set(ids)) != len(ids) or any(not i for i in ids):
            raise ValueError("ambiguous_actor_key")
        actor_ids = {v: i for i, v in enumerate(ids)}
        team_ids = {v: i for i, v in enumerate(sorted({r["team_id"] for r in participants}))}
        actors = [
            {
                "actor": i,
                "team": team_ids[r["team_id"]],
                "kind": 1
                if ids[i].startswith("account.")
                else 2
                if ids[i].startswith("ai.")
                else 0,
                "rank": r["win_place"],
            }
            for i, r in enumerate(participants)
        ]
        teams = {}
        for r in participants:
            team = team_ids[r["team_id"]]
            value = {"team": team, "rank": r["rank"], "won": r["won"]}
            if team in teams and teams[team] != value:
                raise ValueError("ambiguous_team_result")
            teams[team] = value
        tables, counts, dropped = {}, {}, {}
        for name, (table, cols, keys) in TABLES.items():
            cur.execute(
                f"SELECT {cols} FROM pubg.{table} WHERE match_id=%s ORDER BY event_time,id", (mid,)
            )
            kept, n, ignored = [], 0, 0
            while True:
                batch = cur.fetchmany(10000)
                if not batch:
                    break
                for values in batch:
                    n += 1
                    r = dict(zip(keys, values.values()))
                    r["t"] = (r["t"] - start).total_seconds()
                    if "actor" in r:
                        if r["actor"] not in actor_ids:
                            ignored += 1
                            continue
                        r["actor"] = actor_ids[r["actor"]]
                    if "attacker" in r:
                        r["attacker"] = actor_ids.get(r["attacker"], -1)
                    for k in (
                        "x",
                        "y",
                        "z",
                        "safe_x",
                        "safe_y",
                        "safe_r",
                        "poison_x",
                        "poison_y",
                        "poison_r",
                    ):
                        if r.get(k) is not None:
                            r[k] /= 100.0
                    invalid = [
                        k for k, v in r.items() if isinstance(v, float) and not math.isfinite(v)
                    ]
                    if invalid:
                        r["invalid_numeric_fields"] = invalid
                        for k in invalid:
                            r[k] = None
                    kept.append(r)
            tables[name], counts[name], dropped[name] = kept, n, ignored
    return {
        "meta": {
            **meta,
            "start_time": start.isoformat(),
            "end_time": ends[-1]["event_time"].isoformat() if len(ends) == 1 else None,
            "end_s": (ends[0]["event_time"] - start).total_seconds() if len(ends) == 1 else None,
            "br_rule_candidate": bool(int(meta["br_rule_candidate"])),
            "start_team_size": starts[0]["team_size"],
            "start_camera": starts[0]["camera_view_type"],
        },
        "actors": actors,
        "teams": [teams[k] for k in sorted(teams)],
        "tables": tables,
        "source_counts": counts,
        "nonparticipant_rows_excluded": dropped,
    }


def one(meta, out, password):
    mid = meta["match_id"]
    marker = out / "meta" / f"{mid}.json"
    bankpath = out / "banks" / f"{mid}.npz"
    rawpath = out / "raw" / f"{mid}.json.gz"
    if marker.exists() and bankpath.exists():
        m = json.loads(marker.read_text())
        if m.get("bank_sha256") != sha(bankpath):
            raise ValueError("cached_bank_hash_mismatch")
        return {"match_id": mid, "cached": True, **m["counts"]}
    start = time.monotonic()
    if rawpath.exists():
        with gzip.open(rawpath, "rt") as f:
            raw = json.load(f)
    else:
        conn = psycopg2.connect(
            host=os.environ.get("PG_HOST", "localhost"),
            dbname=os.environ.get("PG_DBNAME", "pubg_survival"),
            user=os.environ.get("PG_USER", "postgres"),
            password=password,
            connect_timeout=8,
            options="-c default_transaction_read_only=on -c statement_timeout=120000",
        )
        conn.set_session(readonly=True, isolation_level="REPEATABLE READ")
        try:
            with conn:
                raw = extract(conn, meta)
        finally:
            conn.close()
        tmp = rawpath.with_suffix(".tmp")
        # mtime=0 for reproducible gzip bytes; runtime timing kept outside data.
        with (
            tmp.open("wb") as handle,
            gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0, compresslevel=3) as z,
        ):
            z.write(json.dumps(raw, separators=(",", ":"), allow_nan=False).encode())
        tmp.replace(rawpath)
    live_patch, evidence = infer_patch(
        {
            "start_time": raw["meta"]["start_time"],
            "end_time": raw["meta"]["end_time"],
            "created_at": meta["created_at"],
            "shard_id": "steam",
        }
    )
    if live_patch != meta["patch"]:
        raise ValueError("live_patch_boundary_disagrees_with_manifest")
    bank, quality = build_bank(raw)
    tmp = bankpath.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **bank)
    tmp.replace(bankpath)
    quality.update(
        match_id=mid,
        role=meta["role"],
        source_counts=raw["source_counts"],
        nonparticipant_rows_excluded=raw["nonparticipant_rows_excluded"],
        raw_sha256=sha(rawpath),
        bank_sha256=sha(bankpath),
        elapsed_s=round(time.monotonic() - start, 3),
        worker_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    save(marker, quality)
    return {
        "match_id": mid,
        "cached": False,
        "worker_maxrss_bytes": quality["worker_maxrss_bytes"],
        **quality["counts"],
    }


def run(out, workers, limit, offline=False):
    frozen(out)
    spec = json.loads((out / "preprocess_spec.json").read_text())
    for rel, expected in spec["code_sha256"].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError(
                "Code changed after prepare; use a new run or explicitly version a not-started run."
            )
    rows = [r for r in csv.DictReader((out / "matches.csv").open()) if r["eligible"] == "1"]
    if limit:
        rows = rows[:limit]
    if offline:
        assert all((out / "raw" / f"{r['match_id']}.json.gz").exists() for r in rows), (
            "Offline requires all canonical raw caches."
        )
        password = None
    else:
        password = (
            os.environ.get("PG_PASSWORD")
            or os.environ.get("PGPASSWORD")
            or getpass.getpass("DB password (not saved): ")
        )
    started = time.monotonic()
    done, failed, max_worker_rss = 0, 0, 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, r, out, password): r for r in rows}
        for f in as_completed(futures):
            r = futures[f]
            try:
                result = f.result()
                max_worker_rss = max(max_worker_rss, result.get("worker_maxrss_bytes", 0))
                done += 1
            except Exception as exc:
                failed += 1
                save(
                    out / "errors" / f"{r['match_id']}.json",
                    {
                        "match_id": r["match_id"],
                        "error_class": type(exc).__name__,
                        "sqlstate": getattr(exc, "pgcode", None),
                    },
                )
            if (done + failed) % 25 == 0 or done + failed == len(rows):
                progress = {
                    "completed": done,
                    "failed": failed,
                    "expected": len(rows),
                    "seconds": round(time.monotonic() - started),
                    "workers": workers,
                    "max_worker_rss_bytes": max_worker_rss,
                    "maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                }
                save(out / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    del password


def finalize(out):
    frozen(out)
    rows = list(csv.DictReader((out / "matches.csv").open()))
    eligible = {r["match_id"]: r for r in rows if r["eligible"] == "1"}
    complete = {p.stem: p for p in (out / "meta").glob("*.json")}
    protected = {r["match_id"] for r in rows if r["role"] not in ("train", "validation")}
    assert not (protected & (complete.keys() | {p.stem for p in (out / "banks").glob("*.npz")}))
    raw_ids = {p.name.removesuffix(".json.gz") for p in (out / "raw").glob("*.json.gz")}
    assert not protected & raw_ids
    assert {r["patch"] for r in rows if r["role"] == "train"} == {"40.2"}
    assert {r["patch"] for r in rows if r["role"] == "validation"} == {"41.1"}
    if set(eligible) - set(complete):
        raise RuntimeError(
            f"Missing {len(set(eligible) - set(complete))} eligible matches; run is not complete"
        )
    db = sqlite3.connect(out / "catalog.sqlite")
    db.executescript("""
      CREATE TABLE matches(match_id TEXT PRIMARY KEY,role TEXT,map TEXT,perspective TEXT,
       nominal_team_size INTEGER,geometry_key TEXT,anchors INTEGER,valid_states INTEGER,
       raw_sha256 TEXT,bank_sha256 TEXT);
      CREATE TABLE issues(match_id TEXT,actor INTEGER,start_s REAL,end_s REAL,code TEXT,action TEXT);
      CREATE TABLE outcomes(match_id TEXT,team INTEGER,rank INTEGER,won INTEGER,
       rank_mask INTEGER,winner_mask INTEGER,terminal_time_mask INTEGER,evidence_json TEXT);
      CREATE TABLE spatial_support(match_id TEXT,geometry_key TEXT,sx INTEGER,sy INTEGER,
       dx INTEGER,dy INTEGER,mode INTEGER,n INTEGER);
      CREATE INDEX issues_lookup ON issues(match_id,actor,start_s);
      CREATE INDEX outcomes_lookup ON outcomes(match_id,team);
      CREATE INDEX support_lookup ON spatial_support(geometry_key,sx,sy);
    """)
    totals = Counter()
    modes = Counter()
    validations = []
    fit_ids = []
    fit_vocab = set()
    for i, (mid, r) in enumerate(sorted(eligible.items())):
        m = json.loads(complete[mid].read_text())
        with np.load(out / "banks" / f"{mid}.npz", allow_pickle=False) as b:
            assert (
                b["source_time"][b["valid"]]
                <= np.broadcast_to(b["times"][:, None], b["valid"].shape)[b["valid"]]
            ).all()
            assert b["x"].shape[:2] == b["valid"].shape
            totals.update(m["counts"])
            modes.update(Counter(b["mobility"][b["valid"]].tolist()))
            db.execute(
                "INSERT INTO matches VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    mid,
                    r["role"],
                    r["map"],
                    r["perspective"],
                    r["nominal_team_size"],
                    r["geometry_key"],
                    m["counts"]["anchors"],
                    m["counts"]["valid_states"],
                    m["raw_sha256"],
                    m["bank_sha256"],
                ),
            )
            db.executemany(
                "INSERT INTO issues VALUES(?,?,?,?,?,?)", [(mid, *v) for v in m["issues"]]
            )
            db.executemany(
                "INSERT INTO outcomes VALUES(?,?,?,?,?,?,?,?)",
                [
                    (
                        mid,
                        v["team"],
                        v["rank"],
                        v["won"],
                        v["rank_mask"],
                        v["winner_mask"],
                        v["terminal_time_mask"],
                        json.dumps(v),
                    )
                    for v in m["outcomes"]
                ],
            )
            if r["role"] == "train":
                fit_ids.append(mid)
                fit_vocab.update(m["item_vocabulary_local"])
                db.executemany(
                    "INSERT INTO spatial_support VALUES(?,?,?,?,?,?,?,?)",
                    [(mid, r["geometry_key"], *map(int, e)) for e in b["spatial_edges"]],
                )
            if len(validations) < 12 and len(b["anchors"]):
                k, p = map(int, b["anchors"][0])
                obs = view_at(b, k, p, "observer")
                server = view_at(b, k, p, "server")
                y = targets_at(b, k, p)
                assert np.isfinite(obs["x"]).all() and np.isfinite(server["x"]).all()
                validations.append(
                    {
                        "match_id": mid,
                        "role": r["role"],
                        "player_count": len(b["team"]),
                        "observer_shape": list(obs["x"].shape),
                        "server_shape": list(server["x"].shape),
                        "target_shape": list(y["xy"].shape),
                    }
                )
        if i % 100 == 0:
            db.commit()
    save(
        out / "priors" / "fit_manifest.json",
        {
            "match_ids": fit_ids,
            "source": "train only",
            "geometry": "map x schedule-inferred patch; no asserted patch geometry",
            "inferred_connections": "not materialized as observed labels; generate adjacency with unknown support",
        },
    )
    save(
        out / "priors" / "item_vocabulary.json",
        {
            "unknown": 1,
            "padding": 0,
            "tokens": {v: i + 2 for i, v in enumerate(sorted(fit_vocab))},
            "fit_match_ids_sha256": hashlib.sha256("\n".join(sorted(fit_ids)).encode()).hexdigest(),
        },
    )
    ablations = []
    fit = [r for r in rows if r["eligible"] == "1" and r["role"] == "train"]
    dev = [r for r in rows if r["eligible"] == "1" and r["role"] == "validation"]
    cases = [("full", lambda r: True, lambda r: True)]
    for field in ("map", "perspective", "nominal_team_size"):
        for value in sorted({r[field] for r in fit + dev}):
            cases.append(
                (
                    f"only_{field}_{value}",
                    lambda r, f=field, v=value: r[f] == v,
                    lambda r, f=field, v=value: r[f] == v,
                )
            )
            cases.append(
                (
                    f"holdout_{field}_{value}",
                    lambda r, f=field, v=value: r[f] != v,
                    lambda r, f=field, v=value: r[f] == v,
                )
            )
    for name in (
        "no_spatial",
        "observed_edges_only",
        "no_team_node",
        "no_outcome_loss",
        "short_history_30s",
        "no_pretraining",
        "observer",
        "server",
        "no_map_condition",
        "no_perspective_condition",
        "no_team_size_condition",
        "non_BR_excluded",
    ):
        cases.append(
            (
                name,
                lambda r, n=name: n != "non_BR_excluded" or r["br_rule_candidate"] == "1",
                lambda r: True,
            )
        )
    for name, ff, df in cases:
        train = [r["match_id"] for r in fit if ff(r)]
        val = [r["match_id"] for r in dev if df(r)]
        assert not set(train) & set(val) and not (set(train) | set(val)) & protected
        record = {
            "name": name,
            "train_match_ids": train,
            "validation_match_ids": val,
            "test_match_ids": [],
            "training_executed": False,
            "fit_priors_and_vocab_must_use_exact_train_ids": True,
            "global_prior_reuse_allowed": set(train) == set(fit_ids),
            "train_patches": sorted({r["patch"] for r in fit if ff(r)}),
            "validation_patches": sorted({r["patch"] for r in dev if df(r)}),
            "status": "ready_manifest" if train and val else "unavailable_empty_domain",
        }
        save(out / "ablations" / f"{name}.json", record)
        ablations.append(
            {"name": name, "train": len(train), "validation": len(val), "status": record["status"]}
        )
    csv_write(out / "ablation_summary.csv", ablations)
    db.commit()
    label_counts = dict(
        db.execute(
            "SELECT 'rank',sum(rank_mask) FROM outcomes UNION ALL SELECT 'winner',sum(winner_mask) FROM outcomes"
        )
    )
    db.close()
    summary = {
        "training_started": False,
        "preprocessed_matches": len(complete),
        "excluded_or_protected_matches": len(protected),
        "totals": dict(totals),
        "mobility_states": {MODES[k]: v for k, v in sorted(modes.items())},
        "label_counts": label_counts,
        "ablations": len(ablations),
        "runtime_examples": validations,
        "pristine_final_test": False,
        "test_detail_reads": 0,
    }
    save(out / "summary.json", summary)
    save(
        out / "COMPLETE.json",
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "summary_sha256": sha(out / "summary.json"),
        },
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "run", "finalize"])
    p.add_argument("--output", type=Path, default=DEFAULT)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int)
    p.add_argument("--offline", action="store_true")
    a = p.parse_args()
    if a.command == "prepare":
        prepare(a.output)
    elif a.command == "run":
        run(a.output, a.workers, a.limit, a.offline)
    else:
        finalize(a.output)


if __name__ == "__main__":
    main()
