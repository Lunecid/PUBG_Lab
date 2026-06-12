"""
stdb/build_available.py
=======================
Resumable batch builder for converting raw PUBG telemetry into the base STDB
and the RQ1 phase-level feature layer.

Targets:
    baltic-squad-tpp  -> Baltic_Main / squad (third-person squad)
    squad-crossmap    -> squad / squad-fpp, excluding Tutorial_Main and Range_Main
    squad-tpp-crossmap -> squad, excluding Tutorial_Main and Range_Main
    squad-fpp-crossmap -> squad-fpp, excluding Tutorial_Main and Range_Main
    tpp-classic       -> solo/duo/squad, excluding Tutorial_Main and Range_Main
    fpp-classic       -> solo-fpp/duo-fpp/squad-fpp, excluding Tutorial_Main and Range_Main
    classic-crossmap  -> solo/duo/squad in both TPP and FPP, excluding Tutorial_Main and Range_Main

The script skips matches that already have RQ1 labels unless --refresh is set.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import time
from pathlib import Path

from stdb.conn import apply_sql_file, connect
from stdb.etl import ingest_match
from stdb.rq1_feature_etl import SCHEMA_PATH as RQ1_SCHEMA_PATH
from stdb.rq1_feature_etl import build_match as build_rq1_features

BASE_SCHEMA_PATH = Path(__file__).with_name("schema.sql")

logger = logging.getLogger("stdb.build_available")


TARGET_SQL = {
    "baltic-squad-tpp": """
        SELECT m.match_id
          FROM pubg.matches m
          JOIN pubg.collection_log c ON c.match_id = m.match_id
         WHERE c.telemetry_fetched = TRUE
           AND m.map_name = 'Baltic_Main'
           AND m.game_mode = 'squad'
         ORDER BY m.created_at, m.match_id
    """,
    "squad-crossmap": """
        SELECT m.match_id
          FROM pubg.matches m
          JOIN pubg.collection_log c ON c.match_id = m.match_id
         WHERE c.telemetry_fetched = TRUE
           AND m.game_mode IN ('squad', 'squad-fpp')
           AND m.map_name NOT IN ('Tutorial_Main', 'Range_Main')
         ORDER BY m.map_name, m.game_mode, m.created_at, m.match_id
    """,
    "squad-tpp-crossmap": """
        SELECT m.match_id
          FROM pubg.matches m
          JOIN pubg.collection_log c ON c.match_id = m.match_id
         WHERE c.telemetry_fetched = TRUE
           AND m.game_mode = 'squad'
           AND m.map_name NOT IN ('Tutorial_Main', 'Range_Main')
         ORDER BY m.map_name, m.created_at, m.match_id
    """,
    "squad-fpp-crossmap": """
        SELECT m.match_id
          FROM pubg.matches m
          JOIN pubg.collection_log c ON c.match_id = m.match_id
         WHERE c.telemetry_fetched = TRUE
           AND m.game_mode = 'squad-fpp'
           AND m.map_name NOT IN ('Tutorial_Main', 'Range_Main')
         ORDER BY m.map_name, m.created_at, m.match_id
    """,
    "tpp-classic": """
        SELECT m.match_id
          FROM pubg.matches m
          JOIN pubg.collection_log c ON c.match_id = m.match_id
         WHERE c.telemetry_fetched = TRUE
           AND m.game_mode IN ('solo', 'duo', 'squad')
           AND m.map_name NOT IN ('Tutorial_Main', 'Range_Main')
         ORDER BY m.map_name, m.game_mode, m.created_at, m.match_id
    """,
    "fpp-classic": """
        SELECT m.match_id
          FROM pubg.matches m
          JOIN pubg.collection_log c ON c.match_id = m.match_id
         WHERE c.telemetry_fetched = TRUE
           AND m.game_mode IN ('solo-fpp', 'duo-fpp', 'squad-fpp')
           AND m.map_name NOT IN ('Tutorial_Main', 'Range_Main')
         ORDER BY m.map_name, m.game_mode, m.created_at, m.match_id
    """,
    "classic-crossmap": """
        SELECT m.match_id
          FROM pubg.matches m
          JOIN pubg.collection_log c ON c.match_id = m.match_id
         WHERE c.telemetry_fetched = TRUE
           AND m.game_mode IN ('solo', 'solo-fpp', 'duo', 'duo-fpp', 'squad', 'squad-fpp')
           AND m.map_name NOT IN ('Tutorial_Main', 'Range_Main')
         ORDER BY m.map_name, m.game_mode, m.created_at, m.match_id
    """,
}


def select_match_ids(
    target: str,
    *,
    limit: int | None,
    refresh: bool,
    shards: int = 1,
    shard_index: int = 0,
) -> list[str]:
    if target not in TARGET_SQL:
        raise ValueError(f"unknown target: {target}")
    if shards < 1:
        raise ValueError("--shards must be >= 1")
    if shard_index < 0 or shard_index >= shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < shards")

    sql = TARGET_SQL[target]
    if not refresh:
        sql = f"""
            WITH target AS ({sql})
            SELECT t.match_id
              FROM target t
             WHERE NOT EXISTS (
                   SELECT 1
                     FROM stdb.team_phase_labels l
                    WHERE l.match_id = t.match_id
             )
               AND NOT EXISTS (
                   SELECT 1
                     FROM stdb.matches m
                    WHERE m.match_id = t.match_id
                      AND NOT EXISTS (
                          SELECT 1
                            FROM stdb.moving_objects mo
                           WHERE mo.match_id = t.match_id
                             AND mo.agent_kind = 'team'
                      )
               )
             ORDER BY t.match_id
        """
    params: tuple[object, ...] = ()

    conn = connect()
    cur = conn.cursor()
    try:
        cur.execute(sql, params)
        match_ids = [r[0] for r in cur.fetchall()]
        if shards > 1:
            match_ids = [
                mid
                for mid in match_ids
                if int(hashlib.md5(mid.encode("utf-8")).hexdigest()[:8], 16) % shards == shard_index
            ]
        if limit:
            match_ids = match_ids[:limit]
        return match_ids
    finally:
        cur.close()
        conn.close()


def target_status(target: str) -> dict:
    if target not in TARGET_SQL:
        raise ValueError(f"unknown target: {target}")

    conn = connect()
    cur = conn.cursor()
    try:
        cur.execute(f"WITH target AS ({TARGET_SQL[target]}) SELECT count(*) FROM target")
        total = int(cur.fetchone()[0])
        cur.execute(
            f"""
            WITH target AS ({TARGET_SQL[target]})
            SELECT count(DISTINCT l.match_id)
              FROM target t
              JOIN stdb.team_phase_labels l ON l.match_id = t.match_id
            """
        )
        feature_done = int(cur.fetchone()[0])
        cur.execute(
            f"""
            WITH target AS ({TARGET_SQL[target]})
            SELECT count(DISTINCT m.match_id)
              FROM target t
              JOIN stdb.matches m ON m.match_id = t.match_id
            """
        )
        base_done = int(cur.fetchone()[0])
        cur.execute(
            f"""
            WITH target AS ({TARGET_SQL[target]})
            SELECT count(DISTINCT m.match_id)
              FROM target t
              JOIN stdb.matches m ON m.match_id = t.match_id
             WHERE NOT EXISTS (
                   SELECT 1
                     FROM stdb.moving_objects mo
                    WHERE mo.match_id = m.match_id
                      AND mo.agent_kind = 'team'
             )
            """
        )
        unusable_empty = int(cur.fetchone()[0])
        return {
            "target": target,
            "total": total,
            "base_done": base_done,
            "feature_done": feature_done,
            "unusable_empty": unusable_empty,
            "remaining": total - feature_done - unusable_empty,
        }
    finally:
        cur.close()
        conn.close()


def build_batch(match_ids: list[str]) -> dict:
    ok = 0
    failed = 0
    failures: list[dict] = []
    started = time.time()

    for idx, match_id in enumerate(match_ids, start=1):
        one_started = time.time()
        base = ingest_match(match_id)
        if not base.get("ok"):
            failed += 1
            failures.append({"match_id": match_id, "stage": "base", "result": base})
            logger.error("[%s/%s] base failed %s", idx, len(match_ids), base)
            continue

        rq1 = build_rq1_features(match_id)
        if not rq1.get("ok"):
            failed += 1
            failures.append({"match_id": match_id, "stage": "rq1", "result": rq1})
            logger.error("[%s/%s] rq1 failed %s", idx, len(match_ids), rq1)
            continue

        ok += 1
        elapsed = round(time.time() - one_started, 2)
        logger.info(
            "[%s/%s] OK %s elapsed=%ss base_rows=(player:%s team:%s event:%s prox:%s) rq1_labels=%s",
            idx,
            len(match_ids),
            match_id,
            elapsed,
            base.get("player_rows"),
            base.get("team_rows"),
            base.get("event_rows"),
            base.get("proximity_rows"),
            rq1.get("team_phase_labels"),
        )

    return {
        "ok": ok,
        "failed": failed,
        "failures": failures[:10],
        "elapsed_s": round(time.time() - started, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=sorted(TARGET_SQL),
        default="baltic-squad-tpp",
        help="Raw match set to convert",
    )
    parser.add_argument("--limit", type=int, help="Build at most N remaining matches")
    parser.add_argument("--refresh", action="store_true", help="Rebuild matches even if RQ1 labels exist")
    parser.add_argument("--shards", type=int, default=1, help="Split selected remaining matches into N stable shards")
    parser.add_argument("--shard-index", type=int, default=0, help="Build only this zero-based shard index")
    parser.add_argument("--apply-schema", action="store_true", help="Apply base and RQ1 schemas before building")
    parser.add_argument("--status", action="store_true", help="Print target conversion status and exit")
    parser.add_argument("--dry-run", action="store_true", help="List selected count without building")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.apply_schema:
        apply_sql_file(BASE_SCHEMA_PATH)
        apply_sql_file(RQ1_SCHEMA_PATH)

    status = target_status(args.target)
    logger.info("status before: %s", status)
    if args.status:
        return 0

    match_ids = select_match_ids(
        args.target,
        limit=args.limit,
        refresh=args.refresh,
        shards=args.shards,
        shard_index=args.shard_index,
    )
    logger.info("selected %s matches (shard %s/%s)", len(match_ids), args.shard_index, args.shards)
    if args.dry_run:
        return 0

    result = build_batch(match_ids)
    logger.info("batch result: %s", result)
    logger.info("status after: %s", target_status(args.target))
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
