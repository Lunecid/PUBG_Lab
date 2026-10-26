"""Register current pubg.* as the preserved PUBG L1 ground layer.

This script is deliberately light:

* it applies the additive br_* schema when requested;
* it records pubg.* table provenance in br_raw.l1_table_registry;
* it materializes only match-level rows into br_ground.matches.

It does not copy the massive telemetry event tables.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT

import argparse
import sys
from pathlib import Path
from typing import Any

from psycopg2.extras import Json

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))

from pubg_lab.data.stdb.conn import apply_sql_file, connect


SOURCE_ID = "pubg_current_l1"

EVENT_KIND_BY_TABLE = {
    "matches": "match_metadata",
    "participants": "participant_outcome",
    "rosters": "team_outcome",
    "collection_log": "collection_provenance",
    "telem_match_start": "match_start",
    "telem_match_end": "match_end",
    "telem_positions": "player_position",
    "telem_game_states": "zone_state",
    "telem_phase_changes": "phase_change",
    "telem_damage": "damage",
    "telem_groggy": "groggy",
    "telem_kills": "kill",
    "telem_item_pickup": "item_pickup",
    "telem_item_equip": "item_equip",
    "telem_item_use": "item_use",
    "telem_parachute_landing": "landing",
}

KNOWN_L1_LIMITATIONS = [
    "byte-identical raw telemetry JSON is not preserved in the DB",
    "raw event ordering index is not preserved for every telemetry event",
    "some official telemetry event families are not normalized into pubg.*",
    "full nested Character/common/Vehicle payloads are partially flattened",
]


def table_rows_estimate(cur, table_name: str) -> int | None:
    cur.execute(
        """
        SELECT c.reltuples::BIGINT
          FROM pg_class c
          JOIN pg_namespace n ON n.oid = c.relnamespace
         WHERE n.nspname = 'pubg'
           AND c.relname = %s
        """,
        (table_name,),
    )
    row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else None


def exact_count(cur, table_name: str) -> int:
    cur.execute(f'SELECT count(*) FROM pubg."{table_name}"')
    return int(cur.fetchone()[0])


def exact_distinct_matches(cur, table_name: str, has_match_id: bool) -> int | None:
    if not has_match_id:
        return None
    cur.execute(f'SELECT count(DISTINCT match_id) FROM pubg."{table_name}"')
    return int(cur.fetchone()[0])


def pubg_tables(cur) -> list[dict[str, Any]]:
    cur.execute(
        """
        SELECT
            t.table_name,
            t.table_type,
            EXISTS (
                SELECT 1
                  FROM information_schema.columns c
                 WHERE c.table_schema = t.table_schema
                   AND c.table_name = t.table_name
                   AND c.column_name = 'match_id'
            ) AS has_match_id
          FROM information_schema.tables t
         WHERE t.table_schema = 'pubg'
         ORDER BY t.table_name
        """
    )
    return [
        {"table_name": row[0], "table_type": row[1], "has_match_id": bool(row[2])}
        for row in cur.fetchall()
    ]


def register(args: argparse.Namespace) -> None:
    if args.apply_schema:
        apply_sql_file(ROOT / "stdb" / "schema_br_layers.sql")

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO br_raw.sources (
                    source_id, source_game, provider, canonical_url,
                    observability, notes
                ) VALUES (
                    %s, 'PUBG: Battlegrounds', 'KRAFTON / PUBG API',
                    'https://documentation.pubg.com/',
                    'full_normalized_telemetry_l1',
                    %s
                )
                ON CONFLICT (source_id) DO UPDATE
                SET observability = EXCLUDED.observability,
                    notes = br_raw.sources.notes || EXCLUDED.notes
                """,
                (
                    SOURCE_ID,
                    Json(
                        {
                            "local_schema": "pubg",
                            "ground_role": "soil/raw-normalized layer; preserve as source of truth",
                            "limitations": KNOWN_L1_LIMITATIONS,
                        }
                    ),
                ),
            )
            cur.execute(
                """
                INSERT INTO br_raw.artifacts (
                    artifact_id, source_id, layer_name, artifact_kind, uri,
                    local_path, content_format, metadata
                ) VALUES (
                    'pubg_current_l1_database', %s, 'L1', 'postgres_schema',
                    'postgresql://pubg_survival/pubg',
                    'pubg.*', 'postgresql', %s
                )
                ON CONFLICT (artifact_id) DO UPDATE
                SET metadata = EXCLUDED.metadata,
                    registered_at = now()
                """,
                (
                    SOURCE_ID,
                    Json({"schema": "pubg", "limitations": KNOWN_L1_LIMITATIONS}),
                ),
            )

            rows = []
            for table in pubg_tables(cur):
                table_name = table["table_name"]
                if table["table_type"] != "BASE TABLE":
                    continue
                if args.exact:
                    row_count = exact_count(cur, table_name)
                    distinct_matches = exact_distinct_matches(
                        cur, table_name, table["has_match_id"]
                    )
                else:
                    row_count = table_rows_estimate(cur, table_name)
                    distinct_matches = None
                rows.append((table_name, row_count, distinct_matches, table["has_match_id"]))

            for table_name, row_count, distinct_matches, _has_match_id in rows:
                cur.execute(
                    """
                    INSERT INTO br_raw.l1_table_registry (
                        source_id, source_table, normalized_event_kind,
                        rows_estimate, distinct_matches_estimate,
                        consistency_level, missing_from_official_raw, notes
                    ) VALUES (
                        %s, %s, %s, %s, %s, 'normalized_raw', %s, %s
                    )
                    ON CONFLICT (source_id, source_table) DO UPDATE
                    SET normalized_event_kind = EXCLUDED.normalized_event_kind,
                        rows_estimate = EXCLUDED.rows_estimate,
                        distinct_matches_estimate = EXCLUDED.distinct_matches_estimate,
                        consistency_level = EXCLUDED.consistency_level,
                        missing_from_official_raw = EXCLUDED.missing_from_official_raw,
                        notes = EXCLUDED.notes,
                        audited_at = now()
                    """,
                    (
                        SOURCE_ID,
                        f"pubg.{table_name}",
                        EVENT_KIND_BY_TABLE.get(table_name, "pubg_l1_table"),
                        row_count,
                        distinct_matches,
                        KNOWN_L1_LIMITATIONS,
                        "Registered by scripts/register_pubg_l1_ground.py",
                    ),
                )

            cur.execute(
                """
                INSERT INTO br_ground.matches (
                    br_match_id, source_id, source_match_id, source_game,
                    map_name, mode_name, perspective, team_size,
                    started_at, duration_s, coordinate_unit,
                    coordinate_scale_to_m, observability, raw_artifact_id,
                    metadata
                )
                SELECT
                    br_match_id, source_id, source_match_id, source_game,
                    map_name, mode_name, perspective, team_size,
                    started_at, duration_s, coordinate_unit,
                    coordinate_scale_to_m, observability,
                    'pubg_current_l1_database',
                    metadata
                  FROM br_ground.v_pubg_matches_from_existing_l1
                ON CONFLICT (br_match_id) DO UPDATE
                SET map_name = EXCLUDED.map_name,
                    mode_name = EXCLUDED.mode_name,
                    perspective = EXCLUDED.perspective,
                    team_size = EXCLUDED.team_size,
                    started_at = EXCLUDED.started_at,
                    duration_s = EXCLUDED.duration_s,
                    observability = EXCLUDED.observability,
                    raw_artifact_id = EXCLUDED.raw_artifact_id,
                    metadata = EXCLUDED.metadata
                """
            )
            materialized = cur.rowcount
        conn.commit()

    print(f"registered pubg.* L1 tables: {len(rows)}")
    print(f"materialized br_ground.matches rows touched: {materialized}")
    print("heavy telemetry rows were not copied")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-schema", action="store_true")
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Use exact count(*) and count(distinct match_id). This can be slow on large telemetry tables.",
    )
    args = parser.parse_args()
    register(args)


if __name__ == "__main__":
    main()
