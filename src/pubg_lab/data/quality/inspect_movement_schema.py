"""Export live database metadata and bounded movement-study availability checks, read-only."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import getpass
import json
import os
from pathlib import Path
import psycopg2

ROOT = REPO_ROOT
OUT = ROOT / "reports/movement_schema_2026-10-15"


def main():
    password = os.environ.get("PG_PASSWORD") or getpass.getpass("DB password: ")
    conn = psycopg2.connect(
        host=os.environ.get("PG_HOST", "localhost"),
        dbname=os.environ.get("PG_DBNAME", "pubg_survival"),
        user=os.environ.get("PG_USER", "postgres"),
        port=os.environ.get("PG_PORT", "5432"),
        password=password,
        connect_timeout=5,
        options="-c default_transaction_read_only=on -c statement_timeout=30000",
    )
    del password
    result = {
        "inspected_date": "2026-10-15",
        "scope": "pubg and stdb schemas; partition child tables excluded",
    }
    with conn, conn.cursor() as cur:

        def query(sql, args=None):
            cur.execute(sql, args)
            columns = [d[0] for d in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

        result["relations"] = (
            query("""SELECT n.nspname AS schema, c.relname AS name, c.relkind AS kind,
            c.reltuples::bigint AS estimated_rows FROM pg_class c JOIN pg_namespace n ON c.relnamespace=n.oid
            WHERE n.nspname IN ('pubg','stdb') AND c.relkind IN ('r','p','v','m') AND NOT c.relispartition
            ORDER BY 1,2""")
        )
        result["columns"] = (
            query("""SELECT table_schema AS schema, table_name AS table, ordinal_position,
            column_name, data_type, udt_name, is_nullable FROM information_schema.columns
            WHERE table_schema IN ('pubg','stdb') AND NOT EXISTS
            (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
            WHERE n.nspname=table_schema AND c.relname=table_name AND c.relispartition)
            ORDER BY table_schema, table_name, ordinal_position""")
        )
        result["constraints"] = (
            query("""SELECT n.nspname AS schema, c.relname AS table, con.conname AS name,
            con.contype AS type, pg_get_constraintdef(con.oid) AS definition FROM pg_constraint con
            JOIN pg_class c ON c.oid=con.conrelid JOIN pg_namespace n ON n.oid=c.relnamespace
            WHERE n.nspname IN ('pubg','stdb') AND NOT c.relispartition ORDER BY 1,2,3""")
        )
        result["match_population"] = (
            query("""SELECT map_name, game_mode, match_type, count(*) AS matches
            FROM pubg.matches GROUP BY 1,2,3 ORDER BY 1,2,3""")
        )
        mids = [
            r["match_id"]
            for r in json.loads(
                (ROOT / "reports/bot_trajectory_audit_2026-10-15/audit.json").read_text()
            )["matches"]
        ]
        result["sample_match_ids"] = mids
        result["sample_game_states"] = query(
            """SELECT match_id, count(*) AS rows,
            count(*) FILTER (WHERE safe_zone_radius>0) AS blue_valid_rows,
            count(*) FILTER (WHERE poison_zone_radius>0) AS white_valid_rows
            FROM pubg.telem_game_states WHERE match_id=ANY(%s) GROUP BY 1 ORDER BY 1""",
            (mids,),
        )
        result["sample_vehicle_values"] = query(
            """SELECT vehicle_type, vehicle_id, count(*) AS rows
            FROM pubg.telem_positions WHERE match_id=ANY(%s) GROUP BY 1,2 ORDER BY 3 DESC""",
            (mids,),
        )
        result["sample_phase_clock"] = query(
            """SELECT count(*) AS rows,
            count(elapsed_time) AS elapsed_time_nonnull, count(event_time) AS event_time_nonnull
            FROM pubg.telem_phase_changes WHERE match_id=ANY(%s)""",
            (mids,),
        )
        result["sample_match_start"] = query(
            """SELECT match_id, event_time, blue_zone_options
            FROM pubg.telem_match_start WHERE match_id=ANY(%s) ORDER BY 1""",
            (mids,),
        )
        result["sample_landing_coverage"] = query(
            """SELECT
            CASE WHEN p.player_id LIKE 'ai.%%' THEN 'bot' WHEN p.player_id LIKE 'account.%%' THEN 'human' ELSE 'other' END AS class,
            count(*) AS participant_rows, count(*) FILTER (WHERE l.n>0) AS with_landing,
            count(*) FILTER (WHERE l.n>1) AS multiple_landing_rows,
            count(*) FILTER (WHERE l.distinct_times>1) AS multiple_landing_timestamps,
            sum(l.n-l.distinct_events) AS duplicate_landing_rows
            FROM pubg.participants p LEFT JOIN LATERAL
            (SELECT count(*) AS n, count(DISTINCT event_time) AS distinct_times,
                count(DISTINCT (event_time,pos_x,pos_y,pos_z,distance)) AS distinct_events
                FROM pubg.telem_parachute_landing l WHERE l.match_id=p.match_id AND l.account_id=p.player_id) l ON true
            WHERE p.match_id=ANY(%s) GROUP BY 1""",
            (mids,),
        )
        result["sample_duplicate_participant_accounts"] = query(
            """SELECT match_id, player_id, count(*) AS rows
            FROM pubg.participants WHERE match_id=ANY(%s) GROUP BY 1,2 HAVING count(*)>1""",
            (mids,),
        )
    conn.close()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "schema_snapshot.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    print("Relations:", len(result["relations"]), "Columns:", len(result["columns"]))
    print("Relations:", ", ".join(r["schema"] + "." + r["name"] for r in result["relations"]))
    print("Phase clock:", result["sample_phase_clock"])
    print("Landing:", result["sample_landing_coverage"])
    print("Vehicle values:", result["sample_vehicle_values"][:15])


if __name__ == "__main__":
    main()
