"""Read-only bounded audit of evidence available for player decision states."""

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
from pathlib import Path

import psycopg2

ROOT = REPO_ROOT
OUT = ROOT / "reports/decision_state_audit_2026-10-15"


def main():
    rows = list(
        csv.DictReader((ROOT / "reports/erangel_100m_40_2_2026-10-15/match_split.csv").open())
    )
    selected = [
        r
        for r in rows
        if r["split"] == "train" and r["match_type"] == "official" and r["mode"] == "squad"
    ]
    selected.sort(
        key=lambda r: hashlib.sha256(("decision-audit-v1:" + r["match_id"]).encode()).hexdigest()
    )
    mids = [r["match_id"] for r in selected[:12]]
    password = getpass.getpass("DB password: ")
    conn = psycopg2.connect(
        host="localhost",
        dbname="pubg_survival",
        user="postgres",
        password=password,
        connect_timeout=5,
        options="-c default_transaction_read_only=on -c statement_timeout=30000",
    )
    del password
    report = {
        "scope": "12 deterministic training-only inferred-40.2 official TPP squad matches; schema/event availability, not full inventory reconstruction",
        "match_ids": mids,
    }
    with conn, conn.cursor() as cur:

        def query(sql, params=None):
            cur.execute(sql, params)
            keys = [d[0] for d in cur.description]
            return [dict(zip(keys, r)) for r in cur.fetchall()]

        cur.execute("SHOW transaction_read_only")
        assert cur.fetchone()[0] == "on"
        report["event_types"] = {}
        for table in ["telem_item_equip", "telem_item_pickup"]:
            report["event_types"][table] = query(
                f"""SELECT event_type,category,sub_category,count(*) AS rows,
                count(*) FILTER (WHERE item_id IS NULL) AS missing_item,
                count(*) FILTER (WHERE attached_items IS NOT NULL) AS attachments_nonnull
                FROM pubg.{table} WHERE match_id=ANY(%s) GROUP BY 1,2,3 ORDER BY 4 DESC""",
                (mids,),
            )
        report["item_use"] = query(
            """SELECT category,sub_category,count(*) AS rows FROM pubg.telem_item_use
            WHERE match_id=ANY(%s) GROUP BY 1,2 ORDER BY 3 DESC""",
            (mids,),
        )
        report["damage_types"] = query(
            """SELECT damage_type,damage_reason,count(*) AS rows,
            count(*) FILTER (WHERE attacker_id IS NOT NULL) AS attacker_nonnull,
            count(*) FILTER (WHERE attacker_x IS NOT NULL AND attacker_y IS NOT NULL) AS attacker_xy_nonnull,
            count(*) FILTER (WHERE damage>0) AS positive_damage
            FROM pubg.telem_damage WHERE match_id=ANY(%s) GROUP BY 1,2 ORDER BY 3 DESC""",
            (mids,),
        )
        report["vehicle_values"] = query(
            """SELECT vehicle_type,vehicle_id,count(*) AS rows,
            count(vehicle_speed) AS speed_nonnull FROM pubg.telem_positions
            WHERE match_id=ANY(%s) GROUP BY 1,2 ORDER BY 3 DESC""",
            (mids,),
        )
        report["player_event_coverage"] = query(
            """SELECT
            CASE WHEN p.player_id LIKE 'ai.%%' THEN 'bot' WHEN p.player_id LIKE 'account.%%' THEN 'human' ELSE 'unknown' END AS actor_class,
            count(*) AS participant_rows,
            count(*) FILTER (WHERE EXISTS (SELECT 1 FROM pubg.telem_item_equip e WHERE e.match_id=p.match_id AND e.account_id=p.player_id)) AS with_equip,
            count(*) FILTER (WHERE EXISTS (SELECT 1 FROM pubg.telem_item_pickup e WHERE e.match_id=p.match_id AND e.account_id=p.player_id)) AS with_pickup_or_drop,
            count(*) FILTER (WHERE EXISTS (SELECT 1 FROM pubg.telem_damage e WHERE e.match_id=p.match_id AND e.victim_id=p.player_id AND e.damage>0)) AS with_received_damage
            FROM pubg.participants p WHERE p.match_id=ANY(%s) GROUP BY 1""",
            (mids,),
        )
    conn.close()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    print(
        json.dumps(
            {
                "scope": report["scope"],
                "event_type_values": {
                    k: sorted({r["event_type"] for r in v})
                    for k, v in report["event_types"].items()
                },
                "player_event_coverage": report["player_event_coverage"],
                "vehicle_values": report["vehicle_values"][:8],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
