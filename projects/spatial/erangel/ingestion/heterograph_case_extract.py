"""Read-only, one-match evidence export. Credentials are prompted, never stored."""

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
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import psycopg2

ROOT = REPO_ROOT
OUT = ROOT / "reports/heterograph_case_40_2_2026-10-15"


def main():
    mid = json.loads((ROOT / "reports/decision_state_audit_2026-10-15/audit.json").read_text())[
        "match_ids"
    ][0]
    meta = next(
        r
        for r in csv.DictReader(
            (ROOT / "reports/erangel_100m_40_2_2026-10-15/match_split.csv").open()
        )
        if r["match_id"] == mid
    )
    pw = getpass.getpass("DB password: ")
    conn = psycopg2.connect(
        host="localhost",
        dbname="pubg_survival",
        user="postgres",
        password=pw,
        connect_timeout=5,
        options="-c default_transaction_read_only=on -c statement_timeout=60000",
    )
    del pw
    queries = []
    with conn, conn.cursor() as cur:

        def q(sql, args=(mid,)):
            queries.append({"sql": sql, "parameters": list(args)})
            cur.execute(sql, args)
            keys = [c[0] for c in cur.description]
            return [dict(zip(keys, r)) for r in cur.fetchall()]

        cur.execute("SHOW transaction_read_only")
        assert cur.fetchone()[0] == "on"
        roster = q(
            "SELECT p.player_id,r.team_id FROM pubg.participants p JOIN pubg.rosters r ON r.roster_id=p.roster_id WHERE p.match_id=%s"
        )
        teams = {r["player_id"]: r["team_id"] for r in roster if r["player_id"]}
        positions = q("SELECT * FROM pubg.telem_positions WHERE match_id=%s ORDER BY event_time,id")
        damages = q("SELECT * FROM pubg.telem_damage WHERE match_id=%s ORDER BY event_time,id")
        equips = q("SELECT * FROM pubg.telem_item_equip WHERE match_id=%s ORDER BY event_time,id")
        sizes = Counter(teams.values())
        counts = {p: Counter() for p in teams if p.startswith("account.") and sizes[teams[p]] == 4}
        for r in positions:
            p = r["account_id"]
            if p in counts:
                counts[p]["positions"] += 1
                if r["vehicle_type"] and r["vehicle_type"] != "TransportAircraft":
                    counts[p]["vehicle_samples"] += 1
        for r in damages:
            p = r["victim_id"]
            a = r["attacker_id"]
            if p in counts and (r["damage"] or 0) > 0 and a in teams and teams[a] != teams[p]:
                counts[p]["enemy_damage_events"] += 1
        for r in equips:
            if r["account_id"] in counts and r["category"] == "Weapon":
                counts[r["account_id"]]["weapon_events"] += 1
        eligible = [
            p
            for p, c in counts.items()
            if c["vehicle_samples"] and c["enemy_damage_events"] and c["weapon_events"]
        ]
        assert eligible, (
            "First predefined match has no eligible observer; review selection explicitly"
        )
        eligible.sort(
            key=lambda p: (
                -counts[p]["vehicle_samples"],
                -counts[p]["enemy_damage_events"],
                hashlib.sha256(p.encode()).hexdigest(),
            )
        )
        ego = eligible[0]
        own = [ego] + sorted(p for p in teams if teams[p] == teams[ego] and p != ego)
        labels = {p: f"P{i + 1}" for i, p in enumerate(own)}
        labels.update({p: f"Other{i + 1}" for i, p in enumerate(sorted(set(teams) - set(own)))})
        data = {
            "positions": [r for r in positions if r["account_id"] in own],
            "damage": [r for r in damages if r["victim_id"] in own or r["attacker_id"] in own],
            "item_equip": [r for r in equips if r["account_id"] in own],
        }
        for name in ["item_pickup", "item_use", "parachute_landing"]:
            data[name] = q(
                f"SELECT * FROM pubg.telem_{name} WHERE match_id=%s AND account_id=ANY(%s) ORDER BY event_time,id",
                (mid, own),
            )
        for name in ["groggy", "kills"]:
            data[name] = q(
                f"SELECT * FROM pubg.telem_{name} WHERE match_id=%s ORDER BY event_time,id"
            )
            data[name] = [
                r
                for r in data[name]
                if r.get("victim_id") in own or r.get("attacker_id", r.get("killer_id")) in own
            ]
        data["game_states"] = q(
            "SELECT * FROM pubg.telem_game_states WHERE match_id=%s ORDER BY event_time,id"
        )
        data["match_start"] = q(
            "SELECT * FROM pubg.telem_match_start WHERE match_id=%s ORDER BY event_time,id"
        )
    conn.close()
    # Raw export is explicitly privileged server evidence, not observer-model input.
    for table, rows in data.items():
        for r in rows:
            for k in list(r):
                if k.endswith("_name") or k in ("player_name", "match_id"):
                    r.pop(k)
                elif k in (
                    "account_id",
                    "attacker_id",
                    "victim_id",
                    "killer_id",
                    "dbno_maker_id",
                    "finisher_id",
                    "lootbox_creator_id",
                ):
                    r[k] = labels.get(r[k], "Unknown" if r[k] else None)
                elif isinstance(r[k], datetime):
                    r[k] = r[k].isoformat()
                elif isinstance(r[k], Decimal):
                    r[k] = float(r[k])
            if table == "damage":
                a, v = r["attacker_id"], r["victim_id"]
                r["relation"] = (
                    "self"
                    if a == v and a
                    else "friendly"
                    if a in ("P1", "P2", "P3", "P4") and v in ("P1", "P2", "P3", "P4")
                    else "opponent"
                    if a and a.startswith("Other") and v in ("P1", "P2", "P3", "P4")
                    else "outgoing"
                    if a in ("P1", "P2", "P3", "P4") and v and v.startswith("Other")
                    else "environment_or_unknown"
                )
    # Query parameters contain private account IDs: publish only parameter shapes.
    for r in queries:
        r["parameters"] = [
            mid if isinstance(v, str) and v == mid else "<selected-team account IDs>"
            for v in r["parameters"]
        ]
    report = {
        "metadata": meta,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "selection": "First match in predefined deterministic 12-match training audit; human in four-person roster, weapon events, vehicle samples and received opponent damage required; maximize vehicle samples, then damage-event count; illustrative selection, not representative.",
        "eligible_humans": len(eligible),
        "observer": "P1",
        "team_members": ["P1", "P2", "P3", "P4"],
        "selection_counts": dict(counts[ego]),
        "patch": "40.2 date-window inferred, executable build unverified",
        "raw_scope": "Privileged server event evidence; NOT directly suitable as observer model input. Identifiers pseudonymized per match.",
        "tables": data,
        "queries": queries,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "source_events_server_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("tables", "queries")},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(json.dumps({k: len(v) for k, v in data.items()}))


if __name__ == "__main__":
    main()
