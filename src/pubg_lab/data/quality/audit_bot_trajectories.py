"""Read-only, stratified feasibility probe; password comes from env or a hidden prompt.

Run from the repository root: .venv/bin/python tools/audit_bot_trajectories.py
This is a purposive 12-match probe, not a population estimate or training dataset.
"""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import csv
import getpass
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import psycopg2

ROOT = REPO_ROOT
OUT = ROOT / "reports/bot_trajectory_audit_2026-10-15"


def kind(account):
    if isinstance(account, str) and account.startswith("ai."):
        return "bot"
    if isinstance(account, str) and account.startswith("account."):
        return "human"
    return "other"


def quantiles(values):
    ordered = sorted(values)
    out = []
    for p in [0, 0.1, 0.5, 0.9, 1]:
        index = p * (len(ordered) - 1)
        lo = int(index)
        hi = min(lo + 1, len(ordered) - 1)
        out.append(ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo))
    return out


def main():
    with (ROOT / "reports/bigcomp2027_data_profile/csv/sample_match_level.csv").open() as f:
        population = list(csv.DictReader(f))
    strata = [
        ("erangel_tpp_squad", "official"),
        ("erangel_tpp_squad", "airoyale"),
        ("erangel_tpp_squad", "competitive"),
        ("taego_tpp_squad", "official"),
        ("sanhok_tpp_squad", "official"),
        ("vikendi_tpp_squad", "official"),
    ]
    selected = []
    for cohort, match_type in strata:
        candidates = sorted(
            (r for r in population if r["cohort"] == cohort and r["match_type"] == match_type),
            key=lambda r: r["match_id"],
        )
        selected.extend(candidates[:2])
    password = (
        os.environ.get("PG_PASSWORD")
        or os.environ.get("PGPASSWORD")
        or getpass.getpass("DB password: ")
    )
    conn = psycopg2.connect(
        host=os.environ.get("PG_HOST", "localhost"),
        port=os.environ.get("PG_PORT", "5432"),
        dbname=os.environ.get("PG_DBNAME", "pubg_survival"),
        user=os.environ.get("PG_USER", "postgres"),
        password=password,
        connect_timeout=5,
        options="-c default_transaction_read_only=on -c statement_timeout=30000",
    )
    del password
    OUT.mkdir(parents=True, exist_ok=True)
    summaries, sequences, examples = [], [], []
    intervals = defaultdict(list)
    with conn, conn.cursor() as cur:
        cur.execute("SHOW transaction_read_only")
        assert cur.fetchone()[0] == "on"
        for meta in selected:
            mid = meta["match_id"]
            cur.execute(
                "SELECT p.player_id, r.team_id FROM pubg.participants p LEFT JOIN pubg.rosters r ON r.match_id=p.match_id AND r.roster_id=p.roster_id WHERE p.match_id=%s",
                (mid,),
            )
            parts = dict(cur.fetchall())
            counts = Counter(kind(p) for p in parts)
            cur.execute(
                """SELECT account_id, elapsed_time, pos_x::float8, pos_y::float8,
                        pos_z::float8, health::float8, team_id, event_time, vehicle_type, id
                        FROM pubg.telem_positions WHERE match_id=%s AND is_game >= 0.1
                        ORDER BY account_id, elapsed_time, event_time, id""",
                (mid,),
            )
            raw = cur.fetchall()
            tracks = defaultdict(list)
            invalid = outsiders = mismatch = 0
            outsider_types = Counter()
            for row in raw:
                acc, t, x, y, z, hp, team, event_time, vehicle, rid = row
                if acc not in parts:
                    outsiders += 1
                    outsider_types[(acc or "missing").split(".")[0]] += 1
                    continue
                if t is None or x is None or y is None:
                    invalid += 1
                    continue
                mismatch += team != parts[acc]
                tracks[acc].append(row)
            summary = {
                "match_id": mid,
                "cohort": meta["cohort"],
                "match_type": meta["match_type"],
                "participants": len(parts),
                "humans": counts["human"],
                "bots": counts["bot"],
                "other_participants": counts["other"],
                "raw_ingame_position_rows": len(raw),
                "nonparticipant_position_rows": outsiders,
                "nonparticipant_prefixes": dict(outsider_types),
                "invalid_participant_position_rows": invalid,
                "team_mismatch_rows": mismatch,
                "participants_without_positions": len(set(parts) - set(tracks)),
                "bot_count_agrees_with_saved_profile": counts["bot"] == int(meta["n_ai"]),
            }
            longest = {}
            for acc, rows in tracks.items():
                cls = kind(acc)
                times = [r[1] for r in rows]
                intervals[cls].extend(b - a for a, b in zip(times, times[1:]))
                # Keep the last event at each integer elapsed second; do not interpolate gaps.
                dedup = list({r[1]: r for r in rows}.values())
                run = best = 1
                for prev, row in zip(dedup, dedup[1:]):
                    run = run + 1 if 0 < row[1] - prev[1] <= 11 else 1
                    best = max(best, run)
                alias = hashlib.sha256((mid + acc).encode()).hexdigest()[:16]
                sequences.append(
                    {
                        "match_id": mid,
                        "class": cls,
                        "trajectory_id": alias,
                        "raw_points": len(rows),
                        "unique_seconds": len(dedup),
                        "span_seconds": int(times[-1] - times[0]),
                        "longest_run_points_gap_le11": best,
                        "vehicle_nonempty_points": sum(bool(r[8]) for r in dedup),
                    }
                )
                if cls not in longest or len(dedup) > len(longest[cls][1]):
                    longest[cls] = (alias, dedup)
            for cls, (alias, rows) in longest.items():
                for acc, t, x, y, z, hp, team, event_time, vehicle, rid in rows:
                    examples.append(
                        {
                            "match_id": mid,
                            "class": cls,
                            "trajectory_id": alias,
                            "elapsed_time_s": t,
                            "event_time": event_time.isoformat(),
                            "x_m": x / 100,
                            "y_m": y / 100,
                            "z_m": None if z is None else z / 100,
                            "health": hp,
                            "team_id": team,
                            "vehicle_type": vehicle,
                        }
                    )
            for table, cols in [
                ("telem_damage", ["attacker_id", "victim_id"]),
                ("telem_kills", ["killer_id", "victim_id"]),
                ("telem_item_use", ["account_id"]),
            ]:
                # Table and columns come only from the fixed literals above.
                cur.execute(f"SELECT {','.join(cols)} FROM pubg.{table} WHERE match_id=%s", (mid,))
                events = cur.fetchall()
                bot_ids = {a for row in events for a in row if kind(a) == "bot"}
                summary[table] = {
                    "rows_with_bot": sum(any(kind(a) == "bot" for a in row) for row in events),
                    "distinct_bot_ids": len(bot_ids),
                    "bot_ids_missing_participant": len(bot_ids - set(parts)),
                    "bot_ids_missing_positions": len(bot_ids - set(tracks)),
                }
            summaries.append(summary)
            print(
                f"Checked {meta['cohort']} {meta['match_type']}: {counts['human']} human / {counts['bot']} bot",
                flush=True,
            )
    conn.close()
    stats = {}
    for cls, deltas in intervals.items():
        seq = [r for r in sequences if r["class"] == cls]
        stats[cls] = {
            "trajectories": len(seq),
            "raw_points": sum(r["raw_points"] for r in seq),
            "interval_count": len(deltas),
            "interval_quantiles_p0_p10_p50_p90_p100": quantiles(deltas) if deltas else [],
            "duplicate_second_intervals": sum(d == 0 for d in deltas),
            "gap_over_20s_intervals": sum(d > 20 for d in deltas),
            "tracks_with_10_contiguous_points_gap_le11": sum(
                r["longest_run_points_gap_le11"] >= 10 for r in seq
            ),
        }
    result = {
        "audit_date": "2026-10-15",
        "scope": "12 purposively selected matches, two per stratum; no population inference",
        "classification": "ai. = bot proxy; account. = human proxy; other retained separately",
        "filter": "is_game >= 0.1, participant membership, valid t/x/y; no death/landing/vehicle filtering",
        "matches": summaries,
        "by_class": stats,
    }
    (OUT / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    for name, rows in [
        ("trajectory_summary.csv", sequences),
        ("example_trajectories.csv", examples),
    ]:
        with (OUT / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
