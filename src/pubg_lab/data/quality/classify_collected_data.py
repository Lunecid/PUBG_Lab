"""Read-only, match-level collection audit. Does not train or change source data."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import argparse
import csv
import getpass
import json
import os
import time
import hashlib
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

ROOT = REPO_ROOT
OUT = ROOT / "reports/corpus_quality_2026-10-20"


def connect(password):
    connection = psycopg2.connect(
        host=os.environ.get("PG_HOST", "localhost"),
        dbname=os.environ.get("PG_DBNAME", "pubg_survival"),
        user=os.environ.get("PG_USER", "postgres"),
        password=password,
        connect_timeout=8,
        options="-c default_transaction_read_only=on -c statement_timeout=300000",
    )
    connection.set_session(readonly=True, autocommit=True)
    return connection


def locked_ids():
    path = ROOT / "data/processed/heterograph_diffusion_v3/matches.csv"
    return {row["match_id"] for row in csv.DictReader(path.open()) if row["split"] == "test"}


# Parameterized queries only. No row-level player identity or coordinates are exported.
# Missing fields in a completed aggregate are NOT the same as a failed query.
QUERIES = {
    "structure": """
WITH p AS (
 SELECT p.match_id, count(*) AS participant_rows,
 count(DISTINCT NULLIF(p.player_id,'')) AS participant_accounts,
 count(*) FILTER (WHERE p.player_id IS NULL OR p.player_id='') AS participant_missing_id,
 count(*) FILTER (WHERE r.roster_id IS NULL) AS participant_orphan_roster,
 count(*) FILTER (WHERE r.match_id<>p.match_id) AS participant_cross_match_roster,
 count(*) FILTER (WHERE p.win_place IS NULL) AS participant_rank_missing,
 count(*) FILTER (WHERE p.win_place<1) AS participant_rank_invalid,
 count(*) FILTER (WHERE p.win_place IS NOT NULL AND r.rank IS NOT NULL AND p.win_place<>r.rank) AS participant_rank_disagree,
 count(*) FILTER (WHERE p.win_place IS NOT NULL AND r.rank IS NOT NULL AND p.win_place<>r.rank AND p.player_id LIKE 'ai.%%') AS bot_rank_disagree,
 count(*) FILTER (WHERE p.win_place IS NOT NULL AND r.rank IS NOT NULL AND p.win_place<>r.rank AND p.player_id LIKE 'account.%%') AS human_rank_disagree,
 count(*) FILTER (WHERE p.player_id LIKE 'ai.%%') AS bot_participants,
 count(*) FILTER (WHERE p.player_id LIKE 'account.%%') AS human_participants
 FROM pubg.participants p LEFT JOIN pubg.rosters r ON r.roster_id=p.roster_id
 WHERE p.match_id=ANY(%s) GROUP BY p.match_id
), r AS (
 SELECT match_id,count(*) AS roster_rows,count(DISTINCT team_id) AS distinct_team_ids,
 count(*) FILTER (WHERE team_id IS NULL) AS roster_team_missing,
 count(*) FILTER (WHERE rank IS NULL) AS roster_rank_missing,
 count(*) FILTER (WHERE rank<1) AS roster_rank_invalid,
 count(DISTINCT rank) AS distinct_ranks,max(rank) AS max_rank,
 count(*) FILTER (WHERE rank=1) AS rank1_rosters,
 count(*) FILTER (WHERE won) AS won_rosters,
 count(*) FILTER (WHERE rank IS NOT NULL AND won<>(rank=1)) AS won_rank_disagree
 FROM pubg.rosters WHERE match_id=ANY(%s) GROUP BY match_id
)
SELECT p.*,r.*,m.match_id,s.event_time AS start_time,e.event_time AS end_time,
 s.map_name AS start_map,s.team_size AS start_team_size,s.camera_view_type AS start_camera,
 s.is_event_mode,extract(epoch FROM (e.event_time-s.event_time)) AS event_duration_s
FROM pubg.matches m LEFT JOIN p ON p.match_id=m.match_id LEFT JOIN r ON r.match_id=m.match_id
LEFT JOIN pubg.telem_match_start s ON s.match_id=m.match_id
LEFT JOIN pubg.telem_match_end e ON e.match_id=m.match_id WHERE m.match_id=ANY(%s)
""",
    "positions": """
WITH raw AS (
 SELECT p.*, lag(event_time) OVER w AS prev_time,lag(pos_x) OVER w AS prev_x,
 lag(pos_y) OVER w AS prev_y,lag(pos_z) OVER w AS prev_z,lag(health) OVER w AS prev_hp,
 lag(vehicle_type) OVER w AS prev_vehicle_type, lag(vehicle_id) OVER w AS prev_vehicle_id,
 lag(is_game) OVER w AS prev_is_game
 FROM pubg.telem_positions p WHERE p.match_id=ANY(%s)
 WINDOW w AS (PARTITION BY match_id,account_id ORDER BY event_time,id)
), q AS (
 SELECT *,extract(epoch FROM (event_time-prev_time)) AS dt,
 (pos_x IS NOT NULL AND pos_y IS NOT NULL AND pos_z IS NOT NULL
 AND pos_x::text NOT IN ('NaN','Infinity','-Infinity')
 AND pos_y::text NOT IN ('NaN','Infinity','-Infinity')
 AND pos_z::text NOT IN ('NaN','Infinity','-Infinity')) AS finite_xyz
 FROM raw
)
SELECT match_id,count(*) AS position_rows,
 count(*) FILTER (WHERE is_game>0) AS ingame_position_rows,
 count(*) FILTER (WHERE is_game>0 AND finite_xyz AND health BETWEEN 0 AND 100
 AND NOT(pos_x=0 AND pos_y=0 AND pos_z=0) AND account_id IS NOT NULL AND account_id<>'') AS basic_valid_position_rows,
 count(*) FILTER (WHERE is_game IS NULL) AS position_phase_missing,
 count(*) FILTER (WHERE is_game>0 AND NOT finite_xyz) AS ingame_invalid_xyz,
 count(*) FILTER (WHERE is_game>0 AND (health IS NULL OR health::text IN ('NaN','Infinity','-Infinity'))) AS ingame_missing_health,
 count(*) FILTER (WHERE is_game>0 AND health::text NOT IN ('NaN','Infinity','-Infinity') AND (health<0 OR health>100)) AS ingame_invalid_health,
 count(*) FILTER (WHERE is_game>0 AND pos_x=0 AND pos_y=0 AND pos_z=0) AS ingame_zero_xyz,
 count(*) FILTER (WHERE is_game>0 AND (account_id IS NULL OR account_id='')) AS ingame_missing_actor,
 count(*) FILTER (WHERE is_game>0 AND finite_xyz AND (pos_x<0 OR pos_y<0 OR pos_x>1500000 OR pos_y>1500000)) AS ingame_broad_xy_outlier,
 count(*) FILTER (WHERE dt=0 AND (pos_x,pos_y,pos_z,health,vehicle_type,vehicle_id)
 IS NOT DISTINCT FROM (prev_x,prev_y,prev_z,prev_hp,prev_vehicle_type,prev_vehicle_id)) AS repeated_state_timestamp_rows,
 count(*) FILTER (WHERE dt=0 AND (pos_x,pos_y,pos_z,health,vehicle_type,vehicle_id)
 IS DISTINCT FROM (prev_x,prev_y,prev_z,prev_hp,prev_vehicle_type,prev_vehicle_id)) AS conflicting_state_timestamp_rows,
 count(*) FILTER (WHERE is_game>0 AND prev_is_game>0 AND dt>0) AS ingame_track_steps,
 count(*) FILTER (WHERE is_game>0 AND prev_is_game>0 AND dt>30) AS ingame_gap_over30s,
 count(*) FILTER (WHERE is_game>0 AND prev_is_game>0 AND dt BETWEEN 5 AND 30
 AND health>0 AND prev_hp>0 AND finite_xyz
 AND sqrt(power((pos_x-prev_x)::float8,2)+power((pos_y-prev_y)::float8,2))/100/NULLIF(dt,0)>200) AS fast_transition_over200mps,
 min(event_time) AS position_first_time,max(event_time) AS position_last_time,
 min(event_time) FILTER (WHERE is_game>0) AS ingame_first_time,
 max(event_time) FILTER (WHERE is_game>0) AS ingame_last_time,
 max(pos_x) FILTER (WHERE is_game>0 AND finite_xyz) AS ingame_max_x_cm,
 max(pos_y) FILTER (WHERE is_game>0 AND finite_xyz) AS ingame_max_y_cm
FROM q GROUP BY match_id
""",
    "position_actors": """
WITH pa AS (
 SELECT match_id,account_id,count(*) AS n,
 count(*) FILTER (WHERE is_game>0 AND health>0) AS positive_ingame
 FROM pubg.telem_positions WHERE match_id=ANY(%s) GROUP BY match_id,account_id
), pp AS (
 SELECT DISTINCT match_id,player_id FROM pubg.participants WHERE match_id=ANY(%s)
)
SELECT pa.match_id,count(*) AS position_actor_count,
 count(*) FILTER (WHERE pp.player_id IS NOT NULL) AS roster_actors_with_positions,
 count(*) FILTER (WHERE pp.player_id IS NULL) AS nonroster_position_actors,
 coalesce(sum(pa.n) FILTER (WHERE pp.player_id IS NULL),0) AS nonroster_position_rows,
 coalesce(sum(pa.n) FILTER (WHERE pp.player_id IS NULL AND pa.account_id LIKE 'Monster.%%'),0) AS monster_position_rows,
 count(*) FILTER (WHERE pp.player_id IS NOT NULL AND pa.positive_ingame>0) AS roster_actors_positive_ingame
FROM pa LEFT JOIN pp ON pp.match_id=pa.match_id AND pp.player_id=pa.account_id GROUP BY pa.match_id
""",
    "game_clock": """
SELECT g.match_id,count(*) AS game_rows,
 count(*) FILTER (WHERE is_game>0) AS ingame_game_rows,
 count(*) FILTER (WHERE is_game>0 AND safe_zone_radius>0 AND poison_zone_radius>0) AS positive_zone_pair_rows,
 count(*) FILTER (WHERE safe_zone_radius<0 OR poison_zone_radius<0) AS negative_zone_radius_rows,
 count(*) FILTER (WHERE num_alive_players<0 OR num_alive_teams<0) AS negative_alive_rows,
 count(*) FILTER (WHERE elapsed_time IS NULL) AS game_elapsed_missing,
 percentile_cont(0.5) WITHIN GROUP(ORDER BY extract(epoch FROM (g.event_time-s.event_time))-g.elapsed_time)
 FILTER (WHERE g.is_game>0 AND g.elapsed_time IS NOT NULL AND s.event_time IS NOT NULL) AS game_clock_offset_median_s,
 min(extract(epoch FROM (g.event_time-s.event_time))-g.elapsed_time)
 FILTER (WHERE g.is_game>0 AND g.elapsed_time IS NOT NULL AND s.event_time IS NOT NULL) AS game_clock_offset_min_s,
 max(extract(epoch FROM (g.event_time-s.event_time))-g.elapsed_time)
 FILTER (WHERE g.is_game>0 AND g.elapsed_time IS NOT NULL AND s.event_time IS NOT NULL) AS game_clock_offset_max_s
FROM pubg.telem_game_states g LEFT JOIN pubg.telem_match_start s ON s.match_id=g.match_id
WHERE g.match_id=ANY(%s) GROUP BY g.match_id
""",
    "kills": """
WITH k AS (
 SELECT match_id,victim_id,event_time,count(*) AS n
 FROM pubg.telem_kills WHERE match_id=ANY(%s) GROUP BY match_id,victim_id,event_time
), a AS (
 SELECT match_id,victim_id,count(*) AS distinct_death_times,sum(n) AS raw_rows
 FROM k GROUP BY match_id,victim_id
), p AS (SELECT DISTINCT match_id,player_id FROM pubg.participants WHERE match_id=ANY(%s))
SELECT a.match_id,sum(raw_rows) AS kill_rows,sum(raw_rows-distinct_death_times) AS repeated_victim_death_timestamp_rows,
 count(*) FILTER (WHERE distinct_death_times>1) AS actors_with_multiple_death_times,
 coalesce(sum(raw_rows) FILTER (WHERE p.player_id IS NULL),0) AS nonroster_victim_kill_rows
FROM a LEFT JOIN p ON p.match_id=a.match_id AND p.player_id=a.victim_id GROUP BY a.match_id
""",
    "damage": """
SELECT match_id,count(*) AS damage_rows,
 count(*) FILTER (WHERE damage IS NULL OR damage::text IN ('NaN','Infinity','-Infinity')) AS invalid_damage_value_rows,
 count(*) FILTER (WHERE damage<0) AS negative_damage_rows,
 count(*) FILTER (WHERE victim_id IS NULL OR victim_id='') AS missing_damage_victim_rows,
 count(*) FILTER (WHERE attacker_id IS NULL OR attacker_id='') AS missing_damage_attacker_rows,
 count(*) FILTER (WHERE damage_type IS NULL OR damage_type='') AS missing_damage_type_rows
FROM pubg.telem_damage WHERE match_id=ANY(%s) GROUP BY match_id
""",
}

RANK_DIAGNOSTIC_SQL = """
WITH deaths AS (
 SELECT match_id,victim_id,count(DISTINCT event_time) AS death_times
 FROM pubg.telem_kills WHERE NOT(match_id=ANY(%s)) GROUP BY match_id,victim_id
)
SELECT m.match_type,m.game_mode,
 CASE WHEN p.player_id LIKE 'account.%%' THEN 'human'
      WHEN p.player_id LIKE 'ai.%%' THEN 'bot' ELSE 'other' END AS actor_kind,
 CASE WHEN coalesce(d.death_times,0)>1 THEN 'multiple_death_timestamps'
      WHEN d.death_times=1 THEN 'one_death_timestamp' ELSE 'no_death_record' END AS death_group,
 count(*) AS participant_rows,
 count(*) FILTER (WHERE p.win_place IS NOT NULL AND r.rank IS NOT NULL) AS comparable_rows,
 count(*) FILTER (WHERE p.win_place<>r.rank) AS rank_disagree,
 count(*) FILTER (WHERE p.win_place<r.rank) AS personal_rank_better_than_team,
 count(*) FILTER (WHERE p.win_place>r.rank) AS personal_rank_worse_than_team,
 count(*) FILTER (WHERE p.win_place<>r.rank AND r.rank=1) AS disagreement_in_winning_team
FROM pubg.participants p JOIN pubg.rosters r ON r.roster_id=p.roster_id AND r.match_id=p.match_id
JOIN pubg.matches m ON m.match_id=p.match_id
LEFT JOIN deaths d ON d.match_id=p.match_id AND d.victim_id=p.player_id
WHERE NOT(p.match_id=ANY(%s))
GROUP BY 1,2,3,4 ORDER BY 1,2,3,4
"""


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    tmp.replace(path)


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def audit_chunk(index, ids, password, resume):
    folder = OUT / "chunks"
    folder.mkdir(exist_ok=True)
    output = {}
    for name, sql in QUERIES.items():
        path = folder / f"{index:04d}_{name}.json"
        if resume and path.exists():
            cached = json.loads(path.read_text())
            if (
                cached.get("match_ids") == ids
                and cached.get("query_sha256") == hashlib.sha256(sql.encode()).hexdigest()
                and cached.get("ok")
            ):
                output[name] = cached
                continue
        started = time.monotonic()
        try:
            with connect(password) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, tuple(ids for _ in range(sql.count("%s"))))
                rows = [dict(row) for row in cur.fetchall()]
            result = {"ok": True, "rows": rows}
        except psycopg2.Error as exc:
            # Server error details can contain data; keep only SQLSTATE and error class.
            result = {
                "ok": False,
                "rows": [],
                "error_class": type(exc).__name__,
                "sqlstate": exc.pgcode,
            }
        result.update(
            match_ids=ids,
            query_sha256=hashlib.sha256(sql.encode()).hexdigest(),
            elapsed_s=round(time.monotonic() - started, 3),
        )
        save_json(path, result)
        output[name] = result
    return index, output


def classify(inventory, results, protected):
    by_match = {r["match_id"]: dict(r) for r in inventory}
    for mid, row in by_match.items():
        row["locked_not_inspected"] = mid in protected
        row["query_failures"] = []
    for chunk in results:
        for name, result in chunk.items():
            assert not protected.intersection(result["match_ids"]), (
                "Protected match reached detail audit"
            )
            for mid in result["match_ids"]:
                if not result["ok"]:
                    by_match[mid]["query_failures"].append(name)
                else:
                    by_match[mid][f"{name}_checked"] = True
            for item in result["rows"]:
                # SELECT p.*,r.* has duplicate match_id columns; guard missing values.
                mid = item.get("match_id")
                if mid:
                    by_match[mid].update(item)
    classified = []
    for mid, row in by_match.items():
        n = lambda key: float(row.get(key) or 0)
        flags, critical, repair, unknown, notes = [], [], [], [], []

        def flag(code, bucket):
            flags.append(code)
            bucket.append(code)

        collected = bool(row.get("telemetry_fetched"))
        br = row["map_name"] not in ("Range_Main", "Tutorial_Main") and row["game_mode"] in (
            "solo",
            "duo",
            "squad",
            "solo-fpp",
            "duo-fpp",
            "squad-fpp",
        )
        row["br_rule_candidate"] = br
        if mid in protected:
            status = "protected_not_inspected"
            trajectory = rank = combat = "not_inspected"
        elif row["query_failures"]:
            flag("query_incomplete", unknown)
            status = "insufficient_evidence"
            trajectory = rank = combat = "not_verified_incomplete_query"
        else:
            if n("position_rows") == 0:
                flag("no_position_records", critical if collected else unknown)
            if n("participant_rows") == 0:
                flag("no_participant_records", critical if br else notes)
            if n("roster_rows") == 0:
                flag("no_roster_records", critical if br else notes)
            if not row.get("start_time"):
                flag("missing_match_start", unknown)
            if not row.get("end_time"):
                flag("missing_match_end", unknown)
            if n("event_duration_s") < 0:
                flag("end_before_start", critical)
            if row.get("start_map") and row["start_map"] != row["map_name"]:
                flag("match_start_map_disagree", repair)
            if (
                n("participant_missing_id")
                or n("participant_orphan_roster")
                or n("participant_cross_match_roster")
            ):
                flag("broken_participant_roster_join", critical)
            if n("participant_rows") - n("participant_missing_id") > n("participant_accounts"):
                flag("duplicate_participant_account", repair)
            if n("roster_team_missing") or n("roster_rows") - n("roster_team_missing") > n(
                "distinct_team_ids"
            ):
                flag("ambiguous_roster_team_key", repair)
            rank_issues = []
            if br:
                for key in (
                    "roster_rank_missing",
                    "roster_rank_invalid",
                    "participant_rank_invalid",
                    "participant_rank_disagree",
                    "won_rank_disagree",
                ):
                    if n(key):
                        rank_issues.append(key)
                if n("rank1_rosters") != 1 or n("won_rosters") != 1:
                    rank_issues.append("winner_not_unique_or_missing")
                if n("max_rank") > n("roster_rows"):
                    rank_issues.append("rank_exceeds_roster_count")
                if n("distinct_ranks") < n("roster_rows") - n("roster_rank_missing"):
                    rank_issues.append("rank_ties_or_duplicate_rank")
                if rank_issues:
                    flag("outcome_needs_reconciliation", repair)
            invalid_pos = sum(
                n(k)
                for k in (
                    "ingame_invalid_xyz",
                    "ingame_missing_health",
                    "ingame_invalid_health",
                    "ingame_zero_xyz",
                    "ingame_missing_actor",
                )
            )
            if invalid_pos:
                flag("invalid_or_missing_ingame_position_fields", repair)
            if n("position_rows") and not n("ingame_position_rows"):
                flag("no_positive_phase_positions", unknown if br else notes)
            if br and n("ingame_position_rows") and not n("basic_valid_position_rows"):
                flag("no_basic_valid_ingame_positions", critical)
            for key in (
                "repeated_state_timestamp_rows",
                "conflicting_state_timestamp_rows",
                "ingame_broad_xy_outlier",
                "nonroster_position_rows",
                "repeated_victim_death_timestamp_rows",
                "invalid_damage_value_rows",
                "negative_damage_rows",
                "negative_zone_radius_rows",
                "negative_alive_rows",
            ):
                if n(key):
                    flag(key, repair)
            if n("roster_actors_with_positions") < n("participant_accounts"):
                flag("participants_without_position_records", repair)
            if n("fast_transition_over200mps"):
                flag("mobility_discontinuity_needs_episode_review", repair)
            if n("ingame_gap_over30s"):
                flag("track_gaps_need_window_mask", repair)
            if abs(n("game_clock_offset_median_s")) > 2:
                flag("elapsed_and_event_clock_need_alignment", repair)
            if n("game_clock_offset_max_s") - n("game_clock_offset_min_s") > 5:
                flag("clock_offset_not_constant", repair)
            if br and not n("positive_zone_pair_rows"):
                flag("no_positive_zone_pair", unknown)
            if not collected and n("position_rows"):
                flag("collection_flag_record_disagree", repair)
            if n("actors_with_multiple_death_times"):
                flag("multiple_deaths_possible_reentry_not_duplicate", notes)
            if n("missing_damage_attacker_rows"):
                flag("missing_attacker_can_be_environment_damage", notes)
            if not br:
                flag("non_BR_rules_task_masks_required", notes)
            if row["query_failures"]:
                status = "insufficient_evidence"
            elif not collected and not n("position_rows"):
                status = "uncollected_or_no_telemetry"
            elif critical:
                status = "quarantine_for_core_tasks"
            elif unknown:
                status = "insufficient_evidence"
            elif repair:
                status = "usable_after_correction_or_masking"
            else:
                status = "no_issue_detected_in_checked_rules"
            trajectory = (
                "unavailable"
                if not n("position_rows")
                else "blocked_join_or_clock"
                if critical
                else "needs_review"
                if unknown
                else "needs_correction_or_masking"
                if repair
                else "basic_checks_passed"
            )
            rank = (
                "not_applicable"
                if not br
                else "needs_reconciliation"
                if rank_issues
                else "not_verified"
                if not row.get("end_time") or row["query_failures"]
                else "source_fields_consistent"
            )
            combat = (
                "not_checked"
                if not row.get("damage_checked")
                else "no_events_recorded"
                if not n("damage_rows")
                else "mask_invalid_events"
                if n("negative_damage_rows") + n("invalid_damage_value_rows")
                else "basic_value_checks_passed_actor_semantics_not_exhaustive"
            )
            row["rank_issue_codes"] = "|".join(rank_issues)
        row.update(
            classification=status,
            trajectory_status=trajectory,
            rank_status=rank,
            combat_status=combat,
            issue_codes="|".join(flags),
            critical_codes="|".join(critical),
            repair_codes="|".join(repair),
            uncertainty_codes="|".join(unknown),
            context_notes="|".join(notes),
            query_failures="|".join(row["query_failures"]),
        )
        # Separate source anomaly evidence from ordinary model-readiness work.
        if status == "protected_not_inspected":
            anomaly = "protected_not_inspected"
        elif status == "uncollected_or_no_telemetry":
            anomaly = "unavailable"
        elif row["query_failures"]:
            anomaly = "unknown_incomplete_audit"
        elif critical:
            anomaly = "core_record_problem"
        elif (
            any(
                n(key)
                for key in (
                    "participant_rank_disagree",
                    "won_rank_disagree",
                    "participant_cross_match_roster",
                )
            )
            or "match_start_map_disagree" in flags
        ):
            anomaly = "source_field_conflict"
        elif any(
            n(key)
            for key in (
                "negative_damage_rows",
                "invalid_damage_value_rows",
                "negative_zone_radius_rows",
                "negative_alive_rows",
                "conflicting_state_timestamp_rows",
            )
        ):
            anomaly = "invalid_or_conflicting_record"
        elif unknown or any(
            code in flags
            for code in (
                "invalid_or_missing_ingame_position_fields",
                "ingame_broad_xy_outlier",
                "mobility_discontinuity_needs_episode_review",
                "track_gaps_need_window_mask",
                "outcome_needs_reconciliation",
                "participants_without_position_records",
                "nonroster_position_rows",
                "duplicate_participant_account",
                "ambiguous_roster_team_key",
            )
        ):
            anomaly = "context_or_completeness_review"
        else:
            anomaly = "no_anomaly_detected_in_checked_rules"
        row["anomaly_class"] = anomaly
        classified.append(row)
    return classified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--rank-diagnostic", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    password = (
        os.environ.get("PG_PASSWORD")
        or os.environ.get("PGPASSWORD")
        or getpass.getpass("DB password (not stored): ")
    )
    with connect(password) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SHOW transaction_read_only")
            assert cur.fetchone()["transaction_read_only"] == "on"
            cur.execute(
                "SELECT count(*) AS n, min(created_at) AS first_created, max(created_at) AS last_created FROM pubg.matches"
            )
            print(json.dumps(dict(cur.fetchone()), default=str), flush=True)
            cur.execute("""SELECT table_name, column_name, data_type FROM information_schema.columns
                WHERE table_schema='pubg' AND table_name IN
                ('matches','participants','rosters','telem_positions','telem_game_states',
                 'telem_match_start','telem_match_end','telem_kills','telem_damage','collection_log')
                ORDER BY table_name,ordinal_position""")
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "live_schema.json").write_text(
                json.dumps([dict(r) for r in cur.fetchall()], indent=2)
            )
            print(
                "Read-only connection and schema verified; protected matches:",
                len(locked_ids()),
                flush=True,
            )
            if args.probe:
                return
            if args.rank_diagnostic:
                blocked = sorted(locked_ids())
                cur.execute(RANK_DIAGNOSTIC_SQL, (blocked, blocked))
                rows = [dict(r) for r in cur.fetchall()]
                save_json(
                    OUT / "rank_lifecycle_diagnostic.json",
                    {
                        "sql": RANK_DIAGNOSTIC_SQL,
                        "protected_ids_excluded": len(blocked),
                        "rows": rows,
                        "note": "Multiple death timestamps are an episode-reentry proxy, not a verified cause.",
                    },
                )
                write_csv(OUT / "rank_lifecycle_diagnostic.csv", rows)
                print(json.dumps(rows, default=str, ensure_ascii=False, indent=2), flush=True)
                return
            cur.execute("""SELECT m.match_id,m.created_at,m.duration,m.game_mode,m.map_name,m.match_type,m.shard_id,
                m.is_custom,c.match_fetched,c.telemetry_fetched,(c.error_message IS NOT NULL) AS collection_has_error,
                c.retry_count FROM pubg.matches m LEFT JOIN pubg.collection_log c ON c.match_id=m.match_id
                ORDER BY m.match_id""")
            inventory = [dict(r) for r in cur.fetchall()]
            if args.explain:
                selected = [r["match_id"] for r in inventory if r["match_id"] not in locked_ids()][
                    :250
                ]
                cur.execute("EXPLAIN (FORMAT JSON) " + QUERIES["positions"], (selected,))
                plan = cur.fetchone()["QUERY PLAN"]
                save_json(OUT / "positions_query_plan.json", plan)
                print(json.dumps(plan, indent=2, default=str)[:14000], flush=True)
                return
    protected = locked_ids()
    save_json(OUT / "inventory.json", inventory)
    save_json(OUT / "queries.json", QUERIES)
    ids = [row["match_id"] for row in inventory if row["match_id"] not in protected]
    if args.limit:
        ids = ids[: args.limit]
        inventory = [r for r in inventory if r["match_id"] in set(ids) | protected]
    chunks = [ids[i : i + args.chunk_size] for i in range(0, len(ids), args.chunk_size)]
    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(audit_chunk, i, part, password, args.resume): i
            for i, part in enumerate(chunks)
        }
        for future in as_completed(futures):
            index, result = future.result()
            completed.append(result)
            failures = [name for name, r in result.items() if not r["ok"]]
            print(
                json.dumps(
                    {
                        "completed_chunks": len(completed),
                        "total_chunks": len(chunks),
                        "chunk": index,
                        "failures": failures,
                        "elapsed_s": round(time.monotonic() - started),
                    }
                ),
                flush=True,
            )
    del password
    rows = classify(inventory, completed, protected)
    write_csv(OUT / "match_quality.csv", rows)
    counts = Counter(r["classification"] for r in rows)
    reasons = Counter(code for r in rows for code in r["issue_codes"].split("|") if code)
    grouped = Counter(
        (r["map_name"], r["game_mode"], r["match_type"], r["classification"]) for r in rows
    )
    write_csv(
        OUT / "quality_by_domain.csv",
        [
            dict(map_name=a, game_mode=b, match_type=c, classification=d, matches=n)
            for (a, b, c, d), n in sorted(
                grouped.items(), key=lambda x: tuple(str(v) for v in x[0])
            )
        ],
    )
    summary = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "unit": "match",
        "inventory_matches": len(inventory),
        "detailed_audit_requested_matches": len(ids),
        "locked_matches_not_opened": len(protected),
        "classification_counts": dict(counts),
        "reason_match_counts": dict(reasons),
        "rank_status_counts": dict(Counter(r["rank_status"] for r in rows)),
        "query_failures": sum(not v["ok"] for c in completed for v in c.values()),
        "elapsed_s": round(time.monotonic() - started, 1),
        "training_started": False,
        "database_read_only": True,
        "source_modified": False,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": [
            "Normal means no issue detected in these checks, not fully validated terrain or semantics.",
            "Motion discontinuity is not proof of corruption; vehicles, transport and reentry require episode review.",
            "Position value checks distinguish positive-phase records; initial aircraft and pregame require separate mode interpretation.",
            "Rank agreement is source consistency, not independent verification of final elimination order.",
            "Map-specific collision and exact coordinate bounds are not verified; broad XY screening only.",
            "Repeated same-state timestamp counts are candidate duplicate states, not necessarily byte-identical events.",
            "Database may be updated during chunked reads; capture interval is recorded, not a frozen exported DB snapshot.",
        ],
    }
    save_json(OUT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
