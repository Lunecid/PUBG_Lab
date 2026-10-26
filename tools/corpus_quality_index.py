"""Persistent, local quality index. Never connects to or changes the source DB."""
import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports/corpus_quality_2026-10-20"
DEFAULT_DB = DEFAULT_REPORT / "quality_index.sqlite"

# Each rule retains its actual aggregate evidence, not just a binary flag.
# Severity is the potential effect on the affected task, not certainty of corruption.
RULES = {
    "outcome_needs_reconciliation": ("structure", "context_candidate", "high",
        "결과 라벨 대조 필요", "순위 목표를 보류하고 로스터·개인 순위·최종 탈락을 대조",
        "rank_issue_codes participant_rank_disagree won_rank_disagree roster_rank_missing roster_rank_invalid participant_rank_invalid rank1_rosters won_rosters max_rank distinct_ranks roster_rows"),
    "participant_rank_disagree": ("structure", "source_conflict", "high",
        "개인·팀 순위 필드 불일치 확인", "어느 필드가 정확한지는 원본 결과와 대조",
        "participant_rank_disagree human_rank_disagree bot_rank_disagree participant_rows"),
    "won_rank_disagree": ("structure", "source_conflict", "high",
        "우승·팀 순위 필드 불일치 확인", "원본 결과 대조 후 결과 목표 재검토",
        "won_rank_disagree won_rosters rank1_rosters"),
    "match_start_map_disagree": ("structure", "source_conflict", "high",
        "시작 이벤트·메타데이터 맵 불일치", "공간 입력 구성 전 맵 식별 확인",
        "start_map map_name"),
    "broken_participant_roster_join": ("structure", "structural_problem", "high",
        "참가자·로스터 연결 문제", "팀 관계 감독 과제를 보류하고 조인 복원",
        "participant_missing_id participant_orphan_roster participant_cross_match_roster"),
    "duplicate_participant_account": ("structure", "context_candidate", "high",
        "참가자 계정 키 중복 후보", "중복 레코드와 실제 상태를 대조",
        "participant_rows participant_accounts participant_missing_id"),
    "ambiguous_roster_team_key": ("structure", "context_candidate", "high",
        "팀 키 결측·중복 후보", "팀 노드 생성 전 키 확인",
        "roster_rows distinct_team_ids roster_team_missing"),
    "end_before_start": ("structure", "structural_problem", "high",
        "종료가 시작보다 빠름", "시간열 감독 보류",
        "start_time end_time event_duration_s"),
    "no_position_records": ("positions", "missing_data", "high",
        "위치 기록 없음", "경로 목표 제외; 수집 상태 확인",
        "telemetry_fetched positions_checked position_rows"),
    "no_participant_records": ("structure", "missing_data", "high",
        "참가자 기록 없음", "BR 참가자 감독 보류; 비-BR 정상 예외 구분",
        "structure_checked participant_rows br_rule_candidate map_name game_mode"),
    "no_roster_records": ("structure", "missing_data", "high",
        "로스터 기록 없음", "BR 팀 감독 보류; 비-BR 정상 예외 구분",
        "structure_checked roster_rows br_rule_candidate map_name game_mode"),
    "missing_match_start": ("structure", "missing_data", "high",
        "시작 사건 없음", "기준 시계와 수집 완전성 확인",
        "structure_checked start_time telemetry_fetched"),
    "missing_match_end": ("structure", "missing_data", "high",
        "종료 사건 없음", "종료 결과 목표 보류; 유효 prefix는 별도 검토",
        "structure_checked end_time telemetry_fetched"),
    "invalid_or_missing_ingame_position_fields": ("positions", "context_candidate", "high",
        "위치·체력·ID 기초 필드 문제", "문제 행과 구간 분리 후 마스크",
        "ingame_invalid_xyz ingame_missing_health ingame_invalid_health ingame_zero_xyz ingame_missing_actor"),
    "no_basic_valid_ingame_positions": ("positions", "structural_problem", "high",
        "양성 페이즈 기초 유효 위치 없음", "해당 경로 목표 보류",
        "ingame_position_rows basic_valid_position_rows"),
    "no_positive_phase_positions": ("positions", "context_candidate", "medium",
        "양성 페이즈 위치 없음", "경기 규칙·페이즈 의미 확인",
        "position_rows ingame_position_rows map_name game_mode"),
    "ingame_broad_xy_outlier": ("positions", "context_candidate", "medium",
        "넓은 XY 범위 밖 후보", "항공·맵 경계 맥락 검토; 자동 clipping 금지",
        "ingame_broad_xy_outlier ingame_max_x_cm ingame_max_y_cm"),
    "repeated_state_timestamp_rows": ("positions", "context_candidate", "medium",
        "동시각 동일 상태 반복 후보", "바이트 단위 중복인지 별도 대조",
        "repeated_state_timestamp_rows position_rows"),
    "conflicting_state_timestamp_rows": ("positions", "conflicting_record", "high",
        "동시각 상태 상충 확인", "상충 상태의 순서·원본 사건 확인",
        "conflicting_state_timestamp_rows position_rows"),
    "nonroster_position_rows": ("position_actors", "context_candidate", "medium",
        "참가자 명단 밖 위치 객체", "NPC·명단 손실 구분; player 노드 자동 편입 금지",
        "nonroster_position_rows nonroster_position_actors monster_position_rows position_actor_count"),
    "participants_without_position_records": ("position_actors", "missing_data", "medium",
        "위치 기록 없는 참가자", "해당 개인의 경로 목표 마스크",
        "participant_accounts roster_actors_with_positions"),
    "mobility_discontinuity_needs_episode_review": ("positions", "context_candidate", "medium",
        "고속 전이 후보", "재진입·수송·항공·공백 episode 대조",
        "fast_transition_over200mps ingame_track_steps"),
    "track_gaps_need_window_mask": ("positions", "context_candidate", "medium",
        "30초 초과 관측 간격", "공백을 가로지르는 경로 window 마스크",
        "ingame_gap_over30s ingame_track_steps"),
    "elapsed_and_event_clock_need_alignment": ("game_clock", "preprocessing_requirement", "medium",
        "시계 원점 정렬 필요", "경과 시계와 사건 시계의 관계 검증 후 정렬",
        "game_clock_offset_median_s game_clock_offset_min_s game_clock_offset_max_s"),
    "clock_offset_not_constant": ("game_clock", "context_candidate", "medium",
        "경기 중 시계 오프셋 변동", "단일 상수 보정으로 충분한지 시간 구간 대조",
        "game_clock_offset_min_s game_clock_offset_max_s game_clock_offset_median_s"),
    "no_positive_zone_pair": ("game_clock", "missing_data", "medium",
        "양의 자기장 반경 쌍 없음", "규칙 확인 후 자기장 과제 마스크",
        "game_clock_checked game_rows positive_zone_pair_rows"),
    "collection_flag_record_disagree": ("inventory+positions", "source_conflict", "medium",
        "수집 플래그·실제 위치 기록 불일치", "수집 상태 메타데이터 확인",
        "telemetry_fetched position_rows"),
    "multiple_deaths_possible_reentry_not_duplicate": ("kills", "normal_exception", "info",
        "반복 사망은 재진입일 수 있음", "서로 다른 시각의 사망을 중복 삭제하지 않음",
        "actors_with_multiple_death_times kill_rows"),
    "missing_attacker_can_be_environment_damage": ("damage", "normal_exception", "info",
        "공격자 없는 피해는 환경 피해일 수 있음", "피해 원인을 확인하고 환경 타입 분리",
        "missing_damage_attacker_rows damage_rows"),
    "non_BR_rules_task_masks_required": ("inventory+structure", "normal_exception", "info",
        "비-BR 목표 적용 범위 구분", "성립하지 않는 순위·자기장 목표 마스크",
        "map_name game_mode br_rule_candidate"),
    "query_incomplete": ("audit", "unknown", "high",
        "조회 미완료", "실패 검사 재실행; 정상 또는 결측으로 단정 금지",
        "query_failures"),
}
for _code, _query in {
    "invalid_damage_value_rows": "damage", "negative_damage_rows": "damage",
    "negative_zone_radius_rows": "game_clock", "negative_alive_rows": "game_clock",
}.items():
    RULES[_code] = (_query, "invalid_record", "high", "무효 값 확인",
                    "해당 사건·필드를 격리하고 원본 대조", _code)
RULES["repeated_victim_death_timestamp_rows"] = (
    "kills", "context_candidate", "medium", "동일 피해자·시각 사망 반복 후보",
    "원본 사건 식별자를 대조", "repeated_victim_death_timestamp_rows kill_rows")
for _code in ("roster_rank_missing", "roster_rank_invalid", "participant_rank_invalid",
              "winner_not_unique_or_missing", "rank_exceeds_roster_count", "rank_ties_or_duplicate_rank"):
    RULES[_code] = ("structure", "context_candidate", "high", "결과 라벨 추가 검토",
        "경기 규칙·결과 원본 대조", RULES["outcome_needs_reconciliation"][-1])

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS audit_runs(
 run_id TEXT PRIMARY KEY, captured_at TEXT NOT NULL, source_sha256 TEXT NOT NULL,
 classification_code_sha256 TEXT NOT NULL, summary_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS match_quality(
 run_id TEXT NOT NULL REFERENCES audit_runs, match_id TEXT NOT NULL,
 map_name TEXT, game_mode TEXT, match_type TEXT, created_at TEXT,
 anomaly_class TEXT NOT NULL, classification TEXT NOT NULL,
 trajectory_status TEXT, rank_status TEXT, combat_status TEXT,
 locked_not_inspected INTEGER NOT NULL, details_json TEXT NOT NULL,
 PRIMARY KEY(run_id,match_id));
CREATE TABLE IF NOT EXISTS issues(
 run_id TEXT NOT NULL, match_id TEXT NOT NULL, code TEXT NOT NULL,
 severity TEXT NOT NULL, nature TEXT NOT NULL, source_query TEXT NOT NULL,
 label_ko TEXT NOT NULL, next_action TEXT NOT NULL, evidence_json TEXT NOT NULL,
 PRIMARY KEY(run_id,match_id,code),
 FOREIGN KEY(run_id,match_id) REFERENCES match_quality(run_id,match_id));
CREATE TABLE IF NOT EXISTS issue_reviews(
 review_id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, match_id TEXT NOT NULL,
 code TEXT NOT NULL, reviewed_at TEXT NOT NULL, reviewer TEXT NOT NULL,
 disposition TEXT NOT NULL CHECK(disposition IN
 ('pending','confirmed','corrected','accepted_context','false_positive')),
 note TEXT NOT NULL,
 FOREIGN KEY(run_id,match_id,code) REFERENCES issues(run_id,match_id,code));
CREATE TRIGGER IF NOT EXISTS reviews_no_update BEFORE UPDATE ON issue_reviews
 BEGIN SELECT RAISE(ABORT,'Reviews are append-only'); END;
CREATE TRIGGER IF NOT EXISTS reviews_no_delete BEFORE DELETE ON issue_reviews
 BEGIN SELECT RAISE(ABORT,'Reviews are append-only'); END;
CREATE INDEX IF NOT EXISTS quality_match ON match_quality(match_id);
CREATE INDEX IF NOT EXISTS quality_class ON match_quality(run_id,anomaly_class,map_name,game_mode);
CREATE INDEX IF NOT EXISTS issue_code ON issues(code,severity,run_id);
CREATE INDEX IF NOT EXISTS review_issue ON issue_reviews(run_id,match_id,code,review_id);
CREATE VIEW IF NOT EXISTS latest_match_quality AS
 SELECT * FROM match_quality WHERE run_id=(
 SELECT run_id FROM audit_runs ORDER BY captured_at DESC,run_id DESC LIMIT 1);
CREATE VIEW IF NOT EXISTS latest_issues AS
 SELECT i.*,coalesce((
 SELECT r.disposition FROM issue_reviews r
 WHERE r.run_id=i.run_id AND r.match_id=i.match_id AND r.code=i.code
 ORDER BY r.review_id DESC LIMIT 1),
 CASE WHEN i.nature='normal_exception' THEN 'context_note' ELSE 'pending' END) AS review_status
 FROM issues i WHERE run_id=(
 SELECT run_id FROM audit_runs ORDER BY captured_at DESC,run_id DESC LIMIT 1);
"""


def split_codes(value):
    return set(filter(None, (value or "").split("|")))


def read_only(db):
    conn = sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def build(report, db):
    report, db = Path(report), Path(db)
    summary = json.loads((report / "summary.json").read_text())
    source_hash = hashlib.sha256((report / "match_quality.csv").read_bytes()).hexdigest()
    run_id = summary["captured_at"] + ":" + source_hash[:16]
    with (report / "match_quality.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == summary["inventory_matches"] == len({r["match_id"] for r in rows})
    db.parent.mkdir(parents=True, exist_ok=True)
    catalog = Counter()
    with sqlite3.connect(db) as conn:
        conn.executescript(SCHEMA)
        conn.execute("INSERT OR IGNORE INTO audit_runs VALUES(?,?,?,?,?)", (
            run_id, summary["captured_at"], source_hash,
            summary["classification_code_sha256"], json.dumps(summary, ensure_ascii=False)))
        for row in rows:
            locked = row["locked_not_inspected"].lower() == "true"
            codes = split_codes(row["issue_codes"]) | split_codes(row.get("rank_issue_codes"))
            if locked:
                assert not codes
            conn.execute("INSERT OR IGNORE INTO match_quality VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                run_id, row["match_id"], row["map_name"], row["game_mode"], row["match_type"],
                row["created_at"], row["anomaly_class"], row["classification"],
                row["trajectory_status"], row["rank_status"], row["combat_status"], int(locked),
                json.dumps(row, ensure_ascii=False)))
            for code in sorted(codes):
                query, nature, severity, label, action, fields = RULES[code]
                if code in split_codes(row["context_notes"]):
                    nature, severity = "normal_exception", "info"
                evidence = {key: row.get(key) or None for key in fields.split()}
                evidence.update(aggregation_unit="match", scope="stored aggregate screening")
                conn.execute("INSERT OR IGNORE INTO issues VALUES(?,?,?,?,?,?,?,?,?)", (
                    run_id, row["match_id"], code, severity, nature, query, label, action,
                    json.dumps(evidence, ensure_ascii=False)))
                catalog[(code, severity, nature, query, label, action)] += 1
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert not conn.execute("PRAGMA foreign_key_check").fetchall()
        counts = {table: conn.execute(f"SELECT count(*) FROM {table} WHERE run_id=?", (run_id,)).fetchone()[0]
                  for table in ("match_quality", "issues", "issue_reviews")}
    with (report / "issue_catalog.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["code", "severity", "nature", "source_query", "label_ko", "next_action", "matches"])
        writer.writerows((*key, n) for key, n in sorted(catalog.items()))
    result = dict(run_id=run_id, source_sha256=source_hash, **counts,
                  path=str(db.resolve()), original_database_modified=False)
    (report / "index_receipt.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def query(db, match=None, issue=None, map_name=None, status=None, list_codes=False, limit=20):
    with read_only(db) as conn:
        if list_codes:
            return [dict(r) for r in conn.execute("""
                SELECT code,nature,severity,label_ko,review_status,count(*) AS matches
                FROM latest_issues GROUP BY 1,2,3,4,5 ORDER BY matches DESC,code""")]
        clauses, params = [], []
        for field, value in (("q.match_id", match), ("q.map_name", map_name), ("q.anomaly_class", status)):
            if value is not None:
                clauses.append(field + "=?"); params.append(value)
        if issue:
            clauses.append("EXISTS(SELECT 1 FROM latest_issues i WHERE i.match_id=q.match_id AND i.code=?)")
            params.append(issue)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        matches = [dict(r) for r in conn.execute(
            "SELECT * FROM latest_match_quality q" + where + " ORDER BY match_id LIMIT ?", [*params, limit])]
        for row in matches:
            row.pop("details_json")
            row["issues"] = [dict(r) for r in conn.execute("""
                SELECT code,severity,nature,label_ko,next_action,evidence_json,review_status
                FROM latest_issues WHERE match_id=? ORDER BY severity,code""", (row["match_id"],))]
            for item in row["issues"]:
                item["evidence"] = json.loads(item.pop("evidence_json"))
        return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build")
    create.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    lookup = commands.add_parser("query")
    lookup.add_argument("--match")
    lookup.add_argument("--issue")
    lookup.add_argument("--map", dest="map_name")
    lookup.add_argument("--status", help="anomaly_class; see classification_summary.csv")
    lookup.add_argument("--list-codes", action="store_true")
    lookup.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if args.command == "build":
        result = build(args.report, args.db)
    else:
        if not 1 <= args.limit <= 1000:
            parser.error("--limit must be between 1 and 1000")
        result = query(args.db, args.match, args.issue, args.map_name, args.status, args.list_codes, args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
