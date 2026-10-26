"""Reclassify saved aggregate queries and build inspectable quality deliverables.

No database access, no observation extraction, no model training.
"""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT
import csv
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = REPO_ROOT
OUT = ROOT / "reports/corpus_quality_2026-10-20"
spec = importlib.util.spec_from_file_location(
    "audit", ROOT / "src/pubg_lab/data/quality/classify_collected_data.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

ANOMALY_LABELS = {
    "no_anomaly_detected_in_checked_rules": "검사 범위에서 이상 미발견",
    "source_field_conflict": "원천 필드 간 불일치",
    "invalid_or_conflicting_record": "무효·상충 기록",
    "core_record_problem": "핵심 기록 문제",
    "context_or_completeness_review": "맥락·완전성 추가 검토",
    "unavailable": "미수집·텔레메트리 없음",
    "unknown_incomplete_audit": "검사 미완료",
    "protected_not_inspected": "보호 중·상세 미검사",
}
READY_LABELS = {
    "no_issue_detected_in_checked_rules": "검사한 기본 규칙 통과",
    "usable_after_correction_or_masking": "전처리·구간 검토 후 사용 후보",
    "quarantine_for_core_tasks": "핵심 과제에서 보류",
    "insufficient_evidence": "근거 부족",
    "uncollected_or_no_telemetry": "미수집·텔레메트리 없음",
    "protected_not_inspected": "보호 중·상세 미검사",
}
REASON_LABELS = {
    "outcome_needs_reconciliation": "순위·우승 필드 대조 필요",
    "ingame_broad_xy_outlier": "넓은 좌표 범위 밖 후보",
    "track_gaps_need_window_mask": "30초 초과 관측 간격",
    "elapsed_and_event_clock_need_alignment": "이벤트·경과 시계 정렬 필요",
    "clock_offset_not_constant": "양성 페이즈 시계 오프셋 범위 5초 초과",
    "nonroster_position_rows": "참가자 명단 밖 위치 기록",
    "mobility_discontinuity_needs_episode_review": "200m/s 초과 이동 전이 후보",
    "invalid_or_missing_ingame_position_fields": "양성 페이즈 위치·체력·ID 결측/범위 문제",
    "participants_without_position_records": "위치 기록 없는 참가자",
    "missing_match_start": "시작 사건 없음",
    "missing_match_end": "종료 사건 없음",
    "no_position_records": "위치 기록 없음",
    "no_positive_zone_pair": "양의 반경 자기장 쌍 없음",
    "no_participant_records": "참가자 기록 없음",
    "no_roster_records": "로스터 기록 없음",
    "repeated_state_timestamp_rows": "같은 시각에 같은 상태가 반복된 후보",
    "conflicting_state_timestamp_rows": "같은 시각의 상태 충돌",
    "negative_damage_rows": "음수 피해량",
    "invalid_damage_value_rows": "결측·비유한 피해량",
    "repeated_victim_death_timestamp_rows": "동일 피해자·시각의 반복 사망 기록",
}


def count_table(counter, labels):
    return "\n".join(
        f"| {labels.get(k, k)} | {v:,} | {v / sum(counter.values()):.2%} |"
        for k, v in counter.most_common()
    )


def main():
    inventory = json.loads((OUT / "inventory.json").read_text())
    protected = audit.locked_ids()
    intended = {r["match_id"] for r in inventory} - protected
    coverage = {name: set() for name in audit.QUERIES}
    results = []
    for path in sorted((OUT / "chunks").glob("*.json")):
        name = path.stem.split("_", 1)[1]
        if name not in audit.QUERIES:
            continue
        result = json.loads(path.read_text())
        expected = hashlib.sha256(audit.QUERIES[name].encode()).hexdigest()
        assert result["query_sha256"] == expected, (name, "stale query cache")
        assert not protected.intersection(result["match_ids"])
        assert not coverage[name].intersection(result["match_ids"]), (name, "duplicate chunks")
        coverage[name].update(result["match_ids"])
        results.append({name: result})
    assert all(ids == intended for ids in coverage.values()), {
        k: len(intended - ids) for k, ids in coverage.items()
    }
    rows = audit.classify(inventory, results, protected)
    assert len(rows) == len(inventory) == len({r["match_id"] for r in rows})
    for row in rows:
        if row.get("event_duration_s") is not None:
            row["event_minus_metadata_duration_s"] = float(row["event_duration_s"]) - float(
                row["duration"]
            )
    audit.write_csv(OUT / "match_quality.csv", rows)
    (OUT / "manifests").mkdir(exist_ok=True)
    bucket_fields = [
        "match_id",
        "map_name",
        "game_mode",
        "match_type",
        "anomaly_class",
        "classification",
        "trajectory_status",
        "rank_status",
        "combat_status",
        "issue_codes",
        "locked_not_inspected",
    ]
    for bucket in sorted({r["anomaly_class"] for r in rows}):
        audit.write_csv(
            OUT / "manifests" / f"{bucket}.csv",
            [
                {key: r.get(key) for key in bucket_fields}
                for r in rows
                if r["anomaly_class"] == bucket
            ],
        )
    anomalies = Counter(r["anomaly_class"] for r in rows)
    readiness = Counter(r["classification"] for r in rows)
    reasons = Counter(code for r in rows for code in r["issue_codes"].split("|") if code)
    group = Counter(
        (r["map_name"], r["game_mode"], r["match_type"], r["anomaly_class"], r["classification"])
        for r in rows
    )
    audit.write_csv(
        OUT / "quality_by_domain.csv",
        [
            dict(
                map_name=a, game_mode=b, match_type=c, anomaly_class=d, classification=e, matches=n
            )
            for (a, b, c, d, e), n in sorted(
                group.items(), key=lambda v: tuple(str(x) for x in v[0])
            )
        ],
    )
    daily = Counter((str(r["created_at"])[:10], r["anomaly_class"]) for r in rows)
    audit.write_csv(
        OUT / "quality_by_day.csv",
        [dict(date_kst=a, anomaly_class=b, matches=n) for (a, b), n in sorted(daily.items())],
    )
    rank_group = defaultdict(Counter)
    numeric = lambda r, k: float(r.get(k) or 0)
    detail = [r for r in rows if not r["locked_not_inspected"]]
    for row in detail:
        key = (row["map_name"], row["game_mode"], row["match_type"])
        rank_group[key]["matches"] += 1
        for field in (
            "participant_rows",
            "human_participants",
            "bot_participants",
            "participant_rank_disagree",
            "human_rank_disagree",
            "bot_rank_disagree",
        ):
            rank_group[key][field] += int(numeric(row, field))
    audit.write_csv(
        OUT / "rank_consistency_by_domain.csv",
        [
            dict(map_name=k[0], game_mode=k[1], match_type=k[2], **v)
            for k, v in sorted(rank_group.items(), key=lambda v: tuple(str(x) for x in v[0]))
        ],
    )
    rollup_keys = [
        "position_rows",
        "ingame_position_rows",
        "basic_valid_position_rows",
        "ingame_invalid_xyz",
        "ingame_missing_health",
        "ingame_invalid_health",
        "ingame_zero_xyz",
        "ingame_missing_actor",
        "nonroster_position_rows",
        "monster_position_rows",
        "damage_rows",
        "kill_rows",
        "participant_rank_disagree",
        "human_rank_disagree",
        "bot_rank_disagree",
        "human_participants",
        "bot_participants",
        "repeated_state_timestamp_rows",
        "conflicting_state_timestamp_rows",
        "negative_damage_rows",
    ]
    totals = {key: int(sum(numeric(r, key) for r in detail)) for key in rollup_keys}
    rank_counts = Counter(r["rank_status"] for r in rows)
    diagnostic_path = OUT / "rank_lifecycle_diagnostic.json"
    diagnostic = json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else None
    diagnostic_groups = defaultdict(Counter)
    if diagnostic:
        for row in diagnostic["rows"]:
            group_key = (row["actor_kind"], row["death_group"])
            for field in (
                "participant_rows",
                "comparable_rows",
                "rank_disagree",
                "personal_rank_better_than_team",
                "personal_rank_worse_than_team",
                "disagreement_in_winning_team",
            ):
                diagnostic_groups[group_key][field] += row[field]
    diagnostic_table = "\n".join(
        f"| {actor} | {group} | {v['comparable_rows']:,} | {v['rank_disagree']:,} | "
        f"{v['rank_disagree'] / v['comparable_rows']:.2%} |"
        for (actor, group), v in sorted(diagnostic_groups.items())
        if v["comparable_rows"]
    )
    br_counts = Counter(
        r["anomaly_class"] for r in rows if r["br_rule_candidate"] and not r["locked_not_inspected"]
    )
    failures = [
        dict(query=name, sqlstate=result.get("sqlstate"), matches=len(result["match_ids"]))
        for chunk in results
        for name, result in chunk.items()
        if not result["ok"]
    ]
    summary = json.loads((OUT / "summary.json").read_text())
    summary.update(
        classification_counts=dict(readiness),
        anomaly_class_counts=dict(anomalies),
        reason_match_counts=dict(reasons),
        rank_status_counts=dict(rank_counts),
        detailed_br_anomaly_counts=dict(br_counts),
        row_totals=totals,
        query_coverage={k: len(v) for k, v in coverage.items()},
        query_failure_details=failures,
        rank_lifecycle_diagnostic=[
            dict(actor_kind=k[0], death_group=k[1], **v) for k, v in diagnostic_groups.items()
        ],
        classification_code_sha256=hashlib.sha256(
            (ROOT / "src/pubg_lab/data/quality/classify_collected_data.py").read_bytes()
        ).hexdigest(),
        summarizer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        audit_completed_not_training=True,
    )
    audit.save_json(OUT / "summary.json", summary)
    audit.save_json(
        OUT / "rulebook.json",
        {
            "unit": "match-level triage with per-task statuses",
            "quality_and_readiness_are_separate": True,
            "positive_phase_position_filter": "is_game > 0",
            "health_basic_range": [0, 100],
            "broad_xy_screen_cm": [0, 1500000],
            "gap_candidate_threshold_seconds": 30,
            "fast_transition_candidate_mps": 200,
            "fast_transition_dt_seconds": [5, 30],
            "clock_offset_alignment_threshold_seconds": 2,
            "clock_offset_range_review_threshold_seconds": 5,
            "thresholds_status": "screening and preprocessing rules, not certified physical limits",
            "source_conflict_priority": [
                "participant versus roster rank",
                "won versus roster rank",
                "start versus metadata map",
            ],
            "never_automatically_called_corruption": [
                "reentry or repeated lives",
                "missing environmental attacker",
                "non-BR missing rank/roster",
                "constant clock offset",
                "unobserved graph edges",
            ],
            "unsupported_claims": [
                "complete collision validation",
                "certified cheat detection",
                "all equipment events checked",
                "final elimination order independently verified",
            ],
            "training_started": False,
        },
    )
    counts_rows = [
        dict(
            anomaly_class=k,
            label_ko=ANOMALY_LABELS[k],
            matches=n,
            share_of_metadata_matches=n / len(rows),
        )
        for k, n in anomalies.most_common()
    ]
    audit.write_csv(OUT / "classification_summary.csv", counts_rows)
    notable = [
        code
        for code in reasons
        if code
        not in (
            "multiple_deaths_possible_reentry_not_duplicate",
            "missing_attacker_can_be_environment_damage",
            "non_BR_rules_task_masks_required",
        )
    ]
    reason_table = "\n".join(
        f"| {REASON_LABELS.get(k, k)} | {reasons[k]:,} | {reasons[k] / len(detail):.2%} |"
        for k in sorted(notable, key=lambda x: -reasons[x])
    )
    date_min = min(str(r["created_at"]) for r in rows)[:10]
    date_max = max(str(r["created_at"]) for r in rows)[:10]
    report = f"""# 수집 코퍼스 데이터 품질 분류

DB를 읽기 전용으로 조회했다. 경기 메타데이터 {len(rows):,}건 중 잠긴 {len(protected):,}경기는 상세 기록을 열지 않았고, 나머지 {len(detail):,}경기를 검사했다. 메타데이터 생성일 범위는 {date_min}–{date_max}다. 모델 학습·원본 수정·데이터 삭제는 수행하지 않았다.

## 1. 판정의 의미

이 표는 경기 단위 요약이다. 한 경기의 일부 잘못된 기록 때문에 경기 전체를 폐기하지 않는다. 원본의 충돌·결측과 모델 사용 전의 시계 정렬·구간 마스킹을 분리했다. ‘이상 미발견’은 아래 검사 범위에 한하며 지형·시야·물리 정확도까지 검증했다는 뜻이 아니다.

| 원본 품질 분류 | 경기 수 | 전체 메타데이터 대비 |
|---|---:|---:|
{count_table(anomalies, ANOMALY_LABELS)}

분류는 상호배타 우선순위다. 한 경기의 모든 문제는 `match_quality.csv`의 `issue_codes`에 별도로 보존한다. 예를 들어 순위 불일치가 우선 분류됐어도 위치 결측이 함께 있을 수 있다. `quality_by_domain.csv`와 `quality_by_day.csv`에서 맵·모드·유형·날짜별 분포를 확인할 수 있다.

`manifests/`에는 분류별 경기 목록을 저장했다. 원본 파일을 이동·삭제한 것이 아니며, 이 목록은 학습/검증 분할을 대체하지 않는다.

## 2. 모델 사용 준비 상태

| 사용 준비 상태 | 경기 수 | 전체 메타데이터 대비 |
|---|---:|---:|
{count_table(readiness, READY_LABELS)}

‘전처리·구간 검토 후 사용 후보’는 보정이 이미 끝났다는 뜻이 아니다. 기준 시계 정렬, 비참가자 제거, 문제 구간 마스킹, 결과 필드 대조 등을 거친 뒤 목표별로 사용할 수 있다는 의미다. `trajectory_status`, `rank_status`, `combat_status`를 분리했다.

## 3. 실제 조회 범위

- 위치 기록 {totals["position_rows"]:,}행, 그중 `is_game>0` 기록 {totals["ingame_position_rows"]:,}행.
- 피해 기록 {totals["damage_rows"]:,}행과 사망 기록 {totals["kill_rows"]:,}행.
- 참가자·로스터 연결, 팀 순위·우승·개인 순위 일치, 경기 시작·종료, 자기장 상태와 시계.
- 위치 시퀀스는 동일 계정별 시간 순으로 비교했다. 양성 페이즈의 결측·비유한 좌표, 체력 범위, 동일 시각 상태 반복/충돌, 관측 간격, 고속 전이 후보를 검사했다.
- 조회 실패: {len(failures)}개 chunk-query. 실패한 검사를 결측 데이터로 바꾸지 않고 미확인으로 분류한다.

`basic_valid_position_rows`는 좌표·체력·ID의 기초 필드 조건 통과 수다. 정확한 지형 위의 통행 가능 위치, 참가자 소속, 관측 시야까지 검증한 usable trajectory row 수가 아니다.

## 4. 발견 항목

아래는 중복 가능한 원인별 경기 수다. 분모는 상세 검사 대상 {len(detail):,}경기이며, 미수집 경기의 기록 부재도 포함한다. 서로 더해 전체 경기 수로 해석하면 안 된다.

| 점검 항목 | 해당 경기 | 상세 검사 대상 대비 |
|---|---:|---:|
{reason_table}

### 순위와 우승

참가자 `win_place`와 연결된 로스터 `rank`의 불일치는 {totals["participant_rank_disagree"]:,}개 참가자 행에서 확인됐다. 인간 식별 규칙에 해당하는 행은 {totals["human_rank_disagree"]:,}개, 봇 식별 규칙에 해당하는 행은 {totals["bot_rank_disagree"]:,}개다. 이는 서로 다른 열의 값이 불일치한다는 직접 증거이며, 어느 열이 실제 최종 순위를 잘못 담았는지를 이번 집계만으로 확정하지 않는다.

이 불일치가 있는 경기 자체는 {sum(numeric(r, "participant_rank_disagree") > 0 for r in detail):,}개다. 미수집·핵심 기록 문제 등이 먼저 분류되는 우선순위 때문에 위 표의 `source_field_conflict` 수와 다를 수 있다. 인덱스는 상위 분류와 관계없이 개별 불일치 코드를 보존한다.

`rank_consistency_by_domain.csv`에는 집단별 분모와 불일치 수를 함께 저장했다. 참가자 순위를 팀 순위로 즉시 복사하지 않고 로스터 결과·최종 탈락 순서·경기 종료를 대조해야 한다. 이번 검사는 최종 탈락 순서 전체 복원까지 실행한 것은 아니다. rank 동률·gap도 기록하고 임의 정렬로 해결하지 않았다.

별도 읽기 전용 집계는 반복 사망 기록과 불일치의 관련성을 확인했다. `no_death_record`는 기록이 없다는 뜻이며 미수집·생존 등 여러 경우를 포함한다. `multiple_death_timestamps`도 재진입의 대리 구분이며 인과적 원인 확정이 아니다.

| 참가자 유형 | 사망 기록 구분 | 비교 가능 참가자 행 | 순위 불일치 | 비율 |
|---|---|---:|---:|---:|
{diagnostic_table}

불일치가 반복 사망 집단 밖에도 있는지 이 표로 확인한다. 재진입 하나로 모든 불일치를 설명하지 않는다. 정확한 발생 원인은 원본 API 응답 및 결과 저장 로직의 후속 대조가 필요하다.

### 위치와 이동

30초 초과 간격이나 200m/s 초과 전이는 검토 후보이며 순간이동 오류·치팅의 증거가 아니다. 사망 이후 재진입, 긴급 수송, 항공기, 로그 공백을 구분해야 한다. 동일 피해자의 서로 다른 시각 사망은 중복으로 처리하지 않았다.

음수 XY 또는 1,500,000cm 초과 XY는 넓은 범위 screening이다. 항공 경로·맵 경계 밖 기록일 수 있으므로 clipping하거나 지상 통행 위반으로 확정하지 않는다. 정확한 맵별 경계·건물·교량·충돌 지형은 이번에 검사하지 않았다.

참가자 roster에 연결되지 않는 위치 기록은 {totals["nonroster_position_rows"]:,}행이고, 그중 `Monster.` 계정 접두 기록은 {totals["monster_position_rows"]:,}행이다. 환경 NPC·훈련장 등장 객체·명단 손실을 구분해야 한다. 모두 인간·봇 플레이어 그래프에 넣으면 안 된다.

체력 0–100 범위를 벗어난 양성 페이즈 기록은 {totals["ingame_invalid_health"]:,}행이다. 이 선별 규칙은 모든 객체가 일반 플레이어라는 가정을 보증하지 않는다. NPC·특수 규칙 객체를 분리하기 전에는 이를 모두 손상된 인간 플레이어 기록으로 해석하지 않는다.

### 시간

`event_time - match_start - elapsed_time`의 경기 내 중앙값과 범위를 측정했다. 시계 원점 차이는 정렬이 필요한 사항이지 단독으로 원본 손상의 증거는 아니다. 시계 안정성 검사는 `is_game>0`인 구간에 제한해 pregame의 0 타이머와 섞지 않았다.

### 정상적인 예외

훈련장·튜토리얼에 BR 로스터·순위·자기장이 없는 것은 그 자체로 손상이 아니다. 상태·공간 자기지도 학습에 활용하되 BR 결과 목표를 mask한다. 피해 공격자가 없는 경우에도 환경 피해가 가능하므로 무조건 불량 처리하지 않는다.

## 5. 사용 방침

| 발견 유형 | 영향도 | 현재 근거 수준 |
|---|---|---|
| 개인·팀 순위 불일치 | 순위 지도학습에 높음 | 불일치 자체는 직접 집계; 실제 원인은 미확정 |
| 핵심 ID·조인·시작/종료 오류 | 개인·팀 시간열 구성에 높음 | 저장 필드/연결 검사에 한함 |
| 무효 값·동시각 충돌 | 해당 row/window에 높음 | 지정한 값 범위와 충돌 규칙으로 확인 |
| 시계 원점 차이 | 혼용 시 누수·오정렬 위험이 높음 | 차이는 측정됨; 단독 원본 손상은 아님 |
| 고속 이동·긴 간격·범위 밖 좌표 | 구간별 검토 필요 | 맥락 없는 후보이며 오류 확정 신뢰도는 낮음 |
| 비-BR의 순위 부재·환경 피해 | 규칙에 따른 정상 예외 | 해당 과제의 적용 범위를 구분 |

1. 이상 미발견 경기: 점검 범위와 미검사 항목을 유지하며 다음 전처리 후보로 사용.
2. 필드 충돌·무효 기록: 문제 field/row/window를 분리하고 해결 전 해당 목표의 label/input을 mask.
3. 맥락 검토: 이동 episode·NPC·로그 공백·모드 의미 확인 후 사용 여부 결정.
4. 핵심 연결 문제: 개인·팀 감독 과제에서 보류. 독립적으로 유효한 상태 기록까지 자동 폐기하지 않음.
5. 미수집: 재수집 대상으로 관리. 없는 경로를 보간해 생성하지 않음.
6. 보호 경기: 잠금 유지. 상세 품질을 정상으로 추정하지 않음.

## 6. 한계와 검증

이 검사는 {len(audit.QUERIES)}종 핵심 SQL 집계와 별도 순위 진단으로 구성된다. 각 핵심 query에 대해 보호 집합을 제외한 모든 경기 ID의 포함 여부를 확인했다. 단일 exported DB snapshot을 고정한 것은 아니므로 조회 중 원천이 변경됐다면 시점 차이가 있을 수 있다. 코드와 query hash, chunk별 소요 시간·오류를 남겼다.

장비 이벤트의 수량·순서 정합성, 공격자/피해자 전체 관계의 의미, 재진입의 최종 탈락 복원, 맵별 정밀 좌표·고도, 패치별 게임 규칙, 모든 이벤트 유형의 중복은 미검사다. 현재 분류는 확정적인 ‘완전히 깨끗한 경기 목록’이 아니다.

분류 규칙의 합성 검사에는 정상 사례, 순위 불일치, 비-BR 명단 부재, 재진입 반복 사망, 조인 오류, query 실패, 잠긴 경기 접근 방지, 시계 원점 차이의 해석을 포함했다. 실제 모델 정확도를 검증한 테스트는 아니다.

## 7. 재현 자료

- [경기별 판정 CSV](match_quality.csv)
- [품질 인덱스 사용법](../../docs/data_layer/CORPUS_QUALITY_INDEX.md)
- [검색 가능한 SQLite](quality_index.sqlite)
- [문제 코드 사전](issue_catalog.csv)
- [요약 JSON](summary.json)
- [맵·모드·유형별 분류](quality_by_domain.csv)
- [날짜별 분류](quality_by_day.csv)
- [순위 불일치와 분모](rank_consistency_by_domain.csv)
- [순위·사망 횟수 관계 집계](rank_lifecycle_diagnostic.csv)
- [실제 SQL](queries.json)
- [분류 규칙과 임계값](rulebook.json)
- [결과 확인 노트북](QUALITY_REVIEW.ipynb)
- [읽기 전용 검사 코드](../../tools/classify_collected_data.py)
- [저장된 집계 재분류 코드](../../tools/summarize_corpus_quality.py)

비밀번호·접속 토큰·개인 이름·플레이어 계정 ID는 결과 파일에 저장하지 않았다. 경기 ID는 재현과 필터링을 위해 유지했다.

경기별 분류를 저장한 뒤 `python3 tools/corpus_quality_index.py build`로 문제 코드·수치 근거·검토 상태를 색인한다. 이 인덱스는 경기 단위이며 개별 문제 시각의 row/window 색인은 아직 생성하지 않았다. 자동 검토 표시를 보정 완료로 해석하지 않는다.
"""
    (OUT / "REPORT_KO.md").write_text(report)
    # Notebook reads saved aggregates only. It never reconnects to the DB.
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 수집 코퍼스 품질 점검\n저장된 집계 결과를 확인하는 노트북입니다. DB 재접속·학습은 수행하지 않습니다. `queries.json`에 실제 검사 SQL이 있습니다.\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\nimport json\nimport pandas as pd\n",
                "HERE = Path.cwd()\n",
                "if not (HERE / 'match_quality.csv').exists():\n",
                "    HERE = HERE / 'reports/corpus_quality_2026-10-20'\n",
                "quality = pd.read_csv(HERE / 'match_quality.csv')\n",
                "assert quality.match_id.is_unique\nquality.anomaly_class.value_counts(dropna=False)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["pd.crosstab(quality.map_name, quality.anomaly_class)\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "rank = pd.read_csv(HERE / 'rank_consistency_by_domain.csv')\n",
                "rank[['human_participants','human_rank_disagree','bot_participants','bot_rank_disagree']].sum()\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "queries = json.loads((HERE / 'queries.json').read_text())\n",
                "for name, sql in queries.items():\n    print(name, sql)\n",
            ],
        },
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"quality-{index}"
    audit.save_json(
        OUT / "QUALITY_REVIEW.ipynb",
        {
            "cells": cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        },
    )
    payload = {
        "schemaVersion": 1,
        "items": [
            {
                "id": "corpus-quality",
                "title": "수집 코퍼스 품질 분류",
                "queries": [
                    {
                        "id": "saved-quality-classification",
                        "source": {
                            "label": "PUBG 수집 코퍼스 품질 집계",
                            "files": [
                                {"label": name}
                                for name in ["match_quality.csv", "summary.json", "queries.json"]
                            ],
                            "tables": [
                                {"name": name}
                                for name in [
                                    "pubg.matches",
                                    "pubg.collection_log",
                                    "pubg.participants",
                                    "pubg.rosters",
                                    "pubg.telem_positions",
                                    "pubg.telem_game_states",
                                    "pubg.telem_kills",
                                    "pubg.telem_damage",
                                ]
                            ],
                            "caveats": [
                                "잠긴 94경기는 상세 검사에서 제외했다.",
                                "이상 미발견은 검사한 규칙 범위에 한한다.",
                                "원본 필드 불일치가 경기 전체의 사용 불가를 뜻하지 않는다.",
                            ],
                            "filters": ["경기별 분류", "보호 경기 상세 미조회"],
                        },
                        "columns": [
                            {"field": "label_ko", "label": "품질 분류"},
                            {"field": "matches", "label": "경기 수"},
                        ],
                        "rows": counts_rows,
                        "preview": {
                            "kind": "aggregate",
                            "note": "상호배타적인 경기별 최우선 분류. 개별 원인은 중복될 수 있다.",
                        },
                    }
                ],
            }
        ],
    }
    for name, sql in audit.QUERIES.items():
        payload["items"][0]["queries"].append(
            {
                "id": "check-" + name.replace("_", "-"),
                "source": {
                    "label": "검사 SQL: " + name,
                    "sql": sql,
                    "filters": ["보호 경기를 제외한 경기 ID를 배치 매개변수로 전달"],
                    "caveats": ["이 SQL의 집계 결과를 경기별 분류 규칙에 결합했다."],
                },
            }
        )
    if diagnostic:
        payload["items"].append(
            {
                "id": "rank-conflicts",
                "title": "개인·팀 순위 불일치와 사망 기록",
                "queries": [
                    {
                        "id": "rank-lifecycle",
                        "source": {
                            "label": "개인·팀 결과 필드 대조",
                            "sql": diagnostic["sql"],
                            "tables": [
                                {"name": name}
                                for name in [
                                    "pubg.participants",
                                    "pubg.rosters",
                                    "pubg.telem_kills",
                                ]
                            ],
                            "filters": ["보호된 94경기 제외"],
                            "caveats": [
                                "사망 기록 횟수는 재진입의 확정 라벨이 아니다.",
                                "불일치는 어느 필드가 실제 순위를 잘못 담았는지 단독으로 증명하지 않는다.",
                            ],
                        },
                        "columns": [
                            "actor_kind",
                            "death_group",
                            "comparable_rows",
                            "rank_disagree",
                        ],
                        "rows": summary["rank_lifecycle_diagnostic"],
                        "preview": {
                            "kind": "aggregate",
                            "note": "맵·모드별 원 집계를 참가자 유형과 사망 기록 수로 합산했다.",
                        },
                    }
                ],
            }
        )
    audit.save_json(OUT / "sources_receipt_input.json", payload)
    print(
        json.dumps(
            {
                "matches": len(rows),
                "anomalies": dict(anomalies),
                "readiness": dict(readiness),
                "row_totals": totals,
                "query_failures": len(failures),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
