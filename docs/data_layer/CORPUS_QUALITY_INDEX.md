# 수집 코퍼스 품질 인덱스

현재 점검: **2026-10-20**, 전체 메타데이터 28,458경기. 상세 검사 대상 28,364경기, 기존 잠금 94경기는 상세 미조회. 모델 학습과 원본 DB 수정 없이 읽기 전용 집계로 분류했다.

## 어디에 표시했는가

- [경기별 전체 판정과 수치 근거](../../reports/corpus_quality_2026-10-20/match_quality.csv)
- [검색 가능한 SQLite 인덱스](../../reports/corpus_quality_2026-10-20/quality_index.sqlite)
- [문제 코드·의미·영향도·다음 조치 목록](../../reports/corpus_quality_2026-10-20/issue_catalog.csv)
- [검사 보고서](../../reports/corpus_quality_2026-10-20/REPORT_KO.md)
- [분류별 경기 목록](../../reports/corpus_quality_2026-10-20/manifests/)
- [실행 요약](../../reports/corpus_quality_2026-10-20/summary.json), [인덱스 생성 기록](../../reports/corpus_quality_2026-10-20/index_receipt.json)

키는 `match_id`다. 플레이어 이름·계정 ID·비밀번호는 출력하지 않았다. 경기별로 여러 문제를 동시에 보존한다. 최상위 `anomaly_class` 하나만 보고 다른 문제가 없다고 판단하면 안 된다.

## 분류와 후속 사용

| 원본 품질 분류 | 경기 수 | 의미 |
|---|---:|---|
| `source_field_conflict` | 16,040 | 서로 다른 원천 필드 값의 불일치 확인. 어느 쪽이 정확한지는 별도 확인 |
| `context_or_completeness_review` | 7,323 | 이동 모드·NPC·관측 공백 등의 맥락 또는 기록 완전성 검토 |
| `no_anomaly_detected_in_checked_rules` | 3,912 | 검사한 규칙에서 이상 미발견. 모든 공간·게임 규칙의 정확성을 보증하지 않음 |
| `core_record_problem` | 122 | 수집 완료 표시가 있으나 위치 기록이 없음. 이 중 3경기는 참가자·로스터도 없음. 핵심 과제 보류 |
| `unavailable` | 967 | 미수집·텔레메트리 없음 |
| `protected_not_inspected` | 94 | 잠금 유지. 정상/이상 판정을 하지 않음 |

`classification`은 별도의 사용 준비 상태다. 이상 미발견이어도 시계 정렬이 필요할 수 있고, 순위 필드가 충돌해도 유효한 이동 기록까지 폐기하지 않는다. `trajectory_status`, `rank_status`, `combat_status`는 과제별 검토 출발점이며 사용 허가 또는 보정 완료 표시가 아니다.

순위 불일치가 있는 경기는 실제로 16,600개다. 위 표는 미수집 등 우선 분류가 먼저 적용된 상호배타 요약이다. 개별 문제 코드 검색은 최상위 분류에 가려진 문제도 반환한다.

`issues.nature`로 직접 확인된 불일치(`source_conflict`), 검토 후보(`context_candidate`), 전처리 필요(`preprocessing_requirement`), 기록 부재(`missing_data`), 정상일 수 있는 예외(`normal_exception`)를 구분한다. `severity`는 해당 과제에 미칠 영향도이며 오류 확률이 아니다.

`review_status=pending`은 후속 검토 전이다. `context_note`는 정상일 수 있는 예외를 보존한 메모다. 자동 분류가 `corrected` 또는 검토 완료로 바뀌지는 않는다. 실제 수정은 이번에 수행하지 않았다.

## 검색

저장소 루트에서 실행한다. 검색 명령은 SQLite 파일을 읽기 전용으로 연다. 원천 PostgreSQL에 연결하지 않는다.

```bash
# 문제 코드별 경기 수·의미·검토 상태
python3 tools/corpus_quality_index.py query --list-codes

# 개인·팀 순위가 불일치한 경기와 수치 근거
python3 tools/corpus_quality_index.py query --issue participant_rank_disagree --limit 5

# 특정 경기의 모든 표시 확인: 실제 match_id로 교체
python3 tools/corpus_quality_index.py query --match MATCH_ID

# Erangel에서 핵심 기록 문제가 있는 경기
python3 tools/corpus_quality_index.py query --map Baltic_Main --status core_record_problem
```

직접 SQL로도 연결 가능하다.

```sql
SELECT q.match_id, q.map_name, i.code, i.nature, i.review_status, i.evidence_json
FROM latest_match_quality q
JOIN latest_issues i USING (run_id, match_id)
WHERE i.code = 'participant_rank_disagree';
```

`evidence_json`은 문제를 발생시킨 **경기 단위 집계값**이다. 현재 인덱스는 원본 행 번호나 문제 시각을 모두 추출한 row/window 색인이 아니다. 구간 마스킹을 구현할 때는 이 목록을 출발점으로 해당 원천 사건·시각을 추가로 찾아야 한다. 결측 집계 필드는 `null`로 남기고 검사 성공 여부와 함께 읽는다.

## 재검사와 이력

- `audit_runs`: 조회 시각, CSV SHA-256, 분류 코드 SHA-256, 요약.
- `match_quality`: `(run_id, match_id)` 단위 판정과 전체 집계값.
- `issues`: `(run_id, match_id, code)` 단위 표시·근거·다음 조치.
- `issue_reviews`: 후속 검토자·시각·판정·근거의 추가 전용 이력. 현재 실제 검토 이력은 **0건**.
- `latest_match_quality`, `latest_issues`: 가장 최근 감사의 검색 뷰.

동일 감사·동일 CSV의 재색인은 중복을 만들지 않는다. 새 감사는 이전 판정과 검토 이력을 유지한다. 이전 감사에서 해결한 문제도 새 감사에 자동으로 해결 처리하지 않는다.

```bash
# 기존 저장 집계를 다시 분류할 때. DB 재조회·학습 없음
python3 tools/summarize_corpus_quality.py
python3 tools/corpus_quality_index.py build

# 새 감사 디렉터리를 누적할 때. 해당 폴더에 동일 스키마의 CSV/summary 필요
python3 tools/corpus_quality_index.py --db reports/corpus_quality_2026-10-20/quality_index.sqlite \
  build --report REPORT_DIRECTORY
```

미래 dataset builder는 split manifest에 `match_id`로 이 인덱스를 조인하고 `quality_run_id`, 해당 문제 코드, 실제 적용한 field/window mask를 기록해야 한다. 품질 목록은 학습/검증/테스트 분할이 아니며 보호된 94경기를 train으로 옮기는 근거가 될 수 없다. 결과 라벨 검토 전에는 관련 목표를 보류하고, 공간 prior·normalization은 실제 fit split 안에서만 다시 만든다.

좌표 범위 밖·고속 전이·체력 0–100 범위 밖 기록은 NPC·특수 규칙·항공·수송 맥락을 확인해야 한다. 모든 객체에 일반 플레이어의 물리·체력 규칙이 성립한다고 가정하지 않는다. 정확한 지형/충돌, 장비 사건 전체, 재진입 이후 최종 탈락 순서는 이번 검사로 검증하지 않았다.

## G2 입력 계약 검사 연결 (2026-10-20)

후속 [G2 보고서](../../reports/world_model_g2_2026-10-20/REPORT_KO.md)는 기존 품질 분류를 보존한다.
허용 train/validation 14,593경기의 입력 계약 오류는 0개였지만,
기존 issue 표시가 있는 12,005경기를 정상으로 재분류한 것은 아니다.
경기별 결과는 `/Users/lunecid/PUBG_Data/runs/20261020_world_model_g2_cpu_v1/audit_02/match_audit.jsonl`에
`match_id`, `role`, `errors`, `quality_issues`, `bank_sha256`로 기록되어 있다.
보호·보류 경기는 이 후속 상세 검사에 포함하지 않았다.

## G3 유형·희귀 이동 색인 (2026-10-20)

[G3 보고서](../../reports/world_model_g3_2026-10-20/REPORT_KO.md)의 새 SQLite
`/Users/lunecid/PUBG_Data/runs/20261020_world_model_g3_v1/index_02/cohort_events.sqlite`는
`matches.match_id`로 기존 품질 인덱스와 연결된다. 튜토리얼·훈련장·AI Royale은 오류로
재분류하지 않고 다른 경기 유형으로 보존했다. 직접 재진입과 추정 재진입도 별개다.
`events.event_id → candidates → selected_events`로 사건, 후보, 선정 표본을 추적할 수 있다.
최종 표본은 같은 실행의 `ready_01/samples.jsonl`이며 `event_ids`와 `tiers`를 보존한다.
이 태그와 미래 정답의 지원 여부는 모델 입력에 넣지 않는다.
기존 품질 문제나 10초 bank의 관측 한계를 “표본 확보”로 해소했다고 표시하지 않는다.

## G4 제한 학습 색인 (2026-10-20)

G4의 실제 fit 경기는 G3 주 집합에서 고른 93경기이며 개발은 48경기다.
`/Users/lunecid/PUBG_Data/runs/20261020_world_model_g4_v1/prepared_01/fit_manifest.json`과
`selection.json`에서 경기 ID·시각·개인·층을 연결한다.
기존 품질 판정은 그대로다. 새로운 `TemporalPriorDataset`은 이 실제 fit ID와
donor 종료 < 대상 시작을 모두 만족하는 공간 기여분만 사용한다.
자연 표본과 사건 중심 표본의 태그는 평가·샘플링 메타데이터로 남기며 입력에 넣지 않는다.
보호·보류 경기의 상세 접근이나 역할 변경은 하지 않았다.
