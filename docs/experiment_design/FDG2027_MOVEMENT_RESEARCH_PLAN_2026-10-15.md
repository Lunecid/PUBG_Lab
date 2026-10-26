# PUBG 인간·봇 이동 비교와 안전구역 조건부 경로 생성 연구 설계

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent plan
- Origin Date: 2026-10-15, Asia/Seoul
- Version Label: movement_plan_v1
- Verification Status: DB 구조와 12경기 추출 가능성 VERIFIED; 데이터셋 규칙·모델·실험은 PROPOSED, 학습 및 성능 검증 전.
- 목적: 수업 논문에서 재현 가능한 오프라인 실험을 완료하고 FDG 2027에서 게임 AI 연구로 확장한다.
- 제안 제목: **Human-Like Macro-Movement Generation in PUBG via Zone-Conditioned Diffusion**.

공식 데이터 정의는 [PUBG 공식 API 스키마 감사](../data_layer/PUBG_OFFICIAL_API_SCHEMA_AUDIT_2026-10-15.md)를 우선한다. DB 컬럼의 존재와 공식 필드의 의미·보존 정도는 별도다. 구역 별칭·반경 단위·단계 전환 의미는 raw trace 검증을 완료한 뒤 확정한다.

## 1. 연구의 중심 주장

인간과 봇이 같은 경기에 존재하는 PUBG 관측 자료에서, **상황을 맞춘 이동 차이를 측정하고, 인간 데이터로 학습한 생성모델이 봇의 현재 상태에서도 인간 이동 특성에 가까운 여러 미래 경로를 제안하는지** 평가한다. 목표는 약 10초 해상도의 거시적 이동이다. 조준, 엄폐물 회피, 정확한 경로 탐색, 전투 실력은 현재 자료로 직접 평가하지 않는다.

Diffusion은 인간성의 보장이 아니라 다봉형 경로 분포를 표현하는 한 방법이다. CVAE·확률적 자기회귀 모델과 비교해 유용성이 없다면 그 결과도 보고한다. 기여는 새 diffusion 수식 자체보다 **구역 정보의 공개 시점 보장, 동일 상황의 인간·봇 비교, 인간 유사성과 제약 위반의 동시 평가**에 둔다.

| 질문 | 설계 | 허용되는 결론 |
|---|---|---|
| RQ1. 유사한 경기 상황에서 인간·봇의 거시 이동이 어떻게 다른가? | 관측 자료의 상황별 매칭·가중 비교 | 조건을 맞춘 행동 차이. 봇이라는 속성의 인과효과는 아님 |
| RQ2. 공개된 구역 조건을 쓰는 인간 학습 모델이 유용한 경로 분포를 생성하는가? | 인간 테스트 상태의 예측 평가 + 봇 테스트 상태의 인간 분포 유사성 | 오프라인 경로 생성의 타당성과 범위 |
| RQ3. 후보 선택이 인간 유사성을 유지하면서 이동 제약·구역 대응을 개선하는가? | 동일 후보 수, 공통 selector, 제약별 제거 실험; FDG에서 반복 제어 | 오프라인 후보 개선 또는 해당 시뮬레이터 내 제어 결과 |

**관측된 봇의 실제 미래는 인간형 생성 경로의 정답이 아니다.** 실제 봇과 생성 경로가 다르다는 사실만으로 개선이라 하지 않는다. 생존율·승률 향상은 상호작용 환경에서 별도로 검증해야 한다.

## 2. 확인된 자료와 아직 확인하지 않은 범위

2026-10-15 읽기 전용 DB 점검에서 `pubg`·`stdb`의 부모 테이블 및 뷰 49개, 컬럼 정의 1,130개를 조회했다. `pubg.matches`는 28,458경기이며 이는 원자료 보유 경기 수이지 학습 적격 경기 수가 아니다. Erangel TPP squad의 `official` 경기는 2,777개이며, 이 중 텔레메트리·착지·연속 구간 조건을 통과하는 수는 아직 세지 않았다.

12경기 표본에는 인간 741명, 봇 361명이 있다. 인간 738명/57,133행과 봇 361명/25,709행의 위치가 추출되었다. 봇 361명 모두 최대 간격 11초 조건으로 10개 이상 연속된 위치점이 있다. 이는 이전의 `is_game >= 0.1` 필터 결과라 비행·차량 구간을 포함한다. 최종 도보 학습 창 수로 사용하면 안 된다.

| 실측 결과 | 설계에 주는 의미 |
|---|---|
| `account_id`, `event_time`, `elapsed_time`, 위치, 체력, 차량 정보 존재 | 개인 이동 시계열 재구성 가능 |
| 위치 기록 간격 중앙값 인간·봇 모두 10초 | 10초 간격, 30초 이력 → 60초 경로를 초기 규격으로 설정 |
| `telem_parachute_landing` 존재, 표본 참가자 1,102명 모두 착지 행 존재 | 착지 이후 시작 가능. 전체 코퍼스 결측률은 별도 확인 |
| 봇 361명 모두 착지 행 복수; 서로 다른 착지 timestamp는 87명 | 행 수를 재배치 횟수로 해석하면 안 됨. 동일 시각 상충 행과 의미 중복을 먼저 조사 |
| 인간 124명도 서로 다른 착지 timestamp 복수 | 첫 착지 이후의 수송·재착지·재배치 후보를 별도 분절 |
| 표본 phase change 172행의 `elapsed_time` 전부 NULL | `event_time − LogMatchStart.event_time`으로 통일 |
| 12경기의 `blue_zone_options`는 JSON 문자열을 해석하면 모두 빈 배열 | 진짜 남은 구역 시간을 입력할 수 있다고 가정하지 않음 |
| 전용 revive/redeploy/swim/vehicle ride 테이블 없음 | 공식 API에는 해당 사건이 존재함. raw JSON 복구·재추출 가능성을 먼저 확인하고 불가능할 때만 휴리스틱 사용 |
| 지형·navmesh 확보 및 실제 PUBG 봇 제어 미확인 | MVP의 이동 제약을 지형 통과 가능성으로 부르지 않음 |

근거: [표본 감사](../../reports/bot_trajectory_audit_2026-10-15/audit.json), [현재 스키마와 추가 점검](../../reports/movement_schema_2026-10-15/schema_snapshot.json), [프로젝트 스키마](../data_layer/FDG2027_MOVEMENT_PROJECT_SCHEMA_2026-10-15.md).

## 3. 수업 MVP의 범위와 모집단

첫 데이터셋은 **Erangel / TPP / squad / official / 비커스텀**으로 제한한다. 코호트명과 실제 `game_mode`, `camera_view_type`, `team_size`가 일치하는지 확인한다. 동일 지도·시점·팀 규모에서 인간·봇이 공존하고, `airoyale`와 일반 경기의 구성 차이를 피할 수 있기 때문이다. 이는 선택한 연구 범위이며 최적 코호트가 실험으로 확인되었다는 뜻은 아니다.

공식 `LogMatchStart` 키는 `cameraViewBehaviour`다. 점검한 수집 코드 사본은 `cameraViewType`을 읽으므로 camera 컬럼은 provenance·누락률 확인 전 신뢰하지 않는다. `gameMode`의 공식 코드와 raw camera 값을 대조한다. 시간순 평가는 `createdAt`(API 객체 저장시각) 대신 match-start 사건 시각을 우선한다.

인간 모델 학습은 이 범위의 인간 계정만 사용한다. 봇 기록은 비교·평가용이며 인간 학습 목표에 섞지 않는다. 경기 단위 분할 후 training에 속한 봇 상태는 overlap 기준·진단 방법 개발에만 쓴다. validation/test의 인간·봇은 같은 경기 분할을 공유한다.

FDG에서는 다음 순서로 늘린다: 같은 지도의 시간순 외부 평가 → Taego/Sanhok의 `official` → `airoyale` 별도 평가. `competitive`는 인간 분포 이동 평가에 쓸 수 있지만 봇이 없으면 인간·봇 비교군이 아니다. 원래 팀 위험 예측의 graph-eligible 코호트나 생존 라벨 필터를 그대로 가져오지 않는다.

인간이라는 접두어 라벨은 숙련자·프로·우수 플레이를 뜻하지 않는다. 생성 목표는 **선택한 인간 표본의 이동 분포**이다. 최종 순위로 인간을 골라 학습하면 다른 모집단이 되므로 MVP에서 사용하지 않는다.

## 4. 전처리 순서

| 순서 | 작업 | 출력 / 검증 |
|---|---|---|
| 1 | 경기·참가자·팀 연결 | `(match_id, account_id)` 자연키 중복·누락, roster team 불일치 검사 |
| 2 | 접두어 분류 | human/bot/other/unknown. `Monster.*` 등 참가자 외 계정 제외 |
| 3 | 공통 시계 | `w_s = event_time − telem_match_start.event_time`, UTC 원시 시각도 보존 |
| 4 | 단위·축 정리 | cm→m. x/y 원래 지도축 유지. 지도 크기는 검증된 맵 메타데이터 사용 |
| 5 | 중복·상충 기록 분리 | 동일 사건 payload 중복 제거; 같은 시각의 다른 좌표는 conflict flag. 같은 정수 elapsed_time만으로 삭제하지 않음 |
| 6 | 착지·생존 구간 | 최초의 유효 착지 이후 시작. 최초 확인 kill에서 primary episode 종료. 그 이후 이동은 별도 재배치 후보로 보존 |
| 7 | 이동 모드 분리 | 비행기·차량·EmergencyPickup 구간 제외. vehicle null은 ‘확인된 차량 없음’이지 도보 확정이 아님 |
| 8 | 추가 품질 관리 | 결측, 지도 밖 위치, 긴 공백, 비정상 변위, 재착지/체력 복귀/수송 후보 분절 |
| 9 | 경기 단위 split | 모든 플레이어·창을 한 split에 유지; 누수 감사 후 고정 |
| 10 | 30초 history·60초 future 창 | 원시 row lineage, 실제 timestamp, mask, 제외 이유 저장 |
| 11 | 조건 정렬 | 과거 방향 as-of join. t 이후 구역·팀원·피해 사건 사용 금지 |
| 12 | training 기준 scaling·sampling | training만으로 임계값·normalizer·matching bins 학습 |

### 구간 정의와 관측 한계

- 공식 재배치·소생·수영·승하차 사건이 보존된 원본이 있으면 먼저 복원해 구간 분절에 사용한다. 아래 휴리스틱은 현재 정규화 DB만 사용할 때의 보수적 대안이다.
- 최초 landing timestamp의 복수 상충 좌표는 하나의 착지 시점 후보로 묶고 전후 위치와 검증한다. 같은 시각의 landing 여러 행으로 episode를 늘리지 않는다.
- 최초 kill 이전이라도 두 번째 착지, 큰 위치 불연속, 알려진 수송 수단이 보이면 window를 분절/제외한다. 원인 확인이 안 되는 구간은 `ambiguous_mobility`로 남긴다.
- `health == 0`만으로 사망을 판정하지 않는다. DBNO일 수 있다. MVP는 incapacitated 구간을 제외하고, `health > 0` 및 최초 kill 전이라는 제한을 명시한다.
- 연속 시점 사이에 잠깐 탔다 내린 차량, 수영·절벽·건물 통과는 10초 로그만으로 모두 검출할 수 없다. 기본 자료 명칭은 **착지 후 관측상 비차량 이동**으로 쓰고, 도보 확정 자료라는 표현은 보류한다.
- kill 없는 이탈은 마지막 관측에서 censor한다. `time_survived`, 최종 순위, 최종 사망 여부는 input으로 금지한다.
- 속도 상한은 먼저 training 자료의 이동 모드별 분포와 이상치를 점검해 정한다. 99.5 percentile은 후보 임계값이지 게임 물리 법칙이 아니다. 인위적 임계값 8/12/20 m/s의 민감도 표를 보조로 만들 수 있으나 정답 속도라고 부르지 않는다.

### 창의 구체적 시간 규칙

anchor `t`는 정제된 현재 위치의 실제 event clock이다. history 기준 시각은 `[t−30,t−20,t−10,t]`초, future 기준 시각은 `[t+10,…,t+60]`초로 정한다.

History는 각 기준 시각 **이하**의 최근 위치를 취하며 age ≤11초, 서로 다른 원시 관측 4개, 연속 실제 관측 간격 ≤15초를 요구한다. anchor는 age 0이다. 과거 grid의 future 값을 이용하는 centered interpolation은 사용하지 않는다. 속도는 선택된 실제 시각 차이로 계산하고 첫 history point의 속도와 dt는 mask 처리한다. 첫 grid의 관측 age 때문에 실제 가장 오래된 위치는 t−41초까지 갈 수 있으므로 nominal 30초 history와 실제 관측 범위를 모두 기록한다.

Future target은 각 기준 시각 ±2초 안의 가장 가까운 **실제 관측**을 라벨로 선택한다. 서로 다른 6개 관측을 요구하고 실제 label time도 보관한다. interpolation은 하지 않는다. 실제 horizon 오차를 보고하고, tolerance 1/2초 민감도 및 actual-time 출력 확장은 보조 실험으로 둔다. 입력은 반드시 t까지에서 끝나며 target 선택은 label 단계에서만 한다.

훈련 후보 anchor는 개인별 약 10초 간격으로 만들되, 매 epoch에서 경기→플레이어→창을 계층 표집해 긴 생존자의 과대표집을 줄인다. validation/test 기본 집계는 같은 개인의 지원 구간 `[t−30,t+60]`이 겹치지 않도록 anchor 간격 ≥100초로 둔다. 경기 내 다른 사람의 상관은 통계 단계에서 처리한다.

6개 target을 모두 요구하면 다음 60초까지 관측·이동하는 생존자에 조건화된다. 따라서 모집단은 ‘완전 관측 비차량 60초 구간’이며, 이 결과를 사망 직전이나 전체 플레이어로 확대하지 않는다. horizon별 유효 창 비율과 class별 제외율을 보고하고 20초 horizon 민감도 분석을 추가한다. FDG에서는 부분 target mask와 종료/관측가능성 head를 별도 검토하며 죽은 뒤 위치를 채워 넣지 않는다.

## 5. 안전구역과 정보 접근 범위

공식 필드 경로는 확인했으며, DB 이름의 연구용 별칭은 다음과 같이 제안한다. ‘현재 blue/공개 white’는 기존 코드의 해석으로, 공식 객체 페이지가 이 시간 의미까지 명시하는 것은 아니다. 원시 구역 전환과 좌표·반경을 대조하기 전에는 별칭의 의미를 검증 완료로 표시하지 않는다.

| 원자료 | 새 이름 | 사용 |
|---|---|---|
| `safe_zone_x/y/radius` ← `safetyZonePosition/Radius` | `blue_current_cx/cy/r_m` | 현재 블루존 경계로 해석; raw 검증 필요 |
| `poison_zone_x/y/radius` ← `poisonGasWarningPosition/Radius` | `white_announced_cx/cy/r_m` | 공개된 목표 원으로 해석; raw 검증 필요 |
| `telem_game_states.is_game` | `phase_index`, `shrinking` | 현재 관측된 단계/축소 상태 |
| `blue_zone_options` | `schedule_source`, 선택적 schedule | 비어 있으면 실제 잔여시간을 알 수 있다고 가정하지 않음 |

기본 입력의 구역은 각 history 시각까지 관측된 최신 game state이다. age ≤15초, 반경>0, 유한 좌표를 검증하며 없으면 mask한다. 다음 원이 공개되기 전에는 미래 행으로 채우지 않는다. 현재 상태에서의 단계 번호는 game-state `is_game`의 정수부, 축소 여부는 소수부 .5로 설계하되 실제 trace와 교차 검증한다. `LogPhaseChange.is_game`은 기존 코드에서 종료된 상태를 가리키는 것으로 처리하므로 그대로 현재 상태로 쓰지 않는다.

`is_game < 1`에서는 phase index를 생성하지 않고 pre-zone 상태로 mask한다. Location의 cm 단위는 공식 정의지만, 반경의 cm 해석은 해당 객체 페이지에 명시되어 있지 않으므로 기존 ETL 변환과 raw 기하를 대조한다. 이 검증은 zone-conditioned 학습 전 필수 완료 항목이다.

MVP는 **실제 다음 단계 종료시각, 다음 공개 원, 미래 적 위치**를 input으로 쓰지 않는다. `state_age`는 과거에서 마지막으로 관측한 현재 상태 전환 이후 시간이며, 기록 시작이 늦으면 좌측 절단 flag를 둔다. 잔여시간은 기본 입력에서 제외한다. 확장 시에는 training에서만 산출한 schedule 추정값을 `estimated`로 표시하고 oracle 시간과 구분한다.

기본 모델의 정보는 본인 이력, 팀원 위치, 공개 구역·생존 인원이다. 전역 적 위치/적 밀도는 게임 내 시야가 복원되지 않으므로 기본 입력에서 제외한다. 분석 매칭 또는 `oracle_context` 상한 실험에는 사용 가능하지만 플레이어가 알 수 있는 정보라는 주장은 하지 않는다. 팀원 정확 위치 역시 텔레메트리 기반 상태 표현이며 실제 UI 관측과 일치하는지는 별도 문제다.

## 6. 모델 입력과 출력 계약

다음은 **제안하는 고정 규격**이다. 실제 dataloader나 모델의 runtime shape 검증 결과가 아니다. feature 순서와 symbolic shape는 [JSON 계약](fdg2027_movement_contract_v1.json)에 저장한다.

| 텐서 | shape | feature |
|---|---|---|
| `ego_hist` | `[B,4,8]` | anchor 상대 x,y; vx,vy; health; 직전10초 받은 damage; 관측 age; 이전 유효 관측과 dt |
| `zone_hist` | `[B,4,10]` | blue 상대 cx,cy,r; white 상대 cx,cy,r; phase_index; shrinking; state_age; zone age |
| `team_hist` | `[B,4,3,6]` | 동료별 anchor 상대 x,y; vx,vy; health; age |
| `context` | `[B,4]` | anchor 절대 x/map_extent, y/map_extent; 현재 생존 인원/100; match-start 경과초/1800 |
| feature masks | 각 입력과 동일 shape | 유효값 1; 없음 0. 0으로 채운 값과 실제 0 구분 |
| `team_present` | `[B,4,3]` | 동료 존재/신선한 관측 여부 |
| `target_delta` | `[B,6,2]` | 다음 관측점까지 x,y 변위. 6개를 누적하면 60초 경로 |
| `target_valid` | `[B,6]` | MVP 모두 1; 부분 관측 확장의 계약을 위해 보관 |
| `target_time_offset_s` | `[B,6]` | 실제 라벨 시각−t. 평가·진단 전용 |
| `samples_delta` | `[B,K,6,2]` | K개 생성 후보. 기본 K=20 |
| `selected_xy_m` | `[B,6,2]` | selector가 고른 경로의 지도좌표 |

좌표는 anchor의 관측 위치를 원점으로 평행 이동하고 **x/y 상대좌표·반경·변위는 고정 100m scale**, 속도는 10m/s, 체력·피해는 100, age/dt/state_age는 명시적 초 scale로 나눈다. 지형의 방향을 보존하기 위해 MVP는 진행 방향 회전을 하지 않는다. 절대 위치를 context에 남겨 동일 방향이어도 지도 내 위치가 다른 상황을 구분한다. 표준화나 clipping 추가 시 training statistics만 사용한다.

동료 슬롯은 경기 내 안정된 로컬 인덱스로 정렬한다. 3명 초과 팀은 MVP에서 제외한다. 계정 ID, 이름, 봇 여부, 최종 순위, 사후 생존시간, match ID는 neural input이 아니다. 원시 z와 vehicle fields는 QC에 남기며 기본 출력은 2D이다. 현지 높낮이·통과 가능성은 모델이 이미 안다고 간주하지 않는다.

확률모형은 `p_theta(Y | ego_hist, zone_hist, team_hist, context)`이다. 인간 창만으로 `Y=[delta x,delta y]`를 학습하고 테스트 시 봇 창의 같은 형식 조건에 넣는다. 봇 상태가 인간 training 분포 밖이면 출력은 생성하더라도 별도 OOD 결과로 보고한다.

## 7. 모델과 학습 설정

| 구성 | MVP 제안 |
|---|---|
| 이력 encoder | ego+zone+mask MLP → GRU, hidden 128 |
| 동료 encoder | 슬롯별 shared MLP → masked mean pool; 시각별 이력 encoder와 결합 |
| denoiser | 길이 6의 temporal Transformer, d_model=128, 4 layers, 4 heads, dropout 0.1 |
| 조건 주입 | context 128차원 + diffusion timestep embedding; 각 block에 conditioning |
| forward process | cosine noise schedule, T=100 |
| 목표 | 무작위 step의 noise prediction MSE. delta 표준규격, target mask 적용 |
| optimizer | AdamW, lr=3e−4, weight_decay=1e−4, gradient clip=1 |
| batch | 256 시작; 실제 메모리 측정 후 모든 모델의 effective batch와 변경 기록 |
| 종료 | 최대 100 epochs, validation Energy Score patience 10; 3 seeds {17,42,2026} |
| sampling | DDIM 20 steps; K=20, 보조 K=1; step ablation 10/20/50 |

표준화된 정답 `Y_0`에 `Y_tau=sqrt(alpha_bar_tau)Y_0+sqrt(1-alpha_bar_tau)epsilon`을 만들고 `E||epsilon−epsilon_theta(Y_tau,tau,C)||²`를 최소화한다. noise level tau는 경기 시간 t와 다른 축이다.

처음에는 training 경기 최대 500개·유효 인간 창 최대 100,000개의 pilot로 코드와 비용을 확인한 뒤 적격 training 전체로 늘린다. 실제 확보량이 이보다 작으면 있는 수를 명시한다. hardware/GPU와 수업 제출일은 아직 미지정이며 총 GPU 시간은 첫 1,000 step 실측으로 산정한다. 전체 학습이 완료됐다고 가정한 성능·비용 수치를 쓰지 않는다.

## 8. 기준 모델과 공정한 비교

| 모델 | 역할 / 출력 |
|---|---|
| 실제 봇 로그 | 동일 anchor의 관측 미래. 실행 가능한 제어정책이 아닌 관측 기준선 |
| 정지·등속 | 난이도 점검. `[B,6,2]` 단일 경로 |
| 구역 방향 휴리스틱 | 관측된 white 원으로 향하되 제한속도 적용. 목표가 없으면 정지/등속 규칙 고정 |
| 조건부 인간 궤적 retrieval | training의 비슷한 상태에서 K개 인간 future를 가져옴. 기억 기반 강한 기준선 |
| GRU 회귀 BC | 같은 조건 encoder + MLP, 좌표/변위 회귀. 평균 경로 모델 |
| 조건부 VAE | 같은 encoder + latent decoder, K개 경로. diffusion의 다봉형 이점을 직접 검증 |
| 조건부 diffusion | 제안 모델, K개 경로 |
| 자기회귀 Transformer-GMM | FDG 확장: multimodal autoregressive baseline |

학습형 모델은 같은 인간 training set·input 정보·출력 horizon을 쓴다. 가능한 범위에서 parameter 수와 튜닝 예산을 맞추고 차이는 보고한다. 각 학습형 모델 동일 6개 hyperparameter configuration 예산, 동일 validation 기준을 제안한다. test로 epoch·selector·threshold를 고르지 않는다.

확률모델·retrieval은 모두 K=20에서 평가한다. deterministic baseline을 20번 복제해 다양성이 있는 것처럼 표시하지 않는다. raw output 비교와 공통 selector 적용 비교를 분리한다. **CVAE+selector와 diffusion+selector**가 없으면 효과가 diffusion 때문인지 selector 때문인지 구분할 수 없다.

기존 KRAFTON win-probability / GBDT / PTR 생존위험 모델은 출력 과제가 다르므로 본 연구의 직접 baseline으로 쓰지 않는다. MLMove 구현도 CS:GO 상태·제어 의미가 다르므로 PUBG에서 그대로 재현한 것처럼 이름 붙이지 않는다.

## 9. 인간·봇 비교와 평가 설계

### A. 상황을 맞춘 관측 비교

RQ1 primary matching은 동일 map/mode/match_type/phase, 가능한 동일 경기의 ±30초 구간 안에서 **현재 위치 구역(초기 500m 격자), blue/white signed distance, 체력, 최근 피해 여부, 신선한 팀원 수**를 맞춘다. spatial bin 폭·continuous caliper는 training에서 overlap과 balance를 보고 정하고 validation 전에 고정한다. 만족하는 쌍이 없으면 억지 매칭하지 않는다. 같은 match 내 불충분 시 다른 match 매칭을 별도 결과로 내며 지역/시기 차이를 통제한다.

과거 속도·방향·팀원 간격까지 맞추는 비교는 history-adjusted 보조 분석이다. 이를 primary에서 모두 통제하면 행동 차이 자체를 제거할 수 있다. class별 매칭 전후 수, 제외율, standardized mean difference와 공통 지지영역을 보고한다. SMD 0.1은 진단 목표이지 인과식별 보증이 아니다. 관측되지 않은 숙련도·장비·목적지 선택 차이는 남는다.

### B. 인간 테스트 상태에서의 생성 검증

실제 인간 future 한 개와 K개 생성 분포를 비교한다. **Energy Score**를 주지표로 사전 지정한다. 경로 벡터를 미터 단위로 펼친 거리 `d=||Y−Y'||_2/sqrt(H)`에 대해 `ES=mean_k d(Y_k,Y*)−(1/(2K²))sum_kl d(Y_k,Y_l)`로 계산한다. 낮을수록 좋다. deterministic baseline은 K=1의 퇴화분포로 평가할 수 있다.

ADE/FDE는 평균 sample, minADE@20/minFDE@20, 실제 selector가 고른 1개를 따로 보고한다. min-of-K는 정답을 이용한 사후 최선이므로 실제 선택 성능이 아니다. 다양성(후보 간 평균 거리), 최종점 region coverage/entropy, ensemble calibration도 보조로 본다. 실제 한 개 future만 있으므로 한 상태의 완전한 다봉형 정답 분포가 검증됐다고 하지 않는다.

### C. 봇 테스트 상태에서의 인간 유사성

모든 모델에 같은 봇 history를 준다. anchor 시점 조건만으로 매칭한 held-out 인간 구간을 reference로 삼고, **실제 봇 future, 생성 future, 인간 reference**의 조건부 특징 분포를 비교한다. 매칭은 미래 이동·최종 순위로 하지 않는다. held-out reference는 평가에만 쓰고 selector·OOD threshold·모델 튜닝에는 쓰지 않는다.

초기 특징은 관측점 사이 총 chord 길이, 순변위, 정지 비율, coarse 방향 변화, frozen blue/white 원까지의 signed distance 변화, reference 지점 대비 공간 점유이다. 이 길이·회전은 10초 관측으로 근사한 값이며 실제 도보 총거리·순간 가속도가 아니다. 속도가 거의 0인 점의 회전각은 제외한다.

주지표는 training-human scale로 정규화한 특징들의 **조건별 Wasserstein-1 거리 평균**이다. 각 특징도 따로 제시해 합성점수에 숨기지 않는다. 인간-vs-인간 독립 split 거리를 비교 기준으로 제공한다. 모델-vs-인간 classifier AUC는 보조이며, 지도·상황·sampling 차이를 이용하거나 분류기 자체가 약할 수 있으므로 AUC≈0.5를 인간성 증명으로 해석하지 않는다.

생성된 ego가 다른 경로를 택했을 때 실제 로그의 미래 팀원/적이 같은 행동을 유지한다고 보장할 수 없다. 로그 기반 미래 팀 간격은 `frozen-others replay` 보조 평가로 명시한다. 기본 경로 selector에는 미래 동료 좌표를 제공하지 않는다.

### D. 제약, 구역 대응, 계산 비용

| 축 | 지표 | 주의 |
|---|---|---|
| 거시적 운동 제약 | 변위 기반 속도 상한 초과율, 지도 범위 위반율, 불연속률 | 건물/물/산의 통과 가능성을 증명하지 않음 |
| 공개 구역 대응 | t의 white 원까지 signed distance 변화, 원 내부 진입 비율 | 미래에 공개될 원을 selector에 넣지 않음 |
| 사후 환경 평가 | 실제 미래 blue 경계 대비 밖에 있는 관측점 비율 | label-only 외생 replay 결과. 인과적 피해/생존 개선 아님 |
| 다양성 | sample pairwise distance, endpoint entropy, 선택 후 분포 변화 | 선택이 모드를 지워버리는지 확인 |
| 효율 | 동일 장비 batch=1 p50/p95, K별 end-to-end latency, peak memory | encoder+sampling+selector 포함, warm-up과 반복 수 명시 |
| 자료 지원 | human-training support 내외 성능·coverage | OOD를 버리고 전체 봇에 좋다고 하지 않음 |

위험도/승률 proxy 모델로 생성 경로를 채점하는 것은 선택적 탐색 분석만 허용한다. 관측 자료에 학습된 위험모델은 생성 경로·새 상호작용에서 잘못 작동할 수 있어 성능 개선 증거로 삼지 않는다.

### E. 통계 단위와 누수 방지

경기 단위 80/10/10 고정 분할을 MVP 기본으로 한다. 모든 overlapping window와 등장 인물은 해당 경기 split에 묶는다. 같은 인간 계정의 여러 경기 출현률을 보고하고, FDG에서는 train에 없던 account만 남긴 test 또는 account-match 연결성 검사를 통한 disjoint split을 추가한다. 한 경기의 player만 쪼개 train/test에 넣지 않는다. 연결 성분이 너무 커 split이 불가능하면 이를 보고하고 unseen-account subset 방식을 쓴다.

결과는 창 평균 → 개인 평균 → 경기 평균 순으로 집계하며 bootstrap은 경기 단위 2,000회로 한다. 모델 간 동일 경기의 paired difference와 95% CI를 보고한다. 동일 계정 반복의 영향은 FDG에서 match+account cluster 민감도 분석으로 확인한다. 3 seeds 결과는 각각과 평균/표준편차를 보이며 seed를 수천 개 독립 표본으로 취급하지 않는다. 여러 주요 비교는 Holm 보정하고 p-value보다 효과크기·CI·제약 trade-off를 중심으로 해석한다.

학습 통계·매칭 규칙·selector weight·classifier·support threshold는 train/validation에서 고정한다. 전체 DB로 만든 normalize 값, 재배치 뒤의 최종 death, 미래 phase 경계를 입력에 섞지 않는다. **prefix replay**에서 t 이후 행을 잘라 다시 만든 입력이 전체 DB 입력과 같아야 한다.

## 10. 제약 후보 선택과 FDG 확장

MVP selector는 K개 경로 중 속도/지도 범위 위반 후보를 먼저 표시하고, 유효 후보에서 `J=lambda_z * frozen_white_endpoint_distance + lambda_j * coarse_velocity_change`를 최소화한다. feature cost는 training scale, weight는 validation으로 고정한다. white 원이 없으면 구역 항은 끈다. 전 후보 실패 시 억지 경로를 성공으로 바꾸지 않고 `no_valid_candidate`와 고정 fallback 결과를 함께 기록한다. 후보 원본과 선택본을 모두 남긴다.

여기서 ‘구역에 빨리 들어감’을 인간성으로 정의하지 않는다. 인간은 의도적으로 원 밖에 머물 수도 있다. weight sweep의 **인간 분포 거리 ↔ 구역 대응 ↔ 위반율** Pareto plot으로 trade-off를 보인다. selector를 모든 확률 baseline에 동일 적용한다. human classifier 점수를 selector로 최적화하고 그 classifier로 다시 인간성을 평가하는 순환은 피한다.

FDG의 반복 제어는 2D 수축 원 환경에서 60초 경로를 생성하고 첫 10초만 실행한 뒤 재계획한다. 실제 기록에서 가져온 외생 zone schedule은 evaluator가 보관하고 모델에는 그 시각까지 공개된 원만 보여준다. 사람·봇 로그 재생은 상호작용 없는 replay로, 정책을 실행하는 closed loop와 분리한다. 정지/등속/구역휴리스틱/CVAE/diffusion을 같은 초기 상태·구역 seed·저수준 controller에서 비교한다.

map navigation이 없다면 open-plane macro simulator라고 부른다. 실제 PUBG geometry를 재현했다고 하지 않는다. terrain 추가 시 walkability mask와 local planner의 출처·정확도를 검증하고 모든 모델에 같은 planner를 사용한다. 첫 확장은 교전 없는 ego+팀 협동·구역 대응으로 제한한다. 전투 성능을 보려면 상대 정책과 피해 모델이 별도로 검증되어야 한다.

사람 평가를 추가하면 같은 길이·색·축척·구역 표시의 top-down 영상을 무작위 순서로 제시한다. 정답 label·모델명은 가리고 인간성 선호와 목적 적합성을 따로 묻는다. 8–12명 pilot 후 효과와 ICC로 표본 수를 산정하고, 30–50명 본실험은 예산상 목표로만 둔다. 평가자와 경기의 random effect를 포함한 mixed model을 사용한다. top-down 궤적 평가를 실제 게임 플레이 경험 평가로 확대하지 않는다. 모집 전 소속기관의 사람 대상 연구 절차를 확인한다.

## 11. 필수 실험표와 단계별 산출물

| 실험 | 수업 | FDG |
|---|---|---|
| E0 데이터 품질·class별 유효율·prefix replay | 필수 | 범위 확대 재검증 |
| E1 매칭된 인간·봇 행동 차이 | 필수 | 지도·시간·match type 일반화 |
| E2 등속/heuristic/retrieval/GRU/CVAE/diffusion | 필수 | AR Transformer-GMM 추가 |
| E3 no-zone / blue-only / blue+white | 필수 | OOD 구역·state strata 분석 |
| E4 no-team / with-team | 여유 시 | 필수 |
| E5 raw / common selector, CVAE와 diffusion 둘 다 | 필수 | weight Pareto·지형 제약 |
| E6 horizon 20/60초, K=1/20, steps=10/20/50 | 20/60초·K 비교 필수, step 비교 축소 가능 | 전체 |
| E7 closed-loop macro simulation | 선택 | 제어를 주장할 경우 필수 |
| E8 blinded human ratings | 선택 | 강한 인간성 주장을 위해 권장 |

학습된 모델 비교에서 zone ablation은 input 차이 이외 구조와 budget을 맞춘다. blue-only 대 blue+white를 분리해야 ‘현재 경계 정보’와 ‘공개 목표 정보’의 기여를 구분할 수 있다. 위치 절대 좌표 유무와 모드별 지지영역은 추가 민감도 분석이다.

수업 8쪽은 수업 규정 확인 전의 작업 가정이다. 배분 예: 문제·선행연구 1.5쪽, 데이터와 누수 통제 1.5쪽, 방법 1.5쪽, 실험 2.5쪽, 한계·결론 1쪽. 참고문헌 포함 여부는 실제 수업 규정에 맞춘다. FDG는 같은 연구의 증거를 확장하며 새 연구인 것처럼 중복 출판하지 않는다. 수업 원고의 공개/출판 상태와 투고 규정을 함께 기록한다.

| 기간 | 완료 기준 |
|---|---|
| 10/15–10/25 | 전수 QC, 착지/모드 분절, cohort manifest, 20개 궤적 수동 점검 |
| 10/26–11/08 | split·loader·prefix replay, RQ1, retrieval/GRU/CVAE |
| 11/09–11/22 | diffusion, E2/E3/E5와 3 seeds, 비용 실측 |
| 11/23–12/06 | 수업 원고·그림·재현 묶음. 실제 수업 마감에 맞춰 조정 |
| 12/07–12/20 | FDG 지도/시간 일반화, closed-loop 평가, 승인된 경우 사람 평가 |
| 12/21–01/03 | 결과 freeze, 한계·ablation 보강, 익명 원고와 메타데이터 |
| 01/04–01/10 | 최종 검증 및 제출 준비 |

FDG 공식 안내의 검색 서비스 제공 본문 기준: metadata/abstract **2027-01-04 23:59 AoE**, full paper **2027-01-11 23:59 AoE**. KST로 각각 1/5·1/12 20:59이다. 단일단 ACM review 형식, 보통 본문 16쪽 이내이며 24쪽 초과는 desk reject 안내다. 직접 페이지 열기는 timeout이 있었으므로 제출 직전 공식 원문을 재확인한다. 트랙은 **Game Artificial Intelligence, Player Characters and NPCs**가 적합하다. [FDG 공식 안내](https://www.fdg2027.org/authorinformation.html)

## 12. 성공 기준과 연구의 중단·축소 조건

데이터 성공은 충분한 인간·봇 공통 지지영역, 해석 가능한 제외율, 관측 시점 누수 0건이다. 특정 유효 창 수를 아직 달성했다고 쓰지 않는다. 초기 작업 목표는 수백 경기·수만 유효 인간 창이며 효과 추정의 단위는 경기이다.

Diffusion 기여는 강한 확률 baseline 대비 human test ES와 bot-state 조건부 분포 거리의 개선 또는 명확한 인간성/제약 Pareto 이점으로 판단한다. 승리를 미리 전제하지 않는다. 차이가 작거나 불확실하면 더 가벼운 CVAE/retrieval의 장점과 diffusion의 비용을 보고한다.

공통 지지영역이 너무 작으면 ‘봇 교정’ 결론을 줄이고 인간 이동 생성 연구로 제한한다. 지형 확보가 안 되면 macro feasibility로 제한한다. 반복 제어가 안 되면 제목과 결론에서 control/improvement를 뺀다. 결과가 좋아도 실제 PUBG 봇 개선은 실제 제어 실험 없이 주장하지 않는다.

## 13. 선행연구와 차별화 근거

- **MLMove, SCA/CGF 2024**: 인간 이동 데이터로 Transformer 제어와 사람 평가를 수행한 가까운 선행연구다. 본 연구는 PUBG의 희소 텔레메트리·공개 구역·관측 봇 상태의 지원 범위에 초점을 둔다. [저자 프로젝트](https://davidbdurst.com/mlmove/)
- **Humanlike Behavior in a Third-Person Shooter, CoG 2024**: identity-conditioned causal Transformer를 사용한 프레임 단위 조준·표적 획득 연구다. 우리의 10초 이동과 과제·관측 단위가 다르다. [저자 프로젝트](https://alexfarhang.github.io/humanlikebehavior)
- **MotionDiffuser, CVPR 2023**: 다중 에이전트 궤적 확산과 미분 가능한 비용을 통한 제약 샘플링의 출발점이다. 우리 MVP는 단일 ego의 조건부 생성+공통 사후 selector이며 원 논문의 재현이 아니다. [CVF 공식 논문](https://openaccess.thecvf.com/content/CVPR2023/html/Jiang_MotionDiffuser_Controllable_Multi-Agent_Motion_Prediction_Using_Diffusion_CVPR_2023_paper.html)
- **Diffusion Policy, RSS 2023**: 관측 조건부 action sequence와 receding horizon control을 참고한다. PUBG 위치 변위는 실제 action command가 아니므로 local controller가 별도로 필요하다. [저자 프로젝트](https://diffusion-policy.cs.columbia.edu/)
- 최신 관련 기술의 하나로 **PUBG Ally**의 계층적 제어가 공개되어 있다. 이것을 기존 수집 자료의 `ai.` 봇과 동일 시스템으로 간주하지 않는다. [NVIDIA 기술 설명](https://developer.nvidia.com/blog/how-krafton-built-pubg-ally-a-co-playable-character-powered-by-nvidia-ace/)

이 검토는 설계에 필요한 핵심 문헌의 제한된 확인이며 ‘최초 PUBG diffusion’이라는 전수 문헌검색 결론이 아니다. 문헌의 주장과 제안 모델의 성능을 구분한다. [PUBG 이벤트](https://documentation.pubg.com/en/telemetry-events.html)와 [객체·좌표·단계 정의](https://documentation.pubg.com/en/telemetry-objects.html)는 원시 필드 해석의 기준으로 사용했다.
