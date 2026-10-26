# G3 경기 유형·희귀 이동 표본·과거 공간 prior

승인: 저자의 경기 유형 분리, 재진입·긴급 호출 표본 확보, 누수 없는 패치 간 prior 설계와 자기 피드백 요청.
이번 단계는 데이터 색인·표본 구성·prior 프로토타입·시험이며 추가 모델 학습은 하지 않는다.
V3 원천·bank·분할, G1/G2 완료 실행은 보존한다. 보호·41.2 상세 접근 금지.
산출물은 `/Users/lunecid/PUBG_Data/runs/20261020_world_model_g3_v1`, 보고서는 저장소에 둔다.

## 적용 전 변경 범위와 근거

새 모듈 `world_model/cohorts.py`: 분류, 사건 구간, 정답 지원, 임시 시각별 adapter.
새 실행 도구: 전체 허용 경기의 SQLite 색인, 중복 제거 표본 목록, 과거 prior 구축·검사.
새 시험: 분류 충돌, 죽음/재진입 경계, 미래 사건 정보 격리, prior 출처/시간/지도 제한.
기존 함수 `actor_timeline`, `targets_at`, `load_sample` 의미를 유지한다.
graphify 기존 색인으로 연결을 찾고 현재 원문으로 확인했다.

## 경기 분류

map와 match_type을 함께 사용한다. tutorialatoz/Tutorial_Main, trainingroom/Range_Main은 별도.
두 신호가 충돌하면 unresolved로 보류한다. 일반 official, ranked competitive,
airoyale를 서로 구분한다. 일반·경쟁 BR 주 집합과 AI royale·튜토리얼·훈련장 보조 집합을
모두 보존하며 각각 fit ID를 가진 manifest를 만든다.
분류는 결과 순위·미래 행동을 쓰지 않는다. BR 형식 후보는 기존 메타데이터를 사용하되
해당 경기의 승패/순위 label 유효성과 동일시하지 않는다.

## 사건과 표본

재진입: 기존 actor_timeline의 reentry_direct / reentry_inferred 기록을 따로 유지한다.
긴급 호출: bank의 mode=4 연속 구간(관측 시각 간격 포함). 사람별 탑승 구간이며
동일 비행기·동일 호출의 고유 수를 알 수 없으므로 이를 “독립 호출 수”로 부르지 않는다.
사건마다 actor, team, 시작/종료, evidence, match ID를 색인한다.
기존 30초 stride anchor 외에 10초 bank 시각에서 사건 주변 -60~+120초의 후보를 만든다.
120초 history 최소, 현재 유효/known focal actor만 입력. 원본 bank를 바꾸지 않는다.
현재 팀원 집합 고정, episode/segment를 가로지르는 future trajectory는 계속 마스크한다.
재진입 이전 시각의 회귀가 죽음/비행 전환을 강제로 연결해서는 안 된다.

선정 정보(미래 사건 종류·발생 시각·정답 support)는 manifest/평가에만 쓰며 입력에 넣지 않는다.
우선 각 사건의 현재/사후 구간에서 actor의 30초 이상(첫 3 horizon 연속) 유효 정답이 있는
첫 시각을 대표 표본으로 고른다. 여러 사건이 같은 (match,frame,ego)를 고르면 합쳐 저장한다.
중복 이벤트 목록을 보존한다. 모든 후보 및 제외 이유는 SQLite에 남긴다.
기존 sparse stride에 놓친 사건 수와 실제 추가된 지원을 별도로 측정한다.

“충분”은 우선 자료 준비 기준으로 각 직접 사건 종류에서 학습 100경기·500 actor-event,
개발 30경기·100 actor-event 이상의 30초 지원을 목표로 둔다.
이는 통계 검정력·수렴 보장이 아니다. 미달하면 임계값을 낮추지 않고 부족을 보고한다.
간접 재진입은 직접 재진입 수에 합치지 않는다.
훈련 oversampling은 사건→경기→팀→구간으로 군집을 고려하고,
일반 표본과의 혼합비는 후속 학습 전에 고정한다. 평가 기본 집합은 자연 분포,
희귀 challenge 집합은 별도 보고하고 합산 전체 성능으로 바꾸지 않는다.

## 패치 간 prior

40.2의 허용 official/competitive fit 경기만 donor로 구축한다. 튜토리얼·훈련장·airoyale 제외.
100m 셀 연결의 비차량/육상/수상 모드를 분리하며 항공·재진입은 지상 통행 prior에서 제외한다.
새 테이블에 map, source patch, donor match, counts, donor end time을 보존한다.
대상과 동일 map, donor 종료 < 대상 시작, donor != target, donor ∈ 실제 fit ID 조건을 모두 요구한다.
40.2 학습에서 temporal prefix를 쓰면 마지막 경기까지 포함한 prior와 표현 분포 차이를 줄일 수 있다.
새 ablation/inner fold는 donor ID를 다시 제한해야 하며 full fit prior를 재사용하지 않는다.

입력 제안: 기존 observed/inferred와 별도로 historical edge channel.
features = log1p(횟수), log1p(기여 경기수), 모드, source/target patch,
최근 donor로부터 경과 일수, known_change, geometry_verified=0.
0은 불통 label이 아니다. 신뢰도는 정확한 확률로 해석하지 않으며 강제 마스크/최단경로 제약에 쓰지 않는다.
41.1의 Erangel 지형 파괴, Rondo/Taego/Miramar/Erangel 굴착 및 ranked 변경,
봇 위치 행동 변경 때문에 과거 관측은 현재 지형 truth가 아니다.
출처: https://www.pubg.com/en-asia/news/9926 (2026-04-08), World / Bot Refinement.
지도 ID 출처: https://raw.githubusercontent.com/pubg/api-assets/master/dictionaries/telemetry/mapName.json .
“패치 노트에 변경 없음”을 불변 지형의 증거로 삼지 않는다.

검사: donor ID 교집합·시간·map·mode, 자기 경기 제거, validation/protected 행 주입 거부,
현재·미래 사건 메타데이터 변조에도 encoder input 불변.
현재 bank 기준으로 과거 prior의 존재를 확인하는 것과 지형/충돌 정확도는 구분한다.
validation 경로를 사용해 prior weight·신뢰도·필터를 fit하지 않는다.

계산: bank 1경기씩 읽고 SQLite batch insert, CPU·private memory 48GiB 미만.
이 단계에서는 가중치 갱신이나 새로운 검증 성능의 우월성 판정을 하지 않는다.
