# 공간·경기 구조 사전학습 V2: 전체 코퍼스와 최종 순위

2026-10-20. 사용자 수정 반영. **설계만 개정했으며 추출·전처리·학습·평가는 시작하지 않았다.**

이 문서가 [초기 설계](PUBG_WORLD_MODEL_PRETRAINING_DESIGN_2026-10-20.md)의 모집단, 공간 차원, 팀 크기, 순위 과제, 도메인 비교 절을 대체한다. 개인·팀 이종 노드, 인과적 시간 입력, observer/server 분리, 미래 정보 누수 방지 원칙은 유지한다.

## 1. 수정 사항

- 주 분석은 Erangel·40.2 추정 구간·squad만의 모델이 아니라 **수집 코퍼스 전체의 사용 가능한 기록을 이용하는 공통 사전학습 모델**이다.
- 맵·인칭·팀 규모는 입력 조건과 ablation 축으로 다룬다. 작은 코호트도 전체 코퍼스에 포함하며 기존 300경기 기준을 제거한다.
- 최종 팀 순위·우승을 핵심 사전학습 목표에 포함한다. 기존 trajectory bank에 해당 열이 없는 것은 데이터 소스에 없다는 뜻이 아니다.
- 역순의 최종 탈락으로 순위를 복원하되, 이미 존재하는 공식 결과 필드를 우선 연결하고 복원값을 대조한다.
- 전체 코퍼스는 모집단 범위다. validation/test까지 사전학습에 넣거나 기존 잠금을 해제한다는 뜻은 아니다.

## 2. 순위: 최종 팀 탈락의 역순

N개 팀이 있고, 최종 탈락한 순서를 빠른 순서로 e=1,…,N−1이라 하면, 동률과 결측이 없는 완결 경기에서:

`rank = N − e + 1`, 우승 팀은 `rank = 1`.

예를 들어 4개 팀이 A→B→C 순서로 최종 탈락하고 D가 남으면 A=4위, B=3위, C=2위, D=1위다. 이는 **탈락 순서의 역순**이며 수학적 역수 `1/e`와는 다르다. 순위의 역수 `1/rank`는 별도의 보상 변환으로 선택할 수 있지만 공식 순위와 구별한다.

개인의 사망 순서를 그대로 팀 순위로 바꾸지 않는다. 듀오·스쿼드는 팀의 최종 탈락을 사용한다. 개인이 일찍 사망해도 팀이 나중에 우승할 수 있다. 솔로는 개인과 1인 팀의 단위가 일치하지만, 생명 에피소드와 최종 결과는 여전히 구분한다.

### 원천 자료에서 확인한 필드

[2026-10-15 DB 사전](../data_layer/DB_SCHEMA_CATALOG_2026-10-15.md)에 다음 열이 있다.

| 원천 | 용도 |
|---|---|
| `pubg.rosters.rank`, `won` | 팀 최종 순위·우승의 우선 후보 |
| `pubg.participants.win_place` | 참가자 결과와 로스터 결과의 일치 검사·보완 |
| `pubg.telem_kills.victim_rank` | 사건 결과의 보조 증거. 단일 사망의 값이 최종 팀 결과인지 대조 필요 |
| `pubg.telem_kills` 및 재진입·생존 증거 | 최종 탈락 순서와 시각 복원 |
| `pubg.telem_match_end` | 경기 종료·완결성 확인. 보존 필드는 별도 대조 |

공식 [Roster schema](https://documentation.pubg.com/en/_static/swagger/en/schemas/roster.yml)는 rank를 로스터의 경기 순위로 정의한다. [Telemetry events](https://documentation.pubg.com/en/telemetry-events.html#logmatchend)는 `LogMatchEnd.gameResultOnFinished`를 우승 플레이어 정보로 설명한다. API에 정의된 필드가 정규화 DB에도 모두 보존됐다고 가정하지 않는다.

이 확인은 저장된 DB 스키마와 공식 문서 확인이다. 현재 전체 행의 non-null 비율·값 정확도·join 성공률을 새로 측정한 것은 아니다.

### 라벨 생성 절차

1. 경기×로스터 결과를 읽고 participant 연결의 중복·누락을 확인한다.
2. `rank`, `won`, 팀원들의 `win_place`를 대조한다. 승패 flag가 단순 기본값인지도 검사한다. 불일치는 임의 우선순위로 숨기지 않고 `conflict`로 보존한다.
3. 완결 경기에서 각 개인의 사망·재진입 생명 이력을 복원한다. 첫 사망이 아니라 마지막 실제 생명의 종료를 고려한다.
4. 모든 구성원의 최종 상태가 확인되는 탈락 팀은 최종 탈락 시각을 복원한다. 우승 팀은 경기 종료를 이긴 상태로 처리하며 가짜 사망 시각을 만들지 않는다.
5. 최종 탈락 시각의 역순과 공식 순위를 비교한다. 동시각 기록은 이벤트 배열 순서나 계정 문자열로 임의 tie-break하지 않는다. 공식 순위가 일관되면 이를 사용하고, 그렇지 않으면 순위 구간·부분 순서·unknown으로 남긴다.
6. 공식 순위가 없지만 전체 최종 탈락 순서·우승·참가 팀 수가 확실한 경기에만 exact reconstructed rank를 부여한다. 일부 기록만 있으면 비교 가능한 팀 쌍의 순서만 정답으로 쓴다.
7. 순위가 확인돼도 정확한 탈락 시각이 없을 수 있다. `rank_mask`와 `terminal_time_mask`는 독립적으로 저장한다.

`final_team_outcomes`의 제안 열:

`match_id, roster_id, team_id, rank_official, rank_reconstructed, rank_lower, rank_upper, winner, team_count, terminal_elimination_time, source, conflict_flags, rank_mask, winner_mask, terminal_time_mask`.

기존 [라벨 감사](../data_layer/BIGCOMP2027_LABEL_AUDIT_2026-10-05.md)는 첫 사망 사용, 위치·사건 시계 혼용, 봇/재진입의 `time_survived` 대체에 문제가 있었음을 기록한다. 이를 재사용하지 않도록 한다. 그 과거 감사 수치는 이번에 재측정하지 않았다.

최종 사망을 사후 확인하는 것은 label 생성에 허용된다. 그 사후 정보를 과거의 alive 상태·episode·관측 mask를 수정하는 데 사용하지 않는다.

## 3. 전체 코퍼스의 정의

현재 로컬에 보관된 2026-10-05 집계 CSV를 다시 합산한 규모:

| 당시 집계 | 경기 수 | 이번 설계에서의 의미 |
|---|---:|---|
| DB 경기 메타 | 28,458 | 텔레메트리 없는 경기도 포함 |
| 수집 텔레메트리 | 27,491 | 전체 코퍼스 inventory의 기존 출발점 |
| 기존 stdb 범위 | 21,833 | 과거 연구가 훈련장·튜토리얼 등을 제외한 결과 |
| 기존 graph-eligible | 21,824 | 과거 파이프라인 적격성; 신규 모델의 필터가 아님 |
| 기존 13개 큰 코호트 | 17,943 | 이번 주 분석을 이 범위로 제한하지 않음 |

자료: `reports/bigcomp2027_data_profile/csv/funnel_map_mode_persp.csv`. 이는 **현재 DB 전수 재조회 수치가 아니라 저장된 과거 스냅샷**이다. 실제 실행 전에 전체 inventory를 새로 만들고 snapshot ID를 고정한다. 위 CSV에는 수집 자료가 있는 57개 맵×모드×인칭 조합이 있다.

새 모집단은 원천 수집 경기를 출발점으로 하며, 과거 stdb feature 생성 성공 여부·최종 순위 결측·작은 코호트라는 이유로 자기지도 학습 가능 기록을 버리지 않는다. 원시 사건과 위치를 기준으로 새 window를 만든다.

| 기록 종류 | 사전학습 활용 |
|---|---|
| 일반·경쟁전·AI 중심 battle royale | 공간·상태·팀·경기 결과 과제. 각 과제의 라벨 mask 적용 |
| 튜토리얼·훈련장·기타 규칙 | 사용 가능한 공간·상태 복원 과제에 포함. BR 순위·자기장 목표는 규칙이 성립할 때만 사용 |
| 결과 누락·중도 기록 종료 | 유효 prefix의 자기지도 과제 사용, 결과·위험은 관측 범위에 맞게 mask/검열 |
| 텔레메트리 미수집 | inventory에는 남기지만 없는 경로를 만들어 학습하지 않음 |
| NPC/환경 객체 | 인간·봇 참가자와 분리. 참가자 roster 근거 없이 player 노드로 승격하지 않음 |

주 결과 평가는 BR 규칙이 성립하는 경기 전체에서 수행하고, 비-BR 자료의 사전학습 기여를 포함/제거 ablation으로 분리한다. 작은 맵·조건도 표본 수와 불확실성을 함께 공개한다.

기존 수집 기간의 모든 패치를 후보로 포함하며 40.2로 고정하지 않는다. 다만 패치 번호가 추정값이면 `patch_estimate`, `patch_source`, `patch_confidence`로 보관한다. 맵 변경 구간을 한 정적 공간 지도에 무조건 합치지 않는다.

## 4. 여러 맵·인칭·인원에 맞춘 모델 변경

공유 encoder를 사용하고 `map`, `perspective`, `nominal_team_size`, `rule_set`, `patch/map_version`을 조건으로 제공한다. 인간·봇 타입은 주 모델에서 입력하지 않고 평가 집단으로 유지한다.

### 가변 공간

- `cell_key = (map_id, geometry_version, ix, iy)`. 동일 cell 번호를 다른 맵에서 공유하지 않는다.
- 100m grid의 크기는 맵의 검증된 좌표 범위로 계산한다. `82×82`, `R=441`은 Erangel 예시일 뿐 공통 상수가 아니다.
- 맵·버전별 graph를 fit 기록으로만 구축하고 공간 encoder 가중치를 공유한다. 역사 지도 없는 맵은 해당 feature mask를 끄고 이동 증거를 사용한다.
- 절대 metre 단위, 맵 크기로 정규화한 위치, 현재 개인/팀 상대 위치를 구분한다. 작은 맵의 100m와 큰 맵의 100m를 같은 물리 거리로 처리한다.
- Paramo처럼 지형 배치가 달라질 수 있는 맵/규칙은 layout 확인 수준을 기록한다. 배치를 식별하지 못하면 고정 지형 edge를 pooling하지 않고 국소 이동 표현·layout-uncertain mask로 제한한다.
- 새로운 맵 평가에서 그 맵의 평가 경기로 prior를 만들지 않는다. 외부 사전 지도 사용 여부를 별도 설정으로 공개한다.

### 가변 팀

- solo·duo·squad를 포함하되 `M_batch`는 batch 내 실제 팀 slot 수로 padding한다. 기타 모드에서 4명을 초과하면 자동 잘라내지 않는다.
- nominal team size(규칙), 최초 roster 수, 현재 알려진 생존 수, 현재 위치가 유효한 수를 서로 다른 변수로 둔다.
- 솔로에도 1인 팀 노드를 유지한다. pair-distance 목표는 pair가 없으므로 mask하며 0점으로 평균에 넣지 않는다.
- 듀오·스쿼드는 실제 관측된 구성과 위치 mask를 사용한다. roster나 팀 규모를 미래 생존 여부에서 역산하지 않는다.

출력은 `trajectory[B,K,M_batch,H,2]`, `region_logits[B,M_batch,H,R_batch]`, `team_latent[B,Q_batch,d]`로 일반화하고 유효 map region·player·team mask를 둔다. 공통 H=6(10–60초)와 초기 잠재 폭 128 제안은 유지한다.

### 인칭과 관측 경계

FPP/TPP 조건은 게임 모드의 차이를 표현한다. 현재 텔레메트리만으로 두 인칭의 실제 시야·카메라 가림을 복원할 수는 없다. `perspective`는 observer/server 구분과 다른 축이다. observer-private masking은 두 인칭 모두 별도로 적용한다.

## 5. 최종 순위·우승을 핵심 학습 신호로 포함

기존 P1 관측 복원, P2 이동 전이, P3 팀 공동 변화, P4 사건·위험에 **P6 경기 결과**를 포함한다. P5 잠재 teacher는 선택 과제로 유지한다. P6는 장기 후속 연구로 미루지 않고 전체 코퍼스 전처리의 필수 라벨 작업으로 다룬다. 검증된 라벨에만 손실이 적용된다는 원칙은 유지한다.

| 목표 | 제안 출력·손실 |
|---|---|
| 최종 순위 | N에 따른 ordinal rank 분포 또는 순위 percentile 예측. exact rank와 partial order를 구분 |
| 우승 | observer의 목표 팀 win BCE; 서버는 검증된 참가 팀 전체가 표현될 때 masked team softmax CE |
| 상대적 결과 | 서버의 관측 가능한 두 팀 표현에서 최종 순위 순서의 pairwise ranking loss. 같은 순위·미확정 쌍 제외 |
| 최종 탈락 위험 | terminal event가 검증된 경우의 hazard. 현재 생명 사망과 별도 head |

서버 graph가 참가 팀 전체를 포함하지 못하면 우승팀 softmax에 잘못된 후보 집합을 사용하지 않는다. 완전 후보 집합을 만들거나 해당 loss를 mask한다.

N>1인 경기에서 높은 값이 좋은 보조 점수는 `u=(N-rank)/(N-1)`로 정의할 수 있다. N=1, 불명확한 경기 규칙, 참가 팀 수 미확정이면 이 target을 mask한다. 정규화 순위가 같다고 서로 다른 맵/모드의 난도까지 같다는 뜻은 아니다.

현재 N이 observer에게 알려져 있지 않으면 정답 정규화에 사용한 최종 팀 수를 입력으로 넘기지 않는다. 입력과 라벨 계산의 가용 정보를 분리한다.

순위·우승은 팀 node의 목표다. 각 개인에게 팀 결과를 독립적으로 복제해 개인의 인과적 기여도처럼 학습·해석하지 않는다. 개인→팀 메시지 경로를 통해 상태 표현은 함께 학습할 수 있다.

경기→팀→시각 표집 또는 이에 상응하는 가중으로 긴 생존 팀이 최종 결과 loss를 과도하게 반복하지 않게 한다. P6 초기 후보 가중은 rank 0.25, win 0.25, 선택적 pair ranking 0.1이며 검증 전 제안값이다. P6 제거 ablation을 통해 이동 학습에 유용한지 확인한다.

이렇게 학습한 것은 **상황과 결과의 관계**다. 특정 이동 경로를 선택하면 승률이 증가한다는 정책 효과는 여전히 별도 검증이 필요하다.

## 6. 주 실험과 ablation

주 모델: **모든 적격 train 도메인의 공통 사전학습 + 도메인 조건 + 개인·팀 노드 + 공간·경기 결과 목표**. 동일 encoder를 회귀·공동 GMM/CVAE·Diffusion에 제공한다.

아래의 도메인 제외 비교는 실제 재학습 ablation이다. 같은 모델의 조건별 성능표는 subgroup 평가이며 ablation과 별도로 보고한다.

| 축 | 비교 | 평가 질문 |
|---|---|---|
| 맵 | 전체 맵 vs 대상 맵만 학습; leave-one-map-out | 다른 지형 경험이 도움이 되는가, 미학습 맵으로 이전되는가 |
| 인칭 | FPP+TPP vs FPP-only/TPP-only; 필요 시 한 인칭 제외 | 인칭 간 공유가 유효한가 |
| 팀 규모 | solo+duo+squad vs 규모별 학습; leave-one-size-out | 1/2/4인 구조 사이에서 무엇이 이전되는가 |
| 조건 입력 | map/perspective/team-size 조건 각각 제거 | 도메인 조건을 명시하는 효과 |
| 패치·규칙 | 전체 호환 버전 vs 제한 버전; 시간 holdout | 지형/재진입 규칙 변화에 대한 강건성 |
| 비-BR 기록 | 보조 자기지도 자료 포함 vs 제외 | 튜토리얼·훈련장 경험의 전이 효과 |
| 결과 목표 | P6 포함 vs 제외 | 순위·우승 학습이 경로와 팀 표현에 도움이 되는가 |
| 팀 구조 | 개인-only vs 개인+팀 | 다른 팀 규모에서도 팀 node의 추가 효과 |

전체 데이터를 쓰는 모델은 자연히 학습량이 많다. 모든 자료를 쓴 실제 이득과 데이터량/compute를 맞춘 이득을 각각 보고한다. FPP가 특정 맵·경쟁전에 몰리는 등 교란을 줄이기 위해 공통 맵×패치×규칙×팀 규모의 overlap에서 matched 비교도 수행한다. 지지 영역 밖의 인칭 차이를 인과효과라고 부르지 않는다.

주 학습은 전체 도메인의 실제 경기 비율을 반영한 경기 균등 표집을 기준으로 한다. 소수 도메인 균형 표집은 별도 실험으로 두며, 위험·우승 확률 보정은 자연 분포의 내부 검증에서 수행한다. 도메인별 성능과 domain-macro/match-macro 평균을 함께 제공한다.

순위 평가는 normalized rank MAE, ordinal score, 경기 내 순서 일치도를 사용한다. 우승은 NLL/Brier와 reliability, 경기별 1/N 또는 남은 후보 팀 수 기반 기준 모델을 비교한다. 생존 단계별 지표를 나눠 종료 직전의 쉬운 예측이 성능을 지배하지 않게 한다. 기존 경로·팀 거리·Energy score도 유지한다.

## 7. 전체 코퍼스의 분할과 기존 노출 이력

- 기존 657/188/94는 이전 Erangel 실험의 분할 기록이며 전체 코퍼스 split 수가 아니다.
- 경기 ID를 전역 키로 deduplicate하고 모든 기존 split·학습·검증 노출 이력을 registry에 합친다.
- 이전 validation/test 경기와 잠긴 94경기를 새 train으로 자동 이동하지 않는다. 기존 보호 집합과 충돌하는 경기는 별도 보존한다.
- full-corpus train은 이 보호 집합을 제외한 사용 가능한 도메인 합집합이다. metadata를 전수 inventory하는 것과 미래 정답을 열어보는 것은 구분한다.
- leave-one-map/perspective/size-out에서는 제외 도메인의 SSL·공간 prior·normalization까지 제외한다. 전체 코퍼스 pretrained checkpoint를 그대로 사용하면 strict holdout이 아니다.
- 반복 사용한 기존 검증은 개발 결과다. 미사용 final test는 별도로 확보하며, 없다면 그 한계를 그대로 보고한다.

## 8. 다음 전처리 산출물

2026-10-20 후속 품질 점검은 완료했으며, 경기별 판정·문제 코드·근거를 [코퍼스 품질 인덱스](../data_layer/CORPUS_QUALITY_INDEX.md)에 저장했다. 아래 전처리에서는 `match_id`로 품질 판정과 감사 버전을 조인하고 실제 field/window mask를 별도로 남긴다. 이 점검은 학습을 실행한 것이 아니며 기존 보호 경기와 분할 규칙을 변경하지 않는다.

이 절은 다음 작업의 명세다. 이번 턴에서는 아래 데이터를 새로 생성하지 않았다.

1. `corpus_inventory`: 전체 map/mode/perspective/rule/patch, 수집·결과 가용성, QA 사유.
2. `exposure_registry`: 모든 기존 경기의 split·모델 노출 이력과 잠금.
3. `final_team_outcomes`: 공식 결과·복원 순위·충돌·결측·출처.
4. `map_graph_registry`: 맵/버전별 grid·좌표 변환·이동 증거·불확실성·fit 범위.
5. `multidomain_windows`: 가변 팀/맵, 목표별 mask, observer/server paired 입력.
6. `ablation_manifest`: 공통 평가 경기, 제외 도메인, 데이터량·학습량 대조.

학습 실행은 여전히 중지 상태다. 먼저 이 산출물의 계약과 QA를 구현·검증한 뒤 학습 단계로 넘어간다.
