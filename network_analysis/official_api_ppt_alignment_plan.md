# 공식 PUBG API + Spatio-Temporal PPT 기반 정식 연구 주제

## 결론

교전 episode 승패 예측은 본 프로젝트의 주제로 추천하지 않는다. 이유는 교전의 시작/종료, 다팀 난전, 승패 판정 기준이 모두 operational definition에 크게 의존하기 때문이다. 모델을 만들 수는 있지만, 발표나 보고서에서 "정답 label이 휴리스틱 아니냐"는 공격을 받기 쉽다.

따라서 본 프로젝트는 공식 PUBG API/telemetry에서 직접 관측되는 결과값인 `final rank`, `survival time`, `top-k placement`를 outcome으로 삼고, `/Users/lunecid/Documents/Spatio-Temporal` 경로의 PPT 내부 개념을 시공간 feature와 분석 방법론의 근거로 사용하는 방향이 더 정식적이다.

추천 주제는 다음과 같다.

> 공식 PUBG telemetry 기반 팀 이동 궤적과 동적 교전 네트워크가 생존 성과에 미치는 영향 분석

영문 제목 후보:

> Spatio-temporal analysis of PUBG squad survival performance using official telemetry trajectories and dynamic combat networks

이 주제는 교전 승패를 임의로 자르지 않는다. 대신 플레이어/팀을 시간에 따라 움직이는 객체로 보고, phase별 trajectory, zone-relative position, local enemy density, dynamic combat graph position이 공식 결과값인 최종 순위/생존 시간과 어떤 관계를 갖는지 분석한다.

## 왜 이 주제가 더 방어 가능한가

| 항목 | 교전 승패 예측 | 추천 주제 |
| --- | --- | --- |
| 분석 단위 | 임의로 자른 engagement episode | 공식 match/team 단위 |
| outcome | 교전 승패 label을 직접 정의해야 함 | 공식/명확한 final rank, survival time |
| 가장 큰 리스크 | episode boundary와 winner 판정 | 인과가 아닌 연관 분석이라는 한계 |
| PPT 정합성 | 가능하지만 event segmentation 중심 | moving object, trajectory, snapshot, distance, topology와 직접 정합 |
| 현재 구현 재사용성 | 새 labeler 필요 | 기존 network/phase/team 분석을 바로 확장 |
| 발표 방어력 | 중간 | 높음 |

핵심 차이는 "누가 특정 교전을 이겼는가"를 묻지 않고, "공식 telemetry에서 관측되는 시공간 행동 구조가 최종 생존 성과와 연결되는가"를 묻는다는 점이다.

## 연구 질문

주 연구 질문:

> PUBG squad match에서 팀의 zone-relative trajectory, 주변 적 밀도, 동적 교전 네트워크 위치는 최종 순위와 생존 시간에 유의미한 정보를 제공하는가?

하위 질문:

1. 팀의 이동 궤적과 자기장 상대 위치는 최종 순위/생존 시간과 관련되는가?
2. 같은 phase에서 주변 적 팀 밀도와 nearest enemy distance는 이후 생존 성과와 관련되는가?
3. 팀의 산개도와 이동 안정성은 성과와 관련되는가?
4. 시간에 따라 누적되는 combat graph의 degree, weighted degree, betweenness, community position은 최종 순위와 관련되는가?
5. early window 또는 phase별 feature만 사용해도 최종 rank group/top-k를 예측할 수 있는가?

## 공식 API로 사용할 수 있는 데이터

공식 telemetry 기준으로 사용 가능한 축은 다음과 같다.

| 축 | 공식 telemetry 근거 | 분석 변수 예시 |
| --- | --- | --- |
| 위치 | `Character.location`, `Location(x, y, z)` | 팀 중심 좌표, 팀 산개도, 이동 거리, 이동 속도 |
| 시간 | 이벤트 timestamp `_D`, `elapsedTime`, `common.isGame`, `LogPhaseChange` | phase, snapshot time, fixed observation window |
| 자기장 | `GameState.safetyZonePosition`, `safetyZoneRadius`, `poisonGasWarningPosition`, `poisonGasWarningRadius` | zone 중심 거리, zone edge distance, zone pressure |
| 생존 상태 | `numAliveTeams`, `numAlivePlayers`, `Character.health`, 팀원 생존 수 | alive members, HP summary, match pressure |
| 교전 | `LogPlayerTakeDamage`, `LogPlayerAttack`, `LogPlayerMakeGroggy`, `LogPlayerKillV2` | 누적 damage, 최근 damage, combat graph edge |
| 장비/행동 | `Item`, `Vehicle`, `LogHeal`, `LogItemUse`, `LogVehicleRide/Leave` | 차량 이동 여부, 회복/아이템 사용, 무기군 보조 변수 |
| 결과 | `GameResult`, team/player rank, death/survival event | final rank, survival time, top-k placement |

공식 `api-assets`는 무기/아이템/맵 이미지/딕셔너리 같은 보조 자산으로만 사용한다. 건물 polygon, cover, 실내/외 구분, 정확한 line-of-sight는 공식 데이터에 없으므로 feature로 사용하지 않는다.

## PPT 내부 데이터 활용 방식

| PPT | 내부 개념 | 프로젝트 적용 |
| --- | --- | --- |
| `SD-Chap-2-Representation.ppt` | raster/vector, point/line/polygon, spatial object representation, OGC Simple Feature | 플레이어를 moving point object로, 팀을 centroid와 dispersion을 가진 aggregate object로 표현한다. |
| `SD-Chap-3-Distance.ppt` | Euclidean distance, similarity/distance measure | 팀 간 거리, nearest enemy distance, zone 중심/경계 거리, trajectory displacement를 정의한다. 도로/건물 기반 constrained distance는 제외한다. |
| `SD-Chap-4-Topology.ppt` | topological relation, connectivity, adjacency | 공간 위상 자체가 아니라 공식 교전 이벤트로 구성한 team-team combat graph의 adjacency, connectivity, community topology를 사용한다. |
| `SD-Chap-7-Spatial_Indexing.ppt` | grid, KD-tree, quadtree, R-tree, spatial query | 각 snapshot에서 반경 내 적 팀 수, nearest enemy, local combat density를 효율적으로 계산하는 방법론 근거로 사용한다. |
| `SD-Chap-8-BuildingData.ppt` | spatial database schema, metadata, data integration, generalization | 위치 snapshot, team-state table, event table, phase table, graph table의 schema 설계 근거로 사용한다. 실제 building data는 사용하지 않는다. |
| `SD-Chap-9-SpatiotemporalData.ppt`, `SD-Chap-9-SpatiotemporalData (1).ppt` | moving object, trajectory, valid time, snapshot, spatio-temporal query, team sports analysis | 핵심 이론 근거다. PUBG 팀의 `x, y, z, t` trajectory와 event sequence를 phase별 snapshot/trajectory로 분석한다. |
| `SD-Chap-5-DTM.ppt` | terrain model, TIN, elevation | 공식 API만으로 terrain surface를 얻을 수 없으므로 feature에서는 제외하고 limitation으로 둔다. |
| `psychology.pptx` | topology/semantic relation 예시 | 직접 feature보다는 의미 기반 위치 해석의 한계를 설명하는 보조 근거로만 사용한다. |

## 분석 단위와 outcome

교전 episode가 아니라 match-team 단위를 기본 표본으로 둔다.

| 항목 | 정의 |
| --- | --- |
| sample | 한 매치 안의 한 팀 |
| observation window | early 300s, phase 1/2/3, 또는 특정 `isGame` snapshot까지 |
| outcome 1 | final rank |
| outcome 2 | survival time |
| outcome 3 | top-k placement 또는 rank quartile |
| split | match 단위 train/validation/test split |
| leakage 방지 | 관측 window 이후의 damage, kill, movement, survival 정보는 input에서 제외 |

이렇게 잡으면 label이 공식 outcome에 기반하므로 휴리스틱성이 크게 줄어든다.

## 시공간 feature set

### 1. Trajectory features

- `travel_distance`: 관측 window 내 팀 centroid 이동 거리
- `net_displacement`: 시작점과 끝점 사이 직선 거리
- `movement_efficiency`: `net_displacement / travel_distance`
- `mean_speed`, `max_speed`
- `direction_change_rate`
- `trajectory_stability`: 이동 방향 변동성
- `squad_spread_mean`, `squad_spread_max`

### 2. Zone-relative features

- `zone_center_distance`
- `zone_edge_distance`
- `zone_pressure`: 안전지대/블루존 경계 기준 압박 정도
- `zone_approach_speed`: 안전지대 중심 또는 경계 방향으로 접근하는 속도
- `time_outside_safe_zone`

### 3. Local spatial context features

- `nearest_enemy_distance`
- `enemy_count_100m`, `enemy_count_200m`, `enemy_count_500m`
- `local_team_density`
- `local_combat_density_recent`

이들은 건물/엄폐 같은 의미 라벨이 아니라, 공식 좌표와 이벤트만으로 계산되는 관측 가능한 spatial context다.

### 4. Dynamic combat graph features

- `degree_until_t`: 관측 window까지 교전한 상대 팀 수
- `weighted_degree_until_t`: 관측 window까지 주고받은 damage 가중치
- `damage_dealt_until_t`, `damage_taken_until_t`
- `betweenness_until_t`
- `pagerank_until_t`
- `community_id_until_t`
- `community_embeddedness_until_t`
- `component_size_until_t`

기존 `network_analysis`의 graph/community/centrality 분석을 이 축으로 재사용할 수 있다.

## 모델과 평가

분석은 예측과 설명을 분리한다.

| 목적 | 방법 | 평가 |
| --- | --- | --- |
| 설명 분석 | 상관, partial correlation, rank residual, Mann-Whitney U, regression | effect size, p-value, confidence interval |
| 순위 예측 | ordinal/logistic/ridge regression, random forest, gradient boosting | MAE, Spearman correlation, top-k AUC |
| 생존 분석 | Cox model 또는 survival regression | concordance index, time-dependent AUC |
| feature contribution | ablation: baseline, +trajectory, +zone, +local density, +graph | 성능 변화량 |

baseline은 `phase`, `alive teams`, `elapsed time`처럼 최소 정보만 사용하고, 이후 PPT 기반 시공간 feature를 단계적으로 추가한다.

## 현재 구현과의 연결

현재 `network_analysis` 구현은 이미 다음 기반을 갖고 있다.

| 현재 구현 | 새 주제에서의 역할 |
| --- | --- |
| team-team damage graph | dynamic combat graph feature |
| phase/H3 movement 분석 | phase별 trajectory/mobility feature |
| strategy type 분석 | 해석용 aggregate 또는 보조 비교 |
| final rank 분석 | 공식 outcome |
| community/betweenness 분석 | topological graph feature |

따라서 완전히 새 프로젝트가 아니라, 기존 분석을 "시공간 moving object + dynamic graph" 관점으로 재구성하는 작업이다.

## 피해야 할 표현

- "교전 승패를 정확히 예측했다" -> 교전 label이 휴리스틱이므로 주제로 삼지 않는다.
- "건물 안/밖을 고려했다" -> 공식 API에 없다.
- "엄폐 상태를 반영했다" -> 직접 관측 불가.
- "정확한 맵 구조를 모델링했다" -> 공식 맵 이미지는 있어도 구조 라벨은 없다.
- "지형 고저차를 반영했다" -> 공식 API 위치 `z`는 사용할 수 있지만, terrain surface를 구축했다고 말하면 안 된다.

대신 다음 표현을 사용한다.

- "공식 telemetry 기반 팀 이동 궤적"
- "zone-relative spatio-temporal context"
- "동적 combat graph topology"
- "moving object trajectory와 event sequence 기반 생존 성과 분석"
- "PPT의 moving object, distance, topology, spatial indexing 개념을 PUBG telemetry에 적용"

## 실현 가능성 평가

| 평가 항목 | 판단 | 이유 |
| --- | --- | --- |
| 데이터 확보성 | 높음 | 공식 telemetry와 현재 캐시가 이미 위치/시간/이벤트/순위 정보를 포함한다. |
| label 신뢰성 | 높음 | final rank, survival time은 match/team 결과 기반이다. |
| 시공간성 | 높음 | `x, y, z, t` trajectory, zone dynamics, local density, time-window graph를 직접 사용한다. |
| PPT 정합성 | 높음 | 특히 `SpatiotemporalData`, `Distance`, `Topology`, `Spatial_Indexing`과 자연스럽게 연결된다. |
| 구현 난이도 | 중간 | episode labeler는 필요 없지만, phase별 snapshot/trajectory feature builder가 필요하다. |
| 연구 방어력 | 높음 | 공식 데이터와 명확한 outcome에 기반하므로 휴리스틱 공격에 강하다. |

## 권장 구현 순서

1. match-team 단위의 공식 outcome table을 확정한다: `final_rank`, `survival_time`, `top_k`.
2. phase 또는 early 300s 기준 observation window를 정의한다.
3. 팀별 moving object trajectory table을 만든다: `match_id`, `team_id`, `t`, `x`, `y`, `z`, `phase`.
4. trajectory feature를 계산한다.
5. zone-relative feature를 계산한다.
6. spatial index를 이용해 nearest enemy와 local enemy density를 계산한다.
7. 관측 window까지의 dynamic combat graph feature를 계산한다.
8. 설명 분석과 예측 분석을 분리해 수행한다.
9. PPT 파일별 개념이 어떤 feature/방법론으로 들어갔는지 결과 보고서에 명시한다.

