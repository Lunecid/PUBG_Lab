# PUBG 공식 Telemetry 기반 시공간 생존 성과 분석 보고서

## 1. 연구 주제

본 프로젝트의 최종 연구 주제는 다음과 같다.

> 공식 PUBG telemetry 기반 팀 이동 궤적과 동적 교전 네트워크가 생존 성과에 미치는 영향 분석

영문 제목은 다음과 같이 제안한다.

> Spatio-temporal analysis of PUBG squad survival performance using official telemetry trajectories and dynamic combat networks

이 연구는 PUBG squad match에서 각 팀을 시간에 따라 이동하는 시공간 객체로 보고, 팀의 이동 궤적, 자기장 상대 위치, 주변 적 밀도, 그리고 시간에 따라 형성되는 교전 네트워크 구조가 최종 순위와 생존 시간에 어떤 관계를 갖는지 분석한다.

## 2. 연구 목적

본 연구의 목적은 공식 PUBG telemetry에서 관측 가능한 시공간 데이터를 활용하여 팀의 생존 성과를 설명하고 예측하는 것이다.

핵심 목적은 세 가지다.

1. 팀의 이동 궤적과 자기장 상대 위치가 최종 순위 및 생존 시간과 관련되는지 분석한다.
2. 특정 시점 또는 phase에서 주변 적 팀 밀도와 근접성이 이후 성과와 관련되는지 분석한다.
3. 교전 이벤트로 구성한 동적 네트워크에서 팀의 위치가 생존 성과와 관련되는지 분석한다.

따라서 본 연구는 단순히 좌표를 사용하는 것이 아니라, `x, y, z, t`로 구성된 이동 객체의 궤적과 시간에 따라 누적되는 사건 관계를 함께 다루는 시공간 데이터 분석이다.

## 3. 사용 데이터 범위

본 연구는 공식 PUBG API 및 telemetry에서 제공되는 데이터만 사용한다.

| 데이터 축 | 사용 내용 |
| --- | --- |
| 위치 | 플레이어 및 팀 위치 좌표 `x, y, z` |
| 시간 | 이벤트 발생 시각, elapsed time, phase |
| 자기장 | safe zone, poison gas warning zone의 중심과 반경 |
| 팀 상태 | 생존 인원, 체력, 팀 ID |
| 교전 이벤트 | damage, attack, DBNO, kill 이벤트 |
| 이동/행동 | 차량 탑승, 아이템 사용, 회복 등 공식 이벤트 |
| 결과 | final rank, survival time, top-k placement |

사용하지 않는 데이터는 다음과 같다.

| 제외 항목 | 제외 이유 |
| --- | --- |
| 건물 내부/외부 여부 | 공식 telemetry에서 제공되지 않음 |
| 엄폐 여부 | 직접 관측 불가 |
| 개활지/건물/숲 같은 맵 의미 라벨 | 공식 구조 데이터가 아님 |
| 정확한 line-of-sight | 공식 API만으로 계산 불가 |
| 외부 height map 또는 비공식 map repo | 공식 데이터 범위를 벗어남 |

본 연구의 범위는 “정확한 맵 구조 해석”이 아니라 “공식 telemetry에서 관측 가능한 시공간 맥락 분석”이다.

## 4. 분석 단위와 결과 변수

기본 분석 단위는 `match-team`이다.

| 항목 | 정의 |
| --- | --- |
| 표본 단위 | 한 매치 안의 한 팀 |
| 관측 구간 | early 300s, phase별 구간, 또는 특정 snapshot 시점 |
| 주요 결과 변수 | final rank |
| 보조 결과 변수 | survival time, top-k placement, rank quartile |
| 데이터 분할 | match 단위 train/validation/test split |
| 누수 방지 | 관측 구간 이후의 이동, 교전, 생존 정보는 input feature에서 제외 |

결과 변수는 공식 match 결과에 기반하므로 별도의 휴리스틱 label 정의가 필요하지 않다.

## 5. 핵심 연구 질문

본 연구의 중심 질문은 다음과 같다.

> PUBG squad match에서 팀의 이동 궤적, 자기장 상대 위치, 주변 적 밀도, 동적 교전 네트워크 위치는 최종 순위와 생존 시간에 유의미한 정보를 제공하는가?

하위 연구 질문은 다음과 같다.

1. 팀의 이동 거리, 이동 효율, 방향 변화는 생존 성과와 관련되는가?
2. 자기장 중심 및 경계와의 상대적 위치는 최종 순위와 관련되는가?
3. 특정 phase에서 주변 적 팀 밀도와 nearest enemy distance는 생존 성과와 관련되는가?
4. 시간에 따라 누적되는 combat graph의 중심성, 커뮤니티 위치, 연결성은 최종 순위와 관련되는가?
5. early window 또는 특정 phase까지의 정보만으로도 top-k placement를 예측할 수 있는가?

## 6. 시공간 Feature 설계

### 6.1 Trajectory Features

팀의 이동 궤적을 요약하는 변수다.

| Feature | 의미 |
| --- | --- |
| `travel_distance` | 관측 구간 동안 팀 centroid가 이동한 총 거리 |
| `net_displacement` | 시작점과 끝점 사이 직선 거리 |
| `movement_efficiency` | `net_displacement / travel_distance` |
| `mean_speed` | 평균 이동 속도 |
| `max_speed` | 최대 이동 속도 |
| `direction_change_rate` | 이동 방향 변화 빈도 |
| `trajectory_stability` | 이동 방향의 안정성 |
| `squad_spread_mean` | 팀원 간 평균 산개도 |
| `squad_spread_max` | 관측 구간 내 최대 산개도 |

### 6.2 Zone-relative Features

팀 위치를 자기장 구조에 상대화한 변수다.

| Feature | 의미 |
| --- | --- |
| `zone_center_distance` | 안전지대 중심까지 거리 |
| `zone_edge_distance` | 안전지대 경계까지 거리 |
| `zone_pressure` | 안전지대/블루존 경계 기준 압박 정도 |
| `zone_approach_speed` | 안전지대 방향으로 접근하는 속도 |
| `time_outside_safe_zone` | 안전지대 밖에 있었던 시간 |

### 6.3 Local Spatial Context Features

공식 좌표와 이벤트만으로 계산 가능한 주변 상황 변수다.

| Feature | 의미 |
| --- | --- |
| `nearest_enemy_distance` | 가장 가까운 적 팀까지 거리 |
| `enemy_count_100m` | 반경 100m 내 적 팀 수 |
| `enemy_count_200m` | 반경 200m 내 적 팀 수 |
| `enemy_count_500m` | 반경 500m 내 적 팀 수 |
| `local_team_density` | 주변 팀 밀도 |
| `local_combat_density_recent` | 최근 주변 교전 이벤트 밀도 |

### 6.4 Dynamic Combat Graph Features

교전 이벤트를 시간에 따라 누적하여 만든 팀 간 네트워크 변수다.

| Feature | 의미 |
| --- | --- |
| `degree_until_t` | 관측 시점까지 교전한 상대 팀 수 |
| `weighted_degree_until_t` | 관측 시점까지 누적 교전 가중치 |
| `damage_dealt_until_t` | 관측 시점까지 가한 damage |
| `damage_taken_until_t` | 관측 시점까지 받은 damage |
| `betweenness_until_t` | 동적 교전 네트워크에서의 중개 중심성 |
| `pagerank_until_t` | 교전 네트워크상의 중요도 |
| `community_id_until_t` | Louvain community ID |
| `community_embeddedness_until_t` | 같은 커뮤니티 내부에 묶인 정도 |
| `component_size_until_t` | 속한 연결 성분의 크기 |

## 7. PPT 내부 데이터 활용 계획

본 프로젝트는 `/Users/lunecid/Documents/Spatio-Temporal` 경로의 PPT 자료를 방법론 근거로 활용한다.

| PPT 파일 | 활용 개념 | 본 연구 적용 |
| --- | --- | --- |
| `SD-Chap-2-Representation.ppt` | spatial object representation, point/line/polygon, raster/vector | 플레이어를 moving point object로, 팀을 centroid와 dispersion을 가진 aggregate object로 표현 |
| `SD-Chap-3-Distance.ppt` | Euclidean distance, distance measure | 팀 간 거리, zone 거리, nearest enemy distance, 이동 거리 정의 |
| `SD-Chap-4-Topology.ppt` | adjacency, connectivity, topological relation | 공식 교전 이벤트 기반 combat graph의 연결성, 중심성, community 구조 분석 |
| `SD-Chap-7-Spatial_Indexing.ppt` | grid, KD-tree, quadtree, R-tree | nearest enemy, 주변 팀 수, local density 계산을 위한 spatial query 근거 |
| `SD-Chap-8-BuildingData.ppt` | spatial database schema, metadata, integration | 위치 snapshot table, event table, team-state table, graph table 설계 근거 |
| `SD-Chap-9-SpatiotemporalData.ppt` | moving object, trajectory, valid time, snapshot | 팀의 `x, y, z, t` trajectory와 phase별 snapshot 분석의 핵심 근거 |
| `SD-Chap-9-SpatiotemporalData (1).ppt` | spatio-temporal query, trajectory analysis, team sports analysis | PUBG 팀 이동과 교전 이벤트를 시공간 질의 및 궤적 분석으로 재구성 |
| `SD-Chap-5-DTM.ppt` | terrain model, elevation | 공식 API만으로 terrain surface를 얻을 수 없으므로 한계점 설명에 활용 |
| `psychology.pptx` | topology/semantic relation 예시 | 의미 기반 공간 해석의 한계와 보조 설명에만 활용 |

핵심 근거는 `SD-Chap-9-SpatiotemporalData` 계열 자료이며, `Distance`, `Topology`, `Spatial_Indexing` 자료가 feature 계산과 분석 방법을 보조한다.

## 8. 분석 방법

본 연구는 설명 분석과 예측 분석을 분리한다.

### 8.1 설명 분석

팀별 시공간 feature와 생존 성과 사이의 관계를 통계적으로 분석한다.

사용 방법:

- 상관 분석
- partial correlation
- rank residual analysis
- Mann-Whitney U test
- regression analysis
- phase별 비교 분석

목표는 어떤 시공간 변수들이 final rank 또는 survival time과 안정적으로 연결되는지 확인하는 것이다.

### 8.2 예측 분석

관측 구간까지의 feature만 사용하여 최종 성과를 예측한다.

예측 target:

- final rank regression
- top-k placement classification
- rank quartile classification
- survival time prediction

사용 가능한 모델:

- logistic regression
- ridge/lasso regression
- random forest
- gradient boosting
- survival regression 또는 Cox model

평가 지표:

- MAE
- Spearman correlation
- AUC
- PR-AUC
- concordance index
- calibration curve

### 8.3 Ablation Study

시공간 feature의 기여도를 검증하기 위해 feature group을 단계적으로 추가한다.

| 실험 | 사용 feature |
| --- | --- |
| Baseline | phase, elapsed time, alive teams |
| +Trajectory | 이동 거리, 속도, 방향 변화, 산개도 |
| +Zone | zone 중심/경계 거리, zone pressure |
| +Local Density | nearest enemy, 주변 적 수, 주변 교전 밀도 |
| +Dynamic Graph | degree, weighted degree, betweenness, community position |

성능 향상이 확인되면, 공식 telemetry 기반 시공간 맥락이 생존 성과 분석에 실질적 정보를 제공한다고 주장할 수 있다.

## 9. 현재 구현과의 연결

현재 프로젝트의 기존 분석 자산은 다음과 같이 활용할 수 있다.

| 기존 구현 | 본 연구에서의 역할 |
| --- | --- |
| team-team damage graph | dynamic combat graph feature |
| phase별 분석 | phase-specific trajectory/context feature |
| movement/H3 분석 | trajectory feature의 기반 |
| strategy type 분석 | 해석용 aggregate 또는 보조 비교 |
| final rank 분석 | 공식 outcome 분석 |
| Louvain/community 분석 | graph topology feature |
| betweenness/PageRank 분석 | dynamic network position feature |

따라서 본 연구는 기존 구현을 폐기하는 것이 아니라, 기존 네트워크 분석을 시공간 moving object 분석과 결합하는 방향으로 확장한다.

## 10. 실현 가능성 평가

| 평가 항목 | 판단 | 근거 |
| --- | --- | --- |
| 데이터 확보성 | 높음 | 공식 telemetry와 현재 캐시가 위치, 시간, 이벤트, 순위 정보를 포함 |
| label 신뢰성 | 높음 | final rank, survival time, top-k는 공식 결과 기반 |
| 시공간 데이터 활용성 | 높음 | trajectory, snapshot, zone dynamics, local density를 직접 사용 |
| PPT 정합성 | 높음 | moving object, distance, topology, spatial indexing 개념과 직접 연결 |
| 구현 난이도 | 중간 | episode labeler는 필요 없지만 phase별 trajectory feature builder가 필요 |
| 연구 방어력 | 높음 | 공식 데이터와 명확한 outcome을 사용하므로 휴리스틱 공격에 강함 |
| 한계 명확성 | 높음 | 건물/엄폐/맵 구조를 주장하지 않아 범위가 명확함 |

종합적으로 본 주제는 현재 프로젝트 조건에서 실현 가능성이 높다. 특히 공식 telemetry만 사용한다는 제약과 Spatio-Temporal PPT 활용 조건을 동시에 만족한다.

## 11. 예상 한계

본 연구의 한계는 다음과 같다.

1. 공식 API만으로 건물, 엄폐, 실내/외 여부를 알 수 없다.
2. 좌표 기반 거리와 밀도는 실제 시야, 지형, 장애물을 완전히 반영하지 않는다.
3. final rank와 시공간 feature의 관계는 인과가 아니라 구조적 연관으로 해석해야 한다.
4. 관측 구간 설정에 따라 결과가 달라질 수 있으므로 early 300s, phase별 window 등 민감도 분석이 필요하다.
5. 생존 시간이 길수록 더 많은 이벤트가 관측되는 노출 시간 편향을 통제해야 한다.

이 한계는 연구 실패 요인이 아니라, 공식 데이터 기반 분석의 범위를 명확히 하는 조건이다.

## 12. 최종 정리

본 프로젝트는 PUBG의 공식 telemetry를 사용하여 팀을 시공간 이동 객체로 모델링하고, 팀의 trajectory, zone-relative context, local density, dynamic combat graph가 생존 성과와 어떤 관련을 갖는지 분석한다.

이 주제는 다음 조건을 만족한다.

- 공식 API 기반 데이터만 사용한다.
- 결과 변수는 공식 match outcome에 기반한다.
- `/Users/lunecid/Documents/Spatio-Temporal`의 PPT 내부 개념을 직접 활용한다.
- 단순 좌표 분석이 아니라 trajectory, snapshot, distance, topology, spatial indexing을 결합한 시공간 데이터 분석이다.
- 기존 PUBG 네트워크 분석 구현을 자연스럽게 확장할 수 있다.

따라서 본 프로젝트의 최종 방향은 다음 문장으로 요약할 수 있다.

> 공식 PUBG telemetry에서 관측 가능한 팀의 시공간 이동 패턴과 동적 교전 네트워크 구조를 이용하여 squad match의 생존 성과를 설명하고 예측한다.

