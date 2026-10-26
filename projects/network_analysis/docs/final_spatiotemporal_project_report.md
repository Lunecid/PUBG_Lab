# PUBG 공식 텔레메트리 기반 시공간 데이터베이스 모델링과 질의 기반 생존 패턴 분석

## 1. 연구 개요

이 연구는 PUBG squad match의 공식 telemetry를 시공간 데이터베이스 관점에서 재구성하고, 팀의 이동 궤적과 시공간 질의에서 추출한 특징이 생존 성과와 어떤 관련을 갖는지 분석하는 것을 목표로 한다. 분석 대상은 플레이어와 팀의 위치 변화, 전투 이벤트, 안전구역 변화, 생존 상태이며, 결과 변수는 최종 순위, 생존 시간, top-k placement로 설정한다.

연구의 핵심은 단순히 좌표를 feature로 사용하는 것이 아니라, 배틀로얄 매치를 다음 세 요소가 결합된 시공간 데이터로 모델링하는 데 있다.

| 구성 요소 | 시공간 데이터베이스 관점 | PUBG 적용 |
| --- | --- | --- |
| 플레이어/팀 위치 | moving object | 시간에 따라 이동하는 플레이어와 팀 centroid |
| 전투 이벤트 | timestamped spatial event | damage, attack, DBNO, kill 이벤트 |
| 안전구역 | dynamic spatial region | 시간에 따라 중심과 반경이 변하는 safe zone |

따라서 이 연구는 “게임 데이터 분석”이 아니라, trajectory, event, dynamic region이 결합된 시공간 데이터베이스 문제로 정의된다.

## 2. 최종 연구 주제

### 2.1 한국어 제목

**PUBG 공식 텔레메트리 기반 시공간 데이터베이스 모델링과 질의 기반 생존 패턴 분석**

### 2.2 영어 제목

**Spatio-Temporal Database Modeling and Query-Based Survival Pattern Analysis in PUBG**

### 2.3 핵심 연구 문장

PUBG 매치를 플레이어/팀 궤적, 전투 이벤트, 동적 안전구역으로 구성된 시공간 데이터베이스로 모델링하고, range query, nearest-neighbor query, trajectory query, encounter query, zone-distance query를 통해 추출한 시공간 특징이 팀 생존 패턴을 어떻게 설명하는지 분석한다.

## 3. 연구 배경과 필요성

배틀로얄 게임에서는 플레이어가 제한된 지도 안에서 이동하고, 시간이 지나면서 안전구역이 축소되며, 다른 팀과의 상호작용을 통해 생존 또는 탈락이 결정된다. 팀의 생존 성과는 단순히 킬 수나 총 데미지량만으로 설명되기 어렵다. 어느 시점에 어디에 있었는지, 안전구역과 얼마나 가까웠는지, 주변에 적 팀이 얼마나 있었는지, 전투 이벤트가 어떤 순서로 발생했는지가 함께 작용한다.

기존의 match-level summary 통계는 이러한 구조를 충분히 반영하지 못한다. 예를 들어 최종 이동 거리, 총 데미지, 킬 수 같은 값은 팀의 전체적인 활동량을 보여줄 수 있지만, 특정 phase에서 팀이 안전구역 안쪽에 있었는지, 주변 적 밀도가 높았는지, 전투 직전 어떤 이동 패턴을 보였는지는 설명하지 못한다.

이 문제를 해결하려면 원시 telemetry를 시간 순서가 있는 위치 데이터와 이벤트 데이터로 재구성해야 한다. 즉, 플레이어와 팀을 moving object로 표현하고, 전투 이벤트를 spatial event로 저장하며, 안전구역을 time-varying region으로 다루는 시공간 데이터베이스 접근이 필요하다.

## 4. 왜 시공간 데이터베이스 문제인가

이 연구가 시공간 데이터베이스 문제인 이유는 세 가지다.

첫째, 분석 대상이 시간에 따라 위치가 변하는 객체다. 각 플레이어의 위치는 `p_i(t) = (x_i(t), y_i(t), z_i(t))`로 표현할 수 있으며, 팀 위치는 생존한 팀원들의 centroid로 정의할 수 있다. 이는 전형적인 moving object data다.

둘째, 전투 이벤트는 특정 시간과 위치에서 발생한다. damage, attack, DBNO, kill 이벤트는 이벤트 유형, 발생 시각, 발생 위치, 행위자, 대상자를 가진다. 따라서 전투 로그는 timestamped spatial event table로 표현할 수 있다.

셋째, 안전구역은 시간에 따라 변하는 공간 영역이다. safe zone은 각 phase마다 중심과 반경이 바뀌므로 dynamic spatial region으로 볼 수 있다. 배틀로얄 게임에서는 이 동적 공간 영역이 플레이어의 이동과 생존에 직접 영향을 준다.

결국 PUBG telemetry는 단일 정적 테이블이 아니라, moving object, spatial event, dynamic region이 결합된 복합 시공간 데이터다. 따라서 range query, nearest-neighbor query, trajectory query, temporal window query, spatial indexing 같은 시공간 데이터베이스 개념을 적용할 수 있다.

## 5. 선행연구 관점과 적용 방향

Trajectory 연구는 일반적으로 원시 좌표를 바로 분석하지 않는다. 보통 다음 흐름을 따른다.

```text
raw trajectory
-> preprocessing / resampling
-> segmentation
-> semantic enrichment
-> spatio-temporal query
-> similarity / clustering / pattern mining
-> outcome analysis
```

이 흐름을 PUBG에 적용하면, 팀 trajectory는 단순 위치열이 아니라 phase, safe zone, 주변 적 밀도, 전투 이벤트가 결합된 semantic trajectory로 구성된다.

### 5.1 Moving object modeling

Moving object database 연구에서는 시간에 따라 위치가 변하는 객체를 데이터베이스 타입과 질의 대상으로 다룬다. PUBG에서는 플레이어가 moving point에 해당하고, 팀은 여러 플레이어 위치를 집계한 aggregate moving object로 볼 수 있다.

### 5.2 Trajectory preprocessing

원시 telemetry는 분석 가능한 형태로 정렬되어야 한다. 매치 시작 시간을 기준으로 timestamp를 정규화하고, 위치 정보를 10초 단위 team-time snapshot으로 변환한다. 이벤트는 원래 timestamp를 유지하되, feature extraction 단계에서 최근 30초, 60초, 300초 같은 temporal window에 맞춰 집계한다.

### 5.3 Trajectory segmentation

전체 match trajectory를 한 덩어리로 비교하면 phase별 차이가 사라진다. 따라서 trajectory는 다음 기준으로 나눈다.

| Segment | 목적 |
| --- | --- |
| early 300s | 초반 이동 전략 분석 |
| phase별 segment | 안전구역 변화에 따른 이동 비교 |
| rolling 60s window | 최근 이동 및 전투 상태 분석 |
| pre-elimination window | 탈락 직전 위험 패턴 분석 |

기본 분석은 early 300s와 phase별 segment를 중심으로 수행한다.

### 5.4 Semantic trajectory

Semantic trajectory 연구에서는 raw movement에 도메인 의미를 부여한다. 이 연구에서는 외부 맵 라벨을 사용하지 않고, 공식 telemetry에서 얻을 수 있는 정보만 사용한다.

| 의미 부여 항목 | 정의 |
| --- | --- |
| phase | phase 또는 `isGame` 상태 |
| safe zone relation | 안전구역 내부, 경계, 외부 |
| zone pressure | 안전구역 또는 블루존 경계와의 거리 |
| combat state | 최근 damage, attack, DBNO, kill 이벤트 |
| enemy proximity | nearest enemy distance, 반경 내 적 팀 수 |
| team state | 생존 인원, HP 합계, HP 평균 |
| mobility state | 정지, 이동, 회전, 안전구역 접근 |

이 방식은 건물, 엄폐, 개활지 같은 비공식 맵 구조를 사용하지 않으면서도 PUBG의 도메인 맥락을 trajectory에 반영할 수 있다.

### 5.5 Movement pattern discovery

Trajectory similarity 연구에서는 DTW, LCSS, Fréchet distance 같은 방법을 사용할 수 있지만, 프로젝트 범위 안에서는 feature-based movement archetype이 더 현실적이다. phase별 trajectory를 요약 feature vector로 변환한 뒤 clustering을 수행하고, 각 cluster의 final rank와 survival time을 비교한다.

예상 가능한 movement archetype은 다음과 같다.

| Movement archetype | 특징 |
| --- | --- |
| early center rotation | 초반부터 안전구역 중심 방향으로 빠르게 이동 |
| edge holding | 안전구역 경계 근처에 오래 머무름 |
| late rotation | 초반 이동이 적고 후반에 급격히 이동 |
| wide roaming | 이동 거리와 방향 변화가 큼 |
| compact slow play | 팀 산개도가 낮고 이동량이 작음 |
| high exposure path | 주변 적 밀도와 encounter duration이 큼 |

이 archetype을 최종 순위 및 생존 시간과 연결하면, 생존 성과와 관련된 이동 패턴을 해석할 수 있다.

## 6. 데이터 설명

### 6.1 데이터 출처

사용 데이터는 PUBG 공식 API 및 telemetry log다. 우선 분석 범위는 현재 프로젝트에 구축된 `Baltic_Main / squad` 데이터셋으로 둔다.

```text
/Users/lunecid/project/PUBG/network_analysis/output/analysis_datasets/Baltic_Main_squad_n1000
```

이 데이터셋은 1000개 매치와 약 26,000개 이상의 team sample을 포함한다. 필요하면 초기 분석에서는 일부 매치로 축소한 뒤, 최종 단계에서 전체 표본으로 확장한다.

### 6.2 주요 데이터 항목

| 데이터 축 | 내용 |
| --- | --- |
| 위치 | 플레이어 위치 `x, y, z` |
| 시간 | event timestamp, elapsed time, phase |
| 안전구역 | safe zone 중심, 반경, phase |
| 팀 상태 | team id, 생존 인원, HP |
| 전투 이벤트 | damage, attack, DBNO, kill |
| 결과 | final rank, survival time, top-k placement |

### 6.3 사용하지 않는 데이터

공식 telemetry만 사용하기 때문에 다음 항목은 분석에 포함하지 않는다.

| 제외 항목 | 이유 |
| --- | --- |
| 건물 내부/외부 여부 | 공식 telemetry에서 제공되지 않음 |
| 엄폐 여부 | 직접 관측 불가 |
| 개활지/숲/건물 같은 맵 의미 라벨 | 공식 구조 데이터가 아님 |
| 정확한 line-of-sight | 공식 API만으로 계산 불가 |
| 외부 height map 또는 비공식 map repo | 공식 데이터 범위를 벗어남 |

이 연구의 범위는 정확한 맵 구조 해석이 아니라, 공식 telemetry에서 관측 가능한 시공간 맥락 분석이다.

## 7. 시공간 데이터베이스 스키마

제안하는 주요 테이블은 다음과 같다.

```sql
matches(
    match_id,
    map_name,
    game_mode,
    duration,
    created_at
)

player_positions(
    match_id,
    player_id,
    team_id,
    timestamp,
    elapsed_time,
    phase,
    x,
    y,
    z,
    health,
    alive
)

team_snapshots(
    match_id,
    team_id,
    timestamp,
    elapsed_time,
    phase,
    centroid_x,
    centroid_y,
    centroid_z,
    alive_members,
    hp_sum,
    hp_mean,
    squad_spread
)

combat_events(
    match_id,
    timestamp,
    elapsed_time,
    event_type,
    attacker_id,
    attacker_team_id,
    victim_id,
    victim_team_id,
    x,
    y,
    z,
    damage,
    weapon
)

zone_states(
    match_id,
    timestamp,
    elapsed_time,
    phase,
    safe_center_x,
    safe_center_y,
    safe_radius,
    poison_center_x,
    poison_center_y,
    poison_radius
)

team_outcomes(
    match_id,
    team_id,
    final_rank,
    survival_time,
    top_k_placement,
    rank_quartile
)
```

`player_positions`와 `team_snapshots`는 moving object table, `combat_events`는 spatial event table, `zone_states`는 time-varying region table, `team_outcomes`는 결과 테이블에 해당한다.

## 8. 문제 정의

### 8.1 입력 데이터

입력은 세 가지로 정의한다.

```text
Player trajectories:
T_i = {(t_1, x_i1, y_i1, z_i1), ..., (t_n, x_in, y_in, z_in)}

Event logs:
E = {e_j = (tau_j, location_j, actor_j, target_j, type_j)}

Zone states:
Z(t) = (center_x(t), center_y(t), radius(t), phase(t))
```

### 8.2 분석 단위

기본 분석 단위는 10초 간격의 team-time snapshot이다. 각 snapshot에서 한 팀은 위치, 생존 인원, 체력, 안전구역 거리, 주변 적 수, 최근 전투 이벤트 수, 동적 교전 네트워크 위치를 가진 하나의 시공간 객체로 표현된다.

최종 성과 분석은 match-team 단위로 수행한다. 즉, team-time snapshot에서 추출한 feature를 phase 또는 window 단위로 집계한 뒤 final rank, survival time, top-k placement와 연결한다.

### 8.3 결과 변수

이 연구는 임의의 교전 승패 label을 만들지 않는다. 결과 변수는 공식 결과 또는 명확히 관측 가능한 값으로 제한한다.

| 결과 변수 | 정의 |
| --- | --- |
| final_rank | 팀 최종 순위 |
| survival_time | 팀 생존 시간 |
| top_k_placement | top 1, top 5, top 10 등 순위 그룹 |
| rank_quartile | 최종 순위 사분위 |

## 9. 연구 질문

연구 질문은 다음과 같다.

**RQ1.** 시간에 따라 축소되는 안전구역은 팀의 공간 분포와 이동 패턴을 어떻게 변화시키는가?

**RQ2.** 특정 시점의 주변 적 밀도, nearest enemy distance, 안전구역 경계 거리는 팀 생존 성과와 어떤 관계를 갖는가?

**RQ3.** phase별 team trajectory feature는 final rank 및 survival time과 관련되는가?

**RQ4.** phase별 movement archetype은 생존 성과의 차이를 설명하는가?

**RQ5.** 전투 이벤트로 구성한 동적 combat graph의 중심성, 연결성, community 위치는 팀 생존 성과와 관련되는가?

**RQ6.** 시공간 query-derived feature는 단순 match-level summary statistic보다 생존 결과를 더 잘 설명하거나 예측하는가?

## 10. 시공간 질의 설계

### 10.1 Range Query

특정 시점 `t`에서 팀 `g` 주변 반경 `r` 안에 있는 적 팀을 찾는다.

```text
Q_range(g, r, t) = {h | h != g and dist(P_g(t), P_h(t)) <= r}
```

이를 통해 `enemy_count_100m`, `enemy_count_200m`, `enemy_count_500m`를 계산한다.

### 10.2 k-Nearest Neighbor Query

특정 시점에서 팀 `g`와 가장 가까운 적 팀을 찾는다.

```text
Q_knn(g, k, t) = top-k enemy teams h by dist(P_g(t), P_h(t))
```

주요 feature는 `nearest_enemy_distance`다.

### 10.3 Trajectory Query

특정 시간 구간 `[t_a, t_b]` 동안 팀의 이동 경로를 추출한다.

```text
Q_traj(g, t_a, t_b) = {P_g(t) | t_a <= t <= t_b}
```

이를 통해 이동 거리, 순변위, 이동 효율, 평균 속도, 방향 변화, 팀 산개도를 계산한다.

### 10.4 Encounter Query

두 팀이 일정 시간 이상 가까운 거리에 있었는지를 계산한다.

```text
Encounter(g, h, [t_a, t_b]) = 1
if sum I[dist(P_g(t), P_h(t)) <= r] >= m
```

이 질의는 교전 승패 label을 만들기 위한 것이 아니라, 주변 적 노출 정도와 근접 상황을 측정하는 feature로 사용한다.

### 10.5 Zone-Distance Query

각 팀이 안전구역 중심 또는 경계로부터 얼마나 떨어져 있는지 계산한다.

```text
d_zone(g, t) = dist(P_g(t), center(t)) - radius(t)
```

`d_zone < 0`이면 안전구역 내부, `d_zone = 0`이면 경계, `d_zone > 0`이면 안전구역 외부로 해석한다.

### 10.6 Event Window Query

특정 시점 이전 `w`초 동안 팀 주변에서 발생한 이벤트를 집계한다.

```text
Q_event(g, r, w, t) =
{e_j | t-w <= tau_j <= t and dist(P_g(t), location_j) <= r}
```

이를 통해 최근 주변 교전 밀도, 최근 damage, 최근 kill/DBNO, 마지막 교전 이후 시간을 계산한다.

## 11. Feature 설계

| Feature group | 주요 변수 |
| --- | --- |
| Geometry | path length, displacement, tortuosity, speed, acceleration |
| Direction | heading change, angular variance, zone approach angle |
| Team shape | squad spread mean/max, split duration |
| Zone-relative | center distance, edge distance, outside time, approach speed |
| Interaction | nearest enemy distance, enemy count in radius, encounter duration |
| Event-enriched | recent damage, recent DBNO/kill, time since combat |
| Graph-enriched | degree, weighted degree, betweenness, PageRank, community position |

특히 safe zone이 매치마다 다르게 움직이므로 절대 좌표와 zone-relative 좌표를 함께 사용한다.

```text
relative_zone_x = (team_x - safe_center_x) / safe_radius
relative_zone_y = (team_y - safe_center_y) / safe_radius
zone_edge_distance = dist(team, safe_center) - safe_radius
```

이 방식은 외부 맵 의미 라벨 없이도 배틀로얄의 핵심 공간 구조를 반영한다.

## 12. 분석 방법

### 12.1 설명 분석

시공간 feature와 생존 성과 사이의 관계를 통계적으로 확인한다.

사용 방법은 다음과 같다.

- Pearson/Spearman correlation
- Mann-Whitney U test
- partial correlation
- regression analysis
- rank residual analysis
- phase별 비교 분석

### 12.2 Movement archetype 분석

phase별 trajectory feature를 이용해 팀 이동 패턴을 군집화한다. 각 cluster는 movement archetype으로 해석하고, cluster별 final rank, survival time, top-k placement 비율을 비교한다.

평가 항목은 다음과 같다.

- cluster별 평균 final rank
- cluster별 survival time 분포
- cluster별 top-k placement 비율
- silhouette score 또는 cluster stability
- trajectory-only, zone-enriched, event-enriched feature 비교

### 12.3 동적 교전 네트워크 분석

각 시점 `t`까지 발생한 팀 간 damage event를 누적하여 그래프를 구성한다.

```text
G_t = (V_t, E_t, W_t)
```

여기서 `V_t`는 팀, `E_t`는 팀 간 교전 관계, `W_t`는 누적 damage weight다.

계산할 graph feature는 다음과 같다.

- degree
- weighted degree
- damage dealt / taken
- betweenness
- PageRank
- community embeddedness
- component size

이 feature는 trajectory feature와 결합하여 생존 성과 분석에 사용한다.

### 12.4 예측 분석

예측은 보조 실험으로 수행한다. 모델의 복잡도보다 feature group의 기여도를 확인하는 데 목적을 둔다.

예측 task는 다음과 같다.

- final rank regression
- top-k placement classification
- rank quartile classification
- 선택적으로 다음 `Delta`초 내 탈락 여부 예측

사용 모델은 다음 수준으로 제한한다.

- logistic regression
- ridge/lasso regression
- random forest
- gradient boosting

평가 지표는 MAE, Spearman correlation, ROC-AUC, PR-AUC, F1-score, Brier score를 사용한다.

## 13. 실험 계획

### 13.1 질의 성능 평가

시공간 데이터베이스 프로젝트로서 query execution time도 평가한다.

| 비교 대상 | 목적 |
| --- | --- |
| naive pairwise distance | index 없는 기준 |
| KD-tree 또는 BallTree | point snapshot 기반 근접 질의 |
| grid index | 빠른 range query |

range query와 nearest-neighbor query의 실행 시간을 데이터 크기 변화에 따라 비교하고, spatial index 사용 전후의 차이를 확인한다.

### 13.2 생존 패턴 분석 평가

| 연구 질문 | 평가 방식 |
| --- | --- |
| 안전구역 축소와 공간 분포 | phase별 density heatmap, spatial entropy |
| 주변 적 밀도와 생존 성과 | eliminated/survived group의 proximity distribution 비교 |
| trajectory와 성과 | 이동 거리, 이동 효율, 산개도와 rank/survival time 상관 |
| movement archetype과 성과 | cluster별 final rank, survival time, top-k 비율 비교 |
| dynamic graph와 성과 | centrality/community feature와 final rank 분석 |

### 13.3 Baseline 비교

| 실험 | 사용 feature |
| --- | --- |
| Baseline 1 | match-level summary statistics |
| Baseline 2 | temporal event counts |
| Proposed 1 | trajectory query features |
| Proposed 2 | zone-distance query features |
| Proposed 3 | range/kNN local context features |
| Proposed 4 | dynamic combat graph features |

이 ablation을 통해 시공간 질의 기반 feature가 실제로 설명력 또는 예측력을 개선하는지 확인한다.

## 14. PPT 자료 활용 계획

`/Users/lunecid/Documents/Spatio-Temporal` 경로의 PPT 자료는 다음 방식으로 활용한다.

| PPT 파일 | 활용 개념 | 적용 |
| --- | --- | --- |
| `SD-Chap-2-Representation.ppt` | spatial object representation | 플레이어를 point object, 팀을 aggregate object로 표현 |
| `SD-Chap-3-Distance.ppt` | distance measure | 팀 간 거리, zone 거리, nearest enemy distance |
| `SD-Chap-4-Topology.ppt` | adjacency, connectivity | dynamic combat graph 분석 |
| `SD-Chap-7-Spatial_Indexing.ppt` | grid, KD-tree, quadtree, R-tree | range/kNN query 처리 |
| `SD-Chap-8-BuildingData.ppt` | spatial database schema | trajectory/event/zone table 설계 |
| `SD-Chap-9-SpatiotemporalData.ppt` | moving object, trajectory, snapshot | 팀 trajectory와 team-time snapshot 분석 |
| `SD-Chap-9-SpatiotemporalData (1).ppt` | spatio-temporal query, trajectory analysis | 질의 기반 feature extraction |
| `SD-Chap-5-DTM.ppt` | terrain model | 공식 API 한계 설명 |
| `psychology.pptx` | topology/semantic relation | 의미 기반 공간 해석의 한계 설명 |

핵심 근거는 `SD-Chap-9-SpatiotemporalData` 계열 자료이며, `Distance`, `Topology`, `Spatial_Indexing` 자료가 feature 계산과 질의 설계를 보조한다.

## 15. 실현 가능성

| 평가 항목 | 판단 | 근거 |
| --- | --- | --- |
| 데이터 확보성 | 높음 | 공식 telemetry와 기존 캐시가 위치, 시간, 이벤트, 순위 정보를 포함 |
| label 신뢰성 | 높음 | final rank, survival time, top-k는 공식 결과 기반 |
| 시공간성 | 높음 | trajectory, snapshot, zone dynamics, local density를 직접 사용 |
| 선행연구 정합성 | 높음 | moving object, semantic trajectory, trajectory mining 흐름과 연결 |
| 구현 난이도 | 중간 | trajectory feature builder와 query 모듈 구현 필요 |
| 연구 방어력 | 높음 | 비공식 맵 구조나 휴리스틱 교전 승패 label에 의존하지 않음 |

프로젝트 기간 안에 실현 가능한 최소 범위는 다음과 같다.

1. 일부 `Baltic_Main / squad` 매치에서 10초 team-time snapshot 생성
2. range, kNN, trajectory, zone-distance query 구현
3. phase별 trajectory feature 추출
4. final rank 또는 top-k placement와의 관계 분석
5. 간단한 baseline 비교와 시각화

전체 범위가 가능하면 movement archetype clustering과 dynamic combat graph feature까지 확장한다.

## 16. 일정 계획

| 기간 | 작업 |
| --- | --- |
| 1주차 | 데이터 구조 확인, 공식 telemetry field 정리, 스키마 설계 |
| 2주차 | 위치 snapshot 정규화, team centroid/health/alive table 생성 |
| 3주차 | range query, kNN query, trajectory query, zone-distance query 구현 |
| 4주차 | trajectory feature 추출, phase별 분포 및 local density 분석 |
| 5주차 | movement archetype clustering, baseline 비교, ablation study |
| 6주차 | 결과 시각화, 보고서 작성, 발표 자료 구성 |

## 17. 한계와 리스크 관리

### 17.1 맵 의미 정보의 부재

공식 telemetry만으로 건물 내부/외부, 엄폐, 개활지, line-of-sight를 알 수 없다. 따라서 이 연구는 맵 구조를 직접 해석한다고 주장하지 않는다. 분석 범위는 공식 좌표, 이벤트, 안전구역에서 관측 가능한 시공간 맥락으로 제한한다.

### 17.2 위치 샘플링 문제

위치 샘플이 모든 순간을 연속적으로 기록하지 않을 수 있다. 이를 해결하기 위해 10초 간격 snapshot으로 resampling하고, 짧은 결측은 nearest previous state 방식으로 처리한다. 긴 결측은 missing으로 표시하거나 분석에서 제외한다.

### 17.3 생존 시간 편향

오래 생존한 팀은 더 많은 이벤트를 관측할 기회를 가진다. 따라서 전체 누적값만 사용하지 않고, early 300s, phase-specific window, per-minute rate, survival-time control을 함께 사용한다.

### 17.4 프로젝트 범위

이 연구는 복잡한 딥러닝 모델 개발보다 시공간 데이터베이스 설계, 질의 정의, feature extraction, 해석 가능한 분석에 초점을 둔다. 예측 모델은 baseline 비교를 위한 보조 실험으로 제한한다.

### 17.5 label 신뢰성

교전 승패처럼 정의가 모호한 label은 주 target으로 사용하지 않는다. final rank, survival time, top-k placement처럼 공식 결과 또는 명확히 관측 가능한 결과를 사용한다.

## 18. 기대 기여

이 연구의 기대 기여는 다음과 같다.

1. PUBG telemetry를 moving object, spatial event, dynamic region으로 구성된 시공간 데이터베이스 스키마로 설계한다.
2. range, kNN, trajectory, encounter, zone-distance, event-window query를 정의하고 생존 패턴 분석 feature로 변환한다.
3. 팀 trajectory를 phase와 safe zone 기준으로 분절하고, combat event와 enemy proximity를 결합한 semantic trajectory로 재구성한다.
4. phase별 movement archetype을 도출하고, 각 archetype의 final rank, survival time, top-k placement 차이를 분석한다.
5. 단순 match-level summary statistic과 시공간 query-derived feature를 비교하여 시공간 데이터베이스 접근의 유용성을 평가한다.

## 19. 참고문헌

1. PUBG Developer Documentation. Telemetry. https://documentation.pubg.com/en/telemetry.html
2. PUBG Developer Documentation. Telemetry Events. https://documentation.pubg.com/en/telemetry-events.html
3. PUBG Developer Documentation. Telemetry Objects. https://documentation.pubg.com/en/telemetry-objects.html
4. Güting, R. H., Böhlen, M. H., Erwig, M., Jensen, C. S., Lorentzos, N. A., Schneider, M., & Vazirgiannis, M. (2000). A Foundation for Representing and Querying Moving Objects. ACM Transactions on Database Systems, 25(1), 1-42. https://doi.org/10.1145/352958.352963
5. Zheng, Y. (2015). Trajectory Data Mining: An Overview. ACM Transactions on Intelligent Systems and Technology, 6(3), 1-41. https://doi.org/10.1145/2743025
6. Lee, J. G., Han, J., & Whang, K. Y. (2007). Trajectory Clustering: A Partition-and-Group Framework. SIGMOD 2007, 593-604. https://doi.org/10.1145/1247480.1247546
7. Giannotti, F., Nanni, M., Pinelli, F., & Pedreschi, D. (2007). Trajectory Pattern Mining. KDD 2007. https://doi.org/10.1145/1281192.1281230
8. Spaccapietra, S., Parent, C., Damiani, M. L., de Macedo, J. A., Porto, F., & Vangenot, C. (2008). A Conceptual View on Trajectories. Data & Knowledge Engineering, 65(1), 126-146. https://doi.org/10.1016/j.datak.2007.10.008
9. Gudmundsson, J., & Horton, M. (2017). Spatio-Temporal Analysis of Team Sports. ACM Computing Surveys, 50(2), 1-34. https://doi.org/10.1145/3054132
10. `/Users/lunecid/Documents/Spatio-Temporal/SD-Chap-2-Representation.ppt`
11. `/Users/lunecid/Documents/Spatio-Temporal/SD-Chap-3-Distance.ppt`
12. `/Users/lunecid/Documents/Spatio-Temporal/SD-Chap-4-Topology.ppt`
13. `/Users/lunecid/Documents/Spatio-Temporal/SD-Chap-7-Spatial_Indexing.ppt`
14. `/Users/lunecid/Documents/Spatio-Temporal/SD-Chap-8-BuildingData.ppt`
15. `/Users/lunecid/Documents/Spatio-Temporal/SD-Chap-9-SpatiotemporalData.ppt`
16. `/Users/lunecid/Documents/Spatio-Temporal/SD-Chap-9-SpatiotemporalData (1).ppt`

## 20. 최종 요약

이 연구는 PUBG 공식 telemetry를 시공간 데이터베이스로 모델링하고, 팀 trajectory와 전투 이벤트, 안전구역 변화를 결합해 생존 패턴을 분석한다. 플레이어와 팀은 moving object, 전투 이벤트는 timestamped spatial event, 안전구역은 dynamic spatial region으로 정의한다. 이후 range query, nearest-neighbor query, trajectory query, encounter query, zone-distance query를 통해 주변 적 밀도, 안전구역 압박, 이동 패턴, 최근 전투 상태를 추출한다.

Trajectory 선행연구의 흐름에 맞춰 팀 이동 경로는 phase별로 분절하고, safe zone과 combat event를 이용해 semantic trajectory로 확장한다. 최종적으로 phase별 movement archetype과 동적 combat graph feature가 final rank, survival time, top-k placement와 어떤 관계를 갖는지 분석한다.

이 접근은 공식 데이터만 사용하고, 임의의 교전 승패 label이나 비공식 맵 구조에 의존하지 않는다. 따라서 수업 프로젝트 기간 안에서 실현 가능하며, 시공간 데이터베이스 과목의 핵심 개념인 moving object, trajectory, spatial event, dynamic region, spatio-temporal query, spatial indexing을 실제 데이터에 적용하는 사례로 정리할 수 있다.

