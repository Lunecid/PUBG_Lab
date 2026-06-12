# Spatio-Temporal Database Modeling and Query-Based Survival Pattern Analysis in PUBG

## 1. Title

**Spatio-Temporal Database Modeling and Query-Based Survival Pattern Analysis in PUBG**

한국어 제목:

**PUBG 공식 텔레메트리 기반 시공간 데이터베이스 모델링과 질의 기반 생존 패턴 분석**

## 2. Abstract / Project Summary

온라인 배틀로얄 게임은 플레이어 위치, 전투 이벤트, 안전구역 변화, 생존 상태가 시간에 따라 함께 변화하는 대표적인 시공간 데이터 환경이다. 본 프로젝트는 PUBG 공식 telemetry를 플레이어/팀 궤적, timestamped combat event, dynamic safe zone으로 구성된 시공간 데이터베이스로 모델링하고, 시공간 질의를 통해 팀 생존 성과와 관련된 해석 가능한 특징을 추출하는 것을 목표로 한다. 플레이어와 팀은 시간에 따라 위치가 변하는 moving object로 표현하고, damage/kill/DBNO 이벤트는 특정 시간과 위치에서 발생한 spatial event로 표현하며, 안전구역은 시간에 따라 중심과 반경이 바뀌는 dynamic spatial region으로 표현한다. 이후 range query, nearest-neighbor query, trajectory query, encounter query, zone-distance query를 정의하여 주변 적 밀도, 안전구역 거리, 이동 안정성, 최근 교전 밀도, 동적 교전 네트워크 위치를 계산한다. 최종적으로 이러한 시공간 query-derived feature가 final rank, survival time, top-k placement와 어떤 관계를 갖는지 분석하고, 단순 match-level summary statistic보다 생존 패턴을 더 잘 설명하는지 평가한다. 본 프로젝트는 복잡한 딥러닝 모델 개발보다 시공간 데이터베이스 스키마, 질의 정의, feature extraction, 해석 가능한 생존 패턴 분석에 초점을 둔다.

## 3. Introduction

### 3.1 Background

시공간 데이터는 단순히 위치 좌표를 포함하는 데이터가 아니라, 시간에 따라 변화하는 객체의 위치, 상태, 상호작용을 함께 다루는 데이터다. 대표적인 예로 차량 GPS trajectory, 택시/배달 이동 기록, 감염병 확산 경로, 스포츠 선수 위치 데이터, 온라인 게임 플레이어 로그가 있다.

PUBG와 같은 배틀로얄 게임은 시공간 데이터베이스 관점에서 특히 적합한 분석 대상이다. 한 매치 안에서 플레이어는 넓은 지도 위를 이동하고, 시간이 지남에 따라 안전구역은 축소되며, 전투 이벤트는 특정 시간과 위치에서 발생한다. 또한 최종 순위와 생존 시간이라는 명확한 결과 변수가 존재한다. 즉 PUBG telemetry는 moving object, timestamped spatial event, dynamic spatial region이 결합된 복합 시공간 데이터다.

### 3.2 Motivation

기존 게임 분석은 대체로 킬 수, 데미지량, 이동 거리, 최종 순위 같은 match-level summary statistic에 의존한다. 그러나 이러한 집계 통계는 특정 시점에 팀이 어디에 있었는지, 주변에 어떤 적이 있었는지, 안전구역과의 상대적 위치가 어땠는지, 교전 이벤트가 어떤 시간 순서로 발생했는지를 잃어버린다.

핵심 문제는 다음과 같다.

> 집계 통계는 시공간 구조를 잃는다.

따라서 생존 패턴을 제대로 이해하려면 좌표를 단순 feature로 사용하는 것을 넘어, 플레이어/팀 trajectory, 안전구역 변화, 주변 적 밀도, 전투 이벤트의 시간적 누적을 하나의 시공간 데이터베이스 구조로 통합해야 한다.

### 3.3 Research Objective

본 프로젝트의 목표는 PUBG 공식 telemetry를 시공간 데이터베이스로 재구성하고, 시공간 질의를 통해 팀 생존 성과를 설명하는 feature를 추출하는 것이다.

구체적인 목표는 다음과 같다.

1. PUBG telemetry를 moving object table, spatial event table, dynamic region table로 설계한다.
2. range, nearest-neighbor, trajectory, encounter, zone-distance query를 정의한다.
3. 질의 결과를 team-time snapshot 및 match-team feature로 변환한다.
4. 추출된 시공간 feature가 final rank, survival time, top-k placement와 어떤 관계를 갖는지 분석한다.
5. 단순 match-level summary baseline과 query-derived spatio-temporal feature를 비교한다.

### 3.4 Expected Contributions

본 프로젝트의 기대 기여는 다음과 같다.

1. PUBG 공식 telemetry를 플레이어 궤적, 전투 이벤트, 동적 안전구역으로 구성된 시공간 데이터베이스 스키마로 설계한다.
2. 배틀로얄 생존 패턴 분석에 필요한 range query, nearest-neighbor query, trajectory query, encounter query, zone-distance query를 정의한다.
3. 시공간 질의로 추출한 해석 가능한 feature가 단순 경기 요약 통계보다 팀 생존 성과를 더 잘 설명하는지 평가한다.
4. 수업 PPT의 moving object, trajectory, distance, topology, spatial indexing 개념을 실제 게임 telemetry 데이터에 적용한다.

## 4. Problem Definition

### 4.1 Spatio-Temporal Objects

본 프로젝트는 PUBG 매치를 세 가지 시공간 객체로 모델링한다.

첫째, 플레이어와 팀은 시간에 따라 위치가 변하는 moving object다.

플레이어 `i`의 위치는 다음과 같이 표현한다.

```text
p_i(t) = (x_i(t), y_i(t), z_i(t))
```

팀 `g`의 위치는 같은 팀 플레이어 위치의 centroid로 표현한다.

```text
P_g(t) = centroid({p_i(t) | player i belongs to team g and is alive at t})
```

둘째, 전투 이벤트는 특정 시간과 위치에서 발생하는 timestamped spatial event다.

```text
e_j = (tau_j, l_j, actor_j, target_j, type_j)
```

여기서 `tau_j`는 이벤트 발생 시각, `l_j`는 이벤트 위치, `actor_j`와 `target_j`는 행위자와 대상, `type_j`는 damage, attack, DBNO, kill 등의 이벤트 유형이다.

셋째, 안전구역은 시간에 따라 중심과 반경이 바뀌는 dynamic spatial region이다.

```text
Z(t) = (center_x(t), center_y(t), radius(t), phase(t))
```

### 4.2 Input Data

입력 데이터는 다음 세 범주로 정의한다.

```text
Player trajectories:
T_i = {(t_1, x_i1, y_i1, z_i1), ..., (t_n, x_in, y_in, z_in)}

Event logs:
E = {e_j = (tau_j, l_j, actor_j, target_j, type_j)}

Zone states:
Z = {Z(t) = (center_x(t), center_y(t), radius(t), phase(t))}
```

### 4.3 Analysis Unit

본 프로젝트의 기본 분석 단위는 **10초 간격의 team-time snapshot**이다. 각 snapshot에서 한 팀은 위치, 생존 인원, 체력, 안전구역과의 거리, 주변 적 수, 최근 전투 이벤트 수, 동적 교전 네트워크 위치를 가진 하나의 시공간 객체로 표현된다.

분석 단위는 두 층으로 구성한다.

| 분석 단위 | 용도 |
| --- | --- |
| `team-time snapshot` | range/kNN/zone-distance query, 시간창 feature 계산 |
| `match-team aggregate` | final rank, survival time, top-k placement와 연결 |

### 4.4 Target Output

본 프로젝트는 임의의 교전 승패 label을 만들지 않는다. 대신 공식 결과 또는 명확히 관측 가능한 결과를 사용한다.

| Target | 정의 |
| --- | --- |
| `final_rank` | 공식 match 결과의 팀 최종 순위 |
| `survival_time` | 팀이 매치에서 생존한 시간 |
| `top_k_placement` | top 1, top 5, top 10 등 순위 그룹 |
| `rank_quartile` | 최종 순위를 사분위로 나눈 결과 |

선택적 예측 task로는 특정 시점 `t`에서 다음 `Delta`초 안에 탈락하는지 여부를 사용할 수 있다.

```text
y_g,t^Delta = 1 if team g is eliminated in (t, t + Delta], otherwise 0
```

이 task는 교전 승패와 달리 공식 생존/탈락 시점에서 직접 파생되므로 label 휴리스틱성이 낮다.

### 4.5 Research Questions

본 프로젝트의 연구 질문은 다음과 같다.

**RQ1.** 시간에 따라 축소되는 안전구역은 팀의 공간 분포와 이동 패턴을 어떻게 변화시키는가?

**RQ2.** 특정 시점의 주변 적 밀도, nearest enemy distance, 안전구역 경계 거리는 팀의 생존 성과와 어떤 관계를 갖는가?

**RQ3.** trajectory query로 추출한 이동 거리, 이동 효율, 방향 변화, 팀 산개도는 final rank 및 survival time과 관련되는가?

**RQ4.** 전투 이벤트로 구성한 동적 combat graph의 중심성, 연결성, community 위치는 팀 생존 성과와 관련되는가?

**RQ5.** 시공간 query-derived feature는 단순 match-level summary statistic보다 생존 결과를 더 잘 설명하거나 예측하는가?

## 5. Related Work / Background

### 5.1 Spatio-Temporal Databases

시공간 데이터베이스는 시간에 따라 위치와 상태가 변화하는 객체를 저장하고 질의하기 위한 데이터 모델과 연산을 다룬다. 본 프로젝트에서는 플레이어와 팀을 moving object로, 안전구역을 dynamic spatial region으로, 전투 이벤트를 timestamped spatial event로 본다.

이 관점은 `/Users/lunecid/Documents/Spatio-Temporal`의 `SD-Chap-9-SpatiotemporalData.ppt` 및 `SD-Chap-9-SpatiotemporalData (1).ppt`에서 다루는 moving object, trajectory, snapshot, valid time 개념과 직접 연결된다.

### 5.2 Trajectory Analysis

Trajectory는 시간 순서가 있는 위치 샘플의 집합이다. 플레이어 또는 팀 trajectory를 구성하면 이동 거리, 이동 속도, 방향 변화, 특정 phase의 경로, 안전구역 방향으로의 접근 여부를 분석할 수 있다.

본 프로젝트에서는 `SD-Chap-2-Representation.ppt`의 spatial object representation 개념을 사용하여 플레이어를 point object로, 팀을 centroid와 dispersion을 가진 aggregate object로 표현한다. 또한 `SD-Chap-3-Distance.ppt`의 distance measure 개념을 사용하여 팀 간 거리, zone distance, nearest enemy distance를 계산한다.

### 5.3 Spatial Indexing and Query Processing

대량의 team-time snapshot에서 주변 적 수나 nearest enemy를 계산하려면 효율적인 spatial query가 필요하다. 본 프로젝트에서는 `SD-Chap-7-Spatial_Indexing.ppt`의 grid, KD-tree, quadtree, R-tree 개념을 바탕으로 range query와 nearest-neighbor query를 구현한다.

### 5.4 Event-Based Game and Sports Analytics

게임 telemetry는 일반 GPS trajectory 데이터와 달리 이동 궤적뿐 아니라 damage, kill, DBNO, heal, vehicle ride, zone change 같은 semantic event를 함께 포함한다. 따라서 geometric proximity와 event sequence를 결합할 수 있다.

본 프로젝트에서는 `SD-Chap-4-Topology.ppt`의 adjacency, connectivity 개념을 교전 이벤트 기반 dynamic combat graph로 확장한다. 팀은 노드, 팀 간 damage relation은 edge, 누적 damage는 edge weight로 표현한다.

## 6. Data Description

### 6.1 Data Source

본 프로젝트는 PUBG 공식 API와 telemetry log를 사용한다. 공식 문서에 따르면 match object의 asset reference를 통해 telemetry file URL을 얻을 수 있으며, telemetry file은 다양한 event object의 배열로 구성된다.

현재 프로젝트에서 우선 사용할 데이터셋은 다음과 같다.

```text
/Users/lunecid/project/PUBG/network_analysis/output/analysis_datasets/Baltic_Main_squad_n1000
```

초기 분석 범위는 `Baltic_Main / squad` 1000개 매치, 약 26,000개 이상의 team sample로 둔다. 필요하면 프로젝트 기간 안에서 표본 수를 줄여 smoke analysis를 수행하고, 최종 단계에서 1000개 매치로 확장한다.

### 6.2 Main Tables

제안하는 시공간 데이터베이스 스키마는 다음과 같다.

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

`player_positions`와 `team_snapshots`는 moving object table, `combat_events`는 spatial event table, `zone_states`는 time-varying region table, `team_outcomes`는 official result table에 해당한다.

### 6.3 Temporal and Spatial Resolution

위치와 상태 정보는 분석 가능성을 위해 10초 간격 snapshot으로 정규화한다. 이벤트는 원본 timestamp를 유지하되, feature extraction 단계에서 최근 30초/60초/300초 같은 temporal window에 맞춰 집계한다.

공식 telemetry의 `Location` 객체는 `x`, `y`, `z` 좌표를 제공하며, 좌표 단위는 centimeters다. 따라서 분석에서는 meter 단위로 변환하여 거리 기반 질의를 수행한다.

### 6.4 Spatio-Temporal Nature of the Data

본 데이터는 세 가지 시공간 요소가 결합되어 있다.

1. 플레이어와 팀 위치는 시간에 따라 변화하는 moving object data다.
2. damage, DBNO, kill, heal 등 이벤트는 특정 시간과 위치에서 발생하는 spatial event다.
3. 안전구역은 시간에 따라 중심과 반경이 변하는 dynamic spatial region이다.

따라서 PUBG telemetry는 trajectory, event, dynamic region이 결합된 시공간 데이터베이스 문제로 모델링할 수 있다.

### 6.5 Preprocessing Plan

전처리 과정은 다음과 같다.

1. Raw telemetry event를 event type별로 파싱한다.
2. 각 매치 내부 timestamp를 `elapsed_time`으로 정규화한다.
3. 플레이어 위치를 10초 단위 snapshot으로 재구성한다.
4. 팀 단위 centroid, HP, alive count, squad spread를 계산한다.
5. zone state를 같은 시간축에 정렬한다.
6. combat event를 attacker/victim team 기준 edge로 변환한다.
7. 관측 window 이후 정보를 제거하여 leakage를 방지한다.

## 7. Proposed Method

### 7.1 Overall Pipeline

본 프로젝트의 분석 흐름은 다음과 같다.

```text
Raw PUBG telemetry
-> event parsing and timestamp normalization
-> spatio-temporal schema construction
-> team-time snapshot generation
-> spatio-temporal query execution
-> query-derived feature extraction
-> survival pattern analysis and baseline comparison
```

### 7.2 Spatio-Temporal Query Design

본 프로젝트의 핵심은 시공간 질의를 정의하고, 그 결과를 해석 가능한 feature로 변환하는 것이다.

#### Query 1. Range Query

특정 시점 `t`에서 팀 `g` 주변 반경 `r` 안에 있는 적 팀을 찾는다.

```text
Q_range(g, r, t) = {h | team h != g and dist(P_g(t), P_h(t)) <= r}
```

Feature:

```text
enemy_count_r(g, t) = |Q_range(g, r, t)|
```

사용 예시:

- `enemy_count_100m`
- `enemy_count_200m`
- `enemy_count_500m`

#### Query 2. k-Nearest Neighbor Query

특정 시점 `t`에서 팀 `g`와 가장 가까운 `k`개 적 팀을 찾는다.

```text
Q_knn(g, k, t) = top-k enemy teams h by dist(P_g(t), P_h(t))
```

Feature:

```text
nearest_enemy_distance(g, t)
```

#### Query 3. Trajectory Query

특정 시간 구간 `[t_a, t_b]` 동안 팀의 이동 경로를 추출한다.

```text
Q_traj(g, t_a, t_b) = {P_g(t) | t_a <= t <= t_b}
```

Feature:

- `travel_distance`
- `net_displacement`
- `movement_efficiency`
- `mean_speed`
- `direction_change_rate`
- `squad_spread_mean`

#### Query 4. Encounter Query

두 팀이 일정 시간 이상 가까운 거리에 있었는지를 계산한다.

```text
Encounter(g, h, [t_a, t_b]) = 1
if sum_{t=t_a}^{t_b} I[dist(P_g(t), P_h(t)) <= r] >= m
```

여기서 `r`은 거리 기준, `m`은 최소 time step 수다. 이 질의는 교전 승패 label을 만들기 위한 것이 아니라, 주변 적 노출 정도와 근접 상황을 측정하기 위한 feature로 사용한다.

Feature:

- `encounter_count_r`
- `encounter_duration`
- `unique_nearby_enemy_teams`

#### Query 5. Zone-Distance Query

각 팀이 안전구역 중심 또는 경계로부터 얼마나 떨어져 있는지 계산한다.

```text
d_zone(g, t) = dist(P_g(t), center(t)) - radius(t)
```

해석:

```text
d_zone(g, t) < 0: safe zone 내부
d_zone(g, t) = 0: safe zone 경계
d_zone(g, t) > 0: safe zone 외부
```

Feature:

- `zone_center_distance`
- `zone_edge_distance`
- `time_outside_safe_zone`
- `zone_approach_speed`

#### Query 6. Event Window Query

특정 시점 `t` 이전 `w`초 동안 팀 주변에서 발생한 이벤트를 집계한다.

```text
Q_event(g, r, w, t) =
{e_j | t-w <= tau_j <= t and dist(P_g(t), l_j) <= r}
```

Feature:

- `local_combat_density_recent`
- `recent_damage_dealt`
- `recent_damage_taken`
- `recent_kill_count`
- `time_since_last_combat`

### 7.3 Feature Extraction

질의 결과는 다음 feature vector로 변환한다.

```text
x_g,t = [
    d_zone(g,t),
    nearest_enemy_distance(g,t),
    enemy_count_200m(g,t),
    local_combat_density_recent(g,t),
    travel_distance(g,[t-w,t]),
    movement_efficiency(g,[t-w,t]),
    squad_spread_mean(g,[t-w,t]),
    degree_until_t(g),
    betweenness_until_t(g)
]
```

feature group은 다음 네 가지로 나눈다.

| Feature group | 주요 변수 |
| --- | --- |
| Trajectory | 이동 거리, 이동 효율, 평균 속도, 방향 변화, 산개도 |
| Zone-relative | zone 중심 거리, 경계 거리, zone pressure, outside time |
| Local context | nearest enemy, 반경 내 적 팀 수, 주변 교전 밀도 |
| Dynamic graph | degree, weighted degree, betweenness, PageRank, community position |

### 7.4 Dynamic Combat Graph

각 시점 `t`까지 발생한 팀 간 damage event를 누적하여 동적 그래프를 구성한다.

```text
G_t = (V_t, E_t, W_t)
```

여기서 `V_t`는 시점 `t`까지 생존하거나 관측된 팀, `E_t`는 팀 간 combat relation, `W_t`는 누적 damage weight다.

계산할 graph feature는 다음과 같다.

- `degree_until_t`
- `weighted_degree_until_t`
- `damage_dealt_until_t`
- `damage_taken_until_t`
- `betweenness_until_t`
- `pagerank_until_t`
- `community_embeddedness_until_t`
- `component_size_until_t`

이 부분은 `SD-Chap-4-Topology.ppt`의 adjacency/connectivity 개념과 기존 `network_analysis` 구현의 team-team damage graph를 연결하는 축이다.

## 8. Experimental Plan / Evaluation

### 8.1 Query System Evaluation

시공간 DB 프로젝트로서 질의 처리 성능도 평가한다.

평가 항목:

- range query 실행 시간
- nearest-neighbor query 실행 시간
- 데이터 크기 증가에 따른 scalability
- spatial index 사용 전후 실행 시간 비교

비교 방식:

| 비교 | 목적 |
| --- | --- |
| naive pairwise distance | index 없는 기준 |
| KD-tree 또는 BallTree | point snapshot 기반 근접 질의 |
| grid index | 빠른 근사 range query |

### 8.2 Survival Pattern Analysis

설명 분석에서는 다음 결과를 제시한다.

| Research Question | Evaluation |
| --- | --- |
| RQ1. 안전구역 축소와 공간 분포 | phase별 team density heatmap, spatial entropy |
| RQ2. 주변 적 밀도와 생존 성과 | eliminated/survived group의 enemy proximity distribution 비교 |
| RQ3. trajectory와 성과 | 이동 거리, 이동 효율, 산개도와 rank/survival time 상관 |
| RQ4. combat graph와 성과 | degree/betweenness/community position과 final rank 분석 |
| RQ5. feature 유용성 | baseline 대비 query-derived feature의 성능 개선 |

통계 방법:

- Pearson/Spearman correlation
- Mann-Whitney U test
- partial correlation
- regression with survival-time or phase controls
- rank residual analysis

### 8.3 Prediction Evaluation

예측은 보조 실험으로 수행한다. 핵심은 모델 복잡도보다 feature group의 기여도를 평가하는 것이다.

Prediction tasks:

- final rank regression
- top-k placement classification
- rank quartile classification
- optional: elimination within next `Delta` seconds

Models:

- Logistic Regression
- Ridge/Lasso Regression
- Random Forest
- Gradient Boosting

Metrics:

- MAE for final rank
- Spearman correlation for rank ordering
- ROC-AUC for top-k classification
- PR-AUC and F1 for imbalanced elimination/top-k tasks
- Brier score for calibration

### 8.4 Baseline Comparison and Ablation

Baseline과 proposed feature를 다음 순서로 비교한다.

| Experiment | Feature set |
| --- | --- |
| Baseline 1 | match-level summary statistics |
| Baseline 2 | temporal event counts |
| Proposed 1 | + trajectory query features |
| Proposed 2 | + zone-distance query features |
| Proposed 3 | + range/kNN local context features |
| Proposed 4 | + dynamic combat graph features |

이 ablation을 통해 시공간 질의 기반 feature가 실제로 설명력 또는 예측력을 개선하는지 확인한다.

### 8.5 Visualization Plan

최종 결과는 다음 방식으로 시각화한다.

- phase별 team density heatmap
- combat event hotspot map
- zone edge distance distribution
- nearest enemy distance vs survival time plot
- dynamic combat graph centrality vs final rank plot
- feature group별 ablation performance chart

## 9. Expected Contributions

본 프로젝트의 결과물은 다음 네 가지다.

1. **시공간 데이터베이스 스키마**  
   PUBG telemetry를 moving object, spatial event, dynamic region으로 분해한 스키마를 제안한다.

2. **질의 기반 feature extraction 방법**  
   range, kNN, trajectory, encounter, zone-distance, event-window query를 정의하고, 이를 생존 패턴 분석 feature로 변환한다.

3. **생존 패턴 분석 결과**  
   안전구역 거리, 주변 적 밀도, 이동 패턴, 동적 교전 네트워크 위치가 final rank/survival time과 어떤 관계를 갖는지 분석한다.

4. **baseline 대비 평가**  
   단순 match-level summary statistic과 시공간 query-derived feature를 비교하여, 시공간 데이터베이스 접근의 실질적 가치를 평가한다.

## 10. Timeline

| 기간 | 작업 |
| --- | --- |
| 1주차 | 데이터 구조 확인, 공식 telemetry field 정리, 스키마 설계 |
| 2주차 | 위치 snapshot 정규화, team centroid/health/alive table 생성 |
| 3주차 | range query, kNN query, trajectory query, zone-distance query 구현 |
| 4주차 | feature extraction, phase별 분포 분석, hotspot/local density 분석 |
| 5주차 | baseline 비교, ablation study, rank/top-k/survival prediction 실험 |
| 6주차 | 결과 시각화, 보고서 작성, 발표 자료 구성 |

프로젝트 기간이 짧을 경우 최소 범위는 다음으로 축소한다.

1. `Baltic_Main / squad` 일부 표본에서 10초 team-time snapshot 생성
2. range/kNN/zone-distance/trajectory query 구현
3. final rank 또는 top-k placement와의 관계 분석
4. 간단한 baseline 비교와 시각화

## 11. Limitations and Risk Management

### 11.1 Map Semantics Limitation

공식 telemetry만으로 건물 내부/외부, 엄폐, 개활지, line-of-sight를 정확히 알 수 없다. 따라서 본 프로젝트는 맵 구조를 직접 해석한다고 주장하지 않고, 공식 좌표와 이벤트에서 관측 가능한 시공간 맥락만 분석한다.

### 11.2 Sampling and Missingness

위치 샘플이 모든 순간을 연속적으로 기록하지 않을 수 있다. 이를 해결하기 위해 10초 간격 snapshot으로 resampling하고, 짧은 결측은 nearest previous state 방식으로 처리한다. 긴 결측은 해당 구간 feature를 missing으로 표시하거나 제외한다.

### 11.3 Exposure-Time Bias

오래 생존한 팀은 더 많은 이벤트를 관측할 기회를 갖는다. 따라서 cumulative feature만 사용하지 않고, early 300s, phase-specific window, per-minute rate, survival-time control을 함께 사용한다.

### 11.4 Scope Management

본 프로젝트는 복잡한 딥러닝 모델 개발보다 시공간 데이터베이스 설계, 질의 정의, feature extraction, 해석 가능한 분석에 초점을 둔다. 모델링은 logistic regression, random forest, gradient boosting 수준의 baseline으로 제한한다.

### 11.5 Label Reliability

교전 승패처럼 정의가 모호한 label은 주 target으로 사용하지 않는다. final rank, survival time, top-k placement처럼 공식 결과 또는 명확히 관측 가능한 결과를 사용한다.

## 12. References

1. PUBG Developer Documentation. **Telemetry**. https://documentation.pubg.com/en/telemetry.html
2. PUBG Developer Documentation. **Telemetry Events**. https://documentation.pubg.com/en/telemetry-events.html
3. PUBG Developer Documentation. **Telemetry Objects**. https://documentation.pubg.com/en/telemetry-objects.html
4. Güting, R. H., Böhlen, M. H., Erwig, M., Jensen, C. S., Lorentzos, N. A., Schneider, M., & Vazirgiannis, M. (2000). **A Foundation for Representing and Querying Moving Objects**. ACM Transactions on Database Systems, 25(1), 1-42. https://doi.org/10.1145/352958.352963
5. Guttman, A. (1984). **R-Trees: A Dynamic Index Structure for Spatial Searching**. Proceedings of ACM SIGMOD. https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Guttman84.pdf
6. `/Users/lunecid/Documents/Spatio-Temporal/SD-Chap-2-Representation.ppt`
7. `/Users/lunecid/Documents/Spatio-Temporal/SD-Chap-3-Distance.ppt`
8. `/Users/lunecid/Documents/Spatio-Temporal/SD-Chap-4-Topology.ppt`
9. `/Users/lunecid/Documents/Spatio-Temporal/SD-Chap-7-Spatial_Indexing.ppt`
10. `/Users/lunecid/Documents/Spatio-Temporal/SD-Chap-8-BuildingData.ppt`
11. `/Users/lunecid/Documents/Spatio-Temporal/SD-Chap-9-SpatiotemporalData.ppt`
12. `/Users/lunecid/Documents/Spatio-Temporal/SD-Chap-9-SpatiotemporalData (1).ppt`

## 13. One-Sentence Summary

본 프로젝트는 PUBG 매치를 플레이어/팀 궤적, 전투 이벤트, 동적 안전구역으로 구성된 시공간 데이터베이스로 모델링하고, range query, nearest-neighbor query, trajectory query, encounter query, zone-distance query를 통해 추출한 시공간 특징이 팀 생존 패턴을 어떻게 설명하는지 분석한다.

