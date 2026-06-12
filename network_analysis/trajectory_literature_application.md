# Trajectory 선행연구 관점과 PUBG 적용 전략

## 1. 결론

trajectory 연구를 참고하면, 본 프로젝트는 단순히 `x, y, t` 좌표에서 이동 거리와 속도를 뽑는 수준에 머물면 약하다. 일반적인 trajectory 연구는 다음 흐름을 가진다.

```text
raw trajectory
-> preprocessing / resampling
-> segmentation
-> semantic enrichment
-> spatio-temporal query
-> similarity / clustering / pattern mining
-> outcome analysis or prediction
```

PUBG에 적용할 때도 이 흐름을 따라야 한다. 다만 공식 telemetry만 사용한다는 조건 때문에 건물, 엄폐, 도로 같은 외부 의미 라벨은 사용하지 않는다. 대신 PUBG 내부에서 공식적으로 관측되는 `phase`, `safe zone`, `combat event`, `team state`를 이용해 trajectory에 의미를 부여한다.

따라서 본 프로젝트의 trajectory 분석 중심은 다음 문장으로 정리할 수 있다.

> PUBG 팀 trajectory를 phase, safe zone, 주변 적 밀도, combat event로 의미화하고, 시공간 질의와 trajectory pattern analysis를 통해 생존 성과와 연결한다.

## 2. 일반적인 trajectory 연구 흐름

### 2.1 Moving object modeling

Moving object database 연구는 시간에 따라 위치가 변하는 객체를 데이터베이스 타입과 질의 대상으로 모델링한다. 대표적으로 Güting et al.은 moving point와 moving region 같은 추상 타입을 통해 시공간 객체를 표현하고 질의하는 기초를 제시했다.

PUBG 적용:

- 플레이어: moving point
- 팀 centroid: aggregate moving point
- 팀 spread: moving extent 또는 dispersion
- safe zone: moving/dynamic region
- combat event: timestamped spatial event

이 관점은 우리 프로젝트의 시공간 DB 정당성을 만든다.

### 2.2 Preprocessing and resampling

trajectory data mining 연구에서는 원시 위치 샘플을 그대로 사용하지 않고, 시간 정렬, 결측 처리, noise filtering, resampling을 수행한다. Zheng의 trajectory data mining survey는 trajectory 분석 흐름을 data derivation, preprocessing, management, mining task로 정리한다.

PUBG 적용:

- 매치 시작 시간을 기준으로 `elapsed_time` 정규화
- 10초 단위 `team-time snapshot` 생성
- 플레이어 좌표를 팀 centroid와 squad spread로 집계
- 이벤트 timestamp를 가장 가까운 snapshot 또는 temporal window에 연결
- match마다 다른 길이를 phase별 또는 early window로 정렬

### 2.3 Segmentation

trajectory는 전체 경로를 한 덩어리로 비교하기보다, 의미 있는 subtrajectory로 나누는 경우가 많다. Lee, Han, Whang의 TRACLUS는 전체 trajectory가 아니라 line segment와 subtrajectory를 군집화해야 공통 이동 패턴을 더 잘 찾을 수 있다고 본다.

PUBG 적용:

- 전체 매치 trajectory를 그대로 비교하지 않는다.
- phase별로 자른다: phase 1, phase 2, phase 3, phase 4+
- 시간창으로 자른다: early 300s, mid-game, late-game
- safe zone 변화 전후로 자른다.
- combat event 전후 window로 자른다.

추천 기본 segmentation:

```text
segment(g, phase p) = team g의 phase p 구간 trajectory
segment(g, early) = team g의 match start부터 300초까지 trajectory
segment(g, pre_combat) = 특정 combat event 이전 60초 trajectory
```

### 2.4 Semantic trajectories

Spaccapietra et al.의 semantic trajectory 관점은 raw movement를 stop/move, event, goal, environment 같은 의미 단위로 enrich하는 것을 강조한다. 우리에게 중요한 점은 trajectory가 단순 좌표열이 아니라, 도메인 의미가 붙은 이동 기록이라는 것이다.

PUBG 적용:

외부 맵 의미 라벨은 사용하지 않지만, 공식 telemetry에서 얻을 수 있는 의미는 사용할 수 있다.

| Semantic annotation | 공식 데이터 기반 정의 |
| --- | --- |
| `phase` | `LogPhaseChange`, `isGame` |
| `safe_zone_relation` | safe zone 내부/경계/외부 |
| `zone_pressure` | safe zone 또는 poison zone 경계와의 거리 |
| `combat_state` | 최근 30/60초 damage/attack/DBNO/kill 이벤트 |
| `enemy_proximity` | nearest enemy distance, range query |
| `team_state` | alive members, HP summary |
| `mobility_state` | stationary, rotating, approaching zone, escaping zone |

이렇게 하면 건물/엄폐를 몰라도 trajectory를 도메인적으로 해석할 수 있다.

### 2.5 Trajectory similarity and clustering

trajectory 연구에서는 DTW, LCSS, Fréchet/Hausdorff distance 같은 유사도나, TRACLUS 같은 subtrajectory clustering을 통해 유사 이동 패턴을 찾는다. 그러나 우리 프로젝트에서 모든 팀 trajectory 간 복잡한 pairwise similarity를 전면에 두는 것은 구현 부담이 크다.

PUBG 적용 권장안:

1. 1차 구현은 feature-based clustering으로 한다.
2. 각 phase trajectory를 요약 feature vector로 바꾼다.
3. K-means, Gaussian Mixture, HDBSCAN 같은 간단한 군집을 적용한다.
4. 군집을 movement archetype으로 해석한다.

예시 movement archetype:

| Archetype | 특징 |
| --- | --- |
| `early_center_rotation` | 초반부터 safe zone 중심 방향으로 빠르게 이동 |
| `edge_holding` | zone edge 근처에 오래 머무름 |
| `late_rotation` | 초반 이동 적고 후반에 급격히 이동 |
| `wide_roaming` | 이동 거리와 방향 변화가 큼 |
| `compact_slow_play` | 팀 산개도가 낮고 이동량이 작음 |
| `high_exposure_path` | 주변 적 밀도와 encounter duration이 큼 |

2차 확장으로 subtrajectory clustering을 적용할 수 있다. 이때 TRACLUS의 아이디어처럼 전체 match trajectory보다 phase별 line segment clustering이 더 적합하다.

### 2.6 Trajectory pattern mining

Giannotti et al.의 trajectory pattern mining은 이동 객체들의 빈번한 이동 패턴을 찾는 방향이다. PUBG에서는 raw coordinate sequence를 그대로 frequent pattern으로 만들기 어렵기 때문에, 공간을 grid 또는 zone-relative bin으로 이산화한 뒤 sequence pattern으로 바꾸는 것이 좋다.

PUBG 적용:

```text
raw position -> zone-relative bin -> phase sequence
```

예시:

```text
edge_outside -> edge_inside -> center_approach -> center_hold
```

또는 주변 적 밀도까지 포함한다.

```text
low_density_edge -> medium_density_rotation -> high_density_center
```

이 pattern을 final rank/top-k와 연결하면, "어떤 trajectory pattern이 생존 성과와 관련되는가"를 분석할 수 있다.

### 2.7 Team sports spatio-temporal analysis

팀 스포츠 시공간 분석 연구는 개별 선수의 이동만이 아니라 선수 간 상호작용과 팀 단위 구조를 함께 본다. Gudmundsson and Horton의 survey는 축구/농구/하키 같은 invasion sport에서 선수 trajectory와 상호작용을 함께 분석해야 성과를 이해할 수 있다고 본다.

PUBG 적용:

PUBG는 전통 스포츠는 아니지만, squad는 팀 단위로 움직이고 다른 팀과 상호작용한다. 따라서 다음을 함께 봐야 한다.

- 팀 centroid trajectory
- 팀원 spread
- nearest enemy distance
- 주변 적 팀 수
- encounter duration
- dynamic combat graph centrality

즉 PUBG trajectory는 단일 객체 이동이 아니라 multi-agent spatio-temporal interaction 문제로 다루는 것이 맞다.

## 3. PUBG에 맞는 최종 trajectory 분석 프레임

### 3.1 Representation

기본 trajectory는 다음 세 수준으로 표현한다.

| 수준 | 표현 | 용도 |
| --- | --- | --- |
| Player trajectory | `p_i(t) = (x_i, y_i, z_i)` | 원천 moving point |
| Team trajectory | `P_g(t) = centroid(players in team g)` | 주 분석 단위 |
| Semantic trajectory | `(P_g(t), phase, zone_relation, enemy_density, combat_state)` | 생존 패턴 분석 |

팀 trajectory는 단순 centroid만 쓰면 팀 형태를 잃기 때문에, 반드시 `squad_spread`를 함께 둔다.

### 3.2 Coordinate normalization

PUBG에서는 매치마다 safe zone이 다르게 움직이므로 절대 좌표만으로 비교하면 패턴이 흐려질 수 있다. 따라서 두 좌표계를 함께 사용한다.

| 좌표계 | 의미 |
| --- | --- |
| absolute map coordinates | 맵의 실제 위치 기반 이동 |
| zone-relative coordinates | 현재 safe zone 중심/반경 기준 상대 위치 |

zone-relative feature 예시:

```text
relative_zone_x = (team_x - safe_center_x) / safe_radius
relative_zone_y = (team_y - safe_center_y) / safe_radius
zone_edge_distance = dist(team, safe_center) - safe_radius
```

이 방식은 map semantic이 없어도 배틀로얄의 핵심 공간 구조인 safe zone을 반영한다.

### 3.3 Segmentation strategy

전체 trajectory를 한 번에 분석하지 않고, 다음 window별로 나눈다.

| Segment | 목적 |
| --- | --- |
| early 300s | 초반 전략과 생존 노출 편향 분리 |
| phase별 segment | safe zone 변화에 따른 이동 비교 |
| rolling 60s window | 최근 이동/교전 상태 feature |
| pre-elimination window | 탈락 직전 위험 패턴 분석 |

주 분석은 `early 300s`와 `phase별 segment`로 충분하다. pre-elimination은 선택 실험으로 둔다.

### 3.4 Feature extraction

trajectory 선행연구를 반영한 feature group은 다음과 같다.

| Feature group | 변수 |
| --- | --- |
| Geometry | path length, displacement, tortuosity, speed, acceleration |
| Direction | heading change, angular variance, zone approach angle |
| Team shape | squad spread mean/max, split duration |
| Zone-relative | center distance, edge distance, outside time, approach speed |
| Interaction | nearest enemy distance, enemy count in radius, encounter duration |
| Event-enriched | recent damage, recent DBNO/kill, time since combat |
| Graph-enriched | degree, weighted degree, betweenness, community position |

### 3.5 Pattern discovery

본 프로젝트 안에서 현실적으로 가능한 trajectory pattern discovery는 두 단계다.

1. Feature-based movement archetype clustering
   - phase별 trajectory feature를 벡터화
   - clustering으로 movement style 도출
   - cluster별 final rank/survival time 비교

2. Zone-relative sequence pattern
   - position을 `center`, `edge_inside`, `edge_outside`, `far_outside` 등으로 이산화
   - phase sequence를 생성
   - frequent sequence와 top-k placement 관계 분석

이 두 가지가 수업 프로젝트 범위 안에서 가장 방어 가능하다.

## 4. 제안서에 반영해야 할 수정점

현재 제안서의 방향은 맞지만, trajectory 선행연구를 반영하려면 다음 내용을 더 강조하는 것이 좋다.

### 4.1 Related Work 보강

기존 Related Work를 다음 네 축으로 정리한다.

1. Moving object databases
2. Trajectory data mining
3. Semantic trajectories
4. Team-sports spatio-temporal analysis

### 4.2 Method 보강

`Proposed Method`에 다음 subsection을 추가한다.

- Trajectory construction and resampling
- Trajectory segmentation by phase/window
- Semantic enrichment using zone/combat/team state
- Movement archetype discovery

### 4.3 Research Question 보강

다음 RQ를 추가한다.

> RQ. Phase별 team trajectory pattern 또는 movement archetype은 final rank와 survival time의 차이를 설명하는가?

### 4.4 Evaluation 보강

trajectory clustering 결과를 평가하는 방법을 추가한다.

- cluster별 final rank 평균 비교
- cluster별 survival time 분포 비교
- top-k placement 비율 비교
- silhouette score 또는 cluster stability
- feature ablation: trajectory-only vs zone-enriched vs event-enriched

## 5. 우리 프로젝트의 추천 최종 연구 구성

선행연구를 반영하면 프로젝트의 중심은 다음처럼 잡는 것이 가장 좋다.

> PUBG 공식 telemetry에서 팀 trajectory를 phase별 semantic trajectory로 재구성하고, 시공간 질의와 movement pattern discovery를 통해 생존 성과와 관련된 이동 전략을 분석한다.

작업 단위:

1. team-time snapshot 생성
2. phase별 team trajectory 구성
3. safe zone 기준 좌표 정규화
4. range/kNN/zone-distance query 수행
5. trajectory feature extraction
6. movement archetype clustering
7. archetype별 final rank/survival time 비교
8. dynamic combat graph feature와 결합한 ablation

## 6. 핵심 메시지

trajectory 선행연구를 적용한 뒤의 논리 흐름은 다음과 같다.

```text
PUBG telemetry는 moving object + event + dynamic region 데이터다.
팀 trajectory는 전체 경로가 아니라 phase별 subtrajectory로 분석해야 한다.
safe zone과 combat event는 trajectory에 의미를 부여하는 semantic annotation이다.
range/kNN/zone-distance query는 주변 상황을 feature로 바꾼다.
trajectory feature와 movement archetype은 팀 생존 성과와 연결해 평가한다.
```

따라서 제안서의 핵심 문장은 다음처럼 보강할 수 있다.

> 본 프로젝트는 PUBG 팀 trajectory를 phase와 safe zone 기준으로 분절하고, combat event와 enemy proximity로 의미를 부여한 semantic trajectory로 재구성한 뒤, 시공간 질의와 movement pattern analysis를 통해 생존 성과와 관련된 이동 전략을 분석한다.

## 7. 참고 선행연구

1. Güting, R. H., Böhlen, M. H., Erwig, M., Jensen, C. S., Lorentzos, N. A., Schneider, M., & Vazirgiannis, M. (2000). **A Foundation for Representing and Querying Moving Objects**. ACM Transactions on Database Systems, 25(1), 1-42. https://doi.org/10.1145/352958.352963
2. Zheng, Y. (2015). **Trajectory Data Mining: An Overview**. ACM Transactions on Intelligent Systems and Technology, 6(3), 1-41. https://doi.org/10.1145/2743025
3. Lee, J. G., Han, J., & Whang, K. Y. (2007). **Trajectory Clustering: A Partition-and-Group Framework**. SIGMOD 2007, 593-604. https://doi.org/10.1145/1247480.1247546
4. Giannotti, F., Nanni, M., Pinelli, F., & Pedreschi, D. (2007). **Trajectory Pattern Mining**. KDD 2007. https://doi.org/10.1145/1281192.1281230
5. Spaccapietra, S., Parent, C., Damiani, M. L., de Macedo, J. A., Porto, F., & Vangenot, C. (2008). **A Conceptual View on Trajectories**. Data & Knowledge Engineering, 65(1), 126-146. https://doi.org/10.1016/j.datak.2007.10.008
6. Gudmundsson, J., & Horton, M. (2017). **Spatio-Temporal Analysis of Team Sports**. ACM Computing Surveys, 50(2), 1-34. https://doi.org/10.1145/3054132
7. PUBG Developer Documentation. **Telemetry Objects**. https://documentation.pubg.com/en/telemetry-objects.html
8. PUBG Developer Documentation. **Telemetry Events**. https://documentation.pubg.com/en/telemetry-events.html

