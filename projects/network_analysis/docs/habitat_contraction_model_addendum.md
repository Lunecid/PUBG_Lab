# 생태학적 서식지 축소 모델을 활용한 PUBG 시공간 분석 제안

## 1. 제안 개요

PUBG의 안전구역 축소는 생태학에서 다루는 서식지 감소 및 서식지 압축 상황과 구조적으로 유사하다. 시간이 지날수록 이용 가능한 공간이 줄어들고, 생존한 팀들은 더 작은 영역 안으로 압축된다. 이 과정에서 팀 밀도, 근접 조우, 교전 발생 가능성, 탈락 위험이 함께 변한다.

따라서 기존 주제는 다음과 같이 재정의할 수 있다.

> PUBG 공식 telemetry에서 안전구역 축소를 동적 서식지 감소 과정으로 모델링하고, 서식지 면적, 팀 밀도, 이동 반응, 교전 네트워크 변화가 생존 성과와 어떤 관계를 갖는지 분석한다.

이 제안은 기존의 시공간 데이터베이스 주제를 유지하면서, safe zone 축소에 대한 이론적 해석을 생태학 모델에서 가져온다.

## 2. 생태학 모델과 PUBG 대응 관계

| 생태학 개념 | PUBG 대응 |
| --- | --- |
| habitat | 안전구역 또는 현재 생존 가능한 공간 |
| habitat loss | safe zone 면적 축소 |
| habitat compression | 생존 팀이 좁아진 공간 안으로 압축되는 과정 |
| population density | 단위 면적당 생존 팀 수 |
| species richness / occupancy | 생존 팀 수 또는 생존 팀 비율 |
| extinction | 팀 탈락 |
| dispersal / movement | 안전구역 안쪽으로 이동하거나 회전하는 과정 |
| patch / region | phase별 safe zone |
| edge effect | 안전구역 경계 근처의 위험 |

이 대응은 생물학적 동일성을 주장하는 것이 아니라, 축소되는 공간 안에서 개체군이 압축되고 탈락 위험이 변한다는 구조적 유사성을 이용하는 것이다.

## 3. 핵심 모델 1: Species-Area Relationship

생태학에서 species-area relationship은 일반적으로 다음 형태로 표현된다.

```text
S = c A^z
```

여기서 `S`는 종 수, `A`는 서식지 면적, `c`와 `z`는 상수다. 이를 PUBG에 적용하면 다음과 같이 바꿀 수 있다.

```text
N_t = c A_t^z
```

| 기호 | 의미 |
| --- | --- |
| `N_t` | 시점 `t`의 생존 팀 수 |
| `A_t` | 시점 `t`의 안전구역 면적 |
| `z` | 안전구역 축소에 대한 생존 팀 수의 민감도 |
| `c` | 매치별 초기 조건을 반영하는 상수 |

로그 변환하면 다음 회귀식으로 추정할 수 있다.

```text
log N_t = log c + z log A_t
```

이 모델은 이미 기존 코드에도 일부 구현되어 있다.

- `dataset.py`의 `species_area_fit`은 `log(alive_teams)`와 `log(zone_area)`의 선형 관계를 fitting한다.
- `result_viz.py`의 `plot_species_area`는 empirical species-area curve를 시각화한다.
- `adapters/base.py`의 `ArenaState`는 `safe_area_ratio`, `shrink_rate`, `alive_groups_ratio`, `density`를 포함한다.

따라서 이 방향은 완전히 새로 시작하는 주제가 아니라, 기존 구현을 시공간 데이터베이스와 생태학 모델의 언어로 재정리하는 방식이다.

## 4. 핵심 모델 2: Habitat Compression Pressure

safe zone 면적이 줄어들면 생존 팀 수가 줄어드는 동시에 단위 면적당 팀 밀도가 증가할 수 있다. 이를 habitat compression pressure로 정의한다.

```text
rho_t = N_t / A_t
```

여기서 `rho_t`는 시점 `t`의 생존 팀 밀도다. 팀 단위로는 주변 적 밀도를 함께 계산한다.

```text
local_density_i,t = enemy_count_r(i,t) / area(r)
```

분석 질문은 다음과 같다.

1. safe zone 면적이 줄어들수록 전체 팀 밀도는 증가하는가?
2. 팀 밀도가 증가하는 phase에서 교전 이벤트와 탈락 위험도 증가하는가?
3. 같은 safe zone 면적에서도 local density가 높은 팀은 생존 성과가 낮은가?
4. 밀도 증가에 빠르게 대응해 zone 내부로 이동한 팀은 더 오래 생존하는가?

이 모델은 시공간 질의와 직접 연결된다.

| 압축 지표 | 계산 방식 |
| --- | --- |
| global density | `alive_teams / safe_zone_area` |
| local density | range query로 반경 `r` 안 적 팀 수 계산 |
| edge pressure | `dist(team, safe_center) - safe_radius` |
| shrink rate | `-dA/dt` 또는 phase 간 면적 감소율 |
| displacement pressure | 이전 safe zone 중심과 다음 safe zone 중심의 이동량 |

## 5. 핵심 모델 3: Team-Level Survival Hazard

species-area relationship은 match 전체의 거시적 관계를 설명한다. 팀 단위 분석에는 survival hazard 모델이 더 적합하다.

예측 대상은 특정 팀이 다음 시간창 안에 탈락하는지 여부로 둘 수 있다.

```text
y_i,t^Delta = 1 if team i is eliminated in (t, t + Delta], otherwise 0
```

입력 feature는 habitat shrinkage와 trajectory response를 결합한다.

```text
hazard_i,t = f(
    safe_area_ratio_t,
    shrink_rate_t,
    global_density_t,
    local_density_i,t,
    zone_edge_distance_i,t,
    zone_approach_speed_i,t,
    squad_spread_i,t,
    recent_combat_i,t
)
```

이 모델은 교전 승패 label을 만들지 않는다. 결과는 공식 telemetry에서 얻는 탈락 시점 또는 생존 시간에 기반한다.

## 6. 시공간 DB 관점에서의 장점

이 제안은 시공간 데이터베이스 과목과 잘 맞는다.

| 시공간 DB 요소 | 적용 |
| --- | --- |
| moving object | 플레이어/팀 trajectory |
| dynamic spatial region | 시간에 따라 축소되는 safe zone |
| spatial event | damage, DBNO, kill, heal 이벤트 |
| range query | 주변 적 팀 수와 local density 계산 |
| nearest-neighbor query | nearest enemy distance 계산 |
| trajectory query | safe zone 축소에 대한 이동 반응 분석 |
| temporal window query | 최근 전투 및 탈락 위험 계산 |
| spatial indexing | 반경 질의와 kNN 질의 최적화 |

특히 safe zone은 단순 배경 변수가 아니라, 시간에 따라 축소되는 동적 공간 영역이다. 이 점이 서식지 축소 모델과 시공간 데이터베이스 모델을 연결한다.

## 7. 연구 질문 재구성

이 프레임을 사용하면 연구 질문을 다음과 같이 정리할 수 있다.

**RQ1.** Safe zone 면적 감소와 생존 팀 수 사이에는 species-area relationship 형태의 관계가 나타나는가?

**RQ2.** Safe zone 축소로 인한 global density와 local enemy density 증가는 탈락 위험과 관련되는가?

**RQ3.** 팀의 zone-relative trajectory는 habitat compression에 대한 이동 반응으로 해석될 수 있는가?

**RQ4.** Habitat compression pressure가 높은 phase에서 dynamic combat graph의 밀도와 중심성 구조는 어떻게 변하는가?

**RQ5.** Habitat shrinkage feature를 추가하면 단순 trajectory/event feature보다 final rank, survival time, 탈락 위험을 더 잘 설명하는가?

## 8. 실험 설계

### 8.1 Match-Level Species-Area Fit

각 매치에서 `log(safe_zone_area)`와 `log(alive_teams)`를 계산하고, 다음 회귀식을 fitting한다.

```text
log N_t = log c + z log A_t
```

평가 항목:

- match별 `z` 분포
- match별 `R^2`
- phase별 잔차
- `z`와 match intensity, final combat density의 관계

### 8.2 Habitat Compression Feature

team-time snapshot마다 다음 feature를 계산한다.

| Feature | 정의 |
| --- | --- |
| `safe_area_ratio` | 현재 safe zone 면적 / 초기 면적 |
| `shrink_rate` | 시간당 safe zone 면적 감소율 |
| `global_team_density` | 생존 팀 수 / safe zone 면적 |
| `local_enemy_density_r` | 반경 `r` 내 적 팀 수 / 원 면적 |
| `zone_edge_distance` | safe zone 경계까지 거리 |
| `outside_safe_zone` | safe zone 외부 여부 |
| `zone_approach_speed` | safe zone 중심 또는 경계 방향 이동 속도 |

### 8.3 Survival Hazard Analysis

팀 단위 결과는 다음 중 하나로 둔다.

- `survival_time`
- `final_rank`
- `top_k_placement`
- 다음 `Delta`초 내 탈락 여부

비교 실험은 다음 순서로 구성한다.

| 실험 | Feature set |
| --- | --- |
| Baseline | phase, elapsed time, alive teams |
| +Trajectory | 이동 거리, 속도, 방향 변화, squad spread |
| +Event | 최근 damage, DBNO, kill, combat density |
| +Habitat shrinkage | safe area ratio, shrink rate, density, edge pressure |
| +Dynamic graph | degree, weighted degree, betweenness, community position |

## 9. 기존 제안서와의 관계

기존 제안서는 trajectory와 시공간 질의 중심이었다. 생태학적 서식지 축소 모델을 도입하면 다음처럼 초점이 바뀐다.

| 기존 초점 | 보강된 초점 |
| --- | --- |
| 팀 trajectory 분석 | 서식지 축소에 대한 이동 반응 분석 |
| zone-distance feature | habitat edge pressure |
| local enemy density | habitat compression pressure |
| dynamic safe zone | shrinking habitat |
| survival outcome | extinction/survival process |
| movement archetype | compression response archetype |

즉, 기존 제안서를 버리는 것이 아니라, safe zone 축소를 설명하는 이론적 프레임을 더하는 것이다.

## 10. 주의할 점

이 프레임은 유용하지만, 다음 표현은 피해야 한다.

- PUBG 팀을 실제 생물종과 동일하게 본다.
- species-area relationship이 PUBG 생존을 완전히 설명한다.
- safe zone이 실제 생태계 서식지와 동일하다.
- 생태학 모델로 인과를 증명한다.

대신 다음처럼 표현하는 것이 안전하다.

- 생태학적 서식지 축소 모델을 분석적 비유로 차용한다.
- safe zone을 시간에 따라 축소되는 가용 공간으로 모델링한다.
- 생존 팀 수와 가용 면적의 관계를 species-area form으로 검정한다.
- 팀 단위 탈락 위험은 habitat compression pressure와 trajectory response의 함수로 분석한다.

## 11. 최종 제안 문장

> 본 연구는 PUBG의 safe zone을 시간에 따라 축소되는 동적 서식지로 모델링하고, 생태학의 species-area relationship과 habitat compression 개념을 차용하여 안전구역 면적, 생존 팀 밀도, 팀 이동 반응, 교전 네트워크 변화가 생존 성과와 어떤 관계를 갖는지 분석한다.

이 제안은 공식 telemetry만으로 구현 가능하며, 기존 시공간 데이터베이스 설계와도 정합된다. 또한 기존 코드의 species-area fitting 및 arena state feature를 재사용할 수 있어 프로젝트 기간 안에서 실현 가능성이 높다.

