# PUBG STDB Dynamic GNN 프로젝트 흐름 정리

> 작성일: 2026-06-05  
> 최신 수치 보정: 2026-06-11  
> 목적: 지금까지의 연구 흐름, 설계 결정, 데이터베이스 재설계, 모델링, ablation 결과, 논문화 방향을 한 문서로 정리한다.  
> 현재 기준 산출물: STDB 구축, phase feature/edge cache, full-DB DynamicSTGNN 학습, full-DB edge ablation, aligned paired test, calibration 완료.

---

## 1. 연구의 출발점

초기 목표는 PUBG telemetry를 이용해 팀의 최종 등수 또는 생존 여부를 예측하는 것이었다. 하지만 단순히 match 종료 후 통계나 snapshot feature를 넣는 방식으로는 연구 주제인 "시공간 데이터베이스 구축 및 시공간 관점의 분석"이 충분히 반영되지 않았다.

핵심 문제의식은 다음과 같이 정리되었다.

- PUBG는 시간이 지날수록 안전구역이 축소되는 동적 공간 게임이다.
- 팀의 생존은 단순 kill, damage, loot 수치만이 아니라 위치, 이동 궤적, 안전구역과의 관계, 주변 적과의 공간 관계, 전투 관계에 의해 결정된다.
- 따라서 raw telemetry를 일반 테이블로만 다루는 것이 아니라, 이동 객체, 동적 공간 영역, 공간 사건으로 재구성한 STDB가 필요하다.
- 모델도 최종 등수 1회 예측이 아니라 phase별 탈락 위험을 예측해야 연구 질문과 정합된다.

이후 프로젝트 방향은 다음 세 가지로 고정되었다.

1. PUBG telemetry를 STDB로 재설계한다.
2. phase 단위 trajectory, exposure, interaction graph feature를 만든다.
3. 팀을 노드로, 팀 간 관계를 동적 엣지로 갖는 Dynamic Spatio-Temporal GNN으로 phase별 탈락 위험을 예측한다.

---

## 2. 연구 질문 정리

현재 프로젝트의 연구 질문은 다음처럼 정리할 수 있다.

| RQ | 질문 | 현재 상태 |
|---|---|---|
| RQ1 | PUBG telemetry는 어떤 시공간 데이터베이스 구조로 표현될 수 있는가? | STDB schema, phase feature, edge cache 구축으로 답변 가능 |
| RQ2 | 안전구역 축소에 따라 팀의 trajectory와 생존 환경은 phase별로 어떻게 달라지는가? | trajectory/exposure/graph feature 및 phase-wise 분석으로 답변 가능 |
| RQ3 | 시공간 feature와 dynamic graph edge는 팀 탈락 위험 예측에 기여하는가? | DynamicSTGNN edge ablation 및 phase-wise evaluation으로 답변 가능 |

---

## 3. 선행 연구와 우리의 차별점

선행 연구는 PUBG 플레이어/팀의 행동을 FightScore, ActiveScore, 전략 archetype 등으로 요약해 생존 또는 성과를 분석하는 방향에 가깝다. 이 접근은 유용하지만 다음 한계가 있다.

- 행동 유형 또는 집계 점수 중심이라 팀이 놓인 공간 환경을 직접 모델링하지 않는다.
- 안전구역 축소, phase별 이동 방향, 경계 거리, 주변 적 밀도 같은 동적 공간 조건을 충분히 표현하지 않는다.
- 팀 간 관계가 시간에 따라 바뀌는 dynamic graph 문제로 다뤄지지 않는다.

우리의 novelty는 "모델만 GNN으로 바꿨다"가 아니다. 핵심은 다음이다.

> PUBG telemetry를 STDB로 재구성하면 팀의 이동 궤적, 안전구역 노출, 전투 사건, 팀 간 근접/교전/공간 압박 관계를 phase별 dynamic graph로 표현할 수 있고, 이 시공간 관계는 팀 탈락 위험 예측에 실제로 기여한다.

---

## 4. 데이터 수집 및 가용량

현재 raw telemetry 수집량과 STDB 변환 상태는 다음과 같다.

| 항목 | 수량 |
|---|---:|
| raw fetched | 27,491 matches |
| base STDB 변환 완료 | 21,833 matches |
| base STDB with team positions | 21,830 matches |
| phase feature 완료 | 21,830 matches |
| target label 완료 | 21,824 matches |
| edge cache 완료 | 21,830 matches |
| graph-eligible (`N_max=128`) | 21,824 matches |

상세 가용량 CSV:

- `docs/DATA_AVAILABILITY_BY_DOMAIN.md`
- `model_stgcn/checkpoints/domain_runs/data_availability_summary.csv`

초기 domain edge ablation은 TPP squad 5개 맵을 대상으로 수행했다. 최종 논문용 main/edge ablation은 full graph-eligible population 21,824 matches 기준으로 재실행했다.

| Split | Matches |
|---|---:|
| Train | 17,460 |
| Validation | 2,182 |
| Test | 2,182 |

Full-DB aligned test comparison uses 134,245 common target-defined team-phase cells.

---

## 5. STDB 재설계 흐름

### 5.1 Raw telemetry에서 STDB로

초기 raw telemetry는 `pubg.*` schema에 event log 형태로 저장되어 있었다. 이를 연구 질문에 맞게 `stdb.*` schema로 재구성했다.

| STDB 객체 | 의미 |
|---|---|
| `stdb.matches`, `stdb.teams` | match/team metadata |
| `stdb.moving_objects` | player 및 team centroid 위치 시계열 |
| `stdb.dynamic_regions` | safe zone / poison zone의 동적 공간 영역 |
| `stdb.spatial_events` | kill, damage, groggy, landing, loot 등 공간 사건 |
| `stdb.team_proximity` | 시점별 팀 간 거리 및 encounter 정보 |

공간 데이터 처리는 PostgreSQL + PostGIS 기반으로 수행했다. 위치 좌표는 게임 내 좌표를 meter 단위로 변환했고, geometry 컬럼은 GiST index, 시간/매치 key는 BRIN 또는 B-tree index를 사용했다.

### 5.2 Phase 복구

초기 문제는 `pubg.telem_phase_changes.elapsed_time`이 NULL인 경우가 많아 phase index가 제대로 채워지지 않았다는 점이었다.

해결 방식:

- `event_time - match_start_time`으로 elapsed seconds를 재계산
- `stdb.dynamic_regions.phase_idx`를 복구
- match별 phase 시작/종료 구간을 정의

이 작업 이후 `(match_id, team_id, phase_idx)` 단위 feature 설계가 가능해졌다.

---

## 6. Phase 단위 Feature 설계

### 6.1 Trajectory feature

팀의 phase별 이동 궤적을 `LINESTRING`으로 만들고, 이동 경로의 모양을 정량화했다.

| Feature | 의미 |
|---|---|
| `path_length_m` | phase 내 실제 이동 거리 |
| `displacement_m` | 시작점과 끝점의 직선 거리 |
| `sinuosity` | 우회 정도, path/displacement |
| `avg_speed_mps` | 평균 이동 속도 |
| `rotation_timing` | 주요 이동이 phase 초반/후반 중 언제 발생했는가 |
| `safe_alignment` | 이동 방향이 안전구역 방향과 얼마나 정렬되는가 |
| `zone_entry_time` | 다음 안전구역 진입 시점 |
| `outside_safe_duration_s` | 안전구역 밖 누적 시간 |
| `boundary_distance_slope` | 경계 거리 변화 추세 |

### 6.2 Exposure feature

단순 점 단위 평균이 아니라 경로 전체가 어떤 위험에 노출됐는지를 측정했다.

| Feature | 의미 |
|---|---|
| `mean_dist_to_boundary_m` | 안전구역 경계까지 평균 거리 |
| `min_dist_to_boundary_m` | 안전구역 경계까지 최소 거리 |
| `outside_safe_ratio` | phase 중 안전구역 밖에 있던 비율 |
| `avg_enemy_density_100m` | 100m 이내 평균 적 팀 수 |
| `min_nearest_enemy_m` | 가장 가까운 적 팀 거리 |
| `path_near_enemy_ratio` | 경로 중 적 100m 이내에 있던 비율 |
| `recent_damage_taken` | phase 내 받은 damage |
| `combat_zone_pass_count` | 교전 지역 통과 횟수 |

### 6.3 Combat / loot / state feature

PUBG 도메인 이해를 반영해 전투력, 피해, 아이템 준비도, 팀 상태도 포함했다.

| Feature group | 예시 |
|---|---|
| team state | alive players, team spread, health slope, vehicle ratio |
| combat | damage dealt/taken, kills, groggy, combat initiated |
| loot/readiness | pickup, equip, use, weapon/ammo/armor/heal/boost count |
| global phase | phase duration, alive teams, safe radius, zone gap |

현재 phase-level tensor의 node feature dimension은 `66`이다.

---

## 7. 라벨과 예측 문제 정의

가장 중요한 설계 수정은 라벨을 명확히 한 것이다. 기존의 최종 등수 회귀 중심 모델은 연구 주제와 덜 맞았다. 현재 main task는 다음과 같다.

> `X(phase <= k)`를 보고, team이 다음 phase 전에 탈락할 확률을 예측한다.

정의:

- 입력 시점: `cut_phase = k`
- 예측 대상: `hazard_next`
- 라벨: phase k 시작 시 살아 있던 팀이 다음 phase로 넘어가기 전에 탈락하면 1, 아니면 0
- 이미 탈락한 팀은 `target_mask=False`
- split은 match 단위로 수행해 같은 match의 팀이 train/test에 동시에 들어가지 않게 했다.

모델 출력:

| Output | Shape | 역할 |
|---|---|---|
| hazard logits | `(B, T-1, N)` | main task, next-phase elimination hazard |
| final rank | `(B, N)` | auxiliary regression |
| top-k logits | `(B, N, 2)` | auxiliary top5/top10 classification |

---

## 8. 노드와 엣지 정의

### 8.1 Node

노드는 팀이다.

| 항목 | 정의 |
|---|---|
| Node | team |
| Time step | phase |
| Node feature | team state + global phase + trajectory + exposure + combat + loot |
| Padding | `N_max = 128` |
| Mask | `team_mask`, `node_mask`, `target_mask` |

### 8.2 Edge

엣지는 phase별로 동적으로 변하는 팀 간 관계다.

현재 edge channel:

| Edge type | 의미 |
|---|---|
| `proximity` | 같은 phase에서 가까이 있는 팀 간 공간 근접 관계 |
| `combat` | damage/kill/groggy 등 전투 사건으로 연결된 관계 |
| `copressure` | 같은 공간 압박, 안전구역 압력, 다팀 밀집 상황에 놓인 관계 |

Edge ablation은 다음 네 가지로 수행했다.

| Edge mode | 의미 |
|---|---|
| `none` | graph edge 없음, sequence-only neural baseline |
| `proximity` | 근접 edge만 사용 |
| `proximity_combat` | 근접 + 전투 edge 사용 |
| `full` | 근접 + 전투 + copressure edge 사용 |

---

## 9. 모델 설계

현재 모델은 `DynamicSTGNN`이다.

입력:

```text
X: (B, T=8, N=128, F=66)
A: (B, T=8, E=3, N=128, N=128)
```

구조:

```text
Node embedding
  -> ST block x 2
       Temporal causal gated conv
       Heterogeneous spatial GCN
       Temporal causal gated conv
  -> hazard head
  -> rank auxiliary head
  -> top-k auxiliary head
```

중요한 구현 포인트:

- temporal convolution은 causal padding을 사용해 미래 phase 정보 누수를 막는다.
- edge channel별 spatial aggregation 후 learnable edge weight로 결합한다.
- `none` mode에서도 residual/temporal path는 유지되어 sequence-only neural baseline 역할을 한다.

관련 코드:

- `model_stgcn/dyn_stgnn.py`
- `model_stgcn/train_phase.py`
- `model_stgcn/eval_phase.py`
- `model_stgcn/run_domain_training.py`
- `model_stgcn/phase_wise_eval.py`

---

## 10. 전체 Map-Wise Edge Ablation 결과

실험 설정:

- 대상: TPP squad 5개 맵
- 각 맵별로 별도 학습
- 각 맵별 edge mode 4개 학습
- train/val/test는 match 단위 split
- checkpoint output: `model_stgcn/checkpoints/domain_runs_expanded_e12`

| Map | Best model | Hazard AUC | Hazard AP | `full` vs `none` AUC delta |
|---|---|---:|---:|---:|
| Baltic_Main | full | 0.7759 | 0.6444 | +0.0106 |
| Tiger_Main | full | 0.7605 | 0.6213 | +0.0059 |
| Desert_Main | full | 0.7458 | 0.5674 | +0.0063 |
| Neon_Main | proximity_combat | 0.7263 | 0.5511 | +0.0079 |
| Savage_Main | full | 0.7457 | 0.6092 | +0.0119 |

`full` 모델은 5개 맵 모두에서 `none` baseline보다 AUC/AP가 개선되었다.

평균 개선:

| Metric | `full` vs `none` |
|---|---:|
| Hazard AUC | +0.0085 |
| Hazard AP | +0.0122 |
| Loss | -0.0062 |

맵 단위 Wilcoxon:

| Metric | p-value |
|---|---:|
| AUC improvement | 0.0312 |
| AP improvement | 0.0312 |
| Loss reduction | 0.0312 |

결론:

> Dynamic edge는 단순 장식이 아니라 실제 성능 개선을 만든다. 특히 full edge는 평균적으로 가장 안정적인 성능을 보인다.

Neon_Main에서는 `proximity_combat`이 최고였으므로, copressure edge가 모든 맵에서 항상 최고라고 과장하면 안 된다. 다만 Neon에서도 `full`은 `none`보다 개선되었다.

---

## 11. Phase-Wise Evaluation 결과

다음 단계로 전체 성능을 phase별로 쪼개어 평가했다. `cut_phase k`는 phase k까지 보고 다음 phase 탈락 위험을 예측한다는 뜻이다.

산출물:

- `model_stgcn/checkpoints/domain_runs_expanded_e12/phase_wise_eval/phase_predictions.csv`
- `model_stgcn/checkpoints/domain_runs_expanded_e12/phase_wise_eval/phase_metrics.csv`
- `model_stgcn/checkpoints/domain_runs_expanded_e12/phase_wise_eval/phase_deltas.csv`
- `model_stgcn/checkpoints/domain_runs_expanded_e12/phase_wise_eval/phase_metrics_aggregate.csv`
- `model_stgcn/checkpoints/domain_runs_expanded_e12/phase_wise_eval/phase_deltas_aggregate.csv`
- `model_stgcn/checkpoints/domain_runs_expanded_e12/phase_wise_eval/phase_bce_wilcoxon.csv`

총 prediction row:

- 276,396 rows

Aggregate `full` vs `none` 결과:

| cut_phase | n | pos_rate | AUC delta | AP delta | BCE delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 20,851 | 0.245 | +0.0117 | +0.0158 | -0.0094 |
| 2 | 16,233 | 0.221 | +0.0078 | +0.0145 | -0.0045 |
| 3 | 13,007 | 0.301 | +0.0115 | +0.0145 | -0.0066 |
| 4 | 9,422 | 0.428 | +0.0101 | +0.0077 | -0.0057 |
| 5 | 5,512 | 0.435 | +0.0068 | +0.0020 | -0.0049 |
| 6 | 2,998 | 0.454 | +0.0024 | +0.0012 | -0.0025 |
| 7 | 1,076 | 0.394 | +0.0066 | +0.0135 | -0.0048 |

해석:

- edge의 이득은 phase 1~4에서 가장 강하다.
- phase 1~4는 5개 맵 모두 또는 거의 모두에서 `full`이 `none`보다 낫다.
- phase 5 이후에는 표본 수가 줄고 생존자 편향이 커져 효과가 약해진다.
- late-game은 stochasticity가 크므로 조심스럽게 해석해야 한다.

도메인-phase 단위 winner count:

| Metric | full | proximity_combat | proximity | none |
|---|---:|---:|---:|---:|
| AUC best | 22 | 5 | 3 | 5 |
| AP best | 19 | 6 | 5 | 5 |
| BCE best | 22 | 8 | 2 | 3 |

Phase별 `full` vs `none` BCE Wilcoxon:

| cut_phase | 유의한 맵 수 | 평균 BCE 변화 |
|---:|---:|---:|
| 1 | 5/5 | -0.0103 |
| 2 | 5/5 | -0.0048 |
| 3 | 5/5 | -0.0064 |
| 4 | 5/5 | -0.0065 |
| 5 | 2/5 | -0.0035 |
| 6 | 1/5 | +0.0001 |
| 7 | 3/5 | -0.0083 |

결론:

> Dynamic graph edge의 효과는 초중반 phase에서 가장 일관되며, 이는 초반 landing/looting, rotation, zone 압박, 교전 관계가 탈락 위험에 중요한 시공간 신호라는 주장을 지지한다.

---

## 12. 현재까지의 핵심 발견

### 12.1 STDB 표현 가능성

PUBG telemetry는 다음 세 가지 STDB 객체로 자연스럽게 재구성된다.

- 이동 객체: team/player trajectory
- 동적 공간 영역: safe zone, poison zone
- 공간 사건: kill, damage, groggy, landing, loot

따라서 RQ1에 대해 "표현 가능하다"는 답을 줄 수 있다.

### 12.2 Phase별 시공간 패턴

초기 분석에서 top team과 bot team은 phase별 trajectory/exposure 패턴이 다르게 나타났다.

- 상위 팀은 초반 이동량과 safe alignment가 높다.
- 하위 팀은 초반 밀집/교전 위험에 더 자주 노출된다.
- phase 2 이후 rotation과 안전구역 대응이 생존 신호로 나타난다.
- 전투는 phase에 따라 부호가 달라진다. 초반 전투는 위험 신호, 후반 전투는 생존자 간 경쟁 신호일 수 있다.

### 12.3 시공간 edge의 예측 기여

5개 맵 전체에서 `full` edge model은 `none`보다 성능이 개선되었다. 특히 phase-wise evaluation에서 phase 1~4의 개선이 강하다.

따라서 RQ3에 대해 다음처럼 답할 수 있다.

> 팀 간 proximity, combat, copressure edge는 phase별 탈락 위험 예측에 기여한다. 이 효과는 초중반 phase에서 가장 뚜렷하다.

---

## 13. 논문화할 때의 주장 구조

논문 결과 섹션은 다음 흐름이 가장 자연스럽다.

1. STDB 구축 결과
   - raw telemetry를 이동 객체, 동적 공간 영역, 공간 사건으로 재구성
   - phase-level feature 및 edge cache 생성

2. Feature ablation
   - prior style score baseline보다 STDB feature가 유의미하게 낫다
   - conventional telemetry에 STDB spatial feature를 추가하면 성능 향상

3. Dynamic graph ablation
   - no graph, proximity, proximity+combat, full 비교
   - full edge가 평균적으로 가장 안정적으로 좋음

4. Phase-wise evaluation
   - edge 효과가 phase 1~4에서 강함
   - late phase는 표본 감소와 생존자 편향 때문에 조심스럽게 해석

5. Novelty statement
   - "PUBG 생존 예측은 단순 행동 유형 분류 문제가 아니라, 축소되는 안전구역 안에서 팀들이 형성하는 동적 시공간 관계망의 문제"라고 주장

---

## 14. 현재 한계

| 한계 | 설명 |
|---|---|
| 단일 실험 축 | 이번 최종 ablation은 TPP squad 5개 맵 중심 |
| late phase 표본 부족 | phase 5 이후부터 생존 팀 수가 줄어 phase-wise 통계가 약해짐 |
| Neon 예외 | Neon에서는 `proximity_combat`이 `full`보다 좋음 |
| calibration 검토 부족 | AUC/AP 중심 결과는 좋지만 calibration은 추가 검토 필요 |
| 논문용 시각화 미완 | phase-wise heatmap/line plot 정리가 필요 |

---

## 15. 바로 다음 작업

논문화 직전까지 가기 위한 다음 작업은 다음 순서가 좋다.

1. Phase-wise result visualization
   - phase별 AUC/AP delta line plot
   - map × phase heatmap
   - `full` vs `none` BCE improvement heatmap

2. 논문 결과 테이블 정리
   - Table 1: 데이터셋/도메인 가용량
   - Table 2: map-wise edge ablation
   - Table 3: phase-wise full vs none delta
   - Table 4: prior baseline vs STDB feature vs DynamicSTGNN

3. 선행 연구 대비 novelty 문단 작성
   - prior score/archetype approach의 한계
   - STDB + dynamic graph의 장점
   - phase-wise evidence로 주장 보강

4. 추가 실험 선택
   - FPP squad로 generalization 확인
   - solo/duo는 별도 모델 또는 별도 분석으로 분리
   - calibration curve와 Brier score 추가

---

## 16. 중요 산출물 경로

### 설계/문서

- `docs/DESIGN.md`
- `docs/RQ1_STDB_REDESIGN.md`
- `docs/MODEL_RESULTS.md`
- `docs/PROJECT_FLOW_SUMMARY.md`

### 데이터 가용량

- `model_stgcn/checkpoints/domain_runs/data_availability_summary.csv`

### Map-wise edge ablation

- `model_stgcn/checkpoints/domain_runs_expanded_e12/domain_edge_ablation_summary.csv`
- `model_stgcn/checkpoints/domain_runs_expanded_e12/domain_edge_ablation_deltas.csv`
- `model_stgcn/checkpoints/domain_runs_expanded_e12/domain_edge_ablation_wilcoxon.json`

### Phase-wise evaluation

- `model_stgcn/checkpoints/domain_runs_expanded_e12/phase_wise_eval/phase_predictions.csv`
- `model_stgcn/checkpoints/domain_runs_expanded_e12/phase_wise_eval/phase_metrics.csv`
- `model_stgcn/checkpoints/domain_runs_expanded_e12/phase_wise_eval/phase_deltas.csv`
- `model_stgcn/checkpoints/domain_runs_expanded_e12/phase_wise_eval/phase_metrics_aggregate.csv`
- `model_stgcn/checkpoints/domain_runs_expanded_e12/phase_wise_eval/phase_deltas_aggregate.csv`
- `model_stgcn/checkpoints/domain_runs_expanded_e12/phase_wise_eval/phase_bce_wilcoxon.csv`

### 핵심 코드

- `stdb/queries.py`
- `model_stgcn/dyn_stgnn.py`
- `model_stgcn/train_phase.py`
- `model_stgcn/eval_phase.py`
- `model_stgcn/run_domain_training.py`
- `model_stgcn/phase_wise_eval.py`

---

## 17. 최종 요약

현재 프로젝트는 다음 단계까지 도달했다.

1. PUBG raw telemetry를 STDB로 재구성했다.
2. phase 단위 trajectory/exposure/combat/loot/state feature를 만들었다.
3. 팀을 노드로, proximity/combat/copressure를 동적 edge로 정의했다.
4. main target을 next-phase elimination hazard로 명확히 고정했다.
5. DynamicSTGNN을 학습하고, 5개 TPP squad 맵에서 edge ablation을 수행했다.
6. `full` edge model은 모든 맵에서 no-graph baseline보다 AUC/AP가 개선되었다.
7. phase-wise evaluation 결과, edge 효과는 phase 1~4에서 가장 강하게 나타났다.

따라서 현재 논문의 핵심 주장은 다음 한 문장으로 정리된다.

> PUBG telemetry를 STDB로 재구성하고 phase별 dynamic graph로 모델링하면, 팀 간 시공간 관계가 탈락 위험 예측에 유의미한 신호로 작동한다. 특히 초중반 phase에서 proximity, combat, copressure edge의 효과가 가장 뚜렷하다.
