# PUBG Feature Design V2

작성일: 2026-06-12

목적: 기존 short paper의 feature/graph 설계를 확장해, 공식 raw telemetry baseline, STDB 표현, network 표현, habitat-inspired 표현을 같은 `team-phase` 단위에서 비교할 수 있는 구현 가능한 feature schema를 정의한다.

이 문서는 `docs/FEATURE_INVENTORY_AND_ABLATION_PLAN.md`의 다음 단계다. Inventory가 "무엇이 있었는가"를 정리했다면, 이 문서는 "이제 무엇을 어떻게 만들 것인가"를 정의한다.

---

## 1. 핵심 진단

기존 연구는 graph를 만들었지만, graph를 충분한 network representation으로 쓰지는 못했다.

정확한 진단은 다음이다.

1. Current DynamicSTGNN은 adjacency message passing을 사용한다.
2. 그러나 node feature에는 degree, strength, PageRank, betweenness, community, bridge score 같은 network role feature가 없다.
3. No-graph baseline도 이미 `avg_enemy_density_100m`, `min_nearest_enemy_m`, `combat_partner_count`, damage taken/dealt 같은 graph-like summary를 받는다.
4. Combat edge는 기본 E=3 구조에서 attacker row가 victim feature를 받는 방향이라, 피해팀이 공격자의 상태를 직접 받지 못한다.
5. 따라서 현재 graph gain은 "관계 행렬을 추가한 작은 이득"이지, "network representation family의 효과"로 해석하면 안 된다.

새 설계의 목표:

> 동일한 official PUBG telemetry를 raw aggregate, prior-paper score, STDB spatial, network role, habitat pressure, graph relation으로 재구성하고, 어떤 표현 family가 next-phase elimination hazard를 가장 잘 보존하는지 비교한다.

---

## 2. Canonical Unit and Risk Set

모든 feature는 아래 key로 정렬한다.

```text
match_id
team_id
phase_idx
```

Main task:

```text
Input:  observations through cut phase k
Label:  hazard_next = 1 if team is eliminated before or by the next phase interval
Risk:   target_defined = true
Phases: k = 1..7 for the canonical DynamicSTGNN comparison
```

필수 원칙:

- 모든 family는 같은 match split을 사용한다.
- 모든 ablation은 같은 test risk cells에서 평가한다.
- feature 계산은 cut phase `k`까지 관측 가능한 정보만 사용한다.
- full-match network, final rank, winner, match end result는 main input으로 금지한다.

---

## 3. Feature Family Overview

Executable manifest:

```text
model_stgcn/feature_manifests/pubg_representation_v2.json
```

이 manifest를 V2 feature family의 source-of-truth로 둔다. 문서는 연구 설계와 정의를 설명하고, manifest는 ablation 코드가 읽을 수 있는 column contract를 제공한다.

| ID | Family | 구현 상태 | 역할 |
| --- | --- | --- | --- |
| `F0` | official raw phase aggregate | 새 view 필요 | 공식 raw telemetry baseline |
| `F1` | prior paper raw inputs | 기존 코드 있음 | Battleground paper 입력 feature 대응 |
| `F2` | prior paper score summary | 기존 코드 있음 | FightScore/ActiveScore/strategy type |
| `F3` | legacy axis recast | 선택 구현 | 초기 5축 feature와 비교 |
| `F4` | current conventional | 기존 view 있음 | short paper conventional aggregate |
| `F5` | current STDB spatial | 기존 view 있음 | trajectory/exposure representation |
| `F6` | network phase role | 새 table 필요 | graph를 node feature로 요약 |
| `F7` | habitat pressure | 새 table 필요 | 축소 공간 경쟁 압박 |
| `F8` | temporal/patch proxy | 새 view 필요 | temporal drift robustness |

Graph family:

| ID | Family | 구현 상태 |
| --- | --- | --- |
| `G0` | no graph | 있음 |
| `G1` | current STDB graph: proximity/combat/copressure | 있음 |
| `G2` | directed combat graph: add `combat_in` | capability 있음, full rerun 필요 |
| `G3` | network graph | 설계 필요 |
| `G4` | habitat competition graph | 설계 필요 |
| `G5` | shuffled/randomized graph controls | control code 수정 필요 |

Column-contract rule:

- `F0`는 official telemetry에서 phase 안에 직접 집계 가능한 raw value만 포함한다.
- `F4`는 기존 short-paper conventional representation이다. `F0`와 겹치지만 같은 family로 취급하지 않는다.
- `F5`는 STDB 질의가 만든 spatial/trajectory/exposure feature다.
- `F6`는 graph adjacency 자체가 아니라, edge cache에서 계산한 node-level network role feature다.
- `F7`는 biological equivalence가 아니라 habitat-inspired shrinking-space pressure다.
- 모든 ablation 코드는 manifest에 정의된 family를 가져오되, 현재 view에 아직 없는 컬럼은 명시적으로 missing으로 보고해야 한다.

Current DB validation against `stdb.v_team_phase_feature_long` after the F5 view fix:

| Family | Present / Wanted | 상태 |
| --- | ---: | --- |
| `F0 official raw phase` | 52 / 85 | raw center, landing, assist/suicide, 일부 match-state 컬럼 추가 필요 |
| `F1 prior paper raw` | 12 / 12 | 현재 view로 실행 가능 |
| `F2 prior score summary` | 1 / 5 | `feature_ablation.py`에서 train split 기준으로 계산 |
| `F4 current conventional` | 57 / 57 | 현재 view로 실행 가능 |
| `F5 STDB spatial` | 18 / 18 | 현재 view로 실행 가능 |
| `F6 network role` | 0 / 43 | 새 `team_phase_network_features` 필요 |
| `F7 habitat pressure` | 0 / 32 | 새 `team_phase_habitat_features` 필요 |
| `F8 temporal/patch proxy` | 0 / 5 | 새 temporal metadata view 필요 |

검증 도구:

```bash
PG_PASSWORD=... python3 tools/check_feature_manifest.py --pretty
```

---

## 4. F0 Official Raw Phase Aggregate

### 4.1 정의

`F0`는 공식 PUBG telemetry에서 직접 나오는 event/state 값을 phase interval 안에서 집계한 baseline이다. STDB의 spatial relation query나 graph algorithm은 넣지 않는다.

중요한 구분:

| 항목 | F0에 포함 | F5/STDB로 분리 |
| --- | --- | --- |
| raw damage/item/health/vehicle count | yes | no |
| safe/poison zone radius and center values | yes | no |
| team centroid start/end and simple speed | yes, raw position summary | optional |
| distance to safe-zone boundary | no | yes |
| safe-zone alignment / entry timing | no | yes |
| nearest enemy / enemy density | no | yes |
| graph centrality/community | no | F6 |
| habitat area per alive team / refuge scarcity | no | F7 |

### 4.2 구현 view

```text
stdb.v_team_phase_official_raw_features
```

Primary key:

```text
match_id, team_id, phase_idx
```

Source tables:

```text
pubg.matches
pubg.telem_match_start
pubg.telem_phase_changes
pubg.telem_game_states
pubg.telem_positions
pubg.telem_damage
pubg.telem_groggy
pubg.telem_kills
pubg.telem_item_pickup
pubg.telem_item_equip
pubg.telem_item_use
pubg.telem_parachute_landing
```

현재 코드에서는 `stdb.match_phase_global`, `stdb.team_phase_state`, `stdb.team_phase_combat`, `stdb.team_phase_loot`이 이 역할의 상당 부분을 이미 수행한다. 단, 논문상 baseline을 명확히 하려면 `F0` view를 별도로 만들고 "official raw aggregate"라고 이름 붙이는 것이 좋다.

### 4.3 Column design

Match/meta:

```text
map_name
game_mode
match_type
match_team_size
is_custom
weather_id
camera_view_type
created_month
created_week
```

Phase/population:

```text
phase_idx
phase_start_s
phase_end_s
phase_duration_s
avg_alive_teams
start_alive_teams
end_alive_teams
avg_alive_players
start_alive_players
end_alive_players
```

Raw zone state:

```text
safe_center_x_start
safe_center_y_start
safe_center_x_end
safe_center_y_end
safe_radius_start_m
safe_radius_end_m
safe_radius_delta_m
poison_center_x_start
poison_center_y_start
poison_center_x_end
poison_center_y_end
poison_radius_start_m
poison_radius_end_m
poison_radius_delta_m
red_zone_active_ratio
black_zone_active_ratio
```

Raw team state:

```text
team_size
position_rows
alive_players_start
alive_players_end
alive_players_min
alive_players_avg
mean_health_avg
min_health_min
health_slope
team_centroid_x_start
team_centroid_y_start
team_centroid_x_end
team_centroid_y_end
team_displacement_raw_m
team_avg_speed_raw_mps
team_spread_avg_m
team_spread_max_m
vehicle_ratio
vehicle_speed_mean
vehicle_speed_max
```

Raw combat:

```text
damage_dealt
damage_taken
damage_events_dealt
damage_events_taken
groggy_dealt
groggy_taken
kills_dealt
kills_taken
assists_count
suicide_count
distinct_opponents_damaged
distinct_opponents_damaged_by
combat_partner_count
first_damage_dealt_s
first_damage_taken_s
combat_initiated
```

Raw loot/readiness:

```text
pickup_count
equip_count
use_count
unique_item_count
weapon_pickup_count
weapon_equip_count
ammo_pickup_count
armor_pickup_count
throwable_pickup_count
heal_use_count
boost_use_count
carepackage_pickup_count
lootbox_pickup_count
```

Landing:

```text
landing_t
landing_x
landing_y
landing_distance
landing_team_spread_m
```

Excluded from F0 main input:

```text
rosters.rank
rosters.won
telem_match_end.game_results
victim_rank
any full-match aggregate
```

---

## 5. F1/F2 Prior Paper Features

Reference paper:

```text
Are you a Hunter or Camper?
FightScore / ActiveScore / Hunter-Sniper-Traveler-Camper
```

### F1 prior paper raw

Original feature mapping:

| Original | Our team-phase feature |
| --- | --- |
| `phase_attack_count` | `damage_events_dealt` |
| `phase_kill_count` | `kills_dealt` |
| `phase_total_damage` | `damage_dealt` |
| `phase_assist_count` | `assists_count` if rebuilt, otherwise `groggy_dealt` and `distinct_opponents_damaged` |
| `distance` | `team_displacement_raw_m`, `team_avg_speed_raw_mps`, or current `path_length_m` |
| `phase_max_density` | current `avg_enemy_density_100m`, but mark as spatial-derived |
| `phase_item_count` | `pickup_count`, `unique_item_count` |
| `carepackage_use` | `carepackage_pickup_count` |

Implementation note:

- 원 논문 baseline을 엄밀히 맞추려면 `assists_count`를 `telem_kills.assists_cnt`에서 phase/team 단위로 새로 집계한다.
- `phase_max_density`는 raw event가 아니라 spatial density query이므로, F1에는 넣되 `prior-derived spatial input`으로 표시한다.

### F2 prior paper summary

Existing implementation in `model_stgcn/feature_ablation.py`:

```text
fight_score
active_score
strategy_type
type_survival_rate
phase_idx
```

Rules:

- `type_survival_rate`는 train split에서만 fit한다.
- test fold 전체 survival rate를 써서 만들면 leakage다.

---

## 6. F4/F5 Current STDB Features

현재 short paper의 기존 feature는 유지하되, 이름과 역할을 명확히 한다.

### F4 current conventional

Source:

```text
stdb.v_team_phase_feature_long
```

Groups:

```text
global phase context
team state
combat aggregate
loot/readiness
```

F4는 F0와 상당히 겹친다. 하지만 F4는 current short-paper pipeline의 conventional representation이고, F0는 새 논문의 official raw baseline이다. 논문에서는 둘을 같은 것으로 취급하지 않는다.

### F5 current STDB spatial

Source:

```text
stdb.team_phase_trajectory
stdb.team_phase_exposure
```

Trajectory:

```text
path_length_m
displacement_m
sinuosity
avg_speed_mps
rotation_timing
safe_alignment
zone_entry_time
outside_safe_duration_s
boundary_distance_slope
```

Exposure:

```text
mean_dist_to_boundary_m
min_dist_to_boundary_m
outside_safe_ratio
avg_enemy_density_100m
min_nearest_enemy_m
path_near_enemy_ratio
recent_damage_taken
combat_zone_pass_count
```

Known issue:

```text
nearby_enemy_combat_events_200m
third_party_exposure_count
lootbox_pickup_count
```

These have near-zero nonzero coverage in current target rows. Rebuild or exclude from main manifest.

---

## 7. F6 Network Phase Role Features

### 7.1 목적

F6는 이번 재설계의 핵심이다. 기존 DynamicSTGNN은 graph adjacency만 사용했다. F6는 같은 graph를 node-level network role feature로 요약한다.

즉:

```text
Graph adjacency = who is connected to whom
Network feature = what role this team occupies in the graph
```

### 7.2 구현 table

```text
stdb.team_phase_network_features
```

Primary key:

```text
match_id, team_id, phase_idx
```

Main source:

```text
stdb.team_phase_edge_cache
```

Use raw edge-cache weights before DynamicSTGNN row-normalization.

### 7.3 Tier 1 features: stable SQL metrics

Combat local:

```text
net_combat_out_degree
net_combat_in_degree
net_combat_total_degree
net_combat_out_strength
net_combat_in_strength
net_combat_total_strength
net_combat_balance
net_combat_unique_targets
net_combat_unique_attackers
net_combat_reciprocal_pairs
net_combat_reciprocity_ratio
net_combat_out_strength_per_min
net_combat_in_strength_per_min
```

Definitions:

```text
out_degree       = number of teams this team attacked
in_degree        = number of teams that attacked this team
out_strength     = sum raw combat edge weight from team to others
in_strength      = sum raw combat edge weight from others to team
balance          = (out_strength - in_strength) / (out_strength + in_strength + eps)
reciprocal_pairs = number of opponent teams with both in and out combat edges
```

Proximity local:

```text
net_prox_degree_100m
net_prox_degree_200m
net_prox_degree_500m
net_prox_strength_decay
net_prox_mean_distance_m
net_prox_min_distance_m
net_prox_top1_weight
net_prox_top3_strength
net_prox_isolation_m
net_prox_neighbor_entropy
```

Definitions:

```text
degree_100m/200m/500m = count of enemy teams with avg_distance_m below threshold
strength_decay        = sum exp(-avg_distance / 200m)
neighbor_entropy      = entropy of normalized proximity weights
isolation_m           = min avg_distance_m, high value means isolated
```

Co-pressure local:

```text
net_copress_degree
net_copress_strength
net_copress_max_weight
net_copress_mean_ratio
net_copress_shared_pressure_count
net_copress_top1_weight
```

Cumulative through phase `1..k`:

```text
cum_combat_out_degree
cum_combat_in_degree
cum_combat_out_strength
cum_combat_in_strength
cum_combat_total_strength
cum_combat_reciprocal_pairs
cum_unique_targets
cum_unique_attackers
cum_copress_strength
cum_prox_close_exposure_200m
```

Phase-normalized variants:

```text
net_combat_out_strength_per_alive_team
net_combat_in_strength_per_alive_team
net_prox_degree_200m_per_alive_team
net_copress_degree_per_alive_team
```

### 7.4 Tier 2 features: graph algorithm metrics

Add after Tier 1 is validated.

```text
net_combat_pagerank
net_combat_betweenness
net_combat_closeness
net_combat_eigenvector
net_louvain_community_id
net_community_size
net_within_degree
net_across_degree
net_across_share
net_community_embeddedness
net_battle_bridge_score
```

Cautions:

- Phase graph can be sparse in late phases.
- Betweenness and Louvain can be noisy for small graphs.
- These must be calculated phase-local or cumulative-through-phase, never full-match.

### 7.5 Network leakage rules

Allowed:

```text
phase k network built only from phase k observations
cumulative network built from phases 1..k
```

Forbidden:

```text
full-match centrality
full-match community
degree or PageRank using future combat
survival-time-normalized metrics that use final survival time as denominator
```

---

## 8. F7 Habitat-Inspired Pressure Features

### 8.1 목적

F7는 "축소되는 playable space가 팀에게 만드는 생존 압박"을 정량화한다.

이 frame은 biological equivalence가 아니라 analytical analogy다. 논문 표현은 `habitat-inspired shrinking-space pressure` 또는 `playable-space competition`을 사용한다.

### 8.2 구현 table

```text
stdb.team_phase_habitat_features
```

Primary key:

```text
match_id, team_id, phase_idx
```

Sources:

```text
stdb.dynamic_regions
stdb.v_match_phase_bounds
stdb.moving_objects
stdb.team_proximity
stdb.team_phase_state
stdb.match_phase_global
```

### 8.3 Global shrinking-space features

```text
hab_safe_area_start_m2
hab_safe_area_end_m2
hab_safe_area_delta_m2
hab_loss_ratio
hab_shrink_rate_m2ps
hab_alive_teams
hab_safe_area_per_alive_team
hab_crowding_pressure
hab_center_shift_m
hab_center_shift_per_radius
hab_zone_gap_start_m
hab_zone_gap_end_m
hab_zone_gap_pressure
```

Definitions:

```text
safe_area = pi * safe_radius^2
loss_ratio = 1 - safe_area_end / safe_area_start
crowding_pressure = alive_teams / safe_area_end
safe_area_per_alive_team = safe_area_end / alive_teams
center_shift_per_radius = center_shift / safe_radius_start
```

### 8.4 Team-level movement pressure

```text
hab_dist_to_safe_boundary_start_m
hab_dist_to_safe_boundary_end_m
hab_signed_boundary_distance_start_m
hab_signed_boundary_distance_end_m
hab_distance_to_target_safe_m
hab_required_speed_to_safe_mps
hab_observed_speed_mps
hab_migration_deficit_mps
hab_outside_safe_ratio
hab_edge_pressure_ratio
hab_rotation_delay_ratio
hab_safe_alignment_to_target
```

Definitions:

```text
distance_to_target_safe = distance from team centroid to target safe region if outside, else 0
required_speed = distance_to_target_safe / remaining_time
migration_deficit = required_speed - observed_speed
edge_pressure_ratio = fraction of snapshots outside safe zone or within boundary threshold
```

### 8.5 Local competition / refuge scarcity

Tier 1:

```text
hab_local_enemy_count_100m
hab_local_enemy_count_200m
hab_local_enemy_count_500m
hab_local_enemy_density_200m
hab_safe_entry_enemy_count_200m
hab_safe_entry_congestion
hab_refuge_scarcity_simple
```

Tier 2:

```text
hab_reachable_safe_area_m2
hab_reachable_safe_area_ratio
hab_reachable_enemy_count
hab_refuge_scarcity
hab_forced_convergence_score
```

Pragmatic first implementation:

```text
hab_refuge_scarcity_simple =
  hab_safe_entry_enemy_count_200m / (hab_safe_area_end_m2 + eps)
```

The full reachable-area version can be added after the first ablation.

### 8.6 Habitat leakage rules

Allowed:

- current phase safe/poison zone state observed by cut phase.
- announced target safe zone if it is already visible in telemetry at cut time.
- current team position and observed speed through cut phase.

Forbidden:

- future safe-zone centers not announced by cut time.
- future enemy positions.
- future combat.
- final rank or winner.

---

## 9. Graph Redesign

Current graph:

```text
proximity
combat
copressure
```

V2 graph conditions:

| Graph | Channels |
| --- | --- |
| `G0 no_graph` | none |
| `G1 current_stdb` | proximity, combat, copressure |
| `G2 directed_combat` | proximity, combat_out, combat_in, copressure |
| `G3 network_graph` | proximity threshold graph, combat role graph, reciprocal combat graph |
| `G4 habitat_graph` | forced convergence, refuge competition, reachable overlap |
| `G5 combined` | selected G2 + G4 channels |

Immediate correction:

- Run E=4 `combat_in` full-DB ablation.
- Fix `edge_sanity.py` so node shuffle is phase-local: active nodes should be `node_mask[b, t]`, not flattened `node_mask[b]`.

---

## 10. Model/Input Design

### 10.1 Feature-only models

Use HGB/LightGBM first.

Reason:

- feature family effect is easier to interpret.
- graph model changes are separated from representation changes.

### 10.2 Graph models

Node feature conditions should be fixed while graph changes.

Recommended graph experiment input:

```text
Node features = F0 + F5 + F6 + F7
Graph changes = G0/G1/G2/G4/G5
```

Also run:

```text
Node features = F0 + F5 only
Graph changes = G0/G1/G2
```

This checks whether graph gain disappears only because F6 already summarizes network role.

---

## 11. Ablation Matrix

### 11.1 Feature-only ablation

| Condition | Feature set | Question |
| --- | --- | --- |
| A0 | `F0` | How strong is official raw telemetry? |
| A1 | `F1` | How strong are prior paper raw ingredients? |
| A2 | `F2` | How much signal is lost by score summarization? |
| A3 | `F4` | How strong is current conventional representation? |
| A4 | `F5` | How strong is STDB spatial alone? |
| A5 | `F0 + F5` | Does STDB spatial add to raw? |
| A6 | `F0 + F6` | Does network role add to raw? |
| A7 | `F0 + F7` | Does habitat pressure add to raw? |
| A8 | `F0 + F5 + F6` | STDB + network |
| A9 | `F0 + F5 + F7` | STDB + habitat |
| A10 | `F0 + F6 + F7` | network + habitat |
| A11 | `F0 + F5 + F6 + F7` | full feature representation |

### 11.2 Graph ablation

| Condition | Node features | Graph |
| --- | --- | --- |
| B0 | `F0 + F5` | no graph |
| B1 | `F0 + F5` | current graph |
| B2 | `F0 + F5` | directed combat graph |
| B3 | `F0 + F5 + F6 + F7` | no graph |
| B4 | `F0 + F5 + F6 + F7` | directed combat graph |
| B5 | `F0 + F5 + F6 + F7` | habitat graph |
| B6 | `F0 + F5 + F6 + F7` | combined graph |
| B7 | same as B6 | shuffled graph control |

Important interpretation:

- If F6 gives large feature-only gain and graph gain shrinks, that is not failure. It means network structure is better captured as explicit node role features than as only adjacency message passing.
- If graph still adds after F6, then relation-specific neighbor states contain extra signal beyond role summaries.

---

## 12. Implementation Order

### Step 1. Fix graph-control correctness

Before making new graph claims:

```text
Fix edge_sanity phase-local node shuffle.
Run observed vs node_shuffle again.
```

### Step 2. Build F6 Tier 1 network features

Create:

```text
stdb.team_phase_network_features
```

Implement only stable SQL metrics first:

```text
combat in/out degree and strength
combat balance
reciprocity
proximity degree/strength/isolation
copressure degree/strength
cumulative 1..k combat/proximity/copressure metrics
```

### Step 3. Build F7 Tier 1 habitat features

Create:

```text
stdb.team_phase_habitat_features
```

Implement:

```text
safe area loss
crowding pressure
safe area per alive team
required movement speed
migration deficit
edge pressure
simple refuge scarcity
```

### Step 4. Create joined V2 view

```text
stdb.v_team_phase_feature_v2
```

Join:

```text
F0 official raw
F4 current conventional
F5 current STDB spatial
F6 network
F7 habitat
labels
```

### Step 5. Run feature-only ablation

Run A0--A11 with same split and same risk cells.

### Step 6. Run graph ablation

Run B0--B7 after F6/F7 feature-only results are stable.

---

## 13. First Coding Target

The first coding target should be `F6 Tier 1`, not habitat and not graph architecture.

Reason:

- It directly addresses the identified weakness: graph structure existed, network role features were absent.
- It can be built from existing `team_phase_edge_cache`.
- It is low leakage if restricted to phase-local and cumulative-through-phase.
- It gives the cleanest ablation: `F0 + F5` vs `F0 + F5 + F6`.

Recommended first deliverable:

```text
stdb/team_phase_network_features.sql or stdb/network_feature_etl.py
model_stgcn/feature_manifests/network_phase.json
feature-only HGB ablation including F6
```

---

## 14. Implementation Log

### 2026-06-12 — F6 Tier 1 구현 완료 (Step 2)

산출물:

```text
stdb/network_feature_etl.py            # 테이블 DDL + match-scoped resumable ETL
tests/test_network_features.py        # 불변식 5종 (아래)
model_stgcn/feature_ablation.py       # F6 merge + 신규 config 4종 + paired 비교 4종
```

설계 대비 결정 사항:

- 별도 `network_phase.json`은 만들지 않았다. F6 컬럼 계약의 단일 정본은
  `pubg_representation_v2.json`의 `f6_network_phase_role_tier1`이며,
  ETL 테이블과 ablation 코드가 모두 이 manifest를 따른다 (사본 = drift 위험).
- strength 계열은 설계 문구("raw edge-cache weights before row-normalization")
  그대로 cache의 log-damped weight 합이다. raw damage 합이 아니다.
- 계약상 중복 컬럼은 그대로 구현했다: phase-local에서
  `unique_targets == out_degree`, `isolation_m == min_distance_m`,
  `copress top1_weight == max_weight` (cache가 pair 단위로 이미 집계돼 있어
  생기는 동치). cum_* 계열은 중복이 아니다 (edge-row 누적 vs distinct 팀).
- NULL 정책: combat/copress는 event 채널이라 부재 = 0. proximity의
  거리/엔트로피/top-k는 부재 = 미관측 → NULL (HGB median impute가 처리).

빌드 결과 (full DB):

```text
stdb.team_phase_network_features: 21,830 매치, 5,419,004 행, 실패 0 (~4.5분, 4샤드)
라벨 risk set 커버리지: target_defined 1,366,821행 중 누락 0
manifest 검증: F6 43/43 (tools/check_feature_manifest.py --view stdb.team_phase_network_features)
```

검증 (tests/test_network_features.py, 전부 통과):

1. phase-local 값 = edge cache 독립 재계산 (combat/prox/copress 전수 대조)
2. cumulative 값 = phases<=k 삼각 참조 + 단조 증가
3. **미래 phase 차단**: phase>2 엣지 삭제 후 재빌드해도 cut<=2 피처 불변
4. 테이블 컬럼 ⊇ manifest F6 43컬럼
5. BUILD_SQL이 라벨/최종순위 소스를 참조하지 않음 (정적 가드)

Ablation 통합 (`model_stgcn/feature_ablation.py`):

- 신규 config: `network_tier1`(F6 단독), `conventional_plus_network`(F4+F6),
  `conventional_stdb_plus_network`(F4+F5+F6), `full_plus_network`.
  `full_node_features`는 과거 의미(F6 제외)를 유지해 기존 run과 비교 가능.
- 신규 paired 비교: `conventional_stdb_plus_network` vs
  `conventional_plus_stdb_spatial` (= F6 증분, 핵심 질문) 외 3종.
- F0가 아직 없으므로 §11의 A6/A8 (F0 기반)이 아니라 현행 표현(F4+F5) 기준
  증분으로 우선 측정한다. F0 view 구축 후 A0–A11 정식 매트릭스로 재실행.
- full-DB run: split=`full_db_reexperiment/split_phase_all_current_seed42.json`,
  align=`phase_result_analysis_edges/hazard_predictions_test_aligned.csv`
  (6/11 feature ablation 재실험과 동일 조건),
  out=`model_stgcn/checkpoints/feature_ablation_full_db_network/`.

### 2026-06-12 — F6 full-DB feature ablation 결과

aligned 134,245셀, 본 실험 동일 split, match-paired Wilcoxon (n=2,182):

| 비교 | AUC | ΔAUC | p |
| --- | --- | --- | --- |
| network_tier1 단독 | 0.7139 | vs prior_scores +0.0653 | 4.1e-193 |
| F4+F6 vs F4 | 0.7592 | +0.0066 | 2.3e-33 |
| **F4+F5+F6 vs F4+F5** | **0.7684** | **+0.0024** | **9.2e-11** |

해석: 역할 피처는 현행 최강 표현 위에서도 유의한 증분. 단 DynamicSTGNN
full(0.7780)과의 격차 ~+0.010이 남음 — §11.2의 두 번째 분기(관계별 이웃
상태는 역할 요약을 넘는 신호를 가짐)를 지지하는 1차 증거. 정식 검증은
B3/B4 graph ablation. F5 exposure와의 중첩 확인(F5 부재 시 F6 증분 2.7배).
산출물: `model_stgcn/checkpoints/feature_ablation_full_db_network/`.

### 2026-06-12 — Step 1: edge_sanity phase-local 교정 완료

구 `_node_shuffle_edges`는 `node_mask[b]`(T,N 2D)를 nonzero→flatten해
**시간 좌표와 노드 좌표가 섞인 인덱스**로 셔플했다 (중복 인덱스 비결정
쓰기 포함). 기존 −0.0769 edge sanity 수치는 broken control의 산물이므로
**maskedpool 재학습 완료 후 재실행 필요** (07 인과 문단 수치 교체).

수정: 순열을 `node_mask[b, t]`(phase-local 생존 집합)에서 추출, 채널 간
단일 순열 공유(노드 재라벨 의미론 — 팀은 하나의 가짜 정체성을 받고 모든
관계가 함께 이동). edge_shuffle도 phase-local화. 합성 텐서 회귀 테스트
6종 추가 (`tests/test_edge_sanity_controls.py`).

### 2026-06-12 — F7 Tier 1 구현 완료 (Step 3)

산출물:

```text
stdb/habitat_feature_etl.py           # 32컬럼, match-scoped resumable ETL
tests/test_habitat_features.py        # 불변식 6종
model_stgcn/feature_ablation.py       # family 부착 일반화 + F7 config 3종 + 비교 5종
```

핵심 의미론 결정 (코드 docstring에 상세):

- **Zone 의미론 실측 확정**: `dynamic_regions 'safe'` = 현재 블루존
  (생존팀 포함률 phase 2/5/7에서 77%/95%/97%), `'poison'` = 예고된 흰
  타깃 원 (37~49%만 진입 상태). 원시 telemetry safetyZone/poisonGasWarning
  매핑 그대로.
- **흰 원 announcement는 stdb phase 경계와 불일치** (phase 내 반경 p90
  ~50% 변동) → 타깃 의존 피처는 전부 **cut 시점(phase 마지막 poison
  snapshot)의 타깃 원** 기준 = 플레이어가 cut 시점 지도에서 보는 원
  (엄밀히 causal).
- `required_speed` = cut 시점 타깃 거리 / 현 phase duration (실제 마감
  카운트다운은 다음 phase 소속 = 미래 정보라 현 주기 길이를 pace proxy로
  사용, 문서화).
- `rotation_delay_ratio`: 타깃 원 첫 진입 시점 정규화, 미진입 = 1.0
  (최대 지연), 시작부터 내부 = 0.0.
- `edge_pressure_ratio`: 블루존 밖 OR 경계 150m 이내 snapshot 비율
  (co-pressure 채널과 동일 임계).
- 재사용: observed_speed = trajectory.avg_speed_mps,
  outside_safe_ratio = exposure.outside_safe_ratio (재계산 않음).

검증 (전부 통과): 전역 블록 mpg 재계산 일치(상대오차, float4 감안),
범위/부호/공식 불변식, **타깃 거리 0 ⟺ cut 시점 흰 원 내부** (전수),
미래 입력(phase>2 regions/positions/proximity) 삭제 후 cut≤2 불변,
manifest 32/32, 정적 라벨소스 가드.

### 2026-06-12 — F6+F7 통합 full-DB feature ablation 결과 (★핵심)

aligned 134,245셀, 본 실험 동일 split, match-paired Wilcoxon (n=2,182).
산출물: `model_stgcn/checkpoints/feature_ablation_full_db_v2tier1/`.

| Config | n_feat | AUC | AP |
| --- | ---: | ---: | ---: |
| prior_scores | 5 | 0.6490 | 0.4549 |
| network_tier1 (F6 단독) | 44 | 0.7140 | 0.5754 |
| **habitat_tier1 (F7 단독)** | **33** | **0.7493** | **0.6307** |
| conventional (F4) | 57 | 0.7522 | 0.6474 |
| conventional_plus_stdb_spatial (F4+F5) | 74 | 0.7664 | 0.6645 |
| conventional_stdb_plus_network (F4+F5+F6) | 117 | 0.7683 | 0.6665 |
| **conventional_stdb_plus_habitat (F4+F5+F7)** | **106** | **0.7824** | **0.6821** |
| conventional_stdb_network_habitat (전체 Tier-1) | 149 | 0.7829 | 0.6831 |

증분 검정 (AUC):

| 비교 | ΔAUC | p | rank-biserial |
| --- | ---: | ---: | ---: |
| **F4+F5+F7 vs F4+F5** | **+0.0168** | **2.5e-96** | **0.52** |
| F4+F5+F6 vs F4+F5 | +0.0022 | 3.7e-08 | 0.14 |
| 전체 vs F4+F5+F7 (F6의 잔여 기여) | +0.0011 | 5.3e-04 | 0.08 |
| habitat 단독 vs prior_scores | +0.0978 | 3.3e-278 | 0.88 |

핵심 발견:

1. **habitat pressure가 최대 단일 family 증분** (+0.0168 — F6의 ~7배).
   33개 피처 단독으로 conventional 57개(0.7522)에 육박(0.7493).
2. **feature-only 0.7824~0.7829 > DynamicSTGNN full graph 0.7780**
   (95% CI [0.7751, 0.7808] 상단 초과). 기존 "graph 이득" 해석은
   habitat 정보가 node feature에 없던 조건에서의 결과였음이 입증됨.
   단, 비교는 HGB(149d) vs GNN(66d 노드피처)로 학습기·입력이 모두 다름 —
   **공정 비교 = B3/B4 (F6+F7 노드피처로 GNN 재학습 후 graph on/off)가
   이제 이 연구의 결정적 실험.**
3. habitat 투입 후 F6의 잔여 기여는 +0.0011로 수축 — network role 신호의
   상당 부분이 공간 압박과 중첩. §11.2의 해석 원칙대로 이것은 실패가
   아니라 표현 family 간 정보 구조의 발견.
4. F7 이득의 출처에는 신규 정보(예고된 타깃 원까지의 거리·요구속도·
   진입 혼잡)가 포함 — 기존 66d에는 블루존 기준 노출만 있었고 타깃 원
   기준 압박은 전무했음.

남은 단계 (우선순위 재조정): **① B3/B4 graph ablation — F6+F7 노드피처
GNN 재학습으로 graph 잔여 기여 검증 (결정적)** → ② F0 official raw view
→ ③ Step 4 (v_team_phase_feature_v2) → ④ 정식 A0–A11 매트릭스 →
⑤ B0–B7 전체 (G2 combat_in, 교정된 edge_sanity 컨트롤 포함).

### 2026-06-12 — B3/B4 배관 구현 완료 + 실험 예약

V2 노드 피처(66+43+32=141d)가 DynamicSTGNN에 주입 가능해졌다:

```text
stdb/queries.py            PHASE_NODE_FEATURE_SPECS_V2 (75개, manifest와
                           1:1 — 테스트 보증), _fetch_phase_node_features에
                           include_v2_features 플래그 (두 테이블 직접 조회,
                           long view 불변 → HGB 경로 무영향)
model_stgcn/dataset.py     Lazy/Cached 데이터셋 플래그 배관
model_stgcn/train_phase.py --v2-node-features (ckpt에 feature_dim=141 기록)
eval_phase / edge_sanity / phase_result_analysis
                           ckpt args·feature_dim로 V2 레이아웃 자동 감지
tests/test_v2_node_features.py
                           ① 스펙↔manifest 1:1 ② 플래그 off 바이트 동일
                           ③ on 시 75채널 = 테이블값/scale (NULL→0)
scripts/run_v2_node_gnn.sh suite GPU 해제 대기 → B3/B4 학습(e12/b8/seed42/
                           동일 split) → eval(bootstrap 1000) →
                           phase_result_analysis (HGB 자체적합 +
                           tuned 슬롯=66d maskedpool full 참조)
```

스케일 상수는 기존 고정 상수 관례를 따름 (거리 /8160, 속도 /40, 면적
/5e7, crowding·scarcity ×1e6, degree류 /20, combat strength /50 등 —
스펙 리터럴에 명시). 스모크: 141d 학습 1 epoch 정상(params 0.344M),
ckpt 자동 감지 eval 일치, 전체 테스트 38개 통과. 러너는 18:18에 가동되어
suite 종료를 대기 중 — 결과(B3 vs B4 vs HGB-149d vs 66d-full)는
`$ROOT/v2_node_features/`에 적재 예정.

해석 프레임(사전 등록): B4>B3 유의 → graph 잔여 신호 입증(효과 크기
갱신); B4≈B3 → 관계 신호가 명시적 role/pressure 피처로 흡수됨(§11.2
원칙대로 발견으로 서술). B3 vs HGB-149d 격차는 학습기 차이 통제 지표.
