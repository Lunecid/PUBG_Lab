# PUBG Representation Feature Inventory and Ablation Plan

작성일: 2026-06-12

목적: 현재 short paper를 저널/석사논문으로 확장하기 위해, 지금까지 사용한 feature와 legacy/network feature를 한 곳에 고정하고, 새 연구에서 비교할 representation family와 ablation 설계를 정의한다.

핵심 원칙은 다음이다.

> Baseline은 반드시 공식 PUBG telemetry raw feature에서 출발한다. 이후 선행연구 feature, legacy feature, STDB feature, network feature, habitat-inspired feature, graph relation을 같은 team-phase risk set 위에서 단계적으로 조합해 비교한다.

이 문서는 구현 전 설계 문서이며, 이후 SQL view/materialized view, feature manifest, ablation script의 source-of-truth로 사용한다.

---

## 1. 연구 방향

현재 논문의 중심은 특정 모델의 성능이 아니라, 동일한 PUBG telemetry를 어떤 representation으로 재구성할 때 팀 생존 과정을 가장 잘 보존하는지 검증하는 것이다.

최종 연구 질문은 다음처럼 둔다.

> PUBG 팀 생존 위험은 단순 전투 집계보다, phase-aligned 시공간 노출, 팀 간 상호작용 네트워크, 축소되는 playable space의 경쟁 압박, 그리고 동적 graph relation을 함께 표현할 때 더 잘 예측되고 해석되는가?

따라서 feature 설계는 다음 family를 명확히 분리해야 한다.

| Family | 역할 | 현재 상태 |
| --- | --- | --- |
| Official raw telemetry | 모든 비교의 최소 baseline | raw table 존재, baseline manifest 필요 |
| Prior-paper score | Battleground 선행연구 재현/비교 | `model_stgcn/feature_ablation.py`에 구현 |
| Legacy ecological axes | 초기 player/snapshot graph feature 축 | `legacy/main.py`에 설계 흔적 |
| Current STDB | short paper의 핵심 team-phase representation | 구축 완료 |
| Network features | team-team graph 위치/역할을 node feature로 요약 | exploratory 결과만 있음, full DB 재구축 필요 |
| Habitat-inspired pressure | safe zone 축소로 생기는 공간 경쟁 압박 | 제안만 있음, 구현 필요 |
| Graph relation | feature summary가 아니라 adjacency로 직접 입력 | current STDB graph 존재, network/habitat graph 추가 필요 |

---

## 2. Canonical Analysis Unit

모든 feature family는 같은 단위로 맞춘다.

| 항목 | 정의 |
| --- | --- |
| Unit | `(match_id, team_id, phase_idx)` |
| Main label | `hazard_next`: cut phase `k`까지 관측하고 phase `k+1` 전에 탈락하면 1 |
| Main risk set | `target_defined = true`, cut phase 1--7 우선 |
| Split | match-level split |
| Graph tensor | `(match, phase, team, feature)` + `(match, phase, edge_type, team, team)` |
| Secondary tasks | final rank, top-k, winner prediction은 downstream/diagnostic으로 분리 |

중요한 비교 원칙:

- 같은 raw data를 사용한다.
- 같은 match split을 사용한다.
- 같은 label과 risk cell set을 사용한다.
- feature family마다 계산 가능한 row가 달라지면 안 된다.
- prediction input은 cut phase `k`까지 관측 가능한 정보만 사용한다.

---

## 3. Official Raw Telemetry Baseline

### 3.1 공식 raw baseline의 의미

여기서 "official raw feature"는 PUBG telemetry event/object에서 직접 나오는 column 또는 이를 phase 내에서 단순 집계한 값이다. STDB의 PostGIS 공간 질의, graph centrality, habitat overlap 같은 재표현은 raw baseline에 넣지 않는다.

raw baseline은 두 층으로 나눈다.

| 층 | 사용 가능성 | 예 |
| --- | --- | --- |
| Causal phase raw aggregate | main hazard task 입력 가능 | phase 내 damage count, item use count, position-derived distance |
| Post-match/final raw metadata | main hazard 입력 금지 또는 diagnostic only | final rank, winner, match end result |

공식 telemetry 문서 기준 event/object는 다음 자료를 참조한다.

- Telemetry overview: https://documentation.pubg.com/en/telemetry.html
- Telemetry events: https://documentation.pubg.com/en/telemetry-events.html
- Telemetry objects: https://documentation.pubg.com/en/telemetry-objects.html

### 3.2 현재 DB에 materialized 된 raw tables

현재 `pubg` schema에 실제 존재하는 raw table과 column은 다음과 같다.

| Table | Available columns |
| --- | --- |
| `pubg.matches` | `match_id`, `created_at`, `duration`, `game_mode`, `map_name`, `is_custom`, `match_type`, `season_state`, `shard_id`, `title_id`, `telemetry_url`, `collected_at` |
| `pubg.rosters` | `roster_id`, `match_id`, `rank`, `team_id`, `won` |
| `pubg.telem_positions` | `id`, `match_id`, `event_time`, `is_game`, `elapsed_time`, `num_alive`, `account_id`, `player_name`, `team_id`, `health`, `pos_x`, `pos_y`, `pos_z`, `vehicle_type`, `vehicle_id`, `vehicle_speed` |
| `pubg.telem_game_states` | `id`, `match_id`, `event_time`, `is_game`, `elapsed_time`, `num_alive_teams`, `num_alive_players`, `num_join_players`, `num_start_players`, `safe_zone_x`, `safe_zone_y`, `safe_zone_z`, `safe_zone_radius`, `poison_zone_x`, `poison_zone_y`, `poison_zone_z`, `poison_zone_radius`, `red_zone_x`, `red_zone_y`, `red_zone_z`, `red_zone_radius`, `black_zone_x`, `black_zone_y`, `black_zone_z`, `black_zone_radius` |
| `pubg.telem_phase_changes` | `id`, `match_id`, `event_time`, `is_game`, `phase`, `elapsed_time` |
| `pubg.telem_damage` | `id`, `match_id`, `event_time`, `is_game`, `attacker_id`, `attacker_name`, `attacker_x`, `attacker_y`, `attacker_z`, `victim_id`, `victim_name`, `victim_x`, `victim_y`, `victim_z`, `damage`, `damage_reason`, `damage_type`, `damage_causer`, `distance`, `is_through_wall` |
| `pubg.telem_groggy` | `id`, `match_id`, `event_time`, `is_game`, `attacker_id`, `attacker_name`, `attacker_x`, `attacker_y`, `attacker_z`, `victim_id`, `victim_name`, `victim_x`, `victim_y`, `victim_z`, `damage_reason`, `damage_type`, `damage_causer`, `distance`, `is_attacker_in_vehicle` |
| `pubg.telem_kills` | `id`, `match_id`, `event_time`, `is_game`, `killer_id`, `killer_name`, `killer_x`, `killer_y`, `killer_z`, `victim_id`, `victim_name`, `victim_x`, `victim_y`, `victim_z`, `dbno_maker_id`, `dbno_maker_name`, `finisher_id`, `finisher_name`, `damage_reason`, `damage_type`, `damage_causer`, `distance`, `is_suicide`, `assists_cnt`, `victim_rank`, `victim_team_id` |
| `pubg.telem_item_pickup` | `id`, `match_id`, `event_time`, `is_game`, `event_type`, `account_id`, `player_name`, `team_id`, `pos_x`, `pos_y`, `pos_z`, `item_id`, `category`, `sub_category`, `stack_count`, `attached_items`, `carepackage_id`, `lootbox_owner_team_id`, `lootbox_creator_id` |
| `pubg.telem_item_equip` | `id`, `match_id`, `event_time`, `is_game`, `event_type`, `account_id`, `player_name`, `team_id`, `pos_x`, `pos_y`, `pos_z`, `item_id`, `category`, `sub_category`, `attached_items` |
| `pubg.telem_item_use` | `id`, `match_id`, `event_time`, `is_game`, `account_id`, `player_name`, `team_id`, `pos_x`, `pos_y`, `pos_z`, `item_id`, `category`, `sub_category`, `stack_count` |
| `pubg.telem_parachute_landing` | `id`, `match_id`, `event_time`, `account_id`, `player_name`, `team_id`, `pos_x`, `pos_y`, `pos_z`, `distance` |
| `pubg.telem_match_start` | `id`, `match_id`, `event_time`, `map_name`, `weather_id`, `camera_view_type`, `team_size`, `is_custom`, `is_event_mode`, `blue_zone_options` |
| `pubg.telem_match_end` | `id`, `match_id`, `event_time`, `game_results` |

### 3.3 Raw phase aggregate baseline manifest

`F0_official_raw_phase`는 main baseline이다. 이 feature set은 STDB 공간 모델을 쓰지 않고 raw event table을 phase interval로 windowing해서 만든다.

| Group | Candidate features | Source |
| --- | --- | --- |
| Match/meta | `map_name`, `game_mode`, `match_type`, `team_size`, `is_custom`, `duration`, `created_month`, `created_week` | `matches`, `telem_match_start` |
| Phase clock | `phase_idx`, `phase_start_s`, `phase_end_s`, `phase_duration_s` | `telem_phase_changes`, `telem_game_states` |
| Alive state | `avg_alive_teams`, `avg_alive_players`, `min_alive_players`, `avg_team_alive_players` | `telem_game_states`, `telem_positions` |
| Zone raw state | `safe_zone_radius_start/end`, `poison_zone_radius_start/end`, `safe_zone_center_x/y_start/end`, `poison_zone_center_x/y_start/end`, red/black zone radii when available | `telem_game_states` |
| Position raw movement | team centroid start/end, path length, displacement, average speed, max speed | `telem_positions` |
| Health/vehicle | mean/min health, health slope, vehicle ratio, mean/max vehicle speed | `telem_positions` |
| Combat raw | damage dealt/taken, damage event count dealt/taken, groggy dealt/taken, kills dealt/taken, assist count, suicide count, distinct opponent teams | `telem_damage`, `telem_groggy`, `telem_kills` |
| Item raw | pickup/equip/use counts, item categories, weapon/ammo/armor/heal/boost/throwable counts, unique item count, carepackage pickup count, lootbox pickup count | `telem_item_*` |
| Landing raw | landing x/y, landing distance, landing time offset | `telem_parachute_landing` |

Raw baseline에서 제외하거나 별도 diagnostic으로만 둘 항목:

- `rosters.rank`, `rosters.won`: final outcome label이므로 main hazard input 금지.
- `telem_match_end.game_results`: post-match result이므로 input 금지.
- `victim_rank`: kill event 시점에 기록되지만 final/order 정보 성격이 강하므로 leakage audit 후 사용 여부 결정.
- full-match aggregate: cut phase 이후 event를 포함하면 main hazard input 금지.

### 3.4 `official_raw_phase`와 `conventional_current`의 차이

`F0 official_raw_phase`와 현재 short paper의 `F4 conventional_current`는 일부 column이 겹친다. 그러나 논문 설계상 둘은 같은 baseline이 아니다.

| Feature family | 기준 |
| --- | --- |
| `F0 official_raw_phase` | `pubg.telem_*` raw table에서 phase window만 적용해 직접 집계한 최소 baseline |
| `F4 conventional_current` | STDB 변환 이후 `v_team_phase_feature_long`에 올라온 current short-paper conventional aggregate |

따라서 새 논문의 baseline 표에서는 `F0 official_raw_phase`를 첫 행으로 두고, `F4 conventional_current`는 "현재 short paper의 conventional representation"으로 별도 비교한다. 이 구분이 있어야 "공식 raw telemetry에서 출발해 어떤 representation 변환이 성능을 올렸는가"를 주장할 수 있다.

---

## 4. Prior Battleground Paper Features

참조 논문:

- Local file: `paper/battleground_paper.pdf`
- Title: "Are you a Hunter or Camper? A phase-based framework for real-time survival prediction in battle royale using FightScore and ActiveScore"
- Journal: Entertainment Computing 55 (2025) 101040

### 4.1 원 논문의 feature 구조

원 논문은 solo player-phase 단위에서 전략을 두 축으로 요약한다.

| Axis | Original features |
| --- | --- |
| Aggressiveness / FightScore | `phase_attack_count`, `phase_kill_count`, `phase_total_damage`, `phase_assist_count` |
| Activeness / ActiveScore | `distance`, `phase_max_density`, `phase_item_count`, `carepackage_use` |

절차:

1. phase별 aggressiveness feature를 합산해 `FightScore`를 만든다.
2. phase별 activeness feature를 합산해 `ActiveScore`를 만든다.
3. 같은 phase 내 평균/기준값보다 높은지 낮은지로 네 유형을 만든다.

| Type | FightScore | ActiveScore | 해석 |
| --- | --- | --- | --- |
| Hunter | high | high | 교전성과 활동성이 모두 높음 |
| Sniper | high | low | 교전성은 높지만 활동성은 낮음 |
| Traveler | low | high | 교전성은 낮고 이동/탐색성이 높음 |
| Camper | low | low | 교전성과 활동성이 모두 낮음 |

원 논문의 prediction input은 in-game raw features, FightScore/ActiveScore, player type, type survival rate로 구성된다.

### 4.2 우리 short paper에서의 선행연구 재구성

현재 `model_stgcn/feature_ablation.py`는 원 논문의 구조를 team-phase task에 맞게 재구성한다. 원 논문은 player/solo 중심이고, 우리 데이터는 team-phase/full telemetry 중심이므로 feature 이름과 계산 방식은 다음처럼 대응된다.

| Prior concept | Our current operationalization |
| --- | --- |
| `phase_attack_count` | `damage_events_dealt` |
| `phase_kill_count` | `kills_dealt` |
| `phase_total_damage` | `damage_dealt` |
| `phase_assist_count` | direct assist count 대신 `groggy_dealt`, `distinct_opponents_damaged` 보강 |
| `distance` | `path_length_m`, `avg_speed_mps` |
| `phase_max_density` | `avg_enemy_density_100m` |
| `phase_item_count` | `pickup_count`, `unique_item_count` |
| `carepackage_use` | `carepackage_pickup_count` |

현재 코드의 prior feature set:

```text
AGGRESSIVENESS_COLS =
  damage_events_dealt
  kills_dealt
  damage_dealt
  groggy_dealt
  distinct_opponents_damaged

ACTIVENESS_COLS =
  path_length_m
  avg_speed_mps
  avg_enemy_density_100m
  pickup_count
  unique_item_count
  carepackage_pickup_count

PRIOR_SCORE_COLS =
  phase_idx
  fight_score
  active_score
  strategy_type
  type_survival_rate
```

구현상 차이:

- 원 논문은 raw 합산 중심이고, 현재 구현은 robust scaling 후 합산한다.
- 원 논문은 player/solo 중심이고, 현재 구현은 team-phase 중심이다.
- 원 논문은 평균 기준 분류이고, 현재 구현은 phase별 median 기준 high/low를 사용한다.
- `type_survival_rate`는 train split에서만 계산해 test leakage를 막아야 한다.

### 4.3 현재 short paper ablation 결과

현재 short paper의 핵심 feature ablation은 다음이다.

| Feature set | d | AUC | AP | BCE |
| --- | ---: | ---: | ---: | ---: |
| Prior-style score summary | 5 | 0.6490 | 0.4547 | 0.5993 |
| Score raw inputs | 12 | 0.7184 | 0.5703 | 0.5584 |
| STDB spatio-temporal only | 16 | 0.7124 | 0.5895 | 0.5529 |
| Conventional aggregates | 57 | 0.7522 | 0.6476 | 0.5242 |
| Conventional + STDB spatio-temporal | 72 | 0.7652 | 0.6626 | 0.5141 |
| Above + prior scores | 76 | 0.7649 | 0.6624 | 0.5141 |

해석:

- 선행연구식 score summary만으로는 AUC 0.649 수준이다.
- score를 만들기 전 raw input만 써도 AUC 0.718까지 올라간다.
- conventional telemetry와 STDB trajectory/exposure가 결합될 때 AUC 0.765 수준이 된다.
- prior score를 full feature에 다시 붙여도 추가 이득이 없다.

따라서 확장 논문의 중심은 "전략 점수"가 아니라 "representation family가 생존 신호를 얼마나 보존하는가"가 되어야 한다.

---

## 5. Legacy Feature Inventory

`legacy/main.py`에는 초기 single-match PyG graph 설계가 남아 있다. 당시 단위는 player/snapshot에 가까웠고, feature는 생태학적 5축으로 설계됐다.

| Legacy axis | Feature intent |
| --- | --- |
| Physiological State | health, groggy, recovery |
| Mobility | position, velocity, radial speed, ETA |
| Habitat Exposure | zone distance, zone damage |
| Competition Pressure | damage windows, density, isolation |
| Resource Readiness | weapon, armor, heal/boost |

초기 graph 설계:

| Element | Legacy design |
| --- | --- |
| Timestep | 10초 interval, closest position within +/-5초 |
| Node | alive player |
| Ally edge | same `team_id` |
| Encounter edge | enemy-only kNN, `k=5` |
| Elimination label | `telem_kills` + by-zone damage |
| Team placement label | roster rank |
| Zone model | safe zone + poison zone dual time series |

새 연구에서의 사용 방식:

- legacy feature는 그대로 쓰지 않고 team-phase 단위로 recast한다.
- `F3_legacy_axis_recast`라는 ablation family로 두어, "초기 ecology-style hand-crafted axes"가 current STDB보다 얼마나 약한지/강한지 확인한다.
- player-level edge 설계는 current team-phase graph와 직접 비교하지 않고, 필요하면 별도 appendix 또는 historical design note로 둔다.

---

## 6. Current STDB Short-Paper Features

현재 short paper의 STDB는 공식 telemetry를 다음 객체로 재구성한다.

| STDB object | Meaning |
| --- | --- |
| `stdb.matches`, `stdb.teams` | match/team metadata and labels |
| `stdb.moving_objects` | player/team centroid trajectory |
| `stdb.dynamic_regions` | safe/poison zone as dynamic spatial regions |
| `stdb.spatial_events` | damage, groggy, kill, landing, phase events |
| `stdb.team_phase_features` / views | team-phase tabular feature rows |
| `stdb.team_phase_edge_cache` | dynamic team graph edge cache |

### 6.1 Current 66 DynamicSTGNN node features

현재 `stdb.queries.PHASE_NODE_FEATURE_SPECS` 기준 node feature는 66개다.

| Group | Count | Features |
| --- | ---: | --- |
| Phase context | 12 | `phase_idx`, phase duration, alive teams/players, safe radius start/end/delta, safe center shift, poison radius start/end, zone gap start/end |
| Team state | 13 | roster size, alive players start/end/min/avg, mean/min health, health slope, team spread mean/max, vehicle ratio, vehicle speed mean/max |
| Movement | 9 | path length, displacement, sinuosity, average speed, rotation timing, safe alignment, zone entry time, outside safe duration, boundary distance slope |
| Exposure | 8 | mean/min boundary distance, outside-safe ratio, 100m enemy density, nearest enemy distance, path-near-enemy ratio, recent damage taken, combat-zone pass count |
| Combat | 10 | damage dealt/taken, damage events dealt/taken, groggy dealt/taken, kills dealt/taken, distinct opponents damaged/damaged-by |
| Readiness | 14 | combat initiated, pickup/equip/use counts, unique items, weapon/ammo/armor/throwable pickups, heal/boost use, carepackage pickups, lootbox pickups |

### 6.2 Current feature configs in code

`model_stgcn/feature_ablation.py` currently defines these nested feature sets.

| Config | Meaning |
| --- | --- |
| `prior_scores` | `phase_idx`, `fight_score`, `active_score`, `strategy_type`, `type_survival_rate` |
| `prior_raw` | phase index + aggressiveness/activeness raw inputs |
| `global_state` | phase context + team state |
| `conventional` | global/state + combat + loot/readiness |
| `stdb_spatial` | phase index + trajectory + exposure |
| `conventional_plus_trajectory` | conventional + movement |
| `conventional_plus_exposure` | conventional + exposure |
| `conventional_plus_stdb_spatial` | conventional + movement + exposure |
| `full_node_features` | all non-label columns from `v_team_phase_feature_long` |

### 6.3 Current graph relations

Current DynamicSTGNN graph uses three observed relation channels.

| Edge type | Meaning | Status |
| --- | --- | --- |
| `proximity` | mean centroid distance converted to distance-decay weight | implemented |
| `combat` | directed damage/groggy/kill interaction | implemented |
| `copressure` | two teams jointly under zone pressure and spatially near | implemented |

Current edge ablation:

| Model | AUC |
| --- | ---: |
| Tabular HGB | 0.7651 |
| DynamicSTGNN no graph | 0.7681 |
| DynamicSTGNN proximity | 0.7741 |
| DynamicSTGNN proximity + combat | 0.7755 |
| DynamicSTGNN full graph | 0.7782 |

### 6.4 Current feature audit notes

현재 DB audit에서 다음 feature는 target-defined rows에서 nonzero coverage가 사실상 없었다.

| Feature | Action |
| --- | --- |
| `nearby_enemy_combat_events_200m` | exclude or rebuild |
| `third_party_exposure_count` | exclude or rebuild |
| `lootbox_pickup_count` | exclude or rebuild |

새 ablation에서는 zero-coverage placeholder를 manifest에서 명확히 표시해야 한다. 그대로 포함하면 feature count만 늘고 representation contribution 해석을 흐린다.

---

## 7. Legacy Network Analysis Feature Inventory

`network_analysis/`에는 Baltic 1,000-match 중심 exploratory network analysis가 있다. 이 결과는 full DB main experiment에 그대로 쓰면 안 되지만, network feature family 설계의 원천으로 사용할 수 있다.

### 7.1 Static team-level network strategy features

`network_analysis/output/analysis_datasets/Baltic_Main_squad_n1000/strategy_type_network_analysis/strategy_type_team_features.csv` 기준 feature inventory:

| Group | Features |
| --- | --- |
| Match/team | `match_id`, `map_name`, `game_mode`, `team_idx`, `team_id`, `n_members` |
| Outcomes/diagnostic | `final_rank`, `survival_seconds`, `ranked_rp_proxy`, `top5/top10` derived values |
| Landing/farming | `landing_t`, `landing_x`, `landing_y`, `landing_dist_center_m`, `early_farm_x`, `early_farm_y`, `early_farm_dist_center_m` |
| Movement | `trajectory_length_m`, `movement_per_min`, `movement_active_steps`, `movement_step_count`, `movement_active_frame_rate`, `net_displacement_m`, `position_std_m`, `mean_step_distance_m` |
| Combat exposure | `damage_event_dealt_count`, `damage_event_taken_count`, `damage_event_involved_count`, `combat_event_rate_per_min`, `combat_active_frames`, `combat_exposure_frames`, `combat_st_participation_rate` |
| Hotzone | `landing_density_score`, `landing_neighbor_count`, `distance_to_hotzone_m`, `landing_distance_to_hotzone_m`, hotzone quantile group variables |
| Graph centrality | `degree`, `weighted_deg`, `c_degree`, `c_closeness`, `c_betweenness`, `c_eigenvector`, `c_pagerank` |
| Community | `community_size`, `community_density`, `within_deg`, `across_deg`, `local_clustering`, `match_modularity_q`, `across_share`, `community_embeddedness`, `community_core_score` |
| Bridge/role | `degree_per_min`, `weighted_deg_per_min`, `combat_events_per_min`, `dmg_dealt_per_min_calc`, `battle_bridge_score` |
| Network strategy score | standardized combat/movement/network variables, `network_fight_score`, `network_active_score`, `fight_high`, `active_high`, `strategy_type` |

Important leakage note:

- 이 CSV의 network metrics는 full-match graph 기반일 가능성이 크다.
- full-match degree/PageRank/community는 future combat를 포함하므로 main next-phase hazard prediction input으로 사용할 수 없다.
- 논문 본류에서는 phase-local 또는 cumulative-up-to-phase metrics로 재계산해야 한다.

### 7.2 Phase-wise network strategy features

`phase_strategy_type_analysis/phase_team_features.csv` 기준 phase-wise feature:

| Group | Features |
| --- | --- |
| Phase metadata | `phase`, `phase_start_s`, `phase_end_s`, `phase_duration_s`, `phase_observed_s`, `phase_observed_min` |
| Movement | `phase_trajectory_m`, `phase_active_steps`, `phase_step_count`, `phase_active_frame_rate`, `phase_position_std_m`, `phase_movement_per_min` |
| Network | `phase_degree`, `phase_weighted_deg`, `phase_degree_per_min`, `phase_weighted_deg_per_min`, `phase_c_pagerank`, `phase_c_betweenness` |
| Community | `phase_community_id`, `phase_within_deg`, `phase_across_deg`, `phase_across_share`, `phase_community_embeddedness` |
| Combat | `phase_damage_dealt`, `phase_damage_taken`, `phase_damage_event_dealt_count`, `phase_damage_event_taken_count`, `phase_damage_event_involved_count`, `phase_combat_event_rate_per_min`, `phase_combat_active_frames`, `phase_combat_exposure_frames`, `phase_combat_st_participation_rate`, `phase_unique_enemy_teams` |
| Bridge | `phase_battle_bridge_score` |
| Strategy | standardized phase variables, `phase_fight_score`, `phase_active_score`, `phase_fight_high`, `phase_active_high`, `phase_strategy_type` |

이 feature set은 새 연구의 `F6_network_phase`에 가장 직접적으로 연결된다. 다만 full DB, same split, same risk set에서 다시 만들어야 한다.

### 7.3 Network feature family to rebuild

새 연구에서는 다음 table/view를 만든다.

```text
stdb.team_phase_network_features
  match_id
  team_id
  phase_idx
  phase-local network metrics
  cumulative-through-phase network metrics
```

우선순위 1: SQL/edge-cache에서 안정적으로 만들 수 있는 metric.

| Group | Candidate features |
| --- | --- |
| Combat local | `combat_out_degree_k`, `combat_in_degree_k`, `combat_out_strength_k`, `combat_in_strength_k`, `combat_total_strength_k`, `combat_balance_k`, `combat_unique_targets_k`, `combat_unique_attackers_k`, `combat_reciprocity_k` |
| Proximity local | `prox_degree_threshold_k`, `prox_strength_k`, `prox_top1_weight_k`, `prox_top3_strength_k`, `prox_isolation_k`, `prox_embeddedness_k` |
| Co-pressure local | `copress_degree_k`, `copress_strength_k`, `copress_max_weight_k`, `copress_shared_pressure_ratio_k` |
| Cumulative combat | same combat metrics accumulated over phases `1..k` |
| Cumulative proximity/copressure | exposure/embeddedness accumulated over phases `1..k` |

우선순위 2: graph algorithm metric.

| Metric | Caution |
| --- | --- |
| PageRank | sparse phase graph에서 불안정할 수 있음 |
| Betweenness | per-match/per-phase 계산 비용 큼 |
| Louvain community | phase마다 node/edge가 적으면 noisy |
| Bridge score | community definition에 의존 |
| Eigenvector centrality | disconnected/sparse graph에서 불안정 |

예측 feature로 허용:

- phase `k` 내부에서 관측된 network metric.
- phase `1..k` 누적 metric.

예측 feature로 금지:

- match 전체 graph로 계산한 degree/PageRank/community.
- 탈락 이후 event를 포함한 누적 metric.

---

## 8. Habitat-Inspired Feature Inventory

Habitat frame은 생물학적 동일성을 주장하지 않는다. 축소되는 playable space가 만드는 경쟁 압박을 정량화하기 위한 analytical analogy로만 사용한다.

권장 용어:

- habitat-inspired pressure
- shrinking-space pressure
- playable-space competition
- refuge scarcity
- forced-convergence pressure

### 8.1 Concept mapping

| Ecology term | PUBG representation |
| --- | --- |
| Habitat | safe/playable zone |
| Habitat loss | safe zone area shrinkage |
| Habitat compression | alive teams compressed into smaller safe area |
| Population density | alive teams per safe area |
| Edge effect | boundary/blue-zone pressure |
| Dispersal/migration | rotation toward next safe zone |
| Extinction | team elimination |
| Refuge scarcity | limited reachable safe space under enemy density |

### 8.2 Habitat feature family to build

새 연구에서는 다음 table/view를 만든다.

```text
stdb.team_phase_habitat_features
  match_id
  team_id
  phase_idx
  global shrinking-space pressure
  team-level migration pressure
  local refuge scarcity
```

Global phase-level habitat features:

| Feature | Definition |
| --- | --- |
| `safe_area_start_m2` | area of safe zone at phase start |
| `safe_area_end_m2` | area of safe zone at phase end or announced target |
| `habitat_loss_ratio` | `1 - safe_area_end_m2 / safe_area_start_m2` |
| `safe_area_per_alive_team` | `safe_area_end_m2 / alive_teams_k` |
| `crowding_pressure` | `alive_teams_k / safe_area_end_m2` |
| `center_shift_per_radius` | safe center shift divided by current radius |
| `zone_gap_pressure` | blue/safe boundary gap normalized by radius |

Team-level habitat pressure features:

| Feature | Definition |
| --- | --- |
| `distance_to_target_safe_m` | distance from team centroid to announced target safe zone |
| `required_movement_speed_mps` | distance to target safe zone divided by remaining phase time |
| `migration_deficit_mps` | required speed minus observed/available speed |
| `boundary_pressure` | high when outside or near safe-zone boundary |
| `edge_pressure_ratio` | fraction of phase spent outside or within boundary threshold |
| `safe_alignment_to_target` | movement direction aligned with target safe zone |
| `rotation_delay_ratio` | how late the team starts moving toward target |
| `local_enemy_density_in_safe` | enemy teams near the team's safe entry/refuge area |
| `refuge_scarcity` | local enemy density divided by reachable safe area |

Habitat graph edge candidates:

| Edge | Meaning |
| --- | --- |
| `habitat_overlap_ij_k` | teams have overlapping reachable safe-zone candidate regions |
| `forced_convergence_ij_k` | teams are forced toward similar target directions under high pressure |
| `refuge_competition_ij_k` | teams compete for the same low-area safe region |
| `co_migration_pressure_ij_k` | teams have high required movement speed toward nearby target cells |

Leakage rule:

- only safe-zone information known at cut phase `k` can be used.
- target safe zone for the current transition is allowed only if it is already announced in telemetry by cut time.
- future phase zone, future enemy movement, future combat, and final winner/rank cannot enter habitat features.

---

## 9. Patch / Temporal Cohort Features

Current DB does not contain exact PUBG patch/build version.

Checked facts:

- No `patch`, `version`, `build`, or `season` feature exists in `pubg` or `stdb` except `matches.season_state`.
- `season_state` is currently `progress` and not useful as a patch label.
- Official telemetry event/object docs do not expose patch version as a standard telemetry field.

Usable proxy features:

| Feature | Use |
| --- | --- |
| `created_at` month/week | temporal cohort/control |
| `telemetry_url` date path | collection-date proxy |
| `days_since_first_collected_match` | temporal drift proxy |
| external patch calendar mapping | optional if exact date ranges are reliable |

Recommendation:

- patch proxy는 main explanatory feature가 아니라 robustness/control로 둔다.
- main ablation은 patch proxy 없이 수행하고, temporal cohort split or stratified evaluation을 추가한다.

---

## 10. Proposed Feature Families for New Study

### 10.1 Feature-only family names

| ID | Name | Definition |
| --- | --- | --- |
| `F0` | `official_raw_phase` | official telemetry raw phase aggregates; mandatory baseline |
| `F1` | `prior_paper_raw` | raw inputs corresponding to Battleground paper aggressiveness/activeness |
| `F2` | `prior_paper_summary` | FightScore, ActiveScore, strategy type, type survival rate |
| `F3` | `legacy_axis_recast` | legacy 5-axis player/snapshot design recast to team-phase |
| `F4` | `conventional_current` | current short-paper conventional telemetry aggregate |
| `F5` | `stdb_spatial_current` | current STDB trajectory/exposure |
| `F6` | `network_phase` | phase-local and cumulative network role features |
| `F7` | `habitat_pressure` | habitat-inspired shrinking-space pressure |
| `F8` | `temporal_patch_proxy` | created time/collection cohort controls |

### 10.2 Graph relation family names

| ID | Name | Definition |
| --- | --- | --- |
| `G0` | `no_graph` | node features only |
| `G1` | `current_stdb_graph` | proximity + combat + copressure |
| `G2` | `network_graph` | graph relation emphasizing combat/proximity network role |
| `G3` | `habitat_graph` | reachable-space/refuge competition relation |
| `G4` | `combined_graph` | current STDB + network + habitat relation channels |
| `G5` | `shuffled_graph_control` | degree/weight-preserving or phase-shuffled negative control |

---

## 11. Ablation Design

### 11.1 Experiment A: Feature representation ablation

Same learner, same split, same risk set, feature set only changes.

Recommended learner:

- HistGradientBoosting or LightGBM for primary table.
- Logistic regression can be added as interpretability/linearity baseline.

Core matrix:

| Condition | Feature set |
| --- | --- |
| A0 | `F0 official_raw_phase` |
| A1 | `F1 prior_paper_raw` |
| A2 | `F2 prior_paper_summary` |
| A3 | `F3 legacy_axis_recast` |
| A4 | `F4 conventional_current` |
| A5 | `F5 stdb_spatial_current` |
| A6 | `F0 + F5` |
| A7 | `F0 + F6` |
| A8 | `F0 + F7` |
| A9 | `F0 + F5 + F6` |
| A10 | `F0 + F5 + F7` |
| A11 | `F0 + F5 + F6 + F7` |
| A12 | `F0 + F5 + F6 + F7 + F8` as robustness, not main claim |

This table answers:

- How much signal is already in official raw telemetry?
- How much does prior score summarization lose?
- How much does STDB trajectory/exposure add?
- Does network role add beyond raw/STDB?
- Does habitat pressure add beyond raw/STDB?

### 11.2 Experiment B: Graph relation ablation

Node feature set is fixed, graph relation changes.

Recommended fixed node feature:

```text
F0 + F5 + F6 + F7
```

Core matrix:

| Condition | Node feature | Graph |
| --- | --- | --- |
| B0 | all selected features | `G0 no_graph` |
| B1 | all selected features | `G1 current_stdb_graph` |
| B2 | all selected features | `G2 network_graph` |
| B3 | all selected features | `G3 habitat_graph` |
| B4 | all selected features | `G4 combined_graph` |
| B5 | all selected features | `G5 shuffled_graph_control` |

This table answers:

- Are graph relations useful after strong node features?
- Which graph relation family contributes most?
- Is the gain relation-specific or just graph regularization/noise?

### 11.3 Experiment C: Phase-wise and group importance

Run on the best feature-only model and best graph model.

Required analyses:

| Analysis | Output |
| --- | --- |
| phase-wise metrics | AUC/AP/BCE by phase |
| early/mid/late grouping | early 1--2, mid 3--5, late 6--7 |
| group permutation importance | drop/permute raw, STDB, network, habitat groups |
| feature SHAP/permutation | top individual features by phase |
| high-pressure subgroup | evaluate separately when `crowding_pressure` is high |
| map/mode robustness | map and game_mode stratified metrics |

### 11.4 Experiment D: Winner/top-k extension

Winner/top-k should be secondary.

Two tasks must be separated.

| Task | Input | Meaning |
| --- | --- | --- |
| Real-time top-k forecast | `X_{\le k}` only | predict final top-k/winner at cut phase |
| Retrospective survival decomposition | observed phase features through match | explain how phase hazards accumulate |

Do not use future phase features to claim real-time winner prediction.

---

## 12. Leakage Audit Table

Every feature group must pass this audit.

| Feature group | Uses future phase? | Uses future zone? | Uses future combat? | Main hazard input? |
| --- | --- | --- | --- | --- |
| Official raw phase aggregate | no, if windowed by cut phase | no | no | yes |
| Prior paper raw | no, if windowed by cut phase | no | no | yes |
| Prior score summary | no, if survival rate fitted on train only | no | no | yes |
| Current conventional | no | no | no | yes |
| Current STDB spatial | no, if target zone known at cut phase | audit needed | no | yes |
| Network phase-local | no | no | no | yes |
| Network cumulative 1..k | no | no | no | yes |
| Network full-match | yes | no | yes | no |
| Habitat pressure | no, if only announced zone used | audit needed | no | yes |
| Habitat reachable-space graph | no, if computed from cut-time state | audit needed | no | yes |
| Final rank/winner | label | label | label | no as input |
| Patch/temporal proxy | no | no | no | robustness only |

---

## 13. Implementation Plan

### Step 1. Freeze feature manifests

Create explicit manifests under `model_stgcn/feature_manifests/`.

Recommended files:

```text
official_raw_phase.json
prior_paper.json
legacy_axis_recast.json
current_stdb.json
network_phase.json
habitat_pressure.json
ablation_matrix.json
```

Each manifest should include:

- feature name
- source table/view
- computation window
- causal status
- family/group
- scaling/transform
- null policy
- whether it is used in main hazard task

### Step 2. Build official raw baseline view

Create a view or materialized view:

```text
stdb.v_team_phase_official_raw_features
```

This should be computed directly from `pubg.telem_*` tables with phase intervals. It should not depend on graph centrality or habitat overlap.

### Step 3. Rebuild network features for full DB

Create:

```text
stdb.team_phase_network_features
```

Use `stdb.team_phase_edge_cache` where possible. Start with degree/strength/reciprocity/cumulative features before PageRank/community.

### Step 4. Build habitat pressure features

Create:

```text
stdb.team_phase_habitat_features
```

Start with area, density, required movement, boundary pressure, refuge scarcity. Add habitat graph only after feature-only effect is measured.

### Step 5. Extend ablation script

Extend or replace:

```text
model_stgcn/feature_ablation.py
```

Required changes:

- load feature manifests
- join raw/STDB/network/habitat feature views on `(match_id, team_id, phase_idx)`
- enforce common risk set
- write feature list per condition
- write row counts and null coverage per condition
- write paired test by match

### Step 6. Extend graph ablation

Current graph edge modes:

```text
none
proximity
proximity_combat
full
```

New edge modes:

```text
none
current_stdb_graph
network_graph
habitat_graph
combined_graph
shuffled_graph_control
```

---

## 14. Immediate Decisions

Recommended immediate choices before coding:

1. Use `F0 official_raw_phase` as the first implemented baseline.
2. Keep current `F2 prior_paper_summary` and `F1 prior_paper_raw` as prior-work comparison.
3. Rebuild `F6 network_phase` from `team_phase_edge_cache`, not from old Baltic CSV outputs.
4. Implement `F7 habitat_pressure` as feature-only first; defer habitat graph until feature effect is known.
5. Keep winner/top-k as secondary extension after hazard ablation is stable.

Main first milestone:

> Produce a single joined team-phase dataset with `F0`, `F1`, `F2`, `F4`, `F5`, `F6`, and `F7` columns, all aligned on the same target-defined risk cells, then run feature-only HGB ablation.

---

## 15. Source Pointers

Local sources used for this inventory:

| Source | Use |
| --- | --- |
| `paper/battleground_paper.pdf` | prior FightScore/ActiveScore framework |
| `model_stgcn/feature_ablation.py` | current prior/current STDB feature ablation implementation |
| `paper/sections/03-data-stdb-modeling.tex` | current STDB feature group and graph definition |
| `paper/sections/07-results-discussion.tex` | current ablation results |
| `paper/sections/appendix-features.tex` | 66 node feature dictionary |
| `stdb/queries.py` | DynamicSTGNN feature and edge specs |
| `legacy/main.py` | legacy 5-axis ecological feature design |
| `network_analysis/*strategy_type*` | network strategy exploratory feature inventory |
| `network_analysis/habitat_contraction_model_addendum.md` | habitat-inspired pressure proposal |
| `docs/DATA_AVAILABILITY_BY_DOMAIN.md` | current data availability and graph-eligible population |
