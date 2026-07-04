# Battle Royale OM Ladder Result And Feature Audit

Date: 2026-07-04
Class: `ExperimentResult`
Lifecycle: `active`

## Purpose

This document freezes the first required-cohort OM0-OM6 result audit. It records:

- the exact feature presets used by the run;
- the exact features added at each OM block;
- the actual cohort split sizes and test evaluation `N`;
- the aggregate result pattern and claim boundary.

The source result file is:

```text
model_stgcn/checkpoints/br_om_ladder_required_v1/om_ladder_metrics_all_cohorts.csv
```

The run root is:

```text
model_stgcn/checkpoints/br_om_ladder_required_v1
```

## Run Contract

| Field | Value |
| --- | --- |
| Cohort rule | one independent model set per map / perspective / team-size cohort |
| Completed cohorts | 13 |
| Presets per cohort | 7 (`OM0` through `OM6`) |
| Metric rows | 91 |
| Model | `mr` |
| Graph encoder | `none` |
| Edge mode | `none` |
| Edge control | `observed` |
| Global fusion | `concat` |
| Team decoder | `mlp` |
| Pairwise rank loss | `0.0` |
| Seed | `42` |
| Epochs | 12 |
| Batch | 8 |
| Eval split | `test` |
| Main hazard metric unit | `at_risk_team_phase` |

`N` below is the number of at-risk team-phase rows in the test evaluation, not the number of matches. Match counts are reported separately from each cohort `split.json`.

## Headline Interpretation

The central ontology claim is supported in this first run: `OM3 Pressure / Habitat`, the `O2 x O6 = H` block, is the only block with a positive AUC increment in all 13 cohorts. Its sequential AUC gain has median `+0.0103`, min `+0.0057`, and max `+0.0211`. This supports the claim that Battle Royale survival is organized by phase-conditioned transformations of the playable survival environment, not only by roster state or movement.

Threat exposure adds smaller and more cohort-dependent gains. The current relation and spatial-proxy blocks are not full tests of O7 graph structure or the planned O6 Z-ladder: `OM5` uses node-summary relation features only, and `OM6` adds only three spatial-exposure proxies.

## Sequential Block Result Summary

| Step | Added construct | Median Delta AUC | Mean Delta AUC | Range | Positive cohorts | Interpretation |
| --- | --- | ---: | ---: | --- | ---: | --- |
| `OM0->OM1` | O1 agent/team survival capital | `-0.0007` | `-0.0001` | `-0.0057` to `+0.0082` | 5/13 | No stable pooled contribution; phase-stratified check required. |
| `OM1->OM2` | O3 movement and rotation response | `+0.0028` | `+0.0031` | `-0.0004` to `+0.0078` | 10/13 | Small movement/rotation contribution. |
| `OM2->OM3` | O2 contraction pressure + O2 x O6 habitat H | `+0.0103` | `+0.0119` | `+0.0057` to `+0.0211` | 13/13 | Universal headline gain. |
| `OM3->OM4` | O4 threat exposure / realized combat pressure | `+0.0020` | `+0.0034` | `-0.0029` to `+0.0238` | 10/13 | Small conditional gain, strongest in compact/high-contact cohorts. |
| `OM4->OM5` | O7 node-summary relation structure | `+0.0003` | `+0.0004` | `-0.0030` to `+0.0049` | 7/13 | Near-zero for node-summary relation projection. |
| `OM5->OM6` | Current O6 spatial-exposure proxies | `+0.0010` | `+0.0008` | `-0.0052` to `+0.0042` | 10/13 | Tiny gain; current implementation is not the planned Z-ladder. |

## Macro Result Summary

| Preset | Total features | Node | Global | AUC | Brier | ECE | Rank Spearman | Top5 AUC | Top10 AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `om0_context` | 3 | 1 | 2 | `0.7375` | `0.1794` | `0.0189` | `0.5891` | `0.8050` | `0.8474` |
| `om1_survival_capital` | 12 | 10 | 2 | `0.7374` | `0.1794` | `0.0178` | `0.5885` | `0.8038` | `0.8464` |
| `om2_movement` | 19 | 17 | 2 | `0.7405` | `0.1785` | `0.0149` | `0.5912` | `0.8059` | `0.8493` |
| `om3_pressure_habitat` | 66 | 41 | 25 | `0.7523` | `0.1752` | `0.0169` | `0.6173` | `0.8227` | `0.8805` |
| `om4_threat_exposure` | 79 | 54 | 25 | `0.7557` | `0.1739` | `0.0169` | `0.6261` | `0.8274` | `0.8812` |
| `om5_relation_structure` | 122 | 97 | 25 | `0.7561` | `0.1736` | `0.0154` | `0.6250` | `0.8264` | `0.8805` |
| `om6_spatial_proxy` | 125 | 100 | 25 | `0.7569` | `0.1733` | `0.0148` | `0.6237` | `0.8257` | `0.8801` |

## Cohort Sizes And Outcomes

| Cohort | Map code | Mode | Train matches | Val matches | Test matches | Total matches | Test N | Pos rate | OM0 AUC | OM3 AUC | OM6 AUC | OM0->OM3 | OM0->OM6 | OM0 rank rho | OM6 rank rho |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Erangel TPP squad | `Baltic_Main` | `squad` | 4,964 | 620 | 620 | 6,204 | 31,628 | `31.7%` | `0.7832` | `0.7937` | `0.7948` | `+0.0106` | `+0.0117` | `0.6200` | `0.6524` |
| Taego TPP squad | `Tiger_Main` | `squad` | 3,329 | 415 | 415 | 4,159 | 21,805 | `29.7%` | `0.7843` | `0.7937` | `0.7967` | `+0.0094` | `+0.0123` | `0.5926` | `0.6325` |
| Miramar / Desert TPP squad | `Desert_Main` | `squad` | 1,235 | 154 | 154 | 1,543 | 9,068 | `28.9%` | `0.7674` | `0.7745` | `0.7782` | `+0.0071` | `+0.0108` | `0.5439` | `0.5735` |
| Rondo TPP squad | `Neon_Main` | `squad` | 833 | 103 | 103 | 1,039 | 5,200 | `28.9%` | `0.7480` | `0.7594` | `0.7624` | `+0.0114` | `+0.0144` | `0.5575` | `0.5972` |
| Sanhok TPP squad | `Savage_Main` | `squad` | 752 | 94 | 94 | 940 | 4,637 | `32.4%` | `0.7574` | `0.7687` | `0.7702` | `+0.0112` | `+0.0128` | `0.6639` | `0.6774` |
| Erangel TPP solo | `Baltic_Main` | `solo` | 640 | 80 | 80 | 800 | 12,065 | `35.8%` | `0.7397` | `0.7669` | `0.7747` | `+0.0273` | `+0.0350` | `0.6051` | `0.6764` |
| Erangel TPP duo | `Baltic_Main` | `duo` | 526 | 65 | 65 | 656 | 5,763 | `33.0%` | `0.7096` | `0.7274` | `0.7260` | `+0.0178` | `+0.0164` | `0.5846` | `0.6038` |
| Taego TPP duo | `Tiger_Main` | `duo` | 459 | 57 | 57 | 573 | 4,610 | `32.8%` | `0.7159` | `0.7407` | `0.7415` | `+0.0249` | `+0.0256` | `0.5883` | `0.6178` |
| Vikendi TPP squad | `DihorOtok_Main` | `squad` | 400 | 49 | 49 | 498 | 2,778 | `30.2%` | `0.7353` | `0.7499` | `0.7540` | `+0.0147` | `+0.0187` | `0.5682` | `0.5885` |
| Taego FPP squad | `Tiger_Main` | `squad-fpp` | 349 | 43 | 43 | 435 | 1,995 | `33.3%` | `0.7213` | `0.7427` | `0.7487` | `+0.0214` | `+0.0275` | `0.6050` | `0.6395` |
| Erangel FPP squad | `Baltic_Main` | `squad-fpp` | 344 | 43 | 43 | 430 | 2,204 | `31.3%` | `0.7201` | `0.7452` | `0.7469` | `+0.0251` | `+0.0269` | `0.4941` | `0.5550` |
| Karakin TPP squad | `Summerland_Main` | `squad` | 267 | 33 | 33 | 333 | 1,159 | `36.8%` | `0.6933` | `0.6962` | `0.7198` | `+0.0029` | `+0.0265` | `0.6513` | `0.6772` |
| Paramo TPP squad | `Chimera_Main` | `squad` | 267 | 33 | 33 | 333 | 948 | `33.0%` | `0.7124` | `0.7212` | `0.7263` | `+0.0088` | `+0.0139` | `0.5835` | `0.6174` |

## Feature Block Inventory

The lists below are the exact normalized tensor feature names selected by `model_stgcn/v3/spec.py` for this run. Each OM preset is cumulative. For readability, each block first lists newly added features, then the cumulative input list used by that preset.

### OM0 Context (`om0_context`)

Ontology role: O0/O2 minimal phase clock anchor

Run note: Minimal time/phase context plus team-size anchor. This is the pooled phase baseline, not a player-skill baseline.

Feature count: total 3, node 1, global 2.

New features at this step:

- Global (2): `phase_idx_norm`, `phase_duration_10min`
- Node (1): `team_size_norm`

Cumulative features used by this preset:

- Global (2): `phase_idx_norm`, `phase_duration_10min`
- Node (1): `team_size_norm`

### OM1 Survival Capital (`om1_survival_capital`)

Ontology role: O1 agent/team survival capital

Run note: Adds roster, health, and team cohesion state.

Feature count: total 12, node 10, global 2.

New features at this step:

- Global (0): none
- Node (9): `alive_players_start_norm`, `alive_players_end_norm`, `alive_players_min_norm`, `alive_players_avg_norm`, `mean_health_avg_norm`, `min_health_min_norm`, `health_slope_norm`, `team_spread_avg_norm`, `team_spread_max_norm`

Cumulative features used by this preset:

- Global (2): `phase_idx_norm`, `phase_duration_10min`
- Node (10): `team_size_norm`, `alive_players_start_norm`, `alive_players_end_norm`, `alive_players_min_norm`, `alive_players_avg_norm`, `mean_health_avg_norm`, `min_health_min_norm`, `health_slope_norm`, `team_spread_avg_norm`, `team_spread_max_norm`

### OM2 Movement (`om2_movement`)

Ontology role: O3 movement and rotation response

Run note: Adds distance, speed, rotation timing, and safe-zone alignment response.

Feature count: total 19, node 17, global 2.

New features at this step:

- Global (0): none
- Node (7): `path_length_norm`, `displacement_norm`, `log1p_sinuosity`, `avg_speed_norm`, `rotation_timing`, `safe_alignment`, `zone_entry_time`

Cumulative features used by this preset:

- Global (2): `phase_idx_norm`, `phase_duration_10min`
- Node (17): `team_size_norm`, `alive_players_start_norm`, `alive_players_end_norm`, `alive_players_min_norm`, `alive_players_avg_norm`, `mean_health_avg_norm`, `min_health_min_norm`, `health_slope_norm`, `team_spread_avg_norm`, `team_spread_max_norm`, `path_length_norm`, `displacement_norm`, `log1p_sinuosity`, `avg_speed_norm`, `rotation_timing`, `safe_alignment`, `zone_entry_time`

### OM3 Pressure / Habitat (`om3_pressure_habitat`)

Ontology role: O2 contraction pressure + O2 x O6 habitat H

Run note: Adds zone contraction pressure plus safe-area scarcity, crowding, migration deficit, and refuge/congestion features. This is the headline habitat block.

Feature count: total 66, node 41, global 25.

New features at this step:

- Global (23): `avg_alive_teams_norm`, `avg_alive_players_norm`, `safe_radius_start_norm`, `safe_radius_end_norm`, `safe_radius_delta_norm`, `safe_center_shift_norm`, `poison_radius_start_norm`, `poison_radius_end_norm`, `zone_gap_start_norm`, `zone_gap_end_norm`, `hab_safe_area_start_m2`, `hab_safe_area_end_m2`, `hab_safe_area_delta_m2`, `hab_loss_ratio`, `hab_shrink_rate_m2ps`, `hab_alive_teams`, `hab_safe_area_per_alive_team`, `hab_crowding_pressure`, `hab_center_shift_m`, `hab_center_shift_per_radius`, `hab_zone_gap_start_m`, `hab_zone_gap_end_m`, `hab_zone_gap_pressure`
- Node (24): `outside_safe_duration_10min`, `boundary_distance_slope_norm`, `mean_dist_to_boundary_norm`, `min_dist_to_boundary_norm`, `outside_safe_ratio`, `hab_dist_to_safe_boundary_start_m`, `hab_dist_to_safe_boundary_end_m`, `hab_signed_boundary_distance_start_m`, `hab_signed_boundary_distance_end_m`, `hab_distance_to_target_safe_m`, `hab_required_speed_to_safe_mps`, `hab_observed_speed_mps`, `hab_migration_deficit_mps`, `hab_outside_safe_ratio`, `hab_edge_pressure_ratio`, `hab_rotation_delay_ratio`, `hab_safe_alignment_to_target`, `hab_local_enemy_count_100m`, `hab_local_enemy_count_200m`, `hab_local_enemy_count_500m`, `hab_local_enemy_density_200m`, `hab_safe_entry_enemy_count_200m`, `hab_safe_entry_congestion`, `hab_refuge_scarcity_simple`

Cumulative features used by this preset:

- Global (25): `phase_idx_norm`, `phase_duration_10min`, `avg_alive_teams_norm`, `avg_alive_players_norm`, `safe_radius_start_norm`, `safe_radius_end_norm`, `safe_radius_delta_norm`, `safe_center_shift_norm`, `poison_radius_start_norm`, `poison_radius_end_norm`, `zone_gap_start_norm`, `zone_gap_end_norm`, `hab_safe_area_start_m2`, `hab_safe_area_end_m2`, `hab_safe_area_delta_m2`, `hab_loss_ratio`, `hab_shrink_rate_m2ps`, `hab_alive_teams`, `hab_safe_area_per_alive_team`, `hab_crowding_pressure`, `hab_center_shift_m`, `hab_center_shift_per_radius`, `hab_zone_gap_start_m`, `hab_zone_gap_end_m`, `hab_zone_gap_pressure`
- Node (41): `team_size_norm`, `alive_players_start_norm`, `alive_players_end_norm`, `alive_players_min_norm`, `alive_players_avg_norm`, `mean_health_avg_norm`, `min_health_min_norm`, `health_slope_norm`, `team_spread_avg_norm`, `team_spread_max_norm`, `path_length_norm`, `displacement_norm`, `log1p_sinuosity`, `avg_speed_norm`, `rotation_timing`, `safe_alignment`, `zone_entry_time`, `outside_safe_duration_10min`, `boundary_distance_slope_norm`, `mean_dist_to_boundary_norm`, `min_dist_to_boundary_norm`, `outside_safe_ratio`, `hab_dist_to_safe_boundary_start_m`, `hab_dist_to_safe_boundary_end_m`, `hab_signed_boundary_distance_start_m`, `hab_signed_boundary_distance_end_m`, `hab_distance_to_target_safe_m`, `hab_required_speed_to_safe_mps`, `hab_observed_speed_mps`, `hab_migration_deficit_mps`, `hab_outside_safe_ratio`, `hab_edge_pressure_ratio`, `hab_rotation_delay_ratio`, `hab_safe_alignment_to_target`, `hab_local_enemy_count_100m`, `hab_local_enemy_count_200m`, `hab_local_enemy_count_500m`, `hab_local_enemy_density_200m`, `hab_safe_entry_enemy_count_200m`, `hab_safe_entry_congestion`, `hab_refuge_scarcity_simple`

### OM4 Threat Exposure (`om4_threat_exposure`)

Ontology role: O4 threat exposure / realized combat pressure

Run note: Adds realized damage, kills, groggy, combat initiation, and combat-zone exposure.

Feature count: total 79, node 54, global 25.

New features at this step:

- Global (0): none
- Node (13): `recent_damage_taken_norm`, `combat_zone_pass_count_norm`, `damage_dealt_norm`, `damage_taken_norm`, `damage_events_dealt_norm`, `damage_events_taken_norm`, `groggy_dealt_norm`, `groggy_taken_norm`, `kills_dealt_norm`, `kills_taken_norm`, `distinct_opponents_damaged_norm`, `distinct_opponents_damaged_by_norm`, `combat_initiated`

Cumulative features used by this preset:

- Global (25): `phase_idx_norm`, `phase_duration_10min`, `avg_alive_teams_norm`, `avg_alive_players_norm`, `safe_radius_start_norm`, `safe_radius_end_norm`, `safe_radius_delta_norm`, `safe_center_shift_norm`, `poison_radius_start_norm`, `poison_radius_end_norm`, `zone_gap_start_norm`, `zone_gap_end_norm`, `hab_safe_area_start_m2`, `hab_safe_area_end_m2`, `hab_safe_area_delta_m2`, `hab_loss_ratio`, `hab_shrink_rate_m2ps`, `hab_alive_teams`, `hab_safe_area_per_alive_team`, `hab_crowding_pressure`, `hab_center_shift_m`, `hab_center_shift_per_radius`, `hab_zone_gap_start_m`, `hab_zone_gap_end_m`, `hab_zone_gap_pressure`
- Node (54): `team_size_norm`, `alive_players_start_norm`, `alive_players_end_norm`, `alive_players_min_norm`, `alive_players_avg_norm`, `mean_health_avg_norm`, `min_health_min_norm`, `health_slope_norm`, `team_spread_avg_norm`, `team_spread_max_norm`, `path_length_norm`, `displacement_norm`, `log1p_sinuosity`, `avg_speed_norm`, `rotation_timing`, `safe_alignment`, `zone_entry_time`, `outside_safe_duration_10min`, `boundary_distance_slope_norm`, `mean_dist_to_boundary_norm`, `min_dist_to_boundary_norm`, `outside_safe_ratio`, `recent_damage_taken_norm`, `combat_zone_pass_count_norm`, `damage_dealt_norm`, `damage_taken_norm`, `damage_events_dealt_norm`, `damage_events_taken_norm`, `groggy_dealt_norm`, `groggy_taken_norm`, `kills_dealt_norm`, `kills_taken_norm`, `distinct_opponents_damaged_norm`, `distinct_opponents_damaged_by_norm`, `combat_initiated`, `hab_dist_to_safe_boundary_start_m`, `hab_dist_to_safe_boundary_end_m`, `hab_signed_boundary_distance_start_m`, `hab_signed_boundary_distance_end_m`, `hab_distance_to_target_safe_m`, `hab_required_speed_to_safe_mps`, `hab_observed_speed_mps`, `hab_migration_deficit_mps`, `hab_outside_safe_ratio`, `hab_edge_pressure_ratio`, `hab_rotation_delay_ratio`, `hab_safe_alignment_to_target`, `hab_local_enemy_count_100m`, `hab_local_enemy_count_200m`, `hab_local_enemy_count_500m`, `hab_local_enemy_density_200m`, `hab_safe_entry_enemy_count_200m`, `hab_safe_entry_congestion`, `hab_refuge_scarcity_simple`

### OM5 Relation Structure (`om5_relation_structure`)

Ontology role: O7 node-summary relation structure

Run note: Adds precomputed net_* and cum_* node summaries only. This run did not use graph message passing or edge streams.

Feature count: total 122, node 97, global 25.

New features at this step:

- Global (0): none
- Node (43): `net_combat_out_degree`, `net_combat_in_degree`, `net_combat_total_degree`, `net_combat_out_strength`, `net_combat_in_strength`, `net_combat_total_strength`, `net_combat_balance`, `net_combat_unique_targets`, `net_combat_unique_attackers`, `net_combat_reciprocal_pairs`, `net_combat_reciprocity_ratio`, `net_combat_out_strength_per_min`, `net_combat_in_strength_per_min`, `net_prox_degree_100m`, `net_prox_degree_200m`, `net_prox_degree_500m`, `net_prox_strength_decay`, `net_prox_mean_distance_m`, `net_prox_min_distance_m`, `net_prox_top1_weight`, `net_prox_top3_strength`, `net_prox_isolation_m`, `net_prox_neighbor_entropy`, `net_copress_degree`, `net_copress_strength`, `net_copress_max_weight`, `net_copress_mean_ratio`, `net_copress_shared_pressure_count`, `net_copress_top1_weight`, `cum_combat_out_degree`, `cum_combat_in_degree`, `cum_combat_out_strength`, `cum_combat_in_strength`, `cum_combat_total_strength`, `cum_combat_reciprocal_pairs`, `cum_unique_targets`, `cum_unique_attackers`, `cum_copress_strength`, `cum_prox_close_exposure_200m`, `net_combat_out_strength_per_alive_team`, `net_combat_in_strength_per_alive_team`, `net_prox_degree_200m_per_alive_team`, `net_copress_degree_per_alive_team`

Cumulative features used by this preset:

- Global (25): `phase_idx_norm`, `phase_duration_10min`, `avg_alive_teams_norm`, `avg_alive_players_norm`, `safe_radius_start_norm`, `safe_radius_end_norm`, `safe_radius_delta_norm`, `safe_center_shift_norm`, `poison_radius_start_norm`, `poison_radius_end_norm`, `zone_gap_start_norm`, `zone_gap_end_norm`, `hab_safe_area_start_m2`, `hab_safe_area_end_m2`, `hab_safe_area_delta_m2`, `hab_loss_ratio`, `hab_shrink_rate_m2ps`, `hab_alive_teams`, `hab_safe_area_per_alive_team`, `hab_crowding_pressure`, `hab_center_shift_m`, `hab_center_shift_per_radius`, `hab_zone_gap_start_m`, `hab_zone_gap_end_m`, `hab_zone_gap_pressure`
- Node (97): `team_size_norm`, `alive_players_start_norm`, `alive_players_end_norm`, `alive_players_min_norm`, `alive_players_avg_norm`, `mean_health_avg_norm`, `min_health_min_norm`, `health_slope_norm`, `team_spread_avg_norm`, `team_spread_max_norm`, `path_length_norm`, `displacement_norm`, `log1p_sinuosity`, `avg_speed_norm`, `rotation_timing`, `safe_alignment`, `zone_entry_time`, `outside_safe_duration_10min`, `boundary_distance_slope_norm`, `mean_dist_to_boundary_norm`, `min_dist_to_boundary_norm`, `outside_safe_ratio`, `recent_damage_taken_norm`, `combat_zone_pass_count_norm`, `damage_dealt_norm`, `damage_taken_norm`, `damage_events_dealt_norm`, `damage_events_taken_norm`, `groggy_dealt_norm`, `groggy_taken_norm`, `kills_dealt_norm`, `kills_taken_norm`, `distinct_opponents_damaged_norm`, `distinct_opponents_damaged_by_norm`, `combat_initiated`, `net_combat_out_degree`, `net_combat_in_degree`, `net_combat_total_degree`, `net_combat_out_strength`, `net_combat_in_strength`, `net_combat_total_strength`, `net_combat_balance`, `net_combat_unique_targets`, `net_combat_unique_attackers`, `net_combat_reciprocal_pairs`, `net_combat_reciprocity_ratio`, `net_combat_out_strength_per_min`, `net_combat_in_strength_per_min`, `net_prox_degree_100m`, `net_prox_degree_200m`, `net_prox_degree_500m`, `net_prox_strength_decay`, `net_prox_mean_distance_m`, `net_prox_min_distance_m`, `net_prox_top1_weight`, `net_prox_top3_strength`, `net_prox_isolation_m`, `net_prox_neighbor_entropy`, `net_copress_degree`, `net_copress_strength`, `net_copress_max_weight`, `net_copress_mean_ratio`, `net_copress_shared_pressure_count`, `net_copress_top1_weight`, `cum_combat_out_degree`, `cum_combat_in_degree`, `cum_combat_out_strength`, `cum_combat_in_strength`, `cum_combat_total_strength`, `cum_combat_reciprocal_pairs`, `cum_unique_targets`, `cum_unique_attackers`, `cum_copress_strength`, `cum_prox_close_exposure_200m`, `net_combat_out_strength_per_alive_team`, `net_combat_in_strength_per_alive_team`, `net_prox_degree_200m_per_alive_team`, `net_copress_degree_per_alive_team`, `hab_dist_to_safe_boundary_start_m`, `hab_dist_to_safe_boundary_end_m`, `hab_signed_boundary_distance_start_m`, `hab_signed_boundary_distance_end_m`, `hab_distance_to_target_safe_m`, `hab_required_speed_to_safe_mps`, `hab_observed_speed_mps`, `hab_migration_deficit_mps`, `hab_outside_safe_ratio`, `hab_edge_pressure_ratio`, `hab_rotation_delay_ratio`, `hab_safe_alignment_to_target`, `hab_local_enemy_count_100m`, `hab_local_enemy_count_200m`, `hab_local_enemy_count_500m`, `hab_local_enemy_density_200m`, `hab_safe_entry_enemy_count_200m`, `hab_safe_entry_congestion`, `hab_refuge_scarcity_simple`

### OM6 Spatial Proxy (`om6_spatial_proxy`)

Ontology role: Current O6 spatial-exposure proxies

Run note: Adds only avg enemy density, nearest enemy, and path-near-enemy ratio. The planned Z-ladder high-ground/relative-height features are not tested here.

Feature count: total 125, node 100, global 25.

New features at this step:

- Global (0): none
- Node (3): `avg_enemy_density_100m_norm`, `min_nearest_enemy_norm`, `path_near_enemy_ratio`

Cumulative features used by this preset:

- Global (25): `phase_idx_norm`, `phase_duration_10min`, `avg_alive_teams_norm`, `avg_alive_players_norm`, `safe_radius_start_norm`, `safe_radius_end_norm`, `safe_radius_delta_norm`, `safe_center_shift_norm`, `poison_radius_start_norm`, `poison_radius_end_norm`, `zone_gap_start_norm`, `zone_gap_end_norm`, `hab_safe_area_start_m2`, `hab_safe_area_end_m2`, `hab_safe_area_delta_m2`, `hab_loss_ratio`, `hab_shrink_rate_m2ps`, `hab_alive_teams`, `hab_safe_area_per_alive_team`, `hab_crowding_pressure`, `hab_center_shift_m`, `hab_center_shift_per_radius`, `hab_zone_gap_start_m`, `hab_zone_gap_end_m`, `hab_zone_gap_pressure`
- Node (100): `team_size_norm`, `alive_players_start_norm`, `alive_players_end_norm`, `alive_players_min_norm`, `alive_players_avg_norm`, `mean_health_avg_norm`, `min_health_min_norm`, `health_slope_norm`, `team_spread_avg_norm`, `team_spread_max_norm`, `path_length_norm`, `displacement_norm`, `log1p_sinuosity`, `avg_speed_norm`, `rotation_timing`, `safe_alignment`, `zone_entry_time`, `outside_safe_duration_10min`, `boundary_distance_slope_norm`, `mean_dist_to_boundary_norm`, `min_dist_to_boundary_norm`, `outside_safe_ratio`, `avg_enemy_density_100m_norm`, `min_nearest_enemy_norm`, `path_near_enemy_ratio`, `recent_damage_taken_norm`, `combat_zone_pass_count_norm`, `damage_dealt_norm`, `damage_taken_norm`, `damage_events_dealt_norm`, `damage_events_taken_norm`, `groggy_dealt_norm`, `groggy_taken_norm`, `kills_dealt_norm`, `kills_taken_norm`, `distinct_opponents_damaged_norm`, `distinct_opponents_damaged_by_norm`, `combat_initiated`, `net_combat_out_degree`, `net_combat_in_degree`, `net_combat_total_degree`, `net_combat_out_strength`, `net_combat_in_strength`, `net_combat_total_strength`, `net_combat_balance`, `net_combat_unique_targets`, `net_combat_unique_attackers`, `net_combat_reciprocal_pairs`, `net_combat_reciprocity_ratio`, `net_combat_out_strength_per_min`, `net_combat_in_strength_per_min`, `net_prox_degree_100m`, `net_prox_degree_200m`, `net_prox_degree_500m`, `net_prox_strength_decay`, `net_prox_mean_distance_m`, `net_prox_min_distance_m`, `net_prox_top1_weight`, `net_prox_top3_strength`, `net_prox_isolation_m`, `net_prox_neighbor_entropy`, `net_copress_degree`, `net_copress_strength`, `net_copress_max_weight`, `net_copress_mean_ratio`, `net_copress_shared_pressure_count`, `net_copress_top1_weight`, `cum_combat_out_degree`, `cum_combat_in_degree`, `cum_combat_out_strength`, `cum_combat_in_strength`, `cum_combat_total_strength`, `cum_combat_reciprocal_pairs`, `cum_unique_targets`, `cum_unique_attackers`, `cum_copress_strength`, `cum_prox_close_exposure_200m`, `net_combat_out_strength_per_alive_team`, `net_combat_in_strength_per_alive_team`, `net_prox_degree_200m_per_alive_team`, `net_copress_degree_per_alive_team`, `hab_dist_to_safe_boundary_start_m`, `hab_dist_to_safe_boundary_end_m`, `hab_signed_boundary_distance_start_m`, `hab_signed_boundary_distance_end_m`, `hab_distance_to_target_safe_m`, `hab_required_speed_to_safe_mps`, `hab_observed_speed_mps`, `hab_migration_deficit_mps`, `hab_outside_safe_ratio`, `hab_edge_pressure_ratio`, `hab_rotation_delay_ratio`, `hab_safe_alignment_to_target`, `hab_local_enemy_count_100m`, `hab_local_enemy_count_200m`, `hab_local_enemy_count_500m`, `hab_local_enemy_density_200m`, `hab_safe_entry_enemy_count_200m`, `hab_safe_entry_congestion`, `hab_refuge_scarcity_simple`

## Exact Added Feature Groups

This section keeps the conceptual block ownership explicit. It is useful when checking that future code changes do not silently move features between ontology blocks.

### OM0 added

- `phase_idx_norm`
- `phase_duration_10min`
- `team_size_norm`

### OM1 added

- `alive_players_start_norm`
- `alive_players_end_norm`
- `alive_players_min_norm`
- `alive_players_avg_norm`
- `mean_health_avg_norm`
- `min_health_min_norm`
- `health_slope_norm`
- `team_spread_avg_norm`
- `team_spread_max_norm`

### OM2 added

- `path_length_norm`
- `displacement_norm`
- `log1p_sinuosity`
- `avg_speed_norm`
- `rotation_timing`
- `safe_alignment`
- `zone_entry_time`

### OM3 pressure global added

- `avg_alive_teams_norm`
- `avg_alive_players_norm`
- `safe_radius_start_norm`
- `safe_radius_end_norm`
- `safe_radius_delta_norm`
- `safe_center_shift_norm`
- `poison_radius_start_norm`
- `poison_radius_end_norm`
- `zone_gap_start_norm`
- `zone_gap_end_norm`

### OM3 local pressure added

- `outside_safe_duration_10min`
- `boundary_distance_slope_norm`
- `mean_dist_to_boundary_norm`
- `min_dist_to_boundary_norm`
- `outside_safe_ratio`

### OM3 habitat added

- `hab_safe_area_start_m2`
- `hab_safe_area_end_m2`
- `hab_safe_area_delta_m2`
- `hab_loss_ratio`
- `hab_shrink_rate_m2ps`
- `hab_alive_teams`
- `hab_safe_area_per_alive_team`
- `hab_crowding_pressure`
- `hab_center_shift_m`
- `hab_center_shift_per_radius`
- `hab_zone_gap_start_m`
- `hab_zone_gap_end_m`
- `hab_zone_gap_pressure`
- `hab_dist_to_safe_boundary_start_m`
- `hab_dist_to_safe_boundary_end_m`
- `hab_signed_boundary_distance_start_m`
- `hab_signed_boundary_distance_end_m`
- `hab_distance_to_target_safe_m`
- `hab_required_speed_to_safe_mps`
- `hab_observed_speed_mps`
- `hab_migration_deficit_mps`
- `hab_outside_safe_ratio`
- `hab_edge_pressure_ratio`
- `hab_rotation_delay_ratio`
- `hab_safe_alignment_to_target`
- `hab_local_enemy_count_100m`
- `hab_local_enemy_count_200m`
- `hab_local_enemy_count_500m`
- `hab_local_enemy_density_200m`
- `hab_safe_entry_enemy_count_200m`
- `hab_safe_entry_congestion`
- `hab_refuge_scarcity_simple`

### OM4 added

- `recent_damage_taken_norm`
- `combat_zone_pass_count_norm`
- `damage_dealt_norm`
- `damage_taken_norm`
- `damage_events_dealt_norm`
- `damage_events_taken_norm`
- `groggy_dealt_norm`
- `groggy_taken_norm`
- `kills_dealt_norm`
- `kills_taken_norm`
- `distinct_opponents_damaged_norm`
- `distinct_opponents_damaged_by_norm`
- `combat_initiated`

### OM5 added

- `net_combat_out_degree`
- `net_combat_in_degree`
- `net_combat_total_degree`
- `net_combat_out_strength`
- `net_combat_in_strength`
- `net_combat_total_strength`
- `net_combat_balance`
- `net_combat_unique_targets`
- `net_combat_unique_attackers`
- `net_combat_reciprocal_pairs`
- `net_combat_reciprocity_ratio`
- `net_combat_out_strength_per_min`
- `net_combat_in_strength_per_min`
- `net_prox_degree_100m`
- `net_prox_degree_200m`
- `net_prox_degree_500m`
- `net_prox_strength_decay`
- `net_prox_mean_distance_m`
- `net_prox_min_distance_m`
- `net_prox_top1_weight`
- `net_prox_top3_strength`
- `net_prox_isolation_m`
- `net_prox_neighbor_entropy`
- `net_copress_degree`
- `net_copress_strength`
- `net_copress_max_weight`
- `net_copress_mean_ratio`
- `net_copress_shared_pressure_count`
- `net_copress_top1_weight`
- `cum_combat_out_degree`
- `cum_combat_in_degree`
- `cum_combat_out_strength`
- `cum_combat_in_strength`
- `cum_combat_total_strength`
- `cum_combat_reciprocal_pairs`
- `cum_unique_targets`
- `cum_unique_attackers`
- `cum_copress_strength`
- `cum_prox_close_exposure_200m`
- `net_combat_out_strength_per_alive_team`
- `net_combat_in_strength_per_alive_team`
- `net_prox_degree_200m_per_alive_team`
- `net_copress_degree_per_alive_team`

### OM6 added

- `avg_enemy_density_100m_norm`
- `min_nearest_enemy_norm`
- `path_near_enemy_ratio`

## Claim Boundary

- This result supports the `O2 x O6 = H` habitat claim because `OM2 -> OM3` is positive in all 13 cohorts.
- `OM1` should not be interpreted as truly useless until phase-stratified evaluation separates between-phase baseline discrimination from within-phase survival-capital signal.
- `OM5` tests only precomputed `net_*` and `cum_*` node summaries. It does not test graph message passing, edge streams, or edge-shuffle controls.
- `OM6` is a current spatial-exposure proxy block. It does not test the planned PUBG Z-ladder of vertical position, contact-gated relative height, and zone interaction features.
- All reported results are single-seed results. `OM3` is robust by sign consistency, while smaller `OM4`, `OM5`, and `OM6` increments require bootstrap confidence intervals or seed repeats before strong claims.

## Immediate Follow-Ups

1. Commit the result CSV, summary JSON, run index, and this audit document.
2. Run phase-stratified aggregation to separate pooled phase effects from within-phase predictive value.
3. Add paired bootstrap confidence intervals for OM deltas, especially `OM4`, `OM5`, and `OM6`.
4. Implement and rerun the intended O6 Z-ladder block before making spatial-affordance claims.
5. Run the true O7 edge-stream / graph-relation matrix after the N-null guard decision.
