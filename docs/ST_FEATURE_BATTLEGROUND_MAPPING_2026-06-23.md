# Battleground Domain ST Feature Mapping (2026-06-23)

## Purpose

This table maps prior-work spatio-temporal representation concepts to the
Battleground/PUBG survival domain and to the current project feature/channel
contract.

Companion documents:

```text
docs/ST_FEATURE_PRIOR_ALIGNED_DESIGN_2026-06-23.md
docs/ST_FEATURE_DEFINITION_2026-06-23.md
docs/GENRE_PRIMITIVE_FEATURE_MANIFEST.csv
```

## Mapping Table

| Prior-work axis | Battleground domain reinterpretation | Redesigned feature block | Current physical/model features | Stream / clock | Paper hypothesis | Novelty claim |
| --- | --- | --- | --- | --- | --- | --- |
| Trajectory data mining | A team is a moving survival actor, not a static row. Raw player positions are projected to a team centroid/path. | Team trajectory state | `pos_x`, `pos_y`, `pos_z`, `speed_mps`, `path_length_m`, `displacement_m`, `sinuosity`, `avg_speed_mps` | `X_s` at `snapshot_10s`; `X_phase` at `phase_window` | Movement history before a phase cut predicts next-phase survival better than a final/latest aggregate alone. | Team-level trajectory is evaluated under phase survival cuts instead of post-game placement rows. |
| Multi-agent team ST tracking | Survival depends on the team's own route plus the local opponent field around that route. | Local opponent field | `edge_snapshot_proximity`, `avg_enemy_density_100m`, `min_nearest_enemy_m`, `path_near_enemy_ratio`, `net_prox_*`, `hab_local_enemy_count_*` | `A_s` at `snapshot_10s`; `X_phase` at `phase_window` | Opponent field adds signal beyond team-only temporal dynamics. | Nearby teams represent survival dependence, route conflict, and safe-entry competition, not homophily. |
| ST forecasting with external context | The arena is not only a background map; phase, safe-zone geometry, and population pressure condition risk. | Domain and episode context | `map_name`, `game_mode`, `perspective`, `team_size`, phase index/duration, `hab_alive_teams`, `hab_safe_area_per_alive_team`, `hab_crowding_pressure` | `U` / global context; `match_static` or `phase_window` | Risk should be stratified or conditioned by phase/map/mode context. | Battle Royale rule-set and phase horizon are part of the survival process, not only metadata. |
| ST forecasting with spatial constraints | The safe zone creates a moving hazard field. Spatial meaning changes as the zone shrinks and shifts. | Shrinking-arena / zone constraint | `rel_zone_x`, `rel_zone_y`, `dist_blue_boundary_m`, `inside_blue`, `dist_target_center_m`, `outside_safe_duration_s`, `outside_safe_ratio`, `hab_distance_to_target_safe_m`, `hab_required_speed_to_safe_mps`, `hab_migration_deficit_mps`, `hab_safe_alignment_to_target` | `X_s`, `X_phase`, `U`; `snapshot_10s` or `phase_window` | Zone-relative ST features should explain survival pressure beyond absolute map coordinates. | The key spatial frame is dynamic safe-space compression, unlike fixed-road or fixed-court ST tasks. |
| Game telemetry / behavioral profiling | Sparse logs become behavior mechanisms only when grouped by clock and causal window. | Interaction event field | `edge_snapshot_combat`, `damage_dealt`, `damage_taken`, kills/groggy dealt/taken, `combat_initiated`, recent damage, third-party exposure | `A_s` at trailing `10s`; `X_phase` at `phase_window` | Directed combat pressure before the cut improves hazard prediction. | Combat is treated as a time-windowed directed pressure relation, not a match-total statistic. |
| Time-varying covariates | Survival capital changes during the match: roster count, health, mobility, and resources. | Team viability and resource state | `mean_health`, `min_health`, `alive_n`, `alive_players_*`, `team_spread_*`, `vehicle_ratio`, pickup/equip/use/heal/boost counts | `X_s` at `snapshot_10s`; `X_phase` at `phase_window` | Current and recent survival capital should explain short-horizon elimination risk. | Player telemetry is projected into team survival capital before the phase-cut target. |
| ST graph learning | Edges are typed survival dependencies: proximity, combat, co-pressure, forced convergence, or relative advantage. | Typed relation topology | `edge_snapshot_proximity`, `edge_snapshot_combat`, `net_combat_*`, `net_copress_*`, optional `E_s` relative-edge features | `A_s`, `A_phase`, optional `E_s`; snapshot or phase clocks | Graph channels matter only if observed relations beat no-graph and graph-null controls. | The graph is a competitive hazard/exposure graph, not a social graph. |
| Dynamic survival / hazard modeling | The prediction row is an at-risk team at a phase cut, not a whole-match independent sample. | Survival-cut label design | `y_hazard`, at-risk mask, phase-cut labels; not input features | `Y` and mask only | Causal cut alignment makes ST feature interpretation valid. | The model predicts next-phase elimination risk from observations available through phase `k`. |
| PUBG prior aggregate baselines | Post-game aggregate predictors are useful baselines but not causal ST representations. | Baseline contrast block | Kaggle-style totals or Sagong-style phase strategy aggregates where available | baseline / `X_phase` depending on implementation | ST representation should be compared against aggregate/static baselines. | Our main claim is phase-causal team-level ST database representation, not simply PUBG feature engineering. |

## Paper-Ready Compression

Use the following short mapping in the paper:

| Reference concept | PUBG/Battleground feature family | Stream | Example features | What is new here |
| --- | --- | --- | --- | --- |
| trajectory state | team trajectory | `X_s`, `X_phase` | position, speed, path length, sinuosity | team path under phase survival cuts |
| external/spatial context | shrinking-arena pressure | `X_s`, `U`, `X_phase` | zone-relative position, boundary distance, required speed | dynamic safe-space frame |
| time-varying covariates | viability/resource capital | `X_s`, `X_phase` | health, alive count, vehicles, resources | roster survival state |
| event-log behavior | combat/event pressure | `A_s`, `X_phase` | combat edge, damage, kills | directed pressure before cut |
| multi-agent tracking | local opponent field | `A_s`, `X_phase` | proximity, density, nearest enemy | opponent field as survival dependence |
| ST graph learning | typed survival topology | `A_s`, `A_phase`, `E_s` | proximity, combat, co-pressure, relative edge | null-controlled competitive graph |
| dynamic survival | phase-cut hazard | `Y`, mask | at-risk team x phase | dynamic risk, not post-game row |

## Reference Sources

```text
Wang et al., trajectory data management survey:
https://arxiv.org/abs/2003.11547

Gudmundsson and Horton, team-sports ST survey:
https://arxiv.org/abs/1602.06994

Zhang et al., ST-ResNet:
https://arxiv.org/abs/1701.02543

Yu et al., STGCN:
https://arxiv.org/abs/1709.04875

Drachen et al., behavioral telemetry clustering:
https://arxiv.org/abs/1407.3950

Bunian et al., HMM game behavior:
https://arxiv.org/abs/1804.00245

Rokad et al., PUBG aggregate prediction:
https://arxiv.org/abs/1905.06052

Jarrett et al., MATCH-Net:
https://arxiv.org/abs/1811.10746

Rava and Bradic, DeepHazard:
https://arxiv.org/abs/2007.13218

PUBG telemetry events:
https://documentation.pubg.com/en/telemetry-events.html
```
