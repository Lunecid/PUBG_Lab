# PUBG Public Telemetry and Local DB Feature Inventory

Generated: 2026-06-23

Sources: official PUBG Developer Documentation telemetry pages and the local PostgreSQL information_schema for schemas `pubg` and `stdb`.

## 1. Official PUBG telemetry event schemas

Every telemetry event has `_D`, `_T`, and `common`; those fields are included below for clarity.

### LogArmorDestroy
`_D`, `_T`, `common`, `attackId`, `attacker`, `victim`, `damageTypeCategory`, `damageReason`, `damageCauserName`, `item`, `distance`

### LogBlackZoneEnded
`_D`, `_T`, `common`, `survivors`

### LogCarePackageLand
`_D`, `_T`, `common`, `itemPackage`

### LogCarePackageSpawn
`_D`, `_T`, `common`, `itemPackage`

### LogCharacterCarry
`_D`, `_T`, `common`, `character`, `carryState`

### LogEmPickupLiftOff
`_D`, `_T`, `common`, `instigator`, `riders`

### LogGameStatePeriodic
`_D`, `_T`, `common`, `gameState`

### LogHeal
`_D`, `_T`, `common`, `character`, `item`, `healamount`

### LogItemAttach
`_D`, `_T`, `common`, `character`, `parentItem`, `childItem`

### LogItemDetach
`_D`, `_T`, `common`, `character`, `parentItem`, `childItem`

### LogItemDrop
`_D`, `_T`, `common`, `character`, `item`

### LogItemEquip
`_D`, `_T`, `common`, `character`, `item`

### LogItemPickup
`_D`, `_T`, `common`, `character`, `item`

### LogItemPickupFromCarepackage
`_D`, `_T`, `common`, `character`, `item`, `carePackageUniqueId`

### LogItemPickupFromCustomPackage
`_D`, `_T`, `common`, `character`, `item`

### LogItemPickupFromLootbox
`_D`, `_T`, `common`, `character`, `item`, `ownerTeamId`, `creatorAccountId`

### LogItemPickupFromVehicleTrunk
`_D`, `_T`, `common`, `character`, `vehicle`, `item`

### LogItemPutToVehicleTrunk
`_D`, `_T`, `common`, `character`, `vehicle`, `item`

### LogItemUnequip
`_D`, `_T`, `common`, `character`, `item`

### LogItemUse
`_D`, `_T`, `common`, `character`, `item`

### LogMatchDefinition
`_D`, `_T`, `common`, `MatchId`, `PingQuality`, `SeasonState`

### LogMatchEnd
`_D`, `_T`, `common`, `characters`, `gameResultOnFinished`

### LogMatchStart
`_D`, `_T`, `common`, `mapName`, `weatherId`, `characters`, `cameraViewBehaviour`, `teamSize`, `isCustomGame`, `isEventMode`, `blueZoneCustomOptions`

### LogObjectDestroy
`_D`, `_T`, `common`, `character`, `objectType`, `objectLocation`

### LogObjectInteraction
`_D`, `_T`, `common`, `character`, `objectType`, `objectTypeStatus`, `objectTypeAdditionalInfo`

### LogParachuteLanding
`_D`, `_T`, `common`, `character`, `distance`

### LogPhaseChange
`_D`, `_T`, `common`, `phase`, `elapsedTime`

### LogPlayerAttack
`_D`, `_T`, `common`, `attackId`, `fireWeaponStackCount`, `attacker`, `attackType`, `weapon`, `vehicle`

### LogPlayerCreate
`_D`, `_T`, `common`, `character`

### LogPlayerDestroyBreachableWall
`_D`, `_T`, `common`, `attacker`, `weapon`

### LogPlayerDestroyProp
`_D`, `_T`, `common`, `attacker`, `objectType`, `objectLocation`

### LogPlayerKill (tournament matches)
`_D`, `_T`, `common`, `attackId`, `killer`, `victim`, `assistant`, `dBNOId`, `damageReason`, `damageTypeCategory`, `damageCauserName`, `damageCauserAdditionalInfo`, `victimWeapon`, `victimWeaponAdditionalInfo`, `distance`, `victimGameResult`, `isThroughPenetrableWall`

### LogPlayerKillV2
`_D`, `_T`, `common`, `attackId`, `dBNOId`, `victimGameResult`, `victim`, `victimWeapon`, `victimWeaponAdditionalInfo`, `dBNOMaker`, `dBNODamageInfo`, `finisher`, `finishDamageInfo`, `killer`, `killerDamageInfo`, `assists_AccountId`, `teamKillers_AccountId`, `isSuicide`

### LogPlayerLogin
`_D`, `_T`, `common`, `accountId`

### LogPlayerLogout
`_D`, `_T`, `common`, `accountId`

### LogPlayerMakeGroggy
`_D`, `_T`, `common`, `attackId`, `attacker`, `victim`, `damageReason`, `damageTypeCategory`, `damageCauserName`, `damageCauserAdditionalInfo`, `VictimWeapon`, `VictimWeaponAdditionalInfo`, `distance`, `isAttackerInVehicle`, `dBNOId`, `isThroughPenetrableWall`

### LogPlayerPosition
`_D`, `_T`, `common`, `character`, `vehicle`, `elapsedTime`, `numAlivePlayers`

### LogPlayerRedeploy
`_D`, `_T`, `common`, `character`

### LogPlayerRedeployBRStart
`_D`, `_T`, `common`, `characters`

### LogPlayerRevive
`_D`, `_T`, `common`, `reviver`, `victim`, `dBNOId`

### LogPlayerTakeDamage
`_D`, `_T`, `common`, `attackId`, `attacker`, `victim`, `damageTypeCategory`, `damageReason`, `damage`, `damageCauserName`, `isThroughPenetrableWall`

### LogPlayerUseFlareGun
`_D`, `_T`, `common`, `attackId`, `fireWeaponStackCount`, `attacker`, `attackType`, `weapon`

### LogPlayerUseThrowable
`_D`, `_T`, `common`, `attackId`, `fireWeaponStackCount`, `attacker`, `attackType`, `weapon`

### LogRedZoneEnded
`_D`, `_T`, `common`, `drivers`

### LogSwimEnd
`_D`, `_T`, `common`, `character`, `swimDistance`, `maxSwimDepthOfWater`

### LogSwimStart
`_D`, `_T`, `common`, `character`

### LogVaultStart
`_D`, `_T`, `common`, `character`, `isLedgeGrab`

### LogVehicleDamage
`_D`, `_T`, `common`, `attackId`, `attacker`, `vehicle`, `damageTypeCategory`, `damageCauserName`, `damage`, `distance`

### LogVehicleDestroy
`_D`, `_T`, `common`, `attackId`, `attacker`, `vehicle`, `damageTypeCategory`, `damageCauserName`, `distance`

### LogVehicleLeave
`_D`, `_T`, `common`, `character`, `vehicle`, `rideDistance`, `seatIndex`, `maxSpeed`, `fellowPassengers`

### LogVehicleRide
`_D`, `_T`, `common`, `character`, `vehicle`, `seatIndex`, `fellowPassengers`

### LogWeaponFireCount
`_D`, `_T`, `common`, `character`, `weaponId`, `fireCount`

### LogWheelDestroy
`_D`, `_T`, `common`, `attackId`, `attacker`, `vehicle`, `damageTypeCategory`, `damageCauserName`

## 2. Official PUBG telemetry object schemas

### BlueZoneCustomOptions
`phaseNum`, `startDelay`, `warningDuration`, `releaseDuration`, `poisonGasDamagePerSecond`, `radiusRate`, `spreadRatio`, `landRatio`, `circleAlgorithm`

### Character
`name`, `teamId`, `health`, `location`, `ranking`, `accountId`, `isInBlueZone`, `isInRedZone`, `zone`

### CharacterWrapper
`character`, `primaryWeaponFirst`, `primaryWeaponSecond`, `secondaryWeapon`, `spawnKitIndex`

### Common
`isGame`

### DamageInfo
`damageReason`, `damageTypeCategory`, `damageCauserName`, `additionalInfo`, `distance`, `isThroughPenetrableWall`

### GameResult
`rank`, `gameResult`, `teamId`, `stats`, `accountId`

### GameResultOnFinished
`results`

### GameState
`elapsedTime`, `numAliveTeams`, `numJoinPlayers`, `numStartPlayers`, `numAlivePlayers`, `safetyZonePosition`, `safetyZoneRadius`, `poisonGasWarningPosition`, `poisonGasWarningRadius`, `redZonePosition`, `redZoneRadius`, `blackZonePosition`, `blackZoneRadius`

### Item
`itemId`, `stackCount`, `category`, `subCategory`, `attachedItems`

### ItemPackage
`itemPackageId`, `location`, `items`

### Location
`x`, `y`, `z`

### Stats
`killCount`, `distanceOnFoot`, `distanceOnSwim`, `distanceOnVehicle`, `distanceOnParachute`, `distanceOnFreefall`

### Vehicle
`vehicleType`, `vehicleId`, `vehicleUniqueId`, `healthPercent`, `feulPercent`, `altitudeAbs`, `altitudeRel`, `velocity`, `seatIndex`, `isWheelsInAir`, `isInWaterVolume`, `isEngineOn`

## 3. Local DB schemas: pubg.* and stdb.*

Total tables/views: 52
Total columns: 1172

### pubg.api_rate_log (7 columns)

- `id`: bigint
- `called_at`: timestamp with time zone
- `endpoint`: text
- `status_code`: integer
- `rate_remaining`: integer
- `rate_reset`: timestamp with time zone
- `response_ms`: integer

### pubg.collection_log (10 columns)

- `match_id`: text
- `shard_id`: text
- `discovered_at`: timestamp with time zone
- `match_fetched`: boolean
- `match_fetched_at`: timestamp with time zone
- `telemetry_url`: text
- `telemetry_fetched`: boolean
- `telemetry_fetched_at`: timestamp with time zone
- `error_message`: text
- `retry_count`: integer

### pubg.matches (12 columns)

- `match_id`: text
- `created_at`: timestamp with time zone
- `duration`: integer
- `game_mode`: text
- `map_name`: text
- `is_custom`: boolean
- `match_type`: text
- `season_state`: text
- `shard_id`: text
- `title_id`: text
- `telemetry_url`: text
- `collected_at`: timestamp with time zone

### pubg.participants (27 columns)

- `participant_id`: text
- `match_id`: text
- `roster_id`: text
- `player_id`: text
- `player_name`: text
- `kills`: integer
- `assists`: integer
- `dbnos`: integer
- `damage_dealt`: numeric
- `headshot_kills`: integer
- `longest_kill`: numeric
- `most_damage`: numeric
- `kill_place`: integer
- `kill_streaks`: integer
- `road_kills`: integer
- `team_kills`: integer
- `vehicle_destroys`: integer
- `death_type`: text
- `time_survived`: numeric
- `win_place`: integer
- `walk_distance`: numeric
- `ride_distance`: numeric
- `swim_distance`: numeric
- `boosts`: integer
- `heals`: integer
- `weapons_acquired`: integer
- `revives`: integer

### pubg.rosters (5 columns)

- `roster_id`: text
- `match_id`: text
- `rank`: integer
- `team_id`: integer
- `won`: boolean

### pubg.telem_damage (20 columns)

- `id`: bigint
- `match_id`: text
- `event_time`: timestamp with time zone
- `is_game`: numeric
- `attacker_id`: text
- `attacker_name`: text
- `attacker_x`: numeric
- `attacker_y`: numeric
- `attacker_z`: numeric
- `victim_id`: text
- `victim_name`: text
- `victim_x`: numeric
- `victim_y`: numeric
- `victim_z`: numeric
- `damage`: numeric
- `damage_reason`: text
- `damage_type`: text
- `damage_causer`: text
- `distance`: numeric
- `is_through_wall`: boolean

### pubg.telem_game_states (25 columns)

- `id`: bigint
- `match_id`: text
- `event_time`: timestamp with time zone
- `is_game`: numeric
- `elapsed_time`: integer
- `num_alive_teams`: integer
- `num_alive_players`: integer
- `num_join_players`: integer
- `num_start_players`: integer
- `safe_zone_x`: numeric
- `safe_zone_y`: numeric
- `safe_zone_z`: numeric
- `safe_zone_radius`: numeric
- `poison_zone_x`: numeric
- `poison_zone_y`: numeric
- `poison_zone_z`: numeric
- `poison_zone_radius`: numeric
- `red_zone_x`: numeric
- `red_zone_y`: numeric
- `red_zone_z`: numeric
- `red_zone_radius`: numeric
- `black_zone_x`: numeric
- `black_zone_y`: numeric
- `black_zone_z`: numeric
- `black_zone_radius`: numeric

### pubg.telem_groggy (19 columns)

- `id`: bigint
- `match_id`: text
- `event_time`: timestamp with time zone
- `is_game`: numeric
- `attacker_id`: text
- `attacker_name`: text
- `attacker_x`: numeric
- `attacker_y`: numeric
- `attacker_z`: numeric
- `victim_id`: text
- `victim_name`: text
- `victim_x`: numeric
- `victim_y`: numeric
- `victim_z`: numeric
- `damage_reason`: text
- `damage_type`: text
- `damage_causer`: text
- `distance`: numeric
- `is_attacker_in_vehicle`: boolean

### pubg.telem_item_equip (15 columns)

- `id`: bigint
- `match_id`: text
- `event_time`: timestamp with time zone
- `is_game`: numeric
- `event_type`: text
- `account_id`: text
- `player_name`: text
- `team_id`: integer
- `pos_x`: numeric
- `pos_y`: numeric
- `pos_z`: numeric
- `item_id`: text
- `category`: text
- `sub_category`: text
- `attached_items`: ARRAY

### pubg.telem_item_pickup (19 columns)

- `id`: bigint
- `match_id`: text
- `event_time`: timestamp with time zone
- `is_game`: numeric
- `event_type`: text
- `account_id`: text
- `player_name`: text
- `team_id`: integer
- `pos_x`: numeric
- `pos_y`: numeric
- `pos_z`: numeric
- `item_id`: text
- `category`: text
- `sub_category`: text
- `stack_count`: integer
- `attached_items`: ARRAY
- `carepackage_id`: numeric
- `lootbox_owner_team_id`: integer
- `lootbox_creator_id`: text

### pubg.telem_item_use (14 columns)

- `id`: bigint
- `match_id`: text
- `event_time`: timestamp with time zone
- `is_game`: numeric
- `account_id`: text
- `player_name`: text
- `team_id`: integer
- `pos_x`: numeric
- `pos_y`: numeric
- `pos_z`: numeric
- `item_id`: text
- `category`: text
- `sub_category`: text
- `stack_count`: integer

### pubg.telem_kills (26 columns)

- `id`: bigint
- `match_id`: text
- `event_time`: timestamp with time zone
- `is_game`: numeric
- `killer_id`: text
- `killer_name`: text
- `killer_x`: numeric
- `killer_y`: numeric
- `killer_z`: numeric
- `victim_id`: text
- `victim_name`: text
- `victim_x`: numeric
- `victim_y`: numeric
- `victim_z`: numeric
- `dbno_maker_id`: text
- `dbno_maker_name`: text
- `finisher_id`: text
- `finisher_name`: text
- `damage_reason`: text
- `damage_type`: text
- `damage_causer`: text
- `distance`: numeric
- `is_suicide`: boolean
- `assists_cnt`: integer
- `victim_rank`: integer
- `victim_team_id`: integer

### pubg.telem_match_end (4 columns)

- `id`: bigint
- `match_id`: text
- `event_time`: timestamp with time zone
- `game_results`: jsonb

### pubg.telem_match_start (10 columns)

- `id`: bigint
- `match_id`: text
- `event_time`: timestamp with time zone
- `map_name`: text
- `weather_id`: text
- `camera_view_type`: text
- `team_size`: integer
- `is_custom`: boolean
- `is_event_mode`: boolean
- `blue_zone_options`: jsonb

### pubg.telem_parachute_landing (10 columns)

- `id`: bigint
- `match_id`: text
- `event_time`: timestamp with time zone
- `account_id`: text
- `player_name`: text
- `team_id`: integer
- `pos_x`: numeric
- `pos_y`: numeric
- `pos_z`: numeric
- `distance`: numeric

### pubg.telem_phase_changes (6 columns)

- `id`: bigint
- `match_id`: text
- `event_time`: timestamp with time zone
- `is_game`: numeric
- `phase`: numeric
- `elapsed_time`: integer

### pubg.telem_positions (16 columns)

- `id`: bigint
- `match_id`: text
- `event_time`: timestamp with time zone
- `is_game`: numeric
- `elapsed_time`: integer
- `num_alive`: integer
- `account_id`: text
- `player_name`: text
- `team_id`: integer
- `health`: numeric
- `pos_x`: numeric
- `pos_y`: numeric
- `pos_z`: numeric
- `vehicle_type`: text
- `vehicle_id`: text
- `vehicle_speed`: numeric

### pubg.telem_positions_2026_03 (16 columns)

- `id`: bigint
- `match_id`: text
- `event_time`: timestamp with time zone
- `is_game`: numeric
- `elapsed_time`: integer
- `num_alive`: integer
- `account_id`: text
- `player_name`: text
- `team_id`: integer
- `health`: numeric
- `pos_x`: numeric
- `pos_y`: numeric
- `pos_z`: numeric
- `vehicle_type`: text
- `vehicle_id`: text
- `vehicle_speed`: numeric

### pubg.telem_positions_2026_04 (16 columns)

- `id`: bigint
- `match_id`: text
- `event_time`: timestamp with time zone
- `is_game`: numeric
- `elapsed_time`: integer
- `num_alive`: integer
- `account_id`: text
- `player_name`: text
- `team_id`: integer
- `health`: numeric
- `pos_x`: numeric
- `pos_y`: numeric
- `pos_z`: numeric
- `vehicle_type`: text
- `vehicle_id`: text
- `vehicle_speed`: numeric

### pubg.telem_positions_2026_05 (16 columns)

- `id`: bigint
- `match_id`: text
- `event_time`: timestamp with time zone
- `is_game`: numeric
- `elapsed_time`: integer
- `num_alive`: integer
- `account_id`: text
- `player_name`: text
- `team_id`: integer
- `health`: numeric
- `pos_x`: numeric
- `pos_y`: numeric
- `pos_z`: numeric
- `vehicle_type`: text
- `vehicle_id`: text
- `vehicle_speed`: numeric

### pubg.telem_positions_2026_06 (16 columns)

- `id`: bigint
- `match_id`: text
- `event_time`: timestamp with time zone
- `is_game`: numeric
- `elapsed_time`: integer
- `num_alive`: integer
- `account_id`: text
- `player_name`: text
- `team_id`: integer
- `health`: numeric
- `pos_x`: numeric
- `pos_y`: numeric
- `pos_z`: numeric
- `vehicle_type`: text
- `vehicle_id`: text
- `vehicle_speed`: numeric

### pubg.v_collection_status (6 columns)

- `total_discovered`: bigint
- `matches_fetched`: bigint
- `telemetry_fetched`: bigint
- `errors`: bigint
- `oldest_discovery`: timestamp with time zone
- `latest_discovery`: timestamp with time zone

### pubg.v_match_summary (11 columns)

- `match_id`: text
- `created_at`: timestamp with time zone
- `duration`: integer
- `game_mode`: text
- `map_name`: text
- `team_count`: bigint
- `player_count`: bigint
- `max_kills`: integer
- `avg_damage`: numeric
- `match_fetched`: boolean
- `telemetry_fetched`: boolean

### stdb.dynamic_regions (7 columns)

- `match_id`: text
- `region_kind`: text
- `snapshot_t`: integer
- `center_pos`: USER-DEFINED
- `radius_m`: real
- `region`: USER-DEFINED
- `phase_idx`: integer

### stdb.match_phase_global (28 columns)

- `match_id`: text
- `phase_idx`: integer
- `phase_start_s`: real
- `phase_end_s`: real
- `phase_duration_s`: real
- `avg_alive_teams`: real
- `start_alive_teams`: integer
- `end_alive_teams`: integer
- `avg_alive_players`: real
- `start_alive_players`: integer
- `end_alive_players`: integer
- `safe_radius_start_m`: real
- `safe_radius_end_m`: real
- `safe_radius_delta_m`: real
- `safe_center_shift_m`: real
- `poison_radius_start_m`: real
- `poison_radius_end_m`: real
- `poison_radius_delta_m`: real
- `zone_gap_start_m`: real
- `zone_gap_end_m`: real
- `safe_area_ratio_start`: real
- `safe_area_ratio_end`: real
- `red_zone_active_ratio`: real
- `black_zone_active_ratio`: real
- `weather_id`: text
- `camera_view_type`: text
- `blue_zone_options`: jsonb
- `created_at`: timestamp with time zone

### stdb.matches (9 columns)

- `match_id`: text
- `started_at`: timestamp with time zone
- `duration_s`: integer
- `map_name`: text
- `game_mode`: text
- `team_size`: integer
- `is_custom`: boolean
- `num_teams`: integer
- `num_players`: integer

### stdb.moving_objects (9 columns)

- `match_id`: text
- `agent_kind`: text
- `agent_id`: text
- `team_id`: integer
- `snapshot_t`: integer
- `pos`: USER-DEFINED
- `health`: real
- `in_vehicle`: boolean
- `is_alive`: boolean

### stdb.spatial_events (15 columns)

- `event_id`: bigint
- `match_id`: text
- `event_kind`: text
- `event_t`: real
- `attacker_id`: text
- `attacker_team`: integer
- `attacker_pos`: USER-DEFINED
- `victim_id`: text
- `victim_team`: integer
- `victim_pos`: USER-DEFINED
- `damage`: real
- `damage_type`: text
- `damage_causer`: text
- `distance_m`: real
- `extra`: jsonb

### stdb.team_phase_combat (20 columns)

- `match_id`: text
- `team_id`: integer
- `phase_idx`: integer
- `damage_dealt`: real
- `damage_taken`: real
- `damage_events_dealt`: integer
- `damage_events_taken`: integer
- `groggy_dealt`: integer
- `groggy_taken`: integer
- `kills_dealt`: integer
- `kills_taken`: integer
- `distinct_opponents_damaged`: integer
- `distinct_opponents_damaged_by`: integer
- `combat_partner_count`: integer
- `first_damage_dealt_s`: real
- `first_damage_taken_s`: real
- `combat_initiated`: boolean
- `nearby_enemy_combat_events_200m`: integer
- `third_party_exposure_count`: integer
- `created_at`: timestamp with time zone

### stdb.team_phase_edge_cache (14 columns)

- `match_id`: text
- `phase_idx`: integer
- `edge_type`: text
- `src_team_id`: integer
- `dst_team_id`: integer
- `weight`: real
- `support_count`: integer
- `avg_distance_m`: real
- `ratio`: real
- `damage_sum`: real
- `event_count`: integer
- `kill_count`: integer
- `groggy_count`: integer
- `created_at`: timestamp with time zone

### stdb.team_phase_exposure (11 columns)

- `match_id`: text
- `team_id`: integer
- `phase_idx`: integer
- `mean_dist_to_boundary_m`: real
- `min_dist_to_boundary_m`: real
- `outside_safe_ratio`: real
- `avg_enemy_density_100m`: real
- `min_nearest_enemy_m`: real
- `path_near_enemy_ratio`: real
- `recent_damage_taken`: real
- `combat_zone_pass_count`: integer

### stdb.team_phase_f0_extra (18 columns)

- `match_id`: text
- `team_id`: integer
- `phase_idx`: integer
- `safe_center_x_start`: real
- `safe_center_y_start`: real
- `safe_center_x_end`: real
- `safe_center_y_end`: real
- `poison_center_x_start`: real
- `poison_center_y_start`: real
- `poison_center_x_end`: real
- `poison_center_y_end`: real
- `team_centroid_x_start`: real
- `team_centroid_y_start`: real
- `team_centroid_x_end`: real
- `team_centroid_y_end`: real
- `team_displacement_raw_m`: real
- `team_avg_speed_raw_mps`: real
- `created_at`: timestamp with time zone

### stdb.team_phase_features (8 columns)

- `match_id`: text
- `team_id`: integer
- `phase_idx`: integer
- `mean_dist_to_boundary_m`: real
- `enemy_density_100m`: real
- `recent_combat_exposure`: real
- `position_stability`: real
- `inside_safe_ratio`: real

### stdb.team_phase_graph (12 columns)

- `match_id`: text
- `team_id`: integer
- `phase_idx`: integer
- `avg_degree_100m`: real
- `max_degree_100m`: integer
- `avg_nearest_enemy_m`: real
- `isolation_score`: real
- `combat_edge_count`: integer
- `combat_partner_count`: integer
- `copress_200m_ratio`: real
- `had_combat`: boolean
- `survives_next_phase`: boolean

### stdb.team_phase_habitat_features (36 columns)

- `match_id`: text
- `team_id`: integer
- `phase_idx`: integer
- `hab_safe_area_start_m2`: real
- `hab_safe_area_end_m2`: real
- `hab_safe_area_delta_m2`: real
- `hab_loss_ratio`: real
- `hab_shrink_rate_m2ps`: real
- `hab_alive_teams`: integer
- `hab_safe_area_per_alive_team`: real
- `hab_crowding_pressure`: real
- `hab_center_shift_m`: real
- `hab_center_shift_per_radius`: real
- `hab_zone_gap_start_m`: real
- `hab_zone_gap_end_m`: real
- `hab_zone_gap_pressure`: real
- `hab_dist_to_safe_boundary_start_m`: real
- `hab_dist_to_safe_boundary_end_m`: real
- `hab_signed_boundary_distance_start_m`: real
- `hab_signed_boundary_distance_end_m`: real
- `hab_distance_to_target_safe_m`: real
- `hab_required_speed_to_safe_mps`: real
- `hab_observed_speed_mps`: real
- `hab_migration_deficit_mps`: real
- `hab_outside_safe_ratio`: real
- `hab_edge_pressure_ratio`: real
- `hab_rotation_delay_ratio`: real
- `hab_safe_alignment_to_target`: real
- `hab_local_enemy_count_100m`: integer
- `hab_local_enemy_count_200m`: integer
- `hab_local_enemy_count_500m`: integer
- `hab_local_enemy_density_200m`: real
- `hab_safe_entry_enemy_count_200m`: integer
- `hab_safe_entry_congestion`: integer
- `hab_refuge_scarcity_simple`: real
- `created_at`: timestamp with time zone

### stdb.team_phase_labels (18 columns)

- `match_id`: text
- `team_id`: integer
- `cut_phase`: integer
- `alive_at_phase_start`: boolean
- `alive_at_phase_end`: boolean
- `next_phase_idx`: integer
- `target_defined`: boolean
- `survives_next_phase`: boolean
- `eliminated_next_phase`: boolean
- `hazard_next`: integer
- `final_rank`: integer
- `final_rank_norm`: real
- `final_top3`: boolean
- `final_top5`: boolean
- `final_top10`: boolean
- `final_winner`: boolean
- `last_phase_alive`: integer
- `created_at`: timestamp with time zone

### stdb.team_phase_loot (17 columns)

- `match_id`: text
- `team_id`: integer
- `phase_idx`: integer
- `pickup_count`: integer
- `equip_count`: integer
- `use_count`: integer
- `unique_item_count`: integer
- `weapon_pickup_count`: integer
- `weapon_equip_count`: integer
- `ammo_pickup_count`: integer
- `armor_pickup_count`: integer
- `throwable_pickup_count`: integer
- `heal_use_count`: integer
- `boost_use_count`: integer
- `carepackage_pickup_count`: integer
- `lootbox_pickup_count`: integer
- `created_at`: timestamp with time zone

### stdb.team_phase_network_features (47 columns)

- `match_id`: text
- `team_id`: integer
- `phase_idx`: integer
- `net_combat_out_degree`: integer
- `net_combat_in_degree`: integer
- `net_combat_total_degree`: integer
- `net_combat_out_strength`: real
- `net_combat_in_strength`: real
- `net_combat_total_strength`: real
- `net_combat_balance`: real
- `net_combat_unique_targets`: integer
- `net_combat_unique_attackers`: integer
- `net_combat_reciprocal_pairs`: integer
- `net_combat_reciprocity_ratio`: real
- `net_combat_out_strength_per_min`: real
- `net_combat_in_strength_per_min`: real
- `net_prox_degree_100m`: integer
- `net_prox_degree_200m`: integer
- `net_prox_degree_500m`: integer
- `net_prox_strength_decay`: real
- `net_prox_mean_distance_m`: real
- `net_prox_min_distance_m`: real
- `net_prox_top1_weight`: real
- `net_prox_top3_strength`: real
- `net_prox_isolation_m`: real
- `net_prox_neighbor_entropy`: real
- `net_copress_degree`: integer
- `net_copress_strength`: real
- `net_copress_max_weight`: real
- `net_copress_mean_ratio`: real
- `net_copress_shared_pressure_count`: integer
- `net_copress_top1_weight`: real
- `cum_combat_out_degree`: integer
- `cum_combat_in_degree`: integer
- `cum_combat_out_strength`: real
- `cum_combat_in_strength`: real
- `cum_combat_total_strength`: real
- `cum_combat_reciprocal_pairs`: integer
- `cum_unique_targets`: integer
- `cum_unique_attackers`: integer
- `cum_copress_strength`: real
- `cum_prox_close_exposure_200m`: integer
- `net_combat_out_strength_per_alive_team`: real
- `net_combat_in_strength_per_alive_team`: real
- `net_prox_degree_200m_per_alive_team`: real
- `net_copress_degree_per_alive_team`: real
- `created_at`: timestamp with time zone

### stdb.team_phase_state (18 columns)

- `match_id`: text
- `team_id`: integer
- `phase_idx`: integer
- `team_size`: integer
- `position_rows`: integer
- `alive_players_start`: integer
- `alive_players_end`: integer
- `alive_players_min`: real
- `alive_players_avg`: real
- `mean_health_avg`: real
- `min_health_min`: real
- `health_slope`: real
- `team_spread_avg_m`: real
- `team_spread_max_m`: real
- `vehicle_ratio`: real
- `vehicle_speed_mean`: real
- `vehicle_speed_max`: real
- `created_at`: timestamp with time zone

### stdb.team_phase_trajectory (18 columns)

- `match_id`: text
- `team_id`: integer
- `phase_idx`: integer
- `waypoints`: integer
- `phase_start_s`: real
- `phase_end_s`: real
- `phase_duration_s`: real
- `any_alive`: boolean
- `trajectory`: USER-DEFINED
- `path_length_m`: real
- `displacement_m`: real
- `sinuosity`: real
- `avg_speed_mps`: real
- `rotation_timing`: real
- `safe_alignment`: real
- `zone_entry_time`: real
- `outside_safe_duration_s`: real
- `boundary_distance_slope`: real

### stdb.team_proximity (6 columns)

- `match_id`: text
- `snapshot_t`: integer
- `team_a`: integer
- `team_b`: integer
- `distance_m`: real
- `encounter_recently`: boolean

### stdb.team_snapshot_features (18 columns)

- `match_id`: text
- `team_id`: integer
- `snapshot_t`: integer
- `phase_idx`: integer
- `pos_x`: real
- `pos_y`: real
- `pos_z`: real
- `rel_zone_x`: real
- `rel_zone_y`: real
- `speed_mps`: real
- `dist_blue_boundary_m`: real
- `inside_blue`: boolean
- `dist_target_center_m`: real
- `mean_health`: real
- `min_health`: real
- `alive_n`: integer
- `vehicle_ratio`: real
- `created_at`: timestamp with time zone

### stdb.teams (5 columns)

- `match_id`: text
- `team_id`: integer
- `final_rank`: integer
- `num_players`: integer
- `won`: boolean

### stdb.v_match_phase_bounds (5 columns)

- `match_id`: text
- `phase_idx`: integer
- `phase_start_s`: real
- `phase_end_s`: real
- `phase_duration_s`: real

### stdb.v_positions_ingame (16 columns)

- `id`: bigint
- `match_id`: text
- `event_time`: timestamp with time zone
- `is_game`: numeric
- `elapsed_time`: integer
- `num_alive`: integer
- `account_id`: text
- `player_name`: text
- `team_id`: integer
- `health`: numeric
- `pos_x`: numeric
- `pos_y`: numeric
- `pos_z`: numeric
- `vehicle_type`: text
- `vehicle_id`: text
- `vehicle_speed`: numeric

### stdb.v_team_by_phase (6 columns)

- `match_id`: text
- `team_id`: integer
- `snapshot_t`: integer
- `pos`: USER-DEFINED
- `is_alive`: boolean
- `phase_idx`: integer

### stdb.v_team_phase_feature_long (86 columns)

- `match_id`: text
- `team_id`: integer
- `phase_idx`: integer
- `map_name`: text
- `game_mode`: text
- `match_team_size`: integer
- `phase_start_s`: real
- `phase_end_s`: real
- `phase_duration_s`: real
- `avg_alive_teams`: real
- `avg_alive_players`: real
- `safe_radius_start_m`: real
- `safe_radius_end_m`: real
- `safe_radius_delta_m`: real
- `safe_center_shift_m`: real
- `poison_radius_start_m`: real
- `poison_radius_end_m`: real
- `zone_gap_start_m`: real
- `zone_gap_end_m`: real
- `team_size`: integer
- `alive_players_start`: integer
- `alive_players_end`: integer
- `alive_players_min`: real
- `alive_players_avg`: real
- `mean_health_avg`: real
- `min_health_min`: real
- `health_slope`: real
- `team_spread_avg_m`: real
- `team_spread_max_m`: real
- `vehicle_ratio`: real
- `vehicle_speed_mean`: real
- `vehicle_speed_max`: real
- `path_length_m`: real
- `displacement_m`: real
- `sinuosity`: real
- `avg_speed_mps`: real
- `rotation_timing`: real
- `safe_alignment`: real
- `zone_entry_time`: real
- `outside_safe_duration_s`: real
- `boundary_distance_slope`: real
- `mean_dist_to_boundary_m`: real
- `min_dist_to_boundary_m`: real
- `outside_safe_ratio`: real
- `avg_enemy_density_100m`: real
- `min_nearest_enemy_m`: real
- `path_near_enemy_ratio`: real
- `damage_dealt`: real
- `damage_taken`: real
- `damage_events_dealt`: integer
- `damage_events_taken`: integer
- `groggy_dealt`: integer
- `groggy_taken`: integer
- `kills_dealt`: integer
- `kills_taken`: integer
- `distinct_opponents_damaged`: integer
- `distinct_opponents_damaged_by`: integer
- `combat_partner_count`: integer
- `combat_initiated`: boolean
- `nearby_enemy_combat_events_200m`: integer
- `third_party_exposure_count`: integer
- `pickup_count`: integer
- `equip_count`: integer
- `use_count`: integer
- `unique_item_count`: integer
- `weapon_pickup_count`: integer
- `weapon_equip_count`: integer
- `ammo_pickup_count`: integer
- `armor_pickup_count`: integer
- `throwable_pickup_count`: integer
- `heal_use_count`: integer
- `boost_use_count`: integer
- `carepackage_pickup_count`: integer
- `lootbox_pickup_count`: integer
- `target_defined`: boolean
- `hazard_next`: integer
- `survives_next_phase`: boolean
- `final_rank`: integer
- `final_rank_norm`: real
- `final_top3`: boolean
- `final_top5`: boolean
- `final_top10`: boolean
- `final_winner`: boolean
- `last_phase_alive`: integer
- `recent_damage_taken`: real
- `combat_zone_pass_count`: integer

### stdb.v_team_phase_feature_v2 (187 columns)

- `match_id`: text
- `team_id`: integer
- `phase_idx`: integer
- `map_name`: text
- `game_mode`: text
- `match_team_size`: integer
- `phase_start_s`: real
- `phase_end_s`: real
- `phase_duration_s`: real
- `avg_alive_teams`: real
- `avg_alive_players`: real
- `safe_radius_start_m`: real
- `safe_radius_end_m`: real
- `safe_radius_delta_m`: real
- `safe_center_shift_m`: real
- `poison_radius_start_m`: real
- `poison_radius_end_m`: real
- `zone_gap_start_m`: real
- `zone_gap_end_m`: real
- `team_size`: integer
- `alive_players_start`: integer
- `alive_players_end`: integer
- `alive_players_min`: real
- `alive_players_avg`: real
- `mean_health_avg`: real
- `min_health_min`: real
- `health_slope`: real
- `team_spread_avg_m`: real
- `team_spread_max_m`: real
- `vehicle_ratio`: real
- `vehicle_speed_mean`: real
- `vehicle_speed_max`: real
- `path_length_m`: real
- `displacement_m`: real
- `sinuosity`: real
- `avg_speed_mps`: real
- `rotation_timing`: real
- `safe_alignment`: real
- `zone_entry_time`: real
- `outside_safe_duration_s`: real
- `boundary_distance_slope`: real
- `mean_dist_to_boundary_m`: real
- `min_dist_to_boundary_m`: real
- `outside_safe_ratio`: real
- `avg_enemy_density_100m`: real
- `min_nearest_enemy_m`: real
- `path_near_enemy_ratio`: real
- `damage_dealt`: real
- `damage_taken`: real
- `damage_events_dealt`: integer
- `damage_events_taken`: integer
- `groggy_dealt`: integer
- `groggy_taken`: integer
- `kills_dealt`: integer
- `kills_taken`: integer
- `distinct_opponents_damaged`: integer
- `distinct_opponents_damaged_by`: integer
- `combat_partner_count`: integer
- `combat_initiated`: boolean
- `nearby_enemy_combat_events_200m`: integer
- `third_party_exposure_count`: integer
- `pickup_count`: integer
- `equip_count`: integer
- `use_count`: integer
- `unique_item_count`: integer
- `weapon_pickup_count`: integer
- `weapon_equip_count`: integer
- `ammo_pickup_count`: integer
- `armor_pickup_count`: integer
- `throwable_pickup_count`: integer
- `heal_use_count`: integer
- `boost_use_count`: integer
- `carepackage_pickup_count`: integer
- `lootbox_pickup_count`: integer
- `target_defined`: boolean
- `hazard_next`: integer
- `survives_next_phase`: boolean
- `final_rank`: integer
- `final_rank_norm`: real
- `final_top3`: boolean
- `final_top5`: boolean
- `final_top10`: boolean
- `final_winner`: boolean
- `last_phase_alive`: integer
- `recent_damage_taken`: real
- `combat_zone_pass_count`: integer
- `weather_id`: text
- `camera_view_type`: text
- `start_alive_teams`: integer
- `end_alive_teams`: integer
- `start_alive_players`: integer
- `end_alive_players`: integer
- `poison_radius_delta_m`: real
- `red_zone_active_ratio`: real
- `black_zone_active_ratio`: real
- `position_rows`: integer
- `first_damage_dealt_s`: real
- `first_damage_taken_s`: real
- `safe_center_x_start`: real
- `safe_center_y_start`: real
- `safe_center_x_end`: real
- `safe_center_y_end`: real
- `poison_center_x_start`: real
- `poison_center_y_start`: real
- `poison_center_x_end`: real
- `poison_center_y_end`: real
- `team_centroid_x_start`: real
- `team_centroid_y_start`: real
- `team_centroid_x_end`: real
- `team_centroid_y_end`: real
- `team_displacement_raw_m`: real
- `team_avg_speed_raw_mps`: real
- `net_combat_out_degree`: integer
- `net_combat_in_degree`: integer
- `net_combat_total_degree`: integer
- `net_combat_out_strength`: real
- `net_combat_in_strength`: real
- `net_combat_total_strength`: real
- `net_combat_balance`: real
- `net_combat_unique_targets`: integer
- `net_combat_unique_attackers`: integer
- `net_combat_reciprocal_pairs`: integer
- `net_combat_reciprocity_ratio`: real
- `net_combat_out_strength_per_min`: real
- `net_combat_in_strength_per_min`: real
- `net_prox_degree_100m`: integer
- `net_prox_degree_200m`: integer
- `net_prox_degree_500m`: integer
- `net_prox_strength_decay`: real
- `net_prox_mean_distance_m`: real
- `net_prox_min_distance_m`: real
- `net_prox_top1_weight`: real
- `net_prox_top3_strength`: real
- `net_prox_isolation_m`: real
- `net_prox_neighbor_entropy`: real
- `net_copress_degree`: integer
- `net_copress_strength`: real
- `net_copress_max_weight`: real
- `net_copress_mean_ratio`: real
- `net_copress_shared_pressure_count`: integer
- `net_copress_top1_weight`: real
- `cum_combat_out_degree`: integer
- `cum_combat_in_degree`: integer
- `cum_combat_out_strength`: real
- `cum_combat_in_strength`: real
- `cum_combat_total_strength`: real
- `cum_combat_reciprocal_pairs`: integer
- `cum_unique_targets`: integer
- `cum_unique_attackers`: integer
- `cum_copress_strength`: real
- `cum_prox_close_exposure_200m`: integer
- `net_combat_out_strength_per_alive_team`: real
- `net_combat_in_strength_per_alive_team`: real
- `net_prox_degree_200m_per_alive_team`: real
- `net_copress_degree_per_alive_team`: real
- `hab_safe_area_start_m2`: real
- `hab_safe_area_end_m2`: real
- `hab_safe_area_delta_m2`: real
- `hab_loss_ratio`: real
- `hab_shrink_rate_m2ps`: real
- `hab_alive_teams`: integer
- `hab_safe_area_per_alive_team`: real
- `hab_crowding_pressure`: real
- `hab_center_shift_m`: real
- `hab_center_shift_per_radius`: real
- `hab_zone_gap_start_m`: real
- `hab_zone_gap_end_m`: real
- `hab_zone_gap_pressure`: real
- `hab_dist_to_safe_boundary_start_m`: real
- `hab_dist_to_safe_boundary_end_m`: real
- `hab_signed_boundary_distance_start_m`: real
- `hab_signed_boundary_distance_end_m`: real
- `hab_distance_to_target_safe_m`: real
- `hab_required_speed_to_safe_mps`: real
- `hab_observed_speed_mps`: real
- `hab_migration_deficit_mps`: real
- `hab_outside_safe_ratio`: real
- `hab_edge_pressure_ratio`: real
- `hab_rotation_delay_ratio`: real
- `hab_safe_alignment_to_target`: real
- `hab_local_enemy_count_100m`: integer
- `hab_local_enemy_count_200m`: integer
- `hab_local_enemy_count_500m`: integer
- `hab_local_enemy_density_200m`: real
- `hab_safe_entry_enemy_count_200m`: integer
- `hab_safe_entry_congestion`: integer
- `hab_refuge_scarcity_simple`: real

### stdb.v_team_phase_official_raw_features (85 columns)

- `match_id`: text
- `team_id`: integer
- `phase_idx`: integer
- `map_name`: text
- `game_mode`: text
- `match_team_size`: integer
- `weather_id`: text
- `camera_view_type`: text
- `phase_start_s`: real
- `phase_end_s`: real
- `phase_duration_s`: real
- `avg_alive_teams`: real
- `start_alive_teams`: integer
- `end_alive_teams`: integer
- `avg_alive_players`: real
- `start_alive_players`: integer
- `end_alive_players`: integer
- `safe_radius_start_m`: real
- `safe_radius_end_m`: real
- `safe_radius_delta_m`: real
- `poison_radius_start_m`: real
- `poison_radius_end_m`: real
- `poison_radius_delta_m`: real
- `safe_center_x_start`: real
- `safe_center_y_start`: real
- `safe_center_x_end`: real
- `safe_center_y_end`: real
- `poison_center_x_start`: real
- `poison_center_y_start`: real
- `poison_center_x_end`: real
- `poison_center_y_end`: real
- `red_zone_active_ratio`: real
- `black_zone_active_ratio`: real
- `team_size`: integer
- `position_rows`: integer
- `alive_players_start`: integer
- `alive_players_end`: integer
- `alive_players_min`: real
- `alive_players_avg`: real
- `mean_health_avg`: real
- `min_health_min`: real
- `health_slope`: real
- `team_spread_avg_m`: real
- `team_spread_max_m`: real
- `vehicle_ratio`: real
- `vehicle_speed_mean`: real
- `vehicle_speed_max`: real
- `team_centroid_x_start`: real
- `team_centroid_y_start`: real
- `team_centroid_x_end`: real
- `team_centroid_y_end`: real
- `team_displacement_raw_m`: real
- `team_avg_speed_raw_mps`: real
- `damage_dealt`: real
- `damage_taken`: real
- `damage_events_dealt`: integer
- `damage_events_taken`: integer
- `groggy_dealt`: integer
- `groggy_taken`: integer
- `kills_dealt`: integer
- `kills_taken`: integer
- `distinct_opponents_damaged`: integer
- `distinct_opponents_damaged_by`: integer
- `combat_partner_count`: integer
- `first_damage_dealt_s`: real
- `first_damage_taken_s`: real
- `combat_initiated`: boolean
- `pickup_count`: integer
- `equip_count`: integer
- `use_count`: integer
- `unique_item_count`: integer
- `weapon_pickup_count`: integer
- `weapon_equip_count`: integer
- `ammo_pickup_count`: integer
- `armor_pickup_count`: integer
- `throwable_pickup_count`: integer
- `heal_use_count`: integer
- `boost_use_count`: integer
- `carepackage_pickup_count`: integer
- `lootbox_pickup_count`: integer
- `target_defined`: boolean
- `hazard_next`: integer
- `final_rank_norm`: real
- `final_top5`: boolean
- `final_top10`: boolean

### stdb.v_team_phase_trajectory (9 columns)

- `match_id`: text
- `team_id`: integer
- `phase_idx`: integer
- `waypoints`: bigint
- `phase_start_s`: integer
- `phase_end_s`: integer
- `any_alive`: boolean
- `trajectory`: USER-DEFINED
- `trajectory_2d`: USER-DEFINED

### stdb.v_team_phase_wide (107 columns)

- `match_id`: text
- `team_id`: integer
- `final_rank`: integer
- `tier`: text
- `last_phase_alive`: integer
- `p1_path_length_m`: real
- `p2_path_length_m`: real
- `p3_path_length_m`: real
- `p4_path_length_m`: real
- `p5_path_length_m`: real
- `p6_path_length_m`: real
- `p1_displacement_m`: real
- `p2_displacement_m`: real
- `p3_displacement_m`: real
- `p4_displacement_m`: real
- `p5_displacement_m`: real
- `p6_displacement_m`: real
- `p1_sinuosity`: real
- `p2_sinuosity`: real
- `p3_sinuosity`: real
- `p4_sinuosity`: real
- `p5_sinuosity`: real
- `p6_sinuosity`: real
- `p1_avg_speed_mps`: real
- `p2_avg_speed_mps`: real
- `p3_avg_speed_mps`: real
- `p4_avg_speed_mps`: real
- `p5_avg_speed_mps`: real
- `p6_avg_speed_mps`: real
- `p1_rotation_timing`: real
- `p2_rotation_timing`: real
- `p3_rotation_timing`: real
- `p4_rotation_timing`: real
- `p5_rotation_timing`: real
- `p6_rotation_timing`: real
- `p1_safe_alignment`: real
- `p2_safe_alignment`: real
- `p3_safe_alignment`: real
- `p4_safe_alignment`: real
- `p5_safe_alignment`: real
- `p6_safe_alignment`: real
- `p1_zone_entry_time`: real
- `p2_zone_entry_time`: real
- `p3_zone_entry_time`: real
- `p4_zone_entry_time`: real
- `p5_zone_entry_time`: real
- `p6_zone_entry_time`: real
- `p1_outside_safe_duratio`: real
- `p2_outside_safe_duratio`: real
- `p3_outside_safe_duratio`: real
- `p4_outside_safe_duratio`: real
- `p5_outside_safe_duratio`: real
- `p6_outside_safe_duratio`: real
- `p1_boundary_distance_sl`: real
- `p2_boundary_distance_sl`: real
- `p3_boundary_distance_sl`: real
- `p4_boundary_distance_sl`: real
- `p5_boundary_distance_sl`: real
- `p6_boundary_distance_sl`: real
- `p1_mean_dist_to_boundar`: real
- `p2_mean_dist_to_boundar`: real
- `p3_mean_dist_to_boundar`: real
- `p4_mean_dist_to_boundar`: real
- `p5_mean_dist_to_boundar`: real
- `p6_mean_dist_to_boundar`: real
- `p1_min_dist_to_boundary`: real
- `p2_min_dist_to_boundary`: real
- `p3_min_dist_to_boundary`: real
- `p4_min_dist_to_boundary`: real
- `p5_min_dist_to_boundary`: real
- `p6_min_dist_to_boundary`: real
- `p1_outside_safe_ratio`: real
- `p2_outside_safe_ratio`: real
- `p3_outside_safe_ratio`: real
- `p4_outside_safe_ratio`: real
- `p5_outside_safe_ratio`: real
- `p6_outside_safe_ratio`: real
- `p1_avg_enemy_density_10`: real
- `p2_avg_enemy_density_10`: real
- `p3_avg_enemy_density_10`: real
- `p4_avg_enemy_density_10`: real
- `p5_avg_enemy_density_10`: real
- `p6_avg_enemy_density_10`: real
- `p1_min_nearest_enemy_m`: real
- `p2_min_nearest_enemy_m`: real
- `p3_min_nearest_enemy_m`: real
- `p4_min_nearest_enemy_m`: real
- `p5_min_nearest_enemy_m`: real
- `p6_min_nearest_enemy_m`: real
- `p1_path_near_enemy_rati`: real
- `p2_path_near_enemy_rati`: real
- `p3_path_near_enemy_rati`: real
- `p4_path_near_enemy_rati`: real
- `p5_path_near_enemy_rati`: real
- `p6_path_near_enemy_rati`: real
- `p1_recent_damage_taken`: real
- `p2_recent_damage_taken`: real
- `p3_recent_damage_taken`: real
- `p4_recent_damage_taken`: real
- `p5_recent_damage_taken`: real
- `p6_recent_damage_taken`: real
- `p1_combat_zone_pass_cou`: integer
- `p2_combat_zone_pass_cou`: integer
- `p3_combat_zone_pass_cou`: integer
- `p4_combat_zone_pass_cou`: integer
- `p5_combat_zone_pass_cou`: integer
- `p6_combat_zone_pass_cou`: integer

### stdb.v_team_truncated_targets (11 columns)

- `match_id`: text
- `team_id`: integer
- `final_rank`: integer
- `last_phase_alive`: integer
- `cut_phase`: integer
- `survives_next_phase`: boolean
- `survives_two_more`: boolean
- `final_top5`: boolean
- `final_top10`: boolean
- `final_winner`: boolean
- `final_rank_target`: integer

## 4. Current short-paper snapshot model-facing features

These are not the full DB. These are the 13 node features used in the current 10-second snapshot experiment.

- Movement: `pos_x`, `pos_y`, `pos_z`, `speed_mps`
- Zone context: `rel_zone_x`, `rel_zone_y`, `dist_blue_boundary_m`, `inside_blue`, `dist_target_center_m`
- Survival state: `mean_health`, `min_health`, `alive_n`
- Mobility resource: `vehicle_ratio`
- Primitive relation tensor `A_s`: `proximity`, `combat`
- Relative edge follow-up `E_s`: contact gate, proximity/combat weights, and relative state differences derived from `X_s` and `A_s`
