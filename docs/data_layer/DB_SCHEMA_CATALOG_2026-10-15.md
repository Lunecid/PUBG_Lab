# PUBG 현재 DB 전체 테이블과 컬럼 사전

이 사전은 로컬 저장 구조다. 공식 API 원형과의 차이는 [공식 API 스키마 대조표](PUBG_OFFICIAL_API_SCHEMA_AUDIT_2026-10-15.md)를 기준으로 확인한다.

2026-10-15 PostgreSQL metadata 실측. 범위는 pubg/stdb의 부모 테이블과 view이며 child partition은 제외했다. 새 연구의 제안 테이블은 포함하지 않는다.

- Relation: 49; column definitions: 1130.
- Source: [schema_snapshot.json](../../reports/movement_schema_2026-10-15/schema_snapshot.json).
- 관계별 PK/FK/UNIQUE와 NULL 제약은 아래에 기재한다. 이름이 같은 컬럼이라도 의미·시계·단위가 같은지는 별도 검증한다.
- 활용 가이드: [이동 프로젝트 스키마](FDG2027_MOVEMENT_PROJECT_SCHEMA_2026-10-15.md).

| Relation | 종류 | 컬럼 수 |
|---|---|---:|
| `pubg.api_rate_log` | table | 7 |
| `pubg.collection_log` | table | 10 |
| `pubg.matches` | table | 12 |
| `pubg.participants` | table | 27 |
| `pubg.rosters` | table | 5 |
| `pubg.telem_damage` | table | 20 |
| `pubg.telem_game_states` | table | 25 |
| `pubg.telem_groggy` | table | 19 |
| `pubg.telem_item_equip` | table | 15 |
| `pubg.telem_item_pickup` | table | 19 |
| `pubg.telem_item_use` | table | 14 |
| `pubg.telem_kills` | table | 26 |
| `pubg.telem_match_end` | table | 4 |
| `pubg.telem_match_start` | table | 10 |
| `pubg.telem_parachute_landing` | table | 10 |
| `pubg.telem_phase_changes` | table | 6 |
| `pubg.telem_positions` | partitioned parent | 16 |
| `pubg.v_collection_status` | view | 6 |
| `pubg.v_match_summary` | view | 11 |
| `stdb.dynamic_regions` | table | 7 |
| `stdb.match_phase_global` | table | 28 |
| `stdb.matches` | table | 9 |
| `stdb.moving_objects` | table | 9 |
| `stdb.spatial_events` | table | 15 |
| `stdb.team_phase_combat` | table | 20 |
| `stdb.team_phase_edge_cache` | table | 14 |
| `stdb.team_phase_exposure` | table | 11 |
| `stdb.team_phase_f0_extra` | table | 18 |
| `stdb.team_phase_features` | table | 8 |
| `stdb.team_phase_graph` | table | 12 |
| `stdb.team_phase_habitat_features` | table | 36 |
| `stdb.team_phase_labels` | table | 18 |
| `stdb.team_phase_loot` | table | 17 |
| `stdb.team_phase_network_features` | table | 47 |
| `stdb.team_phase_state` | table | 18 |
| `stdb.team_phase_trajectory` | table | 18 |
| `stdb.team_phase_z_affordance_features` | table | 13 |
| `stdb.team_proximity` | table | 6 |
| `stdb.team_snapshot_features` | table | 18 |
| `stdb.teams` | table | 5 |
| `stdb.v_match_phase_bounds` | view | 5 |
| `stdb.v_positions_ingame` | view | 16 |
| `stdb.v_team_by_phase` | view | 6 |
| `stdb.v_team_phase_feature_long` | view | 86 |
| `stdb.v_team_phase_feature_v2` | view | 196 |
| `stdb.v_team_phase_official_raw_features` | view | 85 |
| `stdb.v_team_phase_trajectory` | view | 9 |
| `stdb.v_team_phase_wide` | view | 107 |
| `stdb.v_team_truncated_targets` | view | 11 |

## pubg.api_rate_log

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `id` | bigint | NO |
| 2 | `called_at` | timestamp with time zone | NO |
| 3 | `endpoint` | text | NO |
| 4 | `status_code` | integer | YES |
| 5 | `rate_remaining` | integer | YES |
| 6 | `rate_reset` | timestamp with time zone | YES |
| 7 | `response_ms` | integer | YES |

키 및 무결성 제약:

- `api_rate_log_pkey`: `PRIMARY KEY (id)`

## pubg.collection_log

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | NO |
| 2 | `shard_id` | text | NO |
| 3 | `discovered_at` | timestamp with time zone | NO |
| 4 | `match_fetched` | boolean | NO |
| 5 | `match_fetched_at` | timestamp with time zone | YES |
| 6 | `telemetry_url` | text | YES |
| 7 | `telemetry_fetched` | boolean | NO |
| 8 | `telemetry_fetched_at` | timestamp with time zone | YES |
| 9 | `error_message` | text | YES |
| 10 | `retry_count` | integer | NO |

키 및 무결성 제약:

- `collection_log_pkey`: `PRIMARY KEY (match_id)`

## pubg.matches

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | NO |
| 2 | `created_at` | timestamp with time zone | NO |
| 3 | `duration` | integer | NO |
| 4 | `game_mode` | text | NO |
| 5 | `map_name` | text | NO |
| 6 | `is_custom` | boolean | NO |
| 7 | `match_type` | text | YES |
| 8 | `season_state` | text | YES |
| 9 | `shard_id` | text | NO |
| 10 | `title_id` | text | YES |
| 11 | `telemetry_url` | text | YES |
| 12 | `collected_at` | timestamp with time zone | NO |

키 및 무결성 제약:

- `matches_pkey`: `PRIMARY KEY (match_id)`

## pubg.participants

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `participant_id` | text | NO |
| 2 | `match_id` | text | NO |
| 3 | `roster_id` | text | YES |
| 4 | `player_id` | text | YES |
| 5 | `player_name` | text | YES |
| 6 | `kills` | integer | YES |
| 7 | `assists` | integer | YES |
| 8 | `dbnos` | integer | YES |
| 9 | `damage_dealt` | numeric | YES |
| 10 | `headshot_kills` | integer | YES |
| 11 | `longest_kill` | numeric | YES |
| 12 | `most_damage` | numeric | YES |
| 13 | `kill_place` | integer | YES |
| 14 | `kill_streaks` | integer | YES |
| 15 | `road_kills` | integer | YES |
| 16 | `team_kills` | integer | YES |
| 17 | `vehicle_destroys` | integer | YES |
| 18 | `death_type` | text | YES |
| 19 | `time_survived` | numeric | YES |
| 20 | `win_place` | integer | YES |
| 21 | `walk_distance` | numeric | YES |
| 22 | `ride_distance` | numeric | YES |
| 23 | `swim_distance` | numeric | YES |
| 24 | `boosts` | integer | YES |
| 25 | `heals` | integer | YES |
| 26 | `weapons_acquired` | integer | YES |
| 27 | `revives` | integer | YES |

키 및 무결성 제약:

- `participants_match_id_fkey`: `FOREIGN KEY (match_id) REFERENCES pubg.matches(match_id)`
- `participants_pkey`: `PRIMARY KEY (participant_id)`
- `participants_roster_id_fkey`: `FOREIGN KEY (roster_id) REFERENCES pubg.rosters(roster_id)`

## pubg.rosters

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `roster_id` | text | NO |
| 2 | `match_id` | text | NO |
| 3 | `rank` | integer | YES |
| 4 | `team_id` | integer | YES |
| 5 | `won` | boolean | NO |

키 및 무결성 제약:

- `rosters_match_id_fkey`: `FOREIGN KEY (match_id) REFERENCES pubg.matches(match_id)`
- `rosters_pkey`: `PRIMARY KEY (roster_id)`

## pubg.telem_damage

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `id` | bigint | NO |
| 2 | `match_id` | text | NO |
| 3 | `event_time` | timestamp with time zone | NO |
| 4 | `is_game` | numeric | YES |
| 5 | `attacker_id` | text | YES |
| 6 | `attacker_name` | text | YES |
| 7 | `attacker_x` | numeric | YES |
| 8 | `attacker_y` | numeric | YES |
| 9 | `attacker_z` | numeric | YES |
| 10 | `victim_id` | text | YES |
| 11 | `victim_name` | text | YES |
| 12 | `victim_x` | numeric | YES |
| 13 | `victim_y` | numeric | YES |
| 14 | `victim_z` | numeric | YES |
| 15 | `damage` | numeric | YES |
| 16 | `damage_reason` | text | YES |
| 17 | `damage_type` | text | YES |
| 18 | `damage_causer` | text | YES |
| 19 | `distance` | numeric | YES |
| 20 | `is_through_wall` | boolean | YES |

키 및 무결성 제약:

- `telem_damage_pkey`: `PRIMARY KEY (id)`

## pubg.telem_game_states

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `id` | bigint | NO |
| 2 | `match_id` | text | NO |
| 3 | `event_time` | timestamp with time zone | NO |
| 4 | `is_game` | numeric | YES |
| 5 | `elapsed_time` | integer | YES |
| 6 | `num_alive_teams` | integer | YES |
| 7 | `num_alive_players` | integer | YES |
| 8 | `num_join_players` | integer | YES |
| 9 | `num_start_players` | integer | YES |
| 10 | `safe_zone_x` | numeric | YES |
| 11 | `safe_zone_y` | numeric | YES |
| 12 | `safe_zone_z` | numeric | YES |
| 13 | `safe_zone_radius` | numeric | YES |
| 14 | `poison_zone_x` | numeric | YES |
| 15 | `poison_zone_y` | numeric | YES |
| 16 | `poison_zone_z` | numeric | YES |
| 17 | `poison_zone_radius` | numeric | YES |
| 18 | `red_zone_x` | numeric | YES |
| 19 | `red_zone_y` | numeric | YES |
| 20 | `red_zone_z` | numeric | YES |
| 21 | `red_zone_radius` | numeric | YES |
| 22 | `black_zone_x` | numeric | YES |
| 23 | `black_zone_y` | numeric | YES |
| 24 | `black_zone_z` | numeric | YES |
| 25 | `black_zone_radius` | numeric | YES |

키 및 무결성 제약:

- `telem_game_states_pkey`: `PRIMARY KEY (id)`

## pubg.telem_groggy

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `id` | bigint | NO |
| 2 | `match_id` | text | NO |
| 3 | `event_time` | timestamp with time zone | NO |
| 4 | `is_game` | numeric | YES |
| 5 | `attacker_id` | text | YES |
| 6 | `attacker_name` | text | YES |
| 7 | `attacker_x` | numeric | YES |
| 8 | `attacker_y` | numeric | YES |
| 9 | `attacker_z` | numeric | YES |
| 10 | `victim_id` | text | YES |
| 11 | `victim_name` | text | YES |
| 12 | `victim_x` | numeric | YES |
| 13 | `victim_y` | numeric | YES |
| 14 | `victim_z` | numeric | YES |
| 15 | `damage_reason` | text | YES |
| 16 | `damage_type` | text | YES |
| 17 | `damage_causer` | text | YES |
| 18 | `distance` | numeric | YES |
| 19 | `is_attacker_in_vehicle` | boolean | YES |

키 및 무결성 제약:

- `telem_groggy_pkey`: `PRIMARY KEY (id)`

## pubg.telem_item_equip

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `id` | bigint | NO |
| 2 | `match_id` | text | NO |
| 3 | `event_time` | timestamp with time zone | NO |
| 4 | `is_game` | numeric | NO |
| 5 | `event_type` | text | NO |
| 6 | `account_id` | text | YES |
| 7 | `player_name` | text | YES |
| 8 | `team_id` | integer | YES |
| 9 | `pos_x` | numeric | YES |
| 10 | `pos_y` | numeric | YES |
| 11 | `pos_z` | numeric | YES |
| 12 | `item_id` | text | YES |
| 13 | `category` | text | YES |
| 14 | `sub_category` | text | YES |
| 15 | `attached_items` | ARRAY | YES |

키 및 무결성 제약:

- `telem_item_equip_pkey`: `PRIMARY KEY (id)`

## pubg.telem_item_pickup

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `id` | bigint | NO |
| 2 | `match_id` | text | NO |
| 3 | `event_time` | timestamp with time zone | NO |
| 4 | `is_game` | numeric | NO |
| 5 | `event_type` | text | NO |
| 6 | `account_id` | text | YES |
| 7 | `player_name` | text | YES |
| 8 | `team_id` | integer | YES |
| 9 | `pos_x` | numeric | YES |
| 10 | `pos_y` | numeric | YES |
| 11 | `pos_z` | numeric | YES |
| 12 | `item_id` | text | YES |
| 13 | `category` | text | YES |
| 14 | `sub_category` | text | YES |
| 15 | `stack_count` | integer | YES |
| 16 | `attached_items` | ARRAY | YES |
| 17 | `carepackage_id` | numeric | YES |
| 18 | `lootbox_owner_team_id` | integer | YES |
| 19 | `lootbox_creator_id` | text | YES |

키 및 무결성 제약:

- `telem_item_pickup_pkey`: `PRIMARY KEY (id)`

## pubg.telem_item_use

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `id` | bigint | NO |
| 2 | `match_id` | text | NO |
| 3 | `event_time` | timestamp with time zone | NO |
| 4 | `is_game` | numeric | NO |
| 5 | `account_id` | text | YES |
| 6 | `player_name` | text | YES |
| 7 | `team_id` | integer | YES |
| 8 | `pos_x` | numeric | YES |
| 9 | `pos_y` | numeric | YES |
| 10 | `pos_z` | numeric | YES |
| 11 | `item_id` | text | YES |
| 12 | `category` | text | YES |
| 13 | `sub_category` | text | YES |
| 14 | `stack_count` | integer | YES |

키 및 무결성 제약:

- `telem_item_use_pkey`: `PRIMARY KEY (id)`

## pubg.telem_kills

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `id` | bigint | NO |
| 2 | `match_id` | text | NO |
| 3 | `event_time` | timestamp with time zone | NO |
| 4 | `is_game` | numeric | YES |
| 5 | `killer_id` | text | YES |
| 6 | `killer_name` | text | YES |
| 7 | `killer_x` | numeric | YES |
| 8 | `killer_y` | numeric | YES |
| 9 | `killer_z` | numeric | YES |
| 10 | `victim_id` | text | YES |
| 11 | `victim_name` | text | YES |
| 12 | `victim_x` | numeric | YES |
| 13 | `victim_y` | numeric | YES |
| 14 | `victim_z` | numeric | YES |
| 15 | `dbno_maker_id` | text | YES |
| 16 | `dbno_maker_name` | text | YES |
| 17 | `finisher_id` | text | YES |
| 18 | `finisher_name` | text | YES |
| 19 | `damage_reason` | text | YES |
| 20 | `damage_type` | text | YES |
| 21 | `damage_causer` | text | YES |
| 22 | `distance` | numeric | YES |
| 23 | `is_suicide` | boolean | YES |
| 24 | `assists_cnt` | integer | YES |
| 25 | `victim_rank` | integer | YES |
| 26 | `victim_team_id` | integer | YES |

키 및 무결성 제약:

- `telem_kills_pkey`: `PRIMARY KEY (id)`

## pubg.telem_match_end

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `id` | bigint | NO |
| 2 | `match_id` | text | NO |
| 3 | `event_time` | timestamp with time zone | NO |
| 4 | `game_results` | jsonb | YES |

키 및 무결성 제약:

- `telem_match_end_match_id_key`: `UNIQUE (match_id)`
- `telem_match_end_pkey`: `PRIMARY KEY (id)`

## pubg.telem_match_start

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `id` | bigint | NO |
| 2 | `match_id` | text | NO |
| 3 | `event_time` | timestamp with time zone | NO |
| 4 | `map_name` | text | YES |
| 5 | `weather_id` | text | YES |
| 6 | `camera_view_type` | text | YES |
| 7 | `team_size` | integer | YES |
| 8 | `is_custom` | boolean | YES |
| 9 | `is_event_mode` | boolean | YES |
| 10 | `blue_zone_options` | jsonb | YES |

키 및 무결성 제약:

- `telem_match_start_match_id_key`: `UNIQUE (match_id)`
- `telem_match_start_pkey`: `PRIMARY KEY (id)`

## pubg.telem_parachute_landing

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `id` | bigint | NO |
| 2 | `match_id` | text | NO |
| 3 | `event_time` | timestamp with time zone | NO |
| 4 | `account_id` | text | YES |
| 5 | `player_name` | text | YES |
| 6 | `team_id` | integer | YES |
| 7 | `pos_x` | numeric | YES |
| 8 | `pos_y` | numeric | YES |
| 9 | `pos_z` | numeric | YES |
| 10 | `distance` | numeric | YES |

키 및 무결성 제약:

- `telem_parachute_landing_pkey`: `PRIMARY KEY (id)`

## pubg.telem_phase_changes

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `id` | bigint | NO |
| 2 | `match_id` | text | NO |
| 3 | `event_time` | timestamp with time zone | NO |
| 4 | `is_game` | numeric | YES |
| 5 | `phase` | numeric | YES |
| 6 | `elapsed_time` | integer | YES |

키 및 무결성 제약:

- `telem_phase_changes_pkey`: `PRIMARY KEY (id)`

## pubg.telem_positions

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `id` | bigint | NO |
| 2 | `match_id` | text | NO |
| 3 | `event_time` | timestamp with time zone | NO |
| 4 | `is_game` | numeric | YES |
| 5 | `elapsed_time` | integer | YES |
| 6 | `num_alive` | integer | YES |
| 7 | `account_id` | text | YES |
| 8 | `player_name` | text | YES |
| 9 | `team_id` | integer | YES |
| 10 | `health` | numeric | YES |
| 11 | `pos_x` | numeric | YES |
| 12 | `pos_y` | numeric | YES |
| 13 | `pos_z` | numeric | YES |
| 14 | `vehicle_type` | text | YES |
| 15 | `vehicle_id` | text | YES |
| 16 | `vehicle_speed` | numeric | YES |

키 및 무결성 제약:

- `telem_positions_pkey`: `PRIMARY KEY (id, event_time)`

## pubg.v_collection_status

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `total_discovered` | bigint | YES |
| 2 | `matches_fetched` | bigint | YES |
| 3 | `telemetry_fetched` | bigint | YES |
| 4 | `errors` | bigint | YES |
| 5 | `oldest_discovery` | timestamp with time zone | YES |
| 6 | `latest_discovery` | timestamp with time zone | YES |

PK/FK/UNIQUE/CHECK 제약 없음 또는 view. nullable 정보는 위 표 참조.

## pubg.v_match_summary

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | YES |
| 2 | `created_at` | timestamp with time zone | YES |
| 3 | `duration` | integer | YES |
| 4 | `game_mode` | text | YES |
| 5 | `map_name` | text | YES |
| 6 | `team_count` | bigint | YES |
| 7 | `player_count` | bigint | YES |
| 8 | `max_kills` | integer | YES |
| 9 | `avg_damage` | numeric | YES |
| 10 | `match_fetched` | boolean | YES |
| 11 | `telemetry_fetched` | boolean | YES |

PK/FK/UNIQUE/CHECK 제약 없음 또는 view. nullable 정보는 위 표 참조.

## stdb.dynamic_regions

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | NO |
| 2 | `region_kind` | text | NO |
| 3 | `snapshot_t` | integer | NO |
| 4 | `center_pos` | geometry | NO |
| 5 | `radius_m` | real | NO |
| 6 | `region` | geometry | YES |
| 7 | `phase_idx` | integer | YES |

키 및 무결성 제약:

- `dynamic_regions_pkey`: `PRIMARY KEY (match_id, region_kind, snapshot_t)`
- `dynamic_regions_region_kind_check`: `CHECK ((region_kind = ANY (ARRAY['safe'::text, 'poison'::text])))`

## stdb.match_phase_global

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | NO |
| 2 | `phase_idx` | integer | NO |
| 3 | `phase_start_s` | real | NO |
| 4 | `phase_end_s` | real | NO |
| 5 | `phase_duration_s` | real | YES |
| 6 | `avg_alive_teams` | real | YES |
| 7 | `start_alive_teams` | integer | YES |
| 8 | `end_alive_teams` | integer | YES |
| 9 | `avg_alive_players` | real | YES |
| 10 | `start_alive_players` | integer | YES |
| 11 | `end_alive_players` | integer | YES |
| 12 | `safe_radius_start_m` | real | YES |
| 13 | `safe_radius_end_m` | real | YES |
| 14 | `safe_radius_delta_m` | real | YES |
| 15 | `safe_center_shift_m` | real | YES |
| 16 | `poison_radius_start_m` | real | YES |
| 17 | `poison_radius_end_m` | real | YES |
| 18 | `poison_radius_delta_m` | real | YES |
| 19 | `zone_gap_start_m` | real | YES |
| 20 | `zone_gap_end_m` | real | YES |
| 21 | `safe_area_ratio_start` | real | YES |
| 22 | `safe_area_ratio_end` | real | YES |
| 23 | `red_zone_active_ratio` | real | YES |
| 24 | `black_zone_active_ratio` | real | YES |
| 25 | `weather_id` | text | YES |
| 26 | `camera_view_type` | text | YES |
| 27 | `blue_zone_options` | jsonb | YES |
| 28 | `created_at` | timestamp with time zone | NO |

키 및 무결성 제약:

- `match_phase_global_match_id_fkey`: `FOREIGN KEY (match_id) REFERENCES stdb.matches(match_id) ON DELETE CASCADE`
- `match_phase_global_pkey`: `PRIMARY KEY (match_id, phase_idx)`

## stdb.matches

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | NO |
| 2 | `started_at` | timestamp with time zone | NO |
| 3 | `duration_s` | integer | NO |
| 4 | `map_name` | text | NO |
| 5 | `game_mode` | text | NO |
| 6 | `team_size` | integer | YES |
| 7 | `is_custom` | boolean | NO |
| 8 | `num_teams` | integer | YES |
| 9 | `num_players` | integer | YES |

키 및 무결성 제약:

- `matches_pkey`: `PRIMARY KEY (match_id)`

## stdb.moving_objects

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | NO |
| 2 | `agent_kind` | text | NO |
| 3 | `agent_id` | text | NO |
| 4 | `team_id` | integer | NO |
| 5 | `snapshot_t` | integer | NO |
| 6 | `pos` | geometry | NO |
| 7 | `health` | real | YES |
| 8 | `in_vehicle` | boolean | NO |
| 9 | `is_alive` | boolean | NO |

키 및 무결성 제약:

- `moving_objects_agent_kind_check`: `CHECK ((agent_kind = ANY (ARRAY['player'::text, 'team'::text])))`
- `moving_objects_pkey`: `PRIMARY KEY (match_id, agent_kind, agent_id, snapshot_t)`

## stdb.spatial_events

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `event_id` | bigint | NO |
| 2 | `match_id` | text | NO |
| 3 | `event_kind` | text | NO |
| 4 | `event_t` | real | NO |
| 5 | `attacker_id` | text | YES |
| 6 | `attacker_team` | integer | YES |
| 7 | `attacker_pos` | geometry | YES |
| 8 | `victim_id` | text | YES |
| 9 | `victim_team` | integer | YES |
| 10 | `victim_pos` | geometry | YES |
| 11 | `damage` | real | YES |
| 12 | `damage_type` | text | YES |
| 13 | `damage_causer` | text | YES |
| 14 | `distance_m` | real | YES |
| 15 | `extra` | jsonb | YES |

키 및 무결성 제약:

- `spatial_events_event_kind_check`: `CHECK ((event_kind = ANY (ARRAY['kill'::text, 'damage'::text, 'groggy'::text, 'landing'::text, 'phase'::text])))`
- `spatial_events_pkey`: `PRIMARY KEY (event_id)`

## stdb.team_phase_combat

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | NO |
| 2 | `team_id` | integer | NO |
| 3 | `phase_idx` | integer | NO |
| 4 | `damage_dealt` | real | NO |
| 5 | `damage_taken` | real | NO |
| 6 | `damage_events_dealt` | integer | NO |
| 7 | `damage_events_taken` | integer | NO |
| 8 | `groggy_dealt` | integer | NO |
| 9 | `groggy_taken` | integer | NO |
| 10 | `kills_dealt` | integer | NO |
| 11 | `kills_taken` | integer | NO |
| 12 | `distinct_opponents_damaged` | integer | NO |
| 13 | `distinct_opponents_damaged_by` | integer | NO |
| 14 | `combat_partner_count` | integer | NO |
| 15 | `first_damage_dealt_s` | real | YES |
| 16 | `first_damage_taken_s` | real | YES |
| 17 | `combat_initiated` | boolean | YES |
| 18 | `nearby_enemy_combat_events_200m` | integer | NO |
| 19 | `third_party_exposure_count` | integer | NO |
| 20 | `created_at` | timestamp with time zone | NO |

키 및 무결성 제약:

- `team_phase_combat_match_id_fkey`: `FOREIGN KEY (match_id) REFERENCES stdb.matches(match_id) ON DELETE CASCADE`
- `team_phase_combat_pkey`: `PRIMARY KEY (match_id, team_id, phase_idx)`

## stdb.team_phase_edge_cache

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | NO |
| 2 | `phase_idx` | integer | NO |
| 3 | `edge_type` | text | NO |
| 4 | `src_team_id` | integer | NO |
| 5 | `dst_team_id` | integer | NO |
| 6 | `weight` | real | NO |
| 7 | `support_count` | integer | NO |
| 8 | `avg_distance_m` | real | YES |
| 9 | `ratio` | real | YES |
| 10 | `damage_sum` | real | YES |
| 11 | `event_count` | integer | NO |
| 12 | `kill_count` | integer | NO |
| 13 | `groggy_count` | integer | NO |
| 14 | `created_at` | timestamp with time zone | NO |

키 및 무결성 제약:

- `team_phase_edge_cache_check`: `CHECK ((src_team_id <> dst_team_id))`
- `team_phase_edge_cache_edge_type_check`: `CHECK ((edge_type = ANY (ARRAY['proximity'::text, 'combat'::text, 'copressure'::text])))`
- `team_phase_edge_cache_pkey`: `PRIMARY KEY (match_id, phase_idx, edge_type, src_team_id, dst_team_id)`

## stdb.team_phase_exposure

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | NO |
| 2 | `team_id` | integer | NO |
| 3 | `phase_idx` | integer | NO |
| 4 | `mean_dist_to_boundary_m` | real | YES |
| 5 | `min_dist_to_boundary_m` | real | YES |
| 6 | `outside_safe_ratio` | real | YES |
| 7 | `avg_enemy_density_100m` | real | YES |
| 8 | `min_nearest_enemy_m` | real | YES |
| 9 | `path_near_enemy_ratio` | real | YES |
| 10 | `recent_damage_taken` | real | YES |
| 11 | `combat_zone_pass_count` | integer | YES |

키 및 무결성 제약:

- `team_phase_exposure_pkey`: `PRIMARY KEY (match_id, team_id, phase_idx)`

## stdb.team_phase_f0_extra

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | NO |
| 2 | `team_id` | integer | NO |
| 3 | `phase_idx` | integer | NO |
| 4 | `safe_center_x_start` | real | YES |
| 5 | `safe_center_y_start` | real | YES |
| 6 | `safe_center_x_end` | real | YES |
| 7 | `safe_center_y_end` | real | YES |
| 8 | `poison_center_x_start` | real | YES |
| 9 | `poison_center_y_start` | real | YES |
| 10 | `poison_center_x_end` | real | YES |
| 11 | `poison_center_y_end` | real | YES |
| 12 | `team_centroid_x_start` | real | YES |
| 13 | `team_centroid_y_start` | real | YES |
| 14 | `team_centroid_x_end` | real | YES |
| 15 | `team_centroid_y_end` | real | YES |
| 16 | `team_displacement_raw_m` | real | YES |
| 17 | `team_avg_speed_raw_mps` | real | YES |
| 18 | `created_at` | timestamp with time zone | NO |

키 및 무결성 제약:

- `team_phase_f0_extra_pkey`: `PRIMARY KEY (match_id, team_id, phase_idx)`

## stdb.team_phase_features

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | NO |
| 2 | `team_id` | integer | NO |
| 3 | `phase_idx` | integer | NO |
| 4 | `mean_dist_to_boundary_m` | real | YES |
| 5 | `enemy_density_100m` | real | YES |
| 6 | `recent_combat_exposure` | real | YES |
| 7 | `position_stability` | real | YES |
| 8 | `inside_safe_ratio` | real | YES |

키 및 무결성 제약:

- `team_phase_features_pkey`: `PRIMARY KEY (match_id, team_id, phase_idx)`

## stdb.team_phase_graph

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | NO |
| 2 | `team_id` | integer | NO |
| 3 | `phase_idx` | integer | NO |
| 4 | `avg_degree_100m` | real | YES |
| 5 | `max_degree_100m` | integer | YES |
| 6 | `avg_nearest_enemy_m` | real | YES |
| 7 | `isolation_score` | real | YES |
| 8 | `combat_edge_count` | integer | YES |
| 9 | `combat_partner_count` | integer | YES |
| 10 | `copress_200m_ratio` | real | YES |
| 11 | `had_combat` | boolean | YES |
| 12 | `survives_next_phase` | boolean | YES |

키 및 무결성 제약:

- `team_phase_graph_pkey`: `PRIMARY KEY (match_id, team_id, phase_idx)`

## stdb.team_phase_habitat_features

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | NO |
| 2 | `team_id` | integer | NO |
| 3 | `phase_idx` | integer | NO |
| 4 | `hab_safe_area_start_m2` | real | YES |
| 5 | `hab_safe_area_end_m2` | real | YES |
| 6 | `hab_safe_area_delta_m2` | real | YES |
| 7 | `hab_loss_ratio` | real | YES |
| 8 | `hab_shrink_rate_m2ps` | real | YES |
| 9 | `hab_alive_teams` | integer | YES |
| 10 | `hab_safe_area_per_alive_team` | real | YES |
| 11 | `hab_crowding_pressure` | real | YES |
| 12 | `hab_center_shift_m` | real | YES |
| 13 | `hab_center_shift_per_radius` | real | YES |
| 14 | `hab_zone_gap_start_m` | real | YES |
| 15 | `hab_zone_gap_end_m` | real | YES |
| 16 | `hab_zone_gap_pressure` | real | YES |
| 17 | `hab_dist_to_safe_boundary_start_m` | real | YES |
| 18 | `hab_dist_to_safe_boundary_end_m` | real | YES |
| 19 | `hab_signed_boundary_distance_start_m` | real | YES |
| 20 | `hab_signed_boundary_distance_end_m` | real | YES |
| 21 | `hab_distance_to_target_safe_m` | real | YES |
| 22 | `hab_required_speed_to_safe_mps` | real | YES |
| 23 | `hab_observed_speed_mps` | real | YES |
| 24 | `hab_migration_deficit_mps` | real | YES |
| 25 | `hab_outside_safe_ratio` | real | YES |
| 26 | `hab_edge_pressure_ratio` | real | YES |
| 27 | `hab_rotation_delay_ratio` | real | YES |
| 28 | `hab_safe_alignment_to_target` | real | YES |
| 29 | `hab_local_enemy_count_100m` | integer | YES |
| 30 | `hab_local_enemy_count_200m` | integer | YES |
| 31 | `hab_local_enemy_count_500m` | integer | YES |
| 32 | `hab_local_enemy_density_200m` | real | YES |
| 33 | `hab_safe_entry_enemy_count_200m` | integer | YES |
| 34 | `hab_safe_entry_congestion` | integer | YES |
| 35 | `hab_refuge_scarcity_simple` | real | YES |
| 36 | `created_at` | timestamp with time zone | NO |

키 및 무결성 제약:

- `team_phase_habitat_features_pkey`: `PRIMARY KEY (match_id, team_id, phase_idx)`

## stdb.team_phase_labels

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | NO |
| 2 | `team_id` | integer | NO |
| 3 | `cut_phase` | integer | NO |
| 4 | `alive_at_phase_start` | boolean | YES |
| 5 | `alive_at_phase_end` | boolean | YES |
| 6 | `next_phase_idx` | integer | YES |
| 7 | `target_defined` | boolean | NO |
| 8 | `survives_next_phase` | boolean | YES |
| 9 | `eliminated_next_phase` | boolean | YES |
| 10 | `hazard_next` | integer | YES |
| 11 | `final_rank` | integer | YES |
| 12 | `final_rank_norm` | real | YES |
| 13 | `final_top3` | boolean | YES |
| 14 | `final_top5` | boolean | YES |
| 15 | `final_top10` | boolean | YES |
| 16 | `final_winner` | boolean | YES |
| 17 | `last_phase_alive` | integer | YES |
| 18 | `created_at` | timestamp with time zone | NO |

키 및 무결성 제약:

- `team_phase_labels_match_id_fkey`: `FOREIGN KEY (match_id) REFERENCES stdb.matches(match_id) ON DELETE CASCADE`
- `team_phase_labels_pkey`: `PRIMARY KEY (match_id, team_id, cut_phase)`

## stdb.team_phase_loot

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | NO |
| 2 | `team_id` | integer | NO |
| 3 | `phase_idx` | integer | NO |
| 4 | `pickup_count` | integer | NO |
| 5 | `equip_count` | integer | NO |
| 6 | `use_count` | integer | NO |
| 7 | `unique_item_count` | integer | NO |
| 8 | `weapon_pickup_count` | integer | NO |
| 9 | `weapon_equip_count` | integer | NO |
| 10 | `ammo_pickup_count` | integer | NO |
| 11 | `armor_pickup_count` | integer | NO |
| 12 | `throwable_pickup_count` | integer | NO |
| 13 | `heal_use_count` | integer | NO |
| 14 | `boost_use_count` | integer | NO |
| 15 | `carepackage_pickup_count` | integer | NO |
| 16 | `lootbox_pickup_count` | integer | NO |
| 17 | `created_at` | timestamp with time zone | NO |

키 및 무결성 제약:

- `team_phase_loot_match_id_fkey`: `FOREIGN KEY (match_id) REFERENCES stdb.matches(match_id) ON DELETE CASCADE`
- `team_phase_loot_pkey`: `PRIMARY KEY (match_id, team_id, phase_idx)`

## stdb.team_phase_network_features

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | NO |
| 2 | `team_id` | integer | NO |
| 3 | `phase_idx` | integer | NO |
| 4 | `net_combat_out_degree` | integer | NO |
| 5 | `net_combat_in_degree` | integer | NO |
| 6 | `net_combat_total_degree` | integer | NO |
| 7 | `net_combat_out_strength` | real | NO |
| 8 | `net_combat_in_strength` | real | NO |
| 9 | `net_combat_total_strength` | real | NO |
| 10 | `net_combat_balance` | real | NO |
| 11 | `net_combat_unique_targets` | integer | NO |
| 12 | `net_combat_unique_attackers` | integer | NO |
| 13 | `net_combat_reciprocal_pairs` | integer | NO |
| 14 | `net_combat_reciprocity_ratio` | real | NO |
| 15 | `net_combat_out_strength_per_min` | real | YES |
| 16 | `net_combat_in_strength_per_min` | real | YES |
| 17 | `net_prox_degree_100m` | integer | NO |
| 18 | `net_prox_degree_200m` | integer | NO |
| 19 | `net_prox_degree_500m` | integer | NO |
| 20 | `net_prox_strength_decay` | real | NO |
| 21 | `net_prox_mean_distance_m` | real | YES |
| 22 | `net_prox_min_distance_m` | real | YES |
| 23 | `net_prox_top1_weight` | real | YES |
| 24 | `net_prox_top3_strength` | real | YES |
| 25 | `net_prox_isolation_m` | real | YES |
| 26 | `net_prox_neighbor_entropy` | real | YES |
| 27 | `net_copress_degree` | integer | NO |
| 28 | `net_copress_strength` | real | NO |
| 29 | `net_copress_max_weight` | real | NO |
| 30 | `net_copress_mean_ratio` | real | NO |
| 31 | `net_copress_shared_pressure_count` | integer | NO |
| 32 | `net_copress_top1_weight` | real | NO |
| 33 | `cum_combat_out_degree` | integer | NO |
| 34 | `cum_combat_in_degree` | integer | NO |
| 35 | `cum_combat_out_strength` | real | NO |
| 36 | `cum_combat_in_strength` | real | NO |
| 37 | `cum_combat_total_strength` | real | NO |
| 38 | `cum_combat_reciprocal_pairs` | integer | NO |
| 39 | `cum_unique_targets` | integer | NO |
| 40 | `cum_unique_attackers` | integer | NO |
| 41 | `cum_copress_strength` | real | NO |
| 42 | `cum_prox_close_exposure_200m` | integer | NO |
| 43 | `net_combat_out_strength_per_alive_team` | real | YES |
| 44 | `net_combat_in_strength_per_alive_team` | real | YES |
| 45 | `net_prox_degree_200m_per_alive_team` | real | YES |
| 46 | `net_copress_degree_per_alive_team` | real | YES |
| 47 | `created_at` | timestamp with time zone | NO |

키 및 무결성 제약:

- `team_phase_network_features_pkey`: `PRIMARY KEY (match_id, team_id, phase_idx)`

## stdb.team_phase_state

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | NO |
| 2 | `team_id` | integer | NO |
| 3 | `phase_idx` | integer | NO |
| 4 | `team_size` | integer | YES |
| 5 | `position_rows` | integer | YES |
| 6 | `alive_players_start` | integer | YES |
| 7 | `alive_players_end` | integer | YES |
| 8 | `alive_players_min` | real | YES |
| 9 | `alive_players_avg` | real | YES |
| 10 | `mean_health_avg` | real | YES |
| 11 | `min_health_min` | real | YES |
| 12 | `health_slope` | real | YES |
| 13 | `team_spread_avg_m` | real | YES |
| 14 | `team_spread_max_m` | real | YES |
| 15 | `vehicle_ratio` | real | YES |
| 16 | `vehicle_speed_mean` | real | YES |
| 17 | `vehicle_speed_max` | real | YES |
| 18 | `created_at` | timestamp with time zone | NO |

키 및 무결성 제약:

- `team_phase_state_match_id_fkey`: `FOREIGN KEY (match_id) REFERENCES stdb.matches(match_id) ON DELETE CASCADE`
- `team_phase_state_pkey`: `PRIMARY KEY (match_id, team_id, phase_idx)`

## stdb.team_phase_trajectory

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | NO |
| 2 | `team_id` | integer | NO |
| 3 | `phase_idx` | integer | NO |
| 4 | `waypoints` | integer | NO |
| 5 | `phase_start_s` | real | NO |
| 6 | `phase_end_s` | real | NO |
| 7 | `phase_duration_s` | real | NO |
| 8 | `any_alive` | boolean | NO |
| 9 | `trajectory` | geometry | YES |
| 10 | `path_length_m` | real | YES |
| 11 | `displacement_m` | real | YES |
| 12 | `sinuosity` | real | YES |
| 13 | `avg_speed_mps` | real | YES |
| 14 | `rotation_timing` | real | YES |
| 15 | `safe_alignment` | real | YES |
| 16 | `zone_entry_time` | real | YES |
| 17 | `outside_safe_duration_s` | real | YES |
| 18 | `boundary_distance_slope` | real | YES |

키 및 무결성 제약:

- `team_phase_trajectory_pkey`: `PRIMARY KEY (match_id, team_id, phase_idx)`

## stdb.team_phase_z_affordance_features

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | NO |
| 2 | `team_id` | integer | NO |
| 3 | `phase_idx` | integer | NO |
| 4 | `team_z_mean_m` | real | YES |
| 5 | `team_z_range_m` | real | YES |
| 6 | `team_z_percentile_alive` | real | YES |
| 7 | `contact_delta_z_mean_m` | real | YES |
| 8 | `contact_delta_z_positive_ratio` | real | YES |
| 9 | `high_ground_contact_advantage` | real | YES |
| 10 | `high_ground_x_boundary_pressure` | real | YES |
| 11 | `z_percentile_x_hab_crowding_pressure` | real | YES |
| 12 | `contact_delta_z_x_outside_safe_ratio` | real | YES |
| 13 | `created_at` | timestamp with time zone | NO |

키 및 무결성 제약:

- `team_phase_z_affordance_features_pkey`: `PRIMARY KEY (match_id, team_id, phase_idx)`

## stdb.team_proximity

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | NO |
| 2 | `snapshot_t` | integer | NO |
| 3 | `team_a` | integer | NO |
| 4 | `team_b` | integer | NO |
| 5 | `distance_m` | real | NO |
| 6 | `encounter_recently` | boolean | NO |

키 및 무결성 제약:

- `team_proximity_check`: `CHECK ((team_a < team_b))`
- `team_proximity_pkey`: `PRIMARY KEY (match_id, snapshot_t, team_a, team_b)`

## stdb.team_snapshot_features

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | NO |
| 2 | `team_id` | integer | NO |
| 3 | `snapshot_t` | integer | NO |
| 4 | `phase_idx` | integer | YES |
| 5 | `pos_x` | real | YES |
| 6 | `pos_y` | real | YES |
| 7 | `pos_z` | real | YES |
| 8 | `rel_zone_x` | real | YES |
| 9 | `rel_zone_y` | real | YES |
| 10 | `speed_mps` | real | YES |
| 11 | `dist_blue_boundary_m` | real | YES |
| 12 | `inside_blue` | boolean | YES |
| 13 | `dist_target_center_m` | real | YES |
| 14 | `mean_health` | real | YES |
| 15 | `min_health` | real | YES |
| 16 | `alive_n` | integer | YES |
| 17 | `vehicle_ratio` | real | YES |
| 18 | `created_at` | timestamp with time zone | NO |

키 및 무결성 제약:

- `team_snapshot_features_pkey`: `PRIMARY KEY (match_id, team_id, snapshot_t)`

## stdb.teams

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | NO |
| 2 | `team_id` | integer | NO |
| 3 | `final_rank` | integer | YES |
| 4 | `num_players` | integer | YES |
| 5 | `won` | boolean | NO |

키 및 무결성 제약:

- `teams_match_id_fkey`: `FOREIGN KEY (match_id) REFERENCES stdb.matches(match_id) ON DELETE CASCADE`
- `teams_pkey`: `PRIMARY KEY (match_id, team_id)`

## stdb.v_match_phase_bounds

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | YES |
| 2 | `phase_idx` | integer | YES |
| 3 | `phase_start_s` | real | YES |
| 4 | `phase_end_s` | real | YES |
| 5 | `phase_duration_s` | real | YES |

PK/FK/UNIQUE/CHECK 제약 없음 또는 view. nullable 정보는 위 표 참조.

## stdb.v_positions_ingame

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `id` | bigint | YES |
| 2 | `match_id` | text | YES |
| 3 | `event_time` | timestamp with time zone | YES |
| 4 | `is_game` | numeric | YES |
| 5 | `elapsed_time` | integer | YES |
| 6 | `num_alive` | integer | YES |
| 7 | `account_id` | text | YES |
| 8 | `player_name` | text | YES |
| 9 | `team_id` | integer | YES |
| 10 | `health` | numeric | YES |
| 11 | `pos_x` | numeric | YES |
| 12 | `pos_y` | numeric | YES |
| 13 | `pos_z` | numeric | YES |
| 14 | `vehicle_type` | text | YES |
| 15 | `vehicle_id` | text | YES |
| 16 | `vehicle_speed` | numeric | YES |

PK/FK/UNIQUE/CHECK 제약 없음 또는 view. nullable 정보는 위 표 참조.

## stdb.v_team_by_phase

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | YES |
| 2 | `team_id` | integer | YES |
| 3 | `snapshot_t` | integer | YES |
| 4 | `pos` | geometry | YES |
| 5 | `is_alive` | boolean | YES |
| 6 | `phase_idx` | integer | YES |

PK/FK/UNIQUE/CHECK 제약 없음 또는 view. nullable 정보는 위 표 참조.

## stdb.v_team_phase_feature_long

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | YES |
| 2 | `team_id` | integer | YES |
| 3 | `phase_idx` | integer | YES |
| 4 | `map_name` | text | YES |
| 5 | `game_mode` | text | YES |
| 6 | `match_team_size` | integer | YES |
| 7 | `phase_start_s` | real | YES |
| 8 | `phase_end_s` | real | YES |
| 9 | `phase_duration_s` | real | YES |
| 10 | `avg_alive_teams` | real | YES |
| 11 | `avg_alive_players` | real | YES |
| 12 | `safe_radius_start_m` | real | YES |
| 13 | `safe_radius_end_m` | real | YES |
| 14 | `safe_radius_delta_m` | real | YES |
| 15 | `safe_center_shift_m` | real | YES |
| 16 | `poison_radius_start_m` | real | YES |
| 17 | `poison_radius_end_m` | real | YES |
| 18 | `zone_gap_start_m` | real | YES |
| 19 | `zone_gap_end_m` | real | YES |
| 20 | `team_size` | integer | YES |
| 21 | `alive_players_start` | integer | YES |
| 22 | `alive_players_end` | integer | YES |
| 23 | `alive_players_min` | real | YES |
| 24 | `alive_players_avg` | real | YES |
| 25 | `mean_health_avg` | real | YES |
| 26 | `min_health_min` | real | YES |
| 27 | `health_slope` | real | YES |
| 28 | `team_spread_avg_m` | real | YES |
| 29 | `team_spread_max_m` | real | YES |
| 30 | `vehicle_ratio` | real | YES |
| 31 | `vehicle_speed_mean` | real | YES |
| 32 | `vehicle_speed_max` | real | YES |
| 33 | `path_length_m` | real | YES |
| 34 | `displacement_m` | real | YES |
| 35 | `sinuosity` | real | YES |
| 36 | `avg_speed_mps` | real | YES |
| 37 | `rotation_timing` | real | YES |
| 38 | `safe_alignment` | real | YES |
| 39 | `zone_entry_time` | real | YES |
| 40 | `outside_safe_duration_s` | real | YES |
| 41 | `boundary_distance_slope` | real | YES |
| 42 | `mean_dist_to_boundary_m` | real | YES |
| 43 | `min_dist_to_boundary_m` | real | YES |
| 44 | `outside_safe_ratio` | real | YES |
| 45 | `avg_enemy_density_100m` | real | YES |
| 46 | `min_nearest_enemy_m` | real | YES |
| 47 | `path_near_enemy_ratio` | real | YES |
| 48 | `damage_dealt` | real | YES |
| 49 | `damage_taken` | real | YES |
| 50 | `damage_events_dealt` | integer | YES |
| 51 | `damage_events_taken` | integer | YES |
| 52 | `groggy_dealt` | integer | YES |
| 53 | `groggy_taken` | integer | YES |
| 54 | `kills_dealt` | integer | YES |
| 55 | `kills_taken` | integer | YES |
| 56 | `distinct_opponents_damaged` | integer | YES |
| 57 | `distinct_opponents_damaged_by` | integer | YES |
| 58 | `combat_partner_count` | integer | YES |
| 59 | `combat_initiated` | boolean | YES |
| 60 | `nearby_enemy_combat_events_200m` | integer | YES |
| 61 | `third_party_exposure_count` | integer | YES |
| 62 | `pickup_count` | integer | YES |
| 63 | `equip_count` | integer | YES |
| 64 | `use_count` | integer | YES |
| 65 | `unique_item_count` | integer | YES |
| 66 | `weapon_pickup_count` | integer | YES |
| 67 | `weapon_equip_count` | integer | YES |
| 68 | `ammo_pickup_count` | integer | YES |
| 69 | `armor_pickup_count` | integer | YES |
| 70 | `throwable_pickup_count` | integer | YES |
| 71 | `heal_use_count` | integer | YES |
| 72 | `boost_use_count` | integer | YES |
| 73 | `carepackage_pickup_count` | integer | YES |
| 74 | `lootbox_pickup_count` | integer | YES |
| 75 | `target_defined` | boolean | YES |
| 76 | `hazard_next` | integer | YES |
| 77 | `survives_next_phase` | boolean | YES |
| 78 | `final_rank` | integer | YES |
| 79 | `final_rank_norm` | real | YES |
| 80 | `final_top3` | boolean | YES |
| 81 | `final_top5` | boolean | YES |
| 82 | `final_top10` | boolean | YES |
| 83 | `final_winner` | boolean | YES |
| 84 | `last_phase_alive` | integer | YES |
| 85 | `recent_damage_taken` | real | YES |
| 86 | `combat_zone_pass_count` | integer | YES |

PK/FK/UNIQUE/CHECK 제약 없음 또는 view. nullable 정보는 위 표 참조.

## stdb.v_team_phase_feature_v2

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | YES |
| 2 | `team_id` | integer | YES |
| 3 | `phase_idx` | integer | YES |
| 4 | `map_name` | text | YES |
| 5 | `game_mode` | text | YES |
| 6 | `match_team_size` | integer | YES |
| 7 | `phase_start_s` | real | YES |
| 8 | `phase_end_s` | real | YES |
| 9 | `phase_duration_s` | real | YES |
| 10 | `avg_alive_teams` | real | YES |
| 11 | `avg_alive_players` | real | YES |
| 12 | `safe_radius_start_m` | real | YES |
| 13 | `safe_radius_end_m` | real | YES |
| 14 | `safe_radius_delta_m` | real | YES |
| 15 | `safe_center_shift_m` | real | YES |
| 16 | `poison_radius_start_m` | real | YES |
| 17 | `poison_radius_end_m` | real | YES |
| 18 | `zone_gap_start_m` | real | YES |
| 19 | `zone_gap_end_m` | real | YES |
| 20 | `team_size` | integer | YES |
| 21 | `alive_players_start` | integer | YES |
| 22 | `alive_players_end` | integer | YES |
| 23 | `alive_players_min` | real | YES |
| 24 | `alive_players_avg` | real | YES |
| 25 | `mean_health_avg` | real | YES |
| 26 | `min_health_min` | real | YES |
| 27 | `health_slope` | real | YES |
| 28 | `team_spread_avg_m` | real | YES |
| 29 | `team_spread_max_m` | real | YES |
| 30 | `vehicle_ratio` | real | YES |
| 31 | `vehicle_speed_mean` | real | YES |
| 32 | `vehicle_speed_max` | real | YES |
| 33 | `path_length_m` | real | YES |
| 34 | `displacement_m` | real | YES |
| 35 | `sinuosity` | real | YES |
| 36 | `avg_speed_mps` | real | YES |
| 37 | `rotation_timing` | real | YES |
| 38 | `safe_alignment` | real | YES |
| 39 | `zone_entry_time` | real | YES |
| 40 | `outside_safe_duration_s` | real | YES |
| 41 | `boundary_distance_slope` | real | YES |
| 42 | `mean_dist_to_boundary_m` | real | YES |
| 43 | `min_dist_to_boundary_m` | real | YES |
| 44 | `outside_safe_ratio` | real | YES |
| 45 | `avg_enemy_density_100m` | real | YES |
| 46 | `min_nearest_enemy_m` | real | YES |
| 47 | `path_near_enemy_ratio` | real | YES |
| 48 | `damage_dealt` | real | YES |
| 49 | `damage_taken` | real | YES |
| 50 | `damage_events_dealt` | integer | YES |
| 51 | `damage_events_taken` | integer | YES |
| 52 | `groggy_dealt` | integer | YES |
| 53 | `groggy_taken` | integer | YES |
| 54 | `kills_dealt` | integer | YES |
| 55 | `kills_taken` | integer | YES |
| 56 | `distinct_opponents_damaged` | integer | YES |
| 57 | `distinct_opponents_damaged_by` | integer | YES |
| 58 | `combat_partner_count` | integer | YES |
| 59 | `combat_initiated` | boolean | YES |
| 60 | `nearby_enemy_combat_events_200m` | integer | YES |
| 61 | `third_party_exposure_count` | integer | YES |
| 62 | `pickup_count` | integer | YES |
| 63 | `equip_count` | integer | YES |
| 64 | `use_count` | integer | YES |
| 65 | `unique_item_count` | integer | YES |
| 66 | `weapon_pickup_count` | integer | YES |
| 67 | `weapon_equip_count` | integer | YES |
| 68 | `ammo_pickup_count` | integer | YES |
| 69 | `armor_pickup_count` | integer | YES |
| 70 | `throwable_pickup_count` | integer | YES |
| 71 | `heal_use_count` | integer | YES |
| 72 | `boost_use_count` | integer | YES |
| 73 | `carepackage_pickup_count` | integer | YES |
| 74 | `lootbox_pickup_count` | integer | YES |
| 75 | `target_defined` | boolean | YES |
| 76 | `hazard_next` | integer | YES |
| 77 | `survives_next_phase` | boolean | YES |
| 78 | `final_rank` | integer | YES |
| 79 | `final_rank_norm` | real | YES |
| 80 | `final_top3` | boolean | YES |
| 81 | `final_top5` | boolean | YES |
| 82 | `final_top10` | boolean | YES |
| 83 | `final_winner` | boolean | YES |
| 84 | `last_phase_alive` | integer | YES |
| 85 | `recent_damage_taken` | real | YES |
| 86 | `combat_zone_pass_count` | integer | YES |
| 87 | `weather_id` | text | YES |
| 88 | `camera_view_type` | text | YES |
| 89 | `start_alive_teams` | integer | YES |
| 90 | `end_alive_teams` | integer | YES |
| 91 | `start_alive_players` | integer | YES |
| 92 | `end_alive_players` | integer | YES |
| 93 | `poison_radius_delta_m` | real | YES |
| 94 | `red_zone_active_ratio` | real | YES |
| 95 | `black_zone_active_ratio` | real | YES |
| 96 | `position_rows` | integer | YES |
| 97 | `first_damage_dealt_s` | real | YES |
| 98 | `first_damage_taken_s` | real | YES |
| 99 | `safe_center_x_start` | real | YES |
| 100 | `safe_center_y_start` | real | YES |
| 101 | `safe_center_x_end` | real | YES |
| 102 | `safe_center_y_end` | real | YES |
| 103 | `poison_center_x_start` | real | YES |
| 104 | `poison_center_y_start` | real | YES |
| 105 | `poison_center_x_end` | real | YES |
| 106 | `poison_center_y_end` | real | YES |
| 107 | `team_centroid_x_start` | real | YES |
| 108 | `team_centroid_y_start` | real | YES |
| 109 | `team_centroid_x_end` | real | YES |
| 110 | `team_centroid_y_end` | real | YES |
| 111 | `team_displacement_raw_m` | real | YES |
| 112 | `team_avg_speed_raw_mps` | real | YES |
| 113 | `net_combat_out_degree` | integer | YES |
| 114 | `net_combat_in_degree` | integer | YES |
| 115 | `net_combat_total_degree` | integer | YES |
| 116 | `net_combat_out_strength` | real | YES |
| 117 | `net_combat_in_strength` | real | YES |
| 118 | `net_combat_total_strength` | real | YES |
| 119 | `net_combat_balance` | real | YES |
| 120 | `net_combat_unique_targets` | integer | YES |
| 121 | `net_combat_unique_attackers` | integer | YES |
| 122 | `net_combat_reciprocal_pairs` | integer | YES |
| 123 | `net_combat_reciprocity_ratio` | real | YES |
| 124 | `net_combat_out_strength_per_min` | real | YES |
| 125 | `net_combat_in_strength_per_min` | real | YES |
| 126 | `net_prox_degree_100m` | integer | YES |
| 127 | `net_prox_degree_200m` | integer | YES |
| 128 | `net_prox_degree_500m` | integer | YES |
| 129 | `net_prox_strength_decay` | real | YES |
| 130 | `net_prox_mean_distance_m` | real | YES |
| 131 | `net_prox_min_distance_m` | real | YES |
| 132 | `net_prox_top1_weight` | real | YES |
| 133 | `net_prox_top3_strength` | real | YES |
| 134 | `net_prox_isolation_m` | real | YES |
| 135 | `net_prox_neighbor_entropy` | real | YES |
| 136 | `net_copress_degree` | integer | YES |
| 137 | `net_copress_strength` | real | YES |
| 138 | `net_copress_max_weight` | real | YES |
| 139 | `net_copress_mean_ratio` | real | YES |
| 140 | `net_copress_shared_pressure_count` | integer | YES |
| 141 | `net_copress_top1_weight` | real | YES |
| 142 | `cum_combat_out_degree` | integer | YES |
| 143 | `cum_combat_in_degree` | integer | YES |
| 144 | `cum_combat_out_strength` | real | YES |
| 145 | `cum_combat_in_strength` | real | YES |
| 146 | `cum_combat_total_strength` | real | YES |
| 147 | `cum_combat_reciprocal_pairs` | integer | YES |
| 148 | `cum_unique_targets` | integer | YES |
| 149 | `cum_unique_attackers` | integer | YES |
| 150 | `cum_copress_strength` | real | YES |
| 151 | `cum_prox_close_exposure_200m` | integer | YES |
| 152 | `net_combat_out_strength_per_alive_team` | real | YES |
| 153 | `net_combat_in_strength_per_alive_team` | real | YES |
| 154 | `net_prox_degree_200m_per_alive_team` | real | YES |
| 155 | `net_copress_degree_per_alive_team` | real | YES |
| 156 | `hab_safe_area_start_m2` | real | YES |
| 157 | `hab_safe_area_end_m2` | real | YES |
| 158 | `hab_safe_area_delta_m2` | real | YES |
| 159 | `hab_loss_ratio` | real | YES |
| 160 | `hab_shrink_rate_m2ps` | real | YES |
| 161 | `hab_alive_teams` | integer | YES |
| 162 | `hab_safe_area_per_alive_team` | real | YES |
| 163 | `hab_crowding_pressure` | real | YES |
| 164 | `hab_center_shift_m` | real | YES |
| 165 | `hab_center_shift_per_radius` | real | YES |
| 166 | `hab_zone_gap_start_m` | real | YES |
| 167 | `hab_zone_gap_end_m` | real | YES |
| 168 | `hab_zone_gap_pressure` | real | YES |
| 169 | `hab_dist_to_safe_boundary_start_m` | real | YES |
| 170 | `hab_dist_to_safe_boundary_end_m` | real | YES |
| 171 | `hab_signed_boundary_distance_start_m` | real | YES |
| 172 | `hab_signed_boundary_distance_end_m` | real | YES |
| 173 | `hab_distance_to_target_safe_m` | real | YES |
| 174 | `hab_required_speed_to_safe_mps` | real | YES |
| 175 | `hab_observed_speed_mps` | real | YES |
| 176 | `hab_migration_deficit_mps` | real | YES |
| 177 | `hab_outside_safe_ratio` | real | YES |
| 178 | `hab_edge_pressure_ratio` | real | YES |
| 179 | `hab_rotation_delay_ratio` | real | YES |
| 180 | `hab_safe_alignment_to_target` | real | YES |
| 181 | `hab_local_enemy_count_100m` | integer | YES |
| 182 | `hab_local_enemy_count_200m` | integer | YES |
| 183 | `hab_local_enemy_count_500m` | integer | YES |
| 184 | `hab_local_enemy_density_200m` | real | YES |
| 185 | `hab_safe_entry_enemy_count_200m` | integer | YES |
| 186 | `hab_safe_entry_congestion` | integer | YES |
| 187 | `hab_refuge_scarcity_simple` | real | YES |
| 188 | `team_z_mean_m` | real | YES |
| 189 | `team_z_percentile_alive` | real | YES |
| 190 | `team_z_range_m` | real | YES |
| 191 | `contact_delta_z_mean_m` | real | YES |
| 192 | `contact_delta_z_positive_ratio` | real | YES |
| 193 | `high_ground_contact_advantage` | real | YES |
| 194 | `high_ground_x_boundary_pressure` | real | YES |
| 195 | `z_percentile_x_hab_crowding_pressure` | real | YES |
| 196 | `contact_delta_z_x_outside_safe_ratio` | real | YES |

PK/FK/UNIQUE/CHECK 제약 없음 또는 view. nullable 정보는 위 표 참조.

## stdb.v_team_phase_official_raw_features

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | YES |
| 2 | `team_id` | integer | YES |
| 3 | `phase_idx` | integer | YES |
| 4 | `map_name` | text | YES |
| 5 | `game_mode` | text | YES |
| 6 | `match_team_size` | integer | YES |
| 7 | `weather_id` | text | YES |
| 8 | `camera_view_type` | text | YES |
| 9 | `phase_start_s` | real | YES |
| 10 | `phase_end_s` | real | YES |
| 11 | `phase_duration_s` | real | YES |
| 12 | `avg_alive_teams` | real | YES |
| 13 | `start_alive_teams` | integer | YES |
| 14 | `end_alive_teams` | integer | YES |
| 15 | `avg_alive_players` | real | YES |
| 16 | `start_alive_players` | integer | YES |
| 17 | `end_alive_players` | integer | YES |
| 18 | `safe_radius_start_m` | real | YES |
| 19 | `safe_radius_end_m` | real | YES |
| 20 | `safe_radius_delta_m` | real | YES |
| 21 | `poison_radius_start_m` | real | YES |
| 22 | `poison_radius_end_m` | real | YES |
| 23 | `poison_radius_delta_m` | real | YES |
| 24 | `safe_center_x_start` | real | YES |
| 25 | `safe_center_y_start` | real | YES |
| 26 | `safe_center_x_end` | real | YES |
| 27 | `safe_center_y_end` | real | YES |
| 28 | `poison_center_x_start` | real | YES |
| 29 | `poison_center_y_start` | real | YES |
| 30 | `poison_center_x_end` | real | YES |
| 31 | `poison_center_y_end` | real | YES |
| 32 | `red_zone_active_ratio` | real | YES |
| 33 | `black_zone_active_ratio` | real | YES |
| 34 | `team_size` | integer | YES |
| 35 | `position_rows` | integer | YES |
| 36 | `alive_players_start` | integer | YES |
| 37 | `alive_players_end` | integer | YES |
| 38 | `alive_players_min` | real | YES |
| 39 | `alive_players_avg` | real | YES |
| 40 | `mean_health_avg` | real | YES |
| 41 | `min_health_min` | real | YES |
| 42 | `health_slope` | real | YES |
| 43 | `team_spread_avg_m` | real | YES |
| 44 | `team_spread_max_m` | real | YES |
| 45 | `vehicle_ratio` | real | YES |
| 46 | `vehicle_speed_mean` | real | YES |
| 47 | `vehicle_speed_max` | real | YES |
| 48 | `team_centroid_x_start` | real | YES |
| 49 | `team_centroid_y_start` | real | YES |
| 50 | `team_centroid_x_end` | real | YES |
| 51 | `team_centroid_y_end` | real | YES |
| 52 | `team_displacement_raw_m` | real | YES |
| 53 | `team_avg_speed_raw_mps` | real | YES |
| 54 | `damage_dealt` | real | YES |
| 55 | `damage_taken` | real | YES |
| 56 | `damage_events_dealt` | integer | YES |
| 57 | `damage_events_taken` | integer | YES |
| 58 | `groggy_dealt` | integer | YES |
| 59 | `groggy_taken` | integer | YES |
| 60 | `kills_dealt` | integer | YES |
| 61 | `kills_taken` | integer | YES |
| 62 | `distinct_opponents_damaged` | integer | YES |
| 63 | `distinct_opponents_damaged_by` | integer | YES |
| 64 | `combat_partner_count` | integer | YES |
| 65 | `first_damage_dealt_s` | real | YES |
| 66 | `first_damage_taken_s` | real | YES |
| 67 | `combat_initiated` | boolean | YES |
| 68 | `pickup_count` | integer | YES |
| 69 | `equip_count` | integer | YES |
| 70 | `use_count` | integer | YES |
| 71 | `unique_item_count` | integer | YES |
| 72 | `weapon_pickup_count` | integer | YES |
| 73 | `weapon_equip_count` | integer | YES |
| 74 | `ammo_pickup_count` | integer | YES |
| 75 | `armor_pickup_count` | integer | YES |
| 76 | `throwable_pickup_count` | integer | YES |
| 77 | `heal_use_count` | integer | YES |
| 78 | `boost_use_count` | integer | YES |
| 79 | `carepackage_pickup_count` | integer | YES |
| 80 | `lootbox_pickup_count` | integer | YES |
| 81 | `target_defined` | boolean | YES |
| 82 | `hazard_next` | integer | YES |
| 83 | `final_rank_norm` | real | YES |
| 84 | `final_top5` | boolean | YES |
| 85 | `final_top10` | boolean | YES |

PK/FK/UNIQUE/CHECK 제약 없음 또는 view. nullable 정보는 위 표 참조.

## stdb.v_team_phase_trajectory

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | YES |
| 2 | `team_id` | integer | YES |
| 3 | `phase_idx` | integer | YES |
| 4 | `waypoints` | bigint | YES |
| 5 | `phase_start_s` | integer | YES |
| 6 | `phase_end_s` | integer | YES |
| 7 | `any_alive` | boolean | YES |
| 8 | `trajectory` | geometry | YES |
| 9 | `trajectory_2d` | geometry | YES |

PK/FK/UNIQUE/CHECK 제약 없음 또는 view. nullable 정보는 위 표 참조.

## stdb.v_team_phase_wide

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | YES |
| 2 | `team_id` | integer | YES |
| 3 | `final_rank` | integer | YES |
| 4 | `tier` | text | YES |
| 5 | `last_phase_alive` | integer | YES |
| 6 | `p1_path_length_m` | real | YES |
| 7 | `p2_path_length_m` | real | YES |
| 8 | `p3_path_length_m` | real | YES |
| 9 | `p4_path_length_m` | real | YES |
| 10 | `p5_path_length_m` | real | YES |
| 11 | `p6_path_length_m` | real | YES |
| 12 | `p1_displacement_m` | real | YES |
| 13 | `p2_displacement_m` | real | YES |
| 14 | `p3_displacement_m` | real | YES |
| 15 | `p4_displacement_m` | real | YES |
| 16 | `p5_displacement_m` | real | YES |
| 17 | `p6_displacement_m` | real | YES |
| 18 | `p1_sinuosity` | real | YES |
| 19 | `p2_sinuosity` | real | YES |
| 20 | `p3_sinuosity` | real | YES |
| 21 | `p4_sinuosity` | real | YES |
| 22 | `p5_sinuosity` | real | YES |
| 23 | `p6_sinuosity` | real | YES |
| 24 | `p1_avg_speed_mps` | real | YES |
| 25 | `p2_avg_speed_mps` | real | YES |
| 26 | `p3_avg_speed_mps` | real | YES |
| 27 | `p4_avg_speed_mps` | real | YES |
| 28 | `p5_avg_speed_mps` | real | YES |
| 29 | `p6_avg_speed_mps` | real | YES |
| 30 | `p1_rotation_timing` | real | YES |
| 31 | `p2_rotation_timing` | real | YES |
| 32 | `p3_rotation_timing` | real | YES |
| 33 | `p4_rotation_timing` | real | YES |
| 34 | `p5_rotation_timing` | real | YES |
| 35 | `p6_rotation_timing` | real | YES |
| 36 | `p1_safe_alignment` | real | YES |
| 37 | `p2_safe_alignment` | real | YES |
| 38 | `p3_safe_alignment` | real | YES |
| 39 | `p4_safe_alignment` | real | YES |
| 40 | `p5_safe_alignment` | real | YES |
| 41 | `p6_safe_alignment` | real | YES |
| 42 | `p1_zone_entry_time` | real | YES |
| 43 | `p2_zone_entry_time` | real | YES |
| 44 | `p3_zone_entry_time` | real | YES |
| 45 | `p4_zone_entry_time` | real | YES |
| 46 | `p5_zone_entry_time` | real | YES |
| 47 | `p6_zone_entry_time` | real | YES |
| 48 | `p1_outside_safe_duratio` | real | YES |
| 49 | `p2_outside_safe_duratio` | real | YES |
| 50 | `p3_outside_safe_duratio` | real | YES |
| 51 | `p4_outside_safe_duratio` | real | YES |
| 52 | `p5_outside_safe_duratio` | real | YES |
| 53 | `p6_outside_safe_duratio` | real | YES |
| 54 | `p1_boundary_distance_sl` | real | YES |
| 55 | `p2_boundary_distance_sl` | real | YES |
| 56 | `p3_boundary_distance_sl` | real | YES |
| 57 | `p4_boundary_distance_sl` | real | YES |
| 58 | `p5_boundary_distance_sl` | real | YES |
| 59 | `p6_boundary_distance_sl` | real | YES |
| 60 | `p1_mean_dist_to_boundar` | real | YES |
| 61 | `p2_mean_dist_to_boundar` | real | YES |
| 62 | `p3_mean_dist_to_boundar` | real | YES |
| 63 | `p4_mean_dist_to_boundar` | real | YES |
| 64 | `p5_mean_dist_to_boundar` | real | YES |
| 65 | `p6_mean_dist_to_boundar` | real | YES |
| 66 | `p1_min_dist_to_boundary` | real | YES |
| 67 | `p2_min_dist_to_boundary` | real | YES |
| 68 | `p3_min_dist_to_boundary` | real | YES |
| 69 | `p4_min_dist_to_boundary` | real | YES |
| 70 | `p5_min_dist_to_boundary` | real | YES |
| 71 | `p6_min_dist_to_boundary` | real | YES |
| 72 | `p1_outside_safe_ratio` | real | YES |
| 73 | `p2_outside_safe_ratio` | real | YES |
| 74 | `p3_outside_safe_ratio` | real | YES |
| 75 | `p4_outside_safe_ratio` | real | YES |
| 76 | `p5_outside_safe_ratio` | real | YES |
| 77 | `p6_outside_safe_ratio` | real | YES |
| 78 | `p1_avg_enemy_density_10` | real | YES |
| 79 | `p2_avg_enemy_density_10` | real | YES |
| 80 | `p3_avg_enemy_density_10` | real | YES |
| 81 | `p4_avg_enemy_density_10` | real | YES |
| 82 | `p5_avg_enemy_density_10` | real | YES |
| 83 | `p6_avg_enemy_density_10` | real | YES |
| 84 | `p1_min_nearest_enemy_m` | real | YES |
| 85 | `p2_min_nearest_enemy_m` | real | YES |
| 86 | `p3_min_nearest_enemy_m` | real | YES |
| 87 | `p4_min_nearest_enemy_m` | real | YES |
| 88 | `p5_min_nearest_enemy_m` | real | YES |
| 89 | `p6_min_nearest_enemy_m` | real | YES |
| 90 | `p1_path_near_enemy_rati` | real | YES |
| 91 | `p2_path_near_enemy_rati` | real | YES |
| 92 | `p3_path_near_enemy_rati` | real | YES |
| 93 | `p4_path_near_enemy_rati` | real | YES |
| 94 | `p5_path_near_enemy_rati` | real | YES |
| 95 | `p6_path_near_enemy_rati` | real | YES |
| 96 | `p1_recent_damage_taken` | real | YES |
| 97 | `p2_recent_damage_taken` | real | YES |
| 98 | `p3_recent_damage_taken` | real | YES |
| 99 | `p4_recent_damage_taken` | real | YES |
| 100 | `p5_recent_damage_taken` | real | YES |
| 101 | `p6_recent_damage_taken` | real | YES |
| 102 | `p1_combat_zone_pass_cou` | integer | YES |
| 103 | `p2_combat_zone_pass_cou` | integer | YES |
| 104 | `p3_combat_zone_pass_cou` | integer | YES |
| 105 | `p4_combat_zone_pass_cou` | integer | YES |
| 106 | `p5_combat_zone_pass_cou` | integer | YES |
| 107 | `p6_combat_zone_pass_cou` | integer | YES |

PK/FK/UNIQUE/CHECK 제약 없음 또는 view. nullable 정보는 위 표 참조.

## stdb.v_team_truncated_targets

| 순서 | 컬럼 | PostgreSQL 타입 | nullable |
|---:|---|---|---|
| 1 | `match_id` | text | YES |
| 2 | `team_id` | integer | YES |
| 3 | `final_rank` | integer | YES |
| 4 | `last_phase_alive` | integer | YES |
| 5 | `cut_phase` | integer | YES |
| 6 | `survives_next_phase` | boolean | YES |
| 7 | `survives_two_more` | boolean | YES |
| 8 | `final_top5` | boolean | YES |
| 9 | `final_top10` | boolean | YES |
| 10 | `final_winner` | boolean | YES |
| 11 | `final_rank_target` | integer | YES |

PK/FK/UNIQUE/CHECK 제약 없음 또는 view. nullable 정보는 위 표 참조.
