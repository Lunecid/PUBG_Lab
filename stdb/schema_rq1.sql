-- stdb/schema_rq1.sql
-- =============================================================================
-- RQ1 STDB feature layer
--
-- This file is additive. It does not drop or rewrite the existing base STDB
-- objects in schema.sql. It formalizes the phase-level PUBG domain features
-- needed by the redesigned RQ1/RQ3 pipeline:
--
--   raw pubg.* telemetry
--      -> base STDB objects
--      -> phase-level global / state / combat / loot / label tables
--      -> model-ready long feature view
--
-- Apply with:
--   python -m stdb.rq1_feature_etl --apply-schema --all-current
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE SCHEMA IF NOT EXISTS stdb;

-- -----------------------------------------------------------------------------
-- Phase bounds derived from dynamic_regions after phase recovery.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW stdb.v_match_phase_bounds AS
SELECT
    match_id,
    phase_idx::integer AS phase_idx,
    min(snapshot_t)::real AS phase_start_s,
    max(snapshot_t)::real AS phase_end_s,
    greatest(max(snapshot_t) - min(snapshot_t), 0)::real AS phase_duration_s
FROM stdb.dynamic_regions
WHERE phase_idx IS NOT NULL
  AND phase_idx > 0
GROUP BY match_id, phase_idx;

-- -----------------------------------------------------------------------------
-- Match/phase context: shrinking zone, alive population, and match settings.
-- One row per (match, phase).
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stdb.match_phase_global (
    match_id                    TEXT    NOT NULL REFERENCES stdb.matches(match_id) ON DELETE CASCADE,
    phase_idx                   INTEGER NOT NULL,
    phase_start_s               REAL    NOT NULL,
    phase_end_s                 REAL    NOT NULL,
    phase_duration_s            REAL,

    avg_alive_teams             REAL,
    start_alive_teams           INTEGER,
    end_alive_teams             INTEGER,
    avg_alive_players           REAL,
    start_alive_players         INTEGER,
    end_alive_players           INTEGER,

    safe_radius_start_m         REAL,
    safe_radius_end_m           REAL,
    safe_radius_delta_m         REAL,
    safe_center_shift_m         REAL,
    poison_radius_start_m       REAL,
    poison_radius_end_m         REAL,
    poison_radius_delta_m       REAL,
    zone_gap_start_m            REAL,
    zone_gap_end_m              REAL,
    safe_area_ratio_start       REAL,
    safe_area_ratio_end         REAL,

    red_zone_active_ratio       REAL,
    black_zone_active_ratio     REAL,
    weather_id                  TEXT,
    camera_view_type            TEXT,
    blue_zone_options           JSONB,

    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (match_id, phase_idx)
);
CREATE INDEX IF NOT EXISTS ix_mpg_match_phase
    ON stdb.match_phase_global (match_id, phase_idx);

-- -----------------------------------------------------------------------------
-- Team state features from player-level position telemetry.
-- One row per (match, team, phase).
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stdb.team_phase_state (
    match_id                    TEXT    NOT NULL REFERENCES stdb.matches(match_id) ON DELETE CASCADE,
    team_id                     INTEGER NOT NULL,
    phase_idx                   INTEGER NOT NULL,

    team_size                   INTEGER,
    position_rows               INTEGER,
    alive_players_start         INTEGER,
    alive_players_end           INTEGER,
    alive_players_min           REAL,
    alive_players_avg           REAL,
    mean_health_avg             REAL,
    min_health_min              REAL,
    health_slope                REAL,

    team_spread_avg_m           REAL,
    team_spread_max_m           REAL,
    vehicle_ratio               REAL,
    vehicle_speed_mean          REAL,
    vehicle_speed_max           REAL,

    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (match_id, team_id, phase_idx)
);
CREATE INDEX IF NOT EXISTS ix_tps_match_phase
    ON stdb.team_phase_state (match_id, phase_idx);
CREATE INDEX IF NOT EXISTS ix_tps_team
    ON stdb.team_phase_state (match_id, team_id);

-- -----------------------------------------------------------------------------
-- Directed combat features from kill / damage / groggy telemetry.
-- One row per (match, team, phase).
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stdb.team_phase_combat (
    match_id                         TEXT    NOT NULL REFERENCES stdb.matches(match_id) ON DELETE CASCADE,
    team_id                          INTEGER NOT NULL,
    phase_idx                        INTEGER NOT NULL,

    damage_dealt                     REAL    NOT NULL DEFAULT 0,
    damage_taken                     REAL    NOT NULL DEFAULT 0,
    damage_events_dealt              INTEGER NOT NULL DEFAULT 0,
    damage_events_taken              INTEGER NOT NULL DEFAULT 0,
    groggy_dealt                     INTEGER NOT NULL DEFAULT 0,
    groggy_taken                     INTEGER NOT NULL DEFAULT 0,
    kills_dealt                      INTEGER NOT NULL DEFAULT 0,
    kills_taken                      INTEGER NOT NULL DEFAULT 0,
    distinct_opponents_damaged       INTEGER NOT NULL DEFAULT 0,
    distinct_opponents_damaged_by    INTEGER NOT NULL DEFAULT 0,
    combat_partner_count             INTEGER NOT NULL DEFAULT 0,
    first_damage_dealt_s             REAL,
    first_damage_taken_s             REAL,
    combat_initiated                 BOOLEAN,
    nearby_enemy_combat_events_200m  INTEGER NOT NULL DEFAULT 0,
    third_party_exposure_count       INTEGER NOT NULL DEFAULT 0,

    created_at                       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (match_id, team_id, phase_idx)
);
CREATE INDEX IF NOT EXISTS ix_tpc_match_phase
    ON stdb.team_phase_combat (match_id, phase_idx);
CREATE INDEX IF NOT EXISTS ix_tpc_team
    ON stdb.team_phase_combat (match_id, team_id);

-- -----------------------------------------------------------------------------
-- Loot / equipment / consumable features from item telemetry.
-- One row per (match, team, phase).
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stdb.team_phase_loot (
    match_id                    TEXT    NOT NULL REFERENCES stdb.matches(match_id) ON DELETE CASCADE,
    team_id                     INTEGER NOT NULL,
    phase_idx                   INTEGER NOT NULL,

    pickup_count                INTEGER NOT NULL DEFAULT 0,
    equip_count                 INTEGER NOT NULL DEFAULT 0,
    use_count                   INTEGER NOT NULL DEFAULT 0,
    unique_item_count           INTEGER NOT NULL DEFAULT 0,
    weapon_pickup_count         INTEGER NOT NULL DEFAULT 0,
    weapon_equip_count          INTEGER NOT NULL DEFAULT 0,
    ammo_pickup_count           INTEGER NOT NULL DEFAULT 0,
    armor_pickup_count          INTEGER NOT NULL DEFAULT 0,
    throwable_pickup_count      INTEGER NOT NULL DEFAULT 0,
    heal_use_count              INTEGER NOT NULL DEFAULT 0,
    boost_use_count             INTEGER NOT NULL DEFAULT 0,
    carepackage_pickup_count    INTEGER NOT NULL DEFAULT 0,
    lootbox_pickup_count        INTEGER NOT NULL DEFAULT 0,

    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (match_id, team_id, phase_idx)
);
CREATE INDEX IF NOT EXISTS ix_tpl_match_phase
    ON stdb.team_phase_loot (match_id, phase_idx);
CREATE INDEX IF NOT EXISTS ix_tpl_team
    ON stdb.team_phase_loot (match_id, team_id);

-- -----------------------------------------------------------------------------
-- Leakage-safe phase labels.
-- cut_phase means: features observed through cut_phase, predict the next phase.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stdb.team_phase_labels (
    match_id                    TEXT    NOT NULL REFERENCES stdb.matches(match_id) ON DELETE CASCADE,
    team_id                     INTEGER NOT NULL,
    cut_phase                   INTEGER NOT NULL,

    alive_at_phase_start        BOOLEAN,
    alive_at_phase_end          BOOLEAN,
    next_phase_idx              INTEGER,
    target_defined             BOOLEAN NOT NULL DEFAULT FALSE,
    survives_next_phase         BOOLEAN,
    eliminated_next_phase       BOOLEAN,
    hazard_next                 INTEGER,

    final_rank                  INTEGER,
    final_rank_norm             REAL,
    final_top3                  BOOLEAN,
    final_top5                  BOOLEAN,
    final_top10                 BOOLEAN,
    final_winner                BOOLEAN,
    last_phase_alive            INTEGER,

    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (match_id, team_id, cut_phase)
);
CREATE INDEX IF NOT EXISTS ix_tplabel_match_phase
    ON stdb.team_phase_labels (match_id, cut_phase);
CREATE INDEX IF NOT EXISTS ix_tplabel_team
    ON stdb.team_phase_labels (match_id, team_id);

-- -----------------------------------------------------------------------------
-- Model-ready long feature view. Graph edge tensors are built separately from
-- team_proximity / spatial_events, so graph summary columns remain analysis-only.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW stdb.v_team_phase_feature_long AS
SELECT
    t.match_id,
    t.team_id,
    b.phase_idx,

    -- meta / global
    m.map_name,
    m.game_mode,
    m.team_size AS match_team_size,
    g.phase_start_s,
    g.phase_end_s,
    g.phase_duration_s,
    g.avg_alive_teams,
    g.avg_alive_players,
    g.safe_radius_start_m,
    g.safe_radius_end_m,
    g.safe_radius_delta_m,
    g.safe_center_shift_m,
    g.poison_radius_start_m,
    g.poison_radius_end_m,
    g.zone_gap_start_m,
    g.zone_gap_end_m,

    -- basic state
    s.team_size,
    s.alive_players_start,
    s.alive_players_end,
    s.alive_players_min,
    s.alive_players_avg,
    s.mean_health_avg,
    s.min_health_min,
    s.health_slope,
    s.team_spread_avg_m,
    s.team_spread_max_m,
    s.vehicle_ratio,
    s.vehicle_speed_mean,
    s.vehicle_speed_max,

    -- trajectory / exposure
    tr.path_length_m,
    tr.displacement_m,
    tr.sinuosity,
    tr.avg_speed_mps,
    tr.rotation_timing,
    tr.safe_alignment,
    tr.zone_entry_time,
    tr.outside_safe_duration_s,
    tr.boundary_distance_slope,
    ex.mean_dist_to_boundary_m,
    ex.min_dist_to_boundary_m,
    ex.outside_safe_ratio,
    ex.avg_enemy_density_100m,
    ex.min_nearest_enemy_m,
    ex.path_near_enemy_ratio,

    -- combat
    c.damage_dealt,
    c.damage_taken,
    c.damage_events_dealt,
    c.damage_events_taken,
    c.groggy_dealt,
    c.groggy_taken,
    c.kills_dealt,
    c.kills_taken,
    c.distinct_opponents_damaged,
    c.distinct_opponents_damaged_by,
    c.combat_partner_count,
    c.combat_initiated,
    c.nearby_enemy_combat_events_200m,
    c.third_party_exposure_count,

    -- loot
    l.pickup_count,
    l.equip_count,
    l.use_count,
    l.unique_item_count,
    l.weapon_pickup_count,
    l.weapon_equip_count,
    l.ammo_pickup_count,
    l.armor_pickup_count,
    l.throwable_pickup_count,
    l.heal_use_count,
    l.boost_use_count,
    l.carepackage_pickup_count,
    l.lootbox_pickup_count,

    -- labels
    lab.target_defined,
    lab.hazard_next,
    lab.survives_next_phase,
    lab.final_rank,
    lab.final_rank_norm,
    lab.final_top3,
    lab.final_top5,
    lab.final_top10,
    lab.final_winner,
    lab.last_phase_alive
FROM stdb.teams t
JOIN stdb.matches m
  ON m.match_id = t.match_id
JOIN stdb.v_match_phase_bounds b
  ON b.match_id = t.match_id
LEFT JOIN stdb.match_phase_global g
  ON g.match_id = t.match_id AND g.phase_idx = b.phase_idx
LEFT JOIN stdb.team_phase_state s
  ON s.match_id = t.match_id AND s.team_id = t.team_id AND s.phase_idx = b.phase_idx
LEFT JOIN stdb.team_phase_trajectory tr
  ON tr.match_id = t.match_id AND tr.team_id = t.team_id AND tr.phase_idx = b.phase_idx
LEFT JOIN stdb.team_phase_exposure ex
  ON ex.match_id = t.match_id AND ex.team_id = t.team_id AND ex.phase_idx = b.phase_idx
LEFT JOIN stdb.team_phase_combat c
  ON c.match_id = t.match_id AND c.team_id = t.team_id AND c.phase_idx = b.phase_idx
LEFT JOIN stdb.team_phase_loot l
  ON l.match_id = t.match_id AND l.team_id = t.team_id AND l.phase_idx = b.phase_idx
LEFT JOIN stdb.team_phase_labels lab
  ON lab.match_id = t.match_id AND lab.team_id = t.team_id AND lab.cut_phase = b.phase_idx;

-- =============================================================================
-- End of RQ1 feature layer
-- =============================================================================
