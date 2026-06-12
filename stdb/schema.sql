-- stdb/schema.sql
-- =============================================================================
-- PUBG STDB (Spatio-Temporal Database) Schema
-- Implements the abstraction defined in 백성은 (2026) §연구방법:
--   1) 이동 객체 (moving objects)         → stdb.moving_objects
--   2) 동적 공간 영역 (dynamic regions)   → stdb.dynamic_regions
--   3) 공간 사건 (spatial events)         → stdb.spatial_events
--
-- Source (read-only):  pubg.* schema (raw KRAFTON telemetry)
-- Target:              stdb.* schema (this file)
--
-- Coordinate system:
--   PUBG telemetry stores positions in centimeters (numeric) without SRID.
--   ETL converts to METERS (real) before insertion. SRID 0 is used for the
--   `geometry` columns so PostGIS spatial indexes work but no datum reprojection
--   is attempted (the game world is a Cartesian plane).
--
-- Apply with:  psycopg2 → cursor.execute(open('stdb/schema.sql').read())
--              (or psql -f stdb/schema.sql if libpq is installed)
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS postgis;

CREATE SCHEMA IF NOT EXISTS stdb;

-- -----------------------------------------------------------------------------
-- 1) Match meta
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stdb.matches (
    match_id      TEXT PRIMARY KEY,
    started_at    TIMESTAMPTZ NOT NULL,
    duration_s    INTEGER     NOT NULL,
    map_name      TEXT        NOT NULL,
    game_mode     TEXT        NOT NULL,
    team_size     INTEGER,
    is_custom     BOOLEAN     NOT NULL DEFAULT FALSE,
    num_teams     INTEGER,
    num_players   INTEGER
);
CREATE INDEX IF NOT EXISTS ix_matches_map_mode ON stdb.matches (map_name, game_mode);
CREATE INDEX IF NOT EXISTS ix_matches_started_at ON stdb.matches (started_at);

-- -----------------------------------------------------------------------------
-- 2) Team meta + final-rank labels (RQ3)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stdb.teams (
    match_id     TEXT    NOT NULL REFERENCES stdb.matches(match_id) ON DELETE CASCADE,
    team_id      INTEGER NOT NULL,
    final_rank   INTEGER,
    num_players  INTEGER,
    won          BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (match_id, team_id)
);
CREATE INDEX IF NOT EXISTS ix_teams_rank ON stdb.teams (match_id, final_rank);

-- -----------------------------------------------------------------------------
-- 3) Moving objects  (보고서 §STDB-1)
--    Both player-level and team-level rows live in this table; agent_kind
--    discriminates. `snapshot_t` is the elapsed game seconds (10s grid).
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stdb.moving_objects (
    match_id     TEXT       NOT NULL,
    agent_kind   TEXT       NOT NULL CHECK (agent_kind IN ('player','team')),
    agent_id     TEXT       NOT NULL,            -- player: account_id; team: team_id::text
    team_id      INTEGER    NOT NULL,
    snapshot_t   INTEGER    NOT NULL,            -- elapsed seconds, 10s grid
    pos          geometry(PointZ, 0) NOT NULL,   -- (x,y,z) in meters
    health       REAL,                           -- 0..100 (NULL for team rows)
    in_vehicle   BOOLEAN    NOT NULL DEFAULT FALSE,
    is_alive     BOOLEAN    NOT NULL DEFAULT TRUE,
    PRIMARY KEY (match_id, agent_kind, agent_id, snapshot_t)
);
CREATE INDEX IF NOT EXISTS ix_mo_gist
    ON stdb.moving_objects USING GIST (pos);
CREATE INDEX IF NOT EXISTS ix_mo_brin
    ON stdb.moving_objects USING BRIN (match_id, snapshot_t);
CREATE INDEX IF NOT EXISTS ix_mo_kind_t
    ON stdb.moving_objects (match_id, agent_kind, snapshot_t);
CREATE INDEX IF NOT EXISTS ix_mo_team
    ON stdb.moving_objects (match_id, team_id, snapshot_t);

-- -----------------------------------------------------------------------------
-- 4) Dynamic spatial regions  (보고서 §STDB-2)
--    Safe / poison zones recorded per snapshot.  Polygon is buffered from the
--    2D centre with `radius_m`.  v1 ignores red_zone / black_zone.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stdb.dynamic_regions (
    match_id     TEXT       NOT NULL,
    region_kind  TEXT       NOT NULL CHECK (region_kind IN ('safe','poison')),
    snapshot_t   INTEGER    NOT NULL,
    center_pos   geometry(PointZ, 0) NOT NULL,   -- (cx, cy, cz) in meters
    radius_m     REAL       NOT NULL,
    region       geometry(Polygon, 0),           -- ST_Buffer(ST_Force2D(center), radius, 64)
    phase_idx    INTEGER,                        -- 0,1,2,... (from telem_phase_changes)
    PRIMARY KEY (match_id, region_kind, snapshot_t)
);
CREATE INDEX IF NOT EXISTS ix_dr_gist
    ON stdb.dynamic_regions USING GIST (region);
CREATE INDEX IF NOT EXISTS ix_dr_brin
    ON stdb.dynamic_regions USING BRIN (match_id, snapshot_t);
CREATE INDEX IF NOT EXISTS ix_dr_phase
    ON stdb.dynamic_regions (match_id, region_kind, phase_idx);

-- -----------------------------------------------------------------------------
-- 5) Spatial events  (보고서 §STDB-3)
--    kill / damage / groggy / landing — all rows from pubg.telem_* whose
--    payload includes attacker+victim coordinates (no separate position lookup
--    needed — confirmed via audit_raw.py).
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stdb.spatial_events (
    event_id        BIGSERIAL PRIMARY KEY,
    match_id        TEXT       NOT NULL,
    event_kind      TEXT       NOT NULL CHECK (event_kind IN ('kill','damage','groggy','landing','phase')),
    event_t         REAL       NOT NULL,         -- elapsed seconds (subseconds from event_time)
    attacker_id     TEXT,
    attacker_team   INTEGER,
    attacker_pos    geometry(PointZ, 0),         -- (x,y,z) in meters; nullable for 'phase'
    victim_id       TEXT,
    victim_team     INTEGER,
    victim_pos      geometry(PointZ, 0),
    damage          REAL,                        -- only for kind='damage'
    damage_type     TEXT,
    damage_causer   TEXT,
    distance_m      REAL,
    extra           JSONB
);
CREATE INDEX IF NOT EXISTS ix_se_match_t
    ON stdb.spatial_events (match_id, event_t);
CREATE INDEX IF NOT EXISTS ix_se_match_kind_t
    ON stdb.spatial_events (match_id, event_kind, event_t);
CREATE INDEX IF NOT EXISTS ix_se_attacker_gist
    ON stdb.spatial_events USING GIST (attacker_pos);
CREATE INDEX IF NOT EXISTS ix_se_victim_gist
    ON stdb.spatial_events USING GIST (victim_pos);
CREATE INDEX IF NOT EXISTS ix_se_victim_team
    ON stdb.spatial_events (match_id, victim_team);

-- -----------------------------------------------------------------------------
-- 6) Team phase-level survival-environment features  (보고서 §연구방법-3)
--    Computed by stdb/etl.py after moving_objects + dynamic_regions +
--    spatial_events are populated.  One row per (match, team, phase).
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stdb.team_phase_features (
    match_id                   TEXT    NOT NULL,
    team_id                    INTEGER NOT NULL,
    phase_idx                  INTEGER NOT NULL,
    mean_dist_to_boundary_m    REAL,
    enemy_density_100m         REAL,
    recent_combat_exposure     REAL,
    position_stability         REAL,
    inside_safe_ratio          REAL,
    PRIMARY KEY (match_id, team_id, phase_idx)
);
CREATE INDEX IF NOT EXISTS ix_tpf_team
    ON stdb.team_phase_features (match_id, team_id);

-- -----------------------------------------------------------------------------
-- 7) Time-varying team adjacency  (ST-GCN A_t cache)
--    Per snapshot, pairwise distance + recent-encounter flag.  Recomputed by
--    stdb/etl.py from moving_objects + spatial_events.  Self-loops excluded.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stdb.team_proximity (
    match_id              TEXT    NOT NULL,
    snapshot_t            INTEGER NOT NULL,
    team_a                INTEGER NOT NULL,
    team_b                INTEGER NOT NULL,
    distance_m            REAL    NOT NULL,
    encounter_recently    BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (match_id, snapshot_t, team_a, team_b),
    CHECK (team_a < team_b)
);
CREATE INDEX IF NOT EXISTS ix_tp_brin
    ON stdb.team_proximity USING BRIN (match_id, snapshot_t);

-- -----------------------------------------------------------------------------
-- View: convenience for `is_game` filter (drop pre-game / lobby positions).
-- pubg.telem_positions.is_game is numeric:
--   0.0 = lobby, 0.1 = parachute / pre-engagement, 1.0 = in-game.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW stdb.v_positions_ingame AS
SELECT *
  FROM pubg.telem_positions
 WHERE is_game >= 0.1
   AND account_id IS NOT NULL
   AND pos_x IS NOT NULL
   AND pos_y IS NOT NULL
   AND pos_z IS NOT NULL;

-- =============================================================================
-- End of stdb schema
-- =============================================================================
