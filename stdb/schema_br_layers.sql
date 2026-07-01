-- stdb/schema_br_layers.sql
-- =============================================================================
-- Battle Royale cross-title data layers.
--
-- This schema is intentionally additive.  It does not replace pubg.* or stdb.*;
-- it registers them as source/ground layers and adds a game-agnostic canonical
-- surface for future adapters such as Activision's Caldera data set.
--
-- Layer contract:
--   L0 br_raw     : byte/file provenance and source registries only
--   L1 br_ground  : source-faithful canonical facts, no research feature edits
--   L2 br_repr    : research representation, features, windows, tensor manifests
--   L3 br_exp     : cohorts, splits, experiment populations
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS postgis;

CREATE SCHEMA IF NOT EXISTS br_raw;
CREATE SCHEMA IF NOT EXISTS br_ground;
CREATE SCHEMA IF NOT EXISTS br_repr;
CREATE SCHEMA IF NOT EXISTS br_exp;

-- -----------------------------------------------------------------------------
-- L0: raw/source archive registry.  Raw bytes stay on disk/object storage; the DB
-- stores where they are, how large they are, and how to verify them.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS br_raw.sources (
    source_id       TEXT PRIMARY KEY,
    source_game     TEXT NOT NULL,
    provider        TEXT NOT NULL,
    canonical_url   TEXT,
    license_name    TEXT,
    license_url     TEXT,
    observability   TEXT NOT NULL,
    notes           JSONB NOT NULL DEFAULT '{}'::jsonb,
    registered_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS br_raw.artifacts (
    artifact_id     TEXT PRIMARY KEY,
    source_id       TEXT NOT NULL REFERENCES br_raw.sources(source_id),
    layer_name      TEXT NOT NULL CHECK (layer_name IN ('L0','L1','L2','L3')),
    artifact_kind   TEXT NOT NULL,
    uri             TEXT NOT NULL,
    local_path      TEXT,
    content_format  TEXT,
    size_bytes      BIGINT,
    sha256          TEXT,
    git_ref         TEXT,
    git_commit      TEXT,
    row_count       BIGINT,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    registered_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_br_raw_artifacts_source
    ON br_raw.artifacts (source_id, layer_name, artifact_kind);

-- Current PUBG raw salvage audit: pubg.* is L1-normalized telemetry, not a
-- byte-identical JSON archive.  Keep table-level coverage explicit.
CREATE TABLE IF NOT EXISTS br_raw.l1_table_registry (
    source_id                  TEXT NOT NULL REFERENCES br_raw.sources(source_id),
    source_table               TEXT NOT NULL,
    normalized_event_kind      TEXT,
    rows_estimate              BIGINT,
    distinct_matches_estimate  BIGINT,
    consistency_level          TEXT NOT NULL CHECK (
        consistency_level IN ('raw_json','normalized_raw','derived','unknown')
    ),
    missing_from_official_raw  TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    notes                      TEXT,
    audited_at                 TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (source_id, source_table)
);

-- -----------------------------------------------------------------------------
-- L1: canonical ground facts.  These tables should be source-faithful and sparse:
-- if a title does not expose a field, leave it NULL and record observability.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS br_ground.matches (
    br_match_id            TEXT PRIMARY KEY,
    source_id              TEXT NOT NULL REFERENCES br_raw.sources(source_id),
    source_match_id        TEXT NOT NULL,
    source_game            TEXT NOT NULL,
    map_name               TEXT,
    mode_name              TEXT,
    perspective            TEXT,
    team_size              TEXT,
    started_at             TIMESTAMPTZ,
    duration_s             REAL,
    coordinate_unit        TEXT NOT NULL DEFAULT 'meter',
    coordinate_scale_to_m  REAL NOT NULL DEFAULT 1.0,
    observability          TEXT NOT NULL,
    raw_artifact_id        TEXT REFERENCES br_raw.artifacts(artifact_id),
    metadata               JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (source_id, source_match_id)
);
CREATE INDEX IF NOT EXISTS ix_br_ground_matches_domain
    ON br_ground.matches (source_game, map_name, perspective, team_size);

CREATE TABLE IF NOT EXISTS br_ground.agents (
    br_agent_id      TEXT PRIMARY KEY,
    br_match_id      TEXT NOT NULL REFERENCES br_ground.matches(br_match_id) ON DELETE CASCADE,
    source_agent_id  TEXT NOT NULL,
    agent_kind       TEXT NOT NULL CHECK (agent_kind IN ('player','team','squad','unknown')),
    team_id          TEXT,
    is_observed      BOOLEAN NOT NULL DEFAULT TRUE,
    metadata         JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (br_match_id, source_agent_id, agent_kind)
);
CREATE INDEX IF NOT EXISTS ix_br_ground_agents_match
    ON br_ground.agents (br_match_id, agent_kind, team_id);

CREATE TABLE IF NOT EXISTS br_ground.trajectory_samples (
    br_match_id          TEXT NOT NULL REFERENCES br_ground.matches(br_match_id) ON DELETE CASCADE,
    br_agent_id          TEXT NOT NULL REFERENCES br_ground.agents(br_agent_id) ON DELETE CASCADE,
    sample_t             REAL NOT NULL,
    sample_kind          TEXT NOT NULL DEFAULT 'observed',
    pos                  geometry(PointZ, 0),
    pos_x_m              REAL,
    pos_y_m              REAL,
    pos_z_m              REAL,
    health               REAL,
    alive_state          TEXT,
    in_vehicle           BOOLEAN,
    source_artifact_id   TEXT REFERENCES br_raw.artifacts(artifact_id),
    metadata             JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (br_match_id, br_agent_id, sample_t, sample_kind)
);
CREATE INDEX IF NOT EXISTS ix_br_ground_traj_match_t
    ON br_ground.trajectory_samples (br_match_id, sample_t);
CREATE INDEX IF NOT EXISTS ix_br_ground_traj_gist
    ON br_ground.trajectory_samples USING GIST (pos);

CREATE TABLE IF NOT EXISTS br_ground.zone_states (
    br_match_id          TEXT NOT NULL REFERENCES br_ground.matches(br_match_id) ON DELETE CASCADE,
    zone_kind            TEXT NOT NULL,
    sample_t             REAL NOT NULL,
    phase_idx            INTEGER,
    center_pos           geometry(PointZ, 0),
    radius_m             REAL,
    region               geometry(Polygon, 0),
    source_artifact_id   TEXT REFERENCES br_raw.artifacts(artifact_id),
    metadata             JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (br_match_id, zone_kind, sample_t)
);
CREATE INDEX IF NOT EXISTS ix_br_ground_zone_match_phase
    ON br_ground.zone_states (br_match_id, phase_idx, zone_kind);

CREATE TABLE IF NOT EXISTS br_ground.events (
    event_id             BIGSERIAL PRIMARY KEY,
    br_match_id          TEXT NOT NULL REFERENCES br_ground.matches(br_match_id) ON DELETE CASCADE,
    event_kind           TEXT NOT NULL,
    event_t              REAL,
    actor_agent_id       TEXT REFERENCES br_ground.agents(br_agent_id),
    target_agent_id      TEXT REFERENCES br_ground.agents(br_agent_id),
    actor_pos            geometry(PointZ, 0),
    target_pos           geometry(PointZ, 0),
    value                REAL,
    source_artifact_id   TEXT REFERENCES br_raw.artifacts(artifact_id),
    metadata             JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS ix_br_ground_events_match_kind_t
    ON br_ground.events (br_match_id, event_kind, event_t);

CREATE TABLE IF NOT EXISTS br_ground.outcomes (
    br_match_id           TEXT NOT NULL REFERENCES br_ground.matches(br_match_id) ON DELETE CASCADE,
    br_agent_id           TEXT NOT NULL REFERENCES br_ground.agents(br_agent_id) ON DELETE CASCADE,
    final_rank            INTEGER,
    survived              BOOLEAN,
    elimination_t         REAL,
    endpoint_pos          geometry(PointZ, 0),
    source_artifact_id    TEXT REFERENCES br_raw.artifacts(artifact_id),
    metadata              JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (br_match_id, br_agent_id)
);

-- Anonymous endpoint point clouds are L1 ground observations but may not expose
-- match/player identity.  Caldera endpoints use this table.
CREATE TABLE IF NOT EXISTS br_ground.endpoint_samples (
    source_id             TEXT NOT NULL REFERENCES br_raw.sources(source_id),
    endpoint_idx          INTEGER NOT NULL,
    sample_t              REAL,
    pos                   geometry(PointZ, 0),
    pos_x_m               REAL,
    pos_y_m               REAL,
    pos_z_m               REAL,
    color_r               REAL,
    color_g               REAL,
    color_b               REAL,
    source_artifact_id    TEXT REFERENCES br_raw.artifacts(artifact_id),
    metadata              JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (source_id, endpoint_idx)
);
CREATE INDEX IF NOT EXISTS ix_br_ground_endpoint_t
    ON br_ground.endpoint_samples (source_id, sample_t);
CREATE INDEX IF NOT EXISTS ix_br_ground_endpoint_gist
    ON br_ground.endpoint_samples USING GIST (pos);

-- Source-faithful environment/geometry catalog.  This does not store raw mesh
-- vertices or OpenUSD blobs; it records the scene graph surface needed to build
-- spatial indexes and derived geometry features later.
CREATE TABLE IF NOT EXISTS br_ground.environment_prims (
    source_id                    TEXT NOT NULL REFERENCES br_raw.sources(source_id),
    environment_id               TEXT NOT NULL,
    prim_path                    TEXT NOT NULL,
    source_artifact_id           TEXT REFERENCES br_raw.artifacts(artifact_id),
    stage_path                   TEXT,
    parent_path                  TEXT,
    prim_name                    TEXT,
    prim_type                    TEXT,
    kind                         TEXT,
    purpose                      TEXT,
    visibility                   TEXT,
    is_active                    BOOLEAN,
    is_loaded                    BOOLEAN,
    is_instance                  BOOLEAN,
    is_instanceable              BOOLEAN,
    is_model                     BOOLEAN,
    has_authored_references      BOOLEAN NOT NULL DEFAULT FALSE,
    has_authored_payloads        BOOLEAN NOT NULL DEFAULT FALSE,
    authored_reference_assets    TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    authored_payload_assets      TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    variant_sets                 JSONB NOT NULL DEFAULT '{}'::jsonb,
    attribute_names              TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    bbox_min_x_m                 REAL,
    bbox_min_y_m                 REAL,
    bbox_min_z_m                 REAL,
    bbox_max_x_m                 REAL,
    bbox_max_y_m                 REAL,
    bbox_max_z_m                 REAL,
    centroid                     geometry(PointZ, 0),
    metadata                     JSONB NOT NULL DEFAULT '{}'::jsonb,
    cataloged_at                 TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (source_id, environment_id, prim_path)
);
CREATE INDEX IF NOT EXISTS ix_br_ground_env_prims_type
    ON br_ground.environment_prims (source_id, environment_id, prim_type);
CREATE INDEX IF NOT EXISTS ix_br_ground_env_prims_parent
    ON br_ground.environment_prims (source_id, environment_id, parent_path);
CREATE INDEX IF NOT EXISTS ix_br_ground_env_prims_centroid_gist
    ON br_ground.environment_prims USING GIST (centroid);

-- -----------------------------------------------------------------------------
-- L2: research representation.  Keep heavy tensors/files external when possible;
-- the DB records feature contracts, windows, and artifact pointers.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS br_repr.feature_sets (
    feature_set_id    TEXT PRIMARY KEY,
    description       TEXT,
    source_id         TEXT REFERENCES br_raw.sources(source_id),
    manifest_path     TEXT,
    manifest_sha256   TEXT,
    metadata          JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS br_repr.observation_windows (
    window_id       TEXT PRIMARY KEY,
    br_match_id     TEXT NOT NULL REFERENCES br_ground.matches(br_match_id) ON DELETE CASCADE,
    clock           TEXT NOT NULL CHECK (clock IN ('snapshot','phase','match')),
    t_start         REAL,
    t_end           REAL,
    phase_idx       INTEGER,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (br_match_id, clock, t_start, t_end, phase_idx)
);

CREATE TABLE IF NOT EXISTS br_repr.tensor_artifacts (
    tensor_id        TEXT PRIMARY KEY,
    feature_set_id   TEXT NOT NULL REFERENCES br_repr.feature_sets(feature_set_id),
    artifact_id      TEXT REFERENCES br_raw.artifacts(artifact_id),
    uri              TEXT NOT NULL,
    local_path       TEXT,
    tensor_kind      TEXT NOT NULL,
    n_matches        INTEGER,
    n_agents_max     INTEGER,
    n_steps          INTEGER,
    n_features       INTEGER,
    metadata         JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS br_repr.trajectory_summaries (
    br_match_id        TEXT NOT NULL REFERENCES br_ground.matches(br_match_id) ON DELETE CASCADE,
    br_agent_id        TEXT NOT NULL REFERENCES br_ground.agents(br_agent_id) ON DELETE CASCADE,
    feature_set_id     TEXT REFERENCES br_repr.feature_sets(feature_set_id),
    n_samples          INTEGER NOT NULL,
    t_start_s          REAL,
    t_end_s            REAL,
    duration_s         REAL,
    path_length_m      REAL,
    displacement_m     REAL,
    sinuosity          REAL,
    endpoint_pos       geometry(PointZ, 0),
    metadata           JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (br_match_id, br_agent_id)
);
CREATE INDEX IF NOT EXISTS ix_br_repr_traj_feature_set
    ON br_repr.trajectory_summaries (feature_set_id);

CREATE TABLE IF NOT EXISTS br_repr.environment_catalog_runs (
    catalog_run_id        TEXT PRIMARY KEY,
    source_id             TEXT NOT NULL REFERENCES br_raw.sources(source_id),
    environment_id        TEXT NOT NULL,
    stage_path            TEXT NOT NULL,
    load_policy           TEXT NOT NULL,
    prim_count            INTEGER NOT NULL,
    type_counts           JSONB NOT NULL DEFAULT '{}'::jsonb,
    reference_asset_count INTEGER,
    payload_asset_count   INTEGER,
    metadata              JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_br_repr_env_catalog_runs_source
    ON br_repr.environment_catalog_runs (source_id, environment_id, created_at);

-- -----------------------------------------------------------------------------
-- L3: cohort/split/experiment population registry.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS br_exp.cohorts (
    cohort_id       TEXT PRIMARY KEY,
    source_scope    TEXT NOT NULL,
    feature_set_id  TEXT REFERENCES br_repr.feature_sets(feature_set_id),
    definition      JSONB NOT NULL,
    n_matches       INTEGER,
    ids_sha1        TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS br_exp.cohort_members (
    cohort_id       TEXT NOT NULL REFERENCES br_exp.cohorts(cohort_id) ON DELETE CASCADE,
    br_match_id     TEXT NOT NULL REFERENCES br_ground.matches(br_match_id) ON DELETE CASCADE,
    domain_key      TEXT,
    split_name      TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (cohort_id, br_match_id)
);
CREATE INDEX IF NOT EXISTS ix_br_exp_members_domain
    ON br_exp.cohort_members (cohort_id, domain_key, split_name);

-- -----------------------------------------------------------------------------
-- Non-destructive registration of known sources.
-- -----------------------------------------------------------------------------
INSERT INTO br_raw.sources (
    source_id, source_game, provider, canonical_url, license_name,
    license_url, observability, notes
) VALUES
(
    'pubg_current_l1',
    'PUBG: Battlegrounds',
    'KRAFTON / PUBG API',
    'https://documentation.pubg.com/',
    NULL,
    NULL,
    'full_normalized_telemetry_l1',
    '{"local_schema":"pubg","ground_role":"soil/raw-normalized layer; preserve as source of truth"}'::jsonb
),
(
    'activision_caldera',
    'Call of Duty: Warzone Caldera',
    'Activision',
    'https://github.com/Activision/caldera',
    'Activision Non-Commercial Use License Agreement',
    'https://github.com/Activision/caldera/blob/main/LICENSE.md',
    'geometry_plus_anonymized_player_point_clouds',
    '{"blog":"https://blog.activision.com/activision/2024/activision-releases-call-of-duty-warzone-caldera-data-set"}'::jsonb
)
ON CONFLICT (source_id) DO UPDATE
SET source_game = EXCLUDED.source_game,
    provider = EXCLUDED.provider,
    canonical_url = EXCLUDED.canonical_url,
    license_name = EXCLUDED.license_name,
    license_url = EXCLUDED.license_url,
    observability = EXCLUDED.observability,
    notes = br_raw.sources.notes || EXCLUDED.notes;

-- View over the existing PUBG L1 match surface.  It makes the current DB usable
-- as BR ground without copying rows.  This intentionally reads pubg.matches
-- directly instead of stdb.matches so the "soil" layer is the normalized raw
-- source, not the already-derived research subset.
CREATE OR REPLACE VIEW br_ground.v_pubg_matches_from_existing_l1 AS
WITH roster_counts AS (
    SELECT match_id, count(*)::INTEGER AS num_teams
      FROM pubg.rosters
     GROUP BY match_id
),
participant_counts AS (
    SELECT match_id, count(*)::INTEGER AS num_players
      FROM pubg.participants
     GROUP BY match_id
)
SELECT
    ('pubg:' || m.match_id) AS br_match_id,
    'pubg_current_l1'::TEXT AS source_id,
    m.match_id AS source_match_id,
    'PUBG: Battlegrounds'::TEXT AS source_game,
    m.map_name,
    m.game_mode AS mode_name,
    CASE WHEN m.game_mode LIKE '%-fpp' THEN 'FPP' ELSE 'TPP' END AS perspective,
    CASE
        WHEN replace(m.game_mode, '-fpp', '') IN ('solo','duo','squad')
            THEN replace(m.game_mode, '-fpp', '')
        ELSE NULL
    END AS team_size,
    COALESCE(ms.event_time, m.created_at) AS started_at,
    m.duration::REAL AS duration_s,
    'meter'::TEXT AS coordinate_unit,
    1.0::REAL AS coordinate_scale_to_m,
    'full_normalized_telemetry_l1'::TEXT AS observability,
    NULL::TEXT AS raw_artifact_id,
    jsonb_build_object(
        'collection_telemetry_fetched', c.telemetry_fetched,
        'stdb_materialized', sm.match_id IS NOT NULL,
        'num_teams', rc.num_teams,
        'num_players', pc.num_players,
        'is_custom', m.is_custom
    ) AS metadata
FROM pubg.matches m
LEFT JOIN pubg.collection_log c ON c.match_id = m.match_id
LEFT JOIN pubg.telem_match_start ms ON ms.match_id = m.match_id
LEFT JOIN stdb.matches sm ON sm.match_id = m.match_id
LEFT JOIN roster_counts rc ON rc.match_id = m.match_id
LEFT JOIN participant_counts pc ON pc.match_id = m.match_id;

-- =============================================================================
-- End of BR layer schema
-- =============================================================================
