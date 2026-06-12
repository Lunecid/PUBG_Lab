"""
stdb/rq1_feature_etl.py
=======================
Build the redesigned RQ1 phase-level feature layer from the raw ``pubg.*``
telemetry already stored in PostgreSQL.

This script is intentionally additive: it reads the existing base STDB tables
(``matches``, ``teams``, ``moving_objects``, ``dynamic_regions``,
``spatial_events``) and raw item/position/game-state telemetry, then fills the
RQ1 tables declared in ``stdb/schema_rq1.sql``.

Usage examples:
    python -m stdb.rq1_feature_etl --apply-schema --limit 1
    python -m stdb.rq1_feature_etl --apply-schema --all-current
    python -m stdb.rq1_feature_etl --match-id <uuid> --verify
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Iterable

from stdb.conn import apply_sql_file, connect

logger = logging.getLogger("stdb.rq1_feature_etl")

SCHEMA_PATH = Path(__file__).with_name("schema_rq1.sql")
ARENA_SIZE_M = 8160.0


def _execute(cur, sql: str, params: tuple = ()) -> None:
    cur.execute(sql, params)


def list_current_match_ids(
    *, map_name: str | None = None, mode: str | None = None, limit: int | None = None
) -> list[str]:
    conn = connect()
    cur = conn.cursor()
    try:
        sql = """
            SELECT match_id
              FROM stdb.matches
             WHERE 1=1
        """
        params: list[object] = []
        if map_name:
            sql += " AND map_name = %s"
            params.append(map_name)
        if mode:
            sql += " AND game_mode = %s"
            params.append(mode)
        sql += " ORDER BY started_at, match_id"
        if limit:
            sql += " LIMIT %s"
            params.append(limit)
        cur.execute(sql, params)
        return [r[0] for r in cur.fetchall()]
    finally:
        cur.close()
        conn.close()


def build_match_phase_global(cur, match_id: str) -> int:
    _execute(cur, "DELETE FROM stdb.match_phase_global WHERE match_id = %s", (match_id,))
    _execute(
        cur,
        """
        WITH b AS (
            SELECT *
              FROM stdb.v_match_phase_bounds
             WHERE match_id = %s
        ),
        game AS (
            SELECT
                b.phase_idx,
                avg(gs.num_alive_teams)::real AS avg_alive_teams,
                (array_agg(gs.num_alive_teams ORDER BY gs.elapsed_time))[1] AS start_alive_teams,
                (array_agg(gs.num_alive_teams ORDER BY gs.elapsed_time DESC))[1] AS end_alive_teams,
                avg(gs.num_alive_players)::real AS avg_alive_players,
                (array_agg(gs.num_alive_players ORDER BY gs.elapsed_time))[1] AS start_alive_players,
                (array_agg(gs.num_alive_players ORDER BY gs.elapsed_time DESC))[1] AS end_alive_players,
                avg(CASE WHEN coalesce(gs.red_zone_radius, 0) > 0 THEN 1.0 ELSE 0.0 END)::real
                    AS red_zone_active_ratio,
                avg(CASE WHEN coalesce(gs.black_zone_radius, 0) > 0 THEN 1.0 ELSE 0.0 END)::real
                    AS black_zone_active_ratio
              FROM b
              JOIN pubg.telem_game_states gs
                ON gs.match_id = b.match_id
               AND gs.elapsed_time BETWEEN b.phase_start_s AND b.phase_end_s
             GROUP BY b.phase_idx
        ),
        safe_start AS (
            SELECT DISTINCT ON (phase_idx)
                   phase_idx, center_pos, radius_m
              FROM stdb.dynamic_regions
             WHERE match_id = %s AND region_kind = 'safe' AND phase_idx IS NOT NULL
             ORDER BY phase_idx, snapshot_t
        ),
        safe_end AS (
            SELECT DISTINCT ON (phase_idx)
                   phase_idx, center_pos, radius_m
              FROM stdb.dynamic_regions
             WHERE match_id = %s AND region_kind = 'safe' AND phase_idx IS NOT NULL
             ORDER BY phase_idx, snapshot_t DESC
        ),
        poison_start AS (
            SELECT DISTINCT ON (phase_idx)
                   phase_idx, center_pos, radius_m
              FROM stdb.dynamic_regions
             WHERE match_id = %s AND region_kind = 'poison' AND phase_idx IS NOT NULL
             ORDER BY phase_idx, snapshot_t
        ),
        poison_end AS (
            SELECT DISTINCT ON (phase_idx)
                   phase_idx, center_pos, radius_m
              FROM stdb.dynamic_regions
             WHERE match_id = %s AND region_kind = 'poison' AND phase_idx IS NOT NULL
             ORDER BY phase_idx, snapshot_t DESC
        ),
        match_start AS (
            SELECT match_id, weather_id::text AS weather_id, camera_view_type::text AS camera_view_type,
                   to_jsonb(blue_zone_options) AS blue_zone_options
              FROM pubg.telem_match_start
             WHERE match_id = %s
             LIMIT 1
        )
        INSERT INTO stdb.match_phase_global (
            match_id, phase_idx, phase_start_s, phase_end_s, phase_duration_s,
            avg_alive_teams, start_alive_teams, end_alive_teams,
            avg_alive_players, start_alive_players, end_alive_players,
            safe_radius_start_m, safe_radius_end_m, safe_radius_delta_m,
            safe_center_shift_m, poison_radius_start_m, poison_radius_end_m,
            poison_radius_delta_m, zone_gap_start_m, zone_gap_end_m,
            safe_area_ratio_start, safe_area_ratio_end,
            red_zone_active_ratio, black_zone_active_ratio,
            weather_id, camera_view_type, blue_zone_options
        )
        SELECT
            b.match_id,
            b.phase_idx,
            b.phase_start_s,
            b.phase_end_s,
            b.phase_duration_s,
            g.avg_alive_teams,
            g.start_alive_teams,
            g.end_alive_teams,
            g.avg_alive_players,
            g.start_alive_players,
            g.end_alive_players,
            ss.radius_m AS safe_radius_start_m,
            se.radius_m AS safe_radius_end_m,
            (se.radius_m - ss.radius_m)::real AS safe_radius_delta_m,
            ST_Distance(ST_Force2D(ss.center_pos), ST_Force2D(se.center_pos))::real
                AS safe_center_shift_m,
            ps.radius_m AS poison_radius_start_m,
            pe.radius_m AS poison_radius_end_m,
            (pe.radius_m - ps.radius_m)::real AS poison_radius_delta_m,
            (ps.radius_m - ss.radius_m)::real AS zone_gap_start_m,
            (pe.radius_m - se.radius_m)::real AS zone_gap_end_m,
            (pi() * ss.radius_m * ss.radius_m / (%s * %s))::real AS safe_area_ratio_start,
            (pi() * se.radius_m * se.radius_m / (%s * %s))::real AS safe_area_ratio_end,
            g.red_zone_active_ratio,
            g.black_zone_active_ratio,
            ms.weather_id,
            ms.camera_view_type,
            ms.blue_zone_options
          FROM b
     LEFT JOIN game g ON g.phase_idx = b.phase_idx
     LEFT JOIN safe_start ss ON ss.phase_idx = b.phase_idx
     LEFT JOIN safe_end se ON se.phase_idx = b.phase_idx
     LEFT JOIN poison_start ps ON ps.phase_idx = b.phase_idx
     LEFT JOIN poison_end pe ON pe.phase_idx = b.phase_idx
     LEFT JOIN match_start ms ON ms.match_id = b.match_id
        """,
        (
            match_id,
            match_id,
            match_id,
            match_id,
            match_id,
            match_id,
            ARENA_SIZE_M,
            ARENA_SIZE_M,
            ARENA_SIZE_M,
            ARENA_SIZE_M,
        ),
    )
    cur.execute("SELECT count(*) FROM stdb.match_phase_global WHERE match_id = %s", (match_id,))
    return int(cur.fetchone()[0])


def build_team_phase_state(cur, match_id: str) -> int:
    _execute(cur, "DELETE FROM stdb.team_phase_state WHERE match_id = %s", (match_id,))
    _execute(
        cur,
        """
        WITH b AS (
            SELECT *
              FROM stdb.v_match_phase_bounds
             WHERE match_id = %s
        ),
        raw_pos AS (
            SELECT
                b.phase_idx,
                p.team_id,
                p.account_id,
                p.elapsed_time::integer AS elapsed_time,
                p.health::real AS health,
                (p.pos_x::real / 100.0) AS x_m,
                (p.pos_y::real / 100.0) AS y_m,
                (p.pos_z::real / 100.0) AS z_m,
                CASE
                    WHEN p.vehicle_type IS NOT NULL
                     AND p.vehicle_type NOT IN ('TransportAircraft', 'DummyTransportAircraft_C')
                    THEN TRUE ELSE FALSE END AS in_vehicle,
                p.vehicle_speed::real AS vehicle_speed
              FROM pubg.telem_positions p
              JOIN b
                ON b.match_id = p.match_id
               AND p.elapsed_time BETWEEN b.phase_start_s AND b.phase_end_s
             WHERE p.match_id = %s
               AND p.account_id IS NOT NULL
               AND p.team_id IS NOT NULL
               AND p.elapsed_time IS NOT NULL
               AND p.is_game >= 0.1
               AND p.pos_x IS NOT NULL
               AND p.pos_y IS NOT NULL
        ),
        snap AS (
            SELECT
                phase_idx,
                team_id,
                elapsed_time,
                count(DISTINCT account_id) FILTER (WHERE coalesce(health, 0) > 0) AS alive_players,
                avg(health) FILTER (WHERE coalesce(health, 0) > 0) AS mean_health,
                min(health) FILTER (WHERE coalesce(health, 0) > 0) AS min_health,
                avg(CASE WHEN in_vehicle THEN 1.0 ELSE 0.0 END) AS vehicle_ratio_snap,
                avg(vehicle_speed) FILTER (WHERE in_vehicle AND vehicle_speed IS NOT NULL) AS vehicle_speed_mean_snap,
                max(vehicle_speed) FILTER (WHERE in_vehicle AND vehicle_speed IS NOT NULL) AS vehicle_speed_max_snap
              FROM raw_pos
             GROUP BY phase_idx, team_id, elapsed_time
        ),
        centroids AS (
            SELECT phase_idx, team_id, elapsed_time, avg(x_m) AS cx, avg(y_m) AS cy
              FROM raw_pos
             WHERE coalesce(health, 0) > 0
             GROUP BY phase_idx, team_id, elapsed_time
        ),
        spread AS (
            SELECT
                p.phase_idx,
                p.team_id,
                p.elapsed_time,
                avg(sqrt(power(p.x_m - c.cx, 2) + power(p.y_m - c.cy, 2)))::real AS spread_avg_m,
                max(sqrt(power(p.x_m - c.cx, 2) + power(p.y_m - c.cy, 2)))::real AS spread_max_m
              FROM raw_pos p
              JOIN centroids c
                ON c.phase_idx = p.phase_idx
               AND c.team_id = p.team_id
               AND c.elapsed_time = p.elapsed_time
             WHERE coalesce(p.health, 0) > 0
             GROUP BY p.phase_idx, p.team_id, p.elapsed_time
        ),
        snap_join AS (
            SELECT
                s.phase_idx,
                s.team_id,
                s.elapsed_time,
                s.alive_players,
                s.mean_health,
                s.min_health,
                s.vehicle_ratio_snap,
                s.vehicle_speed_mean_snap,
                s.vehicle_speed_max_snap,
                sp.spread_avg_m,
                sp.spread_max_m
              FROM snap s
         LEFT JOIN spread sp
                ON sp.phase_idx = s.phase_idx
               AND sp.team_id = s.team_id
               AND sp.elapsed_time = s.elapsed_time
        ),
        agg AS (
            SELECT
                phase_idx,
                team_id,
                count(*)::integer AS position_rows,
                min(alive_players)::real AS alive_players_min,
                avg(alive_players)::real AS alive_players_avg,
                avg(mean_health)::real AS mean_health_avg,
                min(min_health)::real AS min_health_min,
                regr_slope(mean_health, elapsed_time)::real AS health_slope,
                avg(spread_avg_m)::real AS team_spread_avg_m,
                max(spread_max_m)::real AS team_spread_max_m,
                avg(vehicle_ratio_snap)::real AS vehicle_ratio,
                avg(vehicle_speed_mean_snap)::real AS vehicle_speed_mean,
                max(vehicle_speed_max_snap)::real AS vehicle_speed_max
              FROM snap_join
             GROUP BY phase_idx, team_id
        ),
        start_snap AS (
            SELECT DISTINCT ON (phase_idx, team_id)
                   phase_idx, team_id, alive_players AS alive_players_start
              FROM snap_join
             ORDER BY phase_idx, team_id, elapsed_time
        ),
        end_snap AS (
            SELECT DISTINCT ON (phase_idx, team_id)
                   phase_idx, team_id, alive_players AS alive_players_end
              FROM snap_join
             ORDER BY phase_idx, team_id, elapsed_time DESC
        )
        INSERT INTO stdb.team_phase_state (
            match_id, team_id, phase_idx, team_size, position_rows,
            alive_players_start, alive_players_end, alive_players_min, alive_players_avg,
            mean_health_avg, min_health_min, health_slope,
            team_spread_avg_m, team_spread_max_m,
            vehicle_ratio, vehicle_speed_mean, vehicle_speed_max
        )
        SELECT
            %s,
            t.team_id,
            b.phase_idx,
            t.num_players,
            coalesce(a.position_rows, 0),
            ss.alive_players_start,
            es.alive_players_end,
            a.alive_players_min,
            a.alive_players_avg,
            a.mean_health_avg,
            a.min_health_min,
            a.health_slope,
            a.team_spread_avg_m,
            a.team_spread_max_m,
            a.vehicle_ratio,
            a.vehicle_speed_mean,
            a.vehicle_speed_max
          FROM stdb.teams t
          JOIN b ON b.match_id = t.match_id
     LEFT JOIN agg a
            ON a.team_id = t.team_id AND a.phase_idx = b.phase_idx
     LEFT JOIN start_snap ss
            ON ss.team_id = t.team_id AND ss.phase_idx = b.phase_idx
     LEFT JOIN end_snap es
            ON es.team_id = t.team_id AND es.phase_idx = b.phase_idx
         WHERE t.match_id = %s
        """,
        (match_id, match_id, match_id, match_id),
    )
    cur.execute("SELECT count(*) FROM stdb.team_phase_state WHERE match_id = %s", (match_id,))
    return int(cur.fetchone()[0])


def build_team_phase_combat(cur, match_id: str) -> int:
    _execute(cur, "DELETE FROM stdb.team_phase_combat WHERE match_id = %s", (match_id,))
    _execute(
        cur,
        """
        WITH b AS (
            SELECT *
              FROM stdb.v_match_phase_bounds
             WHERE match_id = %s
        ),
        team_phase AS (
            SELECT t.match_id, t.team_id, b.phase_idx
              FROM stdb.teams t
              JOIN b ON b.match_id = t.match_id
             WHERE t.match_id = %s
        ),
        role_events AS (
            SELECT
                b.phase_idx,
                se.attacker_team AS team_id,
                se.victim_team AS opponent_team,
                'dealt'::text AS role,
                se.event_kind,
                se.event_t,
                coalesce(se.damage, 0)::real AS damage
              FROM stdb.spatial_events se
              JOIN b
                ON b.match_id = se.match_id
               AND se.event_t BETWEEN b.phase_start_s AND b.phase_end_s
             WHERE se.match_id = %s
               AND se.event_kind IN ('damage', 'groggy', 'kill')
               AND se.attacker_team IS NOT NULL
               AND se.victim_team IS NOT NULL
               AND se.attacker_team <> se.victim_team
            UNION ALL
            SELECT
                b.phase_idx,
                se.victim_team AS team_id,
                se.attacker_team AS opponent_team,
                'taken'::text AS role,
                se.event_kind,
                se.event_t,
                coalesce(se.damage, 0)::real AS damage
              FROM stdb.spatial_events se
              JOIN b
                ON b.match_id = se.match_id
               AND se.event_t BETWEEN b.phase_start_s AND b.phase_end_s
             WHERE se.match_id = %s
               AND se.event_kind IN ('damage', 'groggy', 'kill')
               AND se.attacker_team IS NOT NULL
               AND se.victim_team IS NOT NULL
               AND se.attacker_team <> se.victim_team
        ),
        agg AS (
            SELECT
                phase_idx,
                team_id,
                sum(damage) FILTER (WHERE role = 'dealt' AND event_kind = 'damage')::real AS damage_dealt,
                sum(damage) FILTER (WHERE role = 'taken' AND event_kind = 'damage')::real AS damage_taken,
                count(*) FILTER (WHERE role = 'dealt' AND event_kind = 'damage')::integer AS damage_events_dealt,
                count(*) FILTER (WHERE role = 'taken' AND event_kind = 'damage')::integer AS damage_events_taken,
                count(*) FILTER (WHERE role = 'dealt' AND event_kind = 'groggy')::integer AS groggy_dealt,
                count(*) FILTER (WHERE role = 'taken' AND event_kind = 'groggy')::integer AS groggy_taken,
                count(*) FILTER (WHERE role = 'dealt' AND event_kind = 'kill')::integer AS kills_dealt,
                count(*) FILTER (WHERE role = 'taken' AND event_kind = 'kill')::integer AS kills_taken,
                count(DISTINCT opponent_team) FILTER (WHERE role = 'dealt' AND event_kind = 'damage')::integer
                    AS distinct_opponents_damaged,
                count(DISTINCT opponent_team) FILTER (WHERE role = 'taken' AND event_kind = 'damage')::integer
                    AS distinct_opponents_damaged_by,
                count(DISTINCT opponent_team)::integer AS combat_partner_count,
                min(event_t) FILTER (WHERE role = 'dealt' AND event_kind = 'damage')::real AS first_damage_dealt_s,
                min(event_t) FILTER (WHERE role = 'taken' AND event_kind = 'damage')::real AS first_damage_taken_s
              FROM role_events
             GROUP BY phase_idx, team_id
        )
        INSERT INTO stdb.team_phase_combat (
            match_id, team_id, phase_idx,
            damage_dealt, damage_taken, damage_events_dealt, damage_events_taken,
            groggy_dealt, groggy_taken, kills_dealt, kills_taken,
            distinct_opponents_damaged, distinct_opponents_damaged_by,
            combat_partner_count, first_damage_dealt_s, first_damage_taken_s,
            combat_initiated, nearby_enemy_combat_events_200m,
            third_party_exposure_count
        )
        SELECT
            tp.match_id,
            tp.team_id,
            tp.phase_idx,
            coalesce(a.damage_dealt, 0),
            coalesce(a.damage_taken, 0),
            coalesce(a.damage_events_dealt, 0),
            coalesce(a.damage_events_taken, 0),
            coalesce(a.groggy_dealt, 0),
            coalesce(a.groggy_taken, 0),
            coalesce(a.kills_dealt, 0),
            coalesce(a.kills_taken, 0),
            coalesce(a.distinct_opponents_damaged, 0),
            coalesce(a.distinct_opponents_damaged_by, 0),
            coalesce(a.combat_partner_count, 0),
            a.first_damage_dealt_s,
            a.first_damage_taken_s,
            CASE
                WHEN a.first_damage_dealt_s IS NULL AND a.first_damage_taken_s IS NULL THEN NULL
                WHEN a.first_damage_taken_s IS NULL THEN TRUE
                WHEN a.first_damage_dealt_s IS NULL THEN FALSE
                ELSE a.first_damage_dealt_s <= a.first_damage_taken_s
            END AS combat_initiated,
            0,
            0
          FROM team_phase tp
     LEFT JOIN agg a
            ON a.team_id = tp.team_id AND a.phase_idx = tp.phase_idx
        """,
        (match_id, match_id, match_id, match_id),
    )
    cur.execute("SELECT count(*) FROM stdb.team_phase_combat WHERE match_id = %s", (match_id,))
    return int(cur.fetchone()[0])


def build_team_phase_loot(cur, match_id: str) -> int:
    _execute(cur, "DELETE FROM stdb.team_phase_loot WHERE match_id = %s", (match_id,))
    _execute(
        cur,
        """
        WITH b AS (
            SELECT *
              FROM stdb.v_match_phase_bounds
             WHERE match_id = %s
        ),
        m AS (
            SELECT match_id, started_at
              FROM stdb.matches
             WHERE match_id = %s
        ),
        team_phase AS (
            SELECT t.match_id, t.team_id, b.phase_idx
              FROM stdb.teams t
              JOIN b ON b.match_id = t.match_id
             WHERE t.match_id = %s
        ),
        item_events AS (
            SELECT p.match_id, 'pickup'::text AS event_kind, p.event_time,
                   p.team_id, p.account_id, p.item_id, p.category, p.sub_category,
                   coalesce(p.stack_count, 1)::integer AS stack_count,
                   p.carepackage_id::text AS carepackage_id,
                   p.lootbox_owner_team_id::text AS lootbox_owner_team_id
              FROM pubg.telem_item_pickup p
             WHERE p.match_id = %s AND p.team_id IS NOT NULL
            UNION ALL
            SELECT e.match_id, 'equip'::text AS event_kind, e.event_time,
                   e.team_id, e.account_id, e.item_id, e.category, e.sub_category,
                   1::integer AS stack_count,
                   NULL::text AS carepackage_id, NULL::text AS lootbox_owner_team_id
              FROM pubg.telem_item_equip e
             WHERE e.match_id = %s AND e.team_id IS NOT NULL
            UNION ALL
            SELECT u.match_id, 'use'::text AS event_kind, u.event_time,
                   u.team_id, u.account_id, u.item_id, u.category, u.sub_category,
                   coalesce(u.stack_count, 1)::integer AS stack_count,
                   NULL::text AS carepackage_id, NULL::text AS lootbox_owner_team_id
              FROM pubg.telem_item_use u
             WHERE u.match_id = %s AND u.team_id IS NOT NULL
        ),
        tagged AS (
            SELECT
                b.phase_idx,
                ie.team_id,
                ie.event_kind,
                ie.item_id,
                lower(coalesce(ie.item_id, '')) AS item_l,
                lower(coalesce(ie.category, '')) AS cat_l,
                lower(coalesce(ie.sub_category, '')) AS subcat_l,
                ie.stack_count,
                ie.carepackage_id,
                ie.lootbox_owner_team_id
              FROM item_events ie
              JOIN m ON m.match_id = ie.match_id
              JOIN b
                ON b.match_id = ie.match_id
               AND extract(epoch FROM (ie.event_time - m.started_at))
                   BETWEEN b.phase_start_s AND b.phase_end_s
        ),
        agg AS (
            SELECT
                phase_idx,
                team_id,
                count(*) FILTER (WHERE event_kind = 'pickup')::integer AS pickup_count,
                count(*) FILTER (WHERE event_kind = 'equip')::integer AS equip_count,
                count(*) FILTER (WHERE event_kind = 'use')::integer AS use_count,
                count(DISTINCT item_id)::integer AS unique_item_count,
                count(*) FILTER (
                    WHERE event_kind = 'pickup'
                      AND (item_l LIKE 'item_weapon_%%' OR cat_l = 'weapon')
                )::integer AS weapon_pickup_count,
                count(*) FILTER (
                    WHERE event_kind = 'equip'
                      AND (item_l LIKE 'item_weapon_%%' OR cat_l = 'weapon')
                )::integer AS weapon_equip_count,
                sum(stack_count) FILTER (
                    WHERE event_kind = 'pickup'
                      AND (item_l LIKE '%%ammo%%' OR cat_l = 'ammunition' OR subcat_l = 'ammo')
                )::integer AS ammo_pickup_count,
                count(*) FILTER (
                    WHERE event_kind = 'pickup'
                      AND (
                          item_l LIKE '%%armor%%'
                          OR item_l LIKE '%%helmet%%'
                          OR item_l LIKE '%%back%%'
                          OR subcat_l IN ('vest', 'helmet', 'backpack')
                      )
                )::integer AS armor_pickup_count,
                count(*) FILTER (
                    WHERE event_kind = 'pickup'
                      AND (subcat_l = 'throwable' OR item_l LIKE '%%throw%%')
                )::integer AS throwable_pickup_count,
                count(*) FILTER (
                    WHERE event_kind = 'use'
                      AND (
                          item_l LIKE '%%heal%%'
                          OR item_l LIKE '%%firstaid%%'
                          OR item_l LIKE '%%medkit%%'
                          OR item_l LIKE '%%bandage%%'
                      )
                )::integer AS heal_use_count,
                count(*) FILTER (
                    WHERE event_kind = 'use'
                      AND (
                          item_l LIKE '%%boost%%'
                          OR item_l LIKE '%%energydrink%%'
                          OR item_l LIKE '%%painkiller%%'
                          OR item_l LIKE '%%adrenaline%%'
                      )
                )::integer AS boost_use_count,
                count(*) FILTER (
                    WHERE event_kind = 'pickup' AND carepackage_id IS NOT NULL
                )::integer AS carepackage_pickup_count,
                count(*) FILTER (
                    WHERE event_kind = 'pickup' AND lootbox_owner_team_id IS NOT NULL
                )::integer AS lootbox_pickup_count
              FROM tagged
             GROUP BY phase_idx, team_id
        )
        INSERT INTO stdb.team_phase_loot (
            match_id, team_id, phase_idx,
            pickup_count, equip_count, use_count, unique_item_count,
            weapon_pickup_count, weapon_equip_count, ammo_pickup_count,
            armor_pickup_count, throwable_pickup_count, heal_use_count,
            boost_use_count, carepackage_pickup_count, lootbox_pickup_count
        )
        SELECT
            tp.match_id,
            tp.team_id,
            tp.phase_idx,
            coalesce(a.pickup_count, 0),
            coalesce(a.equip_count, 0),
            coalesce(a.use_count, 0),
            coalesce(a.unique_item_count, 0),
            coalesce(a.weapon_pickup_count, 0),
            coalesce(a.weapon_equip_count, 0),
            coalesce(a.ammo_pickup_count, 0),
            coalesce(a.armor_pickup_count, 0),
            coalesce(a.throwable_pickup_count, 0),
            coalesce(a.heal_use_count, 0),
            coalesce(a.boost_use_count, 0),
            coalesce(a.carepackage_pickup_count, 0),
            coalesce(a.lootbox_pickup_count, 0)
          FROM team_phase tp
     LEFT JOIN agg a
            ON a.team_id = tp.team_id AND a.phase_idx = tp.phase_idx
        """,
        (match_id, match_id, match_id, match_id, match_id, match_id),
    )
    cur.execute("SELECT count(*) FROM stdb.team_phase_loot WHERE match_id = %s", (match_id,))
    return int(cur.fetchone()[0])


def build_team_phase_labels(cur, match_id: str) -> int:
    _execute(cur, "DELETE FROM stdb.team_phase_labels WHERE match_id = %s", (match_id,))
    _execute(
        cur,
        """
        WITH b AS (
            SELECT
                match_id,
                phase_idx,
                lead(phase_idx) OVER (PARTITION BY match_id ORDER BY phase_idx) AS next_phase_idx
              FROM stdb.v_match_phase_bounds
             WHERE match_id = %s
        ),
        team_phase AS (
            SELECT t.match_id, t.team_id, t.final_rank, t.won, b.phase_idx AS cut_phase, b.next_phase_idx
              FROM stdb.teams t
              JOIN b ON b.match_id = t.match_id
             WHERE t.match_id = %s
        ),
        state_now AS (
            SELECT match_id, team_id, phase_idx,
                   coalesce(alive_players_start, 0) AS alive_players_start,
                   coalesce(alive_players_end, 0) AS alive_players_end
              FROM stdb.team_phase_state
             WHERE match_id = %s
        ),
        last_alive AS (
            SELECT match_id, team_id, max(phase_idx)::integer AS last_phase_alive
              FROM state_now
             WHERE alive_players_end > 0
             GROUP BY match_id, team_id
        ),
        team_count AS (
            SELECT match_id, count(*)::real AS n_teams
              FROM stdb.teams
             WHERE match_id = %s
             GROUP BY match_id
        )
        INSERT INTO stdb.team_phase_labels (
            match_id, team_id, cut_phase,
            alive_at_phase_start, alive_at_phase_end, next_phase_idx,
            target_defined, survives_next_phase, eliminated_next_phase,
            hazard_next, final_rank, final_rank_norm, final_top3, final_top5,
            final_top10, final_winner, last_phase_alive
        )
        SELECT
            tp.match_id,
            tp.team_id,
            tp.cut_phase,
            (coalesce(sn.alive_players_start, 0) > 0) AS alive_at_phase_start,
            (coalesce(sn.alive_players_end, 0) > 0) AS alive_at_phase_end,
            tp.next_phase_idx,
            ((coalesce(sn.alive_players_end, 0) > 0) AND tp.next_phase_idx IS NOT NULL)
                AS target_defined,
            CASE WHEN tp.next_phase_idx IS NULL THEN NULL
                 ELSE (coalesce(nx.alive_players_end, 0) > 0)
            END AS survives_next_phase,
            CASE WHEN tp.next_phase_idx IS NULL THEN NULL
                 ELSE ((coalesce(sn.alive_players_end, 0) > 0) AND NOT (coalesce(nx.alive_players_end, 0) > 0))
            END AS eliminated_next_phase,
            CASE WHEN tp.next_phase_idx IS NULL THEN NULL
                 WHEN ((coalesce(sn.alive_players_end, 0) > 0) AND NOT (coalesce(nx.alive_players_end, 0) > 0))
                 THEN 1 ELSE 0 END AS hazard_next,
            tp.final_rank,
            (tp.final_rank::real / greatest(tc.n_teams, 1.0))::real AS final_rank_norm,
            (tp.final_rank <= 3) AS final_top3,
            (tp.final_rank <= 5) AS final_top5,
            (tp.final_rank <= 10) AS final_top10,
            tp.won AS final_winner,
            la.last_phase_alive
          FROM team_phase tp
     LEFT JOIN state_now sn
            ON sn.match_id = tp.match_id
           AND sn.team_id = tp.team_id
           AND sn.phase_idx = tp.cut_phase
     LEFT JOIN state_now nx
            ON nx.match_id = tp.match_id
           AND nx.team_id = tp.team_id
           AND nx.phase_idx = tp.next_phase_idx
     LEFT JOIN last_alive la
            ON la.match_id = tp.match_id AND la.team_id = tp.team_id
     LEFT JOIN team_count tc
            ON tc.match_id = tp.match_id
        """,
        (match_id, match_id, match_id, match_id),
    )
    cur.execute("SELECT count(*) FROM stdb.team_phase_labels WHERE match_id = %s", (match_id,))
    return int(cur.fetchone()[0])


def build_match(match_id: str) -> dict:
    t0 = time.time()
    conn = connect()
    cur = conn.cursor()
    try:
        cur.execute("SELECT count(*) FROM stdb.matches WHERE match_id = %s", (match_id,))
        if cur.fetchone()[0] == 0:
            return {"match_id": match_id, "ok": False, "reason": "match is not in stdb.matches"}

        counts = {
            "match_phase_global": build_match_phase_global(cur, match_id),
            "team_phase_state": build_team_phase_state(cur, match_id),
            "team_phase_combat": build_team_phase_combat(cur, match_id),
            "team_phase_loot": build_team_phase_loot(cur, match_id),
        }
        counts["team_phase_labels"] = build_team_phase_labels(cur, match_id)
        conn.commit()
        return {
            "match_id": match_id,
            "ok": True,
            "elapsed_s": round(time.time() - t0, 2),
            **counts,
        }
    except Exception as exc:
        conn.rollback()
        logger.exception("RQ1 feature build failed for %s", match_id)
        return {"match_id": match_id, "ok": False, "reason": repr(exc)}
    finally:
        cur.close()
        conn.close()


def verify_match(match_id: str) -> dict:
    conn = connect()
    cur = conn.cursor()
    try:
        out = {"match_id": match_id}
        for table in (
            "match_phase_global",
            "team_phase_state",
            "team_phase_combat",
            "team_phase_loot",
            "team_phase_labels",
        ):
            cur.execute(f"SELECT count(*) FROM stdb.{table} WHERE match_id = %s", (match_id,))
            out[table] = int(cur.fetchone()[0])
        cur.execute(
            """
            SELECT
                count(*) AS rows,
                count(*) FILTER (WHERE target_defined) AS target_rows,
                count(*) FILTER (WHERE hazard_next = 1) AS hazard_positive_rows
              FROM stdb.team_phase_labels
             WHERE match_id = %s
            """,
            (match_id,),
        )
        rows, target_rows, positives = cur.fetchone()
        out["label_rows"] = int(rows)
        out["target_rows"] = int(target_rows)
        out["hazard_positive_rows"] = int(positives)
        return out
    finally:
        cur.close()
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply-schema", action="store_true", help="Apply stdb/schema_rq1.sql first")
    parser.add_argument("--match-id", help="Build a single match")
    parser.add_argument("--map", dest="map_name", help="Filter current stdb.matches by map_name")
    parser.add_argument("--mode", help="Filter current stdb.matches by game_mode")
    parser.add_argument("--limit", type=int, help="Limit selected matches")
    parser.add_argument("--all-current", action="store_true", help="Build all current stdb.matches")
    parser.add_argument("--verify", action="store_true", help="Print verification counts after build")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.apply_schema:
        logger.info("applying %s", SCHEMA_PATH)
        apply_sql_file(SCHEMA_PATH)

    if args.match_id:
        match_ids: Iterable[str] = [args.match_id]
    elif args.all_current or args.map_name or args.mode or args.limit:
        match_ids = list_current_match_ids(
            map_name=args.map_name, mode=args.mode, limit=args.limit
        )
    else:
        parser.error("pass --match-id or --all-current/--map/--mode/--limit")
        return 2

    ok = 0
    failed = 0
    for match_id in match_ids:
        result = build_match(match_id)
        if result.get("ok"):
            ok += 1
            logger.info("built %s", result)
            if args.verify:
                logger.info("verify %s", verify_match(match_id))
        else:
            failed += 1
            logger.error("failed %s", result)

    logger.info("done ok=%s failed=%s", ok, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
