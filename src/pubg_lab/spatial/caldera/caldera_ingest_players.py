"""Extract Caldera player endpoint/breadcrumb data into br_ground/br_repr.

The script reads only the minimal player USD layers downloaded by
scripts/caldera_dataset_manager.py.  Coordinates are converted from USD stage
units to meters using the stage metersPerUnit metadata.

Examples:

    .venv/bin/python scripts/caldera_ingest_players.py --apply-schema --dry-run
    PG_PASSWORD=... .venv/bin/python scripts/caldera_ingest_players.py --endpoint-limit 1000 --match-limit 2
    PG_PASSWORD=... .venv/bin/python scripts/caldera_ingest_players.py
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

import psycopg2.extras
from psycopg2.extras import Json
from pxr import Usd


ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))

from pubg_lab.data.stdb.conn import apply_sql_file, connect


SOURCE_ID = "activision_caldera"
SOURCE_GAME = "Call of Duty: Warzone Caldera"
FEATURE_SET_ID = "caldera_spatial_trajectory_v1"
ENDPOINT_ARTIFACT_ID = "caldera_minimal_layers_endpoints_usd"
BREADCRUMB_ARTIFACT_ID = "caldera_minimal_layers_breadcrumbs_usd"

DEFAULT_DATA_DIR = ROOT / "data" / "br_l0" / "caldera" / "raw_minimal_players"
DEFAULT_ENDPOINTS = DEFAULT_DATA_DIR / "layers" / "endpoints.usd"
DEFAULT_BREADCRUMBS = DEFAULT_DATA_DIR / "layers" / "breadcrumbs.usd"


def vec_to_meters(vec, scale: float) -> tuple[float, float, float]:
    return (float(vec[0]) * scale, float(vec[1]) * scale, float(vec[2]) * scale)


def sample_time_s(timecode: float, fps: float) -> float:
    # Activision's sample code documents a 60x scale from USD timeline to game time.
    return round(60.0 * (float(timecode) / float(fps)), 3)


def life_state(life: float | None) -> str | None:
    if life is None:
        return None
    if life < 0:
        return "pre_spawn"
    if life <= 1:
        return "in_play"
    return "post_play"


def pairwise_path_length(points: list[tuple[float, float, float]]) -> float:
    total = 0.0
    for prev, cur in zip(points, points[1:]):
        total += math.dist(prev, cur)
    return total


def chunked(values: Iterable[tuple], size: int) -> Iterable[list[tuple]]:
    chunk: list[tuple] = []
    for value in values:
        chunk.append(value)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def ensure_feature_set(cur) -> None:
    cur.execute(
        """
        INSERT INTO br_repr.feature_sets (
            feature_set_id, description, source_id, manifest_path, metadata
        ) VALUES (
            %s, %s, %s, %s, %s
        )
        ON CONFLICT (feature_set_id) DO UPDATE
        SET description = EXCLUDED.description,
            manifest_path = EXCLUDED.manifest_path,
            metadata = EXCLUDED.metadata
        """,
        (
            FEATURE_SET_ID,
            "Caldera low-observability spatial trajectory representation from endpoints and breadcrumbs.",
            SOURCE_ID,
            "data/br_l0/caldera/manifest/caldera_minimal_players_manifest.json",
            Json(
                {
                    "has_geometry": True,
                    "has_endpoint_time": True,
                    "has_breadcrumb_trajectory": True,
                    "has_team_id": False,
                    "has_combat": False,
                    "has_loot": False,
                    "has_zone": False,
                }
            ),
        ),
    )


def endpoint_rows(stage: Usd.Stage, limit: int | None) -> Iterable[tuple]:
    scale = float(stage.GetMetadata("metersPerUnit") or 1.0)
    prim = stage.GetPrimAtPath("/players/endpoints/points_0")
    if not prim:
        raise RuntimeError("Missing /players/endpoints/points_0")
    points = prim.GetAttribute("points").Get()
    times = prim.GetAttribute("primvars:time").Get()
    colors = prim.GetAttribute("primvars:displayColor").Get()
    n = len(points) if limit is None else min(limit, len(points))
    for idx in range(n):
        x, y, z = vec_to_meters(points[idx], scale)
        color = colors[idx] if colors is not None else (None, None, None)
        raw_time_ms = int(times[idx]) if times is not None else None
        t_s = float(raw_time_ms) / 1000.0 if raw_time_ms is not None else None
        yield (
            SOURCE_ID,
            idx,
            t_s,
            x,
            y,
            z,
            x,
            y,
            z,
            float(color[0]) if color[0] is not None else None,
            float(color[1]) if color[1] is not None else None,
            float(color[2]) if color[2] is not None else None,
            ENDPOINT_ARTIFACT_ID,
            Json(
                {"raw_time_ms": raw_time_ms, "raw_unit": "usd_stage_unit", "meters_per_unit": scale}
            ),
        )


def ingest_endpoints(args: argparse.Namespace) -> int:
    stage = Usd.Stage.Open(str(args.endpoints))
    if stage is None:
        raise RuntimeError(f"Could not open endpoints layer: {args.endpoints}")
    prim = stage.GetPrimAtPath("/players/endpoints/points_0")
    total_available = len(prim.GetAttribute("points").Get())
    n_expected = (
        total_available
        if args.endpoint_limit is None
        else min(args.endpoint_limit, total_available)
    )
    print(f"endpoints_available={total_available} ingesting={n_expected}")
    if args.dry_run:
        return n_expected

    insert_sql = """
        INSERT INTO br_ground.endpoint_samples (
            source_id, endpoint_idx, sample_t, pos, pos_x_m, pos_y_m, pos_z_m,
            color_r, color_g, color_b, source_artifact_id, metadata
        ) VALUES (
            %s, %s, %s,
            ST_MakePoint(%s, %s, %s)::geometry(PointZ, 0),
            %s, %s, %s,
            %s, %s, %s, %s, %s
        )
        ON CONFLICT (source_id, endpoint_idx) DO UPDATE
        SET sample_t = EXCLUDED.sample_t,
            pos = EXCLUDED.pos,
            pos_x_m = EXCLUDED.pos_x_m,
            pos_y_m = EXCLUDED.pos_y_m,
            pos_z_m = EXCLUDED.pos_z_m,
            color_r = EXCLUDED.color_r,
            color_g = EXCLUDED.color_g,
            color_b = EXCLUDED.color_b,
            source_artifact_id = EXCLUDED.source_artifact_id,
            metadata = EXCLUDED.metadata
    """

    inserted = 0
    with connect() as conn:
        with conn.cursor() as cur:
            ensure_feature_set(cur)
            for batch in chunked(endpoint_rows(stage, args.endpoint_limit), args.batch_size):
                psycopg2.extras.execute_batch(cur, insert_sql, batch, page_size=args.batch_size)
                inserted += len(batch)
                conn.commit()
                print(f"endpoint_rows_upserted={inserted}")
    return inserted


def player_translate_attr(player_prim):
    for attr in player_prim.GetAttributes():
        if attr.GetName() == "xformOp:translate":
            return attr
    return None


def player_life_attr(player_prim):
    attr = player_prim.GetAttribute("primvars:life")
    return attr if attr else None


def collect_player_series(stage: Usd.Stage, player_prim) -> list[dict]:
    fps = float(stage.GetFramesPerSecond() or 30.0)
    scale = float(stage.GetMetadata("metersPerUnit") or 1.0)
    translate = player_translate_attr(player_prim)
    life_attr = player_life_attr(player_prim)
    if translate is None:
        return []
    rows = []
    for timecode in translate.GetTimeSamples():
        pos = translate.Get(time=timecode)
        life = float(life_attr.Get(time=timecode)) if life_attr else None
        x, y, z = vec_to_meters(pos, scale)
        rows.append(
            {
                "timecode": float(timecode),
                "sample_t": sample_time_s(float(timecode), fps),
                "pos": (x, y, z),
                "life": life,
                "alive_state": life_state(life),
                "meters_per_unit": scale,
            }
        )
    return rows


def upsert_match(
    cur, br_match_id: str, source_match_id: str, duration_s: float | None, player_count: int
) -> None:
    cur.execute(
        """
        INSERT INTO br_ground.matches (
            br_match_id, source_id, source_match_id, source_game, map_name,
            mode_name, perspective, team_size, duration_s, coordinate_unit,
            coordinate_scale_to_m, observability, raw_artifact_id, metadata
        ) VALUES (
            %s, %s, %s, %s, 'Caldera', NULL, NULL, NULL, %s,
            'meter', 0.0254, 'breadcrumb_trajectory_low_observability', %s, %s
        )
        ON CONFLICT (br_match_id) DO UPDATE
        SET duration_s = EXCLUDED.duration_s,
            observability = EXCLUDED.observability,
            raw_artifact_id = EXCLUDED.raw_artifact_id,
            metadata = EXCLUDED.metadata
        """,
        (
            br_match_id,
            SOURCE_ID,
            source_match_id,
            SOURCE_GAME,
            duration_s,
            BREADCRUMB_ARTIFACT_ID,
            Json({"source_match_name": source_match_id, "player_count": player_count}),
        ),
    )


def upsert_agent(
    cur, br_match_id: str, br_agent_id: str, source_agent_id: str, source_path: str
) -> None:
    cur.execute(
        """
        INSERT INTO br_ground.agents (
            br_agent_id, br_match_id, source_agent_id, agent_kind, team_id,
            is_observed, metadata
        ) VALUES (
            %s, %s, %s, 'player', NULL, TRUE, %s
        )
        ON CONFLICT (br_match_id, source_agent_id, agent_kind) DO UPDATE
        SET br_agent_id = EXCLUDED.br_agent_id,
            metadata = EXCLUDED.metadata
        """,
        (br_agent_id, br_match_id, source_agent_id, Json({"source_prim_path": source_path})),
    )


def trajectory_sample_rows(
    br_match_id: str, br_agent_id: str, series: list[dict], sample_stride: int
) -> list[tuple]:
    rows = []
    for idx, row in enumerate(series):
        if sample_stride > 1 and idx % sample_stride:
            continue
        x, y, z = row["pos"]
        rows.append(
            (
                br_match_id,
                br_agent_id,
                row["sample_t"],
                x,
                y,
                z,
                x,
                y,
                z,
                row["alive_state"],
                BREADCRUMB_ARTIFACT_ID,
                Json(
                    {
                        "life_norm": row["life"],
                        "source_timecode": row["timecode"],
                        "raw_unit": "usd_stage_unit",
                        "meters_per_unit": row["meters_per_unit"],
                    }
                ),
            )
        )
    return rows


def summarize_series(series: list[dict]) -> dict:
    in_play = [row for row in series if row["alive_state"] == "in_play"]
    usable = in_play or series
    if not usable:
        return {
            "n_samples": 0,
            "t_start_s": None,
            "t_end_s": None,
            "duration_s": None,
            "path_length_m": None,
            "displacement_m": None,
            "sinuosity": None,
            "endpoint_pos": None,
            "metadata": {},
        }
    points = [row["pos"] for row in usable]
    path_length = pairwise_path_length(points)
    displacement = math.dist(points[0], points[-1]) if len(points) > 1 else 0.0
    sinuosity = path_length / displacement if displacement > 0 else None
    t_start = usable[0]["sample_t"]
    t_end = usable[-1]["sample_t"]
    return {
        "n_samples": len(usable),
        "t_start_s": t_start,
        "t_end_s": t_end,
        "duration_s": t_end - t_start if t_start is not None and t_end is not None else None,
        "path_length_m": path_length,
        "displacement_m": displacement,
        "sinuosity": sinuosity,
        "endpoint_pos": points[-1],
        "metadata": {
            "in_play_samples": len(in_play),
            "all_samples": len(series),
            "summary_window": "in_play" if in_play else "all_samples_fallback",
        },
    }


def upsert_summary_and_outcome(cur, br_match_id: str, br_agent_id: str, summary: dict) -> None:
    endpoint = summary["endpoint_pos"]
    endpoint_args = endpoint if endpoint is not None else (None, None, None)
    cur.execute(
        """
        INSERT INTO br_repr.trajectory_summaries (
            br_match_id, br_agent_id, feature_set_id, n_samples,
            t_start_s, t_end_s, duration_s, path_length_m, displacement_m,
            sinuosity, endpoint_pos, metadata
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            CASE WHEN %s IS NULL THEN NULL ELSE ST_MakePoint(%s, %s, %s)::geometry(PointZ, 0) END,
            %s
        )
        ON CONFLICT (br_match_id, br_agent_id) DO UPDATE
        SET feature_set_id = EXCLUDED.feature_set_id,
            n_samples = EXCLUDED.n_samples,
            t_start_s = EXCLUDED.t_start_s,
            t_end_s = EXCLUDED.t_end_s,
            duration_s = EXCLUDED.duration_s,
            path_length_m = EXCLUDED.path_length_m,
            displacement_m = EXCLUDED.displacement_m,
            sinuosity = EXCLUDED.sinuosity,
            endpoint_pos = EXCLUDED.endpoint_pos,
            metadata = EXCLUDED.metadata,
            created_at = now()
        """,
        (
            br_match_id,
            br_agent_id,
            FEATURE_SET_ID,
            summary["n_samples"],
            summary["t_start_s"],
            summary["t_end_s"],
            summary["duration_s"],
            summary["path_length_m"],
            summary["displacement_m"],
            summary["sinuosity"],
            endpoint_args[0],
            endpoint_args[0],
            endpoint_args[1],
            endpoint_args[2],
            Json(summary["metadata"]),
        ),
    )
    cur.execute(
        """
        INSERT INTO br_ground.outcomes (
            br_match_id, br_agent_id, final_rank, survived, elimination_t,
            endpoint_pos, source_artifact_id, metadata
        ) VALUES (
            %s, %s, NULL, NULL, %s,
            CASE WHEN %s IS NULL THEN NULL ELSE ST_MakePoint(%s, %s, %s)::geometry(PointZ, 0) END,
            %s, %s
        )
        ON CONFLICT (br_match_id, br_agent_id) DO UPDATE
        SET elimination_t = EXCLUDED.elimination_t,
            endpoint_pos = EXCLUDED.endpoint_pos,
            source_artifact_id = EXCLUDED.source_artifact_id,
            metadata = EXCLUDED.metadata
        """,
        (
            br_match_id,
            br_agent_id,
            summary["t_end_s"],
            endpoint_args[0],
            endpoint_args[0],
            endpoint_args[1],
            endpoint_args[2],
            BREADCRUMB_ARTIFACT_ID,
            Json({"outcome_type": "breadcrumb_last_in_play_position"}),
        ),
    )


def ingest_breadcrumbs(args: argparse.Namespace) -> tuple[int, int, int]:
    stage = Usd.Stage.Open(str(args.breadcrumbs))
    if stage is None:
        raise RuntimeError(f"Could not open breadcrumbs layer: {args.breadcrumbs}")
    root = stage.GetPrimAtPath("/players/breadcrumbs")
    if not root:
        raise RuntimeError("Missing /players/breadcrumbs")

    matches = list(root.GetChildren())
    if args.match_limit is not None:
        matches = matches[: args.match_limit]
    print(f"breadcrumb_matches_ingesting={len(matches)}")

    total_players = 0
    total_samples = 0
    if args.dry_run:
        for match in matches:
            players = list(match.GetChildren())
            if args.player_limit_per_match is not None:
                players = players[: args.player_limit_per_match]
            total_players += len(players)
            for player in players:
                attr = player_translate_attr(player)
                total_samples += len(attr.GetTimeSamples()) if attr else 0
        return len(matches), total_players, total_samples

    sample_sql = """
        INSERT INTO br_ground.trajectory_samples (
            br_match_id, br_agent_id, sample_t, sample_kind, pos,
            pos_x_m, pos_y_m, pos_z_m, alive_state, source_artifact_id, metadata
        ) VALUES (
            %s, %s, %s, 'breadcrumb',
            ST_MakePoint(%s, %s, %s)::geometry(PointZ, 0),
            %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (br_match_id, br_agent_id, sample_t, sample_kind) DO UPDATE
        SET pos = EXCLUDED.pos,
            pos_x_m = EXCLUDED.pos_x_m,
            pos_y_m = EXCLUDED.pos_y_m,
            pos_z_m = EXCLUDED.pos_z_m,
            alive_state = EXCLUDED.alive_state,
            source_artifact_id = EXCLUDED.source_artifact_id,
            metadata = EXCLUDED.metadata
    """

    with connect() as conn:
        with conn.cursor() as cur:
            ensure_feature_set(cur)
            for match_idx, match in enumerate(matches, start=1):
                match_name = match.GetName()
                players = list(match.GetChildren())
                if args.player_limit_per_match is not None:
                    players = players[: args.player_limit_per_match]

                series_by_player = []
                duration_s = 0.0
                for player in players:
                    series = collect_player_series(stage, player)
                    if series:
                        duration_s = max(duration_s, max(row["sample_t"] for row in series))
                    series_by_player.append((player, series))

                br_match_id = f"caldera:{match_name}"
                upsert_match(cur, br_match_id, match_name, duration_s, len(players))

                match_samples = 0
                for player, series in series_by_player:
                    source_agent_id = player.GetName()
                    br_agent_id = f"caldera:{match_name}:{source_agent_id}"
                    upsert_agent(
                        cur, br_match_id, br_agent_id, source_agent_id, str(player.GetPath())
                    )

                    rows = trajectory_sample_rows(
                        br_match_id, br_agent_id, series, args.sample_stride
                    )
                    if rows:
                        psycopg2.extras.execute_batch(
                            cur, sample_sql, rows, page_size=args.batch_size
                        )
                    summary = summarize_series(series)
                    upsert_summary_and_outcome(cur, br_match_id, br_agent_id, summary)
                    total_players += 1
                    total_samples += len(rows)
                    match_samples += len(rows)

                conn.commit()
                print(
                    f"breadcrumb_match={match_name} "
                    f"idx={match_idx}/{len(matches)} players={len(players)} samples={match_samples}"
                )

    return len(matches), total_players, total_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-schema", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-endpoints", action="store_true")
    parser.add_argument("--skip-breadcrumbs", action="store_true")
    parser.add_argument("--endpoints", type=Path, default=DEFAULT_ENDPOINTS)
    parser.add_argument("--breadcrumbs", type=Path, default=DEFAULT_BREADCRUMBS)
    parser.add_argument("--endpoint-limit", type=int)
    parser.add_argument("--match-limit", type=int)
    parser.add_argument("--player-limit-per-match", type=int)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    if args.apply_schema and not args.dry_run:
        apply_sql_file(ROOT / "stdb" / "schema_br_layers.sql")

    if not args.dry_run:
        with connect() as conn:
            with conn.cursor() as cur:
                ensure_feature_set(cur)
            conn.commit()

    endpoint_count = 0
    if not args.skip_endpoints:
        endpoint_count = ingest_endpoints(args)

    breadcrumb_counts = (0, 0, 0)
    if not args.skip_breadcrumbs:
        breadcrumb_counts = ingest_breadcrumbs(args)

    print("summary")
    print(f"  endpoint_rows={endpoint_count}")
    print(f"  breadcrumb_matches={breadcrumb_counts[0]}")
    print(f"  breadcrumb_players={breadcrumb_counts[1]}")
    print(f"  breadcrumb_samples={breadcrumb_counts[2]}")
    print(f"  dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
