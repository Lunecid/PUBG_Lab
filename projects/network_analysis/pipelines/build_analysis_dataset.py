"""
Build analysis-ready intermediate datasets from the existing PUBG DB + .pt cache.

This intentionally does not regenerate the PyG .pt graph cache.  The .pt files
provide match/static/frame position data, while PostgreSQL telemetry supplies the
actual damage events used to build team-team combat networks.

Outputs are CSV files plus metadata.json:
  - matches.csv
  - teams.csv
  - team_edges.csv
  - team_centroids.csv

Example:
  python3 -m network_analysis.build_analysis_dataset \
      --map Baltic_Main --mode squad --n 5
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

import argparse
import csv
import json
import math
import os
import tempfile
from datetime import datetime
from typing import Any

from pubg_lab.analysis.network.features.archetype import (
    assign_archetype_labels,
    compute_per_match_archetype_features,
)
from pubg_lab.analysis.network.config.config import (
    DEFAULT_MAP,
    DEFAULT_MODE,
    OUTPUT_ROOT,
    LANDING_OFFSET_S,
    MOVEMENT_WINDOW_S,
)
from pubg_lab.analysis.network.data.data_loader import build_match_data, list_matches
from pubg_lab.analysis.network.data.db_loader import load_match_damage
from projects.network_analysis.studies.h1_archetype import compute_h1_archetype
from projects.network_analysis.studies.h2_battle_center import compute_h2_battle_center
from projects.network_analysis.studies.h3_movement import compute_h3_movement
from projects.network_analysis.studies.hypotheses import attach_community_features
from pubg_lab.analysis.network.features.movement import (
    assign_movement_labels,
    compute_movement_features,
)
from pubg_lab.analysis.network.features.team_centrality import (
    attach_centralities_to_rows,
    compute_team_centralities,
)
from pubg_lab.analysis.network.features.team_network import (
    build_early_team_graph,
    build_team_graph,
    graph_globals,
)


DEFAULT_OUT_ROOT = os.path.join(OUTPUT_ROOT, "analysis_datasets")


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _jsonable(value: Any) -> Any:
    """Convert common scientific values to CSV/JSON-safe scalars."""
    if value is None:
        return ""
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _clean_json(value: Any) -> Any:
    """Recursively replace non-JSON floats with None."""
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)

    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=os.path.dirname(path),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _jsonable(row.get(key)) for key in columns})
        os.replace(tmp_path, path)
        os.chmod(path, 0o644)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=os.path.dirname(path),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(
                _clean_json(payload),
                f,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            f.write("\n")
        os.replace(tmp_path, path)
        os.chmod(path, 0o644)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _edge_rows(match_id: str, window: str, G) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, target, data in G.edges(data=True):
        rows.append(
            {
                "match_id": match_id,
                "window": window,
                "source_team_idx": int(source),
                "target_team_idx": int(target),
                "weight": float(data.get("weight", 0.0)),
                "damage": float(data.get("dmg", 0.0)),
                "events": int(data.get("events", 0)),
            }
        )
    rows.sort(
        key=lambda r: (r["match_id"], r["window"], r["source_team_idx"], r["target_team_idx"])
    )
    return rows


def _centroid_rows(
    md: dict,
    *,
    sample_sec: int,
) -> list[dict[str, Any]]:
    static = md["static"]
    rows: list[dict[str, Any]] = []
    last_kept_t: float | None = None

    for frame in md.get("frames", []):
        t = float(frame.get("t", 0.0))
        if last_kept_t is not None and t - last_kept_t < sample_sec - 1e-9:
            continue
        last_kept_t = t

        team_alive = frame.get("team_alive_count") or []
        centroids = frame.get("team_centroid") or []
        for team_idx, centroid in enumerate(centroids):
            if centroid is None:
                continue
            rows.append(
                {
                    "match_id": static["match_id"],
                    "t": t,
                    "team_idx": int(team_idx),
                    "alive_count": int(team_alive[team_idx]) if team_idx < len(team_alive) else "",
                    "centroid_x": float(centroid[0]),
                    "centroid_y": float(centroid[1]),
                }
            )
    return rows


def _merge_early_features(rows: list[dict], early_rows: list[dict]) -> None:
    early_by_team = {r["team_idx"]: r for r in early_rows}
    keys = [
        "degree",
        "weighted_deg",
        "combat_events",
        "dmg_dealt",
        "dmg_taken",
        "c_degree",
        "c_closeness",
        "c_betweenness",
        "c_eigenvector",
        "c_pagerank",
    ]
    for row in rows:
        early = early_by_team.get(row["team_idx"], {})
        for key in keys:
            row[f"early_{key}"] = early.get(key, 0.0)


def _match_summary_row(
    *,
    md: dict,
    pt_path: str,
    G,
    G_early,
    globals_full: dict,
    events: list[dict],
    early_window_sec: int,
) -> dict[str, Any]:
    static = md["static"]
    total_damage = sum(float(e.get("damage") or 0.0) for e in events)
    early_damage = sum(
        float(e.get("damage") or 0.0)
        for e in events
        if float(e.get("elapsed") or 0.0) <= early_window_sec
    )
    return {
        "match_id": static["match_id"],
        "map_name": static.get("map_name"),
        "game_mode": static.get("game_mode"),
        "duration_s": float(static.get("duration_s") or 0.0),
        "n_players": int(static.get("n_players") or 0),
        "n_teams": int(static.get("n_teams") or 0),
        "pt_path": pt_path,
        "damage_events": len(events),
        "damage_total": total_damage,
        "early_window_sec": early_window_sec,
        "early_damage_total": early_damage,
        "full_n_edges": G.number_of_edges(),
        "early_n_edges": G_early.number_of_edges(),
        **{f"full_{k}": v for k, v in globals_full.items()},
    }


def _process_match(
    pt_path: str,
    *,
    weight_mode: str,
    early_window_sec: int,
    centroid_sample_sec: int,
    include_centroids: bool,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    # fetch_damage=False keeps frame building independent of DB; damage is loaded once below.
    md = build_match_data(pt_path, fetch_damage=False, stride=1)
    match_id = md["static"]["match_id"]
    events = load_match_damage(match_id).get("events", []) or []

    G, rows = build_team_graph(
        md,
        weight_mode=weight_mode,
        preloaded_events=events,
    )
    G_early, rows_early = build_early_team_graph(
        md,
        window_sec=early_window_sec,
        weight_mode=weight_mode,
        preloaded_events=events,
    )

    centr = compute_team_centralities(G)
    attach_centralities_to_rows(rows, centr)
    early_centr = compute_team_centralities(G_early)
    attach_centralities_to_rows(rows_early, early_centr)
    _merge_early_features(rows, rows_early)

    attach_community_features(rows, G, seed=0)
    compute_per_match_archetype_features(rows, md)
    compute_movement_features(rows, md)

    globals_full = graph_globals(G)
    match_row = _match_summary_row(
        md=md,
        pt_path=pt_path,
        G=G,
        G_early=G_early,
        globals_full=globals_full,
        events=events,
        early_window_sec=early_window_sec,
    )

    edge_rows = _edge_rows(match_id, "full", G) + _edge_rows(
        match_id, f"early_{early_window_sec}s", G_early
    )
    centroid_rows = _centroid_rows(md, sample_sec=centroid_sample_sec) if include_centroids else []
    return match_row, rows, edge_rows, centroid_rows


def _build_metadata(
    *,
    args: argparse.Namespace,
    out_dir: str,
    match_rows: list[dict],
    team_rows: list[dict],
    edge_rows: list[dict],
    centroid_rows: list[dict],
    archetype_meta: dict,
    movement_meta: dict,
    hypotheses: dict,
) -> dict[str, Any]:
    files = {}
    for name in (
        "matches.csv",
        "teams.csv",
        "team_edges.csv",
        "team_centroids.csv",
        "metadata.json",
    ):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            files[name] = os.path.getsize(path)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "map_name": args.map,
        "game_mode": args.mode,
        "weight_mode": args.weight_mode,
        "early_window_sec": args.early_window_sec,
        "centroid_sample_sec": args.centroid_sample_sec,
        "landing_offset_s": LANDING_OFFSET_S,
        "movement_window_s": MOVEMENT_WINDOW_S,
        "n_matches": len(match_rows),
        "n_teams": len(team_rows),
        "n_team_edges": len(edge_rows),
        "n_team_centroids": len(centroid_rows),
        "out_dir": out_dir,
        "files": files,
        "archetype_meta": archetype_meta,
        "movement_meta": movement_meta,
        "hypotheses": hypotheses,
        "notes": [
            "Source telemetry is reused from PostgreSQL; .pt graph files are not regenerated.",
            "team_edges.csv has both full-match and early-window team combat edges.",
            "teams.csv includes provisional archetype/movement labels for analysis; labels can be revised after hypothesis discussion.",
        ],
    }


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    pt_paths = list_matches(args.map, args.mode, limit=None)
    if args.offset:
        pt_paths = pt_paths[args.offset :]
    if not args.all and args.n is not None:
        pt_paths = pt_paths[: args.n]
    if not pt_paths:
        raise RuntimeError(f"No .pt files found for map={args.map} mode={args.mode}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"{args.map}_{args.mode}_{stamp}"
    out_dir = os.path.abspath(args.out or os.path.join(DEFAULT_OUT_ROOT, default_name))
    os.makedirs(out_dir, exist_ok=True)

    match_rows: list[dict[str, Any]] = []
    team_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    centroid_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    total = len(pt_paths)
    for i, pt_path in enumerate(pt_paths, start=1):
        if args.verbose:
            print(f"[{i}/{total}] {pt_path}", flush=True)
        try:
            m_row, t_rows, e_rows, c_rows = _process_match(
                pt_path,
                weight_mode=args.weight_mode,
                early_window_sec=args.early_window_sec,
                centroid_sample_sec=args.centroid_sample_sec,
                include_centroids=not args.no_centroids,
            )
        except Exception as exc:
            failures.append({"pt_path": pt_path, "error": repr(exc)})
            print(f"[warn] skipped {pt_path}: {exc!r}", flush=True)
            if not args.keep_going:
                raise
            continue
        match_rows.append(m_row)
        team_rows.extend(t_rows)
        edge_rows.extend(e_rows)
        centroid_rows.extend(c_rows)

    archetype_meta = assign_archetype_labels(team_rows)
    movement_meta = assign_movement_labels(team_rows)
    hypotheses = {
        "h1": compute_h1_archetype(team_rows),
        "h2": compute_h2_battle_center(team_rows),
        "h3": compute_h3_movement(team_rows),
    }

    _write_csv(os.path.join(out_dir, "matches.csv"), match_rows)
    _write_csv(os.path.join(out_dir, "teams.csv"), team_rows)
    _write_csv(os.path.join(out_dir, "team_edges.csv"), edge_rows)
    _write_csv(os.path.join(out_dir, "team_centroids.csv"), centroid_rows)

    metadata = _build_metadata(
        args=args,
        out_dir=out_dir,
        match_rows=match_rows,
        team_rows=team_rows,
        edge_rows=edge_rows,
        centroid_rows=centroid_rows,
        archetype_meta=archetype_meta,
        movement_meta=movement_meta,
        hypotheses=hypotheses,
    )
    metadata["failures"] = failures
    _write_json(os.path.join(out_dir, "metadata.json"), metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build analysis CSV datasets from existing PUBG DB and .pt graphs."
    )
    parser.add_argument("--map", default=DEFAULT_MAP, help="PUBG map code")
    parser.add_argument("--mode", default=DEFAULT_MODE, help="Game mode")
    parser.add_argument("--n", type=int, default=10, help="Number of .pt matches to process")
    parser.add_argument(
        "--all", action="store_true", help="Process all available .pt matches for the map/mode"
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="Skip this many .pt files before processing"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory; default uses timestamped analysis_datasets dir",
    )
    parser.add_argument("--weight-mode", choices=["dmg", "count", "binary"], default="dmg")
    parser.add_argument("--early-window-sec", type=int, default=300)
    parser.add_argument("--centroid-sample-sec", type=int, default=30)
    parser.add_argument(
        "--no-centroids", action="store_true", help="Do not write team_centroids.csv"
    )
    parser.add_argument(
        "--keep-going", action="store_true", help="Skip failed matches instead of stopping"
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    metadata = build_dataset(parse_args())
    print(
        json.dumps(
            {
                "out_dir": metadata["out_dir"],
                "n_matches": metadata["n_matches"],
                "n_teams": metadata["n_teams"],
                "n_team_edges": metadata["n_team_edges"],
                "n_team_centroids": metadata["n_team_centroids"],
                "files": metadata["files"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
