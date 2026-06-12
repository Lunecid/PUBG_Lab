"""
model_stgcn/infer.py
====================
Run a trained STGCN on a single match and emit a JSON in the same shape
that `visualize.py` consumes (compatible with simulation_result_*.json).

The model is invoked once over the T_WIN snapshots used during training.  The
predicted rank and Top-K probabilities are then *broadcast* to every snapshot
in the match (the 3D viewer overlays them across the entire timeline) so that
the viewer can still show team movement and zone shrinking continuously.

Per-frame `is_dying` is derived from the STDB `is_alive` transition between
snapshots, and `dying_teams` lists the teams that dropped at that step.

Output JSON shape (matches visualize.py expectations):
{
  "meta":   {match_id, map_name, map_size, model, …},
  "frames": [
      { step, elapsed, phase, n_alive,
        zone: {safe_x,safe_y,safe_radius,poison_x,poison_y,poison_radius,safe_area},
        teams: [{team_idx,x,y,z,hazard,hazard_rank,is_dying,final_rank}, …],
        dying_teams: [team_idx, …]
      }, …
  ]
}
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch

from model_stgcn.stgcn import STGCN
from stdb.conn import connect
from stdb.queries import ARENA_SIZE_M, fetch_match_tensors

logger = logging.getLogger("model_stgcn.infer")


def _load_model(checkpoint: str, device: str = "cpu"):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    args_d = ckpt.get("args", {})
    model = STGCN(
        in_features=8,
        hidden=args_d.get("hidden", 64),
        num_blocks=args_d.get("num_blocks", 2),
        dropout=args_d.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model, ckpt


def _query_match_timeline(match_id: str) -> dict:
    """Fetch all the frame-level info needed for the JSON, straight from STDB."""
    conn = connect()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT duration_s, map_name FROM stdb.matches WHERE match_id = %s",
            (match_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"match {match_id} not in stdb")
        duration_s, map_name = int(row[0]), row[1]

        cur.execute(
            """
            SELECT team_id, final_rank, num_players
              FROM stdb.teams
             WHERE match_id = %s
             ORDER BY coalesce(final_rank, 9999)
            """,
            (match_id,),
        )
        teams_meta = [
            {"team_id": tid, "final_rank": rnk, "num_players": np_}
            for (tid, rnk, np_) in cur.fetchall()
        ]
        final_rank_map = {t["team_id"]: t["final_rank"] for t in teams_meta}

        cur.execute(
            """
            SELECT snapshot_t, team_id,
                   ST_X(pos), ST_Y(pos), ST_Z(pos), is_alive
              FROM stdb.moving_objects
             WHERE match_id = %s AND agent_kind = 'team'
             ORDER BY snapshot_t, team_id
            """,
            (match_id,),
        )
        snaps: dict[int, list[dict]] = {}
        for st, tid, x, y, z, alive in cur.fetchall():
            snaps.setdefault(int(st), []).append(
                {"team_id": tid, "x": float(x), "y": float(y), "z": float(z), "alive": bool(alive)}
            )

        cur.execute(
            """
            SELECT region_kind, snapshot_t,
                   ST_X(center_pos), ST_Y(center_pos), radius_m, phase_idx
              FROM stdb.dynamic_regions
             WHERE match_id = %s
            """,
            (match_id,),
        )
        regions: dict[int, dict[str, dict]] = {}
        for kind, st, cx, cy, r, phase in cur.fetchall():
            regions.setdefault(int(st), {})[kind] = {
                "x": float(cx), "y": float(cy), "r": float(r),
                "phase": int(phase) if phase is not None else 0,
            }

        return {
            "duration_s": duration_s,
            "map_name": map_name,
            "teams_meta": teams_meta,
            "final_rank_map": final_rank_map,
            "snaps": snaps,
            "regions": regions,
        }
    finally:
        cur.close(); conn.close()


def infer_match(match_id: str, checkpoint: str, *, device: str = "cpu") -> dict:
    model, ckpt = _load_model(checkpoint, device=device)

    mt = fetch_match_tensors(match_id)
    if mt is None:
        raise RuntimeError(f"match {match_id} not in stdb")

    X = torch.from_numpy(mt.X).unsqueeze(0).to(device)
    A = torch.from_numpy(mt.A_t).unsqueeze(0).to(device)
    with torch.no_grad():
        rp, cp = model(X, A)
    rank_pred = (rp[0].cpu().numpy() * 25.0)               # (N_MAX,)
    top_probs = torch.sigmoid(cp[0]).cpu().numpy()         # (N_MAX, 2)

    # Map: real team_id -> index in mt.team_ids (only the first `mask.sum()` are real)
    n_real = int(mt.mask.sum())
    team_id_to_idx = {tid: i for i, tid in enumerate(mt.team_ids[:n_real])}

    # hazard_rank: 1 = highest hazard (most likely to die soon)
    # higher predicted-rank value = worse final standing = higher hazard
    real_ranks = rank_pred[:n_real]
    sorted_by_hazard = np.argsort(-real_ranks)  # descending: highest hazard first
    hazard_rank = {mt.team_ids[idx]: pos + 1 for pos, idx in enumerate(sorted_by_hazard)}

    timeline = _query_match_timeline(match_id)
    snaps = timeline["snaps"]
    regions = timeline["regions"]
    final_rank_map = timeline["final_rank_map"]

    snapshot_ts = sorted(snaps.keys())
    prev_alive: set[int] | None = None
    frames: list[dict] = []
    for step, st in enumerate(snapshot_ts):
        team_rows: list[dict] = []
        cur_alive: set[int] = set()
        for entry in snaps[st]:
            tid = entry["team_id"]
            if tid not in final_rank_map:
                continue
            if entry["alive"]:
                cur_alive.add(tid)
            idx = team_id_to_idx.get(tid)
            hazard = float(top_probs[idx, 0]) if idx is not None else 0.0  # P(top5) complement is "danger"
            danger = 1.0 - hazard
            is_dying = (prev_alive is not None) and (tid in prev_alive) and (tid not in cur_alive)
            team_rows.append({
                "team_idx": tid,
                "x": float(np.clip(entry["x"] / ARENA_SIZE_M, 0.0, 1.0)),
                "y": float(np.clip(entry["y"] / ARENA_SIZE_M, 0.0, 1.0)),
                "x_m": entry["x"],
                "y_m": entry["y"],
                "z_m": entry["z"],
                "alive": bool(entry["alive"]),
                "hazard": float(danger),
                "hazard_rank": int(hazard_rank.get(tid, 99)),
                "is_dying": bool(is_dying),
                "final_rank": int(final_rank_map[tid]) if final_rank_map[tid] is not None else 99,
            })

        dying_teams = [t["team_idx"] for t in team_rows if t["is_dying"]]
        regs = regions.get(st, {})
        safe = regs.get("safe", {"x": 0.0, "y": 0.0, "r": 0.0, "phase": 0})
        poison = regs.get("poison", {"x": 0.0, "y": 0.0, "r": 0.0, "phase": 0})

        frames.append({
            "step": step,
            "elapsed": float(st),
            "phase": int(safe.get("phase") or poison.get("phase") or 0),
            "n_alive": int(sum(1 for t in team_rows if t["alive"])),
            "zone": {
                "safe_x": safe["x"],
                "safe_y": safe["y"],
                "safe_radius": safe["r"],
                "poison_x": poison["x"],
                "poison_y": poison["y"],
                "poison_radius": poison["r"],
                "safe_area": math.pi * (safe["r"] ** 2),
            },
            "teams": team_rows,
            "dying_teams": dying_teams,
        })
        prev_alive = cur_alive

    return {
        "meta": {
            "match_id": match_id,
            "map_name": timeline["map_name"],
            "map_size": ARENA_SIZE_M,
            "model": "stgcn-yu-dyn",
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        },
        "frames": frames,
    }


def build_elevation_grid(
    match_id: str | None = None,
    *,
    map_size_m: float = 8160.0,
    gs: int = 80,
    z_clip_m: float = 500.0,
    map_name: str = "Baltic_Main",
) -> dict | None:
    """Aggregate stdb.moving_objects pos_z into a coarse 80×80 elevation grid.

    Erangel terrain spans roughly 0–400m, so we hard-clip outliers above
    `z_clip_m` to drop airborne positions (TransportAircraft sits at ~1502m)
    that would otherwise dominate cell averages.

    When `match_id` is None the grid is aggregated across every match of
    `map_name` currently in stdb — this gives far denser coverage and a
    much more recognisable terrain than the player traces of a single match.

    Returns dict {grid, elev_min_m, elev_max_m, cell_size_m, grid_size} that
    matches the shape consumed by visualize.py's heatmap renderer.
    Returns None if no positions found.
    """
    conn = connect()
    cur = conn.cursor()
    try:
        if match_id is None:
            cur.execute(
                """
                SELECT ST_X(mo.pos), ST_Y(mo.pos), ST_Z(mo.pos)
                  FROM stdb.moving_objects mo
                  JOIN stdb.matches m ON m.match_id = mo.match_id
                 WHERE mo.agent_kind = 'player'
                   AND m.map_name = %s
                   AND ST_Z(mo.pos) BETWEEN -50 AND %s
                """,
                (map_name, z_clip_m),
            )
        else:
            cur.execute(
                """
                SELECT ST_X(pos), ST_Y(pos), ST_Z(pos)
                  FROM stdb.moving_objects
                 WHERE match_id = %s
                   AND agent_kind = 'player'
                   AND ST_Z(pos) BETWEEN -50 AND %s
                """,
                (match_id, z_clip_m),
            )
        rows = cur.fetchall()
    finally:
        cur.close(); conn.close()

    if not rows:
        return None

    cell = map_size_m / gs
    sumg = [[0.0] * gs for _ in range(gs)]
    cnt = [[0] * gs for _ in range(gs)]
    z_min, z_max = float("inf"), float("-inf")
    for x, y, z in rows:
        i = max(0, min(gs - 1, int(float(x) / cell)))
        j = max(0, min(gs - 1, int(float(y) / cell)))
        sumg[j][i] += float(z)
        cnt[j][i] += 1
        if z < z_min:
            z_min = float(z)
        if z > z_max:
            z_max = float(z)

    grid = [[(sumg[j][i] / cnt[j][i]) if cnt[j][i] > 0 else 0.0 for i in range(gs)] for j in range(gs)]
    # smooth empty cells with a 3×3 mean of nonzero neighbours so the mesh isn't
    # full of holes; do two passes.
    for _ in range(2):
        new = [row[:] for row in grid]
        for j in range(gs):
            for i in range(gs):
                if cnt[j][i] == 0:
                    acc = 0.0
                    n = 0
                    for dj in (-1, 0, 1):
                        for di in (-1, 0, 1):
                            jj, ii = j + dj, i + di
                            if 0 <= jj < gs and 0 <= ii < gs and grid[jj][ii] != 0.0:
                                acc += grid[jj][ii]
                                n += 1
                    if n > 0:
                        new[j][i] = acc / n
        grid = new

    if z_min == float("inf"):
        z_min, z_max = 0.0, 400.0
    span = max(1.0, z_max - z_min)
    norm = [[max(0.0, min(1.0, (grid[j][i] - z_min) / span)) for i in range(gs)] for j in range(gs)]
    return {
        "grid":         norm,
        "elev_min_m":   z_min,
        "elev_max_m":   z_max,
        "cell_size_m":  cell,
        "grid_size":    gs,
    }


def to_visualize_compat(infer_result: dict, *, elevation: dict | None = None) -> dict:
    """Convert STDB-style infer JSON → visualize.py-compatible payload.

    The viewer expects:
      • frames[i].nodes  — pid-indexed array (same shape every frame),
      • frames[i].ally / .enc — edge lists (we leave both empty),
      • frames[i].zone — {scx,scy,sr,pcx,pcy,pr,alive},
      • meta : match_id, total_players, total_teams, map_name, map_size,
               z_min, z_max, heatmaps (optional), terrain (optional).
    Each team is rendered as a single "player" node so the existing 3D viewer
    works without modification.
    """
    meta = infer_result["meta"]
    frames_in = infer_result["frames"]

    # Determine a stable list of team ids across all frames
    all_team_ids: list[int] = []
    seen: set[int] = set()
    for fr in frames_in:
        for t in fr["teams"]:
            if t["team_idx"] not in seen:
                seen.add(t["team_idx"])
                all_team_ids.append(t["team_idx"])
    pid_of = {tid: i for i, tid in enumerate(all_team_ids)}
    total_players = len(all_team_ids)

    # z range from team z values for the viewer's Z_SCALE
    z_vals = [t["z_m"] for fr in frames_in for t in fr["teams"]]
    z_min = float(min(z_vals)) if z_vals else 0.0
    z_max = float(max(z_vals)) if z_vals else 400.0

    out_frames: list[dict] = []
    for fr in frames_in:
        nodes = [
            {
                "pid": pid, "alive": 0,
                "x": 0.0, "y": 0.0, "z": 0.0, "hp": 0.0,
                "sd": 0, "sb": 0, "is": 0,
                "pd": 0, "pb": 0, "ip": 0,
                "vs": 0, "iv": 0,
                "dd": 0, "dt": 0,
                "t": all_team_ids[pid],
                "aid": f"team_{all_team_ids[pid]}",
                "tid": str(all_team_ids[pid]),
            }
            for pid in range(total_players)
        ]
        for t in fr["teams"]:
            pid = pid_of[t["team_idx"]]
            nodes[pid].update({
                "alive": 1 if t["alive"] else 0,
                "x": t["x_m"], "y": t["y_m"], "z": t["z_m"],
                "hp": 1.0 if t["alive"] else 0.0,
                "is": 1,
                "ip": 1,
                "dd": float(t.get("hazard", 0.0)),       # piggy-back hazard in dd
                "dt": float(t.get("hazard_rank", 99)),   # and rank in dt
            })

        zone_in = fr["zone"]
        out_frames.append({
            "t": fr["elapsed"],
            "nodes": nodes,
            "ally": [],
            "enc":  [],
            "zone": {
                "scx": zone_in["safe_x"], "scy": zone_in["safe_y"],
                "sr":  zone_in["safe_radius"],
                "pcx": zone_in["poison_x"], "pcy": zone_in["poison_y"],
                "pr":  zone_in["poison_radius"],
                "alive": fr["n_alive"],
            },
        })

    heatmaps = None
    terrain = None
    if elevation is not None:
        heatmaps = {
            "grid_size":  elevation["grid_size"],
            "cell_size_m": elevation["cell_size_m"],
            "elev_min_m": elevation["elev_min_m"],
            "elev_max_m": elevation["elev_max_m"],
            "elevation":  elevation["grid"],
            "density":    [[0.0] * elevation["grid_size"] for _ in range(elevation["grid_size"])],
            "combat":     [[0.0] * elevation["grid_size"] for _ in range(elevation["grid_size"])],
        }
        terrain = {
            "grid":        elevation["grid"],
            "elev_min_m":  elevation["elev_min_m"],
            "elev_max_m":  elevation["elev_max_m"],
            "flip_y":      False,
            "rot_deg":     0,
        }

    return {
        "match_id":      meta["match_id"],
        "total_players": total_players,
        "total_teams":   total_players,
        "map_name":      meta.get("map_name", ""),
        "map_size":      meta.get("map_size", 8160.0),
        "z_min":         z_min,
        "z_max":         z_max,
        "frames":        out_frames,
        "heatmaps":      heatmaps,
        "terrain":       terrain,
        "model":         meta.get("model", "stgcn-yu-dyn"),
    }


def emit_html(viz_json: dict, out_path: Path) -> None:
    """Inject the viz JSON into visualize.HTML and write a standalone .html file."""
    from visualize import HTML  # noqa: WPS433 (local import to avoid torch deps at import time)

    js = json.dumps(viz_json, separators=(",", ":"))
    html = HTML.replace("__JSON_DATA__", js).replace(
        "__TOTAL__", str(viz_json["total_players"])
    )
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--checkpoint", default="model_stgcn/checkpoints/best.pt")
    ap.add_argument("--out-json", help="Path for STDB-style JSON")
    ap.add_argument("--out-html", help="Path for visualize-compat HTML")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--no-elevation",
        action="store_true",
        help="Skip elevation-grid generation (faster, flat terrain in HTML)",
    )
    ap.add_argument(
        "--out-elevation-png",
        help="Optional path to dump the aggregated elevation as a grayscale PNG",
    )
    args = ap.parse_args()
    if not (args.out_json or args.out_html):
        ap.error("at least one of --out-json / --out-html is required")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = infer_match(args.match_id, args.checkpoint, device=args.device)
    logger.info("STDB infer ok  frames=%d", len(result["frames"]))

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result))
        logger.info("wrote %s", out_json)

    if args.out_html:
        # Aggregate over every stdb match on the same map for a dense, accurate
        # heightfield (a single match's player trace is far too sparse).
        elevation = None if args.no_elevation else build_elevation_grid(
            match_id=None,
            map_name=result["meta"].get("map_name", "Baltic_Main"),
        )
        if elevation is None:
            logger.info("elevation grid: none (flat terrain fallback)")
        else:
            logger.info(
                "elevation grid: %d×%d  z=%.0fm…%.0fm",
                elevation["grid_size"], elevation["grid_size"],
                elevation["elev_min_m"], elevation["elev_max_m"],
            )
            if args.out_elevation_png:
                _export_elevation_png(elevation, Path(args.out_elevation_png))
                logger.info("wrote elevation preview %s", args.out_elevation_png)
        viz = to_visualize_compat(result, elevation=elevation)
        out_html = Path(args.out_html)
        out_html.parent.mkdir(parents=True, exist_ok=True)
        emit_html(viz, out_html)
        logger.info("wrote %s", out_html)


def _export_elevation_png(elevation: dict, out_path: Path) -> None:
    """Write the normalised elevation grid as an 8-bit grayscale PNG.

    Useful for eyeballing how close the player-trace heightfield is to the real
    Erangel topography (compare against the map image in the repo).
    """
    import struct
    import zlib

    grid = elevation["grid"]
    gs = elevation["grid_size"]
    raw = bytearray()
    for j in range(gs - 1, -1, -1):  # flip Y so map north is up on the PNG
        raw.append(0)
        for i in range(gs):
            v = max(0.0, min(1.0, grid[j][i]))
            raw.append(int(v * 255))

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", gs, gs, 8, 0, 0, 0, 0)  # 8-bit grayscale
    idat = zlib.compress(bytes(raw), 9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


if __name__ == "__main__":
    main()
