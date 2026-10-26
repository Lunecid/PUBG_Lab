"""Export one Caldera player trajectory to JSON for Blender overlays."""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pubg_lab.data.stdb.connection.conn import connect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--source-artifact-id", default="caldera_minimal_layers_breadcrumbs_usd")
    parser.add_argument("--sample-t-min", type=float)
    parser.add_argument("--sample-t-max", type=float)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def fetch_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    filters = [
        "br_match_id = %s",
        "br_agent_id = %s",
        "source_artifact_id = %s",
        "pos_x_m IS NOT NULL",
        "pos_y_m IS NOT NULL",
        "pos_z_m IS NOT NULL",
    ]
    params: list[Any] = [args.match_id, args.agent_id, args.source_artifact_id]
    if args.sample_t_min is not None:
        filters.append("sample_t >= %s")
        params.append(args.sample_t_min)
    if args.sample_t_max is not None:
        filters.append("sample_t <= %s")
        params.append(args.sample_t_max)

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT sample_t, pos_x_m, pos_y_m, pos_z_m, alive_state
              FROM br_ground.trajectory_samples
             WHERE {" AND ".join(filters)}
             ORDER BY sample_t
            """,
            params,
        )
        return [
            {
                "t": float(sample_t),
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "alive_state": alive_state,
            }
            for sample_t, x, y, z, alive_state in cur.fetchall()
        ]


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = fetch_rows(args)
    if len(rows) < 2:
        raise SystemExit("Need at least two trajectory rows.")
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    xs = [row["x"] for row in rows]
    ys = [row["y"] for row in rows]
    zs = [row["z"] for row in rows]
    print(f"trajectory_rows={len(rows)}")
    print(f"output={args.output}")
    print(f"time_range_s={[rows[0]['t'], rows[-1]['t']]}")
    print(
        f"bounds={{'x': {[min(xs), max(xs)]}, 'y': {[min(ys), max(ys)]}, 'z': {[min(zs), max(zs)]}}}"
    )


if __name__ == "__main__":
    main()
