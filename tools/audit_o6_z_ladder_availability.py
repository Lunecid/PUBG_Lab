"""Audit availability of the planned PUBG O6 Z-ladder features.

This script does not train a model.  It checks whether the intended Z1-Z3
spatial-affordance proxies are already available on the V3 phase tensor surface
or whether ETL/tensor mapping is required before running an OM6Z sensitivity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from model_stgcn.v3.cohorts import REQUIRED_COHORTS, CohortSpec
from model_stgcn.v3.spec import (
    O6_SPATIAL_PROXY_FEATURES,
    O6_Z_AFFORDANCE_FEATURES,
    expected_phase_feature_names,
)
from stdb.conn import get_cur


DEFAULT_OUT_DIR = Path("model_stgcn/checkpoints/br_om_ladder_required_v1/o6_z_ladder_availability")

Z1_TARGET_FEATURES = (
    "team_z_mean_norm",
    "team_z_percentile_alive",
    "team_z_range_norm",
)
Z2_TARGET_FEATURES = (
    "contact_delta_z_mean_norm",
    "contact_delta_z_positive_ratio",
    "high_ground_contact_advantage",
)
Z3_TARGET_FEATURES = (
    "high_ground_x_boundary_pressure",
    "z_percentile_x_hab_crowding_pressure",
    "contact_delta_z_x_outside_safe_ratio",
)

SOURCE_TABLES = (
    "v_team_phase_feature_long",
    "v_team_phase_feature_v2",
    "team_snapshot_features",
    "team_phase_edge_cache",
    "team_phase_network_features",
    "team_phase_habitat_features",
    "team_phase_z_affordance_features",
)

SPATIAL_NAME_RE = re.compile(r"(?:^|_)(?:pos_z|team_z|z_|delta_z|height|elev|altitude|ground)(?:_|$)")

TARGET_SOURCE_COLUMNS = {
    "team_z_mean_norm": "team_z_mean_m",
    "team_z_percentile_alive": "team_z_percentile_alive",
    "team_z_range_norm": "team_z_range_m",
    "contact_delta_z_mean_norm": "contact_delta_z_mean_m",
    "contact_delta_z_positive_ratio": "contact_delta_z_positive_ratio",
    "high_ground_contact_advantage": "high_ground_contact_advantage",
    "high_ground_x_boundary_pressure": "high_ground_x_boundary_pressure",
    "z_percentile_x_hab_crowding_pressure": "z_percentile_x_hab_crowding_pressure",
    "contact_delta_z_x_outside_safe_ratio": "contact_delta_z_x_outside_safe_ratio",
}


@dataclass
class FeatureAvailabilityRow:
    block: str
    feature: str
    source_column: str
    in_v3_phase_tensor: bool
    in_phase_long_columns: bool
    in_phase_v2_columns: bool
    in_snapshot_columns: bool
    in_edge_cache_columns: bool
    in_z_affordance_columns: bool
    status: str


@dataclass
class CohortSnapshotZRow:
    cohort: str
    cohort_slug: str
    map: str
    perspective: str
    team_size: str
    n_rows: int
    n_matches: int
    n_teams: int
    nonnull_pos_z: int
    pos_z_mean: float
    pos_z_std: float
    pos_z_min: float
    pos_z_p05: float
    pos_z_median: float
    pos_z_p95: float
    pos_z_max: float
    degenerate: bool


def _write_csv(path: Path, rows: Iterable[object]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _safe_float(value) -> float:
    if value is None:
        return math.nan
    return float(value)


def _columns_by_table() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with get_cur(dict_cursor=True) as (_conn, cur):
        cur.execute(
            """
            SELECT table_name, column_name
              FROM information_schema.columns
             WHERE table_schema = 'stdb'
               AND table_name = ANY(%s)
             ORDER BY table_name, ordinal_position
            """,
            (list(SOURCE_TABLES),),
        )
        for row in cur.fetchall():
            out.setdefault(row["table_name"], set()).add(row["column_name"])
    for table in SOURCE_TABLES:
        out.setdefault(table, set())
    return out


def _feature_rows(columns: dict[str, set[str]]) -> list[FeatureAvailabilityRow]:
    phase_names = set(expected_phase_feature_names(include_v2=True))
    source_pos_z = "pos_z" in columns["team_snapshot_features"]
    rows: list[FeatureAvailabilityRow] = []
    targets = (
        [("Z1", f) for f in Z1_TARGET_FEATURES]
        + [("Z2", f) for f in Z2_TARGET_FEATURES]
        + [("Z3", f) for f in Z3_TARGET_FEATURES]
    )
    for block, feature in targets:
        source_column = TARGET_SOURCE_COLUMNS[feature]
        in_tensor = feature in phase_names
        in_long = source_column in columns["v_team_phase_feature_long"]
        in_v2 = source_column in columns["v_team_phase_feature_v2"]
        in_snap = source_column in columns["team_snapshot_features"]
        in_edge = source_column in columns["team_phase_edge_cache"]
        in_z = source_column in columns["team_phase_z_affordance_features"]
        if in_tensor and in_z:
            status = "ready_in_tensor"
        elif in_tensor:
            status = "tensor_ready_missing_source_table"
        elif in_long or in_v2:
            status = "db_column_needs_tensor_mapping"
        elif in_snap or in_edge or source_pos_z:
            status = "source_signal_needs_phase_etl"
        else:
            status = "missing"
        rows.append(
            FeatureAvailabilityRow(
                block=block,
                feature=feature,
                source_column=source_column,
                in_v3_phase_tensor=in_tensor,
                in_phase_long_columns=in_long,
                in_phase_v2_columns=in_v2,
                in_snapshot_columns=in_snap,
                in_edge_cache_columns=in_edge,
                in_z_affordance_columns=in_z,
                status=status,
            )
        )
    return rows


def _snapshot_z_row(spec: CohortSpec) -> CohortSnapshotZRow:
    with get_cur(dict_cursor=True) as (_conn, cur):
        cur.execute(
            """
            SELECT
                count(*)::int AS n_rows,
                count(DISTINCT s.match_id)::int AS n_matches,
                count(DISTINCT (s.match_id, s.team_id))::int AS n_teams,
                count(s.pos_z)::int AS nonnull_pos_z,
                avg(s.pos_z)::float AS pos_z_mean,
                stddev_pop(s.pos_z)::float AS pos_z_std,
                min(s.pos_z)::float AS pos_z_min,
                percentile_cont(0.05) WITHIN GROUP (ORDER BY s.pos_z)::float AS pos_z_p05,
                percentile_cont(0.50) WITHIN GROUP (ORDER BY s.pos_z)::float AS pos_z_median,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY s.pos_z)::float AS pos_z_p95,
                max(s.pos_z)::float AS pos_z_max
              FROM stdb.team_snapshot_features s
              JOIN stdb.matches m ON m.match_id = s.match_id
             WHERE m.map_name = %s
               AND m.game_mode = %s
               AND s.phase_idx IS NOT NULL
               AND s.phase_idx BETWEEN 1 AND 7
               AND s.alive_n > 0
            """,
            (spec.map_name, spec.game_mode),
        )
        row = cur.fetchone()
    std = _safe_float(row["pos_z_std"])
    nonnull = int(row["nonnull_pos_z"] or 0)
    degenerate = bool(nonnull == 0 or (math.isfinite(std) and std < 1e-8))
    return CohortSnapshotZRow(
        cohort=spec.label,
        cohort_slug=spec.slug,
        map=spec.map_label,
        perspective=spec.perspective,
        team_size=spec.team_size,
        n_rows=int(row["n_rows"] or 0),
        n_matches=int(row["n_matches"] or 0),
        n_teams=int(row["n_teams"] or 0),
        nonnull_pos_z=nonnull,
        pos_z_mean=_safe_float(row["pos_z_mean"]),
        pos_z_std=std,
        pos_z_min=_safe_float(row["pos_z_min"]),
        pos_z_p05=_safe_float(row["pos_z_p05"]),
        pos_z_median=_safe_float(row["pos_z_median"]),
        pos_z_p95=_safe_float(row["pos_z_p95"]),
        pos_z_max=_safe_float(row["pos_z_max"]),
        degenerate=degenerate,
    )


def _spatial_columns(columns: dict[str, set[str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for table, names in columns.items():
        out[table] = sorted(name for name in names if SPATIAL_NAME_RE.search(name))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    columns = _columns_by_table()
    feature_rows = _feature_rows(columns)
    cohort_rows = [_snapshot_z_row(spec) for spec in REQUIRED_COHORTS]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "o6_z_ladder_feature_availability.csv", feature_rows)
    _write_csv(args.out_dir / "o6_z_ladder_snapshot_pos_z_by_cohort.csv", cohort_rows)

    ready = sum(1 for row in feature_rows if row.status == "ready_in_tensor")
    source_signal = bool("pos_z" in columns["team_snapshot_features"])
    old_o6 = list(O6_SPATIAL_PROXY_FEATURES)
    intended_o6z = list(O6_Z_AFFORDANCE_FEATURES)
    summary = {
        "out_dir": str(args.out_dir),
        "target_feature_count": len(feature_rows),
        "target_features_ready_in_v3_phase_tensor": ready,
        "target_features_missing_or_need_etl": len(feature_rows) - ready,
        "source_snapshot_pos_z_available": source_signal,
        "snapshot_pos_z_degenerate_cohorts": [row.cohort for row in cohort_rows if row.degenerate],
        "current_om6_spatial_proxy_features": old_o6,
        "intended_om6z_spatial_affordance_features": intended_o6z,
        "spatial_like_columns_by_table": _spatial_columns(columns),
        "verdict": (
            "ETL_REQUIRED_BEFORE_OM6Z"
            if ready < len(feature_rows)
            else "READY_FOR_OM6Z_PRESET"
        ),
    }
    (args.out_dir / "o6_z_ladder_availability_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
