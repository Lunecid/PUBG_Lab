"""
V2 unified feature view (FEATURE_DESIGN_V2 §12 Step 4).

stdb.v_team_phase_feature_v2 joins every representation family on
(match_id, team_id, phase_idx) so the A0-A11 / B0-B7 ablations can read one
view:

    long view (F4 conventional + F5 STDB spatial + labels)
  + F0 base-exposed columns (weather, start/end alive, red/black, position_rows,
    first_damage) not surfaced by the long view
  + F0 extras (zone centers, team centroid, raw kinematics)  [team_phase_f0_extra]
  + F6 network role                                          [team_phase_network_features]
  + F7 habitat pressure                                      [team_phase_habitat_features]
  + O6 Z-affordance sensitivity                              [team_phase_z_affordance_features]

F6/F7/O6Z column lists are read from the manifest (single source of truth) so
the view never drifts from the ablation contract. F0 raw pass-2 columns
(assists/suicide/landing) are intentionally absent until built.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[5]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

import json
from pathlib import Path

from pubg_lab.data.stdb.connection.conn import connect

MANIFEST = Path("src/pubg_lab/survival/features/manifests/pubg_representation_v2.json")

# F0 columns the long view does not expose (live in base tables)
F0_BASE_EXPOSE = [
    ("g2", "weather_id"),
    ("g2", "camera_view_type"),
    ("g2", "start_alive_teams"),
    ("g2", "end_alive_teams"),
    ("g2", "start_alive_players"),
    ("g2", "end_alive_players"),
    ("g2", "poison_radius_delta_m"),
    ("g2", "red_zone_active_ratio"),
    ("g2", "black_zone_active_ratio"),
    ("s2", "position_rows"),
    ("c2", "first_damage_dealt_s"),
    ("c2", "first_damage_taken_s"),
]
# F0 extras (team_phase_f0_extra) — zone centers, centroid, kinematics
F0_EXTRA = [
    "safe_center_x_start",
    "safe_center_y_start",
    "safe_center_x_end",
    "safe_center_y_end",
    "poison_center_x_start",
    "poison_center_y_start",
    "poison_center_x_end",
    "poison_center_y_end",
    "team_centroid_x_start",
    "team_centroid_y_start",
    "team_centroid_x_end",
    "team_centroid_y_end",
    "team_displacement_raw_m",
    "team_avg_speed_raw_mps",
]


def build_sql() -> str:
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    f6 = man["families"]["f6_network_phase_role_tier1"]["columns"]
    f7 = man["families"]["f7_habitat_pressure_tier1"]["columns"]
    z = man["families"].get("o6_z_affordance_tier1", {}).get("columns", [])

    cols = [f"{a}.{c}" for a, c in F0_BASE_EXPOSE]
    cols += [f'f0."{c}"' for c in F0_EXTRA]
    cols += [f'n."{c}"' for c in f6]
    cols += [f'h."{c}"' for c in f7]
    cols += [f'z."{c}"' for c in z]
    extra = ",\n    ".join(cols)

    return f"""
CREATE OR REPLACE VIEW stdb.v_team_phase_feature_v2 AS
SELECT
    long.*,
    {extra}
FROM stdb.v_team_phase_feature_long long
LEFT JOIN stdb.match_phase_global g2
       ON g2.match_id = long.match_id AND g2.phase_idx = long.phase_idx
LEFT JOIN stdb.team_phase_state s2
       ON s2.match_id = long.match_id AND s2.team_id = long.team_id AND s2.phase_idx = long.phase_idx
LEFT JOIN stdb.team_phase_combat c2
       ON c2.match_id = long.match_id AND c2.team_id = long.team_id AND c2.phase_idx = long.phase_idx
LEFT JOIN stdb.team_phase_f0_extra f0
       ON f0.match_id = long.match_id AND f0.team_id = long.team_id AND f0.phase_idx = long.phase_idx
LEFT JOIN stdb.team_phase_network_features n
       ON n.match_id = long.match_id AND n.team_id = long.team_id AND n.phase_idx = long.phase_idx
LEFT JOIN stdb.team_phase_habitat_features h
       ON h.match_id = long.match_id AND h.team_id = long.team_id AND h.phase_idx = long.phase_idx
LEFT JOIN stdb.team_phase_z_affordance_features z
       ON z.match_id = long.match_id AND z.team_id = long.team_id AND z.phase_idx = long.phase_idx
"""


def apply() -> None:
    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(build_sql())


if __name__ == "__main__":
    apply()
    print("v_team_phase_feature_v2 applied")
