# pubg_lab.data.stdb.pipelines

Explicit orchestration.

| Component | Purpose |
|---|---|
| `base_stage.py` | Transient re-materialisation of the base STDB rows for ONE match. |
| `build_available.py` | stdb/build_available.py |
| `etl.py` | stdb/etl.py |
| `f0_feature_etl.py` | F0 official-raw extras (Tier-1: stdb-base derivable) — FEATURE_DESIGN_V2 §4. |
| `habitat_feature_etl.py` | F7 Tier-1 habitat-inspired pressure feature builder (FEATURE_DESIGN_V2 §8). |
| `network_feature_etl.py` | F6 Tier-1 network role feature builder (FEATURE_DESIGN_V2 §7). |
| `patch_combat_zone_pass.py` | Patch stdb.team_phase_exposure.combat_zone_pass_count to remove the |
| `phase_edge_etl.py` | Phase-level edge cache builder for DynamicSTGNN. |
| `phase_spatial_feature_etl.py` | Phase-level trajectory and exposure feature builder. |
| `rq1_feature_etl.py` | stdb/rq1_feature_etl.py |
| `skill_proxy_etl.py` | Build leakage-safe historical outcome proxy tiers. |
| `skill_rating_proxy_etl.py` | Build opponent-adjusted historical skill rating proxies. |
| `snapshot_feature_etl.py` | Snapshot-resolution (10s) team node-feature builder for the multi-resolution |
| `z_affordance_feature_etl.py` | O6 Z-ladder spatial-affordance feature builder. |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
