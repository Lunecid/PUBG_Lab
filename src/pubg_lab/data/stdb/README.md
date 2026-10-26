# pubg_lab.data.stdb

Responsibility: STDB access, quality rules and data preparation.

Dependency boundary: Shared infrastructure and scientific source contracts.

| Component | Responsibility |
|---|---|
| `audit_raw.py` | stdb/audit_raw.py |
| `base_stage.py` | Transient re-materialisation of the base STDB rows for ONE match. |
| `build_available.py` | stdb/build_available.py |
| `causal_cut.py` | Causal-cut guard for phase-tier features and edges (Fix B + single clock, 2026-10-05). |
| `conn.py` | stdb/conn.py |
| `etl.py` | stdb/etl.py |
| `f0_feature_etl.py` | F0 official-raw extras (Tier-1: stdb-base derivable) — FEATURE_DESIGN_V2 §4. |
| `habitat_feature_etl.py` | F7 Tier-1 habitat-inspired pressure feature builder (FEATURE_DESIGN_V2 §8). |
| `network_feature_etl.py` | F6 Tier-1 network role feature builder (FEATURE_DESIGN_V2 §7). |
| `patch_combat_zone_pass.py` | Patch stdb.team_phase_exposure.combat_zone_pass_count to remove the |
| `phase_edge_etl.py` | Phase-level edge cache builder for DynamicSTGNN. |
| `phase_spatial_feature_etl.py` | Phase-level trajectory and exposure feature builder. |
| `queries.py` | stdb/queries.py |
| `rq1_feature_etl.py` | stdb/rq1_feature_etl.py |
| `skill_coverage_audit.py` | Audit whether current PUBG data can support repeated-player skill proxies. |
| `skill_proxy_etl.py` | Build leakage-safe historical outcome proxy tiers. |
| `skill_rating_proxy_etl.py` | Build opponent-adjusted historical skill rating proxies. |
| `snapshot_feature_etl.py` | Snapshot-resolution (10s) team node-feature builder for the multi-resolution |
| `snapshot_queries.py` | Snapshot-resolution (10s) tensor fetch for the MR-STGNN Tier-S. |
| `v2_view.py` | V2 unified feature view (FEATURE_DESIGN_V2 §12 Step 4). |
| `z_affordance_feature_etl.py` | O6 Z-ladder spatial-affordance feature builder. |

The implementation lives here. Old root paths are compatibility entry points.
Import reusable components by their canonical `pubg_lab` path; invoke applications through `python -m pubg_lab run <command>`.
Listing commands does not import applications, access data, or start services.
Scientific contracts and completed evidence remain under the repository docs and original artifact paths.
