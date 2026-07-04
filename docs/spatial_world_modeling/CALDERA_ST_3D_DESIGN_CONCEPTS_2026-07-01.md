# Caldera ST-Aware 3D Design Concepts

Date: 2026-07-01

This note preserves the concepts behind the Caldera ST-aware 3D work before
generated artifacts and caches are cleaned from the workspace.

## Core Research Object

The Caldera work is not just a visual map.  The intended object is a layered
spatial-temporal research representation:

```text
official raw asset / telemetry
-> preserved ground layer
-> semantic spatial map layer
-> ST trajectory-grounded analysis layer
-> model features / paper figures
```

The key claim is conservative: official Caldera OpenUSD geometry can be used to
build a metric semantic spatial layer, and observed player trajectories can be
aligned into the same `x/y/z/t` coordinate system.  This enables terrain-aware
and context-aware ST feature construction without claiming full game-engine
reconstruction.

## Preserved Inputs

The cleanup should preserve the inputs and implementation structure needed to
rebuild the work:

- PostgreSQL Battle Royale ground tables, especially trajectory and endpoint
  data exposed through the local DB layer.
- Caldera OpenUSD source assets under `data/br_l0/caldera`.
- Schema and data-layer contracts under `stdb/`.
- Design and handoff documents under `docs/`.
- Rebuild scripts under `scripts/`.
- Model/source code under `model_stgcn/`, excluding generated checkpoints.

Generated reports, HTML scenes, sampled pickle caches, model checkpoints,
prediction CSVs, logs, and package installs are reproducible artifacts rather
than primary research state.

## Spatial Layer Concepts

The spatial layer is geometry-backed:

- Mesh vertices and faces come from official OpenUSD assets.
- `x/y/z` coordinates are real world-space asset coordinates.
- Terrain elevation is derived from geometry rather than from a flat image.
- Broad semantic categories are inferred from USD path and asset naming
  semantics.

Current semantic categories:

```text
terrain
coast_beach
water
vegetation
road
runway
building
prop
```

The semantic map is a research visualization, not an official minimap texture.
It can express terrain, relief, roads, runways, water/coast, building mass, and
vegetation context, but it should not be described as material-accurate,
collision-accurate, navmesh-accurate, or an exact game renderer.

## ST Alignment Concepts

The trajectory layer uses observed agent samples:

```text
match_id
agent_id
sample_t
x
y
z
```

The important design property is that trajectory coordinates and asset geometry
are rendered in the same metric coordinate system.  This makes the scene useful
as an analysis object, not only as a picture.

The first physical diagnostic is:

```text
height_above_terrain = agent_z - nearest_terrain_z
```

This diagnostic is approximate because the nearest terrain estimate may use a
sampled/displayed mesh rather than the full terrain surface.  Still, a median
near ground level is useful evidence that the trajectory and terrain spaces are
reasonably aligned.

## Phase-Aware Interpretation

Large positive `height_above_terrain` values should not automatically be treated
as errors.  They may encode phase or movement state:

- pre-spawn
- aircraft/drop/airborne movement
- redeploy
- vertical traversal
- high structures
- sparse terrain lookup artifacts

The model feature design should therefore separate ground movement from
airborne or transition phases instead of collapsing every sample into one
ordinary movement channel.

Initial phase-channel idea:

```text
pre_spawn
airborne
in_play_ground
vertical_transition
final_state
unknown_or_sparse
```

## Feature Extraction Direction

The next durable artifact should be a spatial feature table, not a larger
render cache.  A useful row contract is:

```text
trajectory_sample_id
match_id
agent_id
sample_t
x
y
z
terrain_z
height_above_terrain
local_slope
local_relief
road_distance
runway_distance
building_density_radius_*
water_distance
coast_distance
semantic_region
phase_channel
```

This table is the bridge from 3D reconstruction to model features and paper
analysis.

## Visualization Tracks

Keep two tracks conceptually separate.

Track A: paper-visible figures

```text
clean semantic basemap
selected camera
selected agents
snapshot or short animation
minimal artifact noise
```

Track B: analysis layer

```text
terrain elevation
relative height
slope / local relief
road and runway proximity
building density
water and coast proximity
phase-aware movement channels
```

All-candidate geometry extraction is useful as a stress test, but it can
amplify proxy, guide, or helper geometry.  Paper figures should favor semantic
filtering, category budgets, and artifact suppression.

## Rebuild Entry Points

The current implementation concepts live in these scripts:

```text
scripts/render_caldera_semantic_basemap.py
scripts/build_caldera_st_3d_scene.py
scripts/build_caldera_map_mesh.py
scripts/render_caldera_asset_snapshot.py
scripts/visualize_caldera_3d.py
```

The handoff and planning documents are:

```text
docs/handoffs/NEXT_CHAT_HANDOFF_2026-07-01_CALDERA_ST_3D.md
docs/spatial_world_modeling/CALDERA_3D_VISUALIZATION_PLAN.md
```

After cleanup, regenerate only the artifacts needed for the next concrete
research step: either the feature table or the final paper figure pair.

## Cleanup Record

Cleanup performed after this note was created:

```text
removed reports/
removed model_stgcn/checkpoints/
removed network_analysis/output/
removed checkpoints/
removed logs/
removed outputs/
removed presentation_build/node_modules/
removed presentation_build/intro_en_png/
removed presentation_build/prev_pptx_png/
removed webapp/dist/
removed prev_pptx_png/
removed Python __pycache__ directories
removed .pytest_cache/
removed .DS_Store files
```

The cleanup reduced the workspace footprint from roughly `22G` to `11G`.
Disk availability on the current volume increased from roughly `23GiB` to
`33GiB`.

The following were intentionally preserved:

```text
data/br_l0/caldera/
.venv/
docs/
scripts/
stdb/
model_stgcn/ source files
paper/ source files
```
