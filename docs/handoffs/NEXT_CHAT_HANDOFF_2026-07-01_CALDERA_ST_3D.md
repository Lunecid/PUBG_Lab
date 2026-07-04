# Next Chat Handoff: Caldera ST-Aware 3D Map

Date: 2026-07-01

## One-Line State

We moved from a bbox/point scaffold to a geometry-backed, semantic, ST-aware
Caldera 3D scene: reconstructed terrain/geometry from official OpenUSD assets
plus observed agent `x/y/z/t` trajectories in the same coordinate system.

## Research Direction Locked Today

The target is not only a visual figure.  The target is a spatial-temporal
research object:

```text
official raw asset/telemetry
-> preserved ground layer
-> semantic spatial map layer
-> ST trajectory-grounded analysis layer
-> model features / paper figures
```

For Warzone Caldera, the geometry layer is especially important because it
provides terrain, elevation, roads/runways, water/coast regions, building mass,
and vegetation/context in ways PUBG telemetry alone does not.

## What We Confirmed

The current Caldera asset data can support a meaningful spatial layer:

- Real OpenUSD mesh coordinates exist in `x/y/z`.
- Terrain elevation is available through geometry, not only 2D map imagery.
- Asset path semantics are strong enough to classify broad spatial layers:
  `terrain`, `coast_beach`, `water`, `vegetation`, `road`, `runway`,
  `building`, and `prop`.
- Agent trajectories can be overlaid in the same world coordinate system.
- `agent_z - terrain_z` can be computed approximately, giving a first physical
  consistency diagnostic.

Important limitation:

- This is not a full game-engine reconstruction.  It is not yet collision,
  navmesh, indoor floor semantics, material-accurate rendering, or official
  minimap texture reproduction.

## Key Artifacts

### Semantic Basemap Builder

```text
scripts/render_caldera_semantic_basemap.py
docs/spatial_world_modeling/CALDERA_3D_VISUALIZATION_PLAN.md
```

Main budgeted semantic cache:

```text
reports/caldera_3d/semantic_basemap/caldera_semantic_triangles.pkl
reports/caldera_3d/semantic_basemap/caldera_semantic_triangles.manifest.json
```

Main budgeted figure candidate:

```text
reports/caldera_3d/semantic_basemap/caldera_semantic_basemap_v4_thin_outline.png
```

All-candidate stress test:

```text
reports/caldera_3d/semantic_basemap/caldera_semantic_triangles_all_candidates.pkl
reports/caldera_3d/semantic_basemap/caldera_semantic_basemap_all_candidates.png
reports/caldera_3d/semantic_basemap/caldera_semantic_basemap_comparison.png
```

### ST-Aware 3D Scene Builder

```text
scripts/build_caldera_st_3d_scene.py
reports/caldera_3d/caldera_st_3d_scene.html
reports/caldera_3d/caldera_st_3d_scene.manifest.json
```

This is the current best prototype for the paper-design direction.

## Current Quantitative State

Budgeted semantic cache selected:

```text
terrain=360,000
coast_beach=80,000
water=35,000
vegetation=95,000
road=150,000
runway=40,000
building=230,000
prop=45,000
```

Total candidate counts seen by current semantic classifier:

```text
terrain=4,365,093
coast_beach=4,642,119
water=989,253
vegetation=657,345
road=1,939,872
runway=979,182
building=11,459,056
prop=153,142
total=25,185,062
```

All-candidate stress test:

```text
all candidate cache size=865 MB
semantic_basemap folder size ~=928 MB after stress test
```

Disk state after stress test:

```text
available disk ~=23 GiB
filesystem capacity used ~=98%
```

Current ST 3D scene display:

```text
display_triangles=252,000
trajectory_points=136
trajectory_agents=16
time_range=0.00..310.99 seconds
html_size=33 MB
```

Terrain consistency diagnostic:

```text
min(agent_z - terrain_z)=-4.82 m
median(agent_z - terrain_z)=1.43 m
max(agent_z - terrain_z)=387.61 m
```

Interpretation:

- Median near `1.43 m` suggests most in-play trajectory samples sit close to
  reconstructed terrain.
- Large positive values likely indicate pre-spawn, airborne, drop, or vertical
  transition phases.  These should be modeled as phase-aware ST events, not
  ordinary ground movement.

## Important Design Decisions

### 1. Do Not Use Every Candidate Blindly For Paper Figures

Using every semantic candidate increases detail but also amplifies guide/proxy
artifacts: large rectangles, triangles, and non-paper visual debris.  The
all-candidate result is useful as an upper-bound diagnostic, not necessarily
as the clean final figure.

Better direction:

```text
semantic filtering + category budgets + artifact suppression
```

### 2. Keep Two Visual/Analysis Tracks

Track A: paper-visible figure

```text
clean semantic basemap
selected camera
selected agents
snapshot or short animation
```

Track B: analysis layer

```text
terrain elevation
relative height
slope / local relief
road/runway proximity
building density / POI context
water/coast proximity
phase-aware movement channels
```

### 3. Treat High Z Samples As Phase Signals

Large `agent_z - terrain_z` values are not necessarily errors.  They can encode:

- pre-spawn
- drop / airborne movement
- redeploy
- vertical traversal
- high structures
- incomplete terrain match in sparse displayed mesh

The next model feature design should separate ground movement from airborne or
transition phases.

## Reproduction Commands

Budgeted semantic basemap:

```bash
python3 scripts/render_caldera_semantic_basemap.py \
  --width 2400 \
  --height 2400 \
  --bounds-mode robust \
  --bounds-trim-percentile 0.05 \
  --coastline-thickness 1 \
  --output reports/caldera_3d/semantic_basemap/caldera_semantic_basemap_v4_thin_outline.png
```

All-candidate semantic stress test:

```bash
python3 scripts/render_caldera_semantic_basemap.py \
  --force-extract \
  --cache reports/caldera_3d/semantic_basemap/caldera_semantic_triangles_all_candidates.pkl \
  --output reports/caldera_3d/semantic_basemap/caldera_semantic_basemap_all_candidates.png \
  --width 2400 \
  --height 2400 \
  --bounds-mode robust \
  --bounds-trim-percentile 0.05 \
  --coastline-thickness 1 \
  --terrain-triangles 4365093 \
  --coast-beach-triangles 4642119 \
  --water-triangles 989253 \
  --vegetation-triangles 657345 \
  --road-triangles 1939872 \
  --runway-triangles 979182 \
  --building-triangles 11459056 \
  --prop-triangles 153142
```

ST-aware 3D scene:

```bash
python3 scripts/build_caldera_st_3d_scene.py \
  --output reports/caldera_3d/caldera_st_3d_scene.html \
  --frames 28 \
  --include-plotlyjs inline
```

## Current Best Interpretation For Paper Writing

A careful claim:

```text
We reconstructed a geometry-backed semantic spatial layer from official
Caldera OpenUSD assets and aligned observed player trajectories in the same
metric coordinate system.  This enables ST feature construction over terrain
elevation, relative height, roads/runways, coast/water regions, building mass,
and vegetation/context, while preserving phase-aware vertical movement events.
```

Avoid overclaiming:

```text
We did not reconstruct the full game-engine simulation layer, collision mesh,
navigation mesh, material-accurate renderer, or official minimap texture.
```

## Next Recommended Work

1. Tighten semantic classification.
   - Suppress guide/proxy artifacts.
   - Separate terrain surface from helper planes.
   - Separate building mass from large non-building rectangles.

2. Build a spatial feature extraction table.
   - `trajectory_sample_id`
   - `terrain_z`
   - `height_above_terrain`
   - `local_slope`
   - `road_distance`
   - `runway_distance`
   - `building_density_radius_*`
   - `water_distance`
   - `coast_distance`
   - `phase_channel`

3. Make ST phase segmentation explicit.
   - pre-spawn / airborne / in-play ground / vertical transition / final state
   - use `agent_z - terrain_z` and `sample_t` as first diagnostics

4. Create a paper figure pair.
   - Figure A: clean semantic Caldera map with feature layers.
   - Figure B: 3D ST trajectory snapshot showing agent paths over terrain.

5. Decide storage policy.
   - The all-candidate cache is useful but large (`865 MB`).
   - If disk pressure increases, keep the manifest/image and delete/rebuild
     the all-candidate pickle only when needed.

## Files To Open First In Next Chat

```text
docs/handoffs/NEXT_CHAT_HANDOFF_2026-07-01_CALDERA_ST_3D.md
docs/spatial_world_modeling/CALDERA_3D_VISUALIZATION_PLAN.md
scripts/build_caldera_st_3d_scene.py
scripts/render_caldera_semantic_basemap.py
reports/caldera_3d/caldera_st_3d_scene.manifest.json
```
