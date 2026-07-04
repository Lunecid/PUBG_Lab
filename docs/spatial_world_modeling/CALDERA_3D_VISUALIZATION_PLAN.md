# Caldera 3D Visualization Plan

## Figure Purpose

`scripts/visualize_caldera_3d.py` is a lightweight prototype for turning the
Caldera side of the common Battle Royale DB layer into an interactive 3D scene.
The first output is a scouting artifact for showing that PUBG and Activision
Caldera can share a spatial representation:

- Caldera breadcrumb trajectories become player line traces.
- Anonymous endpoint samples become a point-cloud trace.
- `br_ground.environment_prims` bbox and centroid metadata become environment
  candidate traces.

The default output is:

```text
reports/caldera_3d/caldera_spatial_scene.html
```

For paper figures, the preferred output is a snapshot over a clay-map style
scene:

```text
reports/caldera_3d/caldera_spatial_snapshot.html
```

Visual target:

- white/gray modeled environment scaffold;
- colored player trajectories up to a selected game time;
- emphasized current player positions at the snapshot time;
- faint endpoint field as a spatial outcome/exposure background;
- fixed camera preset for repeatable paper screenshots.

Current generated sample:

```text
trajectory_matches=2
trajectory_agents=12
trajectory_points=161
endpoint_points=2000
environment_prims=300
html_size=5.2 MB
```

Current generated paper snapshot:

```text
trajectory_matches=2
trajectory_agents=16
trajectory_points=301
endpoint_points=2000
environment_prims=500
snapshot_t=310.992
html_size=5.1 MB
```

Example usage:

```bash
PG_PASSWORD=... python3 scripts/visualize_caldera_3d.py \
  --match-limit 2 \
  --players-per-match 8 \
  --endpoint-limit 3000 \
  --environment-limit 800
```

Paper snapshot usage:

```bash
PG_PASSWORD=... python3 scripts/visualize_caldera_3d.py \
  --figure-mode paper-snapshot \
  --match-limit 2 \
  --players-per-match 8 \
  --endpoint-limit 2000 \
  --environment-limit 500 \
  --trajectory-sample-stride 8 \
  --camera-preset oblique
```

For a cheap DB/query check without Plotly:

```bash
PG_PASSWORD=... python3 scripts/visualize_caldera_3d.py --dry-run
```

## Current Design

The prototype reads only bounded samples from PostgreSQL through
`stdb.conn.connect`. The main controls are:

- `--match-limit`: number of Caldera matches selected from
  `br_ground.trajectory_samples`.
- `--players-per-match`: number of player agents selected per match.
- `--trajectory-sample-stride`: temporal downsampling per selected player.
- `--trajectory-point-limit`: hard cap after trajectory filtering.
- `--endpoint-limit`: cap for `br_ground.endpoint_samples`.
- `--environment-limit`: cap for `br_ground.environment_prims` bbox/centroid
  candidates.
- `--figure-mode`: `explore` for all sampled traces, `paper-snapshot` for
  publication-style user movement snapshots.
- `--snapshot-t`: fixed game second for the movement snapshot.
- `--snapshot-percentile`: automatic snapshot time when `--snapshot-t` is not
  supplied.
- `--camera-preset`: repeatable camera presets: `overview`, `oblique`,
  `topdown`, or `low`.
- `--static-output`: optional PNG/SVG export path once Kaleido is installed.

Plotly is imported only when HTML is written. If Plotly is missing, the script
prints an install hint and exits cleanly. `--dry-run` can still validate the DB
sampling path without Plotly.

## Asset Mesh Snapshot Path

The bbox/centroid HTML is not sufficient for paper figures that need to show
the actual Caldera environment.  The asset-backed path is:

```text
scripts/render_caldera_asset_snapshot.py
```

This script extracts real OpenUSD mesh vertices/faces from downloaded Caldera
visual prefabs, caches the simplified mesh as a local file, then renders a
clay-style snapshot with Caldera DB trajectories projected into the same world
coordinate system.

Current working paper candidates:

```text
reports/caldera_3d/map_capital_visual_ground_agent_snapshot_1m_points.png
reports/caldera_3d/map_capital_visual_ground_agent_snapshot_closeup.png
```

Current cache:

```text
reports/caldera_3d/map_capital_visual_ground_crop_cache_1m.pkl
```

## Full Map Mesh Layer

Before drawing agents, the map itself must exist as a reproducible 3D mesh
layer.  The current map-modeling path is:

```text
scripts/build_caldera_map_mesh.py
reports/caldera_3d/map_mesh_lod0/caldera_map_lod0.obj
reports/caldera_3d/map_mesh_lod0/manifest.json
reports/caldera_3d/map_mesh_lod0/caldera_map_lod0_preview.png
```

This builder uses `map_source/mp_wz_island.usd` as the root stage so terrain
and district branches keep their world placement.  It sets terrain/district LOD
variants to `full`, filters non-visual gameplay/audio/loot/trigger layers, and
streams a bounded LOD OBJ instead of trying to flatten the full dataset.

Current LOD0 summary:

```text
source_count=41
faces=688,027
meshes_used=67,968
obj_size=79 MB
bounds_x=-2038.45..1621.69 m
bounds_y=-1607.72..1953.01 m
bounds_z=-16.20..246.60 m
```

The release README describes the full Caldera scene as roughly 17.5M
primitives and more than 2B points, so a full-fidelity monolithic mesh is not a
safe local target.  The practical target is a tile/LOD mesh package: broad map
LOD first, then higher-detail district meshes for paper close-ups.

## Semantic Basemap Layer

The paper map should not rely on the earlier bbox/centroid scaffold.  The
strongest local representation currently available is a semantic basemap built
from the real OpenUSD mesh geometry:

```text
scripts/render_caldera_semantic_basemap.py
reports/caldera_3d/semantic_basemap/caldera_semantic_triangles.pkl
reports/caldera_3d/semantic_basemap/caldera_semantic_triangles.manifest.json
reports/caldera_3d/semantic_basemap/caldera_semantic_basemap_v4_thin_outline.png
reports/caldera_3d/semantic_basemap/caldera_semantic_basemap_v4_no_outline.png
```

This renderer opens `map_source/mp_wz_island.usd`, scans all terrain and
district branches, classifies mesh paths into terrain, coast/beach, water,
vegetation, road, runway, building, and prop layers, then rasterizes those
layers into a top-down image.  The current extraction filled every category
budget:

```text
terrain=360,000 selected from 4,365,093 candidates
coast_beach=80,000 selected from 4,642,119 candidates
water=35,000 selected from 989,253 candidates
vegetation=95,000 selected from 657,345 candidates
road=150,000 selected from 1,939,872 candidates
runway=40,000 selected from 979,182 candidates
building=230,000 selected from 11,459,056 candidates
prop=45,000 selected from 153,142 candidates
```

Example:

```bash
python3 scripts/render_caldera_semantic_basemap.py \
  --width 2400 \
  --height 2400 \
  --bounds-mode robust \
  --bounds-trim-percentile 0.05 \
  --coastline-thickness 1 \
  --output reports/caldera_3d/semantic_basemap/caldera_semantic_basemap_v4_thin_outline.png
```

The same script also has an optional `--overlay-trajectories` mode that draws
Caldera player trajectories and snapshot positions on top of the semantic
basemap.  That mode requires an explicit local PostgreSQL connection through
`PG_PASSWORD`; it was not executed in the sandboxed run.

Important interpretation note: this is not an official Caldera minimap texture.
It is a reconstructed research visualization from geometry plus USD path
semantics.  It can show the island-scale spatial structure, terrain relief,
roads/runways, POI building masses, vegetation/clutter, and movement overlays.
It cannot exactly reproduce official satellite-map colors or artist-authored
minimap labels unless a licensed raster/minimap texture or a stable USD
material renderer is added.

All-candidate stress test:

```text
reports/caldera_3d/semantic_basemap/caldera_semantic_triangles_all_candidates.pkl
reports/caldera_3d/semantic_basemap/caldera_semantic_basemap_all_candidates.png
reports/caldera_3d/semantic_basemap/caldera_semantic_basemap_comparison.png
```

This run stored every semantic candidate captured by the current filter:
25,185,062 triangles total, producing an 865 MB cache.  The image contains more
fine detail, but it also amplifies large guide/proxy-looking rectangles and
triangles around the map.  For paper figures, the all-candidate result is useful
as an upper-bound diagnostic, while the budgeted semantic sample is cleaner for
publication unless the classifier is tightened further.

## ST-Aware 3D Scene Layer

The next research object is not only a visual map but an interactive
space-time scene where player movement can be inspected against reconstructed
terrain and geometry:

```text
scripts/build_caldera_st_3d_scene.py
reports/caldera_3d/caldera_st_3d_scene.html
reports/caldera_3d/caldera_st_3d_scene.manifest.json
```

This builder combines the budgeted semantic mesh cache with observed
trajectory traces extracted from `caldera_spatial_snapshot.html`.  The output
is a standalone Plotly 3D HTML with:

- real x/y/z mesh geometry for terrain, coast, water, vegetation, roads,
  runway, buildings, and props;
- observed agent trajectories rendered in the same x/y/z coordinate system;
- an animation slider for cumulative ST movement and current agent positions;
- hover diagnostics containing time, position, nearest reconstructed terrain
  elevation, and `agent_z - terrain_z`.

Current prototype summary:

```text
display_triangles=252,000
trajectory_points=136
trajectory_agents=16
time_range=0.00..310.99 seconds
median_agent_z_minus_terrain_z=1.43 m
max_agent_z_minus_terrain_z=387.61 m
```

The median height-above-terrain value indicates that most in-play samples are
physically close to the reconstructed terrain.  Large positive values mark
airborne/pre-spawn or vertical-transition phases and should be treated as
phase-aware ST events rather than ordinary ground movement.

This scene is suitable for designing paper features such as local elevation,
relative height, slope exposure, road/runway proximity, building-density
context, water/coast proximity, and phase-specific movement channels.  It is
not yet a full collision/navmesh reconstruction, so walkability and indoor
floor-level semantics still need a separate layer if they become central to the
paper.

Example:

```bash
PG_PASSWORD=... python3 scripts/render_caldera_asset_snapshot.py \
  --renderer raster \
  --mesh-cache reports/caldera_3d/map_capital_visual_ground_crop_cache_1m.pkl \
  --players 8 \
  --trajectory-stride 5 \
  --agent-z-max 90 \
  --sample-t-min 250 \
  --mesh-point-alpha 0.42 \
  --output reports/caldera_3d/map_capital_visual_ground_agent_snapshot_1m_points.png
```

Important renderer note: the full Caldera assets are present locally under
`data/br_l0/caldera/repo/assets`, but `/usr/bin/usdrecord` currently crashes on
these stages with the installed macOS Metal renderer, and `usdview` is not
installed in this environment.  Until a stable USD viewport renderer is
available, the reproducible local figure path is mesh extraction plus the
software raster renderer above.

Static image export is available through:

```bash
PG_PASSWORD=... python3 scripts/visualize_caldera_3d.py \
  --figure-mode paper-snapshot \
  --static-output reports/caldera_3d/caldera_spatial_snapshot.png
```

This requires Kaleido:

```bash
python3 -m pip install kaleido
```

Kaleido is not currently installed in the checked environment, so the generated
artifact for now is HTML. The HTML can still be opened interactively and used to
choose camera presets before static export.

## Current Limits

This is a DB-catalog visualization, not a full OpenUSD renderer:

- `environment_prims` stores prim metadata, bbox columns, and centroids. It does
  not store terrain mesh vertices or textured geometry.
- Endpoint samples are anonymous point clouds and do not expose match/player
  identity.
- Trajectory lines are deliberately downsampled, so they are useful for spatial
  shape and density, not frame-accurate animation.
- Bboxes are only candidates for visible environmental structure. They may
  include helper prims, referenced assets, broad containers, or catalog artifacts
  that are not publication-quality scene geometry.
- The HTML scene is interactive and useful for inspection. `paper-snapshot`
  mode now fixes camera presets and snapshot time selection, but final static
  export still needs Kaleido or an equivalent renderer.
- The asset-mesh raster path uses real USD mesh vertices/faces, but it is still
  a clay/point overlay renderer.  It does not reproduce the full `usdview`
  textured viewport shown in the Activision example images.

## Next Steps

1. Terrain mesh layer
   - Add a mesh extraction step from `data/br_l0/caldera/repo/map_source` or
     derived assets.
   - Keep heavy mesh artifacts outside PostgreSQL and register paths/metadata in
     the DB.
   - Render simplified terrain as a surface or mesh trace under trajectories.

2. Camera path and figure presets
   - Add named camera presets for overview, endpoint density, trajectory-detail,
     and environment-catalog views.
   - Save camera settings with the HTML so paper screenshots are repeatable.
   - Consider a short camera-path animation for talk slides or supplemental
     material.

3. Snapshot image export
   - Add optional static export via Plotly Kaleido once the figure composition is
     stable.
   - Generate `png` and `svg` variants under `reports/caldera_3d/`.
   - Lock output size, camera, trace visibility, and color palette for the final
     paper figure.

4. Semantic filtering
   - Promote useful `prim_type`, `kind`, and asset-reference filters into CLI
     options.
   - Separate terrain, buildings, vegetation, roads, and large props once those
     categories are reliable enough for explanatory figures.
