# pubg_lab.spatial.caldera

Responsibility: Erangel and Caldera spatial preparation and visualization.

Dependency boundary: Shared workspace paths and explicit data inputs.

| Component | Responsibility |
|---|---|
| `assemble_caldera_o6_construct_validation_artifacts.py` | Assemble the Caldera O6 construct-validation artifact package. |
| `build_caldera_map_mesh.py` | Build a tile-based Caldera map mesh from OpenUSD assets. |
| `build_caldera_st_3d_scene.py` | Build an interactive ST-aware Caldera 3D scene. |
| `caldera_dataset_manager.py` | Manage Activision Caldera as an external BR L0 data source. |
| `caldera_ingest_geometry_catalog.py` | Catalog Caldera OpenUSD geometry metadata into the BR ground layer. |
| `caldera_ingest_players.py` | Extract Caldera player endpoint/breadcrumb data into br_ground/br_repr. |
| `export_caldera_mesh_cache_obj.py` | Export a Caldera local mesh cache to OBJ/MTL for Blender rendering. |
| `export_caldera_trajectory_json.py` | Export one Caldera player trajectory to JSON for Blender overlays. |
| `extract_caldera_local_mesh.py` | Extract a local Caldera mesh window directly from the OpenUSD source stage. |
| `extract_caldera_terrain_experience_features.py` | Extract terrain-experience features for one Caldera trajectory segment. |
| `make_caldera_rq3_feature_design_board.py` | Create an RQ3 feature-design board from Caldera geometry features. |
| `render_caldera_agent_experience_snapshot.py` | Render a terrain-aware Caldera trajectory experience snapshot. |
| `render_caldera_agent_pov_animation.py` | Render a first-person-ish Caldera trajectory animation from semantic mesh data. |
| `render_caldera_asset_snapshot.py` | Render a Caldera asset-mesh snapshot with player trajectories. |
| `render_caldera_blender_clay.py` | Render a Caldera OBJ mesh in Blender as a clay/wireframe scene. |
| `render_caldera_rq3_clean_pov_proxy.py` | Render clean player-POV proxy panels for Caldera RQ3. |
| `render_caldera_rq3_clean_situation_panels.py` | Render clean Caldera RQ3 situation panels from geometry features. |
| `render_caldera_rq3_zbuffer_pov.py` | Render Caldera player-POV panels with a small CPU z-buffer rasterizer. |
| `render_caldera_semantic_basemap.py` | Render a semantic top-down Caldera basemap from local OpenUSD assets. |
| `render_caldera_single_agent_topdown.py` | Render one Caldera agent trajectory over a semantic top-down mesh cache. |
| `visualize_caldera_3d.py` | Build a sampled 3D Caldera spatial scene from the common BR DB layer. |

The implementation lives here. Old root paths are compatibility entry points.
Import reusable components by their canonical `pubg_lab` path; invoke applications through `python -m pubg_lab run <command>`.
Listing commands does not import applications, access data, or start services.
Scientific contracts and completed evidence remain under the repository docs and original artifact paths.
