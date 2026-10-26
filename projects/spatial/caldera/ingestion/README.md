# projects.spatial.caldera.ingestion

Explicit spatial ingestion commands.

| Component | Purpose |
|---|---|
| `caldera_dataset_manager.py` | Manage Activision Caldera as an external BR L0 data source. |
| `caldera_ingest_geometry_catalog.py` | Catalog Caldera OpenUSD geometry metadata into the BR ground layer. |
| `caldera_ingest_players.py` | Extract Caldera player endpoint/breadcrumb data into br_ground/br_repr. |
| `export_caldera_mesh_cache_obj.py` | Export a Caldera local mesh cache to OBJ/MTL for Blender rendering. |
| `export_caldera_trajectory_json.py` | Export one Caldera player trajectory to JSON for Blender overlays. |
| `extract_caldera_local_mesh.py` | Extract a local Caldera mesh window directly from the OpenUSD source stage. |
| `extract_caldera_terrain_experience_features.py` | Extract terrain-experience features for one Caldera trajectory segment. |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
