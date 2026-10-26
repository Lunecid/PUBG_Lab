# pubg_lab.spatial.erangel

Responsibility: Erangel and Caldera spatial preparation and visualization.

Dependency boundary: Shared workspace paths and explicit data inputs.

| Component | Responsibility |
|---|---|
| `erangel_100m_build.py` | Build 100m graph from TRAIN matches only; then evaluate held-out matches. |
| `erangel_100m_extract.py` | Read-only, resumable full 40.2 inferred Erangel extraction. |
| `erangel_100m_verify.py` | Independent structural/accounting checks on the completed run. |
| `erangel_100m_visualize.py` | Render the frozen training graph and held-out coverage, without refitting. |
| `erangel_spatial_build.py` | Version-aware, diagnostic Erangel registration and observed-transition graph. |
| `erangel_spatial_extract.py` | Read-only Erangel patch inventory and bounded, pseudonymous spatial probe. |
| `erangel_spatial_patch_inventory.py` | Reproduce schedule-based patch inference from the saved read-only inventory. |
| `erangel_spatial_visualize.py` | Render actual raster/telemetry/graph artifacts; no generated map imagery. |
| `heterograph_case_build.py` | Causal as-of replay of a single real match; retrospective selection is explicit. |
| `heterograph_case_extract.py` | Read-only, one-match evidence export. Credentials are prompted, never stored. |
| `heterograph_case_figure.py` | Reproducible four-scene evidence figure, plotted from saved observer frames. |

The implementation lives here. Old root paths are compatibility entry points.
Import reusable components by their canonical `pubg_lab` path; invoke applications through `python -m pubg_lab run <command>`.
Listing commands does not import applications, access data, or start services.
Scientific contracts and completed evidence remain under the repository docs and original artifact paths.
