# pubg_lab.projects.figures

Responsibility: SAC, BR and figure applications composing reusable components.

Dependency boundary: Domain libraries; do not become dependencies of common infrastructure.

| Component | Responsibility |
|---|---|
| `create_pubg_safe_zone_proposal_docx.py` | Owned implementation component |
| `make_labeling_figure.py` | Figure: how the two prediction targets are labeled (leakage-safe, cut-phase). |
| `make_paper_figures_svg.py` | Generate publication figures for the PUBG STDB/DynamicSTGNN paper. |
| `make_phase_summary_figure.py` | Result figures (academic tone, revised spec): |
| `make_presentation_figures.py` | Presentation/paper figures from verified V3 results (see docs/PRESENTATION_DATA_2026-06-21.md). |
| `make_realmap_figure.py` | Real-match graph on the actual Erangel (Baltic_Main) map. |

The implementation lives here. Old root paths are compatibility entry points.
Import reusable components by their canonical `pubg_lab` path; invoke applications through `python -m pubg_lab run <command>`.
Listing commands does not import applications, access data, or start services.
Scientific contracts and completed evidence remain under the repository docs and original artifact paths.
