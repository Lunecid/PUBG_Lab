# pubg_lab.movement

Responsibility: Joint player/team movement models, datasets, metrics and owned CLI.

Dependency boundary: Shared infrastructure and spatial/data components.

| Component | Responsibility |
|---|---|
| `bank.py` | Causal 10-second state banks, shared by both information views. |
| `bank_v3.py` | As-of multi-life and mobility evidence. No retrospective revival backfilling. |
| `dataset.py` | Paired views share sample IDs and future labels; graph construction is as-of only. |
| `dataset_v3.py` | V3 paired views with causal episode/mobility and direct circle geometry. |
| `evaluation.py` | Masked joint-scenario metrics in meters; matched sample IDs, match-level averages. |
| `model.py` | Joint, member-equivariant graph-conditioned residual diffusion (v prediction). |
| `model_v3.py` | Same-input joint regression and smooth-knot diffusion with direct circle conditioning. |
| `model_v4.py` | Team-center/relative-formation prediction and frozen-mean residual diffusion. |

The implementation lives here. Old root paths are compatibility entry points.
Import reusable components by their canonical `pubg_lab` path; invoke applications through `python -m pubg_lab run <command>`.
Listing commands does not import applications, access data, or start services.
Scientific contracts and completed evidence remain under the repository docs and original artifact paths.
