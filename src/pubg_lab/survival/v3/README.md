# pubg_lab.survival.v3

Responsibility: BR/PTR/survival representation and model components.

Dependency boundary: Data contracts; private split access remains outside the public package.

| Component | Responsibility |
|---|---|
| `adv_model.py` | Phase-tier edge-conditioned GNN for the learned-advantage edge experiment. |
| `advantage_edges.py` | Contact-gated, relative-state edge features with a FUSED dynamic proximity gate. |
| `cohorts.py` | Named PUBG cohort filters for V3 experiment runs. |
| `edge_controls.py` | Phase-graph randomization controls for V3 factorized experiments. |
| `eval.py` | Evaluate V3 factorized checkpoints. |
| `experiment_matrix.py` | Canonical V3 experiment matrix. |
| `factorized_data.py` | Factorized V3 tensors. |
| `models.py` | Factorized V3 GNN models. |
| `run_matrix.py` | Run V3 experiment groups with a shared split. |
| `spec.py` | Feature role specification for the final factorized experiment. |
| `train.py` | Train V3 factorized models. |

The implementation lives here. Old root paths are compatibility entry points.
Import reusable components by their canonical `pubg_lab` path; invoke applications through `python -m pubg_lab run <command>`.
Listing commands does not import applications, access data, or start services.
Scientific contracts and completed evidence remain under the repository docs and original artifact paths.
