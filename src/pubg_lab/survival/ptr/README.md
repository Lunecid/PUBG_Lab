# pubg_lab.survival.ptr

Responsibility: BR/PTR/survival representation and model components.

Dependency boundary: Data contracts; private split access remains outside the public package.

| Component | Responsibility |
|---|---|
| `aggregate.py` | Results aggregator of the PTR matrix: design doc sections 4.1 (v2.2 decision rules) and 5. |
| `arms.py` | Experiment arms: design doc section 4 and section 4.1 (v2.1). |
| `cohorts.py` | Cohort discovery from the dump directory (design doc v2.3). |
| `data.py` | PTR dump loading, standardisation, dataset and collate (DB-free). |
| `derived.py` | Exploratory analyses derived from the existing prediction files and contrast tables (no retraining, no change to the |
| `ecomodel.py` | Ecological hazard model (EHM): an explicit discrete-time local-extinction model of team death (exploratory). |
| `extra.py` | Extra team-level features of the UNION arms (design doc 4.6, post hoc / exploratory): the join of the ``kf_*`` features |
| `gatv2.py` | Snapshot GATv2 baseline using the pinned PyG 2.7.0 operator directly. |
| `hgb.py` | HGB reference baselines built from the same dump (identical rows to the neural arms). |
| `learning_curve.py` | Learning curve of design doc 4.4 (exploratory): test AUC against the number of training matches. |
| `loss.py` | PTR loss: hazard BCE over ``target_mask`` + ``rank_weight`` * pairwise logistic rank loss. |
| `metrics.py` | Metrics, paired match-level bootstrap and DerSimonian-Laird pooling for PTR prediction files. |
| `model.py` | PTR-Former: phase-axis, team-axis and relation-biased attention over team tokens. |
| `paths.py` | The roots of the v2 re-run in ONE place, and the guard against mixing v1 artifacts with a v2 dump. |
| `player.py` | Stage-2 player layer: the player dump, its join to the phase dump, and the player relation tensors. |
| `relations.py` | Relation matrices for the team-axis attention bias, and the within-match edge shuffle. |
| `run_matrix.py` | Run the whole PTR experiment matrix on parallel CPU workers; resumable, one log per run, a status CSV. |
| `supplement.py` | Supplementary analyses E1-E3 of the BigComp 2027 results (post hoc, exploratory: no Holm family, no pre-registered claim). |
| `synthetic.py` | Synthetic PTR dumps (DB-free) following the cohort dump schema (design doc v2.3). |
| `train.py` | Train one PTR arm on one cohort dump (DB-free) and write predictions + metrics. |
| `util.py` | Small shared helpers: provenance, seeding, strict-JSON dumping. |

The implementation lives here. Old root paths are compatibility entry points.
Import reusable components by their canonical `pubg_lab` path; invoke applications through `python -m pubg_lab run <command>`.
Listing commands does not import applications, access data, or start services.
Scientific contracts and completed evidence remain under the repository docs and original artifact paths.
