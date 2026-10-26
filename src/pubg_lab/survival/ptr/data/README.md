# pubg_lab.survival.ptr.data

Dataset contracts, assembly and source adapters.

| Component | Purpose |
|---|---|
| `cohorts.py` | Cohort discovery from the dump directory (design doc v2.3). |
| `data.py` | PTR dump loading, standardisation, dataset and collate (DB-free). |
| `extra.py` | Extra team-level features of the UNION arms (design doc 4.6, post hoc / exploratory): the join of the ``kf_*`` features |
| `paths.py` | The roots of the v2 re-run in ONE place, and the guard against mixing v1 artifacts with a v2 dump. |
| `synthetic.py` | Synthetic PTR dumps (DB-free) following the cohort dump schema (design doc v2.3). |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
