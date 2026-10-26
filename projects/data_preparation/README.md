# projects.data_preparation

data preparation package.

| Component | Purpose |
|---|---|
| `build_data_availability_domains.py` | Build availability tables by perspective, map, and team size. |
| `build_experiment_domain_datasets.py` | Create experiment-ready domain datasets from the raw PUBG population. |
| `build_research_cohort_flat.py` | Build a flat, non-destructive research cohort manifest. |
| `convert_raw_to_usable.py` | Convert raw PUBG matches into experiment-usable STDB/phase/graph data. |
| `import_research_cohort_to_br_exp.py` | Import the existing paper cohort manifest into br_exp. |
| `register_pubg_l1_ground.py` | Register current pubg.* as the preserved PUBG L1 ground layer. |
| `update_data_availability_summary.py` | Update map/mode data availability summary for phase-level GNN runs. |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
