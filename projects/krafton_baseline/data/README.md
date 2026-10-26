# projects.krafton_baseline.data

Dataset contracts, assembly and source adapters.

| Component | Purpose |
|---|---|
| `data.py` | In-memory sample sets and batch loaders for the KRAFTON baseline (reads the shards written by extract.py). |
| `extract.py` | Extraction of KRAFTON-style squad samples from the raw ``pubg.*`` tables (READ-ONLY database access). |
| `features.py` | Causal reconstruction of KRAFTON's per-squad inputs from the raw ``pubg.telem_*`` rows of ONE match (pure numpy). |
| `kftable.py` | Persisted KRAFTON feature table at the PTR phase cuts (reusable by other models; READ-ONLY database access). |
| `labels.py` | Compact label tables of the PTR dumps (model_stgcn/dumps/bigcomp2027_phase_v2) for the KRAFTON baseline. |
| `schema.py` | Feature schema, item / weapon dictionaries and constants of the adaptation. |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
