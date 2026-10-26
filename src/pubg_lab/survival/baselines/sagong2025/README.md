# pubg_lab.survival.baselines.sagong2025

sagong2025 package.

| Component | Purpose |
|---|---|
| `__main__.py` | ``python -m model_stgcn.sagong2025 {plan,extract,fit} ...`` |
| `db.py` | Read-only PostgreSQL access for the Sagong et al. (2025) feature extraction (SELECT only). |
| `extract.py` | Stage ``extract``: raw telemetry (DB, read-only SELECT) -> the 8 Sagong in-game features per team and phase. |
| `features.py` | The 8 in-game features of Sagong et al. (2025, Table 4) per unit (team) and phase from raw telemetry (DB-free core). |
| `fit.py` | Stage ``fit``: Sagong et al. (2025) types, Eq. 4 survival rate and the three classifiers on the PTR rows. |
| `hab.py` | Zone + habitat columns of the PTR dump per risk-set row, for the ``SAGONG-hgb-hab`` control (needs torch, no DB). |
| `plan.py` | Stage ``plan``: PTR dumps -> the exact evaluation rows and phase windows (needs torch, no DB). |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
