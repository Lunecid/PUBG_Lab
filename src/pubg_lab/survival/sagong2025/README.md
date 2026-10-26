# pubg_lab.survival.sagong2025

Responsibility: BR/PTR/survival representation and model components.

Dependency boundary: Data contracts; private split access remains outside the public package.

| Component | Responsibility |
|---|---|
| `__main__.py` | ``python -m model_stgcn.sagong2025 {plan,extract,fit} ...`` |
| `db.py` | Read-only PostgreSQL access for the Sagong et al. (2025) feature extraction (SELECT only). |
| `extract.py` | Stage ``extract``: raw telemetry (DB, read-only SELECT) -> the 8 Sagong in-game features per team and phase. |
| `features.py` | The 8 in-game features of Sagong et al. (2025, Table 4) per unit (team) and phase from raw telemetry (DB-free core). |
| `fit.py` | Stage ``fit``: Sagong et al. (2025) types, Eq. 4 survival rate and the three classifiers on the PTR rows. |
| `hab.py` | Zone + habitat columns of the PTR dump per risk-set row, for the ``SAGONG-hgb-hab`` control (needs torch, no DB). |
| `plan.py` | Stage ``plan``: PTR dumps -> the exact evaluation rows and phase windows (needs torch, no DB). |

The implementation lives here. Old root paths are compatibility entry points.
Import reusable components by their canonical `pubg_lab` path; invoke applications through `python -m pubg_lab run <command>`.
Listing commands does not import applications, access data, or start services.
Scientific contracts and completed evidence remain under the repository docs and original artifact paths.
