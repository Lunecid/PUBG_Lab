# pubg_lab.survival.ptr.experiments

Experiment composition under each research protocol.

| Component | Purpose |
|---|---|
| `derived.py` | Exploratory analyses derived from the existing prediction files and contrast tables (no retraining, no change to the |
| `learning_curve.py` | Learning curve of design doc 4.4 (exploratory): test AUC against the number of training matches. |
| `run_matrix.py` | Run the whole PTR experiment matrix on parallel CPU workers; resumable, one log per run, a status CSV. |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
