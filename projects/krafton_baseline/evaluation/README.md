# projects.krafton_baseline.evaluation

Evaluation primitives and explicit verification commands.

| Component | Purpose |
|---|---|
| `diag_phase_init.py` | Diagnostic (not part of the K3 matrix): K3 with the phase offsets initialised at ZERO instead of the train base rates. |
| `diagnostics.py` | Diagnostic (not part of the baseline protocol): how much of the pooled hazard-AUC gap is the missing phase base rate. |
| `k3_contrasts.py` | Paired match-level bootstrap contrasts of the K3 test (pooled and within-phase AUC), with random-effects pooling. |
| `phase_auc.py` | Pooled and within-phase test AUC of the K-arms and reference arms (the K3 table of the paper's headline test). |
| `validate_causal.py` | Prefix-replay check of the feature reconstruction on REAL matches (read-only DB access). |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
