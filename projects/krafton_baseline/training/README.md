# projects.krafton_baseline.training

Explicit training entry points and objective components.

| Component | Purpose |
|---|---|
| `h0_baseline.py` | H0 arms: Cox scores + an estimated phase baseline hazard (grouped-time Cox, Prentice-Gloeckler cloglog). |
| `h0_fit.py` | Maximum likelihood of the grouped-time Cox baseline: P(Y = 1 / s, k) = 1 - exp(-exp(gamma_k + beta * s)) (numpy only). |
| `hgb_kraftonfeat.py` | HGB-kraftonfeat: gradient-boosted trees on the reconstructed KRAFTON squad inputs, evaluated at the phase cuts. |
| `ourfeat_hazard.py` | K3a / K3: KRAFTON-tf-ourfeat (K2) with the weighted-Cox objective replaced by a per-team BCE on the next-phase hazard. |
| `train.py` | Train and evaluate KRAFTON's model (adapted copy) on one BigComp 2027 cohort and write PTR-format outputs. |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
