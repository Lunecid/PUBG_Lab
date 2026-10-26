# projects.krafton_baseline.reporting

Reports built from explicit artifacts.

| Component | Purpose |
|---|---|
| `h0_report.py` | Tables and paired contrasts of the H0 arms (``adaptation.h0_baseline``) next to the Cox, hazard and PTR arms (seed 42, test split). |
| `k3_facts.py` | Run facts of K2 / K3a / K3 (epochs, early stopping, Platt scaling, intercept movement, run time) from the metrics.json files. |
| `parity.py` | Feature parity table of the reconstruction (definitional status + descriptive statistics of the train rows). |
| `summarize.py` | Summary table of the KRAFTON baseline runs (KRAFTON-tf, KRAFTON-mlp, KRAFTON-tf-ourfeat) next to the PTR reference arms. |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
