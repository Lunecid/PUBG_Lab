# pubg_lab.world_model.data

Dataset contracts, assembly and source adapters.

| Component | Purpose |
|---|---|
| `cohorts.py` | G3 cohort and event index. Future event selection is NEVER encoder input. |
| `dataset.py` | Lazy graph/target loader with ablation-specific train-only spatial support. |
| `indexed_dataset.py` | Spatial query acceleration with identical frozen-loader output. |
| `patches.py` | Official PC maintenance-window inference, not executable build verification. |
| `preprocess.py` | Causal, variable-roster preprocessing. Coordinates are metres, time is seconds. |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
