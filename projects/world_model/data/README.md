# projects.world_model.data

Dataset contracts, assembly and source adapters.

| Component | Purpose |
|---|---|
| `build_world_model_g3.py` | G3 read-only full allowed corpus classification and rare-event index. |
| `build_world_model_historical_prior.py` | Build independent historical evidence from explicitly approved primary fit IDs. |
| `finalize_world_model_g3.py` | Merge tiers without duplicate samples, recheck every short label, smoke inputs. |
| `mine_world_model_short_transport.py` | Supplementary short-flight tier; never relabel as 30s-complete trajectories. |
| `world_model_preprocess.py` | Versioned full-corpus preprocessing; no optimizer, training, or test loader. |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
