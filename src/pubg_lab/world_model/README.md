# pubg_lab.world_model

Responsibility: World-model inputs, encoder components, pretraining and owned CLI.

Dependency boundary: Shared infrastructure, typed graph inputs and approved data views.

| Component | Responsibility |
|---|---|
| `cohorts.py` | G3 cohort and event index. Future event selection is NEVER encoder input. |
| `dataset.py` | Lazy graph/target loader with ablation-specific train-only spatial support. |
| `encoder.py` | Untrained G1 typed temporal/set encoder; no decoder or optimizer. |
| `historical_prior.py` | Past-only behavioral edge evidence, NEVER a current passability label. |
| `indexed_dataset.py` | Spatial query acceleration with identical frozen-loader output. |
| `inputs.py` | G1 causal input boundary for the frozen V3 loader. |
| `patches.py` | Official PC maintenance-window inference, not executable build verification. |
| `preprocess.py` | Causal, variable-roster preprocessing. Coordinates are metres, time is seconds. |
| `pretraining.py` | G2 P1/P2/P3: bounded CPU feasibility heads, not a full game simulator. |
| `temporal_prior.py` | G4 single-owner past-only spatial evidence. |

The implementation lives here. Old root paths are compatibility entry points.
Import reusable components by their canonical `pubg_lab` path; invoke applications through `python -m pubg_lab run <command>`.
Listing commands does not import applications, access data, or start services.
Scientific contracts and completed evidence remain under the repository docs and original artifact paths.
