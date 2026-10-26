# pubg_lab.data.quality

quality package.

| Component | Purpose |
|---|---|
| `audit_bot_trajectories.py` | Read-only, stratified feasibility probe; password comes from env or a hidden prompt. |
| `audit_decision_state_schema.py` | Read-only bounded audit of evidence available for player decision states. |
| `classify_collected_data.py` | Read-only, match-level collection audit. Does not train or change source data. |
| `corpus_quality_index.py` | Persistent, local quality index. Never connects to or changes the source DB. |
| `inspect_movement_schema.py` | Export live database metadata and bounded movement-study availability checks, read-only. |
| `summarize_corpus_quality.py` | Reclassify saved aggregate queries and build inspectable quality deliverables. |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
