# pubg_lab.data.quality

Responsibility: STDB access, quality rules and data preparation.

Dependency boundary: Shared infrastructure and scientific source contracts.

| Component | Responsibility |
|---|---|
| `audit_bot_trajectories.py` | Read-only, stratified feasibility probe; password comes from env or a hidden prompt. |
| `audit_decision_state_schema.py` | Read-only bounded audit of evidence available for player decision states. |
| `classify_collected_data.py` | Read-only, match-level collection audit. Does not train or change source data. |
| `corpus_quality_index.py` | Persistent, local quality index. Never connects to or changes the source DB. |
| `inspect_movement_schema.py` | Export live database metadata and bounded movement-study availability checks, read-only. |
| `summarize_corpus_quality.py` | Reclassify saved aggregate queries and build inspectable quality deliverables. |

The implementation lives here. Old root paths are compatibility entry points.
Import reusable components by their canonical `pubg_lab` path; invoke applications through `python -m pubg_lab run <command>`.
Listing commands does not import applications, access data, or start services.
Scientific contracts and completed evidence remain under the repository docs and original artifact paths.
