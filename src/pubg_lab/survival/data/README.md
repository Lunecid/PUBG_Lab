# pubg_lab.survival.data

Dataset contracts, assembly and source adapters.

Subpackages: `phase_v2/`.

| Component | Purpose |
|---|---|
| `split_access.py` | Approved public API boundary to the unchanged restricted split implementation |
| `dataset.py` | model_stgcn/dataset.py |
| `dump_phase_tier.py` | Per-cohort tensor dumps of the phase tier (design v2.3: one clock + Fix B cap) for the PTR study. |
| `dump_phase_tier_v2.py` | Leak-free phase-tier dump v2 for the BigComp 2027 PTR study (writes ``bigcomp2027_phase_v2``; v1 is untouched). |
| `dump_player_tier.py` | Per-cohort PLAYER-level tensor dumps (design stage 2, section 3) next to the phase dumps. |
| `mr_dataset.py` | Dataset + collate for the MR-STGNN: each item carries BOTH the phase tier |
| `snapshot_phase_dataset.py` | Dataset for snapshot-resolution phase survival training. |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
