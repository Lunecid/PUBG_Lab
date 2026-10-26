# pubg_lab.movement.data

Dataset contracts, assembly and source adapters.

| Component | Purpose |
|---|---|
| `bank.py` | Causal 10-second state banks, shared by both information views. |
| `bank_v3.py` | As-of multi-life and mobility evidence. No retrospective revival backfilling. |
| `dataset.py` | Paired views share sample IDs and future labels; graph construction is as-of only. |
| `dataset_v3.py` | V3 paired views with causal episode/mobility and direct circle geometry. |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
