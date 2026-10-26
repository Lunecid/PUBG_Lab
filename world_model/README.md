# PUBG predictive world-model preprocessing and G1 encoder

This package prepares telemetry and implements an untrained G1 encoder.
There is no optimizer, pretraining task head or trajectory decoder.

The approved primary fold is **40.2 train → 41.1 development**. Patch 41.2 is
reserved. All discovered historical test assignments remain protected, and
previously validated 40.2 matches are not promoted into training. Patch identity
is inferred from official Steam/PC maintenance windows, not executable hashes.

Current run:
`/Users/lunecid/PUBG_Data/runs/20261020_world_model_patch_preprocess_v3`

## Read a graph

```python
from world_model.dataset import WorldDataset

root = "/Users/lunecid/PUBG_Data/runs/20261020_world_model_patch_preprocess_v3"
dataset = WorldDataset(root, role="validation", view="observer")
for match_id in dataset.match_ids:
    if len(dataset.bank(match_id)["anchors"]):
        inputs, targets = dataset.sample(match_id, anchor_index=0)
        break
```

`role="test"` is rejected. `bank()` is a privileged internal evidence container;
never feed it directly to the observer encoder. Only `sample()`/`view_at()`
apply the information boundary. Human/bot labels and final outcomes are not
encoder fields.

For the server comparison use `view="server"` or `ablation="server"`. Anchors,
target membership and labels are shared. The own-team spatial crop is shared;
the server can additionally represent recorded other players/teams.

## Shapes and meanings

`H <= 13` history frames at 10-second steps, variable visible players `P`,
variable teams `Q`, variable local cells `C`, future steps `F=6`.

| Field | Shape | Meaning |
|---|---|---|
| `x`, `mask` | `[H,P,17]` | Numeric player history; metres/seconds/logged counts, not fitted normalization |
| `member_edges` | `[P,2]` | Player index → team index |
| `team_x` | `[Q,7]` | Known member count, members with valid current positions, mean XY, XY standard deviation, mean pair distance |
| `items_sparse` | `[N,3]` | Frame, match-local actor index, fold-specific item vocabulary code; 0 padding/1 unknown |
| `mobility`, `mobility_mask` | `[H,P]` | Known/inferred movement mode under the selected information view |
| `reentry_confidence` | `[H,P]` | 2 direct evidence, 1 inferred; use the paired mask, never interpret it as calibrated probability |
| `vehicle_evidence` | `[V,3]` | Visible player index, mode, observation age; not unique vehicle instances |
| `game`, `game_mask` | `[H,9]` | Phase, alive counts and as-of circle geometry; observer HUD/announcement timing remains a proxy |
| `cell_coordinates` | `[C,2]` | Integer 100m cell coordinates, scoped by map and inferred patch |
| `cell_features` | `[C,6]` | Centre XY, observed outgoing count, incoming count, incident mode-edge count, any support flag |
| `observed_connections` | `[E,5]` | Source cell, destination, mode, observation count, supporting match count |
| `inferred_adjacency` | `[U,2]` | Hypothetical 4-neighbour links; no traversability truth label |
| `targets["xy"]`, `targets["mask"]` | `[M,6,2]`, `[M,6]` | Current known team members' future coordinates; episode/gap/end-aware masks |
| `targets["outcomes"]` | focal team only | Final rank/winner candidates and independent masks; never inputs |

`items_sparse` retains match-local actor IDs, mapped through `player_ids` for
graph construction. Frame numbers are match-bank frame indices. Unknown item
vocabulary entries are not learned from validation/held-out domains.

`source_time` is event-relative seconds, not a prediction target. Exact
visibility, sounds, complete ammunition inventory, unique vehicle instance,
terminal team elimination time and surveyed collision geometry are unavailable.

The actor's recorded Z coordinate is included in player history. It is not a
certified terrain height; the cell features do not claim to be a heightmap.

## Ablations

Each JSON under `ablations/` contains exact train and development IDs and the
patch boundary. The loader limits both vocabulary and spatial support to those
train IDs. `only_*`, `holdout_*` and other plans do not implicitly execute a model.
Check `status` in the manifest before scheduling an experiment.

`no_pretraining` is a later training-schedule control; `no_outcome_loss` masks
the result targets. `no_team_node` removes the explicit team nodes/edges but
preserves the same joint prediction target. The model-training implementation
must respect these contracts. A deterministic regression and probabilistic
heads have not yet been implemented in this package.

In the strict 40.2→41.1 fold, 41.1 has no observed spatial prior. This deliberately
avoids relabeling 40.2 connections as 41.1 observations. Any future transfer of
old geometry requires a separately marked inferred prior and an explicit test.

## Reproduce and inspect

```bash
python3 -m unittest discover -s tests/world_model -v
python3 -m tools.world_model_toy_checks
python3 tools/world_model_preprocess.py prepare --output /path/to/new_run
python3 tools/world_model_preprocess.py run --output /path/to/new_run --workers 8
python3 tools/world_model_preprocess.py finalize --output /path/to/new_run
python3 tools/verify_world_model_preprocess.py --root /path/to/new_run --output /path/to/report
python3 tools/report_world_model_preprocess.py --root /path/to/new_run --report /path/to/report
```

The DB password is prompted without echo and never saved. No package installation
or training command is required. The existing completed run must not be
overwritten. A cached raw-data run can use `run --offline` once every canonical
raw file is present and its provenance has been recorded.

The SQLite catalog indexes match quality, timestamped issues, outcome masks and
per-training-match spatial support. Raw files use pseudonymous within-match IDs.
The earlier v1 hash-split smoke test and v2 pre-terminal-censoring banks are
superseded; only v3 should feed a new training experiment.

## G1 encoder boundary (2026-10-20)

Use `load_sample` for model inputs, not the raw bank or `dataset.sample()` result.
It remaps sparse indices, applies unit scales and observable segment masks,
removes unverified observer `alive_teams`, and excludes the focal training
match's entire movement contribution from its spatial prior. Other fit matches
remain population-level prior knowledge. Frozen preprocessing is unchanged.

```python
import torch
from world_model.inputs import ConditionVocabulary, load_sample, collate_samples
from world_model.encoder import WorldEncoder

torch.set_num_threads(1)
torch.manual_seed(20261020)
torch.use_deterministic_algorithms(True)
torch.backends.mha.set_fastpath_enabled(False)  # one CPU path with/without gradients
vocabulary = ConditionVocabulary.from_dataset(dataset)
sample = load_sample(dataset, match_id, 0, vocabulary)
encoder = WorldEncoder(len(dataset.item_vocabulary)+2, vocabulary.sizes).cpu().eval()
latent = encoder(sample.inputs)
# sample.targets is separate; no future mask selects encoder nodes.
batch = collate_samples([sample])
latents = encoder.forward_batch(batch.inputs)
```

Outputs are player/team/cell/zone/game representations, with d=128 by default,
and a current player-valid mask. They are untrained, not generated future paths.
The encoder keeps individuals and represents the team as a member set with
pair geometry. Permutation tests include five-member synthetic teams.

Run `python3 -m pytest tests/world_model -q` for preprocessing and encoder tests.
See [G1 report](../reports/world_model_g1_2026-10-20/REPORT_KO.md) for failed
counterexamples, corrections, CPU receipts, limitations and the next stage.
