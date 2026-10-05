# Battle Royale O7 Edge-Shuffle Audit

Date: 2026-07-05
Class: `ExperimentResult`
Lifecycle: `active`

## Purpose

This audit checks the immediate pre-run guard for the locked O7
advantage-edge protocol:

```text
Verify that edge_shuffle rewires only within the active-node mask and preserves
the intended edge-vector distribution for the advantage-edge implementation.
```

This is not an O7 predictive result. It is a null-control readiness check
before running the A/B/C advantage-edge experiment.

## Code Surface

| Component | Role |
| --- | --- |
| `model_stgcn/v3/advantage_edges.py` | builds contact-gated relative-advantage edge tensors |
| `model_stgcn/v3/adv_model.py::shuffle_adj` | applies node / edge shuffle controls |
| `tools/audit_o7_advantage_edge_shuffle.py` | real-match audit of edge-shuffle invariants |
| `scripts/adv_precompute.py` | cohort-aware advantage-edge cache generation |
| `scripts/run_br_o7_advantage_edge_early_readout.sh` | A/B/C early-readout runner |
| `tests/test_v3_edge_controls.py` | synthetic unit test for advantage-edge shuffle invariants |

## Audit Contract

For each audited match-phase alive-node block, the edge-shuffle control must:

1. keep shuffled mask and edge features inside the alive-node block;
2. preserve the number of active edges;
3. preserve the paired mask + edge-feature vector multiset inside the block;
4. rewire nontrivial edge blocks so the null is not accidentally identical.

The checked edge stream is:

```text
gate = knn
k = 5
beta = 1.0
edge features = g_combat, g_knn, affinity, d_over_ell,
                combat_dmg_log, combat_kills,
                dz, dalive, dhealth, dzone
```

## Real-Match Audit Result

Command:

```text
PG_PASSWORD=... python3 tools/audit_o7_advantage_edge_shuffle.py \
  --cohort "Erangel TPP squad" \
  --max-matches 2
```

Artifact:

```text
model_stgcn/checkpoints/br_o7_advantage_edge_required_v1/erangel_tpp_squad/o7_edge_shuffle_audit.json
```

Result:

| Field | Value |
| --- | ---: |
| Status | `passed` |
| Cohort | Erangel TPP squad |
| Split | test |
| Matches audited | 2 |
| Phase blocks audited | 13 |
| Edge-containing blocks | 13 |
| Rewired blocks | 13 |
| Total active edges checked | 724 |
| Failures | 0 |

Interpretation:

```text
The current O7 advantage-edge edge_shuffle implementation passes the short
real-match null-control audit on Erangel TPP squad.
```

## Cache / Runner Smoke

A tiny cache-only smoke also passed:

```text
O7_ROOT=/private/tmp/pubg_o7_smoke
O7_COHORT="Erangel TPP squad"
O7_MAX_TRAIN=2
O7_MAX_VAL=1
O7_MAX_TEST=1
O7_CACHE_ONLY=1
bash scripts/run_br_o7_advantage_edge_early_readout.sh
```

The generated cache used:

| Field | Value |
| --- | --- |
| Feature preset | `om5_relation_structure` |
| Node absolute control features | enabled |
| Gates | `knn` |
| Split JSON | `model_stgcn/checkpoints/br_om_ladder_required_v1/erangel_tpp_squad/split.json` |

A tiny 1-epoch CPU A/B/C training smoke also passed on the same cache:

| Arm | Gate | Control | Prediction artifact |
| --- | --- | --- | --- |
| `A` | `none` | `observed` | `/private/tmp/pubg_o7_smoke/erangel_tpp_squad/A_none_observed/pred_test.csv` |
| `B` | `knn` | `observed` | `/private/tmp/pubg_o7_smoke/erangel_tpp_squad/B_knn_observed/pred_test.csv` |
| `C` | `knn` | `edge_shuffle` | `/private/tmp/pubg_o7_smoke/erangel_tpp_squad/C_knn_edge_shuffle/pred_test.csv` |

The smoke outputs are not predictive evidence because they use only a tiny
2/1/1 match cache and 1 training epoch.

## Claim Boundary

Allowed:

```text
The O7 advantage-edge null-control and cohort-aware cache/run path are ready
for high-power early readout.
```

Not allowed:

```text
O7 advantage-edge topology improves survival prediction.
```

That claim requires the gen-2 A/B/C required-cohort runs and phase-group
analysis.

## Next Action

Launch the high-power early readout cohorts under:

```text
model_stgcn/checkpoints/br_o7_advantage_edge_required_v1/
```

Start with:

```text
Erangel TPP squad
Taego TPP squad
Erangel TPP solo
Miramar / Desert TPP squad
```
