# Paper ST Feature Fast MPS Result (2026-06-23)

## Run

Purpose:

```text
Fast paper-facing orchestration for the prior-aligned ST feature design.
```

Command envelope:

```text
PAPER_ST_OUT_DIR=model_stgcn/checkpoints/paper_st_feature_mps_fast500_e2_20260623
PAPER_ST_LIMIT=500
PAPER_ST_EPOCHS=2
PAPER_ST_BATCH=8
PAPER_ST_DEVICE=mps
PAPER_ST_WORKERS=4
PAPER_ST_EVAL_WORKERS=2
PAPER_ST_INCLUDE_SPARSE=0
```

Artifacts:

```text
model_stgcn/checkpoints/paper_st_feature_mps_fast500_e2_20260623/paper_st_metrics_seed42.csv
model_stgcn/checkpoints/paper_st_feature_mps_fast500_e2_20260623/paper_st_config_seed42.json
model_stgcn/checkpoints/paper_st_feature_mps_fast500_e2_20260623/*_seed42.summary.json
```

## Metrics

| Variant | Input contract | Best val loss | Test loss | Test acc | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| `snapshot_last_only` | `X_s`, no graph | 0.6272 | 0.6246 | 0.6705 | latest snapshot baseline |
| `snapshot_tcn_no_graph` | `X_s`, no graph | 0.5827 | 0.5746 | 0.7263 | best fast-run result; temporal ST dynamics are useful |
| `snapshot_rgcn` | `X_s + A_s` | 0.5877 | 0.5823 | 0.7247 | relation channel does not beat TCN on this fast split |
| `snapshot_rgcn_edge_shuffle` | `X_s + shuffled A_s` | 0.5878 | 0.5829 | 0.7227 | observed topology is nearly tied with topology null |
| `snapshot_gate_only` | `X_s + binary contact A_s` | 0.5874 | 0.5819 | 0.7237 | graph capacity/contact control is slightly best among graph variants |

## Reading

Primary fast-run signal:

```text
snapshot_tcn_no_graph improves over snapshot_last_only by about 0.0500 test
loss and +0.0557 test accuracy.
```

Safe interpretation:

```text
The prior-aligned snapshot ST node stream carries temporal signal. The sequence
of team trajectory, zone pressure, viability, and resource state is materially
better than using only the latest cut snapshot on this 500-match fast split.
```

Relation-channel interpretation:

```text
snapshot_rgcn is worse than snapshot_tcn_no_graph by about 0.0077 test loss.
snapshot_rgcn, edge_shuffle, and gate_only are nearly tied, with gate_only
slightly best among graph variants. This does not support an observed-topology
benefit in this fast run.
```

Paper claim status:

```text
Supported as a fast-pilot claim:
  temporal ST feature representation is useful.

Not supported by this fast pilot:
  observed graph topology adds robust signal beyond temporal node features.
```

This result should be treated as a fast MPS pilot, not the final paper result.
It should be followed by either seed repeats at the same 500-match scale or a
focused all-match continuation if wall-clock budget permits.
