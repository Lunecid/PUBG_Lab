# Paper ST Feature Matrix Runbook (2026-06-23)

## Purpose

This runner is separated from the exploratory graph/null matrix for the short
paper question:

```text
Does a phase-aligned 10-second spatio-temporal database representation improve
survival prediction over basic non-relational baselines, and do relation
channels add signal beyond temporal-only and graph-null controls?
```

Use:

```text
scripts/run_snapshot_paper_st_feature_matrix.sh
```

ST feature definition:

```text
docs/feature_engineering/ST_FEATURE_DEFINITION_2026-06-23.md
docs/ST_FEATURE_PRIOR_ALIGNED_DESIGN_2026-06-23.md
```

Do not use the broader sparse/null matrix as the primary paper runner.

## Core Variants

| Variant | Role | Channel contract |
| --- | --- | --- |
| `snapshot_last_only` | basic non-temporal baseline | `X_s`, no graph |
| `snapshot_tcn_no_graph` | temporal ST feature baseline | `X_s`, no graph |
| `snapshot_rgcn` | observed relation-channel model | `X_s + A_s` |
| `snapshot_rgcn_edge_shuffle` | topology-null graph control | `X_s + shuffled A_s` |
| `snapshot_gate_only` | contact-gate/capacity control | `X_s + binary contact A_s` |

Sparse relative-edge variants are not part of the default paper matrix. They
can be added with `PAPER_ST_INCLUDE_SPARSE=1`, but should be treated as
preliminary/future unless a later repeat justifies them.

## Smoke Command

```bash
PG_PASSWORD=... \
PAPER_ST_OUT_DIR=model_stgcn/checkpoints/paper_st_feature_smoke20 \
PAPER_ST_LIMIT=20 \
PAPER_ST_EPOCHS=1 \
PAPER_ST_FORCE=1 \
./scripts/run_snapshot_paper_st_feature_matrix.sh
```

Smoke result:

```text
model_stgcn/checkpoints/paper_st_feature_smoke20/paper_st_metrics_seed42.csv
```

The smoke completed all five core variants and wrote a metrics CSV. Treat it
only as orchestration validation, not as a result for the paper.

## Paper Run Command

Use the recovered all-match split from the seed-42 narrow matrix unless a new
split policy is intentionally chosen:

```bash
PG_PASSWORD=... \
PAPER_ST_OUT_DIR=model_stgcn/checkpoints/paper_st_feature_e3_seed42 \
PAPER_ST_SPLIT_JSON=model_stgcn/checkpoints/snapshot_narrow_full_sleep6h_20260623_0022/split_seed42.json \
PAPER_ST_EPOCHS=3 \
PAPER_ST_BATCH=2 \
PAPER_ST_HIDDEN=16 \
PAPER_ST_MID=32 \
PAPER_ST_NUM_BLOCKS=1 \
PAPER_ST_EDGE_HIDDEN=8 \
PAPER_ST_WORKERS=2 \
PAPER_ST_EVAL_WORKERS=1 \
PAPER_ST_PREFETCH_FACTOR=2 \
PAPER_ST_CAFFEINATE=1 \
./scripts/run_snapshot_paper_st_feature_matrix.sh
```

Primary outputs:

```text
paper_st_config_seed42.json
paper_st_metrics_seed42.csv
paper_st_status_seed42.json
*_seed42.summary.json
*_seed42.pt
```

## Resume / Recovery

The runner is resumable:

- if `*.summary.json` exists and `PAPER_ST_FORCE != 1`, the variant is skipped;
- if `*.pt` exists but `*.summary.json` is missing, the runner calls
  `tools/eval_snapshot_phase_checkpoint.py` to recover val/test metrics without
  retraining;
- use `PAPER_ST_ONLY=variant_a,variant_b` to resume or extend selected
  variants.

Example:

```bash
PG_PASSWORD=... \
PAPER_ST_OUT_DIR=model_stgcn/checkpoints/paper_st_feature_e3_seed42 \
PAPER_ST_ONLY=snapshot_tcn_no_graph,snapshot_rgcn \
./scripts/run_snapshot_paper_st_feature_matrix.sh
```

## Interpretation Plan

Read the resulting table in this order:

| Comparison | Paper meaning |
| --- | --- |
| `snapshot_last_only` vs `snapshot_tcn_no_graph` | temporal ST feature value |
| `snapshot_tcn_no_graph` vs `snapshot_rgcn` | relation-channel added value |
| `snapshot_rgcn` vs `snapshot_rgcn_edge_shuffle` | observed topology value |
| `snapshot_rgcn` vs `snapshot_gate_only` | relation richness beyond binary contact/capacity |

Safe current claim:

```text
The all-match one-epoch evidence favors the R-GCN relation-channel model, but
the paper claim should remain centered on the ST database representation until
2-3 epoch or seed-repeat confirmation is available.
```

Prior-work framing:

```text
The experiment is not just "more PUBG features." It tests a prior-aligned ST
representation: trajectory state, shrinking-arena context, time-varying
viability/resource covariates, interaction event fields, local opponent fields,
and typed relation topology under phase-cut survival labels.
```
