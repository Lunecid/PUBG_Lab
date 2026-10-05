# RQ4 Realtime Prefix Probability Demo

Date: 2026-07-06
Class: `ExperimentArtifact`
Lifecycle: `descriptive_demo_bounded`

## Purpose

This artifact demonstrates that the snapshot LSTM checkpoint can emit a
changing probability readout on the native 10-second snapshot clock.

## Contract

```text
S_i(tau)  = S_i(k-1) * (1 - lambda_i(k | x_{<=tau}))
pi_i(tau) = S_i(tau) / sum_j S_j(tau)
```

`tau` is a 10-second snapshot time inside phase `k`.

## Outputs

- csv: `/Users/lunecid/project/PUBG/reports/rq4_dynamic_probability_fluctuation/realtime_prefix_probability_trajectory.csv`
- figure: `/Users/lunecid/project/PUBG/reports/rq4_dynamic_probability_fluctuation/realtime_prefix_probability_example_match.png`

## Metadata

```json
{
  "checkpoint": "/Users/lunecid/project/PUBG/model_stgcn/checkpoints/br_rq4_sequence_model_zoo_required_v1/erangel_tpp_squad/snapshot_lstm_no_graph/model.pt",
  "model_variant": "snapshot_lstm_no_graph",
  "match_id": "10691322-1ab3-42be-b896-c9c76f3e10f7",
  "snapshots_total": 200,
  "prefix_rows": 2234,
  "test_match": true,
  "readout_contract": "10-second prefix readout from phase-cut-trained LSTM; descriptive demo, not separately supervised prefix metric.",
  "outputs": {
    "csv": "/Users/lunecid/project/PUBG/reports/rq4_dynamic_probability_fluctuation/realtime_prefix_probability_trajectory.csv",
    "figure": "/Users/lunecid/project/PUBG/reports/rq4_dynamic_probability_fluctuation/realtime_prefix_probability_example_match.png",
    "note": "/Users/lunecid/project/PUBG/reports/rq4_dynamic_probability_fluctuation/RQ4_REALTIME_PREFIX_PROBABILITY_DEMO_2026-07-06.md",
    "manifest": "/Users/lunecid/project/PUBG/reports/rq4_dynamic_probability_fluctuation/realtime_prefix_probability_manifest.json"
  }
}
```

## Boundary

This is a descriptive prefix-readout demo from a phase-cut-trained model.
Do not report it as a separately validated 10-second supervised metric until
a prefix-label evaluation protocol is added.
