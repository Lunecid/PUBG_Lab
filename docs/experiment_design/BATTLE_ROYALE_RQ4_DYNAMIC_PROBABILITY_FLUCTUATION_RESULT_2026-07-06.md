# Battle Royale RQ4 Dynamic Probability Fluctuation Result

Date: 2026-07-06
Class: `ExperimentResult`
Lifecycle: `analysis_complete_bounded`

## Purpose

This experiment adds the realtime survival readout that the prior Sagong-style score
baseline does not provide: a changing team-level probability trajectory during
the match.

The question is not:

```text
Does a sequence model beat every tabular baseline?
```

The question is:

```text
Can the phase-wise Battle Royale survival formulation be rendered as a
dynamic probability model whose outputs change as the match progresses?
```

## Model Definition

For each match `m`, team `i`, and observed phase cut `k`:

```text
lambda_i(k) = predicted next-phase elimination hazard
S_i(k)      = product_{j <= k} (1 - lambda_i(j))
pi_i(k)     = S_i(k) / sum_j S_j(k) within the same match and phase cut
```

`pi_i(k)` is a match-local survival-share / win-probability proxy.  It is not a
bookmaker-calibrated win probability.

## Source

The current artifact uses the completed sequence-model zoo readout:

```text
model_stgcn/checkpoints/br_rq4_sequence_model_zoo_required_v1
```

Primary model variant:

```text
snapshot_lstm_no_graph
```

This variant was selected because it is the best completed sequence-family
required-cohort model and has stored phase-cut prediction rows.  The dynamic
trajectory transformation is model-agnostic: any leakage-safe phase-wise hazard
predictor can be converted into `S_i(k)` and `pi_i(k)`.

## Execution

Executable:

```text
tools/build_dynamic_probability_fluctuation.py
```

Command:

```bash
python3 tools/build_dynamic_probability_fluctuation.py \
  --root model_stgcn/checkpoints/br_rq4_sequence_model_zoo_required_v1 \
  --model-variant snapshot_lstm_no_graph \
  --out-dir reports/rq4_dynamic_probability_fluctuation \
  --example-cohort erangel_tpp_squad \
  --top-n 8
```

The script carries cumulative survival forward through missing phase rows so
that match-local normalization does not change merely because a team lacks an
explicit readout at a later cut.

## Outputs

| Artifact | Path |
| --- | --- |
| trajectory rows | `reports/rq4_dynamic_probability_fluctuation/dynamic_probability_trajectories.csv` |
| per-team summaries | `reports/rq4_dynamic_probability_fluctuation/dynamic_probability_team_summary.csv` |
| per-match summaries | `reports/rq4_dynamic_probability_fluctuation/dynamic_probability_match_summary.csv` |
| cohort metrics | `reports/rq4_dynamic_probability_fluctuation/dynamic_probability_metrics_by_cohort.csv` |
| example figure | `reports/rq4_dynamic_probability_fluctuation/dynamic_probability_example_match.png` |
| generated report | `reports/rq4_dynamic_probability_fluctuation/BATTLE_ROYALE_RQ4_DYNAMIC_PROBABILITY_FLUCTUATION_RESULT_2026-07-06.md` |
| manifest | `reports/rq4_dynamic_probability_fluctuation/dynamic_probability_manifest.json` |
| 10-second prefix demo CSV | `reports/rq4_dynamic_probability_fluctuation/realtime_prefix_probability_trajectory.csv` |
| 10-second prefix demo figure | `reports/rq4_dynamic_probability_fluctuation/realtime_prefix_probability_example_match.png` |
| 10-second prefix demo note | `reports/rq4_dynamic_probability_fluctuation/RQ4_REALTIME_PREFIX_PROBABILITY_DEMO_2026-07-06.md` |
| phase-transition board CSV | `reports/rq4_dynamic_probability_fluctuation/phase_transition_risk_advantage_board.csv` |
| phase-transition board result | `docs/experiment_design/BATTLE_ROYALE_RQ4_PHASE_TRANSITION_RISK_ADVANTAGE_BOARD_RESULT_2026-07-06.md` |

## Coverage

| Field | Value |
| --- | ---: |
| prediction files | 13 |
| cohorts | 13 |
| matches | 1,789 |
| teams | 33,584 |
| observed prediction rows | 103,860 |
| trajectory rows after carry-forward | 105,491 |

## Aggregate Result

| Model | Cohorts | Median hazard AUC | Mean hazard AUC | Median latest top-5 AUC from `pi` | Median latest top-10 AUC from `pi` | Median latest rank Spearman | Median team probability volatility | Mean leader changes / match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `snapshot_lstm_no_graph` | 13 | 0.7441 | 0.7379 | 0.9290 | 0.9033 | 0.7914 | 0.0296 | 2.5292 |

Interpretation:

```text
The LSTM hazard model does not beat the strongest tabular representation
models on hazard AUC, but its phase-cut hazards can be accumulated into
dynamic survival-share trajectories that update throughout a match.
```

## Example Figure

```text
reports/rq4_dynamic_probability_fluctuation/dynamic_probability_example_match.png
```

Selected example:

| Field | Value |
| --- | --- |
| cohort | `erangel_tpp_squad` |
| model | `snapshot_lstm_no_graph` |
| match | `10691322-1ab3-42be-b896-c9c76f3e10f7` |

The upper panel plots `pi_i(k)`, the match-local survival share.  The lower
panel plots `lambda_i(k)`, the next-phase elimination hazard.

## 10-Second Prefix Demo

A descriptive real-time-style demo was also generated from the same Erangel TPP
squad LSTM checkpoint and the same test match:

```text
S_i(tau)  = S_i(k-1) * (1 - lambda_i(k | x_{<=tau}))
pi_i(tau) = S_i(tau) / sum_j S_j(tau)
```

where `tau` is the native 10-second snapshot time inside phase `k`.

| Field | Value |
| --- | ---: |
| snapshots in match tensor | 200 |
| prefix probability rows | 2,234 |
| unique teams in prefix rows | 28 |
| unique prefix snapshots represented | 146 |
| phase range | 1-7 |

Use the prefix figure as a demonstration that the trained snapshot LSTM can emit
changing probability readouts at the 10-second clock.  Do not report it as a
new validated metric yet.

## Phase-Transition Board Addendum

The dynamic trajectory was also converted into a phase-transition board:

```text
RiskBoard_m(k)      = sort_i(lambda_i(k), descending=True)
AdvantageBoard_m(k) = sort_i(pi_i(k), descending=True)
```

Aggregate board readout:

| Field | Value |
| --- | ---: |
| phase boards | 10,616 |
| median risk top-1 hit | 0.7004 |
| median risk top-3 hit | 0.9172 |
| median risk MRR | 0.8121 |
| median advantage top-team final top-10 | 0.9490 |
| median advantage top-5 contains final top-10 | 1.0000 |

Detailed result:

```text
docs/experiment_design/BATTLE_ROYALE_RQ4_PHASE_TRANSITION_RISK_ADVANTAGE_BOARD_RESULT_2026-07-06.md
```

## Supported Claim

```text
The proposed Battle Royale survival formulation supports a dynamic probability
readout: at each observed phase cut, the model updates next-phase elimination
hazard, cumulative survival, and match-local survival share for teams still in
the at-risk set.
```

This claim directly separates the thesis from a static score or final-rank
prediction baseline.  The output is not only a final label or a phase score; it
is a probability trajectory over the course of the match.

## Boundary

Do not claim:

```text
The LSTM sequence model is the best overall predictor.
```

The strongest completed hazard-AUC models remain the mechanism-aligned tabular
representation models, especially the pressure-habitat family.

Do not claim:

```text
This is a separately validated 10-second real-time deployment model.
```

The aggregate metrics in this document are phase-cut metrics.  The 10-second
prefix demo shows an executable readout path, but strict real-time validation
still requires a dedicated prefix-label evaluation protocol, for example:

```text
h_i(tau) -> lambda_i(k | x_{<= tau}) -> S_i(tau) -> pi_i(tau)
```

where `tau` is a 10-second snapshot time inside or before phase cut `k`.

## Manuscript Placement

Use this as the revised RQ3 realtime prediction evidence after the prior-baseline
and same-condition sequence comparison:

```text
Sagong-style / basic tabular baseline
-> mechanism-aligned phase hazard prediction
-> cumulative survival score
-> dynamic match-local probability trajectory
-> final top-k / rank bridge
```

Then use the phase-transition board result as revised RQ4:

```text
dynamic match-local probability trajectory
-> risk board and advantage board
-> most-at-risk / most-advantaged team identification
```

The manuscript-safe wording is:

```text
Prior PUBG phase-score work motivates phase-aware prediction, but the proposed
formulation treats Battle Royale outcome prediction as a dynamic survival
process: phase hazards update cumulative survival and match-local probability
trajectories as the match unfolds.
```
