# Battle Royale RQ4 Phase-Transition Risk/Advantage Board Result

Date: 2026-07-06
Class: `ExperimentResult`
Lifecycle: `analysis_complete_bounded`

## Purpose

This experiment evaluates the proposed realtime risk-advantage framework as a
phase-transition board.

The question is:

```text
At each phase cut, can the model identify the teams most likely to be
eliminated next and the teams currently in the strongest survival position?
```

This is the direct experiment for the manuscript idea:

```text
realtime elimination-risk prediction
+ phase-transition most-at-risk / most-advantaged team board
```

## Source

Input trajectory artifact:

```text
reports/rq4_dynamic_probability_fluctuation/dynamic_probability_trajectories.csv
```

Model variant:

```text
snapshot_lstm_no_graph
```

The input trajectories were produced from completed LSTM phase-cut hazard
predictions across the 13 required cohorts.

## Board Definitions

Risk board:

```text
RiskBoard_m(k) = sort_i(lambda_i(k), descending=True)
```

Advantage board:

```text
AdvantageBoard_m(k) = sort_i(pi_i(k), descending=True)
```

Where:

```text
lambda_i(k) = predicted next-phase elimination hazard
pi_i(k)     = match-local survival share
```

## Execution

Executable:

```text
tools/eval_phase_transition_risk_board.py
```

Command:

```bash
python3 tools/eval_phase_transition_risk_board.py \
  --trajectories reports/rq4_dynamic_probability_fluctuation/dynamic_probability_trajectories.csv \
  --out-dir reports/rq4_dynamic_probability_fluctuation \
  --top-k 5
```

## Outputs

| Artifact | Path |
| --- | --- |
| phase board rows | `reports/rq4_dynamic_probability_fluctuation/phase_transition_risk_advantage_board.csv` |
| cohort summary | `reports/rq4_dynamic_probability_fluctuation/phase_transition_risk_advantage_by_cohort.csv` |
| aggregate summary | `reports/rq4_dynamic_probability_fluctuation/phase_transition_risk_advantage_aggregate.csv` |
| generated report | `reports/rq4_dynamic_probability_fluctuation/RQ4_PHASE_TRANSITION_RISK_ADVANTAGE_BOARD_RESULT_2026-07-06.md` |
| manifest | `reports/rq4_dynamic_probability_fluctuation/phase_transition_risk_advantage_manifest.json` |

## Coverage

| Field | Value |
| --- | ---: |
| cohorts | 13 |
| matches | 1,789 |
| phase-transition boards | 10,616 |
| top-k list size | 5 |

## Aggregate Result

| Model | Cohorts | Phase boards | Median risk top-1 hit | Median risk top-3 hit | Median risk MRR | Median advantage top-team final top-5 | Median advantage top-team final top-10 | Median advantage top-5 contains final top-10 | Median advantage NDCG@5 vs top-10 | Mean leader change rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `snapshot_lstm_no_graph` | 13 | 10,616 | 0.7004 | 0.9172 | 0.8121 | 0.7768 | 0.9490 | 1.0000 | 0.9381 | 0.4214 |

Global row-weighted means:

| Metric | Mean |
| --- | ---: |
| risk top-1 hit | 0.7174 |
| risk top-3 hit | 0.9182 |
| risk top-5 hit | 0.9668 |
| risk MRR | 0.8226 |
| advantage top-team final top-5 | 0.7683 |
| advantage top-team final top-10 | 0.9331 |
| advantage top-5 contains final top-10 | 0.9972 |

## Interpretation

The risk board is meaningful as a phase-transition alert:

```text
The highest-risk team is actually eliminated in the next phase in roughly
70-72% of evaluated phase boards with at least one next-phase elimination, and
the top-3 risk list contains a next-phase eliminated team in about 92% of
boards.
```

The advantage board is meaningful as a final-outcome bridge:

```text
The top-ranked advantaged team at a phase cut finishes in the final top-10 in
about 93-95% of boards, and the top-5 advantage list almost always contains a
final top-10 team.
```

This supports the framework form:

```text
phase hazard -> live risk board
cumulative survival share -> phase advantage board
```

## Supported Claim

```text
The dynamic survival readout can be converted into phase-transition boards
that identify the highest next-phase elimination risk and the strongest
match-local survival advantage at each phase cut.
```

## Boundary

Do not claim:

```text
This is already the final validated board model.
```

This board uses the existing phase-cut trajectory artifact.  It must still be
compared against explicit board baselines:

```text
Sagong-style phase score
official raw / tabular baseline
health / alive-count baseline
zone-distance baseline
pressure-habitat HGB baseline
```

Do not claim:

```text
This validates the 10-second prefix streaming demo.
```

The board metrics here are phase-cut metrics.  The 10-second prefix readout is
currently a descriptive demonstration only.

## Manuscript Placement

Use this result after the dynamic probability trajectory result:

```text
phase-wise hazard model
-> dynamic probability trajectory
-> phase-transition risk/advantage board
-> final rank/top-k bridge
```

Recommended wording:

```text
At each phase transition, the proposed framework can rank teams by next-phase
elimination risk and by match-local survival advantage, producing an
interpretable risk/advantage board rather than only a scalar final-placement
prediction.
```
