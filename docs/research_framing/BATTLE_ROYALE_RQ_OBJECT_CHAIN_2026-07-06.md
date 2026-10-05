# Battle Royale RQ Object Chain

Date: 2026-07-06
Package: research_framing
Lifecycle: active
Responsibility: Define the object-oriented research-question chain for the manuscript.
Depends on: docs/research_framing/BATTLE_ROYALE_RESEARCH_QUESTION_CHAIN_2026-07-05.md
Used by: docs/paper_writing/BATTLE_ROYALE_MANUSCRIPT_ASSEMBLY_PLAN_2026-07-05.md

## Purpose

This document turns the manuscript's four research questions into explicit
documentation objects.

Each research question owns:

```text
RQ definition
-> main claim
-> claim objects
-> experiment objects
-> evidence objects
-> result objects
-> claim boundary
```

The goal is to prevent the thesis from becoming a chronological experiment log.
Experiments are children of claims.  Claims are children of research questions.
Research questions are children of the manuscript chain.

## Object Model

| Object | Responsibility |
| --- | --- |
| `RQObject` | states the research question and its role in the paper |
| `ClaimObject` | states one manuscript-safe claim under that RQ |
| `ExperimentObject` | validates or bounds one or more claim objects |
| `EvidenceObject` | points to the actual protocol, result, table, figure, or generated artifact |
| `ResultObject` | records the result value or qualitative decision used in the manuscript |
| `BoundaryObject` | states what the claim does not support |

Dependency rule:

```text
RQObject owns ClaimObject.
ClaimObject depends on ExperimentObject.
ExperimentObject produces EvidenceObject and ResultObject.
BoundaryObject constrains ClaimObject.
```

## RQ Chain

| RQ object | Question | Main claim | Canonical object document |
| --- | --- | --- | --- |
| `RQ1GenreFormalization` | How can the Battle Royale genre be formalized? | Battle Royale can be modeled as phase-conditioned multi-agent survival with reusable mechanism families. | `BATTLE_ROYALE_RQ1_GENRE_FORMALIZATION_OBJECT_2026-07-06.md` |
| `RQ2PUBGRepresentation` | How can that formalization be represented in PUBG official telemetry? | PUBG official telemetry can instantiate the formalization as leakage-guarded team-phase and 10-second snapshot representations. | `BATTLE_ROYALE_RQ2_PUBG_REPRESENTATION_OBJECT_2026-07-06.md` |
| `RQ3RealtimeSurvivalReadout` | How can prior tabular and phase-aware prediction be extended into realtime elimination-risk modeling? | Prior phase-aware prediction can be extended into a causal telemetry survival readout that updates hazard, cumulative survival, and match-local survival share. | `BATTLE_ROYALE_RQ3_REALTIME_SURVIVAL_READOUT_OBJECT_2026-07-06.md` |
| `RQ4RiskAdvantageBoard` | Can the realtime survival readout identify unfavorable and favorable teams at phase transitions? | Phase-transition risk and advantage boards can identify most-at-risk and most-advantaged teams, with an HGB-scoring boundary. | `BATTLE_ROYALE_RQ4_RISK_ADVANTAGE_BOARD_OBJECT_2026-07-06.md` |

## Chain Invariants

1. `RQ1` must define the genre before any PUBG-specific feature claim appears.
2. `RQ2` must map the formalization into observed PUBG telemetry before any
   predictive result is interpreted.
3. `RQ3` must position prior tabular / phase-aware work before introducing the
   realtime readout.
4. `RQ4` must use the realtime survival readout as a board, not merely as a
   row-level AUC score.
5. Supplementary O6Z, Caldera / Warzone, and O7 edge-stream evidence can defend
   boundaries, but they do not replace the four RQ objects.

## Manuscript Assembly Interface

The manuscript can use the object chain as:

```text
Introduction:
  problem -> RQ chain

Method:
  RQ1 formalization
  RQ2 telemetry representation
  RQ3 realtime survival readout
  RQ4 board construction

Results:
  claim objects in the same order

Discussion:
  boundary objects and future work
```

## Current Status

| RQ object | Status | Main boundary |
| --- | --- | --- |
| `RQ1GenreFormalization` | supported as framework vocabulary | not an exhaustive universal ontology |
| `RQ2PUBGRepresentation` | supported as data contract | PUBG is geometry-limited |
| `RQ3RealtimeSurvivalReadout` | supported with phase-cut / demo boundary | no general deep-learning superiority or deployment claim |
| `RQ4RiskAdvantageBoard` | supported with HGB-scoring boundary | LSTM is not the strongest board scorer under all comparisons |
