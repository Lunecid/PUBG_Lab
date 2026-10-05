# Battle Royale Claim-Experiment Mapping

Date: 2026-07-06
Class: `ExperimentSynthesis`
Lifecycle: `active`

## Purpose

This document maps the revised Battle Royale thesis claims to the experiments
and results that currently support them.

It should be used as the manuscript-facing bridge between:

```text
research question -> required methodology -> experiment/result -> allowed claim
```

It is not a new experiment result.  It is a claim-to-evidence map for writing
the thesis in the revised RQ1-RQ4 order.

## Canonical Build-Up

The thesis build-up is now:

```text
RQ1. Battle Royale genre formalization
RQ2. PUBG official-telemetry representation as the representative game case
RQ3. Prior tabular / phase-aware prediction plus 10-second realtime
     elimination-risk modeling
RQ4. Phase-transition most-at-risk / most-advantaged team prediction
```

Warzone / Caldera and O7 graph-relation experiments are supplementary boundary
or reviewer-defense evidence.  They are not core RQ sections.

## Claim Status Legend

| Status | Meaning |
| --- | --- |
| `supported` | completed evidence can be used in main text with bounded wording |
| `supported with boundary` | main claim is usable, but a limitation must be stated next to it |
| `supplementary` | useful for boundary, robustness, or future-work positioning, not core contribution |
| `pending` | needed before stronger wording can be used |
| `not supported` | should be explicitly avoided or framed as a negative result |

## RQ-Level Claim Map

| RQ | Thesis claim | Required methodology | Supporting experiments / artifacts | Current result | Manuscript-safe wording | Do not claim |
| --- | --- | --- | --- | --- | --- | --- |
| `RQ1` | Battle Royale can be formalized as a genre-level survival system. | Define phase-conditioned multi-agent survival and reusable mechanism families `O0-O8`. | ontology lock, survival formalization, mini corpus coverage | `O0-O8` covered 15/15 source-rich games/modes; all families appear across the mini corpus without requiring `O9`. | Battle Royale can be modeled as phase-conditioned multi-agent survival with reusable mechanism families. | Do not claim an exhaustive universal ontology or causal proof of the genre. |
| `RQ2` | The formalization can be represented in PUBG official telemetry. | Map formal objects to leakage-guarded team x phase-cut variables and 10-second snapshot histories. | official telemetry observability table, operational manifest, feature availability audit, required cohort registry, OM0-OM6 ladder | PUBG fields are separated into direct telemetry, pre-cut derived features, proxy-only O6 signals, label-only outcomes, and not-provided geometry; OM0-OM6 feature blocks are executable on required cohorts. | PUBG is a label-rich supervised instantiation of the formalization. | Do not claim PUBG official telemetry directly provides terrain mesh, cover, LOS, or building occlusion. |
| `RQ3` | Prior tabular / phase-aware PUBG prediction can be extended into realtime elimination-risk readouts using 10-second telemetry. | Compare prior/basic tabular baselines, evaluate same-snapshot-surface temporal models, then convert phase hazards to cumulative survival and match-local survival share. | Sagong-style/basic tabular baseline result, same-condition sequence result, dynamic probability trajectory, 10-second prefix demo | Representation-rich pressure/habitat features beat prior/basic tabular baselines in 13/13 cohorts; phase-wise LSTM hazards produce dynamic survival-share trajectories over 13 cohorts; one-match 10-second prefix demo exists. | Prior phase-aware prediction can be extended into a causal telemetry survival readout that updates hazard, cumulative survival, and match-local survival share. | Do not claim general deep-learning superiority or separately validated realtime deployment. |
| `RQ4` | Realtime survival readouts can identify the most unfavorable and most favorable teams at phase transitions. | Convert hazard into a risk board and survival share into an advantage board; evaluate board-level top-k metrics and board baselines. | realtime risk-advantage framework, phase-transition board result, board-baseline comparison | LSTM board: 10,616 phase boards across 13 cohorts; median risk top-1 hit `0.7004`, median risk top-3 hit `0.9172`, median advantage top-team final top-10 `0.9490`; beats simple/prior boards, while pressure-habitat HGB is stronger as a phase-cut scorer. | Phase-transition boards can identify most-at-risk and most-advantaged teams; frame LSTM as the realtime readout form. | Do not claim LSTM is the strongest board scorer under all comparisons or that the 10-second deployment is separately validated. |

## Detailed Evidence Crosswalk

### RQ1. Genre Formalization

| Claim element | Evidence | Result | Claim status | Manuscript use |
| --- | --- | --- | --- | --- |
| Battle Royale should not be reduced to PUBG-specific features. | `BATTLE_ROYALE_ONTOLOGY_LOCK_2026-07-03.md` | `O0-O8` mechanism families are frozen for the current thesis path. | `supported` | Theory / genre formalization section |
| The mechanism families are not PUBG-only. | `BATTLE_ROYALE_MINI_CORPUS_COVERAGE_TABLE_2026-07-04.md` | `O0` through `O8` each receive `15/15` mini-corpus coverage. | `supported with boundary` | Table: genre coverage by family |
| Final rank alone is insufficient for the problem. | `BATTLE_ROYALE_SURVIVAL_FORMALIZATION_2026-07-03.md` | Survival is expressed as phase-cut hazard and cumulative survival. | `supported` | Formal problem definition |

Safe RQ1 sentence:

```text
This thesis formalizes Battle Royale as a phase-conditioned multi-agent
survival process with reusable mechanism families, rather than as a
PUBG-specific final-rank prediction task.
```

Boundary:

```text
The mini corpus supports the framework vocabulary; it does not prove that the
ontology is exhaustive for all possible Battle Royale variants.
```

### RQ2. PUBG Official-Telemetry Representation

| Claim element | Evidence | Result | Claim status | Manuscript use |
| --- | --- | --- | --- | --- |
| PUBG can instantiate the formalization. | `PUBG_OFFICIAL_TELEMETRY_OBSERVABILITY_TABLE_2026-07-05.md` | Direct, derived, proxy, label-only, and not-provided fields are separated. | `supported` | Main RQ2 data table |
| The feature mapping is executable without forbidden future inputs. | `BATTLE_ROYALE_FEATURE_AVAILABILITY_AUDIT_2026-07-03.md` | OM0-OM6 features pass availability / leakage gates on required cohorts. | `supported` | Data preprocessing / representation section |
| Cohort claims are not pooled across incompatible game conditions. | `BATTLE_ROYALE_REQUIRED_COHORT_SET_2026-07-03.md` | Required cohorts are stratified by map, perspective, and team size. | `supported` | Experiment setup |
| Representation blocks matter for supervised hazard prediction. | `BATTLE_ROYALE_OM_LADDER_RESULT_FEATURE_AUDIT_2026-07-04.md` | `OM2 -> OM3` pressure-habitat improves hazard AUC in 13/13 cohorts. | `supported` | RQ2/RQ4 mechanism validation |
| The pressure-habitat result is not merely pooled phase base-rate encoding. | `BATTLE_ROYALE_PHASE_STRATIFIED_OM_RESULT_2026-07-04.md`, `BATTLE_ROYALE_PHASE_GROUP_BOOTSTRAP_RESULT_2026-07-04.md` | Phase-stratified and paired bootstrap analyses support the mechanism boundary. | `supported` | Robustness paragraph |

Safe RQ2 sentence:

```text
PUBG official telemetry can instantiate the Battle Royale formalization as
leakage-guarded team-phase and 10-second snapshot representations, while also
making explicit which spatial constructs are only proxies or not observed.
```

Boundary:

```text
PUBG is label-rich but geometry-limited; it should not be used to claim true
terrain mesh, LOS, cover, or building-affordance measurement from official
telemetry alone.
```

### RQ3. Prior Baselines And Realtime Survival Readout

| Claim element | Evidence | Result | Claim status | Manuscript use |
| --- | --- | --- | --- | --- |
| The thesis improves over prior-style and basic tabular baselines on the same risk set. | `BATTLE_ROYALE_RQ4_PRIOR_BASELINE_RESULT_2026-07-05.md` | `a7_raw_plus_habitat - a0_official_raw`: 13/13 AUC wins, mean `+0.021690`; `a11_full_representation - a1_prior_raw`: 13/13 wins, mean `+0.059714`. | `supported` | Prior-work comparison table |
| Graph architecture is not the core contribution. | bounded graph-design early readout in prior-baseline result | no-graph AUC `0.776948`; observed graph AUC `0.774473`; shuffle controls comparable. | `not retained as core` | Design exclusion note |
| 10-second state histories can be modeled causally. | `BATTLE_ROYALE_RQ4_HIERARCHICAL_SURVIVAL_PROTOCOL_2026-07-05.md`, `BATTLE_ROYALE_RQ4_SAME_CONDITION_SEQUENCE_RESULT_2026-07-05.md` | Sequence accumulation beats neural last-only in most cohorts, but `snapshot_summary_hgb` remains strongest same-condition baseline. | `supported with boundary` | Model-form comparison |
| The model can output a changing probability trajectory, not only a static score. | `BATTLE_ROYALE_RQ4_DYNAMIC_PROBABILITY_FLUCTUATION_RESULT_2026-07-06.md` | 13 cohorts, 1,789 matches, 33,584 teams, 105,491 trajectory rows; median hazard AUC `0.7441`; median latest top-5 AUC from `pi` `0.9290`; median latest rank Spearman `0.7914`. | `supported with phase-cut boundary` | Dynamic trajectory figure/table |
| A 10-second prefix readout is technically demonstrable. | `RQ4_REALTIME_PREFIX_PROBABILITY_DEMO_2026-07-06.md` | one Erangel TPP squad test-match demo, 200 snapshots, 28 teams, 2,234 prefix rows. | `descriptive` | Optional illustration, not headline metric |

Safe RQ3 sentence:

```text
Compared with prior-style and basic tabular baselines, the proposed
representation supports a phase-wise survival readout that can be accumulated
into changing match-local survival probabilities from 10-second telemetry.
```

Boundary:

```text
The current evidence supports realtime readout form and phase-cut dynamic
trajectories; it does not establish general neural superiority over strong
tabular summary baselines, nor a separately validated deployed streaming model.
```

### RQ4. Phase-Transition Risk / Advantage Board

| Claim element | Evidence | Result | Claim status | Manuscript use |
| --- | --- | --- | --- | --- |
| Hazard can be interpreted as next-phase elimination risk. | `BATTLE_ROYALE_REALTIME_RISK_ADVANTAGE_FRAMEWORK_2026-07-06.md` | `RiskBoard_m(k) = sort_i(lambda_i(k), descending=True)`. | `supported as model contract` | Method section |
| Cumulative survival share can be interpreted as match-local advantage. | `BATTLE_ROYALE_REALTIME_RISK_ADVANTAGE_FRAMEWORK_2026-07-06.md` | `AdvantageBoard_m(k) = sort_i(pi_i(k), descending=True)`. | `supported as model contract` | Method section |
| The board identifies likely next-phase eliminations. | `BATTLE_ROYALE_RQ4_PHASE_TRANSITION_RISK_ADVANTAGE_BOARD_RESULT_2026-07-06.md`, `BATTLE_ROYALE_RQ4_BOARD_BASELINE_COMPARISON_RESULT_2026-07-06.md` | LSTM board: 10,616 phase boards; median risk top-1 hit `0.7004`; median risk top-3 hit `0.9172`; median risk MRR `0.8121`. It beats random, health/alive-count, zone-distance, and Sagong-style boards in 13/13 cohorts on risk top-1. | `supported with HGB-scoring boundary` | Main RQ4 result table |
| The board identifies high final-outcome advantage. | same board result plus board-baseline comparison | LSTM board median advantage top-team final top-10 `0.9490`; pressure-habitat HGB is slightly stronger on several phase-cut board metrics, so LSTM should be framed as the realtime readout form. | `supported with HGB-scoring boundary` | Main RQ4 result table |
| Board state changes over time. | same board result | mean leader change rate `0.4214`. | `supported descriptive` | Optional trajectory / board example |

Safe RQ4 sentence:

```text
The realtime survival readout can be converted into phase-transition boards
that rank teams by immediate elimination risk and match-local survival
advantage.
```

Boundary:

```text
The board comparison is complete, but it does not support a blanket claim that
LSTM is the strongest board scorer.  The safe claim is that the representation
pipeline supports strong phase-cut board scoring, while LSTM supplies the
dynamic realtime readout form.
```

## Supplementary Evidence Map

| Supplement | Why it exists | Result | How to use | How not to use |
| --- | --- | --- | --- | --- |
| O6Z vertical-proxy sensitivity | test whether PUBG `pos_z` can support fine terrain/high-ground affordance | corrected `O6Z` is a bounded negative result | shows PUBG official telemetry is geometry-limited | do not claim PUBG validates fine terrain / cover / LOS |
| Warzone / Caldera observability bridge | show what geometry-rich data can represent when PUBG official telemetry cannot | terrain, route, building, rough-mobility, and compound constructs can be represented descriptively | supplement / future O6 extension | do not claim supervised survival/rank validation from Caldera |
| O7 advantage-edge null-control | defend relation-topology boundary for one contact-gated edge stream | observed contact-gated advantage edges beat edge-shuffled null in required-cohort hazard and hazard-implied rank bridge | reviewer-defense supplement | do not make graph architecture the thesis contribution |
| O1 survival-capital audit | explain why weak incremental OM1 result is not feature breakage | O1 tensors are valid and have univariate hazard signal | boundary note on feature-block interpretation | do not claim O1 is irrelevant or broken |

## Manuscript Figure And Table Placement

| Manuscript location | Figure / table | Source artifact | Purpose |
| --- | --- | --- | --- |
| RQ1 / theory | Table: `O0-O8` mechanism families and mini-corpus coverage | mini corpus coverage table | show genre-level vocabulary support |
| RQ1 / method | Equation block: hazard, cumulative survival, match-local survival share | survival formalization and realtime framework | define the prediction object |
| RQ2 / data | Table: PUBG official telemetry observability | official telemetry observability table | show direct / derived / proxy / label / missing fields |
| RQ2 / validation | Table: required cohorts and feature availability | required cohort set and feature audit | show reproducible data contract |
| RQ2/RQ4 / mechanism | Table: OM ladder and `OM2 -> OM3` result | OM ladder, phase-stratified, bootstrap results | support pressure-habitat mechanism |
| RQ3 / prior comparison | Table: Sagong-style/basic tabular vs representation-rich features | prior-baseline result | position against previous tabular/phase prediction |
| RQ3 / model form | Table: same snapshot surface tabular vs TCN/GRU/LSTM | same-condition sequence result | bound the DL-vs-tabular claim |
| RQ3 / realtime readout | Figure: dynamic probability trajectory | dynamic probability result | show changing survival probabilities |
| RQ4 / board | Table: phase-transition risk/advantage board metrics | board result | show most-at-risk / most-advantaged prediction |
| Supplement | O6 observability figure/table | Warzone / Caldera bridge and construct artifact | show geometry-rich future extension |
| Supplement | O7 null-control table | O7 required-cohort hazard/rank bridge | reviewer-defense relation-topology evidence |

## Completed Gap Before Stronger RQ4 Wording

The previously missing experiment is now complete:

```text
RQ4 risk/advantage board baseline comparison
```

Minimum baseline panel:

| Baseline board | Role |
| --- | --- |
| random alive-team board | sanity floor |
| health / alive-count board | simple survival-capital baseline |
| zone-distance / outside-safe board | simple pressure baseline |
| Sagong-style score board | prior-work score baseline |
| official raw tabular hazard board | basic telemetry baseline |
| pressure-habitat HGB board | strongest mechanism-aligned tabular baseline |
| proposed LSTM survival-share board | current realtime readout board |

Produced outputs:

```text
aggregate board table
cohort-level board table
sign-count comparison
one phase-transition example board
claim-boundary note
```

RQ4 now moves to:

```text
supported against simple/prior-work board baselines with HGB-scoring boundary
```

The boundary branch is the one selected by the result:

```text
supported as an interpretable board readout, but not superior to the strongest
baseline board.
```

Both outcomes are thesis-usable as long as the claim boundary is explicit.
