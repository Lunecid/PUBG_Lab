# Battle Royale Manuscript Assembly Plan

Date: 2026-07-05
Class: `ManuscriptSurface`
Lifecycle: `active`

## Decision

The next core work is manuscript-directed.  The RQ4 model-form comparison is
now complete and bounded, so the prediction chapter can be finalized without
starting a new architecture search:

```text
Sagong-style / official raw tabular baselines
-> feature-only Battle Royale representation ablation
-> OM2 -> OM3 pressure-habitat mechanism
-> same-snapshot-surface tabular baselines vs hierarchical phase-survival model
```

This is not exploratory feature hunting.  The prior-work/basic-tabular
comparison is complete.  The same-snapshot-surface comparison shows that
temporal sequence models usually improve over a neural phase-cut-only lower
bound, but strong tabular snapshot-summary reductions remain the aggregate
winner.  Graph design has been removed from the manuscript core after the
bounded early readout failed to support a hazard-AUC improvement over no-graph
and shuffle controls.

The active thesis-writing root is:

```text
/Users/lunecid/학위논문
```

The manuscript is written as LaTeX/TEX files in that folder, following the
Korean master's thesis guide stored there:

```text
/Users/lunecid/학위논문/학위논문작성가이드(국문석사).pdf
```

The paper must be built in this order:

```text
problem formalization
-> required methodology / representation
-> data construction and preprocessing justification
-> experiments and results
-> interpretation, claim boundary, and future work
```

## Master's Thesis Mode

This manuscript is a master's thesis, not a compressed conference or journal
paper.

Therefore the writing goal is not only to prove one sharp contribution.  The
goal is to demonstrate that the researcher can:

- define a research problem;
- understand and position related work;
- construct and audit the data;
- design representations and experiments;
- interpret both positive and weak/null results;
- explain limits without overclaiming;
- identify future research directions.

Weak or bounded results should not be hidden.  They should be used to show
research judgment.  For example, the bounded graph early readout explains why
the core thesis does not rely on graph architecture, and the corrected O6Z
negative result justifies the PUBG-to-Warzone geometry observability bridge.

## Immediate Priority

Create the manuscript-facing package that turns the current research artifacts
into section drafts, figure/table specifications, captions, and claim-safe
result wording.

## Work Items

| Priority | Work item | Output | Source artifacts | Done when |
| --- | --- | --- | --- | --- |
| `P0` | Freeze the RQ4 comparison requirement | one paragraph in the manuscript plan and checklist | RQ chain, prior-work frame, evidence ledger, RQ4 prior baseline result, hierarchical survival protocol | RQ4 no longer jumps from OM ladder directly to final rank claims |
| `P0` | Integrate RQ4 prior baseline comparison | required-cohort result package | `BATTLE_ROYALE_RQ4_PRIOR_BASELINE_RESULT_2026-07-05.md` | Sagong-style/basic tabular results are available and graph is removed from the core |
| `P0` | Integrate RQ4 recurrent survival comparison | required-cohort model-form result package | `BATTLE_ROYALE_RQ4_HIERARCHICAL_SURVIVAL_PROTOCOL_2026-07-05.md`, `BATTLE_ROYALE_RQ4_SAME_CONDITION_SEQUENCE_RESULT_2026-07-05.md` | last-only, temporal-conv, causal-GRU, and same-snapshot HGB baselines are compared and framed as a bounded model-form result |
| `P1` | Build the paper skeleton | section outline with problem -> method -> experiment -> result -> boundary for every section | canonical flow, RQ chain | each main section has a clear job and no experiment-first ordering |
| `P1` | Expand to master's thesis structure | thesis chapter map with related work, data, preprocessing, method, experiment design, result interpretation, limitations, and future work | user's master's thesis framing, thesis guide | the TEX root no longer reads like a short paper skeleton |
| `P1` | Build the figure/table storyboard | list of required figures, tables, captions, and source files | telemetry table, OM results, Caldera bridge, survival formalization | every RQ has at least one manuscript-facing visual/table |
| `P1` | Draft Section 1-2 prose | Introduction and genre formalization draft | problem definition, ontology lock, mini corpus, survival formalization | RQ1 is readable without starting from PUBG |
| `P1` | Draft Section 3 prose | PUBG official telemetry mapping draft | official telemetry observability table, operational manifest, feature audit | RQ2 shows exactly what PUBG provides and how it maps to formal objects |
| `P1` | Draft Section 4-5 prose | observability boundary and Warzone / Caldera bridge draft | O6Z result, PUBG-Warzone bridge, Caldera construct artifact | RQ3 is a gap-to-representation argument, not a standalone image |
| `P1` | Draft Section 6 prose | prediction experiments and results draft | RQ4 baseline result, hierarchical survival result, OM ladder, phase stratification, paired bootstrap, rank/top-k bridge | RQ4 presents phase survival and final outcome prediction in the right order |
| `P2` | Build claim-boundary text | limitations / not-claimed paragraphs | evidence ledger, gen-1/gen-2 boundary, Caldera boundary | every major claim has a matching forbidden-claim guard |
| `P2` | Supplement triage | appendix/supplement map | O1 integrity, O7 advantage-edge, gen-1/gen-2 note | auxiliary experiments stop competing with the core spine |
| `P3` | Optional robustness decision | small list of optional, not-blocking checks | seed-repeat note, citation verification, LOS stretch note | optional work is clearly separated from manuscript blockers |

## Required Manuscript Figures And Tables

| Slot | Type | Role | Source |
| --- | --- | --- | --- |
| Figure 1 | conceptual diagram | Battle Royale as phase-conditioned multi-agent survival | ontology lock, survival formalization |
| Table 1 | problem-method-experiment map | show how each RQ moves from problem to method to validation | problem-method-experiment structure |
| Table 2 | PUBG telemetry observability table | show what official telemetry directly provides, derives, labels, proxies, or lacks | `PUBG_OFFICIAL_TELEMETRY_OBSERVABILITY_TABLE_2026-07-05.md` |
| Figure 2 | PUBG formal-object mapping | map `U, C, Z, B, H, E, R, Q, Y` to team x phase-cut features | operational manifest, feature audit |
| Figure 3 | PUBG-Warzone observability bridge | show PUBG gap -> Warzone / Caldera representation -> evidence -> claim status | PUBG-Warzone bridge board |
| Figure 4 | RQ4 baseline comparison | show Sagong-style/basic tabular baseline and where OM3 fits | RQ4 prior baseline result |
| Figure 5 | phase survival prediction result | show `OM2 -> OM3` hazard result across cohorts and phase groups, plus same-condition sequence comparison as a bounded model-form result | OM ladder, phase-stratified analysis, paired bootstrap, hierarchical survival result |
| Table 3 | outcome bridge | summarize hazard-to-rank/top-k support and boundaries | rank/top-k bridge |
| Table 4 | claim boundary | supported / bounded / pending / not claimed | evidence ledger |

## Active Thesis Chapter Map

The TEX root now uses the following master's thesis structure:

| Chapter | Role |
| --- | --- |
| 1. 서론 | research background, motivation, questions, and thesis structure |
| 2. 관련 연구 | game telemetry, spatiotemporal data, survival analysis, tabular PUBG prediction, and research gap |
| 3. 연구 문제 정식화 | Battle Royale as phase-conditioned multi-agent survival |
| 4. 데이터와 전처리 | PUBG telemetry layer, observability classes, cohort design, leakage guard |
| 5. 표현 설계와 방법론 | team-phase representation, OM ladder, survival-habitat design, geometry bridge |
| 6. 실험 설계 | why each experiment is needed and what it validates or bounds |
| 7. 실험 결과와 분석 | OM ladder, OM2->OM3, bootstrap, rank/top-k bridge, O6Z, Caldera interpretation |
| 8. 논의와 한계 | master's-thesis-style interpretation of positive and weak/null results |
| 9. 결론 | integrated contribution and future direction |

## Core-Experiment Rule

Do not start unrelated exploratory experiments before the first manuscript
assembly pass.  The RQ4 prior baseline comparison and hierarchical survival
comparison are now complete and should be used to position the prediction
claim against prior work, basic tabular alternatives, same-snapshot-surface
tabular reductions, and the phase-wise survival model form.

Allowed before the first draft:

- cite/source verification;
- figure/table extraction from existing results;
- caption drafting;
- consistency checks between artifacts.
- RQ4 result table extraction from completed Sagong-style/basic tabular
  baselines;
- RQ4 result table extraction from the completed same-snapshot-surface tabular
  vs hierarchical phase-survival required-cohort comparison;

Not blocking the first draft:

- more O7 runs or reinterpretation;
- LOS ray-casting extension;
- Caldera supervised prediction;
- training-seed repeats;
- new feature-family invention.

These can be listed as supplement, robustness, or future work after the core
manuscript spine exists.

## First Draft Order

1. Maintain the active TEX skeleton in `/Users/lunecid/학위논문`.
2. Write the figure/table storyboard.
3. Draft Chapter 1-4 first because they define the research problem, related
   work position, formalization, and data/preprocessing responsibility.
4. Draft Chapter 5-6 next because they justify representation and experiment
   design.
5. Draft Chapter 7 after source numbers and tables are copied from the evidence
   ledger.
6. Draft Chapter 8-9 after positive, weak/null, and future-work claims are
   aligned.

Original RQ-driven order remains conceptually useful:

1. Draft RQ1 and RQ2 first because they define the formalization and data
   representation.
2. Draft RQ3 next, using the PUBG-Warzone bridge as the logic artifact and the
   Caldera composite as the visual example.
3. Draft RQ4 last, using the prior baseline comparison first, then the
   same-snapshot-surface tabular/DL comparison, OM ladder, and bootstrap
   results as the main supervised evidence.
4. Add limitations and supplement placement after the core draft is coherent.

## One-Sentence Next Step

```text
Turn the completed RQ1-RQ4 evidence chain into the manuscript skeleton,
figure/table storyboard, and claim-safe Chapter 6-7 result prose.
```
