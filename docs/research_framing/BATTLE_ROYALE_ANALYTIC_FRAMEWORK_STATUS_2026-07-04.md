# Battle Royale Analytic Framework Status

Date: 2026-07-04
Class: `ClaimBoundary`
Lifecycle: `active`

## Decision

The current paper should frame the contribution as an ontology-guided Battle
Royale survival analysis framework, not as the establishment of a formal or
universal Battle Royale ontology.

This means:

```text
O0-O8 is a stable lightweight domain vocabulary inside an analytic framework
for Battle Royale team survival prediction and construct validation.
```

It does not mean:

```text
O0-O8 is a formal ontology, an OWL/RDF knowledge model, or the final taxonomy
of every Battle Royale game.
```

The correct claim level is therefore:

```text
We propose an ontology-guided Battle Royale survival analysis framework and
validate its central pressure-habitat mechanism through cohort-separated PUBG
prediction experiments.
```

Avoid:

```text
We establish the universal ontology of all Battle Royale games.
```

## Basis

### 1. Controlled Vocabulary

`O0` through `O8` are locked in
`docs/research_framing/BATTLE_ROYALE_ONTOLOGY_LOCK_2026-07-03.md`.

The locked families provide the controlled vocabulary of the framework:

- rule and match context;
- agent or team survival capital;
- contraction / elimination pressure;
- movement / rotation response;
- interaction / threat exposure;
- resource / equipment / mobility support;
- spatial / terrain / environment affordance;
- relation / interaction structure;
- outcome / win condition.

This defines the conceptual vocabulary used by the rest of the project.  In the
paper, the preferred term is controlled mechanism vocabulary inside an
ontology-guided analytic framework.  If the word ontology is used, it should be
qualified as lightweight domain ontology and explicitly distinguished from a
formal ontology.

### 2. Operationalization

The framework is not only a textual taxonomy. Its vocabulary has been mapped
into executable PUBG feature presets:

```text
OM0 -> OM1 -> OM2 -> OM3 -> OM4 -> OM5 -> OM6
```

The exact implemented feature list is frozen in:

```text
docs/experiment_design/BATTLE_ROYALE_OM_LADDER_RESULT_FEATURE_AUDIT_2026-07-04.md
```

The code-level owner is:

```text
model_stgcn/v3/spec.py
```

### 3. Predictive Validation Of The Central Mechanism

The central framework mechanism is:

```text
O2 contraction / elimination pressure x O6 spatial affordance
-> H, the phase-conditioned survival habitat
```

This mechanism is operationalized in the OM ladder as:

```text
OM2 movement -> OM3 pressure / habitat
```

The required-cohort run completed 13 independently trained map / perspective /
team-size cohorts. In all 13 cohorts, `OM2 -> OM3` increased hazard AUC.

Observed summary:

| Quantity | Value |
| --- | ---: |
| Cohorts | 13 |
| Positive `OM2 -> OM3` AUC deltas | 13/13 |
| Median `OM2 -> OM3` AUC delta | `+0.0103` |
| Min / max `OM2 -> OM3` AUC delta | `+0.0057` / `+0.0211` |
| Sign-test evidence | 13/13 sign consistency, two-sided p approximately `0.00024` |

This is the strongest current evidence that the framework's vocabulary is doing
predictive work rather than merely assigning names to arbitrary feature groups.

### 4. Cross-Cohort Stability

The OM ladder was not trained on a pooled mixed cohort. It was run separately
for each map / perspective / team-size cohort.

Completed required cohorts:

- Erangel TPP squad;
- Taego TPP squad;
- Miramar / Desert TPP squad;
- Rondo TPP squad;
- Sanhok TPP squad;
- Vikendi TPP squad;
- Paramo TPP squad;
- Karakin TPP squad;
- Erangel TPP solo;
- Erangel TPP duo;
- Taego TPP duo;
- Taego FPP squad;
- Erangel FPP squad.

The positive `OM2 -> OM3` result across all cohorts supports the claim that
the pressure / habitat mechanism is not just an artifact of one map, one
perspective, or one team size.

### 5. Heterogeneous Projection Boundary

PUBG and Caldera/Warzone play different roles:

| Dataset | Current role |
| --- | --- |
| PUBG | label-rich, geometry-poor supervised prediction test |
| Caldera / Warzone | geometry-rich, label-poor construct-validation evidence for O6 |

This supports the framework as a mechanism-and-observability scheme. It does
not make Caldera a supervised survival-prediction dataset under the current
labels.

## What Is Supported

The following are now supported for the current paper:

- O0-O8 is stable enough to use as the framework vocabulary without reopening
  family names;
- the framework vocabulary can be operationalized as leakage-guarded PUBG
  feature blocks;
- the central `O2 x O6 = H` habitat mechanism has robust first-run predictive
  support across the required PUBG cohorts;
- phase-stratified and paired match-cluster bootstrap analyses support that
  the pressure-habitat signal is not merely a pooled phase-base-rate artifact;
- Battle Royale survival should be framed as phase-conditioned multi-agent
  survival, not final-rank prediction from a bag of match statistics;
- corrected PUBG `pos_z`-only O6Z features do not robustly improve prediction
  after OM5, bounding PUBG as geometry-poor for fine spatial-affordance claims;
- Caldera/Warzone belongs in the paper as construct-validation evidence for
  richer spatial affordance measurement, not as supervised prediction evidence.

## What Is Not Claimed

The following claims are not yet established:

- O0-O8 is the final taxonomy of every Battle Royale game;
- O0-O8 is a formal ontology in the semantic-web or knowledge-representation
  sense;
- `O1` survival capital is useless, because the integrity audit shows direct
  univariate hazard association even though incremental OM1 gains are small;
- O7 graph structure is fully tested, because the completed OM5 run used
  node-summary `net_*` and `cum_*` features only;
- O6 spatial affordance is fully tested in general, because the corrected
  OM6Z result only tests PUBG telemetry-derived `pos_z` proxies, not true
  terrain mesh, cover, line of sight, or building occlusion;
- small OM4/OM5/OM6 gains are statistically settled, because the first run is
  single-seed and needs bootstrap or seed-repeat support.

## Allowed Manuscript Wording

Use:

```text
We propose an ontology-guided analytic framework for Battle Royale survival.
The framework uses O0-O8 as a lightweight domain vocabulary of reusable
mechanism families. In PUBG, this vocabulary is operationalized as a
cohort-separated OM0-OM6 feature ladder, and the central pressure-habitat
mechanism improves next-phase survival prediction in all 13 required cohorts.
```

Use:

```text
The results support the framework's central mechanism: survival is shaped by
how contraction pressure transforms the playable habitat available to teams.
```

Avoid:

```text
The Battle Royale ontology is established because game descriptions mention
similar concepts.
```

Avoid:

```text
The OM ladder proves that all Battle Royale games have the same feature
importance structure.
```

## Next Work

The next work is not ontology construction. The next work is targeted evidence
completion:

1. resolve the gen-1 / gen-2 result boundary so manuscript claims only cite
   results that are reusable under the current cohort / feature / split
   contract;
2. run the locked O7 edge-stream / graph-relation protocol after the null-guard
   audit;
3. produce the Caldera O6 construct-validation artifact to show what richer
   geometry-derived spatial affordance can measure;
4. convert the framework narrative and completed result audits into manuscript
   section drafts;
5. keep small-cohort seed repeats and extended corpus / negative-control work
   as optional robustness or appendix work if schedule permits.
