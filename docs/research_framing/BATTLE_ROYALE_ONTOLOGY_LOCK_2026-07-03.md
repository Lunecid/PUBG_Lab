# Battle Royale Ontology Lock

Date: 2026-07-03

## Decision

The Battle Royale ontology is locked as v1 for the current paper and for the
OM0-OM6 PUBG prediction ladder.

Locked families:

| Family | Locked name | Role |
| --- | --- | --- |
| `O0` | Rule And Match Context | comparison frame and match rules |
| `O1` | Agent / Team Survival Capital | capacity to remain alive or qualified |
| `O2` | Contraction / Elimination Pressure | pressure schedule that compresses viable survival conditions |
| `O3` | Movement / Rotation Behavior | behavioral response to pressure |
| `O4` | Interaction / Threat Exposure | exposure to opponent, contest, or combat pressure |
| `O5` | Resource / Equipment / Mobility Support | resources that enable survival, contact, or movement |
| `O6` | Spatial / Terrain / Environment Affordance | spatial opportunities and constraints under the available geometry mask |
| `O7` | Relation / Interaction Structure | opponent, ally, proximity, contest, and pressure relations among agents |
| `O8` | Outcome / Win Condition | survival, rank, qualification, or winner condition |

## What Is Frozen

The following are frozen for the next experiment cycle:

- top-level family count: `O0` through `O8`;
- top-level family names;
- `O2` as contraction / elimination pressure, with shrinking zones as one
  projection;
- `O4` as interaction / threat exposure, with gun combat as one projection;
- the conceptual portability axis:
  `BR_GENERAL`, `GAME_SPECIFIC`, `DATASET_SPECIFIC`, `GEOMETRY_SPECIFIC`;
- the measurement observability axis:
  telemetry-observed, geometry-observed, label-observed, proxy, or masked;
- the rule that O6 must declare spatial profile and observability mask.

## What Can Still Change

The following can still change without reopening the ontology:

- game-specific projections under an existing family;
- source URLs, source grades, lifecycle status, and archive links;
- PUBG implementation details and feature formulas;
- Caldera/Warzone geometry feature formulas;
- ablation presets and model architecture;
- corpus-codebook rows and chunk labels;
- wording that clarifies a family without renaming or splitting it.

Adding `O9` or renaming a locked family is not allowed before OM0-OM6 unless a
genuine blocker is found and documented.

## Dynamic Mechanism

The simple chain is the primary explanatory direction for phase-cut prediction,
not a full one-way causal graph:

```text
O2 pressure schedule x O6 spatial affordance
-> H_i(k), the survivable habitat available to agent/team i
O1_i(k), O5_i(k), and H_i(k)
-> O3_i(k), behavioral response
O3_i(k), O7_ij(k), and shared pressure
-> O4_i(k), interaction / threat exposure
O1_i(k), H_i(k), O4_i(k), and O7_i(k)
-> O8_i(k+1), survival / rank / qualification outcome
```

The ontology also permits state feedback:

| Feedback | Meaning |
| --- | --- |
| `O4 -> O1` | interaction or combat can deplete health, roster, lives, or survival capital |
| `O8 -> O2/O7` | eliminations change alive population, crowding, and the relation field |
| `O3_j/O7_ij -> H_i` | other agents' movement, blocking, camping, or third-party pressure changes my local habitat |

This feedback is why late phases can show rank lock-in or selection effects
without contradicting the phase-cut explanatory chain.

## Composite Ownership Rules

### Habitat

`Habitat` is an `O2 x O6` interaction construct:

```text
habitat = contraction / elimination pressure applied to the available spatial
affordance field.
```

Examples:

- safe area per alive team;
- zone gap pressure;
- required speed to safe area;
- route or sector pressure induced by the pressure schedule;
- viable terrain or geometry under a future geometry-rich projection.

Ablation attribution rule:

```text
Habitat features are attributed to OM3, the O2 environmental-pressure block.
OM6 should add standalone spatial/geometry affordance features and should not
duplicate habitat signals already counted in OM3.
```

### Interaction Exposure Versus Relation Structure

`O7` owns the relation field itself: edges, relation channels, roles, global
structure, and graph summaries.

`O4` owns realized hostile or contest events and scalar ego exposure summaries
derived from a declared relation field.

Rule:

```text
Every O4 exposure summary must declare its source relation graph or observation
window, such as proximity radius, combat event window, or co-pressure channel.
Graph edges, centrality, bridge roles, and global graph state remain O7.
```

Ablation attribution rule:

- realized damage/kill/contact events and simple scalar exposure summaries are
  OM4;
- edge streams, centralities, bridge roles, community/global graph metrics, and
  relation-message-passing inputs are OM5;
- the same raw signal should not be duplicated across OM4 and OM5.

### Information And Intel

Information mechanics do not create a new family in v1.

Rule:

```text
The information-acquisition means is O5.
The exposure state changed by that information is O4.
The spatial visibility or occlusion structure is O6.
The known or inferred relation edge can be represented in O7.
```

Examples:

- UAV, radar, scan, ping, contract reveal, or intel tablet item: O5;
- revealed enemy threat or contact awareness: O4;
- line of sight, fog, occlusion, or visibility field: O6;
- known enemy relation or team-to-team reveal edge: O7.

Corpus coding should add an `intel_flag` note for these chunks.

## Revision Procedure

Clarifying wording can be patched in place.  A true ontology revision requires:

1. create a new version label, such as `ontology_v2`;
2. update the ontology, operational manifest, validation plan, and corpus
   codebook schema together;
3. mark whether already-run OM experiments remain valid, need relabeling, or
   need rerunning.

## Lock Basis

The lock is justified by triangulation rather than by an exhaustive genre
survey.

### T1. Operationalization

O0-O8 are already mapped into the PUBG operational manifest and model-facing
feature blocks.  The ontology is therefore not merely a textual taxonomy; it has
an implementation path into DB tables, columns, and model inputs.

### T2. Heterogeneous Projection

The same ontology projects into:

- PUBG as a label-rich, geometry-poor supervised prediction setting;
- Caldera/Warzone as a geometry-rich, label-poor construct-validation setting.

This supports the claim that the ontology owns genre mechanisms, while each
dataset supplies a masked projection.

### T3. Corpus Sanity

The initial corpus pass over shooter, mobile, top-down, party, racing, puzzle,
and mode-based Battle Royale variants produced projection differences rather
than new top-level families.

The corpus did revise wording:

- `O2`: from phase/zone pressure to contraction / elimination pressure;
- `O4`: from contact/combat/threat to interaction / threat exposure.

After these revisions, the other top-level families remain stable.

## Claim Level

Allowed paper claim:

```text
The O0-O8 vocabulary supports an ontology-guided analytic framework for
operationalizing Battle Royale survival prediction and construct validation
across observability regimes.
```

Avoid:

```text
This paper defines the final taxonomy of the Battle Royale genre.
```

The paper should frame O0-O8 as a lightweight domain vocabulary inside an
analytic framework with sufficient coverage for prediction, not as a formal
ontology or exhaustive genre-definition project.

## Post-Lock Framework Evidence Status

As of 2026-07-04, the paper claim has been lowered from "ontology established"
to "ontology-guided analytic framework proposed and empirically supported."  The
supporting status document is:

```text
docs/research_framing/BATTLE_ROYALE_ANALYTIC_FRAMEWORK_STATUS_2026-07-04.md
```

The strongest evidence is the completed required-cohort OM ladder:

```text
docs/experiment_design/BATTLE_ROYALE_OM_LADDER_RESULT_FEATURE_AUDIT_2026-07-04.md
```

The central mechanism `O2 x O6 = H` is operationalized as `OM2 -> OM3` and
shows positive AUC gain in all 13 required PUBG cohorts.  This supports using
O0-O8 as the framework vocabulary.

This does not upgrade the claim to a formal ontology or exhaustive genre
taxonomy.  The corpus track remains confirmatory, and Caldera remains construct
validation for geometry-rich O6 projection.

## Corpus Track

The corpus track is confirmatory.

For the current paper, required corpus work is limited to:

- one mini coverage table over representative games with S0/S1 or
  publisher-controlled sources;
- a short limitations paragraph on source circularity and long-tail
  discontinued games;
- optional negative controls if they can be completed without delaying OM0-OM6.

The current mini coverage artifact is complete:

```text
docs/research_framing/BATTLE_ROYALE_MINI_CORPUS_COVERAGE_TABLE_2026-07-04.md
```

The full C1-C4 protocol is not a blocking prerequisite for PUBG prediction.

Stop rule for extended corpus work:

```text
After the mini corpus, stop when two consecutive new strata add zero new
top-level families and only add projections under existing O0-O8 families.
```

## Next Priority

The next priority is no longer ontology construction or the first OM ladder
execution.  It is:

```text
1. run phase-stratified OM aggregation and paired uncertainty checks;
2. resolve the gen-1 / gen-2 experiment naming and result boundary;
3. implement or explicitly defer the intended O6 Z-ladder;
4. keep Caldera as O6 geometry-rich construct validation unless labels are
   rebuilt.
```

The corpus codebook can continue in parallel, but it should not delay the
prediction track.
