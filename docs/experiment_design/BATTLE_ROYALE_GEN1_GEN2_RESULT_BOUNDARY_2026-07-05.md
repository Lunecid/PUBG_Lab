# Battle Royale Gen-1 / Gen-2 Result Boundary

Date: 2026-07-05
Class: `ExperimentBoundary`
Lifecycle: `active`

## Purpose

This note closes the manuscript blocker recorded in the execution checklist:

```text
Resolve the gen-1 / gen-2 result boundary before launching long O7 runs.
```

The goal is not to reinterpret every older run.  The goal is to decide which
legacy or preliminary results may be cited as manuscript evidence, which may
only be used as design motivation, and which must be rerun under the current
required-cohort contract.

## Boundary Decision

Use this rule:

```text
Only gen-2 required-cohort artifacts are manuscript evidence for the current
Battle Royale framework paper.
```

Gen-1 results may be used for:

- historical motivation;
- method development;
- pilot evidence;
- explaining why the O7 edge-stream protocol was designed.

Gen-1 results must not be used as headline evidence for:

- the central pressure-habitat mechanism;
- required-cohort cross-map robustness;
- O7 observed relation-topology support;
- exact rank lock-in timing;
- corrected O6 spatial-affordance support.

## Definitions

| Generation | Meaning | Claim role |
| --- | --- | --- |
| `gen-1` | June 2026 legacy / presentation / pooled V3 and preliminary graph results | motivation or pilot only |
| `gen-2` | July 2026 required-cohort, cohort-separated, claim-boundary-controlled Battle Royale framework artifacts | current manuscript evidence |

## Gen-2 Evidence Source Of Truth

The current manuscript evidence is:

| Result family | Artifact root / document | Status |
| --- | --- | --- |
| Required-cohort OM ladder | `model_stgcn/checkpoints/br_om_ladder_required_v1/` | manuscript evidence |
| OM result feature audit | `docs/experiment_design/BATTLE_ROYALE_OM_LADDER_RESULT_FEATURE_AUDIT_2026-07-04.md` | manuscript evidence |
| Phase-stratified OM analysis | `docs/experiment_design/BATTLE_ROYALE_PHASE_STRATIFIED_OM_RESULT_2026-07-04.md` | manuscript evidence |
| Phase-group paired bootstrap | `docs/experiment_design/BATTLE_ROYALE_PHASE_GROUP_BOOTSTRAP_RESULT_2026-07-04.md` | manuscript evidence |
| O1 integrity audit | `docs/experiment_design/BATTLE_ROYALE_O1_SURVIVAL_CAPITAL_INTEGRITY_AUDIT_2026-07-04.md` | boundary evidence |
| Corrected O6Z required-cohort sensitivity | `model_stgcn/checkpoints/br_om6z_required_v1/` | bounded negative manuscript evidence |
| Experiment evidence ledger | `docs/experiment_design/BATTLE_ROYALE_EXPERIMENT_EVIDENCE_LEDGER_2026-07-05.md` | source-of-truth claim ledger |
| O7 edge-stream protocol | `docs/experiment_design/BATTLE_ROYALE_O7_EDGE_STREAM_NULL_GUARD_PROTOCOL_2026-07-05.md` | pre-run contract, not result evidence |

## Gen-1 Inventory And Decision

| Gen-1 family | Main artifact / source | Decision | Reason |
| --- | --- | --- | --- |
| `v3_full_main_causal` pooled MR/V3 architecture matrix | `docs/archive/legacy_inventory_2026-06/LEGACY_RESULT_CODE_REGISTRY_2026-06-22.md` | background only | not cohort-separated under the current required map / perspective / team-size contract |
| `v3_full_controls` resolution / decoder matrix | same legacy registry | background only | useful for historical ST/MR motivation, but not the current ontology-guided OM evidence |
| `v3_full_feature_causal` feature preset matrix | same legacy registry | background only | superseded by clean OM0-OM6 required-cohort ladder |
| `feature_ablation_full_db_manifest_v2` tabular HGB ablation | same legacy registry | optional background baseline | phase-row tabular baseline, not current neural/cohort OM ladder |
| `v3_full_graph_relation` core-habitat graph/null-control matrix | same legacy registry | O7 design motivation only | uses legacy pooled `graph_relation` run, not current O7 advantage-edge A/B/C required-cohort contract |
| June presentation rank lock-in analysis | `docs/archive/legacy_status_reports_2026-06/PRESENTATION_DATA_2026-06-21.md` | motivation only unless rerun | exact `phase 3->6 +0.085` and `16-24 min` timing are not yet gen-2 required-cohort claims |
| Preliminary advantage-edge result | `docs/research_framing/FUTURE_PLAN.md` and generated-presentation archive | pilot only | current `adv_precompute.py` default is squad-wide, not exact required cohorts; result must be rerun with A/B/C null guard |
| Snapshot pilot / narrow pilot results | `docs/experiment_design/PRE_EXPERIMENT_DESIGN_LOCK_2026-06-22.md` and handoff archive | engineering pilot only | useful for feasibility, not manuscript evidence |

## Claim-Specific Decisions

### Core-Habitat Null-Control CLEAN

Allowed wording:

```text
An earlier pooled V3 graph-relation matrix motivated the current O7 protocol:
the unshuffled proximity/combat/co-pressure graph did not beat an edge-shuffled
null in that legacy run.
```

Avoid:

```text
The current Battle Royale paper has proven that observed O7 topology carries no
signal.
```

Reason:

The old `graph_core_habitat_*` rows are valuable because they separated graph
smoothing from observed topology in a legacy phase-tier graph experiment.  They
do not satisfy the current required-cohort, exact map / perspective / team-size
contract and they are not the locked O7 headline stream.  The current O7 stream
is contact-gated relative-advantage edges.

### Rank Lock-In

Allowed wording:

```text
Legacy analyses suggested a mid-to-late decision window, motivating phase-cut
rank and top-k evaluation.
```

Avoid:

```text
The current gen-2 framework proves final rank locks in from phase 3 to phase 6
with delta Spearman +0.085.
```

Reason:

The current gen-2 evidence supports a hazard-to-rank bridge after OM3,
especially in early and mid phase groups.  The exact old lock-in number and
game-time window should be rerun or re-derived from gen-2 artifacts before
appearing as a current manuscript result.

### Preliminary Advantage-Edge Evidence

Allowed wording:

```text
Preliminary advantage-edge pilots motivated the locked O7 A/B/C protocol.
```

Avoid:

```text
Advantage-edge topology is already validated as a current O7 result.
```

Reason:

The pilot values such as `+0.0033` against edge-shuffle and `+0.0023` under
node-absolute control were not produced under the current 13-cohort contract.
They also predate the locked O7 interpretation rule:

```text
primary evidence = B - C, observed advantage edges minus edge-shuffled null
```

## What Must Be Rerun For O7 Claims

Before any O7 manuscript claim, run the locked protocol as gen-2:

| Arm | Code-level condition | Interpretation |
| --- | --- | --- |
| `A` | `gate=none`, `control=observed` | node/global baseline |
| `B` | `gate=knn`, `control=observed` | observed advantage-edge graph |
| `C` | `gate=knn`, `control=edge_shuffle` | edge-shuffled null |

Required changes before a long run:

1. Add cohort-aware cache generation to the advantage-edge precompute path, or
   wrap it so each required cohort uses its exact `map_name`, `game_mode`, and
   existing OM ladder `split.json`.
2. Add a short edge-shuffle audit proving shuffling occurs only within each
   match-phase alive-node block and preserves edge count / edge-feature
   distribution at the intended level.
3. Run high-power early readout cohorts first:

```text
Erangel TPP squad
Taego TPP squad
Erangel TPP solo
Miramar / Desert TPP squad
```

4. Expand to all 13 required cohorts before any headline O7 claim.

Suggested gen-2 root:

```text
model_stgcn/checkpoints/br_o7_advantage_edge_required_v1/
```

Each cohort should write:

```text
<root>/<cohort_slug>/adv_cache/{train,val,test}.pt
<root>/<cohort_slug>/A_none_observed/
<root>/<cohort_slug>/B_knn_observed/
<root>/<cohort_slug>/C_knn_edge_shuffle/
<root>/<cohort_slug>/o7_edge_shuffle_audit.json
```

## Immediate Next Action

The short O7 edge-shuffle audit and cohort-aware cache/run smoke are complete:

```text
docs/experiment_design/BATTLE_ROYALE_O7_EDGE_SHUFFLE_AUDIT_2026-07-05.md
```

Next launch the high-power O7 early readout cohorts.  Do not cite any gen-1
graph, rank-lock, or advantage-edge value as current manuscript evidence unless
it is rerun or re-derived under gen-2.
