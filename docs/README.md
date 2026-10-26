# PUBG Research Docs

This directory is organized as a documentation object model.

```text
folder = package / module
README = public interface
markdown file = object with one responsibility
archive = deprecated object store
```

## Public Entry Points

| Concern | Read first |
| --- | --- |
| Documentation object model | `DOC_OBJECT_MODEL_2026-07-02.md` |
| Folder routing rules | `DESIGN_DOC_STRUCTURE_2026-07-02.md` |
| Battle Royale problem definition | `research_framing/BATTLE_ROYALE_PROBLEM_DEFINITION_2026-07-03.md` |
| Battle Royale paper argument chain | `research_framing/BATTLE_ROYALE_PAPER_ARGUMENT_CHAIN_2026-07-03.md` |
| PUBG cohort registry | `data_layer/BATTLE_ROYALE_COHORT_REGISTRY_2026-07-03.md` |
| Required experiment cohort set | `experiment_design/BATTLE_ROYALE_REQUIRED_COHORT_SET_2026-07-03.md` |
| Battle Royale ontology validation protocol | `experiment_design/BATTLE_ROYALE_ONTOLOGY_VALIDATION_PROTOCOL_2026-07-03.md` |
| Battle Royale feature availability audit | `experiment_design/BATTLE_ROYALE_FEATURE_AVAILABILITY_AUDIT_2026-07-03.md` |
| Battle Royale ontology lock | `research_framing/BATTLE_ROYALE_ONTOLOGY_LOCK_2026-07-03.md` |
| Battle Royale survival formula | `model_design/BATTLE_ROYALE_SURVIVAL_FORMALIZATION_2026-07-03.md` |
| Current research execution checklist | `experiment_design/BATTLE_ROYALE_RESEARCH_EXECUTION_CHECKLIST_2026-07-03.md` |
| Battle Royale ontology corpus design | `research_framing/BATTLE_ROYALE_ONTOLOGY_CORPUS_DESIGN_2026-07-02.md` |
| Current terrain-affordance claim boundary | `research_framing/TERRAIN_AFFORDANCE_CAPABILITY_BOUNDARY_2026-07-02.md` |
| Current terrain-experience feature contract | `feature_engineering/TERRAIN_EXPERIENCE_FEATURE_DESIGN_2026-07-02.md` |
| Caldera/OpenUSD world modeling | `spatial_world_modeling/README.md` |
| Graph/model architecture | `model_design/README.md` |
| Experiment plans and results | `experiment_design/README.md` |

Generated presentation artifacts and legacy inventory files are under
`archive/`. They are preserved for provenance but are not active design docs.

## Dependency Direction

```text
data_layer
-> spatial_world_modeling
-> feature_engineering
-> model_design
-> experiment_design
-> research_framing / paper_writing
```

Use folder READMEs as public interfaces before opening individual docs.

## Public papers, talks and repository organization

- [Dated paper/talk catalog](PAPERS_AND_TALKS.md) retained from the previous main branch.
- [Repository management](repository/README.md): versions, cleanup evidence and outstanding modularization.
- [Previous main README snapshots](archive/main_before_cleanup_merge_2026-10-26/README.md).
