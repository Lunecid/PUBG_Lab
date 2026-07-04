# Documentation Object Model

Date: 2026-07-02

## Purpose

This repository now treats Markdown files as documentation objects rather than
loose notes. Each document should have one responsibility, a clear package, a
small public interface, and an explicit lifecycle.

In short:

```text
folder = package / module
README = public interface
markdown file = object with single responsibility
archive = deprecated object store
cross-reference = dependency
```

## OOP Principles For Markdown

### 1. Single Responsibility

Each Markdown file should answer one durable question.

Good:

```text
What terrain-affordance features do we compute?
What can the current Caldera mesh pipeline claim?
What graph edge semantics does the model use?
```

Bad:

```text
One file mixing feature math, model architecture, paper prose, chat history,
and run logs.
```

### 2. Encapsulation

Each folder owns a design surface. Other folders should refer to its public
README or a named design document rather than duplicating its details.

Example:

```text
model_design -> may reference feature_engineering/TERRAIN_EXPERIENCE_FEATURE_DESIGN
model_design -> should not redefine terrain feature formulas unless changing the model interface
```

### 3. Public Interface

Every active package folder must expose:

- a `README.md`;
- the folder's responsibility;
- its canonical documents;
- what should not be placed there.

The root `docs/README.md` is the public entry point for the entire documentation
package.

### 4. Dependency Direction

Documentation dependencies should flow in this direction:

```text
data_layer
-> spatial_world_modeling
-> feature_engineering
-> model_design
-> experiment_design
-> research_framing / paper_writing
```

Allowed exceptions:

- `research_framing` may point to every package because it defines claims and
  contribution boundaries.
- `handoffs` may point anywhere because it is operational context.
- `archive` may contain stale references because it preserves history.

### 5. Interface Segregation

Do not make one "master design" file serve every reader. Separate:

- feature contract;
- model contract;
- data-layer contract;
- experiment contract;
- paper prose.

### 6. Open / Closed

Active docs should be open for extension but closed for accidental mutation.
When a new idea changes the design surface, create a new dated document or
update the canonical document intentionally. Do not append unrelated chat notes
to a canonical design doc.

### 7. Lifecycle

Every document is in one of these states:

| State | Meaning |
| --- | --- |
| `active` | current design contract or current reference |
| `supporting` | useful background, not the canonical contract |
| `handoff` | operational continuation state |
| `generated` | reproducible output or presentation artifact |
| `legacy` | historical context, not active design |
| `deprecated` | kept only until references are removed |

## Package Classes

| Package | Class role | Owns |
| --- | --- | --- |
| `data_layer` | `DataContract` | DB/layer ownership, cohort, availability |
| `spatial_world_modeling` | `WorldModel` | mesh extraction, z-up alignment, rendering, ray casting |
| `feature_engineering` | `FeatureContract` | feature math, clocks, extraction definitions |
| `model_design` | `ModelContract` | architecture, graph semantics, loss, model input channels |
| `experiment_design` | `ExperimentContract` | splits, runbooks, ablations, metrics |
| `research_framing` | `ClaimBoundary` | contribution, capability boundary, related-work positioning |
| `paper_writing` | `ManuscriptSurface` | paper prose and figure narrative |
| `handoffs` | `ContinuationState` | operational resumption notes |
| `archive` | `DeprecatedObjectStore` | generated or legacy docs |

## Naming Pattern

Use:

```text
TOPIC_OBJECT_YYYY-MM-DD.md
```

Examples:

```text
TERRAIN_EXPERIENCE_FEATURE_DESIGN_2026-07-02.md
TERRAIN_AFFORDANCE_CAPABILITY_BOUNDARY_2026-07-02.md
THREAT_EDGE_MODEL_DESIGN_2026-07-02.md
CALDERA_ZUP_WORLD_MODEL_2026-07-02.md
```

## Object Header Pattern

New canonical design documents should start with:

```markdown
# Title

Date: YYYY-MM-DD
Package: feature_engineering | model_design | ...
Lifecycle: active | supporting | legacy | generated
Responsibility: one sentence
Depends on: path/to/dependency.md
Used by: path/to/consumer.md
```

This header is recommended for new documents. It does not need to be backfilled
into every legacy file immediately.

## Refactoring Rule

When a Markdown file becomes too large or starts owning multiple concerns:

1. identify the separate responsibilities;
2. split into package-specific docs;
3. keep a short reference from the old doc if needed;
4. archive the old mixed file when all references are updated.

## Current Canonical Entry Points

| Concern | Canonical entry |
| --- | --- |
| docs structure | `docs/DESIGN_DOC_STRUCTURE_2026-07-02.md` |
| docs object model | `docs/DOC_OBJECT_MODEL_2026-07-02.md` |
| terrain-affordance feature contract | `docs/feature_engineering/TERRAIN_EXPERIENCE_FEATURE_DESIGN_2026-07-02.md` |
| terrain-affordance lineage | `docs/feature_engineering/TERRAIN_EXPERIENCE_RESEARCH_LINEAGE_2026-07-02.md` |
| capability boundary | `docs/research_framing/TERRAIN_AFFORDANCE_CAPABILITY_BOUNDARY_2026-07-02.md` |
| cleanup audit | `docs/research_framing/DOCS_CLEANUP_AUDIT_2026-07-02.md` |
