# Archive Docs

Package class: `DeprecatedObjectStore`

## Responsibility

Preserve generated presentation artifacts, legacy inventories, old status
reports, and historical documents that are no longer active design contracts.

## Public Interface

| Object group | Lifecycle | Notes |
| --- | --- | --- |
| `generated_presentation_2026-06/` | generated | deck, figures, scripts, and presentation outputs |
| `legacy_inventory_2026-06/` | legacy | old code/result inventories and audits |
| `legacy_status_reports_2026-06/` | legacy | old chat/status/presentation data |

## Rules

- Do not add new active design docs here.
- Do not update archived objects as canonical sources.
- If an archived object becomes active again, move it to the owning package and
  update that package README.
- Large generated files here are deletion candidates after reference checks.

See `../DOC_OBJECT_MODEL_2026-07-02.md` for lifecycle definitions.
