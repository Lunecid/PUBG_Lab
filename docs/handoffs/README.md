# Handoff Docs

Package class: `ContinuationState`

## Responsibility

Own next-chat handoffs, continuation notes, and short operational summaries for
resuming work.

## Public Interface

| Object | Role |
| --- | --- |
| `NEXT_CHAT_HANDOFF_2026-07-05_BATTLE_ROYALE_FRAMEWORK.md` | current Battle Royale framework continuation handoff after OM6Z |
| `NEXT_CHAT_HANDOFF_2026-07-01_CALDERA_ST_3D.md` | Caldera ST/3D continuation handoff |
| `NEXT_CHAT_HANDOFF_2026-06-22.md` | older ST paper continuation handoff |

## Does Not Own

- canonical design contracts;
- durable feature/model definitions;
- final manuscript prose.

Handoffs may reference any package, but should not become the canonical source
of a design decision.

See `../DESIGN_DOC_STRUCTURE_2026-07-02.md` and
`../DOC_OBJECT_MODEL_2026-07-02.md` for routing rules.
