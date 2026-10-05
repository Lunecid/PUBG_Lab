# Paper Writing Docs

Package class: `ManuscriptSurface`

## Responsibility

Own manuscript-facing prose, section drafts, figure narratives, contribution
statements, and submission text.

## Public Interface

| Object | Role |
| --- | --- |
| `BATTLE_ROYALE_MANUSCRIPT_ASSEMBLY_PLAN_2026-07-05.md` | next-action plan for turning the locked RQ1-RQ4 chain into manuscript sections, figures, tables, captions, and claim-boundary prose |
| `BATTLE_ROYALE_RQ1_MANUSCRIPT_CLAIM_2026-07-06.md` | manuscript-facing RQ1 build-up from general Battle Royale definition to PUBG empirical instantiation and RQ2 observability bridge |

## Active TEX Root

The actual thesis TEX files live outside this repository:

```text
/Users/lunecid/학위논문
```

Follow the writing guide PDF in that folder when editing or generating TEX.

## Does Not Own

- feature contracts: use `../feature_engineering/`;
- model design: use `../model_design/`;
- experiment runbooks: use `../experiment_design/`.

See `../DESIGN_DOC_STRUCTURE_2026-07-02.md` and
`../DOC_OBJECT_MODEL_2026-07-02.md` for routing rules.
