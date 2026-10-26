"""Bring the anonymized release in line with the v17 manuscript (evaluation-unit framing, combined objective).

Copies changed analysis code from the main repository, adds the new scripts and the supplementary notes, removes the
superseded v16 figure script, neutralizes version-labelled default paths, and updates README/NOTES references.
Every edit asserts its anchor so a stale release fails loudly.
"""
import re
import shutil
from pathlib import Path

SRC = Path('/Users/lunecid/project/PUBG')
DST = Path('/Users/lunecid/project/nextphase-elimination-code')
FORBIDDEN = ('lunecid', 'pusan', 'bigcomp', '/Users/', 'v14_review', 'v16_revision', 'v17_revision', 'todtjddms', 'jdc05124')


def edit(rel, pairs):
    p = DST / rel
    s = p.read_text(encoding='utf-8')
    for a, b in pairs:
        assert s.count(a) == 1, (rel, a[:80], s.count(a))
        s = s.replace(a, b)
    p.write_text(s, encoding='utf-8')


# 6) README
edit('README.md', [
    ('Code for the anonymous ACM SAC 2027 submission *Predicting Next-Phase Elimination in Battle Royale: Target Alignment,\n'
     'Habitat Context, and Model Structure*.',
     'Code for the anonymous ACM SAC 2027 submission *Ranking Teams or Matches? Evaluation Units for Next-Phase Elimination\n'
     'Prediction in Battle Royale*.'),
    ('adaptations of the two prior designs. It holds no telemetry, derived samples, model outputs, or results; PUBG telemetry is\n'
     'public through the PUBG developer API, and raw and processed data are not redistributed.',
     'adaptations of the two prior designs. It holds no telemetry, derived samples, or model outputs; PUBG telemetry is\n'
     'public through the PUBG developer API, and raw and processed data are not redistributed. `supplement/supplementary_notes.pdf`\n'
     'holds the supplementary analyses that the paper cites (habitat hazard model, path decomposition, tuning, further controls).'),
    ('| `model_stgcn/sagong2025/` | Adaptation of Sagong et al. (2025): input reconstruction (`plan`, `extract`) and fitting (`fit`). | Sec. 4.3 |',
     '| `model_stgcn/sagong2025/` | Adaptation of Sagong et al. (2025): input reconstruction (`plan`, `extract`) and fitting (`fit`). | Sec. 4.2 |'),
    ("with our adaptation in `adaptation/` (input reconstruction, training, phase baseline, native metrics). | Sec. 4.3 |",
     "with our adaptation in `adaptation/` (input reconstruction, training, phase baseline, native metrics). | Sec. 4.2 |"),
    ('| Equal-trial tuning (16 configurations) and the exploratory GATv2 with its relation-relabeling control. | Table 8, Sec. 6.3 |',
     '| Equal-trial tuning (16 configurations) and the exploratory GATv2 with its relation-relabeling control. | Sec. 6.3, supplement |'),
    ('| Matched objective and horizon controls (weighted survival, Cox, cross-entropy) and the objective x mapping factorial. | Table 6, Sec. 6.3 |',
     '| Matched objective and horizon controls (weighted survival, Cox, cross-entropy, next-phase Cox plus cross-entropy) and the objective x mapping factorial. | Table 4, Sec. 6.1, supplement |'),
    ('| Population (AI input, human-only loss) and history controls. | Sec. 6.3 |',
     '| Population (AI input, human-only loss) and history controls. | Sec. 6.3, supplement |'),
    ('| Full-feature L2 logistic baseline. | Table 4 |', '| Full-feature L2 logistic baseline. | Table 5 |'),
    ('| Matched test contrasts, human-team evaluation, winner accuracy and C-index, Table-4 metrics of the logistic baseline. | Table 4, Sec. 6.2 |',
     '| Matched test contrasts, human-team evaluation, winner accuracy and C-index, main-table metrics of the logistic baseline. | Tables 4-5, Sec. 6.1-6.2 |'),
    ('| Within-phase AUC before and after match-phase centering (post hoc diagnostic). | Fig. 3c, Sec. 6.2-6.3, 7.1 |',
     '| Within-phase AUC before and after match-phase centering (post hoc diagnostic). | Fig. 2, Sec. 6.1-6.3, 7.1 |\n'
     '| `tools/sac2027_pair_share.py` | Share of same-match pairs among within-phase AUC pairs, per cohort. | Sec. 5.1, supplement |'),
    ('| RQ1 hazard models, sensitivity analyses, phase-band equality tests. | Sec. 6.1, Fig. 2 |',
     '| Habitat hazard models, sensitivity analyses, phase-band equality tests. | Sec. 4.1, supplement |'),
    ('| `tools/sac2027_paper_tables.py`, `tools/sac2027_figure3.py`, `tools/sac2027_plot_rq1.py`, `tools/sac2027_result_figures.py`, `tools/make_result_figures_v2.py` | Tables and figures built from the analysis outputs. | Tables 4-8, Figs. 2-3 |',
     '| `tools/sac2027_paper_tables.py`, `tools/sac2027_figure_eval_units.py`, `tools/sac2027_plot_rq1.py`, `tools/sac2027_result_figures.py`, `tools/make_result_figures_v2.py` | Tables and figures built from the analysis outputs. | Tables 4-6, Fig. 2, supplement |'),
    ('   `python tools/sac2027_sesoi_top3.py`.',
     '   `python tools/sac2027_sesoi_top3.py`, `python tools/sac2027_pair_share.py`.'),
    ('5. Tables, figures, and checks: `python tools/sac2027_paper_tables.py`, `python tools/sac2027_figure3.py <output-dir>`,',
     '5. Tables, figures, and checks: `python tools/sac2027_paper_tables.py`, `python tools/sac2027_figure_eval_units.py <output-dir>`,'),
    ('- Logistic baseline: phase indicators plus the GBDT\'s 151 inputs standardized on training data, L2 penalty',
     '- Objective controls: the KRAFTON-derived backbone at a configuration fixed before tuning; six losses (weighted survival\n'
     '  and standard Cox, each with full follow-up or truncated at the next phase; next-phase BCE; next-phase Cox plus BCE with\n'
     '  weight 1.0, fixed before training); seeds 42/13/2026; stopping on validation within-phase AUC; a common validation\n'
     '  logistic calibration with phase indicators.\n'
     '- Logistic baseline: phase indicators plus the GBDT\'s 151 inputs standardized on training data, L2 penalty'),
])

# 7) NOTES
edit('NOTES.md', [
    ('- `tools/sac2027_match_offset.py`: objective controls (five losses x three seeds x two cohorts). It splits within-phase AUC',
     '- `tools/sac2027_match_offset.py`: objective controls (six losses x three seeds x two cohorts). It splits within-phase AUC'),
    ('- Bootstrap draws reuse the objective suite\'s seeds, so the intervals share resampled matches with the objective table.',
     '- Bootstrap draws reuse the objective suite\'s seeds, so the intervals share resampled matches with the objective table.\n'
     '- Combined objective (`cox_truncated_bce` in `tools/sac2027_objective_controls.py`): next-phase Cox plus next-phase BCE with\n'
     '  weight 1.0. The weight and three predictions were recorded before training: higher within-phase AUC than next-phase Cox\n'
     '  (95% CI above zero), within-phase AUC practically equivalent to BCE (90% CI within +-0.005), and within-match AUC\n'
     '  practically equivalent to next-phase Cox (90% CI within +-0.005), in both cohorts. Contrasts with 90% intervals are in\n'
     '  `objective_match_offset/contrasts.csv` and `objective/contrasts.csv` (columns `ci90_low`, `ci90_high`).\n'
     '- `tools/sac2027_pair_share.py`: share of same-match pairs among within-phase AUC pairs per cohort (about 1/M for M test\n'
     '  matches). Output: `reports/sac2027_patch_v14/control_evaluation/pair_share/pair_share.csv`.'),
    ('- `tools/sac2027_logistic_summary.py`: Table-4 metrics of the logistic baseline, computed with the code behind the other',
     '- `tools/sac2027_logistic_summary.py`: main-table metrics of the logistic baseline, computed with the code behind the other'),
])

# 8) scan everything that changed for identity or version cues
changed = ['tools/sac2027_objective_controls.py', 'tools/sac2027_match_offset.py', 'tools/sac2027_evaluate_controls.py',
           'tools/sac2027_paper_tables.py', 'tests/test_sac2027_objective_controls.py', 'tools/sac2027_figure_eval_units.py',
           'tools/sac2027_pair_share.py', 'tools/sac2027_result_figures.py', 'tools/sac2027_plot_rq1.py',
           'analysis/rq1_robustness.py', '.gitignore', 'README.md', 'NOTES.md',
           'third_party/krafton_pubg_winprob/ADAPTATION.patch']
for rel in changed:
    s = (DST / rel).read_text(encoding='utf-8').lower()
    hits = [w for w in FORBIDDEN if w.lower() in s]
    assert not hits, (rel, hits)
    assert not re.search(r'[가-힣]', s), rel
print('README, NOTES updated; scan clean')
