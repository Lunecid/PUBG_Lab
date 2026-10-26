#!/usr/bin/env python3
"""RQ1 contrast table of the SAC 2027 manuscript (v20h): matched objective controls in the two largest cohorts.

One row per (contrast, evaluation unit); columns are Erangel and Taego TPP squad; cells are seed-averaged estimates with
95% match-bootstrap intervals, or closure shares in percent.  Every value is read from stored analysis outputs:

* control_evaluation/objective_match_offset/contrasts.csv  (horizon, BCE vs next-phase Cox, combined loss, closure share)
* control_evaluation/condlogit/contrasts_per_cohort.csv    (conditional logistic loss, weight 4, phase-minus-match)

Usage: python tools/sac2027_rq1_table.py <output-dir>
"""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CE = REPO / 'reports/sac2027_patch_v14/control_evaluation'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / 'paper/tables'
COHORTS = ('erangel_tpp_squad', 'taego_tpp_squad')

# (label, source, contrast, metric, kind)
ROWS = [
    ('group', 'Horizon'),
    (r' & Cox$_{\mathrm{next}}-$Cox$_{\mathrm{full}}$, match', 'om', 'cox_truncated_minus_cox_full', 'within_match', 'diff'),
    ('group', 'BCE versus next-phase Cox'),
    (r' & phase', 'om', 'bce_minus_cox_truncated', 'original_within_phase', 'diff'),
    (r' & match', 'om', 'bce_minus_cox_truncated', 'within_match', 'diff'),
    (r' & phase $-$ match$^{a}$', 'cl', 'bce_minus_cox_truncated', 'phase_minus_match', 'diff'),
    (r' & centered, phase', 'om', 'bce_minus_cox_truncated', 'centred_within_phase', 'diff'),
    ('group', 'Conditional logistic (CLogit)$^{b}$'),
    (r' & CLogit$-$Cox, phase', 'cl', 'cond_logit_minus_cox_truncated', 'original_within_phase', 'diff'),
    (r' & CLogit$-$Cox, match', 'cl', 'cond_logit_minus_cox_truncated', 'within_match', 'diff'),
    (r' & BCE$-$CLogit, phase', 'cl', 'bce_minus_cond_logit', 'original_within_phase', 'diff'),
    (r' & BCE$-$CLogit, match', 'cl', 'bce_minus_cond_logit', 'within_match', 'diff'),
    ('group', 'Combined loss Cox + $w$BCE ($w{=}4$ rows$^{b}$)'),
    (r'H4a & $w{=}1$ $-$ Cox, phase', 'om', 'cox_truncated_bce_minus_cox_truncated', 'original_within_phase', 'diff'),
    (r'H4a & $w{=}1$ $-$ Cox, match', 'om', 'cox_truncated_bce_minus_cox_truncated', 'within_match', 'diff'),
    (r'H4a & BCE $-$ $w{=}1$, phase', 'om', 'bce_minus_cox_truncated_bce', 'original_within_phase', 'diff'),
    (r' & gap closed, $w{=}1$ (\%)', 'om', 'closure_share_cox_truncated_bce', 'original_within_phase', 'share'),
    (r' & gap closed, $w{=}4$ (\%)', 'cl', 'closure_share:cox_truncated_bce_w4', 'original_within_phase', 'share'),
    (r' & $w{=}4$ $-$ Cox, match', 'cl', 'cox_truncated_bce_w4_minus_cox_truncated', 'within_match', 'diff'),
    (r' & BCE $-$ $w{=}4$, phase', 'cl', 'bce_minus_cox_truncated_bce_w4', 'original_within_phase', 'diff'),
]


def load():
    om = pd.read_csv(CE / 'objective_match_offset/contrasts.csv')
    om = om[om.population == 'all']
    cl = pd.read_csv(CE / 'condlogit/contrasts_per_cohort.csv')
    oc = pd.read_csv(CE / 'offset_only_cutoff/contrasts.csv')
    return {'om': om, 'cl': cl, 'oc': oc}


def signed(v, digits=4):
    s = f'{abs(v):.{digits}f}'
    return ('$+$' if v >= 0 else '$-$') + s


def plain(v, digits=4):
    return ('$-$' if v < 0 else '') + f'{abs(v):.{digits}f}'


def cell(src, contrast, metric, kind, cohort):
    r = src[(src.cohort == cohort) & (src.contrast == contrast) & (src.metric == metric)]
    assert len(r) == 1, (contrast, metric, cohort, len(r))
    r = r.iloc[0]
    if kind == 'share':
        pct = lambda v: ('$-$' if v < 0 else '') + f'{abs(100 * v):.0f}'
        return f'{pct(r.estimate)} {{\\scriptsize[{pct(r.ci_low)}, {pct(r.ci_high)}]}}', r
    return f'{signed(r.estimate)} {{\\scriptsize[{plain(r.ci_low)}, {plain(r.ci_high)}]}}', r


def main():
    srcs = load()
    lines, values = [], []
    for row in ROWS:
        if row[0] == 'group':
            lines.append(r'\multicolumn{4}{l}{\emph{' + row[1] + r'}} \\')
            continue
        label, key, contrast, metric, kind = row
        cells = []
        for cohort in COHORTS:
            text, r = cell(srcs[key], contrast, metric, kind, cohort)
            cells.append(text)
            values.append(dict(label=label, cohort=cohort, contrast=contrast, metric=metric, kind=kind,
                               estimate=r.estimate, ci_low=r.ci_low, ci_high=r.ci_high, source=key))
        lines.append(f'{label} & {cells[0]} & {cells[1]} \\\\')
    body = '\n'.join(lines)
    tex = (r'''\begin{table}[t]
\centering\footnotesize
\caption{Matched objective controls in the two largest cohorts: AUC differences (three-seed means) with 95\% match-bootstrap intervals, and the share of BCE's within-phase advantage over next-phase Cox that a condition closes (phase, match: within-phase and within-match AUC). H4a: hypothesis of Section~\ref{sec:account}, with predictions recorded before these runs.}
\label{tab:objective}
\setlength{\tabcolsep}{1.6pt}
\begin{tabular}{@{}llrr@{}}
\toprule
 & Contrast and unit & Erangel & Taego \\
\midrule
''' + body + r'''
\bottomrule
\end{tabular}
\smallskip
\parbox{\columnwidth}{\scriptsize $^{a}$Within-phase difference minus the within-match difference with equal phase weights (Section~\ref{sec:method:evaluation}). $^{b}$Predictions recorded before running.}
\end{table}
''')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'rq1_contrasts.tex').write_text(tex)
    pd.DataFrame(values).to_csv(OUT / 'rq1_contrasts_values.csv', index=False)
    print(OUT / 'rq1_contrasts.tex')


if __name__ == '__main__':
    main()
