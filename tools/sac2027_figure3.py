#!/usr/bin/env python3
"""Figure 3 of the SAC 2027 manuscript (v16): per-phase calibration, horizon alignment, and match-phase centering.

(a) Observed and predicted elimination rates by phase (unchanged from the earlier figure; 11 primary cohorts).
(b) Matched objective controls in the two tuned cohorts: cross-entropy minus full-follow-up and next-phase Cox, for
    within-match and within-phase AUC.  Aligning the horizon removes the within-match gap but not the within-phase gap.
(c) Within-phase gaps before and after centering logits within each match-phase, next to the within-match gap, for
    cross-entropy minus next-phase Cox (two cohorts), the team axis, PTR-Former minus GBDT, and PTR-Former minus
    phase-adjusted KRAFTON (11 cohorts, random-effects pooled).
Every value is read from stored analysis outputs; nothing is refitted.  Plotted values go to a CSV next to the figure.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from tools import make_result_figures_v2 as F  # noqa: E402

BASE = REPO / 'reports/sac2027_patch_v14'
CE = BASE / 'control_evaluation'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / 'paper/sac2027/v16_revision/figures'
COHORT_LABEL = {'erangel_tpp_squad': 'Erangel', 'taego_tpp_squad': 'Taego'}


def ci(s):
    lo, hi = (float(x) for x in str(s).strip('()[]').split(','))
    return lo, hi


def objective_rows():
    c = pd.read_csv(CE / 'objective_match_offset/contrasts.csv')
    c = c[c.population == 'all']
    get = lambda cohort, contrast, metric: c[(c.cohort == cohort) & (c.contrast == contrast) & (c.metric == metric)].iloc[0]
    rows_b, rows_c = [], []
    for cohort in ('erangel_tpp_squad', 'taego_tpp_squad'):
        for contrast, label in (('bce_minus_cox_full', 'full Cox'), ('bce_minus_cox_truncated', 'next-phase Cox')):
            for metric, series in (('within_match', 'within match'), ('original_within_phase', 'within phase')):
                r = get(cohort, contrast, metric)
                rows_b.append(dict(panel='b', row=f'{COHORT_LABEL[cohort]}, {label}', series=series, estimate=r.estimate,
                                   ci_low=r.ci_low, ci_high=r.ci_high, source='objective_match_offset/contrasts.csv'))
        for metric, series in (('original_within_phase', 'within phase'), ('centred_within_phase', 'within phase, centered'),
                               ('within_match', 'within match')):
            r = get(cohort, 'bce_minus_cox_truncated', metric)
            rows_c.append(dict(panel='c', row=f'BCE − next-phase Cox, {COHORT_LABEL[cohort]}', series=series, estimate=r.estimate,
                               ci_low=r.ci_low, ci_high=r.ci_high, source='objective_match_offset/contrasts.csv'))
    return rows_b, rows_c


def pooled_rows():
    rows = []
    s = pd.read_csv(CE / 'structure_match_offset/pooled.csv')
    s = s[s.population == 'all']
    for contrast, label in (('team_main', 'Team axis: on − off (pooled)'), ('ptr_minus_gbdt', 'PTR − GBDT (pooled)')):
        for metric, series in (('original_within_phase', 'within phase'), ('centred_within_phase', 'within phase, centered'),
                               ('within_match', 'within match')):
            r = s[(s.contrast == contrast) & (s.metric == metric)].iloc[0]
            lo, hi = ci(r.ci95)
            rows.append(dict(panel='c', row=label, series=series, estimate=r.theta, ci_low=lo, ci_high=hi,
                             source='structure_match_offset/pooled.csv'))
    # PTR-Former minus phase-adjusted KRAFTON: the main-population contrasts quoted in Sec. 6.2, centering from the RQ2 run
    mp = {(r['contrast'], r['metric']): r for r in json.loads((CE / 'main_population/pooled.json').read_text()) if r['population'] == 'all'}
    q = pd.read_csv(CE / 'rq2_match_offset/pooled.csv')
    label = 'PTR − KRAFTON+phase (pooled)'
    for series, src in (('within phase', ('M11R_minus_KRAFTON-tf-h0', 'auc_within_phase')),
                        ('within phase, centered', None), ('within match', ('M11R_minus_KRAFTON-tf-h0', 'auc_within_match_phase'))):
        if src:
            r = mp[src]
            rows.append(dict(panel='c', row=label, series=series, estimate=r['theta'], ci_low=r['ci95'][0], ci_high=r['ci95'][1],
                             source='main_population/pooled.json'))
        else:
            r = q[(q.contrast == 'ptr_minus_krafton_phase') & (q.metric == 'centred_within_phase')].iloc[0]
            lo, hi = ci(r.ci95)
            rows.append(dict(panel='c', row=label, series=series, estimate=r.theta, ci_low=lo, ci_high=hi,
                             source='rq2_match_offset/pooled.csv'))
    return rows


STYLE = {  # marker, face, edge
    'within match': ('o', F.PTR, F.PTR),
    'within phase': ('s', 'white', F.KRAFTON_C),
    'within phase, centered': ('D', F.INK2, F.INK2),
}


def dot_panel(ax, frame, order, series, offsets, xlim, xticks, xlabel):
    y = {r: len(order) - 1 - i for i, r in enumerate(order)}
    ax.axvline(0, color=F.AXIS, lw=.6, ls='--', zorder=0)
    for s, off in zip(series, offsets):
        m, face, edge = STYLE[s]
        g = frame[frame.series == s]
        for _, r in g.iterrows():
            yy = y[r.row] + off
            ax.plot([r.ci_low, r.ci_high], [yy, yy], color=edge, lw=.85, solid_capstyle='butt', zorder=2)
            ax.plot([r.estimate], [yy], marker=m, ms=3.6, mfc=face, mec=edge, mew=.8, ls='none', zorder=3)
    ax.set_yticks([y[r] for r in order], order)
    ax.set_ylim(-.6, len(order) - .4)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks, [f'{t:+.2f}' if t else '0' for t in xticks])
    ax.set_xlabel(xlabel)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color=F.GRID, lw=.4)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def main():
    from matplotlib.lines import Line2D
    counts = pd.read_csv(BASE / 'cohort_counts.csv')
    cohorts = sorted(counts.loc[counts.analysis_role == 'main', 'cohort'])
    assert len(cohorts) == 11
    F.RUNS = BASE / 'checkpoints/ptr_v2'
    primary = BASE / 'primary_results_v2'
    F.SRC.update(contrasts_pooled=primary / 'contrasts_pooled.csv', contrasts_per_cohort=primary / 'contrasts_per_cohort.csv',
                 arm_per_cohort=primary / 'arm_metrics_per_cohort.csv', e2_pooled=primary / 'supplement/e2_contrasts_pooled_dl.csv')
    phase = F.per_phase_table(cohorts)
    rows_b, rows_c = objective_rows()
    rows_c += pooled_rows()
    values = pd.DataFrame(rows_b + rows_c)
    phase.to_csv(OUT / 'fig_contrast_panel_a_per_phase.csv', index=False)
    values.to_csv(OUT / 'fig_contrast_panels_bc_values.csv', index=False)

    F.setup_style()
    W, H = 17.8, 5.8
    fig = F.new_figure(W, H)
    bottom = 2.2
    axa = F.add_axes_cm(fig, 1., bottom, 3.35, H - .55 - bottom)
    handles, labels = F.draw_panel_a(axa, phase)
    la = fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.12 / W, (bottom - .66) / H), frameon=False,
                    handlelength=2., handletextpad=.5, labelspacing=.08, borderpad=0, borderaxespad=0, fontsize=7)
    top = H - .75
    yb, hb = 1.55, top - 1.55
    axb = F.add_axes_cm(fig, 7.65, yb, 2.95, hb)
    order_b = [r['row'] for r in rows_b[::2]]
    dot_panel(axb, values[values.panel == 'b'], order_b, ('within match', 'within phase'), (.13, -.13),
              (-.004, .024), [0, .01, .02], 'BCE minus Cox, AUC')
    axc = F.add_axes_cm(fig, 14.55, yb, 3.05, hb)
    order_c = list(dict.fromkeys(values[values.panel == 'c'].row))
    dot_panel(axc, values[values.panel == 'c'], order_c, ('within phase', 'within phase, centered', 'within match'), (.22, 0, -.22),
              (-.007, .038), [0, .01, .02, .03], 'AUC difference')
    legend_handles = [Line2D([0], [0], marker=STYLE[s][0], mfc=STYLE[s][1], mec=STYLE[s][2], mew=.8, ms=3.6, ls='none')
                      for s in ('within phase', 'within phase, centered', 'within match')]
    lb = fig.legend(legend_handles, ['within phase', 'within phase, centered on match–phase means', 'within match'],
                    loc='lower left', bbox_to_anchor=(5.0 / W, .05 / H), ncol=3, frameon=False, handletextpad=.3,
                    columnspacing=1.2, borderpad=0, borderaxespad=0, fontsize=7)
    for x, letter, title in ((.06, '(a)', 'Per-phase calibration'), (4.95, '(b)', 'Horizon alignment (matched controls)'),
                             (10.95, '(c)', 'Centering within match–phases')):
        F.panel_title(fig, x, H - .30, letter, title)
    fig._legends_to_check = [la, lb]
    notes = F.save_and_check(fig, OUT / 'fig_contrast', 17.8, 9.0, True)
    (OUT / 'fig_contrast_qa.txt').write_text('\n'.join(notes) + '\n')
    print('\n'.join(notes))


if __name__ == '__main__':
    main()
