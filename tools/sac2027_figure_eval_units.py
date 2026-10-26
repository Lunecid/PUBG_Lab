#!/usr/bin/env python3
"""Figure 2 of the SAC 2027 manuscript (v17e): AUC differences on four evaluation units.

Columns are evaluation units (within phase, within phase after match-phase centering, within match, within match with
the phase weights of within-phase AUC); rows are contrasts.  Top: cross-entropy (BCE) minus next-phase Cox in the two
cohorts where the explanation was formed and pooled over the other nine.  Middle: BCE minus the conditional logistic
loss, same rows.  Third: habitat block with - without (GBDT) and team attention on - off (PTR-Former), 11 cohorts.
Bottom: our designs minus phase-adjusted
KRAFTON, pooled over 11 cohorts, with KRAFTON centered on its native Cox score.  One colour, no legend; the gray band is
the SESOI (+-0.005) and every column shares the x range, so a point that moves into the band from left to right was a
match-level component.  Every value is read from stored analysis outputs; plotted values go to a CSV next to the figure.
Usage: python tools/sac2027_figure_eval_units.py <output-dir>
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
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / 'paper/figures'
SESOI = 0.005
UNITS = (('original_within_phase', 'Within phase'), ('within_match', 'Within match'),
         ('within_match_phase_mean', 'Within match, equal phase weights'), ('centred_within_phase', 'Within phase, centered'))
# Column heads: a short bold name over a grey line saying which pairs the column scores (Section 3: A_B, A_W).
HEADS = {'original_within_phase': ('Within phase', 'mostly different-match pairs'),
         'within_match': ('Within match', 'pairs pooled over boundaries'),
         'within_match_phase_mean': ('Within match', 'equal phase weights'),
         'centred_within_phase': ('Within phase', 'match-phase mean removed')}
ORDER = (('top', 'Erangel'), ('top', 'Taego'), ('top', 'Other nine (pooled)'),
         ('mid', 'Erangel'), ('mid', 'Taego'), ('mid', 'Other nine (pooled)'),
         ('rq2', 'Habitat block (GBDT)'), ('rq2', 'Team attention (PTR)'),
         ('bottom', 'PTR-Former'), ('bottom', 'GBDT'))


def ci(s):
    lo, hi = (float(x) for x in str(s).strip('()[]').split(','))
    return lo, hi


def rows():
    out = []
    c = pd.read_csv(CE / 'objective_match_offset/contrasts.csv')
    c = c[(c.population == 'all') & (c.contrast == 'bce_minus_cox_truncated')]
    for cohort, label in (('erangel_tpp_squad', 'Erangel'), ('taego_tpp_squad', 'Taego')):
        for metric, unit in UNITS:
            r = c[(c.cohort == cohort) & (c.metric == metric)].iloc[0]
            out.append(dict(group='top', row=label, unit=unit, estimate=r.estimate, ci_low=r.ci_low, ci_high=r.ci_high,
                            source='objective_match_offset/contrasts.csv'))
    p = pd.read_csv(CE / 'objective_all_cohorts/pooled.csv')
    p = p[(p.selection == 'within_phase_auc') & (p.scope == 'nine_new') & (p.contrast == 'bce_minus_cox_truncated')]
    for metric, unit in UNITS:
        r = p[p.metric == metric].iloc[0]
        lo, hi = ci(r.ci95)
        out.append(dict(group='top', row='Other nine (pooled)', unit=unit, estimate=r.theta, ci_low=lo, ci_high=hi,
                        source='objective_all_cohorts/pooled.csv'))
    # BCE minus the conditional logistic loss (control_evaluation/condlogit; CONDLOGIT_PLAN_KO_2026-10-09.md).
    cl = pd.read_csv(CE / 'condlogit/contrasts_per_cohort.csv')
    cl = cl[cl.contrast == 'bce_minus_cond_logit']
    for cohort, label in (('erangel_tpp_squad', 'Erangel'), ('taego_tpp_squad', 'Taego')):
        for metric, unit in UNITS:
            r = cl[(cl.cohort == cohort) & (cl.metric == metric)].iloc[0]
            out.append(dict(group='mid', row=label, unit=unit, estimate=r.estimate, ci_low=r.ci_low, ci_high=r.ci_high,
                            source='condlogit/contrasts_per_cohort.csv'))
    cp = pd.read_csv(CE / 'condlogit/pooled.csv')
    cp = cp[(cp.scope == 'nine_new') & (cp.contrast == 'bce_minus_cond_logit')]
    for metric, unit in UNITS:
        r = cp[cp.metric == metric].iloc[0]
        lo, hi = ci(r.ci95)
        out.append(dict(group='mid', row='Other nine (pooled)', unit=unit, estimate=r.theta, ci_low=lo, ci_high=hi,
                        source='condlogit/pooled.csv'))
    # Inputs vs. structure (RQ2): habitat block with - without (GBDT) and team attention on - off (PTR-Former 2x2 main
    # effect), all four columns from the RQ2 offset diagnostic pooled over 11 cohorts.
    q2 = pd.read_csv(CE / 'rq2_match_offset/pooled.csv')
    for key, label in (('hgb_cut_nohab', 'Habitat block (GBDT)'), ('team_main', 'Team attention (PTR)')):
        for metric, unit in UNITS:
            r = q2[(q2.contrast == key) & (q2.metric == metric)].iloc[0]
            (lo, hi) = ci(r.ci95)
            out.append(dict(group='rq2', row=label, unit=unit, estimate=r.theta, ci_low=lo, ci_high=hi,
                            source='rq2_match_offset/pooled.csv'))
    # Designs minus phase-adjusted KRAFTON: within phase and within match as quoted in Sec. 6.2 (main-population run);
    # centering on KRAFTON's native Cox score and the phase-weighted within-match AUC from the RQ2 diagnostic.
    mp = {(r['contrast'], r['metric']): r for r in json.loads((CE / 'main_population/pooled.json').read_text())
          if r['population'] == 'all'}
    q = pd.read_csv(CE / 'rq2_match_offset/pooled.csv')
    for arm, key, label in (('M11R', 'ptr_minus_krafton_phase', 'PTR-Former'), ('HGB-cut', 'gbdt_minus_krafton_phase', 'GBDT')):
        for metric, unit in UNITS:
            if metric in ('original_within_phase', 'within_match'):
                name = 'auc_within_phase' if metric == 'original_within_phase' else 'auc_within_match_phase'
                r = mp[(f'{arm}_minus_KRAFTON-tf-h0', name)]
                est, lo, hi, src = r['theta'], r['ci95'][0], r['ci95'][1], 'main_population/pooled.json'
            else:
                contrast = key + '_native' if metric == 'centred_within_phase' else key
                r = q[(q.contrast == contrast) & (q.metric == metric)].iloc[0]
                (lo, hi), est, src = ci(r.ci95), r.theta, 'rq2_match_offset/pooled.csv'
            out.append(dict(group='bottom', row=label, unit=unit, estimate=est, ci_low=lo, ci_high=hi, source=src))
    return pd.DataFrame(out)


def main():
    values = rows()
    assert values.notna().all().all() and len(values) == len(ORDER) * len(UNITS)
    OUT.mkdir(parents=True, exist_ok=True)
    values.to_csv(OUT / 'fig_eval_units_values.csv', index=False)

    F.setup_style()
    W, H = 17.8, 8.6
    fig = F.new_figure(W, H)
    gap = 1.0  # extra row units between the groups, holding the bottom group's header
    ypos, y = {}, 0.0
    for key in reversed(ORDER):
        ypos[key] = y
        y += 1.0 + (gap if key in (('bottom', 'PTR-Former'), ('mid', 'Erangel'), ('rq2', 'Habitat block (GBDT)')) else 0.0)
    ylo, yhi = -0.6, y - 1.0 + 1.0
    left, width, sep, bottom, top = 3.55, 3.22, 0.30, 0.95, H - 0.85
    xlim, ticks = (-0.008, 0.038), [0, 0.01, 0.02, 0.03]
    color = F.PTR
    for j, (metric, unit) in enumerate(UNITS):
        ax = F.add_axes_cm(fig, left + j * (width + sep), bottom, width, top - bottom)
        ax.axvspan(-SESOI, SESOI, color='#e4e4e4', zorder=0, lw=0)
        for t in ticks[1:]:  # grid lines away from zero only, so the zero line is never doubled by a grid line
            ax.axvline(t, color=F.GRID, lw=.4, zorder=0.5)
        ax.axvline(0, color='#333333', lw=.9, zorder=1.5)
        for _, r in values[values.unit == unit].iterrows():
            yy = ypos[(r.group, r.row)]
            ax.plot([r.ci_low, r.ci_high], [yy, yy], color=color, lw=1.0, solid_capstyle='butt', zorder=2)
            ax.plot([r.estimate], [yy], marker='o', ms=4.0, mfc=color, mec=color, ls='none', zorder=3)
        ax.set_xlim(*xlim)
        ax.set_ylim(ylo, yhi)
        ax.set_xticks(ticks, ['0' if t == 0 else f'{t:.2f}' for t in ticks])
        ax.grid(False)
        head, sub = HEADS[metric]
        ax.set_title(head, fontsize=7, fontweight='bold', pad=11)
        ax.annotate(sub, xy=(0.5, 1.0), xycoords='axes fraction', xytext=(0, 3), textcoords='offset points',
                    fontsize=7, color='#666666', ha='center', va='bottom', annotation_clip=False)
        ax.tick_params(axis='y', length=0)
        ax.set_yticks([ypos[k] for k in ORDER], [r for _, r in ORDER] if j == 0 else [''] * len(ORDER))
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        if j == 0:
            for key, text in ((('top', 'Erangel'), 'BCE − next-phase Cox'),
                              (('mid', 'Erangel'), 'BCE − conditional logit'),
                              (('rq2', 'Habitat block (GBDT)'), 'Inputs or structure'),
                              (('bottom', 'PTR-Former'), 'Reference design −\nKRAFTON+phase')):
                ax.annotate(text, xy=(0, ypos[key] + 0.75), xycoords=('axes fraction', 'data'), xytext=(-3.3 / 2.54 * 72, 0),
                            textcoords='offset points', fontsize=7, fontweight='bold', ha='left', va='center',
                            annotation_clip=False)
    span_mid = left + (4 * width + 3 * sep) / 2
    fig.text(span_mid / W, 0.12 / H, 'AUC difference', fontsize=7, ha='center', va='bottom')
    notes = F.save_and_check(fig, OUT / 'fig_eval_units', 17.8, 8.8, True)
    (OUT / 'fig_eval_units_qa.txt').write_text('\n'.join(notes) + '\n')
    print('\n'.join(notes))


if __name__ == '__main__':
    main()
