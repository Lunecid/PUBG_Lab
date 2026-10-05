#!/usr/bin/env python3
"""Figures: (1) next-phase elimination rate by cut phase per cohort, (2) zone schedule (phase timing) by map.  Reads csv/*.csv only."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D; from matplotlib.patches import Patch
ROOT = '/Users/lunecid/project/PUBG/reports/bigcomp2027_data_profile'; CSV = ROOT + '/csv'; FIG = ROOT + '/figures'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e3e2de', '#ffffff'
MAPCOL = {'Erangel': '#2a78d6', 'Taego': '#eb6834', 'Miramar': '#1baf7a', 'Rondo': '#eda100', 'Sanhok': '#e87ba4', 'Vikendi': '#008300', 'Paramo': '#4a3aa7', 'Karakin': '#e34948'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.linewidth': 0.6, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.edgecolor': INK2, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'text.color': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})
hz = pd.read_csv(f'{CSV}/hazard_by_cohort_cutphase.csv'); hz = hz[hz.cut_phase <= 7]; cs = pd.read_csv(f'{CSV}/cohort_summary.csv').set_index('slug')
lab = lambda s: f"{cs.loc[s].map_label} {cs.loc[s].perspective} {cs.loc[s].team_size}"
def wil(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; ctr = (p + z * z / (2 * n)) / d; hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return ctr - hw, ctr + hw
pool = hz.groupby('cut_phase')[['risk', 'pos']].sum().reset_index(); pool['hazard'] = pool.pos / pool.risk; ci = [wil(r.pos, r.risk) for r in pool.itertuples()]; pool['lo'] = [x[0] for x in ci]; pool['hi'] = [x[1] for x in ci]
layout = [['erangel_tpp_squad', 'taego_tpp_squad', 'miramar_desert_tpp_squad', 'rondo_tpp_squad', 'sanhok_tpp_squad'],
          ['vikendi_tpp_squad', 'karakin_tpp_squad', 'paramo_tpp_squad', 'erangel_fpp_squad', 'taego_fpp_squad'],
          ['erangel_tpp_solo', 'erangel_tpp_duo', 'taego_tpp_duo', '__all__', '__legend__']]
fig, axes = plt.subplots(3, 5, figsize=(7.2, 5.0), sharex=True, sharey=True)
for r in range(3):
    for cidx in range(5):
        ax = axes[r][cidx]; key = layout[r][cidx]
        if key == '__legend__':
            ax.axis('off'); hd = [Line2D([0], [0], color=col, lw=1.6, marker='o', ms=3, label=mp) for mp, col in MAPCOL.items()]
            hd += [Line2D([0], [0], color='#8d8c86', lw=0.9, ls='--', label='pooled, 13 cohorts'), Patch(facecolor='#8d8c86', alpha=0.25, label='95% Wilson CI')]
            ax.legend(handles=hd, loc='center', frameon=False, fontsize=6.2, handlelength=1.6, labelspacing=0.45, title='Map', title_fontsize=6.5); continue
        ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
        if key == '__all__':
            for s_, d in hz.groupby('cohort'): ax.plot(d.cut_phase, d.hazard, color='#c4c3bc', lw=0.7)
            ax.fill_between(pool.cut_phase, pool.lo, pool.hi, color='#8d8c86', alpha=0.25, lw=0); ax.plot(pool.cut_phase, pool.hazard, color=INK, lw=1.4, marker='o', ms=2.8)
            ax.set_title('All 13 cohorts', fontsize=7, pad=11); ax.text(0.5, 1.015, f"pooled risk n={int(pool.risk.iloc[0]):,} to {int(pool.risk.iloc[-1]):,}", transform=ax.transAxes, ha='center', va='bottom', fontsize=5.6, color=INK2)
        else:
            d = hz[hz.cohort == key].sort_values('cut_phase'); col = MAPCOL[cs.loc[key].map_label]
            ax.plot(pool.cut_phase, pool.hazard, color='#8d8c86', lw=0.9, ls='--'); ax.fill_between(d.cut_phase, d.ci_lo, d.ci_hi, color=col, alpha=0.20, lw=0); ax.plot(d.cut_phase, d.hazard, color=col, lw=1.5, marker='o', ms=3)
            ax.set_title(lab(key), fontsize=7, pad=11); ax.text(0.5, 1.015, f"risk n={int(d.risk.iloc[0]):,} to {int(d.risk.iloc[-1]):,}", transform=ax.transAxes, ha='center', va='bottom', fontsize=5.6, color=INK2)
        ax.set_xlim(0.6, 7.4); ax.set_ylim(0, 0.76); ax.set_xticks(range(1, 8)); ax.set_yticks([0, 0.2, 0.4, 0.6]); ax.tick_params(length=2, width=0.5, labelsize=6)
fig.supxlabel('Cut phase k  (label: team eliminated during phase k+1, given alive at the end of phase k)', fontsize=7, color=INK2); fig.supylabel('Next-phase elimination rate', fontsize=7, color=INK2)
fig.tight_layout(rect=(0.01, 0.02, 1, 1), h_pad=1.0, w_pad=0.6)
for ext, kw in (('pdf', {}), ('png', {'dpi': 300})): fig.savefig(f'{FIG}/fig1_hazard_by_phase_per_cohort.{ext}', **kw)
plt.close(fig)
# ---- figure 2: zone schedule by map ----
import json
pm = pd.read_csv(f'{CSV}/phase_timing_by_map.csv'); pm = pm[pm.phase_idx <= 9]
R = '/Users/lunecid/project/PUBG/model_stgcn/checkpoints/br_om_ladder_required_v1'; ids = set()
for d_ in os.listdir(R):
    p_ = f'{R}/{d_}/split.json'
    if os.path.isfile(p_): ids |= set(sum(json.load(open(p_))['splits'].values(), []))
flat = pd.read_csv('/Users/lunecid/project/PUBG/model_stgcn/experiment_datasets/research_cohort_flat/match_cohort_flat.csv', usecols=['match_id', 'map_name', 'duration'])
dur = flat[flat.match_id.isin(ids)].groupby('map_name').duration.median(); pd.DataFrame({'map_name': dur.index, 'duration_median_s': dur.values}).to_csv(f'{CSV}/duration_median_by_map.csv', index=False)
RAMP = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#0d366b']
p5 = pm[pm.phase_idx == 5].set_index('map_label').start_med; n1 = pm[pm.phase_idx == 1].set_index('map_label').n_matches; m2n = pm.drop_duplicates('map_label').set_index('map_label').map_name
maps = sorted(p5.index, key=lambda x: (-p5[x], x)); fig, ax = plt.subplots(figsize=(7.2, 3.2))
for yi, mp in enumerate(maps):
    rows = [r for r in pm[pm.map_label == mp].sort_values('phase_idx').itertuples() if r.n_matches >= 30]
    for i, r in enumerate(rows):
        end = (rows[i + 1].start_med - 10) if i + 1 < len(rows) else r.end_med; w = end - r.start_med
        ax.barh(yi, w, left=r.start_med, height=0.64, color=RAMP[int(r.phase_idx) - 1], edgecolor=SURF, linewidth=1.0)
        if w >= 24: ax.text(r.start_med + w / 2, yi, str(int(r.phase_idx)), ha='center', va='center', fontsize=6, color=('#ffffff' if r.phase_idx >= 5 else INK))
    ax.plot([dur[m2n[mp]]] * 2, [yi - 0.46, yi + 0.46], color=INK, lw=1.5, solid_capstyle='butt')
ax.set_yticks(range(len(maps))); ax.set_yticklabels([f"{mp}  (n={int(n1[mp]):,})" for mp in maps], fontsize=6.6); ax.set_ylim(len(maps) - 0.4, -0.85); ax.set_xlim(0, 2050)
ax.set_xlabel('Match clock (s since match start); bar = median start of phase k to median start of phase k+1', fontsize=7)
ax.axvline(2000, color=INK2, lw=0.7, ls=':'); ax.text(1994, -0.62, '2000 s', ha='right', va='center', fontsize=6, color=INK2); ax.grid(axis='x', color=GRID, lw=0.5); ax.set_axisbelow(True)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
ax.tick_params(length=2, width=0.5, labelsize=6.4)
hd = [Patch(facecolor=RAMP[i], label=f'Phase {i + 1}') for i in range(9)] + [Line2D([0], [0], color=INK, lw=1.5, label='median match duration')]
ax.legend(handles=hd, ncol=10, loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=5.8, handlelength=1.1, columnspacing=0.8)
fig.tight_layout()
for ext, kw in (('pdf', {}), ('png', {'dpi': 300})): fig.savefig(f'{FIG}/fig2_phase_timing_by_map.{ext}', **kw)
print('figures written:', sorted(os.listdir(FIG)), '| median duration by map (s):', dur.round(0).to_dict())
