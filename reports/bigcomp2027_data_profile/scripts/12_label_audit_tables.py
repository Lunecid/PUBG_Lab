#!/usr/bin/env python3
"""Aggregate tier-1 counts into the label-audit metric tables (reads csv/label_audit_{TAG}_*.csv; no DB)."""
import json, os, sys
import numpy as np, pandas as pd
C = '/Users/lunecid/project/PUBG/reports/bigcomp2027_data_profile/csv'; TAG = sys.argv[1] if len(sys.argv) > 1 else 'full'; pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)
cnt = pd.read_csv(f'{C}/label_audit_{TAG}_counts.csv'); cs = pd.read_csv(f'{C}/cohort_summary.csv').set_index('slug'); order = list(cs.sort_values('n_total', ascending=False).index)
def met(df, tr='tr_evt'):
    S = lambda m: int(df[m].n.sum()); n = S(df.n > -1) if False else int(df.n.sum())
    pr, tk = S((df.pst >= 1)), S((df[tr] >= 1)); pp, tp, bp = S(df.pst == 2), S(df[tr] == 2), S((df.pst == 2) & (df[tr] == 2)); ag = S(df.pst == df[tr])
    return pd.Series(dict(rows=n, agree=ag / n, changed=n - ag, pipe_risk=pr, truth_risk=tk, risk_fp=S((df.pst >= 1) & (df[tr] == 0)), risk_fn=S((df.pst == 0) & (df[tr] >= 1)), pipe_pos=pp, truth_pos=tp, both_pos=bp,
                          label_fp=S((df.pst == 2) & (df[tr] == 1)), label_fn=S((df.pst == 1) & (df[tr] == 2)), precision=bp / pp if pp else np.nan, recall=bp / tp if tp else np.nan, hazard_pipe=pp / pr if pr else np.nan, hazard_truth=tp / tk if tk else np.nan))
def sel(df, mask): return df[mask]
# helper: boolean count on the n-weighted frame
class W:
    def __init__(self, df): self.df = df
def met2(df, tr='tr_evt'):
    def S(mask): return int(df.loc[mask, 'n'].sum())
    n = int(df.n.sum()); pr, tk = S(df.pst >= 1), S(df[tr] >= 1); pp, tp, bp = S(df.pst == 2), S(df[tr] == 2), S((df.pst == 2) & (df[tr] == 2)); ag = S(df.pst == df[tr])
    return pd.Series(dict(rows=n, agree=ag / n, changed=n - ag, pipe_risk=pr, truth_risk=tk, risk_fp=S((df.pst >= 1) & (df[tr] == 0)), risk_fn=S((df.pst == 0) & (df[tr] >= 1)), pipe_pos=pp, truth_pos=tp, both_pos=bp, label_fp=S((df.pst == 2) & (df[tr] == 1)), label_fn=S((df.pst == 1) & (df[tr] == 2)),
                          precision=bp / pp if pp else np.nan, recall=bp / tp if tp else np.nan, hazard_pipe=pp / pr if pr else np.nan, hazard_truth=tp / tk if tk else np.nan))
allm = met2(cnt); print('OVERALL (events truth):'); print(allm.round(4).to_string())
bys = cnt.groupby('k').apply(lambda d: met2(d, 'tr_evt_strict')); print('\nBY k with STRICT truth (terminal next phase ends at last grid point): hazard_truth', bys.hazard_truth.round(4).to_dict(), '| overall changed vs extended:', int(cnt[cnt.tr_evt != cnt.tr_evt_strict].n.sum()))
byt = cnt.groupby('term_n').apply(met2); print('\nBY terminal-next flag:'); print(byt[['rows', 'agree', 'risk_fp', 'risk_fn', 'label_fp', 'label_fn', 'precision', 'recall', 'hazard_pipe', 'hazard_truth']].round(4).to_string())
byk = cnt.groupby('k').apply(met2); byk.loc['ALL'] = allm; byk.to_csv(f'{C}/label_audit_metrics_by_k.csv'); print('\nBY k:'); print(byk[['rows', 'agree', 'pipe_risk', 'truth_risk', 'risk_fp', 'risk_fn', 'pipe_pos', 'truth_pos', 'both_pos', 'label_fp', 'label_fn', 'precision', 'recall', 'hazard_pipe', 'hazard_truth']].round(4).to_string())
byc = cnt.groupby('cohort').apply(met2).loc[order]; byc.to_csv(f'{C}/label_audit_metrics_by_cohort.csv'); print('\nBY cohort:'); print(byc[['rows', 'agree', 'risk_fp', 'risk_fn', 'label_fp', 'label_fn', 'precision', 'recall', 'hazard_pipe', 'hazard_truth']].round(4).to_string())
bym = cnt.groupby('match_type').apply(met2); bym.to_csv(f'{C}/label_audit_metrics_by_matchtype.csv'); print('\nBY match_type:'); print(bym[['rows', 'agree', 'risk_fp', 'risk_fn', 'label_fp', 'label_fn', 'precision', 'recall', 'hazard_pipe', 'hazard_truth']].round(4).to_string())
bya = cnt.groupby('ai_bucket').apply(met2); bya.to_csv(f'{C}/label_audit_metrics_by_aibucket.csv'); print('\nBY ai_bucket:'); print(bya[['rows', 'agree', 'risk_fp', 'risk_fn', 'label_fp', 'label_fn', 'precision', 'recall', 'hazard_pipe', 'hazard_truth']].round(4).to_string())
byck = cnt.groupby(['cohort', 'k']).apply(met2).reset_index(); byck.to_csv(f'{C}/label_audit_metrics_by_cohort_k.csv', index=False)
bykt = cnt[cnt.ai_bucket == 'no_ai'].groupby('k').apply(lambda d: met2(d, 'tr_off')); print('\nHUMAN-ONLY teams: pipeline vs OFFICIAL (time_survived) truth by k:'); print(bykt[['rows', 'agree', 'risk_fp', 'risk_fn', 'label_fp', 'label_fn', 'precision', 'recall', 'hazard_pipe', 'hazard_truth']].round(4).to_string())
bykh = cnt[cnt.ai_bucket == 'no_ai'].groupby('k').apply(met2); print('\nHUMAN-ONLY teams: pipeline vs EVENTS truth by k:'); print(bykh[['rows', 'agree', 'risk_fp', 'risk_fn', 'label_fp', 'label_fn', 'precision', 'recall', 'hazard_pipe', 'hazard_truth']].round(4).to_string())
h = cnt[cnt.ai_bucket == 'no_ai']; print('human-only: events-vs-official truth agree share:', round(float(h[h.tr_evt == h.tr_off].n.sum() / h.n.sum()), 4), '| all-AI:', round(float(cnt[(cnt.ai_bucket == 'all_ai') & (cnt.tr_evt == cnt.tr_off)].n.sum() / cnt[cnt.ai_bucket == 'all_ai'].n.sum()), 4))
mc = pd.read_csv(f'{C}/label_audit_{TAG}_match_clock.csv'); print('\nMATCH CLOCK (17,943 matches): delta_gs q', mc.delta_gs.quantile([0, .05, .25, .5, .75, .95, 1]).round(2).to_dict(), '| share >5/10/15/20 s:', [round(float((mc.delta_gs > x).mean()), 4) for x in (5, 10, 15, 20)], '| off_ts (human) n/q:', int(mc.off_ts.notna().sum()), mc.off_ts.quantile([.05, .25, .5, .75, .95]).round(2).to_dict(), '| mad_ts median:', round(float(mc.mad_ts.median()), 2))
print('players per match / matches with redeploy players:', round(float(mc.n_players.mean()), 1), int((mc.n_multi > 0).sum()), 'of', len(mc), '| share', round(float((mc.n_multi > 0).mean()), 4), '| by match_type:', mc.groupby('match_type').apply(lambda g: round(float((g.n_multi > 0).mean()), 3)).to_dict())
fl = pd.read_csv(f'{C}/label_audit_{TAG}_flag_counts.csv'); fa = fl.groupby('agree').n.sum(); print('\nFLAG base rates (agree vs disagree rows):')
for f in ['near15', 'nopos_k', 'nopos_n', 'knock_k', 'knock_n', 'ai_team', 'airoyale', 'logout', 'redeploy']: print(f'  {f:9s} agree {fl[(fl.agree) & fl[f]].n.sum() / fa[True]:.4f}  disagree {fl[(~fl.agree) & fl[f]].n.sum() / fa[False]:.4f}')
print('TEAM T agreement:'); print(pd.read_csv(f'{C}/label_audit_{TAG}_team_T_agreement.csv').to_string(index=False))
dis = pd.read_csv(f'{C}/label_audit_{TAG}_disagreements.csv.gz', usecols=['typ', 'cat', 'ai_bucket', 'airoyale', 'k', 'redeploy']); print('\nDISAGREE rows by cat x ai_bucket:'); print(dis.groupby(['cat', 'ai_bucket']).size().unstack(fill_value=0).to_string()); print('by typ x k:'); print(dis.groupby(['typ', 'k']).size().unstack(fill_value=0).to_string())
