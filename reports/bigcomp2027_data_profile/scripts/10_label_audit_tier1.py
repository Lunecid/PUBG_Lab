#!/usr/bin/env python3
"""Label audit tier 1: all 13 cohorts, pipeline label (stdb.team_phase_labels) vs truth from kill telemetry and participants.time_survived.  READ-ONLY."""
import argparse, os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import dbh
from label_audit_lib import *
OUT = ROOT + '/reports/bigcomp2027_data_profile/csv'
ap = argparse.ArgumentParser(); ap.add_argument('--chunk', type=int, default=1500); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--tag', default='full'); a = ap.parse_args()
id2coh = cohort_map(); ids = sorted(id2coh)
if a.limit: ids = list(np.random.RandomState(1).choice(ids, a.limit, replace=False))
cur = dbh.conn().cursor(); t0 = time.time(); CNT, DIS, MT, FL, TC, PO, OUTL, RD = [], [], [], [], [], [], [], []; dh, da = [], []
KEEP = ['match_id', 'team_id', 'cohort', 'match_type', 'k', 'nk', 'e_k', 'e_n', 'T_evt', 'T_off', 'pst', 'tr_evt', 'tr_off', 'pr_k', 'ape_k', 'pr_n', 'ape_n', 'n_mem', 'n_ai', 'n_logout', 'n_fb', 'won', 'rrank', 'near_ek', 'near_en', 'near15', 'nopos_k', 'nopos_n', 'knock_k', 'knock_n', 'ai_team', 'ai_all', 'airoyale', 'logout', 'redeploy', 'ai_bucket', 'n_multi', 'term_n', 'tr_evt_strict']
for i in range(0, len(ids), a.chunk):
    r, tt, mt, pl = build_rows(cur, ids[i:i + a.chunk], id2coh)
    CNT.append(r.groupby(['cohort', 'match_type', 'ai_bucket', 'term_n', 'k', 'pst', 'tr_evt', 'tr_evt_strict', 'tr_off']).size().rename('n').reset_index())
    dis = r[(r.pst != r.tr_evt)]; typ, cat = categorize(dis); dis = dis.assign(typ=typ, cat=cat); DIS.append(dis[KEEP + ['typ', 'cat']])
    FL.append(r.assign(agree=(r.pst == r.tr_evt)).groupby(['agree', 'near15', 'nopos_k', 'nopos_n', 'knock_k', 'knock_n', 'ai_team', 'airoyale', 'logout', 'redeploy']).size().rename('n').reset_index()); MT.append(mt)
    both = tt[np.isfinite(tt.T_evt) | np.isfinite(tt.T_off)].copy(); both['ai_bucket'] = np.select([both.n_ai == 0, both.n_ai == both.n_mem], ['no_ai', 'all_ai'], 'mixed'); both['d'] = (both.T_off - both.T_evt)
    TC.append(both.assign(match_ok=both.T_evt.notna()).groupby(['ai_bucket']).apply(lambda g: pd.Series(dict(n=len(g), both_inf=int((~np.isfinite(g.T_evt) & ~np.isfinite(g.T_off)).sum()), evt_only_inf=int((~np.isfinite(g.T_evt) & np.isfinite(g.T_off)).sum()), off_only_inf=int((np.isfinite(g.T_evt) & ~np.isfinite(g.T_off)).sum()),
        both_fin=int((np.isfinite(g.T_evt) & np.isfinite(g.T_off)).sum()), within5=int(((g.d.abs() <= 5) & np.isfinite(g.d)).sum()), within15=int(((g.d.abs() <= 15) & np.isfinite(g.d)).sum())))).reset_index())
    h = pl[(~pl.is_ai) & pl.d_kill.notna() & pl.ts.notna()]; dh.append((h.d_kill - h.ts).values.astype('float32')); x = pl[pl.is_ai & pl.d_kill.notna() & pl.ts.notna()]; da.append((x.d_kill - x.ts).values.astype('float32'))
    hh = pl[(~pl.is_ai) & pl.d_kill.notna() & pl.ts.notna()].copy(); hh['df'] = hh.ep_first - hh.S - hh.ts; hh['dl'] = hh.d_kill - hh.ts; hh['b'] = np.select([hh.n_ev >= 3, hh.n_ev == 2], ['n_ev3+', 'n_ev2'], 'n_ev1'); hh['first_ok'] = hh.df.abs() <= 2; hh['last_ok'] = hh.dl.abs() <= 2
    PO.append(hh.groupby(['b', 'first_ok', 'last_ok']).size().rename('n').reset_index()); OUTL.append(hh[~hh.first_ok & ~hh.last_ok].head(12)[['match_id', 'player_id', 'team_id', 'death_type', 'ts', 'df', 'dl', 'n_ev']])
    RD.append(tt.assign(cohort=tt.match_id.map(id2coh)).groupby('cohort').apply(lambda g: pd.Series(dict(teams=len(g), teams_redeploy=int((g.n_multi > 0).sum()), teams_logout=int((g.n_logout > 0).sum())))).reset_index())
    print(f'[{time.time()-t0:6.1f}s] chunk {i // a.chunk + 1}/{(len(ids) + a.chunk - 1) // a.chunk} rows={len(r):,} disagree={len(dis):,}', flush=True)
cnt = pd.concat(CNT).groupby(['cohort', 'match_type', 'ai_bucket', 'term_n', 'k', 'pst', 'tr_evt', 'tr_evt_strict', 'tr_off']).n.sum().reset_index(); cnt.to_csv(f'{OUT}/label_audit_{a.tag}_counts.csv', index=False)
dis = pd.concat(DIS); dis.to_csv(f'{OUT}/label_audit_{a.tag}_disagreements.csv.gz', index=False, compression='gzip'); mt = pd.concat(MT); mt.to_csv(f'{OUT}/label_audit_{a.tag}_match_clock.csv', index=False)
fl = pd.concat(FL).groupby(['agree', 'near15', 'nopos_k', 'nopos_n', 'knock_k', 'knock_n', 'ai_team', 'airoyale', 'logout', 'redeploy']).n.sum().reset_index(); fl.to_csv(f'{OUT}/label_audit_{a.tag}_flag_counts.csv', index=False)
tc = pd.concat(TC).groupby('ai_bucket').sum(numeric_only=True).reset_index(); tc.to_csv(f'{OUT}/label_audit_{a.tag}_team_T_agreement.csv', index=False)
dh = np.concatenate(dh); da = np.concatenate(da); qq = [0, .01, .05, .25, .5, .75, .95, .99, 1]
json.dump(dict(human=dict(n=len(dh), q={str(p): float(np.quantile(dh, p)) for p in qq}, share_abs_le2=float((np.abs(dh) <= 2).mean())), ai=dict(n=len(da), q={str(p): float(np.quantile(da, p)) for p in qq}, share_abs_le2=float((np.abs(da) <= 2).mean()))), open(f'{OUT}/label_audit_{a.tag}_player_offsets.json', 'w'), indent=1)
pd.concat(PO).groupby(['b', 'first_ok', 'last_ok']).n.sum().reset_index().to_csv(f'{OUT}/label_audit_{a.tag}_player_outliers.csv', index=False); pd.concat(OUTL).to_csv(f'{OUT}/label_audit_{a.tag}_player_outlier_examples.csv', index=False)
rd = pd.concat(RD).groupby('cohort').sum().reset_index(); rd.to_csv(f'{OUT}/label_audit_{a.tag}_redeploy_by_cohort.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print('HUMAN kill-vs-time_survived outliers (b=kill-event count, first_ok/last_ok = |diff|<=2s):'); print(pd.read_csv(f'{OUT}/label_audit_{a.tag}_player_outliers.csv').to_string(index=False)); print('redeploy/logout teams by cohort:'); print(rd.to_string(index=False))
print('strict (terminal next phase ends at last grid point) vs extended truth, rows changed:', int(cnt[cnt.tr_evt != cnt.tr_evt_strict].n.sum()), '| terminal-next rows:', int(cnt[cnt.term_n].n.sum()))
tot = cnt.groupby(['pst', 'tr_evt']).n.sum().unstack(fill_value=0); print('CONFUSION pipe (rows: 0 out,1 neg,2 pos) x truth_events (cols):'); print(tot.to_string()); print('rows', int(cnt.n.sum()), 'agree', round(float(np.trace(tot.values) / tot.values.sum()), 4))
tot2 = cnt.groupby(['pst', 'tr_off']).n.sum().unstack(fill_value=0); print('CONFUSION pipe x truth_official (cols -1,0,1,2):'); print(tot2.to_string())
print('events-vs-official truth agree share:', round(float(cnt[cnt.tr_evt == cnt.tr_off].n.sum() / cnt.n.sum()), 4)); print(dis.groupby(['typ', 'cat']).size().unstack(fill_value=0).to_string()); print(tc.to_string(index=False))
po = json.load(open(f'{OUT}/label_audit_{a.tag}_player_offsets.json')); print('player offsets d_kill_last - time_survived: human', {k: round(v, 2) for k, v in po['human']['q'].items()}, 'share|d|<=2:', round(po['human']['share_abs_le2'], 4), '| AI', {k: round(v, 1) for k, v in po['ai']['q'].items()}, 'share|d|<=2:', round(po['ai']['share_abs_le2'], 4)); print('match clock: off_ts q', mt.off_ts.quantile([.05, .5, .95]).round(2).to_dict(), 'delta_gs q', mt.delta_gs.quantile([.05, .5, .95, 1]).round(2).to_dict())
