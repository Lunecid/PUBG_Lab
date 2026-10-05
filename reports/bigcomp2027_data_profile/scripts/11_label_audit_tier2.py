#!/usr/bin/env python3
"""Label audit tier 2: position-level replication of the pipeline label on a stratified match sample, a clock-fixed variant, and boundary member states.  READ-ONLY."""
import argparse, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import dbh
from label_audit_lib import *
OUT = ROOT + '/reports/bigcomp2027_data_profile/csv'
ap = argparse.ArgumentParser(); ap.add_argument('--per-cohort', type=int, default=250); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--chunk', type=int, default=40); ap.add_argument('--tag', default='t2'); a = ap.parse_args()
id2coh = cohort_map(); m = pd.read_csv(f'{OUT}/sample_match_level.csv'); ids = m.groupby('cohort').head(a.per_cohort).match_id.tolist()
if a.limit: ids = ids[::max(1, len(ids) // a.limit)][:a.limit]
cur = dbh.conn().cursor(); t0 = time.time()
POS = "SELECT match_id, account_id AS acc, team_id AS team, elapsed_time AS t, health::float8 AS health, extract(epoch from event_time)::float8 AS ep FROM pubg.telem_positions WHERE match_id = ANY(%s) AND is_game >= 0.1 AND account_id IS NOT NULL AND team_id IS NOT NULL AND pos_x IS NOT NULL AND pos_y IS NOT NULL AND elapsed_time IS NOT NULL"
def window_feats(P, t, hp, bdm, shift):
    bs, be, bp = bdm.s.values, bdm.e.values, bdm.p.values; tt = t + shift; i = np.searchsorted(bs, tt, side='right') - 1; ic = np.clip(i, 0, None); ok = (i >= 0) & (tt <= be[ic]); ph = np.where(ok, bp[ic], -1)
    D = pd.DataFrame({'team': P.team.values[ok], 'acc': P.acc.values[ok], 't': t[ok], 'ph': ph[ok], 'hp': hp[ok]})
    if not len(D): return pd.DataFrame(columns=['team_id', 'ph', 'tl', 'nh', 'na', 'ns'])
    nh = D[D.hp].groupby(['team', 'ph', 't']).acc.nunique().rename('nh'); na = D.groupby(['team', 'ph', 't']).acc.nunique().rename('na')
    sn = pd.concat([na, nh], axis=1).fillna(0).reset_index().sort_values('t'); last = sn.groupby(['team', 'ph']).tail(1).set_index(['team', 'ph'])[['t', 'nh', 'na']].rename(columns={'t': 'tl'}); last['ns'] = sn.groupby(['team', 'ph']).size()
    return last.reset_index().rename(columns={'team': 'team_id'})
def tier2_match(r_m, P, bdm, S, mem):
    P = P.sort_values('t', kind='mergesort').reset_index(drop=True); dpos = float(np.median(P.ep.values - P.t.values) - S); t = P.t.values.astype(float); hp = (P.health.fillna(0).values > 0)
    A = window_feats(P, t, hp, bdm, 0.0); F = window_feats(P, t, hp, bdm, dpos); BF = []; mem = mem[['player_id', 'team_id', 'd_evt']]
    for p, e_p in zip(bdm.p.values, bdm.e.values):
        b_el = e_p - dpos; i1 = np.searchsorted(t, b_el, side='right'); sub = P.iloc[:i1]; last = sub.drop_duplicates('acc', keep='last')[['acc', 't', 'health']]
        x = mem[mem.d_evt > e_p].merge(last, left_on='player_id', right_on='acc', how='left'); fresh = x.t.notna() & ((b_el - x.t) <= 30); h0 = x.health.fillna(0) <= 0
        x['healthy'] = fresh & ~h0; x['knock'] = fresh & h0; x['unsamp'] = ~fresh; ah = set(P.acc.values[i1:][hp[i1:]]); x['revive'] = x.knock & x.player_id.isin(ah)
        g = x.groupby('team_id').agg(n_alive_b=('player_id', 'size'), n_healthy_b=('healthy', 'sum'), n_knock_b=('knock', 'sum'), n_unsamp_b=('unsamp', 'sum'), n_revive_b=('revive', 'sum'))
        w30 = P.iloc[np.searchsorted(t, b_el - 30, side='left'):i1]; w15 = P.iloc[np.searchsorted(t, b_el - 15, side='left'):i1]
        g = g.join(w30.groupby('team').size().rename('rows30')).join(w15.groupby('team').size().rename('rows15')); g['p'] = p; BF.append(g.reset_index())
    BF = pd.concat(BF) if BF else pd.DataFrame(columns=['team_id', 'p'])
    x = r_m.copy()
    for tag, key in (('k', 'k'), ('n', 'nk')):
        x = x.merge(A.rename(columns={'ph': key, 'tl': f'tl_{tag}', 'nh': f'nh_{tag}', 'na': f'na_{tag}', 'ns': f'ns_{tag}'}), on=['team_id', key], how='left')
        x = x.merge(F.rename(columns={'ph': key, 'tl': f'tlfx_{tag}', 'nh': f'nhfx_{tag}', 'na': f'nafx_{tag}', 'ns': f'nsfx_{tag}'}), on=['team_id', key], how='left')
        x = x.merge(BF.rename(columns={'p': key, **{c: f'{c}_{tag}' for c in BF.columns if c not in ('team_id', 'p')}}), on=['team_id', key], how='left')
    x = x.merge((P.groupby('team').t.max() + dpos).rename('lp_st').reset_index().rename(columns={'team': 'team_id'}), on='team_id', how='left'); x['dpos'] = dpos; return x
OUTR = []
for i in range(0, len(ids), a.chunk):
    b = ids[i:i + a.chunk]; r, tt, mt, pl = build_rows(cur, b, id2coh); P = q(cur, POS, (b,)); bd = q(cur, "SELECT match_id, phase_idx AS p, phase_start_s::float8 AS s, phase_end_s::float8 AS e FROM stdb_bak_20261005.match_phase_global WHERE match_id = ANY(%s) ORDER BY match_id, phase_idx", (b,))
    Pg = dict(tuple(P.groupby('match_id'))); Bg = dict(tuple(bd.groupby('match_id'))); Rg = dict(tuple(r.groupby('match_id'))); Mg = dict(tuple(pl.groupby('match_id'))); Sg = mt.set_index('match_id').S
    for mid in b:
        if mid in Pg and mid in Rg and mid in Bg: OUTR.append(tier2_match(Rg[mid], Pg[mid], Bg[mid], float(Sg[mid]), Mg[mid]))
    print(f'[{time.time()-t0:6.1f}s] {min(i + a.chunk, len(ids))}/{len(ids)} rows={sum(len(x) for x in OUTR):,}', flush=True)
R2 = pd.concat(OUTR, ignore_index=True)
for tag in ('', 'fx'):
    pre = {'': 'nh', 'fx': 'nhfx'}[tag]; td = R2[f'{pre}_k'].fillna(0) > 0; hz = td & ~(R2[f'{pre}_n'].fillna(0) > 0); R2['pst_re' if tag == '' else 'pst_fx'] = np.where(~td, 0, np.where(hz, 2, 1))
R2['typ'], R2['cat'] = categorize(R2); R2.loc[R2.pst == R2.tr_evt, ['typ', 'cat']] = ['agree', 'agree']
ctrl = R2[R2.typ == 'agree'].sample(frac=0.03, random_state=3); keep = pd.concat([R2[R2.typ != 'agree'], ctrl]); keep.to_csv(f'{OUT}/label_audit_{a.tag}_rows.csv.gz', index=False, compression='gzip')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
hu = R2[(R2.n_ai == 0) & np.isfinite(R2.T_evt) & np.isfinite(R2.T_off) & ((R2.T_evt - R2.T_off).abs() > 5)].drop_duplicates(['match_id', 'team_id']); hu = hu.assign(evt_ok=hu.T_evt >= hu.lp_st - 2, off_ok=hu.T_off >= hu.lp_st - 2)
print('ADJUDICATION human teams with |T_evt-T_off|>5 s:', len(hu), '| T_evt >= last position sample:', int(hu.evt_ok.sum()), '| T_off >= last position sample:', int(hu.off_ok.sum()), '| both:', int((hu.evt_ok & hu.off_ok).sum()), '| neither:', int((~hu.evt_ok & ~hu.off_ok).sum()), '| redeploy teams among them:', int(hu.redeploy.sum()))
allt = R2[np.isfinite(R2.T_evt)].drop_duplicates(['match_id', 'team_id']); print('ALL teams with finite T_evt:', len(allt), '| T_evt < last position sample - 2s:', int((allt.T_evt < allt.lp_st - 2).sum()), '| of which redeploy:', int(((allt.T_evt < allt.lp_st - 2) & allt.redeploy).sum()), '| T_off < last sample - 2s (human teams):', int(((allt.T_off < allt.lp_st - 2) & (allt.n_ai == 0)).sum()))
print('rows', len(R2), 'matches', R2.match_id.nunique(), 'dpos q05/50/95/max:', R2.drop_duplicates('match_id').dpos.quantile([.05, .5, .95, 1]).round(2).to_dict(), 'share dpos>10:', round(float((R2.drop_duplicates('match_id').dpos > 10).mean()), 3))
print('replication: pst_re == table pst:', round(float((R2.pst_re == R2.pst).mean()), 4), '| nh_k == table ape_k (rows with state):', round(float((R2.nh_k.fillna(0) == R2.ape_k.fillna(0)).mean()), 4))
for nm in ('pst', 'pst_re', 'pst_fx'): print(nm, 'vs truth_events agree:', round(float((R2[nm] == R2.tr_evt).mean()), 4), '| risk-set FP/FN:', int(((R2[nm] >= 1) & (R2.tr_evt == 0)).sum()), int(((R2[nm] == 0) & (R2.tr_evt >= 1)).sum()), '| label FP/FN:', int(((R2[nm] == 2) & (R2.tr_evt == 1)).sum()), int(((R2[nm] == 1) & (R2.tr_evt == 2)).sum()))
bk = R2.typ.isin(['risk_FN', 'risk_FP']).values
for c_ in ['n_alive_b', 'n_healthy_b', 'n_knock_b', 'n_unsamp_b', 'n_revive_b', 'rows30', 'rows15']: R2[c_ + '_rel'] = np.where(bk, R2[c_ + '_k'], R2[c_ + '_n'])
kn = R2[R2.cat == 'knocked_last_snapshot']; print('KNOCKED-snapshot rows', len(kn), '| teammate healthy+sampled at boundary (n_healthy>=1):', int((kn.n_healthy_b_rel.fillna(0) >= 1).sum()), '| all alive members knocked (healthy==0 & knock>=1):', int(((kn.n_healthy_b_rel.fillna(0) == 0) & (kn.n_knock_b_rel.fillna(0) >= 1)).sum()), '| none sampled:', int(((kn.n_healthy_b_rel.fillna(0) == 0) & (kn.n_knock_b_rel.fillna(0) == 0)).sum()), '| revive evidence (knocked member later healthy):', int((kn.n_revive_b_rel.fillna(0) > 0).sum()))
un = R2[R2.cat == 'unobserved_no_rows']; print('UNOBSERVED rows', len(un), '| team rows within 30 s before physical boundary == 0:', int((un.rows30_rel.fillna(0) == 0).sum()), '| within 15 s == 0:', int((un.rows15_rel.fillna(0) == 0).sum()))
d = R2[R2.typ != 'agree']; print('disagreeing rows with NO team position rows in the 30 s before the physical boundary:', int((d.rows30_rel.fillna(0) == 0).sum()), 'of', len(d), '| agreeing rows:', round(float((R2[R2.typ == 'agree'].rows30_rel.fillna(0) == 0).mean()), 4), '| disagree share:', round(float((d.rows30_rel.fillna(0) == 0).mean()), 4))
print('events-disagreeing rows', len(d), '| fixed by clock-fix:', int((d.pst_fx == d.tr_evt).sum())); print(d.groupby(['typ', 'cat']).apply(lambda g: pd.Series(dict(n=len(g), fixed_by_clock=int((g.pst_fx == g.tr_evt).sum())))).to_string())
