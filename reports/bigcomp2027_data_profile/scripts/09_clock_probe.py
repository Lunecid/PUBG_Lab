#!/usr/bin/env python3
"""Label audit step 1: which clocks do phase bounds, telemetry elapsed_time, event_time and participants.time_survived use?  READ-ONLY.
Bounds are read from stdb_bak_20261005.match_phase_global (copy made before the Fix B rewrite)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import dbh
C = '/Users/lunecid/project/PUBG/reports/bigcomp2027_data_profile/csv'; pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
m = pd.read_csv(f'{C}/sample_match_level.csv'); rng = np.random.RandomState(11); ids = []
for co, g in m.groupby('cohort'): ids += rng.choice(g.match_id.values, 30, replace=False).tolist()
cur = dbh.conn().cursor()
def q(sql, p): cur.execute(sql, p); return pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
PH, PL, MM = [], [], []
for i in range(0, len(ids), 30):
    b = ids[i:i + 30]
    S = q("SELECT match_id, extract(epoch from event_time)::float8 AS s_ep FROM pubg.telem_match_start WHERE match_id = ANY(%s)", (b,)).drop_duplicates('match_id').set_index('match_id').s_ep
    ogs = q("SELECT match_id, percentile_cont(0.5) WITHIN GROUP (ORDER BY extract(epoch from event_time) - elapsed_time) AS o_gs FROM pubg.telem_game_states WHERE match_id = ANY(%s) AND elapsed_time IS NOT NULL GROUP BY match_id", (b,)).set_index('match_id').o_gs
    opos = q("SELECT match_id, percentile_cont(0.5) WITHIN GROUP (ORDER BY extract(epoch from event_time) - elapsed_time) AS o_pos FROM pubg.telem_positions WHERE match_id = ANY(%s) AND is_game >= 0.1 AND elapsed_time IS NOT NULL GROUP BY match_id", (b,)).set_index('match_id').o_pos
    pc = q("SELECT match_id, floor(phase)::int AS p, min(extract(epoch from event_time))::float8 AS ep FROM pubg.telem_phase_changes WHERE match_id = ANY(%s) GROUP BY 1,2", (b,))
    gsf = q("SELECT match_id, floor(is_game)::int AS p, min(elapsed_time) AS gs_first_el FROM pubg.telem_game_states WHERE match_id = ANY(%s) AND elapsed_time IS NOT NULL GROUP BY 1,2", (b,))
    bd = q("SELECT match_id, phase_idx AS p, phase_start_s::float8 AS s, phase_end_s::float8 AS e FROM stdb_bak_20261005.match_phase_global WHERE match_id = ANY(%s)", (b,))
    part = q("SELECT p.match_id, p.player_id, r.team_id, p.death_type, p.time_survived::float8 AS ts FROM pubg.participants p JOIN pubg.rosters r ON r.roster_id = p.roster_id AND r.match_id = p.match_id WHERE p.match_id = ANY(%s)", (b,))
    kil = q("SELECT match_id, victim_id, min(extract(epoch from event_time))::float8 AS ep FROM pubg.telem_kills WHERE match_id = ANY(%s) GROUP BY 1,2", (b,))
    grg = q("SELECT match_id, victim_id, min(extract(epoch from event_time))::float8 AS gep FROM pubg.telem_groggy WHERE match_id = ANY(%s) GROUP BY 1,2", (b,))
    mm = pd.DataFrame({'s_ep': S, 'o_gs': ogs, 'o_pos': opos}).dropna(); mm['delta'] = mm.o_gs - mm.s_ep; mm['pos_minus_gs'] = mm.o_pos - mm.o_gs; MM.append(mm.reset_index())
    x = pc.merge(bd, on=['match_id', 'p']).merge(gsf, on=['match_id', 'p'], how='left').join(mm[['s_ep', 'o_gs']], on='match_id', how='inner')
    x['b_el'] = x.ep - x.o_gs; x['b_st'] = x.ep - x.s_ep; x['d_el'] = x.s - x.b_el; x['d_st'] = x.s - x.b_st; x['g_minus_b_el'] = x.gs_first_el - x.b_el; PH.append(x)
    y = part.merge(kil, left_on=['match_id', 'player_id'], right_on=['match_id', 'victim_id'], how='left').merge(grg, left_on=['match_id', 'player_id'], right_on=['match_id', 'victim_id'], how='left').join(mm[['s_ep', 'o_gs', 'delta']], on='match_id', how='inner')
    y['diff_el'] = y.ep - y.o_gs - y.ts; y['diff_st'] = y.ep - y.s_ep - y.ts; y['knocked_first'] = y.gep.notna() & (y.gep < y.ep); y['is_ai'] = y.player_id.fillna('').str.startswith('ai.')
    y['team_n'] = y.groupby(['match_id', 'team_id']).player_id.transform('count'); PL.append(y)
MM = pd.concat(MM); PH = pd.concat(PH); PL = pd.concat(PL); PL.to_csv(f'{C}/label_audit_clock_probe_players.csv', index=False); PH.to_csv(f'{C}/label_audit_clock_probe_phases.csv', index=False); MM.to_csv(f'{C}/label_audit_clock_probe_matches.csv', index=False)
qs = lambda s, nm: print(f'{nm:52s} n={len(s.dropna()):6d}', s.quantile([0, .05, .25, .5, .75, .95, 1]).round(2).to_dict())
print('matches probed', len(MM))
qs(MM.delta, 'delta = O_gs - S  (elapsed zero minus LogMatchStart, s)'); qs(MM.pos_minus_gs, 'O_pos - O_gs (positions vs game_states clock)')
qs(PH.d_st, 'phase_start_s - (phase change epoch - S)'); qs(PH.d_el, 'phase_start_s - (phase change epoch - O_gs)'); qs(PH.g_minus_b_el, 'first game_state with floor(is_game)=p minus B_el')
print('phase_start_s / phase_end_s multiples of 10:', float((PH.s % 10 == 0).mean()), float((PH.e % 10 == 0).mean()))
print('corr(d_el, delta):', round(PH[['d_el']].assign(dl=PH.o_gs - PH.s_ep).corr().iloc[0, 1], 3), '| median d_el - delta:', round((PH.d_el - (PH.o_gs - PH.s_ep)).median(), 2), '| q05/q95:', (PH.d_el - (PH.o_gs - PH.s_ep)).quantile([.05, .95]).round(1).to_dict())
k = PL[PL.ep.notna()]
qs(k.diff_el, 'player: kill_epoch - O_gs - time_survived (elapsed clock)'); qs(k.diff_st, 'player: kill_epoch - S - time_survived (started clock)')
for nm, msk in [('knocked before death', k.knocked_first), ('no knock before death', ~k.knocked_first), ('AI', k.is_ai), ('human', ~k.is_ai), ('solo (team_n=1)', k.team_n == 1), ('squad (team_n>=3)', k.team_n >= 3)]: qs(k[msk].diff_st, f'  started-clock diff | {nm}')
pm = k.groupby('match_id').agg(med_st=('diff_st', 'median'), mad_st=('diff_st', lambda x: (x - x.median()).abs().median()), med_el=('diff_el', 'median'), n=('diff_st', 'size')).join(MM.set_index('match_id').delta)
qs(pm.med_st, 'per-match median started-clock diff'); qs(pm.mad_st, 'per-match MAD of started-clock diff'); qs(pm.med_el, 'per-match median elapsed-clock diff'); print('corr(per-match med_el, -delta):', round(pm[['med_el']].assign(nd=-pm.delta).corr().iloc[0, 1], 3))
print('death_type counts:', PL.death_type.value_counts().to_dict(), '| logout players with a kill event:', int(PL[(PL.death_type == 'logout')].ep.notna().sum()), 'of', int((PL.death_type == 'logout').sum()), '| alive players with kill event:', int(PL[PL.death_type == 'alive'].ep.notna().sum()))
print('byplayer/byzone players WITHOUT kill event:', int(((PL.death_type.isin(['byplayer', 'byzone', 'suicide'])) & PL.ep.isna()).sum()), 'of', int(PL.death_type.isin(['byplayer', 'byzone', 'suicide']).sum()))
print('time_survived of alive players vs match duration (first 5):', PL[PL.death_type == 'alive'].groupby('match_id').ts.median().head(5).round(1).tolist())
