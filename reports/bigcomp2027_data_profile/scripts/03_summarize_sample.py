#!/usr/bin/env python3
"""Aggregate the player-level sample (csv/<tag>_*.csv from 02_sample_player_level.py) into per-cohort stage-2 estimates."""
import json, os, sys
import numpy as np, pandas as pd
ROOT = '/Users/lunecid/project/PUBG'; C = ROOT + '/reports/bigcomp2027_data_profile/csv'
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 80); pd.set_option('display.max_rows', 400)
tag = sys.argv[1] if len(sys.argv) > 1 else 'sample'
rd = lambda n: pd.read_csv(f'{C}/{tag}_{n}.csv')
m, c, e, a, h, tmg = rd('match_level'), rd('cut_level'), rd('events_by_phase'), rd('label_audit'), rd('position_interval_hist'), rd('batch_timing')
meta = json.load(open(f'{C}/{tag}_run_meta.json'))
cs = pd.read_csv(f'{C}/cohort_summary.csv').set_index('slug').sort_values('n_total', ascending=False); order = list(cs.index); NT = cs.n_total.to_dict()
print('RUN META', meta); print('matches ok per cohort:', m.groupby('cohort').size().loc[order].to_dict())
# ---- (1) cut-level means ----
cm = ['nodes_A', 'teams_A', 'nodes_C', 'teams_C', 'tm_A', 'tm_C', 'obs_nodes', 'fresh_nodes', 'tm_fresh', 'enemy_all', 'enemy_fresh', 'enemy_fresh_C', 'bin_lt50', 'bin_50_100', 'bin_100_200', 'bin_200_300',
      'nodes_with_enemy_fresh', 'cb_enemy', 'cb_friendly', 'cb_unknown', 'cb_enemy_inA', 'cb_enemy_obs', 'cb_enemy_within300', 'cb_events_enemy', 'cb_dmg_enemy', 'C_players', 'C_miss15', 'C_miss30', 'C_teams',
      'C_team_miss15', 'C_team_miss30', 'A_nopos', 'fresh_C_n', 'fresh_C_h0', 'stale_med', 'stale_p90', 'phase_dur']
g = c.groupby(['cohort', 'k']); ck = g[cm].mean().reset_index(); ck.insert(2, 'n_matches', g.size().values); ck.to_csv(f'{C}/stage2_cut_means_by_cohort_phase.csv', index=False)
pk = c.groupby('k')[cm].mean().round(2); pk.insert(0, 'n', c.groupby('k').size())
print('\nALL-COHORT mean per cut k (sample):'); print(pk[['n', 'nodes_A', 'teams_A', 'nodes_C', 'tm_A', 'tm_C', 'obs_nodes', 'fresh_nodes', 'enemy_all', 'enemy_fresh', 'enemy_fresh_C', 'cb_enemy', 'cb_enemy_obs', 'cb_enemy_within300', 'cb_events_enemy']].to_string())
print('\nEnemy proximity (alive-at-cut variant) mean per cut, cohort x k:'); print(ck.pivot(index='cohort', columns='k', values='enemy_fresh_C').loc[order].round(1).to_string())
print('\nnodes_A mean per cut, cohort x k:'); print(ck.pivot(index='cohort', columns='k', values='nodes_A').loc[order].round(1).to_string())
print('\ncb_enemy mean per cut, cohort x k:'); print(ck.pivot(index='cohort', columns='k', values='cb_enemy').loc[order].round(1).to_string())
# ---- (2) per-match totals ----
met = ['nodes_A', 'teams_A', 'nodes_C', 'tm_A', 'tm_C', 'enemy_all', 'enemy_fresh', 'enemy_fresh_C', 'cb_enemy', 'cb_enemy_obs', 'cb_events_enemy', 'nodes_with_enemy_fresh']; rows = []
for scope, df in [('cuts_with_target', c[c.has_next == 1]), ('all_k1to7', c)]:
    t = df.groupby(['cohort', 'match_id'])[met].sum().reset_index()
    for co in order:
        gg = t[t.cohort == co]; r = dict(scope=scope, cohort=co, n_sample=len(gg), n_total=NT[co])
        for mm in met: r.update({mm + '_mean': gg[mm].mean(), mm + '_med': gg[mm].median(), mm + '_p10': gg[mm].quantile(.1), mm + '_p90': gg[mm].quantile(.9)})
        rows.append(r)
pm = pd.DataFrame(rows); pm.to_csv(f'{C}/stage2_per_match_totals_by_cohort.csv', index=False)
print('\nPER-MATCH TOTALS over cuts with target (k<=7 and phase k+1 exists):'); print(pm[pm.scope == 'cuts_with_target'][['cohort', 'n_sample', 'nodes_A_mean', 'nodes_A_med', 'nodes_A_p10', 'nodes_A_p90', 'teams_A_mean', 'tm_A_mean', 'tm_C_mean', 'enemy_fresh_C_mean', 'enemy_fresh_mean', 'enemy_all_mean', 'cb_enemy_mean', 'cb_events_enemy_mean']].round(1).to_string(index=False))
print('\nPER-MATCH TOTALS over all k=1..7 present:'); print(pm[pm.scope == 'all_k1to7'][['cohort', 'nodes_A_mean', 'tm_A_mean', 'enemy_fresh_C_mean', 'enemy_all_mean', 'cb_enemy_mean']].round(1).to_string(index=False))
# ---- (3) tensor size model (ragged layout, float32) ----
D_P, D_T, B_EDGE, B_ZONE = 12, 150, 17, 21
def tens(r, col):
    N, T = r['nodes_A_mean'], r['teams_A_mean']; E = 2 * r['tm_A_mean'] + 2 * r[col] + r['cb_enemy_mean']
    node, team, bc_, edge, zone = N * D_P * 4, T * D_T * 4, N * D_T * 4, E * B_EDGE, N * 2 * B_ZONE
    return pd.Series(dict(edges_directed=E, MB_nodes=node / 1e6, MB_team_once=team / 1e6, MB_team_broadcast=bc_ / 1e6, MB_edges=edge / 1e6, MB_zone=zone / 1e6, MB_match_compact=(node + team + edge + zone) / 1e6, MB_match_broadcast=(node + bc_ + edge + zone) / 1e6))
tz = []
for scope in ['cuts_with_target', 'all_k1to7']:
    d = pm[pm.scope == scope].copy()
    for variant, col in [('enemy_alive_at_cut', 'enemy_fresh_C_mean'), ('enemy_all_nodes_upper', 'enemy_all_mean')]:
        x = pd.concat([d[['scope', 'cohort', 'n_total', 'nodes_A_mean', 'teams_A_mean', 'tm_A_mean']].reset_index(drop=True), d.apply(lambda r: tens(r, col), axis=1).reset_index(drop=True)], axis=1); x['variant'] = variant
        x['GB_cohort_compact'] = x.MB_match_compact * x.n_total / 1e3; x['GB_cohort_broadcast'] = x.MB_match_broadcast * x.n_total / 1e3; tz.append(x)
tz = pd.concat(tz); tz.to_csv(f'{C}/stage2_tensor_size_estimate.csv', index=False)
print('\nTENSOR SIZE (MB/match, GB/cohort), scope=cuts_with_target:'); print(tz[tz.scope == 'cuts_with_target'][['variant', 'cohort', 'n_total', 'nodes_A_mean', 'edges_directed', 'MB_nodes', 'MB_team_once', 'MB_team_broadcast', 'MB_edges', 'MB_zone', 'MB_match_compact', 'MB_match_broadcast', 'GB_cohort_compact', 'GB_cohort_broadcast']].round(3).to_string(index=False))
print(tz.groupby(['scope', 'variant'])[['GB_cohort_compact', 'GB_cohort_broadcast']].sum().round(2).to_string())
# ---- (4) events (raw telemetry) ----
ec = ['kills_all', 'kills_enemy', 'kills_nokiller', 'groggy_all', 'groggy_enemy', 'dmg_ev_all', 'dmg_ev_enemy', 'dmg_ev_noattacker', 'dmg_sum_all', 'dmg_sum_enemy']
eg = e.groupby(['cohort', 'k']); ep = eg[ec].mean().reset_index(); ep.insert(2, 'n_matches', eg.size().values); ep.to_csv(f'{C}/sample_events_per_phase_by_cohort.csv', index=False)
et = e.groupby(['cohort', 'match_id'])[ec].sum().reset_index(); rows = []
for co in order:
    gg = et[et.cohort == co]; r = dict(cohort=co, n=len(gg))
    for x in ec: r.update({x + '_mean': gg[x].mean(), x + '_med': gg[x].median(), x + '_p10': gg[x].quantile(.1), x + '_p90': gg[x].quantile(.9)})
    rows.append(r)
evm = pd.DataFrame(rows); evm.to_csv(f'{C}/sample_events_per_match_by_cohort.csv', index=False)
print('\nRAW EVENTS per match (sample): kills / groggy / dmg events / dmg sum'); print(evm[['cohort', 'n', 'kills_all_mean', 'kills_enemy_mean', 'groggy_all_mean', 'groggy_enemy_mean', 'dmg_ev_all_mean', 'dmg_ev_all_med', 'dmg_ev_all_p10', 'dmg_ev_all_p90', 'dmg_ev_enemy_mean', 'dmg_ev_noattacker_mean', 'dmg_sum_all_mean', 'dmg_sum_enemy_mean']].round(1).to_string(index=False))
pp = e.groupby('k')[ec].mean().round(1); pp.insert(0, 'n', e.groupby('k').size()); print('\nRAW EVENTS per phase (all cohorts pooled mean per match; k=0 is before phase 1):'); print(pp[['n', 'kills_all', 'kills_enemy', 'groggy_all', 'groggy_enemy', 'dmg_ev_all', 'dmg_ev_enemy', 'dmg_ev_noattacker', 'dmg_sum_all', 'dmg_sum_enemy']].to_string())
# ---- (5) position sampling interval ----
rows = []
for co in order + ['ALL']:
    hh = h if co == 'ALL' else h[h.cohort == co]; hv = hh.groupby('interval_s').n.sum().reindex(range(121), fill_value=0).values.astype(float); tot = hv.sum(); cdf = hv.cumsum() / tot
    qf = lambda p: int(np.searchsorted(cdf, p)); rows.append(dict(cohort=co, n_intervals=int(tot), mean_s=float((np.arange(121) * hv).sum() / tot), median_s=qf(.5), p10_s=qf(.1), p90_s=qf(.9), p99_s=qf(.99),
        share_le11=float(hv[:12].sum() / tot), share_eq10=float(hv[10] / tot), share_gt20=float(hv[21:].sum() / tot), share_gt60=float(hv[61:].sum() / tot)))
iv = pd.DataFrame(rows); iv.to_csv(f'{C}/position_interval_stats_by_cohort.csv', index=False); print('\nPOSITION SAMPLING INTERVAL (s):'); print(iv.round(4).to_string(index=False))
hall = h.groupby('interval_s').n.sum(); print('top interval values (all):', (hall / hall.sum()).sort_values(ascending=False).head(8).round(4).to_dict())
# ---- (6) missing positions near cuts ----
mk = c.groupby(['cohort', 'k'])[['C_players', 'C_miss15', 'C_miss30', 'C_teams', 'C_team_miss15', 'C_team_miss30', 'A_nopos', 'nodes_A']].sum().reset_index()
for w in (15, 30): mk[f'player_miss{w}_share'] = mk[f'C_miss{w}'] / mk.C_players; mk[f'team_miss{w}_share'] = mk[f'C_team_miss{w}'] / mk.C_teams
mk['A_nopos_share'] = mk.A_nopos / mk.nodes_A; mk.to_csv(f'{C}/missing_positions_near_cut_by_cohort_phase.csv', index=False)
ov = c.groupby('k')[['C_players', 'C_miss15', 'C_miss30', 'C_teams', 'C_team_miss15', 'C_team_miss30', 'A_nopos', 'nodes_A']].sum()
for w in (15, 30): ov[f'player_miss{w}'] = ov[f'C_miss{w}'] / ov.C_players; ov[f'team_miss{w}'] = ov[f'C_team_miss{w}'] / ov.C_teams
ov['A_nopos_share'] = ov.A_nopos / ov.nodes_A; print('\nMISSING POSITIONS near the cut (alive at cut; window 15 s / 30 s), all cohorts by k:'); print(ov[['C_players', 'player_miss15', 'player_miss30', 'C_teams', 'team_miss15', 'team_miss30', 'A_nopos_share']].round(4).to_string())
tot = c[['C_players', 'C_miss15', 'C_miss30', 'C_teams', 'C_team_miss15', 'C_team_miss30', 'A_nopos', 'nodes_A']].sum(); print('TOTAL player miss15/30', round(tot.C_miss15 / tot.C_players, 4), round(tot.C_miss30 / tot.C_players, 4), '| team miss15/30', round(tot.C_team_miss15 / tot.C_teams, 4), round(tot.C_team_miss30 / tot.C_teams, 4), '| A_nopos share', round(tot.A_nopos / tot.nodes_A, 5))
print('alive-at-cut with health==0 (knocked?) share:', round(c.fresh_C_h0.sum() / c.fresh_C_n.sum(), 4), '| stale_med mean', round(c.stale_med.mean(), 2), 'stale_p90 mean', round(c.stale_p90.mean(), 2))
# ---- (7) label audit ----
sc = [x for x in a.columns if x.startswith('n_')]; ag = a.groupby(['cohort', 'k'])[sc].sum().reset_index(); ao = a.groupby('k')[sc].sum(); ao.loc['ALL k1-7'] = ao.sum()
def ratios(d):
    d = d.copy(); d['risk_agree_of_label'] = d.n_risk_both / d.n_label_risk; d['risk_agree_of_true'] = d.n_risk_both / d.n_true_risk; d['pos_precision'] = d.n_pos_both / d.n_label_pos; d['pos_recall'] = d.n_pos_both / d.n_true_pos
    d['labelpos_dead_before_cut_share'] = d.n_labelpos_dead_before_cut / d.n_label_pos; d['labelpos_alive_after_next_share'] = d.n_labelpos_alive_after_next / d.n_label_pos
    d['hazard_label'] = d.n_label_pos / d.n_label_risk; d['hazard_true'] = d.n_true_pos / d.n_true_risk; return d
ag = ratios(ag); ag.to_csv(f'{C}/label_audit_by_cohort_phase.csv', index=False); ao = ratios(ao); ao.to_csv(f'{C}/label_audit_overall_by_phase.csv')
print('\nLABEL AUDIT vs ground truth from kills/participants (sample, all cohorts):'); print(ao[['n_label_risk', 'n_true_risk', 'n_risk_both', 'n_label_pos', 'n_true_pos', 'n_pos_both', 'pos_precision', 'pos_recall', 'risk_agree_of_label', 'labelpos_dead_before_cut_share', 'labelpos_alive_after_next_share', 'hazard_label', 'hazard_true']].round(4).to_string())
# ---- (8) bots / match type ----
m['n_human'] = m.n_players - m.n_ai
bt = m.groupby(['cohort', 'match_type']).agg(n=('match_id', 'size'), players_mean=('n_players', 'mean'), ai_mean=('n_ai', 'mean'), ai_med=('n_ai', 'median'), ai_p90=('n_ai', lambda x: x.quantile(.9)), human_mean=('n_human', 'mean'), share_matches_with_ai=('n_ai', lambda x: (x > 0).mean()), ai_sum=('n_ai', 'sum'), pl_sum=('n_players', 'sum')).reset_index()
bt['ai_share_players'] = bt.ai_sum / bt.pl_sum; bt.to_csv(f'{C}/bots_by_cohort_matchtype.csv', index=False); print('\nBOTS (account_id starts with "ai.") by cohort x match_type (sample):'); print(bt.drop(columns=['ai_sum', 'pl_sum']).round(3).to_string(index=False))
bc_ = m.groupby('cohort').agg(n=('match_id', 'size'), ai_sum=('n_ai', 'sum'), pl_sum=('n_players', 'sum'), human_mean=('n_human', 'mean'), share_matches_with_ai=('n_ai', lambda x: (x > 0).mean())).loc[order]; bc_['ai_share'] = bc_.ai_sum / bc_.pl_sum; bc_.to_csv(f'{C}/bots_by_cohort.csv'); print(bc_.round(3).to_string()); print('ALL: AI share of players', round(m.n_ai.sum() / m.n_players.sum(), 4), '| matches with >=1 AI', round((m.n_ai > 0).mean(), 4), '| human players mean/match', round(m.n_human.mean(), 2))
# ---- (9) raw-table consistency & knock/revive evidence ----
ev_m = e.groupby('match_id')[['dmg_sum_enemy', 'dmg_sum_all']].sum(); mm_ = m.set_index('match_id').join(ev_m)
cons = mm_.groupby('cohort').apply(lambda d: pd.Series(dict(kill_rows_per_p_kills=d.kill_rows.sum() / max(d.p_kills.sum(), 1), groggy_rows_per_p_dbnos=d.groggy_rows.sum() / max(d.p_dbnos.sum(), 1), enemy_dmg_per_p_damage=d.dmg_sum_enemy.sum() / max(d.p_damage.sum(), 1),
    knock_next_h0=d.knock_next_sample_h0.sum() / max(d.knock_events.sum(), 1), knock_revive_like=d.knock_revive_like.sum() / max(d.knock_events.sum(), 1), revives_per_match=d.p_revives.mean(), heals_per_match=d.p_heals.mean(), boosts_per_match=d.p_boosts.mean(), pos_rows_mean=d.pos_rows.mean(), dmg_rows_mean=d.dmg_rows.mean()))).loc[order]
cons.to_csv(f'{C}/raw_consistency_checks_by_cohort.csv'); print('\nRAW CONSISTENCY / KNOCK-REVIVE EVIDENCE:'); print(cons.round(3).to_string())
print('ALL: kill_rows/p_kills', round(m.kill_rows.sum() / m.p_kills.sum(), 3), '| groggy_rows/p_dbnos', round(m.groggy_rows.sum() / max(m.p_dbnos.sum(), 1), 3), '| knock next-sample health0', round(m.knock_next_sample_h0.sum() / m.knock_events.sum(), 3), '| revive-like', round(m.knock_revive_like.sum() / m.knock_events.sum(), 3), '| p_revives/match', round(m.p_revives.mean(), 2), '| pos_health0_share mean', round(m.pos_health0_share.mean(), 4), '| health NaN share', round(m.pos_health_nan_share.mean(), 5))
print('clock: off_minus_started median/p10/p90/max', m.off_minus_started.quantile([.5, .1, .9, 1]).round(2).to_dict(), '| team_mismatch_share max', m.team_mismatch_share.max(), '| ts_diff_mad median', m.ts_diff_mad.median().round(2))
# ---- (10) extraction throughput ----
nm = tmg.n.sum(); tabs = [x for x in ['meta', 'phase', 'part', 'pos', 'kill', 'grg', 'dmg', 'lab'] if x in tmg.columns]
per = {x: round(tmg[x].sum() / nm, 4) for x in tabs}; fetch_pm = tmg.fetch_total.sum() / nm; proc_pm = tmg.proc.sum() / nm
print('\nTHROUGHPUT: matches', int(nm), '| wall', round(meta['wall_s'], 1), 's with', meta['args']['workers'], 'workers | fetch s/match', round(fetch_pm, 4), '| process s/match', round(proc_pm, 4), '| per table s/match', per)
tot_n = int(cs.n_total.sum()); print('EXTRAPOLATION to', tot_n, 'matches: serial', round(tot_n * (fetch_pm + proc_pm) / 60, 1), 'min; with 2 workers ~', round(tot_n * (fetch_pm + proc_pm) / 120, 1), 'min (measured wall/match x N =', round(tot_n * meta['wall_s'] / nm / 60, 1), 'min)')
# ---- (11) items, coordinates ----
ip = f'{C}/{tag}_item_use_taxonomy.csv'
if os.path.exists(ip):
    it = pd.read_csv(ip); print('\nITEM USE categories (sample of first batch per cohort):'); print(it.groupby(['category', 'sub_category'], dropna=False).n.sum().sort_values(ascending=False).head(14).to_string()); print(it.head(14).to_string(index=False))
print('\nCOORD range (cm) by map:'); print(m.groupby('map_name')[['x_min', 'x_max', 'y_min', 'y_max']].agg(['min', 'max']).round(0).to_string())
print('players/match (sample) mean', round(m.n_players.mean(), 2), '| pos rows/match mean/med/p90', round(m.pos_rows.mean()), m.pos_rows.median(), m.pos_rows.quantile(.9), '| dmg rows/match', round(m.dmg_rows.mean()), '| kill rows', round(m.kill_rows.mean(), 1), '| groggy rows', round(m.groggy_rows.mean(), 1))
