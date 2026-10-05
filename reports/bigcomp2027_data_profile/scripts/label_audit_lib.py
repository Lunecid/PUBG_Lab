"""Shared code for the hazard-label audit (Part 1b).  READ-ONLY.  Truth is built in the started clock (LogMatchStart = stdb.matches.started_at),
which is the numeric clock of phase_start_s/phase_end_s (see 09_clock_probe.py) and of participants.time_survived."""
import json, os
import numpy as np, pandas as pd
INF = np.inf; FALLBACK_OFF = 0.70   # median (kill epoch - S - time_survived) for human players in the 390-match probe
ROOT = '/Users/lunecid/project/PUBG'; R = ROOT + '/model_stgcn/checkpoints/br_om_ladder_required_v1'
def cohort_map():
    m = {}
    for d in sorted(os.listdir(R)):
        p = f'{R}/{d}/split.json'
        if os.path.isfile(p):
            for sp in json.load(open(p))['splits'].values():
                for mid in sp: m[mid] = d
    return m
def q(cur, sql, params):
    cur.execute(sql, params); return pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
def build_rows(cur, ids, id2coh):
    ms = q(cur, "SELECT m.match_id, extract(epoch from m.started_at)::float8 AS s_ep, pm.match_type FROM stdb.matches m JOIN pubg.matches pm ON pm.match_id = m.match_id WHERE m.match_id = ANY(%s)", (ids,)).set_index('match_id')
    bd = q(cur, "SELECT match_id, phase_idx AS p, phase_start_s::float8 AS s, phase_end_s::float8 AS e FROM stdb_bak_20261005.match_phase_global WHERE match_id = ANY(%s)", (ids,))
    lab = q(cur, "SELECT match_id, team_id, cut_phase AS k, next_phase_idx AS nk, target_defined AS tdef, hazard_next AS haz, final_rank FROM stdb.team_phase_labels WHERE match_id = ANY(%s) AND cut_phase BETWEEN 1 AND 7", (ids,))
    st = q(cur, "SELECT match_id, team_id, phase_idx AS p, position_rows AS pr, alive_players_end AS ape FROM stdb.team_phase_state WHERE match_id = ANY(%s)", (ids,))
    part = q(cur, "SELECT p.match_id, p.player_id, r.team_id, r.rank AS rrank, r.won, p.death_type, p.time_survived::float8 AS ts FROM pubg.participants p JOIN pubg.rosters r ON r.roster_id = p.roster_id AND r.match_id = p.match_id WHERE p.match_id = ANY(%s)", (ids,))
    kil = q(cur, "SELECT match_id, victim_id, min(extract(epoch from event_time))::float8 AS ep_first, max(extract(epoch from event_time))::float8 AS ep, count(*) AS n_ev FROM pubg.telem_kills WHERE match_id = ANY(%s) GROUP BY 1,2", (ids,))
    gs = q(cur, "SELECT match_id, percentile_cont(0.5) WITHIN GROUP (ORDER BY extract(epoch from event_time) - elapsed_time) AS o_gs FROM pubg.telem_game_states WHERE match_id = ANY(%s) AND elapsed_time IS NOT NULL GROUP BY match_id", (ids,)).set_index('match_id').o_gs
    pl = part.merge(kil, left_on=['match_id', 'player_id'], right_on=['match_id', 'victim_id'], how='left').drop(columns='victim_id')
    pl['S'] = pl.match_id.map(ms.s_ep); pl['d_kill'] = pl.ep - pl.S; pl['multi'] = (pl.n_ev.fillna(1) > 1) & ((pl.ep - pl.ep_first) > 5); pl['is_ai'] = pl.player_id.fillna('').str.startswith('ai.')
    pl['is_alive'] = pl.death_type.eq('alive'); pl['is_logout'] = pl.death_type.eq('logout')
    hm = pl[(~pl.is_ai) & pl.d_kill.notna() & pl.ts.notna()]; dd = hm.d_kill - hm.ts; gb = dd.groupby(hm.match_id)
    off = gb.median(); mad = (dd - gb.transform('median')).abs().groupby(hm.match_id).median(); nhk = gb.size()
    pl['off'] = pl.match_id.map(off).fillna(FALLBACK_OFF); pl['fb'] = (~pl.is_alive) & pl.d_kill.isna()
    pl['d_evt'] = np.where(pl.is_alive, INF, np.where(pl.d_kill.notna(), pl.d_kill, np.where(pl.ts.notna(), pl.ts + pl.off, INF)))
    pl['d_off'] = np.where(pl.is_alive, INF, pl.ts + pl.off); pl['won_b'] = pl.won.fillna(False).astype(bool)
    tt = pl.groupby(['match_id', 'team_id']).agg(T_evt=('d_evt', 'max'), T_off_raw=('d_off', 'max'), n_mem=('player_id', 'size'), n_ai=('is_ai', 'sum'), n_alive=('is_alive', 'sum'), n_logout=('is_logout', 'sum'), n_fb=('fb', 'sum'), n_multi=('multi', 'sum'), won=('won_b', 'max'), rrank=('rrank', 'min')).reset_index()
    tt['T_off'] = np.where(tt.won.astype(bool) | (tt.rrank == 1), INF, tt.T_off_raw); tt = tt.drop(columns=['T_off_raw'])
    r = lab[lab.nk.notna()].copy(); r['nk'] = r.nk.astype(int)
    r = r.merge(bd.rename(columns={'p': 'k', 's': 's_k', 'e': 'e_k'}), on=['match_id', 'k'], how='left').merge(bd.rename(columns={'p': 'nk', 'e': 'e_n'})[['match_id', 'nk', 'e_n']], on=['match_id', 'nk'], how='left')
    r = r.merge(tt, on=['match_id', 'team_id'], how='left')
    r = r.merge(st.rename(columns={'p': 'k', 'pr': 'pr_k', 'ape': 'ape_k'}), on=['match_id', 'team_id', 'k'], how='left').merge(st.rename(columns={'p': 'nk', 'pr': 'pr_n', 'ape': 'ape_n'})[['match_id', 'team_id', 'nk', 'pr_n', 'ape_n']], on=['match_id', 'team_id', 'nk'], how='left')
    r = r[r.T_evt.notna() & r.e_k.notna() & r.e_n.notna()].copy()
    r['pst'] = np.where(~r.tdef.fillna(False).astype(bool), 0, np.where(r.haz.fillna(0).astype(int) == 1, 2, 1))
    r['term_n'] = r.nk == r.match_id.map(bd.groupby('match_id').p.max()); r['e_n_eff'] = np.where(r.term_n, INF, r.e_n)   # terminal next phase lasts until the match ends (winner censored)
    for nm, T in (('tr_evt', r.T_evt), ('tr_off', r.T_off)): r[nm] = np.where(T.isna(), -1, np.where(T <= r.e_k, 0, np.where(np.isfinite(T) & (T <= r.e_n_eff), 2, 1)))
    r['tr_evt_strict'] = np.where(r.T_evt <= r.e_k, 0, np.where(r.T_evt <= r.e_n, 2, 1))
    r['cohort'] = r.match_id.map(id2coh); r['match_type'] = r.match_id.map(ms.match_type); fin = np.isfinite(r.T_evt)
    r['near_ek'] = fin & ((r.T_evt - r.e_k).abs() < 15); r['near_en'] = fin & ((r.T_evt - r.e_n).abs() < 15) & ~r.term_n; r['near15'] = r.near_ek | r.near_en
    r['nopos_k'] = r.pr_k.fillna(0) == 0; r['nopos_n'] = r.pr_n.fillna(0) == 0
    r['knock_k'] = (r.pr_k.fillna(0) > 0) & (r.ape_k.fillna(0) == 0); r['knock_n'] = (r.pr_n.fillna(0) > 0) & (r.ape_n.fillna(0) == 0)
    r['ai_team'] = r.n_ai > 0; r['ai_all'] = r.n_ai == r.n_mem; r['airoyale'] = r.match_type == 'airoyale'; r['logout'] = r.n_logout > 0; r['redeploy'] = r.n_multi > 0; r['ai_bucket'] = np.select([r.n_ai == 0, r.n_ai == r.n_mem], ['no_ai', 'all_ai'], 'mixed')
    mt = pd.DataFrame({'S': ms.s_ep, 'match_type': ms.match_type, 'off_ts': off, 'mad_ts': mad, 'n_human_kills': nhk, 'delta_gs': gs - ms.s_ep}).reset_index().rename(columns={'index': 'match_id'})
    mt['ai_share'] = mt.match_id.map(pl.groupby('match_id').is_ai.mean()); mt['n_multi'] = mt.match_id.map(pl.groupby('match_id').multi.sum()); mt['n_players'] = mt.match_id.map(pl.groupby('match_id').size()); mt['cohort'] = mt.match_id.map(id2coh)
    return r, tt, mt, pl
def typ_of(pipe, tr):
    return np.select([(pipe >= 1) & (tr == 0), (pipe == 0) & (tr >= 1), (pipe == 2) & (tr == 1), (pipe == 1) & (tr == 2)], ['risk_FP', 'risk_FN', 'label_FP', 'label_FN'], 'agree')
def categorize(d):
    """Exclusive category for rows where pipe != tr_evt (priority order); flags stay available as columns."""
    typ = typ_of(d['pst'].values, d.tr_evt.values)
    unobs = ((typ == 'risk_FN') & d.nopos_k.values) | ((typ == 'label_FP') & d.nopos_n.values)
    knock = ((typ == 'risk_FN') & d.knock_k.values) | ((typ == 'label_FP') & d.knock_n.values)
    stale = np.isin(typ, ['risk_FP', 'label_FN'])
    return typ, np.select([d.logout.values, d.redeploy.values, d.near15.values, unobs, knock, stale], ['afk_logout', 'revive_redeploy', 'near_boundary_15s', 'unobserved_no_rows', 'knocked_last_snapshot', 'stale_last_snapshot'], 'other')
