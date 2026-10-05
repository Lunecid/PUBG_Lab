#!/usr/bin/env python3
"""BigComp 2027 data profile part 2: player-level (stage-2) corpus estimate on a random match sample.  READ-ONLY (SELECT only).
Sources: pubg.telem_positions/telem_damage/telem_kills/telem_groggy/telem_item_use, pubg.participants+rosters,
stdb.match_phase_global (phase bounds), stdb.team_phase_labels (audit only).  No derived feature tables are read."""
import argparse, json, os, sys, time, traceback
from collections import Counter
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dbh
ROOT = '/Users/lunecid/project/PUBG'; OUT = ROOT + '/reports/bigcomp2027_data_profile/csv'
R = ROOT + '/model_stgcn/checkpoints/br_om_ladder_required_v1'
INF = float('inf'); KMAX = 7; _conn = None
SQL = dict(
 meta="SELECT m.match_id, m.map_name, m.game_mode, m.duration_s, extract(epoch from m.started_at)::float8 AS started_ep, pm.match_type FROM stdb.matches m JOIN pubg.matches pm ON pm.match_id = m.match_id WHERE m.match_id = ANY(%s)",
 phase="SELECT match_id, phase_idx, phase_start_s::float8 AS s, phase_end_s::float8 AS e FROM stdb.match_phase_global WHERE match_id = ANY(%s) ORDER BY match_id, phase_idx",
 part="""SELECT p.match_id, p.player_id, r.team_id, p.death_type, p.time_survived::float8 AS time_survived, p.kills AS p_kills, p.dbnos AS p_dbnos, p.damage_dealt::float8 AS p_damage,
         p.revives AS p_revives, p.heals AS p_heals, p.boosts AS p_boosts FROM pubg.participants p JOIN pubg.rosters r ON r.roster_id = p.roster_id AND r.match_id = p.match_id WHERE p.match_id = ANY(%s)""",
 pos="""SELECT match_id, account_id AS acc, team_id AS team, elapsed_time AS t, health::float8 AS health, pos_x::float8 AS x, pos_y::float8 AS y, extract(epoch from event_time)::float8 AS ep
        FROM pubg.telem_positions WHERE match_id = ANY(%s) AND is_game >= 0.1 AND account_id IS NOT NULL AND team_id IS NOT NULL AND pos_x IS NOT NULL AND pos_y IS NOT NULL AND elapsed_time IS NOT NULL""",
 kill="SELECT match_id, victim_id, killer_id, is_suicide, extract(epoch from event_time)::float8 AS ep FROM pubg.telem_kills WHERE match_id = ANY(%s)",
 grg="SELECT match_id, victim_id, attacker_id, extract(epoch from event_time)::float8 AS ep FROM pubg.telem_groggy WHERE match_id = ANY(%s)",
 dmg="SELECT match_id, attacker_id, victim_id, damage::float8 AS damage, extract(epoch from event_time)::float8 AS ep FROM pubg.telem_damage WHERE match_id = ANY(%s)",
 lab="SELECT match_id, team_id, cut_phase, target_defined, eliminated_next_phase FROM stdb.team_phase_labels WHERE match_id = ANY(%s) AND cut_phase BETWEEN 1 AND 7",
 item="SELECT category, sub_category, item_id, count(*)::bigint AS n, coalesce(sum(stack_count),0)::bigint AS stack FROM pubg.telem_item_use WHERE match_id = ANY(%s) GROUP BY 1,2,3")
def getcur():
    global _conn
    if _conn is None: _conn = dbh.conn()
    return _conn.cursor()
def q(c, sql, params):
    c.execute(sql, params); return pd.DataFrame(c.fetchall(), columns=[d[0] for d in c.description])
def comb2(n): return n * (n - 1) // 2
def epairs(xy, team, r=300.0):
    z = (np.zeros(0, int), np.zeros(0, int), np.zeros(0))
    if len(xy) < 2: return z
    pr = cKDTree(xy).query_pairs(r, output_type='ndarray')
    if len(pr) == 0: return z
    i, j = pr[:, 0], pr[:, 1]; m = team[i] != team[j]; i, j = i[m], j[m]
    return i, j, np.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1])

def process(mid, cohort, T):
    meta, ph, part, pos, kill, grg, dmg, lab = (T[k] for k in ['meta', 'phase', 'part', 'pos', 'kill', 'grg', 'dmg', 'lab'])
    if len(ph) == 0 or len(pos) == 0 or len(part) == 0: raise ValueError(f'missing data ph={len(ph)} pos={len(pos)} part={len(part)}')
    ph = ph.sort_values('phase_idx'); phases = [int(x) for x in ph.phase_idx]; S = dict(zip(phases, ph.s.astype(float))); E = dict(zip(phases, ph.e.astype(float)))
    nxt = {phases[i]: (phases[i + 1] if i + 1 < len(phases) else None) for i in range(len(phases))}
    part = part.copy(); part['player_id'] = part.player_id.fillna('')
    team_of = dict(zip(part.player_id, part.team_id)); universe = list(part.player_id)
    pos = pos.sort_values('t', kind='mergesort').reset_index(drop=True); tt = pos.t.values.astype(float); off = float(np.median(pos.ep.values - tt))
    started = float(meta.started_ep.iloc[0]) if len(meta) else np.nan
    # ---- death times (kills first, participants fallback) ----
    death = {}
    if len(kill):
        for v, e_ in zip(kill.victim_id.values, kill.ep.values - off):
            if isinstance(v, str) and (v not in death or e_ < death[v]): death[v] = float(e_)
    kdeath = dict(death); diffs = []
    for r in part.itertuples(index=False):
        p = r.player_id
        if p in kdeath:
            if np.isfinite(r.time_survived): diffs.append(kdeath[p] - r.time_survived)
        else:
            dtp = r.death_type; death[p] = INF if (not isinstance(dtp, str) or dtp == 'alive') else (float(r.time_survived) if np.isfinite(r.time_survived) else INF)
    members = {}
    for p in universe: members.setdefault(team_of[p], []).append(p)
    Tdeath = {t_: (max(death[p] for p in ms) if all(np.isfinite(death[p]) for p in ms) else INF) for t_, ms in members.items()}
    # ---- position interval histogram, knock evidence ----
    ps = pos.sort_values(['acc', 't'], kind='mergesort'); a_ = ps.acc.values; t_ = ps.t.values.astype(float)
    dd = np.diff(t_)[a_[1:] == a_[:-1]]; hist = np.bincount(np.clip(dd, 0, 120).astype(int), minlength=121)
    grp = {a: (g.t.values.astype(float), g.health.values) for a, g in ps.groupby('acc', sort=False)}
    kn_total = kn_h0 = kn_revive_like = 0
    for v, e_ in zip(grg.victim_id.values, grg.ep.values - off):
        if v not in grp: continue
        ta_, ha_ = grp[v]; ix = np.searchsorted(ta_, e_, side='right'); kn_total += 1
        if ix < len(ta_) and ha_[ix] == 0: kn_h0 += 1
        dth = death.get(v, INF)
        if np.any((ta_ >= e_ + 60) & (ha_ > 0) & (ta_ < dth)): kn_revive_like += 1
    # ---- events binned by phase (half-open [S_k, S_next)) ----
    edges = np.array([S[k] for k in phases]); nb = len(phases) + 1
    def bc(t, mask, w=None): return np.bincount(np.searchsorted(edges, t[mask], side='right'), weights=None if w is None else w[mask], minlength=nb)
    if len(dmg):
        dt_ = dmg.ep.values - off; dam = dmg.damage.values.astype(float); att = dmg.attacker_id.values; vic = dmg.victim_id.values
        aok = np.array([isinstance(a, str) and a != '' for a in att]); vok = np.array([isinstance(v, str) and v != '' for v in vic])
        ta = dmg.attacker_id.map(team_of); tv = dmg.victim_id.map(team_of); enemy_d = (ta.notna() & tv.notna() & (ta != tv)).values
        dvalid = aok & vok & (dam > 0) & (att != vic)
    else:
        dt_ = dam = att = vic = np.zeros(0); aok = vok = enemy_d = dvalid = np.zeros(0, bool)
    if len(kill):
        kt_ = (kill.ep.values - off); kk = kill.killer_id.map(team_of); kv = kill.victim_id.map(team_of); enemy_k = (kk.notna() & kv.notna() & (kk != kv)).values; nokiller = kill.killer_id.isna().values
    else: kt_ = np.zeros(0); enemy_k = nokiller = np.zeros(0, bool)
    if len(grg):
        gt_ = (grg.ep.values - off); ga = grg.attacker_id.map(team_of); gv = grg.victim_id.map(team_of); enemy_g = (ga.notna() & gv.notna() & (ga != gv)).values
    else: gt_ = np.zeros(0); enemy_g = np.zeros(0, bool)
    allk = np.ones(len(kt_), bool); allg = np.ones(len(gt_), bool); alld = np.ones(len(dt_), bool)
    ev = dict(kills_all=bc(kt_, allk), kills_enemy=bc(kt_, enemy_k), kills_nokiller=bc(kt_, nokiller), groggy_all=bc(gt_, allg), groggy_enemy=bc(gt_, enemy_g),
              dmg_ev_all=bc(dt_, alld), dmg_ev_enemy=bc(dt_, enemy_d), dmg_ev_noattacker=bc(dt_, ~aok), dmg_sum_all=bc(dt_, alld, dam), dmg_sum_enemy=bc(dt_, enemy_d, dam))
    ev_rows = []
    for i in range(nb):
        r = dict(match_id=mid, cohort=cohort, k=(0 if i == 0 else phases[i - 1])); r.update({n: float(v[i]) for n, v in ev.items()}); ev_rows.append(r)
    # ---- label audit ----
    aud_rows = []
    for k in phases:
        if k > KMAX: break
        rows = lab[lab.cut_phase == k]
        if not len(rows): continue
        td = rows.team_id.map(Tdeath); ok = td.notna().values; td = td.values.astype(float); nk = nxt[k]
        lr = rows.target_defined.fillna(False).astype(bool).values; lp = lr & rows.eliminated_next_phase.fillna(False).astype(bool).values
        tr_alive = td > E[k]; true_risk = tr_alive & (nk is not None); true_pos = true_risk & (td <= E[nk]) if nk is not None else np.zeros(len(td), bool)
        m = ok
        aud_rows.append(dict(match_id=mid, cohort=cohort, k=k, n_rows=int(m.sum()), n_true_risk=int((true_risk & m).sum()), n_label_risk=int((lr & m).sum()), n_risk_both=int((true_risk & lr & m).sum()),
            n_true_pos=int((true_pos & m).sum()), n_label_pos=int((lp & m).sum()), n_pos_both=int((true_pos & lp & m).sum()), n_labelpos_dead_before_cut=int((lp & (td <= E[k]) & m).sum()),
            n_labelpos_alive_after_next=int((lp & (td > E[nk]) & m).sum()) if nk is not None else 0))
    # ---- per-cut quantities ----
    cut_rows = []
    for k in phases:
        if k > KMAX: break
        ts, te = S[k], E[k]; tc = te - 10.0
        A = [p for p in universe if death[p] > ts]; Aset = set(A); C = [p for p in A if death[p] > tc]; Cset = set(C)
        a_team = Counter(team_of[p] for p in A); c_team = Counter(team_of[p] for p in C)
        sub = pos[pos.t <= tc]; last = sub.drop_duplicates('acc', keep='last'); lastA = last[last.acc.isin(Aset)]
        stale = tc - lastA.t.values.astype(float); fm = stale <= 30.0
        xy_all = np.c_[lastA.x.values / 100.0, lastA.y.values / 100.0]; tm_all = lastA.team.values
        xy_f = xy_all[fm]; tm_f = tm_all[fm]; acc_f = lastA.acc.values[fm]; h_f = lastA.health.values[fm]
        i_all, j_all, d_all = epairs(xy_all, tm_all); i_f, j_f, d_f = epairs(xy_f, tm_f)
        inC_f = np.array([a in Cset for a in acc_f], dtype=bool)
        bins = np.histogram(d_f, bins=[0, 50, 100, 200, 300.0001])[0] if len(d_f) else np.zeros(4, int)
        tmf = Counter(tm_f.tolist()); lastt = dict(zip(lastA.acc, stale))
        miss = lambda p, w: (lastt.get(p) is None) or (lastt[p] > w)
        team_C = {}
        for p in C: team_C.setdefault(team_of[p], []).append(p)
        row = dict(match_id=mid, cohort=cohort, k=k, has_next=int(nxt[k] is not None), phase_s=ts, phase_e=te, phase_dur=te - ts, nodes_A=len(A), teams_A=len(a_team), nodes_C=len(C), teams_C=len(c_team),
                   tm_A=sum(comb2(n) for n in a_team.values()), tm_C=sum(comb2(n) for n in c_team.values()), obs_nodes=len(lastA), fresh_nodes=int(fm.sum()),
                   stale_med=float(np.median(stale)) if len(stale) else np.nan, stale_p90=float(np.percentile(stale, 90)) if len(stale) else np.nan, tm_fresh=sum(comb2(n) for n in tmf.values()),
                   enemy_all=len(i_all), enemy_fresh=len(i_f), enemy_fresh_C=int((inC_f[i_f] & inC_f[j_f]).sum()) if len(i_f) else 0, bin_lt50=int(bins[0]), bin_50_100=int(bins[1]), bin_100_200=int(bins[2]),
                   bin_200_300=int(bins[3]), nodes_with_enemy_fresh=len(set(i_f.tolist()) | set(j_f.tolist())), C_players=len(C), C_miss15=sum(miss(p, 15) for p in C), C_miss30=sum(miss(p, 30) for p in C),
                   C_teams=len(team_C), C_team_miss15=sum(all(miss(p, 15) for p in ms) for ms in team_C.values()), C_team_miss30=sum(all(miss(p, 30) for p in ms) for ms in team_C.values()),
                   A_nopos=sum(1 for p in A if p not in lastt), fresh_C_n=int(inC_f.sum()), fresh_C_h0=int(((h_f == 0) & inC_f).sum()))
        z = dict(cb_enemy=0, cb_friendly=0, cb_unknown=0, cb_enemy_inA=0, cb_enemy_obs=0, cb_enemy_within300=0, cb_events_enemy=0, cb_dmg_enemy=0.0)
        if tc > ts and len(dmg):
            wm = (dt_ >= ts) & (dt_ <= tc) & dvalid
            if wm.any():
                pairs = pd.DataFrame({'a': att[wm], 'v': vic[wm]}).drop_duplicates(); pa = pairs.a.map(team_of); pv = pairs.v.map(team_of)
                known = (pa.notna() & pv.notna()).values; same = (pa == pv).values; em = known & ~same
                inA = (pairs.a.isin(Aset) & pairs.v.isin(Aset)).values
                xm = dict(zip(acc_f, xy_f[:, 0])); ym = dict(zip(acc_f, xy_f[:, 1])); ax = pairs.a.map(xm); vx = pairs.v.map(xm); ay = pairs.a.map(ym); vy = pairs.v.map(ym)
                om = em & ax.notna().values & vx.notna().values; dist = np.hypot((ax - vx).values[om], (ay - vy).values[om]); evm = wm & enemy_d
                z = dict(cb_enemy=int(em.sum()), cb_friendly=int((known & same).sum()), cb_unknown=int((~known).sum()), cb_enemy_inA=int((em & inA).sum()), cb_enemy_obs=int(om.sum()),
                         cb_enemy_within300=int((dist <= 300).sum()), cb_events_enemy=int(evm.sum()), cb_dmg_enemy=float(dam[evm].sum()))
        row.update(z); cut_rows.append(row)
    mrow = dict(match_id=mid, cohort=cohort, match_type=(meta.match_type.iloc[0] if len(meta) else None), map_name=(meta.map_name.iloc[0] if len(meta) else None), duration_s=(float(meta.duration_s.iloc[0]) if len(meta) else np.nan), n_phases=len(phases), max_phase=max(phases),
                n_players=len(universe), n_ai=int(part.player_id.str.startswith('ai.').sum()), n_teams=len(members), pos_rows=len(pos), pos_players=int(pos.acc.nunique()), dmg_rows=len(dmg),
                kill_rows=len(kill), groggy_rows=len(grg), x_min=float(pos.x.min()), x_max=float(pos.x.max()), y_min=float(pos.y.min()), y_max=float(pos.y.max()), off_minus_started=off - started,
                team_mismatch_share=float(np.mean([team_of.get(a, t) != t for a, t in zip(*[pos.drop_duplicates('acc')[c].values for c in ('acc', 'team')])])),
                ts_diff_med=float(np.median(diffs)) if diffs else np.nan, ts_diff_mad=float(np.median(np.abs(diffs))) if diffs else np.nan, n_kill_events_victims=len(kdeath),
                p_kills=int(part.p_kills.sum()), p_dbnos=int(part.p_dbnos.sum()), p_damage=float(part.p_damage.sum()), p_revives=int(part.p_revives.sum()), p_heals=int(part.p_heals.sum()), p_boosts=int(part.p_boosts.sum()),
                knock_events=kn_total, knock_next_sample_h0=kn_h0, knock_revive_like=kn_revive_like, pos_health0_share=float((pos.health.values == 0).mean()), pos_health_nan_share=float(np.isnan(pos.health.values).mean()))
    return mrow, cut_rows, ev_rows, aud_rows, hist

def run_batch(job):
    cohort, mids, with_items = job; timing = {}; out = dict(cohort=cohort, mrows=[], crows=[], erows=[], arows=[], hist=np.zeros(121, np.int64), items=None, errors=[], timing=timing, n=len(mids))
    try:
        c = getcur(); tabs = {}
        for name in ['meta', 'phase', 'part', 'pos', 'kill', 'grg', 'dmg', 'lab']:
            t0 = time.time(); tabs[name] = q(c, SQL[name], (mids,)); timing[name] = time.time() - t0
        if with_items:
            t0 = time.time(); out['items'] = q(c, SQL['item'], (mids,)).values.tolist(); timing['item'] = time.time() - t0
        timing['fetch_total'] = sum(v for k_, v in timing.items() if k_ != 'item')
        grouped = {n: {k_: g for k_, g in df.groupby('match_id', sort=False)} for n, df in tabs.items()}
        t0 = time.time()
        for mid in mids:
            try:
                T = {n: grouped[n].get(mid, tabs[n].iloc[0:0]) for n in tabs}
                m, cr, er, ar, h = process(mid, cohort, T); out['mrows'].append(m); out['crows'] += cr; out['erows'] += er; out['arows'] += ar; out['hist'] += h
            except Exception:
                out['errors'].append((mid, traceback.format_exc()[-600:]))
        timing['proc'] = time.time() - t0
    except Exception:
        out['errors'].append(('BATCH', traceback.format_exc()[-800:]))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--per-cohort', type=int, default=200); ap.add_argument('--workers', type=int, default=3); ap.add_argument('--batch', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--tag', default='sample'); args = ap.parse_args()
    cohorts = {}
    for d in sorted(os.listdir(R)):
        p = f'{R}/{d}/split.json'
        if os.path.isfile(p): cohorts[d] = sorted(sum(json.load(open(p))['splits'].values(), []))
    rng = np.random.RandomState(args.seed); jobs = []
    for c, ids in cohorts.items():
        pick = rng.choice(ids, size=min(args.per_cohort, len(ids)), replace=False).tolist()
        for i in range(0, len(pick), args.batch): jobs.append((c, pick[i:i + args.batch], i == 0))
    print(f'jobs={len(jobs)} matches={sum(len(j[1]) for j in jobs)} workers={args.workers}', flush=True)
    t0 = time.time(); res = []
    if args.workers <= 1:
        for j in jobs:
            res.append(run_batch(j)); print(f'[{time.time()-t0:7.1f}s] {len(res)}/{len(jobs)}', flush=True)
    else:
        with Pool(args.workers) as pool:
            for r in pool.imap_unordered(run_batch, jobs):
                res.append(r)
                if len(res) % 20 == 0 or len(res) == len(jobs): print(f'[{time.time()-t0:7.1f}s] {len(res)}/{len(jobs)} errors_so_far={sum(len(x["errors"]) for x in res)}', flush=True)
    wall = time.time() - t0; tag = args.tag
    mdf = pd.DataFrame([m for r in res for m in r['mrows']]); cdf = pd.DataFrame([x for r in res for x in r['crows']]); edf = pd.DataFrame([x for r in res for x in r['erows']]); adf = pd.DataFrame([x for r in res for x in r['arows']])
    mdf.to_csv(f'{OUT}/{tag}_match_level.csv', index=False); cdf.to_csv(f'{OUT}/{tag}_cut_level.csv', index=False); edf.to_csv(f'{OUT}/{tag}_events_by_phase.csv', index=False); adf.to_csv(f'{OUT}/{tag}_label_audit.csv', index=False)
    hist = []
    for c in cohorts:
        h = sum((r['hist'] for r in res if r['cohort'] == c), np.zeros(121, np.int64)); hist += [dict(cohort=c, interval_s=i, n=int(h[i])) for i in range(121) if h[i]]
    pd.DataFrame(hist).to_csv(f'{OUT}/{tag}_position_interval_hist.csv', index=False)
    items = [row for r in res if r['items'] for row in r['items']]
    if items: pd.DataFrame(items, columns=['category', 'sub_category', 'item_id', 'n', 'stack']).groupby(['category', 'sub_category', 'item_id'], dropna=False).sum().reset_index().sort_values('n', ascending=False).to_csv(f'{OUT}/{tag}_item_use_taxonomy.csv', index=False)
    errs = [e for r in res for e in r['errors']]
    tm = pd.DataFrame([dict(cohort=r['cohort'], n=r['n'], **r['timing']) for r in res]); tm.to_csv(f'{OUT}/{tag}_batch_timing.csv', index=False)
    meta = dict(args=vars(args), wall_s=wall, n_matches_ok=len(mdf), n_errors=len(errs), fetch_s_total=float(tm.fetch_total.sum()), proc_s_total=float(tm.proc.sum()), n_batches=len(res))
    json.dump(meta, open(f'{OUT}/{tag}_run_meta.json', 'w'), indent=1); print('META', json.dumps(meta))
    for e in errs[:5]: print('ERR', e[0], e[1])
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 80)
    if len(cdf):
        print(cdf.groupby('k')[['nodes_A', 'nodes_C', 'tm_A', 'obs_nodes', 'fresh_nodes', 'enemy_all', 'enemy_fresh', 'cb_enemy', 'cb_enemy_inA', 'C_miss30', 'C_team_miss30']].mean().round(2).to_string())
        print(mdf[['pos_rows', 'dmg_rows', 'kill_rows', 'groggy_rows', 'n_players', 'n_ai', 'off_minus_started', 'team_mismatch_share', 'ts_diff_med', 'ts_diff_mad', 'x_max', 'y_max', 'pos_health0_share']].describe().round(3).to_string())
        print(adf.groupby('k')[[c for c in adf.columns if c.startswith('n_')]].sum().to_string())
if __name__ == '__main__': main()
