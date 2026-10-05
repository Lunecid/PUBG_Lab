#!/usr/bin/env python3
"""BigComp 2027 data profile, part 1: funnel, per-cohort profile, labels, event volumes, stage-2 counts.
READ-ONLY. Uses raw pubg.* and stable stdb.* tables only (matches, match_phase_global, team_phase_state/combat/labels)."""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dbh
ROOT = '/Users/lunecid/project/PUBG'; OUT = ROOT + '/reports/bigcomp2027_data_profile/csv'
R = ROOT + '/model_stgcn/checkpoints/br_om_ladder_required_v1'
FLAT = ROOT + '/model_stgcn/experiment_datasets/research_cohort_flat/match_cohort_flat.csv'
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 300)
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)
conn = dbh.conn(); cur = conn.cursor()
def q(sql, params=None):
    cur.execute(sql, params); return pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
def qchunk(sql, ids, n=3000):
    return pd.concat([q(sql, (ids[i:i+n],)) for i in range(0, len(ids), n)], ignore_index=True)
def save(df, name): df.to_csv(f'{OUT}/{name}', index=False)
MAPLBL = {'Baltic_Main':'Erangel','Tiger_Main':'Taego','Desert_Main':'Miramar','Neon_Main':'Rondo','Savage_Main':'Sanhok',
          'DihorOtok_Main':'Vikendi','Chimera_Main':'Paramo','Summerland_Main':'Karakin','Kiki_Main':'Deston'}
def mode_persp(gm):
    if not isinstance(gm, str): return ('other', 'TPP')
    p = 'FPP' if gm.endswith('-fpp') else 'TPP'; b = gm[:-4] if gm.endswith('-fpp') else gm
    return (b if b in ('solo', 'duo', 'squad') else 'other:' + b, p)
def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (c - h, c + h)

# ---------- cohorts from split files ----------
rows = []; id2coh = {}
for d in sorted(os.listdir(R)):
    p = f'{R}/{d}/split.json'
    if not os.path.isfile(p): continue
    s = json.load(open(p)); md = s['metadata']; sp = s['splits']
    for part, lst in sp.items():
        for m in lst: id2coh[m] = d
    rows.append(dict(slug=d, cohort=md['cohort'], map_name=md['map'], mode_meta=md['mode'], n_train=len(sp['train']), n_val=len(sp['val']),
                     n_test=len(sp['test']), n_total=s['n_matches']))
cohdf = pd.DataFrame(rows).sort_values('n_total', ascending=False).reset_index(drop=True)
all_ids = sorted(id2coh); order = list(cohdf.slug)
log('cohorts', len(cohdf), 'unique ids', len(all_ids), 'sum n_total', int(cohdf.n_total.sum()))

# ---------- flat csv ----------
flat = pd.read_csv(FLAT)
flat['date_kst'] = pd.to_datetime(flat['created_at'], utc=True).dt.tz_convert('Asia/Seoul').dt.strftime('%Y-%m-%d')
flat['cohort'] = flat.match_id.map(id2coh)
flat[['mode', 'persp']] = flat.game_mode.apply(lambda g: pd.Series(mode_persp(g)))
agree = ((flat.persp == flat.perspective) | flat.perspective.isna()).mean()
log('flat rows', len(flat), 'eligible', int(flat.graph_eligible_n128.sum()), 'in cohorts', int(flat.cohort.notna().sum()), 'persp agree', round(agree, 4))
chk = flat.dropna(subset=['cohort']).groupby('cohort')[['map_name', 'game_mode', 'perspective', 'team_size']].nunique().max()
log('max nunique within cohort (map,game_mode,persp,team_size):', chk.to_dict())
cm = flat.dropna(subset=['cohort']).groupby('cohort')[['game_mode', 'perspective', 'team_size']].first()
cohdf = cohdf.merge(cm, left_on='slug', right_index=True, how='left'); cohdf['map_label'] = cohdf.map_name.map(MAPLBL)

# ---------- 1. funnel ----------
raw = q("""SELECT map_name, game_mode, match_type, is_custom, count(*)::bigint AS n, min(created_at) AS first_created, max(created_at) AS last_created,
                  min(collected_at) AS first_collected, max(collected_at) AS last_collected FROM pubg.matches GROUP BY 1,2,3,4 ORDER BY 1,2,3,4""")
save(raw, 'raw_db_by_map_mode_matchtype_custom.csv')
raw[['mode', 'persp']] = raw.game_mode.apply(lambda g: pd.Series(mode_persp(g)))
log('pubg.matches total', int(raw.n.sum()), '| custom', int(raw[raw.is_custom == True].n.sum()), '| match_type counts', raw.groupby('match_type').n.sum().to_dict())
rng = q("SELECT min(created_at) a, max(created_at) b, min(collected_at) c, max(collected_at) d FROM pubg.matches"); log('pubg.matches created_at/collected_at range:', rng.iloc[0].to_dict())
f_raw = raw.groupby(['map_name', 'mode', 'persp']).n.sum().rename('raw_db')
f_csv = flat.groupby(['map_name', 'mode', 'persp']).size().rename('csv_total')
f_in = flat[flat.cohort_status != 'raw_only_not_ready'].groupby(['map_name', 'mode', 'persp']).size().rename('csv_in_stdb')
f_el = flat[flat.graph_eligible_n128].groupby(['map_name', 'mode', 'persp']).size().rename('graph_eligible')
f_13 = flat[flat.cohort.notna()].groupby(['map_name', 'mode', 'persp']).size().rename('main13')
fun = pd.concat([f_raw, f_csv, f_in, f_el, f_13], axis=1).fillna(0).astype(int).reset_index()
fun['map_label'] = fun.map_name.map(MAPLBL); fun = fun.sort_values(['raw_db', 'csv_total'], ascending=False)
save(fun, 'funnel_map_mode_persp.csv')
print(fun.drop(columns='map_label').to_string(index=False)); print('TOTAL', fun[['raw_db', 'csv_total', 'csv_in_stdb', 'graph_eligible', 'main13']].sum().to_dict())
print('csv cohort_status:', flat.cohort_status.value_counts().to_dict(), '| exclude_reason:', flat.exclude_reason.value_counts().to_dict())
print('csv raw_fetched true:', int(flat.raw_fetched.sum()), '| csv non-eligible maps:', flat[~flat.graph_eligible_n128].map_name.value_counts().to_dict())
csv_ids = flat.match_id.tolist()
cov = dict(csv_in_pubg_matches=int(q("SELECT count(*) c FROM pubg.matches WHERE match_id = ANY(%s)", (csv_ids,)).c[0]),
           csv_in_stdb_matches=int(q("SELECT count(*) c FROM stdb.matches WHERE match_id = ANY(%s)", (csv_ids,)).c[0]),
           csv_in_telem_match_start=int(q("SELECT count(DISTINCT match_id) c FROM pubg.telem_match_start WHERE match_id = ANY(%s)", (csv_ids,)).c[0]),
           csv_in_telem_phase_changes=int(q("SELECT count(DISTINCT match_id) c FROM pubg.telem_phase_changes WHERE match_id = ANY(%s)", (csv_ids,)).c[0]),
           pubg_matches_not_in_csv=int(q("SELECT count(*) c FROM pubg.matches WHERE NOT (match_id = ANY(%s))", (csv_ids,)).c[0]),
           stdb_matches_total=int(q("SELECT count(*) c FROM stdb.matches").c[0]))
log('coverage:', cov); json.dump(cov, open(f'{OUT}/funnel_coverage_checks.json', 'w'), indent=1)
try: print(q("SELECT * FROM pubg.v_collection_status LIMIT 30").to_string(index=False))
except Exception as e: log('v_collection_status err', e)
pdays = q("""SELECT (created_at AT TIME ZONE 'Asia/Seoul')::date::text AS date_kst, count(*)::bigint AS raw_db FROM pubg.matches GROUP BY 1 ORDER BY 1""")
el_d = flat[flat.graph_eligible_n128].groupby('date_kst').size().rename('graph_eligible'); c13_d = flat[flat.cohort.notna()].groupby('date_kst').size().rename('main13')
csv_d = flat.groupby('date_kst').size().rename('csv_total')
pdays = pd.concat([pdays.set_index('date_kst').raw_db, csv_d, el_d, c13_d], axis=1).fillna(0).astype(int).rename_axis('date_kst').reset_index(); save(pdays, 'matches_per_day.csv')
nz = pdays[pdays.raw_db > 0]
log('days with raw_db>0:', len(nz), 'range', nz.date_kst.min(), nz.date_kst.max(), '| per-day raw_db min/med/p10/p90/max', nz.raw_db.min(), nz.raw_db.median(), np.percentile(nz.raw_db, 10), np.percentile(nz.raw_db, 90), nz.raw_db.max(),
    '| calendar days in range', (pd.Timestamp(nz.date_kst.max()) - pd.Timestamp(nz.date_kst.min())).days + 1)
nz2 = pdays[pdays.csv_total > 0]; log('csv days', len(nz2), nz2.date_kst.min(), nz2.date_kst.max(), 'csv per-day min/med/max', nz2.csv_total.min(), nz2.csv_total.median(), nz2.csv_total.max(),
    '| eligible per day min/med/max', nz2.graph_eligible.min(), nz2.graph_eligible.median(), nz2.graph_eligible.max(), '| main13 per day med', nz2.main13.median())

mt = qchunk("SELECT match_id, match_type FROM pubg.matches WHERE match_id = ANY(%s)", all_ids); mt['cohort'] = mt.match_id.map(id2coh)
mtp = mt.groupby(['cohort', 'match_type']).size().unstack(fill_value=0).loc[order]; mtp.to_csv(f'{OUT}/cohort_match_type.csv'); print('MATCH TYPE per cohort:'); print(mtp.to_string()); log('match_type total (13 cohorts):', mtp.sum().to_dict())
# ---------- 2. per-cohort profile ----------
sub = flat[flat.cohort.notna()].copy()
pg = qchunk("""SELECT match_id, phase_idx, phase_start_s::float8 s, phase_end_s::float8 e, phase_duration_s::float8 dur, start_alive_teams, end_alive_teams,
                      start_alive_players, end_alive_players FROM stdb.match_phase_global WHERE match_id = ANY(%s)""", all_ids)
pg['cohort'] = pg.match_id.map(id2coh); log('match_phase_global rows', len(pg), 'matches', pg.match_id.nunique())
mp = pg.groupby('match_id').agg(max_phase=('phase_idx', 'max'), min_phase=('phase_idx', 'min'), n_phase=('phase_idx', 'nunique'))
sub = sub.merge(mp, left_on='match_id', right_index=True, how='left')
sub['gap'] = sub.max_phase.notna() & ((sub.n_phase != sub.max_phase - sub.min_phase + 1) | (sub.min_phase != 1))
def stats(x, pre):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return {pre + 'med': np.median(x), pre + 'p10': np.percentile(x, 10), pre + 'p90': np.percentile(x, 90), pre + 'mean': x.mean(), pre + 'min': x.min(), pre + 'max': x.max()}
srows = []
for c in order:
    g = sub[sub.cohort == c]; r = dict(slug=c); r.update(stats(g.num_teams, 'teams_')); r.update(stats(g.num_players, 'players_')); r.update(stats(g.duration, 'dur_'))
    r['dur_over2000_share'] = float((g.duration > 2000).mean()); r['dur_over2000_n'] = int((g.duration > 2000).sum()); r.update(stats(g.max_phase, 'maxphase_'))
    r['n_matches'] = len(g); r['n_no_phase_rows'] = int(g.max_phase.isna().sum()); r['n_phase_gap_or_min!=1'] = int(g.gap.sum())
    r['n_maxphase_le3'] = int((g.max_phase <= 3).sum()); r['n_maxphase_le5'] = int((g.max_phase <= 5).sum()); r['n_maxphase_ge10'] = int((g.max_phase >= 10).sum())
    srows.append(r)
csum = cohdf.merge(pd.DataFrame(srows), on='slug'); save(csum, 'cohort_summary.csv')
print(csum[['slug', 'n_train', 'n_val', 'n_test', 'n_total', 'teams_med', 'teams_p10', 'teams_p90', 'players_med', 'players_p10', 'players_p90']].to_string(index=False))
print(csum[['slug', 'dur_med', 'dur_p10', 'dur_p90', 'dur_over2000_share', 'dur_over2000_n', 'maxphase_med', 'maxphase_p10', 'maxphase_p90', 'maxphase_min', 'maxphase_max',
            'n_no_phase_rows', 'n_phase_gap_or_min!=1', 'n_maxphase_le3', 'n_maxphase_le5', 'n_maxphase_ge10']].round(3).to_string(index=False))
mpd = sub.groupby(['cohort', 'max_phase']).size().rename('n').reset_index(); mpd['share'] = mpd.n / mpd.groupby('cohort').n.transform('sum'); save(mpd, 'max_phase_distribution.csv')
print(mpd.pivot(index='cohort', columns='max_phase', values='n').fillna(0).astype(int).loc[order].to_string())

sub2 = sub.merge(mt[['match_id', 'match_type']], on='match_id', how='left'); mrows_ = []
for (c_, t_), g in sub2.groupby(['cohort', 'match_type']):
    mrows_.append(dict(cohort=c_, match_type=t_, n_matches=len(g), teams_med=g.num_teams.median(), players_med=g.num_players.median(), dur_med=g.duration.median(), dur_over2000_share=float((g.duration > 2000).mean()),
                       maxphase_med=g.max_phase.median(), maxphase_p10=g.max_phase.quantile(.1), maxphase_p90=g.max_phase.quantile(.9)))
cmt = pd.DataFrame(mrows_); save(cmt, 'cohort_matchtype_profile.csv'); print('PROFILE by cohort x match_type:'); print(cmt.round(3).to_string(index=False))
# phase timing
rp = qchunk("""SELECT pc.match_id, floor(pc.phase)::int AS ph, min(coalesce(pc.elapsed_time::float8, extract(epoch FROM (pc.event_time - ms.event_time)))) AS t_first,
                      count(*) FILTER (WHERE pc.elapsed_time IS NULL)::bigint AS n_null_elapsed, count(*)::bigint AS n_rows
               FROM pubg.telem_phase_changes pc JOIN pubg.telem_match_start ms ON ms.match_id = pc.match_id WHERE pc.match_id = ANY(%s) GROUP BY 1,2""", all_ids)
log('telem_phase_changes (cohort matches): rows with NULL elapsed_time', int(rp.n_null_elapsed.sum()), 'of', int(rp.n_rows.sum()), '| matches with raw phase rows', rp.match_id.nunique())
rp['cohort'] = rp.match_id.map(id2coh); pg = pg.merge(rp.rename(columns={'ph': 'phase_idx'})[['match_id', 'phase_idx', 't_first']], on=['match_id', 'phase_idx'], how='left')
pg['map_name'] = pg.cohort.map(cohdf.set_index('slug').map_name); pg['start_minus_raw'] = pg.s - pg.t_first
print('raw telem_phase_changes values:', q("SELECT phase::float8 AS phase, count(*)::bigint n FROM pubg.telem_phase_changes GROUP BY 1 ORDER BY 1").set_index('phase').n.to_dict())
def tim(g):
    return pd.Series(dict(n_matches=g.match_id.nunique(), start_med=g.s.median(), start_p10=g.s.quantile(.1), start_p90=g.s.quantile(.9), end_med=g.e.median(), end_p10=g.e.quantile(.1),
                          end_p90=g.e.quantile(.9), dur_med=g.dur.median(), raw_start_med=g.t_first.median(), start_minus_raw_med=g.start_minus_raw.median(),
                          dur_lt20_share=(g.dur < 20).mean(), dur_lt10_share=(g.dur < 10).mean()))
pc = pg[pg.phase_idx.between(1, 12)].groupby(['cohort', 'phase_idx']).apply(tim).reset_index(); save(pc, 'phase_timing_by_cohort.csv')
pm = pg[pg.phase_idx.between(1, 12)].groupby(['map_name', 'phase_idx']).apply(tim).reset_index(); pm['map_label'] = pm.map_name.map(MAPLBL); save(pm, 'phase_timing_by_map.csv')
print('PHASE START median (s) by map:'); print(pm.pivot(index='map_label', columns='phase_idx', values='start_med').round(0).to_string())
print('PHASE END median (s) by map:'); print(pm.pivot(index='map_label', columns='phase_idx', values='end_med').round(0).to_string())
print('n matches reaching phase by map:'); print(pm.pivot(index='map_label', columns='phase_idx', values='n_matches').astype('Int64').to_string())
print('start_minus_raw median by map:'); print(pm.pivot(index='map_label', columns='phase_idx', values='start_minus_raw_med').round(1).to_string())

# ---------- 3. labels ----------
lb = qchunk("""SELECT match_id, cut_phase, count(*) FILTER (WHERE target_defined)::bigint AS risk, count(*) FILTER (WHERE target_defined AND eliminated_next_phase)::bigint AS pos,
                      count(*) FILTER (WHERE alive_at_phase_start)::bigint AS alive_start, count(*) FILTER (WHERE alive_at_phase_end)::bigint AS alive_end
               FROM stdb.team_phase_labels WHERE match_id = ANY(%s) GROUP BY match_id, cut_phase""", all_ids)
lb['cohort'] = lb.match_id.map(id2coh)
hz = lb[lb.cut_phase.between(1, 10)].groupby(['cohort', 'cut_phase']).agg(n_matches=('risk', lambda x: int((x > 0).sum())), risk=('risk', 'sum'), pos=('pos', 'sum')).reset_index()
hz['hazard'] = hz.pos / hz.risk; ci = hz.apply(lambda r: wilson(r.pos, r.risk), axis=1); hz['ci_lo'] = [a for a, b in ci]; hz['ci_hi'] = [b for a, b in ci]
hz['map_label'] = hz.cohort.map(cohdf.set_index('slug').map_label); save(hz, 'hazard_by_cohort_cutphase.csv')
lbm = lb.merge(mt[['match_id', 'match_type']], on='match_id', how='left')
hzm = lbm[lbm.cut_phase.between(1, 7)].groupby(['cohort', 'match_type', 'cut_phase']).agg(n_matches=('risk', lambda x: int((x > 0).sum())), risk=('risk', 'sum'), pos=('pos', 'sum')).reset_index()
hzm['hazard'] = hzm.pos / hzm.risk; save(hzm, 'hazard_by_cohort_matchtype_cutphase.csv')
print('HAZARD by cohort x match_type (cut 1..7):'); print(hzm.pivot_table(index=['cohort', 'match_type'], columns='cut_phase', values='hazard').round(4).to_string())
print('RISK by cohort x match_type (cut 1..7):'); print(hzm.pivot_table(index=['cohort', 'match_type'], columns='cut_phase', values='risk').astype(int).to_string())
print('HAZARD (pos/risk) cut 1..7:'); print(hz[hz.cut_phase <= 7].pivot(index='cohort', columns='cut_phase', values='hazard').loc[order].round(4).to_string())
print('RISK SET size cut 1..7:'); print(hz[hz.cut_phase <= 7].pivot(index='cohort', columns='cut_phase', values='risk').loc[order].astype(int).to_string())
print('POSITIVES cut 1..7:'); print(hz[hz.cut_phase <= 7].pivot(index='cohort', columns='cut_phase', values='pos').loc[order].astype(int).to_string())

# ---------- 4./5. state + combat aggregates ----------
st = qchunk("""SELECT match_id, phase_idx,
      count(*) FILTER (WHERE alive_players_start > 0)::bigint AS teams_start, coalesce(sum(alive_players_start) FILTER (WHERE alive_players_start > 0), 0)::bigint AS players_start,
      coalesce(sum(alive_players_start*(alive_players_start-1)/2) FILTER (WHERE alive_players_start > 0), 0)::bigint AS tm_edges_start,
      count(*) FILTER (WHERE alive_players_end > 0)::bigint AS teams_end, coalesce(sum(alive_players_end) FILTER (WHERE alive_players_end > 0), 0)::bigint AS players_end,
      coalesce(sum(alive_players_end*(alive_players_end-1)/2) FILTER (WHERE alive_players_end > 0), 0)::bigint AS tm_edges_end,
      count(*) FILTER (WHERE alive_players_start > 0 AND position_rows = 0)::bigint AS teams_nopos, count(*) FILTER (WHERE alive_players_start > 0 AND position_rows < 3)::bigint AS teams_lowpos,
      coalesce(sum(position_rows) FILTER (WHERE alive_players_start > 0), 0)::bigint AS pos_rows
    FROM stdb.team_phase_state WHERE match_id = ANY(%s) GROUP BY match_id, phase_idx""", all_ids)
st['cohort'] = st.match_id.map(id2coh); log('team_phase_state agg rows', len(st))
cb = qchunk("""SELECT match_id, phase_idx, coalesce(sum(kills_dealt), 0)::bigint AS kills, coalesce(sum(groggy_dealt), 0)::bigint AS groggy, coalesce(sum(damage_events_dealt), 0)::bigint AS dmg_events,
      coalesce(sum(damage_dealt), 0)::float8 AS dmg_sum FROM stdb.team_phase_combat WHERE match_id = ANY(%s) GROUP BY match_id, phase_idx""", all_ids)
cb['cohort'] = cb.match_id.map(id2coh); log('team_phase_combat agg rows', len(cb))
def per_phase(df, cols, name):
    d = df[df.phase_idx.between(1, 12)].groupby(['cohort', 'phase_idx'])[cols].agg('mean').reset_index(); n = df[df.phase_idx.between(1, 12)].groupby(['cohort', 'phase_idx']).match_id.nunique().rename('n_matches').reset_index()
    d = d.merge(n, on=['cohort', 'phase_idx']); save(d, name); return d
ev_p = per_phase(cb, ['kills', 'groggy', 'dmg_events', 'dmg_sum'], 'events_per_phase_by_cohort_teamlevel.csv')
s2_p = per_phase(st, ['teams_start', 'players_start', 'tm_edges_start', 'teams_end', 'players_end', 'tm_edges_end'], 'state_table_snapshot_counts_per_phase_by_cohort.csv')
def per_match_tot(df, cols, lo, hi, name):
    t = df[df.phase_idx.between(lo, hi)].groupby(['cohort', 'match_id'])[cols].sum().reset_index(); rows = []
    for c in order:
        g = t[t.cohort == c]; r = dict(cohort=c, n=len(g))
        for col in cols: r.update(stats(g[col], col + '_')); 
        rows.append(r)
    o = pd.DataFrame(rows); save(o, name); return o
ev_m = per_match_tot(cb, ['kills', 'groggy', 'dmg_events', 'dmg_sum'], 1, 12, 'events_per_match_by_cohort_teamlevel.csv')
s2_m = per_match_tot(st, ['players_start', 'tm_edges_start', 'teams_start', 'players_end'], 1, 7, 'state_table_snapshot_counts_per_match_k1to7_by_cohort.csv')
print('EVENTS per match (team-level enemy-directed, all phases) mean/median:'); print(ev_m[['cohort', 'n', 'kills_mean', 'kills_med', 'groggy_mean', 'groggy_med', 'dmg_events_mean', 'dmg_events_med', 'dmg_events_p10', 'dmg_events_p90', 'dmg_sum_mean', 'dmg_sum_med']].round(1).to_string(index=False))
print('STATE-TABLE snapshot counts (NOT true alive) per match (sum over phases 1..7) players_start / tm_edges_start / teams_start:'); print(s2_m[['cohort', 'n', 'players_start_mean', 'players_start_med', 'players_start_p10', 'players_start_p90', 'tm_edges_start_mean', 'tm_edges_start_med', 'teams_start_mean', 'players_end_mean']].round(1).to_string(index=False))
print('STATE-TABLE snapshot counts (NOT true alive) players at phase start (mean per match) cohort x phase:'); print(s2_p.pivot(index='cohort', columns='phase_idx', values='players_start').loc[order].iloc[:, :9].round(1).to_string())
print('STATE-TABLE snapshot counts (NOT true alive) teammate edges at phase start (mean) cohort x phase:'); print(s2_p.pivot(index='cohort', columns='phase_idx', values='tm_edges_start').loc[order].iloc[:, :9].round(1).to_string())
print('EVENTS per phase mean (team-level) kills:'); print(ev_p.pivot(index='cohort', columns='phase_idx', values='kills').loc[order].iloc[:, :9].round(2).to_string())
print('EVENTS per phase mean dmg_events:'); print(ev_p.pivot(index='cohort', columns='phase_idx', values='dmg_events').loc[order].iloc[:, :9].round(1).to_string())

# ---------- 6. quality (state-based missing positions) ----------
qm = st[st.phase_idx.between(1, 7)].groupby(['cohort', 'phase_idx'])[['teams_start', 'teams_nopos', 'teams_lowpos']].sum().reset_index()
qm['nopos_share'] = qm.teams_nopos / qm.teams_start; qm['lowpos_share'] = qm.teams_lowpos / qm.teams_start; save(qm, 'quality_team_position_rows_state_by_cohort_phase.csv')
tot = st[st.phase_idx.between(1, 7)][['teams_start', 'teams_nopos', 'teams_lowpos']].sum()
log('team_phase_state (phase 1..7, alive at start): team-phase rows', int(tot.teams_start), 'no position rows', int(tot.teams_nopos), 'share', round(tot.teams_nopos / tot.teams_start, 5), '| <3 rows', int(tot.teams_lowpos), 'share', round(tot.teams_lowpos / tot.teams_start, 5))
print(qm.pivot(index='cohort', columns='phase_idx', values='nopos_share').loc[order].round(4).to_string())
log('DONE')
