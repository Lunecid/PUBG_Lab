#!/usr/bin/env python3
"""Check: stdb.team_phase_state.alive_players_* (count at one team-second snapshot) vs true alive players per team.  READ-ONLY."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import dbh
C = '/Users/lunecid/project/PUBG/reports/bigcomp2027_data_profile/csv'
m = pd.read_csv(f'{C}/sample_match_level.csv'); cut = pd.read_csv(f'{C}/sample_cut_level.csv'); rng = np.random.RandomState(7); ids = []
for co in ['erangel_tpp_squad', 'taego_tpp_squad', 'rondo_tpp_squad']: ids += rng.choice(m[m.cohort == co].match_id.values, 20, replace=False).tolist()
cur = dbh.conn().cursor()
cur.execute("""SELECT match_id, team_id, elapsed_time AS t, count(DISTINCT account_id) AS n FROM pubg.telem_positions WHERE match_id = ANY(%s) AND is_game >= 0.1 AND account_id IS NOT NULL AND team_id IS NOT NULL
               AND pos_x IS NOT NULL AND pos_y IS NOT NULL AND elapsed_time IS NOT NULL AND coalesce(health, 0) > 0 GROUP BY 1,2,3""", (ids,))
d = pd.DataFrame(cur.fetchall(), columns=['match_id', 'team_id', 't', 'n'])
dist = d.n.value_counts(normalize=True).sort_index().round(4).to_dict()
print('distinct healthy players with a position row per (team, elapsed second): share by count', dist, '| mean', round(d.n.mean(), 3), '| rows', len(d), '| matches', len(ids))
cur.execute("""SELECT phase_idx, count(*) AS n_rows, avg(alive_players_start) AS a_start, avg(alive_players_end) AS a_end, avg(alive_players_min) AS a_min, avg(alive_players_avg) AS a_avg,
               avg((alive_players_start = 1)::int) AS share_start_eq1, avg(team_size) AS team_size FROM stdb.team_phase_state WHERE match_id = ANY(%s) AND alive_players_start > 0 AND phase_idx BETWEEN 1 AND 7 GROUP BY 1 ORDER BY 1""", (ids,))
st = pd.DataFrame(cur.fetchall(), columns=['phase_idx', 'n_rows', 'state_alive_start_mean', 'state_alive_end_mean', 'state_alive_min_mean', 'state_alive_avg_mean', 'share_start_eq1', 'team_size_mean'])
tr = cut[cut.match_id.isin(ids)].groupby('k').apply(lambda g: pd.Series(dict(true_alive_per_team_start=g.nodes_A.sum() / g.teams_A.sum(), true_alive_per_team_at_cut=g.nodes_C.sum() / g.teams_C.sum()))).reset_index().rename(columns={'k': 'phase_idx'})
out = st.merge(tr, on='phase_idx'); out['ratio_true_over_state_start'] = out.true_alive_per_team_start / out.state_alive_start_mean; out.to_csv(f'{C}/alive_definition_check.csv', index=False); pd.set_option('display.width', 250); print(out.round(3).to_string(index=False))
pd.DataFrame([dict(share_1=dist.get(1, 0), share_2=dist.get(2, 0), share_3=dist.get(3, 0), share_4=dist.get(4, 0), mean=d.n.mean(), rows=len(d), matches=len(ids))]).to_csv(f'{C}/alive_definition_check_team_second.csv', index=False)
