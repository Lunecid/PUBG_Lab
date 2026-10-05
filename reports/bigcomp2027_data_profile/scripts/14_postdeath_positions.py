#!/usr/bin/env python3
"""Do telem_positions rows continue after a player's kill event?  (player level, human players with one kill event).  READ-ONLY."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import dbh
C = '/Users/lunecid/project/PUBG/reports/bigcomp2027_data_profile/csv'; pd.set_option('display.width', 220)
m = pd.read_csv(f'{C}/sample_match_level.csv'); rng = np.random.RandomState(21); ids = []
for co, g in m.groupby('cohort'): ids += rng.choice(g.match_id.values, 6, replace=False).tolist()
cur = dbh.conn().cursor()
def q(sql, p): cur.execute(sql, p); return pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
rows = []
for i in range(0, len(ids), 13):
    b = ids[i:i + 13]
    S = q("SELECT match_id, extract(epoch from started_at)::float8 AS s_ep FROM stdb.matches WHERE match_id = ANY(%s)", (b,)).set_index('match_id').s_ep
    kil = q("SELECT match_id, victim_id, min(extract(epoch from event_time))::float8 AS ep1, max(extract(epoch from event_time))::float8 AS ep, count(*) AS n_ev FROM pubg.telem_kills WHERE match_id = ANY(%s) GROUP BY 1,2", (b,))
    pos = q("SELECT match_id, account_id AS acc, max(extract(epoch from event_time))::float8 AS lp, count(*) AS n_rows, count(*) FILTER (WHERE coalesce(health,0) <= 0) AS n_zero FROM pubg.telem_positions WHERE match_id = ANY(%s) AND is_game >= 0.1 AND account_id IS NOT NULL GROUP BY 1,2", (b,))
    pa = q("SELECT p.match_id, p.player_id, p.death_type, p.time_survived::float8 AS ts FROM pubg.participants p WHERE p.match_id = ANY(%s)", (b,))
    x = kil.merge(pos, left_on=['match_id', 'victim_id'], right_on=['match_id', 'acc'], how='left').merge(pa, left_on=['match_id', 'victim_id'], right_on=['match_id', 'player_id'], how='left'); x['S'] = x.match_id.map(S); rows.append(x)
x = pd.concat(rows); x = x[(~x.victim_id.fillna('').str.startswith('ai.')) & (x.n_ev == 1) & x.lp.notna()].copy(); x['gap_pos_minus_kill'] = x.lp - x.ep; x['gap_pos_minus_ts'] = x.lp - x.S - x.ts
q_ = lambda s: s.quantile([.01, .05, .25, .5, .75, .95, .99]).round(2).to_dict()
print('human players with exactly 1 kill event and positions:', len(x)); print('last position epoch - kill epoch (s):', q_(x.gap_pos_minus_kill)); print('share last position AFTER kill by >2 s:', round(float((x.gap_pos_minus_kill > 2).mean()), 4), '| by >10 s:', round(float((x.gap_pos_minus_kill > 10).mean()), 4), '| before kill by >2 s:', round(float((x.gap_pos_minus_kill < -2).mean()), 4))
print('last position epoch - S - time_survived (s):', q_(x.gap_pos_minus_ts)); print('players whose LAST position row has health 0 is not measured here; share of players with any zero-health rows:', round(float((x.n_zero > 0).mean()), 4), '| zero rows per player median/p90:', x.n_zero.quantile([.5, .9]).to_dict())
x.to_csv(f'{C}/label_audit_postdeath_positions.csv', index=False)
