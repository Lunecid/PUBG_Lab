#!/usr/bin/env python3
"""Vikendi anomaly check: position rows per match, duplicate rows, stdb.team_phase_combat damage vs raw.  READ-ONLY."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import dbh
C = '/Users/lunecid/project/PUBG/reports/bigcomp2027_data_profile/csv'
m = pd.read_csv(f'{C}/sample_match_level.csv'); h = pd.read_csv(f'{C}/sample_position_interval_hist.csv'); e = pd.read_csv(f'{C}/sample_events_by_phase.csv')
pd.set_option('display.width', 250)
print('median rows per match by cohort:'); print(m.groupby('cohort')[['pos_rows', 'pos_players', 'dmg_rows']].median().round(0).to_string())
for co in ['vikendi_tpp_squad', 'erangel_tpp_squad']:
    hh = h[h.cohort == co].set_index('interval_s').n; hh = hh / hh.sum(); print(co, 'interval share >0.2%:', hh[hh > 0.002].round(4).to_dict())
rng = np.random.RandomState(3); cur = dbh.conn().cursor(); rows = []
es = e.groupby('match_id').dmg_sum_enemy.sum()
for co in ['vikendi_tpp_squad', 'erangel_tpp_squad', 'taego_tpp_squad']:
    for mid in rng.choice(m[m.cohort == co].match_id.values, 8, replace=False).tolist():
        cur.execute("SELECT count(*), count(DISTINCT (account_id, elapsed_time)), count(DISTINCT account_id), count(DISTINCT id) FROM pubg.telem_positions WHERE match_id=%s AND is_game>=0.1 AND account_id IS NOT NULL", (mid,)); a = cur.fetchone()
        cur.execute("SELECT count(*), count(DISTINCT (attacker_id, victim_id, event_time, damage)), count(DISTINCT id) FROM pubg.telem_damage WHERE match_id=%s", (mid,)); b = cur.fetchone()
        cur.execute("SELECT coalesce(sum(damage_dealt),0)::float8, coalesce(sum(damage_events_dealt),0) FROM stdb.team_phase_combat WHERE match_id=%s", (mid,)); c = cur.fetchone()
        r = m[m.match_id == mid].iloc[0]; rows.append(dict(cohort=co, pos_rows=a[0], pos_distinct_acc_t=a[1], pos_players=a[2], pos_distinct_id=a[3], dmg_rows=b[0], dmg_distinct_tuple=b[1], dmg_distinct_id=b[2], raw_enemy_dmg=es.get(mid, np.nan), participants_dmg=r.p_damage, stdb_dmg=c[0], stdb_dmg_events=c[1]))
d = pd.DataFrame(rows); d['pos_dup_ratio'] = d.pos_rows / d.pos_distinct_acc_t; d['dmg_dup_ratio'] = d.dmg_rows / d.dmg_distinct_tuple; d['stdb_over_raw'] = d.stdb_dmg / d.raw_enemy_dmg; d['rows_per_player'] = d.pos_rows / d.pos_players
d.to_csv(f'{C}/vikendi_anomaly_check_matches.csv', index=False)
print(d.groupby('cohort')[['pos_rows', 'pos_dup_ratio', 'rows_per_player', 'dmg_rows', 'dmg_dup_ratio', 'raw_enemy_dmg', 'participants_dmg', 'stdb_dmg', 'stdb_over_raw']].median().round(3).to_string())
print(d[d.cohort == 'vikendi_tpp_squad'][['pos_rows', 'pos_distinct_acc_t', 'pos_distinct_id', 'dmg_rows', 'dmg_distinct_tuple', 'raw_enemy_dmg', 'participants_dmg', 'stdb_dmg']].round(0).to_string(index=False))
