#!/usr/bin/env python3
"""Accounts present in telem_positions but absent from pubg.participants (Vikendi anomaly follow-up).  READ-ONLY."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import dbh
C = '/Users/lunecid/project/PUBG/reports/bigcomp2027_data_profile/csv'
m = pd.read_csv(f'{C}/sample_match_level.csv'); cur = dbh.conn().cursor(); rng = np.random.RandomState(5); rows = []
for co in ['vikendi_tpp_squad', 'erangel_tpp_squad', 'sanhok_tpp_squad', 'miramar_desert_tpp_squad']:
    for mid in rng.choice(m[m.cohort == co].match_id.values, 10, replace=False).tolist():
        cur.execute("""SELECT DISTINCT p.account_id, p.team_id FROM pubg.telem_positions p WHERE p.match_id = %s AND p.is_game >= 0.1 AND p.account_id IS NOT NULL
                       AND NOT EXISTS (SELECT 1 FROM pubg.participants pp WHERE pp.match_id = p.match_id AND pp.player_id = p.account_id)""", (mid,)); ex = cur.fetchall()
        cur.execute("SELECT DISTINCT r.team_id FROM pubg.rosters r WHERE r.match_id = %s", (mid,)); pteams = {x[0] for x in cur.fetchall()}
        cur.execute("SELECT count(DISTINCT player_id), count(*) FILTER (WHERE left(player_id, 3) = 'ai.') FROM pubg.participants WHERE match_id = %s", (mid,)); np_, nai = cur.fetchone()
        ids = [x[0] for x in ex]; nd = 0; nk = 0
        if ids:
            cur.execute("SELECT count(*) FROM pubg.telem_damage WHERE match_id = %s AND (attacker_id = ANY(%s) OR victim_id = ANY(%s))", (mid, ids, ids)); nd = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM pubg.telem_kills WHERE match_id = %s AND (killer_id = ANY(%s) OR victim_id = ANY(%s))", (mid, ids, ids)); nk = cur.fetchone()[0]
        rows.append(dict(cohort=co, match_id=mid, n_participants=np_, n_participants_ai=nai, n_extra_accounts=len(ids), extra_ai_prefix=sum(1 for i in ids if i.startswith('ai.')), extra_team_ids=len({x[1] for x in ex}),
                         extra_team_in_roster=sum(1 for x in ex if x[1] in pteams), extra_damage_rows=nd, extra_kill_rows=nk, example_prefix=(ids[0][:8] if ids else '')))
d = pd.DataFrame(rows); d.to_csv(f'{C}/nonparticipant_accounts_check.csv', index=False); pd.set_option('display.width', 250)
print(d.groupby('cohort')[['n_participants', 'n_participants_ai', 'n_extra_accounts', 'extra_ai_prefix', 'extra_team_ids', 'extra_team_in_roster', 'extra_damage_rows', 'extra_kill_rows']].median().round(1).to_string())
print(d[d.cohort == 'vikendi_tpp_squad'].drop(columns='match_id').to_string(index=False))
