"""Read-only Erangel patch inventory and bounded, pseudonymous spatial probe.

The 40.2 cohort is inferred from Steam live maintenance dates, not a stored
engine build. Position filters do not certify foot-only or ground contact.
"""
import csv
import getpass
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import psycopg2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/erangel_alignment_2026-10-15'


def main():
    OUT.mkdir(exist_ok=True)
    conn = psycopg2.connect(
        host='localhost', dbname='pubg_survival', user='postgres',
        password=getpass.getpass('DB password: '), connect_timeout=5,
        options='-c default_transaction_read_only=on -c statement_timeout=30000')
    points, pairs, matches = [], [], []
    totals = Counter()
    with conn, conn.cursor() as cur:
        cur.execute('SHOW transaction_read_only')
        assert cur.fetchone()[0] == 'on'
        cur.execute("""SELECT m.match_id,m.map_name,m.game_mode,m.match_type,m.shard_id,
            m.created_at,s.event_time,s.weather_id,s.is_event_mode
            FROM pubg.matches m LEFT JOIN pubg.telem_match_start s USING(match_id)
            WHERE m.map_name IN ('Baltic_Main','Erangel_Main') ORDER BY m.match_id""")
        population = cur.fetchall()
        cols = ['match_id','map','mode','match_type','shard','created_at','start_at','weather','event_mode']
        with (OUT/'erangel_match_inventory.csv').open('w', newline='') as f:
            w = csv.writer(f); w.writerow(cols); w.writerows(population)
        eligible = [r for r in population if r[1:5] == ('Baltic_Main','squad','official','steam')
                    and r[6] is not None and r[8] is False
                    and '2026-03-11T08:30' <= r[6].astimezone(__import__('datetime').timezone.utc).isoformat() < '2026-04-08T00:00']
        eligible.sort(key=lambda r: hashlib.sha256(('erangel-align-40.2-v1:'+r[0]).encode()).hexdigest())
        assert len(eligible) >= 40
        for mi, row in enumerate(eligible[:40]):
            mid = row[0]
            cur.execute('SELECT player_id FROM pubg.participants WHERE match_id=%s', (mid,))
            humans = {a for (a,) in cur.fetchall() if a and a.startswith('account.')}
            cur.execute('SELECT account_id,min(event_time) FROM pubg.telem_parachute_landing WHERE match_id=%s GROUP BY 1', (mid,))
            land = dict(cur.fetchall())
            cur.execute('SELECT victim_id,min(event_time) FROM pubg.telem_kills WHERE match_id=%s GROUP BY 1', (mid,))
            death = dict(cur.fetchall())
            cur.execute('''SELECT account_id,event_time,pos_x::float8,pos_y::float8,pos_z::float8,
                health::float8,vehicle_type,vehicle_id FROM pubg.telem_positions
                WHERE match_id=%s ORDER BY account_id,event_time,id''', (mid,))
            seq = cur.fetchall()
            dup = Counter((r[0],r[1]) for r in seq)
            prev = None
            local = Counter()
            for a,t,x,y,z,health,v,vid in seq:
                if a not in humans:
                    prev = None; continue
                local['raw_human_points'] += 1
                if (t is None or any(q is None or not np.isfinite(q) for q in [x,y,z])
                    or dup[a,t] != 1 or a not in land or t < land[a]
                    or (a in death and t >= death[a]) or v or vid
                    or health is None or health <= 0 or not (0<=x<816000 and 0<=y<816000)):
                    prev = None; continue
                idx = len(points)
                points.append((mi,(t-row[6]).total_seconds(),x/100,y/100,z/100))
                local['candidate_points'] += 1
                if prev and prev[0] == a:
                    dt=(t-prev[1]).total_seconds();distance=np.hypot(x/100-prev[2],y/100-prev[3])
                    if 5<=dt<=15 and distance/dt<=12:
                        pairs.append((prev[4],idx,dt,distance))
                        local['candidate_pairs'] += 1
                prev=(a,t,x/100,y/100,idx)
            matches.append(dict(match_index=mi,match_id=mid,split='fit' if mi<32 else 'held_out',
                                start_at=row[6].isoformat(),weather=row[7],**local))
            totals.update(local)
            print(f'{mi+1}/40: {local["candidate_points"]} points',flush=True)
    conn.close()
    np.savez_compressed(OUT/'telemetry_probe.npz',points=np.asarray(points,dtype=np.float64),
                        pairs=np.asarray(pairs,dtype=np.float64))
    (OUT/'extraction.json').write_text(json.dumps(dict(
        population_matches=len(population),eligible_40_2_squad_matches=len(eligible),
        patch_status='inferred_from_UTC_start_and_official_PC_schedule_not_exact_build',
        filters='human prefix; landing to first kill; alive; no recorded vehicle; finite XYZ; unique timestamp; dt 5..15s; speed <=12m/s',
        caveat='not certified foot-only or ground contact; no interpolation; no account IDs exported',
        point_columns=['match_index','seconds_since_start','x_m','y_m','player_z_m'],
        pair_columns=['source_point','target_point','dt_s','displacement_m'],
        totals=dict(totals),matches=matches),indent=2))


if __name__ == '__main__':
    main()
