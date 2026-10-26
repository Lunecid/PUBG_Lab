"""Bounded read-only spatial support audit, not a walkability or learning test.

40 deterministically selected matches from an existing purposive profile;
32 reference matches and 8 held-out matches. No interpolation or training.
"""
import csv
import getpass
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import psycopg2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/spatial_feasibility_2026-10-15'


def main():
    rows = list(csv.DictReader((ROOT / 'reports/bigcomp2027_data_profile/csv/sample_match_level.csv').open()))
    candidates = [r['match_id'] for r in rows if r['cohort'] == 'erangel_tpp_squad' and r['match_type'] == 'official']
    mids = sorted(set(candidates), key=lambda x: hashlib.sha256(('spatial-probe-v1:' + x).encode()).hexdigest())[:40]
    assert len(mids) == 40
    password = getpass.getpass('DB password: ')
    conn = psycopg2.connect(host='localhost', dbname='pubg_survival', user='postgres', password=password,
                           connect_timeout=5, options='-c default_transaction_read_only=on -c statement_timeout=30000')
    del password
    resolutions = [25, 50, 100]
    cell_matches = {s: defaultdict(set) for s in resolutions}
    edge_matches = {s: defaultdict(set) for s in resolutions}
    held = {s: [] for s in resolutions}
    counts = Counter()
    per_match = []
    displacements = []
    gaps = []
    with conn, conn.cursor() as cur:
        cur.execute('SHOW transaction_read_only')
        assert cur.fetchone()[0] == 'on'
        for mi, mid in enumerate(mids):
            cur.execute('SELECT player_id FROM pubg.participants WHERE match_id=%s', (mid,))
            humans = {a for (a,) in cur.fetchall() if a and a.startswith('account.')}
            cur.execute('SELECT account_id,min(event_time) FROM pubg.telem_parachute_landing WHERE match_id=%s GROUP BY 1', (mid,))
            landing = dict(cur.fetchall())
            cur.execute('SELECT victim_id,min(event_time) FROM pubg.telem_kills WHERE match_id=%s GROUP BY 1', (mid,))
            kills = dict(cur.fetchall())
            cur.execute('''SELECT account_id,event_time,pos_x::float8,pos_y::float8,pos_z::float8,
                                  health::float8,vehicle_type,vehicle_id
                           FROM pubg.telem_positions WHERE match_id=%s
                           ORDER BY account_id,event_time,id''', (mid,))
            tracks = defaultdict(list)
            for a,t,x,y,z,h,v,vid in cur.fetchall():
                if a in humans:
                    tracks[a].append((t,x,y,z,h,v,vid))
            local = Counter(humans=len(humans))
            split = 'reference' if mi < 32 else 'held_out'
            for a, seq in tracks.items():
                prev = None
                timecounts = Counter(p[0] for p in seq)
                for p in seq:
                    t,x,y,z,h,v,vid = p
                    local['raw_human_rows'] += 1
                    reason = None
                    if t is None or any(q is None or not math.isfinite(q) for q in [x,y]): reason='invalid_xy_time'
                    elif timecounts[t] > 1: reason='duplicate_timestamp'
                    elif a not in landing or t < landing[a]: reason='before_or_missing_landing'
                    elif a in kills and t >= kills[a]: reason='at_or_after_first_kill'
                    elif v or vid: reason='vehicle_recorded'
                    elif h is None or h <= 0: reason='health_nonpositive_or_missing'
                    elif not (0 <= x < 816000 and 0 <= y < 816000): reason='outside_documented_extent'
                    if reason:
                        local['excluded_' + reason] += 1
                        prev = None
                        continue
                    x,y = x/100, y/100
                    local['candidate_points'] += 1
                    for s in resolutions:
                        cell=(int(x//s),int(y//s))
                        if split == 'reference': cell_matches[s][cell].add(mid)
                    if prev:
                        pt,px,py = prev
                        dt=(t-pt).total_seconds()
                        d=math.hypot(x-px,y-py)
                        if 5 <= dt <= 15 and d/dt <= 12:
                            local['candidate_transitions'] += 1
                            displacements.append(d)
                            gaps.append(dt)
                            for s in resolutions:
                                edge=((int(px//s),int(py//s)),(int(x//s),int(y//s)))
                                if split == 'reference': edge_matches[s][edge].add(mid)
                                else: held[s].append((mid,edge))
                        else: local['excluded_pair_gap_or_speed'] += 1
                    prev=(t,x,y)
            counts.update(local)
            per_match.append(dict(match_id=mid, split=split, **local))
            print(f'{mi+1}/40 {split}: points={local["candidate_points"]}, transitions={local["candidate_transitions"]}', flush=True)
    conn.close()
    def percentile(values, p):
        a=sorted(values)
        return a[min(int((len(a)-1)*p),len(a)-1)] if a else None
    metrics=[]
    for s in resolutions:
        cm,em=cell_matches[s],edge_matches[s]
        # Exclude self-transitions when evaluating connectivity support.
        hv=[(mid,e) for mid,e in held[s] if e[0] != e[1]]
        distinct_edges={e for _,e in hv}
        metric=dict(cell_size_m=s, reference_visited_cells=len(cm),
                    reference_cells_in_at_least_3_matches=sum(len(v)>=3 for v in cm.values()),
                    reference_directed_nonself_edges=sum(e[0]!=e[1] for e in em),
                    reference_nonself_edges_in_at_least_3_matches=sum(e[0]!=e[1] and len(v)>=3 for e,v in em.items()),
                    heldout_nonself_transitions=len(hv), heldout_unique_nonself_edges=len(distinct_edges))
        for label,fn in [('both_cells_seen',lambda e: e[0] in cm and e[1] in cm),
                         ('directed_edge_seen',lambda e: e in em),
                         ('either_direction_edge_seen',lambda e: e in em or (e[1],e[0]) in em),
                         ('directed_edge_in_3_matches',lambda e: len(em.get(e,()))>=3)]:
            metric[label+'_micro_share']=sum(fn(e) for _,e in hv)/len(hv) if hv else None
            bymatch=defaultdict(list)
            for mid,e in hv: bymatch[mid].append(fn(e))
            metric[label+'_match_macro_share']=sum(sum(v)/len(v) for v in bymatch.values())/len(bymatch) if bymatch else None
        metrics.append(metric)
    result=dict(status='READ_ONLY_SUPPORT_PROBE_NOT_TRAINING',date='2026-10-15',
                sampling='40 hash-selected from 171 saved purposive-profile candidates; not random sample of entire DB',
                scope='Baltic_Main erangel_tpp_squad official; account. heuristic humans',
                filters='first landing to first kill; positive health; no recorded vehicle; in extent; drop all repeated timestamps; pair dt 5..15s and speed<=12m/s',
                limitations=['not verified foot-only','no raw-event redeploy/swim recovery','map patch identity unverified',
                             '12m/s is probe cutoff not game physics','support is not traversability or model accuracy',
                             'reference/held-out split is diagnostic only, not final benchmark split'],
                counts=dict(counts),displacement_m_quantiles={str(p):percentile(displacements,p) for p in [.1,.5,.9,.99]},
                interval_s_quantiles={str(p):percentile(gaps,p) for p in [.1,.5,.9,.99]},
                resolutions=metrics, matches=per_match)
    OUT.mkdir(exist_ok=True)
    (OUT/'telemetry_support_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='matches'},indent=2))


if __name__ == '__main__':
    main()
