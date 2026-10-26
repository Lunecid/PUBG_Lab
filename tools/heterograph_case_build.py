"""Causal as-of replay of a single real match; retrospective selection is explicit.
Run after heterograph_case_extract.py. No DB access or credential use here.
"""
import csv, hashlib, json, math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/heterograph_case_40_2_2026-10-15'
MEMBERS=['P1','P2','P3','P4']; FRESH=15.;CONTACT=30.
TERRAIN={int(r['node_id']):{k:r[k] for k in ('historical_height_proxy_m','historical_land_fraction','known_change_buffer','exact_patch_match','traversability_verified')} for r in csv.DictReader((ROOT/'reports/erangel_100m_40_2_2026-10-15/nodes.csv').open())}

def short(item):
    return (item or '?').removeprefix('Item_Weapon_').removeprefix('Item_').removesuffix('_C')

def load():
    src=json.loads((OUT/'source_events_server_audit.json').read_text());start=datetime.fromisoformat(src['metadata']['start_at']);tables={};dups={}
    for table,rows in src['tables'].items():
        seen=set();tables[table]=[]
        for original in rows:
            r=dict(original);r['t']=round((datetime.fromisoformat(r['event_time'])-start).total_seconds(),3)
            key=json.dumps({k:v for k,v in r.items() if k!='id'},sort_keys=True)
            if key in seen:continue
            seen.add(key);tables[table].append(r)
        tables[table].sort(key=lambda r:r['t']);dups[table]=len(rows)-len(tables[table])
    return src,tables,dups

def evidence(table,row,t,scope='self',kind='recorded'):
    return {'source':table,'source_id':row['id'],'source_t':row['t'],'age_s':round(t-row['t'],3),'scope':scope,'kind':kind}

def equipment(rows,p,t):
    groups=defaultdict(list)
    for r in rows:
        if r['account_id']==p and r['t']<=t and (r['category']=='Weapon' or r['sub_category'] in ('Headgear','Vest','Throwable')):groups[r['t'],r['item_id']].append(r)
    state={}
    for (ts,item),events in sorted(groups.items()):
        actions={r['event_type'] for r in events}
        if actions=={'equip','unequip'}:state[item]={'item':item,'label':short(item),'status':'ambiguous_same_time','source_t':ts,'source_ids':sorted(r['id'] for r in events),'category':events[0]['category']}
        elif actions=={'equip'}:state[item]={'item':item,'label':short(item),'status':'last_equip_not_reversed','source_t':ts,'source_ids':sorted(r['id'] for r in events),'category':events[0]['category']}
        elif actions=={'unequip'}:state.pop(item,None)
    return list(state.values())

def snapshot(t,tables):
    players=[];nodes=[];edges=[]
    def node(id,type,label,kind='derived',**extra):
        nodes.append(dict(id=id,type=type,label=label,kind=kind,**extra))
    def edge(a,b,type,kind='derived',**extra):
        previous=next((e for e in edges if (e['source'],e['target'],e['type'])==(a,b,type)),None)
        if previous is not None:
            if type=='occupies':previous['weight']+=extra.get('weight',1);previous['via'].extend(extra.get('via',[]))
            return
        edges.append(dict(source=a,target=b,type=type,kind=kind,**extra))
    node('team','team','우리 팀');node('game','game','경기 구조','recorded')
    # Region nodes connect to global game context; no extra latent context node.
    for p in MEMBERS:
        pos=[r for r in tables['positions'] if r['account_id']==p and r['t']<=t]
        # Multiple conflicting positions at the latest timestamp are unknown.
        last=pos[-1] if pos else None
        conflict=bool(last and len([r for r in pos if r['t']==last['t']])>1)
        kills=[r for r in tables['kills'] if r['victim_id']==p and r['t']<=t]
        death=kills[-1] if kills else None
        # Positive post-death evidence would flag an unresolved respawn; never silently reuse dead state.
        respawn=bool(death and last and last['t']>death['t'] and (last['health'] or 0)>0)
        dead=bool(death and not respawn)
        landings=[r for r in tables['parachute_landing'] if r['account_id']==p and r['t']<=t]
        land=bool(landings)
        ground_sample=bool(landings and last and last['t']>=landings[-1]['t'])
        age=round(t-last['t'],3) if last else None
        fresh=bool(last and 0<=age<=FRESH and not conflict)
        spatial=bool(fresh and ground_sample and not dead and not respawn and all(last[k] is not None for k in ('pos_x','pos_y')) and 0<=last['pos_x']<816000 and 0<=last['pos_y']<816000)
        scope='self' if p=='P1' else 'teammate_hud_proxy'
        equip=equipment(tables['item_equip'],p,t) if p=='P1' else None
        state={'id':p,'age_s':age,'fresh':fresh,'landed':land,'dead_recorded':dead,'respawn_ambiguous':respawn,'position_valid':spatial,'xy_m':[round(last['pos_x']/100,2),round(last['pos_y']/100,2)] if spatial else None,'health_sample':last['health'] if fresh and not dead else None,'position_evidence':evidence('telem_positions',last,t,scope) if last else None,'equipment':equip,'equipment_scope':'self_event_reconstruction' if p=='P1' else 'unknown_to_observer','active_weapon_in_hand':None,'ammunition_total':None,'vehicle_class':last['vehicle_id'] if fresh and not dead else None,'vehicle_type':last['vehicle_type'] if fresh and not dead else None,'vehicle_scope':scope if p=='P1' else 'server_only','knock_evidence_age_s':None}
        knocks=[r for r in tables['groggy'] if r['victim_id']==p and r['t']<=t]
        if knocks:state['knock_evidence_age_s']=round(t-knocks[-1]['t'],3)
        # Other members' vehicle class is not guaranteed available in HUD.
        if p!='P1':state['vehicle_class']=None;state['vehicle_type']=None
        players.append(state)
        node(p,'player',p+(' 관측자' if p=='P1' else ''),'unknown' if not fresh or dead else 'recorded',scope=scope,source_t=last['t'] if last else None)
        edge(p,'team','member_of','recorded',scope='roster_assumed_known')
        if spatial:
            x,y=state['xy_m'];cell=f'C{int(x//100)}:{int(y//100)}';reg=f'R{int(x//400)}:{int(y//400)}'
            if not any(n['id']==cell for n in nodes):node(cell,'cell',cell+' · 100m',xy_m=[int(x//100)*100+50,int(y//100)*100+50],terrain_prior=TERRAIN[int(y//100)*82+int(x//100)],terrain_scope='historical assets; no match-derived occupancy counts or current geometry guarantee')
            if not any(n['id']==reg for n in nodes):node(reg,'region',reg+' · 400m');edge(reg,'game','spatial_context')
            edge(p,cell,'located_at',source_t=last['t'],scope=scope);edge(cell,reg,'within');edge('team',cell,'occupies',weight=1,via=[p])
        if p=='P1' and state['vehicle_class'] and state['vehicle_type']!='TransportAircraft':
            vid='vehicle:P1';node(vid,'vehicle','P1 차량 증거','recorded',vehicle_class=state['vehicle_class'],source_t=last['t']);edge(p,vid,'vehicle_sample','recorded',source_t=last['t'])
        for it in equip or []:
            id='gear:'+it['item'];node(id,'equipment',it['label'],'unknown' if it['status'].startswith('ambiguous') else 'derived',source_t=it['source_t'],status=it['status']);edge(p,id,'equip_history',source_t=it['source_t'])
    valid=[p for p in players if p['position_valid']]
    team={'valid_members':[p['id'] for p in valid],'known_deaths':[p['id'] for p in players if p['dead_recorded']],'center_m':None,'medoid_member':None,'max_pair_distance_m':None,'radius_m':None,'center_is_traversable':None}
    if valid:
        xy=np.array([p['xy_m'] for p in valid]);w=np.exp(-np.array([p['age_s'] for p in valid])/10);center=np.average(xy,axis=0,weights=w);dist=np.linalg.norm(xy[:,None]-xy[None,:],axis=2)
        team.update(center_m=np.round(center,2).tolist(),medoid_member=valid[int(np.argmin(dist.sum(axis=1)))]['id'],max_pair_distance_m=round(float(dist.max()),2),radius_m=round(float(np.linalg.norm(xy-center,axis=1).max()),2))
    damage=[r for r in tables['damage'] if t-CONTACT<r['t']<=t and r['victim_id']=='P1' and (r['damage'] or 0)>0]
    contacts=[]
    for typ in sorted({r['damage_type'] or 'Unknown' for r in damage}):
        group=[r for r in damage if (r['damage_type'] or 'Unknown')==typ]
        combat=typ in ('Damage_Gun','Damage_Explosion_Grenade','Damage_Explosion_Molotov')
        contact={'id':'contact:'+typ,'damage_type':typ,'count':len(group),'health_loss':round(sum(r['damage'] for r in group),2),'latest_t':max(r['t'] for r in group),'interpretation':'incoming_combat_proxy' if combat else 'damage_context','enemy_visible':None,'enemy_location':None,'attacker_identity':None,'friendly_or_enemy':None,'source_ids':[r['id'] for r in group]}
        contacts.append(contact);node(contact['id'],'contact','최근 피격' if combat else typ,'derived',source_t=contact['latest_t']);edge('P1',contact['id'],'received_damage_proxy',source_t=contact['latest_t'])
    gs=[r for r in tables['game_states'] if r['t']<=t];game=None
    if gs:
        g=gs[-1];age=t-g['t'];game={'source_t':g['t'],'age_s':round(age,3),'fresh':age<=15,'phase_raw':g['is_game'],'alive_players':g['num_alive_players'],'zones':[]}
        for name in ('safe_zone','poison_zone'):
            if g[name+'_radius'] and g[name+'_radius']>0:
                z={'id':name,'xy_m':[g[name+'_x']/100,g[name+'_y']/100],'radius_m':g[name+'_radius']/100,'source_t':g['t'],'semantic':'raw_api_field; map-visible-zone proxy, UI timing unverified'};game['zones'].append(z)
                node(name,'zone',name,'recorded' if age<=15 else 'unknown',source_t=g['t']);edge('game',name,'zone_sample','recorded',source_t=g['t']);edge('team',name,'zone_context')
    ownuse=[r for r in tables['item_use'] if r['account_id']=='P1' and t-30<r['t']<=t]
    recent_actions=[{'item':short(r['item_id']),'category':r['category'],'t':r['t'],'count_field':r['stack_count']} for r in ownuse]
    return {'t':t,'players':players,'team':team,'game':game,'contacts':contacts,'recent_item_use':recent_actions,'nodes':nodes,'edges':edges}

def main():
    src,tables,dups=load();end=max(r['t'] for r in tables['kills'] if r['victim_id']=='P1')
    times=set(float(t) for t in range(0,int(end)+1,5));times.add(end)
    milestones=[{'t':0.,'label':'경기 시작'}]
    for r in tables['parachute_landing']:
        if r['account_id']=='P1':milestones.append({'t':r['t'],'label':'P1 착지'})
    for r in tables['kills']:
        if r['victim_id'] in MEMBERS:milestones.append({'t':r['t'],'label':r['victim_id']+' 사망 기록'})
    lastv=None
    for r in tables['positions']:
        if r['account_id']!='P1' or r['t']<100:continue
        v=r['vehicle_id']
        if v!=lastv:
            milestones.append({'t':r['t'],'label':('차량 기록: '+v if v else '차량 없음 표본')});lastv=v
    incoming=[r for r in tables['damage'] if r['victim_id']=='P1' and r['damage_type']=='Damage_Gun' and r['damage']>0]
    milestones.append({'t':incoming[0]['t'],'label':'P1 첫 총기 피해 기록'})
    weapon=[r for r in tables['item_equip'] if r['account_id']=='P1' and r['category']=='Weapon']
    for r in weapon:milestones.append({'t':r['t'],'label':r['event_type']+' '+short(r['item_id'])})
    for m in milestones:times.add(m['t'])
    # Exact damage-event frames keep health-loss evidence visible without snapping backward.
    times.update(r['t'] for r in tables['damage'] if r['victim_id']=='P1' and 0<=r['t']<=end)
    times=sorted(t for t in times if 0<=t<=end);frames=[snapshot(t,tables) for t in times]
    milestones=sorted([m for m in milestones if 0<=m['t']<=end],key=lambda m:m['t'])
    ledger=[]
    for table,rows in tables.items():
        if table in ('positions','game_states','match_start'):continue
        for r in rows:
            if not 0<=r['t']<=end:continue
            p=r.get('account_id',r.get('victim_id'))
            if table in ('kills','groggy') and p not in MEMBERS:continue
            if table=='damage' and p!='P1':continue
            scope='self' if p=='P1' else 'teammate_hud_proxy' if table in ('kills','groggy','parachute_landing') else 'server_only'
            detail=short(r.get('item_id')) if 'item_id' in r else (f"{r.get('damage_type')} · loss {r.get('damage')}" if table=='damage' else table)
            ledger.append({'t':r['t'],'player':p,'table':table,'event':r.get('event_type',table),'detail':detail,'scope':scope,'source_id':r['id']})
    ledger.sort(key=lambda r:r['t'])
    audit=[{'t':t,'teammate_equipment':{p:equipment(tables['item_equip'],p,t) for p in MEMBERS[1:]},'outgoing_damage_events_p1':sum(1 for r in tables['damage'] if r['attacker_id']=='P1' and r['t']<=t and r['damage']>0)} for t in times]
    for a in audit:
        a['teammate_server_state']={}
        for p in MEMBERS[1:]:
            positions=[r for r in tables['positions'] if r['account_id']==p and r['t']<=a['t']]
            last=positions[-1] if positions else None
            damage=[r for r in tables['damage'] if r['victim_id']==p and a['t']-30<r['t']<=a['t'] and (r['damage'] or 0)>0]
            a['teammate_server_state'][p]={'last_vehicle_class':last['vehicle_id'] if last else None,'sample_age_s':round(a['t']-last['t'],3) if last else None,'received_damage_last30':round(sum(r['damage'] for r in damage),2),'damage_events_last30':len(damage),'scope':'server_only; no team communication evidence; last sample may be stale or after death'}
    history={p:[{'t':r['t'],'xy_m':[r['pos_x']/100,r['pos_y']/100]} for r in tables['positions'] if r['account_id']==p and r['t']>=min(x['t'] for x in tables['parachute_landing'] if x['account_id']==p) and r['t']<=end and (r['health'] or 0)>0] for p in MEMBERS}
    # Four summary scenes: all landed, transport swap, clustered UAZ evidence, first incoming gun damage.
    scene_times=[240.,755.,850.,incoming[0]['t']];scene_indices=[min(range(len(times)),key=lambda i:abs(times[i]-t)) for t in scene_times]
    meta={**src['metadata'],'patch':src['patch'],'selection':src['selection'],'selection_counts':src['selection_counts'],'observer':'P1','members':MEMBERS,'freshness_s':FRESH,'contact_window_s':CONTACT,'source_sha256':hashlib.sha256((OUT/'source_events_server_audit.json').read_bytes()).hexdigest(),'map':'Official historical Erangel map, last asset change 2024-10-28; not certified patch 40.2 geometry','source_captured_at':src['captured_at'],'observer_contract':'Telemetry-derived observable-information proxy; no actual screen, gaze, audio, communications, reaction or perception ground truth. Team vehicle/equipment and global alive-team count are withheld. Exact HP/positions and zone timing are HUD proxies.'}
    payload={'meta':meta,'frames':frames,'milestones':milestones,'history':history,'ledger':[e for e in ledger if e['scope']!='server_only'],'scene_indices':scene_indices}
    (OUT/'replay_observer.json').write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':')))
    (OUT/'server_audit_overlay.json').write_text(json.dumps(audit,ensure_ascii=False,separators=(',',':')))
    with (OUT/'event_ledger.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
    # Independent as-of assertions: future rows never alter a current frame.
    tests={}
    for t in [80.,225.,755.,850.,1322.758,1452.151]:
        prefix={k:[r for r in rows if r['t']<=t] for k,rows in tables.items()};assert snapshot(t,prefix)==snapshot(t,tables)
    tests['prefix_invariance_at_6_times']=True
    for f in frames:
        assert all(n.get('source_t',f['t']) is None or n.get('source_t',f['t'])<=f['t'] for n in f['nodes'])
        assert all(p['equipment'] is None and p['vehicle_class'] is None for p in f['players'][1:])
        assert len({n['id'] for n in f['nodes']})==len(f['nodes'])
        ids={n['id'] for n in f['nodes']};assert all(e['source'] in ids and e['target'] in ids for e in f['edges'])
        assert all(p['age_s']<=FRESH for p in f['players'] if p['position_valid'])
        for p in f['players']:
            if p['position_valid']:
                assert p['position_evidence']['source_t']>=min(r['t'] for r in tables['parachute_landing'] if r['account_id']==p['id'])
        assert len({(e['source'],e['target'],e['type']) for e in f['edges']})==len(f['edges'])
        assert not (set(f['team']['valid_members']) & set(f['team']['known_deaths']))
        assert all(c['enemy_location'] is None and c['enemy_visible'] is None for c in f['contacts'])
    tests.update(future_source_time_absent=True,teammate_private_state_withheld=True,graph_references_valid=True,stale_position_mask=True,post_landing_ground_position_only=True,unique_typed_edges=True,dead_excluded_from_team_center=True,enemy_visibility_and_location_unknown=True)
    assert equipment(list(reversed(tables['item_equip'])),'P1',end)==equipment(tables['item_equip'],'P1',end);tests['equipment_same_time_order_invariance']=True
    checks={'checks':tests,'frames':len(frames),'duplicates_removed':dups,'event_counts':{k:len(v) for k,v in tables.items()},'scenes':[{'t':frames[i]['t'],'team':frames[i]['team'],'nodes':len(frames[i]['nodes']),'edges':len(frames[i]['edges'])} for i in scene_indices],'first_incoming_gun_t':incoming[0]['t'],'incoming_gun_events':len(incoming),'incoming_gun_damage':round(sum(r['damage'] for r in incoming),2),'equipment_same_timestamp_ambiguities':sum(1 for r in equipment(tables['item_equip'],'P1',end) if r['status']=='ambiguous_same_time'),'positive_positions_after_first_death':{p:sum(1 for r in tables['positions'] if r['account_id']==p and r['t']>min(d['t'] for d in tables['kills'] if d['victim_id']==p) and (r['health'] or 0)>0) for p in MEMBERS}}
    (OUT/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    Image.open(ROOT/'data/spatial/erangel/source/official_notext.png').convert('RGB').resize((1600,1600)).save(OUT/'map.jpg',quality=88)
    print(json.dumps(checks,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
