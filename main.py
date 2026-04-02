"""
PUBG Single-Match Graph Prototype
==================================
단일 매치의 전체 타임스텝을 PyG HeteroData로 구성하는 프로토타입.

설계 결정 (데이터 감사 결과 기반):
- 타임스텝: 10초 간격, ±5초 윈도우 내 closest position
- 노드: 생존 플레이어 (마지막 position 기준 dropout)
- ally 엣지: team_id 기반 (positions ↔ rosters 정합 확인 완료)
- encounter 엣지: k-NN (k=5), 적 팀 플레이어만
- 탈락 라벨: telem_kills + telem_damage(byzone) 결합
- 팀 배치 라벨: rosters.rank
- ID 매핑: account_id = player_id 직접 매핑
- 자기장: game_states의 safe_zone 시계열

사용법:
  1. DB_CONFIG를 본인 환경에 맞게 수정
  2. python3 graph_prototype.py
  3. 출력되는 진단 정보로 그래프 품질 확인
"""

import psycopg2
import psycopg2.extras
import numpy as np
from collections import defaultdict
from datetime import datetime
import torch
from torch_geometric.data import HeteroData
from scipy.spatial import cKDTree

# ============================================================
# 1. DB 연결 설정
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "pubg_survival",
    "user": "postgres",       # ← 수정
    "password": "104204",   # ← 수정
}

SCHEMA = "pubg"
SNAPSHOT_INTERVAL = 10    # 초
SNAPSHOT_HALF_WINDOW = 5  # ±5초
K_NEIGHBORS = 5           # encounter 엣지 k-NN
CM_TO_M = 100.0           # 좌표 cm → m 변환


import decimal

def get_conn():
    # Decimal → float 자동 변환 등록
    DEC2FLOAT = psycopg2.extensions.new_type(
        psycopg2.extensions.DECIMAL.values,
        'DEC2FLOAT',
        lambda value, curs: float(value) if value is not None else None
    )
    psycopg2.extensions.register_type(DEC2FLOAT)
    return psycopg2.connect(**DB_CONFIG)


# ============================================================
# 2. 데이터 로드: 단일 매치
# ============================================================

def pick_sample_match(cur):
    """Erangel 스쿼드 매치 중 텔레메트리가 있는 첫 매치 선택."""
    cur.execute(f"""
        SELECT match_id
        FROM {SCHEMA}.v_match_summary
        WHERE map_name = 'Baltic_Main'
          AND game_mode IN ('squad', 'squad-fpp')
          AND telemetry_fetched = true
        ORDER BY match_id
        LIMIT 1
    """)
    return cur.fetchone()[0]


def load_positions(cur, match_id):
    """telem_positions 전체 로드. 리스트 of dict."""
    cur.execute(f"""
        SELECT account_id, team_id, elapsed_time,
               pos_x, pos_y, pos_z, health,
               vehicle_type, vehicle_speed
        FROM {SCHEMA}.telem_positions
        WHERE match_id = %s
          AND account_id IS NOT NULL
        ORDER BY elapsed_time
    """, (match_id,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def load_game_states(cur, match_id):
    """telem_game_states 로드. elapsed_time 기준 정렬."""
    cur.execute(f"""
        SELECT elapsed_time, num_alive_players, num_alive_teams,
               safe_zone_x, safe_zone_y, safe_zone_z, safe_zone_radius,
               poison_zone_x, poison_zone_y, poison_zone_z, poison_zone_radius
        FROM {SCHEMA}.telem_game_states
        WHERE match_id = %s
        ORDER BY elapsed_time
    """, (match_id,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def load_match_start_time(cur, match_id):
    """telem_match_start의 event_time을 기준 시각으로 가져온다."""
    cur.execute(f"""
        SELECT event_time
        FROM {SCHEMA}.telem_match_start
        WHERE match_id = %s
        LIMIT 1
    """, (match_id,))
    row = cur.fetchone()
    return row[0] if row else None


def event_time_to_elapsed(event_time, match_start_time):
    """event_time(timestamp) → elapsed_time(초) 변환."""
    if event_time is None or match_start_time is None:
        return None
    diff = event_time - match_start_time
    return diff.total_seconds()


def load_kills(cur, match_id, match_start_time):
    """telem_kills 로드. victim별 탈락 시점 (event_time → elapsed 변환)."""
    cur.execute(f"""
        SELECT victim_id, event_time,
               killer_id, damage_causer, distance, is_suicide
        FROM {SCHEMA}.telem_kills
        WHERE match_id = %s
          AND victim_id IS NOT NULL
    """, (match_id,))
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    # victim_id → 최종 kill 이벤트
    kill_map = {}
    for r in rows:
        r["elapsed_time"] = event_time_to_elapsed(r["event_time"], match_start_time)
        vid = r["victim_id"]
        if vid not in kill_map or r["event_time"] > kill_map[vid]["event_time"]:
            kill_map[vid] = r
    return kill_map


def load_byzone_deaths(cur, match_id, match_start_time):
    """
    byzone 사망자의 마지막 블루존 데미지 시점.
    damage_causer = 'Bluezonebomb_EffectActor_C'인 마지막 이벤트.
    """
    cur.execute(f"""
        SELECT victim_id, MAX(event_time) AS last_event_time
        FROM {SCHEMA}.telem_damage
        WHERE match_id = %s
          AND damage_causer = 'Bluezonebomb_EffectActor_C'
          AND victim_id IS NOT NULL
        GROUP BY victim_id
    """, (match_id,))
    result = {}
    for row in cur.fetchall():
        elapsed = event_time_to_elapsed(row[1], match_start_time)
        if elapsed is not None:
            result[row[0]] = elapsed
    return result


def load_rosters(cur, match_id):
    """rosters 로드. team_id → rank 매핑."""
    cur.execute(f"""
        SELECT r.team_id, r.rank,
               p.player_id, p.player_name
        FROM {SCHEMA}.rosters r
        JOIN {SCHEMA}.participants p
            ON r.roster_id = p.roster_id AND r.match_id = p.match_id
        WHERE r.match_id = %s
    """, (match_id,))
    team_rank = {}
    player_team = {}
    for row in cur.fetchall():
        team_id, rank, player_id, player_name = row
        team_rank[team_id] = rank
        player_team[player_id] = team_id
    return team_rank, player_team


def load_recent_damage(cur, match_id, match_start_time):
    """
    telem_damage 전체 로드. 스냅샷별 최근 데미지 집계에 사용.
    event_time → elapsed_time 변환 포함.
    """
    cur.execute(f"""
        SELECT attacker_id, victim_id, event_time, damage,
               damage_causer, damage_type
        FROM {SCHEMA}.telem_damage
        WHERE match_id = %s
          AND attacker_id IS NOT NULL
          AND victim_id IS NOT NULL
        ORDER BY event_time
    """, (match_id,))
    cols = [d[0] for d in cur.description]
    rows = []
    for row in cur.fetchall():
        d = dict(zip(cols, row))
        d["elapsed_time"] = event_time_to_elapsed(d["event_time"], match_start_time)
        rows.append(d)
    return rows


# ============================================================
# 3. 타임스텝 구성
# ============================================================

def determine_match_time_range(game_states):
    """매치의 시작/종료 elapsed_time 결정."""
    times = [gs["elapsed_time"] for gs in game_states if gs["elapsed_time"] is not None]
    return min(times), max(times)


def get_snapshot_times(start, end, interval=SNAPSHOT_INTERVAL):
    """interval 간격의 스냅샷 시점 리스트 생성."""
    # 첫 스냅샷은 start 이후 첫 interval 배수
    first = int(np.ceil(start / interval)) * interval
    times = list(range(first, int(end) + 1, interval))
    return times


def find_closest_zone_state(game_states, target_time):
    """target_time에 가장 가까운 game_state 반환."""
    best = None
    best_diff = float("inf")
    for gs in game_states:
        if gs["elapsed_time"] is None:
            continue
        diff = abs(gs["elapsed_time"] - target_time)
        if diff < best_diff:
            best_diff = diff
            best = gs
    return best


def determine_death_times(positions, kills, byzone_deaths):
    """
    각 플레이어의 탈락 elapsed_time 결정.

    전략:
    - telem_kills에 있으면: 해당 플레이어의 마지막 position elapsed_time 사용
      (kill 이벤트가 position보다 ~50초 늦을 수 있으므로)
    - byzone 사망이면: byzone_deaths의 마지막 블루존 데미지 elapsed_time 사용
    - 둘 다 없으면: 매치 종료 시 생존 (censored)

    반환: {account_id: death_elapsed} (생존자는 포함 안 됨)
    """
    # 플레이어별 마지막 position elapsed_time
    last_pos_time = {}
    for p in positions:
        aid = p["account_id"]
        et = p["elapsed_time"]
        if aid not in last_pos_time or et > last_pos_time[aid]:
            last_pos_time[aid] = et

    death_times = {}

    # kill 기록이 있는 플레이어: 마지막 position 시점을 사망 시점으로
    for vid, kill_info in kills.items():
        if vid in last_pos_time:
            death_times[vid] = last_pos_time[vid]

    # byzone 사망자 중 kills에 없는 경우 (드물지만 가능)
    for vid, death_elapsed in byzone_deaths.items():
        if vid not in death_times:
            death_times[vid] = death_elapsed

    return death_times


# ============================================================
# 4. 스냅샷 → 그래프 변환
# ============================================================

def build_player_index(positions, snapshot_time, half_window, death_times):
    """
    스냅샷 시점에 생존 중인 플레이어의 closest position을 추출.

    반환: {account_id: position_dict}
    """
    window_start = snapshot_time - half_window
    window_end = snapshot_time + half_window

    # 윈도우 내 레코드 필터링
    candidates = defaultdict(list)
    for p in positions:
        if window_start <= p["elapsed_time"] <= window_end:
            candidates[p["account_id"]].append(p)

    # 각 플레이어의 closest position 선택 + 생존 여부 필터
    result = {}
    for aid, pos_list in candidates.items():
        # 사망 시점 이전인지 확인
        if aid in death_times and death_times[aid] < window_start:
            continue  # 이미 탈락

        # snapshot_time에 가장 가까운 레코드
        best = min(pos_list, key=lambda p: abs(p["elapsed_time"] - snapshot_time))
        result[aid] = best

    return result


def compute_velocity(positions, account_id, snapshot_time, half_window=15):
    """
    이전 스냅샷과의 위치 차이로 속도 추정.
    positions 전체에서 해당 플레이어의 이전 기록을 찾아 차분.
    """
    prev_window_start = snapshot_time - half_window - SNAPSHOT_INTERVAL
    prev_window_end = snapshot_time - half_window

    prev_records = [
        p for p in positions
        if p["account_id"] == account_id
        and prev_window_start <= p["elapsed_time"] <= prev_window_end
    ]

    if not prev_records:
        return 0.0, 0.0

    prev = min(prev_records, key=lambda p: abs(p["elapsed_time"] - (snapshot_time - SNAPSHOT_INTERVAL)))
    # 현재 위치는 호출자가 알고 있으므로 여기서는 이전 위치만 반환
    return prev["pos_x"], prev["pos_y"]


def aggregate_recent_damage(damage_events, account_id, snapshot_time, lookback=30):
    """최근 lookback초 내 가한/받은 데미지 합산."""
    window_start = snapshot_time - lookback
    dealt = 0.0
    taken = 0.0
    for d in damage_events:
        if d["elapsed_time"] < window_start or d["elapsed_time"] > snapshot_time:
            continue
        if d["attacker_id"] == account_id:
            dealt += d["damage"] if d["damage"] else 0
        if d["victim_id"] == account_id:
            taken += d["damage"] if d["damage"] else 0
    return dealt, taken


def build_snapshot_graph(
    snapshot_time,
    player_positions,   # {account_id: pos_dict}
    zone_state,          # game_state dict
    team_rank,           # {team_id: rank}
    damage_events,       # full damage list
    positions_all,       # full positions list (for velocity)
    k=K_NEIGHBORS,
):
    """
    단일 스냅샷에서 PyG HeteroData 그래프 구성.

    노드 타입: 'player'
    엣지 타입: ('player', 'ally', 'player'), ('player', 'encounter', 'player')
    """
    data = HeteroData()

    players = list(player_positions.keys())
    n = len(players)

    if n == 0:
        return None

    pid_to_idx = {pid: i for i, pid in enumerate(players)}

    # ----------------------------------------------------------
    # 노드 피처
    # ----------------------------------------------------------
    node_feats = []
    coords = []  # 엣지 구성용

    safe_x = zone_state["safe_zone_x"] if zone_state else 0
    safe_y = zone_state["safe_zone_y"] if zone_state else 0
    safe_r = zone_state["safe_zone_radius"] if zone_state else 1
    poison_r = zone_state["poison_zone_radius"] if zone_state else 0
    alive_count = zone_state["num_alive_players"] if zone_state else n

    for pid in players:
        p = player_positions[pid]
        px, py, pz = p["pos_x"], p["pos_y"], p["pos_z"]
        health = p["health"] if p["health"] is not None else 100.0

        # 자기장 관련 피처
        dx_zone = (px - safe_x) / CM_TO_M
        dy_zone = (py - safe_y) / CM_TO_M
        dist_to_zone_center = np.sqrt(dx_zone**2 + dy_zone**2)
        dist_to_boundary = max(0, (safe_r / CM_TO_M) - dist_to_zone_center)
        inside_zone = 1.0 if dist_to_zone_center < (safe_r / CM_TO_M) else 0.0

        # 속도 (이전 스냅샷 대비 차분)
        veh_speed = p["vehicle_speed"] if p["vehicle_speed"] else 0.0
        in_vehicle = 1.0 if p["vehicle_type"] and p["vehicle_type"] != "" else 0.0

        # 최근 데미지
        dmg_dealt, dmg_taken = aggregate_recent_damage(
            damage_events, pid, snapshot_time, lookback=30
        )

        feat = [
            px / CM_TO_M,            # pos_x (m)
            py / CM_TO_M,            # pos_y (m)
            pz / CM_TO_M,            # pos_z (m) - 고도
            health / 100.0,          # 체력 (0~1)
            dist_to_zone_center,     # 자기장 중심 거리 (m)
            dist_to_boundary,        # 자기장 경계까지 거리 (m)
            inside_zone,             # 자기장 내부 여부
            veh_speed / CM_TO_M,     # 차량 속도 (m/s)
            in_vehicle,              # 탑승 여부
            dmg_dealt,               # 최근 30초 가한 데미지
            dmg_taken,               # 최근 30초 받은 데미지
        ]

        node_feats.append(feat)
        coords.append([px, py])

    data["player"].x = torch.tensor(node_feats, dtype=torch.float32)
    data["player"].num_nodes = n

    # ----------------------------------------------------------
    # 글로벌 자기장 컨텍스트
    # ----------------------------------------------------------
    effective_area = np.pi * (safe_r / CM_TO_M) ** 2
    density = alive_count / max(effective_area, 1.0)

    zone_context = torch.tensor([[
        safe_x / CM_TO_M,
        safe_y / CM_TO_M,
        safe_r / CM_TO_M,
        poison_r / CM_TO_M,
        effective_area,
        density,
        alive_count,
        snapshot_time,
    ]], dtype=torch.float32)
    data["zone"].x = zone_context
    data["zone"].num_nodes = 1

    # ----------------------------------------------------------
    # ally 엣지: 같은 팀 내 모든 쌍 (양방향)
    # ----------------------------------------------------------
    team_groups = defaultdict(list)
    for pid in players:
        tid = player_positions[pid]["team_id"]
        team_groups[tid].append(pid_to_idx[pid])

    ally_src, ally_dst = [], []
    ally_feat = []
    for tid, members in team_groups.items():
        for i in range(len(members)):
            for j in range(len(members)):
                if i != j:
                    src_idx = members[i]
                    dst_idx = members[j]
                    ally_src.append(src_idx)
                    ally_dst.append(dst_idx)

                    # 엣지 피처: 상호 거리, 고도차
                    sx, sy = coords[src_idx]
                    dx, dy = coords[dst_idx]
                    dist = np.sqrt((sx - dx)**2 + (sy - dy)**2) / CM_TO_M
                    sz = node_feats[src_idx][2]  # pos_z in m
                    dz = node_feats[dst_idx][2]
                    alt_diff = sz - dz  # 양수면 src가 높음

                    ally_feat.append([dist, alt_diff])

    if ally_src:
        data["player", "ally", "player"].edge_index = torch.tensor(
            [ally_src, ally_dst], dtype=torch.long
        )
        data["player", "ally", "player"].edge_attr = torch.tensor(
            ally_feat, dtype=torch.float32
        )
    else:
        data["player", "ally", "player"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["player", "ally", "player"].edge_attr = torch.zeros((0, 2), dtype=torch.float32)

    # ----------------------------------------------------------
    # encounter 엣지: k-NN (적 팀만, 양방향)
    # ----------------------------------------------------------
    coords_arr = np.array(coords)
    team_ids = [player_positions[pid]["team_id"] for pid in players]

    enc_src, enc_dst = [], []
    enc_feat = []

    if n > 1:
        tree = cKDTree(coords_arr)

        for i, pid in enumerate(players):
            # 자기 포함 k+팀원수 만큼 쿼리해서 적만 필터링
            query_k = min(n, k + 10)  # 충분히 많이 쿼리
            dists, indices = tree.query(coords_arr[i], k=query_k)

            enemy_count = 0
            for d, j in zip(dists, indices):
                if j == i:
                    continue
                if team_ids[j] == team_ids[i]:
                    continue  # 같은 팀 스킵
                if enemy_count >= k:
                    break

                enc_src.append(i)
                enc_dst.append(j)

                dist_m = d / CM_TO_M
                alt_diff = node_feats[i][2] - node_feats[j][2]
                enc_feat.append([dist_m, alt_diff])

                enemy_count += 1

    if enc_src:
        data["player", "encounter", "player"].edge_index = torch.tensor(
            [enc_src, enc_dst], dtype=torch.long
        )
        data["player", "encounter", "player"].edge_attr = torch.tensor(
            enc_feat, dtype=torch.float32
        )
    else:
        data["player", "encounter", "player"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["player", "encounter", "player"].edge_attr = torch.zeros((0, 2), dtype=torch.float32)

    return data


# ============================================================
# 5. 매치 전체 그래프 시퀀스 생성
# ============================================================

def build_match_graph_sequence(match_id, conn):
    """
    단일 매치의 전체 타임스텝 그래프 시퀀스 생성.

    반환:
      graphs: list of HeteroData (타임스텝별)
      snapshot_times: list of float (각 그래프의 시점)
      meta: dict (매치 메타데이터)
    """
    cur = conn.cursor()

    print(f"[1/6] 데이터 로드: {match_id[:20]}...")
    match_start_time = load_match_start_time(cur, match_id)
    positions = load_positions(cur, match_id)
    game_states = load_game_states(cur, match_id)
    kills = load_kills(cur, match_id, match_start_time)
    byzone_deaths = load_byzone_deaths(cur, match_id, match_start_time)
    team_rank, player_team = load_rosters(cur, match_id)
    damage_events = load_recent_damage(cur, match_id, match_start_time)
    cur.close()

    print(f"  positions: {len(positions):,}, game_states: {len(game_states)}")
    print(f"  kills: {len(kills)}, byzone_deaths: {len(byzone_deaths)}")
    print(f"  teams: {len(team_rank)}, damage_events: {len(damage_events):,}")

    # 탈락 시점 결정
    print("[2/6] 탈락 시점 계산...")
    death_times = determine_death_times(positions, kills, byzone_deaths)
    all_players = set(p["account_id"] for p in positions)
    survivors = all_players - set(death_times.keys())
    print(f"  총 플레이어: {len(all_players)}, 탈락: {len(death_times)}, 생존: {len(survivors)}")

    # 타임스텝 결정
    print("[3/6] 타임스텝 구성...")
    t_start, t_end = determine_match_time_range(game_states)
    snapshot_times = get_snapshot_times(t_start, t_end, SNAPSHOT_INTERVAL)
    print(f"  매치 범위: {t_start}~{t_end}초, 스냅샷 수: {len(snapshot_times)}")

    # positions를 elapsed_time 기준 인덱싱 (속도 최적화)
    # → 윈도우 필터링을 전체 positions에서 매번 하면 느림
    # → 미리 정렬된 상태이므로 이진 탐색 가능하지만, 프로토타입에서는 단순 필터
    print("[4/6] 그래프 생성 중...")
    graphs = []
    for i, st in enumerate(snapshot_times):
        # 플레이어 스냅샷
        player_positions = build_player_index(
            positions, st, SNAPSHOT_HALF_WINDOW, death_times
        )

        # 자기장 상태
        zone_state = find_closest_zone_state(game_states, st)

        # 그래프 구성
        g = build_snapshot_graph(
            snapshot_time=st,
            player_positions=player_positions,
            zone_state=zone_state,
            team_rank=team_rank,
            damage_events=damage_events,
            positions_all=positions,
            k=K_NEIGHBORS,
        )

        if g is not None:
            # 메타데이터 추가
            g.snapshot_time = st
            g.num_alive = len(player_positions)
            graphs.append(g)

        if (i + 1) % 20 == 0 or i == len(snapshot_times) - 1:
            alive = len(player_positions) if player_positions else 0
            print(f"  [{i+1}/{len(snapshot_times)}] t={st}s, alive={alive}")

    # 라벨 구성
    print("[5/6] 라벨 생성...")
    meta = {
        "match_id": match_id,
        "team_rank": team_rank,           # {team_id: rank}
        "death_times": death_times,       # {account_id: elapsed_time}
        "player_team": player_team,       # {player_id: team_id}
        "survivors": survivors,           # set of account_id
        "total_players": len(all_players),
        "snapshot_times": snapshot_times,
    }

    print("[6/6] 완료.")
    return graphs, snapshot_times, meta


# ============================================================
# 6. 진단 출력
# ============================================================

def print_diagnostics(graphs, snapshot_times, meta):
    """그래프 시퀀스의 품질 진단 정보 출력."""
    print("\n" + "=" * 60)
    print("그래프 시퀀스 진단")
    print("=" * 60)

    print(f"\n매치: {meta['match_id'][:40]}...")
    print(f"총 플레이어: {meta['total_players']}")
    print(f"총 팀: {len(meta['team_rank'])}")
    print(f"총 스냅샷: {len(graphs)}")
    print(f"시간 범위: {snapshot_times[0]}~{snapshot_times[-1]}초")

    # 스냅샷별 통계
    print(f"\n{'시점':>6} {'노드':>5} {'ally엣지':>8} {'enc엣지':>8} "
          f"{'ally/node':>9} {'enc/node':>9}")
    print("-" * 55)

    sample_indices = [0, len(graphs)//4, len(graphs)//2, 3*len(graphs)//4, len(graphs)-1]
    sample_indices = sorted(set(min(i, len(graphs)-1) for i in sample_indices))

    for i in sample_indices:
        g = graphs[i]
        n_nodes = g["player"].num_nodes
        n_ally = g["player", "ally", "player"].edge_index.shape[1]
        n_enc = g["player", "encounter", "player"].edge_index.shape[1]
        ally_per = n_ally / max(n_nodes, 1)
        enc_per = n_enc / max(n_nodes, 1)
        print(f"{g.snapshot_time:6.0f} {n_nodes:5d} {n_ally:8d} {n_enc:8d} "
              f"{ally_per:9.1f} {enc_per:9.1f}")

    # 노드 피처 통계 (첫 스냅샷)
    g0 = graphs[0]
    x = g0["player"].x
    feat_names = [
        "pos_x(m)", "pos_y(m)", "pos_z(m)", "health",
        "zone_dist", "boundary_dist", "inside_zone",
        "veh_speed", "in_vehicle", "dmg_dealt", "dmg_taken"
    ]
    print(f"\n노드 피처 통계 (첫 스냅샷, {x.shape[0]}노드):")
    print(f"{'피처':<15} {'min':>10} {'max':>10} {'mean':>10} {'std':>10}")
    print("-" * 57)
    for j, name in enumerate(feat_names):
        col = x[:, j]
        print(f"{name:<15} {col.min().item():10.1f} {col.max().item():10.1f} "
              f"{col.mean().item():10.1f} {col.std().item():10.1f}")

    # encounter 엣지 거리 통계
    print(f"\nencounter 엣지 거리 (m):")
    for i in sample_indices:
        g = graphs[i]
        if g["player", "encounter", "player"].edge_attr.shape[0] > 0:
            dists = g["player", "encounter", "player"].edge_attr[:, 0]
            print(f"  t={g.snapshot_time:.0f}s: "
                  f"min={dists.min().item():.0f}, "
                  f"median={dists.median().item():.0f}, "
                  f"max={dists.max().item():.0f}, "
                  f"edges={dists.shape[0]}")

    # zone context 시계열
    print(f"\n자기장 축소 시계열:")
    for i in sample_indices:
        g = graphs[i]
        z = g["zone"].x[0]
        print(f"  t={g.snapshot_time:.0f}s: "
              f"safe_r={z[2].item():.0f}m, "
              f"density={z[5].item():.6f}, "
              f"alive={z[6].item():.0f}")

    # 탈락 라벨 검증
    print(f"\n탈락 라벨:")
    print(f"  기록된 사망: {len(meta['death_times'])}")
    print(f"  생존(censored): {len(meta['survivors'])}")
    print(f"  팀 배치 범위: {min(meta['team_rank'].values())}~{max(meta['team_rank'].values())}")

    # 메모리 사용량 추정
    total_bytes = 0
    for g in graphs:
        total_bytes += g["player"].x.element_size() * g["player"].x.nelement()
        for et in [("player", "ally", "player"), ("player", "encounter", "player")]:
            total_bytes += g[et].edge_index.element_size() * g[et].edge_index.nelement()
            total_bytes += g[et].edge_attr.element_size() * g[et].edge_attr.nelement()
    print(f"\n총 메모리 (추정): {total_bytes / 1024 / 1024:.1f} MB")
    print(f"스냅샷당 평균: {total_bytes / len(graphs) / 1024:.1f} KB")


# ============================================================
# 메인
# ============================================================

if __name__ == "__main__":
    print("PUBG Single-Match Graph Prototype")
    print("=" * 40)

    conn = get_conn()
    cur = conn.cursor()

    match_id = pick_sample_match(cur)
    cur.close()
    print(f"선택된 매치: {match_id}")

    graphs, snapshot_times, meta = build_match_graph_sequence(match_id, conn)
    print_diagnostics(graphs, snapshot_times, meta)

    # 저장 (선택)
    save_path = f"match_graphs_{match_id[:8]}.pt"
    torch.save({
        "graphs": graphs,
        "snapshot_times": snapshot_times,
        "meta": meta,
    }, save_path)
    print(f"\n저장: {save_path}")

    conn.close()