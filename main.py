"""
PUBG Single-Match Graph — V2 Feature Expansion
================================================
단일 매치의 전체 타임스텝을 PyG HeteroData로 구성.

V2 피처 설계 (생태학 5축, ~39d agent 노드 피처):
  축1: Physiological State (5d)  — health, groggy, recovery
  축2: Mobility (8d)            — position, velocity, radial_speed, ETA
  축3: Habitat Exposure (5d)    — zone distance, zone damage
  축4: Competition Pressure (13d)— damage windows, density, isolation
  축5: Resource Readiness (8d)  — weapon, armor, heal/boost

설계 결정 (데이터 감사 결과 기반):
- 타임스텝: 10초 간격, ±5초 윈도우 내 closest position
- 노드: 생존 플레이어 (마지막 position 기준 dropout)
- ally 엣지: team_id 기반 (positions ↔ rosters 정합 확인 완료)
- encounter 엣지: k-NN (k=5), 적 팀 플레이어만
- 탈락 라벨: telem_kills + telem_damage(byzone) 결합
- 팀 배치 라벨: rosters.rank
- ID 매핑: account_id = player_id 직접 매핑
- 자기장: game_states의 safe_zone(목표) + poison_zone(현재 경계) 이중 시계열

사용법:
  1. DB_CONFIG를 본인 환경에 맞게 수정
  2. python3 main.py
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
from scipy.ndimage import uniform_filter

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
               damage_causer, damage_type,
               attacker_x, attacker_y, victim_x, victim_y
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


def load_groggy_events(cur, match_id, match_start_time):
    """
    telem_groggy (LogPlayerMakeGroggy) 로드.
    victim_id → 다운 시점 목록. groggy 상태 + decay 피처 계산에 사용.
    테이블 없으면 빈 리스트 반환 (fallback).
    """
    try:
        cur.execute(f"""
            SELECT victim_id, event_time, attacker_id, damage_causer
            FROM {SCHEMA}.telem_groggy
            WHERE match_id = %s
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
    except Exception:
        cur.execute("ROLLBACK")
        return []


def load_item_equip(cur, match_id, match_start_time):
    """
    telem_item_equip (LogItemEquip) 로드.
    플레이어별 장착 장비 추적. weapon_class, armor_level 피처에 사용.
    테이블 없으면 빈 리스트 반환.
    """
    try:
        cur.execute(f"""
            SELECT account_id, event_time, item_id,
                   item_category, item_sub_category
            FROM {SCHEMA}.telem_item_equip
            WHERE match_id = %s
              AND account_id IS NOT NULL
            ORDER BY event_time
        """, (match_id,))
        cols = [d[0] for d in cur.description]
        rows = []
        for row in cur.fetchall():
            d = dict(zip(cols, row))
            d["elapsed_time"] = event_time_to_elapsed(d["event_time"], match_start_time)
            rows.append(d)
        return rows
    except Exception:
        cur.execute("ROLLBACK")
        return []


def load_item_use(cur, match_id, match_start_time):
    """
    telem_item_use (LogItemUse) 로드.
    heal/boost 사용 추적. hp_recovery_recent, heal/boost_use 피처에 사용.
    테이블 없으면 빈 리스트 반환.
    """
    try:
        cur.execute(f"""
            SELECT account_id, event_time, item_id,
                   item_category, item_sub_category
            FROM {SCHEMA}.telem_item_use
            WHERE match_id = %s
              AND account_id IS NOT NULL
            ORDER BY event_time
        """, (match_id,))
        cols = [d[0] for d in cur.description]
        rows = []
        for row in cur.fetchall():
            d = dict(zip(cols, row))
            d["elapsed_time"] = event_time_to_elapsed(d["event_time"], match_start_time)
            rows.append(d)
        return rows
    except Exception:
        cur.execute("ROLLBACK")
        return []


# ============================================================
# 2b. 피처 계산 유틸리티
# ============================================================

# 무기 클래스 점수 (CLAUDE.md §4 축5)
WEAPON_CLASS_SCORE = {
    "AR": 0.7, "DMR": 0.75, "SR": 0.8, "SMG": 0.5,
    "SG": 0.4, "LMG": 0.6, "Pistol": 0.2, "Melee": 0.1,
}

# item_sub_category → 무기 클래스 매핑
WEAPON_SUBCATEGORY_MAP = {
    "Main": None,  # 일부 아이템 일반 카테고리
    "AssaultRifle": "AR", "Rifle": "AR",
    "DMR": "DMR", "DesignatedMarksmanRifle": "DMR",
    "SR": "SR", "SniperRifle": "SR",
    "SMG": "SMG", "SubMachineGun": "SMG",
    "Shotgun": "SG", "SG": "SG",
    "LMG": "LMG", "LightMachineGun": "LMG",
    "Handgun": "Pistol", "Pistol": "Pistol",
    "Melee": "Melee",
}

# 방어구 레벨 추출: item_id에서 Lv1/Lv2/Lv3 파싱
def parse_armor_level(item_id):
    """item_id에서 방어구 레벨 추출 (0~3)."""
    if not item_id:
        return 0
    item_id_upper = item_id.upper()
    if "LV3" in item_id_upper or "LEVEL3" in item_id_upper or "_03" in item_id:
        return 3
    if "LV2" in item_id_upper or "LEVEL2" in item_id_upper or "_02" in item_id:
        return 2
    if "LV1" in item_id_upper or "LEVEL1" in item_id_upper or "_01" in item_id:
        return 1
    return 0


def get_weapon_score(item_id, item_sub_category):
    """무기 item_id/sub_category → 점수."""
    # sub_category 우선
    if item_sub_category:
        wclass = WEAPON_SUBCATEGORY_MAP.get(item_sub_category)
        if wclass and wclass in WEAPON_CLASS_SCORE:
            return WEAPON_CLASS_SCORE[wclass]
    # item_id fallback: PUBG item naming convention
    if item_id:
        item_upper = item_id.upper()
        # PUBG 무기 이름 패턴 매칭 (Item_Weapon_AK47_C 등)
        # 먼저 구체적 무기명 → 카테고리 매핑
        WEAPON_ID_PATTERNS = {
            # AR
            "AK47": "AR", "M416": "AR", "SCAR": "AR", "HK416": "AR",
            "GROZA": "AR", "AUG": "AR", "QBZ": "AR", "G36C": "AR",
            "BERYL": "AR", "ACE32": "AR", "M16A4": "AR", "FAMAS": "AR",
            # DMR
            "MINI14": "DMR", "SKS": "DMR", "SLR": "DMR", "QBU": "DMR",
            "MK47": "DMR", "MK12": "DMR", "VSS": "DMR", "MK14": "DMR",
            # SR
            "KAR98": "SR", "M24": "SR", "AWM": "SR", "WIN94": "SR",
            "MOSIN": "SR", "LYNX": "SR",
            # SMG
            "UMP": "SMG", "VECTOR": "SMG", "UZI": "SMG", "MP5K": "SMG",
            "BIZON": "SMG", "THOMPSON": "SMG", "P90": "SMG", "MP9": "SMG",
            # SG
            "S12K": "SG", "S1897": "SG", "S686": "SG", "DBS": "SG",
            # LMG
            "DP28": "LMG", "M249": "LMG", "MG3": "LMG",
            # Pistol
            "P92": "Pistol", "P1911": "Pistol", "R45": "Pistol",
            "DEAGLE": "Pistol", "P18C": "Pistol", "R1895": "Pistol",
            "SKORPION": "Pistol", "FLARE": "Pistol",
        }
        for wname, wclass in WEAPON_ID_PATTERNS.items():
            if wname in item_upper:
                return WEAPON_CLASS_SCORE.get(wclass, 0.0)
        # generic fallback
        if "WEAPON" in item_upper:
            return 0.3  # 알 수 없는 무기 기본 점수
    return 0.0


def build_equipment_state(item_equip_events, snapshot_time):
    """
    장비 이벤트 → 스냅샷 시점의 플레이어별 장비 상태.

    반환: {account_id: {
        "weapons": [score1, score2],  # 상위 2개 무기 점수
        "armor_level": int,
        "helmet_level": int,
        "backpack_level": int,
        "attachment_count": int,
    }}
    """
    # 시점까지의 최신 장비 상태 추적
    player_weapons = defaultdict(list)     # 현재 무기 목록
    player_armor = defaultdict(int)
    player_helmet = defaultdict(int)
    player_backpack = defaultdict(int)

    for evt in item_equip_events:
        if evt["elapsed_time"] is None or evt["elapsed_time"] > snapshot_time:
            continue
        aid = evt["account_id"]
        cat = (evt.get("item_category") or "").lower()
        subcat = evt.get("item_sub_category") or ""
        item_id = evt.get("item_id") or ""

        if cat == "weapon" or "weapon" in cat:
            score = get_weapon_score(item_id, subcat)
            if score > 0:
                player_weapons[aid].append(score)
        elif cat == "equipment" or "armor" in cat.lower() or "vest" in item_id.lower():
            if "vest" in item_id.lower() or "armor" in cat.lower():
                player_armor[aid] = max(player_armor[aid], parse_armor_level(item_id))
            elif "helmet" in item_id.lower() or "head" in item_id.lower():
                player_helmet[aid] = max(player_helmet[aid], parse_armor_level(item_id))
            elif "backpack" in item_id.lower() or "bag" in item_id.lower():
                player_backpack[aid] = max(player_backpack[aid], parse_armor_level(item_id))
        elif "helmet" in item_id.lower():
            player_helmet[aid] = max(player_helmet[aid], parse_armor_level(item_id))
        elif "backpack" in item_id.lower() or "bag" in item_id.lower():
            player_backpack[aid] = max(player_backpack[aid], parse_armor_level(item_id))

    result = {}
    all_aids = set(player_weapons.keys()) | set(player_armor.keys()) | \
               set(player_helmet.keys()) | set(player_backpack.keys())
    for aid in all_aids:
        weapons = sorted(player_weapons.get(aid, []), reverse=True)
        result[aid] = {
            "weapons": weapons[:2] + [0.0] * (2 - len(weapons[:2])),
            "armor_level": player_armor.get(aid, 0),
            "helmet_level": player_helmet.get(aid, 0),
            "backpack_level": player_backpack.get(aid, 0),
        }
    return result


def build_item_use_state(item_use_events, snapshot_time, lookback=60):
    """
    아이템 사용 이벤트 → 최근 heal/boost 사용 횟수.

    반환: {account_id: {"heal_count": int, "boost_count": int}}
    """
    window_start = snapshot_time - lookback
    result = defaultdict(lambda: {"heal_count": 0, "boost_count": 0})

    for evt in item_use_events:
        et = evt.get("elapsed_time")
        if et is None or et < window_start or et > snapshot_time:
            continue
        aid = evt["account_id"]
        cat = (evt.get("item_category") or "").lower()
        subcat = (evt.get("item_sub_category") or "").lower()
        item_id = (evt.get("item_id") or "").lower()

        is_heal = any(k in item_id for k in ["firstaid", "medkit", "bandage", "healthkit"])
        is_boost = any(k in item_id for k in ["painkiller", "energydrink", "adrenaline"])

        if is_heal or "heal" in cat:
            result[aid]["heal_count"] += 1
        elif is_boost or "boost" in cat:
            result[aid]["boost_count"] += 1

    return dict(result)


def build_groggy_state(groggy_events, snapshot_time):
    """
    groggy 이벤트 → 현재 groggy 상태 + decay 피처.

    반환: {victim_id: {"is_groggy": 0/1, "groggy_decay": float}}

    판정 로직:
    - 최근 30초 내 groggy 발생 → is_groggy 후보 (실제로는 revive로 풀릴 수 있음)
    - groggy_decay = exp(-Δt/30) — 마지막 groggy 이후 시간 경과
    """
    result = {}
    for evt in groggy_events:
        et = evt.get("elapsed_time")
        if et is None or et > snapshot_time:
            continue
        vid = evt["victim_id"]
        dt = snapshot_time - et
        decay = np.exp(-dt / 30.0)

        # 가장 최근 이벤트 기준
        if vid not in result or et > result[vid]["_last_time"]:
            result[vid] = {
                "is_groggy": 1.0 if dt < 30.0 else 0.0,
                "groggy_decay": decay,
                "_last_time": et,
            }

    # 내부 키 제거
    for vid in result:
        del result[vid]["_last_time"]

    return result


def compute_velocity(positions_all, account_id, snapshot_time, lookback=10):
    """
    position 차분으로 속도 벡터 계산.

    반환: (vx, vy, speed_m_s) — m/s 단위.
    이전 위치가 없으면 (0, 0, 0).
    """
    prev_pos = None
    curr_pos = None
    prev_time = None
    curr_time = None

    for p in positions_all:
        if p["account_id"] != account_id:
            continue
        et = p["elapsed_time"]
        if et is None:
            continue

        # snapshot_time 이하의 가장 최근 두 위치
        if et <= snapshot_time:
            if curr_pos is None or et > curr_time:
                prev_pos = curr_pos
                prev_time = curr_time
                curr_pos = p
                curr_time = et
            elif prev_pos is None or et > prev_time:
                prev_pos = p
                prev_time = et

    if prev_pos is None or curr_pos is None or curr_time == prev_time:
        return 0.0, 0.0, 0.0

    dt = curr_time - prev_time
    if dt <= 0 or dt > lookback * 2:
        return 0.0, 0.0, 0.0

    vx = (curr_pos["pos_x"] - prev_pos["pos_x"]) / CM_TO_M / dt
    vy = (curr_pos["pos_y"] - prev_pos["pos_y"]) / CM_TO_M / dt
    speed = np.sqrt(vx**2 + vy**2)
    return vx, vy, speed


def aggregate_damage_detailed(damage_events, account_id, snapshot_time):
    """
    10s/30s 윈도우 + unique attacker/target + decay 피처를 한번에 계산.

    반환: dict with keys:
      dealt_10s, taken_10s, dealt_30s, taken_30s,
      n_attackers_30s, n_targets_30s,
      dmg_taken_decay, dmg_dealt_decay,
      zone_dmg_30s
    """
    result = {
        "dealt_10s": 0.0, "taken_10s": 0.0,
        "dealt_30s": 0.0, "taken_30s": 0.0,
        "n_attackers_30s": 0, "n_targets_30s": 0,
        "dmg_taken_decay": 0.0, "dmg_dealt_decay": 0.0,
        "zone_dmg_30s": 0.0,
    }

    w10_start = snapshot_time - 10
    w30_start = snapshot_time - 30
    attackers_30s = set()
    targets_30s = set()
    last_dmg_taken_time = None
    last_dmg_dealt_time = None

    for d in damage_events:
        et = d.get("elapsed_time")
        if et is None or et > snapshot_time:
            continue

        dmg = d.get("damage") or 0
        if dmg <= 0:
            continue

        causer = d.get("damage_causer") or ""
        is_zone_dmg = "bluezonebomb" in causer.lower()

        # victim 측
        if d["victim_id"] == account_id:
            if is_zone_dmg:
                if et >= w30_start:
                    result["zone_dmg_30s"] += dmg
            else:
                if et >= w30_start:
                    result["taken_30s"] += dmg
                    if d["attacker_id"]:
                        attackers_30s.add(d["attacker_id"])
                if et >= w10_start:
                    result["taken_10s"] += dmg
                if last_dmg_taken_time is None or et > last_dmg_taken_time:
                    last_dmg_taken_time = et

        # attacker 측
        if d["attacker_id"] == account_id and not is_zone_dmg:
            if et >= w30_start:
                result["dealt_30s"] += dmg
                if d["victim_id"]:
                    targets_30s.add(d["victim_id"])
            if et >= w10_start:
                result["dealt_10s"] += dmg
            if last_dmg_dealt_time is None or et > last_dmg_dealt_time:
                last_dmg_dealt_time = et

    result["n_attackers_30s"] = len(attackers_30s)
    result["n_targets_30s"] = len(targets_30s)

    if last_dmg_taken_time is not None:
        result["dmg_taken_decay"] = np.exp(-(snapshot_time - last_dmg_taken_time) / 10.0)
    if last_dmg_dealt_time is not None:
        result["dmg_dealt_decay"] = np.exp(-(snapshot_time - last_dmg_dealt_time) / 10.0)

    return result


# ============================================================
# 3. 타임스텝 구성
# ============================================================

def determine_match_time_range(game_states):
    """매치의 시작/종료 elapsed_time 결정."""
    times = [gs["elapsed_time"] for gs in game_states if gs["elapsed_time"] is not None]
    return min(times), max(times)


def get_snapshot_times(start, end, interval=SNAPSHOT_INTERVAL):
    """interval 간격의 스냅샷 시점 리스트 생성."""
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
    last_pos_time = {}
    for p in positions:
        aid = p["account_id"]
        et = p["elapsed_time"]
        if aid not in last_pos_time or et > last_pos_time[aid]:
            last_pos_time[aid] = et

    death_times = {}

    for vid, kill_info in kills.items():
        if vid in last_pos_time:
            death_times[vid] = last_pos_time[vid]

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

    candidates = defaultdict(list)
    for p in positions:
        if window_start <= p["elapsed_time"] <= window_end:
            candidates[p["account_id"]].append(p)

    result = {}
    for aid, pos_list in candidates.items():
        if aid in death_times and death_times[aid] < window_start:
            continue

        best = min(pos_list, key=lambda p: abs(p["elapsed_time"] - snapshot_time))
        result[aid] = best

    return result


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
    global_team_to_idx,  # 글로벌 팀→인덱스 매핑 (고정)
    global_pid_to_idx,   # 글로벌 플레이어→인덱스 매핑 (고정)
    groggy_events=None,  # V2: groggy 이벤트
    equip_state=None,    # V2: {aid: equipment_dict}
    item_use_state=None, # V2: {aid: {heal_count, boost_count}}
    prev_positions=None, # V2: 이전 스냅샷 {aid: pos_dict} (정지 비율용)
    k=K_NEIGHBORS,
):
    """
    단일 스냅샷에서 PyG HeteroData 그래프 구성 (V2).

    노드 타입: 'player'
    엣지 타입: ('player', 'ally', 'player'), ('player', 'encounter', 'player')

    노드 피처 (39d, 생태학 5축):
      축1 Physiological:  health, shield, groggy, groggy_decay, hp_recovery
      축2 Mobility:       x, y, z, speed, in_vehicle, radial_speed, eta, stationary
      축3 Habitat:        dist_boundary, dist_safe, inside_safe, inside_boundary, zone_dmg
      축4 Competition:    dmg_dealt/taken 10s/30s, attackers, targets, decay, ally/enemy density
      축5 Resource:       weapons, attachment, armor, helmet, backpack, heal, boost
    """
    data = HeteroData()

    players = sorted(player_positions.keys())
    n = len(players)

    if n == 0:
        return None

    if groggy_events is None:
        groggy_events = []
    if equip_state is None:
        equip_state = {}
    if item_use_state is None:
        item_use_state = {}

    pid_to_idx = {pid: i for i, pid in enumerate(players)}

    # ── zone 기본 데이터 ──
    safe_x = zone_state["safe_zone_x"] if zone_state else 0
    safe_y = zone_state["safe_zone_y"] if zone_state else 0
    safe_r = zone_state["safe_zone_radius"] if zone_state else 1
    poison_x = zone_state["poison_zone_x"] if zone_state else 0
    poison_y = zone_state["poison_zone_y"] if zone_state else 0
    poison_r = zone_state["poison_zone_radius"] if zone_state else 0
    alive_count = zone_state["num_alive_players"] if zone_state else n

    ARENA_SIZE = 816000.0    # cm (8160m)
    ARENA_DIAG = ARENA_SIZE * np.sqrt(2)
    MAX_SPEED = 130.0        # m/s (차량 최대)
    COMBAT_NORM = 200.0      # 30초 데미지 정규화 기준

    # V2: groggy 상태 계산
    groggy_state = build_groggy_state(groggy_events, snapshot_time)

    # ── 좌표 배열 (팀별 그루핑, 밀도 계산용) ──
    coords_m = []       # meters
    coords_cm = []      # cm (원본)
    team_id_list = []

    for pid in players:
        p = player_positions[pid]
        coords_cm.append([p["pos_x"], p["pos_y"]])
        coords_m.append([p["pos_x"] / CM_TO_M, p["pos_y"] / CM_TO_M])
        team_id_list.append(p["team_id"])

    coords_cm_arr = np.array(coords_cm)
    coords_m_arr = np.array(coords_m)

    # ── 팀 그루핑 (밀도/거리 계산용) ──
    team_members_idx = defaultdict(list)  # team_id → [player indices]
    for i, pid in enumerate(players):
        team_members_idx[player_positions[pid]["team_id"]].append(i)

    # ── KD-tree for spatial queries ──
    tree = cKDTree(coords_cm_arr) if n > 1 else None
    RADIUS_100M_CM = 100.0 * CM_TO_M  # 100m in cm

    # ----------------------------------------------------------
    # 노드 피처 (39d)
    # ----------------------------------------------------------
    node_feats = []

    for idx, pid in enumerate(players):
        p = player_positions[pid]
        px, py, pz = p["pos_x"], p["pos_y"], p["pos_z"]
        health = p["health"] if p["health"] is not None else 100.0
        tid = team_id_list[idx]

        # ── 축1: Physiological State (5d) ──
        health_ratio = np.clip(health / 100.0, 0, 1)
        shield_ratio = 0.0  # PUBG: 방탄복은 축5에서 처리

        gs = groggy_state.get(pid, {})
        is_groggy = gs.get("is_groggy", 0.0)
        groggy_decay = gs.get("groggy_decay", 0.0)

        use_state = item_use_state.get(pid, {})
        hp_recovery = min((use_state.get("heal_count", 0)) / 3.0, 1.0)

        # ── 축2: Mobility (8d) ──
        arena_x = np.clip(px / ARENA_SIZE, 0, 1)
        arena_y = np.clip(py / ARENA_SIZE, 0, 1)
        arena_z = np.clip((pz / CM_TO_M) / 400.0, 0, 1)  # Erangel 0~400m

        # 속도 (position 차분)
        vx, vy, speed_ms = compute_velocity(positions_all, pid, snapshot_time)
        speed_norm = min(speed_ms / MAX_SPEED, 1.0)

        veh_speed = p["vehicle_speed"] if p["vehicle_speed"] else 0.0
        in_vehicle = 1.0 if p["vehicle_type"] and p["vehicle_type"] != "" else 0.0

        # radial_speed_to_safe: -(p-c)·v / |p-c|
        safe_cx_m = safe_x / CM_TO_M
        safe_cy_m = safe_y / CM_TO_M
        px_m = px / CM_TO_M
        py_m = py / CM_TO_M
        dp_x = px_m - safe_cx_m
        dp_y = py_m - safe_cy_m
        dp_dist = np.sqrt(dp_x**2 + dp_y**2)
        if dp_dist > 1.0 and speed_ms > 0.1:
            radial_speed = -(dp_x * vx + dp_y * vy) / dp_dist
            radial_speed_norm = np.clip(radial_speed / MAX_SPEED, -1, 1)
        else:
            radial_speed_norm = 0.0

        # time_to_safe_est: d_boundary / (|v| + ε)
        safe_r_m = safe_r / CM_TO_M
        dist_to_safe_boundary = max(0, dp_dist - safe_r_m)
        if dist_to_safe_boundary > 0 and speed_ms > 0.1:
            eta = dist_to_safe_boundary / speed_ms
            time_to_safe = min(eta / 120.0, 1.0)  # 120초 cap
        else:
            time_to_safe = 0.0

        # time_stationary_recent: 이전 스냅샷과 비교해서 정지 판단
        if prev_positions and pid in prev_positions:
            pp = prev_positions[pid]
            move_dist = np.sqrt(
                (px - pp["pos_x"])**2 + (py - pp["pos_y"])**2
            ) / CM_TO_M
            time_stationary = 1.0 if move_dist < 2.0 else 0.0  # 2m 미만 = 정지
        else:
            time_stationary = 0.0

        # ── 축3: Habitat Exposure (5d) ──
        # poison boundary 거리 (정규화)
        if poison_r > 0:
            dp_poison_x = (px - poison_x) / CM_TO_M
            dp_poison_y = (py - poison_y) / CM_TO_M
            dist_poison_center = np.sqrt(dp_poison_x**2 + dp_poison_y**2)
            dist_boundary = max(0, dist_poison_center - poison_r / CM_TO_M)
            inside_boundary = 1.0 if dist_poison_center < poison_r / CM_TO_M else 0.0
        else:
            dist_boundary = 0.0
            inside_boundary = 1.0

        dist_boundary_norm = min(dist_boundary / (ARENA_DIAG / CM_TO_M), 1.0)

        # safe zone 거리 (정규화)
        inside_safe = 1.0 if dp_dist < safe_r_m else 0.0
        dist_safe_norm = min(dist_to_safe_boundary / max(safe_r_m, 1), 1.0)

        # V2: 축4 상세 데미지 + zone damage
        dmg_detail = aggregate_damage_detailed(damage_events, pid, snapshot_time)
        zone_dmg_30s = min(dmg_detail["zone_dmg_30s"] / 100.0, 1.0)

        # ── 축4: Competition Pressure (13d) ──
        dmg_dealt_10s = min(dmg_detail["dealt_10s"] / COMBAT_NORM, 1.0)
        dmg_taken_10s = min(dmg_detail["taken_10s"] / COMBAT_NORM, 1.0)
        dmg_dealt_30s = min(dmg_detail["dealt_30s"] / COMBAT_NORM, 1.0)
        dmg_taken_30s = min(dmg_detail["taken_30s"] / COMBAT_NORM, 1.0)
        n_attackers = min(dmg_detail["n_attackers_30s"] / 8.0, 1.0)
        n_targets = min(dmg_detail["n_targets_30s"] / 8.0, 1.0)
        dmg_taken_decay = dmg_detail["dmg_taken_decay"]
        dmg_dealt_decay = dmg_detail["dmg_dealt_decay"]

        # dist_nearest_ally, dist_team_centroid
        my_team_indices = [j for j in team_members_idx[tid] if j != idx]
        if my_team_indices:
            ally_coords = coords_cm_arr[my_team_indices]
            ally_dists = np.sqrt(((ally_coords - coords_cm_arr[idx])**2).sum(axis=1))
            dist_nearest_ally = (ally_dists.min() / CM_TO_M) / (ARENA_DIAG / CM_TO_M)
            centroid = ally_coords.mean(axis=0)
            centroid = np.append(centroid, coords_cm_arr[idx])  # include self
            team_all_coords = np.vstack([ally_coords, coords_cm_arr[idx:idx+1]])
            team_centroid = team_all_coords.mean(axis=0)
            dist_centroid = np.sqrt(((coords_cm_arr[idx] - team_centroid)**2).sum())
            dist_team_centroid = min(dist_centroid / CM_TO_M / (ARENA_DIAG / CM_TO_M), 1.0)
            dist_nearest_ally = min(dist_nearest_ally, 1.0)
        else:
            dist_nearest_ally = 1.0
            dist_team_centroid = 1.0

        # enemy_count_100m, ally_count_100m, is_isolated
        if tree is not None:
            nearby = tree.query_ball_point(coords_cm_arr[idx], RADIUS_100M_CM)
            enemy_100m = sum(1 for j in nearby if j != idx and team_id_list[j] != tid)
            ally_100m = sum(1 for j in nearby if j != idx and team_id_list[j] == tid)
        else:
            enemy_100m = 0
            ally_100m = 0

        team_size = len(team_members_idx[tid])
        enemy_count_100m = min(enemy_100m / 10.0, 1.0)
        ally_count_100m = min(ally_100m / max(team_size - 1, 1), 1.0)
        is_isolated = 1.0 if ally_100m == 0 and team_size > 1 else 0.0

        # ── 축5: Resource Readiness (8d) ──
        eq = equip_state.get(pid, {})
        weapons = eq.get("weapons", [0.0, 0.0])
        weapon_primary = weapons[0] if len(weapons) > 0 else 0.0
        weapon_secondary = weapons[1] if len(weapons) > 1 else 0.0
        attachment_score = 0.0  # TODO: 부착물 데이터 상세 추적 시 구현
        armor_level = eq.get("armor_level", 0) / 3.0
        helmet_level = eq.get("helmet_level", 0) / 3.0
        backpack_level = eq.get("backpack_level", 0) / 3.0
        heal_use = min(use_state.get("heal_count", 0) / 3.0, 1.0)
        boost_use = min(use_state.get("boost_count", 0) / 3.0, 1.0)

        # ── 39d 피처 벡터 조립 ──
        feat = [
            # 축1: Physiological State (5d)
            health_ratio,           # 0
            shield_ratio,           # 1
            is_groggy,              # 2
            groggy_decay,           # 3
            hp_recovery,            # 4
            # 축2: Mobility (8d)
            arena_x,                # 5
            arena_y,                # 6
            arena_z,                # 7
            speed_norm,             # 8
            in_vehicle,             # 9
            radial_speed_norm,      # 10
            time_to_safe,           # 11
            time_stationary,        # 12
            # 축3: Habitat Exposure (5d)
            dist_boundary_norm,     # 13
            dist_safe_norm,         # 14
            inside_safe,            # 15
            inside_boundary,        # 16
            zone_dmg_30s,           # 17
            # 축4: Competition Pressure (13d)
            dmg_dealt_10s,          # 18
            dmg_taken_10s,          # 19
            dmg_dealt_30s,          # 20
            dmg_taken_30s,          # 21
            n_attackers,            # 22
            n_targets,              # 23
            dmg_taken_decay,        # 24
            dmg_dealt_decay,        # 25
            dist_nearest_ally,      # 26
            dist_team_centroid,     # 27
            enemy_count_100m,       # 28
            ally_count_100m,        # 29
            is_isolated,            # 30
            # 축5: Resource Readiness (8d)
            weapon_primary,         # 31
            weapon_secondary,       # 32
            attachment_score,       # 33
            armor_level,            # 34
            helmet_level,           # 35
            backpack_level,         # 36
            heal_use,               # 37
            boost_use,              # 38
        ]

        node_feats.append(feat)

    data["player"].x = torch.tensor(node_feats, dtype=torch.float32)
    data["player"].num_nodes = n

    data["player"].team_idx = torch.tensor(
        [global_team_to_idx.get(t, 0) for t in team_id_list], dtype=torch.long
    )
    data["player"].global_pid = torch.tensor(
        [global_pid_to_idx[pid] for pid in players], dtype=torch.long
    )
    data["player"].account_ids = players
    data["player"].team_ids = team_id_list

    # ----------------------------------------------------------
    # 글로벌 자기장 컨텍스트 (safe + poison 모두 포함)
    # ----------------------------------------------------------
    safe_area = np.pi * (safe_r / CM_TO_M) ** 2
    poison_area = np.pi * (poison_r / CM_TO_M) ** 2 if poison_r > 0 else 0
    effective_area = poison_area if poison_r > 0 else safe_area
    density = alive_count / max(effective_area, 1.0)

    zone_context = torch.tensor([[
        safe_x / CM_TO_M,       # 0: safe center x
        safe_y / CM_TO_M,       # 1: safe center y
        safe_r / CM_TO_M,       # 2: safe radius
        poison_x / CM_TO_M,     # 3: poison center x
        poison_y / CM_TO_M,     # 4: poison center y
        poison_r / CM_TO_M,     # 5: poison radius
        effective_area,          # 6: effective area
        density,                 # 7: density
        alive_count,             # 8: alive count
        snapshot_time,           # 9: time
    ]], dtype=torch.float32)
    data["zone"].x = zone_context
    data["zone"].num_nodes = 1

    # ----------------------------------------------------------
    # ally 엣지
    # ----------------------------------------------------------
    ally_src, ally_dst = [], []
    ally_feat = []
    for tid, members in team_members_idx.items():
        for i in range(len(members)):
            for j in range(len(members)):
                if i != j:
                    si, di = members[i], members[j]
                    ally_src.append(si)
                    ally_dst.append(di)

                    dist = np.sqrt(((coords_cm_arr[si] - coords_cm_arr[di])**2).sum()) / CM_TO_M
                    alt_diff = (node_feats[si][7] - node_feats[di][7]) * 400.0  # z 역정규화 → m
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
    # encounter 엣지: k-NN (적 팀만)
    # ----------------------------------------------------------
    enc_src, enc_dst = [], []
    enc_feat = []

    if n > 1 and tree is not None:
        for i, pid in enumerate(players):
            query_k = min(n, k + 10)
            dists, indices = tree.query(coords_cm_arr[i], k=query_k)

            enemy_count_k = 0
            for d, j in zip(dists, indices):
                if j == i:
                    continue
                if team_id_list[j] == team_id_list[i]:
                    continue
                if enemy_count_k >= k:
                    break

                enc_src.append(i)
                enc_dst.append(j)

                dist_m = d / CM_TO_M
                alt_diff = (node_feats[i][7] - node_feats[j][7]) * 400.0
                enc_feat.append([dist_m, alt_diff])

                enemy_count_k += 1

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

    # V2: 추가 텔레메트리 (테이블 없으면 빈 리스트)
    groggy_events = load_groggy_events(cur, match_id, match_start_time)
    item_equip_events = load_item_equip(cur, match_id, match_start_time)
    item_use_events = load_item_use(cur, match_id, match_start_time)
    cur.close()

    print(f"  positions: {len(positions):,}, game_states: {len(game_states)}")
    print(f"  kills: {len(kills)}, byzone_deaths: {len(byzone_deaths)}")
    print(f"  teams: {len(team_rank)}, damage_events: {len(damage_events):,}")
    print(f"  V2: groggy={len(groggy_events)}, equip={len(item_equip_events)}, "
          f"item_use={len(item_use_events)}")

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

    print("[4/6] 그래프 생성 중...")

    all_team_ids = sorted(team_rank.keys())
    global_team_to_idx = {t: i for i, t in enumerate(all_team_ids)}

    all_player_ids = sorted(all_players)
    global_pid_to_idx = {pid: i for i, pid in enumerate(all_player_ids)}

    graphs = []
    prev_positions = None  # V2: 이전 스냅샷 위치 (정지 비율용)

    for i, st in enumerate(snapshot_times):
        player_positions = build_player_index(
            positions, st, SNAPSHOT_HALF_WINDOW, death_times
        )

        zone_state = find_closest_zone_state(game_states, st)

        # V2: 장비/아이템 상태 계산
        equip_state = build_equipment_state(item_equip_events, st)
        item_use_state = build_item_use_state(item_use_events, st, lookback=60)

        g = build_snapshot_graph(
            snapshot_time=st,
            player_positions=player_positions,
            zone_state=zone_state,
            team_rank=team_rank,
            damage_events=damage_events,
            positions_all=positions,
            global_team_to_idx=global_team_to_idx,
            global_pid_to_idx=global_pid_to_idx,
            groggy_events=groggy_events,
            equip_state=equip_state,
            item_use_state=item_use_state,
            prev_positions=prev_positions,
            k=K_NEIGHBORS,
        )

        if g is not None:
            g.snapshot_time = st
            g.num_alive = len(player_positions)
            graphs.append(g)

        prev_positions = player_positions  # V2: 다음 스냅샷 정지 비율용

        if (i + 1) % 20 == 0 or i == len(snapshot_times) - 1:
            alive = len(player_positions) if player_positions else 0
            print(f"  [{i+1}/{len(snapshot_times)}] t={st}s, alive={alive}")

    # 히트맵 그리드 계산
    print("[5/6] 히트맵 생성...")
    GRID_SIZE = 80
    MAP_MAX_CM = 816000.0
    cell_size = MAP_MAX_CM / GRID_SIZE

    elev_sum = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    elev_cnt = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    density_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    combat_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)

    for p in positions:
        gx = int(p["pos_x"] / cell_size)
        gy = int(p["pos_y"] / cell_size)
        gx = max(0, min(GRID_SIZE - 1, gx))
        gy = max(0, min(GRID_SIZE - 1, gy))
        z_val = p["pos_z"] if p["pos_z"] else 0
        elev_sum[gy, gx] += z_val
        elev_cnt[gy, gx] += 1
        density_grid[gy, gx] += 1

    elev_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    mask = elev_cnt > 0
    elev_grid[mask] = elev_sum[mask] / elev_cnt[mask]
    if mask.sum() > 0:
        filled = uniform_filter(elev_sum, size=3) / np.maximum(uniform_filter(elev_cnt.astype(float), size=3), 1)
        elev_grid[~mask] = filled[~mask]

    for d in damage_events:
        dmg = d.get("damage") or 0
        if dmg <= 0:
            continue
        ax = d.get("attacker_x")
        ay = d.get("attacker_y")
        if ax and ay:
            gx = int(float(ax) / cell_size)
            gy = int(float(ay) / cell_size)
            gx = max(0, min(GRID_SIZE - 1, gx))
            gy = max(0, min(GRID_SIZE - 1, gy))
            combat_grid[gy, gx] += dmg
        vx = d.get("victim_x")
        vy = d.get("victim_y")
        if vx and vy:
            gx = int(float(vx) / cell_size)
            gy = int(float(vy) / cell_size)
            gx = max(0, min(GRID_SIZE - 1, gx))
            gy = max(0, min(GRID_SIZE - 1, gy))
            combat_grid[gy, gx] += dmg * 0.5

    elev_grid_m = elev_grid / CM_TO_M

    def normalize(g):
        mn, mx = g.min(), g.max()
        if mx - mn < 1e-6:
            return np.zeros_like(g)
        return (g - mn) / (mx - mn)

    heatmaps = {
        "grid_size": GRID_SIZE,
        "cell_size_m": cell_size / CM_TO_M,
        "elevation": normalize(elev_grid_m).round(3).tolist(),
        "density": normalize(density_grid.astype(float)).round(3).tolist(),
        "combat": normalize(combat_grid).round(3).tolist(),
        "elev_min_m": round(float(elev_grid_m[mask].min()) if mask.sum() > 0 else 0, 1),
        "elev_max_m": round(float(elev_grid_m[mask].max()) if mask.sum() > 0 else 0, 1),
    }
    print(f"  그리드: {GRID_SIZE}x{GRID_SIZE}, "
          f"고도 범위: {heatmaps['elev_min_m']}~{heatmaps['elev_max_m']}m, "
          f"전투 셀: {(combat_grid > 0).sum()}")

    # 라벨 구성
    print("[6/6] 라벨 생성...")
    meta = {
        "match_id": match_id,
        "team_rank": team_rank,
        "death_times": death_times,
        "player_team": player_team,
        "survivors": survivors,
        "total_players": len(all_players),
        "snapshot_times": snapshot_times,
        "heatmaps": heatmaps,
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
        # 축1: Physiological (5d)
        "health", "shield", "groggy", "groggy_decay", "hp_recovery",
        # 축2: Mobility (8d)
        "arena_x", "arena_y", "arena_z", "speed", "in_vehicle",
        "radial_spd", "eta_safe", "stationary",
        # 축3: Habitat (5d)
        "dist_bndry", "dist_safe", "in_safe", "in_bndry", "zone_dmg",
        # 축4: Competition (13d)
        "dmg_d_10s", "dmg_t_10s", "dmg_d_30s", "dmg_t_30s",
        "n_atk", "n_tgt", "dmg_t_dec", "dmg_d_dec",
        "d_ally", "d_centroid", "enemy100", "ally100", "isolated",
        # 축5: Resource (8d)
        "wpn_1", "wpn_2", "attach", "armor", "helmet", "backpk",
        "heal", "boost",
    ]
    print(f"\n노드 피처 통계 (첫 스냅샷, {x.shape[0]}노드, {x.shape[1]}d):")
    print(f"{'피처':<15} {'min':>10} {'max':>10} {'mean':>10} {'std':>10}")
    print("-" * 57)
    for j, name in enumerate(feat_names[:x.shape[1]]):
        col = x[:, j]
        print(f"{name:<15} {col.min().item():10.3f} {col.max().item():10.3f} "
              f"{col.mean().item():10.3f} {col.std().item():10.3f}")

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

    # zone context 시계열 (safe + poison 모두 출력)
    print(f"\n자기장 시계열 (safe=흰원/목표, poison=파란원/현재경계):")
    for i in sample_indices:
        g = graphs[i]
        z = g["zone"].x[0]
        print(f"  t={g.snapshot_time:.0f}s: "
              f"safe_r={z[2].item():.0f}m, "
              f"poison_r={z[5].item():.0f}m, "
              f"density={z[7].item():.6f}, "
              f"alive={z[8].item():.0f}")

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

    # 저장
    import os
    os.makedirs("data/graphs", exist_ok=True)
    save_path = f"data/graphs/match_{match_id[:8]}.pt"
    torch.save({
        "graphs": graphs,
        "snapshot_times": snapshot_times,
        "meta": meta,
    }, save_path)
    print(f"\n저장: {save_path}")

    conn.close()