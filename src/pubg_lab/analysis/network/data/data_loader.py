"""
매치 데이터 로더 (v2)
=====================
.pt 그래프 (포지션·ally·enemy) + DB telem_damage (교전) 를 결합해
프레임 시퀀스를 만든다. 시각화 JS가 바로 먹을 수 있는 dict로 반환.

프레임(스냅샷) 하나의 구조:
    {
      "t": 10,
      "alive_mask": [1, 1, 0, ...],          # per player
      "positions": [[x, y], ...],             # per player (arena meter)
      "hp":        [1.0, 0.8, ...],
      "team_alive_count": [...],              # per team
      "team_centroid":    [[x, y], ...],     # per team (alive만 평균, 없으면 None)
      "edges": {
        "ally":   [[i, j], ...],             # undirected player-player (same team, proximity)
        "enemy":  [[i, j], ...],             # undirected player-player (encounter, cross-team)
        "combat": [[i, j, dmg], ...]         # directed: i shot j, window-aggregated dmg
      },
      "zone": {...},
      "metrics": {...}                       # 외부에서 채움
    }

매치 전체 반환 구조는 build_match_data() 참조.
"""

import os
import glob
import torch
import numpy as np

from pubg_lab.analysis.network.config.config import (
    GRAPHS_ROOT,
    ARENA_SIZE_M,
    COMBAT_WINDOW_SEC,
    COMBAT_MIN_DMG,
)
from pubg_lab.analysis.network.data.db_loader import load_match_damage, aggregate_combat_edges


# ============================================================
# 0. 매치 목록
# ============================================================


def list_matches(map_name, mode, limit=None):
    root = os.path.join(GRAPHS_ROOT, map_name, mode)
    paths = sorted(glob.glob(os.path.join(root, "*.pt")))
    if limit is not None:
        paths = paths[:limit]
    return paths


# ============================================================
# 1. 매치 고정 정보 (전체 플레이어 목록, 팀, 랭크)
# ============================================================


def build_match_static(raw):
    """
    .pt의 meta + 첫 스냅샷의 account_ids/team_ids 를 결합해
    매치 전역 고정 정보 구성.

    Returns:
        dict:
            match_id, map_name, game_mode, duration_s, n_players, n_teams
            players: list[{idx, account_id, short, team_idx, team_id}]
            teams:   list[{idx, team_id, rank, members:[player_idx,...]}]
            account_to_idx: {account_id -> player_idx}
    """
    meta = raw["meta"]
    graphs = raw["graphs"]
    snapshot_times = raw["snapshot_times"]

    # 모든 스냅샷에서 플레이어 account_id / team 수집 → 매치 전역 idx 고정
    account_to_team = {}
    seen_order = []
    for g in graphs:
        aids = list(g["player"].account_ids)
        tids = list(g["player"].team_ids)
        for aid, tid in zip(aids, tids):
            if aid not in account_to_team:
                account_to_team[aid] = tid
                seen_order.append(aid)

    # fallback: main.py meta["player_team"] 도 머지
    for aid, tid in meta["player_team"].items():
        if aid not in account_to_team:
            account_to_team[aid] = tid
            seen_order.append(aid)

    # 플레이어 인덱스 고정
    players = []
    account_to_idx = {}
    for i, aid in enumerate(seen_order):
        account_to_idx[aid] = i
        players.append(
            {
                "idx": i,
                "account_id": aid,
                "short": aid.split(".")[-1][:8] if isinstance(aid, str) else str(aid),
                "team_id": account_to_team[aid],
            }
        )

    # 팀 인덱스 고정
    team_id_set = sorted(
        {p["team_id"] for p in players}, key=lambda x: meta["team_rank"].get(x, 99)
    )
    team_to_idx = {tid: i for i, tid in enumerate(team_id_set)}
    teams = []
    for tid in team_id_set:
        teams.append(
            {
                "idx": team_to_idx[tid],
                "team_id": tid,
                "rank": meta["team_rank"].get(tid),
                "members": [p["idx"] for p in players if p["team_id"] == tid],
            }
        )
    for p in players:
        p["team_idx"] = team_to_idx[p["team_id"]]

    duration = snapshot_times[-1] if snapshot_times else 1800.0

    return {
        "match_id": meta["match_id"],
        "map_name": meta.get("map_name", "?"),
        "game_mode": meta.get("game_mode", "?"),
        "arena_size_m": ARENA_SIZE_M,
        "duration_s": float(duration),
        "n_players": len(players),
        "n_teams": len(teams),
        "players": players,
        "teams": teams,
        "account_to_idx": account_to_idx,
        "death_times": dict(meta["death_times"]),
    }


# ============================================================
# 2. 프레임 단위 변환
# ============================================================


def _build_frame(g, elapsed, static, damage_events):
    """
    단일 HeteroData 스냅샷 + DB 데미지 이벤트 → 프레임 dict.
    """
    n_players = static["n_players"]
    n_teams = static["n_teams"]

    # 기본 배열
    alive_mask = [0] * n_players
    positions = [[0.0, 0.0] for _ in range(n_players)]
    hp = [0.0] * n_players

    x = g["player"].x  # [n_alive, 39]
    team_ids_snap = list(g["player"].team_ids)
    account_ids_snap = list(g["player"].account_ids)
    n_alive = x.shape[0]

    # 스냅샷 local_idx → global player idx 매핑
    local_to_global = {}
    for li, aid in enumerate(account_ids_snap):
        gi = static["account_to_idx"].get(aid)
        if gi is None:
            continue
        local_to_global[li] = gi
        alive_mask[gi] = 1
        positions[gi] = [
            float(x[li, 5].item() * ARENA_SIZE_M),
            float(x[li, 6].item() * ARENA_SIZE_M),
        ]
        hp[gi] = float(np.clip(x[li, 0].item(), 0.0, 1.0))

    # ── 엣지: ally (global idx) ──
    ally_pairs = set()
    ei = g["player", "ally", "player"].edge_index
    for e in range(ei.shape[1]):
        s, d = int(ei[0, e].item()), int(ei[1, e].item())
        if s >= n_alive or d >= n_alive:
            continue
        gs, gd = local_to_global.get(s), local_to_global.get(d)
        if gs is None or gd is None or gs == gd:
            continue
        key = (min(gs, gd), max(gs, gd))
        ally_pairs.add(key)

    # ── 엣지: enemy (encounter) ──
    enemy_pairs = set()
    ei = g["player", "encounter", "player"].edge_index
    for e in range(ei.shape[1]):
        s, d = int(ei[0, e].item()), int(ei[1, e].item())
        if s >= n_alive or d >= n_alive:
            continue
        gs, gd = local_to_global.get(s), local_to_global.get(d)
        if gs is None or gd is None or gs == gd:
            continue
        key = (min(gs, gd), max(gs, gd))
        enemy_pairs.add(key)

    # ── 엣지: combat (DB 데미지) ──
    combat_list = []
    if damage_events:
        agg = aggregate_combat_edges(damage_events, elapsed, COMBAT_WINDOW_SEC)
        for (a_aid, v_aid), dmg in agg.items():
            if dmg < COMBAT_MIN_DMG:
                continue
            ai = static["account_to_idx"].get(a_aid)
            vi = static["account_to_idx"].get(v_aid)
            if ai is None or vi is None:
                continue
            combat_list.append([ai, vi, float(dmg)])

    # ── 팀 centroid (alive만 평균) ──
    team_centroids = [None] * n_teams
    team_alive = [0] * n_teams
    team_accum_x = [0.0] * n_teams
    team_accum_y = [0.0] * n_teams
    for p in static["players"]:
        if alive_mask[p["idx"]]:
            ti = p["team_idx"]
            team_accum_x[ti] += positions[p["idx"]][0]
            team_accum_y[ti] += positions[p["idx"]][1]
            team_alive[ti] += 1
    for ti in range(n_teams):
        if team_alive[ti] > 0:
            team_centroids[ti] = [
                team_accum_x[ti] / team_alive[ti],
                team_accum_y[ti] / team_alive[ti],
            ]

    # ── zone ──
    zv = g["zone"].x[0].numpy()
    zone = {
        "sx": float(zv[0]),
        "sy": float(zv[1]),
        "sr": float(zv[2]),
        "px": float(zv[3]),
        "py": float(zv[4]),
        "pr": float(zv[5]),
        "area": float(zv[6]),
        "alive_total": float(zv[8]),
    }

    return {
        "t": float(elapsed),
        "alive_mask": alive_mask,
        "positions": positions,
        "hp": hp,
        "team_alive_count": team_alive,
        "team_centroid": team_centroids,
        "edges": {
            "ally": sorted([list(p) for p in ally_pairs]),
            "enemy": sorted([list(p) for p in enemy_pairs]),
            "combat": combat_list,
        },
        "zone": zone,
    }


# ============================================================
# 3. 매치 전체 빌드
# ============================================================


def build_match_data(pt_path, fetch_damage=True, stride=1):
    """
    .pt + DB → 매치 전체 프레임 시퀀스.

    Parameters:
        pt_path: .pt 파일 경로
        fetch_damage: False면 DB 조회 스킵 (combat 엣지 빈 리스트)
        stride: 스냅샷 샘플링 간격 (1=모든 10초)
    Returns:
        dict: {static: ..., frames: [...]}
    """
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    static = build_match_static(raw)

    damage_events = []
    if fetch_damage:
        dmg_pack = load_match_damage(static["match_id"])
        damage_events = dmg_pack["events"]

    graphs = raw["graphs"]
    times = raw["snapshot_times"]

    frames = []
    for i, (g, t) in enumerate(zip(graphs, times)):
        if stride > 1 and (i % stride) != 0:
            continue
        frames.append(_build_frame(g, t, static, damage_events))

    return {"static": static, "frames": frames}
