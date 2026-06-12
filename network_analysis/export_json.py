"""
JSON shard exporter (v2)
=========================
run_v2 가 분석을 마친 후 webapp/public/data/ 아래로 컴팩트한 JSON 파일을 작성.

생성되는 파일 트리:
    <out_dir>/
    ├── index.json                      # 전역 메타 + match_id 목록
    ├── hypotheses.json                 # H1/H2/H3 verdict 헤드라인
    ├── archetype/teams.json            # H1 페이지 — 모든 팀 평탄 테이블
    ├── battle_center/teams.json        # H2 페이지
    ├── movement/teams.json             # H3 페이지
    └── matches/<match_id>.json         # 매치당 1개 (Explorer lazy-fetch)
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Any

import networkx as nx
import numpy as np

from network_analysis.config import MAP_CDN_BY_NAME, MAP_IMAGE_URLS, LANDING_OFFSET_S


# ============================================================
# 직렬화 유틸
# ============================================================

def _safe_num(v, ndigits: int | None = None):
    """None / NaN / inf 안전, 옵션으로 round."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    if ndigits is not None:
        return round(f, ndigits)
    return f


def _safe_int(v, default: int | None = None):
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _sanitize(obj):
    """NaN / Inf → None 재귀 변환 (표준 JSON 호환)."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    return obj


def _atomic_write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            _sanitize(data), f,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    os.replace(tmp, path)


# ============================================================
# Per-match shard
# ============================================================

def _build_match_shard(
    match_data: dict,
    G: nx.Graph,
    rows: list[dict],
    events: list[dict] | None = None,
    frame_stride_sec: int = 30,
    team_state_by_t: dict[int, dict] | None = None,
) -> dict:
    static = match_data["static"]
    duration = float(static.get("duration_s") or 0.0)
    arena_m = float(static.get("arena_size_m") or 8160.0)

    # 팀별 row by team_idx
    by_idx = {r["team_idx"]: r for r in rows}

    nodes = []
    for ti in sorted(by_idx.keys()):
        r = by_idx[ti]
        nodes.append({
            "id":          int(r["team_idx"]),
            "team":        _safe_int(r.get("team_id"), -1),
            "rank":        _safe_int(r.get("final_rank")),
            "n_members":   _safe_int(r.get("n_members"), 0),
            "degree":      _safe_int(r.get("degree"), 0),
            "events":      _safe_int(r.get("combat_events"), 0),
            "dmg":         _safe_num(r.get("dmg_dealt"), 1),
            "dmg_taken":   _safe_num(r.get("dmg_taken"), 1),
            "community":   _safe_int(r.get("community_id"), -1),
            "csize":       _safe_int(r.get("community_size"), 0),
            "position":    r.get("position_label", "mid"),
            "survival":    _safe_num(r.get("survival_seconds"), 1),
            "archetype":   r.get("archetype", "neutral"),
            "movement_label":       r.get("movement_label", "mid"),
            "combat_participation": _safe_num(r.get("combat_participation"), 3),
            "battle_center_score":  _safe_num(r.get("battle_center_score"), 3),
            "trajectory_length_m":  _safe_num(r.get("trajectory_length_m"), 1),
            "displacement_per_min": _safe_num(r.get("displacement_per_min"), 2),
            "position_std_m":       _safe_num(r.get("position_std_m"), 1),
            "dist_from_center_m":   _safe_num(r.get("dist_from_center_m"), 1),
            "centroid_first_xy":    r.get("centroid_first_xy"),
            "centroid_last_xy":     r.get("centroid_last_xy"),
        })

    edges = []
    for u, v, d in G.edges(data=True):
        edges.append({
            "source":  int(u),
            "target":  int(v),
            "weight":  _safe_num(d.get("weight", 1.0), 2),
            "dmg":     _safe_num(d.get("dmg", 0.0), 1),
            "events":  _safe_int(d.get("events"), 0),
        })

    # Frame timeline: 프레임 stride 적용 (≥30초 간격), centroid + alive + zone 만
    frames_out = []
    src_frames = match_data.get("frames", [])
    last_t_emitted = -1e9
    for f in src_frames:
        t = float(f.get("t", 0.0))
        if t - last_t_emitted < frame_stride_sec:
            continue
        last_t_emitted = t
        cx, cy = {}, {}
        for ti, c in enumerate(f.get("team_centroid") or []):
            if c is None:
                continue
            cx[str(ti)] = round(float(c[0]), 1)
            cy[str(ti)] = round(float(c[1]), 1)
        alive = [
            ti for ti, n_alive in enumerate(f.get("team_alive_count") or [])
            if n_alive > 0
        ]
        zone = f.get("zone") or {}
        z_out = None
        if zone:
            z_out = {
                "px": round(float(zone.get("px") or 0.0), 1),
                "py": round(float(zone.get("py") or 0.0), 1),
                "pr": round(float(zone.get("pr") or 0.0), 1),
                "sr": round(float(zone.get("sr") or 0.0), 1),
            }
        ft_int = int(round(t))
        # 동적 team_state 부착 (가장 가까운 ≤ ft_int frame_t)
        ts_attached: dict | None = None
        if team_state_by_t:
            best_key: int | None = None
            for k in team_state_by_t.keys():
                if k <= ft_int and (best_key is None or k > best_key):
                    best_key = k
            if best_key is not None:
                ts_attached = team_state_by_t.get(best_key)
        frame_obj: dict = {
            "t":     ft_int,
            "alive": alive,
            "cx":    cx,
            "cy":    cy,
            "zone":  z_out,
        }
        if ts_attached:
            frame_obj["ts"] = ts_attached
        frames_out.append(frame_obj)

    # Team-level damage event timeline (ST 시각화용)
    team_events_out: list[list] = []
    if events:
        account_to_team = {p["account_id"]: p["team_idx"]
                           for p in static["players"]}
        for e in events:
            a_t = account_to_team.get(e.get("attacker"))
            v_t = account_to_team.get(e.get("victim"))
            if a_t is None or v_t is None or a_t == v_t:
                continue
            team_events_out.append([
                int(a_t), int(v_t),
                round(float(e.get("elapsed", 0.0)), 1),
                round(float(e.get("damage", 0.0)), 1),
            ])
        team_events_out.sort(key=lambda x: x[2])

    return {
        "match_id":  static["match_id"],
        "map_name":  static.get("map_name"),
        "game_mode": static.get("game_mode"),
        "duration_s": int(round(duration)),
        "arena_m":   arena_m,
        "n_nodes":   G.number_of_nodes(),
        "n_edges":   G.number_of_edges(),
        "n_players": _safe_int(static.get("n_players"), 0),
        "modularity_q": _safe_num(rows[0].get("match_modularity_q") if rows else None, 3),
        "frame_stride_s": frame_stride_sec,
        "landing_offset_s": float(LANDING_OFFSET_S),
        "nodes":  nodes,
        "edges":  edges,
        "frames": frames_out,
        "events": team_events_out,
    }


# ============================================================
# Flat team table per hypothesis
# ============================================================

def _flat_table_archetype(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        out.append({
            "match_id":  r.get("match_id"),
            "team_idx":  _safe_int(r.get("team_idx"), -1),
            "team_id":   _safe_int(r.get("team_id"), -1),
            "rank":      _safe_int(r.get("final_rank")),
            "archetype": r.get("archetype", "neutral"),
            "combat_participation": _safe_num(r.get("combat_participation"), 3),
            "combat_per_min":       _safe_num(r.get("combat_per_min"), 3),
            "dmg_per_min":          _safe_num(r.get("dmg_per_min"), 1),
            "degree":               _safe_int(r.get("degree"), 0),
            "dist_from_center_m":   _safe_num(r.get("dist_from_center_m"), 1),
            "survival":             _safe_num(r.get("survival_seconds"), 1),
        })
    return out


def _flat_table_battle_center(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        out.append({
            "match_id":   r.get("match_id"),
            "team_idx":   _safe_int(r.get("team_idx"), -1),
            "team_id":    _safe_int(r.get("team_id"), -1),
            "rank":       _safe_int(r.get("final_rank")),
            "battle_center_score": _safe_num(r.get("battle_center_score"), 3),
            "c_betweenness":       _safe_num(r.get("c_betweenness"), 4),
            "c_degree":            _safe_num(r.get("c_degree"), 4),
            "local_clustering":    _safe_num(r.get("local_clustering"), 4),
            "degree":              _safe_int(r.get("degree"), 0),
            "position_label":      r.get("position_label", "mid"),
        })
    return out


def _flat_table_movement(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        out.append({
            "match_id":  r.get("match_id"),
            "team_idx":  _safe_int(r.get("team_idx"), -1),
            "team_id":   _safe_int(r.get("team_id"), -1),
            "rank":      _safe_int(r.get("final_rank")),
            "movement_label":      r.get("movement_label", "mid"),
            "displacement_per_min": _safe_num(r.get("displacement_per_min"), 2),
            "trajectory_length_m":  _safe_num(r.get("trajectory_length_m"), 1),
            "position_std_m":       _safe_num(r.get("position_std_m"), 1),
            "net_displacement_m":   _safe_num(r.get("net_displacement_m"), 1),
            "n_alive_frames":       _safe_int(r.get("n_alive_frames"), 0),
            "survival":             _safe_num(r.get("survival_seconds"), 1),
            "archetype":            r.get("archetype", "neutral"),
        })
    return out


# ============================================================
# Public entry
# ============================================================

def _aggregate_network_stats(globals_all: list[dict], rows: list[dict]) -> dict:
    """매치별 graph_globals 와 row 의 community feature 로 데이터셋 평균 계산."""
    if not globals_all:
        return {}
    keys = ["n_nodes", "n_edges", "density", "avg_degree",
            "avg_clustering", "diameter", "avg_path_length", "giant_size"]
    stats: dict[str, float] = {}
    for k in keys:
        vals = [
            float(g.get(k) or 0.0)
            for g in globals_all
            if g.get(k) is not None and math.isfinite(float(g.get(k) or 0.0))
        ]
        stats[f"avg_{k}"] = float(sum(vals) / len(vals)) if vals else 0.0

    # avg modularity Q (rows 에 부착된 match_modularity_q 사용)
    qs = []
    seen_match: set = set()
    for r in rows:
        mid = r.get("match_id")
        if mid in seen_match:
            continue
        seen_match.add(mid)
        q = r.get("match_modularity_q")
        if q is not None and math.isfinite(float(q)):
            qs.append(float(q))
    stats["avg_modularity_q"] = float(sum(qs) / len(qs)) if qs else 0.0
    stats["n_matches_in_avg"] = len(globals_all)
    return stats


def write_all(
    out_dir: str,
    *,
    rows: list[dict],
    matches_payload: list,
    globals_all: list[dict] | None = None,
    state_by_match: dict[str, dict] | None = None,
    h1: dict,
    h2: dict,
    h3: dict,
    archetype_meta: dict,
    movement_meta: dict,
    map_name: str,
    game_mode: str,
    weight_mode: str,
    frame_stride_sec: int = 30,
) -> dict:
    """모든 JSON shard 를 out_dir 아래에 작성.

    Returns:
        write summary dict (파일별 byte size)
    """
    os.makedirs(out_dir, exist_ok=True)
    sizes: dict[str, int] = {}

    # 1. matches/<id>.json
    matches_dir = os.path.join(out_dir, "matches")
    os.makedirs(matches_dir, exist_ok=True)
    match_summaries = []
    for entry in matches_payload:
        # backward compat: accept (md, G, rows) or (md, G, rows, events)
        if len(entry) == 4:
            md, G, m_rows, evs = entry
        else:
            md, G, m_rows = entry  # type: ignore[misc]
            evs = []
        ts_for_match = (state_by_match or {}).get(md["static"]["match_id"])
        shard = _build_match_shard(
            md, G, m_rows, evs, frame_stride_sec,
            team_state_by_t=ts_for_match,
        )
        path = os.path.join(matches_dir, f"{shard['match_id']}.json")
        _atomic_write_json(path, shard)
        sizes[os.path.relpath(path, out_dir)] = os.path.getsize(path)
        match_summaries.append({
            "match_id":   shard["match_id"],
            "map_name":   shard["map_name"],
            "duration_s": shard["duration_s"],
            "n_nodes":    shard["n_nodes"],
            "n_edges":    shard["n_edges"],
            "modularity_q": shard["modularity_q"],
        })

    # 2. archetype / battle_center / movement flat tables
    arc_path = os.path.join(out_dir, "archetype", "teams.json")
    bc_path = os.path.join(out_dir, "battle_center", "teams.json")
    mv_path = os.path.join(out_dir, "movement", "teams.json")
    _atomic_write_json(arc_path, _flat_table_archetype(rows))
    _atomic_write_json(bc_path, _flat_table_battle_center(rows))
    _atomic_write_json(mv_path, _flat_table_movement(rows))
    for p in (arc_path, bc_path, mv_path):
        sizes[os.path.relpath(p, out_dir)] = os.path.getsize(p)

    # 3. hypotheses.json
    hyp_path = os.path.join(out_dir, "hypotheses.json")
    _atomic_write_json(hyp_path, {"h1": h1, "h2": h2, "h3": h3})
    sizes[os.path.relpath(hyp_path, out_dir)] = os.path.getsize(hyp_path)

    # 4. index.json
    n_with_rank = sum(1 for r in rows if r.get("final_rank") is not None)
    network_stats = _aggregate_network_stats(globals_all or [], rows)
    map_urls = list(MAP_CDN_BY_NAME.get(map_name) or MAP_IMAGE_URLS)
    idx_path = os.path.join(out_dir, "index.json")
    idx = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "map_name":     map_name,
        "game_mode":    game_mode,
        "weight_mode":  weight_mode,
        "n_matches":    len(matches_payload),
        "n_teams":      len(rows),
        "n_teams_with_rank": n_with_rank,
        "frame_stride_s":    frame_stride_sec,
        "map_urls":          map_urls,
        "network_stats":     network_stats,
        "archetype_meta":    archetype_meta,
        "movement_meta":     movement_meta,
        "verdicts": {
            "h1": h1.get("verdict"),
            "h2": h2.get("verdict"),
            "h3": h3.get("verdict"),
        },
        "matches": match_summaries,
    }
    _atomic_write_json(idx_path, idx)
    sizes[os.path.relpath(idx_path, out_dir)] = os.path.getsize(idx_path)

    return {
        "out_dir": out_dir,
        "n_files": len(sizes),
        "total_bytes": sum(sizes.values()),
        "files": sizes,
    }
