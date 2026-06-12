"""
v2 unified pipeline (CLI)
==========================
재설계된 H1·H2·H3 가설 (외곽 파밍 vs 교전 참여 / 다중 교전 중심 / 존버 vs
활발 이동) 을 한 번에 분석하고 webapp/public/data/ 아래로 JSON shard 를 쓴다.

Usage:
    python3 -m network_analysis.run_v2 \
        --map Baltic_Main --mode squad --n 5 \
        --out /Users/lunecid/project/PUBG/webapp/public/data
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np

from network_analysis.config import (
    DEFAULT_MAP, DEFAULT_MODE, DEFAULT_N_MATCHES, OUTPUT_ROOT,
    LANDING_OFFSET_S, MOVEMENT_WINDOW_S,
)
import math

import networkx as nx

from network_analysis.data_loader import list_matches, build_match_data
from network_analysis.team_network import (
    build_team_graph, build_early_team_graph, graph_globals,
)
from network_analysis.team_centrality import (
    compute_team_centralities, attach_centralities_to_rows,
)
from network_analysis.community_detect import detect_for_frame, louvain_partition
from network_analysis.hypotheses import attach_community_features
from network_analysis.db_loader import load_match_damage

from network_analysis.archetype import (
    compute_per_match_archetype_features, assign_archetype_labels,
)
from network_analysis.movement import (
    compute_movement_features, assign_movement_labels,
)
from network_analysis.h1_archetype import compute_h1_archetype
from network_analysis.h2_battle_center import compute_h2_battle_center
from network_analysis.h3_movement import compute_h3_movement
from network_analysis.export_json import write_all


# ============================================================
# 매치 단위 처리
# ============================================================

def _process_match(pt_path: str, weight_mode: str, stride: int):
    """매치 1개 → (match_data, G, rows, globals, events).

    events 는 raw damage event list (account_id 단위). export_json 에서
    team_idx 단위 [src, dst, t, dmg] 로 매핑되어 ST 시각화 용도로 사용.
    """
    md = build_match_data(pt_path, fetch_damage=True, stride=stride)
    match_id = md["static"]["match_id"]

    events = load_match_damage(match_id).get("events", []) or []
    G, rows = build_team_graph(
        md, weight_mode=weight_mode, preloaded_events=events,
    )

    centr = compute_team_centralities(G)
    rows = attach_centralities_to_rows(rows, centr)
    attach_community_features(rows, G, seed=0)
    compute_per_match_archetype_features(rows, md)
    compute_movement_features(rows, md)

    glb = graph_globals(G)
    glb["match_id"] = match_id
    return md, G, rows, glb, events


# ============================================================
# 시간축 동적 팀 상태 (community / position / archetype / movement at t)
# ============================================================

def _team_survival_map(static: dict) -> dict[int, float]:
    """team_idx → survival_seconds. team_network._compute_team_survival 와 동일 의미."""
    duration = float(static.get("duration_s") or 0.0)
    death_times = static.get("death_times", {}) or {}
    players = static.get("players", [])
    idx_to_aid = {p["idx"]: p["account_id"] for p in players}
    out: dict[int, float] = {}
    for t in static.get("teams", []):
        ti = t["idx"]
        members = t.get("members", []) or []
        if not members:
            out[ti] = 0.0
            continue
        ts: list[float] = []
        for m in members:
            aid = idx_to_aid.get(m)
            dt = death_times.get(aid)
            ts.append(duration if dt is None else float(dt))
        out[ti] = max(ts) if ts else 0.0
    return out


def _safe_z(v: float, mean: float, std: float) -> float:
    if std < 1e-9:
        return 0.0
    return (v - mean) / std


def _compute_frame_team_states(
    md: dict,
    events: list[dict],
    frame_stride_sec: int,
    weight_mode: str,
    archetype_meta: dict,
    movement_meta: dict,
) -> dict[int, dict]:
    """매치의 frame_stride_sec 간격마다 동적 팀 상태 dict 를 빌드.

    반환: { frame_t: { str(team_idx): {deg, wdeg, ev, dmg, dmg_taken,
                                       comm, pos, lc, bet,
                                       cp, arc, dpm, mvm, traj} } }
    """
    static = md["static"]
    duration = float(static.get("duration_s") or 0.0)
    md_stub = {"static": static}

    survival_map = _team_survival_map(static)

    # 매 프레임 frame_data 의 team_centroid 시계열로 누적 trajectory 계산
    # ★ 측정 구간: [LANDING_OFFSET_S, LANDING_OFFSET_S + MOVEMENT_WINDOW_S]
    #   비행기·낙하 (전) 와 후반 zone 강제 이동 (후) 모두 제외
    frames_src = md.get("frames", [])
    window_end = LANDING_OFFSET_S + MOVEMENT_WINDOW_S
    trail: dict[int, list[tuple[float, float, float]]] = {}
    for f in frames_src:
        ft = float(f.get("t", 0.0))
        if ft < LANDING_OFFSET_S or ft > window_end:
            continue
        for ti, c in enumerate(f.get("team_centroid") or []):
            if c is None:
                continue
            (trail.setdefault(ti, [])).append((ft, float(c[0]), float(c[1])))
    for arr in trail.values():
        arr.sort(key=lambda r: r[0])

    # 샘플링할 frame_t 들
    last_t = int(round(duration)) if duration > 0 else 0
    sample_times: list[int] = list(range(frame_stride_sec, last_t + 1, frame_stride_sec))
    if not sample_times or sample_times[-1] < last_t:
        sample_times.append(last_t)

    # archetype_meta / movement_meta cutoffs
    q_high = float(archetype_meta.get("q_high_combat_participation", 0.0))
    q_low = float(archetype_meta.get("q_low_combat_participation", 0.0))
    cp_mean = float(archetype_meta.get("combat_per_min_mean", 0.0))
    cp_std = float(archetype_meta.get("combat_per_min_std", 1.0))
    dp_mean = float(archetype_meta.get("dmg_per_min_mean", 0.0))
    dp_std = float(archetype_meta.get("dmg_per_min_std", 1.0))
    dg_mean = float(archetype_meta.get("degree_mean", 0.0))
    dg_std = float(archetype_meta.get("degree_std", 1.0))
    d_median = float(archetype_meta.get("median_dist_from_center_m", 0.0))

    mv_q1 = float(movement_meta.get("q1_displacement_per_min", 0.0))
    mv_q3 = float(movement_meta.get("q3_displacement_per_min", 0.0))

    # 첫 alive centroid 와 map 중심 거리
    arena_m = float(static.get("arena_size_m") or 8160.0)
    cx_map = arena_m / 2.0
    cy_map = arena_m / 2.0
    dist_first: dict[int, float] = {}
    for ti, pts in trail.items():
        if pts:
            dist_first[ti] = math.hypot(pts[0][1] - cx_map, pts[0][2] - cy_map)

    # team_idx → cumulative trajectory length up to time t (greedy 계산)
    # 각 t 별로 정확한 누적 거리 = 이전 frame 까지 합 + 마지막 점에서 t 시점 보간 거리
    # 정확도 vs 비용 trade-off — frame stride 보다 큰 step 이라 stride 단위로 충분
    # 미리 cum_at_t[ti][k] 형태로 저장
    cum_at: dict[int, list[tuple[float, float]]] = {}  # ti -> [(t, cum)]
    for ti, pts in trail.items():
        cum = 0.0
        seq: list[tuple[float, float]] = [(pts[0][0], 0.0)] if pts else []
        for k in range(1, len(pts)):
            dx = pts[k][1] - pts[k - 1][1]
            dy = pts[k][2] - pts[k - 1][2]
            cum += math.hypot(dx, dy)
            seq.append((pts[k][0], cum))
        cum_at[ti] = seq

    def _traj_at(ti: int, t: float) -> float:
        seq = cum_at.get(ti, [])
        if not seq:
            return 0.0
        # 마지막 frame_t <= t 의 cum
        prev = 0.0
        for ft, cv in seq:
            if ft <= t:
                prev = cv
            else:
                break
        return prev

    out: dict[int, dict] = {}
    for ft in sample_times:
        G_t, rows_t = build_early_team_graph(
            md_stub, window_sec=ft,
            weight_mode=weight_mode, preloaded_events=events,
        )
        # community
        team_to_comm: dict[int, int] = {}
        if G_t.number_of_edges() > 0:
            try:
                parts = louvain_partition(G_t, seed=0)
                parts_sorted = sorted(parts, key=lambda s: (-len(s), min(s) if s else 0))
                for cid, comm in enumerate(parts_sorted):
                    for n in comm:
                        team_to_comm[n] = cid
            except Exception:
                pass
        for n in G_t.nodes():
            team_to_comm.setdefault(n, -1)

        # betweenness → position label
        try:
            bet = nx.betweenness_centrality(G_t, weight="weight", normalized=True)
        except Exception:
            bet = {n: 0.0 for n in G_t.nodes()}
        bvals = sorted(bet.values(), reverse=True)
        n = len(bvals)
        hi_cut = bvals[max(0, n // 4 - 1)] if n >= 4 else float("inf")
        lo_cut = bvals[max(0, (3 * n) // 4 - 1)] if n >= 4 else -float("inf")

        # local clustering
        try:
            lc_dict = (
                nx.clustering(G_t, weight="weight")
                if G_t.number_of_edges() > 0 else {}
            )
        except Exception:
            lc_dict = {}

        state: dict[str, dict] = {}
        for r in rows_t:
            ti = int(r["team_idx"])
            deg = int(r.get("degree", 0))
            b = float(bet.get(ti, 0.0))
            if deg == 0:
                pos = "isolated"
            elif b >= hi_cut and n >= 4:
                pos = "core"
            elif b <= lo_cut and n >= 4:
                pos = "peripheral"
            else:
                pos = "mid"

            # 동적 archetype: 누적 events / dmg / degree → z-score → cp_t
            # 기준 시간: 현재까지의 alive 시간 (min(ft, survival_s)) 의 분
            surv = float(survival_map.get(ti, 0.0))
            elapsed = min(ft, max(surv, 1.0)) if surv > 0 else max(ft, 1.0)
            minutes_alive = max(elapsed / 60.0, 1.0 / 60.0)
            ev = int(r.get("combat_events", 0))
            dmg = float(r.get("dmg_dealt", 0.0))
            cp_per_min = ev / minutes_alive
            dp_per_min = dmg / minutes_alive
            cp_t = (
                _safe_z(cp_per_min, cp_mean, cp_std)
                + _safe_z(dp_per_min, dp_mean, dp_std)
                + _safe_z(float(deg), dg_mean, dg_std)
            )

            d_first = dist_first.get(ti, float("nan"))
            if cp_t >= q_high:
                arc = "combat_participant"
            elif (cp_t <= q_low and deg <= 1
                  and not math.isnan(d_first) and d_first > d_median):
                arc = "outer_farmer"
            else:
                arc = "neutral"

            # 동적 movement (윈도우 안에서 alive 였던 시간으로 정규화)
            # 분자: 윈도우 [landing, window_end] 안에서 t 까지 누적된 trajectory
            traj_t_capped = _traj_at(ti, min(ft, window_end))
            in_win_s = max(min(elapsed, window_end) - LANDING_OFFSET_S, 0.0)
            dpm_t = traj_t_capped / max(in_win_s / 60.0, 1.0 / 60.0)
            if not math.isfinite(dpm_t):
                dpm_t = 0.0
            traj = traj_t_capped
            if dpm_t <= mv_q1:
                mvm = "sedentary"
            elif dpm_t >= mv_q3:
                mvm = "mobile"
            else:
                mvm = "mid"

            state[str(ti)] = {
                "deg":        deg,
                "wdeg":       round(float(r.get("weighted_deg", 0.0)), 1),
                "ev":         ev,
                "dmg":        round(dmg, 1),
                "dmg_taken":  round(float(r.get("dmg_taken", 0.0)), 1),
                "comm":       int(team_to_comm.get(ti, -1)),
                "pos":        pos,
                "lc":         round(float(lc_dict.get(ti, 0.0)), 4),
                "bet":        round(b, 4),
                "cp":         round(cp_t, 3),
                "arc":        arc,
                "traj":       round(traj, 1),
                "dpm":        round(dpm_t, 2),
                "mvm":        mvm,
            }
        out[int(ft)] = state
    return out


# ============================================================
# 무결성 어서션
# ============================================================

def _assert_invariants(rows: list[dict], matches_payload, archetype_meta) -> list[str]:
    out: list[str] = []

    # 1) sum(node.degree) == 2 * edges 매치별 검증
    for entry in matches_payload:
        md, G, m_rows = entry[0], entry[1], entry[2]
        deg_sum = sum(int(r.get("degree") or 0) for r in m_rows)
        if deg_sum != 2 * G.number_of_edges():
            out.append(
                f"[!] {md['static']['match_id'][:8]}: sum(degree)={deg_sum} "
                f"!= 2*|E|={2 * G.number_of_edges()}"
            )

    # 2) archetype 분포 sanity (대략 25/50/25)
    counts = archetype_meta.get("archetype_counts", {})
    n_total = sum(counts.values())
    if n_total > 0:
        cp_pct = counts.get("combat_participant", 0) / n_total
        of_pct = counts.get("outer_farmer", 0) / n_total
        if not (0.10 <= cp_pct <= 0.40):
            out.append(f"[?] combat_participant 비율 {cp_pct:.1%} 이상치")
        if not (0.00 <= of_pct <= 0.40):
            out.append(f"[?] outer_farmer 비율 {of_pct:.1%} 이상치")

    # 3) degree==0 팀이 H2 eligible 에서 제외되었는지 (간접 확인)
    deg0_with_bcs = sum(
        1 for r in rows
        if int(r.get("degree") or 0) == 0
        and r.get("battle_center_score") is not None
    )
    # battle_center_score 는 z 기반이므로 degree=0 팀에도 값이 부여됨 (정상).
    # 단 H2 분석시 degree>0 팀만 사용 → h2_battle_center.compute_* 가 보장.
    return out


# ============================================================
# 엔트리포인트
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="v2: 외곽 파밍 vs 교전 참여 · 다중 교전 중심 · 존버 vs 활발",
    )
    ap.add_argument("--map",  default=DEFAULT_MAP)
    ap.add_argument("--mode", default=DEFAULT_MODE)
    ap.add_argument("--n",    type=int, default=DEFAULT_N_MATCHES)
    ap.add_argument("--weight", default="dmg",
                    choices=["dmg", "count", "binary"])
    ap.add_argument("--stride", type=int, default=1,
                    help="snapshot stride (1=10s 마다 모두 사용)")
    ap.add_argument("--frame-stride-s", type=int, default=30,
                    help="JSON shard 의 frame timeline stride (초)")
    ap.add_argument("--out", default=None,
                    help="JSON shard 출력 디렉토리 (기본: webapp/public/data)")
    args = ap.parse_args()

    # 기본 출력 위치
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_out = os.path.join(project_root, "webapp", "public", "data")
    out_dir = args.out or default_out

    print(f"[v2] map={args.map}  mode={args.mode}  n={args.n}  "
          f"weight={args.weight}  stride={args.stride}")
    print(f"[v2] out_dir = {out_dir}\n")

    pts = list_matches(args.map, args.mode, limit=args.n)
    if not pts:
        print(f"[!] no .pt under data/graphs/{args.map}/{args.mode}")
        sys.exit(1)
    print(f"[v2] matches found: {len(pts)}")

    # 1) 매치 단위 처리
    t0 = time.time()
    rows_all: list[dict] = []
    matches_payload: list[tuple[dict, "nx.Graph", list[dict], list[dict]]] = []
    globals_all: list[dict] = []
    for i, pt in enumerate(pts, 1):
        try:
            md, G, m_rows, glb, evs = _process_match(pt, args.weight, args.stride)
        except Exception as e:
            print(f"  [{i}/{len(pts)}] {pt.split('/')[-1]} FAILED: {e}")
            continue
        rows_all.extend(m_rows)
        matches_payload.append((md, G, m_rows, evs))
        globals_all.append(glb)
        print(f"  [{i}/{len(pts)}] {md['static']['match_id'][:8]}  "
              f"|V|={G.number_of_nodes()}  |E|={G.number_of_edges()}  "
              f"density={glb['density']:.3f}  Q={m_rows[0].get('match_modularity_q', 0.0):.2f}  "
              f"frames={len(md.get('frames', []))}  events={len(evs)}")
    print(f"\n[v2] 매치 처리 완료: {time.time()-t0:.1f}s, "
          f"총 {len(rows_all)} 팀 row")

    if not rows_all:
        print("[!] 처리된 row 없음 — 종료")
        sys.exit(1)

    # 2) 데이터셋 단위 라벨 부착 (archetype + movement)
    archetype_meta = assign_archetype_labels(rows_all)
    movement_meta = assign_movement_labels(rows_all)

    print("\n[v2] archetype 분포:")
    for k, v in archetype_meta.get("archetype_counts", {}).items():
        print(f"    {k:20s}: {v}")
    print("[v2] movement 분포:")
    for k, v in movement_meta.get("movement_counts", {}).items():
        print(f"    {k:10s}: {v}")

    # 3) 가설 검증
    h1 = compute_h1_archetype(rows_all)
    h2 = compute_h2_battle_center(rows_all)
    h3 = compute_h3_movement(rows_all)

    print("\n[v2] === H1: 외곽 파밍 vs 교전 참여 ===")
    print(f"  verdict: {h1.get('verdict')}")
    for ln in h1.get("conclusion_lines", []):
        print(f"    · {ln}")
    print("[v2] === H2: 다중 교전 중심 (C) ===")
    print(f"  verdict: {h2.get('verdict')}")
    for ln in h2.get("conclusion_lines", []):
        print(f"    · {ln}")
    print("[v2] === H3: 존버 vs 활발 이동 ===")
    print(f"  verdict: {h3.get('verdict')}")
    for ln in h3.get("conclusion_lines", []):
        print(f"    · {ln}")

    # 4) 무결성 어서션
    issues = _assert_invariants(rows_all, matches_payload, archetype_meta)
    if issues:
        print("\n[v2] sanity issues:")
        for s in issues:
            print(f"    {s}")

    # 5) 시간축 동적 팀 상태 계산
    print("\n[v2] 시간축 team_state 계산 중...")
    state_by_match: dict[str, dict] = {}
    t_state0 = time.time()
    for entry in matches_payload:
        md, _G, _m_rows, evs = entry
        mid = md["static"]["match_id"]
        try:
            state_by_match[mid] = _compute_frame_team_states(
                md, evs, args.frame_stride_s, args.weight,
                archetype_meta, movement_meta,
            )
        except Exception as e:
            print(f"  [!] {mid[:8]} team_state FAILED: {e}")
            state_by_match[mid] = {}
    print(f"[v2] team_state 완료: {time.time()-t_state0:.1f}s")

    # 6) JSON shard 작성
    print("\n[v2] JSON shard 작성 중...")
    summary = write_all(
        out_dir,
        rows=rows_all,
        matches_payload=matches_payload,
        globals_all=globals_all,
        state_by_match=state_by_match,
        h1=h1, h2=h2, h3=h3,
        archetype_meta=archetype_meta,
        movement_meta=movement_meta,
        map_name=args.map,
        game_mode=args.mode,
        weight_mode=args.weight,
        frame_stride_sec=args.frame_stride_s,
    )
    total_mb = summary["total_bytes"] / 1e6
    print(f"[v2] ✅ 작성 완료: {summary['n_files']} files, "
          f"총 {total_mb:.2f} MB")
    print(f"[v2] out_dir: {summary['out_dir']}")
    return summary


if __name__ == "__main__":
    main()
