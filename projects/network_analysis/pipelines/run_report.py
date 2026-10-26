"""
Unified report CLI (H1 · H2 · H3)
===================================
N 개 매치에 대해 세 가설을 모두 분석하고, 좌측 메뉴 네비게이션을 갖춘
단일 HTML 리포트로 저장.

Usage:
  python3 -m network_analysis.run_report
  python3 -m network_analysis.run_report --n 10
  python3 -m network_analysis.run_report --n 5 --window-sec 240
"""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

import os
import sys
import time
import argparse
from collections import defaultdict
from datetime import datetime

from pubg_lab.analysis.network.config.config import (
    DEFAULT_MAP,
    DEFAULT_MODE,
    DEFAULT_N_MATCHES,
    OUTPUT_ROOT,
    MAP_IMAGE_URLS,
    MAP_CDN_BY_NAME,
)
from pubg_lab.analysis.network.data.data_loader import list_matches, build_match_data
from pubg_lab.analysis.network.features.team_centrality import MEASURES
from pubg_lab.analysis.network.legacy.h1_analysis import (
    collect_all_matches,
    correlations_for_scatter,
    fourway_correlation_table,
    quartile_group_compare,
    generate_h1_conclusion,
)
from projects.network_analysis.studies.hypotheses import (
    attach_community_features,
    frame_community_map,
    frame_position_map,
)
from pubg_lab.analysis.network.features.team_network import build_early_team_graph
from pubg_lab.analysis.network.data.db_loader import load_match_damage
from pubg_lab.analysis.network.reporting.visualize_match import _load_map_base64
from pubg_lab.analysis.network.legacy.h2_analysis import compute_h2
from pubg_lab.analysis.network.legacy.h3_analysis import compute_h3
from projects.network_analysis.reporting.report import write_report


def _ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    ap = argparse.ArgumentParser(
        description="Unified H1 · H2 · H3 report for PUBG team networks",
    )
    ap.add_argument("--map", default=DEFAULT_MAP)
    ap.add_argument("--mode", default=DEFAULT_MODE)
    ap.add_argument("--n", type=int, default=DEFAULT_N_MATCHES)
    ap.add_argument("--weight", default="dmg", choices=["dmg", "count", "binary"])
    ap.add_argument("--window-sec", type=int, default=300)
    ap.add_argument("--skip-early", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_id = _ts()
    out_path = args.out or os.path.join(
        OUTPUT_ROOT,
        f"report_{args.map}_{args.mode}_{run_id}.html",
    )
    early_window = None if args.skip_early else args.window_sec

    print(
        f"[report] map={args.map} mode={args.mode} n={args.n} "
        f"weight={args.weight} early_window={early_window}s"
    )
    print(f"[report] out: {out_path}\n")

    pts = list_matches(args.map, args.mode, limit=args.n)
    if not pts:
        print(f"[!] no .pt under data/graphs/{args.map}/{args.mode}")
        sys.exit(1)
    print(f"[report] matches found: {len(pts)}")

    # 1. 매치 수집 (H1 이 원하는 full + early feature)
    t0 = time.time()
    collected = collect_all_matches(
        pts,
        build_match_data_fn=build_match_data,
        weight_mode=args.weight,
        early_window_sec=early_window,
        verbose=True,
    )
    print(f"\n[report] 수집 완료: {time.time() - t0:.1f}s")

    # 2. 각 팀 row 에 community / position feature 부착
    rows = collected["rows"]
    # graph index 를 match_id 로 찾아 row-graph 매칭
    mid_to_graph = {mid: g for mid, g in zip(collected["match_ids"], collected["graphs"])}
    # row 가 어느 매치 것인지에 따라 해당 그래프로 community feature 부착
    # 단, hypotheses.attach_community_features 는 (rows subset, G) 단위
    from collections import defaultdict

    rows_by_match = defaultdict(list)
    for r in rows:
        rows_by_match[r["match_id"]].append(r)
    for mid, sub_rows in rows_by_match.items():
        G = mid_to_graph.get(mid)
        if G is None:
            continue
        attach_community_features(sub_rows, G, seed=0)

    # 3. H1 분석
    h1_four = fourway_correlation_table(rows)
    h1_quart_keys = ["degree", "combat_events", "dmg_dealt"] + [f"c_{m}" for m in MEASURES]
    h1_quart = [quartile_group_compare(rows, k) for k in h1_quart_keys]
    h1_scatter_keys = [
        "degree",
        "combat_events",
        "dmg_dealt",
        "c_degree",
        "c_pagerank",
        "c_betweenness",
    ]
    h1_scatter = correlations_for_scatter(rows, h1_scatter_keys)
    h1_verdict, h1_lines, h1_hero = generate_h1_conclusion(
        h1_four,
        h1_quart,
        n_matches=len(collected["match_ids"]),
        n_teams_total=len(rows),
        early_window_sec=early_window,
    )
    h1 = {
        "fourway_table": h1_four,
        "corr_for_scatter": h1_scatter,
        "quartile_results": h1_quart,
        "conclusion_lines": h1_lines,
        "verdict": h1_verdict,
        "hero": h1_hero,
    }

    # 4. H2 분석
    h2 = compute_h2(rows)

    # 5. H3 분석
    h3 = compute_h3(rows)

    # 6. 매치 탐색기용 payload (맵 좌표 기반)
    FRAME_STRIDE_SEC = 30  # 30초 간격 타임라인
    print(f"\n[report] 탐색기 프레임 빌드 (stride={FRAME_STRIDE_SEC}s)...")
    explorer_matches = []

    # 맵 이미지: 공통이므로 첫 매치에만 embed, 나머지는 참조 플래그만
    shared_map_b64 = _load_map_base64()

    # 매치별 static / events / duration 재조회 (collect 때 이미 load 됐지만
    # events 는 캐시되지 않아 다시 로드; 수십 ms 수준)
    for idx, mid in enumerate(collected["match_ids"]):
        G = collected["graphs"][idx]
        m_rows = [r for r in rows if r["match_id"] == mid]

        # 기본 팀 메타
        node_payload = []
        for r in sorted(m_rows, key=lambda x: x["team_idx"]):
            node_payload.append(
                {
                    "id": int(r["team_idx"]),
                    "team": int(r["team_id"]) if r["team_id"] is not None else -1,
                    "rank": int(r["final_rank"]) if r["final_rank"] is not None else None,
                    "degree": int(r.get("degree", 0)),
                    "events": int(r.get("combat_events", 0)),
                    "dmg": float(round(r.get("dmg_dealt", 0.0), 1)),
                    "community": int(r.get("community_id", -1)),
                    "csize": int(r.get("community_size", 0)),
                    "position": r.get("position_label", "mid"),
                    "survival": float(round(r.get("survival_seconds", 0.0), 1)),
                }
            )

        # 매치 전체 기간 최종 edge (reference)
        edge_payload_final = []
        for u, v, d in G.edges(data=True):
            edge_payload_final.append(
                {
                    "source": int(u),
                    "target": int(v),
                    "weight": float(round(d.get("weight", 1.0), 2)),
                    "dmg": float(round(d.get("dmg", 0.0), 1)),
                    "events": int(d.get("events", 0)),
                }
            )

        # ── events (source_team, target_team, elapsed, damage) ──
        # DB 재로드 + 팀 매핑
        damage_pack = load_match_damage(mid)
        raw_events = damage_pack.get("events", [])
        survival_by_team = {
            int(r["team_idx"]): float(r.get("survival_seconds") or 0.0) for r in m_rows
        }

        # 팀 메타 + 프레임별 centroid 를 위해 매치 데이터 재로드.
        # stride=3 (= 30초 간격) 로 충분. team_centroid 는 data_loader
        # 가 각 snapshot 마다 alive 평균 위치로 채워둠.
        match_pt = [pt for pt in pts if pt.endswith(f"{mid[:8]}.pt")]
        match_pt = match_pt[0] if match_pt else pts[idx]
        md_vis = build_match_data(match_pt, fetch_damage=False, stride=3)
        players = md_vis["static"]["players"]
        teams_meta = md_vis["static"]["teams"]
        account_to_team = {p["account_id"]: p["team_idx"] for p in players}
        duration_s = float(md_vis["static"].get("duration_s") or 0.0)
        arena_m = float(md_vis["static"].get("arena_size_m") or 8160.0)
        map_name = md_vis["static"].get("map_name") or ""
        n_teams_total = md_vis["static"].get("n_teams") or len(teams_meta)

        # vis frame: {t -> team_centroid list}
        vis_by_t = {}
        for vf in md_vis["frames"]:
            vis_by_t[int(round(vf["t"]))] = vf

        # event list (팀 수준)
        team_events = []
        for e in raw_events:
            a = account_to_team.get(e["attacker"])
            v = account_to_team.get(e["victim"])
            if a is None or v is None or a == v:
                continue
            team_events.append(
                [int(a), int(v), round(float(e["elapsed"]), 1), round(float(e["damage"]), 1)]
            )
        team_events.sort(key=lambda x: x[2])

        # ── 플레이어 단위 데이터 (사용자 정의: 개인 노드 + ally/combat 분리) ──
        # 각 플레이어의 사망 시각 (account_id → death_seconds)
        death_times = md_vis["static"].get("death_times", {}) or {}
        idx_to_aid = {p["idx"]: p["account_id"] for p in players}
        player_meta = []
        n_players = md_vis["static"]["n_players"]
        for p in players:
            aid = p["account_id"]
            dt = death_times.get(aid)
            survival_p = float(dt) if dt is not None else float(duration_s)
            player_meta.append(
                {
                    "id": int(p["idx"]),
                    "team": int(p["team_idx"]),
                    "team_id": int(p["team_id"]) if p.get("team_id") is not None else -1,
                    "short": p.get("short", ""),
                    "survival": round(survival_p, 1),
                }
            )

        # 프레임별 player 위치 + ally pair (proximity 기반)
        # data_loader._build_frame 가 이미 frame["positions"] / frame["edges"]["ally"] 를 만든다.
        # md_vis 의 frames 를 t -> frame 로 인덱싱.
        vis_frames_for_pos = md_vis["frames"]

        # combat events (player level, directed): attacker → victim 단방향
        account_to_pidx = md_vis["static"].get("account_to_idx", {}) or {}
        player_combat = []  # [src_pidx, dst_pidx, t, dmg]
        for e in raw_events:
            ai = account_to_pidx.get(e["attacker"])
            vi = account_to_pidx.get(e["victim"])
            if ai is None or vi is None or ai == vi:
                continue
            player_combat.append(
                [
                    int(ai),
                    int(vi),
                    round(float(e["elapsed"]), 1),
                    round(float(e["damage"]), 1),
                ]
            )
        player_combat.sort(key=lambda x: x[2])

        # ── 프레임 시퀀스 ──
        last_t = (
            int(duration_s) if duration_s > 0 else int(max([e[2] for e in team_events], default=0))
        )
        # duration 내부로 clamp (초과 프레임이 alive=0 만 보여주는 dead frame
        # 을 만들지 않도록)
        frame_times = list(range(FRAME_STRIDE_SEC, last_t + 1, FRAME_STRIDE_SEC))
        if not frame_times or frame_times[-1] < last_t:
            frame_times.append(last_t)

        frames = []
        # build_early_team_graph 은 raw events 를 elapsed <= t 로 필터해서 동일
        # stub 만 필요 — 최소한의 md 재구성
        md_stub = {"static": md_vis["static"]}
        vis_keys = sorted(vis_by_t.keys())

        # combat 이벤트를 프레임 시점별 누적 (player + team in/out unique pair)
        # frame_times 는 오름차순. events 도 정렬됨.
        ev_idx = 0
        cur_player_out = defaultdict(set)  # pidx → {victim pidx, ...}
        cur_player_in = defaultdict(set)  # pidx → {attacker pidx, ...}
        cur_team_out = defaultdict(set)  # team_idx → {victim team_idx, ...}
        cur_team_in = defaultdict(set)
        team_of_player = {p["idx"]: p["team_idx"] for p in players}

        first_combat_t = None
        for t in frame_times:
            # 누적 events ≤ t
            while ev_idx < len(player_combat) and player_combat[ev_idx][2] <= t:
                src, dst, te, _dmg = player_combat[ev_idx]
                if first_combat_t is None:
                    first_combat_t = te
                cur_player_out[src].add(dst)
                cur_player_in[dst].add(src)
                ts_, td_ = team_of_player.get(src), team_of_player.get(dst)
                if ts_ is not None and td_ is not None and ts_ != td_:
                    cur_team_out[ts_].add(td_)
                    cur_team_in[td_].add(ts_)
                ev_idx += 1

            # team-level 그래프 (기존 가설용 weight 그래프)
            G_t, _ = build_early_team_graph(
                md_stub,
                window_sec=t,
                weight_mode=args.weight,
                preloaded_events=raw_events,
            )
            comm_map = frame_community_map(G_t, seed=0)
            pos_map = frame_position_map(G_t)
            alive = [int(ti) for ti, sv in survival_by_team.items() if sv >= t]

            # team in/out (unique pair count)
            team_io = {}
            for ti in range(int(n_teams_total)):
                team_io[str(ti)] = {
                    "out": len(cur_team_out.get(ti, ())),
                    "in": len(cur_team_in.get(ti, ())),
                }

            # centroid: 가장 가까운 vis frame 에서 추출
            closest = min(vis_keys, key=lambda k: abs(k - t)) if vis_keys else None
            vf = vis_by_t.get(closest) if closest is not None else None
            cx, cy = {}, {}
            ally_pairs = []
            player_pos = {}  # str(pidx) -> [x, y]
            player_alive = []  # alive player idx list
            zone = None
            if vf is not None:
                centroids = vf.get("team_centroid") or []
                for ti, c in enumerate(centroids):
                    if c is None:
                        continue
                    cx[str(ti)] = round(float(c[0]), 1)
                    cy[str(ti)] = round(float(c[1]), 1)
                # player 위치
                positions = vf.get("positions") or []
                am = vf.get("alive_mask") or []
                for pi, pos in enumerate(positions):
                    if pi < len(am) and am[pi]:
                        player_pos[str(pi)] = [round(float(pos[0]), 1), round(float(pos[1]), 1)]
                        player_alive.append(int(pi))
                # ally pairs (proximity 기반, 같은 팀)
                for s, d in vf.get("edges", {}).get("ally") or []:
                    ally_pairs.append([int(s), int(d)])
                # zone
                zv = vf.get("zone") or {}
                if zv:
                    zone = {
                        "sx": round(zv.get("sx", 0.0), 1),
                        "sy": round(zv.get("sy", 0.0), 1),
                        "sr": round(zv.get("sr", 0.0), 1),
                        "px": round(zv.get("px", 0.0), 1),
                        "py": round(zv.get("py", 0.0), 1),
                        "pr": round(zv.get("pr", 0.0), 1),
                    }

            # player in/out
            player_io = {}
            for pi in player_alive:
                player_io[str(pi)] = {
                    "out": len(cur_player_out.get(pi, ())),
                    "in": len(cur_player_in.get(pi, ())),
                }

            frames.append(
                {
                    "t": int(t),
                    "alive": alive,
                    "comm": {str(k): v for k, v in comm_map.items()},
                    "pos": {str(k): v for k, v in pos_map.items()},
                    "n_edges": G_t.number_of_edges(),
                    "cx": cx,
                    "cy": cy,
                    "team_io": team_io,
                    # player level
                    "p_alive": player_alive,
                    "p_pos": player_pos,
                    "p_io": player_io,
                    "ally": ally_pairs,
                    "zone": zone,
                }
            )

        # ── sanity 로그 ──
        first_alive_pos = None
        for f in frames:
            if f["p_alive"]:
                pi = f["p_alive"][0]
                p = f["p_pos"].get(str(pi))
                if p:
                    first_alive_pos = (pi, p[0], p[1], f["t"])
                    break
        sx, sy = ("—", "—")
        if first_alive_pos:
            _, fx, fy, _ft = first_alive_pos
            sx = round(fx / arena_m * 1000, 1)
            sy = round(fy / arena_m * 1000, 1)
        print(
            f"  [{idx + 1}/{len(collected['match_ids'])}] {mid[:12]}  "
            f"map={map_name}  arena={int(arena_m)}m"
        )
        if first_alive_pos:
            pi, fx, fy, ft = first_alive_pos
            print(f"    first_player P{pi} @ t={ft}s = ({fx:.0f}m, {fy:.0f}m) → SVG ({sx}, {sy})")
        print(
            f"    duration={last_t}s  frames={len(frames)}  "
            f"team_events={len(team_events)}  player_events={len(player_combat)}  "
            f"first_combat_at={first_combat_t}s"
        )

        # 맵 base64: 매치별 map_name 의 assets 파일 자동 검색 (첫 매치만)
        if idx == 0:
            assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
            map_b64_resolved = None
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                cand = os.path.join(assets_dir, f"{map_name}{ext}")
                if os.path.exists(cand):
                    map_b64_resolved = _load_map_base64(cand)
                    if map_b64_resolved:
                        break
            # 폴백: config 의 일반 MAP_IMAGE_PATH
            if map_b64_resolved is None:
                map_b64_resolved = shared_map_b64
        else:
            map_b64_resolved = None

        # 맵 URL chain: map_name 우선, 없으면 일반 URLS
        if idx == 0:
            map_urls_resolved = MAP_CDN_BY_NAME.get(map_name) or list(MAP_IMAGE_URLS)
        else:
            map_urls_resolved = []

        explorer_matches.append(
            {
                "match_id": mid[:12],
                "duration_s": last_t,
                "arena_m": float(arena_m),
                "map_name": map_name,
                "map_b64": map_b64_resolved,
                "map_urls": map_urls_resolved,
                "n_nodes": G.number_of_nodes(),
                "n_edges": G.number_of_edges(),
                "n_players": int(n_players),
                "modularity": round(
                    collected["communities"][idx].get("louvain", {}).get("modularity_q", 0.0), 3
                ),
                "nodes": node_payload,
                "players": player_meta,
                "edges": edge_payload_final,
                "events": team_events,  # team-level [src_t, dst_t, t, dmg]
                "p_events": player_combat,  # player-level [src_p, dst_p, t, dmg]
                "frames": frames,
                "frame_stride": FRAME_STRIDE_SEC,
            }
        )

    print(f"\n[report] H1 verdict: {h1_verdict}")
    print(f"[report] H2 verdict: {h2.get('verdict')}")
    print(f"[report] H3 verdict: {h3.get('verdict')}")

    print("\nH2 결론:")
    for l in h2.get("conclusion_lines", []):
        print(f"  · {l}")
    print("H3 결론:")
    for l in h3.get("conclusion_lines", []):
        print(f"  · {l}")

    # 7. 리포트 쓰기
    analyses = {"h1": h1, "h2": h2, "h3": h3, "explorer_matches": explorer_matches}
    write_report(collected, analyses, out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n[report] ✅ {out_path} ({size_mb:.2f} MB)")
    return out_path


if __name__ == "__main__":
    main()
