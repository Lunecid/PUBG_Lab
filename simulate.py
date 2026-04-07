"""
Match Simulation
=================
한 매치를 처음부터 끝까지 재생하며 모든 스냅샷에서 예측을 수행.
결과를 JSON으로 저장하여 시각화에 사용.

사용법:
  # 대화형 모드 (인자 없이 — 목록에서 선택)
  python3 simulate.py

  # 직접 지정
  python3 simulate.py --match data/graphs/Baltic_Main/squad-fpp/match_xxx.pt \
                      --checkpoint checkpoints/Baltic_Main/squad-fpp/best_model.pt \
                      --output simulation_result.json
"""

import sys
import torch
import numpy as np
import json
import argparse

from model.arena_survival_net import ArenaSurvivalNet
from dataset import MatchSurvivalData, load_match_meta, build_team_graph
from run_meta import (make_run_meta, stamp_filename,
                      discover_checkpoints, discover_matches)


def extract_team_centroids(graph, alive_teams):
    """
    그래프에서 alive 팀별 중심 좌표 추출.

    Returns:
        dict: {team_idx: (x, y)}
    """
    x = graph["player"].x
    team_idx = graph["player"].team_idx
    positions = {}

    for tidx in alive_teams:
        mask = (team_idx == tidx)
        if mask.sum() == 0:
            continue
        members = x[mask]
        # 피처 인덱스 5=arena_x, 6=arena_y (main.py build_snapshot_graph 기준)
        cx = members[:, 5].mean().item()
        cy = members[:, 6].mean().item()
        positions[tidx] = (cx, cy)

    return positions


def extract_zone_info(graph):
    """zone context에서 자기장 정보 추출."""
    z = graph["zone"].x[0]
    return {
        "safe_x": z[0].item(),
        "safe_y": z[1].item(),
        "safe_radius": z[2].item(),
        "poison_x": z[3].item(),
        "poison_y": z[4].item(),
        "poison_radius": z[5].item(),
        "safe_area": z[6].item(),
    }


def simulate_match(model, match_data, window_size=5, min_alive=3):
    """
    한 매치의 모든 스냅샷을 순차 처리.

    Parameters:
        model: ArenaSurvivalNet (eval mode)
        match_data: MatchSurvivalData
        window_size: temporal window
        min_alive: 최소 alive 팀 수

    Returns:
        frames: list[dict] -- 각 프레임의 예측 결과
        meta: dict -- 매치 메타데이터
    """
    graphs = match_data.graphs
    n_steps = match_data.n_steps
    team_rank = match_data.team_rank
    n_teams = match_data.n_teams

    frames = []
    model.eval()

    for step in range(n_steps):
        alive_teams = sorted(match_data.get_alive_teams_at(step))

        if len(alive_teams) < min_alive:
            continue

        # 윈도우 구성
        w_start = max(0, step - window_size + 1)
        player_graphs = graphs[w_start:step + 1]
        while len(player_graphs) < window_size:
            player_graphs.insert(0, player_graphs[0])

        # zone 시퀀스
        zone_seq = torch.stack([g["zone"].x[0] for g in player_graphs])

        # team graph
        last_graph = player_graphs[-1]
        p_team_idx = last_graph["player"].team_idx
        n_teams_local = p_team_idx.max().item() + 1 if last_graph["player"].x.shape[0] > 0 else 0
        team_graph = build_team_graph(
            last_graph, p_team_idx, n_teams_local, set(alive_teams)
        )

        # 탈락 팀 (현재 step에서 탈락하는 팀)
        dying_teams = []
        for local_idx, tidx in enumerate(alive_teams):
            death_step = match_data.team_death_step.get(tidx, None)
            if death_step is not None and death_step == step:
                dying_teams.append(local_idx)

        # 메타
        zone_phase = match_data.get_zone_phase(step)
        elapsed = match_data.snapshot_times[step] if step < len(match_data.snapshot_times) else 0
        norm_zone_area = match_data.get_zone_area_normalized(step)

        sample = {
            "player_graphs": player_graphs,
            "team_graph": team_graph,
            "zone_seq": zone_seq,
            "alive_teams": alive_teams,
            "meta": {
                "zone_phase": zone_phase,
                "elapsed": elapsed,
                "norm_zone_area": norm_zone_area,
                "n_alive": len(alive_teams),
            },
        }

        with torch.no_grad():
            hazard_logits, risk_scores, alphas = model.forward_snapshot(sample)

        # 팀 위치 추출
        positions = extract_team_centroids(graphs[step], alive_teams)
        zone_info = extract_zone_info(graphs[step])

        # hazard 순위
        sorted_indices = torch.argsort(hazard_logits, descending=True).tolist()

        # 프레임 구성
        teams_data = []
        for i, tidx in enumerate(alive_teams):
            rank_in_hazard = sorted_indices.index(i) + 1
            pos = positions.get(tidx, (0, 0))
            teams_data.append({
                "team_idx": int(tidx),
                "x": pos[0],
                "y": pos[1],
                "hazard": float(risk_scores[i].item()),
                "hazard_rank": rank_in_hazard,
                "is_dying": i in dying_teams,
                "final_rank": int(team_rank.get(tidx, -1)),
            })

        frame = {
            "step": step,
            "elapsed": float(elapsed),
            "phase": zone_phase,
            "n_alive": len(alive_teams),
            "zone": zone_info,
            "teams": teams_data,
            "dying_teams": dying_teams,
        }

        frames.append(frame)

    return frames, {
        "match_id": match_data.match_id,
        "n_teams": n_teams,
        "n_frames": len(frames),
        "total_steps": n_steps,
    }


def save_simulation(frames, sim_meta, output_path, run_meta=None):
    """시뮬레이션 결과 JSON 저장."""
    result = {
        "meta": sim_meta,
        "frames": frames,
    }
    if run_meta is not None:
        result["run_meta"] = run_meta
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"시뮬레이션 저장: {output_path} ({len(frames)} frames)")


# ============================================================
# 대화형 선택
# ============================================================

def _ask_choice(items, prompt_msg, format_fn):
    """
    번호 목록을 출력하고 사용자 선택을 받는 공통 함수.

    Parameters:
        items: list — 선택 대상
        prompt_msg: str — 입력 프롬프트
        format_fn: callable(idx, item) → str — 항목 표시 함수

    Returns:
        선택된 item, 또는 items가 비어있으면 None
    """
    if not items:
        return None

    if len(items) == 1:
        print(f"  [1] {format_fn(0, items[0])}")
        print(f"  → 자동 선택 (1개)")
        return items[0]

    for i, item in enumerate(items):
        marker = "  ← 최신" if i == 0 else ""
        print(f"  [{i+1}] {format_fn(i, item)}{marker}")

    while True:
        try:
            user_input = input(f"\n  {prompt_msg} (번호, Enter=1) > ").strip()
            if not user_input:
                return items[0]
            idx = int(user_input) - 1
            if 0 <= idx < len(items):
                return items[idx]
            print(f"  ⚠ 1~{len(items)} 범위의 번호를 입력하세요.")
        except ValueError:
            print("  ⚠ 숫자를 입력하세요.")
        except EOFError:
            print("  → 자동 선택: 1")
            return items[0]


def ask_checkpoint(checkpoints):
    """대화형 체크포인트 선택."""
    print("\n사용 가능한 체크포인트:")
    return _ask_choice(
        checkpoints,
        "체크포인트 선택",
        lambda i, c: f"{c['map']} / {c['mode']}  ({c['run_id']})",
    )


def ask_match(matches):
    """대화형 매치 선택. 메타데이터(팀 수, 스텝 수) 표시."""
    if not matches:
        return None

    # 매치 메타 로드 (경량)
    print("\n  매치 정보 로딩 중...")
    enriched = []
    for m in matches:
        try:
            meta = load_match_meta(m["path"])
            m["n_teams"] = meta["n_teams"]
            m["n_steps"] = meta["n_steps"]
            m["match_id"] = meta["match_id"]
        except Exception:
            m["n_teams"] = "?"
            m["n_steps"] = "?"
            m["match_id"] = m["filename"]
        enriched.append(m)

    def fmt(i, m):
        mid = m.get("match_id", m["filename"])
        # match_id가 길면 앞 8자 + ...
        if isinstance(mid, str) and len(mid) > 20:
            mid = mid[:17] + "..."
        return f"{m['filename']:<30s} ({m['n_teams']} teams, {m['n_steps']} steps)"

    print(f"\n매치 파일 ({enriched[0]['map']} / {enriched[0]['mode']}):")
    return _ask_choice(enriched, "매치 선택", fmt)


def interactive_select(ckpt_path=None, match_path=None):
    """
    미지정 인자를 대화형으로 보완.

    Returns:
        (ckpt_path, match_path) — 선택된 경로 tuple
    """
    # ── 체크포인트 선택 ──
    if ckpt_path is None:
        checkpoints = discover_checkpoints()
        if not checkpoints:
            print("ERROR: checkpoints/ 에서 best_model.pt를 찾을 수 없습니다.")
            print("  먼저 train.py로 학습을 실행하세요.")
            return None, None
        selected_ckpt = ask_checkpoint(checkpoints)
        if selected_ckpt is None:
            return None, None
        ckpt_path = selected_ckpt["path"]
        ckpt_map = selected_ckpt["map"]
        ckpt_mode = selected_ckpt["mode"]
    else:
        # 주어진 체크포인트에서 map/mode 추출
        ckpt_map, ckpt_mode = None, None
        checkpoints = discover_checkpoints()
        for c in checkpoints:
            if c["path"] == ckpt_path:
                ckpt_map, ckpt_mode = c["map"], c["mode"]
                break

    # ── 매치 선택 ──
    if match_path is None:
        matches = discover_matches(map_filter=ckpt_map, mode_filter=ckpt_mode)
        if not matches:
            print(f"ERROR: 매치 파일을 찾을 수 없습니다.")
            if ckpt_map and ckpt_mode:
                print(f"  경로: data/graphs/{ckpt_map}/{ckpt_mode}/")
            return ckpt_path, None
        selected_match = ask_match(matches)
        if selected_match is None:
            return ckpt_path, None
        match_path = selected_match["path"]

    return ckpt_path, match_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="매치 시뮬레이션")
    parser.add_argument("--match", default=None,
                        help=".pt 매치 파일 경로 (미지정 시 대화형 선택)")
    parser.add_argument("--checkpoint", default=None,
                        help="모델 체크포인트 (미지정 시 대화형 선택)")
    parser.add_argument("--output", default="simulation_result.json")
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--min_alive", type=int, default=3)
    args = parser.parse_args()

    # ── 대화형 선택 (인자 미지정 시) ──
    ckpt_path = args.checkpoint
    match_path = args.match

    if ckpt_path is None or match_path is None:
        ckpt_path, match_path = interactive_select(ckpt_path, match_path)

    if ckpt_path is None or match_path is None:
        sys.exit(1)

    # ── 모델 로드 ──
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    model = ArenaSurvivalNet(
        agent_feat_dim=config["agent_feat_dim"],
        hidden_dim=config["hidden_dim"],
        n_encoder_layers=config["n_encoder_layers"],
        n_group_gnn_layers=config["n_group_gnn_layers"],
        n_gru_layers=config["n_gru_layers"],
        n_heads=config["n_heads"],
        dropout=config["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── Run metadata ──
    run_meta = make_run_meta(
        match=match_path,
        checkpoint=ckpt_path,
        window_size=args.window_size,
        min_alive=args.min_alive,
    )
    if "run_meta" in ckpt:
        run_meta["train_run_meta"] = ckpt["run_meta"]

    # 출력 파일명에 타임스탬프 적용
    output_path = args.output
    if output_path == "simulation_result.json":
        output_path = stamp_filename(output_path, run_meta["run_id"])

    # ── 매치 로드 + 시뮬레이션 ──
    match_data = MatchSurvivalData(match_path)
    print(f"\n매치: {match_data.match_id}")
    print(f"  팀: {match_data.n_teams}, 스텝: {match_data.n_steps}")
    print(f"  run_id: {run_meta['run_id']}")

    frames, sim_meta = simulate_match(
        model, match_data, args.window_size, args.min_alive
    )
    save_simulation(frames, sim_meta, output_path, run_meta=run_meta)
