"""
Match Simulation
=================
한 매치를 처음부터 끝까지 재생하며 모든 스냅샷에서 예측을 수행.
결과를 JSON으로 저장하여 시각화에 사용.

사용법:
  python3 simulate.py --match data/graphs/Baltic_Main/squad-fpp/match_xxx.pt \
                      --checkpoint checkpoints/Baltic_Main/squad-fpp/best_model.pt \
                      --output simulation_result.json
"""

import torch
import numpy as np
import json
import argparse

from model.arena_survival_net import ArenaSurvivalNet
from dataset import MatchSurvivalData, build_team_graph
from run_meta import make_run_meta, stamp_filename


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="매치 시뮬레이션")
    parser.add_argument("--match", required=True, help=".pt 매치 파일 경로")
    parser.add_argument("--checkpoint", required=True, help="모델 체크포인트 경로")
    parser.add_argument("--output", default="simulation_result.json")
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--min_alive", type=int, default=3)
    args = parser.parse_args()

    # 모델 로드
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
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

    # Run metadata
    run_meta = make_run_meta(
        match=args.match,
        checkpoint=args.checkpoint,
        window_size=args.window_size,
        min_alive=args.min_alive,
    )
    # 체크포인트에 저장된 run_meta가 있으면 학습 조건도 포함
    if "run_meta" in ckpt:
        run_meta["train_run_meta"] = ckpt["run_meta"]

    # 출력 파일명에 타임스탬프 적용
    output_path = args.output
    if output_path == "simulation_result.json":
        output_path = stamp_filename(output_path, run_meta["run_id"])

    # 매치 로드 + 시뮬레이션
    match_data = MatchSurvivalData(args.match)
    print(f"매치: {match_data.match_id}")
    print(f"  팀: {match_data.n_teams}, 스텝: {match_data.n_steps}")
    print(f"  run_id: {run_meta['run_id']}")

    frames, sim_meta = simulate_match(
        model, match_data, args.window_size, args.min_alive
    )
    save_simulation(frames, sim_meta, output_path, run_meta=run_meta)
