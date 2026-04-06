"""
Feature Attribution
====================
Integrated Gradients로 스냅샷 내 각 팀의 탈락/생존 요인을 분석.

사용법:
  python3 explain.py --match data/graphs/Baltic_Main/squad-fpp/match_xxx.pt \
                     --checkpoint checkpoints/Baltic_Main/squad-fpp/best_model.pt \
                     --step 25 --team 7
"""

import torch
import numpy as np
import argparse

from model.arena_survival_net import ArenaSurvivalNet
from dataset import SnapshotDataset

# 39d 플레이어 피처 → 5개 카테고리 매핑
# main.py build_snapshot_graph 피처 순서 기준
FEATURE_GROUPS = {
    "생체 상태 (Physiology)":   list(range(0, 5)),     # health, shield, groggy, groggy_decay, hp_recovery
    "이동성 (Mobility)":        list(range(5, 13)),    # arena_xyz, speed, vehicle, radial_speed, time_to_safe, stationary
    "서식 환경 (Habitat)":      list(range(13, 18)),   # dist_boundary, dist_safe, inside_safe, inside_boundary, zone_dmg
    "경쟁 (Competition)":       list(range(18, 31)),   # dmg_dealt/taken, attackers, targets, decay, distances, enemy/ally count, isolated
    "자원 (Resource)":          list(range(31, 39)),   # weapons, attachment, armor, helmet, backpack, heal, boost
}


def integrated_gradients(model, sample, target_team_local_idx,
                         alive_teams, steps=50):
    """
    Integrated Gradients: 입력 피처의 기여도를 계산.

    baseline(zero features) -> actual features 경로를 따라
    gradient를 적분.

    Parameters:
        model: ArenaSurvivalNet
        sample: forward_snapshot용 dict
        target_team_local_idx: int -- alive_teams 내 인덱스
        alive_teams: list[int]
        steps: 적분 스텝 수

    Returns:
        attributions: dict[str, float] -- 카테고리별 기여도 (%)
        raw_attr: np.array [39] -- 피처별 raw attribution
    """
    model.eval()

    # 마지막 스텝의 player_graph에서 타겟 팀 멤버 피처 추출
    last_pg = sample["player_graphs"][-1]
    x_orig = last_pg["player"].x.clone()
    team_idx = last_pg["player"].team_idx
    target_global = alive_teams[target_team_local_idx]
    member_mask = (team_idx == target_global)

    if member_mask.sum() == 0:
        return {g: 0.0 for g in FEATURE_GROUPS}, np.zeros(39)

    # baseline: zero features
    baseline = torch.zeros_like(x_orig)

    # integrated gradients 계산
    accumulated_grads = torch.zeros_like(x_orig[member_mask])

    for k in range(1, steps + 1):
        alpha = k / steps
        interpolated = baseline.clone()
        interpolated[member_mask] = baseline[member_mask] + \
            alpha * (x_orig[member_mask] - baseline[member_mask])
        interpolated.requires_grad_(True)

        # forward
        last_pg["player"].x = interpolated
        hazard_logits, _, _ = model.forward_snapshot(sample)

        # target 팀의 hazard에 대한 gradient
        target_hazard = hazard_logits[target_team_local_idx]
        target_hazard.backward(retain_graph=True)

        if interpolated.grad is not None:
            accumulated_grads += interpolated.grad[member_mask].detach()

        model.zero_grad()

    # 원본 피처 복원
    last_pg["player"].x = x_orig

    # IG = (x - baseline) * mean(gradients)
    diff = x_orig[member_mask] - baseline[member_mask]
    ig = (diff * accumulated_grads / steps).detach()

    # 팀 멤버 평균 -> 피처별 attribution [39]
    raw_attr = ig.mean(dim=0).abs().numpy()

    # 카테고리별 합산 -> 비율
    total = raw_attr.sum() + 1e-8
    category_attr = {}
    for name, indices in FEATURE_GROUPS.items():
        valid_idx = [i for i in indices if i < len(raw_attr)]
        category_attr[name] = float(raw_attr[valid_idx].sum() / total * 100)

    return category_attr, raw_attr


def format_attribution(category_attr, team_idx, phase):
    """Attribution 결과를 텍스트 리포트로."""
    lines = [f"탈락 기여 요인 (Team #{team_idx}, Phase {phase}):"]
    sorted_attr = sorted(category_attr.items(), key=lambda x: -x[1])
    for name, pct in sorted_attr:
        bar_len = int(pct / 5)
        bar = "█" * bar_len
        lines.append(f"  {bar:<20s} {name:<24s} {pct:.1f}%")
    return "\n".join(lines)


def extract_model_interpretables(model, sample):
    """
    모델 내장 해석 정보 추출.

    Returns:
        dict:
            hazard_logits: 전체 팀 hazard
            risk_scores: 전체 팀 risk
            competition_alphas: 팀 간 경쟁 계수 list[Tensor]
    """
    with torch.no_grad():
        hazard_logits, risk_scores, alphas = model.forward_snapshot(sample)

    return {
        "hazard_logits": hazard_logits,
        "risk_scores": risk_scores,
        "competition_alphas": alphas,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Attribution (Integrated Gradients)")
    parser.add_argument("--match", required=True, help=".pt 매치 파일 경로")
    parser.add_argument("--checkpoint", required=True, help="모델 체크포인트 경로")
    parser.add_argument("--step", type=int, default=25, help="분석할 스냅샷 스텝")
    parser.add_argument("--team", type=int, default=0, help="분석할 팀 (alive_teams 내 로컬 인덱스)")
    parser.add_argument("--ig_steps", type=int, default=50, help="IG 적분 스텝 수")
    parser.add_argument("--window_size", type=int, default=5, help="temporal window size")
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

    # 매치 로드
    from dataset import MatchSurvivalData, build_team_graph
    match_data = MatchSurvivalData(args.match)

    step = min(args.step, match_data.n_steps - 1)
    alive_teams = sorted(match_data.get_alive_teams_at(step))

    if len(alive_teams) < 2:
        print(f"Step {step}: alive 팀 부족 ({len(alive_teams)})")
        exit(1)

    # 윈도우 구성
    w_start = max(0, step - args.window_size + 1)
    player_graphs = match_data.graphs[w_start:step + 1]
    while len(player_graphs) < args.window_size:
        player_graphs.insert(0, player_graphs[0])

    zone_seq = torch.stack([g["zone"].x[0] for g in player_graphs])

    # team graph
    last_graph = player_graphs[-1]
    p_team_idx = last_graph["player"].team_idx
    n_teams = p_team_idx.max().item() + 1 if last_graph["player"].x.shape[0] > 0 else 0
    team_graph = build_team_graph(last_graph, p_team_idx, n_teams, set(alive_teams))

    sample = {
        "player_graphs": player_graphs,
        "team_graph": team_graph,
        "zone_seq": zone_seq,
        "alive_teams": alive_teams,
        "meta": {
            "zone_phase": match_data.get_zone_phase(step),
            "elapsed": match_data.snapshot_times[step] if step < len(match_data.snapshot_times) else 0,
            "norm_zone_area": match_data.get_zone_area_normalized(step),
            "n_alive": len(alive_teams),
        },
    }

    # Attribution 계산
    target_local = min(args.team, len(alive_teams) - 1)
    target_global = alive_teams[target_local]
    print(f"\nStep {step}, Team #{target_global} (local idx {target_local})")
    print(f"Alive teams: {len(alive_teams)}")

    # 모델 해석 정보
    interp = extract_model_interpretables(model, sample)
    print(f"\nHazard logits range: [{interp['hazard_logits'].min():.3f}, {interp['hazard_logits'].max():.3f}]")

    # Integrated Gradients
    print(f"\nIntegrated Gradients ({args.ig_steps} steps)...")
    category_attr, raw_attr = integrated_gradients(
        model, sample, target_local, alive_teams, steps=args.ig_steps
    )

    phase = sample["meta"]["zone_phase"]
    print(format_attribution(category_attr, target_global, phase))
