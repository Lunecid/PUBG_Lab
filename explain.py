"""
Feature Attribution
====================
Integrated Gradients로 스냅샷 내 각 팀의 탈락/생존 요인을 분석.

사용법:
  # 전체 탈락 이벤트 일괄 분석 (기본)
  python3 explain.py --match data/graphs/Baltic_Main/squad-fpp/match_xxx.pt \
                     --checkpoint checkpoints/Baltic_Main/squad-fpp/best_model.pt

  # 특정 스텝/팀만 분석
  python3 explain.py --match data/graphs/Baltic_Main/squad-fpp/match_xxx.pt \
                     --checkpoint checkpoints/Baltic_Main/squad-fpp/best_model.pt \
                     --step 25 --team 7

  # 인자 없이 실행 (최신 체크포인트 + 첫 번째 매치 자동 선택)
  python3 explain.py
"""

import os
import sys
import glob
import json
import torch
import numpy as np
import argparse

from model.arena_survival_net import ArenaSurvivalNet
from dataset import build_team_graph
from run_meta import make_run_meta, stamp_filename

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

    baseline(zero features) → actual features 경로를 따라
    gradient를 적분.

    Parameters:
        model: ArenaSurvivalNet
        sample: forward_snapshot용 dict
        target_team_local_idx: int — alive_teams 내 인덱스
        alive_teams: list[int]
        steps: 적분 스텝 수

    Returns:
        attributions: dict[str, float] — 카테고리별 기여도 (%)
        raw_attr: np.array [39] — 피처별 raw attribution
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

    # 팀 멤버 평균 → 피처별 attribution [39]
    raw_attr = ig.mean(dim=0).abs().numpy()

    # 카테고리별 합산 → 비율
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


def analyze_all_eliminations(model, match_data, window_size=5, ig_steps=30):
    """
    매치 내 모든 탈락 이벤트에 대해 attribution 일괄 계산.

    Parameters:
        model: ArenaSurvivalNet (eval mode)
        match_data: torch.load 결과 dict (graphs, snapshot_times, meta)
        window_size: temporal window
        ig_steps: Integrated Gradients 적분 스텝 수

    Returns:
        events: list[dict] — 각 탈락 이벤트의 attribution
        summary: dict — 카테고리별 평균 기여도
    """
    graphs = match_data["graphs"]
    snapshot_times = match_data["snapshot_times"]
    meta = match_data["meta"]
    team_rank = meta["team_rank"]
    death_times = meta["death_times"]
    n_teams = len(team_rank)

    # team_death_step 계산
    team_death_step = {}
    for tid, dt in death_times.items():
        if dt is not None:
            for step_i, st in enumerate(snapshot_times):
                if st >= dt:
                    team_death_step[tid] = step_i
                    break

    events = []
    category_sums = {g: 0.0 for g in FEATURE_GROUPS}
    n_analyzed = 0

    for step in range(len(graphs)):
        alive_teams = sorted([
            tid for tid in team_rank.keys()
            if tid not in team_death_step or team_death_step[tid] > step
        ])

        if len(alive_teams) < 3:
            continue

        # 이 스텝에서 탈락한 팀 찾기
        dying_local = []
        for local_idx, tidx in enumerate(alive_teams):
            if tidx in team_death_step:
                ds = team_death_step[tidx]
                if ds == step or ds == step + 1:
                    dying_local.append((local_idx, tidx))

        if not dying_local:
            continue

        # 스냅샷 샘플 구성
        w_start = max(0, step - window_size + 1)
        player_graphs = graphs[w_start:step + 1]
        while len(player_graphs) < window_size:
            player_graphs.insert(0, player_graphs[0])

        zone_seq = torch.stack([g["zone"].x[0] for g in player_graphs])

        last_graph = player_graphs[-1]
        alive_set = set(alive_teams)
        p_team_idx = last_graph["player"].team_idx
        n_teams_local = p_team_idx.max().item() + 1 if last_graph["player"].x.shape[0] > 0 else 0
        team_graph = build_team_graph(
            last_graph, p_team_idx, n_teams_local, alive_set,
        )

        sample = {
            "player_graphs": player_graphs,
            "team_graph": team_graph,
            "zone_seq": zone_seq,
            "alive_teams": alive_teams,
            "meta": {
                "zone_phase": 0,
                "elapsed": snapshot_times[step],
                "norm_zone_area": 1.0,
                "n_alive": len(alive_teams),
            },
        }

        # 각 탈락 팀에 대해 attribution 계산
        for local_idx, global_idx in dying_local:
            cat_attr, raw = integrated_gradients(
                model, sample, local_idx, alive_teams, steps=ig_steps
            )

            events.append({
                "step": step,
                "team_idx": int(global_idx),
                "elapsed": float(snapshot_times[step]),
                "n_alive": len(alive_teams),
                "attribution": cat_attr,
            })

            for g in FEATURE_GROUPS:
                category_sums[g] += cat_attr[g]
            n_analyzed += 1

            print(f"  Step {step:3d} | Team #{global_idx:2d} | "
                  f"alive={len(alive_teams):2d} | analyzed")

    # 전체 평균
    summary = {}
    if n_analyzed > 0:
        for g in FEATURE_GROUPS:
            summary[g] = category_sums[g] / n_analyzed

    return events, summary


def find_default_paths():
    """
    인자 없이 실행 시 기본 경로 자동 탐색.
    checkpoints/에서 가장 최근 best_model.pt,
    data/graphs/에서 첫 번째 .pt 매치 파일.
    """
    # 체크포인트 탐색
    ckpt_candidates = glob.glob("checkpoints/**/best_model.pt", recursive=True)
    if not ckpt_candidates:
        print("ERROR: checkpoints/ 에서 best_model.pt를 찾을 수 없습니다.")
        print("  먼저 train.py로 학습을 실행하세요.")
        return None, None
    ckpt_path = sorted(ckpt_candidates)[-1]  # 가장 최근

    # 매치 파일 탐색 (체크포인트 경로에서 map/mode 추출)
    # checkpoints/Baltic_Main/squad-fpp/best_model.pt
    #           → data/graphs/Baltic_Main/squad-fpp/*.pt
    parts = ckpt_path.replace("checkpoints/", "").replace("/best_model.pt", "")
    data_dir = os.path.join("data/graphs", parts)

    match_candidates = glob.glob(os.path.join(data_dir, "*.pt"))
    if not match_candidates:
        print(f"ERROR: {data_dir} 에서 매치 파일을 찾을 수 없습니다.")
        return ckpt_path, None
    match_path = sorted(match_candidates)[0]

    return ckpt_path, match_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Attribution 분석")
    parser.add_argument("--match", default=None, help=".pt 매치 파일 경로 (미지정 시 자동 탐색)")
    parser.add_argument("--checkpoint", default=None, help="모델 체크포인트 (미지정 시 자동 탐색)")
    parser.add_argument("--step", type=int, default=None, help="분석할 스텝 (미지정 시 전체 탈락 이벤트)")
    parser.add_argument("--team", type=int, default=None, help="분석할 팀 인덱스 (미지정 시 탈락 팀 전체)")
    parser.add_argument("--output", default=None, help="결과 JSON 저장 경로")
    parser.add_argument("--ig_steps", type=int, default=30, help="Integrated Gradients 적분 스텝")
    args = parser.parse_args()

    # 경로 자동 탐색
    ckpt_path = args.checkpoint
    match_path = args.match

    if ckpt_path is None or match_path is None:
        auto_ckpt, auto_match = find_default_paths()
        ckpt_path = ckpt_path or auto_ckpt
        match_path = match_path or auto_match

    if ckpt_path is None or match_path is None:
        sys.exit(1)

    # Run metadata
    run_meta = make_run_meta(
        match=match_path,
        checkpoint=ckpt_path,
        step=args.step,
        team=args.team,
        ig_steps=args.ig_steps,
    )

    print(f"매치: {match_path}")
    print(f"체크포인트: {ckpt_path}")
    print(f"run_id: {run_meta['run_id']}")

    # 모델 로드
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

    # 매치 로드
    match_data = torch.load(match_path, map_location="cpu", weights_only=False)

    if args.step is not None and args.team is not None:
        # ── 단일 팀/스텝 분석 ──
        graphs = match_data["graphs"]
        snapshot_times = match_data["snapshot_times"]
        meta = match_data["meta"]
        team_rank = meta["team_rank"]
        death_times = meta["death_times"]

        # team_death_step 계산
        team_death_step = {}
        for tid, dt in death_times.items():
            if dt is not None:
                for step_i, st in enumerate(snapshot_times):
                    if st >= dt:
                        team_death_step[tid] = step_i
                        break

        step = min(args.step, len(graphs) - 1)
        alive_teams = sorted([
            tid for tid in team_rank.keys()
            if tid not in team_death_step or team_death_step[tid] > step
        ])

        if len(alive_teams) < 2:
            print(f"Step {step}: alive 팀 부족 ({len(alive_teams)})")
            sys.exit(1)

        window_size = 5
        w_start = max(0, step - window_size + 1)
        player_graphs = graphs[w_start:step + 1]
        while len(player_graphs) < window_size:
            player_graphs.insert(0, player_graphs[0])

        zone_seq = torch.stack([g["zone"].x[0] for g in player_graphs])

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
                "zone_phase": 0,
                "elapsed": snapshot_times[step],
                "norm_zone_area": 1.0,
                "n_alive": len(alive_teams),
            },
        }

        target_local = min(args.team, len(alive_teams) - 1)
        target_global = alive_teams[target_local]

        print(f"\n단일 분석: step={step}, Team #{target_global} (local idx {target_local})")
        print(f"Alive teams: {len(alive_teams)}")

        # 모델 해석 정보
        interp = extract_model_interpretables(model, sample)
        print(f"Hazard logits range: [{interp['hazard_logits'].min():.3f}, {interp['hazard_logits'].max():.3f}]")

        # Integrated Gradients
        print(f"\nIntegrated Gradients ({args.ig_steps} steps)...")
        category_attr, raw_attr = integrated_gradients(
            model, sample, target_local, alive_teams, steps=args.ig_steps
        )

        print(format_attribution(category_attr, target_global, 0))

    else:
        # ── 전체 탈락 이벤트 일괄 분석 ──
        print(f"\n전체 탈락 이벤트 분석 중...")
        events, summary = analyze_all_eliminations(
            model, match_data, ig_steps=args.ig_steps
        )

        # 결과 출력
        print(f"\n{'='*50}")
        print(f"분석 완료: {len(events)}건의 탈락 이벤트")
        print(f"{'='*50}")

        print(f"\n전체 평균 탈락 기여 요인:")
        sorted_summary = sorted(summary.items(), key=lambda x: -x[1])
        for name, pct in sorted_summary:
            bar_len = int(pct / 3)
            bar = "█" * bar_len
            print(f"  {bar:<30s} {name:<24s} {pct:.1f}%")

        # 개별 이벤트 출력 (최근 5건)
        print(f"\n개별 탈락 분석 (최근 5건):")
        for ev in events[-5:]:
            print(f"\n  Step {ev['step']} | Team #{ev['team_idx']} | "
                  f"alive={ev['n_alive']}")
            sorted_attr = sorted(ev["attribution"].items(), key=lambda x: -x[1])
            for name, pct in sorted_attr:
                bar_len = int(pct / 5)
                bar = "█" * bar_len
                print(f"    {bar:<20s} {name:<24s} {pct:.1f}%")

        # JSON 저장 (run_meta 포함)
        if args.output:
            output_path = args.output
        else:
            base_name = os.path.basename(match_path).replace(".pt", "_attribution.json")
            output_path = stamp_filename(base_name, run_meta["run_id"])
            output_path = os.path.join(os.path.dirname(match_path), output_path)

        # 체크포인트에 저장된 학습 조건도 포함
        train_run_meta = ckpt.get("run_meta", None)

        with open(output_path, "w") as f:
            json.dump({
                "run_meta": run_meta,
                "train_run_meta": train_run_meta,
                "events": events,
                "summary": summary,
            }, f, indent=2)
        print(f"\n결과 저장: {output_path}")
