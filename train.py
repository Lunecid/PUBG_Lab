"""
Training Loop
==============
ArenaSurvivalNet 학습 파이프라인.

기능:
  - Match 단위 chrono split (train/val/test)
  - 3-loss 학습 (survival + rank + encounter)
  - Epoch별 검증 + 메트릭 계산
  - Early stopping + 체크포인트 저장
  - 재현성: seed 고정

사용법:
  python3 train.py --pt_dir data/graphs --epochs 50 --hidden_dim 128

CLAUDE.md §15 하이퍼파라미터 시작점:
  hidden_dim=64~128, GNN 2~3층, GRU 1층, L=5, k=5
  lr=1e-3 (Adam), λ₁=0.1, λ₂=0.05, batch=32~64
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from model.arena_survival_net import ArenaSurvivalNet
from dataset import TeamSurvivalDataset, collate_survival_batch
from metrics import evaluate_survival_model, format_eval_results


# ============================================================
# 1. 재현성 시드 고정
# ============================================================

def set_seed(seed=42):
    """torch, numpy, python 시드 고정."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# 2. Match 단위 Split
# ============================================================

def split_dataset_by_match(dataset, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    매치 단위 chrono split.
    같은 매치의 모든 샘플은 같은 split에 배치.

    Returns:
        train_indices, val_indices, test_indices
    """
    # 매치별 샘플 인덱스 그룹핑
    match_to_indices = {}
    for idx, (mi, tidx, step) in enumerate(dataset.samples):
        if mi not in match_to_indices:
            match_to_indices[mi] = []
        match_to_indices[mi].append(idx)

    # 매치 인덱스 순서 (MatchSurvivalData 로딩 순 = 파일명 순)
    match_indices = sorted(match_to_indices.keys())
    n_matches = len(match_indices)

    # chrono split (파일명 순 = 시간순 가정)
    n_train = max(1, int(n_matches * train_ratio))
    n_val = max(1, int(n_matches * val_ratio)) if n_matches > 1 else 0
    # test는 나머지
    n_test = n_matches - n_train - n_val
    if n_test < 0:
        n_val = n_matches - n_train
        n_test = 0

    train_matches = match_indices[:n_train]
    val_matches = match_indices[n_train:n_train + n_val]
    test_matches = match_indices[n_train + n_val:]

    train_idx = []
    val_idx = []
    test_idx = []

    for mi in train_matches:
        train_idx.extend(match_to_indices[mi])
    for mi in val_matches:
        val_idx.extend(match_to_indices[mi])
    for mi in test_matches:
        test_idx.extend(match_to_indices[mi])

    return train_idx, val_idx, test_idx


# ============================================================
# 3. 평가 루프
# ============================================================

@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    검증/테스트 평가.

    Returns:
        loss_dict: dict with mean losses
        eval_results: dict with survival metrics
    """
    model.eval()

    all_hazard_logits = []
    all_risk_scores = []
    all_events = []
    all_times = []
    all_phases = []
    all_ranks = []
    all_match_ids = []
    total_loss = 0
    total_survival = 0
    total_rank = 0
    n_batches = 0

    for batch in dataloader:
        # device 이동 (zone_seqs만 — 그래프는 CPU)
        batch["zone_seqs"] = batch["zone_seqs"].to(device)
        batch["events"] = batch["events"].to(device)
        batch["at_risks"] = batch["at_risks"].to(device)

        loss_dict = model.compute_loss(batch)

        total_loss += loss_dict["total"].item()
        total_survival += loss_dict["survival"].item()
        total_rank += loss_dict["rank"].item()
        n_batches += 1

        all_hazard_logits.append(loss_dict["hazard_logits"].cpu().numpy())
        all_risk_scores.append(loss_dict["risk_scores"].cpu().numpy())
        all_events.append(batch["events"].cpu().numpy())

        for m in batch["metas"]:
            all_times.append(m["elapsed"])
            all_phases.append(m["zone_phase"])
            all_ranks.append(m["final_rank"])
            all_match_ids.append(m["match_id"])

    if n_batches == 0:
        return {}, {}

    # 집계
    hazard_logits = np.concatenate(all_hazard_logits)
    risk_scores = np.concatenate(all_risk_scores)
    events = np.concatenate(all_events)
    times = np.array(all_times)
    phases = np.array(all_phases)
    ranks = np.array(all_ranks, dtype=float)
    match_ids = np.array(all_match_ids)

    loss_means = {
        "total": total_loss / n_batches,
        "survival": total_survival / n_batches,
        "rank": total_rank / n_batches,
    }

    # 메트릭 계산
    eval_results = evaluate_survival_model(
        hazard_logits=hazard_logits,
        events=events,
        times=times,
        phases=phases,
        risk_scores=risk_scores,
        final_ranks=ranks,
        match_ids=match_ids,
    )

    return loss_means, eval_results


# ============================================================
# 4. 학습 루프
# ============================================================

def train(
    pt_dir,
    output_dir="checkpoints",
    # 모델 하이퍼파라미터
    hidden_dim=128,
    n_encoder_layers=2,
    n_group_gnn_layers=2,
    n_gru_layers=1,
    n_heads=4,
    dropout=0.1,
    # 학습 하이퍼파라미터
    epochs=50,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=32,
    lambda_rank=0.1,
    lambda_encounter=0.05,
    # 데이터
    window_size=5,
    skip_first_steps=10,
    stride=2,
    min_alive_teams=3,
    # 기타
    patience=10,
    seed=42,
    device=None,
):
    """
    전체 학습 파이프라인.

    Returns:
        model: 학습된 모델
        history: dict with loss/metric history
    """
    set_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Device: {device}")

    # ── 데이터셋 로드 ──
    print("\n[1/4] 데이터셋 구성...")
    dataset = TeamSurvivalDataset(
        pt_dir=pt_dir,
        window_size=window_size,
        min_alive_teams=min_alive_teams,
        skip_first_steps=skip_first_steps,
        stride=stride,
    )

    if len(dataset) == 0:
        print("ERROR: 데이터셋이 비어 있습니다.")
        return None, None

    # ── Split ──
    train_idx, val_idx, test_idx = split_dataset_by_match(dataset, seed=seed)
    print(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_survival_batch,
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_survival_batch,
        num_workers=0,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_survival_batch,
        num_workers=0,
    ) if test_idx else None

    # ── 모델 생성 ──
    print("\n[2/4] 모델 생성...")
    model = ArenaSurvivalNet(
        agent_feat_dim=dataset.PLAYER_FEAT_DIM,
        hidden_dim=hidden_dim,
        n_encoder_layers=n_encoder_layers,
        n_group_gnn_layers=n_group_gnn_layers,
        n_gru_layers=n_gru_layers,
        n_heads=n_heads,
        dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  파라미터: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2,
    )

    # ── 학습 ──
    print(f"\n[3/4] 학습 시작 (epochs={epochs}, batch={batch_size}, lr={lr})...")
    os.makedirs(output_dir, exist_ok=True)

    history = {
        "train_loss": [], "val_loss": [],
        "val_c_index": [], "val_spearman": [],
        "best_epoch": 0, "best_c_index": 0,
    }

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        epoch_loss = 0
        epoch_survival = 0
        epoch_rank = 0
        n_batches = 0

        for batch in train_loader:
            batch["zone_seqs"] = batch["zone_seqs"].to(device)
            batch["events"] = batch["events"].to(device)
            batch["at_risks"] = batch["at_risks"].to(device)

            optimizer.zero_grad()

            loss_dict = model.compute_loss(
                batch, lambda_rank=lambda_rank, lambda_encounter=lambda_encounter
            )

            loss_dict["total"].backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            epoch_loss += loss_dict["total"].item()
            epoch_survival += loss_dict["survival"].item()
            epoch_rank += loss_dict["rank"].item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        avg_train_surv = epoch_survival / max(n_batches, 1)
        avg_train_rank = epoch_rank / max(n_batches, 1)

        # ── Validation ──
        val_losses, val_metrics = evaluate(model, val_loader, device)
        val_loss = val_losses.get("total", float("inf"))
        val_c_index = val_metrics.get("summary", {}).get("c_index", 0)
        val_spearman = val_metrics.get("summary", {}).get("spearman_rho", 0)

        scheduler.step(val_loss)

        # History
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_c_index"].append(val_c_index)
        history["val_spearman"].append(val_spearman)

        elapsed = time.time() - t0

        print(f"  Epoch {epoch:3d}/{epochs} ({elapsed:.1f}s) | "
              f"train: {avg_train_loss:.4f} (surv={avg_train_surv:.4f}, rank={avg_train_rank:.4f}) | "
              f"val: {val_loss:.4f}, C={val_c_index:.4f}, ρ={val_spearman:.3f}")

        # ── Early stopping + checkpoint ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            history["best_epoch"] = epoch
            history["best_c_index"] = val_c_index

            ckpt_path = os.path.join(output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_c_index": val_c_index,
                "config": {
                    "hidden_dim": hidden_dim,
                    "n_encoder_layers": n_encoder_layers,
                    "n_group_gnn_layers": n_group_gnn_layers,
                    "n_gru_layers": n_gru_layers,
                    "n_heads": n_heads,
                    "dropout": dropout,
                    "agent_feat_dim": dataset.PLAYER_FEAT_DIM,
                },
            }, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    # ── 최종 평가 (Test) ──
    print(f"\n[4/4] 최종 평가...")
    print(f"  Best epoch: {history['best_epoch']}, "
          f"best val C-index: {history['best_c_index']:.4f}")

    # 최적 모델 로드
    ckpt = torch.load(os.path.join(output_dir, "best_model.pt"),
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Validation 최종 평가
    val_losses, val_results = evaluate(model, val_loader, device)
    print(format_eval_results(val_results, prefix="  [Val] "))

    # Test 평가
    if test_loader:
        test_losses, test_results = evaluate(model, test_loader, device)
        print(format_eval_results(test_results, prefix="  [Test] "))
        history["test_results"] = test_results.get("summary", {})

    # History 저장
    history_path = os.path.join(output_dir, "training_history.json")
    # numpy → python 변환
    history_serializable = {}
    for k, v in history.items():
        if isinstance(v, list):
            history_serializable[k] = [float(x) if isinstance(x, (float, np.floating)) else x for x in v]
        elif isinstance(v, dict):
            history_serializable[k] = {str(kk): float(vv) if isinstance(vv, (float, np.floating)) else vv for kk, vv in v.items()}
        else:
            history_serializable[k] = v
    with open(history_path, "w") as f:
        json.dump(history_serializable, f, indent=2)
    print(f"\n  History 저장: {history_path}")

    return model, history


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Arena Survival Network 학습")

    # 데이터
    parser.add_argument("--pt_dir", type=str, default="data/graphs",
                        help=".pt 파일 디렉토리")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="체크포인트 저장 디렉토리")

    # 모델
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_encoder_layers", type=int, default=2)
    parser.add_argument("--n_group_gnn_layers", type=int, default=2)
    parser.add_argument("--n_gru_layers", type=int, default=1)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    # 학습
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lambda_rank", type=float, default=0.1)
    parser.add_argument("--lambda_encounter", type=float, default=0.05)

    # 데이터 처리
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--skip_first_steps", type=int, default=10)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--min_alive_teams", type=int, default=3)

    # 기타
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Arena Survival Network 학습")
    print("=" * 50)
    print(f"Config: {vars(args)}")
    print()

    model, history = train(**vars(args))

    if model is not None:
        print("\n학습 완료!")
    else:
        print("\n학습 실패.")
