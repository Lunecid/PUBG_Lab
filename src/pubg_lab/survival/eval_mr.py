"""
Test-set evaluation for the MR-STGNN (hazard AUC/AP/acc), the snapshot-tier
counterpart of eval_phase. Reuses the checkpoint's own split (test_ids) and
the V2/snapshot feature dims recorded at train time, so MR vs phase-only is a
controlled comparison on the same risk cells.

    python -m model_stgcn.eval_mr --checkpoint model_stgcn/checkpoints/mr_tiger_squad/mr.pt
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[2]))

import argparse
import json
import logging

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from pubg_lab.survival.mr_dataset import CachedMRDataset, LazyMRDataset, mr_collate
from pubg_lab.survival.mr_stgnn import MRSTGNN

logger = logging.getLogger("model_stgcn.eval_mr")


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _shuffle_time(batch: dict, generator: torch.Generator) -> None:
    """In-place: permute each match's REAL snapshots along the time axis.

    Destroys snapshot temporal order (and the encoder's causal patterns) while
    keeping the exact same snapshot SET per phase — X_s/A_s/node_mask/phase_of_snap
    move together, so phase-pooling (order-invariant masked mean) is unchanged
    and ONLY the temporal convolution's view is scrambled. If the +AUC survives
    this, the gain is "just more information"; if it collapses to the phase-only
    level, snapshot temporal STRUCTURE is what matters.
    """
    nm = batch["node_mask_s"]  # (B, T_s, N)
    B = nm.shape[0]
    for b in range(B):
        real = nm[b].any(dim=-1).nonzero(as_tuple=False).flatten()
        if real.numel() <= 1:
            continue
        order = torch.randperm(real.numel(), generator=generator).to(real.device)
        perm = real[order]
        for key in ("X_s", "A_s", "node_mask_s", "phase_of_snap"):
            t = batch[key]
            t[b, real] = t[b, perm].clone()


def _build_model(ckpt: dict) -> MRSTGNN:
    a = ckpt.get("args", {})
    return MRSTGNN(
        snap_in_features=int(ckpt["snap_feature_dim"]),
        phase_in_features=int(ckpt["phase_feature_dim"]),
        snap_hidden=int(a.get("snap_hidden", 48)),
        phase_hidden=int(a.get("hidden", 64)),
        dropout=float(a.get("dropout", 0.1)),
    )


def evaluate(args: argparse.Namespace) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ids = ckpt[f"{args.split}_ids"]
    logger.info("%s ids=%d", args.split, len(ids))

    model = _build_model(ckpt)
    model.load_state_dict(ckpt["model_state"])
    device = _device(args.device)
    model.to(device).eval()

    ds_cls = CachedMRDataset if args.cached else LazyMRDataset
    loader = DataLoader(
        ds_cls(ids), batch_size=args.batch, shuffle=False, collate_fn=mr_collate, num_workers=0
    )

    gen = torch.Generator()
    if args.shuffle_time:
        gen.manual_seed(args.shuffle_seed)

    probs: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            if args.shuffle_time:
                _shuffle_time(moved, gen)
            hazard, _surv, _rank, _topk = model(
                moved["X_s"].float(),
                moved["A_s"].float(),
                moved["phase_of_snap"],
                moved["node_mask_s"].bool(),
                moved["X"].float(),
                moved["A"].float(),
                phase_mask=moved["phase_mask"],
            )
            p = torch.sigmoid(hazard).cpu().numpy()
            y = batch["y_hazard"].numpy()
            m = batch["target_mask"].numpy().astype(bool)
            probs.append(p[m])
            trues.append(y[m])

    P = np.concatenate(probs)
    Y = np.concatenate(trues).astype(int)
    pred = (P >= 0.5).astype(int)
    out = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n": int(Y.size),
        "pos_rate": float(Y.mean()),
        "acc": float(accuracy_score(Y, pred)),
        "auc": float(roc_auc_score(Y, P)) if Y.min() != Y.max() else float("nan"),
        "ap": float(average_precision_score(Y, P)) if Y.min() != Y.max() else float("nan"),
    }
    logger.info("MR eval: %s", out)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--cached", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--shuffle-time",
        action="store_true",
        help="permute snapshot time order (adversarial: temporal-structure test)",
    )
    ap.add_argument("--shuffle-seed", type=int, default=0)
    return ap.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
