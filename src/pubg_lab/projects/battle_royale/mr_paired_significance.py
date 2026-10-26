"""Match-paired significance: MR vs phase-only vs HGB on the full-squad test set.

per-match hazard AUC for each model on the SAME test matches, then one-sided
Wilcoxon (MR > baseline). Reuses the shared split's test_ids so the cells line
up; HGB predictions come from the feature_ablation run.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

OUT = "model_stgcn/checkpoints/mr_full_squad"
DEV = torch.device("cpu")


def _per_match(pred: dict[str, list[tuple[int, float]]]) -> dict[str, float]:
    out = {}
    for mid, cells in pred.items():
        y = np.array([c[0] for c in cells])
        p = np.array([c[1] for c in cells])
        if len(np.unique(y)) == 2:
            out[mid] = float(roc_auc_score(y, p))
    return out


def collect_mr(ckpt_path: str) -> dict[str, list[tuple[int, float]]]:
    from pubg_lab.survival.mr_dataset import CachedMRDataset, mr_collate
    from pubg_lab.survival.mr_stgnn import MRSTGNN

    c = torch.load(ckpt_path, map_location="cpu")
    a = c.get("args", {})
    m = MRSTGNN(
        snap_in_features=c["snap_feature_dim"],
        phase_in_features=c["phase_feature_dim"],
        snap_hidden=a.get("snap_hidden", 48),
        phase_hidden=a.get("hidden", 64),
        dropout=a.get("dropout", 0.1),
    )
    m.load_state_dict(c["model_state"])
    m.to(DEV).eval()
    loader = DataLoader(CachedMRDataset(c["test_ids"]), batch_size=8, collate_fn=mr_collate)
    d: dict[str, list] = {}
    with torch.no_grad():
        for b in loader:
            mv = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in b.items()}
            hz, _, _, _ = m(
                mv["X_s"].float(),
                mv["A_s"].float(),
                mv["phase_of_snap"],
                mv["node_mask_s"].bool(),
                mv["X"].float(),
                mv["A"].float(),
                phase_mask=mv["phase_mask"],
            )
            P = torch.sigmoid(hz).cpu().numpy()
            Y = b["y_hazard"].numpy()
            M = b["target_mask"].numpy().astype(bool)
            for bi, mid in enumerate(b["match_id"]):
                for t in range(P.shape[1]):
                    for i in range(P.shape[2]):
                        if M[bi, t, i]:
                            d.setdefault(mid, []).append((int(Y[bi, t, i]), float(P[bi, t, i])))
    return d


def collect_phase(ckpt_path: str) -> dict[str, list[tuple[int, float]]]:
    from pubg_lab.survival.dataset import CachedPhaseMatchDataset, phase_collate
    from pubg_lab.survival.dyn_stgnn import DynamicSTGNN
    from pubg_lab.survival.edge_modes import apply_edge_mode, edge_active_mask

    c = torch.load(ckpt_path, map_location="cpu")
    a = c.get("args", {})
    et = list(c.get("edge_types", ["proximity", "combat", "copressure"]))
    fd = int(c.get("feature_dim"))
    m = DynamicSTGNN(
        in_features=fd,
        hidden=int(a.get("hidden", 64)),
        mid=int(a.get("mid", 128)),
        num_blocks=int(a.get("num_blocks", 2)),
        num_edge_types=len(et),
        dropout=float(a.get("dropout", 0.1)),
    )
    m.load_state_dict(c["model_state"])
    m.to(DEV).eval()
    inc_v2 = bool(a.get("v2_node_features")) or fd > 66
    loader = DataLoader(
        CachedPhaseMatchDataset(c["test_ids"], include_v2_features=inc_v2),
        batch_size=8,
        collate_fn=phase_collate,
    )
    em = a.get("edge_mode", "full")
    am = edge_active_mask(em, edge_types=et, device=DEV)
    d: dict[str, list] = {}
    with torch.no_grad():
        for b in loader:
            mv = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in b.items()}
            A = apply_edge_mode(mv["A"].float(), em, edge_types=et)
            hz, _, _ = m(mv["X"].float(), A, edge_active_mask=am, phase_mask=mv["phase_mask"])
            P = torch.sigmoid(hz).cpu().numpy()
            Y = b["y_hazard"].numpy()
            M = b["target_mask"].numpy().astype(bool)
            for bi, mid in enumerate(b["match_id"]):
                for t in range(P.shape[1]):
                    for i in range(P.shape[2]):
                        if M[bi, t, i]:
                            d.setdefault(mid, []).append((int(Y[bi, t, i]), float(P[bi, t, i])))
    return d


def collect_hgb() -> dict[str, float]:
    metrics = pd.read_csv(f"{OUT}/hgb/feature_ablation_metrics.csv")
    best = metrics.loc[metrics["auc"].idxmax(), "model"]
    preds = pd.read_csv(f"{OUT}/hgb/feature_ablation_predictions.csv")
    preds = preds[preds["model"] == best]
    out = {}
    for mid, g in preds.groupby("match_id"):
        if g["y"].nunique() == 2:
            out[str(mid)] = float(roc_auc_score(g["y"], g["prob"]))
    print(f"HGB best config: {best}")
    return out


def paired(a: dict[str, float], b: dict[str, float], name_a: str, name_b: str) -> None:
    common = sorted(set(a) & set(b))
    diff = np.array([a[k] - b[k] for k in common])
    nz = diff[diff != 0]
    if nz.size:
        stat = wilcoxon(nz, alternative="greater")
        p = stat.pvalue
    else:
        p = 1.0
    print(
        f"  {name_a} vs {name_b}: n={len(common)} "
        f"mean({name_a})={np.mean([a[k] for k in common]):.4f} "
        f"mean({name_b})={np.mean([b[k] for k in common]):.4f} "
        f"ΔAUC={diff.mean():+.4f} wins={int((diff > 0).sum())}/{len(common)} "
        f"Wilcoxon p={p:.3e}"
    )


def main() -> int:
    print("collecting MR predictions...")
    mr = _per_match(collect_mr(f"{OUT}/mr.pt"))
    print("collecting phase-only predictions...")
    ph = _per_match(collect_phase(f"{OUT}/phase.pt"))
    print("collecting HGB predictions...")
    hgb = collect_hgb()
    print("\n=== match-paired significance (full squad test) ===")
    paired(mr, ph, "MR", "phase-only")
    paired(mr, hgb, "MR", "HGB")
    paired(ph, hgb, "phase-only", "HGB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
