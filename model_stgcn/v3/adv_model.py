"""Phase-tier edge-conditioned GNN for the learned-advantage edge experiment.

Fast by design: phase tier only, N<=32, tensors preloaded from disk cache
(scripts/adv_precompute.py), folded (B,T)->batch, dense bmm on MPS.

The edge-conditioned layer learns the "advantage" function: message weight from
node i to j = MLP(edge_feature[i,j]).  Null controls (node_shuffle / edge_shuffle)
destroy the advantage topology to test whether observed structure carries signal.
"""
from __future__ import annotations
import argparse, json, logging, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("adv_model")


def device_of(name):
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


class AdvDataset(Dataset):
    def __init__(self, path, gate):
        blob = torch.load(path, weights_only=False)
        self.items = blob["items"]
        self.gate = gate

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        Th = it["y_hazard"].shape[0]  # T-1 (cut phases)
        out = {
            "X_node": torch.from_numpy(it["X_node"][:Th]).float(),
            "G": torch.from_numpy(it["G"][:Th]).float(),
            "y": torch.from_numpy(it["y_hazard"]).float(),
            "tmask": torch.from_numpy(it["target_mask"]).bool(),
            "nmask": torch.from_numpy(it["node_mask"][:Th]).bool(),
            "phase_idx": torch.from_numpy(it["phase_indices"][:Th]).long(),
            "match_id": it["match_id"],
        }
        if self.gate != "none":
            out["A_feat"] = torch.from_numpy(it[f"A_feat_{self.gate}"][:Th].astype(np.float32))
            out["A_mask"] = torch.from_numpy(it[f"A_mask_{self.gate}"][:Th]).float()
        return out


def collate(batch):
    out = {}
    for k in batch[0]:
        if k == "match_id":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


def shuffle_adj(A_feat, A_mask, nmask, mode, gen):
    """A_feat [B,T,N,N,F], A_mask [B,T,N,N], nmask [B,T,N]. In-place-safe copy."""
    if mode == "observed":
        return A_feat, A_mask
    B, T, N, _, F = A_feat.shape
    Af = torch.zeros_like(A_feat); Am = torch.zeros_like(A_mask)
    for b in range(B):
        for t in range(T):
            alive = torch.nonzero(nmask[b, t], as_tuple=False).flatten()
            if alive.numel() <= 1:
                continue
            sub_f = A_feat[b, t][alive][:, alive]      # [a,a,F]
            sub_m = A_mask[b, t][alive][:, alive]      # [a,a]
            perm = torch.randperm(alive.numel(), generator=gen)
            if mode == "node_shuffle":         # relabel nodes (keep structure, break identity)
                sf = sub_f[perm][:, perm]; sm = sub_m[perm][:, perm]
            elif mode == "edge_shuffle":       # destroy topology: permute flat positions
                a = alive.numel()
                flatf = sub_f.reshape(a * a, F); flatm = sub_m.reshape(a * a)
                p2 = torch.randperm(a * a, generator=gen)
                sf = flatf[p2].reshape(a, a, F); sm = flatm[p2].reshape(a, a)
            else:
                raise ValueError(mode)
            Af[b, t][alive[:, None], alive[None, :]] = sf
            Am[b, t][alive[:, None], alive[None, :]] = sm
    return Af, Am


class EdgeCondLayer(nn.Module):
    def __init__(self, dim, edge_dim, ehid=16):
        super().__init__()
        self.w_self = nn.Linear(dim, dim)
        self.w_msg = nn.Linear(dim, dim)
        self.edge_mlp = nn.Sequential(nn.Linear(edge_dim, ehid), nn.ReLU(), nn.Linear(ehid, 1))

    def forward(self, h, A_feat, A_mask):
        # h [BT,N,dim]; A_feat [BT,N,N,F]; A_mask [BT,N,N]
        logit = self.edge_mlp(A_feat).squeeze(-1)            # [BT,N,N]
        logit = logit.masked_fill(A_mask < 0.5, -30.0)
        attn = torch.softmax(logit, dim=-1)
        has = (A_mask.sum(-1, keepdim=True) > 0).float()
        attn = attn * has                                    # zero rows with no edges
        msg = torch.bmm(attn, self.w_msg(h))
        return torch.relu(self.w_self(h) + msg)


class EdgeCondGNN(nn.Module):
    def __init__(self, node_dim, global_dim, edge_dim, hidden=64, layers=2, use_graph=True):
        super().__init__()
        self.use_graph = use_graph
        self.enc = nn.Linear(node_dim + global_dim, hidden)
        self.gnn = nn.ModuleList([EdgeCondLayer(hidden, edge_dim) for _ in range(layers)]) if use_graph else None
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, X_node, G, A_feat, A_mask):
        B, T, N, D = X_node.shape
        Gb = G[:, :, None, :].expand(-1, -1, N, -1)
        h = torch.relu(self.enc(torch.cat([X_node, Gb], -1)))   # [B,T,N,hid]
        if self.use_graph:
            hid = h.shape[-1]
            h = h.reshape(B * T, N, hid)
            af = A_feat.reshape(B * T, N, N, -1)
            am = A_mask.reshape(B * T, N, N)
            for layer in self.gnn:
                h = layer(h, af, am)
            h = h.reshape(B, T, N, hid)
        return self.head(h).squeeze(-1)                          # [B,T,N]


def run_epoch(model, loader, dev, edge_dim, gate, control, opt=None, gen=None):
    train = opt is not None
    model.train(train)
    tot, n = 0.0, 0
    P, Y = [], []
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for batch in loader:
        X = batch["X_node"].to(dev); G = batch["G"].to(dev)
        y = batch["y"].to(dev); tm = batch["tmask"].to(dev)
        if gate == "none":
            B, T, N, _ = X.shape
            af = torch.zeros(B, T, N, N, edge_dim, device=dev)
            am = torch.zeros(B, T, N, N, device=dev)
        else:
            af, am = batch["A_feat"], batch["A_mask"]
            if control != "observed":
                af, am = shuffle_adj(af, am, batch["nmask"], control, gen)
            af = af.to(dev); am = am.to(dev)
        with torch.set_grad_enabled(train):
            logit = model(X, G, af, am)
            loss_all = bce(logit, y)
            m = tm.float()
            loss = (loss_all * m).sum() / m.sum().clamp_min(1)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss.detach()) * int(m.sum()); n += int(m.sum())
        p = torch.sigmoid(logit).detach().cpu().numpy()[tm.cpu().numpy()]
        P.append(p); Y.append(y.cpu().numpy()[tm.cpu().numpy()])
    P = np.concatenate(P); Y = np.concatenate(Y).astype(int)
    auc = roc_auc_score(Y, P) if Y.min() != Y.max() else float("nan")
    return tot / max(n, 1), auc


def evaluate_dump(model, ds, dev, edge_dim, gate, control, gen, out_csv):
    model.eval()
    rows = []
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)
    with torch.no_grad():
        for batch in loader:
            X = batch["X_node"].to(dev); G = batch["G"].to(dev)
            tm = batch["tmask"].numpy(); y = batch["y"].numpy()
            ph = batch["phase_idx"].numpy()
            if gate == "none":
                B, T, N, _ = X.shape
                af = torch.zeros(B, T, N, N, edge_dim, device=dev); am = torch.zeros(B, T, N, N, device=dev)
            else:
                af, am = batch["A_feat"], batch["A_mask"]
                if control != "observed":
                    af, am = shuffle_adj(af, am, batch["nmask"], control, gen)
                af = af.to(dev); am = am.to(dev)
            p = torch.sigmoid(model(X, G, af, am)).cpu().numpy()
            for b in range(X.shape[0]):
                mid = batch["match_id"][b]
                for t in range(tm.shape[1]):
                    for i in range(tm.shape[2]):
                        if tm[b, t, i]:
                            rows.append((mid, int(ph[b, t]), i, int(y[b, t, i]), float(p[b, t, i])))
    import csv
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["match_id", "cut_phase", "team_idx", "hazard_true", "hazard_prob"])
        w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="model_stgcn/checkpoints/adv_cache")
    ap.add_argument("--gate", default="knn", choices=["knn", "strict", "full", "none"])
    ap.add_argument("--control", default="observed", choices=["observed", "node_shuffle", "edge_shuffle"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = device_of(args.device)
    gen = torch.Generator(); gen.manual_seed(args.seed)
    cd = Path(args.cache_dir)
    tr = AdvDataset(cd / "train.pt", args.gate)
    va = AdvDataset(cd / "val.pt", args.gate)
    te = AdvDataset(cd / "test.pt", args.gate)
    sample = tr[0]
    node_dim = sample["X_node"].shape[-1]; gdim = sample["G"].shape[-1]
    edge_dim = sample["A_feat"].shape[-1] if args.gate != "none" else 10
    log.info("gate=%s control=%s dev=%s node_dim=%d gdim=%d edge_dim=%d train=%d",
             args.gate, args.control, dev, node_dim, gdim, edge_dim, len(tr))

    model = EdgeCondGNN(node_dim, gdim, edge_dim, hidden=args.hidden, use_graph=(args.gate != "none")).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    trl = DataLoader(tr, batch_size=args.batch, shuffle=True, collate_fn=collate)
    val = DataLoader(va, batch_size=128, shuffle=False, collate_fn=collate)

    best, best_state, t0 = -1, None, time.time()
    for ep in range(args.epochs):
        loss, auc = run_epoch(model, trl, dev, edge_dim, args.gate, args.control, opt=opt, gen=gen)
        vl, vauc = run_epoch(model, val, dev, edge_dim, args.gate, "observed", gen=gen)
        log.info("ep%02d train_loss=%.4f train_auc=%.4f | val_loss=%.4f val_auc=%.4f (%.0fs)",
                 ep, loss, auc, vl, vauc, time.time() - t0)
        if vauc > best:
            best, best_state = vauc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    pred_csv = str(Path(args.out) / "pred_test.csv")
    n = evaluate_dump(model, te, dev, edge_dim, args.gate, args.control, gen, pred_csv)
    Path(args.out, "summary.json").write_text(json.dumps(
        {"gate": args.gate, "control": args.control, "best_val_auc": best,
         "n_test_cells": n, "pred_csv": pred_csv, "elapsed_s": round(time.time() - t0, 1)}, indent=2))
    log.info("DONE gate=%s control=%s best_val_auc=%.4f test_cells=%d -> %s",
             args.gate, args.control, best, n, pred_csv)


if __name__ == "__main__":
    main()
