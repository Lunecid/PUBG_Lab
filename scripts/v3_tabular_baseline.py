"""V3-comparable tabular hazard baselines (LR, HGB).

Trains on the SAME match-level split as the V3 GNN runs
(model_stgcn/checkpoints/v3_full_main_causal/split.json), over the SAME at-risk
team-phase hazard cells, using the SAME phase-level features the GNN node/global
encoder sees: concat(X_node[p,i], G[p]) under the 'full' preset.  Tabular models
get phase features only (no 10s snapshot tier, no graph) -> the intended fair
comparison.  Writes per-cell predictions and computes per-cut-phase match-level
bootstrap AUC + paired ΔAUC vs the V3 model.
"""
from __future__ import annotations
import argparse, csv, json, logging, time
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from model_stgcn.splits import load_or_create_split
from model_stgcn.v3.factorized_data import fetch_factorized_phase_match
from stdb.queries import list_phase_eligible_match_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("v3_baseline")
SPLIT = "model_stgcn/checkpoints/v3_full_main_causal/split.json"
PREDDIR = "model_stgcn/checkpoints/v3_pred"
V3_MODEL_CSV = f"{PREDDIR}/v3_full_main_causal_main_mr_gat.pred.csv"
N_BOOT = 1000


def build_xy(ids, preset="full", cap=None):
    if cap:
        ids = ids[:cap]
    X, y, meta = [], [], []
    t0 = time.time()
    for k, mid in enumerate(ids):
        m = fetch_factorized_phase_match(mid, feature_preset=preset)
        if m is None:
            continue
        T, N = m.y_hazard.shape
        for p in range(T):
            cut = m.phase_indices[p] if p < len(m.phase_indices) and m.phase_indices[p] else p + 1
            if cut <= 0:
                continue
            for i in range(N):
                if not m.target_mask[p, i]:
                    continue
                X.append(np.concatenate([m.X_node[p, i], m.G[p]]))
                y.append(int(m.y_hazard[p, i]))
                meta.append((m.match_id, int(m.team_ids[i]), int(cut)))
        if (k + 1) % 500 == 0:
            log.info("  loaded %d/%d matches (%.0fs, %d cells)", k + 1, len(ids), time.time() - t0, len(y))
    X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.asarray(y, dtype=np.int64), meta


def clusters(groups):
    uniq = np.unique(groups)
    order = np.argsort(groups, kind="stable")
    sg = groups[order]
    s = np.searchsorted(sg, uniq, "left"); e = np.searchsorted(sg, uniq, "right")
    return uniq, [order[a:b] for a, b in zip(s, e)]


def auc(y, p):
    return roc_auc_score(y, p) if y.min() != y.max() else np.nan


def boot_auc_ci(y, p, g, n=N_BOOT):
    uniq, rows = clusters(g)
    rng = np.random.default_rng(42); st = []
    for _ in range(n):
        idx = np.concatenate([rows[k] for k in rng.integers(0, len(uniq), len(uniq))])
        yy = y[idx]
        if yy.min() != yy.max():
            st.append(roc_auc_score(yy, p[idx]))
    return auc(y, p), (np.percentile(st, [2.5, 97.5]) if st else (np.nan, np.nan))


def boot_paired_delta(y, pa, pb, g, n=N_BOOT):
    """CI of AUC(a)-AUC(b) paired on resampled matches."""
    uniq, rows = clusters(g)
    rng = np.random.default_rng(42); d = []
    for _ in range(n):
        idx = np.concatenate([rows[k] for k in rng.integers(0, len(uniq), len(uniq))])
        yy = y[idx]
        if yy.min() != yy.max():
            d.append(roc_auc_score(yy, pa[idx]) - roc_auc_score(yy, pb[idx]))
    return (auc(y, pa) - auc(y, pb)), (np.percentile(d, [2.5, 97.5]) if d else (np.nan, np.nan))


def load_v3(path):
    """key (match,team,cut) -> hazard_prob, and the true label."""
    pr = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["match_id"], int(row["team_id"]), int(row["cut_phase"]))
            pr[key] = (float(row["hazard_prob"]), int(float(row["hazard_true"])))
    return pr


def write_pred(path, meta, prob, ytrue):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["match_id", "team_id", "cut_phase", "hazard_true", "hazard_prob"])
        for (mid, tid, cut), p, y in zip(meta, prob, ytrue):
            w.writerow([mid, tid, cut, y, f"{p:.6f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--preset", default="full")
    args = ap.parse_args()

    ids = list_phase_eligible_match_ids(None, "squad", None)
    tr, va, te = load_or_create_split(ids, split_json=SPLIT, seed=42, val_frac=0.1, test_frac=0.1)
    log.info("split: train=%d val=%d test=%d", len(tr), len(va), len(te))

    log.info("building TRAIN features...")
    Xtr, ytr, _ = build_xy(tr, args.preset, args.max_train)
    log.info("building TEST features...")
    Xte, yte, meta = build_xy(te, args.preset, args.max_test)
    log.info("train cells=%d (pos %.3f) | test cells=%d (pos %.3f) | dim=%d",
             len(ytr), ytr.mean(), len(yte), yte.mean(), Xtr.shape[1])

    groups = np.asarray([m[0] for m in meta])
    cutarr = np.asarray([m[2] for m in meta])

    models = {}
    sc = StandardScaler().fit(Xtr)
    log.info("training LogisticRegression...")
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(sc.transform(Xtr), ytr)
    models["lr"] = lr.predict_proba(sc.transform(Xte))[:, 1]
    log.info("training HistGradientBoosting...")
    hgb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                         max_depth=None, l2_regularization=1.0,
                                         early_stopping=True, validation_fraction=0.1,
                                         random_state=42)
    hgb.fit(Xtr, ytr)
    models["hgb"] = hgb.predict_proba(Xte)[:, 1]

    for name, prob in models.items():
        write_pred(f"{PREDDIR}/baseline_{name}.pred.csv", meta, prob, yte)
        log.info("wrote baseline_%s predictions; overall AUC=%.4f", name, auc(yte, prob))

    # V3 alignment
    v3 = load_v3(V3_MODEL_CSV)
    v3prob = np.full(len(yte), np.nan)
    nmiss = 0
    for j, (mid, tid, cut) in enumerate(meta):
        hit = v3.get((mid, tid, cut))
        if hit is None:
            nmiss += 1
        else:
            v3prob[j] = hit[0]
    log.info("V3 alignment: %d/%d cells matched (%d missing)", len(yte) - nmiss, len(yte), nmiss)
    have = ~np.isnan(v3prob)

    print("\n" + "=" * 96)
    print("V3 (mr_gat) vs TABULAR baselines: per-cut-phase hazard AUC + paired ΔAUC, match-level 95% CI")
    print("=" * 96)
    for name, prob in models.items():
        print(f"\n--- V3 mr_gat  vs  {name.upper()} ---")
        print(f"{'phase':>5}{'n':>8}{'AUC_V3':>9}{'AUC_'+name:>9}{'dAUC(V3-base)':>15}{'95% CI':>20}{'sig':>6}")
        for cut in sorted(set(cutarr.tolist())):
            mask = (cutarr == cut) & have
            if mask.sum() < 20:
                continue
            y = yte[mask]; pv = v3prob[mask]; pb = prob[mask]; g = groups[mask]
            if y.min() == y.max():
                continue
            d, (lo, hi) = boot_paired_delta(y, pv, pb, g)
            sig = "YES" if lo > 0 else ("neg" if hi < 0 else "ns")
            print(f"{cut:>5}{mask.sum():>8}{auc(y,pv):>9.4f}{auc(y,pb):>9.4f}{d:>+15.4f}  [{lo:+.4f},{hi:+.4f}]{sig:>6}")
        # overall
        m = have
        d, (lo, hi) = boot_paired_delta(yte[m], v3prob[m], prob[m], groups[m])
        sig = "YES" if lo > 0 else ("neg" if hi < 0 else "ns")
        print(f"{'ALL':>5}{m.sum():>8}{auc(yte[m],v3prob[m]):>9.4f}{auc(yte[m],prob[m]):>9.4f}{d:>+15.4f}  [{lo:+.4f},{hi:+.4f}]{sig:>6}")

    # also per-phase baseline AUC CI (is each baseline phase significant)
    print("\n" + "=" * 60 + "\nHGB baseline per-phase AUC 95% CI (sanity)\n" + "=" * 60)
    for cut in sorted(set(cutarr.tolist())):
        mask = cutarr == cut
        y = yte[mask]
        if y.min() == y.max():
            continue
        pt, (lo, hi) = boot_auc_ci(y, models["hgb"][mask], groups[mask])
        print(f"  phase {cut}: HGB AUC={pt:.4f} [{lo:.4f},{hi:.4f}]")
    print("\nDONE")


if __name__ == "__main__":
    main()
