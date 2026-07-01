"""Analyze the learned-advantage edge experiment: per-cut-phase hazard AUC per
condition + paired observed-vs-null ΔAUC with match-level cluster bootstrap.
"""
from __future__ import annotations
import csv, glob, os
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score

import sys
ROOT = sys.argv[1] if len(sys.argv) > 1 else "model_stgcn/checkpoints/adv_runs"
N_BOOT = 1000


def load(cond):
    path = f"{ROOT}/{cond}/pred_test.csv"
    if not os.path.exists(path):
        return None
    per = defaultdict(lambda: {"y": [], "p": [], "mid": []})
    allc = {"y": [], "p": [], "mid": []}
    with open(path) as f:
        for r in csv.DictReader(f):
            ph = int(r["cut_phase"]); y = int(r["hazard_true"]); p = float(r["hazard_prob"]); mid = r["match_id"]
            per[ph]["y"].append(y); per[ph]["p"].append(p); per[ph]["mid"].append(mid)
            allc["y"].append(y); allc["p"].append(p); allc["mid"].append(mid)
    for d in list(per.values()) + [allc]:
        for k in d: d[k] = np.array(d[k]) if k != "mid" else np.array(d[k])
    return per, allc


def auc(y, p):
    return roc_auc_score(y, p) if y.size and y.min() != y.max() else float("nan")


def clusters(g):
    u = np.unique(g); o = np.argsort(g, kind="stable"); sg = g[o]
    s = np.searchsorted(sg, u, "left"); e = np.searchsorted(sg, u, "right")
    return u, [o[a:b] for a, b in zip(s, e)]


def paired_delta(yA, pA, pB, g):
    u, rows = clusters(g); rng = np.random.default_rng(42); d = []
    for _ in range(N_BOOT):
        idx = np.concatenate([rows[k] for k in rng.integers(0, len(u), len(u))])
        yy = yA[idx]
        if yy.min() != yy.max():
            d.append(auc(yy, pA[idx]) - auc(yy, pB[idx]))
    return (auc(yA, pA) - auc(yA, pB)), (np.percentile(d, [2.5, 97.5]) if d else (np.nan, np.nan))


def overall_table():
    print(f"\n{'='*70}\nOverall test hazard AUC by condition\n{'='*70}")
    conds = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{ROOT}/*/pred_test.csv"))
    res = {}
    for c in conds:
        r = load(c)
        if r is None: continue
        _, allc = r
        res[c] = auc(allc["y"], allc["p"])
        print(f"  {c:24} AUC={res[c]:.4f}")
    return res


def paired(condA, condB, label):
    a = load(condA); b = load(condB)
    if a is None or b is None:
        print(f"  [{label}] missing run"); return
    perA, allA = a; perB, allB = b
    # align by (match_id order assumed identical: same test set, same order)
    print(f"\n{'='*78}\n{label}: {condA}  -  {condB}  (paired ΔAUC, match-level CI)\n{'='*78}")
    print(f"{'phase':>5}{'n':>8}{'AUC_obs':>9}{'AUC_null':>9}{'dAUC':>9}{'  95% CI':>20}{'sig':>6}")
    for ph in sorted(perA):
        if ph not in perB: continue
        dA = perA[ph]; dB = perB[ph]
        if not np.array_equal(dA["mid"], dB["mid"]):
            print(f"{ph:>5}  (order mismatch, skip)"); continue
        d, (lo, hi) = paired_delta(dA["y"], dA["p"], dB["p"], dA["mid"])
        sig = "YES" if lo > 0 else ("neg" if hi < 0 else "ns")
        print(f"{ph:>5}{dA['y'].size:>8}{auc(dA['y'],dA['p']):>9.4f}{auc(dA['y'],dB['p']):>9.4f}{d:>+9.4f}  [{lo:+.4f},{hi:+.4f}]{sig:>6}")
    d, (lo, hi) = paired_delta(allA["y"], allA["p"], allB["p"], allA["mid"])
    sig = "YES" if lo > 0 else ("neg" if hi < 0 else "ns")
    print(f"{'ALL':>5}{allA['y'].size:>8}{auc(allA['y'],allA['p']):>9.4f}{auc(allA['y'],allB['p']):>9.4f}{d:>+9.4f}  [{lo:+.4f},{hi:+.4f}]{sig:>6}")


if __name__ == "__main__":
    overall_table()
    # does the advantage topology beat its null? (the key question)
    paired("knn_observed", "knn_edge_shuffle", "kNN advantage vs edge_shuffle (topology signal?)")
    paired("knn_observed", "knn_node_shuffle", "kNN advantage vs node_shuffle (team-alignment signal?)")
    paired("strict_observed", "strict_edge_shuffle", "combat advantage vs edge_shuffle")
    paired("full_observed", "full_edge_shuffle", "full+features vs edge_shuffle")
    # does any graph beat no-graph?
    paired("knn_observed", "none_observed", "kNN advantage vs no-graph")
