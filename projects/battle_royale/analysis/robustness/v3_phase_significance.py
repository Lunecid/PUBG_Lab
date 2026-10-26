"""V3 per-cut-phase significance via match-level (cluster) bootstrap.

Inputs: prediction CSVs from `model_stgcn.v3.eval --pred-out`.
Answers, for the phase-wise causal forecasting interpretation:
  (A) is each cut phase's hazard AUC / rank Spearman significant (CI excludes
      chance) and how tight?
  (B) rank LOCK-IN: is the final-rank alignment significantly higher at a late
      phase than an early phase (e.g. p6 > p3)?  -> decision-window claim.
  (C) graph TOPOLOGY: is observed > edge_shuffle significant per phase (paired)?
Match-level resampling preserves within-match correlation (bootstrap_ci.py C7).
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))
import csv
import sys
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score

PRED = "model_stgcn/checkpoints/v3_pred"
N_BOOT = 1000
SEED = 42


def _rankdata(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    r = np.empty(a.size)
    i = 0
    while i < a.size:
        j = i + 1
        while j < a.size and sa[j] == sa[i]:
            j += 1
        r[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return r


def spearman(yt, yp):
    if yt.size < 2:
        return np.nan
    rt, rp = _rankdata(yt), _rankdata(yp)
    if rt.std() == 0 or rp.std() == 0:
        return np.nan
    return float(np.corrcoef(rt, rp)[0, 1])


def load_pred(path):
    """cut_phase -> dict of np arrays (mid, hz_y, hz_p, rk_y, rk_p)."""
    per = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for row in csv.DictReader(f):
            ph = int(row["cut_phase"])
            d = per[ph]
            d["mid"].append(row["match_id"])
            d["hz_y"].append(int(float(row["hazard_true"])))
            d["hz_p"].append(float(row["hazard_prob"]))
            d["rk_y"].append(float(row["final_rank_norm"]))
            d["rk_p"].append(float(row["rank_pred_norm"]))
    out = {}
    for ph, d in per.items():
        out[ph] = {k: np.asarray(v) if k != "mid" else np.asarray(v) for k, v in d.items()}
    return out


def _clusters(groups):
    uniq = np.unique(groups)
    order = np.argsort(groups, kind="stable")
    sg = groups[order]
    s = np.searchsorted(sg, uniq, "left")
    e = np.searchsorted(sg, uniq, "right")
    return uniq, [order[a:b] for a, b in zip(s, e)]


def boot_ci(fn, y, p, groups, n_boot=N_BOOT):
    uniq, rows = _clusters(groups)
    rng = np.random.default_rng(SEED)
    point = fn(y, p)
    st = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[k] for k in rng.integers(0, len(uniq), len(uniq))])
        try:
            v = fn(y[idx], p[idx])
            if v == v:
                st.append(v)
        except ValueError:
            pass
    lo, hi = np.percentile(st, [2.5, 97.5]) if st else (np.nan, np.nan)
    return point, lo, hi


def auc(y, p):
    return roc_auc_score(y, p) if y.min() != y.max() else np.nan


def report_single(name):
    path = f"{PRED}/{name}.pred.csv"
    per = load_pred(path)
    print(
        f"\n{'=' * 94}\n(A) {name}: per-cut-phase hazard AUC & rank Spearman, match-level 95% CI\n{'=' * 94}"
    )
    print(
        f"{'phase':>5}{'n_cell':>8}{'n_mtch':>7}{'haz%':>6}"
        f"{'hazAUC':>8}{'  AUC 95%CI':>18}{'rankSpr':>9}{'  Spr 95%CI':>18}"
    )
    for ph in sorted(per):
        d = per[ph]
        ap, alo, ahi = boot_ci(auc, d["hz_y"], d["hz_p"], d["mid"])
        sp, slo, shi = boot_ci(spearman, d["rk_y"], d["rk_p"], d["mid"])
        print(
            f"{ph:>5}{d['hz_y'].size:>8}{len(np.unique(d['mid'])):>7}{d['hz_y'].mean() * 100:>5.0f}%"
            f"{ap:>8.4f}  [{alo:.4f},{ahi:.4f}]{sp:>9.4f}  [{slo:.4f},{shi:.4f}]"
        )


def report_lockin(name, pairs):
    path = f"{PRED}/{name}.pred.csv"
    per = load_pred(path)
    print(
        f"\n{'=' * 94}\n(B) {name}: rank LOCK-IN — Spearman(phase_j) - Spearman(phase_i), match-level CI\n{'=' * 94}"
    )
    # joint match-level bootstrap across phases
    uniq = np.unique(np.concatenate([per[ph]["mid"] for ph in per]))
    rng = np.random.default_rng(SEED)
    # precompute per-phase cluster index
    pc = {ph: _clusters(per[ph]["mid"]) for ph in per}
    midpos = {ph: {m: k for k, m in enumerate(pc[ph][0])} for ph in per}
    print(f"{'i->j':>8}{'Spr_i':>8}{'Spr_j':>8}{'dSpr':>9}{'  95% CI of dSpr':>22}{'sig':>6}")
    for i, j in pairs:
        si = spearman(per[i]["rk_y"], per[i]["rk_p"])
        sj = spearman(per[j]["rk_y"], per[j]["rk_p"])
        deltas = []
        for _ in range(N_BOOT):
            pick = rng.integers(0, len(uniq), len(uniq))
            picks = uniq[pick]

            # gather rows for phase i and j for picked matches
            def gather(ph):
                rows = pc[ph][1]
                mp = midpos[ph]
                idxs = [rows[mp[m]] for m in picks if m in mp]
                return np.concatenate(idxs) if idxs else np.asarray([], int)

            ii, jj = gather(i), gather(j)
            if ii.size < 2 or jj.size < 2:
                continue
            d = spearman(per[j]["rk_y"][jj], per[j]["rk_p"][jj]) - spearman(
                per[i]["rk_y"][ii], per[i]["rk_p"][ii]
            )
            if d == d:
                deltas.append(d)
        lo, hi = np.percentile(deltas, [2.5, 97.5]) if deltas else (np.nan, np.nan)
        sig = "YES" if lo > 0 else ("(neg)" if hi < 0 else "ns")
        print(
            f"{f'{i}->{j}':>8}{si:>8.4f}{sj:>8.4f}{sj - si:>+9.4f}  [{lo:+.4f},{hi:+.4f}]{sig:>6}"
        )


def report_paired(name_a, name_b, label):
    pa = load_pred(f"{PRED}/{name_a}.pred.csv")
    pb = load_pred(f"{PRED}/{name_b}.pred.csv")
    print(
        f"\n{'=' * 94}\n(C) {label}: per-phase paired ΔAUC & ΔSpearman ({name_a} - {name_b})\n{'=' * 94}"
    )
    print(
        f"{'phase':>5}{'dAUC':>9}{'  AUC 95%CI':>20}{'sig':>5}{'dSpr':>10}{'  Spr 95%CI':>20}{'sig':>5}"
    )
    for ph in sorted(set(pa) & set(pb)):
        a, b = pa[ph], pb[ph]
        # observed & shuffle share the same test split + dataset -> identical
        # cell ordering, so positional alignment is valid (guarded below).
        if a["mid"].size != b["mid"].size or not np.array_equal(a["mid"], b["mid"]):
            print(f"{ph:>5}  (cell mismatch: {a['mid'].size} vs {b['mid'].size}) - skipping")
            continue
        uniq, rows = _clusters(a["mid"])
        rng = np.random.default_rng(SEED)
        dA = auc(a["hz_y"], a["hz_p"]) - auc(b["hz_y"], b["hz_p"])
        dS = spearman(a["rk_y"], a["rk_p"]) - spearman(b["rk_y"], b["rk_p"])
        bootA, bootS = [], []
        for _ in range(N_BOOT):
            idx = np.concatenate([rows[k] for k in rng.integers(0, len(uniq), len(uniq))])
            yy = a["hz_y"][idx]
            if yy.min() != yy.max():
                bootA.append(
                    auc(a["hz_y"][idx], a["hz_p"][idx]) - auc(b["hz_y"][idx], b["hz_p"][idx])
                )
            s = spearman(a["rk_y"][idx], a["rk_p"][idx]) - spearman(b["rk_y"][idx], b["rk_p"][idx])
            if s == s:
                bootS.append(s)
        aLo, aHi = np.percentile(bootA, [2.5, 97.5]) if bootA else (np.nan, np.nan)
        sLo, sHi = np.percentile(bootS, [2.5, 97.5]) if bootS else (np.nan, np.nan)
        sigA = "YES" if aLo > 0 else ("neg" if aHi < 0 else "ns")
        sigS = "YES" if sLo > 0 else ("neg" if sHi < 0 else "ns")
        print(
            f"{ph:>5}{dA:>+9.4f}  [{aLo:+.4f},{aHi:+.4f}]{sigA:>5}{dS:>+10.4f}  [{sLo:+.4f},{sHi:+.4f}]{sigS:>5}"
        )


if __name__ == "__main__":
    RANK = "mr_rgcn_attn_rank"
    report_single(RANK)
    report_single("v3_full_main_causal_main_mr_gat")
    report_lockin(RANK, [(3, 5), (3, 6), (4, 6), (1, 6), (5, 6), (6, 7)])
    # graph topology: observed vs edge_shuffle (core_habitat), and observed vs no_graph
    OBS = "v3_full_graph_relation_graph_relation_graph_core_habitat_full_observed"
    SHUF = "v3_full_graph_relation_graph_relation_graph_core_habitat_full_edge_shuffle"
    NOG = "v3_full_graph_relation_graph_relation_graph_core_habitat_no_graph"
    report_single(OBS)
    report_paired(OBS, SHUF, "observed vs edge_shuffle (graph TOPOLOGY signal)")
    report_paired(OBS, NOG, "observed vs no_graph (does a graph help at all)")
