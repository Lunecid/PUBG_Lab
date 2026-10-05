#!/usr/bin/env python
"""PTR-Former interpretability: EXPLORATORY and DESCRIPTIVE (no causal claim).  Read-only on code, checkpoints, dumps.

  python interp.py a1   [--cohorts c ...]                    learned relation biases b[r,h]: raw / effective, real vs shuffled
  python interp.py a1sum                                     summaries and paired tests from relation_bias_long.csv
  python interp.py a23  [--cohorts c ...] [--max-matches N]  attention mass by pair class (M11R vs M11), zone token (M11RZ),
                                                             learned team<->zone bias curves (M11RZ)
  python interp.py a23sum                                    summaries from the a23 CSVs
  (--out DIR redirects every CSV, e.g. for a smoke test)

Resource cap: 2 threads.  Runs analysed: M11R / M11RZ / M11R-shuf (every seed on disk) for analysis 1; seed 42 for 2 and 3.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "2"

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

REPO = Path("/Users/lunecid/project/PUBG")
sys.path.insert(0, str(REPO))
from model_stgcn.ptr import model as M  # noqa: E402
from model_stgcn.ptr import relations as R  # noqa: E402
from model_stgcn.ptr.data import PTRDataset, Standardizer, collate_batch, load_dump  # noqa: E402
from model_stgcn.ptr.model import PTRConfig, PTRFormer, zone_geometry  # noqa: E402

CK = REPO / "model_stgcn/checkpoints/bigcomp2027_ptr_v1"
DUMPS = REPO / "model_stgcn/dumps/bigcomp2027_phase_v1"
OUT = REPO / "reports/bigcomp2027_results/interpretability"
CH = tuple(R.RELATION_TYPES)  # prox, combat_out, combat_in, copress
A1_ARMS = ("M11R", "M11RZ", "M11R-shuf")
SEL_SEED = 20261006
KEYS = ["cohort", "seed", "block", "head"]
MEAS = {"raw": "b_raw", "eff_med": "eff_med", "eff_p90": "eff_p90", "delta_raw": "b_delta"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def write_csv(df: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False, float_format="%.6g")
    log("wrote", name, df.shape)


# ============================================================================ common helpers
def cohorts_all() -> list[str]:
    return sorted(p.name for p in CK.iterdir() if p.is_dir() and (DUMPS / p.name / "test.pt").exists())


def run_dirs(cohort: str, arm: str) -> list[Path]:
    ds = [p for p in (CK / cohort / arm).glob("seed*") if (p / "best.pt").exists()]
    return sorted(ds, key=lambda p: int(p.name[4:]))


def seed_of(p: Path) -> int:
    return int(p.name[4:])


def load_ckpt(run_dir: Path) -> dict:
    return torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)


def build_model(ck: dict) -> PTRFormer:
    m = PTRFormer(PTRConfig(**ck["config"]))
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def cohort_mode(cohort: str) -> str:
    """Resolved combat-weight mode recorded in metrics.json ('log': the dump already holds ln(1+intensity))."""
    modes = set()
    for arm in A1_ARMS:  # M11 records the unused default 'raw' (it has no relation channel), so it is not consulted
        for rd in run_dirs(cohort, arm):
            rel = json.load(open(rd / "metrics.json")).get("relations") or {}
            if rel.get("combat_weights"):
                modes.add(rel["combat_weights"])
    assert len(modes) == 1, (cohort, modes)
    return modes.pop()


def load_test(cohort: str) -> dict:
    return load_dump(DUMPS / cohort / "test.pt", "v2")


def make_ds(dump: dict, ck: dict, mode: str, shuffle_seed: int | None = None) -> PTRDataset:
    arm = ck["arm"]
    assert ck["labels"] == "v2"
    ds = PTRDataset(
        dump, Standardizer.from_state_dict(ck["standardizer"]), drop_groups=tuple(arm["drop_groups"]),
        keep_names=tuple(arm["keep_names"]), pos_in_token=True, relation_types=CH, combat_weights=mode,
        shuffle_seed=shuffle_seed, labels="v2", validate=False,
    )  # fmt: skip
    assert ds.n_features == ck["config"]["in_features"], (ds.n_features, ck["config"]["in_features"])
    return ds


def signflip_p(d) -> float:
    """Exact two-sided sign-flip permutation p-value of the mean of paired differences (n <= ~16)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan")
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(d))))
    return float((np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())


def boot_ci(d, n=10000, seed=0):
    d = np.asarray(d, float)
    if len(d) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ============================================================================ analysis 1
def edge_weight_stats(ds: PTRDataset, bs: int = 16) -> dict:
    """Non-zero edge weights per channel over the alive off-diagonal pairs of the whole test split, as the model sees them
    (after escalation, symmetrisation and the combat weight mode)."""
    vals = {t: [] for t in CH}
    n_pairs = 0
    for s in range(0, len(ds), bs):
        batch = ds.get_batch(list(range(s, min(s + bs, len(ds)))))
        alive = batch["node_mask"].bool()
        pm = alive.unsqueeze(-1) & alive.unsqueeze(-2) & ~torch.eye(alive.shape[-1], dtype=torch.bool)
        n_pairs += int(pm.sum())
        for ri, t in enumerate(CH):
            w = batch["rel"][:, :, ri]
            vals[t].append(w[pm & (w > 0)].double().numpy())
    out = {"_n_pairs": n_pairs}
    for t in CH:
        v = np.concatenate(vals[t])
        out[t] = dict(n_nonzero=int(v.size), density=v.size / max(n_pairs, 1), mean=float(v.mean()),
                      median=float(np.median(v)), p90=float(np.percentile(v, 90)), max=float(v.max()))  # fmt: skip
    return out


def a1(cohorts: list[str]) -> None:
    long_rows, w_rows = [], []
    for cohort in cohorts:
        t0 = time.time()
        dump = load_test(cohort)
        mode = cohort_mode(cohort)
        runs = {arm: run_dirs(cohort, arm) for arm in A1_ARMS}
        ck_ref = load_ckpt(runs["M11R"][0])
        ds = make_ds(dump, ck_ref, mode)
        st = edge_weight_stats(ds)
        st_sh = edge_weight_stats(make_ds(dump, ck_ref, mode, shuffle_seed=42))  # multiset check for the shuffled arm
        for ch in CH:
            a, b = st[ch], st_sh[ch]
            w_rows.append(dict(
                cohort=cohort, channel=ch, combat_mode=mode, n_test_matches=len(ds), n_alive_pairs=st["_n_pairs"],
                n_nonzero=a["n_nonzero"], density=a["density"], w_mean=a["mean"], w_median=a["median"], w_p90=a["p90"],
                w_max=a["max"], shuf_n_nonzero=b["n_nonzero"], shuf_w_median=b["median"], shuf_w_p90=b["p90"],
                shuf_max_rel_diff=max(abs(a["median"] - b["median"]) / a["median"], abs(a["p90"] - b["p90"]) / a["p90"]),
            ))  # fmt: skip
        for arm in A1_ARMS:
            for rd in runs[arm]:
                seed = seed_of(rd)
                ck = load_ckpt(rd)
                assert ck["seed"] == seed and ck["labels"] == "v2" and ck["arm"]["name"] == arm
                cfg = PTRConfig(**ck["config"])
                assert tuple(cfg.relation_types) == CH
                # exact initial b: train.fit builds PTRFormer(config, init_seed=seed), RNG scoped per component
                b_init = [blk.rel_b.detach().numpy().copy() for blk in PTRFormer(cfg, init_seed=seed).blocks]
                for l in range(cfg.n_layers):
                    b_fin = ck["state_dict"][f"blocks.{l}.rel_b"].numpy()
                    for ri, ch in enumerate(CH):
                        for h in range(cfg.n_heads):
                            b0, b1 = float(b_init[l][ri, h]), float(b_fin[ri, h])
                            wm, w9 = st[ch]["median"], st[ch]["p90"]
                            long_rows.append(dict(
                                arm=arm, cohort=cohort, seed=seed, block=l, head=h, channel=ch, b_raw=b1, b_init=b0,
                                b_delta=b1 - b0, w_med=wm, w_p90=w9, eff_med=b1 * wm, eff_p90=b1 * w9,
                                eff_med_delta=(b1 - b0) * wm, eff_p90_delta=(b1 - b0) * w9,
                            ))  # fmt: skip
        log(f"a1 {cohort}: mode={mode} runs={ {a: len(r) for a, r in runs.items()} } {time.time() - t0:.0f}s")
    write_csv(pd.DataFrame(long_rows), "relation_bias_long.csv")
    write_csv(pd.DataFrame(w_rows), "relation_bias_edge_weight_stats.csv")
    a1_summarize()


def sign_stats(x) -> dict:
    x = np.asarray(x, float)
    n = len(x)
    pos, neg = float((x > 0).mean()), float((x < 0).mean())
    return dict(n=n, mean=float(x.mean()), sd=float(x.std(ddof=1)) if n > 1 else np.nan, median=float(np.median(x)),
                frac_pos=pos, sign_consistency=max(pos, neg), majority_sign="+" if pos >= neg else "-",
                mean_abs=float(np.abs(x).mean()))  # fmt: skip


def cohort_vals(df: pd.DataFrame, arm: str, ch: str, col: str, absval: bool) -> pd.Series:
    d = df[df["arm"] == arm]
    f = (lambda s: s.abs()) if absval else (lambda s: s)
    if ch == "combat_mean":  # average of the two directed combat channels
        a = d[d["channel"] == "combat_out"].set_index(KEYS)[col]
        b = d[d["channel"] == "combat_in"].set_index(KEYS)[col]
        v = (f(a) + f(b)) / 2.0
    else:
        v = f(d[d["channel"] == ch].set_index(KEYS)[col])
    return v.groupby(level="cohort").mean()


def a1_summarize() -> None:
    df = pd.read_csv(OUT / "relation_bias_long.csv")
    rows = []
    for (arm, ch), g in df.groupby(["arm", "channel"], sort=False):
        for mname, col in MEAS.items():
            lv = {"entry": g[col].to_numpy(), "run": g.groupby(["cohort", "seed"])[col].mean().to_numpy(),
                  "cohort": g.groupby("cohort")[col].mean().to_numpy()}  # fmt: skip
            for level, x in lv.items():
                rows.append(dict(arm=arm, channel=ch, measure=mname, level=level, **sign_stats(x)))
    summ = pd.DataFrame(rows)
    summ["rank_mean_abs"] = summ.groupby(["arm", "measure", "level"])["mean_abs"].rank(ascending=False)
    write_csv(summ, "relation_bias_summary.csv")

    bh = df.groupby(["arm", "channel", "block", "head"]).agg(
        n_runs=("b_raw", "size"), mean_b_raw=("b_raw", "mean"), sd_b_raw=("b_raw", "std"),
        frac_pos_b_raw=("b_raw", lambda x: float((x > 0).mean())), mean_b_init=("b_init", "mean"),
        mean_b_delta=("b_delta", "mean"), mean_eff_med=("eff_med", "mean"), mean_eff_p90=("eff_p90", "mean"),
    ).reset_index()  # fmt: skip
    write_csv(bh, "relation_bias_by_block_head.csv")
    bc = df.groupby(["arm", "cohort", "channel"]).agg(
        n_runs=("seed", "nunique"), mean_b_raw=("b_raw", "mean"), mean_b_delta=("b_delta", "mean"), w_med=("w_med", "first"),
        w_p90=("w_p90", "first"), mean_eff_med=("eff_med", "mean"), mean_eff_p90=("eff_p90", "mean"),
        frac_pos_b=("b_raw", lambda x: float((x > 0).mean())),
    ).reset_index()  # fmt: skip
    write_csv(bc, "relation_bias_by_cohort.csv")

    tests = []
    for real in ("M11R", "M11RZ"):
        for ch in CH + ("combat_mean",):
            for mname, col in MEAS.items():
                for tr in ("signed", "abs"):
                    r = cohort_vals(df, real, ch, col, tr == "abs")
                    s = cohort_vals(df, "M11R-shuf", ch, col, tr == "abs")
                    j = pd.concat([r.rename("real"), s.rename("shuf")], axis=1).dropna()
                    d = (j["real"] - j["shuf"]).to_numpy()
                    lo, hi = boot_ci(d)
                    tests.append(dict(
                        real_arm=real, vs="M11R-shuf", channel=ch, measure=mname, transform=tr, n_cohorts=len(d),
                        mean_real=float(j["real"].mean()), mean_shuf=float(j["shuf"].mean()), mean_diff=float(d.mean()),
                        ci95_lo=lo, ci95_hi=hi, n_real_gt_shuf=int((d > 0).sum()), p_signflip=signflip_p(d),
                    ))  # fmt: skip
    write_csv(pd.DataFrame(tests), "relation_bias_real_vs_shuf.csv")


# ============================================================================ attention capture
CAP = {"on": False, "store": []}
_ORIG_MA = M.masked_attention


def _patched_ma(q, k, v, mask, bias=None, dropout_p=0.0, training=False):
    """Same arithmetic as model.masked_attention, additionally stores the attention weights while CAP['on']."""
    logits = torch.matmul(q, k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    if bias is not None:
        logits = logits + bias
    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min / 2)
    attn = torch.softmax(logits, dim=-1)
    attn = attn * mask.any(dim=-1, keepdim=True).to(attn.dtype)
    if CAP["on"]:
        CAP["store"].append(attn.detach())
    attn = F.dropout(attn, dropout_p, training)
    return torch.matmul(attn, v)


def _pre(mod, args):
    CAP["on"] = True


def _post(mod, args, out):
    CAP["on"] = False


def hook_team_attn(model: PTRFormer) -> None:
    for blk in model.blocks:  # only the team-axis attention of each block is captured, not the phase axis
        blk.attn_t.register_forward_pre_hook(_pre)
        blk.attn_t.register_forward_hook(_post)


def forward_capture(model, batch):
    CAP["store"] = []
    M.masked_attention = _patched_ma
    try:
        out = model(batch)
    finally:
        M.masked_attention = _ORIG_MA
        CAP["on"] = False
    return out, list(CAP["store"])


def forward_plain(model, batch):
    M.masked_attention = _ORIG_MA
    return model(batch)


# ============================================================================ analysis 2 and 3
POOL_CLASSES = ("combat", "prox_only", "copress_only", "prox_copress", "none", "self")
ZONE_BINS = torch.tensor([0.5, 1.0, 1.5, 2.0])
ZONE_BIN_NAMES = ["dnext[0,0.5)", "dnext[0.5,1)", "dnext[1,1.5)", "dnext[1.5,2)", "dnext[2,inf)"]
H_NAMES = ("h0", "h1", "h2", "h3")


class Acc:
    """Sums per key: [sum_lift (H), n, sum_mass (H), sum_uniform_mass]."""

    def __init__(self, H: int) -> None:
        self.H, self.d = H, {}

    def add(self, key, sum_lift, n, sum_mass, sum_unif) -> None:
        e = self.d.setdefault(key, [np.zeros(self.H), 0.0, np.zeros(self.H), 0.0])
        e[0] += sum_lift
        e[1] += n
        e[2] += sum_mass
        e[3] += sum_unif


def pair_class_masks(batch: dict, cex: torch.Tensor):
    """Mutually exclusive classes of the (query i, key j != i) pairs of alive teams, plus self.  'prox_only' / 'copress_only'
    exclude the other channel, 'prox_copress' holds pairs with both and no combat edge; combat = an edge in either direction."""
    alive = batch["node_mask"].bool()
    eye = torch.eye(alive.shape[-1], dtype=torch.bool)
    pm = alive.unsqueeze(-1) & alive.unsqueeze(-2) & ~eye
    comb = (cex | cex.transpose(-1, -2)) & pm
    prox = (batch["rel"][:, :, 0] > 0) & pm
    cop = (batch["rel"][:, :, 3] > 0) & pm
    masks = {
        "combat": comb, "prox_only": prox & ~cop & ~comb, "copress_only": cop & ~prox & ~comb,
        "prox_copress": prox & cop & ~comb, "none": pm & ~comb & ~prox & ~cop, "self": alive.unsqueeze(-1) & eye,
    }  # fmt: skip
    return masks, alive


def acc_pairs(acc: Acc, arm: str, blk: int, A: torch.Tensor, masks: dict, alive, target, K) -> None:
    """A [B,P,H,N,N] team->team attention.  lift = (attention mass on the class keys / n class keys) / (1 / K), K = alive keys."""
    Kq = K[:, :, None].clamp(min=1.0)  # [B,P,1]
    for cname, cm in masks.items():
        cmf = cm.float()
        n_c = cmf.sum(-1)  # [B,P,N]
        mass = torch.einsum("bphij,bpij->bphi", A, cmf).double()  # [B,P,H,N]
        lift_q = mass * Kq[:, :, None, :].double() / n_c.clamp(min=1.0)[:, :, None, :].double()
        unif = (n_c / Kq).double()  # uniform mass n_c / K
        for sname, sm in (("all_alive", alive), ("target_rows", target)):
            valid = (sm & alive & (n_c > 0)).double()  # [B,P,N]
            acc.add((arm, blk, sname, cname), (lift_q * valid[:, :, None, :]).sum((0, 1, 3)).numpy(), float(valid.sum()),
                    (mass * valid[:, :, None, :]).sum((0, 1, 3)).numpy(), float((unif * valid).sum()))  # fmt: skip


def zone_groups(geo, geo_ok) -> dict:
    dc, dn = geo[..., 2], geo[..., 5]
    ic, inn = dc < 1.0, dn < 1.0
    g = {
        "all": geo_ok, "next_inside": geo_ok & inn, "next_outside": geo_ok & ~inn, "cur_inside": geo_ok & ic,
        "cur_outside": geo_ok & ~ic, "cur_in_next_in": geo_ok & ic & inn, "cur_in_next_out": geo_ok & ic & ~inn,
        "cur_out_next_in": geo_ok & ~ic & inn, "cur_out_next_out": geo_ok & ~ic & ~inn,
    }  # fmt: skip
    bins = torch.bucketize(dn, ZONE_BINS, right=True)
    for i, nm in enumerate(ZONE_BIN_NAMES):
        g[nm] = geo_ok & (bins == i)
    return g


def acc_zone(acc: Acc, blk: int, attn, alive, zone_ok, geo, geo_ok, target) -> None:
    """Attention of the team queries on the zone token (key index N); lift = mass x K, K = alive teams + zone token."""
    N = alive.shape[-1]
    Az = attn[..., :N, N].double()  # [B,P,H,N]
    K = (alive.sum(-1).float() + zone_ok.float()).clamp(min=1.0)
    lift = Az * K[:, :, None, None].double()
    unif = (1.0 / K)[:, :, None].expand(alive.shape).double()
    for gname, g in zone_groups(geo, geo_ok).items():
        for sname, sm in (("all_alive", alive), ("target_rows", target)):
            valid = (g & sm).double()
            acc.add((blk, sname, gname), (lift * valid[:, :, None, :]).sum((0, 1, 3)).numpy(), float(valid.sum()),
                    (Az * valid[:, :, None, :]).sum((0, 1, 3)).numpy(), float((unif * valid).sum()))  # fmt: skip


def zone_curves(model: PTRFormer, geo_all: np.ndarray, med: np.ndarray, grid: np.ndarray, cohort: str) -> pd.DataFrame:
    """Learned team<->zone bias per block / head vs the d/r of the current or the next zone circle.
    'median': the other 5 geometry inputs at their test medians (as specified).  'pdp': partial dependence, i.e. the
    relative-position vector of the varied circle rescaled to length d for every test token, other inputs as observed."""
    G = len(grid)
    rng = np.random.default_rng(0)
    samp = torch.from_numpy(geo_all[rng.choice(len(geo_all), size=min(4000, len(geo_all)), replace=False)].astype(np.float32))
    g_t = torch.from_numpy(grid.astype(np.float32))
    frames = []
    with torch.no_grad():
        for l, blk in enumerate(model.blocks):
            f = blk.zone_bias
            for which, (ix, iy, idist) in (("cur", (0, 1, 2)), ("next", (3, 4, 5))):
                g = np.tile(med.astype(np.float32), (G, 1))
                g[:, idist] = grid
                arrs = {"median": f(torch.from_numpy(g)).numpy()}
                rel = samp[:, [ix, iy]]
                nrm = rel.norm(dim=1, keepdim=True)
                direc = torch.where(nrm > 1e-6, rel / nrm.clamp(min=1e-6), torch.tensor([[1.0, 0.0]]))
                big = samp[:, None, :].repeat(1, G, 1)
                big[:, :, [ix, iy]] = direc[:, None, :] * g_t[None, :, None]
                big[:, :, idist] = g_t[None, :]
                arrs["pdp"] = f(big).mean(0).numpy()
                for variant, arr in arrs.items():
                    for h in range(arr.shape[1]):
                        frames.append(pd.DataFrame(dict(cohort=cohort, variant=variant, which=which, block=l, head=h, d=grid, bias=arr[:, h])))
    return pd.concat(frames, ignore_index=True)


def _div(a, b):
    return np.divide(a, b, out=np.full_like(np.asarray(a, float), np.nan), where=np.asarray(b) > 0)


def rows_from_acc(acc: Acc, names: tuple, base: dict, count_name: str) -> list[dict]:
    rows = []
    for key, (sl, n, sm, su) in acc.d.items():
        for h in (*range(acc.H), "mean"):
            if h == "mean":
                ml, pl, mm = (float(np.mean(v)) if n > 0 else np.nan for v in (_div(sl, n), _div(sm, su), _div(sm, n)))
            else:
                ml, pl, mm = _div(sl[h], n), _div(sm[h], su), _div(sm[h], n)
            rows.append({**base, **dict(zip(names, key)), "head": "mean" if h == "mean" else H_NAMES[h], count_name: int(n),
                         "mean_lift": float(ml), "pooled_lift": float(pl), "mean_mass": float(mm),
                         "mean_uniform_mass": float(su / n) if n > 0 else np.nan})  # fmt: skip
    return rows


def a23(cohorts: list[str], max_matches: int) -> None:
    lift_rows, zone_rows, san_rows, curve_frames, med_rows, used_rows = [], [], [], [], [], []
    grid = np.linspace(0.0, 2.0, 81)
    torch.set_grad_enabled(False)
    for cohort in cohorts:
        t0 = time.time()
        dump = load_test(cohort)
        n_test = len(dump["matches"])
        mode = cohort_mode(cohort)
        cks = {arm: load_ckpt(CK / cohort / arm / "seed42") for arm in ("M11R", "M11", "M11RZ")}
        ref = cks["M11R"]
        for arm, ck in cks.items():  # one standardiser (train-only z-scoring) shared by the three arms
            for k, v in ref["standardizer"].items():
                assert np.allclose(v, ck["standardizer"][k], rtol=1e-9, atol=0, equal_nan=True), (cohort, arm, k)
            assert list(ref["keep_columns"]) == list(ck["keep_columns"]), (cohort, arm)
        ds = make_ds(dump, ref, mode)
        models = {arm: build_model(ck) for arm, ck in cks.items()}
        for m in models.values():
            hook_team_attn(m)
        H = models["M11R"].cfg.n_heads

        # ---- zone geometry over the whole test split (medians for analysis 3)
        geo_l = []
        for s in range(0, n_test, 16):
            b = collate_batch([ds[i] for i in range(s, min(s + 16, n_test))])
            al = b["node_mask"].bool()
            geo, gok = zone_geometry(b["pos"], b["zone"], al & b["pos_valid"].bool(), b["zone_valid"].bool())
            geo_l.append(geo[gok].numpy())
        geo_all = np.concatenate(geo_l)
        med = np.median(geo_all, axis=0)
        names = ["cur_relx", "cur_rely", "cur_dr", "next_relx", "next_rely", "next_dr"]
        q = {f"q{int(p * 100):02d}": np.quantile(geo_all[:, [2, 5]], p, axis=0) for p in (0.05, 0.25, 0.5, 0.75, 0.95)}
        med_rows.append(dict(
            cohort=cohort, n_geo_tokens=len(geo_all), **{f"median_{n}": float(v) for n, v in zip(names, med)},
            **{f"cur_dr_{k}": float(v[0]) for k, v in q.items()}, **{f"next_dr_{k}": float(v[1]) for k, v in q.items()},
            frac_inside_cur=float((geo_all[:, 2] < 1).mean()), frac_inside_next=float((geo_all[:, 5] < 1).mean()),
            frac_next_dr_clamped=float((geo_all[:, 5] >= 49.99).mean()),
        ))  # fmt: skip
        curve_frames.append(zone_curves(models["M11RZ"], geo_all, med, grid, cohort))

        # ---- forward passes with attention capture on up to max_matches test matches
        idx = np.sort(np.random.default_rng(SEL_SEED).choice(n_test, size=min(n_test, max_matches), replace=False))
        accP, accZ = Acc(H), Acc(H)
        san = {arm: dict(max_logit_diff=0.0, max_rowsum_dev=0.0, max_partition_dev=0.0) for arm in models}
        n_alive_tok = n_target_tok = 0
        for s in range(0, len(idx), 8):
            items = [ds[int(i)] for i in idx[s : s + 8]]
            P, n_max = items[0]["x"].shape[0], max(it["x"].shape[1] for it in items)
            dense = R.densify_edges_batch([it["edges"] for it in items], P, n_max)
            batch = collate_batch(items)
            batch["rel"] = R.build_relation_tensor(dense, CH, mode)
            masks, alive = pair_class_masks(batch, dense["combat_exists"])
            target = batch["target_mask"].bool() & alive
            zone_ok = batch["zone_valid"].bool()
            geo, gok = zone_geometry(batch["pos"], batch["zone"], alive & batch["pos_valid"].bool(), zone_ok)
            n_alive_tok += int(alive.sum())
            n_target_tok += int(target.sum())
            tot_n = sum(m.float().sum(-1) for m in masks.values())
            part_dev = float(((tot_n - alive.sum(-1, keepdim=True).float()).abs() * alive).max())
            N = alive.shape[-1]
            for arm, model in models.items():
                out, attns = forward_capture(model, batch)
                if s == 0:  # patched attention must reproduce the unpatched forward pass
                    ref_out = forward_plain(model, batch)
                    dif = float(((out["hazard_logit"] - ref_out["hazard_logit"]).abs() * alive).max())
                    san[arm]["max_logit_diff"] = max(san[arm]["max_logit_diff"], dif)
                san[arm]["max_partition_dev"] = max(san[arm]["max_partition_dev"], part_dev)
                use_zone = model.cfg.zone
                K = alive.sum(-1).float() + (zone_ok.float() if use_zone else 0.0)
                assert len(attns) == model.cfg.n_layers
                for l, attn in enumerate(attns):
                    rs = float(((attn.sum(-1)[..., :N] - 1).abs() * alive[:, :, None, :]).max())
                    san[arm]["max_rowsum_dev"] = max(san[arm]["max_rowsum_dev"], rs)
                    acc_pairs(accP, arm, l, attn[..., :N, :N], masks, alive, target, K)
                    if use_zone:
                        acc_zone(accZ, l, attn, alive, zone_ok, geo, gok, target)
        base = dict(cohort=cohort)
        lift_rows += rows_from_acc(accP, ("arm", "block", "subset", "class"), base, "n_query")
        zone_rows += rows_from_acc(accZ, ("block", "subset", "group"), base, "n_tokens")
        for arm, d in san.items():
            san_rows.append(dict(cohort=cohort, arm=arm, **d))
        used_rows.append(dict(cohort=cohort, n_test_matches=n_test, n_matches_used=len(idx), n_alive_tokens=n_alive_tok,
                              n_target_rows=n_target_tok, combat_mode=mode, selection_seed=SEL_SEED))  # fmt: skip
        log(f"a23 {cohort}: test={n_test} used={len(idx)} sanity={ {a: round(d['max_logit_diff'], 8) for a, d in san.items()} } {time.time() - t0:.0f}s")

    write_csv(pd.DataFrame(lift_rows), "attention_lift_by_cohort.csv")
    write_csv(pd.DataFrame(zone_rows), "attention_zone_token.csv")
    write_csv(pd.DataFrame(san_rows), "attention_sanity.csv")
    write_csv(pd.DataFrame(used_rows), "attention_matches_used.csv")
    write_csv(pd.DataFrame(med_rows), "zone_geo_medians.csv")
    write_csv(pd.concat(curve_frames, ignore_index=True), "zone_bias_curve.csv")
    a23_summarize()


def a23_summarize() -> None:
    # ---- pair-class attention lift: cohort level (head mean), then across cohorts, then M11R - M11 paired
    df = pd.read_csv(OUT / "attention_lift_by_cohort.csv")
    h = df[df["head"] == "mean"].drop(columns="head")
    both = h.groupby(["cohort", "arm", "subset", "class"], as_index=False)[
        ["n_query", "mean_lift", "pooled_lift", "mean_mass", "mean_uniform_mass"]].mean()  # fmt: skip
    both["block"] = "both"
    h = pd.concat([h.assign(block=h["block"].map(lambda b: f"L{b}")), both], ignore_index=True)
    rows = []
    for (arm, blk, sub, cl), g in h.groupby(["arm", "block", "subset", "class"], sort=False):
        for col in ("mean_lift", "pooled_lift"):
            x = g[col].dropna().to_numpy()
            if len(x) == 0:
                continue
            rows.append(dict(arm=arm, block=blk, subset=sub, **{"class": cl}, stat=col, n_cohorts=len(x), mean=x.mean(),
                             sd=x.std(ddof=1) if len(x) > 1 else np.nan, min=x.min(), max=x.max(), n_gt1=int((x > 1).sum())))  # fmt: skip
    write_csv(pd.DataFrame(rows), "attention_lift_summary.csv")
    pr = []
    for (blk, sub, cl), g in h.groupby(["block", "subset", "class"], sort=False):
        for col in ("mean_lift", "pooled_lift"):
            w = g.pivot(index="cohort", columns="arm", values=col).dropna()
            if len(w) == 0 or not {"M11R", "M11"} <= set(w.columns):
                continue
            d = (w["M11R"] - w["M11"]).to_numpy()
            lo, hi = boot_ci(d)
            pr.append(dict(block=blk, subset=sub, **{"class": cl}, stat=col, n_cohorts=len(d), mean_M11R=w["M11R"].mean(),
                           mean_M11=w["M11"].mean(), mean_diff=d.mean(), ci95_lo=lo, ci95_hi=hi, n_M11R_gt_M11=int((d > 0).sum()),
                           p_signflip=signflip_p(d)))  # fmt: skip
    write_csv(pd.DataFrame(pr), "attention_lift_M11R_vs_M11.csv")

    # ---- zone token attention: inside vs outside the next zone
    z = pd.read_csv(OUT / "attention_zone_token.csv")
    zh = z[z["head"] == "mean"].drop(columns="head")
    zb = zh.groupby(["cohort", "subset", "group"], as_index=False)[["n_tokens", "mean_lift", "pooled_lift", "mean_mass", "mean_uniform_mass"]].mean()
    zb["block"] = "both"
    zh = pd.concat([zh.assign(block=zh["block"].map(lambda b: f"L{b}")), zb], ignore_index=True)
    zr = []
    for (blk, sub, gr), g in zh.groupby(["block", "subset", "group"], sort=False):
        x = g["mean_lift"].dropna().to_numpy()
        if len(x) == 0:
            continue
        zr.append(dict(block=blk, subset=sub, group=gr, n_cohorts=len(x), mean_lift=x.mean(), sd_lift=x.std(ddof=1) if len(x) > 1 else np.nan,
                       mean_mass=g["mean_mass"].mean(), mean_uniform_mass=g["mean_uniform_mass"].mean(), total_tokens=int(g["n_tokens"].sum())))  # fmt: skip
    write_csv(pd.DataFrame(zr), "attention_zone_token_summary.csv")
    zp = []
    for (blk, sub), g in zh.groupby(["block", "subset"], sort=False):
        for a_, b_ in (("next_inside", "next_outside"), ("cur_inside", "cur_outside")):
            w = g[g["group"].isin([a_, b_])].pivot(index="cohort", columns="group", values="mean_lift").dropna()
            if len(w) == 0 or not {a_, b_} <= set(w.columns):
                continue
            d = (w[a_] - w[b_]).to_numpy()
            lo, hi = boot_ci(d)
            zp.append(dict(block=blk, subset=sub, contrast=f"{a_} - {b_}", n_cohorts=len(d), mean_a=w[a_].mean(), mean_b=w[b_].mean(),
                           mean_diff=d.mean(), ci95_lo=lo, ci95_hi=hi, n_a_gt_b=int((d > 0).sum()), p_signflip=signflip_p(d)))  # fmt: skip
    write_csv(pd.DataFrame(zp), "attention_zone_token_inside_vs_outside.csv")

    # ---- zone bias curves: mean over blocks and heads
    c = pd.read_csv(OUT / "zone_bias_curve.csv")
    gm = c.groupby(["cohort", "variant", "which", "d"]).agg(bias_mean=("bias", "mean"), bias_sd_over_block_head=("bias", "std")).reset_index()
    pb = c.groupby(["cohort", "variant", "which", "d", "block"])["bias"].mean().unstack("block")
    pb.columns = [f"bias_L{b}" for b in pb.columns]
    gm = gm.merge(pb.reset_index(), on=["cohort", "variant", "which", "d"])
    allm = gm.groupby(["variant", "which", "d"]).agg(bias_mean=("bias_mean", "mean"), bias_sd_over_cohorts=("bias_mean", "std")).reset_index()
    allm["cohort"] = "mean_over_cohorts"
    write_csv(pd.concat([gm, allm], ignore_index=True), "zone_bias_curve_mean.csv")
    sr = []
    for (co, var, wh), g in pd.concat([gm, allm.rename(columns={"bias_mean": "bias_mean"})], ignore_index=True).groupby(["cohort", "variant", "which"]):
        g = g.sort_values("d")
        at = lambda d0: float(np.interp(d0, g["d"], g["bias_mean"]))  # noqa: E731
        dd = np.diff(g["bias_mean"].to_numpy())
        sr.append(dict(cohort=co, variant=var, which=wh, bias_d0=at(0), bias_d05=at(0.5), bias_d1=at(1.0), bias_d15=at(1.5), bias_d2=at(2.0),
                       range=float(g["bias_mean"].max() - g["bias_mean"].min()), d_at_max=float(g["d"].iloc[g["bias_mean"].argmax()]),
                       d_at_min=float(g["d"].iloc[g["bias_mean"].argmin()]), delta_d1_minus_d0=at(1.0) - at(0.0),
                       delta_d2_minus_d1=at(2.0) - at(1.0), monotone_up=bool((dd >= -1e-9).all()), monotone_down=bool((dd <= 1e-9).all())))  # fmt: skip
    write_csv(pd.DataFrame(sr), "zone_bias_curve_summary.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["a1", "a1sum", "a23", "a23sum"])
    ap.add_argument("--cohorts", nargs="*", default=None)
    ap.add_argument("--max-matches", type=int, default=300)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out:
        globals()["OUT"] = Path(a.out)
    cohorts = a.cohorts or cohorts_all()
    log("cmd", a.cmd, "cohorts", cohorts, "out", OUT)
    {"a1": lambda: a1(cohorts), "a1sum": a1_summarize, "a23": lambda: a23(cohorts, a.max_matches), "a23sum": a23_summarize}[a.cmd]()


if __name__ == "__main__":
    main()
