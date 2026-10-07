"""Train and evaluate KRAFTON's model (adapted copy) on one BigComp 2027 cohort and write PTR-format outputs.

    python -m adaptation.train --cohort erangel_tpp_squad --encoder transformer --seed 42

* data: shards of adaptation.extract (train = KRAFTON sampling, val / test = the dump's phase cuts), standardised
  with train statistics (adaptation.data).
* model / loss: upstream TransformerBackbone (d64, 8 heads, 8 layers, FFN 256, dropout 0.1, 8 Fourier bands) +
  HazardHead, ``weighted_cox`` (their paper loss), AdamW lr 1e-4 / wd 1e-4, batch 512, early stopping on the val
  loss (patience 4 = their paper setting; max epochs 30).  ``--encoder mlp`` is their own no-attention ablation.
  Patches of the vendored code (key padding mask, no 16-squad cap, Cox sign, 57-d input) are listed in
  README_ADAPTATION.md.
* outputs (``<out-dir>/<cohort>/KRAFTON-tf|KRAFTON-mlp/seed<seed>/``): pred_val.csv, pred_test.csv in the PTR
  prediction format (+ ``raw_score``), metrics.json computed with model_stgcn.ptr.metrics, best.pt, history.json.
  ``rank_score = -raw_score`` (survival direction), ``p_hazard`` = Platt scaling of the hazard score fitted on VAL.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
for p in (str(ROOT), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from adaptation import schema as S  # noqa: E402
from adaptation.data import BatchLoader, fit_standardizer, load_split  # noqa: E402
from adaptation.labels import check_label_version, label_version, load_labels  # noqa: E402
from model_stgcn.ptr import metrics as M  # noqa: E402  (read-only use: prediction format and metrics)
from model_stgcn.ptr.util import config_hash, git_info, to_jsonable  # noqa: E402
from src.models import TransformerBackbone  # noqa: E402
from src.models.heads import get_head  # noqa: E402
from src.training.metrics import survival_score  # noqa: E402
from src.training.trainer import Trainer, set_seed  # noqa: E402

logger = logging.getLogger("krafton.train")
ARM_NAME = {"transformer": "KRAFTON-tf", "mlp": "KRAFTON-mlp"}
DEFAULT_DATA = REPO / "model_stgcn/dumps/krafton_winprob_v2"
DEFAULT_OUT = REPO / "model_stgcn/checkpoints/bigcomp2027_ptr_v2"
PAPER = dict(embed_dim=64, num_heads=8, num_layers=8, dropout=0.1, batch_size=512, lr=1e-4, weight_decay=1e-4, patience=4)
LOSS = "weighted_cox"


def key_mask_of(batch: dict, device=None) -> torch.Tensor:
    """Attention keys of a batch: ``key_mask & padding_mask`` (v2: the causally alive squads) or, for batches without a
    ``key_mask`` (v1), ``squad_alive_mask & padding_mask`` exactly as upstream.  A v2 ``squad_alive_mask`` is the Cox risk
    set (the loss rows), so it must not be used as the key mask there: a comeback squad is a loss row but never a key."""
    m = (batch["key_mask"] & batch["padding_mask"]) if "key_mask" in batch else (batch["squad_alive_mask"] & batch["padding_mask"])
    return m if device is None else m.to(device)


class TimedTrainer(Trainer):
    """Upstream Trainer + seconds per epoch (+ the v2 key mask: attention keys are the causally alive squads, the Cox loss
    rows are the risk-set squads, see :func:`key_mask_of`)."""

    epoch_sec: list

    def _forward(self, batch):
        x = batch["x_continuous"].to(self.device)
        emb = self.backbone(x, batch["positions"].to(self.device), batch["bluezone_info"].to(self.device),
                            batch["whitezone_info"].to(self.device), batch["map_idx"].to(self.device),
                            valid_mask=key_mask_of(batch, self.device))  # fmt: skip
        return self.head(emb)

    def train_epoch(self, epoch, num_epochs=None):
        t0 = time.perf_counter()
        out = super().train_epoch(epoch, num_epochs)
        if not hasattr(self, "epoch_sec"):
            self.epoch_sec = []
        self.epoch_sec.append(time.perf_counter() - t0)
        logger.info("epoch %d train_loss=%.5f (%.1fs)", epoch, out, self.epoch_sec[-1])
        return out


# --------------------------------------------------------------------------- prediction
@torch.no_grad()
def predict_scores(backbone, head, ss, device, batch_size: int = 512) -> np.ndarray:
    """Hazard score (higher = eliminated earlier) of every stored squad row, aligned with the rows of ``ss``."""
    backbone.eval()
    head.eval()
    out = np.empty(ss.n_rows, dtype=np.float64)
    for a in range(0, len(ss), batch_size):
        idx = np.arange(a, min(len(ss), a + batch_size))
        b = ss.batch(idx)
        valid = key_mask_of(b, device)
        emb = backbone(b["x_continuous"].to(device), b["positions"].to(device), b["bluezone_info"].to(device),
                       b["whitezone_info"].to(device), b["map_idx"].to(device), valid_mask=valid)  # fmt: skip
        g = head(emb).float().cpu().numpy()
        for j, i in enumerate(idx):
            n = int(ss.S_n[i])
            out[ss.off[i] : ss.off[i] + n] = g[j, :n]
    return out


def build_frame(ss, g: np.ndarray, labs: dict) -> pd.DataFrame:
    """Risk-set rows of the dump (v2) from the cut samples; ``raw_score`` = hazard score, y / ranks from the dump."""
    recs = {k: [] for k in ("match_id", "team_id", "phase", "y", "raw_score", "final_rank_norm", "top5", "match_type")}
    for i in range(len(ss)):
        k = int(ss.S_cut[i])
        assert k >= 1, "evaluation samples must be phase cuts"
        lab = labs[str(ss.match_ids[ss.S_mi[i]])]
        p = k - 1
        r0, r1 = ss.off[i], ss.off[i + 1]
        tidx = ss.TIDX[r0:r1].astype(np.int64)
        sel = np.flatnonzero(lab["target"][p, tidx])
        if sel.size != int(lab["target"][p].sum()):
            raise AssertionError(f"{lab['match_id']} cut {k}: {int(lab['target'][p].sum())} risk-set rows, {sel.size} found among the alive squads")
        t = tidx[sel]
        recs["match_id"] += [lab["match_id"]] * sel.size
        recs["team_id"] += lab["team_ids"][t].tolist()
        recs["phase"] += [k] * sel.size
        recs["y"] += lab["y"][p, t].tolist()
        recs["raw_score"] += g[r0:r1][sel].tolist()
        recs["final_rank_norm"] += lab["final_rank_norm"][t].tolist()
        recs["top5"] += lab["top5"][t].tolist()
        recs["match_type"] += [lab["match_type"]] * sel.size
    return pd.DataFrame(recs)


def fit_platt(g: np.ndarray, y: np.ndarray) -> dict:
    """Platt scaling p = sigmoid(a * g + b) of the hazard score (higher = riskier), fitted by maximum likelihood."""
    from sklearn.linear_model import LogisticRegression

    mu, sd = float(g.mean()), float(g.std()) or 1.0
    lr = LogisticRegression(C=1e8, solver="lbfgs", max_iter=5000)
    lr.fit(((g - mu) / sd)[:, None], y.astype(int))
    a_s, b_s = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    return {"a": a_s / sd, "b": b_s - a_s * mu / sd, "n": int(len(g)), "mu": mu, "sd": sd}


def apply_platt(g: np.ndarray, platt: dict) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(platt["a"] * g + platt["b"])))


def finalize_frame(df: pd.DataFrame, platt: dict) -> pd.DataFrame:
    df = df.copy()
    df["p_hazard"] = apply_platt(df["raw_score"].to_numpy(), platt)
    df["rank_score"] = -df["raw_score"]  # survival direction: higher = survives longer (PTR convention)
    return df


def write_pred(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Canonical PTR prediction file (PRED_COLUMNS, sorted) with ``raw_score`` appended; returns the canonical frame."""
    canon = M.read_predictions(df[M.PRED_COLUMNS])
    raw = df[M.KEY_COLUMNS + ["raw_score"]].copy()
    raw["match_id"] = raw["match_id"].astype(str)
    raw["team_id"] = raw["team_id"].astype(np.int64)
    raw["phase"] = raw["phase"].astype(np.int64)
    full = canon.merge(raw, on=M.KEY_COLUMNS, how="left", validate="one_to_one")
    assert full["raw_score"].notna().all() and len(full) == len(canon)
    path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(path, index=False, float_format="%.9g")
    return canon


def assert_rows_match(canon: pd.DataFrame, out_base: Path, cohort: str, split: str) -> dict:
    """Rows, labels and match types must equal those of the PTR arms (same risk set).  Raises ValueError otherwise."""
    checked = []
    for ref in ("M00", "HGB-cut", "M11R"):
        path = out_base / cohort / ref / "seed42" / f"pred_{split}.csv"
        if path.is_file():
            M.align_predictions(canon, M.read_predictions(path))  # raises on any difference
            checked.append(ref)
    if not checked:
        raise FileNotFoundError(f"no reference prediction file for {cohort}/{split}")
    return {"split": split, "n_rows": int(len(canon)), "matches_arms": checked}


# --------------------------------------------------------------------------- KRAFTON-native metrics at the cuts
def target_rows(ss, labs: dict) -> np.ndarray:
    """``[R]`` bool over the stored rows of ``ss``: the dump's risk-set rows of each sample's cut (``target`` of the label table, ``T > b_k``).

    These are the rows of the prediction files (:func:`build_frame`, same completeness check): every risk-set row of the cut must be among the
    stored squads.  They are NOT ``ss.RISK``: for K1 / K2 that is the Cox loss set ``T > a_k`` (``a_k = b_k - 10``), which also holds the squads
    that die in ``(a_k, b_k]``; only K3a / K3 have ``RISK == target``."""
    out = np.zeros(ss.n_rows, dtype=bool)
    for i in range(len(ss)):
        k = int(ss.S_cut[i])
        assert k >= 1, "evaluation samples must be phase cuts"
        lab = labs[str(ss.match_ids[ss.S_mi[i]])]
        r0, r1 = ss.off[i], ss.off[i + 1]
        sel = np.asarray(lab["target"][k - 1, ss.TIDX[r0:r1].astype(np.int64)], dtype=bool)
        n_dump = int(lab["target"][k - 1].sum())
        if n_dump < 1 or int(sel.sum()) != n_dump:
            raise AssertionError(f"{lab['match_id']} cut {k}: {n_dump} risk-set rows, {int(sel.sum())} found among the stored squads")
        out[r0:r1] = sel
    return out


def native_metrics(ss, g: np.ndarray, labs: dict) -> dict:
    """Winner accuracy and C-index of upstream (metrics.py) on the (match, cut) samples, survival score = -hazard.

    Rows scored, for every arm: the dump's risk-set rows of the cut (``labs[match]['target']``, ``T > b_k``, :func:`target_rows`) = the rows of
    pred_*.csv; ``ss.RISK`` (the Cox loss set of K1 / K2, ``T > a_k``) is not read.  Winner = the squad with the largest death time target
    (survivors censored at max(2000 s, match end)) among those rows.  ``c_index`` is the pooled pair concordance of upstream
    (ties in the score count 1/2); ``c_index_mean`` averages the per-sample values.  The ``*_unfixed_direction``
    entries read the raw hazard as 'higher = wins', which is what the upstream evaluation did for Cox losses."""
    acc, acc_raw, chance, conc, comp, per_c = [], [], [], [], [], []
    cut = []
    rows = target_rows(ss, labs)
    for i in range(len(ss)):
        r0, r1 = ss.off[i], ss.off[i + 1]
        sel = rows[r0:r1]
        t = ss.TGT[r0:r1].astype(np.float64)[sel]
        s = -g[r0:r1][sel]
        winner = int(np.argmax(t))
        acc.append(float(np.argmax(s) == winner))
        acc_raw.append(float(np.argmax(g[r0:r1][sel]) == winner))
        chance.append(1.0 / len(t))
        dt = t[:, None] - t[None, :]
        ds = s[:, None] - s[None, :]
        pair = np.triu(dt != 0, 1)
        c = float((((dt * ds) > 0) & pair).sum() + 0.5 * (((ds == 0) & pair).sum()))
        n = float(pair.sum())
        conc.append(c)
        comp.append(n)
        per_c.append(c / n if n else np.nan)
        cut.append(int(ss.S_cut[i]))
    acc, acc_raw, chance, conc, comp, per_c, cut = map(np.asarray, (acc, acc_raw, chance, conc, comp, per_c, cut))

    def block(m):
        return {
            "n_samples": int(m.sum()), "winner_accuracy": float(acc[m].mean()), "winner_accuracy_unfixed_direction": float(acc_raw[m].mean()),
            "winner_accuracy_chance": float(chance[m].mean()), "c_index": float(conc[m].sum() / comp[m].sum()) if comp[m].sum() else None,
            "c_index_mean": float(np.nanmean(per_c[m])) if np.isfinite(per_c[m]).any() else None,
        }  # fmt: skip

    out = {"all": block(np.ones(len(cut), bool)), "by_cut_phase": {int(k): block(cut == k) for k in sorted(set(cut.tolist()))}}
    return out


# --------------------------------------------------------------------------- run
def code_fingerprint() -> str:
    """SHA-1 over the vendored src/ and the adaptation modules that decide a result."""
    import hashlib

    h = hashlib.sha1()
    files = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "adaptation" / f"{n}.py" for n in ("data", "train", "features", "extract", "labels", "schema")]
    for f in files:
        h.update(f.relative_to(ROOT).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def spec_of(args) -> dict:
    return {
        "arm": ARM_NAME[args.encoder], "cohort": args.cohort, "seed": args.seed, "loss": LOSS, "max_epochs": args.max_epochs, "min_steps": args.min_steps,
        "paper": PAPER, "data": args.data_dir.name, "train_mode": "krafton192", "device": args.device, "code": code_fingerprint(),
    }  # fmt: skip


def run(args) -> dict:
    t_start = time.perf_counter()
    arm = ARM_NAME[args.encoder]
    out = args.out_dir / args.cohort / arm / f"seed{args.seed}"
    if (out / "metrics.json").is_file() and not args.redo:
        logger.info("skip %s: metrics.json exists", out)
        return json.loads((out / "metrics.json").read_text())
    torch.set_num_threads(args.threads)
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")
    out.mkdir(parents=True, exist_ok=True)

    # ---- data
    t0 = time.perf_counter()
    train = load_split(args.data_dir, args.cohort, "train")
    val = load_split(args.data_dir, args.cohort, "val")
    test = load_split(args.data_dir, args.cohort, "test")
    mean, std = fit_standardizer(train)
    for ss in (train, val, test):
        ss.finalize(mean, std)
    recs = {sp: load_labels(args.data_dir, args.cohort, sp) for sp in ("train", "val", "test")}
    for sp, ss in (("train", train), ("val", val), ("test", test)):  # shards and label tables must come from the same dump version
        check_label_version(recs[sp], ss.dump_version, against=f"the {sp} shards of {args.cohort}")
    labs = {sp: {m["match_id"]: m for m in recs[sp]["matches"]} for sp in ("val", "test")}
    t_data = time.perf_counter() - t0
    data_info = {sp: {"n_matches": int(len(set(ss.S_mi.tolist()))), "n_samples": int(len(ss)), "n_rows": int(ss.n_rows),
                      "mean_alive_squads": float(ss.n_rows / max(len(ss), 1)), "max_alive_squads": int(ss.S_n.max())}
                 for sp, ss in (("train", train), ("val", val), ("test", test))}  # fmt: skip
    logger.info("data %s  (%.1fs)", data_info, t_data)

    # ---- model (paper configuration)
    set_seed(args.seed)
    backbone = TransformerBackbone(input_dim=S.N_INPUT, embed_dim=PAPER["embed_dim"], num_heads=PAPER["num_heads"],
                                   num_layers=PAPER["num_layers"], dropout=PAPER["dropout"], encoder_type=args.encoder,
                                   num_maps=len(S.MAPS))  # fmt: skip
    head = get_head(LOSS, embed_dim=PAPER["embed_dim"], dropout=PAPER["dropout"])
    n_params = sum(p.numel() for p in backbone.parameters()) + sum(p.numel() for p in head.parameters())
    trainer = TimedTrainer(
        backbone=backbone, head=head, train_loader=BatchLoader(train, PAPER["batch_size"], True, args.seed),
        val_loader=BatchLoader(val, PAPER["batch_size"], False), loss_type=LOSS, lr=PAPER["lr"],
        weight_decay=PAPER["weight_decay"], device=device, checkpoint_dir=str(out),
        scaler_params={"mean": mean, "std": std},
    )  # fmt: skip
    steps_per_epoch = len(trainer.train_loader)
    # the epoch cap must not bind before convergence on small cohorts (21 steps / epoch on Karakin): max(30 epochs, 5000 steps)
    max_epochs_eff = max(args.max_epochs, int(np.ceil(args.min_steps / steps_per_epoch)))
    logger.info("steps/epoch=%d  max_epochs=%d (requested %d, min_steps %d)", steps_per_epoch, max_epochs_eff, args.max_epochs, args.min_steps)
    t0 = time.perf_counter()
    history = trainer.train(num_epochs=max_epochs_eff, patience=PAPER["patience"], verbose=True)
    t_train = time.perf_counter() - t0
    ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
    backbone.load_state_dict(ckpt["backbone_state_dict"])
    head.load_state_dict(ckpt["head_state_dict"])
    if (out / "latest.pt").exists():
        (out / "latest.pt").unlink()  # best.pt is enough (optimizer state of the last epoch is not needed)

    # ---- evaluation at the phase cuts
    g_val = predict_scores(backbone, head, val, device)
    g_test = predict_scores(backbone, head, test, device)
    df_val = build_frame(val, g_val, labs["val"])
    df_test = build_frame(test, g_test, labs["test"])
    platt = fit_platt(df_val["raw_score"].to_numpy(), df_val["y"].to_numpy())  # VAL only
    df_val, df_test = finalize_frame(df_val, platt), finalize_frame(df_test, platt)
    canon_val = write_pred(df_val, out / "pred_val.csv")
    canon_test = write_pred(df_test, out / "pred_test.csv")
    rows_check = {sp: assert_rows_match(c, args.ref_dir or args.out_dir, args.cohort, sp) for sp, c in (("val", canon_val), ("test", canon_test))}
    res = {
        "arm": arm, "cohort": args.cohort, "seed": args.seed, "device": device, "labels": "v2", "encoder": args.encoder,
        "dump_version": label_version(recs["train"]), "config_hash": config_hash(spec_of(args)),
        "baseline": {
            "source": "krafton-ai/pubg-winprob 99ed9d1, adapted (third_party/krafton_pubg_winprob/README_ADAPTATION.md)",
            "loss": LOSS, "paper_hparams": PAPER, "max_epochs": args.max_epochs, "train_sampling": "krafton192 (178 random + 14 anchors, points with < 2 alive squads dropped)",
            "input_dim": S.N_INPUT, "n_params": int(n_params), "p_hazard": "Platt scaling of the Cox hazard score, fitted on val rows only",
            "rank_score": "-raw_score (raw_score = hazard: higher = eliminated earlier)", "platt": platt,
            "zero_filled_features": [f for f, v in S.PARITY.items() if v[0] == "zero"],
            "proxy_features": [f for f, v in S.PARITY.items() if v[0] == "proxy"],
        },  # fmt: skip
        "data": data_info, "n_matches": {sp: data_info[sp]["n_matches"] for sp in ("train", "val", "test")}, "row_assertion": rows_check,
        "steps_per_epoch": int(steps_per_epoch), "max_epochs_effective": int(max_epochs_eff), "min_steps": int(args.min_steps),
        "epochs_run": int(history["stopped_epoch"]), "best_epoch": int(history["best_epoch"]), "best_val_loss": float(history["best_val_loss"]),
        "stopped_early": bool(history["early_stopped"]), "train_sec": t_train, "data_sec": t_data,
        "history": [{"epoch": e + 1, "train_loss": float(tl), "val_loss": float(vl), "sec": float(sec)}
                    for e, (tl, vl, sec) in enumerate(zip(history["train_loss"], history["val_loss"], trainer.epoch_sec))],
        "git": git_info(), "val": M.all_metrics(canon_val), "test": M.all_metrics(canon_test),
        "krafton_native": {"val": native_metrics(val, g_val, labs["val"]), "test": native_metrics(test, g_test, labs["test"]),
                           "note": "winner accuracy / C-index of upstream metrics.py at the phase cuts over the dump's risk-set rows (T > b_k, the rows of "
                                   "pred_*.csv), survival score = -hazard"},  # fmt: skip
    }  # fmt: skip
    res["elapsed_sec"] = time.perf_counter() - t_start
    (out / "metrics.json").write_text(json.dumps(to_jsonable(res), indent=1))
    logger.info("%s %s seed=%d  best_val_loss=%.4f  test AUC=%.4f  platt_a=%.3g  elapsed %.0fs", args.cohort, arm, args.seed,
                res["best_val_loss"], res["test"]["hazard"]["auc"], platt["a"], res["elapsed_sec"])
    return res


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--encoder", default="transformer", choices=["transformer", "mlp"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--ref-dir", type=Path, default=None, help="where the PTR arms' prediction files live (default: --out-dir)")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--min-steps", type=int, default=5000, help="the epoch cap is raised to reach this many optimizer steps (small cohorts)")
    ap.add_argument("--threads", type=int, default=4, help="torch threads (resource cap: <= 4)")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args(argv)
    if a.threads > 4:
        raise SystemExit("resource cap: at most 4 CPU threads")
    return a


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
