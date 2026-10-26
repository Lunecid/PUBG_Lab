"""K3a / K3: KRAFTON-tf-ourfeat (K2) with the weighted-Cox objective replaced by a per-team BCE on the next-phase hazard.

    python -m adaptation.ourfeat_hazard --cohort erangel_tpp_squad --variant bce     --seed 42    # K3a
    python -m adaptation.ourfeat_hazard --cohort erangel_tpp_squad --variant hazard  --seed 42    # K3

Identical to K2 (``adaptation/ourfeat.py``) in everything except the objective: the same official, bug-fixed
``TransformerBackbone`` (``TokenOnlyBackbone``: key padding mask, no position / zone / map embeddings), the same
hyperparameters (embed 64, 8 heads, 8 layers, FFN 256, dropout 0.1, batch 512, AdamW lr 1e-4, wd 1e-4), the same input
(the dump's team token at the cut: X(150) + team centre(2) + ``team_ai_frac`` = 153, train-token z-score), the same
cut-only training samples (one sample per (match, cut phase) with >= 1 risk-set row, squads alive at the cut), the same
steps per epoch (the steps of a KRAFTON-tf epoch on the cohort), the same early stopping (patience 4, epoch cap
max(30, 5000 steps)) and the same initial weights (same seed, same construction order as K2).

Only the objective changes:

* K3a  ``KRAFTON-tf-ourfeat-bce``     per-team BCE on ``y_hazard_v2`` over the v2 risk-set rows (winner censoring: winners
  are at risk at every existing cut and never positive); the head is upstream's scalar ``HazardHead`` read as a logit.  No
  phase information enters the head.
* K3   ``KRAFTON-tf-ourfeat-hazard``  K3a + a learned per-cut-phase bias added to the logit: ``logit = head(h) + b[k]``,
  ``b`` = ``nn.Embedding(7, 1)`` over k = 1..7 (the discrete-time hazard with a phase-specific baseline).

Squads that are alive at the cut but outside the risk set (they die in the last 10 s before the phase end) stay in the
attention set as in K2 and carry no loss.  On a v2 dump the rows are the dump's tokens: the attention keys are the causally
alive squads (``node_mask``), and a comeback row (risk set but causally dead at the cut) is a loss row without being a key.  Early stopping is on the val NLL (pooled BCE over the val risk-set rows).
Platt scaling on val as in K1 / K2 (for a BCE logit it should be close to the identity; a and b are stored, ``raw_score``
is the raw logit, and ``uncalibrated`` in metrics.json holds the metrics of sigmoid(raw logit)).

Intercept initialisation (both arms, train labels only): the scalar bias of the head's last layer starts at the pooled
train log-odds of the hazard; in K3 the phase embedding starts at logit(rate_k) - pooled log-odds, so that the offsets of
the phases start at their train base rates.  Reason: AdamW with lr 1e-4 moves a scalar by <= ~1e-4 per step, so a
zero-initialised embedding could not reach the baseline logits (-1.3 ... +0.2) within the few thousand steps of a run,
and K3 would degenerate to K3a.  The embedding is learned (it is in the optimiser) from that start.
"""

from __future__ import annotations

# Project code is separate from the immutable vendored reference.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))
from pubg_lab.common.paths import REPO_ROOT
from projects.krafton_baseline.common.commands import source_for

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = REPO_ROOT / "third_party/krafton_pubg_winprob"
REPO = REPO_ROOT
for p in (str(ROOT), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import projects.krafton_baseline.models.ourfeat as OF  # noqa: E402  (K2: backbone, samples, loader, helpers)
import projects.krafton_baseline.training.train as T  # noqa: E402  (K1 helpers: prediction frame, Platt, files, native metrics, trainer)
from projects.krafton_baseline.data.labels import check_label_version, load_labels  # noqa: E402
import pubg_lab.survival.ptr.evaluation.metrics as M  # noqa: E402
from pubg_lab.survival.ptr.data.data import Standardizer, dump_version, load_cohort  # noqa: E402  (read-only use)
from pubg_lab.survival.ptr.common.util import config_hash, git_info, to_jsonable  # noqa: E402
from src.models.heads import get_head  # noqa: E402

logger = logging.getLogger("krafton.ourfeat_hazard")
ARMS = {"bce": "KRAFTON-tf-ourfeat-bce", "hazard": "KRAFTON-tf-ourfeat-hazard"}
N_CUTS = 7  # cut phases k = 1..7
NO_LOSS = -1.0  # label value of a token that is not in the risk set
RATE_EPS = 1e-3


# --------------------------------------------------------------------------- samples
class HazardSampleSet(OF.TokenSampleSet):
    """K2's cut samples + per-row hazard labels: ``RISK`` (row in the v2 risk set) and ``Y`` (``y_hazard_v2``).

    ``batch`` returns the K2 batch with ``y`` = the hazard label of the risk-set rows and ``-1`` elsewhere (no loss), and
    ``cut_phase`` (the cut k = 1..7 of every sample)."""

    def __init__(self, base: OF.TokenSampleSet, risk: np.ndarray, y: np.ndarray) -> None:
        self.__dict__.update(base.__dict__)  # same arrays (X57, TGT, TIDX, S_*, off ...)
        self.RISK = risk
        self.Y = y

    def batch(self, idx: np.ndarray) -> dict:
        b = super().batch(idx)
        n = self.S_n[idx]
        pad = int(n.max())
        ar = np.arange(pad)[None, :]
        valid = ar < n[:, None]
        rows = np.where(valid, self.off[idx][:, None] + ar, 0)
        lab = np.where(valid & self.RISK[rows], self.Y[rows], NO_LOSS).astype(np.float32)
        b["y"] = torch.from_numpy(lab)
        b["cut_phase"] = torch.from_numpy(self.S_cut[idx].astype(np.int64))
        return b


def attach_labels(ss: OF.TokenSampleSet, labs: dict) -> HazardSampleSet:
    """Hazard labels of the K2 rows from the dump's label tables (``select_labels`` v2: risk set, winner censoring)."""
    risk = np.zeros(ss.n_rows, dtype=bool)
    y = np.zeros(ss.n_rows, dtype=np.float32)
    for i in range(len(ss)):
        lab = labs[str(ss.match_ids[ss.S_mi[i]])]
        p = int(ss.S_cut[i]) - 1
        r0, r1 = ss.off[i], ss.off[i + 1]
        tidx = ss.TIDX[r0:r1].astype(np.int64)
        risk[r0:r1] = lab["target"][p, tidx]
        y[r0:r1] = lab["y"][p, tidx]
        if int(risk[r0:r1].sum()) != int(lab["target"][p].sum()):
            raise AssertionError(
                f"{lab['match_id']} cut {p + 1}: risk-set rows missing among the alive squads"
            )
    return HazardSampleSet(ss, risk, y)


def phase_rates(ss: HazardSampleSet) -> tuple[float, np.ndarray, np.ndarray]:
    """(pooled rate, per-phase rate [7], per-phase risk rows [7]) of the hazard over the risk-set rows."""
    ph = np.repeat(ss.S_cut.astype(np.int64), ss.S_n)
    m = ss.RISK
    pooled = float(ss.Y[m].mean())
    rates, n = np.zeros(N_CUTS), np.zeros(N_CUTS, dtype=np.int64)
    for k in range(1, N_CUTS + 1):
        sel = m & (ph == k)
        n[k - 1] = int(sel.sum())
        rates[k - 1] = float(ss.Y[sel].mean()) if n[k - 1] else pooled
    return pooled, rates, n


def logit(p: float) -> float:
    p = float(np.clip(p, RATE_EPS, 1.0 - RATE_EPS))
    return float(np.log(p / (1.0 - p)))


# --------------------------------------------------------------------------- model
class PhaseHead(nn.Module):
    """``HazardHead`` + a learned per-cut-phase bias on the logit: ``logit = base(h) + b[k]`` (k = 1..7)."""

    def __init__(self, base: nn.Module, n_phases: int = N_CUTS) -> None:
        super().__init__()
        self.base = base
        self.phase_bias = nn.Embedding(n_phases, 1)
        nn.init.zeros_(self.phase_bias.weight)

    def forward(self, x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.phase_bias(phase - 1)  # [B, N] + [B, 1]


def last_layer(head: nn.Module) -> nn.Linear:
    base = head.base if isinstance(head, PhaseHead) else head
    return base.mlp[-1]


def init_intercepts(head: nn.Module, pooled_rate: float, rates: np.ndarray | None) -> dict:
    """Head bias := pooled train log-odds; phase embedding := logit(rate_k) - pooled log-odds (K3 only)."""
    bar = logit(pooled_rate)
    with torch.no_grad():
        last_layer(head).bias.fill_(bar)
        offsets = None
        if isinstance(head, PhaseHead):
            offsets = np.array([logit(r) - bar for r in rates], dtype=np.float64)
            head.phase_bias.weight.copy_(torch.as_tensor(offsets, dtype=torch.float32)[:, None])
    return {"head_bias": bar, "phase_offsets": None if offsets is None else offsets.tolist()}


def build_models(
    n_features: int, seed: int, phased: bool, pooled_rate: float, rates: np.ndarray | None
):
    """K2's construction order (seed -> backbone -> head), so the initial weights equal K2's; then the intercepts."""
    T.set_seed(seed)
    backbone = OF.TokenOnlyBackbone(input_dim=n_features, embed_dim=T.PAPER["embed_dim"], num_heads=T.PAPER["num_heads"],
                                    num_layers=T.PAPER["num_layers"], dropout=T.PAPER["dropout"])  # fmt: skip
    head = get_head(
        T.LOSS, embed_dim=T.PAPER["embed_dim"], dropout=T.PAPER["dropout"]
    )  # upstream HazardHead: a scalar per squad
    if phased:
        head = PhaseHead(head)
    init = init_intercepts(head, pooled_rate, rates)
    return backbone, head, init


# --------------------------------------------------------------------------- trainer
def masked_bce(
    pred: torch.Tensor, y: torch.Tensor, valid: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """BCE with logits over the rows with ``valid`` and ``y >= 0`` (the risk-set rows); elementwise + mask (no dynamic shapes)."""
    m = valid & (y >= 0)
    per = F.binary_cross_entropy_with_logits(pred, y.clamp(min=0.0), reduction="none") * m.to(
        pred.dtype
    )
    total = per.sum()
    return total if reduction == "sum" else total / m.sum().clamp(min=1)


class BCETrainer(T.TimedTrainer):
    """Upstream ``Trainer`` (via the K1 timing wrapper) with the BCE objective, the cut-phase bias and a pooled val NLL."""

    def __init__(self, *args, phased: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.phased = phased

    def _forward(self, batch):
        x = batch["x_continuous"].to(self.device)
        valid = T.key_mask_of(
            batch, self.device
        )  # attention keys: the alive squads (v2: node_mask; comeback rows are queries only)
        emb = self.backbone(x, None, None, None, None, valid_mask=valid)
        if self.phased:
            return self.head(emb, batch["cut_phase"].to(self.device))
        return self.head(emb)

    def _compute_masked_loss(self, pred, y, alive_mask, padding_mask):
        return masked_bce(pred, y, alive_mask & padding_mask)

    @torch.no_grad()
    def validate(self) -> float:
        """Pooled NLL over all val risk-set rows (not a mean of batch means)."""
        self.backbone.eval()
        self.head.eval()
        total, n = 0.0, 0
        for batch in self.val_loader:
            y = batch["y"].to(self.device)
            valid = (batch["squad_alive_mask"] & batch["padding_mask"]).to(
                self.device
            )  # loss rows (y >= 0 selects the risk set)
            pred = self._forward(batch)
            total += float(masked_bce(pred, y, valid, reduction="sum"))
            n += int((valid & (y >= 0)).sum())
        return total / max(n, 1)


@torch.no_grad()
def predict_logits(
    backbone, head, ss: OF.TokenSampleSet, device: str, phased: bool, batch_size: int = 512
) -> np.ndarray:
    """Logit (higher = eliminated within the next phase) of every stored squad row, aligned with the rows of ``ss``."""
    backbone.eval()
    head.eval()
    out = np.empty(ss.n_rows, dtype=np.float64)
    for a in range(0, len(ss), batch_size):
        idx = np.arange(a, min(len(ss), a + batch_size))
        b = ss.batch(idx)
        valid = T.key_mask_of(b, device)
        emb = backbone(b["x_continuous"].to(device), None, None, None, None, valid_mask=valid)
        g = (head(emb, b["cut_phase"].to(device)) if phased else head(emb)).float().cpu().numpy()
        for j, i in enumerate(idx):
            n = int(ss.S_n[i])
            out[ss.off[i] : ss.off[i] + n] = g[j, :n]
    return out


# --------------------------------------------------------------------------- run
def code_fingerprint() -> str:
    import hashlib

    h = hashlib.sha1()
    files = sorted((ROOT / "src").rglob("*.py")) + [
        source_for(n) for n in ("ourfeat_hazard", "ourfeat", "train", "data", "labels", "schema")
    ]
    for f in files:
        h.update(f.relative_to(REPO).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def _k2_data_block(out_dir: Path, cohort: str, seed: int) -> dict | None:
    p = out_dir / cohort / "KRAFTON-tf-ourfeat" / f"seed{seed}" / "metrics.json"
    return json.loads(p.read_text())["data"] if p.is_file() else None


def run(args) -> dict:
    t_start = time.perf_counter()
    arm = ARMS[args.variant]
    phased = args.variant == "hazard"
    out = args.out_dir / args.cohort / arm / f"seed{args.seed}"
    if (out / "metrics.json").is_file() and not args.redo:
        logger.info("skip %s: metrics.json exists", out)
        return json.loads((out / "metrics.json").read_text())
    torch.set_num_threads(args.threads)
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")
    out.mkdir(parents=True, exist_ok=True)

    # ---- data: K2's cut samples (the K2 builder itself) + the hazard labels of the label tables
    t0 = time.perf_counter()
    splits = load_cohort(args.dump_dir, args.cohort, labels="v2")
    std = Standardizer.fit(splits["train"], "v2")
    t_win = OF._t_win_map(args.data_dir)
    base_sets, infos = {}, {}
    for sp in ("train", "val", "test"):
        base_sets[sp], infos[sp] = OF.build_samples(splits[sp], std, t_win, sp)
    n_features = int(base_sets["train"].X57.shape[1])
    dump_ver = dump_version(splits["train"])
    del splits
    recs = {sp: load_labels(args.data_dir, args.cohort, sp) for sp in ("train", "val", "test")}
    for (
        sp,
        rec,
    ) in recs.items():  # label tables must come from the very dump the tokens were read from
        check_label_version(
            rec, dump_ver, against=f"the {sp} dump of {args.cohort} under {args.dump_dir}"
        )
    labs = {sp: {m["match_id"]: m for m in recs[sp]["matches"]} for sp in ("train", "val", "test")}
    sets = {sp: attach_labels(base_sets[sp], labs[sp]) for sp in base_sets}
    for sp in sets:
        expect = int(sum(m["target"].sum() for m in labs[sp].values()))
        if int(sets[sp].RISK.sum()) != expect:
            raise AssertionError(
                f"{args.cohort}/{sp}: {int(sets[sp].RISK.sum())} labelled rows vs {expect} risk-set rows in the dump"
            )
        infos[sp]["n_risk_rows"] = expect
    token_check = {
        sp: OF.check_against_krafton_tf(sets[sp], args.data_dir, args.cohort, sp)
        for sp in ("val", "test")
    }
    k2 = _k2_data_block(args.out_dir, args.cohort, args.seed)
    same_as_k2 = None
    if k2 is not None:
        same_as_k2 = {
            sp: all(
                int(k2[sp][key]) == int(infos[sp][key])
                for key in ("n_matches", "n_samples", "n_rows")
            )
            for sp in ("train", "val", "test")
        }
        if not all(same_as_k2.values()):
            raise AssertionError(f"{args.cohort}: the samples differ from K2's: {same_as_k2}")
    t_data = time.perf_counter() - t0
    for sp, ss in sets.items():
        infos[sp].update(
            {
                "mean_alive_squads": float(ss.n_rows / max(len(ss), 1)),
                "max_alive_squads": int(ss.S_n.max()),
            }
        )
    logger.info(
        "data %s  token check %s  same samples as K2: %s (%.1fs)",
        infos,
        token_check,
        same_as_k2,
        t_data,
    )

    # ---- model: K2's, with the intercepts of the logit initialised at the train base rates
    pooled, rates, n_phase_rows = phase_rates(sets["train"])
    epoch_steps = OF.k1_epoch_steps(args.out_dir, args.data_dir, args.cohort, args.seed)
    backbone, head, init = build_models(
        n_features, args.seed, phased, pooled, rates if phased else None
    )
    n_params = sum(p.numel() for p in backbone.parameters()) + sum(
        p.numel() for p in head.parameters()
    )
    trainer = BCETrainer(
        backbone=backbone, head=head, train_loader=OF.StepLoader(sets["train"], T.PAPER["batch_size"], True, args.seed, epoch_steps),
        val_loader=OF.StepLoader(sets["val"], T.PAPER["batch_size"], False), loss_type=T.LOSS, lr=T.PAPER["lr"], weight_decay=T.PAPER["weight_decay"],
        device=device, checkpoint_dir=str(out), scaler_params={"x_mean": std.x_mean, "x_std": std.x_std}, phased=phased,
    )  # fmt: skip
    steps_per_epoch = len(trainer.train_loader)
    max_epochs_eff = max(args.max_epochs, int(np.ceil(args.min_steps / steps_per_epoch)))
    passes = steps_per_epoch * T.PAPER["batch_size"] / max(len(sets["train"]), 1)
    logger.info(
        "%s: train cut samples %d, epoch = %d steps (= %.1f passes), max_epochs=%d, pooled train hazard %.4f, init %s",
        arm,
        len(sets["train"]),
        steps_per_epoch,
        passes,
        max_epochs_eff,
        pooled,
        init,
    )
    t0 = time.perf_counter()
    history = trainer.train(num_epochs=max_epochs_eff, patience=T.PAPER["patience"], verbose=True)
    t_train = time.perf_counter() - t0
    ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
    backbone.load_state_dict(ckpt["backbone_state_dict"])
    head.load_state_dict(ckpt["head_state_dict"])
    if (out / "latest.pt").exists():
        (out / "latest.pt").unlink()

    # ---- evaluation at the cuts (same code as K1 / K2)
    g_val = predict_logits(backbone, head, sets["val"], device, phased)
    g_test = predict_logits(backbone, head, sets["test"], device, phased)
    df_val, df_test = (
        T.build_frame(sets["val"], g_val, labs["val"]),
        T.build_frame(sets["test"], g_test, labs["test"]),
    )
    platt = T.fit_platt(df_val["raw_score"].to_numpy(), df_val["y"].to_numpy())  # val only
    sig = lambda g: 1.0 / (1.0 + np.exp(-g))  # noqa: E731
    uncalibrated = {
        sp: M.hazard_metrics(df["y"].to_numpy(), sig(df["raw_score"].to_numpy()))
        for sp, df in (("val", df_val), ("test", df_test))
    }
    df_val, df_test = T.finalize_frame(df_val, platt), T.finalize_frame(df_test, platt)
    canon_val, canon_test = (
        T.write_pred(df_val, out / "pred_val.csv"),
        T.write_pred(df_test, out / "pred_test.csv"),
    )
    ref_dir = args.ref_dir or args.out_dir
    rows_check = {
        sp: T.assert_rows_match(c, ref_dir, args.cohort, sp)
        for sp, c in (("val", canon_val), ("test", canon_test))
    }
    final_offsets = head.phase_bias.weight.detach().cpu().numpy()[:, 0].tolist() if phased else None
    final_bias = float(last_layer(head).bias.detach().cpu())
    native = {
        sp: T.native_metrics(sets[sp], g, labs[sp]) for sp, g in (("val", g_val), ("test", g_test))
    }
    for sp in native:  # the 'unfixed direction' entry belongs to the Cox convention of K1 / K2; meaningless for a hazard logit
        native[sp]["all"].pop("winner_accuracy_unfixed_direction", None)
        for blk in native[sp]["by_cut_phase"].values():
            blk.pop("winner_accuracy_unfixed_direction", None)
    spec = {"arm": arm, "cohort": args.cohort, "seed": args.seed, "objective": "bce", "phased": phased, "max_epochs": args.max_epochs, "min_steps": args.min_steps,
            "paper": T.PAPER, "epoch_steps": epoch_steps, "device": device, "code": code_fingerprint()}  # fmt: skip
    res = {
        "arm": arm, "cohort": args.cohort, "seed": args.seed, "device": device, "labels": "v2", "encoder": "transformer", "dump_version": dump_ver,
        "config_hash": config_hash(spec),
        "baseline": {
            "source": "krafton-ai/pubg-winprob 99ed9d1 transformer backbone, adapted (third_party/krafton_pubg_winprob/README_ADAPTATION.md), on the PTR team features; objective changed vs K2",
            "objective": "per-team BCE on y_hazard_v2 over the v2 risk-set rows (winners censored), scalar logit head (upstream HazardHead)",
            "phase_conditioning": "learned per-cut-phase bias added to the logit (nn.Embedding(7, 1), k = 1..7)" if phased else "none",
            "intercept_init": init, "train_hazard_pooled": pooled, "train_hazard_by_phase": rates.tolist(), "train_risk_rows_by_phase": n_phase_rows.tolist(),
            "final_head_bias": final_bias, "final_phase_offsets": final_offsets,
            "loss": "weighted_cox in K1 / K2; here BCE", "val_loss": "val NLL = pooled BCE over the val risk-set rows (early stopping, patience 4)",
            "paper_hparams": T.PAPER, "max_epochs": args.max_epochs, "input_dim": n_features, "pos_zone_map_embeddings": "off", "n_params": int(n_params),
            "train_sampling": "phase cuts of the train matches (as K2); epoch = steps of one KRAFTON-tf epoch on this cohort",
            "p_hazard": "Platt scaling of the raw logit, fitted on val rows only (near-identity check; raw logit in raw_score)",
            "rank_score": "-raw_score (raw_score = logit: higher = eliminated earlier)", "platt": platt,
        },  # fmt: skip
        "data": infos, "n_matches": {sp: infos[sp]["n_matches"] for sp in ("train", "val", "test")}, "token_check_vs_krafton_tf": token_check,
        "same_samples_as_k2": same_as_k2, "row_assertion": rows_check, "steps_per_epoch": int(steps_per_epoch), "train_passes_per_epoch": float(passes),
        "max_epochs_effective": int(max_epochs_eff), "min_steps": int(args.min_steps), "epochs_run": int(history["stopped_epoch"]),
        "best_epoch": int(history["best_epoch"]), "best_val_loss": float(history["best_val_loss"]), "best_val_nll": float(history["best_val_loss"]),
        "stopped_early": bool(history["early_stopped"]), "train_sec": t_train, "data_sec": t_data,
        "history": [{"epoch": e + 1, "train_loss": float(tl), "val_nll": float(vl), "sec": float(sec)}
                    for e, (tl, vl, sec) in enumerate(zip(history["train_loss"], history["val_loss"], trainer.epoch_sec))],
        "git": git_info(), "val": M.all_metrics(canon_val), "test": M.all_metrics(canon_test), "uncalibrated": uncalibrated,
        "krafton_native": {**native, "note": "winner accuracy / C-index of upstream metrics.py at the phase cuts over the dump's risk-set rows (T > b_k, the rows of pred_*.csv), survival score = -logit"},
    }  # fmt: skip
    res["elapsed_sec"] = time.perf_counter() - t_start
    (out / "metrics.json").write_text(json.dumps(to_jsonable(res), indent=1))
    logger.info(
        "%s %s seed=%d  best_val_nll=%.4f  test AUC=%.4f  NLL raw %.4f / platt %.4f  platt a=%.3f b=%.3f  elapsed %.0fs",
        args.cohort,
        arm,
        args.seed,
        res["best_val_loss"],
        res["test"]["hazard"]["auc"],
        uncalibrated["test"]["nll"],
        res["test"]["hazard"]["nll"],
        platt["a"],
        platt["b"],
        res["elapsed_sec"],
    )
    return res


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--variant", required=True, choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump-dir", type=Path, default=OF.DEFAULT_DUMP)
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=OF.DEFAULT_DATA,
        help="KRAFTON-tf data dir (label tables, shards for the token check, extraction log)",
    )
    ap.add_argument("--out-dir", type=Path, default=OF.DEFAULT_OUT)
    ap.add_argument("--ref-dir", type=Path, default=None)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--min-steps", type=int, default=5000)
    ap.add_argument(
        "--threads",
        type=int,
        default=3,
        help="torch threads (resource cap: <= 3 while other jobs share the machine)",
    )
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args(argv)
    if a.threads > 3:
        raise SystemExit("resource cap: at most 3 CPU threads")
    return a


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
