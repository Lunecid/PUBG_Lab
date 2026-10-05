"""K2 = KRAFTON-tf-ourfeat: upstream's transformer backbone on OUR phase-dump team features (isolates input vs model).

    python -m adaptation.ourfeat --cohort erangel_tpp_squad --seed 42

* model: the bug-fixed upstream ``TransformerBackbone`` (key padding mask for padded / dead squads) + ``HazardHead``,
  same hyperparameters as KRAFTON-tf (embed 64, 8 heads, 8 layers, FFN 256, dropout 0.1, batch 512, AdamW lr 1e-4,
  wd 1e-4, patience 4) and the same ``weighted_cox`` loss on ``team_death_s`` (winners censored).  The squad token is
  the dump's team token at the cut: X (150) + team centre (2) + ``team_ai_frac`` (1) = 153, z-scored with train
  statistics (``model_stgcn.ptr.data.Standardizer`` / ``PTRDataset``, read-only use).  Upstream's 55 features, the
  per-player Fourier xyz, the zone embeddings and the map embedding are not used (``TokenOnlyBackbone`` removes
  them): our features already carry the zone information.
* data: phase cuts only (train, val and test): one sample = (match, cut phase k) with every squad alive at the cut
  t = ``phase_end_s[k-1] - 10 s`` (``team_death_s > t``, winners censored): the squad set and the target of KRAFTON-tf
  at the same cut; the features are the dump's tokens of slot k-1.  192-point sampling is not used (our features exist
  only at the cuts).
* "epoch": a pass over ~6 cut samples per match is only 30-40 optimizer steps, so patience 4 epochs would stop the
  training after ~150 steps.  An epoch here has the SAME number of optimizer steps as an epoch of KRAFTON-tf on the
  same cohort (the train cut samples are re-shuffled and repeated to fill it), so patience / epoch cap act on the
  same step scale in both models.
* outputs: same files and format as KRAFTON-tf, in ``<out>/<cohort>/KRAFTON-tf-ourfeat/seed<seed>/``.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
for p in (str(ROOT), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from adaptation import schema as S  # noqa: E402
from adaptation import train as T  # noqa: E402  (K1 helpers: prediction frame, Platt, files, native metrics, trainer)
from adaptation.data import load_split  # noqa: E402
from adaptation.labels import load_labels  # noqa: E402
from model_stgcn.ptr import metrics as M  # noqa: E402
from model_stgcn.ptr.data import PTRDataset, Standardizer, load_cohort  # noqa: E402  (read-only use)
from model_stgcn.ptr.util import config_hash, git_info, to_jsonable  # noqa: E402
from src.models import TransformerBackbone  # noqa: E402
from src.models.heads import get_head  # noqa: E402

logger = logging.getLogger("krafton.ourfeat")
ARM = "KRAFTON-tf-ourfeat"
DEFAULT_DUMP = REPO / "model_stgcn/dumps/bigcomp2027_phase_v1"
DEFAULT_DATA = REPO / "model_stgcn/dumps/krafton_winprob_v1"
DEFAULT_OUT = REPO / "model_stgcn/checkpoints/bigcomp2027_ptr_v1"


# --------------------------------------------------------------------------- model
class TokenOnlyBackbone(TransformerBackbone):
    """Upstream's backbone with the position / zone / map embeddings removed: token MLP -> encoder -> LayerNorm.

    Same modules (``TokenEmbedding``, ``nn.TransformerEncoder`` with 8 layers, final ``LayerNorm``) and the same key
    padding mask as the patched upstream class; the extra arguments of ``forward`` are accepted and ignored so that
    the upstream ``Trainer`` and the evaluation helpers of ``adaptation.train`` work unchanged."""

    def __init__(self, input_dim, embed_dim=64, num_heads=8, num_layers=8, dropout=0.1, encoder_type="transformer"):
        super().__init__(input_dim=input_dim, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout,
                         encoder_type=encoder_type, num_maps=1)  # fmt: skip
        for name in ("position_encoding", "bluezone_embedding", "whitezone_embedding", "map_embedding"):
            delattr(self, name)

    def forward(self, x, positions=None, bluezone_info=None, whitezone_info=None, map_idx=None, valid_mask=None):
        tokens = self.token_embedding(x)
        if self.encoder_type == "transformer" and valid_mask is not None:
            encoded = self.transformer(tokens, src_key_padding_mask=~valid_mask.to(torch.bool))
        else:
            encoded = self.transformer(tokens)
        return self.norm(encoded)


# --------------------------------------------------------------------------- samples
class TokenSampleSet:
    """Cut samples with the PTR team tokens (already standardised); same attribute names as adaptation.data.SampleSet."""

    def __init__(self, arrays: dict, match_ids: np.ndarray) -> None:
        self.X57 = arrays["X"]  # [R, 153] float32 (name kept for the shared helpers)
        self.TGT = arrays["TGT"]
        self.TIDX = arrays["TIDX"]
        self.S_n = arrays["S_n"].astype(np.int64)
        self.S_cut = arrays["S_cut"]
        self.S_mi = arrays["S_mi"]
        self.S_t = arrays["S_t"]
        self.match_ids = match_ids
        self.off = np.concatenate([[0], np.cumsum(self.S_n)])
        self.n_samples = len(self.S_n)
        self.n_rows = int(self.off[-1])

    def __len__(self) -> int:
        return self.n_samples

    def batch(self, idx: np.ndarray) -> dict:
        n = self.S_n[idx]
        pad = int(n.max())
        B = len(idx)
        ar = np.arange(pad)[None, :]
        valid = ar < n[:, None]
        rows = np.where(valid, self.off[idx][:, None] + ar, 0)
        x = np.where(valid[:, :, None], self.X57[rows], 0.0).astype(np.float32)
        y = np.where(valid, self.TGT[rows] / S.TARGET_SCALE, 0.0).astype(np.float32)
        return {
            "x_continuous": torch.from_numpy(x),
            "positions": torch.zeros(B, pad, 4, 3),  # unused by TokenOnlyBackbone (kept for the shared Trainer interface)
            "bluezone_info": torch.zeros(B, 3), "whitezone_info": torch.zeros(B, 3),
            "y": torch.from_numpy(y),
            "squad_alive_mask": torch.from_numpy(valid.copy()), "padding_mask": torch.from_numpy(valid.copy()),
            "num_squads": torch.from_numpy(n), "time_point": torch.from_numpy(self.S_t[idx].astype(np.float32)),
            "map_idx": torch.zeros(B, dtype=torch.long),
        }  # fmt: skip


class StepLoader:
    """Epoch = ``epoch_steps`` batches; the (shuffled) samples are repeated to fill it; validation: one ordered pass."""

    def __init__(self, data: TokenSampleSet, batch_size: int, shuffle: bool, seed: int = 0, epoch_steps: int | None = None) -> None:
        self.data, self.batch_size, self.shuffle, self.seed = data, batch_size, shuffle, seed
        self.epoch = 0
        n_pass = (len(data) + batch_size - 1) // batch_size
        self.steps = int(epoch_steps) if (shuffle and epoch_steps) else n_pass
        self.dataset = data

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        n = len(self.data)
        if not self.shuffle:
            order = np.arange(n)
        else:
            need = self.steps * self.batch_size
            reps = int(np.ceil(need / n))
            rng = np.random.RandomState(self.seed * 1000003 + self.epoch)
            order = np.concatenate([rng.permutation(n) for _ in range(reps)])[:need]
            self.epoch += 1
        for i in range(0, len(order), self.batch_size):
            yield self.data.batch(order[i : i + self.batch_size])


def _t_win_map(data_dir: Path) -> dict[str, float]:
    """match_id -> censoring time of KRAFTON-tf (max(2000 s, match end, latest death + 1 s)), from the extraction QA log."""
    files = sorted(glob.glob(str(data_dir / "logs" / "extract_*.jsonl")), key=lambda f: Path(f).stat().st_mtime)
    out: dict[str, float] = {}
    for f in files:  # later files override earlier ones
        for line in open(f):
            q = json.loads(line)
            if q.get("ok") and "t_win" in q:
                out[q["match_id"]] = float(q["t_win"])
    return out


def build_samples(dump: dict, std: Standardizer, t_win: dict[str, float], split: str) -> tuple[TokenSampleSet, dict]:
    ds = PTRDataset(dump, std, relation_types=None, labels="v2", validate=False)
    xs, tg, ti, sn, sc, smi, st, mids = [], [], [], [], [], [], [], []
    n_fallback = n_outside = 0
    for i, m in enumerate(dump["matches"]):
        mid = str(m["match_id"])
        it = ds[i]
        x = it["x"].numpy()
        target = it["target_mask"].numpy()
        node = it["node_mask"].numpy()
        death = np.asarray(m["team_death_s"], dtype=np.float64)
        won = np.asarray(m["team_won"]).astype(bool)
        death_eff = np.where(won, np.inf, death)  # winner censoring (labels v2)
        pe = np.asarray(m["phase_end_s"], dtype=np.float64)
        fin = death_eff[np.isfinite(death_eff)]
        if mid in t_win:
            tw = t_win[mid]
        else:
            n_fallback += 1
            tw = max(2000.0, float(fin.max()) + 1.0 if fin.size else 0.0)
        tgt_all = np.where(np.isinf(death_eff), tw, death_eff).astype(np.float32)
        mids.append(mid)
        for p in range(x.shape[0]):
            if not target[p].any():
                continue
            t_cut = pe[p] - 10.0
            alive_cut = death_eff > t_cut  # NaN -> False
            tok = alive_cut & node[p]
            n_outside += int((alive_cut & ~node[p]).sum())
            if int(tok.sum()) < 2 or bool((target[p] & ~tok).any()):
                raise ValueError(f"{mid} cut {p + 1}: token set {int(tok.sum())} does not cover the risk set")
            idx = np.flatnonzero(tok)
            xs.append(x[p][idx])
            tg.append(tgt_all[idx])
            ti.append(idx.astype(np.int16))
            sn.append(len(idx))
            sc.append(p + 1)
            smi.append(len(mids) - 1)
            st.append(t_cut)
    arrays = {"X": np.concatenate(xs).astype(np.float32), "TGT": np.concatenate(tg), "TIDX": np.concatenate(ti),
              "S_n": np.asarray(sn, dtype=np.int16), "S_cut": np.asarray(sc, dtype=np.int8), "S_mi": np.asarray(smi, dtype=np.int32),
              "S_t": np.asarray(st, dtype=np.float32)}  # fmt: skip
    info = {"split": split, "n_matches": len(mids), "n_samples": len(sn), "n_rows": int(sum(sn)), "t_win_fallback": n_fallback,
            "alive_at_cut_but_not_at_phase_start": n_outside}  # fmt: skip
    return TokenSampleSet(arrays, np.asarray(mids)), info


def check_against_krafton_tf(ss: TokenSampleSet, data_dir: Path, cohort: str, split: str) -> dict:
    """Same (match, cut) samples and the same alive squads as the KRAFTON-tf shards (val / test)."""
    ref = load_split(data_dir, cohort, split)
    key = lambda s, i: (str(s.match_ids[s.S_mi[i]]), int(s.S_cut[i]))  # noqa: E731
    mine = {key(ss, i): tuple(ss.TIDX[ss.off[i] : ss.off[i + 1]].tolist()) for i in range(len(ss))}
    theirs = {key(ref, i): tuple(ref.TIDX[ref.off[i] : ref.off[i + 1]].tolist()) for i in range(len(ref))}
    if set(mine) != set(theirs):
        raise AssertionError(f"{cohort}/{split}: the cut samples differ from KRAFTON-tf ({len(mine)} vs {len(theirs)})")
    same = sum(mine[k] == theirs[k] for k in mine)
    return {"split": split, "n_samples": len(mine), "same_alive_set": int(same)}


def k1_epoch_steps(out_dir: Path, data_dir: Path, cohort: str, seed: int) -> int:
    """Optimizer steps of one KRAFTON-tf epoch on this cohort (metrics.json of the run; else from the extraction QA log)."""
    p = out_dir / cohort / "KRAFTON-tf" / f"seed{seed}" / "metrics.json"
    if p.is_file():
        return int(json.loads(p.read_text())["steps_per_epoch"])
    files = sorted(glob.glob(str(data_dir / "logs" / "extract_*.jsonl")), key=lambda f: Path(f).stat().st_mtime)
    n = 0
    seen = set()
    for f in files:
        for line in open(f):
            q = json.loads(line)
            if q.get("ok") and q.get("cohort") == cohort and q.get("split") == "train" and q["match_id"] not in seen:
                seen.add(q["match_id"])
                n += int(q["n_samples"])
    return int(np.ceil(n / T.PAPER["batch_size"]))


# --------------------------------------------------------------------------- run
def code_fingerprint() -> str:
    import hashlib

    h = hashlib.sha1()
    files = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "adaptation" / f"{n}.py" for n in ("ourfeat", "train", "data", "labels", "schema")]
    for f in files:
        h.update(f.relative_to(ROOT).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def run(args) -> dict:
    t_start = time.perf_counter()
    out = args.out_dir / args.cohort / ARM / f"seed{args.seed}"
    if (out / "metrics.json").is_file() and not args.redo:
        logger.info("skip %s: metrics.json exists", out)
        return json.loads((out / "metrics.json").read_text())
    torch.set_num_threads(args.threads)
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")
    out.mkdir(parents=True, exist_ok=True)

    # ---- data: the dump's team tokens at the cuts
    t0 = time.perf_counter()
    splits = load_cohort(args.dump_dir, args.cohort, labels="v2")
    std = Standardizer.fit(splits["train"], "v2")  # train alive tokens
    t_win = _t_win_map(args.data_dir)
    sets, infos = {}, {}
    for sp in ("train", "val", "test"):
        sets[sp], infos[sp] = build_samples(splits[sp], std, t_win, sp)
    n_features = int(sets["train"].X57.shape[1])
    del splits
    labs = {sp: {m["match_id"]: m for m in load_labels(args.data_dir, args.cohort, sp)["matches"]} for sp in ("val", "test")}
    token_check = {sp: check_against_krafton_tf(sets[sp], args.data_dir, args.cohort, sp) for sp in ("val", "test")}
    t_data = time.perf_counter() - t0
    for sp, ss in sets.items():
        infos[sp].update({"mean_alive_squads": float(ss.n_rows / max(len(ss), 1)), "max_alive_squads": int(ss.S_n.max())})
    logger.info("data %s  token check %s (%.1fs)", infos, token_check, t_data)

    # ---- model: upstream backbone, tokens only
    epoch_steps = k1_epoch_steps(args.out_dir, args.data_dir, args.cohort, args.seed)
    T.set_seed(args.seed)
    backbone = TokenOnlyBackbone(input_dim=n_features, embed_dim=T.PAPER["embed_dim"], num_heads=T.PAPER["num_heads"], num_layers=T.PAPER["num_layers"],
                                 dropout=T.PAPER["dropout"])  # fmt: skip
    head = get_head(T.LOSS, embed_dim=T.PAPER["embed_dim"], dropout=T.PAPER["dropout"])
    n_params = sum(p.numel() for p in backbone.parameters()) + sum(p.numel() for p in head.parameters())
    trainer = T.TimedTrainer(
        backbone=backbone, head=head, train_loader=StepLoader(sets["train"], T.PAPER["batch_size"], True, args.seed, epoch_steps),
        val_loader=StepLoader(sets["val"], T.PAPER["batch_size"], False), loss_type=T.LOSS, lr=T.PAPER["lr"], weight_decay=T.PAPER["weight_decay"],
        device=device, checkpoint_dir=str(out), scaler_params={"x_mean": std.x_mean, "x_std": std.x_std},
    )  # fmt: skip
    steps_per_epoch = len(trainer.train_loader)
    max_epochs_eff = max(args.max_epochs, int(np.ceil(args.min_steps / steps_per_epoch)))
    passes = steps_per_epoch * T.PAPER["batch_size"] / max(len(sets["train"]), 1)
    logger.info("train cut samples %d, epoch = %d steps (= %.1f passes), max_epochs=%d", len(sets["train"]), steps_per_epoch, passes, max_epochs_eff)
    t0 = time.perf_counter()
    history = trainer.train(num_epochs=max_epochs_eff, patience=T.PAPER["patience"], verbose=True)
    t_train = time.perf_counter() - t0
    ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
    backbone.load_state_dict(ckpt["backbone_state_dict"])
    head.load_state_dict(ckpt["head_state_dict"])
    if (out / "latest.pt").exists():
        (out / "latest.pt").unlink()

    # ---- evaluation at the cuts (same code as KRAFTON-tf)
    g_val = T.predict_scores(backbone, head, sets["val"], device)
    g_test = T.predict_scores(backbone, head, sets["test"], device)
    df_val, df_test = T.build_frame(sets["val"], g_val, labs["val"]), T.build_frame(sets["test"], g_test, labs["test"])
    platt = T.fit_platt(df_val["raw_score"].to_numpy(), df_val["y"].to_numpy())
    df_val, df_test = T.finalize_frame(df_val, platt), T.finalize_frame(df_test, platt)
    canon_val, canon_test = T.write_pred(df_val, out / "pred_val.csv"), T.write_pred(df_test, out / "pred_test.csv")
    ref_dir = args.ref_dir or args.out_dir
    rows_check = {sp: T.assert_rows_match(c, ref_dir, args.cohort, sp) for sp, c in (("val", canon_val), ("test", canon_test))}
    spec = {"arm": ARM, "cohort": args.cohort, "seed": args.seed, "loss": T.LOSS, "max_epochs": args.max_epochs, "min_steps": args.min_steps, "paper": T.PAPER,
            "epoch_steps": epoch_steps, "device": device, "code": code_fingerprint()}  # fmt: skip
    res = {
        "arm": ARM, "cohort": args.cohort, "seed": args.seed, "device": device, "labels": "v2", "encoder": "transformer",
        "config_hash": config_hash(spec),
        "baseline": {
            "source": "krafton-ai/pubg-winprob 99ed9d1 transformer backbone, adapted (third_party/krafton_pubg_winprob/README_ADAPTATION.md), on the PTR team features",
            "loss": T.LOSS, "paper_hparams": T.PAPER, "max_epochs": args.max_epochs, "input": "dump team token at the cut: X(150) + team centre(2) + team_ai_frac(1), train-token z-score",
            "input_dim": n_features, "pos_zone_map_embeddings": "off", "n_params": int(n_params),
            "train_sampling": "phase cuts of the train matches (one sample per match and cut with >= 1 risk-set row); epoch = steps of one KRAFTON-tf epoch on this cohort",
            "p_hazard": "Platt scaling of the Cox hazard score, fitted on val rows only", "rank_score": "-raw_score (raw_score = hazard: higher = eliminated earlier)", "platt": platt,
        },  # fmt: skip
        "data": infos, "n_matches": {sp: infos[sp]["n_matches"] for sp in ("train", "val", "test")}, "token_check_vs_krafton_tf": token_check,
        "row_assertion": rows_check, "steps_per_epoch": int(steps_per_epoch), "train_passes_per_epoch": float(passes), "max_epochs_effective": int(max_epochs_eff),
        "min_steps": int(args.min_steps), "epochs_run": int(history["stopped_epoch"]), "best_epoch": int(history["best_epoch"]),
        "best_val_loss": float(history["best_val_loss"]), "stopped_early": bool(history["early_stopped"]), "train_sec": t_train, "data_sec": t_data,
        "history": [{"epoch": e + 1, "train_loss": float(tl), "val_loss": float(vl), "sec": float(sec)}
                    for e, (tl, vl, sec) in enumerate(zip(history["train_loss"], history["val_loss"], trainer.epoch_sec))],
        "git": git_info(), "val": M.all_metrics(canon_val), "test": M.all_metrics(canon_test),
        "krafton_native": {"val": T.native_metrics(sets["val"], g_val), "test": T.native_metrics(sets["test"], g_test),
                           "note": "winner accuracy / C-index of upstream metrics.py at the phase cuts over the alive squads, survival score = -hazard"},  # fmt: skip
    }  # fmt: skip
    res["elapsed_sec"] = time.perf_counter() - t_start
    (out / "metrics.json").write_text(json.dumps(to_jsonable(res), indent=1))
    logger.info("%s %s seed=%d  best_val_loss=%.4f  test AUC=%.4f  platt_a=%.3g  elapsed %.0fs", args.cohort, ARM, args.seed, res["best_val_loss"],
                res["test"]["hazard"]["auc"], platt["a"], res["elapsed_sec"])
    return res


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump-dir", type=Path, default=DEFAULT_DUMP)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA, help="KRAFTON-tf data dir (label tables, shards for the token check, extraction log)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--ref-dir", type=Path, default=None)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--min-steps", type=int, default=5000)
    ap.add_argument("--threads", type=int, default=4)
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
