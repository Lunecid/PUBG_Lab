"""H0 arms: Cox scores + an estimated phase baseline hazard (grouped-time Cox, Prentice-Gloeckler cloglog).

    python -m adaptation.h0_baseline [--cohorts a,b,...] [--device cpu] [--threads 3] [--redo]
        [--ckpt-dir model_stgcn/checkpoints/bigcomp2027_ptr_v2] [--dump-dir model_stgcn/dumps/bigcomp2027_phase_v2]
        [--data-dir model_stgcn/dumps/krafton_winprob_v2]

``--ckpt-dir`` (the run directory that holds the K1 / K2 arms and receives the H0 arms), ``--dump-dir`` (the phase dump the K2 samples are
built from) and ``--data-dir`` (the KRAFTON label tables and extraction logs) are explicit; the defaults are the v2 roots of
``model_stgcn.ptr.paths``.  The dump version (``meta['version']``) is compared with the version of the other two and a v1 root next to a v2
dump (or the reverse) is refused before anything is read.

The Cox arms (K1 = KRAFTON-tf, K2 = KRAFTON-tf-ourfeat) are trained with a ranking loss, so their raw score ``s`` (higher =
eliminated earlier) is a log relative hazard without a baseline.  ``p_hazard`` of those arms is a phase-agnostic Platt map
fitted on val.  The interval label ``Y`` = "eliminated by the end of cut phase k + 1" (risk set = teams alive at the cut k,
labels v2) is a grouped-time outcome, so a Cox model gives

    P(Y = 1 | s, k) = 1 - exp(-exp(gamma_k + s)),        gamma_k = log(H0(k + 1) - H0(k))

(cloglog link, offset ``s`` with the coefficient fixed at 1, one baseline increment ``gamma_k`` per cut phase), fitted by
maximum likelihood per cohort and phase.  Arms written to ``<ckpt>/<cohort>/<ARM>/seed42/{pred_val.csv, pred_test.csv, metrics.json}``:

    KRAFTON-tf-ourfeat-h0          "KR-Cox+H0"      K2 scores, gamma_k fitted on the TRAIN risk-set rows at our cuts   (train_scores.csv.gz)
    KRAFTON-tf-ourfeat-h0-slope                      + one shared slope beta on s, fitted jointly with gamma_k (sensitivity)
    KRAFTON-tf-h0                  "KRAFTON+H0"     K1 scores, gamma_k fitted on the VAL risk-set rows (see below)
    KRAFTON-tf-h0-slope                              + shared slope
    KRAFTON-tf-ourfeat-h0val       control          K2 scores, gamma_k fitted on VAL (the like-for-like twin of KRAFTON-tf-h0)
    KRAFTON-tf-ourfeat-h0val-slope

K1 deviation: the TRAIN-cut input of the K1 backbone is not on disk.  Its token carries the 55 reconstructed features (persisted
in the kf table ``bigcomp2027_kraftonfeat_v2``) AND the per-player xyz positions, the two zone circles and the map embedding; the K1 train
shards hold the 192 random / anchor times of upstream's sampler (no cut), and positions / zones at the train cuts would need the
database (not used here).  K1's val / test shards are exactly the cuts, so K1's gamma_k is fitted on val; the K2 control arms
(``-h0val``) show what the train -> val change does for the same recipe.

Every arm asserts: (i) the val / test raw_score of the parent is reproduced by inference with the saved best checkpoint (max abs
diff recorded); (ii) orientation, corr(s, y) > 0 on the fit rows; (iii) rows, labels and match types equal those of the PTR
arms (HGB-cut, M00, M11R); (iv) the per-phase AUC equals the parent's (the map is monotone within a phase; max abs diff recorded);
(v) beta > 0 for the slope variants.  ``raw_score`` is kept, ``rank_score`` = -raw_score as in the parent.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "3")  # resource cap: at most 3 CPU threads

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
for p in (str(ROOT), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from adaptation import ourfeat as OF  # noqa: E402  (K2: backbone, cut samples)
from adaptation import schema as S  # noqa: E402
from adaptation import train as T  # noqa: E402  (K1 helpers: prediction frame, files, row assertion)
from adaptation.data import load_split  # noqa: E402
from adaptation.h0_fit import cloglog_prob, fit_cloglog_offset, predict_h0  # noqa: E402,F401  (the estimator; numpy only)
from adaptation.labels import load_labels  # noqa: E402
from adaptation.phase_auc import COHORTS, per_phase_auc  # noqa: E402
from model_stgcn.ptr import metrics as M  # noqa: E402  (read-only use)
from model_stgcn.ptr import paths as P  # noqa: E402  (the v2 roots and the version guard)
from model_stgcn.ptr.data import Standardizer, load_cohort  # noqa: E402
from model_stgcn.ptr.util import config_hash, git_info, to_jsonable  # noqa: E402
from src.models import TransformerBackbone  # noqa: E402
from src.models.heads import get_head  # noqa: E402

logger = logging.getLogger("krafton.h0")
CKPT = P.RUNS_DIR  # defaults of the command line = the v2 roots; every function below takes its roots explicitly
DUMP = P.DUMP_DIR
DATA = P.KRAFTON_DATA_DIR
SEED = 42
K1, K2 = "KRAFTON-tf", "KRAFTON-tf-ourfeat"
TOL_REPRO = 1e-4  # val / test raw_score from inference vs the stored file (float32 net, MPS vs CPU: observed ~1e-6)
TOL_AUC_FILES = 5e-5  # per-phase AUC, written child file vs parent file (new ties where p rounds to 1 in the 9-digit file)
# arm -> (parent, score source for the fit, shared slope)
ARMS = {
    "KRAFTON-tf-ourfeat-h0": (K2, "train", False), "KRAFTON-tf-ourfeat-h0-slope": (K2, "train", True),
    "KRAFTON-tf-h0": (K1, "val", False), "KRAFTON-tf-h0-slope": (K1, "val", True),
    "KRAFTON-tf-ourfeat-h0val": (K2, "val", False), "KRAFTON-tf-ourfeat-h0val-slope": (K2, "val", True),
}  # fmt: skip


# --------------------------------------------------------------------------- the roots
@dataclass(frozen=True)
class Roots:
    """The three directories of an H0 run: ``ckpt`` (K1 / K2 arms in, H0 arms out), ``dump`` (phase dump), ``data`` (KRAFTON labels and logs)."""

    ckpt: Path = CKPT
    dump: Path = DUMP
    data: Path = DATA

    def __post_init__(self) -> None:
        for name in ("ckpt", "dump", "data"):
            object.__setattr__(self, name, Path(getattr(self, name)))

    def check_versions(self) -> str:
        """Refuse a v1 root next to a v2 dump (or the reverse); returns the dump's version."""
        try:
            return P.check_same_version(self.dump, {"ckpt (--ckpt-dir)": self.ckpt, "KRAFTON data (--data-dir)": self.data})
        except P.VersionMixError as exc:
            raise SystemExit(f"refusing to run: {exc}") from exc


# --------------------------------------------------------------------------- scores of the parents
def _load_ckpt(arm: str, cohort: str, ckpt: Path = CKPT) -> dict:
    return torch.load(Path(ckpt) / cohort / arm / f"seed{SEED}" / "best.pt", map_location="cpu", weights_only=False)


def _stored(arm: str, cohort: str, split: str, ckpt: Path = CKPT) -> pd.DataFrame:
    return pd.read_csv(Path(ckpt) / cohort / arm / f"seed{SEED}" / f"pred_{split}.csv", dtype={"match_id": str})


def _repro(df_new: pd.DataFrame, stored: pd.DataFrame) -> float:
    """Max abs difference of raw_score (inference vs stored file); the key sets and the labels must be identical."""
    key = ["match_id", "team_id", "phase"]
    a = df_new.sort_values(key, kind="mergesort").reset_index(drop=True)
    b = stored.sort_values(key, kind="mergesort").reset_index(drop=True)
    for c in key:
        if not np.array_equal(a[c].to_numpy(), b[c].to_numpy()):
            raise AssertionError(f"inference rows differ from the stored file in '{c}' ({len(a)} vs {len(b)} rows)")
    if not np.array_equal(a["y"].to_numpy(), b["y"].to_numpy()):
        raise AssertionError("labels differ between the inference frame and the stored file")
    return float(np.abs(a["raw_score"].to_numpy() - b["raw_score"].to_numpy()).max())


def k2_frames(cohort: str, device: str, roots: Roots = Roots()) -> tuple[dict, dict]:
    """Raw K2 scores of the risk-set rows at our cuts for train / val / test by inference with best.pt (as ourfeat.py)."""
    splits = load_cohort(roots.dump, cohort, labels="v2")
    std = Standardizer.fit(splits["train"], "v2")
    ck = _load_ckpt(K2, cohort, roots.ckpt)
    for k_ck, k_sd in (("x_mean", std.x_mean), ("x_std", std.x_std)):
        if not np.allclose(ck["scaler_params"][k_ck], k_sd, rtol=0, atol=1e-9):
            raise AssertionError(f"{cohort}: the refitted standardiser differs from the checkpoint's {k_ck}")
    t_win = OF._t_win_map(roots.data)
    sets = {sp: OF.build_samples(splits[sp], std, t_win, sp)[0] for sp in ("train", "val", "test")}
    del splits
    n_features = int(sets["train"].X57.shape[1])
    backbone = OF.TokenOnlyBackbone(input_dim=n_features, embed_dim=T.PAPER["embed_dim"], num_heads=T.PAPER["num_heads"], num_layers=T.PAPER["num_layers"],
                                    dropout=T.PAPER["dropout"])  # fmt: skip
    head = get_head(T.LOSS, embed_dim=T.PAPER["embed_dim"], dropout=T.PAPER["dropout"])
    backbone.load_state_dict(ck["backbone_state_dict"])
    head.load_state_dict(ck["head_state_dict"])
    backbone.to(device)
    head.to(device)
    frames, repro = {}, {}
    for sp in ("train", "val", "test"):
        labs = {m["match_id"]: m for m in load_labels(roots.data, cohort, sp)["matches"]}
        frames[sp] = T.build_frame(sets[sp], T.predict_scores(backbone, head, sets[sp], device), labs)
        if sp != "train":
            repro[sp] = _repro(frames[sp], _stored(K2, cohort, sp, roots.ckpt))
    return frames, repro


def k1_cut_repro(cohort: str, device: str, roots: Roots = Roots()) -> dict:
    """K1 val / test: inference with best.pt on the cut shards reproduces the stored raw_score (the fit uses the stored scores)."""
    ck = _load_ckpt(K1, cohort, roots.ckpt)
    backbone = TransformerBackbone(input_dim=S.N_INPUT, embed_dim=T.PAPER["embed_dim"], num_heads=T.PAPER["num_heads"], num_layers=T.PAPER["num_layers"],
                                   dropout=T.PAPER["dropout"], encoder_type="transformer", num_maps=len(S.MAPS))  # fmt: skip
    head = get_head(T.LOSS, embed_dim=T.PAPER["embed_dim"], dropout=T.PAPER["dropout"])
    backbone.load_state_dict(ck["backbone_state_dict"])
    head.load_state_dict(ck["head_state_dict"])
    backbone.to(device)
    head.to(device)
    out = {}
    for sp in ("val", "test"):
        ss = load_split(roots.data, cohort, sp).finalize(ck["scaler_params"]["mean"], ck["scaler_params"]["std"])
        labs = {m["match_id"]: m for m in load_labels(roots.data, cohort, sp)["matches"]}
        out[sp] = _repro(T.build_frame(ss, T.predict_scores(backbone, head, ss, device), labs), _stored(K1, cohort, sp, roots.ckpt))
    return out


# --------------------------------------------------------------------------- checks and arm writing
def orientation(df: pd.DataFrame) -> dict:
    """corr(s, y) pooled (asserted > 0) and the smallest per-phase correlation (recorded)."""
    c = float(np.corrcoef(df["raw_score"], df["y"])[0, 1])
    per = {int(k): float(np.corrcoef(g["raw_score"], g["y"])[0, 1]) for k, g in df.groupby("phase") if g["y"].nunique() == 2}
    if not c > 0:
        raise AssertionError(f"corr(raw_score, y) = {c:.4f} <= 0: the score is not oriented as a hazard")
    return {"corr_pooled": c, "corr_by_phase": per, "corr_min_phase": float(min(per.values()))}


def _auc_diff(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    m = a.merge(b, on="phase", suffixes=("_a", "_b"))
    ok = m["auc_a"].notna() & m["auc_b"].notna()
    d = (m.loc[ok, "auc_a"] - m.loc[ok, "auc_b"]).abs()
    return {"max_abs_diff": float(d.max()) if len(d) else 0.0, "n_phases": int(ok.sum())}


def phase_auc_checks(child_mem: pd.DataFrame, child_csv: pd.DataFrame, parent_csv: pd.DataFrame) -> dict:
    """Child vs parent per-phase AUC (every phase with both classes), max abs difference over the phases, from the written files.

    The map s -> p is non-decreasing within a phase (``monotone_within_phase``, asserted on the unrounded p), so a per-phase AUC can only
    change through NEW TIES: rows whose p rounds to the same 9-significant-digit value in the child file and not in the parent's.  That
    happens at the top, where exp(-exp(gamma_k + s)) < 5e-10 (``n_rows_p_tied_at_one``: rows written as 1)."""
    mono = True
    for _k, g in child_mem.groupby("phase"):
        o = np.argsort(g["raw_score"].to_numpy(), kind="mergesort")
        mono &= bool((np.diff(g["p_hazard"].to_numpy()[o]) >= 0).all())
    files = _auc_diff(per_phase_auc(child_csv, "p_hazard", 1), per_phase_auc(parent_csv, "p_hazard", 1))
    return {"vs_parent_files": files, "monotone_within_phase": mono, "n_rows_p_tied_at_one": int((child_csv["p_hazard"] >= 1.0).sum()), "n_rows": int(len(child_csv))}


def cal_gap(df: pd.DataFrame) -> float:
    """Row-weighted mean over the cut phases of |mean p - observed hazard rate| (as phase_auc.py)."""
    return float(sum(abs(g["p_hazard"].mean() - g["y"].mean()) * len(g) for _k, g in df.groupby("phase")) / len(df))


def write_arm(cohort: str, arm: str, fit: dict, checks: dict, repro: dict, ckpt: Path = CKPT) -> dict:
    ckpt = Path(ckpt)
    parent, source, slope = ARMS[arm]
    out = ckpt / cohort / arm / f"seed{SEED}"
    out.mkdir(parents=True, exist_ok=True)
    canon, frames, mem = {}, {}, {}
    for sp in ("val", "test"):
        base = _stored(parent, cohort, sp, ckpt)
        df = base.copy()
        df["p_hazard"] = predict_h0(df["raw_score"].to_numpy(), df["phase"].to_numpy(), fit)
        df["rank_score"] = -df["raw_score"]  # survival direction, as the parent
        mem[sp] = df
        canon[sp] = T.write_pred(df, out / f"pred_{sp}.csv")
        frames[sp] = pd.read_csv(out / f"pred_{sp}.csv", dtype={"match_id": str})  # what the downstream tools read
    row_check = {sp: T.assert_rows_match(canon[sp], ckpt, cohort, sp) for sp in ("val", "test")}
    auc_check = {sp: phase_auc_checks(mem[sp], M.read_predictions(out / f"pred_{sp}.csv"), M.read_predictions(ckpt / cohort / parent / f"seed{SEED}" / f"pred_{sp}.csv"))
                 for sp in ("val", "test")}  # fmt: skip
    if slope and not fit["beta"] > 0:
        raise AssertionError(f"{cohort}/{arm}: beta = {fit['beta']:.4f} <= 0, the map is not increasing in the score")
    for sp in ("val", "test"):
        if not auc_check[sp]["monotone_within_phase"] or auc_check[sp]["vs_parent_files"]["max_abs_diff"] > TOL_AUC_FILES:
            raise AssertionError(f"{cohort}/{arm}/{sp}: per-phase AUC differs from the parent: {auc_check[sp]}")
    spec = {"arm": arm, "cohort": cohort, "seed": SEED, "parent": parent, "fit_split": source, "slope": slope, "method": "cloglog-offset-per-phase"}
    res = {
        "arm": arm, "cohort": cohort, "seed": SEED, "parent_arm": parent, "labels": "v2", "config_hash": config_hash(spec),
        "baseline": {
            "method": "grouped-time Cox baseline (Prentice-Gloeckler): P(Y=1 | s, k) = 1 - exp(-exp(gamma_k + beta * s)); gamma_k = log(H0(k+1) - H0(k)); "
                      "maximum likelihood per cohort and cut phase k = 1..7" + ("; beta shared across phases and fitted jointly" if slope else "; beta fixed at 1 (offset)"),
            "fit_split": source,
            "fit_note": ("train risk-set rows at our cuts (in-sample scores of the parent)" if source == "train"
                         else "VAL risk-set rows (val is the parent's early-stopping / Platt split); the train-cut input of the K1 backbone is not on disk" if parent == K1
                         else "VAL risk-set rows (control: same scores as the train-fit arm, other fit split)"),
            "fit": fit, "p_hazard": "1 - exp(-exp(gamma_k + beta * raw_score))", "rank_score": "-raw_score (the parent's)",
            "score": "raw_score of the parent (hazard: higher = eliminated earlier), unchanged",
        },
        "checks": {"orientation_on_fit_rows": checks["orientation"], "raw_score_reproduced_by_inference_max_abs_diff": repro,
                   "row_assertion": row_check, "within_phase_auc_vs_parent": auc_check},  # fmt: skip
        "calibration_gap_by_phase": {sp: cal_gap(frames[sp]) for sp in ("val", "test")},
        "val": M.all_metrics(canon["val"]), "test": M.all_metrics(canon["test"]), "git": git_info(),
        "note": "val metrics are in-sample for the arms fitted on val",
    }  # fmt: skip
    (out / "metrics.json").write_text(json.dumps(to_jsonable(res), indent=1))
    return res


def run_cohort(cohort: str, device: str, redo: bool, roots: Roots = Roots()) -> dict:
    t0 = time.perf_counter()
    if not redo and all((roots.ckpt / cohort / a / f"seed{SEED}" / "metrics.json").is_file() for a in ARMS):
        logger.info("skip %s: every H0 arm exists", cohort)
        return {}
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
    frames2, repro2 = k2_frames(cohort, device, roots)
    repro1 = k1_cut_repro(cohort, device, roots)
    for name, rp in (("K2", repro2), ("K1", repro1)):
        for sp, d in rp.items():
            if d > TOL_REPRO:
                raise AssertionError(f"{cohort}: {name} {sp} raw_score from inference differs from the stored file by {d:.2e} (> {TOL_REPRO})")
    ori = {"K2_train": orientation(frames2["train"]), "K2_val": orientation(_stored(K2, cohort, "val", roots.ckpt)), "K1_val": orientation(_stored(K1, cohort, "val", roots.ckpt))}
    fit_frames = {(K2, "train"): frames2["train"], (K2, "val"): _stored(K2, cohort, "val", roots.ckpt), (K1, "val"): _stored(K1, cohort, "val", roots.ckpt)}
    res = {}
    for arm, (parent, source, slope) in ARMS.items():
        df = fit_frames[(parent, source)]
        fit = fit_cloglog_offset(df["raw_score"], df["y"], df["phase"], slope=slope)
        key = f"{'K2' if parent == K2 else 'K1'}_{source}"
        res[arm] = write_arm(cohort, arm, fit, {"orientation": ori[key]}, repro2 if parent == K2 else repro1, roots.ckpt)
        logger.info("%s %-34s fit=%s beta=%.3f  test AUC %.4f NLL %.4f Brier %.4f", cohort, arm, source, fit["beta"], res[arm]["test"]["hazard"]["auc"],
                    res[arm]["test"]["hazard"]["nll"], res[arm]["test"]["hazard"]["brier"])
    d = roots.ckpt / cohort / "KRAFTON-tf-ourfeat-h0" / f"seed{SEED}"
    frames2["train"][["match_id", "team_id", "phase", "y", "raw_score"]].to_csv(d / "train_scores.csv.gz", index=False, float_format="%.9g", compression="gzip")
    logger.info("%s done in %.0fs", cohort, time.perf_counter() - t0)
    return res


def resolve_cohorts(spec: str, roots: Roots) -> list[str]:
    """``all`` = the cohorts of the dump directory (the hard-coded 12-cohort list when there is none), in the order of that list."""
    if spec != "all":
        return [c for c in spec.split(",") if c]
    if roots.dump.is_dir():
        from model_stgcn.ptr.cohorts import discover_cohorts

        have = set(discover_cohorts(roots.dump))
        if not have:
            raise SystemExit(f"no complete cohort (train / val / test .pt) in the dump directory {roots.dump}")
        return [c for c in COHORTS if c in have] + sorted(have - set(COHORTS))
    return list(COHORTS)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohorts", default="all")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--ckpt-dir", type=Path, default=CKPT, help="run directory: K1 / K2 arms are read from it and the H0 arms are written to it (default: the v2 root)")
    ap.add_argument("--dump-dir", type=Path, default=DUMP, help="phase dump directory (default: the v2 dump)")
    ap.add_argument("--data-dir", type=Path, default=DATA, help="KRAFTON data directory: label tables, extraction logs (default: the v2 root)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.threads > 3:
        raise SystemExit("resource cap: at most 3 CPU threads")
    roots = Roots(a.ckpt_dir, a.dump_dir, a.data_dir)
    roots.check_versions()
    P.claim_root(roots.ckpt, roots.dump, role="runs")
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for c in resolve_cohorts(a.cohorts, roots):
        run_cohort(c, a.device, a.redo, roots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
