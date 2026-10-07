"""HGB-kraftonfeat: gradient-boosted trees on the reconstructed KRAFTON squad inputs, evaluated at the phase cuts.

    python -m adaptation.hgb_kraftonfeat --cohort erangel_tpp_squad --seed 42 [--threads 1]

* rows: every (match, team, cut phase) of the dump's v2 risk set (winner censoring as ``model_stgcn.ptr.data.select_labels``)
  of the train / val / test split, ordered by (match_id, team_id, phase) like ``model_stgcn/ptr/hgb.py``; the label is
  ``y_hazard_v2`` of the next phase.
* features: the 55 reconstructed upstream squad features (``schema.FEATURES``, raw, un-standardised) at the cut
  t = ``phase_end_s - 10 s`` on the unified clock, read from the persisted table ``model_stgcn/dumps/bigcomp2027_kraftonfeat_v2``
  (``adaptation/kftable.py``; the risk-set rows ``in_risk_set == 1`` of train, val and test).  The table, the label tables and the
  dump must come from the same dump version (``DumpVersionError`` otherwise); on the v2 dump the risk-set rows include the comeback
  rows (causally dead at the cut, ``alive_at_cut == 0``) and the features are the causal, strictly-before-the-cut ones.  Left out: the per-player xyz,
  the zone values, the map, the PTR team features (``team_ai_frac`` included), the cut phase.  No other squad-level
  aggregate was built.
* model: ``model_stgcn.ptr.hgb.make_classifier(seed)`` (``HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06,
  l2_regularization=0.01, random_state=seed, early_stopping=False)``), ``p_hazard`` = predicted probability,
  ``rank_score = -p_hazard`` (the convention of the PTR HGB arms).
* outputs: ``<out>/<cohort>/HGB-kraftonfeat/seed<seed>/{pred_val.csv, pred_test.csv, metrics.json}`` in the PTR prediction
  format (``model_stgcn.ptr.metrics``); val / test rows are asserted equal to those of the PTR arms and the train rows
  are asserted equal to the rows ``model_stgcn.ptr.hgb.build_rows`` builds from the dump (same keys, same labels).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
for p in (str(ROOT), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from adaptation import schema as S  # noqa: E402
from adaptation.labels import check_label_version, load_labels  # noqa: E402
from model_stgcn.ptr import extra as EX  # noqa: E402  (read-only use: the table's dump version)
from model_stgcn.ptr import metrics as M  # noqa: E402  (read-only use)
from model_stgcn.ptr.data import DEFAULT_MATCH_TYPES, dump_version, load_cohort  # noqa: E402
from model_stgcn.ptr.hgb import build_rows as ptr_build_rows  # noqa: E402
from model_stgcn.ptr.hgb import make_classifier  # noqa: E402
from model_stgcn.ptr.util import config_hash, git_info, to_jsonable  # noqa: E402

logger = logging.getLogger("krafton.hgbkf")
ARM = "HGB-kraftonfeat"
DEFAULT_DATA = REPO / "model_stgcn/dumps/krafton_winprob_v2"  # label tables
DEFAULT_TABLE = REPO / "model_stgcn/dumps/bigcomp2027_kraftonfeat_v2"
DEFAULT_DUMP = REPO / "model_stgcn/dumps/bigcomp2027_phase_v2"
DEFAULT_OUT = REPO / "model_stgcn/checkpoints/bigcomp2027_ptr_v2"


def read_table(table_dir: Path, cohort: str, split: str) -> pd.DataFrame:
    return pd.read_csv(table_dir / cohort / f"{split}.csv.gz", dtype={"match_id": str}, float_precision="round_trip")  # exact float32 values


def build_rows(df: pd.DataFrame, labs: dict) -> dict:
    """Risk-set rows of one split from the table (``in_risk_set == 1``): raw 55 features + keys + labels, ordered like ptr.hgb.build_rows."""
    risk = df[df["in_risk_set"] == 1].reset_index(drop=True)
    kf = [f"kf_{n}" for n in S.FEATURES]
    keys = risk[["match_id", "team_id", "phase"]].copy()
    keys["team_id"] = keys["team_id"].astype(np.int64)
    keys["phase"] = keys["phase"].astype(np.int64)
    n = len(risk)
    frn, t5, mt, y_lab = np.empty(n, np.float32), np.empty(n, np.float32), np.empty(n, dtype=object), np.empty(n, np.float32)
    tid = keys["team_id"].to_numpy()
    ph = keys["phase"].to_numpy()
    for mid, idx in risk.groupby("match_id").indices.items():
        lab = labs[str(mid)]
        order = np.argsort(lab["team_ids"])
        j = order[np.searchsorted(lab["team_ids"][order], tid[idx])]
        assert np.array_equal(lab["team_ids"][j], tid[idx]), f"{mid}: team ids not found in the label table"
        assert lab["target"][ph[idx] - 1, j].all(), f"{mid}: a table risk-set row is not in the dump's risk set"
        frn[idx], t5[idx], mt[idx] = lab["final_rank_norm"][j], lab["top5"][j], lab["match_type"]
        y_lab[idx] = lab["y"][ph[idx] - 1, j]
    y = risk["y_hazard_v2"].to_numpy(dtype=np.float32)
    assert np.array_equal(y, y_lab), "table labels differ from the label tables"
    order = np.lexsort((keys["phase"].to_numpy(), keys["team_id"].to_numpy(), keys["match_id"].to_numpy()))  # match, team, phase
    return {
        "keys": keys.iloc[order].reset_index(drop=True), "X": risk[kf].to_numpy(dtype=np.float32)[order], "y": y[order],
        "final_rank_norm": frn[order], "top5": t5[order], "match_type": mt[order],
    }  # fmt: skip


def prediction_frame(rows: dict, p: np.ndarray) -> pd.DataFrame:
    df = rows["keys"].copy()
    df["y"] = rows["y"]
    df["p_hazard"] = p
    df["rank_score"] = -p  # survival-like, as the PTR HGB arms
    df["final_rank_norm"] = rows["final_rank_norm"]
    df["top5"] = rows["top5"]
    df["match_type"] = rows["match_type"]
    return df


def check_train_rows(rows: dict, dump_train: dict) -> dict:
    """The train rows equal the dump's v2 risk-set rows (keys and labels) as built by the PTR HGB baseline."""
    ref = ptr_build_rows(dump_train, "HGB-cut")
    same_keys = ref.keys.reset_index(drop=True).equals(rows["keys"])
    same_y = bool(np.array_equal(ref.y, rows["y"]))
    if not (same_keys and same_y):
        raise AssertionError(f"train rows differ from the PTR HGB rows (keys equal: {same_keys}, labels equal: {same_y}; {len(rows['y'])} vs {len(ref.y)})")
    return {"n_rows": int(len(ref.y)), "keys_equal": True, "labels_equal": True, "ref": "model_stgcn.ptr.hgb.build_rows(train, 'HGB-cut')"}


def code_fingerprint() -> str:
    import hashlib

    h = hashlib.sha1()
    for f in [ROOT / "adaptation" / f"{n}.py" for n in ("hgb_kraftonfeat", "kftable", "features", "schema", "labels")]:
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def run(args) -> dict:
    t_start = time.perf_counter()
    out = args.out_dir / args.cohort / ARM / f"seed{args.seed}"
    if (out / "metrics.json").is_file() and not args.redo:
        logger.info("skip %s: metrics.json exists", out)
        return json.loads((out / "metrics.json").read_text())
    from threadpoolctl import threadpool_limits

    t0 = time.perf_counter()
    dump = load_cohort(args.dump_dir, args.cohort, labels="v2", match_types=DEFAULT_MATCH_TYPES)
    dump_ver = dump_version(dump["train"])
    EX.check_table_version(args.table_dir, dump)  # a v1 table next to a v2 dump (or the reverse) is refused
    recs = {sp: load_labels(args.data_dir, args.cohort, sp) for sp in ("train", "val", "test")}
    for sp, rec in recs.items():
        check_label_version(rec, dump_ver, against=f"the {sp} dump of {args.cohort} under {args.dump_dir}")
    tables = {sp: read_table(args.table_dir, args.cohort, sp) for sp in ("train", "val", "test")}
    labs = {sp: {m["match_id"]: m for m in recs[sp]["matches"]} for sp in ("train", "val", "test")}
    rows = {sp: build_rows(tables[sp], labs[sp]) for sp in tables}
    train_check = check_train_rows(rows["train"], dump["train"])
    del dump
    train = rows["train"]
    if len(np.unique(train["y"])) < 2:
        raise ValueError(f"{args.cohort}: a single hazard class in the train rows")
    t_data = time.perf_counter() - t0
    clf = make_classifier(args.seed)
    t1 = time.perf_counter()
    with threadpool_limits(limits=args.threads):
        clf.fit(train["X"], train["y"])
    fit_sec = time.perf_counter() - t1
    out.mkdir(parents=True, exist_ok=True)
    res = {
        "arm": ARM, "variant": ARM, "cohort": args.cohort, "seed": args.seed, "labels": "v2", "label_fields": {"y": "y_hazard_v2", "target": "target_mask_v2"},
        "dump_version": dump_ver,
        "config_hash": config_hash({"arm": ARM, "cohort": args.cohort, "seed": args.seed, "code": code_fingerprint(), "classifier": clf.get_params()}),
        "baseline": {
            "source": "reconstructed KRAFTON squad inputs (third_party/krafton_pubg_winprob, adaptation/features.py) at the phase cuts, GBDT of model_stgcn/ptr/hgb.py",
            "feature_table": f"{args.table_dir}/<cohort>/{{train,val,test}}.csv.gz (rows with in_risk_set = 1)",
            "features": list(S.FEATURES), "left_out": "per-player xyz, zone values, map, PTR team features (team_ai_frac included), cut phase",
            "zero_filled_features": [f for f, v in S.PARITY.items() if v[0] == "zero"], "proxy_features": [f for f, v in S.PARITY.items() if v[0] == "proxy"],
            "train_rows": "risk-set rows of the train matches at the phase cuts (t = phase_end_s - 10 s), winners censored (v2)",
        },  # fmt: skip
        "n_matches": {sp: len(labs[sp]) for sp in labs}, "n_features": int(train["X"].shape[1]), "n_train_rows": int(len(train["y"])),
        "n_iter": int(clf.n_iter_), "classifier": clf.get_params(), "fit_sec": fit_sec, "data_sec": t_data, "train_rows_check": train_check, "git": git_info(),
    }  # fmt: skip
    ref_dir = args.ref_dir or args.out_dir
    from adaptation.train import assert_rows_match

    res["row_assertion"] = {}
    for sp in ("val", "test"):
        p = clf.predict_proba(rows[sp]["X"])[:, 1]
        df = prediction_frame(rows[sp], p)
        M.write_predictions(df, out / f"pred_{sp}.csv")
        canon = M.read_predictions(df)
        res["row_assertion"][sp] = assert_rows_match(canon, ref_dir, args.cohort, sp)
        res[sp] = M.all_metrics(canon)
    res["elapsed_sec"] = time.perf_counter() - t_start
    (out / "metrics.json").write_text(json.dumps(to_jsonable(res), indent=1))
    logger.info("%s %s seed=%d  n_train_rows=%d  test AUC=%.4f  (fit %.1fs, elapsed %.1fs)", args.cohort, ARM, args.seed, res["n_train_rows"],
                res["test"]["hazard"]["auc"], fit_sec, res["elapsed_sec"])
    return res


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA, help="label tables (adaptation.labels)")
    ap.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE, help="persisted KRAFTON feature table (adaptation.kftable)")
    ap.add_argument("--dump-dir", type=Path, default=DEFAULT_DUMP)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--ref-dir", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=1, help="OpenMP threads of the boosting (resource cap: <= 4 in total)")
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
