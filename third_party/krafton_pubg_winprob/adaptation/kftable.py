"""Persisted KRAFTON feature table at the PTR phase cuts (reusable by other models; READ-ONLY database access).

    python -m adaptation.kftable --cohorts <c1,c2,...> --splits train,val,test --workers 1 [--out-dir <root>]

Output ``<root>/<cohort>/{train,val,test}.csv.gz`` (+ ``<root>/meta.json``), default root
``model_stgcn/dumps/bigcomp2027_kraftonfeat_v1``.  One row per (match_id, team_id, phase) for the cut phases k = 1..P-1
(PTR convention: k = dump slot + 1, 1..7) and for EVERY team that is alive at the start of phase k (the PTR token set
``node_mask_v2`` = ``model_stgcn.ptr.data.event_alive_mask``), which contains every row of the dump's v2 risk set
(winner censoring as ``select_labels``).  Columns:

    match_id, team_id, phase            keys
    cut_t                               phase_end_s - 10 s on the unified clock W (event_time - stdb.matches.started_at)
    in_risk_set                         1 if the row is in the dump's v2 risk set (a hazard label exists)
    y_hazard_v2                         the v2 next-phase hazard label of the risk-set rows (empty otherwise)
    alive_at_cut                        1 if the team is alive at cut_t (team_death_s > cut_t, winners always alive)
    kf_<name> (55)                      the reconstructed upstream squad features at cut_t (adaptation/features.py)

Strictly causal: every feature at cut_t reads events / samples with time < cut_t (cumulative windows) or <= cut_t
(snapshots) only.  A team that is alive at phase start but dead at the cut keeps its cumulative action features and gets
the upstream dead-squad snapshot values (-1 = no alive member for the means, 0 for sums / levels / counts).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REPO = ROOT.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from adaptation import extract as E  # noqa: E402
from adaptation import schema as S  # noqa: E402
from adaptation.features import compute_features  # noqa: E402
from adaptation.labels import load_labels  # noqa: E402

DEFAULT_ROOT = REPO / "model_stgcn/dumps/bigcomp2027_kraftonfeat_v1"
KF = [f"kf_{n}" for n in S.FEATURES]
FLAG_COLUMNS = ["cut_t", "in_risk_set", "y_hazard_v2", "alive_at_cut"]
COLUMNS = ["match_id", "team_id", "phase"] + FLAG_COLUMNS + KF
BATCH = 8


def _event_alive(lab: dict) -> np.ndarray:
    """[P, N] alive at phase start: the PTR token rule (model_stgcn.ptr.data.event_alive_mask, read-only use)."""
    from model_stgcn.ptr.data import event_alive_mask

    return event_alive_mask(lab).numpy()


def table_rows(mid: str, lab: dict, raw: dict, extra: dict) -> dict:
    """Rows of one match: arrays of keys, flags and the 55 features (see the module docstring)."""
    P = len(lab["phase_end_s"])
    pe, pm = lab["phase_end_s"], lab["phase_mask"]
    alive_start = _event_alive(lab)
    T_eff = np.where(lab["team_won"], np.inf, lab["team_death_s"])
    slots = [p for p in range(P - 1) if bool(pm[p]) and np.isfinite(pe[p])]  # cut phases k = 1..P-1
    if not slots:
        return {"n": 0}
    ts = pe[slots] - 10.0
    if (ts < extra["flags"]["first_state_w"]).any():
        raise ValueError(f"a cut falls before the first game state ({extra['flags']['first_state_w']:.1f}s)")
    out = compute_features(raw, ts)
    team_ids = lab["team_ids"]
    keep_cols = {k: [] for k in ("team_id", "phase", "cut_t", "in_risk_set", "y", "alive_at_cut")}
    feats = []
    for kk, p in enumerate(slots):
        idx = np.flatnonzero(alive_start[p])
        risk = lab["target"][p]
        if (risk & ~alive_start[p]).any():
            raise AssertionError(f"{mid} slot {p}: risk-set rows outside the alive-at-start tokens")
        if (risk & ~(T_eff > ts[kk])).any():
            raise AssertionError(f"{mid} slot {p}: risk-set rows that are dead at the cut")
        if idx.size == 0:
            continue
        keep_cols["team_id"].append(team_ids[idx])
        keep_cols["phase"].append(np.full(idx.size, p + 1, dtype=np.int64))
        keep_cols["cut_t"].append(np.full(idx.size, ts[kk], dtype=np.float64))
        keep_cols["in_risk_set"].append(risk[idx].astype(np.int8))
        keep_cols["y"].append(np.where(risk[idx], lab["y"][p, idx], np.nan).astype(np.float32))
        keep_cols["alive_at_cut"].append((T_eff[idx] > ts[kk]).astype(np.int8))
        feats.append(out["F"][idx, kk, :])
    if not feats:
        return {"n": 0}
    res = {k: np.concatenate(v) for k, v in keep_cols.items()}
    res["F"] = np.concatenate(feats).astype(np.float32)
    res["n"] = len(res["team_id"])
    res["match_id"] = mid
    res["n_risk"] = int(sum(int(lab["target"][p].sum()) for p in slots))
    return res


def run_job(job: dict) -> dict:
    t0 = time.time()
    cohort, split, bidx, labs = job["cohort"], job["split"], job["bidx"], job["labs"]
    qa, parts = [], []
    try:
        mids = [lab["match_id"] for lab in labs]
        T = E.fetch_batch(mids)
        meta = T["meta"].set_index("match_id")
        TT = {k: E._by_match(v) for k, v in T.items() if k != "meta"}
        for lab in labs:
            mid = lab["match_id"]
            try:
                if mid not in meta.index or meta.loc[mid].started_ep is None:
                    raise ValueError("no stdb.matches row / started_at")
                row = meta.loc[mid]
                raw, extra = E.build_raw(mid, lab, TT, row, float(row.started_ep))
                if extra["flags"]["n_death_mismatch"]:
                    raise ValueError(f"{extra['flags']['n_death_mismatch']} death time mismatches vs the dump")
                r = table_rows(mid, lab, raw, extra)
                if r["n"]:
                    parts.append(r)
                qa.append({"match_id": mid, "ok": True, "n_rows": int(r["n"]), "n_risk": int(r.get("n_risk", 0))})
            except Exception:
                qa.append({"match_id": mid, "ok": False, "error": traceback.format_exc()[-500:]})
    except Exception:
        qa.append({"match_id": "BATCH", "ok": False, "error": traceback.format_exc()[-600:]})
    arrays = None
    if parts:
        arrays = {
            "match_id": np.concatenate([np.full(r["n"], r["match_id"], dtype=object) for r in parts]),
            **{k: np.concatenate([r[k] for r in parts]) for k in ("team_id", "phase", "cut_t", "in_risk_set", "y", "alive_at_cut", "F")},
        }  # fmt: skip
    return {"cohort": cohort, "split": split, "bidx": bidx, "qa": qa, "arrays": arrays, "sec": time.time() - t0}


def to_frame(chunks: list[dict]) -> pd.DataFrame:
    cols = {
        "match_id": np.concatenate([c["match_id"] for c in chunks]),
        "team_id": np.concatenate([c["team_id"] for c in chunks]).astype(np.int64),
        "phase": np.concatenate([c["phase"] for c in chunks]).astype(np.int64),
        "cut_t": np.concatenate([c["cut_t"] for c in chunks]),
        "in_risk_set": np.concatenate([c["in_risk_set"] for c in chunks]).astype(np.int8),
        "y_hazard_v2": np.concatenate([c["y"] for c in chunks]),
        "alive_at_cut": np.concatenate([c["alive_at_cut"] for c in chunks]).astype(np.int8),
    }
    F = np.concatenate([c["F"] for c in chunks])
    df = pd.DataFrame(cols)
    for j, name in enumerate(KF):
        df[name] = F[:, j]
    df = df.sort_values(["match_id", "team_id", "phase"], kind="mergesort").reset_index(drop=True)
    if df.duplicated(["match_id", "team_id", "phase"]).any():
        raise AssertionError("duplicate (match_id, team_id, phase) keys")
    return df[COLUMNS]


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip", float_format="%.9g")


def feature_meta() -> list[dict]:
    from adaptation.parity import block_of

    out = []
    for n in S.FEATURES:
        st, src, note = S.PARITY[n]
        out.append({"column": f"kf_{n}", "upstream_name": n, "block": block_of(n), "status": st, "source": src, "note": note,
                    "zero_filled": st == "zero", "approximate": st == "proxy"})  # fmt: skip
    return out


def write_meta(root: Path, counts: dict, args) -> None:
    from model_stgcn.ptr.util import git_info, to_jsonable

    meta = {
        "name": "bigcomp2027_kraftonfeat_v1", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": git_info(),
        "what": "KRAFTON (krafton-ai/pubg-winprob 99ed9d1) squad-level continuous features reconstructed from the raw pubg.* tables at the PTR phase cuts",
        "code": "third_party/krafton_pubg_winprob/adaptation/{features,extract,kftable,schema}.py; definitions: README_ADAPTATION.md, PARITY_TABLE.csv",
        "files": "<cohort>/{train,val,test}.csv.gz (gzip CSV, floats with 9 significant digits; pyarrow is not installed in the interpreter)",
        "keys": ["match_id", "team_id", "phase"],
        "phase": "cut phase k = 1..7 (PTR CSV convention: k = dump slot + 1); cut_t = phase_end_s[k-1] - 10 s on the unified clock W = event_time - stdb.matches.started_at",
        "rows": "every team alive at the start of phase k (PTR token set node_mask_v2 = model_stgcn.ptr.data.event_alive_mask, winners always alive) for every cut phase "
                "k = 1..P-1 whose phase exists; contains every row of the dump's v2 risk set (winner censoring as select_labels)",
        "flag_columns": {
            "cut_t": "float seconds on W; phase_end_s = cut_t + 10",
            "in_risk_set": "1 = row of the dump's v2 risk set (target_mask_v2 after winner censoring)",
            "y_hazard_v2": "next-phase hazard label of the risk-set rows (v2, winners 0); empty for the other rows",
            "alive_at_cut": "1 = team_death_s > cut_t (winners always alive); 0 = alive at phase start but dead at the cut",
        },
        "causality": "features at cut_t read events with time < cut_t (cumulative windows) or <= cut_t (snapshots) only; prefix replay on real matches gives identical features (validate_causal)",
        "dead_squad_convention": "rows with alive_at_cut = 0: cumulative action features as upstream (summed over all members); mean snapshot features (n_teammates, n_enemies, dist_*) = -1, "
                                 "dist_*_v2 = -1, sums / levels / counts over alive members = 0 (squad_alive_count, squad_total_health, items, is_vehicle_count, ...)",
        "unknown_values": "-1 in dist_* / n_teammates / n_enemies = no alive member (or no valid zone, zone radius < 1 m is treated as no zone)",
        "left_out": "per-player xyz, zone values, map, phase",
        "feature_columns": feature_meta(),
        "zero_filled": [f"kf_{n}" for n, v in S.PARITY.items() if v[0] == "zero"],
        "approximate": [f"kf_{n}" for n, v in S.PARITY.items() if v[0] == "proxy"],
        "counts": counts,
    }
    (root / "meta.json").write_text(json.dumps(to_jsonable(meta), indent=1))


def verify(cohort: str, root: Path, data_dir: Path, label_dir: Path) -> dict:
    """The table's alive-at-cut rows of val / test equal the KRAFTON-tf cut shards (same features, same squads, same cuts)."""
    from adaptation.data import load_split

    out = {}
    for split in ("val", "test"):
        ss = load_split(data_dir, cohort, split)
        labs = {m["match_id"]: m for m in load_labels(label_dir, cohort, split)["matches"]}
        tab = pd.read_csv(root / cohort / f"{split}.csv.gz", dtype={"match_id": str}, float_precision="round_trip")  # exact float32 round trip
        tab = tab.set_index(["match_id", "team_id", "phase"])
        n_rows = n_missing = 0
        max_diff = 0.0
        for i in range(len(ss)):
            mid = str(ss.match_ids[ss.S_mi[i]])
            k = int(ss.S_cut[i])
            r0, r1 = ss.off[i], ss.off[i + 1]
            tid = labs[mid]["team_ids"][ss.TIDX[r0:r1].astype(np.int64)]
            idx = pd.MultiIndex.from_arrays([[mid] * len(tid), tid, [k] * len(tid)])
            sub = tab.reindex(idx)
            if sub["kf_n_kill"].isna().any():
                n_missing += int(sub["kf_n_kill"].isna().sum())
                continue
            if not (sub["alive_at_cut"].to_numpy() == 1).all():
                raise AssertionError(f"{mid} cut {k}: a K1 squad is not flagged alive_at_cut in the table")
            a_, b_ = sub[KF].to_numpy(dtype=np.float64), ss.raw_X[r0:r1].astype(np.float64)
            # equal up to float32 rounding: BLAS sums over a different number of cut times can differ in the last bit
            max_diff = max(max_diff, float((np.abs(a_ - b_) / (1.0 + np.abs(b_))).max()))
            n_rows += len(tid)
        out[split] = {"rows_compared": n_rows, "rows_missing_in_table": n_missing, "max_rel_diff": max_diff}
        if n_missing or max_diff > 2e-6:
            raise AssertionError(f"{cohort}/{split}: table differs from the KRAFTON-tf cut shards: {out[split]}")
    return out


def _init_worker():
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohorts", required=True)
    ap.add_argument("--splits", default="val,test,train")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--label-dir", default=str(E.DEFAULT_OUT), help="where adaptation.labels wrote the label tables")
    ap.add_argument("--out-dir", default=str(DEFAULT_ROOT))
    ap.add_argument("--limit-matches", type=int, default=0)
    ap.add_argument("--verify", action="store_true", help="compare the val / test tables with the KRAFTON-tf cut shards")
    ap.add_argument("--data-dir", default=str(E.DEFAULT_OUT))
    a = ap.parse_args(argv)
    if a.workers > 3:
        raise SystemExit("resource cap: at most 3 extraction workers")
    root = Path(a.out_dir)
    cohorts = [c for c in a.cohorts.split(",") if c]
    splits = [s for s in a.splits.split(",") if s]
    counts_path = root / "counts_partial.json"
    counts = json.loads(counts_path.read_text()) if counts_path.exists() else {}
    t0 = time.time()
    ctx = mp.get_context("spawn")
    for cohort in cohorts:
        for split in splits:
            if counts.get(cohort, {}).get(split) and (root / cohort / f"{split}.csv.gz").exists():
                print(f"skip {cohort}/{split} (done)", flush=True)
                continue
            labs_all = load_labels(a.label_dir, cohort, split)["matches"]
            labs = labs_all[: a.limit_matches] if a.limit_matches else labs_all
            jobs = [{"cohort": cohort, "split": split, "bidx": bi, "labs": labs[i : i + BATCH]} for bi, i in enumerate(range(0, len(labs), BATCH))]
            chunks, n_bad, risk_total = [], 0, 0
            with ctx.Pool(a.workers, initializer=_init_worker) as pool:
                for r in pool.imap_unordered(run_job, jobs):
                    for q in r["qa"]:
                        if not q["ok"]:
                            n_bad += 1
                            print("FAILED", q["match_id"], q.get("error", "")[-300:], flush=True)
                        else:
                            risk_total += q["n_risk"]
                    if r["arrays"] is not None:
                        chunks.append(r["arrays"])
            if n_bad:
                raise SystemExit(f"{cohort}/{split}: {n_bad} matches failed; table not written")
            df = to_frame(chunks)
            expect = int(sum(m["target"][: len(m["phase_end_s"]) - 1].sum() for m in labs))
            if int(df["in_risk_set"].sum()) != expect or risk_total != expect:
                raise AssertionError(f"{cohort}/{split}: {int(df['in_risk_set'].sum())} risk-set rows in the table, {expect} in the dump")
            write_table(df, root / cohort / f"{split}.csv.gz")
            counts.setdefault(cohort, {})[split] = {
                "rows": int(len(df)), "risk_set_rows": int(df["in_risk_set"].sum()), "alive_at_cut_rows": int(df["alive_at_cut"].sum()),
                "matches": int(df["match_id"].nunique()), "dump_matches": len(labs_all),
            }  # fmt: skip
            counts_path.parent.mkdir(parents=True, exist_ok=True)
            counts_path.write_text(json.dumps(counts, indent=1))
            print(f"[{time.time() - t0:7.1f}s] {cohort}/{split}: {counts[cohort][split]}", flush=True)
    if a.verify:
        for cohort in cohorts:
            if all((root / cohort / f"{sp}.csv.gz").exists() for sp in ("val", "test")):
                print("verify", cohort, verify(cohort, root, Path(a.data_dir), Path(a.label_dir)), flush=True)
    write_meta(root, counts, a)
    print("meta written", root / "meta.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
