"""Summary table of the KRAFTON baseline runs (KRAFTON-tf, KRAFTON-mlp, KRAFTON-tf-ourfeat) next to the PTR reference arms.

    python -m adaptation.summarize [--seed 42] [--out reports/krafton_baseline_v2/summary_seed42.csv]

Reads ``<ckpt>/<cohort>/KRAFTON-tf|KRAFTON-mlp|KRAFTON-tf-ourfeat/seed<seed>/metrics.json`` and ``reports/
bigcomp2027_results_v2/arm_metrics_per_cohort.csv`` (HGB-cut, M01, M11R, seed 42, stratum all).  Prints one row per
cohort; facts only.

Both defaults are the v2 roots of ``model_stgcn.ptr.paths``.  The run directory (``--ckpt-dir``) and the directory that holds the reference table
(``--ref``, the aggregator's results root) must come from the same dump version (``paths.artifact_version``: stamp, run metrics, summary.json,
``_v<N>`` in the name): a v1 table next to v2 runs (or the reverse), or a source whose version cannot be told next to one that can, is refused
before anything is read (:func:`check_sources`).
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

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = REPO_ROOT / "third_party/krafton_pubg_winprob"
REPO = REPO_ROOT
for _p in (str(ROOT), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pubg_lab.survival.ptr.data.paths as P  # noqa: E402  (the v2 roots and the version guard)

COHORTS = ["erangel_tpp_squad", "taego_tpp_squad", "miramar_desert_tpp_squad", "rondo_tpp_squad", "sanhok_tpp_squad", "erangel_tpp_solo", "erangel_tpp_duo",
           "taego_tpp_duo", "taego_fpp_squad", "erangel_fpp_squad", "karakin_tpp_squad", "paramo_tpp_squad"]  # fmt: skip


def check_sources(ckpt_dir: Path, ref: Path) -> str | None:
    """Refuse to pair a v1 source with a v2 source; returns the common version (None when neither source says).

    The sources are the run directory ``ckpt_dir`` and the directory that holds the reference table ``ref`` (the aggregator's results root).
    Their versions come from ``paths.artifact_version`` (stamp, run metrics, summary.json's ``runs_dir``, ``_v<N>`` in the name).  A mismatch is
    refused, and so is a source whose version cannot be told next to one that can (it may be a v1 root that was renamed or copied).
    Raises ``paths.VersionMixError``."""
    ckpt_dir, ref = (
        Path(os.path.abspath(ckpt_dir)),
        Path(os.path.abspath(ref)),
    )  # absolute: a bare file name or '.' has no directory name to read a version from
    (cv, c_why), (rv, r_why) = P.artifact_version(ckpt_dir), P.artifact_version(ref.parent)
    if cv == rv:
        return cv
    say = (
        lambda role, path, v, why: f"{role} {path} is {v} ({why})"
        if v
        else f"{role} {path} carries no version evidence"
    )  # noqa: E731
    raise P.VersionMixError(
        f"refusing to pair {say('--ckpt-dir', ckpt_dir, cv, c_why)} with {say('--ref', ref, rv, r_why)}: runs and reference table must come from the "
        f"same dump (v2 roots: {P.RUNS_DIR}, {P.RESULTS_DIR}; a root without evidence can be stamped with "
        "`python -m model_stgcn.ptr.paths stamp <root> --dump-dir <dump>`)"
    )


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--ckpt-dir",
        type=Path,
        default=P.RUNS_DIR,
        help="run directory with the KRAFTON arms (default: the v2 root)",
    )
    ap.add_argument("--ref", type=Path, default=P.RESULTS_DIR / "arm_metrics_per_cohort.csv",
                    help="PTR reference table of the aggregator (default: the v2 results root); must be of the same dump version as --ckpt-dir")  # fmt: skip
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--md", action="store_true", help="print a markdown table")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    try:
        check_sources(a.ckpt_dir, a.ref)
    except P.VersionMixError as exc:
        raise SystemExit(f"refusing to run: {exc}") from exc
    ref = pd.read_csv(a.ref)
    ref = ref[(ref.seed == 42) & (ref.stratum == "all")]
    rows = []
    for c in COHORTS:
        row = {"cohort": c}
        for arm, key in (
            ("KRAFTON-tf", "tf"),
            ("KRAFTON-mlp", "mlp"),
            ("KRAFTON-tf-ourfeat", "k2"),
            ("HGB-kraftonfeat", "hkf"),
        ):
            p = a.ckpt_dir / c / arm / f"seed{a.seed}" / "metrics.json"
            if not p.is_file():
                continue
            m = json.loads(p.read_text())
            t = m["test"]
            nat = (m.get("krafton_native") or {}).get("test", {}).get("all", {})
            row.update({
                f"{key}_auc": t["hazard"]["auc"], f"{key}_ap": t["hazard"]["ap"], f"{key}_nll": t["hazard"]["nll"], f"{key}_brier": t["hazard"]["brier"],
                f"{key}_spearman": t["rank"]["spearman"], f"{key}_top5_auc": t["rank"]["top5_auc"], f"{key}_cindex": t["rank"]["cindex"],
                f"{key}_win_acc": nat.get("winner_accuracy"), f"{key}_win_acc_chance": nat.get("winner_accuracy_chance"), f"{key}_native_cindex": nat.get("c_index"),
                f"{key}_epochs": m.get("epochs_run"), f"{key}_best_epoch": m.get("best_epoch"), f"{key}_stopped_early": m.get("stopped_early"),
                f"{key}_train_sec": m.get("train_sec", m.get("fit_sec")), f"{key}_total_sec": m.get("elapsed_sec"),
                f"{key}_platt_a": (m.get("baseline") or {}).get("platt", {}).get("a"),
            })  # fmt: skip
            row["n_test_rows"] = t["hazard"]["n"]
        for arm in ("HGB-cut", "M01", "M11R"):
            for metric in ("auc", "spearman"):
                r = ref[(ref.cohort == c) & (ref.arm == arm) & (ref.metric == metric)]
                if len(r):
                    row[f"{arm.replace('-', '_')}_{metric}"] = float(r.value.iloc[0])
        rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    cols = [
        "cohort",
        "n_test_rows",
        "tf_auc",
        "mlp_auc",
        "k2_auc",
        "hkf_auc",
        "M01_auc",
        "HGB_cut_auc",
        "M11R_auc",
        "tf_epochs",
        "mlp_epochs",
        "k2_epochs",
        "tf_total_sec",
        "mlp_total_sec",
        "k2_total_sec",
        "hkf_total_sec",
    ]
    print(df[[c for c in cols if c in df.columns]].round(4).to_string(index=False))
    if a.md:
        f = lambda v: "-" if pd.isna(v) else f"{v:.3f}"  # noqa: E731
        print(
            "| cohort | test rows | KRAFTON-tf | KRAFTON-mlp | KRAFTON-tf-ourfeat | HGB-kraftonfeat | M01 | HGB-cut | M11R |"
        )
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, r in df.iterrows():
            g = lambda k: r.get(k, float("nan"))  # noqa: E731
            n = g("n_test_rows")
            print(
                f"| {r['cohort']} | {int(n) if not pd.isna(n) else '-'} | {f(g('tf_auc'))} | {f(g('mlp_auc'))} | {f(g('k2_auc'))} | {f(g('hkf_auc'))} | {f(g('M01_auc'))} | {f(g('HGB_cut_auc'))} | {f(g('M11R_auc'))} |"
            )
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out, index=False, float_format="%.9g")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
