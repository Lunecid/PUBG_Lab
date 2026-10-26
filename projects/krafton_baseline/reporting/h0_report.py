"""Tables and paired contrasts of the H0 arms (``adaptation.h0_baseline``) next to the Cox, hazard and PTR arms (seed 42, test split).

    python -m adaptation.h0_report [--n-boot 2000] [--boot-seed 0] [--ckpt-dir model_stgcn/checkpoints/bigcomp2027_ptr_v2]
        [--dump-dir model_stgcn/dumps/bigcomp2027_phase_v2] [--out-dir reports/krafton_baseline_v2] [--cohorts a,b,...]

The run directory (``--ckpt-dir``, where ``adaptation.h0_baseline`` wrote the H0 arms) and the output directory are explicit (the defaults are the v2
roots of ``model_stgcn.ptr.paths``); ``--dump-dir`` (optional) gives the cohorts of ``all`` and its meta.version is compared with the other two: a v1 root
next to a v2 dump (or the reverse) is refused before anything is read (without it ``--ckpt-dir`` and ``--out-dir`` are compared with each other).

* ``h0_arm_metrics.csv``   one row per cohort and arm: pooled AUC, within-phase AUC (phases with both classes and >= 30 rows, unweighted
                           mean), NLL, Brier (``metrics.hazard_metrics``, clip 1e-7), per-phase calibration gap (row-weighted mean over the
                           cut phases of |mean p - observed rate|, as ``phase_auc.py``); pooled AUC / NLL / Brier are asserted equal to the
                           arm's ``metrics.json``.  Prints the 12-cohort means.
* ``h0_contrasts.csv``     paired match-level bootstrap contrasts A - B (the SAME 2,000 resamples of whole test matches for every arm, as
                           ``k3_contrasts.py``), per cohort and DerSimonian-Laird pooled over the 12 cohorts, pooled and within-phase AUC;
                           ``n_same_sign`` = cohorts whose point difference has the sign of the pooled estimate.
* ``h0_fit_summary.csv``   one row per cohort and H0 arm: fit split, rows, beta, gamma_1..7, log-likelihood, and the assertions of ``metrics.json``.
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
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = REPO_ROOT / "third_party/krafton_pubg_winprob"
REPO = REPO_ROOT
for p in (str(ROOT), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from projects.krafton_baseline.evaluation.k3_contrasts import MIN_ROWS, summarise  # noqa: E402
from projects.krafton_baseline.evaluation.phase_auc import COHORTS  # noqa: E402
import pubg_lab.survival.ptr.evaluation.metrics as M  # noqa: E402  (read-only use)
import pubg_lab.survival.ptr.data.paths as P  # noqa: E402  (the v2 roots and the version guard)

CKPT = P.RUNS_DIR  # default of --ckpt-dir; every function below takes the run directory explicitly
ARMS = {  # short name -> arm directory
    "K2": "KRAFTON-tf-ourfeat", "KR-Cox+H0": "KRAFTON-tf-ourfeat-h0", "KR-Cox+H0-slope": "KRAFTON-tf-ourfeat-h0-slope",
    "KR-Cox+H0val": "KRAFTON-tf-ourfeat-h0val", "KR-Cox+H0val-slope": "KRAFTON-tf-ourfeat-h0val-slope",
    "K3a": "KRAFTON-tf-ourfeat-bce", "K3": "KRAFTON-tf-ourfeat-hazard",
    "KRAFTON-tf": "KRAFTON-tf", "KRAFTON+H0": "KRAFTON-tf-h0", "KRAFTON+H0-slope": "KRAFTON-tf-h0-slope",
    "M01": "M01", "M11R": "M11R", "HGB-cut": "HGB-cut",
}  # fmt: skip
TABLE_ORDER = ["K2", "KR-Cox+H0", "K3a", "K3", "KRAFTON-tf", "KRAFTON+H0", "M01", "M11R", "HGB-cut"]
EXTRA_ORDER = ["KR-Cox+H0-slope", "KRAFTON+H0-slope", "KR-Cox+H0val", "KR-Cox+H0val-slope"]
# (A, B) contrasts A - B; the first five are the requested ones, then the slope variants, then diagnostics
CONTRASTS = [
    ("K3a", "KR-Cox+H0"), ("KR-Cox+H0", "K2"), ("M01", "KR-Cox+H0"), ("M11R", "KRAFTON+H0"), ("HGB-cut", "KRAFTON+H0"),
    ("K3a", "KR-Cox+H0-slope"), ("KR-Cox+H0-slope", "K2"), ("M01", "KR-Cox+H0-slope"), ("M11R", "KRAFTON+H0-slope"), ("HGB-cut", "KRAFTON+H0-slope"),
    ("KRAFTON+H0", "KRAFTON-tf"), ("KR-Cox+H0", "KR-Cox+H0val"), ("KR-Cox+H0val", "KRAFTON+H0"), ("KR-Cox+H0val-slope", "KRAFTON+H0-slope"),
]  # fmt: skip


def read_arm(cohort: str, arm: str, split: str = "test", ckpt: Path = CKPT) -> pd.DataFrame:
    return M.read_predictions(Path(ckpt) / cohort / ARMS[arm] / "seed42" / f"pred_{split}.csv")


def arm_row(cohort: str, short: str, ckpt: Path = CKPT) -> dict:
    df = read_arm(cohort, short, "test", ckpt)
    y, pr = df["y"].to_numpy(), df["p_hazard"].to_numpy()
    h = M.hazard_metrics(y, pr)
    ref = json.loads((Path(ckpt) / cohort / ARMS[short] / "seed42" / "metrics.json").read_text())[
        "test"
    ]["hazard"]
    for k in ("auc", "nll", "brier"):
        if abs(h[k] - ref[k]) > 1e-5:
            raise AssertionError(
                f"{cohort}/{short}: {k} {h[k]:.7f} from the CSV differs from metrics.json {ref[k]:.7f}"
            )
    wp = M.WithinPhaseAUC(y, pr, df["phase"].to_numpy(), MIN_ROWS)
    gaps = [(abs(g["p_hazard"].mean() - g["y"].mean()), len(g)) for _k, g in df.groupby("phase")]
    gap = sum(d * n for d, n in gaps) / len(df)
    return {"cohort": cohort, "arm": short, "dir": ARMS[short], "n_rows": len(df), "pooled": h["auc"], "within_phase": wp.mean(np.ones(len(df))), "n_phases": len(wp.phases),
            "nll": h["nll"], "brier": h["brier"], "cal_gap": float(gap), "cal_gap_unweighted": float(np.mean([d for d, _n in gaps]))}  # fmt: skip


def cohort_draws(
    cohort: str, arms: list[str], n_boot: int, boot_seed: int, ckpt: Path = CKPT
) -> dict:
    """Pooled and within-phase AUC of every arm on the same bootstrap resamples of whole test matches (k3_contrasts.cohort_draws, many arms)."""
    frames = {a: read_arm(cohort, a, "test", ckpt) for a in arms}
    base = frames[arms[0]]
    for df in frames.values():
        M.align_predictions(base, df)
    match_ids, codes = np.unique(base["match_id"].to_numpy(), return_inverse=True)
    counts = M.resample_counts(len(match_ids), n_boot, boot_seed)
    y, phase = base["y"].to_numpy(), base["phase"].to_numpy()
    phases = [
        k
        for k in sorted(set(phase.tolist()))
        if (phase == k).sum() >= MIN_ROWS and len(np.unique(y[phase == k])) == 2
    ]
    rows_k = {k: np.flatnonzero(phase == k) for k in phases}
    eng = {
        a: (
            M.SortedBinary(y, df["p_hazard"].to_numpy()),
            {k: M.SortedBinary(y[r], df["p_hazard"].to_numpy()[r]) for k, r in rows_k.items()},
        )
        for a, df in frames.items()
    }
    full = np.ones(len(y))
    out = {a: {"pooled": [eng[a][0].auc(full), np.empty(n_boot)], "within_phase": [float(np.mean([eng[a][1][k].auc(full[rows_k[k]]) for k in phases])), np.empty(n_boot)]}
           for a in arms}  # fmt: skip
    for b in range(n_boot):
        w = counts[b][codes].astype(np.float64)
        for a in arms:
            pooled, per = eng[a]
            out[a]["pooled"][1][b] = pooled.auc(w)
            out[a]["within_phase"][1][b] = np.mean([per[k].auc(w[rows_k[k]]) for k in phases])
    return out


def fit_rows(ckpt: Path = CKPT, cohorts: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for c in COHORTS if cohorts is None else cohorts:
        for short in (
            "KR-Cox+H0",
            "KR-Cox+H0-slope",
            "KRAFTON+H0",
            "KRAFTON+H0-slope",
            "KR-Cox+H0val",
            "KR-Cox+H0val-slope",
        ):
            m = json.loads((Path(ckpt) / c / ARMS[short] / "seed42" / "metrics.json").read_text())
            f, ck = m["baseline"]["fit"], m["checks"]
            row = {"cohort": c, "arm": short, "dir": ARMS[short], "parent": m["parent_arm"], "fit_split": m["baseline"]["fit_split"], "n_fit_rows": f["n_rows"], "n_events": f["n_events"],
                   "beta": f["beta"], "beta_se": f["beta_se"], "loglik": f["loglik"], "converged": f["converged"], "n_iter": f["n_iter"]}  # fmt: skip
            row.update({f"gamma_{k}": v for k, v in f["gamma"].items()})
            ori = ck["orientation_on_fit_rows"]
            row.update(
                {"corr_s_y_fit": ori["corr_pooled"], "corr_s_y_min_phase": ori["corr_min_phase"]}
            )
            rp = ck["raw_score_reproduced_by_inference_max_abs_diff"]
            row.update({"repro_max_abs_diff_val": rp["val"], "repro_max_abs_diff_test": rp["test"]})
            for sp in ("val", "test"):
                a = ck["within_phase_auc_vs_parent"][sp]
                row.update(
                    {
                        f"phase_auc_max_abs_diff_{sp}": a["vs_parent_files"]["max_abs_diff"],
                        f"monotone_{sp}": a["monotone_within_phase"],
                        f"rows_p_tied_at_one_{sp}": a["n_rows_p_tied_at_one"],
                    }
                )
            row.update({"rows_match": ",".join(ck["row_assertion"]["test"]["matches_arms"])})
            rows.append(row)
    return pd.DataFrame(rows)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument(
        "--ckpt-dir",
        type=Path,
        default=CKPT,
        help="run directory with the K1 / K2 / H0 arms (default: the v2 root)",
    )
    ap.add_argument(
        "--dump-dir",
        type=Path,
        default=None,
        help="phase dump directory: its meta.version is compared with --ckpt-dir / --out-dir and its cohorts "
        "select 'all' (v2: model_stgcn/dumps/bigcomp2027_phase_v2; without it --ckpt-dir and --out-dir are compared with each other)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=P.KRAFTON_REPORTS_DIR,
        help="report directory (default: the v2 root)",
    )
    ap.add_argument(
        "--cohorts",
        default="all",
        help="comma separated; 'all' = the 12 cohorts of phase_auc.COHORTS that the dump directory has",
    )
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    try:
        if a.dump_dir is not None:
            P.check_same_version(
                a.dump_dir, {"ckpt (--ckpt-dir)": a.ckpt_dir, "reports (--out-dir)": a.out_dir}
            )
        else:
            cv, ov = P.artifact_version(a.ckpt_dir)[0], P.artifact_version(a.out_dir)[0]
            if cv and ov and cv != ov:
                raise P.VersionMixError(
                    f"--ckpt-dir {a.ckpt_dir} is {cv} but --out-dir {a.out_dir} is {ov}"
                )
    except P.VersionMixError as exc:
        raise SystemExit(f"refusing to run: {exc}") from exc
    ckpt = Path(a.ckpt_dir)
    if a.cohorts == "all":
        cohorts = list(COHORTS)
        if a.dump_dir is not None and Path(a.dump_dir).is_dir():
            from pubg_lab.survival.ptr.data.cohorts import discover_cohorts

            have = set(discover_cohorts(a.dump_dir))
            cohorts = (
                [c for c in COHORTS if c in have] + sorted(have - set(COHORTS)) if have else cohorts
            )
    else:
        cohorts = [c for c in a.cohorts.split(",") if c]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    # ---- arm metrics
    arms_all = TABLE_ORDER + EXTRA_ORDER
    tab = pd.DataFrame([arm_row(c, s, ckpt) for c in cohorts for s in arms_all])
    tab.to_csv(a.out_dir / "h0_arm_metrics.csv", index=False, float_format="%.9g")
    mean = (
        tab.groupby("arm")[
            ["pooled", "within_phase", "nll", "brier", "cal_gap", "cal_gap_unweighted"]
        ]
        .mean()
        .reindex(arms_all)
    )
    print(f"\n{len(cohorts)}-cohort means (test, seed 42)\n")
    print(mean.round(4).to_string())
    print("\nper cohort pooled AUC / NLL / calibration gap of the H0 arms\n")
    print(
        tab[tab.arm.isin(["KR-Cox+H0", "KRAFTON+H0", "K2", "KRAFTON-tf"])]
        .pivot(index="cohort", columns="arm", values=["pooled", "nll", "cal_gap"])
        .reindex(cohorts)
        .round(4)
        .to_string()
    )

    for child, parent in (
        ("KR-Cox+H0", "K2"),
        ("KR-Cox+H0-slope", "K2"),
        ("KR-Cox+H0val", "K2"),
        ("KRAFTON+H0", "KRAFTON-tf"),
        ("KRAFTON+H0-slope", "KRAFTON-tf"),
    ):
        w = tab.pivot(index="cohort", columns="arm", values="within_phase")
        print(
            f"max |within-phase AUC {child} - {parent}| over the {len(cohorts)} cohorts: {(w[child] - w[parent]).abs().max():.2e}"
        )

    # ---- fit summary + checks
    fs = fit_rows(ckpt, cohorts)
    fs.to_csv(a.out_dir / "h0_fit_summary.csv", index=False, float_format="%.9g")
    print(
        "\nchecks over",
        len(fs),
        "fits: max raw_score reproduction diff %.2e, max per-phase AUC diff %.2e, monotone all %s, orientation corr min %.3f"
        % (
            max(fs.repro_max_abs_diff_val.max(), fs.repro_max_abs_diff_test.max()),
            max(fs.phase_auc_max_abs_diff_val.max(), fs.phase_auc_max_abs_diff_test.max()),
            bool(fs.monotone_val.all() and fs.monotone_test.all()),
            fs.corr_s_y_fit.min(),
        ),
    )
    print(
        "beta of the slope fits (mean over cohorts):",
        fs[fs.arm.str.endswith("slope")]
        .groupby("arm")["beta"]
        .agg(["mean", "min", "max"])
        .round(3)
        .to_dict("index"),
    )

    # ---- paired contrasts
    need = sorted({x for p in CONTRASTS for x in p}, key=arms_all.index)
    rows = []
    for c in cohorts:
        r = cohort_draws(c, need, a.n_boot, a.boot_seed, ckpt)
        for x, y in CONTRASTS:
            for metric in ("pooled", "within_phase"):
                est = r[x][metric][0] - r[y][metric][0]
                rows.append(
                    {
                        "cohort": c,
                        "contrast": f"{x} - {y}",
                        "metric": metric,
                        "level": "cohort",
                        **summarise(est, r[x][metric][1] - r[y][metric][1]),
                    }
                )
        print("bootstrapped", c, flush=True)
    cohort_df = pd.DataFrame(rows)
    pooled_rows = []
    for (contrast, metric), g in cohort_df.groupby(["contrast", "metric"], sort=False):
        if (
            g["estimate"].abs().max() < 1e-4 and g["se"].max() < 1e-4
        ):  # the two arms rank identically (monotone map within a phase): nothing to pool
            theta, se_re, ci, tau2, p1, k = 0.0, 0.0, (0.0, 0.0), 0.0, float("nan"), 0
            note = f"0 by construction (monotone map within a phase); max |cohort difference| {g['estimate'].abs().max():.1e} from p rounded to 1 in the files"
        else:
            re = M.dersimonian_laird(g["estimate"].to_numpy(), g["se"].to_numpy())
            theta, se_re, ci, tau2, p1, k, note = (
                re["theta"],
                re["se"],
                re["ci95"],
                re["tau2"],
                re["p_one_sided"],
                re["k"],
                "",
            )
        same = int((np.sign(g["estimate"]) == np.sign(theta)).sum()) if abs(theta) > 0 else -1
        pooled_rows.append({"cohort": f"RE pooled (k={k})", "contrast": contrast, "metric": metric, "level": "pooled", "estimate": theta, "se": se_re, "ci_low": ci[0],
                            "ci_high": ci[1], "tau2": tau2, "p_one_sided": p1, "mean_unweighted": float(g["estimate"].mean()),
                            "n_cohorts_positive": int((g["estimate"] > 0).sum()), "n_same_sign": same, "n_cohorts": int(len(g)), "note": note})  # fmt: skip
    out = pd.concat([cohort_df, pd.DataFrame(pooled_rows)], ignore_index=True)
    out.to_csv(a.out_dir / "h0_contrasts.csv", index=False, float_format="%.6g")
    pr = pd.DataFrame(pooled_rows)
    print("\nDL-pooled paired contrasts A - B (2,000 paired match bootstraps per cohort)\n")
    for _i, r in pr.iterrows():
        print(
            f"{r['contrast']:<40s} {r['metric']:<13s} {r['estimate']:+.4f} ({r['ci_low']:+.4f}, {r['ci_high']:+.4f})  same sign {r['n_same_sign'] if r['n_same_sign'] >= 0 else 'n/a'}/{r['n_cohorts']}  positive {r['n_cohorts_positive']}/{r['n_cohorts']}  tau2 {r['tau2']:.2e}  {r['note']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
