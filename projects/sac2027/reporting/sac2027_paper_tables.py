#!/usr/bin/env python3
"""Source-linked manuscript tables; all values come from completed patch analyses."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT
import argparse
from pathlib import Path
import sys

REPO = REPO_ROOT
sys.path.insert(0, str(REPO))
import numpy as np
import pandas as pd

BASE = REPO / "reports/sac2027_patch_v14"
OUT = REPO / "paper/sac2027/v14_review/tables"


def main_table():
    primary = pd.read_csv(BASE / "primary_results_v2/arm_metrics_per_cohort.csv")
    metrics = pd.read_csv(BASE / "control_evaluation/main_population/arm_metrics.csv")
    native = pd.read_csv(BASE / "native_evaluation/per_cohort.csv")
    blocks = [
        [
            ("HGB-cut", "Ours, GBDT"),
            ("M11R", "Ours, PTR-Former"),
            ("KRAFTON-tf", "KRAFTON adaptation"),
            ("KRAFTON-tf-h0", "KRAFTON + phase baseline"),
            ("SAGONG-xgb", "Sagong adaptation (XGBoost)"),
        ],
        [
            ("KRAFTON-tf-ourfeat", "Weighted survival objective"),
            ("KRAFTON-tf-ourfeat-bce", "Next-phase cross-entropy"),
        ],
        [("HGB-union", "GBDT, ours + KRAFTON inputs")],
    ]
    columns = (
        "auc",
        "human_auc",
        "auc_within_phase",
        "auc_within_match_phase",
        "nll",
        "brier",
        "phase_calibration_error",
        "winner_accuracy",
        "c_index",
    )
    directions = np.array([1, 1, 1, 1, -1, -1, -1, 1, 1])
    records = []
    for block, arms in enumerate(blocks):
        for arm, label in arms:
            g = metrics[(metrics.arm == arm) & (metrics.population == "all")]
            human = metrics[(metrics.arm == arm) & (metrics.population == "human")]
            assert len(g) == len(human) == 11
            row = {m: float(g[m].mean()) for m in columns[:7] if m != "human_auc"}
            row["human_auc"] = float(human.auc.mean())
            # Independent main aggregator must agree on point estimates.
            for metric in ("auc", "auc_within_phase", "nll", "brier"):
                ref = primary[
                    (primary.arm == arm)
                    & (primary.seed == 42)
                    & (primary.stratum == "all")
                    & (primary.metric == metric)
                ]
                assert len(ref) == 11 and abs(ref.value.mean() - row[metric]) < 1e-10
            n = native[native.arm == arm]
            assert len(n) == 11 or arm == "SAGONG-xgb"
            row.update({m: float(n[m].mean()) if len(n) else np.nan for m in columns[-2:]})
            records.append(dict(block=block, arm=arm, label=label, **row))
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "main_results_values.csv", index=False)
    lines = [
        r"\begin{table*}[t]",
        r"\centering\footnotesize",
        r"\caption{Patch-41.2 performance: arithmetic means over 11 eligible cohorts. Human AUC restricts evaluation to human teams; other columns use all teams.}",
        r"\label{tab:main}",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrrrr@{}}",
        r"\toprule",
        r" & \multicolumn{4}{c}{Discrimination $\uparrow$} & \multicolumn{3}{c}{Probability accuracy $\downarrow$} & \multicolumn{2}{c}{Complementary task $\uparrow$} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}",
        r"Design & All & Human & Within & Within & Log loss & Brier & Phase & Winner & C-index \\",
        r" & AUC & AUC & phase & match & & & cal. error & accuracy & \\",
        r"\midrule",
    ]
    for block, g in frame.groupby("block", sort=False):
        if block:
            title = (
                "KRAFTON backbone, our inputs (reference recipes)"
                if block == 1
                else "Both input sets combined"
            )
            lines.extend([r"\midrule", r"\multicolumn{10}{l}{\emph{" + title + r"}} \\"])
        for _, row in g.iterrows():
            cells = []
            for metric, direction in zip(columns, directions):
                value = row[metric]
                best = round(float((g[metric] * direction).max()), 3)
                cell = "---" if pd.isna(value) else f"{value:.3f}"
                if len(g) > 1 and not pd.isna(value) and round(value * direction, 3) == best:
                    cell = r"\textbf{" + cell + "}"
                cells.append(cell)
            lines.append(row.label + " & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular*}",
            r"\smallskip",
            r"\parbox{\textwidth}{\scriptsize Bold identifies the best rounded value within each comparison block. Within phase and within match denote AUC. Reference recipes differ in stopping and intercept initialization; matched objective controls are reported separately. Winner accuracy and C-index use the survival-oriented ranking output (PTR: placement-ranking head).}",
            r"\end{table*}",
        ]
    )
    (OUT / "main_results.tex").write_text("\n".join(lines) + "\n")


def tuning_table():
    frame = pd.read_csv(BASE / "control_evaluation/tuning/arm_metrics.csv")
    lines = [
        r"\begin{table}[t]",
        r"\centering\footnotesize",
        r"\caption{Matched-input, equal-trial tuning. Each value averages three selected-model seeds; all use the BCE target.}",
        r"\label{tab:tuning}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & All AUC & Human & Within & Log loss \\",
        r" & & AUC & match & \\",
        r"\midrule",
    ]
    for cohort, name in [
        ("erangel_tpp_squad", "Erangel TPP squad"),
        ("taego_tpp_squad", "Taego TPP squad"),
    ]:
        lines.append(r"\multicolumn{5}{l}{\emph{" + name + r"}} \\")
        for arm, label in [
            ("gbdt", "GBDT"),
            ("krafton", "KRAFTON-derived"),
            ("ptr", "PTR (no rank loss)"),
        ]:
            g = frame[(frame.cohort == cohort) & (frame.arm == arm)].set_index("population")
            assert len(g) == 2
            vals = [
                g.loc["all", "auc"],
                g.loc["human", "auc"],
                g.loc["all", "auc_within_match_phase"],
                g.loc["all", "nll"],
            ]
            lines.append(label + " & " + " & ".join(f"{v:.3f}" for v in vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUT / "tuning_results.tex").write_text("\n".join(lines) + "\n")


def objective_table():
    f = pd.read_csv(BASE / "control_evaluation/objective/arm_metrics.csv")
    f = f[f.population == "all"].set_index(["cohort", "arm"])
    labels = [
        ("weighted_full", "Weighted, full"),
        ("weighted_truncated", "Weighted, next phase"),
        ("cox_full", "Cox, full"),
        ("cox_truncated", "Cox, next phase"),
        ("bce", "BCE, next phase"),
        ("cox_truncated_bce", "Cox + BCE, next phase"),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering\footnotesize",
        r"\caption{Matched objective and horizon controls. AUCs average three seeds after a common phase-aware validation mapping.}",
        r"\label{tab:objective}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Erangel} & \multicolumn{2}{c}{Taego} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Loss and horizon & Within & Within & Within & Within \\",
        r" & phase & match & phase & match \\",
        r"\midrule",
    ]
    for arm, label in labels:
        values = [
            f.loc[(c, arm), m]
            for c in ("erangel_tpp_squad", "taego_tpp_squad")
            for m in ("auc_within_phase", "auc_within_match_phase")
        ]
        lines.append(label + " & " + " & ".join(f"{v:.3f}" for v in values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUT / "objective_results.tex").write_text("\n".join(lines) + "\n")


def steps_table():
    f = pd.read_csv(BASE / "primary_results_v2/contrasts_pooled.csv")
    f = f[f.stratum == "all"].set_index(["contrast", "metric"])
    w = pd.read_csv(BASE / "primary_results_v2/supplement/e2_contrasts_pooled_dl.csv").set_index(
        "contrast"
    )
    names = [
        ("hgbcut_vs_sagong_hgb", r"(a) Sagong inputs $\to$ ours"),
        ("hgbcut_vs_hgb_kraftonfeat", r"(b) KRAFTON inputs $\to$ ours"),
        ("k2h0_vs_k2", r"(c1) + phase baseline"),
        ("k3a_vs_k2h0", r"(c2) $\to$ next-phase BCE recipe"),
        ("k3_vs_k3a", r"(d) + explicit phase intercepts"),
        ("m01norank_vs_k3", r"(e) $\to$ team-axis PTR"),
        ("m11rnorank_vs_m01norank", r"(f) + phase axis, relation bias"),
        ("hgb_union_vs_hgbcut", r"(g) + KRAFTON inputs (union)"),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering\footnotesize",
        r"\caption{Conditional reference-path changes in AUC, pooled across 11 cohorts.}",
        r"\label{tab:steps}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Step & Overall & Within phase & Within match \\",
        r"\midrule",
    ]
    for key, label in names:
        cells = []
        for metric in ("auc", "auc_within_phase", "auc_within_match_phase"):
            if key == "k2h0_vs_k2" and metric != "auc":
                cells.append(r"$0^{\dagger}$")
                continue
            row = w.loc[key] if metric == "auc_within_match_phase" else f.loc[(key, metric)]
            v = row.theta
            cell = f"{v:+.3f}" if abs(v) >= 0.0005 else f"{v:+.4f}"
            if abs(v) >= 0.005 and (row.ci95_low > 0 or row.ci95_high < 0):
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\smallskip",
            r"\parbox{\columnwidth}{\scriptsize Random-effects pooled estimates. Bold: 95\% CI excludes 0 and $|\Delta|\geq0.005$. $\dagger$Zero by monotonicity within a phase. Steps (a), (b), (g) use GBDT. Other steps change reference recipes and may bundle stopping, initialization, or baseline-fit split; matched controls follow.}",
            r"\end{table}",
        ]
    )
    (OUT / "step_results.tex").write_text("\n".join(lines) + "\n")


def inputs_table():
    """Input rows (a), (b), (g) of the reference path: GBDT fixed, so only the input set changes."""
    f = pd.read_csv(BASE / "primary_results_v2/contrasts_pooled.csv")
    f = f[f.stratum == "all"].set_index(["contrast", "metric"])
    w = pd.read_csv(BASE / "primary_results_v2/supplement/e2_contrasts_pooled_dl.csv").set_index(
        "contrast"
    )
    names = [
        ("hgbcut_vs_sagong_hgb", r"Sagong inputs $\to$ ours"),
        ("hgbcut_vs_hgb_kraftonfeat", r"KRAFTON inputs $\to$ ours"),
        ("hgb_union_vs_hgbcut", r"Ours $\to$ ours + KRAFTON inputs"),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering\footnotesize",
        r"\caption{Input contrasts with GBDT fixed: AUC changes pooled across 11 cohorts.}",
        r"\label{tab:inputs}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Input change & Overall & Within phase & Within match \\",
        r"\midrule",
    ]
    for key, label in names:
        cells = []
        for metric in ("auc", "auc_within_phase", "auc_within_match_phase"):
            row = w.loc[key] if metric == "auc_within_match_phase" else f.loc[(key, metric)]
            v = row.theta
            cell = f"{v:+.3f}"
            if abs(v) >= 0.005 and (row.ci95_low > 0 or row.ci95_high < 0):
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\smallskip",
            r"\parbox{\columnwidth}{\scriptsize Random-effects pooled estimates. Bold: 95\% CI excludes 0 and $|\Delta|\geq0.005$.}",
            r"\end{table}",
        ]
    )
    (OUT / "input_results.tex").write_text("\n".join(lines) + "\n")


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--table", choices=("main", "tuning", "objective", "steps", "inputs", "all"), default="all"
    )
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    OUT = a.out
    OUT.mkdir(parents=True, exist_ok=True)
    if a.table in ("main", "all"):
        main_table()
    if a.table in ("tuning", "all"):
        tuning_table()
    if a.table in ("objective", "all"):
        objective_table()
    if a.table in ("steps", "all"):
        steps_table()
    if a.table in ("inputs", "all"):
        inputs_table()


if __name__ == "__main__":
    main()
