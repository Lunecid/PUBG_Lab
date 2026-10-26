"""The results aggregator (design doc 4.1 / 5): per-cohort paired bootstrap, DerSimonian-Laird pooling, Holm on
P1-P5 only, verdicts, kappa gate, seed SDs, strata, tidy CSVs, LaTeX tables and the forest plot.

All inputs are synthetic predictions with KNOWN effects (``synthetic.write_prediction_runs``): arm k has hazard
AUC ``Phi(d'_k / sqrt(2))`` by construction, so the pooled estimates can be compared with the truth."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from scipy.stats import t as t_dist

import pubg_lab.survival.ptr.evaluation.aggregate as A
import pubg_lab.survival.ptr.evaluation.metrics as M
from pubg_lab.survival.ptr.data.synthetic import write_prediction_runs

BASE = 0.9
# latent hazard separation d' per arm.  Truth: phase axis +0.4, team axis +0 (hazard), relation +0.2 over its
# shuffle, Hab costs 0.4, the zone object recovers 0.2 of it.
QUALITY = {
    "M00": 0.9, "M10": 1.3, "M01": 0.9, "M11": 1.3, "M11R": 1.5, "M11R-shuf": 1.3, "M11RZ": 1.6,
    "M11R-noHab": 1.1, "M11RZ-noHab": 1.3, "M11R-noProx": 1.45, "M11R-noCombat": 1.42, "M11R-noCopress": 1.49,
    "HGB-cut": 0.8, "HGB-lag": 0.95, "HGB-noHab": 0.7,
}  # fmt: skip
# the team axis helps the relative standing (rank head), not the hazard: M01 / M11 reuse the hazard scores of M00 / M10
RANK_QUALITY = {**QUALITY, "M01": 1.8, "M11": 2.2, "M10": 0.9, "M11R": 2.2, "M11RZ": 2.3}
NEAR_COPY = {"M01": "M00", "M11": "M10"}
COHORTS = ["c_alpha", "c_beta", "c_gamma", "c_delta"]
SCALE = {"c_alpha": 1.0, "c_beta": 1.3, "c_gamma": 0.7, "c_delta": 1.0}  # heterogeneity of the effects
EXPLORATORY = ("M11R-noProx", "M11R-noCombat", "M11R-noCopress")
SKIP_EXPLORATORY = {(c, a) for c in ("c_gamma", "c_delta") for a in EXPLORATORY}  # the two "large" cohorts only


def auc_theory(d):
    return float(norm.cdf(d / np.sqrt(2.0)))


def eff(cohort, arm, quality=QUALITY):
    return BASE + SCALE[cohort] * (quality[arm] - BASE)


def make_runs(root, **kw):
    args = dict(
        cohorts=COHORTS, quality=QUALITY, rank_quality=RANK_QUALITY, n_matches=60, seed=3, cohort_scale=SCALE,
        seeds={"c_alpha": (13, 2026), "c_beta": (13, 2026)}, near_copy=NEAR_COPY, skip=SKIP_EXPLORATORY,
    )  # fmt: skip
    args.update(kw)
    return write_prediction_runs(root, args.pop("cohorts"), args.pop("quality"), args.pop("rank_quality"), **args)


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    root, out = tmp_path_factory.mktemp("runs"), tmp_path_factory.mktemp("results")
    make_runs(root)
    res = A.aggregate(root, out, n_boot=150, seed=5, workers=2)
    return res, root, out


def pooled(res, formula, metric, stratum="all"):
    rows = res.pooled[(res.pooled["contrast"] == formula) & (res.pooled["metric"] == metric) & (res.pooled["stratum"] == stratum)]
    assert len(rows) == 1, (formula, metric, stratum, len(rows))
    return rows.iloc[0]


# --------------------------------------------------------------------------- pure helpers
def test_constants_follow_the_design_doc():
    assert A.SESOI == {"auc": 0.005, "spearman": 0.01, "top5_auc": 0.005, "cindex": 0.005}
    assert A.PRIMARY == {
        "P1": ("phase_main", "auc"), "P2": ("team_main", "auc"), "P3": ("team_main", "spearman"),
        "P4": ("rel_vs_shuf", "auc"), "P5": ("zone_nohab", "auc"),
    }  # fmt: skip
    assert A.N_BOOT == 2000 and A.ALPHA == 0.05
    f = A.FORMULAS
    assert f["phase_main"].coefs == {"M10": 0.5, "M00": -0.5, "M11": 0.5, "M01": -0.5}
    assert f["team_main"].coefs == {"M01": 0.5, "M00": -0.5, "M11": 0.5, "M10": -0.5}
    assert f["rel_vs_shuf"].coefs == {"M11R": 1, "M11R-shuf": -1}
    assert f["zone_nohab"].coefs == {"M11RZ-noHab": 1, "M11R-noHab": -1}
    assert f["interaction"].coefs == {"M11": 1, "M10": -1, "M01": -1, "M00": 1}
    assert f["m11r_m11"].coefs == {"M11R": 1, "M11": -1} and f["m11rz_m11r"].coefs == {"M11RZ": 1, "M11R": -1}
    assert f["m11r_noprox"].coefs == {"M11R": 1, "M11R-noProx": -1}
    assert f["hgb_cut_nohab"].coefs == {"HGB-cut": 1, "HGB-noHab": -1}
    for name in ("m10_m00", "m11_m01", "m01_m00", "m11_m10"):
        assert sum(f[name].coefs.values()) == 0 and len(f[name].coefs) == 2


def test_holm_matches_hand_computation():
    p = {"P1": 0.001, "P2": 0.04, "P3": 0.02, "P4": 0.5, "P5": 0.01}
    adj = A.holm(p)
    # ascending: P1 .001*5, P5 .01*4, P3 .02*3, P2 .04*2, P4 .5*1, then the running maximum
    assert adj == pytest.approx({"P1": 0.005, "P5": 0.04, "P3": 0.06, "P2": 0.08, "P4": 0.5})
    assert A.holm({"P1": 0.3, "P2": 0.5, "P3": 0.9, "P4": 0.99, "P5": 0.6})["P1"] == pytest.approx(1.0)  # capped at 1
    # monotone: a larger raw p never gets a smaller adjusted p
    raw = {"P1": 0.012, "P2": 0.011, "P3": 0.2, "P4": 0.013, "P5": 0.0001}
    adj = A.holm(raw)
    order = sorted(raw, key=raw.get)
    assert all(adj[a] <= adj[b] + 1e-15 for a, b in zip(order, order[1:]))
    # a missing contrast keeps the family size at 5 (it counts as p = 1)
    assert A.holm({"P1": 0.001})["P1"] == pytest.approx(0.005)
    assert A.holm({"P1": 0.001, "P5": 0.01}) == pytest.approx({"P1": 0.005, "P5": 0.04})
    assert np.isnan(A.holm({"P1": float("nan"), "P2": 0.01})["P1"])  # an undefined p stays undefined
    assert A.holm({"P1": float("nan"), "P2": 0.01})["P2"] == pytest.approx(0.05)  # smallest valid p, still x5


def test_verdict_rules_of_design_doc_4_1_v2_2():
    v = A.verdict
    # gain: Holm-adjusted p < 0.05 AND pooled estimate >= SESOI
    assert v(0.010, (0.006, 0.014), 0.01, 0.005, +1) == "gain"
    assert v(0.005, (0.002, 0.012), 0.049, 0.005, +1) == "gain"  # exactly at the SESOI counts
    assert v(0.010, (0.002, 0.018), 0.06, 0.005, +1) == "inconclusive"  # p not below 0.05, CI wider than SESOI
    assert v(0.004, (0.0045, 0.0048), 0.001, 0.005, +1) == "equivalent"  # significant but below SESOI, CI in +-SESOI
    assert v(0.004, (0.001, 0.0095), 0.001, 0.005, +1) == "inconclusive"  # below SESOI and CI not inside +-SESOI
    # equivalent: pooled 90% CI inside +-SESOI (inclusive), whatever the p-value
    assert v(0.001, (-0.002, 0.004), 0.9, 0.005, +1) == "equivalent"
    assert v(0.0, (-0.005, 0.005), 1.0, 0.005, +1) == "equivalent"
    assert v(0.0, (-0.0051, 0.005), 1.0, 0.005, +1) == "inconclusive"
    # a clear loss is not a gain and not an equivalence
    assert v(-0.02, (-0.03, -0.01), 1.0, 0.005, +1) == "inconclusive"
    # metrics where lower is better flip the sign before the rules apply; no SESOI -> no verdict
    assert v(-0.020, (-0.025, -0.015), 0.01, 0.005, -1) == "gain"
    assert v(-0.020, (-0.025, -0.015), 0.01, None, -1) == "n/a"
    assert v(float("nan"), (float("nan"), float("nan")), float("nan"), 0.005, +1) == "n/a"
    # Spearman has SESOI 0.01
    assert v(0.008, (0.004, 0.012), 0.001, 0.01, +1) == "inconclusive"
    assert v(0.012, (0.008, 0.016), 0.001, 0.01, +1) == "gain"


def test_number_formatting_for_the_tables():
    assert A.fmt_delta(0.0123, 0.006, 0.0181) == "+0.012 [0.006, 0.018]"
    assert A.fmt_delta(-0.0123, -0.0181, -0.006) == "$-$0.012 [$-$0.018, $-$0.006]"
    assert A.fmt_delta(0.0004, -0.0011, 0.0019) == "+0.000 [$-$0.001, 0.002]"
    assert A.fmt_delta(float("nan"), 0, 0) == "---"
    assert A.fmt_p(0.0004) == "$<$0.001" and A.fmt_p(0.0234) == "0.023" and A.fmt_p(1.0) == "1.000"  # OT1-safe: no bare "<"
    assert A.fmt_p(float("nan")) == "---"
    assert A.fmt_tau2(0.0) == "0" and A.fmt_tau2(0.01234) == "0.0123"
    assert A.fmt_tau2(2.1e-5) == r"$2.1\!\times\!10^{-5}$" and A.fmt_tau2(float("nan")) == "---"
    assert A.fmt_value(0.78456) == "0.785" and A.fmt_value(float("nan")) == "---"
    assert A.cohort_label("erangel_tpp_squad") == "Erangel TPP squad"
    assert A.cohort_label("miramar_desert_tpp_squad") == "Miramar Desert TPP squad"
    assert A.cohort_label("taego_fpp_squad") == "Taego FPP squad"


def test_bootstrap_arms_is_the_shared_engine_of_the_paired_helpers(tmp_path):
    root = make_runs(tmp_path, cohorts=["c_alpha"], skip=SKIP_EXPLORATORY)
    paths = {a: root / "c_alpha" / a / "seed42" / "pred_test.csv" for a in ("M00", "M10", "M01", "M11")}
    info, point, samples = M.bootstrap_arms(paths, n_boot=60, seed=4)
    ref = M.factorial_main_effect(paths["M10"], paths["M00"], paths["M11"], paths["M01"], n_boot=60, seed=4)
    got = A.contrast_from_samples(point, samples, A.FORMULAS["phase_main"].coefs, "auc")
    assert got["estimate"] == pytest.approx(ref["metrics"]["auc"]["estimate"])
    assert got["se"] == pytest.approx(ref["metrics"]["auc"]["se"]) and got["ci_low"] == pytest.approx(ref["metrics"]["auc"]["ci_low"])
    assert info["n_boot"] == 60 and set(point) == set(paths) and samples["M00"]["auc"].shape == (60,)


# --------------------------------------------------------------------------- the full pipeline on known effects
def test_all_outputs_are_written(full):
    res, root, out = full
    for name in ("arm_metrics_per_cohort.csv", "arm_metrics_mean.csv", "contrasts_per_cohort.csv", "contrasts_pooled.csv",
                 "kappa.csv", "seed_stability.csv", "run_inventory.csv", "issues.csv", "summary.json"):  # fmt: skip
        assert (out / name).is_file() and (out / name).stat().st_size > 0, name
    for name in ("tab_primary.tex", "tab_exploratory.tex", "tab_arms_metrics.tex", "tab_primary_strata.tex"):
        assert (out / "tables" / name).is_file(), name
    for name in ("forest_primary.pdf", "forest_primary.png", "forest_primary_data.csv"):
        assert (out / "figures" / name).is_file() and (out / "figures" / name).stat().st_size > 0, name
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_boot"] == 150 and summary["labels"] == "v2" and summary["n_cohorts"] == 4
    assert summary["sesoi"] == A.SESOI and sorted(summary["cohorts"]) == sorted(COHORTS)
    assert summary["strata"] == ["all", "competitive", "official"]
    assert summary["holm_family"] == ["P1", "P2", "P3", "P4", "P5"] and summary["primary"]["P3"] == ["team_main", "spearman"]
    assert pd.read_csv(out / "contrasts_pooled.csv").shape[0] == len(res.pooled)


def test_per_cohort_metrics_are_the_plain_metrics_of_the_prediction_files(full):
    res, root, out = full
    df = M.read_predictions(root / "c_beta" / "M11R" / "seed42" / "pred_test.csv")
    plain = M.all_metrics(df, by_match_type=False)
    row = res.arm_metrics
    sel = lambda metric, stratum="all": float(row[(row.cohort == "c_beta") & (row.arm == "M11R") & (row.seed == 42) & (row.metric == metric) & (row.stratum == stratum)].value.iloc[0])  # noqa: E731
    assert sel("auc") == pytest.approx(plain["hazard"]["auc"], abs=1e-12) and sel("nll") == pytest.approx(plain["hazard"]["nll"], abs=1e-12)
    assert sel("spearman") == pytest.approx(plain["rank"]["spearman"], abs=1e-12) and sel("cindex") == pytest.approx(plain["rank"]["cindex"], abs=1e-12)
    off = M.all_metrics(df[df.match_type == "official"], by_match_type=False)
    assert sel("auc", "official") == pytest.approx(off["hazard"]["auc"], abs=1e-12)
    # arm means are unweighted means over cohorts at seed 42
    m = res.arm_means
    want = np.mean([float(row[(row.cohort == c) & (row.arm == "M00") & (row.seed == 42) & (row.metric == "auc") & (row.stratum == "all")].value.iloc[0]) for c in COHORTS])
    got = float(m[(m.arm == "M00") & (m.metric == "auc") & (m.stratum == "all")].iloc[0]["mean"])
    assert got == pytest.approx(want, abs=1e-12) and int(m[(m.arm == "M00") & (m.metric == "auc")].iloc[0].n_cohorts) == 4


def test_primary_contrasts_recover_the_known_effects_and_verdicts(full):
    res, root, out = full
    P = {k: pooled(res, *v) for k, v in A.PRIMARY.items()}
    truth_p1 = np.mean([auc_theory(eff(c, "M10")) - auc_theory(eff(c, "M00")) for c in COHORTS])
    assert P["P1"].theta == pytest.approx(truth_p1, abs=0.025)  # phase-axis main effect on AUC
    assert P["P1"].verdict == "gain" and P["P1"].p_holm < 0.001 and P["P1"].k == 4
    assert P["P1"].ci95_low < P["P1"].theta < P["P1"].ci95_high and P["P1"].primary_id == "P1"
    # team axis: no hazard effect (M01 ~ M00, M11 ~ M10) ...
    assert abs(P["P2"].theta) < 0.003 and P["P2"].verdict == "equivalent"
    assert P["P2"].ci90_low > -A.SESOI["auc"] and P["P2"].ci90_high < A.SESOI["auc"]
    # ... but a clear gain in relative standing
    assert P["P3"].theta > 0.05 and P["P3"].verdict == "gain" and P["P3"].sesoi == 0.01 and P["P3"].p_holm < 0.001
    truth_p4 = np.mean([auc_theory(eff(c, "M11R")) - auc_theory(eff(c, "M11R-shuf")) for c in COHORTS])
    assert P["P4"].theta == pytest.approx(truth_p4, abs=0.025) and P["P4"].verdict == "gain"
    truth_p5 = np.mean([auc_theory(eff(c, "M11RZ-noHab")) - auc_theory(eff(c, "M11R-noHab")) for c in COHORTS])
    assert P["P5"].theta == pytest.approx(truth_p5, abs=0.025) and P["P5"].verdict == "gain"
    assert (P["P1"].tau2 >= 0) and P["P1"].tau2 < 0.01 and np.isfinite(P["P1"].Q) and np.isfinite(P["P1"].I2)


def test_pooling_uses_the_per_cohort_bootstrap_ses_with_dersimonian_laird(full):
    res, root, out = full
    for pid, (formula, metric) in A.PRIMARY.items():
        rows = res.contrasts[(res.contrasts.contrast == formula) & (res.contrasts.metric == metric) & (res.contrasts.stratum == "all")]
        assert len(rows) == 4 and (rows.se > 0).all()
        dl = M.dersimonian_laird(rows.estimate.to_numpy(), rows.se.to_numpy())
        got = pooled(res, formula, metric)
        assert got.theta == pytest.approx(dl["theta"]) and got.se == pytest.approx(dl["se"]) and got.tau2 == pytest.approx(dl["tau2"])
        assert (got.ci95_low, got.ci95_high) == pytest.approx(dl["ci95"]) and (got.ci90_low, got.ci90_high) == pytest.approx(dl["ci90"])
        assert got.z == pytest.approx(dl["z"]) and got.p_one_sided == pytest.approx(dl["p_one_sided"], abs=1e-12)
        assert got.same_sign == int(np.sum(np.sign(rows.estimate.to_numpy()) == np.sign(dl["theta"])))
        assert got.k == 4 and got.k_total == 4 and set(got.cohorts.split(";")) == set(COHORTS)


def test_holm_covers_exactly_the_five_primary_tests(full):
    res, root, out = full
    df = res.pooled
    prim = df[(df.primary_id != "") & (df.stratum == "all")].set_index("primary_id")
    assert list(sorted(prim.index)) == ["P1", "P2", "P3", "P4", "P5"] and (prim.kind == "primary").all()
    want = A.holm({k: float(prim.loc[k, "p_one_sided"]) for k in prim.index})
    for k in prim.index:
        assert prim.loc[k, "p_holm"] == pytest.approx(want[k], abs=1e-12)
        assert prim.loc[k, "p_holm"] >= prim.loc[k, "p_one_sided"] - 1e-15
    others = df[~((df.primary_id != "") & (df.stratum == "all"))]
    assert others["p_holm"].isna().all(), "no Holm adjustment outside the five primary tests (incl. other strata)"
    assert (others[others.kind == "secondary"].exploratory == True).all()  # noqa: E712
    assert (df[df.kind == "primary"].exploratory == False).all()  # noqa: E712
    # every row has the unadjusted one-sided p and its two-sided sibling
    assert df.p_one_sided.between(0, 1).all()


def test_secondary_contrasts_are_complete_and_unadjusted(full):
    res, root, out = full
    need = ["m10_m00", "m11_m01", "m01_m00", "m11_m10", "interaction", "m11r_m11", "m11rz_m11r",
            "m11r_noprox", "m11r_nocombat", "m11r_nocopress", "hgb_cut_nohab", "hgb_lag_cut"]  # fmt: skip
    for formula in need:
        for metric in A.METRICS:  # the seven metrics and the exploratory within-phase AUC
            row = pooled(res, formula, metric)
            assert row.kind == "secondary" and np.isfinite(row.theta) and np.isfinite(row.se)
    assert pooled(res, "m11r_noprox", "auc").k == 2 and pooled(res, "m11r_nocombat", "auc").k == 2  # two large cohorts
    assert pooled(res, "m10_m00", "auc").k == 4
    assert pooled(res, "m10_m00", "auc").theta == pytest.approx(auc_theory(1.3) - auc_theory(0.9), abs=0.03)
    assert pooled(res, "interaction", "auc").theta == pytest.approx(0.0, abs=0.02)
    assert pooled(res, "m11r_noprox", "auc").theta > 0  # removing prox lowered AUC in the synthetic truth
    # the other metrics of the primary formulas are reported too (exploratory)
    assert pooled(res, "phase_main", "ap").kind == "secondary" and np.isfinite(pooled(res, "team_main", "cindex").theta)
    # lower-is-better metrics: the one-sided p is for an improvement (a decrease)
    nll = pooled(res, "m10_m00", "nll")
    assert nll.theta < 0 and nll.p_one_sided < 0.01
    assert nll.verdict == "n/a" and pooled(res, "m10_m00", "brier").verdict == "n/a"  # no SESOI for them
    assert pooled(res, "m10_m00", "auc").verdict == "gain" and pooled(res, "m10_m00", "ap").verdict == "n/a"


def test_recovery_ratio_kappa_with_the_denominator_gate(full):
    res, root, out = full
    k = res.kappa
    pooled_row = k[(k.level == "pooled") & (k.stratum == "all")].iloc[0]
    assert bool(pooled_row.estimable) and pooled_row.den_ci95_low >= A.SESOI["auc"]
    truth = np.mean([(auc_theory(eff(c, "M11RZ-noHab")) - auc_theory(eff(c, "M11R-noHab")))
                     / (auc_theory(eff(c, "M11R")) - auc_theory(eff(c, "M11R-noHab"))) for c in COHORTS])  # fmt: skip
    assert pooled_row.kappa == pytest.approx(truth, abs=0.2)
    assert pooled_row.ci_low < pooled_row.kappa < pooled_row.ci_high and pooled_row.k == 4 and pooled_row.n_valid > 100
    assert "reportable" in k.columns
    coh = k[(k.level == "cohort") & (k.stratum == "all")].set_index("cohort")
    assert set(coh.index) == set(COHORTS)
    for c in COHORTS:
        assert coh.loc[c, "kappa"] == pytest.approx(coh.loc[c, "numerator"] / coh.loc[c, "denominator"])
        assert bool(coh.loc[c, "reportable"]) is True  # the pooled gate passed, so the descriptive values may be shown
    # numerator and denominator are the pooled contrasts P5 and M11R - M11R-noHab
    p5 = pooled(res, "zone_nohab", "auc")
    assert pooled_row.numerator == pytest.approx(p5.theta) and pooled_row.denominator > 0.03


def test_kappa_is_not_estimable_when_the_denominator_does_not_qualify(tmp_path):
    flat = {**QUALITY, "M11R": QUALITY["M11R-noHab"]}  # Hab carries nothing: M11R == M11R-noHab
    root = make_runs(tmp_path / "r", cohorts=COHORTS[:3], quality=flat, near_copy={**NEAR_COPY, "M11R": "M11R-noHab"})
    res = A.aggregate(root, tmp_path / "o", n_boot=80, seed=1, workers=1, figures=False)
    row = res.kappa[(res.kappa.level == "pooled") & (res.kappa.stratum == "all")].iloc[0]
    assert not bool(row.estimable) and np.isnan(row.kappa) and np.isnan(row.ci_low)
    assert row.den_ci95_low < A.SESOI["auc"] and "not estimable" in row.note
    coh = res.kappa[(res.kappa.level == "cohort") & (res.kappa.stratum == "all")]
    assert not coh.reportable.any()  # per-cohort values are descriptive only and withheld
    assert "not estimable" in (tmp_path / "o" / "tables" / "tab_exploratory.tex").read_text()


def test_seed_standard_deviations_are_reported_separately_and_not_pooled(full):
    res, root, out = full
    s = res.seed_stability
    assert set(s.cohort) == {"c_alpha", "c_beta"}  # only the cohorts that have the extra seeds
    arms = s[s.kind == "arm"]
    assert set(arms.name) == {"M00", "M11", "M11R", "M11R-shuf", "M11RZ"}
    assert (arms.n_seeds == 3).all() and (arms.sd > 0).all()
    row = arms[(arms.cohort == "c_alpha") & (arms.name == "M11R") & (arms.metric == "auc")].iloc[0]
    values = json.loads(row["values"])
    assert sorted(json.loads(row["seeds"])) == [13, 42, 2026] and len(values) == 3
    assert row["mean"] == pytest.approx(np.mean(values)) and row["sd"] == pytest.approx(np.std(values, ddof=1))
    plain = M.all_metrics(M.read_predictions(root / "c_alpha" / "M11R" / "seed13" / "pred_test.csv"), by_match_type=False)
    assert plain["hazard"]["auc"] in [pytest.approx(v) for v in values]
    contrasts = s[s.kind == "contrast"]
    assert {"rel_vs_shuf", "m11_m00", "m11r_m11", "m11rz_m11r"} <= set(contrasts.name)  # contrasts available at every seed
    assert (contrasts[contrasts.name == "rel_vs_shuf"].n_seeds == 3).all()
    # the pooled results never see the extra seeds: same numbers with and without them
    assert (res.inventory[res.inventory.seed != 42].shape[0]) == 2 * 2 * 5 and pooled(res, "rel_vs_shuf", "auc").k == 4
    assert res.contrasts.cohort.nunique() == 4 and len(res.contrasts[res.contrasts.contrast == "rel_vs_shuf"]) == 4 * len(A.METRICS) * 3


def test_strata_by_match_type_are_analysed_with_their_own_bootstrap(full):
    res, root, out = full
    assert set(res.pooled.stratum) == {"all", "official", "competitive"}
    for stratum in ("official", "competitive"):
        row = pooled(res, "phase_main", "auc", stratum)
        assert row.k == 4 and row.verdict in ("gain", "inconclusive", "equivalent") and np.isfinite(row.se)
        assert pd.isna(row.p_holm)  # a stratum analysis is exploratory
        sub = res.contrasts[(res.contrasts.stratum == stratum) & (res.contrasts.contrast == "phase_main") & (res.contrasts.metric == "auc")]
        df = M.read_predictions(root / "c_alpha" / "M00" / "seed42" / "pred_test.csv")
        n_rows = int((df.match_type == stratum).sum())
        assert int(sub[sub.cohort == "c_alpha"].n_rows.iloc[0]) == n_rows
    # more matches -> a smaller bootstrap SE.  Compare the per-cohort SEs (the random-effects pooled SE also carries tau2,
    # which differs between strata and is not a function of the sample size alone)
    def mean_se(stratum):
        sub = res.contrasts[(res.contrasts.contrast == "phase_main") & (res.contrasts.metric == "auc") & (res.contrasts.stratum == stratum)]
        return float(sub.se.mean())

    assert mean_se("all") < mean_se("official") < mean_se("competitive")  # 60 vs 40 vs 20 matches per cohort
    assert set(res.arm_metrics.stratum) == {"all", "official", "competitive"}
    assert (res.kappa.stratum.isin(["all", "official", "competitive"])).all()


# --------------------------------------------------------------------------- tables
def parse_tabular(text):
    body = re.search(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", text, re.S)
    spec, rest = body.group(1), body.group(2)
    lines = [ln.strip() for ln in rest.splitlines() if ln.strip() and not ln.strip().startswith("%")]
    rows = [[c.strip() for c in ln.rstrip("\\").rstrip().split("&")] for ln in lines if ln.endswith("\\\\")]
    return spec, rows


# The three table skeletons of the pre-registered paper (paper/bigcomp2027/sections/05-experiments.tex at commit f59bac9, before the
# paper started to build its tables from the aggregator's CSVs).  The generated tables must keep this structure; the test no
# longer reads the paper, which is free to change.
PAPER_SKELETON = (
    r"""\begin{tabular}{@{}l l l c c c c c@{}}
\toprule
 & Contrast & Metric & $\Delta$ [95\% CI] & Same sign /13 & $\tau^2$ & Holm $p$ & Verdict \\
\midrule
P1 (RQ1) & phase-axis main effect & AUC & & & & & \\
P2 (RQ2) & team-axis main effect & AUC & & & & & \\
P3 (RQ2) & team-axis main effect & Spearman & & & & & \\
P4 (RQ3) & M11R $-$ M11R-shuf & AUC & & & & & \\
P5 (RQ4) & M11RZ-noHab $-$ M11R-noHab & AUC & & & & & \\
\bottomrule
\end{tabular}""",
    r"""\begin{tabular}{@{}l l c c c c@{}}
\toprule
RQ & Contrast & $\Delta$AUC & $\Delta$Spearman & $\Delta$Top-5 AUC & $\Delta$C-index \\
\midrule
RQ1 & M10 $-$ M00 & & & & \\
RQ1 & M11 $-$ M01 & & & & \\
RQ2 & M01 $-$ M00 & & & & \\
RQ2 & M11 $-$ M10 & & & & \\
RQ1, RQ2 & interaction, $(\mathrm{M11}-\mathrm{M10})-(\mathrm{M01}-\mathrm{M00})$ & & & & \\
RQ3 & M11R $-$ M11 & & & & \\
RQ3 & M11R $-$ M11R$-$prox, $-$combat, $-$copress (two cohorts; three rows) & & & & \\
RQ4 & M11RZ $-$ M11R & & & & \\
RQ4 & $\kappa$ (recovery ratio) & & --- & --- & --- \\
RQ4 & HGB-cut $-$ HGB-noHab (reference) & & & & \\
% STAGE2: delete if not run by 10/9
Player layer & M11RZ-P $-$ M11RZ & & & & \\
RQ3 (teammates) & M11RZ-P $-$ M11RZ-P$-$tm & & & & \\
% STAGE2 end
\bottomrule
\end{tabular}""",
    r"""\begin{tabular}{@{}lccccccc@{}}
\toprule
 & \multicolumn{4}{c}{Hazard} & \multicolumn{3}{c}{Relative standing} \\
\cmidrule(lr){2-5}\cmidrule(l){6-8}
Arm & AUC & AP & NLL & Brier & Spearman & Top-5 AUC & C-index \\
\midrule
M00 & & & & & & & \\
M10 & & & & & & & \\
M01 & & & & & & & \\
M11 & & & & & & & \\
M11R & & & & & & & \\
M11R-shuf & & & & & & & \\
M11RZ & & & & & & & \\
M11R-noHab & & & & & & & \\
M11RZ-noHab & & & & & & & \\
% STAGE2: delete if not run by 10/9
M11RZ-P & & & & & & & \\
M11RZ-P$-$tm & & & & & & & \\
% STAGE2 end
HGB-cut & & & & & & & \\
HGB-lag & & & & & & & \\
HGB-noHab & & & & & & & \\
\bottomrule
\end{tabular}""",
)  # fmt: skip


def test_tables_have_the_structure_of_the_paper_skeletons(full):
    res, root, out = full
    blocks = list(PAPER_SKELETON)
    assert len(blocks) == 3
    sk_primary, sk_expl, sk_arms = (parse_tabular(b) for b in blocks)
    mine = {n: parse_tabular((out / "tables" / f"tab_{n}.tex").read_text()) for n in ("primary", "exploratory", "arms_metrics")}

    # primary table: same column spec, same header cells (except the cohort-count in "Same sign"), P1..P5 in order
    assert mine["primary"][0] == sk_primary[0]
    assert len(mine["primary"][1][0]) == len(sk_primary[1][0]) == 8
    assert [c for c in mine["primary"][1][0] if not c.startswith("Same sign")] == [c for c in sk_primary[1][0] if not c.startswith("Same sign")]
    assert mine["primary"][1][0][4].startswith("Same sign")
    sk_rows = [r for r in sk_primary[1][1:] if r[0].startswith("P")]
    my_rows = [r for r in mine["primary"][1][1:] if r[0].startswith("P")]
    assert [r[:3] for r in my_rows] == [r[:3] for r in sk_rows] and len(my_rows) == 5
    assert all(len(r) == 8 for r in my_rows)
    # exploratory table: same spec and header, skeleton rows in order (the relation-type row expanded to three rows)
    assert mine["exploratory"][0] == sk_expl[0] and mine["exploratory"][1][0] == sk_expl[1][0]
    sk_first = [(r[0], r[1]) for r in sk_expl[1][1:] if "Player layer" not in r[0] and "teammates" not in r[0]]
    my_first = [(r[0], r[1]) for r in mine["exploratory"][1][1:]]
    idx = next(i for i, r in enumerate(sk_first) if "prox" in r[1])
    expected = sk_first[:idx] + [("RQ3", f"M11R $-$ M11R$-$ {rel}") for rel in ("prox", "combat", "copress")] + sk_first[idx + 1:]
    norm_ = lambda rows: [(a, re.sub(r"\s*\(.*?\)|\s+", " ", b).replace("$-$ ", "$-$").strip()) for a, b in rows]  # noqa: E731
    assert norm_(my_first) == norm_(expected)
    assert all(len(r) == 6 for r in mine["exploratory"][1][1:])
    # arms table: same spec and the skeleton's arm order, 8 cells per row
    assert mine["arms_metrics"][0] == sk_arms[0]
    sk_arm_names = [r[0] for r in sk_arms[1] if re.match(r"(M\d|HGB)", r[0]) and "-P" not in r[0] and "P$-$" not in r[0]]
    my_arm_names = [r[0] for r in mine["arms_metrics"][1] if re.match(r"(M\d|HGB)", r[0])]
    assert my_arm_names == sk_arm_names and all(len(r) == 8 for r in mine["arms_metrics"][1] if re.match(r"(M\d|HGB)", r[0]))


def test_primary_table_cells_carry_the_pooled_numbers(full):
    res, root, out = full
    text = (out / "tables" / "tab_primary.tex").read_text()
    spec, rows = parse_tabular(text)
    byid = {r[0].split()[0]: r for r in rows if r[0].startswith("P")}
    for pid in ("P1", "P2", "P3", "P4", "P5"):
        row = pooled(res, *A.PRIMARY[pid])
        cells = byid[pid]
        assert cells[3] == A.fmt_delta(row.theta, row.ci95_low, row.ci95_high)
        assert cells[4] == str(int(row.same_sign))  # uniform k -> integer counts, the header carries the /K
        assert cells[5] == A.fmt_tau2(row.tau2) and cells[6] == A.fmt_p(row.p_holm) and cells[7] == row.verdict
    assert "Same sign /$4$" in text or "Same sign /4" in text
    assert byid["P1"][7] == "gain" and byid["P2"][7] == "equivalent"
    assert "generated by model_stgcn.ptr.aggregate" in text and "\\input" in text  # how to include it


def test_exploratory_table_cells_and_kappa_row(full):
    res, root, out = full
    spec, rows = parse_tabular((out / "tables" / "tab_exploratory.tex").read_text())
    by = {(r[0], r[1].split(" (")[0]): r for r in rows[1:]}
    row = pooled(res, "m10_m00", "auc")
    cells = by[("RQ1", "M10 $-$ M00")]
    assert cells[2] == A.fmt_delta(row.theta, row.ci95_low, row.ci95_high)
    cm = pooled(res, "m10_m00", "cindex")
    assert cells[5] == A.fmt_delta(cm.theta, cm.ci95_low, cm.ci95_high)
    kap = next(r for r in rows[1:] if r[1].startswith("$\\kappa$"))
    p = res.kappa[(res.kappa.level == "pooled") & (res.kappa.stratum == "all")].iloc[0]
    assert kap[2].startswith(f"{p.kappa:.2f}".replace("-", "$-$")) and kap[3:] == ["---", "---", "---"]
    assert f"{p.ci_low:.2f}" in kap[2] or f"{p.ci_low:.2f}".replace("-", "$-$") in kap[2]
    rel = [r for r in rows[1:] if "M11R $-$ M11R$-$" in r[1]]
    assert len(rel) == 3 and all("2 cohorts" in r[1] for r in rel)
    hgb = next(r for r in rows[1:] if r[1].startswith("HGB-cut"))
    hp = pooled(res, "hgb_cut_nohab", "auc")
    assert hgb[2] == A.fmt_delta(hp.theta, hp.ci95_low, hp.ci95_high)


def test_arms_table_is_the_unweighted_mean_over_cohorts(full):
    res, root, out = full
    spec, rows = parse_tabular((out / "tables" / "tab_arms_metrics.tex").read_text())
    by = {r[0]: r for r in rows if re.match(r"(M\d|HGB)", r[0])}
    m = res.arm_means
    for arm in ("M00", "M11RZ", "HGB-lag"):
        cells = by[arm]
        for j, metric in enumerate(("auc", "ap", "nll", "brier", "spearman", "top5_auc", "cindex"), start=1):
            val = float(m[(m.arm == arm) & (m.metric == metric) & (m.stratum == "all")].iloc[0]["mean"])
            assert cells[j] == A.fmt_value(val), (arm, metric)
    assert not any(a in by for a in ("M11R-noProx", "M11R-noCombat", "M11R-noCopress"))  # exploratory arms: not in the table
    spec, rows = parse_tabular((out / "tables" / "tab_primary_strata.tex").read_text())
    assert len([r for r in rows if r[0].startswith("P")]) == 5 and "official" in " ".join(rows[0]).lower()


def test_stage_two_rows_appear_only_when_those_arms_were_run(tmp_path):
    quality = {**QUALITY, "M11RZ-P": 1.7, "M11RZ-P-tm": 1.65}
    root = make_runs(tmp_path / "r", cohorts=COHORTS[:3], quality=quality, rank_quality={**RANK_QUALITY, "M11RZ-P": 2.3, "M11RZ-P-tm": 2.3})
    res = A.aggregate(root, tmp_path / "o", n_boot=60, seed=1, workers=1, figures=False)
    assert pooled(res, "p_m11rz", "auc").k == 3 and pooled(res, "p_tm", "auc").k == 3
    expl = (tmp_path / "o" / "tables" / "tab_exploratory.tex").read_text()
    arms = (tmp_path / "o" / "tables" / "tab_arms_metrics.tex").read_text()
    assert "M11RZ-P $-$ M11RZ" in expl and "Player layer" in expl and "teammates" in expl
    assert "M11RZ-P &" in arms and "M11RZ-P$-$tm &" in arms
    # the main fixture has no stage-2 arms: no such rows
    base = make_runs(tmp_path / "r2", cohorts=COHORTS[:2])
    res2 = A.aggregate(base, tmp_path / "o2", n_boot=40, seed=1, workers=1, figures=False)
    assert (res2.pooled.contrast == "p_m11rz").sum() == 0
    assert "M11RZ-P" not in (tmp_path / "o2" / "tables" / "tab_exploratory.tex").read_text()
    assert "M11RZ-P" not in (tmp_path / "o2" / "tables" / "tab_arms_metrics.tex").read_text()


# --------------------------------------------------------------------------- forest plot
def test_forest_plot_files_and_data_twin(full):
    res, root, out = full
    data = pd.read_csv(out / "figures" / "forest_primary_data.csv")
    assert {"panel", "contrast", "metric", "cohort", "kind", "estimate", "ci_low", "ci_high"} <= set(data.columns)
    assert len(data) == 5 * (4 + 1) and set(data.panel) == {"P1", "P2", "P3", "P4", "P5"}
    assert (data.kind == "pooled").sum() == 5 and (data.kind == "cohort").sum() == 20
    p1 = data[(data.panel == "P1") & (data.kind == "pooled")].iloc[0]
    row = pooled(res, *A.PRIMARY["P1"])
    assert p1.estimate == pytest.approx(row.theta) and p1.ci_low == pytest.approx(row.ci95_low)
    assert (out / "figures" / "forest_primary.pdf").read_bytes()[:4] == b"%PDF"
    assert (out / "figures" / "forest_primary.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    from matplotlib import image

    img = image.imread(out / "figures" / "forest_primary.png")
    assert img.shape[0] > 300 and img.shape[1] > 1000 and img.shape[1] > img.shape[0]  # wide, double-column layout
    assert img[..., :3].std() > 0.02  # not blank


# --------------------------------------------------------------------------- robustness
def test_partial_matrices_pool_over_the_cohorts_that_have_the_arms(tmp_path):
    root = make_runs(tmp_path / "r", cohorts=COHORTS[:3], skip={("c_gamma", "M01"), ("c_beta", "M11RZ-noHab")} | SKIP_EXPLORATORY)
    res = A.aggregate(root, tmp_path / "o", n_boot=60, seed=2, workers=1, figures=False)
    assert pooled(res, "phase_main", "auc").k == 2 and pooled(res, "phase_main", "auc").k_total == 3  # c_gamma lacks M01
    assert pooled(res, "zone_nohab", "auc").k == 2  # c_beta lacks M11RZ-noHab
    assert pooled(res, "m10_m00", "auc").k == 3  # needs only M10 and M00
    assert pooled(res, "m11r_noprox", "auc").k == 2  # only c_alpha and c_beta ran the exploratory arms
    msgs = " ".join(res.issues.message)
    assert "m11r_noprox" not in msgs and "noProx" not in msgs  # an optional contrast that a cohort lacks is not a problem
    assert "c_gamma" in msgs and "phase_main" in msgs and "M01" in msgs
    assert "c_beta" in msgs and "zone_nohab" in msgs
    # the table prints --- for a contrast no cohort can compute
    root2 = make_runs(tmp_path / "r2", cohorts=COHORTS[:2], skip={(c, "M11RZ-noHab") for c in COHORTS[:2]} | SKIP_EXPLORATORY)
    res2 = A.aggregate(root2, tmp_path / "o2", n_boot=40, seed=2, workers=1, figures=False)
    assert (res2.pooled.primary_id == "P5").sum() == 0
    _, rows = parse_tabular((tmp_path / "o2" / "tables" / "tab_primary.tex").read_text())
    p5 = next(r for r in rows if r[0].startswith("P5"))
    assert p5[3:] == ["---"] * 5
    k = res2.kappa
    assert k.empty or not k[(k.level == "pooled")].estimable.any()


def test_mixed_label_choices_are_rejected(tmp_path):
    root = make_runs(tmp_path / "r", cohorts=COHORTS[:2])
    meta = root / "c_beta" / "M00" / "seed42" / "metrics.json"
    meta.write_text(json.dumps({**json.loads(meta.read_text()), "labels": "pipeline"}))
    with pytest.raises(ValueError, match="labels"):
        A.aggregate(root, tmp_path / "o", n_boot=20, workers=1, figures=False)


def test_misaligned_rows_fail_loudly_unless_skipping_is_requested(tmp_path):
    root = make_runs(tmp_path / "r", cohorts=COHORTS[:2])
    path = root / "c_beta" / "M11R" / "seed42" / "pred_test.csv"
    df = pd.read_csv(path)
    df.iloc[1:].to_csv(path, index=False)  # one row missing
    with pytest.raises(ValueError, match=r"c_beta.*M11R|M11R.*c_beta"):
        A.aggregate(root, tmp_path / "o", n_boot=20, workers=1, figures=False)
    res = A.aggregate(root, tmp_path / "o2", n_boot=30, seed=1, workers=1, figures=False, skip_misaligned=True)
    assert "M11R" in " ".join(res.issues.message) and "c_beta" in " ".join(res.issues.message)
    assert pooled(res, "m11r_m11", "auc").k == 1  # only c_alpha still has both arms
    assert pooled(res, "phase_main", "auc").k == 2  # unaffected contrasts keep both cohorts


def test_runs_without_predictions_or_metrics_are_ignored_and_reported(tmp_path):
    root = make_runs(tmp_path / "r", cohorts=COHORTS[:2], seeds={})  # seed 42 only: c_alpha/M00 has no other run
    (root / "c_alpha" / "M00" / "seed42" / "metrics.json").unlink()  # an unfinished run
    (root / "c_alpha" / "M10" / "seed42" / "pred_test.csv").unlink()  # a broken one
    (root / "logs").mkdir()
    (root / "run_matrix_status.csv").write_text("run_id\n")
    res = A.aggregate(root, tmp_path / "o", n_boot=30, seed=1, workers=1, figures=False)
    assert pooled(res, "m10_m00", "auc").k == 1  # c_alpha lost M00 and M10
    msgs = " ".join(res.issues.message)
    assert "c_alpha" in msgs and "M00" in msgs and "M10" in msgs
    assert not ((res.inventory.cohort == "c_alpha") & (res.inventory.arm == "M00")).any()


def test_results_are_deterministic_and_independent_of_the_number_of_workers(tmp_path):
    root = make_runs(tmp_path / "r", cohorts=COHORTS[:3])
    a = A.aggregate(root, tmp_path / "a", n_boot=50, seed=9, workers=1, figures=False)
    b = A.aggregate(root, tmp_path / "b", n_boot=50, seed=9, workers=3, figures=False)
    c = A.aggregate(root, tmp_path / "c", n_boot=50, seed=10, workers=1, figures=False)
    for name in ("contrasts_pooled.csv", "contrasts_per_cohort.csv", "kappa.csv", "arm_metrics_per_cohort.csv"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name
    assert (tmp_path / "a" / "contrasts_pooled.csv").read_bytes() != (tmp_path / "c" / "contrasts_pooled.csv").read_bytes()
    assert a.pooled.equals(b.pooled)


def test_cohort_filter_and_cli(tmp_path, capsys):
    root = make_runs(tmp_path / "r", cohorts=COHORTS[:3])
    rc = A.main(["--runs-dir", str(root), "--out-dir", str(tmp_path / "o"), "--n-boot", "40", "--seed", "1",
                 "--workers", "1", "--cohorts", "c_alpha,c_beta", "--no-figures"])  # fmt: skip
    assert rc == 0
    pooled_df = pd.read_csv(tmp_path / "o" / "contrasts_pooled.csv")
    assert pooled_df.k.max() == 2 and not (tmp_path / "o" / "figures" / "forest_primary.pdf").exists()
    text = capsys.readouterr().out
    assert "P1" in text and "gain" in text  # the CLI prints the primary table
    with pytest.raises(SystemExit):
        A.main(["--runs-dir", str(root), "--out-dir", str(tmp_path / "o2"), "--n-boot", "0"])
    with pytest.raises(FileNotFoundError):
        A.main(["--runs-dir", str(tmp_path / "nope"), "--out-dir", str(tmp_path / "o3"), "--n-boot", "10"])
    assert A.main(["--runs-dir", str(root), "--out-dir", str(tmp_path / "o4"), "--n-boot", "20", "--workers", "1",
                   "--no-figures", "--strata", "none"]) == 0  # fmt: skip
    assert set(pd.read_csv(tmp_path / "o4" / "contrasts_pooled.csv").stratum) == {"all"}


# --------------------------------------------------------------------------- meta-regression on the training size (design 4.4)
def reference_metareg(y, se, x):
    """Independent implementation: generalised DerSimonian-Laird tau2, then weighted least squares by row scaling."""
    y, se, x = map(np.asarray, (y, se, x))
    k = len(y)
    xc = x - x.mean()
    X = np.column_stack([np.ones(k), xc])
    w = 1 / se**2
    beta0 = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    qe = float(np.sum(w * (y - X @ beta0) ** 2))
    xtwx_inv = np.linalg.inv(X.T @ (X * w[:, None]))
    tr_p = w.sum() - np.trace(xtwx_inv @ (X.T @ (X * (w**2)[:, None])))
    tau2 = max(0.0, (qe - (k - 2)) / tr_p)
    w2 = 1 / (se**2 + tau2)
    beta = np.linalg.lstsq(X * np.sqrt(w2)[:, None], y * np.sqrt(w2), rcond=None)[0]
    cov = np.linalg.inv(X.T @ (X * w2[:, None]))
    scale = float(np.sum(w2 * (y - X @ beta) ** 2) / (k - 2))  # Knapp-Hartung residual scale
    return beta, np.sqrt(np.diag(cov)), tau2, qe, scale


def test_meta_regression_matches_an_independent_implementation():
    rng = np.random.default_rng(3)
    for k in (5, 12):
        x = np.log(rng.uniform(150, 4000, k))
        se = rng.uniform(0.002, 0.01, k)
        y = 0.03 * (x - 6) - 0.01 + rng.normal(0, 0.012, k) + rng.normal(0, se)
        fit = A.meta_regression(y, se, x)
        beta, beta_se, tau2, qe, scale = reference_metareg(y, se, x)
        assert fit["slope"] == pytest.approx(beta[1]) and fit["intercept"] == pytest.approx(beta[0])
        kh_se = beta_se[1] * np.sqrt(max(1.0, scale))  # Knapp-Hartung: variance scaled by max(1, s^2), t with k - 2 df
        assert fit["p_kh"] == pytest.approx(2 * t_dist.sf(abs(beta[1] / kh_se), k - 2))
        assert (fit["ci95_low_kh"], fit["ci95_high_kh"]) == pytest.approx((beta[1] - t_dist.ppf(0.975, k - 2) * kh_se, beta[1] + t_dist.ppf(0.975, k - 2) * kh_se))
        assert fit["se"] == pytest.approx(beta_se[1]) and fit["tau2"] == pytest.approx(tau2) and fit["QE"] == pytest.approx(qe)
        assert fit["k"] == k and fit["QE_df"] == k - 2 and fit["mean_x"] == pytest.approx(x.mean())
        assert fit["z"] == pytest.approx(beta[1] / beta_se[1]) and fit["p"] == pytest.approx(2 * norm.sf(abs(beta[1] / beta_se[1])))
        assert fit["QM"] == pytest.approx(fit["z"] ** 2)
        assert (fit["ci95_low"], fit["ci95_high"]) == pytest.approx((beta[1] - 1.959964 * beta_se[1], beta[1] + 1.959964 * beta_se[1]), rel=1e-5)
        assert fit["p_kh"] >= fit["p"] - 1e-12  # Knapp-Hartung with the max(1, s^2) rule is never more liberal
        assert fit["ci95_low_kh"] <= fit["ci95_low"] + 1e-12 and fit["ci95_high_kh"] >= fit["ci95_high"] - 1e-12
        assert fit["weights"] == pytest.approx(1 / (se**2 + tau2)) and len(fit["fitted"]) == k


def test_meta_regression_recovers_a_noiseless_line_and_its_crossover():
    n = np.array([150.0, 300, 600, 1200, 2400, 3300])
    x = np.log(n)
    y = 0.02 * (x - np.log(500.0))  # zero at N = 500, slope 0.02 per ln N
    fit = A.meta_regression(y, np.full(6, 0.004), x)
    assert fit["slope"] == pytest.approx(0.02) and fit["tau2"] == 0.0 and fit["QE"] == pytest.approx(0.0, abs=1e-18)
    assert np.exp(fit["crossover_x"]) == pytest.approx(500.0)
    assert fit["tau2_no_moderator"] > 0 and fit["r2_heterogeneity"] == pytest.approx(1.0)  # N explains all the between-cohort spread
    lo, hi = np.exp(fit["crossover_ci_low"]), np.exp(fit["crossover_ci_high"])
    assert lo < 500 < hi  # Fieller interval brackets the crossover; the line is (almost) exact so it is narrow
    assert (hi - lo) / 500 < 0.5
    # the fitted line is zero at the crossover and the sign flips around it
    a, b = fit["intercept"], fit["slope"]
    assert a + b * (fit["crossover_x"] - fit["mean_x"]) == pytest.approx(0.0, abs=1e-12)


def test_meta_regression_with_heterogeneity_and_with_no_slope():
    x = np.log([200.0, 400, 800, 1600, 3200, 6400])
    pattern = np.array([1.0, -1, -1, -1, -1, 1])  # sums to zero against the centred x: a null slope by construction
    assert abs(np.sum(pattern * (x - x.mean()))) < 1e-12
    se = np.full(6, 0.003)
    y = 0.01 + 0.02 * pattern  # large between-cohort spread, no relation with x
    fit = A.meta_regression(y, se, x)
    assert fit["slope"] == pytest.approx(0.0, abs=1e-12) and fit["p"] > 0.95 and fit["tau2"] > 1e-4
    assert fit["intercept"] == pytest.approx(0.01 + 0.02 * pattern.mean())
    assert np.isnan(fit["crossover_ci_low"]) and np.isnan(fit["crossover_ci_high"]) and "unbounded" in fit["crossover_note"]
    assert fit["p_kh"] > 0.95 and fit["r2_heterogeneity"] == pytest.approx(0.0, abs=1e-9)  # the moderator explains nothing
    # a clear slope on top of the same spread: heterogeneity shrinks (R2 > 0), the interval is bounded
    y2 = y - 0.03 + 0.04 * (x - x.mean())
    fit2 = A.meta_regression(y2, se, x)
    assert fit2["slope"] == pytest.approx(0.04) and fit2["p"] < 0.01
    assert fit2["tau2"] == pytest.approx(fit["tau2"]) and fit2["tau2"] < 0.5 * fit2["tau2_no_moderator"] and fit2["r2_heterogeneity"] > 0.5
    assert np.isfinite(fit2["crossover_ci_low"]) and fit2["crossover_ci_low"] < fit2["crossover_x"] < fit2["crossover_ci_high"]
    # heterogeneity widens the SE of the slope w.r.t. the fixed-effect fit (weights 1/se^2)
    fixed_se = 0.003 / np.sqrt(np.sum((x - x.mean()) ** 2))
    assert fit2["se"] > 2 * fixed_se


def test_the_fieller_interval_is_the_set_of_crossovers_not_rejected_at_the_95_percent_level():
    """Direct check of the definition: u in the interval <=> (a + b u)^2 <= z^2 Var(a + b u), u = ln N* - mean ln N."""
    rng = np.random.default_rng(8)
    z = norm.ppf(0.975)
    checked = 0
    for trial in range(5):
        x = np.log(rng.uniform(150, 4000, 10))
        se = rng.uniform(0.002, 0.006, 10)
        y = 0.02 * (x - 6.8) + rng.normal(0, 0.006, 10) + rng.normal(0, se)
        f = A.meta_regression(y, se, x)
        if not np.isfinite(f["crossover_ci_low"]):
            continue
        a, b, c = f["intercept"], f["slope"], f["cov"]
        stat = lambda u: (a + b * u) ** 2 - z**2 * (c[0, 0] + 2 * u * c[0, 1] + u**2 * c[1, 1])  # <= 0 inside  # noqa: E731
        lo, hi = f["crossover_ci_low"] - f["mean_x"], f["crossover_ci_high"] - f["mean_x"]
        assert abs(stat(lo)) < 1e-12 and abs(stat(hi)) < 1e-12  # the endpoints solve the equation
        mid = 0.5 * (lo + hi)
        assert stat(mid) < 0 and stat(lo - 0.05) > 0 and stat(hi + 0.05) > 0  # inside accepted, outside rejected
        assert lo < f["crossover_x"] - f["mean_x"] < hi
        checked += 1
    assert checked >= 3  # most draws have a significant slope, i.e. a bounded interval


def test_meta_regression_degenerate_inputs():
    for y, se, x in (([1.0, 2.0], [0.1, 0.1], [0.0, 1.0]),  # two studies: no degrees of freedom
                     ([1.0, 2.0, 3.0], [0.1, 0.1, 0.1], [2.0, 2.0, 2.0]),  # no spread in the moderator
                     ([np.nan, 2.0, 3.0], [0.1, 0.1, 0.1], [0.0, 1.0, 2.0])):  # a missing estimate leaves two
        fit = A.meta_regression(y, se, x)
        assert np.isnan(fit["slope"]) and np.isnan(fit["p"]) and np.isnan(fit["crossover_x"]) and "at least 3" in fit["crossover_note"]
    assert A.meta_regression([1, 2, 3, 4], [0.1, 0, 0.1, 0.1], [0, 1, 2, 3])["k"] == 3  # SE <= 0 dropped


def test_meta_regression_is_nearly_unbiased_and_the_knapp_hartung_interval_covers():
    rng = np.random.default_rng(5)
    hits_z = hits_kh = 0
    slopes = []
    reps = 400
    for _ in range(reps):
        x = np.log(rng.uniform(150, 4000, 12))
        se = rng.uniform(0.003, 0.008, 12)
        y = 0.02 * (x - 6.5) + rng.normal(0, 0.01, 12) + rng.normal(0, se)
        f = A.meta_regression(y, se, x)
        slopes.append(f["slope"])
        hits_z += f["ci95_low"] <= 0.02 <= f["ci95_high"]
        hits_kh += f["ci95_low_kh"] <= 0.02 <= f["ci95_high_kh"]
    assert np.mean(slopes) == pytest.approx(0.02, abs=0.002)
    assert hits_kh / reps > 0.92 and hits_z / reps > 0.88  # k = 12: the normal interval is a little liberal, KH is not


@pytest.fixture(scope="module")
def sized(tmp_path_factory):
    """Seven cohorts whose planted effects grow with ln(training matches); the HGB reference does not."""
    sizes = {f"s{i}": n for i, n in enumerate((150, 300, 600, 1200, 2400, 3300, 5000))}
    scale = {c: 0.35 + 0.28 * (np.log(n) - np.log(150)) for c, n in sizes.items()}
    arms = ("M00", "M10", "M01", "M11", "M11R", "M11R-shuf", "M11R-noHab", "M11RZ-noHab", "M11RZ", "HGB-cut")
    quality = {a: QUALITY[a] for a in arms}
    root, out = tmp_path_factory.mktemp("sized_runs"), tmp_path_factory.mktemp("sized_res")
    write_prediction_runs(root, list(sizes), quality, {a: RANK_QUALITY[a] for a in arms}, n_matches=40, seed=7, cohort_scale=scale,
                          near_copy=NEAR_COPY, n_train=sizes)  # fmt: skip
    res = A.aggregate(root, out, n_boot=100, seed=3, workers=2)
    return res, sizes, scale, root, out


def test_metareg_of_the_five_contrasts_is_written_and_labelled_exploratory(sized):
    res, sizes, scale, root, out = sized
    assert list(res.metareg.panel) == ["P1", "P2", "P5", "m11_m00", "m11r_vs_hgbcut"]
    assert list(res.metareg.columns) == A.METAREG_COLUMNS and (res.metareg.metric == "auc").all() and (res.metareg.k == 7).all()
    assert set(res.metareg_data.panel) == set(res.metareg.panel) and len(res.metareg_data) == 5 * 7
    for name in ("metareg.csv", "metareg_data.csv"):
        assert (out / name).stat().st_size > 0
    for name in ("metareg_bubble.pdf", "metareg_bubble.png"):
        assert (out / "figures" / name).stat().st_size > 0
    assert (out / "figures" / "metareg_bubble.pdf").read_bytes()[:4] == b"%PDF"
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["metareg"]) == set(res.metareg.panel)
    assert pd.read_csv(out / "metareg.csv").shape[0] == 5
    # P1 grows with the training size by construction: a clear positive slope per ln N
    p1 = res.metareg.set_index("panel").loc["P1"]
    assert p1.slope > 0.01 and p1.p < 0.01 and p1.ci95_low > 0 and p1.n_min == 150 and p1.n_max == 5000
    assert p1.mean_log_n == pytest.approx(np.mean(np.log(list(sizes.values()))))
    # the regression is made on the per-cohort estimates and SEs of the contrasts table, with ln(n)
    sub = res.contrasts[(res.contrasts.contrast == "phase_main") & (res.contrasts.metric == "auc") & (res.contrasts.stratum == "all")].sort_values("cohort")
    beta, beta_se, tau2, *_ = reference_metareg(sub.estimate, sub.se, np.log([sizes[c] for c in sub.cohort]))
    assert p1.slope == pytest.approx(beta[1]) and p1.se == pytest.approx(beta_se[1]) and p1.tau2 == pytest.approx(tau2)
    d = res.metareg_data[res.metareg_data.panel == "P1"].sort_values("cohort")
    assert list(d.n_train) == [sizes[c] for c in sub.cohort] and np.allclose(d.log_n, np.log(d.n_train))
    assert np.allclose(d.weight_re, 1 / (sub.se**2 + tau2)) and np.allclose(d.estimate - d.fitted, d.residual)
    # the HGB reference is flat in N while the neural model improves: M11R - HGB-cut grows with N and crosses zero in the range
    h = res.metareg.set_index("panel").loc["m11r_vs_hgbcut"]
    assert h.slope > 0 and np.isfinite(h.crossover_n) and h.crossover_n > 0
    assert h.crossover_where in ("below the observed sizes", "inside the observed sizes", "above the observed sizes")
    # the new contrast is a normal pooled secondary contrast too
    assert pooled(res, "m11r_vs_hgbcut", "auc").k == 7 and pooled(res, "m11_m00", "auc").kind == "secondary"


def test_metareg_crossover_matches_the_fitted_line(sized):
    res, *_ = sized
    for r in res.metareg.itertuples():
        if np.isfinite(r.crossover_n):
            line = r.intercept_at_mean_log_n + r.slope * (np.log(r.crossover_n) - r.mean_log_n)
            assert line == pytest.approx(0.0, abs=1e-9), r.panel
        if np.isfinite(r.crossover_ci_low):
            assert r.crossover_ci_low < r.crossover_n < r.crossover_ci_high
        assert r.p_kh >= r.p - 1e-12 and r.pred_at_n_max == pytest.approx(r.intercept_at_mean_log_n + r.slope * (np.log(r.n_max) - r.mean_log_n))


def test_training_sizes_from_the_runs_the_argument_and_the_population_file(tmp_path):
    sizes = {"s0": 200, "s1": 400, "s2": 800, "s3": 1600}
    arms = ("M00", "M10", "M01", "M11")
    quality = {a: QUALITY[a] for a in arms}
    root = tmp_path / "r"
    write_prediction_runs(root, list(sizes), quality, {a: RANK_QUALITY[a] for a in arms}, n_matches=30, seed=2, n_train={"s0": 200, "s1": 400})  # fmt: skip
    res = A.aggregate(root, None, n_boot=30, seed=1, workers=1, figures=False)
    assert (res.metareg_data.cohort.unique() == ["s0", "s1"]).all() or set(res.metareg_data.cohort) == {"s0", "s1"}
    assert res.metareg.empty or (res.metareg.k.max() <= 2 and res.metareg.slope.isna().all())  # two cohorts: no slope
    msgs = " ".join(res.issues.message)
    assert "s2" in msgs and "training size unknown" in msgs
    res2 = A.aggregate(root, None, n_boot=30, seed=1, workers=1, figures=False, train_matches=sizes)  # the argument fills the gaps
    assert set(res2.metareg_data.cohort) == set(sizes) and (res2.metareg.k == 4).all()
    # the CLI reads population.json of a dump directory
    dumps = tmp_path / "dumps"
    dumps.mkdir()
    (dumps / "population.json").write_text(json.dumps({"cohorts": {c: {"status": "kept", "after": {"train": n}} for c, n in sizes.items()}}))
    rc = A.main(["--runs-dir", str(root), "--out-dir", str(tmp_path / "o"), "--n-boot", "30", "--workers", "1", "--no-figures", "--dump-dir", str(dumps)])
    assert rc == 0 and (pd.read_csv(tmp_path / "o" / "metareg.csv").k == 4).all()


def test_runs_that_disagree_on_the_training_size_are_reported(tmp_path):
    arms = ("M00", "M10", "M01", "M11")
    quality = {a: QUALITY[a] for a in arms}
    root = tmp_path / "r"
    write_prediction_runs(root, ["s0", "s1", "s2"], quality, {a: RANK_QUALITY[a] for a in arms}, n_matches=30, seed=2,
                          n_train={"s0": 200, "s1": 400, "s2": 800})  # fmt: skip
    meta = root / "s1" / "M10" / "seed42" / "metrics.json"
    meta.write_text(json.dumps({**json.loads(meta.read_text()), "n_matches": {"train": 777}}))
    res = A.aggregate(root, None, n_boot=20, seed=1, workers=1, figures=False)
    assert "different training sizes" in " ".join(res.issues.message) and "s1" in " ".join(res.issues.message)
    assert int(res.metareg_data[res.metareg_data.cohort == "s1"].n_train.iloc[0]) == 400  # the majority (the mode: 3 of 4 arms)


def test_the_cli_prints_the_metareg_table(tmp_path, capsys):
    sizes = {"s0": 200, "s1": 400, "s2": 800, "s3": 1600, "s4": 3200}
    arms = ("M00", "M10", "M01", "M11", "HGB-cut", "M11R")
    quality = {a: QUALITY[a] for a in arms}
    scale = {c: 0.4 + 0.3 * i for i, c in enumerate(sizes)}
    root = tmp_path / "r"
    write_prediction_runs(root, list(sizes), quality, {a: RANK_QUALITY[a] for a in arms}, n_matches=30, seed=4, cohort_scale=scale, n_train=sizes)  # fmt: skip
    assert A.main(["--runs-dir", str(root), "--out-dir", str(tmp_path / "o"), "--n-boot", "40", "--workers", "1", "--no-figures"]) == 0
    text = capsys.readouterr().out
    assert "meta-regression" in text and "P1" in text and "m11r_vs_hgbcut" in text and "crossover" in text
    assert not (tmp_path / "o" / "figures" / "metareg_bubble.png").exists()  # --no-figures


# --------------------------------------------------------------------------- external baselines (design doc 4.3)
EXTERNAL_QUALITY = {
    "SAGONG-xgb": 0.75, "SAGONG-rf": 0.7, "SAGONG-lr": 0.6, "KRAFTON-tf": 0.85, "KRAFTON-tf-h0": 0.95, "KRAFTON-mlp": 0.8, "KRAFTON-tf-ourfeat": 1.2,
    "KRAFTON-tf-ourfeat-h0": 1.3, "KRAFTON-tf-ourfeat-bce": 1.1, "KRAFTON-tf-ourfeat-hazard": 1.25, "SAGONG-hgb": 0.72, "HGB-kraftonfeat": 0.95,
}  # fmt: skip  (K2 = KRAFTON-tf-ourfeat, K2H0 = -h0, K3a = -bce, K3 = -hazard; KRAFTON-tf-h0 = K1 scores + a phase baseline; SAGONG-hgb = HGB-cut's learner on Sagong's 12 inputs)
EXTERNAL_RANK_QUALITY = {
    "KRAFTON-tf": 1.3, "KRAFTON-tf-h0": 1.3, "KRAFTON-mlp": 1.1, "KRAFTON-tf-ourfeat": 1.6, "KRAFTON-tf-ourfeat-h0": 1.6, "KRAFTON-tf-ourfeat-bce": 1.5,
    "KRAFTON-tf-ourfeat-hazard": 1.7,
}  # fmt: skip
NORANK_QUALITY = {"M01-norank": 0.88, "M11R-norank": 1.46}  # d' (M01 0.9, M11R 1.5): the PTR arms trained with the hazard BCE alone
NORANK_RANK_QUALITY = {"M01-norank": 1.0, "M11R-norank": 1.4}  # the rank head is untrained: the rank score is the hazard head's ranking
EXT_FORMULAS = {
    "m11r_vs_sagong_xgb": {"M11R": 1, "SAGONG-xgb": -1}, "hgbcut_vs_sagong_xgb": {"HGB-cut": 1, "SAGONG-xgb": -1},
    "m11r_vs_krafton_tf": {"M11R": 1, "KRAFTON-tf": -1}, "hgbcut_vs_krafton_tf": {"HGB-cut": 1, "KRAFTON-tf": -1},
    "krafton_tf_mlp": {"KRAFTON-tf": 1, "KRAFTON-mlp": -1},
    "m11r_vs_sagong_rf": {"M11R": 1, "SAGONG-rf": -1}, "m11r_vs_sagong_lr": {"M11R": 1, "SAGONG-lr": -1},
    "m11r_vs_krafton_mlp": {"M11R": 1, "KRAFTON-mlp": -1}, "m11r_vs_krafton_tf_ourfeat": {"M11R": 1, "KRAFTON-tf-ourfeat": -1},
    "krafton_ourfeat_input": {"KRAFTON-tf-ourfeat": 1, "KRAFTON-tf": -1},  # the input effect: their backbone and loss fixed
    "m01_vs_krafton_tf_ourfeat": {"M01": 1, "KRAFTON-tf-ourfeat": -1},  # our model and loss vs theirs on the same features
    "hgbcut_vs_hgb_kraftonfeat": {"HGB-cut": 1, "HGB-kraftonfeat": -1},  # C1 grid, input effect with the GBDT fixed
    "hgb_kraftonfeat_vs_krafton_tf": {"HGB-kraftonfeat": 1, "KRAFTON-tf": -1},  # bundled: GBDT vs their pipeline (learner and more)
    "hgbcut_vs_sagong_hgb": {"HGB-cut": 1, "SAGONG-hgb": -1},  # the clean input effect: same HGB, rows and labels, Sagong's 12 inputs vs ours
    "sagong_hgb_vs_sagong_xgb": {"SAGONG-hgb": 1, "SAGONG-xgb": -1},  # the learner check on Sagong's inputs
    "k3a_vs_k2": {"KRAFTON-tf-ourfeat-bce": 1, "KRAFTON-tf-ourfeat": -1},  # K3 ladder: the objective (BCE vs Cox)
    "k3_vs_k3a": {"KRAFTON-tf-ourfeat-hazard": 1, "KRAFTON-tf-ourfeat-bce": -1},  # the phase baseline
    "k3_vs_k2": {"KRAFTON-tf-ourfeat-hazard": 1, "KRAFTON-tf-ourfeat": -1},  # the formulation in total
    "m01_vs_k3": {"M01": 1, "KRAFTON-tf-ourfeat-hazard": -1},  # the residual: our architecture vs theirs, same formulation and features
    "k2h0_vs_k2": {"KRAFTON-tf-ourfeat-h0": 1, "KRAFTON-tf-ourfeat": -1},  # (c1) the same Cox scores plus a training-set phase baseline
    "k3a_vs_k2h0": {"KRAFTON-tf-ourfeat-bce": 1, "KRAFTON-tf-ourfeat-h0": -1},  # (c2) the objective with the baseline held
    "m11r_vs_krafton_tf_h0": {"M11R": 1, "KRAFTON-tf-h0": -1}, "hgbcut_vs_krafton_tf_h0": {"HGB-cut": 1, "KRAFTON-tf-h0": -1},  # RQ2 additions
    "hgbcut_vs_sagong_rf": {"HGB-cut": 1, "SAGONG-rf": -1}, "hgbcut_vs_sagong_lr": {"HGB-cut": 1, "SAGONG-lr": -1},  # RQ2: HGB-cut vs every SAGONG model
}  # fmt: skip
EXT_TABLE_ORDER = [  # the rows of tab_external_contrasts, top to bottom, with the blocks separated by a rule
    ("M11R $-$ SAGONG-xgb", "m11r_vs_sagong_xgb"), ("M11R $-$ SAGONG-rf", "m11r_vs_sagong_rf"), ("M11R $-$ SAGONG-lr", "m11r_vs_sagong_lr"),
    ("HGB-cut $-$ SAGONG-xgb (bundled)", "hgbcut_vs_sagong_xgb"), ("HGB-cut $-$ SAGONG-rf (bundled)", "hgbcut_vs_sagong_rf"),
    ("HGB-cut $-$ SAGONG-lr (bundled)", "hgbcut_vs_sagong_lr"),
    ("HGB-cut $-$ SAGONG-hgb (input)", "hgbcut_vs_sagong_hgb"), ("SAGONG-hgb $-$ SAGONG-xgb (learner check)", "sagong_hgb_vs_sagong_xgb"), None,
    ("M11R $-$ KRAFTON-tf", "m11r_vs_krafton_tf"), ("M11R $-$ KRAFTON-tf-h0", "m11r_vs_krafton_tf_h0"), ("M11R $-$ KRAFTON-mlp", "m11r_vs_krafton_mlp"),
    ("M11R $-$ KRAFTON-tf-ourfeat", "m11r_vs_krafton_tf_ourfeat"), ("HGB-cut $-$ KRAFTON-tf", "hgbcut_vs_krafton_tf"),
    ("HGB-cut $-$ KRAFTON-tf-h0", "hgbcut_vs_krafton_tf_h0"), None,
    ("KRAFTON-tf $-$ KRAFTON-mlp", "krafton_tf_mlp"), ("KRAFTON-tf-ourfeat $-$ KRAFTON-tf (bundled)", "krafton_ourfeat_input"),
    ("M01 $-$ KRAFTON-tf-ourfeat (bundled)", "m01_vs_krafton_tf_ourfeat"), None,
    ("HGB-cut $-$ HGB-kraftonfeat (input)", "hgbcut_vs_hgb_kraftonfeat"), ("HGB-kraftonfeat $-$ KRAFTON-tf (bundled)", "hgb_kraftonfeat_vs_krafton_tf"), None,
    ("K2+H0 $-$ K2 (phase baseline for Cox)", "k2h0_vs_k2"), ("K3a $-$ K2+H0 (objective, baseline held)", "k3a_vs_k2h0"),
    ("K3a $-$ K2 (objective)", "k3a_vs_k2"), ("K3 $-$ K3a (phase baseline)", "k3_vs_k3a"), ("K3 $-$ K2 (total)", "k3_vs_k2"),
    ("M01 $-$ K3 (residual)", "m01_vs_k3"),
]  # fmt: skip
EXT_PAIRS = {
    "m11r_vs_sagong_xgb": ("M11R", "SAGONG-xgb"), "hgbcut_vs_sagong_xgb": ("HGB-cut", "SAGONG-xgb"),
    "m11r_vs_krafton_tf": ("M11R", "KRAFTON-tf"), "hgbcut_vs_krafton_tf": ("HGB-cut", "KRAFTON-tf"),
    "krafton_tf_mlp": ("KRAFTON-tf", "KRAFTON-mlp"), "m11r_vs_sagong_rf": ("M11R", "SAGONG-rf"), "m11r_vs_sagong_lr": ("M11R", "SAGONG-lr"),
    "m11r_vs_krafton_mlp": ("M11R", "KRAFTON-mlp"), "m11r_vs_krafton_tf_ourfeat": ("M11R", "KRAFTON-tf-ourfeat"),
    "krafton_ourfeat_input": ("KRAFTON-tf-ourfeat", "KRAFTON-tf"), "m01_vs_krafton_tf_ourfeat": ("M01", "KRAFTON-tf-ourfeat"),
    "hgbcut_vs_hgb_kraftonfeat": ("HGB-cut", "HGB-kraftonfeat"), "hgb_kraftonfeat_vs_krafton_tf": ("HGB-kraftonfeat", "KRAFTON-tf"),
    "hgbcut_vs_sagong_hgb": ("HGB-cut", "SAGONG-hgb"), "sagong_hgb_vs_sagong_xgb": ("SAGONG-hgb", "SAGONG-xgb"),
    "k3a_vs_k2": ("KRAFTON-tf-ourfeat-bce", "KRAFTON-tf-ourfeat"), "k3_vs_k3a": ("KRAFTON-tf-ourfeat-hazard", "KRAFTON-tf-ourfeat-bce"),
    "k3_vs_k2": ("KRAFTON-tf-ourfeat-hazard", "KRAFTON-tf-ourfeat"), "m01_vs_k3": ("M01", "KRAFTON-tf-ourfeat-hazard"),
    "k2h0_vs_k2": ("KRAFTON-tf-ourfeat-h0", "KRAFTON-tf-ourfeat"), "k3a_vs_k2h0": ("KRAFTON-tf-ourfeat-bce", "KRAFTON-tf-ourfeat-h0"),
    "m11r_vs_krafton_tf_h0": ("M11R", "KRAFTON-tf-h0"), "hgbcut_vs_krafton_tf_h0": ("HGB-cut", "KRAFTON-tf-h0"),
    "hgbcut_vs_sagong_rf": ("HGB-cut", "SAGONG-rf"), "hgbcut_vs_sagong_lr": ("HGB-cut", "SAGONG-lr"),
}  # fmt: skip


def external_runs(tmp_path, cohorts=COHORTS[:3], arms=tuple(EXTERNAL_QUALITY), skip=frozenset(), norank=False):
    quality = {**{a: q for a, q in QUALITY.items() if a not in EXPLORATORY}, **{a: EXTERNAL_QUALITY[a] for a in arms}, **(NORANK_QUALITY if norank else {})}
    rank_quality = {**{a: q for a, q in RANK_QUALITY.items() if a in quality}, **{a: q for a, q in EXTERNAL_RANK_QUALITY.items() if a in quality},
                    **{a: q for a, q in NORANK_RANK_QUALITY.items() if a in quality}}  # fmt: skip
    return make_runs(tmp_path, cohorts=cohorts, quality=quality, rank_quality=rank_quality, skip=set(skip), seeds={})


def test_external_arms_and_their_exploratory_contrasts_are_registered():
    assert A.EXTERNAL_ARMS == (
        "SAGONG-xgb", "SAGONG-rf", "SAGONG-lr", "KRAFTON-tf", "KRAFTON-tf-h0", "KRAFTON-mlp", "KRAFTON-tf-ourfeat", "KRAFTON-tf-ourfeat-h0",
        "KRAFTON-tf-ourfeat-bce", "KRAFTON-tf-ourfeat-hazard", "SAGONG-hgb", "HGB-kraftonfeat",
    )  # fmt: skip  (the H0 arms follow their parents; the K3 ladder follows K2H0; the two arms that run OUR learner on their inputs close the list)
    assert A.ARM_ORDER[-12:] == A.EXTERNAL_ARMS  # after the HGB references and the union arms
    assert "HGB-kraftonfeat" not in A.HGB_ARMS and "SAGONG-hgb" not in A.HGB_ARMS  # not HGB-* variants of the main matrix
    assert set(EXT_FORMULAS) == set(EXT_PAIRS) and len(EXT_FORMULAS) == 25
    for name, coefs in EXT_FORMULAS.items():
        f = A.FORMULAS[name]
        assert f.coefs == coefs and f.optional and f.pooled and f.group == "external"
        assert name not in {k for k, _ in A.PRIMARY.values()}  # never part of the Holm family
    ext_rows = [row for row in A.EXTERNAL_CONTRAST_ROWS if row[2] != "union"]  # the union block follows them (design doc 4.6)
    assert [name for _, name, *_ in ext_rows] == [row[1] for row in EXT_TABLE_ORDER if row]  # the order of the table
    assert {name for _, name, *_ in ext_rows} == set(EXT_FORMULAS)
    assert {name: tag for _, name, _, tag in ext_rows if tag} == {
        "krafton_ourfeat_input": "bundled", "m01_vs_krafton_tf_ourfeat": "bundled", "hgbcut_vs_sagong_xgb": "bundled",
        "hgbcut_vs_sagong_rf": "bundled", "hgbcut_vs_sagong_lr": "bundled",
        "hgbcut_vs_hgb_kraftonfeat": "input", "hgb_kraftonfeat_vs_krafton_tf": "bundled",
        "hgbcut_vs_sagong_hgb": "input", "sagong_hgb_vs_sagong_xgb": "learner check",
        "k2h0_vs_k2": "phase baseline for Cox", "k3a_vs_k2h0": "objective, baseline held",
        "k3a_vs_k2": "objective", "k3_vs_k3a": "phase baseline", "k3_vs_k2": "total", "m01_vs_k3": "residual"}  # fmt: skip
    assert [g for _, _, g, _ in ext_rows] == ["sagong"] * 8 + ["krafton"] * 6 + ["ablation"] * 3 + ["grid"] * 2 + ["formulation"] * 6
    # arm order puts the new arm after KRAFTON-mlp: it is listed last in the external arms table
    assert A.arm_rank("KRAFTON-tf-ourfeat") > A.arm_rank("KRAFTON-mlp") > A.arm_rank("KRAFTON-tf") > A.arm_rank("HGB-noHab")
    assert A.arm_rank("KRAFTON-tf") < A.arm_rank("KRAFTON-tf-h0") < A.arm_rank("KRAFTON-mlp") < A.arm_rank("KRAFTON-tf-ourfeat") < A.arm_rank("KRAFTON-tf-ourfeat-h0")


def test_external_runs_are_kind_external_and_the_contrasts_are_pooled_without_holm(tmp_path):
    root = external_runs(tmp_path / "r")
    res = A.aggregate(root, tmp_path / "o", n_boot=80, seed=2, workers=1, figures=False)
    inv = res.inventory
    assert set(inv[inv.arm.str.startswith(("SAGONG", "KRAFTON"))].kind) == {"external"}
    assert set(inv[inv.arm == "KRAFTON-tf-ourfeat"].kind) == {"external"} and len(inv[inv.arm == "KRAFTON-tf-ourfeat"]) == 3
    assert set(inv[inv.arm.isin(A.HGB_ARMS)].kind) == {"hgb"} and set(inv[inv.arm.str.startswith("M")].kind) == {"neural"}
    # HGB-kraftonfeat starts with "HGB" but is an external model (a GBDT on KRAFTON's inputs), listed like the other external arms
    assert set(inv[inv.arm == "HGB-kraftonfeat"].kind) == {"external"} and len(inv[inv.arm == "HGB-kraftonfeat"]) == 3
    qual = {**QUALITY, **EXTERNAL_QUALITY}
    for name, (a, b) in EXT_PAIRS.items():
        row = pooled(res, name, "auc")
        want = np.mean([auc_theory(eff(c, a, qual)) - auc_theory(eff(c, b, qual)) for c in COHORTS[:3]])
        assert row.k == 3 and row.kind == "secondary" and row.exploratory and np.isnan(row.p_holm) and row.verdict_basis == "unadjusted"
        assert row.theta == pytest.approx(want, abs=0.03), name
        assert row.primary_id == ""
        for metric in A.METRICS:  # all metrics (the seven and the within-phase AUC) are in the CSVs, in every stratum
            assert np.isfinite(pooled(res, name, metric).theta) and pooled(res, name, metric).se > 0
        assert (res.pooled.contrast == name).sum() == len(A.METRICS) * 3
    # verdicts follow the planted truth: our M11R is far better than every SAGONG model, the input effect is a gain, and our M01
    # (hazard d' 0.9) is clearly worse than their transformer on our features (d' 1.2): never a gain, never equivalent
    assert pooled(res, "m11r_vs_sagong_rf", "auc").verdict == "gain" and pooled(res, "m11r_vs_sagong_lr", "auc").verdict == "gain"
    assert pooled(res, "krafton_ourfeat_input", "auc").verdict == "gain" and pooled(res, "krafton_ourfeat_input", "auc").theta > 0
    assert pooled(res, "m01_vs_krafton_tf_ourfeat", "auc").theta < -0.02 and pooled(res, "m01_vs_krafton_tf_ourfeat", "auc").verdict == "inconclusive"
    assert pooled(res, "krafton_tf_mlp", "ap").verdict == "n/a"
    # C1 grid contrasts: GBDT fixed, ours (d' 0.8) vs KRAFTON's inputs (0.95): negative; KRAFTON inputs fixed, GBDT (0.95) vs their transformer (0.85): positive
    assert pooled(res, "hgbcut_vs_hgb_kraftonfeat", "auc").theta < -0.01 and pooled(res, "hgbcut_vs_hgb_kraftonfeat", "auc").verdict == "inconclusive"
    assert pooled(res, "hgb_kraftonfeat_vs_krafton_tf", "auc").theta > 0.005 and pooled(res, "hgb_kraftonfeat_vs_krafton_tf", "auc").verdict in ("gain", "inconclusive")
    # the contrast on identical rows is a paired bootstrap of the per-arm resamples
    sub = res.contrasts[(res.contrasts.contrast == "krafton_ourfeat_input") & (res.contrasts.metric == "auc") & (res.contrasts.stratum == "all")]
    assert (sub.se > 0).all() and len(sub) == 3 and "no external" not in " ".join(res.issues.message)
    # the interpretation contrasts are linear in the per-arm results: input effect = (M11R - tf) - (M11R - ourfeat) per cohort
    a, b, c = (res.contrasts[(res.contrasts.contrast == n) & (res.contrasts.metric == "auc") & (res.contrasts.stratum == "all")].set_index("cohort").estimate
               for n in ("krafton_ourfeat_input", "m11r_vs_krafton_tf", "m11r_vs_krafton_tf_ourfeat"))  # fmt: skip
    assert np.allclose(a, b - c, atol=1e-12)


def test_sagong_hgb_is_an_external_arm_with_a_clean_input_effect_and_a_learner_check(tmp_path):
    root = external_runs(tmp_path / "r")
    res = A.aggregate(root, tmp_path / "o", n_boot=60, seed=2, workers=1, figures=False)
    inv = res.inventory
    assert set(inv[inv.arm == "SAGONG-hgb"].kind) == {"external"} and len(inv[inv.arm == "SAGONG-hgb"]) == 3  # it is not an HGB-* variant
    assert A.arm_rank("KRAFTON-tf-ourfeat-hazard") < A.arm_rank("SAGONG-hgb") < A.arm_rank("HGB-kraftonfeat")  # the two "our learner, their inputs" arms close the list
    # clean input effect: the same HGB (settings, rows, labels), Sagong's 12 inputs vs ours; learner check: HGB vs XGBoost on Sagong's inputs
    assert A.FORMULAS["hgbcut_vs_sagong_hgb"].coefs == {"HGB-cut": 1, "SAGONG-hgb": -1}
    assert A.FORMULAS["sagong_hgb_vs_sagong_xgb"].coefs == {"SAGONG-hgb": 1, "SAGONG-xgb": -1}
    point = lambda arm: res.arm_metrics[(res.arm_metrics.arm == arm) & (res.arm_metrics.metric == "auc") & (res.arm_metrics.seed == 42) & (res.arm_metrics.stratum == "all")].set_index("cohort").value  # noqa: E731
    assert np.allclose(cohort_estimates(res, "hgbcut_vs_sagong_hgb"), point("HGB-cut") - point("SAGONG-hgb"), atol=1e-12)
    assert np.allclose(cohort_estimates(res, "sagong_hgb_vs_sagong_xgb"), point("SAGONG-hgb") - point("SAGONG-xgb"), atol=1e-12)
    for metric in A.METRICS:  # the old contrast HGB-cut - SAGONG-xgb is exactly input + learner (bundled), in every cohort
        a, b, c = (cohort_estimates(res, n, metric) for n in ("hgbcut_vs_sagong_hgb", "sagong_hgb_vs_sagong_xgb", "hgbcut_vs_sagong_xgb"))
        assert np.allclose(a + b, c, atol=1e-12), metric
    for name in ("hgbcut_vs_sagong_hgb", "sagong_hgb_vs_sagong_xgb"):
        row = pooled(res, name, "auc")
        assert row.k == 3 and row.kind == "secondary" and row.exploratory and np.isnan(row.p_holm) and row.verdict_basis == "unadjusted"
        assert (res.pooled.contrast == name).sum() == len(A.METRICS) * 3
    assert pooled(res, "hgbcut_vs_sagong_hgb", "auc").theta > 0 and pooled(res, "sagong_hgb_vs_sagong_xgb", "auc").theta < 0.01  # d': 0.8 vs 0.72; 0.72 vs 0.75
    # without SAGONG-hgb (the Sagong agent has not delivered it) the two contrasts do not exist and nothing warns
    old = external_runs(tmp_path / "r0", arms=("SAGONG-xgb",))
    res0 = A.aggregate(old, None, n_boot=30, seed=1, workers=1, figures=False)
    assert all((res0.pooled.contrast == n).sum() == 0 for n in ("hgbcut_vs_sagong_hgb", "sagong_hgb_vs_sagong_xgb"))
    assert pooled(res0, "hgbcut_vs_sagong_xgb", "auc").k == 3 and "sagong_hgb" not in " ".join(res0.issues.message).lower()


def test_adding_external_arms_changes_none_of_the_primary_results(tmp_path):
    with_ext = A.aggregate(external_runs(tmp_path / "a"), tmp_path / "oa", n_boot=60, seed=4, workers=1, figures=False)
    without = A.aggregate(external_runs(tmp_path / "b", arms=()), tmp_path / "ob", n_boot=60, seed=4, workers=1, figures=False)
    key = ["contrast", "metric", "stratum"]
    a = with_ext.pooled.merge(without.pooled, on=key, suffixes=("_a", "_b"))
    assert len(a) == len(without.pooled) > 0
    for col in ("theta", "se", "ci95_low", "ci95_high", "tau2", "p_one_sided", "p_holm"):
        assert np.array_equal(a[col + "_a"].to_numpy(), a[col + "_b"].to_numpy(), equal_nan=True), col  # bitwise, shared resamples
    pa = with_ext.arm_metrics.merge(without.arm_metrics, on=["cohort", "arm", "seed", "stratum", "metric"], suffixes=("_a", "_b"))
    assert np.array_equal(pa.value_a.to_numpy(), pa.value_b.to_numpy())
    assert with_ext.summary["primary_results"] == without.summary["primary_results"]


def test_external_tables_follow_the_arms_table_layout(tmp_path):
    root = external_runs(tmp_path / "r")
    res = A.aggregate(root, tmp_path / "o", n_boot=60, seed=2, workers=1, figures=False)
    spec, rows = parse_tabular((tmp_path / "o" / "tables" / "tab_external.tex").read_text())
    arms_spec, arms_rows = parse_tabular((tmp_path / "o" / "tables" / "tab_arms_metrics.tex").read_text())
    raw = (tmp_path / "o" / "tables" / "tab_external.tex").read_text()
    assert spec == arms_spec and "\\begin{tabular}{@{}lccccccc@{}}" in raw
    assert rows[0][1:] == arms_rows[0][1:] and rows[1][1:] == arms_rows[1][1:]  # the two header rows of the arms table
    body = [r for r in rows if r[0].startswith(("HGB", "M11R", "SAGONG", "KRAFTON"))]
    assert [r[0] for r in body] == [
        "HGB-cut (reference)", "M11R (reference)", "SAGONG-xgb", "SAGONG-rf", "SAGONG-lr", "KRAFTON-tf", "KRAFTON-tf-h0", "KRAFTON-mlp",
        "KRAFTON-tf-ourfeat", "KRAFTON-tf-ourfeat-h0", "KRAFTON-tf-ourfeat-bce", "KRAFTON-tf-ourfeat-hazard", "SAGONG-hgb", "HGB-kraftonfeat",
    ]  # fmt: skip
    m = res.arm_means
    for r in body:
        arm = r[0].replace(" (reference)", "")
        for j, metric in enumerate(("auc", "ap", "nll", "brier", "spearman", "top5_auc", "cindex"), start=1):
            val = float(m[(m.arm == arm) & (m.metric == metric) & (m.stratum == "all")].iloc[0]["mean"])
            assert r[j] == A.fmt_value(val), (arm, metric)
    assert "generated by model_stgcn.ptr.aggregate" in raw and "\\midrule" in raw
    # the main arms table does not list the external models (it follows the paper skeleton)
    assert not any(r[0].startswith(("SAGONG", "KRAFTON")) for r in arms_rows)
    # contrasts table: eleven rows in three blocks separated by a rule, tags on the two interpretation rows
    path = tmp_path / "o" / "tables" / "tab_external_contrasts.tex"
    text = path.read_text()
    spec2, rows2 = parse_tabular(text)
    assert "\\begin{tabular}{@{}l c c c c@{}}" in text and rows2[0][0] == "Contrast"
    assert [r[0] for r in rows2[1:]] == [row[0] for row in EXT_TABLE_ORDER if row]
    body_lines = text.split("\\midrule", 1)[1].split("\\bottomrule")[0]  # between the header rule and the bottom rule
    blocks = [[ln for ln in blk.splitlines() if ln.strip().endswith("\\\\")] for blk in body_lines.split("\\midrule")]
    assert [len(b) for b in blocks] == [8, 6, 3, 2, 6]  # SAGONG, KRAFTON, KRAFTON ablations, the C1 grid effects, the K3 ladder (with the H0 steps)
    for tag in ("(input)", "(bundled)", "(learner check)"):
        assert tag in text  # tags explained in the header comment (and used by rows)
    assert "(learner)" not in text and "(model)" not in text  # the old tags are gone: those rows are bundled, not single-factor effects
    assert text.count("(bundled)") >= 5  # four bundled rows and the legend
    for needle in ("a clean single-factor input effect", "(learner check) = SAGONG-hgb - SAGONG-xgb", "(bundled) = more than one factor differs",
                   "stay out of tab_decomp.tex", "HGB-cut - SAGONG-xgb (input and learner at once)",
                   "M01 - KRAFTON-tf-ourfeat (= objective + phase baseline + residual of the K ladder below)",
                   "(objective) = K3a - K2", "(phase baseline) = K3 - K3a", "(total) = K3 - K2", "(residual) = M01 - K3"):  # fmt: skip
        assert needle in text, needle
    assert text.count("%") >= 6
    flat = [row for row in EXT_TABLE_ORDER if row]
    for r, (label, name) in zip(rows2[1:], flat):
        for cell, metric in zip(r[1:], ("auc", "spearman", "top5_auc", "cindex")):
            p = pooled(res, name, metric)
            assert cell == A.fmt_delta(p.theta, p.ci95_low, p.ci95_high), (name, metric)


def test_external_tables_only_list_what_was_run_and_flag_partial_coverage(tmp_path):
    only_sagong = external_runs(tmp_path / "r", arms=("SAGONG-xgb", "SAGONG-rf", "SAGONG-lr"), skip={("c_gamma", "SAGONG-xgb"), ("c_gamma", "SAGONG-rf"), ("c_gamma", "SAGONG-lr")})
    res = A.aggregate(only_sagong, tmp_path / "o", n_boot=40, seed=1, workers=1, figures=False)
    text = (tmp_path / "o" / "tables" / "tab_external.tex").read_text()
    assert "SAGONG-xgb (2 cohorts)" in text and "KRAFTON" not in text and "HGB-cut (reference)" in text
    _, rows = parse_tabular((tmp_path / "o" / "tables" / "tab_external_contrasts.tex").read_text())
    labels = [r[0] for r in rows[1:]]
    assert labels == ["M11R $-$ SAGONG-xgb (2 cohorts)", "M11R $-$ SAGONG-rf (2 cohorts)", "M11R $-$ SAGONG-lr (2 cohorts)",
                      "HGB-cut $-$ SAGONG-xgb (bundled, 2 cohorts)", "HGB-cut $-$ SAGONG-rf (bundled, 2 cohorts)",
                      "HGB-cut $-$ SAGONG-lr (bundled, 2 cohorts)"]  # no KRAFTON rows: not run  # fmt: skip
    assert pooled(res, "m11r_vs_sagong_xgb", "auc").k == 2 and (res.pooled.contrast == "m11r_vs_krafton_tf").sum() == 0
    for gone in ("m11r_vs_krafton_tf_ourfeat", "krafton_ourfeat_input", "m01_vs_krafton_tf_ourfeat", "m11r_vs_krafton_mlp"):
        assert (res.pooled.contrast == gone).sum() == 0
    msgs = " ".join(res.issues.message)
    assert "KRAFTON" not in msgs and "SAGONG" not in msgs  # optional external arms never raise warnings
    # no external arm at all: no external tables (and a stale file is removed)
    plain = external_runs(tmp_path / "p", arms=())
    out = tmp_path / "o2"
    (out / "tables").mkdir(parents=True)
    (out / "tables" / "tab_external.tex").write_text("stale")
    (out / "tables" / "tab_grid.tex").write_text("stale")
    A.aggregate(plain, out, n_boot=30, seed=1, workers=1, figures=False)
    assert not (out / "tables" / "tab_external.tex").exists() and not (out / "tables" / "tab_external_contrasts.tex").exists()
    assert not (out / "tables" / "tab_grid.tex").exists()


def test_tags_and_partial_coverage_share_one_parenthesis_and_missing_arms_drop_only_their_rows(tmp_path):
    # KRAFTON-tf-ourfeat is missing in c_gamma only; KRAFTON-tf and -mlp are complete
    root = external_runs(tmp_path / "r", skip={("c_gamma", "KRAFTON-tf-ourfeat")})
    res = A.aggregate(root, tmp_path / "o", n_boot=40, seed=1, workers=1, figures=False)
    _, rows = parse_tabular((tmp_path / "o" / "tables" / "tab_external_contrasts.tex").read_text())
    labels = {r[0] for r in rows[1:]}
    assert "KRAFTON-tf-ourfeat $-$ KRAFTON-tf (bundled, 2 cohorts)" in labels
    assert "M01 $-$ KRAFTON-tf-ourfeat (bundled, 2 cohorts)" in labels and "M11R $-$ KRAFTON-tf-ourfeat (2 cohorts)" in labels
    assert "KRAFTON-tf $-$ KRAFTON-mlp" in labels and "M11R $-$ KRAFTON-mlp" in labels  # complete contrasts carry no count
    assert pooled(res, "krafton_ourfeat_input", "auc").k == 2 and pooled(res, "m11r_vs_krafton_tf", "auc").k == 3
    text = (tmp_path / "o" / "tables" / "tab_external.tex").read_text()
    assert "KRAFTON-tf-ourfeat (2 cohorts)" in text
    assert not any("))" in label or ") (" in label for label in labels)


def grid_cell(res, arm):
    row = res.arm_means[(res.arm_means.arm == arm) & (res.arm_means.metric == "auc") & (res.arm_means.stratum == "all")].iloc[0]
    return f"{A.fmt_value(float(row['mean']))} ({int(row.n_cohorts)} coh.)"


def test_the_c1_grid_table_has_inputs_in_rows_learners_in_columns_and_cohort_counts(tmp_path):
    root = external_runs(tmp_path / "r", skip={("c_gamma", "KRAFTON-tf-ourfeat")})
    res = A.aggregate(root, tmp_path / "o", n_boot=40, seed=2, workers=1, figures=False)
    text = (tmp_path / "o" / "tables" / "tab_grid.tex").read_text()
    spec, rows = parse_tabular(text)
    assert "\\begin{tabular}{@{}l c c c@{}}" in text and "generated by model_stgcn.ptr.aggregate" in text
    assert rows[0] == ["Inputs $\\backslash$ learner", "GBDT", "KRAFTON transformer", "PTR-Former"]
    assert [r[0] for r in rows[1:]] == ["Sagong features", "KRAFTON features", "Our representation"]
    # the grid of the paper focus: Sagong x GBDT = SAGONG-hgb (the HGB-cut learner on Sagong's inputs, so the GBDT column is learner-clean),
    # KRAFTON x GBDT = HGB-kraftonfeat, KRAFTON x transformer = KRAFTON-tf, ours x GBDT = HGB-cut,
    # ours x transformer = KRAFTON-tf-ourfeat (K2), ours x PTR-Former = M11R; the other cells are empty
    assert A.GRID_CELLS[("sagong", "gbdt")] == "SAGONG-hgb"
    assert rows[1][1:] == [grid_cell(res, "SAGONG-hgb"), "---", "---"]
    assert rows[2][1:] == [grid_cell(res, "HGB-kraftonfeat"), grid_cell(res, "KRAFTON-tf"), "---"]
    assert rows[3][1:] == [grid_cell(res, "HGB-cut"), grid_cell(res, "KRAFTON-tf-ourfeat"), grid_cell(res, "M11R")]
    assert rows[3][2].endswith("(2 coh.)") and rows[3][1].endswith("(3 coh.)") and rows[2][2].endswith("(3 coh.)")  # K2 misses c_gamma
    for arm in ("SAGONG-hgb", "HGB-kraftonfeat", "KRAFTON-tf", "HGB-cut", "KRAFTON-tf-ourfeat", "M11R"):
        assert arm in text  # the cells are named in the header comment
    assert "GBDT column = HistGradientBoosting with the HGB-cut settings in every row" in text  # why the column is learner-clean
    # the number in a cell is the unweighted mean of the per-cohort test AUCs, the same as arm_metrics_mean.csv
    per = res.arm_metrics[(res.arm_metrics.arm == "HGB-cut") & (res.arm_metrics.metric == "auc") & (res.arm_metrics.seed == 42) & (res.arm_metrics.stratum == "all")]
    assert rows[3][1].startswith(A.fmt_value(float(per.value.mean())))


def test_grid_cells_without_a_run_stay_empty_and_the_table_needs_an_external_arm(tmp_path):
    only = external_runs(tmp_path / "r", arms=("SAGONG-hgb", "KRAFTON-tf"))  # K2 and HGB-kraftonfeat not produced yet
    res = A.aggregate(only, tmp_path / "o", n_boot=30, seed=1, workers=1, figures=False)
    _, rows = parse_tabular((tmp_path / "o" / "tables" / "tab_grid.tex").read_text())
    assert rows[1][1:] == [grid_cell(res, "SAGONG-hgb"), "---", "---"]
    assert rows[2][1:] == ["---", grid_cell(res, "KRAFTON-tf"), "---"]
    assert rows[3][1:] == [grid_cell(res, "HGB-cut"), "---", grid_cell(res, "M11R")]
    # the two grid contrasts need HGB-kraftonfeat: not computed, no rows, no warnings
    assert (res.pooled.contrast == "hgbcut_vs_hgb_kraftonfeat").sum() == 0 and (res.pooled.contrast == "hgb_kraftonfeat_vs_krafton_tf").sum() == 0
    assert "kraftonfeat" not in " ".join(res.issues.message)
    # Sagong's own XGBoost does not stand in for the cell (a cell is one model): with SAGONG-xgb only, the Sagong row is empty
    only_xgb = external_runs(tmp_path / "rx", arms=("SAGONG-xgb", "KRAFTON-tf"))
    A.aggregate(only_xgb, tmp_path / "ox", n_boot=30, seed=1, workers=1, figures=False)
    _, rows_x = parse_tabular((tmp_path / "ox" / "tables" / "tab_grid.tex").read_text())
    assert rows_x[1][1:] == ["---", "---", "---"]
    assert rows_x[2][1] == "---" and rows_x[2][2].endswith("(3 coh.)") and rows_x[2][3] == "---"  # the KRAFTON transformer cell is unaffected
    # with HGB-kraftonfeat alone (a GBDT on KRAFTON's inputs) the KRAFTON row gets its GBDT cell
    only2 = external_runs(tmp_path / "r2", arms=("HGB-kraftonfeat",))
    res2 = A.aggregate(only2, tmp_path / "o2", n_boot=30, seed=1, workers=1, figures=False)
    _, rows2 = parse_tabular((tmp_path / "o2" / "tables" / "tab_grid.tex").read_text())
    assert rows2[2][1:] == [grid_cell(res2, "HGB-kraftonfeat"), "---", "---"] and pooled(res2, "hgbcut_vs_hgb_kraftonfeat", "auc").k == 3
    assert (res2.pooled.contrast == "hgb_kraftonfeat_vs_krafton_tf").sum() == 0  # needs KRAFTON-tf too


def test_the_two_grid_contrasts_are_the_input_and_the_learner_effect(tmp_path):
    root = external_runs(tmp_path / "r")
    res = A.aggregate(root, tmp_path / "o", n_boot=60, seed=3, workers=1, figures=False)
    est = lambda name: res.contrasts[(res.contrasts.contrast == name) & (res.contrasts.metric == "auc") & (res.contrasts.stratum == "all")].set_index("cohort").estimate  # noqa: E731
    point = lambda arm: res.arm_metrics[(res.arm_metrics.arm == arm) & (res.arm_metrics.metric == "auc") & (res.arm_metrics.seed == 42) & (res.arm_metrics.stratum == "all")].set_index("cohort").value  # noqa: E731
    # input effect (GBDT fixed) = HGB-cut - HGB-kraftonfeat; learner effect (KRAFTON inputs fixed) = HGB-kraftonfeat - KRAFTON-tf
    assert np.allclose(est("hgbcut_vs_hgb_kraftonfeat"), point("HGB-cut") - point("HGB-kraftonfeat"), atol=1e-12)
    assert np.allclose(est("hgb_kraftonfeat_vs_krafton_tf"), point("HGB-kraftonfeat") - point("KRAFTON-tf"), atol=1e-12)
    # together with the existing contrast they close the loop: (ours - KRAFTON inputs | GBDT) + (KRAFTON GBDT - KRAFTON tf) = HGB-cut - KRAFTON-tf
    assert np.allclose(est("hgbcut_vs_hgb_kraftonfeat") + est("hgb_kraftonfeat_vs_krafton_tf"), est("hgbcut_vs_krafton_tf"), atol=1e-12)
    for name in ("hgbcut_vs_hgb_kraftonfeat", "hgb_kraftonfeat_vs_krafton_tf"):
        row = pooled(res, name, "auc")
        assert row.kind == "secondary" and row.exploratory and np.isnan(row.p_holm) and row.verdict_basis == "unadjusted" and row.k == 3


def test_an_external_run_with_other_rows_is_reported_or_rejected(tmp_path):
    root = external_runs(tmp_path / "r", cohorts=COHORTS[:2])
    path = root / "c_beta" / "KRAFTON-tf-ourfeat" / "seed42" / "pred_test.csv"
    df = pd.read_csv(path)
    df.iloc[2:].to_csv(path, index=False)  # the baseline scored fewer rows
    with pytest.raises(ValueError, match=r"c_beta.*KRAFTON-tf-ourfeat|KRAFTON-tf-ourfeat.*c_beta"):
        A.aggregate(root, tmp_path / "o", n_boot=20, workers=1, figures=False)
    res = A.aggregate(root, tmp_path / "o2", n_boot=30, seed=1, workers=1, figures=False, skip_misaligned=True)
    assert "KRAFTON-tf-ourfeat" in " ".join(res.issues.message) and pooled(res, "m11r_vs_krafton_tf_ourfeat", "auc").k == 1
    assert pooled(res, "m11r_vs_krafton_tf", "auc").k == 2 and pooled(res, "phase_main", "auc").k == 2  # nothing else is affected


# --------------------------------------------------------------------------- forest plot layout and redrawing from the CSVs
def text_boxes(fig):
    """Pixel boxes of everything written inside the panels: annotations, visible tick labels and the x labels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    items = []
    for k, ax in enumerate(fig.axes):
        lo, hi = ax.get_xlim()
        items += [(k, "text", t.get_text(), t.get_window_extent(renderer)) for t in ax.texts]
        items += [(k, "tick", t.get_text(), t.get_window_extent(renderer)) for loc, t in zip(ax.get_xticks(), ax.get_xticklabels()) if lo <= loc <= hi and t.get_text()]
        items.append((k, "xlabel", ax.xaxis.label.get_text(), ax.xaxis.label.get_window_extent(renderer)))
    return items


def forest_inputs(res, overrides=None):
    pooled_df = res.pooled.copy()
    for (name, metric), (verdict, p_holm) in (overrides or {}).items():
        sel = (pooled_df.contrast == name) & (pooled_df.metric == metric) & (pooled_df.stratum == "all")
        pooled_df.loc[sel, "verdict"], pooled_df.loc[sel, "p_holm"] = verdict, p_holm
    return A.forest_data(pooled_df, res.contrasts), pooled_df


def test_forest_panel_texts_never_overlap_even_with_the_longest_verdicts(full):
    res, root, out = full
    # the real-data case that overlapped: P3 "inconclusive, Holm p <0.001" next to P4 "equivalent, Holm p 0.080"
    worst = {("team_main", "spearman"): ("inconclusive", 0.0003), ("rel_vs_shuf", "auc"): ("equivalent", 0.0798),
             ("team_main", "auc"): ("equivalent", 0.262), ("phase_main", "auc"): ("inconclusive", 0.0501), ("zone_nohab", "auc"): ("equivalent", 0.0023)}  # fmt: skip
    data, pooled_df = forest_inputs(res, worst)
    fig = A.build_forest_figure(data, pooled_df)
    items = text_boxes(fig)
    clashes = [(a[:3], b[:3]) for i, a in enumerate(items) for b in items[i + 1 :] if a[3].overlaps(b[3])]
    assert not clashes, clashes
    fig_box = fig.bbox
    assert all(fig_box.x0 <= b.x0 and b.x1 <= fig_box.x1 and fig_box.y0 <= b.y0 and b.y1 <= fig_box.y1 for *_, b in items)  # inside the figure
    # three separate lines per panel, top to bottom: title, verdict, Holm p; the verdict and the p-value are never on one line
    for k, ax in enumerate(fig.axes):
        lines = sorted((t for t in ax.texts), key=lambda t: -t.get_window_extent(fig.canvas.get_renderer()).y0)
        assert [t.get_text().split()[0] for t in lines] == [f"P{[1, 2, 3, 4, 5][k]}", worst_verdict(k, worst), "Holm"]
    # neighbouring panels keep a clear horizontal gap between their widest text lines (the P3 / P4 collision)
    r = fig.canvas.get_renderer()
    for k in range(4):
        right = max(t.get_window_extent(r).x1 for t in fig.axes[k].texts)
        left_next = min(t.get_window_extent(r).x0 for t in fig.axes[k + 1].texts)
        assert left_next - right > 0.05 * fig.dpi, k  # more than 0.05 in


def worst_verdict(k, worst):
    keys = list(A.PRIMARY.values())
    return worst[keys[k]][0]


def test_forest_panel_text_helper():
    class Row:
        verdict, p_holm = "gain", 0.00004

    assert A.forest_panel_text(None) == ("not estimable", "")
    assert A.forest_panel_text(Row()) == ("gain", "Holm p <0.001")
    Row.p_holm, Row.verdict = 0.0798, "equivalent"
    assert A.forest_panel_text(Row()) == ("equivalent", "Holm p 0.080")


def test_a_panel_without_a_pooled_estimate_says_so_in_one_line(full):
    res, root, out = full
    pooled_df = res.pooled[~((res.pooled.contrast == "zone_nohab") & (res.pooled.stratum == "all"))]
    data = A.forest_data(pooled_df, res.contrasts)
    fig = A.build_forest_figure(data, pooled_df)
    last = [t.get_text() for t in fig.axes[4].texts]
    assert last == ["P5  zone object", "not estimable"]
    assert not text_overlaps(fig)


def text_overlaps(fig):
    items = text_boxes(fig)
    return [(a[:3], b[:3]) for i, a in enumerate(items) for b in items[i + 1 :] if a[3].overlaps(b[3])]


def test_figures_can_be_redrawn_from_the_csvs_without_the_bootstrap(full, tmp_path, monkeypatch):
    import shutil

    res, root, out = full
    work = tmp_path / "copy"
    shutil.copytree(out, work)
    for name in ("forest_primary.png", "forest_primary.pdf", "metareg_bubble.png", "metareg_bubble.pdf", "forest_primary_data.csv"):
        (work / "figures" / name).unlink(missing_ok=True)
    before = {p.name: p.read_bytes() for p in work.glob("*.csv")}
    written = A.redraw_figures(work)
    assert {p.name for p in written} >= {"forest_primary.png", "forest_primary.pdf"}
    assert (work / "figures" / "forest_primary.pdf").read_bytes()[:4] == b"%PDF" and (work / "figures" / "forest_primary.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert {p.name: p.read_bytes() for p in work.glob("*.csv")} == before  # nothing but the figures changed
    redrawn = pd.read_csv(work / "figures" / "forest_primary_data.csv")
    original = pd.read_csv(out / "figures" / "forest_primary_data.csv")
    pd.testing.assert_frame_equal(redrawn, original, check_exact=False, rtol=1e-8)
    from matplotlib import image

    img = image.imread(work / "figures" / "forest_primary.png")
    assert img.shape[1] > 1500 and img[..., :3].std() > 0.02
    # the same through the CLI; a directory without results fails clearly
    assert A.main(["--out-dir", str(work), "--redraw-figures"]) == 0
    with pytest.raises(FileNotFoundError, match="run the aggregator"):
        A.redraw_figures(tmp_path / "empty")
    # without --runs-dir the aggregator reads the default v2 root; with that root missing it fails loudly.  The default is patched to a directory that does not
    # exist, so that the test does not depend on the real run directories (the v2 root exists once the re-run has produced it)
    no_runs = tmp_path / "no_v2_runs"
    monkeypatch.setattr(A.P, "RUNS_DIR", no_runs)
    assert A.parse_args(["--out-dir", str(work)]).runs_dir == str(no_runs)
    with pytest.raises(FileNotFoundError, match="runs directory not found"):
        A.main(["--out-dir", str(work)])  # neither --runs-dir nor --redraw-figures


def test_the_bubble_plot_is_redrawn_too_when_the_results_have_a_meta_regression(sized, tmp_path):
    import shutil

    res, sizes, scale, root, out = sized
    work = tmp_path / "copy"
    shutil.copytree(out, work)
    for name in ("forest_primary.png", "forest_primary.pdf", "metareg_bubble.png", "metareg_bubble.pdf"):
        (work / "figures" / name).unlink()
    written = {p.name for p in A.redraw_figures(work)}
    assert written == {"forest_primary.png", "forest_primary.pdf", "metareg_bubble.png", "metareg_bubble.pdf"}
    assert (work / "figures" / "metareg_bubble.pdf").read_bytes()[:4] == b"%PDF" and (work / "figures" / "metareg_bubble.png").stat().st_size > 10_000
    # results without a meta-regression (no training sizes): only the forest plot
    bare = tmp_path / "bare"
    shutil.copytree(out, bare)
    (bare / "figures" / "metareg_bubble.png").unlink()
    (bare / "figures" / "metareg_bubble.pdf").unlink()
    (bare / "metareg.csv").write_text(",".join(A.METAREG_COLUMNS) + "\n")
    assert {p.name for p in A.redraw_figures(bare)} == {"forest_primary.png", "forest_primary.pdf"}
    assert not (bare / "figures" / "metareg_bubble.png").exists()


# --------------------------------------------------------------------------- union (design doc 4.6, post hoc exploratory)
UNION_QUALITY = {"HGB-union": 1.05, "M11R-union": 1.7}  # d' (HGB-cut 0.8, M11R 1.5, KRAFTON's inputs on a GBDT 0.95)
UNION_RANK_QUALITY = {"M11R-union": 2.4}
UNION_FORMULAS = {
    "hgb_union_vs_hgbcut": {"HGB-union": 1, "HGB-cut": -1},
    "m11r_union_vs_m11r": {"M11R-union": 1, "M11R": -1},
    "m11r_union_vs_krafton_tf": {"M11R-union": 1, "KRAFTON-tf": -1},
    "m11r_union_vs_sagong_xgb": {"M11R-union": 1, "SAGONG-xgb": -1},
    "hgb_union_vs_hgb_kraftonfeat": {"HGB-union": 1, "HGB-kraftonfeat": -1},
}  # fmt: skip
UNION_TABLE_ORDER = [  # the rows of the union block of tab_external_contrasts, in the order of the coordinator's list
    ("HGB-union $-$ HGB-cut (+kf)", "hgb_union_vs_hgbcut"), ("M11R-union $-$ M11R (+kf)", "m11r_union_vs_m11r"),
    ("M11R-union $-$ KRAFTON-tf", "m11r_union_vs_krafton_tf"), ("M11R-union $-$ SAGONG-xgb", "m11r_union_vs_sagong_xgb"),
    ("HGB-union $-$ HGB-kraftonfeat (+ours)", "hgb_union_vs_hgb_kraftonfeat"),
]  # fmt: skip
UNION_PAIRS = {name: tuple(coefs) for name, coefs in UNION_FORMULAS.items()}  # (minuend, subtrahend)
UNION_GRID_LABELS = ["Sagong features", "KRAFTON features", "Our representation", "Union (ours + KRAFTON)"]


def union_runs(tmp_path, cohorts=COHORTS[:3], arms=tuple(UNION_QUALITY), external=tuple(EXTERNAL_QUALITY), skip=frozenset(), norank=False):
    quality = {**{a: q for a, q in QUALITY.items() if a not in EXPLORATORY}, **{a: EXTERNAL_QUALITY[a] for a in external},
               **{a: UNION_QUALITY[a] for a in arms}, **(NORANK_QUALITY if norank else {})}  # fmt: skip
    rank_quality = {**{a: q for a, q in RANK_QUALITY.items() if a in quality},
                    **{a: q for a, q in EXTERNAL_RANK_QUALITY.items() if a in quality},
                    **{a: q for a, q in UNION_RANK_QUALITY.items() if a in quality},
                    **{a: q for a, q in NORANK_RANK_QUALITY.items() if a in quality}}  # fmt: skip
    return make_runs(tmp_path, cohorts=cohorts, quality=quality, rank_quality=rank_quality, skip=set(skip), seeds={})


def test_union_arms_and_their_exploratory_contrasts_are_registered():
    assert A.UNION_ARMS == ("HGB-union", "M11R-union")
    for group in (A.MAIN_ARMS, A.HGB_ARMS, A.EXTERNAL_ARMS, A.EXPLORATORY_ARMS, A.STAGE2_ARMS, A.NORANK_ARMS):
        assert not set(A.UNION_ARMS) & set(group)  # the union arms belong to no other family
    assert A.ARM_ORDER == A.MAIN_ARMS + A.EXPLORATORY_ARMS + A.STAGE2_ARMS + A.NORANK_ARMS + A.HGB_ARMS + A.ECOLOGICAL_ARMS + A.UNION_ARMS + A.EXTERNAL_ARMS
    assert A.arm_rank("HGB-noHab") < A.arm_rank("HGB-union") < A.arm_rank("M11R-union") < A.arm_rank("SAGONG-xgb")
    assert len(UNION_FORMULAS) == 5
    for name, coefs in UNION_FORMULAS.items():
        f = A.FORMULAS[name]
        assert f.coefs == coefs and f.optional and f.pooled and f.group == "union"
        assert name not in {k for k, _ in A.PRIMARY.values()}  # never part of the Holm family
    union_rows = [row for row in A.EXTERNAL_CONTRAST_ROWS if row[2] == "union"]
    assert [(label, name) for label, name, *_ in union_rows] == [(label.split(" (")[0], name) for label, name in UNION_TABLE_ORDER]
    assert [row[2] for row in A.EXTERNAL_CONTRAST_ROWS][-5:] == ["union"] * 5  # the fifth block, after the C1 grid effects
    assert {name: tag for _, name, _, tag in union_rows if tag} == {
        "hgb_union_vs_hgbcut": "+kf", "m11r_union_vs_m11r": "+kf", "hgb_union_vs_hgb_kraftonfeat": "+ours"}  # fmt: skip
    assert A.GRID_ROWS[-1][0] == "union" and A.GRID_ROWS[-1][1] == "Union (ours + KRAFTON)"
    assert A.GRID_CELLS[("union", "gbdt")] == "HGB-union" and A.GRID_CELLS[("union", "ptr")] == "M11R-union"
    assert ("union", "krafton_tf") not in A.GRID_CELLS  # no KRAFTON transformer on the union inputs was run


def test_union_contrasts_are_pooled_without_holm_and_follow_the_planted_truth(tmp_path):
    root = union_runs(tmp_path / "r")
    res = A.aggregate(root, tmp_path / "o", n_boot=80, seed=2, workers=1, figures=False)
    inv = res.inventory
    assert set(inv[inv.arm == "HGB-union"].kind) == {"hgb"} and set(inv[inv.arm == "M11R-union"].kind) == {"neural"}  # ours, not external
    assert len(inv[inv.arm == "HGB-union"]) == len(inv[inv.arm == "M11R-union"]) == 3
    qual = {**QUALITY, **EXTERNAL_QUALITY, **UNION_QUALITY}
    for name, (a, b) in UNION_PAIRS.items():
        row = pooled(res, name, "auc")
        want = np.mean([auc_theory(eff(c, a, qual)) - auc_theory(eff(c, b, qual)) for c in COHORTS[:3]])
        assert row.k == 3 and row.kind == "secondary" and row.exploratory and np.isnan(row.p_holm) and row.verdict_basis == "unadjusted"
        assert row.theta == pytest.approx(want, abs=0.03), name
        assert row.primary_id == ""
        for metric in A.METRICS:
            assert np.isfinite(pooled(res, name, metric).theta) and pooled(res, name, metric).se > 0
        assert (res.pooled.contrast == name).sum() == len(A.METRICS) * 3
    assert pooled(res, "m11r_union_vs_sagong_xgb", "auc").verdict == "gain" and pooled(res, "m11r_union_vs_m11r", "auc").theta > 0.01
    assert pooled(res, "hgb_union_vs_hgbcut", "auc").theta > 0.02 and pooled(res, "m11r_union_vs_krafton_tf", "auc").theta > 0.05
    # linear in the per-arm results: (union - ours) + (ours - KRAFTON inputs) = union - KRAFTON inputs, in every cohort
    est = lambda n: res.contrasts[(res.contrasts.contrast == n) & (res.contrasts.metric == "auc") & (res.contrasts.stratum == "all")].set_index("cohort").estimate  # noqa: E731
    assert np.allclose(est("hgb_union_vs_hgbcut") + est("hgbcut_vs_hgb_kraftonfeat"), est("hgb_union_vs_hgb_kraftonfeat"), atol=1e-12)
    assert np.allclose(est("m11r_union_vs_m11r") + est("m11r_vs_sagong_xgb"), est("m11r_union_vs_sagong_xgb"), atol=1e-12)
    assert "union" not in " ".join(res.issues.message)  # optional arms never raise warnings


def test_adding_union_arms_changes_none_of_the_other_results(tmp_path):
    with_union = A.aggregate(union_runs(tmp_path / "a"), tmp_path / "oa", n_boot=60, seed=4, workers=1, figures=False)
    without = A.aggregate(union_runs(tmp_path / "b", arms=()), tmp_path / "ob", n_boot=60, seed=4, workers=1, figures=False)
    key = ["contrast", "metric", "stratum"]
    a = with_union.pooled.merge(without.pooled, on=key, suffixes=("_a", "_b"))
    assert len(a) == len(without.pooled) > 0 and len(with_union.pooled) > len(without.pooled)
    for col in ("theta", "se", "ci95_low", "ci95_high", "tau2", "p_one_sided", "p_holm"):
        assert np.array_equal(a[col + "_a"].to_numpy(), a[col + "_b"].to_numpy(), equal_nan=True), col  # bitwise, shared resamples
    assert with_union.summary["primary_results"] == without.summary["primary_results"]


def test_the_union_block_of_the_contrasts_table_and_the_union_row_of_the_grid(tmp_path):
    root = union_runs(tmp_path / "r")
    res = A.aggregate(root, tmp_path / "o", n_boot=60, seed=2, workers=1, figures=False)
    tables = tmp_path / "o" / "tables"
    text = (tables / "tab_external_contrasts.tex").read_text()
    _, rows = parse_tabular(text)
    flat = [row for row in EXT_TABLE_ORDER if row] + UNION_TABLE_ORDER
    assert [r[0] for r in rows[1:]] == [label for label, _ in flat]  # the union block comes last, after the C1 grid effects
    body_lines = text.split("\\midrule", 1)[1].split("\\bottomrule")[0]  # between the header rule and the bottom rule
    blocks = [[ln for ln in blk.splitlines() if ln.strip().endswith("\\\\")] for blk in body_lines.split("\\midrule")]
    assert [len(b) for b in blocks] == [8, 6, 3, 2, 6, 5]  # ... the K3 ladder, then the union block
    assert "(+kf)" in text and "(+ours)" in text and "post hoc" in text  # the tags are explained in the header comment
    for r, (label, name) in zip(rows[1:], flat):
        for cell, metric in zip(r[1:], ("auc", "spearman", "top5_auc", "cindex")):
            p = pooled(res, name, metric)
            assert cell == A.fmt_delta(p.theta, p.ci95_low, p.ci95_high), (name, metric)
    # the grid: a fourth row, union, with the GBDT and the PTR-Former cell
    gtext = (tables / "tab_grid.tex").read_text()
    _, grows = parse_tabular(gtext)
    assert grows[0] == ["Inputs $\\backslash$ learner", "GBDT", "KRAFTON transformer", "PTR-Former"]
    assert [r[0] for r in grows[1:]] == UNION_GRID_LABELS
    assert grows[3][1:] == [grid_cell(res, "HGB-cut"), grid_cell(res, "KRAFTON-tf-ourfeat"), grid_cell(res, "M11R")]  # the rows above are as before
    assert grows[4][1:] == [grid_cell(res, "HGB-union"), "---", grid_cell(res, "M11R-union")]
    assert "HGB-union" in gtext and "M11R-union" in gtext  # the cells are named in the header comment


def test_the_union_row_only_appears_when_a_union_arm_was_run(tmp_path):
    plain = external_runs(tmp_path / "p")
    A.aggregate(plain, tmp_path / "op", n_boot=30, seed=1, workers=1, figures=False)
    _, rows = parse_tabular((tmp_path / "op" / "tables" / "tab_grid.tex").read_text())
    assert [r[0] for r in rows[1:]] == UNION_GRID_LABELS[:3]  # no dashed row for an experiment that was not run
    _, crows = parse_tabular((tmp_path / "op" / "tables" / "tab_external_contrasts.tex").read_text())
    assert not any("union" in r[0] for r in crows)
    # one union arm is enough for the row; the missing cell is a dash and the count follows the cohorts that have the arm
    one = union_runs(tmp_path / "o1", arms=("HGB-union",), skip={("c_gamma", "HGB-union")})
    res = A.aggregate(one, tmp_path / "oo1", n_boot=30, seed=1, workers=1, figures=False)
    _, rows1 = parse_tabular((tmp_path / "oo1" / "tables" / "tab_grid.tex").read_text())
    assert rows1[4][1:] == [grid_cell(res, "HGB-union"), "---", "---"] and rows1[4][1].endswith("(2 coh.)")
    assert pooled(res, "hgb_union_vs_hgbcut", "auc").k == 2 and (res.pooled.contrast == "m11r_union_vs_m11r").sum() == 0


def test_union_arms_without_any_external_arm_still_write_the_grid_and_the_contrasts_table(tmp_path):
    root = union_runs(tmp_path / "r", external=())
    out = tmp_path / "o"
    res = A.aggregate(root, out, n_boot=40, seed=1, workers=1, figures=False)
    tables = out / "tables"
    assert not (tables / "tab_external.tex").exists()  # that table is about the external baselines
    _, rows = parse_tabular((tables / "tab_external_contrasts.tex").read_text())
    assert [r[0] for r in rows[1:]] == ["HGB-union $-$ HGB-cut (+kf)", "M11R-union $-$ M11R (+kf)"]  # the other three need external arms
    _, grows = parse_tabular((tables / "tab_grid.tex").read_text())
    assert grows[1][1:] == ["---", "---", "---"] and grows[2][1:] == ["---", "---", "---"]
    assert grows[3][1:] == [grid_cell(res, "HGB-cut"), "---", grid_cell(res, "M11R")]
    assert grows[4][1:] == [grid_cell(res, "HGB-union"), "---", grid_cell(res, "M11R-union")]
    for gone in ("m11r_union_vs_krafton_tf", "m11r_union_vs_sagong_xgb", "hgb_union_vs_hgb_kraftonfeat"):
        assert (res.pooled.contrast == gone).sum() == 0
    assert "union" not in " ".join(res.issues.message)


# --------------------------------------------------------------------------- within-phase AUC, the K3 ladder and the decomposition
DECOMP_ROWS = [  # (step, contrast) of tab_decomp.tex, top to bottom: the single-factor steps (a)-(g), then the rank auxiliary loss group
    ("(a) Input vs Sagong (same HGB)", "HGB-cut $-$ SAGONG-hgb"),
    ("(b) Input vs KRAFTON (same HGB)", "HGB-cut $-$ HGB-kraftonfeat"),
    ("(c1) Phase baseline for the Cox score (+H0)", "K2+H0 $-$ K2"),
    ("(c2) Objective (Cox+H0 $\\to$ BCE)", "K3a $-$ K2+H0"),
    ("(d) Explicit phase baseline $\\alpha_k$", "K3 $-$ K3a"),
    ("(e) Architecture (theirs $\\to$ ours, no rank loss)", "M01-norank $-$ K3"),
    ("(f) Structure (phase axis, relations; no rank loss)", "M11R-norank $-$ M01-norank"),
    ("(g) Extended representation (+kf, HGB)", "HGB-union $-$ HGB-cut"),
    ("Rank auxiliary loss, team axis", "M01 $-$ M01-norank"),
    ("Rank auxiliary loss, phase + relations", "M11R $-$ M11R-norank"),
]  # fmt: skip
DECOMP_FORMULAS = ["hgbcut_vs_sagong_hgb", "hgbcut_vs_hgb_kraftonfeat", "k2h0_vs_k2", "k3a_vs_k2h0", "k3_vs_k3a", "m01norank_vs_k3", "m11rnorank_vs_m01norank",
                   "hgb_union_vs_hgbcut", "m01_vs_m01norank", "m11r_vs_m11rnorank"]  # fmt: skip
DECOMP_PAIRS = {  # formula -> (to, from): a plain difference of two models, the pairs of the plan
    "hgbcut_vs_sagong_hgb": ("HGB-cut", "SAGONG-hgb"), "hgbcut_vs_hgb_kraftonfeat": ("HGB-cut", "HGB-kraftonfeat"),
    "k2h0_vs_k2": ("KRAFTON-tf-ourfeat-h0", "KRAFTON-tf-ourfeat"), "k3a_vs_k2h0": ("KRAFTON-tf-ourfeat-bce", "KRAFTON-tf-ourfeat-h0"),
    "k3_vs_k3a": ("KRAFTON-tf-ourfeat-hazard", "KRAFTON-tf-ourfeat-bce"), "m01norank_vs_k3": ("M01-norank", "KRAFTON-tf-ourfeat-hazard"),
    "m11rnorank_vs_m01norank": ("M11R-norank", "M01-norank"), "hgb_union_vs_hgbcut": ("HGB-union", "HGB-cut"),
    "m01_vs_m01norank": ("M01", "M01-norank"), "m11r_vs_m11rnorank": ("M11R", "M11R-norank"),
}  # fmt: skip
DECOMP_BLOCKS = [2, 4, 1, 1, 2]  # rows per block between the rules: inputs | the K ladder | structure | extension | the rank auxiliary loss
BUNDLED_CONTRASTS = ("krafton_ourfeat_input", "hgb_kraftonfeat_vs_krafton_tf", "hgbcut_vs_sagong_xgb", "m01_vs_krafton_tf_ourfeat", "m01_vs_k3")  # several factors at once
RQ2_ROWS = [  # (label, formula) of tab_rq2_external.tex, top to bottom: our two models against the external baselines
    ("M11R $-$ KRAFTON-tf", "m11r_vs_krafton_tf"), ("M11R $-$ KRAFTON-tf-h0", "m11r_vs_krafton_tf_h0"), ("M11R $-$ SAGONG-xgb", "m11r_vs_sagong_xgb"),
    ("M11R $-$ SAGONG-rf", "m11r_vs_sagong_rf"), ("M11R $-$ SAGONG-lr", "m11r_vs_sagong_lr"), ("HGB-cut $-$ KRAFTON-tf", "hgbcut_vs_krafton_tf"),
    ("HGB-cut $-$ KRAFTON-tf-h0", "hgbcut_vs_krafton_tf_h0"), ("HGB-cut $-$ SAGONG-xgb", "hgbcut_vs_sagong_xgb"),
    ("HGB-cut $-$ SAGONG-rf", "hgbcut_vs_sagong_rf"), ("HGB-cut $-$ SAGONG-lr", "hgbcut_vs_sagong_lr"),
]  # fmt: skip


def decomp_blocks(text):
    """Rows per block (blocks are separated by a \\midrule) of the body of a generated tabular; a \\multicolumn title row is not a row."""
    body = text.split("\\midrule", 1)[1].split("\\bottomrule")[0]
    return [len([ln for ln in blk.splitlines() if ln.strip().endswith("\\\\") and not ln.strip().startswith("\\multicolumn")]) for blk in body.split("\\midrule")]


def data_rows(rows):
    """The rows of a parsed tabular without the title rows of a group."""
    return [r for r in rows if not r[0].startswith("\\multicolumn")]


def cohort_estimates(res, name, metric="auc", stratum="all"):
    sub = res.contrasts[(res.contrasts.contrast == name) & (res.contrasts.metric == metric) & (res.contrasts.stratum == stratum)]
    return sub.set_index("cohort").estimate


def test_the_within_phase_auc_is_an_exploratory_metric_next_to_the_seven(tmp_path):
    root = external_runs(tmp_path / "r")
    res = A.aggregate(root, tmp_path / "o", n_boot=60, seed=2, workers=1, figures=False)
    assert A.METRICS == M.ALL_METRIC_NAMES and A.TABLE_METRICS == M.METRIC_NAMES and len(A.METRICS) == 8
    assert "auc_within_phase" not in A.SESOI and A.DIRECTION["auc_within_phase"] == 1 and A.METRIC_LABEL["auc_within_phase"] == "AUC within phase"
    am = res.arm_metrics[res.arm_metrics.metric == "auc_within_phase"]
    assert len(am) == len(res.arm_metrics[res.arm_metrics.metric == "auc"]) > 0 and np.isfinite(am.value).all()  # every arm, every stratum
    row = am[(am.cohort == COHORTS[0]) & (am.arm == "M11R") & (am.seed == 42) & (am.stratum == "all")].iloc[0]
    plain = M.within_phase_auc(M.read_predictions(root / COHORTS[0] / "M11R" / "seed42" / "pred_test.csv"))["auc_within_phase"]
    assert row.value == pytest.approx(plain, abs=1e-12)  # the value is the plain metric of the prediction file
    sub = res.pooled[res.pooled.metric == "auc_within_phase"]
    assert len(sub) == len(res.pooled[res.pooled.metric == "auc"]) > 0 and (sub.kind == "secondary").all() and sub.exploratory.all()
    assert sub.p_holm.isna().all() and (sub.verdict == "n/a").all() and (sub.verdict_basis == "unadjusted").all() and sub.sesoi.isna().all()
    assert "auc_within_phase" not in {m for _, m in A.PRIMARY.values()}  # the pre-specified primary metric stays the pooled AUC
    big = pooled(res, "m11r_vs_sagong_lr", "auc_within_phase")  # a large planted gain: one-sided p for an improvement is tiny, theta ~ the pooled one
    assert big.theta > 0.1 and big.p_one_sided < 1e-3 and big.theta == pytest.approx(pooled(res, "m11r_vs_sagong_lr", "auc").theta, abs=0.02)
    assert len(res.contrasts[(res.contrasts.metric == "auc_within_phase") & (res.contrasts.contrast == "k3_vs_k2")]) == 3 * 3  # cohorts x strata
    for name in ("P1", "P2", "P4", "P5"):  # the primary contrasts also get the within-phase twin, as secondary rows
        contrast, _ = A.PRIMARY[name]
        assert (sub.contrast == contrast).any()
    # the paper tables keep their seven metric columns
    for table in ("tab_arms_metrics.tex", "tab_external.tex"):
        _, rows = parse_tabular((tmp_path / "o" / "tables" / table).read_text())
        assert rows[1][1:] == ["AUC", "AP", "NLL", "Brier", "Spearman", "Top-5 AUC", "C-index"] and all(len(r) == 8 for r in rows[1:]), table
    assert "within" not in (tmp_path / "o" / "tables" / "tab_arms_metrics.tex").read_text()


def test_asking_for_the_within_phase_auc_changes_none_of_the_seven_other_metrics(tmp_path, monkeypatch):
    root = external_runs(tmp_path / "r", arms=("KRAFTON-tf", "SAGONG-xgb"))
    with_new = A.aggregate(root, None, n_boot=50, seed=3, workers=1, figures=False)
    monkeypatch.setattr(A, "METRICS", M.METRIC_NAMES)  # the old behaviour: seven metrics
    without = A.aggregate(root, None, n_boot=50, seed=3, workers=1, figures=False)
    assert set(without.pooled.metric) == set(M.METRIC_NAMES) and "auc_within_phase" in set(with_new.pooled.metric)
    key = ["contrast", "metric", "stratum"]
    a = with_new.pooled.merge(without.pooled, on=key, suffixes=("_a", "_b"))
    assert len(a) == len(without.pooled) > 0
    for col in ("theta", "se", "ci95_low", "ci95_high", "tau2", "p_one_sided", "p_holm", "I2"):
        assert np.array_equal(a[col + "_a"].to_numpy(), a[col + "_b"].to_numpy(), equal_nan=True), col  # bitwise: same resamples, same draws
    c = with_new.contrasts.merge(without.contrasts, on=["cohort", *key], suffixes=("_a", "_b"))
    assert len(c) == len(without.contrasts) > 0
    for col in ("estimate", "se", "ci_low", "ci_high"):
        assert np.array_equal(c[col + "_a"].to_numpy(), c[col + "_b"].to_numpy(), equal_nan=True), col
    assert with_new.summary["primary_results"] == without.summary["primary_results"]


def test_a_phase_baseline_gains_pooled_auc_but_not_within_phase_auc(tmp_path):
    arms = ("M00", "M10", "M01", "M11", "M11R", "M11R-shuf", "M11RZ", "M11R-noHab", "M11RZ-noHab", "HGB-cut", "HGB-lag", "HGB-noHab")
    quality = {**{a: QUALITY[a] for a in arms}, "KRAFTON-tf-ourfeat-bce": 1.1, "KRAFTON-tf-ourfeat-hazard": 1.1}  # same d'
    rank_quality = {a: RANK_QUALITY.get(a, 1.1) for a in quality}
    root = make_runs(tmp_path / "r", cohorts=COHORTS[:3], quality=quality, rank_quality=rank_quality, skip=set(), seeds={},
                     phase_rate=(0.05, 0.6), phase_slope={"KRAFTON-tf-ourfeat-hazard": 1.2})  # fmt: skip
    res = A.aggregate(root, tmp_path / "o", n_boot=60, seed=2, workers=1, figures=False)
    pooled_auc, within = pooled(res, "k3_vs_k3a", "auc"), pooled(res, "k3_vs_k3a", "auc_within_phase")
    assert pooled_auc.theta > 0.02 and pooled_auc.ci95_low > 0  # the phase baseline looks like a gain in the pooled AUC ...
    assert abs(within.theta) < 0.02 and within.ci95_low < 0 < within.ci95_high  # ... and is nothing within a phase
    text = (tmp_path / "o" / "tables" / "tab_decomp.tex").read_text()
    _, rows = parse_tabular(text)
    step = next(r for r in rows if r[1] == "K3 $-$ K3a")
    assert step[2] == A.fmt_delta(pooled_auc.theta, pooled_auc.ci95_low, pooled_auc.ci95_high)
    assert step[4] == A.fmt_delta(within.theta, within.ci95_low, within.ci95_high)  # the table shows both side by side


def test_the_k3_ladder_contrasts_are_linear_exploratory_and_follow_the_planted_truth(tmp_path):
    root = external_runs(tmp_path / "r")
    res = A.aggregate(root, tmp_path / "o", n_boot=60, seed=2, workers=1, figures=False)
    inv = res.inventory
    assert set(inv[inv.arm.isin(["KRAFTON-tf-ourfeat-bce", "KRAFTON-tf-ourfeat-hazard"])].kind) == {"external"}
    for metric in ("auc", "auc_within_phase"):
        # objective + phase baseline = formulation in total; residual + formulation = M01 against K2 (the existing contrast)
        assert np.allclose(cohort_estimates(res, "k3a_vs_k2", metric) + cohort_estimates(res, "k3_vs_k3a", metric),
                           cohort_estimates(res, "k3_vs_k2", metric), atol=1e-12)  # fmt: skip
        assert np.allclose(cohort_estimates(res, "m01_vs_k3", metric) + cohort_estimates(res, "k3_vs_k2", metric),
                           cohort_estimates(res, "m01_vs_krafton_tf_ourfeat", metric), atol=1e-12)  # fmt: skip
    assert pooled(res, "k3a_vs_k2", "auc").theta < 0 < pooled(res, "k3_vs_k3a", "auc").theta  # d': BCE 1.1 < Cox 1.2, K3 1.25 > BCE 1.1
    assert pooled(res, "m01_vs_k3", "auc").theta < -0.02  # our M01 (d' 0.9) against K3 (1.25): the residual is negative here
    for name in ("k3a_vs_k2", "k3_vs_k3a", "k3_vs_k2", "m01_vs_k3"):
        row = pooled(res, name, "auc")
        assert row.k == 3 and row.kind == "secondary" and row.exploratory and np.isnan(row.p_holm) and row.verdict_basis == "unadjusted"
    # only the K3 arms that exist produce rows: without K3 (the KRAFTON agent has not delivered it) there is no warning either
    only = external_runs(tmp_path / "r2", arms=("KRAFTON-tf-ourfeat",))
    res2 = A.aggregate(only, None, n_boot=30, seed=1, workers=1, figures=False)
    assert all((res2.pooled.contrast == n).sum() == 0 for n in ("k3a_vs_k2", "k3_vs_k3a", "k3_vs_k2", "m01_vs_k3"))
    assert "KRAFTON-tf-ourfeat-" not in " ".join(res2.issues.message)


def test_the_decomposition_table_lists_the_single_factor_steps_and_the_rank_loss_group_with_pooled_and_within_phase_deltas(tmp_path):
    root = union_runs(tmp_path / "r", norank=True)  # the external arms (incl. SAGONG-hgb, K2H0 and the K3 ladder), the union arms and the no-rank-loss arms
    res = A.aggregate(root, tmp_path / "o", n_boot=60, seed=2, workers=1, figures=False)
    text = (tmp_path / "o" / "tables" / "tab_decomp.tex").read_text()
    _, rows = parse_tabular(text)
    assert "\\begin{tabular}{@{}l l c c c c@{}}" in text and "generated by model_stgcn.ptr.aggregate" in text
    assert rows[0] == ["Step", "Contrast", "$\\Delta$AUC [95\\% CI]", "Coh.", "$\\Delta$AUC within phase [95\\% CI]", "Coh."]
    title = [r for r in rows if r[0].startswith("\\multicolumn")]
    assert len(title) == 1 and "Rank auxiliary loss" in title[0][0]  # the separate group has a title row, placed right after its rule
    assert text.index("\\multicolumn") > text.index("HGB-union $-$ HGB-cut") and text.count("\\midrule") == 1 + 4  # the header rule + one per group change
    rows = data_rows(rows)
    assert [(r[0], r[1]) for r in rows[1:]] == DECOMP_ROWS and [name for *_, name in A.DECOMP_STEPS] == DECOMP_FORMULAS
    assert [g for g, *_ in A.DECOMP_STEPS] == ["inputs"] * 2 + ["ladder"] * 4 + ["structure", "extension"] + [A.RANK_AUX_GROUP] * 2
    assert decomp_blocks(text) == DECOMP_BLOCKS  # a rule between the groups: inputs | K ladder | structure | extension | rank auxiliary loss
    # every step changes ONE factor: a plain difference of two models, exactly the pairs of the plan, written to - from
    for name, (to, frm) in DECOMP_PAIRS.items():
        assert A.FORMULAS[name].coefs == {to: 1, frm: -1}, name
    # the bundled contrasts (several factors at once) are NOT steps: they stay in the external contrasts table, tagged
    assert not set(BUNDLED_CONTRASTS) & set(DECOMP_FORMULAS)
    assert not {r[1] for r in rows[1:]} & {"KRAFTON-tf-ourfeat $-$ KRAFTON-tf", "HGB-kraftonfeat $-$ KRAFTON-tf", "HGB-cut $-$ SAGONG-xgb",
                                           "M01 $-$ KRAFTON-tf-ourfeat", "M01 $-$ K3", "M11R $-$ M01"}  # fmt: skip
    _, crows = parse_tabular((tmp_path / "o" / "tables" / "tab_external_contrasts.tex").read_text())
    labels = [r[0] for r in crows[1:]]
    assert all(f"{c} (bundled)" in labels for c in ("KRAFTON-tf-ourfeat $-$ KRAFTON-tf", "HGB-kraftonfeat $-$ KRAFTON-tf",
                                                      "HGB-cut $-$ SAGONG-xgb", "M01 $-$ KRAFTON-tf-ourfeat"))  # fmt: skip
    # the structure step with the rank loss on both sides (M11R - M01) stays a formula and is the sum of the two simple effects of the main matrix
    assert not A.FORMULAS["m11r_vs_m01"].optional and pooled(res, "m11r_vs_m01", "auc").theta > 0.05
    assert np.allclose(cohort_estimates(res, "m11r_vs_m01"), cohort_estimates(res, "m11r_m11") + cohort_estimates(res, "m11_m01"), atol=1e-12)
    assert pooled(res, "hgb_union_vs_hgbcut", "auc").theta > 0.02 and pooled(res, "hgbcut_vs_sagong_hgb", "auc").theta > 0  # planted: +0.057, +0.020
    # the steps are linear in the arms: (c1) + (c2) = the old objective step K3a - K2, (e) + the rank loss of M01 = M01 - K3, and the chain of the structure
    for metric in ("auc", "auc_within_phase"):
        assert np.allclose(cohort_estimates(res, "k2h0_vs_k2", metric) + cohort_estimates(res, "k3a_vs_k2h0", metric), cohort_estimates(res, "k3a_vs_k2", metric), atol=1e-12)
        assert np.allclose(cohort_estimates(res, "m01norank_vs_k3", metric) + cohort_estimates(res, "m01_vs_m01norank", metric), cohort_estimates(res, "m01_vs_k3", metric), atol=1e-12)
        assert np.allclose(cohort_estimates(res, "m11rnorank_vs_m01norank", metric) + cohort_estimates(res, "m11r_vs_m11rnorank", metric)
                           - cohort_estimates(res, "m01_vs_m01norank", metric), cohort_estimates(res, "m11r_vs_m01", metric), atol=1e-12)  # fmt: skip
    for r, name in zip(rows[1:], DECOMP_FORMULAS):
        auc, wp = pooled(res, name, "auc"), pooled(res, name, "auc_within_phase")
        assert r[2:] == [A.fmt_delta(auc.theta, auc.ci95_low, auc.ci95_high), f"{int(auc.k)}/{int(auc.k_total)}",
                         A.fmt_delta(wp.theta, wp.ci95_low, wp.ci95_high), f"{int(wp.k)}/{int(wp.k_total)}"]  # fmt: skip
        assert r[3] == r[5] == "3/3"  # three cohorts, all pooled
    for needle in ("every row is (to) - (from) and changes ONE factor", "K2 = KRAFTON-tf-ourfeat", "K2+H0 = KRAFTON-tf-ourfeat-h0", "K3a = KRAFTON-tf-ourfeat-bce",
                   "K3 = KRAFTON-tf-ourfeat-hazard", "within phase = unweighted mean over the cut phases", "Coh. = cohorts pooled", "exploratory, no Holm", "bundled",
                   "tab_external_contrasts.tex", "tab_rq2_external.tex", "(c1) K2+H0 - K2 adds a training-set phase baseline to the Cox score",
                   "(c2) K3a - K2+H0 changes the objective, Cox with baseline -> BCE", "(d) K3 - K3a adds the explicit phase baseline",
                   "WITHOUT the rank auxiliary loss", "rank auxiliary loss (separate, not a step of the ladder)",
                   "KRAFTON-tf-ourfeat - KRAFTON-tf, HGB-kraftonfeat - KRAFTON-tf, HGB-cut - SAGONG-xgb, M01 - KRAFTON-tf-ourfeat"):  # fmt: skip
        assert needle in text, needle
    assert "steps not run" not in text  # all ten rows are there
    # the pre-specified analysis does not know the no-rank-loss arms: P1-P5 and the Holm family are what they were
    assert set(A.PRIMARY) == {"P1", "P2", "P3", "P4", "P5"} and A.HOLM_FAMILY == ("P1", "P2", "P3", "P4", "P5")
    assert all(not any(a in A.NORANK_ARMS for a in A.FORMULAS[f].coefs) for f, _ in A.PRIMARY.values())
    assert set(res.pooled[res.pooled.primary_id != ""].contrast) == {"phase_main", "team_main", "rel_vs_shuf", "zone_nohab"}


def test_h0_arms_are_ingested_as_regular_arms_with_paired_bootstrap_and_pooling(tmp_path):
    root = external_runs(tmp_path / "r", norank=True)
    res = A.aggregate(root, tmp_path / "o", n_boot=60, seed=2, workers=1, figures=False)
    inv = res.inventory
    for arm in ("KRAFTON-tf-h0", "KRAFTON-tf-ourfeat-h0"):
        assert set(inv[inv.arm == arm].kind) == {"external"} and len(inv[inv.arm == arm]) == 3 and inv[inv.arm == arm].aligned.all()
        assert (res.arm_metrics.arm == arm).sum() == len(A.METRICS) * 3 * 3  # every metric, every stratum, every cohort: like any other arm
    assert set(inv[inv.arm.isin(A.NORANK_ARMS)].kind) == {"neural"} and len(inv[inv.arm == "M01-norank"]) == 3
    for name in ("k2h0_vs_k2", "k3a_vs_k2h0", "m11r_vs_krafton_tf_h0", "hgbcut_vs_krafton_tf_h0"):
        row = pooled(res, name, "auc")
        assert row.k == 3 and row.kind == "secondary" and row.exploratory and np.isnan(row.p_holm) and row.verdict_basis == "unadjusted"
        assert (res.contrasts.contrast == name).sum() == len(A.METRICS) * 3 * 3 and np.isfinite(pooled(res, name, "auc_within_phase").theta)
    qual = {**QUALITY, **EXTERNAL_QUALITY, **NORANK_QUALITY}
    for name in ("k2h0_vs_k2", "k3a_vs_k2h0", "m11r_vs_krafton_tf_h0", "m01norank_vs_k3", "m11rnorank_vs_m01norank", "m01_vs_m01norank", "m11r_vs_m11rnorank"):
        f = A.FORMULAS[name]
        (to, _), (frm, _) = [(a, c) for a, c in f.coefs.items() if c > 0][0], [(a, c) for a, c in f.coefs.items() if c < 0][0]
        want = np.mean([auc_theory(eff(c, to, qual)) - auc_theory(eff(c, frm, qual)) for c in COHORTS[:3]])
        assert pooled(res, name, "auc").theta == pytest.approx(want, abs=0.03), name
    # adding the H0 and the no-rank-loss arms changes none of the primary results (the resamples depend on the cohort and stratum only)
    base = A.aggregate(external_runs(tmp_path / "b"), None, n_boot=60, seed=2, workers=1, figures=False)
    key = ["contrast", "metric", "stratum"]
    a = res.pooled.merge(base.pooled, on=key, suffixes=("_a", "_b"))
    assert len(a) == len(base.pooled) > 0
    for col in ("theta", "se", "ci95_low", "ci95_high", "tau2", "p_one_sided", "p_holm"):
        assert np.array_equal(a[col + "_a"].to_numpy(), a[col + "_b"].to_numpy(), equal_nan=True), col
    assert res.summary["primary_results"] == base.summary["primary_results"]
    # the sensitivity variants (-slope, -h0val) are inventoried by prefix without being registered
    assert A.Run("c", "KRAFTON-tf-ourfeat-h0-slope", 42, Path("p"), "v2", "h").kind == "external"


def test_the_rq2_table_holds_the_bundled_external_contrasts_in_a_table_of_its_own(tmp_path):
    root = external_runs(tmp_path / "r")
    res = A.aggregate(root, tmp_path / "o", n_boot=60, seed=2, workers=1, figures=False)
    text = (tmp_path / "o" / "tables" / "tab_rq2_external.tex").read_text()
    _, rows = parse_tabular(text)
    assert "\\begin{tabular}{@{}l c c c c@{}}" in text and "generated by model_stgcn.ptr.aggregate (tab_rq2_external)" in text
    assert rows[0] == ["Contrast", "$\\Delta$AUC [95\\% CI]", "Coh.", "$\\Delta$AUC within phase [95\\% CI]", "Coh."]
    assert [r[0] for r in rows[1:]] == [label for label, _ in RQ2_ROWS]  # M11R against the five baselines, then HGB-cut against the same five
    assert [(m, b, f) for m, b, f in A.RQ2_ROWS] == [(label.split(" $-$ ")[0], label.split(" $-$ ")[1], name) for label, name in RQ2_ROWS]
    assert decomp_blocks(text) == [5, 5]  # one block per model of ours
    for r, (label, name) in zip(rows[1:], RQ2_ROWS):
        auc, wp = pooled(res, name, "auc"), pooled(res, name, "auc_within_phase")
        assert r[1:] == [A.fmt_delta(auc.theta, auc.ci95_low, auc.ci95_high), f"{int(auc.k)}/{int(auc.k_total)}",
                         A.fmt_delta(wp.theta, wp.ci95_low, wp.ci95_high), f"{int(wp.k)}/{int(wp.k_total)}"]  # fmt: skip
    assert not {r[0] for r in rows[1:]} & {r[1] for r in DECOMP_ROWS}  # none of them is a decomposition step
    for needle in ("BUNDLED", "not decomposition steps", "KRAFTON-tf-h0 = KRAFTON-tf scores with a phase baseline fitted on VAL", "exploratory, no Holm"):
        assert needle in text, needle
    # the two new contrasts are plain differences of the planned pairs
    assert A.FORMULAS["m11r_vs_krafton_tf_h0"].coefs == {"M11R": 1, "KRAFTON-tf-h0": -1}
    assert A.FORMULAS["hgbcut_vs_krafton_tf_h0"].coefs == {"HGB-cut": 1, "KRAFTON-tf-h0": -1}
    # a baseline that was not run leaves its rows out and is named; without any external arm there is no table (and a stale file is removed)
    part = external_runs(tmp_path / "p", arms=("KRAFTON-tf", "SAGONG-xgb"))
    A.aggregate(part, tmp_path / "op", n_boot=30, seed=1, workers=1, figures=False)
    text2 = (tmp_path / "op" / "tables" / "tab_rq2_external.tex").read_text()
    _, rows2 = parse_tabular(text2)
    assert [r[0] for r in rows2[1:]] == ["M11R $-$ KRAFTON-tf", "M11R $-$ SAGONG-xgb", "HGB-cut $-$ KRAFTON-tf", "HGB-cut $-$ SAGONG-xgb"]
    assert "contrasts not run (left out): m11r_vs_krafton_tf_h0, m11r_vs_sagong_rf, m11r_vs_sagong_lr, hgbcut_vs_krafton_tf_h0, hgbcut_vs_sagong_rf, hgbcut_vs_sagong_lr" in text2
    out = tmp_path / "o2"
    (out / "tables").mkdir(parents=True)
    (out / "tables" / "tab_rq2_external.tex").write_text("stale")
    A.aggregate(external_runs(tmp_path / "q", arms=()), out, n_boot=20, seed=1, workers=1, figures=False)
    assert not (out / "tables" / "tab_rq2_external.tex").exists()


def test_decomposition_steps_without_their_models_are_left_out_and_named(tmp_path):
    root = external_runs(tmp_path / "r", arms=("SAGONG-hgb", "SAGONG-xgb", "KRAFTON-tf-ourfeat", "KRAFTON-tf-ourfeat-h0", "KRAFTON-tf-ourfeat-bce"),
                         skip={("c_gamma", "KRAFTON-tf-ourfeat-bce")})  # fmt: skip
    res = A.aggregate(root, tmp_path / "o", n_boot=40, seed=1, workers=1, figures=False)
    text = (tmp_path / "o" / "tables" / "tab_decomp.tex").read_text()
    _, rows = parse_tabular(text)
    rows = data_rows(rows)
    assert [r[0] for r in rows[1:]] == [DECOMP_ROWS[0][0], DECOMP_ROWS[2][0], DECOMP_ROWS[3][0]]  # input vs Sagong, the Cox baseline, the objective step
    assert ("steps not run (left out): hgbcut_vs_hgb_kraftonfeat, k3_vs_k3a, m01norank_vs_k3, m11rnorank_vs_m01norank, hgb_union_vs_hgbcut, "
            "m01_vs_m01norank, m11r_vs_m11rnorank") in text  # fmt: skip
    assert decomp_blocks(text) == [1, 2] and "multicolumn" not in text  # rules only between the groups that have rows; no title for a group without rows
    c2 = rows[3]
    assert c2[3] == "2/3" and c2[5] == "2/3" and pooled(res, "k3a_vs_k2h0", "auc").k == 2  # c_gamma lacks K3a: pooled over two cohorts
    assert rows[1][3] == "3/3" and rows[2][3] == "3/3"
    # a bundled contrast never stands in for a missing clean step: without SAGONG-hgb there is no "input vs Sagong" row,
    # although HGB-cut - SAGONG-xgb (input + learner) exists; only the steps whose arms exist (here: with the no-rank-loss arms) are listed
    only_xgb = external_runs(tmp_path / "x", arms=("SAGONG-xgb",), norank=True)
    res_x = A.aggregate(only_xgb, tmp_path / "ox", n_boot=30, seed=1, workers=1, figures=False)
    text_x = (tmp_path / "ox" / "tables" / "tab_decomp.tex").read_text()
    _, rows_x = parse_tabular(text_x)
    assert [r[0] for r in data_rows(rows_x)[1:]] == [DECOMP_ROWS[6][0], DECOMP_ROWS[8][0], DECOMP_ROWS[9][0]] and pooled(res_x, "hgbcut_vs_sagong_xgb", "auc").k == 3
    assert "hgbcut_vs_sagong_hgb" in text_x.split("steps not run (left out):")[1].splitlines()[0]
    # no step at all: no table, and a stale file is removed
    few = ("M00", "M10", "M01", "M11")
    quality = {a: QUALITY[a] for a in few}
    plain = make_runs(tmp_path / "p", cohorts=COHORTS[:2], quality=quality, rank_quality={a: RANK_QUALITY[a] for a in few}, skip=set(), seeds={})
    out = tmp_path / "o2"
    (out / "tables").mkdir(parents=True)
    (out / "tables" / "tab_decomp.tex").write_text("stale")
    A.aggregate(plain, out, n_boot=20, seed=1, workers=1, figures=False)
    assert not (out / "tables" / "tab_decomp.tex").exists()
    # the main arms alone: no step exists (the structure step needs the no-rank-loss arms now); with the norank arms: (f) and the rank loss group
    main_only = external_runs(tmp_path / "m", arms=())
    A.aggregate(main_only, tmp_path / "o3", n_boot=20, seed=1, workers=1, figures=False)
    assert not (tmp_path / "o3" / "tables" / "tab_decomp.tex").exists()
    nr = external_runs(tmp_path / "n", arms=(), norank=True)
    A.aggregate(nr, tmp_path / "o4", n_boot=20, seed=1, workers=1, figures=False)
    _, rows4 = parse_tabular((tmp_path / "o4" / "tables" / "tab_decomp.tex").read_text())
    assert [r[0] for r in data_rows(rows4)[1:]] == [DECOMP_ROWS[6][0], DECOMP_ROWS[8][0], DECOMP_ROWS[9][0]]


def test_rerendering_the_tables_from_the_csvs_reproduces_the_tables_of_the_run(tmp_path):
    root = union_runs(tmp_path / "r", norank=True)
    res = A.aggregate(root, tmp_path / "o", n_boot=40, seed=2, workers=1, figures=False)
    tables = tmp_path / "o" / "tables"
    before = {p.name: p.read_text() for p in tables.glob("*.tex")}
    assert {"tab_decomp.tex", "tab_rq2_external.tex", "tab_external_contrasts.tex", "tab_grid.tex", "tab_external.tex"} <= set(before)
    for p in tables.glob("*.tex"):
        p.write_text("stale")
    (tables / "tab_rq2_external.tex").unlink()
    written = A.rerender_tables(tmp_path / "o")
    assert {p.name for p in written} == set(before)
    assert {p.name: p.read_text() for p in tables.glob("*.tex")} == before  # the tables are a function of the CSVs alone
    assert A.main(["--rerender-tables", "--out-dir", str(tmp_path / "o")]) == 0
    with pytest.raises(FileNotFoundError, match="run the aggregator once"):
        A.rerender_tables(tmp_path / "nowhere")
    # the norank arms are rows of the arms table only when they were run
    _, arows = parse_tabular((tables / "tab_arms_metrics.tex").read_text())
    assert any(r[0] == "M01$-$norank" for r in arows) and any(r[0] == "M11R$-$norank" for r in arows)
    plain = A.aggregate(external_runs(tmp_path / "p"), tmp_path / "op", n_boot=20, seed=2, workers=1, figures=False)
    assert "norank" not in (tmp_path / "op" / "tables" / "tab_arms_metrics.tex").read_text() and plain is not None


# --------------------------------------------------------------------------- ecological arms (PHASE-only, EHM, EHM-int)
ECO_QUALITY = {"PHASE-only": 0.45, "EHM": 1.0, "EHM-int": 1.02, "EHM-lite": 0.95}  # d' (HGB-cut 0.8, M11R 1.5)
ECO_FORMULAS = {
    "ehm_vs_phase": {"EHM": 1, "PHASE-only": -1}, "ehmint_vs_ehm": {"EHM-int": 1, "EHM": -1},
    "hgbcut_vs_ehm": {"HGB-cut": 1, "EHM": -1}, "m11r_vs_ehm": {"M11R": 1, "EHM": -1},
    "ehmlite_vs_phase": {"EHM-lite": 1, "PHASE-only": -1}, "ehm_vs_ehmlite": {"EHM": 1, "EHM-lite": -1},
    "hgbcut_vs_ehmlite": {"HGB-cut": 1, "EHM-lite": -1},
}  # fmt: skip


def eco_runs(tmp_path, cohorts=COHORTS[:3], arms=tuple(ECO_QUALITY), skip=frozenset()):
    quality = {**{a: q for a, q in QUALITY.items() if a not in EXPLORATORY}, **{a: ECO_QUALITY[a] for a in arms}}
    rank_quality = {**{a: q for a, q in RANK_QUALITY.items() if a in quality}, **{a: 1.0 for a in arms}}
    return make_runs(tmp_path, cohorts=cohorts, quality=quality, rank_quality=rank_quality, skip=set(skip), seeds={})


def test_ecological_arms_are_their_own_kind_and_have_seven_exploratory_contrasts(tmp_path):
    assert A.ECOLOGICAL_ARMS == ("PHASE-only", "EHM", "EHM-int", "EHM-lite")
    for group in (A.MAIN_ARMS, A.EXPLORATORY_ARMS, A.STAGE2_ARMS, A.NORANK_ARMS, A.HGB_ARMS, A.UNION_ARMS, A.EXTERNAL_ARMS):
        assert not set(A.ECOLOGICAL_ARMS) & set(group)  # a family of their own
    assert A.arm_rank("HGB-noHab") < A.arm_rank("PHASE-only") < A.arm_rank("EHM") < A.arm_rank("EHM-int") < A.arm_rank("EHM-lite") < A.arm_rank("HGB-union")
    for name, coefs in ECO_FORMULAS.items():
        f = A.FORMULAS[name]
        assert f.coefs == coefs and f.optional and f.pooled and f.group == "ecological"
        assert name not in {k for k, _ in A.PRIMARY.values()}  # never part of the Holm family
    res = A.aggregate(eco_runs(tmp_path / "r"), tmp_path / "o", n_boot=60, seed=2, workers=1, figures=False)
    inv = res.inventory
    assert set(inv[inv.arm.isin(A.ECOLOGICAL_ARMS)].kind) == {"ecological"} and len(inv[inv.arm == "EHM"]) == 3
    assert set(inv[inv.arm.isin(A.HGB_ARMS)].kind) == {"hgb"} and set(inv[inv.arm.isin(["M11R", "M00"])].kind) == {"neural"}
    qual = {**QUALITY, **ECO_QUALITY}
    pairs = {"ehm_vs_phase": ("EHM", "PHASE-only"), "ehmint_vs_ehm": ("EHM-int", "EHM"), "hgbcut_vs_ehm": ("HGB-cut", "EHM"), "m11r_vs_ehm": ("M11R", "EHM"),
             "ehmlite_vs_phase": ("EHM-lite", "PHASE-only"), "ehm_vs_ehmlite": ("EHM", "EHM-lite"), "hgbcut_vs_ehmlite": ("HGB-cut", "EHM-lite")}  # fmt: skip
    for name, (a, b) in pairs.items():
        row = pooled(res, name, "auc")
        want = np.mean([auc_theory(eff(c, a, qual)) - auc_theory(eff(c, b, qual)) for c in COHORTS[:3]])
        assert row.k == 3 and row.kind == "secondary" and row.exploratory and np.isnan(row.p_holm) and row.verdict_basis == "unadjusted"
        assert row.theta == pytest.approx(want, abs=0.03), name
        assert (res.pooled.contrast == name).sum() == len(A.METRICS) * 3 and np.isfinite(pooled(res, name, "auc_within_phase").theta)
    assert pooled(res, "ehm_vs_phase", "auc").theta > 0.1 and pooled(res, "hgbcut_vs_ehm", "auc").theta < 0
    # no external or union arm: none of their tables; the ecological arms are not in the paper tables either
    assert not (tmp_path / "o" / "tables" / "tab_external.tex").exists() and not (tmp_path / "o" / "tables" / "tab_grid.tex").exists()
    _, rows = parse_tabular((tmp_path / "o" / "tables" / "tab_arms_metrics.tex").read_text())
    assert not any(r[0] in A.ECOLOGICAL_ARMS for r in rows)


def test_without_ecological_arms_nothing_changes_and_nothing_warns(tmp_path):
    res = A.aggregate(eco_runs(tmp_path / "r", arms=()), None, n_boot=30, seed=1, workers=1, figures=False)
    assert all((res.pooled.contrast == n).sum() == 0 for n in ECO_FORMULAS)
    assert "ehm" not in " ".join(res.issues.message).lower() and "phase-only" not in " ".join(res.issues.message).lower()
    with_eco = A.aggregate(eco_runs(tmp_path / "r2"), None, n_boot=30, seed=1, workers=1, figures=False)
    key = ["contrast", "metric", "stratum"]
    a = with_eco.pooled.merge(res.pooled, on=key, suffixes=("_a", "_b"))
    assert len(a) == len(res.pooled) > 0
    for col in ("theta", "se", "ci95_low", "ci95_high", "tau2", "p_one_sided", "p_holm"):
        assert np.array_equal(a[col + "_a"].to_numpy(), a[col + "_b"].to_numpy(), equal_nan=True), col  # adding arms changes no other contrast
