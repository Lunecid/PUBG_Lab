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

from model_stgcn.ptr import aggregate as A
from model_stgcn.ptr import metrics as M
from model_stgcn.ptr.synthetic import write_prediction_runs

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
        for metric in M.METRIC_NAMES:
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
    assert res.contrasts.cohort.nunique() == 4 and len(res.contrasts[res.contrasts.contrast == "rel_vs_shuf"]) == 4 * 7 * 3


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


def test_tables_have_the_structure_of_the_paper_skeletons(full):
    res, root, out = full
    skeleton_path = Path(__file__).resolve().parents[1] / "paper" / "bigcomp2027" / "sections" / "05-experiments.tex"
    if not skeleton_path.is_file():
        pytest.skip("paper skeleton not present")
    skeleton = skeleton_path.read_text()
    blocks = re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", skeleton, re.S)
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
