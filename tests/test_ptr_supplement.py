"""Supplementary analyses E1-E3 (``model_stgcn.ptr.supplement``): the within match-phase AUC, REML + HKSJ pooling, the sensitivity table."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score

from model_stgcn.ptr import aggregate as A
from model_stgcn.ptr import metrics as M
from model_stgcn.ptr import supplement as S
from tests.test_ptr_metrics import make_preds


# --------------------------------------------------------------------------- E2: within match-phase AUC
def brute_force(df: pd.DataFrame) -> tuple[float, int, int]:
    num = den = 0.0
    used = 0
    for _, g in df.groupby(["match_id", "phase"]):
        if g["y"].nunique() < 2:
            continue
        pos = g["y"].sum()
        pairs = pos * (len(g) - pos)
        num += roc_auc_score(g["y"], g["p_hazard"]) * pairs
        den += pairs
        used += 1
    return num / den, used, int(df.groupby(["match_id", "phase"]).ngroups)


@pytest.mark.parametrize("round_scores", [False, True])  # rounded scores: many ties inside a risk set
def test_risk_set_auc_equals_pair_weighted_mean_of_per_risk_set_auc(round_scores):
    df = make_preds(n_matches=25, seed=3, round_scores=round_scores)
    eng = S.RiskSetAUC(df)
    want, used, total = brute_force(df)
    assert eng.evaluate(np.ones(eng.n_matches)) == pytest.approx(want, abs=1e-12)
    assert (eng.n_sets_used, eng.n_sets) == (used, total)
    per_set = eng.set_auc()
    assert len(per_set) == used and per_set["auc"].between(0, 1).all()


def test_risk_set_auc_bootstrap_equals_recomputation_on_a_duplicated_frame():
    df = make_preds(n_matches=20, seed=4)
    eng = S.RiskSetAUC(df)
    counts = M.resample_counts(eng.n_matches, 5, 11)
    dups = []
    for row in counts:
        parts = []
        for mid, cnt in zip(eng.match_ids, row):
            for r in range(cnt):
                g = df[df["match_id"] == mid].copy()
                g["match_id"] = f"{mid}#{r}"
                parts.append(g)
        dups.append(pd.concat(parts, ignore_index=True))
    explicit = [S.RiskSetAUC(d).evaluate(np.ones(len(d["match_id"].unique()))) for d in dups]
    np.testing.assert_allclose(eng.evaluate(counts.astype(float)), explicit, atol=1e-12)


def test_risk_sets_with_one_label_are_excluded_and_counted():
    rows = []
    for m, labels in enumerate(([0, 0, 0], [1, 1], [0, 1, 0, 1], [0, 1, 1])):  # no positive | no negative | two pairs each | mixed
        for k, y in enumerate(labels):
            rows.append({"match_id": f"m{m}", "team_id": k, "phase": 1, "y": float(y), "p_hazard": 0.1 * (k + 1) + 0.01 * m, "rank_score": -0.1 * k,
                         "final_rank_norm": 0.1 * k, "top5": 1.0, "match_type": "official"})  # fmt: skip
    df = pd.DataFrame(rows)
    eng = S.RiskSetAUC(df)
    ex = eng.exclusions()
    assert (ex["n_risk_sets"], ex["n_risk_sets_used"], ex["n_risk_sets_no_positive"], ex["n_risk_sets_no_negative"]) == (4, 2, 1, 1)
    assert (ex["n_rows"], ex["n_rows_used"]) == (12, 7)
    assert ex["share_risk_sets_excluded"] == pytest.approx(0.5) and ex["share_rows_excluded"] == pytest.approx(5 / 12)
    assert ex["n_pairs"] == 4 + 2  # 2 * 2 + 1 * 2
    want, _, _ = brute_force(df)
    assert eng.evaluate(np.ones(eng.n_matches)) == pytest.approx(want)
    # a resample that keeps only the single-label matches has no pair: NaN, not an error
    w = np.array([1.0, 1.0, 0.0, 0.0])
    assert np.isnan(eng.evaluate(w))


def test_risk_set_auc_ignores_phase_and_match_level_differences():
    # every risk set is scored 0.5 by a score that depends on the match and phase only: the pooled AUC would be far from 0.5
    rng = np.random.default_rng(0)
    rows = []
    for m in range(12):
        for p in range(1, 6):
            for t in range(6):
                rows.append({"match_id": f"m{m}", "team_id": t, "phase": p, "y": float(rng.random() < 0.15 * p), "p_hazard": 0.1 * p + 0.01 * m,
                             "rank_score": 0.0, "final_rank_norm": 0.5, "top5": 0.0, "match_type": "official"})  # fmt: skip
    df = pd.DataFrame(rows)
    assert M.hazard_metrics(df["y"].to_numpy(), df["p_hazard"].to_numpy())["auc"] > 0.6
    assert S.RiskSetAUC(df).evaluate(np.ones(12)) == pytest.approx(0.5)


def test_e_cohort_matches_paired_contrasts_and_the_aggregator_seed(tmp_path):
    base = make_preds(n_matches=24, seed=5)
    rng = np.random.default_rng(6)
    for arm, noise in (("HGB-cut", 0.0), ("SAGONG-hgb", 0.6), ("M11R", 0.3)):
        d = base.copy()
        p = np.clip(d["p_hazard"] + noise * rng.standard_normal(len(d)) * 0.1, 1e-3, 1 - 1e-3)
        d["p_hazard"], d["rank_score"] = p, -p
        path = S.pred_path(tmp_path, "c_squad", arm)
        M.write_predictions(d, path)
    task = {"cohort": "c_squad", "arms": ("HGB-cut", "SAGONG-hgb", "M11R"), "formulas": {"x": {"HGB-cut": 1.0, "SAGONG-hgb": -1.0}}, "standard": True,
            "runs_dir": str(tmp_path), "n_boot": 60, "seed": 0}  # fmt: skip
    res = S.e_cohort(task)
    frames = S.load_frames(tmp_path, "c_squad", task["arms"])
    ref = M.paired_contrasts({a: frames[a] for a in task["arms"]}, {"x": {"HGB-cut": 1.0, "SAGONG-hgb": -1.0}}, n_boot=60, seed=A.task_seed("c_squad", "all", 0),
                             metrics=A.METRICS)  # fmt: skip
    got = pd.DataFrame(res["contrast_rows"]).set_index("metric")
    for m in ("auc", "spearman", "auc_within_phase", "cindex"):
        for key, col in (("estimate", "estimate"), ("se", "se"), ("ci_low", "ci_low"), ("ci_high", "ci_high")):
            assert got.loc[m, col] == pytest.approx(ref["contrasts"]["x"][m][key], abs=1e-12)
    # the E2 metric rides on the same resamples: its draws equal an explicit count-weighted evaluation
    counts = M.resample_counts(24, 60, A.task_seed("c_squad", "all", 0))
    e_a, e_b = S.RiskSetAUC(frames["HGB-cut"]), S.RiskSetAUC(frames["SAGONG-hgb"])
    draws = e_a.evaluate(counts) - e_b.evaluate(counts)
    est = e_a.evaluate(np.ones(24)) - e_b.evaluate(np.ones(24))
    s = M.summarise_draws(est, draws, 0.95)
    for key in ("estimate", "se", "ci_low", "ci_high"):
        assert got.loc[S.WMP, key] == pytest.approx(s[key], abs=1e-12)
    assert res["exclusions"]["n_risk_sets"] == e_a.n_sets and {r["arm"] for r in res["arm_rows"]} == set(task["arms"])


# --------------------------------------------------------------------------- E3: REML + HKSJ
def bcg() -> tuple[np.ndarray, np.ndarray]:
    t = np.array([[4, 119, 11, 128], [6, 300, 29, 274], [3, 228, 11, 209], [62, 13536, 248, 12619], [33, 5036, 47, 5761], [180, 1361, 372, 1079],
                  [8, 2537, 10, 619], [505, 87886, 499, 87892], [29, 7470, 45, 7232], [17, 1699, 65, 1600], [186, 50448, 141, 27197],
                  [5, 2493, 3, 2338], [27, 16886, 29, 17825]], dtype=float)  # fmt: skip
    a, b, c, d = t.T
    yi = np.log((a / (a + b)) / (c / (c + d)))
    vi = 1 / a - 1 / (a + b) + 1 / c - 1 / (c + d)
    return yi, vi


def test_reml_matches_metafor_on_the_bcg_data_and_maximises_the_restricted_likelihood():
    yi, vi = bcg()
    se = np.sqrt(vi)
    t2 = S.reml_tau2(yi, se)
    assert t2 == pytest.approx(0.3132, abs=5e-4)  # metafor::rma(yi, vi, data = dat.bcg), method = "REML"
    w = 1 / (vi + t2)
    assert (w * yi).sum() / w.sum() == pytest.approx(-0.7145, abs=5e-4)
    assert S.pool_dl(yi, se)["tau2"] == pytest.approx(0.3088, abs=5e-4)  # method = "DL"
    grid = np.concatenate([[0.0], np.linspace(1e-4, 3.0, 6000)])
    ll = np.array([S.reml_loglik(yi, se, g) for g in grid])
    assert abs(grid[ll.argmax()] - t2) < 1e-3 and S.reml_loglik(yi, se, t2) >= ll.max() - 1e-9


def test_reml_takes_the_global_maximum_when_the_likelihood_has_a_boundary_and_an_interior_maximum():
    # the per-cohort M11R - M11 Brier differences of the real matrix: the restricted likelihood falls from tau2 = 0 and rises again to a higher peak
    y = [-0.0001138276, 0.000173139, -4.668113e-06, -0.001000999, -0.001455018, -0.0004394606, -0.000243862, -0.0004395668, -0.0009396475, 0.0002866286,
         2.803984e-05, -0.0007861473]  # fmt: skip
    se = [0.0003106716, 0.0001029161, 1.424511e-05, 0.0003957105, 0.0005017032, 0.0003388488, 0.0002691346, 0.0004622516, 0.0005353394, 0.0004996867,
          8.350517e-05, 0.0003257675]  # fmt: skip
    t2 = S.reml_tau2(y, se)
    assert t2 > 0.0 and S.reml_loglik(y, se, t2) > S.reml_loglik(y, se, 0.0) + 1.0  # the interior peak beats the boundary by > 1 log-likelihood unit
    grid = np.concatenate([[0.0], np.linspace(1e-10, 5e-7, 20001)])
    ll = np.array([S.reml_loglik(y, se, g) for g in grid])
    assert S.reml_loglik(y, se, t2) >= ll.max() - 1e-7 and abs(grid[ll.argmax()] - t2) < 1e-10


def test_reml_is_zero_for_homogeneous_studies_and_positive_for_heterogeneous_ones():
    rng = np.random.default_rng(1)
    se = np.full(12, 0.1)
    assert S.reml_tau2(0.3 + 0.01 * rng.standard_normal(12), se) == 0.0  # far less spread than the sampling errors imply
    t2 = S.reml_tau2(0.3 + 0.5 * rng.standard_normal(12), se)
    assert t2 > 0.05


def test_hksj_interval_is_the_t_interval_with_the_hk_variance():
    yi, vi = bcg()
    se = np.sqrt(vi)
    hk = S.pool_hksj(yi, se)
    t2 = S.reml_tau2(yi, se)
    w = 1 / (vi + t2)
    theta = (w * yi).sum() / w.sum()
    q = (w * (yi - theta) ** 2).sum() / (len(yi) - 1)
    se_hk = np.sqrt(q / w.sum())
    t = stats.t.ppf(0.975, len(yi) - 1)
    assert hk["theta"] == pytest.approx(theta) and hk["se"] == pytest.approx(se_hk) and hk["df"] == len(yi) - 1
    assert (hk["ci95_low"], hk["ci95_high"]) == pytest.approx((theta - t * se_hk, theta + t * se_hk))
    assert hk["p_two_sided"] == pytest.approx(2 * stats.t.sf(abs(theta / se_hk), len(yi) - 1))
    trunc = S.pool_hksj(yi, se, truncate=True)  # max(1, q): never narrower than the Wald interval of the same REML fit
    assert trunc["se"] >= np.sqrt(1 / w.sum()) - 1e-15 and trunc["se"] == pytest.approx(np.sqrt(max(1.0, q) / w.sum()))


def test_pool_dl_matches_the_aggregators_pooling_and_drops_unusable_cohorts():
    rng = np.random.default_rng(2)
    est = 0.01 + 0.004 * rng.standard_normal(12)
    se = 0.002 + 0.001 * rng.random(12)
    est2, se2 = np.r_[est, np.nan, 0.1], np.r_[se, 0.002, 0.0]  # a NaN estimate and a zero SE are dropped
    dl = M.dersimonian_laird(est, se)
    got = S.pool_dl(est2, se2)
    assert got["k"] == 12 and got["theta"] == pytest.approx(dl["theta"]) and got["tau2"] == pytest.approx(dl["tau2"]) and got["I2"] == pytest.approx(dl["I2"])
    assert (got["ci95_low"], got["ci95_high"]) == pytest.approx(dl["ci95"])
    assert got["p_one_sided"] == pytest.approx(float(stats.norm.sf(dl["z"])))
    # a metric where lower is better flips the one-sided p
    assert S.pool_dl(est, se, -1)["p_one_sided"] == pytest.approx(float(stats.norm.sf(-dl["z"])))


def test_sensitivity_table_flags_sign_ci_and_verdict_changes():
    rng = np.random.default_rng(3)
    cohorts = [f"c{i:02d}" for i in range(12)]
    rows = []

    def add(name, metric, est, se):
        rows.extend({"cohort": c, "contrast": name, "metric": metric, "estimate": e, "se": s} for c, e, s in zip(cohorts, est, se))

    se = np.full(12, 0.002)
    add("clear", "auc", 0.03 + 0.001 * rng.standard_normal(12), se)  # large and homogeneous: gain under both methods
    add("null", "auc", 0.0 + 0.002 * rng.standard_normal(12), se)  # nothing: equivalent
    het = np.r_[np.full(6, 0.02), np.full(6, -0.012)]  # heterogeneous with a small SE: the t interval is much wider than the Wald one
    add("hetero", "auc", het, se)
    add("win", S.WMP, 0.02 + 0.001 * rng.standard_normal(12), se)  # a metric without SESOI
    df = pd.DataFrame(rows)
    sets = {("clear", "auc"): "x", ("null", "auc"): "x", ("hetero", "auc"): "x", ("win", S.WMP): "x"}
    out = S.sensitivity_table(df, sets, {}).set_index("contrast")
    assert out.loc["clear", "dl_verdict"] == "gain" == out.loc["clear", "hksj_verdict"]
    assert out.loc["null", "dl_verdict"] == "equivalent"
    assert not out.loc["clear", "sign_changed"] and not out.loc["clear", "excl0_changed"] and not out.loc["clear", "verdict_changed"]
    assert out.loc["win", "dl_verdict"] == "n/a" and out.loc["win", "hksj_verdict"] == "n/a" and out.loc["win", "verdict_illustrative_dl"] == "gain"
    h = out.loc["hetero"]
    assert h["hksj_ci95_high"] - h["hksj_ci95_low"] > h["dl_ci95_high"] - h["dl_ci95_low"] - 1e-12  # the t(k - 1) interval is not narrower here
    assert h["excl0_dl"] == (h["dl_ci95_low"] > 0 or h["dl_ci95_high"] < 0)
    assert set(out.columns) == set(S.SENS_COLUMNS) - {"contrast"}


def test_sensitivity_table_holm_is_recomputed_inside_each_method():
    rng = np.random.default_rng(4)
    cohorts = [f"c{i:02d}" for i in range(12)]
    rows, sets, primary = [], {}, {}
    for pid, mu in zip(("P1", "P2", "P3", "P4", "P5"), (0.0, 0.004, 0.01, 0.0005, 0.0032)):
        rows += [{"cohort": c, "contrast": f"t_{pid}", "metric": "auc", "estimate": mu + 0.001 * rng.standard_normal(), "se": 0.0008} for c in cohorts]
        sets[(f"t_{pid}", "auc")], primary[(f"t_{pid}", "auc")] = f"primary {pid}", pid
    out = S.sensitivity_table(pd.DataFrame(rows), sets, {}, primary)
    for tag in ("dl", "hksj"):
        p = out[f"{tag}_p_one_sided"].to_numpy()
        want = A.holm({f"P{i + 1}": v for i, v in enumerate(p)})
        np.testing.assert_allclose(out[f"{tag}_p_holm"].to_numpy(), [want[f"P{i + 1}"] for i in range(5)])


# --------------------------------------------------------------------------- the v2 run: the supplement reads the aggregator's outputs
def _aggregated(tmp_path, *, n_boot=40, seed=0, **kw):
    """Synthetic runs with every arm of the decomposition (external arms incl. the H0 arms, the no-rank-loss arms) and the aggregator run on them."""
    from tests.test_ptr_aggregate import union_runs

    root = union_runs(tmp_path / "runs", norank=True, **kw)
    results = tmp_path / "results"
    res = A.aggregate(root, results, n_boot=n_boot, seed=seed, workers=1, figures=False)
    return root, results, res


def test_the_supplement_follows_the_decomposition_of_the_aggregator():
    from model_stgcn.ptr import supplement as S2

    assert S2.E2_DECOMP == tuple(name for _, _, _, name in A.DECOMP_STEPS) and len(S2.E2_DECOMP) == 10  # steps (a)-(g) and the rank auxiliary loss group
    assert S2.E2_RQ2 == tuple(name for _, _, name in A.RQ2_ROWS) and len(S2.E2_RQ2) == 10
    assert "hgbcut_vs_sagong_hgb" not in S2.E2_RQ2 and "m01norank_vs_k3" in S2.E2_DECOMP and "k2h0_vs_k2" in S2.E2_DECOMP
    for name, (label, coefs) in S2.E1_FORMULAS.items():  # one definition: the E1 coefficients ARE the aggregator's
        assert coefs == {a: float(c) for a, c in A.FORMULAS[S2.E1_AGG_NAME[name]].coefs.items()}
    assert set(S2.E1_ORIGINAL) == {"k3_to_m01", "m01_to_m11r"}
    assert {"KRAFTON-tf-ourfeat-h0", "KRAFTON-tf-h0", "M01-norank", "M11R-norank"} <= set(S2.E2_ARMS)


def test_defaults_are_the_v2_roots_and_the_bootstrap_follows_the_aggregator(tmp_path):
    from model_stgcn.ptr import paths as P
    from model_stgcn.ptr import supplement as S2

    a = S2.parse_args([])
    assert a.runs_dir == str(P.RUNS_DIR) and a.results_dir == str(P.RESULTS_DIR) and a.out_dir is None and a.n_boot is None and a.seed is None
    root, results, _ = _aggregated(tmp_path, n_boot=30, seed=3)
    agg = S2.read_aggregator_summary(results)
    assert agg["n_boot"] == 30 and agg["seed"] == 3 and S2.read_aggregator_summary(tmp_path / "nowhere") is None
    assert S2.main(["--runs-dir", str(root), "--results-dir", str(results), "--workers", "1"]) == 0  # n_boot / seed taken from summary.json
    summary = json.loads((results / "supplement" / "supplement_summary.json").read_text())
    assert summary["n_boot"] == 30 and summary["seed"] == 3  # out-dir defaults to <results-dir>/supplement
    assert summary["aggregator"]["same_bootstrap"] is True and summary["aggregator"]["summary_found"] is True


def test_the_e1_contrasts_reproduce_the_aggregators_numbers_and_every_decomposition_row_gets_the_within_match_phase_metric(tmp_path):
    from model_stgcn.ptr import supplement as S2

    root, results, agg = _aggregated(tmp_path, n_boot=40, seed=0)
    out = tmp_path / "supp"
    res = S2.run(root, results, out, n_boot=40, workers=1, seed=0)
    chk = res["summary"]["aggregator"]["e1_reproduces_aggregator"]
    assert chk["n_compared"] > 0 and chk["max_abs_diff_estimate"] < 1e-9 and chk["max_abs_diff_se"] < 1e-9  # same arms, rows and resamples
    # the per-cohort E1 numbers ARE the aggregator's for the pairs it also has (e.g. (e) M01-norank - K3)
    pc = pd.read_csv(results / "contrasts_per_cohort.csv")
    pc = pc[pc.stratum == "all"]
    for e1, ag in S2.E1_AGG_NAME.items():
        a = pc[(pc.contrast == ag) & (pc.metric == "auc")].set_index("cohort").estimate
        b = res["e1_pc"][(res["e1_pc"].contrast == e1) & (res["e1_pc"].metric == "auc")].set_index("cohort").estimate
        assert np.allclose(a.sort_index().to_numpy(), b.sort_index().to_numpy(), atol=1e-12), e1
    # E2: the within match-phase AUC is pooled for the ten rows of tab_decomp.tex and the ten RQ2 contrasts
    pooled2 = res["e2_pooled"]
    assert set(S2.E2_FORMULAS) <= set(pooled2.contrast) and set(pooled2.metric) == {S2.WMP} and (pooled2.k == 3).all()
    # E3: P1-P5, every decomposition row and every RQ2 contrast under DL and REML + HKSJ, plus the two original steps with the rank loss
    e3 = res["e3"]
    sets = e3.groupby("set").size()
    assert sum(1 for s in sets.index if s.startswith("primary")) == 5
    assert sum(1 for s in sets.index if s.startswith("decomposition:")) == 10 and (sets[[s for s in sets.index if s.startswith("decomposition:")]] == 4).all()
    assert sets["RQ2 contrast"] == 10 * 4 and sets["E1 original step (with the rank loss)"] == 2 * 4
    assert e3[e3.set.str.startswith("decomposition:")].contrast.nunique() == 10
    for col in ("dl_theta", "hksj_theta", "hksj_ci95_low", "hksj_verdict", "excl0_changed"):
        assert col in e3.columns
    names = {p.name for p in out.iterdir()}
    assert {"e1_contrasts_per_cohort.csv", "e2_contrasts_pooled_dl.csv", "e3_pooling_sensitivity.csv", "supplement_summary.json"} <= names


def test_the_supplement_refuses_an_aggregator_output_of_other_runs_or_another_bootstrap(tmp_path):
    from model_stgcn.ptr import supplement as S2

    root, results, _ = _aggregated(tmp_path, n_boot=30, seed=0)
    with pytest.raises(ValueError, match="differ from the aggregator's"):  # other resamples: the bootstrap SEs do not reproduce
        S2.run(root, results, tmp_path / "s1", n_boot=45, workers=1, seed=0)
    elsewhere = tmp_path / "other_runs"
    elsewhere.mkdir()
    with pytest.raises(ValueError, match="was aggregated from"):  # another run directory than the one the aggregator summarised
        S2.run(elsewhere, results, tmp_path / "s2", n_boot=30, workers=1, seed=0)
    assert not (tmp_path / "s2").exists()  # refused before anything is written


def test_arms_without_a_prediction_file_are_skipped_and_named(tmp_path):
    from model_stgcn.ptr import supplement as S2

    root, results, _ = _aggregated(tmp_path, n_boot=30, seed=0)
    import shutil

    shutil.rmtree(root / "c_beta" / "KRAFTON-tf-ourfeat-h0")  # one cohort lacks K2H0 (the aggregator was run before)
    res = S2.run(root, results, tmp_path / "s", n_boot=30, workers=1, seed=0, cohorts=["c_alpha", "c_beta", "c_gamma"])
    missing = res["summary"]["missing_arms"]
    assert "KRAFTON-tf-ourfeat-h0" in missing["e2"]["c_beta"] and "KRAFTON-tf-ourfeat-h0" not in missing["e2"].get("c_alpha", []) and not missing["e1"]
    assert set(missing["e2"]["c_beta"]) == {"KRAFTON-tf-ourfeat-h0", "EHM-lite"}  # the ecological arm was never part of these synthetic runs
    k2h0 = res["e2_pc"][(res["e2_pc"].contrast == "k2h0_vs_k2") & (res["e2_pc"].metric == S2.WMP)]
    assert sorted(k2h0.cohort) == ["c_alpha", "c_gamma"]  # the contrast is computed where both arms exist


def test_the_supplement_refuses_a_v1_root_next_to_a_v2_dump(tmp_path, capsys):
    from model_stgcn.ptr import supplement as S2
    from model_stgcn.ptr.synthetic import write_cohort_dumps

    v2 = tmp_path / "dumps" / "bigcomp2027_phase_v2"
    write_cohort_dumps(v2, "c_alpha", sizes={"train": 4, "val": 3, "test": 3}, seed=1)
    man = json.loads((v2 / "c_alpha" / "dump_manifest.json").read_text())
    man["version"] = "v2"
    (v2 / "c_alpha" / "dump_manifest.json").write_text(json.dumps(man))
    v1_runs = tmp_path / "checkpoints" / "bigcomp2027_ptr_v1"
    v1_runs.mkdir(parents=True)
    results = tmp_path / "reports" / "bigcomp2027_results_v2"
    results.mkdir(parents=True)
    with pytest.raises(SystemExit, match="refusing to run"):
        S2.main(["--runs-dir", str(v1_runs), "--results-dir", str(results), "--dump-dir", str(v2)])


def test_pool_dl_degenerate_contrast_is_its_common_value():
    """A contrast with zero standard error in every cohort (identical rankings by construction) pools to its value, no p-value."""
    import numpy as np

    from model_stgcn.ptr import supplement as S

    r = S.pool_dl(np.zeros(12), np.zeros(12))
    assert r["k"] == 0 and r["theta"] == 0.0 and r["ci95_low"] == r["ci95_high"] == 0.0 and np.isnan(r["p_one_sided"])
