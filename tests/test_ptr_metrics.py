"""PTR metrics, paired match-level bootstrap and DerSimonian-Laird pooling.

References are sklearn / scipy / brute force; the weighted (bootstrap) path is checked against
metrics recomputed on an explicitly duplicated frame.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

import pubg_lab.survival.ptr.evaluation.metrics as M


def make_preds(n_matches=30, seed=0, signal=2.0, noise=1.0, round_scores=False):
    """Prediction frame in the CSV format.

    final_rank_norm: larger = worse placement.  rank_score is survival-like (higher = better survival, so
    it decreases with final_rank_norm); p_hazard is a risk (increases with final_rank_norm).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_matches):
        n = int(rng.integers(6, 14))
        order = rng.permutation(n)
        rank = order + 1  # unique ranks, 1 = winner
        frn = rank / n
        for p in range(1, 7):
            size = int(rng.integers(3, n + 1))
            teams = rng.choice(n, size=size, replace=False)
            risk = signal * (frn[teams] - 0.5) + noise * rng.standard_normal(size)
            prob = 1 / (1 + np.exp(-(risk - 1.0)))
            y = (rng.random(size) < prob).astype(float)
            score = -(frn[teams] + 0.3 * noise * rng.standard_normal(size))
            if round_scores:
                prob, score = np.round(prob, 1), np.round(score, 1)
            for k, t in enumerate(teams):
                rows.append(
                    {
                        "match_id": f"m{m:03d}",
                        "team_id": int(100 + t),
                        "phase": p,
                        "y": y[k],
                        "p_hazard": prob[k],
                        "rank_score": score[k],
                        "final_rank_norm": frn[t],
                        "top5": float(rank[t] <= 5),
                        "match_type": "official" if m % 3 else "competitive",
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- hazard
def test_hazard_metrics_match_sklearn():
    df = make_preds(seed=1)
    h = M.hazard_metrics(df["y"].to_numpy(), df["p_hazard"].to_numpy())
    y, p = df["y"].to_numpy(), df["p_hazard"].to_numpy()
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    assert h["auc"] == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    assert h["ap"] == pytest.approx(average_precision_score(y, p), abs=1e-12)
    assert h["nll"] == pytest.approx(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)), abs=1e-12)
    assert h["brier"] == pytest.approx(np.mean((p - y) ** 2), abs=1e-12)
    assert h["n"] == len(df) and h["pos_rate"] == pytest.approx(y.mean())


def test_hazard_metrics_degenerate_inputs_are_nan_not_errors():
    h = M.hazard_metrics(np.zeros(5), np.full(5, 0.2))  # a single class: ranking metrics undefined
    assert np.isnan(h["auc"]) and np.isnan(h["ap"])
    assert np.isfinite(h["nll"]) and np.isfinite(h["brier"])
    empty = M.hazard_metrics(np.array([]), np.array([]))
    assert empty["n"] == 0 and np.isnan(empty["nll"]) and np.isnan(empty["auc"])


# --------------------------------------------------------------------------- rank
def _risk(df, col):
    """rank_score is survival-like (higher = better), p_hazard is a risk."""
    return -df[col] if col == "rank_score" else df[col]


def _reference_rank_metrics(df, col):
    sp, ci = [], []
    risk = _risk(df, col)
    for _, g in df.assign(_risk=risk).groupby(["match_id", "phase"]):
        s, f = g["_risk"].to_numpy(), g["final_rank_norm"].to_numpy()
        if len(g) >= 2 and np.ptp(s) > 0 and np.ptp(f) > 0:
            sp.append(spearmanr(s, f).statistic)
        num = den = 0.0
        for i, j in itertools.permutations(range(len(g)), 2):
            if f[i] > f[j]:  # i is the worse placement: should have the higher risk
                den += 1
                num += 1.0 if s[i] > s[j] else 0.5 if s[i] == s[j] else 0.0
        if den:
            ci.append(num / den)
    top5_auc = roc_auc_score(1 - df["top5"], risk)
    return float(np.mean(sp)), float(top5_auc), float(np.mean(ci))


@pytest.mark.parametrize("round_scores", [False, True])
def test_rank_metrics_match_reference(round_scores):
    df = make_preds(seed=2, round_scores=round_scores)
    r = M.rank_metrics(df, "rank_score")
    sp, auc, ci = _reference_rank_metrics(df, "rank_score")
    assert r["spearman"] == pytest.approx(sp, abs=1e-10)
    assert r["top5_auc"] == pytest.approx(auc, abs=1e-12)
    assert r["cindex"] == pytest.approx(ci, abs=1e-12)
    r2 = M.rank_metrics(df, "p_hazard")  # same metrics driven by the hazard probability
    sp, auc, ci = _reference_rank_metrics(df, "p_hazard")
    assert (r2["spearman"], r2["top5_auc"], r2["cindex"]) == pytest.approx((sp, auc, ci), abs=1e-10)


def test_rank_direction_better_placement_means_higher_rank_score():
    """v2.2: rank_score is survival-like; every rank metric reads higher-is-better (+1 / 1 = perfect)."""
    df = make_preds(seed=3)
    df["rank_score"] = -df["final_rank_norm"]  # perfect within every group: better placement -> higher s
    r = M.rank_metrics(df, "rank_score")
    assert r["spearman"] == pytest.approx(1.0) and r["cindex"] == pytest.approx(1.0)
    assert r["top5_auc"] > 0.9  # pooled over matches of different size: rank / n is not comparable
    df["rank_score"] = df["final_rank_norm"]  # worst possible
    r = M.rank_metrics(df, "rank_score")
    assert r["spearman"] == pytest.approx(-1.0) and r["cindex"] == pytest.approx(0.0)
    assert r["top5_auc"] < 0.1
    # top-5 AUC: a survival score that is high exactly for the teams that WILL finish top 5 is perfect
    df["rank_score"] = df["top5"]
    assert M.rank_metrics(df, "rank_score")["top5_auc"] == pytest.approx(1.0)
    df["rank_score"] = 1.0 - df["top5"]
    assert M.rank_metrics(df, "rank_score")["top5_auc"] == pytest.approx(0.0)
    df["rank_score"] = 0.0  # constant score: no information
    r = M.rank_metrics(df, "rank_score")
    assert r["cindex"] == pytest.approx(0.5) and r["top5_auc"] == pytest.approx(0.5)
    assert np.isnan(r["spearman"])
    # the hazard probability is a risk: it must rank with the opposite sign
    df["p_hazard"] = df["final_rank_norm"]
    r = M.rank_metrics(df, "p_hazard")
    assert r["spearman"] == pytest.approx(1.0) and r["cindex"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="rank_col"):
        M.rank_metrics(df, "final_rank_norm")


def test_all_metrics_structure_and_csv_roundtrip(tmp_path):
    df = make_preds(seed=4)
    out = M.all_metrics(df)
    assert out["n_rows"] == len(df) and out["n_matches"] == df["match_id"].nunique()
    assert set(out["hazard"]) >= {"auc", "ap", "nll", "brier"}
    assert set(out["rank"]) >= {"spearman", "top5_auc", "cindex"}
    assert set(out["rank_from_hazard"]) >= {"spearman", "top5_auc", "cindex"}

    path = tmp_path / "p.csv"
    M.write_predictions(df.sample(frac=1.0, random_state=0), path)
    back = pd.read_csv(path)
    assert list(back.columns) == M.PRED_COLUMNS
    assert M.PRED_COLUMNS[:7] == [
        "match_id", "team_id", "phase", "y", "p_hazard", "rank_score", "final_rank_norm",
    ]  # fmt: skip
    assert M.PRED_COLUMNS[7:] == ["top5", "match_type"]  # appended: the AUC label and the stratum (v2.3)
    canon = M.read_predictions(path)
    keys = list(zip(canon["match_id"], canon["team_id"], canon["phase"]))
    assert keys == sorted(keys)  # canonical row order
    pd.testing.assert_frame_equal(
        canon.reset_index(drop=True),
        M.read_predictions(df).reset_index(drop=True),
        check_exact=False,
        rtol=1e-8,
    )
    with pytest.raises(ValueError):
        M.read_predictions(df.drop(columns=["rank_score"]))


# --------------------------------------------------------------------------- weighted core
@pytest.mark.parametrize("ties", [False, True])
def test_weighted_auc_ap_match_sklearn_sample_weight(ties):
    rng = np.random.default_rng(5)
    n = 400
    y = (rng.random(n) < 0.3).astype(float)
    s = rng.standard_normal(n) + 1.2 * y
    if ties:
        s = np.round(s, 1)
    w = rng.integers(0, 4, size=n).astype(float)  # includes zero weights
    sb = M.SortedBinary(y, s)
    assert sb.auc(w) == pytest.approx(roc_auc_score(y, s, sample_weight=w), abs=1e-12)
    assert sb.ap(w) == pytest.approx(average_precision_score(y, s, sample_weight=w), abs=1e-12)
    ones = np.ones(n)
    assert sb.auc(ones) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert sb.ap(ones) == pytest.approx(average_precision_score(y, s), abs=1e-12)
    assert np.isnan(sb.auc(np.where(y == 1, w, 0.0)))  # no negatives left -> NaN


def test_weighted_metrics_equal_metrics_on_explicitly_duplicated_frame():
    df = M.read_predictions(make_preds(seed=6, round_scores=True))
    suite = M.MetricSuite(df)
    rng = np.random.default_rng(0)
    codes = np.unique(df["match_id"], return_inverse=True)[1]
    n_matches = int(codes.max()) + 1
    counts = rng.multinomial(n_matches, np.full(n_matches, 1 / n_matches))
    assert (counts == 0).any() and (counts > 1).any()
    dup_ids = np.repeat(np.arange(n_matches), counts)
    parts = []
    for new_id, m in enumerate(dup_ids):  # each copy becomes its own match
        part = df[codes == m].copy()
        part["match_id"] = f"copy{new_id:04d}"
        parts.append(part)
    dup = M.read_predictions(pd.concat(parts, ignore_index=True))
    direct = M.all_metrics(dup)
    got = suite.evaluate(counts.astype(float))
    for name in ("auc", "ap", "nll", "brier"):
        assert got[name] == pytest.approx(direct["hazard"][name], abs=1e-10), name
    for name in ("spearman", "top5_auc", "cindex"):
        assert got[name] == pytest.approx(direct["rank"][name], abs=1e-10), name
    # identity weights give the plain metrics
    plain = M.all_metrics(df)
    ident = suite.evaluate(np.ones(n_matches))
    for name in ("auc", "ap", "nll", "brier"):
        assert ident[name] == pytest.approx(plain["hazard"][name], abs=1e-12), name
    for name in ("spearman", "top5_auc", "cindex"):
        assert ident[name] == pytest.approx(plain["rank"][name], abs=1e-12), name


# --------------------------------------------------------------------------- bootstrap
def test_bootstrap_defaults_and_resampling_is_by_match():
    assert M.N_BOOT == 2000
    counts = M.resample_counts(37, 50, seed=3)
    assert counts.shape == (50, 37) and (counts.sum(1) == 37).all() and counts.dtype.kind in "iu"
    assert (counts == M.resample_counts(37, 50, seed=3)).all()
    assert (counts != M.resample_counts(37, 50, seed=4)).any()


def test_paired_bootstrap_identical_files_have_zero_difference(tmp_path):
    df = make_preds(seed=7)
    res = M.paired_bootstrap(df, df.copy(), n_boot=60, seed=1)
    for name, r in res["metrics"].items():
        assert r["diff"] == 0 and r["se"] == pytest.approx(0, abs=1e-15), name
        assert r["ci_low"] == pytest.approx(0, abs=1e-15) and r["ci_high"] == pytest.approx(0, abs=1e-15)
    assert set(res["metrics"]) == {"auc", "ap", "nll", "brier", "spearman", "top5_auc", "cindex"}
    assert res["n_boot"] == 60 and res["n_matches"] == df["match_id"].nunique()


def test_paired_bootstrap_detects_better_model_and_point_estimates_are_plain_differences():
    good = make_preds(seed=8, signal=3.0, noise=0.5)
    bad = good.copy()
    rng = np.random.default_rng(0)
    bad["p_hazard"] = rng.random(len(bad))
    bad["rank_score"] = rng.standard_normal(len(bad))
    res = M.paired_bootstrap(good, bad, n_boot=300, seed=2)
    mg, mb = M.all_metrics(good), M.all_metrics(bad)
    assert res["metrics"]["auc"]["diff"] == pytest.approx(mg["hazard"]["auc"] - mb["hazard"]["auc"])
    assert res["metrics"]["cindex"]["diff"] == pytest.approx(mg["rank"]["cindex"] - mb["rank"]["cindex"])
    assert res["metrics"]["auc"]["ci_low"] > 0 and res["metrics"]["cindex"]["ci_low"] > 0
    assert res["metrics"]["nll"]["diff"] < 0  # lower NLL is better; diff is raw A - B
    lo, hi = res["metrics"]["auc"]["ci_low"], res["metrics"]["auc"]["ci_high"]
    assert lo <= res["metrics"]["auc"]["diff"] <= hi
    assert res["metrics"]["auc"]["se"] > 0
    # deterministic given the seed, different with another seed
    res2 = M.paired_bootstrap(good, bad, n_boot=300, seed=2)
    assert res2["metrics"]["auc"]["se"] == res["metrics"]["auc"]["se"]
    res3 = M.paired_bootstrap(good, bad, n_boot=300, seed=3)
    assert res3["metrics"]["auc"]["se"] != res["metrics"]["auc"]["se"]
    samples = M.paired_bootstrap(good, bad, n_boot=40, seed=2, return_samples=True)
    assert samples["samples"]["auc"].shape == (40,)


def test_paired_bootstrap_aligns_rows_by_key_and_rejects_mismatches(tmp_path):
    a = make_preds(seed=9)
    shuffled = a.sample(frac=1.0, random_state=1).reset_index(drop=True)
    shuffled["p_hazard"] = np.clip(shuffled["p_hazard"] * 0.9, 0, 1)
    res = M.paired_bootstrap(a, shuffled, n_boot=20, seed=0)  # row order is irrelevant
    assert res["n_rows"] == len(a)
    with pytest.raises(ValueError, match="rows"):
        M.paired_bootstrap(a, a.iloc[1:], n_boot=5)
    changed = a.copy()
    changed.loc[changed.index[0], "y"] = 1 - changed.loc[changed.index[0], "y"]
    with pytest.raises(ValueError, match="label"):
        M.paired_bootstrap(a, changed, n_boot=5)
    other = a.copy()
    other["team_id"] = other["team_id"] + 1
    with pytest.raises(ValueError, match="rows"):
        M.paired_bootstrap(a, other, n_boot=5)
    pa, pb = tmp_path / "a.csv", tmp_path / "b.csv"  # file paths are accepted
    M.write_predictions(a, pa)
    M.write_predictions(shuffled, pb)
    res = M.paired_bootstrap(pa, pb, n_boot=10, seed=0)
    assert res["n_rows"] == len(a)


# --------------------------------------------------------------------------- DerSimonian-Laird
def test_dersimonian_laird_hand_example():
    r = M.dersimonian_laird([0.1, 0.3, 0.5], [0.1, 0.1, 0.1])
    # Q = 100 * (0.04 + 0 + 0.04) = 8, df = 2, C = 300 - 30000/300 = 200 -> tau2 = 6/200
    assert r["Q"] == pytest.approx(8.0) and r["df"] == 2
    assert r["tau2"] == pytest.approx(0.03)
    assert r["theta"] == pytest.approx(0.3)
    assert r["se"] == pytest.approx((1 / 75) ** 0.5)
    assert r["ci95"] == pytest.approx((0.3 - 1.959964 * r["se"], 0.3 + 1.959964 * r["se"]), abs=1e-5)
    assert r["ci90"] == pytest.approx((0.3 - 1.644854 * r["se"], 0.3 + 1.644854 * r["se"]), abs=1e-5)
    assert r["I2"] == pytest.approx((8 - 2) / 8)
    assert r["k"] == 3
    from scipy.stats import norm

    z = 0.3 / r["se"]
    assert r["z"] == pytest.approx(z)
    assert r["p"] == pytest.approx(2 * norm.sf(z))
    assert r["p_one_sided"] == pytest.approx(norm.sf(z)) and r["p_one_sided"] == pytest.approx(r["p"] / 2)
    neg = M.dersimonian_laird([-0.1, -0.3, -0.5], [0.1, 0.1, 0.1])
    assert neg["p_one_sided"] > 0.99


def test_dersimonian_laird_homogeneous_and_single_study_and_nan_dropping():
    r = M.dersimonian_laird([0.2, 0.21, 0.19], [0.1, 0.1, 0.1])  # Q < df -> tau2 = 0
    assert r["tau2"] == 0.0 and r["I2"] == 0.0
    w = 1 / np.array([0.1, 0.1, 0.1]) ** 2
    assert r["theta"] == pytest.approx(np.sum(w * [0.2, 0.21, 0.19]) / w.sum())
    assert r["se"] == pytest.approx((1 / w.sum()) ** 0.5)
    one = M.dersimonian_laird([0.4], [0.2])
    assert one["theta"] == pytest.approx(0.4) and one["se"] == pytest.approx(0.2)
    assert one["tau2"] == 0.0 and one["k"] == 1
    dropped = M.dersimonian_laird([0.1, np.nan, 0.5], [0.1, 0.1, 0.1])
    assert dropped["k"] == 2
    with pytest.raises(ValueError):
        M.dersimonian_laird([np.nan], [0.1])
    with pytest.raises(ValueError):
        M.dersimonian_laird([0.1, 0.2], [0.1])


def test_dersimonian_laird_matches_independent_formula_on_random_data():
    rng = np.random.default_rng(10)
    for _ in range(20):
        k = int(rng.integers(2, 14))
        se = rng.uniform(0.01, 0.2, k)
        y = rng.normal(0.02, 0.1, k)
        r = M.dersimonian_laird(y, se)
        w = 1 / se**2
        fixed = (w * y).sum() / w.sum()
        q = (w * (y - fixed) ** 2).sum()
        tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w**2).sum() / w.sum()))
        ws = 1 / (se**2 + tau2)
        theta = (ws * y).sum() / ws.sum()
        assert r["tau2"] == pytest.approx(tau2, rel=1e-9, abs=1e-12)
        assert r["theta"] == pytest.approx(theta, rel=1e-9)
        assert r["se"] == pytest.approx((1 / ws.sum()) ** 0.5, rel=1e-9)
        assert r["ci95"][0] < r["theta"] < r["ci95"][1]
        assert r["ci90"][0] > r["ci95"][0] and r["ci90"][1] < r["ci95"][1]


# --------------------------------------------------------------------------- contrasts / factorial / kappa
def with_quality(df, alpha, seed):
    """Same rows and labels; hazard / rank scores whose informativeness grows with alpha."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["p_hazard"] = 1 / (1 + np.exp(-(alpha * (2 * df["y"] - 1) + rng.standard_normal(len(df)))))
    out["rank_score"] = -alpha * 2 * (df["final_rank_norm"] - 0.5) + rng.standard_normal(len(df))  # survival-like
    return out


@pytest.fixture(scope="module")
def arms4():
    base = make_preds(n_matches=40, seed=30)
    return {k: with_quality(base, a, 100 + i) for i, (k, a) in enumerate(
        {"M00": 0.2, "M10": 0.7, "M01": 0.5, "M11": 1.4}.items()
    )}  # fmt: skip


def _point(df):
    m = M.all_metrics(df)
    return {**m["hazard"], **m["rank"]}


def test_factorial_main_effect_equals_half_sum_of_simple_effects(arms4):
    a, b, c, d = arms4["M10"], arms4["M00"], arms4["M11"], arms4["M01"]
    res = M.factorial_main_effect(a, b, c, d, n_boot=80, seed=1)
    pa, pb, pc, pd_ = (_point(x) for x in (a, b, c, d))
    for name in M.METRIC_NAMES:
        want = 0.5 * ((pa[name] - pb[name]) + (pc[name] - pd_[name]))
        assert res["metrics"][name]["estimate"] == pytest.approx(want, abs=1e-12), name
        r = res["metrics"][name]
        assert r["se"] > 0 and r["ci_low"] < r["ci_high"] and r["n_valid"] == 80
    assert res["n_boot"] == 80 and res["n_matches"] == 40


def test_factorial_interaction_and_2x2_use_the_section_4_1_definitions(arms4):
    p = {k: _point(v) for k, v in arms4.items()}
    inter = M.factorial_interaction(arms4["M11"], arms4["M10"], arms4["M01"], arms4["M00"], n_boot=60, seed=2)
    full = M.factorial_2x2(arms4["M00"], arms4["M10"], arms4["M01"], arms4["M11"], n_boot=60, seed=2)
    for name in M.METRIC_NAMES:
        want = (p["M11"][name] - p["M10"][name]) - (p["M01"][name] - p["M00"][name])
        assert inter["metrics"][name]["estimate"] == pytest.approx(want, abs=1e-12), name
        c = {k: v[name]["estimate"] for k, v in full["contrasts"].items()}
        assert c["interaction"] == pytest.approx(want, abs=1e-12)
        assert c["phase_main"] == pytest.approx(0.5 * ((p["M10"][name] - p["M00"][name]) + (p["M11"][name] - p["M01"][name])), abs=1e-12)
        assert c["team_main"] == pytest.approx(0.5 * ((p["M01"][name] - p["M00"][name]) + (p["M11"][name] - p["M10"][name])), abs=1e-12)
        assert c["phase_given_team_off"] == pytest.approx(p["M10"][name] - p["M00"][name], abs=1e-12)
        assert c["phase_given_team_on"] == pytest.approx(p["M11"][name] - p["M01"][name], abs=1e-12)
        assert c["team_given_phase_off"] == pytest.approx(p["M01"][name] - p["M00"][name], abs=1e-12)
        assert c["team_given_phase_on"] == pytest.approx(p["M11"][name] - p["M10"][name], abs=1e-12)
    # one shared bootstrap: same seed -> same resamples -> same se as the stand-alone helper
    a, b, c_, d = arms4["M01"], arms4["M00"], arms4["M11"], arms4["M10"]
    team = M.factorial_main_effect(a, b, c_, d, n_boot=60, seed=2)
    assert team["metrics"]["auc"]["se"] == pytest.approx(full["contrasts"]["team_main"]["auc"]["se"])
    assert full["models"]["M11"]["auc"] == pytest.approx(p["M11"]["auc"])


def test_contrast_engine_is_consistent_with_paired_bootstrap(arms4):
    good, bad = arms4["M11"], arms4["M00"]
    single = M.paired_bootstrap(good, bad, n_boot=70, seed=3)
    # main effect of (good - bad, good - bad) is the plain difference, with identical resamples
    twice = M.factorial_main_effect(good, bad, good, bad, n_boot=70, seed=3)
    zero = M.factorial_interaction(good, bad, good, bad, n_boot=70, seed=3)
    for name in M.METRIC_NAMES:
        assert twice["metrics"][name]["estimate"] == pytest.approx(single["metrics"][name]["diff"], abs=1e-12)
        assert twice["metrics"][name]["se"] == pytest.approx(single["metrics"][name]["se"], rel=1e-9)
        assert twice["metrics"][name]["ci_low"] == pytest.approx(single["metrics"][name]["ci_low"], abs=1e-12)
        assert zero["metrics"][name]["estimate"] == pytest.approx(0.0, abs=1e-12)
        assert zero["metrics"][name]["se"] == pytest.approx(0.0, abs=1e-12)
    ident = M.factorial_main_effect(good, good, bad, bad, n_boot=20, seed=3)
    assert all(ident["metrics"][m]["estimate"] == 0 for m in M.METRIC_NAMES)


def test_paired_contrasts_validation_and_samples(arms4):
    with pytest.raises(ValueError, match="unknown models"):
        M.paired_contrasts({"x": arms4["M00"], "y": arms4["M10"]}, {"c": {"x": 1.0, "z": -1.0}}, n_boot=5)
    with pytest.raises(ValueError, match="at least two"):
        M.paired_contrasts({"x": arms4["M00"]}, {"c": {"x": 1.0}}, n_boot=5)
    with pytest.raises(ValueError, match="unknown metrics"):
        M.paired_contrasts({"x": arms4["M00"], "y": arms4["M10"]}, {"c": {"x": 1.0}}, n_boot=5, metrics=("f1",))
    shifted = arms4["M10"].copy()
    shifted["team_id"] += 1
    with pytest.raises(ValueError, match="rows"):
        M.factorial_main_effect(arms4["M00"], arms4["M10"], arms4["M01"], shifted, n_boot=5)
    res = M.paired_contrasts(
        {"x": arms4["M00"], "y": arms4["M10"]}, {"c": {"x": 1.0, "y": -1.0}},
        n_boot=25, metrics=("auc", "nll"), return_samples=True,
    )  # fmt: skip
    assert set(res["contrasts"]["c"]) == {"auc", "nll"} and res["samples"]["c"]["auc"].shape == (25,)


def test_recovery_ratio_matches_formula_and_has_sane_limits(arms4):
    a_noz, b_nohab, c_full = arms4["M10"], arms4["M00"], arms4["M11"]  # b < a < c in quality
    res = M.recovery_ratio(a_noz, b_nohab, c_full, n_boot=200, seed=4, metrics=("auc", "cindex"))
    pa, pb, pc = _point(a_noz), _point(b_nohab), _point(c_full)
    for name in ("auc", "cindex"):
        r = res["metrics"][name]
        want = (pa[name] - pb[name]) / (pc[name] - pb[name])
        assert r["kappa"] == pytest.approx(want, abs=1e-12)
        assert r["numerator"]["estimate"] == pytest.approx(pa[name] - pb[name], abs=1e-12)
        assert r["denominator"]["estimate"] == pytest.approx(pc[name] - pb[name], abs=1e-12)
        assert 0 < r["kappa"] < 1 and r["ci_low"] < r["ci_high"] and r["n_valid"] > 150
        assert r["den_nonpositive_frac"] == 0.0
        assert r["denominator"]["se"] > 0 and r["numerator"]["se"] > 0
    # limits: a == c -> kappa 1, a == b -> kappa 0 (with degenerate intervals)
    full = M.recovery_ratio(c_full, b_nohab, c_full, n_boot=30, seed=4)["metrics"]["auc"]
    assert full["kappa"] == pytest.approx(1.0) and full["ci_low"] == pytest.approx(1.0) and full["ci_high"] == pytest.approx(1.0)
    none = M.recovery_ratio(b_nohab, b_nohab, c_full, n_boot=30, seed=4)["metrics"]["auc"]
    assert none["kappa"] == pytest.approx(0.0) and none["ci_high"] == pytest.approx(0.0)
    # no denominator (c == b): not estimable, flagged instead of crashing
    flat = M.recovery_ratio(a_noz, b_nohab, b_nohab, n_boot=30, seed=4)["metrics"]["auc"]
    assert np.isnan(flat["kappa"]) and flat["n_valid"] == 0 and flat["den_nonpositive_frac"] == 1.0
    # a harmful "denominator" is reported through den_nonpositive_frac
    neg = M.recovery_ratio(a_noz, c_full, b_nohab, n_boot=30, seed=4)["metrics"]["auc"]
    assert neg["denominator"]["estimate"] < 0 and neg["den_nonpositive_frac"] > 0.9
    samples = M.recovery_ratio(a_noz, b_nohab, c_full, n_boot=15, seed=4, return_samples=True)
    assert samples["samples"]["auc"]["kappa"].shape == (15,)


# --------------------------------------------------------------------------- v2.3: match_type
def test_match_type_is_part_of_the_prediction_format_and_is_validated(tmp_path):
    df = make_preds(seed=70)
    assert df.columns[-1] == "match_type" and set(df["match_type"]) == {"official", "competitive"}
    path = tmp_path / "p.csv"
    M.write_predictions(df, path)
    back = M.read_predictions(path)
    assert list(back.columns) == M.PRED_COLUMNS and back["match_type"].dtype == object
    assert set(back["match_type"]) == {"official", "competitive"}
    with pytest.raises(ValueError, match="match_type"):
        M.read_predictions(df.drop(columns=["match_type"]))
    blank = df.copy()
    blank.loc[blank.index[0], "match_type"] = None
    with pytest.raises(ValueError, match="match_type"):
        M.read_predictions(blank)
    mixed = df.copy()  # a match must have one type
    mixed.loc[mixed.index[0], "match_type"] = "airoyale"
    with pytest.raises(ValueError, match="match_type"):
        M.read_predictions(mixed)


def test_by_match_type_metrics_equal_metrics_on_the_subset():
    df = M.read_predictions(make_preds(seed=71))
    out = M.all_metrics(df)
    assert set(out["by_match_type"]) == {"official", "competitive"}
    for mt, block in out["by_match_type"].items():
        sub = df[df["match_type"] == mt]
        direct = M.all_metrics(sub.drop(columns=[]), by_match_type=False)
        assert "by_match_type" not in direct
        assert block == direct  # the stratum block is exactly the metrics of its rows
        assert block["n_rows"] == len(sub) and block["n_matches"] == sub["match_id"].nunique()
    assert sum(b["n_rows"] for b in out["by_match_type"].values()) == out["n_rows"]
    assert "by_match_type" not in M.all_metrics(df, by_match_type=False)


def test_paired_bootstrap_can_be_restricted_to_a_match_type():
    good = M.read_predictions(make_preds(n_matches=36, seed=72, signal=3.0, noise=0.5))
    bad = good.copy()
    rng = np.random.default_rng(0)
    bad["p_hazard"] = rng.random(len(bad))
    bad["rank_score"] = rng.standard_normal(len(bad))
    full = M.paired_bootstrap(good, bad, n_boot=40, seed=1)
    assert full["match_type"] is None
    for mt in ("official", "competitive"):
        sub = M.paired_bootstrap(good, bad, n_boot=40, seed=1, match_type=mt)
        rows = good[good["match_type"] == mt]
        assert sub["match_type"] == [mt]
        assert sub["n_rows"] == len(rows) and sub["n_matches"] == rows["match_id"].nunique()
        want = M.all_metrics(rows)["hazard"]["auc"] - M.all_metrics(bad[bad["match_type"] == mt])["hazard"]["auc"]
        assert sub["metrics"]["auc"]["diff"] == pytest.approx(want, abs=1e-12)
    both = M.paired_bootstrap(good, bad, n_boot=10, seed=1, match_type=("official", "competitive"))
    assert both["n_rows"] == len(good)
    with pytest.raises(ValueError, match="match_type"):
        M.paired_bootstrap(good, bad, n_boot=5, match_type="airoyale")  # no such rows
    # contrasts and kappa accept the same filter
    res = M.factorial_main_effect(good, bad, good, bad, n_boot=10, seed=1, match_type="official")
    assert res["n_rows"] == int((good["match_type"] == "official").sum())
    rec = M.recovery_ratio(good, bad, good, n_boot=10, seed=1, match_type="competitive")
    assert rec["n_rows"] == int((good["match_type"] == "competitive").sum())


def test_paired_bootstrap_rejects_files_with_different_match_types():
    a = M.read_predictions(make_preds(seed=73))
    b = a.copy()
    flip = b["match_id"] == b["match_id"].iloc[0]
    b.loc[flip, "match_type"] = np.where(b.loc[flip, "match_type"] == "official", "competitive", "official")
    with pytest.raises(ValueError, match="match_type"):
        M.paired_bootstrap(a, b, n_boot=5)


# --------------------------------------------------------------------------- within-phase AUC (exploratory, complementary)
def phase_frame(spec, seed=0, signal=1.5):
    """One block of rows per cut phase: ``spec = {phase: (n_rows, n_positive)}``; rows are spread over 12 matches."""
    rng = np.random.default_rng(seed)
    rows = []
    for phase, (n, n_pos) in spec.items():
        y = np.zeros(n)
        y[rng.permutation(n)[:n_pos]] = 1.0
        p = 1 / (1 + np.exp(-(signal * (y - 0.5) + rng.standard_normal(n))))
        for i in range(n):
            rows.append({"match_id": f"m{i % 12:02d}", "team_id": 1000 * phase + i, "phase": phase, "y": y[i], "p_hazard": p[i],
                         "rank_score": -p[i], "final_rank_norm": 0.5, "top5": 0.0, "match_type": "official"})  # fmt: skip
    return pd.DataFrame(rows)


def per_phase_reference(df, phases=None):
    return {k: roc_auc_score(g["y"], g["p_hazard"]) for k, g in df.groupby("phase") if phases is None or k in phases}


def test_the_exploratory_metric_is_outside_the_default_set_and_metrics_json():
    assert M.EXPLORATORY_METRICS == ("auc_within_phase",) and M.MIN_PHASE_ROWS == 30
    assert M.ALL_METRIC_NAMES == M.METRIC_NAMES + ("auc_within_phase",)
    assert M.METRIC_NAMES == ("auc", "ap", "nll", "brier", "spearman", "top5_auc", "cindex")  # unchanged
    df = make_preds(seed=1)
    out = M.all_metrics(df)
    assert "auc_within_phase" not in out["hazard"] and "auc_within_phase" not in out["rank"]  # metrics.json keeps its structure
    suite = M.MetricSuite(df)
    assert list(suite.evaluate(np.ones(suite.n_matches))) == list(M.METRIC_NAMES)  # asked for only when needed
    assert set(suite.evaluate(np.ones(suite.n_matches), M.ALL_METRIC_NAMES)) == set(M.ALL_METRIC_NAMES)
    with pytest.raises(ValueError, match="unknown metrics"):
        M.paired_bootstrap(df, df.copy(), n_boot=5, metrics=("auc", "nonsense"))


def test_within_phase_auc_is_the_mean_of_the_per_phase_aucs():
    df = make_preds(seed=1)
    ref = per_phase_reference(df)
    res = M.within_phase_auc(df)
    assert res["n_phases"] == 6 and res["excluded"] == {} and set(res["phases"]) == set(ref)
    for k, v in ref.items():
        assert res["phases"][k] == pytest.approx(v, abs=1e-12)
    assert res["auc_within_phase"] == pytest.approx(np.mean(list(ref.values())), abs=1e-12)
    pooled = M.hazard_metrics(df["y"].to_numpy(), df["p_hazard"].to_numpy())["auc"]
    assert res["auc_within_phase"] != pytest.approx(pooled, abs=1e-6)  # a different quantity
    suite = M.MetricSuite(df)
    assert suite.evaluate(np.ones(suite.n_matches), ("auc_within_phase",))["auc_within_phase"] == pytest.approx(res["auc_within_phase"], abs=1e-12)


def test_within_phase_auc_is_the_unweighted_mean_over_phases_each_phase_counts_once():
    """The definition used everywhere: every eligible phase counts once, whatever its number of rows (not a row-weighted mean)."""
    big = phase_frame({1: (900, 225)}, seed=11, signal=3.5)  # one large phase with a very good score
    small = phase_frame({2: (40, 10), 3: (40, 10), 4: (40, 10)}, seed=12, signal=0.1)  # three small phases with a near-chance score
    df = pd.concat([big, small], ignore_index=True)
    res = M.within_phase_auc(df)
    per = per_phase_reference(df)
    assert sorted(res["phases"]) == [1, 2, 3, 4] and per[1] > 0.9 and max(per[2], per[3], per[4]) < 0.75
    unweighted = float(np.mean(list(per.values())))
    row_weighted = float(sum(len(g) * per[k] for k, g in df.groupby("phase")) / len(df))
    assert abs(unweighted - row_weighted) > 0.1  # the two definitions are far apart on this frame
    assert res["auc_within_phase"] == pytest.approx(unweighted, abs=1e-12)
    suite = M.MetricSuite(df)
    assert suite.evaluate(np.ones(suite.n_matches), ("auc_within_phase",))["auc_within_phase"] == pytest.approx(unweighted, abs=1e-12)
    # a 40-row phase is one term of the mean, like the 900-row phase: dropping the small phase 2 leaves the mean of the other three
    without_small = M.within_phase_auc(df[df["phase"] != 2])["auc_within_phase"]
    assert without_small == pytest.approx(np.mean([per[1], per[3], per[4]]), abs=1e-12)


def test_within_phase_auc_uses_only_phases_with_both_classes_and_at_least_30_rows():
    df = phase_frame({1: (100, 30), 2: (29, 10), 3: (30, 10), 4: (200, 0), 5: (200, 200), 6: (50, 5)}, seed=3)
    res = M.within_phase_auc(df)
    ref = per_phase_reference(df, phases={1, 3, 6})
    assert sorted(res["phases"]) == [1, 3, 6] and res["n_phases"] == 3  # 30 rows is enough, 29 is not
    assert set(res["excluded"]) == {2, 4, 5}
    assert "29 rows" in res["excluded"][2] and "single class" in res["excluded"][4] and "single class" in res["excluded"][5]
    assert res["auc_within_phase"] == pytest.approx(np.mean(list(ref.values())), abs=1e-12)
    assert M.within_phase_auc(df, min_rows=29)["n_phases"] == 4  # the threshold is a parameter, 30 is the default
    suite = M.MetricSuite(df)
    assert suite.within_phase.phases == [1, 3, 6]


def test_within_phase_auc_is_nan_when_no_phase_qualifies():
    df = phase_frame({1: (20, 5), 2: (25, 5), 3: (60, 0)}, seed=4)
    res = M.within_phase_auc(df)
    assert np.isnan(res["auc_within_phase"]) and res["n_phases"] == 0 and set(res["excluded"]) == {1, 2, 3}
    suite = M.MetricSuite(df)
    assert np.isnan(suite.evaluate(np.ones(suite.n_matches), ("auc_within_phase",))["auc_within_phase"])


def test_within_phase_auc_removes_the_phase_baseline_that_the_pooled_auc_rewards():
    rng = np.random.default_rng(5)
    rows = []
    for phase in range(1, 7):
        n = 600
        y = (rng.random(n) < 0.05 + 0.07 * phase).astype(float)  # the hazard rises with the phase
        p = 0.05 * phase + 0.01 * rng.standard_normal(n)  # a score that only knows the phase, nothing within it
        rows += [{"match_id": f"m{i % 30:02d}", "team_id": 1000 * phase + i, "phase": phase, "y": y[i], "p_hazard": p[i],
                  "rank_score": -p[i], "final_rank_norm": 0.5, "top5": 0.0, "match_type": "official"} for i in range(n)]  # fmt: skip
    df = pd.DataFrame(rows)
    pooled = M.hazard_metrics(df["y"].to_numpy(), df["p_hazard"].to_numpy())["auc"]
    within = M.within_phase_auc(df)["auc_within_phase"]
    assert pooled > 0.60 and abs(within - 0.5) < 0.03  # the baseline model looks good pooled and uninformative within phases


def test_weighted_within_phase_auc_equals_the_duplicated_frame_on_the_phases_fixed_by_the_data():
    df = M.read_predictions(phase_frame({1: (150, 30), 2: (150, 20), 3: (45, 6), 4: (150, 40)}, seed=6))
    suite = M.MetricSuite(df)
    codes = np.unique(df["match_id"], return_inverse=True)[1]
    n_matches = int(codes.max()) + 1
    rng = np.random.default_rng(1)
    counts = rng.multinomial(n_matches, np.full(n_matches, 1 / n_matches))
    assert (counts == 0).any() and (counts > 1).any()
    parts = []
    for new_id, m in enumerate(np.repeat(np.arange(n_matches), counts)):  # each copy becomes its own match
        part = df[codes == m].copy()
        part["match_id"] = f"copy{new_id:04d}"
        parts.append(part)
    dup = M.read_predictions(pd.concat(parts, ignore_index=True))
    ref = np.mean(list(per_phase_reference(dup, phases=set(suite.within_phase.phases)).values()))  # the eligible set of the ORIGINAL data
    got = suite.evaluate(counts.astype(float), ("auc_within_phase",))["auc_within_phase"]
    assert got == pytest.approx(ref, abs=1e-12)
    ident = suite.evaluate(np.ones(n_matches), ("auc_within_phase",))["auc_within_phase"]
    assert ident == pytest.approx(M.within_phase_auc(df)["auc_within_phase"], abs=1e-12)


def test_a_phase_that_loses_a_class_in_a_draw_is_skipped_in_that_draw_only():
    df = M.read_predictions(phase_frame({1: (120, 30), 2: (60, 1), 3: (120, 30)}, seed=7))  # phase 2 has ONE positive row
    suite = M.MetricSuite(df)
    assert suite.within_phase.phases == [1, 2, 3]  # eligible on the data (both classes, >= 30 rows)
    codes = suite.match_codes
    pos_match = int(codes[(df["phase"] == 2).to_numpy() & (df["y"] == 1).to_numpy()][0])
    w = np.ones(suite.n_matches)
    w[pos_match] = 0.0  # the draw misses the match of that positive: phase 2 has no AUC in this draw
    per = suite.within_phase.per_phase(w[codes])
    assert np.isnan(per[2]) and np.isfinite(per[1]) and np.isfinite(per[3])
    assert suite.evaluate(w, ("auc_within_phase",))["auc_within_phase"] == pytest.approx(np.mean([per[1], per[3]]), abs=1e-12)
    assert np.isfinite(suite.evaluate(np.ones(suite.n_matches), ("auc_within_phase",))["auc_within_phase"])


def test_within_phase_auc_in_the_paired_bootstrap_and_in_a_match_type_stratum():
    good = make_preds(seed=8, signal=3.0, noise=0.5)
    bad = good.copy()
    bad["p_hazard"] = np.random.default_rng(0).random(len(bad))
    res = M.paired_bootstrap(good, bad, n_boot=80, seed=2, metrics=("auc", "auc_within_phase"))
    want = M.within_phase_auc(good)["auc_within_phase"] - M.within_phase_auc(bad)["auc_within_phase"]
    m = res["metrics"]["auc_within_phase"]
    assert m["diff"] == pytest.approx(want, abs=1e-12) and m["a"] == pytest.approx(M.within_phase_auc(good)["auc_within_phase"], abs=1e-12)
    assert m["ci_low"] > 0 and m["se"] > 0 and m["n_valid"] == 80 and m["ci_low"] <= m["diff"] <= m["ci_high"]
    same = M.paired_bootstrap(good, good.copy(), n_boot=30, seed=1, metrics=("auc_within_phase",))["metrics"]["auc_within_phase"]
    assert same["diff"] == 0 and same["se"] == pytest.approx(0, abs=1e-15)
    # a stratum fixes its own eligible phases from its own rows
    strat = M.paired_bootstrap(good, bad, n_boot=40, seed=2, metrics=("auc_within_phase",), match_type="competitive")["metrics"]["auc_within_phase"]
    comp = lambda d: M.within_phase_auc(d[d["match_type"] == "competitive"])["auc_within_phase"]  # noqa: E731
    assert strat["diff"] == pytest.approx(comp(good) - comp(bad), abs=1e-12)
    # the other metrics are untouched by asking for it too (same resamples, same draws)
    both = M.paired_bootstrap(good, bad, n_boot=80, seed=2, metrics=("auc", "auc_within_phase"))["metrics"]["auc"]
    alone = M.paired_bootstrap(good, bad, n_boot=80, seed=2, metrics=("auc",))["metrics"]["auc"]
    assert both == alone
