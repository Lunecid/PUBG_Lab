"""The ecological hazard model (EHM): logistic engine, covariates, arms, run directories and the report.  DB-free.

Planted truth: dumps whose hazard labels follow a logistic model of the cut phase and the habitat columns (the real dump names
are written into a synthetic dump), so that the fit can be compared with the coefficients that generated the data."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from model_stgcn.ptr import data as D
from model_stgcn.ptr import ecomodel as E
from model_stgcn.ptr import hgb as H
from model_stgcn.ptr import metrics as M
from model_stgcn.ptr.synthetic import make_dump

COLUMNS = {  # real dump name -> column of the synthetic dump that carries it
    "hab_signed_boundary_distance_end_m": 3, "hab_required_speed_to_safe_mps": 4, "hab_local_enemy_count_200m": 5,
    "hab_alive_teams": 6, "alive_players_end_norm": 7, "hab_migration_deficit_mps": 8,
}  # fmt: skip
TRUTH = {  # raw-scale coefficients of the planted hazard (logit)
    "outside_depth": 9.0, "inside_cushion": -2.0, "required_speed": 3.0, "local_crowding": 4.0, "log_alive_teams": 0.0, "team_members": -0.6,
}  # fmt: skip
PHASE_LOGIT = {1: -1.6, 2: -1.3, 3: -0.9, 4: -0.5, 5: -0.1, 6: 0.3, 7: 0.7}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ehm_dump(n_matches, seed, *, split="train", cohort="syn", interaction=0.0, solo=False, truth=TRUTH):
    """A synthetic dump with the real habitat names and labels drawn from the planted logistic hazard."""
    dump = make_dump(n_matches, seed=seed, n_features=24, edge_prob=0.0, cohort=cohort, split=split, team_range=(6, 12))
    names = dump["meta"]["feature_names"]
    for name, c in COLUMNS.items():
        names[c] = name
    rng = np.random.default_rng(seed + 777)
    rng_deficit = np.random.default_rng(seed + 999)  # its own stream: adding the column leaves every other draw as it was
    for m in dump["matches"]:
        m.pop("team_won", None)  # no winner censoring: the planted labels are the truth
        P, N = m["node_mask"].shape
        tgt = m["target_mask_v2"].numpy()
        d = rng.normal(0.02, 0.1, size=(P, N))
        out, ins = np.maximum(0.0, -d), np.maximum(0.0, d)
        speed = out * rng.uniform(0.0, 1.0, size=(P, N)) * (rng.random((P, N)) < 0.8)
        crowd = rng.poisson(0.9, size=(P, N)) / 20.0
        at_risk = tgt.sum(axis=1)  # teams at risk at the cut
        teams = np.repeat((at_risk / 100.0)[:, None], N, axis=1)
        members = np.full((P, N), 0.25) if solo else rng.integers(1, 5, size=(P, N)) / 4.0
        X = m["X"]
        X[..., COLUMNS["hab_signed_boundary_distance_end_m"]] = torch.from_numpy(d).float()
        X[..., COLUMNS["hab_required_speed_to_safe_mps"]] = torch.from_numpy(speed).float()
        X[..., COLUMNS["hab_local_enemy_count_200m"]] = torch.from_numpy(crowd).float()
        X[..., COLUMNS["hab_alive_teams"]] = torch.from_numpy(teams).float()
        X[..., COLUMNS["alive_players_end_norm"]] = torch.from_numpy(members).float()
        X[..., COLUMNS["hab_migration_deficit_mps"]] = torch.from_numpy(-0.05 + out + 0.08 * rng_deficit.normal(size=(P, N))).float()
        k = np.arange(1, P + 1)[:, None] * np.ones((1, N))  # cut phase of slot p is p + 1
        logit = (
            np.vectorize(PHASE_LOGIT.get)(k.astype(int), -0.0)
            + truth["outside_depth"] * out + truth["inside_cushion"] * ins + truth["required_speed"] * speed
            + truth["local_crowding"] * crowd + truth["team_members"] * members + interaction * out * (k - E.PHASE_CENTER)
        )  # fmt: skip
        y = (rng.random((P, N)) < sigmoid(logit)).astype(np.float32)
        m["y_hazard_v2"] = torch.from_numpy(np.where(tgt, y, 0.0)).float()
    return dump


def write_cohort(root, cohort, sizes, seed, **kw):
    base = root / cohort
    base.mkdir(parents=True, exist_ok=True)
    for i, (split, n) in enumerate(sizes.items()):
        torch.save(ehm_dump(n, seed * 100 + i, split=split, cohort=cohort, **kw), base / f"{split}.pt")
    return base


@pytest.fixture(scope="module")
def big():
    """Train / val / test dumps of one cohort with the planted hazard, and their HGB-cut rows / inputs."""
    dumps = {s: ehm_dump(n, 11 + i, split=s) for i, (s, n) in enumerate({"train": 500, "val": 120, "test": 120}.items())}
    rows = {s: H.build_rows(d, "HGB-cut", "v2") for s, d in dumps.items()}
    return dumps, rows, {s: E.make_inputs(r) for s, r in rows.items()}


# --------------------------------------------------------------------------- the logistic engine
def random_clustered(n=4000, groups=150, k=4, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, groups, size=n)
    X = np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))])
    shared = rng.normal(0.0, 0.6, size=groups)[g]  # cluster effect: the model-based SE is too small, the robust one is not
    beta = np.array([-0.4, 0.8, -0.5, 0.3][:k])
    y = (rng.random(n) < sigmoid(X @ beta + shared)).astype(float)
    return X, y, g


def test_logit_matches_statsmodels_coefficients_and_both_standard_errors():
    sm = pytest.importorskip("statsmodels.api")
    X, y, g = random_clustered()
    fit = E.fit_logit(X, y, groups=g)
    plain = sm.Logit(y, X).fit(disp=0)
    robust = sm.Logit(y, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": g})
    assert fit.converged and fit.penalty == "none" and fit.l2 == 0.0 and fit.fallback_reason == ""
    np.testing.assert_allclose(fit.beta, plain.params, atol=1e-7)
    np.testing.assert_allclose(fit.se_model, plain.bse, rtol=1e-6)
    np.testing.assert_allclose(fit.se_robust, robust.bse, rtol=1e-6)  # the same small-sample factor as statsmodels' default
    assert fit.se_robust[0] > 1.4 * fit.se_model[0]  # the planted cluster effect shifts the intercept: only the robust SE sees it
    assert fit.loglik == pytest.approx(plain.llf, abs=1e-6)


def test_logit_matches_sklearn_without_penalty():
    from sklearn.linear_model import LogisticRegression

    X, y, _ = random_clustered(seed=3)
    fit = E.fit_logit(X, y)
    ref = LogisticRegression(penalty=None, fit_intercept=False, tol=1e-12, max_iter=1000).fit(X, y)
    np.testing.assert_allclose(fit.beta, ref.coef_[0], atol=1e-6)
    assert fit.cov_robust is None and np.isnan(fit.se_robust).all()  # no groups, no robust covariance


def test_the_l2_fit_is_the_optimum_of_the_penalised_likelihood_and_shrinks():
    from scipy.optimize import minimize

    X, y, _ = random_clustered(seed=4)
    free = E.fit_logit(X, y)
    lam = 3.0

    def negative_objective(b):  # -loglik + lam/2 * |b|^2 over the non-intercept columns
        eta = X @ b
        return -float(np.sum(y * eta - np.logaddexp(0.0, eta))) + 0.5 * lam * float(b[1:] @ b[1:])

    def gradient(b):
        g = -(X.T @ (y - sigmoid(X @ b)))
        g[1:] += lam * b[1:]
        return g

    ref = minimize(negative_objective, np.zeros(X.shape[1]), jac=gradient, method="BFGS", options={"gtol": 1e-10, "maxiter": 500})
    ridge = E.fit_logit(X, y, l2=lam)
    np.testing.assert_allclose(ridge.beta, ref.x, atol=1e-6)
    assert ridge.penalty == "l2" and ridge.l2 == lam and ridge.converged
    assert np.all(np.abs(ridge.beta[1:]) < np.abs(free.beta[1:]))  # shrunk; the intercept is free
    assert ridge.loglik < free.loglik  # the unpenalised log-likelihood is at its maximum only without the penalty


def test_separation_falls_back_to_a_tiny_l2_and_says_so():
    rng = np.random.default_rng(1)
    n = 600
    dummy = (np.arange(n) % 6 == 0).astype(float)
    X = np.column_stack([np.ones(n), dummy, rng.normal(size=n)])
    y = (rng.random(n) < 0.35).astype(float)
    y[dummy == 1] = 1.0  # every row of the dummy is an event: complete separation of that coefficient
    assert not E.fit_logit(X, y).converged  # the unpenalised optimum is at infinity
    fit = E.fit_logit_auto(X, y, groups=np.arange(n) // 5)
    assert fit.penalty == "l2" and fit.l2 == E.L2_LAMBDA and fit.converged
    assert "did not converge" in fit.fallback_reason and np.isfinite(fit.beta).all() and float(np.max(np.abs(fit.beta))) < E.COEF_LIMIT
    ok = E.fit_logit_auto(*random_clustered(seed=5)[:2])
    assert ok.penalty == "none" and ok.fallback_reason == ""  # an ordinary problem is never penalised


def hard_newton_case():
    """A heavy-tailed problem on which a plain Newton iteration (no step halving) diverges from the intercept-only start."""
    rng = np.random.default_rng(89)
    n = int(rng.integers(40, 400))
    x = rng.standard_t(1.5, size=(n, 2)) * rng.uniform(0.5, 4)
    beta = rng.normal(0, 3, size=3)
    X = np.column_stack([np.ones(n), x])
    return X, (rng.random(n) < sigmoid(X @ beta)).astype(float)


def test_a_full_newton_step_that_overshoots_is_halved_and_the_fit_still_converges():
    X, y = hard_newton_case()
    fit = E.fit_logit(X, y)
    p = sigmoid(X @ fit.beta)
    assert fit.converged and fit.penalty == "none" and np.isfinite(fit.beta).all()
    assert float(np.max(np.abs(X.T @ (y - p)))) < 1e-6  # the score is zero at the solution: the maximum likelihood estimate
    sm = pytest.importorskip("statsmodels.api")
    np.testing.assert_allclose(fit.beta, sm.Logit(y, X).fit(disp=0, method="newton", maxiter=200).params, atol=1e-5)


def test_a_coefficient_beyond_the_limit_counts_as_non_convergence(monkeypatch):
    X, y, _ = random_clustered(seed=7)
    assert E.fit_logit(X, y).converged
    monkeypatch.setattr(E, "COEF_LIMIT", 0.5)  # the planted coefficients (0.8, -0.5, ...) are now "diverged"
    assert not E.fit_logit(X, y).converged
    auto = E.fit_logit_auto(X, y)
    assert auto.penalty == "l2" and "largest |coefficient|" in auto.fallback_reason


def test_the_coefficient_bound_can_be_applied_per_standard_deviation_of_the_design_column():
    # a column with a tiny spread has a large coefficient that is only a unit effect (0.08 per SD of the column): not separation
    rng = np.random.default_rng(3)
    n = 30000
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n) * 0.002])
    y = (rng.random(n) < sigmoid(-0.5 + 0.5 * X[:, 1] + 40.0 * X[:, 2])).astype(float)
    free = E.fit_logit(X, y)
    assert float(np.max(np.abs(free.beta))) > E.COEF_LIMIT and not free.converged and free.newton_converged  # flagged on the raw scale
    auto = E.fit_logit_auto(X, y)
    assert auto.penalty == "l2" and "left the coefficient bound" in auto.fallback_reason and "did not converge" not in auto.fallback_reason
    scaled = E.fit_logit(X, y, coef_scale=X.std(axis=0))
    assert scaled.converged and scaled.penalty == "none" and scaled.newton_converged
    np.testing.assert_allclose(scaled.beta, free.beta)  # the same maximum likelihood estimate: only the verdict differs
    assert E.fit_logit_auto(X, y, coef_scale=X.std(axis=0)).penalty == "none"
    # real separation is still caught with the scaled bound (the iterations never converge)
    n = 600
    dummy = (np.arange(n) % 6 == 0).astype(float)
    Xs = np.column_stack([np.ones(n), dummy, rng.normal(size=n)])
    ys = (rng.random(n) < 0.35).astype(float)
    ys[dummy == 1] = 1.0
    sep = E.fit_logit(Xs, ys, coef_scale=Xs.std(axis=0))
    assert not sep.converged and not sep.newton_converged
    assert "did not converge" in E.fit_logit_auto(Xs, ys, coef_scale=Xs.std(axis=0)).fallback_reason


def test_robust_covariance_is_brute_force_sandwich():
    X, y, g = random_clustered(n=500, groups=40, seed=6)
    fit = E.fit_logit(X, y, groups=g)
    p = sigmoid(X @ fit.beta)
    bread = np.linalg.inv((X * (p * (1 - p))[:, None]).T @ X)
    meat = np.zeros((X.shape[1],) * 2)
    for c in np.unique(g):
        s = (X[g == c] * (y - p)[g == c][:, None]).sum(axis=0)
        meat += np.outer(s, s)
    n, k, G = X.shape[0], X.shape[1], len(np.unique(g))
    np.testing.assert_allclose(fit.cov_robust, (G / (G - 1)) * ((n - 1) / (n - k)) * bread @ meat @ bread, rtol=1e-8)


# --------------------------------------------------------------------------- the covariates
def test_the_habitat_terms_are_built_from_the_named_dump_columns(big):
    dumps, rows, inputs = big
    r, inp = rows["train"], inputs["train"]
    col = lambda name: r.X[:, r.feature_names.index(name)].astype(np.float64)  # noqa: E731
    d = col("hab_signed_boundary_distance_end_m")
    assert list(inp.raw.columns) == list(E.TERMS) and len(inp) == len(r.y)
    np.testing.assert_array_equal(inp.raw["outside_depth"], np.maximum(0, -d))  # positive outside (d < 0)
    np.testing.assert_array_equal(inp.raw["inside_cushion"], np.maximum(0, d))  # positive inside (d > 0)
    assert (inp.raw["outside_depth"] * inp.raw["inside_cushion"] == 0).all() and (d < 0).any() and (d > 0).any()
    np.testing.assert_array_equal(inp.raw["required_speed"], col("hab_required_speed_to_safe_mps"))
    np.testing.assert_array_equal(inp.raw["local_crowding"], col("hab_local_enemy_count_200m"))
    np.testing.assert_allclose(inp.raw["log_alive_teams"], np.log(np.maximum(100.0 * col("hab_alive_teams"), 1.0)))
    np.testing.assert_array_equal(inp.raw["team_members"], col("alive_players_end_norm"))
    np.testing.assert_array_equal(inp.phase, r.keys["phase"].to_numpy())
    assert inp.phase.min() == 1 and inp.phase.max() <= 7
    assert len(np.unique(inp.match)) == r.keys["match_id"].nunique()


def test_log_alive_teams_is_the_log_of_the_count_and_never_minus_infinity(big):
    _, rows, _ = big
    r = rows["train"]
    j = r.feature_names.index("hab_alive_teams")
    r2 = H.Rows(r.keys, r.X.copy(), r.y, r.final_rank_norm, r.top5, r.match_type, r.feature_names, r.removal)
    r2.X[:5, j] = 0.0  # a cell without the value: the dump rule is NULL -> 0
    inp = E.make_inputs(r2)
    assert np.isfinite(inp.raw["log_alive_teams"]).all() and (inp.raw["log_alive_teams"][:5] == 0.0).all()  # log(max(100 x, 1)) = log 1
    r2.X[7, j] = 0.12  # 12 teams
    assert E.make_inputs(r2).raw["log_alive_teams"][7] == pytest.approx(np.log(12.0))


def test_a_missing_dump_column_is_an_error_that_names_it(big):
    dumps, rows, _ = big
    r = rows["train"]
    names = [n if n != "hab_local_enemy_count_200m" else "other" for n in r.feature_names]
    broken = H.Rows(r.keys, r.X, r.y, r.final_rank_norm, r.top5, r.match_type, names, r.removal)
    with pytest.raises(ValueError, match="hab_local_enemy_count_200m"):
        E.make_inputs(broken)


def test_non_finite_habitat_values_are_refused(big):
    _, rows, _ = big
    r = rows["train"]
    X = r.X.copy()
    X[3, r.feature_names.index("hab_required_speed_to_safe_mps")] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        E.make_inputs(H.Rows(r.keys, X, r.y, r.final_rank_norm, r.top5, r.match_type, r.feature_names, r.removal))


# --------------------------------------------------------------------------- the arms
def test_phase_only_predicts_the_train_base_rate_of_every_cut_phase(big):
    _, rows, inputs = big
    model = E.fit_arm(E.PHASE_ONLY, inputs["train"], rows["train"].y)
    assert model.used == [] and model.terms[0] == "intercept" and model.terms[1:] == [f"phase_{k}" for k in model.phases[1:]]
    p = E.predict(model, inputs["train"])
    for k in model.phases:
        m = inputs["train"].phase == k
        assert p[m] == pytest.approx(rows["train"].y[m].mean(), abs=1e-7)  # the MLE of a saturated phase model is the empirical rate
        assert len(np.unique(np.round(p[m], 12))) == 1
    pt = E.predict(model, inputs["test"])
    df = H._prediction_frame(rows["test"], pt)
    w = M.within_phase_auc(df)
    assert all(v == 0.5 for v in w["phases"].values()) and w["auc_within_phase"] == 0.5  # ties only: nothing within a phase
    assert M.hazard_metrics(df["y"].to_numpy(), df["p_hazard"].to_numpy())["auc"] > 0.55  # but the phase baseline ranks rows across phases


def test_design_columns_and_the_interaction_only_in_ehm_int(big):
    _, rows, inputs = big
    y = rows["train"].y
    ehm = E.fit_arm(E.EHM, inputs["train"], y)
    inte = E.fit_arm(E.EHM_INT, inputs["train"], y)
    assert ehm.terms == ["intercept", *[f"phase_{k}" for k in ehm.phases[1:]], *E.TERMS]
    assert inte.terms == [*ehm.terms, E.INTERACTION] and E.INTERACTION == "outside_depth:phase"
    X_e, X_i = E.design_matrix(ehm, inputs["train"]), E.design_matrix(inte, inputs["train"])
    assert X_e.shape == (len(y), len(ehm.terms)) and X_i.shape == (len(y), len(inte.terms))
    np.testing.assert_array_equal(X_i[:, :-1], X_e)  # the same columns, one more
    z_out = X_e[:, ehm.terms.index("outside_depth")]
    np.testing.assert_allclose(X_i[:, -1], z_out * (inputs["train"].phase - E.PHASE_CENTER))  # standardised depth x (k - 4)
    assert E.PHASE_CENTER == 4
    # phase dummies: reference = the first phase, one column per other phase
    for j, k in enumerate(ehm.phases[1:], start=1):
        np.testing.assert_array_equal(X_e[:, j], (inputs["train"].phase == k).astype(float))
    assert (X_e[:, 0] == 1).all()


def test_the_habitat_terms_are_standardised_on_the_train_rows_only(big):
    _, rows, inputs = big
    model = E.fit_arm(E.EHM, inputs["train"], rows["train"].y)
    X_tr = E.design_matrix(model, inputs["train"])
    X_te = E.design_matrix(model, inputs["test"])
    first = 1 + len(model.phases) - 1
    for j, t in enumerate(model.used):
        assert X_tr[:, first + j].mean() == pytest.approx(0.0, abs=1e-9) and X_tr[:, first + j].std() == pytest.approx(1.0, abs=1e-9)
        want = (inputs["test"].raw[t].to_numpy() - inputs["train"].raw[t].mean()) / inputs["train"].raw[t].std(ddof=0)
        np.testing.assert_allclose(X_te[:, first + j], want)  # the TEST rows use the train mean / sd
    assert abs(X_te[:, first].mean()) > 1e-6  # so the test columns are not centred by themselves


def test_a_constant_term_is_dropped_and_recorded(tmp_path):
    dump = ehm_dump(60, 5, solo=True)  # team size 1: alive_players_end_norm is the constant 0.25
    rows = H.build_rows(dump, "HGB-cut", "v2")
    inp = E.make_inputs(rows)
    model = E.fit_arm(E.EHM, inp, rows.y)
    assert "team_members" not in model.used and model.dropped == {"team_members": "constant in the train rows (zero variance)"}
    assert "team_members" not in model.terms and len(model.terms) == 1 + len(model.phases) - 1 + 5
    assert np.isfinite(model.fit.beta).all() and model.fit.converged
    inte = E.fit_arm(E.EHM_INT, inp, rows.y)
    assert E.INTERACTION in inte.terms and "team_members" in inte.dropped


def test_the_fit_recovers_the_planted_coefficients_and_the_arms_order_by_discrimination(big):
    dumps, rows, inputs = big
    y = rows["train"].y
    ehm = E.fit_arm(E.EHM, inputs["train"], y)
    assert ehm.fit.converged and ehm.fit.penalty == "none"
    coefs = {c["term"]: c for c in ehm.coefficients()}
    sd = ehm.sd
    for term, raw in TRUTH.items():
        want = raw * sd[term]  # the planted effect per SD of the term
        c = coefs[term]
        assert abs(c["beta"] - want) < 4.0 * c["se_robust"], (term, c["beta"], want, c["se_robust"])
    assert coefs["outside_depth"]["beta"] > 0 and coefs["inside_cushion"]["beta"] < 0 and coefs["local_crowding"]["beta"] > 0
    assert coefs["log_alive_teams"]["se_robust"] > 0 and abs(coefs["log_alive_teams"]["beta"]) < 4 * coefs["log_alive_teams"]["se_robust"]  # planted 0
    auc = {}
    for arm in (E.PHASE_ONLY, E.EHM):
        model = ehm if arm == E.EHM else E.fit_arm(arm, inputs["train"], y)
        df = H._prediction_frame(rows["test"], E.predict(model, inputs["test"]))
        auc[arm] = (M.hazard_metrics(df["y"].to_numpy(), df["p_hazard"].to_numpy())["auc"], M.within_phase_auc(df)["auc_within_phase"])
    assert auc[E.EHM][0] > auc[E.PHASE_ONLY][0] + 0.03 and auc[E.EHM][1] > 0.6 and auc[E.PHASE_ONLY][1] == 0.5


def test_the_interaction_is_recovered_with_its_sign_and_vanishes_without_one():
    steep = {s: ehm_dump(n, 31 + i, split=s, interaction=4.0) for i, (s, n) in enumerate({"train": 600, "val": 100, "test": 100}.items())}
    flat = {s: ehm_dump(n, 41 + i, split=s, interaction=0.0) for i, (s, n) in enumerate({"train": 600, "val": 100, "test": 100}.items())}
    for dumps, planted in ((steep, 4.0), (flat, 0.0)):
        r = H.build_rows(dumps["train"], "HGB-cut", "v2")
        inp = E.make_inputs(r)
        model = E.fit_arm(E.EHM_INT, inp, r.y)
        c = {x["term"]: x for x in model.coefficients()}[E.INTERACTION]
        want = planted * model.sd["outside_depth"]  # per SD of depth per phase step
        assert abs(c["beta"] - want) < 4.0 * c["se_robust"], (planted, c["beta"], want, c["se_robust"])
        if planted:
            assert c["beta"] > 0 and c["beta"] / c["se_robust"] > 4  # the edge gradient steepens as the habitat shrinks
        else:
            assert abs(c["beta"] / c["se_robust"]) < 4


def test_rows_of_an_unseen_phase_take_the_nearest_train_phase(big):
    _, rows, inputs = big
    y = rows["train"].y
    keep = inputs["train"].phase <= 5
    sub = E.Inputs(inputs["train"].phase[keep], inputs["train"].raw[keep].reset_index(drop=True), inputs["train"].match[keep])
    model = E.fit_arm(E.EHM, sub, y[keep])
    assert model.phases == [1, 2, 3, 4, 5]
    late = E.Inputs(np.array([6, 7, 5]), inputs["train"].raw.iloc[:3].reset_index(drop=True), np.array([0, 1, 2]))
    assert E.n_unseen_phase_rows(model, late) == 2
    same = E.Inputs(np.array([5, 5, 5]), late.raw, late.match)
    np.testing.assert_allclose(E.predict(model, late), E.predict(model, same))  # 6 and 7 are coded as 5 (the dummies only)


# --------------------------------------------------------------------------- EHM-lite: no collinear edge terms
def test_ehm_lite_keeps_phase_speed_crowding_alive_teams_and_members_only(big):
    _, rows, inputs = big
    y = rows["train"].y
    lite = E.fit_arm(E.EHM_LITE, inputs["train"], y)
    ehm = E.fit_arm(E.EHM, inputs["train"], y)
    assert E.ARMS == ("PHASE-only", "EHM", "EHM-int", "EHM-lite")
    assert E.LITE_TERMS == ("required_speed", "local_crowding", "log_alive_teams", "team_members")
    assert lite.used == list(E.LITE_TERMS) and lite.dropped == {}
    assert lite.terms == ["intercept", *[f"phase_{k}" for k in lite.phases[1:]], *E.LITE_TERMS]
    assert "outside_depth" not in lite.terms and "inside_cushion" not in lite.terms and E.INTERACTION not in lite.terms
    X_l, X_e = E.design_matrix(lite, inputs["train"]), E.design_matrix(ehm, inputs["train"])
    for t in E.LITE_TERMS:  # the very same z-scored columns as in EHM (standardised on the same train rows)
        np.testing.assert_array_equal(X_l[:, lite.terms.index(t)], X_e[:, ehm.terms.index(t)])
    assert lite.fit.converged and lite.fit.penalty == "none" and lite.fit.cov_robust is not None  # same fitting, same SEs
    sm = pytest.importorskip("statsmodels.api")
    fit = sm.Logit(y, X_l).fit(disp=0)
    np.testing.assert_allclose(lite.fit.beta, fit.params, atol=1e-6)
    np.testing.assert_allclose(lite.fit.se_model, fit.bse, rtol=1e-5)


def test_ehm_lite_drops_the_constant_term_too_and_the_edge_terms_add_discrimination_when_planted(big):
    dumps, rows, inputs = big
    solo = ehm_dump(60, 5, solo=True)
    r = H.build_rows(solo, "HGB-cut", "v2")
    lite = E.fit_arm(E.EHM_LITE, E.make_inputs(r), r.y)
    assert lite.used == ["required_speed", "local_crowding", "log_alive_teams"] and lite.dropped == {"team_members": "constant in the train rows (zero variance)"}
    auc = {}
    for arm in (E.EHM, E.EHM_LITE):
        model = E.fit_arm(arm, inputs["train"], rows["train"].y)
        df = H._prediction_frame(rows["test"], E.predict(model, inputs["test"]))
        auc[arm] = M.hazard_metrics(df["y"].to_numpy(), df["p_hazard"].to_numpy())["auc"]
    assert auc[E.EHM] > auc[E.EHM_LITE] + 0.01  # the planted outside-depth / cushion effects live only in the full model


def test_ehm_lite_recovers_planted_coefficients_when_the_edge_terms_are_not_in_the_truth():
    truth = {**TRUTH, "outside_depth": 0.0, "inside_cushion": 0.0}
    dumps = {s: ehm_dump(n, 61 + i, split=s, truth=truth) for i, (s, n) in enumerate({"train": 600, "val": 80, "test": 80}.items())}
    r = H.build_rows(dumps["train"], "HGB-cut", "v2")
    inp = E.make_inputs(r)
    lite = E.fit_arm(E.EHM_LITE, inp, r.y)
    coefs = {c["term"]: c for c in lite.coefficients()}
    for term in E.LITE_TERMS:
        want = truth[term] * lite.sd[term]
        assert abs(coefs[term]["beta"] - want) < 4.0 * coefs[term]["se_robust"], (term, coefs[term]["beta"], want)
    assert coefs["required_speed"]["beta"] > 0 and coefs["local_crowding"]["beta"] > 0 and coefs["team_members"]["beta"] < 0
    ehm = E.fit_arm(E.EHM, inp, r.y)  # and the full model has nothing to add: its edge terms are zero within their standard errors
    e = {c["term"]: c for c in ehm.coefficients()}
    assert abs(e["outside_depth"]["beta"]) < 4.0 * e["outside_depth"]["se_robust"] and abs(e["inside_cushion"]["beta"]) < 4.0 * e["inside_cushion"]["se_robust"]


def test_vif_is_the_inverse_correlation_diagonal_one_for_a_single_term_and_lower_without_the_edge_terms(big):
    _, _, inputs = big
    inp = inputs["train"]
    six = E.vif(inp, list(E.TERMS))
    np.testing.assert_allclose([six[t] for t in E.TERMS], np.diag(np.linalg.inv(inp.raw.corr().to_numpy())), rtol=1e-8)
    assert E.vif(inp, ["required_speed"]) == {"required_speed": 1.0} and E.vif(inp, []) == {}
    lite = E.vif(inp, list(E.LITE_TERMS))
    np.testing.assert_allclose(
        [lite[t] for t in E.LITE_TERMS], np.diag(np.linalg.inv(inp.raw[list(E.LITE_TERMS)].corr().to_numpy())), rtol=1e-8
    )
    assert max(lite.values()) < max(six.values())  # removing the correlated edge terms removes the inflation


# --------------------------------------------------------------------------- diagnostics: collinearity and single-term coefficients
def test_diagnostics_give_correlations_vifs_and_the_phase_adjusted_single_term_fits(big):
    _, rows, inputs = big
    y = rows["train"].y
    diag = E.diagnostics(inputs["train"], y)
    raw = inputs["train"].raw
    assert diag["terms"] == list(E.TERMS)
    ref = raw.corr().to_numpy()
    for i, a in enumerate(E.TERMS):
        assert diag["corr"][a][a] == pytest.approx(1.0)
        for j, b in enumerate(E.TERMS):
            assert diag["corr"][a][b] == pytest.approx(ref[i, j], abs=1e-9)
    np.testing.assert_allclose([diag["vif"][t] for t in E.TERMS], np.diag(np.linalg.inv(ref)), rtol=1e-8)
    assert all(v >= 1.0 for v in diag["vif"].values())
    # a single-term fit is PHASE-only + that one standardised term
    t = "required_speed"
    z = (raw[t].to_numpy() - raw[t].mean()) / raw[t].std(ddof=0)
    ph = inputs["train"].phase
    X = np.column_stack([np.ones(len(y))] + [(ph == k).astype(float) for k in sorted(np.unique(ph))[1:]] + [z])
    fit = E.fit_logit(X, y, groups=inputs["train"].match)
    assert diag["single_term"][t]["beta"] == pytest.approx(fit.beta[-1], abs=1e-9)
    assert diag["single_term"][t]["se_robust"] == pytest.approx(fit.se_robust[-1], abs=1e-9)


def test_collinear_terms_give_a_suppression_pair_in_the_joint_model_but_not_alone():
    """Two terms with r ~ 0.95 and one planted effect: alone each has the (shared) positive effect; jointly the weights split."""
    rng = np.random.default_rng(3)
    n, n_match = 30000, 600
    latent = rng.normal(size=n)
    a = latent + 0.3 * rng.normal(size=n)
    b = latent + 0.3 * rng.normal(size=n)  # r(a, b) ~ 0.92
    match = rng.integers(0, n_match, size=n)
    phase = rng.integers(1, 5, size=n)
    y = (rng.random(n) < sigmoid(-1.0 + 0.8 * b)).astype(float)  # only b matters
    raw = pd.DataFrame({t: rng.normal(size=n) for t in E.TERMS})
    raw["outside_depth"], raw["required_speed"] = a, b
    inp = E.Inputs(phase, raw, match)
    diag = E.diagnostics(inp, y)
    assert diag["corr"]["outside_depth"]["required_speed"] > 0.85 and diag["vif"]["outside_depth"] > 5 and diag["vif"]["required_speed"] > 5
    alone_a, alone_b = diag["single_term"]["outside_depth"]["beta"], diag["single_term"]["required_speed"]["beta"]
    assert alone_a > 0.2 and alone_b > alone_a  # a only borrows b's effect when alone
    joint = E.fit_arm(E.EHM, inp, y)
    coefs = {c["term"]: c["beta"] for c in joint.coefficients()}
    assert coefs["required_speed"] > 0.5 > abs(coefs["outside_depth"]) + 0.2  # jointly the planted term takes the weight


def test_the_report_writes_the_diagnostic_tables(reported):
    runs, out, summary = reported
    coll = pd.read_csv(out / "collinearity.csv")
    single = pd.read_csv(out / "single_term_coefficients.csv")
    pooled = pd.read_csv(out / "single_term_pooled.csv")
    assert set(coll.cohort) == {"c_squad", "c_duo", "c_solo"} and {"vif", "corr_outside_depth", "corr_required_speed"} <= set(coll.columns)
    assert set(coll[coll.cohort == "c_solo"].term) == set(E.TERMS) - {"team_members"}  # the constant term has no collinearity row
    np.testing.assert_allclose(coll[coll.term == "outside_depth"].corr_outside_depth, 1.0, atol=1e-9)  # a term with itself
    assert set(single.term) == set(E.TERMS) and len(single) == 3 * 6 - 1
    assert set(pooled.term) == set(E.TERMS) and pooled.set_index("term").loc["team_members", "n_cohorts"] == 2
    g = single[single.term == "outside_depth"]
    ref = M.dersimonian_laird(g.beta.to_numpy(), g.se_robust.to_numpy())
    assert pooled.set_index("term").loc["outside_depth", "theta"] == pytest.approx(ref["theta"], abs=1e-12)
    res = json.loads((runs / "c_squad" / "EHM" / "seed42" / "metrics.json").read_text())
    assert "diagnostics" in res and "diagnostics" not in json.loads((runs / "c_squad" / "PHASE-only" / "seed42" / "metrics.json").read_text())
    r = summary["corr_outside_depth_required_speed"]
    assert set(r) == {"min", "median", "max"} and r["min"] <= r["median"] <= r["max"] and len(summary["single_term_pooled"]) == 6


# --------------------------------------------------------------------------- run directories (standard format)
@pytest.fixture(scope="module")
def cohort_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("eco")
    dumps, runs = root / "dumps", root / "runs"
    for cohort, seed, kw in (("c_squad", 1, {}), ("c_duo", 2, {}), ("c_solo", 3, {"solo": True})):
        write_cohort(dumps, cohort, {"train": 160, "val": 40, "test": 40}, seed, **kw)
        E.run_cohort(dumps, cohort, runs)
        H.run(dumps, cohort, "HGB-cut", 42, runs)  # the reference arm of the report (same rows)
    return dumps, runs


def test_run_cohort_writes_the_standard_format_on_the_rows_of_hgb_cut(cohort_runs):
    dumps, runs = cohort_runs
    ref = {s: M.read_predictions(runs / "c_squad" / "HGB-cut" / "seed42" / f"pred_{s}.csv") for s in ("val", "test")}
    for arm in E.ARMS:
        d = runs / "c_squad" / arm / "seed42"
        assert sorted(p.name for p in d.iterdir()) == ["metrics.json", "pred_test.csv", "pred_val.csv"]
        for s in ("val", "test"):
            df = M.read_predictions(d / f"pred_{s}.csv")
            M.align_predictions(df, ref[s])  # identical (match, team, phase) rows and labels as HGB-cut
            assert list(pd.read_csv(d / f"pred_{s}.csv").columns) == M.PRED_COLUMNS
            assert df["p_hazard"].between(0, 1).all() and np.isfinite(df["p_hazard"]).all()
            np.testing.assert_allclose(df["rank_score"], -df["p_hazard"])  # no rank head: ranking by hazard, like the tabular arms
        res = json.loads((d / "metrics.json").read_text())
        assert res["arm"] == arm and res["kind"] == "ecological" and res["labels"] == "v2" and res["seed"] == 42 and res["cohort"] == "c_squad"
        assert res["label_fields"] == {"y": "y_hazard_v2", "target": "target_mask_v2"} and len(res["config_hash"]) == 16
        assert res["n_matches"] == {"train": 160, "val": 40, "test": 40} and res["n_train_rows"] > 100
        assert {c["term"] for c in res["covariates"]} == set(E.TERMS) and res["covariates"][0]["column"] == "hab_signed_boundary_distance_end_m"
        m = res["model"]
        assert m["penalty"] == "none" and m["converged"] is True and m["l2_lambda"] == 0.0 and m["reference_phase"] == 1
        assert set(m["standardisation"]) == set(E.TERMS) and m["phase_center"] == 4
        assert [c["term"] for c in res["coefficients"]] == m["terms"] and res["n_features"] == len(m["terms"])
        assert res["val"]["hazard"]["n"] > 0 and res["test"]["within_phase"]["n_phases"] >= 1 and res["n_unseen_phase_rows"] == {"val": 0, "test": 0}
        want = M.all_metrics(M.read_predictions(d / "pred_test.csv"))["hazard"]["auc"]
        assert res["test"]["hazard"]["auc"] == pytest.approx(want, abs=1e-12)


def test_the_run_directories_are_read_by_the_aggregator_scan(cohort_runs):
    from model_stgcn.ptr import aggregate as A

    _, runs = cohort_runs
    found, issues = A.scan_runs(runs)
    arms = {(r.cohort, r.arm) for r in found}
    assert {("c_squad", a) for a in E.ARMS} <= arms and all(r.labels == "v2" and r.config_hash for r in found)
    assert not [i for i in issues if "ecological" in str(i) or "PHASE" in str(i)]


def test_the_config_hash_follows_the_dump_and_the_arm(cohort_runs, tmp_path):
    dumps, runs = cohort_runs
    h = {a: json.loads((runs / "c_squad" / a / "seed42" / "metrics.json").read_text())["config_hash"] for a in E.ARMS}
    assert len(set(h.values())) == 4
    E.run_cohort(dumps, "c_squad", tmp_path / "again", arms=(E.EHM,))
    assert json.loads((tmp_path / "again" / "c_squad" / "EHM" / "seed42" / "metrics.json").read_text())["config_hash"] == h[E.EHM]  # deterministic
    p1 = pd.read_csv(runs / "c_squad" / "EHM" / "seed42" / "pred_test.csv")
    p2 = pd.read_csv(tmp_path / "again" / "c_squad" / "EHM" / "seed42" / "pred_test.csv")
    pd.testing.assert_frame_equal(p1, p2)  # no randomness anywhere


def test_ehm_lite_run_directory_records_its_terms_and_the_vif(cohort_runs):
    dumps, runs = cohort_runs
    res = json.loads((runs / "c_squad" / "EHM-lite" / "seed42" / "metrics.json").read_text())
    assert res["arm"] == "EHM-lite" and res["kind"] == "ecological" and res["model"]["used_covariates"] == list(E.LITE_TERMS)
    assert res["model"]["terms"][-4:] == list(E.LITE_TERMS) and res["n_features"] == len(res["model"]["terms"])
    assert set(res["vif"]) == set(E.LITE_TERMS) and all(v >= 1.0 for v in res["vif"].values()) and "diagnostics" not in res
    full = json.loads((runs / "c_squad" / "EHM" / "seed42" / "metrics.json").read_text())
    assert set(full["vif"]) == set(E.TERMS) and max(full["vif"].values()) >= max(res["vif"].values())
    solo = json.loads((runs / "c_solo" / "EHM-lite" / "seed42" / "metrics.json").read_text())
    assert set(solo["vif"]) == set(E.LITE_TERMS) - {"team_members"}


def test_the_solo_cohort_drops_the_constant_term_and_says_so(cohort_runs):
    _, runs = cohort_runs
    res = json.loads((runs / "c_solo" / "EHM" / "seed42" / "metrics.json").read_text())
    assert "team_members" in res["model"]["dropped"] and "team_members" not in res["model"]["terms"]
    assert json.loads((runs / "c_squad" / "EHM" / "seed42" / "metrics.json").read_text())["model"]["dropped"] == {}


# --------------------------------------------------------------------------- the report
def test_pooled_dl_counts_signs_and_matches_the_metrics_helper():
    beta, se = [0.30, 0.25, -0.05, 0.40, 0.10], [0.05, 0.04, 0.06, 0.08, 0.05]
    out = E.pooled_dl(beta, se)
    ref = M.dersimonian_laird(beta, se)
    assert out["k"] == 5 and out["n_positive"] == 4 and out["n_negative"] == 1 and out["same_sign_as_pooled"] == 4
    assert out["theta"] == pytest.approx(ref["theta"]) and (out["ci95_low"], out["ci95_high"]) == pytest.approx(ref["ci95"])
    assert out["ci95_low"] < out["theta"] < out["ci95_high"] and out["tau2"] == pytest.approx(ref["tau2"])
    neg = E.pooled_dl([-0.3, -0.2, -0.1], [0.05, 0.05, 0.05])
    assert neg["same_sign_as_pooled"] == 3 and neg["n_positive"] == 0 and neg["theta"] < 0
    assert E.pooled_dl([float("nan")], [0.1]) == {"k": 0}  # nothing usable


def test_share_of_discrimination_formula():
    assert E.share_of_discrimination(0.75, 0.80) == pytest.approx(0.25 / 0.30)
    assert E.share_of_discrimination(0.5, 0.8) == 0.0 and E.share_of_discrimination(0.8, 0.8) == 1.0
    assert E.share_of_discrimination(0.85, 0.80) > 1.0  # an arm can exceed the reference
    assert np.isnan(E.share_of_discrimination(0.7, 0.5)) and np.isnan(E.share_of_discrimination(0.7, float("nan")))


@pytest.fixture(scope="module")
def reported(cohort_runs, tmp_path_factory):
    _, runs = cohort_runs
    rng = np.random.default_rng(0)
    for cohort in ("c_squad", "c_duo", "c_solo"):  # the other arms of the comparison: noisy copies of HGB-cut
        ref = M.read_predictions(runs / cohort / "HGB-cut" / "seed42" / "pred_test.csv")
        for arm, noise in (("M11R", 0.3), ("KRAFTON-tf-ourfeat-bce", 0.8), ("KRAFTON-tf", 1.2), ("SAGONG-hgb", 1.6)):
            df = ref.copy()
            df["p_hazard"] = 1 / (1 + np.exp(-(np.log(df["p_hazard"] / (1 - df["p_hazard"])) + noise * rng.standard_normal(len(df)))))
            df["rank_score"] = -df["p_hazard"]
            M.write_predictions(df, runs / cohort / arm / "seed42" / "pred_test.csv")
    out = tmp_path_factory.mktemp("report")
    summary = E.report(runs, out, ["c_squad", "c_duo", "c_solo"], n_boot=60)
    return runs, out, summary


def test_the_auc_comparison_has_every_arm_pooled_and_within_phase(reported):
    runs, out, _ = reported
    t = pd.read_csv(out / "auc_comparison.csv")
    assert set(t.arm) == {*E.ARMS, *E.COMPARISON_ARMS} and len(t) == 3 * 9 and E.EHM_LITE in set(t.arm)
    assert {"cohort", "arm", "auc", "auc_within_phase", "n_phases_within", "n_rows"} <= set(t.columns)
    row = t[(t.cohort == "c_squad") & (t.arm == "EHM")].iloc[0]
    df = M.read_predictions(runs / "c_squad" / "EHM" / "seed42" / "pred_test.csv")
    assert row.auc == pytest.approx(M.hazard_metrics(df["y"].to_numpy(), df["p_hazard"].to_numpy())["auc"], abs=1e-12)
    assert row.auc_within_phase == pytest.approx(M.within_phase_auc(df)["auc_within_phase"], abs=1e-12)
    assert (t[t.arm == "PHASE-only"].auc_within_phase == 0.5).all()  # nothing within a phase


def test_share_of_discrimination_table_per_cohort_and_pooled(reported):
    _, out, _ = reported
    s = pd.read_csv(out / "share_of_discrimination.csv")
    auc = pd.read_csv(out / "auc_comparison.csv")
    assert set(s.cohort) == {"c_squad", "c_duo", "c_solo", "pooled"} and set(s.metric) == {"auc", "auc_within_phase"} and set(s.arm) == set(E.ARMS)
    for arm in E.ARMS:
        r = s[(s.cohort == "c_squad") & (s.arm == arm) & (s.metric == "auc")].iloc[0]
        a = auc[(auc.cohort == "c_squad") & (auc.arm == arm)].auc.iloc[0]
        h = auc[(auc.cohort == "c_squad") & (auc.arm == "HGB-cut")].auc.iloc[0]
        assert r.share == pytest.approx((a - 0.5) / (h - 0.5), abs=1e-9) and r.auc_arm == pytest.approx(a) and r.auc_reference == pytest.approx(h)
        assert r.reference == "HGB-cut" and r.ci_low <= r.share <= r.ci_high + 1e-9
        pooled = s[(s.cohort == "pooled") & (s.arm == arm) & (s.metric == "auc")].iloc[0]
        ma, mh = auc[auc.arm == arm].auc.mean(), auc[auc.arm == "HGB-cut"].auc.mean()
        assert pooled.share == pytest.approx((ma - 0.5) / (mh - 0.5), abs=1e-9)  # ratio of the cross-cohort means
    w = s[(s.cohort == "pooled") & (s.arm == "PHASE-only") & (s.metric == "auc_within_phase")].iloc[0]
    assert w.share == 0.0 and w.ci_low == 0.0 and w.ci_high == 0.0  # nothing within a phase
    ehm, phase = (s[(s.cohort == "pooled") & (s.arm == a) & (s.metric == "auc")].share.iloc[0] for a in (E.EHM, E.PHASE_ONLY))
    assert 0 < phase < ehm  # the habitat terms add to the phase baseline (a share above 1 is possible: the planted truth IS the EHM)


def test_coefficient_tables_pool_the_cohorts_with_sign_agreement(reported):
    runs, out, summary = reported
    c = pd.read_csv(out / "coefficients_per_cohort.csv")
    p = pd.read_csv(out / "coefficients_pooled.csv")
    assert set(c.arm) == {E.EHM, E.EHM_INT, E.EHM_LITE} and {"beta", "se_robust", "se_model", "z_robust", "kind"} <= set(c.columns)
    np.testing.assert_allclose(c.z_robust, c.beta / c.se_robust)  # the z statistic uses the cluster-robust SE
    for cohort in ("c_squad", "c_solo"):  # and the table carries the two standard errors of the run's metrics.json, not one for both
        res = json.loads((runs / cohort / "EHM" / "seed42" / "metrics.json").read_text())
        for coef in res["coefficients"]:
            row = c[(c.cohort == cohort) & (c.arm == E.EHM) & (c.term == coef["term"])].iloc[0]
            assert row.beta == pytest.approx(coef["beta"], abs=1e-12) and row.se_robust == pytest.approx(coef["se_robust"], abs=1e-12)
            assert row.se_model == pytest.approx(coef["se_model"], abs=1e-12) and row.se_robust != row.se_model
    ehm = p[p.arm == E.EHM].set_index("term")
    assert set(ehm.index) == set(E.TERMS)
    assert ehm.loc["team_members", "n_cohorts"] == 2 and ehm.loc["outside_depth", "n_cohorts"] == 3  # the solo cohort has no team-size term
    for term in E.TERMS:
        g = c[(c.arm == E.EHM) & (c.term == term)]
        ref = M.dersimonian_laird(g.beta.to_numpy(), g.se_robust.to_numpy())
        assert ehm.loc[term, "theta"] == pytest.approx(ref["theta"], abs=1e-12) and ehm.loc[term, "ci95_low"] == pytest.approx(ref["ci95"][0], abs=1e-12)
        sign = np.sign(ref["theta"])
        assert ehm.loc[term, "same_sign_as_pooled"] == int((np.sign(g.beta) == sign).sum())
    assert ehm.loc["outside_depth", "theta"] > 0 and ehm.loc["outside_depth", "same_sign_as_pooled"] == 3  # planted
    assert E.INTERACTION in set(p[p.arm == E.EHM_INT].term) and E.INTERACTION not in set(ehm.index)
    assert (c[c.kind == "phase"].term.str.startswith("phase_")).all() and not set(p.term) & {"intercept", "phase_2"}
    assert pd.read_csv(out / "coefficients_pooled_model_se.csv").se_type.eq("se_model").all() and p.se_type.eq("se_robust").all()
    fi = pd.read_csv(out / "fit_info.csv")
    assert len(fi) == 12 and fi.penalty.eq("none").all() and fi.converged.all()
    assert summary["penalty_fallbacks"] == [] and "c_solo/EHM" in summary["dropped_terms"]
    assert json.loads((out / "summary.json").read_text())["cohorts"] == ["c_squad", "c_duo", "c_solo"]


def test_auc_difference_ehm_minus_lite_per_cohort_and_pooled_with_a_paired_bootstrap(reported):
    runs, out, summary = reported
    d = pd.read_csv(out / "auc_difference_ehm_minus_lite.csv")
    auc = pd.read_csv(out / "auc_comparison.csv")
    assert set(d.cohort) == {"c_squad", "c_duo", "c_solo", "pooled"} and set(d.metric) == {"auc", "auc_within_phase"}
    assert {"difference", "ci_low", "ci_high", "auc_a", "auc_b", "a", "b"} <= set(d.columns) and set(d.a) == {"EHM"} and set(d.b) == {"EHM-lite"}
    for metric, col in (("auc", "auc"), ("auc_within_phase", "auc_within_phase")):
        for cohort in ("c_squad", "c_duo", "c_solo"):
            a = auc[(auc.cohort == cohort) & (auc.arm == "EHM")][col].iloc[0]
            b = auc[(auc.cohort == cohort) & (auc.arm == "EHM-lite")][col].iloc[0]
            r = d[(d.cohort == cohort) & (d.metric == metric)].iloc[0]
            assert r.difference == pytest.approx(a - b, abs=1e-9) and r.auc_a == pytest.approx(a) and r.auc_b == pytest.approx(b)
            assert r.ci_low <= r.difference <= r.ci_high + 1e-9
        pooled = d[(d.cohort == "pooled") & (d.metric == metric)].iloc[0]
        ma, mb = auc[auc.arm == "EHM"][col].mean(), auc[auc.arm == "EHM-lite"][col].mean()
        assert pooled.difference == pytest.approx(ma - mb, abs=1e-9) and pooled.ci_low <= pooled.difference <= pooled.ci_high + 1e-9
    # the interval is the percentile interval of the PAIRED difference of the draws (both arms on the same match resamples)
    draws = {}
    for cohort in ("c_squad", "c_duo", "c_solo"):
        frames = {a: M.read_predictions(runs / cohort / a / "seed42" / "pred_test.csv") for a in ("EHM", "EHM-lite")}
        _, _, samples = M.bootstrap_arms(frames, metrics=("auc",), n_boot=60, seed=E._cohort_seed(cohort))
        draws[cohort] = samples
        want = np.percentile(samples["EHM"]["auc"] - samples["EHM-lite"]["auc"], [2.5, 97.5])
        r = d[(d.cohort == cohort) & (d.metric == "auc")].iloc[0]
        assert (r.ci_low, r.ci_high) == pytest.approx(tuple(want), abs=1e-12)
    pooled_draws = np.mean([draws[c]["EHM"]["auc"] for c in draws], axis=0) - np.mean([draws[c]["EHM-lite"]["auc"] for c in draws], axis=0)
    r = d[(d.cohort == "pooled") & (d.metric == "auc")].iloc[0]
    assert (r.ci_low, r.ci_high) == pytest.approx(tuple(np.percentile(pooled_draws, [2.5, 97.5])), abs=1e-12)
    assert summary["ehm_minus_lite"]["auc"]["difference"] == pytest.approx(d[(d.cohort == "pooled") & (d.metric == "auc")].difference.iloc[0])
    v = pd.read_csv(out / "vif.csv")
    assert set(v.arm) == {"EHM", "EHM-int", "EHM-lite"} and {"cohort", "arm", "term", "vif"} <= set(v.columns) and (v.vif >= 1.0).all()
    assert summary["max_vif"]["EHM-lite"] == pytest.approx(v[v.arm == "EHM-lite"].vif.max()) and summary["max_vif"]["EHM"] == pytest.approx(v[v.arm == "EHM"].vif.max())
    assert set(pd.read_csv(out / "coefficients_pooled.csv").query("arm == 'EHM-lite'").term) == set(E.LITE_TERMS)


def test_the_cli_fits_and_reports(tmp_path):
    dumps, runs, out = tmp_path / "dumps", tmp_path / "runs", tmp_path / "out"
    write_cohort(dumps, "c_a", {"train": 60, "val": 20, "test": 20}, 8)
    H.run(dumps, "c_a", "HGB-cut", 42, runs)
    assert E.main(["--dump-dir", str(dumps), "--runs-dir", str(runs), "--out-dir", str(out), "--n-boot", "0"]) == 0
    assert (out / "auc_comparison.csv").is_file() and (runs / "c_a" / "EHM-int" / "seed42" / "pred_test.csv").is_file()
    s = pd.read_csv(out / "share_of_discrimination.csv")
    assert s.ci_low.isna().all()  # --n-boot 0: points only
    only = tmp_path / "only"
    assert E.main(["--dump-dir", str(dumps), "--runs-dir", str(runs), "--out-dir", str(only), "--stage", "report", "--n-boot", "0"]) == 0
    assert (only / "summary.json").is_file()
    # --arms refits only the named arms and leaves the other run directories untouched
    before = (runs / "c_a" / "EHM" / "seed42" / "metrics.json").read_text()
    (runs / "c_a" / "EHM-lite" / "seed42" / "pred_test.csv").unlink()
    assert E.main(["--dump-dir", str(dumps), "--runs-dir", str(runs), "--out-dir", str(only), "--stage", "fit", "--arms", "EHM-lite"]) == 0
    assert (runs / "c_a" / "EHM-lite" / "seed42" / "pred_test.csv").is_file()
    assert (runs / "c_a" / "EHM" / "seed42" / "metrics.json").read_text() == before
    with pytest.raises(SystemExit):
        E.parse_args(["--dump-dir", "d", "--runs-dir", "r", "--out-dir", "o", "--arms", "EHM,bogus"])


# =========================================================================== RQ1: exposure, movement burden, competition, team size
RQ1_SLOPES = {  # raw-scale planted slopes (early, mid, late) of the RQ1 hazard: only outside depth changes with the phase group
    "outside_depth": (0.0, 8.0, 16.0), "inside_cushion": (-3.0, -3.0, -3.0), "migration_deficit": (2.0, 2.0, 2.0),
    "local_crowding": (3.0, 3.0, 3.0), "log_alive_teams": (0.0, 0.0, 0.0), "team_members": (-0.6, -0.6, -0.6),
}  # fmt: skip


def rq1_dump(n_matches, seed, *, split="train", cohort="syn", slopes=RQ1_SLOPES, solo=False, spread=None):
    """A synthetic dump (real habitat names incl. hab_migration_deficit_mps) whose hazard has the planted phase-group slopes.

    ``spread`` = (early, mid, late) factors on the spread of the signed boundary distance (the real late-phase zones are small, so the
    exposure terms hardly vary there); None leaves every draw as it was."""
    dump = make_dump(n_matches, seed=seed, n_features=24, edge_prob=0.0, cohort=cohort, split=split, team_range=(6, 12))
    names = dump["meta"]["feature_names"]
    for name, c in COLUMNS.items():
        names[c] = name
    rng = np.random.default_rng(seed + 555)
    for m in dump["matches"]:
        m.pop("team_won", None)
        P, N = m["node_mask"].shape
        tgt = m["target_mask_v2"].numpy()
        d = rng.normal(0.02, 0.1, size=(P, N))
        if spread is not None:
            slot_group = np.where(np.arange(1, P + 1) <= 2, 0, np.where(np.arange(1, P + 1) <= 5, 1, 2))
            d = 0.02 + (d - 0.02) * np.asarray(spread, dtype=float)[slot_group][:, None]
        out, ins = np.maximum(0.0, -d), np.maximum(0.0, d)
        deficit = -0.05 + out + 0.08 * rng.normal(size=(P, N))  # r ~ 0.5 with the exposure term, like the real dump (0.3 - 0.5)
        crowd = rng.poisson(0.9, size=(P, N)) / 20.0
        at_risk = tgt.sum(axis=1)
        teams = np.repeat((at_risk / 100.0)[:, None], N, axis=1)
        members = np.full((P, N), 0.25) if solo else rng.integers(1, 5, size=(P, N)) / 4.0
        X = m["X"]
        for name, values in (("hab_signed_boundary_distance_end_m", d), ("hab_migration_deficit_mps", deficit), ("hab_local_enemy_count_200m", crowd),
                             ("hab_alive_teams", teams), ("alive_players_end_norm", members), ("hab_required_speed_to_safe_mps", out * 0.5)):  # fmt: skip
            X[..., COLUMNS[name]] = torch.from_numpy(values).float()
        k = np.arange(1, P + 1)[:, None] * np.ones((1, N), dtype=int)
        group = np.where(k <= 2, 0, np.where(k <= 5, 1, 2))
        raw = {"outside_depth": out, "inside_cushion": ins, "migration_deficit": deficit, "local_crowding": crowd,
               "log_alive_teams": np.log(np.maximum(teams * 100.0, 1.0)), "team_members": members}  # fmt: skip
        logit = np.vectorize(PHASE_LOGIT.get)(k.astype(int), 0.0).astype(float)
        for term, (e, mid, late) in slopes.items():
            logit = logit + np.choose(group, [e, mid, late]) * raw[term]
        y = (rng.random((P, N)) < sigmoid(logit)).astype(np.float32)
        m["y_hazard_v2"] = torch.from_numpy(np.where(tgt, y, 0.0)).float()
    return dump


@pytest.fixture(scope="module")
def rq1_big():
    dumps = {s: rq1_dump(n, 71 + i, split=s) for i, (s, n) in enumerate({"train": 800, "val": 100, "test": 100}.items())}
    rows = {s: H.build_rows(d, "HGB-cut", "v2") for s, d in dumps.items()}
    return dumps, rows, {s: E.make_rq1_inputs(r) for s, r in rows.items()}


RQ1_SPREAD = (1.0, 0.45, 0.15)  # the spread of the signed distance in the early / mid / late groups


def _exposure_sd(factor, n=2_000_000):
    """SD of max(0, -d) for d ~ N(0.02, 0.1 * factor): the within-group SD of the raw outside depth."""
    z = np.random.default_rng(0).standard_normal(n)
    return float(np.maximum(0.0, -(0.02 + 0.1 * factor * z)).std())


@pytest.fixture(scope="module")
def rq1_scaled():
    """Per-unit slopes of the outside depth that make its slope per within-group SD equal (0.5) in the three groups."""
    slopes = dict(RQ1_SLOPES, outside_depth=tuple(0.5 / _exposure_sd(f) for f in RQ1_SPREAD))
    dumps = {s: rq1_dump(n, 91 + i, split=s, slopes=slopes, spread=RQ1_SPREAD) for i, (s, n) in enumerate({"train": 1200, "val": 100, "test": 100}.items())}
    rows = {s: H.build_rows(d, "HGB-cut", "v2") for s, d in dumps.items()}
    return dumps, rows, {s: E.make_rq1_inputs(r) for s, r in rows.items()}


def test_rq1_terms_columns_phase_groups_and_the_dropped_required_speed():
    assert E.RQ1_TERMS == ("outside_depth", "inside_cushion", "migration_deficit", "local_crowding", "log_alive_teams", "team_members")
    assert "required_speed" not in E.RQ1_TERMS  # collinear with exposure
    cols = {c.term: c.column for c in E.RQ1_COVARIATES}
    assert cols["migration_deficit"] == "hab_migration_deficit_mps" and cols["local_crowding"] == "hab_local_enemy_count_200m"
    assert cols["outside_depth"] == cols["inside_cushion"] == "hab_signed_boundary_distance_end_m"
    assert cols["log_alive_teams"] == "hab_alive_teams" and cols["team_members"] == "alive_players_end_norm"
    assert E.RQ1_GROUPS == (("early", (1, 2)), ("mid", (3, 4, 5)), ("late", (6, 7)))
    assert list(E.phase_group_index(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]))) == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert E.TERMS == ("outside_depth", "inside_cushion", "required_speed", "local_crowding", "log_alive_teams", "team_members")  # EHM unchanged


def test_rq1_inputs_are_built_from_the_named_dump_columns(rq1_big):
    _, rows, inputs = rq1_big
    r, inp = rows["train"], inputs["train"]
    col = lambda name: r.X[:, r.feature_names.index(name)].astype(np.float64)  # noqa: E731
    d = col("hab_signed_boundary_distance_end_m")
    assert list(inp.raw.columns) == list(E.RQ1_TERMS) and len(inp) == len(r.y)
    np.testing.assert_array_equal(inp.raw["outside_depth"], np.maximum(0, -d))
    np.testing.assert_array_equal(inp.raw["inside_cushion"], np.maximum(0, d))
    np.testing.assert_array_equal(inp.raw["migration_deficit"], col("hab_migration_deficit_mps"))
    np.testing.assert_array_equal(inp.raw["local_crowding"], col("hab_local_enemy_count_200m"))
    np.testing.assert_allclose(inp.raw["log_alive_teams"], np.log(np.maximum(100.0 * col("hab_alive_teams"), 1.0)))
    np.testing.assert_array_equal(inp.raw["team_members"], col("alive_players_end_norm"))
    assert (inp.raw["migration_deficit"] < 0).any() and (inp.raw["migration_deficit"] > 0).any()  # negative = faster than needed
    # the EHM inputs do not need the new column; the RQ1 inputs name it when it is missing
    names = [n if n != "hab_migration_deficit_mps" else "other" for n in r.feature_names]
    broken = H.Rows(r.keys, r.X, r.y, r.final_rank_norm, r.top5, r.match_type, names, r.removal)
    assert list(E.make_inputs(broken).raw.columns) == list(E.TERMS)
    with pytest.raises(ValueError, match="hab_migration_deficit_mps"):
        E.make_rq1_inputs(broken)


def test_m_main_is_phase_dummies_plus_the_standardised_terms_and_matches_statsmodels(rq1_big):
    _, rows, inputs = rq1_big
    y = rows["train"].y
    model = E.fit_rq1("main", inputs["train"], y)
    assert model.kind == "main" and model.used == list(E.RQ1_TERMS) and model.dropped == {}
    assert model.terms == ["intercept", *[f"phase_{k}" for k in model.phases[1:]], *E.RQ1_TERMS]
    X = E.design_rq1(model, inputs["train"])
    assert X.shape == (len(y), len(model.terms)) and (X[:, 0] == 1).all()
    first = len(model.phases)  # intercept + (n_phases - 1) dummies
    for j, t in enumerate(model.used):
        assert X[:, first + j].mean() == pytest.approx(0.0, abs=1e-9) and X[:, first + j].std() == pytest.approx(1.0, abs=1e-9)
    assert model.fit.converged and model.fit.penalty == "none" and model.fit.cov_robust is not None
    assert model.phases == sorted(set(inputs["train"].phase.tolist())) == list(range(1, int(inputs["train"].phase.max()) + 1)) and len(model.phases) == 7
    for k in model.phases[1:]:  # a dummy for every train phase but the first, which is the reference level
        np.testing.assert_array_equal(X[:, model.terms.index(f"phase_{k}")], (inputs["train"].phase == k).astype(float))
    assert X[inputs["train"].phase == model.phases[0], 1:first].sum() == 0
    sm = pytest.importorskip("statsmodels.api")
    fit = sm.Logit(y, X).fit(disp=0)
    np.testing.assert_allclose(model.fit.beta, fit.params, atol=1e-6)
    np.testing.assert_allclose(model.fit.se_model, fit.bse, rtol=1e-5)
    p = E.predict_rq1(model, inputs["test"])
    assert p.shape == (len(inputs["test"]),) and ((p > 0) & (p < 1)).all()


def test_vif_with_phase_dummies_is_the_brute_force_r_squared_and_never_below_the_terms_only_vif(rq1_big):
    _, rows, inputs = rq1_big
    inp, y = inputs["train"], rows["train"].y
    model = E.fit_rq1("main", inp, y)
    full, terms_only = E.vif_design(model, inp), E.vif(inp, model.used)
    assert list(full) == list(model.used) == list(terms_only)
    X = E.design_rq1(model, inp)  # intercept, dummies, terms
    for j, t in enumerate(model.used):
        col = len(model.phases) + j
        others = np.delete(X, col, axis=1)  # every other column, the intercept included
        resid = X[:, col] - others @ np.linalg.lstsq(others, X[:, col], rcond=None)[0]
        r2 = 1.0 - float(resid @ resid) / float(((X[:, col] - X[:, col].mean()) ** 2).sum())
        assert full[t] == pytest.approx(1.0 / (1.0 - r2), rel=1e-6), t
        assert full[t] >= terms_only[t] - 1e-9  # more regressors can only raise R^2
    # a term that follows the phase is the one the dummies inflate: here the alive teams, which the planted data tie to the phase
    assert full["log_alive_teams"] > 1.5 * terms_only["log_alive_teams"]
    one = E.fit_rq1("main", inp, y)
    one.used, one.terms = ["local_crowding"], ["intercept", "local_crowding"]  # a design with a single column besides the intercept
    one.phases = [1]
    assert E.vif_design(one, inp) == {"local_crowding": 1.0}


def test_m_main_drops_the_constant_team_size_term(tmp_path):
    dump = rq1_dump(80, 6, solo=True)
    r = H.build_rows(dump, "HGB-cut", "v2")
    inp = E.make_rq1_inputs(r)
    for kind in ("main", "phase"):
        model = E.fit_rq1(kind, inp, r.y)
        assert "team_members" not in model.used and model.dropped == {"team_members": "constant in the train rows (zero variance)"}
        assert not any(t.startswith("team_members") for t in model.terms) and model.fit.converged
    assert E.vif(inp, [t for t in E.RQ1_TERMS if t != "team_members"])  # the VIF is over the terms that are in


def test_m_phase_adds_covariate_by_group_interactions_with_early_as_the_reference(rq1_big):
    _, rows, inputs = rq1_big
    y, inp = rows["train"].y, inputs["train"]
    main, phase = E.fit_rq1("main", inp, y), E.fit_rq1("phase", inp, y)
    assert phase.terms == [*main.terms, *[f"{t}:mid" for t in E.RQ1_TERMS], *[f"{t}:late" for t in E.RQ1_TERMS]]
    Xm, Xp = E.design_rq1(main, inp), E.design_rq1(phase, inp)
    np.testing.assert_array_equal(Xp[:, : Xm.shape[1]], Xm)
    g = E.phase_group_index(inp.phase)
    n0 = Xm.shape[1]
    for j, t in enumerate(phase.used):
        z = Xm[:, len(main.phases) + j]
        np.testing.assert_allclose(Xp[:, n0 + j], z * (g == 1))  # t:mid
        np.testing.assert_allclose(Xp[:, n0 + len(phase.used) + j], z * (g == 2))  # t:late
    assert Xp[g == 0, n0:].sum() == 0  # no interaction column is on in the early rows: early is the reference
    assert phase.fit.converged and phase.fit.penalty == "none"


def test_group_slopes_come_from_the_covariance_matrix_and_equal_the_group_specific_parameterisation(rq1_big):
    _, rows, inputs = rq1_big
    y, inp = rows["train"].y, inputs["train"]
    phase = E.fit_rq1("phase", inp, y)
    tables = E.rq1_tables(phase, "c")
    est = tables["estimates"]
    V, Vm = phase.fit.cov_robust, phase.fit.cov_model
    ix = {t: i for i, t in enumerate(phase.terms)}
    for t in phase.used:
        a, b, c = ix[t], ix[f"{t}:mid"], ix[f"{t}:late"]
        beta = phase.fit.beta
        want = {"early": (beta[a], V[a, a], Vm[a, a]), "mid": (beta[a] + beta[b], V[a, a] + V[b, b] + 2 * V[a, b], Vm[a, a] + Vm[b, b] + 2 * Vm[a, b]),
                "late": (beta[a] + beta[c], V[a, a] + V[c, c] + 2 * V[a, c], Vm[a, a] + Vm[c, c] + 2 * Vm[a, c])}  # fmt: skip
        for grp, (b_hat, var_r, var_m) in want.items():
            row = est[(est.covariate == t) & (est.group == grp)].iloc[0]
            assert row.estimate == pytest.approx(b_hat, abs=1e-12) and row.se_robust == pytest.approx(np.sqrt(var_r), rel=1e-10)
            assert row.se_model == pytest.approx(np.sqrt(var_m), rel=1e-10) and row.analysis == "M-phase" and row.cohort == "c"
    con = tables["contrasts"].set_index(["covariate", "contrast"])
    t = "outside_depth"
    a, b, c = ix[t], ix[f"{t}:mid"], ix[f"{t}:late"]
    assert con.loc[(t, "mid-early"), "estimate"] == pytest.approx(phase.fit.beta[b]) and con.loc[(t, "late-early"), "estimate"] == pytest.approx(phase.fit.beta[c])
    assert con.loc[(t, "late-mid"), "estimate"] == pytest.approx(phase.fit.beta[c] - phase.fit.beta[b])
    assert con.loc[(t, "late-mid"), "se_robust"] == pytest.approx(np.sqrt(V[b, b] + V[c, c] - 2 * V[b, c]), rel=1e-10)
    # the same model written with one slope per group: identical predictions, identical group slopes and standard errors
    g = E.phase_group_index(inp.phase)
    Xm = E.design_rq1(E.fit_rq1("main", inp, y), inp)
    base = Xm[:, : len(phase.phases)]
    cols = [Xm[:, len(phase.phases) + j] * (g == k) for j in range(len(phase.used)) for k in range(3)]
    alt = E.fit_logit(np.column_stack([base, *cols]), y, groups=inp.match)
    np.testing.assert_allclose(E._sigmoid(np.column_stack([base, *cols]) @ alt.beta), E.predict_rq1(phase, inp), atol=1e-6)
    for j, t in enumerate(phase.used):
        for k, grp in enumerate(E.GROUP_NAMES):
            row = est[(est.covariate == t) & (est.group == grp)].iloc[0]
            pos = len(phase.phases) + 3 * j + k
            assert row.estimate == pytest.approx(alt.beta[pos], abs=1e-5) and row.se_robust == pytest.approx(alt.se_robust[pos], rel=1e-4)


def test_a_large_late_phase_slope_on_a_column_with_a_tiny_spread_is_not_mistaken_for_separation():
    # the real late-phase exposure terms hardly vary (rondo / taego_fpp: ratio 0.004 - 0.02 of the train spread): the slope per train SD
    # is then beyond the quasi-separation bound although the fit is perfectly regular
    rng = np.random.default_rng(12)
    n = 80000
    phase = rng.integers(1, 8, size=n)
    group = E.phase_group_index(phase)
    spread = np.array([0.05, 0.05, 0.0006])[group]
    raw = pd.DataFrame({t: rng.normal(size=n) for t in E.RQ1_TERMS})
    raw["outside_depth"] = np.abs(rng.normal(size=n)) * spread
    raw["inside_cushion"] = np.abs(rng.normal(size=n)) * 0.05
    unit = np.array([0.0, 0.0, 1500.0])[group]  # per unit; per within-group SD of the late rows: 1500 * 0.00036 = 0.54
    eta = -1.0 + 0.2 * (phase - 4) + unit * raw["outside_depth"].to_numpy()
    y = (rng.random(n) < sigmoid(eta)).astype(float)
    inp = E.Inputs(phase=phase, raw=raw, match=rng.integers(0, 4000, size=n))
    model = E.fit_rq1("phase", inp, y)
    ix = {t: i for i, t in enumerate(model.terms)}
    late = model.fit.beta[ix["outside_depth"]] + model.fit.beta[ix["outside_depth:late"]]
    assert late > E.COEF_LIMIT  # the premise: beyond the unscaled bound ...
    assert model.fit.penalty == "none" and model.fit.converged and model.fit.fallback_reason == ""  # ... and still no ridge fallback
    ratio = model.group_ratio["outside_depth"]["late"]
    assert ratio < 0.05
    row = E.rq1_tables(model, "c", scale="group_sd")["estimates"].query("covariate == 'outside_depth' and group == 'late'").iloc[0]
    want = 1500.0 * float(raw["outside_depth"].to_numpy()[group == 2].std())  # the planted slope per within-group SD
    assert abs(row.estimate - want) < 4.0 * row.se_robust and row.se_robust < 0.1  # ... is recovered, and well determined


def test_group_sd_ratio_is_the_within_group_sd_over_the_train_sd(rq1_scaled):
    _, rows, inputs = rq1_scaled
    inp = inputs["train"]
    phase = E.fit_rq1("phase", inp, rows["train"].y)
    g = E.phase_group_index(inp.phase)
    for t in phase.used:
        x = inp.raw[t].to_numpy()
        for i, grp in enumerate(E.GROUP_NAMES):
            assert phase.group_ratio[t][grp] == pytest.approx(x[g == i].std() / x.std(), rel=1e-12)
    assert phase.group_ratio["outside_depth"]["late"] < 0.5 * phase.group_ratio["outside_depth"]["mid"] < 0.5 * phase.group_ratio["outside_depth"]["early"]
    assert phase.group_ratio["inside_cushion"]["late"] < 0.3  # the late-phase spread of the exposure terms is a fraction of the train spread
    assert E.fit_rq1("main", inp, rows["train"].y).group_ratio == phase.group_ratio  # a property of the train rows, not of the model
    x = np.array([1.0, 2.0, 3.0, 5.0])
    assert E._group_ratio(x, np.array([True, True, False, False]), 2.0) == pytest.approx(0.25)
    assert np.isnan(E._group_ratio(x, np.array([False] * 4), 2.0)) and np.isnan(E._group_ratio(np.full(4, 3.0), np.array([True] * 4), 2.0))  # no group / no spread
    table = E.group_ratio_table(phase, "c")
    assert list(table.columns) == ["cohort", "covariate", "group", "sd_ratio"] and len(table) == len(phase.used) * 3
    row = table[(table.covariate == "inside_cushion") & (table.group == "late")].iloc[0]
    assert row.sd_ratio == phase.group_ratio["inside_cushion"]["late"]


def test_group_sd_slopes_equal_the_train_sd_slopes_times_the_ratio_and_a_fit_on_group_standardised_columns(rq1_scaled):
    _, rows, inputs = rq1_scaled
    y, inp = rows["train"].y, inputs["train"]
    phase = E.fit_rq1("phase", inp, y)
    train_t, gsd_t = E.rq1_tables(phase, "c"), E.rq1_tables(phase, "c", scale="group_sd")
    assert set(train_t["estimates"].scale) == {"train_sd"} and set(gsd_t["estimates"].scale) == {"group_sd"}
    for name in ("estimates", "contrasts", "interactions"):
        assert len(train_t[name]) == len(gsd_t[name]) > 0
    a = train_t["estimates"].set_index(["covariate", "group"])
    b = gsd_t["estimates"].set_index(["covariate", "group"])
    for (t, grp), row in a.iterrows():
        r = phase.group_ratio[t][grp]
        assert b.loc[(t, grp), "estimate"] == pytest.approx(row.estimate * r, rel=1e-12)
        assert b.loc[(t, grp), "se_robust"] == pytest.approx(row.se_robust * r, rel=1e-12) and b.loc[(t, grp), "se_model"] == pytest.approx(row.se_model * r, rel=1e-12)
    # the same slopes from a model written with one column per group, each standardised within its group
    g = E.phase_group_index(inp.phase)
    Xm = E.design_rq1(E.fit_rq1("main", inp, y), inp)
    n_base = len(phase.phases)
    cols = [Xm[:, n_base + j] * (g == k) / phase.group_ratio[t][grp] for j, t in enumerate(phase.used) for k, grp in enumerate(E.GROUP_NAMES)]
    Xa = np.column_stack([Xm[:, :n_base], *cols])
    alt = E.fit_logit(Xa, y, groups=inp.match)
    np.testing.assert_allclose(E._sigmoid(Xa @ alt.beta), E.predict_rq1(phase, inp), atol=1e-6)  # the same model
    for j, t in enumerate(phase.used):
        for k, grp in enumerate(E.GROUP_NAMES):
            pos = n_base + 3 * j + k
            assert b.loc[(t, grp), "estimate"] == pytest.approx(alt.beta[pos], abs=1e-5) and b.loc[(t, grp), "se_robust"] == pytest.approx(alt.se_robust[pos], rel=1e-4)
    # contrasts and the Wald test of equal group-SD slopes from the same alternative parameterisation
    con = gsd_t["contrasts"].set_index(["covariate", "contrast"])
    inter = gsd_t["interactions"].set_index("covariate")
    for j, t in enumerate(phase.used):
        pos = {grp: n_base + 3 * j + k for k, grp in enumerate(E.GROUP_NAMES)}
        for name, hi, lo in E.CONTRASTS:
            L = np.zeros(len(alt.beta))
            L[pos[hi]], L[pos[lo]] = 1.0, -1.0
            assert con.loc[(t, name), "estimate"] == pytest.approx(L @ alt.beta, abs=1e-5)
            assert con.loc[(t, name), "se_robust"] == pytest.approx(float(np.sqrt(L @ alt.cov_robust @ L)), rel=1e-4)
        L = np.zeros((2, len(alt.beta)))
        L[0, pos["mid"]], L[0, pos["early"]], L[1, pos["late"]], L[1, pos["early"]] = 1.0, -1.0, 1.0, -1.0
        d, S = L @ alt.beta, L @ alt.cov_robust @ L.T
        assert inter.loc[t, "wald"] == pytest.approx(float(d @ np.linalg.solve(S, d)), rel=1e-3)
    with pytest.raises(ValueError, match="group_sd"):
        E.rq1_tables(E.fit_rq1("main", inp, y), "c", scale="group_sd")  # M-main has one slope: no group scale
    with pytest.raises(ValueError, match="scale"):
        E.rq1_tables(phase, "c", scale="bogus")


def test_the_two_scales_answer_different_equality_questions(rq1_scaled):
    # planted: the same effect per within-group SD (0.5) of the outside depth in every group, i.e. very different slopes per unit
    _, rows, inputs = rq1_scaled
    phase = E.fit_rq1("phase", inputs["train"], rows["train"].y)
    train_t, gsd_t = E.rq1_tables(phase, "c"), E.rq1_tables(phase, "c", scale="group_sd")
    per_sd = gsd_t["estimates"]
    od = per_sd[per_sd.covariate == "outside_depth"].set_index("group")
    for grp in E.GROUP_NAMES:
        assert abs(od.loc[grp, "estimate"] - 0.5) < 4.0 * od.loc[grp, "se_robust"], (grp, od.loc[grp, "estimate"], od.loc[grp, "se_robust"])
    unit = train_t["estimates"]
    od_unit = unit[unit.covariate == "outside_depth"].set_index("group")
    assert od_unit.loc["late", "estimate"] > 2.0 * od_unit.loc["early", "estimate"]  # the train-SD slope grows as the spread shrinks
    p_train = train_t["interactions"].set_index("covariate").loc["outside_depth", "p_wald"]
    p_group = gsd_t["interactions"].set_index("covariate").loc["outside_depth", "p_wald"]
    assert p_train < 1e-6 < 0.01 < p_group  # unequal per unit (and per train SD), equal per within-group SD


def test_the_per_cohort_wald_test_of_equal_slopes_uses_the_two_interaction_coefficients(rq1_big):
    from scipy import stats

    _, rows, inputs = rq1_big
    phase = E.fit_rq1("phase", inputs["train"], rows["train"].y)
    inter = E.rq1_tables(phase, "c")["interactions"].set_index("covariate")
    V, ix = phase.fit.cov_robust, {t: i for i, t in enumerate(phase.terms)}
    for t in phase.used:
        b, c = ix[f"{t}:mid"], ix[f"{t}:late"]
        d = np.array([phase.fit.beta[b], phase.fit.beta[c]])
        S = V[np.ix_([b, c], [b, c])]
        r = inter.loc[t]
        assert (r.d_mid, r.d_late) == pytest.approx(tuple(d)) and (r.s11, r.s12, r.s22) == pytest.approx((S[0, 0], S[0, 1], S[1, 1]))
        assert r.wald == pytest.approx(float(d @ np.linalg.solve(S, d)), rel=1e-9) and r.p_wald == pytest.approx(float(stats.chi2.sf(r.wald, 2)), rel=1e-9)


def test_planted_group_slopes_are_recovered_and_equality_is_rejected_only_where_it_was_planted(rq1_big):
    _, rows, inputs = rq1_big
    phase = E.fit_rq1("phase", inputs["train"], rows["train"].y)
    tables = E.rq1_tables(phase, "c")
    est = tables["estimates"]
    for term, planted in RQ1_SLOPES.items():
        for k, grp in enumerate(E.GROUP_NAMES):
            row = est[(est.covariate == term) & (est.group == grp)].iloc[0]
            want = planted[k] * phase.sd[term]  # per SD of the term
            assert abs(row.estimate - want) < 4.0 * row.se_robust, (term, grp, row.estimate, want, row.se_robust)
    inter = tables["interactions"].set_index("covariate")
    assert inter.loc["outside_depth", "p_wald"] < 1e-6  # slopes 0 / 0.4 / 0.8 per SD
    for term in ("inside_cushion", "migration_deficit", "local_crowding", "team_members"):
        assert inter.loc[term, "p_wald"] > 1e-3, term  # planted equal across the groups
    assert est[(est.covariate == "outside_depth") & (est.group == "late")].estimate.iloc[0] > est[(est.covariate == "outside_depth") & (est.group == "early")].estimate.iloc[0] + 0.4


def test_m_main_group_free_slopes_lie_between_the_group_slopes_of_m_phase(rq1_big):
    _, rows, inputs = rq1_big
    y, inp = rows["train"].y, inputs["train"]
    main = E.rq1_tables(E.fit_rq1("main", inp, y), "c")["estimates"]
    ph = E.rq1_tables(E.fit_rq1("phase", inp, y), "c")["estimates"]
    row = main[(main.covariate == "outside_depth")].iloc[0]
    assert row.group == "all" and row.analysis == "M-main"
    groups = ph[ph.covariate == "outside_depth"].set_index("group").estimate
    assert groups.min() - 0.05 <= row.estimate <= groups.max() + 0.05  # an average of the group slopes (weights = the groups' information)


# --------------------------------------------------------------------------- the pooled test of equal slopes (multivariate random effects)
def test_mv_random_effects_closed_forms():
    theta = np.array([0.30, -0.10])
    S = np.array([[0.04, 0.01], [0.01, 0.09]])
    out = E.mv_random_effects(np.tile(theta, (6, 1)), np.tile(S, (6, 1, 1)))  # six identical cohorts, no dispersion
    np.testing.assert_allclose(out["T"], 0.0, atol=1e-12)
    np.testing.assert_allclose(out["theta"], theta, atol=1e-12)
    np.testing.assert_allclose(out["cov"], S / 6, atol=1e-12)
    assert out["wald"] == pytest.approx(float(6 * theta @ np.linalg.solve(S, theta)), rel=1e-9) and out["df"] == 2 and out["k"] == 6
    # no dispersion, different precisions: the inverse-variance (fixed effect) estimate
    rng = np.random.default_rng(0)
    Ss = np.array([np.diag(rng.uniform(0.01, 0.05, 2)) for _ in range(5)])
    d = np.tile(theta, (5, 1)) + 0.0
    fe = np.linalg.inv(sum(np.linalg.inv(x) for x in Ss)) @ sum(np.linalg.inv(x) @ y for x, y in zip(Ss, d))
    np.testing.assert_allclose(E.mv_random_effects(d, Ss)["theta"], fe, atol=1e-12)
    # the between-cohort covariance is recovered with many cohorts and is positive semi-definite
    k, T = 4000, np.array([[0.05, 0.03], [0.03, 0.08]])
    Sk = np.tile(np.array([[0.01, 0.0], [0.0, 0.02]]), (k, 1, 1))
    dk = rng.multivariate_normal([0.1, 0.2], T + Sk[0], size=k)
    big = E.mv_random_effects(dk, Sk)
    np.testing.assert_allclose(big["T"], T, atol=0.006)
    assert np.all(np.linalg.eigvalsh(big["T"]) >= -1e-12)
    np.testing.assert_allclose(big["theta"], [0.1, 0.2], atol=0.02)


def test_mv_random_effects_p_values_are_the_stated_references_and_coincide_when_the_sampling_error_vanishes():
    from scipy import stats

    rng = np.random.default_rng(4)
    k, p = 9, 2
    d = rng.multivariate_normal([0.2, -0.1], [[0.05, 0.02], [0.02, 0.04]], size=k)
    out = E.mv_random_effects(d, np.tile(np.diag([0.001, 0.002]), (k, 1, 1)))
    assert out["p_chi2"] == pytest.approx(stats.chi2.sf(out["wald"], p), rel=1e-9)
    assert out["p_f"] == pytest.approx(stats.f.sf(out["wald"] * (k - p) / (p * (k - 1)), p, k - p), rel=1e-9)
    dbar, cov = d.mean(axis=0), np.cov(d, rowvar=False)  # Hotelling's T^2 of the unweighted vectors, from first principles
    t2 = float(k * dbar @ np.linalg.solve(cov, dbar))
    assert out["hotelling_t2"] == pytest.approx(t2, rel=1e-9)
    assert out["p_hotelling"] == pytest.approx(stats.f.sf(t2 * (k - p) / (p * (k - 1)), p, k - p), rel=1e-9)
    # without sampling error the random-effects Wald statistic is Hotelling's T^2 (equal weights, T = the sample covariance): the F references agree
    zero = E.mv_random_effects(d, np.zeros((k, p, p)))
    assert zero["wald"] == pytest.approx(t2, rel=1e-9) and zero["p_f"] == pytest.approx(zero["p_hotelling"], rel=1e-9)
    np.testing.assert_allclose(zero["theta"], dbar, atol=1e-12)


def _mv_rejection_rates(k, T, sd_range, theta_alt, reps, seed):
    """Rejection rates at the 5% level of the three references under theta = 0 and theta = theta_alt."""
    rng = np.random.default_rng(seed)
    rates = {b: {r: 0 for r in ("p_chi2", "p_f", "p_hotelling")} for b in ("null", "alt")}
    for _ in range(reps):
        sd = rng.uniform(*sd_range, size=(k, 2))
        S = np.array([np.diag(s**2) + 0.2 * np.outer(s, s) * (1 - np.eye(2)) for s in sd])
        for theta, bucket in ((np.zeros(2), "null"), (theta_alt, "alt")):
            d = np.array([rng.multivariate_normal(theta, T + S[c]) for c in range(k)])
            out = E.mv_random_effects(d, S)
            for r in rates[bucket]:
                rates[bucket][r] += out[r] < 0.05
    return {b: {r: v / reps for r, v in rr.items()} for b, rr in rates.items()}


def test_mv_random_effects_headline_reference_keeps_its_size_where_the_chi_square_one_over_rejects():
    # large between-cohort variation relative to the sampling error (the case of the real cohorts): the Wald statistic is Hotelling's T^2
    big = np.array([[0.02, 0.012], [0.012, 0.03]])
    rates = _mv_rejection_rates(12, big, (0.03, 0.12), np.array([0.12, 0.2]), reps=500, seed=11)
    assert rates["null"]["p_f"] < 0.085 and rates["null"]["p_hotelling"] < 0.085  # the F reference and the Hotelling check keep the size
    assert rates["null"]["p_chi2"] > 0.09  # ... the chi-square reference does not (it is why p_f is the headline)
    assert rates["alt"]["p_f"] > 0.7 and rates["alt"]["p_hotelling"] > 0.7  # and both have power against a real difference
    # no between-cohort variation: the F reference is conservative, the chi-square one is nominal
    rates0 = _mv_rejection_rates(12, np.zeros((2, 2)), (0.03, 0.12), np.array([0.08, 0.12]), reps=500, seed=12)
    assert rates0["null"]["p_f"] < 0.04 and rates0["null"]["p_chi2"] < 0.075 and rates0["alt"]["p_f"] > 0.9


def test_mv_random_effects_edge_cases():
    S = np.tile(np.eye(2) * 0.01, (2, 1, 1))
    out = E.mv_random_effects(np.array([[0.1, 0.2], [0.12, 0.18]]), S)
    assert out["k"] == 2 and np.isnan(out["p_hotelling"]) and np.isnan(out["p_f"]) and np.isfinite(out["p_chi2"])  # the F references need k > p
    one = E.mv_random_effects(np.array([[0.1, 0.2]]), S[:1])
    assert one["k"] == 1 and np.isnan(one["wald"]) and np.isnan(one["p_chi2"]) and np.isnan(one["p_f"])
    nan = E.mv_random_effects(np.array([[0.1, 0.2], [np.nan, 0.1], [0.1, 0.3], [0.2, 0.1]]), np.tile(np.eye(2) * 0.01, (4, 1, 1)))
    assert nan["k"] == 3  # a cohort without a finite estimate is dropped


# --------------------------------------------------------------------------- pooling the cohorts
TOY_RATIO = {"early": 1.0, "mid": 0.5, "late": 0.1}  # within-group SD / train SD of the toy covariates


def toy_estimates():
    """Per-cohort estimates, contrasts and interactions of two covariates on both scales (the group-SD ones are the train-SD ones times the ratio)."""
    rng = np.random.default_rng(5)
    rows, contrasts, inter = [], [], []
    ratio = np.array([TOY_RATIO[g] for g in E.GROUP_NAMES])
    p_cohort = {"a": 0.01, "b": 0.2, "c": 0.03, "d": 0.5, "e": 0.049}  # three of the five cohorts reject at 5%
    for cohort in ["a", "b", "c", "d", "e"]:
        for t, centre in (("outside_depth", (0.1, 0.3, 0.5)), ("migration_deficit", (-0.1, -0.1, -0.1))):
            slopes = np.array(centre) + rng.normal(0, 0.05, 3)
            se = rng.uniform(0.03, 0.06, 3)
            rows.append({"cohort": cohort, "analysis": "M-main", "covariate": t, "group": "all", "scale": "train_sd", "estimate": float(slopes.mean()),
                         "se_robust": 0.03, "se_model": 0.02})  # fmt: skip
            for scale, r in (("train_sd", np.ones(3)), ("group_sd", ratio)):
                sl, sd = slopes * r, se * r
                for g, bb, ss in zip(E.GROUP_NAMES, sl, sd):
                    rows.append({"cohort": cohort, "analysis": "M-phase", "covariate": t, "group": g, "scale": scale, "estimate": float(bb),
                                 "se_robust": float(ss), "se_model": float(ss) / 2})  # fmt: skip
                d = sl[1:] - sl[0]
                S = np.array([[sd[1] ** 2 + sd[0] ** 2, sd[0] ** 2], [sd[0] ** 2, sd[2] ** 2 + sd[0] ** 2]])
                inter.append({"cohort": cohort, "covariate": t, "scale": scale, "d_mid": d[0], "d_late": d[1], "s11": S[0, 0], "s12": S[0, 1],
                              "s22": S[1, 1], "wald": 1.0, "p_wald": p_cohort[cohort]})  # fmt: skip
                contrasts += [{"cohort": cohort, "covariate": t, "scale": scale, "contrast": "mid-early", "estimate": d[0], "se_robust": np.sqrt(S[0, 0]), "se_model": 0.01},
                              {"cohort": cohort, "covariate": t, "scale": scale, "contrast": "late-early", "estimate": d[1], "se_robust": np.sqrt(S[1, 1]), "se_model": 0.01},
                              {"cohort": cohort, "covariate": t, "scale": scale, "contrast": "late-mid", "estimate": d[1] - d[0],
                               "se_robust": np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1]), "se_model": 0.01}]  # fmt: skip
    return pd.DataFrame(rows), pd.DataFrame(contrasts), pd.DataFrame(inter)


def test_pooling_helpers_use_dl_per_covariate_and_group_and_the_multivariate_test_per_covariate():
    est, con, inter = toy_estimates()
    main = E.rq1_pool_main(est)
    groups = E.rq1_pool_groups(est)
    eq = E.rq1_equality(inter, con)
    covs = ("outside_depth", "migration_deficit")
    assert list(main.covariate) == list(covs) and set(main.group) == {"all"} and set(main.analysis) == {"M-main"} and set(main.scale) == {"train_sd"}
    g = est[(est.analysis == "M-main") & (est.covariate == "outside_depth")]
    ref = E.pooled_dl(g.estimate, g.se_robust)
    r = main.iloc[0]
    assert r.theta == pytest.approx(ref["theta"]) and (r.ci95_low, r.ci95_high) == pytest.approx((ref["ci95_low"], ref["ci95_high"]))
    assert r.tau2 == pytest.approx(ref["tau2"]) and r.I2 == pytest.approx(ref["I2"]) and r.k == 5
    assert r.same_sign_as_pooled == ref["same_sign_as_pooled"] and r.sign_agreement == f"{ref['same_sign_as_pooled']}/5"
    assert list(zip(groups.scale, groups.covariate, groups.group)) == [(sc, t, g) for sc in E.SCALES for t in covs for g in E.GROUP_NAMES]
    for scale in E.SCALES:  # DL per scale, covariate and group
        late = est[(est.analysis == "M-phase") & (est.covariate == "outside_depth") & (est.group == "late") & (est.scale == scale)]
        ref = M.dersimonian_laird(late.estimate.to_numpy(), late.se_robust.to_numpy())
        r = groups[(groups.scale == scale) & (groups.covariate == "outside_depth") & (groups.group == "late")].iloc[0]
        assert r.theta == pytest.approx(ref["theta"]) and r.se == pytest.approx(ref["se"]) and set(groups.analysis) == {"M-phase"}
    late_train = groups[(groups.scale == "train_sd") & (groups.covariate == "outside_depth") & (groups.group == "late")].iloc[0]
    late_group = groups[(groups.scale == "group_sd") & (groups.covariate == "outside_depth") & (groups.group == "late")].iloc[0]
    assert late_group.theta == pytest.approx(late_train.theta * TOY_RATIO["late"]) and late_group.se == pytest.approx(late_train.se * TOY_RATIO["late"])  # a common factor
    assert late_group.sign_agreement == late_train.sign_agreement and late_group.k == late_train.k == 5
    # the equality test: the multivariate random-effects Wald of the two contrasts, per scale and covariate, plus the univariate pooled differences
    assert list(zip(eq.scale, eq.covariate)) == [(sc, t) for sc in E.SCALES for t in covs]
    eqi = eq.set_index(["scale", "covariate"])
    e = eqi.loc[("train_sd", "outside_depth")]
    x = inter[(inter.covariate == "outside_depth") & (inter.scale == "train_sd")]
    mv = E.mv_random_effects(x[["d_mid", "d_late"]].to_numpy(), np.array([[[a, b], [b, c]] for a, b, c in zip(x.s11, x.s12, x.s22)]))
    assert e.wald == pytest.approx(mv["wald"]) and e.p_equal == pytest.approx(mv["p_f"]) and e.p_chi2 == pytest.approx(mv["p_chi2"]) and e.k == 5
    assert e.p_hotelling == pytest.approx(mv["p_hotelling"]) and e.p_chi2 < e.p_equal  # the F reference is the more conservative of the two
    cm = con[(con.covariate == "outside_depth") & (con.contrast == "late-early") & (con.scale == "train_sd")]
    ref = M.dersimonian_laird(cm.estimate.to_numpy(), cm.se_robust.to_numpy())
    assert e.diff_late_early == pytest.approx(ref["theta"]) and e.diff_late_early_low == pytest.approx(ref["ci95"][0])
    assert e.p_equal < 0.01 and eqi.loc[("train_sd", "outside_depth"), "p_equal"] < eqi.loc[("train_sd", "migration_deficit"), "p_equal"]  # the planted gap is the one detected
    assert eqi.loc[("train_sd", "outside_depth"), "diff_late_early"] > 0.2 and abs(eqi.loc[("train_sd", "migration_deficit"), "diff_late_early"]) < 0.1
    eg = eqi.loc[("group_sd", "outside_depth")]  # on the group-SD scale the planted slopes (0.1, 0.15, 0.05) are close: a smaller difference
    assert abs(eg.diff_late_early) < abs(e.diff_late_early) and eg.p_equal != e.p_equal
    assert (eq.n_cohorts_p05 == 3).all()  # the cohorts whose own Wald test rejects at 5%: 0.01, 0.03, 0.049
    # the sign agreement is the number of cohorts with the sign of the pooled estimate, for every pooled row (negative pooled slopes included)
    assert (main.theta < 0).any() and (groups.theta < 0).any() and (groups.theta > 0).any()
    for table in (main, groups):
        for r in table.itertuples():
            cell = est[(est.analysis == r.analysis) & (est.covariate == r.covariate) & (est.group == r.group) & (est.scale == r.scale)]
            same = int((np.sign(cell.estimate) == np.sign(r.theta)).sum())
            assert r.same_sign_as_pooled == same and r.sign_agreement == f"{same}/{len(cell)}", (r.covariate, r.group, r.scale)
            assert r.n_positive + r.n_negative == len(cell) and same == (r.n_positive if r.theta > 0 else r.n_negative)
    tidy = E.rq1_tidy(main, groups, eq)
    assert list(tidy.columns[:6]) == ["analysis", "covariate", "construct", "group", "scale", "k"] and len(tidy) == 2 * 7
    assert [(r.analysis, r.group, r.scale) for r in tidy.iloc[:7].itertuples()] == [("M-main", "all", "train_sd")] + [("M-phase", g, sc) for sc in E.SCALES for g in E.GROUP_NAMES]
    assert list(tidy.covariate[:7].unique()) == ["outside_depth"]
    for col in ("equality_wald", "equality_p", "equality_p_chi2", "equality_p_hotelling"):
        assert tidy[tidy.analysis == "M-main"][col].isna().all() and tidy[tidy.analysis == "M-phase"][col].notna().all(), col
    for scale in E.SCALES:
        row = eqi.loc[(scale, "outside_depth")]
        od = tidy[(tidy.analysis == "M-phase") & (tidy.covariate == "outside_depth") & (tidy.scale == scale)]
        assert (od.equality_p == row.p_equal).all() and (od.equality_p_chi2 == row.p_chi2).all() and (od.equality_p_hotelling == row.p_hotelling).all()
    assert set(tidy.construct) == {"exposure", "movement burden"}
    # estimates without a scale column (one scale) are read as train-SD
    old = est.drop(columns="scale")[lambda f: ~((f.analysis == "M-phase") & f.duplicated(["cohort", "covariate", "group"]))]
    assert set(E.rq1_pool_groups(old).scale) == {"train_sd"} and len(E.rq1_pool_groups(old)) == 6


# --------------------------------------------------------------------------- end to end
def test_rq1_end_to_end_on_synthetic_cohorts_writes_the_tables_and_the_forest(cohort_runs, tmp_path):
    dumps, runs = cohort_runs
    out = tmp_path / "rq1"
    summary = E.rq1_run(dumps, ["c_squad", "c_duo", "c_solo"], runs, out, n_boot=40)
    files = {p.name for p in out.iterdir()}
    for name in ("rq1_tidy.csv", "rq1_cohort_estimates.csv", "rq1_cohort_contrasts.csv", "rq1_interactions_per_cohort.csv", "rq1_slope_equality.csv",
                 "rq1_group_sd_ratio.csv", "rq1_vif.csv", "rq1_vif_max.csv", "rq1_auc.csv", "rq1_auc_difference.csv", "rq1_fit_info.csv", "rq1_forest_data.csv",
                 "rq1_forest.pdf", "rq1_forest.png", "rq1_forest_train_sd.pdf", "rq1_forest_train_sd.png", "rq1_summary.json"):  # fmt: skip
        assert name in files, name
    tidy = pd.read_csv(out / "rq1_tidy.csv")
    assert set(tidy.analysis) == {"M-main", "M-phase"} and set(tidy.covariate) == set(E.RQ1_TERMS) and len(tidy) == 6 * 7  # M-main + 3 groups x 2 scales
    assert set(tidy.scale) == {"train_sd", "group_sd"} and set(tidy[tidy.analysis == "M-main"].scale) == {"train_sd"}
    tm = tidy[(tidy.covariate == "team_members") & (tidy.analysis == "M-main")].iloc[0]
    assert tm.k == 2 and tidy[(tidy.covariate == "outside_depth") & (tidy.analysis == "M-main")].iloc[0].k == 3  # the solo cohort has no team-size term
    est = pd.read_csv(out / "rq1_cohort_estimates.csv")
    assert set(est.cohort) == {"c_squad", "c_duo", "c_solo"} and not ((est.cohort == "c_solo") & (est.covariate == "team_members")).any()
    assert set(est.group) == {"all", "early", "mid", "late"} and (est.se_robust > 0).all() and set(est.scale) == {"train_sd", "group_sd"}
    ratio = pd.read_csv(out / "rq1_group_sd_ratio.csv")
    assert set(ratio.cohort) == {"c_squad", "c_duo", "c_solo"} and (ratio.sd_ratio > 0).all() and not ((ratio.cohort == "c_solo") & (ratio.covariate == "team_members")).any()
    a = est[(est.cohort == "c_duo") & (est.covariate == "local_crowding") & (est.group == "late") & (est.scale == "train_sd")].iloc[0]
    b = est[(est.cohort == "c_duo") & (est.covariate == "local_crowding") & (est.group == "late") & (est.scale == "group_sd")].iloc[0]
    r = ratio[(ratio.cohort == "c_duo") & (ratio.covariate == "local_crowding") & (ratio.group == "late")].sd_ratio.iloc[0]
    assert b.estimate == pytest.approx(a.estimate * r, rel=1e-9) and b.se_robust == pytest.approx(a.se_robust * r, rel=1e-9)
    vif = pd.read_csv(out / "rq1_vif.csv")
    vmax = pd.read_csv(out / "rq1_vif_max.csv").set_index("cohort")
    assert set(vif.cohort) == {"c_squad", "c_duo", "c_solo"} and (vif.vif >= 1.0).all()
    for cohort in vmax.index:
        assert vmax.loc[cohort, "max_vif"] == pytest.approx(vif[vif.cohort == cohort].vif.max()) and vmax.loc[cohort, "term"] in E.RQ1_TERMS
        assert vmax.loc[cohort, "max_vif_with_phase"] == pytest.approx(vif[vif.cohort == cohort].vif_with_phase.max()) and vmax.loc[cohort, "term_with_phase"] in E.RQ1_TERMS
    assert (vif.vif_with_phase >= vif.vif - 1e-9).all() and list(vif.columns) == ["cohort", "term", "vif", "vif_with_phase"]
    auc = pd.read_csv(out / "rq1_auc.csv")
    assert set(auc.arm) == {"M-main", "M-phase", "EHM-lite", "EHM"} and len(auc) == 3 * 4
    row = auc[(auc.cohort == "c_squad") & (auc.arm == "EHM-lite")].iloc[0]
    ref = M.read_predictions(runs / "c_squad" / "EHM-lite" / "seed42" / "pred_test.csv")
    assert row.auc == pytest.approx(M.hazard_metrics(ref["y"].to_numpy(), ref["p_hazard"].to_numpy())["auc"], abs=1e-12)
    diff = pd.read_csv(out / "rq1_auc_difference.csv")
    assert set(diff.cohort) == {"c_squad", "c_duo", "c_solo", "pooled"} and set(diff.a) == {"M-main"} and set(diff.b) == {"EHM-lite"}
    a = auc[(auc.cohort == "c_squad") & (auc.arm == "M-main")].auc.iloc[0]
    assert diff[(diff.cohort == "c_squad") & (diff.metric == "auc")].difference.iloc[0] == pytest.approx(a - row.auc, abs=1e-9)
    splits = D.load_cohort(dumps, "c_squad", labels="v2", match_types=None)
    rows = {k: H.build_rows(splits[k], "HGB-cut", "v2") for k in ("train", "test")}
    inp = {k: E.make_rq1_inputs(r) for k, r in rows.items()}
    main_model = E.fit_rq1("main", inp["train"], rows["train"].y)
    for term, value in E.vif(inp["train"], main_model.used).items():  # the VIF of the train rows
        assert vif[(vif.cohort == "c_squad") & (vif.term == term)].vif.iloc[0] == pytest.approx(value, rel=1e-9)
    for term, value in E.vif_design(main_model, inp["train"]).items():
        assert vif[(vif.cohort == "c_squad") & (vif.term == term)].vif_with_phase.iloc[0] == pytest.approx(value, rel=1e-9)
    frame = H._prediction_frame(rows["test"], E.predict_rq1(main_model, inp["test"]))  # the test AUC of M-main, recomputed from scratch
    assert a == pytest.approx(M.hazard_metrics(frame["y"].to_numpy(), frame["p_hazard"].to_numpy())["auc"], abs=1e-12)
    fi = pd.read_csv(out / "rq1_fit_info.csv")
    assert len(fi) == 6 and fi.penalty.eq("none").all() and fi.converged.all()
    assert summary["cohorts"] == ["c_squad", "c_duo", "c_solo"] and set(summary["max_vif_per_cohort"]) == {"c_squad", "c_duo", "c_solo"}
    assert set(summary["auc_main_minus_ehm_lite"]) == {"auc", "auc_within_phase"} and len(summary["slope_equality"]) == 6 * 2 and summary["scales"] == ["train_sd", "group_sd"]
    forest = pd.read_csv(out / "rq1_forest_data.csv")
    assert len(forest) == 2 * 6 * 4 and set(forest.figure_scale) == {"train_sd", "group_sd"}
    for stem in ("rq1_forest", "rq1_forest_train_sd"):
        assert (out / f"{stem}.pdf").read_bytes()[:4] == b"%PDF" and (out / f"{stem}.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_rq1_forest_draws_every_pooled_estimate_with_its_interval(tmp_path):
    est, con, inter = toy_estimates()
    tidy = E.rq1_tidy(E.rq1_pool_main(est), E.rq1_pool_groups(est), E.rq1_equality(inter, con))
    data = E.rq1_forest_data(tidy)
    assert list(data.columns[:5]) == ["figure_scale", "covariate", "construct", "group", "y"] and len(data) == 2 * 4 and set(data.figure_scale) == {"group_sd"}
    assert list(data.group[:4]) == ["all", "early", "mid", "late"] and data.y.is_monotonic_increasing
    assert list(data.y) == [0, 1, 2, 3, 5, 6, 7, 8]  # one empty row between the covariates, where the header sits
    # the row of each group is the pooled slope of the requested scale; the all-phases row is the M-main coefficient on both
    train = E.rq1_forest_data(tidy, "train_sd")
    assert list(train.group) == list(data.group) and set(train.figure_scale) == {"train_sd"}
    first, first_t = data[data.covariate == "outside_depth"].set_index("group"), train[train.covariate == "outside_depth"].set_index("group")
    assert first.loc["all", "theta"] == first_t.loc["all", "theta"]
    assert first.loc["late", "theta"] == pytest.approx(first_t.loc["late", "theta"] * TOY_RATIO["late"])
    for scale, frame in (("group_sd", data), ("train_sd", train)):
        fig = E.build_rq1_forest(frame, scale)
        ax = fig.axes[0]
        segments = sorted((round(float(s[0][0]), 9), round(float(s[1][0]), 9), round(float(s[0][1]), 6)) for c in ax.collections for s in c.get_segments())
        want = sorted((round(float(r.ci95_low), 9), round(float(r.ci95_high), 9), round(float(r.y), 6)) for r in frame.itertuples())
        assert segments == want  # one interval per pooled estimate at its row
        points = sorted((round(float(line.get_xdata()[0]), 9), round(float(line.get_ydata()[0]), 6)) for line in ax.get_lines() if len(line.get_xdata()) == 1)
        assert points == sorted((round(float(r.theta), 9), round(float(r.y), 6)) for r in frame.itertuples())  # one marker per pooled estimate, at its row
        lo, hi = ax.get_xlim()
        assert lo <= frame.ci95_low.min() and hi >= frame.ci95_high.max()
        labels = [t.get_text() for t in ax.texts]
        assert all(a in labels for a in frame.sign_agreement) and "n/k" in labels  # the sign agreement at every row
        assert ax.get_xlabel().startswith("Log-odds per " + ("SD" if scale == "group_sd" else "train SD"))
        bold = [t for t in ax.texts if t.get_fontweight() == "bold"]
        assert [t.get_text() for t in bold] == [E.RQ1_COVARIATE_LABELS[t] for t in dict.fromkeys(frame.covariate)]  # a header per covariate
        assert [round(t.get_position()[1], 6) for t in bold] == [-0.95, 4.05]  # in the empty row above its block of rows
        assert all(E.RQ1_CONSTRUCT[c] in labels for c in frame.covariate)  # and the construct next to it
        fig.canvas.draw()
        assert all(t.get_window_extent().x0 >= 0 for t in bold)  # not clipped by the left edge of the figure
    E.draw_rq1_forest(data, tmp_path / "f")
    assert (tmp_path / "f.pdf").read_bytes()[:4] == b"%PDF" and (tmp_path / "f.png").read_bytes()[:4] == b"\x89PNG"


def test_a_ridge_fallback_is_recorded_in_the_fit_information_and_the_summary(cohort_runs, tmp_path, monkeypatch):
    dumps, runs = cohort_runs
    monkeypatch.setattr(E, "COEF_LIMIT", 0.2)  # every fit now leaves the coefficient bound: the tiny-L2 fallback
    res = E.rq1_cohort(dumps, "c_duo", runs)
    assert {r["model"] for r in res["fit_info"]} == {"M-main", "M-phase"}
    for r in res["fit_info"]:
        assert r["penalty"] == "l2" and r["l2_lambda"] == E.L2_LAMBDA and "left the coefficient bound" in r["fallback_reason"]
    out = tmp_path / "rq1"
    summary = E.rq1_run(dumps, ["c_duo", "c_solo"], runs, out, n_boot=0)
    assert summary["penalty_fallbacks"] == [f"{c}/{m}" for c in ("c_duo", "c_solo") for m in ("M-main", "M-phase")]
    fi = pd.read_csv(out / "rq1_fit_info.csv")
    assert fi.penalty.eq("l2").all() and fi.fallback_reason.str.contains("coefficient bound").all()
    assert summary["dropped_terms"] == {"c_solo/M-main": "team_members", "c_solo/M-phase": "team_members"}


def test_the_cli_runs_the_rq1_stage_into_the_rq1_subdirectory(cohort_runs, tmp_path):
    dumps, runs = cohort_runs
    out = tmp_path / "eco"
    assert E.main(["--dump-dir", str(dumps), "--runs-dir", str(runs), "--out-dir", str(out), "--stage", "rq1", "--cohorts", "c_squad,c_duo,c_solo", "--n-boot", "0"]) == 0
    assert (out / "rq1" / "rq1_tidy.csv").is_file() and not (out / "auc_comparison.csv").exists()
    d = pd.read_csv(out / "rq1" / "rq1_auc_difference.csv")
    assert d.ci_low.isna().all()  # --n-boot 0: points only


# =========================================================================== E4: RQ1 in physical units and average marginal effects
def test_the_physical_units_follow_the_dump_scaling_constants():
    assert set(E.RQ1_UNITS) == set(E.RQ1_TERMS)
    assert E.RQ1_UNITS["outside_depth"] == E.RQ1_UNITS["inside_cushion"] == (100.0 / 8160.0, "per 100 m")  # distances are stored / 8160 m
    assert E.RQ1_UNITS["migration_deficit"] == (1.0 / 40.0, "per 1 m/s")  # speeds / 40 m/s
    assert E.RQ1_UNITS["local_crowding"][0] == 1.0 / 20.0 and E.RQ1_UNITS["team_members"][0] == 1.0 / 4.0  # counts / 20, / 4
    assert E.RQ1_UNITS["log_alive_teams"][0] == 1.0  # unchanged
    assert E.rq1_units() == {t: E.RQ1_UNITS[t][0] for t in E.RQ1_TERMS}


def test_check_units_accepts_integer_counts_and_rejects_a_changed_scaling(rq1_big):
    _, rows, inputs = rq1_big
    dev = E.check_units(inputs["train"])
    assert set(dev) == {"local_crowding", "team_members", "log_alive_teams"} and max(dev.values()) < 1e-3
    bad = E.Inputs(inputs["train"].phase, inputs["train"].raw.assign(local_crowding=inputs["train"].raw["local_crowding"] * 1.37), inputs["train"].match)
    with pytest.raises(ValueError, match="scaling constants"):
        E.check_units(bad)


def test_the_physical_fit_is_a_reparametrisation_of_the_per_sd_fit(rq1_big):
    _, rows, inputs = rq1_big
    y, train = rows["train"].y, inputs["train"]
    for kind, name in (("main", "M-main"), ("phase", "M-phase")):
        sd_model = E.fit_rq1(kind, train, y)
        phys = E.fit_rq1(kind, train, y, E.rq1_units())
        assert phys.scale == pytest.approx(E.rq1_units()) and sd_model.scale is None and phys.terms == sd_model.terms
        assert phys.fit.loglik == pytest.approx(sd_model.fit.loglik, abs=1e-8) and phys.fit.penalty == "none"
        np.testing.assert_allclose(E.predict_rq1(phys, inputs["test"]), E.predict_rq1(sd_model, inputs["test"]), atol=1e-9)
        for i, t in enumerate(phys.terms):  # beta_per_SD = beta_physical * SD / unit, coefficient by coefficient (SEs too)
            base = t.split(":")[0]
            k = phys.sd[base] / phys.scale[base] if base in phys.used else 1.0
            assert phys.fit.beta[i] * k == pytest.approx(sd_model.fit.beta[i], rel=1e-6, abs=1e-7), t
            assert phys.fit.se_robust[i] * k == pytest.approx(sd_model.fit.se_robust[i], rel=1e-6, abs=1e-8), t
        rows_eq = E._equivalence_rows("c", name, phys, sd_model)
        assert max(r["rel_diff"] for r in rows_eq) < 1e-5 and {r["quantity"] for r in rows_eq} == {"beta", "se_robust"}
    # a physical column is centred but not scaled to unit SD: its SD is the SD in physical units
    X = E.design_rq1(E.fit_rq1("main", train, y, E.rq1_units()), train)
    j = len(E.fit_rq1("main", train, y).phases)  # first term column (outside depth)
    assert X[:, j].mean() == pytest.approx(0.0, abs=1e-9)
    assert X[:, j].std() == pytest.approx(train.raw["outside_depth"].to_numpy().std() / (100.0 / 8160.0), rel=1e-9)


def test_the_physical_scale_label_belongs_to_physical_models_only(rq1_big):
    _, rows, inputs = rq1_big
    y, train = rows["train"].y, inputs["train"]
    phys, sd_model = E.fit_rq1("phase", train, y, E.rq1_units()), E.fit_rq1("phase", train, y)
    tabs = E.rq1_tables(phys, "c", scale=E.PHYSICAL)
    assert set(tabs["estimates"].scale) == {"physical"} and len(tabs["estimates"]) == 6 * 3
    with pytest.raises(ValueError, match="physical"):
        E.rq1_tables(phys, "c", scale="train_sd")
    with pytest.raises(ValueError, match="physical"):
        E.rq1_tables(sd_model, "c", scale=E.PHYSICAL)
    with pytest.raises(ValueError, match="physical"):
        E.rq1_tables(E.fit_rq1("main", train, y, E.rq1_units()), "c", scale="group_sd")  # no group-SD slopes for a physical model
    with pytest.raises(ValueError, match="group_sd"):
        E.rq1_tables(E.fit_rq1("main", train, y), "c", scale="group_sd")  # M-main has one slope per covariate (unchanged behaviour)


def test_physical_slopes_recover_the_planted_slopes_per_unit(rq1_big):
    _, rows, inputs = rq1_big
    model = E.fit_rq1("phase", inputs["train"], rows["train"].y, E.rq1_units())
    est = E.rq1_tables(model, "c", scale=E.PHYSICAL)["estimates"].set_index(["covariate", "group"])
    for term, planted in RQ1_SLOPES.items():  # planted slopes are per DUMP unit: x the unit width = per physical unit
        for g, slope in zip(("early", "mid", "late"), planted):
            want = slope * E.RQ1_UNITS[term][0]
            r = est.loc[(term, g)]
            assert abs(r.estimate - want) < 4.0 * r.se_robust, (term, g, r.estimate, want)


def finite_difference_gradient(f, beta, h=1e-6):
    g = np.zeros_like(beta)
    for i in range(len(beta)):
        d = np.zeros_like(beta)
        d[i] = h
        g[i] = (f(beta + d) - f(beta - d)) / (2 * h)
    return g


@pytest.mark.parametrize("kind", ["main", "phase"])
def test_ame_equals_brute_force_derivatives_with_a_delta_method_se(rq1_big, kind):
    _, rows, inputs = rq1_big
    train, y = inputs["train"], rows["train"].y
    model = E.fit_rq1(kind, train, y, E.rq1_units())
    ame = E.ame_table(model, train, "c", "M-" + kind).set_index(["covariate", "group"])
    assert len(ame) == 6 * 3 and set(ame.unit) == {u for _, u in E.RQ1_UNITS.values()}
    X = E.design_rq1(model, train)
    group = E.phase_group_index(train.phase)
    ix = {t: i for i, t in enumerate(model.terms)}
    for term in ("outside_depth", "migration_deficit", "team_members"):
        for gi, g in enumerate(E.GROUP_NAMES):
            sel = group == gi
            L = np.zeros(len(model.fit.beta))
            L[ix[term]] = 1.0
            if kind == "phase" and g != "early":
                L[ix[f"{term}:{g}"]] = 1.0

            def mean_derivative(b, sel=sel, L=L):  # d p / d x averaged over the group's rows, brute force from the probabilities
                eps = 1e-5
                lo, hi = E._sigmoid(X[sel] @ b - eps * (L @ b)), E._sigmoid(X[sel] @ b + eps * (L @ b))
                return float(np.mean((hi - lo) / (2 * eps)))

            def mean_discrete(b, sel=sel, L=L):
                return float(np.mean(E._sigmoid(X[sel] @ b + L @ b) - E._sigmoid(X[sel] @ b)))

            r = ame.loc[(term, g)]
            assert r.ame == pytest.approx(mean_derivative(model.fit.beta), rel=1e-6, abs=1e-9)
            assert r.ame_disc == pytest.approx(mean_discrete(model.fit.beta), rel=1e-9, abs=1e-12)
            V = model.fit.cov_robust
            for est, se, f in ((r.ame, r.se, mean_derivative), (r.ame_disc, r.se_disc, mean_discrete)):
                grad = finite_difference_gradient(f, model.fit.beta)
                assert se == pytest.approx(float(np.sqrt(grad @ V @ grad)), rel=2e-4), (term, g)
            assert r.n_rows == int(sel.sum()) and r.mean_p == pytest.approx(E._sigmoid(X[sel] @ model.fit.beta).mean())
            assert r.slope == pytest.approx(float(L @ model.fit.beta)) and r.group_sd > 0


def test_ame_is_invariant_to_the_scaling_of_the_covariates(rq1_big):
    # the AME of the per-SD model, converted to a physical unit (x unit / SD), is the physical-model AME: the same fit, the same rows
    _, rows, inputs = rq1_big
    train, y = inputs["train"], rows["train"].y
    sd_model, phys = E.fit_rq1("phase", train, y), E.fit_rq1("phase", train, y, E.rq1_units())
    a_sd = E.ame_table(sd_model, train, "c", "M-phase").set_index(["covariate", "group"])
    a_ph = E.ame_table(phys, train, "c", "M-phase").set_index(["covariate", "group"])
    for key in a_ph.index:
        k = E.RQ1_UNITS[key[0]][0] / sd_model.sd[key[0]]  # one physical unit in SDs
        assert a_ph.loc[key, "ame"] == pytest.approx(a_sd.loc[key, "ame"] * k, rel=1e-6, abs=1e-12)
        assert a_ph.loc[key, "se"] == pytest.approx(a_sd.loc[key, "se"] * k, rel=1e-5, abs=1e-12)


def test_ame_pooled_is_the_dl_pooling_of_the_per_cohort_ames():
    rng = np.random.default_rng(0)
    rows = []
    for c in range(8):
        for g in ("early", "mid", "late"):
            rows.append({"cohort": f"c{c}", "analysis": "M-phase", "covariate": "local_crowding", "group": g, "unit": "per 1 enemy team within 200 m", "n_rows": 100,
                         "mean_p": 0.3, "group_sd": 1.0, "slope": 0.1, "ame": 0.02 + 0.005 * rng.standard_normal(), "se": 0.004, "ame_disc": 0.021 + 0.005 * rng.standard_normal(),
                         "se_disc": 0.004})  # fmt: skip
    frame = pd.DataFrame(rows)
    pooled = E.ame_pooled(frame).set_index("group")
    for g in ("early", "mid", "late"):
        part = frame[frame.group == g]
        dl = M.dersimonian_laird(part.ame.to_numpy(), part.se.to_numpy())
        r = pooled.loc[g]
        assert r.theta == pytest.approx(dl["theta"]) and r.k == 8 and r.I2 == pytest.approx(dl["I2"]) and r.tau2 == pytest.approx(dl["tau2"])
        assert (r.ci95_low, r.ci95_high) == pytest.approx(dl["ci95"]) and r.theta_pp == pytest.approx(100 * dl["theta"])
        assert r.n_positive + r.n_negative == 8 and r.sign_agreement.endswith("/8")


def test_the_cli_runs_the_physical_units_stage_into_its_own_subdirectory(cohort_runs, tmp_path):
    dumps, runs = cohort_runs
    out = tmp_path / "eco"
    sd = out / "rq1"
    E.rq1_run(dumps, ["c_squad", "c_duo", "c_solo"], runs, sd, n_boot=0)
    assert E.main(["--dump-dir", str(dumps), "--runs-dir", str(runs), "--out-dir", str(out), "--stage", "rq1", "--rq1-units", "physical",
                   "--cohorts", "c_squad,c_duo,c_solo"]) == 0  # fmt: skip
    d = out / "rq1_physical"
    for name in ("rq1_physical_tidy.csv", "rq1_physical_cohort_estimates.csv", "rq1_physical_cohort_contrasts.csv", "rq1_physical_ame_per_cohort.csv",
                 "rq1_physical_ame_pooled.csv", "rq1_physical_units.csv", "rq1_physical_check_vs_sd.csv", "rq1_physical_fit_info.csv", "rq1_physical_summary.json",
                 "rq1_physical_vs_sd_pooled.csv"):  # fmt: skip
        assert (d / name).is_file(), name
    assert (sd / "rq1_tidy.csv").is_file()
    tidy = pd.read_csv(d / "rq1_physical_tidy.csv")
    assert set(tidy.scale) == {"physical"} and set(tidy.analysis) == {"M-main", "M-phase"} and len(tidy) == 6 * 4
    assert {*tidy.unit} == {u for _, u in E.RQ1_UNITS.values()} and np.allclose(tidy.odds_ratio, np.exp(tidy.theta), rtol=1e-12)
    assert tidy[(tidy.covariate == "team_members") & (tidy.analysis == "M-main")].k.iloc[0] == 2  # the solo cohort has no team-size term
    assert list(tidy.analysis[:6]) == ["M-main"] * 6 and list(tidy.covariate[:6]) == list(E.RQ1_TERMS)  # M-main first, covariates in table order
    ame = pd.read_csv(d / "rq1_physical_ame_pooled.csv")
    assert set(ame.group) == {"early", "mid", "late"} and len(ame) == 2 * 6 * 3 and (ame.k >= 2).all()
    summary = json.loads((d / "rq1_physical_summary.json").read_text())
    assert summary["max_rel_diff_vs_per_sd_fit"] < 1e-5 and summary["sign_counts_equal_to_per_sd_tables"] is True
    assert summary["units"]["outside_depth"] == "per 100 m" and summary["dump_scaling"]["distance_m"] == 8160.0
    # the pooled physical coefficient lies between the per-cohort ones and the sign counts match the per-SD analysis
    est = pd.read_csv(d / "rq1_physical_cohort_estimates.csv")
    for t in E.RQ1_TERMS:
        e = est[(est.covariate == t) & (est.analysis == "M-main")].estimate
        th = tidy[(tidy.covariate == t) & (tidy.analysis == "M-main")].theta.iloc[0]
        assert e.min() - 1e-12 <= th <= e.max() + 1e-12
