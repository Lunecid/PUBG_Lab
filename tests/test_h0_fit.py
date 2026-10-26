"""Tests of the grouped-time Cox baseline estimator (third_party/krafton_pubg_winprob/adaptation/h0_fit.py): the cloglog fit with the
score as an offset (and with a shared slope) against an independent optimiser, the closed form without a score, monotonicity and
numerical stability of the probabilities, and the refusal of phases that hold a single class.  numpy / scipy only."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

KT = Path(__file__).resolve().parents[1] / "third_party" / "krafton_pubg_winprob"
if str(KT) not in sys.path:
    sys.path.insert(0, str(KT))

from projects.krafton_baseline.training.h0_fit import cloglog_prob, fit_cloglog_offset, predict_h0  # noqa: E402

GAMMA = np.array([-1.6, -1.2, -0.9, -0.6, -0.3, 0.0, 0.4])


def simulate(seed: int = 0, n: int = 6000, beta: float = 1.0):
    rng = np.random.default_rng(seed)
    phase = rng.integers(1, 8, n)
    s = rng.normal(-0.3, 0.8, n)
    y = (rng.random(n) < cloglog_prob(GAMMA[phase - 1] + beta * s)).astype(float)
    return s, y, phase


def reference_nll(theta, s, y, phase, slope):
    """Independent negative log-likelihood (direct formula) for scipy's BFGS."""
    g = theta[:7]
    b = theta[7] if slope else 1.0
    mu = np.exp(g[phase - 1] + b * s)
    return -(y * np.log(-np.expm1(-mu)) - (1 - y) * mu).sum()


@pytest.mark.parametrize("slope", [False, True])
def test_matches_independent_optimiser(slope):
    s, y, phase = simulate(1, beta=0.8)
    fit = fit_cloglog_offset(s, y, phase, slope=slope)
    x0 = np.r_[np.full(7, -1.0), 1.0] if slope else np.full(7, -1.0)
    ref = minimize(reference_nll, x0, args=(s, y, phase, slope), method="BFGS", options={"gtol": 1e-9, "maxiter": 500})
    mine = np.array([fit["gamma"][k] for k in range(1, 8)] + ([fit["beta"]] if slope else []))
    assert fit["converged"]
    assert np.allclose(mine, ref.x, atol=2e-4)
    assert fit["loglik"] >= -ref.fun - 1e-6  # at least as good as the generic optimiser
    assert (fit["beta"] > 0) if slope else fit["beta"] == 1.0


def test_slope_recovers_the_generating_slope():
    s, y, phase = simulate(2, n=40000, beta=0.7)
    fit = fit_cloglog_offset(s, y, phase, slope=True)
    assert abs(fit["beta"] - 0.7) < 4 * fit["beta_se"]


def test_without_score_gamma_is_the_log_cumulative_hazard_increment():
    rng = np.random.default_rng(3)
    phase = rng.integers(1, 8, 5000)
    y = (rng.random(5000) < 1 - np.exp(-np.exp(GAMMA[phase - 1]))).astype(float)
    fit = fit_cloglog_offset(np.zeros(5000), y, phase)
    for k in range(1, 8):
        rate = y[phase == k].mean()
        assert fit["gamma"][k] == pytest.approx(np.log(-np.log1p(-rate)), abs=1e-6)


def test_probability_is_monotone_and_stable():
    eta = np.array([-800.0, -40.0, -5.0, 0.0, 5.0, 40.0, 800.0])
    p = cloglog_prob(eta)
    assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1)) and np.all(np.diff(p) >= 0)
    assert p[0] == 0.0 and p[-1] == 1.0
    s, y, phase = simulate(4)
    fit = fit_cloglog_offset(s, y, phase)
    grid = np.linspace(-8, 8, 50)
    for k in range(1, 8):
        assert np.all(np.diff(predict_h0(grid, np.full(50, k), fit)) >= 0)  # increasing in the score inside a phase
    with pytest.raises(KeyError):
        predict_h0(grid, np.full(50, 9), fit)


def test_single_class_phase_is_refused():
    s, y, phase = simulate(5, n=500)
    y[phase == 3] = 0.0
    with pytest.raises(ValueError, match="single class"):
        fit_cloglog_offset(s, y, phase)
