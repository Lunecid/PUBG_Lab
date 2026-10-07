"""Maximum likelihood of the grouped-time Cox baseline: P(Y = 1 | s, k) = 1 - exp(-exp(gamma_k + beta * s)) (numpy only).

Prentice-Gloeckler: an interval outcome of a proportional-hazards model is a cloglog regression with the Cox score ``s`` as an offset
(beta = 1) and one baseline increment ``gamma_k = log(H0(k+1) - H0(k))`` per cut phase k.  ``beta`` may instead be shared across the
phases and fitted jointly (the "-slope" sensitivity).  Used by ``adaptation.h0_baseline``; the module has no torch / database dependency
so that it can be tested alone (``tests/test_h0_fit.py``).
"""

from __future__ import annotations

import numpy as np


def cloglog_prob(eta: np.ndarray) -> np.ndarray:
    """1 - exp(-exp(eta)), stable for very small and very large eta."""
    with np.errstate(over="ignore"):
        return -np.expm1(-np.exp(np.minimum(eta, 50.0)))


def _loglik_derivs(eta: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Log-likelihood and its first / second derivative with respect to eta, per row (mu = exp(eta), P = 1 - exp(-mu)).

    y = 1: log(1 - e^-mu), d/deta = a(mu) = mu / (e^mu - 1), d2/deta2 = -mu (a + (a - 1) / (e^mu - 1)) (< 0)
    y = 0: -mu, d/deta = -mu, d2/deta2 = -mu."""
    mu = np.exp(np.minimum(eta, 50.0))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        em1 = np.expm1(mu)
        a = np.where(em1 > 0, mu / em1, 1.0)
        h1 = -mu * (a + (a - 1.0) / em1)
        h1 = np.where(np.isfinite(h1), h1, 0.0)
        ll1 = np.log(-np.expm1(-mu))
    pos = y > 0.5
    ll = float(np.where(pos, ll1, -mu).sum())
    g = np.where(pos, a, -mu)
    h = np.where(pos, h1, -mu)
    return ll, g, h


def fit_cloglog_offset(s, y, phase, *, slope: bool = False, max_iter: int = 100, tol: float = 1e-10) -> dict:
    """Maximum likelihood of P(Y=1 | s, k) = 1 - exp(-exp(gamma_k + beta * s)): beta = 1 (offset) or shared and fitted.

    The log-likelihood is concave in (gamma, beta) (cloglog: both log F and log(1 - F) are concave in eta), so damped Newton
    converges to the unique maximiser when each phase holds both classes.  Returns gamma {phase: value}, SEs from the observed
    information, beta, the log-likelihood and the convergence record."""
    s, y, phase = np.asarray(s, np.float64), np.asarray(y, np.float64), np.asarray(phase, np.int64)
    phases = np.unique(phase)
    K = len(phases)
    idx = np.searchsorted(phases, phase)
    ev = np.bincount(idx, weights=y, minlength=K)
    n = np.bincount(idx, minlength=K).astype(np.float64)
    if (ev <= 0).any() or (ev >= n).any():
        bad = [int(phases[i]) for i in np.flatnonzero((ev <= 0) | (ev >= n))]
        raise ValueError(f"phase(s) {bad} hold a single class: the maximum likelihood estimate of gamma_k is not finite")
    d = K + 1 if slope else K
    # start: gamma_k from the phase rate at beta = 1 (mean eta matched), beta = 1
    rate = ev / n
    sbar = np.bincount(idx, weights=s, minlength=K) / n
    theta = np.zeros(d)
    theta[:K] = np.log(-np.log1p(-rate)) - sbar
    if slope:
        theta[K] = 1.0

    def eval_at(th):
        beta = th[K] if slope else 1.0
        return _loglik_derivs(th[:K][idx] + beta * s, y)

    ll, g_row, h_row = eval_at(theta)
    converged, it = False, 0
    for it in range(1, max_iter + 1):
        grad = np.zeros(d)
        grad[:K] = np.bincount(idx, weights=g_row, minlength=K)
        H = np.zeros((d, d))
        H[np.arange(K), np.arange(K)] = np.bincount(idx, weights=h_row, minlength=K)
        if slope:
            grad[K] = float((g_row * s).sum())
            hs = np.bincount(idx, weights=h_row * s, minlength=K)
            H[:K, K] = H[K, :K] = hs
            H[K, K] = float((h_row * s * s).sum())
        step = np.linalg.solve(-H + 1e-12 * np.eye(d), grad)
        dec = float(grad @ step)  # Newton decrement (>= 0 for a concave function)
        if dec < tol:
            converged = True
            break
        t = 1.0
        while True:
            cand = theta + t * step
            ll_c, g_c, h_c = eval_at(cand)
            if np.isfinite(ll_c) and ll_c >= ll + 1e-4 * t * dec:
                break
            t *= 0.5
            if t < 1e-12:
                raise RuntimeError("line search failed in fit_cloglog_offset")
        theta, ll, g_row, h_row = cand, ll_c, g_c, h_c
    # observed information at the solution
    H = np.zeros((d, d))
    H[np.arange(K), np.arange(K)] = -np.bincount(idx, weights=h_row, minlength=K)
    if slope:
        hs = -np.bincount(idx, weights=h_row * s, minlength=K)
        H[:K, K] = H[K, :K] = hs
        H[K, K] = float(-(h_row * s * s).sum())
    cov = np.linalg.inv(H)
    se = np.sqrt(np.diag(cov))
    return {"gamma": {int(k): float(v) for k, v in zip(phases, theta[:K])}, "gamma_se": {int(k): float(v) for k, v in zip(phases, se[:K])},
            "beta": float(theta[K]) if slope else 1.0, "beta_se": float(se[K]) if slope else None, "slope_fitted": bool(slope), "loglik": float(ll),
            "n_rows": int(len(y)), "n_events": int(y.sum()), "n_iter": int(it), "converged": bool(converged), "newton_decrement": float(dec),
            "rows_by_phase": {int(k): int(v) for k, v in zip(phases, n)}, "events_by_phase": {int(k): int(v) for k, v in zip(phases, ev)}}  # fmt: skip


def predict_h0(s, phase, fit: dict) -> np.ndarray:
    """p = 1 - exp(-exp(gamma_k + beta * s)) for rows of the given cut phases (every phase must have been fitted)."""
    phase = np.asarray(phase, np.int64)
    miss = sorted(set(phase.tolist()) - set(fit["gamma"]))
    if miss:
        raise KeyError(f"phase(s) {miss} have no fitted gamma")
    g = np.array([fit["gamma"][int(k)] for k in phase])
    return cloglog_prob(g + fit["beta"] * np.asarray(s, np.float64))
