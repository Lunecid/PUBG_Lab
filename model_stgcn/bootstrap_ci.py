"""Match-level (cluster) bootstrap confidence intervals for evaluation metrics.

같은 매치의 평가 셀들은 상관되어 있으므로 row-level bootstrap은 CI를 과소추정
한다.  매치(클러스터) 단위로 복원추출해 within-match 상관을 보존한다.
docs/CODE_REVIEW_2026-06-12.md C7.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Percentile bootstrap CI with match-level resampling.

    Parameters:
        y_true, y_prob: (n,) flattened evaluation cells
        groups: (n,) cluster label per cell (match_id)
        metric_fn: e.g. sklearn roc_auc_score — may raise ValueError on
            single-class resamples; those draws are counted in n_failed.

    Returns:
        {point, lo, hi, n_boot, n_failed, n_clusters}
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    groups = np.asarray(groups)
    if y_true.size == 0:
        return {
            "point": float("nan"), "lo": float("nan"), "hi": float("nan"),
            "n_boot": 0, "n_failed": 0, "n_clusters": 0,
        }

    point = float(metric_fn(y_true, y_prob))

    uniq = np.unique(groups)
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    starts = np.searchsorted(sorted_groups, uniq, side="left")
    ends = np.searchsorted(sorted_groups, uniq, side="right")
    cluster_rows = [order[s:e] for s, e in zip(starts, ends)]

    rng = np.random.default_rng(seed)
    stats: list[float] = []
    failed = 0
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([cluster_rows[k] for k in pick])
        try:
            stats.append(float(metric_fn(y_true[idx], y_prob[idx])))
        except ValueError:
            failed += 1

    if stats:
        lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    else:
        lo = hi = float("nan")
    return {
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "n_boot": len(stats),
        "n_failed": failed,
        "n_clusters": int(len(uniq)),
    }
