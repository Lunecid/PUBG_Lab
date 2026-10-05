"""HGB determinism contract (design doc section 6, items 2-3; 2026-10-05).

DB-free: the two HGB baselines must (a) not early-stop on a row-order-dependent validation
split, (b) read their rows in a fixed ``match_id, team_id, cut_phase`` order, and
(c) the tabular evaluation rows are restricted to phases 1..7 like every other arm.
"""

from __future__ import annotations

import argparse
import contextlib

import pytest


class _FakeCursor:
    def __init__(self, sink: list[str]) -> None:
        self.sink = sink
        self.description = [("match_id",), ("team_id",), ("phase_idx",)]

    def execute(self, sql, params=None):
        self.sink.append(sql)

    def fetchall(self):
        return []


class _FakeConn:
    def __init__(self, sink: list[str]) -> None:
        self.sink = sink

    @contextlib.contextmanager
    def cursor(self):
        yield _FakeCursor(self.sink)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _hgb_of(pipeline):
    return pipeline.steps[-1][1]


def test_snapshot_tabular_hgb_has_no_early_stopping():
    from model_stgcn import snapshot_tabular_baseline as sb

    args = argparse.Namespace(
        max_iter=40, learning_rate=0.1, l2_regularization=0.0, seed=42, n_estimators=10, n_jobs=1
    )
    model = _hgb_of(sb._classifier(args, learner="hgb"))
    assert model.get_params()["early_stopping"] is False


def test_feature_ablation_hgb_has_no_early_stopping():
    from model_stgcn import feature_ablation as fa

    model = _hgb_of(fa._classifier(["a", "b"], learner="hgb"))
    assert model.get_params()["early_stopping"] is False


def test_snapshot_tabular_rows_have_a_fixed_order():
    from model_stgcn import snapshot_tabular_baseline as sb

    sql = " ".join(sb._snapshot_tabular_sql().split())
    assert "ROW_NUMBER() OVER (ORDER BY l.match_id, l.team_id, l.cut_phase) AS rid" in sql
    assert "ROW_NUMBER() OVER () " not in sql
    assert sql.endswith("ORDER BY l.match_id, l.team_id, l.cut_phase")


def test_feature_ablation_rows_are_ordered_and_limited_to_phases_1_to_7(monkeypatch):
    from model_stgcn import feature_ablation as fa

    seen: list[str] = []
    monkeypatch.setattr(fa, "connect", lambda *a, **k: _FakeConn(seen))
    rows, cols = fa.load_phase_rows_from_view(["m1"], "stdb.v_team_phase_feature_long")
    assert rows == [] and cols == ["match_id", "team_id", "phase_idx"]
    assert len(seen) == 1
    sql = " ".join(seen[0].split())
    assert "AND phase_idx BETWEEN 1 AND 7" in sql
    assert sql.endswith("ORDER BY match_id, team_id, phase_idx")


@pytest.mark.parametrize("learner", ["hgb"])
def test_two_fits_on_shuffled_rows_agree(learner):
    """With early stopping off the HGB fit depends only on the data, not on a hidden split."""
    import numpy as np

    from model_stgcn import feature_ablation as fa

    rng = np.random.default_rng(0)
    n = 12_000  # > 10_000 would have switched 'auto' early stopping on
    X = rng.standard_normal((n, 4)).astype("float64")
    y = ((X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.standard_normal(n)) > 0).astype(int)
    import pandas as pd

    df = pd.DataFrame(X, columns=list("abcd"))
    clf_a = fa._classifier(list("abcd"), learner=learner, max_iter=20)
    clf_a.fit(df, y)
    # same rows, identical order -> bit-identical probabilities across two fits
    clf_b = fa._classifier(list("abcd"), learner=learner, max_iter=20)
    clf_b.fit(df, y)
    pa, pb = clf_a.predict_proba(df)[:, 1], clf_b.predict_proba(df)[:, 1]
    assert np.array_equal(pa, pb)
    # iterations actually ran to max_iter (no early stop)
    assert _hgb_of(clf_a).n_iter_ == 20
