from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from pubg_lab.survival.v3.data.cohorts import cohort_choices, resolve_cohort, resolve_map_mode
from pubg_lab.survival.v3.experiments.run_matrix import _cmd_eval, _cmd_train


def _runner_args(**overrides):
    base = {
        "limit": None,
        "cohort": None,
        "map": None,
        "mode": None,
        "epochs": 1,
        "batch": 2,
        "workers": 0,
        "prefetch_factor": 2,
        "log_every": 10,
        "device": "cpu",
        "split_json": Path("/tmp/split.json"),
        "seed": 42,
        "cached": False,
        "combat_in": False,
        "no_v2_features": False,
        "eval_split": "test",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _matrix_row():
    return {
        "model": "mr",
        "graph_encoder": "none",
        "global_fusion": "concat",
        "team_decoder": "mlp",
        "pairwise_rank_weight": "0.1",
        "edge_mode": "none",
        "edge_control": "observed",
        "feature_preset": "om0_context",
    }


def test_resolve_required_cohort_exact_label():
    spec = resolve_cohort("Erangel TPP squad")
    assert spec.map_name == "Baltic_Main"
    assert spec.game_mode == "squad"
    assert spec.perspective == "TPP"
    assert spec.team_size == "squad"


def test_resolve_required_cohort_slug_and_alias():
    by_slug = resolve_cohort("erangel_tpp_squad")
    by_internal = resolve_cohort("Baltic_Main__TPP__squad")
    assert by_slug == by_internal
    assert resolve_cohort("Miramar TPP squad").map_name == "Desert_Main"


def test_fpp_cohort_maps_to_fpp_game_mode():
    spec = resolve_cohort("Taego FPP squad")
    assert spec.map_name == "Tiger_Main"
    assert spec.game_mode == "squad-fpp"


def test_resolve_map_mode_rejects_mixed_cohort_and_raw_filters():
    with pytest.raises(ValueError, match="--cohort"):
        resolve_map_mode(cohort="Erangel TPP squad", map_name="Baltic_Main", mode=None)


def test_unknown_cohort_error_lists_human_readable_choices():
    with pytest.raises(ValueError) as exc:
        resolve_cohort("Atlantis TPP squad")
    assert "Erangel TPP squad" in str(exc.value)
    assert cohort_choices()[0] == "Erangel TPP squad"


def test_run_matrix_passes_cohort_without_duplicate_raw_filters():
    args = _runner_args(cohort="Erangel TPP squad", map="Baltic_Main", mode="squad")
    train_cmd = _cmd_train(_matrix_row(), args, Path("/tmp/model.pt"))
    eval_cmd = _cmd_eval(_matrix_row(), args, Path("/tmp/model.pt"), Path("/tmp/eval.json"))

    assert "--cohort" in train_cmd
    assert "--cohort" in eval_cmd
    assert "--map" not in train_cmd
    assert "--mode" not in train_cmd
    assert "--map" not in eval_cmd
    assert "--mode" not in eval_cmd
