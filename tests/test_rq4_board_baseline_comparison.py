from __future__ import annotations

import math

import pandas as pd

from projects.battle_royale.evaluation.rq4.compare_rq4_board_baselines import build_sign_count, evaluate_board_scores, filter_common_score_universe, summarize_board_scores


def _row(variant: str, phase: int, team: int, risk: float, adv: float, hazard: int, top10: int) -> dict:
    return {
        "cohort_slug": "toy",
        "model_variant": variant,
        "match_id": "m1",
        "team_idx": team - 1,
        "team_id": str(team),
        "phase_slot": phase,
        "cut_phase": phase,
        "hazard_true": hazard,
        "risk_score": risk,
        "advantage_score": adv,
        "final_rank_norm": 0.1 * team,
        "final_top5": int(team == 1),
        "final_top10": top10,
        "observed_readout": 1,
    }


def test_evaluate_board_scores_includes_delta_metrics() -> None:
    rows = [
        _row("lstm_realtime_board", 1, 1, 0.20, 0.80, 0, 1),
        _row("lstm_realtime_board", 1, 2, 0.30, 0.60, 1, 0),
        _row("lstm_realtime_board", 1, 3, 0.10, 0.50, 0, 0),
        _row("lstm_realtime_board", 2, 1, 0.25, 0.90, 0, 1),
        _row("lstm_realtime_board", 2, 2, 0.35, 0.50, 0, 0),
        _row("lstm_realtime_board", 2, 3, 0.60, 0.40, 1, 0),
    ]
    board = evaluate_board_scores(pd.DataFrame(rows), top_k=2)

    phase1 = board[board["phase_slot"] == 1].iloc[0]
    phase2 = board[board["phase_slot"] == 2].iloc[0]
    assert phase1["risk_top1_hit"] == 1.0
    assert math.isnan(phase1["risk_delta_top1_hit"])
    assert phase2["risk_top1_hit"] == 1.0
    assert phase2["risk_delta_top1_hit"] == 1.0
    assert phase2["adv_delta_top_final_top10"] == 1


def test_risk_top1_uses_same_no_elimination_denominator_as_topk() -> None:
    rows = [
        _row("lstm_realtime_board", 1, 1, 0.20, 0.80, 0, 1),
        _row("lstm_realtime_board", 1, 2, 0.30, 0.60, 0, 0),
        _row("lstm_realtime_board", 1, 3, 0.10, 0.50, 0, 0),
    ]
    board = evaluate_board_scores(pd.DataFrame(rows), top_k=2)

    phase1 = board.iloc[0]
    assert phase1["next_phase_eliminations"] == 0
    assert math.isnan(phase1["risk_top1_hit"])
    assert math.isnan(phase1["risk_top3_hit"])
    assert math.isnan(phase1["risk_top2_hit"])


def test_common_score_universe_intersects_match_phase_team_rows() -> None:
    rows = [
        _row("lstm_realtime_board", 1, 1, 0.10, 0.80, 0, 1),
        _row("lstm_realtime_board", 1, 2, 0.90, 0.20, 1, 0),
        _row("random_alive_board", 1, 1, 0.90, 0.10, 0, 1),
    ]

    filtered, coverage = filter_common_score_universe(pd.DataFrame(rows))

    assert len(filtered) == 2
    assert set(filtered["team_id"].astype(str)) == {"1"}
    assert coverage["input_rows"] == 3
    assert coverage["output_rows"] == 2
    assert coverage["dropped_rows_by_variant"]["lstm_realtime_board"] == 1


def test_summarize_and_sign_count_compare_lstm_to_baseline() -> None:
    rows = [
        _row("lstm_realtime_board", 1, 1, 0.10, 0.80, 0, 1),
        _row("lstm_realtime_board", 1, 2, 0.90, 0.20, 1, 0),
        _row("random_alive_board", 1, 1, 0.90, 0.10, 0, 1),
        _row("random_alive_board", 1, 2, 0.10, 0.20, 1, 0),
    ]
    board = evaluate_board_scores(pd.DataFrame(rows), top_k=2)
    by_cohort, aggregate = summarize_board_scores(board, top_k=2)
    signs = build_sign_count(by_cohort, metrics=["risk_top1_hit"])

    assert set(aggregate["model_variant"]) == {"lstm_realtime_board", "random_alive_board"}
    row = signs.iloc[0]
    assert row["baseline_variant"] == "random_alive_board"
    assert row["lstm_wins"] == 1
    assert row["lstm_losses"] == 0
