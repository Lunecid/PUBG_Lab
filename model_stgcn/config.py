"""Central path/template constants for the model_stgcn experiment suite.

Producer (run_domain_training) and consumers (edge_sanity, phase_wise_eval)
must agree on the checkpoint file name — change it HERE only.
See docs/CODE_REVIEW_2026-06-12.md D11.
"""

from __future__ import annotations

from pathlib import Path

CHECKPOINT_ROOT = Path("model_stgcn/checkpoints")

PHASE_CHECKPOINT_TEMPLATE = "phase_{edge_mode}_e{epochs}.pt"
DEFAULT_PHASE_EPOCHS = 12

DEFAULT_PHASE_CKPT = str(CHECKPOINT_ROOT / "dynamic_stgnn_phase.pt")
DEFAULT_SPLIT_JSON = str(CHECKPOINT_ROOT / "split_phase_all_seed42.json")
DEFAULT_ALIGNED_PREDICTIONS = str(
    CHECKPOINT_ROOT / "phase_result_analysis" / "hazard_predictions_test_aligned.csv"
)


def phase_checkpoint_name(edge_mode: str, epochs: int = DEFAULT_PHASE_EPOCHS) -> str:
    return PHASE_CHECKPOINT_TEMPLATE.format(edge_mode=edge_mode, epochs=epochs)
