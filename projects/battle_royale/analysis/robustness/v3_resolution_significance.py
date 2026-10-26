"""Resolution-group significance: does the multi-resolution snapshot model beat
the phase-only model per cut phase (match-level paired bootstrap)?
Reuses v3_phase_significance helpers; reads resolution pred CSVs in v3_pred/.
"""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from projects.battle_royale.analysis.robustness.v3_phase_significance import (
    report_single,
    report_paired,
)

# per-phase AUC/Spearman CI for the MR (with graph) and phase (with graph) models
report_single("resolution_mr_rgcn")
report_single("resolution_phase_rgcn")
# the headline: snapshot tier effect, holding graph fixed and holding graph-off
report_paired(
    "resolution_mr_rgcn",
    "resolution_phase_rgcn",
    "MR vs phase (both with graph) — pure snapshot effect",
)
report_paired(
    "resolution_mr_tcn",
    "resolution_phase_tcn",
    "MR vs phase (both no graph) — pure snapshot effect",
)
