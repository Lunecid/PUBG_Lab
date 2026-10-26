"""Canonical module names for explicit baseline subprocesses; no imports of applications."""

from pubg_lab.common.paths import REPO_ROOT

COMMANDS = {
    "schema": "projects.krafton_baseline.data.schema",
    "features": "projects.krafton_baseline.data.features",
    "data": "projects.krafton_baseline.data.data",
    "labels": "projects.krafton_baseline.data.labels",
    "extract": "projects.krafton_baseline.data.extract",
    "kftable": "projects.krafton_baseline.data.kftable",
    "ourfeat": "projects.krafton_baseline.models.ourfeat",
    "train": "projects.krafton_baseline.training.train",
    "ourfeat_hazard": "projects.krafton_baseline.training.ourfeat_hazard",
    "hgb_kraftonfeat": "projects.krafton_baseline.training.hgb_kraftonfeat",
    "h0_baseline": "projects.krafton_baseline.training.h0_baseline",
    "h0_fit": "projects.krafton_baseline.training.h0_fit",
    "phase_auc": "projects.krafton_baseline.evaluation.phase_auc",
    "k3_contrasts": "projects.krafton_baseline.evaluation.k3_contrasts",
    "diagnostics": "projects.krafton_baseline.evaluation.diagnostics",
    "validate_causal": "projects.krafton_baseline.evaluation.validate_causal",
    "diag_phase_init": "projects.krafton_baseline.evaluation.diag_phase_init",
    "k3_facts": "projects.krafton_baseline.reporting.k3_facts",
    "h0_report": "projects.krafton_baseline.reporting.h0_report",
    "parity": "projects.krafton_baseline.reporting.parity",
    "summarize": "projects.krafton_baseline.reporting.summarize",
    "run_all": "projects.krafton_baseline.pipelines.run_all",
}


def module_for(name: str) -> str:
    return COMMANDS[name]


def source_for(name: str):
    """Locate current adapter source for code identities without importing it."""
    return REPO_ROOT / (module_for(name).replace(".", "/") + ".py")
