"""Results aggregator of the PTR matrix: design doc sections 4.1 (v2.2 decision rules) and 5.

Input: the run directory written by :mod:`model_stgcn.ptr.run_matrix`,
``<runs>/<cohort>/<arm>/seed<s>/{pred_test.csv, metrics.json}`` (a run without both files is not finished and is ignored,
with a note in ``issues.csv``).  Everything is computed from the prediction files; nothing is retrained.

Pipeline
--------
1. Per cohort and stratum (``all`` matches, then each ``match_type``): all seed-42 arms are scored on the *same*
   match-level bootstrap resamples (:func:`model_stgcn.ptr.metrics.bootstrap_arms`, 2,000 resamples, a fixed seed per
   cohort/stratum), so every contrast is a linear combination of per-arm resampled metrics on identical rows.  A
   contrast is ``estimate`` (full data), ``se`` (bootstrap SD) and a percentile 95% interval.
2. Pooling over cohorts per (contrast, metric, stratum): DerSimonian-Laird random effects on the per-cohort estimates
   with their bootstrap SEs (``tau2``, ``Q``, ``I2``, pooled 95% / 90% interval, same-sign count, z test).  The p-value is
   one-sided for an *improvement* (H0: effect <= 0 for AUC / AP / correlations; for NLL / Brier a decrease is the
   improvement).
3. Holm correction over the five primary tests P1-P5 (stratum ``all``, a missing test counts as p = 1) and nowhere else.
4. Verdicts (``gain`` / ``equivalent`` / ``inconclusive``, design v2.2) for metrics that have a SESOI.  ``gain``: Holm
   adjusted p < 0.05 and pooled estimate >= SESOI; ``equivalent``: pooled 90% interval inside +-SESOI; else
   ``inconclusive``.  Secondary contrasts and strata carry no Holm adjustment, so their verdict uses the unadjusted
   one-sided p (``verdict_basis``); they are exploratory.
5. Recovery ratio kappa = (M11RZ-noHab - M11R-noHab) / (M11R - M11R-noHab) (AUC): reported only when the pooled 95%
   interval of its denominator starts at or above the SESOI; otherwise ``not estimable``.  The pooled interval is a
   percentile interval of the ratio of the pooled numerator / denominator formed from the per-cohort resamples with the
   DL weights held fixed.  Per-cohort kappa is descriptive.
6. Seed 42 feeds every contrast; seeds 13 / 2026 only give the seed-to-seed SD (``seed_stability.csv``), never pooled.
7. External baselines (design doc 4.3, exploratory): ``SAGONG-xgb/rf/lr`` and ``KRAFTON-tf/mlp/tf-ourfeat`` (their transformer on
   our features) are inventoried as kind ``external`` and enter only eleven paired, DL-pooled contrasts without Holm
   (``FORMULAS`` group ``external``): M11R against every SAGONG / KRAFTON model, HGB-cut against SAGONG-xgb and KRAFTON-tf,
   KRAFTON-tf - KRAFTON-mlp, the input effect ``krafton_ourfeat_input`` (KRAFTON-tf-ourfeat - KRAFTON-tf: their backbone and
   loss fixed, our features vs theirs) and ``m01_vs_krafton_tf_ourfeat`` (our model and loss vs theirs on the same features);
   ``tables/tab_external.tex`` (arm means next to the references) and ``tab_external_contrasts.tex`` (three blocks) exist
   only when such runs exist.  Adding arms never changes another contrast: the bootstrap resamples depend on the cohort and
   stratum only.
   ``SAGONG-hgb`` is HGB-cut's learner (same settings, rows and labels) on Sagong's 12 inputs: ``hgbcut_vs_sagong_hgb`` is the clean
   input effect and ``sagong_hgb_vs_sagong_xgb`` the learner check.  Contrasts in which more than one factor differs are tagged
   ``(bundled)`` in ``tab_external_contrasts.tex`` (KRAFTON-tf -> K2, HGB-kraftonfeat - KRAFTON-tf, HGB-cut - SAGONG-xgb, M01 - K2) and are not
   decomposition steps.
8. Meta-regression on the training size (design doc 4.4, exploratory): the per-cohort AUC contrasts of P1, P2, P5, M11 - M00 and
   M11R - HGB-cut regressed on ln(training matches), random effects with the method-of-moments (generalised DerSimonian-Laird)
   tau^2 and inverse-variance weights; slope, normal and Knapp-Hartung intervals / p, tau^2 with and without the moderator,
   and the crossover N where the fitted effect is zero (Fieller interval, bounded only for a significant slope):
   ``metareg.csv``, ``metareg_data.csv``, ``figures/metareg_bubble.{pdf,png}``.

9. Union arms (design doc 4.6, post hoc and exploratory): ``HGB-union`` (HGB-cut plus KRAFTON's reconstructed ``kf_*`` features) and
   ``M11R-union`` (M11R plus the same features) are our own models (kind ``hgb`` / ``neural``), not external ones.  Five paired,
   DL-pooled contrasts without Holm (``FORMULAS`` group ``union``): ``hgb_union_vs_hgbcut`` and ``m11r_union_vs_m11r`` (what the
   KRAFTON features add to ours), ``m11r_union_vs_krafton_tf``, ``m11r_union_vs_sagong_xgb`` and ``hgb_union_vs_hgb_kraftonfeat``
   (what our representation adds to KRAFTON's inputs with the same GBDT); they form the fifth block of
   ``tab_external_contrasts.tex`` and the ``union`` row of ``tab_grid.tex``, which exist when external or union arms were run.
   Adding arms never changes another contrast (the resamples depend on the cohort and stratum only).

10. Within-phase AUC (exploratory, complementary): ``auc_within_phase`` = the UNWEIGHTED mean over cut phases of the per-phase hazard AUC
   (each eligible phase counts once, whatever its row count; phases with both classes and at least 30 rows, fixed from the data of
   the evaluated frame; ``model_stgcn.ptr.metrics``).  This one definition is used for every arm, contrast, stratum and table.  It is computed
   for every arm, every contrast and every stratum next to the seven metrics (paired match bootstrap: the per-phase AUCs are
   recomputed on each resample), DL-pooled like them, never in the Holm family and without a verdict (no SESOI): the pre-specified
   primary metric stays the pooled AUC.  The paper tables keep their seven metric columns.
11. K3 ladder and decomposition (exploratory): the KRAFTON transformer on OUR features comes in three formulations, K2 =
   ``KRAFTON-tf-ourfeat`` (their Cox loss), K3a = ``KRAFTON-tf-ourfeat-bce`` (BCE on the hazard rows) and K3 =
   ``KRAFTON-tf-ourfeat-hazard`` (BCE with a phase baseline).  Contrasts ``k3a_vs_k2`` (objective), ``k3_vs_k3a`` (phase baseline),
   ``k3_vs_k2`` (formulation total) and ``m01_vs_k3`` (the residual: our architecture against theirs under the same formulation and
   features); ``tables/tab_decomp.tex`` lines up seven CLEAN single-factor steps (pooled and within-phase delta AUC with the cohort
   counts, every row to - from): input vs Sagong (HGB-cut - SAGONG-hgb), input vs KRAFTON (HGB-cut - HGB-kraftonfeat), objective (K3a - K2),
   phase baseline (K3 - K3a), architecture (M01 - K3), structure (M11R - M01) and the extended representation (HGB-union - HGB-cut).
   Bundled contrasts (KRAFTON-tf -> K2, HGB-kraftonfeat - KRAFTON-tf, HGB-cut - SAGONG-xgb, M01 - K2) are not steps; a step whose arms were
   not run is left out, never replaced by a bundled one.  The Sagong x GBDT cell of ``tab_grid.tex`` is SAGONG-hgb, so the GBDT column
   is one learner (HistGradientBoosting, HGB-cut settings) with four inputs.

Outputs (``--out-dir``, default ``reports/bigcomp2027_results``): tidy CSVs, ``summary.json``, LaTeX ``tabular`` files
(``tables/``) with the structure of the pre-registered paper skeletons, and the forest plot of P1-P5 (``figures/``, PDF + PNG
and a CSV twin of the plotted numbers).  ``--redraw-figures`` regenerates the forest and bubble plots from the CSVs of an
earlier run without any bootstrap.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import logging
import math
import multiprocessing
import os
import re
import sys
import zlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from model_stgcn.ptr import metrics as M
from model_stgcn.ptr.util import REPO_ROOT, git_info, to_jsonable, write_json

logger = logging.getLogger("model_stgcn.ptr.aggregate")

# --------------------------------------------------------------------------- constants (design doc 4.1 / 5)
N_BOOT = 2000
ALPHA = 0.05
MAIN_SEED = 42
RANK_COL = "rank_score"
STRATUM_ALL = "all"
MIN_STRATUM_MATCHES = 10  # a stratum with fewer test matches in a cohort is skipped (a bootstrap of 3 matches means nothing)
# phase-group strata (exploratory): rows of the cut phases k in the group, next to the match-type strata.  A stratum name
# "phase:early" etc.; the cut phase k is the CSV column ``phase`` (1-based).
PHASE_GROUPS: dict[str, tuple[int, ...]] = {"early": (1, 2), "mid": (3, 4), "late": (5, 6, 7)}
PHASE_PREFIX = "phase:"
METRICS: tuple[str, ...] = M.ALL_METRIC_NAMES  # auc, ap, nll, brier, spearman, top5_auc, cindex + the exploratory auc_within_phase
TABLE_METRICS: tuple[str, ...] = M.METRIC_NAMES  # the seven metrics of the paper tables (the exploratory metric has no column there)
SESOI = {"auc": 0.005, "spearman": 0.01, "top5_auc": 0.005, "cindex": 0.005}  # auc_within_phase has none: no verdict, no Holm
DIRECTION = {
    "auc": 1, "ap": 1, "nll": -1, "brier": -1, "spearman": 1, "top5_auc": 1, "cindex": 1, "auc_within_phase": 1,
}  # +1: higher is better
METRIC_LABEL = {
    "auc": "AUC", "ap": "AP", "nll": "NLL", "brier": "Brier", "spearman": "Spearman", "top5_auc": "Top-5 AUC",
    "cindex": "C-index", "auc_within_phase": "AUC within phase",
}  # fmt: skip

MAIN_ARMS: tuple[str, ...] = (
    "M00", "M10", "M01", "M11", "M11R", "M11R-shuf", "M11RZ", "M11R-noHab", "M11RZ-noHab",
)  # fmt: skip
EXPLORATORY_ARMS: tuple[str, ...] = ("M11R-noProx", "M11R-noCombat", "M11R-noCopress")
STAGE2_ARMS: tuple[str, ...] = ("M11RZ-P", "M11RZ-P-tm")
HGB_ARMS: tuple[str, ...] = ("HGB-cut", "HGB-lag", "HGB-noHab")
# external baselines (design doc 4.3, exploratory): re-implementations of Sagong et al. (2025) and the KRAFTON win-probability
# models, run on the same cohorts / splits / label v2 / rows; they only ever enter exploratory contrasts, never the primary family
# KRAFTON-tf-ourfeat is their transformer on OUR token features (same backbone and loss as KRAFTON-tf, other input).
# HGB-kraftonfeat is a GBDT on KRAFTON's reconstructed inputs (C1 grid: input x learner); it is external although its name
# starts with HGB, and none of the HGB-* formulas (HGB_ARMS) involves it.
# The K3 ladder (their transformer on OUR features, three formulations): K2 = KRAFTON-tf-ourfeat (their Cox loss), K3a =
# KRAFTON-tf-ourfeat-bce (BCE on the hazard rows), K3 = KRAFTON-tf-ourfeat-hazard (BCE with a phase baseline).
# SAGONG-hgb and HGB-kraftonfeat run OUR learner (the HGB-cut settings, rows and labels) on THEIR inputs (Sagong's 12 inputs, KRAFTON's
# reconstructed inputs): the clean input effects are HGB-cut - SAGONG-hgb and HGB-cut - HGB-kraftonfeat.
EXTERNAL_ARMS: tuple[str, ...] = (
    "SAGONG-xgb", "SAGONG-rf", "SAGONG-lr", "KRAFTON-tf", "KRAFTON-mlp", "KRAFTON-tf-ourfeat", "KRAFTON-tf-ourfeat-bce",
    "KRAFTON-tf-ourfeat-hazard", "SAGONG-hgb", "HGB-kraftonfeat",
)  # fmt: skip
EXTERNAL_PREFIXES: tuple[str, ...] = ("SAGONG-", "KRAFTON-")
# union arms (design doc 4.6, post hoc / exploratory): our representation united with KRAFTON's reconstructed kf_* features; they are
# our own models (kind hgb / neural), separate from the HGB_ARMS of the main matrix and from the external baselines
UNION_ARMS: tuple[str, ...] = ("HGB-union", "M11R-union")
ARM_ORDER: tuple[str, ...] = MAIN_ARMS + EXPLORATORY_ARMS + STAGE2_ARMS + HGB_ARMS + UNION_ARMS + EXTERNAL_ARMS
KAPPA_ARMS = {"num_a": "M11RZ-noHab", "num_b": "M11R-noHab", "den_a": "M11R", "den_b": "M11R-noHab"}


@dataclass(frozen=True)
class Formula:
    """A linear contrast of arm metrics, ``sum_k coefs[k] * metric(arm_k)``."""

    name: str
    coefs: dict
    label: str
    group: str  # primary | simple | interaction | rq3 | rq4 | hgb | stage2 | stability
    optional: bool = False  # arms are not expected in every cohort (exploratory / stage 2): no issue when missing
    pooled: bool = True  # False: only used for the seed-to-seed SD


_H = 0.5
FORMULAS: dict[str, Formula] = {
    f.name: f
    for f in (
        Formula("phase_main", {"M10": _H, "M00": -_H, "M11": _H, "M01": -_H}, "phase-axis main effect", "primary"),
        Formula("team_main", {"M01": _H, "M00": -_H, "M11": _H, "M10": -_H}, "team-axis main effect", "primary"),
        Formula("rel_vs_shuf", {"M11R": 1, "M11R-shuf": -1}, "M11R - M11R-shuf", "primary"),
        Formula("zone_nohab", {"M11RZ-noHab": 1, "M11R-noHab": -1}, "M11RZ-noHab - M11R-noHab", "primary"),
        Formula("m10_m00", {"M10": 1, "M00": -1}, "M10 - M00", "simple"),
        Formula("m11_m01", {"M11": 1, "M01": -1}, "M11 - M01", "simple"),
        Formula("m01_m00", {"M01": 1, "M00": -1}, "M01 - M00", "simple"),
        Formula("m11_m10", {"M11": 1, "M10": -1}, "M11 - M10", "simple"),
        Formula("interaction", {"M11": 1, "M10": -1, "M01": -1, "M00": 1}, "(M11 - M10) - (M01 - M00)", "interaction"),
        Formula("m11r_m11", {"M11R": 1, "M11": -1}, "M11R - M11", "rq3"),
        Formula("m11r_noprox", {"M11R": 1, "M11R-noProx": -1}, "M11R - M11R-noProx", "rq3", optional=True),
        Formula("m11r_nocombat", {"M11R": 1, "M11R-noCombat": -1}, "M11R - M11R-noCombat", "rq3", optional=True),
        Formula("m11r_nocopress", {"M11R": 1, "M11R-noCopress": -1}, "M11R - M11R-noCopress", "rq3", optional=True),
        Formula("m11rz_m11r", {"M11RZ": 1, "M11R": -1}, "M11RZ - M11R", "rq4"),
        Formula("hab_value", {"M11R": 1, "M11R-noHab": -1}, "M11R - M11R-noHab (kappa denominator)", "rq4"),
        Formula("hgb_cut_nohab", {"HGB-cut": 1, "HGB-noHab": -1}, "HGB-cut - HGB-noHab", "hgb"),
        Formula("hgb_lag_cut", {"HGB-lag": 1, "HGB-cut": -1}, "HGB-lag - HGB-cut", "hgb"),
        Formula("p_m11rz", {"M11RZ-P": 1, "M11RZ": -1}, "M11RZ-P - M11RZ", "stage2", optional=True),
        Formula("p_tm", {"M11RZ-P": 1, "M11RZ-P-tm": -1}, "M11RZ-P - M11RZ-P-tm", "stage2", optional=True),
        Formula("m11_m00", {"M11": 1, "M00": -1}, "M11 - M00", "simple"),
        Formula("m11r_vs_hgbcut", {"M11R": 1, "HGB-cut": -1}, "M11R - HGB-cut", "hgb"),
        Formula("m11r_vs_sagong_xgb", {"M11R": 1, "SAGONG-xgb": -1}, "M11R - SAGONG-xgb", "external", optional=True),
        Formula("hgbcut_vs_sagong_xgb", {"HGB-cut": 1, "SAGONG-xgb": -1}, "HGB-cut - SAGONG-xgb", "external", optional=True),
        Formula("m11r_vs_krafton_tf", {"M11R": 1, "KRAFTON-tf": -1}, "M11R - KRAFTON-tf", "external", optional=True),
        Formula("hgbcut_vs_krafton_tf", {"HGB-cut": 1, "KRAFTON-tf": -1}, "HGB-cut - KRAFTON-tf", "external", optional=True),
        Formula("krafton_tf_mlp", {"KRAFTON-tf": 1, "KRAFTON-mlp": -1}, "KRAFTON-tf - KRAFTON-mlp", "external", optional=True),
        Formula("m11r_vs_sagong_rf", {"M11R": 1, "SAGONG-rf": -1}, "M11R - SAGONG-rf", "external", optional=True),
        Formula("m11r_vs_sagong_lr", {"M11R": 1, "SAGONG-lr": -1}, "M11R - SAGONG-lr", "external", optional=True),
        Formula("m11r_vs_krafton_mlp", {"M11R": 1, "KRAFTON-mlp": -1}, "M11R - KRAFTON-mlp", "external", optional=True),
        Formula("m11r_vs_krafton_tf_ourfeat", {"M11R": 1, "KRAFTON-tf-ourfeat": -1}, "M11R - KRAFTON-tf-ourfeat", "external", optional=True),
        # the input effect: their backbone and loss fixed, our features vs theirs
        Formula("krafton_ourfeat_input", {"KRAFTON-tf-ourfeat": 1, "KRAFTON-tf": -1}, "KRAFTON-tf-ourfeat - KRAFTON-tf", "external", optional=True),
        # our model and loss vs theirs on the same (our) features
        Formula("m01_vs_krafton_tf_ourfeat", {"M01": 1, "KRAFTON-tf-ourfeat": -1}, "M01 - KRAFTON-tf-ourfeat", "external", optional=True),
        # C1 grid: the input effect with the GBDT fixed (our representation vs KRAFTON's inputs) ...
        Formula("hgbcut_vs_hgb_kraftonfeat", {"HGB-cut": 1, "HGB-kraftonfeat": -1}, "HGB-cut - HGB-kraftonfeat", "external", optional=True),
        # ... and the learner effect with KRAFTON's inputs fixed (GBDT vs their transformer)
        Formula("hgb_kraftonfeat_vs_krafton_tf", {"HGB-kraftonfeat": 1, "KRAFTON-tf": -1}, "HGB-kraftonfeat - KRAFTON-tf", "external", optional=True),
        # Sagong's inputs with OUR learner: the clean input effect (same HGB settings, rows and labels) and the learner check
        Formula("hgbcut_vs_sagong_hgb", {"HGB-cut": 1, "SAGONG-hgb": -1}, "HGB-cut - SAGONG-hgb", "external", optional=True),
        Formula("sagong_hgb_vs_sagong_xgb", {"SAGONG-hgb": 1, "SAGONG-xgb": -1}, "SAGONG-hgb - SAGONG-xgb", "external", optional=True),
        # K3 ladder: their transformer on our features in three formulations (K2 Cox, K3a BCE, K3 BCE + phase baseline) and the residual
        Formula("k3a_vs_k2", {"KRAFTON-tf-ourfeat-bce": 1, "KRAFTON-tf-ourfeat": -1}, "KRAFTON-tf-ourfeat-bce - KRAFTON-tf-ourfeat", "external", optional=True),
        Formula("k3_vs_k3a", {"KRAFTON-tf-ourfeat-hazard": 1, "KRAFTON-tf-ourfeat-bce": -1}, "KRAFTON-tf-ourfeat-hazard - KRAFTON-tf-ourfeat-bce", "external", optional=True),
        Formula("k3_vs_k2", {"KRAFTON-tf-ourfeat-hazard": 1, "KRAFTON-tf-ourfeat": -1}, "KRAFTON-tf-ourfeat-hazard - KRAFTON-tf-ourfeat", "external", optional=True),
        Formula("m01_vs_k3", {"M01": 1, "KRAFTON-tf-ourfeat-hazard": -1}, "M01 - KRAFTON-tf-ourfeat-hazard", "external", optional=True),
        # the structure step of the decomposition: what the phase axis and the relation bias add to the team axis alone
        Formula("m11r_vs_m01", {"M11R": 1, "M01": -1}, "M11R - M01", "decomp"),
        # union (design doc 4.6, post hoc): what KRAFTON's kf_* features add to our representation, with the learner fixed ...
        Formula("hgb_union_vs_hgbcut", {"HGB-union": 1, "HGB-cut": -1}, "HGB-union - HGB-cut", "union", optional=True),
        Formula("m11r_union_vs_m11r", {"M11R-union": 1, "M11R": -1}, "M11R-union - M11R", "union", optional=True),
        # ... the union against the external baselines ...
        Formula("m11r_union_vs_krafton_tf", {"M11R-union": 1, "KRAFTON-tf": -1}, "M11R-union - KRAFTON-tf", "union", optional=True),
        Formula("m11r_union_vs_sagong_xgb", {"M11R-union": 1, "SAGONG-xgb": -1}, "M11R-union - SAGONG-xgb", "union", optional=True),
        # ... and what our representation adds to KRAFTON's inputs with the same GBDT
        Formula("hgb_union_vs_hgb_kraftonfeat", {"HGB-union": 1, "HGB-kraftonfeat": -1}, "HGB-union - HGB-kraftonfeat", "union", optional=True),
    )
}  # fmt: skip
PRIMARY: dict[str, tuple[str, str]] = {
    "P1": ("phase_main", "auc"),
    "P2": ("team_main", "auc"),
    "P3": ("team_main", "spearman"),
    "P4": ("rel_vs_shuf", "auc"),
    "P5": ("zone_nohab", "auc"),
}
HOLM_FAMILY: tuple[str, ...] = tuple(PRIMARY)
SEED_CONTRASTS: tuple[str, ...] = ("m11_m00", "m11r_m11", "m11rz_m11r", "rel_vs_shuf")

# row labels of the paper skeletons (paper/bigcomp2027/sections/05-experiments.tex)
PRIMARY_ROWS = {
    "P1": ("P1 (RQ1)", "phase-axis main effect", "AUC"),
    "P2": ("P2 (RQ2)", "team-axis main effect", "AUC"),
    "P3": ("P3 (RQ2)", "team-axis main effect", "Spearman"),
    "P4": ("P4 (RQ3)", "M11R $-$ M11R-shuf", "AUC"),
    "P5": ("P5 (RQ4)", "M11RZ-noHab $-$ M11R-noHab", "AUC"),
}
FOREST_TITLES = {
    "P1": "phase axis", "P2": "team axis", "P3": "team axis", "P4": "relation", "P5": "zone object",
}  # fmt: skip
EXPLORATORY_ROWS = [  # (RQ cell, contrast cell, formula | "kappa", stage-2 only)
    ("RQ1", "M10 $-$ M00", "m10_m00", False),
    ("RQ1", "M11 $-$ M01", "m11_m01", False),
    ("RQ2", "M01 $-$ M00", "m01_m00", False),
    ("RQ2", "M11 $-$ M10", "m11_m10", False),
    ("RQ1, RQ2", "interaction, $(\\mathrm{M11}-\\mathrm{M10})-(\\mathrm{M01}-\\mathrm{M00})$", "interaction", False),
    ("RQ3", "M11R $-$ M11", "m11r_m11", False),
    ("RQ3", "M11R $-$ M11R$-$prox", "m11r_noprox", False),
    ("RQ3", "M11R $-$ M11R$-$combat", "m11r_nocombat", False),
    ("RQ3", "M11R $-$ M11R$-$copress", "m11r_nocopress", False),
    ("RQ4", "M11RZ $-$ M11R", "m11rz_m11r", False),
    ("RQ4", "$\\kappa$ (recovery ratio)", "kappa", False),
    ("RQ4", "HGB-cut $-$ HGB-noHab (reference)", "hgb_cut_nohab", False),
    ("Player layer", "M11RZ-P $-$ M11RZ", "p_m11rz", True),
    ("RQ3 (teammates)", "M11RZ-P $-$ M11RZ-P$-$tm", "p_tm", True),
]  # fmt: skip
EXTERNAL_TABLE_REFERENCES = (("HGB-cut", "HGB-cut (reference)"), ("M11R", "M11R (reference)"))
EXTERNAL_CONTRAST_ROWS = [  # (label, formula, group, tag): exploratory, paired, pooled over the cohorts, no Holm
    ("M11R $-$ SAGONG-xgb", "m11r_vs_sagong_xgb", "sagong", ""),
    ("M11R $-$ SAGONG-rf", "m11r_vs_sagong_rf", "sagong", ""),
    ("M11R $-$ SAGONG-lr", "m11r_vs_sagong_lr", "sagong", ""),
    ("HGB-cut $-$ SAGONG-xgb", "hgbcut_vs_sagong_xgb", "sagong", "bundled"),
    ("HGB-cut $-$ SAGONG-hgb", "hgbcut_vs_sagong_hgb", "sagong", "input"),
    ("SAGONG-hgb $-$ SAGONG-xgb", "sagong_hgb_vs_sagong_xgb", "sagong", "learner check"),
    ("M11R $-$ KRAFTON-tf", "m11r_vs_krafton_tf", "krafton", ""),
    ("M11R $-$ KRAFTON-mlp", "m11r_vs_krafton_mlp", "krafton", ""),
    ("M11R $-$ KRAFTON-tf-ourfeat", "m11r_vs_krafton_tf_ourfeat", "krafton", ""),
    ("HGB-cut $-$ KRAFTON-tf", "hgbcut_vs_krafton_tf", "krafton", ""),
    ("KRAFTON-tf $-$ KRAFTON-mlp", "krafton_tf_mlp", "ablation", ""),
    ("KRAFTON-tf-ourfeat $-$ KRAFTON-tf", "krafton_ourfeat_input", "ablation", "bundled"),
    ("M01 $-$ KRAFTON-tf-ourfeat", "m01_vs_krafton_tf_ourfeat", "ablation", "bundled"),
    ("HGB-cut $-$ HGB-kraftonfeat", "hgbcut_vs_hgb_kraftonfeat", "grid", "input"),
    ("HGB-kraftonfeat $-$ KRAFTON-tf", "hgb_kraftonfeat_vs_krafton_tf", "grid", "bundled"),
    # the K3 ladder: their transformer on our features, K2 (Cox) -> K3a (BCE) -> K3 (BCE + phase baseline), and what is left for our architecture
    ("K3a $-$ K2", "k3a_vs_k2", "formulation", "objective"),
    ("K3 $-$ K3a", "k3_vs_k3a", "formulation", "phase baseline"),
    ("K3 $-$ K2", "k3_vs_k2", "formulation", "total"),
    ("M01 $-$ K3", "m01_vs_k3", "formulation", "residual"),
    # union (design doc 4.6, post hoc, exploratory): (+kf) = our representation plus KRAFTON's features vs ours, (+ours) = vs KRAFTON's inputs
    ("HGB-union $-$ HGB-cut", "hgb_union_vs_hgbcut", "union", "+kf"),
    ("M11R-union $-$ M11R", "m11r_union_vs_m11r", "union", "+kf"),
    ("M11R-union $-$ KRAFTON-tf", "m11r_union_vs_krafton_tf", "union", ""),
    ("M11R-union $-$ SAGONG-xgb", "m11r_union_vs_sagong_xgb", "union", ""),
    ("HGB-union $-$ HGB-kraftonfeat", "hgb_union_vs_hgb_kraftonfeat", "union", "+ours"),
]  # fmt: skip
# the C1 grid (paper focus 2026-10-06): rows = inputs, columns = learner, cell = the model of that input and learner
GRID_ROWS: tuple[tuple[str, str], ...] = (
    ("sagong", "Sagong features"), ("krafton", "KRAFTON features"), ("ours", "Our representation"), ("union", "Union (ours + KRAFTON)"),
)  # fmt: skip
GRID_OPTIONAL_ROWS: tuple[str, ...] = ("union",)  # post hoc (design doc 4.6): the row is shown only when one of its models was run
GRID_COLUMNS: tuple[tuple[str, str], ...] = (("gbdt", "GBDT"), ("krafton_tf", "KRAFTON transformer"), ("ptr", "PTR-Former"))
GRID_CELLS: dict[tuple[str, str], str] = {
    ("sagong", "gbdt"): "SAGONG-hgb",  # HGB-cut's learner on Sagong's inputs: every cell of the GBDT column is the same HistGradientBoosting
    ("krafton", "gbdt"): "HGB-kraftonfeat", ("krafton", "krafton_tf"): "KRAFTON-tf",
    ("ours", "gbdt"): "HGB-cut", ("ours", "krafton_tf"): "KRAFTON-tf-ourfeat", ("ours", "ptr"): "M11R",
    ("union", "gbdt"): "HGB-union", ("union", "ptr"): "M11R-union",  # no KRAFTON transformer on the union inputs was run
}  # fmt: skip
# the decomposition table (tab_decomp.tex): seven CLEAN single-factor steps, every row "to - from" (paired, DL-pooled, exploratory, no Holm);
# contrasts in which several factors differ (KRAFTON-tf -> K2, HGB-kraftonfeat - KRAFTON-tf, HGB-cut - SAGONG-xgb, M01 - K2) are not steps
DECOMP_STEPS: tuple[tuple[str, str, str, str], ...] = (  # (group, step, contrast cell, formula); every step changes ONE factor
    ("inputs", "Input vs Sagong (same HGB)", "HGB-cut $-$ SAGONG-hgb", "hgbcut_vs_sagong_hgb"),
    ("inputs", "Input vs KRAFTON (same HGB)", "HGB-cut $-$ HGB-kraftonfeat", "hgbcut_vs_hgb_kraftonfeat"),
    ("ladder", "Objective (Cox $\\to$ BCE)", "K3a $-$ K2", "k3a_vs_k2"),
    ("ladder", "Phase baseline", "K3 $-$ K3a", "k3_vs_k3a"),
    ("ladder", "Architecture (theirs $\\to$ ours)", "M01 $-$ K3", "m01_vs_k3"),
    ("structure", "Structure (phase axis, relations)", "M11R $-$ M01", "m11r_vs_m01"),
    ("extension", "Extended representation (+kf, HGB)", "HGB-union $-$ HGB-cut", "hgb_union_vs_hgbcut"),
)  # fmt: skip
ARMS_TABLE_ROWS = [  # (arm, row label); the stage-2 rows appear only when those arms were run
    *[(a, a) for a in MAIN_ARMS], ("M11RZ-P", "M11RZ-P"), ("M11RZ-P-tm", "M11RZ-P$-$tm"),
    *[(a, a) for a in HGB_ARMS],
]  # fmt: skip

POOLED_COLUMNS = [
    "contrast", "metric", "stratum", "kind", "primary_id", "exploratory", "theta", "se", "ci95_low", "ci95_high",
    "ci90_low", "ci90_high", "z", "p_one_sided", "p_two_sided", "p_holm", "tau2", "Q", "I2", "p_Q", "k", "k_total",
    "same_sign", "cohorts", "sesoi", "direction", "verdict", "verdict_basis",
]  # fmt: skip
CONTRAST_COLUMNS = [
    "cohort", "contrast", "metric", "stratum", "estimate", "se", "ci_low", "ci_high", "n_valid", "n_rows", "n_matches",
]  # fmt: skip
ARM_METRIC_COLUMNS = ["cohort", "arm", "seed", "stratum", "metric", "value", "n_rows", "n_matches"]
KAPPA_COLUMNS = [
    "level", "stratum", "cohort", "kappa", "ci_low", "ci_high", "numerator", "num_se", "denominator", "den_se",
    "den_ci95_low", "den_ci95_high", "estimable", "reportable", "k", "n_valid", "den_nonpositive_frac", "note",
]  # fmt: skip
SEED_COLUMNS = [
    "kind", "cohort", "name", "metric", "n_seeds", "seeds", "values", "mean", "sd", "min", "max", "sesoi",
    "sd_over_sesoi",
]  # fmt: skip
ISSUE_COLUMNS = ["level", "cohort", "arm", "contrast", "message"]
INVENTORY_COLUMNS = [
    "cohort", "arm", "seed", "kind", "role", "labels", "config_hash", "n_rows", "n_matches", "aligned", "pred_test",
]  # fmt: skip


# --------------------------------------------------------------------------- pure helpers
def arm_rank(arm: str) -> tuple[int, str]:
    return (ARM_ORDER.index(arm), arm) if arm in ARM_ORDER else (len(ARM_ORDER), arm)


def task_seed(cohort: str, stratum: str, seed: int) -> int:
    """Bootstrap seed of one (cohort, stratum): stable across processes and runs (no salted ``hash``)."""
    return (zlib.crc32(f"{cohort}|{stratum}".encode("utf-8")) ^ int(seed)) & 0xFFFFFFFF


def holm(pvals: Mapping[str, float], family: int | None = None) -> dict[str, float]:
    """Holm step-down adjustment.  The family has at least ``len(PRIMARY)`` members: a test that is absent counts as
    p = 1 and so never shortens the multiplier; an undefined p stays undefined and takes no rank."""
    m = max(len(pvals), len(HOLM_FAMILY) if family is None else family)
    out = {k: float("nan") for k in pvals}
    valid = sorted(((float(p), k) for k, p in pvals.items() if p is not None and np.isfinite(p)), key=lambda t: (t[0], t[1]))
    running = 0.0
    for rank, (p, k) in enumerate(valid):
        running = max(running, min(1.0, (m - rank) * p))
        out[k] = running
    return out


def verdict(theta: float, ci90: tuple[float, float], p_adj: float, sesoi: float | None, direction: int = 1) -> str:
    """Design v2.2: ``gain`` = adjusted p < 0.05 and estimate >= SESOI; ``equivalent`` = 90% interval within
    +-SESOI; else ``inconclusive``.  Metrics where lower is better flip the sign first; no SESOI -> ``n/a``."""
    lo, hi = ci90
    if sesoi is None or not np.isfinite(sesoi) or not all(np.isfinite(v) for v in (theta, lo, hi)):
        return "n/a"
    if direction < 0:
        theta, lo, hi = -theta, -hi, -lo
    if np.isfinite(p_adj) and p_adj < ALPHA and theta >= sesoi:
        return "gain"
    if lo >= -sesoi and hi <= sesoi:
        return "equivalent"
    return "inconclusive"


def contrast_from_samples(point: Mapping, samples: Mapping, coefs: Mapping[str, float], metric: str, ci: float = 0.95) -> dict:
    """``estimate`` / ``se`` / percentile interval of ``sum_k c_k metric(arm_k)`` from one shared bootstrap."""
    est = sum(c * point[arm][metric] for arm, c in coefs.items())
    draws = sum(c * samples[arm][metric] for arm, c in coefs.items())
    return M.summarise_draws(est, draws, ci)


def _tex_num(x: float, digits: int) -> str:
    s = f"{x:.{digits}f}"
    return "$-$" + s[1:] if s.startswith("-") else s


def fmt_delta(est: float, lo: float, hi: float) -> str:
    if not all(math.isfinite(v) for v in (est, lo, hi)):
        return "---"
    e = _tex_num(est, 3)
    return f"{e if e.startswith('$') else '+' + e} [{_tex_num(lo, 3)}, {_tex_num(hi, 3)}]"


def fmt_p(p: float) -> str:
    if not math.isfinite(p):
        return "---"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def fmt_tau2(t: float) -> str:
    if not math.isfinite(t):
        return "---"
    if t == 0:
        return "0"
    if t < 1e-3:
        mant, exp = f"{t:.1e}".split("e")
        return f"${mant}\\!\\times\\!10^{{{int(exp)}}}$"
    return f"{t:.4f}"


def fmt_value(x: float) -> str:
    return "---" if not math.isfinite(x) else _tex_num(x, 3)


def fmt_kappa(kappa: float, lo: float, hi: float) -> str:
    return f"{_tex_num(kappa, 2)} [{_tex_num(lo, 2)}, {_tex_num(hi, 2)}]"


_LABEL_TOKENS = {"tpp": "TPP", "fpp": "FPP", "solo": "solo", "duo": "duo", "squad": "squad"}


def cohort_label(name: str) -> str:
    return " ".join(_LABEL_TOKENS.get(t, t.capitalize()) for t in name.split("_"))


def _f(x) -> float:
    return float("nan") if x is None else float(x)


def _frame(rows: Sequence[Mapping], columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(columns))


# --------------------------------------------------------------------------- scanning the run directory
@dataclass(frozen=True)
class Run:
    cohort: str
    arm: str
    seed: int
    pred: Path
    labels: str | None
    config_hash: str | None
    n_train: int | None = None  # training matches of the cohort (metrics.json n_matches.train), None if not recorded

    @property
    def kind(self) -> str:
        if self.arm in EXTERNAL_ARMS or self.arm.startswith(EXTERNAL_PREFIXES):
            return "external"
        return "hgb" if self.arm.startswith("HGB") else "neural"

    @property
    def role(self) -> str:
        return "main" if self.seed == MAIN_SEED else "extra_seed"


def scan_runs(runs_dir: str | Path, cohorts: Iterable[str] | None = None) -> tuple[list[Run], list[tuple]]:
    """Finished runs (``pred_test.csv`` and ``metrics.json`` both present) and issue records for the others."""
    root = Path(runs_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"runs directory not found: {root}")
    wanted = None if cohorts is None else set(cohorts)
    runs: list[Run] = []
    issues: list[tuple] = []
    for cdir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "logs" and not p.name.startswith(".")):
        if wanted is not None and cdir.name not in wanted:
            continue
        for adir in sorted(p for p in cdir.iterdir() if p.is_dir()):
            for sdir in sorted(p for p in adir.iterdir() if p.is_dir()):
                m = re.fullmatch(r"seed(\d+)", sdir.name)
                if not m:
                    continue
                seed, pred, meta_path = int(m.group(1)), sdir / "pred_test.csv", sdir / "metrics.json"
                where = f"{cdir.name} {adir.name} seed {seed}"
                if not pred.is_file() or not meta_path.is_file():
                    missing = [p.name for p in (pred, meta_path) if not p.is_file()]
                    issues.append(("warning", cdir.name, adir.name, "", f"{where}: run not usable, missing {' and '.join(missing)}; ignored"))
                    continue
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (ValueError, OSError) as exc:
                    issues.append(("warning", cdir.name, adir.name, "", f"{where}: metrics.json unreadable ({exc}); run ignored"))
                    continue
                n_train = (meta.get("n_matches") or {}).get("train") if isinstance(meta.get("n_matches"), dict) else None
                runs.append(Run(cdir.name, adir.name, seed, pred, meta.get("labels"), meta.get("config_hash"),
                                int(n_train) if n_train is not None else None))  # fmt: skip
    if wanted is not None:
        for name in sorted(wanted - {r.cohort for r in runs}):
            issues.append(("warning", name, "", "", f"{name}: no finished runs found"))
    return runs, issues


def check_labels(runs: Sequence[Run]) -> str:
    """All runs must use one label choice; returns it ('unknown' if no run records it)."""
    known = sorted({r.labels for r in runs if r.labels})
    if len(known) > 1:
        by = {lab: sorted({f"{r.cohort}/{r.arm}/seed{r.seed}" for r in runs if r.labels == lab})[:3] for lab in known}
        raise ValueError(f"the runs use different label choices {known} (labels in metrics.json), e.g. {by}; "
                         "a result matrix needs one choice")  # fmt: skip
    return known[0] if known else "unknown"


# --------------------------------------------------------------------------- worker tasks (module level: picklable)
def _fingerprint(df: pd.DataFrame) -> str:
    """Identity of the rows and labels of a prediction frame (what the paired bootstrap needs to be equal)."""
    h = hashlib.sha1()
    h.update("\x1f".join(df["match_id"].tolist()).encode("utf-8") + b"\0")
    h.update(df["team_id"].to_numpy(np.int64).tobytes() + df["phase"].to_numpy(np.int64).tobytes())
    for c in ("y", "top5", "final_rank_norm"):
        h.update(np.round(df[c].to_numpy(np.float64), 6).tobytes())
    h.update("\x1f".join(df["match_type"].tolist()).encode("utf-8"))
    return h.hexdigest()


def _prep_task(spec: dict) -> dict:
    """Read every prediction file of a cohort once: row fingerprint, sizes and match types."""
    files, errors = {}, {}
    for arm, seed, path in spec["runs"]:
        try:
            df = M.read_predictions(path)
        except (ValueError, OSError, KeyError) as exc:
            errors[(arm, seed)] = str(exc)
            continue
        mt = df.groupby("match_type")["match_id"].nunique()
        files[(arm, seed)] = {
            "fp": _fingerprint(df), "n_rows": int(len(df)), "n_matches": int(df["match_id"].nunique()),
            "types": {str(k): int(v) for k, v in mt.items()},
        }  # fmt: skip
    return {"cohort": spec["cohort"], "files": files, "errors": errors}


def _stratum_task(spec: dict) -> dict:
    """Point metrics of every run and the shared-resample bootstrap of the seed-42 arms for one (cohort, stratum)."""
    cohort, stratum = spec["cohort"], spec["stratum"]
    out: dict = {"cohort": cohort, "stratum": stratum, "skipped": None, "arm_rows": [], "contrast_rows": [], "kappa": None}
    frames: dict[tuple[str, int], pd.DataFrame] = {}
    for arm, seed, path in spec["runs"]:
        df = M.read_predictions(path)
        if stratum.startswith(PHASE_PREFIX):  # a phase group: the rows of its cut phases (matches stay whole clusters)
            df = df[df["phase"].isin(spec["phases"])].reset_index(drop=True)
        elif stratum != STRATUM_ALL:
            df = df[df["match_type"] == stratum].reset_index(drop=True)
        frames[(arm, seed)] = df
    if not frames:
        out["skipped"] = "no runs"
        return out
    first = next(iter(frames.values()))
    n_matches, n_rows = int(first["match_id"].nunique()), int(len(first))
    if n_matches < spec["min_matches"]:
        out["skipped"] = f"{n_matches} test matches (fewer than {spec['min_matches']})"
        return out
    for (arm, seed), df in frames.items():
        suite = M.MetricSuite(df, RANK_COL)
        vals = suite.evaluate(np.ones(suite.n_matches), METRICS)
        for metric in METRICS:
            out["arm_rows"].append(
                {"cohort": cohort, "arm": arm, "seed": seed, "stratum": stratum, "metric": metric, "value": float(vals[metric]),
                 "n_rows": suite.n_rows, "n_matches": suite.n_matches}  # fmt: skip
            )
    main = {arm: df for (arm, seed), df in frames.items() if seed == MAIN_SEED}
    if len(main) < 2:
        return out
    _, point, samples = M.bootstrap_arms(
        main, rank_col=RANK_COL, metrics=METRICS, n_boot=spec["n_boot"], seed=task_seed(cohort, stratum, spec["seed"])
    )
    for f in FORMULAS.values():
        if not f.pooled or not all(a in samples for a in f.coefs):
            continue
        for metric in METRICS:
            c = contrast_from_samples(point, samples, f.coefs, metric)
            out["contrast_rows"].append(
                {"cohort": cohort, "contrast": f.name, "metric": metric, "stratum": stratum, "estimate": c["estimate"],
                 "se": c["se"], "ci_low": c["ci_low"], "ci_high": c["ci_high"], "n_valid": c["n_valid"], "n_rows": n_rows,
                 "n_matches": n_matches}  # fmt: skip
            )
    k = KAPPA_ARMS
    if all(a in samples for a in set(k.values())):
        num = contrast_from_samples(point, samples, FORMULAS["zone_nohab"].coefs, "auc")
        den = contrast_from_samples(point, samples, FORMULAS["hab_value"].coefs, "auc")
        out["kappa"] = {
            "num": num["estimate"], "num_se": num["se"], "den": den["estimate"], "den_se": den["se"],
            "num_draws": samples[k["num_a"]]["auc"] - samples[k["num_b"]]["auc"],
            "den_draws": samples[k["den_a"]]["auc"] - samples[k["den_b"]]["auc"],
        }  # fmt: skip
    return out


class _Parallel:
    """A process pool (spawn) when ``workers > 1``; plain in-process execution otherwise.  Results keep input order."""

    def __init__(self, workers: int) -> None:
        self.workers = max(1, int(workers))
        self.pool: cf.ProcessPoolExecutor | None = None

    def __enter__(self) -> "_Parallel":
        if self.workers > 1:
            self.pool = cf.ProcessPoolExecutor(max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"))
        return self

    def map(self, fn: Callable, items: Sequence, cost: Callable | None = None) -> list:
        items = list(items)
        if self.pool is None or len(items) <= 1:
            return [fn(x) for x in items]
        order = sorted(range(len(items)), key=(lambda i: -cost(items[i])) if cost else (lambda i: i))
        futures = {i: self.pool.submit(fn, items[i]) for i in order}  # biggest first: better packing
        return [futures[i].result() for i in range(len(items))]

    def __exit__(self, *exc) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)


# --------------------------------------------------------------------------- pooling
def _index_rows(df: pd.DataFrame) -> dict:
    return {(r.contrast, r.metric, r.stratum): r for r in df.itertuples(index=False)}


def _stratum_order(strata: Iterable[str]) -> list[str]:
    s = set(strata)
    return ([STRATUM_ALL] if STRATUM_ALL in s else []) + sorted(s - {STRATUM_ALL})


def _display_strata(strata: Iterable[str]) -> list[str]:
    """all, official, competitive, then any other match type (tables only; files keep the sorted order)."""
    s = set(strata)
    first = [x for x in (STRATUM_ALL, "official", "competitive") if x in s]
    return first + sorted(s - set(first))


def pool_contrasts(contrasts: pd.DataFrame, n_cohorts: int) -> pd.DataFrame:
    """DerSimonian-Laird pooling per (contrast, metric, stratum); Holm over P1-P5 (stratum ``all``) and verdicts."""
    rows = []
    for (name, metric, stratum), g in contrasts.groupby(["contrast", "metric", "stratum"], sort=False):
        ok = g[np.isfinite(g["estimate"]) & np.isfinite(g["se"]) & (g["se"] > 0)].sort_values("cohort")
        if ok.empty:
            continue
        dl = M.dersimonian_laird(ok["estimate"].to_numpy(), ok["se"].to_numpy())
        direction = DIRECTION[metric]
        pid = next((p for p, key in PRIMARY.items() if key == (name, metric)), "")
        is_primary = bool(pid) and stratum == STRATUM_ALL
        rows.append({
            "contrast": name, "metric": metric, "stratum": stratum, "kind": "primary" if is_primary else "secondary",
            "primary_id": pid, "exploratory": not is_primary, "theta": dl["theta"], "se": dl["se"],
            "ci95_low": dl["ci95"][0], "ci95_high": dl["ci95"][1], "ci90_low": dl["ci90"][0], "ci90_high": dl["ci90"][1],
            "z": dl["z"], "p_one_sided": float(stats.norm.sf(direction * dl["z"])), "p_two_sided": dl["p"],
            "p_holm": float("nan"), "tau2": dl["tau2"], "Q": dl["Q"], "I2": dl["I2"], "p_Q": dl["p_Q"], "k": dl["k"],
            "k_total": int(n_cohorts), "same_sign": int(np.sum(np.sign(ok["estimate"].to_numpy()) == np.sign(dl["theta"]))),
            "cohorts": ";".join(ok["cohort"]), "sesoi": SESOI.get(metric, float("nan")), "direction": direction,
            "verdict": "", "verdict_basis": "",
        })  # fmt: skip
    df = _frame(rows, POOLED_COLUMNS)
    if df.empty:
        return df
    prim = df[df["kind"] == "primary"]
    adj = holm({r.primary_id: r.p_one_sided for r in prim.itertuples()})
    for idx, r in prim.iterrows():
        df.loc[idx, "p_holm"] = adj[r["primary_id"]]
    holm_basis = df["kind"] == "primary"
    for idx, r in df.iterrows():
        p_used = r["p_holm"] if holm_basis[idx] else r["p_one_sided"]
        df.loc[idx, "verdict"] = verdict(
            r["theta"], (r["ci90_low"], r["ci90_high"]), p_used, SESOI.get(r["metric"]), int(r["direction"])
        )
        df.loc[idx, "verdict_basis"] = "holm" if holm_basis[idx] else "unadjusted"
    fo = list(FORMULAS)
    key = df.apply(lambda r: (fo.index(r["contrast"]), METRICS.index(r["metric"])), axis=1)
    strata = _stratum_order(df["stratum"])
    order = sorted(range(len(df)), key=lambda i: (key.iloc[i][0], key.iloc[i][1], strata.index(df["stratum"].iloc[i])))
    return df.iloc[order].reset_index(drop=True)


def pool_kappa(inputs: Mapping[str, list[dict]], n_cohorts: int, n_boot: int) -> pd.DataFrame:
    """Recovery ratio per stratum: descriptive per-cohort values and the pooled, gated estimate."""
    rows = []
    gate = SESOI["auc"]
    for stratum in _stratum_order(inputs):
        items = sorted(inputs[stratum], key=lambda d: d["cohort"])
        good = [d for d in items if all(np.isfinite([d["num"], d["den"]])) and d["num_se"] > 0 and d["den_se"] > 0
                and np.isfinite(d["num_se"]) and np.isfinite(d["den_se"])]  # fmt: skip
        pooled = {
            "level": "pooled", "stratum": stratum, "cohort": "", "kappa": float("nan"), "ci_low": float("nan"),
            "ci_high": float("nan"), "numerator": float("nan"), "num_se": float("nan"), "denominator": float("nan"),
            "den_se": float("nan"), "den_ci95_low": float("nan"), "den_ci95_high": float("nan"), "estimable": False,
            "reportable": False, "k": len(good), "n_valid": 0, "den_nonpositive_frac": float("nan"), "note": "",
        }  # fmt: skip
        estimable = False
        if good:
            num = M.dersimonian_laird([d["num"] for d in good], [d["num_se"] for d in good])
            den = M.dersimonian_laird([d["den"] for d in good], [d["den_se"] for d in good])
            pooled.update(numerator=num["theta"], num_se=num["se"], denominator=den["theta"], den_se=den["se"],
                          den_ci95_low=den["ci95"][0], den_ci95_high=den["ci95"][1])  # fmt: skip
            estimable = bool(den["ci95"][0] >= gate)
            if estimable:
                wn = 1.0 / (np.array([d["num_se"] for d in good]) ** 2 + num["tau2"])
                wd = 1.0 / (np.array([d["den_se"] for d in good]) ** 2 + den["tau2"])
                nd = (wn[:, None] * np.stack([d["num_draws"] for d in good])).sum(0) / wn.sum()
                dd = (wd[:, None] * np.stack([d["den_draws"] for d in good])).sum(0) / wd.sum()
                fin = np.isfinite(nd) & np.isfinite(dd)
                pos = fin & (dd > 0)
                draws = nd[pos] / dd[pos]
                pooled.update(
                    kappa=num["theta"] / den["theta"], n_valid=int(pos.sum()),
                    den_nonpositive_frac=float(np.mean(dd[fin] <= 0)) if fin.any() else float("nan"),
                    ci_low=float(np.percentile(draws, 2.5)) if draws.size else float("nan"),
                    ci_high=float(np.percentile(draws, 97.5)) if draws.size else float("nan"),
                )  # fmt: skip
                pooled["note"] = f"pooled over {len(good)} cohorts; interval: percentile of the pooled ratio over {n_boot} resamples"
            else:
                pooled["note"] = (f"not estimable: the pooled 95% interval of the denominator (M11R - M11R-noHab) starts at "
                                  f"{den['ci95'][0]:.4f}, below the SESOI {gate}")  # fmt: skip
        else:
            pooled["note"] = "not estimable: no cohort has a valid numerator and denominator"
        pooled["estimable"] = estimable
        pooled["reportable"] = estimable
        rows.append(pooled)
        for d in items:
            den_draws = np.asarray(d["den_draws"])
            ok = np.isfinite(d["num_draws"]) & np.isfinite(den_draws) & (den_draws != 0)
            ratio = np.asarray(d["num_draws"])[ok] / den_draws[ok]
            rows.append({
                "level": "cohort", "stratum": stratum, "cohort": d["cohort"],
                "kappa": d["num"] / d["den"] if d["den"] != 0 else float("nan"),
                "ci_low": float(np.percentile(ratio, 2.5)) if ratio.size else float("nan"),
                "ci_high": float(np.percentile(ratio, 97.5)) if ratio.size else float("nan"),
                "numerator": d["num"], "num_se": d["num_se"], "denominator": d["den"], "den_se": d["den_se"],
                "den_ci95_low": float("nan"), "den_ci95_high": float("nan"), "estimable": estimable,
                "reportable": estimable, "k": 1, "n_valid": int(ok.sum()),
                "den_nonpositive_frac": float(np.mean(den_draws[np.isfinite(den_draws)] <= 0)) if np.isfinite(den_draws).any() else float("nan"),
                "note": "descriptive (per cohort)" if estimable else "withheld: the pooled denominator does not qualify",
            })  # fmt: skip
    return _frame(rows, KAPPA_COLUMNS)


# --------------------------------------------------------------------------- arm tables and seed stability
def arm_means(arm_metrics: pd.DataFrame) -> pd.DataFrame:
    """Unweighted mean over cohorts of the seed-42 point metrics, per (arm, metric, stratum)."""
    sub = arm_metrics[(arm_metrics["seed"] == MAIN_SEED) & np.isfinite(arm_metrics["value"])]
    rows = []
    for (arm, metric, stratum), g in sub.groupby(["arm", "metric", "stratum"], sort=False):
        v = g["value"].to_numpy()
        rows.append({"arm": arm, "metric": metric, "stratum": stratum, "mean": float(v.mean()),
                     "sd_cohorts": float(v.std(ddof=1)) if v.size > 1 else float("nan"), "n_cohorts": int(v.size)})  # fmt: skip
    df = _frame(rows, ["arm", "metric", "stratum", "mean", "sd_cohorts", "n_cohorts"])
    if df.empty:
        return df
    strata = _stratum_order(df["stratum"])
    order = sorted(range(len(df)), key=lambda i: (arm_rank(df["arm"].iloc[i]), strata.index(df["stratum"].iloc[i]), METRICS.index(df["metric"].iloc[i])))
    return df.iloc[order].reset_index(drop=True)


def seed_stability(arm_metrics: pd.DataFrame) -> pd.DataFrame:
    """Seed-to-seed SD per arm and per contrast (stratum ``all``) for cohorts that ran more than one seed."""
    sub = arm_metrics[(arm_metrics["stratum"] == STRATUM_ALL) & np.isfinite(arm_metrics["value"])]
    vals: dict[tuple[str, str, str], dict[int, float]] = {}
    for r in sub.itertuples(index=False):
        vals.setdefault((r.cohort, r.arm, r.metric), {})[int(r.seed)] = float(r.value)
    rows = []

    def add(kind: str, cohort: str, name: str, metric: str, per_seed: dict[int, float]) -> None:
        seeds = sorted(per_seed)
        v = np.array([per_seed[s] for s in seeds])
        sd = float(v.std(ddof=1))
        sesoi = SESOI.get(metric, float("nan"))
        rows.append({
            "kind": kind, "cohort": cohort, "name": name, "metric": metric, "n_seeds": len(seeds),
            "seeds": json.dumps(seeds), "values": json.dumps([float(x) for x in v]), "mean": float(v.mean()), "sd": sd,
            "min": float(v.min()), "max": float(v.max()), "sesoi": sesoi, "sd_over_sesoi": sd / sesoi if np.isfinite(sesoi) else float("nan"),
        })  # fmt: skip

    cohorts = sorted({k[0] for k in vals})
    for cohort in cohorts:
        arms = sorted({k[1] for k in vals if k[0] == cohort}, key=arm_rank)
        for arm in arms:
            for metric in METRICS:
                per_seed = vals.get((cohort, arm, metric), {})
                if len(per_seed) >= 2:
                    add("arm", cohort, arm, metric, per_seed)
        for name in SEED_CONTRASTS:
            coefs = FORMULAS[name].coefs
            for metric in METRICS:
                have = [vals.get((cohort, a, metric), {}) for a in coefs]
                common = sorted(set.intersection(*[set(h) for h in have])) if have else []
                if len(common) >= 2:
                    add("contrast", cohort, name, metric, {s: sum(c * vals[(cohort, a, metric)][s] for a, c in coefs.items()) for s in common})
    return _frame(rows, SEED_COLUMNS)


# --------------------------------------------------------------------------- LaTeX tables
def _tex_comment(name: str, extra: Sequence[str] = ()) -> str:
    lines = [
        f"% generated by model_stgcn.ptr.aggregate ({name}); do not edit by hand.",
        "% Include it inside the table environment with \\input{...} in place of the tabular of the skeleton.",
        *[f"% {e}" for e in extra],
    ]
    return "\n".join(lines) + "\n"


def _lookup(idx: Mapping, name: str, metric: str, stratum: str = STRATUM_ALL):
    return idx.get((name, metric, stratum))


def _delta_cell(row) -> str:
    return "---" if row is None else fmt_delta(row.theta, row.ci95_low, row.ci95_high)


def render_primary_table(pooled: pd.DataFrame) -> str:
    idx = _index_rows(pooled)
    rows = {pid: _lookup(idx, *PRIMARY[pid]) for pid in PRIMARY}
    ks = {int(r.k) for r in rows.values() if r is not None}
    uniform = len(ks) == 1
    same_head = f"Same sign /${next(iter(ks))}$" if uniform else "Same sign"
    out = [
        _tex_comment("tab_primary", ["one-sided pooled z tests, Holm over P1-P5 only; the 95% intervals are unadjusted"]),
        "\\begin{tabular}{@{}l l l c c c c c@{}}",
        "\\toprule",
        f" & Contrast & Metric & $\\Delta$ [95\\% CI] & {same_head} & $\\tau^2$ & Holm $p$ & Verdict \\\\",
        "\\midrule",
    ]
    for pid, r in rows.items():
        label, contrast, metric = PRIMARY_ROWS[pid]
        if r is None:
            cells = ["---"] * 5
        else:
            same = str(int(r.same_sign)) if uniform else f"{int(r.same_sign)}/{int(r.k)}"
            cells = [_delta_cell(r), same, fmt_tau2(r.tau2), fmt_p(r.p_holm), r.verdict]
        out.append(" & ".join([label, contrast, metric, *cells]) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(out)


def render_exploratory_table(pooled: pd.DataFrame, kappa: pd.DataFrame) -> str:
    idx = _index_rows(pooled)
    out = [
        _tex_comment("tab_exploratory", ["no Holm correction; (k cohorts) marks rows pooled over fewer cohorts than the rest"]),
        "\\begin{tabular}{@{}l l c c c c@{}}",
        "\\toprule",
        "RQ & Contrast & $\\Delta$AUC & $\\Delta$Spearman & $\\Delta$Top-5 AUC & $\\Delta$C-index \\\\",
        "\\midrule",
    ]
    k_total = int(pooled["k_total"].max()) if len(pooled) else 0
    for rq, contrast, name, stage2 in EXPLORATORY_ROWS:
        if name == "kappa":
            krow = kappa[(kappa["level"] == "pooled") & (kappa["stratum"] == STRATUM_ALL)] if len(kappa) else kappa
            if krow.empty:
                cell = "---"
            else:
                r = krow.iloc[0]
                cell = fmt_kappa(r["kappa"], r["ci_low"], r["ci_high"]) if bool(r["estimable"]) else "not estimable"
            out.append(" & ".join([rq, contrast, cell, "---", "---", "---"]) + " \\\\")
            continue
        auc = _lookup(idx, name, "auc")
        if stage2 and all(_lookup(idx, name, m) is None for m in METRICS):
            continue  # stage-2 arms were not run: no row
        label = contrast
        if auc is not None and k_total and int(auc.k) < k_total and not name.startswith("m11r_no"):
            label += f" ({int(auc.k)} cohorts)"
        elif name.startswith("m11r_no") and auc is not None:
            label += f" ({int(auc.k)} cohorts)"
        cells = [_delta_cell(_lookup(idx, name, m)) for m in ("auc", "spearman", "top5_auc", "cindex")]
        out.append(" & ".join([rq, label, *cells]) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(out)


def render_arms_table(means: pd.DataFrame) -> str:
    sel = means[means["stratum"] == STRATUM_ALL] if len(means) else means
    lookup = {(r.arm, r.metric): r.mean for r in sel.itertuples(index=False)}
    out = [
        _tex_comment("tab_arms_metrics", ["unweighted mean over the test cohorts, seed 42 (descriptive)"]),
        "\\begin{tabular}{@{}lccccccc@{}}",
        "\\toprule",
        " & \\multicolumn{4}{c}{Hazard} & \\multicolumn{3}{c}{Relative standing} \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(l){6-8}",
        "Arm & AUC & AP & NLL & Brier & Spearman & Top-5 AUC & C-index \\\\",
        "\\midrule",
    ]
    for arm, label in ARMS_TABLE_ROWS:
        have = [lookup.get((arm, m), float("nan")) for m in TABLE_METRICS]
        if arm in STAGE2_ARMS and not any(math.isfinite(v) for v in have):
            continue
        out.append(" & ".join([label, *[fmt_value(v) for v in have]]) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(out)


def render_external_table(means: pd.DataFrame) -> str:
    """Arm-level metrics of the external baselines next to the references M11R and HGB-cut (unweighted mean over the
    cohorts that have the arm, seed 42, descriptive).  Arms that were not run are left out."""
    sel = means[means["stratum"] == STRATUM_ALL] if len(means) else means
    lookup = {(r.arm, r.metric): r.mean for r in sel.itertuples(index=False)}
    n_coh = {r.arm: int(r.n_cohorts) for r in sel[sel["metric"] == "auc"].itertuples(index=False)} if len(sel) else {}
    top = max(n_coh.values()) if n_coh else 0
    out = [
        _tex_comment("tab_external", [
            "external baselines (design doc 4.3, exploratory): unweighted mean over the cohorts that have the model, seed 42;",
            "a '(k cohorts)' label means fewer cohorts than the references, so its row is not comparable with theirs",
        ]),
        "\\begin{tabular}{@{}lccccccc@{}}",
        "\\toprule",
        " & \\multicolumn{4}{c}{Hazard} & \\multicolumn{3}{c}{Relative standing} \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(l){6-8}",
        "Model & AUC & AP & NLL & Brier & Spearman & Top-5 AUC & C-index \\\\",
        "\\midrule",
    ]

    def row(arm: str, label: str) -> str | None:
        have = [lookup.get((arm, m), float("nan")) for m in TABLE_METRICS]
        if not any(math.isfinite(v) for v in have):
            return None
        suffix = f" ({n_coh[arm]} cohorts)" if arm in n_coh and n_coh[arm] < top else ""
        return " & ".join([label + suffix, *[fmt_value(v) for v in have]]) + " \\\\"

    for arm, label in EXTERNAL_TABLE_REFERENCES:
        line = row(arm, label)
        if line:
            out.append(line)
    out.append("\\midrule")
    for arm in EXTERNAL_ARMS:
        line = row(arm, arm)
        if line:
            out.append(line)
    out += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(out)


def render_grid_table(means: pd.DataFrame) -> str:
    """The C1 grid: rows = inputs (Sagong / KRAFTON / ours), columns = learner (GBDT / KRAFTON transformer / PTR-Former); each
    cell is the unweighted mean test AUC (seed 42, all matches) over the cohorts that have that model, with the number of cohorts.
    A cell with no model (the grid has none for it, or it was not run) is ``---``: a cell is one model and no other model stands in
    for it (Sagong's own XGBoost, SAGONG-xgb, is in ``tab_external.tex``).  The GBDT column is HistGradientBoosting with the HGB-cut
    settings in every row, so its differences are single-factor input effects.  Same column: input effect; same row: learner effect (the
    paired, pooled differences are in ``tab_external_contrasts.tex``).  Means over different cohort sets are not comparable."""
    sel = means[(means["stratum"] == STRATUM_ALL) & (means["metric"] == "auc")] if len(means) else means
    info = {r.arm: (r.mean, int(r.n_cohorts)) for r in sel.itertuples(index=False)}
    cells = ", ".join(f"{GRID_ROWS_DICT[r]} x {GRID_COLUMNS_DICT[c]}: {a}" for (r, c), a in GRID_CELLS.items())
    out = [
        _tex_comment("tab_grid", [
            "C1 grid, mean test AUC over cohorts (seed 42, all matches), (k coh.) = number of cohorts averaged; --- = no model in the grid",
            "same column: input effect, same row: learner effect; the paired, pooled differences are in tab_external_contrasts.tex",
            "GBDT column = HistGradientBoosting with the HGB-cut settings in every row (a clean input effect down the column); Sagong's own XGBoost is in tab_external.tex",
            "cells: " + cells,
        ]),
        "\\begin{tabular}{@{}l c c c@{}}",
        "\\toprule",
        "Inputs $\\backslash$ learner & " + " & ".join(label for _, label in GRID_COLUMNS) + " \\\\",
        "\\midrule",
    ]
    for rkey, rlabel in GRID_ROWS:
        if rkey in GRID_OPTIONAL_ROWS and not any(
            (a := GRID_CELLS.get((rkey, c))) is not None and a in info and math.isfinite(info[a][0]) for c, _ in GRID_COLUMNS
        ):
            continue  # an experiment that was not run gets no row of dashes
        row = []
        for ckey, _ in GRID_COLUMNS:
            arm = GRID_CELLS.get((rkey, ckey))
            if arm is None or arm not in info or not math.isfinite(info[arm][0]):
                row.append("---")
            else:
                mean, k = info[arm]
                row.append(f"{fmt_value(mean)} ({k} coh.)")
        out.append(" & ".join([rlabel, *row]) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(out)


GRID_ROWS_DICT = dict(GRID_ROWS)
GRID_COLUMNS_DICT = dict(GRID_COLUMNS)


def render_external_contrasts_table(pooled: pd.DataFrame) -> str:
    """Paired, DL-pooled exploratory contrasts between our models and the external baselines (no Holm correction).

    Three blocks (a rule between them): our models against SAGONG, against KRAFTON, and the KRAFTON ablations on our data:
    the input effect (their transformer on our features minus on theirs: backbone and loss fixed) and the model and loss
    effect (our M01 minus their transformer, both on our features)."""
    idx = _index_rows(pooled)
    out = [
        _tex_comment("tab_external_contrasts", [
            "delta [95% CI], paired match-level bootstrap on identical rows per cohort, DerSimonian-Laird pooled; exploratory, no Holm",
            "(k cohorts) marks a contrast pooled over fewer cohorts than the analysis has",
            "tags: (input) = a clean single-factor input effect (same HistGradientBoosting settings, rows and labels): HGB-cut - SAGONG-hgb",
            "(our representation vs Sagong's 12 inputs) and HGB-cut - HGB-kraftonfeat (ours vs KRAFTON's reconstructed inputs);",
            "(learner check) = SAGONG-hgb - SAGONG-xgb: HistGradientBoosting vs XGBoost on Sagong's inputs;",
            "(bundled) = more than one factor differs between the two models, not a single-factor effect (the bundled contrasts stay out of tab_decomp.tex):",
            "KRAFTON-tf-ourfeat - KRAFTON-tf, HGB-kraftonfeat - KRAFTON-tf, HGB-cut - SAGONG-xgb (input and learner at once) and",
            "M01 - KRAFTON-tf-ourfeat (= objective + phase baseline + residual of the K ladder below);",
            "K3 ladder (their transformer on our features): K2 = KRAFTON-tf-ourfeat (Cox loss), K3a = -bce (BCE), K3 = -hazard (BCE + phase baseline):",
            "(objective) = K3a - K2, (phase baseline) = K3 - K3a, (total) = K3 - K2, (residual) = M01 - K3, i.e. our architecture against theirs",
            "under the same formulation and features;",
            "last block, union (post hoc, design doc 4.6): (+kf) = our representation plus KRAFTON's kf_* features vs ours alone with the same learner,",
            "(+ours) = HGB-union - HGB-kraftonfeat: our representation added to KRAFTON's inputs with the same GBDT",
        ]),
        "\\begin{tabular}{@{}l c c c c@{}}",
        "\\toprule",
        "Contrast & $\\Delta$AUC & $\\Delta$Spearman & $\\Delta$Top-5 AUC & $\\Delta$C-index \\\\",
        "\\midrule",
    ]
    present = [row for row in EXTERNAL_CONTRAST_ROWS if any(_lookup(idx, row[1], m) is not None for m in METRICS)]
    previous = None
    for label, name, group, tag in present:
        if previous is not None and group != previous:
            out.append("\\midrule")
        previous = group
        auc = _lookup(idx, name, "auc")
        notes = [tag] if tag else []
        if auc is not None and int(auc.k) < int(auc.k_total):
            notes.append(f"{int(auc.k)} cohorts")
        suffix = f" ({', '.join(notes)})" if notes else ""
        cells = [_delta_cell(_lookup(idx, name, m)) for m in ("auc", "spearman", "top5_auc", "cindex")]
        out.append(" & ".join([label + suffix, *cells]) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(out)


def render_decomposition_table(pooled: pd.DataFrame) -> str:
    """The decomposition table: the CLEAN single-factor steps of :data:`DECOMP_STEPS` (every row ``to - from``, one factor changes), in
    four groups separated by a rule (inputs | the K ladder | structure | extension); pooled ``Delta AUC`` and ``Delta AUC within phase``
    (unweighted mean over cut phases of the per-phase AUC), each as ``delta [95% CI]`` with the number of cohorts that were pooled over
    the cohorts of the analysis (``k/k_total``).  Paired match-level bootstrap per cohort, DerSimonian-Laird pooling, exploratory, no Holm.
    A step whose models were not run is left out and named in the header comment; a bundled contrast (several factors at once) never
    stands in for it."""
    idx = _index_rows(pooled)
    legend = [
        "decomposition in single-factor steps, one row per step, every row is (to) - (from) and changes ONE factor; stratum all, exploratory, no Holm;",
        "delta [95% CI]: paired match-level bootstrap on identical rows per cohort, DerSimonian-Laird pooled; Coh. = cohorts pooled / cohorts analysed;",
        "within phase = unweighted mean over the cut phases (>= 30 rows and both classes) of the per-phase AUC, the pooled AUC stays the primary metric;",
        "K2 = KRAFTON-tf-ourfeat (their transformer and Cox loss on our features), K3a = KRAFTON-tf-ourfeat-bce (BCE), K3 = KRAFTON-tf-ourfeat-hazard (BCE + phase baseline);",
        "the bundled contrasts (several factors at once) are not steps, they are in tab_external_contrasts.tex: KRAFTON-tf-ourfeat - KRAFTON-tf, HGB-kraftonfeat - KRAFTON-tf, HGB-cut - SAGONG-xgb, M01 - KRAFTON-tf-ourfeat",
    ]
    rows, missing, previous = [], [], None
    for group, step, cell, name in DECOMP_STEPS:
        auc, within = _lookup(idx, name, "auc"), _lookup(idx, name, "auc_within_phase")
        if auc is None and within is None:
            missing.append(name)
            continue
        if previous is not None and group != previous:
            rows.append("\\midrule")  # a rule only between groups that have rows
        previous = group

        def part(r):
            return ("---", "---") if r is None else (_delta_cell(r), f"{int(r.k)}/{int(r.k_total)}")

        rows.append(" & ".join([step, cell, *part(auc), *part(within)]) + " \\\\")
    if missing:
        legend.append("steps not run (left out): " + ", ".join(missing))
    out = [
        _tex_comment("tab_decomp", legend),
        "\\begin{tabular}{@{}l l c c c c@{}}",
        "\\toprule",
        "Step & Contrast & $\\Delta$AUC [95\\% CI] & Coh. & $\\Delta$AUC within phase [95\\% CI] & Coh. \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    return "\n".join(out)


def render_strata_table(pooled: pd.DataFrame) -> str:
    idx = _index_rows(pooled)
    strata = _display_strata(pooled["stratum"]) if len(pooled) else [STRATUM_ALL]
    names = {STRATUM_ALL: "All matches"} | {s: s.capitalize() for s in strata if s != STRATUM_ALL}
    out = [
        _tex_comment("tab_primary_strata", [
            "primary contrasts by match type (delta [95% CI]); verdicts per stratum are in contrasts_pooled.csv (strata use the",
            "unadjusted one-sided p, exploratory)",
        ]),
        f"\\begin{{tabular}}{{@{{}}l l l {'c ' * len(strata)}@{{}}}}".replace(" @{}", "@{}"),
        "\\toprule",
        " & Contrast & Metric & " + " & ".join(names[s] for s in strata) + " \\\\",
        "\\midrule",
    ]
    for pid, key in PRIMARY.items():
        label, contrast, metric = PRIMARY_ROWS[pid]
        cells = []
        for s in strata:
            r = _lookup(idx, key[0], key[1], s)
            cells.append(_delta_cell(r))
        out.append(" & ".join([label, contrast, metric, *cells]) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------- forest plot
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BAND = "#ecebe6"
ACCENT = "#2a78d6"


def forest_data(pooled: pd.DataFrame, contrasts: pd.DataFrame) -> pd.DataFrame:
    """The numbers behind the forest plot (also its table view): per-cohort and pooled rows of P1-P5."""
    idx = _index_rows(pooled)
    rows = []
    for pid, (name, metric) in PRIMARY.items():
        sub = contrasts[(contrasts["contrast"] == name) & (contrasts["metric"] == metric) & (contrasts["stratum"] == STRATUM_ALL)]
        for r in sub.sort_values("cohort").itertuples(index=False):
            if np.isfinite([r.estimate, r.ci_low, r.ci_high]).all():
                rows.append({"panel": pid, "contrast": name, "metric": metric, "cohort": r.cohort, "kind": "cohort",
                             "estimate": r.estimate, "ci_low": r.ci_low, "ci_high": r.ci_high, "se": r.se, "n_matches": r.n_matches})  # fmt: skip
        p = _lookup(idx, name, metric)
        if p is not None:
            rows.append({"panel": pid, "contrast": name, "metric": metric, "cohort": "pooled", "kind": "pooled",
                         "estimate": p.theta, "ci_low": p.ci95_low, "ci_high": p.ci95_high, "se": p.se, "n_matches": float("nan")})  # fmt: skip
    return _frame(rows, ["panel", "contrast", "metric", "cohort", "kind", "estimate", "ci_low", "ci_high", "se", "n_matches"])


def _plain_p(p: float) -> str:
    return "n/a" if not math.isfinite(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


FOREST_FIGSIZE = (7.2, 3.65)
# text above every panel, in points from the panel's top-left corner: title, verdict, Holm p (one short line each, so that
# neighbouring panels (about 1.2 in wide) can never run into each other)
FOREST_TEXT_OFFSETS = (22.0, 13.0, 4.5)


def forest_panel_text(pooled_row) -> tuple[str, str]:
    """``(verdict, Holm p)`` lines of a panel (``("not estimable", "")`` without a pooled estimate)."""
    if pooled_row is None:
        return "not estimable", ""
    return str(pooled_row.verdict), f"Holm p {_plain_p(pooled_row.p_holm)}"


def build_forest_figure(data: pd.DataFrame, pooled: pd.DataFrame):
    """Small multiples, one panel per primary contrast: cohort estimates in grey, the pooled estimate in the accent.

    Above each panel three short lines: ``P3  team axis`` (bold), the verdict and ``Holm p ...``."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    idx = _index_rows(pooled)
    coh = data[data["kind"] == "cohort"]
    size = coh.groupby("cohort")["n_matches"].max()
    order = sorted(size.index, key=lambda c: (-size[c], c)) if len(size) else []
    n = len(order)
    fig = Figure(figsize=FOREST_FIGSIZE, dpi=300, facecolor="white")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, len(PRIMARY), sharey=True, gridspec_kw={"wspace": 0.15})
    fig.subplots_adjust(left=0.19, right=0.995, top=0.80, bottom=0.21)
    ypos = {c: j for j, c in enumerate(order)}
    y_pool = n + 0.7
    for ax, (pid, (name, metric)) in zip(axes, PRIMARY.items()):
        sub = data[data["panel"] == pid]
        sesoi = SESOI[metric]
        lo = min([-sesoi, 0.0] + sub["ci_low"].tolist())
        hi = max([sesoi, 0.0] + sub["ci_high"].tolist())
        pad = 0.08 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(n + 1.45, -0.9)
        ax.axvspan(-sesoi, sesoi, color=BAND, lw=0, zorder=0)
        ax.axvline(0.0, color=MUTED, lw=0.7, zorder=1)
        ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.axhline(n - 0.15, color=BASELINE, lw=0.6, zorder=1)
        for r in sub[sub["kind"] == "cohort"].itertuples(index=False):
            y = ypos[r.cohort]
            ax.hlines(y, r.ci_low, r.ci_high, color=MUTED, lw=1.0, zorder=2)
            ax.plot([r.estimate], [y], "o", ms=3.6, color=MUTED, mec="white", mew=0.5, zorder=3)
        for r in sub[sub["kind"] == "pooled"].itertuples(index=False):
            ax.hlines(y_pool, r.ci_low, r.ci_high, color=ACCENT, lw=1.9, zorder=2)
            ax.plot([r.estimate], [y_pool], "D", ms=5.2, color=ACCENT, mec="white", mew=0.6, zorder=3)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.spines["bottom"].set_linewidth(0.7)
        ax.xaxis.set_major_locator(MaxNLocator(3, prune=None))
        ax.tick_params(axis="x", labelsize=6, length=2, width=0.6, color=BASELINE, labelcolor=INK2, pad=1.5)
        ax.tick_params(axis="y", length=0, labelsize=6.2, labelcolor=INK2)
        ax.set_xlabel(f"\u0394 {METRIC_LABEL[metric]}", fontsize=6.5, color=INK2, labelpad=2)
        verdict_txt, p_txt = forest_panel_text(_lookup(idx, name, metric))
        for text, off, size_pt, weight, color in (
            (f"{pid}  {FOREST_TITLES[pid]}", FOREST_TEXT_OFFSETS[0], 6.8, "bold", INK),
            (verdict_txt, FOREST_TEXT_OFFSETS[1], 6.2, "normal", INK2),
            (p_txt, FOREST_TEXT_OFFSETS[2], 6.0, "normal", INK2),
        ):
            if text:
                ax.annotate(text, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0.0, off), textcoords="offset points",
                            fontsize=size_pt, color=color, fontweight=weight, ha="left", va="bottom")  # fmt: skip
    axes[0].set_yticks([ypos[c] for c in order] + [y_pool])
    axes[0].set_yticklabels([cohort_label(c) for c in order] + ["Pooled (RE)"])
    if order:
        axes[0].get_yticklabels()[-1].set_fontweight("bold")
        axes[0].get_yticklabels()[-1].set_color(INK)
    handles = [
        Line2D([], [], marker="o", color=MUTED, lw=1.0, ms=3.6, label="Cohort (95% bootstrap interval)"),
        Line2D([], [], marker="D", color=ACCENT, lw=1.9, ms=5.0, label="Pooled (DerSimonian\u2013Laird, 95% interval)"),
        Patch(facecolor=BAND, label="\u00b1SESOI"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.3, bbox_to_anchor=(0.58, 0.0), handlelength=1.8)
    for t in leg.get_texts():
        t.set_color(INK2)
    return fig


def draw_forest(data: pd.DataFrame, pooled: pd.DataFrame, stem: Path) -> None:
    """Write the forest plot of P1-P5 as ``<stem>.png`` (300 dpi) and ``<stem>.pdf`` (TrueType fonts)."""
    from matplotlib import rc_context

    fig = build_forest_figure(data, pooled)
    stem.parent.mkdir(parents=True, exist_ok=True)
    with rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
        fig.savefig(stem.with_suffix(".pdf"), facecolor="white")


# --------------------------------------------------------------------------- meta-regression on the training size (4.4)
METAREG_CONTRASTS: dict[str, tuple[str, str]] = {  # panel -> (formula, metric); all exploratory (design doc 4.4)
    "P1": ("phase_main", "auc"),
    "P2": ("team_main", "auc"),
    "P5": ("zone_nohab", "auc"),
    "m11_m00": ("m11_m00", "auc"),
    "m11r_vs_hgbcut": ("m11r_vs_hgbcut", "auc"),
}
METAREG_TITLES = {
    "P1": "P1 phase axis", "P2": "P2 team axis", "P5": "P5 zone object (noHab)", "m11_m00": "M11 \u2212 M00",
    "m11r_vs_hgbcut": "M11R \u2212 HGB-cut",
}  # fmt: skip
METAREG_COLUMNS = [
    "panel", "contrast", "metric", "exploratory", "k", "n_min", "n_max", "slope", "se", "ci95_low", "ci95_high", "z", "p", "p_kh",
    "ci95_low_kh", "ci95_high_kh", "intercept_at_mean_log_n", "mean_log_n", "tau2", "tau2_no_moderator", "r2_heterogeneity",
    "QE", "QE_df", "QE_p", "QM", "crossover_n", "crossover_ci_low", "crossover_ci_high", "crossover_where", "pred_at_n_min",
    "pred_at_n_max", "note",
]  # fmt: skip
METAREG_DATA_COLUMNS = ["panel", "contrast", "exploratory", "cohort", "n_train", "log_n", "estimate", "se", "weight_re", "fitted", "residual"]


def meta_regression(y, se, x, *, alpha: float = ALPHA) -> dict:
    """Random-effects meta-regression ``y_i = a + b (x_i - mean x) + u_i + e_i`` with a method-of-moments tau^2.

    ``y`` / ``se``: per-study estimate and standard error (here the per-cohort contrast and its bootstrap SE), ``x`` the
    moderator (``ln`` of the training matches).  tau^2 is the generalised DerSimonian-Laird moment estimator,
    ``max(0, (Q_E - (k - 2)) / tr(P))`` with ``Q_E`` the residual weighted sum of squares of the fixed-effect fit and
    ``tr(P) = sum w - tr((X'WX)^-1 X'W^2 X)``, ``w = 1 / se^2``; the coefficients are then the weighted least squares
    fit with weights ``1 / (se^2 + tau2)``.

    Returned: ``slope`` (per unit of ``x``), ``se``, normal-theory interval and two-sided ``p``; the Knapp-Hartung
    variant (t with k - 2 df, variance scaled by ``max(1, s^2)``) as ``p_kh`` / ``ci95_*_kh``; ``tau2`` of the
    regression and of the intercept-only model and ``r2_heterogeneity`` (share of tau^2 the moderator removes), the
    residual heterogeneity test ``QE``, the moderator test ``QM = z^2``; and the point where the fitted effect
    crosses zero, ``crossover_x = mean x - a / b``, with a Fieller interval (bounded only when the slope is
    significant at ``alpha``; otherwise NaN).  Studies with a non-finite estimate or SE <= 0 are dropped; fewer than
    three leave every statistic NaN."""
    y, se, x = (np.asarray(v, dtype=np.float64) for v in (y, se, x))
    ok = np.isfinite(y) & np.isfinite(se) & (se > 0) & np.isfinite(x)
    y, se, x = y[ok], se[ok], x[ok]
    k = int(y.size)
    nan = float("nan")
    out = {key: nan for key in (
        "slope", "se", "ci95_low", "ci95_high", "z", "p", "p_kh", "ci95_low_kh", "ci95_high_kh", "intercept", "mean_x", "tau2",
        "tau2_no_moderator", "r2_heterogeneity", "QE", "QE_p", "QM", "crossover_x", "crossover_ci_low", "crossover_ci_high",
    )}  # fmt: skip
    out.update(k=k, QE_df=max(k - 2, 0), crossover_note="", weights=np.full(k, nan), fitted=np.full(k, nan))
    if k < 3 or np.ptp(x) <= 0:
        out["crossover_note"] = "needs at least 3 cohorts with different training sizes"
        return out
    mean_x = float(x.mean())
    X = np.column_stack([np.ones(k), x - mean_x])
    w = 1.0 / se**2
    a_fe = X.T @ (w[:, None] * X)
    beta_fe = np.linalg.solve(a_fe, X.T @ (w * y))
    qe = float((w * (y - X @ beta_fe) ** 2).sum())
    trace_p = float(w.sum() - np.trace(np.linalg.solve(a_fe, X.T @ ((w**2)[:, None] * X))))
    tau2 = max(0.0, (qe - (k - 2)) / trace_p) if trace_p > 0 else 0.0
    ws = 1.0 / (se**2 + tau2)
    cov = np.linalg.inv(X.T @ (ws[:, None] * X))
    beta = cov @ (X.T @ (ws * y))
    a, b = float(beta[0]), float(beta[1])
    b_se = float(np.sqrt(cov[1, 1]))
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    z = b / b_se
    resid = y - X @ beta
    scale = float((ws * resid**2).sum() / (k - 2))  # Knapp-Hartung residual scale
    kh_se = b_se * np.sqrt(max(1.0, scale))
    t_crit = float(stats.t.ppf(1 - alpha / 2, k - 2))
    tau2_0 = M.dersimonian_laird(y, se)["tau2"]
    out.update(
        slope=b, se=b_se, ci95_low=b - z_crit * b_se, ci95_high=b + z_crit * b_se, z=z, p=float(2 * stats.norm.sf(abs(z))),
        p_kh=float(2 * stats.t.sf(abs(b / kh_se), k - 2)), ci95_low_kh=b - t_crit * kh_se, ci95_high_kh=b + t_crit * kh_se,
        intercept=a, mean_x=mean_x, tau2=float(tau2), tau2_no_moderator=float(tau2_0),
        r2_heterogeneity=float(min(1.0, max(0.0, 1.0 - tau2 / tau2_0))) if tau2_0 > 0 else nan,
        QE=qe, QE_p=float(stats.chi2.sf(qe, k - 2)), QM=float(z**2), weights=ws, fitted=X @ beta, cov=cov,
    )  # fmt: skip
    if b != 0.0:
        out["crossover_x"] = mean_x - a / b
        # Fieller: all u = x* - mean_x with (a + b u)^2 <= z^2 Var(a + b u); bounded iff b is significantly non-zero
        v00, v01, v11 = cov[0, 0], cov[0, 1], cov[1, 1]
        qa, qb, qc = b**2 - z_crit**2 * v11, 2.0 * (a * b - z_crit**2 * v01), a**2 - z_crit**2 * v00
        disc = qb**2 - 4.0 * qa * qc
        if qa > 0 and disc >= 0:
            r1, r2 = (-qb - np.sqrt(disc)) / (2 * qa), (-qb + np.sqrt(disc)) / (2 * qa)
            out["crossover_ci_low"], out["crossover_ci_high"] = mean_x + min(r1, r2), mean_x + max(r1, r2)
        else:
            out["crossover_note"] = "slope not significant: the crossover interval is unbounded"
    else:
        out["crossover_note"] = "zero slope: no crossover"
    return out


def metareg_tables(contrasts: pd.DataFrame, train_matches: Mapping[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Meta-regressions of the per-cohort contrasts of :data:`METAREG_CONTRASTS` (stratum ``all``) on ``ln`` of the cohort's
    training matches.  Returns ``(one row per contrast, one row per contrast and cohort)``."""
    rows, data = [], []
    for panel, (name, metric) in METAREG_CONTRASTS.items():
        sub = contrasts[(contrasts["contrast"] == name) & (contrasts["metric"] == metric) & (contrasts["stratum"] == STRATUM_ALL)]
        sub = sub[sub["cohort"].isin(train_matches)].sort_values("cohort")
        if sub.empty:
            continue
        n = sub["cohort"].map(train_matches).to_numpy(dtype=float)
        fit = meta_regression(sub["estimate"].to_numpy(), sub["se"].to_numpy(), np.log(n))
        usable = np.isfinite(sub["estimate"].to_numpy()) & np.isfinite(sub["se"].to_numpy()) & (sub["se"].to_numpy() > 0)
        n_ok = n[usable]
        crossover = float(np.exp(fit["crossover_x"])) if np.isfinite(fit["crossover_x"]) else float("nan")
        lo = float(np.exp(fit["crossover_ci_low"])) if np.isfinite(fit["crossover_ci_low"]) else float("nan")
        hi = float(np.exp(fit["crossover_ci_high"])) if np.isfinite(fit["crossover_ci_high"]) else float("nan")
        where = ""
        if np.isfinite(crossover) and n_ok.size:
            where = "below the observed sizes" if crossover < n_ok.min() else "above the observed sizes" if crossover > n_ok.max() else "inside the observed sizes"
        pred = lambda nn: (fit["intercept"] + fit["slope"] * (np.log(nn) - fit["mean_x"])) if np.isfinite(fit["slope"]) else float("nan")  # noqa: E731
        rows.append({
            "panel": panel, "contrast": name, "metric": metric, "exploratory": True, "k": fit["k"], "n_min": float(n_ok.min()) if n_ok.size else float("nan"),
            "n_max": float(n_ok.max()) if n_ok.size else float("nan"), "slope": fit["slope"], "se": fit["se"], "ci95_low": fit["ci95_low"],
            "ci95_high": fit["ci95_high"], "z": fit["z"], "p": fit["p"], "p_kh": fit["p_kh"], "ci95_low_kh": fit["ci95_low_kh"],
            "ci95_high_kh": fit["ci95_high_kh"], "intercept_at_mean_log_n": fit["intercept"], "mean_log_n": fit["mean_x"],
            "tau2": fit["tau2"], "tau2_no_moderator": fit["tau2_no_moderator"], "r2_heterogeneity": fit["r2_heterogeneity"],
            "QE": fit["QE"], "QE_df": fit["QE_df"], "QE_p": fit["QE_p"], "QM": fit["QM"], "crossover_n": crossover,
            "crossover_ci_low": lo, "crossover_ci_high": hi, "crossover_where": where,
            "pred_at_n_min": pred(n_ok.min()) if n_ok.size else float("nan"), "pred_at_n_max": pred(n_ok.max()) if n_ok.size else float("nan"),
            "note": fit["crossover_note"],
        })  # fmt: skip
        weights, fitted = iter(fit["weights"]), iter(fit["fitted"])
        for r, good, nn in zip(sub.itertuples(index=False), usable, n):
            w_i, f_i = (next(weights), next(fitted)) if good else (float("nan"), float("nan"))
            data.append({"panel": panel, "contrast": name, "exploratory": True, "cohort": r.cohort, "n_train": int(nn), "log_n": float(np.log(nn)),
                         "estimate": r.estimate, "se": r.se, "weight_re": w_i, "fitted": f_i, "residual": r.estimate - f_i})  # fmt: skip
    return _frame(rows, METAREG_COLUMNS), _frame(data, METAREG_DATA_COLUMNS)


def _nice_ticks(lo: float, hi: float) -> list[float]:
    """Round tick values on a log axis between ``lo`` and ``hi``, at least a factor 1.9 apart."""
    cand = [50, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000]
    ticks: list[float] = []
    for c in cand:
        if lo / 1.15 <= c <= hi * 1.15 and (not ticks or c / ticks[-1] >= 1.9):
            ticks.append(c)
    return ticks if len(ticks) >= 2 else [lo, hi]


def draw_metareg(table: pd.DataFrame, data: pd.DataFrame, stem: Path) -> None:
    """Bubble plot (area ~ random-effects weight) of the per-cohort contrast against the training matches, with the
    meta-regression line and its 95% band; one panel per contrast.  Exploratory."""
    from matplotlib import rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

    panels = [p for p in METAREG_CONTRASTS if (table["panel"] == p).any()]
    ncol = 3
    nrow = max(1, -(-len(panels) // ncol))
    fig = Figure(figsize=(7.2, 2.35 * nrow + 0.65), dpi=300, facecolor="white")
    FigureCanvasAgg(fig)
    axes = fig.subplots(nrow, ncol, squeeze=False, gridspec_kw={"wspace": 0.34, "hspace": 0.62})
    top = 1 - 0.50 / (2.35 * nrow + 0.65)
    fig.subplots_adjust(left=0.085, right=0.985, top=top - 0.045, bottom=0.215 / nrow + 0.02)
    wmax = float(np.nanmax(data["weight_re"])) if len(data) and np.isfinite(data["weight_re"]).any() else 1.0
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, panel in zip(axes.ravel(), panels):
        ax.set_visible(True)
        t = table[table["panel"] == panel].iloc[0]
        d = data[(data["panel"] == panel) & np.isfinite(data["weight_re"])]
        name, metric = METAREG_CONTRASTS[panel]
        x = np.log(d["n_train"].to_numpy(dtype=float))
        grid = np.linspace(x.min() - 0.12, x.max() + 0.12, 80) if len(x) else np.array([0.0, 1.0])
        ax.axhline(0.0, color=MUTED, lw=0.7, zorder=1)
        ax.axhspan(-SESOI[metric], SESOI[metric], color=BAND, lw=0, zorder=0)
        if np.isfinite(t["slope"]):
            cov_fit = meta_regression(d["estimate"].to_numpy(), d["se"].to_numpy(), x)
            c = cov_fit["cov"]
            u = grid - cov_fit["mean_x"]
            mean = cov_fit["intercept"] + cov_fit["slope"] * u
            sd = np.sqrt(np.maximum(c[0, 0] + 2 * u * c[0, 1] + u**2 * c[1, 1], 0.0))
            ax.fill_between(np.exp(grid), mean - 1.96 * sd, mean + 1.96 * sd, color=ACCENT, alpha=0.16, lw=0, zorder=2)
            ax.plot(np.exp(grid), mean, color=ACCENT, lw=1.4, zorder=3)
        sizes = 14 + 150 * d["weight_re"].to_numpy() / wmax
        ax.scatter(d["n_train"], d["estimate"], s=sizes, color=MUTED, alpha=0.82, edgecolor="white", linewidth=0.5, zorder=4)
        ends = d.sort_values("n_train").iloc[[0, -1]] if len(d) > 1 else d.iloc[:0]
        for j, r in enumerate(ends.itertuples(index=False)):  # label the smallest and the largest cohort
            ax.annotate(cohort_label(r.cohort).replace(" squad", "").replace(" TPP", " T").replace(" FPP", " F"), (r.n_train, r.estimate),
                        xytext=(4 if j == 0 else -4, 5), textcoords="offset points", fontsize=5.2, color=INK2,
                        ha="left" if j == 0 else "right")  # fmt: skip
        ax.set_xscale("log")
        ticks = _nice_ticks(float(d["n_train"].min()), float(d["n_train"].max())) if len(d) else [100, 1000]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.set_xticklabels([f"{v:,.0f}" for v in ticks])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(float(d["n_train"].min()) / 1.35, float(d["n_train"].max()) * 1.35) if len(d) else None
        ax.xaxis.grid(True, color=GRID, lw=0.5)
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
            ax.spines[side].set_linewidth(0.7)
        ax.tick_params(axis="both", labelsize=5.8, length=2, width=0.6, color=BASELINE, labelcolor=INK2, pad=1.5)
        ax.set_xlabel("training matches", fontsize=6.0, color=INK2, labelpad=1.5)
        ax.set_ylabel(f"\u0394 {METRIC_LABEL[metric]}", fontsize=6.0, color=INK2, labelpad=1.5)
        ax.text(0.0, 1.20, METAREG_TITLES[panel], transform=ax.transAxes, fontsize=6.8, color=INK, fontweight="bold", ha="left", va="bottom")
        p_txt = "p < 0.001" if t["p"] < 0.001 else f"p = {t['p']:.3f}"
        line = f"slope {t['slope']:+.4f} per ln N, {p_txt}" if np.isfinite(t["slope"]) else "not estimable"
        ax.text(0.0, 1.05, line, transform=ax.transAxes, fontsize=5.6, color=INK2, ha="left", va="bottom")
    legend_ax = axes.ravel()[len(panels)] if len(panels) < axes.size else None
    handles = [
        Line2D([], [], marker="o", color=MUTED, lw=0, ms=4.2, label="cohort (area ~ RE weight)"),
        Line2D([], [], color=ACCENT, lw=1.4, label="meta-regression, 95% band"),
        Patch(facecolor=BAND, label="\u00b1SESOI"),
    ]
    if legend_ax is not None:
        legend_ax.set_visible(True)
        legend_ax.axis("off")
        leg = legend_ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=6.0, bbox_to_anchor=(0.0, 0.95))
        for tx in leg.get_texts():
            tx.set_color(INK2)
        legend_ax.text(0.0, 0.40, f"Exploratory (design 4.4). Random-effects\nmeta-regression on ln(training matches),\nmethod-of-moments \u03c4\u00b2, {int(table['k'].max())} cohorts.", fontsize=5.8,
                       color=INK2, va="top", ha="left", transform=legend_ax.transAxes)  # fmt: skip
    else:
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.0, bbox_to_anchor=(0.5, 0.0))
    stem.parent.mkdir(parents=True, exist_ok=True)
    with rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
        fig.savefig(stem.with_suffix(".pdf"), facecolor="white")


# --------------------------------------------------------------------------- phase-group strata (exploratory)
PHASE_COLUMNS = [
    "level", "contrast", "metric", "phase_group", "phases", "cohort", "estimate", "se", "ci95_low", "ci95_high", "n_valid", "n_rows",
    "n_matches", "k", "k_total", "same_sign", "tau2", "Q", "I2", "p_one_sided", "p_two_sided", "ci90_low", "ci90_high", "sesoi",
    "direction", "verdict", "exploratory", "cohorts",
]  # fmt: skip


def phase_group_of(stratum: str) -> str:
    return stratum[len(PHASE_PREFIX):]


def phase_contrast_table(contrasts: pd.DataFrame, n_cohorts: int, groups: Mapping[str, Sequence[int]] = PHASE_GROUPS) -> pd.DataFrame:
    """Per-cohort and DL-pooled contrasts of the phase-group strata, one tidy table (``level`` = ``cohort`` | ``pooled``).

    ``contrasts``: the per-cohort rows of the ``phase:<group>`` strata (same columns as the main per-cohort table).  Every
    stratum is analysed with its own paired match bootstrap (cohort rows) and pooled with DerSimonian-Laird (pooled rows,
    ``contrasts_pooled.csv`` conventions: one-sided p for an improvement, unadjusted verdict).  All rows are exploratory."""
    if contrasts.empty:
        return _frame([], PHASE_COLUMNS)
    pooled = pool_contrasts(contrasts, n_cohorts)
    rows = []
    for r in pooled.itertuples(index=False):
        group = phase_group_of(r.stratum)
        part = contrasts[(contrasts["contrast"] == r.contrast) & (contrasts["metric"] == r.metric) & (contrasts["stratum"] == r.stratum)]
        rows.append({
            "level": "pooled", "contrast": r.contrast, "metric": r.metric, "phase_group": group,
            "phases": "-".join(f"{p}" for p in (groups[group][0], groups[group][-1])) if len(groups[group]) > 1 else str(groups[group][0]),
            "cohort": "", "estimate": r.theta, "se": r.se, "ci95_low": r.ci95_low, "ci95_high": r.ci95_high,
            "n_valid": np.nan, "n_rows": int(part["n_rows"].sum()), "n_matches": int(part["n_matches"].sum()), "k": int(r.k),
            "k_total": int(r.k_total), "same_sign": int(r.same_sign), "tau2": r.tau2, "Q": r.Q, "I2": r.I2,
            "p_one_sided": r.p_one_sided, "p_two_sided": r.p_two_sided, "ci90_low": r.ci90_low, "ci90_high": r.ci90_high,
            "sesoi": r.sesoi, "direction": int(r.direction), "verdict": r.verdict, "exploratory": True, "cohorts": r.cohorts,
        })  # fmt: skip
    for r in contrasts.itertuples(index=False):
        group = phase_group_of(r.stratum)
        rows.append({
            "level": "cohort", "contrast": r.contrast, "metric": r.metric, "phase_group": group,
            "phases": "-".join(f"{p}" for p in (groups[group][0], groups[group][-1])) if len(groups[group]) > 1 else str(groups[group][0]),
            "cohort": r.cohort, "estimate": r.estimate, "se": r.se, "ci95_low": r.ci_low, "ci95_high": r.ci_high,
            "n_valid": r.n_valid, "n_rows": r.n_rows, "n_matches": r.n_matches, "k": 1, "k_total": np.nan, "same_sign": np.nan,
            "tau2": np.nan, "Q": np.nan, "I2": np.nan, "p_one_sided": np.nan, "p_two_sided": np.nan, "ci90_low": np.nan,
            "ci90_high": np.nan, "sesoi": SESOI.get(r.metric, np.nan), "direction": DIRECTION[r.metric], "verdict": "",
            "exploratory": True, "cohorts": r.cohort,
        })  # fmt: skip
    df = _frame(rows, PHASE_COLUMNS)
    fo, go = list(FORMULAS), list(groups)
    key = [(fo.index(c), METRICS.index(m), go.index(g), 0 if lv == "pooled" else 1, co)
           for c, m, g, lv, co in zip(df["contrast"], df["metric"], df["phase_group"], df["level"], df["cohort"])]
    order = sorted(range(len(df)), key=key.__getitem__)
    return df.iloc[order].reset_index(drop=True)


# --------------------------------------------------------------------------- driver
@dataclass
class AggregateResult:
    inventory: pd.DataFrame
    arm_metrics: pd.DataFrame
    arm_means: pd.DataFrame
    contrasts: pd.DataFrame
    pooled: pd.DataFrame
    kappa: pd.DataFrame
    seed_stability: pd.DataFrame
    issues: pd.DataFrame
    summary: dict
    labels: str
    cohorts: list[str]
    strata: list[str]
    out_dir: Path | None = None
    metareg: pd.DataFrame | None = None  # exploratory meta-regression on the training size (design doc 4.4)
    metareg_data: pd.DataFrame | None = None
    by_phase: pd.DataFrame | None = None  # exploratory phase-group strata (early / mid / late), see ``phase_strata``


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.10g")


def _reference_fingerprint(fps: Mapping[tuple[str, int], str]) -> str:
    """The row set most arms agree on (seed-42 files first); ties go to the first arm in the canonical order."""
    main = {k: v for k, v in fps.items() if k[1] == MAIN_SEED} or dict(fps)
    counts = Counter(main.values())
    best = max(counts.values())
    tied = {fp for fp, c in counts.items() if c == best}
    for key in sorted(main, key=lambda k: (arm_rank(k[0]), k[1])):
        if main[key] in tied:
            return main[key]
    raise AssertionError("unreachable")


def aggregate(
    runs_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    n_boot: int = N_BOOT,
    seed: int = 0,
    workers: int = 1,
    cohorts: Iterable[str] | None = None,
    strata: str | Sequence[str] | None = "auto",
    figures: bool = True,
    skip_misaligned: bool = False,
    min_stratum_matches: int = MIN_STRATUM_MATCHES,
    train_matches: Mapping[str, int] | None = None,
    phase_strata: str = "off",
) -> AggregateResult:
    """Run the whole analysis on ``runs_dir``; writes the outputs when ``out_dir`` is given.

    ``phase_strata``: ``"off"`` (default), ``"add"`` (also analyse the phase groups early / mid / late of
    :data:`PHASE_GROUPS` as strata next to the match types; they fill ``result.by_phase`` and ``contrasts_by_phase.csv`` and
    leave every main table untouched) or ``"only"`` (skip everything else: only the phase strata are computed and
    written).  Each phase stratum has its own paired match bootstrap and is DL-pooled over the cohorts.

    ``train_matches``: ``{cohort: training matches}`` for the meta-regression on the training size (design doc 4.4);
    cohorts missing from it take the count recorded in their runs' ``metrics.json`` (``n_matches.train``)."""
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if phase_strata not in ("off", "add", "only"):
        raise ValueError("phase_strata must be 'off', 'add' or 'only'")
    runs_dir = Path(runs_dir)
    issues: list[tuple] = []
    runs, scan_issues = scan_runs(runs_dir, cohorts)
    issues += scan_issues
    if not runs:
        raise ValueError(f"no finished runs (pred_test.csv and metrics.json) under {runs_dir}")
    labels = check_labels(runs)
    cohort_list = sorted({r.cohort for r in runs})
    by_cohort = {c: sorted((r for r in runs if r.cohort == c), key=lambda r: (arm_rank(r.arm), r.seed)) for c in cohort_list}

    with _Parallel(workers) as par:
        # ---- phase 1: read every file once, fingerprint the rows, decide which runs share rows
        preps = par.map(
            _prep_task,
            [{"cohort": c, "runs": [(r.arm, r.seed, str(r.pred)) for r in by_cohort[c]]} for c in cohort_list],
            cost=lambda s: len(s["runs"]),
        )
        usable: dict[str, list[Run]] = {}
        dropped: dict[str, dict[tuple[str, int], str]] = {c: {} for c in cohort_list}
        run_info: dict[tuple[str, str, int], dict] = {}
        cohort_types: dict[str, dict[str, int]] = {}
        for prep in preps:
            c = prep["cohort"]
            for key, msg in prep["errors"].items():
                dropped[c][key] = f"unreadable prediction file ({msg})"
                issues.append(("warning", c, key[0], "", f"{c} {key[0]} seed {key[1]}: unreadable prediction file ({msg}); run ignored"))
            files = prep["files"]
            if not files:
                usable[c] = []
                continue
            ref = _reference_fingerprint({k: v["fp"] for k, v in files.items()})
            bad = sorted((k for k, v in files.items() if v["fp"] != ref), key=lambda k: (arm_rank(k[0]), k[1]))
            if bad:
                names = ", ".join(f"{a} (seed {s})" for a, s in bad)
                detail = (f"{c}: the predictions of {names} do not cover the same rows and labels as the other arms "
                          f"({len(files) - len(bad)} runs agree); the paired bootstrap needs identical rows")  # fmt: skip
                if not skip_misaligned:
                    raise ValueError(detail + " (pass skip_misaligned=True to drop the misaligned runs and continue)")
                for k in bad:
                    dropped[c][k] = "misaligned rows"
                    issues.append(("warning", c, k[0], "", f"{c} {k[0]} seed {k[1]}: rows differ from the other arms; run dropped"))
            for r in by_cohort[c]:
                if (r.arm, r.seed) in files:
                    run_info[(c, r.arm, r.seed)] = {**files[(r.arm, r.seed)], "aligned": (r.arm, r.seed) not in dropped[c]}
            usable[c] = [r for r in by_cohort[c] if (r.arm, r.seed) in files and (r.arm, r.seed) not in dropped[c]]
            good = next(v for k, v in files.items() if (k not in dropped[c]))
            cohort_types[c] = good["types"]

        if strata in (None, "none"):
            wanted_types: set[str] | None = set()
        elif strata == "auto":
            wanted_types = None
        else:
            wanted_types = set(strata.split(",")) if isinstance(strata, str) else set(strata)
        tasks = []
        for c in cohort_list:
            if not usable.get(c):
                continue
            types = sorted(t for t in cohort_types.get(c, {}) if wanted_types is None or t in wanted_types)
            specs = [] if phase_strata == "only" else [(stratum, ()) for stratum in [STRATUM_ALL, *types]]
            if phase_strata in ("add", "only"):
                specs += [(PHASE_PREFIX + g, tuple(ph)) for g, ph in PHASE_GROUPS.items()]
            for stratum, phases in specs:
                tasks.append({
                    "cohort": c, "stratum": stratum, "phases": phases, "n_boot": int(n_boot), "seed": int(seed),
                    "min_matches": int(min_stratum_matches), "runs": [(r.arm, r.seed, str(r.pred)) for r in usable[c]],
                })  # fmt: skip
        results = par.map(_stratum_task, tasks, cost=lambda t: sum(run_info[(t["cohort"], a, s)]["n_rows"] for a, s, _ in t["runs"]))

    # ---- assemble
    arm_rows, contrast_rows, phase_rows = [], [], []
    kappa_in: dict[str, list[dict]] = {}
    analysed: set[str] = {STRATUM_ALL}
    for res in results:
        c, stratum = res["cohort"], res["stratum"]
        if stratum.startswith(PHASE_PREFIX):  # phase groups are analysed separately: no main table sees them
            if res["skipped"]:
                issues.append(("info", c, "", "", f"{c}: stratum '{stratum}' skipped ({res['skipped']})"))
            else:
                phase_rows += res["contrast_rows"]
            continue
        if res["skipped"]:
            level = "warning" if stratum == STRATUM_ALL else "info"
            issues.append((level, c, "", "", f"{c}: stratum '{stratum}' skipped ({res['skipped']})"))
            continue
        if stratum != STRATUM_ALL:
            analysed.add(stratum)
        arm_rows += res["arm_rows"]
        contrast_rows += res["contrast_rows"]
        if res["kappa"] is not None:
            kappa_in.setdefault(stratum, []).append({"cohort": c, **res["kappa"]})
    for c in cohort_list:
        have = {r.arm for r in usable.get(c, []) if r.seed == MAIN_SEED}
        for f in FORMULAS.values():
            missing = [a for a in f.coefs if a not in have]
            if f.pooled and missing and not f.optional:
                why = [a for a in missing if (a, MAIN_SEED) in dropped.get(c, {})]
                note = f" ({', '.join(why)}: {dropped[c][(why[0], MAIN_SEED)]})" if why else ""
                issues.append(("warning", c, "", f.name, f"{c}: contrast {f.name} not computed, missing arm(s) {', '.join(missing)} at seed {MAIN_SEED}{note}"))

    inventory = _frame(
        [{"cohort": r.cohort, "arm": r.arm, "seed": r.seed, "kind": r.kind, "role": r.role, "labels": r.labels or "",
          "config_hash": r.config_hash or "", "n_rows": run_info.get((r.cohort, r.arm, r.seed), {}).get("n_rows", np.nan),
          "n_matches": run_info.get((r.cohort, r.arm, r.seed), {}).get("n_matches", np.nan),
          "aligned": run_info.get((r.cohort, r.arm, r.seed), {}).get("aligned", False),
          "pred_test": str(r.pred.relative_to(runs_dir))}
         for c in cohort_list for r in by_cohort[c] if (r.cohort, r.arm, r.seed) in run_info],
        INVENTORY_COLUMNS,
    )  # fmt: skip
    arm_df = _frame(arm_rows, ARM_METRIC_COLUMNS)
    contrast_df = _frame(contrast_rows, CONTRAST_COLUMNS)
    n_cohorts = len(cohort_list)
    pooled_df = pool_contrasts(contrast_df, n_cohorts) if len(contrast_df) else _frame([], POOLED_COLUMNS)
    kappa_df = pool_kappa(kappa_in, n_cohorts, n_boot)
    means_df = arm_means(arm_df) if len(arm_df) else _frame([], ["arm", "metric", "stratum", "mean", "sd_cohorts", "n_cohorts"])
    seeds_df = seed_stability(arm_df) if len(arm_df) else _frame([], SEED_COLUMNS)
    sizes = {c: int(n) for c, n in (train_matches or {}).items()}
    for c in cohort_list:
        seen = Counter(r.n_train for r in runs if r.cohort == c and r.n_train)
        if c not in sizes and seen:
            sizes[c] = int(seen.most_common(1)[0][0])
            if len(seen) > 1:
                issues.append(("warning", c, "", "", f"{c}: the runs record different training sizes {sorted(seen)}; {sizes[c]} used for the meta-regression"))
    for c in cohort_list:
        if c not in sizes:
            issues.append(("info", c, "", "", f"{c}: training size unknown, left out of the meta-regression"))
    metareg_df, metareg_data = metareg_tables(contrast_df, {c: sizes[c] for c in cohort_list if c in sizes}) if len(contrast_df) else (
        _frame([], METAREG_COLUMNS), _frame([], METAREG_DATA_COLUMNS))  # fmt: skip
    strata_list = _stratum_order(analysed)
    for df, cols in ((arm_df, ["cohort", "arm", "seed", "stratum", "metric"]), (contrast_df, ["cohort", "contrast", "metric", "stratum"])):
        if len(df):
            sortkey = {"arm": arm_rank, "metric": METRICS.index, "stratum": strata_list.index,
                       "contrast": list(FORMULAS).index, "cohort": str, "seed": int}  # fmt: skip
            df.sort_values(cols, key=lambda s, _k=sortkey: s.map(_k.get(s.name, str)), kind="mergesort", inplace=True)
            df.reset_index(drop=True, inplace=True)
    issue_df = _frame(sorted(set(issues), key=lambda t: (t[1], t[2], t[3], t[4], t[0])), ISSUE_COLUMNS)

    primary_results = {}
    idx = _index_rows(pooled_df) if len(pooled_df) else {}
    for pid, (name, metric) in PRIMARY.items():
        r = idx.get((name, metric, STRATUM_ALL))
        primary_results[pid] = None if r is None else {
            "contrast": name, "metric": metric, "theta": r.theta, "ci95": [r.ci95_low, r.ci95_high], "ci90": [r.ci90_low, r.ci90_high],
            "p_one_sided": r.p_one_sided, "p_holm": r.p_holm, "tau2": r.tau2, "k": int(r.k), "same_sign": int(r.same_sign),
            "verdict": r.verdict,
        }  # fmt: skip
    krow = kappa_df[(kappa_df["level"] == "pooled") & (kappa_df["stratum"] == STRATUM_ALL)] if len(kappa_df) else kappa_df
    summary = {
        "n_boot": int(n_boot), "seed": int(seed), "labels": labels, "n_cohorts": n_cohorts, "cohorts": cohort_list,
        "strata": strata_list, "sesoi": dict(SESOI), "alpha": ALPHA, "holm_family": list(HOLM_FAMILY),
        "primary": {k: list(v) for k, v in PRIMARY.items()}, "primary_results": primary_results,
        "kappa": None if krow.empty else {k: krow.iloc[0][k] for k in ("kappa", "ci_low", "ci_high", "numerator", "denominator", "den_ci95_low", "estimable", "k", "note")},
        "n_runs": int(len(inventory)), "n_issues": int(len(issue_df)), "skip_misaligned": bool(skip_misaligned),
        "runs_dir": str(runs_dir), "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": git_info(),
    }  # fmt: skip
    summary["metareg"] = {
        r.panel: {"contrast": r.contrast, "k": int(r.k), "slope_per_ln_n": r.slope, "ci95": [r.ci95_low, r.ci95_high], "p": r.p,
                  "crossover_n": r.crossover_n, "crossover_ci": [r.crossover_ci_low, r.crossover_ci_high]}  # fmt: skip
        for r in metareg_df.itertuples(index=False)
    }
    by_phase = phase_contrast_table(_frame(phase_rows, CONTRAST_COLUMNS), n_cohorts) if phase_strata != "off" else None
    result = AggregateResult(inventory, arm_df, means_df, contrast_df, pooled_df, kappa_df, seeds_df, issue_df, summary, labels,
                             cohort_list, strata_list, Path(out_dir) if out_dir is not None else None, metareg_df, metareg_data,
                             by_phase)  # fmt: skip
    if out_dir is not None:
        if phase_strata != "only":
            write_outputs(result, figures=figures)
        if by_phase is not None:
            write_phase_outputs(result, figures=figures)
    return result


def write_phase_outputs(res: AggregateResult, *, figures: bool = True) -> None:
    """``contrasts_by_phase.csv`` and ``figures/phase_strata.{pdf,png}``; nothing else in ``out_dir`` is touched."""
    out = res.out_dir
    assert out is not None and res.by_phase is not None
    _write_csv(res.by_phase, out / "contrasts_by_phase.csv")
    if figures and len(res.by_phase):
        from model_stgcn.ptr import derived

        derived.draw_phase_strata(res.by_phase, out / "figures" / "phase_strata")


def redraw_figures(out_dir: str | Path) -> list[Path]:
    """Regenerate the forest plot (+ its data twin) and the meta-regression bubble plot from the CSVs of an earlier run
    (``contrasts_pooled.csv``, ``contrasts_per_cohort.csv``, ``metareg*.csv``); no bootstrap, nothing else is rewritten."""
    out = Path(out_dir)
    for name in ("contrasts_pooled.csv", "contrasts_per_cohort.csv"):
        if not (out / name).is_file():
            raise FileNotFoundError(f"{out / name} not found: run the aggregator once before redrawing its figures")
    pooled = pd.read_csv(out / "contrasts_pooled.csv")
    contrasts = pd.read_csv(out / "contrasts_per_cohort.csv")
    written = []
    data = forest_data(pooled, contrasts)
    _write_csv(data, out / "figures" / "forest_primary_data.csv")
    if len(data):
        draw_forest(data, pooled, out / "figures" / "forest_primary")
        written += [out / "figures" / "forest_primary.png", out / "figures" / "forest_primary.pdf"]
    if (out / "metareg.csv").is_file() and (out / "metareg_data.csv").is_file():
        table, mdata = pd.read_csv(out / "metareg.csv"), pd.read_csv(out / "metareg_data.csv")
        if len(table):
            draw_metareg(table, mdata, out / "figures" / "metareg_bubble")
            written += [out / "figures" / "metareg_bubble.png", out / "figures" / "metareg_bubble.pdf"]
    return written


def write_outputs(res: AggregateResult, *, figures: bool = True) -> None:
    out = res.out_dir
    assert out is not None
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(res.arm_metrics, out / "arm_metrics_per_cohort.csv")
    _write_csv(res.arm_means, out / "arm_metrics_mean.csv")
    _write_csv(res.contrasts, out / "contrasts_per_cohort.csv")
    _write_csv(res.pooled, out / "contrasts_pooled.csv")
    _write_csv(res.kappa, out / "kappa.csv")
    _write_csv(res.seed_stability, out / "seed_stability.csv")
    _write_csv(res.inventory, out / "run_inventory.csv")
    _write_csv(res.issues, out / "issues.csv")
    if res.metareg is not None:
        _write_csv(res.metareg, out / "metareg.csv")
        _write_csv(res.metareg_data, out / "metareg_data.csv")
    write_json(out / "summary.json", res.summary)
    tables = out / "tables"
    tables.mkdir(exist_ok=True)
    (tables / "tab_primary.tex").write_text(render_primary_table(res.pooled), encoding="utf-8")
    (tables / "tab_exploratory.tex").write_text(render_exploratory_table(res.pooled, res.kappa), encoding="utf-8")
    (tables / "tab_arms_metrics.tex").write_text(render_arms_table(res.arm_means), encoding="utf-8")
    (tables / "tab_primary_strata.tex").write_text(render_strata_table(res.pooled), encoding="utf-8")
    # external baselines (4.3): the tables exist only when at least one external model was run (a stale file is removed)
    have_external = len(res.arm_means) > 0 and bool(res.arm_means["arm"].isin(EXTERNAL_ARMS).any())
    have_union = len(res.arm_means) > 0 and bool(res.arm_means["arm"].isin(UNION_ARMS).any())  # post hoc (4.6): same tables, one more block / row
    ext_means = res.arm_means[res.arm_means["arm"].isin((*EXTERNAL_ARMS, *(a for a, _ in EXTERNAL_TABLE_REFERENCES)))] if have_external else res.arm_means
    for name, text in (("tab_external.tex", render_external_table(ext_means) if have_external else None),
                       ("tab_external_contrasts.tex", render_external_contrasts_table(res.pooled) if have_external or have_union else None),
                       ("tab_grid.tex", render_grid_table(res.arm_means) if have_external or have_union else None),
                       ("tab_decomp.tex", render_decomposition_table(res.pooled))):  # fmt: skip
        path = tables / name
        if text is not None and "\\midrule\n\\midrule" not in text and "\\midrule\n\\bottomrule" not in text:
            path.write_text(text, encoding="utf-8")
        elif path.exists():
            path.unlink()
    if figures:
        data = forest_data(res.pooled, res.contrasts)
        _write_csv(data, out / "figures" / "forest_primary_data.csv")
        if len(data):
            draw_forest(data, res.pooled, out / "figures" / "forest_primary")
        if res.metareg is not None and len(res.metareg):
            draw_metareg(res.metareg, res.metareg_data, out / "figures" / "metareg_bubble")


# --------------------------------------------------------------------------- CLI
def format_primary(res: AggregateResult) -> str:
    """Plain-text table of P1-P5 (stratum all) for the console."""
    idx = _index_rows(res.pooled) if len(res.pooled) else {}
    head = f"{'':3} {'contrast':<28} {'metric':<9} {'pooled delta [95% CI]':<30} {'same sign':>9} {'tau2':>10} {'Holm p':>8}  verdict"
    lines = [head]
    for pid, (name, metric) in PRIMARY.items():
        r = idx.get((name, metric, STRATUM_ALL))
        label = FORMULAS[name].label
        if r is None:
            lines.append(f"{pid:3} {label:<28} {METRIC_LABEL[metric]:<9} {'not computed':<30}")
            continue
        ci = f"{r.theta:+.4f} [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}]"
        lines.append(f"{pid:3} {label:<28} {METRIC_LABEL[metric]:<9} {ci:<30} {int(r.same_sign):>4}/{int(r.k):<4} {r.tau2:>10.2e} {r.p_holm:>8.4f}  {r.verdict}")
    krow = res.kappa[(res.kappa["level"] == "pooled") & (res.kappa["stratum"] == STRATUM_ALL)] if len(res.kappa) else res.kappa
    if not krow.empty:
        k = krow.iloc[0]
        lines.append("kappa: " + (f"{k['kappa']:.3f} [{k['ci_low']:.3f}, {k['ci_high']:.3f}]" if bool(k["estimable"]) else str(k["note"])))
    return "\n".join(lines)


def format_metareg(res: AggregateResult) -> str:
    """Plain-text table of the exploratory meta-regression on the training size (design doc 4.4)."""
    t = res.metareg
    if t is None or t.empty:
        return "meta-regression: no cohort with a known training size"
    lines = ["exploratory meta-regression of the per-cohort AUC contrast on ln(training matches); random effects, method-of-moments tau2",
             f"{'panel':<16} {'k':>2} {'slope per ln N [95% CI]':<34} {'p':>8} {'tau2':>9}  crossover N [95% CI] (where the effect is 0)"]  # fmt: skip
    for r in t.itertuples(index=False):
        if not np.isfinite(r.slope):
            lines.append(f"{r.panel:<16} {int(r.k):>2} {'not estimable':<34}")
            continue
        ci = f"{r.slope:+.4f} [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}]"
        if np.isfinite(r.crossover_n) and not (1.0 <= r.crossover_n <= 1e7):
            cross = "outside any plausible size (slope ~ 0)"
        elif np.isfinite(r.crossover_n):
            bounds = f"[{r.crossover_ci_low:,.0f}, {r.crossover_ci_high:,.0f}]" if np.isfinite(r.crossover_ci_low) else "[unbounded]"
            cross = f"{r.crossover_n:,.0f} {bounds} {r.crossover_where}"
        else:
            cross = "none"
        lines.append(f"{r.panel:<16} {int(r.k):>2} {ci:<34} {r.p:>8.4f} {r.tau2:>9.2e}  {cross}")
    return "\n".join(lines)


def _positive_int(text: str) -> int:
    v = int(text)
    if v < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return v


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Aggregate a PTR run matrix: bootstrap, pooling, Holm, verdicts, tables, forest plot.")
    ap.add_argument("--runs-dir", default=None, help="run_matrix output: <runs>/<cohort>/<arm>/seed<s>/ (required unless --redraw-figures)")
    ap.add_argument("--redraw-figures", action="store_true",
                    help="only redraw the forest and bubble plots from the CSVs already in --out-dir (no bootstrap, nothing else changes)")  # fmt: skip
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "reports" / "bigcomp2027_results"))
    ap.add_argument("--n-boot", type=_positive_int, default=N_BOOT, help="match-level bootstrap resamples (default 2000)")
    ap.add_argument("--seed", type=int, default=0, help="base seed of the bootstrap (a fixed offset per cohort / stratum is added)")
    ap.add_argument("--workers", type=_positive_int, default=None, help="parallel processes (default: cores / 2)")
    ap.add_argument("--cohorts", default=None, help="comma-separated cohorts to include (default: all found)")
    ap.add_argument("--strata", default="auto", help="'auto' (every match type found), 'none', or a comma-separated list of match types")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--skip-misaligned", action="store_true", help="drop runs whose rows differ from the other arms instead of failing")
    ap.add_argument("--dump-dir", default=None,
                    help="dump directory: its population.json gives the training sizes of cohorts whose runs do not record them")  # fmt: skip
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    ap_args = parse_args(argv)
    args = ap_args
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("fontTools", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if args.redraw_figures:
        for path in redraw_figures(args.out_dir):
            print(f"wrote {path}")
        return 0
    if not args.runs_dir:
        raise SystemExit("--runs-dir is required (unless --redraw-figures)")
    workers = args.workers or max(1, (os.cpu_count() or 2) // 2)
    cohorts = [c for c in args.cohorts.split(",") if c] if args.cohorts else None
    sizes = None
    if args.dump_dir:
        pop = Path(args.dump_dir) / "population.json"
        if pop.is_file():
            info = json.loads(pop.read_text(encoding="utf-8")).get("cohorts", {})
            sizes = {c: int(v["after"]["train"]) for c, v in info.items() if v.get("status") == "kept"}
    res = aggregate(
        args.runs_dir, args.out_dir, n_boot=args.n_boot, seed=args.seed, workers=workers, cohorts=cohorts, strata=args.strata,
        figures=not args.no_figures, skip_misaligned=args.skip_misaligned, train_matches=sizes,
    )  # fmt: skip
    print(f"{res.summary['n_cohorts']} cohorts, {res.summary['n_runs']} runs, labels={res.labels}, {args.n_boot} resamples")
    print(format_primary(res))
    print(format_metareg(res))
    if len(res.issues):
        print(f"{len(res.issues)} issue(s), see {Path(args.out_dir) / 'issues.csv'}", file=sys.stderr)
    print(f"outputs in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
