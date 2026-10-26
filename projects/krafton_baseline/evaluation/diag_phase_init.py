"""Diagnostic (not part of the K3 matrix): K3 with the phase offsets initialised at ZERO instead of the train base rates.

    python -m adaptation.diag_phase_init --cohorts paramo_tpp_squad,karakin_tpp_squad [--out-dir <diag root>]

Answers whether the learned per-cut-phase bias can reach the baseline logits from zero at lr 1e-4 within a run (the reason for the
base-rate initialisation of K3, ``ourfeat_hazard`` docstring).  Everything else is K3: the head bias still starts at the pooled
train log-odds, the same samples, steps, early stopping and seed.  ``adaptation.ourfeat_hazard.init_intercepts`` is wrapped here,
the module itself is not modified.  Results go to a SEPARATE root (default ``model_stgcn/checkpoints/bigcomp2027_ptr_v2_diag``) so
that the aggregator of the main directory never sees them; the PTR prediction files are still asserted against the main directory.
Prints, per cohort, the offsets at the start, the offsets that the training reached, the base-rate offsets they would need and the
test AUC next to K3a / K3 of the main directory.
"""

from __future__ import annotations

# Project code is separate from the immutable vendored reference.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))
from pubg_lab.common.paths import REPO_ROOT

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = REPO_ROOT / "third_party/krafton_pubg_winprob"
REPO = REPO_ROOT
for p in (str(ROOT), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import projects.krafton_baseline.training.ourfeat_hazard as H  # noqa: E402

MAIN = REPO / "model_stgcn/checkpoints/bigcomp2027_ptr_v2"
DIAG = REPO / "model_stgcn/checkpoints/bigcomp2027_ptr_v2_diag"
_orig_init = H.init_intercepts


def zero_phase_init(head, pooled_rate, rates):
    info = _orig_init(head, pooled_rate, rates)
    if isinstance(head, H.PhaseHead):
        with torch.no_grad():
            head.phase_bias.weight.zero_()
        info["phase_offsets"] = [0.0] * H.N_CUTS
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohorts", required=True)
    ap.add_argument("--out-dir", type=Path, default=DIAG)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args(argv)
    H.init_intercepts = zero_phase_init
    for c in [x for x in a.cohorts.split(",") if x]:
        args = H.parse_args(
            [
                "--cohort",
                c,
                "--variant",
                "hazard",
                "--out-dir",
                str(a.out_dir),
                "--ref-dir",
                str(MAIN),
                "--threads",
                str(a.threads),
                "--redo",
            ]
        )
        H.run(args)
        m = json.loads((a.out_dir / c / H.ARMS["hazard"] / "seed42" / "metrics.json").read_text())
        k3 = json.loads((MAIN / c / H.ARMS["hazard"] / "seed42" / "metrics.json").read_text())
        k3a = json.loads((MAIN / c / H.ARMS["bce"] / "seed42" / "metrics.json").read_text())
        need = np.array(k3["baseline"]["intercept_init"]["phase_offsets"])
        got = np.array(m["baseline"]["final_phase_offsets"])
        print(
            f"{c}: zero-init offsets after training {np.round(got, 3).tolist()}  (base-rate offsets {np.round(need, 3).tolist()}; largest |move| {np.abs(got).max():.3f})"
        )
        print(
            f"   test AUC: zero-init K3 {m['test']['hazard']['auc']:.4f}   K3 {k3['test']['hazard']['auc']:.4f}   K3a {k3a['test']['hazard']['auc']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
