#!/usr/bin/env python3
"""Match-offset decomposition of the RQ2 within-phase gaps and a spread check for the team axis (stored seed-42 test
predictions only; post hoc diagnostic).

1. RQ2: PTR-Former (``M11R``) and the reference GBDT (``HGB-cut``) minus phase-adjusted KRAFTON (``KRAFTON-tf-h0``).  KRAFTON's
   objective is normalized within risk sets, so its within-phase deficit should sit in match-phase mean logits and shrink
   after match-phase centring, while the within-match part reflects the prediction horizon.
2. Spread: the team-axis main effect (P2) and PTR minus GBDT keep part of their within-phase gain after centring.  If the
   team axis rescales scores by match context, dividing the centred logits by their match-phase SD (``scaled``) should
   remove that remainder while leaving within-match AUC unchanged.
Uses :class:`tools.sac2027_match_offset.Decomposition` with the opt-in ``scaled`` score and pools 11 cohorts with
DerSimonian-Laird.  No model is retrained.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from model_stgcn.ptr import metrics as M  # noqa: E402
from model_stgcn.ptr.supplement import RiskSetAUC  # noqa: E402
from tools.sac2027_match_offset import BASE, N_BOOT, SCALED, Decomposition  # noqa: E402

OUT = BASE / "control_evaluation/rq2_match_offset"
ARMS = ("M00", "M01", "M10", "M11", "M11R", "HGB-cut", "KRAFTON-tf-h0", "M11R-shuf", "M11RZ-noHab", "M11R-noHab", "HGB-noHab")
NATIVE = "KRAFTON-tf-h0-native"  # the Cox score s itself; p = 1 - exp(-exp(gamma_k + s)) is not linear in s on the logit scale
CONTRASTS = {
    "ptr_minus_krafton_phase": {"M11R": 1.0, "KRAFTON-tf-h0": -1.0},
    "gbdt_minus_krafton_phase": {"HGB-cut": 1.0, "KRAFTON-tf-h0": -1.0},
    "team_main": {"M01": 0.5, "M00": -0.5, "M11": 0.5, "M10": -0.5},
    "ptr_minus_gbdt": {"M11R": 1.0, "HGB-cut": -1.0},
    # v17 additions: centering on native scores, the other primary decisions, and the habitat-block removal
    "ptr_minus_krafton_phase_native": {"M11R": 1.0, NATIVE: -1.0},
    "gbdt_minus_krafton_phase_native": {"HGB-cut": 1.0, NATIVE: -1.0},
    "phase_main": {"M10": 0.5, "M00": -0.5, "M11": 0.5, "M01": -0.5},
    "rel_vs_shuf": {"M11R": 1.0, "M11R-shuf": -1.0},
    "zone_nohab": {"M11RZ-noHab": 1.0, "M11R-noHab": -1.0},
    "hgb_cut_nohab": {"HGB-cut": 1.0, "HGB-noHab": -1.0},
}


def analyze(cohort: str) -> tuple[list[dict], list[dict]]:
    frames, provenance = {}, []
    for arm in ARMS:
        path = BASE / "checkpoints/ptr_v2" / cohort / arm / "seed42" / "pred_test.csv"
        frames[arm] = M.read_predictions(path)
        provenance.append(dict(path=str(path.relative_to(REPO)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    native = frames["KRAFTON-tf-h0"].copy()
    native["p_hazard"] = 1.0 / (1.0 + np.exp(native["rank_score"].to_numpy()))  # rank_score = -s, so logit(p_hazard) = s
    frames[NATIVE] = native
    ref = frames["HGB-cut"]
    for arm, f in frames.items():
        assert f[M.KEY_COLUMNS + ["y"]].equals(ref[M.KEY_COLUMNS + ["y"]]), (cohort, arm)
    seed = int(hashlib.sha256(f"20261008|rq2_offset|{cohort}".encode()).hexdigest()[:8], 16)
    n_matches = ref["match_id"].nunique()
    counts = M.resample_counts(n_matches, N_BOOT, seed)
    point, draws = {}, {}
    for arm, frame in frames.items():
        dec = Decomposition(frame, names=SCALED)
        point[arm] = {m: float(v[0]) for m, v in dec.evaluate(np.ones(n_matches)).items()}
        draws[arm] = dec.evaluate(counts)
        rs = RiskSetAUC(frame)
        point[arm]["within_match"], draws[arm]["within_match"] = rs.evaluate(np.ones(n_matches)), rs.evaluate(counts)
    rows = []
    for name, weights in CONTRASTS.items():
        for m in point[ARMS[0]]:
            if m == "same_match_pair_share" or (NATIVE in weights and m not in ("original_within_phase", "centred_within_phase", "scaled_within_phase", "match_mean_between_match", "within_match")):
                continue
            est = sum(c * point[a][m] for a, c in weights.items())
            d = sum(c * draws[a][m] for a, c in weights.items())
            d = d[np.isfinite(d)]
            rows.append(dict(cohort=cohort, contrast=name, metric=m, estimate=float(est), se=float(d.std(ddof=1)),
                             ci_low=float(np.quantile(d, 0.025)), ci_high=float(np.quantile(d, 0.975))))
    return rows, provenance


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    roles = pd.read_csv(BASE / "cohort_counts.csv")
    cohorts = roles.loc[roles["analysis_role"] == "main", "cohort"].tolist()
    rows, provenance = [], []
    for c in cohorts:
        print("START", c, flush=True)
        r, files = analyze(c)
        rows += r
        provenance += files
        pd.DataFrame(rows).to_csv(OUT / "contrasts_per_cohort.csv", index=False)
    per = pd.DataFrame(rows)
    pooled = []
    for (name, m), g in per.groupby(["contrast", "metric"]):
        dl = M.dersimonian_laird(g["estimate"].to_numpy(), g["se"].to_numpy())
        pooled.append(dict(contrast=name, metric=m, same_sign=int((np.sign(g["estimate"]) == np.sign(dl["theta"])).sum()), **dl))
    pd.DataFrame(pooled).to_csv(OUT / "pooled.csv", index=False)
    (OUT / "complete.json").write_text(json.dumps(dict(cohorts=cohorts, arms=ARMS, contrasts=CONTRASTS, n_boot=N_BOOT,
                                                       note="post hoc diagnostic; seed 42", files=provenance), indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
