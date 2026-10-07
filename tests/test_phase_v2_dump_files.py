"""Checks of the WRITTEN phase-tier dump v2 (``model_stgcn/dumps/bigcomp2027_phase_v2``): schema, population, invariants.

Skipped when the dump has not been produced.  File-level only (no database).  The test splits are loaded in full, train / val
through their manifests (counts) so the test stays fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from model_stgcn import dump_phase_tier as V1
from model_stgcn.phase_v2 import features as FT

DUMP = Path("model_stgcn/dumps/bigcomp2027_phase_v2")
V1_DUMP = Path("model_stgcn/dumps/bigcomp2027_phase_v1")
pytestmark = pytest.mark.skipif(not (DUMP / "population.json").exists(), reason="v2 dump not produced")

FLOAT64_TIME_KEYS = ("phase_start_s", "phase_end_s", "cut_t", "cut_wall_s", "label_window_end_s", "team_death_s")
REQUIRED = (
    "match_id", "match_type", "team_ids", "X", "node_mask", "query_mask", "target_mask_v2", "target_mask", "y_hazard_v2",
    "y_hazard", "team_death_s", "team_won", "cut_mask", "cut_valid", "cut_excluded", "cut_terminal", "label_window_end_s",
    "match_end_s", "phase_mask", "phase_start_s", "phase_end_s", "cut_t", "cut_wall_s", "phase_duration_sched_s", "edges",
    "team_pos", "zone", "zone_valid", "map_size", "final_rank_norm", "top5", "top10", "team_ai_frac",
    "team_has_monster_dmg", "last_phase_idx", "clock_delta_s", "clock_delta_median_s", "node_mask_label",
)  # fmt: skip


def cohorts():
    pop = json.loads((DUMP / "population.json").read_text())
    return [c for c, r in pop["cohorts"].items() if r["status"] == "kept"]


def test_population_is_the_v1_population():
    pop2 = json.loads((DUMP / "population.json").read_text())
    pop1 = json.loads((V1_DUMP / "population.json").read_text())
    assert pop2["cohorts"] == pop1["cohorts"]
    assert len(cohorts()) == 12
    total = 0
    for slug in cohorts():
        man = json.loads((DUMP / slug / "dump_manifest.json").read_text())
        man1 = json.loads((V1_DUMP / slug / "dump_manifest.json").read_text())
        for split, rec in man["splits"].items():
            assert rec["n_skipped"] == 0 and rec["n_dumped"] == rec["n_split_ids"] == man1["splits"][split]["n_split_ids"]
            total += rec["n_dumped"]
    assert total == 13948


@pytest.mark.parametrize("slug", [
    "erangel_tpp_squad", "taego_tpp_squad", "erangel_tpp_solo", "erangel_tpp_duo", "karakin_tpp_squad", "paramo_tpp_squad",
    "miramar_desert_tpp_squad", "rondo_tpp_squad", "sanhok_tpp_squad", "taego_tpp_duo", "taego_fpp_squad", "erangel_fpp_squad",
])  # fmt: skip
def test_test_split_schema_and_invariants(slug):
    d = torch.load(DUMP / slug / "test.pt", weights_only=False)
    meta, matches = d["meta"], d["matches"]
    assert meta["version"] == "v2" and meta["v2"]["version"] == "v2"
    assert list(meta["feature_names"]) == FT.FEATURE_NAMES == V1.FEATURE_NAMES
    assert "team_death_source" not in meta  # would make the v1 loader rebuild the alive mask from the final death time
    assert set(meta["v2"]["changed_features"]["feature_change_class"]) == set(FT.FEATURE_NAMES)
    ids1 = {m["match_id"] for m in torch.load(V1_DUMP / slug / "test.pt", weights_only=False)["matches"]}
    assert {m["match_id"] for m in matches} == ids1
    comeback = 0
    for m in matches:
        missing = [k for k in REQUIRED if k not in m]
        assert not missing, (m["match_id"], missing)
        assert "node_mask_v2" not in m
        for k in FLOAT64_TIME_KEYS:
            assert m[k].dtype == torch.float64, k
        assert m["X"].dtype == torch.float32 and torch.isfinite(m["X"]).all()
        node, tgt, qry = m["node_mask"], m["target_mask_v2"], m["query_mask"]
        assert (qry == (node | tgt)).all() and (m["target_mask"] == tgt).all() and (m["y_hazard"] == m["y_hazard_v2"]).all()
        assert not tgt[7].any() and (tgt <= m["cut_valid"][:, None]).all() and (m["cut_valid"] <= m["cut_mask"]).all()
        assert not m["X"][~m["cut_mask"]].any() and not node[~m["cut_mask"]].any()
        y = m["y_hazard_v2"]
        assert torch.isin(y[tgt], torch.tensor([0.0, 1.0])).all() and not (y[~tgt] != 0).any()
        T = m["team_death_s"]
        b = m["phase_end_s"]
        for k in range(7):
            if m["cut_mask"][k]:
                assert abs(float(m["cut_t"][k]) - (float(b[k]) - 10.0)) < 1e-9 and m["cut_t"][k] > m["phase_start_s"][k] - 20
                assert (T[tgt[k]] > b[k]).all(), "risk set must be {T > b_k}"
                pos = (y[k] > 0.5) & tgt[k]
                assert torch.isfinite(T[pos]).all() and (T[pos] <= m["label_window_end_s"][k]).all()
        assert not (y[:, m["team_won"]] > 0).any() and torch.isinf(T[m["team_won"]]).all()  # winners censored
        comeback += int((tgt & ~node).sum())
    assert comeback == meta["stats"]["comeback_rows"]
    assert meta["stats"]["excluded_match_cuts"] == {} or all(
        r in ("no_match_end", "position_gap") for r in meta["stats"]["excluded_match_cuts"]
    )
