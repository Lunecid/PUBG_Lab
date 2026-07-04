"""V2 노드 피처(F6+F7+O6Z → DynamicSTGNN 입력) 배관 검증 — FEATURE_DESIGN_V2 B3/B4.

보장:
1. 스펙 ↔ manifest family 컬럼 1:1 (43+32+9), base 66과 무충돌, 총 150
2. 플래그 off = 기존 66-d 텐서와 바이트 동일 (기존 ckpt 재현성)
3. 플래그 on = 앞 66채널 불변 + 추가 84채널이 테이블 값/scale과 일치,
   NULL→0.0 채움
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import requires_db

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "model_stgcn" / "feature_manifests" / "pubg_representation_v2.json"


def test_v2_specs_match_manifest_families():
    from stdb.queries import (
        PHASE_FEATURE_DIM,
        PHASE_FEATURE_DIM_V2,
        PHASE_NODE_FEATURE_NAMES,
        PHASE_NODE_FEATURE_SPECS_V2,
    )

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    f6 = manifest["families"]["f6_network_phase_role_tier1"]["columns"]
    f7 = manifest["families"]["f7_habitat_pressure_tier1"]["columns"]
    z = manifest["families"]["o6_z_affordance_tier1"]["columns"]

    spec_cols = [col for (_name, col, _scale, _tr) in PHASE_NODE_FEATURE_SPECS_V2]
    assert spec_cols == f6 + f7 + z, "V2 spec column order must mirror manifest F6+F7+O6Z"
    assert len(spec_cols) == 84
    assert PHASE_FEATURE_DIM_V2 == PHASE_FEATURE_DIM + 84 == 150

    overlap = set(spec_cols) & set(PHASE_NODE_FEATURE_NAMES)
    assert not overlap, f"V2 columns collide with base feature names: {overlap}"

    for _name, _col, scale, transform in PHASE_NODE_FEATURE_SPECS_V2:
        assert scale != 0 and transform == "linear"


@pytest.fixture(scope="module")
def v2_match_pair():
    from stdb.queries import fetch_phase_match_tensors, list_phase_eligible_match_ids

    ids = list_phase_eligible_match_ids(None, None, 1)
    if not ids:
        pytest.skip("no phase-eligible matches in stdb")
    base = fetch_phase_match_tensors(ids[0])
    v2 = fetch_phase_match_tensors(ids[0], include_v2_features=True)
    if base is None or v2 is None:
        pytest.skip(f"match {ids[0]} not phase-ready")
    return base, v2


@requires_db
def test_flag_off_unchanged_and_flag_on_extends(v2_match_pair):
    from stdb.queries import PHASE_FEATURE_DIM, PHASE_FEATURE_DIM_V2

    base, v2 = v2_match_pair
    assert base.X.shape[-1] == PHASE_FEATURE_DIM
    assert v2.X.shape[-1] == PHASE_FEATURE_DIM_V2
    assert len(base.feature_names) == PHASE_FEATURE_DIM
    assert len(v2.feature_names) == PHASE_FEATURE_DIM_V2

    np.testing.assert_array_equal(
        v2.X[:, :, :PHASE_FEATURE_DIM], base.X,
        err_msg="V2 플래그가 base 66채널 값을 바꿨다",
    )
    # 텐서의 나머지 구성요소는 플래그와 무관해야 한다
    np.testing.assert_array_equal(v2.A, base.A)
    np.testing.assert_array_equal(v2.y_hazard, base.y_hazard)
    np.testing.assert_array_equal(v2.target_mask, base.target_mask)
    np.testing.assert_array_equal(v2.node_mask, base.node_mask)


@requires_db
def test_v2_channels_match_tables(v2_match_pair):
    from stdb.conn import get_cur
    from stdb.queries import PHASE_FEATURE_DIM, PHASE_NODE_FEATURE_SPECS_V2

    _base, v2 = v2_match_pair
    team_ids = [t for t in v2.team_ids if t]
    phase_indices = [p for p in v2.phase_indices if p]
    team_index = {tid: i for i, tid in enumerate(team_ids)}
    phase_index = {p: t for t, p in enumerate(phase_indices)}

    rows: dict[tuple[int, int], dict] = {}
    with get_cur(dict_cursor=True) as (_conn, cur):
        for table in (
            "stdb.team_phase_network_features",
            "stdb.team_phase_habitat_features",
            "stdb.team_phase_z_affordance_features",
        ):
            cur.execute(
                f"SELECT * FROM {table} WHERE match_id = %s",
                (v2.match_id,),
            )
            for r in cur.fetchall():
                key = (int(r["team_id"]), int(r["phase_idx"]))
                rows.setdefault(key, {}).update(r)

    assert rows, "fixture match must have V2 feature rows"

    checked = 0
    mismatches: list[str] = []
    for (tid, p), row in rows.items():
        if tid not in team_index or p not in phase_index:
            continue
        i, t = team_index[tid], phase_index[p]
        for f, (_name, col, scale, _tr) in enumerate(PHASE_NODE_FEATURE_SPECS_V2):
            got = float(v2.X[t, i, PHASE_FEATURE_DIM + f])
            raw = row.get(col)
            want = 0.0 if raw is None else float(raw) / scale
            if not np.isfinite(want):
                want = 0.0
            if abs(got - want) > max(1e-5, 1e-5 * abs(want)):
                mismatches.append(f"{col}@(team={tid},phase={p}): got={got} want={want}")
            checked += 1
    assert checked > 0
    assert not mismatches, f"{len(mismatches)}/{checked} mismatches, first: {mismatches[:3]}"
