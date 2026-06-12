"""공용 fixture — 실 DB 1매치 기반의 leakage-safety 불변식 검증용.

DB가 없으면 (PG_PASSWORD 미설정 / 접속 실패) DB 의존 테스트는 skip된다:
    PG_PASSWORD=... python -m pytest tests/ -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "legacy"))  # legacy 스택 모듈 (model.*, dataset)


def _db_available() -> bool:
    if not os.environ.get("PG_PASSWORD"):
        return False
    try:
        from stdb.conn import get_cur

        with get_cur() as (_conn, cur):
            cur.execute("SELECT 1")
        return True
    except Exception:
        return False


DB_AVAILABLE = _db_available()

requires_db = pytest.mark.skipif(
    not DB_AVAILABLE, reason="PostgreSQL STDB unavailable (set PG_PASSWORD)"
)


@pytest.fixture(scope="session")
def phase_tensors():
    """phase-eligible 매치 1개의 텐서 (E=3 기본 레이아웃)."""
    from stdb.queries import fetch_phase_match_tensors, list_phase_eligible_match_ids

    ids = list_phase_eligible_match_ids(None, None, 1)
    if not ids:
        pytest.skip("no phase-eligible matches in stdb")
    mt = fetch_phase_match_tensors(ids[0])
    if mt is None:
        pytest.skip(f"match {ids[0]} not phase-ready")
    return mt


@pytest.fixture(scope="session")
def phase_tensors_directed(phase_tensors):
    """같은 매치의 E=4 (combat_in 포함) 텐서."""
    from stdb.queries import fetch_phase_match_tensors

    mt = fetch_phase_match_tensors(phase_tensors.match_id, include_combat_in=True)
    assert mt is not None
    return mt
