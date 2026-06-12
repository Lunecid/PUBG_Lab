-- E18 (docs/CODE_REVIEW_2026-06-12.md): phase 테이블 복합 인덱스
-- 실행: PG_PASSWORD=... python -c "from stdb.conn import apply_sql_file; apply_sql_file('stdb/migrations/2026-06-12_phase_indexes.sql')"
-- (현 규모 2.2만 매치에선 선택 사항, 데이터 확장 시 권장)

CREATE INDEX IF NOT EXISTS ix_tpt_match_phase
    ON stdb.team_phase_trajectory (match_id, phase_idx);

CREATE INDEX IF NOT EXISTS ix_tpe_match_phase
    ON stdb.team_phase_exposure (match_id, phase_idx);
