# projects.collector

collector package.

| Component | Purpose |
|---|---|
| `config.py` | 공통 설정 및 DB 연결. |
| `fetch_match_ids.py` | 01 — /samples에서 match_id 수집 → collection_log에 저장. |
| `fetch_matches.py` | 02 — /matches에서 매치 상세 수집 → matches, rosters, participants 적재. |
| `fetch_telemetry.py` | 03 — 텔레메트리 JSON 다운로드 → telem_* 테이블 적재. |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
