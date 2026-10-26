# pubg_lab.collectors.pubg

Responsibility: Explicitly invoked data-collection applications.

Dependency boundary: Operational API and DB configuration; never started by package discovery.

| Component | Responsibility |
|---|---|
| `config.py` | 공통 설정 및 DB 연결. |
| `fetch_match_ids.py` | 01 — /samples에서 match_id 수집 → collection_log에 저장. |
| `fetch_matches.py` | 02 — /matches에서 매치 상세 수집 → matches, rosters, participants 적재. |
| `fetch_telemetry.py` | 03 — 텔레메트리 JSON 다운로드 → telem_* 테이블 적재. |

The implementation lives here. Old root paths are compatibility entry points.
Import reusable components by their canonical `pubg_lab` path; invoke applications through `python -m pubg_lab run <command>`.
Listing commands does not import applications, access data, or start services.
Scientific contracts and completed evidence remain under the repository docs and original artifact paths.
