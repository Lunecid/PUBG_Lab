# pubg_lab.analysis.network

Responsibility: Network-analysis features, metrics and application orchestration.

Dependency boundary: Scientific datasets and shared infrastructure.

| Component | Responsibility |
|---|---|
| `advanced_graph_report.py` | Advanced graph analysis and paper-style report for PUBG team combat networks. |
| `archetype.py` | Team archetype labeler (v2) |
| `basic_graph_analysis.py` | Basic graph-network analysis for the intermediate PUBG team network dataset. |
| `build_analysis_dataset.py` | Build analysis-ready intermediate datasets from the existing PUBG DB + .pt cache. |
| `community_detect.py` | Graph Community Detection (Chapter 5) |
| `config.py` | 네트워크 분석 설정. |
| `data_loader.py` | 매치 데이터 로더 (v2) |
| `db_loader.py` | DB에서 매치별 교전 이벤트 로드. |
| `export_json.py` | JSON shard exporter (v2) |
| `h1_archetype.py` | H1 (v2) · 외곽 파밍형 vs 교전 참여형 |
| `h2_battle_center.py` | H2 (v2) · 다중 교전 중심 (Battle Center "C") vs 평균 순위 |
| `h3_movement.py` | H3 (v2) · 존버형 vs 활발 이동형 평균 순위 비교 |
| `hypotheses.py` | 가설 검증용 공용 feature 엔진 |
| `metrics.py` | 프레임별 네트워크 지표 계산 (v2) |
| `movement.py` | Team movement / trajectory features (v2) |
| `narrative.py` | Per-frame narrative generation (프레임별 자연어 해설) |
| `phase_h3_analysis.py` | Phase-wise H3 movement hypothesis analysis. |
| `phase_strategy_type_analysis.py` | Phase-wise Hunter/Sniper/Traveler/Camper strategy analysis. |
| `run.py` | Network Analysis 오케스트레이터 (v2) |
| `run_v2.py` | v2 unified pipeline (CLI) |
| `st_hypothesis_analysis.py` | Spatio-temporal hypothesis analysis for PUBG team outcomes. |
| `strategy_type_network_analysis.py` | Network-based Hunter/Sniper/Traveler/Camper strategy-type analysis. |
| `team_centrality.py` | Team-level Centrality Measures (CH3 + CH4 반영) |
| `team_network.py` | Team-level Network Builder |
| `visualize_match.py` | 단일 매치 인터랙티브 HTML 시각화 (v4 · 노드 안정화 + 팀 식별 + 맵 + Pan/Zoom/Rotate) |
| `visualize_summary.py` | 여러 매치 크로스 분석 HTML |

The implementation lives here. Old root paths are compatibility entry points.
Import reusable components by their canonical `pubg_lab` path; invoke applications through `python -m pubg_lab run <command>`.
Listing commands does not import applications, access data, or start services.
Scientific contracts and completed evidence remain under the repository docs and original artifact paths.
