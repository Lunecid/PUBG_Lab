# pubg_lab.analysis.network.legacy

Responsibility: Network-analysis features, metrics and application orchestration.

Dependency boundary: Scientific datasets and shared infrastructure.

| Component | Responsibility |
|---|---|
| `h1_analysis.py` | H1 · Bias-controlled combat engagement vs placement |
| `h1_report.py` | H1 · Report writer |
| `h2_analysis.py` | H2 · 교전 밀집도와 최종 순위의 관계 |
| `h3_analysis.py` | H3 · 네트워크 위치(외곽 vs 중심)와 생존의 관계 |
| `report.py` | Unified network analysis report (H1 · H2 · H3) |
| `run_h1.py` | H1 CLI · Bias-controlled combat engagement vs placement |
| `run_report.py` | Unified report CLI (H1 · H2 · H3) |

The implementation lives here. Old root paths are compatibility entry points.
Import reusable components by their canonical `pubg_lab` path; invoke applications through `python -m pubg_lab run <command>`.
Listing commands does not import applications, access data, or start services.
Scientific contracts and completed evidence remain under the repository docs and original artifact paths.
