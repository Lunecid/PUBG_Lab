# pubg_lab.world_model.cli

Responsibility: World-model inputs, encoder components, pretraining and owned CLI.

Dependency boundary: Shared infrastructure, typed graph inputs and approved data views.

| Component | Responsibility |
|---|---|
| `audit_world_model_g4.py` | Post-run support and parameter audit; no fitting and no protected data loader. |
| `audit_world_model_inputs.py` | Full allowed-bank audit; never opens protected match banks. |
| `build_world_model_g3.py` | G3 read-only full allowed corpus classification and rare-event index. |
| `build_world_model_historical_prior.py` | Build independent historical evidence from explicitly approved primary fit IDs. |
| `finalize_world_model_g3.py` | Merge tiers without duplicate samples, recheck every short label, smoke inputs. |
| `mine_world_model_short_transport.py` | Supplementary short-flight tier; never relabel as 30s-complete trajectories. |
| `report_world_model_g2.py` | Read-only result diagnostics and standalone scientific figures. |
| `report_world_model_g3.py` | Reproduce G3 figures and aggregate-only evidence notebook from saved runs. |
| `report_world_model_g4.py` | Saved-run diagnostics only; never fit or select checkpoints from development. |
| `report_world_model_preprocess.py` | Readable preprocessing receipt and offline notebook; no model training. |
| `train_world_model_g2.py` | Preregistered bounded CPU pretraining and frozen representation probe. |
| `train_world_model_g4.py` | G4 preregistered CPU pilot; fixed fit universe, sampling and prior ablations. |
| `verify_world_model_encoder.py` | Bounded CPU G1 integration check. No optimizer, training or protected loader. |
| `verify_world_model_g3.py` | All primary selected labels + per-match/kind two-view adapter verification. |
| `verify_world_model_historical_prior.py` | Read-only historical side-channel demonstration on fixed current-state crops. |
| `verify_world_model_preprocess.py` | Offline audit of the completed preprocessing; never opens protected matches. |
| `world_model_preprocess.py` | Versioned full-corpus preprocessing; no optimizer, training, or test loader. |
| `world_model_toy_checks.py` | Hand-checkable unit/time/label examples; no DB and no training. |

The implementation lives here. Old root paths are compatibility entry points.
Import reusable components by their canonical `pubg_lab` path; invoke applications through `python -m pubg_lab run <command>`.
Listing commands does not import applications, access data, or start services.
Scientific contracts and completed evidence remain under the repository docs and original artifact paths.
