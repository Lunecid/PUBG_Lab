# projects.sac2027.release

Release verification and packaging.

| Component | Purpose |
|---|---|
| `sac2027_package.py` | Package only a verified paper; rebuild the standalone source before delivery. |
| `sac2027_patch_validate.py` | Validate the new split, all rebuilt labels, and raw-prefix replay with train-only schedules. |
| `sac2027_release_summary.py` | Summarize completed, immutable v14 analysis outputs for the paper handoff. |
| `sac2027_verify_paper.py` | Verify the completed v14 source/PDF and trace its experimental inputs. |
| `sac2027_verify_results.py` | Completion and selection audit for the fixed v14 experiment plan. |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
