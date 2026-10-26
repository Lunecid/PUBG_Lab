# pubg_lab.repository

Responsibility: Release inventory and static codebase checks.

Dependency boundary: Python standard library and common workspace paths.

| Component | Responsibility |
|---|---|
| `check_structure.py` | Check code ownership and import boundaries without executing research code. |
| `repository_release.py` | Create and verify immutable, local repository inventories; never run experiments. |

The implementation lives here. Old root paths are compatibility entry points.
Import reusable components by their canonical `pubg_lab` path; invoke applications through `python -m pubg_lab run <command>`.
Listing commands does not import applications, access data, or start services.
Scientific contracts and completed evidence remain under the repository docs and original artifact paths.

`check_structure.py` statically validates ownership, canonical imports, library/application direction and domain acyclicity. It does not execute research modules.
