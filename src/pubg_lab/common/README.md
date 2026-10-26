# pubg_lab.common

Responsibility: Repository paths and compatibility execution only.

Dependency boundary: Python standard library only.

| Component | Responsibility |
|---|---|
| `compat.py` | Compatibility forwarding with one canonical implementation per Python module. |
| `paths.py` | Resolve the workspace explicitly without opening data or private settings. |

The implementation lives here. Old root paths are compatibility entry points.
Import reusable components by their canonical `pubg_lab` path; invoke applications through `python -m pubg_lab run <command>`.
Listing commands does not import applications, access data, or start services.
Scientific contracts and completed evidence remain under the repository docs and original artifact paths.
