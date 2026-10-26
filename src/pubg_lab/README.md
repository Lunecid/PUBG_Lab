# pubg_lab

Responsibility: Canonical implementation package for the PUBG research monorepo.

| Component | Responsibility |
|---|---|
| `__main__.py` | Discover and explicitly run a registered research application. |

The implementation lives here. Old root paths are compatibility entry points.
Import reusable components by their canonical `pubg_lab` path; invoke applications through `python -m pubg_lab run <command>`.
Listing commands does not import applications, access data, or start services.
Scientific contracts and completed evidence remain under the repository docs and original artifact paths.
