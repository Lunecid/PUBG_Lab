# pubg_lab.spatial

Responsibility: Erangel and Caldera spatial preparation and visualization.

Dependency boundary: Shared workspace paths and explicit data inputs.

The implementation lives here. Old root paths are compatibility entry points.
Import reusable components by their canonical `pubg_lab` path; invoke applications through `python -m pubg_lab run <command>`.
Listing commands does not import applications, access data, or start services.
Scientific contracts and completed evidence remain under the repository docs and original artifact paths.
