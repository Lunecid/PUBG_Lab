"""Reproduce schedule-based patch inference from the saved read-only inventory."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

OUT = (REPO_ROOT) / "reports/erangel_alignment_2026-10-15"
WINDOWS = [
    ("40.2", "2026-03-11T08:30:00+00:00", "2026-04-08T00:00:00+00:00"),
    ("41.1", "2026-04-08T08:30:00+00:00", "2026-05-13T00:00:00+00:00"),
    # Observation bound covers this inventory, not an asserted next-patch date.
    ("41.2", "2026-05-13T08:30:00+00:00", "2026-06-03T00:00:00+00:00"),
]


def main():
    counts = Counter()
    grouped = defaultdict(Counter)
    rows = list(csv.DictReader((OUT / "erangel_match_inventory.csv").open()))
    for row in rows:
        patch = "missing_start"
        if row["start_at"]:
            time = datetime.fromisoformat(row["start_at"])
            patch = "maintenance_or_unassigned"
            for label, start, end in WINDOWS:
                if datetime.fromisoformat(start) <= time < datetime.fromisoformat(end):
                    patch = label
                    break
        row["inferred_patch"] = patch
        counts[patch] += 1
        grouped[row["match_type"]][patch] += 1
    with (OUT / "erangel_match_patch_inventory.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "patch_assignment.json").write_text(
        json.dumps(
            dict(
                basis="inferred PC schedule; not exact build identity",
                all_counts=counts,
                by_match_type=grouped,
                windows_utc=WINDOWS,
                last_window_end="inventory observation bound, not asserted next-patch date",
                sources=[
                    "https://pubg.com/en/news/9809",
                    "https://www.pubg.com/en-asia/news/9926",
                    "https://www.pubg.com/en/news/10080",
                ],
            ),
            indent=2,
        )
        + "\n"
    )
    print(dict(counts))


if __name__ == "__main__":
    main()
