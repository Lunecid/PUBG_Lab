"""Official PC maintenance-window inference, not executable build verification.

Sources checked 2026-10-20:
https://pubg.com/en/news/9809 (40.2)
https://www.pubg.com/en-asia/news/9926 (41.1)
https://www.pubg.com/en/news/10080 (41.2)
https://www.pubg.com/en/news/10179 (42.1 boundary)
Maintenance periods and matches straddling a release are never forced into a patch.
"""
from datetime import datetime, timezone

PATCH_WINDOWS = [
    {"patch": "40.2", "start": "2026-03-11T08:30:00+00:00",
     "end": "2026-04-08T00:00:00+00:00", "source": "https://pubg.com/en/news/9809"},
    {"patch": "41.1", "start": "2026-04-08T08:30:00+00:00",
     "end": "2026-05-13T00:00:00+00:00", "source": "https://www.pubg.com/en-asia/news/9926"},
    {"patch": "41.2", "start": "2026-05-13T08:30:00+00:00",
     "end": "2026-06-17T00:00:00+00:00", "source": "https://www.pubg.com/en/news/10080"},
]


def parse(value):
    if not value:
        return None
    t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ValueError("Naive source timestamp; timezone must not be guessed")
    return t.astimezone(timezone.utc)


def at_time(value):
    t = parse(value)
    if t is None:
        return None
    for w in PATCH_WINDOWS:
        if parse(w["start"]) <= t < parse(w["end"]):
            return w["patch"]
    return None


def infer_patch(row):
    if row.get("shard_id", "steam") != "steam":
        return "unknown", "unsupported_platform"
    start = row.get("start_time")
    if not start:
        return "unknown", "missing_match_start"
    patch = at_time(start)
    if patch is None:
        return "unknown", "maintenance_or_outside_verified_schedule"
    created = row.get("created_at")
    if created and at_time(created) != patch:
        return "unknown", "metadata_start_patch_disagreement"
    end = row.get("end_time")
    if end and at_time(end) != patch:
        return "unknown", "match_crosses_patch_or_maintenance_boundary"
    return patch, "schedule_inferred_build_unverified"
