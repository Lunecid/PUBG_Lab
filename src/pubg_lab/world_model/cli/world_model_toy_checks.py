"""Hand-checkable unit/time/label examples; no DB and no training."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))
import json
import numpy as np
from pubg_lab.world_model.patches import at_time


def main():
    xy_cm = np.array([20100.0, 9900.0])
    xy_m = xy_cm / 100
    cell = np.floor(xy_m / 100).astype(int)
    np.testing.assert_array_equal(cell, [2, 0])
    distance = float(np.linalg.norm(np.array([3.0, 1.0])))
    assert np.isclose(distance, np.sqrt(10))
    t0 = 110.0
    observations = np.array([98.0, 108.2, 115.0])
    chosen = observations[np.searchsorted(observations, t0, side="right") - 1]
    assert chosen == 108.2 and chosen <= t0
    ranks = [4 - e + 1 for e in (1, 2, 3)]
    assert ranks == [4, 3, 2]
    assert at_time("2026-04-08T08:59:59+09:00") == "40.2"
    assert at_time("2026-04-08T09:00:00+09:00") is None
    print(
        json.dumps(
            {
                "xy_m": xy_m.tolist(),
                "cell": cell.tolist(),
                "distance_m": distance,
                "asof_source_s": chosen,
                "age_s": round(t0 - chosen, 6),
                "eliminated_team_ranks": ranks,
                "winner_rank": 1,
                "checks": "passed",
            }
        )
    )


if __name__ == "__main__":
    main()
