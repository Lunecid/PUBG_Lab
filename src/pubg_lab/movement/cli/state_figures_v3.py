"""Evidence-led examples of life reset and transport restoration, fit matches only."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import csv
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from pubg_lab.movement.bank import read_raw
from pubg_lab.movement.bank_v3 import actor_states, MODE_NAMES

DATA = ROOT / "data/processed/heterograph_diffusion_v3"
OLD = ROOT / "data/processed/heterograph_diffusion_v1"
OUT = ROOT / "reports/heterograph_diffusion_experiment_v3"


def main():
    quality = json.loads((DATA / "per_match_quality.json").read_text())
    wanted = {5: "Reentry aircraft", 4: "Emergency pickup", 8: "Motor glider"}
    found = {}
    for mid in sorted(quality):
        if quality[mid]["split"] != "train":
            continue
        needs = [
            mode
            for mode in wanted
            if mode not in found and quality[mid].get("mode_" + MODE_NAMES[mode], 0)
        ]
        if not needs:
            continue
        raw = read_raw(OLD / "raw" / f"{mid}.json.gz")
        for actor in range(len(raw["actors"])):
            states, transitions, _ = actor_states(raw["tables"], actor)
            for mode in needs:
                choices = [
                    r
                    for r in states
                    if r["mode"] == mode and r["valid"] and (mode != 5 or r["episode"] > 0)
                ]
                if choices and mode not in found:
                    t = choices[0]["t"]
                    found[mode] = (mid, actor, t, states, transitions)
        if len(found) == 3:
            break
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), layout="constrained")
    manifest = []
    for col, mode in enumerate([5, 4, 8]):
        mid, actor, start, states, transitions = found[mode]
        window = [r for r in states if start - 60 <= r["t"] <= start + 180]
        ts = np.array([r["t"] for r in window])
        xy = np.array([r["xyz"][:2] for r in window])
        valid = np.array([r["valid"] for r in window])
        ep = np.array([r["episode"] for r in window])
        ms = np.array([r["mode"] for r in window])
        ax = axes[0, col]
        for j in range(1, len(window)):
            if valid[j] and valid[j - 1] and ep[j] == ep[j - 1]:
                ax.plot(xy[j - 1 : j + 1, 0], xy[j - 1 : j + 1, 1], color="#777", lw=1)
        im = ax.scatter(xy[valid, 0], xy[valid, 1], c=ts[valid], cmap="viridis", s=22)
        ax.set_aspect("equal", adjustable="datalim")
        ax.invert_yaxis()
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"{wanted[mode]}\n{mid[:8]} / actor {actor}")
        fig.colorbar(im, ax=ax, label="match time (s)", shrink=0.7)
        ax = axes[1, col]
        z = np.array([r["xyz"][2] for r in window])
        ax.plot(ts, np.where(valid, z, np.nan), "-o", ms=3, color="#167dad")
        ax.set_xlabel("Match time (s)")
        ax.set_ylabel("Recorded height Z (m)")
        ax.grid(alpha=0.2)
        ax = axes[2, col]
        ax.step(ts, np.where(valid, ms, np.nan), where="post", color="#9a5b23")
        ax.scatter(ts[valid], ms[valid], s=18, color="#9a5b23")
        present = sorted(set(ms[valid].tolist()))
        ax.set_yticks(present, [MODE_NAMES[m].replace("_", "\n") for m in present], fontsize=7)
        ax.set_xlabel("Match time (s)")
        ax.grid(alpha=0.2)
        for t, alive, e, _, conf in transitions:
            if ts.min() <= t <= ts.max():
                for ax in axes[1:, col]:
                    ax.axvline(t, color="#2e8d5e" if alive else "#ba3636", ls="--", alpha=0.8)
                axes[1, col].text(
                    t,
                    axes[1, col].get_ylim()[1],
                    f" episode {e}" if alive else " death",
                    rotation=90,
                    va="top",
                    fontsize=8,
                )
        manifest.append(
            {
                "label": wanted[mode],
                "match_id": mid,
                "actor_index": actor,
                "anchor_evidence_s": start,
                "source_split": "train",
                "selection": "first sorted fit match with specified class, first matching actor",
                "transitions": [
                    {"t": t, "alive": alive, "episode": e, "evidence": conf}
                    for t, alive, e, _, conf in transitions
                ],
                "rows": [
                    dict(
                        t=r["t"],
                        x=r["x"],
                        y=r["y"],
                        z=r["z"],
                        health=r["hp"],
                        vehicle_type=r["vehicle_type"],
                        vehicle_class=r["vehicle_class"],
                        episode=r["episode"],
                        mode=r["mode"],
                        valid=r["valid"],
                    )
                    for r in window
                ],
            }
        )
    fig.suptitle(
        "V3 restoration examples | direct vehicle evidence and explicitly inferred exit state",
        fontsize=15,
    )
    fig.supxlabel(
        "No line joins a death/reentry boundary. Height is recorded player altitude, not ground elevation. Exit state is inferred until landing or direct ground/water/mount evidence.\nExamples selected by vehicle class in training data; not an accuracy estimate and no locked-test access.",
        fontsize=10,
    )
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"state_reconstruction.{ext}", dpi=150)
    plt.close(fig)
    (OUT / "state_reconstruction_evidence.json").write_text(json.dumps(manifest, indent=2))
    print(
        [
            {"label": r["label"], "match_id": r["match_id"], "actor": r["actor_index"]}
            for r in manifest
        ]
    )


if __name__ == "__main__":
    main()
