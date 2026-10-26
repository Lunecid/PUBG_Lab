"""Re-verify the V3 dynamic-graph signal WITHOUT retraining.

Mines existing eval_test.json phase_metrics across the 3 graph_relation seed
runs and asks: is there ANY (cut_phase x task) regime where the observed graph
consistently beats the edge_shuffle null control? If not, the graph lift is a
smoothing/regularization effect, not relational topology.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

import json
from pathlib import Path
import numpy as np

ROOT = Path("/Users/lunecid/project/PUBG/model_stgcn/checkpoints")
RUNS = {
    "main42": ROOT / "v3_full_graph_relation" / "graph_relation",
    "seed13": ROOT / "v3_graph_relation_seed_repeats" / "seed13" / "graph_relation",
    "seed2026": ROOT / "v3_graph_relation_seed_repeats" / "seed2026" / "graph_relation",
}
# canonical condition -> checkpoint dir name
CONDS = {
    "no_graph": "graph_core_habitat_no_graph",
    "proximity": "graph_core_habitat_proximity",
    "combat": "graph_core_habitat_combat",
    "copressure": "graph_core_habitat_copressure",
    "prox+combat": "graph_core_habitat_proximity_combat",
    "prox+copress": "graph_core_habitat_proximity_copressure",
    "full_observed": "graph_core_habitat_full_observed",
    "node_shuffle": "graph_core_habitat_full_node_shuffle",
    "edge_shuffle": "graph_core_habitat_full_edge_shuffle",
}


def load(run_dir: Path, cond_dir: str):
    f = run_dir / cond_dir / "eval_test.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def overall(metric="hazard_auc"):
    print(f"\n{'=' * 78}\nOVERALL {metric}  (mean across seeds available)\n{'=' * 78}")
    rows = []
    for cond, cdir in CONDS.items():
        vals = []
        for run in RUNS.values():
            d = load(run, cdir)
            if d is not None and metric in d:
                vals.append(d[metric])
        if vals:
            rows.append((cond, np.mean(vals), np.std(vals), len(vals)))
    base = next((m for c, m, s, n in rows if c == "no_graph"), None)
    shuf = next((m for c, m, s, n in rows if c == "edge_shuffle"), None)
    print(f"{'condition':16}{'mean':>9}{'std':>9}{'n':>4}{'  vs no_graph':>14}{'  vs shuffle':>13}")
    for cond, m, s, n in rows:
        dv = f"{m - base:+.4f}" if base is not None else ""
        ds = f"{m - shuf:+.4f}" if shuf is not None else ""
        print(f"{cond:16}{m:>9.4f}{s:>9.4f}{n:>4}{dv:>14}{ds:>13}")


def per_phase(task="hazard", metric="auc"):
    print(f"\n{'=' * 78}\nPER cut_phase: {task}.{metric}  (mean +/- std across seeds)\n{'=' * 78}")
    # collect: cond -> phase -> [values across seeds]
    data: dict[str, dict[int, list]] = {c: {} for c in CONDS}
    pos: dict[int, list] = {}
    nmatch: dict[int, list] = {}
    for cond, cdir in CONDS.items():
        for run in RUNS.values():
            d = load(run, cdir)
            if d is None:
                continue
            for ph, pm in d.get("phase_metrics", {}).items():
                v = pm.get(task, {}).get(metric)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    data[cond].setdefault(int(ph), []).append(v)
                if cond == "no_graph":
                    pos.setdefault(int(ph), []).append(pm.get("hazard", {}).get("pos_rate"))
                    nmatch.setdefault(int(ph), []).append(pm.get("n_matches"))
    phases = sorted(pos.keys())
    focus = ["no_graph", "proximity", "prox+combat", "full_observed", "edge_shuffle"]
    header = f"{'cut_phase':>9}{'pos%':>7}{'#mtch':>7}"
    for c in focus:
        header += f"{c:>14}"
    header += f"{'  obs-shuf':>11}"
    print(header)
    win_obs = 0
    tot = 0
    for ph in phases:
        line = f"{ph:>9}{np.mean(pos[ph]) * 100:>6.1f}%{int(np.mean(nmatch[ph])):>7}"
        obs = shuf = None
        for c in focus:
            vs = data[c].get(ph, [])
            if vs:
                m = np.mean(vs)
                line += f"{m:>9.4f}±{np.std(vs):>3.3f}"
                if c == "full_observed":
                    obs = m
                if c == "edge_shuffle":
                    shuf = m
            else:
                line += f"{'-':>14}"
        if obs is not None and shuf is not None:
            diff = obs - shuf
            line += f"{diff:>+11.4f}"
            tot += 1
            if diff > 0:
                win_obs += 1
        print(line)
    print(f"\n  -> observed beats edge_shuffle in {win_obs}/{tot} cut_phases for {task}.{metric}")


def phase_verdict(task="hazard", metric="auc"):
    """PRIMARY result: per cut_phase, the core 3-way comparison with a
    per-phase significance verdict (seed-paired sign + noise floor)."""
    print(f"\n{'#' * 78}\n# PRIMARY PER-PHASE RESULT  |  {task}.{metric}\n{'#' * 78}")
    core = ["no_graph", "full_observed", "edge_shuffle"]
    # cond -> phase -> {run: value}
    data: dict[str, dict[int, dict[str, float]]] = {c: {} for c in core}
    pos: dict[int, list] = {}
    nmatch: dict[int, list] = {}
    for cond in core:
        cdir = CONDS[cond]
        for run_name, run in RUNS.items():
            d = load(run, cdir)
            if d is None:
                continue
            for ph, pm in d.get("phase_metrics", {}).items():
                v = pm.get(task, {}).get(metric)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    data[cond].setdefault(int(ph), {})[run_name] = v
                if cond == "no_graph":
                    pos.setdefault(int(ph), []).append(pm.get("hazard", {}).get("pos_rate"))
                    nmatch.setdefault(int(ph), []).append(pm.get("n_matches"))
    print(
        f"{'phase':>5}{'pos%':>7}{'#mtch':>7}{'no_graph':>11}{'observed':>11}{'shuffle':>11}"
        f"{'obs-shuf':>10}{'seeds':>7}  verdict"
    )
    for ph in sorted(pos.keys()):
        ng = list(data["no_graph"].get(ph, {}).values())
        ob = data["full_observed"].get(ph, {})
        sh = data["edge_shuffle"].get(ph, {})
        common = sorted(set(ob) & set(sh))
        if not common:
            continue
        ob_v = np.array([ob[r] for r in common])
        sh_v = np.array([sh[r] for r in common])
        diffs = ob_v - sh_v
        mdiff = diffs.mean()
        n_obs_win = int((diffs > 0).sum())
        noise = max(ob_v.std(), sh_v.std(), 1e-4)
        higher = "obs" if "auc" in metric or "spearman" in metric else "obs"
        if n_obs_win == len(common) and mdiff > 2 * noise:
            verdict = "observed WINS (sig)"
        elif n_obs_win == len(common):
            verdict = "observed wins (within noise)"
        elif n_obs_win == 0 and -mdiff > 2 * noise:
            verdict = "SHUFFLE wins (sig)"
        elif n_obs_win == 0:
            verdict = "shuffle wins (within noise)"
        else:
            verdict = "tie/mixed"
        print(
            f"{ph:>5}{np.mean(pos[ph]) * 100:>6.1f}%{int(np.mean(nmatch[ph])):>7}"
            f"{np.mean(ng):>11.4f}{ob_v.mean():>11.4f}{sh_v.mean():>11.4f}"
            f"{mdiff:>+10.4f}{n_obs_win}/{len(common):>2}  {verdict}"
        )


if __name__ == "__main__":
    phase_verdict("hazard", "auc")
    phase_verdict("hazard", "ap")
    phase_verdict("rank", "spearman")
    phase_verdict("top10", "auc")
    print("\n\n----- secondary: pooled-over-all-phases (diagnostic only) -----")
    overall("hazard_auc")
