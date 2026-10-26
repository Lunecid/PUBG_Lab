"""Reproduce G3 figures and aggregate-only evidence notebook from saved runs."""
import argparse
import json
import sqlite3
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run, out = Path(a.run), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    read = lambda name: json.loads((run / name).read_text())
    idx = read("index_02/summary.json")
    short = read("short_transport_01/summary.json")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    colors = ["#21618c", "#d17a22"]
    cohorts = ["normal_br", "ranked_br", "airoyale", "tutorial", "training"]
    labels = ["Normal BR", "Ranked BR", "AI Royale", "Tutorial", "Training"]
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    x = np.arange(5)
    for i, role in enumerate(("train", "validation")):
        counts = [next(r["matches"] for r in idx["cohorts"]
                       if r["role"] == role and r["cohort"] == c) for c in cohorts]
        bars = ax.bar(x + (i-.5)*.37, counts, .37, color=colors[i],
                      label="40.2 train" if i == 0 else "41.1 development")
        ax.bar_label(bars, padding=3, fmt="{:,.0f}")
    ax.set(xticks=x, xticklabels=labels, ylabel="Matches",
           title="14,593 allowed matches classified; primary BR keeps normal + ranked",
           ylim=(0, 6400))
    ax.legend(frameon=False)
    fig.savefig(out/"cohorts.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    kinds = ["emergency_pickup", "reentry_direct", "reentry_inferred"]
    for i, role in enumerate(("train", "validation")):
        values = [sum(r["supported30_events"] for r in idx["rare_support"]
                      if r["role"] == role and r["kind"] == k
                      and r["cohort"] in ("normal_br", "ranked_br")) for k in kinds]
        bars = axes[0].bar(np.arange(3)+(i-.5)*.37, values, .37, color=colors[i],
                          label="Train" if i == 0 else "Development")
        axes[0].bar_label(bars, padding=3, fmt="{:,.0f}", fontsize=9)
        r = next(r for r in short["strata"]
                 if r["role"] == role and r["kind"] == "reentry_direct")
        vals = [r["actor_events"], r["first20_complete"], r["first30_complete"]]
        bars = axes[1].bar(np.arange(3)+(i-.5)*.37, vals, .37, color=colors[i])
        axes[1].bar_label(bars, padding=3, fmt="{:,.0f}", fontsize=9)
    axes[0].set(xticks=range(3), xticklabels=["Emergency\npickup", "Direct\nreentry", "Inferred\nreentry"],
                ylabel="Actor-event links", ylim=(0, 25500),
                title="Main tier: first 30 s labels complete")
    axes[0].legend(frameon=False)
    axes[1].set(xticks=range(3), xticklabels=["Any valid\nhorizon", "First 20 s\ncomplete", "First 30 s\ncomplete"],
                ylabel="Actor-event links", ylim=(0, 4200),
                title="Supplement: actual redeploy-aircraft target")
    fig.suptitle("Reentry trajectories are abundant; long aircraft targets remain scarce", fontsize=13)
    fig.savefig(out/"rare_support.png"); plt.close(fig)

    examples = read("prior_verify_02/examples.json")
    e = next(r for r in examples if r["role"] == "validation")
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    cells = np.array(e["cells"], dtype=float)*100+50
    edge_colors = {1: "#287ca4", 2: "#b16828", 3: "#3e8c60"}
    for r in e["historical_edges"]:
        x0,y0,x1,y1 = [r[k]*100+50 for k in ("sx","sy","dx","dy")]
        ax.annotate("", xy=(x1,y1), xytext=(x0,y0),
                    arrowprops=dict(arrowstyle="->", color=edge_colors[r["mode"]],
                                    alpha=.35, lw=.5+min(np.log1p(r["count"])/4,1.5),
                                    linestyle="--", shrinkA=4, shrinkB=4))
    ax.scatter(cells[:,0], cells[:,1], s=30, c="#263746", zorder=3)
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set(xlabel="Map x (m)", ylabel="Map y (m)",
           title=f'{e["map"]}: current team crop, 41.1 development\n'
                 f'{len(e["historical_edges"])} historical mode-edges; '
                 f'{e["exact_patch_edges"]} exact-patch observed edges')
    ax.legend(handles=[Line2D([0],[0], color=edge_colors[i], linestyle="--", label=l)
                       for i,l in [(1,"Past nonvehicle"),(2,"Past land vehicle"),(3,"Past water vehicle")]],
              frameon=False, loc="best")
    fig.text(.5,.005,"Past behavior only; no verified terrain, collision or present traversability.", ha="center", fontsize=9)
    fig.savefig(out/"historical_crop.png", bbox_inches="tight"); plt.close(fig)
    (out/"figure_provenance.json").write_text(json.dumps({
        "historical_crop_selection":"first development example in fixed SHA-order audit",
        "match_id":e["match_id"],"frame":e["frame"],"actor":e["actor"],
        "no_future_crop":True,"geometry_verified":False,
        "cohort_source":"index_02/summary.json",
        "rare_source":["index_02/summary.json","short_transport_01/summary.json"],
        "historical_source":"prior_verify_02/examples.json"}, indent=2))

    query = ("SELECT role,cohort,count(*) AS matches FROM matches "
             "GROUP BY role,cohort ORDER BY role,cohort")
    db = sqlite3.connect(f"file:{run/'index_02/cohort_events.sqlite'}?mode=ro", uri=True)
    rows = [list(r) for r in db.execute(query)]
    db.close()
    code = ("import sqlite3, json\nfrom pathlib import Path\n"
            f"run = Path({str(run)!r})\n"
            "db = sqlite3.connect(f\"file:{run/'index_02/cohort_events.sqlite'}?mode=ro\", uri=True)\n"
            f"query = {query!r}\n"
            "rows = [list(r) for r in db.execute(query)]\n"
            "db.close()\nprint(json.dumps(rows, indent=2))\n")
    # The exact cell was executed above; only aggregate rows are embedded.
    nb = {"nbformat":4,"nbformat_minor":5,
          "metadata":{"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"}},
          "cells":[{"id":"scope","cell_type":"markdown","metadata":{},"source":[
              "# G3 aggregate evidence\nRead-only local audit. No protected details, DB secrets, or model fitting.\n",
              "The primary cohort contains normal_br + ranked_br. Reentry inferred remains a weak label.\n"]},
              {"id":"cohorts","cell_type":"code","metadata":{},"execution_count":1,
               "source":code.splitlines(True),"outputs":[{"output_type":"stream","name":"stdout",
                                                        "text":json.dumps(rows,indent=2)+"\n"}]}]}
    (out/"audit_notebook.ipynb").write_text(json.dumps(nb,indent=2))
    receipt = {"schemaVersion":1,"items":[{"id":"g3","title":"PUBG G3: cohort and rare-movement audit",
        "assumptions":["Patch assignment follows official PC schedules; executable build is unverified.",
                       "Counts are actor-event links, not independent aircraft flights."],
        "queries":[{"id":"cohorts","source":{"label":"G3 frozen local index",
            "files":[{"label":"cohort_events.sqlite"}],"sql":query,
            "filters":["Allowed 40.2 train and 41.1 development only"],
            "caveats":["Protected and reserved matches were not inspected.","Historical behavior does not verify terrain."]},
            "sourceRoles":[{"kind":"file","label":"cohort_events.sqlite","role":"Reproducible cohort counts"}],
            "reportingPeriod":"40.2 train / 41.1 development",
            "columns":["role","cohort","matches"],
            "rows":[dict(zip(["role","cohort","matches"],r)) for r in rows],
            "methods":[{"language":"calculation","code":"Count matches within each (role, cohort); keep original patch roles."}]}]}]}
    for key, title, filename, columns, records in [
        ("rare", "Actual transport label support", "short_transport_01/summary.json",
         ["role","kind","actor_events","first20_complete","first30_complete"],
         short["strata"]),
        ("ready", "Deduplicated final samples", "ready_01/summary.json",
         ["role","samples"], [{"role":k,"samples":v} for k,v in read("ready_01/summary.json")["role_counts"].items()]),
        ("prior", "Historical support in fixed crop audit", "prior_verify_02/summary.json",
         ["role","sampled_matches","historical_edges","exact_patch_edges"],
         [dict(role=k,**v) for k,v in read("prior_verify_02/summary.json")["roles"].items()])
    ]:
        receipt["items"][0]["queries"].append({
            "id":key,"source":{"label":title,"files":[{"label":filename}],
                              "caveats":["No protected match detail inspected; no new model training.",
                                         "Event support and historical edges are not accuracy estimates."]},
            "sourceRoles":[{"kind":"file","label":filename,"role":"Saved verification aggregate"}],
            "reportingPeriod":"40.2 train / 41.1 development",
            "columns":columns,"rows":[{c:r[c] for c in columns} for r in records],
            "methods":[{"language":"calculation","code":"Read named fields from saved summary; do not pool event links as independent matches."}]
        })
    (out/"sources.json").write_text(json.dumps(receipt,indent=2))
    print(json.dumps({"figures":3,"notebook":"audit_notebook.ipynb","receipt":"sources.json"}))


if __name__ == "__main__":
    main()
