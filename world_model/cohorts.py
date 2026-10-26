"""G3 cohort and event index. Future event selection is NEVER encoder input.

Grounding: frozen preprocess.actor_timeline/targets_at; G3 design document.
Official map names: pubg/api-assets dictionaries/telemetry/mapName.json.
"""
import math
import json
import csv
from pathlib import Path
import numpy as np
from .inputs import load_sample
from .indexed_dataset import IndexedWorldDataset

PRIMARY = ("normal_br", "ranked_br")
BR_MODES = {"solo", "duo", "squad", "solo-fpp", "duo-fpp", "squad-fpp"}


class CohortDataset(IndexedWorldDataset):
    """Frozen banks, but vocabulary and all spatial support fit this cohort only."""
    def __init__(self, root, manifest, role="train", view="observer"):
        if role not in ("train", "validation"):
            raise PermissionError("Protected/reserved loader prohibited")
        self.root = Path(root)
        self.plan = json.loads(Path(manifest).read_text())
        original = json.loads((self.root/"ablations/full.json").read_text())
        for r in ("train", "validation"):
            if not set(self.plan[r+"_match_ids"]) <= set(original[r+"_match_ids"]):
                raise PermissionError("Cohort crosses frozen role boundary")
        self.match_ids = self.plan[role+"_match_ids"]
        self.fit_ids = set(self.plan["train_match_ids"])
        self.metadata = {r["match_id"]: r for r in csv.DictReader((self.root/"matches.csv").open())}
        if any(self.metadata[m]["role"] != role for m in self.match_ids):
            raise PermissionError("Role metadata conflict")
        self.view, self.ablation = view, self.plan["cohort"]
        self.bank_cache, self.spatial_cache = {}, {}
        tokens = set()
        for mid in sorted(self.fit_ids):
            tokens.update(json.loads((self.root/"meta"/f"{mid}.json").read_text())["item_vocabulary_local"])
        self.item_vocabulary = {v: i+2 for i, v in enumerate(sorted(tokens))}


def classify_match(row):
    m, typ = row["map"], row["match_type"]
    if m == "Tutorial_Main" or typ == "tutorialatoz":
        return "tutorial" if m == "Tutorial_Main" and typ == "tutorialatoz" else "unresolved"
    if m == "Range_Main" or typ == "trainingroom":
        return "training" if m == "Range_Main" and typ == "trainingroom" else "unresolved"
    if row["game_mode"] not in BR_MODES or str(row["br_rule_candidate"]) not in ("1", "True"):
        return "unresolved"
    return {"official": "normal_br", "competitive": "ranked_br", "airoyale": "airoyale"}.get(typ, "unresolved")


def movement_events(bank, meta):
    events = []
    for actor, start, end, code, evidence in meta["issues"]:
        if code not in ("reentry_direct", "reentry_inferred"):
            continue
        actor = int(actor)
        k = min(int(np.searchsorted(bank["times"], start)), len(bank["times"])-1)
        stop = k
        episode = int(bank["episode"][k, actor])
        while (stop+1 < len(bank["times"]) and bank["mobility"][stop+1, actor] == 5
               and bank["episode"][stop+1, actor] == episode):
            stop += 1
        events.append(dict(actor=actor, team=int(bank["team"][actor]), kind=code,
                           start_s=float(start), end_s=max(float(start), float(bank["times"][stop])),
                           evidence=evidence, source_time=float(start), event_episode=episode))
    for actor in range(len(bank["team"])):
        rows = np.flatnonzero((bank["mobility"][:, actor] == 4) & bank["valid"][:, actor])
        runs = []
        for k in rows:
            if (not runs or k != runs[-1][-1]+1
                    or bank["episode"][k, actor] != bank["episode"][runs[-1][-1], actor]):
                runs.append([])
            runs[-1].append(int(k))
        for run in runs:
            events.append(dict(actor=actor, team=int(bank["team"][actor]), kind="emergency_pickup",
                               start_s=float(bank["times"][run[0]]), end_s=float(bank["times"][run[-1]]),
                               evidence="recorded_vehicle_mode", source_time=float(bank["source_time"][run[0], actor]),
                               event_episode=int(bank["episode"][run[0], actor])))
    return sorted(events, key=lambda e: (e["start_s"], e["actor"], e["kind"]))


def label_support(b, k, actor):
    members = np.flatnonzero((b["team"] == b["team"][actor]) & b["valid"][k] & b["known"][k])
    if actor not in members:
        return None
    q = k+np.arange(1, 7); qc = np.minimum(q, len(b["times"])-1)
    valid = (b["target_valid"][qc][:, members] & (q < len(b["times"]))[:, None]
             & (b["target_episode"][qc][:, members] == b["episode"][k, members])
             & (b["target_segment"][qc][:, members] == b["segment"][k, members]))
    ego = int(np.flatnonzero(members == actor)[0])
    return dict(ego_points=int(valid[:, ego].sum()), ego_first30=bool(valid[:3, ego].all()),
                ego_full60=bool(valid[:, ego].all()), joint_full60=bool(valid.all()),
                focal_members=len(members), team_points=int(valid.sum()),
                rare_mode_points=int(((b["target_mode"][qc][:, members] == 4)
                                      | (b["target_mode"][qc][:, members] == 5))[valid].sum()))


def candidate_windows(bank, event):
    lo = max(12, math.ceil((event["start_s"]-60)/10))
    hi = min(len(bank["times"])-1, math.floor((event["end_s"]+120)/10))
    for k in range(lo, hi+1):
        support = label_support(bank, k, event["actor"])
        if support is None:
            continue
        t = bank["times"][k]
        stage = "pre" if t < event["start_s"] else "during" if t <= event["end_s"] else "post"
        if stage != "pre" and bank["episode"][k, event["actor"]] != event["event_episode"]:
            continue
        yield {"frame": k, "stage": stage, **support}


def load_at(dataset, mid, frame, actor, vocabulary):
    """Temporary single anchor overlay; no changes to frozen files or metadata."""
    bank = dataset.bank(mid)  # enforces role/protected boundary
    if frame < 12 or frame >= len(bank["times"]) or not bank["valid"][frame, actor]:
        raise ValueError("Not an eligible current input")
    overlay = dict(bank)
    overlay["anchors"] = np.asarray([[frame, actor]], np.int32)
    dataset.bank_cache[mid] = overlay
    try:
        sample = load_sample(dataset, mid, 0, vocabulary)
        sample.provenance["anchor_index"] = None
        sample.provenance["anchor_source"] = "g3_10s_overlay"
        return sample
    finally:
        dataset.bank_cache[mid] = bank
