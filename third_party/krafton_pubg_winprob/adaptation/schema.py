"""Feature schema, item / weapon dictionaries and constants of the adaptation.

Everything that upstream defines (feature list, helmet / vest / backpack / throwable / aid item ids, weapon
classes, sampling schedule, scales) is read from the vendored upstream files, so there is one source of truth.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # third_party/krafton_pubg_winprob
_FE = ROOT / "data_generation" / "feature_engineering"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _list_assignments(path: Path) -> dict[str, list]:
    """Top-level ``NAME = [literal, ...]`` assignments of a python file (no import, so no pyspark / pandas)."""
    out: dict[str, list] = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                val = ast.literal_eval(node.value)
            except Exception:
                continue
            if isinstance(val, list):
                out[node.targets[0].id] = val
    return out


_VARS = _load_module(_FE / "variables.py", "krafton_variables")  # plain dicts, no imports
_ATT = _list_assignments(_FE / "attach_score.py")

HELMET_ITEMS: dict[str, int] = dict(_VARS.HELMET_ITEMS)
VEST_ITEMS: dict[str, int] = dict(_VARS.VEST_ITEMS)
BACKPACK_ITEMS: dict[str, int] = dict(_VARS.BACKPACK_ITEMS)
THROWABLE_ITEMS: dict[str, str] = dict(_VARS.THROWABLE_ITEMS)
AIDS_ITEMS: dict[str, str] = dict(_VARS.AIDS_ITEMS)

# weapon classes: 'Item_Weapon_AK47_C'.split('_')[2] -> 'AK47' (extracting_pgc_features.py, td_*_WEAPONS)
WEAPON_CLASSES = ("AR", "SR", "DMR", "SMG", "SHOTGUN")
WEAPON_LISTS: dict[str, list[str]] = {
    "AR": [a.split("_")[2] for a in _ATT["AR_WEAPONS"]],
    "SR": [a.split("_")[2] for a in _ATT["SR_WEAPONS"]],
    "DMR": [a.split("_")[2] for a in _ATT["DMR_WEAPONS"]],
    "SMG": [a.split("_")[2] for a in _ATT["SMG_WEAPONS"]],
    "SHOTGUN": [a.split("_")[2] for a in _ATT["SHOTGUN_WEAPONS"]],
}
WEAPON_TO_CLASS: dict[str, str] = {w: c for c, ws in WEAPON_LISTS.items() for w in ws}
# damage causers that are never counted in the by-weapon features (feature_dmg_by_weapon)
EXCLUDED_CAUSER_PATTERNS = ("BP_", "Proj", "Debuff", "Fire")
# pickups / drops of these sub categories are removed (parse_pickup_item_seq -> remove_item_log) and the
# start parachute is removed from the pickups only
REMOVED_SUBCATEGORIES = ("Shirts", "Mask", "Gloves", "Bottoms", "Footwear", "Eyewear", "Jacket")
START_PARACHUTE = "StartParachutePack"

# --------------------------------------------------------------------------- upstream constants
POSITION_SCALE = 80000.0  # dataset_v2.py
ZONE_SCALE = 800000.0
TARGET_SCALE = 2000.0
NEAR_RADIUS_CM = 150.0 * 100.0  # parse_num_near_players_seq_vectorized(d_meter=150), coordinates in cm
# adaptation: upstream treats a zone radius <= 0 as "no zone" (distance -1 / NaN -> 0).  The game state of this database
# ends with a degenerate final circle of radius 2.31 / 7.75 cm (the 23 m circle x 1e-3); distance / radius then reaches
# 1e5 - 1e6 and wrecks the z-scoring (std of dist_from_whitezone 355 on Erangel, max |z| 240-375) and the raw 57-d copy.
# A radius below 1 m is treated as "no zone" exactly like upstream's radius <= 0.
MIN_ZONE_RADIUS_CM = 100.0

# KRAFTON's sampling schedule (extracting_pgc_features.py, SquadFeatureExtractor.__init__)
PHASE_CONFIG = {
    1: (0, 600, 10), 2: (600, 810, 11), 3: (810, 990, 9), 4: (990, 1170, 18), 5: (1170, 1350, 18),
    6: (1350, 1530, 18), 7: (1530, 1680, 30), 8: (1680, 1800, 24), 9: (1800, 1970, 34), 10: (1970, 2000, 6),
}  # fmt: skip
PHASE_BOUNDARY_POINTS = (599, 601, 809, 811, 989, 991, 1169, 1171, 1349, 1351, 1529, 1531, 1679, 1681)
assert sum(n for _, _, n in PHASE_CONFIG.values()) == 178 and len(PHASE_BOUNDARY_POINTS) == 14

# maps: upstream embeds 4 maps (erangel, miramar, taego, rondo); the cohorts of this study need three more
MAPS = ("erangel", "miramar", "taego", "rondo", "sanhok", "karakin", "paramo")
COHORT_MAP = {
    "erangel": "erangel", "miramar_desert": "miramar", "taego": "taego", "rondo": "rondo", "sanhok": "sanhok",
    "karakin": "karakin", "paramo": "paramo",
}  # fmt: skip


def cohort_map(cohort: str) -> str:
    """'erangel_tpp_squad' -> 'erangel', 'miramar_desert_tpp_squad' -> 'miramar_desert' -> 'miramar'."""
    stem = cohort.rsplit("_", 2)[0]
    return COHORT_MAP[stem]


def load_feature_list() -> list[str]:
    """CONTINUOUS_FEATURES of upstream (55 names, the CSV columns that dataset_v2 standardises)."""
    return list(_list_assignments(ROOT / "src" / "data" / "continuous_features.py")["CONTINUOUS_FEATURES"])


FEATURES: list[str] = load_feature_list()
assert len(FEATURES) == 55, len(FEATURES)
F_IDX = {n: i for i, n in enumerate(FEATURES)}
V2_COMPUTED = ("dist_from_bluezone_v2", "dist_from_whitezone_v2")  # appended, unscaled, by dataset_v2 -> 57-d
N_INPUT = len(FEATURES) + len(V2_COMPUTED)  # 57

# --------------------------------------------------------------------------- parity table
# status: exact = same definition computed from the same event type of this database
#         proxy = same quantity approximated from a different / coarser signal (stated in ``note``)
#         zero  = cannot be built from the available tables; the column is filled with 0
PARITY: dict[str, tuple[str, str, str]] = {}


def _p(names, status, source, note=""):
    for n in ([names] if isinstance(names, str) else names):
        PARITY[n] = (status, source, note)


_p("n_pickup", "exact", "telem_item_pickup event_type='pickup'", "LogItemPickup only; clothing sub categories and StartParachutePack removed as upstream; carepackage / trunk / custom pickups excluded as upstream")
_p("n_drop", "exact", "telem_item_pickup event_type='drop'", "LogItemDrop (the collector stores drops in the pickup table); clothing sub categories removed as upstream")
_p("n_kill", "exact", "telem_kills.killer_id", "kill_time of the killer, window [0, t)")
_p("n_finish", "exact", "telem_kills.finisher_id", "window [0, t)")
_p("n_groggy", "exact", "telem_groggy.attacker_id", "LogPlayerMakeGroggy by the attacker, window [0, t)")
_p(["n_dmg", "sum_dmg"], "exact", "telem_damage attacker_id, damage>0", "all causers, window (0, t); includes friendly / self damage as upstream")
_p([f"n_dmg_by_{c}" for c in WEAPON_CLASSES] + [f"sum_dmg_by_{c}" for c in WEAPON_CLASSES], "exact",
   "telem_damage attacker_id, damage_causer", "causer 'WeapAK47_C' -> 'AK47' -> class by the upstream weapon lists; BP_/Proj/Debuff/Fire causers excluded as upstream")  # fmt: skip
_p(["n_take_dmg", "sum_take_dmg"], "exact", "telem_damage victim_id, damage>0", "includes environment / blue-zone damage (attacker NULL), window (0, t)")
_p("n_heal", "exact", "telem_item_use item_id in (Bandage, FirstAid, MedKit)", "upstream also reads LogItemUse for these counts")
_p("sum_heal_amount", "zero", "-", "needs LogHeal.healAmount, which the database does not hold")
_p(["n_heal_bandage", "n_heal_firstaid", "n_heal_medkit", "n_heal_painkiller", "n_heal_energydrink", "n_heal_adrenaline"],
   "exact", "telem_item_use item_id", "upstream reads LogItemUse for these counts")  # fmt: skip
_p("n_vehicle_ride", "proxy", "telem_positions.vehicle_type", "no LogVehicleRide / LogVehicleLeave: count of 10 s position samples that enter a vehicle (TransportAircraft excluded); rides shorter than the sampling interval are missed")
_p(["n_teammates", "n_enemies"], "proxy", "telem_positions xyz", "alive players within 150 m (3-D, cm) at the latest position sample <= t (10 s sampling; upstream forward-fills a 10 s grid built from every event type); teammate count includes the player itself as upstream")
_p(["dist_from_bluezone", "dist_from_whitezone"], "proxy", "telem_positions + telem_game_states", "ratio at the latest position sample and the latest game state <= t (upstream uses the zone at the time of the position sample); mean over alive members, -1 if none")
_p(V2_COMPUTED, "exact", "telem_positions + telem_game_states", "squad centre (mean xy of alive members) to zone centre / zone radius; -1 if no alive member with a position")
_p("is_vehicle_count", "proxy", "telem_positions.vehicle_type", "alive members whose latest position sample (<= t) carries a vehicle (TransportAircraft excluded)")
_p(["item_helmet_level", "item_vest_level", "item_backpack_level"], "exact", "telem_item_equip event_type='equip'", "level of the latest equip event, upstream item-id dictionaries, no un-equip handling as upstream; mean over alive members")
_p(["item_throwables_count", "item_grenade_count", "item_smoke_count", "item_flashbang_count", "item_molotov_count", "item_bluezone_grenade_count"],
   "proxy", "telem_item_pickup pickup - drop", "no LogPlayerUseThrowable: thrown items are never subtracted, so counts are upper bounds; sum over alive members")  # fmt: skip
_p(["item_aids_count", "item_bandage_count", "item_firstaid_count", "item_medkit_count", "item_painkiller_count", "item_energydrink_count", "item_adrenaline_count"],
   "proxy", "telem_item_pickup pickup - drop, telem_item_use", "max(0, picked up - dropped - used) per item id; upstream also subtracts LogHeal events (absent)")  # fmt: skip
_p("squad_total_health", "proxy", "telem_positions.health", "sum of the latest sampled health (<= t) over alive members (upstream reads health from every event type)")
_p("squad_alive_count", "exact", "participants + telem_kills (+ time_survived fallback)", "member alive at t from the records before t (v2 dump, phase_v2.alive): a kill ends a life until a return, inferred from the first position sample with health > 0 at least 20 s after the kill and before the next kill (no revive / redeploy events); v1 dump: last death time (dump rule) later than t, revivals not modelled, consistent with team_death_s")
_p("squad_non_groggy_count", "proxy", "telem_groggy victim events + telem_positions.health", "alive member is groggy if the latest position sample has health 0 or a knock event follows it (no LogPlayerRevive: recovery is seen only through health > 0 in a later sample)")
_p("squad_out_bluezone_count", "proxy", "telem_positions + telem_game_states", "alive members outside the current safe circle (safe_zone_*), geometry instead of the isInBlueZone flag")
for _f in FEATURES:
    assert _f in PARITY, f"parity entry missing for {_f}"
assert set(PARITY) == set(FEATURES), set(PARITY) ^ set(FEATURES)
