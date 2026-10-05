"""Tests of the KRAFTON baseline adaptation (third_party/krafton_pubg_winprob): feature reconstruction against a
naive per-(squad, time) reference, causality (prefix replay), padding / dead-squad invariance of the patched
backbone, the Cox sign convention, the collate cap and the 57-d input.  No database needed."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

KT = Path(__file__).resolve().parents[1] / "third_party" / "krafton_pubg_winprob"
if str(KT) not in sys.path:
    sys.path.insert(0, str(KT))

from adaptation import schema as S  # noqa: E402
from adaptation.features import compute_features  # noqa: E402

torch = pytest.importorskip("torch")


# --------------------------------------------------------------------------- synthetic match
def make_raw(seed: int, N: int = 6, size: int = 4):
    rng = np.random.RandomState(seed)
    team = np.repeat(np.arange(N), size)
    P = len(team)
    d = rng.uniform(100, 1200, P)
    d[rng.rand(P) < 0.15] = np.inf
    slot = np.full((N, 4), -1, dtype=np.int64)
    for s in range(N):
        mem = np.flatnonzero(team == s)[:4]
        slot[s, : len(mem)] = mem

    def ev(n, hi=1300.0):
        return rng.randint(-1, P, n), np.sort(rng.uniform(0, hi, n))

    kp, kt = ev(60)
    fp, _ = ev(60)
    ga, gt = ev(40)
    gv, _ = ev(40)
    da, dt = ev(300)
    dv, _ = ev(300)
    dmg = np.where(rng.rand(300) < 0.2, 0.0, rng.uniform(1, 40, 300))
    wc = rng.choice(np.array(["", "AR", "SR", "DMR", "SMG", "SHOTGUN"], dtype=object), 300)
    # positions every ~10 s per player, random walk, vehicles, health
    rows = []
    for p in range(P):
        t = np.arange(8.0 + rng.rand() * 3, 1300, 10.0)
        xy = np.cumsum(rng.normal(0, 700, (len(t), 2)), axis=0) + rng.uniform(1e5, 5e5, 2)
        hp = np.where(rng.rand(len(t)) < 0.05, 0.0, rng.uniform(20, 100, len(t)))
        vh = rng.rand(len(t)) < 0.1
        for i in range(len(t)):
            rows.append((p, t[i], hp[i], xy[i, 0], xy[i, 1], rng.uniform(0, 2000), vh[i]))
    rows = np.array(rows)
    rows = rows[rng.permutation(len(rows))]  # unsorted on purpose
    zt = np.arange(10.0, 1300, 10.0)
    zone = {"t": zt, "sx": np.full(len(zt), 3e5), "sy": np.full(len(zt), 3e5), "sr": np.concatenate([np.linspace(5e5, 5e4, len(zt) - 25), np.full(25, 2.31)]),  # degenerate final circle
            "px": np.where(zt < 100, 0.0, 3.2e5), "py": np.where(zt < 100, 0.0, 2.9e5), "pr": np.where(zt < 100, 0.0, np.where(zt > 1000, 7.75, 1.5e5))}  # fmt: skip
    items = list(S.AIDS_ITEMS) + list(S.THROWABLE_ITEMS) + ["Item_Ammo_556mm_C", "Item_Weapon_AK47_C", "Item_Back_B_01_StartParachutePack_C"]
    npick = 400
    pp, pt = ev(npick)
    pitem = rng.choice(np.array(items, dtype=object), npick)
    kind = (rng.rand(npick) < 0.3).astype(np.int8)
    stack = rng.choice([0, 1, 1, 2, 5], npick).astype(float)
    removed = rng.rand(npick) < 0.05
    parachute = np.array(["StartParachutePack" in x for x in pitem])
    eqp, eqt = ev(60)
    eqi = rng.choice(np.array(sorted(set(S.HELMET_ITEMS) | set(S.VEST_ITEMS) | set(S.BACKPACK_ITEMS)) + ["Item_Back_BlueBlocker"], dtype=object), 60)
    up, ut = ev(120)
    ui = rng.choice(np.array(list(S.AIDS_ITEMS), dtype=object), 120)
    raw = {
        "pid": np.array([f"p{i}" for i in range(P)]), "team": team, "d": d, "is_ai": np.zeros(P, bool), "N": N, "slot": slot,
        "kills": {"killer_pidx": kp, "finisher_pidx": fp, "t": kt},
        "groggy": {"attacker_pidx": ga, "victim_pidx": gv, "t": gt},
        "damage": {"attacker_pidx": da, "victim_pidx": dv, "dmg": dmg, "t": dt, "wclass": wc},
        "pos": {"pidx": rows[:, 0].astype(np.int64), "t": rows[:, 1], "health": rows[:, 2], "x": rows[:, 3], "y": rows[:, 4],
                "z": rows[:, 5], "veh": rows[:, 6] > 0.5},
        "zone": zone,
        "pick": {"pidx": pp, "t": pt, "kind": kind, "item_id": pitem, "stack": stack, "removed": removed, "parachute": parachute},
        "equip": {"pidx": eqp, "t": eqt, "item_id": eqi},
        "use": {"pidx": up, "t": ut, "item_id": ui},
    }  # fmt: skip
    return raw


# --------------------------------------------------------------------------- naive reference (upstream semantics)
def naive(raw, t):
    N, P, team, d = raw["N"], len(raw["pid"]), raw["team"], raw["d"]
    F = {n: np.zeros(N) for n in S.FEATURES}
    alive = d > t
    pos = raw["pos"]
    zn = raw["zone"]
    zi = np.searchsorted(zn["t"], t, side="right") - 1
    bz = np.array([zn["sx"][zi], zn["sy"][zi], zn["sr"][zi]])
    wz = np.array([zn["px"][zi], zn["py"][zi], zn["pr"][zi]])
    if (wz == 0).all():
        wz = bz.copy()
    last = {}
    for p in range(P):
        m = (pos["pidx"] == p) & (pos["t"] <= t)
        if m.any():
            i = np.flatnonzero(m)[np.argmax(pos["t"][m])]
            last[p] = (pos["x"][i], pos["y"][i], pos["z"][i], pos["health"][i], bool(pos["veh"][i]), pos["t"][i])
    cnt = lambda a, b, c, strict=True: ((a == c) & (b < t if strict else b <= t)).sum()  # noqa: E731
    per = {}
    for p in range(P):
        k, f, g = raw["kills"], raw["kills"], raw["groggy"]
        dm, pc, us, eq = raw["damage"], raw["pick"], raw["use"], raw["equip"]
        r = {}
        r["n_kill"] = ((k["killer_pidx"] == p) & (k["t"] < t)).sum()
        r["n_finish"] = ((f["finisher_pidx"] == p) & (f["t"] < t)).sum()
        r["n_groggy"] = ((g["attacker_pidx"] == p) & (g["t"] < t)).sum()
        a = (dm["attacker_pidx"] == p) & (dm["t"] < t) & (dm["t"] > 0) & (dm["dmg"] > 0)
        r["n_dmg"], r["sum_dmg"] = a.sum(), dm["dmg"][a].sum()
        v = (dm["victim_pidx"] == p) & (dm["t"] < t) & (dm["t"] > 0) & (dm["dmg"] > 0)
        r["n_take_dmg"], r["sum_take_dmg"] = v.sum(), dm["dmg"][v].sum()
        for c in S.WEAPON_CLASSES:
            mc = a & (dm["wclass"] == c)
            r[f"n_dmg_by_{c}"], r[f"sum_dmg_by_{c}"] = mc.sum(), dm["dmg"][mc].sum()
        keep = (pc["pidx"] == p) & ~pc["removed"] & (pc["t"] < t)
        r["n_pickup"] = (keep & (pc["kind"] == 0) & ~pc["parachute"]).sum()
        r["n_drop"] = (keep & (pc["kind"] == 1)).sum()
        used = {i: ((us["pidx"] == p) & (us["item_id"] == i) & (us["t"] <= t)).sum() for i in S.AIDS_ITEMS}
        held = {}
        for item in list(S.AIDS_ITEMS) + list(S.THROWABLE_ITEMS):
            sel = (pc["pidx"] == p) & (pc["item_id"] == item) & (pc["t"] <= t)
            st = np.where(pc["stack"] > 0, pc["stack"], 1.0)
            got = st[sel & (pc["kind"] == 0)].sum()
            gone = st[sel & (pc["kind"] == 1)].sum()
            held[item] = max(0.0, got - gone - used.get(item, 0))
        r["n_heal"] = sum(used[i] for i in ("Item_Heal_Bandage_C", "Item_Heal_FirstAid_C", "Item_Heal_MedKit_C"))
        for nm, key in (("bandage", "Item_Heal_Bandage_C"), ("firstaid", "Item_Heal_FirstAid_C"), ("medkit", "Item_Heal_MedKit_C"),
                        ("painkiller", "Item_Boost_PainKiller_C"), ("energydrink", "Item_Boost_EnergyDrink_C"), ("adrenaline", "Item_Boost_AdrenalineSyringe_C")):  # fmt: skip
            r[f"n_heal_{nm}"] = used[key]
            r[f"item_{nm}_count"] = held[key]
        r["item_aids_count"] = sum(held[i] for i in S.AIDS_ITEMS)
        r["item_throwables_count"] = sum(held[i] for i in S.THROWABLE_ITEMS)
        for nm, key in (("grenade", "Item_Weapon_Grenade_C"), ("smoke", "Item_Weapon_SmokeBomb_C"), ("flashbang", "Item_Weapon_FlashBang_C"),
                        ("molotov", "Item_Weapon_Molotov_C"), ("bluezone_grenade", "Item_Weapon_BluezoneGrenade_C")):  # fmt: skip
            r[f"item_{nm}_count"] = held[key]
        for name, table in (("helmet", S.HELMET_ITEMS), ("vest", S.VEST_ITEMS), ("backpack", S.BACKPACK_ITEMS)):
            lv, tl = 0, -1.0
            idx = np.flatnonzero((eq["pidx"] == p) & (eq["t"] <= t) & np.isin(eq["item_id"], list(table)))
            if len(idx):
                j = idx[np.argmax(eq["t"][idx])]
                lv = table[eq["item_id"][j]]
            r[f"item_{name}_level"] = lv
        # rides: samples in a vehicle whose predecessor (same player, by time) was not
        idx = np.flatnonzero(pos["pidx"] == p)
        idx = idx[np.argsort(pos["t"][idx])]
        prev, rides = False, 0
        for j in idx:
            if pos["veh"][j] and not prev and pos["t"][j] <= t:
                rides += 1
            prev = bool(pos["veh"][j])
        r["n_vehicle_ride"] = rides
        per[p] = r
    # snapshot values per alive player
    for p in range(P):
        if not alive[p] or p not in last:
            continue
    near_t, near_e = {}, {}
    ok = [p for p in range(P) if alive[p] and p in last]
    for p in ok:
        near_t[p] = near_e[p] = 0
        for q in ok:
            dist = np.sqrt((last[p][0] - last[q][0]) ** 2 + (last[p][1] - last[q][1]) ** 2 + (last[p][2] - last[q][2]) ** 2)
            if dist <= S.NEAR_RADIUS_CM:
                if team[p] == team[q]:
                    near_t[p] += 1
                else:
                    near_e[p] += 1
    for s in range(N):
        mem = np.flatnonzero(team == s)
        al = [p for p in mem if alive[p]]
        for name in per[mem[0]]:
            if name.startswith(("item_helmet", "item_vest", "item_backpack")):
                continue
            if name.startswith("item_"):
                F[name][s] = sum(per[p][name] for p in al)
            else:
                F[name][s] = sum(per[p][name] for p in mem)
        for name in ("item_helmet_level", "item_vest_level", "item_backpack_level"):
            F[name][s] = np.mean([per[p][name] for p in al]) if al else 0.0
        F["squad_alive_count"][s] = len(al)

        def zd(p, z):
            if p not in last or z[2] < S.MIN_ZONE_RADIUS_CM:
                return 0.0
            return np.hypot(last[p][0] - z[0], last[p][1] - z[1]) / z[2]

        for nm, vals in (("n_teammates", [near_t.get(p, 0) for p in al]), ("n_enemies", [near_e.get(p, 0) for p in al]),
                         ("dist_from_bluezone", [zd(p, bz) for p in al]), ("dist_from_whitezone", [zd(p, wz) for p in al])):  # fmt: skip
            F[nm][s] = np.mean(vals) if al else -1.0
        F["squad_total_health"][s] = sum(last[p][3] if p in last else 100.0 for p in al)
        F["is_vehicle_count"][s] = sum(1 for p in al if p in last and last[p][4])
        out_b = 0
        for p in al:
            if p in last and bz[2] >= S.MIN_ZONE_RADIUS_CM and np.hypot(last[p][0] - bz[0], last[p][1] - bz[1]) > bz[2]:
                out_b += 1
        F["squad_out_bluezone_count"][s] = out_b
        ng = 0
        for p in al:
            if p in last:
                h0 = last[p][3] <= 0
                kn = ((raw["groggy"]["victim_pidx"] == p) & (raw["groggy"]["t"] <= t) & (raw["groggy"]["t"] > last[p][5])).any()
                ng += int(h0 or kn)
            else:
                ng += int(((raw["groggy"]["victim_pidx"] == p) & (raw["groggy"]["t"] <= t)).any())
        F["squad_non_groggy_count"][s] = len(al) - ng
        posp = [p for p in al if p in last]
        if posp:
            cx, cy = np.mean([last[p][0] for p in posp]), np.mean([last[p][1] for p in posp])
            F["dist_from_bluezone_v2"][s] = np.hypot(cx - bz[0], cy - bz[1]) / bz[2] if bz[2] >= S.MIN_ZONE_RADIUS_CM else -1.0
            F["dist_from_whitezone_v2"][s] = np.hypot(cx - wz[0], cy - wz[1]) / wz[2] if wz[2] >= S.MIN_ZONE_RADIUS_CM else -1.0
        else:
            F["dist_from_bluezone_v2"][s] = F["dist_from_whitezone_v2"][s] = -1.0
    return F


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_features_match_naive_reference(seed):
    raw = make_raw(seed)
    ts = np.array([55.0, 130.5, 400.0, 777.7, 1100.0, 1200.0, 1290.0])  # the last two see the degenerate final circle
    out = compute_features(raw, ts)
    for k, t in enumerate(ts):
        ref = naive(raw, t)
        for name in S.FEATURES:
            if name == "sum_heal_amount":
                assert np.all(out["F"][:, k, S.F_IDX[name]] == 0)
                continue
            np.testing.assert_allclose(out["F"][:, k, S.F_IDX[name]], ref[name], rtol=1e-4, atol=1e-3, err_msg=f"{name} t={t} seed={seed}")


def _truncate(raw, t):
    """The raw rows that exist at time t (everything after t removed)."""
    out = dict(raw)
    for key, tk in (("kills", "t"), ("groggy", "t"), ("damage", "t"), ("pos", "t"), ("pick", "t"), ("equip", "t"), ("use", "t")):
        sub = raw[key]
        m = sub[tk] <= t
        out[key] = {k: v[m] for k, v in sub.items()}
    z = raw["zone"]
    mz = z["t"] <= t
    out["zone"] = {k: v[mz] for k, v in z.items()}
    return out


def test_features_are_causal_prefix_replay():
    """Features at t from the rows available at t equal the features from the full log (no look-ahead).
    The death times are left untouched: they only decide who is alive at t (d > t is a statement about the past)."""
    raw = make_raw(7)
    ts = np.array([200.0, 640.0, 1000.0])
    full = compute_features(raw, ts)
    for k, t in enumerate(ts):
        part = compute_features(_truncate(raw, t), np.array([t]))
        np.testing.assert_allclose(full["F"][:, k, :], part["F"][:, 0, :], rtol=1e-5, atol=1e-4)
        np.testing.assert_array_equal(np.isnan(full["POS"][:, k]), np.isnan(part["POS"][:, 0]))


# --------------------------------------------------------------------------- vendored model / loss / collate
def _import_upstream():
    import importlib

    for m in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        del sys.modules[m]
    importlib.invalidate_caches()
    from src.data import dataset_v2
    from src.models.backbone import TransformerBackbone
    from src.training import losses, metrics

    return dataset_v2, TransformerBackbone, losses, metrics


def test_padding_and_dead_squads_do_not_change_alive_outputs():
    dataset_v2, Backbone, _, _ = _import_upstream()
    torch.manual_seed(0)
    bb = Backbone(input_dim=S.N_INPUT, embed_dim=16, num_heads=2, num_layers=2, dropout=0.0, num_maps=len(S.MAPS)).eval()
    n = 3
    x = torch.randn(1, n, S.N_INPUT)
    pos = torch.rand(1, n, 4, 3)
    z = torch.rand(1, 3)
    mp = torch.tensor([2])
    valid = torch.ones(1, n, dtype=torch.bool)
    ref = bb(x, pos, z, z, mp, valid_mask=valid)
    # pad with 13 junk tokens (random features, different positions) that are masked
    xp = torch.cat([x, torch.randn(1, 13, S.N_INPUT) * 5], dim=1)
    pp = torch.cat([pos, torch.rand(1, 13, 4, 3)], dim=1)
    vp = torch.cat([valid, torch.zeros(1, 13, dtype=torch.bool)], dim=1)
    out = bb(xp, pp, z, z, mp, valid_mask=vp)
    assert torch.allclose(ref, out[:, :n], atol=1e-5)
    # without the mask the junk tokens leak (this is the audited upstream behaviour)
    leak = bb(xp, pp, z, z, mp)
    assert not torch.allclose(ref, leak[:, :n], atol=1e-3)


def test_collate_does_not_truncate_beyond_16_squads():
    dataset_v2, *_ = _import_upstream()
    n = 20
    sample = {
        "x_continuous": torch.randn(n, S.N_INPUT), "positions": torch.rand(n, 4, 3), "bluezone_info": torch.rand(3),
        "whitezone_info": torch.rand(3), "y": torch.rand(n), "squad_alive_mask": torch.ones(n, dtype=torch.bool),
        "num_squads": n, "time_point": torch.tensor(100.0), "map_idx": torch.tensor(1),
    }  # fmt: skip
    b = dataset_v2.collate_fn([sample])
    assert b["x_continuous"].shape[1] == n and int(b["padding_mask"].sum()) == n


def test_cox_sign_convention_and_survival_score():
    _, _, losses, metrics = _import_upstream()
    target = torch.tensor([[0.2, 0.5, 1.0]])
    hazard = torch.tensor([[2.0, 0.0, -2.0]])  # the earliest death has the highest hazard
    valid = torch.ones(1, 3, dtype=torch.bool)
    lf = losses.get_loss_fn("weighted_cox")
    assert float(lf(hazard, target, valid)) < float(lf(-hazard, target, valid))  # the loss prefers this ordering
    # upstream evaluation reads "higher = survives longer": it must be fed the survival score
    assert metrics.compute_winner_accuracy(hazard, target, valid) == 0.0
    s = metrics.survival_score(hazard, "weighted_cox")
    assert metrics.compute_winner_accuracy(s, target, valid) == 1.0
    assert metrics.compute_c_index(s, target, valid) == 1.0
    assert metrics.survival_score(hazard, "mse") is hazard  # regression / classification heads are not flipped


def test_input_dimension_is_57_and_consistent():
    assert S.N_INPUT == 57 and len(S.FEATURES) == 55
    assert S.FEATURES[-2:] != list(S.V2_COMPUTED)  # the two v2 names are inside the 55 and appended again unscaled
    assert set(S.V2_COMPUTED) <= set(S.FEATURES)


def test_csv_inference_uses_57d_input_and_survival_direction(tmp_path):
    """The audited crash (57-d model, 55-d CSV inference) is gone and the win probability follows the survival score."""
    import json

    import pandas as pd

    _import_upstream()
    from src.data.continuous_features import CONTINUOUS_FEATURES
    from src.models.backbone import TransformerBackbone
    from src.models.heads import get_head
    from src.training import inference

    rng = np.random.RandomState(0)
    n_sq, times = 4, [300.0, 900.0]
    rows = []
    for t in times:
        for s in range(n_sq):
            feats = {f: float(rng.rand() * 10) for f in CONTINUOUS_FEATURES}
            feats["squad_total_health"] = 100.0 + 50 * s
            pos = [[1e5 + 1e4 * s, 2e5, 100.0], [1e5 + 1e4 * s, 2.1e5, 100.0], [float("nan")] * 3, [float("nan")] * 3]
            rows.append({"match_id": "m0", "squad_number": s, "time_point": t, **feats, "positions": str(pos),
                         "bluezone_info": "(np.float64(400000.0), np.float64(400000.0), np.float64(300000.0))",
                         "whitezone_info": "(np.float64(380000.0), np.float64(390000.0), np.float64(150000.0))",
                         "map": "erangel", "squad_death_time": 500.0 + 100 * s})  # fmt: skip
    csv = tmp_path / "m0.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    embed, heads, layers = 16, 2, 1
    cfg = {"embed_dim": embed, "num_heads": heads, "num_layers": layers, "dropout": 0.0, "input_dim": 57, "loss_type": "weighted_cox",
           "use_dataset_v2": True, "encoder_type": "transformer"}  # fmt: skip
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    torch.manual_seed(0)
    bb = TransformerBackbone(input_dim=57, embed_dim=embed, num_heads=heads, num_layers=layers, dropout=0.0)
    hd = get_head("weighted_cox", embed, 0.0)
    scaler = (np.zeros(len(CONTINUOUS_FEATURES)), np.ones(len(CONTINUOUS_FEATURES)))
    torch.save({"backbone_state_dict": bb.state_dict(), "head_state_dict": hd.state_dict(), "scaler_params": scaler}, tmp_path / "best.pt")
    df = inference.run_inference_on_csv(str(csv), str(tmp_path / "best.pt"), save_predictions=False)  # raised a shape error upstream
    assert len(df) == len(times) * n_sq
    g = df.groupby("time_point").win_probability.sum()
    np.testing.assert_allclose(g.to_numpy(), 1.0, atol=1e-5)
    # direction: the squad with the LOWEST hazard score has the highest win probability
    from src.data.dataset_v2 import PGCDatasetV2

    ds = PGCDatasetV2(str(tmp_path), ["m0.csv"], scaler_params=scaler)
    bb.eval()
    hd.eval()
    with torch.no_grad():
        s0 = ds[0]
        h = hd(bb(s0["x_continuous"][None], s0["positions"][None], s0["bluezone_info"][None], s0["whitezone_info"][None], s0["map_idx"][None])).squeeze(0)
    first = df[df.time_point == df.time_point.min()].sort_values("squad_idx")
    assert int(np.argmax(first.win_probability.to_numpy())) == int(torch.argmin(h))
