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

import projects.krafton_baseline.data.schema as S  # noqa: E402
from projects.krafton_baseline.data.features import compute_features  # noqa: E402

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


def test_token_only_backbone_has_no_position_zone_map_and_ignores_padding():
    """K2 (KRAFTON-tf-ourfeat): upstream's modules without the position / zone / map embeddings, same key padding mask."""
    _import_upstream()
    from projects.krafton_baseline.models.ourfeat import TokenOnlyBackbone

    torch.manual_seed(0)
    bb = TokenOnlyBackbone(input_dim=153, embed_dim=16, num_heads=2, num_layers=2, dropout=0.0).eval()
    assert not ({"position_encoding", "bluezone_embedding", "whitezone_embedding", "map_embedding"} & set(dict(bb.named_children())))
    x = torch.randn(1, 3, 153)
    valid = torch.ones(1, 3, dtype=torch.bool)
    ref = bb(x, None, None, None, None, valid_mask=valid)
    xp = torch.cat([x, torch.randn(1, 9, 153) * 5], dim=1)
    vp = torch.cat([valid, torch.zeros(1, 9, dtype=torch.bool)], dim=1)
    out = bb(xp, torch.zeros(1, 12, 4, 3), torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, dtype=torch.long), valid_mask=vp)
    assert torch.allclose(ref, out[:, :3], atol=1e-5)


def test_step_loader_fills_an_epoch_by_repeating_shuffled_samples():
    _import_upstream()
    from projects.krafton_baseline.models.ourfeat import StepLoader, TokenSampleSet

    n_s = 10
    n = np.full(n_s, 3, dtype=np.int16)
    R = int(n.sum())
    arrays = {"X": np.random.RandomState(0).randn(R, 153).astype(np.float32), "TGT": np.linspace(100, 900, R).astype(np.float32),
              "TIDX": np.tile(np.arange(3), n_s).astype(np.int16), "S_n": n, "S_cut": np.ones(n_s, dtype=np.int8),
              "S_mi": np.arange(n_s, dtype=np.int32), "S_t": np.zeros(n_s, dtype=np.float32)}  # fmt: skip
    ds = TokenSampleSet(arrays, np.array([f"m{i}" for i in range(n_s)]))
    loader = StepLoader(ds, batch_size=4, shuffle=True, seed=1, epoch_steps=7)
    batches = list(loader)
    assert len(batches) == 7 == len(loader)
    assert sum(b["x_continuous"].shape[0] for b in batches) == 28  # 28 sample draws = 2.8 passes over 10 samples
    seen = np.concatenate([b["time_point"].numpy() for b in batches])
    assert len(seen) == 28 and batches[0]["positions"].shape[-2:] == (4, 3)
    val = list(StepLoader(ds, batch_size=4, shuffle=False))
    assert len(val) == 3 and sum(b["x_continuous"].shape[0] for b in val) == n_s  # validation: one ordered pass


def test_feature_table_rows_cover_alive_at_phase_start_and_risk_set():
    """adaptation.kftable.table_rows: one row per team alive at the start of each cut phase, risk-set rows flagged, causal cut."""
    import projects.krafton_baseline.data.kftable as kftable

    raw = make_raw(3, N=6, size=4)
    N = 6
    rng = np.random.RandomState(1)
    death = np.array([150.0, 420.0, 610.0, 900.0, np.inf, np.nan])
    won = np.array([False, False, False, False, True, False])
    pe = np.array([290.0, 500.0, 680.0, 800.0, 920.0, 960.0, np.nan, np.nan])
    ps = np.array([60.0, 300.0, 510.0, 690.0, 810.0, 930.0, np.nan, np.nan])
    pm = np.array([True] * 6 + [False] * 2)
    T_eff = np.where(won, np.inf, death)
    target = np.zeros((8, N), dtype=bool)
    for p in range(5):  # a risk set at cuts 1..5: alive after the phase end and phase p + 2 exists
        target[p] = T_eff > pe[p]
    y = np.where(target, (T_eff <= np.append(pe[1:], np.inf)[:, None].T[0][:, None]).astype(np.float32), 0.0).astype(np.float32)
    lab = {"match_id": "synthetic", "team_ids": np.arange(10, 10 + N), "team_death_s": death, "team_won": won, "phase_end_s": pe, "phase_start_s": ps,
           "phase_mask": pm, "target": target, "y": y}  # fmt: skip
    r = kftable.table_rows("synthetic", lab, raw, {"flags": {"first_state_w": 10.0}})
    from pubg_lab.survival.ptr.data.data import event_alive_mask

    alive_start = event_alive_mask(lab).numpy()
    slots = [p for p in range(7) if pm[p]]
    assert r["n"] == int(sum(alive_start[p].sum() for p in slots))  # every team alive at phase start, cut phases 1..7 that exist
    assert int(r["in_risk_set"].sum()) == int(target[slots].sum())
    assert set(np.unique(r["phase"])) <= set(range(1, 8))
    cut = r["cut_t"]
    assert np.allclose(cut, pe[r["phase"] - 1] - 10.0)  # strictly the unified-clock cut
    team_idx = r["team_id"] - 10
    assert np.array_equal(r["alive_at_cut"].astype(bool), T_eff[team_idx] > cut)
    assert np.isnan(r["y"][r["in_risk_set"] == 0]).all() and np.isfinite(r["y"][r["in_risk_set"] == 1]).all()
    # the features are those of compute_features at the cut times
    ts = pe[slots] - 10.0
    full = compute_features(raw, ts)
    k = slots.index(2)
    sel = np.flatnonzero(r["phase"] == 3)
    np.testing.assert_allclose(r["F"][sel], full["F"][team_idx[sel], k, :], rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- K3a / K3 (BCE objective, cut-phase bias)
def _tiny_hazard_samples(seed: int = 0, n_s: int = 9, d: int = 12):
    """Synthetic cut samples (a TokenSampleSet + label tables) with a known risk set."""
    from projects.krafton_baseline.models.ourfeat import TokenSampleSet

    rng = np.random.RandomState(seed)
    n = rng.randint(3, 7, n_s).astype(np.int16)
    R = int(n.sum())
    cuts = (np.arange(n_s) % 7 + 1).astype(np.int8)
    mids = np.array([f"m{i // 3}" for i in range(n_s)])  # three samples per match
    uniq = list(dict.fromkeys(mids.tolist()))
    S_mi = np.array([uniq.index(m) for m in mids], dtype=np.int32)
    tidx = np.concatenate([np.arange(k) for k in n]).astype(np.int16)
    arrays = {"X": rng.randn(R, d).astype(np.float32), "TGT": rng.uniform(100, 2000, R).astype(np.float32), "TIDX": tidx, "S_n": n, "S_cut": cuts,
              "S_mi": S_mi, "S_t": rng.uniform(100, 1500, n_s).astype(np.float32)}  # fmt: skip
    ss = TokenSampleSet(arrays, np.array(uniq))
    labs = {}
    for mi, mid in enumerate(uniq):
        N = 8
        target = np.zeros((8, N), dtype=bool)
        y = np.zeros((8, N), dtype=np.float32)
        for i in np.flatnonzero(S_mi == mi):
            p = int(cuts[i]) - 1
            k = int(n[i])
            risk = rng.rand(k) < 0.7
            risk[0] = True  # at least one risk-set row per sample
            target[p, :k] = risk
            y[p, :k] = np.where(risk, (rng.rand(k) < 0.4).astype(np.float32), 0.0)
        labs[mid] = {"match_id": mid, "team_ids": np.arange(N) + 1, "target": target, "y": y}
    return ss, labs


def test_k3_attach_labels_and_batch_encode_risk_rows():
    _import_upstream()
    from projects.krafton_baseline.training.ourfeat_hazard import NO_LOSS, attach_labels

    ss, labs = _tiny_hazard_samples()
    hs = attach_labels(ss, labs)
    for i in range(len(hs)):  # rows equal the label tables
        lab = labs[str(hs.match_ids[hs.S_mi[i]])]
        p = int(hs.S_cut[i]) - 1
        r0, r1 = hs.off[i], hs.off[i + 1]
        t = hs.TIDX[r0:r1].astype(np.int64)
        assert np.array_equal(hs.RISK[r0:r1], lab["target"][p, t])
        assert np.array_equal(hs.Y[r0:r1], lab["y"][p, t])
    b = hs.batch(np.arange(len(hs)))
    y = b["y"].numpy()
    valid = b["padding_mask"].numpy()
    assert (y[~valid] == NO_LOSS).all()  # padding carries no loss
    n_risk = int(hs.RISK.sum())
    assert int(((y >= 0) & valid).sum()) == n_risk  # exactly the risk-set rows are labelled
    assert np.array_equal(b["cut_phase"].numpy(), hs.S_cut.astype(np.int64))
    assert b["x_continuous"].shape[:2] == b["y"].shape  # same attention set as K2: every alive squad stays a key
    ref = hs.__class__.__mro__[1].batch(hs, np.arange(len(hs)))  # K2's batch of the same samples
    assert torch.equal(ref["x_continuous"], b["x_continuous"]) and torch.equal(ref["squad_alive_mask"], b["squad_alive_mask"])


def test_k3_phase_head_bias_and_gradient():
    _import_upstream()
    from projects.krafton_baseline.training.ourfeat_hazard import PhaseHead
    from src.models.heads import HazardHead

    torch.manual_seed(0)
    head = PhaseHead(HazardHead(16, dropout=0.0)).eval()
    with torch.no_grad():
        head.phase_bias.weight.copy_(torch.tensor([[0.5], [-1.0], [2.0], [0.0], [0.0], [0.0], [3.0]]))
    x = torch.randn(3, 4, 16)
    phase = torch.tensor([1, 3, 7])
    out = head(x, phase)
    base = head.base(x)
    assert torch.allclose(out - base, torch.tensor([0.5, 2.0, 3.0])[:, None].expand(3, 4), atol=1e-6)
    head.zero_grad()
    out[0].sum().backward()  # a loss on sample 0 (phase 1) only
    g = head.phase_bias.weight.grad[:, 0]
    assert g[0].abs() > 0 and (g[1:] == 0).all()


def test_k3_init_intercepts_and_shared_initial_weights_with_k2():
    _import_upstream()
    import projects.krafton_baseline.training.train as T
    from projects.krafton_baseline.models.ourfeat import TokenOnlyBackbone
    from projects.krafton_baseline.training.ourfeat_hazard import PhaseHead, build_models, logit
    from src.models.heads import get_head

    rates = np.array([0.2, 0.25, 0.3, 0.4, 0.45, 0.5, 0.55])
    pooled = 0.33
    bb3, h3, init3 = build_models(153, 42, True, pooled, rates)
    bba, ha, inita = build_models(153, 42, False, pooled, None)
    assert isinstance(h3, PhaseHead) and not isinstance(ha, PhaseHead)
    # head bias + phase offset = the logit of the phase base rate; K3a has the pooled intercept only
    off = h3.phase_bias.weight.detach().numpy()[:, 0]
    bias = float(h3.base.mlp[-1].bias)
    np.testing.assert_allclose(bias + off, [logit(r) for r in rates], atol=1e-5)
    assert abs(float(ha.mlp[-1].bias) - logit(pooled)) < 1e-6 and inita["phase_offsets"] is None
    # same seed and construction order as K2: backbone and head weights equal, except the last bias
    T.set_seed(42)
    bb2 = TokenOnlyBackbone(input_dim=153, embed_dim=T.PAPER["embed_dim"], num_heads=T.PAPER["num_heads"], num_layers=T.PAPER["num_layers"], dropout=T.PAPER["dropout"])
    h2 = get_head(T.LOSS, embed_dim=T.PAPER["embed_dim"], dropout=T.PAPER["dropout"])
    for (k, v2), (_, va), (_, v3) in zip(bb2.state_dict().items(), bba.state_dict().items(), bb3.state_dict().items()):
        assert torch.equal(v2, va) and torch.equal(v2, v3), k
    for (k, v2), (_, va) in zip(h2.state_dict().items(), ha.state_dict().items()):
        if k != "mlp.3.bias":
            assert torch.equal(v2, va), k
    for (k, v2), (k3, v3) in zip(h2.state_dict().items(), h3.base.state_dict().items()):
        if k != "mlp.3.bias":
            assert torch.equal(v2, v3), k


def test_k3_trainer_loss_masks_and_pooled_val_nll(tmp_path):
    _import_upstream()
    import torch.nn.functional as F

    import projects.krafton_baseline.models.ourfeat as OF
    import projects.krafton_baseline.training.train as T
    from projects.krafton_baseline.training.ourfeat_hazard import BCETrainer, HazardSampleSet, PhaseHead, attach_labels, masked_bce, predict_logits
    from src.models.heads import HazardHead

    # masked BCE: only rows with a label >= 0 inside the valid mask count
    pred = torch.randn(2, 5)
    y = torch.tensor([[1.0, 0.0, -1.0, 1.0, -1.0], [0.0, -1.0, 1.0, 0.0, 1.0]])
    valid = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
    m = valid & (y >= 0)
    ref = F.binary_cross_entropy_with_logits(pred[m], y[m])
    assert torch.allclose(masked_bce(pred, y, valid), ref, atol=1e-6)
    assert torch.allclose(masked_bce(pred, y, valid, reduction="sum"), ref * m.sum(), atol=1e-5)

    ss, labs = _tiny_hazard_samples(seed=3, d=12)
    hs = attach_labels(ss, labs)
    torch.manual_seed(0)
    bb = OF.TokenOnlyBackbone(input_dim=12, embed_dim=16, num_heads=2, num_layers=2, dropout=0.0)
    head = PhaseHead(HazardHead(16, dropout=0.0))
    trainer = BCETrainer(backbone=bb, head=head, train_loader=OF.StepLoader(hs, 4, True, 0, 3), val_loader=OF.StepLoader(hs, 4, False), loss_type="weighted_cox",
                         lr=1e-3, weight_decay=0.0, device="cpu", checkpoint_dir=str(tmp_path), phased=True)  # fmt: skip
    nll = trainer.validate()  # batches of 4 samples -> unequal row counts per batch: must be the pooled value
    g = predict_logits(bb, head, hs, "cpu", True)
    p = 1 / (1 + np.exp(-g[hs.RISK]))
    y_rows = hs.Y[hs.RISK].astype(np.float64)
    pooled = float(-(y_rows * np.log(p) + (1 - y_rows) * np.log(1 - p)).mean())
    assert abs(nll - pooled) < 1e-5
    # a training epoch runs, updates the phase bias and keeps the loss finite
    before = head.phase_bias.weight.detach().clone()
    loss = trainer.train_epoch(1)
    assert np.isfinite(loss) and not torch.equal(before, head.phase_bias.weight.detach())
