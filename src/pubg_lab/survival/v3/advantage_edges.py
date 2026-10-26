"""Contact-gated, relative-state edge features with a FUSED dynamic proximity gate.

Proximity is NOT a fixed metric threshold (sample/scale dependent).  We fuse:
  (A) graph-design: kNN topology (relevance, density-adaptive)
  (B) physical spatio-temporal: the characteristic inter-team spacing of a
      contracting 2D point process (Clark-Evans):  ell(t) = (sqrt(pi)/2) * R(t)/sqrt(N(t))
All distances are measured in units of ell(t) (dimensionless d/ell), so the gate
adapts to zone size R(t) and density N(t) and is free of any calibrated meters.

Edge feature vector per directed (src -> dst) at phase p (F = 10):
    [g_combat, g_knn,
     affinity,        # exp(-(d/(beta*ell))^2), self-tuned by the physical scale
     d_over_ell,      # dimensionless distance d/ell(t)
     combat_dmg_log, combat_kills,
     dz, dalive, dhealth, dzone]   # relative state (src - dst), causal <= p

Gates:
    knn    = each team's k nearest (by distance) among alive teams  [fused proximity]
    strict = combat edges only (teams that actually fought; no threshold)
    full   = all proximity pairs (original null graph) + features
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from dataclasses import dataclass
import math
import numpy as np

from pubg_lab.data.stdb.conn import connect

EDGE_FEATURE_NAMES = [
    "g_combat",
    "g_knn",
    "affinity",
    "d_over_ell",
    "combat_dmg_log",
    "combat_kills",
    "dz",
    "dalive",
    "dhealth",
    "dzone",
]
EDGE_FEATURE_DIM = len(EDGE_FEATURE_NAMES)

CLARK_EVANS = math.sqrt(math.pi) / 2.0  # ~0.886; 2D Poisson mean nearest-neighbor coeff
Z_SCALE, ALIVE_SCALE, HEALTH_SCALE, ZONE_SCALE = 400.0, 4.0, 100.0, 1000.0


@dataclass
class RawMatch:
    state: dict  # (team, phase) -> {z, alive, health, zone}
    pairs: dict  # phase -> {(src,dst): {dist, g_combat, combat_w, kills, g_prox}}
    radius: dict  # phase -> safe-zone radius R(t) (m)


def fetch_raw(match_id: str) -> RawMatch:
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT team_id, phase_idx, avg(pos_z), avg(alive_n), avg(mean_health), avg(dist_blue_boundary_m)
              FROM stdb.team_snapshot_features
             WHERE match_id=%s AND phase_idx IS NOT NULL AND phase_idx>0
             GROUP BY 1,2
            """,
            (match_id,),
        )
        state = {}
        for t, p, z, al, h, zn in cur.fetchall():
            state[(int(t), int(p))] = {
                "z": float(z or 0),
                "alive": float(al or 0),
                "health": float(h or 0),
                "zone": float(zn or 0),
            }

        cur.execute(
            """
            SELECT phase_idx, edge_type, src_team_id, dst_team_id, weight, avg_distance_m, damage_sum, kill_count
              FROM stdb.team_phase_edge_cache WHERE match_id=%s AND phase_idx>0
            """,
            (match_id,),
        )
        pairs: dict = {}
        for ph, et, s, d, w, dist, dmg, kills in cur.fetchall():
            pp = pairs.setdefault(int(ph), {})
            e = pp.setdefault(
                (int(s), int(d)),
                {"dist": 1e9, "g_combat": 0.0, "combat_w": 0.0, "kills": 0.0, "g_prox": 0.0},
            )
            if et == "proximity":
                e["g_prox"] = 1.0
                e["dist"] = float(dist) if dist is not None else 1e9
            elif et == "combat":
                e["g_combat"] = 1.0
                e["combat_w"] = float(w or 0.0)
                e["kills"] = float(kills or 0.0)

        cur.execute(
            "SELECT phase_idx, max(safe_radius_start_m) FROM stdb.v_team_phase_feature_v2 WHERE match_id=%s GROUP BY 1",
            (match_id,),
        )
        radius = {int(p): float(r) for p, r in cur.fetchall() if r is not None}
        cur.close()
    return RawMatch(state, pairs, radius)


def build_adv_tensor(
    match_id,
    team_ids,
    phase_indices,
    node_mask,
    *,
    gate="knn",
    k=5,
    beta=1.0,
    raw: RawMatch | None = None,
    radius_by_phase: dict | None = None,
):
    """Aligned A_feat[T,N,N,F] + A_mask[T,N,N] using the fused dynamic gate."""
    raw = raw if raw is not None else fetch_raw(match_id)
    if radius_by_phase:  # allow batched radius override (avoids per-match view query)
        raw.radius = radius_by_phase
    T, N, F = len(phase_indices), len(team_ids), EDGE_FEATURE_DIM
    nm = np.asarray(node_mask).astype(bool)
    idx_of = {int(t): i for i, t in enumerate(team_ids) if t}
    A_feat = np.zeros((T, N, N, F), dtype=np.float32)
    A_mask = np.zeros((T, N, N), dtype=np.float32)

    for t, ph in enumerate(phase_indices):
        ph = int(ph)
        if ph <= 0 or ph not in raw.pairs:
            continue
        alive_idx = np.where(nm[t])[0]
        if alive_idx.size < 2:
            continue
        Nt = int(nm[t].sum())
        R = raw.radius.get(ph, 0.0)
        ell = CLARK_EVANS * R / math.sqrt(max(Nt, 1)) if R > 0 else None

        # candidate directed edges among alive teams, with distance
        cand = {}  # (a,b) local idx -> dict
        for (s_team, d_team), e in raw.pairs[ph].items():
            a = idx_of.get(s_team)
            b = idx_of.get(d_team)
            if a is None or b is None or not (nm[t, a] and nm[t, b]):
                continue
            cand[(a, b)] = (e, raw.state.get((s_team, ph)), raw.state.get((d_team, ph)))

        # gate -> which (a,b) survive
        kept = set()
        if gate == "strict":
            kept = {(a, b) for (a, b), (e, _, _) in cand.items() if e["g_combat"] == 1.0}
        elif gate == "full":
            kept = {
                (a, b)
                for (a, b), (e, _, _) in cand.items()
                if e["g_prox"] == 1.0 or e["g_combat"] == 1.0
            }
        elif gate == "knn":
            # per source a, keep k nearest b (by distance); combat pairs always kept
            by_src: dict = {}
            for (a, b), (e, _, _) in cand.items():
                if e["g_prox"] == 1.0 or e["g_combat"] == 1.0:
                    by_src.setdefault(a, []).append((e["dist"], b, e))
            for a, lst in by_src.items():
                lst.sort(key=lambda x: x[0])
                for dist, b, e in lst[:k]:
                    kept.add((a, b))
                for dist, b, e in lst:
                    if e["g_combat"] == 1.0:
                        kept.add((a, b))
        else:
            raise ValueError(f"unknown gate {gate}")

        for a, b in kept:
            e, ss, sd = cand[(a, b)]
            if ss is None or sd is None:
                continue
            d = e["dist"] if e["dist"] < 1e8 else (ell or 1.0)
            doe = (d / ell) if ell else 0.0
            aff = math.exp(-((d / (beta * ell)) ** 2)) if ell else e["g_prox"]
            A_feat[t, a, b] = [
                e["g_combat"],
                1.0 if gate == "knn" else 0.0,
                aff,
                doe,
                e["combat_w"],
                e["kills"],
                (ss["z"] - sd["z"]) / Z_SCALE,
                (ss["alive"] - sd["alive"]) / ALIVE_SCALE,
                (ss["health"] - sd["health"]) / HEALTH_SCALE,
                (ss["zone"] - sd["zone"]) / ZONE_SCALE,
            ]
            A_mask[t, a, b] = 1.0
    return A_feat, A_mask


if __name__ == "__main__":
    from pubg_lab.survival.v3.factorized_data import fetch_factorized_phase_match

    with connect() as c:
        cur = c.cursor()
        cur.execute("SELECT match_id FROM stdb.team_phase_edge_cache LIMIT 1")
        mid = cur.fetchone()[0]
        cur.close()
    m = fetch_factorized_phase_match(mid)
    raw = fetch_raw(mid)
    print(
        "match", mid[:12], "| radius/phase:", {p: round(r) for p, r in sorted(raw.radius.items())}
    )
    for gate in ("knn", "strict", "full"):
        A_feat, A_mask = build_adv_tensor(
            mid, m.team_ids, m.phase_indices, m.node_mask, gate=gate, raw=raw
        )
        deg = A_mask.reshape(A_mask.shape[0], -1).sum(1).astype(int).tolist()
        # avg out-degree among alive
        print(f"  gate={gate:6} edges/phase={deg}")
        for t in range(len(m.phase_indices)):
            if A_mask[t].sum() > 0:
                ii, jj = np.where(A_mask[t] > 0)
                a, b = ii[0], jj[0]
                print(
                    "    sample:",
                    ", ".join(f"{n}={v:+.3f}" for n, v in zip(EDGE_FEATURE_NAMES, A_feat[t, a, b])),
                )
                break
