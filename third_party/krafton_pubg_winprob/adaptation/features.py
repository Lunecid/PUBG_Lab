"""Causal reconstruction of KRAFTON's per-squad inputs from the raw ``pubg.telem_*`` rows of ONE match (pure numpy).

Input of :func:`compute_features` is a dict of numpy arrays with every time already on the unified clock W
(``event_time - stdb.matches.started_at`` in seconds, the clock of ``team_death_s`` / ``phase_end_s`` of the
dumps).  Position rows are placed on W through their own ``event_time`` (exact wall clock), so the per-match
game-clock delta of stdb/causal_cut.py (needed only for ``elapsed_time`` columns) is not required.

Causality: the features at time ``t`` read events / samples with time ``< t`` (cumulative windows
``[0, t)`` / ``(0, t)`` as upstream) or ``<= t`` (snapshots as upstream) only.  Death times enter as "alive at t
iff last death time > t" (labels v2 rule); the death time itself is never read before it has happened.

**v2 (causal) mode** (``raw['life']`` present, see :func:`causal_mode`): the leak-free dump of
``model_stgcn.dump_phase_tier_v2``.  Then EVERY read is strict, a record exists at ``t`` iff its time is ``< t`` (snapshots
too: latest position, zone state, equip level, use / ride / held counts), and the member alive status is the v2 causal
one of ``model_stgcn.phase_v2.alive`` (no kill event of the player ``< t``, or a healthy position sample after its last
kill and ``< t``; knocked players are alive), never "last death time > t": that rule reads the future (revives, the
final-death fallback of leavers).  ``raw['d']`` is not used.  With ``causal=False`` (the v1 path, kept for the v1 tests
and the v1 results) nothing changes.

Output layout: ``F[N, K, 55]`` squad features in the order of upstream ``CONTINUOUS_FEATURES``, ``POS[N, K, 4, 3]``
player positions (cm, NaN = dead / unknown), zone ``Z[K, 6]`` = (blue x, y, r, white x, y, r) and the alive
flags.  See schema.PARITY for the status of every feature.
"""

from __future__ import annotations

import numpy as np

from adaptation import schema as S

BIG = 1.0e5  # composite key = player * BIG + (t + OFF); t + OFF must stay in [0, BIG)
OFF = 1.0e3
F = S.F_IDX


class Events:
    """Per-player time-sorted event stream with prefix sums.

    ``cum(ts, strict)`` -> [P, K] sum of the weights of the events of every player with time < ts[k]
    (strict) or <= ts[k]; ``last(ts)`` -> [P, K] index (into the sorted arrays) of the latest event <= ts[k] of
    the player, -1 if there is none.
    """

    def __init__(self, pidx: np.ndarray, t: np.ndarray, w: np.ndarray | None, P: int) -> None:
        pidx = np.asarray(pidx, dtype=np.int64)
        t = np.asarray(t, dtype=np.float64)
        order = np.lexsort((t, pidx))
        self.P = P
        self.p = pidx[order]
        self.t = t[order]
        self.order = order
        ww = np.ones(len(t)) if w is None else np.asarray(w, dtype=np.float64)[order]
        self.w = ww
        self.keys = self.p * BIG + (self.t + OFF)
        self.starts = np.searchsorted(self.p, np.arange(P), side="left")
        ends = np.searchsorted(self.p, np.arange(P), side="right")
        # Block-local prefix sums, one zero-padded block per player: ``cwp[starts[p] + p + j]`` = sum of the first j event weights
        # of player p.  A global running sum minus the block offset would round differently once events of OTHER players (or later
        # events) are removed; the block-local sum of a player depends on that player's events before the cut only, so a prefix
        # replay (records < t) reproduces the full-data features bit for bit.
        self.cwp = np.zeros(len(ww) + P)
        for p_ in np.flatnonzero(ends > self.starts):
            s0, e0 = int(self.starts[p_]), int(ends[p_])
            self.cwp[s0 + p_ + 1 : e0 + p_ + 1] = np.cumsum(ww[s0:e0])
        self._pad = np.arange(P)[:, None]

    def _q(self, ts: np.ndarray) -> np.ndarray:
        return (np.arange(self.P, dtype=np.float64)[:, None] * BIG + (ts[None, :] + OFF)).ravel()

    def cum(self, ts: np.ndarray, strict: bool) -> np.ndarray:
        K = len(ts)
        if len(self.t) == 0:
            return np.zeros((self.P, K))
        ix = np.searchsorted(self.keys, self._q(ts), side="left" if strict else "right").reshape(self.P, K)
        return self.cwp[ix + self._pad]

    def last(self, ts: np.ndarray, strict: bool = False) -> np.ndarray:
        """Index of the latest event ``<= ts[k]`` (``< ts[k]`` if ``strict``) of every player, -1 if there is none."""
        K = len(ts)
        if len(self.t) == 0:
            return np.full((self.P, K), -1, dtype=np.int64)
        ix = np.searchsorted(self.keys, self._q(ts), side="left" if strict else "right").reshape(self.P, K) - 1
        return np.where(ix >= self.starts[:, None], ix, -1)


def _noevents(P: int) -> Events:
    return Events(np.zeros(0, np.int64), np.zeros(0), None, P)


def _as_idx(ids, pid2idx: dict) -> np.ndarray:
    """player id strings -> player index, -1 where unknown / missing."""
    return np.fromiter((pid2idx.get(x, -1) if isinstance(x, str) else -1 for x in ids), dtype=np.int64, count=len(ids))


def member_death_times(death_type, time_survived, last_kill_w, is_ai, fallback_offset_s: float = 0.70):
    """Per-member last death time on W: the dump's rule (model_stgcn.dump_phase_tier.member_death_times).

    alive -> +inf; else the last victim kill event; else ``time_survived + off``; else +inf, where ``off`` is the
    median over human members with both a kill event and a time_survived of ``kill event - time_survived``."""
    human = [lk - ts for lk, ts, ai in zip(last_kill_w, time_survived, is_ai) if (not ai) and np.isfinite(lk) and np.isfinite(ts)]
    off = float(np.median(human)) if human else float(fallback_offset_s)
    d = np.empty(len(death_type))
    for i, (dt, lk, ts) in enumerate(zip(death_type, last_kill_w, time_survived)):
        if dt == "alive":
            d[i] = np.inf
        elif np.isfinite(lk):
            d[i] = lk
        elif np.isfinite(ts):
            d[i] = ts + off
        else:
            d[i] = np.inf
    return d, off


def member_life(kill_w: np.ndarray, kill_victim: np.ndarray, pos_w: np.ndarray, pos_member: np.ndarray, pos_health: np.ndarray) -> dict:
    """``{member index: model_stgcn.phase_v2.alive.PlayerTimeline}`` of the members that have a kill event (all others are
    always alive).  Same inputs and rule as the v2 dump (``alive.build_timelines``); indices are the members' own indices."""
    from model_stgcn.phase_v2 import alive as AL  # read-only: the v2 contract of the causal alive status

    kill_w, kill_victim = np.asarray(kill_w, dtype=np.float64), np.asarray(kill_victim, dtype=np.int64)
    ok = kill_victim >= 0
    return AL.build_timelines(kill_w[ok], kill_victim[ok], np.asarray(pos_w, dtype=np.float64), np.asarray(pos_member, dtype=np.int64),
                              np.asarray(pos_health, dtype=np.float64))  # fmt: skip


def member_alive_causal(life: dict, n_members: int, ts: np.ndarray) -> np.ndarray:
    """``[P, K]`` bool: member alive at ``ts[k]`` using only records strictly before it.

    The status is evaluated by the v2 contract's own function (``model_stgcn.phase_v2.alive.player_alive_at`` on the timelines of
    :func:`member_life`, whose revive rule, including the 20 s stale-health guard, lives in ``alive.build_timelines``): nothing of
    it is re-implemented here.  The contract's functions read records ``<= t``; the dump truncates at ``w < a_k`` and evaluates at
    ``a_k``, which is the same as evaluating the full timelines at ``nextafter(t, -inf)``.  Members without a kill event are
    always alive."""
    from model_stgcn.phase_v2 import alive as AL

    ts = np.asarray(ts, dtype=np.float64)
    tb = np.nextafter(ts, -np.inf)
    alive = np.ones((n_members, len(ts)), dtype=bool)
    for a, tl in life.items():
        if not 0 <= int(a) < n_members or len(tl.kills) == 0:
            continue
        alive[int(a)] = [AL.player_alive_at(life, int(a), float(t)) for t in tb]
    return alive


def team_alive_causal(raw: dict, ts: np.ndarray, member_alive: np.ndarray | None = None) -> np.ndarray:
    """``[N, K]`` bool: the squad has >= 1 causally alive member at ``ts[k]`` (v2 rule, records ``< t``).  ``member_alive`` may
    hand in the result of :func:`member_alive_causal` for the same ``ts`` (saves the second evaluation)."""
    alive = member_alive_causal(raw["life"], len(raw["pid"]), ts) if member_alive is None else member_alive
    return _squad_matrix(raw) @ alive.astype(np.float64) > 0


def _squad_matrix(raw: dict) -> np.ndarray:
    P, N = len(raw["pid"]), int(raw["N"])
    M = np.zeros((N, P))
    M[raw["team"], np.arange(P)] = 1.0
    return M


def causal_mode(raw: dict) -> bool:
    """True iff ``raw`` carries the v2 member life timelines (``raw['life']``): the strict, causal reconstruction."""
    return "life" in raw


def _hypot_ratio(x, y, cx, cy, r):
    """distance / radius; NaN where r < MIN_ZONE_RADIUS_CM (upstream: r <= 0)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.hypot(x - cx, y - cy) / r
    return np.where(r >= S.MIN_ZONE_RADIUS_CM, out, np.nan)


def compute_features(raw: dict, ts: np.ndarray, member_alive: np.ndarray | None = None) -> dict:
    """Features at the (sorted) times ``ts`` of one match.  ``raw`` keys (all numpy, times on W):

    players   : pid (str[P]), team (int[P] squad index 0..N-1), d (float[P] last death time, inf = alive), is_ai
    N         : number of squads, ``slot`` int[N, 4] player indices of the first four members (-1 = none)
    kills     : killer_pidx, finisher_pidx, t, victim_pidx (victim not needed here)
    groggy    : attacker_pidx, victim_pidx, t
    damage    : attacker_pidx, victim_pidx, dmg, causer (object[str]), t
    pos       : pidx, t, health, x, y, z, veh (bool: in a vehicle other than the aircraft)
    zone      : t, sx, sy, sr, px, py, pr
    pick      : pidx, t, kind (0 pickup / 1 drop), item_id (str), stack, removed (bool: clothing subcategory), parachute (bool)
    equip     : pidx, t, item_id (str)
    use       : pidx, t, item_id (str)
    """
    ts = np.asarray(ts, dtype=np.float64)
    K = len(ts)
    P = len(raw["pid"])
    N = int(raw["N"])
    team = raw["team"]
    causal = causal_mode(raw)  # v2: strict (< t) reads and the causal alive status; v1: the legacy rule below
    if causal:
        alive = member_alive_causal(raw["life"], P, ts) if member_alive is None else member_alive  # [P, K] causally alive (records < t)
    else:
        alive = raw["d"][:, None] > ts[None, :]  # [P, K] member alive (dump rule, v1)
    Mmat = np.zeros((N, P))
    Mmat[team, np.arange(P)] = 1.0

    def msum(a):
        """Per-squad sum of a ``[P, K]`` member array, members added in index order (``np.add.at``): the order does not depend on K
        or on the BLAS blocking, so one cut and many cuts give bit-identical values (``Mmat @ a`` does not guarantee that)."""
        out = np.zeros((N, a.shape[1]))
        np.add.at(out, team, np.asarray(a, dtype=np.float64))
        return out

    # ---------------------------------------------------------------- latest position sample per (player, time)
    pz = raw["pos"]
    pos_ev = Events(pz["pidx"], pz["t"], None, P)
    li = pos_ev.last(ts, strict=causal)  # indices into the sorted position arrays
    has = li >= 0
    o = pos_ev.order
    sx, sy, sz = pz["x"][o], pz["y"][o], pz["z"][o]
    sh, sv, st = pz["health"][o], pz["veh"][o], pz["t"][o]
    lic = np.where(has, li, 0)
    if len(o) == 0:
        X = Y = Z = np.full((P, K), np.nan)
        H = np.full((P, K), np.nan)
        V = np.zeros((P, K), bool)
        ST = np.full((P, K), np.nan)
    else:
        X = np.where(has, sx[lic], np.nan)
        Y = np.where(has, sy[lic], np.nan)
        Z = np.where(has, sz[lic], np.nan)
        H = np.where(has, sh[lic], np.nan)
        V = np.where(has, sv[lic], False)
        ST = np.where(has, st[lic], np.nan)
    pos_ok = alive & has & np.isfinite(X) & np.isfinite(Y)

    # ---------------------------------------------------------------- zones at t (latest game state <= t)
    zn = raw["zone"]
    zix = np.searchsorted(zn["t"], ts, side="left" if causal else "right") - 1
    zvalid = zix >= 0
    zc = np.clip(zix, 0, max(len(zn["t"]) - 1, 0))
    if len(zn["t"]) == 0:
        zvalid = np.zeros(K, bool)
        zc = np.zeros(K, np.int64)
        bz = np.zeros((K, 3))
        wz = np.zeros((K, 3))
    else:
        bz = np.stack([zn["sx"][zc], zn["sy"][zc], zn["sr"][zc]], axis=1)
        wz = np.stack([zn["px"][zc], zn["py"][zc], zn["pr"][zc]], axis=1)
        none_white = (wz[:, 0] == 0) & (wz[:, 1] == 0) & (wz[:, 2] == 0)  # upstream: unset white zone -> blue zone
        wz = np.where(none_white[:, None], bz, wz)
    Zt = np.concatenate([bz, wz], axis=1)  # [K, 6]

    # ---------------------------------------------------------------- near players (150 m, 3-D), per time step
    n_team = np.zeros((P, K))
    n_enemy = np.zeros((P, K))
    for k in range(K):
        v = np.flatnonzero(pos_ok[:, k])
        if v.size == 0:
            continue
        c = np.stack([X[v, k], Y[v, k], Z[v, k]], axis=1)
        c = np.where(np.isfinite(c), c, 0.0)  # z missing -> 0 (never happens in the data; keeps the count defined)
        diff = c[:, None, :] - c[None, :, :]
        within = (diff * diff).sum(-1) <= S.NEAR_RADIUS_CM**2
        same = team[v][:, None] == team[v][None, :]
        n_team[v, k] = (within & same).sum(1)  # includes the player itself, as upstream
        n_enemy[v, k] = (within & ~same).sum(1)

    # ---------------------------------------------------------------- zone distances per player
    dist_b = np.where(pos_ok & zvalid[None, :], _hypot_ratio(X, Y, bz[None, :, 0], bz[None, :, 1], bz[None, :, 2]), 0.0)
    dist_w = np.where(pos_ok & zvalid[None, :], _hypot_ratio(X, Y, wz[None, :, 0], wz[None, :, 1], wz[None, :, 2]), 0.0)
    dist_b = np.nan_to_num(dist_b, nan=0.0)  # upstream: no zone info -> 0.0
    dist_w = np.nan_to_num(dist_w, nan=0.0)
    with np.errstate(invalid="ignore"):
        out_blue = pos_ok & zvalid[None, :] & (bz[None, :, 2] >= S.MIN_ZONE_RADIUS_CM) & (
            np.hypot(X - bz[None, :, 0], Y - bz[None, :, 1]) > bz[None, :, 2]
        )

    # ---------------------------------------------------------------- groggy / health / vehicle
    gk = raw["groggy"]
    grg_victim = Events(gk["victim_pidx"][gk["victim_pidx"] >= 0], gk["t"][gk["victim_pidx"] >= 0], None, P)
    g_now = grg_victim.cum(ts, strict=causal)  # knocks <= t (v2: < t)
    g_at_sample = np.zeros((P, K))
    if len(st):
        for_p = np.arange(P)[:, None] * BIG + (np.where(has, ST, 0.0) + OFF)
        if len(grg_victim.t):
            ix = np.searchsorted(grg_victim.keys, for_p.ravel(), side="right").reshape(P, K)
            g_at_sample = grg_victim.cwp[ix + grg_victim._pad]
    knock_after = (g_now - g_at_sample) > 0
    groggy = alive & ((has & (np.nan_to_num(H, nan=100.0) <= 0)) | knock_after)
    health = np.where(has & np.isfinite(H), H, 100.0)  # upstream default 100 when nothing observed

    # ---------------------------------------------------------------- cumulative action features per player
    kl = raw["kills"]
    pk = lambda a: a >= 0  # noqa: E731
    m = pk(kl["killer_pidx"])
    n_kill = Events(kl["killer_pidx"][m], kl["t"][m], None, P).cum(ts, True)
    m = pk(kl["finisher_pidx"])
    n_finish = Events(kl["finisher_pidx"][m], kl["t"][m], None, P).cum(ts, True)
    m = pk(gk["attacker_pidx"])
    n_groggy = Events(gk["attacker_pidx"][m], gk["t"][m], None, P).cum(ts, True)

    dm = raw["damage"]
    pos_dmg = (dm["dmg"] > 0) & (dm["t"] > 0)
    am = pos_dmg & pk(dm["attacker_pidx"])
    n_dmg = Events(dm["attacker_pidx"][am], dm["t"][am], None, P).cum(ts, True)
    sum_dmg = Events(dm["attacker_pidx"][am], dm["t"][am], dm["dmg"][am], P).cum(ts, True)
    vm = pos_dmg & pk(dm["victim_pidx"])
    n_take = Events(dm["victim_pidx"][vm], dm["t"][vm], None, P).cum(ts, True)
    sum_take = Events(dm["victim_pidx"][vm], dm["t"][vm], dm["dmg"][vm], P).cum(ts, True)
    cls = dm["wclass"]  # object[str] weapon class or '' (precomputed by the caller, same rules as upstream)
    n_by, sum_by = {}, {}
    for c in S.WEAPON_CLASSES:
        mc = am & (cls == c)
        n_by[c] = Events(dm["attacker_pidx"][mc], dm["t"][mc], None, P).cum(ts, True)
        sum_by[c] = Events(dm["attacker_pidx"][mc], dm["t"][mc], dm["dmg"][mc], P).cum(ts, True)

    pc = raw["pick"]
    keep = pk(pc["pidx"]) & ~pc["removed"]
    pick_m = keep & (pc["kind"] == 0) & ~pc["parachute"]
    drop_m = keep & (pc["kind"] == 1)
    n_pickup = Events(pc["pidx"][pick_m], pc["t"][pick_m], None, P).cum(ts, True)
    n_drop = Events(pc["pidx"][drop_m], pc["t"][drop_m], None, P).cum(ts, True)

    us = raw["use"]
    uid = us["item_id"]
    use_cnt = {}
    for item in S.AIDS_ITEMS:
        mi = pk(us["pidx"]) & (uid == item)
        use_cnt[item] = Events(us["pidx"][mi], us["t"][mi], None, P).cum(ts, causal)  # t <= time_point (v2: < t)

    # held counts: per item id max(0, picked up - dropped - used) with stack counts (0 or missing -> 1), as upstream
    stack = np.where(pc["stack"] > 0, pc["stack"], 1.0)
    held = {}
    for item in list(S.AIDS_ITEMS) + list(S.THROWABLE_ITEMS):
        # snapshot pickups of aids / throwables use LogItemPickup (event_type pickup) incl. the clothing filter, as upstream
        mp = pk(pc["pidx"]) & (pc["item_id"] == item) & (pc["kind"] == 0)
        md = pk(pc["pidx"]) & (pc["item_id"] == item) & (pc["kind"] == 1)
        got = Events(pc["pidx"][mp], pc["t"][mp], stack[mp], P).cum(ts, causal)
        gone = Events(pc["pidx"][md], pc["t"][md], stack[md], P).cum(ts, causal)
        held[item] = np.maximum(0.0, got - gone - (use_cnt[item] if item in use_cnt else 0.0))

    # rides: a position sample in a vehicle whose predecessor (same player) was not
    ride_cnt = np.zeros((P, K))
    if len(o):
        prev_in = np.concatenate(([False], sv[:-1]))
        first = np.concatenate(([True], pos_ev.p[1:] != pos_ev.p[:-1]))
        prev_in = np.where(first, False, prev_in)
        ride = sv & ~prev_in
        ride_cnt = Events(pos_ev.p[ride], pos_ev.t[ride], None, P).cum(ts, causal)

    # equipment levels: level of the latest matching equip event <= t
    eq = raw["equip"]
    levels = {}
    for name, table in (("helmet", S.HELMET_ITEMS), ("vest", S.VEST_ITEMS), ("backpack", S.BACKPACK_ITEMS)):
        lv = np.zeros((P, K))
        sel = pk(eq["pidx"]) & np.isin(eq["item_id"], list(table))
        if sel.any():
            ev = Events(eq["pidx"][sel], eq["t"][sel], None, P)
            vals = np.array([table[i] for i in eq["item_id"][sel][ev.order]], dtype=np.float64)
            ix = ev.last(ts, strict=causal)
            lv = np.where(ix >= 0, vals[np.clip(ix, 0, len(vals) - 1)], 0.0)
        levels[name] = lv

    # ---------------------------------------------------------------- squad aggregation
    def squad_sum_all(a):  # all members (cumulative action features)
        return msum(a)

    def squad_sum_alive(a):
        return msum(a * alive)

    n_alive = msum(alive)  # [N, K]
    with np.errstate(invalid="ignore", divide="ignore"):

        def squad_mean_alive(a, empty):
            return np.where(n_alive > 0, msum(a * alive) / n_alive, empty)

        # squad centre of alive members with a position
        cnt_pos = msum(pos_ok)
        cx = msum(np.where(pos_ok, X, 0.0)) / cnt_pos
        cy = msum(np.where(pos_ok, Y, 0.0)) / cnt_pos
        has_c = cnt_pos > 0
        v2_b = np.where(has_c & zvalid[None, :] & (bz[None, :, 2] >= S.MIN_ZONE_RADIUS_CM), _hypot_ratio(cx, cy, bz[None, :, 0], bz[None, :, 1], bz[None, :, 2]), -1.0)
        v2_w = np.where(has_c & zvalid[None, :] & (wz[None, :, 2] >= S.MIN_ZONE_RADIUS_CM), _hypot_ratio(cx, cy, wz[None, :, 0], wz[None, :, 1], wz[None, :, 2]), -1.0)
        v2_b = np.nan_to_num(v2_b, nan=-1.0)
        v2_w = np.nan_to_num(v2_w, nan=-1.0)

        Fm = np.zeros((N, K, len(S.FEATURES)), dtype=np.float32)

        def put(name, arr):
            Fm[:, :, F[name]] = arr

        put("n_pickup", squad_sum_all(n_pickup))
        put("n_drop", squad_sum_all(n_drop))
        put("n_kill", squad_sum_all(n_kill))
        put("n_finish", squad_sum_all(n_finish))
        put("n_groggy", squad_sum_all(n_groggy))
        put("n_dmg", squad_sum_all(n_dmg))
        put("sum_dmg", squad_sum_all(sum_dmg))
        for c in S.WEAPON_CLASSES:
            put(f"n_dmg_by_{c}", squad_sum_all(n_by[c]))
            put(f"sum_dmg_by_{c}", squad_sum_all(sum_by[c]))
        put("n_take_dmg", squad_sum_all(n_take))
        put("sum_take_dmg", squad_sum_all(sum_take))
        heal_items = ("Item_Heal_Bandage_C", "Item_Heal_FirstAid_C", "Item_Heal_MedKit_C")
        put("n_heal", squad_sum_all(sum(use_cnt[i] for i in heal_items)))
        # sum_heal_amount stays 0 (LogHeal is not in the database)
        put("n_heal_bandage", squad_sum_all(use_cnt["Item_Heal_Bandage_C"]))
        put("n_heal_firstaid", squad_sum_all(use_cnt["Item_Heal_FirstAid_C"]))
        put("n_heal_medkit", squad_sum_all(use_cnt["Item_Heal_MedKit_C"]))
        put("n_heal_painkiller", squad_sum_all(use_cnt["Item_Boost_PainKiller_C"]))
        put("n_heal_energydrink", squad_sum_all(use_cnt["Item_Boost_EnergyDrink_C"]))
        put("n_heal_adrenaline", squad_sum_all(use_cnt["Item_Boost_AdrenalineSyringe_C"]))
        put("n_vehicle_ride", squad_sum_all(ride_cnt))

        put("n_teammates", squad_mean_alive(n_team, -1.0))
        put("n_enemies", squad_mean_alive(n_enemy, -1.0))
        put("dist_from_bluezone", squad_mean_alive(dist_b, -1.0))
        put("dist_from_whitezone", squad_mean_alive(dist_w, -1.0))
        put("dist_from_bluezone_v2", v2_b)
        put("dist_from_whitezone_v2", v2_w)
        put("is_vehicle_count", squad_sum_alive(V.astype(np.float64) * alive))
        put("item_helmet_level", squad_mean_alive(levels["helmet"], 0.0))
        put("item_vest_level", squad_mean_alive(levels["vest"], 0.0))
        put("item_backpack_level", squad_mean_alive(levels["backpack"], 0.0))
        thr_total = sum(held[i] for i in S.THROWABLE_ITEMS)
        put("item_throwables_count", squad_sum_alive(thr_total))
        for nm, key in (("grenade", "Item_Weapon_Grenade_C"), ("smoke", "Item_Weapon_SmokeBomb_C"), ("flashbang", "Item_Weapon_FlashBang_C"),
                        ("molotov", "Item_Weapon_Molotov_C"), ("bluezone_grenade", "Item_Weapon_BluezoneGrenade_C")):  # fmt: skip
            put(f"item_{nm}_count", squad_sum_alive(held[key]))
        put("item_aids_count", squad_sum_alive(sum(held[i] for i in S.AIDS_ITEMS)))
        for nm, key in (("bandage", "Item_Heal_Bandage_C"), ("firstaid", "Item_Heal_FirstAid_C"), ("medkit", "Item_Heal_MedKit_C"),
                        ("painkiller", "Item_Boost_PainKiller_C"), ("energydrink", "Item_Boost_EnergyDrink_C"),
                        ("adrenaline", "Item_Boost_AdrenalineSyringe_C")):  # fmt: skip
            put(f"item_{nm}_count", squad_sum_alive(held[key]))
        put("squad_total_health", squad_sum_alive(health))
        put("squad_alive_count", n_alive)
        put("squad_non_groggy_count", n_alive - msum(groggy & alive))
        put("squad_out_bluezone_count", msum(out_blue))

    # ---------------------------------------------------------------- positions of the first four members
    slot = raw["slot"]  # [N, 4]
    POS = np.full((N, K, 4, 3), np.nan, dtype=np.float32)
    for j in range(4):
        pj = slot[:, j]
        ok = pj >= 0
        pjc = np.where(ok, pj, 0)
        valid = ok[:, None] & pos_ok[pjc, :]
        POS[:, :, j, 0] = np.where(valid, X[pjc, :], np.nan)
        POS[:, :, j, 1] = np.where(valid, Y[pjc, :], np.nan)
        POS[:, :, j, 2] = np.where(valid, np.nan_to_num(Z[pjc, :], nan=0.0), np.nan)
    return {"F": Fm, "POS": POS, "Zt": Zt.astype(np.float32), "zvalid": zvalid, "member_alive": alive, "pos_ok": pos_ok}
