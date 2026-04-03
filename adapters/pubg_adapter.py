"""
PUBG Adapter
=============
main.py가 생성한 .pt 그래프 데이터를 CanonicalMatch로 변환.

기존 파이프라인(DB → main.py → .pt)은 그대로 두고,
.pt 파일을 읽어서 정규화된 형태로 변환하는 역할만 한다.

사용법:
  from arena_survival.adapters.pubg_adapter import PUBGAdapter
  adapter = PUBGAdapter()
  match = adapter.load_match("data/graphs/match_0046ac50.pt")
"""

import torch
import numpy as np
from collections import defaultdict

from adapters.base import (
    ArenaAdapter, CanonicalMatch, CanonicalSnapshot,
    AgentState, ArenaState, AllyEdge, EncounterEdge, GroupOutcome,
    normalize_position, normalize_distance, normalize_combat,
    compute_arena_diagonal,
)


class PUBGAdapter(ArenaAdapter):
    """PUBG .pt → CanonicalMatch 변환."""

    # Erangel 기준 상수 (다른 맵은 서브클래스 또는 config로)
    ARENA_SIZE_M = 8160.0       # 맵 크기 (m)
    MAX_HP = 100.0
    MAX_SHIELD = 0.0            # PUBG는 별도 쉴드 없음 (방탄복은 resource로)
    MAX_SPEED_M_S = 130.0       # 차량 최대 속도 (m/s)
    COMBAT_NORM = 200.0         # 30초 내 200dmg = 1.0
    TYPICAL_DURATION = 1800.0   # 30분
    MAX_PHASES = 9              # Erangel 자기장 페이즈

    SNAPSHOT_INTERVAL = 10.0    # 초

    def game_name(self) -> str:
        return "pubg"

    def get_normalization_constants(self) -> dict:
        return {
            "arena_size": self.ARENA_SIZE_M,
            "max_hp": self.MAX_HP,
            "max_shield": self.MAX_SHIELD,
            "max_speed": self.MAX_SPEED_M_S,
            "combat_norm": self.COMBAT_NORM,
            "typical_match_duration": self.TYPICAL_DURATION,
            "max_phases": self.MAX_PHASES,
        }

    def load_match(self, pt_path) -> CanonicalMatch:
        """
        .pt 파일 → CanonicalMatch.

        .pt 구조 (main.py 출력):
          graphs: list[HeteroData]
          snapshot_times: list[float]
          meta: dict
        """
        raw = torch.load(pt_path, map_location="cpu", weights_only=False)
        graphs = raw["graphs"]
        snapshot_times = raw["snapshot_times"]
        meta = raw["meta"]

        diag = compute_arena_diagonal(self.ARENA_SIZE_M)
        team_rank = meta["team_rank"]
        death_times = meta["death_times"]
        player_team = meta["player_team"]

        # 초기 zone 면적 (정규화 기준)
        if graphs:
            z0 = graphs[0]["zone"].x[0]
            initial_area = z0[6].item()
        else:
            initial_area = np.pi * (self.ARENA_SIZE_M / 2) ** 2

        # 총 페이즈 수 추정
        total_phases = self._count_phases(graphs)

        # ── 스냅샷 변환 ──
        snapshots = []
        for g, t in zip(graphs, snapshot_times):
            snap = self._convert_snapshot(g, t, diag, initial_area, total_phases)
            snapshots.append(snap)

        # ── 그룹 결과 ──
        outcomes = self._build_outcomes(meta)

        return CanonicalMatch(
            match_id=meta["match_id"],
            game="pubg",
            snapshots=snapshots,
            outcomes=outcomes,
            snapshot_interval=self.SNAPSHOT_INTERVAL,
            total_agents=meta["total_players"],
            total_groups=len(team_rank),
            game_meta={
                "map": "erangel",
                "pt_path": pt_path,
            },
        )

    def _convert_snapshot(self, g, elapsed, diag, initial_area, total_phases):
        """HeteroData → CanonicalSnapshot."""
        x = g["player"].x          # [n, 39] (V2) or [n, 14] (V1 legacy)
        n = x.shape[0]
        feat_dim = x.shape[1]
        account_ids = g["player"].account_ids
        team_ids = g["player"].team_ids
        team_idx = g["player"].team_idx

        # zone context: [safe_cx, safe_cy, safe_r, poison_cx, poison_cy, poison_r,
        #                area, density, alive, time]
        zv = g["zone"].x[0]
        safe_cx, safe_cy, safe_r = zv[0].item(), zv[1].item(), zv[2].item()
        poison_cx, poison_cy, poison_r = zv[3].item(), zv[4].item(), zv[5].item()
        area = zv[6].item()
        alive_count = zv[8].item()

        is_v2 = (feat_dim >= 39)

        # ── 에이전트 변환 ──
        agents = []
        for i in range(n):
            if is_v2:
                # V2 피처 (39d): main.py에서 이미 정규화됨 → 직접 매핑
                a = AgentState(
                    agent_id=account_ids[i],
                    group_id=str(team_ids[i]),
                    # 축1: Physiological State
                    health_ratio=np.clip(x[i, 0].item(), 0, 1),
                    shield_ratio=x[i, 1].item(),
                    is_groggy=x[i, 2].item(),
                    groggy_decay=x[i, 3].item(),
                    hp_recovery_recent=x[i, 4].item(),
                    # 축2: Mobility
                    arena_x=np.clip(x[i, 5].item(), 0, 1),
                    arena_y=np.clip(x[i, 6].item(), 0, 1),
                    arena_z=np.clip(x[i, 7].item(), 0, 1),
                    speed_norm=x[i, 8].item(),
                    in_vehicle=x[i, 9].item(),
                    radial_speed_to_safe=x[i, 10].item(),
                    time_to_safe_est=x[i, 11].item(),
                    time_stationary_recent=x[i, 12].item(),
                    # 축3: Habitat Exposure
                    dist_to_boundary_norm=x[i, 13].item(),
                    dist_to_safe_norm=x[i, 14].item(),
                    inside_safe=x[i, 15].item(),
                    inside_boundary=x[i, 16].item(),
                    zone_damage_taken_30s=x[i, 17].item(),
                    # 축4: Competition Pressure
                    dmg_dealt_10s=x[i, 18].item(),
                    dmg_taken_10s=x[i, 19].item(),
                    dmg_dealt_30s=x[i, 20].item(),
                    dmg_taken_30s=x[i, 21].item(),
                    n_unique_attackers_30s=x[i, 22].item(),
                    n_unique_targets_30s=x[i, 23].item(),
                    dmg_taken_decay=x[i, 24].item(),
                    dmg_dealt_decay=x[i, 25].item(),
                    dist_nearest_ally=x[i, 26].item(),
                    dist_team_centroid=x[i, 27].item(),
                    enemy_count_100m=x[i, 28].item(),
                    ally_count_100m=x[i, 29].item(),
                    is_isolated=x[i, 30].item(),
                    # 축5: Resource Readiness
                    weapon_class_primary=x[i, 31].item(),
                    weapon_class_secondary=x[i, 32].item(),
                    attachment_score=x[i, 33].item(),
                    armor_level=x[i, 34].item(),
                    helmet_level=x[i, 35].item(),
                    backpack_level=x[i, 36].item(),
                    heal_use_recent=x[i, 37].item(),
                    boost_use_recent=x[i, 38].item(),
                )
            else:
                # V1 legacy (14d): 기존 변환 로직 유지
                pos_x = x[i, 0].item()
                pos_y = x[i, 1].item()
                pos_z = x[i, 2].item()
                ax, ay = normalize_position(pos_x, pos_y, self.ARENA_SIZE_M)
                az = self._normalize_altitude(pos_z)
                a = AgentState(
                    agent_id=account_ids[i],
                    group_id=str(team_ids[i]),
                    arena_x=np.clip(ax, 0, 1),
                    arena_y=np.clip(ay, 0, 1),
                    arena_z=az,
                    health_ratio=np.clip(x[i, 3].item(), 0, 1),
                    dist_to_boundary_norm=normalize_distance(x[i, 7].item(), diag),
                    dist_to_safe_norm=normalize_distance(x[i, 4].item(), max(safe_r, 1)),
                    inside_safe=x[i, 6].item(),
                    inside_boundary=x[i, 9].item(),
                    dmg_dealt_30s=normalize_combat(x[i, 12].item(), self.COMBAT_NORM),
                    dmg_taken_30s=normalize_combat(x[i, 13].item(), self.COMBAT_NORM),
                    speed_norm=min(x[i, 10].item() / self.MAX_SPEED_M_S, 1.0),
                    in_vehicle=x[i, 11].item(),
                )
            agents.append(a)

        # ── 아레나 상태 ──
        arena = ArenaState(
            safe_center_x=safe_cx / self.ARENA_SIZE_M,
            safe_center_y=safe_cy / self.ARENA_SIZE_M,
            safe_radius_norm=safe_r / (self.ARENA_SIZE_M / 2),
            boundary_center_x=poison_cx / self.ARENA_SIZE_M if poison_r > 0 else 0.5,
            boundary_center_y=poison_cy / self.ARENA_SIZE_M if poison_r > 0 else 0.5,
            boundary_radius_norm=poison_r / (self.ARENA_SIZE_M / 2) if poison_r > 0 else 1.0,
            safe_area_ratio=min(area / max(initial_area, 1), 1.0),
            shrink_rate=0.0,  # TODO: 이전 스냅샷 대비 변화율
            alive_ratio=alive_count / 100.0,  # 최대 100명 기준
            alive_groups_ratio=0.0,  # 스냅샷 단독으로는 팀 수 모름 → 데이터셋에서 채움
            game_progress=min(elapsed / self.TYPICAL_DURATION, 1.0),
            phase=0.0,  # 데이터셋에서 채움
            density=0.0,  # 데이터셋에서 채움
        )

        # ── 엣지 변환 ──
        ally_edges = self._convert_ally_edges(g, agents, diag)
        enc_edges = self._convert_encounter_edges(g, agents, diag)

        return CanonicalSnapshot(
            elapsed=elapsed,
            agents=agents,
            arena=arena,
            ally_edges=ally_edges,
            encounter_edges=enc_edges,
        )

    def _convert_ally_edges(self, g, agents, diag):
        """ally 엣지 → AllyEdge 리스트."""
        ei = g["player", "ally", "player"].edge_index
        ea = g["player", "ally", "player"].edge_attr
        edges = []
        for e in range(ei.shape[1]):
            si, di = ei[0, e].item(), ei[1, e].item()
            if si >= len(agents) or di >= len(agents):
                continue
            dist = ea[e, 0].item() if ea.shape[0] > 0 else 0
            alt = ea[e, 1].item() if ea.shape[0] > 0 else 0
            edges.append(AllyEdge(
                src_id=agents[si].agent_id,
                dst_id=agents[di].agent_id,
                distance_norm=normalize_distance(dist, diag),
                altitude_diff=alt / 500.0,  # 500m 기준 정규화
            ))
        return edges

    def _convert_encounter_edges(self, g, agents, diag):
        """encounter 엣지 → EncounterEdge 리스트."""
        ei = g["player", "encounter", "player"].edge_index
        ea = g["player", "encounter", "player"].edge_attr
        edges = []
        for e in range(ei.shape[1]):
            si, di = ei[0, e].item(), ei[1, e].item()
            if si >= len(agents) or di >= len(agents):
                continue
            dist = ea[e, 0].item() if ea.shape[0] > 0 else 0
            alt = ea[e, 1].item() if ea.shape[0] > 0 else 0

            # 양쪽 교전 강도
            s_combat = agents[si].recent_dmg_dealt_norm + agents[si].recent_dmg_taken_norm
            d_combat = agents[di].recent_dmg_dealt_norm + agents[di].recent_dmg_taken_norm
            combat_intensity = min((s_combat + d_combat) / 2, 1.0)

            edges.append(EncounterEdge(
                src_id=agents[si].agent_id,
                dst_id=agents[di].agent_id,
                distance_norm=normalize_distance(dist, diag),
                altitude_diff=alt / 500.0,
                recent_combat=combat_intensity,
            ))
        return edges

    def _build_outcomes(self, meta):
        """매치 결과 → GroupOutcome 리스트."""
        team_rank = meta["team_rank"]
        death_times = meta["death_times"]
        player_team = meta["player_team"]
        survivors = meta["survivors"]

        # 팀별 멤버 수 + 사망 수
        team_members = defaultdict(list)
        for pid, tid in player_team.items():
            team_members[tid].append(pid)

        outcomes = []
        for tid, rank in team_rank.items():
            members = team_members.get(tid, [])
            dead_members = [p for p in members if p in death_times]
            all_dead = len(dead_members) == len(members) and len(members) > 0

            elim_time = None
            if all_dead:
                elim_time = max(death_times[p] for p in dead_members)

            outcomes.append(GroupOutcome(
                group_id=tid,
                final_rank=rank,
                elimination_time=elim_time,
                member_count=len(members),
                is_censored=not all_dead,
            ))
        return outcomes

    def _normalize_altitude(self, z_meters):
        """고도 정규화. PUBG Erangel: 0~400m 범위를 0~1로."""
        return np.clip(z_meters / 400.0, 0, 1)

    def _count_phases(self, graphs):
        """poison_r 변화 횟수로 페이즈 수 추정."""
        if len(graphs) < 2:
            return 1
        phases = 1
        prev_pr = graphs[0]["zone"].x[0, 5].item()
        shrinking = False
        for g in graphs[1:]:
            pr = g["zone"].x[0, 5].item()
            if not shrinking and pr < prev_pr - 1.0:
                shrinking = True
            elif shrinking and abs(pr - prev_pr) < 0.5:
                shrinking = False
                phases += 1
            prev_pr = pr
        return max(phases, 1)


# ============================================================
# 검증
# ============================================================

if __name__ == "__main__":
    import sys, os

    pt_dir = sys.argv[1] if len(sys.argv) > 1 else "data/graphs"
    pt_files = sorted(f for f in os.listdir(pt_dir) if f.endswith(".pt"))
    if not pt_files:
        print(f"No .pt in {pt_dir}")
        sys.exit(1)

    adapter = PUBGAdapter()
    print(f"Game: {adapter.game_name()}")
    print(f"Constants: {adapter.get_normalization_constants()}")

    fp = os.path.join(pt_dir, pt_files[0])
    print(f"\nLoading: {fp}")
    match = adapter.load_match(fp)

    print(f"  match_id: {match.match_id[:20]}...")
    print(f"  agents: {match.total_agents}, groups: {match.total_groups}")
    print(f"  snapshots: {len(match.snapshots)}")
    print(f"  outcomes: {len(match.outcomes)}")

    # 첫 스냅샷 검증
    s0 = match.snapshots[0]
    print(f"\n  Snapshot 0 (t={s0.elapsed}s):")
    print(f"    agents: {len(s0.agents)}")
    print(f"    ally_edges: {len(s0.ally_edges)}")
    print(f"    encounter_edges: {len(s0.encounter_edges)}")
    if s0.agents:
        a0 = s0.agents[0]
        t = a0.to_tensor()
        print(f"    agent feat_dim: {t.shape[0]} (expected {AgentState.FEAT_DIM})")
        # 축별 요약 출력
        for name, val in zip(AgentState.FEAT_NAMES, t.tolist()):
            print(f"      {name:<20s} = {val:.3f}")
    print(f"    arena: area_ratio={s0.arena.safe_area_ratio:.3f}, "
          f"progress={s0.arena.game_progress:.3f}")

    # HeteroData 변환 검증
    hd = s0.to_hetero_data()
    if hd:
        print(f"\n  HeteroData:")
        print(f"    agent.x: {hd['agent'].x.shape}")
        print(f"    arena.x: {hd['arena'].x.shape}")
        print(f"    ally edges: {hd['agent', 'ally', 'agent'].edge_index.shape}")
        print(f"    enc edges: {hd['agent', 'encounter', 'agent'].edge_index.shape}")

    # 결과 검증
    print(f"\n  Outcomes (sample):")
    for o in match.outcomes[:5]:
        print(f"    group={o.group_id}, rank={o.final_rank}, "
              f"elim={o.elimination_time:.0f}s" if o.elimination_time else
              f"    group={o.group_id}, rank={o.final_rank}, survived")

    print("\n완료.")
