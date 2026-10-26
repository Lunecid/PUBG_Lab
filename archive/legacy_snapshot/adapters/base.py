"""
Canonical Arena Graph — Base Adapter
=====================================
배틀 아레나 장르 공통 그래프 추상화.

모든 게임별 어댑터는 이 인터페이스를 구현한다.
모델은 CanonicalSnapshot만 보고, 원시 텔레메트리는 모른다.

설계 원칙:
  - 모든 피처는 0~1 정규화 (게임 간 스케일 불변)
  - 공간 좌표는 경기장 크기 기준 정규화
  - 전투 지표는 게임별 TTK/DPS 기준 정규화
  - 자원 지표는 게임별 최대값 기준 정규화
  - 에이전트 = 개별 플레이어, 그룹 = 팀/스쿼드/듀오

용어:
  agent  = 개별 플레이어 (게임 무관)
  group  = 팀/스쿼드/듀오 (협력 단위)
  arena  = 경기장 (맵)
  boundary = 데미지를 주는 현재 경계 (PUBG: poison zone, Fortnite: storm)
  safe_zone = 다음 안전 구역 목표 (PUBG: white circle)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import torch
import numpy as np


# ============================================================
# 1. 정규화된 데이터 구조
# ============================================================

@dataclass
class AgentState:
    """
    단일 에이전트의 정규화된 상태 (V2, ~39d).

    생태학 5축 프레임워크:
      축1: Physiological State (5d)  — 생리 상태
      축2: Mobility (8d)            — 이동/공간
      축3: Habitat Exposure (5d)    — 서식지 노출
      축4: Competition Pressure (13d)— 경쟁 압력
      축5: Resource Readiness (8d)  — 자원 준비도
    """
    agent_id: str
    group_id: str

    # ── 축1: Physiological State (5d) ──
    health_ratio: float = 1.0           # 0: HP / max_HP
    shield_ratio: float = 0.0           # 1: 쉴드 / max (PUBG=0)
    is_groggy: float = 0.0              # 2: 다운 상태 여부
    groggy_decay: float = 0.0           # 3: exp(-Δt/30) since last groggy
    hp_recovery_recent: float = 0.0     # 4: 최근 회복 아이템 사용 (0~1)

    # ── 축2: Mobility (8d) ──
    arena_x: float = 0.0                # 5: 정규화 x좌표
    arena_y: float = 0.0                # 6: 정규화 y좌표
    arena_z: float = 0.0                # 7: 정규화 z좌표 (고도)
    speed_norm: float = 0.0             # 8: 이동 속도 / max_speed
    in_vehicle: float = 0.0             # 9: 차량 탑승 여부
    radial_speed_to_safe: float = 0.0   # 10: safe zone 접근 속도 (정규화)
    time_to_safe_est: float = 0.0       # 11: safe zone 도달 예상 시간 (정규화)
    time_stationary_recent: float = 0.0 # 12: 최근 30초 내 정지 비율

    # ── 축3: Habitat Exposure (5d) ──
    dist_to_boundary_norm: float = 0.0  # 13: poison boundary까지 / 대각선
    dist_to_safe_norm: float = 0.0      # 14: safe zone까지 / safe_r
    inside_safe: float = 1.0            # 15: safe zone 내부 여부
    inside_boundary: float = 1.0        # 16: poison boundary 내부 (1=안전)
    zone_damage_taken_30s: float = 0.0  # 17: 30초 내 자기장 데미지 / 100

    # ── 축4: Competition Pressure (13d) ──
    dmg_dealt_10s: float = 0.0          # 18: 10초 내 가한 데미지
    dmg_taken_10s: float = 0.0          # 19: 10초 내 받은 데미지
    dmg_dealt_30s: float = 0.0          # 20: 30초 내 가한 데미지
    dmg_taken_30s: float = 0.0          # 21: 30초 내 받은 데미지
    n_unique_attackers_30s: float = 0.0 # 22: 30초 내 고유 공격자 수 / 8
    n_unique_targets_30s: float = 0.0   # 23: 30초 내 고유 공격 대상 수 / 8
    dmg_taken_decay: float = 0.0        # 24: exp(-Δt/10) since last dmg taken
    dmg_dealt_decay: float = 0.0        # 25: exp(-Δt/10) since last dmg dealt
    dist_nearest_ally: float = 0.0      # 26: 가장 가까운 팀원 거리 / 대각선
    dist_team_centroid: float = 0.0     # 27: 팀 중심까지 거리 / 대각선
    enemy_count_100m: float = 0.0       # 28: 100m 내 적 수 / 10
    ally_count_100m: float = 0.0        # 29: 100m 내 아군 수 / team_size
    is_isolated: float = 0.0            # 30: 100m 내 아군 == 0

    # ── 축5: Resource Readiness (8d) ──
    weapon_class_primary: float = 0.0   # 31: 주무기 점수
    weapon_class_secondary: float = 0.0 # 32: 보조무기 점수
    attachment_score: float = 0.0       # 33: 부착물 점수 (0~1)
    armor_level: float = 0.0            # 34: 방어구 레벨 / 3
    helmet_level: float = 0.0           # 35: 헬멧 레벨 / 3
    backpack_level: float = 0.0         # 36: 배낭 레벨 / 3
    heal_use_recent: float = 0.0        # 37: 최근 회복 아이템 사용 횟수 / 3
    boost_use_recent: float = 0.0       # 38: 최근 부스트 사용 횟수 / 3

    def to_tensor(self) -> torch.Tensor:
        """39차원 피처 벡터."""
        return torch.tensor([
            # 축1: Physiological State (5d)
            self.health_ratio,              # 0
            self.shield_ratio,              # 1
            self.is_groggy,                 # 2
            self.groggy_decay,              # 3
            self.hp_recovery_recent,        # 4
            # 축2: Mobility (8d)
            self.arena_x,                   # 5
            self.arena_y,                   # 6
            self.arena_z,                   # 7
            self.speed_norm,                # 8
            self.in_vehicle,                # 9
            self.radial_speed_to_safe,      # 10
            self.time_to_safe_est,          # 11
            self.time_stationary_recent,    # 12
            # 축3: Habitat Exposure (5d)
            self.dist_to_boundary_norm,     # 13
            self.dist_to_safe_norm,         # 14
            self.inside_safe,               # 15
            self.inside_boundary,           # 16
            self.zone_damage_taken_30s,     # 17
            # 축4: Competition Pressure (13d)
            self.dmg_dealt_10s,             # 18
            self.dmg_taken_10s,             # 19
            self.dmg_dealt_30s,             # 20
            self.dmg_taken_30s,             # 21
            self.n_unique_attackers_30s,    # 22
            self.n_unique_targets_30s,      # 23
            self.dmg_taken_decay,           # 24
            self.dmg_dealt_decay,           # 25
            self.dist_nearest_ally,         # 26
            self.dist_team_centroid,        # 27
            self.enemy_count_100m,          # 28
            self.ally_count_100m,           # 29
            self.is_isolated,               # 30
            # 축5: Resource Readiness (8d)
            self.weapon_class_primary,      # 31
            self.weapon_class_secondary,    # 32
            self.attachment_score,          # 33
            self.armor_level,               # 34
            self.helmet_level,              # 35
            self.backpack_level,            # 36
            self.heal_use_recent,           # 37
            self.boost_use_recent,          # 38
        ], dtype=torch.float32)

    FEAT_DIM = 39

    FEAT_NAMES = [
        # 축1: Physiological State
        "health_ratio", "shield_ratio", "is_groggy", "groggy_decay", "hp_recovery_recent",
        # 축2: Mobility
        "arena_x", "arena_y", "arena_z", "speed_norm", "in_vehicle",
        "radial_speed_to_safe", "time_to_safe_est", "time_stationary_recent",
        # 축3: Habitat Exposure
        "dist_to_boundary", "dist_to_safe", "inside_safe", "inside_boundary",
        "zone_dmg_30s",
        # 축4: Competition Pressure
        "dmg_dealt_10s", "dmg_taken_10s", "dmg_dealt_30s", "dmg_taken_30s",
        "n_attackers_30s", "n_targets_30s", "dmg_taken_decay", "dmg_dealt_decay",
        "dist_nearest_ally", "dist_team_centroid",
        "enemy_100m", "ally_100m", "is_isolated",
        # 축5: Resource Readiness
        "weapon_primary", "weapon_secondary", "attachment_score",
        "armor_level", "helmet_level", "backpack_level",
        "heal_use", "boost_use",
    ]


@dataclass
class ArenaState:
    """글로벌 경기장 상태."""
    # 경계 (정규화)
    safe_center_x: float = 0.5    # 0~1
    safe_center_y: float = 0.5
    safe_radius_norm: float = 1.0  # / 초기 반경

    boundary_center_x: float = 0.5
    boundary_center_y: float = 0.5
    boundary_radius_norm: float = 1.0

    # 매치 상태
    safe_area_ratio: float = 1.0    # 현재 면적 / 초기 면적
    shrink_rate: float = 0.0        # 축소 속도 (정규화)
    alive_ratio: float = 1.0        # 생존자 / 초기
    alive_groups_ratio: float = 1.0 # 생존팀 / 초기팀
    game_progress: float = 0.0      # 0→1
    phase: float = 0.0              # 현재 페이즈 / 총 페이즈
    density: float = 0.0            # 생존수 / 면적 (정규화)

    def to_tensor(self) -> torch.Tensor:
        """13차원 아레나 컨텍스트 벡터."""
        return torch.tensor([
            self.safe_center_x, self.safe_center_y, self.safe_radius_norm,
            self.boundary_center_x, self.boundary_center_y, self.boundary_radius_norm,
            self.safe_area_ratio, self.shrink_rate,
            self.alive_ratio, self.alive_groups_ratio,
            self.game_progress, self.phase, self.density,
        ], dtype=torch.float32)

    FEAT_DIM = 13


@dataclass
class AllyEdge:
    """팀 내 엣지."""
    src_id: str
    dst_id: str
    distance_norm: float = 0.0  # / 경기장 대각선
    altitude_diff: float = 0.0  # 정규화

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([self.distance_norm, self.altitude_diff],
                          dtype=torch.float32)

    FEAT_DIM = 2


@dataclass
class EncounterEdge:
    """적대 엣지."""
    src_id: str
    dst_id: str
    distance_norm: float = 0.0
    altitude_diff: float = 0.0
    recent_combat: float = 0.0   # 최근 교전 강도 (0~1)

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([self.distance_norm, self.altitude_diff,
                           self.recent_combat], dtype=torch.float32)

    FEAT_DIM = 3


@dataclass
class CanonicalSnapshot:
    """단일 타임스텝의 정규화된 그래프 스냅샷."""
    elapsed: float                           # 경과 시간 (초)
    agents: list                             # list[AgentState]
    arena: ArenaState = field(default_factory=ArenaState)
    ally_edges: list = field(default_factory=list)       # list[AllyEdge]
    encounter_edges: list = field(default_factory=list)  # list[EncounterEdge]

    def to_hetero_data(self):
        """PyG HeteroData로 변환."""
        from torch_geometric.data import HeteroData

        data = HeteroData()
        n = len(self.agents)
        if n == 0:
            return None

        # 에이전트 ID → 로컬 인덱스
        aid_to_idx = {a.agent_id: i for i, a in enumerate(self.agents)}

        # 노드 피처
        data["agent"].x = torch.stack([a.to_tensor() for a in self.agents])
        data["agent"].num_nodes = n
        data["agent"].agent_ids = [a.agent_id for a in self.agents]
        data["agent"].group_ids = [a.group_id for a in self.agents]

        # 아레나 컨텍스트
        data["arena"].x = self.arena.to_tensor().unsqueeze(0)
        data["arena"].num_nodes = 1

        # Ally 엣지
        a_src, a_dst, a_feat = [], [], []
        for e in self.ally_edges:
            if e.src_id in aid_to_idx and e.dst_id in aid_to_idx:
                a_src.append(aid_to_idx[e.src_id])
                a_dst.append(aid_to_idx[e.dst_id])
                a_feat.append(e.to_tensor())
        if a_src:
            data["agent", "ally", "agent"].edge_index = torch.tensor(
                [a_src, a_dst], dtype=torch.long)
            data["agent", "ally", "agent"].edge_attr = torch.stack(a_feat)
        else:
            data["agent", "ally", "agent"].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data["agent", "ally", "agent"].edge_attr = torch.zeros((0, AllyEdge.FEAT_DIM), dtype=torch.float32)

        # Encounter 엣지
        e_src, e_dst, e_feat = [], [], []
        for e in self.encounter_edges:
            if e.src_id in aid_to_idx and e.dst_id in aid_to_idx:
                e_src.append(aid_to_idx[e.src_id])
                e_dst.append(aid_to_idx[e.dst_id])
                e_feat.append(e.to_tensor())
        if e_src:
            data["agent", "encounter", "agent"].edge_index = torch.tensor(
                [e_src, e_dst], dtype=torch.long)
            data["agent", "encounter", "agent"].edge_attr = torch.stack(e_feat)
        else:
            data["agent", "encounter", "agent"].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data["agent", "encounter", "agent"].edge_attr = torch.zeros((0, EncounterEdge.FEAT_DIM), dtype=torch.float32)

        data.elapsed = self.elapsed
        return data


@dataclass
class GroupOutcome:
    """그룹(팀)의 매치 결과."""
    group_id: str
    final_rank: int               # 1 = 우승
    elimination_time: Optional[float] = None  # 탈락 시점 (초), None=생존
    member_count: int = 1
    is_censored: bool = False     # 매치 종료 시 생존


@dataclass
class CanonicalMatch:
    """단일 매치의 전체 정규화된 데이터."""
    match_id: str
    game: str                      # "pubg", "apex", "fortnite", ...
    snapshots: list                # list[CanonicalSnapshot]
    outcomes: list                 # list[GroupOutcome]
    snapshot_interval: float       # 스냅샷 간격 (초)
    total_agents: int
    total_groups: int

    # 게임별 메타 (선택)
    game_meta: dict = field(default_factory=dict)


# ============================================================
# 2. 어댑터 인터페이스
# ============================================================

class ArenaAdapter(ABC):
    """
    게임별 텔레메트리 → CanonicalMatch 변환 어댑터.

    각 게임은 이 클래스를 상속해서 구현한다.
    모델 코드는 이 인터페이스만 의존한다.
    """

    @abstractmethod
    def game_name(self) -> str:
        """게임 식별자 (예: 'pubg', 'apex')."""
        ...

    @abstractmethod
    def load_match(self, source) -> CanonicalMatch:
        """
        원시 데이터를 CanonicalMatch로 변환.

        Parameters:
            source: 게임별 데이터 소스
                    (파일 경로, DB 커넥션, API 응답 등)

        Returns:
            CanonicalMatch
        """
        ...

    @abstractmethod
    def get_normalization_constants(self) -> dict:
        """
        게임별 정규화 상수.

        Returns:
            dict with keys:
              - arena_size: float (맵 크기, meters)
              - max_hp: float
              - max_shield: float
              - max_speed: float (최대 이동 속도)
              - combat_norm: float (데미지 정규화 기준값)
              - typical_match_duration: float (초)
              - max_phases: int (최대 자기장 페이즈 수)
        """
        ...


# ============================================================
# 3. 정규화 유틸리티
# ============================================================

def normalize_position(x, y, arena_size):
    """월드 좌표 → 0~1 정규화."""
    return x / arena_size, y / arena_size


def normalize_distance(d, arena_diagonal):
    """거리 → 0~1 정규화."""
    return min(d / arena_diagonal, 1.0)


def normalize_combat(value, combat_norm):
    """전투 지표 → 0~1 정규화 (tanh 스케일링)."""
    return min(value / combat_norm, 1.0) if combat_norm > 0 else 0.0


def compute_arena_diagonal(arena_size):
    """정사각형 맵의 대각선 길이."""
    return arena_size * np.sqrt(2)
