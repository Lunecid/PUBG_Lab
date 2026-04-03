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
    """단일 에이전트의 정규화된 상태."""
    agent_id: str
    group_id: str

    # 공간 (0~1, 경기장 정규화)
    arena_x: float = 0.0
    arena_y: float = 0.0
    arena_z: float = 0.0     # 고도 (정규화)

    # 생존
    health_ratio: float = 1.0   # HP / max_HP
    shield_ratio: float = 0.0   # 방어구/쉴드 / max

    # 서식지
    dist_to_boundary_norm: float = 0.0   # 현재 경계까지 / 대각선
    dist_to_safe_norm: float = 0.0       # 다음 안전구역까지 / 경계 반경
    inside_safe: float = 1.0             # 다음 안전구역 내부
    inside_boundary: float = 1.0         # 현재 경계 내부 (1=안전)

    # 전투
    recent_dmg_dealt_norm: float = 0.0   # / 기준값
    recent_dmg_taken_norm: float = 0.0

    # 이동
    speed_norm: float = 0.0    # / max_speed
    in_vehicle: float = 0.0

    # 자원 (게임별 정의)
    resource_level: float = 0.0  # 0~1

    def to_tensor(self) -> torch.Tensor:
        """14차원 피처 벡터."""
        return torch.tensor([
            self.arena_x, self.arena_y, self.arena_z,
            self.health_ratio, self.shield_ratio,
            self.dist_to_boundary_norm, self.dist_to_safe_norm,
            self.inside_safe, self.inside_boundary,
            self.recent_dmg_dealt_norm, self.recent_dmg_taken_norm,
            self.speed_norm, self.in_vehicle,
            self.resource_level,
        ], dtype=torch.float32)

    FEAT_DIM = 14

    FEAT_NAMES = [
        "arena_x", "arena_y", "arena_z",
        "health_ratio", "shield_ratio",
        "dist_to_boundary", "dist_to_safe",
        "inside_safe", "inside_boundary",
        "dmg_dealt", "dmg_taken",
        "speed", "in_vehicle",
        "resource_level",
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
