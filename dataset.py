"""
TeamSurvivalDataset
====================
main.py가 생성한 .pt 그래프 데이터를 받아
(match, team, timestep) 단위의 생존 분석 샘플로 변환.

구조:
  MatchSurvivalData  — 단일 매치 전처리 (팀 사망시점, zone phase 등)
  TeamSurvivalDataset — PyTorch Dataset (인덱싱 + 윈도우 + 라벨)
  build_team_graph    — 플레이어 그래프에서 팀-팀 상호작용 그래프 구성
  collate_fn          — 가변 크기 배치 처리

샘플 구조:
  input:
    - player_graphs: list[HeteroData]  (최근 L 스텝)
    - team_mask: tensor  (해당 팀 소속 플레이어 boolean mask)
    - all_team_ids: tensor  (모든 플레이어의 팀 인덱스)
    - team_graph: Data  (팀-팀 상호작용 그래프)
    - zone_seq: tensor [L, zone_dim]  (zone context 시퀀스)

  label:
    - event: 0 or 1  (다음 구간 내 팀 탈락 여부)
    - at_risk: 0 or 1  (현 시점 생존 여부)
    - censored: 0 or 1  (매치 종료 시 생존)

  meta:
    - match_id: str
    - team_id: str
    - team_idx: int  (글로벌 팀 인덱스)
    - timestep: int  (스냅샷 인덱스)
    - elapsed: float  (초)
    - final_rank: int
    - zone_phase: int
    - alive_teams: int
    - norm_zone_area: float  (정규화된 zone 면적, 0~1)
"""

import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData, Data
from collections import defaultdict


# ============================================================
# 1. 단일 매치 전처리
# ============================================================

class MatchSurvivalData:
    """
    단일 매치의 .pt 파일을 로드해서 팀 수준 생존 메타데이터를 계산.

    계산하는 것:
    - team_death_step: {team_idx: 마지막 멤버 탈락 스냅샷 인덱스}
    - zone_phases: 각 스냅샷의 zone phase 번호
    - team_alive_at: [n_teams, n_steps] boolean 텐서
    """

    def __init__(self, pt_path):
        raw = torch.load(pt_path, map_location="cpu", weights_only=False)
        self.graphs = raw["graphs"]
        self.snapshot_times = raw["snapshot_times"]
        self.meta = raw["meta"]
        self.pt_path = pt_path

        self.match_id = self.meta["match_id"]
        self.n_steps = len(self.graphs)

        # ── 팀/플레이어 매핑 ──
        # meta["team_rank"]: {team_id_str: rank}
        # meta["player_team"]: {player_id: team_id_str}
        # meta["death_times"]: {account_id: elapsed_time}

        team_rank = self.meta["team_rank"]
        player_team = self.meta["player_team"]
        death_times = self.meta["death_times"]

        # 팀 정렬 → 글로벌 인덱스
        self.team_ids = sorted(team_rank.keys())
        self.n_teams = len(self.team_ids)
        self.team_to_idx = {t: i for i, t in enumerate(self.team_ids)}
        self.team_ranks = {self.team_to_idx[t]: r for t, r in team_rank.items()}

        # 플레이어 → 팀 인덱스
        self.player_to_team_idx = {}
        for pid, tid in player_team.items():
            if tid in self.team_to_idx:
                self.player_to_team_idx[pid] = self.team_to_idx[tid]

        # ── 팀 사망 시점 ──
        # 팀의 마지막 멤버가 탈락한 시점 = 팀 탈락 시점
        team_last_death = {}  # team_idx → max death elapsed
        team_member_count = defaultdict(int)
        team_dead_count = defaultdict(int)

        for pid, tid in player_team.items():
            if tid not in self.team_to_idx:
                continue
            tidx = self.team_to_idx[tid]
            team_member_count[tidx] += 1
            if pid in death_times:
                team_dead_count[tidx] += 1
                et = death_times[pid]
                if tidx not in team_last_death or et > team_last_death[tidx]:
                    team_last_death[tidx] = et

        self.team_member_count = dict(team_member_count)

        # 팀 탈락 시점 → 스냅샷 인덱스 변환
        # (전 멤버 사망 = 팀 탈락)
        self.team_death_step = {}  # team_idx → step index
        self.team_death_elapsed = {}
        self.team_censored = set()  # 매치 종료 시 생존한 팀

        for tidx in range(self.n_teams):
            if tidx in team_last_death and team_dead_count.get(tidx, 0) == team_member_count.get(tidx, 0):
                # 전원 사망
                death_t = team_last_death[tidx]
                self.team_death_elapsed[tidx] = death_t
                # 가장 가까운 스냅샷 인덱스 찾기
                step = self._elapsed_to_step(death_t)
                self.team_death_step[tidx] = step
            else:
                # 생존 (censored)
                self.team_censored.add(tidx)

        # ── Zone phase 계산 ──
        # poison_radius가 변하는 시점을 phase 경계로 사용
        self.zone_phases = self._compute_zone_phases()

        # ── 팀 생존 텐서 ──
        # team_alive_at[team_idx, step] = 1 if alive
        self.team_alive_at = torch.ones(self.n_teams, self.n_steps, dtype=torch.bool)
        for tidx, step in self.team_death_step.items():
            if step + 1 < self.n_steps:
                self.team_alive_at[tidx, step + 1:] = False

        # ── 초기 zone 면적 (정규화 기준) ──
        if self.n_steps > 0:
            z0 = self.graphs[0]["zone"].x[0]
            # zone context: [safe_cx, safe_cy, safe_r, poison_cx, poison_cy, poison_r, area, density, alive, time]
            self.initial_zone_area = max(z0[6].item(), 1.0)
        else:
            self.initial_zone_area = 1.0

    def _elapsed_to_step(self, elapsed):
        """elapsed time → 가장 가까운 스냅샷 인덱스."""
        best_i, best_d = 0, float("inf")
        for i, st in enumerate(self.snapshot_times):
            d = abs(st - elapsed)
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    def _compute_zone_phases(self):
        """
        poison_radius 변화를 감지해서 zone phase 번호 부여.
        poison_r이 감소하기 시작하면 새 phase.
        """
        phases = np.zeros(self.n_steps, dtype=np.int32)
        if self.n_steps == 0:
            return phases

        current_phase = 0
        prev_pr = self.graphs[0]["zone"].x[0, 5].item()
        shrinking = False

        for i in range(1, self.n_steps):
            pr = self.graphs[i]["zone"].x[0, 5].item()
            if not shrinking and pr < prev_pr - 1.0:
                # 축소 시작
                shrinking = True
            elif shrinking and abs(pr - prev_pr) < 0.5:
                # 축소 멈춤 → 새 phase
                shrinking = False
                current_phase += 1
            phases[i] = current_phase
            prev_pr = pr

        return phases

    def get_alive_teams_at(self, step):
        """step 시점에 생존한 팀 인덱스 리스트."""
        return torch.where(self.team_alive_at[:, step])[0].tolist()

    def get_zone_area_normalized(self, step):
        """정규화된 zone 면적 (초기 대비 비율, 0~1)."""
        if step >= self.n_steps:
            return 0.0
        area = self.graphs[step]["zone"].x[0, 6].item()
        return min(area / self.initial_zone_area, 1.0)


# ============================================================
# 2. 팀-팀 상호작용 그래프 구성
# ============================================================

def build_team_graph(player_graph, team_indices, n_teams, alive_team_set):
    """
    플레이어 그래프의 encounter 엣지를 집계해서
    팀-팀 상호작용 그래프를 구성.

    Parameters:
        player_graph: HeteroData (단일 스냅샷)
        team_indices: tensor [n_players] (각 플레이어의 팀 인덱스)
        n_teams: int
        alive_team_set: set of alive team indices

    Returns:
        Data with:
          - x: [n_alive_teams, team_feat_dim]  팀 노드 피처
          - edge_index: [2, n_edges]  팀-팀 엣지
          - edge_attr: [n_edges, edge_feat_dim]  엣지 피처
          - team_map: dict {team_idx: local_node_idx}
    """
    alive_teams = sorted(alive_team_set)
    n_alive = len(alive_teams)
    team_to_local = {t: i for i, t in enumerate(alive_teams)}

    if n_alive == 0:
        return _empty_team_graph()

    # ── 팀 노드 피처: 멤버 통계 집계 ──
    player_x = player_graph["player"].x  # [n_players, 14]
    n_players = player_x.shape[0]

    # 플레이어 → 팀 매핑 (이 스냅샷에 존재하는 플레이어만)
    # player_graph의 account_ids와 team_ids 사용
    p_team_idx = player_graph["player"].team_idx  # [n_players]

    team_feats = []
    team_member_positions = defaultdict(list)  # team_local → [(x, y)]

    for local_t in range(n_alive):
        global_t = alive_teams[local_t]
        # 이 팀 소속 플레이어 마스크
        mask = (p_team_idx == global_t)
        if mask.sum() == 0:
            # 이 스냅샷에 데이터 없음 (positions 누락)
            team_feats.append(torch.zeros(8))
            continue

        members = player_x[mask]  # [k, 14]

        # 팀 집계 피처 (평균 + 최소 + 분산 기반)
        mean_hp = members[:, 3].mean()
        min_hp = members[:, 3].min()
        alive_count = mask.sum().float()
        mean_zone_dist = members[:, 7].mean()      # poison center dist
        min_zone_boundary = members[:, 8].min()     # 가장 가까운 멤버의 poison 경계 거리
        inside_ratio = members[:, 9].mean()         # poison 내부 비율
        total_dmg_dealt = members[:, 12].sum()
        total_dmg_taken = members[:, 13].sum()

        feat = torch.tensor([
            mean_hp.item(),
            min_hp.item(),
            alive_count.item(),
            mean_zone_dist.item(),
            min_zone_boundary.item(),
            inside_ratio.item(),
            total_dmg_dealt.item(),
            total_dmg_taken.item(),
        ])
        team_feats.append(feat)

        # 위치 수집 (팀 중심 계산용)
        for j in range(members.shape[0]):
            team_member_positions[local_t].append(
                (members[j, 0].item(), members[j, 1].item())
            )

    team_x = torch.stack(team_feats)  # [n_alive, 8]

    # ── 팀 중심 위치 ──
    team_centers = {}
    for lt, positions in team_member_positions.items():
        if positions:
            cx = np.mean([p[0] for p in positions])
            cy = np.mean([p[1] for p in positions])
            team_centers[lt] = (cx, cy)

    # ── 팀-팀 엣지: encounter 엣지 집계 ──
    enc_ei = player_graph["player", "encounter", "player"].edge_index
    enc_attr = player_graph["player", "encounter", "player"].edge_attr

    # 팀 쌍별 상호작용 집계
    pair_stats = defaultdict(lambda: {"count": 0, "min_dist": float("inf"),
                                       "total_dist": 0, "dmg_exchange": 0})

    for e in range(enc_ei.shape[1]):
        si, di = enc_ei[0, e].item(), enc_ei[1, e].item()
        st = p_team_idx[si].item()
        dt = p_team_idx[di].item()

        if st == dt:
            continue  # 같은 팀
        if st not in team_to_local or dt not in team_to_local:
            continue

        lt_s, lt_d = team_to_local[st], team_to_local[dt]
        key = (min(lt_s, lt_d), max(lt_s, lt_d))

        dist = enc_attr[e, 0].item() if enc_attr.shape[0] > 0 else 0
        pair_stats[key]["count"] += 1
        pair_stats[key]["min_dist"] = min(pair_stats[key]["min_dist"], dist)
        pair_stats[key]["total_dist"] += dist

    # 교전 데미지 반영 (player dmg_dealt/taken 기반)
    for lt in range(n_alive):
        gt = alive_teams[lt]
        mask = (p_team_idx == gt)
        if mask.sum() == 0:
            continue
        members = player_x[mask]
        dealt = members[:, 12].sum().item()
        taken = members[:, 13].sum().item()
        # 이 팀이 관여된 모든 페어에 분배
        for (a, b), stats in pair_stats.items():
            if a == lt or b == lt:
                stats["dmg_exchange"] += dealt + taken

    # 엣지 구성
    edge_src, edge_dst = [], []
    edge_feats = []

    for (a, b), stats in pair_stats.items():
        avg_dist = stats["total_dist"] / max(stats["count"], 1)

        # 팀 중심 거리
        if a in team_centers and b in team_centers:
            ca, cb = team_centers[a], team_centers[b]
            center_dist = np.sqrt((ca[0]-cb[0])**2 + (ca[1]-cb[1])**2)
        else:
            center_dist = avg_dist

        feat = [
            stats["min_dist"],         # 최근접 멤버 거리
            center_dist,               # 팀 중심 거리
            float(stats["count"]),     # encounter 엣지 수
            stats["dmg_exchange"],     # 교전 데미지 교환량
        ]

        # 양방향
        edge_src.extend([a, b])
        edge_dst.extend([b, a])
        edge_feats.extend([feat, feat])

    # 완전 연결 추가 (encounter 없는 팀 쌍에도 최소 연결)
    # → 가까운 팀은 encounter 없어도 위협이 될 수 있으므로
    for i in range(n_alive):
        for j in range(i + 1, n_alive):
            if (i, j) not in pair_stats:
                if i in team_centers and j in team_centers:
                    ci, cj = team_centers[i], team_centers[j]
                    cd = np.sqrt((ci[0]-cj[0])**2 + (ci[1]-cj[1])**2)
                    if cd < 2000:  # 2km 이내만
                        feat = [cd, cd, 0.0, 0.0]
                        edge_src.extend([i, j])
                        edge_dst.extend([j, i])
                        edge_feats.extend([feat, feat])

    tg = Data()
    tg.x = team_x
    tg.num_nodes = n_alive

    if edge_src:
        tg.edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        tg.edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
    else:
        tg.edge_index = torch.zeros((2, 0), dtype=torch.long)
        tg.edge_attr = torch.zeros((0, 4), dtype=torch.float32)

    tg.team_map = team_to_local
    tg.alive_teams = alive_teams
    return tg


def _empty_team_graph():
    tg = Data()
    tg.x = torch.zeros((0, 8), dtype=torch.float32)
    tg.num_nodes = 0
    tg.edge_index = torch.zeros((2, 0), dtype=torch.long)
    tg.edge_attr = torch.zeros((0, 4), dtype=torch.float32)
    tg.team_map = {}
    tg.alive_teams = []
    return tg


# ============================================================
# 3. 데이터셋
# ============================================================

class TeamSurvivalDataset(Dataset):
    """
    (match, team, timestep) 단위 생존 분석 데이터셋.

    Parameters:
        pt_dir: str — .pt 파일들이 있는 디렉토리
        window_size: int — 입력 윈도우 크기 (L 스텝, 기본 5 = 50초)
        min_alive_teams: int — 최소 생존 팀 수 (너무 이른 단계 스킵)
        skip_first_steps: int — 초반 N 스텝 스킵 (비행기/낙하 단계)
        stride: int — 스텝 간격 (매 stride번째 스텝만 샘플링)

    샘플 구조:
        dict with keys:
          - player_graphs: list[HeteroData], len=window_size
          - team_idx: int (글로벌 팀 인덱스)
          - team_mask_per_step: list[Tensor] (각 스텝에서 해당 팀 플레이어 mask)
          - team_graph: Data (마지막 스텝의 팀-팀 그래프)
          - zone_seq: Tensor [window_size, zone_dim]
          - event: int (0 or 1)
          - at_risk: int (0 or 1)
          - censored: int (0 or 1)
          - meta: dict
    """

    # 노드 피처 차원 (main.py 기준)
    PLAYER_FEAT_DIM = 14
    ZONE_DIM = 10
    TEAM_FEAT_DIM = 8
    TEAM_EDGE_DIM = 4

    def __init__(
        self,
        pt_dir,
        window_size=5,
        min_alive_teams=3,
        skip_first_steps=10,
        stride=1,
    ):
        super().__init__()
        self.window_size = window_size
        self.min_alive_teams = min_alive_teams
        self.skip_first_steps = skip_first_steps
        self.stride = stride

        # .pt 파일 로드
        pt_files = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files in {pt_dir}")

        print(f"[TeamSurvivalDataset] {len(pt_files)}개 매치 로딩...")
        self.matches = []
        for fp in pt_files:
            try:
                m = MatchSurvivalData(fp)
                if m.n_steps >= window_size + skip_first_steps:
                    self.matches.append(m)
            except Exception as e:
                print(f"  스킵: {fp} ({e})")

        print(f"  유효 매치: {len(self.matches)}개")

        # ── 샘플 인덱스 구성 ──
        # 각 샘플 = (match_idx, team_idx, step)
        self.samples = []
        for mi, match in enumerate(self.matches):
            # 유효 스텝 범위
            start_step = max(self.skip_first_steps, self.window_size - 1)
            end_step = match.n_steps - 1  # 마지막 스텝은 라벨용으로 필요하므로 -1

            for step in range(start_step, end_step, self.stride):
                alive_teams = match.get_alive_teams_at(step)
                if len(alive_teams) < self.min_alive_teams:
                    continue

                for tidx in alive_teams:
                    self.samples.append((mi, tidx, step))

        print(f"  총 샘플: {len(self.samples):,}개")

        # 통계
        n_events = sum(
            1 for mi, tidx, step in self.samples
            if tidx in self.matches[mi].team_death_step
            and self.matches[mi].team_death_step[tidx] == step
        )
        print(f"  이벤트(탈락): {n_events:,}개 "
              f"({100*n_events/max(len(self.samples),1):.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mi, tidx, step = self.samples[idx]
        match = self.matches[mi]

        # ── 윈도우 그래프 시퀀스 ──
        w_start = max(0, step - self.window_size + 1)
        w_end = step + 1
        player_graphs = match.graphs[w_start:w_end]

        # 윈도우가 부족하면 첫 그래프로 패딩
        while len(player_graphs) < self.window_size:
            player_graphs.insert(0, player_graphs[0])

        # ── 팀 마스크 (각 스텝에서 해당 팀 플레이어 boolean) ──
        team_masks = []
        for g in player_graphs:
            t_idx_tensor = g["player"].team_idx
            mask = (t_idx_tensor == tidx)
            team_masks.append(mask)

        # ── Zone 시퀀스 ──
        zone_seq = torch.stack([g["zone"].x[0] for g in player_graphs])

        # ── 팀-팀 그래프 (마지막 스텝) ──
        last_graph = player_graphs[-1]
        alive_set = set(match.get_alive_teams_at(step))
        team_graph = build_team_graph(
            last_graph,
            last_graph["player"].team_idx,
            match.n_teams,
            alive_set,
        )

        # ── 라벨 ──
        # event: 이 스텝에서 (정확히는 step ~ step+1 구간에서) 팀이 탈락했는가
        event = 0
        if tidx in match.team_death_step:
            death_step = match.team_death_step[tidx]
            if death_step == step or death_step == step + 1:
                event = 1

        at_risk = 1  # 이 데이터셋은 at_risk인 샘플만 수집
        censored = 1 if tidx in match.team_censored else 0

        # ── 메타데이터 ──
        elapsed = match.snapshot_times[step]
        alive_teams_count = len(alive_set)
        norm_zone_area = match.get_zone_area_normalized(step)
        zone_phase = int(match.zone_phases[step])

        meta = {
            "match_id": match.match_id,
            "team_id": match.team_ids[tidx],
            "team_idx": tidx,
            "step": step,
            "elapsed": elapsed,
            "final_rank": match.team_ranks.get(tidx, -1),
            "zone_phase": zone_phase,
            "alive_teams": alive_teams_count,
            "norm_zone_area": norm_zone_area,
            "team_members_total": match.team_member_count.get(tidx, 0),
        }

        return {
            "player_graphs": player_graphs,
            "team_masks": team_masks,
            "team_graph": team_graph,
            "zone_seq": zone_seq,
            "event": event,
            "at_risk": at_risk,
            "censored": censored,
            "meta": meta,
        }


# ============================================================
# 4. Collate (배치 구성)
# ============================================================

def collate_survival_batch(batch):
    """
    가변 크기 그래프를 배치로 묶는 커스텀 collate.

    반환:
      batched_player_graphs: list[list[HeteroData]]  [B, L]
      batched_team_masks: list[list[Tensor]]  [B, L]
      team_graphs: list[Data]  [B]
      zone_seqs: Tensor [B, L, zone_dim]
      events: Tensor [B]
      at_risks: Tensor [B]
      censoreds: Tensor [B]
      metas: list[dict]  [B]
    """
    B = len(batch)

    return {
        "player_graphs": [b["player_graphs"] for b in batch],
        "team_masks": [b["team_masks"] for b in batch],
        "team_graphs": [b["team_graph"] for b in batch],
        "zone_seqs": torch.stack([b["zone_seq"] for b in batch]),
        "events": torch.tensor([b["event"] for b in batch], dtype=torch.float32),
        "at_risks": torch.tensor([b["at_risk"] for b in batch], dtype=torch.float32),
        "censoreds": torch.tensor([b["censored"] for b in batch], dtype=torch.float32),
        "metas": [b["meta"] for b in batch],
    }


# ============================================================
# 5. 생존 분석 손실함수
# ============================================================

def discrete_survival_nll(hazard_logits, events, at_risks):
    """
    Discrete-time survival negative log-likelihood.

    h_i(k) = sigmoid(logit_i(k)) = P(T_i = k | T_i >= k)

    NLL = -Σ [event * log(h) + (1 - event) * log(1 - h)]
    at_risk 샘플만 사용.

    Parameters:
        hazard_logits: Tensor [B] — 모델 출력 (sigmoid 전)
        events: Tensor [B] — 0 or 1
        at_risks: Tensor [B] — 0 or 1

    Returns:
        loss: scalar
    """
    mask = at_risks > 0.5
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)

    h_logits = hazard_logits[mask]
    y = events[mask]

    # Binary cross-entropy (numerically stable)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        h_logits, y, reduction="mean"
    )
    return loss


def pairwise_rank_loss(risk_scores, final_ranks, margin=0.1):
    """
    보조 손실: 팀 순위 기반 pairwise ranking loss.
    risk score가 높은 팀이 먼저 탈락해야 함 (rank 숫자가 큼 = 일찍 탈락).

    Parameters:
        risk_scores: Tensor [B] — 모델이 출력한 위험 점수
        final_ranks: Tensor [B] — 최종 순위 (1=치킨디너, 높을수록 일찍 탈락)
        margin: float

    Returns:
        loss: scalar
    """
    B = risk_scores.shape[0]
    if B < 2:
        return torch.tensor(0.0, requires_grad=True)

    # 같은 매치 내에서만 비교해야 하지만,
    # 프로토타입에서는 배치 내 랜덤 페어로 근사
    n_pairs = min(B * 2, B * (B - 1) // 2)
    idx = torch.randint(0, B, (n_pairs, 2))

    ri, rj = final_ranks[idx[:, 0]], final_ranks[idx[:, 1]]
    si, sj = risk_scores[idx[:, 0]], risk_scores[idx[:, 1]]

    # rank가 큰 (일찍 탈락) 쪽이 risk score가 높아야 함
    # → ri > rj이면 si > sj + margin 이어야 함
    target = torch.sign(ri.float() - rj.float())
    diff = si - sj

    loss = torch.nn.functional.margin_ranking_loss(
        si, sj, target, margin=margin, reduction="mean"
    )
    return loss


# ============================================================
# 6. 검증 유틸리티
# ============================================================

def species_area_fit(match_data):
    """
    단일 매치에서 log(alive_teams) vs log(zone_area) 선형 회귀.
    Species-Area relationship S = cA^z 검증용.

    Returns:
        z: float (기울기, 생태학 z값)
        c: float (절편)
        r2: float (결정계수)
        points: list of (log_area, log_teams)
    """
    points = []
    for step in range(match_data.n_steps):
        alive = match_data.get_alive_teams_at(step)
        n_alive = len(alive)
        if n_alive < 2:
            continue
        area = match_data.graphs[step]["zone"].x[0, 6].item()
        if area < 1:
            continue
        points.append((np.log(area), np.log(n_alive)))

    if len(points) < 3:
        return None, None, None, points

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    # 단순 최소제곱
    A = np.vstack([x, np.ones(len(x))]).T
    result = np.linalg.lstsq(A, y, rcond=None)
    z_val, log_c = result[0]
    c_val = np.exp(log_c)

    # R²
    y_pred = z_val * x + log_c
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-10)

    return z_val, c_val, r2, points


# ============================================================
# 메인: 데이터셋 검증
# ============================================================

if __name__ == "__main__":
    import sys

    pt_dir = sys.argv[1] if len(sys.argv) > 1 else "data/graphs"

    print("=" * 50)
    print("TeamSurvivalDataset 검증")
    print("=" * 50)

    # 단일 매치 전처리 테스트
    import glob
    pt_files = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))
    if not pt_files:
        print(f"Error: {pt_dir}에 .pt 파일 없음")
        sys.exit(1)

    print(f"\n[1] MatchSurvivalData 테스트: {pt_files[0]}")
    m = MatchSurvivalData(pt_files[0])
    print(f"  매치: {m.match_id[:20]}...")
    print(f"  팀: {m.n_teams}, 스텝: {m.n_steps}")
    print(f"  팀 탈락: {len(m.team_death_step)}, 생존: {len(m.team_censored)}")
    print(f"  Zone phases: {m.zone_phases.max() + 1}개")
    print(f"  초기 zone 면적: {m.initial_zone_area:.0f} m²")

    # 팀 생존 타임라인
    print(f"\n  팀 생존 타임라인 (샘플):")
    for tidx in range(min(5, m.n_teams)):
        alive_steps = m.team_alive_at[tidx].sum().item()
        rank = m.team_ranks.get(tidx, "?")
        death = m.team_death_step.get(tidx, "survived")
        print(f"    Team {tidx}: rank={rank}, alive_steps={alive_steps}, death_step={death}")

    # Species-Area 검증
    print(f"\n[2] Species-Area Relationship 검증")
    z_val, c_val, r2, pts = species_area_fit(m)
    if z_val is not None:
        print(f"  S = {c_val:.2f} * A^{z_val:.4f}")
        print(f"  R² = {r2:.4f}")
        print(f"  데이터 포인트: {len(pts)}개")
        # 생태학 참고: 대륙 z ≈ 0.15, 해양섬 z ≈ 0.25~0.35
        print(f"  (참고: 생태학 대륙 z≈0.15, 해양섬 z≈0.25~0.35)")
    else:
        print("  데이터 부족")

    # 팀-팀 그래프 테스트
    print(f"\n[3] 팀-팀 그래프 테스트 (중간 스텝)")
    mid_step = m.n_steps // 2
    g = m.graphs[mid_step]
    alive_set = set(m.get_alive_teams_at(mid_step))
    tg = build_team_graph(g, g["player"].team_idx, m.n_teams, alive_set)
    print(f"  스텝 {mid_step}: alive_teams={tg.num_nodes}, "
          f"edges={tg.edge_index.shape[1]}, "
          f"node_feat={tg.x.shape}, edge_feat={tg.edge_attr.shape}")

    # 데이터셋 구성 테스트
    print(f"\n[4] TeamSurvivalDataset 구성")
    ds = TeamSurvivalDataset(
        pt_dir,
        window_size=5,
        min_alive_teams=3,
        skip_first_steps=10,
        stride=2,
    )

    if len(ds) > 0:
        print(f"\n[5] 샘플 검증")
        sample = ds[0]
        print(f"  player_graphs: {len(sample['player_graphs'])} 스텝")
        print(f"  team_masks: {[m.shape for m in sample['team_masks']]}")
        print(f"  team_graph: nodes={sample['team_graph'].num_nodes}, "
              f"edges={sample['team_graph'].edge_index.shape[1]}")
        print(f"  zone_seq: {sample['zone_seq'].shape}")
        print(f"  event={sample['event']}, at_risk={sample['at_risk']}, "
              f"censored={sample['censored']}")
        print(f"  meta: {sample['meta']}")

        # 배치 테스트
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=4, collate_fn=collate_survival_batch,
                           shuffle=True)
        batch = next(iter(loader))
        print(f"\n[6] 배치 검증")
        print(f"  zone_seqs: {batch['zone_seqs'].shape}")
        print(f"  events: {batch['events']}")
        print(f"  censoreds: {batch['censoreds']}")

        # 손실함수 테스트
        print(f"\n[7] 손실함수 테스트")
        dummy_logits = torch.randn(4)
        nll = discrete_survival_nll(dummy_logits, batch["events"], batch["at_risks"])
        print(f"  survival NLL: {nll.item():.4f}")

        dummy_ranks = torch.tensor([m["final_rank"] for m in batch["metas"]], dtype=torch.float32)
        rl = pairwise_rank_loss(dummy_logits, dummy_ranks)
        print(f"  rank loss: {rl.item():.4f}")

    print("\n완료.")