"""
[한] 동적 시공간 그래프 신경망(Dynamic Spatio-Temporal GNN) - PUBG 페이즈 단위 팀 생존 모델링.
    - 입력:
        X : (B, T=8, N<=128 패딩, F) - 각 배치, 시간(페이즈), 노드(팀), 피처 차원의 노드 특성 텐서
        A : (B, T=8, E=3, N<=128 패딩, N<=128 패딩) - 페이즈별 다중 엣지 유형(근접, 교전 등) 인접 행렬
    - 출력:
        hazard_logits : (B, T-1=7, N) - 다음 페이즈 탈락 위험도(Hazard) 예측 로짓 (주 작업)
        rank_pred     : (B, N)        - 정규화된 최종 순위 보조 예측 (Auxiliary Task 1)
        topk_logits   : (B, N, 2)     - [Top-5, Top-10] 잔존 여부 보조 로짓 (Auxiliary Task 2)

[EN] Dynamic Spatio-Temporal GNN for phase-level PUBG team survival modeling.
    - Inputs:
        X : (B, T=8, N<=128 padded, F) - Node feature tensor across batch, phase cut, team node, and features
        A : (B, T=8, E=3, N<=128 padded, N<=128 padded) - Dense multi-relational adjacency tensor
    - Outputs:
        hazard_logits : (B, T-1=7, N)    - Next-phase elimination hazard logits (Primary task)
        rank_pred     : (B, N)           - Normalized final-rank auxiliary prediction
        topk_logits   : (B, N, 2)        - [top5, top10] survival auxiliary logits
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
# [한] 독립 스크립트 실행과 패키지 임포트 방식 모두를 지원하기 위한 워크스페이스 경로 등록 블록
# [EN] Path setup block to support both standalone direct execution and package imports
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalTemporalGatedConv(nn.Module):
    """
    [한] 인과적 1D 시간 게이트 합성곱 (Causal 1D Temporal Convolution with GLU).
         - 역할: 미래 시점의 정보 누설을 엄격히 차단(A3 인과 법칙 준수)하면서, 과거부터 현재 페이즈(t<=t0)까지의
           시간적 변화 패턴을 1D 합성곱과 GLU(Gated Linear Unit) 게이팅 메커니즘을 통해 추출합니다.
         - 구현: 커널 크기 K에 대해 좌측에만 (K-1) 패딩을 적용하여 미래 프레임 참조를 원천 방지하고,
           출력 채널을 2배(2 * c_out)로 확장한 뒤 h = a * sigmoid(b) 형태로 게이트 활성화를 수행합니다.

    [EN] Causal 1D Temporal Gated Convolution with GLU activation.
         - Role: Extracts temporal sequence dynamics from past to current phases (t<=t0) without
           future information leakage (satisfying A3 causality), utilizing 1D temporal convolution with GLU.
         - Implementation: Applies asymmetric left-padding of (kernel_size - 1) to strictly prevent future
           frame lookahead, projects to 2 * c_out channels, and applies a GLU activation (a * sigmoid(b)).
    """

    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3) -> None:
        super().__init__()
        # [한] 인과성 보장을 위한 좌측 패딩 크기 계산 (미래 시점 참조 방지)
        # [EN] Left padding size to ensure causal temporal convolution (prevents lookahead)
        self.left_pad = kernel_size - 1
        # [한] GLU 활성화(절반은 선형, 절반은 게이트)를 위해 채널 수를 2 * c_out으로 설정하는 Conv2D
        # [EN] 2D Conv with kernel (kernel_size, 1) producing 2 * c_out channels for GLU gating
        self.conv = nn.Conv2d(
            c_in,
            2 * c_out,
            kernel_size=(kernel_size, 1),
            padding=(0, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [한] 입력 형상: (B, C, T, N) - 배치, 채널, 시간(페이즈), 노드(팀)
        # [EN] Input shape: (B, C, T, N) - Batch, Channels, Time (phases), Nodes (teams)
        
        # [한] 시간 축(T)의 좌측(과거 방향)에만 패딩을 추가: pad = (left_N, right_N, left_T, right_T)
        # [EN] Pad only on the past side along the temporal dimension T: (pad_left, pad_right, pad_top, pad_bottom)
        x = F.pad(x, (0, 0, self.left_pad, 0))
        h = self.conv(x)
        
        # [한] GLU 연산: 채널 차원(dim=1)을 반으로 나누어 값 벡터 a와 게이트 벡터 b를 구하고 a * sigmoid(b) 계산
        # [EN] GLU activation: split channel dimension into linear vector `a` and gate vector `b`, returning a * sigmoid(b)
        a, b = h.chunk(2, dim=1)
        return a * torch.sigmoid(b)


class HeterogeneousSpatialGCN(nn.Module):
    """
    [한] 다중 엣지 유형에 대한 조밀 페이즈별 이종 그래프 합성곱 (Dense Heterogeneous Spatial GCN).
         - 역할: 각 게임 페이즈에서 팀 간 상호작용(예: 지리적 근접성, 교전 발생, 시야 공유 등 E개 엣지 유형)을
           반영하는 동적 이종 그래프로부터 이웃 노드의 특징을 집약(aggregation)합니다.
         - 구현: 엣지 유형별 독립적인 선형 변환 가중치(edge_linears)와 학습 가능한 엣지 중요도 로짓(edge_logits)에
           소프트맥스를 취해 가중합을 계산하며, 잔차 연결(skip connection)을 통해 노드 고유의 특징을 보존합니다.

    [EN] Dense Per-Phase Graph Convolution over Heterogeneous Edge Types.
         - Role: Aggregates neighbor representations across multiple interaction types (e.g., spatial
           proximity, combat engagement edges) in a dynamic game phase graph.
         - Implementation: Employs type-specific linear projections, weights their contributions via
           softmax over learnable edge logits, and adds a residual skip connection to preserve ego-features.
    """

    def __init__(self, c_in: int, c_out: int, num_edge_types: int) -> None:
        super().__init__()
        # [한] 각 엣지 유형별 독립 선형 변환 레이어 리스트
        # [EN] Independent linear transformations for each edge relation type
        self.edge_linears = nn.ModuleList(
            nn.Linear(c_in, c_out, bias=False) for _ in range(num_edge_types)
        )
        # [한] 엣지 유형별 상대적 중요도를 학습하기 위한 파라미터 로짓 (소프트맥스로 정규화)
        # [EN] Learnable logits representing the relative importance of each edge type
        self.edge_logits = nn.Parameter(torch.zeros(num_edge_types))
        # [한] 노드 자체의 입력 특징을 보존하기 위한 선형 잔차 변환 (Skip connection)
        # [EN] Linear skip connection preserving ego-node features
        self.skip = nn.Linear(c_in, c_out)

    def forward(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        edge_active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # [한] x 형상: (B, C, T, N), A 형상: (B, T, E, N, N)
        # [EN] x shape: (B, C, T, N), A shape: (B, T, E, N, N)

        # [한] 선형 변환 및 행렬 곱을 위해 노드 특성을 (B, T, N, C) 형태로 전치
        # [EN] Permute to (B, T, N, C) for linear projections and batch matrix multiplication
        xp = x.permute(0, 2, 3, 1)

        # [한] 활성 엣지 마스크가 주어지지 않은 경우 모든 엣지 유형을 기본 활성화
        # [EN] Default to all edge types active if no mask is supplied
        if edge_active_mask is None:
            edge_active_mask = torch.ones_like(self.edge_logits, dtype=torch.bool)
        edge_active_mask = edge_active_mask.to(device=self.edge_logits.device, dtype=torch.bool)

        out = torch.zeros(
            (*xp.shape[:-1], self.skip.out_features),
            dtype=xp.dtype,
            device=xp.device,
        )

        # [한] 활성화된 엣지들에 대해서만 소프트맥스 가중치 계산 (비활성 엣지는 -1e9로 마스킹)
        # [EN] Compute softmax weights over active edges only (mask inactive edges with -1e9)
        if edge_active_mask.any():
            masked_logits = self.edge_logits.masked_fill(~edge_active_mask, -1e9)
            edge_w = torch.softmax(masked_logits, dim=0)
        else:
            edge_w = torch.zeros_like(self.edge_logits)

        # [한] 엣지 유형별로 인접 행렬과 노드 특성을 행렬 곱하여 이웃 정보를 집약하고 가중합 누적
        # [EN] For each edge type, aggregate neighbor features via matrix multiplication and accumulate weighted outputs
        for e, lin in enumerate(self.edge_linears):
            if not bool(edge_active_mask[e]):
                continue
            ae = A[:, :, e]  # (B, T, N, N)
            agg = torch.matmul(ae, xp)
            out = out + edge_w[e] * lin(agg)

        # [한] 노드 자체의 특징(skip)을 더한 후 (B, C, T, N) 형태로 복원하여 반환
        # [EN] Add ego skip connection and permute back to (B, C, T, N)
        out = out + self.skip(xp)
        return out.permute(0, 3, 1, 2)

    def edge_weights(self, edge_active_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        [한] 학습된 엣지 유형별 정규화 가중치를 반환 (해석 및 분석용).
        [EN] Returns the learned normalized edge importance weights for interpretability.
        """
        if edge_active_mask is None:
            return torch.softmax(self.edge_logits.detach(), dim=0)
        mask = edge_active_mask.to(device=self.edge_logits.device, dtype=torch.bool)
        if not mask.any():
            return torch.zeros_like(self.edge_logits.detach())
        masked_logits = self.edge_logits.detach().masked_fill(~mask, -1e9)
        return torch.softmax(masked_logits, dim=0)


class DynamicSTBlock(nn.Module):
    """
    [한] 동적 시공간 블록 (Dynamic Spatio-Temporal Block - T-S-T 샌드위치 구조).
         - 역할: 시간 합성곱(t1) -> 공간 그래프 합성곱(s) -> 시간 합성곱(t2) 구조로
           각 페이즈 내의 공간적 상호작용과 페이즈 간 시간적 전이 과정을 순차적으로 통합 표현합니다.
         - 구현: LayerNorm 및 드롭아웃으로 정규화하고 입력 특징과 잔차 연결(proj/identity)을 수행합니다.

    [EN] Dynamic Spatio-Temporal Block (T-S-T Sandwich Architecture).
         - Role: Interweaves temporal and spatial operations via Temporal Conv (t1) -> Spatial GCN (s) ->
           Temporal Conv (t2) to simultaneously capture multi-agent interactions and temporal evolution.
         - Implementation: Applies LayerNorm, dropout, and a residual shortcut projection.
    """

    def __init__(
        self,
        c_in: int,
        c_hid: int,
        c_out: int,
        num_edge_types: int,
        *,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # [한] 1단계 시간 게이트 합성곱 (입력 채널 -> 은닉 채널)
        # [EN] First temporal gated convolution stage (c_in -> c_hid)
        self.t1 = CausalTemporalGatedConv(c_in, c_hid, kernel_size)
        # [한] 2단계 공간 이종 그래프 합성곱 (페이즈 내 팀 간 상호작용 집약)
        # [EN] Spatial heterogeneous graph convolution stage (intra-phase team interactions)
        self.s = HeterogeneousSpatialGCN(c_hid, c_hid, num_edge_types)
        # [한] 3단계 시간 게이트 합성곱 (은닉 채널 -> 출력 채널)
        # [EN] Second temporal gated convolution stage (c_hid -> c_out)
        self.t2 = CausalTemporalGatedConv(c_hid, c_out, kernel_size)
        # [한] 안정적 학습을 위한 LayerNorm 및 드롭아웃
        # [EN] LayerNorm and dropout for training stabilization
        self.norm = nn.LayerNorm(c_out)
        self.dropout = nn.Dropout(dropout)
        # [한] 입출력 채널 수가 다를 경우 차원을 맞춰주는 1x1 투영 잔차 연결
        # [EN] 1x1 Conv residual projection if channel dimensions differ; otherwise Identity
        self.proj = nn.Conv2d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        edge_active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # [한] 시간 합성곱 1 -> 공간 GCN + ReLU -> 시간 합성곱 2 파이프라인
        # [EN] Temporal Conv 1 -> Spatial GCN + ReLU -> Temporal Conv 2 pipeline
        h = self.t1(x)
        h = F.relu(self.s(h, A, edge_active_mask=edge_active_mask))
        h = self.t2(h)

        # [한] LayerNorm 적용을 위해 (B, T, N, C)로 전치 후 다시 원복
        # [EN] Permute to (B, T, N, C) for LayerNorm over feature channels, then permute back
        h = h.permute(0, 2, 3, 1)
        h = self.norm(h)
        h = h.permute(0, 3, 1, 2)
        h = self.dropout(h)

        # [한] 잔차 연결(Residual shortcut) 및 ReLU 활성화
        # [EN] Residual addition with input shortcut projection, followed by ReLU
        return F.relu(h + self.proj(x))


class DynamicSTGNN(nn.Module):
    """
    [한] 동적 시공간 그래프 신경망 메인 모델 (Dynamic Spatio-Temporal GNN).
         - 역할: PUBG 경기 내 128개 팀 노드의 페이즈 단위 시공간 그래프를 입력받아,
           1) 다음 페이즈 탈락 위험도(hazard_logits, 주 목표)
           2) 정규화된 최종 순위 점수(rank_pred, 보조 목표)
           3) Top-5 / Top-10 잔존 확률(topk_logits, 보조 목표)을 동시 예측합니다.
         - 특징: A3 인과성 보장(t시점의 예측은 오직 t이하의 관측 정보만 사용) 및 불변식 검증 포함.

    [EN] End-to-End Dynamic Spatio-Temporal GNN Architecture.
         - Role: Processes phase-level spatio-temporal graphs of up to 128 team nodes to jointly predict:
           1) Next-phase elimination hazards (hazard_logits, primary task)
           2) Normalized final placement ranks (rank_pred, auxiliary task)
           3) Top-5 and Top-10 survival logits (topk_logits, auxiliary task)
         - Properties: Strictly causal (A3 causality: predictions at time t observe only X<=t),
           with runtime validation of adjacency invariants.
    """

    def __init__(
        self,
        in_features: int,
        hidden: int = 64,
        mid: int = 128,
        num_blocks: int = 2,
        num_edge_types: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # [한] 입력 피처 선형 임베딩 레이어
        # [EN] Input feature projection layer to hidden dimension
        self.input = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # [한] 지정된 개수(num_blocks)만큼의 DynamicSTBlock 적층
        # [EN] Stack of multiple DynamicSTBlock layers
        blocks: list[nn.Module] = []
        for _ in range(num_blocks):
            blocks.append(
                DynamicSTBlock(
                    hidden,
                    mid,
                    hidden,
                    num_edge_types,
                    dropout=dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # [한] 주 작업 헤드: 각 페이즈별 다음 페이즈 탈락 위험도(Hazard) 예측 헤드 (출력: 1개 스칼라 로짓)
        # [EN] Primary head: Next-phase elimination hazard head per phase (outputs 1 logit per team)
        self.hazard_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

        # [한] 보조 작업 헤드 1: 최종 순위 정규화 점수 예측 헤드
        # [EN] Auxiliary head 1: Normalized final ranking prediction head
        self.rank_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

        # [한] 보조 작업 헤드 2: [Top-5, Top-10] 잔존 여부 이진 분류 로짓 헤드
        # [EN] Auxiliary head 2: [Top-5, Top-10] binary classification survival logits head
        self.topk_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )
        self._adjacency_checked = False

    def _validate_adjacency(self, A: torch.Tensor) -> None:
        """
        [한] 인접 행렬 정규화 불변식 1회성 검사 (First-forward Adjacency Invariant Check).
             - 인접 행렬 A의 각 행의 합이 1(row-stochastic) 또는 0(고립 노드/패딩)인지 확인합니다.
             - ETL/전처리 파이프라인에서 불변식이 조용히 깨지는 오류를 조기 감지합니다.

        [EN] Single runtime assertion checking that adjacency rows sum to 1 or 0 (row-stochastic).
             - Catches subtle ETL invariant violations early at model initialization.
        """
        if A.dim() != 5:
            raise ValueError(f"A must be (B,T,E,N,N), got {tuple(A.shape)}")
        rs = A.sum(dim=-1)
        ok = torch.isclose(rs, torch.ones_like(rs), atol=1e-3) | torch.isclose(
            rs, torch.zeros_like(rs), atol=1e-6
        )
        if not bool(ok.all()):
            bad = (~ok).nonzero()[0].tolist()
            raise ValueError(
                f"adjacency rows must sum to 1 (or 0 for edge-less rows); "
                f"first violation at (b,t,e,n)={bad}, row_sum={rs[tuple(bad)]:.4f}"
            )

    def forward(
        self,
        X: torch.Tensor,
        A: torch.Tensor,
        edge_active_mask: torch.Tensor | None = None,
        phase_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [한] 인접 행렬 규약 검증 (최초 1회만 실행)
        # [EN] Validate adjacency invariants on first forward call
        if not self._adjacency_checked:
            self._validate_adjacency(A)
            self._adjacency_checked = True

        # [한] 1. 노드 특성 임베딩: X (B,T,N,F) -> embed -> 전치하여 (B,C,T,N)으로 변환
        # [EN] 1. Node feature embedding: X (B,T,N,F) -> embed -> permute to (B,C,T,N)
        h = self.input(X)
        h = h.permute(0, 3, 1, 2)

        # [한] 2. 적층된 시공간 블록(ST-Blocks)을 순차 통과
        # [EN] 2. Pass through sequential spatio-temporal blocks
        for block in self.blocks:
            h = block(h, A, edge_active_mask=edge_active_mask)
        htnc = h.permute(0, 2, 3, 1)  # (B,T,N,C)

        # [한] 3. 주 작업 위험도 예측: 인과적 합성곱 덕분에 t 시점의 은닉 상태는 X<=t만 관측하므로
        #        t번째 페이즈 상태로부터 (t+1)번째 페이즈 탈락 위험도를 인과 누설 없이 예측 (B, T-1, N)
        # [EN] 3. Primary hazard prediction: Temporal convolutions are causal, so hidden[t]
        #        observes only X<=t, validly predicting next-phase elimination hazard (B, T-1, N).
        hazard_logits = self.hazard_head(htnc[:, :-1]).squeeze(-1)

        # [한] 4. 보조 작업 풀링: 관측된 유효 페이즈들에 대해 마스크 가중 평균 풀링 수행
        #        (패딩된 페이즈의 bias로 인한 노이즈 영향을 방지하기 위해 phase_mask 적용)
        # [EN] 4. Auxiliary sequence pooling: Computes masked average over valid observed phases
        #        (prevents bias noise from zero-padded phases via phase_mask).
        if phase_mask is not None:
            m = phase_mask.to(dtype=htnc.dtype, device=htnc.device)
            m = m.view(m.shape[0], m.shape[1], 1, 1)  # (B,T,1,1)
            pooled = (htnc * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        else:
            pooled = htnc.mean(dim=1)

        # [한] 5. 보조 헤드 출력 산출 (순위 예측 및 Top-K 예측)
        # [EN] 5. Auxiliary head predictions (rank prediction and top-k survival logits)
        rank_pred = self.rank_head(pooled).squeeze(-1)
        topk_logits = self.topk_head(pooled)
        return hazard_logits, rank_pred, topk_logits

    def edge_weights(self, edge_active_mask: torch.Tensor | None = None) -> list[torch.Tensor]:
        """
        [한] 모델 내 모든 ST-Block의 엣지 가중치를 리스트로 반환.
        [EN] Returns list of learned edge weights across all ST-blocks.
        """
        return [
            block.s.edge_weights(edge_active_mask=edge_active_mask)
            for block in self.blocks
            if isinstance(block, DynamicSTBlock)
        ]


def dynamic_stgnn_loss(
    hazard_logits: torch.Tensor,
    rank_pred: torch.Tensor,
    topk_logits: torch.Tensor,
    y_hazard: torch.Tensor,
    target_mask: torch.Tensor,
    y_rank: torch.Tensor,
    y_topk: torch.Tensor,
    team_mask: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 0.3,
    gamma: float = 0.3,
    hazard_pos_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    [한] Dynamic STGNN 복합 다중 작업 손실 함수 (Multi-task Loss Function).
         - 총 손실 = alpha * L_hazard + beta * L_rank + gamma * L_topk
         - 1) L_hazard: 유효 타겟(target_mask)에 대해 탈락 위험도를 평가하는 가중치 부여 BCE With Logits Loss
         - 2) L_rank: 실제 생존 팀(team_mask)에 대해 정규화 최종 순위 오차를 평가하는 L1 Loss
         - 3) L_topk: 실제 생존 팀(team_mask)에 대해 Top-5 및 Top-10 잔존 여부를 평가하는 다중 라벨 BCE Loss

    [EN] Multi-task Loss for Dynamic STGNN.
         - Total Loss = alpha * L_hazard + beta * L_rank + gamma * L_topk
         - 1) L_hazard: Weighted BCEWithLogitsLoss over active observation targets (target_mask)
         - 2) L_rank: L1 Loss between predicted and true normalized ranks on valid teams (team_mask)
         - 3) L_topk: Multi-label BCEWithLogitsLoss for Top-5 and Top-10 outcomes on valid teams
    """
    target_f = target_mask.float()
    team_f = team_mask.float()
    hazard_denom = target_f.sum().clamp(min=1.0)
    team_denom = team_f.sum().clamp(min=1.0)

    # [한] 1. Hazard BCE 손실 계산 (탈락 클래스 불균형 조절을 위한 pos_weight 반영)
    # [EN] 1. Hazard BCE loss with pos_weight adjustment for class imbalance
    pos_weight = torch.tensor(
        hazard_pos_weight,
        dtype=hazard_logits.dtype,
        device=hazard_logits.device,
    )
    hazard_bce = F.binary_cross_entropy_with_logits(
        hazard_logits,
        y_hazard,
        reduction="none",
        pos_weight=pos_weight,
    )
    l_hazard = (hazard_bce * target_f).sum() / hazard_denom

    # [한] 2. 최종 순위 L1 손실 계산 (유효 팀 노드에 대해서만 평균 산출)
    # [EN] 2. Rank prediction L1 loss averaged over valid team nodes
    l_rank = ((rank_pred - y_rank).abs() * team_f).sum() / team_denom

    # [한] 3. Top-K 잔존 BCE 손실 계산 (Top-5, Top-10 다중 라벨 이진 분류)
    # [EN] 3. Top-K survival BCE loss over valid team nodes
    topk_bce = F.binary_cross_entropy_with_logits(topk_logits, y_topk, reduction="none")
    l_topk = (topk_bce * team_f.unsqueeze(-1)).sum() / (team_denom * y_topk.size(-1))

    # [한] 가중합을 통한 총 손실 계산 및 각 구성요소 분리 로깅
    # [EN] Total weighted loss computation and component readout for experiment logging
    total = alpha * l_hazard + beta * l_rank + gamma * l_topk
    parts = {
        "loss": float(total.detach().item()),
        "l_hazard": float(l_hazard.detach().item()),
        "l_rank": float(l_rank.detach().item()),
        "l_topk": float(l_topk.detach().item()),
        # [한] 타겟이 전무한 배치(denom clamp로 loss=0)를 학습 곡선에서 식별하기 위한 유효 타겟 카운트
        # [EN] Count of valid targets to identify zero-target batches in learning curves
        "hazard_denom": float(target_f.sum().detach().item()),
    }
    return total, parts


if __name__ == "__main__":
    # [한] 실제 파이프라인 분포를 흉내낸 스모크 테스트:
    #      희소 인접 행렬 A, zero-row 허용, 생존 팀에만 self-loop, 패딩 노드/페이즈는 0
    # [EN] Smoke test mimicking production pipeline distributions:
    #      Sparse adjacency A, zero-row tolerance, self-loops only on alive teams, zeroed padding
    torch.manual_seed(0)
    B, T, E, N, F_ = 2, 8, 3, 128, 66
    n_alive, n_phase = 24, 6  # 실 노드 24팀, 실 phase 6개

    X = torch.zeros(B, T, N, F_)
    X[:, :n_phase, :n_alive] = torch.randn(B, n_phase, n_alive, F_)

    A = torch.zeros(B, T, E, N, N)
    dense = torch.rand(B, n_phase, E, n_alive, n_alive)
    dense = dense * (torch.rand_like(dense) < 0.2)  # ~80% 희소
    eye = torch.eye(n_alive)
    dense[:, :, 0] = dense[:, :, 0] + eye  # proximity self-loop
    rs = dense.sum(dim=-1, keepdim=True)
    dense = torch.where(rs > 0, dense / rs.clamp(min=1e-9), dense)  # zero-row 유지
    A[:, :n_phase, :, :n_alive, :n_alive] = dense

    phase_mask = torch.zeros(B, T, dtype=torch.bool)
    phase_mask[:, :n_phase] = True

    model = DynamicSTGNN(in_features=F_).eval()
    with torch.no_grad():
        yh, yr, yk = model(X, A, phase_mask=phase_mask)
    print(tuple(yh.shape), tuple(yr.shape), tuple(yk.shape))

    # [한] 정규화 위반 시 명확히 AssertionError/ValueError로 실패하는지 반증 시험
    # [EN] Falsification check: assert that row normalization violations raise ValueError
    try:
        bad = A.clone()
        bad[:, 0, 0, 0, :n_alive] = 0.5
        model._adjacency_checked = False
        model(X, bad)
        raise SystemExit("validation should have failed")
    except ValueError as e:
        print("adjacency validation OK:", str(e)[:60])
