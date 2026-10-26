"""
[한] PUBG 월드 모델 G1 계층형 시공간/집합 인코더 (World Model G1 Hierarchical Spatio-Temporal/Set Encoder).
    - 이론적 배경 및 영감:
        • 타입화 메시지 전달(Typed messages): HGT (Hu et al., arXiv:2003.01332) 영감.
        • 다중 개체 집합 인터페이스(Joint entity-set interface): MotionDiffuser (Jiang et al., arXiv:2306.03083).
        • 수학적 규약 및 제약 조건: WORLD_MODEL_G1_IMPLEMENTATION_2026-10-20.md 준수.
    - 주요 구성:
        1) TemporalBlock: 인과적 멀티헤드 셀프 어텐션 기반 플레이어/게임 시계열 인코딩.
        2) SpatialBlock: 관측된 이동 통행성과 가설적 공간 연결을 통합하는 그리드 셀 그래프 GNN.
        3) WorldEncoder: 플레이어, 팀 대형, 공간 그리드, 안전구역 기하학, 글로벌 게임 상태를 결합하는 엔드투엔드 인코더.

[EN] Untrained G1 Typed Temporal/Set Encoder for PUBG World Model.
    - References and Background:
        • Typed messages: Inspired by HGT (Hu et al., arXiv:2003.01332).
        • Joint entity-set interface: MotionDiffuser (Jiang et al., arXiv:2306.03083).
        • Mathematical contracts and constraints: Defined in WORLD_MODEL_G1_IMPLEMENTATION_2026-10-20.md.
    - Key Modules:
        1) TemporalBlock: Causal multi-head self-attention for player and game state histories.
        2) SpatialBlock: Grid cell GNN integrating observed passability and hypothetical spatial connectivity.
        3) WorldEncoder: End-to-end encoder joining player dynamics, team formations, spatial cells,
           safe zone geometries, and global game conditions.
"""

import torch
from torch import nn

from pubg_lab.world_model.features.inputs import EncoderInput


def mean_to(messages: torch.Tensor, destinations: torch.Tensor, size: int) -> torch.Tensor:
    """
    [한] 가변 개수의 메시지를 목적지 인덱스로 산란 가중 평균(scatter mean)하는 유틸리티 함수.
         - 역할: 인덱스 집합 destinations에 대해 messages를 index_add_로 누적하고 개수로 나누어 평균 산출.
         - 안전성: 카운트 텐서에 clamp_min(1)을 적용하여 분모가 0이 되는 것을 방지.

    [EN] Scatter-Mean Messages to Destination Indices.
         - Role: Accumulates messages into their target destinations via index_add_ and computes the mean.
         - Stability: Clamps division counts to minimum 1 to prevent division-by-zero NaNs.
    """
    out = messages.new_zeros((size, messages.shape[-1]))
    counts = messages.new_zeros((size, 1))
    if len(messages):
        out.index_add_(0, destinations, messages)
        counts.index_add_(0, destinations, messages.new_ones((len(messages), 1)))
    return out / counts.clamp_min(1)


class TemporalBlock(nn.Module):
    """
    [한] 인과적 시계열 어텐션 블록 (Causal Temporal Multi-Head Attention Block).
         - 역할: 플레이어 및 게임의 과거 시계열 궤적을 인과적(causal tril mask)으로 인코딩하여
           미래 프레임 정보 누설 없이 시간적 맥락을 학습합니다 (A3 인과 법칙).
         - 수치 안정성: 유효하지 않은(invalid) 쿼리는 자기 자신에게만 어텐션을 수행하도록 마스킹하여
           all-masked로 인한 softmax NaN 발생을 방지한 후 출력에서 0으로 마스킹합니다.

    [EN] Causal Temporal Attention Block.
         - Role: Encodes temporal trajectory history causally (lower-triangular mask) to satisfy A3 causality.
         - Numerical Stability: Invalid queries attend to themselves to avoid all-masked softmax NaNs;
           their outputs are subsequently zeroed and never serve as valid keys.
    """

    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        # [한] 배치 우선(batch_first=True) 인과적 멀티헤드 어텐션
        # [EN] MultiheadAttention with batch_first=True
        self.attention = nn.MultiheadAttention(d, heads, dropout=0, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        # [한] 2배 확장 은닉층과 GELU 활성화를 갖는 피드포워드 신경망
        # [EN] Feedforward network with 2x expansion and GELU activation
        self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        self.heads = heads

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # [한] x 형상: (P, L, d) - 플레이어 개체(P), 시계열 프레임(L), 피처 차원(d)
        # [EN] x shape: (P, L, d) - Players (P), Sequence length (L), Feature dim (d)
        P, L, _ = x.shape

        # [한] 1. 하삼각 행렬(tril)을 이용한 인과적 마스크 및 유효 프레임 마스크 결합
        # [EN] 1. Lower-triangular causal mask combined with valid history indicators
        causal = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
        allowed = causal[None] & valid[:, None, :]
        # [한] 유효하지 않은 쿼리가 all-masked softmax NaN을 일으키지 않도록 대각 성분 허용
        # [EN] Allow diagonal self-attention for invalid queries to prevent NaN in softmax
        allowed = allowed | (
            torch.eye(L, dtype=torch.bool, device=x.device)[None] & ~valid[:, :, None]
        )
        blocked = (~allowed).repeat_interleave(self.heads, dim=0)

        # [한] 2. 셀프 어텐션 및 잔차 연결 + LayerNorm
        # [EN] 2. Self-attention with residual connection and LayerNorm
        h, _ = self.attention(x, x, x, attn_mask=blocked, need_weights=False)
        x = self.norm1(x + h)
        x = self.norm2(x + self.ff(x))

        # [한] 3. 유효하지 않은 프레임의 출력을 0으로 마스킹하여 반환
        # [EN] 3. Zero-out invalid frames before returning
        return x * valid[:, :, None]


class SpatialBlock(nn.Module):
    """
    [한] 공간 그리드 셀 그래프 합성곱 블록 (Spatial Grid Cell GNN Block).
         - 역할: 맵의 공간 그리드 셀 간에 1) 실제 관측된 이동 통행성(observed passability)과
           2) 인접성에 기반한 가설적(inferred/hypothetical) 연결성을 구분하여 메시지 패싱을 수행합니다.
         - 근거 강도 스케일링: matches / (matches + 1) 형태로 관측 표본 크기에 비례하는 가중치를 부여.

    [EN] Spatial Grid Cell Graph Convolution Block.
         - Role: Propagates spatial features across terrain grid cells by distinguishing
           1) empirically observed player transitions, and 2) inferred hypothetical spatial adjacencies.
         - Evidence Weighting: Applies evidence-strength weighting (matches / (matches + 1)) to observed flows.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        # [한] 실제 관측된 이동 경로에 대한 선형 변환
        # [EN] Linear projection for empirically observed transitions
        self.observed = nn.Linear(d, d, bias=False)
        # [한] 가설적/추론된 공간 연결에 대한 선형 변환
        # [EN] Linear projection for hypothetical/inferred connections
        self.hypothetical = nn.Linear(d, d, bias=False)
        # [한] 이동 모드(도보, 차량 등 10개) 임베딩
        # [EN] Embedding for transit modes (on-foot, vehicle types, up to 10)
        self.mode = nn.Embedding(10, d)
        # [한] 관측 지원 통계(표본 수, 일치 횟수) 선형 투영
        # [EN] Linear projection of transition support statistics (sample counts, matches)
        self.support = nn.Linear(2, d, bias=False)
        # [한] 가설적 메시지의 신뢰도를 조절하는 학습 가능한 게이트 파라미터
        # [EN] Learnable gating parameter modulating hypothetical message influence
        self.hypothesis_gate = nn.Parameter(torch.tensor(-2.0))
        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        cells: torch.Tensor,
        observed: torch.Tensor,
        inferred: torch.Tensor,
    ) -> torch.Tensor:
        if not len(cells):
            return cells
        update = torch.zeros_like(cells)

        # [한] 1. 실제 관측된 이동 전이 메시지 집약 (출발 셀 특성 + 이동 수단 + 로그 지지도)
        # [EN] 1. Aggregate observed transition messages (source cell + mode + log-support)
        if len(observed):
            src, dst, mode, n, matches = observed.unbind(1)
            support = torch.stack([n, matches], -1).float()
            weight = (matches.float() / (matches + 1)).unsqueeze(-1)
            msg = (
                self.observed(cells[src]) + self.mode(mode) + self.support(torch.log1p(support))
            ) * weight
            update = update + mean_to(msg, dst, len(cells))

        # [한] 2. 가설적 인접 공간 전이 메시지 집약 (게이팅 적용)
        # [EN] 2. Aggregate hypothetical spatial adjacency messages (modulated by sigmoid gate)
        if len(inferred):
            src, dst = inferred.unbind(1)
            msg = self.hypothetical(cells[src]) * self.hypothesis_gate.sigmoid()
            update = update + mean_to(msg, dst, len(cells))

        # [한] 3. 잔차 갱신 및 LayerNorm 정규화
        # [EN] 3. Residual update and LayerNorm
        return self.norm(cells + update)


class WorldEncoder(nn.Module):
    """
    [한] PUBG 통합 월드 모델 인코더 (Unified PUBG World Model Encoder).
         - 역할: 경기 내 모든 관측 가능 개체(플레이어, 팀, 그리드 셀, 안전구역 자기장, 글로벌 게임 상황)를
           단일 계층형 잠재 공간으로 통합 인코딩합니다.
         - 구성 요소 및 인코딩 흐름:
           1) 플레이어 입력 피처 + 장비 토큰 + 이동 상태 + 신뢰도 + 접촉 피처 투영
           2) 인과적 시간 블록(TemporalBlock)으로 플레이어 궤적 인코딩
           3) 공간 그리드 그래프 GNN(SpatialBlock) 및 위치-셀 매핑
           4) 안전구역 원형 자기장과의 상대 거리/경계 기하학 인코딩
           5) 관측 가능 개체 간 상호 어텐션(player_context: O(P^2 d) 상대 기하 관계)
           6) 팀 단위 순열 등변성(Permutation Equivariance)을 보존하는 팀 대형 풀링

    [EN] Unified World Model Encoder for PUBG.
         - Role: Jointly maps all observable game entities (players, teams, terrain cells, safe zones,
           and macro game state) into unified hierarchical latent representations.
         - Pipeline:
           1) Projects player features + equipment tokens + mobility modes + confidence + contact
           2) TemporalBlock: Causal temporal trajectory encoding
           3) SpatialBlock: Terrain cell GNN and player-to-cell spatial grounding
           4) Safe zone geometric relative vectors and boundary distance encoding
           5) Inter-entity relative attention (player_context preserving permutation equivariance)
           6) Team formation pooling preserving relative squad configurations
    """

    def __init__(self, item_count: int, condition_sizes: list[int], d: int = 128, heads: int = 4) -> None:
        super().__init__()
        self.d = d
        # [한] 플레이어 기본 피처 투영 (36차원: 상태 + 결측마스크 + 절대좌표)
        # [EN] Player input feature projection (36-dim: state + mask + absolute xy)
        self.player_input = nn.Linear(36, d)
        # [한] 아이템 장비 임베딩 (패딩 인덱스 0)
        # [EN] Equipment item embedding (padding_idx=0)
        self.equipment = nn.Embedding(item_count, d, padding_idx=0)
        # [한] 이동 상태 모드 임베딩 (10개 모드)
        # [EN] Mobility mode embedding (10 discrete mobility states)
        self.mobility = nn.Embedding(10, d, padding_idx=0)
        # [한] 관측 신뢰도 선형 변환
        # [EN] Observation confidence projection
        self.confidence = nn.Linear(2, d, bias=False)
        # [한] 교전/시야 접촉 피처 선형 변환
        # [EN] Combat contact feature projection
        self.contact = nn.Linear(6, d, bias=False)
        # [한] 상대 시간 선형 투영
        # [EN] Relative time encoding projection
        self.time = nn.Linear(1, d, bias=False)
        # [한] 플레이어 궤적 시간 인코더 (2개 TemporalBlock)
        # [EN] Player trajectory temporal encoder stack (2 TemporalBlocks)
        self.temporal = nn.ModuleList([TemporalBlock(d, heads) for _ in range(2)])

        # [한] 글로벌 게임 상태 시계열 인코더 (18차원 상태 + 마스크)
        # [EN] Global game state temporal encoder (18-dim game features)
        self.game_input = nn.Linear(18, d)
        self.game_temporal = TemporalBlock(d, heads)
        # [한] 날씨, 맵 등 불연속 경기 조건 임베딩 모듈 리스트
        # [EN] Categorical condition embeddings (weather, map type, etc.)
        self.conditions = nn.ModuleList(
            [nn.Embedding(n, d, padding_idx=0) for n in condition_sizes]
        )

        # [한] 안전구역(Bluezone/Whitezone) 기하학 및 상대 위치 인코더
        # [EN] Safe zone (bluezone/whitezone) geometry and relative distance encoders
        self.zone_input = nn.Linear(6, d)
        self.zone_relative = nn.Sequential(nn.Linear(8, d), nn.GELU(), nn.Linear(d, d))

        # [한] 지형 그리드 셀 인풋 및 공간 GNN 블록 (2개 SpatialBlock)
        # [EN] Terrain grid cell inputs and spatial GNN stack (2 SpatialBlocks)
        self.cell_input = nn.Linear(8, d)
        self.spatial = nn.ModuleList([SpatialBlock(d) for _ in range(2)])
        self.cell_to_player = nn.Linear(d, d, bias=False)

        # [한] 개체 간 상호작용 컨텍스트 어텐션 (상대 거리 및 기하학 반영)
        # [EN] Inter-entity context attention (relative distance and geometry vectors)
        self.context_player = nn.Linear(d, d, bias=False)
        self.context_geometry = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.context_score = nn.Sequential(nn.Tanh(), nn.Linear(d, 1))

        # [한] 팀 대형(Team formation) 인코딩: 멤버 집약, 상대 쌍 벡터(pair), 팀 생존 카운트
        # [EN] Team formation encoding: member pooling, relative pairwise vectors, squad size counts
        self.team_member = nn.Linear(d, d)
        self.team_pair = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d))
        self.team_count = nn.Linear(3, d)
        self.team_norm = nn.LayerNorm(d)
        self.team_to_player = nn.Linear(d, d, bias=False)
        self.player_norm = nn.LayerNorm(d)

    def player_context(
        self,
        player: torch.Tensor,
        positions: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """
        [한] 관측 가능한 개체 간 상대적 기하 어텐션 컨텍스트 (O(P^2 d)).
             - 역할: 현재 시점에서 유효한 개체들 사이의 상대 변위(delta)와 유클리드 거리(norm)를
               기하학적 피처로 변환하여 상호 어텐션 가중합 메시지를 교환합니다.
             - 대칭성 (A1 법칙): 공유 쌍 연산자를 통해 플레이어 순열 등변성(Permutation Equivariance)을 보존합니다.

        [EN] Inter-Entity Pairwise Geometric Attention Context (O(P^2 d)).
             - Role: Exchanges pairwise attention messages among active entities based on relative
               displacement vectors and Euclidean distances.
             - Symmetry (A1 Law): Preserves player permutation equivariance through shared pairwise operators.
        """
        indices = torch.nonzero(valid, as_tuple=True)[0]
        if len(indices) < 2:
            return torch.zeros_like(player)
        points = positions[indices]
        # [한] 개체 간 상대 위치 벡터 및 거리 계산: (len, len, 3)
        # [EN] Relative displacement vector and Euclidean distance: (len, len, 3)
        delta = points[None] - points[:, None]
        relation = torch.cat([delta, delta.norm(dim=-1, keepdim=True)], -1)
        messages = self.context_player(player[indices])[None] + self.context_geometry(relation)
        scores = self.context_score(messages).squeeze(-1)
        # [한] 자기 자신과의 상호작용은 -inf로 마스킹하여 제외
        # [EN] Mask out diagonal self-interaction with -inf
        scores = scores.masked_fill(
            torch.eye(len(indices), dtype=torch.bool, device=player.device), -torch.inf
        )
        context = (scores.softmax(-1)[:, :, None] * messages).sum(1)
        return torch.zeros_like(player).index_copy(0, indices, context)

    def encode_spatial(self, g: EncoderInput) -> torch.Tensor:
        """
        [한] 지형 그리드 셀 특성 인코딩 및 다중 공간 블록 통과.
        [EN] Encodes terrain cell features through the spatial GNN stack.
        """
        cells = self.cell_input(g.cells)
        for block in self.spatial:
            cells = block(cells, g.observed, g.inferred)
        return cells

    def forward(self, inputs: EncoderInput) -> dict[str, torch.Tensor]:
        # [한] 입력 타입 검증 (EncoderInput 전용 규약)
        # [EN] Validate input type (strict EncoderInput protocol)
        if not isinstance(inputs, EncoderInput):
            raise TypeError("WorldEncoder accepts EncoderInput only, never a target dictionary")
        g = inputs
        L, P, _ = g.x.shape
        valid = g.history_valid

        # [한] 1. 플레이어 입력 피처 및 장비/이동/신뢰도/접촉 임베딩 결합
        # [EN] 1. Combine player input projections with equipment, mobility, confidence, and contact
        x = self.player_input(torch.cat([g.x, g.mask.float(), g.absolute_xy], -1))
        item_context = x.new_zeros((L * P, self.d))
        if len(g.items):
            frame, player, token = g.items.unbind(1)
            item_context = mean_to(self.equipment(token), frame * P + player, L * P)
        x = (
            x
            + item_context.reshape(L, P, self.d)
            + self.mobility(g.mobility) * g.mobility_mask[:, :, None]
            + self.confidence(g.confidence)
            + self.contact(torch.cat([g.x[:, :, 9:12], g.mask[:, :, 9:12].float()], -1))
        )

        # [한] 2. 상대 시간 인코딩 부여 후 인과적 시간 어텐션 통과
        # [EN] 2. Add relative time encoding and pass through causal temporal blocks
        time = torch.arange(1 - L, 1, dtype=x.dtype, device=x.device).reshape(L, 1) / 12
        x = x + self.time(time)[:, None, :]
        x = x.transpose(0, 1)
        for block in self.temporal:
            x = block(x, valid.T)
        player = x[:, -1]

        # [한] 3. 지형 그리드 셀 공간 인코딩 및 플레이어 위치-셀 투영 결합
        # [EN] 3. Encode terrain grid cells and scatter groundings to players
        cells = self.encode_spatial(g)
        if len(g.located):
            p, c = g.located.unbind(1)
            player = player + mean_to(self.cell_to_player(cells[c]), p, P)

        # [한] 4. 글로벌 게임 진행 상황 및 환경 조건 인코딩
        # [EN] 4. Encode global game progression and environmental conditions
        game_history = self.game_input(torch.cat([g.game, g.game_mask.float()], -1))
        game_history = game_history + self.time(time)
        game = self.game_temporal(game_history[None], g.game_mask.any(-1)[None])[0, -1]
        for j, embed in enumerate(self.conditions):
            game = game + embed(g.conditions[j])

        # [한] 5. 안전구역 자기장 원과의 상대 위치 벡터 및 경계 거리 기하학 결합
        # [EN] 5. Safe zone circle relative vectors and boundary distance geometry
        zone = self.zone_input(torch.cat([g.zone, g.zone_mask.float()], -1))
        zone_valid = g.zone_mask.all(-1)
        zone = zone * zone_valid[:, None]
        relative = g.zone[None, :, :2] - g.x[-1, :, None, :2]
        boundary = relative.norm(dim=-1) - g.zone[None, :, 2]
        spatial_valid = valid[-1, :, None] & zone_valid[None, :]
        geometry = torch.cat([relative, boundary[:, :, None]], -1)
        geometry = geometry * spatial_valid[:, :, None]
        relative_context = self.zone_relative(
            torch.cat([geometry.reshape(P, 6), spatial_valid.float()], -1)
        )
        player = (player + relative_context + game + zone.sum(0)) * valid[-1, :, None]

        # [한] 6. 플레이어 간 상대 기하 어텐션 컨텍스트 반영
        # [EN] 6. Apply inter-entity pairwise geometric attention
        player = player + self.player_context(player, g.x[-1, :, :2], valid[-1])

        # [한] 7. 팀 대형(Formation) 인코딩: 멤버 평균 + 상대 쌍 벡터(shape) + 생존 팀원 수
        # [EN] 7. Team formation encoding: member mean + relative pairwise vectors + alive squad size
        team_rows = []
        for q in range(g.team_count):
            indices = g.members[g.members[:, 1] == q, 0]
            selected = indices[valid[-1, indices]]
            if len(selected):
                pooled = self.team_member(player[selected]).mean(0)
            else:
                pooled = player.new_zeros(self.d)
            pair = player.new_zeros(self.d)
            n = len(selected)
            if n > 1:
                points = g.x[-1, selected, :2]
                delta = (points[:, None] - points[None])[
                    ~torch.eye(n, dtype=torch.bool, device=player.device)
                ]
                pair = self.team_pair(
                    torch.cat([delta, delta.square(), delta.norm(dim=-1, keepdim=True)], -1)
                ).mean(0)
            count = player.new_tensor([len(indices) / 10, n / 10, float(n > 1)])
            team_rows.append(self.team_norm(pooled + pair + self.team_count(count)))
        team = torch.stack(team_rows) if team_rows else player.new_zeros((0, self.d))

        # [한] 8. 팀 수준 표현을 다시 소속 팀원 플레이어에게 피드백하고 최종 LayerNorm 적용
        # [EN] 8. Scatter team-level representations back to members and apply final LayerNorm
        if len(g.members):
            p, q = g.members.unbind(1)
            player = player + mean_to(self.team_to_player(team[q]), p, P)
        player = self.player_norm(player) * valid[-1, :, None]

        return {
            "player": player,
            "team": team,
            "cell": cells,
            "zone": zone,
            "game": game,
            "player_valid": valid[-1],
        }

    def forward_batch(self, inputs: list[EncoderInput]) -> list[dict[str, torch.Tensor]]:
        """
        [한] 가변 크기 그래프 배치를 개별 순전파하여 리스트 형태로 반환.
        [EN] Forwards a list of variable-sized graph inputs individually.
        """
        return [self(g) for g in inputs]
