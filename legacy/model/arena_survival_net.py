"""
Arena Survival Network
=======================
전체 파이프라인 조립.

Agent graph (t) → Agent Encoder → Group Pooling → Group GNN → GRU → Hazard Head
                  (HeteroGNN)     (Attention)      (α_ij)     (L steps) (h_i(k))

입력: TeamSurvivalDataset 샘플 (player_graphs, team_masks, team_graph, zone_seq)
출력: hazard_logit, risk_score, alphas (경쟁 계수)
"""

import torch
import torch.nn as nn

from model.agent_encoder import AgentEncoder
from model.group_pooling import GroupPooling, GroupPoolingScatter
from model.group_gnn import GroupGNN
from model.temporal import TemporalEncoder
from model.hazard_head import HazardHead


class ArenaSurvivalNet(nn.Module):
    """
    Shrinking-Habitat Survival Network.

    단일 (match, team, timestep) 샘플을 처리하여
    해당 팀의 discrete-time hazard를 예측.
    """

    def __init__(
        self,
        agent_feat_dim=39,
        ally_edge_dim=2,
        encounter_edge_dim=2,
        team_feat_dim=8,
        team_edge_dim=4,
        zone_dim=10,
        hidden_dim=128,
        n_encoder_layers=2,
        n_group_gnn_layers=2,
        n_gru_layers=1,
        n_heads=4,
        max_phases=9,
        dropout=0.1,
        use_scatter_pooling=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 1. Agent Encoder: HeteroGNN
        self.agent_encoder = AgentEncoder(
            agent_feat_dim=agent_feat_dim,
            ally_edge_dim=ally_edge_dim,
            encounter_edge_dim=encounter_edge_dim,
            hidden_dim=hidden_dim,
            n_layers=n_encoder_layers,
            dropout=dropout,
        )

        # 2. Group Pooling: Attention
        PoolingClass = GroupPoolingScatter if use_scatter_pooling else GroupPooling
        self.group_pooling = PoolingClass(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # 3. Group GNN: 팀-팀 상호작용
        self.group_gnn = GroupGNN(
            hidden_dim=hidden_dim,
            team_feat_dim=team_feat_dim,
            edge_dim=team_edge_dim,
            n_layers=n_group_gnn_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

        # 4. Temporal: GRU
        self.temporal = TemporalEncoder(
            hidden_dim=hidden_dim,
            zone_dim=zone_dim,
            n_gru_layers=n_gru_layers,
            dropout=dropout,
        )

        # 5a. GNN skip-connection gate (gradient highway)
        self.gnn_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        # 5. Hazard Head: Phase-conditioned
        self.hazard_head = HazardHead(
            hidden_dim=hidden_dim,
            max_phases=max_phases,
            dropout=dropout,
        )

    def forward_single(self, sample):
        """
        단일 샘플 forward.

        Parameters:
            sample: dict from TeamSurvivalDataset.__getitem__
                - player_graphs: list[HeteroData], len=L
                - team_masks: list[Tensor], len=L
                - team_graph: Data
                - zone_seq: Tensor [L, zone_dim]
                - meta: dict with team_idx, zone_phase, ...

        Returns:
            hazard_logit: Tensor [1]
            risk_score: Tensor [1]
            alphas: list — 경쟁 계수 (해석용)
            attn_weights: Tensor — agent attention (해석용)
        """
        player_graphs = sample["player_graphs"]
        team_masks = sample["team_masks"]
        team_graph = sample["team_graph"]
        zone_seq = sample["zone_seq"]
        meta = sample["meta"]

        L = len(player_graphs)
        team_idx_target = meta["team_idx"]

        # ── L 스텝에 대해 agent encode → group pool ──
        team_h_seq = []
        last_attn = None

        for step in range(L):
            pg = player_graphs[step]
            n_players = pg["player"].x.shape[0]

            if n_players == 0:
                team_h_seq.append(torch.zeros(self.hidden_dim, device=zone_seq.device))
                continue

            # 1. Agent encode
            agent_h = self.agent_encoder(pg)  # [n_players, hidden_dim]

            # 2. Group pool → 팀 임베딩
            p_team_idx = pg["player"].team_idx  # [n_players]
            n_teams = p_team_idx.max().item() + 1 if n_players > 0 else 0
            team_h_all, attn_w = self.group_pooling(agent_h, p_team_idx, n_teams)

            # 타겟 팀 임베딩 추출
            if team_idx_target < team_h_all.shape[0]:
                target_team_h = team_h_all[team_idx_target]
            else:
                target_team_h = torch.zeros(self.hidden_dim, device=zone_seq.device)

            team_h_seq.append(target_team_h)
            last_attn = attn_w

        # 마지막 스텝에서 Group GNN 적용
        last_pg = player_graphs[-1]
        n_players_last = last_pg["player"].x.shape[0]
        alphas = []
        target_gnn_h = None

        if n_players_last > 0 and team_graph.num_nodes > 0:
            agent_h_last = self.agent_encoder(last_pg)
            p_team_idx_last = last_pg["player"].team_idx
            alive_teams_list = team_graph.alive_teams
            n_teams_last = max(
                p_team_idx_last.max().item() + 1,
                (max(alive_teams_list) + 1) if alive_teams_list else 0,
            )
            team_h_pooled, _ = self.group_pooling(agent_h_last, p_team_idx_last, n_teams_last)

            team_h_gnn, alphas = self.group_gnn(team_h_pooled, team_graph)

            # GNN 출력에서 타겟 팀 추출
            alive_teams = team_graph.alive_teams
            if isinstance(alive_teams, list):
                if team_idx_target in alive_teams:
                    local_idx = alive_teams.index(team_idx_target)
                    team_h_seq[-1] = team_h_seq[-1] + team_h_gnn[local_idx]
                    target_gnn_h = team_h_gnn[local_idx]

        # 3. Temporal: GRU
        team_h_seq_t = torch.stack(team_h_seq)  # [L, hidden_dim]
        h_temporal = self.temporal(team_h_seq_t, zone_seq)  # [hidden_dim]

        # GNN gradient highway
        if target_gnn_h is not None:
            gate = self.gnn_gate(torch.cat([h_temporal, target_gnn_h], dim=-1))
            h_final = h_temporal + gate * target_gnn_h
        else:
            h_final = h_temporal

        # 4. Hazard Head
        zone_phase = meta.get("zone_phase", 0)
        norm_zone_area = meta.get("norm_zone_area", 1.0)
        alive_teams_count = meta.get("alive_teams", 25)

        context = torch.tensor([
            norm_zone_area,
            alive_teams_count / 25.0,  # ~25팀 기준 정규화
            meta.get("elapsed", 0) / 1800.0,  # 30분 기준
            norm_zone_area * (alive_teams_count / 25.0),  # density proxy
        ], device=h_final.device, dtype=torch.float32)

        hazard_logit, risk_score = self.hazard_head(h_final, zone_phase, context)

        return hazard_logit, risk_score, alphas, last_attn

    def forward(self, batch):
        """
        배치 forward.

        Parameters:
            batch: collate_survival_batch 출력
                - player_graphs: list[list[HeteroData]], [B, L]
                - team_masks: list[list[Tensor]], [B, L]
                - team_graphs: list[Data], [B]
                - zone_seqs: Tensor [B, L, zone_dim]
                - metas: list[dict], [B]

        Returns:
            hazard_logits: Tensor [B]
            risk_scores: Tensor [B]
        """
        B = len(batch["player_graphs"])
        device = batch["zone_seqs"].device

        hazard_logits = []
        risk_scores = []

        for b in range(B):
            sample = {
                "player_graphs": batch["player_graphs"][b],
                "team_masks": batch["team_masks"][b],
                "team_graph": batch["team_graphs"][b],
                "zone_seq": batch["zone_seqs"][b],
                "meta": batch["metas"][b],
            }

            h_logit, r_score, _, _ = self.forward_single(sample)
            hazard_logits.append(h_logit)
            risk_scores.append(r_score)

        return torch.stack(hazard_logits), torch.stack(risk_scores)

    def compute_loss(self, batch, lambda_rank=0.1, lambda_encounter=0.05):
        """
        3-loss 계산.

        L_total = L_survival + λ₁·L_rank + λ₂·L_encounter

        Parameters:
            batch: collate_survival_batch 출력
            lambda_rank: float — ranking loss 가중치
            lambda_encounter: float — encounter loss 가중치 (현재 미구현, placeholder)

        Returns:
            loss_dict: dict with total, survival, rank, encounter
        """
        from dataset import discrete_survival_nll, pairwise_rank_loss

        hazard_logits, risk_scores = self.forward(batch)

        events = batch["events"]
        at_risks = batch["at_risks"]

        # L_survival: discrete-time NLL (pos_weight 자동 계산)
        l_survival = discrete_survival_nll(hazard_logits, events, at_risks)

        # L_rank: match-aware pairwise ranking
        final_ranks = torch.tensor(
            [m["final_rank"] for m in batch["metas"]],
            dtype=torch.float32,
            device=hazard_logits.device,
        )
        match_ids = [m["match_id"] for m in batch["metas"]]
        l_rank = pairwise_rank_loss(risk_scores, final_ranks, match_ids=match_ids)

        # L_encounter: attention 고정 (TODO: 교전 발생 예측)
        l_encounter = torch.tensor(0.0, device=hazard_logits.device)

        total = l_survival + lambda_rank * l_rank + lambda_encounter * l_encounter

        return {
            "total": total,
            "survival": l_survival,
            "rank": l_rank,
            "encounter": l_encounter,
            "hazard_logits": hazard_logits.detach(),
            "risk_scores": risk_scores.detach(),
        }

    # ================================================================
    # Snapshot-level forward (모든 생존 팀을 한번에 처리)
    # ================================================================

    def forward_snapshot(self, sample):
        """
        스냅샷 단위 forward: 한 시점의 모든 생존 팀을 한번에 처리.
        GNN이 그래프당 1회만 실행되고, 모든 팀 임베딩이 동시에 계산됨.

        Parameters:
            sample: SnapshotDataset.__getitem__ 출력 (단일 샘플)

        Returns:
            hazard_logits: Tensor [n_alive]
            risk_scores: Tensor [n_alive]
            alphas: list — GroupGNN 경쟁 계수
        """
        player_graphs = sample["player_graphs"]
        team_graph = sample["team_graph"]
        zone_seq = sample["zone_seq"]
        alive_teams = sample["alive_teams"]
        meta = sample["meta"]

        L = len(player_graphs)
        n_alive = len(alive_teams)
        device = zone_seq.device

        if n_alive == 0:
            return (torch.zeros(0, device=device),
                    torch.zeros(0, device=device), [])

        # ── L 스텝: AgentEncoder + GroupPooling ──
        all_team_h = []  # [step] → [n_alive, hidden_dim]
        cached_agent_h_last = None

        for step_i in range(L):
            pg = player_graphs[step_i]
            n_players = pg["player"].x.shape[0]

            if n_players == 0:
                all_team_h.append(
                    torch.zeros(n_alive, self.hidden_dim, device=device)
                )
                continue

            agent_h = self.agent_encoder(pg)  # [n_players, hidden_dim]

            if step_i == L - 1:
                cached_agent_h_last = agent_h

            p_team_idx = pg["player"].team_idx
            n_teams_total = p_team_idx.max().item() + 1
            team_h_all, _ = self.group_pooling(
                agent_h, p_team_idx, n_teams_total
            )  # [n_teams_total, hidden_dim]

            # alive 팀만 추출
            alive_h = []
            for tidx in alive_teams:
                if tidx < team_h_all.shape[0]:
                    alive_h.append(team_h_all[tidx])
                else:
                    alive_h.append(torch.zeros(self.hidden_dim, device=device))

            all_team_h.append(torch.stack(alive_h))  # [n_alive, hidden_dim]

        # ── GroupGNN (마지막 스텝) ──
        alphas = []
        gnn_aligned = None  # [n_alive, hidden_dim] aligned to alive_teams order
        if team_graph.num_nodes > 0 and cached_agent_h_last is not None:
            pg_last = player_graphs[-1]
            p_team_idx_last = pg_last["player"].team_idx
            n_teams_last = max(
                p_team_idx_last.max().item() + 1,
                max(alive_teams) + 1 if alive_teams else 0,
            )
            team_h_pooled_last, _ = self.group_pooling(
                cached_agent_h_last, p_team_idx_last, n_teams_last
            )

            team_h_gnn, alphas = self.group_gnn(
                team_h_pooled_last, team_graph
            )

            # GNN 출력을 마지막 스텝에 더하기 (기존 경로 유지)
            graph_alive = team_graph.alive_teams
            if isinstance(graph_alive, list):
                # GNN 출력을 alive_teams 순서에 맞게 정렬
                gnn_parts = []
                for i, tidx in enumerate(alive_teams):
                    if tidx in graph_alive:
                        gnn_local = graph_alive.index(tidx)
                        if gnn_local < team_h_gnn.shape[0]:
                            all_team_h[-1][i] = (
                                all_team_h[-1][i] + team_h_gnn[gnn_local]
                            )
                            gnn_parts.append(team_h_gnn[gnn_local])
                        else:
                            gnn_parts.append(torch.zeros(self.hidden_dim, device=device))
                    else:
                        gnn_parts.append(torch.zeros(self.hidden_dim, device=device))
                gnn_aligned = torch.stack(gnn_parts)  # [n_alive, hidden_dim]

        # ── Temporal: 팀별 GRU (배치 처리) ──
        team_seqs = torch.stack(all_team_h, dim=1)  # [n_alive, L, hidden_dim]
        zone_expanded = zone_seq.unsqueeze(0).expand(n_alive, -1, -1)

        h_temporal = self.temporal.forward_batch(
            team_seqs, zone_expanded
        )  # [n_alive, hidden_dim]

        # ── GNN gradient highway: gated skip connection ──
        if gnn_aligned is not None:
            gate = self.gnn_gate(torch.cat([h_temporal, gnn_aligned], dim=-1))
            h_final = h_temporal + gate * gnn_aligned
        else:
            h_final = h_temporal

        # ── HazardHead (배치 처리) ──
        zone_phase = meta.get("zone_phase", 0)
        zone_phases = torch.full(
            (n_alive,), zone_phase, dtype=torch.long, device=device
        )

        norm_zone_area = meta.get("norm_zone_area", 1.0)
        ctx_base = torch.tensor([
            norm_zone_area,
            n_alive / 25.0,
            meta.get("elapsed", 0) / 1800.0,
            norm_zone_area * (n_alive / 25.0),
        ], device=device, dtype=torch.float32)
        context = ctx_base.unsqueeze(0).expand(n_alive, -1)  # [n_alive, 4]

        hazard_logits, risk_scores = self.hazard_head.forward_batch(
            h_final, zone_phases, context
        )

        return hazard_logits, risk_scores, alphas

    def compute_snapshot_loss(self, batch, lambda_survival=0.5):
        """
        Multi-task loss: L_ranking + λ · L_survival.

        Returns:
            dict: total, hazard_logits list, risk_scores list, dying_teams list
        """
        from dataset import snapshot_ranking_loss, snapshot_survival_nll

        B = len(batch["player_graphs"])
        device = batch["zone_seqs"][0].device

        total_loss = torch.tensor(0.0, device=device)
        all_hazard_logits = []
        all_risk_scores = []
        all_dying = []
        n_valid = 0

        for b in range(B):
            sample = {
                "player_graphs": batch["player_graphs"][b],
                "team_graph": batch["team_graphs"][b],
                "zone_seq": batch["zone_seqs"][b],
                "alive_teams": batch["alive_teams"][b],
                "meta": batch["metas"][b],
            }

            hazard_logits, risk_scores, _ = self.forward_snapshot(sample)
            dying_teams = batch["dying_teams"][b]

            # Multi-task loss
            l_ranking = snapshot_ranking_loss(hazard_logits, dying_teams)
            l_survival = snapshot_survival_nll(hazard_logits, dying_teams)
            loss = l_ranking + lambda_survival * l_survival

            total_loss = total_loss + loss
            n_valid += 1

            all_hazard_logits.append(hazard_logits.detach())
            all_risk_scores.append(risk_scores.detach())
            all_dying.append(dying_teams)

        if n_valid > 0:
            total_loss = total_loss / n_valid

        return {
            "total": total_loss,
            "hazard_logits": all_hazard_logits,
            "risk_scores": all_risk_scores,
            "dying_teams": all_dying,
        }
