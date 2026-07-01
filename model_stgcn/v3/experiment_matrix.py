"""Canonical V3 experiment matrix.

This is intentionally declarative so shell runners and paper tables can be
generated from the same source.
"""

from __future__ import annotations

import json


def _row(
    *,
    group: str,
    name: str,
    model: str,
    graph_encoder: str,
    edge_mode: str = "full",
    edge_control: str = "observed",
    global_fusion: str = "concat",
    team_decoder: str = "mlp",
    pairwise_rank_weight: str = "0.0",
    feature_preset: str = "full",
    note: str = "",
) -> dict[str, str]:
    return {
        "group": group,
        "name": name,
        "model": model,
        "global_fusion": global_fusion,
        "graph_encoder": graph_encoder,
        "edge_mode": edge_mode,
        "edge_control": edge_control,
        "team_decoder": team_decoder,
        "pairwise_rank_weight": pairwise_rank_weight,
        "feature_preset": feature_preset,
        "note": note,
    }


MAIN_ARCHITECTURE_EXPERIMENTS: list[dict[str, str]] = [
    _row(group="main", name="mr_tcn", model="mr", graph_encoder="none", edge_mode="none", note="MR without graph"),
    _row(group="main", name="mr_gcn", model="mr", graph_encoder="gcn", note="MR with simple graph convolution"),
    _row(group="main", name="mr_sage", model="mr", graph_encoder="sage", note="MR with GraphSAGE-style aggregation"),
    _row(group="main", name="mr_gat", model="mr", graph_encoder="gat", note="MR with graph attention"),
    _row(group="main", name="mr_rgcn", model="mr", graph_encoder="rgcn", note="MR with relation-aware graph convolution"),
    _row(
        group="main",
        name="mr_rgcn_film",
        model="mr",
        graph_encoder="rgcn",
        global_fusion="film",
        note="isolates global context conditioning",
    ),
    _row(
        group="main",
        name="mr_rgcn_attn_rank",
        model="mr",
        graph_encoder="rgcn",
        team_decoder="attention",
        pairwise_rank_weight="0.1",
        note="isolates attention rank decoder",
    ),
    _row(
        group="main",
        name="mr_farnet",
        model="mr",
        graph_encoder="rgcn",
        global_fusion="film",
        team_decoder="attention",
        pairwise_rank_weight="0.1",
        note="final MR-FARNet candidate",
    ),
]


RESOLUTION_ABLATION_EXPERIMENTS: list[dict[str, str]] = [
    _row(group="resolution", name="phase_tcn", model="phase", graph_encoder="none", edge_mode="none"),
    _row(group="resolution", name="phase_rgcn", model="phase", graph_encoder="rgcn"),
    _row(group="resolution", name="mr_tcn", model="mr", graph_encoder="none", edge_mode="none"),
    _row(group="resolution", name="mr_rgcn", model="mr", graph_encoder="rgcn"),
]


DECODER_ABLATION_EXPERIMENTS: list[dict[str, str]] = [
    _row(group="decoder", name="mr_rgcn_hazard_rank", model="mr", graph_encoder="rgcn", team_decoder="mlp", pairwise_rank_weight="0.0"),
    _row(group="decoder", name="mr_rgcn_pairwise_rank", model="mr", graph_encoder="rgcn", team_decoder="mlp", pairwise_rank_weight="0.1"),
    _row(group="decoder", name="mr_rgcn_attn_rank", model="mr", graph_encoder="rgcn", team_decoder="attention", pairwise_rank_weight="0.1"),
]


GRAPH_CONTROL_EXPERIMENTS: list[dict[str, str]] = [
    _row(group="graph_control", name="mr_rgcn_no_graph", model="mr", graph_encoder="none", edge_mode="none"),
    _row(group="graph_control", name="mr_rgcn_proximity", model="mr", graph_encoder="rgcn", edge_mode="proximity"),
    _row(group="graph_control", name="mr_rgcn_combat_only", model="mr", graph_encoder="rgcn", edge_mode="combat_only"),
    _row(group="graph_control", name="mr_rgcn_copressure_only", model="mr", graph_encoder="rgcn", edge_mode="copressure_only"),
    _row(group="graph_control", name="mr_rgcn_full_observed", model="mr", graph_encoder="rgcn", edge_mode="full"),
]


GRAPH_RELATION_EXPERIMENTS: list[dict[str, str]] = [
    _row(
        group="graph_relation",
        name="graph_core_habitat_no_graph",
        model="mr",
        graph_encoder="none",
        edge_mode="none",
        feature_preset="core_habitat",
        note="fixed core+habitat node/global features; no phase graph",
    ),
    _row(
        group="graph_relation",
        name="graph_core_habitat_proximity",
        model="mr",
        graph_encoder="rgcn",
        edge_mode="proximity",
        feature_preset="core_habitat",
        note="phase graph uses proximity relation only",
    ),
    _row(
        group="graph_relation",
        name="graph_core_habitat_combat",
        model="mr",
        graph_encoder="rgcn",
        edge_mode="combat_only",
        feature_preset="core_habitat",
        note="phase graph uses combat relation only",
    ),
    _row(
        group="graph_relation",
        name="graph_core_habitat_copressure",
        model="mr",
        graph_encoder="rgcn",
        edge_mode="copressure_only",
        feature_preset="core_habitat",
        note="phase graph uses copressure/habitat relation only",
    ),
    _row(
        group="graph_relation",
        name="graph_core_habitat_proximity_combat",
        model="mr",
        graph_encoder="rgcn",
        edge_mode="proximity_combat",
        feature_preset="core_habitat",
        note="phase graph uses proximity plus combat",
    ),
    _row(
        group="graph_relation",
        name="graph_core_habitat_proximity_copressure",
        model="mr",
        graph_encoder="rgcn",
        edge_mode="proximity_copressure",
        feature_preset="core_habitat",
        note="phase graph uses proximity plus copressure",
    ),
    _row(
        group="graph_relation",
        name="graph_core_habitat_full_observed",
        model="mr",
        graph_encoder="rgcn",
        edge_mode="full",
        feature_preset="core_habitat",
        note="phase graph uses all observed relation channels",
    ),
    _row(
        group="graph_relation",
        name="graph_core_habitat_full_node_shuffle",
        model="mr",
        graph_encoder="rgcn",
        edge_mode="full",
        edge_control="node_shuffle",
        feature_preset="core_habitat",
        note="trained/evaluated with phase-local node-relabelled graph control",
    ),
    _row(
        group="graph_relation",
        name="graph_core_habitat_full_edge_shuffle",
        model="mr",
        graph_encoder="rgcn",
        edge_mode="full",
        edge_control="edge_shuffle",
        feature_preset="core_habitat",
        note="trained/evaluated with phase-local density-preserving edge shuffle",
    ),
]


FEATURE_ABLATION_EXPERIMENTS: list[dict[str, str]] = [
    _row(
        group="feature",
        name="feat_core",
        model="mr",
        graph_encoder="none",
        edge_mode="none",
        feature_preset="core",
        note="conventional/core phase features with MR snapshots, no graph",
    ),
    _row(
        group="feature",
        name="feat_stdb",
        model="mr",
        graph_encoder="none",
        edge_mode="none",
        feature_preset="stdb",
        note="STDB spatial/exposure phase features with minimal temporal context",
    ),
    _row(
        group="feature",
        name="feat_network",
        model="mr",
        graph_encoder="none",
        edge_mode="none",
        feature_preset="network",
        note="network role phase features with minimal temporal context",
    ),
    _row(
        group="feature",
        name="feat_habitat",
        model="mr",
        graph_encoder="none",
        edge_mode="none",
        feature_preset="habitat",
        note="habitat-inspired pressure phase features",
    ),
    _row(
        group="feature",
        name="feat_core_stdb",
        model="mr",
        graph_encoder="none",
        edge_mode="none",
        feature_preset="core_stdb",
        note="core plus STDB spatial/exposure",
    ),
    _row(
        group="feature",
        name="feat_core_network",
        model="mr",
        graph_encoder="none",
        edge_mode="none",
        feature_preset="core_network",
        note="core plus network role",
    ),
    _row(
        group="feature",
        name="feat_core_habitat",
        model="mr",
        graph_encoder="none",
        edge_mode="none",
        feature_preset="core_habitat",
        note="core plus habitat pressure",
    ),
    _row(
        group="feature",
        name="feat_base66",
        model="mr",
        graph_encoder="none",
        edge_mode="none",
        feature_preset="base66",
        note="legacy/current 66 phase features without V2 network/habitat",
    ),
    _row(
        group="feature",
        name="feat_full",
        model="mr",
        graph_encoder="none",
        edge_mode="none",
        feature_preset="full",
        note="all phase feature families with MR snapshots, no graph",
    ),
]


EXPERIMENT_GROUPS: dict[str, list[dict[str, str]]] = {
    "main": MAIN_ARCHITECTURE_EXPERIMENTS,
    "resolution": RESOLUTION_ABLATION_EXPERIMENTS,
    "decoder": DECODER_ABLATION_EXPERIMENTS,
    "graph_control": GRAPH_CONTROL_EXPERIMENTS,
    "graph_relation": GRAPH_RELATION_EXPERIMENTS,
    "feature": FEATURE_ABLATION_EXPERIMENTS,
}


def all_experiments(*, groups: list[str] | None = None) -> list[dict[str, str]]:
    selected = groups or ["main", "resolution", "decoder", "graph_control"]
    rows: list[dict[str, str]] = []
    for group in selected:
        rows.extend(EXPERIMENT_GROUPS[group])
    return rows


if __name__ == "__main__":
    print(json.dumps(all_experiments(), indent=2))
