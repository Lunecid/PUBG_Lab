"""Feature role specification for the final factorized experiment.

The previous phase tensor stores all phase features in a single node vector.
Some columns are truly team/node attributes, while others are global phase
context that was broadcast to every team. V3 makes this contract explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from stdb.queries import PHASE_NODE_FEATURE_NAMES, PHASE_NODE_FEATURE_NAMES_V2


# Names are the normalized tensor feature names, not raw DB column names.
BASE_GLOBAL_FEATURES: tuple[str, ...] = (
    "phase_idx_norm",
    "phase_duration_10min",
    "avg_alive_teams_norm",
    "avg_alive_players_norm",
    "safe_radius_start_norm",
    "safe_radius_end_norm",
    "safe_radius_delta_norm",
    "safe_center_shift_norm",
    "poison_radius_start_norm",
    "poison_radius_end_norm",
    "zone_gap_start_norm",
    "zone_gap_end_norm",
)

HABITAT_GLOBAL_FEATURES: tuple[str, ...] = (
    "hab_safe_area_start_m2",
    "hab_safe_area_end_m2",
    "hab_safe_area_delta_m2",
    "hab_loss_ratio",
    "hab_shrink_rate_m2ps",
    "hab_alive_teams",
    "hab_safe_area_per_alive_team",
    "hab_crowding_pressure",
    "hab_center_shift_m",
    "hab_center_shift_per_radius",
    "hab_zone_gap_start_m",
    "hab_zone_gap_end_m",
    "hab_zone_gap_pressure",
)

GLOBAL_FEATURES: tuple[str, ...] = BASE_GLOBAL_FEATURES + HABITAT_GLOBAL_FEATURES
GLOBAL_FEATURE_SET = set(GLOBAL_FEATURES)

MINIMAL_TEMPORAL_CONTEXT_FEATURES: tuple[str, ...] = (
    "phase_idx_norm",
    "phase_duration_10min",
)

# Base phase feature blocks.  These index ranges follow stdb.queries
# PHASE_NODE_FEATURE_SPECS: base global context, team state, STDB
# trajectory/exposure, combat, and loot/readiness.
BASE66_FEATURES: tuple[str, ...] = tuple(PHASE_NODE_FEATURE_NAMES)
CONVENTIONAL_NODE_FEATURES: tuple[str, ...] = tuple(
    list(PHASE_NODE_FEATURE_NAMES[12:25])
    + list(PHASE_NODE_FEATURE_NAMES[42:65])
)
STDB_SPATIAL_NODE_FEATURES: tuple[str, ...] = tuple(PHASE_NODE_FEATURE_NAMES[25:42])
NETWORK_NODE_FEATURES: tuple[str, ...] = tuple(
    name
    for name in PHASE_NODE_FEATURE_NAMES_V2
    if name.startswith("net_") or name.startswith("cum_")
)
HABITAT_FEATURES: tuple[str, ...] = tuple(
    name for name in PHASE_NODE_FEATURE_NAMES_V2 if name.startswith("hab_")
)

FEATURE_PRESETS: tuple[str, ...] = (
    "full",
    "base66",
    "core",
    "stdb",
    "network",
    "habitat",
    "core_stdb",
    "core_network",
    "core_habitat",
    "core_stdb_network",
    "core_stdb_habitat",
)

# Exact feature names that must never appear in model inputs.  Keep this exact
# rather than substring-based: terms like "target" are valid in causal features
# such as hab_distance_to_target_safe_m, while final_rank_norm is a label.
FORBIDDEN_INPUT_FEATURES: tuple[str, ...] = (
    "target_defined",
    "hazard_next",
    "survives_next_phase",
    "eliminated_next_phase",
    "final_rank",
    "final_rank_norm",
    "final_top3",
    "final_top5",
    "final_top10",
    "final_winner",
    "last_phase_alive",
    "won",
    "rank",
    "victim_rank",
    "match_end_rank",
    "match_end_result",
)
FORBIDDEN_INPUT_FEATURE_SET = set(FORBIDDEN_INPUT_FEATURES)


@dataclass(frozen=True)
class FeatureIndexSplit:
    feature_names: tuple[str, ...]
    node_indices: tuple[int, ...]
    global_indices: tuple[int, ...]
    node_feature_names: tuple[str, ...]
    global_feature_names: tuple[str, ...]

    @property
    def node_dim(self) -> int:
        return len(self.node_indices)

    @property
    def global_dim(self) -> int:
        return len(self.global_indices)


def expected_phase_feature_names(*, include_v2: bool = True) -> list[str]:
    names = list(PHASE_NODE_FEATURE_NAMES)
    if include_v2:
        names += list(PHASE_NODE_FEATURE_NAMES_V2)
    return names


def feature_names_for_preset(preset: str, available_feature_names: Iterable[str]) -> tuple[str, ...]:
    """Return available normalized feature names for a representation preset."""
    if preset not in FEATURE_PRESETS:
        raise ValueError(f"unknown feature_preset={preset!r}; choices={FEATURE_PRESETS}")

    available = tuple(available_feature_names)
    available_set = set(available)
    if preset == "full":
        selected = set(available)
    elif preset == "base66":
        selected = set(BASE66_FEATURES)
    elif preset == "core":
        selected = set(BASE_GLOBAL_FEATURES) | set(CONVENTIONAL_NODE_FEATURES)
    elif preset == "stdb":
        selected = set(MINIMAL_TEMPORAL_CONTEXT_FEATURES) | set(STDB_SPATIAL_NODE_FEATURES)
    elif preset == "network":
        selected = set(MINIMAL_TEMPORAL_CONTEXT_FEATURES) | set(NETWORK_NODE_FEATURES)
    elif preset == "habitat":
        selected = set(MINIMAL_TEMPORAL_CONTEXT_FEATURES) | set(HABITAT_FEATURES)
    elif preset == "core_stdb":
        selected = set(BASE_GLOBAL_FEATURES) | set(CONVENTIONAL_NODE_FEATURES) | set(STDB_SPATIAL_NODE_FEATURES)
    elif preset == "core_network":
        selected = set(BASE_GLOBAL_FEATURES) | set(CONVENTIONAL_NODE_FEATURES) | set(NETWORK_NODE_FEATURES)
    elif preset == "core_habitat":
        selected = set(BASE_GLOBAL_FEATURES) | set(CONVENTIONAL_NODE_FEATURES) | set(HABITAT_FEATURES)
    elif preset == "core_stdb_network":
        selected = (
            set(BASE_GLOBAL_FEATURES)
            | set(CONVENTIONAL_NODE_FEATURES)
            | set(STDB_SPATIAL_NODE_FEATURES)
            | set(NETWORK_NODE_FEATURES)
        )
    elif preset == "core_stdb_habitat":
        selected = (
            set(BASE_GLOBAL_FEATURES)
            | set(CONVENTIONAL_NODE_FEATURES)
            | set(STDB_SPATIAL_NODE_FEATURES)
            | set(HABITAT_FEATURES)
        )
    else:  # pragma: no cover - kept for exhaustiveness if presets change.
        selected = set(available)

    kept = tuple(name for name in available if name in available_set and name in selected)
    assert_no_forbidden_input_features(kept)
    if not kept:
        raise ValueError(f"feature_preset={preset!r} selected no available features")
    return kept


def split_feature_names(feature_names: Iterable[str]) -> FeatureIndexSplit:
    names = tuple(feature_names)
    assert_no_forbidden_input_features(names)
    node_idx: list[int] = []
    global_idx: list[int] = []
    node_names: list[str] = []
    global_names: list[str] = []

    for i, name in enumerate(names):
        if name in GLOBAL_FEATURE_SET:
            global_idx.append(i)
            global_names.append(name)
        else:
            node_idx.append(i)
            node_names.append(name)

    return FeatureIndexSplit(
        feature_names=names,
        node_indices=tuple(node_idx),
        global_indices=tuple(global_idx),
        node_feature_names=tuple(node_names),
        global_feature_names=tuple(global_names),
    )


def role_for_feature(name: str) -> str:
    assert_no_forbidden_input_features([name])
    if name in GLOBAL_FEATURE_SET:
        return "global"
    return "node"


def role_summary(feature_names: Iterable[str]) -> dict[str, int]:
    split = split_feature_names(feature_names)
    return {
        "total": len(split.feature_names),
        "node": split.node_dim,
        "global": split.global_dim,
    }


def forbidden_input_hits(feature_names: Iterable[str]) -> list[str]:
    names = list(feature_names)
    return [name for name in names if name in FORBIDDEN_INPUT_FEATURE_SET]


def assert_no_forbidden_input_features(feature_names: Iterable[str]) -> None:
    hits = forbidden_input_hits(feature_names)
    if hits:
        raise ValueError(f"forbidden leakage input features present: {hits}")
