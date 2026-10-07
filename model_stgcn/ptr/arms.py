"""Experiment arms: design doc section 4 and section 4.1 (v2.1).

Main arms (``MAIN_ARMS``, all seeds / cohorts as in section 4)::

    arm            phase  team  relation   zone  features
    M00            off    off   off        off   all
    M10            on     off   off        off   all
    M01            off    on    off        off   all
    M11            on     on    off        off   all
    M11R           on     on    on         off   all
    M11R-shuf      on     on    shuffled   off   all; relation matrices relabelled by a fixed
                                                  within-(match, phase) permutation of alive teams
    M11RZ          on     on    on         on    all  (PTR-Former)
    M11R-noHab     on     on    on         off   zone + habitat feature groups removed *
    M11RZ-noHab    on     on    on         on    zone + habitat feature groups removed *

    * alive-count variables inside those groups are kept (``NOHAB_KEEP``: avg_alive_teams_norm,
      avg_alive_players_norm, hab_alive_teams); the removed column names are written to metrics.json.

Exploratory arms (``EXPLORATORY_ARMS``; section 4.1, run on the two large cohorts with seed 42):
``M11R-noProx`` / ``M11R-noCombat`` / ``M11R-noCopress`` = M11R with one relation type removed
(``noCombat`` removes both directed channels and also switches escalation off, so the prox channel
cannot leak the combat edge set).

Stage-2 arms (``STAGE2_ARMS``; player hierarchy, design doc 3 / 4.1, they need the player dump
``model_stgcn/dumps/bigcomp2027_player_v1``): ``M11RZ-P`` = M11RZ plus the player layer (relation-biased attention over
all alive players with the teammate, enemy-proximity and combat biases, attention pooling to team tokens), and
``M11RZ-P-tm`` = the same without the teammate bias only.  ``STUB_ARMS`` is empty now: no arm is a stub any more.

Union arm (``UNION_ARMS``; design doc 4.6, post hoc and exploratory): ``M11R-union`` = M11R plus the ``kf_*`` features that
reconstruct KRAFTON's input representation (``model_stgcn.ptr.extra``), appended to the team token features after ``X``, the team
centre and ``team_ai_frac``.  It needs ``--extra-features DIR`` (the kf tables); its HGB counterpart is ``hgb.HGB-union``.

Rank-loss control arms (``NORANK_ARMS``; post hoc and exploratory, supplementary experiment E1): ``M01-norank`` and ``M11R-norank`` are M01 and
M11R trained with the hazard BCE ALONE (``ArmSpec.rank_weight = 0``: no pairwise rank loss), everything else identical (seed, recipe,
architecture, rows).  The KRAFTON arms (K2 / K3a / K3) train without a rank loss, so the architecture step K3 -> M01 also adds one;
``M01-norank`` removes that confound.  Their rank head is untrained, so ``train.py`` writes ``rank_score = -hazard logit`` for them (the
convention of the tabular and KRAFTON arms) and the rank metrics are the hazard head's ranking.
"""

from __future__ import annotations

from dataclasses import dataclass

from model_stgcn.ptr.data import ALIVE_COUNT_FEATURES
from model_stgcn.ptr.model import PTRConfig
from model_stgcn.ptr.relations import RELATION_TYPES
from model_stgcn.ptr.player import PLAYER_RELATION_TYPES

NOHAB_GROUPS: tuple[str, ...] = ("zone", "habitat")
# noHab removes the zone and habitat groups but keeps the alive-count variables that sit inside them
NOHAB_KEEP: tuple[str, ...] = ALIVE_COUNT_FEATURES


@dataclass(frozen=True)
class ArmSpec:
    name: str
    phase_axis: bool
    team_axis: bool
    relation: bool
    zone: bool
    shuffle: bool = False
    drop_groups: tuple[str, ...] = ()
    keep_names: tuple[str, ...] = ()  # feature names kept although they sit inside a dropped group
    relation_types: tuple[str, ...] = RELATION_TYPES
    exploratory: bool = False
    stage: int = 1
    implemented: bool = True
    player: bool = False  # stage 2: player tokens + relation-biased player attention + pooling to team tokens
    player_relation_types: tuple[str, ...] = ()  # bias channels of the player attention (subset of PLAYER_RELATION_TYPES)
    extra: bool = False  # union arm: the kf_* features of --extra-features are appended to the team token features
    rank_weight: float | None = None  # loss override: None = the --rank-weight of the command line (default 0.1); 0.0 = BCE only (E1 controls)


def _arm(name: str, phase: bool, team: bool, relation: bool, zone: bool, **kw) -> ArmSpec:
    return ArmSpec(name, phase, team, relation, zone, **kw)


_MAIN = (
    _arm("M00", False, False, False, False),
    _arm("M10", True, False, False, False),
    _arm("M01", False, True, False, False),
    _arm("M11", True, True, False, False),
    _arm("M11R", True, True, True, False),
    _arm("M11R-shuf", True, True, True, False, shuffle=True),
    _arm("M11RZ", True, True, True, True),
    _arm("M11R-noHab", True, True, True, False, drop_groups=NOHAB_GROUPS, keep_names=NOHAB_KEEP),
    _arm("M11RZ-noHab", True, True, True, True, drop_groups=NOHAB_GROUPS, keep_names=NOHAB_KEEP),
)
_EXPLORATORY = (
    _arm("M11R-noProx", True, True, True, False, exploratory=True,
         relation_types=("combat_out", "combat_in", "copress")),
    _arm("M11R-noCombat", True, True, True, False, exploratory=True,
         relation_types=("prox", "copress")),
    _arm("M11R-noCopress", True, True, True, False, exploratory=True,
         relation_types=("prox", "combat_out", "combat_in")),
)  # fmt: skip
_STAGE2 = (
    # stage 2 (player layer, decided 10/9 18:00): M11RZ + players; -tm removes the teammate bias channel only
    _arm("M11RZ-P", True, True, True, True, stage=2, player=True, player_relation_types=PLAYER_RELATION_TYPES),
    _arm("M11RZ-P-tm", True, True, True, True, stage=2, player=True,
         player_relation_types=tuple(t for t in PLAYER_RELATION_TYPES if t != "teammate")),
)  # fmt: skip

_UNION = (
    # design doc 4.6: our representation united with KRAFTON's reconstructed features; the architecture is M11R's
    _arm("M11R-union", True, True, True, False, exploratory=True, extra=True),
)  # fmt: skip

_NORANK = (
    # supplementary experiment E1 (post hoc): the PTR loss without the pairwise rank term, same architecture / recipe otherwise
    _arm("M01-norank", False, True, False, False, exploratory=True, rank_weight=0.0),
    _arm("M11R-norank", True, True, True, False, exploratory=True, rank_weight=0.0),
)  # fmt: skip

ARMS: dict[str, ArmSpec] = {a.name: a for a in (*_MAIN, *_EXPLORATORY, *_STAGE2, *_UNION, *_NORANK)}
MAIN_ARMS: tuple[str, ...] = tuple(a.name for a in _MAIN)
EXPLORATORY_ARMS: tuple[str, ...] = tuple(a.name for a in _EXPLORATORY)
STAGE2_ARMS: tuple[str, ...] = tuple(a.name for a in _STAGE2)
UNION_ARMS: tuple[str, ...] = tuple(a.name for a in _UNION)
NORANK_ARMS: tuple[str, ...] = tuple(a.name for a in _NORANK)
STUB_ARMS: tuple[str, ...] = ()  # no arm is registered-but-unimplemented any more


def get_arm(name: str, *, implemented_only: bool = False) -> ArmSpec:
    arm = ARMS.get(name)
    if arm is None:
        raise KeyError(f"unknown arm '{name}'; known: {sorted(ARMS)}")
    if implemented_only and not arm.implemented:
        raise NotImplementedError(f"arm {name} (stage 2, player-level hierarchy) is not implemented")
    return arm


def model_config(
    arm: ArmSpec,
    in_features: int,
    *,
    n_phases: int = 8,
    zone_dim: int = 8,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.1,
    ffn_mult: int = 4,
    zone_hidden: int = 16,
    player_features: int = 0,
    player_layers: int = 1,
) -> PTRConfig:
    """Model configuration of an arm (defaults are the shared spec: d=128, H=4, L=2, dropout 0.1).

    ``player_features``: columns of the player feature tensor (``Xp``), required by the stage-2 arms."""
    if not arm.implemented:
        raise NotImplementedError(f"arm {arm.name} is not implemented")
    if arm.player and player_features < 1:
        raise ValueError(f"arm {arm.name} needs the player dump: player_features must be >= 1")
    return PTRConfig(
        in_features=in_features,
        n_phases=n_phases,
        zone_dim=zone_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ffn_mult=ffn_mult,
        dropout=dropout,
        phase_axis=arm.phase_axis,
        team_axis=arm.team_axis,
        relation=arm.relation,
        relation_types=arm.relation_types,
        zone=arm.zone,
        zone_hidden=zone_hidden,
        player=arm.player,
        player_features=player_features if arm.player else 0,
        player_relation_types=arm.player_relation_types if arm.player else (),
        player_layers=player_layers if arm.player else 0,
    )
