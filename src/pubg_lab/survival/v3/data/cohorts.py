"""Named PUBG cohort filters for V3 experiment runs."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CohortSpec:
    slug: str
    label: str
    map_label: str
    map_name: str
    perspective: str
    team_size: str
    game_mode: str
    tier: str = "main"
    aliases: tuple[str, ...] = ()


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


REQUIRED_COHORTS: tuple[CohortSpec, ...] = (
    CohortSpec(
        "erangel_tpp_squad",
        "Erangel TPP squad",
        "Erangel",
        "Baltic_Main",
        "TPP",
        "squad",
        "squad",
        aliases=("C_MAIN_01", "Baltic_Main__TPP__squad"),
    ),
    CohortSpec(
        "taego_tpp_squad",
        "Taego TPP squad",
        "Taego",
        "Tiger_Main",
        "TPP",
        "squad",
        "squad",
        aliases=("C_MAIN_02", "Tiger_Main__TPP__squad"),
    ),
    CohortSpec(
        "miramar_desert_tpp_squad",
        "Miramar / Desert TPP squad",
        "Miramar / Desert",
        "Desert_Main",
        "TPP",
        "squad",
        "squad",
        aliases=("C_MAIN_03", "Desert_Main__TPP__squad", "Miramar TPP squad", "Desert TPP squad"),
    ),
    CohortSpec(
        "rondo_tpp_squad",
        "Rondo TPP squad",
        "Rondo",
        "Neon_Main",
        "TPP",
        "squad",
        "squad",
        aliases=("C_MAIN_04", "Neon_Main__TPP__squad"),
    ),
    CohortSpec(
        "sanhok_tpp_squad",
        "Sanhok TPP squad",
        "Sanhok",
        "Savage_Main",
        "TPP",
        "squad",
        "squad",
        aliases=("C_MAIN_05", "Savage_Main__TPP__squad"),
    ),
    CohortSpec(
        "erangel_tpp_solo",
        "Erangel TPP solo",
        "Erangel",
        "Baltic_Main",
        "TPP",
        "solo",
        "solo",
        aliases=("C_MAIN_06", "Baltic_Main__TPP__solo"),
    ),
    CohortSpec(
        "erangel_tpp_duo",
        "Erangel TPP duo",
        "Erangel",
        "Baltic_Main",
        "TPP",
        "duo",
        "duo",
        aliases=("C_MAIN_07", "Baltic_Main__TPP__duo"),
    ),
    CohortSpec(
        "taego_tpp_duo",
        "Taego TPP duo",
        "Taego",
        "Tiger_Main",
        "TPP",
        "duo",
        "duo",
        aliases=("C_MAIN_08", "Tiger_Main__TPP__duo"),
    ),
    CohortSpec(
        "vikendi_tpp_squad",
        "Vikendi TPP squad",
        "Vikendi",
        "DihorOtok_Main",
        "TPP",
        "squad",
        "squad",
        aliases=("C_MAIN_09", "DihorOtok_Main__TPP__squad"),
    ),
    CohortSpec(
        "taego_fpp_squad",
        "Taego FPP squad",
        "Taego",
        "Tiger_Main",
        "FPP",
        "squad",
        "squad-fpp",
        aliases=("C_MAIN_10", "Tiger_Main__FPP__squad"),
    ),
    CohortSpec(
        "erangel_fpp_squad",
        "Erangel FPP squad",
        "Erangel",
        "Baltic_Main",
        "FPP",
        "squad",
        "squad-fpp",
        aliases=("C_MAIN_11", "Baltic_Main__FPP__squad"),
    ),
    CohortSpec(
        "paramo_tpp_squad",
        "Paramo TPP squad",
        "Paramo",
        "Chimera_Main",
        "TPP",
        "squad",
        "squad",
        aliases=("C_MAIN_12", "Chimera_Main__TPP__squad"),
    ),
    CohortSpec(
        "karakin_tpp_squad",
        "Karakin TPP squad",
        "Karakin",
        "Summerland_Main",
        "TPP",
        "squad",
        "squad",
        aliases=("C_MAIN_13", "Summerland_Main__TPP__squad"),
    ),
)


def _build_index() -> dict[str, CohortSpec]:
    out: dict[str, CohortSpec] = {}
    for spec in REQUIRED_COHORTS:
        keys = (spec.slug, spec.label, *spec.aliases)
        for key in keys:
            norm = _norm(key)
            if norm in out and out[norm] != spec:
                raise RuntimeError(f"duplicate cohort key: {key}")
            out[norm] = spec
    return out


COHORTS_BY_KEY = _build_index()


def cohort_choices() -> list[str]:
    return [spec.label for spec in REQUIRED_COHORTS]


def resolve_cohort(value: str) -> CohortSpec:
    key = _norm(value)
    try:
        return COHORTS_BY_KEY[key]
    except KeyError as exc:
        choices = "; ".join(cohort_choices())
        raise ValueError(f"unknown cohort {value!r}; expected one of: {choices}") from exc


def resolve_map_mode(
    *,
    cohort: str | None,
    map_name: str | None,
    mode: str | None,
) -> tuple[str | None, str | None, CohortSpec | None]:
    if cohort:
        if map_name or mode:
            raise ValueError("--cohort cannot be combined with --map or --mode")
        spec = resolve_cohort(cohort)
        return spec.map_name, spec.game_mode, spec
    return map_name, mode, None
