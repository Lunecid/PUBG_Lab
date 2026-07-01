"""Build a flat, non-destructive research cohort manifest.

The goal is to separate current DB contents by map / perspective / team size
without deleting source data.  Each match receives a status and exclude_reason
so experiments can use exact cohorts while raw/normalized data remains intact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_stgcn.splits import split_ids, write_split_manifest
from stdb.conn import connect
from stdb.queries import PHASE_N_MAX


DEFAULT_OUT_DIR = Path("model_stgcn/experiment_datasets/research_cohort_flat")
DEFAULT_DOC = Path("docs/RESEARCH_COHORT_FLAT.md")
PAPER_COHORT_ID = "paper_graph_n128_v1"
STANDARD_TEAM_SIZES = {"solo", "duo", "squad"}
OUT_OF_SCOPE_MAPS = {"Tutorial_Main", "Range_Main"}
MAP_ALIASES = {
    "Baltic_Main": "Erangel",
    "Desert_Main": "Miramar/Desert",
    "DihorOtok_Main": "Vikendi",
    "Savage_Main": "Sanhok",
    "Tiger_Main": "Taego",
    "Neon_Main": "Rondo",
    "Kiki_Main": "Deston",
    "Chimera_Main": "Paramo",
    "Summerland_Main": "Karakin",
    "Tutorial_Main": "Tutorial",
    "Range_Main": "Training Range",
}


FIELDS = [
    "match_id",
    "map_name",
    "game_mode",
    "perspective",
    "team_size",
    "created_at",
    "duration",
    "raw_fetched",
    "base_done",
    "base_with_team_positions",
    "phase_feature_done",
    "target_label_done",
    "edge_cache_done",
    "num_teams",
    "num_players",
    "graph_eligible_n128",
    "cohort_status",
    "exclude_reason",
]


SUMMARY_FIELDS = [
    "map_name",
    "perspective",
    "team_size",
    "game_mode",
    "raw_fetched",
    "base_done",
    "phase_feature_done",
    "target_label_done",
    "edge_cache_done",
    "graph_eligible_n128",
    "excluded",
]

PIVOT_FIELDS = [
    "map_label",
    "map_name",
    "fpp_solo",
    "fpp_duo",
    "fpp_squad",
    "fpp_total",
    "tpp_solo",
    "tpp_duo",
    "tpp_squad",
    "tpp_total",
    "graph_eligible_total",
    "raw_fetched",
    "excluded",
]

MEMBERSHIP_FIELDS = [
    "match_id",
    "split",
    "map_label",
    "map_name",
    "perspective",
    "team_size",
    "game_mode",
    "created_at",
    "duration",
    "num_teams",
    "num_players",
]

DOMAIN_SPLIT_FIELDS = [
    "map_label",
    "map_name",
    "perspective",
    "team_size",
    "game_mode",
    "train",
    "val",
    "test",
    "total",
]


def perspective(game_mode: str) -> str:
    return "FPP" if game_mode.endswith("-fpp") else "TPP"


def team_size(game_mode: str) -> str:
    base = game_mode.removesuffix("-fpp")
    if base in STANDARD_TEAM_SIZES:
        return base
    return base or "unknown"


def classify(row: dict[str, Any]) -> tuple[str, str]:
    if not row["raw_fetched"]:
        return "not_raw_fetched", "telemetry_not_fetched"
    if row["map_name"] in OUT_OF_SCOPE_MAPS:
        return "raw_only_not_ready", "out_of_scope_map"
    if row["team_size"] not in STANDARD_TEAM_SIZES:
        return "raw_only_not_ready", "non_standard_team_size"
    if not row["base_done"]:
        return "raw_only_not_ready", "not_converted_to_stdb"
    if not row["base_with_team_positions"]:
        return "base_incomplete", "missing_team_positions"
    if not row["phase_feature_done"]:
        return "base_incomplete", "missing_phase_features"
    if not row["target_label_done"]:
        return "feature_incomplete", "missing_target_labels"
    if not row["edge_cache_done"]:
        return "feature_only", "missing_edge_cache"
    if row["num_teams"] is not None and int(row["num_teams"]) < 4:
        return "feature_only", "too_few_teams_for_graph"
    if row["num_teams"] is not None and int(row["num_teams"]) > PHASE_N_MAX:
        return "feature_only", f"too_many_teams_gt_{PHASE_N_MAX}"
    return "graph_eligible", ""


def fetch_rows() -> list[dict[str, Any]]:
    sql = """
    WITH raw AS (
        SELECT
            m.match_id,
            m.map_name,
            m.game_mode,
            m.created_at,
            m.duration,
            COALESCE(c.telemetry_fetched, false) AS raw_fetched
          FROM pubg.matches m
          LEFT JOIN pubg.collection_log c ON c.match_id = m.match_id
    ),
    base AS (
        SELECT
            m.match_id,
            true AS base_done,
            m.num_teams,
            m.num_players,
            EXISTS (
                SELECT 1
                  FROM stdb.moving_objects mo
                 WHERE mo.match_id = m.match_id
                   AND mo.agent_kind = 'team'
            ) AS base_with_team_positions
          FROM stdb.matches m
    ),
    phase AS (
        SELECT
            match_id,
            true AS phase_feature_done,
            bool_or(target_defined) AS target_label_done
          FROM stdb.team_phase_labels
         GROUP BY match_id
    ),
    edge AS (
        SELECT match_id, true AS edge_cache_done
          FROM stdb.team_phase_edge_cache
         GROUP BY match_id
    )
    SELECT
        r.match_id,
        r.map_name,
        r.game_mode,
        r.created_at,
        r.duration,
        r.raw_fetched,
        COALESCE(b.base_done, false) AS base_done,
        COALESCE(b.base_with_team_positions, false) AS base_with_team_positions,
        COALESCE(p.phase_feature_done, false) AS phase_feature_done,
        COALESCE(p.target_label_done, false) AS target_label_done,
        COALESCE(e.edge_cache_done, false) AS edge_cache_done,
        b.num_teams,
        b.num_players
      FROM raw r
 LEFT JOIN base b ON b.match_id = r.match_id
 LEFT JOIN phase p ON p.match_id = r.match_id
 LEFT JOIN edge e ON e.match_id = r.match_id
     WHERE r.raw_fetched
     ORDER BY r.map_name, r.game_mode, r.created_at, r.match_id
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            rows: list[dict[str, Any]] = []
            for values in cur.fetchall():
                row = dict(zip(cols, values))
                row["perspective"] = perspective(row["game_mode"])
                row["team_size"] = team_size(row["game_mode"])
                status, reason = classify(row)
                row["cohort_status"] = status
                row["exclude_reason"] = reason
                row["graph_eligible_n128"] = status == "graph_eligible"
                rows.append(row)
            return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_id_file(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


def ids_sha1(ids: list[str]) -> str:
    return hashlib.sha1("\n".join(ids).encode("utf-8")).hexdigest()


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["map_name"], row["perspective"], row["team_size"], row["game_mode"])
        if key not in grouped:
            grouped[key] = {
                "map_name": key[0],
                "perspective": key[1],
                "team_size": key[2],
                "game_mode": key[3],
                "raw_fetched": 0,
                "base_done": 0,
                "phase_feature_done": 0,
                "target_label_done": 0,
                "edge_cache_done": 0,
                "graph_eligible_n128": 0,
                "excluded": 0,
            }
        out = grouped[key]
        for field in (
            "raw_fetched",
            "base_done",
            "phase_feature_done",
            "target_label_done",
            "edge_cache_done",
            "graph_eligible_n128",
        ):
            out[field] += int(bool(row[field]))
        out["excluded"] += int(row["cohort_status"] != "graph_eligible")
    rows_out = list(grouped.values())
    rows_out.sort(key=lambda r: (-r["graph_eligible_n128"], -r["raw_fetched"], r["map_name"], r["game_mode"]))
    return rows_out


def summarize_pivot(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in summary:
        map_name = row["map_name"]
        if map_name not in grouped:
            grouped[map_name] = {
                "map_label": MAP_ALIASES.get(map_name, map_name),
                "map_name": map_name,
                "fpp_solo": 0,
                "fpp_duo": 0,
                "fpp_squad": 0,
                "fpp_total": 0,
                "tpp_solo": 0,
                "tpp_duo": 0,
                "tpp_squad": 0,
                "tpp_total": 0,
                "graph_eligible_total": 0,
                "raw_fetched": 0,
                "excluded": 0,
            }
        out = grouped[map_name]
        eligible = int(row["graph_eligible_n128"])
        key = f"{row['perspective'].lower()}_{row['team_size']}"
        if key in out:
            out[key] += eligible
        if row["perspective"] == "FPP":
            out["fpp_total"] += eligible
        elif row["perspective"] == "TPP":
            out["tpp_total"] += eligible
        out["graph_eligible_total"] += eligible
        out["raw_fetched"] += int(row["raw_fetched"])
        out["excluded"] += int(row["excluded"])
    rows_out = list(grouped.values())
    rows_out.sort(key=lambda r: (-r["graph_eligible_total"], -r["raw_fetched"], r["map_label"]))
    return rows_out


def write_paper_cohort(
    out_dir: Path,
    rows: list[dict[str, Any]],
    *,
    seed: int,
    val_frac: float,
    test_frac: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    graph_rows = [row for row in rows if row["graph_eligible_n128"]]
    ids = [row["match_id"] for row in graph_rows]
    train_ids, val_ids, test_ids = split_ids(ids, seed=seed, val_frac=val_frac, test_frac=test_frac)

    split_by_id = {match_id: "train" for match_id in train_ids}
    split_by_id.update({match_id: "val" for match_id in val_ids})
    split_by_id.update({match_id: "test" for match_id in test_ids})

    membership: list[dict[str, Any]] = []
    domain_counts: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in graph_rows:
        split = split_by_id[row["match_id"]]
        map_name = row["map_name"]
        membership.append(
            {
                "match_id": row["match_id"],
                "split": split,
                "map_label": MAP_ALIASES.get(map_name, map_name),
                "map_name": map_name,
                "perspective": row["perspective"],
                "team_size": row["team_size"],
                "game_mode": row["game_mode"],
                "created_at": row["created_at"],
                "duration": row["duration"],
                "num_teams": row["num_teams"],
                "num_players": row["num_players"],
            }
        )
        key = (map_name, row["perspective"], row["team_size"], row["game_mode"])
        if key not in domain_counts:
            domain_counts[key] = {
                "map_label": MAP_ALIASES.get(map_name, map_name),
                "map_name": map_name,
                "perspective": row["perspective"],
                "team_size": row["team_size"],
                "game_mode": row["game_mode"],
                "train": 0,
                "val": 0,
                "test": 0,
                "total": 0,
            }
        domain_counts[key][split] += 1
        domain_counts[key]["total"] += 1

    domain_summary = list(domain_counts.values())
    domain_summary.sort(key=lambda r: (-r["total"], r["map_name"], r["perspective"], r["team_size"]))

    cohort_id_file = out_dir / f"{PAPER_COHORT_ID}_match_ids.txt"
    split_file = out_dir / f"{PAPER_COHORT_ID}_split_seed{seed}.json"
    membership_file = out_dir / f"{PAPER_COHORT_ID}_membership.csv"
    domain_split_file = out_dir / f"{PAPER_COHORT_ID}_domain_split_summary.csv"
    manifest_file = out_dir / f"{PAPER_COHORT_ID}_manifest.json"

    write_id_file(cohort_id_file, ids)
    write_split_manifest(
        split_file,
        ids=ids,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        seed=seed,
        val_frac=val_frac,
        test_frac=test_frac,
        metadata={
            "cohort_id": PAPER_COHORT_ID,
            "phase_n_max": PHASE_N_MAX,
            "source": "scripts/build_research_cohort_flat.py",
            "eligibility_rule": "raw_fetched AND stdb base/features/target labels/edge cache complete AND 4 <= num_teams <= PHASE_N_MAX",
            "paper_policy": "All main paper experiments must use this cohort or deterministic subsets joined through this membership file.",
        },
    )
    write_csv(membership_file, membership, MEMBERSHIP_FIELDS)
    write_csv(domain_split_file, domain_summary, DOMAIN_SPLIT_FIELDS)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cohort_id": PAPER_COHORT_ID,
        "phase_n_max": PHASE_N_MAX,
        "n_matches": len(ids),
        "ids_sha1": ids_sha1(ids),
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "split_counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "eligibility_rule": {
            "raw_fetched": True,
            "base_done": True,
            "base_with_team_positions": True,
            "phase_feature_done": True,
            "target_label_done": True,
            "edge_cache_done": True,
            "min_num_teams": 4,
            "max_num_teams": PHASE_N_MAX,
            "excluded_maps": sorted(OUT_OF_SCOPE_MAPS),
            "team_sizes": sorted(STANDARD_TEAM_SIZES),
        },
        "paths": {
            "match_ids": str(cohort_id_file),
            "split": str(split_file),
            "membership": str(membership_file),
            "domain_split_summary": str(domain_split_file),
        },
    }
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest, domain_summary


def markdown_table(rows: list[dict[str, Any]], fields: list[str], *, limit: int | None = None) -> str:
    selected = rows if limit is None else rows[:limit]
    text_fields = {
        "cohort_id",
        "cohort_status",
        "exclude_reason",
        "game_mode",
        "ids_sha1",
        "map_label",
        "map_name",
        "perspective",
        "team_size",
    }
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" if field in text_fields else "---:" for field in fields) + " |",
    ]
    for row in selected:
        vals = []
        for field in fields:
            value = row.get(field, "")
            vals.append(f"{value:,}" if isinstance(value, int) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    pivot: list[dict[str, Any]],
    paper_manifest: dict[str, Any],
    paper_domain_summary: list[dict[str, Any]],
) -> None:
    status_counts = Counter(row["cohort_status"] for row in rows)
    reason_counts = Counter(row["exclude_reason"] or "included" for row in rows)
    total = {
        "raw_fetched": len(rows),
        "graph_eligible_n128": status_counts["graph_eligible"],
        "excluded": len(rows) - status_counts["graph_eligible"],
    }
    status_rows = [
        {"cohort_status": status, "count": count}
        for status, count in sorted(status_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    reason_rows = [
        {"exclude_reason": reason, "count": count}
        for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    split_counts = paper_manifest["split_counts"]
    paper_rows = [
        {
            "cohort_id": paper_manifest["cohort_id"],
            "n_matches": paper_manifest["n_matches"],
            "train": split_counts["train"],
            "val": split_counts["val"],
            "test": split_counts["test"],
            "ids_sha1": paper_manifest["ids_sha1"],
        }
    ]
    lines = [
        "# Research Cohort Flat Manifest",
        "",
        "이 문서는 현재 DB의 raw-fetched match를 삭제 없이 연구용 cohort로 평탄화한 결과이다.",
        "",
        "## Totals",
        "",
        markdown_table([total], ["raw_fetched", "graph_eligible_n128", "excluded"]),
        "",
        "## Status Counts",
        "",
        markdown_table(status_rows, ["cohort_status", "count"]),
        "",
        "## Exclusion Reasons",
        "",
        markdown_table(reason_rows, ["exclude_reason", "count"]),
        "",
        "## Map Pivot Summary",
        "",
        "연구에 실제로 투입 가능한 `graph_eligible_n128` 기준의 맵/인칭/팀 크기 피벗이다.",
        "",
        markdown_table(pivot, PIVOT_FIELDS),
        "",
        "## Paper Cohort Contract",
        "",
        "논문용 본실험은 아래 master cohort를 공통 모집단으로 고정한다. 모델 variant, ablation, tabular baseline은 동일한 match split을 재사용하고, 맵/인칭/팀 크기 분석은 이 cohort membership의 부분집합으로만 수행한다.",
        "",
        markdown_table(paper_rows, ["cohort_id", "n_matches", "train", "val", "test", "ids_sha1"]),
        "",
        "- `feature_eligible`가 더 많은 분석도 논문 본표에서는 이 common-denominator cohort로 제한한다.",
        "- 도메인별 실험은 독립 split을 새로 뽑지 않고 master split과 도메인 membership의 교집합을 사용한다.",
        "- 탐색 실험에서 다른 cohort를 쓰면 보조 분석으로 분리하고, 본 주장과 직접 비교하지 않는다.",
        "",
        "## Paper Cohort Domain Splits",
        "",
        markdown_table(paper_domain_summary, DOMAIN_SPLIT_FIELDS, limit=80),
        "",
        "## Domain Summary",
        "",
        markdown_table(summary, SUMMARY_FIELDS, limit=80),
        "",
        "## Policy",
        "",
        "- `pubg.*` raw/normalized layer는 삭제하지 않는다.",
        "- 숫자를 맞출 때는 `graph_eligible_n128 = true`인 match만 연구 cohort에 포함한다.",
        "- 제외 대상은 `exclude_reason`으로 격리한다.",
        "- 삭제가 필요하다면 `stdb.*` 같은 재생성 가능한 파생 layer에서만 수행한다.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    args = parser.parse_args()

    rows = fetch_rows()
    summary = summarize(rows)
    pivot = summarize_pivot(summary)
    paper_manifest, paper_domain_summary = write_paper_cohort(
        args.out_dir,
        rows,
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )
    write_csv(args.out_dir / "match_cohort_flat.csv", rows, FIELDS)
    write_csv(args.out_dir / "domain_cohort_summary.csv", summary, SUMMARY_FIELDS)
    write_csv(args.out_dir / "map_perspective_team_pivot.csv", pivot, PIVOT_FIELDS)
    write_report(args.doc, rows, summary, pivot, paper_manifest, paper_domain_summary)
    print(f"wrote {args.out_dir / 'match_cohort_flat.csv'}")
    print(f"wrote {args.out_dir / 'domain_cohort_summary.csv'}")
    print(f"wrote {args.out_dir / 'map_perspective_team_pivot.csv'}")
    print(f"wrote {args.out_dir / f'{PAPER_COHORT_ID}_match_ids.txt'}")
    print(f"wrote {args.out_dir / f'{PAPER_COHORT_ID}_split_seed{args.seed}.json'}")
    print(f"wrote {args.out_dir / f'{PAPER_COHORT_ID}_membership.csv'}")
    print(f"wrote {args.out_dir / f'{PAPER_COHORT_ID}_domain_split_summary.csv'}")
    print(f"wrote {args.out_dir / f'{PAPER_COHORT_ID}_manifest.json'}")
    print(f"wrote {args.doc}")


if __name__ == "__main__":
    main()
