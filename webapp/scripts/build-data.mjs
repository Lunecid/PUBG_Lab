#!/usr/bin/env node
/**
 * PUBG webapp 데이터 빌드 스크립트
 *
 * 입력:
 *   /Users/lunecid/project/PUBG/network_analysis/output/analysis_datasets/Baltic_Main_squad_n1000/
 *   webapp/public/data/matches/{matchId}.json (기존 shard 재활용)
 *
 * 출력:
 *   webapp/public/data/index.json
 *   webapp/public/data/network_summary.json
 *   webapp/public/data/strategy_summary.json
 *   webapp/public/data/phase_summary.json
 *   webapp/public/data/hypotheses.json
 *   webapp/public/data/match_index.json
 *   webapp/public/data/glossary.json
 *   webapp/public/data/methods.json
 *   webapp/public/data/matches/{matchId}.json (선정된 30개만, strategy_type 머지된 nodes)
 *
 * 30개 매치는 modularity 분위 + 매치 길이 분위로 stratified random sampling.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const DATASET =
  "/Users/lunecid/project/PUBG/network_analysis/output/analysis_datasets/Baltic_Main_squad_n1000";
const OUT = path.join(ROOT, "public/data");
const SHARDS_OUT = path.join(OUT, "matches");
const SAMPLE_SIZE = 30;
const RNG_SEED = 42;

// ---------- 유틸 ----------

function parseCsv(content) {
  const lines = content.split(/\r?\n/).filter((l) => l.length > 0);
  if (lines.length === 0) return [];
  const headers = parseCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const vals = parseCsvLine(line);
    const obj = {};
    headers.forEach((h, i) => {
      obj[h] = vals[i] ?? "";
    });
    return obj;
  });
}

function parseCsvLine(line) {
  const out = [];
  let cur = "";
  let inQ = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQ && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else {
        inQ = !inQ;
      }
    } else if (ch === "," && !inQ) {
      out.push(cur);
      cur = "";
    } else {
      cur += ch;
    }
  }
  out.push(cur);
  return out;
}

const num = (v) => {
  if (v === undefined || v === null || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};
const intv = (v) => {
  const n = num(v);
  return n === null ? null : Math.round(n);
};
const round = (v, n = 3) => {
  const x = num(v);
  if (x === null) return null;
  const k = Math.pow(10, n);
  return Math.round(x * k) / k;
};

function readCsv(p) {
  return parseCsv(fs.readFileSync(p, "utf-8"));
}
function readJson(p) {
  return JSON.parse(fs.readFileSync(p, "utf-8"));
}
function writeJson(p, data) {
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, JSON.stringify(data));
  console.log(`  wrote ${path.relative(ROOT, p)}`);
}

// 시드 고정 PRNG
function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------- 1. index.json ----------

function buildIndex(meta, props) {
  const n = props.length;
  const mean = (k) =>
    round(
      props.reduce((s, r) => s + (num(r[k]) ?? 0), 0) / n,
      3,
    );
  const giantTeamRatio = round(
    props.reduce(
      (s, r) => s + (num(r.full_giant_size) ?? 0) / (num(r.full_n_nodes) || 1),
      0,
    ) / n,
    3,
  );
  return {
    generated_at: meta.generated_at,
    map_name: meta.map_name,
    game_mode: meta.game_mode,
    n_matches: meta.n_matches,
    n_teams: meta.n_teams,
    n_team_edges: meta.n_team_edges,
    n_team_centroids: meta.n_team_centroids,
    early_window_sec: meta.early_window_sec,
    centroid_sample_sec: meta.centroid_sample_sec,
    landing_offset_s: meta.landing_offset_s,
    movement_window_s: meta.movement_window_s,
    /** Baltic_Main(=Erangel) 미니맵. 8192x8192 PNG가 8x8km 맵에 정확히 매핑됨.
     *  PUBG 텔레메트리 좌표 (top-left 원점, 미터 단위) → 이미지 픽셀 좌표 그대로.
     *  높은 해상도부터 시도, 실패 시 fallback. */
    map_urls: ["/maps/baltic_main.jpg", "/maps/baltic_main_2k.jpg"],
    map_arena_m: 8160,
    map_attribution: {
      source: "Web-Engine/pubg-minimap-replay (GitHub)",
      url: "https://github.com/Web-Engine/pubg-minimap-replay",
      note: "PUBG community asset — fair use for academic analysis",
    },
    network_avg: {
      n_edges: mean("full_n_edges"),
      n_nodes: mean("full_n_nodes"),
      density: mean("full_density"),
      avg_degree: mean("full_avg_degree"),
      avg_clustering: mean("full_avg_clustering"),
      diameter: mean("full_diameter"),
      avg_path_length: mean("full_avg_path_length"),
      modularity_q: mean("full_modularity_q") ?? 0.592, // metric 일부 누락 시 보고서 평균값
      giant_size: mean("full_giant_size"),
      giant_team_ratio: giantTeamRatio,
    },
  };
}

// ---------- 2. network_summary.json ----------

function buildNetworkSummary(props, posSummary, basicSummary, advSummary) {
  // §3.1 매치 단위 네트워크 구조 (보고서 표 그대로)
  const matchStats = [
    { label: "팀 간 교전 엣지 수", value: 42.832, format: "decimal" },
    { label: "네트워크 밀도", value: 0.13, format: "decimal" },
    { label: "평균 degree", value: 3.245, format: "decimal" },
    { label: "평균 clustering", value: 0.401, format: "decimal" },
    { label: "giant component 비율", value: 0.973, format: "decimal" },
    { label: "평균 최단경로", value: 2.766, format: "decimal" },
    { label: "Louvain modularity Q", value: 0.592, format: "decimal" },
    { label: "평균 커뮤니티 수", value: 6.215, format: "decimal" },
    { label: "최대 커뮤니티 비율", value: 0.289, format: "decimal" },
  ];

  // §3.3 위치별 평균 순위 (basic_graph_analysis/position_rank_summary.csv)
  // posSummary는 advanced_graph_analysis/position_rank_summary.csv (isolated 포함)
  const positionTable = posSummary.map((r) => ({
    position: r.position_label,
    n: intv(r.n),
    mean_rank: round(r.mean_rank, 3),
    median_rank: round(r.median_rank, 1),
    mean_degree: round(r.mean_degree, 3),
    mean_betweenness: round(r.mean_betweenness, 4),
    mean_weighted_degree: round(r.mean_weighted_degree, 1),
    mean_survival_s: round(r.mean_survival_s, 1),
  }));

  // §3.4 battle bridge / betweenness / across-deg / PageRank
  const bridgeTable = advSummary
    .filter((r) =>
      [
        "battle_bridge_score",
        "c_betweenness",
        "across_deg",
        "c_pagerank",
        "community_embeddedness",
      ].includes(r.metric),
    )
    .map((r) => ({
      metric: r.metric,
      label: r.label,
      low_n: intv(r.low_n),
      high_n: intv(r.high_n),
      low_mean_rank: round(r.low_mean_rank, 3),
      high_mean_rank: round(r.high_mean_rank, 3),
      delta: round(r.delta_high_minus_low, 3),
      mw_p: num(r.mw_p),
      cliffs_delta: round(r.cliffs_delta_low_vs_high, 3),
    }));

  return {
    match_stats: matchStats,
    position_table: positionTable,
    bridge_table: bridgeTable,
  };
}

// ---------- 3. strategy_summary.json ----------

function buildStrategySummary(typeSummary, edgeMixing) {
  // §6.1: 유형별 성과
  const typePerformance = typeSummary.map((r) => ({
    strategy_type: r.strategy_type,
    n: intv(r.n),
    share: round(r.share, 4),
    fight_score_mean: round(r.fight_score_mean, 3),
    active_score_mean: round(r.active_score_mean, 3),
    final_rank_mean: round(r.final_rank_mean, 3),
    final_rank_median: round(r.final_rank_median, 1),
    ranked_rp_proxy_mean: round(r.ranked_rp_proxy_mean, 3),
    team_kills_mean: round(r.team_kills_mean, 3),
    team_assists_mean: round(r.team_assists_mean, 3),
    top10_rate: round(r.top10_rate, 3),
    survival_seconds_mean: round(r.survival_seconds_mean, 1),
  }));

  // §6.2: 네트워크 프로필
  const typeProfiles = typeSummary.map((r) => ({
    strategy_type: r.strategy_type,
    degree_per_min_mean: round(r.degree_per_min_mean, 3),
    c_pagerank_mean: round(r.c_pagerank_mean, 4),
    c_betweenness_mean: round(r.c_betweenness_mean, 4),
    battle_bridge_score_mean: round(r.battle_bridge_score_mean, 3),
    community_embeddedness_mean: round(r.community_embeddedness_mean, 3),
    movement_per_min_mean: round(r.movement_per_min_mean, 1),
    combat_st_participation_rate_mean: round(
      r.combat_st_participation_rate_mean,
      3,
    ),
  }));

  // §6.4: 유형 간 mixing — top 5 share
  const sortedMix = edgeMixing
    .map((r) => ({
      type_pair: r.type_pair,
      n_edges: intv(r.n_edges),
      total_events: intv(r.total_events),
      total_damage: round(r.total_damage, 0),
      mean_weight: round(r.mean_weight, 1),
      edge_share: round(r.edge_share, 4),
      event_share: round(r.event_share, 4),
    }))
    .sort((a, b) => (b.edge_share || 0) - (a.edge_share || 0));
  const typeMixing = sortedMix.slice(0, 10);

  return { type_performance: typePerformance, type_profiles: typeProfiles, type_mixing: typeMixing };
}

// ---------- 4. phase_summary.json ----------

function buildPhaseSummary(phaseSummary, transitions) {
  const phaseTypePerformance = phaseSummary.map((r) => ({
    phase: intv(r.phase),
    strategy_type: r.phase_strategy_type,
    n: intv(r.n),
    final_rank_mean: round(r.final_rank_mean, 3),
    ranked_rp_proxy_mean: round(r.ranked_rp_proxy_mean, 3),
    team_kills_mean: round(r.team_kills_mean, 3),
    team_assists_mean: round(r.team_assists_mean, 3),
    survived_phase_mean: round(r.survived_phase_mean, 3),
    fight_score_mean: round(r.phase_fight_score_mean, 3),
    active_score_mean: round(r.phase_active_score_mean, 3),
    movement_per_min_mean: round(r.phase_movement_per_min_mean, 1),
    combat_st_participation_rate_mean: round(
      r.phase_combat_st_participation_rate_mean,
      3,
    ),
    top10_rate: round(r.top10_rate, 3),
  }));

  // 전환: 각 from_phase → to_phase 의 상위 5개
  const grouped = {};
  for (const r of transitions) {
    const key = `${r.from_phase}_${r.to_phase}`;
    if (!grouped[key]) grouped[key] = [];
    grouped[key].push({
      from_phase: intv(r.from_phase),
      to_phase: intv(r.to_phase),
      from_type: r.from_type,
      to_type: r.to_type,
      transition: r.transition,
      n: intv(r.n),
    });
  }
  const topTransitions = Object.values(grouped).flatMap((arr) =>
    arr.sort((a, b) => b.n - a.n).slice(0, 5),
  );

  // 전환 매트릭스 (각 from_phase 기준 4×4)
  const matrices = {};
  for (const r of transitions) {
    const key = `${r.from_phase}_${r.to_phase}`;
    if (!matrices[key]) matrices[key] = {};
    matrices[key][`${r.from_type}->${r.to_type}`] = intv(r.n);
  }

  return {
    phase_type_performance: phaseTypePerformance,
    top_transitions: topTransitions,
    transition_matrices: matrices,
  };
}

// ---------- 5. hypotheses.json ----------

function buildHypotheses(stTests, h3Completed, advSummary) {
  // H1 — st_hypothesis_tests.csv 의 H1_outer_farming 행 + 보고서 §4.1 표
  const h1Tests = stTests.filter((r) => r.hypothesis === "H1_outer_farming");
  const h2Tests = stTests.filter((r) => r.hypothesis === "H2_engagement_intensity");
  const h3Tests = stTests.filter((r) => r.hypothesis === "H3_movement_passive");

  // 보고서 §4.1: hotzone vs middle vs outer_farm 의 결과 표
  const h1Groups = [
    {
      group: "hotzone",
      label: "핫존",
      mean_rank: 16.684,
      ranked_rp_proxy: -0.216,
      team_kills_mean: 4.037,
      team_assists_mean: 0.784,
      survival_seconds_mean: 676.161,
      combat_st_participation_rate: 0.301,
      movement_per_min: 115.641,
    },
    {
      group: "middle",
      label: "중간",
      mean_rank: 13.587,
      ranked_rp_proxy: 0.018,
      team_kills_mean: 3.181,
      team_assists_mean: 0.714,
      survival_seconds_mean: 846.547,
      combat_st_participation_rate: 0.223,
      movement_per_min: 163.222,
    },
    {
      group: "outer_farm",
      label: "외곽 파밍",
      mean_rank: 12.189,
      ranked_rp_proxy: 0.183,
      team_kills_mean: 3.054,
      team_assists_mean: 0.715,
      survival_seconds_mean: 920.933,
      combat_st_participation_rate: 0.206,
      movement_per_min: 193.66,
    },
  ];

  // 보고서 §4.2: low / middle / high combat rate
  const h2Groups = [
    {
      group: "low_combat",
      label: "낮은 교전",
      mean_rank: 10.717,
      ranked_rp_proxy: -0.083,
      team_kills_mean: 1.514,
      team_assists_mean: 0.306,
      survival_seconds_mean: 1000.206,
      combat_st_participation_rate: 0.09,
      movement_per_min: 200.208,
    },
    {
      group: "middle",
      label: "중간",
      mean_rank: 12.764,
      ranked_rp_proxy: 0.42,
      team_kills_mean: 4.009,
      team_assists_mean: 0.986,
      survival_seconds_mean: 893.705,
      combat_st_participation_rate: 0.216,
      movement_per_min: 168.367,
    },
    {
      group: "high_combat",
      label: "높은 교전",
      mean_rank: 19.598,
      ranked_rp_proxy: -0.655,
      team_kills_mean: 4.118,
      team_assists_mean: 0.714,
      survival_seconds_mean: 514.313,
      combat_st_participation_rate: 0.429,
      movement_per_min: 100.09,
    },
  ];

  // 보고서 §4.3: H3 600초 고정창
  const h3Groups = [
    {
      group: "fixed_sedentary",
      label: "고정창 저이동 (하위 25%)",
      n: 4103,
      mean_rank: 9.229,
      ranked_rp_proxy: 0.999,
      team_kills_mean: 4.662,
      team_assists_mean: 1.039,
      survival_seconds_mean: 1099.204,
      combat_st_participation_rate: 0.187,
      fixed_window_movement_per_min: 84.49,
    },
    {
      group: "fixed_middle",
      label: "고정창 중간",
      n: 6680,
      mean_rank: 8.432,
      ranked_rp_proxy: 1.112,
      team_kills_mean: 4.724,
      team_assists_mean: 1.083,
      survival_seconds_mean: 1145.376,
      combat_st_participation_rate: 0.166,
      fixed_window_movement_per_min: 189.533,
    },
    {
      group: "fixed_mobile",
      label: "고정창 고이동 (상위 25%)",
      n: 4103,
      mean_rank: 7.427,
      ranked_rp_proxy: 1.478,
      team_kills_mean: 5.662,
      team_assists_mean: 1.277,
      survival_seconds_mean: 1189.025,
      combat_st_participation_rate: 0.16,
      fixed_window_movement_per_min: 383.194,
    },
  ];

  // §4.4: 페이즈별 H3 (completed_phase_only) — phase × {sedentary, mobile}
  const phaseH3 = h3Completed
    .filter((r) => r.cohort === "completed_phase_only")
    .map((r) => ({
      phase: intv(r.phase),
      group: r.phase_movement_group, // phase_sedentary | phase_middle | phase_mobile
      n: intv(r.team_phase_n),
      final_rank_mean: round(r.final_rank_mean, 3),
      ranked_rp_proxy_mean: round(r.ranked_rp_proxy_mean, 3),
      survived_phase_mean: round(r.survived_phase_mean, 3),
      top10_rate: round(r.top10_rate, 3),
    }));

  return {
    h1: {
      title: "H1: 핫존보다 외곽 파밍이 평균 순위가 높다",
      verdict: "outer_farm_better",
      verdict_label: "외곽 파밍 우세",
      conclusion:
        "외곽 파밍 팀이 평균 순위와 RP proxy 모두에서 핫존보다 우세하다. 핫존 팀은 평균 킬은 더 높지만 조기 사망 위험이 보상보다 크다.",
      groups: h1Groups,
      tests: h1Tests.map((t) => ({
        target: t.target,
        target_label: t.target_label,
        low_label: t.low_label,
        high_label: t.high_label,
        low_mean_target: round(t.low_mean_target, 3),
        high_mean_target: round(t.high_mean_target, 3),
        delta: round(t.delta_high_minus_low, 3),
        mw_p: num(t.mw_p),
        cliffs_delta: round(t.cliffs_delta_low_vs_high, 3),
        statistically_supported: t.statistically_supported === "True",
      })),
    },
    h2: {
      title: "H2: 교전 참여율이 높을수록 평균 순위가 높다",
      verdict: "rejected",
      verdict_label: "기각",
      conclusion:
        "교전 참여율이 높기만 한 팀은 오히려 평균 순위와 RP proxy가 낮다. 중간 교전 그룹이 RP proxy가 가장 높고, high combat은 짧은 생존시간으로 손해를 본다.",
      groups: h2Groups,
      tests: h2Tests.map((t) => ({
        target: t.target,
        target_label: t.target_label,
        low_label: t.low_label,
        high_label: t.high_label,
        low_mean_target: round(t.low_mean_target, 3),
        high_mean_target: round(t.high_mean_target, 3),
        delta: round(t.delta_high_minus_low, 3),
        mw_p: num(t.mw_p),
        cliffs_delta: round(t.cliffs_delta_low_vs_high, 3),
        statistically_supported: t.statistically_supported === "True",
      })),
    },
    h3: {
      title: "H3: 존버형 팀이 활발히 이동하는 팀보다 평균 순위가 높다",
      verdict: "rejected",
      verdict_label: "기각 (mobile 우세)",
      conclusion:
        "착륙 후 동일 600초 고정창에서도 존버 우세는 지지되지 않는다. fixed_mobile 이 순위·RP proxy·평균 킬 모두에서 우세. phase별 재분석에서도 고이동 팀이 RP proxy에서 일관되게 우세.",
      groups: h3Groups,
      tests: h3Tests.map((t) => ({
        target: t.target,
        target_label: t.target_label,
        low_label: t.low_label,
        high_label: t.high_label,
        low_mean_target: round(t.low_mean_target, 3),
        high_mean_target: round(t.high_mean_target, 3),
        delta: round(t.delta_high_minus_low, 3),
        mw_p: num(t.mw_p),
        cliffs_delta: round(t.cliffs_delta_low_vs_high, 3),
        statistically_supported: t.statistically_supported === "True",
      })),
      phase_breakdown: phaseH3,
    },
  };
}

// ---------- 6. match_index.json + match 선정 ----------

function pickMatches(propsRows, n = SAMPLE_SIZE) {
  // modularity 분위 (저/중/고 = q33, q67) 와 매치 길이 분위로 stratified 샘플
  const rows = propsRows
    .map((r) => ({
      match_id: r.match_id,
      duration_s: num(r.duration_s),
      n_teams: intv(r.full_n_nodes),
      modularity_q: num(r.full_modularity_q),
      n_edges: intv(r.full_n_edges),
      density: num(r.full_density),
      avg_clustering: num(r.full_avg_clustering),
      avg_path_length: num(r.full_avg_path_length),
      diameter: intv(r.full_diameter),
    }))
    .filter(
      (r) =>
        r.modularity_q !== null &&
        r.duration_s !== null &&
        r.match_id &&
        fs.existsSync(path.join(SHARDS_OUT, `${r.match_id}.json`)),
    );

  // 분위 계산
  const sortedQ = [...rows].sort((a, b) => a.modularity_q - b.modularity_q);
  const q33Q = sortedQ[Math.floor(sortedQ.length / 3)].modularity_q;
  const q67Q = sortedQ[Math.floor((sortedQ.length * 2) / 3)].modularity_q;

  const sortedD = [...rows].sort((a, b) => a.duration_s - b.duration_s);
  const q33D = sortedD[Math.floor(sortedD.length / 3)].duration_s;
  const q67D = sortedD[Math.floor((sortedD.length * 2) / 3)].duration_s;

  const bin = (r) => {
    const qb = r.modularity_q < q33Q ? "L" : r.modularity_q < q67Q ? "M" : "H";
    const db = r.duration_s < q33D ? "S" : r.duration_s < q67D ? "M" : "L";
    return `${qb}-${db}`;
  };

  // 9개 bin 각각에서 ~3-4개씩 균등 샘플
  const buckets = {};
  for (const r of rows) {
    const k = bin(r);
    if (!buckets[k]) buckets[k] = [];
    buckets[k].push(r);
  }

  const rng = mulberry32(RNG_SEED);
  const picked = [];
  const perBucket = Math.ceil(n / Object.keys(buckets).length);
  for (const k of Object.keys(buckets).sort()) {
    const arr = buckets[k];
    // 셔플
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    picked.push(...arr.slice(0, perBucket));
  }
  // n 개로 자름
  return picked.slice(0, n).sort((a, b) => a.match_id.localeCompare(b.match_id));
}

// ---------- 7. shard 머지 ----------

function buildShards(picked, teamFeatures) {
  // teamFeatures를 (match_id, team_idx) 키로 인덱싱
  const tfByKey = new Map();
  for (const r of teamFeatures) {
    const k = `${r.match_id}|${r.team_idx}`;
    tfByKey.set(k, r);
  }

  // 모든 기존 shard 디렉터리에서 선정되지 않은 것들 제거
  const keepSet = new Set(picked.map((p) => `${p.match_id}.json`));
  const allShardFiles = fs.readdirSync(SHARDS_OUT);
  let removed = 0;
  for (const f of allShardFiles) {
    if (!f.endsWith(".json")) continue;
    if (!keepSet.has(f)) {
      fs.unlinkSync(path.join(SHARDS_OUT, f));
      removed++;
    }
  }
  console.log(`  removed ${removed} unused shards`);

  // 선정된 shard 머지
  for (const m of picked) {
    const shardPath = path.join(SHARDS_OUT, `${m.match_id}.json`);
    const shard = readJson(shardPath);
    for (const node of shard.nodes) {
      const key = `${m.match_id}|${node.id}`;
      const tf = tfByKey.get(key);
      if (!tf) continue;
      node.strategy_type = tf.strategy_type || null;
      node.fight_score = round(tf.network_fight_score, 3);
      node.active_score = round(tf.network_active_score, 3);
      node.fight_high = tf.fight_high === "True";
      node.active_high = tf.active_high === "True";
      node.team_kills = intv(tf.team_kills);
      node.team_assists = intv(tf.team_assists);
      node.ranked_rp_proxy = round(tf.ranked_rp_proxy, 3);
      node.movement_per_min = round(tf.movement_per_min, 1);
      node.combat_st_participation_rate = round(
        tf.combat_st_participation_rate,
        3,
      );
      node.battle_bridge_score = round(tf.battle_bridge_score, 3);
      node.distance_to_hotzone_m = round(tf.distance_to_hotzone_m, 1);
      node.farm_position_group = tf.farm_position_group || null;
    }
    writeJson(shardPath, shard);
  }
  console.log(`  augmented ${picked.length} shards with strategy_type fields`);
}

// ---------- 8. match_index.json ----------

function buildMatchIndex(picked, shardsDir) {
  return picked.map((m) => {
    const shardPath = path.join(shardsDir, `${m.match_id}.json`);
    if (!fs.existsSync(shardPath)) {
      return {
        match_id: m.match_id,
        duration_s: m.duration_s,
        n_teams: m.n_teams,
        modularity_q: round(m.modularity_q, 3),
        density: round(m.density, 3),
      };
    }
    const shard = readJson(shardPath);
    const typeCounts = { Hunter: 0, Sniper: 0, Traveler: 0, Camper: 0 };
    for (const n of shard.nodes) {
      if (n.strategy_type && typeCounts[n.strategy_type] !== undefined)
        typeCounts[n.strategy_type]++;
    }
    return {
      match_id: m.match_id,
      duration_s: m.duration_s,
      n_teams: m.n_teams,
      n_edges: m.n_edges,
      modularity_q: round(m.modularity_q, 3),
      density: round(m.density, 3),
      avg_clustering: round(m.avg_clustering, 3),
      avg_path_length: round(m.avg_path_length, 3),
      diameter: m.diameter,
      type_counts: typeCounts,
    };
  });
}

// ---------- 9. glossary.json (보고서 §2.1) ----------

function buildGlossary() {
  return [
    {
      term: "Node",
      general: "그래프를 구성하는 개별 행위자",
      pubg: "한 squad/team을 하나의 노드로 둔다. 동적 분석에서는 같은 팀도 시간 구간별로 노드 속성이 달라진다.",
    },
    {
      term: "Edge",
      general: "두 노드 사이의 관계",
      pubg: "두 팀 사이에 플레이어 damage event가 1회 이상 있으면 edge를 만든다.",
    },
    {
      term: "Weight",
      general: "관계의 강도",
      pubg: "팀 간 누적 damage를 기본 weight로 사용한다. 보조적으로 event count도 사용한다.",
    },
    {
      term: "ST 구조",
      general: "공간(Space)과 시간(Time)이 함께 있는 데이터 구조",
      pubg: "플레이어/팀 위치가 10초 간격으로 기록되고, 교전 이벤트가 시간에 따라 누적된다.",
    },
    {
      term: "Phase",
      general: "PUBG 경기 진행 단계",
      pubg: "phase change 이벤트로 나눈 시간 구간. 같은 팀도 phase마다 다른 노드 속성과 유형을 가진다.",
    },
    {
      term: "Degree",
      general: "직접 연결된 노드 수",
      pubg: "한 팀이 직접 교전한 상대 팀 수.",
    },
    {
      term: "Weighted degree",
      general: "연결 weight의 합",
      pubg: "한 팀이 팀 간 교전에서 누적한 damage 관계의 총량.",
    },
    {
      term: "Betweenness centrality",
      general: "다른 노드 쌍 사이 최단경로를 얼마나 자주 매개하는가",
      pubg: "여러 교전권 사이를 잇는 전투 중개 팀인지 나타낸다.",
    },
    {
      term: "PageRank",
      general: "중요한 노드와 연결된 정도",
      pubg: "강하거나 중심적인 교전 상대와 연결된 정도.",
    },
    {
      term: "Louvain community",
      general: "그래프 내부에서 연결이 조밀한 하위 집단",
      pubg: "한 매치 안에서 형성되는 국소 교전권 또는 전투 클러스터.",
    },
    {
      term: "Battle bridge",
      general: "커뮤니티 사이를 잇는 중개 위치",
      pubg: "서로 다른 교전권을 연결하거나 여러 전투 흐름에 걸친 팀.",
    },
    {
      term: "Hotzone",
      general: "많은 팀이 밀집 착륙한 지역",
      pubg: "맵 중심이 아니라, 매치별 착륙 밀도가 높은 위치로 정의.",
    },
    {
      term: "Outer farm",
      general: "핫존에서 먼 초반 파밍 위치",
      pubg: "많은 팀이 내린 고밀도 착륙지에서 멀리 떨어진 파밍 선택.",
    },
    {
      term: "RP proxy",
      general: "경쟁전 점수 근사값",
      pubg: "팀 순위, 킬, 어시스트를 표준화해 합산한 성과 지표.",
    },
  ];
}

// ---------- 10. methods.json (보고서 §2.6) ----------

function buildMethods() {
  return {
    data_sources: [
      {
        source: ".pt match graph cache",
        features: [
          "match_id",
          "team_id",
          "account_id",
          "team_rank",
          "death_times",
          "10초별 player position",
        ],
        role: "팀 노드 구성, 팀 중심 좌표, 생존시간 계산",
      },
      {
        source: "PostgreSQL telem_damage",
        features: ["attacker_id", "victim_id", "event_time", "damage", "damage_causer"],
        role: "팀-팀 교전 edge 생성의 핵심 입력",
      },
      {
        source: "PostgreSQL telem_phase_changes",
        features: ["event_time", "phase", "is_game"],
        role: "phase별 동적 그래프의 시작/종료 시점 정의",
      },
      {
        source: "PostgreSQL participants",
        features: ["kills", "assists", "dbnos", "damage_dealt"],
        role: "팀 단위 킬/어시스트/RP proxy 계산",
      },
      {
        source: "PostgreSQL rosters",
        features: ["team_id", "rank", "roster_id"],
        role: "팀 최종 순위와 participant-team 매핑",
      },
    ],
    graph_definition: {
      V: "해당 매치에 참가한 팀 집합",
      E: "팀 A와 팀 B 사이에 damage event가 1회 이상 관측된 관계",
      weight: "팀 A와 팀 B 사이의 누적 damage",
      events: "팀 A와 팀 B 사이의 damage event 횟수",
      direction: "무방향",
      excluded: ["같은 팀 내부 damage event", "attacker/victim 팀 매핑 불가능 event", "bluezone 등 환경 피해"],
    },
    graph_windows: [
      { window: "full", definition: "매치 전체 damage event", purpose: "전체 네트워크 구조, 4유형 분석" },
      { window: "early_300s", definition: "매치 시작 후 300초 damage event", purpose: "초반 교전 편향 분석" },
      { window: "phase_k", definition: "특정 phase 시간 구간 안의 damage event", purpose: "phase별 동적 분석" },
    ],
    feature_groups: [
      {
        group: "식별/기본",
        features: ["match_id", "map_name", "game_mode", "team_idx", "team_id", "n_members"],
        usage: "어느 매치의 어느 팀인지 정의",
      },
      {
        group: "성과/결과",
        features: [
          "final_rank",
          "top10",
          "team_kills",
          "team_assists",
          "ranked_placement_percentile",
          "ranked_rp_proxy",
          "survival_seconds",
        ],
        usage: "결과 비교, RP proxy, 유형별 성과표",
      },
      {
        group: "edge 구성",
        features: ["source_team_idx", "target_team_idx", "weight", "damage", "events", "window", "phase"],
        usage: "팀 전투 그래프 생성",
      },
      {
        group: "기본 교전량",
        features: ["combat_events", "dmg_dealt", "dmg_taken", "combat_per_min", "dmg_per_min"],
        usage: "기본/고급 네트워크 분석",
      },
      {
        group: "early 교전",
        features: ["early_degree", "early_weighted_deg", "early_combat_events", "early_dmg_dealt", "early_dmg_taken"],
        usage: "early 300초 보조 분석",
      },
      {
        group: "중심성",
        features: [
          "degree",
          "weighted_deg",
          "c_degree",
          "c_closeness",
          "c_betweenness",
          "c_eigenvector",
          "c_pagerank",
          "local_clustering",
        ],
        usage: "기본/고급 네트워크 분석, 4유형 FightScore",
      },
      {
        group: "커뮤니티",
        features: [
          "community_id",
          "community_size",
          "community_density",
          "within_deg",
          "across_deg",
          "across_share",
          "community_embeddedness",
          "community_core_score",
          "match_modularity_q",
        ],
        usage: "Louvain community 분석",
      },
      {
        group: "네트워크 위치",
        features: ["position_label", "betweenness_raw", "battle_center_score", "battle_bridge_score"],
        usage: "core/mid/peripheral/isolated 분석, battle bridge 분석",
      },
      {
        group: "ST 위치",
        features: [
          "landing_x",
          "landing_y",
          "early_farm_x",
          "early_farm_y",
          "centroid_x",
          "centroid_y",
        ],
        usage: "hotzone/outer farm, 이동 분석",
      },
      {
        group: "핫존",
        features: [
          "landing_density_score",
          "landing_neighbor_count",
          "distance_to_hotzone_m",
          "hotzone_density_q75",
          "hotzone_distance_q25",
          "outer_distance_q75",
          "farm_position_group",
        ],
        usage: "H1 hotzone vs outer farm",
      },
      {
        group: "이동성",
        features: [
          "trajectory_length_m",
          "movement_per_min",
          "movement_active_steps",
          "movement_step_count",
          "movement_active_frame_rate",
          "net_displacement_m",
          "position_std_m",
          "fixed_window_movement_per_min",
        ],
        usage: "H3, ActiveScore, 유형 분류",
      },
      {
        group: "ST 교전 참여",
        features: [
          "damage_event_dealt_count",
          "damage_event_taken_count",
          "damage_event_involved_count",
          "combat_event_rate_per_min",
          "combat_active_frames",
          "combat_exposure_frames",
          "combat_st_participation_rate",
        ],
        usage: "H2, FightScore",
      },
      {
        group: "페이즈별 동적 피쳐",
        features: [
          "phase",
          "phase_start_s",
          "phase_end_s",
          "phase_movement_per_min",
          "phase_active_frame_rate",
          "phase_combat_st_participation_rate",
          "phase_degree_per_min",
          "phase_weighted_deg_per_min",
          "phase_c_betweenness",
          "phase_c_pagerank",
          "phase_battle_bridge_score",
        ],
        usage: "페이즈별 4유형 정의와 전환 분석",
      },
      {
        group: "보정/표준화",
        features: ["rank_residual_after_survival", "rp_z_placement", "rp_z_kills", "rp_z_assists"],
        usage: "생존시간 효과와 매치별 차이 보정",
      },
      {
        group: "4유형 구성",
        features: [
          "z_combat_st_participation_rate",
          "z_combat_event_rate_per_min",
          "z_degree_per_min",
          "z_weighted_deg_per_min",
          "z_c_pagerank",
          "z_battle_bridge_score",
          "z_movement_per_min",
          "z_movement_active_frame_rate",
          "z_position_std_m",
          "z_landing_density_score",
          "z_landing_neighbor_count",
          "network_fight_score",
          "network_active_score",
          "fight_high",
          "active_high",
          "strategy_type",
        ],
        usage: "Hunter/Sniper/Traveler/Camper 정의",
      },
    ],
    rp_proxy_formula: {
      formula: "ranked_rp_proxy = z(placement_percentile) + z(team_kills) + 0.5 * z(team_assists)",
      note: "각 z-score는 매치 내부에서 계산. 정확한 RP 가중치는 비공개이므로 관측 가능한 팀 순위/킬/어시스트만 사용.",
      source:
        "https://support.pubg.com/hc/ko/articles/360048004774-경쟁전-관련-궁금증-해소하기",
    },
    limitations: [
      {
        limitation: "RP 공식 비공개",
        detail: "실제 RP 가중치와 티어 보너스는 알 수 없다.",
        next_step: "공식 점수 대신 proxy임을 명시하고 민감도 분석 수행.",
      },
      {
        limitation: "사용 피쳐 제한",
        detail: "현재 4유형은 실제 사용한 전투/이동/착륙 밀도/네트워크 중심성 피쳐 중심.",
        next_step: "추가 피쳐를 쓰려면 별도 feature 생성 후 분석 재실행.",
      },
      {
        limitation: "페이즈 내부 시간 해상도",
        detail: "phase별 동적 그래프까지 확장했지만 phase 내부 10초 단위 전환은 유형화하지 않았다.",
        next_step: "10초 snapshot graph 또는 rolling window graph로 초단기 전투 전환 분석.",
      },
      {
        limitation: "인과 해석 제한",
        detail: "네트워크 중심성이 좋은 순위의 원인인지 결과인지 완전히 분리하기 어렵다.",
        next_step: "시간 순서를 이용해 early/mid/late 네트워크 지표의 예측력 비교.",
      },
    ],
  };
}

// ---------- 메인 ----------

function main() {
  console.log("Building webapp data from", DATASET);

  // 입력 읽기
  console.log("Reading input CSVs/JSONs...");
  const meta = readJson(`${DATASET}/metadata.json`);
  const props = readCsv(
    `${DATASET}/basic_graph_analysis/match_network_properties.csv`,
  );
  // matches.csv 와 louvain_match_summary.csv 머지하여 modularity 추가
  const louvain = readCsv(
    `${DATASET}/advanced_graph_analysis/louvain_match_summary.csv`,
  );
  const louvainByMatch = new Map(louvain.map((r) => [r.match_id, r]));
  for (const r of props) {
    const lv = louvainByMatch.get(r.match_id);
    if (lv) {
      r.full_modularity_q = lv.modularity_q;
      r.full_n_communities = lv.n_communities;
    }
  }
  const posSummary = readCsv(
    `${DATASET}/advanced_graph_analysis/position_rank_summary.csv`,
  );
  const advSummary = readCsv(
    `${DATASET}/advanced_graph_analysis/battle_bridge_tests.csv`,
  );
  const typeSummary = readCsv(
    `${DATASET}/strategy_type_network_analysis/strategy_type_summary.csv`,
  );
  const edgeMixing = readCsv(
    `${DATASET}/strategy_type_network_analysis/strategy_type_edge_mixing.csv`,
  );
  const phaseSummary = readCsv(
    `${DATASET}/phase_strategy_type_analysis/phase_type_summary.csv`,
  );
  const transitions = readCsv(
    `${DATASET}/phase_strategy_type_analysis/phase_type_transitions.csv`,
  );
  const stTests = readCsv(
    `${DATASET}/st_hypothesis_analysis/st_hypothesis_tests.csv`,
  );
  const h3Completed = readCsv(
    `${DATASET}/phase_h3_analysis/phase_h3_completed_summary.csv`,
  );

  console.log("Building aggregate JSONs...");
  writeJson(path.join(OUT, "index.json"), buildIndex(meta, props));
  writeJson(
    path.join(OUT, "network_summary.json"),
    buildNetworkSummary(props, posSummary, null, advSummary),
  );
  writeJson(
    path.join(OUT, "strategy_summary.json"),
    buildStrategySummary(typeSummary, edgeMixing),
  );
  writeJson(
    path.join(OUT, "phase_summary.json"),
    buildPhaseSummary(phaseSummary, transitions),
  );
  writeJson(
    path.join(OUT, "hypotheses.json"),
    buildHypotheses(stTests, h3Completed, advSummary),
  );
  writeJson(path.join(OUT, "glossary.json"), buildGlossary());
  writeJson(path.join(OUT, "methods.json"), buildMethods());

  // 30개 매치 선정 + shard 머지
  console.log(`\nPicking ${SAMPLE_SIZE} representative matches...`);
  const picked = pickMatches(props, SAMPLE_SIZE);
  console.log(`  picked ${picked.length} matches:`);
  picked.forEach((m) =>
    console.log(
      `    ${m.match_id}  dur=${m.duration_s}s  teams=${m.n_teams}  Q=${m.modularity_q?.toFixed(3)}`,
    ),
  );

  console.log("\nLoading strategy_type_team_features.csv (large file)...");
  const teamFeatures = readCsv(
    `${DATASET}/strategy_type_network_analysis/strategy_type_team_features.csv`,
  );
  console.log(`  loaded ${teamFeatures.length} team-level rows`);

  console.log("\nMerging strategy_type fields into selected shards...");
  buildShards(picked, teamFeatures);

  console.log("\nBuilding match_index.json...");
  writeJson(
    path.join(OUT, "match_index.json"),
    buildMatchIndex(picked, SHARDS_OUT),
  );

  // 기존 불필요 데이터 제거
  console.log("\nCleaning up old data files...");
  const oldFiles = [
    "hypotheses.json", // (재생성됨)
    "network_insights.json",
  ];
  // 기존 디렉터리 삭제
  for (const subdir of ["archetype", "battle_center", "movement"]) {
    const p = path.join(OUT, subdir);
    if (fs.existsSync(p)) {
      fs.rmSync(p, { recursive: true, force: true });
      console.log(`  removed ${subdir}/`);
    }
  }
  for (const f of oldFiles) {
    const p = path.join(OUT, f);
    if (fs.existsSync(p) && f !== "hypotheses.json") {
      fs.unlinkSync(p);
      console.log(`  removed ${f}`);
    }
  }

  console.log("\nDone.");
}

main();
