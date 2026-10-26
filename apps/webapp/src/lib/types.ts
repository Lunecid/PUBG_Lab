/**
 * 새 webapp 데이터 타입.
 * 매치 shard 는 기존 export_json.py 스키마 유지 (AnimatedNetwork 호환).
 * 추가 집계 JSON 은 build-data.mjs 출력 형식.
 */

export type Archetype = "outer_farmer" | "combat_participant" | "neutral";
export type MovementLabel = "sedentary" | "mid" | "mobile";
export type PositionLabel = "core" | "mid" | "peripheral" | "isolated";
export type StrategyType = "Hunter" | "Sniper" | "Traveler" | "Camper";

// ---------- Match shard (AnimatedNetwork 입력) ----------

export interface MatchNode {
  id: number;
  team: number;
  rank: number | null;
  n_members: number;
  degree: number;
  events: number;
  dmg: number | null;
  dmg_taken: number | null;
  community: number;
  csize: number;
  position: PositionLabel | string;
  survival: number | null;
  archetype: Archetype;
  movement_label: MovementLabel;
  combat_participation: number | null;
  battle_center_score: number | null;
  trajectory_length_m: number | null;
  displacement_per_min: number | null;
  position_std_m: number | null;
  dist_from_center_m: number | null;
  centroid_first_xy: [number, number] | null;
  centroid_last_xy: [number, number] | null;
  // build-data.mjs 가 머지한 strategy_type 필드
  strategy_type?: StrategyType;
  fight_score?: number | null;
  active_score?: number | null;
  fight_high?: boolean;
  active_high?: boolean;
  team_kills?: number | null;
  team_assists?: number | null;
  ranked_rp_proxy?: number | null;
  movement_per_min?: number | null;
  combat_st_participation_rate?: number | null;
  battle_bridge_score?: number | null;
  distance_to_hotzone_m?: number | null;
  farm_position_group?: string | null;
}

export interface MatchEdge {
  source: number;
  target: number;
  weight: number;
  dmg: number;
  events: number;
}

export interface TeamStateAtT {
  deg: number;
  wdeg: number;
  ev: number;
  dmg: number;
  dmg_taken: number;
  comm: number;
  pos: PositionLabel | string;
  lc: number;
  bet: number;
  cp: number;
  arc: Archetype;
  traj: number;
  dpm: number;
  mvm: MovementLabel;
}

export interface MatchFrame {
  t: number;
  alive: number[];
  cx: Record<string, number>;
  cy: Record<string, number>;
  zone: { px: number; py: number; pr: number; sr: number } | null;
  ts?: Record<string, TeamStateAtT>;
}

export type TeamEvent = [number, number, number, number];

export interface MatchShard {
  match_id: string;
  map_name: string;
  game_mode: string;
  duration_s: number;
  arena_m: number;
  n_nodes: number;
  n_edges: number;
  n_players: number;
  modularity_q: number | null;
  frame_stride_s: number;
  landing_offset_s?: number;
  nodes: MatchNode[];
  edges: MatchEdge[];
  frames: MatchFrame[];
  events?: TeamEvent[];
}

// ---------- 집계 JSON ----------

export interface IndexFile {
  generated_at: string;
  map_name: string;
  game_mode: string;
  n_matches: number;
  n_teams: number;
  n_team_edges: number;
  n_team_centroids: number;
  early_window_sec: number;
  centroid_sample_sec: number;
  landing_offset_s: number;
  movement_window_s: number;
  /** 우선순위 순 미니맵 URL (high → low res) */
  map_urls?: string[];
  map_arena_m?: number;
  map_attribution?: {
    source: string;
    url: string;
    note: string;
  };
  network_avg: {
    n_edges: number | null;
    n_nodes: number | null;
    density: number | null;
    avg_degree: number | null;
    avg_clustering: number | null;
    diameter: number | null;
    avg_path_length: number | null;
    modularity_q: number | null;
    giant_size: number | null;
    giant_team_ratio: number | null;
  };
}

export interface NetworkSummary {
  match_stats: { label: string; value: number; format: string }[];
  position_table: {
    position: string;
    n: number | null;
    mean_rank: number | null;
    median_rank: number | null;
    mean_degree: number | null;
    mean_betweenness: number | null;
    mean_weighted_degree: number | null;
    mean_survival_s: number | null;
  }[];
  bridge_table: {
    metric: string;
    label: string;
    low_n: number | null;
    high_n: number | null;
    low_mean_rank: number | null;
    high_mean_rank: number | null;
    delta: number | null;
    mw_p: number | null;
    cliffs_delta: number | null;
  }[];
}

export interface StrategySummary {
  type_performance: {
    strategy_type: StrategyType;
    n: number;
    share: number;
    fight_score_mean: number;
    active_score_mean: number;
    final_rank_mean: number;
    final_rank_median: number;
    ranked_rp_proxy_mean: number;
    team_kills_mean: number;
    team_assists_mean: number;
    top10_rate: number;
    survival_seconds_mean: number;
  }[];
  type_profiles: {
    strategy_type: StrategyType;
    degree_per_min_mean: number;
    c_pagerank_mean: number;
    c_betweenness_mean: number;
    battle_bridge_score_mean: number;
    community_embeddedness_mean: number;
    movement_per_min_mean: number;
    combat_st_participation_rate_mean: number;
  }[];
  type_mixing: {
    type_pair: string;
    n_edges: number;
    total_events: number;
    total_damage: number;
    mean_weight: number;
    edge_share: number;
    event_share: number;
  }[];
}

export interface PhaseSummary {
  phase_type_performance: {
    phase: number;
    strategy_type: StrategyType;
    n: number;
    final_rank_mean: number;
    ranked_rp_proxy_mean: number;
    team_kills_mean: number;
    team_assists_mean: number;
    survived_phase_mean: number;
    fight_score_mean: number;
    active_score_mean: number;
    movement_per_min_mean: number;
    combat_st_participation_rate_mean: number;
    top10_rate: number;
  }[];
  top_transitions: {
    from_phase: number;
    to_phase: number;
    from_type: StrategyType;
    to_type: StrategyType;
    transition: string;
    n: number;
  }[];
  transition_matrices: Record<string, Record<string, number>>;
}

export interface HypothesisGroup {
  group: string;
  label: string;
  n?: number;
  mean_rank: number;
  ranked_rp_proxy: number;
  team_kills_mean: number;
  team_assists_mean: number;
  survival_seconds_mean: number;
  combat_st_participation_rate?: number;
  movement_per_min?: number;
  fixed_window_movement_per_min?: number;
}

export interface HypothesisTest {
  target: string;
  target_label: string;
  low_label: string;
  high_label: string;
  low_mean_target: number | null;
  high_mean_target: number | null;
  delta: number | null;
  mw_p: number | null;
  cliffs_delta: number | null;
  statistically_supported: boolean;
}

export interface HypothesisResult {
  title: string;
  verdict: string;
  verdict_label: string;
  conclusion: string;
  groups: HypothesisGroup[];
  tests: HypothesisTest[];
  phase_breakdown?: {
    phase: number;
    group: string;
    n: number;
    final_rank_mean: number;
    ranked_rp_proxy_mean: number;
    survived_phase_mean: number;
    top10_rate: number;
  }[];
}

export interface HypothesesFile {
  h1: HypothesisResult;
  h2: HypothesisResult;
  h3: HypothesisResult;
}

export interface MatchIndexEntry {
  match_id: string;
  duration_s: number;
  n_teams: number;
  n_edges?: number;
  modularity_q: number | null;
  density?: number | null;
  avg_clustering?: number | null;
  avg_path_length?: number | null;
  diameter?: number | null;
  type_counts?: Record<StrategyType, number>;
}

export interface GlossaryEntry {
  term: string;
  general: string;
  pubg: string;
}

export interface MethodsFile {
  data_sources: {
    source: string;
    features: string[];
    role: string;
  }[];
  graph_definition: {
    V: string;
    E: string;
    weight: string;
    events: string;
    direction: string;
    excluded: string[];
  };
  graph_windows: {
    window: string;
    definition: string;
    purpose: string;
  }[];
  feature_groups: {
    group: string;
    features: string[];
    usage: string;
  }[];
  rp_proxy_formula: {
    formula: string;
    note: string;
    source: string;
  };
  limitations: {
    limitation: string;
    detail: string;
    next_step: string;
  }[];
}
