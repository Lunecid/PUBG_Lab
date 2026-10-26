import { useStrategySummary, useMatchIndex, useMatchShard, useIndex } from "../lib/data";
import Section from "../components/Section";
import Callout from "../components/Callout";
import DataTable, { Column } from "../components/DataTable";
import FormulaBlock from "../components/FormulaBlock";
import QuadrantChart from "../components/QuadrantChart";
import BarCompare from "../components/BarCompare";
import StrategyBadge, { STRATEGY_COLORS } from "../components/StrategyBadge";
import AnimatedNetwork from "../components/AnimatedNetwork";
import { useTimePlayer } from "../lib/useTimePlayer";
import TimeControls from "../components/TimeControls";
import { extractPhaseMarkers } from "../lib/phaseMarkers";
import { fmtDec, fmtNum, fmtPct } from "../lib/format";
import { useMemo } from "react";
import type { StrategyType } from "../lib/types";

const TYPES: StrategyType[] = ["Hunter", "Sniper", "Traveler", "Camper"];

const TYPE_BLURB: Record<StrategyType, { headline: string; body: string }> = {
  Hunter: {
    headline: "순위·킬·RP proxy 모두 강함",
    body: "전투 네트워크의 중심에 들어가면서도 활발하게 이동한다. PageRank, betweenness, battle bridge score 가 가장 높고, 평균 킬도 압도적이다. 본 데이터에서 점수형 최강 전략.",
  },
  Traveler: {
    headline: "순위 안정형, RP 는 부족",
    body: "교전성은 낮지만 활발히 이동하고 공간을 탐색한다. 평균 순위는 Hunter 와 거의 같지만 평균 킬이 1.86 으로 낮아 RP proxy 는 ≈0.",
  },
  Sniper: {
    headline: "위험 대비 성과 최저",
    body: "FightScore 는 높지만 ActiveScore 가 낮다. 교전 참여율은 가장 높지만 평균 생존시간이 555초로 가장 짧고 평균 순위가 19.16 으로 가장 나쁘다.",
  },
  Camper: {
    headline: "단순 존버는 한계",
    body: "교전성과 활동성이 모두 낮다. Sniper 보단 낫지만 Hunter/Traveler 보다 순위와 RP proxy 가 모두 낮다. 단순 존버 전략의 상한선을 보여준다.",
  },
};

export default function Strategy() {
  const sum = useStrategySummary();
  const matches = useMatchIndex();
  const idx = useIndex();
  const sigId = useMemo(() => matches.data?.[0]?.match_id, [matches.data]);
  const shard = useMatchShard(sigId);
  const player = useTimePlayer({ duration: shard.data?.duration_s ?? 600, autoplay: true, speed: 24, loop: true });
  const phaseMarkers = useMemo(
    () => (shard.data ? extractPhaseMarkers(shard.data.frames) : []),
    [shard.data],
  );

  if (sum.isLoading || !sum.data) return <div className="text-muted">불러오는 중…</div>;
  const { type_performance, type_profiles, type_mixing } = sum.data;

  const perfCols: Column<typeof type_performance[number]>[] = [
    { key: "type", header: "유형", cell: (r) => <StrategyBadge type={r.strategy_type} />, align: "left" },
    { key: "n", header: "n", cell: (r) => fmtNum(r.n), numeric: true, align: "right" },
    { key: "share", header: "비율", cell: (r) => fmtPct(r.share, 1), numeric: true, align: "right" },
    { key: "rank", header: "평균 순위", cell: (r) => fmtDec(r.final_rank_mean, 2), numeric: true, align: "right" },
    { key: "med", header: "중앙 순위", cell: (r) => fmtDec(r.final_rank_median, 0), numeric: true, align: "right" },
    { key: "rp", header: "RP proxy", cell: (r) => fmtDec(r.ranked_rp_proxy_mean, 2), numeric: true, align: "right" },
    { key: "kills", header: "평균 킬", cell: (r) => fmtDec(r.team_kills_mean, 2), numeric: true, align: "right" },
    { key: "top10", header: "Top10", cell: (r) => fmtPct(r.top10_rate, 1), numeric: true, align: "right" },
    { key: "surv", header: "생존(s)", cell: (r) => fmtDec(r.survival_seconds_mean, 0), numeric: true, align: "right" },
  ];

  const profileCols: Column<typeof type_profiles[number]>[] = [
    { key: "type", header: "유형", cell: (r) => <StrategyBadge type={r.strategy_type} />, align: "left" },
    { key: "deg", header: "degree/min", cell: (r) => fmtDec(r.degree_per_min_mean, 3), numeric: true, align: "right" },
    { key: "pr", header: "PageRank", cell: (r) => fmtDec(r.c_pagerank_mean, 3), numeric: true, align: "right" },
    { key: "bet", header: "betweenness", cell: (r) => fmtDec(r.c_betweenness_mean, 3), numeric: true, align: "right" },
    { key: "br", header: "battle bridge", cell: (r) => fmtDec(r.battle_bridge_score_mean, 2), numeric: true, align: "right" },
    { key: "em", header: "embeddedness", cell: (r) => fmtDec(r.community_embeddedness_mean, 2), numeric: true, align: "right" },
    { key: "mv", header: "이동 m/min", cell: (r) => fmtDec(r.movement_per_min_mean, 0), numeric: true, align: "right" },
    { key: "cp", header: "교전 참여율", cell: (r) => fmtDec(r.combat_st_participation_rate_mean, 3), numeric: true, align: "right" },
  ];

  const mixCols: Column<typeof type_mixing[number]>[] = [
    { key: "pair", header: "유형 조합", cell: (r) => r.type_pair, align: "left" },
    { key: "n", header: "edges", cell: (r) => fmtNum(r.n_edges), numeric: true, align: "right" },
    { key: "ev", header: "events", cell: (r) => fmtNum(r.total_events), numeric: true, align: "right" },
    { key: "dmg", header: "총 damage", cell: (r) => fmtNum(r.total_damage, 0), numeric: true, align: "right" },
    { key: "share", header: "edge share", cell: (r) => fmtPct(r.edge_share, 2), numeric: true, align: "right" },
  ];

  // 사분면 차트용 데이터 (각 유형 평균 위치)
  const quadrantPoints = type_performance.map((t) => ({
    x: t.fight_score_mean,
    y: t.active_score_mean,
    type: t.strategy_type,
  }));
  const quadrantLabels = {
    Hunter: `n=${fmtNum(type_performance.find((t) => t.strategy_type === "Hunter")?.n ?? 0)}`,
    Sniper: `n=${fmtNum(type_performance.find((t) => t.strategy_type === "Sniper")?.n ?? 0)}`,
    Traveler: `n=${fmtNum(type_performance.find((t) => t.strategy_type === "Traveler")?.n ?? 0)}`,
    Camper: `n=${fmtNum(type_performance.find((t) => t.strategy_type === "Camper")?.n ?? 0)}`,
  };

  // 평균 순위 비교 막대
  const rankBars = TYPES.map((t) => {
    const r = type_performance.find((p) => p.strategy_type === t)!;
    return {
      label: t,
      value: r.final_rank_mean,
      meta: `RP=${r.ranked_rp_proxy_mean.toFixed(2)}`,
      color: STRATEGY_COLORS[t],
    };
  }).sort((a, b) => a.value - b.value);

  return (
    <div className="space-y-10">
      <header className="space-y-2">
        <div className="jet text-[11px] uppercase tracking-wider text-accent">
          §5–6 · strategy types
        </div>
        <h1 className="news text-3xl font-semibold leading-tight">
          4유형 전략: Hunter / Sniper / Traveler / Camper
        </h1>
        <p className="max-w-prose text-sm text-muted">
          각 팀의 행동을 두 점수 — Network FightScore (전투 연결성) 와 Network ActiveScore (이동/탐색)
          — 로 요약하고, 매치 내 중앙값을 기준으로 4유형으로 분류한다. 킬과 어시스트는 결과
          변수이므로 유형 정의에는 넣지 않는다.
        </p>
      </header>

      {/* 점수 정의 */}
      <Section title="두 점수: FightScore × ActiveScore" kicker="§5.1–5.2">
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <h3 className="news mb-2 text-lg font-medium">Network FightScore</h3>
            <p className="mb-3 text-sm text-muted">
              팀이 전투 네트워크 안에서 얼마나 자주, 넓게, 중심적으로 교전에 연결됐는지.
            </p>
            <FormulaBlock>{`Network FightScore =
mean(
  z(combat_st_participation_rate),
  z(combat_event_rate_per_min),
  z(degree_per_min),
  z(weighted_deg_per_min),
  z(c_pagerank),
  z(battle_bridge_score)
)`}</FormulaBlock>
          </div>
          <div>
            <h3 className="news mb-2 text-lg font-medium">Network ActiveScore</h3>
            <p className="mb-3 text-sm text-muted">
              착륙 후 얼마나 공간을 탐색하고, 이동하며, 고밀도 지역에서 활동했는지.
            </p>
            <FormulaBlock>{`Network ActiveScore =
mean(
  z(movement_per_min),
  z(movement_active_frame_rate),
  z(position_std_m),
  z(landing_density_score),
  z(landing_neighbor_count)
)`}</FormulaBlock>
          </div>
        </div>
      </Section>

      {/* 사분면 차트 */}
      <Section title="2×2 매트릭스" kicker="§5.3">
        <p className="max-w-prose text-sm text-muted">
          매치 안에서 두 점수의 중앙값을 기준으로 high/low 를 나눈 뒤 4사분면으로 분류한다.
          아래는 1,000매치 평균 위치.
        </p>
        <div className="grid items-center gap-4 md:grid-cols-2">
          <div className="rounded border border-rule bg-paper p-4">
            <QuadrantChart points={quadrantPoints} labels={quadrantLabels} size={420} />
          </div>
          <div className="space-y-3 text-sm">
            {TYPES.map((t) => (
              <div key={t} className="rounded border border-rule bg-paper p-3">
                <StrategyBadge type={t} />
                <div className="mt-1 font-medium">{TYPE_BLURB[t].headline}</div>
                <div className="mt-1 text-xs text-muted">{TYPE_BLURB[t].body}</div>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* 유형별 성과 */}
      <Section title="유형별 성과" kicker="§6.1">
        <DataTable columns={perfCols} rows={type_performance} rowKey={(r) => r.strategy_type} />
        <div className="rounded border border-rule bg-paper p-4">
          <div className="jet mb-3 text-[10px] uppercase text-muted">평균 순위 (낮을수록 좋음)</div>
          <BarCompare items={rankBars} valueDigits={2} unit="등" labelWidth={90} />
        </div>
        <Callout variant="key">
          순위 관점에서는 <strong>Hunter ≈ Traveler &gt; Camper &gt; Sniper</strong>. 하지만 RP proxy 에서는
          킬 보상까지 가져가는 <strong>Hunter 가 압도적</strong>이다.
        </Callout>
      </Section>

      {/* 네트워크 프로필 */}
      <Section title="유형별 네트워크 프로필" kicker="§6.2">
        <DataTable columns={profileCols} rows={type_profiles} rowKey={(r) => r.strategy_type} />
        <Callout>
          Hunter 는 교전성·활동성·중심성 모두 가장 높다. Sniper 는 degree/min 은 높지만 PageRank /
          betweenness / battle bridge 가 Hunter 보다 낮다 — <em>많이 싸우지만 전략적 네트워크
          중심을 확보하지 못한다</em>.
        </Callout>
      </Section>

      {/* 유형 간 mixing */}
      <Section title="유형 간 교전 조합" kicker="§6.4">
        <p className="max-w-prose text-sm text-muted">
          어떤 유형 페어가 가장 많이 교전했는지. Hunter 가 모든 주요 조합에서 비중 높게 등장한다 —
          단순히 킬이 많은 팀이 아니라 <strong>매치 내 교전 네트워크의 여러 관계에 반복적으로 등장하는
          중심적 전략 유형</strong>이다.
        </p>
        <DataTable columns={mixCols} rows={type_mixing} rowKey={(r) => r.type_pair} />
      </Section>

      {/* 인라인 그래프 */}
      <Section title="라이브 매치 (4유형 색상)" kicker="example">
        {shard.data ? (
          <div>
            <AnimatedNetwork
              nodes={shard.data.nodes}
              edges={shard.data.edges}
              events={shard.data.events ?? []}
              frames={shard.data.frames}
              arenaM={shard.data.arena_m}
              t={player.t}
              colorBy="archetype"
              size={540}
              mapUrls={idx.data?.map_urls}
            />
            <div className="mt-2">
              <TimeControls
                duration={shard.data.duration_s}
                t={player.t}
                onSeek={player.seek}
                playing={player.playing}
                onTogglePlay={player.togglePlay}
                speed={player.speed}
                onSpeedChange={player.setSpeed}
                phaseMarkers={phaseMarkers}
                compact
              />
            </div>
          </div>
        ) : null}
      </Section>
    </div>
  );
}
