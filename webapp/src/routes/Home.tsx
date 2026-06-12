import { Link } from "react-router-dom";
import { useIndex, useMatchIndex, useMatchShard, useStrategySummary } from "../lib/data";
import StatCard from "../components/StatCard";
import Section from "../components/Section";
import Callout from "../components/Callout";
import StrategyBadge from "../components/StrategyBadge";
import AnimatedNetwork from "../components/AnimatedNetwork";
import { useTimePlayer } from "../lib/useTimePlayer";
import TimeControls from "../components/TimeControls";
import { extractPhaseMarkers } from "../lib/phaseMarkers";
import { fmtDec, fmtNum } from "../lib/format";
import { useMemo } from "react";

export default function Home() {
  const idx = useIndex();
  const matches = useMatchIndex();
  const strat = useStrategySummary();

  // 시그니처 매치: 첫 번째 (또는 modularity 중간값)
  const signatureMatchId = useMemo(() => {
    if (!matches.data || matches.data.length === 0) return undefined;
    const sorted = [...matches.data].sort(
      (a, b) => (a.modularity_q ?? 0) - (b.modularity_q ?? 0),
    );
    return sorted[Math.floor(sorted.length / 2)].match_id;
  }, [matches.data]);
  const shard = useMatchShard(signatureMatchId);
  const player = useTimePlayer({
    duration: shard.data?.duration_s ?? 600,
    autoplay: true,
    speed: 24,
    loop: true,
  });
  const phaseMarkers = useMemo(
    () => (shard.data ? extractPhaseMarkers(shard.data.frames) : []),
    [shard.data],
  );

  if (idx.isLoading || strat.isLoading) {
    return <div className="text-muted">불러오는 중…</div>;
  }
  if (idx.error || !idx.data) {
    return <div className="text-bad">데이터 로딩 실패</div>;
  }

  const i = idx.data;
  const types = strat.data?.type_performance ?? [];

  return (
    <div className="space-y-10">
      {/* Hero */}
      <section className="space-y-3">
        <div className="jet text-[11px] uppercase tracking-wider text-accent">
          PUBG · Baltic_Main · squad
        </div>
        <h1 className="news max-w-3xl text-4xl font-semibold leading-tight md:text-5xl">
          1,000개 PUBG 매치를 팀 전투 네트워크로 분석했다.
        </h1>
        <p className="max-w-2xl text-base leading-relaxed text-muted">
          팀-팀 damage event 를 엣지로 잇고, 팀의 네트워크 위치와 시공간(ST) 행동을 결합해
          Hunter / Sniper / Traveler / Camper 4유형의 성과 차이를 측정했다.
        </p>
      </section>

      {/* 핵심 숫자 */}
      <section className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatCard label="매치" value={fmtNum(i.n_matches)} hint="Baltic_Main · squad" />
        <StatCard label="팀 노드" value={fmtNum(i.n_teams)} />
        <StatCard label="교전 엣지" value={fmtNum(i.n_team_edges)} hint="팀-팀 damage 관계" />
        <StatCard
          label="Modularity Q"
          value={fmtDec(i.network_avg.modularity_q, 3)}
          hint="Louvain 평균"
        />
      </section>

      {/* 시그니처 그래프 */}
      <Section title="움직이는 전투 네트워크" kicker="signature match">
        <p className="max-w-prose text-sm text-muted">
          한 매치를 자동 재생한다. 각 점은 팀이고, 색은 4유형 분류(Hunter/Sniper/Traveler/Camper),
          연결선은 누적 damage 교전을 나타낸다. 시간이 지나면서 팀이 이동하고, 교전권이
          형성되며, 자기장이 좁혀진다.
        </p>
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
              mapUrls={i.map_urls}
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
            <div className="mt-2 flex items-center justify-end">
              <Link
                to={`/explorer/${signatureMatchId}`}
                className="jet rounded border border-rule bg-paper px-2 py-1 text-xs text-muted hover:bg-cream hover:no-underline"
              >
                매치 상세에서 더 정밀하게 조작 ▸
              </Link>
            </div>
          </div>
        ) : (
          <div className="flex h-[540px] items-center justify-center rounded border border-rule bg-cream/30 text-muted">
            매치 그래프 로딩 중…
          </div>
        )}
      </Section>

      {/* 4유형 한 줄 요약 */}
      <Section title="4유형 핵심 성과" kicker="key finding">
        <div className="grid gap-3 md:grid-cols-2">
          {types.map((t) => (
            <Link
              key={t.strategy_type}
              to="/strategy"
              className="rounded border border-rule bg-paper p-4 transition-colors hover:border-accent hover:no-underline"
            >
              <div className="flex items-center justify-between">
                <StrategyBadge type={t.strategy_type} size="md" />
                <div className="jet text-[10px] text-muted">
                  n={fmtNum(t.n)}
                </div>
              </div>
              <div className="mt-3 grid grid-cols-3 gap-2">
                <div>
                  <div className="jet text-[10px] uppercase text-muted">평균 순위</div>
                  <div className="numeric text-lg">{fmtDec(t.final_rank_mean, 2)}</div>
                </div>
                <div>
                  <div className="jet text-[10px] uppercase text-muted">RP proxy</div>
                  <div className="numeric text-lg">{fmtDec(t.ranked_rp_proxy_mean, 2)}</div>
                </div>
                <div>
                  <div className="jet text-[10px] uppercase text-muted">평균 킬</div>
                  <div className="numeric text-lg">{fmtDec(t.team_kills_mean, 2)}</div>
                </div>
              </div>
            </Link>
          ))}
        </div>
        <Callout variant="key" label="결론 1줄">
          순위에선 Hunter ≈ Traveler 가 강하지만, 킬 보상이 들어간 RP proxy 에서는 <strong>Hunter 가 압도적</strong>이다.
          Sniper 는 교전성은 높지만 활동성이 낮아 위험 대비 성과가 가장 낮다.
        </Callout>
      </Section>

      {/* 6개 RQ */}
      <Section title="6개 연구 질문" kicker="research questions">
        <div className="grid gap-3 md:grid-cols-2">
          {[
            { id: "RQ1", text: "PUBG 팀 간 교전 네트워크는 어떤 구조를 가지는가?", to: "/network" },
            { id: "RQ2", text: "네트워크 중심 위치에 있는 팀은 더 좋은 순위를 얻는가?", to: "/network" },
            { id: "RQ3", text: "핫존/외곽, 교전 참여율, 이동성은 순위에 어떤 차이를 만드는가?", to: "/hypotheses" },
            { id: "RQ4", text: "Hunter/Sniper/Traveler/Camper 는 네트워크 지표로 어떻게 정의되는가?", to: "/strategy" },
            { id: "RQ5", text: "4유형 전략은 순위·킬·RP proxy 에서 어떤 차이를 보이는가?", to: "/strategy" },
            { id: "RQ6", text: "같은 팀의 전략 유형은 phase 가 바뀌면서 어떻게 전환되는가?", to: "/phases" },
          ].map((r) => (
            <Link
              key={r.id}
              to={r.to}
              className="group rounded border border-rule bg-paper p-4 transition-colors hover:border-accent hover:bg-cream/30 hover:no-underline"
            >
              <div className="jet text-[10px] uppercase tracking-wider text-accent">{r.id}</div>
              <div className="mt-1 text-sm leading-relaxed">{r.text}</div>
              <div className="jet mt-2 text-[10px] text-muted opacity-0 transition-opacity group-hover:opacity-100">
                자세히 →
              </div>
            </Link>
          ))}
        </div>
      </Section>
    </div>
  );
}
