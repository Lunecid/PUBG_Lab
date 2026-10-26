import { useNetworkSummary, useMatchIndex, useMatchShard, useIndex } from "../lib/data";
import StatCard from "../components/StatCard";
import Section from "../components/Section";
import Callout from "../components/Callout";
import DataTable, { Column } from "../components/DataTable";
import BarCompare from "../components/BarCompare";
import AnimatedNetwork from "../components/AnimatedNetwork";
import { useTimePlayer } from "../lib/useTimePlayer";
import TimeControls from "../components/TimeControls";
import { extractPhaseMarkers } from "../lib/phaseMarkers";
import { fmtDec, fmtNum, fmtP, fmtSigned } from "../lib/format";
import { useMemo } from "react";

export default function Network() {
  const sum = useNetworkSummary();
  const matches = useMatchIndex();
  const idx = useIndex();
  // network 페이지는 modularity 가 평균에 가까운 매치 사용
  const sigId = useMemo(() => {
    if (!matches.data) return undefined;
    const sorted = [...matches.data].sort(
      (a, b) =>
        Math.abs((a.modularity_q ?? 0) - 0.59) -
        Math.abs((b.modularity_q ?? 0) - 0.59),
    );
    return sorted[0]?.match_id;
  }, [matches.data]);
  const shard = useMatchShard(sigId);
  const player = useTimePlayer({ duration: shard.data?.duration_s ?? 600, autoplay: true, speed: 24, loop: true });
  const phaseMarkers = useMemo(
    () => (shard.data ? extractPhaseMarkers(shard.data.frames) : []),
    [shard.data],
  );

  if (sum.isLoading || !sum.data) return <div className="text-muted">불러오는 중…</div>;
  const { match_stats, position_table, bridge_table } = sum.data;

  // 보고서 §3.3 위치별 평균 순위 색상 (core=좋음 → peripheral=나쁨)
  const POS_COLOR: Record<string, string> = {
    core: "#15803d",
    mid: "#3a4a5a",
    peripheral: "#d97706",
    isolated: "#b91c1c",
  };
  const positionColumns: Column<typeof position_table[number]>[] = [
    { key: "pos", header: "위치", cell: (r) => r.position, align: "left" },
    { key: "n", header: "팀 수", cell: (r) => fmtNum(r.n), numeric: true, align: "right" },
    { key: "rank", header: "평균 순위", cell: (r) => fmtDec(r.mean_rank, 2), numeric: true, align: "right" },
    { key: "deg", header: "평균 degree", cell: (r) => fmtDec(r.mean_degree, 2), numeric: true, align: "right" },
    { key: "wd", header: "평균 weighted deg", cell: (r) => fmtDec(r.mean_weighted_degree, 1), numeric: true, align: "right" },
    { key: "bet", header: "평균 betweenness", cell: (r) => fmtDec(r.mean_betweenness, 3), numeric: true, align: "right" },
    { key: "surv", header: "평균 생존(s)", cell: (r) => fmtDec(r.mean_survival_s, 0), numeric: true, align: "right" },
  ];

  const bridgeColumns: Column<typeof bridge_table[number]>[] = [
    { key: "label", header: "지표", cell: (r) => r.label, align: "left" },
    { key: "ln", header: "low n", cell: (r) => fmtNum(r.low_n), numeric: true, align: "right" },
    { key: "hn", header: "high n", cell: (r) => fmtNum(r.high_n), numeric: true, align: "right" },
    { key: "lr", header: "low 평균 순위", cell: (r) => fmtDec(r.low_mean_rank, 2), numeric: true, align: "right" },
    { key: "hr", header: "high 평균 순위", cell: (r) => fmtDec(r.high_mean_rank, 2), numeric: true, align: "right" },
    { key: "delta", header: "Δ (high-low)", cell: (r) => fmtSigned(r.delta, 2), numeric: true, align: "right" },
    { key: "p", header: "p", cell: (r) => fmtP(r.mw_p), numeric: true, align: "right" },
    { key: "cliffs", header: "Cliff's δ", cell: (r) => fmtDec(r.cliffs_delta, 2), numeric: true, align: "right" },
  ];

  const positionBars = position_table
    .map((r) => ({
      label: r.position,
      value: r.mean_rank ?? 0,
      meta: `n=${fmtNum(r.n)}`,
      color: POS_COLOR[r.position] ?? "#3a4a5a",
    }))
    .sort((a, b) => a.value - b.value);

  return (
    <div className="space-y-10">
      <header className="space-y-2">
        <div className="jet text-[11px] uppercase tracking-wider text-accent">
          §3 · basic network analysis
        </div>
        <h1 className="news text-3xl font-semibold leading-tight">
          기본 네트워크 분석
        </h1>
        <p className="max-w-prose text-sm text-muted">
          PUBG 매치 한 번을 그래프 한 개로 본다. 노드는 팀, 엣지는 팀-팀 damage event,
          weight 는 누적 damage. 1,000매치 평균에서 어떤 구조가 보이는지, 그리고 네트워크
          위치가 최종 순위와 어떻게 연결되는지를 정리한다.
        </p>
      </header>

      {/* §3.1 매치 단위 구조 */}
      <Section title="매치 단위 네트워크 구조" kicker="§3.1">
        <p className="max-w-prose text-sm text-muted">
          매치 한 개의 평균 그래프는 <strong>약 43개의 팀-팀 교전 엣지</strong> 와 <strong>13%의 밀도</strong>를 가진다.
          평균 팀이 약 3.2 개의 상대 팀과 교전하고, Louvain 커뮤니티 평균 6.2 개로 분절된다.
        </p>
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-4">
          {match_stats.map((s) => (
            <StatCard key={s.label} label={s.label} value={fmtDec(s.value, 3)} />
          ))}
        </div>
        <Callout variant="key">
          PUBG 교전은 무작위 난전이 아니다. <strong>대부분의 팀이 giant component 에 포함</strong>되지만
          내부는 평균 6개 이상의 국소 교전권으로 나뉜다.
        </Callout>
      </Section>

      {/* §3.2 위치 정의 */}
      <Section title="네트워크 위치 정의" kicker="§3.2">
        <p className="max-w-prose text-sm text-muted">
          위치 라벨은 <strong>맵 좌표가 아니다</strong>. 각 매치의 팀 그래프에서 betweenness
          centrality 분위로 core / mid / peripheral / isolated 를 나눈다. core 는 “여러 교전 흐름
          사이를 잇는 중개 위치”다.
        </p>
        <div className="grid gap-3 md:grid-cols-2">
          {[
            { name: "core", desc: "degree > 0 이고 betweenness 상위 25%. 여러 교전권을 잇는 중개 팀.", color: POS_COLOR.core },
            { name: "mid", desc: "core/peripheral/isolated 이 아닌 팀. 중간 수준의 연결성.", color: POS_COLOR.mid },
            { name: "peripheral", desc: "degree > 0 이고 betweenness 하위 25%. 교전은 있지만 주변부.", color: POS_COLOR.peripheral },
            { name: "isolated", desc: "degree = 0. damage 교전이 관측되지 않은 팀.", color: POS_COLOR.isolated },
          ].map((p) => (
            <div key={p.name} className="rounded border border-rule bg-paper p-4">
              <div className="flex items-center gap-2">
                <span className="inline-block h-3 w-3 rounded-sm" style={{ background: p.color }} />
                <span className="jet text-sm font-medium uppercase">{p.name}</span>
              </div>
              <p className="mt-2 text-sm text-muted">{p.desc}</p>
            </div>
          ))}
        </div>
      </Section>

      {/* §3.3 위치별 평균 순위 */}
      <Section title="네트워크 위치 → 최종 순위" kicker="§3.3">
        <p className="max-w-prose text-sm text-muted">
          core 팀의 평균 순위는 <strong>7.6</strong>, peripheral 은 <strong>17.7</strong>, isolated 도{" "}
          <strong>17.8</strong>. 고립은 안전이 아니라 영향력 부재로 이어진다.
        </p>
        <div className="rounded border border-rule bg-paper p-4">
          <BarCompare items={positionBars} valueDigits={2} unit="등" labelWidth={110} />
        </div>
        <DataTable columns={positionColumns} rows={position_table} rowKey={(r) => r.position} />
      </Section>

      {/* §3.4 battle bridge */}
      <Section title="Battle bridge 와 중심성" kicker="§3.4">
        <p className="max-w-prose text-sm text-muted">
          여러 교전권을 잇는 <strong>battle bridge score</strong> 가 높은 팀의 평균 순위는{" "}
          <strong>6.86</strong>, 낮은 팀은 <strong>19.43</strong>. betweenness, across-deg, PageRank 도
          모두 같은 방향이다. 반대로 한 커뮤니티에 갇힌 <strong>community embeddedness</strong> 가 높은
          팀은 평균 순위가 낮다.
        </p>
        <DataTable columns={bridgeColumns} rows={bridge_table} rowKey={(r) => r.metric} />
        <Callout variant="key">
          좋은 순위의 팀은 한 교전권에 갇힌 팀이 아니라, <strong>여러 교전권 사이를 연결하거나
          여러 전투 흐름을 제어하는 팀</strong>이다.
        </Callout>
      </Section>

      {/* 인라인 그래프 */}
      <Section title="라이브 매치 (네트워크 위치 색상)" kicker="example">
        <p className="max-w-prose text-sm text-muted">
          한 매치를 colorBy=position 으로 재생한다. core (붉은) 팀이 그래프 중앙에 모이는 경향과,
          isolated 팀이 외곽에 머물다 일찍 사라지는 모습을 볼 수 있다.
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
              colorBy="position"
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
