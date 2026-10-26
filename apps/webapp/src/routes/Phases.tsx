import { usePhaseSummary } from "../lib/data";
import Section from "../components/Section";
import Callout from "../components/Callout";
import DataTable, { Column } from "../components/DataTable";
import Heatmap from "../components/Heatmap";
import SankeyTransitions from "../components/SankeyTransitions";
import StrategyBadge from "../components/StrategyBadge";
import { fmtDec, fmtNum, fmtPct } from "../lib/format";
import type { StrategyType } from "../lib/types";

const TYPES: StrategyType[] = ["Hunter", "Sniper", "Traveler", "Camper"];

export default function Phases() {
  const ps = usePhaseSummary();

  if (ps.isLoading || !ps.data) return <div className="text-muted">불러오는 중…</div>;
  const { phase_type_performance, transition_matrices } = ps.data;

  const phases = Array.from(new Set(phase_type_performance.map((p) => p.phase))).sort((a, b) => a - b);

  // 히트맵 cells
  const rankCells = phase_type_performance.map((p) => ({
    row: p.phase,
    col: p.strategy_type as StrategyType,
    value: p.final_rank_mean,
  }));
  const rpCells = phase_type_performance.map((p) => ({
    row: p.phase,
    col: p.strategy_type as StrategyType,
    value: p.ranked_rp_proxy_mean,
  }));
  const survCells = phase_type_performance.map((p) => ({
    row: p.phase,
    col: p.strategy_type as StrategyType,
    value: p.survived_phase_mean,
  }));

  // 표
  const cols: Column<typeof phase_type_performance[number]>[] = [
    { key: "phase", header: "phase", cell: (r) => r.phase, align: "right", numeric: true },
    { key: "type", header: "유형", cell: (r) => <StrategyBadge type={r.strategy_type as StrategyType} />, align: "left" },
    { key: "n", header: "n", cell: (r) => fmtNum(r.n), numeric: true, align: "right" },
    { key: "rank", header: "최종 평균 순위", cell: (r) => fmtDec(r.final_rank_mean, 2), numeric: true, align: "right" },
    { key: "rp", header: "RP proxy", cell: (r) => fmtDec(r.ranked_rp_proxy_mean, 2), numeric: true, align: "right" },
    { key: "kills", header: "평균 킬", cell: (r) => fmtDec(r.team_kills_mean, 2), numeric: true, align: "right" },
    { key: "psurv", header: "phase 생존율", cell: (r) => fmtPct(r.survived_phase_mean, 1), numeric: true, align: "right" },
    { key: "fight", header: "fight z", cell: (r) => fmtDec(r.fight_score_mean, 2), numeric: true, align: "right" },
    { key: "active", header: "active z", cell: (r) => fmtDec(r.active_score_mean, 2), numeric: true, align: "right" },
  ];

  // 전환 매트릭스 — phase 1→2, 2→3 등 (인접 phase 만)
  const adjacentTransitions = phases
    .slice(0, -1)
    .map((from) => ({ from, to: from + 1, key: `${from}_${from + 1}` }))
    .filter((t) => transition_matrices[t.key]);

  return (
    <div className="space-y-10">
      <header className="space-y-2">
        <div className="jet text-[11px] uppercase tracking-wider text-accent">
          §7 · phase-based dynamic typing
        </div>
        <h1 className="news text-3xl font-semibold leading-tight">
          페이즈별 동적 4유형
        </h1>
        <p className="max-w-prose text-sm text-muted">
          정적 4유형은 한 팀을 매치 전체에서 하나의 유형으로 분류한다. 그러나 PUBG 의 전략은
          phase 가 진행되며 바뀐다. 같은 팀이 phase 1 에서 Traveler 였다가 phase 2 에서
          Hunter 가 되는 식이다. 본 분석은 <strong>74,329 개의 team-phase 관측치</strong> 를 단위로
          유형 전환과 성과를 본다.
        </p>
      </header>

      {/* phase × type 히트맵 (3종) */}
      <Section title="페이즈 × 유형 히트맵" kicker="§7.3">
        <div className="grid gap-6 md:grid-cols-2">
          <div className="rounded border border-rule bg-paper p-4">
            <div className="jet mb-2 text-[11px] uppercase text-muted">최종 평균 순위 (낮을수록 좋음)</div>
            <Heatmap
              cells={rankCells}
              rows={phases}
              cols={TYPES}
              scale={{ min: 2, max: 20, lowColor: "#15803d", highColor: "#b91c1c" }}
              valueDigits={1}
            />
          </div>
          <div className="rounded border border-rule bg-paper p-4">
            <div className="jet mb-2 text-[11px] uppercase text-muted">RP proxy (높을수록 좋음, 0 기준)</div>
            <Heatmap
              cells={rpCells}
              rows={phases}
              cols={TYPES}
              scale={{ min: -1, max: 4.5, midZero: true, highColor: "#b02a2a" }}
              valueDigits={2}
            />
          </div>
          <div className="rounded border border-rule bg-paper p-4 md:col-span-2">
            <div className="jet mb-2 text-[11px] uppercase text-muted">phase 생존율</div>
            <Heatmap
              cells={survCells}
              rows={phases}
              cols={TYPES}
              scale={{ min: 0.3, max: 1.0, lowColor: "#fefcf7", highColor: "#15803d" }}
              valueDigits={2}
            />
          </div>
        </div>
        <Callout variant="key">
          후반 phase 일수록 모든 유형의 RP proxy 가 올라간다 (살아남은 팀일수록 점수 기여 큼).
          하지만 모든 phase 에서 <strong>Hunter 의 RP proxy 가 가장 높고, Traveler 의 phase 생존율이
          가장 안정적</strong>이다.
        </Callout>
      </Section>

      {/* 페이즈 전환 sankey */}
      <Section title="유형 전환" kicker="§7.5">
        <p className="max-w-prose text-sm text-muted">
          인접한 두 phase 사이의 4유형 전환 흐름. 각 색은 from-type 색.
          초반 Traveler 가 다양한 phase 2 유형으로 갈라지는 점, 그리고 중반 이후 Hunter→Hunter 자기
          유지가 강하게 나타나는 점이 보인다.
        </p>
        <div className="grid gap-4 md:grid-cols-2">
          {adjacentTransitions.slice(0, 4).map((tr) => (
            <div key={tr.key} className="rounded border border-rule bg-paper p-4">
              <div className="jet mb-2 text-[10px] uppercase text-muted">
                phase {tr.from} → phase {tr.to}
              </div>
              <SankeyTransitions
                matrix={transition_matrices[tr.key]}
                fromLabel={`phase ${tr.from}`}
                toLabel={`phase ${tr.to}`}
              />
            </div>
          ))}
        </div>
      </Section>

      {/* 페이즈별 성과 표 */}
      <Section title="페이즈 × 유형 전체 표" kicker="reference">
        <DataTable
          columns={cols}
          rows={phase_type_performance}
          rowKey={(r) => `${r.phase}-${r.strategy_type}`}
        />
      </Section>

      <Section title="해석" kicker="§7.4">
        <div className="space-y-2 text-sm leading-relaxed">
          <Callout label="초반 phase 1">
            Traveler 로 관측된 팀들이 최종 평균 순위 9.79 / phase 생존율 89.6% 로 가장 안정적.
            Hunter 는 평균 킬 6.07 / RP proxy 0.75 로 점수형 우세.
            <strong> 초반에는 Traveler = 생존 안정형, Hunter = 점수 기대형으로 분리</strong>.
          </Callout>
          <Callout label="중반 phase 2–3">
            Traveler 와 Hunter 의 평균 순위가 비슷해진다. 그러나 RP proxy 는 Hunter 가 phase 2 에서
            1.82, phase 3 에서 2.27 로 가장 높다.
          </Callout>
          <Callout label="후반 phase 4–6">
            Hunter 가 평균 순위와 RP proxy 모두 가장 강하다. Phase 6 Hunter 는 평균 순위 2.56 /
            RP proxy 4.07 / 평균 킬 11.90.
            <strong> 후반에 살아남은 Hunter 는 단순 교전형이 아니라 중심성과 이동성을 유지하면서
            킬 보상을 회수하는 팀</strong>이다.
          </Callout>
          <Callout variant="warn" label="Camper 후반 평균 순위 함정">
            Camper 의 후반 평균 순위가 좋아 보이지만, 이미 살아남은 팀만 후반 phase 에 남는
            <strong> 선택 효과</strong> 다. 같은 phase 안에서 비교하면 Hunter / Traveler 가 Camper 보다 RP
            proxy 가 높다.
          </Callout>
        </div>
      </Section>
    </div>
  );
}
