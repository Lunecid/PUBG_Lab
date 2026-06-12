import { useHypotheses } from "../lib/data";
import Section from "../components/Section";
import Callout from "../components/Callout";
import HypothesisCard from "../components/HypothesisCard";
import FormulaBlock from "../components/FormulaBlock";
import DataTable, { Column } from "../components/DataTable";
import { fmtDec, fmtNum } from "../lib/format";

export default function Hypotheses() {
  const hyp = useHypotheses();

  if (hyp.isLoading || !hyp.data) return <div className="text-muted">불러오는 중…</div>;
  const { h1, h2, h3 } = hyp.data;

  // H3 phase breakdown 표
  const phaseRows = (h3.phase_breakdown ?? []).filter((r) => r.group === "phase_sedentary" || r.group === "phase_mobile");
  const phaseGrouped = Array.from(
    phaseRows.reduce((m, r) => {
      const arr = m.get(r.phase) ?? {};
      arr[r.group] = r;
      m.set(r.phase, arr);
      return m;
    }, new Map<number, Record<string, typeof phaseRows[number]>>()).entries(),
  ).map(([phase, g]) => ({
    phase,
    sedN: g.phase_sedentary?.n ?? null,
    mobN: g.phase_mobile?.n ?? null,
    sedRank: g.phase_sedentary?.final_rank_mean ?? null,
    mobRank: g.phase_mobile?.final_rank_mean ?? null,
    sedRp: g.phase_sedentary?.ranked_rp_proxy_mean ?? null,
    mobRp: g.phase_mobile?.ranked_rp_proxy_mean ?? null,
  }));
  const phaseCols: Column<typeof phaseGrouped[number]>[] = [
    { key: "phase", header: "phase", cell: (r) => r.phase, numeric: true, align: "right" },
    { key: "sedN", header: "sed n", cell: (r) => fmtNum(r.sedN), numeric: true, align: "right" },
    { key: "mobN", header: "mob n", cell: (r) => fmtNum(r.mobN), numeric: true, align: "right" },
    { key: "sedR", header: "sed 순위", cell: (r) => fmtDec(r.sedRank, 2), numeric: true, align: "right" },
    { key: "mobR", header: "mob 순위", cell: (r) => fmtDec(r.mobRank, 2), numeric: true, align: "right" },
    { key: "sedRp", header: "sed RP", cell: (r) => fmtDec(r.sedRp, 2), numeric: true, align: "right" },
    { key: "mobRp", header: "mob RP", cell: (r) => fmtDec(r.mobRp, 2), numeric: true, align: "right" },
  ];

  return (
    <div className="space-y-10">
      <header className="space-y-2">
        <div className="jet text-[11px] uppercase tracking-wider text-accent">
          §4 · ST hypotheses
        </div>
        <h1 className="news text-3xl font-semibold leading-tight">
          ST 가설: H1 · H2 · H3
        </h1>
        <p className="max-w-prose text-sm text-muted">
          시공간(ST) 행동 패턴이 평균 순위와 RP proxy 에 어떤 차이를 만드는지 세 가설로 검정한다.
          각 가설은 정의 → 그룹별 결과 → 통계 검정 (Mann-Whitney U + Cliff's δ) → 결론.
        </p>
      </header>

      <div className="space-y-8">
        <HypothesisCard
          result={h1}
          definition={
            <div className="space-y-2 text-sm">
              <p>
                핫존을 “많은 팀이 실제로 내린 위치”로 정의 (맵 중심이 아님).
                Gaussian density 의 상위 25% 가 hotzone seed.
              </p>
              <FormulaBlock>{`hotzone seed =
  매치별 착륙 위치 Gaussian density 상위 25%

distance_to_hotzone_m =
  팀의 착륙 후 300초 평균 위치에서
  가장 가까운 hotzone seed 까지의 거리

hotzone =     distance_to_hotzone_m 하위 25%
outer_farm =  distance_to_hotzone_m 상위 25%`}</FormulaBlock>
            </div>
          }
        />

        <HypothesisCard
          result={h2}
          definition={
            <div className="space-y-2 text-sm">
              <p>교전 참여율은 누적 damage 가 아니라 <strong>damage event 횟수/프레임</strong> 으로 계산.</p>
              <FormulaBlock>{`combat_st_participation_rate =
  착륙 후 생존 10초 프레임 중
  팀이 damage event 의 attacker 또는 victim 으로
  등장한 프레임 비율`}</FormulaBlock>
            </div>
          }
        />

        <HypothesisCard
          result={h3}
          definition={
            <div className="space-y-2 text-sm">
              <p>
                초기 H3 는 “전체 생존기간 movement_per_min 하위 25% = sedentary” 였으나,
                조기 탈락팀 오염과 후반 zone rotation 효과를 제거하기 위해
                <strong> 착륙 후 동일 600초 고정 창</strong> 으로 다시 정의.
              </p>
              <FormulaBlock>{`H3 분석 대상 =
  착륙 후 600초 고정 창을 완주한 팀

fixed_window_movement_per_min =
  착륙 후 600초 동안의 팀 중심 이동거리 / 10분

fixed_sedentary = 위 분석 대상에서 하위 25%
fixed_mobile    = 위 분석 대상에서 상위 25%`}</FormulaBlock>
            </div>
          }
        />

        <Section title="H3 페이즈별 재분석" kicker="§4.4">
          <p className="max-w-prose text-sm text-muted">
            phase 별로 다시 계산해도 “존버 우세” 가설은 지지되지 않는다. 해당 phase 를 끝까지 생존한
            팀만 비교한 <strong>completed_phase_only</strong> 코호트 결과:
          </p>
          <DataTable columns={phaseCols} rows={phaseGrouped} rowKey={(r) => String(r.phase)} />
          <Callout variant="key">
            phase 1–3 에서는 mobile 팀의 최종 순위가 뚜렷하게 좋고, phase 4–6 에서는 이미 살아남은
            팀만 남아 순위 차이는 작아진다. 그러나 <strong>RP proxy 는 거의 모든 phase 에서 mobile
            팀이 더 높다</strong>.
          </Callout>
        </Section>

        <Section title="세 가설을 종합하면" kicker="§9 함의">
          <div className="space-y-2">
            <Callout label="H1 함의">
              핫존은 킬 기대값을 높이지만, 조기 탈락 위험이 더 커서 평균 순위와 RP proxy 에서는
              외곽 파밍이 유리하다. 핫존 평균 킬 4.04 vs 외곽 3.05, 평균 순위 16.68 vs 12.19.
            </Callout>
            <Callout label="H2 함의">
              교전 참여율이 높기만 한 팀은 평균 순위 19.60 / RP proxy -0.66. 좋은 전략은
              “많이 싸우는 것” 이 아니라 <strong>“살아남을 수 있는 교전만 선택하고, 그 교전을 점수로
              전환하는 것”</strong>.
            </Callout>
            <Callout label="H3 함의">
              저이동 자체가 항상 나쁘다고 단정할 수는 없다. 다만 <strong>동일한 초반-중반 관측 창에서
              고이동 / 고전환 팀이 순위와 점수 proxy 에서 더 유리</strong>하게 나타난다.
            </Callout>
          </div>
        </Section>
      </div>
    </div>
  );
}
