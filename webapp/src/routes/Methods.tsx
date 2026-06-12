import { useMethods } from "../lib/data";
import Section from "../components/Section";
import Callout from "../components/Callout";
import FormulaBlock from "../components/FormulaBlock";
import DataTable, { Column } from "../components/DataTable";

export default function Methods() {
  const m = useMethods();
  if (m.isLoading || !m.data) return <div className="text-muted">불러오는 중…</div>;
  const data = m.data;

  const sourceCols: Column<typeof data.data_sources[number]>[] = [
    { key: "source", header: "원천", cell: (r) => r.source, align: "left" },
    {
      key: "features",
      header: "사용 피쳐",
      cell: (r) => (
        <div className="flex flex-wrap gap-1">
          {r.features.map((f) => (
            <span key={f} className="jet rounded bg-cream px-1.5 py-0.5 text-[10px]">
              {f}
            </span>
          ))}
        </div>
      ),
    },
    { key: "role", header: "그래프에서의 역할", cell: (r) => r.role, align: "left" },
  ];

  const windowCols: Column<typeof data.graph_windows[number]>[] = [
    { key: "w", header: "그래프", cell: (r) => <span className="jet">{r.window}</span>, align: "left" },
    { key: "d", header: "정의", cell: (r) => r.definition, align: "left" },
    { key: "p", header: "용도", cell: (r) => r.purpose, align: "left" },
  ];

  const featCols: Column<typeof data.feature_groups[number]>[] = [
    { key: "g", header: "묶음", cell: (r) => r.group, align: "left" },
    {
      key: "f",
      header: "피쳐",
      cell: (r) => (
        <div className="flex flex-wrap gap-1">
          {r.features.map((f) => (
            <span key={f} className="jet rounded bg-cream px-1.5 py-0.5 text-[10px]">
              {f}
            </span>
          ))}
        </div>
      ),
    },
    { key: "u", header: "사용 위치", cell: (r) => r.usage, align: "left" },
  ];

  const limCols: Column<typeof data.limitations[number]>[] = [
    { key: "l", header: "한계", cell: (r) => r.limitation, align: "left" },
    { key: "d", header: "설명", cell: (r) => r.detail, align: "left" },
    { key: "n", header: "다음 보완", cell: (r) => r.next_step, align: "left" },
  ];

  return (
    <div className="space-y-10">
      <header className="space-y-2">
        <div className="jet text-[11px] uppercase tracking-wider text-accent">
          §2 · §10 · methods
        </div>
        <h1 className="news text-3xl font-semibold leading-tight">데이터와 방법</h1>
        <p className="max-w-prose text-sm text-muted">
          어떤 원천 데이터를 어떻게 그래프로 변환했는지, 어떤 피쳐를 분석에 사용했는지,
          RP proxy 는 어떻게 정의했는지, 그리고 본 분석의 한계는 무엇인지 정리한다.
        </p>
      </header>

      <Section title="데이터 원천" kicker="§2.2">
        <DataTable columns={sourceCols} rows={data.data_sources} rowKey={(r) => r.source} />
      </Section>

      <Section title="그래프 정의" kicker="§2.3">
        <FormulaBlock>{`V = 해당 매치에 참가한 팀 집합

E = 팀 A와 팀 B 사이에 damage event가 1회 이상 관측된 관계

weight(A, B) = 팀 A와 팀 B 사이의 누적 damage
events(A, B) = 팀 A와 팀 B 사이의 damage event 횟수

direction = ${data.graph_definition.direction}`}</FormulaBlock>
        <Callout label="제외">
          <ul className="list-disc pl-5">
            {data.graph_definition.excluded.map((e) => (
              <li key={e}>{e}</li>
            ))}
          </ul>
        </Callout>
        <DataTable columns={windowCols} rows={data.graph_windows} rowKey={(r) => r.window} />
      </Section>

      <Section title="사용 피쳐" kicker="§2.6">
        <p className="max-w-prose text-sm text-muted">
          본 분석에 실제로 계산하고 검정/해석에 사용한 피쳐만 정리. 별도 생성하지 않은 후보
          피쳐는 제외.
        </p>
        <DataTable columns={featCols} rows={data.feature_groups} rowKey={(r) => r.group} />
      </Section>

      <Section title="RP proxy 공식" kicker="§2.5">
        <p className="max-w-prose text-sm text-muted">{data.rp_proxy_formula.note}</p>
        <FormulaBlock caption={
          <>
            출처:{" "}
            <a href={data.rp_proxy_formula.source} target="_blank" rel="noreferrer">
              PUBG Support 경쟁전 FAQ
            </a>
          </>
        }>
          {data.rp_proxy_formula.formula}
        </FormulaBlock>
      </Section>

      <Section title="한계와 다음 분석" kicker="§10">
        <DataTable columns={limCols} rows={data.limitations} rowKey={(r) => r.limitation} />
      </Section>
    </div>
  );
}
