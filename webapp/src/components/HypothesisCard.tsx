import { ReactNode } from "react";
import type { HypothesisResult } from "../lib/types";
import Callout from "./Callout";
import DataTable, { Column } from "./DataTable";
import { fmtDec, fmtP, fmtSigned } from "../lib/format";

interface Props {
  result: HypothesisResult;
  definition: ReactNode;
}

export default function HypothesisCard({ result, definition }: Props) {
  const verdictColor =
    result.verdict.includes("better") || result.verdict === "outer_farm_better"
      ? "good"
      : result.verdict === "rejected"
      ? "bad"
      : "warn";

  // 그룹 결과 표
  const groupCols: Column<typeof result.groups[number]>[] = [
    { key: "label", header: "그룹", cell: (r) => r.label, align: "left" },
    { key: "n", header: "n", cell: (r) => (r.n != null ? r.n.toLocaleString() : "—"), numeric: true, align: "right" },
    { key: "rank", header: "평균 순위", cell: (r) => fmtDec(r.mean_rank, 2), numeric: true, align: "right" },
    { key: "rp", header: "RP proxy", cell: (r) => fmtSigned(r.ranked_rp_proxy, 2), numeric: true, align: "right" },
    { key: "kills", header: "평균 킬", cell: (r) => fmtDec(r.team_kills_mean, 2), numeric: true, align: "right" },
    { key: "surv", header: "생존(s)", cell: (r) => fmtDec(r.survival_seconds_mean, 0), numeric: true, align: "right" },
  ];

  // tests 헤더 동적 라벨
  const lowLabel = result.tests[0]?.low_label ?? "low";
  const highLabel = result.tests[0]?.high_label ?? "high";

  // 검정 결과 표
  const testCols: Column<typeof result.tests[number]>[] = [
    { key: "target", header: "검정 대상", cell: (r) => r.target_label, align: "left" },
    { key: "low", header: lowLabel, cell: (r) => fmtDec(r.low_mean_target, 3), numeric: true, align: "right" },
    { key: "high", header: highLabel, cell: (r) => fmtDec(r.high_mean_target, 3), numeric: true, align: "right" },
    { key: "delta", header: "Δ", cell: (r) => fmtSigned(r.delta, 3), numeric: true, align: "right" },
    { key: "p", header: "p", cell: (r) => fmtP(r.mw_p), numeric: true, align: "right" },
    { key: "cliffs", header: "Cliff's δ", cell: (r) => fmtDec(r.cliffs_delta, 3), numeric: true, align: "right" },
    {
      key: "sig",
      header: "지지",
      cell: (r) => (r.statistically_supported ? "✓" : "—"),
      align: "center",
      className: "text-good",
    },
  ];

  return (
    <article className="space-y-4 rounded-lg border border-rule bg-paper p-5">
      <header className="flex items-baseline justify-between gap-3 border-b border-rule pb-2">
        <h3 className="news text-lg font-semibold leading-tight">{result.title}</h3>
        <span
          className={`jet rounded px-2 py-0.5 text-[11px] font-medium uppercase tracking-wider ${
            verdictColor === "good"
              ? "bg-good text-paper"
              : verdictColor === "bad"
              ? "bg-bad text-paper"
              : "bg-warn text-paper"
          }`}
        >
          {result.verdict_label}
        </span>
      </header>

      <Callout label="정의">{definition}</Callout>

      <div>
        <h4 className="mb-2 jet text-[10px] uppercase tracking-wider text-muted">그룹별 결과</h4>
        <DataTable columns={groupCols} rows={result.groups} rowKey={(r) => r.group} />
      </div>

      {result.tests.length > 0 ? (
        <div>
          <h4 className="mb-2 jet text-[10px] uppercase tracking-wider text-muted">통계 검정</h4>
          <DataTable columns={testCols} rows={result.tests} rowKey={(r, i) => `${r.target}-${i}`} />
        </div>
      ) : null}

      <Callout variant="key" label="결론">
        {result.conclusion}
      </Callout>
    </article>
  );
}
