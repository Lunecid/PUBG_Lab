import { useState, useMemo } from "react";
import { useMatchIndex } from "../lib/data";
import Section from "../components/Section";
import MatchCard from "../components/MatchCard";
import { fmtDec } from "../lib/format";
import type { MatchIndexEntry } from "../lib/types";

type SortKey = "modularity" | "duration" | "n_teams" | "n_edges";
const SORT_LABEL: Record<SortKey, string> = {
  modularity: "Modularity Q",
  duration: "매치 길이",
  n_teams: "팀 수",
  n_edges: "엣지 수",
};

export default function Explorer() {
  const matches = useMatchIndex();
  const [sort, setSort] = useState<SortKey>("modularity");
  const [desc, setDesc] = useState(true);

  const sorted = useMemo(() => {
    if (!matches.data) return [];
    const arr = [...matches.data];
    arr.sort((a, b) => {
      const av = sortValue(a, sort);
      const bv = sortValue(b, sort);
      return desc ? bv - av : av - bv;
    });
    return arr;
  }, [matches.data, sort, desc]);

  if (matches.isLoading || !matches.data)
    return <div className="text-muted">매치 목록 로딩 중…</div>;

  const stats = useMemo(() => {
    const arr = matches.data!;
    const dur = arr.map((m) => m.duration_s);
    const mod = arr.map((m) => m.modularity_q ?? 0);
    return {
      n: arr.length,
      durMin: Math.min(...dur),
      durMax: Math.max(...dur),
      modMin: Math.min(...mod),
      modMax: Math.max(...mod),
    };
  }, [matches.data]);

  return (
    <div className="space-y-8">
      <header className="space-y-2">
        <div className="jet text-[11px] uppercase tracking-wider text-accent">
          interactive · 30 sample matches
        </div>
        <h1 className="news text-3xl font-semibold leading-tight">매치 탐색</h1>
        <p className="max-w-prose text-sm text-muted">
          n=1,000 매치 중 modularity 와 매치 길이 분위로 균등 샘플링한 30개. 카드 클릭 시 해당
          매치의 움직이는 그래프와 팀별 상세 패널로 이동.
        </p>
      </header>

      <Section title={`${stats.n}개 매치`} kicker="filter / sort">
        <div className="flex flex-wrap items-center gap-3 rounded border border-rule bg-paper p-3 text-sm">
          <span className="jet text-[10px] uppercase text-muted">정렬</span>
          {(Object.keys(SORT_LABEL) as SortKey[]).map((k) => (
            <button
              key={k}
              onClick={() => {
                if (sort === k) setDesc(!desc);
                else {
                  setSort(k);
                  setDesc(true);
                }
              }}
              className={
                "rounded px-2 py-1 transition-colors " +
                (sort === k
                  ? "bg-cream text-ink"
                  : "text-muted hover:bg-cream/50")
              }
            >
              {SORT_LABEL[k]} {sort === k ? (desc ? "↓" : "↑") : ""}
            </button>
          ))}
          <div className="ml-auto jet text-[10px] text-muted">
            Q {fmtDec(stats.modMin, 2)}–{fmtDec(stats.modMax, 2)} ·{" "}
            {Math.round(stats.durMin)}–{Math.round(stats.durMax)}s
          </div>
        </div>

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {sorted.map((m) => (
            <MatchCard key={m.match_id} match={m} />
          ))}
        </div>
      </Section>
    </div>
  );
}

function sortValue(m: MatchIndexEntry, key: SortKey): number {
  switch (key) {
    case "modularity":
      return m.modularity_q ?? 0;
    case "duration":
      return m.duration_s ?? 0;
    case "n_teams":
      return m.n_teams ?? 0;
    case "n_edges":
      return m.n_edges ?? 0;
  }
}
