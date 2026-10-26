import { Link } from "react-router-dom";
import type { MatchIndexEntry, StrategyType } from "../lib/types";
import { fmtDec, fmtDuration } from "../lib/format";
import { STRATEGY_COLORS } from "./StrategyBadge";

interface Props {
  match: MatchIndexEntry;
}

const TYPES: StrategyType[] = ["Hunter", "Traveler", "Sniper", "Camper"];

export default function MatchCard({ match }: Props) {
  const total = match.type_counts
    ? Object.values(match.type_counts).reduce((s, n) => s + n, 0)
    : 0;

  return (
    <Link
      to={`/explorer/${match.match_id}`}
      className="group block rounded-lg border border-rule bg-paper p-4 transition-colors hover:border-accent hover:bg-cream/30 hover:no-underline"
    >
      <div className="flex items-baseline justify-between gap-2">
        <div className="jet truncate text-[11px] text-muted">
          {match.match_id.slice(0, 8)}
        </div>
        <div className="jet text-[10px] text-muted">
          {fmtDuration(match.duration_s)}
        </div>
      </div>

      <div className="mt-2 grid grid-cols-3 gap-2 text-xs">
        <div>
          <div className="jet text-[9px] uppercase text-muted">팀</div>
          <div className="numeric">{match.n_teams}</div>
        </div>
        <div>
          <div className="jet text-[9px] uppercase text-muted">엣지</div>
          <div className="numeric">{match.n_edges ?? "—"}</div>
        </div>
        <div>
          <div className="jet text-[9px] uppercase text-muted">Q</div>
          <div className="numeric">{fmtDec(match.modularity_q, 2)}</div>
        </div>
      </div>

      {/* 4유형 분포 막대 */}
      {match.type_counts && total > 0 ? (
        <div className="mt-3">
          <div className="jet mb-1 text-[9px] uppercase text-muted">4유형 분포</div>
          <div className="flex h-2 overflow-hidden rounded">
            {TYPES.map((t) => {
              const c = match.type_counts![t] ?? 0;
              const w = (c / total) * 100;
              if (w === 0) return null;
              return (
                <div
                  key={t}
                  style={{ width: `${w}%`, background: STRATEGY_COLORS[t] }}
                  title={`${t}: ${c}팀`}
                />
              );
            })}
          </div>
          <div className="jet mt-1 flex flex-wrap gap-x-2 text-[9px] text-muted">
            {TYPES.map((t) => (
              <span key={t}>
                {t.slice(0, 3).toUpperCase()} {match.type_counts![t] ?? 0}
              </span>
            ))}
          </div>
        </div>
      ) : null}

      <div className="mt-3 jet text-[10px] text-accent opacity-0 transition-opacity group-hover:opacity-100">
        그래프 열기 →
      </div>
    </Link>
  );
}
