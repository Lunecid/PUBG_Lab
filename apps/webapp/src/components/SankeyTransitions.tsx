/**
 * 단순 sankey-스타일 페이즈 전환 다이어그램.
 * 두 phase 사이의 4유형 흐름 시각화.
 * 자체 SVG (베지어 곡선).
 */
import type { StrategyType } from "../lib/types";

const TYPES: StrategyType[] = ["Hunter", "Traveler", "Sniper", "Camper"];
const COLOR: Record<StrategyType, string> = {
  Hunter: "#b02a2a",
  Sniper: "#d97706",
  Traveler: "#3a4a5a",
  Camper: "#6b6558",
};

interface Props {
  /** matrix[from_type][to_type] = count */
  matrix: Record<string, number>;
  fromLabel: string;
  toLabel: string;
  /** 너무 작은 흐름은 숨김 */
  minShare?: number;
  width?: number;
  height?: number;
}

export default function SankeyTransitions({
  matrix,
  fromLabel,
  toLabel,
  minShare = 0.02,
  width = 480,
  height = 320,
}: Props) {
  // 각 유형의 from/to 합 계산
  const fromTotals: Record<StrategyType, number> = {
    Hunter: 0, Sniper: 0, Traveler: 0, Camper: 0,
  };
  const toTotals: Record<StrategyType, number> = {
    Hunter: 0, Sniper: 0, Traveler: 0, Camper: 0,
  };
  let total = 0;
  for (const key in matrix) {
    const [a, b] = key.split("->");
    if (!TYPES.includes(a as StrategyType) || !TYPES.includes(b as StrategyType)) continue;
    const n = matrix[key];
    fromTotals[a as StrategyType] += n;
    toTotals[b as StrategyType] += n;
    total += n;
  }
  if (total === 0) return null;

  const padTop = 24;
  const padBot = 16;
  const innerH = height - padTop - padBot;
  const colW = 40;
  const gap = 6;

  // 좌측 (from) 위치: 비율로 누적 stack
  let fy = padTop;
  const fromBox: Record<StrategyType, { y: number; h: number; cx: number; cy: number }> = {} as never;
  for (const t of TYPES) {
    const h = (fromTotals[t] / total) * (innerH - gap * (TYPES.length - 1));
    fromBox[t] = { y: fy, h, cx: colW + colW / 2, cy: fy + h / 2 };
    fy += h + gap;
  }
  let ty = padTop;
  const toBox: Record<StrategyType, { y: number; h: number; cx: number; cy: number }> = {} as never;
  for (const t of TYPES) {
    const h = (toTotals[t] / total) * (innerH - gap * (TYPES.length - 1));
    toBox[t] = { y: ty, h, cx: width - colW - colW / 2, cy: ty + h / 2 };
    ty += h + gap;
  }

  // 흐름: 각 (from, to) → 곡선
  // 각 from 안에서 다시 누적 위치를 계산해 segment 분할
  const fromCursor: Record<StrategyType, number> = {
    Hunter: 0, Sniper: 0, Traveler: 0, Camper: 0,
  };
  const toCursor: Record<StrategyType, number> = {
    Hunter: 0, Sniper: 0, Traveler: 0, Camper: 0,
  };
  type Flow = { from: StrategyType; to: StrategyType; n: number; share: number };
  const flows: Flow[] = [];
  for (const a of TYPES) {
    for (const b of TYPES) {
      const n = matrix[`${a}->${b}`] ?? 0;
      if (n === 0) continue;
      const share = n / total;
      if (share < minShare) continue;
      flows.push({ from: a, to: b, n, share });
    }
  }
  // 굵은 흐름 위로
  flows.sort((a, b) => b.n - a.n);

  return (
    <div>
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" role="img">
        {/* col labels */}
        <text x={colW + colW / 2} y={padTop - 8} textAnchor="middle" fontSize={11} fontFamily="JetBrains Mono, monospace" fill="#6b6558">
          {fromLabel}
        </text>
        <text x={width - colW - colW / 2} y={padTop - 8} textAnchor="middle" fontSize={11} fontFamily="JetBrains Mono, monospace" fill="#6b6558">
          {toLabel}
        </text>

        {/* flows (밑에 깔림) */}
        {flows.map((f, i) => {
          const fb = fromBox[f.from];
          const tb = toBox[f.to];
          // segment 위치
          const sH = (f.n / Math.max(1, fromTotals[f.from])) * fb.h;
          const tH = (f.n / Math.max(1, toTotals[f.to])) * tb.h;
          const sy = fb.y + fromCursor[f.from] + sH / 2;
          const ty2 = tb.y + toCursor[f.to] + tH / 2;
          fromCursor[f.from] += sH;
          toCursor[f.to] += tH;
          const x0 = fb.cx + colW / 2;
          const x1 = tb.cx - colW / 2;
          const cx0 = (x0 + x1) / 2;
          const path = `M ${x0} ${sy} C ${cx0} ${sy}, ${cx0} ${ty2}, ${x1} ${ty2}`;
          const color = COLOR[f.from];
          const sw = Math.max(2, (sH + tH) / 2);
          return (
            <path
              key={i}
              d={path}
              stroke={color}
              strokeOpacity={0.35}
              strokeWidth={sw}
              fill="none"
              strokeLinecap="round"
            >
              <title>{f.from} → {f.to} : {f.n.toLocaleString()}팀 ({(f.share * 100).toFixed(1)}%)</title>
            </path>
          );
        })}

        {/* from boxes */}
        {TYPES.map((t) => {
          const b = fromBox[t];
          if (b.h < 1) return null;
          return (
            <g key={`fb-${t}`}>
              <rect x={colW} y={b.y} width={colW} height={b.h} fill={COLOR[t]} />
              <text x={colW - 6} y={b.cy} textAnchor="end" dominantBaseline="middle" fontSize={11} fontFamily="JetBrains Mono, monospace" fill="#1a1a1a" fontWeight={500}>
                {t}
              </text>
              <text x={colW + colW + 4} y={b.cy} textAnchor="start" dominantBaseline="middle" fontSize={9.5} fontFamily="IBM Plex Mono, monospace" fill="#fefcf7">
                {fromTotals[t].toLocaleString()}
              </text>
            </g>
          );
        })}
        {/* to boxes */}
        {TYPES.map((t) => {
          const b = toBox[t];
          if (b.h < 1) return null;
          return (
            <g key={`tb-${t}`}>
              <rect x={width - colW - colW} y={b.y} width={colW} height={b.h} fill={COLOR[t]} />
              <text x={width - colW + 6} y={b.cy} textAnchor="start" dominantBaseline="middle" fontSize={11} fontFamily="JetBrains Mono, monospace" fill="#1a1a1a" fontWeight={500}>
                {t}
              </text>
              <text x={width - colW - colW - 4} y={b.cy} textAnchor="end" dominantBaseline="middle" fontSize={9.5} fontFamily="IBM Plex Mono, monospace" fill="#fefcf7">
                {toTotals[t].toLocaleString()}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
