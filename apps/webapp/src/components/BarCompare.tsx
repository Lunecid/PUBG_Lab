/**
 * 가로 막대 비교 차트 (SVG).
 * 두 그룹 (또는 N 그룹) 의 단일 metric 비교에 사용.
 */
interface Item {
  label: string;
  value: number;
  color?: string;
  /** 보조 라벨 (예: n=...) */
  meta?: string;
}

interface Props {
  items: Item[];
  /** 도메인 (없으면 자동 계산: 0 ~ max*1.1) */
  max?: number;
  min?: number;
  width?: number;
  rowHeight?: number;
  unit?: string;
  /** 작은 값일수록 좋다고 표시할 색 토글 (색상 그대로 유지) */
  valueDigits?: number;
  /** label 영역 폭 (px) */
  labelWidth?: number;
}

export default function BarCompare({
  items,
  max,
  min = 0,
  width = 560,
  rowHeight = 28,
  unit = "",
  valueDigits = 2,
  labelWidth = 120,
}: Props) {
  if (items.length === 0) return null;
  const vmax = max ?? Math.max(...items.map((i) => i.value)) * 1.08;
  const vmin = min;
  const span = Math.max(1e-6, vmax - vmin);
  const chartW = width - labelWidth - 80; // 80 = right value label
  const h = items.length * rowHeight + 12;

  const x = (v: number) => labelWidth + ((v - vmin) / span) * chartW;
  const baselineX = x(vmin);

  return (
    <svg
      viewBox={`0 0 ${width} ${h}`}
      width="100%"
      style={{ display: "block" }}
      role="img"
    >
      {/* baseline */}
      <line
        x1={baselineX}
        x2={baselineX}
        y1={0}
        y2={h - 12}
        stroke="#d8d1c2"
        strokeWidth={1}
      />
      {items.map((it, i) => {
        const yTop = i * rowHeight + 4;
        const yMid = yTop + rowHeight / 2;
        const barX1 = x(vmin);
        const barX2 = x(it.value);
        const barW = Math.max(1, barX2 - barX1);
        const color = it.color ?? "#3a4a5a";
        return (
          <g key={`${it.label}-${i}`}>
            <text
              x={labelWidth - 6}
              y={yMid}
              textAnchor="end"
              dominantBaseline="middle"
              fontSize={12}
              fill="#1a1a1a"
              fontFamily="Inter, sans-serif"
            >
              {it.label}
            </text>
            {it.meta ? (
              <text
                x={labelWidth - 6}
                y={yMid + 13}
                textAnchor="end"
                dominantBaseline="middle"
                fontSize={9.5}
                fill="#6b6558"
                fontFamily="IBM Plex Mono, monospace"
              >
                {it.meta}
              </text>
            ) : null}
            <rect
              x={barX1}
              y={yMid - 7}
              width={barW}
              height={14}
              fill={color}
              opacity={0.85}
              rx={2}
            />
            <text
              x={barX2 + 6}
              y={yMid}
              textAnchor="start"
              dominantBaseline="middle"
              fontSize={12}
              fill="#1a1a1a"
              fontFamily="IBM Plex Mono, monospace"
              style={{ fontVariantNumeric: "tabular-nums" }}
            >
              {it.value.toFixed(valueDigits)}
              {unit ? <tspan className="text-muted">{` ${unit}`}</tspan> : null}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
