/**
 * phase × type 히트맵.
 * 행: phase (1~6), 열: 4유형, 셀: metric value (color scale).
 */
import type { StrategyType } from "../lib/types";

interface Cell {
  row: number; // phase
  col: StrategyType;
  value: number;
}

interface Props {
  cells: Cell[];
  rows: number[]; // [1,2,3,4,5,6]
  cols: StrategyType[];
  /** value -> color */
  scale: { min: number; max: number; lowColor?: string; highColor?: string; midZero?: boolean };
  cellSize?: number;
  valueDigits?: number;
  caption?: string;
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}
function hex(c: string) {
  return [
    parseInt(c.slice(1, 3), 16),
    parseInt(c.slice(3, 5), 16),
    parseInt(c.slice(5, 7), 16),
  ];
}
function rgb(arr: number[]) {
  return `rgb(${Math.round(arr[0])},${Math.round(arr[1])},${Math.round(arr[2])})`;
}

function colorFor(v: number, scale: Props["scale"]): string {
  const lo = hex(scale.lowColor ?? "#fefcf7");
  const hi = hex(scale.highColor ?? "#b02a2a");
  if (scale.midZero) {
    // 0 기준 양/음 색 분리: 음수는 파랑, 양수는 빨강
    const neg = hex("#1d4ed8");
    const pos = hex(scale.highColor ?? "#b02a2a");
    if (v >= 0) {
      const t = Math.min(1, v / Math.max(1e-6, scale.max));
      return rgb([
        lerp(lo[0], pos[0], t),
        lerp(lo[1], pos[1], t),
        lerp(lo[2], pos[2], t),
      ]);
    } else {
      const t = Math.min(1, -v / Math.max(1e-6, -scale.min));
      return rgb([
        lerp(lo[0], neg[0], t),
        lerp(lo[1], neg[1], t),
        lerp(lo[2], neg[2], t),
      ]);
    }
  }
  const t = Math.max(
    0,
    Math.min(1, (v - scale.min) / Math.max(1e-6, scale.max - scale.min)),
  );
  return rgb([lerp(lo[0], hi[0], t), lerp(lo[1], hi[1], t), lerp(lo[2], hi[2], t)]);
}

export default function Heatmap({
  cells,
  rows,
  cols,
  scale,
  cellSize = 56,
  valueDigits = 2,
  caption,
}: Props) {
  const rowLabelW = 70;
  const colLabelH = 28;
  const w = rowLabelW + cols.length * cellSize + 16;
  const h = colLabelH + rows.length * cellSize + 16;
  const cellByKey = new Map<string, Cell>();
  for (const c of cells) cellByKey.set(`${c.row}|${c.col}`, c);

  return (
    <div>
      <svg viewBox={`0 0 ${w} ${h}`} width="100%" role="img">
        {/* col headers */}
        {cols.map((c, i) => (
          <text
            key={`ch${i}`}
            x={rowLabelW + i * cellSize + cellSize / 2}
            y={colLabelH - 8}
            textAnchor="middle"
            dominantBaseline="middle"
            fontSize={11}
            fontFamily="JetBrains Mono, monospace"
            fontWeight={600}
            fill="#1a1a1a"
          >
            {c}
          </text>
        ))}
        {/* row headers */}
        {rows.map((r, i) => (
          <text
            key={`rh${i}`}
            x={rowLabelW - 8}
            y={colLabelH + i * cellSize + cellSize / 2}
            textAnchor="end"
            dominantBaseline="middle"
            fontSize={11}
            fontFamily="IBM Plex Mono, monospace"
            fill="#6b6558"
          >
            phase {r}
          </text>
        ))}
        {/* cells */}
        {rows.map((r, ri) =>
          cols.map((c, ci) => {
            const cell = cellByKey.get(`${r}|${c}`);
            const x = rowLabelW + ci * cellSize;
            const y = colLabelH + ri * cellSize;
            if (!cell) {
              return (
                <rect key={`x${ri}-${ci}`} x={x} y={y} width={cellSize - 2} height={cellSize - 2} fill="#f3eddd" />
              );
            }
            const fill = colorFor(cell.value, scale);
            // 텍스트 색상: 어두운 배경엔 흰색, 밝은 배경엔 검정
            const [rr, gg, bb] = hex(fill.startsWith("rgb")
              ? `#${fill.match(/\d+/g)!.map((n) => Number(n).toString(16).padStart(2, "0")).join("")}`
              : fill);
            const lum = (0.299 * rr + 0.587 * gg + 0.114 * bb) / 255;
            const txt = lum > 0.55 ? "#1a1a1a" : "#fefcf7";
            return (
              <g key={`c${ri}-${ci}`}>
                <rect x={x} y={y} width={cellSize - 2} height={cellSize - 2} fill={fill} stroke="#d8d1c2" strokeWidth={0.5} />
                <text
                  x={x + (cellSize - 2) / 2}
                  y={y + (cellSize - 2) / 2}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fontSize={11}
                  fontFamily="IBM Plex Mono, monospace"
                  fill={txt}
                  style={{ fontVariantNumeric: "tabular-nums" }}
                >
                  {cell.value.toFixed(valueDigits)}
                </text>
              </g>
            );
          }),
        )}
      </svg>
      {caption ? (
        <div className="mt-2 text-center text-xs text-muted">{caption}</div>
      ) : null}
    </div>
  );
}
