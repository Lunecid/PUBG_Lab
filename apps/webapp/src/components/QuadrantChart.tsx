/**
 * FightScore × ActiveScore 2×2 사분면 산점도.
 * 4유형 분포를 시각화.
 */
import type { StrategyType } from "../lib/types";

interface Point {
  x: number; // FightScore
  y: number; // ActiveScore
  type: StrategyType;
}

interface Props {
  /** 각 유형의 평균 위치 (또는 모든 팀 산점) */
  points: Point[];
  /** 사분면 라벨 위치별 카운트 (선택) */
  labels?: { Hunter?: string; Sniper?: string; Traveler?: string; Camper?: string };
  size?: number;
}

const COLOR: Record<StrategyType, string> = {
  Hunter: "#b02a2a",
  Sniper: "#d97706",
  Traveler: "#3a4a5a",
  Camper: "#6b6558",
};

const QUADRANT_FILL: Record<StrategyType, string> = {
  Hunter: "#b02a2a",
  Sniper: "#d97706",
  Traveler: "#3a4a5a",
  Camper: "#6b6558",
};

export default function QuadrantChart({ points, labels, size = 480 }: Props) {
  const pad = 40;
  const inner = size - pad * 2;
  // 도메인: -1.5 ~ 1.5 z-score 범위
  const dom = 1.5;
  const x = (v: number) => pad + ((v + dom) / (2 * dom)) * inner;
  const y = (v: number) => pad + ((dom - v) / (2 * dom)) * inner;

  return (
    <div>
      <svg viewBox={`0 0 ${size} ${size}`} width="100%" role="img">
        {/* 사분면 배경 */}
        <rect x={x(0)} y={pad} width={x(dom) - x(0)} height={y(0) - pad} fill={QUADRANT_FILL.Hunter} fillOpacity={0.07} />
        <rect x={x(0)} y={y(0)} width={x(dom) - x(0)} height={pad + inner - y(0)} fill={QUADRANT_FILL.Sniper} fillOpacity={0.07} />
        <rect x={pad} y={pad} width={x(0) - pad} height={y(0) - pad} fill={QUADRANT_FILL.Traveler} fillOpacity={0.07} />
        <rect x={pad} y={y(0)} width={x(0) - pad} height={pad + inner - y(0)} fill={QUADRANT_FILL.Camper} fillOpacity={0.07} />

        {/* 축 */}
        <line x1={pad} x2={pad + inner} y1={y(0)} y2={y(0)} stroke="#d8d1c2" strokeWidth={1} />
        <line x1={x(0)} x2={x(0)} y1={pad} y2={pad + inner} stroke="#d8d1c2" strokeWidth={1} />

        {/* 사분면 라벨 */}
        <g fontFamily="JetBrains Mono, monospace" fontSize={11} textAnchor="middle">
          <text x={(x(0) + x(dom)) / 2} y={pad + 14} fill={QUADRANT_FILL.Hunter} fontWeight={600}>
            Hunter {labels?.Hunter ?? ""}
          </text>
          <text x={(x(0) + x(dom)) / 2} y={pad + inner - 4} fill={QUADRANT_FILL.Sniper} fontWeight={600}>
            Sniper {labels?.Sniper ?? ""}
          </text>
          <text x={(pad + x(0)) / 2} y={pad + 14} fill={QUADRANT_FILL.Traveler} fontWeight={600}>
            Traveler {labels?.Traveler ?? ""}
          </text>
          <text x={(pad + x(0)) / 2} y={pad + inner - 4} fill={QUADRANT_FILL.Camper} fontWeight={600}>
            Camper {labels?.Camper ?? ""}
          </text>
        </g>

        {/* 축 라벨 */}
        <text
          x={pad + inner + 4}
          y={y(0) - 4}
          fontSize={10}
          fontFamily="JetBrains Mono, monospace"
          fill="#6b6558"
          textAnchor="end"
        >
          FightScore →
        </text>
        <text
          x={x(0) + 4}
          y={pad + 10}
          fontSize={10}
          fontFamily="JetBrains Mono, monospace"
          fill="#6b6558"
          textAnchor="start"
        >
          ActiveScore ↑
        </text>

        {/* 0 선 라벨 */}
        <text x={x(0)} y={pad + inner + 14} fontSize={10} fontFamily="IBM Plex Mono, monospace" fill="#6b6558" textAnchor="middle">0</text>
        <text x={pad - 6} y={y(0)} fontSize={10} fontFamily="IBM Plex Mono, monospace" fill="#6b6558" textAnchor="end" dominantBaseline="middle">0</text>

        {/* 점 */}
        {points.map((p, i) => (
          <circle
            key={i}
            cx={x(p.x)}
            cy={y(p.y)}
            r={9}
            fill={COLOR[p.type]}
            stroke="#fefcf7"
            strokeWidth={2}
          >
            <title>
              {p.type} · FightScore={p.x.toFixed(3)} · ActiveScore={p.y.toFixed(3)}
            </title>
          </circle>
        ))}
      </svg>
    </div>
  );
}
