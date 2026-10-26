// v4 spatial network — dynamic per-frame team state (community/position/archetype/movement)
import { useMemo, useRef } from "react";
import type {
  Archetype,
  MatchEdge,
  MatchFrame,
  MatchNode,
  MovementLabel,
  TeamEvent,
  TeamStateAtT,
} from "../lib/types";
import { convexHull, expandPolygon } from "../lib/convexHull";
import { usePanZoom } from "../lib/usePanZoom";
import MapBackground from "./MapBackground";

export type AnimColorBy = "archetype" | "movement" | "position" | "community" | "rank";

const ARC_COLOR: Record<Archetype, string> = {
  outer_farmer: "#15803d",
  combat_participant: "#b91c1c",
  neutral: "#a8a29e",
};
const MV_COLOR: Record<MovementLabel, string> = {
  sedentary: "#1d4ed8",
  mid: "#a8a29e",
  mobile: "#b91c1c",
};
const POS_COLOR: Record<string, string> = {
  core: "#b02a2a",
  mid: "#d97706",
  peripheral: "#1d4ed8",
  isolated: "#6b6558",
};
const COMM_PALETTE = [
  "#b02a2a", "#1f4e8c", "#2d5f3f", "#a05a00", "#6f4a88",
  "#8a6a00", "#007a8a", "#7a3a5a", "#3a6a3a", "#8a3a3a",
  "#5a5a7a", "#7a5a3a", "#3a5a7a", "#5a7a3a", "#7a3a7a",
];

function rankColor(rank: number | null, max: number): string {
  if (rank === null) return "#a8a29e";
  const t = Math.max(0, Math.min(1, (rank - 1) / Math.max(1, max - 1)));
  // 1 -> 진한 녹색, 마지막 -> 진한 빨강
  const r = Math.round(20 + t * 200);
  const g = Math.round(160 - t * 130);
  return `rgb(${r}, ${g}, 60)`;
}

/**
 * 노드 색상. 동적 ts 가 있으면 그 시점 라벨 사용 (community/position/archetype/movement),
 * 없으면 매치 종료 정적 라벨 fallback.
 */
function nodeColor(
  n: MatchNode,
  ts: TeamStateAtT | undefined,
  by: AnimColorBy,
  maxRank: number,
): string {
  if (by === "archetype") {
    const arc = (ts?.arc as Archetype | undefined) ?? n.archetype;
    return ARC_COLOR[arc] ?? "#6b6558";
  }
  if (by === "movement") {
    const mvm = (ts?.mvm as MovementLabel | undefined) ?? n.movement_label;
    return MV_COLOR[mvm] ?? "#6b6558";
  }
  if (by === "position") {
    const pos = ts?.pos ?? n.position;
    return POS_COLOR[pos] ?? "#6b6558";
  }
  if (by === "community") {
    const comm = ts?.comm ?? n.community;
    if (comm < 0) return "#a8a29e";
    return COMM_PALETTE[comm % COMM_PALETTE.length];
  }
  if (by === "rank") return rankColor(n.rank, maxRank);
  return "#6b6558";
}

interface Props {
  nodes: MatchNode[];
  edges?: MatchEdge[];
  events: TeamEvent[];
  frames: MatchFrame[];
  arenaM: number;
  /** 현재 시각 (초) */
  t: number;
  colorBy: AnimColorBy;
  /** 맵 배경 이미지 URL fallback chain (index.json.map_urls) */
  mapUrls?: string[];
  /** SVG 한 변 길이 (px) */
  size?: number;
  /** community 힐 표시 여부 */
  showHulls?: boolean;
}

interface Trail {
  t: number;
  x: number;
  y: number;
}

const TRAIL_WINDOW_S = 60;
const ZONE_NUM_STR = "1.2";

export default function AnimatedNetwork({
  nodes,
  events,
  frames,
  arenaM,
  t,
  colorBy,
  mapUrls,
  size = 540,
  showHulls = true,
}: Props) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const pz = usePanZoom(svgRef);

  const maxRank = useMemo(
    () => Math.max(1, ...nodes.map((n) => n.rank ?? 0)),
    [nodes],
  );

  const nodeByIdx = useMemo(() => {
    const m = new Map<number, MatchNode>();
    for (const n of nodes) m.set(n.id, n);
    return m;
  }, [nodes]);

  const trailsByTeam = useMemo(() => {
    const m = new Map<number, Trail[]>();
    for (const f of frames) {
      for (const ti of Object.keys(f.cx)) {
        const id = parseInt(ti, 10);
        const x = f.cx[ti];
        const y = f.cy[ti];
        if (typeof x !== "number" || typeof y !== "number") continue;
        if (!m.has(id)) m.set(id, []);
        m.get(id)!.push({ t: f.t, x, y });
      }
    }
    for (const arr of m.values()) arr.sort((a, b) => a.t - b.t);
    return m;
  }, [frames]);

  const edgeIndex = useMemo(() => {
    type Acc = { events: { t: number; dmg: number }[]; src: number; dst: number };
    const m = new Map<string, Acc>();
    for (const [s, d, te, dmg] of events) {
      const lo = Math.min(s, d);
      const hi = Math.max(s, d);
      const key = `${lo}-${hi}`;
      if (!m.has(key)) m.set(key, { events: [], src: lo, dst: hi });
      m.get(key)!.events.push({ t: te, dmg });
    }
    return m;
  }, [events]);

  const curFrame = useMemo(() => {
    let cur: MatchFrame | null = null;
    for (const f of frames) {
      if (f.t > t) break;
      cur = f;
    }
    return cur;
  }, [frames, t]);

  // 현재 시점의 동적 team_state (id string → TeamStateAtT)
  const curTeamState = curFrame?.ts ?? {};

  // 보간된 팀 위치
  const teamPositions = useMemo(() => {
    const aliveSet = new Set(curFrame?.alive ?? []);
    const out = new Map<
      number,
      { x: number; y: number; lastT: number; alive: boolean }
    >();
    for (const [id, trail] of trailsByTeam.entries()) {
      if (trail.length === 0) continue;
      let prev: Trail | null = null;
      let next: Trail | null = null;
      for (const pt of trail) {
        if (pt.t <= t) prev = pt;
        else {
          next = pt;
          break;
        }
      }
      if (!prev) prev = trail[0];

      let x = prev.x;
      let y = prev.y;
      if (next && aliveSet.has(id)) {
        const span = next.t - prev.t;
        const ratio = span > 0 ? Math.min(1, Math.max(0, (t - prev.t) / span)) : 0;
        x = prev.x + (next.x - prev.x) * ratio;
        y = prev.y + (next.y - prev.y) * ratio;
      }
      out.set(id, { x, y, lastT: prev.t, alive: aliveSet.has(id) });
    }
    return out;
  }, [trailsByTeam, curFrame, t]);

  const visibleEdges = useMemo(() => {
    const out: { key: string; src: number; dst: number; dmg: number }[] = [];
    let maxDmg = 1;
    for (const [key, acc] of edgeIndex.entries()) {
      let sum = 0;
      for (const ev of acc.events) {
        if (ev.t <= t) sum += ev.dmg;
        else break;
      }
      if (sum > 0) {
        out.push({ key, src: acc.src, dst: acc.dst, dmg: sum });
        if (sum > maxDmg) maxDmg = sum;
      }
    }
    return { edges: out, maxDmg };
  }, [edgeIndex, t]);

  // 라이브 KPI: 현재 alive 팀 수, 총 누적 events, 활성 엣지 수, 평균 degree, 최고 교전 팀
  const liveKpi = useMemo(() => {
    const aliveSet = new Set(curFrame?.alive ?? []);
    const teamDeg = new Map<number, number>();
    let totalEvents = 0;
    for (const [, acc] of edgeIndex.entries()) {
      let n = 0;
      for (const ev of acc.events) {
        if (ev.t <= t) n++;
        else break;
      }
      if (n > 0) {
        totalEvents += n;
        teamDeg.set(acc.src, (teamDeg.get(acc.src) ?? 0) + 1);
        teamDeg.set(acc.dst, (teamDeg.get(acc.dst) ?? 0) + 1);
      }
    }
    let topTeam = -1;
    let topDeg = 0;
    let aliveDegSum = 0;
    let aliveDegN = 0;
    for (const [tid, deg] of teamDeg.entries()) {
      if (deg > topDeg) {
        topDeg = deg;
        topTeam = tid;
      }
      if (aliveSet.has(tid)) {
        aliveDegSum += deg;
        aliveDegN += 1;
      }
    }
    const topNode = topTeam >= 0 ? nodeByIdx.get(topTeam) : null;
    return {
      alive: aliveSet.size,
      total: nodes.length,
      activeEdges: visibleEdges.edges.length,
      totalEvents,
      avgAliveDegree: aliveDegN > 0 ? aliveDegSum / aliveDegN : 0,
      topTeamId: topNode?.team ?? null,
      topDegree: topDeg,
    };
  }, [curFrame, edgeIndex, t, visibleEdges.edges.length, nodes.length, nodeByIdx]);

  // community hulls (alive 팀들로 구성). 동적 ts.comm 사용.
  const hulls = useMemo(() => {
    if (!showHulls) return [];
    const groups = new Map<number, [number, number][]>();
    for (const [id, pos] of teamPositions.entries()) {
      if (!pos.alive) continue;
      const node = nodeByIdx.get(id);
      if (!node) continue;
      const ts = curTeamState[String(id)];
      const comm = ts?.comm ?? node.community;
      if (comm < 0) continue;
      if (!groups.has(comm)) groups.set(comm, []);
      groups.get(comm)!.push([pos.x, pos.y]);
    }
    const out: { cid: number; poly: [number, number][] }[] = [];
    for (const [cid, pts] of groups.entries()) {
      if (pts.length < 2) continue;
      let poly: [number, number][];
      if (pts.length === 2) {
        // 두 점 → 미세하게 띄운 사각형
        poly = pts.map(([x, y]) => [x, y]);
      } else {
        poly = convexHull(pts);
      }
      out.push({ cid, poly: expandPolygon(poly, 1.18) });
    }
    return out;
  }, [teamPositions, nodeByIdx, curTeamState, showHulls]);

  const toX = (mx: number) => (mx / arenaM) * size;
  const toY = (my: number) => (my / arenaM) * size;

  // pan/zoom transform
  const tf = `translate(${pz.state.x.toFixed(2)} ${pz.state.y.toFixed(2)}) scale(${pz.state.k.toFixed(3)})`;

  return (
    <div
      className="relative mx-auto"
      style={{
        // 정사각: 화면 폭 100% 와 화면 높이 88vh 중 작은 쪽
        width: "min(100%, 88vh)",
        aspectRatio: "1 / 1",
      }}
    >
      <svg
        ref={svgRef}
        viewBox={`0 0 ${size} ${size}`}
        width="100%"
        height="100%"
        preserveAspectRatio="xMidYMid meet"
        style={{
          display: "block",
          background: "#0f1115",
          borderRadius: 6,
          border: "1px solid #d8d1c2",
          touchAction: "none",
        }}
      >
        <defs>
          <pattern
            id="anet-grid2"
            width={size / 8}
            height={size / 8}
            patternUnits="userSpaceOnUse"
          >
            <path
              d={`M ${size / 8} 0 L 0 0 0 ${size / 8}`}
              fill="none"
              stroke="#fefcf7"
              strokeOpacity={0.08}
              strokeWidth={0.5}
            />
          </pattern>
        </defs>

        <g transform={tf}>
          {/* 맵 이미지 */}
          <MapBackground size={size} urls={mapUrls} />
          {/* grid overlay */}
          <rect width={size} height={size} fill="url(#anet-grid2)" />

          {/* zone */}
          {curFrame?.zone && curFrame.zone.pr > 0 ? (
            <circle
              cx={toX(curFrame.zone.px)}
              cy={toY(curFrame.zone.py)}
              r={Math.max(2, (curFrame.zone.pr / arenaM) * size)}
              fill="none"
              stroke="#fafafa"
              strokeWidth={ZONE_NUM_STR}
              strokeDasharray="4 3"
              opacity={0.7}
            />
          ) : null}

          {/* community hulls */}
          {hulls.map(({ cid, poly }) => {
            const color = COMM_PALETTE[cid % COMM_PALETTE.length];
            const points = poly
              .map(([x, y]) => `${toX(x)},${toY(y)}`)
              .join(" ");
            return (
              <polygon
                key={`h${cid}`}
                points={points}
                fill={color}
                fillOpacity={0.13}
                stroke={color}
                strokeOpacity={0.55}
                strokeWidth={1}
              />
            );
          })}

          {/* trail (지난 60s) */}
          {Array.from(teamPositions.entries()).map(([id, pos]) => {
            if (!pos.alive) return null;
            const trail = trailsByTeam.get(id) ?? [];
            const recent: { x: number; y: number }[] = [];
            for (const pt of trail) {
              if (pt.t <= t && pt.t >= t - TRAIL_WINDOW_S) {
                recent.push({ x: pt.x, y: pt.y });
              }
            }
            recent.push({ x: pos.x, y: pos.y });
            if (recent.length < 2) return null;
            const node = nodeByIdx.get(id);
            const ts = curTeamState[String(id)];
            const color = node ? nodeColor(node, ts, colorBy, maxRank) : "#a8a29e";
            const points = recent.map((p) => `${toX(p.x)},${toY(p.y)}`).join(" ");
            return (
              <polyline
                key={`tr${id}`}
                points={points}
                fill="none"
                stroke={color}
                strokeWidth={1}
                opacity={0.45}
                strokeLinejoin="round"
              />
            );
          })}

          {/* edges */}
          {visibleEdges.edges.map((e) => {
            const sPos = teamPositions.get(e.src);
            const dPos = teamPositions.get(e.dst);
            if (!sPos || !dPos) return null;
            const w = Math.max(0.6, Math.log10(Math.max(1, e.dmg)) * 1.7);
            const op = 0.2 + 0.55 * Math.min(1, e.dmg / visibleEdges.maxDmg);
            const bothDead = !sPos.alive && !dPos.alive;
            return (
              <line
                key={e.key}
                x1={toX(sPos.x)}
                y1={toY(sPos.y)}
                x2={toX(dPos.x)}
                y2={toY(dPos.y)}
                stroke="#ff5a5a"
                strokeWidth={w}
                opacity={bothDead ? op * 0.4 : op}
                strokeLinecap="round"
              />
            );
          })}

          {/* nodes */}
          {Array.from(teamPositions.entries()).map(([id, pos]) => {
            const node = nodeByIdx.get(id);
            if (!node) return null;
            const cxV = toX(pos.x);
            const cyV = toY(pos.y);
            if (!Number.isFinite(cxV) || !Number.isFinite(cyV)) return null;
            const ts = curTeamState[String(id)];
            const color = nodeColor(node, ts, colorBy, maxRank);
            const opacity = pos.alive
              ? 1
              : Math.max(0.18, 1 - Math.max(0, t - pos.lastT) / 90);
            // 노드 크기는 t 시점의 dynamic degree 기반
            const degT = ts?.deg ?? node.degree;
            const r = 5 + Math.min(8, Math.sqrt(Math.max(1, degT * 1.4)));
            return (
              <g key={`n${id}`} opacity={opacity} data-pz="block">
                <circle
                  cx={cxV}
                  cy={cyV}
                  r={r}
                  fill={color}
                  stroke="#0f1115"
                  strokeWidth={1.1}
                >
                  <title>
                    T{node.team} · rank {node.rank ?? "—"}{"\n"}
                    @t={Math.round(t)}s · arc={ts?.arc ?? node.archetype} ·
                    mvm={ts?.mvm ?? node.movement_label}{"\n"}
                    pos={ts?.pos ?? node.position} · comm={ts?.comm ?? node.community} ·
                    deg={degT}
                  </title>
                </circle>
                <text
                  x={cxV}
                  y={cyV + 0.5}
                  fontSize={9}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fontFamily="IBM Plex Mono, monospace"
                  fontWeight={600}
                  fill="#fefcf7"
                  style={{ paintOrder: "stroke" }}
                  stroke="#0f1115"
                  strokeWidth={2.2}
                  pointerEvents="none"
                >
                  T{node.team}
                </text>
              </g>
            );
          })}
        </g>
      </svg>

      {/* zoom controls */}
      <div className="absolute right-2 top-2 flex flex-col gap-1" data-pz="block">
        <button
          onClick={() => pz.zoomBy(1.4)}
          className="rounded border border-rule bg-paper/90 px-2 py-1 text-xs hover:bg-gridline"
          title="확대"
        >
          +
        </button>
        <button
          onClick={() => pz.zoomBy(1 / 1.4)}
          className="rounded border border-rule bg-paper/90 px-2 py-1 text-xs hover:bg-gridline"
          title="축소"
        >
          −
        </button>
        <button
          onClick={pz.reset}
          className="rounded border border-rule bg-paper/90 px-2 py-1 text-xs hover:bg-gridline"
          title="초기화"
        >
          ⌂
        </button>
      </div>

      {/* Live KPI HUD (좌상단 오버레이) */}
      <div
        className="absolute left-2 top-2 rounded border border-rule bg-paper/85 px-3 py-2 text-[11px] shadow-sm pointer-events-none"
        data-pz="block"
      >
        <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 numeric leading-tight">
          <span className="text-muted">alive</span>
          <span className="text-right">
            <strong>{liveKpi.alive}</strong> / {liveKpi.total}
          </span>
          <span className="text-muted">events</span>
          <span className="text-right">{liveKpi.totalEvents.toLocaleString()}</span>
          <span className="text-muted">edges</span>
          <span className="text-right">{liveKpi.activeEdges}</span>
          <span className="text-muted">avg deg</span>
          <span className="text-right">{liveKpi.avgAliveDegree.toFixed(2)}</span>
          {liveKpi.topTeamId !== null ? (
            <>
              <span className="text-muted">top team</span>
              <span className="text-right">
                T{liveKpi.topTeamId}{" "}
                <span className="text-muted">({liveKpi.topDegree})</span>
              </span>
            </>
          ) : null}
        </div>
      </div>
    </div>
  );
}
