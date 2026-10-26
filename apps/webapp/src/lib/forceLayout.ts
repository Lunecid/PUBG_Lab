/**
 * Fruchterman–Reingold 간이 force-directed layout.
 *
 * 결정성을 위해 seed 기반 PRNG (mulberry32) 로 초기 위치 산출.
 * 입력으로 노드 수와 엣지 리스트만 받아 결과적으로 [0,1] × [0,1] 범위에
 * 정규화된 위치를 반환한다. nodes/edges 가 매치마다 다르므로 client-side
 * 계산용. ~26 노드, ~50 엣지에 대해 ms 단위.
 */

export interface FREdge {
  source: number;
  target: number;
  /** 가중치. 0~1 정규화는 호출 측 책임이지만 이 함수 내에선 무시. */
  weight?: number;
}

export interface FRPos {
  x: number;
  y: number;
}

function mulberry32(seed: number) {
  let s = seed | 0;
  return function () {
    s = (s + 0x6d2b79f5) | 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface FRConfig {
  nNodes: number;
  edges: FREdge[];
  /** 기본 200. 노드가 많으면 늘리는 게 좋음. */
  iterations?: number;
  /** 기본 17. 결정성 보장. */
  seed?: number;
  /** 가상 캔버스 크기 (정규화 전). 기본 1×1. */
  width?: number;
  height?: number;
  /** community id per node — 같은 community 끼리 약하게 묶어 줌. */
  communities?: number[];
}

export function computeForceLayout(cfg: FRConfig): FRPos[] {
  const {
    nNodes,
    edges,
    iterations = 220,
    seed = 17,
    width = 1,
    height = 1,
    communities,
  } = cfg;
  if (nNodes === 0) return [];

  const rand = mulberry32(seed);
  const area = width * height;
  // ideal 거리: k = sqrt(area / N). 너무 작은 N 에서도 안정적이도록 floor.
  const k = Math.sqrt(area / Math.max(nNodes, 4));

  // 초기 위치: communities 가 있으면 community 별로 약간 분리
  let pos: FRPos[] = [];
  if (communities && communities.length === nNodes) {
    const uniq = Array.from(new Set(communities)).filter((c) => c >= 0);
    const cCenters = new Map<number, { x: number; y: number }>();
    uniq.forEach((c, i) => {
      const angle = (i / Math.max(uniq.length, 1)) * Math.PI * 2;
      cCenters.set(c, {
        x: width / 2 + Math.cos(angle) * width * 0.25,
        y: height / 2 + Math.sin(angle) * height * 0.25,
      });
    });
    pos = communities.map((c) => {
      const ctr = cCenters.get(c) ?? { x: width / 2, y: height / 2 };
      return {
        x: ctr.x + (rand() - 0.5) * width * 0.18,
        y: ctr.y + (rand() - 0.5) * height * 0.18,
      };
    });
  } else {
    pos = Array.from({ length: nNodes }, () => ({
      x: rand() * width,
      y: rand() * height,
    }));
  }

  let temperature = Math.min(width, height) / 5;
  const cooling = temperature / (iterations + 1);

  const fa = (d: number) => (d * d) / k; // attractive
  const fr = (d: number) => (k * k) / d; // repulsive

  const disp: FRPos[] = Array.from({ length: nNodes }, () => ({ x: 0, y: 0 }));

  for (let it = 0; it < iterations; it++) {
    // repulsive
    for (let i = 0; i < nNodes; i++) {
      disp[i].x = 0;
      disp[i].y = 0;
      for (let j = 0; j < nNodes; j++) {
        if (i === j) continue;
        const dx = pos[i].x - pos[j].x;
        const dy = pos[i].y - pos[j].y;
        const dist = Math.hypot(dx, dy) || 0.001;
        const f = fr(dist);
        disp[i].x += (dx / dist) * f;
        disp[i].y += (dy / dist) * f;
      }
    }
    // attractive (edges)
    for (const e of edges) {
      const a = e.source;
      const b = e.target;
      if (a < 0 || a >= nNodes || b < 0 || b >= nNodes) continue;
      const dx = pos[a].x - pos[b].x;
      const dy = pos[a].y - pos[b].y;
      const dist = Math.hypot(dx, dy) || 0.001;
      const f = fa(dist) * (1 + (e.weight ?? 0) * 0.4);
      const ux = dx / dist;
      const uy = dy / dist;
      disp[a].x -= ux * f;
      disp[a].y -= uy * f;
      disp[b].x += ux * f;
      disp[b].y += uy * f;
    }
    // apply with temperature
    for (let i = 0; i < nNodes; i++) {
      const d = Math.hypot(disp[i].x, disp[i].y) || 0.001;
      const limited = Math.min(d, temperature);
      pos[i].x += (disp[i].x / d) * limited;
      pos[i].y += (disp[i].y / d) * limited;
      // bound box
      pos[i].x = Math.max(0, Math.min(width, pos[i].x));
      pos[i].y = Math.max(0, Math.min(height, pos[i].y));
    }
    temperature -= cooling;
    if (temperature < 0.001) temperature = 0.001;
  }

  // 정규화 [0, 1]
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of pos) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  const sx = maxX - minX || 1;
  const sy = maxY - minY || 1;
  return pos.map((p) => ({
    x: (p.x - minX) / sx,
    y: (p.y - minY) / sy,
  }));
}
