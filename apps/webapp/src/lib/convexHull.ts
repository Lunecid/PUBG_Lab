/**
 * Andrew's monotone chain convex hull.
 * 입력: [x, y] 점 배열. 출력: 시계 반대 방향 hull 점들 (ccw).
 */
export function convexHull(points: [number, number][]): [number, number][] {
  if (points.length <= 1) return points.slice();
  const pts = points
    .slice()
    .sort((a, b) => (a[0] === b[0] ? a[1] - b[1] : a[0] - b[0]));

  const cross = (
    o: [number, number],
    a: [number, number],
    b: [number, number],
  ) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);

  const lower: [number, number][] = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }
  const upper: [number, number][] = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }
  upper.pop();
  lower.pop();
  return lower.concat(upper);
}

/** 폴리곤을 중심에서 factor 배 확장 (커뮤니티 hull 시각적 padding 용). */
export function expandPolygon(
  poly: [number, number][],
  factor: number,
): [number, number][] {
  if (poly.length === 0) return poly;
  let cx = 0, cy = 0;
  for (const [x, y] of poly) { cx += x; cy += y; }
  cx /= poly.length;
  cy /= poly.length;
  return poly.map(([x, y]) => [
    cx + (x - cx) * factor,
    cy + (y - cy) * factor,
  ]);
}
