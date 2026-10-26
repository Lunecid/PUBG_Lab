// shard.frames 의 zone 변화에서 phase 경계 추출.
// PUBG zone 은 phase 마다 다음 zone 이 표시 (sr → 다음 phase 의 목표 반경) 후 일정 시간이 지나면 축소(pr 가 변함).
// 단순화: pr 가 변하는 시점을 phase 경계로 간주.

import type { MatchFrame } from "./types";

export interface PhaseMarker {
  /** 1, 2, 3, ... */
  phase: number;
  /** 매치 시작부터 초 */
  t: number;
  /** 그 시점의 zone 반경 (m). null 이면 zone 미존재. */
  radius: number | null;
}

/**
 * frames 에서 zone radius 가 처음 0 이 아닌 값으로 등장하는 시점을 phase 1 시작,
 * 이후 radius 가 의미 있게(>5%) 변하는 시점을 다음 phase 시작으로 본다.
 */
export function extractPhaseMarkers(frames: MatchFrame[]): PhaseMarker[] {
  if (frames.length === 0) return [];
  const out: PhaseMarker[] = [];
  let lastR: number | null = null;
  let phase = 0;
  for (const f of frames) {
    const r = f.zone?.pr ?? 0;
    if (r > 0) {
      if (lastR === null) {
        // phase 1 시작
        phase = 1;
        out.push({ phase, t: f.t, radius: r });
        lastR = r;
      } else if (r < lastR * 0.95) {
        // 5% 이상 축소 → 새 phase
        phase += 1;
        out.push({ phase, t: f.t, radius: r });
        lastR = r;
      }
    }
  }
  return out;
}
