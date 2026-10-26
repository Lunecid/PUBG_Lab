import { useEffect, useRef, useState } from "react";

interface Opts {
  /** 매치 길이 (초) */
  duration: number;
  /** 시작 시 자동 재생 */
  autoplay?: boolean;
  /** 1배속 = 1초/실초. 기본 24 (24× 빠르게 = 25초 매치를 1초에 재생) */
  speed?: number;
  /** duration 도달 시 처음으로 되감기 */
  loop?: boolean;
  /** 시작 시각 (없으면 0) */
  initialT?: number;
}

/**
 * 시간축 재생 훅. 매치 시간(초)을 [0, duration] 범위로 다루고,
 * 재생/일시정지/배속/시킹을 관리.
 */
export function useTimePlayer(opts: Opts) {
  const { duration, autoplay = false, speed: initSpeed = 24, loop = false, initialT = 0 } = opts;
  const [t, setT] = useState<number>(initialT);
  const [playing, setPlaying] = useState<boolean>(autoplay);
  const [speed, setSpeed] = useState<number>(initSpeed);
  const lastFrameRef = useRef<number | null>(null);

  useEffect(() => {
    if (!playing) {
      lastFrameRef.current = null;
      return;
    }
    let raf = 0;
    const tick = (ts: number) => {
      if (lastFrameRef.current == null) {
        lastFrameRef.current = ts;
        raf = requestAnimationFrame(tick);
        return;
      }
      const dt = (ts - lastFrameRef.current) / 1000; // 초 단위 실 경과
      lastFrameRef.current = ts;
      setT((prev) => {
        const next = prev + dt * speed;
        if (next >= duration) {
          if (loop) return 0;
          setPlaying(false);
          return duration;
        }
        return next;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, duration, loop]);

  function play() {
    if (t >= duration) setT(0);
    setPlaying(true);
  }
  function pause() {
    setPlaying(false);
  }
  function togglePlay() {
    if (playing) pause();
    else play();
  }
  function seek(target: number) {
    setT(Math.max(0, Math.min(duration, target)));
  }
  function reset() {
    setPlaying(false);
    setT(0);
  }
  function jumpToEnd() {
    setPlaying(false);
    setT(duration);
  }

  return { t, setT, playing, play, pause, togglePlay, speed, setSpeed, seek, reset, jumpToEnd };
}
