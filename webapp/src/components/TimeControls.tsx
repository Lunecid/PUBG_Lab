/**
 * 통일된 시간축 컨트롤.
 *
 * 기능:
 *  - 슬라이더 (드래그·클릭으로 정밀 시킹)
 *  - phase 마커 (zone 축소 시점)
 *  - 재생/일시정지/리셋/끝으로
 *  - 속도 토글 (12/24/48/96/240×)
 *  - phase jump 버튼 (선택)
 *  - 키보드 단축키
 *      Space: 재생/멈춤
 *      ← / →: ±10초 (Shift = ±60초)
 *      Home / End / 0: 처음 / 끝으로
 *      ,/.: 속도 다운/업
 *      1~9: 해당 phase 시작점으로 점프
 */
import { useEffect, useRef, useState } from "react";
import type { PhaseMarker } from "../lib/phaseMarkers";
import { fmtDuration } from "../lib/format";

interface Props {
  /** 매치 길이 (초) */
  duration: number;
  /** 현재 시각 */
  t: number;
  /** 시킹 콜백 */
  onSeek: (t: number) => void;
  /** 재생 중 여부 */
  playing: boolean;
  /** 토글 콜백 */
  onTogglePlay: () => void;
  /** 현재 배속 */
  speed: number;
  /** 배속 변경 */
  onSpeedChange: (s: number) => void;
  /** phase 마커 (옵션) */
  phaseMarkers?: PhaseMarker[];
  /** 키보드 단축키 활성화 (페이지당 하나만 활성화 권장) */
  enableShortcuts?: boolean;
  /** compact = 인라인용 작은 버전 */
  compact?: boolean;
}

const SPEED_OPTIONS = [12, 24, 48, 96, 240];

export default function TimeControls({
  duration,
  t,
  onSeek,
  playing,
  onTogglePlay,
  speed,
  onSpeedChange,
  phaseMarkers,
  enableShortcuts = false,
  compact = false,
}: Props) {
  const [dragging, setDragging] = useState(false);
  const sliderRef = useRef<HTMLDivElement | null>(null);

  // 키보드 단축키
  useEffect(() => {
    if (!enableShortcuts) return;
    const handler = (e: KeyboardEvent) => {
      // 입력 필드에 포커스 시 무시
      const tgt = e.target as HTMLElement;
      if (tgt.tagName === "INPUT" || tgt.tagName === "TEXTAREA" || tgt.isContentEditable) return;
      const big = e.shiftKey ? 60 : 10;
      switch (e.key) {
        case " ":
          e.preventDefault();
          onTogglePlay();
          break;
        case "ArrowLeft":
          e.preventDefault();
          onSeek(Math.max(0, t - big));
          break;
        case "ArrowRight":
          e.preventDefault();
          onSeek(Math.min(duration, t + big));
          break;
        case "Home":
        case "0":
          e.preventDefault();
          onSeek(0);
          break;
        case "End":
          e.preventDefault();
          onSeek(duration);
          break;
        case ",":
          e.preventDefault();
          {
            const idx = SPEED_OPTIONS.indexOf(speed);
            if (idx > 0) onSpeedChange(SPEED_OPTIONS[idx - 1]);
          }
          break;
        case ".":
          e.preventDefault();
          {
            const idx = SPEED_OPTIONS.indexOf(speed);
            if (idx >= 0 && idx < SPEED_OPTIONS.length - 1) onSpeedChange(SPEED_OPTIONS[idx + 1]);
          }
          break;
        default:
          // 1-9: phase jump
          if (phaseMarkers && /^[1-9]$/.test(e.key)) {
            const p = Number(e.key);
            const m = phaseMarkers.find((x) => x.phase === p);
            if (m) {
              e.preventDefault();
              onSeek(m.t);
            }
          }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [enableShortcuts, t, duration, speed, phaseMarkers, onSeek, onTogglePlay, onSpeedChange]);

  // 슬라이더 드래그 (마우스+터치)
  function pickFromEvent(clientX: number) {
    const el = sliderRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    onSeek(ratio * duration);
  }
  useEffect(() => {
    if (!dragging) return;
    const onMove = (e: MouseEvent) => pickFromEvent(e.clientX);
    const onTouchMove = (e: TouchEvent) => {
      if (e.touches[0]) pickFromEvent(e.touches[0].clientX);
    };
    const onUp = () => setDragging(false);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    window.addEventListener("touchmove", onTouchMove);
    window.addEventListener("touchend", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("touchmove", onTouchMove);
      window.removeEventListener("touchend", onUp);
    };
  }, [dragging, duration]);

  const pct = duration > 0 ? (t / duration) * 100 : 0;

  return (
    <div
      className={
        "rounded border border-rule bg-paper " + (compact ? "p-2" : "p-3")
      }
    >
      {/* 상단 행: 재생 + 속도 + 시각 */}
      <div className="flex items-center gap-2">
        <button
          onClick={onTogglePlay}
          className={
            "jet rounded border border-rule bg-cream font-medium hover:bg-cream/70 " +
            (compact ? "px-2 py-1 text-xs" : "px-3 py-1.5 text-sm")
          }
          title={playing ? "일시정지 (Space)" : "재생 (Space)"}
        >
          {playing ? "⏸" : "▶"}
        </button>
        <button
          onClick={() => onSeek(0)}
          className="jet rounded border border-rule bg-paper px-2 py-1 text-xs hover:bg-cream"
          title="처음으로 (Home / 0)"
        >
          ⟲
        </button>
        <button
          onClick={() => onSeek(Math.max(0, t - (compact ? 30 : 60)))}
          className="jet rounded border border-rule bg-paper px-2 py-1 text-xs hover:bg-cream"
          title={"이전 " + (compact ? 30 : 60) + "초 (← / Shift+←)"}
        >
          ⏪
        </button>
        <button
          onClick={() => onSeek(Math.min(duration, t + (compact ? 30 : 60)))}
          className="jet rounded border border-rule bg-paper px-2 py-1 text-xs hover:bg-cream"
          title={"다음 " + (compact ? 30 : 60) + "초 (→ / Shift+→)"}
        >
          ⏩
        </button>
        <button
          onClick={() => onSeek(duration)}
          className="jet rounded border border-rule bg-paper px-2 py-1 text-xs hover:bg-cream"
          title="끝으로 (End)"
        >
          ⏭
        </button>

        <div className="jet ml-2 numeric flex items-baseline gap-1 text-xs text-muted">
          <span className="tabular-nums">{fmtDuration(t)}</span>
          <span className="text-faint">/</span>
          <span className="tabular-nums">{fmtDuration(duration)}</span>
          <span className="ml-2 text-[10px] text-faint">
            (t={Math.round(t)}s)
          </span>
        </div>

        <div className="ml-auto flex items-center gap-1">
          <span className="jet text-[10px] uppercase text-muted">속도</span>
          {SPEED_OPTIONS.map((s) => (
            <button
              key={s}
              onClick={() => onSpeedChange(s)}
              className={
                "jet rounded px-1.5 py-0.5 text-[10px] " +
                (speed === s
                  ? "bg-accent text-paper"
                  : "bg-paper text-muted hover:bg-cream")
              }
              title={"배속 " + s + "× (, / .)"}
            >
              {s}×
            </button>
          ))}
        </div>
      </div>

      {/* 슬라이더 + phase 마커 */}
      <div
        ref={sliderRef}
        className="relative mt-3 h-6 cursor-pointer select-none"
        onMouseDown={(e) => {
          setDragging(true);
          pickFromEvent(e.clientX);
        }}
        onTouchStart={(e) => {
          setDragging(true);
          if (e.touches[0]) pickFromEvent(e.touches[0].clientX);
        }}
      >
        {/* 트랙 */}
        <div className="absolute left-0 right-0 top-1/2 h-1.5 -translate-y-1/2 rounded bg-cream" />
        {/* 진행 채움 */}
        <div
          className="absolute left-0 top-1/2 h-1.5 -translate-y-1/2 rounded bg-accent"
          style={{ width: `${pct}%` }}
        />
        {/* phase 마커 (트랙 위 작은 막대) */}
        {phaseMarkers?.map((m) => {
          const left = (m.t / duration) * 100;
          return (
            <div
              key={m.phase}
              className="absolute top-0 flex h-full -translate-x-1/2 flex-col items-center"
              style={{ left: `${left}%` }}
              title={`phase ${m.phase} 시작 (t=${Math.round(m.t)}s, r=${m.radius}m)`}
            >
              <div className="h-3 w-px bg-muted opacity-70" />
              <div className="jet text-[8px] leading-none text-muted">
                P{m.phase}
              </div>
            </div>
          );
        })}
        {/* 핸들 */}
        <div
          className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-paper bg-accent shadow"
          style={{ left: `${pct}%` }}
        />
      </div>

      {/* 단축키 힌트 (compact 일 땐 숨김) */}
      {!compact && enableShortcuts ? (
        <div className="jet mt-2 flex flex-wrap gap-x-3 text-[10px] text-faint">
          <span>Space 재생</span>
          <span>←→ ±10s</span>
          <span>Shift+←→ ±60s</span>
          <span>Home/End</span>
          {phaseMarkers && phaseMarkers.length > 0 ? <span>1~{Math.min(9, phaseMarkers.length)} phase</span> : null}
          <span>, / . 속도</span>
        </div>
      ) : null}
    </div>
  );
}
