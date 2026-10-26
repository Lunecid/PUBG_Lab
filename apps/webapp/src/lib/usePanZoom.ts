import { useEffect, useRef, useState } from "react";

export interface PanZoomState {
  x: number;
  y: number;
  k: number;
}

const MIN_K = 0.7;
const MAX_K = 8;

/**
 * SVG 컨테이너에 wheel(zoom) + drag(pan) 을 부착하고 transform 을 반환.
 * outer <g transform={...}> 에 그대로 넣어 사용한다.
 */
export function usePanZoom(svgRef: React.RefObject<SVGSVGElement | null>) {
  const [state, setState] = useState<PanZoomState>({ x: 0, y: 0, k: 1 });
  const dragRef = useRef<{
    startX: number;
    startY: number;
    sx: number;
    sy: number;
  } | null>(null);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = svg.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      // SVG viewBox 좌표로 변환
      const vbW = svg.viewBox.baseVal.width || rect.width;
      const vbH = svg.viewBox.baseVal.height || rect.height;
      const cx = (px / rect.width) * vbW;
      const cy = (py / rect.height) * vbH;

      setState((prev) => {
        const dk = e.deltaY < 0 ? 1.18 : 1 / 1.18;
        const newK = Math.max(MIN_K, Math.min(MAX_K, prev.k * dk));
        if (Math.abs(newK - prev.k) < 1e-9) return prev;
        // zoom around cursor
        const ratio = newK / prev.k;
        return {
          k: newK,
          x: cx - (cx - prev.x) * ratio,
          y: cy - (cy - prev.y) * ratio,
        };
      });
    };

    const onPointerDown = (e: PointerEvent) => {
      // 노드/버튼 위 클릭은 무시 (data-pz="block" 이거나 버튼)
      const tgt = e.target as Element;
      if (tgt.closest && tgt.closest("[data-pz='block']")) return;
      if (e.button !== 0) return;
      dragRef.current = {
        startX: e.clientX,
        startY: e.clientY,
        sx: state.x,
        sy: state.y,
      };
      svg.setPointerCapture(e.pointerId);
    };

    const onPointerMove = (e: PointerEvent) => {
      if (!dragRef.current) return;
      const rect = svg.getBoundingClientRect();
      const vbW = svg.viewBox.baseVal.width || rect.width;
      const vbH = svg.viewBox.baseVal.height || rect.height;
      const dx = ((e.clientX - dragRef.current.startX) / rect.width) * vbW;
      const dy = ((e.clientY - dragRef.current.startY) / rect.height) * vbH;
      setState((prev) => ({
        ...prev,
        x: dragRef.current!.sx + dx,
        y: dragRef.current!.sy + dy,
      }));
    };

    const onPointerUp = (e: PointerEvent) => {
      dragRef.current = null;
      try {
        svg.releasePointerCapture(e.pointerId);
      } catch {/* ignore */}
    };

    svg.addEventListener("wheel", onWheel, { passive: false });
    svg.addEventListener("pointerdown", onPointerDown);
    svg.addEventListener("pointermove", onPointerMove);
    svg.addEventListener("pointerup", onPointerUp);
    svg.addEventListener("pointercancel", onPointerUp);

    return () => {
      svg.removeEventListener("wheel", onWheel);
      svg.removeEventListener("pointerdown", onPointerDown);
      svg.removeEventListener("pointermove", onPointerMove);
      svg.removeEventListener("pointerup", onPointerUp);
      svg.removeEventListener("pointercancel", onPointerUp);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [svgRef, state.x, state.y]);

  function reset() {
    setState({ x: 0, y: 0, k: 1 });
  }

  function zoomBy(factor: number) {
    setState((prev) => {
      const newK = Math.max(MIN_K, Math.min(MAX_K, prev.k * factor));
      if (Math.abs(newK - prev.k) < 1e-9) return prev;
      // zoom around viewBox center
      const svg = svgRef.current;
      const vbW = svg?.viewBox.baseVal.width || 540;
      const vbH = svg?.viewBox.baseVal.height || 540;
      const cx = vbW / 2;
      const cy = vbH / 2;
      const ratio = newK / prev.k;
      return {
        k: newK,
        x: cx - (cx - prev.x) * ratio,
        y: cy - (cy - prev.y) * ratio,
      };
    });
  }

  return { state, reset, zoomBy };
}
