// 맵 배경 렌더러.
// PUBG 좌표계 (top-left 원점, x→오른쪽, y→아래) → SVG 좌표 (top-left 원점)에 그대로 매핑.
// 8x8km 맵의 미니맵 PNG 가 정사각형 (8192x8192) 이므로 size×size 영역에 stretch 없이 fit.
//
// urls fallback chain: 첫 URL이 로드 실패하면 다음 URL 시도. (high → low res)

import { useEffect, useState } from "react";

interface Props {
  size: number;
  /** 우선순위 순 URL 배열. 없으면 어두운 기본 배경. */
  urls?: string[];
}

export default function MapBackground({ size, urls }: Props) {
  const [activeIdx, setActiveIdx] = useState(0);
  const [allFailed, setAllFailed] = useState(false);

  // urls 변경 시 첫 번째부터 다시 시도
  useEffect(() => {
    setActiveIdx(0);
    setAllFailed(false);
  }, [urls?.join("|")]);

  if (!urls || urls.length === 0 || allFailed) {
    // fallback: 어두운 배경 (격자는 AnimatedNetwork 가 별도로 그림)
    return <rect width={size} height={size} fill="#0f1115" />;
  }

  const currentUrl = urls[activeIdx];

  return (
    <>
      {/* 어두운 배경 (이미지 로드 전 깜빡임 방지) */}
      <rect width={size} height={size} fill="#0f1115" />
      {/* 미니맵 이미지. 8192x8192 PNG/JPG 가 0~size px 영역에 그대로 매핑됨.
          PUBG 텔레메트리의 (x, y) 미터값을 (x/arenaM, y/arenaM)*size 로 변환하면
          이미지 위 정확한 픽셀 좌표가 된다. */}
      <image
        href={currentUrl}
        x={0}
        y={0}
        width={size}
        height={size}
        preserveAspectRatio="none"
        opacity={0.85}
        onError={() => {
          if (activeIdx < urls.length - 1) setActiveIdx(activeIdx + 1);
          else setAllFailed(true);
        }}
      />
      {/* 어두운 오버레이 — 이미지 위 노드/엣지 가독성 확보 */}
      <rect width={size} height={size} fill="#000" opacity={0.18} />
    </>
  );
}
