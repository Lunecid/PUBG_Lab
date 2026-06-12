// =============================================================================
// presentation_build/build.js
// -----------------------------------------------------------------------------
// Renders PUBG_STDB_STGNN.pptx — research-proposal presentation.
//
// Title (B): "축소되는 안전구역에서의 동적 시공간 그래프 기반 PUBG 팀 생존 예측"
// Author    : 백성은 (202589713) · 부산대 데이터사이언스
//
// Changes vs prior build (per reviewer feedback A·B·C·D·E·G; RQ kept as-is):
//   A) Model name      : "ST-GCN" → "Dynamic Spatio-Temporal GNN"
//                        (vanilla ST-GCN assumes a fixed graph; PUBG team
//                         interaction graphs are dynamic.)
//   B) Title           : updated to highlight dynamic graph + survival framing.
//   C) STDB formalism  : slide adds a *derived-relation* card next to the
//                        three primitives.
//   D) Main target     : survival hazard h_i(t) (auxiliary: final rank).
//   E) Edge typology   : 3 edge types — Proximity, Combat, Zone-pressure.
//   G) Evaluation plan : survival + rank + classification + ablation families
//                        listed (names & directions only — no numbers).
//
// Proposal tone is preserved: no measured metric values, no schema/variable
// names, no hyperparameters in slide bodies.  Speaker notes carry the backups.
// =============================================================================

const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const {
  FaDatabase,
  FaProjectDiagram,
  FaChartLine,
  FaCrosshairs,
  FaMapMarkedAlt,
  FaNetworkWired,
  FaCogs,
  FaFlagCheckered,
  FaLayerGroup,
  FaSitemap,
  FaBullseye,
  FaRoute,
  FaShieldAlt,
  FaBolt,
  FaTrophy,
  FaEye,
  FaWalking,
  FaTree,
  FaFistRaised,
  FaBoxOpen,
  FaSeedling,
  FaSearchPlus,
  FaShareAlt,
  FaQuestionCircle,
  FaCubes,
  FaMapMarkerAlt,
  FaClock,
  FaPlane,
  FaSatelliteDish,
  FaGamepad,
  FaUsers,
  FaGlobe,
  FaCalendarAlt,
  FaParachuteBox,
  FaHandshake,
  FaArrowRight,
  FaCompass,
  FaHeartbeat,
  FaBalanceScale,
  FaTachometerAlt,
  FaFire,
} = require("react-icons/fa");

// =============================================================================
// COLORS — academic tone (white bg + navy accent)
// =============================================================================
const C = {
  bg:        "FFFFFF",
  text:      "1A1A1A",
  accent:    "0D1B2A",
  subtle:    "6B7280",
  line:      "E5E7EB",
  cardBg:    "F8FAFC",
  cardEdge:  "D1D8E0",
  hi:        "1E3A8A",
  hazard:    "B91C1C",   // muted red for hazard / survival emphasis
};

// =============================================================================
// HELPERS
// =============================================================================
function renderIconSvg(IconComponent, color = "#0D1B2A", size = 256) {
  return ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color, size: String(size) })
  );
}
async function iconToBase64Png(IconComponent, color = "#0D1B2A", size = 256) {
  const svg = renderIconSvg(IconComponent, color, size);
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

function addFooter(slide, pageNum, totalPages) {
  slide.addShape("line", {
    x: 0.4, y: 5.28, w: 9.2, h: 0,
    line: { color: C.line, width: 0.5 },
  });
  slide.addText(
    "축소되는 안전구역에서의 동적 시공간 그래프 기반 PUBG 팀 생존 예측 — 연구 계획",
    {
      x: 0.4, y: 5.30, w: 7.2, h: 0.22,
      fontSize: 9, fontFace: "Calibri", color: C.subtle, margin: 0,
    }
  );
  slide.addText(`${pageNum} / ${totalPages}`, {
    x: 8.6, y: 5.30, w: 1.0, h: 0.22,
    fontSize: 9, fontFace: "Calibri",
    color: C.subtle, align: "right", margin: 0,
  });
}

function addSlideTitle(slide, sectionNum, title, subtitle) {
  if (sectionNum) {
    slide.addText(sectionNum, {
      x: 0.4, y: 0.35, w: 2.4, h: 0.30,
      fontSize: 10, fontFace: "Calibri", bold: true,
      color: C.accent, charSpacing: 4, margin: 0,
    });
  }
  slide.addText(title, {
    x: 0.4, y: 0.62, w: 9.2, h: 0.55,
    fontSize: 24, fontFace: "Calibri", bold: true,
    color: C.accent, margin: 0,
  });
  slide.addShape("rect", {
    x: 0.4, y: 1.16, w: 1.6, h: 0.03,
    fill: { color: C.accent }, line: { color: C.accent },
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.4, y: 1.26, w: 9.2, h: 0.30,
      fontSize: 12, fontFace: "Calibri", italic: true,
      color: C.subtle, margin: 0,
    });
  }
}

function addCard(slide, x, y, w, h) {
  slide.addShape("roundRect", {
    x, y, w, h,
    fill: { color: C.cardBg },
    line: { color: C.cardEdge, width: 0.75 },
    rectRadius: 0.07,
  });
}

// =============================================================================
// BUILD
// =============================================================================
async function build() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.author = "Sungeun Baek (백성은)";
  pres.title =
    "Dynamic Spatio-Temporal Graph Modeling for Team Survival Prediction in PUBG Telemetry";

  const TOTAL = 12;
  const I = {
    db:       await iconToBase64Png(FaDatabase),
    graph:    await iconToBase64Png(FaProjectDiagram),
    chart:    await iconToBase64Png(FaChartLine),
    cross:    await iconToBase64Png(FaCrosshairs),
    map:      await iconToBase64Png(FaMapMarkedAlt),
    net:      await iconToBase64Png(FaNetworkWired),
    cogs:     await iconToBase64Png(FaCogs),
    flag:     await iconToBase64Png(FaFlagCheckered),
    layer:    await iconToBase64Png(FaLayerGroup),
    sitemap:  await iconToBase64Png(FaSitemap),
    bullseye: await iconToBase64Png(FaBullseye),
    route:    await iconToBase64Png(FaRoute),
    shield:   await iconToBase64Png(FaShieldAlt),
    bolt:     await iconToBase64Png(FaBolt),
    trophy:   await iconToBase64Png(FaTrophy),
    eye:      await iconToBase64Png(FaEye),
    walk:     await iconToBase64Png(FaWalking),
    fist:     await iconToBase64Png(FaFistRaised),
    seed:     await iconToBase64Png(FaSeedling),
    search:   await iconToBase64Png(FaSearchPlus),
    share:    await iconToBase64Png(FaShareAlt),
    question: await iconToBase64Png(FaQuestionCircle),
    cubes:    await iconToBase64Png(FaCubes),
    pin:      await iconToBase64Png(FaMapMarkerAlt),
    clock:    await iconToBase64Png(FaClock),
    plane:    await iconToBase64Png(FaPlane),
    sat:      await iconToBase64Png(FaSatelliteDish),
    pad:      await iconToBase64Png(FaGamepad),
    users:    await iconToBase64Png(FaUsers),
    globe:    await iconToBase64Png(FaGlobe),
    cal:      await iconToBase64Png(FaCalendarAlt),
    chute:    await iconToBase64Png(FaParachuteBox),
    hand:     await iconToBase64Png(FaHandshake),
    arrow:    await iconToBase64Png(FaArrowRight),
    compass:  await iconToBase64Png(FaCompass),
    heart:    await iconToBase64Png(FaHeartbeat),
    balance:  await iconToBase64Png(FaBalanceScale),
    tach:     await iconToBase64Png(FaTachometerAlt),
    fire:     await iconToBase64Png(FaFire, "#B91C1C"),
  };

  // ────────────────────────────────────────────────────────────────────────
  // SLIDE 1 — COVER (B: new title)
  // ────────────────────────────────────────────────────────────────────────
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };

    s.addShape("rect", {
      x: 0.4, y: 0.6, w: 0.08, h: 4.4,
      fill: { color: C.accent }, line: { color: C.accent },
    });

    s.addText("Research Proposal · 2026-05-27", {
      x: 0.7, y: 0.65, w: 6, h: 0.32,
      fontSize: 11, fontFace: "Calibri",
      color: C.subtle, charSpacing: 3,
    });

    s.addText("축소되는 안전구역에서의", {
      x: 0.7, y: 1.00, w: 9, h: 0.55,
      fontSize: 28, fontFace: "Calibri", bold: true, color: C.accent,
    });
    s.addText("동적 시공간 그래프 기반", {
      x: 0.7, y: 1.55, w: 9, h: 0.55,
      fontSize: 28, fontFace: "Calibri", bold: true, color: C.accent,
    });
    s.addText("PUBG 팀 생존 예측", {
      x: 0.7, y: 2.10, w: 9, h: 0.55,
      fontSize: 28, fontFace: "Calibri", bold: true, color: C.accent,
    });

    s.addText(
      "Dynamic Spatio-Temporal Graph Modeling for Team Survival Prediction in PUBG Telemetry",
      {
        x: 0.7, y: 2.85, w: 9, h: 0.55,
        fontSize: 13.5, fontFace: "Calibri", italic: true, color: C.subtle,
      }
    );

    s.addText("백성은  Sungeun Baek", {
      x: 0.7, y: 3.80, w: 8, h: 0.30,
      fontSize: 14, fontFace: "Calibri", bold: true, color: C.text,
    });
    s.addText("부산대학교 데이터사이언스전문대학원 · 데이터사이언스학과", {
      x: 0.7, y: 4.10, w: 8, h: 0.26,
      fontSize: 11, fontFace: "Calibri", color: C.subtle,
    });
    s.addText("202589713", {
      x: 0.7, y: 4.36, w: 8, h: 0.26,
      fontSize: 11, fontFace: "Calibri", color: C.subtle,
    });
  }

  // ────────────────────────────────────────────────────────────────────────
  // SLIDE 2 — PUBG 게임 소개  (변경 없음)
  // ────────────────────────────────────────────────────────────────────────
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    addSlideTitle(s, "Introduction", "PUBG, 어떤 게임인가",
      "PLAYERUNKNOWN'S BATTLEGROUNDS — KRAFTON이 개발한 대표 배틀로얄 게임");

    const items = [
      { icon: I.pad,    head: "배틀로얄 장르",       body: "한 매치에 최대 100명의 플레이어가 참가, 단 하나의 팀만이 살아남는다" },
      { icon: I.globe,  head: "광활한 야외 지도",     body: "약 8km × 8km 크기의 오픈맵에서 자유롭게 이동하며 자원을 확보" },
      { icon: I.shield, head: "축소되는 안전구역",    body: "시간이 지나면서 안전구역이 축소되어 플레이어들을 점점 좁은 공간으로 몰아넣음" },
      { icon: I.users,  head: "솔로·듀오·스쿼드 팀전", body: "1인부터 4인 팀까지 다양한 모드 · 1·3인칭 시점 선택 가능" },
      { icon: I.trophy, head: "마지막 1팀이 우승",     body: "기존 PVP 슈팅 게임과 달리 *살아남기* 자체가 핵심 목표가 된다" },
    ];
    let y = 1.75;
    for (const it of items) {
      s.addImage({ data: it.icon, x: 0.55, y: y + 0.05, w: 0.42, h: 0.42 });
      s.addText(it.head, {
        x: 1.10, y: y, w: 8.4, h: 0.30,
        fontSize: 13.5, bold: true, color: C.accent, fontFace: "Calibri",
      });
      s.addText(it.body, {
        x: 1.10, y: y + 0.30, w: 8.4, h: 0.40,
        fontSize: 11.5, color: C.text, fontFace: "Calibri",
      });
      y += 0.66;
    }
    addFooter(s, 2, TOTAL);
  }

  // ────────────────────────────────────────────────────────────────────────
  // SLIDE 3 — 공식 Telemetry  (변경 없음)
  // ────────────────────────────────────────────────────────────────────────
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    addSlideTitle(s, "Data", "공식 Telemetry — 시공간 데이터의 공급원",
      "KRAFTON Public API가 매 경기마다 수십만 건의 이벤트를 시간·좌표와 함께 제공한다");

    addCard(s, 0.40, 1.75, 5.80, 3.30);
    s.addText("주요 이벤트와 시공간 필드", {
      x: 0.55, y: 1.85, w: 5.55, h: 0.30,
      fontSize: 12.5, bold: true, color: C.accent, fontFace: "Calibri",
    });
    const rows = [
      ["LogPlayerPosition",       "플레이어 위치 (시간 · X/Y/Z 좌표)"],
      ["LogGameStatePeriodic",    "안전구역 중심 · 반경 (시간 진행에 따른 동적 영역)"],
      ["LogPlayerKillV2",         "킬 이벤트 (시간 · killer/victim 좌표)"],
      ["LogPhaseChange",          "안전구역 phase 전환 시점"],
      ["LogPlayerTakeDamage",     "피해 이벤트 (시간 · 가해/피해 좌표)"],
      ["LogParachuteLanding",     "낙하 착지 (시간 · 위치)"],
    ];
    let ry = 2.20;
    for (const [evt, desc] of rows) {
      s.addShape("line", {
        x: 0.55, y: ry + 0.32, w: 5.50, h: 0,
        line: { color: C.line, width: 0.4 },
      });
      s.addText(evt, {
        x: 0.55, y: ry, w: 2.40, h: 0.30,
        fontSize: 10, color: C.accent, bold: true, fontFace: "Courier New",
      });
      s.addText(desc, {
        x: 2.95, y: ry, w: 3.10, h: 0.30,
        fontSize: 10, color: C.text, fontFace: "Calibri",
      });
      ry += 0.34;
    }

    addCard(s, 6.30, 1.75, 3.30, 3.30);
    s.addImage({ data: I.layer, x: 6.45, y: 1.88, w: 0.42, h: 0.42 });
    s.addText("세 가지 자연스러운 묶음", {
      x: 6.95, y: 1.88, w: 2.60, h: 0.32,
      fontSize: 12, bold: true, color: C.accent, fontFace: "Calibri",
    });
    s.addText([
      { text: "▸ 이동 객체\n",       options: { color: C.text, bold: true } },
      { text: "   플레이어 / 팀의 위치 시계열\n\n", options: { color: C.text } },
      { text: "▸ 동적 공간 영역\n",  options: { color: C.text, bold: true } },
      { text: "   안전구역 (시간에 따라 변함)\n\n", options: { color: C.text } },
      { text: "▸ 공간 사건\n",       options: { color: C.text, bold: true } },
      { text: "   전투 / 킬 / 낙하 이벤트", options: { color: C.text } },
    ], { x: 6.45, y: 2.40, w: 3.10, h: 2.55, fontSize: 11, fontFace: "Calibri" });

    s.addText([
      { text: "출처: ", options: { color: C.subtle } },
      { text: "documentation.pubg.com/en/telemetry-events.html",
        options: { color: C.hi, italic: true,
                   hyperlink: { url: "https://documentation.pubg.com/en/telemetry-events.html" } } },
    ], { x: 0.40, y: 5.10, w: 9.2, h: 0.20,
         fontSize: 9, fontFace: "Calibri" });
    addFooter(s, 3, TOTAL);
  }

  // ────────────────────────────────────────────────────────────────────────
  // SLIDE 4 — 선행 연구  (변경 없음)
  // ────────────────────────────────────────────────────────────────────────
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    addSlideTitle(s, "01 · Related Work", "선행 연구 — Sagong, Woo, Kim (2025)",
      "phase-based 행동 분류로 실시간 생존 예측을 시도한 대표 사례");

    addCard(s, 0.40, 1.75, 9.20, 1.65);
    s.addText("4 archetype — FightScore · ActiveScore", {
      x: 0.60, y: 1.85, w: 9, h: 0.30,
      fontSize: 13, bold: true, fontFace: "Calibri", color: C.accent,
    });

    const arche = [
      { icon: I.cross,    label: "Hunter",   desc: "공격성 높음 · 활동성 높음" },
      { icon: I.bullseye, label: "Sniper",   desc: "공격성 높음 · 활동성 낮음" },
      { icon: I.route,    label: "Traveler", desc: "공격성 낮음 · 활동성 높음" },
      { icon: I.shield,   label: "Camper",   desc: "공격성 낮음 · 활동성 낮음" },
    ];
    const arX = [0.60, 2.85, 5.10, 7.35];
    for (let i = 0; i < arche.length; i++) {
      s.addImage({ data: arche[i].icon, x: arX[i], y: 2.24, w: 0.30, h: 0.30 });
      s.addText(arche[i].label, {
        x: arX[i] + 0.36, y: 2.22, w: 1.9, h: 0.30,
        fontSize: 12, bold: true, color: C.text, fontFace: "Calibri",
      });
      s.addText(arche[i].desc, {
        x: arX[i], y: 2.55, w: 2.25, h: 0.32,
        fontSize: 10, color: C.subtle, fontFace: "Calibri",
      });
    }

    addCard(s, 0.40, 3.55, 9.20, 1.45);
    s.addText("핵심 기여", {
      x: 0.60, y: 3.65, w: 8.9, h: 0.30,
      fontSize: 12, bold: true, color: C.accent, fontFace: "Calibri",
    });
    s.addText([
      { text: "플레이어의 행동을 ", options: { color: C.text } },
      { text: "공격성·활동성", options: { color: C.text, bold: true } },
      { text: " 두 축으로 점수화해 archetype을 정의하고, 각 phase마다 archetype 분포가 다음 phase 생존·최종 승리에 어떤 영향을 주는지 분석. ", options: { color: C.text } },
      { text: "단계별(phase-based) 분석", options: { color: C.text, bold: true } },
      { text: "의 유효성을 입증.", options: { color: C.text } },
    ], { x: 0.60, y: 3.95, w: 8.9, h: 1.00, fontSize: 12, fontFace: "Calibri" });

    s.addText([
      { text: "Sagong, Woo, Kim (2025) — ScienceDirect: ", options: { color: C.subtle } },
      { text: "S187595212500120X", options: { color: C.hi, italic: true,
        hyperlink: { url: "https://www.sciencedirect.com/science/article/abs/pii/S187595212500120X" } } },
    ], { x: 0.40, y: 5.05, w: 9.2, h: 0.20,
         fontSize: 9, fontFace: "Calibri" });
    addFooter(s, 4, TOTAL);
  }

  // ────────────────────────────────────────────────────────────────────────
  // SLIDE 5 — 연구 배경 + 우리의 접근  (차별점 명확화)
  // ────────────────────────────────────────────────────────────────────────
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    addSlideTitle(s, "02 · Motivation", "연구 배경 및 우리의 접근",
      "행동 유형 분류에서 한 발 더 — 팀이 놓인 동적 시공간 환경 그 자체를 본다");

    // Left: 기존 접근의 한계
    addCard(s, 0.40, 1.75, 4.55, 3.30);
    s.addImage({ data: I.chart, x: 0.60, y: 1.88, w: 0.36, h: 0.36 });
    s.addText("기존 접근의 한계", {
      x: 1.05, y: 1.88, w: 3.65, h: 0.34,
      fontSize: 14, bold: true, color: C.accent, fontFace: "Calibri",
    });
    s.addText([
      { text: "기존 PUBG 분석은 두 가지 흐름:\n\n", options: { color: C.text } },
      { text: "▸ 경기 종료 후 집계 통계\n", options: { color: C.text, bold: true } },
      { text: "  킬 수 · 피해량 · 이동거리 · 생존시간\n\n", options: { color: C.text } },
      { text: "▸ 행동 유형 분류\n", options: { color: C.text, bold: true } },
      { text: "  Hunter / Sniper / Traveler / Camper 등\n\n", options: { color: C.text } },
      { text: "→ ", options: { color: C.hi } },
      { text: "특정 시점에 팀이 어떤 ", options: { color: C.text, italic: true } },
      { text: "공간 환경 · 팀 간 관계", options: { color: C.text, italic: true, bold: true } },
      { text: " 속에 놓여 있었는지는 잘 보이지 않는다.", options: { color: C.text, italic: true } },
    ], { x: 0.60, y: 2.35, w: 4.25, h: 2.65, fontSize: 11.5, fontFace: "Calibri" });

    // Right: 우리의 접근
    addCard(s, 5.05, 1.75, 4.55, 3.30);
    s.addImage({ data: I.compass, x: 5.25, y: 1.88, w: 0.36, h: 0.36 });
    s.addText("우리의 접근", {
      x: 5.70, y: 1.88, w: 3.65, h: 0.34,
      fontSize: 14, bold: true, color: C.accent, fontFace: "Calibri",
    });
    s.addText([
      { text: "PUBG Telemetry를 ", options: { color: C.text } },
      { text: "시공간 데이터베이스 (STDB)", options: { color: C.text, bold: true } },
      { text: " 로 재구성하고, ", options: { color: C.text } },
      { text: "축소되는 안전구역 속 팀 간 관계", options: { color: C.text, bold: true } },
      { text: "를 ", options: { color: C.text } },
      { text: "동적 시공간 그래프", options: { color: C.text, bold: true } },
      { text: "로 모델링한다.\n\n", options: { color: C.text } },
      { text: "팀 = 노드, 시점별 근접·교전·안전구역 압력 = 엣지로 두어, 그래프 구조 자체가 시간에 따라 진화하도록 표현한다.\n\n", options: { color: C.text } },
      { text: "그래프 신경망(Dynamic ST-GNN)으로 ", options: { color: C.text } },
      { text: "다음 시점 탈락 위험(hazard)", options: { color: C.hazard, bold: true } },
      { text: "을 예측하고, 그 결과를 phase별 생존환경 해석과 연결한다.", options: { color: C.text } },
    ], { x: 5.25, y: 2.35, w: 4.20, h: 2.65, fontSize: 11.5, fontFace: "Calibri" });

    addFooter(s, 5, TOTAL);
  }

  // ────────────────────────────────────────────────────────────────────────
  // SLIDE 6 — 연구 질문 (RQ1·RQ2·RQ3)  — 사용자 요청에 따라 기존 3개 유지
  // ────────────────────────────────────────────────────────────────────────
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    addSlideTitle(s, "03 · Research Questions", "연구 질문",
      "세 개의 질문이 STDB → 생존환경 분석 → 동적 그래프 예측으로 이어진다");

    const rqs = [
      {
        num: "RQ1",
        icon: I.db,
        title: "STDB 구조 표현",
        body: "PUBG Telemetry는 어떤 시공간 데이터베이스 구조로 표현될 수 있는가?",
      },
      {
        num: "RQ2",
        icon: I.shield,
        title: "Phase별 환경 변화",
        body: "안전구역 축소에 따라 팀의 이동 궤적과 생존환경은 어떻게 변화하는가?",
      },
      {
        num: "RQ3",
        icon: I.trophy,
        title: "등수 예측 가능성",
        body: "시공간 데이터로 각 팀의 등수를 예측할 수 있는가? 가능하다면 어떤 요인이 중요한가?",
      },
    ];
    const colX = [0.40, 3.65, 6.90];
    for (let i = 0; i < rqs.length; i++) {
      addCard(s, colX[i], 1.75, 2.85, 3.30);
      s.addText(rqs[i].num, {
        x: colX[i] + 0.18, y: 1.90, w: 1.0, h: 0.30,
        fontSize: 11, bold: true, color: C.accent, charSpacing: 3, fontFace: "Calibri",
      });
      s.addImage({ data: rqs[i].icon, x: colX[i] + 1.10, y: 1.95, w: 1.0, h: 1.0 });
      s.addText(rqs[i].title, {
        x: colX[i] + 0.18, y: 3.10, w: 2.55, h: 0.36,
        fontSize: 15, bold: true, color: C.accent, fontFace: "Calibri",
      });
      s.addText(rqs[i].body, {
        x: colX[i] + 0.18, y: 3.50, w: 2.55, h: 1.50,
        fontSize: 11, color: C.text, fontFace: "Calibri",
        paraSpaceAfter: 2, valign: "top",
      });
    }
    addFooter(s, 6, TOTAL);
  }

  // ────────────────────────────────────────────────────────────────────────
  // SLIDE 7 — 4단계 파이프라인 (D: ④ 단계 hazard로 수정)
  // ────────────────────────────────────────────────────────────────────────
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    addSlideTitle(s, "04 · Methodology", "연구 방법 — 4단계 파이프라인",
      "데이터 정리 → STDB 모델링 → phase별 생존환경 분석 → 동적 그래프 생존 예측");

    const steps = [
      { num: "①", icon: I.cubes,  title: "데이터 정리",
        body: "원시 Telemetry를\n분석에 적합한 형태로\n시간 격자에 정렬" },
      { num: "②", icon: I.db,     title: "STDB 모델링",
        body: "이동 객체 · 동적 영역\n· 공간 사건 + 파생\n관계의 시공간 구조" },
      { num: "③", icon: I.layer,  title: "Phase별 분석",
        body: "안전구역 phase 단위\n생존환경 지표\n계산 및 비교" },
      { num: "④", icon: I.graph,  title: "Dynamic ST-GNN",
        body: "다음 시점 탈락 위험\n예측(hazard)\n+ 최종 등수 (보조)" },
    ];
    const stepX = [0.40, 2.70, 5.00, 7.30];
    const stepY = 1.80;
    for (let i = 0; i < steps.length; i++) {
      addCard(s, stepX[i], stepY, 2.20, 2.95);
      s.addText(steps[i].num, {
        x: stepX[i] + 0.15, y: stepY + 0.12, w: 0.6, h: 0.40,
        fontSize: 24, bold: true, color: C.accent, fontFace: "Calibri",
      });
      s.addImage({ data: steps[i].icon, x: stepX[i] + 0.75, y: stepY + 0.20, w: 0.60, h: 0.60 });
      s.addText(steps[i].title, {
        x: stepX[i] + 0.18, y: stepY + 0.95, w: 2.0, h: 0.34,
        fontSize: 14, bold: true, color: C.accent, fontFace: "Calibri",
      });
      s.addText(steps[i].body, {
        x: stepX[i] + 0.18, y: stepY + 1.32, w: 2.0, h: 1.55,
        fontSize: 11, color: C.text, fontFace: "Calibri", valign: "top",
      });
      if (i < steps.length - 1) {
        s.addImage({ data: I.arrow,
          x: stepX[i] + 2.23, y: stepY + 1.32, w: 0.20, h: 0.20 });
      }
    }
    addFooter(s, 7, TOTAL);
  }

  // ────────────────────────────────────────────────────────────────────────
  // SLIDE 8 — STDB 모델링 + Formalization (C)
  //   좌 : 세 가지 추상 (Moving Object / Dynamic Region / Spatial Event)
  //   우 : 그 위에서 유도되는 파생 관계 (proximity / combat / zone-pressure / enemy-density)
  // ────────────────────────────────────────────────────────────────────────
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    addSlideTitle(s, "04 · ①② Methodology", "STDB 모델링 — 세 가지 추상 + 파생 관계",
      "저장만이 아니라, 시공간 질의·feature 생성이 가능한 구조로 재구성한다");

    // LEFT — 세 가지 추상 (compact list)
    addCard(s, 0.40, 1.75, 4.55, 3.30);
    s.addText("기본 객체 (Primitives)", {
      x: 0.55, y: 1.85, w: 4.30, h: 0.30,
      fontSize: 12, bold: true, color: C.accent, fontFace: "Calibri",
    });

    const prims = [
      { icon: I.walk,   title: "Moving Object",        body: "player_i(t),  team_j(t)" },
      { icon: I.shield, title: "Dynamic Spatial Region", body: "safe_zone(t),  blue_zone(t)" },
      { icon: I.fist,   title: "Spatial Event",         body: "damage(e,t,x,y),  kill(e,t,x,y)" },
    ];
    let py = 2.25;
    for (const p of prims) {
      s.addImage({ data: p.icon, x: 0.60, y: py + 0.05, w: 0.36, h: 0.36 });
      s.addText(p.title, {
        x: 1.05, y: py, w: 3.70, h: 0.30,
        fontSize: 12, bold: true, color: C.accent, fontFace: "Calibri",
      });
      s.addText(p.body, {
        x: 1.05, y: py + 0.30, w: 3.85, h: 0.55,
        fontSize: 10.5, color: C.text, fontFace: "Courier New",
      });
      py += 0.92;
    }

    // RIGHT — 파생 관계 (key contribution highlight)
    addCard(s, 5.05, 1.75, 4.55, 3.30);
    s.addImage({ data: I.sitemap, x: 5.20, y: 1.88, w: 0.40, h: 0.40 });
    s.addText("파생 관계 (Derived Relations)", {
      x: 5.70, y: 1.88, w: 3.85, h: 0.34,
      fontSize: 12, bold: true, color: C.accent, fontFace: "Calibri",
    });

    s.addText([
      { text: "proximity_edge(team_i, team_j, t)\n",
        options: { color: C.text, bold: true, fontFace: "Courier New" } },
      { text: "  ▸ 같은 시점, 일정 거리 이내의 팀 쌍\n\n",
        options: { color: C.text } },
      { text: "combat_edge(team_i, team_j, t)\n",
        options: { color: C.text, bold: true, fontFace: "Courier New" } },
      { text: "  ▸ 최근 Δt 내 damage / knock / kill로 연결된 팀 쌍\n\n",
        options: { color: C.text } },
      { text: "zone_pressure(team_i, t)\n",
        options: { color: C.text, bold: true, fontFace: "Courier New" } },
      { text: "  ▸ 안전구역 경계까지의 거리·이동 압력\n\n",
        options: { color: C.text } },
      { text: "enemy_density(team_i, t)\n",
        options: { color: C.text, bold: true, fontFace: "Courier New" } },
      { text: "  ▸ 주변 반경 내 적 팀 밀도",
        options: { color: C.text } },
    ], { x: 5.25, y: 2.32, w: 4.25, h: 2.70, fontSize: 10.5, fontFace: "Calibri" });

    s.addText("→ 위 파생 관계가 동적 시공간 그래프의 노드 feature와 엣지를 정의한다.", {
      x: 0.40, y: 5.05, w: 9.20, h: 0.22,
      fontSize: 10.5, italic: true, color: C.subtle, fontFace: "Calibri",
    });
    addFooter(s, 8, TOTAL);
  }

  // ────────────────────────────────────────────────────────────────────────
  // SLIDE 9 — Phase별 생존환경 분석  (변경 최소)
  // ────────────────────────────────────────────────────────────────────────
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    addSlideTitle(s, "04 · ③ Methodology", "Phase별 생존환경 분석",
      "안전구역 phase가 진행될 때마다, 네 가지 시공간 지표가 함께 변한다");

    const metrics = [
      { icon: I.pin,   title: "안전구역 경계 거리\n(zone pressure)",
        body: "팀이 안전구역\n경계로부터 얼마나\n떨어져 있는가" },
      { icon: I.cubes, title: "주변 적 팀 밀도\n(enemy density)",
        body: "팀 주변 일정 반경 내\n적 팀이 몇 개나\n존재하는가" },
      { icon: I.fire,  title: "최근 교전 노출\n(combat exposure)",
        body: "최근 일정 시간 동안\n받은 피해의\n누적량" },
      { icon: I.clock, title: "위치 안정성\n(mobility stability)",
        body: "팀 centroid가\nphase 내에서\n얼마나 안정적인가" },
    ];
    const mX = [0.40, 2.70, 5.00, 7.30];
    for (let i = 0; i < metrics.length; i++) {
      addCard(s, mX[i], 1.75, 2.20, 2.30);
      s.addImage({ data: metrics[i].icon, x: mX[i] + 0.85, y: 1.92, w: 0.55, h: 0.55 });
      s.addText(metrics[i].title, {
        x: mX[i] + 0.10, y: 2.55, w: 2.05, h: 0.55,
        fontSize: 11.5, bold: true, color: C.accent, align: "center", fontFace: "Calibri",
      });
      s.addText(metrics[i].body, {
        x: mX[i] + 0.10, y: 3.12, w: 2.05, h: 0.88,
        fontSize: 10, color: C.text, align: "center", fontFace: "Calibri",
      });
    }

    addCard(s, 0.40, 4.20, 9.20, 0.90);
    s.addText([
      { text: "이 네 지표를 phase 단위로 집계해, ", options: { color: C.text } },
      { text: "상위권 팀과 하위권 팀의 시공간 패턴 차이", options: { color: C.text, bold: true } },
      { text: "를 비교하고, 동적 그래프의 노드 feature로 투입한다.", options: { color: C.text } },
    ], { x: 0.55, y: 4.28, w: 8.90, h: 0.75,
         fontSize: 11.5, fontFace: "Calibri", paraSpaceAfter: 2 });

    addFooter(s, 9, TOTAL);
  }

  // ────────────────────────────────────────────────────────────────────────
  // SLIDE 10 — Dynamic Spatio-Temporal GNN (A · D · E)
  // ────────────────────────────────────────────────────────────────────────
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    addSlideTitle(s, "04 · ④ Methodology", "Dynamic Spatio-Temporal GNN",
      "팀 그래프가 시간에 따라 진화 — 노드·엣지·feature 모두 동적");

    // top architecture row
    const blocks = [
      { label: "Temporal Conv",  sub: "(시간축 합성곱)" },
      { label: "Dynamic\nSpatial GCN",    sub: "(시점별 A_t)" },
      { label: "Temporal Conv",  sub: "(시간축 합성곱)" },
    ];
    const bX = [0.40, 2.80, 5.20];
    for (let i = 0; i < blocks.length; i++) {
      addCard(s, bX[i], 1.75, 2.20, 1.30);
      s.addText(blocks[i].label, {
        x: bX[i] + 0.10, y: 1.83, w: 2.0, h: 0.55,
        fontSize: 12, bold: true, color: C.accent, fontFace: "Calibri", align: "center",
      });
      s.addText(blocks[i].sub, {
        x: bX[i] + 0.10, y: 2.42, w: 2.0, h: 0.55,
        fontSize: 10, color: C.subtle, fontFace: "Calibri", align: "center",
      });
      if (i < 2) {
        s.addImage({ data: I.arrow, x: bX[i] + 2.23, y: 2.30, w: 0.20, h: 0.20 });
      }
    }
    s.addImage({ data: I.arrow, x: 7.43, y: 2.30, w: 0.20, h: 0.20 });

    addCard(s, 7.65, 1.75, 1.95, 1.30);
    s.addText([
      { text: "예측 출력\n\n", options: { color: C.accent, bold: true } },
      { text: "▸ 탈락 위험 (hazard)\n", options: { color: C.hazard, bold: true } },
      { text: "▸ 최종 등수 (보조)",     options: { color: C.text } },
    ], {
      x: 7.75, y: 1.83, w: 1.80, h: 1.10,
      fontSize: 10.5, fontFace: "Calibri", align: "center", valign: "top",
    });

    // ── Edge typology (E) — three edge types in one row
    addCard(s, 0.40, 3.20, 9.20, 0.95);
    s.addText("엣지 3종 (heterogeneous dynamic graph)", {
      x: 0.55, y: 3.27, w: 9.0, h: 0.26,
      fontSize: 11.5, bold: true, color: C.accent, fontFace: "Calibri",
    });
    const edgeCols = [
      { title: "Proximity",     desc: "일정 거리 이내 팀 쌍" },
      { title: "Combat",        desc: "최근 Δt damage/knock/kill 발생" },
      { title: "Zone-pressure", desc: "같은 안전구역 경계·좁은 영역 밀집" },
    ];
    const eX = [0.55, 3.65, 6.75];
    for (let i = 0; i < edgeCols.length; i++) {
      s.addShape("ellipse", {
        x: eX[i], y: 3.62, w: 0.14, h: 0.14,
        fill: { color: C.accent }, line: { color: C.accent },
      });
      s.addText(edgeCols[i].title, {
        x: eX[i] + 0.22, y: 3.55, w: 2.4, h: 0.26,
        fontSize: 11, bold: true, color: C.accent, fontFace: "Calibri",
      });
      s.addText(edgeCols[i].desc, {
        x: eX[i] + 0.22, y: 3.78, w: 2.55, h: 0.34,
        fontSize: 10, color: C.text, fontFace: "Calibri",
      });
    }

    // bottom: legend (Node / Temporal / Target)
    const legend = [
      { label: "Node",     body: "한 매치에 참가한 각 팀" },
      { label: "Temporal", body: "동일 노드의 연속 시점 연결" },
      { label: "Primary target", body: "h_i(t) = 다음 Δt 탈락 확률" },
    ];
    const lX = [0.40, 3.55, 6.70];
    for (let i = 0; i < legend.length; i++) {
      s.addShape("ellipse", {
        x: lX[i] + 0.05, y: 4.41, w: 0.14, h: 0.14,
        fill: { color: i === 2 ? C.hazard : C.accent }, line: { color: i === 2 ? C.hazard : C.accent },
      });
      s.addText(legend[i].label, {
        x: lX[i] + 0.26, y: 4.34, w: 3.0, h: 0.26,
        fontSize: 11, bold: true, color: i === 2 ? C.hazard : C.accent, fontFace: "Calibri",
      });
      s.addText(legend[i].body, {
        x: lX[i] + 0.05, y: 4.58, w: 3.10, h: 0.34,
        fontSize: 10, color: C.text, fontFace: "Calibri",
      });
    }

    s.addText([
      { text: "Yan, Xiong, Lin (2018) — ", options: { color: C.subtle } },
      { text: "arxiv:1801.07455", options: { color: C.hi,
        hyperlink: { url: "https://arxiv.org/abs/1801.07455" } } },
      { text: "    ·    Yu, Yin, Zhu (2018) — ", options: { color: C.subtle } },
      { text: "arxiv:1709.04875", options: { color: C.hi,
        hyperlink: { url: "https://arxiv.org/abs/1709.04875" } } },
    ], { x: 0.40, y: 5.05, w: 9.20, h: 0.22,
         fontSize: 9.5, fontFace: "Calibri", italic: true });
    addFooter(s, 10, TOTAL);
  }

  // ────────────────────────────────────────────────────────────────────────
  // SLIDE 11 — 평가 방향 (포괄적)
  //   세부 지표를 나열하지 않고, "어떤 카테고리의 지표를 설계할 것인가"만 보여준다.
  // ────────────────────────────────────────────────────────────────────────
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    addSlideTitle(s, "05 · Evaluation", "평가 방향",
      "세 가지 카테고리의 지표를 설계하고, baseline 비교로 그래프 기반 접근의 기여를 검증한다");

    // Top row — 3 indicator categories
    const cats = [
      {
        icon: I.heart,
        title: "생존 예측",
        body: "다음 시점 탈락 위험\n(hazard) 의 정확도와\n시간별 보정도를 설계",
        color: C.hazard,
      },
      {
        icon: I.trophy,
        title: "등수 예측",
        body: "최종 등수의 순위\n상관·평균 오차 등을\n보조 지표로 설계",
        color: C.hi,
      },
      {
        icon: I.bullseye,
        title: "상위권 진입",
        body: "Top-K 진입 여부의\n분류 성능 지표를\n함께 설계",
        color: C.hi,
      },
    ];
    const catX = [0.40, 3.65, 6.90];
    for (let i = 0; i < cats.length; i++) {
      addCard(s, catX[i], 1.75, 2.85, 2.10);
      s.addImage({ data: cats[i].icon, x: catX[i] + 1.18, y: 1.92, w: 0.50, h: 0.50 });
      s.addText(cats[i].title, {
        x: catX[i] + 0.10, y: 2.50, w: 2.65, h: 0.36,
        fontSize: 14, bold: true, color: cats[i].color, align: "center", fontFace: "Calibri",
      });
      s.addText(cats[i].body, {
        x: catX[i] + 0.10, y: 2.92, w: 2.65, h: 0.88,
        fontSize: 11, color: C.text, align: "center", fontFace: "Calibri",
      });
    }

    // Bottom row — comparison and interpretation (2 cards)
    addCard(s, 0.40, 4.05, 4.55, 1.05);
    s.addImage({ data: I.balance, x: 0.55, y: 4.18, w: 0.40, h: 0.40 });
    s.addText("Baseline 비교 (Ablation)", {
      x: 1.05, y: 4.15, w: 3.80, h: 0.30,
      fontSize: 13, bold: true, color: C.accent, fontFace: "Calibri",
    });
    s.addText("집계 통계·비그래프 모델 등과 비교해 동적 그래프 접근의 기여를 검증", {
      x: 1.05, y: 4.48, w: 3.80, h: 0.55,
      fontSize: 10.5, color: C.text, fontFace: "Calibri",
    });

    addCard(s, 5.05, 4.05, 4.55, 1.05);
    s.addImage({ data: I.search, x: 5.20, y: 4.18, w: 0.40, h: 0.40 });
    s.addText("설명 가능성", {
      x: 5.70, y: 4.15, w: 3.80, h: 0.30,
      fontSize: 13, bold: true, color: C.accent, fontFace: "Calibri",
    });
    s.addText("어떤 시공간 요인이 생존에 가장 크게 기여하는지 분석", {
      x: 5.70, y: 4.48, w: 3.80, h: 0.55,
      fontSize: 10.5, color: C.text, fontFace: "Calibri",
    });

    addFooter(s, 11, TOTAL);
  }

  // ────────────────────────────────────────────────────────────────────────
  // SLIDE 12 — References
  // ────────────────────────────────────────────────────────────────────────
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    addSlideTitle(s, "References", "참고 문헌", null);

    const refs = [
      {
        label: "[1]",
        text:  "KRAFTON.  PUBG API Documentation — Telemetry and Telemetry Events.",
        url:   "https://documentation.pubg.com/en/telemetry-events.html",
      },
      {
        label: "[2]",
        text:  "Sagong C., Woo J. Y., Kim H. K.  Are you a Hunter or Camper? A phase-based framework for real-time survival prediction in battle royale using FightScore and ActiveScore.  (2025)",
        url:   "https://www.sciencedirect.com/science/article/abs/pii/S187595212500120X",
      },
      {
        label: "[3]",
        text:  "Yan S., Xiong Y., Lin D.  Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition.  (2018)",
        url:   "https://arxiv.org/abs/1801.07455",
      },
      {
        label: "[4]",
        text:  "Yu B., Yin H., Zhu Z.  Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting.  (2018)",
        url:   "https://arxiv.org/abs/1709.04875",
      },
    ];
    let ry = 1.85;
    for (const r of refs) {
      s.addText(r.label, {
        x: 0.45, y: ry, w: 0.55, h: 0.30,
        fontSize: 11, bold: true, color: C.accent, fontFace: "Calibri",
      });
      s.addText([
        { text: r.text + "\n", options: { color: C.text } },
        { text: r.url,         options: { color: C.hi, italic: true,
                                          hyperlink: { url: r.url } } },
      ], { x: 1.00, y: ry, w: 8.60, h: 0.80,
           fontSize: 11, fontFace: "Calibri" });
      ry += 0.78;
    }
    addFooter(s, 12, TOTAL);
  }

  // ────────────────────────────────────────────────────────────────────────
  // OUTPUT
  // ────────────────────────────────────────────────────────────────────────
  const path = require("path");
  const outPath = path.resolve(__dirname, "..", "PUBG_STDB_STGNN.pptx");
  await pres.writeFile({ fileName: outPath });
  console.log("wrote", outPath);
}

build().catch(e => {
  console.error(e);
  process.exit(1);
});
