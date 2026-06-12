// =============================================================================
// presentation_build/build_intro_en_5min.js
// -----------------------------------------------------------------------------
// Renders PUBG_Research_Intro_EN_5min.pptx — a 5-slide, 5-minute "this is what
// I am working on" research introduction.  Intentionally NO results, NO
// conclusions: question → data → idea → approach.
//
//   node build_intro_en_5min.js
// =============================================================================

const pptxgen = require("pptxgenjs");

const C = {
  bg:       "FFFFFF",
  text:     "1A1A1A",
  accent:   "0D1B2A",
  subtle:   "6B7280",
  cardBg:   "F7F8FA",
  cardEdge: "E5E7EB",
  line:     "E5E7EB",
};

const F = "Calibri";
const FOOTER = "Predicting Team Elimination in PUBG — Research Introduction";

function addFooter(slide, pageNum, totalPages) {
  slide.addShape("line", {
    x: 0.4, y: 5.28, w: 9.2, h: 0,
    line: { color: C.line, width: 0.5 },
  });
  slide.addText(FOOTER, {
    x: 0.4, y: 5.30, w: 7.2, h: 0.22,
    fontSize: 9, fontFace: F, color: C.subtle, margin: 0,
  });
  slide.addText(`${pageNum} / ${totalPages}`, {
    x: 8.6, y: 5.30, w: 1.0, h: 0.22,
    fontSize: 9, fontFace: F, color: C.subtle, align: "right", margin: 0,
  });
}

function addSlideTitle(slide, sectionNum, title, subtitle) {
  if (sectionNum) {
    slide.addText(sectionNum, {
      x: 0.4, y: 0.35, w: 4.0, h: 0.30,
      fontSize: 10, fontFace: F, bold: true,
      color: C.accent, charSpacing: 4, margin: 0,
    });
  }
  slide.addText(title, {
    x: 0.4, y: 0.62, w: 9.2, h: 0.55,
    fontSize: 24, fontFace: F, bold: true, color: C.accent, margin: 0,
  });
  slide.addShape("rect", {
    x: 0.4, y: 1.16, w: 1.6, h: 0.03,
    fill: { color: C.accent }, line: { color: C.accent },
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.4, y: 1.26, w: 9.2, h: 0.30,
      fontSize: 12, fontFace: F, italic: true, color: C.subtle, margin: 0,
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

async function build() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  const TOTAL = 5;

  // ---------------------------------------------------------------- 1 · TITLE
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    s.addShape("rect", {
      x: 0, y: 0, w: 10, h: 0.18, fill: { color: C.accent },
    });
    s.addText("RESEARCH INTRODUCTION · JUNE 2026", {
      x: 0.7, y: 1.10, w: 8.6, h: 0.3,
      fontSize: 11, fontFace: F, bold: true, color: C.subtle,
      charSpacing: 4, margin: 0,
    });
    s.addText("Predicting Team Elimination in PUBG", {
      x: 0.7, y: 1.50, w: 8.6, h: 0.65,
      fontSize: 30, fontFace: F, bold: true, color: C.accent, margin: 0,
    });
    s.addText(
      "Modeling game telemetry as a spatio-temporal database",
      {
        x: 0.7, y: 2.25, w: 8.6, h: 0.45,
        fontSize: 15, fontFace: F, italic: true, color: C.subtle, margin: 0,
      }
    );
    s.addShape("rect", {
      x: 0.7, y: 3.05, w: 1.6, h: 0.03,
      fill: { color: C.accent }, line: { color: C.accent },
    });
    s.addText("Seongeun Baek", {
      x: 0.7, y: 3.25, w: 8.6, h: 0.3,
      fontSize: 14, fontFace: F, bold: true, color: C.text, margin: 0,
    });
    s.addText("Data Science · Pusan National University", {
      x: 0.7, y: 3.55, w: 8.6, h: 0.3,
      fontSize: 11, fontFace: F, color: C.subtle, margin: 0,
    });
  }

  // ------------------------------------------------- 2 · THE GAME & MY QUESTION
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    addSlideTitle(s, "THE GAME · THE QUESTION", "A shrinking world, 28 teams, one survivor");
    const left = [
      "Up to ~28 squads parachute onto one large map",
      "A circular safe zone shrinks step by step (“phases”) — staying outside drains health",
      "The last team standing wins; everyone else is eliminated along the way",
    ];
    addCard(s, 0.4, 1.75, 4.5, 2.60);
    s.addText("PUBG in one minute", {
      x: 0.58, y: 1.89, w: 4.14, h: 0.32,
      fontSize: 13, bold: true, fontFace: F, color: C.accent, margin: 0,
    });
    s.addText(
      left.map((t) => ({
        text: t,
        options: { bullet: { code: "2022", indent: 10 }, breakLine: true },
      })),
      {
        x: 0.58, y: 2.25, w: 4.14, h: 2.0,
        fontSize: 11.5, fontFace: F, color: C.text,
        valign: "top", margin: 0, lineSpacingMultiple: 1.15,
      }
    );
    addCard(s, 5.1, 1.75, 4.5, 2.60);
    s.addText("My question", {
      x: 5.28, y: 1.89, w: 4.14, h: 0.32,
      fontSize: 13, bold: true, fontFace: F, color: C.accent, margin: 0,
    });
    s.addText(
      [
        { text: "While the match is still running:", options: { breakLine: true } },
        { text: "“Which teams will be eliminated in the next zone phase?”",
          options: { bold: true, color: C.accent, breakLine: true } },
        { text: "", options: { breakLine: true } },
        { text: "Not who wins in the end — but who is in danger right now, and why.",
          options: { italic: true, color: C.subtle } },
      ],
      {
        x: 5.28, y: 2.25, w: 4.14, h: 2.0,
        fontSize: 12.5, fontFace: F, color: C.text,
        valign: "top", margin: 0, lineSpacingMultiple: 1.18,
      }
    );
    addFooter(s, 2, TOTAL);
  }

  // ----------------------------------------------------------------- 3 · DATA
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    addSlideTitle(s, "THE DATA", "The game records everything",
      "Official PUBG API telemetry — every position, shot, and zone change, with a time and a place.");
    const stats = [
      ["27,491", "real squad matches collected"],
      ["115M+", "combat events with coordinates"],
    ];
    stats.forEach(([num, label], i) => {
      const x = 0.4 + i * 4.7;
      addCard(s, x, 1.85, 4.5, 1.35);
      s.addText(num, {
        x: x + 0.2, y: 1.97, w: 2.0, h: 1.10,
        fontSize: 28, bold: true, fontFace: F, color: C.accent,
        valign: "middle", margin: 0,
      });
      s.addText(label, {
        x: x + 2.2, y: 1.97, w: 2.2, h: 1.10,
        fontSize: 11.5, fontFace: F, color: C.text,
        valign: "middle", margin: 0,
      });
    });
    addCard(s, 0.4, 3.45, 9.2, 1.15);
    s.addText(
      [
        { text: "Rich, but raw.  ", options: { bold: true, color: C.accent } },
        { text: "The telemetry is a flat log of events — it does not say where a team stood relative to the zone, who was close to whom, or how pressure built up over time. That structure has to be built.",
          options: { color: C.text } },
      ],
      {
        x: 0.62, y: 3.55, w: 8.8, h: 0.95,
        fontSize: 12, fontFace: F, valign: "middle", margin: 0,
      }
    );
    addFooter(s, 3, TOTAL);
  }

  // ------------------------------------------------------------- 4 · THE IDEA
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    addSlideTitle(s, "THE IDEA", "Treat the match as a spatio-temporal database",
      "Reorganize the raw log into three spatio-temporal objects, aligned to zone phases.");
    const prims = [
      ["Moving Objects", "Teams as trajectories — where each team is, over time"],
      ["Dynamic Regions", "The safe zone as a region whose center and size change each phase"],
      ["Spatial Events", "Kills and damage as events pinned to a time and a location"],
    ];
    prims.forEach(([head, body], i) => {
      const x = 0.4 + i * 3.13;
      addCard(s, x, 1.80, 2.95, 1.95);
      s.addText(head, {
        x: x + 0.18, y: 1.95, w: 2.6, h: 0.35,
        fontSize: 13, bold: true, fontFace: F, color: C.accent, margin: 0,
      });
      s.addText(body, {
        x: x + 0.18, y: 2.35, w: 2.6, h: 1.25,
        fontSize: 11.5, fontFace: F, color: C.text, valign: "top", margin: 0,
      });
    });
    addCard(s, 0.4, 4.00, 9.2, 0.90);
    s.addText(
      [
        { text: "The zone phase becomes the clock of the game.  ", options: { bold: true, color: C.accent } },
        { text: "Phase by phase, simple spatial queries describe each team's situation: distance to the zone, nearby enemies, recent fights.",
          options: { color: C.text } },
      ],
      {
        x: 0.62, y: 4.10, w: 8.8, h: 0.70,
        fontSize: 12, fontFace: F, valign: "middle", margin: 0,
      }
    );
    addFooter(s, 4, TOTAL);
  }

  // --------------------------------------------------------- 5 · THE APPROACH
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    addSlideTitle(s, "THE APPROACH", "A graph between teams, rewired every phase",
      "Survival is not a solo property — it depends on who you are entangled with.");
    addCard(s, 0.4, 1.80, 5.4, 2.85);
    s.addText("Teams as a dynamic graph", {
      x: 0.58, y: 1.94, w: 5.0, h: 0.32,
      fontSize: 13, bold: true, fontFace: F, color: C.accent, margin: 0,
    });
    s.addText(
      [
        "Each team = a node carrying its phase-by-phase situation",
        "Edges connect related teams — and are rebuilt every phase from what actually happened",
        "A spatio-temporal graph neural network reads the histories and relations together",
      ].map((t) => ({
        text: t,
        options: { bullet: { code: "2022", indent: 10 }, breakLine: true },
      })),
      {
        x: 0.58, y: 2.30, w: 5.0, h: 2.2,
        fontSize: 11.5, fontFace: F, color: C.text,
        valign: "top", margin: 0, lineSpacingMultiple: 1.15,
      }
    );
    const edges = [
      ["Proximity", "teams physically near each other"],
      ["Combat", "teams that exchanged damage"],
      ["Co-pressure", "teams squeezed by the same zone edge"],
    ];
    edges.forEach(([head, body], i) => {
      const y = 1.80 + i * 0.98;
      addCard(s, 6.0, y, 3.6, 0.85);
      s.addText(head, {
        x: 6.18, y: y + 0.08, w: 3.25, h: 0.3,
        fontSize: 12, bold: true, fontFace: F, color: C.accent, margin: 0,
      });
      s.addText(body, {
        x: 6.18, y: y + 0.38, w: 3.25, h: 0.4,
        fontSize: 10.5, fontFace: F, color: C.text, margin: 0,
      });
    });
    s.addText(
      "What I want to find out: does where you are — and who you are entangled with — help predict survival, beyond your own stats?",
      {
        x: 0.4, y: 4.80, w: 9.2, h: 0.40,
        fontSize: 12, italic: true, bold: true, fontFace: F, color: C.accent, margin: 0,
      }
    );
    addFooter(s, 5, TOTAL);
  }

  const outPath = "PUBG_Research_Intro_EN_5min.pptx";
  await pres.writeFile({ fileName: outPath });
  console.log(`Wrote ${outPath}`);
}

build().catch((err) => {
  console.error(err);
  process.exit(1);
});
