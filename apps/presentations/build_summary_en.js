// =============================================================================
// apps/presentations/build_summary_en.js
// -----------------------------------------------------------------------------
// Renders PUBG_Research_Summary_EN.pptx — a short, simple English overview of
// the completed research (with measured results, unlike the proposal deck).
//
//   node build_summary_en.js
//
// 10 slides: title / problem / RQs / data / STDB / task / model / results /
// robustness / takeaways.  All numbers verified against
// docs/RESEARCH_STATUS_2026-06-10.md and model_stgcn checkpoints.
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
  good:     "1D6F42",
};

const F = "Calibri";
const FOOTER = "Predicting Team Elimination in PUBG — Research Summary";

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

function cardWithHead(slide, x, y, w, h, head, lines, opts = {}) {
  addCard(slide, x, y, w, h);
  slide.addText(head, {
    x: x + 0.18, y: y + 0.14, w: w - 0.36, h: 0.32,
    fontSize: opts.headSize || 13, bold: true, fontFace: F,
    color: C.accent, margin: 0,
  });
  slide.addText(
    lines.map((t) => ({
      text: t,
      options: { bullet: { code: "2022", indent: 10 }, breakLine: true },
    })),
    {
      x: x + 0.18, y: y + 0.50, w: w - 0.36, h: h - 0.62,
      fontSize: opts.bodySize || 11, fontFace: F, color: C.text,
      valign: "top", margin: 0, lineSpacingMultiple: 1.12,
    }
  );
}

async function build() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  const TOTAL = 10;

  // ---------------------------------------------------------------- 1 · TITLE
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    s.addShape("rect", {
      x: 0, y: 0, w: 10, h: 0.18, fill: { color: C.accent },
    });
    s.addText("RESEARCH SUMMARY · JUNE 2026", {
      x: 0.7, y: 1.10, w: 8.6, h: 0.3,
      fontSize: 11, fontFace: F, bold: true, color: C.subtle,
      charSpacing: 4, margin: 0,
    });
    s.addText("Predicting Team Elimination in PUBG", {
      x: 0.7, y: 1.50, w: 8.6, h: 0.65,
      fontSize: 30, fontFace: F, bold: true, color: C.accent, margin: 0,
    });
    s.addText(
      "Modeling game telemetry as a spatio-temporal database\nfor phase-level survival prediction",
      {
        x: 0.7, y: 2.25, w: 8.6, h: 0.75,
        fontSize: 15, fontFace: F, italic: true, color: C.subtle, margin: 0,
      }
    );
    s.addShape("rect", {
      x: 0.7, y: 3.25, w: 1.6, h: 0.03,
      fill: { color: C.accent }, line: { color: C.accent },
    });
    s.addText("Seongeun Baek", {
      x: 0.7, y: 3.45, w: 8.6, h: 0.3,
      fontSize: 14, fontFace: F, bold: true, color: C.text, margin: 0,
    });
    s.addText("Data Science · Pusan National University", {
      x: 0.7, y: 3.75, w: 8.6, h: 0.3,
      fontSize: 11, fontFace: F, color: C.subtle, margin: 0,
    });
  }

  // -------------------------------------------------------------- 2 · PROBLEM
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    addSlideTitle(s, "BACKGROUND", "The game, and the question",
      "PUBG is a 100-player battle royale played in squads.");
    cardWithHead(s, 0.4, 1.75, 4.5, 2.85, "PUBG in one minute", [
      "Up to ~28 squads parachute onto one large map",
      "A circular safe zone shrinks in discrete phases; staying outside drains health",
      "Eliminated teams are out for good — the last team standing wins",
      "Every position, shot, and zone change is recorded as telemetry",
    ]);
    cardWithHead(s, 5.1, 1.75, 4.5, 2.85, "Our question", [
      "Most prior work predicts the final placement after the match is over",
      "We predict during the match: which teams will be eliminated in the NEXT zone phase?",
      "Hard problem: ~32% of alive teams die each phase — survival depends on space, time, and other teams",
    ]);
    addFooter(s, 2, TOTAL);
  }

  // ------------------------------------------------------ 3 · RESEARCH QUESTIONS
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    addSlideTitle(s, "RESEARCH QUESTIONS", "Three questions drive the work");
    const rqs = [
      ["RQ1 · Representation",
       "How can raw telemetry be organized as a spatio-temporal database — moving teams, shrinking zones, combat events, all aligned to game phases?"],
      ["RQ2 · Dynamics",
       "As the safe zone contracts, how do team trajectories and survival conditions change from one phase to the next?"],
      ["RQ3 · Prediction",
       "Do spatio-temporal features and dynamic team-to-team relations improve next-phase elimination prediction beyond strong tabular baselines?"],
    ];
    rqs.forEach(([head, body], i) => {
      const y = 1.65 + i * 1.18;
      addCard(s, 0.4, y, 9.2, 1.05);
      s.addText(head, {
        x: 0.62, y: y + 0.12, w: 2.4, h: 0.8,
        fontSize: 13, bold: true, fontFace: F, color: C.accent,
        valign: "middle", margin: 0,
      });
      s.addText(body, {
        x: 3.15, y: y + 0.10, w: 6.25, h: 0.85,
        fontSize: 11.5, fontFace: F, color: C.text,
        valign: "middle", margin: 0,
      });
    });
    addFooter(s, 3, TOTAL);
  }

  // ----------------------------------------------------------------- 4 · DATA
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    addSlideTitle(s, "DATA", "Official telemetry at scale",
      "PUBG official API → PostgreSQL + PostGIS");
    const stats = [
      ["27,491", "matches fetched from the official API"],
      ["21,824", "model-ready squad matches (5 maps, TPP)"],
      ["~1.37M", "team-phase observations after reconstruction"],
      ["115M+", "raw damage events processed"],
    ];
    stats.forEach(([num, label], i) => {
      const x = 0.4 + (i % 2) * 4.7;
      const y = 1.75 + Math.floor(i / 2) * 1.32;
      addCard(s, x, y, 4.5, 1.18);
      s.addText(num, {
        x: x + 0.2, y: y + 0.12, w: 1.9, h: 0.95,
        fontSize: 26, bold: true, fontFace: F, color: C.accent,
        valign: "middle", margin: 0,
      });
      s.addText(label, {
        x: x + 2.1, y: y + 0.12, w: 2.3, h: 0.95,
        fontSize: 11, fontFace: F, color: C.text,
        valign: "middle", margin: 0,
      });
    });
    s.addText(
      "Split by match — 17,460 train / 2,182 validation / 2,182 test. No match appears in two sets.",
      {
        x: 0.4, y: 4.55, w: 9.2, h: 0.35,
        fontSize: 11, italic: true, fontFace: F, color: C.subtle, margin: 0,
      }
    );
    addFooter(s, 4, TOTAL);
  }

  // ----------------------------------------------------------- 5 · STEP 1 STDB
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    addSlideTitle(s, "STEP 1 · REPRESENTATION", "A spatio-temporal database of the match",
      "Raw logs are reconstructed into three spatio-temporal primitives.");
    const prims = [
      ["Moving Objects", "Player and team-center trajectories sampled over time"],
      ["Dynamic Regions", "Safe and blue zones whose center and radius change every phase"],
      ["Spatial Events", "Kills, damage, knock-downs — each with a time and a location"],
    ];
    prims.forEach(([head, body], i) => {
      const x = 0.4 + i * 3.13;
      addCard(s, x, 1.80, 2.95, 1.85);
      s.addText(head, {
        x: x + 0.18, y: 1.95, w: 2.6, h: 0.35,
        fontSize: 13, bold: true, fontFace: F, color: C.accent, margin: 0,
      });
      s.addText(body, {
        x: x + 0.18, y: 2.35, w: 2.6, h: 1.15,
        fontSize: 11, fontFace: F, color: C.text, valign: "top", margin: 0,
      });
    });
    addCard(s, 0.4, 3.90, 9.2, 0.95);
    s.addText(
      [
        { text: "Key idea — phase alignment.  ", options: { bold: true, color: C.accent } },
        { text: "The game's zone phases become the unit of time: 66 features per team per phase (position vs. zone, mobility, combat pressure, resources) computed with PostGIS queries.",
          options: { color: C.text } },
      ],
      {
        x: 0.62, y: 4.00, w: 8.8, h: 0.75,
        fontSize: 11.5, fontFace: F, valign: "middle", margin: 0,
      }
    );
    addFooter(s, 5, TOTAL);
  }

  // ----------------------------------------------------------- 6 · STEP 2 TASK
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    addSlideTitle(s, "STEP 2 · TASK", "Next-phase elimination hazard",
      "A discrete-time survival problem, by construction free of future leakage.");
    addCard(s, 0.4, 1.80, 9.2, 1.05);
    s.addText(
      "h(team, k)  =  P( eliminated in phase k+1  |  alive at phase k, history ≤ k )",
      {
        x: 0.62, y: 1.92, w: 8.8, h: 0.80,
        fontSize: 13.5, fontFace: "Courier New", color: C.accent,
        valign: "middle", align: "center", margin: 0,
      }
    );
    cardWithHead(s, 0.4, 3.10, 4.5, 1.75, "Why this task", [
      "Answerable live, during the match",
      "Actionable: “who is in danger next” rather than “who will win eventually”",
      "Censoring and class balance handled by the survival framing",
    ]);
    cardWithHead(s, 5.1, 3.10, 4.5, 1.75, "Leakage control", [
      "Model observes phases ≤ k, predicts phase k+1 — never sees the future",
      "Evaluated on 2,182 fully unseen test matches",
      "Quality measured by AUC / AP and probability calibration",
    ]);
    addFooter(s, 6, TOTAL);
  }

  // ---------------------------------------------------------- 7 · STEP 3 MODEL
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    addSlideTitle(s, "STEP 3 · MODEL", "A dynamic graph between teams",
      "DynamicSTGNN — the match as a graph that rewires every phase.");
    cardWithHead(s, 0.4, 1.75, 5.4, 2.85, "DynamicSTGNN", [
      "Each team = a node with its 66-feature phase sequence",
      "Temporal gated convolution reads each team's history",
      "Graph convolution mixes information between related teams",
      "Edges are rebuilt every phase from what actually happened",
    ]);
    const edges = [
      ["Proximity", "teams physically near each other"],
      ["Combat", "teams that exchanged damage"],
      ["Co-pressure", "teams squeezed by the same zone edge"],
    ];
    edges.forEach(([head, body], i) => {
      const y = 1.75 + i * 0.98;
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
      "Baselines: gradient boosting on the same features (no graph) · the same network with all edges removed.",
      {
        x: 0.4, y: 4.75, w: 9.2, h: 0.35,
        fontSize: 11, italic: true, fontFace: F, color: C.subtle, margin: 0,
      }
    );
    addFooter(s, 7, TOTAL);
  }

  // -------------------------------------------------------------- 8 · RESULTS
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    addSlideTitle(s, "RESULTS", "Team relations carry real signal",
      "Aligned test set: 134,245 team-phase predictions from 2,182 unseen matches.");
    const headOpts = {
      fill: { color: C.accent }, color: "FFFFFF", bold: true,
      fontFace: F, fontSize: 12, align: "center", valign: "middle",
    };
    const cell = (t, extra = {}) => ({
      text: t,
      options: {
        fontFace: F, fontSize: 12, align: "center", valign: "middle",
        color: C.text, ...extra,
      },
    });
    s.addTable(
      [
        [
          { text: "Model", options: { ...headOpts, align: "left" } },
          { text: "AUC ↑", options: headOpts },
          { text: "AP ↑", options: headOpts },
          { text: "Calibration error ↓", options: headOpts },
        ],
        [cell("Gradient boosting (tabular)", { align: "left" }),
          cell("0.7651"), cell("0.6626"), cell("0.0062")],
        [cell("DynamicSTGNN — no graph", { align: "left" }),
          cell("0.7681"), cell("0.6695"), cell("0.0039")],
        [cell("DynamicSTGNN — full graph", { align: "left", bold: true, color: C.good }),
          cell("0.7782", { bold: true, color: C.good }),
          cell("0.6808", { bold: true, color: C.good }),
          cell("0.0015", { bold: true, color: C.good })],
      ],
      {
        x: 0.7, y: 1.90, w: 8.6, rowH: 0.46,
        border: { type: "solid", color: C.cardEdge, pt: 0.75 },
        colW: [3.8, 1.6, 1.6, 1.6],
      }
    );
    addCard(s, 0.7, 4.05, 8.6, 0.85);
    s.addText(
      [
        { text: "Every gain is statistically significant ", options: { bold: true, color: C.accent } },
        { text: "(match-paired Wilcoxon, p < 10⁻²² on all metrics) — and the full model is also the best-calibrated: predicted risk 0.326 vs. actual rate 0.325.",
          options: { color: C.text } },
      ],
      {
        x: 0.92, y: 4.15, w: 8.2, h: 0.65,
        fontSize: 11.5, fontFace: F, valign: "middle", margin: 0,
      }
    );
    addFooter(s, 8, TOTAL);
  }

  // ----------------------------------------------------------- 9 · ROBUSTNESS
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    addSlideTitle(s, "ROBUSTNESS", "Why we believe the result");
    const checks = [
      ["15 / 15", "The graph beats the no-graph control in every run — 3 seeds × 5 maps"],
      ["AUC −0.065", "Shuffling who-relates-to-whom destroys the gain → the signal lives in the real relations"],
      ["+0.099 AUC", "The reconstructed features beat prior-work-style features — representation is the biggest single win"],
      ["ECE 0.0015", "Predicted probabilities match observed elimination rates — usable as real risk scores"],
    ];
    checks.forEach(([num, label], i) => {
      const x = 0.4 + (i % 2) * 4.7;
      const y = 1.65 + Math.floor(i / 2) * 1.55;
      addCard(s, x, y, 4.5, 1.40);
      s.addText(num, {
        x: x + 0.2, y: y + 0.12, w: 1.75, h: 1.15,
        fontSize: 19, bold: true, fontFace: F, color: C.accent,
        valign: "middle", margin: 0,
      });
      s.addText(label, {
        x: x + 1.95, y: y + 0.12, w: 2.45, h: 1.15,
        fontSize: 10.5, fontFace: F, color: C.text,
        valign: "middle", margin: 0,
      });
    });
    addFooter(s, 9, TOTAL);
  }

  // ------------------------------------------------------------ 10 · TAKEAWAYS
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    addSlideTitle(s, "TAKEAWAYS", "What the research says");
    const takes = [
      ["1", "Representation first.",
       "Reconstructing telemetry into a phase-aligned spatio-temporal database delivers the largest gain — before any model choice."],
      ["2", "Team relations matter.",
       "Dynamic proximity / combat / co-pressure relations between teams add a consistent, verified predictive signal on top of strong features."],
      ["3", "Predictive, not causal.",
       "The model identifies who is at risk — it does not prove that changing strategy would change the outcome."],
    ];
    takes.forEach(([n, head, body], i) => {
      const y = 1.60 + i * 0.98;
      addCard(s, 0.4, y, 9.2, 0.85);
      s.addText(n, {
        x: 0.62, y: y + 0.10, w: 0.45, h: 0.65,
        fontSize: 20, bold: true, fontFace: F, color: C.accent,
        valign: "middle", margin: 0,
      });
      s.addText(
        [
          { text: head + "  ", options: { bold: true, color: C.accent } },
          { text: body, options: { color: C.text } },
        ],
        {
          x: 1.15, y: y + 0.10, w: 8.25, h: 0.65,
          fontSize: 11.5, fontFace: F, valign: "middle", margin: 0,
        }
      );
    });
    s.addText(
      "Status: data, model, and main results complete · paper in final writing · target venue: ICE 2027",
      {
        x: 0.4, y: 4.68, w: 9.2, h: 0.35,
        fontSize: 11, italic: true, fontFace: F, color: C.subtle, margin: 0,
      }
    );
    addFooter(s, 10, TOTAL);
  }

  const outPath = "PUBG_Research_Summary_EN.pptx";
  await pres.writeFile({ fileName: outPath });
  console.log(`Wrote ${outPath}`);
}

build().catch((err) => {
  console.error(err);
  process.exit(1);
});
