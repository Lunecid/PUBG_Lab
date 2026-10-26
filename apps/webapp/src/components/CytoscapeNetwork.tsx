import { useEffect, useRef } from "react";
import cytoscape, { type ElementDefinition, type Core } from "cytoscape";
import coseBilkent from "cytoscape-cose-bilkent";
import type { MatchEdge, MatchNode } from "../lib/types";

if (typeof (cytoscape as unknown as { _registered?: boolean })._registered !== "boolean") {
  cytoscape.use(coseBilkent);
  (cytoscape as unknown as { _registered: boolean })._registered = true;
}

export type ColorBy =
  | "archetype"
  | "movement"
  | "position"
  | "community"
  | "rank";

interface Props {
  nodes: MatchNode[];
  edges: MatchEdge[];
  colorBy: ColorBy;
  height?: number;
  onNodeClick?: (n: MatchNode) => void;
}

const ARC_COLOR: Record<string, string> = {
  outer_farmer: "#15803d",
  combat_participant: "#b91c1c",
  neutral: "#a8a29e",
};
const MV_COLOR: Record<string, string> = {
  sedentary: "#1d4ed8",
  mid: "#a8a29e",
  mobile: "#b91c1c",
};
const POS_COLOR: Record<string, string> = {
  core: "#b02a2a",
  mid: "#d97706",
  peripheral: "#1d4ed8",
  isolated: "#6b6558",
};

function communityColor(c: number): string {
  // golden-ratio hue cycling
  const h = ((c * 137.508) % 360);
  return `hsl(${h.toFixed(0)} 60% 45%)`;
}

function rankColor(rank: number | null, max: number): string {
  if (rank === null) return "#a8a29e";
  // 1 = green, max = red
  const t = Math.max(0, Math.min(1, (rank - 1) / Math.max(1, max - 1)));
  const r = Math.round(20 + t * 200);
  const g = Math.round(160 - t * 130);
  return `rgb(${r} ${g} 60)`;
}

function colorForNode(n: MatchNode, colorBy: ColorBy, maxRank: number): string {
  switch (colorBy) {
    case "archetype":
      return ARC_COLOR[n.archetype] ?? "#6b6558";
    case "movement":
      return MV_COLOR[n.movement_label] ?? "#6b6558";
    case "position":
      return POS_COLOR[n.position] ?? "#6b6558";
    case "community":
      return communityColor(n.community);
    case "rank":
      return rankColor(n.rank, maxRank);
  }
}

export default function CytoscapeNetwork({
  nodes,
  edges,
  colorBy,
  height = 540,
  onNodeClick,
}: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<Core | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const maxRank = Math.max(
      1,
      ...nodes.map((n) => (n.rank ?? 0)),
    );

    const elements: ElementDefinition[] = [
      ...nodes.map((n) => ({
        data: {
          id: `n${n.id}`,
          label: `T${n.team}`,
          rank: n.rank,
          degree: n.degree,
          color: colorForNode(n, colorBy, maxRank),
          // size scaled by degree (weighted_deg ish)
          size: 12 + Math.min(36, Math.sqrt(Math.max(1, n.degree * 4))),
        },
      })),
      ...edges.map((e, i) => ({
        data: {
          id: `e${i}`,
          source: `n${e.source}`,
          target: `n${e.target}`,
          weight: Math.max(1, Math.log10(Math.max(1, e.dmg)) * 1.6),
          dmg: e.dmg,
          events: e.events,
        },
      })),
    ];

    cyRef.current?.destroy();
    cyRef.current = cytoscape({
      container: containerRef.current,
      elements,
      style: [
        {
          selector: "node",
          style: {
            "background-color": "data(color)",
            "border-color": "#1a1a1a",
            "border-width": 1,
            label: "data(label)",
            "font-size": "10px",
            "font-family": "IBM Plex Mono, monospace",
            color: "#1a1a1a",
            "text-valign": "center",
            "text-halign": "center",
            "text-outline-color": "#fefcf7",
            "text-outline-width": 2,
            width: "data(size)",
            height: "data(size)",
          },
        },
        {
          selector: "edge",
          style: {
            "curve-style": "bezier",
            "line-color": "#a8a29e",
            "line-opacity": 0.55,
            width: "data(weight)",
          },
        },
        {
          selector: "node:selected",
          style: {
            "border-color": "#b02a2a",
            "border-width": 3,
          },
        },
      ],
      layout: {
        name: "cose-bilkent",
        animate: false,
        idealEdgeLength: 110,
        nodeRepulsion: 6000,
        gravity: 0.25,
        randomize: true,
      } as unknown as cytoscape.LayoutOptions,
      wheelSensitivity: 0.2,
    });

    if (onNodeClick) {
      cyRef.current.on("tap", "node", (evt) => {
        const id: string = evt.target.id();
        const teamId = parseInt(id.replace("n", ""), 10);
        const n = nodes.find((x) => x.id === teamId);
        if (n) onNodeClick(n);
      });
    }

    return () => {
      cyRef.current?.destroy();
      cyRef.current = null;
    };
  }, [nodes, edges, colorBy, onNodeClick]);

  return (
    <div
      ref={containerRef}
      style={{ height, width: "100%", background: "#fefcf7" }}
      className="rounded border border-rule"
    />
  );
}
