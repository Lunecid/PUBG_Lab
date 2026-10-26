import type { StrategyType } from "../lib/types";

const COLOR: Record<StrategyType, string> = {
  Hunter: "bg-accent text-paper",
  Sniper: "bg-warn text-paper",
  Traveler: "bg-accent2 text-paper",
  Camper: "bg-muted text-paper",
};

const HEX: Record<StrategyType, string> = {
  Hunter: "#b02a2a",
  Sniper: "#d97706",
  Traveler: "#3a4a5a",
  Camper: "#6b6558",
};

export const STRATEGY_COLORS = HEX;

interface Props {
  type: StrategyType;
  size?: "sm" | "md";
}

export default function StrategyBadge({ type, size = "sm" }: Props) {
  const sz = size === "md" ? "px-2 py-0.5 text-xs" : "px-1.5 py-0.5 text-[10px]";
  return (
    <span
      className={["jet inline-block rounded font-medium uppercase tracking-wider", sz, COLOR[type]].join(" ")}
    >
      {type}
    </span>
  );
}
