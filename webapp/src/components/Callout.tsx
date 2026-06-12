import { ReactNode } from "react";

interface Props {
  children: ReactNode;
  variant?: "default" | "key" | "warn";
  label?: string;
}

export default function Callout({
  children,
  variant = "default",
  label,
}: Props) {
  const styles =
    variant === "key"
      ? "border-l-4 border-accent bg-accent/[0.04] text-ink"
      : variant === "warn"
      ? "border-l-4 border-warn bg-warn/[0.06] text-ink"
      : "border-l-4 border-rule bg-cream/40 text-ink";
  return (
    <div className={"rounded-r px-4 py-3 " + styles}>
      {label ? (
        <div className="jet mb-1 text-[10px] uppercase tracking-wider text-muted">
          {label}
        </div>
      ) : null}
      <div className="text-sm leading-relaxed">{children}</div>
    </div>
  );
}
