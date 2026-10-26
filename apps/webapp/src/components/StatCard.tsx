import { ReactNode } from "react";

interface Props {
  label: string;
  value: ReactNode;
  unit?: string;
  hint?: string;
  size?: "md" | "lg";
}

export default function StatCard({ label, value, unit, hint, size = "md" }: Props) {
  return (
    <div className="rounded border border-rule bg-cream/40 p-4">
      <div className="jet text-[10px] uppercase tracking-wider text-muted">
        {label}
      </div>
      <div
        className={
          "numeric mt-1 leading-none " +
          (size === "lg" ? "text-3xl font-medium" : "text-2xl")
        }
      >
        {value}
        {unit ? (
          <span className="ml-1 text-base font-normal text-muted">{unit}</span>
        ) : null}
      </div>
      {hint ? <div className="mt-2 text-xs text-muted">{hint}</div> : null}
    </div>
  );
}
