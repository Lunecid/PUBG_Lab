import { ReactNode } from "react";

interface Props {
  children: ReactNode;
  caption?: ReactNode;
}

export default function FormulaBlock({ children, caption }: Props) {
  return (
    <div className="rounded border border-rule bg-cream/30 p-4">
      <pre className="jet whitespace-pre-wrap break-words text-[13px] leading-relaxed text-ink">
        {children}
      </pre>
      {caption ? (
        <div className="mt-2 text-xs text-muted">{caption}</div>
      ) : null}
    </div>
  );
}
