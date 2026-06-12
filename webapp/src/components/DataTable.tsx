import { ReactNode } from "react";

export interface Column<T> {
  key: string;
  header: ReactNode;
  cell: (row: T, idx: number) => ReactNode;
  align?: "left" | "right" | "center";
  className?: string;
  numeric?: boolean;
}

interface Props<T> {
  columns: Column<T>[];
  rows: T[];
  rowKey?: (row: T, idx: number) => string;
  caption?: ReactNode;
  hoverHighlight?: boolean;
}

export default function DataTable<T>({
  columns,
  rows,
  rowKey,
  caption,
  hoverHighlight = true,
}: Props<T>) {
  return (
    <div className="overflow-x-auto rounded border border-rule">
      <table className="w-full border-collapse text-sm">
        {caption ? (
          <caption className="border-b border-rule bg-cream/40 px-3 py-2 text-left text-xs text-muted">
            {caption}
          </caption>
        ) : null}
        <thead>
          <tr className="border-b border-rule bg-cream/30">
            {columns.map((c) => (
              <th
                key={c.key}
                className={[
                  "jet whitespace-nowrap px-3 py-2 text-[11px] font-medium uppercase tracking-wider text-muted",
                  c.align === "right"
                    ? "text-right"
                    : c.align === "center"
                    ? "text-center"
                    : "text-left",
                  c.className ?? "",
                ].join(" ")}
              >
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr
              key={rowKey ? rowKey(row, idx) : String(idx)}
              className={
                "border-b border-rule/60 last:border-b-0 " +
                (hoverHighlight ? "hover:bg-cream/30" : "")
              }
            >
              {columns.map((c) => (
                <td
                  key={c.key}
                  className={[
                    "px-3 py-2",
                    c.numeric ? "numeric" : "",
                    c.align === "right"
                      ? "text-right"
                      : c.align === "center"
                      ? "text-center"
                      : "text-left",
                    c.className ?? "",
                  ].join(" ")}
                >
                  {c.cell(row, idx)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
