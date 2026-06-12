import { NavLink } from "react-router-dom";

const LINKS: { to: string; label: string }[] = [
  { to: "/", label: "개요" },
  { to: "/network", label: "네트워크" },
  { to: "/strategy", label: "4유형" },
  { to: "/phases", label: "페이즈" },
  { to: "/hypotheses", label: "가설" },
  { to: "/explorer", label: "매치 탐색" },
  { to: "/methods", label: "방법" },
  { to: "/glossary", label: "용어" },
];

export default function Nav() {
  return (
    <nav className="flex flex-wrap items-center gap-x-1 gap-y-1 text-sm">
      {LINKS.map((l) => (
        <NavLink
          key={l.to}
          to={l.to}
          end={l.to === "/"}
          className={({ isActive }) =>
            [
              "rounded px-2 py-1 transition-colors hover:bg-cream hover:no-underline",
              isActive
                ? "bg-cream text-ink font-medium"
                : "text-muted",
            ].join(" ")
          }
        >
          {l.label}
        </NavLink>
      ))}
    </nav>
  );
}
