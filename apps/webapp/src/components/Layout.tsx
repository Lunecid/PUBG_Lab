import { ReactNode } from "react";
import Nav from "./Nav";

interface Props {
  children: ReactNode;
}

export default function Layout({ children }: Props) {
  return (
    <div className="min-h-screen bg-paper text-ink">
      <header className="sticky top-0 z-30 border-b border-rule bg-paper/90 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
          <a href="/" className="flex items-baseline gap-3 hover:no-underline">
            <span className="news text-xl font-semibold tracking-tight">
              PUBG Network Atlas
            </span>
            <span className="jet text-[10px] uppercase text-muted">
              Baltic_Main · squad · n=1,000
            </span>
          </a>
          <Nav />
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-8">{children}</main>

      <footer className="mt-16 border-t border-rule">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-6 text-xs text-muted">
          <div>
            데이터 ·{" "}
            <span className="jet">Baltic_Main_squad_n1000</span> · 2026
          </div>
          <div>
            팀 전투 네트워크 분석 · Hunter / Sniper / Traveler / Camper
          </div>
        </div>
      </footer>
    </div>
  );
}
