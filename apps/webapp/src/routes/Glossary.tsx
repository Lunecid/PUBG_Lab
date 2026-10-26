import { useGlossary } from "../lib/data";
import Section from "../components/Section";

export default function Glossary() {
  const g = useGlossary();
  if (g.isLoading || !g.data) return <div className="text-muted">불러오는 중…</div>;

  return (
    <div className="space-y-8">
      <header className="space-y-2">
        <div className="jet text-[11px] uppercase tracking-wider text-accent">
          §2.1 · glossary
        </div>
        <h1 className="news text-3xl font-semibold leading-tight">용어집</h1>
        <p className="max-w-prose text-sm text-muted">
          분석에 사용한 14개 핵심 용어. 일반 그래프 정의와 PUBG 적용을 분리해 설명한다.
        </p>
      </header>

      {/* 사이드 인덱스 */}
      <nav className="rounded border border-rule bg-cream/30 p-3">
        <div className="jet mb-2 text-[10px] uppercase text-muted">목차</div>
        <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
          {g.data.map((e) => (
            <a key={e.term} href={`#${e.term}`} className="text-muted hover:text-accent">
              {e.term}
            </a>
          ))}
        </div>
      </nav>

      <Section title="용어 정의" kicker="terms">
        <div className="grid gap-3">
          {g.data.map((e) => (
            <article
              key={e.term}
              id={e.term}
              className="scroll-mt-20 rounded border border-rule bg-paper p-4"
            >
              <h3 className="news text-lg font-medium">{e.term}</h3>
              <div className="mt-2 grid gap-2 text-sm md:grid-cols-2">
                <div>
                  <div className="jet text-[10px] uppercase text-muted">general</div>
                  <div>{e.general}</div>
                </div>
                <div>
                  <div className="jet text-[10px] uppercase text-muted">PUBG 적용</div>
                  <div>{e.pubg}</div>
                </div>
              </div>
            </article>
          ))}
        </div>
      </Section>
    </div>
  );
}
