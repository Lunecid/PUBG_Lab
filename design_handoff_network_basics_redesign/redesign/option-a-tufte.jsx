// Option A — Editorial Tufte
// Wide page with side-margin notes, sparse rules, small inline figures.
// Hierarchy via type weight + scale, not borders/cards. Calm and precise.

const A = {
  paper: '#fbf8ee',
  ink: '#1d1a14',
  muted: '#7a715f',
  faint: '#b8af9a',
  rule: '#dcd2bc',
  accent: '#a23232',
  page_w: 1280,
  // 3-col: main 720 / gap 60 / margin notes 320
  main_w: 720,
  margin_w: 320,
  gap: 60,
};

function MarginNote({ children, top = 0 }) {
  return (
    <div style={{
      position: 'absolute', left: A.main_w + A.gap, top, width: A.margin_w,
      fontFamily: '"Newsreader", "Noto Serif KR", Georgia, serif',
      fontSize: 13, lineHeight: 1.55, color: A.muted, fontStyle: 'italic',
    }}>{children}</div>
  );
}

// Custom inline figure: degree distribution as Tufte sparkbars
function TufteSparkbars({ width = 220, height = 46 }) {
  const counts = [3, 8, 16, 22, 19, 14, 9, 5, 3, 1, 0.5, 0.2];
  const max = Math.max(...counts);
  const bw = width / counts.length;
  return (
    <svg width={width} height={height + 14} viewBox={`0 0 ${width} ${height + 14}`}>
      {counts.map((c, i) => {
        const h = (c / max) * height;
        const isMean = i === 3;
        return (
          <g key={i}>
            <rect x={i * bw + 1} y={height - h} width={bw - 2} height={h}
              fill={isMean ? A.accent : A.ink} />
          </g>
        );
      })}
      <text x={3 * bw + bw / 2} y={height + 11} fontSize="9" textAnchor="middle"
        fill={A.accent} fontStyle="italic"
        fontFamily="Newsreader, serif">평균 3.2</text>
    </svg>
  );
}

// Custom centrality comparison: small horizontal bars per centrality, all on one figure
function CentralityRanks() {
  const centrals = [
    { name: 'Degree',         k: '직접 교전 다양성',    h: 'H1·H2' },
    { name: 'Closeness',      k: '그래프 중앙 위치',    h: '—' },
    { name: 'Betweenness',    k: '클러스터 중개자',     h: 'H2 핵심', star: true },
    { name: 'Eigenvector',    k: '강팀과의 연결',       h: '—' },
    { name: 'PageRank',       k: 'Eigenvector + 텔레포트', h: '—' },
    { name: 'Local clustering', k: '삼각 교전 밀도',     h: 'H2 보조' },
  ];
  return (
    <table style={{
      width: '100%', borderCollapse: 'collapse', marginTop: 4,
      fontFamily: '"Newsreader", "Noto Serif KR", Georgia, serif', fontSize: 14,
    }}>
      <thead>
        <tr style={{ borderBottom: `1px solid ${A.ink}` }}>
          <th style={{ textAlign: 'left', padding: '8px 0', fontWeight: 500, fontSize: 11, letterSpacing: '.04em', textTransform: 'uppercase', color: A.muted, width: '24%' }}>지표</th>
          <th style={{ textAlign: 'left', padding: '8px 0', fontWeight: 500, fontSize: 11, letterSpacing: '.04em', textTransform: 'uppercase', color: A.muted }}>PUBG 의미</th>
          <th style={{ textAlign: 'left', padding: '8px 0', fontWeight: 500, fontSize: 11, letterSpacing: '.04em', textTransform: 'uppercase', color: A.muted, width: '20%' }}>가설</th>
        </tr>
      </thead>
      <tbody>
        {centrals.map((c) => (
          <tr key={c.name} style={{ borderBottom: `1px solid ${A.rule}` }}>
            <td style={{ padding: '12px 0', fontWeight: 500, color: c.star ? A.accent : A.ink }}>
              {c.name}{c.star ? ' ★' : ''}
            </td>
            <td style={{ padding: '12px 0', color: A.ink }}>{c.k}</td>
            <td style={{ padding: '12px 0', color: A.muted, fontSize: 13 }}>{c.h}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function OptionATufte() {
  const C = A;
  return (
    <div style={{
      width: 1280, minHeight: 2200, background: C.paper, color: C.ink,
      fontFamily: '"Inter", "Noto Sans KR", system-ui, sans-serif',
      padding: '72px 80px 80px',
      position: 'relative',
    }}>
      {/* Header */}
      <header style={{ marginBottom: 72, position: 'relative' }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 16, fontSize: 11,
                      fontFamily: '"JetBrains Mono", monospace', color: C.muted,
                      letterSpacing: '.08em', textTransform: 'uppercase', marginBottom: 24 }}>
          <span>PUBG · Network</span>
          <span style={{ width: 24, borderTop: `1px solid ${C.rule}` }} />
          <span>Reference §0</span>
          <span style={{ width: 24, borderTop: `1px solid ${C.rule}` }} />
          <span>가설 검증 토대</span>
        </div>

        <h1 className="news" style={{
          fontSize: 64, lineHeight: 1.04, fontWeight: 500,
          letterSpacing: '-0.02em', margin: 0, maxWidth: 900,
        }}>
          한 매치를 그래프로 본다는 것
        </h1>
        <p className="news" style={{
          marginTop: 22, fontSize: 22, lineHeight: 1.45, fontStyle: 'italic',
          color: C.muted, maxWidth: 720, fontWeight: 400,
        }}>
          26팀이 200km² 위에 흩어져 1,440초 동안 싸운다. 그 모든 교전을
          하나의 무방향 가중 그래프 <span style={{ fontFamily: '"JetBrains Mono"', fontStyle: 'normal' }}>G = (V, E, w)</span> 로
          요약하고, 이 위에서 H1·H2·H3 가설을 검증한다.
        </p>

        {/* hairline + meta */}
        <div style={{ marginTop: 40, display: 'flex', gap: 40, paddingTop: 16,
                      borderTop: `1px solid ${C.ink}` }}>
          {[
            ['Dataset',   'Baltic_Main · squad'],
            ['Matches',   '3,595'],
            ['Teams',     '94,964'],
            ['Weight',    'damage (default)'],
            ['Updated',   '2026·04·27'],
          ].map(([k, v]) => (
            <div key={k}>
              <div style={{ fontSize: 10, letterSpacing: '.1em', textTransform: 'uppercase', color: C.muted, fontFamily: '"JetBrains Mono"' }}>{k}</div>
              <div className="numeric" style={{ fontSize: 15, marginTop: 4, fontWeight: 500 }}>{v}</div>
            </div>
          ))}
        </div>
      </header>

      {/* ─── §1 Definition ─── */}
      <section style={{ position: 'relative', marginBottom: 96 }}>
        <SectionHead num="§1" kicker="Definition" title="그래프" />

        <div style={{ width: A.main_w }}>
          <p className="news" style={{ fontSize: 19, lineHeight: 1.62, marginTop: 0 }}>
            한 매치의 모든 팀을 노드로, 두 팀 사이의 교전을 엣지로 두는
            무방향 가중 그래프를 구성한다. 모든 후속 분석은 이 그래프 위에서 계산된다.
          </p>

          <dl style={{ marginTop: 28, fontSize: 16, lineHeight: 1.65 }}>
            <DefRow term="V" detail="매치 참가 팀 집합. 보통 |V| ≈ 26–28." />
            <DefRow term="E" detail="두 팀 (u, v) 사이에 damage event 가 한 번이라도 있으면 추가되는 무방향 엣지. 자기 팀 데미지(아군 오사)는 제외." />
            <DefRow term="w(u,v)" detail="(u, v) 가 주고받은 누적 damage 합. weight_mode = count → 이벤트 수, binary → 1." />
          </dl>

          <div style={{
            marginTop: 36, padding: '24px 0', borderTop: `1px solid ${C.rule}`, borderBottom: `1px solid ${C.rule}`,
            textAlign: 'center', fontSize: 18,
          }}>
            <TeX tex={"G = (V, E),\\quad w(u,v) = \\sum_{e \\in \\mathrm{events}_{uv}} \\mathrm{dmg}_e"} />
          </div>

          <p style={{ marginTop: 24, fontSize: 12, color: C.muted, fontFamily: '"JetBrains Mono"', lineHeight: 1.6 }}>
            데이터 소스 — match telemetry damage events (attacker, victim, dmg, elapsed).
            자살 / Bluezone 제외, exclude_self_team=True.
            <br />구현 — network_analysis/team_network.py :: _events_to_team_graph()
          </p>
        </div>

        <MarginNote top={64}>
          “노드는 팀, 엣지는 교전, 가중치는 데미지 강도.” 이 한 줄로
          <br />
          <span style={{ color: C.ink, fontStyle: 'normal' }}>— degree → 직접 부딪힌 팀 수</span><br />
          <span style={{ color: C.ink, fontStyle: 'normal' }}>— weighted_deg → 받고 준 데미지 총합</span><br />
          <span style={{ color: C.ink, fontStyle: 'normal' }}>— edge weight → 두 팀의 교전 강도</span><br />
          <span style={{ color: C.ink, fontStyle: 'normal' }}>— community → 같이 얽혀 싸운 팀 군집</span><br />
          이 모두 정의된다.
        </MarginNote>

        {/* Inline figure: a single typical match's graph at right */}
        <MarginNote top={320}>
          <div style={{ fontFamily: 'Inter', fontStyle: 'normal', fontSize: 11, letterSpacing: '.08em', textTransform: 'uppercase', color: C.muted }}>
            Fig. 1 · 매치 #00036c37
          </div>
          <div style={{ background: C.paper, marginTop: 8, border: `1px solid ${C.rule}`, padding: 8 }}>
            <MiniNetwork width={300} height={220} seed={11} n={26}
              bg={C.paper} edgeColor={C.faint}
              nodeColors={[C.ink, C.accent, C.muted]}
              nodeStroke={C.paper} />
          </div>
          <div style={{ fontFamily: 'Inter', fontStyle: 'normal', marginTop: 10, fontSize: 12, color: C.muted, lineHeight: 1.5 }}>
            26 팀 · 44 엣지 · Q = 0.657. 색은 Louvain community.
          </div>
        </MarginNote>
      </section>

      {/* ─── §2 Dataset stats — Tufte table style ─── */}
      <section style={{ position: 'relative', marginBottom: 96 }}>
        <SectionHead num="§2" kicker="Aggregate" title="이 데이터셋의 그래프는 평균적으로 어떻게 생겼는가" />

        <div style={{ width: A.main_w }}>
          <p className="news" style={{ fontSize: 17, lineHeight: 1.6, color: C.muted, fontStyle: 'italic', marginTop: 0 }}>
            3,595 매치 평균. Baltic_Main · squad. 가중치 = damage.
          </p>

          {/* Stat table — left-aligned big numbers, no boxes */}
          <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 32 }}>
            <tbody>
              {[
                ['|V|',        '26.4',   '매치당 평균 팀 수',                      false],
                ['|E|',        '42.6',   '실제 교전 쌍',                          false],
                ['density',    '0.131',  '가능한 343쌍 중 13.1%',                  false],
                ['avg degree', '3.24',   '한 팀이 직접 교전하는 다른 팀 수',           false],
                ['clustering', '0.398',  '삼각 교전 밀도',                         false],
                ['diameter',   '5.7',    '가장 먼 두 팀의 단계 수',                 false],
                ['avg path',   '2.76',   '평균 최단 경로 길이',                     false],
                ['Q (Louvain)', '0.591', '강한 community 구조 (>0.5)',           true],
              ].map(([k, v, h, hi], i) => (
                <tr key={k} style={{ borderBottom: i === 7 ? `1px solid ${C.ink}` : `1px solid ${C.rule}` }}>
                  <td style={{ padding: '14px 0', fontFamily: '"JetBrains Mono"', fontSize: 13, color: C.muted, width: 120 }}>{k}</td>
                  <td className="numeric" style={{
                    padding: '14px 0', fontSize: 28, fontWeight: 500, fontFamily: '"Newsreader", serif',
                    color: hi ? C.accent : C.ink, width: 140,
                  }}>{v}</td>
                  <td style={{ padding: '14px 0', fontSize: 14, color: C.ink, fontFamily: '"Newsreader"', fontStyle: 'italic' }}>{h}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <p className="news" style={{ marginTop: 28, fontSize: 16, lineHeight: 1.65, fontStyle: 'italic', color: C.muted }}>
            낮은 density 는 자연스럽다 — 200km² 에 26팀이 흩어져 있으니까. 그러나 Q ≈ 0.59 는
            그 적은 교전이 무작위가 아니라 <span style={{ color: C.accent, fontStyle: 'normal' }}>지역 진영</span> 형태로
            묶여 있다는 신호다.
          </p>
        </div>

        <MarginNote top={64}>
          <div style={{ fontFamily: 'Inter', fontStyle: 'normal', fontSize: 11, letterSpacing: '.08em', textTransform: 'uppercase', color: C.muted }}>
            Fig. 2 · degree 분포
          </div>
          <div style={{ marginTop: 10 }}>
            <TufteSparkbars width={300} height={62} />
          </div>
          <div style={{ fontFamily: 'Inter', fontStyle: 'normal', marginTop: 10, fontSize: 12, color: C.muted, lineHeight: 1.5 }}>
            대부분 팀은 1–4 팀과만 직접 교전한다.<br />long tail 의 8+ 팀이 H2 의 후보군.
          </div>
        </MarginNote>
      </section>

      {/* ─── §3 Adjacency matrix ─── */}
      <section style={{ position: 'relative', marginBottom: 96 }}>
        <SectionHead num="§3" kicker="Algebra" title="인접 행렬과 차수" />

        <div style={{ width: A.main_w, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 40 }}>
          <div>
            <p className="news" style={{ fontSize: 17, lineHeight: 1.62, marginTop: 0 }}>
              그래프 G 를 행렬로 보면 A = [aᵢⱼ] 에서 aᵢⱼ = w(i, j) (없으면 0). A 는 대칭이다.
            </p>
            <div style={{ marginTop: 20 }}>
              <TeX tex={"A_{ij} = w(i,j),\\quad d_i = \\sum_j A_{ij}"} />
            </div>
            <ul style={{ paddingLeft: 0, listStyle: 'none', marginTop: 20, fontSize: 14, lineHeight: 1.6 }}>
              <li style={{ marginBottom: 12 }}>
                <span style={{ color: C.accent, fontWeight: 500 }}>degree(i)</span> — 팀 i 의 이웃 수.
                직접 교전한 서로 다른 팀의 수.
              </li>
              <li style={{ marginBottom: 12 }}>
                <span style={{ color: C.accent, fontWeight: 500 }}>weighted_deg(i)</span> — Σⱼ Aᵢⱼ.
                팀 i 가 주고받은 총 누적 damage.
              </li>
              <li>
                <span style={{ color: C.accent, fontWeight: 500 }}>density</span> — 2|E| / (|V|·(|V|−1)).
                가능한 모든 팀 쌍 중 실제 교전한 비율.
              </li>
            </ul>
          </div>
          {/* Mini adjacency matrix heatmap */}
          <AdjacencyHeatmap />
        </div>
      </section>

      {/* ─── §4 Centralities ─── */}
      <section style={{ position: 'relative', marginBottom: 96 }}>
        <SectionHead num="§4" kicker="Centralities" title="다섯 가지 중심성" />

        <div style={{ width: A.main_w }}>
          <p className="news" style={{ fontSize: 17, lineHeight: 1.62, marginTop: 0, color: C.muted, fontStyle: 'italic' }}>
            distance 기반 지표(Closeness, Betweenness)에서는 weight 가 클수록 “가깝다”고
            해석하므로, distance = 1 / weight 로 변환해 사용한다.
          </p>
          <div style={{ marginTop: 32 }}>
            <CentralityRanks />
          </div>
        </div>

        <MarginNote top={140}>
          별표(★) 는 H2 가설의 핵심 지표.
          서로 다른 community 를 잇는 다리 위치에 있는 팀이 누구인지가
          “다중 교전 중심” 정의의 본질이다.
        </MarginNote>
      </section>

      {/* ─── §5 Community ─── */}
      <section style={{ position: 'relative', marginBottom: 96 }}>
        <SectionHead num="§5" kicker="Community" title="모듈러도 Q 와 Louvain 분할" />

        <div style={{ width: A.main_w }}>
          <p className="news" style={{ fontSize: 17, lineHeight: 1.62, marginTop: 0 }}>
            Louvain greedy 알고리즘으로 modularity Q 를 최대화하는 팀 분할을 찾는다.
            우리 데이터셋의 평균 <span className="numeric" style={{ color: C.accent }}>Q = 0.591</span> 은
            Newman (2004) 의 의미 임계 0.3 을 한참 넘는다.
          </p>
          <div style={{ marginTop: 32, padding: '24px 0', borderTop: `1px solid ${C.rule}`, borderBottom: `1px solid ${C.rule}`, textAlign: 'center' }}>
            <TeX tex={"Q = \\frac{1}{2m}\\sum_{ij}\\Big[A_{ij} - \\frac{k_i k_j}{2m}\\Big]\\delta(c_i, c_j)"} />
          </div>
          <dl style={{ marginTop: 28, fontSize: 15, lineHeight: 1.65 }}>
            <DefRow term="community_size" detail="한 팀이 속한 community 의 팀 수." />
            <DefRow term="community_density" detail="community 서브그래프의 density." />
            <DefRow term="position_label" detail="betweenness 사분위. 상위 25% = core, 하위 25% = peripheral, 중간 = mid, degree=0 = isolated." />
          </dl>
        </div>
      </section>

      {/* ─── §6 Mapping to hypotheses — three column poster ─── */}
      <section style={{ position: 'relative' }}>
        <SectionHead num="§6" kicker="Mapping" title="이 정의가 H1·H2·H3 에 어떻게 들어가는가" />

        <div style={{ width: A.page_w - 160, display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 40, marginTop: 8 }}>
          {[
            {
              tag: 'H1', title: '외곽 파밍 vs 교전 참여',
              uses: ['degree (직접 교전 팀 수)', 'combat_per_min, dmg_per_min', 'centroid_first 의 map 중앙 거리'],
              tex: 'P_i = z(\\text{combat\\_per\\_min}_i) + z(\\text{dmg\\_per\\_min}_i) + z(\\deg_i)',
            },
            {
              tag: 'H2', title: '다중 교전 중심 (C)',
              uses: ['betweenness (중개자성)', 'degree centrality', 'local_clustering'],
              tex: 'C_i = z(c^{bet}_i) + z(c^{deg}_i) + z(\\text{local\\_clustering}_i)',
            },
            {
              tag: 'H3', title: '존버 vs 활발 이동',
              uses: ['team_centroid 시퀀스', 'trajectory_length_m', 'displacement_per_min'],
              tex: 'L_i = \\sum_t \\| c_i(t) - c_i(t-1)\\|',
            },
          ].map((h) => (
            <article key={h.tag} style={{ borderTop: `2px solid ${C.ink}`, paddingTop: 16 }}>
              <div style={{ fontFamily: '"JetBrains Mono"', fontSize: 11, color: C.accent, letterSpacing: '.1em', fontWeight: 600 }}>{h.tag}</div>
              <h3 className="news" style={{ fontSize: 24, marginTop: 6, marginBottom: 16, fontWeight: 500, lineHeight: 1.2 }}>{h.title}</h3>
              <div style={{ fontSize: 11, letterSpacing: '.08em', textTransform: 'uppercase', color: C.muted, fontFamily: '"JetBrains Mono"' }}>사용 지표</div>
              <ul style={{ paddingLeft: 0, listStyle: 'none', margin: '8px 0 16px', fontSize: 14, lineHeight: 1.55 }}>
                {h.uses.map((u) => (
                  <li key={u} style={{ paddingLeft: 14, position: 'relative', marginBottom: 4 }}>
                    <span style={{ position: 'absolute', left: 0, color: C.accent }}>·</span>{u}
                  </li>
                ))}
              </ul>
              <div style={{ fontSize: 11, letterSpacing: '.08em', textTransform: 'uppercase', color: C.muted, fontFamily: '"JetBrains Mono"' }}>합성 점수</div>
              <div style={{ marginTop: 6, fontSize: 13, overflowX: 'auto' }}>
                <TeX tex={h.tex} />
              </div>
              <a style={{ display: 'inline-block', marginTop: 14, fontSize: 13, color: C.accent, fontFamily: '"Newsreader"', fontStyle: 'italic', borderBottom: `1px solid ${C.accent}` }}>
                가설 페이지 →
              </a>
            </article>
          ))}
        </div>

        <div style={{ marginTop: 80, paddingTop: 24, borderTop: `1px solid ${C.rule}`,
                      display: 'flex', justifyContent: 'space-between', fontSize: 11,
                      color: C.muted, fontFamily: '"JetBrains Mono"', letterSpacing: '.08em', textTransform: 'uppercase' }}>
          <span>PUBG · Network · v2</span>
          <span>§0 · End of Reference</span>
          <span>2026</span>
        </div>
      </section>
    </div>
  );
}

function SectionHead({ num, kicker, title }) {
  return (
    <div style={{ marginBottom: 32 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 14, marginBottom: 8 }}>
        <span className="numeric" style={{ fontSize: 12, color: A.accent, fontFamily: '"JetBrains Mono"', letterSpacing: '.08em', fontWeight: 600 }}>{num}</span>
        <span style={{ fontSize: 11, color: A.muted, letterSpacing: '.12em', textTransform: 'uppercase', fontFamily: '"JetBrains Mono"' }}>{kicker}</span>
      </div>
      <h2 className="news" style={{ fontSize: 36, fontWeight: 500, letterSpacing: '-0.01em', margin: 0, lineHeight: 1.15 }}>
        {title}
      </h2>
    </div>
  );
}

function DefRow({ term, detail }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr', gap: 24,
                  padding: '14px 0', borderTop: `1px solid ${A.rule}` }}>
      <dt className="numeric" style={{ fontSize: 14, color: A.accent, fontWeight: 500 }}>{term}</dt>
      <dd className="news" style={{ margin: 0, fontSize: 16, lineHeight: 1.55 }}>{detail}</dd>
    </div>
  );
}

function AdjacencyHeatmap({ size = 14 }) {
  // Build a small symmetric matrix from a tiny network
  const { nodes, edges } = React.useMemo(() => makeNetwork(3, size), []);
  const W = Array.from({ length: size }, () => Array(size).fill(0));
  edges.forEach(e => { W[e.a][e.b] = e.w; W[e.b][e.a] = e.w; });
  const cell = 18;
  const dim = size * cell;
  return (
    <div>
      <div style={{ fontFamily: '"JetBrains Mono"', fontSize: 11, letterSpacing: '.08em', textTransform: 'uppercase', color: A.muted, marginBottom: 8 }}>
        Fig. 3 · 인접 행렬 A
      </div>
      <svg width={dim} height={dim} viewBox={`0 0 ${dim} ${dim}`} style={{ border: `1px solid ${A.rule}` }}>
        {W.map((row, i) =>
          row.map((v, j) => {
            const hue = v === 0 ? A.paper : A.ink;
            const op = v === 0 ? 0 : 0.2 + Math.min(v, 1) * 0.7;
            return <rect key={`${i},${j}`} x={j * cell} y={i * cell} width={cell} height={cell}
              fill={hue} fillOpacity={op} />;
          })
        )}
        {/* diagonal hint */}
        <line x1="0" y1="0" x2={dim} y2={dim} stroke={A.accent} strokeWidth="0.5" strokeDasharray="2 3" opacity="0.5" />
      </svg>
      <div style={{ marginTop: 6, fontSize: 11, color: A.muted, fontFamily: '"Newsreader"', fontStyle: 'italic' }}>
        대각선 0, 대칭. 짙을수록 큰 weight.
      </div>
    </div>
  );
}

window.OptionATufte = OptionATufte;


Object.assign(window, { OptionATufte });
