// Option C — Lab Notebook / Field Manual
// Technical document feel: grid paper background, monospace-forward,
// blueprints/diagrams, coordinate-marked figures. Thinks of itself as a
// research operations manual. Strong dot-grid rhythm + plate numbers.

const Ck = {
  paper: '#f4efe2',
  ink: '#0e0c08',
  muted: '#5e574a',
  faint: '#a8a08e',
  rule: '#c9c0a8',
  accent: '#7a2018',  // deeper oxblood
  ink2: '#2c2820',
  blueprint: '#26405a',
};

// Grid paper background as an inline SVG pattern
function GridPaperBG({ width, height, gap = 14 }) {
  return (
    <svg width={width} height={height} style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
      <defs>
        <pattern id="gridC" width={gap} height={gap} patternUnits="userSpaceOnUse">
          <path d={`M ${gap} 0 L 0 0 0 ${gap}`} fill="none" stroke="#0e0c08" strokeOpacity="0.07" strokeWidth="0.5" />
        </pattern>
        <pattern id="gridC2" width={gap * 5} height={gap * 5} patternUnits="userSpaceOnUse">
          <path d={`M ${gap * 5} 0 L 0 0 0 ${gap * 5}`} fill="none" stroke="#0e0c08" strokeOpacity="0.13" strokeWidth="0.6" />
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill="url(#gridC)" />
      <rect width="100%" height="100%" fill="url(#gridC2)" />
    </svg>
  );
}

// Plate label — top-right of a figure
function Plate({ num, label }) {
  return (
    <div style={{
      display: 'inline-flex', alignItems: 'center', gap: 8,
      fontFamily: '"JetBrains Mono"', fontSize: 11, color: Ck.muted, letterSpacing: '.1em',
      textTransform: 'uppercase',
    }}>
      <span style={{ background: Ck.ink, color: Ck.paper, padding: '2px 6px', fontWeight: 600 }}>PL.{num}</span>
      <span>{label}</span>
    </div>
  );
}

// Network with drafting-style annotations: tick marks on edges, coordinate labels
function FieldNetwork({ width = 480, height = 340 }) {
  const { nodes, edges } = React.useMemo(() => makeNetwork(23, 26), []);
  const pad = 30;
  const px = (x) => pad + x * (width - pad * 2);
  const py = (y) => pad + y * (height - pad * 2);

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ background: Ck.paper, border: `1px solid ${Ck.ink}` }}>
      {/* corner crop marks */}
      {[
        [10, 10, 22, 10], [10, 10, 10, 22],
        [width - 10, 10, width - 22, 10], [width - 10, 10, width - 10, 22],
        [10, height - 10, 22, height - 10], [10, height - 10, 10, height - 22],
        [width - 10, height - 10, width - 22, height - 10], [width - 10, height - 10, width - 10, height - 22],
      ].map(([x1, y1, x2, y2], i) => (
        <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={Ck.ink} strokeWidth="1" />
      ))}
      {/* edges */}
      {edges.map((e, i) => {
        const a = nodes[e.a], b = nodes[e.b];
        return <line key={i} x1={px(a.x)} y1={py(a.y)} x2={px(b.x)} y2={py(b.y)}
          stroke={Ck.ink2} strokeWidth={0.6 + e.w * 1.2} strokeOpacity={0.55} />;
      })}
      {/* nodes — squares with id labels */}
      {nodes.map((nd) => {
        const r = 4 + Math.min(nd.degree, 8) * 0.7;
        const isHi = nd.id === 7;
        return (
          <g key={nd.id}>
            <rect x={px(nd.x) - r} y={py(nd.y) - r} width={r * 2} height={r * 2}
              fill={isHi ? Ck.accent : Ck.ink} stroke={Ck.paper} strokeWidth="1" />
            {(nd.degree >= 4) && (
              <text x={px(nd.x)} y={py(nd.y) - r - 4} fontSize="8" textAnchor="middle"
                fill={Ck.muted} fontFamily="JetBrains Mono">T{nd.id.toString().padStart(2,'0')}</text>
            )}
          </g>
        );
      })}
      {/* coordinate ticks */}
      {[0.25, 0.5, 0.75].map((t) => (
        <g key={t}>
          <line x1={px(t)} y1={height - pad} x2={px(t)} y2={height - pad + 4} stroke={Ck.ink} strokeWidth="0.6" />
          <text x={px(t)} y={height - pad + 14} fontSize="8" textAnchor="middle"
            fill={Ck.muted} fontFamily="JetBrains Mono">{t.toFixed(2)}</text>
          <line x1={pad - 4} y1={py(t)} x2={pad} y2={py(t)} stroke={Ck.ink} strokeWidth="0.6" />
          <text x={pad - 7} y={py(t) + 3} fontSize="8" textAnchor="end"
            fill={Ck.muted} fontFamily="JetBrains Mono">{t.toFixed(2)}</text>
        </g>
      ))}
      {/* north arrow */}
      <g transform={`translate(${width - 38}, ${height - 38})`}>
        <circle cx="0" cy="0" r="14" fill="none" stroke={Ck.ink} strokeWidth="0.6" />
        <line x1="0" y1="-12" x2="0" y2="12" stroke={Ck.ink} strokeWidth="0.6" />
        <text x="0" y="-16" fontSize="8" textAnchor="middle" fill={Ck.ink} fontFamily="JetBrains Mono" fontWeight="600">N</text>
      </g>
    </svg>
  );
}

function OptionCField() {
  const C = Ck;
  return (
    <div style={{ width: 1280, minHeight: 2400, background: C.paper, color: C.ink,
                  fontFamily: '"JetBrains Mono", "Inter", system-ui, monospace',
                  position: 'relative', overflow: 'hidden' }}>
      <GridPaperBG width="100%" height="100%" />

      <div style={{ position: 'relative', padding: '56px 64px 80px' }}>

        {/* Tag strip */}
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10,
                      fontFamily: '"JetBrains Mono"', color: C.muted, letterSpacing: '.14em',
                      textTransform: 'uppercase', borderBottom: `1px solid ${C.ink}`, paddingBottom: 10 }}>
          <span>FILE / 00-NETWORK-BASICS</span>
          <span>BALTIC_MAIN · SQUAD · DMG</span>
          <span>REV 2026·04·27</span>
          <span>CLASS · REFERENCE</span>
        </div>

        {/* Title block */}
        <header style={{ marginTop: 40, display: 'grid', gridTemplateColumns: '1fr auto', gap: 40, alignItems: 'end' }}>
          <div>
            <div style={{ fontSize: 11, color: C.accent, letterSpacing: '.16em', fontWeight: 600 }}>§ 0 · GRAPH PRIMER</div>
            <h1 style={{ fontFamily: '"JetBrains Mono"', fontSize: 56, lineHeight: 1.04,
                          fontWeight: 600, letterSpacing: '-0.02em', margin: '12px 0 0', color: C.ink }}>
              네트워크 기초 정의
            </h1>
            <p className="news" style={{ marginTop: 18, fontSize: 17, lineHeight: 1.5, color: C.muted, fontStyle: 'italic',
                                          fontFamily: '"Newsreader"', maxWidth: 720 }}>
              매치 1건을 무방향 가중 그래프 G = (V, E, w) 로 요약하기 위한 정의·표기·구현
              레퍼런스. H1·H2·H3 는 모두 이 그래프 위에서 계산된다.
            </p>
          </div>
          {/* Stamp */}
          <div style={{ border: `1.5px solid ${C.accent}`, color: C.accent, padding: '8px 14px',
                        fontSize: 10, letterSpacing: '.18em', textAlign: 'center', transform: 'rotate(-3deg)', fontWeight: 600 }}>
            VERIFIED<br/><span style={{ fontSize: 8 }}>v2 · n=3,595</span>
          </div>
        </header>

        {/* Status board */}
        <div style={{ marginTop: 40, border: `1px solid ${C.ink}`, padding: 0, background: C.paper }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, 1fr)', borderBottom: `1px solid ${C.ink}` }}>
            {[
              ['|V|',     '26.4'],
              ['|E|',     '42.6'],
              ['ρ',       '0.131'],
              ['k̄',     '3.24'],
              ['C̄',     '0.398'],
              ['diam',    '5.7'],
              ['L̄',     '2.76'],
              ['Q',       '0.591', true],
            ].map(([k, v, hi], i) => (
              <div key={k} style={{ padding: '10px 14px', borderRight: i < 7 ? `1px solid ${C.rule}` : 'none',
                                     background: hi ? C.accent : 'transparent', color: hi ? C.paper : C.ink }}>
                <div style={{ fontSize: 9, letterSpacing: '.12em', opacity: 0.7 }}>{k}</div>
                <div className="numeric" style={{ fontSize: 22, fontWeight: 600, marginTop: 2 }}>{v}</div>
              </div>
            ))}
          </div>
          <div style={{ padding: '8px 14px', fontSize: 10, color: C.muted, letterSpacing: '.06em',
                        display: 'flex', justifyContent: 'space-between' }}>
            <span>STATUS BOARD · 매치 평균 통계</span>
            <span>n_matches = 3,595 · n_teams = 94,964</span>
          </div>
        </div>

        {/* §1 — Definition with diagram */}
        <SectionC num="01" title="GRAPH DEFINITION" />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 48, marginTop: 24 }}>
          <div>
            <p className="news" style={{ fontSize: 16, lineHeight: 1.6, marginTop: 0, fontFamily: '"Newsreader"' }}>
              한 매치의 모든 팀을 노드로, 두 팀 사이의 교전을 엣지로 두는 무방향 가중 그래프를 구성한다.
            </p>
            <table style={{ width: '100%', marginTop: 24, borderCollapse: 'collapse', fontSize: 13 }}>
              <tbody>
                {[
                  ['V',      '매치 참가 팀 집합 (|V| ≈ 26–28)'],
                  ['E',      '두 팀 사이에 damage event 가 1회 이상 → 무방향 엣지'],
                  ['w(u,v)', '(u,v) 가 주고받은 누적 damage 합'],
                  ['—',      '자기 팀 데미지 / Bluezone / 자살 제외'],
                ].map(([k, v]) => (
                  <tr key={k} style={{ borderBottom: `1px dashed ${C.rule}` }}>
                    <td style={{ padding: '10px 0', fontFamily: '"JetBrains Mono"', color: C.accent, fontWeight: 600, width: 80, verticalAlign: 'top' }}>{k}</td>
                    <td className="news" style={{ padding: '10px 0', fontFamily: '"Newsreader"', fontSize: 14, lineHeight: 1.5 }}>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div style={{ marginTop: 24, background: C.paper, color: C.ink, padding: 18, fontSize: 12,
                          fontFamily: '"JetBrains Mono"', border: `1px solid ${C.ink}`,
                          borderLeft: `3px solid ${C.accent}` }}>
              <span style={{ color: C.accent, fontWeight: 600 }}>$ </span>
              network_analysis/team_network.py<br/>
              <span style={{ color: C.muted }}>  → _events_to_team_graph(<br/>     telemetry, weight_mode="dmg",<br/>     exclude_self_team=True)</span>
            </div>
          </div>
          <div>
            <Plate num="01" label="MATCH GRAPH · 평균 케이스" />
            <div style={{ marginTop: 12 }}>
              <FieldNetwork width={520} height={360} />
            </div>
            <div style={{ marginTop: 14, padding: 14, border: `1px solid ${C.ink}`, fontSize: 12, color: C.ink, background: C.paper }}>
              <TeX tex={"G = (V,E),\\ w(u,v) = \\sum_{e \\in \\mathrm{events}_{uv}} \\mathrm{dmg}_e"} />
            </div>
          </div>
        </div>

        {/* §2 — Aggregate as a long form data spec */}
        <SectionC num="02" title="DATASET PROFILE" subtitle="3,595 matches · Baltic_Main · squad" />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 48, marginTop: 24, alignItems: 'start' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, border: `1px solid ${C.ink}` }}>
            <thead>
              <tr style={{ background: C.paper, color: C.ink, borderBottom: `2px solid ${C.ink}` }}>
                <th style={{ textAlign: 'left', padding: '10px 12px', fontSize: 10, letterSpacing: '.14em', fontWeight: 700 }}>SYM</th>
                <th style={{ textAlign: 'right', padding: '10px 12px', fontSize: 10, letterSpacing: '.14em', fontWeight: 700 }}>VALUE</th>
                <th style={{ textAlign: 'left', padding: '10px 12px', fontSize: 10, letterSpacing: '.14em', fontWeight: 700 }}>SEMANTICS</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['|V|',          '26.4',  'avg # nodes / match'],
                ['|E|',          '42.6',  'avg # combat-pairs / match'],
                ['density ρ',    '0.131', '2|E| / |V|(|V|−1)'],
                ['avg degree',   '3.24',  'mean directly-engaged opponents'],
                ['clustering C̄','0.398', 'triangle-engagement density'],
                ['diameter',     '5.7',   'max shortest-path hop count'],
                ['avg path L̄',  '2.76',  'mean shortest-path length'],
                ['giant size',   '25.6',  'largest connected component'],
                ['mod. Q',       '0.591', 'Louvain · strong community signal'],
              ].map((row, i) => (
                <tr key={row[0]} style={{ borderTop: `1px solid ${C.rule}`, background: i === 8 ? '#f7e8e6' : 'transparent' }}>
                  <td style={{ padding: '10px 12px', fontFamily: '"JetBrains Mono"', color: i === 8 ? C.accent : C.ink, fontWeight: i === 8 ? 600 : 400 }}>{row[0]}</td>
                  <td className="numeric" style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 600, color: i === 8 ? C.accent : C.ink }}>{row[1]}</td>
                  <td className="news" style={{ padding: '10px 12px', fontStyle: 'italic', color: C.muted, fontFamily: '"Newsreader"', fontSize: 13.5 }}>{row[2]}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div>
            <Plate num="02" label="DEGREE DISTRIBUTION · k(i)" />
            <div style={{ marginTop: 12, padding: 16, border: `1px solid ${C.ink}`, background: C.paper }}>
              <DegreeHistogram width={460} height={200} color={C.ink} accent={C.accent} />
            </div>
            <div style={{ marginTop: 16, padding: 16, background: C.paper, color: C.ink, fontSize: 12,
                          lineHeight: 1.7, fontFamily: '"JetBrains Mono"',
                          border: `1px solid ${C.ink}`, borderLeft: `3px solid ${C.accent}` }}>
              <span style={{ color: C.accent, fontWeight: 700, letterSpacing: '.08em' }}>NOTE.</span> ρ ≪ 1 자연스러움 (200km² / 26팀).
              그러나 Q ≈ 0.59 → 적은 교전이 무작위가 아닌 <span style={{ color: C.accent, fontWeight: 600 }}>지역 진영</span> 구조.
            </div>
          </div>
        </div>

        {/* §3 — Adjacency matrix as schematic */}
        <SectionC num="03" title="ADJACENCY · DEGREE" />
        <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 48, marginTop: 24, alignItems: 'start' }}>
          <div>
            <p className="news" style={{ fontSize: 16, lineHeight: 1.6, marginTop: 0, fontFamily: '"Newsreader"' }}>
              그래프 G 를 행렬로 보면 A = [aᵢⱼ] 에서 aᵢⱼ = w(i, j) (없으면 0). A 는 대칭이다. degree, weighted degree, density 가 여기서 따라 나온다.
            </p>
            <div style={{ marginTop: 20, padding: 18, border: `1px solid ${C.ink}`, background: C.paper }}>
              <TeX tex={"A_{ij} = w(i,j),\\ d_i = \\sum_j A_{ij},\\ D = \\mathrm{diag}(d_1, \\ldots, d_n)"} />
            </div>
            <ul style={{ marginTop: 24, paddingLeft: 0, listStyle: 'none', fontSize: 13, lineHeight: 1.7, fontFamily: '"JetBrains Mono"' }}>
              <li><span style={{ color: C.accent }}>▪ degree(i)</span><br/><span style={{ color: C.muted }}>  → 직접 교전한 팀 수 (binary)</span></li>
              <li style={{ marginTop: 10 }}><span style={{ color: C.accent }}>▪ weighted_deg(i)</span><br/><span style={{ color: C.muted }}>  → Σⱼ Aᵢⱼ, 누적 damage</span></li>
              <li style={{ marginTop: 10 }}><span style={{ color: C.accent }}>▪ density</span><br/><span style={{ color: C.muted }}>  → 2|E| / |V|(|V|−1)</span></li>
            </ul>
          </div>
          <div>
            <Plate num="03" label="MATRIX A · 16×16 SAMPLE" />
            <div style={{ marginTop: 12 }}>
              <AdjacencyHeatmapC size={16} />
            </div>
          </div>
        </div>

        {/* §4 — Centralities as spec sheet */}
        <SectionC num="04" title="CENTRALITY MEASURES" subtitle="distance = 1 / weight 변환 후 계산" />
        <div style={{ marginTop: 24, border: `1px solid ${C.ink}` }}>
          {[
            { name: 'C_D · Degree',          tex: 'C_D(i) = k(i)/(|N|-1)',                                 ko: '이웃 팀 수 정규화. 단순 직접 교전 빈도.', tag: 'H1·H2' },
            { name: 'C_C · Closeness',       tex: 'C_C(i) = (|N|-1)/\\sum_j d(i,j)',                       ko: '다른 모든 팀까지 평균 거리의 역수. 클수록 그래프 중앙.', tag: '—' },
            { name: 'C_B · Betweenness',     tex: 'C_B(i) = \\sum_{j\\ne k} \\sigma_{jk}(i)/\\sigma_{jk}', ko: '최단경로가 i 를 지나는 비율. 다른 클러스터를 잇는 중개자.', tag: 'H2 ★', star: true },
            { name: 'C_E · Eigenvector',     tex: '\\lambda c = A c',                                       ko: '중요한 이웃과 연결될수록 본인도 중요.', tag: '—' },
            { name: 'C_R · PageRank',        tex: 'r = \\beta M r + (1-\\beta)/N',                          ko: 'Eigenvector 의 random-teleport 확장. β = 0.85.', tag: '—' },
            { name: 'Cl · Local Clustering', tex: 'C_i = 2|\\{(j,k): j,k\\in N(i),(j,k)\\in E\\}|/(k_i(k_i-1))', ko: '팀 i 의 이웃끼리도 서로 교전한 비율. 삼각 교전 밀도.', tag: 'H2 보조' },
          ].map((row, i) => (
            <div key={row.name} style={{ display: 'grid', gridTemplateColumns: '220px 1fr 1fr 100px',
                                          gap: 24, padding: '16px 20px',
                                          borderTop: i === 0 ? 'none' : `1px solid ${C.rule}`,
                                          background: row.star ? '#f7e8e6' : 'transparent',
                                          alignItems: 'center' }}>
              <div style={{ fontFamily: '"JetBrains Mono"', fontSize: 13, fontWeight: 600,
                            color: row.star ? C.accent : C.ink }}>{row.name}</div>
              <div style={{ fontSize: 12 }}>
                <TeX tex={row.tex} />
              </div>
              <div className="news" style={{ fontSize: 13.5, color: C.muted, lineHeight: 1.45,
                                              fontFamily: '"Newsreader"', fontStyle: 'italic' }}>{row.ko}</div>
              <div style={{ textAlign: 'right', fontSize: 10, fontFamily: '"JetBrains Mono"',
                            color: row.star ? C.accent : C.muted, letterSpacing: '.1em', fontWeight: row.star ? 600 : 400 }}>
                {row.tag}
              </div>
            </div>
          ))}
        </div>

        {/* §5 — Community */}
        <SectionC num="05" title="COMMUNITY · MODULARITY Q" />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 48, marginTop: 24 }}>
          <div>
            <p className="news" style={{ fontSize: 16, lineHeight: 1.6, marginTop: 0, fontFamily: '"Newsreader"' }}>
              Louvain greedy 알고리즘으로 modularity Q 를 최대화하는 팀 분할을 찾는다.
              데이터셋 평균 <span className="numeric" style={{ color: C.accent, fontWeight: 600 }}>Q = 0.591</span> 은
              Newman (2004) 의 의미 임계 0.3 을 한참 넘는다.
            </p>
            <div style={{ marginTop: 20, padding: 18, border: `1px solid ${C.ink}`, background: C.paper }}>
              <TeX tex={"Q = \\frac{1}{2m}\\sum_{ij}\\Big[A_{ij} - \\frac{k_i k_j}{2m}\\Big]\\delta(c_i,c_j)"} />
            </div>
            <table style={{ width: '100%', marginTop: 24, borderCollapse: 'collapse', fontSize: 13 }}>
              <tbody>
                {[
                  ['community_size',    '한 팀이 속한 community 의 팀 수'],
                  ['community_density', 'community 서브그래프의 density'],
                  ['position_label',    'betweenness 사분위 → core / mid / peripheral / isolated'],
                ].map((r) => (
                  <tr key={r[0]} style={{ borderBottom: `1px dashed ${C.rule}` }}>
                    <td style={{ padding: '10px 0', fontFamily: '"JetBrains Mono"', color: C.accent, fontWeight: 600, width: 200, verticalAlign: 'top' }}>{r[0]}</td>
                    <td className="news" style={{ padding: '10px 0', fontFamily: '"Newsreader"', fontSize: 14, lineHeight: 1.5 }}>{r[1]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div>
            <Plate num="04" label="LOUVAIN PARTITION" />
            <div style={{ marginTop: 12, padding: 14, border: `1px solid ${C.ink}`, background: C.paper }}>
              <MiniNetwork width={500} height={340} seed={41} n={26}
                bg={C.paper} edgeColor={C.faint}
                nodeColors={[C.ink, C.blueprint, '#6a5520']}
                nodeStroke={C.paper} />
            </div>
            <div style={{ marginTop: 12, display: 'flex', justifyContent: 'space-between', fontSize: 11,
                          color: C.muted, fontFamily: '"JetBrains Mono"' }}>
              <span>3 communities</span><span>Q = 0.62 · core: 4팀 · peripheral: 7팀</span>
            </div>
          </div>
        </div>

        {/* §6 — Hypothesis spec sheet */}
        <SectionC num="06" title="HYPOTHESIS WIRING" subtitle="이 정의가 H1·H2·H3 에 어떻게 들어가는가" />
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 24, marginTop: 24 }}>
          {[
            { tag: 'H1', title: '외곽 파밍 vs 교전 참여',
              uses: ['degree', 'combat_per_min', 'dmg_per_min', 'centroid_first dist'],
              tex: 'P_i = z(\\text{combat}) + z(\\text{dmg}) + z(\\deg)' },
            { tag: 'H2', title: '다중 교전 중심 (C)',
              uses: ['betweenness ★', 'degree centrality', 'local_clustering ★'],
              tex: 'C_i = z(c^{bet}) + z(c^{deg}) + z(\\text{clust})', star: true },
            { tag: 'H3', title: '존버 vs 활발 이동',
              uses: ['team_centroid 시퀀스', 'trajectory_length_m', 'displacement_per_min'],
              tex: 'L_i = \\sum_t \\| c_i(t) - c_i(t-1)\\|' },
          ].map((h) => (
            <div key={h.tag} style={{ border: `1.5px solid ${h.star ? C.accent : C.ink}`, padding: 20, background: C.paper, position: 'relative' }}>
              <div style={{ position: 'absolute', top: -10, left: 14, background: C.paper, padding: '0 8px',
                             fontFamily: '"JetBrains Mono"', fontSize: 11, color: h.star ? C.accent : C.muted, letterSpacing: '.16em', fontWeight: 600 }}>
                · {h.tag} ·
              </div>
              <h3 style={{ fontFamily: '"JetBrains Mono"', fontSize: 17, marginTop: 6, marginBottom: 14, fontWeight: 600 }}>{h.title}</h3>
              <div style={{ fontSize: 10, letterSpacing: '.14em', color: C.muted, fontFamily: '"JetBrains Mono"' }}>USES</div>
              <ul style={{ paddingLeft: 0, listStyle: 'none', margin: '6px 0 16px',
                            fontFamily: '"JetBrains Mono"', fontSize: 12, lineHeight: 1.7 }}>
                {h.uses.map((u, i) => (
                  <li key={i} style={{ paddingLeft: 14, position: 'relative' }}>
                    <span style={{ position: 'absolute', left: 0 }}>{i + 1}.</span>{u}
                  </li>
                ))}
              </ul>
              <div style={{ paddingTop: 14, borderTop: `1px dashed ${C.rule}`, fontSize: 12 }}>
                <TeX tex={h.tex} />
              </div>
            </div>
          ))}
        </div>

        {/* footer */}
        <div style={{ marginTop: 64, paddingTop: 16, borderTop: `2px solid ${C.ink}`,
                      display: 'flex', justifyContent: 'space-between', fontSize: 10,
                      color: C.muted, fontFamily: '"JetBrains Mono"', letterSpacing: '.14em', textTransform: 'uppercase' }}>
          <span>END · 00-NETWORK-BASICS</span>
          <span>NEXT · 01-H1-ARCHETYPE</span>
          <span>PUBG · NETWORK · v2 · 2026</span>
        </div>
      </div>
    </div>
  );
}

function SectionC({ num, title, subtitle }) {
  return (
    <div style={{ marginTop: 80, marginBottom: 4 }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr auto', gap: 16, alignItems: 'center', marginBottom: 8 }}>
        <span style={{ fontFamily: '"JetBrains Mono"', fontSize: 11, color: Ck.paper, background: Ck.ink, padding: '4px 10px', letterSpacing: '.16em', fontWeight: 600 }}>§ {num}</span>
        <h2 style={{ fontFamily: '"JetBrains Mono"', fontSize: 24, fontWeight: 600, letterSpacing: '0.04em', margin: 0, textTransform: 'uppercase' }}>
          {title}
        </h2>
        <span style={{ fontFamily: '"JetBrains Mono"', fontSize: 10, color: Ck.muted, letterSpacing: '.14em' }}>SECTION {num} OF 06</span>
      </div>
      {subtitle && (
        <div className="news" style={{ fontSize: 14, color: Ck.muted, fontStyle: 'italic',
                                        fontFamily: '"Newsreader"', marginTop: 6, marginLeft: 56 }}>{subtitle}</div>
      )}
      <div style={{ height: 4, background: Ck.ink, marginTop: 12 }} />
    </div>
  );
}

function AdjacencyHeatmapC({ size = 16 }) {
  const { edges } = React.useMemo(() => makeNetwork(3, size), []);
  const W = Array.from({ length: size }, () => Array(size).fill(0));
  edges.forEach(e => { W[e.a][e.b] = e.w; W[e.b][e.a] = e.w; });
  const cell = 22;
  const dim = size * cell;
  return (
    <div style={{ background: Ck.paper, border: `1px solid ${Ck.ink}`, padding: 12, display: 'inline-block' }}>
      <svg width={dim} height={dim} viewBox={`0 0 ${dim} ${dim}`}>
        {/* axis ticks */}
        {Array.from({ length: size }).map((_, i) => (
          <g key={i}>
            <text x={i * cell + cell / 2} y={-4} fontSize="7" textAnchor="middle"
              fill={Ck.muted} fontFamily="JetBrains Mono">{i}</text>
          </g>
        ))}
        {W.map((row, i) =>
          row.map((v, j) => {
            const op = v === 0 ? 0 : 0.2 + Math.min(v, 1) * 0.7;
            const fill = v === 0 ? 'transparent' : Ck.ink;
            return (
              <rect key={`${i},${j}`} x={j * cell} y={i * cell} width={cell} height={cell}
                fill={fill} fillOpacity={op}
                stroke={Ck.rule} strokeWidth="0.4" />
            );
          })
        )}
        <line x1="0" y1="0" x2={dim} y2={dim} stroke={Ck.accent} strokeWidth="0.6" strokeDasharray="3 3" />
      </svg>
    </div>
  );
}

window.OptionCField = OptionCField;


Object.assign(window, { OptionCField });
