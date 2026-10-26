// Option B — Pudding-style data journalism
// Big hero network, strong scrollytelling rhythm, varied per-section layouts,
// confident accent color, oversized numbers as visual anchors.

const B = {
  paper: '#fefcf7',
  cream: '#f3eddd',
  ink: '#15110a',
  muted: '#6b6558',
  faint: '#cdc5b3',
  rule: '#d8d1c2',
  accent: '#b02a2a',
  accent2: '#3a4a5a', // steel blue community
  accent3: '#6b8e3a', // olive community
};

// Hero network — large, two communities clearly visible, with a labeled
// "bridge" node highlighted as H2 motif
function HeroNetwork({ width = 1080, height = 460 }) {
  const { nodes, edges } = React.useMemo(() => makeNetwork(17, 28), []);
  // pick a node with most cross-community edges as bridge
  let bridgeIdx = 0, bridgeScore = -1;
  for (let i = 0; i < nodes.length; i++) {
    const cnt = edges.filter(e =>
      (e.a === i || e.b === i) &&
      nodes[e.a].community !== nodes[e.b].community
    ).length;
    if (cnt > bridgeScore) { bridgeScore = cnt; bridgeIdx = i; }
  }
  const pad = 40;
  const px = (x) => pad + x * (width - pad * 2);
  const py = (y) => pad + y * (height - pad * 2);
  const colors = [B.ink, B.accent2, B.accent3];

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      {edges.map((e, i) => {
        const a = nodes[e.a], b = nodes[e.b];
        const cross = a.community !== b.community;
        return (
          <line key={i}
            x1={px(a.x)} y1={py(a.y)} x2={px(b.x)} y2={py(b.y)}
            stroke={cross ? B.accent : '#a89e88'}
            strokeWidth={cross ? 1.6 + e.w * 1.4 : 0.6 + e.w * 1.2}
            strokeOpacity={cross ? 0.85 : 0.5}
          />
        );
      })}
      {nodes.map((nd) => {
        const r = 5 + Math.min(nd.degree, 8) * 1.1;
        const isBridge = nd.id === bridgeIdx;
        return (
          <g key={nd.id}>
            <circle cx={px(nd.x)} cy={py(nd.y)} r={isBridge ? r + 4 : r}
              fill={isBridge ? B.accent : colors[nd.community]}
              stroke={B.paper} strokeWidth={2}
            />
            {isBridge && (
              <>
                <circle cx={px(nd.x)} cy={py(nd.y)} r={r + 14} fill="none" stroke={B.accent} strokeWidth="1" strokeDasharray="3 3" opacity="0.6" />
                <text x={px(nd.x) + r + 22} y={py(nd.y) + 4} fontSize="13" fill={B.accent}
                  fontFamily="Inter, sans-serif" fontWeight="600">
                  bridge · 높은 betweenness
                </text>
              </>
            )}
          </g>
        );
      })}
      {/* community labels */}
      <text x={pad + 30} y={pad + 30} fontSize="11" fill={B.ink} fontFamily="JetBrains Mono, monospace" letterSpacing="1.5">CLUSTER A</text>
      <text x={width - pad - 110} y={pad + 30} fontSize="11" fill={B.accent2} fontFamily="JetBrains Mono, monospace" letterSpacing="1.5">CLUSTER B</text>
      <text x={pad + 30} y={height - pad - 12} fontSize="11" fill={B.accent3} fontFamily="JetBrains Mono, monospace" letterSpacing="1.5">CLUSTER C</text>
    </svg>
  );
}

// Scrolly-style stat row — gigantic numbers, label below
function GiantStat({ label, value, hint, span = 1, color }) {
  return (
    <div style={{ gridColumn: `span ${span}`, paddingTop: 18, borderTop: `2px solid ${B.ink}` }}>
      <div style={{ fontSize: 10, letterSpacing: '.14em', textTransform: 'uppercase',
                    fontFamily: '"JetBrains Mono"', color: B.muted, marginBottom: 14 }}>{label}</div>
      <div className="numeric" style={{ fontSize: 88, lineHeight: 0.95, fontWeight: 500, color: color || B.ink, letterSpacing: '-0.04em' }}>{value}</div>
      <div className="news" style={{ marginTop: 16, fontSize: 14, lineHeight: 1.5, color: B.muted, fontStyle: 'italic' }}>{hint}</div>
    </div>
  );
}

function OptionBPudding() {
  const C = B;
  return (
    <div style={{
      width: 1280, minHeight: 2400, background: C.paper, color: C.ink,
      fontFamily: '"Inter", "Noto Sans KR", system-ui, sans-serif',
      position: 'relative',
    }}>
      {/* HERO — full bleed, dark cream background */}
      <header style={{ background: C.cream, padding: '64px 80px 56px', position: 'relative', overflow: 'hidden' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 32 }}>
          <div style={{ width: 28, height: 1, background: C.accent }} />
          <span style={{ fontSize: 11, letterSpacing: '.16em', textTransform: 'uppercase',
                         fontFamily: '"JetBrains Mono"', color: C.accent, fontWeight: 600 }}>Reference · 가설 검증 토대</span>
        </div>
        <h1 className="news" style={{
          fontSize: 96, lineHeight: 0.96, fontWeight: 500, letterSpacing: '-0.035em',
          margin: 0, maxWidth: 1100,
        }}>
          26팀, 1,440초,<br />
          하나의 <span style={{ color: C.accent, fontStyle: 'italic' }}>그래프</span>.
        </h1>
        <p style={{ marginTop: 32, fontSize: 19, lineHeight: 1.55, maxWidth: 720, color: C.muted }}>
          한 매치를 무방향 가중 그래프 G = (V, E, w) 로 요약하면,
          누가 누구와 얼마나 부딪혔는지가 <strong style={{ color: C.ink, fontWeight: 600 }}>구조</strong>로 드러난다.
          이 페이지는 H1·H2·H3 가설이 그 위에서 어떻게 정의되는지 보여준다.
        </p>

        <div style={{ marginTop: 48 }}>
          <HeroNetwork width={1120} height={420} />
        </div>

        <div style={{ marginTop: 28, display: 'flex', alignItems: 'center', gap: 28, fontSize: 12,
                      fontFamily: '"JetBrains Mono"', color: C.muted, letterSpacing: '.06em' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 10, height: 10, borderRadius: 5, background: C.ink }} /> Cluster A
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 10, height: 10, borderRadius: 5, background: C.accent2 }} /> Cluster B
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 10, height: 10, borderRadius: 5, background: C.accent3 }} /> Cluster C
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 22, height: 1.5, background: C.accent }} /> 클러스터 간 다리
          </span>
          <span style={{ marginLeft: 'auto' }}>match #00036c37 · n=26 · m=44 · Q=0.657</span>
        </div>
      </header>

      {/* §1 — Asymmetric two-up: definition list left, formula card right */}
      <section style={{ padding: '120px 80px 0', position: 'relative' }}>
        <SectionMarkB num="01" title="그래프 정의" />
        <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 80, marginTop: 24 }}>
          <div>
            <p className="news" style={{ fontSize: 24, lineHeight: 1.4, fontWeight: 400, marginTop: 0 }}>
              한 매치의 모든 팀을 노드로, 두 팀 사이의 교전을 엣지로 두는
              무방향 가중 그래프를 구성한다.
              <span style={{ color: C.muted }}> 모든 후속 분석은 이 그래프 위에서 계산된다.</span>
            </p>
            <div style={{ marginTop: 36, display: 'grid', gridTemplateColumns: '80px 1fr', rowGap: 20, columnGap: 20, fontSize: 15 }}>
              <code style={{ fontFamily: '"JetBrains Mono"', fontSize: 18, color: C.accent, fontWeight: 600 }}>V</code>
              <div className="news" style={{ lineHeight: 1.55 }}>매치 참가 팀 집합. 보통 |V| ≈ 26–28.</div>
              <code style={{ fontFamily: '"JetBrains Mono"', fontSize: 18, color: C.accent, fontWeight: 600 }}>E</code>
              <div className="news" style={{ lineHeight: 1.55 }}>두 팀 (u, v) 사이에 damage event 가 한 번이라도 있으면 추가되는 무방향 엣지. 자기 팀 데미지(아군 오사)는 제외.</div>
              <code style={{ fontFamily: '"JetBrains Mono"', fontSize: 18, color: C.accent, fontWeight: 600 }}>w(u,v)</code>
              <div className="news" style={{ lineHeight: 1.55 }}>(u, v) 가 주고받은 누적 damage. 가중치 모드에 따라 count / binary 도 가능.</div>
            </div>
          </div>
          <div style={{ background: C.cream, color: C.ink, padding: 36, borderRadius: 4,
                         alignSelf: 'start', borderLeft: `3px solid ${C.accent}` }}>
            <div style={{ fontSize: 10, letterSpacing: '.16em', textTransform: 'uppercase',
                          fontFamily: '"JetBrains Mono"', color: C.accent, fontWeight: 600, marginBottom: 20 }}>FORMULA</div>
            <div style={{ color: C.ink }}>
              <TeX tex={"G = (V, E)"} />
              <div style={{ marginTop: 12 }}>
                <TeX tex={"w(u,v) = \\sum_{e \\in \\mathrm{events}_{uv}} \\mathrm{dmg}_e"} />
              </div>
            </div>
            <div style={{ marginTop: 28, paddingTop: 20, borderTop: `1px solid ${C.rule}`,
                          fontSize: 12, fontFamily: '"JetBrains Mono"', color: C.muted, lineHeight: 1.7 }}>
              <span style={{ color: C.accent, fontWeight: 600 }}>▸ 도메인 번역</span><br />
              노드 = 팀, 엣지 = 교전, weight = 데미지 강도.<br />
              <span style={{ color: C.accent, fontWeight: 600, display: 'inline-block', marginTop: 8 }}>▸ 구현</span><br />
              network_analysis/team_network.py<br />
              :: _events_to_team_graph()
            </div>
          </div>
        </div>
      </section>

      {/* §2 — Stats as a magazine spread, oversized numbers */}
      <section style={{ padding: '120px 80px 0' }}>
        <SectionMarkB num="02" title="이 데이터셋의 그래프, 한눈에" subtitle="3,595 매치 · Baltic_Main · squad" />
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 32, marginTop: 40 }}>
          <GiantStat label="|V|" value="26.4" hint="매치당 평균 팀 수" />
          <GiantStat label="|E|" value="42.6" hint="실제 교전 쌍" />
          <GiantStat label="density" value="0.131" hint="가능한 343쌍 중 13.1%만 부딪힌다" />
          <GiantStat label="modularity Q" value="0.591" hint={<>Newman 임계 0.3을 한참 넘는, <span style={{color: C.accent, fontStyle: 'normal'}}>강한 community 구조</span></>} color={C.accent} />
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 32, marginTop: 56 }}>
          <GiantStat label="avg degree" value="3.24" hint="한 팀이 직접 교전하는 다른 팀 수" />
          <GiantStat label="clustering" value="0.398" hint="삼각 교전 밀도" />
          <GiantStat label="diameter" value="5.7" hint="가장 먼 두 팀의 단계 수" />
          <GiantStat label="avg path" value="2.76" hint="평균 최단 경로 길이" />
        </div>

        {/* Pull quote */}
        <blockquote className="news" style={{
          marginTop: 80, padding: '32px 0 32px 32px', borderLeft: `4px solid ${C.accent}`,
          fontSize: 28, lineHeight: 1.4, fontStyle: 'italic', maxWidth: 1000, color: C.ink, fontWeight: 400,
        }}>
          density 가 낮은 건 자연스럽다 — 200km² 에 26팀이 흩어져 있으니까.
          그러나 Q ≈ 0.59 는 그 적은 교전이 무작위가 아니라
          <span style={{ color: C.accent, fontStyle: 'normal', fontWeight: 600 }}> 지역 진영 형태로 묶여 있다 </span>
          는 신호다.
        </blockquote>
      </section>

      {/* §3 — Adjacency: text + matrix + degree histogram */}
      <section style={{ padding: '120px 80px 0' }}>
        <SectionMarkB num="03" title="인접 행렬 A 와 차수" />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 48, marginTop: 24, alignItems: 'start' }}>
          <div>
            <p className="news" style={{ fontSize: 18, lineHeight: 1.55, marginTop: 0 }}>
              그래프를 행렬로 보면 <code style={{fontFamily: '"JetBrains Mono"'}}>aᵢⱼ = w(i, j)</code> (없으면 0).
              A 는 대칭. 여기서 degree, weighted degree, density 가 모두 따라 나온다.
            </p>
            <div style={{ marginTop: 24, padding: 20, background: C.cream, borderRadius: 4 }}>
              <TeX tex={"A_{ij} = w(i,j),\\quad d_i = \\sum_j A_{ij}"} />
            </div>
          </div>
          <div>
            <div style={{ fontSize: 11, letterSpacing: '.12em', textTransform: 'uppercase',
                          fontFamily: '"JetBrains Mono"', color: C.muted, marginBottom: 12 }}>Fig · 인접 행렬</div>
            <AdjacencyHeatmapB />
          </div>
          <div>
            <div style={{ fontSize: 11, letterSpacing: '.12em', textTransform: 'uppercase',
                          fontFamily: '"JetBrains Mono"', color: C.muted, marginBottom: 12 }}>Fig · degree 분포</div>
            <DegreeHistogram width={340} height={180} color={C.ink} accent={C.accent} />
            <p style={{ fontSize: 13, color: C.muted, marginTop: 12, fontStyle: 'italic',
                        fontFamily: '"Newsreader"' }}>
              대부분 팀은 1–4 팀과만 교전.<br />
              long tail 의 8+ 팀이 H2 후보.
            </p>
          </div>
        </div>
      </section>

      {/* §4 — Centralities: 6-cell explorer */}
      <section style={{ padding: '120px 80px 0' }}>
        <SectionMarkB num="04" title="중심성 다섯 가지" subtitle="누가 그래프의 중심인가, 다섯 가지 다른 답" />
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 0, marginTop: 32, borderTop: `1px solid ${C.rule}`, borderLeft: `1px solid ${C.rule}` }}>
          {[
            { name: 'Degree',          tex: 'C_D(i) = \\frac{k(i)}{|N| - 1}',                                   ko: '이웃 팀 수의 정규화. 단순 직접 교전 빈도.', highSeed: 5 },
            { name: 'Closeness',       tex: 'C_C(i) = \\frac{|N| - 1}{\\sum_j d(i, j)}',                        ko: '다른 모든 팀까지 평균 거리의 역수. 클수록 그래프 중앙.', highSeed: 9 },
            { name: 'Betweenness ★',   tex: 'C_B(i) = \\sum_{j \\ne k} \\frac{\\sigma_{jk}(i)}{\\sigma_{jk}}',  ko: '다른 두 팀 쌍의 최단경로가 i 를 지나는 비율. 다른 클러스터를 잇는 중개자.', star: true, highSeed: 13 },
            { name: 'Eigenvector',     tex: '\\lambda c = A c',                                                  ko: '중요한 이웃과 연결될수록 본인도 중요.', highSeed: 17 },
            { name: 'PageRank',        tex: 'r = \\beta M r + (1 - \\beta)/N',                                   ko: 'eigenvector 의 random-teleport 확장. β = 0.85.', highSeed: 21 },
            { name: 'Local clustering ★', tex: 'C_i = \\frac{2 \\cdot \\text{tri}(i)}{k_i(k_i - 1)}',           ko: '팀 i 의 이웃끼리도 서로 교전한 비율. 삼각 교전 밀도.', star: true, highSeed: 25 },
          ].map((c) => (
            <div key={c.name} style={{ borderRight: `1px solid ${C.rule}`, borderBottom: `1px solid ${C.rule}`, padding: 24, background: c.star ? C.cream : 'transparent' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                <h4 style={{ margin: 0, fontSize: 17, fontWeight: 600 }}>{c.name}</h4>
              </div>
              <div style={{ marginTop: 12, fontSize: 13, minHeight: 36 }}>
                <TeX tex={c.tex} />
              </div>
              <p style={{ marginTop: 12, fontSize: 13, color: C.muted, lineHeight: 1.5,
                          fontFamily: '"Newsreader"', fontStyle: 'italic' }}>{c.ko}</p>
              <div style={{ marginTop: 16, height: 90, background: C.paper, border: `1px solid ${C.rule}`, padding: 4 }}>
                <MiniNetwork width={300} height={82} seed={c.highSeed} n={20}
                  bg={C.paper} edgeColor="#cdc5b3"
                  nodeColors={[C.ink, C.accent2, C.accent3]}
                  nodeStroke={C.paper}
                  highlight={c.highSeed % 8} />
              </div>
            </div>
          ))}
        </div>
        <p style={{ marginTop: 24, fontSize: 13, color: C.muted, fontStyle: 'italic',
                    fontFamily: '"Newsreader"' }}>
          ★ = H2 가설의 핵심·보조 지표. distance 기반 지표는 distance = 1 / weight 로 변환해 사용한다.
        </p>
      </section>

      {/* §5 — Community visual + formula */}
      <section style={{ padding: '120px 80px 0' }}>
        <SectionMarkB num="05" title="커뮤니티 · 모듈러도 Q" />
        <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1fr', gap: 64, marginTop: 24 }}>
          <div>
            <p className="news" style={{ fontSize: 18, lineHeight: 1.55, marginTop: 0 }}>
              Louvain greedy 알고리즘으로 modularity Q 를 최대화하는 팀 분할을 찾는다.
              우리 데이터셋의 평균 <span className="numeric" style={{color: C.accent, fontWeight: 600}}>Q = 0.591</span>은
              Newman (2004) 의 의미 임계 0.3 을 한참 넘는다.
            </p>
            <div style={{ marginTop: 24, padding: 20, background: C.cream, borderRadius: 4 }}>
              <TeX tex={"Q = \\frac{1}{2m}\\sum_{ij}\\Big[A_{ij} - \\frac{k_i k_j}{2m}\\Big]\\delta(c_i, c_j)"} />
            </div>
            <ul style={{ marginTop: 24, paddingLeft: 0, listStyle: 'none', fontSize: 14, lineHeight: 1.7 }}>
              <li><strong>community_size</strong> — 한 팀이 속한 community 의 팀 수.</li>
              <li><strong>community_density</strong> — community 서브그래프의 density.</li>
              <li><strong>position_label</strong> — betweenness 사분위. 상위 25% = core, 하위 25% = peripheral, degree=0 = isolated.</li>
            </ul>
          </div>
          <div style={{ background: C.cream, padding: 24, borderRadius: 4 }}>
            <div style={{ fontSize: 11, letterSpacing: '.12em', textTransform: 'uppercase',
                          fontFamily: '"JetBrains Mono"', color: C.muted, marginBottom: 12 }}>Fig · Louvain 분할</div>
            <MiniNetwork width={460} height={320} seed={31} n={26}
              bg={C.cream} edgeColor="#bdb39e"
              nodeColors={[C.ink, C.accent2, C.accent3]}
              nodeStroke={C.cream} />
            <div style={{ marginTop: 14, display: 'flex', justifyContent: 'space-between', fontSize: 12,
                          color: C.muted, fontFamily: '"JetBrains Mono"' }}>
              <span>3 communities</span><span>Q = 0.62</span>
            </div>
          </div>
        </div>
      </section>

      {/* §6 — Hypothesis map: as a "what feeds into what" diagram */}
      <section style={{ padding: '120px 80px 80px', marginTop: 80, background: C.cream, color: C.ink }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 16 }}>
          <div style={{ width: 28, height: 1, background: C.accent }} />
          <span style={{ fontSize: 11, letterSpacing: '.16em', textTransform: 'uppercase',
                         fontFamily: '"JetBrains Mono"', color: C.accent, fontWeight: 600 }}>06 · Mapping</span>
        </div>
        <h2 className="news" style={{ fontSize: 56, lineHeight: 1.05, fontWeight: 500, letterSpacing: '-0.025em', margin: 0, maxWidth: 900 }}>
          이 정의가 <span style={{color: C.accent}}>H1·H2·H3</span> 에<br />어떻게 들어가는가
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 32, marginTop: 64 }}>
          {[
            { tag: 'H1', title: '외곽 파밍 vs 교전 참여', uses: ['degree', 'combat_per_min, dmg_per_min', 'centroid_first 의 map 중앙 거리'], tex: 'P_i = z(\\text{combat\\_pm}) + z(\\text{dmg\\_pm}) + z(\\deg)' },
            { tag: 'H2', title: '다중 교전 중심 (C)', uses: ['betweenness', 'degree centrality', 'local_clustering'], tex: 'C_i = z(c^{bet}) + z(c^{deg}) + z(\\text{clust})' },
            { tag: 'H3', title: '존버 vs 활발 이동', uses: ['team_centroid 시퀀스', 'trajectory_length_m', 'displacement_per_min'], tex: 'L_i = \\sum_t \\| c_i(t) - c_i(t-1)\\|' },
          ].map((h) => (
            <div key={h.tag} style={{ background: C.paper, padding: 28, borderRadius: 4,
                                       borderTop: `3px solid ${C.accent}`,
                                       boxShadow: '0 1px 3px rgba(0,0,0,.04), 0 8px 24px rgba(20,15,5,.06)' }}>
              <div className="numeric" style={{ fontSize: 64, lineHeight: 1, fontWeight: 500, color: C.accent, letterSpacing: '-0.03em' }}>{h.tag}</div>
              <h3 className="news" style={{ fontSize: 22, marginTop: 14, marginBottom: 18, fontWeight: 500 }}>{h.title}</h3>
              <div style={{ fontSize: 10, letterSpacing: '.14em', textTransform: 'uppercase', fontFamily: '"JetBrains Mono"', color: C.muted, fontWeight: 600 }}>USES</div>
              <ul style={{ paddingLeft: 0, listStyle: 'none', margin: '8px 0 20px', fontSize: 13.5, lineHeight: 1.6 }}>
                {h.uses.map((u) => (
                  <li key={u} style={{ paddingLeft: 16, position: 'relative' }}>
                    <span style={{ position: 'absolute', left: 0, color: C.accent }}>▸</span>{u}
                  </li>
                ))}
              </ul>
              <div style={{ paddingTop: 16, borderTop: `1px solid ${C.rule}`, color: C.ink }}>
                <TeX tex={h.tex} />
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function SectionMarkB({ num, title, subtitle }) {
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 14 }}>
        <span className="numeric" style={{ fontSize: 13, color: B.accent, fontFamily: '"JetBrains Mono"', letterSpacing: '.1em', fontWeight: 600 }}>{num}</span>
        <span style={{ flex: 1, height: 1, background: B.rule }} />
      </div>
      <h2 className="news" style={{ fontSize: 48, fontWeight: 500, letterSpacing: '-0.025em',
                                     margin: '20px 0 4px', lineHeight: 1.05 }}>{title}</h2>
      {subtitle && <div className="news" style={{ fontSize: 17, color: B.muted, fontStyle: 'italic' }}>{subtitle}</div>}
    </div>
  );
}

function AdjacencyHeatmapB({ size = 14 }) {
  const { edges } = React.useMemo(() => makeNetwork(3, size), []);
  const W = Array.from({ length: size }, () => Array(size).fill(0));
  edges.forEach(e => { W[e.a][e.b] = e.w; W[e.b][e.a] = e.w; });
  const cell = 22;
  const dim = size * cell;
  return (
    <svg width={dim} height={dim} viewBox={`0 0 ${dim} ${dim}`}>
      {W.map((row, i) =>
        row.map((v, j) => {
          const op = v === 0 ? 0.04 : 0.25 + Math.min(v, 1) * 0.7;
          const fill = v === 0 ? B.muted : B.accent;
          return <rect key={`${i},${j}`} x={j * cell} y={i * cell} width={cell - 1} height={cell - 1}
            fill={fill} fillOpacity={op} />;
        })
      )}
    </svg>
  );
}

window.OptionBPudding = OptionBPudding;


Object.assign(window, { OptionBPudding });
