# PUBG Paper Writing Notes

이 파일은 본 논문 작성 시 유지해야 할 규칙과 2026-06-11 작업 내용을 정리한다.

## 작성 규칙

- **문체는 IEEE ToG류의 선언적·정돈된 학술 산문을 따른다 (2026-06-11 추가).**
  - 수사의문문 금지 (예: "Why a graph?" → "A second representational choice concerns relations."처럼 선언문으로).
  - 한 문단 = 하나의 주장, 문단 첫 문장이 그 주장을 직접 진술한다.
  - Methods subsection 제목은 "내용 (RQ태그)" 형식: 내용이 먼저, RQ 매핑은 괄호 접미사 — 설계 속성(누수 안전 등)은 제목이 아니라 본문 속성 문장으로 쓴다.
  - RQ 태그는 RQ1, RQ2-i, RQ2-ii로 통일 (표·제목·본문 prose "part (i) of RQ2" 호환).

- 논문은 Jennifer Widom style을 따른다.
  - 먼저 큰 문제와 직관을 제시한다.
  - 개념은 처음 등장할 때 정의한다.
  - 세부 구현보다 독자가 이해해야 할 추상화와 데이터 흐름을 먼저 설명한다.
  - 표나 그림에 이미 있는 내용을 본문에서 길게 반복하지 않는다.

- 모든 새 섹션은 영어 버전과 한글 버전을 함께 만든다.
  - 영어 main 파일에 들어가는 섹션과 한글 mirror 섹션을 같이 유지한다.
  - RQ, figure, table, caption의 논리 구조도 양쪽이 일치해야 한다.

- 4-page, 2-column 원고를 목표로 한다.
  - 문장은 짧게 쓴다.
  - 한 문단은 하나의 역할만 갖게 한다.
  - feature 나열은 본문보다 표로 처리한다.
  - 코드명, raw event name, 구현 view/table name은 본문에 직접 노출하지 않는다.
  - 필요한 경우 raw telemetry의 "의미"를 학술적으로 서술한다.

- RQ와 Methods는 반드시 1:1로 매핑되어야 한다. (2026-06-11 RQ 2개 체계로 개편 — 구 RQ1·RQ2는 둘 다 "표현" 질문의 중복이라 병합)
  - RQ1 (representation): telemetry를 phase-정렬 STDB로 재구성하고, **위험 관련 조건이 언제 어디서 변하는지(집계가 버리는 정보)를 보존하는** team-phase feature와 dynamic team graph로 요약한다. → Methods 3.1(STDB)·3.2(features)·3.3(graphs). ("어떻게 표현하는가"만 묻는 구현 질문이 되지 않도록 보존 목적을 RQ에 내장 — 2026-06-11)
  - RQ2 (prediction): (i) feature만으로 next-phase elimination hazard를 얼마나 정확히 예측할 수 있는지, (ii) dynamic inter-team relation이 feature-only model을 넘어 추가 신호를 주는지 검증한다. → Methods 3.4(task)·3.5(model). ("강한(strong)"은 결과 선취라 RQ 문장에서 제거; 구 문구 "(i) feature가 tabular baseline을 넘어"는 순환이라 폐기 — 2026-06-11)

- 그림 규칙
  - Overleaf 기준 figure 경로는 `figures/...`를 사용한다.
  - gameplay figure: `figure`, `width=0.48\textwidth`.
  - graph figure: `figure*`, `width=0.86\textwidth`.
  - 그림 안에는 시각적으로 필요한 정보만 넣고, 수식과 긴 정의는 본문으로 옮긴다.

## 현재 논리 구조

1. Introduction
   - PUBG telemetry가 시공간적이라는 문제를 제시한다.
   - post-match aggregate prediction의 한계를 짧게 제시한다.
   - phase-based survival work를 Related Work의 직접 출발점으로 예고한다.
   - RQ1/RQ2/RQ3를 제시한다.

2. Related Work
   - post-match aggregate PUBG prediction 연구.
   - Sagong et al.의 phase-based survival framework.
   - shrinking safe zone을 artificial habitat-contraction process로 보는 ecological analogy.
   - 본 연구의 차별점: official telemetry를 STDB와 dynamic team graph로 재구성한다.

3. Methods
   - 표 `tab:rq-method-map`에서 RQ와 방법론 산출물을 직접 매핑한다.
   - RQ1: telemetry event stream을 phase-aligned STDB로 변환한다.
   - RQ2: team-phase feature row를 만든다.
   - RQ3: leakage-safe next-phase hazard와 dynamic graph model을 평가한다.

## 2026-06-11 주요 작업

- `battleground_paper.pdf`를 검토했다.
  - 논문: Sagong et al., "Are You a Hunter or Camper?"
  - 핵심: phase-specific FightScore/ActiveScore로 strategy type을 만들고 next-phase survival을 예측한다.
  - 본 논문에서는 이 연구를 직접 출발점으로 삼되, representation을 STDB와 graph로 확장하는 방향으로 정리했다.

- Related Work를 compact하게 다시 작성했다.
  - 영어: `sections/02-background-related-work.tex`
  - 한글: `sections/02-background-related-work-ko.tex`
  - 추가 citation:
    - `sagong2025hunterCamper`
    - `fahrig2003fragmentation`
    - `nathan2008movement`

- Methods를 4-page용으로 압축했다.
  - 영어:
    - `sections/03-methods.tex`
    - `sections/03-data-stdb-modeling.tex`
    - `sections/04-prediction-task.tex`
    - `sections/05-dynamic-team-graph-model.tex`
  - 한글:
    - `sections/03-methods-ko.tex`
    - `sections/03-data-stdb-modeling-ko.tex`
    - `sections/04-prediction-task-ko.tex`
    - `sections/05-dynamic-team-graph-model-ko.tex`

- RQ 매핑을 수정했다.
  - Introduction의 RQ 문구와 Methods table/subsection의 핵심어가 맞도록 조정했다.
  - 특히 RQ1의 `dynamic safe and blue zones`, `spatial events`, `phase structure`가 Methods에도 드러나도록 수정했다.
  - RQ3는 tabular baseline과 dynamic graph relation을 분리해서 설명했다.

- Figure 설정을 Overleaf 기준으로 맞췄다.
  - `figures/team_phase_feature_example.png`
  - `figures/dynamic_heterogeneous_graph_example.png`

## 2026-06-11 후속 작업: 서론·방법론 정리

- Introduction 변형 6파일을 canonical 1쌍으로 수렴했다.
  - 영어: `sections/01-introduction.tex` (wrapper 제거, 본문 직접 포함)
  - 한글: `sections/01-introduction-ko.tex`
  - `sections/archive/`로 이동: 장문 서론 2종(`01-introduction-long-en/ko.tex`), redirect placeholder 2종, 기존 short-en/short-ko 원본
- 기여 진술은 RQ식으로 통일했다. RQ 목록 뒤에 pipeline과 결과를 미리 보여주는 문단 하나를 추가했다 (bullet 기여 목록은 장문 서론과 함께 archive). 서론의 수치는 21,824 matches만 사용한다.
- STDB 약어를 서론 첫 등장 위치에서 정의했다 (기존에는 Methods 표에서 정의 없이 처음 등장했다).
- RQ2 문구를 Methods 표와 1:1이 되도록 재정렬했다: "어떻게 변화하는가" → "위험 상태를 기술하는 phase 단위 feature로 어떻게 요약할 수 있는가". Methods가 기술통계 없이 feature 도출로 RQ2에 답하는 현재 구조와 일치시킨 것이다.
- `04-prediction-task`: "current split" 표현을 "match-level split"로 교체했고, 한글 mirror에 calibration/loss 진단 지표 문장을 보강해 영어판과 parity를 맞췄다.

## 2026-06-11 후속 작업 2: 방법론 가이드 정합 수술 (코드 검증 기반)

코드(stdb/phase_edge_etl.py, stdb/queries.py, model_stgcn/dyn_stgnn.py, model_stgcn/baselines.py, 체크포인트 JSON)에서 사실관계를 검증한 뒤 다음을 수정했다.

- **구조**: 03을 RQ별 subsection으로 재편 — `RQ1: Phase-Aligned STDB Construction` + `RQ2: Team-Phase Features`(신설, `sec:methods-features`). RQ1·RQ2가 paragraph로 끼어 있던 비대칭을 해소했다. 게임 맥락 문단(서론과 중복)은 삭제했다.
- **서론**: phase 정의를 첫 등장 위치(KRAFTON 문단)에 삽입했다("번호가 매겨진 phase, 다음 safe zone 제시 후 수축"). RQ3를 (i) feature / (ii) graph relation 2단으로 명시해 04("first part")·05("second part")가 받을 곳을 만들었다. "activity space"→"playable area"로 용어를 통일했다.
- **모집단 명시(코드 검증)**: full-DB 21,824는 TPP squad 한정이 아니라 **12개 map × 전 모드(solo/duo/squad × TPP/FPP)**, TPP squad 15,213이 최대 stratum. 03 RQ1에 명시했다. ⚠ 08 TODO의 "FPP not part of the primary experiment" 전제는 틀림 — 08 작성 시 "FPP 포함되나 소수(약 2.9k)"로 수정할 것.
- **Feature 표 신설**(`tab:feature-groups`): 6개 group 대표 수치, 총 66. 정규화는 z-score가 아니라 **feature별 고정 scale 상수 + heavy-tail log 압축**(train-fit 통계 아님)으로 정정 서술.
- **04**: cut phase를 본문에서 정의, censoring/horizon 추가(phase 1–8 사용, landing interval 제외, cut phase 1–7; 다음 phase 미관측 시 label 없음 → 우승팀은 음성 label만), (m,i,k) 중복 정의 제거(3.2 참조), split 수치·metric 문장은 06으로 이관.
- **05 엣지 정의 신설(코드 검증)**: proximity = 대칭, exp(−avg dist/200m); combat = attacker→victim 방향성, log(damage + knockdown/kill 고정 가중); co-pressure = "safe zone 밖 또는 boundary 150m 이내" 압박 상태의 두 팀이 200m 이내일 때, 공동 압박 시간 비율 × 거리 감쇠. 채널별 row-normalize, proximity만 self-loop. 모두 phase k 관측만 사용.
- **05 아키텍처 교정(코드 검증)**: 입력 embedding(66→64) + **ST block ×2**(gated causal temporal conv → channel별 집계+학습된 softmax weight 결합 heterogeneous graph conv → gated temporal conv, LayerNorm+residual), 보조 head는 final-placement·top-k(loss weight 0.3), 총 ~0.34M params. 기존 "T-S-T 1회" 서술은 부정확했음.
- **06**: HGB가 **72컬럼 view**(raw timing + map/mode 범주형 — 그래프 노드에 없는 컨텍스트 포함)임을 명시, "Adam-style" → AdamW(lr 1e-3, wd 1e-4, cosine annealing, grad clip)로 구체화, 보조 head weight 0.3 명시, Calibration 문단을 Metrics 문단으로 통합(04에서 이관).
- 검증: 본문 \ref↔\label 전수 일치 확인, EN/KO 라벨 parity 확인(03/04/05), 잔여 초안 표현("current split", "Adam-style", "activity space") 부재 확인.

## 2026-06-11 후속 작업 3: 서론 피드백 반영 + RQ 2개 체계 개편

사용자 피드백을 반영했다.

- KRAFTON 문단: 사용자 제공 버전으로 교체 (명시적 "번호가 매겨진 phase" 정의 문장 제거, "각 phase에서 게임은 다음 safe zone을 제시하고 수축한다"로 암묵 정의). EN 미러 동일.
- "누수 방지(leakage-safe)" 용어는 서론에서 제거 — 정의 없이 등장하는 전문용어이므로 미리보기 문단에서는 "관측 구간이 항상 예측 구간보다 앞서도록 유지한다"는 평문으로 표현. 용어 자체는 Methods 3.3에서만 사용.
- 미리보기 문단 재작성: 미도입 용어(relation channel 이름, gradient-boosting baseline, no-graph neural control, ranking/calibration, match-paired test) 제거, 서론에서 이미 정의된 개념(phase, safe zone, STDB, hazard, team graph)만으로 4문장 요약.
- "왜 graph인가" 동기 문단을 서론에 추가: 탈락 위험은 주변 팀 배치(근접·교전·공간 공유)에 의존하고, 팀별 속성 row로는 관계를 표현할 수 없다는 논리. RQ2(ii)와 Methods 3.4의 존재 이유를 서론에서 제시.
- 02 생태 비유 순서 교정: "생태학 문헌이 먼저 있고(habitat fragmentation·movement ecology) → safe zone이 같은 종류의 수축을 부과하므로 그 lens를 분석적 비유로 차용 → 결론: 생존 조건의 시공간적 모델링" 순서로 재배열. EN/KO 동일.
- **RQ 체계 3개 → 2개**: 구 RQ1(STDB 표현)과 구 RQ2(feature 요약)는 같은 표현 질문의 두 단계라 RQ1로 병합. 구 RQ3는 RQ2가 되고 (i) feature / (ii) graph relation 2단 구조 유지. 전파: 서론 RQ 목록, 03-methods pipeline 표(4행: RQ1×2, RQ2(i), RQ2(ii)), 3.2 제목 "RQ1: Team-Phase Features", 04 제목·본문 "RQ2 (i)", 05 제목·본문 "RQ2 (ii)". EN/KO 모두 적용. ⚠ docs/(RESEARCH_STATUS, PROJECT_FLOW)의 3-RQ 서술은 과거 스냅샷이므로 논문이 정본.

## 2026-06-11 후속 작업 4: graph 구조 서술 보강

구현된 graph의 유형을 05에 명명했다: **이산 시간(discrete-time) snapshot 동적 graph** — match당 \(G_1,\dots,G_T\) 가중 이질 snapshot sequence (연속시간 이벤트 graph 아님; related work에서 동적 GNN 문헌과 연결할 고리). 추가로 proximity가 거리 컷오프 없는 전 쌍 연결임을, row-normalize된 edge weight가 집계 계수로 직접 쓰임을 명시했다. EN/KO 동일.

## 2026-06-11 후속 작업 5: 07 수치 동기화 (산출물 검증)

- **정본 확정**: `model_stgcn/checkpoints/full_db_reexperiment/phase_result_analysis_edges/` + `calibration_edges/` (6/10 17:42, 5개 모델 단일 분석런). 구 분석런 `phase_result_analysis/`(6/10 12:34, 3모델)는 HGB 재평가값이 미세하게 달라 폐기 — 신경망 행은 두 런에서 동일, HGB만 차이. 논문 표는 한 분석런에서 나와야 하므로 5모델 런이 정본.
- 07 수정 3건: 메인 표 HGB Acc 0.7489→**0.7483**; full vs tabular Wilcoxon 3행 0.0122/8.18e-77→**0.0120/3.30e-74**, 0.0117/1.51e-39→**0.0113/2.60e-37**, 0.0158/1.17e-37→**0.0154/9.28e-36**; 말미 stale 문장("edge ablation is the next step")을 채널 귀속 서술로 교체(proximity 최대 단일 증분 +0.0060 AUC; full은 두 부분 변형 대비 match-paired 유의; seed 반복 확인은 future work).
- 나머지 수치 전수 대조 완료: 메인 표 5행, paired 12행 중 9행, calibration 표 5행, 본문 델타(0.0100/0.0112/0.0130/0.0181), 양성률 0.3245, mean prob 0.3260 — 모두 신 분석런과 일치.
- 주의: prox+combat vs prox(combat 단독 증분)의 paired test는 산출물에 없음 — combat 증분(+0.0014)에 유의성 주장 금지. full vs prox / full vs prox+combat만 검정 존재.

## 2026-06-11 후속 작업 6: 서사 감사 및 흐름 수정

RQ→설계→결과가 한 서사로 닫히는지 전수 감사 후 수정했다.

- **RQ2(i) 순환 논리 교정**: "(i) feature가 강한 tabular baseline을 넘어 기여하는가"는 baseline이 그 feature를 쓰므로 검증 불가능한 질문이었음. "(i) feature만으로 얼마나 잘 예측되는가(절대 성능), (ii) graph 관계가 feature-only model을 넘어 무엇을 더하는가"로 교정. 전파: 서론 RQ2, 04 도입(beyond 구문 제거), 06 도입.
- **07에 RQ 루프 클로저 신설**: 토론 문단을 RQ 구조로 재작성 — RQ1은 구성으로 답함(27,491→21,824 funnel 참조 + 모든 수치가 그 표현에서 산출), RQ2(i)는 feature-only AUC 0.765–0.768 + near-nominal calibration, RQ2(ii)가 중심 발견(관계의 추가 신호). 말미에 habitat-contraction 회수(02 호응).
- **07 MCE 인용 제거**: 표에 없는 maximum calibration error 대신 "ECE가 full 대비 약 5배"(0.0075 vs 0.0015, 산출물 검증)로 교체.
- **연결 조직**: 03 도입부가 RQ1(3.1–3.2 구축)/RQ2(3.3–3.4 검증)를 명시 호명, 06 도입부가 RQ2를 호명.
- **서론 문단 재구획**: 2문장 토막 문단(PUBG 소개)을 telemetry 문단과 병합, "왜 graph인가"를 별도 문단으로 분리(한 문단 한 역할).
- **02 인용 위치 교정**: Allison/Singer를 "baseline 검증" 문장에서 "이산 시간 hazard 정식화" 절로 이동.
- EN/KO 동일 적용(06·07은 KO 미러 없음 — 기존 관행 유지).

## 2026-06-11 후속 작업 7: 깔때기 해명 + feature ablation full-DB 재실행

- **"왜 27,491을 다 쓰지 않았나" 해명 (DB 산출 문서로 검증)**: 27,491→21,833 탈락 5,658개 = Tutorial_Main 3,696 + Range_Main clansolo 1,571 + Range_Main solo 391 — 전부 튜토리얼/훈련장 세션(비경쟁 플레이)으로 합이 정확히 일치. 나머지 9개만 비정상 telemetry(위치 기록 없음 3, next-phase label 정의 불가 3, edge cache 없음 3). 03 RQ1 절에 두 문장으로 명시(EN/KO). 08 작성 시 "filtering and missingness" TODO의 근거로 재사용.
- **RQ2(i)의 비교 증거 = feature ablation**: `model_stgcn/feature_ablation.py` — 선행연구식 FightScore/ActiveScore를 충실히 재구현(prior_scores 5피처: z-score 합성 점수+strategy type+type survival rate; fit은 train에서만) → prior_raw(12) → global_state(30) → conventional(57) → +stdb_spatial(72) → full(76)의 가산 비교. 구 스냅샷(squad 16,340매치, 82,662셀) 결과: prior_scores 0.6691 → full 0.7601 AUC, match-paired ΔAUC +0.088/ΔAP +0.142(p≈6e-222); stdb_spatial 증분 +0.0035(p=2.5e-10); full vs +stdb_spatial 차이 없음(−0.0005, p=0.987 — 점수 피처 4개는 기여 없음).
- **Full-DB 재실행 완료 및 논문 반영 (같은 날)**: aligned 134,245셀(메인 표와 동일), 72피처 구성이 메인 HGB baseline을 0.0001 AUC 이내로 재현(별도 fit임을 06에 명시). 결과: prior_scores(5) 0.6490 → prior_raw(12) 0.7184 → conventional(57) 0.7522 → +stdb_spatial(72) **0.7652** → +scores(76) 0.7649(무기여, p=0.75). Match-paired **full vs prior_scores ΔAUC +0.112/ΔAP +0.170**(p 수치 정밀도 아래, rank-biserial 0.957/0.967 — 구 스냅샷 +0.088보다 확대), **stdb_spatial 증분 +0.0123 AUC**(p=8.79e-79 — 구 +0.0035의 3.5배, exposure>trajectory), stdb_spatial 단독(16) vs prior_scores +0.057. 반영: 06 "Representation Conditions" 문단 신설 + Population Consistency에서 feature ablation을 구 스냅샷 목록에서 제거, 07에 `tab:feature-ablation` 표(6행)+해석 문단 신설, 루프 클로저의 RQ2(i)를 비교 증거("같은 learner가 선행식 점수로 제한되면 0.112 AUC 손실")로 강화. docs/FULL_DB_REEXPERIMENT_REPORT.md에 §8로 전체 결과 기록.

## 2026-06-11 후속 작업 8: 구 스냅샷 robustness 3종의 full-DB 전환

- **Phase-wise: 이미 full-DB 산출물 존재 확인** (`phase_result_analysis_edges/hazard_phase_metrics.csv`, aligned 134,245셀과 동일 모집단). Full vs none ΔAUC: p1 +0.0125, p2 +0.0111, p3 +0.0070, p4 +0.0103, p5 +0.0111, p6 +0.0192, p7 +0.0245 — **전 phase 양수, 후반이 최대**. ⚠ 구 스냅샷의 "이득은 초·중반 집중, 후반 약화" 서사는 full DB에서 성립하지 않음 — 논문에 구 서사 반입 금지. 07 paired 문단에 한 문장 반영("positive at every cut phase, largest in the sparse late phases"), 보고서 §9 기록.
- **Edge sanity**: 추론 전용이라 재학습 불필요. `edge_sanity.py`의 도메인 규약을 shim 디렉토리(`full_db_reexperiment/sanity_root/All__current/` — 정본 split 사본 + full 체크포인트 symlink)로 충족시켜 node_shuffle ×5 실행 중 → `full_db_reexperiment/edge_sanity/`.
- **Seed 반복**: 정본 split 고정 + 학습 seed만 변경(구 실험과 같은 13, 2026; 42는 기존 메인 런) × {full, none} = 학습 4회 + eval. sanity 종료 후 순차 실행 → `full_db_reexperiment/seed_repeats/`. 오케스트레이션: `scripts/run_full_db_robustness.sh`, 로그 `full_db_robustness.log`. 예상 총 ~10–12시간.
- 완료 후 할 일: seed별 full−none ΔAUC/ΔAP 집계(42/13/2026 3쌍), sanity ΔAUC/ΔAP/ΔBCE 집계 → 07 한두 문장 + 08 재료로 반영, 보고서 §10 갱신. 이로써 구 스냅샷 의존 결과가 전부 소멸.

## 2026-06-11 후속 작업 9: 방법론에 검증 설계 선언 보강

결과 절에 등장하는 실험(representation ablation, shuffle/seed 통제)이 방법론에 선언되지 않은 비대칭을 해소했다.

- 03 pipeline 표: RQ2(i) 산출물에 "representation ablation", RQ2(ii) 산출물에 "edge ablations and controls" 추가.
- 04 말미: (i)의 답을 "feature-only 절대 성능 + 선행식 score로 축소해 가는 representation ablation" 두 방향으로 구한다고 선언 (Experiments 절 참조).
- 05 말미: (ii)의 반증 기준 2개 선언 — "이득이 '누가 누구와 관계 맺는가'에서 온다면 추론 시 team identity shuffle로 사라져야 하고, 학습 seed를 바꿔도 유지되어야 한다" (결과 주장 아님, 설계 선언 — 실험은 실행 중).
- 05: HGB baseline 서술을 "tabular view of the same team-phase rows"로 정밀화 (06의 72컬럼 서술과 정합).
- EN/KO 동일 적용, \ref 무결성 확인.

## 2026-06-11 후속 작업 10: 서론 전면 재작성 + 제목 체계 교정 (ToG 스타일)

사용자 피드백("매핑 안 됨, 지저분하고 산만, RQ2 제목 오류") 반영.

- **서론 전면 재작성** (EN/KO): 4문단 + RQ + 결과 미리보기의 정돈 구조 —
  ① 맥락(telemetry·BR 구조) ② 사례+데이터+phase 정의+시공간성 ③ 난점(위험 변동)과 선행 갭 ④ 응답(STDB+hazard 정식화, 관계 선택의 선언적 근거). "Why a graph?" 수사의문문 제거. 미리보기 문단에 핵심 수치 위계 반영: 표현 +0.112 AUC가 지배, 그 위에 graph 관계의 유의한 추가 이득.
- **RQ 문구 정돈**: RQ1 "phase-정렬 STDB(이동 팀·동적 자기장·공간 사건·phase interval)로 재구성하고 team-phase feature로 요약할 수 있는가" — methods 표의 keyword와 1:1. RQ2 "(i) feature만으로 얼마나 정확히 / (ii) graph 관계가 feature-only model을 넘어 추가 신호".
- **제목 체계 교정**: "RQ2: 누수 방지 예측 문제"는 RQ 내용이 아니라 설계 속성을 제목으로 단 오류 → **"Next-Phase Hazard Prediction Task (RQ2-i)"**로 교정, 누수 안전성은 본문 한 문장("어떤 입력도 cut phase 종료를 넘지 않으므로 구성상 누수 안전")으로 이동. 전체 체계: Phase-Aligned STDB Construction (RQ1) / Team-Phase Features (RQ1) / Next-Phase Hazard Prediction Task (RQ2-i) / Dynamic Team Graph Model (RQ2-ii). Pipeline 표 태그도 RQ2-i/RQ2-ii로 통일.
- 검증: refs/labels·cites 전수 통과, EN/KO 동기.

## 2026-06-11 후속 작업 11: 방법론 ToG식 재작성 (참고: battleground_paper.pdf §4)

Sagong et al.의 방법론 작성 관례를 직접 읽고 차용했다: ① 방법론 도입부에서 전체 단계를 명시적으로 열거하고 개요 그림에 연결("proceeds in four steps, summarized in Figure/Table"), ② dataset 절에 범위·제외 근거 명시, ③ 정의·수식 중심의 선언적 문단, ④ 한 문단 한 역할.

- **03 도입부**: "First/Second/Third/Fourth" 4단계 열거형으로 재작성 — 각 단계가 subsection ref와 RQ 귀속을 명시 (1–2단계=RQ1, 3–4단계=RQ2).
- **3.1**: 역할별 3문단으로 분리 — (a) 데이터 출처·모집단 범위, (b) 객체 모델(4 object types), (c) funnel과 제외 근거(별도 문단으로 승격).
- **3.4**: 과적재 문단 분해 — P1 목적+graph 형식화(G_k, 이산시간 동적 graph 분류를 도입부로 이동: 객체 정의 → 관계 정의 순서), P2 relation 3채널 정의+후처리, P3 모델 아키텍처, P4 head(hazard/보조)+파라미터, P5 평가 설계(ablation+baseline+falsification 통제). **그림 2(fig:dynamic-graph-example)가 본문에서 미참조 상태였던 것을 발견·수정** — 이제 4개 그림 전부 본문 참조 1회.
- 04는 후속 작업 10에서 이미 재작성됨(변경 없음). EN/KO 동기, refs/labels 전수 통과.

## 2026-06-11 후속 작업 12: 그림 2종 Figma 재디자인 → SVG 파이프라인 확정

- 사용자 요청: matplotlib 그림을 Figma/Claude Design 기반의 학술 논문 스타일로 재제작 (참고: "STGCN Architecture"·"Pipeline Overview" 구글 이미지 관례).
- **Figma 파일**: https://www.figma.com/design/cD94Pj3nD8rJKbeKpSXQJ4 ("PUBG Paper Figures") — Fig1(pipeline)은 Figma에서 완성·검증. ⚠ **Starter 플랜 View 시트는 Figma MCP 호출 월 6회 한도** — Fig1 빌드로 소진되어 Fig2와 고해상도 export는 MCP로 불가.
- **확정 파이프라인**: Figma에서 검증한 디자인 스펙(좌표·파스텔 팔레트·Helvetica 계열 타이포·라운드 카드)을 `scripts/make_paper_figures_svg.py`가 SVG로 1:1 재현 → headless Chrome으로 2.5× 렌더(약 4200px 폭) → `paper/figures/pipeline_overview.png`·`dynstgnn_architecture.png` (파일명 유지로 LaTeX 무변경). SVG 원본은 `paper/figures/src/`.
- 디자인 관례: STGCN식 (b)전체망+(c)블록 줌 샌드위치(temporal→spatial 그룹 박스→temporal→LayerNorm→⊕residual rail), EvolveGCN식 (a)스냅샷 시퀀스 G₁…G_T, 파이프라인은 단계 카드+RQ 브레이스. 엣지 색은 논문 Figure 2 범례와 동일(prox 파랑/combat 빨강 방향/co-pr 보라 점선).
- 구 matplotlib 생성기(make_paper_diagrams.py)는 삭제 — SVG 생성기가 단일 정본.

## 2026-06-11 후속 작업 13: 그래프 subsection 독립 + 표준 그래프 서술 보강

사용자 지적 3건 반영: ① graph 구성이 모델 절에 섞여 있던 것, ② graph 논문 표준 요소(완결된 형식 정의·전파 수식·그래프 통계) 부재, ③ 설계 이점 미서술.

- **Methods 5-subsection 체계로 재편**: 3.1 STDB (RQ1) / 3.2 Features (RQ1) / **3.3 Dynamic Team Graphs (RQ1, 신설 — graph "구성"은 표현 작업이므로 RQ1로 귀속; Fig1에서 graphs 카드가 RQ1 브레이스 아래 있던 것과 이제 본문이 일치)** / 3.4 Hazard Task (RQ2-i) / 3.5 DynamicSTGNN (RQ2-ii, 모델 전용). 브리지 5단계 열거, pipeline 표 5행(RQ1×3), 서론 RQ1에 "dynamic team graph" 추가. fig:dynamic-graph-example은 3.3으로 이동.
- **3.3 표준 그래프 서술**: 형식 정의(V_k=co-observed teams, 가중 이질, 채널별 대칭/방향 명시), snapshot 동적 분류, **그래프 통계(DB 2,000매치 표본 실측: phase당 평균 ~14팀, proximity 밀도 정확히 1.0=완전 가중, combat 평균 ~14 방향 edge, co-pressure ~7 pair — 희소)**.
- **3.3 설계 이점 문단(4성질)**: 이질 채널=위험 기제 분리+학습된 가중, 컷오프 없는 완전 가중 prox=거리 임계 hyperparameter 제거+순서 보존, 방향성 combat=교전 주체 보존, snapshot+row-norm=hazard의 phase 시계 정렬+생존 팀 수 감소에도 집계 규모 유지.
- **3.5에 전파 수식 추가**: H' = ReLU(Σ_r α_r Ā_k^r H W_r + H W_0), α=softmax(a) — 코드(dyn_stgnn.py)와 일치 검증된 형태.
- EN/KO 동기, refs/labels·중복 라벨 전수 통과. 그래프 통계 출처: stdb.team_phase_edge_cache 2,000매치 표본 질의(전수 재계산 시 수치 재확인 필요).

## 2026-06-11 후속 작업 14: 서론 전면 재작성 v2 (25개 항목 심층 피드백 반영)

사용자 피드백의 5대 핵심 문제(핵심 문제 제기 지연 / 기여 분산 / 선행연구 차이 불명 / RQ 느슨 / 용어 정의 부족)를 반영해 서론을 다시 썼다. 새 구조와 결정:

- **P1 = 문제 선행**: "팀의 운명은 최종 결과가 아니라 생존 과정" — 위험의 시간 변동을 첫 문단의 축으로. **시장 규모 문장 삭제**(newzoo 인용 제거). 집계의 한계를 시간 손실 예시로 구체화(같은 damage라도 phase 2와 후반 좁은 구역에서 위험이 다름; 선택된 이동 vs 강요된 이동).
- **P2 = 구조적 적합성**: "대표적이고 선도적인" 삭제 → PUBG가 적합한 구조적 이유(100인, phase 구조, 공식 API의 시간순 이벤트 스트림). **안전구역/블루존 관계 정의**(머무는 영역 vs 바깥 피해 영역). **분석 단위 = 팀 명시**(플레이어 기록을 팀 trajectory·노출·관계로 집계, 마지막 팀원 탈락 시 팀 탈락). API 제공 방식 정밀화("developer API를 통해 match별 stream").
- **P3 = 선행과의 차이를 그 자리에서**: Sagong et al.을 서론에서 직접 인용하고 차이를 한 문장으로(플레이어→팀, 파생 점수→전체 telemetry 재구성, 생존 분류→정식화된 위험). "Related Work 절에서 다룬다" 메타 문장 삭제. final-rank 비판은 "목적이 다르다" 톤("designed for"가 아님).
- **P4 = 쉬운 정의 먼저**: hazard를 평문으로 먼저("각 phase 종료까지의 관측으로 다음 phase 동안 탈락할 확률") → 이름 부여 → label 재정의 즉시 연결. "왜 DB인가" 논리 보강(공유 시간·공간 축 위에서 무누수 phase 단위 도출의 체계화). graph 동기 유지. 과적재 한 문장을 3문장으로 분해.
- **RQ 보강**: RQ1에 보존 목적 내장, RQ2에서 "강한" 제거.
- **명시적 기여 3개 + payoff 신설**(기존 "RQ식 통일" 방침 변경): C1 hazard 재정식화(중심축), C2 표현(STDB+feature+graph, 21,824), C3 정보 소재 정량화(+0.112 표현, 관계 추가 이득, ablation·통제 검증). 마지막에 응용 가치(중계 분석·경쟁 피드백·live-service telemetry 생존 분석 일반).
- EN/KO 동기, cites/refs 전수 통과.

## 2026-06-11 후속 작업 15: 서론 2차 피드백 반영 (표현 정밀화)

구조는 유지(피드백: "뼈대는 좋다"), 표현·주장 수위만 수술.

- **Sagong 차이 재구성(#3, 최중요)**: "생존 분류 → 정식화된 위험" 대비를 전면에서 제거 — "라벨을 1−survival로 바꾼 것 아니냐" 공격 지점. 차이의 축을 **분석 단위(플레이어→팀) + 표현(점수→전체 telemetry 재구성) + 해석 목표(분류를 넘어 위험 형성 조건 분석)**로 교체. ⚠ 규칙: hazard 명명 자체를 novelty로 주장하지 말 것.
- **문장 교정(#15)**: 첫 문장 "운명" 제거 → "최종 순위는 정적 결과가 아니라 phase마다 변화하는 생존 위험이 누적된 결과"; "숫자 하나로 평탄화" → "시간 순서가 제거된 사후 통계로 평탄화"; "마지막 팀원이 쓰러질 때" → "마지막 생존 팀원이 제거되는 시점을 팀 탈락 시점으로 정의"(knockdown/elimination 모호성 제거).
- **번역체 절감 규칙(KO, #7)**: framing→문제 설정, learner→학습 알고리즘, feature-only model→"~feature만 사용하는 모델". (EN은 해당 없음)
- **RQ 교정(#8, #9)**: RQ1을 건조·검증가능형으로("…trajectory, 구역 노출, 전투 사건, 동적 팀 간 관계를 feature와 graph로 도출할 수 있는가") — 수사 삽입구 제거. RQ2는 (i)/(ii) 표지 유지(방법론이 참조)하되 "feature-only model" 한국어화. **"누수 없이"는 RQ에서 제거**(방법이 보장할 품질 속성이지 RQ의 내용이 아님 — 서론에서는 P4 동기 서술과 기여 C2에서 한 번씩만, 증명은 방법론 3.2–3.4가 담당).
- **기여 C3 분해(#10)**: 표현 효과와 graph 효과를 별도 문장으로; "선행연구식" → **"선행 연구의 문제 설정을 반영한 strategy-score baseline"**(정확 재현이 아니므로 안전 표현); "동일한 학습 알고리즘에서 AUC 0.112 절대값 향상"으로 정밀화.
- **기타**: "공식 match" → "공식 developer API에서 수집한 match"(#11); payoff "기반이 된다" → "출발점이 될 수 있다"(#13); 3.2에 무누수 보험 문장 추가("row의 모든 수치는 해당 phase 안의 관측만, 종료 시점을 넘지 않음" — 서론의 '누수 없이' 주장 뒷받침, #12).
- 논문 위계(#16) 확인: C1 hazard 재정식화=주인공, STDB=가능케 하는 표현, graph=검증 요소, phase 해석=응용 — 서론 기여 순서가 이미 이 위계.

## 2026-06-11 후속 작업 16: 방법론을 서론 v3에 최종 정렬

- 브리지 2단계와 pipeline 표 RQ1 2행을 RQ1 키워드와 1:1로("trajectory, 구역 노출, 전투, 준비 상태").
- 3.2 첫 문단의 요약 대상을 같은 키워드로 통일("trajectory와 survival condition" → "trajectory, 구역 노출, 전투, 준비 상태").
- 3.4의 \(T_i\)에 서론의 팀 탈락 정의를 연결("마지막 생존 팀원이 제거되는 시점").
- EN/KO 동기, refs/cites 통과. 이로써 서론 P1–P4·RQ·기여 ↔ 방법론 3.1–3.5 ↔ pipeline 그림/표가 같은 어휘로 닫힘.

## 2026-06-11 후속 작업 17: 방법론 엄밀화 대수술 (22항목 심층 리뷰 반영)

리뷰의 우선순위 5개를 전부 처리. 코드 검증 선행: **combat 집계 방향 = A[attacker,victim]이고 Ā·H에서 행이 열을 집계하므로 "공격한 팀이 피격 팀 상태를 집계"** (역방향은 combat-taken node feature가 운반 — 3.3에 명문화); safe_alignment·zone_entry·outside_duration은 **phase k 동안 공개된 그 phase의 target zone** 기준(미래 zone 아님 — 3.2에 명문화).

- **1순위 feature 사전**: Appendix~A 신설(`appendix-features(-ko).tex`, main.tex \appendix 연결) — 66개 전부 정의+scaling 2표. 3.2에 포인터.
- **2순위 누수 서술**: "구성상 누수 안전" 강주장 → "모든 입력은 b_k까지 관측 가능한 telemetry로 구성, 이후는 label 정의에만" (리뷰 제안 채택). 3.2에 centroid 정의식+분할 squad 한계·보완(spread/최근접 적), target zone 공개 시점 명시. 3.5에 masking·causality 정밀 문단(T=8 slot이지만 cut 1–7 출력, LayerNorm은 feature 차원만, padding/탈락 mask). **aux head**: "미래 outcome은 target으로만, 입력 아님, 추론 시 비활성" + hazard-only ablation 약속 — **실험 예약됨**(`run_full_db_aux_ablation.sh`, robustness 체인 종료 후 자동, β=γ=0 full/none).
- **3순위 censoring+loss**: "우승팀은 음성 label만" → right-censoring 정식 서술(관측 interval은 Y=0, 이후는 censored) + **masked 이산시간 생존 우도 수식**(risk set R 정의) 추가.
- **4순위 graph 정밀화**: 3.3에 채널 수식 3개(λ=200m, γ_e=100, γ_k=50, τ_b=150m, τ_d=200m — "사전 고정, 미조정, 민감도는 한계에서"), row-norm 수식, **방향 관례 명문화**, row-norm vs combat 강도(절대 강도는 node feature에 보존, adjacency는 상대 분포), complete graph oversmoothing 인정+감쇠·통제로 상한 제한, N_max=128 padding 설명(3.5).
- **5순위 평가 프로토콜(06)**: random/비층화 split+mode 비율 균형(69.7/68.7/70.9%), "match 내 상관 → 모든 paired test는 row가 아닌 match 재표집", 양성률 0.32·무가중, hyperparameter 무탐색 정직 서술, mode-wise 보고 예고.
- **mode confounder 방어(실측)**: aligned 예측×game_mode 조인 — **full graph가 6개 mode 전부에서 두 기준 초과**(ΔAUC +0.008~+0.020). 07에 한 문장, 보고서 §11에 표.
- **기준 모델 역할 구분(3.5)**: HGB=강한 tabular 검증 / no-edge=architecture·parameter·multi-task 통제 + **"두 기준 모두 이웃 요약 feature(적 밀도·최근접 거리)를 이미 받음 → 이득은 그 요약 너머의 관계 구조에 귀속"** (graph 이득=이웃 통계 재학습이라는 공격 방어).
- 서론 중계 주장에 "phase 경계마다 갱신" 한정어. global α 정직 서술(맥락 의존 중요도 모델링 아님 — 신호 존재 검증 목적). EN/KO 동기, refs/cites/중복 라벨 전수 통과.
- 미해결(한계로): λ·τ 민감도 sweep(엣지 캐시 재구축+재학습 비용), radius-normalized threshold, phase-조건 attention — 08 작성 시 명시.

## 검증 상태

- 정적 검색으로 RQ mapping, figure path, 핵심 citation, 구현 코드명 노출 여부를 확인했다.
- 현재 환경에는 `latexmk`, `pdflatex`, `xelatex`가 없어 PDF 컴파일은 실행하지 못했다.
