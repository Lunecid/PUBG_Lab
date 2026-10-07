# 그림 1(파이프라인) 매핑 — BigComp 2027

작성: 2026-10-06. 그림 자체는 수정하지 않고, 나중에 원고에 넣을 때 필요한 연결 정보만 정리한다.

버전은 둘이다. 원고에는 **간결판 `fig_pipeline_clean.*`을 우선 후보**로 쓴다. 사용자가 준 MLOps 그림 예시의 스타일을 따른 판이다. 간결판 매핑은 7절에 있다. 1–6절은 상세판 `fig_pipeline.*` 기준이며, 상세판은 보존용이다.

## 1. 파일과 빌드

| 항목 | 값 |
|---|---|
| 소스 | `paper/bigcomp2027/figures/fig_pipeline.tex` (standalone TikZ, 310줄) |
| 결과물 | `fig_pipeline.pdf` (벡터, 원고용), `fig_pipeline.png` (300 dpi 미리보기) |
| 크기 | 17.80 × 9.44 cm, 최소 글자 7 pt |
| 빌드 | `cd paper/bigcomp2027/figures && /opt/homebrew/bin/tectonic fig_pipeline.tex && pdftoppm -png -r 300 -singlefile fig_pipeline.pdf fig_pipeline` |
| 언어 | 그림 라벨은 영어(최종 원고 기준). 한국어 초안과의 용어 대응은 4절 |

## 2. 원고 삽입

- 위치: 두 단 전체 폭 `figure*`, 2쪽 상단(`[t]`). IEEE 규칙상 본문에서 먼저 참조해야 한다.
- 크기: 글자 크기 7 pt를 유지하려면 원래 크기로 넣는다. `width=\textwidth`(약 18.1 cm)로 넣어도 약 2%만 커진다.

```latex
\begin{figure*}[t]
\centering
\includegraphics{figures/fig_pipeline.pdf}
\caption{...}  % 3절 초안
\label{fig:pipeline}
\end{figure*}
```

- 본문 참조 위치(제안):
  - 서론 P5, "…파이프라인을 구축한다" 뒤에 `(그림~\ref{fig:pipeline})`. 첫 참조다.
  - III장 도입 또는 III-B 첫 문장에 `(그림~\ref{fig:pipeline} 위)`.
  - IV장 도입에 `(그림~\ref{fig:pipeline} 아래)`.

## 3. 캡션 초안 (그림 밖에 둔다)

- KO: 그림 1. 데이터 파이프라인(위, III장)과 모델링·평가 절차(아래, IV장). 공개 API에서 수집한 텔레메트리를 하나의 시간축에 놓고 사건 기반 단계 경계 $b_k$를 정한다. 모든 입력은 특징 절단 시각 $a_k=b_k-10$초 이전의 기록으로, 라벨은 처치 사건으로 구성한다. 각 설계는 자기 학습 방식을 유지하되 같은 경기 분할과 같은 시험 행에서 평가하며, 코호트별 차이를 확률효과 모형으로 통합해 세 연구 질문에 답한다. 점선은 라벨의 흐름이다.
- EN: Fig. 1. Data pipeline (top, Sec. III) and modeling and evaluation (bottom, Sec. IV). Telemetry is placed on one clock and split at event-based phase boundaries $b_k$. Every input uses only records before the feature cut $a_k=b_k-10$ s, and labels come from kill events. Each design keeps its own training scheme but is scored on the same match split and identical test rows. Per-cohort differences are pooled with a random-effects model to answer RQ1–RQ3. Dashed arrows carry labels only.

## 4. 노드 ↔ 원고 매핑

그림 소스의 노드 ID(`(N..)`)와 원고 `\label`을 기준으로 정리했다. "용어 대응"은 한국어 초안에서 같은 개념을 가리키는 표현이다.

| 노드 | 그림 라벨(요약) | 원고 위치 | 용어 대응 / 일치시킬 내용 | 근거 |
|---|---|---|---|---|
| N0 START | PUBG public API (Steam) | `sec:data:telemetry` P1 | 공개 API, 스팀 서버 | `\cite{PUBGAPI}` |
| N1 | Collect matches & telemetry | `sec:data:telemetry` P2 | 수집 경기(사용분만 서술) | collector `01_fetch_match_ids.py` |
| N2 | Raw telemetry: positions ~10 s · events | `sec:data:telemetry` P1–P2, `tab:corpus` | 위치 표본 약 10초 간격, 사건 기록 | 표 1 건수(DB COUNT) |
| N3 | Filter & form 12 cohorts | `sec:data:benchmark` P5 | 튜토리얼·훈련장·airoyale·Vikendi 제외, 12개 코호트 | `population.json` |
| N4 | Align clocks; phase boundaries from phase-change events | `sec:data:samples` P3 첫째·둘째 | 공통 시간축, 단계 전환 사건 | v2 `CONTRACT.md` §0 |
| N5 | Prediction time $b_k$; feature cut $a_k=b_k-10$ s (+시간축 삽도) | `sec:data:samples` P3 셋째, `sec:method:formulation` P1 | 예측 시점, 특징 절단 시각, $\Delta=10$초, 라벨 구간 | `CONTRACT.md` §0 |
| N6a | Inputs before $a_k$: features, alive status, relations, masks | `sec:data:samples` P3, `sec:method:formulation` P1, `sec:method:representation` P4 | 입력 특징, 생존 인원, 팀 간 관계, attention 마스크; 접두 재생 시험(204경기, 1,293개 경계, 불일치 0) | `prefix_replay.json` |
| N6b | Event-based labels: risk set, $Y$, knocked ≠ eliminated, winners censored, final phase | `sec:data:samples` P4, `sec:method:formulation` P1, `eq:hazard` | 위험집합, 기절과 최종 처치 구분, 우승 팀 검열, $b_{k+1}=C_m$ | `CONTRACT.md` §1 |
| N7 | Team–phase samples: 13,948 matches · one row per (match, team, $k$) | `sec:data:benchmark` P6 | 분석 단위 (경기, 팀, 단계 경계), 817,766행 | `run_summary.json` |
| N8 | Match-level split 80/10/10, fixed, per cohort | `sec:data:benchmark` P6 | 경기 단위 8:1:1, 고정 분할 | split 파일 |
| N9a | Habitat representation: 4 constructs $z$ · 150-d $X$ · team graph $G$ | `sec:method:representation` P3–P4 | 구역 경계 노출, 이동 부담(이동 부족분), 주변 경쟁 밀도, 남은 팀원 수; 150차원; 근접·교전·공동 압박 관계 | `meta["feature_names"]` (F=150) |
| N9b | Sagong inputs (12) | `sec:method:rq2`, `tab:adapt` Sagong 열 | 활동·교전 점수, 플레이 유형 | `sagong2025/fit.py:57-61` |
| N9c | KRAFTON inputs (55 + positions) | `sec:method:rq2`, `tab:adapt` KRAFTON 열 | 행동, 장비, 주변 수 | `adaptation/schema.py:103` |
| N10 (RQ1) | Ecological hazard model: logit $\lambda=\alpha_k+\beta^\top z$ | `sec:method:rq1`, `eq:ehm` | 생태 hazard 모델 | `ptr/ecomodel.py` |
| N11 | Per-cohort $\beta$, clustered SE | `sec:method:rq1` | 코호트별 계수, 경기 단위 군집 표준오차 | 동일 |
| N12 (RQ2) | Train compared designs: Ours (HGB · PTR-Former) vs Adapted (Sagong XGB/RF/LR · KRAFTON Transformer, Cox) | `sec:method:rq2`, `tab:adapt` | 비교 설계, 적용판 | `ptr/arms.py`, adaptation |
| N13 (RQ3) | Single-factor ladder: (a)(b)(g) input swaps; KR-Cox (+H0) → KR-Hazard → $+\alpha_k$ → PTR-Team → PTR-Former, (c)–(f) no rank loss; + rank auxiliary loss (separate) | `sec:method:decomposition`, `tab:ladder` | 단일 요인 분해, 순위 보조 손실 | `aggregate.DECOMP_STEPS` (c1/c2 분리) |
| N14 | Predictions $\hat p$, identical test rows | `sec:method:rq2` 범위 문장 | 같은 경기 분할, 같은 위험집합·라벨에서 평가 | `pred_test.csv` |
| N15 | Metrics + paired bootstrap ×2,000 | `sec:method:evaluation` P9–P10 | 전체 AUC, 단계 안 AUC, 경기–단계 내부 AUC, 로그 손실, Brier, 보정 | `aggregate.py`, `supplement.py` |
| N16 | Random-effects pooling: 12 cohorts · DL (REML+HK check) · gain/equivalent/harm/inconclusive | `sec:method:evaluation` P10 | DerSimonian–Laird, REML+HK 민감도, 판정 범주 | 동일 |
| N17 END | Answers: RQ1 · RQ2 · RQ3 | 결과 장(미작성), 결론 | — | — |

간선 의미:

- 실선은 데이터 흐름이다.
- 점선은 라벨 흐름이다: N6b→N7 "targets $Y$", N8→N15 "$Y$ (test)".
- "inputs per design"은 N9a/b/c에서 N12/N13으로 가는 입력이고, "same match split"은 N8에서 N9a/b/c로 가는 흐름이다.
- 그림 가장자리의 주석 "each design keeps its own training sampling; evaluated on identical test rows"는 IV-D의 범위 문장과 대응한다.

## 5. 나중에 그림을 고칠 때 반영할 것 (지금은 보류)

| # | 노드 | 현재 라벨 | 문제 | 수정안 |
|---|---|---|---|---|
| 1 | N4 | "Align clocks: timestamp ↔ elapsed time" | v2는 경과 시간 필드를 쓰지 않는다. 모든 기록의 타임스탬프를 경기 시작 기준으로 바꿔 시계 하나만 쓴다(III-B 첫째) | "One clock: event timestamps (time since match start)" |
| 2 | N16 | "gain / equivalent / harm / inconclusive" | 한국어 초안은 개선/실질적 동등/저하/불확정이다. 영어 원고와 단어를 맞춰야 한다 | 영어 원고를 그림 단어에 맞추는 쪽을 권장(더 짧음) |
| 3 | N12 | "KRAFTON (Transformer, Cox)" | RQ2에는 KRAFTON+H0(검증 집합에서 추정한 기저위험)도 들어간다 | 선택: "KRAFTON (Transformer, Cox; +H0)" |
| 4 | N15 | 지표 목록 | Spearman과 상위 5위 AUC가 빠져 있다 | 선택: 공간이 남으면 "rank metrics" 추가 |
| 5 | 공통 | 숫자 라벨 | 재실행 뒤 다시 확인한다: 12 cohorts, 13,948 matches, 150-d, Sagong 12, KRAFTON 55, 80/10/10, ×2,000 | v2 계약상 F=150은 그대로이고, 나머지도 바뀔 이유가 없다 |
| 6 | 결과 | — | 그림에는 결과 수치가 없어야 한다(서론·방법의 인과 순서 규칙) | 유지 |

## 6. 이중 맹검·표기 점검

- 저자, 소속, 로고는 없다. 아이콘은 일반 기호(낙하산, 구역 원, 헬멧 등)다.
- 색은 Okabe–Ito 계열이다. 선 종류(실선/점선)와 범례로도 구분되므로 흑백 인쇄에서도 의미가 남는다.
- 그림 안에 캡션 문장을 넣지 않는다. 캡션은 3절 초안을 원고 쪽에 둔다.

## 7. 간결판 `fig_pipeline_clean.*` 매핑 (우선 후보)

- 파일: `paper/bigcomp2027/figures/fig_pipeline_clean.{tex,pdf,png}`
- 크기: 17.80 × 8.97 cm, 최소 글자 7 pt
- 빌드: `tectonic fig_pipeline_clean.tex`, 이어서 `pdftoppm -png -r 300 -singlefile fig_pipeline_clean.pdf fig_pipeline_clean`
- 삽입: 2절의 LaTeX 코드에서 파일명만 `figures/fig_pipeline_clean.pdf`로 바꾼다. 캡션은 3절 초안을 그대로 쓴다.

| 패널/상자 | 그림 라벨(요약) | 원고 위치 |
|---|---|---|
| A1 START | PUBG public API | `sec:data:telemetry` P1 |
| A2 | Raw telemetry: positions ~10 s, timestamped events | `sec:data:telemetry` P1–P2, `tab:corpus` |
| A 띠 | Steam · 13,948 matches · 322 M records | `sec:data:telemetry` P2, `tab:corpus` |
| B1 | One event clock; phase boundaries from phase-change events | `sec:data:samples` P3 첫째·둘째 |
| B2 | Prediction time at each phase end (next zone revealed) | `sec:data:samples` P3 셋째, `sec:method:formulation` P1 |
| B3a | Inputs cut 10 s earlier: features · alive status · relations · masks | `sec:data:samples` P3, `sec:method:formulation` P1 |
| B3b | Event-based labels: teams alive at prediction; 1 = eliminated in next phase | `sec:data:samples` P4, `eq:hazard` |
| B4 | Team–phase samples, one row per match · team · phase | `sec:data:samples` P4 끝 문장(단위), `sec:data:benchmark` P6(표본 수) |
| B 띠 | 시간축: inputs / cut / 10 s / phase end / label window / next phase end | `sec:data:samples` P3–P4 |
| 복귀선 | 12 cohorts · match-level split 8:1:1 → C1–C3 | `sec:data:benchmark` P5(코호트)→P6(분할) |
| C1 (ours) | Habitat representation: edge exposure, movement burden, competition, team members | `sec:method:representation` P3–P4 |
| C2 | Sagong inputs | `sec:method:rq2`, `tab:adapt` |
| C3 | KRAFTON inputs | `sec:method:rq2`, `tab:adapt` |
| M1 (RQ1, ours) | Ecological hazard model: phase baseline + 4 habitat constructs | `sec:method:rq1`, `eq:ehm` |
| M2 (RQ2) | Hazard objective (HGB · PTR-Former) vs Adapted (Sagong · KRAFTON) | `sec:method:rq2`, `tab:adapt` |
| M3 (RQ3) | Single-factor ladder: input · objective · phase baseline · structure | `sec:method:decomposition`, `tab:ladder` |
| C 띠 | same split · own training scheme · adapted designs also scored on their own task | `sec:method:rq2` 범위 문장, P7b(충실도 점검) |
| D1 | Predicted risk, identical test rows | `sec:method:rq2` 범위 문장 |
| D2 | Metrics: AUC · within-phase AUC, log loss · Brier, per-phase calibration | `sec:method:evaluation` P9 |
| D3 | Random-effects pooling, 12 cohorts | `sec:method:evaluation` P10 |
| END | Answers RQ1 · RQ2 · RQ3 | 결과 장, 결론 |
| D 띠 | paired match bootstrap ×2,000 · DerSimonian–Laird | `sec:method:evaluation` P10 |

간선 라벨: C1→M1은 라벨 없음, M1→D3는 "coefficients", B4→D2(점선)는 "test labels". 그림 안에는 "×2,000" 말고 기호가 없다(2026-10-06 사용자 지시). 상자 순서와 내용은 본문 순서에 맞췄다.

간결판 기준으로 5절의 보류 항목을 다시 보면 다음과 같다.

- 1번(시계 표기)과 2번(판정 범주 단어)은 간결판에서 해소되었다. B1에 "on one event clock"이 있고, 판정 범주는 그림에서 빠졌다.
- 3번(KRAFTON+H0)과 4번(순위 지표)은 여전히 선택 사항이다.
- 간결판에서 생략한 세부는 본문과 캡션이 맡는다: 마지막 단계의 $b_{k+1}=C_m$, 경기–단계 내부 AUC, REML+HK 민감도, 순위 보조 손실.

### 7.1 간결판 캡션 (그림 안 수식 제거판, 2026-10-06 사용자 지시)

그림 안에는 수식과 기호를 두지 않는다. 기호와 그림 속 말의 대응은 캡션에서 한 번만 밝힌다.

- KO: 그림 1. 데이터 파이프라인(위, III장)과 모델링·평가 절차(아래, IV장). 예측 시점은 각 단계의 끝($b_k$, 다음 안전구역이 공개되는 시각)이다. 모든 입력은 그보다 10초 앞선 절단 시각($a_k=b_k-10$초) 이전의 기록으로 만든다. 라벨은 그 팀이 다음 단계 안에 탈락하면 1이다($Y=\mathbb{1}[b_k<T\le b_{k+1}]$). 각 설계는 자기 학습 방식을 유지하되 같은 경기 분할과 같은 시험 행에서 평가한다. 코호트별 차이는 확률효과 모형으로 통합한다. 실선은 데이터, 점선은 라벨의 흐름이다.
- EN: Fig. 1. Data pipeline (top, Sec. III) and modeling and evaluation (bottom, Sec. IV). The prediction time is the end of each phase ($b_k$, when the next safe zone is revealed). Every input uses only records before the cut 10 s earlier ($a_k=b_k-10$ s). The label is 1 if the team is eliminated within the next phase ($Y=\mathbb{1}[b_k<T\le b_{k+1}]$). Each design keeps its own training scheme but is scored on the same match split and identical test rows. Per-cohort differences are pooled with a random-effects model. Solid arrows carry data; dashed arrows carry labels.
