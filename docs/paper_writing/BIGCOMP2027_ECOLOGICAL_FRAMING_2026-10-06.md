# BigComp 2027 생태학적 프레이밍 근거 정리

- 작성일: 2026-10-06 · 브랜치 `paper/bigcomp2027` · 성격: 읽기 전용 조사 결과 (원고·코드·결과 파일은 수정하지 않았고 커밋하지 않았다)
- 표기: **[검증]** 직접 연 자료로 확인 / **[내 계산]** 결과 CSV에서 새로 계산 / **[유비]** 개념적 대응이며 인과 주장이 아님 / **미확인** 이번 조사에서 확인하지 못함
- 인용 규칙: 서지는 모두 Crossref API 레코드를 직접 열어 확인했고 일부는 OpenAlex·Semantic Scholar로 교차 확인했다. 초록까지 열어 본 논문은 Fahrig 2003, Nathan 외 2008, Sibly 외 2005, Dunne 외 2002 네 편이다. 나머지는 서지 레코드와 Semantic Scholar 자동 요약(TLDR)만 열었으므로 내용 귀속을 제목 수준으로 제한했다. 전문을 열람한 논문은 없다.
- §6의 한국어 문단은 사용자가 검토·수정할 초안이다. 원고 본문은 영어이므로 반영할 때 번역과 압축이 필요하다.

## 0. 결론

### 0.1 사용자가 말한 "이전 연구"는 무엇인가

1. **폴더에는 원문이 없다.** `연구자료/`의 PDF 77개를 전문 검색했고, PPTX 14개·DOCX 1개의 슬라이드와 발표자 노트를 열었고, 홈 폴더 Spotlight 검색도 했다. 서식지 손실과 멸종을 다룬 생태학 논문 PDF는 없다 (§1.1).
2. **이 관점이 프로젝트에 들어온 문서는 사용자 본인의 2026-04 발표자료다.** `/Users/lunecid/연구자료/게임연구/발표·계획/Graph_PPT.pptx` (제목 "Modeling Team Elimination in PUBG as Shrinking-Habitat Survival: A Graph Neural Network Approach", 파일 수정일 2026-04-07)와 같은 폴더의 `answer.pptx` (2026-04-04). 핵심 문장은 슬라이드 5의 "Zone contraction = habitat loss, Elimination = local extinction"이고, 발표자 노트는 "존의 축소는 서식지의 축소 그리고 탈락은 종의 절멸로 모델링"이라고 적었다.
3. **학술 근거로 실제 인용된 외부 생태학 문헌은 두 편이다.** Fahrig (2003)와 Nathan 외 (2008)이다. 두 논문 모두 PDF는 폴더에 없고, 서지만 `paper/references.bib`, 본인 원고 PDF의 참고문헌, `/Users/lunecid/학위논문/references.bib`에 있다.
4. **판정 (확신도 중간).** 사용자가 기억하는 "서식지 감소 → 멸종" 원리의 학술 근거로 가장 가까운 문헌은 **Fahrig (2003), "Effects of Habitat Fragmentation on Biodiversity", Annu. Rev. Ecol. Evol. Syst. 34(1):487–515** 이다. 핵심 주장(초록)은 "habitat loss has large, consistently negative effects on biodiversity"이다 [검증, Crossref 초록]. 다만 이 논문의 결론은 서식지 손실(양)과 쪼개짐 자체(배치)를 구분하라는 것이며, 개별 멸종의 시간 경과를 모형화한 논문이 아니다. 발표자료 슬라이드 6의 "종간 상호작용 네트워크의 연결도가 멸종 순서를 예측한다"는 연구는 이름이 적혀 있지 않아 식별하지 못했다 (미확인).

### 0.2 가장 강한 매핑 3개 (전체 표는 §4)

1. **안전구역 = 줄어드는 서식지 (서식지 손실).** Fahrig 2003이 뒷받침하는 손실 쪽에 해당한다. 안전구역은 하나의 원이 면적을 잃는 과정이라 쪼개짐이 없다. Hab 블록(45열) 제거 시 AUC가 신경망 −0.0111 [0.0095, 0.0127], HGB −0.0106 [내 계산]이고 둘 다 12/12 코호트에서 같은 방향이다. 우리 모델 구성요소 가운데 SESOI(0.005)를 넘는 효과가 확인된 것은 이 블록뿐이다.
2. **구역 경계 거리 = 가장자리 효과, 그리고 안전구역 객체 노드.** 객체 노드는 수작업 구역·서식지 특징을 뺀 조건에서 +0.0032 [0.0012, 0.0051] (P5, Holm p=0.0023)를 더하고, 그 정보의 κ = 0.29 [0.19, 0.39]를 회복한다. 효과는 신뢰할 만하지만 작다.
3. **팀 탈락 = 지역 멸종이 현재 서식지 상태에 조건부인 이산 hazard.** 같은 KRAFTON 백본·입력·표본에서 Cox 사망 순위를 다음 단계 hazard로 바꾸면 +0.051 [0.044, 0.058]이다. 단계 이력은 추가 정보를 주지 않는다(P1 +0.0007, 동등). 즉 위험은 현재 상태로 대부분 설명된다. 이 +0.051은 정식화 선택의 증거이지 생태 원리의 검증이 아니다.

밀도·국소 경쟁(팀당 면적, 주변 적 팀 수)은 개념적으로 가장 가깝지만 증거가 Hab 블록 단위뿐이라 4순위로 둔다.

### 0.3 고쳐야 할 점과 주의

- **인용과 문장의 불일치 가능성.** 기존 원고(`paper/sections/02-background-related-work.tex` 17–21행, 본인 원고 PDF)는 "서식지 단편화와 movement ecology는 이용 가능한 공간, 이동, 노출, 밀도를 시간–공간 변수로 다룬다"고 쓰고 두 논문을 인용한다. 내가 연 초록은 Fahrig가 손실과 쪼개짐 자체의 구분·효과 크기를, Nathan이 이동의 4요소 틀을 다룬다고 말하며, 노출과 밀도를 결합 변수로 다룬다는 서술은 두 초록에서 확인되지 않는다. 전문은 열람하지 못했으므로 쪽수 근거를 찾거나 문장을 초록이 말하는 내용으로 고쳐야 한다.
- **Hab 효과 수치의 출처.** (i) 요청에 있던 "HGB +0.0095"는 결과 파일에서 찾지 못했다. 0.0095는 신경망 풀링 95% 구간의 하한(0.009548)과 일치한다. HGB는 코호트 평균 +0.0106(12/12)이다 [내 계산]. (ii) Hab 제거 대조는 `contrasts_pooled.csv`에서 kind=secondary, exploratory=True이다. 사전 지정 1차 대조(P1–P5)가 아니므로 "pre-specified"라고 쓰면 안 된다. (iii) "유일한 특징 효과"라는 표현은 한정이 필요하다. 개별 제거를 수행한 특징 계열이 Hab뿐이고, 특징 집합 교체·추가 대조인 (b) +0.014, (g) +0.008도 SESOI를 넘는다. "우리 모델의 구성요소 중 SESOI를 넘는 유일한 효과"로 쓰는 편이 안전하다.
- **수치 정합성.** `paper/bigcomp2027/sections/01-introduction.tex`와 `06-results.tex`는 +0.102 / +0.103 / +0.057을 쓰고, `BIGCOMP2027_PAPER_FOCUS_2026-10-06.md` v5는 +0.091(같은 학습기 HGB), +0.051(Cox → hazard)을 쓴다. 서로 다른 대조로 보이지만 최종 원고에서 어느 쪽을 쓸지 정해 맞춰야 한다.
- **BigComp 원고에는 현재 생태 프레이밍이 없다.** `paper/bigcomp2027/`에서 "Habitat"은 특징 계열 이름일 뿐이고 `references.bib`에 Fahrig·Nathan 항목이 없다 (6월 초안 `paper/references.bib`에만 있다).
- **구역 피해량 변수가 없다.** "구역 피해 = 서식지 손실 사망"에 대응하는 직접 변수는 `hab_*`에 없다. 대리변수는 `hab_outside_safe_ratio`와 `hab_edge_pressure_ratio`뿐이다 (§3.2, §4 행 8).

## 1. 출처 탐색

### 1.1 범위와 방법

| 대상 | 방법 | 결과 |
|---|---|---|
| `연구자료/` PDF 77개 (참고논문, 게임연구, 논문 묶음, 아동상담, 참고자료) | `pdfinfo` 제목과 첫 쪽 상단 4줄로 제목 추출. 전문에서 habitat, extinction, ecolog, species, Fahrig, biodiversity, island biogeograph, metapopulation, 서식지, 멸종, 생태 검색 | 생태학 논문 없음. 히트 12개 파일: 도시공원 현황 보고서(48줄, 무관), Paper_367(Pokémon VGC), Paper_246(게임 AI 즐거움 메타분석), 본인 원고 PDF 7개(Fahrig·Nathan 인용), 본인 발표 PDF 2개. Paper_367·246의 히트 줄은 열어 보지 않았고 제목상 서식지 연구가 아니다 |
| PPTX 14개, DOCX 1개 (`연구자료/` 전체) | 슬라이드·발표자 노트 XML 텍스트 정규식 검색 | `Graph_PPT.pptx`와 사본(각 26줄), `answer.pptx`(10줄)가 생태 프레이밍을 담고 있다. 덱 4개에는 "contracting habitat" 표어 1줄뿐이고 나머지는 0줄. PIBA_PROJECT_PROPOSAL_v2.docx 0줄 |
| 홈 폴더 Spotlight | habitat, extinction, species-area, Fahrig, fragmentation 본문 검색 (쿼리당 상위 8건만 확인) | 외부 생태학 PDF 없음. 이전 프로젝트 사본이 나왔다: `/Users/lunecid/project/arena_survival/dataset.py`, `/Users/lunecid/project/_repo_uploads/PUBG/` |
| 저장소 `docs/`, `paper/`, `network_analysis/`, `legacy/`, `stdb/`, git log, 외부 학위논문 | grep | §1.2 |
| 하지 않은 것 | 텍스트 층이 없는 PDF(`게임 장르.pdf`, `engagement_pair.pdf`, 도로망 궤적 유사도 논문의 1쪽) OCR, zip 2개(`논문 묶음/Paper.zip`, `참고논문/3491101.3516487.zip`) 개봉 | 제목상 생태학과 무관하다고 판단했다 |

임시 파일은 세션 스크래치패드에만 만들었고 저장소에는 이 문서 1개만 추가했다.

### 1.2 프로젝트 안에서 생태 프레이밍이 들어온 경로

| 시점 | 위치 | 내용 |
|---|---|---|
| 2026-04-03 | git `c7df6ec`, `legacy/main.py`, `legacy/dataset.py`, `legacy/model/arena_survival_net.py` | 커밋 메시지 "V2 feature expansion — 14d → 39d agent features (5 ecological axes)". 노드 특징 5축 Physiology / Mobility / Habitat Exposure / Competition / Resource. `species_area_fit`은 S = cA^z를 log(생존 팀 수)–log(구역 면적) 회귀로 검증하며 주석에 "생태학 참고: 대륙 z≈0.15, 해양섬 z≈0.25~0.35"가 있으나 출처가 없다 (미확인). 모델 docstring은 "Shrinking-Habitat Survival Network" |
| 2026-04-04~07 | `연구자료/게임연구/발표·계획/answer.pptx` (슬라이드 9), `Graph_PPT.pptx`·`Graph_PPT 복사본.pptx` (슬라이드 3, 5, 6) | "Players = individuals, Teams = populations, Arena = habitat", "Zone contraction = habitat loss, Elimination = local extinction", "Inspired by movement ecology, edge effect, foraging theory". 슬라이드 6: "species interaction network connectedness predicts extinction order" (발표자 노트: 연구가 있어 참고했다고만 하고 이름 없음). `answer.pptx` 슬라이드 9: Lotka–Volterra 경쟁 계수 α_ij와 인용 [7][8] (덱에 참고문헌 목록 없음, 식별 불가) |
| 시점 미확인 | `network_analysis/habitat_contraction_model_addendum.md` | 생태 개념 ↔ PUBG 대응표(habitat, habitat loss, habitat compression, population density, species richness, extinction, dispersal, edge effect), 종-면적 N_t = c·A_t^z, 압축 압력 ρ_t = N_t/A_t, 팀 단위 hazard. §10은 "분석적 비유로 차용"하고 "생태학 모델로 인과를 증명한다" 같은 표현은 피하라고 적었다. 인용 문헌은 없다 |
| 2026-06-13 | git `2560b1e`, `f80006a`; `docs/feature_engineering/FEATURE_DESIGN_V2.md` §8; `FEATURE_INVENTORY_AND_ABLATION_PLAN.md` §8.1 | F7 habitat-pressure 계열(32열) 구현과 노드 입력 연결. 설계서 §8.1: "biological equivalence가 아니라 analytical analogy", 논문 표현은 `habitat-inspired shrinking-space pressure` 또는 `playable-space competition` 권고 |
| 2026-06 | `paper/sections/02-background-related-work.tex` 17–21행, `paper/references.bib`, `paper/WRITING_NOTES.md` | Fahrig 2003과 Nathan 2008만 인용하고 "analytic analogy, not a biological claim"으로 한정. WRITING_NOTES: 생태학 문헌 → 안전구역이 같은 종류의 수축을 부과 → 분석적 비유로 차용 → 결론의 순서로 교정 |
| 2026 | 본인 원고 PDF 7개 (`논문 묶음/Paper/PUBG_Spatio_Temporal.pdf` 등) | 같은 두 인용과 "생물학적 주장이 아니라 분석적/구조적 비유" 한정. `_2026__PUBG_Spatio_Temporal.pdf`는 키워드에 "shrinking habitat" |
| 2026-07 | `/Users/lunecid/학위논문/references.bib` | 같은 두 항목이 있다. `sections/*.tex`에는 생태 어휘가 없어 (grep) 서지만 남았고 본문 논거는 없다 |
| 2026-10 | `paper/bigcomp2027/` | 생태 프레이밍 없음 (§0.3). 이 문서가 채울 공백 |

### 1.3 "이전 연구" 판정

- **(A) 가장 유력: Fahrig (2003).** 프로젝트가 실제로 인용하는 유일한 서식지 손실 문헌이다. 폴더에 PDF는 없다. 이 논문이 지지하는 것은 "서식지 손실이 생물다양성에 해롭다"이며, 쪼개짐 자체는 약하다는 단서가 붙는다.
- **(B) 프레이밍의 출처 문서: 사용자 본인의 `Graph_PPT.pptx` / `answer.pptx`.** 외부 선행연구가 아니라 사용자가 4월에 세운 유비다. 사용자가 "폴더에 있는 이전 연구"로 기억한다면 이 덱이나 본인 원고 PDF의 참고문헌 목록일 가능성이 높다.
- **(C) 식별 불가: 슬라이드 6의 "연결도가 멸종 순서를 예측" 연구.** 가장 가까운 검증 문헌은 Dunne 외 (2002)이다. 초록은 먹이그물 16개 시뮬레이션에서 연결도(connectance)가 높을수록 종 제거에 강건하고, 연결이 많은 종을 선택적으로 제거하면 임계점 이후 급격히 취약해진다고 말한다 [검증, Crossref 초록]. 사용자가 말한 연구와 같은지는 미확인이다.

## 2. 검증한 생태학 문헌

| 키 | 서지 (Crossref 레코드 기준) | 열람 범위 | 이 문서가 의존한 내용 |
|---|---|---|---|
| fahrig2003fragmentation | Lenore Fahrig. Effects of Habitat Fragmentation on Biodiversity. *Annu. Rev. Ecol. Evol. Syst.* 34(1):487–515, 2003. doi:10.1146/annurev.ecolsys.34.011802.132419 | 서지 + 초록 전문 | 단편화는 보통 서식지 손실과 서식지 쪼개짐을 함께 가리키는 경관 규모 과정이다. 지금까지의 실증 연구는 서식지 손실의 효과가 크고 일관되게 부정적이라고 시사한다. 손실을 통제한 쪼개짐 자체의 효과는 훨씬 약하고 양·음 가능성이 비슷하다. 두 요소를 독립적으로 측정해야 하며 "fragmentation"은 쪼개짐에만 쓰자고 제안한다 |
| nathan2008movement | Ran Nathan, Wayne M. Getz, Eloy Revilla, Marcel Holyoak, Ronen Kadmon, David Saltz, Peter E. Smouse. A movement ecology paradigm for unifying organismal movement research. *PNAS* 105(49):19052–19059, 2008. doi:10.1073/pnas.0800375105 | 서지 + 초록 | 이동 연구의 통합 틀. 내적 상태(왜 움직이나), 운동 능력(어떻게), 항법 능력(언제·어디로), 외부 요인의 상호작용을 기본 구성요소로 둔다 |
| sibly2005regulation | Richard M. Sibly, Daniel Barker, Michael C. Denham, Jim Hone, Mark Pagel. On the Regulation of Populations of Mammals, Birds, Fish, and Insects. *Science* 309(5734):607–610, 2005. doi:10.1126/science.1110760 | 서지 + 초록 | 1780개 시계열에서 개체군 성장률은 저밀도에서 높고 개체군 크기가 커질수록 빠르게 감소한 뒤 평탄해진다 (밀도 의존성) |
| dunne2002connectance | Jennifer A. Dunne, Richard J. Williams, Neo D. Martinez. Network structure and biodiversity loss in food webs: robustness increases with connectance. *Ecology Letters* 5(4):558–567, 2002. doi:10.1046/j.1461-0248.2002.00354.x | 서지 + 초록 | §1.3 (C) 참조 |
| tilman1994extinctiondebt | David Tilman, Robert M. May, Clarence L. Lehman, Martin A. Nowak. Habitat destruction and the extinction debt. *Nature* 371(6492):65–66, 1994. doi:10.1038/371065a0 | 서지 + S2 TLDR (초록 미열람) | TLDR: 패치 서식지의 다종 공존 모형이 현재 서식지 파괴의 미래 생태적 비용(지연되는 멸종)을 예측한다는 요지 |
| kuussaari2009extinctiondebt | Mikko Kuussaari 외 13인. Extinction debt: a challenge for biodiversity conservation. *Trends Ecol. Evol.* 24(10):564–571, 2009. doi:10.1016/j.tree.2009.04.011 | 서지 + S2 TLDR (초록 미열람) | 제목이 말하는 범위. TLDR은 장기 모니터링의 필요성 |
| murcia1995edge | Carolina Murcia. Edge effects in fragmented forests: implications for conservation. *Trends Ecol. Evol.* 10(2):58–62, 1995. doi:10.1016/S0169-5347(00)88977-6 | 서지 + S2 TLDR (초록 미열람) | 파편화된 숲의 가장자리 효과라는 개념의 존재와 보전상의 중요성 |
| hanski1998metapopulation | Ilkka Hanski. Metapopulation dynamics. *Nature* 396(6706):41–49, 1998. doi:10.1038/23876 | 서지 + S2 TLDR (초록 미열람) | 패치 면적과 격리가 이동·정착·국소 개체군 멸절에 미치는 효과가 메타개체군 동태에 통합되었다는 요지 |
| hanski2000capacity | Ilkka Hanski, Otso Ovaskainen. The metapopulation capacity of a fragmented landscape. *Nature* 404(6779):755–758, 2000. doi:10.1038/35008063 | 서지만 (제목) | 제목이 말하는 범위 |
| keymer2000dynamic | Juan E. Keymer, Pablo A. Marquet, Jorge X. Velasco-Hernández, Simon A. Levin. Extinction thresholds and metapopulation persistence in dynamic landscapes. *Am. Nat.* 156(5):478–494, 2000. doi:10.1086/303407 | 서지만 (제목) | 제목이 말하는 범위 |
| connor1979speciesarea | Edward F. Connor, Earl D. McCoy. The statistics and biology of the species-area relationship. *Am. Nat.* 113(6):791–833, 1979. doi:10.1086/283438 | 서지 + S2 TLDR (초록 미열람) | 종-면적 관계의 지수·거듭제곱 모형을 다루는 논문이라는 범위 |
| arrhenius1921species | Olof Arrhenius. Species and area. *J. Ecol.* 9(1):95–, 1921. doi:10.2307/2255763 | 서지만. Crossref·OpenAlex 모두 시작 쪽 95만 기재, 종료 쪽 미확인 | 제목만. 종-면적 관계의 효시라는 귀속은 열람해 확인하지 않았다 |
| volterra1926fluctuations | Vito Volterra. Fluctuations in the abundance of a species considered mathematically. *Nature* 118(2972):558–560, 1926. doi:10.1038/118558a0 | 서지만 | Lotka–Volterra 경쟁 유비를 쓸 때만 필요 (§4 행 13은 사용 비권고) |
| macarthur2001island | Robert H. MacArthur, Edward O. Wilson. The Theory of Island Biogeography. Princeton University Press. doi:10.1515/9781400881376 | 서지만. Crossref 레코드는 2001년 재간행본 | 1967년 원본 연도는 열람한 페이지로 확인하지 못했다. 본문 인용은 권하지 않는다 |

## 3. 프로젝트의 운영화

### 3.1 Hab 블록의 구성

150차원 입력 중 Zone(15)에서 생존 수 2개를 뺀 13개와 Habitat(32)을 합친 45개를 Hab 블록이라 부른다 (`paper/bigcomp2027/sections/03-problem-data.tex` 표 `tab:features`). Habitat 32개 = 전역 수축 13 + 구역 진입 12 + 국소 경쟁 7이고, 구현은 `stdb/habitat_feature_etl.py`의 `stdb.team_phase_habitat_features`이다. 정의는 설계서 `docs/feature_engineering/FEATURE_DESIGN_V2.md` §8.3–8.5와 ETL docstring을 따랐고, BUILD_SQL 본문의 계산식은 대조하지 않았다.

### 3.2 `hab_*` 32열의 정의와 생태 구성개념

| 군 | 열 | 정의 | 대응 구성개념 [유비] |
|---|---|---|---|
| 전역 수축 (13) | `hab_safe_area_{start,end,delta}_m2` | 현재 파란 구역 면적 πr²의 단계 시작·끝·차이 | 서식지 면적과 손실량 |
| | `hab_loss_ratio` | 1 − A_end / A_start | 서식지 손실 비율 |
| | `hab_shrink_rate_m2ps` | 면적 감소율 (정확한 분모는 미대조) | 손실 속도 |
| | `hab_alive_teams` | 단계 끝 생존 팀 수 (압박에 들어서는 개체군) | 개체군 크기 |
| | `hab_safe_area_per_alive_team` | A_end / 생존 팀 수 | 개체군당 면적 |
| | `hab_crowding_pressure` | 생존 팀 수 / A_end | 개체군 밀도 |
| | `hab_center_shift_m`, `hab_center_shift_per_radius` | 구역 중심 이동량, 시작 반경으로 정규화 | 서식지 이동 |
| | `hab_zone_gap_{start,end}_m`, `hab_zone_gap_pressure` | 흰 원(다음 구역) 반경 − 현재 구역 반경(수축 중에는 보통 음수). 끝 간격을 목표 반경으로 정규화(0 = 목표에 닿음) | 서식지 이동·수렴 잔여량 |
| 구역 진입 (12) | `hab_dist_to_safe_boundary_{start,end}_m`, `hab_signed_boundary_distance_{start,end}_m` | 경계까지 거리. 부호 있는 거리는 안쪽 +(완충), 바깥 −(노출) | 가장자리 효과 |
| | `hab_outside_safe_ratio`, `hab_edge_pressure_ratio` | 구역 밖 체류 비율. 구역 밖이거나 경계 안쪽 150 m 이내인 스냅샷 비율 | 노출, 가장자리 압력 |
| | `hab_distance_to_target_safe_m` | 팀 중심에서 컷 시점의 흰 원까지 거리 (안이면 0) | 이동 필요 거리 |
| | `hab_required_speed_to_safe_mps` | 위 거리 ÷ 해당 단계 길이 (다음 단계 길이는 미래 정보라 현재 단계 길이로 대리) | 이동 요구 (분산 비용 유비) |
| | `hab_observed_speed_mps`, `hab_migration_deficit_mps` | 관측 평균 속력, 필요 속력 − 관측 속력 | 이동 부족분 |
| | `hab_rotation_delay_ratio` | 컷 시점 흰 원으로 첫 진입한 시각의 정규화값 (진입 못 하면 1.0, 이미 안이면 0.0) | 이동 지연 |
| | `hab_safe_alignment_to_target` | 목표 구역 방향 정렬도 (정의 식은 설계서에 없음, 미확인) | 항법 방향성 |
| 국소 경쟁 (7) | `hab_local_enemy_count_{100,200,500}m`, `hab_local_enemy_density_200m` | 팀 마지막 스냅샷 기준 반경 내 적 팀 수와 밀도 | 국소 밀도 |
| | `hab_safe_entry_enemy_count_200m`, `hab_safe_entry_congestion` | 목표 구역 진입 시점 주변 적 팀 수와 혼잡 | 피난처 접근 경쟁 |
| | `hab_refuge_scarcity_simple` | 진입 시 적 수 / (파란 구역 면적 + 1 m²) | 피난처 희소성 |

구역 피해량(blue-zone damage)을 직접 담은 열은 없다. `outside_safe_ratio`는 `stdb/phase_spatial_feature_etl.py`에서 오고, Exposure 계열(13)의 damage가 구역 피해를 구분하는지는 확인하지 못했다.

### 3.3 안전구역 객체, co-pressure 엣지, 위상 조건부 이산 hazard

- **안전구역 객체 노드.** 원시 기하만 담는다: `cur_cx, cur_cy, cur_r, next_cx, next_cy, next_r, phase_frac, phase_no` (`model_stgcn/ptr/data.py` `ZONE_COLUMNS`). 경계 거리 같은 가공 변수는 일부러 뺐고, RQ4가 "모델이 이를 스스로 복원하는가"를 묻는다. 팀↔구역 어텐션 편향은 팀 위치의 현재·다음 구역 대비 정규화 위치 (pos − centre)/radius와 정규화 중심거리 d/r (1 = 경계)을 받는 작은 MLP이다 (`model_stgcn/ptr/model.py` docstring). 회복률은 κ = [AUC(M11RZ-noHab) − AUC(M11R-noHab)] / [AUC(M11R) − AUC(M11R-noHab)]이다 (`04-method.tex`).
- **co-pressure 엣지.** 가중치 (c_ij / n_ij)·exp(−d̄co_ij / 200), 대칭. c_ij / n_ij는 두 팀이 서로 200 m 이내이면서 구역 밖이거나 경계 안쪽 150 m 이내인 공동 관측의 비율이다 (`03-problem-data.tex`).
- **위상 조건부 이산 hazard.** 단계 k=1..7은 안전구역 일정이 정하는 이산 시계이고, 위험집합은 단계 k 끝에 살아 있는 팀이다. 팀 탈락은 마지막 구성원의 사망으로 정의하고 기절한 구성원은 생존으로 센다. 라벨은 Y_i(k+1) = 1{팀 i가 단계 k+1에 탈락}, 우승팀은 검열이다. 목표는 λ_i(k) = Pr(Y_i(k+1) = 1 조건 F(k))이고 손실은 위험집합 행의 BCE + 0.1·순위 손실이다 (`03-problem-data.tex`, `04-method.tex`).
- **누수 규칙.** 모든 특징·중심·구역 기하·엣지는 정렬된 시계에서 단계 경계 10초 이전까지의 표본만 쓴다. 허용: 컷 시점에 이미 보이는 현재 구역과 발표된 목표 구역, 컷까지의 위치·속력. 금지: 미래 구역 중심, 미래 적 위치, 미래 교전, 최종 순위 (설계서 §8.6, `stdb/causal_cut.py`).

## 4. 매핑 표

지위 열의 **[유비]**는 개념적 대응이며 인과 주장이 아니다. 증거 열의 수치는 §5 대장에서 출처를 확인할 수 있다. 증거는 대부분 Hab 블록 단위이며, 하위 군(면적, 밀도, 경계, 이동, 국소 경쟁)을 따로 제거한 실험은 없다.

| # | 생태학 원리 | 검증된 인용 | 배틀로얄 대응 | 우리 설계 요소 | 우리 결과의 증거 | 지위 |
|---|---|---|---|---|---|---|
| 1 | 서식지 손실(면적 감소)은 생물다양성에 크고 일관되게 해롭고, 서식지가 쪼개지는 배치 변화 자체의 효과는 훨씬 약하다 | Fahrig 2003 (초록 열람) | 안전구역 = 줄어드는 서식지. 하나의 원이 면적을 잃는 과정이라 쪼개짐 없이 손실만 있다 | `hab_safe_area_*`, `hab_loss_ratio`, `hab_shrink_rate_m2ps`, Zone 계열(구역 이동, 경계 거리, 구역 밖 체류). 단계 = 구역 일정이 정하는 이산 시계 | Hab 블록(45) 제거: 신경망 −0.0111 [0.0095, 0.0127], HGB −0.0106 [내 계산], 둘 다 12/12 코호트, SESOI 초과. 블록 단위이고 2차·탐색적 대조 | [유비] + 예측 정보로 지지 |
| 2 | 종-면적 관계: 종 수 S는 면적 A의 거듭제곱 S = cA^z로 변한다 | Connor & McCoy 1979 (서지 + TLDR), Arrhenius 1921 (제목만) | 생존 팀 수 N_t = c·A_t^z (addendum §3) | legacy `species_area_fit`. BigComp 입력에는 `hab_alive_teams`, `hab_safe_area_per_alive_team`만 있고 z는 추정하지 않는다 | 없음. BigComp 결과에 z 적합이 없고 legacy 산출물은 확인하지 않았다 | [유비], 미검정 |
| 3 | 밀도 의존성: 개체군 크기가 커질수록 1인당 성장률이 떨어진다 | Sibly 외 2005 (초록 열람) | 구역 안 팀 밀도와 팀당 남은 면적 (개체군당 면적). Sibly는 성장률이고 우리는 다음 단계 탈락 확률이라 직접 대응은 아니다 | `hab_alive_teams`, `hab_safe_area_per_alive_team`, `hab_crowding_pressure` | Hab 블록에 포함되어 블록 단위로만 지지. 쌍별 상호작용 구조 없이도 정보가 전달되는 정황: 관계 bias와 셔플 관계가 동등 (P4 +0.0005 [−0.0001, 0.0010]) | [유비], 블록 단위 |
| 4 | 밀집이 경쟁을 키운다 (밀도 의존 경쟁, 피난처 희소성) | 생태학 인용 미검증 (Sibly는 개체군 수준 밀도 의존성만 다룬다) | 강제 수렴: 같은 경계로 몰린 팀 사이의 조우 증가, 피난처 접근 경쟁 | `hab_local_enemy_count_*`, `hab_local_enemy_density_200m`, `hab_safe_entry_*`, `hab_refuge_scarcity_simple`, co-pressure 엣지 | 국소 경쟁 7열도 Hab 블록 안(분해 없음). co-pressure는 가장 작은 기여(06-results: 유형 제거 시 −0.0003, 최대 두 코호트, 탐색적). 학습된 유효 편향 +0.026±0.033 대 combat +0.46~+0.61, 어텐션 lift copress(+prox) 0.89 대 combat 2.20 (기술적 분석). 모델은 쌍별 교전에 쏠리고 밀집 정보는 특징에 담긴다 | [유비]. 쌍별 co-pressure 구조는 지지되지 않음 |
| 5 | 가장자리 효과: 서식지 경계 근처에서 환경 조건과 개체군 변동이 달라진다 | Murcia 1995 (서지 + TLDR) | 구역 경계까지 거리, 구역 밖 노출. 파란 구역 경계는 피해를 주는 경계라 숲 가장자리의 미기후 변화와 기전이 다르다 | `hab_dist_to_safe_boundary_*`, `hab_signed_boundary_distance_*`, `hab_edge_pressure_ratio`, `hab_outside_safe_ratio`. 안전구역 객체 노드와 팀↔구역 어텐션 편향 | 객체 노드 P5 +0.0032 [0.0012, 0.0051] (1차 대조, Holm p=0.0023, 10/12 코호트 같은 부호), κ = 0.286 [0.189, 0.388]. 학습된 팀↔구역 편향이 d/r 0→1→2에서 +0.05→+0.14→+0.23으로 12/12 코호트 단조 증가. 구역 토큰 어텐션 lift는 구역 밖 팀이 더 큼(2.05 대 1.37, p=0.002). 후자 둘은 seed 42의 기술적 분석이며 설명도 인과도 아님 | [유비], 작지만 신뢰할 만한 효과 |
| 6 | 동적 경관: 서식지 패치가 파괴·이동하는 경관에서는 멸종 임계값과 지속성이 달라진다 | Keymer 외 2000 (제목만, 내용 미열람) | 다음 구역(흰 원)이 현재 구역과 다른 위치·크기로 정해져 팀이 재배치를 강요받는다 (구역 간격, 중심 이동) | `hab_center_shift_*`, `hab_zone_gap_*`, 객체 노드의 next center/radius | Hab 블록 안(분해 없음) | [유비]. 인용을 쓰려면 초록 확인 필요 |
| 7 | 이동은 내적 상태, 운동 능력, 항법 능력, 외부 요인의 상호작용으로 결정된다 | Nathan 외 2008 (초록 열람) | 목표 구역까지 필요한 속도 대 실제 속도 (이동 요구, "분산 비용"은 우리 해석), 회전 지연. 내적 상태 = 팀 상태, 운동 = Movement 계열, 항법 = 구역 방향, 외부 요인 = 구역·적 팀 | `hab_distance_to_target_safe_m`, `hab_required_speed_to_safe_mps`, `hab_observed_speed_mps`, `hab_migration_deficit_mps`, `hab_rotation_delay_ratio`, `hab_safe_alignment_to_target`; Movement(7), Team state(9) | Hab 블록 안(분해 없음). Movement·Team state 계열의 개별 제거는 수행하지 않았다 | [유비] |
| 8 | 서식지 손실이 개체를 직접 죽인다 (서식지 손실 사망) | 해당 문헌 미검증 | 구역 밖 체류가 만드는 구역 피해 | 직접 변수 없음. 대리변수 `hab_outside_safe_ratio`, `hab_edge_pressure_ratio`. Exposure 계열 damage가 구역 피해를 구분하는지는 미확인 | 없음 | 현재 설계로는 주장하기 어렵다 |
| 9 | 국소 멸종: 메타개체군에서 패치 면적과 격리가 정착·국소 멸절을 좌우한다 | Hanski 1998 (서지 + TLDR), Hanski & Ovaskainen 2000 (제목만) | 팀 탈락 = 지역 멸종. 탈락은 재정착(rescue)이 없는 흡수 상태라 정착–멸절 균형과 다르다 | 라벨 Y_i(k+1): 마지막 구성원 사망 이벤트가 단계 k+1에 있으면 1, 기절은 생존, 우승팀 검열 | 해당 없음 (정의의 문제). 라벨 감사: 옛 스냅샷 라벨은 훈련 행의 11.4%가 오류, 양성 정밀도 70.4% | [유비] |
| 10 | 멸종 부채: 서식지 파괴의 효과가 시차를 두고 멸종으로 나타난다 | Tilman 외 1994, Kuussaari 외 2009 (둘 다 서지 + TLDR) | 이미 구역 밖에 몰렸거나 회전이 늦은 팀의 위험이 현재 상태뿐 아니라 지난 단계 이력에 의존하는가 (RQ1). 시간 척도가 세대 대 분이라 형식적 유비다 | 단계 축 causal attention (단계 ≤ k 이력), HGB-lag (이전 단계 값·차분 추가) | P1 +0.0007 [−0.0013, 0.0027] (Holm p=0.24, 7/12 코호트 양, 동등). HGB-lag는 HGB-cut보다 평균 0.0010 낮음 [내 계산, 10/12 코호트에서 낮음]. 단계 해상도에서 시차 신호가 없고 현재 상태로 충분하다. 단계 축 주효과는 데이터가 늘면 커진다 (+0.0025/ln N, 탐색적 메타회귀) | [유비], 결과는 null |
| 11 | 서식지 상태가 위험률을 정한다 (상태 의존 위험) | 생태학 인용 없음. 정식화는 이산 시간 생존분석에서 왔다 (Allison 1982, `paper/references.bib`, 재검증하지 않음) | 안전구역이 단계별로 계단식 수축 → 단계 = 이산 시계, 단계 끝 상태에 조건부인 다음 단계 hazard | λ_i(k) = Pr(Y_i(k+1) = 1 조건 F(k)); 위험집합 = 단계 k 끝 생존 팀, k=1..7; BCE + 0.1·순위 손실 | 같은 KRAFTON 백본·입력·표본에서 Cox 사망 순위 → 다음 단계 hazard: +0.051 [0.044, 0.058] (단계 내 +0.027 [0.020, 0.034]) | 방법론 선택의 증거. 생태 원리의 검증은 아님 |
| 12 | 개체 → 개체군 계층 (발표자료: players = individuals, teams = populations) | 없음 | 팀 = 개체군, 플레이어 = 개체 | 팀 토큰 단위 모델(기본), 플레이어 층 변형 M11RZ-P (stage 2) | M11RZ-P − M11RZ +0.0014 [−0.0009, 0.0037] 동등. 팀원 bias 제거 −0.0001 [−0.0004, 0.0001] 동등 (탐색적) | [유비] |
| 13 | 종간 경쟁 계수 α_ij (Lotka–Volterra), `answer.pptx` 슬라이드 9 | Volterra 1926 (서지만). 슬라이드의 [7][8]은 식별 불가 | 팀 간 관계 행렬이 α_ij에 해당 | 관계 편향 어텐션 b_{r,h}·W_r[i,j] | 지지되지 않음: 명시적 관계가 셔플 관계와 동등 (P4). 학습된 편향 크기는 combat에 치우침 | 본문에서는 쓰지 않거나 "지지되지 않았다"로 한정 권고 |
| 14 | (맥락) 위험은 개체 속성이 아니라 서식지 맥락에 달려 있다 | 해당 없음 | Sagong 입력(개인 활동 지표 위주, 타 선수는 100 m 이내 인원 수 하나) 대 우리 입력(구역·서식지·관계 맥락 150열) | 단계 조건부 표현 150-d | 같은 학습기(HGB)에서 Sagong 입력 → 우리 입력 +0.091 [0.085, 0.097] (단계 내 +0.119 [0.114, 0.125]). 많은 차이를 묶은 값이라 생태 원리의 검증이 아니다. Hab 제거 효과(0.011)는 이 격차의 약 1/8이지만 중복 때문에 총정보를 과소평가할 수 있다. Sagong 입력에 Hab만 더하는 실험은 하지 않았다 | 표현의 중요성 증거. 생태 프레임과는 간접 연결 |
| 15 | 종간 상호작용 네트워크의 연결도가 멸종 순서를 예측한다 (발표자료 슬라이드 6, 인용 없음) | 식별 못함. 가장 가까운 검증 문헌은 Dunne 외 2002이나 동일 여부 미확인 | degree가 높은 팀이 먼저 탈락하는가 | 4월 계획의 네트워크 가설. BigComp에서는 degree 분석을 하지 않았다 | 해당 없음 | BigComp 범위 밖 |

## 5. 증거 대장

경로의 기준은 `/Users/lunecid/project/PUBG/`이다.

| 주장 | 값 | 출처 | 검증 메모 |
|---|---|---|---|
| Hab 블록 제거, 신경망 (M11R − M11R-noHab) | +0.0111 [0.0095, 0.0127] 풀링. 코호트 평균 +0.0113, 범위 +0.0066~+0.0172, 12/12 양 | `reports/bigcomp2027_results/kappa.csv` (분모), `contrasts_pooled.csv` 행 `hab_value`, `paper/bigcomp2027/sections/06-results.tex` | 풀링 값은 직접 확인. 코호트별 값은 `arm_metrics_per_cohort.csv` (seed 42, auc, stratum all)에서 [내 계산]. `hab_value`는 kind=secondary, exploratory=True |
| Hab 블록 제거, HGB (HGB-cut − HGB-noHab) | 코호트 평균 +0.0106, 범위 +0.0031~+0.0163, 12/12 양 | `arm_metrics_per_cohort.csv` | [내 계산]. 풀링(DerSimonian–Laird) 추정은 결과 파일에 없다. 요청의 "+0.0095 (HGB)"는 확인하지 못했다 |
| 구역 객체 효과 P5와 회복률 κ | P5 +0.0032 [0.0012, 0.0051] (Holm p=0.0023, 10/12 같은 부호). κ = 0.286 [0.189, 0.388] | `kappa.csv`, `contrasts_pooled.csv` (zone_nohab), `06-results.tex` | 직접 확인 |
| 표현 vs Sagong, 같은 학습기(HGB) | +0.091 [0.085, 0.097], 단계 내 +0.119 [0.114, 0.125] | `docs/paper_writing/BIGCOMP2027_PAPER_FOCUS_2026-10-06.md` v5 (a) | 문서 수치만 확인, CSV 재계산 안 함. `01-introduction.tex`·`06-results.tex`는 구 수치 +0.102/+0.103 |
| 정식화: Cox 사망 순위 → 다음 단계 hazard | +0.051 [0.044, 0.058], 단계 내 +0.027 [0.020, 0.034] | PAPER_FOCUS v5 (c) | 위와 같음 |
| 단계 이력 P1 | +0.0007 [−0.0013, 0.0027], Holm p=0.24, 7/12, 동등 | `06-results.tex` | |
| 명시적 관계 P4 | +0.0005 [−0.0001, 0.0010], Holm p=0.08, 동등 | `06-results.tex` | |
| 팀 간 attention P2 / P3 | +0.0033 AUC [0.0024, 0.0042] (10/12) / +0.0094 Spearman [0.0045, 0.0143] (9/12) | `06-results.tex` | 신뢰 가능하나 SESOI 미만 |
| SESOI | 0.005 AUC, 0.01 Spearman | `06-results.tex` | |
| HGB-lag | HGB-cut − HGB-lag 평균 +0.0010, 10/12 양 | `arm_metrics_per_cohort.csv` | [내 계산] |
| 학습된 구역 편향, 관계 편향, 어텐션 lift | §4 행 4·5 | `reports/bigcomp2027_results/interpretability/README.md` | 탐색적·기술적 분석, 인과 주장 아님. 구역 객체는 seed 42, 12 코호트는 독립이 아님 |
| 과거 OM-ladder habitat | +0.0103, 13/13 (2026-07-04) | `docs/experiment_design/BATTLE_ROYALE_OM_LADDER_RESULT_FEATURE_AUDIT_2026-07-04.md` | 라벨 감사(2026-10-05) 이전 결과라 인용 불가. 방향이 같다는 참고용 |

## 6. 프레이밍 초안 (서론·문제 정의용, 한국어)

### 문단 1 (서론: 유비의 도입)

배틀로얄에서는 안전구역이 단계마다 줄어들어 남은 팀이 쓸 수 있는 공간이 단조롭게 감소한다. 이 과정은 생태학이 서식지 손실이라 부르는 현상과 구조가 닮았다. Fahrig [fahrig2003fragmentation]는 서식지 단편화 문헌을 검토하여, 서식지 손실은 생물다양성에 크고 일관되게 부정적인 효과를 내는 반면 서식지가 쪼개지는 배치 변화 자체의 효과는 훨씬 약하고 방향도 일정하지 않다고 정리하였다. 안전구역은 하나의 원이 면적을 잃어 가는 과정이어서 쪼개짐 없이 손실만 있으므로 이 구분에서 손실에 해당한다. 면적이 줄면 유지되는 종 수가 줄어든다는 종-면적 관계 [connor1979speciesarea]와, 서식지 파괴의 생태적 비용이 시차를 두고 나타난다는 멸종 부채 [tilman1994extinctiondebt] 논의도 같은 방향을 가리킨다. 우리는 이 대응을 생물학적 주장이 아닌 분석적 유비로만 사용한다. 안전구역을 서식지에, 팀을 개체군에, 팀 탈락을 지역 멸종에 대응시키고, 남은 공간이 줄어들수록 다음 단계 탈락 위험이 달라진다는 가설을 예측 정보의 관점에서 검토한다.

### 문단 2 (문제 정의: 설계로의 번역)

이 유비는 세 가지 설계 요소로 옮겨진다. 첫째, 위험은 단계 k 경계 직전까지 관측된 서식지 상태에 조건부로 정의한다. 단계는 안전구역 일정이 정하는 이산 시계이고, 단계 k의 위험집합에 속한 팀 i의 위험률은 λ_i(k) = Pr(Y_i(k+1) = 1 조건 F(k))이다. 서식지 상태는 구역 면적과 손실률, 중심 이동, 팀당 남은 면적, 경계까지의 부호 있는 거리, 목표 구역까지 필요한 이동 속도, 주변 적 팀 수 등으로 구성한 Zone·Habitat 특징 45개(Hab 블록)로 나타내며, 각각 서식지 손실, 개체군당 면적, 가장자리 효과 [murcia1995edge], 이동 요구 [nathan2008movement], 국소 경쟁에 대응한다. 둘째, 안전구역을 특징으로만 쓰지 않고 현재·다음 구역의 중심과 반경, 경과 비율, 단계 번호만 담은 객체 노드로 모델에 넣어, 모델이 경계 거리 같은 가공 변수를 스스로 복원하는지 묻는다(RQ4). 셋째, 두 팀이 구역 경계 부근에 동시에 몰리는 정도를 co-pressure 관계로 표현해 밀집에 따른 경쟁에 대응시킨다. 이 대응은 모두 개념적 유비이며 생물학적 동치나 인과를 뜻하지 않는다.

### 문단 3 (증거의 범위와 유비의 한계)

증거는 예측 정보 수준에 머문다. 12개 코호트에서 Hab 블록을 제거하면 신경망의 AUC가 0.0111 [0.0095, 0.0127] 낮아지고 이 방향은 12개 코호트 모두에서 같다. 구조 대조 P1–P5의 효과가 모두 사전 지정한 최소 유의 효과(SESOI, 0.005 AUC)보다 작았던 것과 대비된다. 안전구역 객체 노드는 이 정보의 κ = 0.29 [0.19, 0.39]만 회복하므로 서식지 압박은 설계한 특징으로 여전히 명시해야 한다. 단계 이력(P1, +0.0007)과 명시적 관계(P4, +0.0005)는 동등으로 판정되어, 위험이 현재 서식지 상태로 대부분 설명되고 시차 효과나 쌍별 상호작용 구조는 이 해상도에서 추가 정보를 주지 않음을 시사한다. 같은 학습기에서 우리 표현은 Sagong 입력보다 AUC가 0.091 높고, 같은 백본에서 목적을 Cox 사망 순위에서 다음 단계 hazard로 바꾸면 0.051 높아지지만, 이는 표현과 정식화의 효과이지 생태학 원리의 검증이 아니다. 유비의 한계도 분명하다. 배틀로얄의 수축은 외부에서 결정적으로 부과되고, 팀은 전략적으로 행동하며, 탈락은 재정착이 없는 흡수 상태이고, 시간 척도는 분 단위이다.

### 반영 시 주의

- 위 초안의 수치는 모두 §5 대장에서 확인했다. HGB 행(+0.0106, 12/12)을 표에 넣고 싶다면 `arm_metrics_per_cohort.csv`에서 풀링 추정을 새로 계산해야 한다. 지금 값은 내 계산이며 논문 표에는 없다.
- BigComp 본문은 참고문헌 포함 8쪽이다. 세 문단을 모두 넣기는 어렵다. 권고: 문단 1을 4문장으로 줄여 서론에 넣고(Fahrig, Tilman), 문단 2의 요지는 표현 절에 한 문장으로, 문단 3의 마지막 두 문장은 한계 절에 둔다.
- 초안은 "Hab 제거가 유일한 효과"라고 쓰지 않았고 "P1–P5가 모두 SESOI보다 작았다"로 한정했다. Hab 제거 대조가 2차·탐색적이라는 점(§0.3)은 원고에서도 밝혀야 한다.

## 7. 추가할 인용

8쪽 제한을 고려하면 생태 인용은 4–5편 이내를 권한다. 아래 서지는 모두 §2에서 Crossref로 확인한 값이다. 새 항목의 제목은 Crossref 표기를 그대로 두었고(`&`는 BibTeX용으로 `\&`), 기존 프로젝트 항목은 제목 대문자 표기가 다르다.

| 우선순위 | 키 | 용도 | 검증 수준 |
|---|---|---|---|
| 필수 | `fahrig2003fragmentation` | 손실 대 쪼개짐 구분, 안전구역은 손실 쪽 | 초록 열람. `paper/references.bib`에는 있고 `paper/bigcomp2027/references.bib`에는 없다 |
| 필수 | `nathan2008movement` | 이동 요구 특징(필요 속도 등)의 틀. 이 특징을 정당화할 때만 | 초록 열람. 위와 같은 상황 |
| 권장 | `tilman1994extinctiondebt` | RQ1(단계 이력) 해석: 시차 효과 가설 | 서지 + TLDR |
| 권장 | `murcia1995edge` | 경계 거리 특징의 가장자리 효과 유비 | 서지 + TLDR |
| 권장 | `connor1979speciesarea` | 종-면적 관계를 언급할 때만 | 서지 + TLDR |
| 권장 | `sibly2005regulation` | 밀도 의존성을 언급할 때만 | 초록 열람 |
| 선택 | `kuussaari2009extinctiondebt`, `hanski1998metapopulation`, `hanski2000capacity`, `keymer2000dynamic`, `dunne2002connectance`, `arrhenius1921species`, `volterra1926fluctuations`, `macarthur2001island` | 지면이 허락할 때 | §2 참조. 본문 인용 전 초록 확인 필요 |

```bibtex
% 기존 항목: paper/references.bib와 동일. bigcomp2027/references.bib에 복사 필요
@article{fahrig2003fragmentation,
  author = {Lenore Fahrig},
  title = {{Effects of Habitat Fragmentation on Biodiversity}},
  journal = {Annual Review of Ecology, Evolution, and Systematics},
  volume = {34},
  pages = {487--515},
  year = {2003},
  doi = {10.1146/annurev.ecolsys.34.011802.132419}
}

@article{nathan2008movement,
  author = {Ran Nathan and Wayne M. Getz and Eloy Revilla and Marcel Holyoak and Ronen Kadmon and David Saltz and Peter E. Smouse},
  title = {{A Movement Ecology Paradigm for Unifying Organismal Movement Research}},
  journal = {Proceedings of the National Academy of Sciences},
  volume = {105},
  number = {49},
  pages = {19052--19059},
  year = {2008},
  doi = {10.1073/pnas.0800375105}
}

% 새 항목: Crossref 레코드 기준
@article{tilman1994extinctiondebt,
  author = {David Tilman and Robert M. May and Clarence L. Lehman and Martin A. Nowak},
  title = {{Habitat destruction and the extinction debt}},
  journal = {Nature},
  volume = {371},
  number = {6492},
  pages = {65--66},
  year = {1994},
  doi = {10.1038/371065a0}
}

@article{murcia1995edge,
  author = {Carolina Murcia},
  title = {{Edge effects in fragmented forests: implications for conservation}},
  journal = {Trends in Ecology \& Evolution},
  volume = {10},
  number = {2},
  pages = {58--62},
  year = {1995},
  doi = {10.1016/S0169-5347(00)88977-6}
}

@article{connor1979speciesarea,
  author = {Edward F. Connor and Earl D. McCoy},
  title = {{The Statistics and Biology of the Species-Area Relationship}},
  journal = {The American Naturalist},
  volume = {113},
  number = {6},
  pages = {791--833},
  year = {1979},
  doi = {10.1086/283438}
}

@article{sibly2005regulation,
  author = {Richard M. Sibly and Daniel Barker and Michael C. Denham and Jim Hone and Mark Pagel},
  title = {{On the Regulation of Populations of Mammals, Birds, Fish, and Insects}},
  journal = {Science},
  volume = {309},
  number = {5734},
  pages = {607--610},
  year = {2005},
  doi = {10.1126/science.1110760}
}

% 선택 항목
@article{kuussaari2009extinctiondebt,
  author = {Mikko Kuussaari and Riccardo Bommarco and Risto K. Heikkinen and Aveliina Helm and Jochen Krauss and Regina Lindborg and Erik {\"O}ckinger and Meelis P{\"a}rtel and Joan Pino and Ferran Rod{\`a} and Constant{\'\i} Stefanescu and Tiit Teder and Martin Zobel and Ingolf Steffan-Dewenter},
  title = {{Extinction debt: a challenge for biodiversity conservation}},
  journal = {Trends in Ecology \& Evolution},
  volume = {24},
  number = {10},
  pages = {564--571},
  year = {2009},
  doi = {10.1016/j.tree.2009.04.011}
}

@article{hanski1998metapopulation,
  author = {Ilkka Hanski},
  title = {{Metapopulation dynamics}},
  journal = {Nature},
  volume = {396},
  number = {6706},
  pages = {41--49},
  year = {1998},
  doi = {10.1038/23876}
}

@article{hanski2000capacity,
  author = {Ilkka Hanski and Otso Ovaskainen},
  title = {{The metapopulation capacity of a fragmented landscape}},
  journal = {Nature},
  volume = {404},
  number = {6779},
  pages = {755--758},
  year = {2000},
  doi = {10.1038/35008063}
}

@article{keymer2000dynamic,
  author = {Juan E. Keymer and Pablo A. Marquet and Jorge X. Velasco-Hern{\'a}ndez and Simon A. Levin},
  title = {{Extinction Thresholds and Metapopulation Persistence in Dynamic Landscapes}},
  journal = {The American Naturalist},
  volume = {156},
  number = {5},
  pages = {478--494},
  year = {2000},
  doi = {10.1086/303407}
}

@article{dunne2002connectance,
  author = {Jennifer A. Dunne and Richard J. Williams and Neo D. Martinez},
  title = {{Network structure and biodiversity loss in food webs: robustness increases with connectance}},
  journal = {Ecology Letters},
  volume = {5},
  number = {4},
  pages = {558--567},
  year = {2002},
  doi = {10.1046/j.1461-0248.2002.00354.x}
}

% 쪽수 미확인: Crossref와 OpenAlex 모두 시작 쪽 95만 기재
@article{arrhenius1921species,
  author = {Olof Arrhenius},
  title = {{Species and Area}},
  journal = {Journal of Ecology},
  volume = {9},
  number = {1},
  pages = {95},
  year = {1921},
  doi = {10.2307/2255763}
}

% Crossref 제목 끝의 각주 숫자 "1"은 뺐다
@article{volterra1926fluctuations,
  author = {Vito Volterra},
  title = {{Fluctuations in the Abundance of a Species considered Mathematically}},
  journal = {Nature},
  volume = {118},
  number = {2972},
  pages = {558--560},
  year = {1926},
  doi = {10.1038/118558a0}
}

% Crossref 레코드는 2001년 재간행본. 1967년 원본 연도는 미확인
@book{macarthur2001island,
  author = {Robert H. MacArthur and Edward O. Wilson},
  title = {{The Theory of Island Biogeography}},
  publisher = {Princeton University Press},
  year = {2001},
  doi = {10.1515/9781400881376}
}
```

## 8. 미확인 항목과 열린 질문

미확인 (이번 조사에서 확인하지 못한 것):

1. 발표자료 슬라이드 6의 "연결도가 멸종 순서를 예측" 연구. 이름이 없다. Dunne 외 2002는 가장 가까운 검증 후보일 뿐이다.
2. `answer.pptx` 슬라이드 9의 인용 [7][8]. 덱에 참고문헌 목록이 없다.
3. legacy 코드 주석의 종-면적 지수 참고값(대륙 z≈0.15, 해양섬 z≈0.25~0.35)의 출처. 확인 전에는 인용하지 말 것.
4. 초록을 열지 못한 문헌의 내용 귀속: Tilman, Kuussaari, Hanski 두 편, Murcia, Connor & McCoy, Keymer, Arrhenius, Volterra. 서지는 확인했다.
5. Arrhenius 1921의 종료 쪽, MacArthur & Wilson 원본의 출판 연도.
6. Fahrig·Nathan 전문. 기존 원고 문장(§0.3)의 근거 쪽수를 확인하지 못했다.
7. 발표자료가 언급한 "foraging theory"와 "refuge scarcity"의 문헌. 검증된 인용이 없다.
8. 텍스트 층이 없는 PDF의 OCR, zip 2개의 내용.
9. `hab_shrink_rate_m2ps`, `hab_safe_alignment_to_target`의 정확한 계산식, Exposure 계열 damage의 구역 피해 포함 여부.
10. Paper_367·Paper_246의 생태 어휘 히트 줄. 제목상 무관하다고 판단했다.

사용자 결정이 필요한 질문과 선택적 후속 분석 (이번 조사에서는 수행하지 않았다):

- 증거가 대부분 블록 단위다. 행 3–7, 9를 각각 지지하려면 Hab 하위 군(면적, 밀도, 경계, 이동, 국소 경쟁)을 따로 제거한 ablation이 필요하다.
- 종-면적 관계(행 2)를 쓰려면 매치·단계 단위로 log N_t = log c + z·log A_t를 적합하는 별도 분석이 필요하다 (`legacy/dataset.py::species_area_fit` 재사용 가능).
- Sagong 입력에 Hab 블록만 더하는 실험이 있으면 행 14의 "맥락 표현" 주장을 생태 프레임과 직접 연결할 수 있다.
- Lotka–Volterra 유비(행 13)와 네트워크 연결도 가설(행 15)은 BigComp에서 지지되지 않거나 다루지 않으므로 본문에서 빼는 편이 안전하다. 최종 판단은 사용자 몫이다.

## 부록: 주요 근거 파일

- `/Users/lunecid/연구자료/게임연구/발표·계획/Graph_PPT.pptx`, `answer.pptx`
- `/Users/lunecid/연구자료/논문 묶음/Paper/PUBG_Spatio_Temporal.pdf` (본인 원고, 참고문헌 [11][12])
- `/Users/lunecid/project/PUBG/network_analysis/habitat_contraction_model_addendum.md`
- `/Users/lunecid/project/PUBG/docs/feature_engineering/FEATURE_DESIGN_V2.md` (§8), `FEATURE_INVENTORY_AND_ABLATION_PLAN.md` (§8.1)
- `/Users/lunecid/project/PUBG/stdb/habitat_feature_etl.py`
- `/Users/lunecid/project/PUBG/paper/bigcomp2027/sections/01-introduction.tex`, `03-problem-data.tex`, `04-method.tex`, `06-results.tex`
- `/Users/lunecid/project/PUBG/model_stgcn/ptr/model.py`, `model_stgcn/ptr/data.py`
- `/Users/lunecid/project/PUBG/reports/bigcomp2027_results/` (`kappa.csv`, `contrasts_pooled.csv`, `arm_metrics_per_cohort.csv`, `interpretability/README.md`)
- `/Users/lunecid/project/PUBG/docs/paper_writing/BIGCOMP2027_PAPER_FOCUS_2026-10-06.md`
- `/Users/lunecid/project/PUBG/paper/references.bib`, `paper/sections/02-background-related-work.tex`, `paper/WRITING_NOTES.md`
