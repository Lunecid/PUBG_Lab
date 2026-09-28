# docs

[한국어](#한국어) · [English](#english)

## 한국어

이 폴더의 논문 원고와 발표 자료를 날짜순으로 정리했습니다. 설명은 파일 자체에 적힌 내용(표지, 첫 슬라이드, 파일 메타데이터)만 옮겼고, 메타데이터 날짜는 UTC 기준입니다.

- 파일마다 데이터, 모형, 예측 대상이 다릅니다. 수치를 서로 비교하거나 합치지 마세요.
- 원고와 발표 자료에 나오는 모형과 스키마 가운데 DynamicSTGNN, R-GCN, PostGIS 시공간 DB 스키마 등은 이 저장소 코드에 없습니다. 저장소의 코드(`main.py`, `model/`)는 2026년 4월에 만든 플레이어 그래프 파이프라인입니다.
- 백성은(Seongeun Baek)의 석사 학위논문은 이와 다른 주제(리그 오브 레전드 교전의 전략적 가치)이며, 이 저장소에 없습니다.

| 날짜 (근거) | 파일 | 파일에 적힌 내용 |
|---|---|---|
| 2026-05-04 (마지막 저장일, 파일 메타데이터) | [talks/PUBG_Network_Analysis.pptx](talks/PUBG_Network_Analysis.pptx) | 16장 발표 자료. 첫 슬라이드 제목은 "PUBG Squad Combat Networks: Phase-Dynamic Strategy Types and Performance"이고, 소속은 Pusan National University, Data Science입니다. 슬라이드에 날짜나 행사 이름은 없습니다. |
| 2026-05-25 (영상 파일 생성 시각) | 발표 영상. 2026-09-28에 저장소에서 삭제했고, 다른 곳에 올리면 여기에 링크를 달 예정입니다. | 약 3분 30초 길이의 발표 영상(1920×1080). 첫 화면에 "REFINED PROPOSAL · BUILDING ON THE PRIOR PLAN"과 제목 "A Heterogeneous Spatio-Temporal GNN for PUBG Team Elimination under a Shrinking-Habitat Framework"가 있습니다. `PUBG_STDB_STGNN.pptx`를 녹화한 영상이 아닙니다. 제목이 다르고, 영상 속 슬라이드는 6장입니다(pptx는 12장). |
| 2026-05-26 (첫 슬라이드, 파일 메타데이터) | [talks/PUBG_STDB_STGNN.pptx](talks/PUBG_STDB_STGNN.pptx) | 12장 발표 자료. 첫 슬라이드에 "Final Presentation · 2026-05-26"과 제목 "PUBG Telemetry의 시공간 데이터베이스 모델링 및 생존/등수 예측 모델 개발"이 있고, 소속은 부산대학교 데이터사이언스전문대학원입니다. |
| 2026-06-11 (PDF 생성일, 꼬리말 "Draft, June 2026") | [papers/PUBG_Spatio_Temporal.pdf](papers/PUBG_Spatio_Temporal.pdf) | 4쪽 한국어 논문 원고(ACM 서식). 제목은 "PUBG 텔레메트리의 시공간 데이터베이스 모델링과 페이즈 단위 팀 탈락 예측"이고 저자는 2명입니다. 꼬리말은 "Draft"이고, 메타데이터의 학회 이름 칸은 서식 기본값 "Proceedings of Draft manuscript (Draft)"입니다. 게재처, DOI, 저작권 표시는 없습니다. |
| 2026-06-21 (파일 메타데이터, 첫 슬라이드에는 "2026") | [talks/PUBG_Graph_Survival_Deck.pptx](talks/PUBG_Graph_Survival_Deck.pptx) | 14장 발표 자료. 슬라이드가 모두 이미지라 글자를 선택할 수 없습니다. 첫 슬라이드에 "GRAPH DATA ANALYSIS · COURSE PROJECT"와 제목 "Graph Modeling for PUBG Survival & Elimination Prediction"이 있습니다. 수업 프로젝트 발표 자료입니다. |
| 2026-08-07 (PDF 생성일, 표지 날짜는 비어 있음) | 91쪽 PDF. 2026-09-28에 저장소에서 삭제했습니다. | 91쪽 A4 문서. 표지에 "석사학위논문"과 제목 "배틀로얄 장르의 단계 조건부 생존 분석 프레임워크"가 있지만, 학과·작성자·지도교수·날짜 칸은 서식 자리표시자("학과명", "작성자명", "지도교수명", "2026년 월 일") 그대로이고 심사위원 인준란도 비어 있습니다. 인준된 학위논문이 아닙니다. PDF 메타데이터에는 제목과 저자가 없습니다. |

짧은 논문 원고의 그림 1과 `PUBG_Graph_Survival_Deck.pptx`의 2번 슬라이드에 쓴 에란겔 지도 이미지는 PUBG 지도 아트워크입니다(© KRAFTON, Inc.).

---

## English

The papers and talks in this folder, in date order. Each description repeats only what the file itself states (title page, first slide, file metadata). Metadata dates are in UTC.

- The files use different data, models and prediction targets. Do not compare or combine their numbers.
- Several models and schemas described in these files, such as DynamicSTGNN, R-GCN and the PostGIS STDB schema, are not in this repository's code. The code here (`main.py`, `model/`) is the player-graph pipeline built in April 2026.
- Seongeun Baek's M.S. thesis is a separate project (the strategic value of League of Legends engagements) and is not in this repository.

| Date (source) | File | What the file states |
|---|---|---|
| 2026-05-04 (last saved, file metadata) | [talks/PUBG_Network_Analysis.pptx](talks/PUBG_Network_Analysis.pptx) | 16-slide deck. Slide 1 title: "PUBG Squad Combat Networks: Phase-Dynamic Strategy Types and Performance", Pusan National University, Data Science. The slides give no date or event. |
| 2026-05-25 (video creation time) | Talk video. Removed from the repository on 2026-09-28; it will be linked here once it is hosted elsewhere. | Talk video, about 3.5 minutes, 1920×1080. The title frame reads "REFINED PROPOSAL · BUILDING ON THE PRIOR PLAN" and "A Heterogeneous Spatio-Temporal GNN for PUBG Team Elimination under a Shrinking-Habitat Framework". It is not a recording of `PUBG_STDB_STGNN.pptx`: the title differs and the video shows 6 slides (the deck has 12). |
| 2026-05-26 (slide 1, file metadata) | [talks/PUBG_STDB_STGNN.pptx](talks/PUBG_STDB_STGNN.pptx) | 12-slide deck. Slide 1 reads "Final Presentation · 2026-05-26", with the title "Spatio-Temporal Database Modeling and Survival / Rank Prediction for PUBG Telemetry" (in Korean and English), Graduate School of Data Science, Pusan National University. |
| 2026-06-11 (PDF creation date; footer "Draft, June 2026") | [papers/PUBG_Spatio_Temporal.pdf](papers/PUBG_Spatio_Temporal.pdf) | 4-page Korean manuscript in the ACM template, titled "PUBG 텔레메트리의 시공간 데이터베이스 모델링과 페이즈 단위 팀 탈락 예측" (spatio-temporal database modeling of PUBG telemetry and phase-level team elimination prediction), with two authors. The footer says "Draft", and the metadata venue field is the template default "Proceedings of Draft manuscript (Draft)". It names no venue, DOI or copyright. |
| 2026-06-21 (file metadata; slide 1 says "2026") | [talks/PUBG_Graph_Survival_Deck.pptx](talks/PUBG_Graph_Survival_Deck.pptx) | 14-slide deck made of slide images, so it has no selectable text. Slide 1 reads "GRAPH DATA ANALYSIS · COURSE PROJECT" and "Graph Modeling for PUBG Survival & Elimination Prediction". It is a course project deck. |
| 2026-08-07 (PDF creation date; the title-page date is blank) | 91-page PDF. Removed from the repository on 2026-09-28. | 91-page A4 document. The title page reads "석사학위논문" (master's thesis) above the title "A Phase-Conditioned Survival Analysis Framework for the Battle Royale Genre", but the department, author, advisor and date fields are still template placeholders ("학과명", "작성자명", "지도교수명", "2026년 월 일") and the committee approval page is blank. It is not an approved thesis. The PDF metadata has no title or author. |

The Erangel map image in Figure 1 of the short manuscript and on slide 2 of `PUBG_Graph_Survival_Deck.pptx` is PUBG map artwork (© KRAFTON, Inc.).
