# 첨부 SAC MLA 리뷰에 대한 선택적 반영 기록 — v15

대상: `ACM_SAC2027_MLA_Review_and_Revision_KO.md` (2026-10-08).
수정 기준: 패치 재학습과 추가 대조가 완료된 v14. 첨부 리뷰는 v12/Widom의 이전 수치·무작위 분할·20쪽 읽기용 조판을 대상으로 하므로, 그 전제를 현재 원고에 그대로 적용하지 않았다. v14의 원고·결과와 v12/v13 원본은 보존했다.

이번 작업은 현재 증거로 가능한 원고 수정과 기존 결과의 재표시다. 새 모델을 학습하거나 위험집합을 바꾸지 않았고, 미수행 실험을 완료했다고 쓰지 않았다. 검증된 결과의 추가 여부와 별개로, 아래의 미해결 항목은 문장 수정만으로 해결되지 않는다.

| 리뷰 항목 | 반영 상태 | v15 변경과 근거 | 남은 범위 |
|---|---|---|---|
| P0-1: 온라인 위험집합 | 부분 반영 | §3.4–3.5에서 입력 cutoff a와 자격 boundary b를 분리. 817,766개 포함 행의 2개 예외 검사는 전체 후보 포함/제외 감사가 아님을 명시. 미래 return이 최종 탈락 시각에 영향을 줄 수 있음을 명시. 같은 시각 사건, 최종 구간, 로그 단절 제외 규칙을 코드와 대조해 추가. | b까지의 모든 후보 팀 상태를 재구성한 전수 교차표, redeploy 이벤트의 수집 완전성 감사, 불일치 원인별 재평가는 미수행. 온라인 위험집합 검증을 완료했다고 주장하지 않음. |
| P0-2: 동일 목표 기준 | 기존 반영 + 보강 | v14의 공통 153입력/BCE/16설정/3시드 비교와 horizon 대조 유지. 기존 phase-only·lite의 절대 AUC/NLL/Brier를 §6.1에 추가. KRAFTON의 전역 Platt 변환식과 전체 weighted 목적함수의 공통 score 이동 불변성을 설명. | full-feature regularized logistic 및 standalone MLP는 학습하지 않음. lite는 이를 대신하는 baseline이 아니며, 부재를 한계에 기재. |
| P0-3: 시간순 평가 | v14에서 완료 + 해석 보강 | 40.2→41.1→41.2의 공통 manifest 유지. 41.1 공식 패치 노트를 확인해 봇 및 Jammer Pack 변화 명시. 동일 zone schedule을 환경 불변성의 근거로 쓰지 않음. | 새로운 미공개 시험 자료, 여러 후속 패치, 새 플레이어 일반화는 검증하지 않음. 무작위 분할 원고의 대체 문단을 그대로 복사하지 않음. |
| P1-1: 이동·경쟁 변수 | 반영 | 예정된 전체 단계 길이로 정규화한 관측 중심점 경로임을 설명. 미래 이동 부족량으로 해석하지 않음. 관측 창, nearest-position ±5초 및 t+5<a, 3차원 중심점 거리 <200m, 팀 중복 집계 배제를 명시. 기존 AI/반경/구성요소 민감도 유지. | 새 물리적 의미의 변수를 만들거나 기존 결과에 새 식을 붙이지 않음. 팀 중심점 변화는 개인의 실제 이동과 동일하지 않음. |
| P1-2: 구조 효과 | 반영 | 지표별 범위를 유지하고 같은 단계의 다른 경기 비교와 같은 경기·단계 비교의 차이를 예시로 설명. attention 내부 기제에 대한 해석을 제한. | 모든 코호트의 다중 시드·동일 계산량 튜닝은 미수행. |
| P1-3: 통계·기여 분해 | 반영 | 5개 주요 구조 비교의 추정치·90% CI·SESOI·Holm p·판정을 Table 7에 집약. Holm은 0 대비 단측 우월성 검정에 적용되고 동등성 구간은 미보정임을 명시. gain이 효과>SESOI의 통계적 입증이 아님을 명시. 임계값 민감도 구간을 기호로 상호 배타적으로 정의하고 원시 숫자를 CSV에 보존. 확인되지 않은 사전등록/dated protocol 표현을 축소. | 현재 선택된 문턱을 사후 변경하지 않음. 표본 bootstrap을 학습 시드 모집단의 불확실성으로 해석하지 않음. |
| P1-4: 활용·재현성 | 부분 반영 | 활용을 사후 spectator-view 경기 검토로 한정. 공개 benchmark 및 운영 효과·실시간성 주장을 하지 않음. 원고 소스 패키지와 실험 재현 자료를 구분하고 파일 해시·출처 기록 추가. | top-k 포착률, calibration slope/intercept, 동일 장비의 지연·메모리, 사용자 효용 연구, 공개 익명 학습 자료는 미완료. |
| 제목·초록·기여 | 반영 | target alignment·team context·model structure 중심 제목. 입력 결합 AUC 0.786, 지표별 PTR 차이, 동일 시행 횟수 튜닝의 무일관 승자, 원 과제 결과를 초록에 명시. | 리뷰의 과거 수치 0.778/0.793 및 12개 주 분석 코호트는 사용하지 않음. |
| 결과 순서 | 부분 반영 | 초록·주 비교 첫 문장에서 입력 결합과 강한 비교를 강조. RQ와 그림의 대응을 보존하기 위해 본문 RQ1→RQ2→RQ3 순서는 유지. | 근거 추가 없이 원고 전체를 새 실험 논문으로 재구성하지 않음. |
| SAC 조판·인용 | 유지·검증 | 기존 익명 acmart 형식, 첫 인용 등장 순 번호, 39개 참고문헌 유지. 이번 PDF도 참고문헌 포함 10쪽. | 업로드·비용 지불·저자 또는 공개자료 계정 설정은 수행하지 않음. |

## 이번에 새로 표시한 수치의 출처

- `tables/primary_decisions_values.csv`: `reports/sac2027_patch_v14/primary_results_v2/contrasts_pooled.csv`의 `kind=primary` 5행. 추론을 다시 실행하거나 다른 구간으로 바꾸지 않았다.
- phase-only/lite 절대 성능: `results_v2/ecological_model/auc_comparison.csv`에서 Karakin을 제외한 11개 코호트의 산술평균. phase-only AUC 0.615300 / 단계 내 0.500000 / NLL 0.574810 / Brier 0.194198; lite 0.719111 / 0.677389 / 0.532137 / 0.177616.
- cutoff와 변수 정의: `model_stgcn/phase_v2/grid.py`, `features.py`, `labels.py`.
- 전역 score 변환: `third_party/krafton_pubg_winprob/adaptation/train.py`의 `fit_platt`/`apply_platt`.
- weighted 목적함수: 고정된 공개 구현 `src/training/losses.py`. 시간 가중치와 survivor-complement도 normalized score에 의존하므로 공통 score 이동에 불변이다. 이는 표준 Cox와 동일한 손실이라는 뜻이 아니다.
- 통계 규칙: `model_stgcn/ptr/aggregate.py`의 `pool_contrasts`/`verdict`.
- 패치 변화: [공식 Update 41.1](https://pubg.com/en/news/9926), 2026-10-08 확인. 기존 `PUBGPatchNotes2026` 인용에 해당 원문 URL이 이미 포함돼 있다.

## 판정

현재 증거로 반영할 수 있는 서술·정의·통계 보고를 수정했다. 새 시험 자료, 온라인 eligibility 전수 감사, MLP/logistic 학습이 있어야 답할 수 있는 지적은 해결 완료로 표시하지 않는다. 첨부 리뷰의 Weak Reject는 과거 원고에 대한 모의 판정이므로, 이번 수정만으로 실제 채택 가능성이 확인됐다고 해석하지 않는다.
