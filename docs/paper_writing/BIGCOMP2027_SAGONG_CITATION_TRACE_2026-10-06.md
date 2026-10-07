# BigComp 2027 Sagong 인용 추적 (후방 + 전방), 2026-10-06

- 성격: 읽기 전용 조사. 이 파일만 새로 썼고 커밋하지 않았다.
- 조사일: 2026-10-06. 모든 API 질의는 이 날 실행했다.
- 원칙: 서지는 아래 검증 소스에서 직접 확인한 값만 적었다. 확인하지 못한 값은 **미확인**으로 표시했다.
- 대상: (1) Sagong, Woo, Kim, Entertainment Computing 55 (2025) 101040 (이하 Sagong-EC), (2) Sagong & Kim, BigComp 2025 (이하 Sagong-SS).

## 0. 요약

**후방 (Sagong-EC 참고문헌 28편)**

- PDF 목록 28편과 Crossref 기탁 28편이 일치한다.
- 분류 (주 분류 1개씩): BR/PUBG 10, e스포츠 결과·승리 예측 11, 플레이어 행동 유형 4, 기타 3, 생존·위험 예측 0, 공간·이동 분석 0 (단독), ML 방법 0.
  - [5]는 궤적(이동) 클러스터링이라 유형과 공간·이동을 겸한다. 주 분류는 유형이다.
- **Sagong-EC의 참고문헌에는 생존분석(Cox, hazard, DeepHit 등) 문헌이 한 편도 없다.** RF, LR, XGBoost에도 방법론 인용이 없다. 이산 hazard 정식화는 이 논문의 인용 범위 밖이다.
- 권고 집계: (a) 8편 (신규 7 + 기존 유지 1), (b) 1편, (c) 19편.
  - (a) [7] [8] [9] [10] [11] [12] [21] [22]
  - (b) [5]
  - (c) 나머지 19편
- 우리가 이미 인용한 항목: Kaggle [11] 1편. [22] Hodge는 arXiv 판(제목이 다름)만 인용 중이라 같은 항목이 아니다. 대상 논문 두 편 자체는 BigComp bib에 있다 (학위 bib에는 Sagong-EC만 있다).

**전방 (이 두 논문을 인용한 논문)**

- 두 논문 모두 **인용 0편**이다. OpenAlex, Crossref, Semantic Scholar 세 곳이 일치한다. 색인 지연 가능성은 있다 (§4).
- 전문 검색에서 "FightScore"가 2026년 책 장 1편에 걸렸지만 PDF가 403이라 인용 여부는 **미확인**이다.

**상위 5개 추가 권고** (서지 검증 수준은 §5와 §9에 항목별로 표기)

1. KRAFTON 논문 제목 반영. GitHub 저장소 설명이 논문 제목을 알려 준다: "Real-Time Win Probability Prediction in Battle Royale Games via Survival Analysis". 논문 기록은 못 찾았다. 가장 가까운 비교 대상이라 서지와 본문 서술을 갱신해야 한다.
2. Hodge 외 2021 (IEEE ToG, Sagong [22]). 현재 bib는 arXiv 판만 있다.
3. Yu 외 2019 MOBA-Slice (Sagong [21]). 시점 조각 단위의 팀 간 상대 우위 평가라서 "팀 단위 경기 중 평가"의 가장 가까운 e스포츠 선행이다.
4. Gałka & Strzelecki 2021 (Sagong [12]) 과 Rosenbusch 외 2020 (검색 중 발견). PUBG와 배틀로얄 결과의 무작위성을 다룬다. 탈락을 확률(hazard)로 두는 근거다.
5. Marshall 외 2022 (검색 중 발견, FDG). CS:GO 경기 중 사망 예측이다. 본문의 Dota 2 사망 예측 문장 옆에 한 편을 더하는 비용이 작다.

**위협 (상세 §6)**

- 최대 위협은 KRAFTON 논문이다. 겹치는 지점이 네 곳이다.
  - 생존분석 정식화
  - 구역 중심·반경 입력
  - LogHaz 기준선 (이산 hazard)
  - 팀 간 attention 제거 변형
- 본문은 이미 "최초" 주장을 피하고 있다. 다만 서술 몇 곳을 정확히 맞춰야 한다 (§6 조치).
- 팀 단위 hazard, 서식지 프레이밍, 분해 설계 자체와 겹치는 선행은 검색 범위 안에서 찾지 못했다.

## 1. 방법과 검증 소스

| 코드 | 소스 | 쓴 곳 |
|---|---|---|
| S1 | Sagong-EC PDF의 참고문헌 목록 (pdftotext로 추출, 28편) | 후방 목록 원본, 각 항목의 PDF 표기 |
| S2 | Crossref REST API (DOI 조회, 서지 질의, 기탁 참고문헌, is-referenced-by-count) | 서지 확정, 28편 교차 확인 |
| S3 | OpenAlex API (works, cites 필터, fulltext.search, 저자별 works, 초록) | 서지 보완, 전방 인용, 초록 |
| S4 | Semantic Scholar Graph API (paper, citations, CorpusId) | 전방 인용, 식별자 |
| S5 | arXiv API | 초록 전문, 사전 게재 판 여부 |
| S6 | GitHub API: krafton-ai/pubg-winprob 의 README 원문, 저장소 메타데이터, 커밋 목록, 커밋 99ed9d1 의 소스(modules.py, backbone.py, heads.py) | KRAFTON 서술 |
| S7 | 연 페이지: WebSearch 결과, WebFetch (요약형 도구라 인용 문구는 S6의 원문으로 다시 확인함) | 보조 |

- 원고 서술 확인에는 `paper/bigcomp2027/sections/*.tex`와 두 bib 파일을 직접 읽었다.
- 한계: Crossref와 OpenAlex는 참고문헌 목록이 기탁·파싱된 논문만 인용으로 센다. 최근 논문이나 본문 언급만 있는 논문은 빠질 수 있다.

## 2. 후방 추적: Sagong-EC 참고문헌 28편

### 2.1 분류 집계

| 분류 | 편수 | 번호 |
|---|---|---|
| BR / PUBG 분석 | 10 | [1] [2] [7] [8] [9] [10] [11] [12] [24] [25] |
| e스포츠 결과·승리 예측 | 11 | [13]–[23] |
| 플레이어 행동 유형 | 4 | [5] [6] [26] [27] |
| 생존·위험 예측 | 0 | 없음 |
| 공간·이동 분석 | 0 (단독) | [5]가 궤적 군집화를 겸함 |
| ML 방법 | 0 | RF·LR·XGBoost는 위 e스포츠 문헌을 근거로 듦 (Sagong-EC §4.3) |
| 기타 | 3 | [3] [4] StarCraft 전략, [28] 몰입(flow) 이론 |

### 2.2 전체 목록

표기: 우리 인용 열의 "예"는 BigComp bib와 본문 모두에 있다는 뜻이다. 서지는 영문, 검증값만 적었다.

| # | 분류 | 서지 | 검증 | 우리 인용 | 권고 |
|---|---|---|---|---|---|
| 1 | BR/PUBG (웹) | Newzoo. "PUBG: Battlegrounds." 2024. https://newzoo.com/game/pubg-battlegrounds (accessed 14 Jun 2024, per Sagong). | S1만, 페이지 미열람 | 아니오 | (c) |
| 2 | BR/PUBG (웹) | T. Burgos. "How long does a round of PUBG last?" 2024. https://www.playbite.com/how-long-does-a-round-of-pubg-last/ (accessed 30 Aug 2024, per Sagong). | S1만 | 아니오 | (c) |
| 3 | 기타 (RTS) | D. Churchill, M. Buro, R. Kelly. "Robust Continuous Build-Order Optimization in StarCraft." 2019 IEEE Conference on Games (CoG), pp. 1-8, 2019. doi:10.1109/CIG.2019.8848109 | S2 | 아니오 | (c) |
| 4 | 기타 (RTS) | P. García-Sánchez, A. Tonda, A. M. Mora, G. Squillero, J. J. Merelo. "Towards automatic StarCraft strategy generation using genetic programming." 2015 IEEE Conference on Computational Intelligence and Games (CIG), pp. 284-291, 2015. doi:10.1109/CIG.2015.7317940 | S2 | 아니오 | (c) |
| 5 | 유형 (궤적) | B. Ingram, B. Rosman, C. van Alten, R. Klein. "Play-style Identification through Deep Unsupervised Clustering of Trajectories." 2022 IEEE Conference on Games (CoG), pp. 393-400, 2022. doi:10.1109/CoG51982.2022.9893680 | S2, S3 초록 | 아니오 | (b) |
| 6 | 유형 | A. Drachen, R. Sifa, C. Bauckhage, C. Thurau. "Guns, swords and data: Clustering of player behavior in computer games in the wild." 2012 IEEE CIG, pp. 163-170, 2012. doi:10.1109/CIG.2012.6374152 | S2 | 아니오 | (c) |
| 7 | BR/PUBG | M. M. Mamulpet. "PUBG Winner Placement Prediction Using Artificial Neural Network." Int. J. Eng. Appl. Sci. Technol. 3(12): 107-118, 2019. doi:10.33564/ijeast.2019.v03i12.020 (저자명 표기: Crossref "Manjunath Mamulpet, Madhurya", OpenAlex "Madhurya Manjunath Mamulpet") | S2, S3 | 아니오 | (a) |
| 8 | BR/PUBG | H. Aggarwal, S. Gautam, S. Malik, A. Khosla, A. Punj, B. Bhasin. "Rank Prediction in PUBG: A Multiplayer Online Battle Royale Game." In: Emerging Technologies in Data Mining and Information Security (Proc. IEMIS 2020, Vol. 2), Advances in Intelligent Systems and Computing, Springer, 2021, pp. 859-867. doi:10.1007/978-981-33-4367-2_82 | S2, S3 | 아니오 | (a) |
| 9 | BR/PUBG | Y. Indulkar. "PUBG Winner Ranking Prediction using R Interface 'h2o' Scalable Machine Learning Platform." 2021 Int. Conf. on Emerging Smart Computing and Informatics (ESCI), IEEE, pp. 300-305, 2021. doi:10.1109/ESCI50559.2021.9396823 | S2, S3 | 아니오 | (a) |
| 10 | BR/PUBG | N. F. Ghazali, N. Sanat, M. A. As'ari. "Esports Analytics on PlayerUnknown's Battlegrounds Player Placement Prediction using Machine Learning." Int. J. Human and Technology Interaction (IJHaTI) 5(1): 17-28, 2021. DOI 미확인 | S1, S3 (W3158249181) | 아니오 | (a) |
| 11 | BR/PUBG | M. Addison Howard, inversion. "PUBG Finish Placement Prediction (Kernels Only)." Kaggle, 2018. https://kaggle.com/competitions/pubg-finish-placement-prediction | S1 | 예 (kagglePubgFinishPlacement; 학위 bib에도 있음) | (a) 유지 |
| 12 | BR/PUBG | P. Gałka, A. Strzelecki. "How Randomness Affects Player Ability to Predict the Chance to Win at PlayerUnknown's Battlegrounds (PUBG)." The Computer Games Journal 10(1-4): 1-18, 2021. doi:10.1007/s40869-020-00117-1 | S2, S4 | 아니오 | (a) |
| 13 | e스포츠 | I. Makarov, D. Savostyanov, B. Litvyakov, D. I. Ignatov. "Predicting Winning Team and Probabilistic Ratings in 'Dota 2' and 'Counter-Strike: Global Offensive' Video Games." AIST 2017, LNCS, Springer, pp. 183-196. doi:10.1007/978-3-319-73013-4_17 (연도: Crossref 2017, Sagong 표기 2018) | S2, S3 | 아니오 | (c) |
| 14 | e스포츠 | K. Song, T. Zhang, C. Ma. "Predicting the winning side of DotA 2." 2015. Semantic Scholar CorpusId 5200255 (venue 공란, DOI 없음). 참고: Sagong 본문은 이 문헌을 League of Legends로 설명하지만 기록의 제목은 DotA 2다. | S1, S4 | 아니오 | (c) |
| 15 | e스포츠 | K. Kalyanaraman. "To win or not to win? A prediction model to determine the outcome of a DotA2 match." Technical report, 2014 (S1). S4 CorpusId 6337723은 연도를 2015로 적음. 그 이상 미확인 | S1, S4 | 아니오 | (c) |
| 16 | e스포츠 (경기 중) | F. Johansson, J. Wikström. "Result Prediction by Mining Replays in Dota 2." KTH DiVA repository, 2015. 문서 유형 미확인 | S3 (W1008770268) | 아니오 | (c) |
| 17 | e스포츠 (경기 중) | R. Ani, V. Harikumar, A. K. Devan, O. S. Deepa. "Victory prediction in League of Legends using Feature Selection and Ensemble methods." 2019 Int. Conf. on Intelligent Computing and Control Systems (ICCS), pp. 74-77, 2019. doi:10.1109/ICCS45141.2019.9065758 | S2, S3 | 아니오 | (c) |
| 18 | e스포츠 | T. D. Smithies, M. J. Campbell, N. Ramsbottom, A. J. Toth. "A Random Forest approach to identify metrics that best predict match outcome and player ranking in the esport Rocket League." Scientific Reports 11(1): 19285, 2021. doi:10.1038/s41598-021-98879-9 | S2 | 아니오 | (c) |
| 19 | e스포츠 | K. U. Birant. "Multi-view rank-based random forest: A new algorithm for prediction in eSports." Expert Systems 39(2): e12857. doi:10.1111/exsy.12857 (Crossref issued 2021) | S2 | 아니오 | (c) |
| 20 | e스포츠 | W. Wang (OpenAlex: Weiqi Wang). "Predicting Multiplayer Online Battle Arena (MOBA) Game Outcome Based on Hero Draft Data." National College of Ireland, 2016. 학위 유형은 미확인 (Sagong 표기: Ph.D. thesis) | S3 (W2592451185) | 아니오 | (c) |
| 21 | e스포츠 (경기 중) | L. Yu, D. Zhang, X. Chen, X. Xie. "MOBA-Slice: A Time Slice Based Evaluation Framework of Relative Advantage Between Teams in MOBA Games." In: Computer Games (CGW 2018), Communications in Computer and Information Science, Springer, 2019, pp. 23-40. doi:10.1007/978-3-030-24337-1_2; arXiv:1807.08360 | S2, S5 초록 | 아니오 | (a) |
| 22 | e스포츠 (경기 중) | V. J. Hodge, S. Devlin, N. Sephton, F. Block, P. I. Cowling, A. Drachen. "Win Prediction in Multiplayer Esports: Live Professional Match Prediction." IEEE Transactions on Games 13(4): 368-379, 2021. doi:10.1109/TG.2019.2948469 | S2, S3 초록 | 부분 (arXiv:1711.06498 판만, 제목이 다름) | (a) |
| 23 | e스포츠 | A. Semenov, P. Romov, S. Korolev, D. Yashkov, K. Neklyudov. "Performance of Machine Learning Algorithms in Predicting Game Outcome from Drafts in Dota 2." AIST 2016, CCIS, Springer, 2017, pp. 26-37. doi:10.1007/978-3-319-52920-2_3 | S2, S3 | 아니오 | (c) |
| 24 | BR/PUBG (장르) | G. Choi, M. Kim. "Battle royale game: In search of a new game genre." Int. J. Culture Technology (IJCT) 2(2): 5, 2018 (S1). Crossref·OpenAlex에 없음. DOI와 쪽수는 미확인 (웹 검색 요약에 pp. 5-11이라는 2차 표기가 있으나 원문 페이지는 열지 않음) | S1만 | 아니오 | (c) |
| 25 | BR/PUBG (웹) | CGMagazine. "Why is PUBG so popular & is it worth playing?" 2022. https://www.cgmagonline.com/articles/why-is-pubg-popular-is-it-worth-playing/ (accessed 29 Aug 2024, per Sagong). | S1만 | 아니오 | (c) |
| 26 | 유형 | S. C. J. Bakkes, P. H. M. Spronck, G. van Lankveld. "Player behavioural modelling for video games." Entertainment Computing 3(3): 71-79, 2012. doi:10.1016/j.entcom.2011.12.001 | S2, S3 | 아니오 | (c) |
| 27 | 유형 | E. Loria, A. Marconi. "Player Types and Player Behaviors: analyzing correlations in an on-the-field gamified system." CHI PLAY '18 Companion Extended Abstracts, pp. 531-538, 2018. doi:10.1145/3270316.3271526 (Crossref 제목은 "Player Types and Player Behaviors") | S2 | 아니오 | (c) |
| 28 | 기타 (flow) | P. Tomlinson, J. H. Salisbury. "Reconciling Csikszentmihalyi's Broader Flow Theory: With Meaning and Value in Digital Games." Trans. Digital Games Research Association 2(2), 2016. doi:10.26503/todigra.v2i2.34 (Crossref 저자 순서는 Salisbury, Tomlinson) | S2 | 아니오 | (c) |

### 2.3 우리와 관련된 항목 상세 (BR/PUBG, 경기 중 e스포츠 예측, 공간·이동)

생존·위험 예측 항목은 Sagong-EC 목록에 없다. 아래 문장 번호는 `paper/bigcomp2027/sections/02-related-work.tex`의 현재 문단 구조를 기준으로 한다.

**[7] [8] [9] [10] 경기 후 순위 예측 묶음.**
- 내용: Kaggle의 경기 후 선수 통계(킬, 피해량, 이동 거리, 아이템 등)로 최종 순위를 회귀·분류한다. Sagong-EC §3.2와 Table 2의 요약이다. [7]은 OpenAlex 초록에서 확인했다 (선수 위치와 통계로 순위를 예측, 신경망 계열). [8]은 OpenAlex에 초록이 없고 [10]은 초록을 조회하지 않아, 둘의 내용은 Sagong-EC의 요약만 근거다.
- 권고 (a): 1문단 첫 문장 "Early PUBG studies predict final placement from post-match aggregates such as the Kaggle placement data [...]"의 인용 묶음에 넣는다. 이 문장의 "입력은 경기 종료 후에야 완성된다"를 뒷받침한다.
- 비용 판단: 8쪽 제한에서 네 편을 한 묶음으로 넣어도 본문 길이는 늘지 않는다. Sagong-SS의 Crossref 기탁 참고문헌 4편이 Ghazali, Indulkar, Aggarwal, Kaggle이므로 같은 저자 그룹이 정석으로 보는 인용은 [8] [9] [10]이다. 줄여야 한다면 DOI가 있는 [8] [9]를 우선한다.
- 우리 인용 여부: 아니오.

**[11] Kaggle.** 이미 인용 중이다. 유지한다.

**[12] Gałka & Strzelecki 2021.**
- 내용: PUBG 승리 가능성에 대한 무작위 요인의 영향이다. Sagong-EC의 요약은 위치, 보급, 안전구역, 무기 정확도를 무작위 요인으로 들고 그중 무기 정확도가 예측에 가장 유용했다고 한다. 초록은 출판사가 제공하지 않아 못 읽었다. Semantic Scholar의 TLDR은 "무작위 요인이 승리 가능성을 좌우할 수 있다"는 취지다.
- 권고 (a): 1문단의 "인-매치 위험" 논의에 "PUBG 결과는 위치와 안전구역 같은 무작위 요인에 크게 좌우된다 [Gałka & Strzelecki; Rosenbusch 외, §5]" 한 문장을 둔다. 탈락 위험을 확률로 다루는 이유가 된다.
- 보조 (b) 아이디어: 무기 정확도 계열 특징을 시험한다. 우리 ETL(`stdb/`, `model_stgcn/ptr/`의 .py)에서 accuracy, hit_ratio, shots_fired, weapon_acc 이름은 검색되지 않았다. 다른 이름의 동등 특징이 있는지는 **미확인**이다. 발사 이벤트 대비 피해 이벤트 비율을 팀 교전 특징으로 계산해 Hab 블록 제거 실험처럼 한 블록으로 넣어 볼 수 있다.
- 우리 인용 여부: 아니오.

**[21] MOBA-Slice (Yu 외 2019).**
- 내용 (arXiv 초록): 시점 조각(time slice) 단위로 팀 간 상대 우위를 평가하는 프레임워크다. AlphaGo의 value network와 비슷한 학습 기반 평가이고, 할인된 평가 함수를 신경망으로 근사한다. DotA2에서 결과 예측 정확도가 DotA Plus Assistant보다 3.7% 높고, 남은 경기 시간 예측과 팀 간 상대 우위 평가도 지원한다.
- 권고 (a): 3문단 "Win prediction has been studied for MOBA games [...]"에 "경기 중 팀 단위 평가는 시점 조각 단위로 이뤄져 왔다 [Yu 외]"를 더한다. 차이는 한 줄로 쓴다: 우리의 조각은 시계 간격이 아니라 안전구역 단계 경계이고, 목표는 승패나 우위가 아니라 다음 단계 탈락 hazard다.
- 우리 인용 여부: 아니오.

**[22] Hodge 외 2021.**
- 내용 (OpenAlex 초록): 프로 e스포츠에서 실시간(live) 승리 예측을 처음 종합적으로 다룬 사례 연구다. 프로 수준의 DotA 2 경기 예측 모델을 평가하고 국제 대회에서 시험했으며, 5분 경과 시점에 최대 85% 정확도를 보고하고 평가 기준을 제안한다.
- 권고 (a, 강): 3문단 MOBA 승리 예측 문장의 첫 인용으로 쓴다. 현재 bib의 hodge2017mobaWinPrediction은 arXiv:1711.06498이고 제목이 다르다 ("Win Prediction in Esports: Mixed-Rank Match Prediction in Multi-player Online Battle Arena Games"). arXiv 초록은 혼합 랭크 데이터로 프로 경기를 예측할 수 있는지를 묻는다. 게재판과 같은 연구의 이전 판인지는 **미확인**이다 (arXiv에 journal_ref와 DOI가 없고 Semantic Scholar도 별개 기록으로 둔다). 게재판을 추가하거나 교체하는 편이 안전하다.
- 우리 인용 여부: 부분 (arXiv 판만).

**[16] Johansson & Wikström, [17] Ani 외.** 둘 다 경기 일부 구간 또는 경기 전·중 특징을 쓰는 RF·앙상블 기준 연구다 (Sagong-EC §3.3의 요약). 우리 본문의 MOBA 문장에는 이미 Yang 2020과 Junior 2023(실시간)이 있어 추가 가치가 작다. (c).

**[5] Ingram 외 2022.**
- 내용 (OpenAlex 초록): 다차원·가변 길이 궤적을 LSTM 오토인코더로 비지도 군집화해 플레이 스타일을 식별한다. 완전한 궤적과 부분 궤적 모두에서 작동한다. Sagong-EC의 요약은 데이터가 GridWorld와 Super Mario Bros라고 한다.
- 권고 (b): 아이디어 하나. 팀별 위치 궤적(약 10초 간격)을 단계 창 안에서 가변 길이로 받는 LSTM 오토인코더를 학습해, 잠재 코드를 hazard 모델의 추가 입력으로 쓴다. 수작업 서식지 특징(Hab 블록 제거 시 AUC -0.011)과의 대비로 "학습된 이동 표현이 수작업 분산·가장자리 특징을 대신하는가"를 묻는 후속 과제다. 8쪽 본문에는 넣지 않고 논의의 후속 과제 한 문장으로 둔다.
- 우리 인용 여부: 아니오.

**[24] Choi & Kim 2018.** Sagong-EC §4.2는 이 문헌의 "actions of play"(이동, 은신·탐색, 차량 조작, 지도로 안전구역·피해구역 정보 획득, 아이템 파밍, 상대 제거)를 Aggressiveness와 Activeness 두 범주로 합쳤다. 우리는 BR 장르 정의 문장이 따로 필요하지 않고 서지(DOI, 쪽수)도 미확인이라 (c)다. 필요해지면 같은 저자의 G. Choi, M. Kim, "Gameplay of Battle Royale Game by Rules and Actions of Play," 2018 IEEE 7th Global Conference on Consumer Electronics (GCCE), pp. 599-600, doi:10.1109/GCCE.2018.8574781 (Crossref에서 확인)가 검증 가능한 대안이다. 두 논문의 관계는 미확인이다.

**[1] [2] [25] 웹 자료.** Newzoo는 Twitch 시청 시간(Sagong-EC 서론), Burgos는 경기 길이 약 30분, CGMagazine은 인기 이유를 뒷받침하는 용도로 쓰였다. 우리 서론은 PUBG 개발자 문서 [API]로 충분하다. 세 항목 모두 (c).

## 3. Sagong 두 논문에서 가져갈 것 (인용 목록 밖, 본문 확인 사항 포함)

Sagong-EC PDF 본문을 직접 읽고 확인한 내용이다.

**데이터와 입력 (우리 서술 검증).**
- 데이터: 2021-01-01~01-07 라이브 경쟁 모드, 솔로만 사용 (듀오·스쿼드는 "팀 전략이 결과에 영향을 주므로" 제외), 451,699명, 7,719경기, 자료 2,868,613건. 위치는 10초마다 기록. 데이터는 비공개 ("The data that has been used is confidential").
- 단계: 0.1(착지)에서 9.5까지 0.5 간격. 1.0~9.5가 18개이고 착지 단계를 더해 19단계다. 본문의 "19 stages"와 일치한다.
- 특징 (Table 4): 공격 4개(attack_count, kill_count, total_damage, assist_count)와 활동 4개(distance, max_density, item_count, carepackage_use), 모두 8개. `phase_max_density`는 반경 100 m 안 플레이어 수의 단계 내 최댓값이다. FightScore와 ActiveScore는 각 범주 특징의 누적 합이다. 구역 거리나 구역 기하 특징은 없다. 본문의 서술(8개 특징, 합산 점수, 100 m 인원수만 타인 정보)은 PDF와 일치한다.
- **점검 필요 (C1의 공정성).** Sagong-EC §4.3은 예측 입력을 "Table 4 특징 + FightScore + ActiveScore + player type + survival rate"라고 쓴다. `02-related-work.tex`는 8개 특징과 두 점수만 적고 있다. 우리 Sagong 재구현이 player type(1~4)과 유형별 생존율까지 넣었는지 확인한다. 빠졌다면 서술이나 구현을 맞춘다.
- Sagong 특징 대부분은 우리 ETL에도 있어 보인다 (`carepackage_pickup_count`, `assists`, `assists_count`가 `stdb/*.py`에서 검색됨). 150열 목록 대비 정확한 포함 여부는 미확인이다.

**논의에서 인용할 문장 (모두 Sagong-EC 본문).**
1. 한계 절: 딥러닝(LSTM, 트랜스포머)은 더 큰 데이터가 필요해 "모델 복잡도와 데이터 요구량의 상충"이 있다. 우리 C3(학습 곡선 교차점 약 700경기, 탐색적)가 이 상충을 직접 측정한다. 논의 한 문장으로 쓸 수 있다.
2. 한계 절: 스쿼드 모드의 팀 협응 변수와 Warzone, Free Fire 같은 다른 BR(단계와 줄어드는 구역을 공유)을 후속 과제로 든다. 우리 팀 단위 + 12개 코호트 설계가 첫 번째를 다룬다. 교차 게임은 여전히 열려 있다.
3. 결과: 우세 전략이 단계에 따라 바뀐다 (Camper는 전 단계에서 우세하고 마지막 단계로 갈수록 늘며, Traveler는 초중반 이후 줄고, Hunter는 마지막 단계에서 급증). 단계 조건부 표현(C1)의 동기로 쓸 수 있다.

**Sagong-SS (BigComp 2025) 확인 사항.**
- Crossref: 제목 "Survival Sequences: Win Prediction from a Strategy Sequence Approach", pp. 105-106, 발행일 2025-02-09, DOI 10.1109/bigcomp64353.2025.00030. BigComp bib와 일치한다.
- OpenAlex 초록: 안전구역이 줄어드는 BR의 단계별 전략이 생존에 중요하다는 전제에서 Fight/Active 두 관점의 행동 분석과, 단계별 전략 시퀀스를 입력으로 하는 승리 예측 딥러닝(LSTM, GRU, RNN)을 제안한다. 즉 이 논문의 목표는 **다음 단계 생존이 아니라 최종 승리**다. `docs`의 서론 v2 초안은 [Sagong25a, Sagong25b]를 묶어 "다음 반 단계 생존 분류"로 서술하지만, 실제 본문(`01-introduction.tex`, `02-related-work.tex`)은 이미 둘을 구분해 쓰고 있다.
- 기탁 참고문헌 4편: Ghazali 2021, Indulkar 2021, Aggarwal 2021, Kaggle 2018.
- 미확인: 전문(IEEE Xplore)은 열지 못했다. 단위가 선수인지 팀인지, 현재 상태 기준선이 있는지는 `02-related-work.tex`의 VERIFY 주석대로 **미확인**이다.

## 4. 전방 추적: 이 두 논문을 인용한 논문

| 질의 | Sagong-EC | Sagong-SS |
|---|---|---|
| OpenAlex work id, cited_by_count | W4415299921, 0 | W4409019416, 0 |
| OpenAlex `filter=cites:` 결과 수 | 0 | 0 |
| Crossref is-referenced-by-count (기탁 참고문헌 수) | 0 (28) | 0 (4) |
| Semantic Scholar citationCount, `/citations` 목록 | 0, 비어 있음 | 0, 비어 있음 |

- OpenAlex 전문 검색: "Hunter or Camper"는 Sagong-EC 자신과 무관한 논문들뿐이다. "FightScore"는 3건이다. Sagong-EC, 무관한 2011년 PLoS ONE 논문, 2026년 책 장 1편이다.
  - 책 장: S. Piatysotska, A. Yefremenko, Наталія Долгополова, V. Romanenko, Y. Tropin (OpenAlex 표기), "Artificial Intelligence in Esports as a Case Of Digital Transformation in Physical Culture Education," pp. 423-462, doi:10.31110/lrdtinedu-3.8. WoS 논문 81편을 검토한 리뷰다. OpenAlex 참고문헌은 0건이고 PDF 요청은 HTTP 403이었다. Sagong을 인용하는지는 **미확인**이다.
- 저자 후속 연구: OpenAlex의 Sagong 2025년 이후 works는 두 논문뿐이다. Huy Kang Kim의 2025년 이후 27건 중 게임 관련 제목은 이 두 편과 치팅 탐지("Battlefield"), 스피드런 포렌식뿐이다. BR 후속 논문은 없다.
- WebSearch ("Hunter or Camper" FightScore 등): 인용 논문은 나오지 않았다 (ScienceDirect 페이지, ResearchGate, OCSLab 출판 목록만 나옴).
- 해석: Sagong-EC는 2025-10-14 게재 승인이라 1년이 안 됐고 Sagong-SS는 2쪽짜리 단신이라 0건은 자연스럽다. 우리 논문이 첫 인용 중 하나일 가능성이 높다 (색인 지연 가능성은 남음).

## 5. 목록 밖에서 찾은 추가 후보 (위협 검색 중 발견, 서지 검증됨)

| 항목 | 서지 | 검증 | 우리 인용 | 권고 |
|---|---|---|---|---|
| KRAFTON 논문 | "Real-Time Win Probability Prediction in Battle Royale Games via Survival Analysis." 저자, 학회, 연도 **미확인**. GitHub 저장소 설명이 "Official implementation of" 뒤에 이 제목을 적는다 (krafton-ai/pubg-winprob, 생성 2026-02-08, 마지막 push 2026-08-26, Apache-2.0, 커밋 99ed9d1). | S6 (저장소 설명과 README 원문). 논문 기록은 arXiv(제목 포함 질의 4건), OpenAlex, Crossref, Semantic Scholar에서 모두 찾지 못함 | 소프트웨어만 인용 (krafton_pubg_winprob) | (a) 제목을 서지와 본문에 반영 (§6, §7) |
| Marshall 외 2022 | S. Marshall, P. Mavromoustakos Blom, P. H. M. Spronck. "Enabling Real-Time Prediction of In-game Deaths through Telemetry in Counter-Strike: Global Offensive." Proc. FDG 2022, pp. 1-10. doi:10.1145/3555858.3555859 | S3만 (Crossref 미조회) | 아니오 | (a) 3문단 사망 예측 문장에 추가. 내용 (OpenAlex 초록): 텔레메트리로 3초 창 안의 사망을 예측, 36개 특징, CNN/RNN/LSTM 비교, LSTM이 최고(F1 0.38) |
| Rosenbusch 외 2020 | H. Rosenbusch, J. Röttger, D. Rosenbusch. "Would Chuck Norris Certainly Win the Hunger Games? Simulating the Result Reliability of Battle Royale Games Through Agent-Based Models." Simulation & Gaming 51(4): 461-476, 2020. doi:10.1177/1046878120914336 | S2, S3 초록 | 아니오 | (a) 선택. 에이전트 기반 시뮬레이션에서 BR 결과가 실제 기량과 약하게만 일치한다 (초록) |
| Katona 외 2019 (게재판) | A. Katona, R. Spick, V. J. Hodge, S. Demediuk, F. Block, A. Drachen, J. A. Walker. "Time to Die: Death Prediction in Dota 2 using Deep Learning." 2019 IEEE Conference on Games (CoG), pp. 1-8. doi:10.1109/CIG.2019.8847997 | S2 | arXiv 판만 (katona2019dotaDeath) | bib 교체 권고 (§7) |
| Xie & Dong 2026 | Q. Xie, H. Dong. "Gameplay Rating and Winning Strategy Evaluation in PlayerUnknown's Battlegrounds." Proc. CIBDA 2026 (ACM), pp. 1520-1526. doi:10.1145/3813822.3814069 | S3 초록 | 아니오 | (c). 초록상 winPlacePerc 예측이라 [7]-[10]과 같은 계열로 보임 |
| Zhou 외 2021 | N. Zhou, M. Doerner, L. Ryan, C. Pierse. "AI Coach for Battle Royale Games." CHI PLAY '21, pp. 280-286. doi:10.1145/3450337.3483456 | S3 초록 | 아니오 | (c). Apex Legends 선수 기량 모델과 코칭 |
| Liu 외 2020 | Jiaxin Liu, Jiaxin Huang, Chen Ru-yun, Tianchang Liu, Liang Zhou (OpenAlex 표기). "A Two-Stage Real-time Prediction Method for Multiplayer Shooting E-Sports." Journal of the Association for Information Systems (OpenAlex 표기). DOI, 권, 쪽수, 대상 게임 **미확인** | S3 초록만 | 아니오 | 인용 전 원문 확인 필요. 초록은 RF 기반 실시간 경기 순위 예측이라고 함 |
| Aşkın & Aşkın 2025 | G. D. Aşkın, O. Aşkın. "Spatial Organization in Battle Royale Games: A Design-Based Analysis on PUBG." SSRN preprint, 2025. doi:10.2139/ssrn.5676547. 초록 **미확인** | S2, S3 (제목만) | 아니오 | 공간 구성의 설계 분석으로 보이나 내용 미확인 |

그 밖에 훑어본 항목 (제목 확인만, 관련성 낮음): "n-Player Stochastic Duel Game Model with Applied Deep Learning and Its Modern Implications" (Sensors 22(6): 2422, 2022), "Team Heart-Rate Dynamics and Physiological Coupling in Professional PUBG Esports: Evidence from Phase-Based Analysis" (SSRN 2026).

## 6. 겹침·위협 점검

판정 기준: 검색 범위 안에서 찾은 것만 쓴다. "없다"고 단정하지 않는다.

### T1. KRAFTON 논문 (위협 큼, 서술 정밀화 필요)

**확인한 사실** (README 원문과 커밋 99ed9d1의 소스. 논문 본문은 못 읽음).
- README 첫 문장: "Real-time win probability prediction for PUBG esports matches using Transformer-based survival analysis." 단위는 스쿼드 (경기당 최대 16).
- 구역: `ZoneEmbedding`이 파란 구역과 흰 구역의 (x, y, 반경)을 임베딩하고, `backbone.py` 185~190행에서 이를 **모든 스쿼드 토큰에 더한다** (별도 구역 토큰이 아니다). 특징 목록에도 "distance to bluezone/whitezone"와 "bluezone/whitezone center and radius"가 있다.
- 머리(head): 생존 시간 회귀, Cox log-hazard (`HazardHead`, "higher = more likely to die soon"), 승자 분류.
- 손실: README의 논문 손실은 `weighted_cox`다 ("Elimination + survival likelihood with early-focus time weighting (paper loss)"). 평범한 `cox`("full retrospective")와 다르다.
- 표집: 단계에 맞춘 192개 시점. 178개는 후반으로 갈수록 촘촘하게 뽑고 (1단계 약 60초 간격에서 7~10단계 약 5초 간격까지), 나머지 14개는 단계 경계 고정 앵커다.
- 기준선 9종 (Table 1): RandomGuess, RuleBased, LogReg, LogHaz, RFClf, LGBMReg, LGBMClf, CoxPH, DeepSurv. README의 LogHaz 하이퍼파라미터는 `num_durations`다. 이산 시간 logistic hazard(pycox)로 보이지만 그것은 추정이다.
- 구조 소거: `--encoder mlp`는 스쿼드 간 attention이 없는 변형이다.
- 지표: Winner Accuracy, C-index, IBS, ECE, log loss. 10개 단계별 분해도 보고한다.

**우리 주장과의 겹침.**
1. 정식화 (C2): 생존분석 틀의 BR 예측이 이미 있고, 기준선에 이산 hazard(LogHaz)가 있어 보인다. 우리 차이는 "경기 전체에서 표집한 시점에서 사망 시각 순위를 학습"과 "안전구역 단계 경계마다 위험집합 위에서 다음 단계 탈락 확률을 학습"의 차이다. 이 차이를 구체적으로 적어야 한다. 본문은 "최초" 주장을 하지 않으므로 그 점은 안전하다.
2. 표현 (C1): KRAFTON도 구역 중심, 반경, 구역까지 거리를 입력한다. 우리 분해에서 KRAFTON 대비 표현의 몫이 +0.014로 작은 것과 일치한다. "KRAFTON은 구역을 다루지 않는다"로 읽힐 문장은 쓰면 안 된다.
3. 구조 (C3, RQ2): 스쿼드 간 attention 소거 실험이 코드에 있어, 그 논문이 같은 질문을 이미 보고했을 수 있다. 우리는 "다른 목표(다음 단계 hazard), 다른 데이터(공개 API 경기), 다른 손실에서의 재검토"로 표현하고 있고 이 표현을 유지한다.

**조치 (제안, 미실행).**
- `02-related-work.tex` KRAFTON 문단: "README refers to an accompanying paper, but we located no paper record"에 제목을 덧붙인다 (저장소 설명이 제시하는 제목, 기록은 못 찾음). 구역 임베딩과 LogHaz 기준선, `weighted_cox` 손실을 코드가 보여 주는 사실로 한 문장 추가한다.
- docs 서론 v2 초안 P4의 "Neither design represents the arena as shrinking habitat"는 구역 중심·반경·거리가 입력에 있으므로 "neither frames elimination as local extinction under stepwise habitat loss"처럼 프레이밍 차이로 고친다. 본문 `01-introduction.tex`는 "explicit object node"를 쓰고 있어 이 점에서는 정확하다 (구역이 별도 토큰이 아님을 소스로 확인).
- **K2/K3a 점검:** 우리 KRAFTON 적응(`05-experiments.tex`)이 README의 논문 손실 `weighted_cox`를 쓰는지 확인한다. 평범한 `cox`를 썼다면 "+0.051 (Cox 사망 순위 대 다음 단계 hazard)"는 논문 손실이 아닌 변형과의 비교이므로 그렇게 명시해야 한다.
- 제출 전에 KRAFTON 논문 공개 여부를 다시 확인한다 (arXiv, OpenReview, 학회).

### T2. Sagong 두 논문 (위협 낮음)

- 팀 단위 hazard와는 직접 겹치지 않는다. Sagong-EC는 선수 단위·솔로 전용이고 팀 모드를 명시적으로 제외했다. Sagong-SS는 최종 승리를 예측한다 (단위 미확인).
- 심사자가 물을 지점: Sagong-EC의 한계 절이 팀 협응 변수와 시퀀스 모델을 후속 과제로 이미 적었다. 우리 서술에서 이 두 가지를 "이 논문이 다룬다"로 연결하되 선점 주장은 하지 않는다.
- Sagong-EC는 "첫 번째 실시간 전략 적응 생존 예측"을 주장한다. 이 주장은 우리와 충돌하지 않는다.

### T3. 구역 = 서식지 프레이밍 (위협 낮음)

- OpenAlex "battle royale AND (habitat OR ecology OR extinction OR ecological)" 274건, "(PUBG OR battle royale) AND (local extinction OR habitat loss OR species-area OR metapopulation)" 4건, arXiv 질의를 훑었다. 결과는 메타휴리스틱 "Battle Royale Optimizer", 생물학 은유 제목, 미디어 연구뿐이었다. BR 분석에서 생태학 프레이밍을 쓴 논문은 이 범위에서 찾지 못했다.
- 가장 가까운 공간 관련 항목은 Aşkın & Aşkın 2025 (SSRN, 설계 분석)이며 초록을 못 읽었다 (미확인).

### T4. 분해 설계 (위협 낮음)

- 정식화, 표현, 구조로 BR 예측 격차를 분해한 연구는 검색 범위에서 찾지 못했다. Sagong 계열 두 논문(분류 대 시퀀스)과 KRAFTON은 여러 축이 동시에 다르므로 분해가 불가능하다는 우리 서술과 일치한다.

### T5. 참고 (위협 아님)

- Xie & Dong 2026은 경기 후 순위 예측이다. 팀 단위 hazard, 서식지, 분해와 겹치지 않는다.
- 2026-09-24 arXiv "PUBG Ally: A Conversational Embodied Agent as an AI Teammate"는 AI 팀원 연구이며 예측 과제가 아니다.

## 7. references.bib 점검 메모 (`paper/bigcomp2027/references.bib`)

| 키 | 점검 결과 | 제안 |
|---|---|---|
| sagong2025hunterCamper | Crossref와 일치: 권 55, 논문 번호 101040, 2025-09, DOI, 저자 3인 | 변경 없음. ScienceDirect PII는 S187595212500120X (검색 결과 URL, 미열람) |
| sagong2025survivalSequences | Crossref와 일치: pp. 105-106, 2025-02-09, DOI, 저자 2인 | 변경 없음 |
| hodge2017mobaWinPrediction | arXiv:1711.06498 제목이 게재판(ToG 2021)과 다름. 관계 미확인 | 게재판 추가 (§9) |
| katona2019dotaDeath | IEEE CoG 2019 게재판이 있음 (Crossref) | @misc를 @inproceedings로 교체 (§9) |
| krafton_pubg_winprob | 저장소 설명이 논문 제목을 알려 줌. 기존 VERIFY 주석의 "no paper record" 판단은 여전히 유효 | note에 제목 추가 (§9) |
| kagglePubgFinishPlacement | Sagong-EC [11]과 같은 항목 | 변경 없음 |

학위 논문 bib (`/Users/lunecid/학위논문/references.bib`)에는 Sagong-EC, Kaggle, Hodge(arXiv), Katona(arXiv)가 있고 Sagong-SS는 없다. 학위 논문에 같은 서술을 옮긴다면 같은 교체가 필요하다.

## 8. 미확인 목록

- 웹 자료 [1] [2] [25]: 페이지를 열지 않았고 S1의 URL과 접속일만 있다.
- [10] DOI, [24] DOI와 쪽수, [14] [15] 발행처, [16] 문서 유형, [20] 학위 유형.
- [13] 연도 (Crossref 2017, Sagong 표기 2018), [19] 호 연도 (Crossref issued 2021).
- [12]의 초록 (출판사가 제공하지 않음). 내용 서술은 S4 TLDR과 Sagong-EC의 요약이다.
- KRAFTON 논문: 저자, 학회, 연도, 공개 여부. 저장소의 "마지막 push 2026-08-26"은 GitHub API 값이다.
- Sagong-SS 전문: 단위(선수 또는 팀), 기준선 여부.
- Piatysotska 외 2026 책 장이 Sagong을 인용하는지.
- Aşkın & Aşkın 2025 초록, Liu 외 2020의 대상 게임과 서지.
- Marshall 외 2022는 OpenAlex로만 확인했다.
- 전방 인용 0건은 색인 지연으로 과소 집계일 수 있다.
- WebFetch는 요약형 도구라, 인용·세부 사실은 GitHub API 원문으로 다시 확인한 것만 적었다.

## 9. 붙여넣기용 BibTeX (검증된 필드만, 미확인은 VERIFY 주석)

```bibtex
% KRAFTON: replace the existing entry's note (paper title from the repository description)
@misc{krafton_pubg_winprob,
  author       = {{KRAFTON AI}},
  title        = {{pubg-winprob}},
  howpublished = {Source code, \url{https://github.com/krafton-ai/pubg-winprob}},
  note         = {Official implementation of ``Real-Time Win Probability Prediction in Battle Royale Games via Survival Analysis'' (repository description; no paper record located). Commit 99ed9d1, 2026-08-26; Apache-2.0; accessed Oct. 2026}
}

@article{hodge2021liveWinPrediction,
  author  = {Victoria J. Hodge and Sam Devlin and Nick Sephton and Florian Block and Peter I. Cowling and Anders Drachen},
  title   = {{Win Prediction in Multiplayer Esports: Live Professional Match Prediction}},
  journal = {IEEE Transactions on Games},
  volume  = {13},
  number  = {4},
  pages   = {368--379},
  year    = {2021},
  doi     = {10.1109/TG.2019.2948469}
}

@inproceedings{yu2019mobaSlice,
  author    = {Lijun Yu and Dawei Zhang and Xiangqun Chen and Xing Xie},
  title     = {{MOBA-Slice: A Time Slice Based Evaluation Framework of Relative Advantage Between Teams in MOBA Games}},
  booktitle = {Computer Games (CGW 2018, held with IJCAI 2018), Communications in Computer and Information Science},
  publisher = {Springer},
  pages     = {23--40},
  year      = {2019},
  doi       = {10.1007/978-3-030-24337-1_2},
  note      = {arXiv:1807.08360}
}

@article{galka2021randomnessPubg,
  author  = {Piotr Ga{\l}ka and Artur Strzelecki},
  title   = {{How Randomness Affects Player Ability to Predict the Chance to Win at PlayerUnknown's Battlegrounds (PUBG)}},
  journal = {The Computer Games Journal},
  volume  = {10},
  number  = {1--4},
  pages   = {1--18},
  year    = {2021},
  doi     = {10.1007/s40869-020-00117-1}
}

@article{rosenbusch2020chuckNorris,
  author  = {Hannes Rosenbusch and Jonas R{\"o}ttger and David Rosenbusch},
  title   = {{Would Chuck Norris Certainly Win the Hunger Games? Simulating the Result Reliability of Battle Royale Games Through Agent-Based Models}},
  journal = {Simulation \& Gaming},
  volume  = {51},
  number  = {4},
  pages   = {461--476},
  year    = {2020},
  doi     = {10.1177/1046878120914336}
}

% VERIFY: OpenAlex record only; confirm in Crossref/ACM before submission.
@inproceedings{marshall2022csgoDeaths,
  author    = {Stefan Marshall and Paris Mavromoustakos Blom and Pieter H. M. Spronck},
  title     = {{Enabling Real-Time Prediction of In-game Deaths through Telemetry in Counter-Strike: Global Offensive}},
  booktitle = {Proceedings of the International Conference on the Foundations of Digital Games (FDG)},
  pages     = {1--10},
  year      = {2022},
  doi       = {10.1145/3555858.3555859}
}

% Replaces the arXiv-only entry katona2019dotaDeath (published version, Crossref-verified).
@inproceedings{katona2019timeToDieCoG,
  author    = {Adam Katona and Ryan Spick and Victoria J. Hodge and Simon Demediuk and Florian Block and Anders Drachen and James Alfred Walker},
  title     = {{Time to Die: Death Prediction in Dota 2 using Deep Learning}},
  booktitle = {2019 IEEE Conference on Games (CoG)},
  pages     = {1--8},
  year      = {2019},
  doi       = {10.1109/CIG.2019.8847997}
}

% Optional group for "post-match placement prediction" (Related Work, first sentence).
@inproceedings{indulkar2021pubgH2o,
  author    = {Yash Indulkar},
  title     = {{PUBG Winner Ranking Prediction using R Interface `h2o' Scalable Machine Learning Platform}},
  booktitle = {2021 International Conference on Emerging Smart Computing and Informatics (ESCI)},
  publisher = {IEEE},
  pages     = {300--305},
  year      = {2021},
  doi       = {10.1109/ESCI50559.2021.9396823}
}

@incollection{aggarwal2021rankPubg,
  author    = {Harsh Aggarwal and Siddharth Gautam and Sandeep Malik and Arushi Khosla and Abhishek Punj and Bismaad Bhasin},
  title     = {{Rank Prediction in PUBG: A Multiplayer Online Battle Royale Game}},
  booktitle = {Emerging Technologies in Data Mining and Information Security: Proceedings of IEMIS 2020, Volume 2},
  series    = {Advances in Intelligent Systems and Computing},
  publisher = {Springer},
  pages     = {859--867},
  year      = {2021},
  doi       = {10.1007/978-981-33-4367-2_82}
}

% VERIFY: author name order (Crossref and OpenAlex differ).
@article{mamulpet2019pubgAnn,
  author  = {Madhurya Manjunath Mamulpet},
  title   = {{PUBG Winner Placement Prediction Using Artificial Neural Network}},
  journal = {International Journal of Engineering Applied Sciences and Technology},
  volume  = {3},
  number  = {12},
  pages   = {107--118},
  year    = {2019},
  doi     = {10.33564/ijeast.2019.v03i12.020}
}

% VERIFY: no DOI found in Crossref or OpenAlex; fields from OpenAlex W3158249181 and Sagong's reference list.
@article{ghazali2021esportsPubg,
  author  = {N. F. Ghazali and N. Sanat and Muhammad Amir As'ari},
  title   = {{Esports Analytics on PlayerUnknown's Battlegrounds Player Placement Prediction using Machine Learning}},
  journal = {International Journal of Human and Technology Interaction (IJHaTI)},
  volume  = {5},
  number  = {1},
  pages   = {17--28},
  year    = {2021}
}
```
