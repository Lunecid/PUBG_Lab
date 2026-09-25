# PUBG_Lab

PUBG 매치를 시공간 그래프로 표현하고, 매 시점 살아 있는 팀 가운데 어느 팀이 곧 탈락할지 예측하는 생존 모델 연구 코드입니다.

[한국어](#한국어) · [English](#english)

## 한국어

### 연구 목적

배틀로얄에서 한 팀이 얼마나 오래 살아남는지는 위치, 자기장, 주변 적 팀과의 교전, 가진 자원이 시간에 따라 얽혀서 정해집니다. 이 저장소는 PUBG 매치를 10초 간격 스냅샷의 그래프로 바꾸고, 그래프 신경망과 순환 신경망으로 팀별 탈락 위험(hazard)을 추정합니다.

백성은(Seongeun Baek)의 석사 연구 주제로, 부산대학교 데이터사이언스전문대학원 [데이터사이언스연구실(DataLab)](https://datalab.pusan.ac.kr/datalab/index.do)에서 진행하고 있습니다(지도교수 권준호).

**그래프** (`main.py`)

- 노드: 그 시점에 살아 있는 플레이어. 39차원 피처를 다섯 축으로 묶었습니다. 신체 상태(체력·기절), 이동, 자기장 노출, 교전 압력, 자원(무기·방어구·회복 아이템)입니다.
- 엣지: 같은 팀끼리 잇는 ally 엣지, 가장 가까운 적 팀 플레이어 5명(k-NN)과 잇는 encounter 엣지.
- 자기장: 목표 원(safe zone)과 현재 경계(poison zone)를 두 개의 시계열로 함께 씁니다.

**모델** (`model/`의 `ArenaSurvivalNet`)

플레이어 인코더(GNN) → 팀 단위 어텐션 풀링 → 팀 간 GNN → 최근 5개 스냅샷을 읽는 GRU(자기장 정보 포함) → 게이트 결합 → 위험도 헤드

**예측 목표와 평가**

- 탈락이 일어나는 스냅샷마다, 살아 있는 팀 중 이번 또는 다음 스냅샷에 전멸하는 팀을 맞힙니다. 손실은 팀 순위 손실(softmax 교차 엔트로피)에 이진 생존 손실을 더한 값입니다.
- 지표: Hit@1/3/5, MRR, Spearman ρ, 시간 의존 C-index, Brier score, ECE(검증 세트로 맞춘 등위 회귀 보정 전후), 우승 팀 적중률.
- `explain.py`는 Integrated Gradients로 어떤 피처가 팀 탈락에 기여했는지 분석합니다.

### 데이터

- PUBG 공식 API에서 수집한 매치 정보와 텔레메트리 이벤트를 PostgreSQL `pubg` 스키마에 적재해 씁니다. 테이블 구조는 [스키마 다이어그램](pubg_survival%20-%20pubg_survival%20-%20pubg.png)에 있습니다.
- `main.py`가 읽는 테이블: `v_match_summary`, `rosters`, `participants`, `telem_match_start`, `telem_positions`, `telem_game_states`, `telem_kills`, `telem_damage`. `telem_groggy`, `telem_item_equip`, `telem_item_use`는 없으면 건너뜁니다.
- 기본 매치 조건: 참가자 20명 이상, 경기 시간 600–2400초, 튜토리얼·연습장 제외.
- 수집·적재 코드와 데이터(`data/`)는 이 저장소에 없습니다.

### 실행 방법

Python 3.12에서 개발했습니다. `torch`와 `torch_geometric`은 쓰는 CUDA 버전에 맞게 설치하세요.

```bash
pip install -r requirements.txt
cp .env.example .env    # DB 접속 정보 입력. .env는 커밋되지 않습니다.
```

맵·모드 이름은 DB에 저장된 값을 그대로 씁니다(예: 에란겔은 `Baltic_Main`).

| 단계 | 명령 | 출력 |
|---|---|---|
| 1. 그래프 생성 | `python main.py --map Baltic_Main --mode squad-fpp` | `data/graphs/{map}/{mode}/match_*.pt` |
| 2. 학습 | `python train.py --map Baltic_Main --mode squad-fpp` | `checkpoints/{map}/{mode}/{run_id}/best_model.pt` |
| 3. 시뮬레이션 | `python simulate.py` | `simulation_result_*.json`, `.html` |
| 4. 기여도 분석 | `python explain.py` | `*_attribution_*.json` |
| 5. 3D 시각화 | `python visualize.py <match_*.pt 경로>` | `data/match_graph_3d.html` |

- `main.py`에서 `--map`, `--mode`를 빼면 조건에 맞는 모든 매치를 처리합니다.
- `train.py`는 시작할 때 train/val/test 비율을 묻습니다. Enter를 누르면 70/15/15입니다.
- `simulate.py`는 인자 없이 실행하면 매치와 체크포인트를 고르게 하고, `explain.py`는 가장 최근 체크포인트를 씁니다. 둘 다 `--match`, `--checkpoint`로 직접 지정할 수 있습니다.

### 프로젝트 상태

진행 중인 석사 연구 코드입니다. 결과 수치는 논문이 확정되면 추가합니다.

알려진 한계:

- 맵 크기(8,160 m)와 시각화 배경 지도가 에란겔 기준이라, 다른 맵에는 그대로 맞지 않습니다.
- `result_viz.py`는 아직 파이프라인에 연결되지 않았고, 실행하려면 `matplotlib`을 따로 설치해야 합니다.

### 라이선스

아직 라이선스를 정하지 않았습니다. 코드는 읽을 수 있지만 재사용 권한은 없으니, 필요하면 먼저 연락해 주세요([@Lunecid](https://github.com/Lunecid)).

---

## English

### Purpose

How long a team survives in a battle royale depends on position, the shrinking zone, fights with nearby teams, and the resources it carries, all changing over time. This repository turns PUBG matches into graphs sampled every 10 seconds and estimates each team's elimination hazard with graph neural networks and a recurrent model.

This is the master's research topic of Seongeun Baek (백성은), carried out at the [Data Science Lab (DataLab)](https://datalab.pusan.ac.kr/datalab/index.do) of the Graduate School of Data Science, Pusan National University (advisor: Prof. Joonho Kwon).

**Graph** (`main.py`)

- Nodes: players alive at that moment, with 39 features grouped into five axes: physical state (health, knocked down), mobility, zone exposure, combat pressure, and resources (weapons, armor, healing items).
- Edges: `ally` edges within a team, and `encounter` edges to the 5 nearest enemy players (k-NN).
- Zone: two time series, the target circle (safe zone) and the current boundary (poison zone).

**Model** (`ArenaSurvivalNet` in `model/`)

player encoder (GNN) → attention pooling to teams → team-level GNN → GRU over the last 5 snapshots, with zone context → gated fusion → hazard head

**Target and evaluation**

- At each snapshot where an elimination happens, the model ranks the surviving teams to find the ones wiped out in this or the next snapshot. The loss is a softmax cross-entropy ranking loss plus a binary survival loss.
- Metrics: Hit@1/3/5, MRR, Spearman ρ, time-dependent C-index, Brier score, ECE (before and after isotonic calibration fitted on the validation set), and winner hit rate.
- `explain.py` uses Integrated Gradients to show which features drove a team's elimination.

### Data

- Match data and telemetry events collected from the official PUBG API, loaded into a PostgreSQL schema named `pubg`. See the [schema diagram](pubg_survival%20-%20pubg_survival%20-%20pubg.png).
- Tables read by `main.py`: `v_match_summary`, `rosters`, `participants`, `telem_match_start`, `telem_positions`, `telem_game_states`, `telem_kills`, `telem_damage`. `telem_groggy`, `telem_item_equip`, and `telem_item_use` are skipped if missing.
- Default match filter: at least 20 players, 600–2400 s long, tutorial and training maps excluded.
- The collection and loading code and the data itself (`data/`) are not in this repository.

### How to run

Developed on Python 3.12. Install `torch` and `torch_geometric` builds that match your CUDA version.

```bash
pip install -r requirements.txt
cp .env.example .env    # fill in the DB connection; .env is gitignored
```

Map and mode names are the raw values stored in the database (Erangel is `Baltic_Main`).

| Step | Command | Output |
|---|---|---|
| 1. Build graphs | `python main.py --map Baltic_Main --mode squad-fpp` | `data/graphs/{map}/{mode}/match_*.pt` |
| 2. Train | `python train.py --map Baltic_Main --mode squad-fpp` | `checkpoints/{map}/{mode}/{run_id}/best_model.pt` |
| 3. Simulate | `python simulate.py` | `simulation_result_*.json`, `.html` |
| 4. Attribution | `python explain.py` | `*_attribution_*.json` |
| 5. 3D view | `python visualize.py <path to match_*.pt>` | `data/match_graph_3d.html` |

- Without `--map`/`--mode`, `main.py` processes every match that passes the filter.
- `train.py` asks for the train/val/test ratio at startup; Enter gives 70/15/15.
- Run without arguments, `simulate.py` lets you pick a match and checkpoint, and `explain.py` uses the latest checkpoint. Both accept `--match` and `--checkpoint`.

### Status

Active master's research code. Results will be added once the thesis is final.

Known limitations:

- Map size (8,160 m) and the background map in the visualizer assume Erangel, so other maps are not handled correctly yet.
- `result_viz.py` is not wired into the pipeline yet and needs `matplotlib`, which is not in `requirements.txt`.

### License

No license has been chosen yet, so all rights are reserved. You are welcome to read the code; please get in touch before reusing it ([@Lunecid](https://github.com/Lunecid)).
