# legacy/ — Snapshot 시대 스택 (2026-04, 데모 전용)

이 디렉토리는 **2026년 4월 snapshot 시대**(10초 스냅샷 + 39d 피처)의 코드를 격리한 것이다.
**ICE 2027 논문 경로가 아니다.** 논문 경로는 `stdb/`(STDB ETL) + `model_stgcn/`(DynamicSTGNN)이다.
배경: `docs/RESEARCH_STATUS_2026-06-10.md` §3 연구 타임라인, `docs/CODE_REVIEW_2026-06-12.md` D9.

## 구성

| 파일 | 역할 |
|------|------|
| `main.py` | raw telemetry → `data/graphs/*.pt` 스냅샷 그래프 생성 (DB 필요) |
| `dataset.py` | .pt 로더 + 팀 그래프 구성 (`TeamSurvivalDataset`, `SnapshotDataset`) |
| `model/` | ArenaSurvivalNet (AgentEncoder → GroupPooling → GroupGNN → Temporal → HazardHead) |
| `train.py` | 학습 루프 (snapshot/legacy 모드) |
| `metrics.py` | 생존 모델 평가 (C-index, Brier 등) |
| `simulate.py` / `visualize.py` / `result_viz.py` | 시뮬레이션·시각화 데모 (발표용) |
| `explain.py` | 탈락 사례 attention 분석 |
| `run_meta.py` | 실행 메타데이터 (run_id, git hash 스탬프) |
| `adapters/` | 게임 어댑터 추상화 시도 (미사용, orphan) |

## 실행 방법

**반드시 repo 루트에서** 실행한다 (데이터 경로가 CWD 상대이므로):

```bash
export PG_PASSWORD='<비밀번호>'      # main.py 등 DB 사용 시
python legacy/train.py --pt_dir data/graphs --output_dir checkpoints
python legacy/simulate.py
```

`python legacy/<script>.py`로 실행하면 sys.path[0]이 `legacy/`가 되므로
스택 내부의 `from dataset import ...`, `from model.* import ...`는 그대로 동작한다.

## 주의

- 이 스택의 결과 수치는 논문에 사용하지 않는다.
- `data/graphs/` 레거시 .pt(~34GB)는 이 스택 전용이다.
- 새 기능은 여기에 추가하지 말 것 — 데모 유지보수(버그 수정)만 한다.
