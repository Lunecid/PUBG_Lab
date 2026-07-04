# 필수 추가 실험 계획 및 실행 명령

이 문서는 현재 논문 방어력을 높이기 위해 추가한 세 가지 필수 실험을 정리한다.

---

## 1. Seed 반복 실험

목적:

- 현재 seed 42 중심 결과가 특정 split/초기화에 의존하지 않는지 확인한다.
- reviewer가 "우연히 좋은 split이 아니었나?"라고 물었을 때 방어한다.

기본 실행:

```bash
python -m model_stgcn.seed_repeat_experiments \
  --eval \
  --seeds 13 42 2026 \
  --edge-modes none proximity_combat full \
  --epochs 12 \
  --batch 16
```

산출물:

- `model_stgcn/checkpoints/required_experiments/seed_repeats/seed_repeat_metrics.csv`
- `model_stgcn/checkpoints/required_experiments/seed_repeats/seed_repeat_deltas.csv`
- `model_stgcn/checkpoints/required_experiments/seed_repeats/seed_repeat_summary.json`

해석 기준:

- `full`의 `delta_hazard_auc`, `delta_hazard_ap`가 seed/domain 전반에서 양수인지 확인한다.
- `proximity_combat`과 `full`의 차이가 seed에 따라 뒤집히는지 확인한다.

---

## 2. Random Edge Sanity Check

목적:

- full edge 모델의 성능 향상이 단순한 regularization 효과인지, 실제 관측된 팀-팀 관계 정보 때문인지 확인한다.
- `node_shuffle`은 active team subgraph의 구조를 node relabeling으로 섞어 density/degree 구조는 유지하되, 팀 feature와 edge의 정렬을 깨뜨린다.

기본 실행:

```bash
python -m model_stgcn.edge_sanity \
  --root model_stgcn/checkpoints/domain_runs_expanded_e12 \
  --edge-mode full \
  --repeats 5 \
  --batch 16
```

산출물:

- `model_stgcn/checkpoints/domain_runs_expanded_e12/edge_sanity/edge_sanity_predictions.csv`
- `model_stgcn/checkpoints/domain_runs_expanded_e12/edge_sanity/edge_sanity_metrics.csv`
- `model_stgcn/checkpoints/domain_runs_expanded_e12/edge_sanity/edge_sanity_metrics_aggregate.csv`
- `model_stgcn/checkpoints/domain_runs_expanded_e12/edge_sanity/edge_sanity_deltas.csv`

해석 기준:

- observed full edge가 node-shuffled edge보다 AUC/AP가 높고 BCE가 낮아야 한다.
- 이 결과가 나오면 "edge density 자체"가 아니라 "관측된 시공간 관계의 정렬"이 의미 있다는 주장을 강화할 수 있다.

---

## 3. Calibration / Brier / ECE

목적:

- AUC/AP와 확률 보정 품질을 분리해 평가한다.
- tuned full 모델은 ranking 성능은 좋지만 BCE가 나빠 calibration 문제가 있으므로 별도로 보고해야 한다.

전체 test cell 기준:

```bash
python -m model_stgcn.calibration \
  --predictions model_stgcn/checkpoints/phase_result_analysis/hazard_predictions_test_aligned.csv \
  --out-dir model_stgcn/checkpoints/required_experiments/calibration \
  --include-overall
```

Phase별 기준:

```bash
python -m model_stgcn.calibration \
  --predictions model_stgcn/checkpoints/phase_result_analysis/hazard_predictions_test_aligned.csv \
  --out-dir model_stgcn/checkpoints/required_experiments/calibration_by_phase \
  --group-cols phase_idx \
  --include-overall
```

산출물:

- `model_stgcn/checkpoints/required_experiments/calibration/calibration_metrics.csv`
- `model_stgcn/checkpoints/required_experiments/calibration/calibration_bins.csv`
- `model_stgcn/checkpoints/required_experiments/calibration_by_phase/calibration_metrics.csv`
- `model_stgcn/checkpoints/required_experiments/calibration_by_phase/calibration_bins.csv`

현재 기존 예측 파일 기준 핵심 결과:

| Model | Brier | BCE | ECE |
|---|---:|---:|---:|
| dynamic_full_e12 | 0.1705 | 0.5147 | 0.0076 |
| dynamic_full_tuned | 0.1871 | 0.5544 | 0.1236 |
| dynamic_none_e12 | 0.1737 | 0.5227 | 0.0067 |
| dynamic_proximity_combat_e12 | 0.1715 | 0.5171 | 0.0041 |
| dynamic_proximity_e12 | 0.1713 | 0.5167 | 0.0037 |
| tabular_baseline_hgb | 0.1704 | 0.5141 | 0.0066 |

해석:

- tuned full은 AUC/AP 최상위지만 calibration이 크게 나쁘다.
- untuned full은 tuned full보다 확률 보정이 훨씬 안정적이다.
- 논문에서는 tuned full을 ranking-oriented 모델, untuned full을 balanced/calibrated default로 구분하는 것이 안전하다.
