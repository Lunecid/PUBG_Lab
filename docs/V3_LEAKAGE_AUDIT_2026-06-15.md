# V3 Leakage Audit 2026-06-15

이 문서는 V3 final experiment의 예측 입력 누수 여부를 점검한 기록이다.

## 1. 결론

직접 label column이 V3 tensor input에 들어간 정황은 확인되지 않았다.

그러나 기존 V3 rank/top-k head는 전체 phase sequence를 pooling해서 final rank/top-k를 예측했기 때문에, 실시간 예측이 아니라 사후 생존 궤적 복원에 가까웠다. 이 때문에 rank Spearman과 top-k AUC가 비정상적으로 높게 나왔다.

수정 사항:

| 항목 | 이전 | 수정 후 |
| --- | --- | --- |
| rank head input | full phase sequence pooled embedding | cut phase `h[:, p]` |
| top-k head input | full phase sequence pooled embedding | cut phase `h[:, p]` |
| rank/top-k output shape | `(B, N)` | `(B, T-1, N)` |
| rank/top-k eval unit | valid teams in match | at-risk team-phase cells |
| rank/top-k mask | `team_mask` | `target_mask` |
| interpretation | retrospective reconstruction | causal final-outcome forecast from cut phase |

## 2. Forbidden Input Columns

다음 column은 label 또는 outcome이므로 절대 input feature로 쓰면 안 된다.

```text
target_defined
hazard_next
survives_next_phase
eliminated_next_phase
final_rank
final_rank_norm
final_top3
final_top5
final_top10
final_winner
last_phase_alive
won
rank
victim_rank
match_end_rank
match_end_result
```

V3에는 `model_stgcn.v3.spec.FORBIDDEN_INPUT_FEATURES` guard를 추가했다. `split_feature_names()`와 `factorize_phase_tensors()`는 이 feature들이 들어오면 즉시 실패한다.

## 3. Manifest Audit

`tools/check_feature_manifest.py`는 각 feature family의 `forbidden_hits`를 보고하도록 수정했다.

현재 `stdb.v_team_phase_feature_v2` 기준 manifest audit 결과:

| Family | Forbidden Hits |
| --- | ---: |
| `f0_official_raw_phase` | 0 |
| `f1_prior_paper_raw_inputs` | 0 |
| `f2_prior_paper_score_summary` | 0 |
| `f4_current_conventional` | 0 |
| `f5_stdb_spatial` | 0 |
| `f6_network_phase_role_tier1` | 0 |
| `f7_habitat_pressure_tier1` | 0 |
| `f8_temporal_patch_proxy` | 0 |

주의: `target`이라는 문자열은 blanket-ban하지 않는다. 예를 들어 `hab_distance_to_target_safe_m`, `net_combat_unique_targets`는 label target이 아니라 safe-zone target 또는 combat target team을 의미한다.

## 4. Team Ordering Audit

과거 legacy snapshot path에는 final-rank 기반 team ordering 함수가 남아 있다.

| Function | Status |
| --- | --- |
| `stdb.queries._team_order` | legacy snapshot model용; final_rank order 사용 |
| `stdb.queries._phase_team_order` | V3 phase/MR path; `team_id` order 사용 |
| `stdb.snapshot_queries.fetch_snapshot_match_tensors` | `_phase_team_order` 사용 |
| `stdb.queries.fetch_phase_match_tensors` | `_phase_team_order` 사용 |

V3 phase/MR tensor는 `team_id` order를 사용하므로 final-rank ordering leak은 현재 경로에 없다.

## 5. Temporal Leakage Audit

현재 task의 cut은 phase end 기준이다.

따라서 phase `p`의 aggregate feature는 phase `p` 종료 시점에서 관측 가능한 값으로 해석한다. 예를 들어 다음은 next-phase hazard 예측에는 허용된다.

| Feature Type | 허용 이유 |
| --- | --- |
| `alive_players_end` | phase p 종료 시점의 상태 |
| `damage_taken` in phase p | phase p 종료 시점까지 관측된 combat |
| `outside_safe_ratio` in phase p | phase p 종료 후 계산 가능한 exposure |
| phase-local network features | phase p 내부까지 관측된 relation |

단, final rank/top-k를 phase `p`에서 예측할 때는 phase `p+1...T` 정보가 들어가면 안 된다. 기존 full-sequence pooling은 이 조건을 위반했다. 수정 후 rank/top-k는 cut hidden state `h[:, p]`만 사용한다.

## 6. Current Model Contract

수정 후 output contract:

```text
hazard_logits: (B, T-1, N)
rank_pred:     (B, T-1, N)
topk_logits:   (B, T-1, N, 2)
```

loss/eval mask:

```text
target_mask: (B, T-1, N)
```

rank/top-k target은 final outcome이지만, prediction input은 cut phase까지만 사용한다. 따라서 task는 다음과 같이 해석한다.

```text
At cut phase p, among teams still at risk, predict eventual final rank/top-k.
```

## 7. Invalidated Checkpoint

`model_stgcn/checkpoints/v3_full_main/main/mr_tcn/model.pt`는 이전 full-sequence rank/top-k head 정의로 학습된 checkpoint다.

이 checkpoint의 hazard 결과는 참고 가능하지만, rank/top-k 결과는 논문용으로 쓰면 안 된다. 수정 후 full run은 `V3_FORCE=1`로 재학습해야 한다.

권장 재시작:

```bash
PG_PASSWORD=104204 V3_FORCE=1 ./scripts/run_v3_full_main_matrix.sh
```

## 8. Remaining Audit Items

아직 남은 점검:

| 항목 | 이유 |
| --- | --- |
| alive-only rank phase bands | early/mid/late에서 final ranking forecast 분석 |
| shuffled graph controls | graph structure 자체의 의미 검증 |
| time-shuffled snapshot controls | MR temporal order의 의미 검증 |
| tabular baseline shared split | feature-family ablation과 같은 split/risk set 보장 |
| feature group importance | survival signal이 어떤 representation family에서 오는지 해석 |

