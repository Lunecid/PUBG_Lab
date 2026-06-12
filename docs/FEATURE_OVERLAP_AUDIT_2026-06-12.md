# Feature 중복·중첩 전수 감사 — 2026-06-12

방법: target_defined 셀 92,943행(매치 1,500개, UUID 정렬 표본)에서
F4+F5+F6+F7 수치 컬럼 147개를 ① exact 복제(NaN 동형 포함), ② 구조적
항등식(공식 잔차 검증), ③ Spearman |ρ|≥0.95 전수 스캔 + 연결성분으로
검사. 표본/스크립트 산출물: `/tmp/feature_audit_sample.pkl`.

**한 줄 결론: 명목 147개 중 |ρ|≥0.95 기준 실효 신호는 ~87개.
사망 3, exact 복제 9, 결정론적 변환 ~10, 고상관 클러스터 27개.**

---

## A. 사망 컬럼 (3) — 전 행 0

| 컬럼 | 소속 | 비고 |
| --- | --- | --- |
| `nearby_enemy_combat_events_200m` | F4 | HGB 입력에만 존재 |
| `third_party_exposure_count` | F4 | HGB 입력에만 존재 |
| `lootbox_pickup_count` | F4 | **GNN 66d에도 포함됨** (상수 0 채널) |

FEATURE_DESIGN_V2 §6의 "near-zero coverage" 경고가 전수 확인됨.
ETL 재구축 전까지 모든 manifest/스펙에서 제외 권장.

## B. Exact 복제 (9) — 값이 완전히 동일

F7이 F4/F5에서 복사한 컬럼 (5):

```text
hab_center_shift_m        == safe_center_shift_m   (F4)
hab_zone_gap_start_m      == zone_gap_start_m      (F4)
hab_zone_gap_end_m        == zone_gap_end_m        (F4)
hab_observed_speed_mps    == avg_speed_mps         (F5)
hab_outside_safe_ratio    == outside_safe_ratio    (F5)
```

F6 계약상 동치 (4, 설계 문서의 redundancy 그대로):

```text
net_combat_unique_targets   == net_combat_out_degree
net_combat_unique_attackers == net_combat_in_degree
net_prox_isolation_m        == net_prox_min_distance_m
net_copress_top1_weight     == net_copress_max_weight
```

## C. 결정론적 변환 (공식 잔차 ≤1e-4로 전부 검증)

단조 변환 (트리 모델에 정보량 0 추가):

```text
hab_safe_area_start_m2      = pi * safe_radius_start_m^2     (rho = +1.000)
hab_safe_area_end_m2        = pi * safe_radius_end_m^2       (rho = +1.000)
hab_local_enemy_density_200m = count_200m / 0.12566          (rho = +1.000)
net_prox_top1_weight        = exp(-min_distance/200)         (rho = -1.000)
hab_crowding_pressure       = 1 / (area_per_alive_team 비례) (rho = -1.000)
```

선형 결합 / 파생:

```text
safe_radius_delta_m   = end - start            (F4 내부)
zone_gap_start/end_m  = poison_r - safe_r      (F4 내부)
net/cum_combat_total_degree·strength = out + in
hab_safe_area_delta_m2 = end - start
hab_shrink_rate_m2ps  ≈ -delta/duration        (rho -0.994)
hab_loss_ratio        = 1 - (r_end/r_start)^2
hab_required_speed    = distance_to_target/duration (rho 0.995 w/ distance)
hab_migration_deficit = required - observed
hab_dist_to_safe_boundary_* = |signed_*|
```

## D. 고상관 클러스터 (|ρ|≥0.95 연결성분, 사망 제외 144 → 87 신호)

주요 다중 클러스터 27개 중 핵심:

1. **[13] 시간-존 메가클러스터**: `phase_idx`·phase 시각 3종·safe/poison
   반경 5종·면적 4종 — 게임 설계상 시계가 존 크기를 결정하므로 전부 한 축.
2. **[7] co-pressure 블록**: F6 copress 6컬럼+per_alive가 사실상 1신호.
3. **[6]×2 combat 중복**: `damage_dealt/events/distinct_opponents_damaged`(F4)
   ≈ `net_combat_out_degree/strength`(F6) (ρ 0.963~0.995), taken 방향 동일.
   → **F6 phase-local combat degree/strength는 F4 집계의 재표현**.
   F6의 진짜 신규 combat 신호 = balance, reciprocity, per_min/per_alive, cum_*.
4. F5 내부: `avg_enemy_density_100m ~ path_near_enemy_ratio` 0.997,
   `outside_safe_duration_s ~ outside_safe_ratio` 0.995.
5. F7 신규 신호 중 겹침: `hab_safe_alignment_to_target ~ safe_alignment`(F5)
   0.957 (phase 말 블루존 중심≈타깃 중심), `hab_dist_to_safe_boundary_start
   ~ mean_dist_to_boundary`(F5) 0.954, `hab_alive_teams ~ avg_alive_teams` 0.981.
6. F4 내부 잔존 쌍: spread avg~max 0.990, damage~events 0.97~0.98,
   equip~weapon_equip 0.966, phase_idx~phase_start_s 0.983.

**겹침 없이 깨끗한 F7 신규 신호** (어느 클러스터에도 안 묶임):
`hab_loss_ratio`(스케일프리), `hab_center_shift_per_radius`,
`hab_zone_gap_pressure`, `hab_signed_boundary_distance_*`,
`hab_distance_to_target_safe_m`(+deficit), `hab_edge_pressure_ratio`,
`hab_rotation_delay_ratio`, `hab_local_enemy_count_100/500m`,
`hab_safe_entry_enemy_count_200m`, `hab_safe_entry_congestion`,
`hab_refuge_scarcity_simple` — **+0.0168 증분의 출처 후보**.

---

## 영향 평가

1. **오늘 밤 B3/B4**: 영향 없음 — 두 팔(graph on/off)이 동일한 141d
   입력을 받으므로 중복은 비교를 편향시키지 않는다. 큐된 실험은 그대로 둔다.
2. **보고된 ablation 증분은 중복에 의해 부풀려지지 않음** — HGB는 중복
   컬럼에서 추가 정보를 얻지 못하므로 +0.0168 등은 순수 신규 신호의 효과.
   부풀려진 것은 "피처 개수" 서술뿐 (149 명목 ≠ 149 신호).
3. **논문 서술 주의**: family 크기(43/32)를 정보량처럼 쓰지 말 것.
   실효 신호 ~87 인용 가능 (이 문서 기준).

## 권장 정리안 (manifest v2.1 — 미적용, 결정 대기)

원칙: **family 정의는 단독-ablation 해석을 보존**해야 하므로, 단독 실행에
의미 있는 cross-family 복사(`hab_outside_safe_ratio` 등)는 family에는 남기되
"결합 스택"(GNN 141d, full-stack HGB)에서는 dedupe된 union을 쓴다.

무손실 제거 (정보 손실 0, family 정의에서도 제거 가능):

```text
사망 3:        nearby_enemy_combat_events_200m, third_party_exposure_count,
               lootbox_pickup_count (GNN 스펙 포함)
F6 동치 4:     unique_targets, unique_attackers, isolation_m, copress_top1_weight
F6 파생 3:     total_degree, total_strength, cum_total_strength
F7 파생 6:     area_delta, shrink_rate, density_200m, required_speed,
               crowding_pressure(또는 per_alive 중 1), unsigned boundary 2종 중
               (signed 유지 시) unsigned 2
```

결합 스택 한정 추가 dedupe (단독 family에는 유지):

```text
F7 복사 5:     center_shift, zone_gap_start/end, observed_speed, outside_safe_ratio
F6 중복:       phase-local combat out/in degree·strength (F4와 ρ≥0.96)
               → balance/reciprocity/per_min/per_alive/cum_*만 결합 스택에
```

적용 시점 권고: B3/B4 결과 해석 후 (현 계약으로 학습된 모델과의
비교 가능성 유지를 위해), v2.1 manifest로 일괄 반영.
