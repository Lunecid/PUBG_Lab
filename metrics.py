"""
Survival Evaluation Metrics
============================
논문 결과표 기본 구성 (CLAUDE.md §8):
  - IPCW concordance index (primary)
  - Cumulative/dynamic AUC
  - Integrated Brier Score (IBS)
  - Phase-wise calibration
  - Final rank Spearman/Kendall correlation
  - Species-Area z-value (empirical vs predicted)

scikit-survival 표준 메트릭 + 커스텀 phase-wise 분석.
"""

import numpy as np
from collections import defaultdict
from scipy import stats as scipy_stats


# ============================================================
# 1. 핵심 생존 분석 메트릭
# ============================================================

def compute_concordance_index(hazard_preds, events, times):
    """
    IPCW concordance index 계산.

    Parameters:
        hazard_preds: np.array [N] — 모델 예측 hazard (높을수록 위험)
        events: np.array [N] — 0/1 (1=탈락)
        times: np.array [N] — 이벤트/관측 시점

    Returns:
        dict with c_index, concordant, discordant, tied
    """
    if events.sum() == 0:
        return {"c_index": 0.5, "concordant": 0, "discordant": 0,
                "tied_risk": 0, "tied_time": 0, "note": "no events"}

    try:
        from sksurv.metrics import concordance_index_censored

        event_bool = events.astype(bool)
        c_index, concordant, discordant, tied_risk, tied_time = \
            concordance_index_censored(event_bool, times, hazard_preds)

        return {
            "c_index": c_index,
            "concordant": concordant,
            "discordant": discordant,
            "tied_risk": tied_risk,
            "tied_time": tied_time,
        }
    except (ImportError, ValueError):
        return _concordance_index_manual(hazard_preds, events, times)


def _concordance_index_manual(hazard_preds, events, times):
    """scikit-survival 없을 때 fallback."""
    concordant = 0
    discordant = 0
    tied = 0
    n = len(hazard_preds)

    for i in range(n):
        if events[i] == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            if times[j] > times[i]:
                if hazard_preds[i] > hazard_preds[j]:
                    concordant += 1
                elif hazard_preds[i] < hazard_preds[j]:
                    discordant += 1
                else:
                    tied += 1

    total = concordant + discordant + tied
    c_index = (concordant + 0.5 * tied) / max(total, 1)
    return {
        "c_index": c_index,
        "concordant": concordant,
        "discordant": discordant,
        "tied_risk": tied,
        "tied_time": 0,
    }


def compute_ipcw_concordance(hazard_preds, events, times, train_events=None, train_times=None):
    """
    IPCW (Inverse Probability of Censoring Weighted) concordance index.
    Primary endpoint metric.

    Parameters:
        hazard_preds: np.array [N]
        events: np.array [N] — 0/1
        times: np.array [N]
        train_events, train_times: 학습 데이터 (censoring 분포 추정용)

    Returns:
        dict with ipcw_c_index, ...
    """
    if events.sum() == 0:
        return {"ipcw_c_index": 0.5, "note": "no events"}

    try:
        from sksurv.metrics import concordance_index_ipcw

        # sksurv 구조화 배열
        def make_structured(events, times):
            dt = np.dtype([('event', bool), ('time', float)])
            arr = np.empty(len(events), dtype=dt)
            arr['event'] = events.astype(bool)
            arr['time'] = times
            return arr

        test_surv = make_structured(events, times)

        if train_events is not None and train_times is not None:
            train_surv = make_structured(train_events, train_times)
        else:
            train_surv = test_surv

        # tau: 최대 평가 시점
        tau = min(times[events == 1].max() if events.sum() > 0 else times.max(),
                  train_times.max() if train_times is not None else times.max())

        c_index, _, _, _, _ = concordance_index_ipcw(
            train_surv, test_surv, hazard_preds, tau=tau
        )

        return {"ipcw_c_index": c_index, "tau": tau}
    except (ImportError, ValueError) as e:
        # fallback to standard concordance
        result = compute_concordance_index(hazard_preds, events, times)
        result["ipcw_c_index"] = result["c_index"]
        result["ipcw_note"] = f"fallback: {e}"
        return result


def compute_dynamic_auc(hazard_preds, events, times, eval_times=None,
                        train_events=None, train_times=None):
    """
    Cumulative/dynamic AUC at specific time points.

    Parameters:
        hazard_preds: np.array [N]
        events: np.array [N]
        times: np.array [N]
        eval_times: np.array — 평가 시점 (없으면 자동 설정)

    Returns:
        dict with mean_auc, auc_per_time
    """
    try:
        from sksurv.metrics import cumulative_dynamic_auc

        def make_structured(events, times):
            dt = np.dtype([('event', bool), ('time', float)])
            arr = np.empty(len(events), dtype=dt)
            arr['event'] = events.astype(bool)
            arr['time'] = times
            return arr

        test_surv = make_structured(events, times)
        if train_events is not None and train_times is not None:
            train_surv = make_structured(train_events, train_times)
        else:
            train_surv = test_surv

        if eval_times is None:
            # 10%, 25%, 50%, 75%, 90% quantile
            event_times = times[events == 1]
            if len(event_times) < 5:
                return {"mean_auc": 0.5, "auc_per_time": {}}
            eval_times = np.percentile(event_times, [10, 25, 50, 75, 90])

        # 범위 제한
        min_t = times.min()
        max_t = times[events == 1].max() if events.sum() > 0 else times.max()
        eval_times = eval_times[(eval_times > min_t) & (eval_times < max_t)]

        if len(eval_times) == 0:
            return {"mean_auc": 0.5, "auc_per_time": {}}

        aucs, mean_auc = cumulative_dynamic_auc(
            train_surv, test_surv, hazard_preds, eval_times
        )

        auc_per_time = {float(t): float(a) for t, a in zip(eval_times, aucs)}
        return {"mean_auc": float(mean_auc), "auc_per_time": auc_per_time}

    except (ImportError, ValueError):
        return {"mean_auc": 0.5, "auc_per_time": {}, "note": "sksurv unavailable"}


def compute_brier_score(survival_probs, events, times, eval_times=None,
                        train_events=None, train_times=None):
    """
    Integrated Brier Score (IBS).

    Parameters:
        survival_probs: np.array [N, T] — 각 시점의 생존 확률
                        또는 np.array [N] — 단일 시점 생존 확률
        events: np.array [N]
        times: np.array [N]
        eval_times: np.array [T]

    Returns:
        dict with ibs, brier_per_time
    """
    try:
        from sksurv.metrics import integrated_brier_score, brier_score

        def make_structured(events, times):
            dt = np.dtype([('event', bool), ('time', float)])
            arr = np.empty(len(events), dtype=dt)
            arr['event'] = events.astype(bool)
            arr['time'] = times
            return arr

        test_surv = make_structured(events, times)
        if train_events is not None and train_times is not None:
            train_surv = make_structured(train_events, train_times)
        else:
            train_surv = test_surv

        if survival_probs.ndim == 1:
            # 단일 시점 → Brier score만
            if eval_times is None:
                return {"ibs": float('nan'), "note": "need eval_times for IBS"}
            # 각 eval_time에 대해 동일 확률 사용 (근사)
            survival_probs = np.tile(survival_probs[:, None], (1, len(eval_times)))

        if eval_times is None:
            return {"ibs": float('nan'), "note": "need eval_times"}

        # 범위 제한
        min_t = times.min()
        max_t = times[events == 1].max() if events.sum() > 0 else times.max()
        valid = (eval_times > min_t) & (eval_times < max_t)
        eval_times = eval_times[valid]
        survival_probs = survival_probs[:, valid] if survival_probs.ndim > 1 else survival_probs

        if len(eval_times) < 2:
            return {"ibs": float('nan'), "note": "insufficient eval_times"}

        ibs = integrated_brier_score(train_surv, test_surv, survival_probs, eval_times)
        return {"ibs": float(ibs)}

    except (ImportError, ValueError) as e:
        return {"ibs": float('nan'), "note": f"error: {e}"}


# ============================================================
# 2. Phase-wise 평가
# ============================================================

def phase_wise_calibration(hazard_preds, events, phases):
    """
    Phase별 예측 확률 vs 실제 탈락률 비교.

    Parameters:
        hazard_preds: np.array [N] — 예측 hazard 확률 (sigmoid 후)
        events: np.array [N] — 0/1
        phases: np.array [N] — zone phase (int)

    Returns:
        dict: {phase: {"predicted": mean_pred, "observed": mean_event,
                        "n_samples": count, "calibration_error": |pred - obs|}}
    """
    result = {}
    unique_phases = sorted(set(phases))

    for p in unique_phases:
        mask = (phases == p)
        n = mask.sum()
        if n == 0:
            continue

        pred_mean = float(hazard_preds[mask].mean())
        obs_mean = float(events[mask].mean())
        cal_error = abs(pred_mean - obs_mean)

        result[int(p)] = {
            "predicted": pred_mean,
            "observed": obs_mean,
            "n_samples": int(n),
            "calibration_error": cal_error,
        }

    return result


def phase_wise_concordance(hazard_preds, events, times, phases):
    """
    Phase별 C-index.

    Returns:
        dict: {phase: c_index}
    """
    result = {}
    unique_phases = sorted(set(phases))

    for p in unique_phases:
        mask = (phases == p)
        if mask.sum() < 10 or events[mask].sum() < 2:
            continue

        ci = compute_concordance_index(
            hazard_preds[mask], events[mask], times[mask]
        )
        result[int(p)] = ci["c_index"]

    return result


# ============================================================
# 2b. Phase-matched AUC (신규 primary)
# ============================================================

def phase_matched_auc(hazard_preds, events, phases):
    """
    Phase별 AUC: 같은 phase 내에서 event=1 vs event=0 분류 성능.
    시간 축 혼동 없이 순수 팀 상태 판별력 측정.

    Returns:
        dict: {phase: auc, "mean_auc": weighted_mean}
    """
    from sklearn.metrics import roc_auc_score

    result = {}
    weighted_sum = 0
    total_weight = 0

    for p in sorted(set(phases)):
        mask = phases == p
        e = events[mask]
        h = hazard_preds[mask]

        if e.sum() < 1 or (1 - e).sum() < 1:
            continue

        try:
            auc = roc_auc_score(e, h)
            n = int(mask.sum())
            result[int(p)] = auc
            weighted_sum += auc * n
            total_weight += n
        except ValueError:
            continue

    result["mean_auc"] = weighted_sum / max(total_weight, 1)
    return result


def per_match_hazard_rank(hazard_preds, events, times, final_ranks, match_ids):
    """
    매치별로 마지막 생존 시점의 hazard를 팀별로 집계 → 최종 순위와 비교.
    팀별 aggregation으로 repeated-measures 문제 해결.

    Returns:
        dict: {spearman_rho, n_matches}
    """
    # 팀별 마지막 hazard 집계
    # (match_id, final_rank)로 그룹핑 — final_rank는 팀 고유값
    team_hazards = defaultdict(list)
    for i in range(len(hazard_preds)):
        key = (match_ids[i], final_ranks[i])
        team_hazards[key].append((times[i], hazard_preds[i]))

    # 매치별 (team_rank, max_time_hazard) 쌍 구성
    match_data = defaultdict(list)
    for (mid, rank), time_haz_list in team_hazards.items():
        # 마지막 시점의 hazard (가장 최신 상태)
        last_time, last_haz = max(time_haz_list, key=lambda x: x[0])
        match_data[mid].append((rank, last_haz))

    spearman_list = []
    for mid, pairs in match_data.items():
        if len(pairs) < 3:
            continue
        ranks = np.array([p[0] for p in pairs])
        haz = np.array([p[1] for p in pairs])

        if np.std(haz) < 1e-8:
            continue

        sp = scipy_stats.spearmanr(haz, ranks)
        rho = float(sp.statistic) if hasattr(sp, 'statistic') else float(sp[0])
        spearman_list.append(rho)

    return {
        "spearman_rho": float(np.mean(spearman_list)) if spearman_list else 0.0,
        "n_matches": len(spearman_list),
    }


# ============================================================
# 3. 최종 순위 상관
# ============================================================

def rank_correlation(predicted_risks, final_ranks, match_ids=None):
    """
    예측 risk score vs 최종 순위 상관.
    매치별로 계산 후 평균.

    Parameters:
        predicted_risks: np.array [N]
        final_ranks: np.array [N]
        match_ids: np.array [N] (optional) — 매치별 분리

    Returns:
        dict with spearman_rho, kendall_tau, per_match stats
    """
    if match_ids is None:
        # 전체 배치
        sp = scipy_stats.spearmanr(predicted_risks, final_ranks)
        kt = scipy_stats.kendalltau(predicted_risks, final_ranks)
        return {
            "spearman_rho": float(sp.statistic) if hasattr(sp, 'statistic') else float(sp[0]),
            "spearman_p": float(sp.pvalue) if hasattr(sp, 'pvalue') else float(sp[1]),
            "kendall_tau": float(kt.statistic) if hasattr(kt, 'statistic') else float(kt[0]),
            "kendall_p": float(kt.pvalue) if hasattr(kt, 'pvalue') else float(kt[1]),
        }

    # 매치별
    unique_matches = sorted(set(match_ids))
    spearman_list = []
    kendall_list = []

    for mid in unique_matches:
        mask = (match_ids == mid)
        if mask.sum() < 3:
            continue

        preds = predicted_risks[mask]
        ranks = final_ranks[mask]

        if np.std(preds) < 1e-8 or np.std(ranks) < 1e-8:
            continue

        sp = scipy_stats.spearmanr(preds, ranks)
        kt = scipy_stats.kendalltau(preds, ranks)

        rho = float(sp.statistic) if hasattr(sp, 'statistic') else float(sp[0])
        tau = float(kt.statistic) if hasattr(kt, 'statistic') else float(kt[0])

        spearman_list.append(rho)
        kendall_list.append(tau)

    return {
        "spearman_rho": float(np.mean(spearman_list)) if spearman_list else 0.0,
        "kendall_tau": float(np.mean(kendall_list)) if kendall_list else 0.0,
        "n_matches": len(spearman_list),
    }


# ============================================================
# 4. 통합 평가 함수
# ============================================================

def evaluate_survival_model(
    hazard_logits,    # np.array [N] — sigmoid 전 logit
    events,           # np.array [N] — 0/1
    times,            # np.array [N] — elapsed seconds
    phases,           # np.array [N] — zone phase
    risk_scores,      # np.array [N] — ranking용 risk
    final_ranks,      # np.array [N] — 최종 순위
    match_ids=None,   # np.array [N] — 매치 식별자
    train_events=None,
    train_times=None,
):
    """
    전체 평가 메트릭 한번에 계산.

    Returns:
        dict with all metrics organized by category
    """
    # hazard probability
    hazard_probs = 1.0 / (1.0 + np.exp(-hazard_logits))

    results = {}

    # ── Primary: IPCW C-index ──
    results["concordance"] = compute_concordance_index(hazard_probs, events, times)

    if train_events is not None and train_times is not None:
        results["ipcw"] = compute_ipcw_concordance(
            hazard_probs, events, times, train_events, train_times
        )
    else:
        results["ipcw"] = compute_ipcw_concordance(hazard_probs, events, times)

    # ── Dynamic AUC ──
    results["dynamic_auc"] = compute_dynamic_auc(
        hazard_probs, events, times, train_events=train_events, train_times=train_times
    )

    # ── Phase-wise ──
    results["phase_calibration"] = phase_wise_calibration(hazard_probs, events, phases)
    results["phase_concordance"] = phase_wise_concordance(hazard_probs, events, times, phases)

    # ── Rank correlation ──
    results["rank_correlation"] = rank_correlation(risk_scores, final_ranks, match_ids)

    # ── Phase-matched AUC (신규 primary) ──
    results["phase_auc"] = phase_matched_auc(hazard_probs, events, phases)

    # ── Per-match hazard rank (team-level aggregation) ──
    results["team_rank_correlation"] = per_match_hazard_rank(
        hazard_probs, events, times, final_ranks, match_ids
    )

    # ── Summary ──
    results["summary"] = {
        "c_index": results["concordance"]["c_index"],
        "ipcw_c_index": results["ipcw"].get("ipcw_c_index", results["concordance"]["c_index"]),
        "mean_auc": results["dynamic_auc"]["mean_auc"],
        "spearman_rho": results["rank_correlation"]["spearman_rho"],
        "phase_auc": results["phase_auc"].get("mean_auc", 0.5),
        "team_rank_rho": results["team_rank_correlation"]["spearman_rho"],
        "n_samples": len(events),
        "n_events": int(events.sum()),
        "event_rate": float(events.mean()),
    }

    return results


def format_eval_results(results, prefix=""):
    """평가 결과를 읽기 쉬운 문자열로 포맷."""
    lines = []
    s = results.get("summary", {})

    lines.append(f"{prefix}=== Survival Model Evaluation ===")
    lines.append(f"{prefix}  C-index:      {s.get('c_index', 0):.4f}")
    lines.append(f"{prefix}  IPCW C-index: {s.get('ipcw_c_index', 0):.4f}")
    lines.append(f"{prefix}  Dynamic AUC:  {s.get('mean_auc', 0):.4f}")
    lines.append(f"{prefix}  Spearman ρ:   {s.get('spearman_rho', 0):.4f}")
    lines.append(f"{prefix}  Phase AUC:    {s.get('phase_auc', 0):.4f}")
    lines.append(f"{prefix}  Team Rank ρ:  {s.get('team_rank_rho', 0):.4f}")
    lines.append(f"{prefix}  Samples:      {s.get('n_samples', 0)}, "
                 f"Events: {s.get('n_events', 0)} "
                 f"({100*s.get('event_rate', 0):.1f}%)")

    # Phase-wise
    pc = results.get("phase_calibration", {})
    if pc:
        lines.append(f"{prefix}  Phase Calibration:")
        for phase, v in sorted(pc.items()):
            lines.append(f"{prefix}    Phase {phase}: pred={v['predicted']:.3f}, "
                        f"obs={v['observed']:.3f}, "
                        f"err={v['calibration_error']:.3f} (n={v['n_samples']})")

    pci = results.get("phase_concordance", {})
    if pci:
        lines.append(f"{prefix}  Phase C-index:")
        for phase, ci in sorted(pci.items()):
            lines.append(f"{prefix}    Phase {phase}: {ci:.4f}")

    pa = results.get("phase_auc", {})
    if pa:
        lines.append(f"{prefix}  Phase-matched AUC:")
        for phase, auc in sorted((k, v) for k, v in pa.items() if isinstance(k, int)):
            lines.append(f"{prefix}    Phase {phase}: {auc:.4f}")

    return "\n".join(lines)


# ============================================================
# 6. 스냅샷 레벨 메트릭
# ============================================================

def snapshot_hit_rate(all_hazard_logits, all_dying_teams, k=1):
    """
    Top-k hit rate: hazard 상위 k개 팀 안에 실제 탈락 팀이 포함되는 비율.
    """
    import torch

    hits = 0
    total = 0

    for logits, dying in zip(all_hazard_logits, all_dying_teams):
        if len(dying) == 0 or len(logits) == 0:
            continue
        topk_indices = torch.topk(
            logits, k=min(k, len(logits))
        ).indices.tolist()

        for dt in dying:
            total += 1
            if dt in topk_indices:
                hits += 1

    return {
        "hit_rate": hits / max(total, 1),
        "n_events": total,
    }


def snapshot_mrr(all_hazard_logits, all_dying_teams):
    """
    Mean Reciprocal Rank: 탈락 팀의 hazard 순위의 역수 평균.
    MRR=1.0이면 매번 탈락 팀이 가장 높은 hazard.
    """
    import torch

    rr_sum = 0.0
    total = 0

    for logits, dying in zip(all_hazard_logits, all_dying_teams):
        if len(dying) == 0 or len(logits) == 0:
            continue
        sorted_indices = torch.argsort(logits, descending=True).tolist()

        for dt in dying:
            rank = sorted_indices.index(dt) + 1
            rr_sum += 1.0 / rank
            total += 1

    return {
        "mrr": rr_sum / max(total, 1),
        "n_events": total,
    }


def snapshot_rank_correlation(all_hazard_logits, all_metas):
    """
    스냅샷별 alive 팀 hazard rank vs final_rank Spearman 상관.
    """
    rho_list = []

    for logits, meta in zip(all_hazard_logits, all_metas):
        ranks = meta.get("team_ranks", [])
        if len(ranks) < 3 or len(logits) < 3:
            continue

        haz = logits.cpu().numpy()
        final = np.array(ranks, dtype=float)

        if np.std(haz) < 1e-8:
            continue

        sp = scipy_stats.spearmanr(haz, final)
        rho = float(sp.statistic) if hasattr(sp, 'statistic') else float(sp[0])
        rho_list.append(rho)

    return {
        "spearman_rho": float(np.mean(rho_list)) if rho_list else 0.0,
        "n_snapshots": len(rho_list),
    }


# ============================================================
# 7. Dual-Axis 메트릭 (Survival + Winner Prediction)
# ============================================================

def snapshot_brier_score(all_hazard_logits, all_dying_teams):
    """
    스냅샷별 Brier Score: sigmoid(hazard)와 실제 탈락 여부의 MSE.
    calibration 품질 측정. 낮을수록 좋음.
    """
    import torch

    brier_sum = 0.0
    n_total = 0

    for logits, dying in zip(all_hazard_logits, all_dying_teams):
        n_alive = len(logits)
        if n_alive == 0:
            continue

        probs = torch.sigmoid(logits).cpu().numpy()
        labels = np.zeros(n_alive)
        for idx in dying:
            labels[idx] = 1.0

        brier_sum += np.sum((probs - labels) ** 2)
        n_total += n_alive

    return brier_sum / max(n_total, 1)


def snapshot_calibration_error(all_hazard_logits, all_dying_teams, n_bins=10):
    """
    Expected Calibration Error (ECE).
    sigmoid(hazard)를 n_bins개 구간으로 나눠 예측 확률 vs 실제 비율 비교.
    """
    import torch

    all_probs = []
    all_labels = []

    for logits, dying in zip(all_hazard_logits, all_dying_teams):
        n_alive = len(logits)
        if n_alive == 0:
            continue
        probs = torch.sigmoid(logits).cpu().numpy()
        labels = np.zeros(n_alive)
        for idx in dying:
            labels[idx] = 1.0
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())

    if len(all_probs) == 0:
        return {"ece": 0.0, "bins": {}}

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bins_detail = {}

    for i in range(n_bins):
        mask = (all_probs >= bin_edges[i]) & (all_probs < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_pred = all_probs[mask].mean()
        bin_obs = all_labels[mask].mean()
        bin_n = int(mask.sum())
        ece += (abs(bin_pred - bin_obs) * bin_n) / len(all_probs)
        bins_detail[i] = {
            "predicted": float(bin_pred),
            "observed": float(bin_obs),
            "n_samples": bin_n,
        }

    return {"ece": float(ece), "bins": bins_detail}


def snapshot_td_c_index(all_hazard_logits, all_dying_teams):
    """
    Time-dependent C-index (스냅샷 레벨).
    각 스냅샷에서 dying 팀이 surviving 팀보다 높은 hazard를 가지는 비율.
    """
    concordant = 0
    discordant = 0
    tied = 0

    for logits, dying in zip(all_hazard_logits, all_dying_teams):
        n_alive = len(logits)
        if n_alive == 0 or len(dying) == 0:
            continue

        dying_set = set(dying)
        surviving = [i for i in range(n_alive) if i not in dying_set]

        if len(surviving) == 0:
            continue

        for d_idx in dying:
            h_d = logits[d_idx].item()
            for s_idx in surviving:
                h_s = logits[s_idx].item()
                if h_d > h_s:
                    concordant += 1
                elif h_d < h_s:
                    discordant += 1
                else:
                    tied += 1

    total = concordant + discordant + tied
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied) / total


def snapshot_topk_precision(all_hazard_logits, all_metas, k=3):
    """
    상위권 예측 정밀도.
    hazard 하위 k팀 (= 가장 안전한 팀)이 실제 최종 상위 k등 안에 포함되는 비율.
    """
    import torch

    hits = 0
    total = 0

    for logits, meta in zip(all_hazard_logits, all_metas):
        ranks = meta.get("team_ranks", [])
        n_alive = len(logits)

        if n_alive < k + 1 or len(ranks) != n_alive:
            continue

        # hazard 가장 낮은 k팀 (= 상위권 예측)
        topk_safe = torch.topk(logits, k=min(k, n_alive), largest=False).indices.tolist()

        # 실제 상위 k등 (rank 값이 가장 낮은 k팀)
        rank_arr = np.array(ranks)
        actual_topk = set(np.argsort(rank_arr)[:k].tolist())

        for idx in topk_safe:
            total += 1
            if idx in actual_topk:
                hits += 1

    return hits / max(total, 1)


def snapshot_winner_hit_rate(all_hazard_logits, all_metas):
    """
    치킨디너 예측 적중률.
    hazard 최저 팀이 실제 1위(rank=1)인 비율.
    """
    import torch

    hits = 0
    total = 0

    for logits, meta in zip(all_hazard_logits, all_metas):
        ranks = meta.get("team_ranks", [])
        n_alive = len(logits)

        if n_alive < 2 or len(ranks) != n_alive:
            continue

        # hazard 최저 팀
        predicted_winner = torch.argmin(logits).item()

        # 실제 1위 (rank 값이 가장 낮은 팀)
        actual_winner = int(np.argmin(ranks))

        total += 1
        if predicted_winner == actual_winner:
            hits += 1

    return hits / max(total, 1)


def evaluate_snapshot_model(all_hazard_logits, all_dying_teams, all_metas):
    """스냅샷 레벨 종합 평가. (dual-axis 확장)"""
    results = {}

    # ── Ranking 축 (기존) ──
    results["hit_1"] = snapshot_hit_rate(all_hazard_logits, all_dying_teams, k=1)
    results["hit_3"] = snapshot_hit_rate(all_hazard_logits, all_dying_teams, k=3)
    results["hit_5"] = snapshot_hit_rate(all_hazard_logits, all_dying_teams, k=5)
    results["mrr"] = snapshot_mrr(all_hazard_logits, all_dying_teams)
    results["rank_corr"] = snapshot_rank_correlation(all_hazard_logits, all_metas)

    # ── Survival 축 (신규) ──
    results["td_c_index"] = snapshot_td_c_index(all_hazard_logits, all_dying_teams)
    results["brier_score"] = snapshot_brier_score(all_hazard_logits, all_dying_teams)
    results["calibration"] = snapshot_calibration_error(all_hazard_logits, all_dying_teams)

    # ── Winner 축 (신규) ──
    results["topk_precision_3"] = snapshot_topk_precision(all_hazard_logits, all_metas, k=3)
    results["topk_precision_5"] = snapshot_topk_precision(all_hazard_logits, all_metas, k=5)
    results["winner_hit"] = snapshot_winner_hit_rate(all_hazard_logits, all_metas)

    results["summary"] = {
        # Ranking
        "hit_1": results["hit_1"]["hit_rate"],
        "hit_3": results["hit_3"]["hit_rate"],
        "hit_5": results["hit_5"]["hit_rate"],
        "mrr": results["mrr"]["mrr"],
        "spearman_rho": results["rank_corr"]["spearman_rho"],
        "n_events": results["mrr"]["n_events"],
        # Survival
        "td_c_index": results["td_c_index"],
        "brier_score": results["brier_score"],
        "ece": results["calibration"]["ece"],
        # Winner
        "topk_precision_3": results["topk_precision_3"],
        "topk_precision_5": results["topk_precision_5"],
        "winner_hit": results["winner_hit"],
    }

    return results


def format_snapshot_eval(results, prefix=""):
    """스냅샷 평가 결과 포맷. (dual-axis 확장)"""
    s = results.get("summary", {})
    lines = [
        f"{prefix}=== Snapshot Model Evaluation ===",
        f"{prefix}  [Ranking]",
        f"{prefix}    Hit@1:      {s.get('hit_1', 0):.4f}",
        f"{prefix}    Hit@3:      {s.get('hit_3', 0):.4f}",
        f"{prefix}    Hit@5:      {s.get('hit_5', 0):.4f}",
        f"{prefix}    MRR:        {s.get('mrr', 0):.4f}",
        f"{prefix}    Spearman ρ: {s.get('spearman_rho', 0):.4f}",
        f"{prefix}  [Survival]",
        f"{prefix}    TD C-index: {s.get('td_c_index', 0):.4f}",
        f"{prefix}    Brier:      {s.get('brier_score', 0):.4f}",
        f"{prefix}    ECE:        {s.get('ece', 0):.4f}",
        f"{prefix}  [Winner]",
        f"{prefix}    Top3 Prec:  {s.get('topk_precision_3', 0):.4f}",
        f"{prefix}    Top5 Prec:  {s.get('topk_precision_5', 0):.4f}",
        f"{prefix}    Winner Hit: {s.get('winner_hit', 0):.4f}",
        f"{prefix}  Events: {s.get('n_events', 0)}",
    ]
    return "\n".join(lines)
