#!/usr/bin/env python3
"""Summarize completed, immutable v14 analysis outputs for the paper handoff."""
import json
from pathlib import Path
import pandas as pd
REPO=Path(__file__).resolve().parents[1]
BASE=REPO/'reports/sac2027_patch_v14'
PAPER=REPO/'paper/sac2027/v14_review'


def main():
    assert (BASE/'final_verification.json').exists()
    rows=[]
    for cohort in ('erangel_tpp_squad','taego_tpp_squad'):
        for arch in ('gbdt','krafton','ptr','gatv2'):
            root=BASE/'tuning'/cohort/arch
            selected=json.loads((root/'selection.json').read_text())
            trials=[json.loads((root/f'trial{i:02d}/trial.json').read_text()) for i in range(16)]
            chosen=trials[selected['chosen_trial']]
            rows.append(dict(cohort=cohort,architecture=arch,chosen_trial=selected['chosen_trial'],
                validation_nll=selected['val_nll'],n_parameters=chosen.get('n_parameters'),
                search_train_seconds=sum(t.get('train_sec',0) for t in trials),
                chosen_train_seconds=chosen.get('train_sec'),setting=json.dumps(selected['setting'],sort_keys=True)))
    pd.DataFrame(rows).to_csv(BASE/'tuning/selected_resources.csv',index=False)
    lines=['# 패치 재학습 결과 확인표', '',
           '아래 수치는 완료된 분석 CSV에서 직접 생성했다. 각 행은 선택한 설정의 3시드 지표 평균이다. 전체/사람 팀 평가를 구분한다. 그래프·모집단 대조는 CPU에서 다시 학습한 신경망 참조를 사용하므로 원 튜닝 표와 신경망 수치가 조금 다를 수 있다.', '',
           '모든 신뢰구간은 경기 단위 paired bootstrap 2,000회로 계산한다. 학습 시드 모집단 전체의 불확실성을 추정하지 않는다. GBDT의 세 시드는 결정적인 적합으로 같은 예측을 낼 수 있다. 추가 대조는 탐색적이며 다중검정 보정된 확증적 결과로 읽지 않는다.', '']
    names={'tuning':'동일 시행 횟수 튜닝','objective':'손실과 예측 구간','history':'과거 요약과 순서','population':'AI 입력과 사람 팀 손실','graph':'GATv2와 관계 대응'}
    for suite,title in names.items():
        frame=pd.read_csv(BASE/'control_evaluation'/suite/'arm_metrics.csv')
        contrasts=pd.read_csv(BASE/'control_evaluation'/suite/'contrasts.csv')
        lines += ['## '+title,'','| 코호트 | 평가 대상 | 조건 | AUC | 단계 내 AUC | 경기 내 AUC | NLL |','|---|---|---|---:|---:|---:|---:|']
        for r in frame.itertuples():
            lines.append(f'| {r.cohort.split("_")[0]} | {r.population} | {r.arm} | {r.auc:.4f} | {r.auc_within_phase:.4f} | {r.auc_within_match_phase:.4f} | {r.nll:.4f} |')
        lines += ['','| 코호트 | 평가 대상 | 차이(앞−뒤) | 지표 | 추정치 [95% CI] |','|---|---|---|---|---:|']
        # Overall and within-match contrasts suffice here; complete CSVs retain all five metrics.
        for r in contrasts[contrasts.metric.isin(['auc','auc_within_match_phase'])].itertuples():
            lines.append(f'| {r.cohort.split("_")[0]} | {r.population} | {r.contrast} | {r.metric} | {r.estimate:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}] |')
        lines += ['']
    lines += ['## 선택된 설정과 학습 자원','',
              '검증 NLL로 설정을 선택했다. 실행시간은 동시 실행과 장치 차이의 영향을 받으므로 속도 우열을 판정하는 벤치마크가 아니다. 정확한 값과 설정은 `reports/sac2027_patch_v14/tuning/selected_resources.csv`에 있다.', '',
              '| 코호트 | 모델 | 선택 trial | 검증 NLL | 파라미터 | 16회 학습 누적 초 |','|---|---|---:|---:|---:|---:|']
    for r in rows:
        params='해당 없음' if r['n_parameters'] is None else str(r['n_parameters'])
        lines.append(f"| {r['cohort'].split('_')[0]} | {r['architecture']} | {r['chosen_trial']} | {r['validation_nll']:.5f} | {params} | {r['search_train_seconds']:.1f} |")
    lines += ['','주 분석·구조 비교는 `primary_results_v2/`, 전체/사람 팀 참조 비교는 `control_evaluation/main_population/`, 원 과제 지표는 `native_evaluation/`를 따른다. 그래프 선정과 원본 코드 검증은 `graph_model_selection/SELECTION_KO.md`, 선행연구 교정은 `baseline_audit/AUDIT_KO.md`에 기록했다.']
    (PAPER/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
