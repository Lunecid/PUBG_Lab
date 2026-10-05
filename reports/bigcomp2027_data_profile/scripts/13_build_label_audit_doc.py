#!/usr/bin/env python3
"""Build docs/data_layer/BIGCOMP2027_LABEL_AUDIT_2026-10-05.md from csv/label_audit_*.csv and logs (numbers are read from files)."""
import json, os, re, sys
import numpy as np, pandas as pd
ROOT = '/Users/lunecid/project/PUBG'; B = ROOT + '/reports/bigcomp2027_data_profile'; C = B + '/csv'; DOC = ROOT + '/docs/data_layer/BIGCOMP2027_LABEL_AUDIT_2026-10-05.md'
T1, T2 = 'full3', 't2c'; LOG2 = open(f'{B}/logs/11_label_audit_tier2.log').read(); LOG1 = open(f'{B}/logs/10_label_audit_tier1.log').read()
rd = lambda n, **k: pd.read_csv(f'{C}/{n}', **k)
L = []
def P(*s): L.extend(s)
def tab(h, rows, left=1, align=None):
    al = ['---' if c == 'l' else '---:' for c in align] if align else ['---'] * left + ['---:'] * (len(h) - left)
    P('| ' + ' | '.join(h) + ' |', '| ' + ' | '.join(al) + ' |')
    for r in rows: P('| ' + ' | '.join(str(x) for x in r) + ' |')
    P('')
fi = lambda x: f"{int(round(x)):,}"; ff = lambda x, d=1: f"{x:,.{d}f}"; fp = lambda x, d=1: f"{100 * x:.{d}f}%"
cs = rd('cohort_summary.csv').set_index('slug').sort_values('n_total', ascending=False); order = list(cs.index); LAB = {s: f"{cs.loc[s].map_label} {cs.loc[s].perspective} {cs.loc[s].team_size}" for s in order}
cnt = rd(f'label_audit_{T1}_counts.csv'); mc = rd(f'label_audit_{T1}_match_clock.csv'); fl = rd(f'label_audit_{T1}_flag_counts.csv'); tc = rd(f'label_audit_{T1}_team_T_agreement.csv').set_index('ai_bucket'); po = json.load(open(f'{C}/label_audit_{T1}_player_offsets.json'))
pout = rd(f'label_audit_{T1}_player_outliers.csv'); rdp = rd(f'label_audit_{T1}_redeploy_by_cohort.csv').set_index('cohort'); dis = rd(f'label_audit_{T1}_disagreements.csv.gz'); t2 = rd(f'label_audit_{T2}_rows.csv.gz')
def met(df, tr='tr_evt'):
    S = lambda m: int(df.loc[m, 'n'].sum()); n = int(df.n.sum()); pr, tk = S(df.pst >= 1), S(df[tr] >= 1); pp, tp, bp = S(df.pst == 2), S(df[tr] == 2), S((df.pst == 2) & (df[tr] == 2)); ag = S(df.pst == df[tr])
    return pd.Series(dict(rows=n, agree=ag / n, changed=n - ag, pipe_risk=pr, truth_risk=tk, risk_fp=S((df.pst >= 1) & (df[tr] == 0)), risk_fn=S((df.pst == 0) & (df[tr] >= 1)), pipe_pos=pp, truth_pos=tp, both_pos=bp, label_fp=S((df.pst == 2) & (df[tr] == 1)), label_fn=S((df.pst == 1) & (df[tr] == 2)), precision=bp / pp if pp else np.nan, recall=bp / tp if tp else np.nan, hazard_pipe=pp / pr if pr else np.nan, hazard_truth=tp / tk if tk else np.nan))
allm = met(cnt); byk = cnt.groupby('k').apply(met); byc = cnt.groupby('cohort').apply(met).loc[order]; bya = cnt.groupby('ai_bucket').apply(met); bym = cnt.groupby('match_type').apply(met); byt = cnt.groupby('term_n').apply(met); byck = cnt.groupby(['cohort', 'k']).apply(met).reset_index()
hum = cnt[cnt.ai_bucket == 'no_ai']; hum_off = hum.groupby('k').apply(lambda d: met(d, 'tr_off')); hum_agree = float(hum[hum.tr_evt == hum.tr_off].n.sum() / hum.n.sum()); ai_agree = float(cnt[(cnt.ai_bucket == 'all_ai') & (cnt.tr_evt == cnt.tr_off)].n.sum() / cnt[cnt.ai_bucket == 'all_ai'].n.sum())
strict = cnt.groupby('k').apply(lambda d: met(d, 'tr_evt_strict')); n_changed_strict = int(cnt[cnt.tr_evt != cnt.tr_evt_strict].n.sum()); nrows = int(cnt.n.sum())
conf = cnt.groupby(['pst', 'tr_evt']).n.sum().unstack(fill_value=0).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0); S_ = lambda x: int(x)
dm = t2.drop_duplicates('match_id').dpos; leak = float(np.clip((dm - 5) / 10, 0, 1).mean()); mcs = mc.delta_gs; ots = mc.off_ts.dropna(); nm = int(len(mc)); n_hum_match = int(len(ots))
def g(pat, txt=LOG2, cast=float, default=np.nan):
    m = re.search(pat, txt); return cast(m.group(1)) if m else default
# ---- tier-2 numbers parsed from the log ----
adj = re.search(r'ADJUDICATION human teams with \|T_evt-T_off\|>5 s: (\d+) \| T_evt >= last position sample: (\d+) \| T_off >= last position sample: (\d+) \| both: (\d+) \| neither: (\d+) \| redeploy teams among them: (\d+)', LOG2)
alt = re.search(r'ALL teams with finite T_evt: (\d+) \| T_evt < last position sample - 2s: (\d+) \| of which redeploy: (\d+) \| T_off < last sample - 2s \(human teams\): (\d+)', LOG2)
t2n = re.search(r'rows (\d+) matches (\d+) dpos q05/50/95/max: \{0.05: ([\d.\-]+), 0.5: ([\d.\-]+), 0.95: ([\d.\-]+), 1.0: ([\d.\-]+)\} share dpos>10: ([\d.]+)', LOG2)
rep = re.search(r'replication: pst_re == table pst: ([\d.]+) \| nh_k == table ape_k \(rows with state\): ([\d.]+)', LOG2)
ag3 = {k: re.search(rf'^{k} vs truth_events agree: ([\d.]+) \| risk-set FP/FN: (\d+) (\d+) \| label FP/FN: (\d+) (\d+)', LOG2, re.M) for k in ('pst', 'pst_re', 'pst_fx')}
kn = re.search(r'KNOCKED-snapshot rows (\d+) \| teammate healthy\+sampled at boundary \(n_healthy>=1\): (\d+) \| all alive members knocked \(healthy==0 & knock>=1\): (\d+) \| none sampled: (\d+) \| revive evidence \(knocked member later healthy\): (\d+)', LOG2)
un = re.search(r'UNOBSERVED rows (\d+) \| team rows within 30 s before physical boundary == 0: (\d+) \| within 15 s == 0: (\d+)', LOG2)
nr = re.search(r'disagreeing rows with NO team position rows in the 30 s before the physical boundary: (\d+) of (\d+) \| agreeing rows: ([\d.]+) \| disagree share: ([\d.]+)', LOG2)
fx = re.search(r'events-disagreeing rows (\d+) \| fixed by clock-fix: (\d+)', LOG2); fxt = re.findall(r'^(?:(label_FN|label_FP|risk_FN|risk_FP)\s+)?(\w+)\s+(\d+)\s+(\d+)\s*$', LOG2, re.M)
# ---- exclusive categories (full corpus) ----
cat = dis.groupby(['typ', 'cat']).size().unstack(fill_value=0).reindex(index=['risk_FN', 'risk_FP', 'label_FP', 'label_FN'], columns=['knocked_last_snapshot', 'stale_last_snapshot', 'revive_redeploy', 'near_boundary_15s', 'unobserved_no_rows', 'afk_logout', 'other'], fill_value=0); ndis = int(len(dis))
fa = fl.groupby('agree').n.sum()
CATN = {'all_ai_team': '전원 AI 팀(원인 무관)', 'knocked_last_snapshot': '마지막 스냅샷이 기절(체력 0)뿐', 'stale_last_snapshot': '마지막 스냅샷이 사망 이전(생존으로 읽음)', 'revive_redeploy': '소생·재투입 팀원(킬 이벤트 2회 이상)', 'near_boundary_15s': '사망이 경계 15초 이내', 'unobserved_no_rows': '해당 단계 위치 행 없음', 'afk_logout': '이탈(logout) 팀원', 'other': '기타'}
TYPN = {'risk_FN': '위험집합 누락: 파이프라인은 사망으로 읽었으나 정답은 생존', 'risk_FP': '위험집합 오포함: 파이프라인은 생존으로 읽었으나 정답은 이미 사망', 'label_FP': '라벨 양성 오류: 파이프라인 양성, 정답 음성', 'label_FN': '라벨 음성 오류: 파이프라인 음성, 정답 양성'}
# ---- document ----
P('# BigComp 2027 hazard 라벨 감사 (2026-10-05)', '')
P('- 목적: 학습에 쓰이는 `y_hazard`/`target_mask`가 실제 팀 탈락 시각과 맞는지 검증한다(Part 1b).',
  '- DB는 읽기 전용 SELECT만 썼다(`default_transaction_read_only=on`, `statement_timeout=120s`). 단계 경계는 Fix B 재작성 전에 만든 사본 `stdb_bak_20261005.match_phase_global`에서 읽었다(라이브 표는 재작성 중이지만 `phase_start_s`·`phase_end_s`가 사본과 다른 행은 0건, `logs/00_stdb_schema_check.log`). `team_phase_labels`·`team_phase_state`는 사본이 없고, `stdb`의 현재 행(5,419,004행, 21,830매치, `max(created_at)=2026-06-09`)에 재작성 흔적이 없어 그대로 읽었다(`logs/` 참조).',
  f'- 범위: 13개 코호트 전수 {nm:,}매치, 라벨 행 {nrows:,}개(컷 k=1~7 중 다음 단계가 있는 행). 위치 단위 진단(4절의 재현과 표 9)은 코호트당 250매치 층화 표본 3,250매치다.',
  '- 스크립트: `reports/bigcomp2027_data_profile/scripts/09_clock_probe.py`, `label_audit_lib.py`, `10_label_audit_tier1.py`, `11_label_audit_tier2.py`, `12_label_audit_tables.py`, `13_build_label_audit_doc.py`. 원자료: `reports/bigcomp2027_data_profile/csv/label_audit_*`. 같은 폴더의 1차 문서(`BIGCOMP2027_DATA_PROFILE_2026-10-05.md` 6.5절)의 감사는 첫 킬 이벤트를 사망 시각으로 써서 소생 팀을 잘못 셌으므로 이 문서가 대체한다.', '')
P('## 0. 결론', '')
P(f'1. 라벨 출처: `stdb.team_phase_labels.hazard_next`(=`y_hazard`)와 `target_defined`(=`target_mask`)를 `stdb/queries.py::_fetch_phase_targets`가 그대로 읽는다. 두 열은 `stdb/rq1_feature_etl.py::build_team_phase_labels`가 `team_phase_state.alive_players_end>0`(팀의 마지막 (팀, 초) 스냅샷에서 체력>0인 관측 선수 수)로 만든다.',
  f'2. 시계: `phase_end_s`는 숫자상 시작 시계(`event_time − telem_match_start.event_time`)다. 단계 경계는 이 시계로, 위치 창(`team_phase_state`)은 `elapsed_time` 시계로 만들어져 위치 창이 경계보다 늦게 끝난다. 표본 매치의 위치 시계 차이 δ_pos는 중앙값 {dm.median():.1f}초, p95 {dm.quantile(.95):.1f}초, 최대 {dm.max():.1f}초이고(전체 매치의 게임 상태 기준 δ는 중앙값 {mcs.median():.1f}초, p95 {mcs.quantile(.95):.1f}초, 최대 {mcs.max():.1f}초), `time_survived`는 시작 시계이며 킬 이벤트와의 차이는 사람 {fp(po["human"]["share_abs_le2"])}가 2초 이내(중앙값 {po["human"]["q"]["0.5"]:.2f}초)다.',
  f'3. 정답: 팀 탈락 시각 = 팀원 마지막 킬 이벤트의 최댓값(생존자가 있으면 검열), 위험집합 = 탈락 시각 > `phase_end_s`(k), 라벨 = 탈락 시각 <= `phase_end_s`(k+1)이며 k+1이 매치의 마지막 단계면 매치 종료까지로 본다.',
  f'4. 불일치: {nrows:,}행 중 {fi(allm.changed)}행({fp(allm.changed / nrows, 2)})에서 파이프라인과 정답의 3상태(위험집합 밖 / 음성 / 양성)가 다르다. 학습 행({fi(allm.pipe_risk)}) 가운데 위험집합 오포함 {fi(allm.risk_fp)}행({fp(allm.risk_fp / allm.pipe_risk)}), 라벨이 틀린 행 {fi(allm.label_fp + allm.label_fn)}행({fp((allm.label_fp + allm.label_fn) / allm.pipe_risk)})이고, 정답 위험집합에서 빠진 행이 {fi(allm.risk_fn)}행이다. 양성 정밀도 {fp(allm.precision)}, 재현율 {fp(allm.recall)}.',
  f'5. hazard 곡선이 다르다. k=1~7에서 파이프라인 {", ".join(f"{byk.loc[k].hazard_pipe:.3f}" for k in range(1, 8))}, 정답 {", ".join(f"{byk.loc[k].hazard_truth:.3f}" for k in range(1, 8))}. 후반(k>=6)은 정답이 훨씬 높다. 마지막 단계의 탈락을 파이프라인이 놓친다(종결 단계 행 양성 재현율 {fp(byt.loc[True].recall)}).',
  f'6. 원인(상호배타, 불일치 {fi(ndis)}행): 마지막 스냅샷이 기절뿐 {fp(cat["knocked_last_snapshot"].sum() / ndis)}, 마지막 스냅샷이 사망 이전 {fp(cat["stale_last_snapshot"].sum() / ndis)}, 소생·재투입 {fp(cat["revive_redeploy"].sum() / ndis)}, 경계 15초 이내 {fp(cat["near_boundary_15s"].sum() / ndis)}, 위치 행 없음 {fp(cat["unobserved_no_rows"].sum() / ndis)}, 이탈 {fp(cat["afk_logout"].sum() / ndis)}.',
  f'7. 판정: 라벨은 실질적으로 틀렸다(행의 {fp(allm.changed / nrows)}, 양성의 약 {fp(1 - allm.precision, 0)}가 거짓, 후반 hazard 크게 과소). 교정 라벨(5절)로 바꾸면 {fi(allm.changed)}행이 바뀐다. 사람 팀만 보면 일치율 {fp(bya.loc["no_ai"].agree)}, AI 팀은 {fp(bya.loc["all_ai"].agree)}다.',
  f'8. 별도 위험: 위치 시계와 경계 시계의 차이 δ_pos는 표본 매치의 {fp((dm > 5).mean())}에서 5초, {fp((dm > 10).mean())}에서 10초를 넘는다(전체 매치의 δ 기준 10초 초과 {fp((mcs > 10).mean())}). Fix B의 `phase_end_s − 10` 보호 구간은 격자 읽기 창(±5초)만 가정한다. 격자 내림 오차를 균등 분포로 두면 컷의 마지막 위치 샘플이 실제 단계 경계 뒤에 있을 매치는 평균 {fp(leak)}로 추정된다(6절, 직접 측정은 아님).', '')
P('## 1. 라벨 출처와 시계', '')
P('표 1. 코드 경로. 줄 번호는 2026-10-05 작업 트리 기준이고, 함수 이름이 기준이다.', '')
tab(['단계', '위치', '내용'], [
    ['학습 입력', '`stdb/queries.py::_fetch_phase_targets`(696~726행), 호출 909행', '`y_hazard[cut_phase−1, 팀] = hazard_next`, `target_mask = target_defined`(NULL이면 0). 컷 1~7, `PHASE_T_MAX` 미만만 읽는다'],
    ['라벨 표', '`stdb.team_phase_labels`', '`target_defined`, `hazard_next`, `eliminated_next_phase`, `alive_at_phase_end`'],
    ['라벨 생성', '`stdb/rq1_feature_etl.py::build_team_phase_labels`', '`target_defined = alive_players_end(k)>0 AND next_phase_idx IS NOT NULL`, `hazard_next = alive_end(k)>0 AND NOT alive_end(k+1)>0`(행이 없으면 0)'],
    ['생존 인원', '`build_team_phase_state`의 `snap`·`start_snap`·`end_snap` CTE', '(팀, 경과 초)별 `count(DISTINCT account_id) FILTER (health>0)`, 단계 안 마지막 스냅샷 한 곳의 값을 `alive_players_end`로 쓴다'],
    ['단계 경계', '`stdb.v_match_phase_bounds`(`schema_rq1.sql`)', '`phase_start_s=min(snapshot_t)`, `phase_end_s=max(snapshot_t)` (`stdb.dynamic_regions`, `phase_idx>0`)'],
    ['격자 시계', '`stdb/etl.py::_ingest_dynamic_regions`', '`snapshot_t = floor(telem_game_states.elapsed_time / 10) × 10` (경과 시간 시계)'],
    ['단계 태깅 시계', '같은 함수의 `telem_phase_changes` 경계', '`coalesce(elapsed_time, event_time − telem_match_start.event_time)`. `elapsed_time`은 코호트 매치 249,427행 전부 NULL이므로 항상 시작 시계 값이다']], align='lll')
P('표 2. 시계 사이의 관계. 출처: `csv/label_audit_clock_probe_*.csv`(390매치, 09 스크립트), `csv/label_audit_full3_match_clock.csv`(전체), `csv/label_audit_full3_player_offsets.json`.', '')
tab(['측정', '값'], [
    [f'경과 시계 0점 − `LogMatchStart`(δ), 전체 {nm:,}매치', f'중앙값 {mcs.median():.2f}초, p5 {mcs.quantile(.05):.2f}, p95 {mcs.quantile(.95):.2f}, 최대 {mcs.max():.2f}. 5초 초과 {fp((mcs > 5).mean())}, 10초 초과 {fp((mcs > 10).mean())}, 15초 초과 {fp((mcs > 15).mean())}, 20초 초과 {fp((mcs > 20).mean())}'],
    ['`phase_start_s` − (단계 변경 시각 − `LogMatchStart`), 2,755 단계', '중앙값 −0.29초, p5 −0.81, p75 −0.09 (경계 숫자가 시작 시계)'],
    ['`phase_start_s` − (단계 변경 시각 − 경과 시계 0점)', '중앙값 +8.10초, p5 0.89, p95 23.26. δ와 상관 0.921'],
    ['킬 이벤트 시각(시작 시계) − `time_survived`, 사람 플레이어', f'중앙값 {po["human"]["q"]["0.5"]:.2f}초, p25 {po["human"]["q"]["0.25"]:.2f}, p75 {po["human"]["q"]["0.75"]:.2f}, 절대 차이 2초 이하 {fp(po["human"]["share_abs_le2"])} (n={po["human"]["n"]:,})'],
    [f'매치별 오프셋(사람 킬 중앙값), {n_hum_match:,}매치', f'중앙값 {ots.median():.2f}초, p5 {ots.quantile(.05):.2f}, p95 {ots.quantile(.95):.2f}, 매치 내 MAD 중앙값 {mc.mad_ts.median():.2f}초'],
    ['같은 차이, AI 플레이어', f'절대 차이 2초 이하 {fp(po["ai"]["share_abs_le2"])}, 중앙값 {po["ai"]["q"]["0.5"]:.1f}초(p1~p95 기준 `time_survived`가 사망보다 {-po["ai"]["q"]["0.95"]:.0f}~{-po["ai"]["q"]["0.01"]:.0f}초 큼, n={po["ai"]["n"]:,})']], align='ll')
P(f'결과: 숫자상 `phase_end_s`는 시작 시계의 단계 끝이다. 그러나 `team_phase_state`의 위치 창은 `elapsed_time BETWEEN phase_start_s AND phase_end_s`로 경과 시계를 쓰므로 단계 k의 "끝 상태"는 시작 시계로 `phase_end_s + δ_pos`까지 읽는다(`δ_pos`는 위치 시계 기준, 표본 매치 중앙값 {float(t2n.group(4)) if t2n else float("nan"):.2f}초, p95 {float(t2n.group(5)) if t2n else float("nan"):.2f}초, 10초 초과 {fp(float(t2n.group(7))) if t2n else "n/a"}). 킬 이벤트와 `time_survived`는 시작 시계라서 정답은 경계 숫자와 직접 비교한다.', '')
pdp = rd('label_audit_postdeath_positions.csv'); gap = pdp.gap_pos_minus_kill
P(f'현행 라벨이 대체로 맞는 이유와 틀리는 방식: 죽은 선수의 위치 행이 킬 이벤트 뒤에도 체력 0으로 최대 약 60초 더 기록된다(사람 단일 사망 선수 {len(pdp):,}명: 마지막 위치 행이 킬보다 2초 넘게 늦은 비율 {fp((gap > 2).mean())}, 10초 넘게 {fp((gap > 10).mean())}, 중앙값 {gap.median():.2f}초, p95 {gap.quantile(.95):.1f}초, 체력 0 행이 하나라도 있는 선수 {fp((pdp.n_zero > 0).mean())}; `csv/label_audit_postdeath_positions.csv`). 그래서 전멸한 팀의 마지막 (팀, 초) 스냅샷은 보통 체력 0 행뿐이라 `alive_players_end=0`이 사망으로 읽힌다. 반대로 즉사해 체력 0 행을 남기지 않은 팀은 마지막 스냅샷이 건강한 행이라 생존으로 읽히고, 살아 있는 팀도 그 (팀, 초)에 기절한 팀원의 행만 있으면 사망으로 읽힌다.', '')
# ---- 2. truth ----
P('## 2. 정답 구성', '')
P('- 이벤트 정답(주): 선수 사망 시각 = `telem_kills`에서 그 선수가 피해자인 마지막 이벤트 시각 − `stdb.matches.started_at`. 자기장·낙하·차량 사망도 같은 표에 있다(`killer_id`가 없는 행 포함). 생존 선수(`death_type=alive`)는 ∞. 킬 이벤트가 없는 사망·이탈 선수는 `time_survived` + 매치 오프셋. 팀 시각 = 팀원 시각의 최댓값. 첫 이벤트가 아니라 마지막 이벤트를 쓰는 이유는 소생·재투입된 팀원(킬 이벤트 2회 이상, 5초 넘게 떨어진 이벤트; `telem_item_use`에 `Item_Bluechip_C` 사용 기록이 있어 재투입 모드로 추정)이 두 번째 사망을 가지기 때문이다.',
  '- 공식 정답(보조): 팀 시각 = max(`participants.time_survived`) + 매치 오프셋, 우승팀(`rosters.won` 또는 rank 1)은 ∞.',
  '- 라벨: 위험집합 = 팀 시각 > `phase_end_s`(k)이고 다음 단계 존재, 양성 = 팀 시각 <= `phase_end_s`(k+1). k+1이 매치 마지막 단계이면 경계를 ∞(매치 종료)로 둔다. 마지막 단계의 `phase_end_s`는 마지막 격자점이라 마지막 킬이 그 뒤에 일어나기 때문이다.', '')
pm_ = float(tc.loc['no_ai', 'within5'] / tc.loc['no_ai', 'both_fin'])
P(f'검증: (1) 사람만의 팀 {fi(tc.loc["no_ai", "both_fin"])}개에서 공식 시각과 이벤트 시각이 5초 이내인 비율 {fp(pm_)}, 행 단위 3상태 일치율 {fp(hum_agree, 2)}. (2) 5초 넘게 다른 사람 팀은 표본에서 {fi(float(adj.group(1)))}팀이고 그중 소생 팀이 {fi(float(adj.group(6)))}팀({fp(float(adj.group(6)) / float(adj.group(1)))})이다. (3) AI 팀 {fi(tc.loc["all_ai", "n"])}개는 5초 이내가 0.0%(행 단위 {fp(ai_agree)})라서 공식 `time_survived`는 AI에 쓸 수 없다. (4) 소생 선수(킬 이벤트 2회)의 `time_survived`는 첫·마지막 킬 시각 어느 쪽과도 {fp(pout[(pout.b == "n_ev2") & ~pout.first_ok & ~pout.last_ok].n.sum() / pout[pout.b == "n_ev2"].n.sum())} 맞지 않는다(단일 사망 선수는 {fp(pout[pout.b == "n_ev1"].query("first_ok and last_ok").n.sum() / pout[pout.b == "n_ev1"].n.sum())}가 킬 시각과 일치). 원인은 확인하지 않았다. 죽은 선수의 위치 행이 사망 뒤에도 남으므로 마지막 위치 시각으로는 사망 시각을 검증할 수 없다. 그래서 킬 이벤트를 주 정답으로 쓰고, 공식 시각은 사람·비소생 팀의 보조 검증으로만 쓴다.', '')
P(f'소생·재투입 팀원이 있는 팀은 코호트 합계 {fi(rdp.teams_redeploy.sum())}팀({fp(rdp.teams_redeploy.sum() / rdp.teams.sum())})이고, 소생 선수가 있는 매치는 {fi((mc.n_multi > 0).sum())}개({fp((mc.n_multi > 0).mean())})다. 매치 유형별로는 competitive {fp((mc[mc.match_type == "competitive"].n_multi > 0).mean())}, official {fp((mc[mc.match_type == "official"].n_multi > 0).mean())}, airoyale {fp((mc[mc.match_type == "airoyale"].n_multi > 0).mean())}다. 이탈(logout) 팀원이 있는 팀은 {fi(rdp.teams_logout.sum())}팀뿐이다. 종결 단계 처리를 쓰지 않은 엄격 정의(마지막 격자점까지)와 비교하면 {fi(n_changed_strict)}행({fp(n_changed_strict / nrows, 2)})이 달라지지만 위험집합이 작은 후반 컷의 hazard는 크게 달라진다: 엄격 정의의 정답 hazard는 k=1~7에서 ' + ', '.join(f'{strict.loc[k].hazard_truth:.3f}' for k in range(1, 8)) + f'이고 이 문서의 정의(매치 종료까지)는 ' + ', '.join(f'{byk.loc[k].hazard_truth:.3f}' for k in range(1, 8)) + '이다. 마지막 단계의 탈락은 마지막 격자점 뒤에 일어나므로 매치 종료까지로 보는 쪽이 맞고, k>=5의 결론은 이 정의에 민감하다.', '')
# ---- 3. confusion ----
P('## 3. 혼동행렬', '')
P(f'표 3. 전체 3상태 혼동행렬(행: 파이프라인, 열: 이벤트 정답). 0=위험집합 밖(`target_defined` 거짓 / 이미 사망), 1=위험집합·음성, 2=위험집합·양성. 출처: `csv/label_audit_{T1}_counts.csv`.', '')
tab(['파이프라인 \\ 정답', '0 밖', '1 음성', '2 양성', '합계'], [[n_, *[fi(conf.loc[i, j]) for j in (0, 1, 2)], fi(conf.loc[i].sum())] for i, n_ in ((0, '0 밖'), (1, '1 음성'), (2, '2 양성'))] + [['합계', *[fi(conf[j].sum()) for j in (0, 1, 2)], fi(conf.values.sum())]])
P(f'일치 {fp(allm.agree, 2)}. 위험집합: 파이프라인 {fi(allm.pipe_risk)}행, 정답 {fi(allm.truth_risk)}행, 오포함 {fi(allm.risk_fp)}, 누락 {fi(allm.risk_fn)}. 양성: 파이프라인 {fi(allm.pipe_pos)}, 정답 {fi(allm.truth_pos)}, 교집합 {fi(allm.both_pos)} (정밀도 {fp(allm.precision)}, 재현율 {fp(allm.recall)}). 위험집합이 같은 행에서 라벨만 다른 행은 양성 오류 {fi(allm.label_fp)}, 음성 오류 {fi(allm.label_fn)}.', '')
P('표 4. 컷 k별. 위험집합 오포함/누락, 라벨 오류, hazard(양성/위험집합). 출처: 같은 파일.', '')
tab(['k', '행', '일치', '위험집합 파이프라인', '위험집합 정답', '오포함', '누락', '양성 오류', '음성 오류', '정밀도', '재현율', 'hazard 파이프라인', 'hazard 정답'], [[str(k), fi(byk.loc[k].rows), fp(byk.loc[k].agree), fi(byk.loc[k].pipe_risk), fi(byk.loc[k].truth_risk), fi(byk.loc[k].risk_fp), fi(byk.loc[k].risk_fn), fi(byk.loc[k].label_fp), fi(byk.loc[k].label_fn), fp(byk.loc[k].precision), fp(byk.loc[k].recall), f"{byk.loc[k].hazard_pipe:.4f}", f"{byk.loc[k].hazard_truth:.4f}"] for k in range(1, 8)] + [['전체', fi(allm.rows), fp(allm.agree), fi(allm.pipe_risk), fi(allm.truth_risk), fi(allm.risk_fp), fi(allm.risk_fn), fi(allm.label_fp), fi(allm.label_fn), fp(allm.precision), fp(allm.recall), f"{allm.hazard_pipe:.4f}", f"{allm.hazard_truth:.4f}"]])
P('표 5. 코호트별. 출처: 같은 파일.', '')
tab(['코호트', '행', '일치', '오포함', '누락', '양성 오류', '음성 오류', '정밀도', '재현율', 'hazard 파이프라인', 'hazard 정답'], [[LAB[s], fi(byc.loc[s].rows), fp(byc.loc[s].agree), fi(byc.loc[s].risk_fp), fi(byc.loc[s].risk_fn), fi(byc.loc[s].label_fp), fi(byc.loc[s].label_fn), fp(byc.loc[s].precision), fp(byc.loc[s].recall), f"{byc.loc[s].hazard_pipe:.4f}", f"{byc.loc[s].hazard_truth:.4f}"] for s in order])
P('표 6. 팀 구성·매치 유형·종결 단계별. 출처: 같은 파일.', '')
rows = [[f'AI 팀원 없음(사람 팀)', fi(bya.loc['no_ai'].rows), fp(bya.loc['no_ai'].agree), fp(bya.loc['no_ai'].precision), fp(bya.loc['no_ai'].recall), f"{bya.loc['no_ai'].hazard_pipe:.4f}", f"{bya.loc['no_ai'].hazard_truth:.4f}"], ['전원 AI 팀', fi(bya.loc['all_ai'].rows), fp(bya.loc['all_ai'].agree), fp(bya.loc['all_ai'].precision), fp(bya.loc['all_ai'].recall), f"{bya.loc['all_ai'].hazard_pipe:.4f}", f"{bya.loc['all_ai'].hazard_truth:.4f}"]]
rows += [[f'매치 유형 {t}', fi(bym.loc[t].rows), fp(bym.loc[t].agree), fp(bym.loc[t].precision), fp(bym.loc[t].recall), f"{bym.loc[t].hazard_pipe:.4f}", f"{bym.loc[t].hazard_truth:.4f}"] for t in ('official', 'competitive', 'airoyale')]
rows += [['다음 단계가 종결 단계', fi(byt.loc[True].rows), fp(byt.loc[True].agree), fp(byt.loc[True].precision), fp(byt.loc[True].recall), f"{byt.loc[True].hazard_pipe:.4f}", f"{byt.loc[True].hazard_truth:.4f}"], ['다음 단계가 비종결', fi(byt.loc[False].rows), fp(byt.loc[False].agree), fp(byt.loc[False].precision), fp(byt.loc[False].recall), f"{byt.loc[False].hazard_pipe:.4f}", f"{byt.loc[False].hazard_truth:.4f}"]]
tab(['구분', '행', '일치', '정밀도', '재현율', 'hazard 파이프라인', 'hazard 정답'], rows)
off_all = met(cnt, 'tr_off')
P(f'공식 정답(`time_survived`)과 비교하면(AI 포함 전체에서는 파이프라인과의 일치율이 {fp(off_all.agree, 2)}로 낮다. AI의 `time_survived`가 틀리기 때문이다) 사람 팀에서 이벤트 정답과 {fp(hum_agree, 2)}가 같아 결론이 같다. 사람 팀 컷별 hazard(공식 정답): ' + ', '.join(f'k={k} {hum_off.loc[k].hazard_truth:.3f}' for k in range(1, 8)) + '. 사람 팀 양성 정밀도·재현율(공식 정답)은 ' + ', '.join(f'k={k} {fp(hum_off.loc[k].precision, 0)}/{fp(hum_off.loc[k].recall, 0)}' for k in range(1, 8)) + '.', '')
# ---- 4. decomposition ----
P('## 4. 불일치 분해', '')
P(f'표 7. 이벤트 정답과 다른 {fi(ndis)}행의 상호배타 분해(우선순위: 이탈 → 소생·재투입 → 경계 15초 → 위치 행 없음 → 기절 스냅샷 → 사망 이전 스냅샷). 출처: `csv/label_audit_{T1}_disagreements.csv.gz`(`typ`, `cat`).', '')
tab(['유형 \\ 원인'] + [CATN[c] for c in cat.columns] + ['합계'], [[f'{TYPN[t]}'] + [fi(cat.loc[t, c]) for c in cat.columns] + [fi(cat.loc[t].sum())] for t in cat.index] + [['합계'] + [fi(cat[c].sum()) for c in cat.columns] + [fi(ndis)]])
P(f'위 우선순위는 한 행에 하나의 원인만 붙인다. 겹치는 원인은 표 8의 비배타 플래그로 본다. 출처: `csv/label_audit_{T1}_flag_counts.csv`.', '')
def rate(f_, ag): return float(fl[(fl.agree == ag) & fl[f_]].n.sum() / fa[ag])
FL_N = [('near15', '정답 사망이 경계(k 끝 또는 k+1 끝) 15초 이내'), ('nopos_k', '단계 k 위치 행 없음'), ('nopos_n', '단계 k+1 위치 행 없음'), ('knock_k', '단계 k 마지막 스냅샷이 기절뿐(`alive_players_end=0`, 행>0)'), ('knock_n', '단계 k+1 마지막 스냅샷이 기절뿐'), ('redeploy', '소생·재투입 팀원 있음'), ('ai_team', 'AI 팀원 있음'), ('airoyale', '`airoyale` 매치'), ('logout', '이탈 팀원 있음')]
tab(['플래그', '일치 행 비율', '불일치 행 비율'], [[n_, fp(rate(f_, True), 2), fp(rate(f_, False), 2)] for f_, n_ in FL_N])
if fx and ag3['pst'] and ag3['pst_fx']:
    a0, a1, a2 = ag3['pst'], ag3['pst_re'], ag3['pst_fx']
    P(f'표본 3,250매치({int(float(t2n.group(1))):,}행) 위치 단위 재현. 파이프라인 규칙을 위치에서 다시 계산하면 저장된 라벨과 {fp(float(rep.group(1)), 2)}가 같다(마지막 스냅샷 생존 수는 {fp(float(rep.group(2)), 2)}). 위치 창을 시작 시계로 옮긴 가상 파이프라인(`pst_fx`: 같은 단일 스냅샷 규칙, 창만 `elapsed_time + δ_pos`)의 정답 일치율은 {fp(float(a2.group(1)), 2)}로 현행 {fp(float(a0.group(1)), 2)}보다 {100 * (float(a2.group(1)) - float(a0.group(1))):.2f}%p 높다. 현행 불일치 {int(float(fx.group(1))):,}행 중 시계를 고치면 맞는 행은 {int(float(fx.group(2))):,}행({fp(float(fx.group(2)) / float(fx.group(1)))})이다. 즉 시계 불일치는 불일치의 일부이고, 나머지는 단일 스냅샷 규칙 자체의 문제다.', '')
if kn and un and nr:
    def cv(d_, c_): return float((d_[c_].fillna(0) == 0).mean())
    ak_, dk_ = t2[(t2.typ == 'agree') & (t2.tr_evt >= 1)], t2[t2.typ == 'risk_FN']; an_, dn_ = t2[(t2.typ == 'agree') & (t2.pst == 1) & (t2.tr_evt == 1)], t2[t2.typ == 'label_FP']
    P(f'기절 스냅샷 {int(float(kn.group(1))):,}행(표본)의 경계 시점 팀 상태: 건강한 생존 팀원이 30초 안에 관측됨 {int(float(kn.group(2))):,}행({fp(float(kn.group(2)) / float(kn.group(1)))}), 생존 팀원 전원이 기절 {int(float(kn.group(3))):,}행, 관측된 팀원 없음 {int(float(kn.group(4))):,}행, 기절한 팀원이 나중에 다시 건강해짐(소생 증거) {int(float(kn.group(5))):,}행. 위치 행 없음 {int(float(un.group(1))):,}행 중 경계 직전 30초에도 팀 위치 행이 없는 행 {int(float(un.group(2))):,}. 정답상 경계에 살아 있는 팀만 보면, 경계 직전 30초에 팀 위치 행이 없는 비율은 파이프라인이 사망으로 읽은 행(위험집합 누락, 경계 k)에서 {fp(cv(dk_, "rows30_k"))}, 일치 행에서 {fp(cv(ak_, "rows30_k"))}이고, 라벨 양성 오류(경계 k+1)에서 {fp(cv(dn_, "rows30_n"))}, 일치 음성 행에서 {fp(cv(an_, "rows30_n"))}다(일치 행은 3% 무작위 표본).', '')
P(f'`team_phase_state.alive_players_*` 사용과의 관계: 모든 라벨이 이 열에서 나오므로 기절 스냅샷·사망 이전 스냅샷·위치 행 없음 세 원인({fi(cat["knocked_last_snapshot"].sum() + cat["stale_last_snapshot"].sum() + cat["unobserved_no_rows"].sum())}행, {fp((cat["knocked_last_snapshot"].sum() + cat["stale_last_snapshot"].sum() + cat["unobserved_no_rows"].sum()) / ndis)})이 직접적인 결과다. 데이터 프로파일 문서 표 18에서 보였듯 이 열은 팀당 한 명만 관측한 스냅샷 값이다. AI·airoyale: AI 팀원이 있는 팀은 전체 행의 {fp(bya.loc["all_ai"].rows / nrows)}이고 불일치 행의 {fp(rate("ai_team", False), 1)}를 차지한다(일치 행의 {fp(rate("ai_team", True), 1)}). 전원 AI 팀은 일치 {fp(bya.loc["all_ai"].agree)}, 양성 정밀도 {fp(bya.loc["all_ai"].precision)}로 사람 팀({fp(bya.loc["no_ai"].agree)}, {fp(bya.loc["no_ai"].precision)})보다 정확하다. `airoyale` 매치는 일치 {fp(bym.loc["airoyale"].agree)}다.', '')
# examples
ex = t2[t2.typ != 'agree'].copy(); bk = ex.typ.isin(['risk_FN', 'risk_FP']).values
for c_ in ['n_alive_b', 'n_healthy_b', 'n_knock_b', 'n_unsamp_b', 'n_revive_b', 'rows30']: ex[c_ + '_rel'] = np.where(bk, ex[c_ + '_k'], ex[c_ + '_n'])
EXC = ['knocked_last_snapshot', 'stale_last_snapshot', 'revive_redeploy', 'near_boundary_15s', 'unobserved_no_rows']; exr = []
for c in EXC:
    d = ex[(ex.cat == c) & (ex.n_ai == 0)]
    if len(d): exr.append(d.sample(min(5, len(d)), random_state=7).assign(cat=c))
d = ex[ex.ai_bucket == 'all_ai']; exr.append(d.sample(5, random_state=7).assign(cat='all_ai_team'))
afk = dis[dis.cat == 'afk_logout']; afk = afk.sample(min(5, len(afk)), random_state=7).assign(cat='afk_logout')
for c_ in ['n_healthy_b_rel', 'n_knock_b_rel', 'n_unsamp_b_rel', 'n_revive_b_rel']: afk[c_] = np.nan
exr.append(afk[[c for c in exr[0].columns if c in afk.columns]])
exd = pd.concat(exr); exd.to_csv(f'{C}/label_audit_examples.csv', index=False)
P('표 9. 범주별 예시 5건(표본 위치 단위 행). 전체 ID와 열은 `csv/label_audit_examples.csv`. k: 컷, 경계: `phase_end_s`(k), `phase_end_s`(k+1), T: 이벤트 정답 팀 사망 시각(초, 시작 시계), 상태: 파이프라인→정답(0 밖, 1 음성, 2 양성), 단계 k/k+1은 (`position_rows`, `alive_players_end`), 경계 팀원은 (건강/기절/미관측/소생증거).', '')
tab(['범주', '매치', '팀', 'k', '경계 k / k+1', 'T', '공식 T', '상태', '단계 k', '단계 k+1', '팀원 AI/전체', '경계 팀원'], [[CATN[r.cat], str(r.match_id)[:8], int(r.team_id), int(r.k), f"{r.e_k:.0f} / {('종결' if r.term_n else f'{r.e_n:.0f}')}", ('생존' if not np.isfinite(r.T_evt) else f"{r.T_evt:.0f}"), ('생존' if not np.isfinite(r.T_off) else f"{r.T_off:.0f}"), f"{int(r.pst)}→{int(r.tr_evt)}", f"({fi(r.pr_k) if pd.notna(r.pr_k) else 0}, {fi(r.ape_k) if pd.notna(r.ape_k) else 0})", f"({fi(r.pr_n) if pd.notna(r.pr_n) else 0}, {fi(r.ape_n) if pd.notna(r.ape_n) else 0})", f"{int(r.n_ai)}/{int(r.n_mem)}", f"({fi(r.n_healthy_b_rel) if pd.notna(r.n_healthy_b_rel) else '-'}/{fi(r.n_knock_b_rel) if pd.notna(r.n_knock_b_rel) else '-'}/{fi(r.n_unsamp_b_rel) if pd.notna(r.n_unsamp_b_rel) else '-'}/{fi(r.n_revive_b_rel) if pd.notna(r.n_revive_b_rel) else '-'})"] for r in exd.itertuples()], align='lllrrrrrrrrr')
# ---- 5. conclusion ----
P('## 5. 결론과 교정안', '')
P(f'판정: 학습 라벨은 실질적으로 틀렸다. 전체 {nrows:,}행 중 {fi(allm.changed)}행({fp(allm.changed / nrows, 2)})의 상태가 정답과 다르고, 학습에 쓰이는 {fi(allm.pipe_risk)}행 중 {fi(allm.risk_fp + allm.label_fp + allm.label_fn)}행({fp((allm.risk_fp + allm.label_fp + allm.label_fn) / allm.pipe_risk)})이 위험집합 오포함이거나 라벨이 틀렸다. 양성 {fi(allm.pipe_pos)}개 중 정답과 겹치는 것은 {fi(allm.both_pos)}개({fp(allm.precision)})이고 정답 양성 {fi(allm.truth_pos)}개 중 {fp(allm.recall)}만 잡는다. 컷별 hazard 곡선이 다르고(표 10) 후반 컷의 위험집합·양성이 구조적으로 어긋난다.', '')
P('교정 정의(제안): 팀 탈락 시각 T = 팀원 마지막 킬 이벤트 시각(`pubg.telem_kills`, victim 기준 `max(event_time)` − `stdb.matches.started_at`)의 최댓값, 생존 팀원이 있으면 ∞(우승팀 검열), 이벤트 없는 사망은 `time_survived` + 매치 오프셋(사람 킬의 중앙값 차이).', '',
  '- `target_mask(k) = (T > phase_end_s(k)) AND (단계 k+1 존재)`',
  '- `y_hazard(k) = (T <= phase_end_s(k+1))`, k+1이 매치의 마지막 단계이면 `phase_end_s(k+1) = ∞`.',
  '- 경계는 지금의 `stdb.match_phase_global.phase_end_s`를 그대로 쓴다(시작 시계). 위치·피처 쪽은 건드리지 않는다. `time_survived`는 사람 매치에서 보조 검증(일치율 ' + f'{fp(hum_agree, 2)})로만 쓴다: AI는 값이 틀리고 소생 팀은 마지막 생존 구간만 센다.', '')
P(f'표 10. 교정 라벨의 컷별 hazard와 변경 행 수. 변경 = 위험집합 포함 여부 또는 라벨이 현행과 다른 행. 출처: `csv/label_audit_{T1}_counts.csv`.', '')
tab(['k', '교정 위험집합', '교정 양성', '교정 hazard', '현행 hazard', '변경 행', '변경 비율(행 대비)', '현행 학습 행 중 폐기', '새로 추가'], [[str(k), fi(byk.loc[k].truth_risk), fi(byk.loc[k].truth_pos), f"{byk.loc[k].hazard_truth:.4f}", f"{byk.loc[k].hazard_pipe:.4f}", fi(byk.loc[k].changed), fp(byk.loc[k].changed / byk.loc[k].rows, 2), fi(byk.loc[k].risk_fp), fi(byk.loc[k].risk_fn)] for k in range(1, 8)] + [['전체', fi(allm.truth_risk), fi(allm.truth_pos), f"{allm.hazard_truth:.4f}", f"{allm.hazard_pipe:.4f}", fi(allm.changed), fp(allm.changed / nrows, 2), fi(allm.risk_fp), fi(allm.risk_fn)]])
P(f'학습 행 기준으로 보면 현행 {fi(allm.pipe_risk)}행 중 {fi(allm.risk_fp)}행을 버리고 {fi(allm.label_fp + allm.label_fn)}행의 라벨을 뒤집고 {fi(allm.risk_fn)}행을 더하면 교정 위험집합 {fi(allm.truth_risk)}행이 된다.', '')
P('표 11. 교정 hazard, 코호트 × 컷. 출처: 같은 파일.', '')
pv = byck.pivot(index='cohort', columns='k', values='hazard_truth').loc[order]; rk = byck.pivot(index='cohort', columns='k', values='truth_risk').loc[order]
tab(['코호트'] + [f'k={k}' for k in range(1, 8)] + ['변경 비율'], [[LAB[s]] + [f"{pv.loc[s, k]:.3f} ({fi(rk.loc[s, k])})" for k in range(1, 8)] + [fp(byc.loc[s].changed / byc.loc[s].rows, 2)] for s in order])
P('괄호는 교정 위험집합 크기. 남는 한계: (1) 위험집합이 `phase_end_s`(k) 생존을 조건으로 하므로 컷(`phase_end_s − 10`)과 `phase_end_s` 사이 10초 안에 탈락한 팀은 위험집합에서 빠진다(조건부 선택, 이번에는 측정하지 않음). (2) 소생 모드에서 "팀 탈락"은 마지막 팀원 사망이다. (3) 사람과 AI 팀의 정확도가 다르므로 `match_type`·AI 플래그별 보고가 필요하다.', '')
P('## 6. 시계 불일치와 Fix B 보호 구간', '')
ov_share = float(np.minimum(dm, 10).div(10).mean()); ov_sec = float(np.where(dm <= 10, dm ** 2 / 20, dm - 5).mean())
P(f'위치 행의 `elapsed_time` t는 시작 시계로 t + δ_pos다. `team_phase_state`는 `elapsed_time BETWEEN phase_start_s AND phase_end_s`로 위치를 읽고(`stdb/causal_cut.py` docstring은 `phase_end_s`에서 끝나는 원시 위치·아이템 창을 바꾸지 않는다고 적었다), 경계 숫자는 시작 시계라서 이 창은 시작 시계로 `phase_end_s + δ_pos`까지 간다. 실제 단계 경계는 격자 내림 때문에 `phase_end_s`보다 u초(0 초과 10 이하) 뒤이므로 창이 다음 단계에 걸치는 길이는 max(δ_pos − u, 0)초다. 표본 3,250매치의 δ_pos는 중앙값 {dm.median():.2f}초, p95 {dm.quantile(.95):.2f}초, 최대 {dm.max():.2f}초이고 5초 초과 {fp((dm > 5).mean())}, 10초 초과 {fp((dm > 10).mean())}, 15초 초과 {fp((dm > 15).mean())}다(전체 17,943매치의 게임 상태 δ: 5초 초과 {fp((mcs > 5).mean())}, 10초 초과 {fp((mcs > 10).mean())}, 20초 초과 {fp((mcs > 20).mean())}). u를 균등 분포로 두면 창이 다음 단계에 걸치는 매치가 평균 {fp(ov_share)}이고 걸치지 않는 매치를 포함한 평균 겹침이 {ov_sec:.1f}초다(추정, 직접 측정은 아니다). Fix B의 `phase_end_s − 10`은 격자 읽기 창(±5초)만 가정하므로 δ_pos > 5 + u인 매치(평균 {fp(leak)})에서는 마지막 격자점도 이미 다음 단계다. 이 겹침은 라벨 쪽에서 "단계 끝 생존"이 실제 경계보다 늦게 평가되는 형태로 나타나고(시계를 고치면 불일치의 {fp(float(fx.group(2)) / float(fx.group(1)))}가 사라진다, 4절), 피처 쪽에서는 단계 k 피처가 단계 k+1의 처음 몇 초를 볼 수 있다는 뜻이다. 해결은 경계를 경과 시계로 옮기거나(`phase_end_s − δ_match`, δ_match = 게임 상태 `event_time − elapsed_time` 중앙값 − `started_at`) 위치 시간에 δ_match를 더해 한 시계로 맞추는 것이다. 이벤트 기반 라벨은 시작 시계끼리 비교하므로 이 문제와 무관하다.', '')
P('## 7. 산출물', '')
P('- `csv/label_audit_full3_*`(전수 집계·불일치 행·매치별 시계), `csv/label_audit_t2c_rows.csv.gz`(표본 위치 단위 불일치 행과 일치 행 3%), `csv/label_audit_metrics_by_*.csv`, `csv/label_audit_examples.csv`, `csv/label_audit_clock_probe_*.csv`.',
  '- 로그: `logs/09_clock_probe.log`, `logs/10_label_audit_tier1.log`, `logs/11_label_audit_tier2.log`.', '')
open(DOC, 'w').write('\n'.join(L) + '\n'); print('doc written', len(L), 'lines', sum(len(x) for x in L), 'chars')
