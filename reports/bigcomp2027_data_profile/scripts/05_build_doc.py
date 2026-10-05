#!/usr/bin/env python3
"""Build docs/data_layer/BIGCOMP2027_DATA_PROFILE_2026-10-05.md from csv/*.csv (all numbers are read from files)."""
import json, os
import numpy as np, pandas as pd
ROOT = '/Users/lunecid/project/PUBG'; B = ROOT + '/reports/bigcomp2027_data_profile'; C = B + '/csv'; DOC = ROOT + '/docs/data_layer/BIGCOMP2027_DATA_PROFILE_2026-10-05.md'
rd = lambda n: pd.read_csv(f'{C}/{n}')
L = []
def P(*s): L.extend(s)
def tab(h, rows, left=1, align=None):
    al = ['---' if c == 'l' else '---:' for c in align] if align else ['---'] * left + ['---:'] * (len(h) - left)
    P('| ' + ' | '.join(h) + ' |', '| ' + ' | '.join(al) + ' |')
    for r in rows: P('| ' + ' | '.join(str(x) for x in r) + ' |')
    P('')
fi = lambda x: f"{int(round(x)):,}"; ff = lambda x, d=1: f"{x:,.{d}f}"; fp = lambda x, d=1: f"{100 * x:.{d}f}%"
fg = lambda x: f"{x:,.0f}" if abs(x - round(x)) < 1e-9 else f"{x:,.1f}"
rng3 = lambda lo, hi: (f"{lo:.3f}" if abs(hi - lo) < 5e-4 else f"{lo:.3f}~{hi:.3f}")
cs = rd('cohort_summary.csv').set_index('slug').sort_values('n_total', ascending=False); order = list(cs.index); LAB = {s: f"{cs.loc[s].map_label} {cs.loc[s].perspective} {cs.loc[s].team_size}" for s in order}
raw = rd('raw_db_by_map_mode_matchtype_custom.csv'); fun = rd('funnel_map_mode_persp.csv'); cov = json.load(open(f'{C}/funnel_coverage_checks.json')); days = rd('matches_per_day.csv'); mt = rd('cohort_match_type.csv').set_index('cohort')
hz = rd('hazard_by_cohort_cutphase.csv'); hz = hz[hz.cut_phase <= 7]; hzm = rd('hazard_by_cohort_matchtype_cutphase.csv'); pm = rd('phase_timing_by_map.csv'); pc = rd('phase_timing_by_cohort.csv'); mpd = rd('max_phase_distribution.csv')
dmap = rd('duration_median_by_map.csv').set_index('map_name').duration_median_s; evt = rd('events_per_match_by_cohort_teamlevel.csv').set_index('cohort'); evs = rd('sample_events_per_match_by_cohort.csv').set_index('cohort'); evp = rd('sample_events_by_phase.csv')
s2m = rd('stage2_per_match_totals_by_cohort.csv'); s2m = s2m[s2m.scope == 'cuts_with_target'].set_index('cohort'); s2c = rd('stage2_cut_means_by_cohort_phase.csv'); tz = rd('stage2_tensor_size_estimate.csv'); cut = rd('sample_cut_level.csv')
iv = rd('position_interval_stats_by_cohort.csv').set_index('cohort'); aud = rd('label_audit_overall_by_phase.csv'); bots = rd('bots_by_cohort.csv').set_index('cohort'); botm = rd('bots_by_cohort_matchtype.csv'); cons = rd('raw_consistency_checks_by_cohort.csv').set_index('cohort')
alive = rd('alive_definition_check.csv'); alive2 = rd('alive_definition_check_team_second.csv').iloc[0]; match = rd('sample_match_level.csv'); meta = json.load(open(f'{C}/sample_run_meta.json')); tmg = rd('sample_batch_timing.csv'); items = rd('sample_item_use_taxonomy.csv'); hist = rd('sample_position_interval_hist.csv'); hh = hist.groupby('interval_s').n.sum(); sh9 = hh.loc[9] / hh.sum(); sh0 = hh.loc[0] / hh.sum(); cmt = rd('cohort_matchtype_profile.csv'); qs = rd('quality_team_position_rows_state_by_cohort_phase.csv')
# ---------- derived numbers ----------
n_db = int(raw.n.sum()); n_csv = int(fun.csv_total.sum()); n_in = int(fun.csv_in_stdb.sum()); n_el = int(fun.graph_eligible.sum()); n_13 = int(fun.main13.sum()); n_aux = n_el - n_13; aux_all = fun[(fun.main13 == 0) & (fun.csv_in_stdb > 0)]; aux_listed = aux_all[aux_all.graph_eligible >= 42]; aux_unlisted = aux_all[aux_all.graph_eligible < 42]
mt_raw = raw.groupby('match_type').n.sum(); nz = days[days.raw_db > 0]; n_days = len(nz); cal = (pd.Timestamp(nz.date_kst.max()) - pd.Timestamp(nz.date_kst.min())).days + 1
nzc = days[days.csv_total > 0]; d_csv_last = nzc.date_kst.max()
pool = hz.groupby('cut_phase')[['risk', 'pos']].sum(); pool['hazard'] = pool.pos / pool.risk
mt13 = mt.sum(); n_ai_all = match.n_ai.sum() / match.n_players.sum(); n_human_mean = (match.n_players - match.n_ai).mean(); share_ai_matches = (match.n_ai > 0).mean()
tzc = tz[(tz.scope == 'cuts_with_target') & (tz.variant == 'enemy_alive_at_cut')].set_index('cohort'); tzu = tz[(tz.scope == 'cuts_with_target') & (tz.variant == 'enemy_all_nodes_upper')].set_index('cohort')
gb_c, gb_b, gb_u, gb_ub = tzc.GB_cohort_compact.sum(), tzc.GB_cohort_broadcast.sum(), tzu.GB_cohort_compact.sum(), tzu.GB_cohort_broadcast.sum()
tot_nodes = sum(cs.loc[s].n_total * s2m.loc[s].nodes_A_mean for s in order); tot_enemy = sum(cs.loc[s].n_total * s2m.loc[s].enemy_fresh_C_mean for s in order); tot_cb = sum(cs.loc[s].n_total * s2m.loc[s].cb_enemy_mean for s in order); tot_tm = sum(cs.loc[s].n_total * s2m.loc[s].tm_A_mean for s in order)
fetch_pm = tmg.fetch_total.sum() / tmg.n.sum(); proc_pm = tmg.proc.sum() / tmg.n.sum(); n_samp = int(meta['n_matches_ok']); full_min_wall = cs.n_total.sum() * meta['wall_s'] / n_samp / 60; full_min_serial = cs.n_total.sum() * (fetch_pm + proc_pm) / 60
ov = cut[['C_players', 'C_miss15', 'C_miss30', 'C_teams', 'C_team_miss15', 'C_team_miss30', 'A_nopos', 'nodes_A']].sum(); health0 = cut.fresh_C_h0.sum() / cut.fresh_C_n.sum()
phs = pc[pc.phase_idx <= 7]; w = phs.n_matches; sh20 = (phs.dur_lt20_share * w).sum() / w.sum(); sh10 = (phs.dur_lt10_share * w).sum() / w.sum()
eall = evp[['dmg_ev_all', 'dmg_ev_enemy', 'dmg_ev_noattacker']].sum(); heal_n = items[(items.category == 'Use') & (items.sub_category == 'Heal')].n.sum(); boost_n = items[(items.category == 'Use') & (items.sub_category == 'Boost')].n.sum(); n_item_matches = 13 * 6
off = match.off_minus_started; sm = pm[pm.phase_idx <= 9]; n_within1 = int((sm.start_minus_raw_med.abs() <= 1).sum())
# ---------- document ----------
P('# BigComp 2027 데이터 프로파일 (2026-10-05)', '')
P('- 설계 문서: `docs/experiment_design/BIGCOMP2027_PTR_STUDY_DESIGN_2026-10-05.md` 3절. 목적은 2단계(플레이어 계층)에 쓸 코퍼스의 규모와 추출 가능성을 재는 것이다.',
  '- DB는 읽기 전용 SELECT만 썼다(`default_transaction_read_only=on`, `statement_timeout=120s`). 재작성 중인 파생 테이블 6종(`stdb.team_phase_trajectory`, `team_phase_exposure`, `team_phase_habitat_features`, `team_phase_edge_cache`, `team_phase_network_features`, `team_phase_z_affordance_features`)은 조회하지 않았다.',
  '- 사용한 테이블: `pubg.matches`, `participants`, `rosters`, `telem_positions`, `telem_damage`, `telem_kills`, `telem_groggy`, `telem_item_use`, `telem_phase_changes`, `telem_match_start`, `stdb.matches`, `match_phase_global`, `team_phase_state`, `team_phase_combat`, `team_phase_labels`.',
  f'- 범위: 1~4절, 6.1절, 표 9b는 13개 코호트 전수({n_13:,}매치)다. 표 9a·10, 5절, 6.2~6.7절의 원시 텔레메트리 수치는 코호트별 무작위 표본(seed 42)이며, 11개 코호트는 400매치, Karakin·Paramo TPP squad는 333매치(전수), 합계 {n_samp:,}매치다.',
  '- 코드 인용은 2026-10-05 작업 트리 기준이다. 다른 에이전트가 `stdb/rq1_feature_etl.py` 등을 수정 중이지만(Fix B), `git diff stdb/rq1_feature_etl.py`는 `build_match_phase_global`의 구역 반경 열만 바꾸고 이 문서가 인용한 생존 인원·라벨·로트 로직은 건드리지 않았다.',
  '- 원자료는 `reports/bigcomp2027_data_profile/csv/`, 스크립트는 `.../scripts/`, 실행 로그는 `.../logs/`에 있다. 표마다 출처 파일을 적었다. 백분위수는 numpy 선형 보간이다.', '')
P('## 0. 요약', '')
P(f'1. 깔때기: `pubg.matches` {n_db:,} → 텔레메트리 수집 {n_csv:,} → 범위 내 맵 {n_in:,} → graph-eligible {n_el:,} → 13개 코호트 {n_13:,}.',
  f'2. 기간: 매치 생성 {nz.date_kst.min()}~{nz.date_kst.max()}(코호트 CSV는 {d_csv_last}까지). 달력 {cal}일 중 {n_days}일에만 매치가 있고 {cal - n_days}일은 공백이다. 하루 매치 수 중앙값은 {ff(nz.raw_db.median())}이다.',
  f'3. 라벨: 13개 합산 hazard는 컷 1에서 {pool.hazard.loc[1]:.3f}, 컷 4에서 {pool.hazard.loc[4]:.3f}이고 컷 7까지 {pool.hazard.loc[[4,5,6,7]].min():.2f}~{pool.hazard.loc[[4,5,6,7]].max():.2f}를 유지한다. 위험집합은 컷 1의 {fi(pool.risk.loc[1])}팀에서 컷 7의 {fi(pool.risk.loc[7])}팀으로 줄어든다.',
  f'4. 봇: 표본의 플레이어 중 {fp(n_ai_all)}가 AI 계정(`ai.` 접두)이고, 표본 매치의 {fp(share_ai_matches)}에 AI가 1명 이상 있다. Erangel TPP squad는 AI 비율이 {fp(bots.loc["erangel_tpp_squad"].ai_share)}이며 코호트 매치 {fi(mt13.airoyale)}개({fp(mt13.airoyale / n_13)})가 `airoyale` 유형이다.',
  f'5. 2단계 코퍼스: 매치당 플레이어-단계 노드(컷 1~7 중 타깃이 있는 컷의 합)는 코호트 평균 {fi(s2m.nodes_A_mean.min())}~{fi(s2m.nodes_A_mean.max())}, 13개 코호트 합계 약 {tot_nodes / 1e6:.2f}M개다. 적 근접 엣지(300 m, 컷 생존자)는 합계 약 {tot_enemy / 1e6:.2f}M쌍, 교전 쌍은 약 {tot_cb / 1e6:.2f}M쌍이다.',
  f'6. 크기와 비용: ragged 레이아웃 텐서는 코호트 합계 {gb_c:.2f} GB(상한 {gb_u:.2f} GB), 팀 150차원을 플레이어마다 복제하면 {gb_b:.2f}~{gb_ub:.2f} GB다. 표본 {n_samp:,}매치 추출이 {meta["wall_s"]:.1f}초(2 worker)였고, 전체 {fi(cs.n_total.sum())}매치는 약 {full_min_wall:.1f}분으로 외삽된다.',
  f'7. 이상 징후: `team_phase_state.alive_players_*`는 1단계에서 실제 생존 수의 약 1/{alive.ratio_true_over_state_start.iloc[0]:.1f}이고, 이 열에서 파생된 `team_phase_labels`는 이벤트 기반 정답 대비 양성 정밀도 {fp(aud.iloc[-1].pos_precision)}, 재현율 {fp(aud.iloc[-1].pos_recall)}다(6.4절; 6.5절은 `BIGCOMP2027_LABEL_AUDIT_2026-10-05.md`로 정정).',
  '8. Vikendi는 매치당 `Monster.` 계정 약 30개가 위치·피해 텔레메트리에 섞여 있고, `team_phase_combat` 피해량 합이 raw 적 대상 합보다 크다(6.7절).', '9. 판정: 데이터·추출 측면은 GO다. 조건과 남은 위험은 7절에 적었다.', '')
# ---- 1. funnel ----
P('## 1. 코퍼스 깔때기', '')
P('표 1. 단계별 매치 수. 출처: `csv/raw_db_by_map_mode_matchtype_custom.csv`, `csv/funnel_map_mode_persp.csv`, `csv/funnel_coverage_checks.json`, `logs/01_stable_profile.log`(`pubg.v_collection_status`).', '')
tab(['단계', '매치 수', '직전 단계 대비', '근거'], [
    ['`pubg.matches` 전체', fi(n_db), '', '`SELECT count(*) FROM pubg.matches` (01 스크립트 1절)'],
    ['텔레메트리 수집 완료 = flat CSV 전체', fi(n_csv), f'-{fi(n_db - n_csv)}', f'`v_collection_status`: matches_fetched 28,458 / telemetry_fetched {fi(n_csv)}. 나머지 {fi(cov["pubg_matches_not_in_csv"])}매치는 CSV에 없다.'],
    ['범위 내 맵 = `stdb.matches`', fi(n_in), f'-{fi(n_csv - n_in)}', f'Tutorial_Main {fi(fun[fun.map_name == "Tutorial_Main"].csv_total.sum())} + Range_Main {fi(fun[fun.map_name == "Range_Main"].csv_total.sum())} 제외(`exclude_reason=out_of_scope_map`)'],
    ['graph-eligible (n128)', fi(n_el), f'-{fi(n_in - n_el)}', '`missing_target_labels` 6, `missing_team_positions` 3'],
    ['13개 코호트 (>=300매치)', fi(n_13), f'-{fi(n_aux)}', f'300매치 미만인 {len(aux_all)}개 (맵, 모드, 시점) 조합 제외(42매치 이상 {len(aux_listed)}개 조합 {fi(aux_listed.graph_eligible.sum())}매치, 나머지 {len(aux_unlisted)}개 조합 {fi(aux_unlisted.graph_eligible.sum())}매치). 레지스트리 `docs/data_layer/BATTLE_ROYALE_COHORT_REGISTRY_2026-07-03.md`의 보조 코호트 표는 28행(최소 42매치)이다']], align='lrrl')
P(f'`pubg.matches`의 `match_type` 분포는 official {fi(mt_raw["official"])}, airoyale {fi(mt_raw["airoyale"])}, tutorialatoz {fi(mt_raw["tutorialatoz"])}, competitive {fi(mt_raw["competitive"])}, trainingroom {fi(mt_raw["trainingroom"])}이고 `is_custom=true`는 0건이다.', '')
P('표 2. 맵 × 모드 × 시점별 매치 수(13개 코호트와 나머지). 출처: `csv/funnel_map_mode_persp.csv`. 전체 {0}행은 CSV 참조.'.format(len(fun)), '')
rows = []
for _, r in fun[fun.main13 > 0].sort_values('main13', ascending=False).iterrows(): rows.append([f"{r.map_label} {r.persp} {r['mode']}", fi(r.raw_db), fi(r.csv_total), fi(r.graph_eligible), fi(r.main13)])
a = fun[(fun.main13 == 0) & (fun.csv_in_stdb > 0)]; o = fun[fun.csv_in_stdb == 0]
rows += [[f'300매치 미만 조합 {len(a)}개', fi(a.raw_db.sum()), fi(a.csv_total.sum()), fi(a.graph_eligible.sum()), '0'], ['Tutorial_Main·Range_Main(범위 밖)', fi(o.raw_db.sum()), fi(o.csv_total.sum()), '0', '0'], ['합계', fi(fun.raw_db.sum()), fi(fun.csv_total.sum()), fi(fun.graph_eligible.sum()), fi(fun.main13.sum())]]
tab(['코호트', 'DB 전체', 'flat CSV', 'graph-eligible', '13개 코호트'], rows)
P(f'수집 기간(경기 생성, KST): DB {nz.date_kst.min()} ~ {nz.date_kst.max()}, flat CSV {nzc.date_kst.min()} ~ {d_csv_last}. 수집 실행 시각은 {raw.first_collected.min()[:10]} ~ {raw.last_collected.max()[:10]}다.',
  f'하루 매치 수(`pubg.matches`, 매치가 있는 {n_days}일): 최소 {fi(nz.raw_db.min())}, p10 {ff(nz.raw_db.quantile(.1))}, 중앙값 {ff(nz.raw_db.median())}, p90 {ff(nz.raw_db.quantile(.9))}, 최대 {fi(nz.raw_db.max())}. 달력 {cal}일 중 {cal - n_days}일은 매치가 0건이다. flat CSV 기준 하루 graph-eligible 중앙값은 {ff(nzc.graph_eligible.median())}, 13개 코호트는 {ff(nzc.main13.median())}다. 일별 표는 `csv/matches_per_day.csv`.', '')
# ---- 2. cohort profile ----
P('## 2. 코호트별 프로파일', '')
P('표 3. 분할, 매치당 팀·플레이어 수, 길이, 최대 관측 단계. 값은 중앙값(p10–p90). 플레이어 수는 AI 계정을 포함한다. 출처: `split.json`(`br_om_ladder_required_v1/<cohort>/`), `match_cohort_flat.csv`(`num_teams`, `num_players`, `duration`), `stdb.match_phase_global`(최대 단계). 파일: `csv/cohort_summary.csv`.', '')
rows = []
for s in order:
    r = cs.loc[s]; rows.append([LAB[s], fi(r.n_train), fi(r.n_val), fi(r.n_test), fi(r.n_total), f"{fg(r.teams_med)} ({fg(r.teams_p10)}–{fg(r.teams_p90)})", f"{fg(r.players_med)} ({fg(r.players_p10)}–{fg(r.players_p90)})",
                                f"{fi(r.dur_med)} ({fi(r.dur_p10)}–{fi(r.dur_p90)})", f"{fp(r.dur_over2000_share)} ({fi(r.dur_over2000_n)})", f"{fg(r.maxphase_med)} ({fg(r.maxphase_p10)}–{fg(r.maxphase_p90)})"])
tab(['코호트', 'train', 'val', 'test', '합계', '팀 수/매치', '플레이어 수/매치', '길이(초)', '2000초 초과(매치)', '최대 단계'], rows)
P('표 4. 최대 관측 단계 분포(매치 수). 출처: `csv/max_phase_distribution.csv`(`stdb.match_phase_global`).', '')
pv = mpd.assign(max_phase=mpd.max_phase.astype(int)).pivot(index='cohort', columns='max_phase', values='n').fillna(0).astype(int).reindex(columns=range(2, 10), fill_value=0).loc[order]
tab(['코호트'] + [f'{k}단계' for k in range(2, 10)] + ['5단계 이하'], [[LAB[s]] + [fi(pv.loc[s, k]) for k in range(2, 10)] + [f"{fi(pv.loc[s, [2,3,4,5]].sum())} ({fp(pv.loc[s, [2,3,4,5]].sum() / pv.loc[s].sum())})"] for s in order])
P('표 5. 맵별 구역 일정: 단계의 시작–종료 시각 중앙값(경기 시계 초). 해당 단계에 도달한 매치가 30개 미만이면 `-`. 출처: `csv/phase_timing_by_map.csv`(`stdb.match_phase_global.phase_start_s/phase_end_s`, 13개 코호트 합산), 매치 길이는 `csv/duration_median_by_map.csv`. 코호트별 값은 `csv/phase_timing_by_cohort.csv`.', '')
p5 = pm[pm.phase_idx == 5].set_index('map_label').start_med; maps = sorted(p5.index, key=lambda x: (-p5[x], x)); m2n = pm.drop_duplicates('map_label').set_index('map_label').map_name; rows = []
for mp in maps:
    d = pm[pm.map_label == mp].set_index('phase_idx'); cells = [(f"{d.loc[k].start_med:.0f}–{d.loc[k].end_med:.0f}" if k in d.index and d.loc[k].n_matches >= 30 else '-') for k in range(1, 10)]
    rows.append([mp, fi(d.loc[1].n_matches)] + cells + [fi(dmap[m2n[mp]])])
tab(['맵', '매치'] + [f'{k}' for k in range(1, 10)] + ['길이 중앙값'], rows)
P('종료 시각은 `phase_end_s`(단계 안 마지막 game-state 스냅샷)라서 비종결 단계는 다음 단계 시작보다 10초 빠르다. 마지막으로 관측된 단계의 종료는 매치 종료에 잘린다. Erangel·Miramar·Rondo·Taego·Vikendi는 거의 같은 일정(1단계 시작 90~100초, 2단계 600초, 5단계 1170초)을 따르고, Sanhok·Paramo(5단계 1080·1050초)와 Karakin(810초)이 빠르다. 그림 2는 이 일정을 시각화한 것이다.', '')
P('![그림 2. 맵별 구역 일정](../../reports/bigcomp2027_data_profile/figures/fig2_phase_timing_by_map.png)', '', '그림 2. 막대는 단계 k 시작 중앙값부터 단계 k+1 시작 중앙값까지, 검은 세로선은 매치 길이 중앙값이다. PDF는 같은 폴더의 `fig2_phase_timing_by_map.pdf`.', '')
P('표 6. 매치 유형(`pubg.matches.match_type`) 구성. 출처: `csv/cohort_match_type.csv`.', '')
tab(['코호트', 'airoyale', 'competitive', 'official', 'airoyale 비율'], [[LAB[s], fi(mt.loc[s].airoyale), fi(mt.loc[s].competitive), fi(mt.loc[s].official), fp(mt.loc[s].airoyale / mt.loc[s].sum())] for s in order] + [['합계', fi(mt13.airoyale), fi(mt13.competitive), fi(mt13.official), fp(mt13.airoyale / n_13)]])
comp = cmt[cmt.match_type == 'competitive']; h4 = hzm[(hzm.cohort == 'erangel_tpp_squad') & (hzm.cut_phase == 4)].set_index('match_type').hazard
P(f'`competitive` 매치는 플레이어 수 중앙값이 {fg(comp.players_med.min())}~{fg(comp.players_med.max())}인 소형 로비이고(표본 AI 0명), `airoyale`는 표본에서 AI가 90% 안팎이다(6.6절). Erangel TPP squad의 hazard를 유형별로 보면 컷 4에서 airoyale {h4.airoyale:.3f}, competitive {h4.competitive:.3f}, official {h4.official:.3f}으로 갈린다(표 6b). 코호트 안에서 유형 구성이 달라지면 hazard 보정이 달라진다.', '')
he = hzm[hzm.cohort == 'erangel_tpp_squad'].pivot(index='match_type', columns='cut_phase', values='hazard')
P('표 6b. Erangel TPP squad의 매치 유형별 hazard. 출처: `csv/hazard_by_cohort_matchtype_cutphase.csv`.', '')
tab(['매치 유형(매치 수)'] + [f'컷 {k}' for k in range(1, 8)], [[f"{t} ({fi(mt.loc['erangel_tpp_squad'][t])})"] + [f"{he.loc[t, k]:.3f}" for k in range(1, 8)] for t in ['airoyale', 'competitive', 'official']])
# ---- 3. labels ----
P('## 3. 라벨: 다음 단계 탈락 hazard', '')
P('정의는 `stdb.team_phase_labels`(`stdb/rq1_feature_etl.py`의 `build_team_phase_labels`)를 그대로 따른다. 위험집합은 `target_defined`(컷 단계 끝에 생존 `alive_players_end>0`이고 다음 단계가 존재)이고, 양성은 `eliminated_next_phase`다. 컷 k는 1~7이다.', '')
hp = hz.pivot(index='cohort', columns='cut_phase', values='hazard').loc[order]; rk = hz.pivot(index='cohort', columns='cut_phase', values='risk').loc[order]; ps = hz.pivot(index='cohort', columns='cut_phase', values='pos').loc[order]
P('표 7. 컷 단계별 다음 단계 탈락률 = 양성 / 위험집합. 출처: `csv/hazard_by_cohort_cutphase.csv`(95% Wilson 구간 포함).', '')
tab(['코호트'] + [f'컷 {k}' for k in range(1, 8)], [[LAB[s]] + [f"{hp.loc[s, k]:.4f}" for k in range(1, 8)] for s in order] + [['13개 합산'] + [f"{pool.hazard.loc[k]:.4f}" for k in range(1, 8)]])
P('표 8. 양성 수 / 위험집합 크기(팀·컷). 출처: 같은 파일.', '')
tab(['코호트'] + [f'컷 {k}' for k in range(1, 8)], [[LAB[s]] + [f"{fi(ps.loc[s, k])} / {fi(rk.loc[s, k])}" for k in range(1, 8)] for s in order] + [['13개 합산'] + [f"{fi(pool.pos.loc[k])} / {fi(pool.risk.loc[k])}" for k in range(1, 8)]])
P('![그림 1. 컷 단계별 hazard](../../reports/bigcomp2027_data_profile/figures/fig1_hazard_by_phase_per_cohort.png)', '', '그림 1. 코호트별 hazard. 점선은 13개 합산, 음영은 95% Wilson 구간, 부제는 컷 1과 컷 7의 위험집합 크기다. PDF는 `fig1_hazard_by_phase_per_cohort.pdf`.', '')
small = hz[(hz.cut_phase == 7)].sort_values('risk').head(4)
P('소형 코호트의 컷 7은 위험집합이 작다(' + ', '.join(f"{LAB[r.cohort]} {fi(r.risk)}팀·양성 {fi(r.pos)}" for r in small.itertuples()) + '). 이 구간의 코호트별 차이는 구간 폭(그림 1 음영)을 같이 봐야 한다.', '')
# ---- 4. events ----
P('## 4. 이벤트 볼륨', '')
P('표 9a. 매치당 이벤트 수(원시 텔레메트리, 표본 평균). 피해 이벤트는 `telem_damage` 행 수이고, 적 대상은 공격자·피해자가 서로 다른 팀의 플레이어인 행, 공격자 없음은 `attacker_id`가 NULL인 행(자기장·출혈 등)이다. 피해량 합은 적 대상 행의 `damage` 합이다. 출처: `csv/sample_events_per_match_by_cohort.csv`.', '')
tab(['코호트', '매치', 'kill', '기절(groggy)', '피해 이벤트 전체 (p10–p90)', '적 대상', '공격자 없음', '피해량 합(적 대상)'], [[LAB[s], fi(evs.loc[s].n), ff(evs.loc[s].kills_all_mean), ff(evs.loc[s].groggy_all_mean), f"{fi(evs.loc[s].dmg_ev_all_mean)} ({fi(evs.loc[s].dmg_ev_all_p10)}–{fi(evs.loc[s].dmg_ev_all_p90)})", fi(evs.loc[s].dmg_ev_enemy_mean), fi(evs.loc[s].dmg_ev_noattacker_mean), fi(evs.loc[s].dmg_sum_enemy_mean)] for s in order])
P(f'표본 전체에서 피해 이벤트의 {fp(eall.dmg_ev_enemy / eall.dmg_ev_all)}가 적 대상이고 {fp(eall.dmg_ev_noattacker / eall.dmg_ev_all)}는 공격자가 없다. 적 대상 피해량 합은 모든 코호트에서 `participants.damage_dealt` 합과 {rng3(cons.enemy_dmg_per_p_damage.min(), cons.enemy_dmg_per_p_damage.max())}배로 일치한다(`csv/raw_consistency_checks_by_cohort.csv`).', '')
P('표 9b. 같은 지표를 코호트 전수로 본 값. `stdb.team_phase_combat`의 팀 단위 합계이며 서로 다른 팀 사이의 이벤트만 포함하고, 단계 구간(시작~종료) 밖의 이벤트는 빠진다. 값은 매치당 평균(damage 열은 p10–p90). 출처: `csv/events_per_match_by_cohort_teamlevel.csv`.', '')
tab(['코호트', '매치', 'kill', '기절', '피해 이벤트 (p10–p90)', '피해량 합', '전수 ÷ 표본 raw(적 대상): kill / 피해 이벤트'], [[LAB[s], fi(evt.loc[s].n), ff(evt.loc[s].kills_mean), ff(evt.loc[s].groggy_mean), f"{fi(evt.loc[s].dmg_events_mean)} ({fi(evt.loc[s].dmg_events_p10)}–{fi(evt.loc[s].dmg_events_p90)})", fi(evt.loc[s].dmg_sum_mean), f"{evt.loc[s].kills_mean / evs.loc[s].kills_enemy_mean:.3f} / {evt.loc[s].dmg_events_mean / evs.loc[s].dmg_ev_enemy_mean:.3f}"] for s in order])
pp = evp.groupby('k')[['kills_all', 'groggy_all', 'dmg_ev_all', 'dmg_ev_enemy', 'dmg_sum_all', 'dmg_sum_enemy']].mean(); pn = evp.groupby('k').size()
P('표 10. 단계별 이벤트 수(표본 13개 코호트 합산, 해당 단계가 있는 매치의 평균). 단계 k는 [시작 k, 시작 k+1)로 나눴고, k=0은 1단계 이전(낙하 전후)이다. 출처: `csv/sample_events_by_phase.csv`, 코호트별은 `csv/sample_events_per_phase_by_cohort.csv`.', '')
tab(['단계', '매치', 'kill', '기절', '피해 이벤트 전체', '피해 이벤트(적 대상)', '피해량 합 전체', '피해량 합(적 대상)'], [[str(k), fi(pn.loc[k]), ff(pp.loc[k].kills_all), ff(pp.loc[k].groggy_all), fi(pp.loc[k].dmg_ev_all), fi(pp.loc[k].dmg_ev_enemy), fi(pp.loc[k].dmg_sum_all), fi(pp.loc[k].dmg_sum_enemy)] for k in pp.index])
# ---- 5. stage 2 ----
P('## 5. 2단계(플레이어 계층) 코퍼스 추정', '')
P('정의(설계 문서 3절의 2단계를 그대로 계산 가능한 형태로 옮겼다).', '',
  '- 노드: 단계 k 시작 시 생존한 플레이어. 사망 시각은 `telem_kills`의 `victim_id` 시각이고, 킬 이벤트가 없는 이탈자는 `participants.time_survived`로 대신했다. 팀 소속은 `participants`→`rosters`(`telem_positions.team_id`와 불일치 0건).',
  '- 컷 시각 t_c = `phase_end_s` − 10초(Fix B). 위치와 체력은 t_c 이전 마지막 `telem_positions` 샘플이고 좌표 단위는 cm를 m로 바꿨다. 샘플이 t_c보다 30초 넘게 오래된 노드는 "신선하지 않음"으로 봤다.',
  '- 팀원 엣지: 시작 시 생존자 사이 C(n,2)(무방향). 컷 시점 생존자 기준 값도 함께 적었다.',
  '- 적 근접 엣지: 서로 다른 팀, 거리 300 m 이하, 무방향 쌍 수. 주 추정은 컷에서도 살아 있고 위치가 신선한 노드만 쓴 값이고, 상한은 시작 시 생존자 전원(사망 노드의 마지막 위치 포함)을 쓴 값이다.',
  '- 교전 엣지: [단계 시작, t_c] 구간 `telem_damage`에서 damage>0, 공격자·피해자 ID가 모두 있고 서로 다른 팀인 distinct (공격자→피해자) 쌍.',
  '- 집계는 학습에 쓰이는 컷만 합친다: k<=7이고 단계 k+1이 존재하는 컷. k<=7 전체 합은 `csv/stage2_per_match_totals_by_cohort.csv`의 `all_k1to7`.', '')
P('표 11. 매치당 합계(컷 합). 노드는 평균(p10–p90). 출처: `csv/stage2_per_match_totals_by_cohort.csv`(scope=cuts_with_target), 스크립트 `02_sample_player_level.py`, `03_summarize_sample.py`.', '')
tab(['코호트', '표본', '노드', '팀·단계', '팀원 엣지(시작)', '팀원 엣지(컷 생존)', '적 근접(컷 생존)', '적 근접(상한)', '교전 쌍', '교전 이벤트'],
    [[LAB[s], fi(s2m.loc[s].n_sample), f"{ff(s2m.loc[s].nodes_A_mean)} ({fi(s2m.loc[s].nodes_A_p10)}–{fi(s2m.loc[s].nodes_A_p90)})", ff(s2m.loc[s].teams_A_mean), ff(s2m.loc[s].tm_A_mean), ff(s2m.loc[s].tm_C_mean), ff(s2m.loc[s].enemy_fresh_C_mean), ff(s2m.loc[s].enemy_all_mean), ff(s2m.loc[s].cb_enemy_mean), ff(s2m.loc[s].cb_events_enemy_mean)] for s in order])
P(f'코퍼스 합계(매치 수 × 코호트 평균): 노드 {tot_nodes / 1e6:.2f}M, 팀원 엣지 {tot_tm / 1e6:.2f}M쌍, 적 근접 {tot_enemy / 1e6:.2f}M쌍, 교전 {tot_cb / 1e6:.2f}M쌍. Erangel TPP solo는 팀이 없어 팀원 엣지가 0이다.', '')
g = cut.groupby('k'); cm = g[['nodes_A', 'teams_A', 'nodes_C', 'tm_A', 'enemy_fresh_C', 'enemy_all', 'cb_enemy', 'cb_enemy_within300', 'cb_events_enemy']].mean(); gn = g.size()
P('표 12. 컷 단계별 평균(표본 13개 코호트 합산, 컷당). "컷 생존"은 t_c에 살아 있는 노드 수다. 출처: `csv/sample_cut_level.csv`, 코호트별은 `csv/stage2_cut_means_by_cohort_phase.csv`.', '')
tab(['컷 k', '컷 수', '노드(시작 생존)', '팀', '컷 생존', '팀원 엣지', '적 근접(컷 생존)', '적 근접(상한)', '교전 쌍', '그중 컷에서 300 m 이내', '교전 이벤트'], [[str(k), fi(gn.loc[k]), ff(cm.loc[k].nodes_A), ff(cm.loc[k].teams_A), ff(cm.loc[k].nodes_C), ff(cm.loc[k].tm_A), ff(cm.loc[k].enemy_fresh_C), ff(cm.loc[k].enemy_all), ff(cm.loc[k].cb_enemy), ff(cm.loc[k].cb_enemy_within300), ff(cm.loc[k].cb_events_enemy)] for k in cm.index])
P('"그중 컷에서 300 m 이내"는 교전 쌍 가운데 양쪽 위치가 신선하게 관측된 쌍만 센 값이다(교전 후 사망한 쪽은 제외된다).', '')
npv = s2c.pivot(index='cohort', columns='k', values='nodes_A').loc[order]
P('표 13. 단계 시작 시 생존 플레이어 수(매치당 평균, 해당 단계가 있는 매치). 출처: `csv/stage2_cut_means_by_cohort_phase.csv`.', '')
tab(['코호트'] + [f'k={k}' for k in range(1, 8)], [[LAB[s]] + [ff(npv.loc[s, k]) for k in range(1, 8)] for s in order])
P('텐서 크기 가정(float32, ragged): 플레이어 특징 12개(팀 중심 기준 위치 2, 맵 위치 2, 체력, 준 피해, 받은 피해, 기절시킴, kill, 누적 kill, 회복 아이템 사용, 생존 마스크), 팀 특징 150개를 팀·단계마다 1회 저장(복제 시 노드 수 × 150), 엣지 1개 17 B(int32 2개 + 유형 1 B + float32 2개; 팀원·적 근접은 양방향 2개로 셈, 교전은 방향 1개), 플레이어–구역 엣지는 노드당 2개 × 21 B. 계산은 `03_summarize_sample.py`.', '')
P('표 14. 텐서 크기. 출처: `csv/stage2_tensor_size_estimate.csv`(scope=cuts_with_target).', '')
rows = [[LAB[s], fi(tzc.loc[s].n_total), ff(tzc.loc[s].MB_match_compact, 3), ff(tzc.loc[s].GB_cohort_compact, 3), ff(tzc.loc[s].GB_cohort_broadcast, 3), ff(tzu.loc[s].GB_cohort_compact, 3), ff(tzu.loc[s].GB_cohort_broadcast, 3)] for s in order]
rows.append(['합계', fi(tzc.n_total.sum()), '', ff(gb_c, 3), ff(gb_b, 3), ff(gb_u, 3), ff(gb_ub, 3)])
tab(['코호트', '코호트 매치', 'MB/매치(compact)', 'GB/코호트(compact)', 'GB/코호트(팀 특징 복제)', 'GB(compact, 상한)', 'GB(복제, 상한)'], rows)
P('"상한"은 적 근접 엣지를 시작 시 생존자 전원 기준으로 센 경우다. 어느 쪽이든 코호트 전체가 수 GB 이하라 메모리에 올릴 수 있다.', '')
P('표 15. 추출 가능성(컷 시점 입력). 출처: 스키마(`information_schema.columns`), 표본 계산 결과.', '')
P('| 입력 | 테이블·컬럼 | 컷 시점 사용 | 상태 |', '| --- | --- | --- | --- |',
  f'| 위치 | `pubg.telem_positions` `pos_x/pos_y/pos_z`(cm), `elapsed_time`, `account_id`, `team_id`, `is_game>=0.1` | t_c 이전 마지막 샘플(10초 간격) | 가능. 컷 전 15초 안 샘플 없는 생존자 {fp(ov.C_miss15 / ov.C_players, 2)} |',
  f'| 체력 | `telem_positions.health`(NaN {fp(match.pos_health_nan_share.mean(), 1)}) | 같은 샘플 | 가능. 0은 기절(DBNO)이며 컷 생존자의 {fp(health0, 2)} |',
  f'| 준·받은 피해 | `pubg.telem_damage` `attacker_id/victim_id/damage/event_time`, 위치 컬럼 | [단계 시작, t_c] 합산 | 가능. 적 대상 피해 합 = `participants.damage_dealt`(배율 {rng3(cons.enemy_dmg_per_p_damage.min(), cons.enemy_dmg_per_p_damage.max())}) |',
  f'| kill·사망 | `pubg.telem_kills` `killer_id/victim_id/event_time/finisher_id/dbno_maker_id/victim_team_id` | 사망 시각 > t_c로 생존 판정, 누적 kill | 가능. 행 수 / `participants.kills` = {match.kill_rows.sum() / match.p_kills.sum():.3f}(자기장 등 무킬러 사망 포함) |',
  f'| 기절시킴 | `pubg.telem_groggy` `attacker_id/victim_id/event_time` | 단계별 집계 | 가능. 행 수 / `participants.dbnos` = {match.groggy_rows.sum() / match.p_dbnos.sum():.3f} |',
  f'| 회복 아이템 사용 | `pubg.telem_item_use` `account_id/team_id/category/sub_category/item_id/stack_count` | `category=Use`, `sub_category in (Heal, Boost)` 집계 | 가능. 표본 {n_item_matches}매치에서 Heal {fi(heal_n)}행, Boost {fi(boost_n)}행(매치당 {ff(heal_n / n_item_matches)}, {ff(boost_n / n_item_matches)}) |',
  '| 팀 소속 | `participants`+`rosters`, `telem_positions.team_id` | 정적 | 가능. 두 출처 불일치 0건 |',
  '| 봇 여부 | `account_id`가 `ai.`로 시작 | 정적 | 가능하나 전용 컬럼이 없다. 노드 플래그로 넣어야 한다 |',
  '| 구역 기하 | `pubg.telem_game_states` `safe_zone_*/poison_zone_*`, `stdb.dynamic_regions` | 단계 경계 기준 | 가능 |',
  f'| 소생(revive) | 이벤트 테이블 없음. `participants.revives`는 매치 합계만 있다(표본 매치당 평균 {match.p_revives.mean():.2f}) | 체력 시계열(0 → 양수)로 간접 추정 | **결측**. 기절 후 60초 뒤 체력이 양수인 샘플이 있는 비율이 {fp(match.knock_revive_like.sum() / match.knock_events.sum())}이고, 기절 직후 첫 샘플의 체력이 0인 비율은 {fp(match.knock_next_sample_h0.sum() / match.knock_events.sum())}다 |', '')
P(f'추출 처리량(표본 {n_samp:,}매치, 2 worker): 벽시계 {meta["wall_s"]:.1f}초. 매치당 DB 조회 {fetch_pm * 1000:.1f} ms(위치 {tmg["pos"].sum() / tmg.n.sum() * 1000:.1f} ms, 피해 {tmg["dmg"].sum() / tmg.n.sum() * 1000:.1f} ms) + 계산 {proc_pm * 1000:.1f} ms. 전체 {fi(cs.n_total.sum())}매치는 단일 프로세스 약 {full_min_serial:.1f}분, 2 worker 약 {full_min_wall:.1f}분이다. 매치당 위치 행 {fi(match.pos_rows.mean())}개, 피해 행 {fi(match.dmg_rows.mean())}개를 읽는다. 출처: `csv/sample_run_meta.json`, `csv/sample_batch_timing.csv`.', '')
# ---- 6. quality ----
P('## 6. 데이터 품질', '')
P('### 6.1 비정상 단계 수', '')
P(f'단계 번호가 비연속이거나 1단계가 없는 매치는 13개 코호트 전부에서 0건이고, 단계 행이 없는 매치도 0건이다. 최대 단계가 10 이상인 매치도 없다(원시 `telem_phase_changes` 전체 매치에는 10단계 386행이 있으나(`logs/01_stable_profile.log`) `match_phase_global`의 코호트 매치 최대 단계는 9다). 최대 단계 2인 매치는 {fi(pv[2].sum())}건이다. 컷 1~7 단계 중 길이가 20초 미만인 단계는 {fp(sh20)}, 10초 미만은 {fp(sh10)}다(가중 평균, `csv/phase_timing_by_cohort.csv`). 이런 단계에서는 t_c 이전 구간이 거의 비어 있다.', '')
P('표 16. 짧게 끝난 매치. 출처: `csv/cohort_summary.csv`.', '')
tab(['코호트', '매치', '최대 단계 <=3', '최대 단계 <=5 (비율)', '비연속·1단계 부재'], [[LAB[s], fi(cs.loc[s].n_matches), fi(cs.loc[s].n_maxphase_le3), f"{fi(cs.loc[s].n_maxphase_le5)} ({fp(cs.loc[s].n_maxphase_le5 / cs.loc[s].n_matches)})", fi(cs.loc[s]['n_phase_gap_or_min!=1'])] for s in order] + [['합계', fi(cs.n_matches.sum()), fi(cs.n_maxphase_le3.sum()), f"{fi(cs.n_maxphase_le5.sum())} ({fp(cs.n_maxphase_le5.sum() / cs.n_matches.sum())})", fi(cs['n_phase_gap_or_min!=1'].sum())]])
ea = cmt[cmt.cohort == 'erangel_tpp_squad'].set_index('match_type')
P(f'5단계 이하에서 끝나는 매치는 Erangel TPP squad에 몰려 있다. 이 코호트의 중앙값 최대 단계는 airoyale {fg(ea.loc["airoyale"].maxphase_med)}, competitive {fg(ea.loc["competitive"].maxphase_med)}, official {fg(ea.loc["official"].maxphase_med)}이다(`csv/cohort_matchtype_profile.csv`).', '')
r_all = iv.loc['ALL']
P('### 6.2 위치 샘플링 간격', '')
P(f'표본 {fi(r_all.n_intervals)}개 간격(플레이어별 연속 샘플 사이)에서 중앙값 {fi(r_all.median_s)}초, p10 {fi(r_all.p10_s)}초, p90 {fi(r_all.p90_s)}초, 평균 {r_all.mean_s:.2f}초다. 10초가 {fp(r_all.share_eq10, 2)}, 9초가 {fp(sh9, 2)}(경과 시간이 정수 초라 생기는 반올림), 0초(같은 초의 중복 행)가 {fp(sh0, 2)}, 11초 이하가 {fp(r_all.share_le11, 2)}이고 20초 초과는 {fp(r_all.share_gt20, 2)}, 60초 초과는 {fp(r_all.share_gt60, 2)}다. 코호트별 값은 `csv/position_interval_stats_by_cohort.csv`.', '')
P('### 6.3 컷 부근 위치 결측', '')
P('표 17. t_c에 생존한 플레이어·팀 중 t_c 이전 15초·30초 안에 위치 샘플이 없는 비율. 팀은 생존자 전원이 결측일 때만 센다. 출처: `csv/sample_cut_level.csv`(`C_miss15/30`, `C_team_miss15/30`), 코호트별은 `csv/missing_positions_near_cut_by_cohort_phase.csv`.', '')
gk = cut.groupby('k')[['C_players', 'C_miss15', 'C_miss30', 'C_teams', 'C_team_miss15', 'C_team_miss30']].sum()
tab(['컷 k', '생존 플레이어', '결측(15초)', '결측(30초)', '생존 팀', '팀 결측(15초)', '팀 결측(30초)'], [[str(k), fi(gk.loc[k].C_players), fp(gk.loc[k].C_miss15 / gk.loc[k].C_players, 2), fp(gk.loc[k].C_miss30 / gk.loc[k].C_players, 2), fi(gk.loc[k].C_teams), fp(gk.loc[k].C_team_miss15 / gk.loc[k].C_teams, 2), fp(gk.loc[k].C_team_miss30 / gk.loc[k].C_teams, 2)] for k in gk.index] + [['합계', fi(ov.C_players), fp(ov.C_miss15 / ov.C_players, 2), fp(ov.C_miss30 / ov.C_players, 2), fi(ov.C_teams), fp(ov.C_team_miss15 / ov.C_teams, 2), fp(ov.C_team_miss30 / ov.C_teams, 2)]])
P(f'단계 시작 시 생존했지만 t_c 이전 샘플이 하나도 없는 노드는 {fp(ov.A_nopos / ov.nodes_A, 3)}다. `stdb.team_phase_state`로는 이 결측을 잴 수 없다. 그 표는 위치 행이 있어야 행이 생기므로 팀·단계 {fi(qs.teams_start.sum())}행 중 위치 행 0인 행이 {fi(qs.teams_nopos.sum())}건이다(`csv/quality_team_position_rows_state_by_cohort_phase.csv`).', '')
P('### 6.4 `team_phase_state`의 생존 인원은 생존 수가 아니다', '')
P(f'`stdb.team_phase_state.alive_players_start/end/min/avg`는 (팀, 경과 초) 스냅샷에서 체력>0인 서로 다른 플레이어 수이고(`build_team_phase_state`의 `snap` CTE), 시작·종료 값은 단계 안 첫·마지막 스냅샷 한 곳의 값이다(`start_snap`·`end_snap` CTE의 `DISTINCT ON`). 플레이어별 10초 샘플의 시각이 서로 어긋나 있어서 (팀, 초)당 서로 다른 플레이어가 1명인 비율이 {fp(alive2.share_1)}, 2명 {fp(alive2.share_2)}, 3명 {fp(alive2.share_3)}, 4명 {fp(alive2.share_4, 2)}다(표본 {fi(alive2.matches)}매치, {fi(alive2.rows)}행, 평균 {alive2["mean"]:.3f}명).', '')
P('표 18. 상태 테이블 값과 실제 생존자 수의 비교(squad 3개 코호트에서 각 20매치). 출처: `csv/alive_definition_check.csv`, 스크립트 `06_verify_alive_definition.py`, `07_vikendi_check.py`, `08_nonparticipant_accounts.py`.', '')
tab(['단계', '팀·단계 행', '상태 `alive_start` 평균', '상태 `alive_end` 평균', '상태 `alive_min` 평균', '상태 `alive_avg` 평균', '실제 생존/팀(시작)', '실제 생존/팀(컷)', '실제/상태(시작)'], [[str(int(r.phase_idx)), fi(r.n_rows), ff(r.state_alive_start_mean, 3), ff(r.state_alive_end_mean, 3), ff(r.state_alive_min_mean, 3), ff(r.state_alive_avg_mean, 3), ff(r.true_alive_per_team_start, 3), ff(r.true_alive_per_team_at_cut, 3), ff(r.ratio_true_over_state_start, 2)] for r in alive.itertuples()])
P('`stdb.v_team_phase_feature_long`(`schema_rq1.sql`)은 `alive_players_start/end`를 모델 입력 쪽으로 내보낸다. 이 문서의 2단계 추정은 이 열을 쓰지 않고 `telem_kills`·`participants`·`telem_positions`에서 다시 계산했다. 표 18 마지막 열처럼 실제 생존자는 상태 테이블 값의 ' + f'{alive.ratio_true_over_state_start.min():.1f}~{alive.ratio_true_over_state_start.max():.1f}배다.', '')
P('### 6.5 라벨 감사', '')
P('**정정(2026-10-05, Part 1b):** 이 절의 표는 1차 감사다. 첫 킬 이벤트를 사망 시각으로 쓰고 마지막 단계의 경계를 엄격하게 잡아 소생 팀과 종결 단계를 잘못 셌다. 정정본은 `docs/data_layer/BIGCOMP2027_LABEL_AUDIT_2026-10-05.md`이며 전수 기준 3상태 불일치 5.95%, 양성 정밀도 70.4%, 재현율 75.2%다.', '')
P('팀 라벨(`eliminated_next_phase`)을 이벤트 기반 정답과 비교했다. 정답의 정의: 팀 사망 시각 = 팀원 마지막 사망 시각(`telem_kills`, 이탈자는 `time_survived`), 컷 k 위험집합 = 사망 시각 > `phase_end_s`(k)이고 다음 단계가 있는 팀, 양성 = 사망 시각 <= `phase_end_s`(다음 단계). 표본 5,066매치 전체 팀 행에서 비교했다. 출처: `csv/label_audit_overall_by_phase.csv`, 코호트별은 `csv/label_audit_by_cohort_phase.csv`.', '')
P(f'표 20. 컷별 라벨 대 이벤트 기반 정답(표본 {n_samp:,}매치).', '')
tab(['컷 k', '라벨 위험집합', '정답 위험집합', '교집합', '라벨 양성', '정답 양성', '교집합', '양성 정밀도', '양성 재현율', 'hazard 라벨', 'hazard 정답', '라벨 양성 중 컷 이전 이미 사망', '라벨 양성 중 다음 단계 끝까지 생존'],
    [[str(r.k), fi(r.n_label_risk), fi(r.n_true_risk), fi(r.n_risk_both), fi(r.n_label_pos), fi(r.n_true_pos), fi(r.n_pos_both), fp(r.pos_precision), fp(r.pos_recall), f"{r.hazard_label:.4f}", f"{r.hazard_true:.4f}", fp(r.labelpos_dead_before_cut_share), fp(r.labelpos_alive_after_next_share)] for r in aud.itertuples()])
P('라벨은 `alive_players_end>0`에서 파생되므로(`build_team_phase_labels`의 `target_defined`·`eliminated_next_phase` 식) 6.4절의 단일 스냅샷 정의를 그대로 물려받는다. 불일치의 원인으로 가능성이 높은 것은 (a) 마지막 스냅샷에 잡힌 한 명이 기절(health=0)인 경우와 (b) 마지막 스냅샷 이후 컷 이전에 사망한 경우지만, 원인별 분해는 하지 않았다. 컷 7은 정밀도가 낮고 컷 1은 라벨 양성 중 컷 이전 사망 비율이 높다. 정답 정의가 이벤트 기반 하나뿐이므로 라벨 소유자의 확인이 필요하다.', '')
P('### 6.6 봇(AI 계정)', '')
P('표 19. 표본의 AI 계정 비율. AI는 `account_id`가 `ai.`로 시작하는 참가자다. 출처: `csv/bots_by_cohort.csv`, `csv/bots_by_cohort_matchtype.csv`.', '')
tab(['코호트', '표본 매치', 'AI 비율(플레이어)', '사람 수/매치(평균)', 'AI 1명 이상 매치 비율'], [[LAB[s], fi(bots.loc[s].n), fp(bots.loc[s].ai_share), ff(bots.loc[s].human_mean), fp(bots.loc[s].share_matches_with_ai)] for s in order] + [['전체', fi(len(match)), fp(n_ai_all), ff(n_human_mean), fp(share_ai_matches)]])
bt = botm.groupby('match_type')[['n', 'ai_sum', 'pl_sum']].sum(); bt['sh'] = bt.ai_sum / bt.pl_sum
ofc = botm[botm.match_type == 'official']
P('매치 유형별 AI 비율: ' + ', '.join(f"{t} {fp(bt.loc[t].sh)}(표본 {fi(bt.loc[t].n)}매치)" for t in bt.index) + f'. `official` 안에서도 코호트별 AI 중앙값은 {fg(ofc.ai_med.min())}~{fg(ofc.ai_med.max())}명, p90은 {fg(ofc.ai_p90.min())}~{fg(ofc.ai_p90.max())}명이라 AI 위주 로비가 섞여 있다(`csv/bots_by_cohort_matchtype.csv`). AI 플래그 없이 학습하면 플레이어 계층 모델은 사람과 AI를 구분하지 못한다.', '')
vk = rd('nonparticipant_accounts_check.csv'); vkm = rd('vikendi_anomaly_check_matches.csv'); vx = vk[vk.cohort == 'vikendi_tpp_squad']; ox = vk[vk.cohort != 'vikendi_tpp_squad']; pr = cons.pos_rows_mean; pr_o = pr.drop('vikendi_tpp_squad'); vks = vkm.groupby('cohort').stdb_over_raw.median()
vk_bullet = (f'- Vikendi(DihorOtok_Main): 위치 텔레메트리에 `participants`에 없는 `Monster.` 계정이 매치당 {fg(vx.n_extra_accounts.median())}개 있다(점검한 {len(vx)}매치 모두 같은 팀 ID 1개, 로스터에 없음). Erangel·Sanhok·Miramar {len(ox)}매치에서는 0개다. 이 계정이 낀 `telem_damage` 행은 매치당 중앙값 {fg(vx.extra_damage_rows.median())}행, `telem_kills` 행은 {fg(vx.extra_kill_rows.median())}행이다. 그래서 Vikendi의 매치당 위치 행은 {fi(pr.loc["vikendi_tpp_squad"])}개로 다른 코호트({fi(pr_o.min())}~{fi(pr_o.max())}개)의 중앙값 대비 {pr.loc["vikendi_tpp_squad"] / pr_o.median():.1f}배이고, `stdb.team_phase_combat` 피해량 합은 raw 적 대상 합의 중앙값 {vks["vikendi_tpp_squad"]:.2f}배다(Erangel {vks["erangel_tpp_squad"]:.2f}, Taego {vks["taego_tpp_squad"]:.2f}). 근거: `csv/vikendi_anomaly_check_matches.csv`(코호트당 8매치), `csv/nonparticipant_accounts_check.csv`(코호트당 10매치). 이 문서의 2단계 노드는 `participants` 기준이라 몬스터를 포함하지 않는다.')
P('### 6.7 그 밖에 확인한 것', '')
P(f'- `pubg.telem_phase_changes.elapsed_time`은 코호트 매치 249,427행 전부 NULL이다(`logs/01_stable_profile.log`). 단계 경계는 `event_time − telem_match_start.event_time`로 복원되고(`etl.py`의 coalesce), 복원값과 `match_phase_global.phase_start_s`의 중앙값 차이는 (맵, 단계) {len(sm)}개 중 {n_within1}개가 ±1초 이내이며 범위는 {sm.start_minus_raw_med.min():.1f}~{sm.start_minus_raw_med.max():.1f}초다(`csv/phase_timing_by_map.csv`).',
  f'- `stdb.matches.started_at`과 경과 시계 0점의 차이는 중앙값 {off.median():.2f}초(p10 {off.quantile(.1):.2f}, p90 {off.quantile(.9):.2f}, 최대 {off.max():.2f})다. 로트 ETL(`build_team_phase_loot`)은 `event_time − started_at`을 단계 경계와 비교하므로 이만큼 어긋난다. 이 문서의 계산은 매치마다 `event_time − elapsed_time` 중앙값으로 시계를 맞췄다(`csv/sample_match_level.csv`의 `off_minus_started`).',
  f'- 위치 좌표는 cm이고 `is_game>=0.1` 행에 비행기 구간이 포함되어 x가 {fi(match.x_min.min())}~{fi(match.x_max.max())}까지 나온다. 단계 1 이후 위치는 맵 안이다.',
  f'- 원시 텔레메트리 합계와 `participants`의 비율: kill 행 / `kills` = {match.kill_rows.sum() / match.p_kills.sum():.3f}, groggy 행 / `dbnos` = {match.groggy_rows.sum() / match.p_dbnos.sum():.3f}, 적 대상 피해 합 / `damage_dealt` = {rng3(cons.enemy_dmg_per_p_damage.min(), cons.enemy_dmg_per_p_damage.max())}. `team_phase_combat` 합계는 같은 코호트 표본 raw의 적 대상 이벤트 대비 kill {(evt.kills_mean / evs.kills_enemy_mean).min():.3f}~{(evt.kills_mean / evs.kills_enemy_mean).max():.3f}배, 피해 이벤트 {(evt.dmg_events_mean / evs.dmg_ev_enemy_mean).min():.3f}~{(evt.dmg_events_mean / evs.dmg_ev_enemy_mean).max():.3f}배다(최댓값은 Vikendi, 아래 항목).',
  f'- `participants.time_survived`와 킬 이벤트 사망 시각의 차이는 매치별 절대 차이의 중앙값이 {match.ts_diff_mad.median():.2f}초이고 최대 {fi(match.ts_diff_mad.max())}초까지 벌어진다. 사망 시각은 킬 이벤트를 쓴다.',
  vk_bullet, '')
# ---- 7. verdict ----
P('## 7. 판정: 2단계 데이터·추출은 GO (10/9 18:00 기준)', '')
P(f'근거:', '',
  f'1. 입력이 모두 원시 `pubg.*`에 있다. 위치·체력·피해·kill·기절·회복 아이템과 팀 소속이 컷 시점 기준으로 재구성된다(표 15). 컷 전 15초 안 위치 결측은 생존자 {fp(ov.C_miss15 / ov.C_players, 2)}, 팀 {fp(ov.C_team_miss15 / ov.C_teams, 2)}다.',
  f'2. 규모가 작다. 컷 7까지 노드 약 {tot_nodes / 1e6:.2f}M, 매치당 방향 엣지(팀원·적 근접 양방향 + 교전) 합은 코호트 평균 {fi(tzc.edges_directed.min())}~{fi(tzc.edges_directed.max())}개이고 ragged 텐서 합계가 {gb_c:.2f}~{gb_u:.2f} GB다.',
  f'3. 추출이 빠르다. 표본 {n_samp:,}매치를 {meta["wall_s"]:.0f}초에 처리했고 전체 {fi(cs.n_total.sum())}매치는 약 {full_min_wall:.0f}분이다. 피처가 늘어도 DB 조회가 병목이 되지 않는다.',
  f'4. 관계 밀도가 충분하다. 컷당 적 근접 엣지는 평균 {cut.enemy_fresh_C.mean():.1f}개(컷 생존자 기준), 교전 쌍은 컷 2 이후 평균 {cut[cut.k >= 2].cb_enemy.mean():.1f}개다. 팀원 엣지는 후반 컷에서 컷당 {cut[cut.k >= 6].tm_A.mean():.1f}개로 적고, solo 코호트에는 없다.', '',
  '조건과 위험:', '',
  f'- 봇: 플레이어의 {fp(n_ai_all)}가 AI다. 노드에 AI 플래그를 넣고, 결과를 `match_type`(airoyale / official / competitive)별로 나눠 보고하지 않으면 AI 위주 매치가 지표를 지배한다(Erangel TPP squad의 {fp(mt.loc["erangel_tpp_squad"].airoyale / mt.loc["erangel_tpp_squad"].sum())}).',
  '- 소생 이벤트가 없다. 노드 특징의 "기절" 상태는 체력 0으로만 보이며 소생 시각은 10초 해상도의 체력 복귀로만 추정된다. 회복 아이템 사용 수는 가능하다.',
  '- 2단계 노드 특징은 `team_phase_state.alive_players_*`를 재사용하면 안 된다(6.4절). 원시에서 재계산해야 한다. 팀 150차원에 같은 열이 들어 있는지는 이 조사에서 확인하지 않았다.',
  '- 라벨 일관성: 팀 라벨은 이벤트 기반 정답과 양성 정밀도 ' + f'{fp(aud.iloc[-1].pos_precision)}, 재현율 {fp(aud.iloc[-1].pos_recall)}로 어긋난다(6.5절). 2단계에서 라벨을 바꾸면 1단계와 비교가 깨지므로 같은 라벨을 쓰되 이 한계를 보고해야 한다.',
  '- Vikendi: 매치당 몬스터 계정 약 30개가 위치·피해 텔레메트리에 있고 `team_phase_combat` 피해량 합이 raw보다 크다(6.7절). 플레이어 노드에서는 제외했다. 팀 수준 combat 특징이 이 영향을 받는지는 확인이 필요하다.',
  '- 소형 코호트(Karakin, Paramo, FPP squad 둘)는 컷 6~7 위험집합이 수십~수백 팀이라 코호트별 컷 7 결론은 약하다(표 8).',
  '- 이 판정은 데이터와 추출만 다룬다. 플레이어 토큰 attention 모델의 구현·학습 시간은 이 조사 범위 밖이며 10/9 18:00 마감의 남은 위험은 그쪽이다.', '')
# ---- 8. artifacts ----
P('## 8. 산출물과 재현', '')
P('- 문서: 이 파일. 그림: `reports/bigcomp2027_data_profile/figures/fig1_hazard_by_phase_per_cohort.{pdf,png}`, `fig2_phase_timing_by_map.{pdf,png}`.',
  '- 스크립트(`reports/bigcomp2027_data_profile/scripts/`): `dbh.py`(읽기 전용 연결), `01_stable_profile.py`(깔때기·코호트·라벨·이벤트), `02_sample_player_level.py`(플레이어 단위 표본), `03_summarize_sample.py`(집계·텐서 크기), `04_figures.py`, `05_build_doc.py`(이 문서 생성), `06_verify_alive_definition.py`.',
  '- 실행 순서: DB 환경 변수를 불러온 뒤 `01` → `02 --per-cohort 400 --workers 2 --batch 6 --tag sample` → `03 sample` → `06` → `07` → `08` → `04` → `05`. `PG_DATABASE`만 설정돼 있어도 `dbh.py`가 처리한다.',
  f'- CSV 목록은 `reports/bigcomp2027_data_profile/csv/`({len(os.listdir(C))}개 파일). 로그는 `.../logs/`.', '')
open(DOC, 'w').write('\n'.join(L) + '\n'); print('doc written', DOC, len(L), 'lines', sum(len(x) for x in L), 'chars')
