"""``tools/paper_numbers_v2.py``: the dump-dependent numbers of the paper drafts with their provenance.  The pure cores are tested on hand-made data, the stages on a
synthetic v2 dump, and -- where the real v1 files are on this disk -- against the documented v1 values; the database stages run on a few v1 matches when the
credentials are in the environment (read-only)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

import pubg_lab.survival.ptr.data.paths as P
from tests.conftest import requires_db
from tests.test_rerun_paths import make_dump

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import projects.battle_royale.reporting.paper_numbers_v2 as PN  # noqa: E402

V1_DUMP = REPO / 'model_stgcn' / "dumps" / "bigcomp2027_phase_v1"
V1_RESULTS = REPO / "reports" / "bigcomp2027_results"
PROFILE = REPO / "reports" / "bigcomp2027_data_profile" / "csv"
needs_v1 = pytest.mark.skipif(not (V1_DUMP / "population.json").is_file(), reason="the v1 dump is not on this disk")


# --------------------------------------------------------------------------- pure cores
def test_population_numbers_sum_the_kept_cohorts():
    pop = {"rules": {"min_matches_per_cohort": 200}, "cohorts": {
        "a_squad": {"status": "kept", "before": {"total": 700}, "after": {"train": 480, "val": 60, "test": 60, "total": 600}, "match_type_after": {"official": 500, "competitive": 100}},
        "b_duo": {"status": "kept", "before": {"total": 260}, "after": {"train": 160, "val": 20, "test": 20, "total": 200}, "match_type_after": {"official": 200}},
        "c_solo": {"status": "excluded", "reason": "fewer than 200 matches", "before": {"total": 90}, "after": {"train": 0, "val": 0, "test": 0, "total": 0}},
    }}  # fmt: skip
    n = PN.population_numbers(pop)
    assert n["n_matches"] == 800 and n["n_cohorts"] == 2 and n["by_split"] == {"train": 640, "val": 80, "test": 80}
    assert n["match_types"] == {"official": 700, "competitive": 100} and n["size_min"] == 200 and n["size_max"] == 600 and n["smallest"] == "b_duo"
    assert n["two_largest"] == ["a_squad", "b_duo"] and n["excluded"] == {"c_solo": "fewer than 200 matches"} and n["before_total"] == 1050


@needs_v1
def test_the_v1_population_reproduces_the_documented_numbers():
    n = PN.population_numbers(json.loads((V1_DUMP / "population.json").read_text()))
    assert (n["n_matches"], n["n_cohorts"]) == (13_948, 12) and n["by_split"] == {"train": 11_188, "val": 1_376, "test": 1_384}
    assert n["match_types"] == {"official": 11_285, "competitive": 2_663} and (n["size_min"], n["size_max"]) == (224, 4_159)
    assert n["two_largest"] == ["taego_tpp_squad", "erangel_tpp_squad"]


def test_interval_stats_follow_the_profile_definitions():
    h = np.zeros(121)
    h[10], h[11], h[12], h[30] = 90, 6, 3, 1
    s = PN.interval_stats(h)
    assert s["n_intervals"] == 100 and s["median_s"] == 10 and s["share_eq10"] == 0.9 and s["share_le11"] == 0.96  # floor(interval) <= 11 s
    assert s["share_gt20"] == 0.01 and s["share_gt60"] == 0.0 and s["p99_s"] == 12 and PN.interval_stats(np.zeros(121)) == {"n_intervals": 0}


@pytest.mark.skipif(not (PROFILE / "sample_position_interval_hist.csv").is_file(), reason="the v1 data profile is not on this disk")
def test_interval_stats_reproduce_the_v1_profile_row():
    hist = pd.read_csv(PROFILE / "sample_position_interval_hist.csv")
    h = np.zeros(121)
    for r in hist.itertuples():
        h[int(r.interval_s)] += r.n
    got = PN.interval_stats(h)
    ref = pd.read_csv(PROFILE / "position_interval_stats_by_cohort.csv").set_index("cohort").loc["ALL"]
    assert got["n_intervals"] == int(ref.n_intervals) and got["median_s"] == ref.median_s
    for key in ("share_le11", "share_eq10", "share_gt20", "share_gt60", "mean_s"):
        assert got[key] == pytest.approx(float(ref[key]), abs=1e-12), key


def test_clock_stats():
    s = PN.clock_stats(np.array([3.0, 5.0, 7.0, 9.0, 41.0, np.nan]))
    assert s["n_matches"] == 5 and s["median_s"] == 7.0 and s["max_s"] == 41.0 and s["min_s"] == 3.0 and s["p95_s"] == pytest.approx(np.quantile([3, 5, 7, 9, 41], 0.95))
    assert PN.clock_stats(np.array([np.nan])) == {"n_matches": 0}


def test_player_cut_coverage_counts_players_without_a_recent_position_sample():
    w = np.array([1.0, 5.0, 9.0, 100.0, 3.0, 7.0])  # account 0: 1, 5, 9, 100 s; account 1: 3, 7 s; account 2 has no sample
    a = np.array([0, 0, 0, 0, 1, 1])
    r = PN.player_cut_coverage(w, a, [(0, 10.0), (0, 12.0), (1, 25.0), (2, 10.0)])
    assert r == {"n_alive": 4, "missing": {"15": 2, "30": 1}, "without_any_sample": 1}  # acct 1 is 18 s stale at 25 s, acct 2 never reported
    # a sample AT the cut does not count (the inputs of a cut are strictly before it)
    assert PN.player_cut_coverage(np.array([10.0]), np.array([0]), [(0, 10.0)])["without_any_sample"] == 1
    assert PN.player_cut_coverage(np.array([10.0]), np.array([0]), [(0, 10.001)])["missing"]["15"] == 0


def test_spearman_pairs_use_the_outside_depth_of_the_signed_distance():
    rng = np.random.default_rng(0)
    d = rng.normal(0, 300, 5000)
    depth = np.maximum(0, -d)
    req = 0.01 * depth + rng.normal(0, 1, 5000) + 5.0
    deficit = 0.2 * req + rng.normal(0, 2, 5000)
    cols = {"hab_required_speed_to_safe_mps": req, "hab_signed_boundary_distance_end_m": d, "hab_migration_deficit_mps": deficit}
    r = PN.spearman_pairs(cols)
    assert r["n_rows"] == 5000 and r["spearman_required_speed_outside_depth"] == pytest.approx(stats.spearmanr(req, depth)[0])
    assert r["spearman_migration_deficit_outside_depth"] == pytest.approx(stats.spearmanr(deficit, depth)[0])
    assert r["spearman_required_speed_migration_deficit"] == pytest.approx(stats.spearmanr(req, deficit)[0]) and r["pearson_required_speed_outside_depth"] == pytest.approx(np.corrcoef(req, depth)[0, 1])
    cols["hab_required_speed_to_safe_mps"] = np.where(np.arange(5000) % 100 == 0, np.nan, req)  # non-finite rows are dropped
    assert PN.spearman_pairs(cols)["n_rows"] == 4950


def test_independent_team_death_and_label_agreement():
    kills = {"p1": 100.0, "p2": 130.0}
    # first kill event, else time_survived + offset unless alive, the team dies with its last member
    assert PN.independent_team_death([("p1", "byplayer", 99.0), ("p2", "byplayer", 129.0)], kills, 0.7) == 130.0
    assert PN.independent_team_death([("p1", "byplayer", 99.0), ("p3", "logout", 200.0)], kills, 0.7) == 200.7
    assert PN.independent_team_death([("p1", "byplayer", 99.0), ("p4", "alive", None)], kills, 0.7) == math.inf and PN.independent_team_death([], kills, 0.7) == math.inf
    # two teams, two valid cuts: team 0 dies at 150 s, team 1 survives
    truth = np.array([150.0, math.inf])
    boundary = np.array([100.0, 200.0, np.nan])
    window_end = np.array([200.0, 300.0, np.nan])
    dump_target = np.array([[True, True], [False, True], [False, False]])
    dump_y = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    valid = np.array([True, True, False])
    r = PN.label_agreement(truth, boundary, window_end, dump_target, dump_y, valid)
    # cut 1: both teams at risk, team 0 positive (150 <= 200); cut 2: only team 1 at risk (team 0 died before 200), no positive
    assert r == {"rows": 4, "risk_true": 3, "risk_dump": 3, "risk_both": 3, "pos_true": 1, "pos_dump": 1, "pos_both": 1}
    wrong = PN.label_agreement(truth, boundary, window_end, dump_target, np.zeros_like(dump_y), valid)
    assert wrong["pos_dump"] == 0 and wrong["pos_true"] == 1 and wrong["pos_both"] == 0


def test_cut_arrays_for_v2_and_v1_records():
    m2 = {"cut_t": np.array([10.0, 20.0, np.nan]), "cut_valid": np.array([True, False, False])}
    cut, valid = PN.cut_arrays(m2)
    assert cut[0] == 10.0 and valid.tolist() == [True, False, False]
    m1 = {"cut_t": np.array([1.0, 2.0, 3.0]), "cut_wall_s": np.array([11.0, 12.0, np.nan]), "phase_mask": np.array([True, True, False])}
    cut, valid = PN.cut_arrays(m1)
    assert cut.tolist()[:2] == [11.0, 12.0] and valid.tolist() == [True, False, False]  # v1: the W-clock cut, valid iff phases k and k + 1 exist


# --------------------------------------------------------------------------- the stages on a synthetic v2 dump
def test_the_dump_stages_write_every_number_with_its_provenance(tmp_path):
    v2 = make_dump(tmp_path / "bigcomp2027_phase_v2", version="v2", sizes={"train": 6, "val": 4, "test": 4})
    nums = PN.run_stages(PN.Ctx(v2, seed=1), ["population", "risk_set"])
    assert nums.errors == []
    n = nums.numbers
    assert n["population.n_matches"]["value"] == 28 and n["population.matches_train"]["value"] == 12 and n["risk_set.n_matches_dumped"]["value"] == 28
    # the risk set is what select_labels returns: recount by hand
    from pubg_lab.survival.ptr.data.data import select_labels

    rows = pos = 0
    by_cut = np.zeros(7, dtype=int)
    for c in ("c_alpha", "c_beta"):
        for s in ("train", "val", "test"):
            for m in torch.load(v2 / c / f"{s}.pt", weights_only=False)["matches"]:
                y, t = select_labels(m, "v2")
                rows += int(t.sum())
                pos += int(((y > 0.5) & t).sum())
                by_cut += t.numpy()[:7].sum(axis=1)
    assert n["risk_set.n_rows"]["value"] == rows > 0 and n["risk_set.n_positives"]["value"] == pos > 0 and n["risk_set.rows_by_cut"]["value"] == by_cut.tolist()
    assert n["risk_set.positive_rate"]["value"] == pytest.approx(pos / rows) and sum(n["risk_set.positives_by_cut"]["value"]) == pos
    assert n["risk_set.n_rows_train"]["value"] + n["risk_set.n_rows_val"]["value"] + n["risk_set.n_rows_test"]["value"] == rows
    assert n["risk_set.comeback_rows"]["value"] >= 0 and n["features.version"]["value"] == "v2" and n["risk_set.meta_stats_agree"]["value"] is True
    for key, rec in n.items():  # provenance on every number
        prov = rec["provenance"]
        assert prov["stage"] in PN.STAGES and prov["source"] in ("dump", "db", "runs", "results") and prov["definition"] and isinstance(prov["inputs"], list), key
    assert nums.not_computed and all(x["reason"] for x in nums.not_computed)
    assert {"edge_by_phase_elimination_rates", "ai_accounts.airoyale_share"} <= {x["key"] for x in nums.not_computed}


def test_main_writes_strict_json_and_refuses_what_it_should(tmp_path, capsys):
    v2 = make_dump(tmp_path / "dumps" / "bigcomp2027_phase_v2", version="v2")
    out = tmp_path / "reports" / "bigcomp2027_results_v2" / "paper_numbers.json"
    argv = ["--dump-dir", str(v2), "--runs-dir", str(tmp_path / "none"), "--results-dir", str(out.parent), "--out", str(out), "--stages", "population,risk_set,headline", "--no-db"]
    assert PN.main(argv) == 0
    doc = json.loads(out.read_text(), parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))
    assert doc["meta"]["dump_version"] == "v2" and doc["meta"]["stages"] == ["population", "risk_set", "headline"] and doc["meta"]["dump_meta_sha1"] == P.dump_meta_sha1(v2)
    assert doc["meta"]["git"]["commit"] is not None or doc["meta"]["git"]["commit"] is None  # recorded, whatever the checkout
    assert doc["numbers"]["population.n_matches"]["value"] == 22 and doc["errors"] == []  # 2 cohorts x (5 + 3 + 3) matches
    assert any(x["key"] == "headline" and "no aggregator output" in x["reason"] for x in doc["not_computed"])  # the headline stage needs the aggregator's outputs
    # a v1 dump needs --allow-v1; a v1 results directory next to the v2 dump is refused
    legacy = make_dump(tmp_path / 'archive/legacy_snapshot')
    with pytest.raises(SystemExit, match="not v2"):
        PN.main(["--dump-dir", str(legacy), "--results-dir", str(tmp_path / "r1"), "--stages", "population", "--no-db"])
    assert PN.main(["--dump-dir", str(legacy), "--results-dir", str(tmp_path / "r1"), "--out", str(tmp_path / "legacy.json"), "--stages", "risk_set", "--no-db", "--allow-v1"]) == 0
    with pytest.raises(SystemExit, match="refusing to run"):
        PN.main(["--dump-dir", str(v2), "--results-dir", str(tmp_path / "reports" / "bigcomp2027_results_v1"), "--stages", "population", "--no-db"])
    with pytest.raises(SystemExit, match="unknown stage"):
        PN.main(["--dump-dir", str(v2), "--stages", "nope", "--no-db"])
    # a failing stage is recorded, the others still run, the exit code says so
    ctx = PN.Ctx(tmp_path / "does_not_exist")
    nums = PN.run_stages(ctx, ["population"])
    assert nums.errors and nums.errors[0]["stage"] == "population" and nums.numbers == {}
    assert PN.resolve_stages("all", no_db=True, have_db=True) == [s for s in PN.STAGES if s not in PN.DB_STAGES]
    assert PN.resolve_stages("all", no_db=False, have_db=False) == [s for s in PN.STAGES if s not in PN.DB_STAGES]
    assert "coverage" in PN.resolve_stages("coverage,intervals", no_db=False, have_db=False)  # an explicit database stage is kept: its failure is reported, not hidden


def test_the_headline_stage_reads_the_aggregator_outputs(tmp_path):
    import pubg_lab.survival.ptr.evaluation.aggregate as A
    from tests.test_ptr_aggregate import union_runs

    root = union_runs(tmp_path / "runs", norank=True)
    A.aggregate(root, tmp_path / "res", n_boot=30, seed=1, workers=1, figures=False)
    v2 = make_dump(tmp_path / "bigcomp2027_phase_v2", version="v2")
    nums = PN.run_stages(PN.Ctx(v2, results_dir=tmp_path / "res"), ["headline"])
    assert nums.errors == []
    n = nums.numbers
    cp = pd.read_csv(tmp_path / "res" / "contrasts_pooled.csv")
    row = cp[(cp.contrast == "k3a_vs_k2h0") & (cp.metric == "auc") & (cp.stratum == "all")].iloc[0]
    got = n["headline.k3a_vs_k2h0.auc"]["value"]
    assert got["theta"] == pytest.approx(row.theta) and got["ci95"] == pytest.approx([row.ci95_low, row.ci95_high]) and got["k"] == 3 and got["k_total"] == 3
    assert n["headline.k3a_vs_k2h0.auc"]["provenance"]["source"] == "results"
    for name in ("k2h0_vs_k2", "m01norank_vs_k3", "m11rnorank_vs_m01norank", "m01_vs_m01norank", "m11r_vs_krafton_tf_h0", "hgb_union_vs_hgbcut", "phase_main"):
        assert f"headline.{name}.auc" in n, name
    assert n["headline.team_main.spearman"]["value"]["verdict"] in ("gain", "equivalent", "inconclusive")
    means = pd.read_csv(tmp_path / "res" / "arm_metrics_mean.csv")
    m = float(means[(means.arm == "M11R") & (means.metric == "auc") & (means.stratum == "all")]["mean"].iloc[0])
    assert n["headline.mean_auc.M11R"]["value"] == pytest.approx(m)
    assert "headline.hab_increment" in {x["key"] for x in nums.not_computed}  # the Sagong hab arm is not part of these runs


# --------------------------------------------------------------------------- real v1 files, where they are on this disk
@needs_v1
def test_the_v1_risk_set_of_two_cohorts_equals_the_sagong_plan_audit(tmp_path):
    plan = REPO / 'model_stgcn' / "dumps" / "bigcomp2027_sagong_v1"
    cohorts = ["paramo_tpp_squad", "karakin_tpp_squad"]
    if not all((plan / c / "plan_audit.json").is_file() for c in cohorts):
        pytest.skip("the Sagong v1 plan audit is not on this disk")
    nums = PN.run_stages(PN.Ctx(V1_DUMP, cohorts=cohorts), ["risk_set"])
    assert nums.errors == []
    audit = [json.loads((plan / c / "plan_audit.json").read_text())["splits"] for c in cohorts]
    assert nums.get("risk_set.n_rows") == sum(sp["n_target"] for a in audit for sp in a.values())
    assert nums.get("risk_set.n_positives") == sum(sp["n_pos"] for a in audit for sp in a.values())
    assert nums.get("risk_set.raw_equals_loader") in (True, False) and nums.get("features.version") == "v1"
    assert "risk_set.comeback_rows" in {x["key"] for x in nums.not_computed}  # a v1 record has no query mask


@needs_v1
def test_the_v1_headline_numbers_are_reproduced_from_the_whole_v1_dump():
    nums = PN.run_stages(PN.Ctx(V1_DUMP), ["population", "risk_set", "correlations"])
    assert nums.errors == []
    assert nums.get("risk_set.n_rows") == 829_640 and nums.get("risk_set.n_positives") == 220_171 and nums.get("risk_set.positive_rate") == pytest.approx(0.2654, abs=5e-5)
    assert nums.get("risk_set.n_matches_dumped") == 13_948 and nums.get("features.n_features") == 150
    assert nums.get("correlations.spearman_required_speed_outside_depth") == pytest.approx(0.795, abs=1e-3)  # the text says 0.79
    assert nums.get("correlations.spearman_migration_deficit_outside_depth") == pytest.approx(0.386, abs=1e-3)  # the text says 0.41: no producer of 0.41 exists
    assert len(nums.get("risk_set.hazard_by_cut")) == 7


@pytest.mark.skipif(not (V1_RESULTS / "contrasts_pooled.csv").is_file(), reason="the v1 results are not on this disk")
def test_the_headline_stage_on_the_v1_results():
    nums = PN.run_stages(PN.Ctx(V1_DUMP if V1_DUMP.exists() else REPO, results_dir=V1_RESULTS), ["headline"])
    assert nums.errors == []
    cp = pd.read_csv(V1_RESULTS / "contrasts_pooled.csv")
    row = cp[(cp.contrast == "m11r_vs_krafton_tf") & (cp.metric == "auc") & (cp.stratum == "all")].iloc[0]
    assert nums.get("headline.m11r_vs_krafton_tf.auc")["theta"] == pytest.approx(row.theta) and 0.05 < row.theta < 0.07  # the blueprint's +0.062


# --------------------------------------------------------------------------- the database stages on a few v1 matches (read-only)
@pytest.fixture(scope="module")
def db_ctx():
    if not V1_DUMP.is_dir():
        pytest.skip("the v1 dump is not on this disk")
    return PN.Ctx(V1_DUMP, cohorts=["paramo_tpp_squad"], limit_matches=3, per_cohort=3, audit_per_cohort=3, lookahead_n=12, workers=1, seed=1)


@requires_db
def test_record_counts_of_a_few_matches_match_a_direct_count(db_ctx):
    nums = PN.run_stages(db_ctx, ["db_counts"])
    assert nums.errors == []
    ids = db_ctx.all_ids()
    cur = db_ctx.conn().cursor()
    cur.execute("SELECT count(*) FROM pubg.telem_kills WHERE match_id = ANY(%s)", (ids,))
    assert nums.get("records.kills") == cur.fetchone()[0] > 0
    assert nums.get("records.total") == sum(nums.get(f"records.{k}") for k, _, _ in PN.TELEMETRY_TABLES) and nums.get("records.events") == nums.get("records.total") - nums.get("records.positions")
    assert nums.get("matches.found_in_db") == len(ids) and nums.get("matches.created_first") <= nums.get("matches.created_last")


@requires_db
def test_interval_clock_and_coverage_stages_on_a_few_v1_matches(db_ctx):
    nums = PN.run_stages(db_ctx, ["intervals", "clock", "coverage", "label_audit"])
    assert nums.errors == []
    assert nums.get("intervals.median_s") == 10 and nums.get("intervals.share_within_11s") > 0.99 and nums.get("intervals.n_intervals") > 10_000
    assert 0 < nums.get("elapsed_time_offset.median_s") < 20 and nums.get("elapsed_time_offset.n_matches") == 3
    assert "elapsed_time FIELD only" in nums.numbers["elapsed_time_offset.median_s"]["provenance"]["note"]  # v2 is on W: the offset only describes the field
    assert nums.get("coverage.n_alive_players") > 100 and 0.95 < nums.get("coverage.share_with_sample_15s") <= 1.0
    assert nums.get("coverage.share_with_sample_30s") >= nums.get("coverage.share_with_sample_15s")
    assert nums.get("label_audit.risk_set_agreement") > 0.9 and nums.get("label_audit.positive_recall") > 0.9


@requires_db
def test_the_lookahead_stage_runs_the_builders_prefix_replay(db_ctx):
    nums = PN.run_stages(db_ctx, ["lookahead"])
    assert nums.errors == []
    assert nums.get("lookahead.n_matches") == 12 and nums.get("lookahead.n_cuts") > 12 and nums.get("lookahead.n_mismatches") == 0
    assert nums.get("lookahead.min_gap_s") > 0 and nums.get("lookahead.mutation_detected")["cuts_detected_as_leaky"] == nums.get("lookahead.mutation_detected")["cuts_compared"]


def test_the_sample_is_deterministic_per_cohort_and_zero_means_every_match(tmp_path):
    v2 = make_dump(tmp_path / "bigcomp2027_phase_v2", version="v2", sizes={"train": 8, "val": 4, "test": 4})
    ctx = PN.Ctx(v2, seed=7)
    a, b = ctx.sample(5), PN.Ctx(v2, seed=7).sample(5)
    assert a == b and len(a) == 10 and {c for c, _, _ in a} == {"c_alpha", "c_beta"} and sum(1 for c, _, _ in a if c == "c_alpha") == 5  # per cohort, seeded
    assert PN.Ctx(v2, seed=8).sample(5) != a
    every = ctx.sample(0)
    assert len(every) == 32 and len({m for _, _, m in every}) == 32 and every == ctx.sample(-1)  # per_cohort <= 0: the whole corpus, no sampling
    assert len(ctx.sample(100)) == 32  # more than a cohort has: all of it
    limited = PN.Ctx(v2, seed=7, limit_matches=3)
    assert len(limited.sample(0)) == 6 and len(limited.all_ids()) == 3
