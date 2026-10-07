"""``tools/rerun_v2.py``: the plan (the final step list), the budget, the version guard, the dry run (it prints the plan and writes nothing), the gate, the lane
runner (resumable, one log per step) and the run manifest.  No step of the real plan is executed: the runner is exercised with small stand-in commands."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from model_stgcn.ptr import paths as P
from tests.test_rerun_paths import make_dump

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import rerun_v2 as R  # noqa: E402

FINAL_STEPS = [
    "gate.dump", "gate.leak_tests",
    "cpu.hgb", "cpu.sagong_plan", "cpu.sagong_extract", "cpu.sagong_habcols", "cpu.sagong_fit", "cpu.ehm_fit", "cpu.ehm_rq1", "cpu.ehm_rq1_physical",
    "cpu.ptr_main", "cpu.ptr_norank", "cpu.union",
    "mps.labels", "mps.extract", "mps.validate_causal", "mps.kftable", "mps.hgbkf", "mps.k2", "mps.k3", "mps.k1", "mps.h0",
    "analysis.aggregate", "analysis.supplement", "reports.ehm_report", "tables.tab_decomp",
    "tables.kftable_h0_report", "tables.kftable_k3_contrasts", "tables.kftable_k3_facts", "tables.kftable_phase_auc", "tables.make_paper_tables", "numbers.paper_numbers",
]  # fmt: skip


def cfg_under(tmp_path, **kw) -> R.Config:
    paths = P.RerunPaths.under(tmp_path)
    return R.Config(paths=paths, cohorts=("paramo_tpp_squad", "karakin_tpp_squad", "erangel_tpp_squad", "taego_tpp_squad"), **kw)


def tree(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")} if root.exists() else set()


# --------------------------------------------------------------------------- the plan
def test_the_final_step_list_and_the_order_of_the_manifest(tmp_path):
    steps = R.build_plan(cfg_under(tmp_path))
    assert [s.id for s in steps] == FINAL_STEPS
    lanes = {}
    for s in steps:
        lanes.setdefault(s.lane, []).append(s.id)
    assert set(lanes) == {"main", "cpu", "mps0"}
    # 1 gate, 2 CPU: HGB family, Sagong, EHM / RQ1, PTR main + seeds, norank (then the union), 3 MPS: features, K2 / K3a / K3, official arms, H0, 4-7 analysis
    cpu = lanes["cpu"]
    assert cpu.index("cpu.hgb") < cpu.index("cpu.sagong_plan") < cpu.index("cpu.sagong_fit") < cpu.index("cpu.ehm_fit") < cpu.index("cpu.ehm_rq1_physical") < cpu.index("cpu.ptr_main") < cpu.index("cpu.ptr_norank")
    mps = lanes["mps0"]
    assert mps.index("mps.labels") < mps.index("mps.extract") < mps.index("mps.kftable") < mps.index("mps.k2") < mps.index("mps.k3") < mps.index("mps.k1") < mps.index("mps.h0")
    main = lanes["main"]
    assert main.index("gate.dump") < main.index("gate.leak_tests") < main.index("analysis.aggregate") < main.index("analysis.supplement") < main.index("tables.tab_decomp") < main.index(
        "tables.make_paper_tables") < main.index("numbers.paper_numbers")  # fmt: skip
    by = {s.id: s for s in steps}
    for s in steps:
        if s.id.startswith(("cpu.", "mps.")):
            assert "gate.leak_tests" in by[s.id].deps or any(d.split(".")[0] in ("cpu", "mps") for d in s.deps) or s.id.startswith(("cpu.hgb", "mps.labels")), s.id
    assert set(by["analysis.aggregate"].deps) == {s.id for s in steps if s.lane != "main"}  # the analysis waits for every producer
    assert by["analysis.supplement"].deps == ("analysis.aggregate",)  # the supplement reads the aggregator's outputs
    assert {"cpu.hgb"} <= set(by["mps.k1"].deps) and {"mps.k3", "mps.k1"} <= set(by["mps.h0"].deps) and {"mps.kftable", "cpu.ptr_norank"} <= set(by["cpu.union"].deps)
    assert by["cpu.sagong_fit"].deps == ("cpu.sagong_habcols",) and "cpu.hgb" in by["cpu.sagong_plan"].deps  # habcols / fit need the HGB runs
    db = {s.id for s in steps if s.needs_db}
    assert db == {"cpu.sagong_extract", "mps.extract", "mps.validate_causal", "mps.kftable", "numbers.paper_numbers"}


def ancestors(steps, sid) -> set[str]:
    by = {s.id: s for s in steps}
    seen: set[str] = set()
    todo = list(by[sid].deps)
    while todo:
        d = todo.pop()
        if d not in seen:
            seen.add(d)
            todo.extend(by[d].deps)
    return seen


def test_hgb_cut_runs_first_every_sagong_and_krafton_stage_waits_for_it(tmp_path):
    """The Sagong and KRAFTON stages assert their rows against HGB-cut's (its prediction files): the HGB family is the first producer, and nothing but the gate
    comes before it."""
    for streams in (1, 2):
        steps = R.build_plan(cfg_under(tmp_path, mps_streams=streams, with_lc=True))
        by = {s.id: s for s in steps}
        assert by["cpu.hgb"].deps == ("gate.leak_tests",) and "--only" in by["cpu.hgb"].cmd and by["cpu.hgb"].cmd[by["cpu.hgb"].cmd.index("--only") + 1] == "group=hgb"
        for s in steps:
            if not s.id.startswith("gate.") and s.id != "cpu.hgb":
                assert "cpu.hgb" in ancestors(steps, s.id), f"{s.id} can start before the HGB family has finished"  # the KRAFTON chain (labels first) and the Sagong chain
        sim = R.simulate(steps)
        first_end = sim["cpu.hgb"][1]
        assert all(start >= first_end for sid, (start, _) in sim.items() if not sid.startswith("gate.") and sid != "cpu.hgb")
    skipped = R.build_plan(cfg_under(tmp_path, skip_leak_tests=True))
    assert next(s for s in skipped if s.id == "cpu.hgb").deps == ("gate.dump",) and "cpu.hgb" in next(s for s in skipped if s.id == "mps.labels").deps


def test_stage_2_the_player_layer_is_not_part_of_the_plan(tmp_path, capsys):
    """M11RZ-P / M11RZ-P-tm need the player dump, which is not rebuilt; the paper does not use them."""
    from model_stgcn.ptr.arms import STAGE2_ARMS

    assert STAGE2_ARMS == ("M11RZ-P", "M11RZ-P-tm") and R.PLAYER_LAYER_TOKENS == ("--stage2", "--player-dump-dir", *STAGE2_ARMS)
    for streams in (1, 2):
        for kw in ({}, {"with_lc": True, "with_mlp": True, "with_m11r_union": True, "with_exploratory": True}):
            steps = R.build_plan(cfg_under(tmp_path, mps_streams=streams, **kw))
            R.validate_plan(steps)  # includes the scope check
            for s in steps:
                for tok in s.cmd or []:
                    assert not any(t in tok for t in R.PLAYER_LAYER_TOKENS), (s.id, tok)
    # what the plan asks run_matrix for contains no stage-2 run (the group exists in run_matrix, the plan never selects it)
    from model_stgcn.ptr import run_matrix as RM

    runs, _ = RM.enumerate_runs(["karakin_tpp_squad", "erangel_tpp_squad"], ("erangel_tpp_squad",))
    assert set(STAGE2_ARMS).isdisjoint({r.arm for r in runs}) and RM.STAGE2_GROUP not in {r.group for r in runs} and {r.arm for r in RM.enumerate_norank_runs(["karakin_tpp_squad"])} == {"M01-norank", "M11R-norank"}
    # the check itself: stage 2, a stage-2 arm or a player dump in any command is refused (M11RZ, the relation-zone arm without the player layer, is fine)
    ok = R.Step(id="ok", lane="cpu", desc="", cmd=["python", "-m", "model_stgcn.ptr.run_matrix", "--only", "arm=M11RZ"])
    R.validate_plan([ok])
    for bad in (["--stage2"], ["--only", "arm=M11RZ-P"], ["--only", "arm=M11RZ-P-tm,M00"], ["--player-dump-dir", "/x/bigcomp2027_player_v2"]):
        with pytest.raises(ValueError, match="stage 2 is not part of the v2 re-run"):
            R.validate_plan([R.Step(id="bad", lane="cpu", desc="", cmd=["python", "-m", "model_stgcn.ptr.run_matrix", *bad])])
        with pytest.raises(ValueError, match="stage 2 is not part"):  # the runner validates too
            R.Runner([R.Step(id="bad", lane="cpu", desc="", cmd=["true", *bad])], R.State(tmp_path / "s.json"), tmp_path / "logs")
    # the dry run says so
    assert R.main(argv_under(tmp_path, "--dry-run")) == 0
    text = capsys.readouterr().out
    assert "excluded      stage 2 (player layer: M11RZ-P, M11RZ-P-tm)" in text and "player dump, which is not rebuilt" in text


def test_the_plan_is_valid_for_every_option_combination(tmp_path):
    for streams in (1, 2):
        for lc in (False, True):
            for mlp in (False, True):
                for union in (False, True):
                    for exploratory, vc, skip in ((False, True, False), (True, False, True)):
                        cfg = cfg_under(tmp_path, mps_streams=streams, with_lc=lc, with_mlp=mlp, with_m11r_union=union, with_exploratory=exploratory, validate_causal=vc, skip_leak_tests=skip)
                        steps = R.build_plan(cfg)
                        R.validate_plan(steps)
                        ids = [s.id for s in steps]
                        assert ("cpu.lc" in ids) == lc == ("reports.learning_curve" in ids) and ("mps.validate_causal" in ids) == vc and ("gate.leak_tests" in ids) == (not skip)
                        assert len({s.lane for s in steps} - {"main", "cpu"}) == streams
                        k1 = next(s for s in steps if s.id == "mps.k1")
                        assert (k1.lane == "mps1") == (streams == 2) and ("KRAFTON-mlp" in k1.desc) == mlp
                        assert ("transformer,mlp" in " ".join(k1.cmd)) == mlp
                        main_cmd = " ".join(next(s for s in steps if s.id == "cpu.ptr_main").cmd)
                        assert ("exploratory" in main_cmd) == exploratory and ("M11R-union" in " ".join(next(s for s in steps if s.id == "cpu.union").cmd)) == union
    with pytest.raises(ValueError, match="mps-streams"):
        R.Config(mps_streams=3)
    with pytest.raises(ValueError, match="positive"):
        R.Config(cpu_workers=0)


def test_two_mps_streams_overlap_k1_with_the_k2_k3_chain(tmp_path):
    one = R.simulate(R.build_plan(cfg_under(tmp_path, mps_streams=1)))
    two = R.simulate(R.build_plan(cfg_under(tmp_path, mps_streams=2)))
    assert max(e for _, e in two.values()) < max(e for _, e in one.values())  # about 3 h instead of 4.4 h: K1 no longer waits for K3
    assert two["mps.k1"][0] < two["mps.k3"][1] and one["mps.k1"][0] >= one["mps.k3"][1]
    assert 4.0 * 60 < max(e for _, e in one.values()) < 4.8 * 60  # the manifest's D.2: about 4.3 h with one stream
    steps2 = {s.id: s for s in R.build_plan(cfg_under(tmp_path, mps_streams=2))}
    assert steps2["mps.k1"].deps == ("mps.extract", "cpu.hgb") and "mps.k1" in steps2["mps.h0"].deps  # K1 needs the extraction only; H0 needs both chains


def test_every_command_forwards_the_configured_roots_and_no_v1_path(tmp_path):
    cfg = cfg_under(tmp_path, cpu_workers=5, cpu_threads=3, mps_threads=2, python="/py/main", python_sagong="/py/sagong")
    p = cfg.paths
    steps = {s.id: s for s in R.build_plan(cfg)}
    for s in steps.values():
        text = " ".join(s.cmd or []) + " " + " ".join(f"{k}={v}" for k, v in s.env.items())
        assert "_v1" not in text and "2026-10-06" not in text, s.id
        if s.cmd and "--dump-dir" in s.cmd:
            assert s.cmd[s.cmd.index("--dump-dir") + 1] == str(p.dump), s.id
    def arg(step, name):
        c = steps[step].cmd
        return c[c.index(name) + 1]
    for sid in ("cpu.hgb", "cpu.ptr_main", "cpu.ptr_norank"):  # run_matrix: the budget and the roots
        assert arg(sid, "--workers") == "5" and arg(sid, "--threads") == "3" and arg(sid, "--out-dir") == str(p.runs) and steps[sid].cmd[0] == "/py/main"
    assert arg("cpu.hgb", "--only") == "group=hgb" and arg("cpu.ptr_main", "--only") == "group=main,seeds" and "--norank" in steps["cpu.ptr_norank"].cmd
    assert arg("cpu.hgb", "--status-name") == "run_matrix_status_hgb.csv" and arg("cpu.ptr_norank", "--status-name") == "run_matrix_status_norank.csv"  # one status file per plan type
    for sid in ("cpu.sagong_plan", "cpu.sagong_extract", "cpu.sagong_habcols", "cpu.sagong_fit"):  # Sagong: the other interpreter, the three roots
        c = steps[sid].cmd
        assert c[0] == "/py/sagong" and c[1:3] == ["-m", "model_stgcn.sagong2025"] and arg(sid, "--plan-dir") == str(p.sagong) and arg(sid, "--runs-dir") == str(p.runs)
    assert arg("cpu.sagong_fit", "--models") == "xgb,rf,lr,hgb,hgb-hab" and arg("cpu.sagong_extract", "--workers") == "2"
    for sid in ("cpu.ehm_fit", "cpu.ehm_rq1", "cpu.ehm_rq1_physical", "reports.ehm_report"):
        assert arg(sid, "--runs-dir") == str(p.runs) and arg(sid, "--out-dir") == str(p.ecological) and steps[sid].env["OMP_NUM_THREADS"] == "2"
    assert arg("cpu.ehm_fit", "--stage") == "fit" and "--rq1-units" in steps["cpu.ehm_rq1_physical"].cmd and arg("reports.ehm_report", "--stage") == "report"
    for sid in ("mps.hgbkf", "mps.k2", "mps.k3", "mps.k1", "mps.h0", "cpu.union"):  # run_all: every root is forwarded
        assert arg(sid, "--out-dir") == str(p.runs) and arg(sid, "--data-dir") == str(p.krafton_data) and arg(sid, "--dump-dir") == str(p.dump) and arg(sid, "--table-dir") == str(p.kf), sid
        assert steps[sid].cwd == R.KR and steps[sid].cmd[1:4] == ["-m", "adaptation.run_all", "--stage"], sid
    assert [arg(s, "--stage") for s in ("mps.hgbkf", "mps.k2", "mps.k3", "mps.k1", "mps.h0", "cpu.union")] == ["hgbkf", "ourfeat", "k3", "main", "h0", "union"]
    assert arg("mps.k2", "--device") == "mps" and arg("mps.k2", "--threads") == "2" and arg("mps.h0", "--device") == "cpu"
    assert arg("cpu.union", "--cpu-workers") == "5" and arg("cpu.union", "--cpu-threads") == "3" and arg("cpu.union", "--union-arms") == "HGB-union"
    assert arg("mps.k1", "--encoders") == "transformer" and arg("mps.labels", "--out-dir") == str(p.krafton_data) and arg("mps.extract", "--out-dir") == str(p.krafton_data)
    assert arg("mps.kftable", "--out-dir") == str(p.kf) and arg("mps.kftable", "--label-dir") == str(p.krafton_data) and "--verify" in steps["mps.kftable"].cmd
    assert arg("mps.extract", "--cohorts") == ",".join(cfg.cohorts)
    assert arg("analysis.aggregate", "--runs-dir") == str(p.runs) and arg("analysis.aggregate", "--out-dir") == str(p.results) and arg("analysis.aggregate", "--workers") == "5"
    assert arg("analysis.supplement", "--results-dir") == str(p.results) and arg("analysis.supplement", "--out-dir") == str(p.supplement) and arg("analysis.supplement", "--workers") == "3"
    assert steps["tables.tab_decomp"].cmd[-3:] == ["--rerender-tables", "--out-dir", str(p.results)]
    assert arg("tables.kftable_h0_report", "--out-dir") == str(p.krafton_reports) and arg("tables.kftable_k3_contrasts", "--out") == str(p.krafton_reports / "k3_contrasts.csv")
    assert arg("tables.make_paper_tables", "--results-dir") == str(p.results) and arg("numbers.paper_numbers", "--out") == str(p.paper_numbers)
    assert arg("mps.validate_causal", "--cohorts") == "karakin_tpp_squad,erangel_tpp_squad"  # the five audit cohorts that this registry has
    # the optional learning curve has its own roots
    lc = {s.id: s for s in R.build_plan(cfg_under(tmp_path, with_lc=True))}
    assert lc["cpu.lc"].cmd[lc["cpu.lc"].cmd.index("--out-dir") + 1] == str(p.runs_lc) and "--lc-steps-matched" in lc["cpu.lc_steps"].cmd
    assert lc["reports.learning_curve"].cmd[lc["reports.learning_curve"].cmd.index("--lc-dir") + 1] == str(p.runs_lc)


def test_validate_plan_catches_duplicates_unknown_dependencies_and_deadlocks():
    a = R.Step(id="a", lane="x", desc="", cmd=["true"])
    b = R.Step(id="b", lane="y", desc="", cmd=["true"], deps=("a",))
    R.validate_plan([a, b])
    with pytest.raises(ValueError, match="duplicate"):
        R.validate_plan([a, a])
    with pytest.raises(ValueError, match="unknown step"):
        R.validate_plan([R.Step(id="c", lane="x", desc="", cmd=["true"], deps=("zzz",))])
    # lane x runs a then c; lane y runs b then d; a needs d and b needs c: every lane waits for the other
    s = [R.Step(id="a", lane="x", desc="", cmd=["t"], deps=("d",)), R.Step(id="c", lane="x", desc="", cmd=["t"]),
         R.Step(id="b", lane="y", desc="", cmd=["t"], deps=("c",)), R.Step(id="d", lane="y", desc="", cmd=["t"])]  # fmt: skip
    with pytest.raises(ValueError, match="deadlock"):
        R.validate_plan(s)
    R.validate_plan([b], universe=[a, b])  # a selection: a dependency outside it counts as satisfied
    assert R.select([a, b], "b", "") == [b] and R.select([a, b], "", "b") == [a]
    assert [s.id for s in R.select(R.build_plan(R.Config()), "mps.k", "mps.k1")] == ["mps.kftable", "mps.k2", "mps.k3"]
    with pytest.raises(SystemExit, match="matches no step"):
        R.select([a, b], "nope", "")


# --------------------------------------------------------------------------- the dry run
def argv_under(tmp_path, *extra):
    p = P.RerunPaths.under(tmp_path)
    return ["--dump-dir", str(p.dump), "--runs-dir", str(p.runs), "--kf-dir", str(p.kf), "--krafton-data-dir", str(p.krafton_data), "--sagong-dir", str(p.sagong),
            "--results-dir", str(p.results), "--krafton-reports-dir", str(p.krafton_reports), "--paper-dir", str(p.paper), *extra]  # fmt: skip


def test_the_dry_run_prints_the_plan_and_writes_nothing(tmp_path, capsys):
    before = tree(tmp_path)
    assert R.main(argv_under(tmp_path, "--dry-run")) == 0  # the dump is not there yet: the plan is printed, the gate preview says it would fail
    text = capsys.readouterr().out
    assert tree(tmp_path) == before == set()  # nothing was created (no results directory, no manifest, no state, no logs, no stamp)
    for sid in FINAL_STEPS:
        assert f"\n{sid} " in text, sid
    assert "32 steps" in text and "lanes main, cpu, mps0" in text and "6 CPU workers x 2 threads + 1 MPS stream(s) x 3 threads = 15 cores" in text
    assert "WOULD FAIL now" in text and "does not exist" in text and "estimate      about 4." in text and "HEAD " in text and "status --porcelain sha1 " in text
    assert "result modules sha1" in text and "main 108 + seeds 20 (hgb 36 first), norank 24, union HGB 12" in text and "_v1" not in text
    assert "credentials in the environment:" in text and "$ " in text and "[needs the database]" in text
    assert "code snapshot " in text and "git diff HEAD --binary = " in text and "tracked files changed" in text and "untracked .py/.md/.json files under model_stgcn, stdb, third_party/krafton_pubg_winprob/adaptation, tools, tests, paper/bigcomp2027/tables" in text
    assert "(written at launch, not by this dry run)" in text and "excluded      stage 2" in text
    # a dump that is there: the preview checks it, the budget options are echoed
    make_dump(P.RerunPaths.under(tmp_path).dump, version="v2", cohorts=("paramo_tpp_squad", "karakin_tpp_squad"))
    before = tree(tmp_path)
    assert R.main(argv_under(tmp_path, "--dry-run", "--mps-streams", "2", "--cpu-workers", "4", "--with-lc", "--with-mlp")) == 0
    text = capsys.readouterr().out
    assert tree(tmp_path) == before  # still nothing written (the dump itself is read, not touched)
    assert "35 steps" in text and "lanes main, cpu, mps0, mps1" in text and "4 CPU workers x 2 threads + 2 MPS stream(s) x 3 threads = 14 cores" in text
    assert "meta sha1 " in text and "cohorts 2: paramo_tpp_squad, karakin_tpp_squad" in text and "WOULD FAIL now" in text  # prefix_replay.json is not there
    assert "KRAFTON-mlp" in text and "cpu.lc_steps" in text and "reports.learning_curve" in text
    (P.RerunPaths.under(tmp_path).dump / "prefix_replay.json").write_text(json.dumps({"n_matches": 1, "n_cuts": 1, "mismatches": [], "min_gap_between_last_kept_record_and_cut_s": 0.5}))
    assert R.main(argv_under(tmp_path, "--dry-run", "--only", "analysis", "--skip", "analysis.supplement")) == 0
    text = capsys.readouterr().out
    assert "the dump passes the manifest-level checks" in text and "1 steps" in text.split("\n")[0] and "\nanalysis.aggregate " in text and "analysis.supplement" not in text


def test_the_dry_run_refuses_a_v1_root_next_to_the_v2_dump(tmp_path, capsys):
    for flag, v1 in (("--runs-dir", "bigcomp2027_ptr_v1"), ("--kf-dir", "bigcomp2027_kraftonfeat_v1"), ("--krafton-data-dir", "krafton_winprob_v1"), ("--results-dir", "bigcomp2027_results_v1"),
                     ("--lc-dir", "bigcomp2027_ptr_v1_lc"), ("--sagong-dir", "bigcomp2027_sagong_v1"), ("--krafton-reports-dir", "krafton_baseline_v1")):  # fmt: skip
        argv = argv_under(tmp_path, "--dry-run")
        if flag in argv:
            argv[argv.index(flag) + 1] = str(tmp_path / v1)
        else:
            argv += [flag, str(tmp_path / v1)]
        with pytest.raises(SystemExit, match="refusing to run"):
            R.main(argv)
    assert tree(tmp_path) == set()  # nothing was created


def test_the_dry_run_of_the_real_repository_state(capsys):
    assert R.main(["--dry-run"]) == 0
    text = capsys.readouterr().out
    assert str(P.RUNS_DIR) in text and str(P.DUMP_DIR) in text and "_v1" not in text and "32 steps" in text


# --------------------------------------------------------------------------- the gate
def good_dump(tmp_path) -> P.RerunPaths:
    paths = P.RerunPaths.under(tmp_path)
    make_dump(paths.dump, version="v2", cohorts=("paramo_tpp_squad", "karakin_tpp_squad"))
    (paths.dump / "prefix_replay.json").write_text(json.dumps({"n_matches": 204, "n_cuts": 1293, "mismatches": [], "min_gap_between_last_kept_record_and_cut_s": 0.001,
                                                                 "mutation_check": {"matches": 3, "cuts_compared": 21, "cuts_detected_as_leaky": 21}}))  # fmt: skip
    return paths


def test_the_gate_step_checks_version_meta_replay_and_roots(tmp_path):
    paths = good_dump(tmp_path)
    cfg = R.Config(paths=paths, cohorts=("paramo_tpp_squad", "karakin_tpp_squad"))
    step = R.build_plan(cfg)[0]
    log = tmp_path / "gate.log"
    assert step.id == "gate.dump" and step.fn(step, log) == 0
    text = log.read_text()
    assert "version v2" in text and "prefix replay: " in text and "cohorts: 2" in text and "error" not in text
    # the replay file with a mismatch, a missing replay file, a v1 dump, planned cohorts that differ
    (paths.dump / "prefix_replay.json").write_text(json.dumps({"n_matches": 5, "n_cuts": 9, "mismatches": [{"k": 2}], "min_gap_between_last_kept_record_and_cut_s": 0.1}))
    assert step.fn(step, log) == 1 and "differ between the full and the database-truncated source" in log.read_text()
    (paths.dump / "prefix_replay.json").unlink()
    assert step.fn(step, log) == 1 and "prefix_replay.json is missing" in log.read_text()
    legacy_cfg = R.Config(paths=P.RerunPaths.under(tmp_path / "other").replace(dump=make_dump(tmp_path / "bigcomp2027_phase_v1")))
    assert R.build_plan(legacy_cfg)[0].fn(step, log) == 1 and "is v1" in log.read_text()  # the gate demands a v2 dump
    other = R.Config(paths=paths, cohorts=("paramo_tpp_squad", "erangel_tpp_squad"))
    (paths.dump / "prefix_replay.json").write_text(json.dumps({"n_matches": 1, "n_cuts": 1, "mismatches": [], "min_gap_between_last_kept_record_and_cut_s": 0.1}))
    assert R.build_plan(other)[0].fn(step, log) == 1 and "differ from the planned" in log.read_text()


# --------------------------------------------------------------------------- provenance: git state, the manifest, resuming
def make_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for cmd in (["init", "-q"], ["config", "commit.gpgsign", "false"]):
        subprocess.run(["git", *cmd], cwd=path, check=True, env=env)
    (path / "a.txt").write_text("a")
    subprocess.run(["git", "add", "a.txt"], cwd=path, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "one"], cwd=path, check=True, env=env)
    return path


def test_git_state_records_head_and_the_hash_of_the_porcelain_status(tmp_path):
    repo = make_repo(tmp_path / "repo")
    clean = R.git_state(repo)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    assert clean["head"] == head and clean["n_status_lines"] == 0 and clean["status_porcelain_sha1"] == __import__("hashlib").sha1(b"").hexdigest()
    (repo / "b.txt").write_text("b")  # an untracked file counts: the hash is of `git status --porcelain`
    dirty = R.git_state(repo)
    assert dirty["head"] == head and dirty["n_status_lines"] == 1 and dirty["status_porcelain_sha1"] != clean["status_porcelain_sha1"]
    assert dirty["status_porcelain_sha1"] == __import__("hashlib").sha1(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo)).hexdigest()
    assert R.git_state(tmp_path / "not_a_repo")["head"] is None


def test_check_resume_refuses_changed_head_modules_or_dump_and_warns_about_the_status(tmp_path):
    base = {"git": {"head": "h1", "branch": "b", "status_porcelain_sha1": "s1", "n_status_lines": 3}, "result_modules_sha1": "m1", "dump": {"dir": "d", "version": "v2", "meta_sha1": "d1"}}
    assert R.check_resume(None, base) == ([], []) and R.check_resume({"provenance": base}, base) == ([], [])
    now = json.loads(json.dumps(base))
    now["git"]["head"] = "h2"
    now["result_modules_sha1"] = "m2"
    now["dump"]["meta_sha1"] = "d2"
    now["git"]["status_porcelain_sha1"] = "s2"
    errors, warnings = R.check_resume({"provenance": base}, now)
    assert len(errors) == 3 and "HEAD changed" in errors[0] and "result-affecting modules" in errors[1] and "dump meta hash changed" in errors[2]
    assert len(warnings) == 1 and "status --porcelain changed" in warnings[0]
    only_status = json.loads(json.dumps(base))
    only_status["git"]["status_porcelain_sha1"] = "s9"
    assert R.check_resume({"provenance": base}, only_status)[0] == []  # a changed status alone is a warning: HEAD and the result modules decide


# --------------------------------------------------------------------------- the code snapshot (git diff HEAD + the untracked sources, with sha256 of each)
GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
BLOB_V1 = bytes(range(256)) * 8
BLOB_V2 = bytes(reversed(range(256))) * 8 + b"\xff\x00\xfe not utf-8 \x80"


def write(path: Path, data: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else data)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True, env={**os.environ, **GIT_ENV}).stdout


def snapshot_repo(tmp_path: Path) -> Path:
    """A small repository in the state of the real one: tracked files modified / deleted / staged (one of them binary), untracked sources in and out of the snapshot scope,
    an ignored file."""
    repo = tmp_path / "srepo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "commit.gpgsign", "false")
    write(repo / ".gitignore", "model_stgcn/ignored.py\nreports/\n")
    write(repo / "model_stgcn/ptr/a.py", "a = 1\n")
    write(repo / "tools/old.py", "old = 1\n")
    write(repo / "data/blob.bin", BLOB_V1)
    write(repo / "README.md", "readme\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    write(repo / "model_stgcn/ptr/a.py", "a = 2  # modified, not staged\n")
    (repo / "tools/old.py").unlink()  # deleted
    write(repo / "data/blob.bin", BLOB_V2)  # a binary file changed (outside the scope of the tar, inside the diff)
    write(repo / "model_stgcn/ptr/staged.py", "staged = 1\n")
    git(repo, "add", "model_stgcn/ptr/staged.py")  # staged, never committed: part of git diff HEAD
    for rel, text in {"model_stgcn/ptr/new.py": "new = 1\n", "tests/test_new.py": "def test_x():\n    pass\n", "tools/sub/dir/y.json": '{"y": 1}\n', "stdb/x.json": "{}\n",
                      "third_party/krafton_pubg_winprob/adaptation/k.md": "# k\n", "model_stgcn/ünï.py": "u = 'ü'\n",
                      "paper/bigcomp2027/tables/make_paper_tables.py": "t = 1\n"}.items():
        write(repo / rel, text)  # untracked, in scope
    for rel in ("docs/note.md", "paper/x.tex", "third_party/krafton_pubg_winprob/src/z.py", "tests/data.csv", "model_stgcn/notes.txt"):
        write(repo / rel, "outside\n")  # untracked, outside the scope (another directory or another extension)
    write(repo / "model_stgcn/ignored.py", "ignored = 1\n")
    write(repo / "reports/r.json", "{}\n")  # both are git-ignored: never listed
    return repo


UNTRACKED_IN_SCOPE = sorted(["model_stgcn/ptr/new.py", "model_stgcn/ünï.py", "stdb/x.json", "third_party/krafton_pubg_winprob/adaptation/k.md", "tests/test_new.py", "tools/sub/dir/y.json",
                             "paper/bigcomp2027/tables/make_paper_tables.py"])  # fmt: skip


def test_the_code_snapshot_holds_the_diff_and_the_untracked_sources_with_their_hashes(tmp_path):
    repo = snapshot_repo(tmp_path)
    out = tmp_path / "snap"
    info = R.take_code_snapshot(repo, out)
    assert {f.name for f in out.iterdir()} == {"git_diff_HEAD.patch", "untracked_files.tar", "snapshot.json", "SHA256SUMS", "untracked_files.sha256"}
    # the diff: exactly `git diff HEAD --binary` (tracked changes, staged and unstaged, the binary file included), not the untracked files
    diff = (out / "git_diff_HEAD.patch").read_bytes()
    want = subprocess.run(["git", "diff", "HEAD", "--binary", "--full-index", "--no-ext-diff", "--no-color", "--no-renames", "--src-prefix=a/", "--dst-prefix=b/"], cwd=repo, check=True, capture_output=True).stdout
    assert diff == want and b"GIT binary patch" in diff and b"a = 2" in diff and b"deleted file mode" in diff and b"staged = 1" in diff and b"new = 1" not in diff
    assert info["git_diff_HEAD"] == {"sha256": hashlib.sha256(diff).hexdigest(), "bytes": len(diff), "n_files_changed": 4}  # a.py, old.py, blob.bin, staged.py
    # the tar: the untracked .py / .md / .json files of the five roots, byte for byte, nothing else
    with tarfile.open(out / "untracked_files.tar") as tf:
        members = {m.name: tf.extractfile(m).read() for m in tf.getmembers()}
    assert sorted(members) == UNTRACKED_IN_SCOPE
    assert all(members[rel] == (repo / rel).read_bytes() for rel in UNTRACKED_IN_SCOPE)
    doc = json.loads((out / "snapshot.json").read_text())
    assert doc["head"] == git(repo, "rev-parse", "HEAD").strip() and doc["branch"] == git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    assert doc["status_porcelain_sha1"] == hashlib.sha1(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo)).hexdigest()
    assert doc["untracked"]["scope"] == {"roots": list(R.SNAPSHOT_ROOTS), "extensions": [".py", ".md", ".json"]}
    assert {f["path"]: f["sha256"] for f in doc["untracked"]["files"]} == {rel: hashlib.sha256((repo / rel).read_bytes()).hexdigest() for rel in UNTRACKED_IN_SCOPE}
    assert doc["untracked"]["sha256"] == hashlib.sha256((out / "untracked_files.tar").read_bytes()).hexdigest() and doc["git_diff_HEAD"]["sha256"] == info["git_diff_HEAD"]["sha256"]
    # what is left out is said, not silently dropped: untracked files of other directories / extensions are counted and named, ignored files are not there at all
    assert doc["untracked"]["outside_scope"]["n"] == 5 and sorted(doc["untracked"]["outside_scope"]["first"]) == sorted(
        ["docs/note.md", "model_stgcn/notes.txt", "paper/x.tex", "tests/data.csv", "third_party/krafton_pubg_winprob/src/z.py"])  # fmt: skip
    assert "model_stgcn/ignored.py" not in json.dumps(doc) and "reports/r.json" not in json.dumps(doc) and info["n_untracked_outside_scope"] == 5
    assert "git apply --binary" in doc["restore"] and doc["restore"].startswith(f"git checkout {doc['head']}")
    # the hash files are in sha256sum format
    sums = dict(line.split("  ")[::-1] for line in (out / "SHA256SUMS").read_text().splitlines())
    assert set(sums) == {"git_diff_HEAD.patch", "untracked_files.tar", "snapshot.json", "untracked_files.sha256"} and all(hashlib.sha256((out / n).read_bytes()).hexdigest() == h for n, h in sums.items())
    assert dict(line.split("  ")[::-1] for line in (out / "untracked_files.sha256").read_text().splitlines()) == {rel: hashlib.sha256((repo / rel).read_bytes()).hexdigest() for rel in UNTRACKED_IN_SCOPE}
    assert R.verify_code_snapshot(out) == []
    # the digest is a function of the code alone: the same tree gives the same digest in another directory, another tracked change gives another one
    again = R.take_code_snapshot(repo, tmp_path / "snap2")
    assert again["digest"] == info["digest"]
    write(repo / "model_stgcn/ptr/a.py", "a = 3\n")
    assert R.take_code_snapshot(repo, tmp_path / "snap3")["digest"] != info["digest"]


def test_the_code_snapshot_restores_the_working_tree_on_a_clean_checkout_of_head(tmp_path):
    repo = snapshot_repo(tmp_path)
    out = tmp_path / "snap"
    R.take_code_snapshot(repo, out)
    clone = tmp_path / "clone"
    git(tmp_path, "clone", "-q", str(repo), str(clone))  # HEAD of the repository: the base commit, none of the changes
    assert (clone / "model_stgcn/ptr/a.py").read_text() == "a = 1\n" and not (clone / "model_stgcn/ptr/new.py").exists()
    git(clone, "apply", "--binary", str(out / "git_diff_HEAD.patch"))
    with tarfile.open(out / "untracked_files.tar") as tf:
        tf.extractall(clone)
    for rel in ["model_stgcn/ptr/a.py", "model_stgcn/ptr/staged.py", "data/blob.bin", "README.md", *UNTRACKED_IN_SCOPE]:
        assert (clone / rel).read_bytes() == (repo / rel).read_bytes(), rel
    assert not (clone / "tools/old.py").exists() and (clone / "data/blob.bin").read_bytes() == BLOB_V2  # the deletion and the binary change came back
    assert not (clone / "docs/note.md").exists() and not (clone / "model_stgcn/ignored.py").exists()  # not captured, as listed


def test_the_code_snapshot_skips_what_is_not_a_regular_or_small_file_and_refuses_an_implausible_set(tmp_path, monkeypatch):
    repo = snapshot_repo(tmp_path)
    (repo / "tools/link.py").symlink_to("../README.md")
    write(repo / "tools/big.json", "x" * 300)
    out = tmp_path / "snap"
    info = R.take_code_snapshot(repo, out, max_file_bytes=200)
    doc = json.loads((out / "snapshot.json").read_text())
    assert {f["reason"].split(" (")[0] for f in doc["untracked"]["skipped"]} == {"not a regular file", "larger than 200 bytes"} and info["untracked_tar"]["n_skipped"] == 2
    assert {s["path"] for s in doc["untracked"]["skipped"]} == {"tools/link.py", "tools/big.json"} and "tools/big.json" not in {f["path"] for f in doc["untracked"]["files"]}
    assert R.verify_code_snapshot(out) == []
    monkeypatch.setattr(R, "SNAPSHOT_MAX_FILES", 3)
    with pytest.raises(RuntimeError, match="something that is not code is not ignored"):
        R.take_code_snapshot(repo, tmp_path / "snap_big")
    monkeypatch.setattr(R, "SNAPSHOT_MAX_FILES", 2000)
    monkeypatch.setattr(R, "SNAPSHOT_MAX_TOTAL_BYTES", 50)
    with pytest.raises(RuntimeError, match="add up to more than"):
        R.take_code_snapshot(repo, tmp_path / "snap_total")
    (tmp_path / "not_a_repo").mkdir()
    with pytest.raises(RuntimeError, match="git rev-parse HEAD failed"):
        R.take_code_snapshot(tmp_path / "not_a_repo", tmp_path / "snap_none")


def test_a_tampered_snapshot_does_not_verify(tmp_path):
    repo = snapshot_repo(tmp_path)
    out = tmp_path / "snap"
    R.take_code_snapshot(repo, out)
    assert R.verify_code_snapshot(out) == []
    (out / "git_diff_HEAD.patch").write_bytes((out / "git_diff_HEAD.patch").read_bytes() + b"\n")
    problems = R.verify_code_snapshot(out)
    assert any("git_diff_HEAD.patch: the sha256 differs" in x for x in problems) and any("SHA256SUMS: git_diff_HEAD.patch" in x for x in problems)
    out2 = tmp_path / "snap2"
    R.take_code_snapshot(repo, out2)
    with tarfile.open(out2 / "untracked_files.tar", "w") as tf:  # another content under the same names
        for rel in UNTRACKED_IN_SCOPE:
            data = b"changed\n"
            info = tarfile.TarInfo(rel)
            info.size = len(data)
            tf.addfile(info, __import__("io").BytesIO(data))
    assert any("differs inside the tar" in x for x in R.verify_code_snapshot(out2))
    (out2 / "SHA256SUMS").unlink()
    assert "SHA256SUMS is missing" in R.verify_code_snapshot(out2)
    assert R.verify_code_snapshot(tmp_path / "nothing")[0].startswith("snapshot.json is unreadable")


def test_a_launch_snapshot_keeps_the_first_one_and_adds_one_only_when_the_code_changed(tmp_path):
    repo = snapshot_repo(tmp_path)
    results = tmp_path / "results"
    first = R.snapshot_launch(repo, results)
    root = results / "code_snapshot"
    assert first["dir"] == str(root) and first["unchanged"] is False and R.verify_code_snapshot(root) == [] and not list(root.glob(".tmp*"))
    assert {f.name for f in root.iterdir()} == {"git_diff_HEAD.patch", "untracked_files.tar", "snapshot.json", "SHA256SUMS", "untracked_files.sha256"}
    same = R.snapshot_launch(repo, results)  # a resume of the same code: nothing new
    assert same["dir"] == str(root) and same["unchanged"] is True and same["digest"] == first["digest"] and not list(root.glob("launch_*"))
    base_files = {f.name: f.read_bytes() for f in root.iterdir() if f.is_file()}
    write(repo / "model_stgcn/ptr/a.py", "a = 4  # edited after the first launch\n")  # the code changed: the resume gets its own snapshot, the first stays
    second = R.snapshot_launch(repo, results)
    d2 = Path(second["dir"])
    assert second["unchanged"] is False and d2.parent == root and d2.name.startswith("launch_") and R.verify_code_snapshot(d2) == [] and second["digest"] != first["digest"]
    assert {f.name: f.read_bytes() for f in root.iterdir() if f.is_file()} == base_files  # the first snapshot is untouched
    assert b"a = 4" in (d2 / "git_diff_HEAD.patch").read_bytes() and b"a = 4" not in (root / "git_diff_HEAD.patch").read_bytes()
    third = R.snapshot_launch(repo, results)  # back to a saved state: points at it
    assert third["dir"] == str(d2) and third["unchanged"] is True and len(list(root.glob("launch_*"))) == 1
    write(repo / "model_stgcn/ptr/a.py", "a = 2  # modified, not staged\n")  # the first code again
    assert R.snapshot_launch(repo, results)["dir"] == str(root)
    (tmp_path / "not_a_repo").mkdir()
    with pytest.raises(RuntimeError, match="git rev-parse HEAD failed"):  # a failing snapshot leaves nothing behind
        R.snapshot_launch(tmp_path / "not_a_repo", tmp_path / "other_results")
    assert not list((tmp_path / "other_results" / "code_snapshot").iterdir())


def test_the_dry_run_previews_the_snapshot_and_writes_nothing(tmp_path):
    repo = snapshot_repo(tmp_path)
    before = {str(p.relative_to(repo)) for p in repo.rglob("*") if ".git" not in p.parts}
    pre = R.preview_code_snapshot(repo)
    assert pre == {"ok": True, "n_files_changed": 4, "diff_bytes": pre["diff_bytes"], "n_untracked": 7, "untracked_bytes": sum((repo / r).stat().st_size for r in UNTRACKED_IN_SCOPE), "n_outside_scope": 5}
    assert pre["diff_bytes"] == len(subprocess.run(["git", "diff", "HEAD", "--binary", "--full-index", "--no-ext-diff", "--no-color", "--no-renames", "--src-prefix=a/", "--dst-prefix=b/"], cwd=repo, capture_output=True).stdout)
    assert {str(p.relative_to(repo)) for p in repo.rglob("*") if ".git" not in p.parts} == before  # nothing was written
    assert R.preview_code_snapshot(tmp_path / "nowhere")["ok"] is False


def test_a_launch_saves_the_snapshot_into_the_run_manifest(tmp_path, capsys, monkeypatch):
    repo = snapshot_repo(tmp_path)
    monkeypatch.setattr(R, "REPO", repo)  # the launch snapshots this repository, not the real one
    paths = good_dump(tmp_path)
    argv = argv_under(tmp_path, "--only", "gate.dump", "--skip-leak-tests")
    assert R.main(argv) == 0
    out = capsys.readouterr().out
    manifest = json.loads(paths.run_manifest.read_text())
    snap = manifest["launches"][0]["code_snapshot"]
    root = paths.results / "code_snapshot"
    assert f"code snapshot: {root} (saved; HEAD " in out and "4 tracked files changed, 7 untracked files" in out
    assert manifest["code_snapshot_dir"] == str(root) and snap["dir"] == str(root) and snap["unchanged"] is False and snap["head"] == git(repo, "rev-parse", "HEAD").strip()
    assert snap["git_diff_HEAD"]["n_files_changed"] == 4 and snap["untracked_tar"]["n_files"] == 7 and snap["digest"] == json.loads((root / "snapshot.json").read_text())["digest"]
    assert R.verify_code_snapshot(root) == [] and snap["git_diff_HEAD"]["sha256"] == hashlib.sha256((root / "git_diff_HEAD.patch").read_bytes()).hexdigest()
    # the same code on the next launch: recorded as unchanged, no second directory; other code: its own directory, and the manifest points at it
    assert R.main(argv) == 0
    assert json.loads(paths.run_manifest.read_text())["launches"][1]["code_snapshot"]["unchanged"] is True and not list(root.glob("launch_*"))
    write(repo / "tools/another.py", "z = 1\n")
    assert R.main(argv) == 0
    third = json.loads(paths.run_manifest.read_text())["launches"][2]["code_snapshot"]
    assert third["unchanged"] is False and Path(third["dir"]).parent == root and R.verify_code_snapshot(Path(third["dir"])) == []
    assert "tools/another.py" in json.loads((Path(third["dir"]) / "snapshot.json").read_text())["untracked"]["files"].__repr__()
    # --no-code-snapshot is recorded, and a failing snapshot stops the launch before any step starts
    other = argv_under(tmp_path / "second_results_root", "--only", "gate.dump", "--skip-leak-tests")
    other[other.index("--dump-dir") + 1] = str(paths.dump)  # the same dump, another root for everything else
    assert R.main([*other, "--no-code-snapshot"]) == 0
    skipped = json.loads(P.RerunPaths.under(tmp_path / "second_results_root").run_manifest.read_text())["launches"][0]["code_snapshot"]
    assert skipped == {"skipped": True, "reason": "--no-code-snapshot"} and not (P.RerunPaths.under(tmp_path / "second_results_root").results / "code_snapshot").exists()
    assert "warning: no code snapshot" in capsys.readouterr().out
    monkeypatch.setattr(R, "REPO", tmp_path / "not_a_repo")
    (tmp_path / "not_a_repo").mkdir()
    state_before = (paths.run_state.read_text() if paths.run_state.exists() else "")
    with pytest.raises(SystemExit, match="the code snapshot failed"):
        R.main(argv)
    assert (paths.run_state.read_text() if paths.run_state.exists() else "") == state_before  # no step ran


def test_a_launch_writes_manifest_state_logs_and_stamps_and_resumes(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(R, "REPO", snapshot_repo(tmp_path))  # the code snapshot of every launch reads a small repository, not the real one
    paths = good_dump(tmp_path)
    argv = argv_under(tmp_path, "--only", "gate.dump", "--skip-leak-tests")
    assert R.main(argv) == 0
    out = capsys.readouterr().out
    assert "gate.dump: start" in out and "gate.dump: done" in out and "finished with exit code 0" in out
    manifest = json.loads(paths.run_manifest.read_text())
    prov = manifest["provenance"]
    assert prov["git"]["head"] == R.git_state()["head"] and prov["git"]["status_porcelain_sha1"] and prov["result_modules_sha1"] == R.code_fingerprint()
    assert prov["dump"]["version"] == "v2" and prov["dump"]["meta_sha1"] == P.dump_meta_sha1(paths.dump, deep=True) and len(manifest["launches"]) == 1
    assert manifest["roots"]["runs"] == str(paths.runs) and manifest["budget"] == {"cpu_workers": 6, "cpu_threads": 2, "mps_streams": 1, "mps_threads": 3}
    assert manifest["launches"][0]["status"] == {"gate.dump": "done"} and manifest["launches"][0]["exit_code"] == 0
    state = json.loads(paths.run_state.read_text())["steps"]
    assert state["gate.dump"]["status"] == "done" and state["gate.dump"]["exit_code"] == 0 and state["gate.dump"]["log"] == str(paths.logs / "gate.dump.log")
    assert (paths.logs / "gate.dump.log").read_text().startswith("dump ")  # one log per step, under <results>/logs
    for root in (paths.runs, paths.krafton_data, paths.kf, paths.sagong, paths.results):
        assert P.artifact_version(root) == ("v2", "stamp"), root  # the roots are claimed for the v2 dump
    assert (paths.krafton_data / "logs").is_dir()  # adaptation.extract opens a log there without creating it
    # the gate runs at every launch (it is cheap and it is the safety check); the launch appends a record without changing the first provenance
    assert R.main(argv) == 0
    out2 = capsys.readouterr().out
    assert "gate.dump: start" in out2 and "gate.dump: done before" not in out2
    again = json.loads(paths.run_manifest.read_text())
    assert again["provenance"] == manifest["provenance"] and len(again["launches"]) == 2 and again["created_utc"] == manifest["created_utc"]
    # the dump is rebuilt: a resume is refused, --allow-code-change accepts it (and records it), --restart starts over
    man = json.loads((paths.dump / "paramo_tpp_squad" / "dump_manifest.json").read_text())
    man["elapsed_s"] = 99.0
    (paths.dump / "paramo_tpp_squad" / "dump_manifest.json").write_text(json.dumps(man))
    with pytest.raises(SystemExit, match="refusing to resume: the dump meta hash changed"):
        R.main(argv)
    assert R.main([*argv, "--allow-code-change"]) == 0
    assert any("dump meta hash changed" in c for c in json.loads(paths.run_manifest.read_text())["launches"][-1]["provenance_changes"])
    assert R.main([*argv, "--restart"]) == 0
    restarted = json.loads(paths.run_manifest.read_text())
    assert restarted["provenance"]["dump"]["meta_sha1"] != prov["dump"]["meta_sha1"] and len(restarted["launches"]) == 1  # the first launch of a fresh run
    # a database step without credentials is refused before anything starts
    monkey_env = {k: v for k, v in os.environ.items() if k not in ("PG_PASSWORD", "PGPASSWORD")}
    old = dict(os.environ)
    os.environ.clear()
    os.environ.update(monkey_env)
    try:
        with pytest.raises(SystemExit, match="need the credentials"):
            R.main(argv_under(tmp_path, "--only", "mps.extract"))
    finally:
        os.environ.clear()
        os.environ.update(old)


# --------------------------------------------------------------------------- the runner (lanes, dependencies, resume, failure)
def py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def stand_in_plan(tmp_path: Path, *, fail: str | None = None) -> list[R.Step]:
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    def writer(name: str, rc: int = 0, delay: float = 0.0) -> list[str]:
        return py(f"import time, pathlib; time.sleep({delay}); pathlib.Path(r'{out}/{name}.txt').write_text('x'); raise SystemExit({rc})")
    return [
        R.Step(id="g", lane="main", desc="gate", cmd=writer("g"), outputs=(out / "g.txt",)),
        R.Step(id="cpu.a", lane="cpu", desc="cpu 1", cmd=writer("a", 3 if fail == "cpu.a" else 0), deps=("g",)),
        R.Step(id="cpu.b", lane="cpu", desc="cpu 2", cmd=writer("b"), deps=("cpu.a",), outputs=(out / "b.txt",)),
        R.Step(id="mps.x", lane="mps0", desc="mps 1", cmd=writer("x", delay=0.3), deps=("g",)),
        R.Step(id="mps.y", lane="mps0", desc="mps 2", cmd=writer("y"), deps=("mps.x",)),
        R.Step(id="final", lane="main", desc="needs both lanes", cmd=writer("final"), deps=("cpu.b", "mps.y"), outputs=(out / "final.txt",)),
    ]  # fmt: skip


def run_plan(tmp_path, steps, **kw):
    logs = tmp_path / "logs"
    state = R.State(tmp_path / "state.json")
    lines: list[str] = []
    runner = R.Runner(steps, state, logs, log=lines.append, **kw)
    return runner.run(), runner, state, lines, logs


def test_the_runner_runs_lanes_in_parallel_in_order_with_one_log_per_step(tmp_path):
    steps = stand_in_plan(tmp_path)
    rc, runner, state, lines, logs = run_plan(tmp_path, steps)
    assert rc == 0 and {s.id: runner.status[s.id] for s in steps} == {s.id: "done" for s in steps}
    assert {p.name for p in logs.iterdir()} == {f"{s.id}.log" for s in steps}  # one log per step
    assert "# " in (logs / "cpu.a.log").read_text() and "exit 0" in (logs / "cpu.a.log").read_text()
    st = json.loads((tmp_path / "state.json").read_text())["steps"]
    assert all(v["status"] == "done" and v["exit_code"] == 0 and v["signature"] for v in st.values())
    # 'final' needs both lanes: it started after cpu.b and mps.y ended
    started = {l.split(": ")[0].split("] ")[1]: i for i, l in enumerate(lines) if l.endswith(": start")}
    ended = {l.split(": ")[0].split("] ")[1]: i for i, l in enumerate(lines) if ": done in" in l}
    assert started["final"] > ended["cpu.b"] and started["final"] > ended["mps.y"] and started["cpu.b"] > ended["cpu.a"] and started["mps.y"] > ended["mps.x"]
    assert started["cpu.a"] < ended["mps.x"] or started["mps.x"] < ended["cpu.a"]  # the lanes overlap in time: neither waited for the other lane's first step
    # a second run skips everything that is recorded as done with its outputs in place
    rc2, runner2, _, lines2, _ = run_plan(tmp_path, stand_in_plan(tmp_path))
    assert rc2 == 0 and {runner2.status[s.id] for s in steps if s.outputs} == {"cached"} and any("done before" in l for l in lines2)


def test_a_failed_step_skips_only_its_dependents_and_runs_again_on_resume(tmp_path):
    rc, runner, state, lines, logs = run_plan(tmp_path, stand_in_plan(tmp_path, fail="cpu.a"))
    assert rc == 1
    assert runner.status["cpu.a"] == "failed" and runner.status["cpu.b"] == "skipped" and runner.status["final"] == "skipped"
    assert runner.status["mps.x"] == "done" and runner.status["mps.y"] == "done"  # the independent lane carried on
    st = json.loads((tmp_path / "state.json").read_text())["steps"]
    assert st["cpu.a"]["status"] == "failed" and st["cpu.a"]["exit_code"] == 3 and st["cpu.b"]["status"] == "skipped"
    assert "exit 3" in (logs / "cpu.a.log").read_text()
    # fixed command: the failed step and its dependents run; what finished before is not repeated
    rc2, runner2, _, lines2, _ = run_plan(tmp_path, stand_in_plan(tmp_path))
    assert rc2 == 0 and runner2.status["cpu.a"] == "done" and runner2.status["cpu.b"] == "done" and runner2.status["final"] == "done" and runner2.status["g"] == "cached"
    assert not any(l.endswith("mps.x: start") for l in lines2)  # mps.x was done with the same command: not started again
    # a step that exits 0 but leaves its output missing counts as failed
    bad = [R.Step(id="s", lane="main", desc="", cmd=py("pass"), outputs=(tmp_path / "never.txt",))]
    rc3, runner3, state3, _, _ = run_plan(tmp_path / "x", bad) if (tmp_path / "x").mkdir() is None else (None,) * 5
    assert rc3 == 1 and runner3.status["s"] == "failed" and "outputs missing" in state3.steps["s"]["note"]
    # a command that cannot start is a failure with a log, not a crash
    nope = [R.Step(id="n", lane="main", desc="", cmd=["/no/such/binary"])]
    (tmp_path / "y").mkdir()
    rc4, runner4, _, _, logs4 = run_plan(tmp_path / "y", nope)
    assert rc4 == 1 and runner4.status["n"] == "failed" and "could not run" in (logs4 / "n.log").read_text()


def test_a_changed_command_invalidates_the_recorded_step_and_stop_on_failure_stops_the_others(tmp_path):
    steps = stand_in_plan(tmp_path)
    run_plan(tmp_path, steps)
    changed = stand_in_plan(tmp_path)
    changed[1] = R.Step(id="cpu.a", lane="cpu", desc="cpu 1", cmd=py("import pathlib; pathlib.Path(r'%s/a2.txt').write_text('y')" % (tmp_path / "out")), deps=("g",))
    rc, runner, _, lines, _ = run_plan(tmp_path, changed)
    assert rc == 0 and runner.status["cpu.a"] == "done"  # run again: its signature changed
    assert runner.status["g"] == "cached" and runner.status["cpu.b"] == "cached"  # the others are not touched
    sleeper = [R.Step(id="g", lane="main", desc="", cmd=py("raise SystemExit(2)")), R.Step(id="slow", lane="cpu", desc="", cmd=py("import time; time.sleep(30)")),
               R.Step(id="after", lane="cpu", desc="", cmd=py("pass"), deps=("g",))]  # fmt: skip
    (tmp_path / "z").mkdir()
    t0 = __import__("time").time()
    rc2, runner2, _, _, _ = run_plan(tmp_path / "z", sleeper, stop_on_failure=True)
    assert rc2 == 1 and __import__("time").time() - t0 < 20 and runner2.status["g"] == "failed"  # the running 30 s step was terminated


def test_selection_and_the_signature_of_steps(tmp_path):
    steps = R.build_plan(cfg_under(tmp_path))
    assert [s.id for s in R.select(steps, "cpu.", "")] == [s.id for s in steps if s.id.startswith("cpu.")]
    assert [s.id for s in R.select(steps, "tables,numbers", "tables.make")] == [s.id for s in steps if s.id.startswith(("tables", "numbers")) and s.id != "tables.make_paper_tables"]
    a, b = R.build_plan(cfg_under(tmp_path)), R.build_plan(cfg_under(tmp_path, cpu_workers=3))
    sig = {s.id: s.signature() for s in a}
    assert sig["cpu.hgb"] == {s.id: s.signature() for s in R.build_plan(cfg_under(tmp_path))}["cpu.hgb"] and {s.id: s.signature() for s in b}["cpu.hgb"] != sig["cpu.hgb"]
    assert {s.id: s.signature() for s in b}["mps.k2"] == sig["mps.k2"]  # the CPU budget does not touch the MPS steps


# --------------------------------------------------------------------------- the commands of the plan are accepted by the children's own parsers
class _Stop(Exception):
    pass


def _parses(monkeypatch, run_main, argv) -> object:
    """Run ``run_main(argv)`` up to the first ``ArgumentParser.parse_args`` of the child: an unknown or malformed option is the child's own SystemExit(2)."""
    import argparse

    real = argparse.ArgumentParser.parse_args

    def spy(self, args=None, namespace=None):
        ns = real(self, args, namespace)  # raises SystemExit(2) on an option the child does not know
        raise _Stop(ns)

    with monkeypatch.context() as m:
        m.setattr(argparse.ArgumentParser, "parse_args", spy)
        with pytest.raises(_Stop) as e:
            run_main(argv)
    return e.value.args[0]


def test_every_command_of_the_plan_is_accepted_by_the_parser_of_the_module_it_runs(tmp_path, monkeypatch):
    import importlib
    import importlib.util

    if str(R.KR) not in sys.path:
        sys.path.insert(0, str(R.KR))
    steps = R.build_plan(cfg_under(tmp_path, with_lc=True, with_mlp=True, with_m11r_union=True, with_exploratory=True, mps_streams=2))
    checked = []
    for s in steps:
        if s.cmd is None or s.id == "gate.leak_tests":
            continue
        cmd = s.cmd
        if cmd[1] == "-m":
            name = cmd[2]
            module = importlib.import_module("model_stgcn.sagong2025.__main__" if name == "model_stgcn.sagong2025" else name)
            ns = _parses(monkeypatch, module.main, cmd[3:])
        elif cmd[1].endswith("make_paper_tables.py"):
            spec = importlib.util.spec_from_file_location("mpt_plan", cmd[1])
            mod = importlib.util.module_from_spec(spec)
            sys.modules["mpt_plan"] = mod
            spec.loader.exec_module(mod)
            ns = _parses(monkeypatch, mod.main, cmd[2:])
        elif cmd[1].endswith("paper_numbers_v2.py"):
            from tools import paper_numbers_v2 as PN

            ns = _parses(monkeypatch, PN.main, cmd[2:])
        else:
            raise AssertionError(f"unhandled command of {s.id}: {cmd[:3]}")
        checked.append(s.id)
        assert ns is not None
    assert len(checked) >= 30 and "mps.k1" in checked and "cpu.union" in checked and "numbers.paper_numbers" in checked and "tables.make_paper_tables" in checked
    # the run_all stage union command is itself a run_matrix command: it must parse with run_matrix's parser (after run_all built it)
    from adaptation import run_all as RA

    roots = RA.Roots(tmp_path / "r", tmp_path / "d", tmp_path / "p", tmp_path / "k")
    from model_stgcn.ptr import run_matrix as RM

    ns = RM.parse_args(RA.union_command(roots, arms=("HGB-union", "M11R-union"), workers=6, threads=2, cohorts="a,b")[3:])
    assert ns.union and ns.only == ["arm=HGB-union,M11R-union", "cohort=a,b"] and ns.extra_features == str(roots.table)
