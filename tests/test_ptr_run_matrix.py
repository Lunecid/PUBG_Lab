"""The experiment-matrix runner: enumeration, filters, dry run, resumability, scheduling (fake Popen) and a few
real-subprocess integration runs on tiny synthetic cohorts.  DB-free."""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

import pytest

import pubg_lab.survival.ptr.models.hgb as H
import pubg_lab.survival.ptr.experiments.run_matrix as RM
import pubg_lab.survival.ptr.training.train as T
from pubg_lab.survival.ptr.models.arms import EXPLORATORY_ARMS, MAIN_ARMS
from pubg_lab.survival.ptr.data.synthetic import write_cohort_dumps, write_extra_tables

LARGE = ("erangel_tpp_squad", "taego_tpp_squad")
SMALL_COHORTS = ["karakin_tpp_squad", "paramo_tpp_squad", "erangel_fpp_squad", "taego_fpp_squad", "taego_tpp_duo",
                 "sanhok_tpp_squad", "rondo_tpp_squad"]  # fmt: skip


# --------------------------------------------------------------------------- enumeration
def test_enumeration_is_the_design_matrix():
    cohorts = SMALL_COHORTS + list(LARGE)  # C = 9, L = 2
    runs, notes = RM.enumerate_runs(cohorts)
    assert notes == [] and len(runs) == 12 * 9 + 13 * 2 == 134
    assert len({r.run_id for r in runs}) == len(runs)  # unique ids
    assert Counter(r.group for r in runs) == {"main": 81, "exploratory": 6, "seeds": 20, "hgb": 27}

    main = {(r.cohort, r.arm, r.seed) for r in runs if r.group == "main"}
    assert main == {(c, a, 42) for c in cohorts for a in MAIN_ARMS}  # 9 main arms x all cohorts at seed 42
    expl = {(r.cohort, r.arm, r.seed) for r in runs if r.group == "exploratory"}
    assert expl == {(c, a, 42) for c in LARGE for a in ("M11R-noProx", "M11R-noCombat", "M11R-noCopress")}
    assert set(EXPLORATORY_ARMS) == {a for _, a, _ in expl}
    seeds = {(r.cohort, r.arm, r.seed) for r in runs if r.group == "seeds"}
    assert seeds == {(c, a, s) for c in LARGE for a in ("M00", "M11", "M11R", "M11R-shuf", "M11RZ") for s in (13, 2026)}
    hgb = {(r.cohort, r.arm, r.seed) for r in runs if r.group == "hgb"}
    assert hgb == {(c, v, 42) for c in cohorts for v in ("HGB-cut", "HGB-lag", "HGB-noHab")}
    assert all(r.kind == ("hgb" if r.group == "hgb" else "neural") for r in runs)
    assert all(r.seed == 42 for r in runs if r.group in ("main", "exploratory", "hgb"))


def test_run_count_formula_and_the_real_registry_size():
    for n_cohorts, n_large in ((12, 2), (6, 0), (11, 1)):
        cohorts = [f"c{i}" for i in range(n_cohorts - n_large)] + list(LARGE[:n_large])
        runs, _ = RM.enumerate_runs(cohorts)
        assert len(runs) == 12 * n_cohorts + 13 * n_large
    assert RM.PROJECTED_REGISTRY.keys() >= set(LARGE)
    assert len(RM.PROJECTED_REGISTRY) == 12  # 13 pre-registered cohorts minus Vikendi (all remaining have >= 200)
    assert "vikendi_tpp_squad" not in RM.PROJECTED_REGISTRY
    runs, _ = RM.enumerate_runs(list(RM.PROJECTED_REGISTRY))
    assert len(runs) == 170


def test_large_cohorts_that_are_not_in_the_dump_dir_are_skipped_with_a_note():
    runs, notes = RM.enumerate_runs(["karakin_tpp_squad", "erangel_tpp_squad"])
    assert {r.cohort for r in runs if r.group in ("exploratory", "seeds")} == {"erangel_tpp_squad"}
    assert any("taego_tpp_squad" in n for n in notes)
    with pytest.raises(ValueError, match="taego_tpp_squad"):
        RM.enumerate_runs(["karakin_tpp_squad", "erangel_tpp_squad"], strict=True)
    runs, notes = RM.enumerate_runs(["karakin_tpp_squad"])  # none present
    assert Counter(r.group for r in runs) == {"main": 9, "hgb": 3} and len(notes) == 1
    custom, _ = RM.enumerate_runs(["a", "b"], large=("b",))
    assert {r.cohort for r in custom if r.group == "seeds"} == {"b"}


# --------------------------------------------------------------------------- --only filters
def test_only_filters_and_their_combination():
    runs, _ = RM.enumerate_runs(SMALL_COHORTS + list(LARGE))
    pick = lambda *only: RM.apply_only(runs, RM.parse_only(only))  # noqa: E731
    assert len(pick()) == len(runs)
    assert {r.cohort for r in pick("cohort=karakin_tpp_squad")} == {"karakin_tpp_squad"}
    assert {r.arm for r in pick("arm=M00,M11RZ")} == {"M00", "M11RZ"}
    assert {r.seed for r in pick("seed=13")} == {13}
    assert {r.kind for r in pick("kind=hgb")} == {"hgb"} and len(pick("kind=hgb")) == 27
    assert {r.group for r in pick("group=exploratory,seeds")} == {"exploratory", "seeds"}
    assert {r.arm for r in pick("arm=HGB-*")} == {"HGB-cut", "HGB-lag", "HGB-noHab"}  # fnmatch patterns
    both = pick("cohort=erangel_tpp_squad", "arm=M11*", "seed=42")  # entries are ANDed, values ORed
    assert {(r.cohort, r.seed) for r in both} == {("erangel_tpp_squad", 42)}
    assert {r.arm for r in both} >= {"M11", "M11R", "M11RZ", "M11R-shuf", "M11R-noHab", "M11RZ-noHab"}
    assert pick("cohort=nope") == []
    for bad in ("nokey", "color=red", "seed=abc"):
        with pytest.raises(ValueError):
            RM.parse_only([bad])


# --------------------------------------------------------------------------- argv / hash agree with the CLIs
def test_run_arguments_reproduce_the_clis_config_hash(tmp_path):
    root = tmp_path / "dumps"
    write_cohort_dumps(root, "karakin_tpp_squad", sizes={"train": 5, "val": 3, "test": 3}, seed=1)
    st = RM.Settings(dump_dir=str(root), out_dir=str(tmp_path / "out"), max_epochs=2, threads=1)
    neural = RM.RunSpec("neural", "karakin_tpp_squad", "M11RZ", 42, "main")
    hgb = RM.RunSpec("hgb", "karakin_tpp_squad", "HGB-lag", 42, "hgb")
    a, b = RM.build_argv(neural, st), RM.build_argv(hgb, st)
    assert a[1:3] == ["-m", 'pubg_lab.survival.ptr.training.train'] and b[1:3] == ["-m", 'pubg_lab.survival.ptr.models.hgb']
    assert a[a.index("--arm") + 1] == "M11RZ" and b[b.index("--variant") + 1] == "HGB-lag"
    assert a[a.index("--max-epochs") + 1] == "2" and "--max-epochs" not in b  # HGB has no epochs
    assert a[a.index("--labels") + 1] == "v2" and a[a.index("--device") + 1] == "cpu"
    assert a[a.index("--threads") + 1] == "1" and a[a.index("--match-types") + 1] == "official,competitive"
    assert RM.expected_hash(neural, st) == T.config_hash(T.parse_args(a[3:]))  # what the subprocess will stamp
    assert RM.expected_hash(hgb, st) == H.config_hash(H.parse_args(b[3:]))
    assert RM.expected_hash(neural, st) != RM.expected_hash(hgb, st)
    assert RM.expected_hash(neural, RM.Settings(**{**st.__dict__, "labels": "pipeline"})) != RM.expected_hash(neural, st)
    assert RM.expected_hash(neural, RM.Settings(**{**st.__dict__, "max_epochs": 3})) != RM.expected_hash(neural, st)
    assert RM.expected_hash(neural, RM.Settings(**{**st.__dict__, "threads": 4})) == RM.expected_hash(neural, st)
    any_types = RM.Settings(**{**st.__dict__, "match_types": None})
    assert RM.build_argv(neural, any_types)[RM.build_argv(neural, any_types).index("--match-types") + 1] == "any"


# --------------------------------------------------------------------------- fake Popen scheduler tests
class FakePopen:
    """Stands in for subprocess.Popen: 'runs' for ``ticks`` polls, then writes metrics.json (or fails)."""

    started: list[list[str]] = []
    live = 0
    max_live = 0
    fail_ids: set[str] = set()
    silent_ids: set[str] = set()  # exit 0 without writing metrics.json
    ticks = 2

    def __init__(self, argv, stdout=None, stderr=None, env=None, cwd=None, **kw):
        self.argv, self.env, self.cwd = argv, env, cwd
        flag = "--arm" if "--arm" in argv else "--variant"
        self.run_id = f"{argv[argv.index('--cohort') + 1]}/{argv[argv.index(flag) + 1]}/seed{argv[argv.index('--seed') + 1]}"
        self.n = 0
        self.returncode = None
        FakePopen.started.append(argv)
        FakePopen.live += 1
        FakePopen.max_live = max(FakePopen.max_live, FakePopen.live)
        if stdout is not None:
            stdout.write(f"fake log of {self.run_id}\n")
            stdout.flush()

    def poll(self):
        self.n += 1
        if self.returncode is None and self.n > FakePopen.ticks:
            FakePopen.live -= 1
            self.returncode = 3 if self.run_id in FakePopen.fail_ids else 0
            if self.returncode == 0 and self.run_id not in FakePopen.silent_ids:
                self._write_metrics()
        return self.returncode

    def terminate(self):
        if self.returncode is None:
            FakePopen.live -= 1
            self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode

    def _write_metrics(self):
        a = self.argv
        mod = "train" if a[2].endswith("train") else "hgb"
        parse = T.parse_args if mod == "train" else H.parse_args
        stamp = (T if mod == "train" else H).config_hash(parse(a[3:]))
        out = Path(a[a.index("--out-dir") + 1]) / a[a.index("--cohort") + 1] / self.run_id.split("/")[1] / self.run_id.split("/")[2]
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps({"config_hash": stamp, "labels": a[a.index("--labels") + 1]}))


@pytest.fixture()
def fake(monkeypatch):
    FakePopen.started, FakePopen.live, FakePopen.max_live = [], 0, 0
    FakePopen.fail_ids, FakePopen.silent_ids, FakePopen.ticks = set(), set(), 2
    return FakePopen


def tiny_dump(tmp_path, names=("karakin_tpp_squad", "paramo_tpp_squad")):
    root = tmp_path / "dumps"
    for k, name in enumerate(names):
        write_cohort_dumps(root, name, sizes={"train": 5, "val": 3, "test": 3}, seed=k)
    return root


def run_main(root, out, *extra):
    return RM.main(["--dump-dir", str(root), "--out-dir", str(out), "--max-epochs", "1", *extra])


def read_status(out):
    with open(Path(out) / "run_matrix_status.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_scheduler_respects_the_worker_limit_runs_everything_and_writes_logs_and_status(tmp_path, fake):
    root, out = tiny_dump(tmp_path), tmp_path / "out"
    rc = RM.main(["--dump-dir", str(root), "--out-dir", str(out), "--workers", "3", "--threads", "2",
                  "--poll-seconds", "0", "--only", "kind=neural"], popen=fake)  # fmt: skip
    assert rc == 0
    assert len(fake.started) == 18 and fake.max_live == 3  # 2 cohorts x 9 arms, never more than 3 at once
    rows = read_status(out)
    assert len(rows) == 18 and {r["status"] for r in rows} == {"done"}
    assert {"run_id", "group", "kind", "cohort", "arm", "seed", "labels", "config_hash", "status", "exit_code",
            "start_utc", "end_utc", "elapsed_sec", "log", "metrics"} <= set(rows[0])  # fmt: skip
    for r in rows:
        assert Path(r["log"]).is_file() and "fake log" in Path(r["log"]).read_text()
        assert re.search(r"logs[/\\]" + r["cohort"] + "__" + re.escape(r["arm"]) + "__seed42.log$", r["log"])
        assert Path(r["metrics"]).is_file() and r["exit_code"] == "0" and float(r["elapsed_sec"]) >= 0
    # children get the thread limits both as arguments and as environment variables
    assert all(a[a.index("--threads") + 1] == "2" for a in fake.started)


def test_worker_environment_limits_threads(tmp_path, monkeypatch):
    seen = []

    class Spy(FakePopen):
        def __init__(self, argv, **kw):
            super().__init__(argv, **kw)
            seen.append(kw.get("env"))

    FakePopen.started, FakePopen.live, FakePopen.max_live = [], 0, 0
    FakePopen.fail_ids, FakePopen.silent_ids = set(), set()
    root = tiny_dump(tmp_path, ("karakin_tpp_squad",))
    RM.main(["--dump-dir", str(root), "--out-dir", str(tmp_path / "o"), "--workers", "1", "--threads", "3",
             "--poll-seconds", "0", "--only", "arm=M00"], popen=Spy)  # fmt: skip
    env = seen[0]
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        assert env[key] == "3"
    assert env["PYTHONUNBUFFERED"] == "1" and str(RM.REPO_ROOT) in env["PYTHONPATH"]


def test_default_workers_and_threads_follow_the_core_count(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    assert RM.default_workers() == 8 and RM.default_threads(8) == 2
    monkeypatch.setattr(os, "cpu_count", lambda: 15)
    assert RM.default_workers() == 7 and RM.default_threads(7) == 2
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    assert RM.default_workers() == 1 and RM.default_threads(1) == 2
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    assert RM.default_threads(32) == 2  # capped at 2 threads per worker
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    assert RM.default_threads(4) == 1


def test_heaviest_runs_are_started_first(tmp_path, fake):
    root = tiny_dump(tmp_path)
    RM.main(["--dump-dir", str(root), "--out-dir", str(tmp_path / "o"), "--workers", "1", "--poll-seconds", "0"],
            popen=fake)  # fmt: skip
    order = [a[a.index("--arm") + 1] if "--arm" in a else a[a.index("--variant") + 1] for a in fake.started]
    assert order.index("M11RZ") < order.index("M00")  # costlier arm of a cohort goes before the cheap one
    neural_first = [("--arm" in a) for a in fake.started]
    assert neural_first.index(False) > 0  # HGB runs are scheduled by cost too, never before every neural run


def test_resume_skips_runs_with_the_same_config_hash_and_reruns_stale_ones(tmp_path, fake):
    root, out = tiny_dump(tmp_path), tmp_path / "out"
    common = ["--workers", "4", "--poll-seconds", "0", "--only", "cohort=karakin_tpp_squad", "--only", "arm=M00,M11,HGB-cut"]
    assert RM.main(["--dump-dir", str(root), "--out-dir", str(out), "--max-epochs", "1", *common], popen=fake) == 0
    assert len(fake.started) == 3
    # second invocation, same settings: everything is skipped, nothing starts
    fake.started.clear()
    assert RM.main(["--dump-dir", str(root), "--out-dir", str(out), "--max-epochs", "1", *common], popen=fake) == 0
    assert fake.started == []
    assert {r["status"] for r in read_status(out)} == {"skipped_done"}
    # a changed setting changes the hash: the neural runs repeat, the HGB run (no epochs) does not
    assert RM.main(["--dump-dir", str(root), "--out-dir", str(out), "--max-epochs", "2", *common], popen=fake) == 0
    assert sorted(a[a.index("--arm") + 1] for a in fake.started) == ["M00", "M11"]
    status = {r["arm"]: r["status"] for r in read_status(out)}
    assert status == {"M00": "done", "M11": "done", "HGB-cut": "skipped_done"}
    # a re-dumped cohort (new file times) invalidates all of its runs
    fake.started.clear()
    os.utime(root / "karakin_tpp_squad" / "train.pt", ns=(5, 5))
    RM.main(["--dump-dir", str(root), "--out-dir", str(out), "--max-epochs", "2", *common], popen=fake)
    assert len(fake.started) == 3
    # a deleted metrics.json re-runs exactly that run; --force re-runs everything
    fake.started.clear()
    (out / "karakin_tpp_squad" / "M00" / "seed42" / "metrics.json").unlink()
    RM.main(["--dump-dir", str(root), "--out-dir", str(out), "--max-epochs", "2", *common], popen=fake)
    assert [a[a.index("--arm") + 1] for a in fake.started] == ["M00"]
    fake.started.clear()
    RM.main(["--dump-dir", str(root), "--out-dir", str(out), "--max-epochs", "2", "--force", *common], popen=fake)
    assert len(fake.started) == 3
    # an unreadable (truncated) metrics.json counts as not done
    (out / "karakin_tpp_squad" / "M11" / "seed42" / "metrics.json").write_text("{trunc")
    fake.started.clear()
    RM.main(["--dump-dir", str(root), "--out-dir", str(out), "--max-epochs", "2", *common], popen=fake)
    assert [a[a.index("--arm") + 1] for a in fake.started] == ["M11"]


def test_failed_runs_are_recorded_and_do_not_stop_the_others(tmp_path, fake):
    root, out = tiny_dump(tmp_path), tmp_path / "out"
    fake.fail_ids = {"karakin_tpp_squad/M00/seed42"}
    fake.silent_ids = {"paramo_tpp_squad/M01/seed42"}  # exit code 0 but no metrics.json: also a failure
    rc = RM.main(["--dump-dir", str(root), "--out-dir", str(out), "--workers", "3", "--poll-seconds", "0",
                  "--only", "kind=neural"], popen=fake)  # fmt: skip
    assert rc == 1  # non-zero if anything failed
    rows = {r["run_id"]: r for r in read_status(out)}
    assert rows["karakin_tpp_squad/M00/seed42"]["status"] == "failed" and rows["karakin_tpp_squad/M00/seed42"]["exit_code"] == "3"
    assert rows["paramo_tpp_squad/M01/seed42"]["status"] == "failed" and "metrics" in rows["paramo_tpp_squad/M01/seed42"]["note"]
    assert Counter(r["status"] for r in rows.values()) == {"done": 16, "failed": 2}
    assert len(fake.started) == 18  # everything was attempted
    # the next invocation retries exactly the failed runs
    fake.fail_ids, fake.silent_ids = set(), set()
    fake.started.clear()
    assert RM.main(["--dump-dir", str(root), "--out-dir", str(out), "--workers", "3", "--poll-seconds", "0",
                    "--only", "kind=neural"], popen=fake) == 0  # fmt: skip
    assert len(fake.started) == 2
    assert Counter(r["status"] for r in read_status(out)) == {"skipped_done": 16, "done": 2}


# --------------------------------------------------------------------------- dry run
def test_dry_run_prints_the_plan_and_changes_nothing(tmp_path, capsys, fake):
    root, out = tiny_dump(tmp_path), tmp_path / "out"
    rc = RM.main(["--dump-dir", str(root), "--out-dir", str(out), "--dry-run", "--workers", "4"], popen=fake)
    text = capsys.readouterr().out
    assert rc == 0 and fake.started == [] and not out.exists()  # nothing launched, nothing written
    assert "runs: 24 = main 18 + exploratory 0 + seeds 0 + hgb 6" in text
    assert "to run: 24" in text and "already done" in text
    assert "workers: 4" in text and "estimated wall-clock" in text
    assert "karakin_tpp_squad" in text and "erangel_tpp_squad" in text  # the note about the missing large cohorts
    rc = RM.main(["--dump-dir", str(root), "--out-dir", str(out), "--dry-run", "--only", "arm=M00", "--list"], popen=fake)
    text = capsys.readouterr().out
    assert rc == 0 and "runs: 2 = main 2" in text
    assert "karakin_tpp_squad/M00/seed42" in text and "paramo_tpp_squad/M00/seed42" in text and "M11RZ" not in text


def test_dry_run_counts_done_runs_and_the_estimate_ignores_them(tmp_path, capsys, fake):
    root, out = tiny_dump(tmp_path), tmp_path / "out"
    common = ["--workers", "2", "--poll-seconds", "0", "--only", "cohort=karakin_tpp_squad", "--only", "arm=M00,M11RZ"]
    RM.main(["--dump-dir", str(root), "--out-dir", str(out), "--max-epochs", "1", *common], popen=fake)
    capsys.readouterr()
    RM.main(["--dump-dir", str(root), "--out-dir", str(out), "--max-epochs", "1", "--dry-run", *common], popen=fake)
    text = capsys.readouterr().out
    assert "to run: 0 (already done with the same config hash: 2)" in text
    RM.main(["--dump-dir", str(root), "--out-dir", str(out), "--max-epochs", "3", "--dry-run", *common], popen=fake)
    assert "to run: 2 (already done with the same config hash: 0)" in capsys.readouterr().out


def test_projected_dry_run_uses_the_registry_sizes_without_any_dump(tmp_path, capsys):
    rc = RM.main(["--dump-dir", str(tmp_path / "does_not_exist"), "--out-dir", str(tmp_path / "o"), "--dry-run",
                  "--projected", "--workers", "7"])  # fmt: skip
    text = capsys.readouterr().out
    assert rc == 0 and "projected registry" in text
    assert "runs: 170 = main 108 + exploratory 6 + seeds 20 + hgb 36" in text
    m = re.search(r"wall-clock with 7 workers: ~([0-9.]+) h", text)
    assert m and 0.05 < float(m.group(1)) < 200
    assert not (tmp_path / "o").exists()


def test_estimates_grow_with_the_data_and_the_arm():
    st = RM.Settings(dump_dir="x", out_dir="y", max_epochs=12, threads=2)
    small = RM.CohortStats(n_train=267, n_val=33, n_test=33, mean_teams=17.7, rows_per_match=36.8)
    big = RM.CohortStats(n_train=2626, n_val=328, n_test=328, mean_teams=26.4, rows_per_match=60.0)
    solo = RM.CohortStats(n_train=179, n_val=22, n_test=22, mean_teams=99.4, rows_per_match=120.0)
    est = lambda arm, stats, kind="neural": RM.estimate_seconds(RM.RunSpec(kind, "c", arm, 42, "main"), stats, st)  # noqa: E731
    assert est("M00", small) < est("M00", big) and est("M00", big) < est("M00", big) * 2
    assert est("M00", big) < est("M11R", big) < est("M11RZ", big) * 1.5 and est("M11", big) < est("M11RZ", big)
    assert est("M00", solo) > est("M00", small) * 0.5  # many teams per match cost more per match
    assert est("HGB-cut", small, "hgb") < est("HGB-lag", small, "hgb") < est("HGB-lag", big, "hgb")
    short = RM.Settings(dump_dir="x", out_dir="y", max_epochs=3, threads=2)
    full = RM.estimate_seconds(RM.RunSpec("neural", "c", "M11RZ", 42, "main"), big, st)
    assert RM.estimate_seconds(RM.RunSpec("neural", "c", "M11RZ", 42, "main"), big, short) < full / 2
    assert RM.makespan([10, 10, 10, 10], 2) == 20 and RM.makespan([30, 10, 10, 10], 2) == 30 and RM.makespan([], 3) == 0


# --------------------------------------------------------------------------- real subprocess integration
def test_real_subprocess_runs_end_to_end_resume_and_failure(tmp_path):
    root = tiny_dump(tmp_path, ("karakin_tpp_squad", "paramo_tpp_squad"))
    out = tmp_path / "out"
    argv = ["--dump-dir", str(root), "--out-dir", str(out), "--max-epochs", "1", "--workers", "2", "--threads", "1",
            "--only", "arm=M00,HGB-cut", "--poll-seconds", "0.1"]  # fmt: skip
    assert RM.main(argv) == 0
    rows = read_status(out)
    assert len(rows) == 4 and {r["status"] for r in rows} == {"done"}
    for r in rows:
        met = json.loads(Path(r["metrics"]).read_text())
        assert met["config_hash"] == r["config_hash"] and met["labels"] == "v2"
        assert "alive tokens" in Path(r["log"]).read_text()  # the child's own log, one file per run
    assert (out / "paramo_tpp_squad" / "HGB-cut" / "seed42" / "pred_test.csv").is_file()
    assert (out / "karakin_tpp_squad" / "M00" / "seed42" / "best.pt").is_file()
    # second invocation: nothing to do
    assert RM.main(argv) == 0
    assert {r["status"] for r in read_status(out)} == {"skipped_done"}
    # the paired utility accepts the matrix's outputs (identical rows between neural and HGB runs)
    import pubg_lab.survival.ptr.evaluation.metrics as M

    res = M.paired_bootstrap(
        out / "karakin_tpp_squad" / "M00" / "seed42" / "pred_test.csv",
        out / "karakin_tpp_squad" / "HGB-cut" / "seed42" / "pred_test.csv",
        n_boot=10,
    )
    assert res["n_rows"] > 0
    # a corrupted cohort fails (loudly, in its log) without stopping a healthy one; labels guard is respected
    (root / "paramo_tpp_squad" / "train.pt").write_bytes(b"not a torch file")
    argv2 = argv[:-2] + ["--poll-seconds", "0.1", "--force"]
    assert RM.main(argv2) == 1
    rows = {(r["cohort"], r["arm"]): r for r in read_status(out)}
    assert rows[("karakin_tpp_squad", "M00")]["status"] == "done"
    bad = rows[("paramo_tpp_squad", "M00")]
    assert bad["status"] == "failed" and Path(bad["log"]).read_text().strip()
    # other label choice into a directory that holds v2 results: refused by the run itself, recorded as failed
    argv3 = ["--dump-dir", str(root), "--out-dir", str(out), "--max-epochs", "1", "--workers", "1", "--threads", "1",
             "--only", "cohort=karakin_tpp_squad", "--only", "arm=M00", "--labels", "pipeline", "--poll-seconds", "0.1"]  # fmt: skip
    assert RM.main(argv3) == 1
    row = read_status(out)[0]
    assert row["status"] == "failed" and "labels" in Path(row["log"]).read_text()
    assert json.loads((out / "karakin_tpp_squad" / "M00" / "seed42" / "metrics.json").read_text())["labels"] == "v2"


def test_population_json_drives_the_projection(tmp_path, capsys):
    import json

    pop = {"cohorts": {
        "a_tpp_squad": {"status": "kept", "after": {"train": 80, "val": 10, "test": 10, "total": 100}},
        "b_tpp_squad": {"status": "excluded", "after": {"train": 0, "val": 0, "test": 0, "total": 0}},
        "erangel_tpp_solo": {"status": "kept", "after": {"train": 160, "val": 20, "test": 20, "total": 200}},
    }}
    (tmp_path / "population.json").write_text(json.dumps(pop))
    stats = RM.population_stats(tmp_path)
    assert sorted(stats) == ["a_tpp_squad", "erangel_tpp_solo"]  # excluded cohorts are not planned
    assert stats["a_tpp_squad"].n_train == 80 and stats["a_tpp_squad"].mean_teams == RM.DEFAULT_TEAMS
    assert stats["erangel_tpp_solo"].mean_teams == 99.4  # team count from the registry when known
    assert RM.population_stats(tmp_path / "nowhere") is None
    assert RM.main(["--dump-dir", str(tmp_path), "--out-dir", str(tmp_path / "o"), "--dry-run", "--projected"]) == 0
    assert "cohorts (2)" in capsys.readouterr().out


# --------------------------------------------------------------------------- stage 2 (player layer)
def test_stage_two_plan_argv_and_hash(tmp_path, capsys):
    from pubg_lab.survival.ptr.data.synthetic import write_player_dumps

    root = tmp_path / "dumps"
    for c in ("karakin_tpp_squad", "paramo_tpp_squad"):
        write_cohort_dumps(root / "bigcomp2027_phase_v1", c, sizes={"train": 5, "val": 3, "test": 3}, seed=1)
    write_player_dumps(root / "bigcomp2027_phase_v1", "karakin_tpp_squad", root / "bigcomp2027_player_v1", seed=2)  # paramo has none
    runs = RM.enumerate_stage2_runs(["karakin_tpp_squad", "paramo_tpp_squad"])
    assert [(r.cohort, r.arm, r.seed, r.group, r.kind) for r in runs] == [
        (c, a, 42, "stage2", "neural") for c in ("karakin_tpp_squad", "paramo_tpp_squad") for a in ("M11RZ-P", "M11RZ-P-tm")
    ]
    assert len(RM.enumerate_stage2_runs(list(RM.PROJECTED_REGISTRY))) == 24
    st = RM.Settings(dump_dir=str(root / "bigcomp2027_phase_v1"), out_dir=str(tmp_path / "o"), max_epochs=2, threads=1)
    argv = RM.build_argv(runs[0], st)
    assert "--player-dump-dir" not in argv  # default: next to the phase dump (train.py resolves it)
    assert RM.expected_hash(runs[0], st) == T.config_hash(T.parse_args(argv[3:]))
    st2 = RM.Settings(**{**st.__dict__, "player_dump_dir": str(root / "elsewhere")})
    argv2 = RM.build_argv(runs[0], st2)
    assert argv2[argv2.index("--player-dump-dir") + 1] == str(root / "elsewhere")
    assert "--player-dump-dir" not in RM.build_argv(RM.RunSpec("neural", "karakin_tpp_squad", "M11RZ", 42, "main"), st2)  # team arms never
    # the dry run plans only the cohorts that have a complete player dump, and says so
    base = ["--dump-dir", str(root / "bigcomp2027_phase_v1"), "--out-dir", str(tmp_path / "o"), "--stage2", "--dry-run", "--workers", "2"]
    assert RM.main(base) == 0
    text = capsys.readouterr().out
    assert "runs: 2 = stage 2 (M11RZ-P, M11RZ-P-tm) x 1 cohorts" in text and "paramo_tpp_squad" in text and "no complete player dump" in text
    with pytest.raises(ValueError, match="no complete player dump"):
        RM.main([*base, "--strict"])
    with pytest.raises(SystemExit):
        RM.main([*base, "--lc"])
    # estimates: the player arms cost more than M11RZ, never less
    stats = RM.CohortStats(n_train=500, n_val=60, n_test=60, mean_teams=26.0, rows_per_match=57.0)
    st3 = RM.Settings(dump_dir="x", out_dir="y", max_epochs=12, threads=2)
    est = lambda arm: RM.estimate_seconds(RM.RunSpec("neural", "c", arm, 42, "stage2"), stats, st3)  # noqa: E731
    assert est("M11RZ") < est("M11RZ-P") and est("M11RZ-P") == est("M11RZ-P-tm")


def test_stage_two_real_subprocess_run_writes_the_player_arm(tmp_path):
    from pubg_lab.survival.ptr.data.synthetic import write_player_dumps

    root = tmp_path / "dumps"
    write_cohort_dumps(root / "bigcomp2027_phase_v1", "karakin_tpp_squad", sizes={"train": 6, "val": 3, "test": 3}, seed=1)
    write_player_dumps(root / "bigcomp2027_phase_v1", "karakin_tpp_squad", root / "bigcomp2027_player_v1", seed=2)
    out = tmp_path / "out"
    argv = ["--dump-dir", str(root / "bigcomp2027_phase_v1"), "--out-dir", str(out), "--stage2", "--max-epochs", "1", "--workers", "2",
            "--threads", "1", "--poll-seconds", "0.1"]  # fmt: skip
    assert RM.main(argv) == 0
    rows = read_status(out)
    assert {(r["arm"], r["group"], r["status"]) for r in rows} == {("M11RZ-P", "stage2", "done"), ("M11RZ-P-tm", "stage2", "done")}
    for r in rows:
        met = json.loads(Path(r["metrics"]).read_text())
        assert met["config_hash"] == r["config_hash"] and "player" in met["data"]
    assert RM.main(argv) == 0 and {r["status"] for r in read_status(out)} == {"skipped_done"}


def test_status_file_name_can_be_changed_so_added_runs_do_not_overwrite_the_matrix_status(tmp_path):
    root = tiny_dump(tmp_path, ("karakin_tpp_squad",))
    out = tmp_path / "out"
    base = ["--dump-dir", str(root), "--out-dir", str(out), "--max-epochs", "1", "--workers", "1", "--threads", "1", "--poll-seconds", "0.1",
            "--only", "arm=M00"]  # fmt: skip
    assert RM.main(base) == 0
    assert RM.main([*base[:-1], "arm=HGB-cut", "--status-name", "run_matrix_status_extra.csv"]) == 0
    first = (out / "run_matrix_status.csv").read_text()
    assert "M00" in first and "HGB-cut" not in first  # the first status file is untouched
    assert "HGB-cut" in (out / "run_matrix_status_extra.csv").read_text()


# --------------------------------------------------------------------------- union (design doc 4.6, post hoc)
def test_union_enumeration_is_the_28_runs_of_the_real_registry():
    cohorts = list(RM.PROJECTED_REGISTRY)
    runs = RM.enumerate_union_runs(cohorts)
    assert RM.UNION_GROUP == "union" and RM.GROUPS == ("main", "exploratory", "seeds", "hgb")  # the main plan is not widened
    assert len(runs) == 12 + 12 + 4 == 28 and Counter(r.group for r in runs) == {"union": 28}
    assert len({r.run_id for r in runs}) == 28
    assert {(r.cohort, r.seed, r.kind) for r in runs if r.arm == "HGB-union"} == {(c, 42, "hgb") for c in cohorts}
    assert {(r.cohort, r.seed, r.kind) for r in runs if r.arm == "M11R-union" and r.seed == 42} == {(c, 42, "neural") for c in cohorts}
    extra = {(r.cohort, r.seed) for r in runs if r.arm == "M11R-union" and r.seed != 42}
    assert extra == {(c, s) for c in LARGE for s in (13, 2026)}  # no other arm / seed is planned
    assert {r.arm for r in runs} == {"HGB-union", "M11R-union"}
    few = RM.enumerate_union_runs(["karakin_tpp_squad"])
    assert [(r.arm, r.seed) for r in few] == [("HGB-union", 42), ("M11R-union", 42)]  # large cohorts absent: no extra seeds


def test_union_plan_argv_and_hash(tmp_path, capsys):
    root = tmp_path / "dumps"
    phase = root / "bigcomp2027_phase_v1"
    for c in ("karakin_tpp_squad", "paramo_tpp_squad"):
        write_cohort_dumps(phase, c, sizes={"train": 5, "val": 3, "test": 3}, seed=1)
    kf = root / "bigcomp2027_kraftonfeat_v1"
    write_extra_tables(phase, "karakin_tpp_squad", kf, seed=2)  # paramo has no table
    st = RM.Settings(dump_dir=str(phase), out_dir=str(tmp_path / "o"), max_epochs=2, threads=1, extra_features_dir=str(kf))
    runs = RM.enumerate_union_runs(["karakin_tpp_squad"])
    for r in runs:
        argv = RM.build_argv(r, st)
        assert argv[argv.index("--extra-features") + 1] == str(kf)
        expected = (T.config_hash(T.parse_args(argv[3:])) if r.kind == "neural" else H.config_hash(H.parse_args(argv[3:])))
        assert RM.expected_hash(r, st) == expected  # what the subprocess will stamp (feature-set hash + table fingerprints inside)
    plain = RM.RunSpec("neural", "karakin_tpp_squad", "M11R", 42, "main")
    assert "--extra-features" not in RM.build_argv(plain, st)  # only the union runs get the flag
    assert RM.expected_hash(plain, st) == T.config_hash(T.parse_args(RM.build_argv(plain, st)[3:]))
    with pytest.raises(ValueError, match="extra"):
        RM.build_argv(runs[0], RM.Settings(dump_dir=str(phase), out_dir="o"))
    # the dry run plans the cohorts that have a complete kf table and says what it left out
    base = ["--dump-dir", str(phase), "--out-dir", str(tmp_path / "o"), "--union", "--dry-run", "--workers", "2"]
    assert RM.main(base) == 0  # default table directory: bigcomp2027_kraftonfeat_v1 next to the phase dump
    text = capsys.readouterr().out
    assert "runs: 2 = union (HGB-union, M11R-union; post hoc" in text and "x 1 cohorts" in text
    assert "no complete kf table" in text and "paramo_tpp_squad" in text
    with pytest.raises(ValueError, match="no complete kf table"):
        RM.main([*base, "--strict"])
    for other in ("--lc", "--stage2", "--projected"):
        with pytest.raises(SystemExit):
            RM.main([*base, other])
    # a table directory that is given explicitly wins
    assert RM.main([*base, "--extra-features", str(root / "elsewhere")]) == 0
    assert "runs: 0 = union" in capsys.readouterr().out
    # estimates: the union HGB has 55 more columns than HGB-cut, the union network costs what M11R costs
    stats = RM.CohortStats(n_train=500, n_val=60, n_test=60, mean_teams=26.0, rows_per_match=57.0)
    st3 = RM.Settings(dump_dir="x", out_dir="y", max_epochs=12, threads=2)
    est = lambda kind, arm: RM.estimate_seconds(RM.RunSpec(kind, "c", arm, 42, "union"), stats, st3)  # noqa: E731
    assert est("hgb", "HGB-union") > est("hgb", "HGB-cut") and est("neural", "M11R-union") == est("neural", "M11R")


def test_union_real_subprocess_run_writes_both_arms_and_resumes(tmp_path):
    root = tmp_path / "dumps"
    phase = root / "bigcomp2027_phase_v1"
    write_cohort_dumps(phase, "karakin_tpp_squad", sizes={"train": 6, "val": 3, "test": 3}, seed=1)
    write_extra_tables(phase, "karakin_tpp_squad", root / "bigcomp2027_kraftonfeat_v1", seed=2)
    out = tmp_path / "out"
    argv = ["--dump-dir", str(phase), "--out-dir", str(out), "--union", "--max-epochs", "1", "--workers", "2", "--threads", "1",
            "--poll-seconds", "0.1", "--status-name", "run_matrix_status_union.csv"]  # fmt: skip
    assert RM.main(argv) == 0
    rows = [r for r in csv.DictReader(open(out / "run_matrix_status_union.csv"))]
    assert {(r["arm"], r["group"], r["status"]) for r in rows} == {("HGB-union", "union", "done"), ("M11R-union", "union", "done")}
    for r in rows:
        met = json.loads(Path(r["metrics"]).read_text())
        assert met["config_hash"] == r["config_hash"] and met["extra_features"]["n_features"] == 4
        assert met["extra_features"]["join"]["test"]["alignment"]["checked"]
    assert RM.main(argv) == 0
    assert {r["status"] for r in csv.DictReader(open(out / "run_matrix_status_union.csv"))} == {"skipped_done"}


# --------------------------------------------------------------------------- rank-loss control (supplementary experiment E1)
def test_norank_enumeration_and_plan(tmp_path):
    cohorts = list(RM.PROJECTED_REGISTRY)
    runs = RM.enumerate_norank_runs(cohorts)
    assert RM.NORANK_GROUP == "norank" and RM.GROUPS == ("main", "exploratory", "seeds", "hgb")  # the main plan is not widened
    assert len(runs) == 24 and Counter(r.group for r in runs) == {"norank": 24} and len({r.run_id for r in runs}) == 24
    assert {r.arm for r in runs} == {"M01-norank", "M11R-norank"} and {r.seed for r in runs} == {42} and {r.kind for r in runs} == {"neural"}
    assert {r.cohort for r in runs} == set(cohorts)
    assert RM.ARM_CLASS["M01-norank"] == "base" and RM.ARM_CLASS["M11R-norank"] == "rel"
    st = RM.Settings(dump_dir=str(tmp_path / "d"), out_dir=str(tmp_path / "o"), max_epochs=3, threads=1)
    for r in runs[:2]:
        argv = RM.build_argv(r, st)
        assert "--rank-weight" not in argv  # the arm carries its loss; the command line does not repeat it
        assert T.parse_args(argv[3:]).rank_weight == 0.0
        assert RM.expected_hash(r, st) == T.config_hash(T.parse_args(argv[3:]))
    main_m01 = RM.RunSpec("neural", runs[0].cohort, "M01", 42, "main")
    assert RM.expected_hash(main_m01, st) != RM.expected_hash(RM.RunSpec("neural", runs[0].cohort, "M01-norank", 42, "norank"), st)


def test_norank_flag_is_exclusive_with_the_other_plans(tmp_path, capsys):
    root = tmp_path / "dumps"
    for c in ("karakin_tpp_squad", "paramo_tpp_squad"):
        write_cohort_dumps(root, c, sizes={"train": 5, "val": 3, "test": 3}, seed=1)
    base = ["--dump-dir", str(root), "--out-dir", str(tmp_path / "o"), "--dry-run"]
    assert RM.main([*base, "--norank"]) == 0
    text = capsys.readouterr().out
    assert "runs: 4 = rank-loss control (M01-norank, M11R-norank" in text and "x 2 cohorts" in text
    for other in ("--union", "--stage2", "--lc"):
        with pytest.raises(SystemExit):
            RM.main([*base, "--norank", other])


# --------------------------------------------------------------------------- the v2 roots, the version guard, no reuse of v1 status files or fingerprints
def v2_marked_dump(tmp_path, names=("karakin_tpp_squad", "paramo_tpp_squad")):
    """A synthetic dump whose manifests say ``version: v2`` (the guard reads the manifests; the fake runner never loads the tensors)."""
    root = tmp_path / "dumps" / "bigcomp2027_phase_v2"
    for k, name in enumerate(names):
        write_cohort_dumps(root, name, sizes={"train": 5, "val": 3, "test": 3}, seed=k)
        man_path = root / name / "dump_manifest.json"
        man = json.loads(man_path.read_text())
        man["version"] = "v2"
        man_path.write_text(json.dumps(man))
    return root


def test_defaults_are_the_v2_roots_and_the_kf_table_follows_the_dump_version(tmp_path):
    import pubg_lab.survival.ptr.data.paths as P

    a = RM.parse_args([])
    assert a.dump_dir == str(P.DUMP_DIR) and a.out_dir == str(P.RUNS_DIR) and a.extra_features is None
    assert P.kf_dir_for(P.DUMP_DIR) == P.KF_DIR  # the union runs of the v2 dump read the v2 table, never the v1 one next to it
    v2 = v2_marked_dump(tmp_path)
    assert P.kf_dir_for(v2) == tmp_path / "dumps" / "bigcomp2027_kraftonfeat_v2"
    assert P.kf_dir_for(tiny_dump(tmp_path / 'archive/legacy_snapshot')) == tmp_path / 'archive/legacy_snapshot/bigcomp2027_kraftonfeat_v1'  # a legacy dump keeps its v1 table


def test_a_v1_root_next_to_a_v2_dump_is_refused_before_anything_runs(tmp_path, fake):
    v2 = v2_marked_dump(tmp_path)
    v1_out = tmp_path / "checkpoints" / "bigcomp2027_ptr_v1"
    for extra in (["--dry-run"], []):
        with pytest.raises(SystemExit, match="refusing to run: refusing to mix versions"):
            RM.main(["--dump-dir", str(v2), "--out-dir", str(v1_out), *extra], popen=fake)
    assert not v1_out.exists() and fake.started == []  # the v1 run directory is never touched
    kf_v1 = tmp_path / "dumps" / "bigcomp2027_kraftonfeat_v1"
    with pytest.raises(SystemExit, match="kf table"):
        RM.main(["--dump-dir", str(v2), "--out-dir", str(tmp_path / "out_v2"), "--union", "--extra-features", str(kf_v1), "--dry-run"], popen=fake)
    # a v2 root of unknown origin that holds runs is refused too (a renamed v1 root), the same root with a stamp is fine
    renamed = tmp_path / "checkpoints" / "renamed"
    (renamed / "karakin_tpp_squad" / "M00" / "seed42").mkdir(parents=True)
    with pytest.raises(SystemExit, match="no version evidence"):
        RM.main(["--dump-dir", str(v2), "--out-dir", str(renamed), "--dry-run"], popen=fake)
    import pubg_lab.survival.ptr.data.paths as P

    P.write_stamp(renamed, version="v2")
    assert RM.main(["--dump-dir", str(v2), "--out-dir", str(renamed), "--dry-run"], popen=fake) == 0
    assert RM.main(["--dump-dir", str(v2), "--out-dir", str(tmp_path / "fresh_v2"), "--dry-run"], popen=fake) == 0 and not (tmp_path / "fresh_v2").exists()
    # a projection reads no dump: no comparison, nothing to refuse
    assert RM.main(["--dump-dir", str(tmp_path / "nowhere"), "--out-dir", str(v1_out), "--dry-run", "--projected"], popen=fake) == 0


def test_stage_2_never_pairs_a_v1_player_dump_with_a_v2_dump(tmp_path, fake):
    """The player dump that sits next to a v2 dump by default is ``bigcomp2027_player_v1``: ``--stage2`` compares it with the dump like every other root,
    so the v2 re-run (which leaves stage 2 out) cannot start it by accident."""
    v2 = v2_marked_dump(tmp_path)
    v1_player = tmp_path / "dumps" / "bigcomp2027_player_v1"
    for c in ("karakin_tpp_squad", "paramo_tpp_squad"):
        (v1_player / c).mkdir(parents=True)
        for split in ("train", "val", "test"):
            (v1_player / c / f"{split}.pt").write_bytes(b"x")
    out = tmp_path / "checkpoints" / "bigcomp2027_ptr_v2"
    for extra in (["--dry-run"], []):
        with pytest.raises(SystemExit, match=r"refusing to run: refusing to mix versions.*player dump"):
            RM.main(["--dump-dir", str(v2), "--out-dir", str(out), "--stage2", *extra], popen=fake)  # the default player dump
        with pytest.raises(SystemExit, match=r"refusing to run: refusing to mix versions.*player dump"):
            RM.main(["--dump-dir", str(v2), "--out-dir", str(out), "--stage2", "--player-dump-dir", str(v1_player), *extra], popen=fake)  # the same one, explicit
    assert not out.exists() and fake.started == []
    # a player dump that is v2 (here: not rebuilt yet, so no cohort has one) is no version conflict; the main matrix never looks at the player dump
    assert RM.main(["--dump-dir", str(v2), "--out-dir", str(out), "--stage2", "--player-dump-dir", str(tmp_path / "dumps" / "bigcomp2027_player_v2"), "--dry-run"], popen=fake) == 0
    assert RM.main(["--dump-dir", str(v2), "--out-dir", str(out), "--dry-run"], popen=fake) == 0
    # a legacy (v1) dump keeps its v1 player dump
    legacy = tiny_dump(tmp_path / 'archive/legacy_snapshot')
    assert RM.main(["--dump-dir", str(legacy), "--out-dir", str(tmp_path / "legacy_out"), "--stage2", "--dry-run"], popen=fake) == 0


def test_the_run_directory_is_stamped_for_a_v2_dump_and_a_legacy_run_stays_untouched(tmp_path, fake):
    import pubg_lab.survival.ptr.data.paths as P

    v2 = v2_marked_dump(tmp_path)
    out = tmp_path / "checkpoints" / "bigcomp2027_ptr_v2"
    assert RM.main(["--dump-dir", str(v2), "--out-dir", str(out), "--workers", "2", "--poll-seconds", "0", "--only", "arm=M00"], popen=fake) == 0
    stamp = json.loads((out / P.STAMP_NAME).read_text())
    assert stamp["version"] == "v2" and stamp["role"] == "runs" and stamp["dump_meta_sha1"] == P.dump_meta_sha1(v2)
    assert (out / "run_matrix_status.csv").is_file() and {r["status"] for r in read_status(out)} == {"done"}
    # a second launch with the same dump keeps the stamp and skips what is done
    first = (out / P.STAMP_NAME).read_text()
    assert RM.main(["--dump-dir", str(v2), "--out-dir", str(out), "--workers", "2", "--poll-seconds", "0", "--only", "arm=M00"], popen=fake) == 0
    assert (out / P.STAMP_NAME).read_text() == first
    # the aggregator and the other readers still see a plain run directory (the stamp is a file, not a cohort)
    import pubg_lab.survival.ptr.evaluation.aggregate as A

    assert [p.name for p in A.scan_runs(out)[0]] == []  # no pred files from the fake runner: nothing to scan, and the stamp did not break the scan
    legacy = tiny_dump(tmp_path / 'archive/legacy_snapshot')
    lout = tmp_path / "legacy_out"
    assert RM.main(["--dump-dir", str(legacy), "--out-dir", str(lout), "--workers", "2", "--poll-seconds", "0", "--only", "arm=M00"], popen=fake) == 0
    assert not (lout / P.STAMP_NAME).exists()  # legacy roots stay byte-identical


def test_a_v1_metrics_file_or_status_file_is_never_reused_in_a_v2_root(tmp_path, fake, capsys):
    v2 = v2_marked_dump(tmp_path)
    out = tmp_path / "checkpoints" / "bigcomp2027_ptr_v2"
    # a metrics.json copied from a v1 run (another config hash) is not 'done': the run is planned again
    run = out / "karakin_tpp_squad" / "M00" / "seed42"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({"config_hash": "0123456789abcdef", "labels": "v2"}))
    import pubg_lab.survival.ptr.data.paths as P

    P.write_stamp(out, version="v2")
    base = ["--dump-dir", str(v2), "--out-dir", str(out), "--dry-run", "--only", "arm=M00"]
    assert RM.main(base, popen=fake) == 0
    text = capsys.readouterr().out
    assert "to run: 2 (already done with the same config hash: 0)" in text
    # a status CSV that was written for another root (its logs and metrics live in the v1 run directory) is refused
    status = out / "run_matrix_status.csv"
    status.write_text(",".join(RM.STATUS_FIELDS) + "\n" + ",".join(
        {"run_id": "karakin_tpp_squad/M00/seed42", "log": str(tmp_path / "checkpoints" / "bigcomp2027_ptr_v1" / "logs" / "x.log"),
         "metrics": str(tmp_path / "checkpoints" / "bigcomp2027_ptr_v1" / "karakin_tpp_squad" / "M00" / "seed42" / "metrics.json")}.get(f, "") for f in RM.STATUS_FIELDS) + "\n")
    with pytest.raises(SystemExit, match="outside .*another root"):
        RM.main(base, popen=fake)
    # the same file with paths inside the directory is a normal status file; a relative path of the repository is resolved there too
    status.write_text(",".join(RM.STATUS_FIELDS) + "\n" + ",".join(
        {"log": str(out / "logs" / "x.log"), "metrics": str(run / "metrics.json")}.get(f, "") for f in RM.STATUS_FIELDS) + "\n")
    assert RM.check_status_files(out) == [status]
    assert RM.main(base, popen=fake) == 0
    # unreadable rows are an error, a missing directory is nothing to check
    assert RM.check_status_files(tmp_path / "nowhere") == []
