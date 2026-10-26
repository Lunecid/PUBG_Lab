"""``adaptation.run_all`` forwards every path to every child (K1, K2, K3a, K3, HGB-kraftonfeat, union, H0); ``h0_baseline`` / ``h0_report`` take explicit paths and
never fall back to a v1 default.  No training: the commands are built, the dry run prints them, a fake subprocess stands in for the children."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import pytest

import pubg_lab.survival.ptr.data.paths as P
from tests.test_rerun_paths import make_dump

KT = Path(__file__).resolve().parents[1] / "third_party" / "krafton_pubg_winprob"
if str(KT) not in sys.path:
    sys.path.insert(0, str(KT))

import projects.krafton_baseline.training.h0_baseline as H0  # noqa: E402
import projects.krafton_baseline.reporting.h0_report as HR  # noqa: E402
import projects.krafton_baseline.pipelines.run_all as RA  # noqa: E402


@pytest.fixture()
def roots(tmp_path):
    base = P.RerunPaths.under(tmp_path)
    v2 = make_dump(base.dump, version="v2", cohorts=("paramo_tpp_squad", "karakin_tpp_squad", "erangel_tpp_squad"))
    return RA.Roots(base.runs, base.krafton_data, v2, base.kf), base


def flag(cmd: list[str], name: str) -> str:
    return cmd[cmd.index(name) + 1]


def test_every_child_gets_the_paths_it_needs_and_no_v1_path(roots):
    r, base = roots
    for stage_run, module, wants in (
        (("karakin_tpp_squad", "transformer", 42), 'projects.krafton_baseline.training.train', {"--out-dir": r.out, "--data-dir": r.data}),  # K1 reads shards, not the dump
        (("karakin_tpp_squad", "mlp", 42), 'projects.krafton_baseline.training.train', {"--out-dir": r.out, "--data-dir": r.data}),
        (("karakin_tpp_squad", "ourfeat", 42), 'projects.krafton_baseline.models.ourfeat', {"--out-dir": r.out, "--data-dir": r.data, "--dump-dir": r.dump}),  # K2
        (("karakin_tpp_squad", "bce", 42), 'projects.krafton_baseline.training.ourfeat_hazard', {"--out-dir": r.out, "--data-dir": r.data, "--dump-dir": r.dump}),  # K3a
        (("karakin_tpp_squad", "hazard", 42), 'projects.krafton_baseline.training.ourfeat_hazard', {"--out-dir": r.out, "--data-dir": r.data, "--dump-dir": r.dump}),  # K3
        (("karakin_tpp_squad", "hgbkf", 42), 'projects.krafton_baseline.training.hgb_kraftonfeat', {"--out-dir": r.out, "--data-dir": r.data, "--dump-dir": r.dump, "--table-dir": r.table}),
    ):
        job = RA.build_job(stage_run, r, device="mps", threads=3)
        assert job.cmd[1:3] == ["-m", module] and job.cwd == RA.ROOT, stage_run
        for name, want in wants.items():
            assert flag(job.cmd, name) == str(want) and Path(flag(job.cmd, name)).is_absolute(), (stage_run, name)  # absolute: the child runs in third_party/...
        if stage_run[1] in ("transformer", "mlp"):
            assert "--dump-dir" not in job.cmd and "--table-dir" not in job.cmd
        assert "_v1" not in " ".join(job.cmd)
        assert job.out == r.out / "karakin_tpp_squad" / RA.ARM[stage_run[1]] / "seed42" and job.log == r.out / "logs" / f"karakin_tpp_squad__{RA.ARM[stage_run[1]]}__seed42.log"
    assert RA.build_job(("karakin_tpp_squad", "hgbkf", 42), r, device="mps", threads=3).env == {"OMP_NUM_THREADS": "1"}  # the boosting runs next to a training process
    assert flag(RA.build_job(("karakin_tpp_squad", "bce", 42), r, device="mps", threads=8).cmd, "--threads") == "3"  # K3 keeps its cap


def test_union_and_h0_stages_forward_the_roots(roots):
    r, base = roots
    u = RA.union_command(r, arms=("HGB-union",), workers=6, threads=2)
    assert u[1:4] == ["-m", 'pubg_lab.survival.ptr.experiments.run_matrix', "--union"] and flag(u, "--dump-dir") == str(r.dump) and flag(u, "--out-dir") == str(r.out)
    assert flag(u, "--extra-features") == str(r.table) and flag(u, "--status-name") == "run_matrix_status_union.csv"
    assert flag(u, "--workers") == "6" and flag(u, "--threads") == "2" and u[u.index("--only") + 1] == "arm=HGB-union"
    assert "cohort=" in " ".join(RA.union_command(r, arms=("HGB-union", "M11R-union"), workers=1, threads=1, cohorts="a,b")) and "arm=HGB-union,M11R-union" in " ".join(
        RA.union_command(r, arms=("HGB-union", "M11R-union"), workers=1, threads=1))  # fmt: skip
    h = RA.h0_command(r, device="cpu", threads=5, cohorts="paramo_tpp_squad", redo=True)
    assert h[1:3] == ["-m", 'projects.krafton_baseline.training.h0_baseline'] and flag(h, "--ckpt-dir") == str(r.out) and flag(h, "--dump-dir") == str(r.dump) and flag(h, "--data-dir") == str(r.data)
    assert flag(h, "--threads") == "3" and flag(h, "--cohorts") == "paramo_tpp_squad" and "--redo" in h  # the module's own cap
    # the commands are accepted by the modules' own parsers (no option is invented)
    ns = H0.parse_args(h[3:])
    assert ns.ckpt_dir == r.out and ns.dump_dir == r.dump and ns.data_dir == r.data and ns.cohorts == "paramo_tpp_squad" and ns.redo


def test_stage_runs_and_the_cohort_order(roots):
    r, _ = roots
    cohorts = RA.resolve_cohorts("", r.dump)
    assert cohorts == ["paramo_tpp_squad", "karakin_tpp_squad", "erangel_tpp_squad"]  # the dump's cohorts in the KRAFTON order (small first, large last)
    assert RA.resolve_cohorts("a,b", r.dump) == ["a", "b"] and RA.resolve_cohorts("", r.dump / "nowhere") == RA.ORDER
    assert RA.stage_runs("main", cohorts, ["transformer", "mlp"]) == [(c, e, 42) for c in cohorts for e in ("transformer", "mlp")]
    assert RA.stage_runs("k3", cohorts, []) == [(c, v, 42) for c in cohorts for v in ("bce", "hazard")]
    assert RA.stage_runs("seeds", cohorts, ["transformer"]) == [("erangel_tpp_squad", "transformer", s) for s in (13, 2026)]
    assert RA.stage_runs("ourfeat", cohorts, []) == [(c, "ourfeat", 42) for c in cohorts] and RA.stage_runs("hgbkf", cohorts, []) == [(c, "hgbkf", 42) for c in cohorts]


def test_the_dry_run_prints_the_commands_and_writes_nothing(roots, capsys):
    r, base = roots
    argv = ["--out-dir", str(r.out), "--data-dir", str(r.data), "--dump-dir", str(r.dump), "--table-dir", str(r.table), "--dry-run"]
    assert RA.main(["--stage", "k3", *argv]) == 0
    text = capsys.readouterr().out
    assert text.count("run ") == 6 and 'projects.krafton_baseline.training.ourfeat_hazard' in text and str(r.dump) in text and "_v1" not in text
    assert RA.main(["--stage", "union", *argv]) == 0 and "run_matrix --union" in capsys.readouterr().out
    assert RA.main(["--stage", "h0", *argv]) == 0 and 'projects.krafton_baseline.training.h0_baseline' in capsys.readouterr().out
    assert not r.out.exists() and not r.data.exists()  # nothing was created: no stamp, no logs, no status
    done = r.out / "paramo_tpp_squad" / "KRAFTON-tf-ourfeat-bce" / "seed42"
    done.mkdir(parents=True)
    (done / "metrics.json").write_text("{}")
    assert RA.main(["--stage", "k3", "--only", "paramo_tpp_squad", *argv]) == 0
    assert "skip (done) paramo_tpp_squad/KRAFTON-tf-ourfeat-bce/seed42" in capsys.readouterr().out


def test_the_runner_skips_done_runs_records_status_and_exits_one_on_a_failure(roots, monkeypatch):
    r, base = roots
    calls = []

    class Done:
        def __init__(self, rc):
            self.returncode = rc

    def fake_run(cmd, cwd=None, stdout=None, stderr=None, env=None, **kw):
        calls.append((cmd, cwd, dict(env or {})))
        arm = cmd[cmd.index("--variant") + 1] if "--variant" in cmd else "KRAFTON-tf-ourfeat"
        cohort = cmd[cmd.index("--cohort") + 1]
        name = {"bce": "KRAFTON-tf-ourfeat-bce", "hazard": "KRAFTON-tf-ourfeat-hazard"}.get(arm, arm)
        if cohort == "karakin_tpp_squad" and arm == "hazard":
            return Done(3)  # a failing run
        d = Path(cmd[cmd.index("--out-dir") + 1]) / cohort / name / "seed42"
        d.mkdir(parents=True, exist_ok=True)
        (d / "metrics.json").write_text("{}")
        return Done(0)

    monkeypatch.setattr(RA.subprocess, "run", fake_run)
    done = r.out / "paramo_tpp_squad" / "KRAFTON-tf-ourfeat-bce" / "seed42"
    done.mkdir(parents=True)
    (done / "metrics.json").write_text("{}")
    argv = ["--stage", "k3", "--out-dir", str(r.out), "--data-dir", str(r.data), "--dump-dir", str(r.dump), "--table-dir", str(r.table), "--status-name", "krafton_run_status_mps1.csv"]
    assert RA.main(argv) == 1  # one run failed: the matrix continued and the exit code says so
    assert len(calls) == 5  # six runs, one already done
    assert all(Path(c[1]) == RA.ROOT for c in calls)
    rows = list(csv.DictReader(open(r.out / "krafton_run_status_mps1.csv")))
    assert {(x["run_id"], x["status"]) for x in rows} == {
        ("paramo_tpp_squad/KRAFTON-tf-ourfeat-hazard/seed42", "done"), ("karakin_tpp_squad/KRAFTON-tf-ourfeat-bce/seed42", "done"),
        ("karakin_tpp_squad/KRAFTON-tf-ourfeat-hazard/seed42", "failed"), ("erangel_tpp_squad/KRAFTON-tf-ourfeat-bce/seed42", "done"),
        ("erangel_tpp_squad/KRAFTON-tf-ourfeat-hazard/seed42", "done"),
    }
    assert (r.out / "logs").is_dir() and not (r.out / "krafton_run_status.csv").exists()  # one status file per stream
    assert RA.main(argv) == 1 and len(calls) == 6  # a second launch re-runs only the failed run
    # the run directory belongs to the v2 dump now
    assert P.artifact_version(r.out) == ("v2", "stamp")


def test_h0_modules_use_explicit_paths_and_have_no_v1_default():
    for module in (H0, HR, RA):
        text = Path(module.__file__).read_text()
        assert not re.search(r"bigcomp2027_ptr_v1|krafton_winprob_v1|bigcomp2027_phase_v1|bigcomp2027_kraftonfeat_v1|krafton_baseline_2026", text), module.__name__
    assert "OF.DEFAULT_" not in Path(H0.__file__).read_text()  # the K2 sample builder is given the roots, not ourfeat's own defaults
    assert H0.CKPT == P.RUNS_DIR and H0.DUMP == P.DUMP_DIR and H0.DATA == P.KRAFTON_DATA_DIR and HR.CKPT == P.RUNS_DIR
    assert HR.read_arm.__defaults__[-1] == P.RUNS_DIR and H0._stored.__defaults__[-1] == P.RUNS_DIR


def test_h0_readers_read_below_the_given_run_directory(tmp_path):
    import pandas as pd

    d = tmp_path / "ckpt" / "karakin_tpp_squad" / "KRAFTON-tf" / "seed42"
    d.mkdir(parents=True)
    pd.DataFrame({"match_id": ["m1", "m2"], "team_id": [1, 2], "phase": [1, 1], "raw_score": [0.1, 0.2]}).to_csv(d / "pred_test.csv", index=False)
    df = H0._stored("KRAFTON-tf", "karakin_tpp_squad", "test", tmp_path / "ckpt")
    assert list(df["match_id"]) == ["m1", "m2"] and df["match_id"].dtype == object
    with pytest.raises(FileNotFoundError):
        H0._stored("KRAFTON-tf", "karakin_tpp_squad", "test", tmp_path / "other")  # nothing is read from a default root
    with pytest.raises(FileNotFoundError):
        H0._load_ckpt("KRAFTON-tf", "karakin_tpp_squad", tmp_path / "ckpt")  # no best.pt there
    roots = H0.Roots(tmp_path / "ckpt", tmp_path / "d_v2", tmp_path / "k_v2")
    with pytest.raises(FileNotFoundError, match="d_v2"):  # the K2 samples are built from THE GIVEN dump (the error names it), not from a default one
        H0.run_cohort("karakin_tpp_squad", "cpu", False, roots)
    for arm in H0.ARMS:  # every H0 arm exists below the given run directory: nothing to do, nothing is loaded
        a = tmp_path / "ckpt" / "karakin_tpp_squad" / arm / "seed42"
        a.mkdir(parents=True, exist_ok=True)
        (a / "metrics.json").write_text("{}")
    assert H0.run_cohort("karakin_tpp_squad", "cpu", False, roots) == {}
