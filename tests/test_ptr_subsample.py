"""Learning-curve plumbing (design doc 4.4): train subsampling in the data layer and both CLIs, the run directory
naming, and the learning-curve plan of the runner.  Synthetic dumps, CPU, DB-free."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import pubg_lab.survival.ptr.data.data as D
import pubg_lab.survival.ptr.models.hgb as H
import pubg_lab.survival.ptr.experiments.run_matrix as RM
import pubg_lab.survival.ptr.training.train as T
from pubg_lab.survival.ptr.data.synthetic import make_dump, write_cohort_dumps
from pubg_lab.survival.ptr.common.util import run_dir_name

SMALL = ["--d-model", "32", "--n-heads", "4", "--n-layers", "2"]
LARGE = ("erangel_tpp_squad", "taego_tpp_squad")


def ids(dump):
    return [str(m["match_id"]) for m in dump["matches"]]


# --------------------------------------------------------------------------- data layer
def test_run_dir_names():
    assert run_dir_name(42) == "seed42" and run_dir_name(13, None, 99) == "seed13"
    assert run_dir_name(13, 300, 13) == "sub300_ss13_seed13"
    assert run_dir_name(42, 600) == "sub600_ss42_seed42"  # the subsample seed defaults to the model seed
    assert run_dir_name(42, 600, 7) == "sub600_ss7_seed42"


def test_subsample_is_deterministic_nested_and_keeps_the_dump_order():
    dump = make_dump(40, seed=5)
    a, info = D.subsample_dump(dump, 10, 13)
    b, info_b = D.subsample_dump(dump, 10, 13)
    assert ids(a) == ids(b) and info == info_b and len(a["matches"]) == 10
    assert info["n_requested"] == 10 and info["n_used"] == 10 and info["n_full"] == 40 and info["seed"] == 13
    order = {i: k for k, i in enumerate(ids(dump))}
    assert [order[i] for i in ids(a)] == sorted(order[i] for i in ids(a))  # dump order preserved
    assert len(set(ids(a))) == 10 and set(ids(a)) <= set(ids(dump))
    # one seed: nested across N (the first N of one permutation); different seeds: different subsets
    small, _ = D.subsample_dump(dump, 5, 13)
    big, _ = D.subsample_dump(dump, 20, 13)
    assert set(ids(small)) <= set(ids(a)) <= set(ids(big))
    other, info_o = D.subsample_dump(dump, 10, 42)
    assert set(ids(other)) != set(ids(a)) and info_o["match_ids_sha1"] != info["match_ids_sha1"]
    # the input dump is not modified; the shared parts (meta) are kept
    assert len(dump["matches"]) == 40 and a["meta"] is dump["meta"]
    # no subsampling / N >= size: the dump itself (info still says what happened)
    assert D.subsample_dump(dump, None, 1) == (dump, None)
    same, info_full = D.subsample_dump(dump, 400, 1)
    assert same is dump and info_full["n_used"] == 40 and info_full["n_requested"] == 400
    with pytest.raises(ValueError):
        D.subsample_dump(dump, 0, 1)


def test_the_standardiser_is_fitted_on_the_subsample_only():
    dump = make_dump(30, seed=6)
    sub, _ = D.subsample_dump(dump, 8, 3)
    full_fit, sub_fit = D.Standardizer.fit(dump), D.Standardizer.fit(sub)
    assert not np.allclose(full_fit.x_mean, sub_fit.x_mean)
    manual = D.Standardizer.fit({**dump, "matches": [m for m in dump["matches"] if str(m["match_id"]) in set(ids(sub))]})
    np.testing.assert_allclose(sub_fit.x_mean, manual.x_mean)
    np.testing.assert_allclose(sub_fit.x_std, manual.x_std)


# --------------------------------------------------------------------------- the CLIs
def neural_args(root, out, arm="M00", seed=13, extra=()):
    return ["--dump-dir", str(root), "--cohort", "c", "--arm", arm, "--seed", str(seed), "--out-dir", str(out),
            "--device", "cpu", "--max-epochs", "1", *SMALL, *extra]  # fmt: skip


def hgb_args(root, out, variant="HGB-cut", seed=13, extra=()):
    return ["--dump-dir", str(root), "--cohort", "c", "--variant", variant, "--seed", str(seed), "--out-dir", str(out), *extra]


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    root = tmp_path_factory.mktemp("lcdumps")
    write_cohort_dumps(root, "c", sizes={"train": 16, "val": 5, "test": 5}, seed=3)
    return root


def test_both_clis_train_on_the_same_subsample_and_leave_val_and_test_alone(tmp_path, cohort):
    out = tmp_path / "o"
    assert T.main(neural_args(cohort, out, extra=["--train-subsample", "6"])) == 0
    assert H.main(hgb_args(cohort, out, extra=["--train-subsample", "6"])) == 0
    assert T.main(neural_args(cohort, out)) == 0  # a full-size run of the same arm and seed next to it
    assert H.main(hgb_args(cohort, out)) == 0
    d_neural, d_hgb = out / "c" / "M00" / "sub6_ss13_seed13", out / "c" / "HGB-cut" / "sub6_ss13_seed13"
    assert (out / "c" / "M00" / "seed13").is_dir() and (out / "c" / "HGB-cut" / "seed13").is_dir()
    mn, mh = (json.loads((d / "metrics.json").read_text()) for d in (d_neural, d_hgb))
    mf = json.loads((out / "c" / "M00" / "seed13" / "metrics.json").read_text())
    sub = mn["data"]["train_subsample"]
    assert sub["n_requested"] == 6 and sub["n_used"] == 6 and sub["n_full"] == 16 and sub["seed"] == 13
    assert mh["train_subsample"] == sub  # identical subset (same sha1 of the match ids)
    assert mn["n_matches"] == {"train": 6, "val": 5, "test": 5} and mh["n_matches"] == {"train": 6, "val": 5, "test": 5}
    assert mf["n_matches"] == {"train": 16, "val": 5, "test": 5} and mf["data"]["train_subsample"] is None
    # val / test rows are the full ones: the prediction files of the subsampled and the full run have identical rows
    import pandas as pd

    for split in ("val", "test"):
        a = pd.read_csv(d_neural / f"pred_{split}.csv")
        b = pd.read_csv(out / "c" / "M00" / "seed13" / f"pred_{split}.csv")
        assert a[["match_id", "team_id", "phase", "y"]].equals(b[["match_id", "team_id", "phase", "y"]])
    assert mn["config_hash"] != mf["config_hash"] and mh["config_hash"] != json.loads(
        (out / "c" / "HGB-cut" / "seed13" / "metrics.json").read_text()
    )["config_hash"]
    assert mn["n_matches"]["train"] < mf["n_matches"]["train"] and mh["n_train_rows"] < json.loads(
        (out / "c" / "HGB-cut" / "seed13" / "metrics.json").read_text()
    )["n_train_rows"]


def test_the_subsample_seed_defaults_to_the_model_seed_and_is_part_of_the_hash(tmp_path, cohort):
    out = tmp_path / "o"
    h = lambda extra, seed=13: T.config_hash(T.parse_args(neural_args(cohort, out, seed=seed, extra=extra)))  # noqa: E731
    base = h([])
    sub = h(["--train-subsample", "6"])
    assert sub != base and h(["--train-subsample", "6", "--subsample-seed", "13"]) == sub  # default = --seed
    assert h(["--train-subsample", "6", "--subsample-seed", "7"]) != sub
    assert h(["--train-subsample", "7"]) != sub
    assert h([], seed=42) != base
    # the hash of a full-size run does not depend on the feature (fields only present when subsampling)
    assert "train_subsample" not in T.config_spec(T.parse_args(neural_args(cohort, out)))
    hh = lambda extra: H.config_hash(H.parse_args(hgb_args(cohort, out, extra=extra)))  # noqa: E731
    assert hh(["--train-subsample", "6"]) != hh([]) and hh(["--train-subsample", "6", "--subsample-seed", "7"]) != hh(["--train-subsample", "6"])
    assert hh(["--train-subsample", "6", "--subsample-seed", "13"]) == hh(["--train-subsample", "6"])


def test_a_subsample_run_with_a_different_seed_trains_on_other_matches(tmp_path, cohort):
    out = tmp_path / "o"
    for ss in (13, 42):
        assert H.main(hgb_args(cohort, out, extra=["--train-subsample", "6", "--subsample-seed", str(ss)])) == 0
    d1, d2 = out / "c" / "HGB-cut" / "sub6_ss13_seed13", out / "c" / "HGB-cut" / "sub6_ss42_seed13"
    m1, m2 = (json.loads((d / "metrics.json").read_text()) for d in (d1, d2))
    assert m1["train_subsample"]["match_ids_sha1"] != m2["train_subsample"]["match_ids_sha1"]


# --------------------------------------------------------------------------- the runner's learning-curve plan
def test_learning_curve_plan_counts_and_names():
    runs = RM.enumerate_lc_runs(LARGE)
    assert len(runs) == 2 * (3 * 3 * 5 + 3 * 2) == 102
    lc = [r for r in runs if r.group == "lc"]
    assert len(lc) == 90 and {(r.subsample, r.seed, r.subsample_seed) for r in lc} == {(n, s, s) for n in (300, 600, 1200) for s in (13, 42, 2026)}
    assert {r.arm for r in lc} == {"HGB-cut", "M00", "M10", "M01", "M11R"}
    assert all(r.kind == ("hgb" if r.arm == "HGB-cut" else "neural") for r in runs)
    full = [r for r in runs if r.group == "lc_full"]
    assert {(r.cohort, r.arm, r.seed) for r in full} == {(c, a, s) for c in LARGE for a in ("HGB-cut", "M10", "M01") for s in (13, 2026)}
    assert all(r.subsample is None and r.epochs is None for r in full)
    assert len({r.run_id for r in runs}) == 102
    spec = next(r for r in lc if r.arm == "M11R" and r.subsample == 600 and r.seed == 2026 and r.cohort == "taego_tpp_squad")
    assert spec.run_id == "taego_tpp_squad/M11R/sub600_ss2026_seed2026"
    assert spec.run_dir("/x").as_posix() == "/x/taego_tpp_squad/M11R/sub600_ss2026_seed2026"
    assert spec.log_path("/x").name == "taego_tpp_squad__M11R__sub600_ss2026_seed2026.log"
    # the main matrix is untouched: names and logs of its runs are what they were
    main = RM.RunSpec("neural", "erangel_tpp_squad", "M00", 42, "main")
    assert main.run_id == "erangel_tpp_squad/M00/seed42" and main.log_path("/x").name == "erangel_tpp_squad__M00__seed42.log"


def test_steps_matched_epochs_keep_the_number_of_steps_near_the_full_run():
    sizes = {"erangel_tpp_squad": 2638, "taego_tpp_squad": 3329}
    runs = RM.enumerate_lc_runs(LARGE, steps_matched=True, full_sizes=sizes)
    assert len(runs) == 90 and all(r.group == "lc" for r in runs)  # the full-size runs are the fixed-epoch ones
    by = {(r.cohort, r.subsample): r.epochs for r in runs if r.kind == "neural"}
    assert by[("taego_tpp_squad", 300)] == math.ceil(12 * 3329 / 300) == 134
    assert by[("erangel_tpp_squad", 1200)] == math.ceil(12 * 2638 / 1200) == 27
    assert all(r.epochs is None for r in runs if r.kind == "hgb")  # trees have no epochs
    for (c, n), e in by.items():  # steps (matches seen) within one batch of the full run
        assert 12 * sizes[c] <= n * e < 12 * sizes[c] + n
    with pytest.raises(ValueError):
        RM.enumerate_lc_runs(LARGE, steps_matched=True)


def test_subsample_arguments_and_hash_reproduce_the_clis(tmp_path, cohort):
    st = RM.Settings(dump_dir=str(cohort), out_dir=str(tmp_path / "o"), max_epochs=2, threads=1)
    neural = RM.RunSpec("neural", "c", "M11R", 13, "lc", 300, 13)
    hgb = RM.RunSpec("hgb", "c", "HGB-cut", 2026, "lc", 600, 2026)
    steps = RM.RunSpec("neural", "c", "M00", 42, "lc", 300, 42, 134)
    a, b, c = RM.build_argv(neural, st), RM.build_argv(hgb, st), RM.build_argv(steps, st)
    assert a[a.index("--train-subsample") + 1] == "300" and a[a.index("--subsample-seed") + 1] == "13"
    assert b[b.index("--train-subsample") + 1] == "600" and b[b.index("--subsample-seed") + 1] == "2026"
    assert a[a.index("--max-epochs") + 1] == "2" and c[c.index("--max-epochs") + 1] == "134"  # the per-run epochs win
    assert RM.expected_hash(neural, st) == T.config_hash(T.parse_args(a[3:]))
    assert RM.expected_hash(hgb, st) == H.config_hash(H.parse_args(b[3:]))
    assert RM.expected_hash(steps, st) != RM.expected_hash(RM.RunSpec("neural", "c", "M00", 42, "lc", 300, 42), st)
    full = RM.RunSpec("neural", "c", "M11R", 13, "lc_full")
    assert "--train-subsample" not in RM.build_argv(full, st)


def test_estimates_scale_with_the_subsample_and_the_epochs():
    st = RM.Settings(dump_dir="x", out_dir="y", max_epochs=12, threads=2)
    stats = RM.CohortStats(n_train=3329, n_val=415, n_test=415, mean_teams=26.8, rows_per_match=59.0)
    est = lambda **kw: RM.estimate_seconds(RM.RunSpec("neural", "c", "M11R", 13, "lc", **kw), stats, st)  # noqa: E731
    assert est(subsample=300, subsample_seed=13) < est(subsample=1200, subsample_seed=13) < RM.estimate_seconds(
        RM.RunSpec("neural", "c", "M11R", 13, "main"), stats, st
    )
    # steps-matched: about as expensive as the full run
    full = RM.estimate_seconds(RM.RunSpec("neural", "c", "M11R", 13, "main"), stats, st)
    matched = est(subsample=300, subsample_seed=13, epochs=134)
    assert 0.8 * full < matched < 1.2 * full


def test_lc_dry_run_and_real_subprocess_run(tmp_path, capsys):
    root = tmp_path / "dumps"
    for c in LARGE:
        write_cohort_dumps(root, c, sizes={"train": 12, "val": 4, "test": 4}, seed=2)
    out = tmp_path / "lc"
    argv = ["--dump-dir", str(root), "--out-dir", str(out), "--lc", "--lc-sizes", "5", "--lc-seeds", "13,42",
            "--lc-models", "HGB-cut,M00", "--max-epochs", "1", "--workers", "2", "--threads", "1", "--poll-seconds", "0.1"]  # fmt: skip
    assert RM.main([*argv, "--dry-run"]) == 0
    text = capsys.readouterr().out
    assert "learning curve 8 + full-size extra seeds 12" in text.replace("  ", " ") or "runs: 20 = learning curve 8" in text
    assert not out.exists()
    assert RM.main([*argv, "--only", "cohort=erangel_tpp_squad", "--only", "arm=HGB-cut", "--only", "group=lc"]) == 0
    assert (out / "erangel_tpp_squad" / "HGB-cut" / "sub5_ss13_seed13" / "metrics.json").is_file()
    assert (out / "erangel_tpp_squad" / "HGB-cut" / "sub5_ss42_seed42" / "pred_test.csv").is_file()
    status = (out / "run_matrix_status.csv").read_text()
    assert "sub5_ss13_seed13" in status and ",5," in status  # the subsample column
    with pytest.raises(SystemExit):
        RM.main(["--dump-dir", str(root), "--out-dir", str(out), "--lc", "--lc-cohorts", "nope_tpp_squad", "--dry-run"])


def test_the_cli_fits_the_standardiser_on_the_subsample(tmp_path, cohort):
    import torch

    out = tmp_path / "o"
    assert T.main(neural_args(cohort, out, extra=["--train-subsample", "6"])) == 0
    ckpt = torch.load(out / "c" / "M00" / "sub6_ss13_seed13" / "best.pt", weights_only=False)
    train = D.load_cohort(cohort, "c")["train"]
    sub, _ = D.subsample_dump(train, 6, 13)
    np.testing.assert_allclose(ckpt["standardizer"]["x_mean"], D.Standardizer.fit(sub).x_mean)
    assert not np.allclose(ckpt["standardizer"]["x_mean"], D.Standardizer.fit(train).x_mean)  # not the full train split
