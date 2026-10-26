"""The learning-curve table and figure (design doc 4.4) on fabricated ``metrics.json`` files with known AUCs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pubg_lab.survival.ptr.experiments.learning_curve as LC

COHORTS = {"taego_tpp_squad": 3329, "erangel_tpp_squad": 2638}
SIZES = (300, 600, 1200)
SEEDS = (13, 42, 2026)
MODELS = LC.MODELS


def auc_of(cohort, arm, n, seed):
    """Known test AUC: grows with log N, the neural models faster than HGB, a small seed term."""
    slope = 0.0 if arm.startswith("HGB") else 0.01 + 0.002 * MODELS.index(arm)
    base = 0.70 + (0.002 if cohort.startswith("taego") else 0.0)
    return base + slope * np.log(n / 300) + 0.0004 * (seed % 7) + 0.0003 * len(arm)


def write_run(root, cohort, arm, n, seed, *, sub=True, hgb_main_style=False, epochs=12, auc=None, ss=None):
    name = f"sub{n}_ss{seed if ss is None else ss}_seed{seed}" if sub else f"seed{seed}"
    d = Path(root) / cohort / arm / name
    d.mkdir(parents=True, exist_ok=True)
    full = COHORTS[cohort]
    meta = {
        "labels": "v2", "epochs_run": epochs, "best_epoch": epochs - 1, "train_config": {"max_epochs": epochs},
        "test": {"hazard": {"auc": auc_of(cohort, arm, n, seed) if auc is None else auc, "ap": 0.5, "nll": 0.4, "brier": 0.1},
                 "rank": {"spearman": 0.2, "top5_auc": 0.6, "cindex": 0.55}},
    }  # fmt: skip
    info = {"n_requested": n, "n_used": min(n, full), "n_full": full, "seed": seed, "match_ids_sha1": "x"} if sub else None
    if arm.startswith("HGB"):
        if not hgb_main_style:
            meta.update(train_subsample=info, n_matches={"train": n if sub else full, "val": 10, "test": 10})
        meta.pop("train_config")
        meta.pop("epochs_run")
        meta.pop("best_epoch")
    else:
        meta.update(n_matches={"train": n if sub else full, "val": 10, "test": 10}, data={"train_subsample": info})
    (d / "metrics.json").write_text(json.dumps(meta))
    return d


@pytest.fixture(scope="module")
def dirs(tmp_path_factory):
    lc, main, steps = (tmp_path_factory.mktemp(n) for n in ("lc", "main", "steps"))
    for cohort in COHORTS:
        for arm in MODELS:
            for n in SIZES:
                for s in SEEDS:
                    write_run(lc, cohort, arm, n, s)
                    write_run(steps, cohort, arm, n, s, epochs=100, auc=auc_of(cohort, arm, n, s) + 0.01)
            # full size: the main matrix has seed 42 of every model and all seeds of M00 / M11R; the lc dir the rest
            for s in SEEDS:
                where = main if (s == 42 or arm in ("M00", "M11R")) else lc
                write_run(where, cohort, arm, COHORTS[cohort], s, sub=False, hgb_main_style=(arm == "HGB-cut" and where is main))
    # things that must not enter the curve: other models / cohorts of the main matrix, a stray directory
    write_run(main, "taego_tpp_squad", "HGB-lag", 3329, 42, sub=False, hgb_main_style=True)
    (main / "taego_tpp_squad" / "M00" / "notes").mkdir()
    return lc, main, steps


def test_tidy_table_has_every_run_once_with_the_training_size(dirs):
    lc, main, steps = dirs
    df = LC.curve_tables([("fixed_epochs", lc)], shared=[main])
    assert list(df.columns) == LC.TIDY_COLUMNS
    # per cohort: 5 models x (3 sizes + full) x 3 seeds
    assert len(df) == 2 * 5 * 4 * 3 and set(df["protocol"]) == {"fixed_epochs"}
    assert set(df["arm"]) == set(MODELS) and set(df["cohort"]) == set(COHORTS)  # HGB-lag, notes dirs are not part of it
    assert not df.duplicated(["cohort", "arm", "n_train", "seed"]).any()
    full = df[df["full_size"]]
    assert set(full["n_train"]) == {3329, 2638} and (full.groupby("cohort")["n_train"].nunique() == 1).all()
    assert full.groupby(["cohort", "arm"]).size().eq(3).all()  # 13 / 42 / 2026 for every model at full size
    # the HGB run of the main matrix has no match count: it takes the cohort's full train size
    hgb42 = full[(full["arm"] == "HGB-cut") & (full["seed"] == 42)]
    assert set(hgb42["n_train"]) == {3329, 2638}
    sub = df[~df["full_size"]]
    assert set(sub["n_train"]) == set(SIZES) and (sub["subsample_seed"] == sub["seed"]).all()
    assert (sub["n_full"] == sub["cohort"].map(COHORTS)).all()
    row = df[(df["cohort"] == "taego_tpp_squad") & (df["arm"] == "M11R") & (df["n_train"] == 600) & (df["seed"] == 42)].iloc[0]
    assert row["test_auc"] == pytest.approx(auc_of("taego_tpp_squad", "M11R", 600, 42)) and row["max_epochs"] == 12 and row["labels"] == "v2"


def test_a_full_size_run_found_in_two_directories_is_taken_from_the_first(dirs, tmp_path):
    lc, main, _ = dirs
    dup_main = tmp_path / "main2"
    write_run(dup_main, "taego_tpp_squad", "M10", 3329, 42, sub=False, auc=0.123)
    lc2 = tmp_path / "lc2"
    write_run(lc2, "taego_tpp_squad", "M10", 3329, 42, sub=False, auc=0.5)
    write_run(lc2, "taego_tpp_squad", "M10", 300, 13)
    df = LC.curve_tables([("fixed_epochs", lc2)], shared=[dup_main])
    got = df[(df["arm"] == "M10") & df["full_size"]]
    assert len(got) == 1 and got["test_auc"].iloc[0] == pytest.approx(0.5)  # the learning-curve dir wins


def test_summary_mean_and_sd_over_seeds(dirs):
    lc, main, _ = dirs
    df = LC.curve_tables([("fixed_epochs", lc)], shared=[main])
    s = LC.summarise(df)
    row = s[(s["cohort"] == "erangel_tpp_squad") & (s["arm"] == "M00") & (s["n_train"] == 1200)].iloc[0]
    vals = [auc_of("erangel_tpp_squad", "M00", 1200, k) for k in SEEDS]
    assert row["n_seeds"] == 3 and row["auc_mean"] == pytest.approx(np.mean(vals)) and row["auc_sd"] == pytest.approx(np.std(vals, ddof=1))
    assert row["auc_min"] == pytest.approx(min(vals)) and row["auc_max"] == pytest.approx(max(vals)) and not row["full_size"]
    assert set(s["n_train"]) == {300, 600, 1200, 3329, 2638}
    # one run only: no SD
    one = LC.summarise(df[(df["arm"] == "M00") & (df["n_train"] == 300) & (df["seed"] == 13)])
    assert np.isnan(one["auc_sd"]).all() and (one["n_seeds"] == 1).all()


def test_steps_matched_protocol_reuses_the_full_size_runs(dirs):
    lc, main, steps = dirs
    df = LC.curve_tables([("fixed_epochs", lc), ("steps_matched", steps)], shared=[main])
    fixed, matched = (df[df["protocol"] == p] for p in ("fixed_epochs", "steps_matched"))
    assert len(fixed) == len(matched) == 120
    neural = lambda d: d[~d["full_size"] & ~d["arm"].str.startswith("HGB")]  # noqa: E731  (trees have no epochs)
    assert set(neural(matched)["max_epochs"]) == {100} and set(neural(fixed)["max_epochs"]) == {12}
    a = fixed[fixed["full_size"]].sort_values(["cohort", "arm", "seed"])["test_auc"].to_numpy()
    b = matched[matched["full_size"]].sort_values(["cohort", "arm", "seed"])["test_auc"].to_numpy()
    assert np.array_equal(a, b)  # the same full-size runs in both curves
    # the subsampled steps-matched runs differ (+0.01 by construction)
    x = fixed[~fixed["full_size"]].sort_values(["cohort", "arm", "n_train", "seed"])["test_auc"].to_numpy()
    y = matched[~matched["full_size"]].sort_values(["cohort", "arm", "n_train", "seed"])["test_auc"].to_numpy()
    assert np.allclose(y - x, 0.01)


def test_run_writes_csvs_and_figures(dirs, tmp_path):
    lc, main, steps = dirs
    out = tmp_path / "res"
    res = LC.run(lc, out, main_dir=main, steps_dir=steps)
    assert (out / "learning_curve.csv").is_file() and (out / "learning_curve_summary.csv").is_file()
    tidy = pd.read_csv(out / "learning_curve.csv")
    assert set(tidy["protocol"]) == {"fixed_epochs", "steps_matched"} and len(tidy) == len(res["runs"]) == 240
    for name in ("learning_curve", "learning_curve_steps"):
        assert (out / "figures" / f"{name}.pdf").read_bytes()[:4] == b"%PDF"
        png = out / "figures" / f"{name}.png"
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    from matplotlib import image

    img = image.imread(out / "figures" / "learning_curve.png")
    assert img.shape[1] > 1500 and img.shape[1] > img.shape[0] and img[..., :3].std() > 0.02
    # without the steps directory and without figures
    out2 = tmp_path / "res2"
    LC.run(lc, out2, main_dir=main, figures=False)
    assert not (out2 / "figures").exists() and set(pd.read_csv(out2 / "learning_curve.csv")["protocol"]) == {"fixed_epochs"}


def test_unreadable_runs_and_missing_directories(dirs, tmp_path):
    lc, main, _ = dirs
    bad = tmp_path / "bad"
    write_run(bad, "taego_tpp_squad", "M00", 300, 13)
    (bad / "taego_tpp_squad" / "M00" / "sub300_ss42_seed42").mkdir()
    (bad / "taego_tpp_squad" / "M00" / "sub300_ss42_seed42" / "metrics.json").write_text("{not json")
    df = LC.collect([("fixed_epochs", bad)])
    assert len(df) == 1  # the unreadable run is skipped
    with pytest.raises(FileNotFoundError):
        LC.collect([("fixed_epochs", tmp_path / "nope")])
    (tmp_path / "empty_lc").mkdir()
    with pytest.raises(ValueError, match="no finished"):
        LC.run(tmp_path / "empty_lc", tmp_path / "o")


def test_cli_prints_the_mean_table(dirs, tmp_path, capsys):
    lc, main, _ = dirs
    rc = LC.main(["--lc-dir", str(lc), "--main-dir", str(main), "--out-dir", str(tmp_path / "o"), "--no-figures"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "taego_tpp_squad" in text and "M11R" in text and "HGB-cut" in text and "3329" in text
