"""Arms registry and the end-to-end training CLI on synthetic dumps (DB-free, CPU)."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

import pubg_lab.survival.ptr.data.data as D
import pubg_lab.survival.ptr.models.hgb as H
import pubg_lab.survival.ptr.evaluation.metrics as M
import pubg_lab.survival.ptr.models.relations as R
import pubg_lab.survival.ptr.training.train as T
from pubg_lab.survival.ptr.models.arms import ARMS, EXPLORATORY_ARMS, MAIN_ARMS, NORANK_ARMS, STAGE2_ARMS, STUB_ARMS, UNION_ARMS, get_arm
from pubg_lab.survival.ptr.data.synthetic import write_cohort_dumps

# design doc section 4 (phase axis, team axis, relation bias, zone object, shuffle, dropped groups)
SPEC = {
    "M00": (False, False, False, False, False, ()),
    "M10": (True, False, False, False, False, ()),
    "M01": (False, True, False, False, False, ()),
    "M11": (True, True, False, False, False, ()),
    "M11R": (True, True, True, False, False, ()),
    "M11R-shuf": (True, True, True, False, True, ()),
    "M11RZ": (True, True, True, True, False, ()),
    "M11R-noHab": (True, True, True, False, False, ("zone", "habitat")),
    "M11RZ-noHab": (True, True, True, True, False, ("zone", "habitat")),
}
SMALL = ["--d-model", "32", "--n-heads", "4", "--n-layers", "2"]


def cli(root, out, arm, *extra, cohort="c", seed=42, epochs=1):
    return [
        "--dump-dir", str(root), "--cohort", cohort, "--arm", arm, "--seed", str(seed),
        "--out-dir", str(out), "--device", "cpu", "--max-epochs", str(epochs), *extra,
    ]  # fmt: skip


def run_dir(out, arm, cohort="c", seed=42):
    return out / cohort / arm / f"seed{seed}"


LABEL_KEYS = {"v2": ("y_hazard_v2", "target_mask_v2"), "pipeline": ("y_hazard", "target_mask")}


def target_keys(dump, labels="v2"):
    """Sorted (match_id, team_id, cut phase k = slot p + 1) of the selected risk set."""
    t_key = LABEL_KEYS[labels][1]
    return sorted(
        (m["match_id"], int(m["team_ids"][i]), int(p) + 1)
        for m in dump["matches"]
        for p, i in m[t_key].nonzero().tolist()
    )


def strip_field(root, cohort, field):
    """Remove one match-level field from every split file of a cohort (simulates an old dump)."""
    for split in ("train", "val", "test"):
        path = root / cohort / f"{split}.pt"
        dump = torch.load(path, weights_only=False)
        for m in dump["matches"]:
            m.pop(field, None)
        torch.save(dump, path)


@pytest.fixture(scope="module")
def cohort_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("dumps")
    write_cohort_dumps(root, "c", sizes={"train": 12, "val": 5, "test": 5}, seed=1)
    return root


# --------------------------------------------------------------------------- registry
def test_arm_registry_matches_design_table():
    assert MAIN_ARMS == tuple(SPEC)  # the nine runnable arms of section 4, in table order
    for name, (phase, team, rel, zone, shuf, drop) in SPEC.items():
        a = ARMS[name]
        assert (a.phase_axis, a.team_axis, a.relation, a.zone, a.shuffle, a.drop_groups) == (
            phase, team, rel, zone, shuf, drop,
        ), name  # fmt: skip
        assert a.name == name and a.implemented and not a.exploratory and a.stage == 1
        assert a.relation_types == ("prox", "combat_out", "combat_in", "copress")
    assert set(ARMS) == set(MAIN_ARMS) | set(EXPLORATORY_ARMS) | set(STAGE2_ARMS) | set(UNION_ARMS) | set(NORANK_ARMS)
    assert all(ARMS[a].rank_weight is None for a in ARMS if a not in NORANK_ARMS)  # only the E1 controls fix their loss
    with pytest.raises(KeyError):
        get_arm("M99")


def test_stage_two_arms_are_registered_with_the_player_layer():
    assert STAGE2_ARMS == ("M11RZ-P", "M11RZ-P-tm") and STUB_ARMS == ()  # nothing is a stub any more
    full, tm = get_arm("M11RZ-P", implemented_only=True), get_arm("M11RZ-P-tm", implemented_only=True)
    for a in (full, tm):
        assert a.implemented and a.stage == 2 and a.player and not a.exploratory and not a.shuffle and a.drop_groups == ()
        assert (a.phase_axis, a.team_axis, a.relation, a.zone) == (True, True, True, True)  # everything of M11RZ
        assert a.relation_types == get_arm("M11RZ").relation_types
    assert full.player_relation_types == ("teammate", "enemy_prox", "combat_out", "combat_in")
    assert tm.player_relation_types == ("enemy_prox", "combat_out", "combat_in")  # -tm: the teammate bias only is gone
    assert not any(get_arm(n).player for n in (*MAIN_ARMS, *EXPLORATORY_ARMS))
    with pytest.raises(ValueError, match="player"):
        from pubg_lab.survival.ptr.models.arms import model_config

        model_config(full, 10)  # a player arm needs the player feature count


def test_norank_arms_are_m01_and_m11r_with_the_rank_loss_switched_off():
    assert NORANK_ARMS == ("M01-norank", "M11R-norank")
    for name, parent in (("M01-norank", "M01"), ("M11R-norank", "M11R")):
        a, base = get_arm(name, implemented_only=True), ARMS[parent]
        assert a.rank_weight == 0.0 and a.exploratory and a.stage == 1 and not a.player and not a.extra
        assert (a.phase_axis, a.team_axis, a.relation, a.zone, a.shuffle, a.drop_groups, a.relation_types) == (
            base.phase_axis, base.team_axis, base.relation, base.zone, base.shuffle, base.drop_groups, base.relation_types,
        )  # fmt: skip
        assert name not in MAIN_ARMS


def test_exploratory_relation_type_arms_follow_section_4_1():
    assert EXPLORATORY_ARMS == ("M11R-noProx", "M11R-noCombat", "M11R-noCopress")
    m11r = ARMS["M11R"]
    expect = {
        "M11R-noProx": ("combat_out", "combat_in", "copress"),
        "M11R-noCombat": ("prox", "copress"),
        "M11R-noCopress": ("prox", "combat_out", "combat_in"),
    }
    for name, types in expect.items():
        a = ARMS[name]
        assert a.exploratory and a.implemented and a.relation_types == types
        assert (a.phase_axis, a.team_axis, a.relation, a.zone, a.shuffle, a.drop_groups) == (
            m11r.phase_axis, m11r.team_axis, m11r.relation, m11r.zone, m11r.shuffle, m11r.drop_groups,
        )  # fmt: skip


# --------------------------------------------------------------------------- early stopping
def test_early_stopper_patience_three_keeps_best_epoch():
    es = T.EarlyStopper(patience=3)
    flags = [es.update(v) for v in (1.0, 0.9, 0.95, 0.96)]
    assert flags == [True, True, False, False] and not es.should_stop
    es.update(0.97)
    assert es.should_stop and es.best_epoch == 2 and es.best == 0.9
    es = T.EarlyStopper(patience=3)
    for v in (1.0, float("nan"), float("nan"), 0.5):
        es.update(v)
    assert es.best == 0.5 and es.best_epoch == 4 and not es.should_stop  # NaN never improves


# --------------------------------------------------------------------------- smoke / end to end
def test_smoke_one_epoch_end_to_end_with_spec_defaults(tmp_path, cohort_dir):
    write_cohort_dumps(cohort_dir, "c150", sizes={"train": 10, "val": 4, "test": 4}, seed=2, n_features=150)
    out = tmp_path / "out"
    args = cli(cohort_dir, out, "M11RZ", cohort="c150")
    assert T.main(args) == 0
    d = run_dir(out, "M11RZ", cohort="c150")
    for f in ("pred_val.csv", "pred_test.csv", "metrics.json", "best.pt"):
        assert (d / f).exists(), f

    splits = {s: torch.load(cohort_dir / "c150" / f"{s}.pt", weights_only=False) for s in ("val", "test")}
    for split in ("val", "test"):
        df = pd.read_csv(d / f"pred_{split}.csv")
        assert list(df.columns) == M.PRED_COLUMNS
        assert list(df.columns[:7]) == [
            "match_id", "team_id", "phase", "y", "p_hazard", "rank_score", "final_rank_norm",
        ]  # fmt: skip
        n_target = sum(int(m["target_mask_v2"].sum()) for m in splits[split]["matches"])  # default labels: v2
        assert len(df) == n_target > 0
        assert np.isfinite(df[["p_hazard", "rank_score"]].to_numpy()).all()
        assert df["p_hazard"].between(0, 1).all()
        keys = list(zip(df["match_id"], df["team_id"], df["phase"]))
        assert keys == sorted(keys)
        # phase = cut phase k = dump slot p + 1 (1-based), rows = the dump's target rows exactly
        assert keys == target_keys(splits[split]) and df["phase"].between(1, 7).all()
        assert df.columns[-1] == "match_type" and set(df["match_type"]) <= {"official", "competitive"}

    met = json.loads((d / "metrics.json").read_text())
    assert met["arm"] == "M11RZ" and met["cohort"] == "c150" and met["seed"] == 42
    assert met["epochs_run"] == 1 and met["best_epoch"] == 1
    assert met["elapsed_sec"] > 0 and met["train_sec"] > 0
    assert len(met["history"]) == 1 and met["history"][0]["sec"] > 0
    assert met["labels"] == "v2" and met["label_fields"] == {"y": "y_hazard_v2", "target": "target_mask_v2"}
    assert met["n_input_features"] == 150 + 2 + 1  # 150 phase-tier columns + team centre (2) + AI share (1)
    cfg, tr = met["config"], met["train_config"]
    assert (cfg["d_model"], cfg["n_heads"], cfg["n_layers"], cfg["dropout"]) == (128, 4, 2, 0.1)
    assert (tr["lr"], tr["weight_decay"], tr["grad_clip"], tr["batch_size"]) == (1e-3, 1e-4, 5.0, 8)
    assert (tr["patience"], tr["rank_weight"], tr["max_epochs"]) == (3, 0.1, 1)
    assert cfg["zone"] and cfg["relation"] and cfg["phase_axis"] and cfg["team_axis"]
    for split in ("val", "test"):
        assert np.isfinite(met[split]["hazard"]["nll"]) and np.isfinite(met[split]["hazard"]["brier"])
        assert set(met[split]) >= {"hazard", "rank", "rank_from_hazard", "n_rows", "n_matches"}


@pytest.mark.parametrize("arm", list(SPEC))
def test_every_arm_trains_and_predicts_one_epoch(tmp_path, cohort_dir, arm):
    out = tmp_path / "out"
    assert T.main(cli(cohort_dir, out, arm, *SMALL)) == 0
    d = run_dir(out, arm)
    df = pd.read_csv(d / "pred_test.csv")
    assert np.isfinite(df[["p_hazard", "rank_score"]].to_numpy()).all()
    met = json.loads((d / "metrics.json").read_text())
    assert met["arm_spec"]["name"] == arm
    spec = SPEC[arm]
    assert (met["config"]["phase_axis"], met["config"]["team_axis"]) == spec[:2]
    assert (met["config"]["relation"], met["config"]["zone"]) == spec[2:4]
    assert met["arm_spec"]["shuffle"] == spec[4] and tuple(met["arm_spec"]["drop_groups"]) == spec[5]
    if spec[5]:  # noHab: zone + habitat columns are really gone from the model input
        assert met["n_input_features"] < 24 + 3
    else:
        assert met["n_input_features"] == 24 + 3  # features + team centre (2) + team_ai_frac (1)


def test_cli_rejects_unknown_arms_and_cohorts(tmp_path, cohort_dir):
    with pytest.raises(SystemExit):
        T.main(cli(cohort_dir, tmp_path, "M99"))
    with pytest.raises(FileNotFoundError):
        T.main(cli(cohort_dir, tmp_path, "M00", cohort="nope"))
    for arm in STAGE2_ARMS:  # the player dump of the cohort is missing: a clear error before anything is trained
        with pytest.raises(FileNotFoundError, match="player"):
            T.main(cli(cohort_dir, tmp_path, arm, *SMALL))


@pytest.mark.parametrize("arm", EXPLORATORY_ARMS)
def test_exploratory_relation_type_arms_train(tmp_path, cohort_dir, arm):
    out = tmp_path / "out"
    assert T.main(cli(cohort_dir, out, arm, *SMALL)) == 0
    met = json.loads((run_dir(out, arm) / "metrics.json").read_text())
    assert met["config"]["relation_types"] == list(ARMS[arm].relation_types)
    assert met["arm_spec"]["exploratory"] is True


def test_training_is_deterministic_on_cpu_and_seed_dependent(tmp_path, cohort_dir):
    out = tmp_path / "out"
    for tag, seed in (("a", 42), ("b", 42), ("c", 43)):
        assert T.main(cli(cohort_dir, out / tag, "M11RZ", *SMALL, seed=seed, epochs=2)) == 0
    read = lambda tag, seed: pd.read_csv(run_dir(out / tag, "M11RZ", seed=seed) / "pred_test.csv")  # noqa: E731
    pd.testing.assert_frame_equal(read("a", 42), read("b", 42), check_exact=True)
    assert not np.allclose(read("a", 42)["p_hazard"], read("c", 43)["p_hazard"])


def test_shuffle_arm_changes_relations_but_not_rows(tmp_path, cohort_dir):
    out = tmp_path / "out"
    for arm in ("M11R", "M11R-shuf"):
        assert T.main(cli(cohort_dir, out, arm, *SMALL, epochs=2)) == 0
    a = pd.read_csv(run_dir(out, "M11R") / "pred_test.csv")
    b = pd.read_csv(run_dir(out, "M11R-shuf") / "pred_test.csv")
    assert a[["match_id", "team_id", "phase", "y"]].equals(b[["match_id", "team_id", "phase", "y"]])
    assert not np.allclose(a["p_hazard"], b["p_hazard"])
    res = M.paired_bootstrap(
        run_dir(out, "M11R") / "pred_test.csv", run_dir(out, "M11R-shuf") / "pred_test.csv", n_boot=20
    )
    assert res["n_rows"] == len(a)


def test_best_checkpoint_is_restored_and_reproduces_predictions(tmp_path, cohort_dir):
    out = tmp_path / "out"
    argv = cli(cohort_dir, out, "M11RZ", *SMALL, epochs=6)
    assert T.main(argv) == 0
    d = run_dir(out, "M11RZ")
    met = json.loads((d / "metrics.json").read_text())
    val = [h["val_nll"] for h in met["history"]]
    assert met["best_epoch"] == int(np.argmin(val)) + 1
    assert met["best_val_nll"] == pytest.approx(min(val))
    assert met["val"]["hazard"]["nll"] == pytest.approx(met["best_val_nll"], abs=1e-4)

    prep = T.prepare(T.parse_args(argv))
    model, ckpt = T.load_checkpoint(d / "best.pt")
    assert ckpt["epoch"] == met["best_epoch"]
    df = T.predict(model, prep.datasets["test"], torch.device("cpu"), batch_size=8)
    saved = pd.read_csv(d / "pred_test.csv")
    np.testing.assert_allclose(df["p_hazard"].to_numpy(), saved["p_hazard"].to_numpy(), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(df["rank_score"].to_numpy(), saved["rank_score"].to_numpy(), rtol=1e-5, atol=1e-7)


def test_early_stopping_in_the_loop_stops_after_patience(tmp_path, cohort_dir):
    out = tmp_path / "out"
    # lr 0 -> val NLL cannot improve after epoch 1; with patience 2 the run must stop at epoch 3
    argv = cli(cohort_dir, out, "M00", *SMALL, "--lr", "0", "--patience", "2", epochs=12)
    assert T.main(argv) == 0
    met = json.loads((run_dir(out, "M00") / "metrics.json").read_text())
    assert met["epochs_run"] == 3 and met["best_epoch"] == 1 and met["stopped_early"] is True


def test_neural_and_hgb_predictions_share_rows(tmp_path, cohort_dir):
    out = tmp_path / "out"
    assert T.main(cli(cohort_dir, out, "M11RZ", *SMALL)) == 0
    H.run(cohort_dir, "c", "HGB-cut", seed=42, out_dir=out)
    a = run_dir(out, "M11RZ") / "pred_test.csv"
    b = out / "c" / "HGB-cut" / "seed42" / "pred_test.csv"
    res = M.paired_bootstrap(a, b, n_boot=30, seed=0)  # raises unless rows are identical
    assert res["n_rows"] > 0
    assert set(res["metrics"]) >= {"auc", "cindex"}


# --------------------------------------------------------------------------- learning sanity
def _auc(out, arm):
    met = json.loads((run_dir(out, arm, cohort="sig") / "metrics.json").read_text())
    return met["test"]["hazard"]["auc"]


def test_model_learns_a_planted_current_state_signal(tmp_path):
    root = tmp_path / "dumps"
    write_cohort_dumps(root, "sig", sizes={"train": 48, "val": 12, "test": 24}, seed=3, signal=3.0)
    out = tmp_path / "out"
    assert T.main(cli(root, out, "M00", *SMALL, cohort="sig", epochs=12)) == 0
    assert _auc(out, "M00") > 0.85


def test_phase_axis_reads_a_signal_that_only_exists_one_phase_earlier(tmp_path):
    root = tmp_path / "dumps"
    write_cohort_dumps(
        root, "sig", sizes={"train": 64, "val": 16, "test": 32}, seed=4, signal=0.0, lag_signal=3.0
    )
    out = tmp_path / "out"
    for arm in ("M00", "M10"):
        assert T.main(cli(root, out, arm, *SMALL, cohort="sig", epochs=15)) == 0
    assert _auc(out, "M10") > _auc(out, "M00") + 0.10
    assert _auc(out, "M10") > 0.70
    assert math.isfinite(_auc(out, "M00"))


# =========================================================================== v2.2 details
def test_nohab_arms_keep_alive_counts_and_other_arms_drop_nothing():
    from pubg_lab.survival.ptr.data.data import ALIVE_COUNT_FEATURES

    for name in MAIN_ARMS:
        expect = ALIVE_COUNT_FEATURES if name.endswith("noHab") else ()
        assert ARMS[name].keep_names == expect, name
        assert (ARMS[name].drop_groups != ()) == name.endswith("noHab")


@pytest.mark.parametrize("arm", ["M11R-noHab", "M11RZ-noHab"])
def test_nohab_run_lists_removed_columns_in_metrics_json(tmp_path, arm):
    root = tmp_path / "dumps"
    write_cohort_dumps(root, "ca", sizes={"train": 10, "val": 4, "test": 4}, seed=7, alive_count_names=True)
    out = tmp_path / "out"
    assert T.main(cli(root, out, arm, *SMALL, cohort="ca")) == 0
    met = json.loads((run_dir(out, arm, cohort="ca") / "metrics.json").read_text())
    rem = met["feature_removal"]
    assert rem["drop_groups"] == ["zone", "habitat"] and rem["n_removed"] == 3
    assert rem["removed_columns"] == ["safe_radius_start_norm", "hab_safe_area_per_alive_team", "hab_loss_ratio"]
    assert set(rem["kept_by_exception"]) == {"avg_alive_teams_norm", "avg_alive_players_norm", "hab_alive_teams"}
    assert met["n_input_features"] == 24 - 3 + 2 + 1
    full = tmp_path / "out_full"
    assert T.main(cli(root, full, "M11R", *SMALL, cohort="ca")) == 0
    assert json.loads((run_dir(full, "M11R", cohort="ca") / "metrics.json").read_text())["feature_removal"]["n_removed"] == 0


def test_rank_reduction_option_is_recorded_and_runs(tmp_path, cohort_dir):
    out = tmp_path / "out"
    for red in ("pair", "group"):
        assert T.main(cli(cohort_dir, out / red, "M00", *SMALL, "--rank-reduction", red)) == 0
        met = json.loads((run_dir(out / red, "M00") / "metrics.json").read_text())
        assert met["train_config"]["rank_reduction"] == red
    with pytest.raises(SystemExit):
        T.main(cli(cohort_dir, out, "M00", "--rank-reduction", "mean"))


@pytest.mark.parametrize("stored,expected", [("log", "log"), ("raw", "raw")])
def test_combat_weight_mode_is_resolved_from_the_dump_and_recorded(tmp_path, stored, expected):
    root = tmp_path / "dumps"
    write_cohort_dumps(root, "cw", sizes={"train": 8, "val": 3, "test": 3}, seed=8, combat_weights=stored)
    out = tmp_path / "out"
    assert T.main(cli(root, out, "M11R", *SMALL, cohort="cw")) == 0
    rel = json.loads((run_dir(out, "M11R", cohort="cw") / "metrics.json").read_text())["relations"]
    assert rel["combat_weights"] == expected and rel["combat_weight_max_train"] > 0
    assert rel["relation_types"] == ["prox", "combat_out", "combat_in", "copress"]
    # the datasets the run trains on really use the resolved mode (not just the recorded one)
    k = ["prox", "combat_out", "combat_in", "copress"].index("combat_out")
    prep = T.prepare(T.parse_args(cli(root, out, "M11R", *SMALL, cohort="cw")))
    got = {s: ds.get_batch([0, 1])["rel"] for s, ds in prep.datasets.items()}
    for split, ds in prep.datasets.items():
        assert ds.combat_weights == expected
    other = "raw" if expected == "log" else "log"
    for split in ("train", "val", "test"):
        dump = {"meta": prep.datasets[split].meta, "matches": prep.datasets[split].matches}
        ref = D.PTRDataset(dump, prep.standardizer, relation_types=R.RELATION_TYPES, combat_weights=expected)
        torch.testing.assert_close(got[split], ref.get_batch([0, 1])["rel"])
        wrong = D.PTRDataset(dump, prep.standardizer, relation_types=R.RELATION_TYPES, combat_weights=other)
        wrong = wrong.get_batch([0, 1])["rel"]
        assert not torch.allclose(got[split][:, :, k], wrong[:, :, k])
    forced = "raw" if expected == "log" else "log"  # an explicit flag overrides the inference
    assert T.main(cli(root, out / "f", "M11R", *SMALL, "--combat-weights", forced, cohort="cw")) == 0
    rel = json.loads((run_dir(out / "f", "M11R", cohort="cw") / "metrics.json").read_text())["relations"]
    assert rel["combat_weights"] == forced
    # non-relation arms do not look at the edges at all
    assert T.main(cli(root, out / "n", "M11", *SMALL, cohort="cw")) == 0
    rel = json.loads((run_dir(out / "n", "M11", cohort="cw") / "metrics.json").read_text())["relations"]
    assert rel["relation_types"] == [] and rel["combat_weight_max_train"] is None


def test_rank_weight_default_and_the_arm_override(tmp_path, cohort_dir):
    args = T.parse_args(cli(cohort_dir, tmp_path, "M01", *SMALL))
    assert args.rank_weight == 0.1  # the shared recipe (None on the command line resolves to 0.1)
    assert T.parse_args(cli(cohort_dir, tmp_path, "M01", *SMALL, "--rank-weight", "0.3")).rank_weight == 0.3
    nr = T.parse_args(cli(cohort_dir, tmp_path, "M01-norank", *SMALL))
    assert nr.rank_weight == 0.0 and isinstance(nr.rank_weight, float)
    assert T.parse_args(cli(cohort_dir, tmp_path, "M01-norank", *SMALL, "--rank-weight", "0")).rank_weight == 0.0  # the same value is fine
    with pytest.raises(SystemExit):  # a conflicting value is an error, never silently replaced
        T.parse_args(cli(cohort_dir, tmp_path, "M11R-norank", *SMALL, "--rank-weight", "0.1"))
    # the hash separates the loss: M01 (0.1) and M01-norank differ in the arm and in the weight; an explicit 0.1 on M01 is the default
    assert T.config_hash(args) == T.config_hash(T.parse_args(cli(cohort_dir, tmp_path, "M01", *SMALL, "--rank-weight", "0.1")))
    assert T.config_spec(nr)["rank_weight"] == 0.0 and T.config_spec(args)["rank_weight"] == 0.1


def test_norank_arm_trains_on_bce_alone_and_writes_the_hazard_ranking(tmp_path, cohort_dir):
    out = tmp_path / "out"
    assert T.main(cli(cohort_dir, out, "M01-norank", *SMALL, epochs=2)) == 0
    assert T.main(cli(cohort_dir, out, "M01", *SMALL, epochs=2)) == 0
    d, d0 = run_dir(out, "M01-norank"), run_dir(out, "M01")
    met, met0 = json.loads((d / "metrics.json").read_text()), json.loads((d0 / "metrics.json").read_text())
    assert met["train_config"]["rank_weight"] == 0.0 and met0["train_config"]["rank_weight"] == 0.1
    assert met["arm_spec"]["name"] == "M01-norank" and met["arm_spec"]["rank_weight"] == 0.0
    assert "untrained" in met["rank_score_source"] and met0["rank_score_source"] == "rank_head"
    assert met["n_params"] == met0["n_params"] and met["config"] == met0["config"]  # the same architecture
    df, df0 = pd.read_csv(d / "pred_test.csv"), pd.read_csv(d0 / "pred_test.csv")
    assert list(df.columns) == M.PRED_COLUMNS and len(df) == len(df0)
    p = df["p_hazard"].to_numpy()
    assert (df[["match_id", "team_id", "phase", "y"]].values == df0[["match_id", "team_id", "phase", "y"]].values).all()  # identical rows / labels
    # rank_score is the (negated) hazard logit, i.e. a monotone function of p_hazard: the rank metrics read the hazard head
    np.testing.assert_allclose(df["rank_score"], -np.log(p / (1 - p)), atol=2e-5)
    assert met["test"]["rank"]["spearman"] == pytest.approx(met["test"]["rank_from_hazard"]["spearman"], abs=1e-6)
    assert not np.allclose(df["p_hazard"], df0["p_hazard"])  # a different objective: different predictions


def test_rank_head_learns_the_survival_direction_end_to_end(tmp_path):
    """Loss direction + metric sign + CSV: better placement must get the HIGHER rank_score."""
    root = tmp_path / "dumps"
    write_cohort_dumps(root, "rk", sizes={"train": 48, "val": 12, "test": 24}, seed=9, signal=0.0, rank_signal=10.0)
    out = tmp_path / "out"
    assert T.main(cli(root, out, "M00", *SMALL, "--rank-weight", "1.0", cohort="rk", epochs=15)) == 0
    d = run_dir(out, "M00", cohort="rk")
    met = json.loads((d / "metrics.json").read_text())
    assert met["test"]["rank"]["cindex"] > 0.75 and met["test"]["rank"]["spearman"] > 0.5
    csv = pd.read_csv(d / "pred_test.csv")
    assert np.corrcoef(csv["rank_score"], csv["final_rank_norm"])[0, 1] < -0.3  # high score <-> small (good) rank
    # and the metric would have flagged the opposite direction
    flipped = csv.assign(rank_score=-csv["rank_score"])
    assert M.rank_metrics(flipped, "rank_score")["cindex"] < 0.25


# =========================================================================== v2.3: labels, AI share, match type
def test_labels_flag_defaults_to_v2_and_selects_the_risk_set_end_to_end(tmp_path):
    """One label choice drives the risk set, the loss targets, the CSV rows and every metric."""
    root = tmp_path / "dumps"
    write_cohort_dumps(root, "lb", sizes={"train": 14, "val": 6, "test": 10}, seed=11)
    test_dump = torch.load(root / "lb" / "test.pt", weights_only=False)
    out = {labels: tmp_path / f"out_{labels}" for labels in ("v2", "pipeline")}
    assert T.main(cli(root, out["v2"], "M11RZ", *SMALL, cohort="lb")) == 0  # no flag -> v2
    assert T.main(cli(root, out["pipeline"], "M11RZ", *SMALL, "--labels", "pipeline", cohort="lb")) == 0
    seen = {}
    for labels in ("v2", "pipeline"):
        d = run_dir(out[labels], "M11RZ", cohort="lb")
        met = json.loads((d / "metrics.json").read_text())
        assert met["labels"] == labels
        assert met["label_fields"] == dict(zip(("y", "target"), LABEL_KEYS[labels]))
        df = pd.read_csv(d / "pred_test.csv")
        keys = list(zip(df["match_id"], df["team_id"], df["phase"]))
        assert keys == target_keys(test_dump, labels)  # the CSV rows are the selected risk set
        y_key, t_key = LABEL_KEYS[labels]
        want_y = {(m["match_id"], int(m["team_ids"][i]), int(p) + 1): float(m[y_key][p, i])
                  for m in test_dump["matches"] for p, i in m[t_key].nonzero().tolist()}  # fmt: skip
        assert all(df["y"].iloc[k] == want_y[key] for k, key in enumerate(keys))  # the selected labels
        # every metric is computed on exactly those rows
        assert met["test"]["n_rows"] == met["test"]["hazard"]["n"] == len(df)
        assert met["data"]["n_target_rows"]["test"] == len(df)
        assert met["data"]["labels"] == labels
        ckpt = torch.load(d / "best.pt", weights_only=False)
        assert ckpt["labels"] == labels
        seen[labels] = (keys, df["y"].to_numpy())
        # ... and the training risk set follows the choice too (what the loss sees)
        prep = T.prepare(T.parse_args(cli(root, out[labels], "M11RZ", *SMALL, "--labels", labels, cohort="lb")))
        train_dump = torch.load(root / "lb" / "train.pt", weights_only=False)
        batch = prep.datasets["train"].get_batch(list(range(len(train_dump["matches"]))))
        assert int(batch["target_mask"].sum()) == sum(int(m[t_key].sum()) for m in train_dump["matches"])
        assert met["data"]["n_target_rows"]["train"] == int(batch["target_mask"].sum())
    assert seen["v2"][0] != seen["pipeline"][0]  # the two label choices select different rows
    assert met["data"]["n_target_rows"]["train"] != json.loads(
        (run_dir(out["v2"], "M11RZ", cohort="lb") / "metrics.json").read_text()
    )["data"]["n_target_rows"]["train"]


def test_neural_and_hgb_rows_agree_per_label_choice_and_paired_bootstrap_refuses_mixing(tmp_path):
    root = tmp_path / "dumps"
    write_cohort_dumps(root, "lb", sizes={"train": 14, "val": 6, "test": 10}, seed=12)
    out = tmp_path / "out"
    for labels in ("v2", "pipeline"):
        o = out / labels
        assert T.main(cli(root, o, "M00", *SMALL, "--labels", labels, cohort="lb")) == 0
        H.run(root, "lb", "HGB-cut", seed=42, out_dir=o, labels=labels)
        a = run_dir(o, "M00", cohort="lb") / "pred_test.csv"
        b = o / "lb" / "HGB-cut" / "seed42" / "pred_test.csv"
        res = M.paired_bootstrap(a, b, n_boot=15, seed=0)  # raises unless the rows are identical
        assert res["n_rows"] == len(pd.read_csv(a))
        assert json.loads((o / "lb" / "HGB-cut" / "seed42" / "metrics.json").read_text())["labels"] == labels
    with pytest.raises(ValueError, match="rows"):
        M.paired_bootstrap(
            run_dir(out / "v2", "M00", cohort="lb") / "pred_test.csv",
            out / "pipeline" / "lb" / "HGB-cut" / "seed42" / "pred_test.csv",
            n_boot=5,
        )


def test_labels_flag_rejects_unknown_choices_and_v2_needs_the_v2_fields(tmp_path):
    root = tmp_path / "dumps"
    write_cohort_dumps(root, "old", sizes={"train": 8, "val": 4, "test": 4}, seed=13, v2_fields=False)
    with pytest.raises(SystemExit):
        T.main(cli(root, tmp_path / "o", "M00", "--labels", "truth", cohort="old"))
    with pytest.raises(ValueError, match=r"y_hazard_v2|target_mask_v2"):  # default v2 on a legacy dump: loud
        T.main(cli(root, tmp_path / "o", "M00", *SMALL, cohort="old"))
    assert T.main(cli(root, tmp_path / "o", "M00", *SMALL, "--labels", "pipeline", cohort="old")) == 0
    met = json.loads((run_dir(tmp_path / "o", "M00", cohort="old") / "metrics.json").read_text())
    assert met["labels"] == "pipeline"


def test_a_run_dir_with_results_of_the_other_label_choice_is_never_overwritten(tmp_path, cohort_dir):
    out = tmp_path / "out"
    assert T.main(cli(cohort_dir, out, "M00", *SMALL)) == 0  # v2
    first = (run_dir(out, "M00") / "pred_test.csv").read_bytes()
    with pytest.raises(FileExistsError, match="labels"):
        T.main(cli(cohort_dir, out, "M00", *SMALL, "--labels", "pipeline"))
    assert (run_dir(out, "M00") / "pred_test.csv").read_bytes() == first  # untouched
    assert T.main(cli(cohort_dir, out, "M00", *SMALL, "--labels", "v2")) == 0  # same choice: re-run is fine
    assert T.main(cli(cohort_dir, tmp_path / "other", "M00", *SMALL, "--labels", "pipeline")) == 0


def test_team_ai_frac_is_part_of_every_arms_input_and_a_missing_field_fails_loudly(tmp_path):
    root = tmp_path / "dumps"
    write_cohort_dumps(root, "ai", sizes={"train": 8, "val": 4, "test": 4}, seed=14, ai_nan_frac=0.2)
    out = tmp_path / "out"
    for arm in ("M00", "M11RZ", "M11RZ-noHab"):
        assert T.main(cli(root, out, arm, *SMALL, cohort="ai")) == 0
        ckpt = torch.load(run_dir(out, arm, cohort="ai") / "best.pt", weights_only=False)
        assert ckpt["feature_names"][-3:] == ["team_pos_x", "team_pos_y", "team_ai_frac"]
        assert ckpt["config"]["in_features"] == len(ckpt["feature_names"])
        assert "ai_mean" in ckpt["standardizer"] and "ai_std" in ckpt["standardizer"]
    assert T.main(cli(root, out / "nopos", "M00", *SMALL, "--no-pos-feature", cohort="ai")) == 0
    names = torch.load(run_dir(out / "nopos", "M00", cohort="ai") / "best.pt", weights_only=False)["feature_names"]
    assert names[-1] == "team_ai_frac" and "team_pos_x" not in names  # only the position flag exists
    strip_field(root, "ai", "team_ai_frac")
    with pytest.raises(ValueError, match="team_ai_frac"):
        T.main(cli(root, tmp_path / "out2", "M00", *SMALL, cohort="ai"))
    assert not (tmp_path / "out2").exists() or not any((tmp_path / "out2").rglob("pred_test.csv"))
    with pytest.raises(ValueError, match="team_ai_frac"):
        H.run(root, "ai", "HGB-cut", seed=42, out_dir=tmp_path / "out3")


def test_match_type_is_in_the_prediction_csv_and_metrics_are_also_reported_per_type(tmp_path, cohort_dir):
    out = tmp_path / "out"
    assert T.main(cli(cohort_dir, out, "M11RZ", *SMALL)) == 0
    d = run_dir(out, "M11RZ")
    test_dump = torch.load(cohort_dir / "c" / "test.pt", weights_only=False)
    want = {m["match_id"]: m["match_type"] for m in test_dump["matches"]}
    for split in ("val", "test"):
        df = pd.read_csv(d / f"pred_{split}.csv")
        assert list(df.columns) == M.PRED_COLUMNS and df.columns[-1] == "match_type"
        assert df["match_type"].notna().all()
    df = pd.read_csv(d / "pred_test.csv")
    assert (df["match_type"] == df["match_id"].map(want)).all()
    met = json.loads((d / "metrics.json").read_text())
    by = met["test"]["by_match_type"]
    assert set(by) == set(df["match_type"]) and len(by) == 2
    assert sum(b["n_rows"] for b in by.values()) == met["test"]["n_rows"] == len(df)
    for mt, block in by.items():
        sub = df[df["match_type"] == mt]
        assert block["n_rows"] == len(sub) and block["n_matches"] == sub["match_id"].nunique()
        assert block["hazard"]["n"] == len(sub)
    assert met["data"]["match_types"]["test"] == {mt: sum(1 for v in want.values() if v == mt) for mt in set(want.values())}


def test_match_type_population_guard_and_override(tmp_path):
    root = tmp_path / "dumps"
    write_cohort_dumps(root, "leaky", sizes={"train": 9, "val": 6, "test": 6}, seed=15, match_types=("official", "airoyale"))
    with pytest.raises(ValueError, match="airoyale"):
        T.main(cli(root, tmp_path / "o", "M00", *SMALL, cohort="leaky"))
    argv = cli(root, tmp_path / "o2", "M00", *SMALL, "--match-types", "official,airoyale", cohort="leaky")
    assert T.main(argv) == 0
    met = json.loads((run_dir(tmp_path / "o2", "M00", cohort="leaky") / "metrics.json").read_text())
    assert met["data"]["match_types_allowed"] == ["official", "airoyale"]
    assert set(met["test"]["by_match_type"]) == {"official", "airoyale"}


def test_cohort_discovery_feeds_the_training_cli(tmp_path):
    from pubg_lab.survival.ptr.data.cohorts import discover_cohorts

    root = tmp_path / "dumps"
    for k, name in enumerate(("erangel_tpp_squad", "taego_tpp_squad")):
        write_cohort_dumps(root, name, sizes={"train": 6, "val": 3, "test": 3}, seed=20 + k)
    out = tmp_path / "out"
    for name in discover_cohorts(root):
        assert T.main(cli(root, out, "M00", *SMALL, cohort=name)) == 0
    assert sorted(p.name for p in out.iterdir()) == ["erangel_tpp_squad", "taego_tpp_squad"]


# =========================================================================== ghost fix: which alive mask was used
def test_alive_mask_rule_is_logged_and_recorded_for_every_dump_format(tmp_path, caplog):
    import logging

    root = tmp_path / "dumps"
    write_cohort_dumps(root, "ev", sizes={"train": 10, "val": 4, "test": 4}, seed=30)  # event-based (new dump)
    write_cohort_dumps(root, "old", sizes={"train": 10, "val": 4, "test": 4}, seed=30, event_death=False)
    out = tmp_path / "out"
    cases = [  # (cohort, extra args, expected rule, ghost tokens removed?, tokens added?)
        ("ev", [], "node_mask_v2", True),
        ("ev", ["--labels", "pipeline"], "node_mask|target_pipeline", False),
        ("old", [], "node_mask|target_v2", False),
    ]
    for cohort, extra, rule, removes in cases:
        caplog.clear()
        with caplog.at_level(logging.INFO, logger='pubg_lab.survival.ptr.training.train'):
            assert T.main(cli(root, out / f"{cohort}{len(extra)}", "M00", *SMALL, *extra, cohort=cohort)) == 0
        assert any(f"alive tokens: {rule}" in r.getMessage() for r in caplog.records), (cohort, extra, rule)
        met = json.loads((run_dir(out / f"{cohort}{len(extra)}", "M00", cohort=cohort) / "metrics.json").read_text())
        am = met["data"]["alive_mask"]
        assert am["source"] == rule
        tr = am["per_split"]["train"]
        assert set(am["per_split"]) == {"train", "val", "test"} and tr["source"] == rule
        assert (tr["ghost_removed"] > 0) == removes
        assert tr["n_alive"] > 0 and tr["n_node_mask"] > 0
        if rule == "node_mask_v2":  # audit against the dump's own node_mask_v2: the loader is the looser rule
            assert tr["only_loader"] >= 0 and tr["only_dump"] == 0 and tr["agree"] > 0
        if rule == "node_mask|target_v2":
            assert tr["raised"] > 0 and tr["ghost_removed"] == 0


def test_event_based_alive_tokens_change_what_the_model_trains_on(tmp_path):
    """Same synthetic matches, event-based vs legacy death times: the v2 runs see different token sets (ghosts)."""
    root = tmp_path / "dumps"
    write_cohort_dumps(root, "ev", sizes={"train": 12, "val": 4, "test": 4}, seed=31)
    write_cohort_dumps(root, "old", sizes={"train": 12, "val": 4, "test": 4}, seed=31, event_death=False)
    sets = {}
    for cohort in ("ev", "old"):
        prep = T.prepare(T.parse_args(cli(root, tmp_path / "o", "M00", *SMALL, cohort=cohort)))
        ds = prep.datasets["train"]
        sets[cohort] = (ds.alive_source, sum(int(ds[i]["node_mask"].sum()) for i in range(len(ds))))
    assert sets["ev"][0] == "node_mask_v2" and sets["old"][0] == "node_mask|target_v2"
    assert sets["ev"][1] < sets["old"][1]  # ghosts removed, nothing else added at this noise level


# =========================================================================== config hash (resumable runner) and threads
def test_config_hash_is_stamped_and_follows_every_input_that_changes_a_result(tmp_path):
    root = tmp_path / "dumps"
    write_cohort_dumps(root, "hc", sizes={"train": 8, "val": 3, "test": 3}, seed=40)
    base = cli(root, tmp_path / "o", "M00", *SMALL, cohort="hc")
    h0 = T.config_hash(T.parse_args(base))
    assert isinstance(h0, str) and len(h0) == 16
    assert h0 == T.config_hash(T.parse_args(base))  # deterministic
    assert T.main(base) == 0
    stamped = json.loads((run_dir(tmp_path / "o", "M00", cohort="hc") / "metrics.json").read_text())
    assert stamped["config_hash"] == h0
    # every setting that changes a result changes the hash ...
    changed = {
        "arm": cli(root, tmp_path / "o", "M11", *SMALL, cohort="hc"),
        "seed": cli(root, tmp_path / "o", "M00", *SMALL, cohort="hc", seed=13),
        "epochs": cli(root, tmp_path / "o", "M00", *SMALL, cohort="hc", epochs=2),
        "labels": cli(root, tmp_path / "o", "M00", *SMALL, "--labels", "pipeline", cohort="hc"),
        "lr": cli(root, tmp_path / "o", "M00", *SMALL, "--lr", "0.01", cohort="hc"),
        "d_model": cli(root, tmp_path / "o", "M00", "--d-model", "64", "--n-heads", "4", "--n-layers", "2", cohort="hc"),
        "match_types": cli(root, tmp_path / "o", "M00", *SMALL, "--match-types", "any", cohort="hc"),
        "rank": cli(root, tmp_path / "o", "M00", *SMALL, "--rank-reduction", "group", cohort="hc"),
        "pos": cli(root, tmp_path / "o", "M00", *SMALL, "--no-pos-feature", cohort="hc"),
        "combat": cli(root, tmp_path / "o", "M00", *SMALL, "--combat-weights", "raw", cohort="hc"),
    }
    seen = {h0}
    for name, argv in changed.items():
        h = T.config_hash(T.parse_args(argv))
        assert h not in seen, name
        seen.add(h)
    # ... and the ones that cannot change a result do not: threads, out dir, log noise
    same = cli(root, tmp_path / "elsewhere", "M00", *SMALL, "--threads", "1", cohort="hc")
    assert T.config_hash(T.parse_args(same)) == h0
    # a re-dump (changed file) invalidates the hash
    import os

    os.utime(root / "hc" / "train.pt", ns=(1, 1))
    assert T.config_hash(T.parse_args(base)) != h0


def test_config_hash_follows_the_result_affecting_code(monkeypatch):
    import pubg_lab.survival.ptr.common.util as util

    h = util.code_fingerprint()
    assert h == util.code_fingerprint() and len(h) == 40
    monkeypatch.setattr(util, "RESULT_MODULES", util.RESULT_MODULES[:-1])
    assert util.code_fingerprint() != h  # the fingerprint covers exactly the listed modules
    assert {"data", "model", "loss", "train", "hgb", "metrics", "relations", "arms"} <= set(util.RESULT_MODULES)
    assert not {"run_matrix", "aggregate", "synthetic", "cohorts"} & set(util.RESULT_MODULES)


def test_threads_flag_limits_torch_threads_during_the_run_and_restores_them(tmp_path, cohort_dir):
    before = torch.get_num_threads()
    assert T.main(cli(cohort_dir, tmp_path / "o", "M00", *SMALL, "--threads", "1")) == 0
    assert torch.get_num_threads() == before  # restored
    with pytest.raises(SystemExit):
        T.main(cli(cohort_dir, tmp_path / "o2", "M00", "--threads", "0"))
