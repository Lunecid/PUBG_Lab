"""HGB baselines built from the same dump (identical rows to the neural arms), DB-free.

Row layout (design doc v2.1 / v2.3): ``HGB-cut`` = ``[X[p] | team_ai_frac]``; ``HGB-lag`` = ``[X[p] |
team_ai_frac | X[p-1] | X[p] - X[p-1]]`` (the AI share is a team property, so it is not lagged);
``HGB-noHab`` = ``[X[p] without the zone / habitat groups | team_ai_frac]``.  Rows are the target rows of the
selected label set (``labels='v2'`` by default).
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

import pubg_lab.survival.ptr.models.hgb as H
import pubg_lab.survival.ptr.evaluation.metrics as M
from pubg_lab.survival.ptr.data.synthetic import make_dump, write_cohort_dumps

LABEL_KEYS = {"v2": ("y_hazard_v2", "target_mask_v2"), "pipeline": ("y_hazard", "target_mask")}


def labels_of(m, labels="v2"):
    y_key, t_key = LABEL_KEYS[labels]
    return m[y_key], m[t_key]


def event_alive(m):
    """Independent ghost-fix reference: alive at the start of slot k iff T > phase_end_s[k-1] (start(0) = 0)."""
    T = m["team_death_s"].double().numpy()
    end = m["phase_end_s"].double().numpy()
    start = np.r_[0.0, end[:-1]]
    with np.errstate(invalid="ignore"):
        alive = (~np.isnan(T))[None, :] & (T[None, :] > start[:, None]) & np.isfinite(end)[:, None]
    return torch.from_numpy(alive)


def alive_of(m, labels="v2"):
    """Alive tokens HGB-lag uses: the event mask for v2 on a new-format dump, else node_mask raised over targets."""
    if labels == "v2" and "node_mask_v2" in m:
        return event_alive(m)
    return m["node_mask"] | labels_of(m, labels)[1]


def target_keys(dump, labels="v2"):
    keys = []
    for m in dump["matches"]:
        p, i = labels_of(m, labels)[1].nonzero(as_tuple=True)
        for pp, ii in zip(p.tolist(), i.tolist()):
            keys.append((m["match_id"], int(m["team_ids"][ii]), pp + 1))  # CSV phase = cut k = slot p + 1
    return sorted(keys)


def row_keys(rows):
    return list(
        zip(rows.keys["match_id"], rows.keys["team_id"].astype(int), rows.keys["phase"].astype(int))
    )


def lookup(dump):
    return {
        m["match_id"]: (m, {int(t): k for k, t in enumerate(m["team_ids"].tolist())})
        for m in dump["matches"]
    }


def kill_token(m, p, i):
    """Make (p, i) a dead token under every label choice (node_mask and both risk sets)."""
    m["node_mask"][p, i] = False
    m["target_mask"][p, i] = False
    m["target_mask_v2"][p, i] = False


def test_variants_and_classifier_settings_match_feature_ablation():
    assert H.HGB_VARIANTS == ("HGB-cut", "HGB-lag", "HGB-noHab")
    p = H.make_classifier(seed=7).get_params()
    # model_stgcn/feature_ablation.py:462-467 (max_iter=200, lr=0.06, l2=0.01) + the design doc's
    # early_stopping=False and random_state=seed
    assert p["early_stopping"] is False
    assert p["max_iter"] == 200 and p["learning_rate"] == 0.06 and p["l2_regularization"] == 0.01
    assert p["random_state"] == 7


@pytest.mark.parametrize("labels", ["v2", "pipeline"])
def test_hgb_cut_rows_are_exactly_the_target_rows_with_x_at_p(labels):
    dump = make_dump(5, seed=1, nan_frac=0.05, ai_nan_frac=0.2)
    rows = H.build_rows(dump, "HGB-cut", labels=labels)
    assert row_keys(rows) == target_keys(dump, labels)  # ordered by (match_id, team_id, phase)
    look = lookup(dump)
    F = len(dump["meta"]["feature_names"])
    for k, (mid, tid, phase) in enumerate(row_keys(rows)):
        m, tix = look[mid]
        i, p = tix[tid], phase - 1  # CSV phase k = slot p + 1
        y, _ = labels_of(m, labels)
        np.testing.assert_array_equal(rows.X[k, :F], m["X"][p, i].numpy())  # NaN kept as NaN
        np.testing.assert_array_equal(rows.X[k, F], m["team_ai_frac"][i].numpy())  # raw share, NaN kept
        assert rows.y[k] == y[p, i]
        assert rows.final_rank_norm[k] == m["final_rank_norm"][i]
        assert rows.top5[k] == m["top5"][i]
        assert rows.match_type[k] == m["match_type"]
    assert rows.X.shape[1] == F + 1
    assert rows.feature_names == dump["meta"]["feature_names"] + ["team_ai_frac"]
    assert np.isnan(rows.X[:, F]).any()  # HGB gets the NaN as a missing value, not a fill


def test_hgb_lag_uses_only_previous_phase_and_is_nan_when_absent():
    dump = make_dump(6, seed=2, dead_garbage=False)
    names = dump["meta"]["feature_names"]
    F = len(names)
    rows = H.build_rows(dump, "HGB-lag")
    assert rows.X.shape[1] == 3 * F + 1
    assert rows.feature_names == (
        names + ["team_ai_frac"] + [f"lag1:{n}" for n in names] + [f"diff1:{n}" for n in names]
    )
    cur, ai = rows.X[:, :F], rows.X[:, F]
    lag, diff = rows.X[:, F + 1 : 2 * F + 1], rows.X[:, 2 * F + 1 :]
    look = lookup(dump)
    absent = 0
    for k, (mid, tid, phase) in enumerate(row_keys(rows)):
        m, tix = look[mid]
        i, p = tix[tid], phase - 1  # CSV phase k = slot p + 1
        np.testing.assert_array_equal(cur[k], m["X"][p, i].numpy())
        assert ai[k] == m["team_ai_frac"][i]
        if p >= 1 and bool(alive_of(m)[p - 1, i]):
            np.testing.assert_array_equal(lag[k], m["X"][p - 1, i].numpy())
            np.testing.assert_allclose(diff[k], (m["X"][p, i] - m["X"][p - 1, i]).numpy(), atol=1e-6)
        else:  # design doc v2.1: no previous record -> NaN (HGB treats it as missing), not 0
            absent += 1
            assert np.isnan(lag[k]).all() and np.isnan(diff[k]).all()
    assert absent > 0
    assert not np.isnan(cur).any() and not np.isnan(ai).any()  # NaN only in the lag / diff blocks

    # nothing but phases p and p-1 can reach a row at phase p
    phase = rows.keys["phase"].to_numpy() - 1  # slot index p
    n_phases = dump["matches"][0]["X"].shape[0]
    for p0 in range(1, n_phases - 1):
        pert = copy.deepcopy(dump)
        for m in pert["matches"]:
            others = [q for q in range(n_phases) if q not in (p0 - 1, p0)]
            m["X"][others] = torch.randn_like(m["X"][others]) * 7
        again = H.build_rows(pert, "HGB-lag")
        np.testing.assert_array_equal(again.X[phase == p0], rows.X[phase == p0])
    # changing p-1 moves the lag/diff blocks but not the current block (nor the team-level AI share)
    pert = copy.deepcopy(dump)
    for m in pert["matches"]:
        m["X"][2] += 3.0
    again = H.build_rows(pert, "HGB-lag")
    sel = phase == 3
    present = ~np.isnan(lag[sel]).any(1)
    assert present.any()
    np.testing.assert_array_equal(again.X[sel][:, : F + 1], rows.X[sel][:, : F + 1])
    assert not np.allclose(again.X[sel][present][:, F + 1 :], rows.X[sel][present][:, F + 1 :])



def test_hgb_lag_is_nan_without_an_alive_token_at_the_previous_phase():
    """Fallback (legacy dump) alive rule: a token that node_mask / the risk set do not cover is no previous record."""
    dump = make_dump(6, seed=2, dead_garbage=False, event_death=False)
    F = len(dump["meta"]["feature_names"])
    rows = H.build_rows(dump, "HGB-lag")
    pert = copy.deepcopy(dump)
    m = pert["matches"][0]
    p_, i_ = [(p, i) for p, i in m["target_mask_v2"].nonzero().tolist() if p >= 2][0]
    kill_token(m, p_ - 1, i_)  # rows at p_ are unaffected
    again = H.build_rows(pert, "HGB-lag")
    k = row_keys(again).index((m["match_id"], int(m["team_ids"][i_]), p_ + 1))
    assert np.isnan(again.X[k, F + 1 :]).all()  # no previous record -> NaN lag / diff although the tensor has numbers
    assert not np.isnan(again.X[k, : F + 1]).any()
    # ... while the same row of the untouched dump has a lag
    k0 = row_keys(rows).index((m["match_id"], int(m["team_ids"][i_]), p_ + 1))
    assert not np.isnan(rows.X[k0, F + 1 :]).any()


def test_hgb_nohab_drops_exactly_zone_and_habitat_columns():
    dump = make_dump(3, n_features=150, seed=3)
    groups = dump["meta"]["feature_groups"]
    cut, nohab = H.build_rows(dump, "HGB-cut"), H.build_rows(dump, "HGB-noHab")
    drop = set(groups["zone"]) | set(groups["habitat"])
    keep = [c for c in range(150) if c not in drop]
    np.testing.assert_array_equal(nohab.X, cut.X[:, keep + [150]])  # the AI share stays
    assert nohab.feature_names == [dump["meta"]["feature_names"][c] for c in keep] + ["team_ai_frac"]
    assert row_keys(nohab) == row_keys(cut)


@pytest.mark.parametrize("labels", ["v2", "pipeline"])
def test_all_variants_use_identical_rows(labels):
    dump = make_dump(4, seed=4)
    keys = [row_keys(H.build_rows(dump, v, labels=labels)) for v in H.HGB_VARIANTS]
    assert keys[0] == keys[1] == keys[2] == target_keys(dump, labels)
    with pytest.raises(ValueError):
        H.build_rows(dump, "HGB-nope")


# --------------------------------------------------------------------------- v2.3: labels
def test_label_choice_selects_different_rows_when_v2_and_pipeline_differ():
    dump = make_dump(10, seed=60)
    for variant in H.HGB_VARIANTS:
        v2 = H.build_rows(dump, variant)  # default = v2
        pipe = H.build_rows(dump, variant, labels="pipeline")
        assert row_keys(v2) == target_keys(dump, "v2") and row_keys(pipe) == target_keys(dump, "pipeline")
        assert row_keys(v2) != row_keys(pipe), "the label sets differ, so must the HGB rows"
        common = set(row_keys(v2)) & set(row_keys(pipe))
        ya = dict(zip(row_keys(v2), v2.y))
        yb = dict(zip(row_keys(pipe), pipe.y))
        assert any(ya[k] != yb[k] for k in common), "shared rows carry different labels"
        # labels of every row come from the selected set
        look = lookup(dump)
        for rows, labels in ((v2, "v2"), (pipe, "pipeline")):
            for k, (mid, tid, phase) in enumerate(row_keys(rows)):
                m, tix = look[mid]
                assert rows.y[k] == labels_of(m, labels)[0][phase - 1, tix[tid]]
    legacy = make_dump(3, seed=60, v2_fields=False)
    assert len(H.build_rows(legacy, "HGB-cut", labels="pipeline").y) > 0
    with pytest.raises(ValueError, match=r"target_mask_v2|y_hazard_v2"):
        H.build_rows(legacy, "HGB-cut")
    with pytest.raises(ValueError, match="labels"):
        H.build_rows(dump, "HGB-cut", labels="truth")


def test_lag_presence_follows_the_alive_rule_of_the_selected_labels():
    """Event-based v2: every team alive at a phase was alive at the one before (monotone), so only the first cut lacks a
    lag.  Pipeline labels: node_mask | pipeline targets, whose gaps (missing tokens) create extra absent lags."""
    dump = make_dump(10, seed=61)
    F = len(dump["meta"]["feature_names"])
    look = lookup(dump)
    absent_after_first = {"v2": 0, "pipeline": 0}
    for labels in ("v2", "pipeline"):
        rows = H.build_rows(dump, "HGB-lag", labels=labels)
        for k, (mid, tid, phase) in enumerate(row_keys(rows)):
            m, tix = look[mid]
            i, p = tix[tid], phase - 1
            has_prev = p >= 1 and bool(alive_of(m, labels)[p - 1, i])
            assert np.isnan(rows.X[k, F + 1]) == (not has_prev), (labels, mid, tid, phase)
            absent_after_first[labels] += int(p >= 1 and not has_prev)
    assert absent_after_first["v2"] == 0
    assert absent_after_first["pipeline"] > 0


def test_lag_presence_under_the_fallback_mask_counts_tokens_raised_over_v2_targets():
    dump = make_dump(10, seed=61, event_death=False)  # legacy dump: alive = node_mask | target(labels)
    F = len(dump["meta"]["feature_names"])
    look = lookup(dump)
    raised_rows = 0
    rows = H.build_rows(dump, "HGB-lag")
    for k, (mid, tid, phase) in enumerate(row_keys(rows)):
        m, tix = look[mid]
        i, p = tix[tid], phase - 1
        has_prev = p >= 1 and bool(alive_of(m, "v2")[p - 1, i])
        assert np.isnan(rows.X[k, F + 1]) == (not has_prev)
        raised_rows += int(p >= 1 and not bool(m["node_mask"][p - 1, i]) and bool(m["target_mask_v2"][p - 1, i]))
    assert raised_rows > 0, "test needs a v2 row whose previous token exists only through the v2 target"


def test_ghost_tokens_never_count_as_a_previous_record_for_event_based_v2():
    """A ghost (node_mask True, events dead) must not give a team at a later phase a lag it cannot have: and a team
    that is alive at p but whose p-1 token is a ghost-free alive token always has its lag."""
    dump = make_dump(10, seed=64)
    ghosts = sum(int((m["node_mask"] & ~event_alive(m)).sum()) for m in dump["matches"])
    assert ghosts > 0
    rows = H.build_rows(dump, "HGB-lag")
    F = len(dump["meta"]["feature_names"])
    p1 = rows.keys["phase"].to_numpy() == 1  # cut phase 1 = slot 0: the first cut has no previous record
    assert np.isnan(rows.X[p1][:, F + 1 :]).all() and not np.isnan(rows.X[~p1][:, F + 1 :]).any()


# --------------------------------------------------------------------------- v2.3: team_ai_frac
def test_team_ai_frac_enters_every_variant_once_and_is_not_lagged():
    dump = make_dump(4, seed=62, ai_nan_frac=0.3)
    for variant in H.HGB_VARIANTS:
        rows = H.build_rows(dump, variant)
        assert rows.feature_names.count("team_ai_frac") == 1
        col = rows.feature_names.index("team_ai_frac")
        look = lookup(dump)
        for k, (mid, tid, phase) in enumerate(row_keys(rows)):
            m, tix = look[mid]
            np.testing.assert_array_equal(rows.X[k, col], m["team_ai_frac"][tix[tid]].numpy())
        assert not any(n.endswith("team_ai_frac") and n != "team_ai_frac" for n in rows.feature_names)


def test_missing_team_ai_frac_fails_loudly():
    dump = make_dump(3, seed=63)
    dump["matches"][1].pop("team_ai_frac")
    for variant in H.HGB_VARIANTS:
        with pytest.raises(ValueError, match="team_ai_frac"):
            H.build_rows(dump, variant)


def test_hgb_end_to_end_writes_same_csv_format_as_neural(tmp_path):
    root = tmp_path / "dumps"
    write_cohort_dumps(root, "c", sizes={"train": 20, "val": 6, "test": 6}, seed=4)
    out = tmp_path / "out"
    for variant in ("HGB-cut", "HGB-lag", "HGB-noHab"):
        res = H.run(root, "c", variant, seed=42, out_dir=out)
        assert np.isfinite(res["test"]["hazard"]["nll"])
    run_dir = out / "c" / "HGB-lag" / "seed42"
    splits = {s: torch.load(root / "c" / f"{s}.pt", weights_only=False) for s in ("val", "test")}
    for split in ("val", "test"):
        df = pd.read_csv(run_dir / f"pred_{split}.csv")
        assert list(df.columns) == M.PRED_COLUMNS and df.columns[-1] == "match_type"
        assert df["p_hazard"].between(0, 1).all()
        assert np.allclose(df["rank_score"], -df["p_hazard"])  # no rank head: -p (survival-like, like the head)
        got = list(zip(df["match_id"], df["team_id"], df["phase"]))
        assert got == target_keys(splits[split])
        assert df[["y", "p_hazard", "rank_score", "final_rank_norm", "top5", "match_type"]].notna().all().all()
        want_type = {m["match_id"]: m["match_type"] for m in splits[split]["matches"]}
        assert (df["match_type"] == df["match_id"].map(want_type)).all()
        assert set(df["match_type"]) == {"official", "competitive"}
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["variant"] == "HGB-lag" and metrics["seed"] == 42 and metrics["cohort"] == "c"
    assert metrics["n_features"] == 3 * 24 + 1 and metrics["classifier"]["early_stopping"] is False
    assert metrics["labels"] == "v2" and metrics["label_fields"] == {"y": "y_hazard_v2", "target": "target_mask_v2"}
    assert {"val", "test"} <= set(metrics) and "auc" in metrics["test"]["hazard"]
    assert set(metrics["test"]["by_match_type"]) == {"official", "competitive"}
    assert metrics["elapsed_sec"] >= 0

    # rows are identical between HGB variants, so paired bootstrap works directly
    a = out / "c" / "HGB-cut" / "seed42" / "pred_test.csv"
    b = out / "c" / "HGB-lag" / "seed42" / "pred_test.csv"
    res = M.paired_bootstrap(a, b, n_boot=20, seed=0)
    assert res["n_rows"] == len(pd.read_csv(a))


def test_hgb_run_follows_the_label_choice_and_records_it(tmp_path):
    root = tmp_path / "dumps"
    write_cohort_dumps(root, "c", sizes={"train": 20, "val": 8, "test": 8}, seed=7)
    test_dump = torch.load(root / "c" / "test.pt", weights_only=False)
    seen = {}
    for labels in ("v2", "pipeline"):
        res = H.run(root, "c", "HGB-cut", seed=42, out_dir=tmp_path / f"out_{labels}", labels=labels)
        run_dir = tmp_path / f"out_{labels}" / "c" / "HGB-cut" / "seed42"
        df = pd.read_csv(run_dir / "pred_test.csv")
        keys = list(zip(df["match_id"], df["team_id"], df["phase"]))
        assert keys == target_keys(test_dump, labels)
        meta = json.loads((run_dir / "metrics.json").read_text())
        assert meta["labels"] == labels == res["labels"]
        assert meta["label_fields"] == dict(zip(("y", "target"), LABEL_KEYS[labels]))
        assert meta["n_train_rows"] == sum(int(labels_of(m, labels)[1].sum()) for m in torch.load(root / "c" / "train.pt", weights_only=False)["matches"])
        # every metric is computed on exactly these rows
        assert meta["test"]["n_rows"] == len(df) and meta["test"]["hazard"]["n"] == len(df)
        seen[labels] = (keys, df["y"].to_numpy())
    assert seen["v2"][0] != seen["pipeline"][0]  # different risk sets
    # the paired utility refuses to compare runs that used different label choices on different rows
    with pytest.raises(ValueError, match="rows"):
        M.paired_bootstrap(
            tmp_path / "out_v2" / "c" / "HGB-cut" / "seed42" / "pred_test.csv",
            tmp_path / "out_pipeline" / "c" / "HGB-cut" / "seed42" / "pred_test.csv",
            n_boot=5,
        )


def test_hgb_run_on_a_legacy_dump_needs_labels_pipeline(tmp_path):
    root = tmp_path / "dumps"
    write_cohort_dumps(root, "old", sizes={"train": 12, "val": 4, "test": 4}, seed=9, v2_fields=False)
    with pytest.raises(ValueError, match=r"y_hazard_v2|target_mask_v2"):  # default labels = v2
        H.run(root, "old", "HGB-cut", seed=42, out_dir=tmp_path / "out")
    res = H.run(root, "old", "HGB-cut", seed=42, out_dir=tmp_path / "out", labels="pipeline")
    assert res["labels"] == "pipeline" and res["test"]["n_rows"] > 0
    assert H.main(["--dump-dir", str(root), "--cohort", "old", "--variant", "HGB-cut", "--seed", "42",
                   "--out-dir", str(tmp_path / "out2"), "--labels", "pipeline"]) == 0  # fmt: skip


def test_hgb_cli_main_labels_flag_defaults_to_v2_and_guards_the_output_dir(tmp_path):
    root = tmp_path / "dumps"
    write_cohort_dumps(root, "c", sizes={"train": 12, "val": 4, "test": 4}, seed=5)
    args = ["--dump-dir", str(root), "--cohort", "c", "--variant", "HGB-cut", "--seed", "42"]
    out = tmp_path / "o1"
    assert H.main(args + ["--out-dir", str(out)]) == 0
    assert H.main(args + ["--out-dir", str(tmp_path / "o2")]) == 0
    run = out / "c" / "HGB-cut" / "seed42"
    assert json.loads((run / "metrics.json").read_text())["labels"] == "v2"  # the default
    a, b = pd.read_csv(run / "pred_test.csv"), pd.read_csv(tmp_path / "o2" / "c" / "HGB-cut" / "seed42" / "pred_test.csv")
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-7)
    # re-running the same labels overwrites; the other label choice must not clobber v2 results
    assert H.main(args + ["--out-dir", str(out), "--labels", "v2"]) == 0
    with pytest.raises(FileExistsError, match="labels"):
        H.main(args + ["--out-dir", str(out), "--labels", "pipeline"])
    assert json.loads((run / "metrics.json").read_text())["labels"] == "v2"
    assert H.main(args + ["--out-dir", str(tmp_path / "o3"), "--labels", "pipeline"]) == 0
    with pytest.raises(SystemExit):
        H.main(args + ["--out-dir", str(tmp_path / "o4"), "--labels", "truth"])


def test_hgb_run_applies_the_match_type_guard(tmp_path):
    root = tmp_path / "dumps"
    write_cohort_dumps(root, "leaky", sizes={"train": 8, "val": 4, "test": 4}, seed=8, match_types=("official", "airoyale"))
    with pytest.raises(ValueError, match="airoyale"):
        H.run(root, "leaky", "HGB-cut", seed=42, out_dir=tmp_path / "out")
    res = H.run(root, "leaky", "HGB-cut", seed=42, out_dir=tmp_path / "out", match_types=("official", "airoyale"))
    assert set(res["test"]["by_match_type"]) == {"official", "airoyale"}


def test_hgb_nohab_keeps_alive_count_variables_and_lists_removed_columns(tmp_path):
    """Design doc v2.2: noHab removes the zone / habitat groups but keeps alive-count variables."""
    dump = make_dump(3, n_features=24, seed=18, alive_count_names=True)
    g = dump["meta"]["feature_groups"]
    names = dump["meta"]["feature_names"]
    removed = {g["zone"][2], g["habitat"][1], g["habitat"][2]}  # safe_radius_start_norm + two hab_* spatial
    keep = [c for c in range(24) if c not in removed]
    cut, nohab = H.build_rows(dump, "HGB-cut"), H.build_rows(dump, "HGB-noHab")
    np.testing.assert_array_equal(nohab.X, cut.X[:, keep + [24]])
    assert nohab.feature_names == [names[c] for c in keep] + ["team_ai_frac"]
    assert {"avg_alive_teams_norm", "avg_alive_players_norm", "hab_alive_teams"} <= set(nohab.feature_names)
    assert nohab.removal["removed_columns"] == [names[c] for c in sorted(removed)]
    assert set(nohab.removal["kept_by_exception"]) == {"avg_alive_teams_norm", "avg_alive_players_norm", "hab_alive_teams"}
    assert cut.removal["n_removed"] == 0 and H.build_rows(dump, "HGB-lag").removal["n_removed"] == 0

    root = tmp_path / "dumps"
    write_cohort_dumps(root, "c", sizes={"train": 14, "val": 4, "test": 4}, seed=6, alive_count_names=True)
    res = H.run(root, "c", "HGB-noHab", seed=42, out_dir=tmp_path / "out")
    saved = json.loads((tmp_path / "out" / "c" / "HGB-noHab" / "seed42" / "metrics.json").read_text())
    assert saved["feature_removal"]["removed_columns"] == [names[c] for c in sorted(removed)]
    assert saved["feature_removal"]["n_removed"] == 3 and res["n_features"] == 21 + 1
    cutres = H.run(root, "c", "HGB-cut", seed=42, out_dir=tmp_path / "out")
    assert cutres["feature_removal"]["removed_columns"] == []


def test_hgb_metrics_record_the_alive_rule(tmp_path):
    root = tmp_path / "dumps"
    write_cohort_dumps(root, "ev", sizes={"train": 12, "val": 4, "test": 4}, seed=65)
    write_cohort_dumps(root, "old", sizes={"train": 12, "val": 4, "test": 4}, seed=65, event_death=False)
    res_ev = H.run(root, "ev", "HGB-lag", seed=42, out_dir=tmp_path / "o")
    res_old = H.run(root, "old", "HGB-lag", seed=42, out_dir=tmp_path / "o")
    assert res_ev["alive_mask"]["train"]["source"] == "node_mask_v2" and res_ev["alive_mask"]["train"]["ghost_removed"] > 0
    assert res_old["alive_mask"]["train"]["source"] == "node_mask|target_v2" and res_old["alive_mask"]["train"]["raised"] > 0
    pipe = H.run(root, "ev", "HGB-cut", seed=42, out_dir=tmp_path / "o2", labels="pipeline")
    assert pipe["alive_mask"]["test"]["source"] == "node_mask|target_pipeline"


def test_hgb_config_hash_is_stamped_and_follows_its_inputs(tmp_path):
    root = tmp_path / "dumps"
    write_cohort_dumps(root, "hc", sizes={"train": 10, "val": 3, "test": 3}, seed=41)
    argv = ["--dump-dir", str(root), "--cohort", "hc", "--variant", "HGB-cut", "--seed", "42", "--out-dir", str(tmp_path / "o")]
    h0 = H.config_hash(H.parse_args(argv))
    assert len(h0) == 16 and h0 == H.config_hash(H.parse_args(argv))
    assert H.main(argv) == 0
    stamped = json.loads((tmp_path / "o" / "hc" / "HGB-cut" / "seed42" / "metrics.json").read_text())
    assert stamped["config_hash"] == h0
    for flag, value in (("--variant", "HGB-lag"), ("--seed", "13"), ("--labels", "pipeline"), ("--match-types", "any")):
        other = argv.copy()
        if flag in other:
            other[other.index(flag) + 1] = value
        else:
            other += [flag, value]
        assert H.config_hash(H.parse_args(other)) != h0, flag
    assert H.config_hash(H.parse_args(argv + ["--threads", "1"])) == h0  # threads never change a result
    # the same configuration as a neural run is a different hash (kind is part of it)
    import pubg_lab.survival.ptr.training.train as T

    neural = T.parse_args(["--dump-dir", str(root), "--cohort", "hc", "--arm", "M00", "--seed", "42", "--out-dir", "x"])
    assert T.config_hash(neural) != h0
    assert H.main(argv + ["--threads", "1"]) == 0
