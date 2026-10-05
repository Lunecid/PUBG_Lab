"""Extra (kf_*) team features of the UNION arms (design doc 4.6): the join onto our tokens / rows, DB-free.

Join contract under test: ``<extra-dir>/<cohort>/<split>.csv.gz`` holds one row per (match_id, team_id, phase) with the
CUT phase ``k`` (= dump slot ``p + 1``... i.e. ``phase = p + 1`` for the cut at the end of slot ``p``) and ``kf_*`` columns;
the values of ``(match, team, k)`` become the extra token features of slot ``k - 1``.  Every alive token of a cut slot (slots
``0 .. P - 2``, which contains every target row) needs a row; a gap fails with a list.
"""

from __future__ import annotations

import copy
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from model_stgcn.ptr import data as D
from model_stgcn.ptr import extra as X
from model_stgcn.ptr import hgb as H
from model_stgcn.ptr import metrics as M
from model_stgcn.ptr import train as T
from model_stgcn.ptr.arms import ARMS, MAIN_ARMS, UNION_ARMS, get_arm
from model_stgcn.ptr.synthetic import make_dump, make_extra_table, write_cohort_dumps, write_extra_tables
from model_stgcn.ptr.util import code_fingerprint, config_hash as util_hash, dump_fingerprint

SMALL = ["--d-model", "32", "--n-heads", "4", "--n-layers", "2"]
KEYS = ["match_id", "team_id", "phase"]
FLAGS = ("cut_t", "in_risk_set", "alive_at_cut", "y_hazard_v2")  # the table's flag / label columns: never features


def planted(table):
    """``{(match, team, phase): kf values}`` of the rows a model needs (note == 'ok'), columns in name order."""
    names = sorted(c for c in table.columns if c.startswith("kf_"))
    ok = table[table["note"] == "ok"]
    return {(r.match_id, int(r.team_id), int(r.phase)): np.array([getattr(r, n) for n in names]) for r in ok.itertuples()}


def alive_cut_tokens(dump, labels="v2"):
    """Independent reference: ``[(match_id, p, i, is_target)]`` of every alive token of a cut slot."""
    event = D.uses_event_alive(dump, labels)
    out = []
    for m in dump["matches"]:
        alive = D.effective_alive(m, labels, event).numpy()
        target = D.select_labels(m, labels)[1].numpy()
        for p in range(alive.shape[0] - 1):
            for i in np.flatnonzero(alive[p]):
                out.append((str(m["match_id"]), p, int(i), bool(target[p, i])))
    return out


@pytest.fixture(scope="module")
def dump():
    return make_dump(6, seed=3)


@pytest.fixture(scope="module")
def table(dump):
    return make_extra_table(dump, seed=5, n_features=5)


@pytest.fixture()
def csv_path(tmp_path, table):
    path = tmp_path / "train.csv.gz"
    table.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def cohort_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("extra_dumps")
    write_cohort_dumps(root, "c", sizes={"train": 12, "val": 5, "test": 5}, seed=1)
    return root


@pytest.fixture()
def extra_dir(tmp_path, cohort_dir):
    write_extra_tables(cohort_dir, "c", tmp_path / "kf", seed=2, n_features=5)
    return tmp_path / "kf"


# --------------------------------------------------------------------------- registry
def test_union_arm_is_a_relation_arm_with_extra_features_and_no_other_difference():
    assert UNION_ARMS == ("M11R-union",)
    u, base = get_arm("M11R-union", implemented_only=True), ARMS["M11R"]
    assert u.extra and not base.extra
    for field in ("phase_axis", "team_axis", "relation", "zone", "shuffle", "drop_groups", "keep_names", "relation_types", "player"):
        assert getattr(u, field) == getattr(base, field), field
    assert u.exploratory and u.name not in MAIN_ARMS
    assert not any(ARMS[a].extra for a in MAIN_ARMS)


# --------------------------------------------------------------------------- reading the table
def test_table_path_prefers_csv_gz_then_csv_then_parquet(tmp_path):
    base = tmp_path / "c"
    base.mkdir()
    for suffix in (".parquet", ".csv", ".csv.gz"):
        (base / f"train{suffix}").write_text("x")
    assert X.table_path(tmp_path, "c", "train").name == "train.csv.gz"
    (base / "train.csv.gz").unlink()
    assert X.table_path(tmp_path, "c", "train").name == "train.csv"
    (base / "train.csv").unlink()
    assert X.table_path(tmp_path, "c", "train").name == "train.parquet"


def test_table_path_missing_lists_what_was_looked_for(tmp_path):
    (tmp_path / "c").mkdir()
    with pytest.raises(FileNotFoundError) as e:
        X.table_path(tmp_path, "c", "val")
    for suffix in X.TABLE_SUFFIXES:
        assert f"val{suffix}" in str(e.value)


def test_read_table_keeps_the_keys_and_the_kf_columns_only_sorted(csv_path, table):
    df = X.read_table(csv_path)
    names = sorted(c for c in table.columns if c.startswith("kf_"))
    assert list(df.columns) == KEYS + names + list(X.ALIGN_COLUMNS)  # the optional alignment columns ride along
    assert not {"note", "cut_t", "alive_at_cut"} & set(df.columns)  # nothing else is read (cut_t / alive_at_cut never)
    assert len(df) == len(table)
    assert df["team_id"].dtype == np.int64 and df["phase"].dtype == np.int64
    assert all(df[n].dtype == np.float32 for n in names)


def test_read_table_match_id_stays_text(tmp_path, dump):
    t = make_extra_table(dump, seed=5, junk_rows=False)  # no row with a text id: every id below is made of digits
    t["match_id"] = t["match_id"].map(lambda s: "00" + s.split("-")[-1])  # '000000' would become the integer 0 if parsed
    path = tmp_path / "x.csv"
    t.to_csv(path, index=False)
    assert t["match_id"].str.isdigit().all() and t["match_id"].str.startswith("00").all()
    got = X.read_table(path)["match_id"]
    assert set(got) == set(t["match_id"]) and all(isinstance(v, str) for v in got)


def test_read_table_reads_plain_csv_like_csv_gz(tmp_path, csv_path):
    plain = tmp_path / "plain.csv"
    pd.read_csv(csv_path, dtype={"match_id": str}).to_csv(plain, index=False)
    pd.testing.assert_frame_equal(X.read_table(plain), X.read_table(csv_path))


def test_read_table_errors(tmp_path, table):
    def write(df, name="t.csv"):
        path = tmp_path / name
        df.to_csv(path, index=False)
        return path

    with pytest.raises(ValueError, match="phase"):
        X.read_table(write(table.drop(columns=["phase"])))
    with pytest.raises(ValueError, match="kf_"):
        X.read_table(write(table[[*KEYS, "note"]]))
    dup = pd.concat([table, table.iloc[[3]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate") as e:
        X.read_table(write(dup))
    assert table.iloc[3]["match_id"] in str(e.value)
    bad = table.copy()
    bad["kf_1"] = bad["kf_1"].astype(str).replace({bad["kf_1"].astype(str).iloc[0]: "high"})
    with pytest.raises(ValueError, match="kf_1"):
        X.read_table(write(bad))
    frac = table.copy()
    frac["phase"] = frac["phase"] + 0.5
    with pytest.raises(ValueError, match="phase"):
        X.read_table(write(frac))
    nan_team = table.copy()
    nan_team.loc[0, "team_id"] = np.nan
    with pytest.raises(ValueError, match="team_id"):
        X.read_table(write(nan_team))


def test_read_table_parquet_needs_an_engine_and_says_so(tmp_path, monkeypatch):
    path = tmp_path / "train.parquet"
    path.write_bytes(b"not parquet")

    def no_engine(*a, **k):
        raise ImportError("Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'.")

    monkeypatch.setattr(pd, "read_parquet", no_engine)
    with pytest.raises(RuntimeError, match=r"pyarrow.*csv\.gz|csv\.gz.*pyarrow"):
        X.read_table(path)


def test_table_feature_names_reads_the_header_only(csv_path, table):
    assert X.table_feature_names(csv_path) == sorted(c for c in table.columns if c.startswith("kf_"))
    assert X.names_sha1(["kf_a", "kf_b"]) != X.names_sha1(["kf_b", "kf_a"]) and len(X.names_sha1(["kf_a"])) == 40


# --------------------------------------------------------------------------- only kf_* columns can be features
def test_check_extra_names_lets_only_kf_columns_in():
    assert D.check_extra_names(["kf_a", "kf_b"]) == ["kf_a", "kf_b"]
    for bad in (
        ["kf_a", "y_hazard_v2"], ["cut_t"], ["in_risk_set"], ["alive_at_cut"], ["match_id"], ["team_id"], ["phase"],
        ["kf_y_hazard_v2"], ["kf_in_risk_set"], ["kf_cut_t"], ["kf_alive_at_cut"], ["team_ai_frac"], ["x"],
    ):  # fmt: skip
        with pytest.raises(ValueError, match="non-kf"):
            D.check_extra_names(bad)
    with pytest.raises(ValueError, match="no extra"):
        D.check_extra_names([])
    with pytest.raises(ValueError, match="duplicate"):
        D.check_extra_names(["kf_a", "kf_a"])
    assert set(D.EXTRA_FORBIDDEN) >= set(FLAGS)


def test_no_non_kf_column_enters_the_neural_tokens_or_the_hgb_rows(dump, table):
    assert set(FLAGS) <= set(table.columns)  # the table carries the flags and the label next to the kf columns
    kf = sorted(c for c in table.columns if c.startswith("kf_"))
    ex = X.join_split(dump, X.read_table_frame(table), "v2")
    assert ex.names == kf and next(iter(ex.tokens.values())).shape[-1] == len(kf)
    cut, rows = H.build_rows(dump, "HGB-cut", "v2"), H.build_rows(dump, "HGB-union", "v2", extra=ex)
    assert rows.feature_names == cut.feature_names + kf and not set(FLAGS) & set(rows.feature_names)
    assert rows.X.shape[1] == cut.X.shape[1] + len(kf)
    base, ds = make_dataset(dump), make_dataset(dump, ex, X.ExtraStandardizer.fit(ex, dump, "v2"))
    assert ds.feature_names == base.feature_names + kf and not set(FLAGS) & set(ds.feature_names)
    assert ds[0]["x"].shape[-1] == len(ds.feature_names) == base.n_features + len(kf)


def test_a_flag_or_the_label_cannot_come_in_under_a_kf_name(dump, table):
    for smuggled in ("kf_y_hazard_v2", "kf_in_risk_set", "kf_cut_t", "kf_alive_at_cut"):
        t = table.copy()
        t[smuggled] = t["y_hazard_v2"].fillna(0.0)
        with pytest.raises(ValueError, match="non-kf"):
            X.read_table_frame(t)
        with pytest.raises(ValueError, match="non-kf"):
            X.join_split(dump, t, "v2")


def test_a_dataset_refuses_extra_names_that_are_not_kf_columns(dump, table):
    ex = X.join_split(dump, X.read_table_frame(table), "v2")
    sz = X.ExtraStandardizer.fit(ex, dump, "v2")
    forged = X.ExtraSplit(["y_hazard_v2", *ex.names[1:]], ex.tokens, ex.report)
    with pytest.raises(ValueError, match="non-kf"):
        make_dataset(dump, forged, sz)
    with pytest.raises(ValueError, match="non-kf"):
        H.build_rows(dump, "HGB-union", "v2", extra=forged)


def test_label_canary_a_union_model_never_reads_the_flag_and_label_columns(tmp_path):
    """The kf columns carry no cue at all, while y_hazard_v2 (= the label), cut_t and alive_at_cut do: a model that
    read any of them would be perfect, a union model on the kf columns alone stays at chance."""
    write_cohort_dumps(tmp_path, "n", sizes={"train": 40, "val": 10, "test": 14}, seed=4, signal=0.0)

    def leak(df):
        df = df.copy()
        y = df["y_hazard_v2"].fillna(0.0)
        df["cut_t"] = 1000.0 * y + df["phase"]
        df["alive_at_cut"] = (y > 0).astype(int)
        return df

    write_extra_tables(tmp_path, "n", tmp_path / "kf", seed=6, signal=0.0, transform=leak)
    tab = pd.read_csv(tmp_path / "kf" / "n" / "test.csv.gz")
    risk = tab[tab["in_risk_set"] == 1]
    assert risk["y_hazard_v2"].nunique() == 2 and (risk["alive_at_cut"] == risk["y_hazard_v2"]).all()  # the leak is real
    res = H.run(tmp_path, "n", "HGB-union", 42, tmp_path / "out", extra_features=tmp_path / "kf")
    assert res["test"]["hazard"]["auc"] < 0.7  # a perfect leak would give ~1.0
    names = res["extra_features"]["names"]
    assert names == sorted(c for c in tab.columns if c.startswith("kf_")) and not set(FLAGS) & set(names)


# --------------------------------------------------------------------------- the join
def test_join_puts_the_cut_phase_values_on_slot_k_minus_one(dump, table):
    ex = X.join_split(dump, X.read_table_frame(table), "v2")
    names = sorted(c for c in table.columns if c.startswith("kf_"))
    assert ex.names == names
    ref = planted(table)
    seen = 0
    for m in dump["matches"]:
        tok = ex.tokens[str(m["match_id"])]
        assert tok.dtype == np.float32 and tok.shape == (*np.asarray(m["node_mask"]).shape, len(names))
        for (mid, team, phase), values in ref.items():
            if mid != str(m["match_id"]):
                continue
            i = int(np.flatnonzero(np.asarray(m["team_ids"]) == team)[0])
            np.testing.assert_array_equal(tok[phase - 1, i], values.astype(np.float32))
            seen += 1
    assert seen == len(ref) == ex.report["alive_tokens_cut_slots"]


def test_join_leaves_dead_tokens_and_the_last_slot_empty(dump, table):
    ex = X.join_split(dump, X.read_table_frame(table), "v2")
    event = D.uses_event_alive(dump, "v2")
    alive = {str(m["match_id"]): D.effective_alive(m, "v2", event).numpy() for m in dump["matches"]}
    n_last = 0
    for mid, tok in ex.tokens.items():
        a = alive[mid]
        assert np.isnan(tok[~a]).all()  # junk rows for dead teams are not picked up
        assert np.isnan(tok[-1]).all()  # no cut after the last slot, so no row (the junk 'after the last slot' row is ignored)
        assert not np.isnan(tok[:-1][a[:-1]]).any()
        n_last += int(a[-1].sum())
    assert ex.report["last_slot_alive_tokens"] == n_last > 0


def test_join_counts_the_rows_it_does_not_use(dump, table):
    ex = X.join_split(dump, X.read_table_frame(table), "v2")
    rep = ex.report
    ref = alive_cut_tokens(dump)
    assert rep["alive_tokens_cut_slots"] == len(ref) and rep["target_rows"] == sum(t for *_, t in ref)
    assert rep["table_rows"] == len(table)
    assert rep["table_rows_used"] == (table["note"] == "ok").sum()
    assert rep["table_rows_unused"] == len(table) - rep["table_rows_used"] == (table["note"] != "ok").sum()
    assert rep["table_rows_other_matches"] == (table["match_id"] == "no-such-match").sum() == 1


def test_join_gap_on_a_target_row_fails_and_lists_it(dump):
    ref = alive_cut_tokens(dump)
    mid, p, i, _ = next(r for r in ref if r[3])
    m = next(mm for mm in dump["matches"] if str(mm["match_id"]) == mid)
    team = int(m["team_ids"][i])
    t = make_extra_table(dump, seed=1, drop=lambda a, b, c: (a, b, c) == (mid, team, p + 1))
    with pytest.raises(X.ExtraCoverageError) as e:
        X.join_split(dump, X.read_table_frame(t), "v2", split="train")
    msg = str(e.value)
    assert mid in msg and str(team) in msg and str(p + 1) in msg and "target" in msg and "train" in msg
    assert len(e.value.gaps) == 1
    gap = e.value.gaps.iloc[0]
    assert (gap["match_id"], int(gap["team_id"]), int(gap["phase"]), gap["kind"]) == (mid, team, p + 1, "target row")


def test_join_gap_on_an_alive_token_that_is_not_a_target_fails_too(dump):
    ref = alive_cut_tokens(dump)
    mid, p, i, _ = next(r for r in ref if not r[3])  # alive at the start of slot p, no risk-set row at cut p
    m = next(mm for mm in dump["matches"] if str(mm["match_id"]) == mid)
    team = int(m["team_ids"][i])
    t = make_extra_table(dump, seed=1, drop=lambda a, b, c: (a, b, c) == (mid, team, p + 1))
    with pytest.raises(X.ExtraCoverageError) as e:
        X.join_split(dump, X.read_table_frame(t), "v2")
    assert list(e.value.gaps["kind"]) == ["alive token"]
    assert "alive token" in str(e.value)


def test_join_lists_every_gap_but_prints_a_bounded_number(dump):
    t = make_extra_table(dump, seed=1, drop=lambda a, b, c: c == 1)  # the whole first cut is missing
    n_first_cut = sum(1 for r in alive_cut_tokens(dump) if r[1] == 0)
    with pytest.raises(X.ExtraCoverageError) as e:
        X.join_split(dump, X.read_table_frame(t), "v2")
    assert len(e.value.gaps) == n_first_cut
    assert str(n_first_cut) in str(e.value)
    assert str(e.value).count("\n") <= X.MAX_LISTED_GAPS + 4


def test_join_nan_cells_are_kept_as_missing_values_not_gaps(dump):
    t = make_extra_table(dump, seed=1, nan_frac=0.3)
    ex = X.join_split(dump, X.read_table_frame(t), "v2")
    assert ex.report["nan_cells"] > 0
    cut_alive = [a for a in alive_cut_tokens(dump)]
    n_nan = sum(int(np.isnan(ex.tokens[mid][p, i]).sum()) for mid, p, i, _ in cut_alive)
    assert n_nan == ex.report["nan_cells"] and ex.report["cells"] == len(cut_alive) * len(ex.names)


def test_join_turns_infinite_values_into_missing_ones_and_counts_them(dump):
    t = make_extra_table(dump, seed=1)
    t.loc[t.index[0], "kf_1"] = np.inf
    t.loc[t.index[1], "kf_2"] = -np.inf
    ex = X.join_split(dump, X.read_table_frame(t), "v2")
    assert ex.report["inf_cells"] == 2 and all(np.isfinite(a[~np.isnan(a)]).all() for a in ex.tokens.values())


def test_join_does_not_depend_on_the_column_order(dump):
    a = X.join_split(dump, X.read_table_frame(make_extra_table(dump, seed=4)), "v2")
    b = X.join_split(dump, X.read_table_frame(make_extra_table(dump, seed=4, shuffle_columns=True)), "v2")
    assert a.names == b.names
    for mid in a.tokens:
        np.testing.assert_array_equal(a.tokens[mid], b.tokens[mid])


def test_join_uses_the_alive_rule_of_the_label_choice(dump):
    v2_table = make_extra_table(dump, seed=1, labels="v2")
    X.join_split(dump, X.read_table_frame(v2_table), "v2")  # covers the event-based alive tokens
    # the pipeline alive rule (node_mask | target) reads ghost tokens as alive: a v2 table has no rows for them
    with pytest.raises(X.ExtraCoverageError) as e:
        X.join_split(dump, X.read_table_frame(v2_table), "pipeline")
    assert len(e.value.gaps) > 0 and "alive token" in set(e.value.gaps["kind"])  # the ghost tokens
    X.join_split(dump, X.read_table_frame(make_extra_table(dump, seed=1, labels="pipeline")), "pipeline")


def test_join_of_a_subsampled_dump_only_needs_its_own_matches(dump, table):
    sub, _ = D.subsample_dump(dump, 3, seed=7)
    assert len(sub["matches"]) == 3
    ex = X.join_split(sub, X.read_table_frame(table), "v2")
    assert set(ex.tokens) == {str(m["match_id"]) for m in sub["matches"]}
    assert ex.report["table_rows_other_matches"] == (~table["match_id"].isin(set(ex.tokens))).sum()


# --------------------------------------------------------------------------- alignment with our labels
def test_alignment_agreement_is_reported_when_the_table_carries_the_label_columns(dump, table):
    rep = X.join_split(dump, X.read_table_frame(table), "v2").report["alignment"]
    assert rep["checked"] and rep["rows_compared"] == (table["note"] == "ok").sum()
    assert rep["risk_set_mismatch"] == rep["label_mismatch"] == rep["table_risk_rows_not_ours"] == 0
    assert rep["target_rows_compared"] == (table["in_risk_set"] == 1).sum()


def test_alignment_is_skipped_without_the_label_columns_and_for_pipeline_labels(dump):
    t = make_extra_table(dump, seed=1, label_columns=False)
    assert X.join_split(dump, X.read_table_frame(t), "v2").report["alignment"] == {"checked": False, "reason": "no label columns"}
    t = make_extra_table(dump, seed=1, labels="pipeline")
    assert X.join_split(dump, X.read_table_frame(t), "pipeline").report["alignment"]["reason"] == "labels=pipeline"


def test_alignment_catches_a_shifted_phase_axis(dump, table):
    shifted = table.copy()
    shifted["phase"] = shifted["phase"] + 1  # a table whose cut phase is one off: covers a different set of tokens
    with pytest.raises(ValueError):  # coverage gaps and/or an alignment mismatch, either way it must not pass
        X.join_split(dump, X.read_table_frame(shifted), "v2")


def test_alignment_catches_a_risk_set_flag_that_disagrees(dump, table):
    t = table.copy()
    row = t[(t["note"] == "ok") & (t["in_risk_set"] == 1)].index[0]
    t.loc[row, "in_risk_set"] = 0
    with pytest.raises(X.ExtraAlignmentError) as e:
        X.join_split(dump, X.read_table_frame(t), "v2", split="val")
    assert t.loc[row, "match_id"] in str(e.value) and "in_risk_set" in str(e.value) and "val" in str(e.value)


def test_alignment_catches_a_hazard_label_that_disagrees(dump, table):
    t = table.copy()
    row = t[(t["note"] == "ok") & (t["in_risk_set"] == 1)].index[0]
    t.loc[row, "y_hazard_v2"] = 1.0 - t.loc[row, "y_hazard_v2"]
    with pytest.raises(X.ExtraAlignmentError, match="y_hazard_v2"):
        X.join_split(dump, X.read_table_frame(t), "v2")


def test_alignment_catches_risk_rows_of_the_table_that_are_not_ours(dump, table):
    t = table.copy()
    row = t[t["note"] == "dead team"].index[0]  # the table calls a token at risk that we call dead
    t.loc[row, "in_risk_set"] = 1
    t.loc[row, "y_hazard_v2"] = 0.0
    with pytest.raises(X.ExtraAlignmentError, match="not in our risk set|table_risk"):
        X.join_split(dump, X.read_table_frame(t), "v2")


# --------------------------------------------------------------------------- loading a cohort
def test_load_extra_joins_every_split_and_checks_the_feature_sets_agree(cohort_dir, extra_dir):
    dumps = D.load_cohort(cohort_dir, "c")
    joined = X.load_extra(extra_dir, "c", dumps, "v2")
    assert set(joined) == {"train", "val", "test"}
    names = joined["train"].names
    assert names == sorted(names) and len(names) == 5 and all(j.names == names for j in joined.values())
    assert joined["val"].report["split"] == "val" and joined["val"].report["file"].endswith("val.csv.gz")
    # a val table with another feature set is refused
    other = make_extra_table(dumps["val"], seed=9, n_features=4)
    other.to_csv(extra_dir / "c" / "val.csv.gz", index=False)
    with pytest.raises(ValueError, match="feature"):
        X.load_extra(extra_dir, "c", dumps, "v2")


def test_load_extra_missing_table_names_the_file(cohort_dir, extra_dir):
    (extra_dir / "c" / "test.csv.gz").unlink()
    with pytest.raises(FileNotFoundError, match="test"):
        X.load_extra(extra_dir, "c", D.load_cohort(cohort_dir, "c"), "v2")


# --------------------------------------------------------------------------- standardiser
def test_standardizer_fits_on_the_covered_train_tokens_only_and_ignores_missing_cells(dump):
    t = make_extra_table(dump, seed=3, nan_frac=0.2, n_features=3)
    t["kf_2"] = 7.0  # a constant column
    ex = X.join_split(dump, X.read_table_frame(t), "v2")
    sz = X.ExtraStandardizer.fit(ex, dump, "v2")
    ok = t[t["note"] == "ok"]
    for j, name in enumerate(ex.names):
        col = ok[name].to_numpy(dtype=np.float32).astype(np.float64)
        col = col[np.isfinite(col)]
        if name == "kf_2":
            assert sz.mean[j] == pytest.approx(7.0) and sz.std[j] == 1.0  # constant column: std floor, never a division by 0
        else:
            assert sz.mean[j] == pytest.approx(col.mean(), rel=1e-5, abs=1e-6)
            assert sz.std[j] == pytest.approx(col.std(), rel=1e-5)
    assert sz.constant_columns(ex.names) == ["kf_2"]


def test_standardizer_transform_zero_fills_missing_values_and_round_trips(dump, table):
    t = make_extra_table(dump, seed=3, nan_frac=0.25)
    ex = X.join_split(dump, X.read_table_frame(t), "v2")
    sz = X.ExtraStandardizer.fit(ex, dump, "v2")
    for state in (sz.state_dict(), X.ExtraStandardizer.from_state_dict(sz.state_dict()).state_dict()):
        assert set(state) == set(X.ExtraStandardizer.FIELDS)
    m = dump["matches"][0]
    tok = ex.tokens[str(m["match_id"])]
    z = sz.transform(tok)
    assert z.dtype == torch.float32 and z.shape == tok.shape and torch.isfinite(z).all()
    assert (z[np.isnan(tok)] == 0).all()  # NaN -> 0 after scaling (the train mean)
    ok = ~np.isnan(tok)
    np.testing.assert_allclose(z.numpy()[ok], ((tok - sz.mean) / sz.std)[ok], rtol=1e-5, atol=1e-6)
    z2 = X.ExtraStandardizer.from_state_dict(sz.state_dict()).transform(tok)
    assert torch.equal(z, z2)


# --------------------------------------------------------------------------- the neural dataset
def make_dataset(dump, ex=None, sz=None, **kw):
    std = D.Standardizer.fit(dump, "v2")
    return D.PTRDataset(dump, std, labels="v2", relation_types=None, extra=ex, extra_standardizer=sz, **kw)


def test_dataset_appends_the_kf_columns_after_x_pos_and_ai(dump, table):
    ex = X.join_split(dump, X.read_table_frame(table), "v2")
    sz = X.ExtraStandardizer.fit(ex, dump, "v2")
    base, union = make_dataset(dump), make_dataset(dump, ex, sz)
    K = len(ex.names)
    assert union.n_features == base.n_features + K
    assert union.feature_names == base.feature_names + ex.names and union.feature_names[-K - 1] == D.AI_FEATURE_NAME
    assert union.extra_names == ex.names and base.extra_names == []
    for i in range(len(dump["matches"])):
        b, u = base[i], union[i]
        assert torch.equal(u["x"][..., : base.n_features], b["x"])  # X, pos and ai are untouched
        np.testing.assert_allclose(u["x"][..., base.n_features :].numpy(), sz.transform(ex.tokens[b["match_id"]]).numpy())
        for key in ("node_mask", "target_mask", "y_hazard", "pos"):
            assert torch.equal(u[key], b[key])


def test_dataset_extra_features_follow_their_cut_slot(dump, table):
    ex = X.join_split(dump, X.read_table_frame(table), "v2")
    sz = X.ExtraStandardizer.fit(ex, dump, "v2")
    ds = make_dataset(dump, ex, sz)
    ref = planted(table)
    m = dump["matches"][0]
    item = ds[0]
    F0 = ds.n_features - len(ex.names)
    checked = 0
    for (mid, team, phase), values in ref.items():
        if mid != str(m["match_id"]):
            continue
        i = int(np.flatnonzero(np.asarray(m["team_ids"]) == team)[0])
        expect = ((values.astype(np.float32) - sz.mean) / sz.std)
        np.testing.assert_allclose(item["x"][phase - 1, i, F0:].numpy(), expect, rtol=1e-4, atol=1e-5)
        checked += 1
    assert checked > 10
    assert (item["x"][-1, :, F0:] == 0).all()  # the last slot has no cut: the train mean (0 after z-scoring)


def test_dataset_without_extra_features_is_what_it_was(dump):
    ds = make_dataset(dump)
    assert ds.feature_names[-1] == D.AI_FEATURE_NAME and ds.extra_names == []
    assert ds.n_features == len(ds.keep) + 2 + 1


def test_dataset_with_extra_requires_a_standardizer_and_every_match(dump, table):
    ex = X.join_split(dump, X.read_table_frame(table), "v2")
    with pytest.raises(ValueError, match="standardi"):
        make_dataset(dump, ex, None)
    sz = X.ExtraStandardizer.fit(ex, dump, "v2")
    broken = X.ExtraSplit(ex.names, {k: v for k, v in list(ex.tokens.items())[1:]}, ex.report)
    with pytest.raises(ValueError, match="no extra tokens|match"):
        make_dataset(dump, broken, sz)


def test_batches_carry_the_extra_columns_through_collate(dump, table):
    ex = X.join_split(dump, X.read_table_frame(table), "v2")
    ds = make_dataset(dump, ex, X.ExtraStandardizer.fit(ex, dump, "v2"))
    batch = ds.get_batch([0, 1, 2])
    assert batch["x"].shape[-1] == ds.n_features and torch.isfinite(batch["x"]).all()


# --------------------------------------------------------------------------- HGB rows
def test_hgb_union_rows_are_the_hgb_cut_rows_plus_the_kf_columns(dump, table):
    ex = X.join_split(dump, X.read_table_frame(table), "v2")
    cut = H.build_rows(dump, "HGB-cut", "v2")
    uni = H.build_rows(dump, "HGB-union", "v2", extra=ex)
    K = len(ex.names)
    assert uni.feature_names == cut.feature_names + ex.names and uni.X.shape == (cut.X.shape[0], cut.X.shape[1] + K)
    pd.testing.assert_frame_equal(uni.keys, cut.keys)
    np.testing.assert_array_equal(uni.y, cut.y)
    np.testing.assert_array_equal(uni.X[:, : cut.X.shape[1]], cut.X)  # NaN-equal
    ref = planted(table)
    for (mid, team, phase), row in zip(zip(uni.keys["match_id"], uni.keys["team_id"], uni.keys["phase"]), uni.X[:, cut.X.shape[1] :]):
        np.testing.assert_array_equal(row, ref[(mid, int(team), int(phase))].astype(np.float32))


def test_hgb_union_keeps_nan_cells_for_the_trees(dump):
    t = make_extra_table(dump, seed=2, nan_frac=0.3)
    ex = X.join_split(dump, X.read_table_frame(t), "v2")
    uni = H.build_rows(dump, "HGB-union", "v2", extra=ex)
    assert np.isnan(uni.X[:, -len(ex.names) :]).any()


def test_hgb_variants_and_extra_arguments_are_strict(dump, table):
    ex = X.join_split(dump, X.read_table_frame(table), "v2")
    assert H.HGB_VARIANTS == ("HGB-cut", "HGB-lag", "HGB-noHab")  # the main matrix is not widened
    assert H.UNION_VARIANTS == ("HGB-union",)
    with pytest.raises(ValueError, match="extra"):
        H.build_rows(dump, "HGB-union", "v2")
    with pytest.raises(ValueError, match="extra"):
        H.build_rows(dump, "HGB-cut", "v2", extra=ex)


# --------------------------------------------------------------------------- command lines
def hgb_cli(root, out, variant, *more, cohort="c", seed=42):
    return ["--dump-dir", str(root), "--cohort", cohort, "--variant", variant, "--seed", str(seed), "--out-dir", str(out), *more]


def train_cli(root, out, arm, *more, cohort="c", seed=42, epochs=1):
    return [
        "--dump-dir", str(root), "--cohort", cohort, "--arm", arm, "--seed", str(seed), "--out-dir", str(out),
        "--device", "cpu", "--max-epochs", str(epochs), *more,
    ]  # fmt: skip


def test_train_cli_requires_the_extra_dir_for_the_union_arm_and_refuses_it_otherwise(cohort_dir, extra_dir, tmp_path):
    with pytest.raises(SystemExit):
        T.parse_args(train_cli(cohort_dir, tmp_path, "M11R-union"))
    with pytest.raises(SystemExit):
        T.parse_args(train_cli(cohort_dir, tmp_path, "M11R", "--extra-features", str(extra_dir)))
    args = T.parse_args(train_cli(cohort_dir, tmp_path, "M11R-union", "--extra-features", str(extra_dir)))
    assert args.extra_features == str(extra_dir)
    assert T.parse_args(train_cli(cohort_dir, tmp_path, "M11R")).extra_features is None


def test_hgb_cli_requires_the_extra_dir_for_the_union_variant_and_refuses_it_otherwise(cohort_dir, extra_dir, tmp_path):
    with pytest.raises(SystemExit):
        H.parse_args(hgb_cli(cohort_dir, tmp_path, "HGB-union"))
    with pytest.raises(SystemExit):
        H.parse_args(hgb_cli(cohort_dir, tmp_path, "HGB-cut", "--extra-features", str(extra_dir)))
    assert H.parse_args(hgb_cli(cohort_dir, tmp_path, "HGB-union", "--extra-features", str(extra_dir))).variant == "HGB-union"


def test_neural_config_hash_covers_the_extra_set_and_the_files_but_only_for_union_arms(cohort_dir, extra_dir, tmp_path):
    plain = T.config_spec(T.parse_args(train_cli(cohort_dir, tmp_path, "M11R")))
    assert "extra_features" not in plain  # the hash of every other run is what it was
    args = T.parse_args(train_cli(cohort_dir, tmp_path, "M11R-union", "--extra-features", str(extra_dir)))
    spec = T.config_spec(args)["extra_features"]
    names = X.table_feature_names(extra_dir / "c" / "train.csv.gz")
    assert spec["names_sha1"] == X.names_sha1(names) and spec["n_features"] == len(names) == 5
    assert set(spec["files"]) == {"train", "val", "test"} and all(len(v) == 2 for v in spec["files"].values())
    h0 = T.config_hash(args)
    assert h0 == T.config_hash(T.parse_args(train_cli(cohort_dir, tmp_path, "M11R-union", "--extra-features", str(extra_dir))))
    p = extra_dir / "c" / "val.csv.gz"
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))  # the table was rewritten
    assert T.config_hash(args) != h0
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert T.config_hash(args) == h0
    # another feature set: another hash
    write_extra_tables(cohort_dir, "c", extra_dir.parent / "kf6", seed=2, n_features=6)
    other = T.parse_args(train_cli(cohort_dir, tmp_path, "M11R-union", "--extra-features", str(extra_dir.parent / "kf6")))
    assert T.config_hash(other) != h0


def test_config_hash_is_total_when_the_tables_do_not_exist_yet(cohort_dir, tmp_path):
    args = T.parse_args(train_cli(cohort_dir, tmp_path, "M11R-union", "--extra-features", str(tmp_path / "nowhere")))
    spec = T.config_spec(args)["extra_features"]
    assert spec["names_sha1"] is None and spec["files"] == {"train": None, "val": None, "test": None}
    assert len(T.config_hash(args)) == 16  # the runner can plan (dry-run) before the tables exist


def test_hgb_config_hash_covers_the_extra_set_and_the_files_but_only_for_union(cohort_dir, extra_dir, tmp_path):
    old_spec = {
        "kind": "hgb", "variant": "HGB-cut", "cohort": "c", "seed": 42, "labels": "v2", "match_types": ["official", "competitive"],
        "dump": dump_fingerprint(cohort_dir, "c"), "code": code_fingerprint(),
    }  # fmt: skip
    assert H.config_hash(H.parse_args(hgb_cli(cohort_dir, tmp_path, "HGB-cut"))) == util_hash(old_spec)  # pinned: unchanged
    a = H.parse_args(hgb_cli(cohort_dir, tmp_path, "HGB-union", "--extra-features", str(extra_dir)))
    h0 = H.config_hash(a)
    assert h0 != H.config_hash(H.parse_args(hgb_cli(cohort_dir, tmp_path, "HGB-cut")))
    p = extra_dir / "c" / "train.csv.gz"
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    assert H.config_hash(a) != h0


def test_code_fingerprint_covers_the_extra_module():
    from model_stgcn.ptr.util import RESULT_MODULES

    assert "extra" in RESULT_MODULES


def test_hgb_union_run_writes_the_extra_block_and_fails_on_a_gap(cohort_dir, extra_dir, tmp_path):
    out = tmp_path / "out"
    assert H.main(hgb_cli(cohort_dir, out, "HGB-union", "--extra-features", str(extra_dir))) == 0
    d = out / "c" / "HGB-union" / "seed42"
    res = json.loads((d / "metrics.json").read_text())
    ex = res["extra_features"]
    assert ex["n_features"] == 5 and ex["names"] == X.table_feature_names(extra_dir / "c" / "train.csv.gz")
    assert ex["names_sha1"] == X.names_sha1(ex["names"])
    assert set(ex["join"]) == {"train", "val", "test"} and ex["join"]["test"]["alive_tokens_cut_slots"] > 0
    assert ex["join"]["train"]["alignment"]["checked"] and ex["join"]["train"]["alignment"]["risk_set_mismatch"] == 0
    assert res["n_features"] == 24 + 1 + 5  # HGB-cut columns + 5 kf
    assert res["config_hash"] == H.config_hash(H.parse_args(hgb_cli(cohort_dir, out, "HGB-union", "--extra-features", str(extra_dir))))
    # rows are the HGB-cut rows
    H.main(hgb_cli(cohort_dir, out, "HGB-cut"))
    a, b = (pd.read_csv(out / "c" / v / "seed42" / "pred_test.csv") for v in ("HGB-union", "HGB-cut"))
    pd.testing.assert_frame_equal(a[KEYS + ["y"]], b[KEYS + ["y"]])
    # a gap: fail before anything is fitted or written, and list it
    t = pd.read_csv(extra_dir / "c" / "test.csv.gz", dtype={"match_id": str})
    gone = t[t["note"] == "ok"].iloc[5]
    t.drop(index=gone.name).to_csv(extra_dir / "c" / "test.csv.gz", index=False)
    out2 = tmp_path / "out2"
    with pytest.raises(X.ExtraCoverageError, match=str(gone["match_id"])):
        H.main(hgb_cli(cohort_dir, out2, "HGB-union", "--extra-features", str(extra_dir)))
    assert not (out2 / "c" / "HGB-union" / "seed42" / "metrics.json").exists()


def test_neural_union_run_trains_on_the_extended_tokens_and_records_the_set(cohort_dir, extra_dir, tmp_path):
    out = tmp_path / "out"
    T.main(train_cli(cohort_dir, out, "M11R-union", "--extra-features", str(extra_dir), *SMALL))
    T.main(train_cli(cohort_dir, out, "M11R", *SMALL))
    d, base = out / "c" / "M11R-union" / "seed42", out / "c" / "M11R" / "seed42"
    res, ref = json.loads((d / "metrics.json").read_text()), json.loads((base / "metrics.json").read_text())
    assert res["n_input_features"] == ref["n_input_features"] + 5 and res["config"]["in_features"] == ref["config"]["in_features"] + 5
    ex = res["extra_features"]
    assert ex["names_sha1"] == X.names_sha1(ex["names"]) and ex["n_features"] == 5
    assert ex["join"]["test"]["alignment"]["checked"] and ex["standardizer"]["constant_train_columns"] == []
    assert res["arm_spec"]["extra"] is True and ref["arm_spec"]["extra"] is False and "extra_features" not in ref
    assert res["config_hash"] == T.config_hash(T.parse_args(train_cli(cohort_dir, out, "M11R-union", "--extra-features", str(extra_dir), *SMALL)))
    # same risk-set rows as the plain arm, and a finite prediction for each
    a, b = (pd.read_csv(x / "pred_test.csv") for x in (d, base))
    pd.testing.assert_frame_equal(a[KEYS + ["y"]], b[KEYS + ["y"]])
    assert np.isfinite(a["p_hazard"]).all()
    ckpt = torch.load(d / "best.pt", weights_only=False)
    assert ckpt["extra_feature_names"] == ex["names"] and set(ckpt["extra_standardizer"]) == set(X.ExtraStandardizer.FIELDS)
    assert ckpt["feature_names"][-5:] == ex["names"]


def test_neural_union_run_fails_on_a_gap_before_training(cohort_dir, extra_dir, tmp_path):
    t = pd.read_csv(extra_dir / "c" / "train.csv.gz", dtype={"match_id": str})
    gone = t[t["note"] == "ok"].iloc[3]
    t.drop(index=gone.name).to_csv(extra_dir / "c" / "train.csv.gz", index=False)
    out = tmp_path / "out"
    with pytest.raises(X.ExtraCoverageError) as e:
        T.main(train_cli(cohort_dir, out, "M11R-union", "--extra-features", str(extra_dir), *SMALL))
    assert str(gone["match_id"]) in str(e.value) and "train" in str(e.value)
    assert not (out / "c" / "M11R-union").exists()


def test_prepare_fits_the_extra_standardizer_on_the_train_table_only(cohort_dir, extra_dir, tmp_path):
    prep = T.prepare(T.parse_args(train_cli(cohort_dir, tmp_path, "M11R-union", "--extra-features", str(extra_dir))))
    tab = pd.read_csv(extra_dir / "c" / "train.csv.gz", dtype={"match_id": str})
    ok = tab[tab["note"] == "ok"]
    names = prep.extra_info["names"]
    np.testing.assert_allclose(
        prep.extra_standardizer.mean, ok[names].to_numpy(np.float32).astype(np.float64).mean(0), rtol=1e-5, atol=1e-6
    )
    assert all(ds.extra_names == names for ds in prep.datasets.values())
    assert prep.config.in_features == prep.datasets["train"].n_features == 24 + 2 + 1 + 5


def test_prepare_with_a_train_subsample_joins_and_standardises_the_subsample_only(cohort_dir, extra_dir, tmp_path):
    prep = T.prepare(T.parse_args(train_cli(cohort_dir, tmp_path, "M11R-union", "--extra-features", str(extra_dir), "--train-subsample", "4")))
    kept = {str(m["match_id"]) for m in prep.datasets["train"].matches}
    assert len(kept) == 4 and prep.extra_info["join"]["train"]["n_matches"] == 4
    tab = pd.read_csv(extra_dir / "c" / "train.csv.gz", dtype={"match_id": str})
    ok = tab[(tab["note"] == "ok") & tab["match_id"].isin(kept)]
    names = prep.extra_info["names"]
    np.testing.assert_allclose(
        prep.extra_standardizer.mean, ok[names].to_numpy(np.float32).astype(np.float64).mean(0), rtol=1e-5, atol=1e-6
    )
    assert prep.extra_info["join"]["val"]["n_matches"] == 5  # val and test are not subsampled


def test_last_slot_tokens_have_no_cut_and_cannot_move_any_prediction(cohort_dir, extra_dir, tmp_path):
    """The only alive tokens without a table row are those of the last slot (no cut after it); whatever they carry, the
    outputs of the cut slots do not change (phase axis is causal), and the last slot itself has no targets."""
    from model_stgcn.ptr.model import PTRFormer

    prep = T.prepare(T.parse_args(train_cli(cohort_dir, tmp_path, "M11R-union", "--extra-features", str(extra_dir), *SMALL)))
    ds = prep.datasets["test"]
    batch = ds.get_batch([0, 1, 2])
    assert not batch["target_mask"][:, -1].any()
    model = PTRFormer(prep.config, init_seed=3).eval()
    with torch.no_grad():
        base = model(batch)
        pert = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        F0 = ds.n_features - len(ds.extra_names)
        pert["x"][:, -1, :, F0:] = torch.randn_like(pert["x"][:, -1, :, F0:]) * 50  # what a row of the last cut would hold
        out = model(pert)
    for key in ("hazard_logit", "rank_score"):
        torch.testing.assert_close(out[key][:, :-1], base[key][:, :-1], atol=1e-5, rtol=1e-5)
    assert prep.extra_info["join"]["test"]["last_slot_alive_tokens"] > 0


# --------------------------------------------------------------------------- the cue reaches the model (end to end)
@pytest.fixture(scope="module")
def cue_dirs(tmp_path_factory):
    """Dumps without any cue in X; the only cue is ``kf_signal`` in the extra table."""
    root = tmp_path_factory.mktemp("cue")
    write_cohort_dumps(root, "u", sizes={"train": 40, "val": 10, "test": 14}, seed=4, signal=0.0, n_features=24)
    write_extra_tables(root, "u", root / "kf", seed=6, signal=3.0, n_features=4)
    return root


def test_hgb_union_uses_a_cue_that_only_the_extra_features_carry(cue_dirs, tmp_path):
    out = tmp_path / "out"
    r_cut = H.run(cue_dirs, "u", "HGB-cut", 42, out)
    r_uni = H.run(cue_dirs, "u", "HGB-union", 42, out, extra_features=cue_dirs / "kf")
    assert r_uni["test"]["hazard"]["auc"] > 0.9 > 0.7 > r_cut["test"]["hazard"]["auc"]


def test_neural_union_uses_a_cue_that_only_the_extra_features_carry(cue_dirs, tmp_path):
    out = tmp_path / "out"
    args = [*SMALL]
    T.main(train_cli(cue_dirs, out, "M11R", *args, cohort="u", epochs=8))
    T.main(train_cli(cue_dirs, out, "M11R-union", "--extra-features", str(cue_dirs / "kf"), *args, cohort="u", epochs=8))
    auc = {a: json.loads((out / "u" / a / "seed42" / "metrics.json").read_text())["test"]["hazard"]["auc"] for a in ("M11R", "M11R-union")}
    assert auc["M11R-union"] > 0.85 and auc["M11R"] < auc["M11R-union"] - 0.1
