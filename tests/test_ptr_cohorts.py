"""Cohort discovery from the dump directory (design doc v2.3).

The dump directory is the registry: airoyale matches, Vikendi and cohorts with < 200 matches are simply not
dumped, so they are not listed and nothing hard-codes the 13 pre-registered cohort names any more.
"""

from __future__ import annotations

import json

import pytest

from model_stgcn.ptr import cohorts as C
from model_stgcn.ptr import data as D
from model_stgcn.ptr.synthetic import write_cohort_dumps

SMALL = {"train": 3, "val": 2, "test": 2}


def make_root(tmp_path, names):
    for k, name in enumerate(names):
        write_cohort_dumps(tmp_path, name, sizes=SMALL, seed=k)
    return tmp_path


def test_discover_lists_complete_cohort_directories_sorted_and_ignores_the_rest(tmp_path):
    root = make_root(tmp_path, ["taego_tpp_squad", "erangel_tpp_squad", "erangel_tpp_solo"])
    (root / "logs").mkdir()  # a directory without split files is not a cohort
    (root / "notes.txt").write_text("not a cohort")
    for scratch in (".backup_taego", "_scratch"):  # hidden / underscore directories are never cohorts,
        write_cohort_dumps(root, scratch, sizes=SMALL, seed=9)  # even when they hold complete dumps
    assert C.discover_cohorts(root) == ["erangel_tpp_solo", "erangel_tpp_squad", "taego_tpp_squad"]
    assert C.discover_cohorts(str(root)) == C.discover_cohorts(root)  # str or Path


def test_cohorts_missing_from_the_dump_directory_are_not_listed(tmp_path):
    """Vikendi (and any airoyale-only / < 200-match cohort) was not dumped -> never discovered, never run."""
    root = make_root(tmp_path, ["erangel_tpp_squad", "taego_tpp_squad"])
    found = C.discover_cohorts(root)
    assert "vikendi_tpp_squad" not in found and found == ["erangel_tpp_squad", "taego_tpp_squad"]
    with pytest.raises(FileNotFoundError, match="vikendi_tpp_squad"):  # ... and asking for it fails clearly
        D.load_cohort(root, "vikendi_tpp_squad")


def test_incomplete_cohorts_are_excluded_but_reported(tmp_path):
    root = make_root(tmp_path, ["a_full", "b_partial"])
    (root / "b_partial" / "test.pt").unlink()
    (root / "c_only_val").mkdir()
    (root / "c_only_val" / "val.pt").write_bytes(b"x")
    assert C.discover_cohorts(root) == ["a_full"]
    assert C.scan_cohorts(root) == {"a_full": [], "b_partial": ["test"], "c_only_val": ["train", "test"]}
    with pytest.raises(ValueError, match=r"b_partial.*test|c_only_val"):
        C.discover_cohorts(root, strict=True)
    assert C.discover_cohorts(make_root(tmp_path / "ok", ["x"]), strict=True) == ["x"]


def test_empty_and_missing_dump_directories(tmp_path):
    assert C.discover_cohorts(tmp_path) == []
    with pytest.raises(FileNotFoundError, match="no_such_dir"):
        C.discover_cohorts(tmp_path / "no_such_dir")


def test_every_discovered_cohort_loads(tmp_path):
    root = make_root(tmp_path, ["erangel_tpp_squad", "miramar_tpp_squad"])
    for name in C.discover_cohorts(root):
        splits = D.load_cohort(root, name)
        assert [len(splits[s]["matches"]) for s in ("train", "val", "test")] == [3, 2, 2]


def test_report_reads_the_dump_manifest_when_present(tmp_path):
    root = make_root(tmp_path, ["erangel_tpp_squad", "taego_tpp_squad"])
    manifest = json.loads((root / "erangel_tpp_squad" / "dump_manifest.json").read_text())
    assert {s: manifest["splits"][s]["n_dumped"] for s in SMALL} == SMALL  # same layout as the dump script
    (root / "taego_tpp_squad" / "dump_manifest.json").unlink()
    rep = {r["cohort"]: r for r in C.cohort_report(root)}
    assert set(rep) == {"erangel_tpp_squad", "taego_tpp_squad"}
    assert rep["erangel_tpp_squad"]["n_matches"] == SMALL
    assert rep["taego_tpp_squad"]["n_matches"] is None  # no manifest: the files are still discovered
    assert rep["erangel_tpp_squad"]["complete"] and rep["taego_tpp_squad"]["complete"]
    assert rep["taego_tpp_squad"]["missing_splits"] == []


def test_cli_prints_one_cohort_per_line_or_json(tmp_path, capsys):
    root = make_root(tmp_path, ["erangel_tpp_squad", "taego_tpp_squad"])
    assert C.main(["--dump-dir", str(root)]) == 0
    assert capsys.readouterr().out.split() == ["erangel_tpp_squad", "taego_tpp_squad"]
    assert C.main(["--dump-dir", str(root), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["cohort"] for r in rows] == ["erangel_tpp_squad", "taego_tpp_squad"]
    (root / "taego_tpp_squad" / "val.pt").unlink()
    assert C.main(["--dump-dir", str(root), "--strict"]) == 1  # incomplete cohort + --strict -> non-zero
    assert "taego_tpp_squad" in capsys.readouterr().err
    assert C.main(["--dump-dir", str(root)]) == 0  # default: skip it and say so on stderr
    captured = capsys.readouterr()
    assert captured.out.split() == ["erangel_tpp_squad"] and "taego_tpp_squad" in captured.err


def test_training_cli_error_for_an_unknown_cohort_lists_the_available_ones(tmp_path):
    from model_stgcn.ptr import train as T

    root = make_root(tmp_path, ["erangel_tpp_squad"])
    argv = [
        "--dump-dir", str(root), "--cohort", "vikendi_tpp_squad", "--arm", "M00",
        "--out-dir", str(tmp_path / "out"), "--max-epochs", "1",
    ]  # fmt: skip
    with pytest.raises(FileNotFoundError) as exc:
        T.main(argv)
    assert "vikendi_tpp_squad" in str(exc.value) and "erangel_tpp_squad" in str(exc.value)
