"""``paper/bigcomp2027/tables/make_paper_tables.py`` on the output of the aggregator (synthetic runs): the explicit directories, the H0 rows, the copies of the
aggregator's decomposition / RQ2 / grid tables, no row for an arm that was not run."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

from model_stgcn.ptr import aggregate as A
from tests.test_ptr_aggregate import (COHORTS, EXPLORATORY, EXTERNAL_QUALITY, EXTERNAL_RANK_QUALITY, NORANK_QUALITY, NORANK_RANK_QUALITY, QUALITY, RANK_QUALITY,
                                      UNION_QUALITY, UNION_RANK_QUALITY, make_runs)  # fmt: skip

SCRIPT = Path(__file__).resolve().parents[1] / "paper" / "bigcomp2027" / "tables" / "make_paper_tables.py"


@pytest.fixture(scope="module")
def mpt():
    spec = importlib.util.spec_from_file_location("make_paper_tables_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_paper_tables_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    root, out = tmp_path_factory.mktemp("runs"), tmp_path_factory.mktemp("results")
    quality = {**{a: q for a, q in QUALITY.items() if a not in EXPLORATORY}, **EXTERNAL_QUALITY, **UNION_QUALITY, **NORANK_QUALITY}
    rank_quality = {**{a: q for a, q in RANK_QUALITY.items() if a in quality}, **EXTERNAL_RANK_QUALITY, **UNION_RANK_QUALITY, **NORANK_RANK_QUALITY}
    sizes = {c: 100 * (i + 1) for i, c in enumerate(COHORTS)}
    make_runs(root, cohorts=COHORTS, quality=quality, rank_quality=rank_quality, skip=set(), seeds={}, n_train=sizes)
    A.aggregate(root, out, n_boot=30, seed=1, workers=1, figures=True)
    return out


def test_the_paper_tables_are_built_from_the_given_results_directory(mpt, results, tmp_path, capsys):
    paper = tmp_path / "paper"
    assert mpt.main(["--results-dir", str(results), "--paper-dir", str(paper)])== 0
    tables = {p.name for p in (paper / "tables").glob("*.tex")}
    assert {"tab_primary.tex", "tab_arms.tex", "tab_external_short.tex", "tab_metareg.tex", "tab_decomp.tex", "tab_rq2_external.tex", "tab_grid.tex"} <= tables
    for name in ("tab_decomp.tex", "tab_rq2_external.tex", "tab_grid.tex"):  # copied verbatim from the aggregator
        assert (paper / "tables" / name).read_text() == (results / "tables" / name).read_text(), name
    assert "Rank auxiliary loss" in (paper / "tables" / "tab_decomp.tex").read_text()
    assert (paper / "figures" / "forest_primary.pdf").is_file() and not (paper / "figures" / "learning_curve.pdf").exists()  # the learning curve is optional
    err = capsys.readouterr().err
    assert "learning_curve.pdf not found" in err
    short = (paper / "tables" / "tab_external_short.tex").read_text()
    assert "\\todo" not in short and "KRAFTON-tf+H0" in short and "K2+H0" in short and "KRAFTON-tf-ourfeat (K2)" in short and "SAGONG-rf" in short
    arms = (paper / "tables" / "tab_arms.tex").read_text()
    assert "STAGE2" not in arms and "M11RZ-P" not in arms and "M11R-noHab" in arms and "HGB-noHab" in arms  # stage 2 is not part of the v2 re-run
    assert "<results-dir>" in (paper / "tables" / "tab_primary.tex").read_text() and "reports/bigcomp2027_results/" not in (paper / "tables" / "tab_primary.tex").read_text()
    # a table the aggregator did not write leaves no stale copy in the paper directory
    other = tmp_path / "results2"
    shutil.copytree(results, other)
    (other / "tables" / "tab_rq2_external.tex").unlink()
    assert mpt.main(["--results-dir", str(other), "--paper-dir", str(paper)])== 0
    assert not (paper / "tables" / "tab_rq2_external.tex").exists() and (paper / "tables" / "tab_decomp.tex").is_file()
