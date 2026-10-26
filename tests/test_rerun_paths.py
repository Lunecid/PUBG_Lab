"""The v2 roots in one place and the guard against mixing v1 artifacts with a v2 dump (``model_stgcn.ptr.paths``), and its use by the entry points of the
re-run: aggregate, supplement, run_matrix (see also tests/test_ptr_run_matrix.py), the H0 modules, run_all and make_paper_tables.  DB-free."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import pubg_lab.survival.ptr.data.paths as P
from pubg_lab.survival.ptr.data.synthetic import write_cohort_dumps

KT = Path(__file__).resolve().parents[1] / "third_party" / "krafton_pubg_winprob"
if str(KT) not in sys.path:
    sys.path.insert(0, str(KT))


# --------------------------------------------------------------------------- helpers
def make_dump(root: Path, *, version: str | None = None, cohorts=("c_alpha", "c_beta"), sizes=None) -> Path:
    """A synthetic dump; ``version='v2'`` turns it into a valid v2 dump (version in the manifests and metas, the v2 record keys, no ``node_mask_v2``)."""
    sizes = sizes or {"train": 5, "val": 3, "test": 3}
    for c in cohorts:
        write_cohort_dumps(root, c, sizes=sizes, seed=1)
    if version is None:
        return root
    pop = {"rules": {}, "cohorts": {}}
    for c in cohorts:
        man_path = root / c / "dump_manifest.json"
        man = json.loads(man_path.read_text())
        man["version"] = version
        for s in sizes:
            man["splits"][s]["n_skipped"] = 0
        man_path.write_text(json.dumps(man))
        for s in sizes:
            path = root / c / f"{s}.pt"
            d = torch.load(path, weights_only=False)
            d["meta"].update({"version": version, "n_phases": 8, "n_features": len(d["meta"]["feature_names"]), "single_clock": True})
            for m in d["matches"]:
                m.pop("node_mask_v2", None)
                target = torch.as_tensor(m["target_mask_v2"]).bool().clone()
                target[-1] = False
                m["target_mask_v2"], m["target_mask"] = target, target.clone()
                m["y_hazard_v2"] = torch.where(target, torch.as_tensor(m["y_hazard_v2"]).float(), torch.zeros(()))
                m["y_hazard"] = m["y_hazard_v2"].clone()
                node = torch.as_tensor(m["node_mask"]).bool()
                m["query_mask"] = node | target
                end = np.asarray(m["phase_end_s"], dtype=np.float64)
                m["cut_t"] = torch.as_tensor(end - 10.0)
                valid = np.ones(len(end), dtype=bool)
                valid[-1] = False
                m["cut_valid"], m["cut_mask"] = torch.as_tensor(valid), torch.as_tensor(valid.copy())
                won = torch.as_tensor(m["team_won"]).bool()
                death = torch.as_tensor(m["team_death_s"]).double().clone()
                death[won] = float("inf")
                m["team_death_s"] = death
            torch.save(d, path)
        pop["cohorts"][c] = {"status": "kept", "before": {s: sizes[s] for s in sizes} | {"total": sum(sizes.values())},
                             "after": {s: sizes[s] for s in sizes} | {"total": sum(sizes.values())}, "match_type_after": {"official": sum(sizes.values())}}
    (root / "population.json").write_text(json.dumps(pop))
    return root


# --------------------------------------------------------------------------- the roots
def test_the_v2_roots_are_defined_in_one_place():
    d = P.RerunPaths()
    assert d.dump == P.REPO_ROOT / "model_stgcn/dumps/bigcomp2027_phase_v2" and d.runs == P.REPO_ROOT / "model_stgcn/checkpoints/bigcomp2027_ptr_v2"
    assert d.results == P.REPO_ROOT / "reports/bigcomp2027_results_v2" and d.kf == P.REPO_ROOT / "model_stgcn/dumps/bigcomp2027_kraftonfeat_v2"
    assert d.krafton_data == P.REPO_ROOT / "model_stgcn/dumps/krafton_winprob_v2" and d.sagong == P.REPO_ROOT / "model_stgcn/dumps/bigcomp2027_sagong_v2"
    assert d.krafton_reports == P.REPO_ROOT / "reports/krafton_baseline_v2"
    assert d.runs_lc.name == "bigcomp2027_ptr_v2_lc" and d.runs_lc_steps.name == "bigcomp2027_ptr_v2_lc_steps"
    assert d.ecological == d.results / "ecological_model" and d.supplement == d.results / "supplement" and d.logs == d.results / "logs"
    assert d.run_manifest == d.results / "run_manifest.json" and d.paper_numbers == d.results / "paper_numbers.json"
    for name, path in {**d.as_dict(), **{k: str(v) for k, v in d.artifact_roots().items()}}.items():
        assert "_v1" not in path, name  # nothing of a v1 root is a default
    assert set(d.artifact_roots()) == {"kf_table", "krafton_data", "sagong_cache", "runs", "runs_lc", "runs_lc_steps", "results", "krafton_reports"}
    u = P.RerunPaths.under("/x")
    assert u.dump == Path("/x/dumps/bigcomp2027_phase_v2") and u.runs == Path("/x/checkpoints/bigcomp2027_ptr_v2") and u.results == Path("/x/reports/bigcomp2027_results_v2")
    assert u.replace(runs="/y").runs == Path("/y") and u.replace(runs="/y").dump == u.dump


def test_every_entry_point_defaults_to_the_v2_roots():
    import pubg_lab.survival.ptr.evaluation.aggregate as A
    import pubg_lab.survival.ptr.experiments.run_matrix as RM
    import pubg_lab.survival.ptr.evaluation.supplement as S

    d = P.RerunPaths()
    rm = RM.parse_args([])
    assert rm.dump_dir == str(d.dump) and rm.out_dir == str(d.runs) and rm.extra_features is None  # the kf table follows the dump (see below)
    ag = A.parse_args([])
    assert ag.runs_dir == str(d.runs) and ag.out_dir == str(d.results) and ag.dump_dir is None
    sp = S.parse_args([])
    assert sp.runs_dir == str(d.runs) and sp.results_dir == str(d.results)
    import projects.krafton_baseline.training.h0_baseline as H0
    import projects.krafton_baseline.reporting.h0_report as HR
    import projects.krafton_baseline.pipelines.run_all as RA

    b = H0.parse_args([])
    assert b.ckpt_dir == d.runs and b.dump_dir == d.dump and b.data_dir == d.krafton_data
    r = HR.parse_args([])
    assert r.ckpt_dir == d.runs and r.out_dir == d.krafton_reports and r.dump_dir is None
    a = RA.parse_args([])
    assert a.out_dir == d.runs and a.data_dir == d.krafton_data and a.dump_dir == d.dump and a.table_dir == d.kf
    import importlib.util

    spec = importlib.util.spec_from_file_location("mpt", Path(__file__).resolve().parents[1] / "projects/battle_royale/reporting/make_paper_tables.py")
    mpt = importlib.util.module_from_spec(spec)
    sys.modules["mpt"] = mpt
    spec.loader.exec_module(mpt)
    m = mpt.parse_args([])
    assert m.results_dir == d.results and m.dump_dir is None
    import pubg_lab.survival.baselines.sagong2025.__main__ as SG

    assert (SG.DEFAULT_DUMP_DIR, SG.DEFAULT_PLAN_DIR, SG.DEFAULT_RUNS_DIR) == ("model_stgcn/dumps/bigcomp2027_phase_v2", "model_stgcn/dumps/bigcomp2027_sagong_v2",
                                                                              "model_stgcn/checkpoints/bigcomp2027_ptr_v2")  # fmt: skip


# --------------------------------------------------------------------------- versions
def test_name_version_reads_the_last_component_only():
    assert P.name_version("a/bigcomp2027_ptr_v1") == "v1" and P.name_version("bigcomp2027_ptr_v2_lc_steps") == "v2"
    assert P.name_version("x/krafton_winprob_v2") == "v2" and P.name_version("reports/krafton_baseline_2026-10-06") is None
    assert P.name_version("/tmp/test_v1_stuff/out") is None  # a parent directory says nothing
    assert P.name_version("a_v1_v2") is None  # ambiguous
    assert P.name_version("version") is None and P.name_version("v2") == "v2"


def test_dump_version_is_v1_without_the_key_and_v2_with_it(tmp_path):
    assert P.dump_version(make_dump(tmp_path / 'archive/legacy_snapshot')) == "v1"
    v2 = make_dump(tmp_path / "bigcomp2027_phase_v2", version="v2")
    assert P.dump_version(v2) == "v2" and P.dump_version(v2, deep=True) == "v2"
    assert P.dump_version(tmp_path / "not_there_v2") == "v2" and P.dump_version(tmp_path / "not_there") == "unknown"  # a dump that is not built yet: its name
    mixed = make_dump(tmp_path / "mixed", version="v2")
    man = json.loads((mixed / "c_beta" / "dump_manifest.json").read_text())
    del man["version"]
    (mixed / "c_beta" / "dump_manifest.json").write_text(json.dumps(man))
    with pytest.raises(P.VersionMixError, match="mixes dump versions"):
        P.dump_version(mixed)
    liar = make_dump(tmp_path / "liar", version="v2")  # the manifest says v2, the meta of a split does not
    d = torch.load(liar / "c_alpha" / "val.pt", weights_only=False)
    d["meta"].pop("version")
    torch.save(d, liar / "c_alpha" / "val.pt")
    assert P.dump_version(liar) == "v2"  # the cheap read trusts the manifest ...
    with pytest.raises(P.VersionMixError, match="meta\\['version'\\]"):
        P.dump_version(liar, deep=True)  # ... the deep read compares it with meta['version']


def test_artifact_version_evidence_order(tmp_path):
    root = tmp_path / "bigcomp2027_ptr_v1"
    root.mkdir()
    assert P.artifact_version(root) == ("v1", "name")
    P.write_stamp(root, version="v2")
    assert P.artifact_version(root) == ("v2", "stamp")  # a stamp beats the name
    kf = tmp_path / "kf"
    kf.mkdir()
    (kf / "meta.json").write_text(json.dumps({"name": "bigcomp2027_kraftonfeat_v1"}))
    assert P.artifact_version(kf) == ("v1", "meta.name")
    (kf / "meta.json").write_text(json.dumps({"name": "bigcomp2027_kraftonfeat_v1", "version": "v2"}))
    assert P.artifact_version(kf) == ("v2", "meta.version")
    res = tmp_path / "results"
    res.mkdir()
    (res / "summary.json").write_text(json.dumps({"runs_dir": "model_stgcn/checkpoints/bigcomp2027_ptr_v1"}))
    assert P.artifact_version(res) == ("v1", "summary.runs_dir")
    assert P.artifact_version(tmp_path / "plain") == (None, "none")


def test_check_same_version_refuses_mixed_roots_and_unproven_ones(tmp_path):
    v2 = make_dump(tmp_path / "dumps" / "bigcomp2027_phase_v2", version="v2")
    ok = tmp_path / "checkpoints" / "bigcomp2027_ptr_v2"
    bad = tmp_path / "checkpoints" / "bigcomp2027_ptr_v1"
    for r in (ok, bad):
        (r / "c_alpha").mkdir(parents=True)
    assert P.check_same_version(v2, {"runs": ok, "kf": None}) == "v2"
    with pytest.raises(P.VersionMixError, match="runs .*bigcomp2027_ptr_v1 is v1"):
        P.check_same_version(v2, {"runs": bad})
    with pytest.raises(P.VersionMixError, match="refusing to mix versions"):
        P.check_same_version(v2, {"runs": ok, "kf": tmp_path / "dumps" / "bigcomp2027_kraftonfeat_v1"})  # not even existing: its name is v1
    renamed = tmp_path / "checkpoints" / "mine"  # a v1 root that was renamed: no evidence, holds files
    (renamed / "c_alpha").mkdir(parents=True)
    with pytest.raises(P.VersionMixError, match="carry no version evidence"):
        P.check_same_version(v2, {"runs": renamed})
    P.write_stamp(renamed, version="v2")
    assert P.check_same_version(v2, {"runs": renamed}) == "v2"  # a stamp is the proof
    empty = tmp_path / "checkpoints" / "fresh"
    empty.mkdir()
    assert P.check_same_version(v2, {"runs": empty, "other": tmp_path / "nowhere"}) == "v2"  # empty or absent: claimed by the first writer
    # the reverse: a v2 root next to a legacy dump; and a legacy dump ignores roots of unknown origin
    legacy = make_dump(tmp_path / "legacy_dump")
    with pytest.raises(P.VersionMixError, match="is v2"):
        P.check_same_version(legacy, {"runs": ok})
    assert P.check_same_version(legacy, {"runs": bad}) == "v1" and P.check_same_version(legacy, {"runs": tmp_path / "checkpoints" / "fresh"}) == "v1"
    assert P.check_same_version(tmp_path / "unknown_dump", {"runs": bad}) == "unknown"  # nothing to compare with


def test_claim_root_stamps_a_v2_root_and_refuses_the_others(tmp_path):
    v2 = make_dump(tmp_path / "bigcomp2027_phase_v2", version="v2")
    root = tmp_path / "runs_v2"
    assert P.claim_root(root, v2, role="runs") == "v2"
    stamp = json.loads((root / P.STAMP_NAME).read_text())
    assert stamp["version"] == "v2" and stamp["role"] == "runs" and stamp["dump_meta_sha1"] == P.dump_meta_sha1(v2) and stamp["dump_dir"] == str(v2)
    created = stamp["created_utc"]
    assert P.claim_root(root, v2, role="runs") == "v2" and json.loads((root / P.STAMP_NAME).read_text())["created_utc"] == created  # idempotent
    assert P.stamp_status(root, v2) == {"stamp": json.loads((root / P.STAMP_NAME).read_text()), "dump_meta_changed": False}
    man = json.loads((v2 / "c_alpha" / "dump_manifest.json").read_text())
    man["elapsed_s"] = 123.0
    (v2 / "c_alpha" / "dump_manifest.json").write_text(json.dumps(man))
    assert P.stamp_status(root, v2)["dump_meta_changed"] is True  # the dump was rebuilt after the stamp (reported, not refused: the runs go stale by their config hash)
    v1_stamped = tmp_path / "old_root"
    P.write_stamp(v1_stamped, version="v1")
    with pytest.raises(P.VersionMixError):
        P.claim_root(v1_stamped, v2)
    holder = tmp_path / "files_without_evidence"
    holder.mkdir()
    (holder / "something").write_text("x")
    with pytest.raises(P.VersionMixError, match="no version evidence"):
        P.claim_root(holder, v2)
    legacy = make_dump(tmp_path / "bigcomp2027_phase_v1")
    out = tmp_path / "legacy_out"
    assert P.claim_root(out, legacy) == "v1" and not out.exists()  # a legacy dump never stamps: its roots stay byte-identical
    with pytest.raises(P.VersionMixError, match="cannot tell the version"):
        P.claim_root(out, tmp_path / "nowhere_dump")
    assert P.kf_dir_for(v2) == tmp_path / "bigcomp2027_kraftonfeat_v2" and P.kf_dir_for(legacy) == tmp_path / "bigcomp2027_kraftonfeat_v1"


def test_dump_meta_sha1_follows_the_manifests_the_sizes_and_with_deep_the_metas(tmp_path):
    v2 = make_dump(tmp_path / "d", version="v2")
    h = P.dump_meta_sha1(v2)
    assert h == P.dump_meta_sha1(v2) and len(h) == 40
    deep = P.dump_meta_sha1(v2, deep=True)
    assert deep != h and deep == P.dump_meta_sha1(v2, deep=True)
    man = json.loads((v2 / "c_beta" / "dump_manifest.json").read_text())
    man["git_commit"] = "x"
    (v2 / "c_beta" / "dump_manifest.json").write_text(json.dumps(man))
    assert P.dump_meta_sha1(v2) != h  # a manifest changed
    h2 = P.dump_meta_sha1(v2)
    with open(v2 / "c_alpha" / "test.pt", "ab") as f:
        f.write(b"0")
    assert P.dump_meta_sha1(v2) != h2  # a split file changed its size
    d = torch.load(v2 / "c_alpha" / "val.pt", weights_only=False)
    h3 = P.dump_meta_sha1(v2, deep=True)
    d["meta"]["created_at"] = "later"
    torch.save(d, v2 / "c_alpha" / "val.pt")
    assert P.dump_meta_sha1(v2, deep=True) != h3


# --------------------------------------------------------------------------- the gate's meta check
def test_a_valid_v2_dump_passes_the_meta_check_and_the_defects_are_named(tmp_path):
    v2 = make_dump(tmp_path / "d", version="v2")
    res = P.validate_v2_dump(v2)
    assert res["errors"] == [] and res["cohorts"] == ["c_alpha", "c_beta"] and res["dump_meta_sha1"] == P.dump_meta_sha1(v2, deep=True)
    assert any("features (the paper plan says 150)" in w for w in res["warnings"])  # the synthetic dump has far fewer columns: a warning, not an error
    assert P.main(["validate", str(v2), "--shallow"]) == 0 and P.main(["validate", str(v2)]) == 0
    # v1 and the leak marker
    legacy = make_dump(tmp_path / 'archive/legacy_snapshot')
    errs = P.validate_v2_dump(legacy)["errors"]
    assert any("population.json" in e for e in errs) and any("dump_manifest.json version is None" in e for e in errs) and any("meta['version']" in e for e in errs)
    assert any("node_mask_v2" in e for e in errs) or any("missing keys" in e for e in errs)
    marked = make_dump(tmp_path / "marked", version="v2")
    d = torch.load(marked / "c_alpha" / "val.pt", weights_only=False)
    d["matches"][0]["node_mask_v2"] = d["matches"][0]["node_mask"].clone()
    torch.save(d, marked / "c_alpha" / "val.pt")
    assert any("leak L3" in e for e in P.validate_v2_dump(marked)["errors"])
    # causal invariants of the records
    bad = make_dump(tmp_path / "bad", version="v2")
    d = torch.load(bad / "c_alpha" / "val.pt", weights_only=False)
    m = d["matches"][0]
    m["cut_t"] = torch.as_tensor(np.asarray(m["phase_end_s"], dtype=np.float64) - 7.0)  # a_k != b_k - 10
    m["target_mask_v2"] = m["target_mask_v2"].clone()
    m["target_mask_v2"][-1, 0] = True  # a risk-set row in the final slot
    torch.save(d, bad / "c_alpha" / "val.pt")
    errs = P.validate_v2_dump(bad)["errors"]
    assert any("cut_t differs from phase_end_s - 10" in e for e in errs) and any("final phase slot" in e for e in errs)
    assert P.main(["validate", str(bad)]) == 1
    # population and manifests
    pop = make_dump(tmp_path / "pop", version="v2")
    p = json.loads((pop / "population.json").read_text())
    p["cohorts"]["c_ghost"] = {"status": "kept", "after": {"train": 1, "val": 1, "test": 1, "total": 3}}
    (pop / "population.json").write_text(json.dumps(p))
    man = json.loads((pop / "c_alpha" / "dump_manifest.json").read_text())
    man["splits"]["val"]["n_skipped"] = 2
    (pop / "c_alpha" / "dump_manifest.json").write_text(json.dumps(man))
    errs = P.validate_v2_dump(pop)["errors"]
    assert any("kept by population.json but not (completely) dumped: ['c_ghost']" in e for e in errs) and any("could not be dumped" in e for e in errs)
    assert P.validate_v2_dump(tmp_path / "nothing")["errors"] == [f"dump directory {tmp_path / 'nothing'} does not exist"]


def test_paths_command_line(tmp_path, capsys):
    assert P.main(["roots"]) == 0
    assert json.loads(capsys.readouterr().out)["runs"].endswith("bigcomp2027_ptr_v2")
    v2 = make_dump(tmp_path / "bigcomp2027_phase_v2", version="v2")
    assert P.main(["version", str(v2)]) == 0 and capsys.readouterr().out.strip() == "v2"
    root = tmp_path / "somewhere"
    assert P.main(["stamp", str(root), "--dump-dir", str(v2)]) == 0 and P.artifact_version(root) == ("v2", "stamp")


# --------------------------------------------------------------------------- the entry points refuse a mix
def test_the_aggregator_refuses_a_v1_root_and_stamps_the_output_of_a_v2_run(tmp_path):
    import pubg_lab.survival.ptr.evaluation.aggregate as A

    v2 = make_dump(tmp_path / "dumps" / "bigcomp2027_phase_v2", version="v2")
    v1_runs = tmp_path / "checkpoints" / "bigcomp2027_ptr_v1"
    v1_runs.mkdir(parents=True)
    v2_runs = tmp_path / "checkpoints" / "bigcomp2027_ptr_v2"
    v2_runs.mkdir()
    res = tmp_path / "reports" / "bigcomp2027_results_v2"
    with pytest.raises(SystemExit, match="refusing to run"):
        A.main(["--runs-dir", str(v1_runs), "--out-dir", str(res), "--dump-dir", str(v2)])  # v1 runs next to the v2 dump
    with pytest.raises(SystemExit, match="refusing to run"):
        A.main(["--runs-dir", str(v1_runs), "--out-dir", str(res)])  # no dump: the two directories disagree with each other
    assert not res.exists()  # refused before anything was written
    assert A.guard_versions(v2_runs, res, v2) == "v2" and P.artifact_version(res) == ("v2", "stamp")  # the output directory is claimed
    with pytest.raises(SystemExit, match="refusing to run"):
        A.guard_versions(v2_runs, tmp_path / "reports" / "bigcomp2027_results_v1", v2)
    assert A.guard_versions(v1_runs, tmp_path / "plain_out", None) is None  # nothing to compare with


def test_the_h0_modules_and_run_all_refuse_a_v1_root_next_to_the_v2_dump(tmp_path):
    import projects.krafton_baseline.training.h0_baseline as H0
    import projects.krafton_baseline.reporting.h0_report as HR
    import projects.krafton_baseline.pipelines.run_all as RA

    v2 = make_dump(tmp_path / "dumps" / "bigcomp2027_phase_v2", version="v2")
    v1 = tmp_path / "checkpoints" / "bigcomp2027_ptr_v1"
    v1.mkdir(parents=True)
    ok_data = tmp_path / "dumps" / "krafton_winprob_v2"
    with pytest.raises(SystemExit, match="refusing to run"):
        H0.Roots(v1, v2, ok_data).check_versions()
    with pytest.raises(SystemExit, match="refusing to run"):
        H0.main(["--ckpt-dir", str(v1), "--dump-dir", str(v2), "--data-dir", str(ok_data)])
    assert H0.Roots(tmp_path / "checkpoints" / "bigcomp2027_ptr_v2", v2, ok_data).check_versions() == "v2"
    with pytest.raises(SystemExit, match="refusing to run"):
        HR.main(["--ckpt-dir", str(v1), "--dump-dir", str(v2), "--out-dir", str(tmp_path / "reports" / "krafton_baseline_v2")])
    with pytest.raises(SystemExit, match="refusing to run"):
        HR.main(["--ckpt-dir", str(v1), "--out-dir", str(tmp_path / "reports" / "krafton_baseline_v2")])  # without a dump: ckpt and out disagree
    with pytest.raises(SystemExit, match="refusing to run"):
        RA.Roots(v1, ok_data, v2, tmp_path / "dumps" / "bigcomp2027_kraftonfeat_v2").check_versions()
    with pytest.raises(SystemExit, match="refusing to run"):
        RA.main(["--stage", "ourfeat", "--dry-run", "--out-dir", str(tmp_path / "checkpoints" / "bigcomp2027_ptr_v2"), "--data-dir", str(ok_data), "--dump-dir", str(v2),
                 "--table-dir", str(tmp_path / "dumps" / "bigcomp2027_kraftonfeat_v1")])  # a v1 kf table
    # the dump of the H0 run: 'all' = the cohorts of the dump directory
    assert H0.resolve_cohorts("all", H0.Roots(tmp_path / "x_v2", v2, ok_data)) == ["c_alpha", "c_beta"]
    assert H0.resolve_cohorts("a,b", H0.Roots(tmp_path / "x_v2", v2, ok_data)) == ["a", "b"]


def test_make_paper_tables_refuses_v1_results_next_to_the_v2_dump(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("mpt2", Path(__file__).resolve().parents[1] / "projects/battle_royale/reporting/make_paper_tables.py")
    mpt = importlib.util.module_from_spec(spec)
    sys.modules["mpt2"] = mpt
    spec.loader.exec_module(mpt)
    v2 = make_dump(tmp_path / "dumps" / "bigcomp2027_phase_v2", version="v2")
    v1_results = tmp_path / "reports" / "bigcomp2027_results_v1"
    v1_results.mkdir(parents=True)
    with pytest.raises(SystemExit, match="refusing to run"):
        mpt.main(["--results-dir", str(v1_results), "--paper-dir", str(tmp_path / "paper"), "--dump-dir", str(v2)])
    assert not (tmp_path / "paper").exists()


def test_the_gate_reads_prefix_replay_json_and_refuses_mismatches(tmp_path, capsys):
    v2 = make_dump(tmp_path / "d", version="v2")
    assert P.validate_v2_dump(v2)["errors"] == [] and P.validate_v2_dump(v2)["prefix_replay"] is None  # optional unless the gate requires it
    missing = P.validate_v2_dump(v2, require_prefix_replay=True)["errors"]
    assert len(missing) == 1 and "prefix_replay.json is missing" in missing[0]
    ok = {"n_matches": 204, "n_cuts": 1293, "mismatches": [], "min_gap_between_last_kept_record_and_cut_s": 0.001,
          "mutation_check": {"matches": 3, "cuts_compared": 21, "cuts_detected_as_leaky": 21}}  # fmt: skip
    (v2 / P.PREFIX_REPLAY_FILE).write_text(json.dumps(ok))
    res = P.validate_v2_dump(v2, require_prefix_replay=True)
    assert res["errors"] == [] and res["prefix_replay"]["n_matches"] == 204 and res["prefix_replay"]["n_cuts"] == 1293 and res["prefix_replay"]["n_mismatches"] == 0
    assert P.main(["validate", str(v2), "--require-prefix-replay"]) == 0
    for patch, needle in (({"mismatches": [{"match_id": "m", "k": 3, "differing": ["X"]}]}, "1 cut(s) differ"), ({"min_gap_between_last_kept_record_and_cut_s": 0.0}, "min gap"),
                          ({"mismatches": None}, "no 'mismatches'")):  # fmt: skip
        (v2 / P.PREFIX_REPLAY_FILE).write_text(json.dumps({**ok, **patch}))
        errs = P.validate_v2_dump(v2)["errors"]  # checked whenever the file is there
        assert any(needle in e for e in errs), (patch, errs)
    assert P.main(["validate", str(v2)]) == 1
    capsys.readouterr()
    (v2 / P.PREFIX_REPLAY_FILE).write_text(json.dumps({**ok, "mutation_check": {"matches": 3, "cuts_compared": 21, "cuts_detected_as_leaky": 17}}))
    res = P.validate_v2_dump(v2)
    assert res["errors"] == [] and any("caught 17 of 21" in w for w in res["warnings"])  # the replay may be blind to a leak: a warning
    (v2 / P.PREFIX_REPLAY_FILE).write_text("not json")
    assert P.validate_v2_dump(v2)["errors"] == [] and any("missing or unreadable" in e for e in P.validate_v2_dump(v2, require_prefix_replay=True)["errors"]) 


def test_incomplete_cohorts_are_a_gate_error(tmp_path):
    v2 = make_dump(tmp_path / "d", version="v2")
    (v2 / "c_beta" / "test.pt").unlink()
    errs = P.validate_v2_dump(v2)["errors"]
    assert any("incomplete cohorts (missing split files): {'c_beta': ['test']}" in e for e in errs)


def test_the_content_of_a_root_is_evidence_of_its_version(tmp_path):
    """A kf table whose meta.json names ``dump_version``, a Sagong cache whose plan audit does, runs whose metrics do: their own record beats the directory name."""
    kf = tmp_path / "bigcomp2027_kraftonfeat_v1"  # named v1 ...
    kf.mkdir()
    (kf / "meta.json").write_text(json.dumps({"name": "bigcomp2027_kraftonfeat_v2", "version": "v2", "dump_version": "v2"}))  # ... but built from a v2 dump
    assert P.artifact_version(kf) == ("v2", "meta.dump_version")
    sg = tmp_path / "bigcomp2027_sagong_v1"
    (sg / "c_alpha").mkdir(parents=True)
    (sg / "c_alpha" / "plan_audit.json").write_text(json.dumps({"cohort": "c_alpha", "version": "v2"}))
    assert P.artifact_version(sg) == ("v2", "c_alpha/plan_audit.json:version")
    runs = tmp_path / "bigcomp2027_ptr_v1"
    (runs / "c_alpha" / "SAGONG-xgb" / "seed42").mkdir(parents=True)
    (runs / "c_alpha" / "SAGONG-xgb" / "seed42" / "metrics.json").write_text(json.dumps({"dump_version": "v2"}))
    assert P.artifact_version(runs) == ("v2", "c_alpha/SAGONG-xgb/seed42/metrics.json:dump_version")
    (runs / "c_alpha" / "SAGONG-xgb" / "seed42" / "metrics.json").write_text(json.dumps({"data": {"dump_version": "v1"}, "arm": "x"}))
    assert P.artifact_version(runs) == ("v1", "c_alpha/SAGONG-xgb/seed42/metrics.json:dump_version")
    (runs / "c_alpha" / "SAGONG-xgb" / "seed42" / "metrics.json").write_text(json.dumps({"data": ["not", "a", "dict"]}))
    assert P.artifact_version(runs) == ("v1", "name")  # no record in the content: the name
    # a v2 dump refuses a root whose content says v1, whatever it is called
    v2 = make_dump(tmp_path / "dumps" / "bigcomp2027_phase_v2", version="v2")
    liar = tmp_path / "checkpoints" / "bigcomp2027_ptr_v2"
    (liar / "c_alpha" / "SAGONG-xgb" / "seed42").mkdir(parents=True)
    (liar / "c_alpha" / "SAGONG-xgb" / "seed42" / "metrics.json").write_text(json.dumps({"dump_version": "v1"}))
    with pytest.raises(P.VersionMixError, match="is v1"):
        P.check_same_version(v2, {"runs": liar})
