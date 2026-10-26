"""v2 guards of the KRAFTON adaptation (``third_party/krafton_pubg_winprob/adaptation``), DB-free.

1. ``labels.build_cohort`` validates the dump like the PTR loader (``validate_dump``) and checks that EVERY record follows ``meta['version']``
   (``is_v2_match(m) == (dump_version(meta) == 'v2')``, ``DumpVersionError``): a v2 dump must never silently take the v1 branch of
   ``select_labels`` (winner override, ``_cut_exists``).
2. ``summarize`` points ``--ref`` at the v2 results root and refuses to pair a v1 source with a v2 source.
3. ``train.native_metrics`` scores the dump's risk-set rows (``target``, ``T > b_k``) for every arm (K1 / K2 used to score ``RISK`` = ``T > a_k``,
   which also holds the squads that die in ``(a_k, b_k]``); training samples, loss and prediction files are untouched.
"""

from __future__ import annotations

import ast
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
KT = REPO / "third_party" / "krafton_pubg_winprob"
if str(KT) not in sys.path:
    sys.path.insert(0, str(KT))

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")

import projects.krafton_baseline.data.data as KD  # noqa: E402
import projects.krafton_baseline.data.extract as E  # noqa: E402
import projects.krafton_baseline.data.labels as L  # noqa: E402
import projects.krafton_baseline.models.ourfeat as OF  # noqa: E402
import projects.krafton_baseline.training.ourfeat_hazard as H3  # noqa: E402
import projects.krafton_baseline.reporting.summarize as SM  # noqa: E402
import projects.krafton_baseline.training.train as T1  # noqa: E402
import pubg_lab.survival.ptr.data.data as D  # noqa: E402
import pubg_lab.survival.ptr.data.paths as P  # noqa: E402
from pubg_lab.survival.ptr.data.synthetic import make_dump  # noqa: E402
from tests.v2_fixtures import krafton_raw_from_source, make_v2_dump, synth_record  # noqa: E402

SPLITS = ("train", "val", "test")


# =========================================================================== 1. labels.build_cohort
def _save(root: Path, cohort: str, dumps: dict) -> Path:
    base = Path(root) / cohort
    base.mkdir(parents=True, exist_ok=True)
    for split, dump in dumps.items():
        torch.save(dump, base / f"{split}.pt")
    return Path(root)


def _v2_dumps(n: int = 3, seed: int = 0, **kw) -> dict:
    return {s: make_v2_dump(n, seed=seed + j, cohort="c", split=s, **kw) for j, s in enumerate(SPLITS)}


def _v1_dumps(n: int = 2, seed: int = 0) -> dict:
    return {s: make_dump(n, seed=seed + j, cohort="c", split=s) for j, s in enumerate(SPLITS)}


def _strip_v2_keys(m: dict) -> None:
    for k in D.V2_ONLY_KEYS:
        m.pop(k, None)


def test_build_cohort_reads_every_record_of_a_v2_dump_through_the_v2_branch(tmp_path):
    dumps = _v2_dumps(4, seed=10, exclude_prob=0.6, comeback_prob=0.15)
    _save(tmp_path / "dump", "c", dumps)
    L.build_cohort(tmp_path / "dump", "c", tmp_path / "out")
    n_excluded = 0
    for split, dump in dumps.items():
        rec = L.load_labels(tmp_path / "out", "c", split)
        assert rec["dump_version"] == "v2" and L.label_version(rec) == "v2" and len(rec["matches"]) == len(dump["matches"])
        for lab, m in zip(rec["matches"], dump["matches"]):
            assert L.label_version(lab) == "v2" and "query_mask" in lab
            assert np.array_equal(lab["target"], m["target_mask_v2"].numpy())  # as stored: no winner re-added, no cut-existence rule
            assert np.array_equal(lab["y"], (m["y_hazard_v2"] * m["target_mask_v2"]).numpy())
            n_excluded += int((~m["cut_valid"][:7]).sum())
    assert n_excluded > 0  # the fixture has excluded cuts: the v1 branch would put the winners back there


def test_build_cohort_still_builds_a_v1_dump(tmp_path):
    _save(tmp_path / "dump", "c", _v1_dumps(2, seed=3))
    L.build_cohort(tmp_path / "dump", "c", tmp_path / "out")
    for split in SPLITS:
        rec = L.load_labels(tmp_path / "out", "c", split)
        assert rec["dump_version"] == "v1" and all(L.label_version(m) == "v1" for m in rec["matches"])


def test_a_v2_dump_with_a_v1_looking_record_is_refused_before_any_label_is_read(tmp_path, monkeypatch):
    dumps = _v2_dumps(4, seed=11, exclude_prob=0.6)
    bad = dumps["train"]["matches"][2]
    orig = copy.deepcopy(bad)
    _strip_v2_keys(bad)
    assert D.is_v2_match(orig) and not D.is_v2_match(bad)
    # why it matters: read through compact_match, the stripped record takes the v1 branch and its winner is put back at the excluded cuts
    assert bool(orig["team_won"].any()) and bool((~orig["cut_valid"][:7]).any())
    assert not np.array_equal(L.compact_match(bad)["target"], L.compact_match(orig)["target"])
    _save(tmp_path / "dump", "c", dumps)
    calls = []
    monkeypatch.setattr(L, "select_labels", lambda *a, **k: calls.append(1))  # any label read would show up here
    with pytest.raises(D.DumpVersionError, match=rf"c/train: meta\['version'\] is 'v2' but 1 of 4 record\(s\) have no v2-only key.*{bad['match_id']}"):
        L.build_cohort(tmp_path / "dump", "c", tmp_path / "out")
    assert calls == [] and not list((tmp_path / "out").rglob("*.pkl"))  # refused up front: nothing compacted, nothing written


def test_a_v1_dump_with_a_v2_record_is_refused(tmp_path):
    dumps = _v1_dumps(2, seed=5)
    m = dumps["train"]["matches"][1]
    m["query_mask"] = m["node_mask"].clone()
    assert D.is_v2_match(m) and D.dump_version(dumps["train"]) == "v1"
    _save(tmp_path / "dump", "c", dumps)
    with pytest.raises(D.DumpVersionError, match=r"c/train: meta\['version'\] is 'v1' but 1 of 2 record\(s\) have a v2-only key"):
        L.build_cohort(tmp_path / "dump", "c", tmp_path / "out")


@pytest.mark.parametrize(
    "corrupt, text",
    [
        (lambda m: m["team_death_s"].__setitem__(int(m["team_won"].nonzero()[0, 0]), 123.0), "censors the winners"),
        (lambda m: m["query_mask"].__ior__(~m["node_mask"] & ~m["target_mask_v2"]), "query_mask must equal"),
        (lambda m: m["target_mask_v2"].__setitem__((7, 0), True), "last phase slot"),
    ],
)
def test_build_cohort_validates_the_dump_like_the_ptr_loader(tmp_path, corrupt, text):
    dumps = _v2_dumps(3, seed=12)
    corrupt(dumps["train"]["matches"][0])
    assert L.check_record_versions(dumps["train"]) == "v2"  # version-consistent: it is validate_dump that refuses it
    _save(tmp_path / "dump", "c", dumps)
    with pytest.raises(ValueError, match=text):
        L.build_cohort(tmp_path / "dump", "c", tmp_path / "out")
    with pytest.raises(ValueError, match=text):
        D.load_dump(tmp_path / "dump" / "c" / "train.pt")  # the PTR loader says the same
    assert not list((tmp_path / "out").rglob("*.pkl"))


def test_check_record_versions_counts_every_bad_record_and_reports_an_unsupported_version():
    dump = make_v2_dump(4, seed=13)
    assert L.check_record_versions(dump, "x") == "v2" and L.check_record_versions(make_dump(2, seed=14)) == "v1"
    for i in (0, 3):
        _strip_v2_keys(dump["matches"][i])
    with pytest.raises(D.DumpVersionError, match=r"x: meta\['version'\] is 'v2' but 2 of 4 record\(s\) have no v2-only key") as ei:
        L.check_record_versions(dump, "x")
    assert dump["matches"][0]["match_id"] in str(ei.value)
    dump["meta"]["version"] = "v3"
    with pytest.raises(D.DumpVersionError, match="unsupported dump version"):
        L.check_record_versions(dump)


def test_labels_cli_builds_a_good_dump_and_refuses_a_mismatched_one(tmp_path):
    n_threads = torch.get_num_threads()  # main() pins torch to one thread
    try:
        _save(tmp_path / "good", "c", _v2_dumps(2, seed=15))
        assert L.main(["--dump-dir", str(tmp_path / "good"), "--out-dir", str(tmp_path / "out"), "--cohorts", "c"]) == 0
        mixed = _v2_dumps(2, seed=16)
        _strip_v2_keys(mixed["test"]["matches"][0])
        _save(tmp_path / "mixed", "c", mixed)
        with pytest.raises(D.DumpVersionError, match="c/test"):
            L.main(["--dump-dir", str(tmp_path / "mixed"), "--out-dir", str(tmp_path / "out2"), "--cohorts", "c"])
    finally:
        torch.set_num_threads(n_threads)


# =========================================================================== 2. summarize
V1_TABLE = REPO / "reports" / "bigcomp2027_results" / "arm_metrics_per_cohort.csv"


def _roots(tmp_path: Path, runs: str = "bigcomp2027_ptr_v2", results: str = "bigcomp2027_results_v2") -> tuple[Path, Path, Path]:
    r, res = tmp_path / runs, tmp_path / results
    r.mkdir()
    res.mkdir()
    return r, res, res / "arm_metrics_per_cohort.csv"


def test_summarize_defaults_are_the_v2_roots_and_form_a_valid_pair():
    a = SM.parse_args([])
    assert a.ref == REPO / "reports" / "bigcomp2027_results_v2" / "arm_metrics_per_cohort.csv"
    assert a.ckpt_dir == REPO / 'model_stgcn' / "checkpoints" / "bigcomp2027_ptr_v2"
    assert SM.check_sources(a.ckpt_dir, a.ref) == "v2"


@pytest.mark.skipif(not V1_TABLE.is_file(), reason="the v1 results are not in this checkout")
def test_the_old_v1_default_ref_is_refused_next_to_the_v2_runs():
    with pytest.raises(P.VersionMixError, match="refusing to pair"):
        SM.check_sources(P.RUNS_DIR, V1_TABLE)
    with pytest.raises(SystemExit, match="refusing to run"):
        SM.main(["--ref", str(V1_TABLE)])  # the v2 --ckpt-dir default next to the v1 table


@pytest.mark.parametrize("runs, results", [("bigcomp2027_ptr_v2", "bigcomp2027_results_v1"), ("bigcomp2027_ptr_v1", "bigcomp2027_results_v2")])
def test_check_sources_refuses_v1_next_to_v2_in_both_directions(tmp_path, runs, results):
    r, _res, ref = _roots(tmp_path, runs, results)
    with pytest.raises(P.VersionMixError, match=r"refusing to pair --ckpt-dir .* is v[12] \(name\) with --ref .* is v[12] \(name\)"):
        SM.check_sources(r, ref)


def test_check_sources_accepts_equal_versions(tmp_path):
    r, _res, ref = _roots(tmp_path)
    assert SM.check_sources(r, ref) == "v2"
    (tmp_path / "old").mkdir()
    r1, _res1, ref1 = _roots(tmp_path / "old", "bigcomp2027_ptr_v1", "bigcomp2027_results_v1")  # a historical v1 pair is a consistent pair
    assert SM.check_sources(r1, ref1) == "v1"


def test_a_v2_named_results_root_that_aggregated_v1_runs_is_refused(tmp_path):
    r, res, ref = _roots(tmp_path)
    (res / "summary.json").write_text(json.dumps({"runs_dir": "model_stgcn/checkpoints/bigcomp2027_ptr_v1"}))  # evidence beats the directory name
    with pytest.raises(P.VersionMixError, match=r"--ref .* is v1 \(summary.runs_dir\)"):
        SM.check_sources(r, ref)


def test_a_source_without_version_evidence_is_refused_next_to_a_versioned_one_unless_stamped(tmp_path):
    r, res, ref = _roots(tmp_path, results="results_copy")  # no _v<N>, no stamp, no summary: could be a renamed v1 root
    with pytest.raises(P.VersionMixError, match=r"--ref .* carries no version evidence"):
        SM.check_sources(r, ref)
    with pytest.raises(P.VersionMixError, match=r"--ckpt-dir .* carries no version evidence"):
        SM.check_sources(tmp_path / "plain_runs", tmp_path / "bigcomp2027_results_v2" / "arm_metrics_per_cohort.csv")
    assert SM.check_sources(tmp_path / "plain_runs", ref) is None  # neither side says anything: there is nothing to compare
    P.write_stamp(res, version="v2", role="results")
    assert SM.check_sources(r, ref) == "v2"
    P.write_stamp(res, version="v1", role="results")  # an explicit stamp beats the name
    with pytest.raises(P.VersionMixError, match="refusing to pair"):
        SM.check_sources(r, ref)


def _fake_metrics(path: Path, auc: float) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "test": {"hazard": {"auc": auc, "ap": 0.5, "nll": 0.4, "brier": 0.1, "n": 100}, "rank": {"spearman": 0.3, "top5_auc": 0.6, "cindex": 0.55}},
        "epochs_run": 5, "best_epoch": 1, "stopped_early": True, "train_sec": 1.0, "elapsed_sec": 2.0, "baseline": {"platt": {"a": 1.0}},
    }))  # fmt: skip


def test_main_pairs_matching_versions_and_refuses_mixed_ones_before_reading(tmp_path):
    runs, _res, ref = _roots(tmp_path)
    _fake_metrics(runs / "erangel_tpp_squad" / "KRAFTON-tf" / "seed42" / "metrics.json", 0.71)
    rows = [("M01", "auc", 0.8), ("HGB-cut", "auc", 0.7), ("M11R", "auc", 0.81), ("M01", "spearman", 0.4)]
    pd.DataFrame([{"cohort": "erangel_tpp_squad", "arm": a, "seed": 42, "stratum": "all", "metric": m, "value": v, "n_rows": 100, "n_matches": 10}
                  for a, m, v in rows]).to_csv(ref, index=False)  # fmt: skip
    out = tmp_path / "summary.csv"
    assert SM.main(["--ckpt-dir", str(runs), "--ref", str(ref), "--out", str(out)]) == 0
    row = pd.read_csv(out).set_index("cohort").loc["erangel_tpp_squad"]
    assert row["tf_auc"] == pytest.approx(0.71) and row["M01_auc"] == pytest.approx(0.8) and row["HGB_cut_auc"] == pytest.approx(0.7)
    # the same run directory next to a v1 table: refused, and the table is never opened (it does not even exist)
    with pytest.raises(SystemExit, match="refusing to run"):
        SM.main(["--ckpt-dir", str(runs), "--ref", str(tmp_path / "results_v1" / "arm_metrics_per_cohort.csv")])


# =========================================================================== 3. native_metrics rows
def _dump(recs):
    import pubg_lab.survival.data.phase_v2.features as FT

    meta = {"feature_names": list(FT.FEATURE_NAMES), "feature_groups": {"zone": list(range(13))}, "version": "v2", "cohort": "synthetic",
            "split": "val", "zone_columns": list(D.ZONE_COLUMNS)}  # fmt: skip
    return {"meta": meta, "matches": recs}


@pytest.fixture(scope="module")
def window_match():
    """The producer's synthetic match with team 13 (index 2) eliminated at 805 s: inside ``(a_2, b_2] = (800, 810]``, so at cut 2 it is a Cox row
    of K1 / K2 (``T > a_2``) and an alive token, but not a row of the dump's risk set (``T > b_2``)."""
    src, rec = synth_record(extra_kills=[(805.0, "d1")])
    lab = L.compact_match(rec)
    assert lab["team_death_s"][2] == 805.0 and not lab["target"][1, 2] and lab["node_mask"][1, 2]
    return rec, lab, krafton_raw_from_source(src, lab)


def _k1(lab, raw):
    extra = {"flags": {"first_state_w": float(raw["zone"]["t"][0])}, "end_w": lab["match_end_s"]}
    res, _ = E.match_samples("synthetic", lab, raw, extra, "val", "krafton192")
    arrays = {k: res[k] for k in ("X", "POS", "TGT", "TIDX", "S_t", "S_cut", "S_n", "S_zone", "KEY", "RISK")}
    arrays.update(S_mi=np.zeros(len(res["S_t"]), dtype=np.int32), dump_version=np.array("v2"))
    return KD.SampleSet(arrays, np.array(["synthetic"]), 0)


def _k2_k3(rec, lab):
    dump = _dump([rec])
    k2, _ = OF.build_samples(dump, D.Standardizer.fit(dump), {}, "val")
    return k2, H3.attach_labels(k2, {"synthetic": lab})


def _cut_slice(ss, cut):
    i = int(np.flatnonzero(ss.S_cut == cut)[0])
    return i, slice(int(ss.off[i]), int(ss.off[i + 1]))


def _brute(ss, g, labs):
    """Winner accuracy / chance / pooled and mean C-index of the dump's risk-set rows, with plain loops (no shared code with train.py)."""
    acc, raw_acc, chance, per_c, conc, comp = [], [], [], [], 0.0, 0.0
    for i in range(len(ss)):
        lab = labs[str(ss.match_ids[ss.S_mi[i]])]
        k = int(ss.S_cut[i])
        rows = [r for r in range(int(ss.off[i]), int(ss.off[i + 1])) if lab["target"][k - 1, int(ss.TIDX[r])]]
        t = [float(ss.TGT[r]) for r in rows]
        s = [-float(g[r]) for r in rows]
        first = lambda v: min(range(len(v)), key=lambda j: (-v[j], j))  # noqa: E731  (np.argmax: the first maximum)
        win = first(t)
        acc.append(float(first(s) == win))
        raw_acc.append(float(first([-x for x in s]) == win))
        chance.append(1.0 / len(rows))
        c = n = 0.0
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                dt, ds = t[a] - t[b], s[a] - s[b]
                if dt == 0:
                    continue
                n += 1
                c += 1.0 if dt * ds > 0 else 0.5 if ds == 0 else 0.0
        conc, comp = conc + c, comp + n
        per_c.append(c / n if n else np.nan)
    return {"winner_accuracy": np.mean(acc), "winner_accuracy_unfixed_direction": np.mean(raw_acc), "winner_accuracy_chance": np.mean(chance),
            "c_index": conc / comp, "c_index_mean": np.nanmean(per_c)}  # fmt: skip


def test_native_metrics_score_the_dump_target_rows_for_every_arm(window_match):
    rec, lab, raw = window_match
    labs = {"synthetic": lab}
    k1 = _k1(lab, raw)
    k2, k3 = _k2_k3(rec, lab)
    i2, sl = _cut_slice(k1, 2)
    j = list(k1.TIDX[sl].astype(int)).index(2)  # team 13 at cut 2
    assert k1.RISK[sl][j] and k1.KEY[sl][j]  # a Cox row and a key of K1 ...
    assert np.array_equal(k1.RISK, k2.RISK) and not np.array_equal(k1.RISK, k3.RISK)  # ... K1 / K2 score T > a_k, K3a / K3 already the dump rows
    rows = T1.target_rows(k1, labs)
    assert not rows[sl][j] and np.array_equal(rows, k3.RISK)  # ... but not a row of the dump's risk set
    assert np.array_equal(rows, T1.target_rows(k2, labs)) and np.array_equal(rows, T1.target_rows(k3, labs))
    # a model that ranks the team dying in (a_2, b_2] as the best survivor and everybody else perfectly: scored on the dump rows it is perfect
    g = -k1.TGT.astype(np.float64)
    g[int(k1.off[i2]) + j] = -1e6
    out = [T1.native_metrics(ss, g, labs) for ss in (k1, k2, k3)]
    assert out[0] == out[1] == out[2]  # the three arms score the same rows
    a = out[0]["all"]
    assert a["n_samples"] == 3 and a["winner_accuracy"] == 1.0 and a["c_index"] == 1.0 and a["c_index_mean"] == 1.0
    assert a["winner_accuracy_chance"] == pytest.approx(np.mean([1 / 5, 1 / 3, 1 / 2]))  # 5, 3 and 2 risk-set rows at cuts 1, 2 and 3
    c2 = out[0]["by_cut_phase"][2]
    assert c2["winner_accuracy"] == 1.0 and c2["winner_accuracy_chance"] == pytest.approx(1 / 3)  # the RISK rows would have been 4 (chance 1/4)
    # ... whereas on the Cox rows of K1 the same predictions would have named team 13 the winner of cut 2
    rk = k1.RISK[sl]
    assert int(np.argmax(-g[sl][rk])) == list(np.flatnonzero(rk)).index(j)


@pytest.mark.parametrize("kind", ["random", "tied", "rounded"])
def test_native_metrics_equal_a_brute_force_over_the_target_rows(window_match, kind):
    rec, lab, raw = window_match
    labs = {"synthetic": lab}
    k1 = _k1(lab, raw)
    k2, k3 = _k2_k3(rec, lab)
    rng = np.random.default_rng(3)
    g = {"random": rng.normal(size=k1.n_rows), "tied": np.zeros(k1.n_rows), "rounded": np.round(rng.normal(size=k1.n_rows))}[kind]
    want = _brute(k1, g, labs)
    for ss in (k1, k2, k3):
        got = T1.native_metrics(ss, g, labs)["all"]
        for key, v in want.items():
            assert got[key] == pytest.approx(v), key


def test_target_rows_are_the_prediction_rows_and_the_training_inputs_are_unchanged(window_match):
    rec, lab, raw = window_match
    labs = {"synthetic": lab}
    k1 = _k1(lab, raw)
    mean, std = KD.fit_standardizer(k1)
    k1.finalize(mean, std)
    risk0, batch0 = k1.RISK.copy(), k1.batch(np.arange(len(k1)))
    g = np.random.default_rng(1).normal(size=k1.n_rows)
    df = T1.build_frame(k1, g, labs)  # the rows of pred_*.csv
    rows = T1.target_rows(k1, labs)
    got = {(int(p), int(t)) for p, t in zip(df["phase"], df["team_id"])}
    want = {(int(k1.S_cut[i]), int(lab["team_ids"][k1.TIDX[r]])) for i in range(len(k1)) for r in range(int(k1.off[i]), int(k1.off[i + 1])) if rows[r]}
    assert got == want and len(df) == int(rows.sum()) == int(lab["target"].sum()) == 10
    assert int(k1.RISK.sum()) == 11  # K1's Cox loss set is wider by team 13 at cut 2, and it stays so
    T1.native_metrics(k1, g, labs)
    assert np.array_equal(k1.RISK, risk0)  # nothing was written back
    batch = k1.batch(np.arange(len(k1)))
    for key in ("squad_alive_mask", "key_mask", "padding_mask", "y", "x_continuous"):
        assert torch.equal(batch[key], batch0[key]), key  # loss rows, keys, targets and inputs of training are the same
    assert torch.equal(batch["squad_alive_mask"].sum(), torch.as_tensor(int(k1.RISK.sum())))


def test_target_rows_refuse_a_train_sample_and_a_cut_with_unstored_risk_rows(window_match):
    rec, lab, raw = window_match
    k1 = _k1(lab, raw)
    g = np.zeros(k1.n_rows)
    train_like = copy.copy(k1)
    train_like.S_cut = np.zeros_like(k1.S_cut)
    with pytest.raises(AssertionError, match="phase cuts"):
        T1.native_metrics(train_like, g, {"synthetic": lab})
    bad = dict(lab)
    bad["target"] = lab["target"].copy()
    bad["target"][1, 1] = True  # team 12 is not among the stored squads of cut 2
    with pytest.raises(AssertionError, match=r"cut 2: 4 risk-set rows, 3 found"):
        T1.native_metrics(k1, g, {"synthetic": bad})
    with pytest.raises(TypeError):
        T1.native_metrics(k1, g)  # the label tables are required: there is no silent fallback to ss.RISK


def test_every_native_metrics_call_of_the_adaptation_passes_the_label_tables():
    """The call sits after a whole training run: a stale call site would only crash there, so it is checked statically."""
    found: dict[str, list[int]] = {}
    for f in sorted((KT / "adaptation").glob("*.py")):
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
                if name == "native_metrics":
                    found.setdefault(f.name, []).append(len(node.args) + sum(k.arg == "labs" for k in node.keywords))
    assert set(found) == {"train.py", "ourfeat.py", "ourfeat_hazard.py"}  # K1, K2, K3a / K3
    assert all(n >= 3 for counts in found.values() for n in counts), found
