#!/usr/bin/env python3
"""Bounded CPU G1 integration check. No optimizer, training or protected loader.

Run twice into different --output directories to check process-level repeatability.
Selection is fixed before results: first 8 eligible nonempty matches per role,
anchors 0 and midpoint, both views. Reports are never overwritten.
"""
import argparse
from collections import Counter
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import threading
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import psutil
import torch
from world_model.dataset import WorldDataset
from world_model.inputs import ConditionVocabulary, load_sample
from world_model.encoder import WorldEncoder

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(items):
    h = hashlib.sha256()
    for name, tensor in sorted(items):
        array = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode())
        h.update(array.tobytes())
    return h.hexdigest()


class ResourceSampler:
    def __init__(self):
        self.stop = threading.Event()
        self.max_uss = self.max_rss = 0
        self.errors = []
        self.over_limit = False
        self.proc = psutil.Process()

    def measure(self):
        try:
            m = self.proc.memory_full_info()
            self.max_uss = max(self.max_uss, m.uss)
            self.max_rss = max(self.max_rss, m.rss)
            self.over_limit |= m.uss > 48*1024**3
        except psutil.Error as exc:
            self.errors.append(type(exc).__name__)

    def loop(self):
        while not self.stop.wait(.05):
            self.measure()

    def __enter__(self):
        self.measure()
        if self.errors:
            raise RuntimeError("Private-memory measurement unavailable; use read-only process permission")
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.measure()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    # Fix one computation path across grad/no-grad. CPU MHA inference fastpath
    # otherwise changes rounding even with deterministic_algorithms=True.
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(20261020)
    competing = [(p.pid, p.name()) for p in psutil.process_iter(["name"])
                 if any(s in (p.info["name"] or "").lower() for s in ("strata", "llama"))]
    if competing:
        raise RuntimeError(f"Competing process check requires review: {competing}")
    rows, source_hashes, plots = [], {}, []
    modes = Counter()
    with ResourceSampler() as memory:
        started = time.perf_counter()
        ds = WorldDataset(args.root, role="train")
        vocab = ConditionVocabulary.from_dataset(ds)
        model = WorldEncoder(len(ds.item_vocabulary)+2, vocab.sizes).cpu().eval()
        before = tensor_hash(model.state_dict().items())
        vocabulary_hash = hashlib.sha256(json.dumps({
            "items": ds.item_vocabulary, "conditions": vocab.tokens
        }, sort_keys=True).encode()).hexdigest()
        fit_hash = hashlib.sha256("\n".join(sorted(ds.fit_ids)).encode()).hexdigest()
        for role in ("train", "validation"):
            if role == "validation":
                ds = WorldDataset(args.root, role=role)
            used = 0
            for mid in sorted(ds.match_ids):
                bank = ds.bank(mid)
                if not len(bank["anchors"]):
                    continue
                source_hashes[mid] = {
                    "bank": sha(ds.root/"banks"/f"{mid}.npz"),
                    "meta": sha(ds.root/"meta"/f"{mid}.json")}
                for anchor in sorted({0, len(bank["anchors"])//2}):
                    paired = []
                    for view in ("observer", "server"):
                        ds.view = view
                        t = time.perf_counter()
                        sample = load_sample(ds, mid, anchor, vocab)
                        load_s = time.perf_counter()-t
                        model.zero_grad(set_to_none=True)
                        t = time.perf_counter()
                        result = model(sample.inputs)
                        forward_s = time.perf_counter()-t
                        assert all(torch.isfinite(v).all() for v in result.values())
                        out_hash = tensor_hash(result.items())
                        with torch.no_grad():
                            assert out_hash == tensor_hash(model(sample.inputs).items())
                        # A fixed scalar probe verifies autograd connectivity; not a task loss.
                        probe = sum(v[..., 0].sum() for v in result.values()
                                    if v.dtype.is_floating_point and v.numel())
                        t = time.perf_counter()
                        probe.backward()
                        backward_s = time.perf_counter()-t
                        gradients = [(k, p.grad) for k, p in model.named_parameters()
                                     if p.grad is not None]
                        assert gradients and all(torch.isfinite(g).all() for _, g in gradients)
                        assert any(torch.count_nonzero(g) for _, g in gradients)
                        g = sample.inputs
                        modes.update(map(int, g.mobility[g.mobility_mask].tolist()))
                        row = {
                            **sample.provenance, "role": role,
                            "input_shapes": {f.name: list(getattr(g, f.name).shape)
                                             for f in fields(g)
                                             if isinstance(getattr(g, f.name), torch.Tensor)},
                            "output_shapes": {k: list(v.shape) for k, v in result.items()},
                            "input_sha256": tensor_hash(
                                (f.name, getattr(g, f.name)) for f in fields(g)
                                if isinstance(getattr(g, f.name), torch.Tensor)),
                            "output_sha256": out_hash,
                            "gradient_sha256": tensor_hash(gradients),
                            "gradient_parameter_tensors": len(gradients),
                            "condition_ids": g.conditions.tolist(),
                            "complete_60s": sample.targets["joint_complete_60s"],
                            "target_rows": sample.targets["selected_rows"].tolist(),
                            "target_sha256": hashlib.sha256(
                                sample.targets["xy"].tobytes()+sample.targets["mask"].tobytes()
                            ).hexdigest(),
                            "load_s": load_s, "forward_s": forward_s, "backward_s": backward_s,
                        }
                        rows.append(row)
                        paired.append(row)
                        if used == 0 and anchor == 0 and view == "observer":
                            plots.append({
                                "role": role, "patch": sample.provenance["patch"], "match_id": mid,
                                "t0": sample.provenance["t0"],
                                "player_xy_m": (g.x[-1, :, :2]*100+g.origin_m).tolist(),
                                "valid": g.history_valid[-1].tolist(),
                                "history_xy_m": (g.x[:, :, :2]*100+g.origin_m).tolist(),
                                "history_valid": g.history_valid.tolist(),
                                "cell_xy_m": (g.cells[:, :2]*100+g.origin_m).tolist(),
                                "observed": g.observed.tolist(), "inferred": g.inferred.tolist(),
                            })
                        del result, probe, gradients, sample
                        if memory.over_limit:
                            raise MemoryError("48 GiB private-memory limit")
                    assert paired[0]["target_sha256"] == paired[1]["target_sha256"]
                    assert paired[0]["target_rows"] == paired[1]["target_rows"]
                used += 1
                if used == 8:
                    break
            assert used == 8
        assert before == tensor_hash(model.state_dict().items()), "Parameters changed"
        elapsed = time.perf_counter()-started
    assert not memory.over_limit and not memory.errors
    report = {
        "status": "G1_CPU_INTERFACE_VERIFIED_NOT_TRAINED", "optimizer_steps": 0,
        "seed": 20261020, "device": "cpu", "threads": 1,
        "mha_fastpath_enabled": torch.backends.mha.get_fastpath_enabled(),
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "numpy": np.__version__, "psutil": psutil.__version__,
        "executable": sys.executable, "elapsed_s": elapsed,
        "memory": {"sampled_peak_uss_bytes": memory.max_uss,
                   "sampled_peak_rss_bytes": memory.max_rss,
                   "ru_maxrss_native": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "sampling_interval_s": .05, "private_limit_bytes": 48*1024**3},
        "competing_processes": competing, "parameter_count": sum(p.numel() for p in model.parameters()),
        "state_sha256_before_and_after": before, "vocabulary_sha256": vocabulary_hash,
        "fit_ids_sha256": fit_hash, "source_root": str(Path(args.root).resolve()),
        "source_hashes": source_hashes,
        "code_hashes": {p: sha(ROOT/p) for p in
                       ("world_model/preprocess.py", "world_model/dataset.py",
                        "world_model/inputs.py", "world_model/encoder.py",
                        "tools/verify_world_model_encoder.py")},
        "mobility_frames_by_code": dict(modes), "examples": rows,
        "limitations": [
            "Bounded interface check: not a population performance estimate.",
            "No optimization, pretraining, decoder, path generation or test evaluation.",
            "41.1 is previously exposed development data, not pristine final test.",
            "Observer visibility is a HUD proxy, not line-of-sight reconstruction.",
            "Strict patch spatial prior may be empty on the next patch.",
        ],
    }
    (output/"receipt.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (output/"plot_data.json").write_text(json.dumps(plots, indent=2))
    print(json.dumps({"status": report["status"], "examples": len(rows),
                      "matches": len(source_hashes), "elapsed_s": elapsed,
                      "peak_private_MiB": memory.max_uss/1024**2,
                      "parameters": report["parameter_count"]}))


if __name__ == "__main__":
    main()
