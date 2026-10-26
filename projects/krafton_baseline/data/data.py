"""In-memory sample sets and batch loaders for the KRAFTON baseline (reads the shards written by extract.py).

A sample is one (match, time point): all ALIVE squads of the match at that time.  Dead squads are not stored (they
are excluded from the attention keys and from the loss in the adapted code, so storing them would only cost time),
and padding is added per batch up to the largest squad count of the batch (no 16-squad cap).

v2 shards (written from a v2 dump, ``dump_version`` array in the shard) also store the squads that are at risk but causally
dead at the sample time (the dump's *comeback* rows) and carry two per-row flags: ``KEY`` (causally alive: an attention key)
and ``RISK`` (``T > t``: a Cox loss row).  A batch then has ``squad_alive_mask`` = ``RISK`` (the rows of the loss, as the
upstream ``Trainer`` reads it), ``key_mask`` = ``KEY`` (the attention keys, read by ``adaptation.train.TimedTrainer``) and
``padding_mask`` = the stored rows.  Shards without the flags (v1) give ``KEY = RISK = stored`` as before.

The input vector is upstream's dataset-v2 vector: the 55 ``CONTINUOUS_FEATURES`` standardised with train mean / std
(pandas ``std``, ddof=1, zero std -> 1) followed by the two zone-distance ratios recomputed unscaled (negative =
unknown -> 0), 57 values.  Positions are cm / 80000 (NaN = dead or unknown player), zones cm / 800000, the target
(squad death time) is divided by 2000, exactly as upstream's PGCDatasetV2.
"""

from __future__ import annotations

# Project code is separate from the immutable vendored reference.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))
from pubg_lab.common.paths import REPO_ROOT

from pathlib import Path

import numpy as np
import torch

import projects.krafton_baseline.data.schema as S


class SampleSet:
    def __init__(self, arrays: dict, match_ids: np.ndarray, map_idx: int) -> None:
        self.raw_X = arrays["X"]  # [R, 55] float32 (unstandardised)
        self.POS = arrays["POS"]
        self.TGT = arrays["TGT"]
        self.TIDX = arrays["TIDX"]
        self.S_n = arrays["S_n"].astype(np.int64)
        self.S_t = arrays["S_t"]
        self.S_cut = arrays["S_cut"]
        self.S_zone = arrays["S_zone"]
        self.S_mi = arrays["S_mi"]  # index into match_ids
        self.KEY = arrays.get(
            "KEY"
        )  # [R] bool, v2 shards only: the row is an attention key (causally alive at t)
        self.RISK = arrays.get(
            "RISK"
        )  # [R] bool, v2 shards only: the row is in the Cox risk set (T > t)
        self.dump_version = str(arrays.get("dump_version", "v1"))
        self.match_ids = match_ids
        self.map_idx = int(map_idx)
        self.off = np.concatenate([[0], np.cumsum(self.S_n)])
        self.n_samples = len(self.S_n)
        self.n_rows = int(self.off[-1])
        self.X57: np.ndarray | None = None
        self.pos_scaled: np.ndarray | None = None

    def __len__(self) -> int:
        return self.n_samples

    def finalize(self, mean: np.ndarray, std: np.ndarray) -> "SampleSet":
        """Standardise with the TRAIN statistics and build the 57-d matrix (float32, kept in memory)."""
        R = self.n_rows
        self.X57 = np.empty((R, S.N_INPUT), dtype=np.float32)
        i_v2 = [S.F_IDX[n] for n in S.V2_COMPUTED]
        for a in range(0, R, 500_000):  # chunked: no float64 copy of the whole matrix
            b = min(R, a + 500_000)
            self.X57[a:b, :55] = ((self.raw_X[a:b].astype(np.float64) - mean) / std).astype(
                np.float32
            )
            v2 = self.raw_X[a:b][:, i_v2]
            self.X57[a:b, 55:] = np.where(
                v2 < 0, 0.0, v2
            )  # upstream V2: unknown (NaN / radius <= 0) -> 0.0
        self.pos_scaled = (self.POS / S.POSITION_SCALE).astype(np.float32)
        assert np.isfinite(self.X57).all(), "non-finite input feature"
        self.zone_scaled = (self.S_zone / S.ZONE_SCALE).astype(np.float32)
        return self

    # ------------------------------------------------------------------ batches
    def batch(self, idx: np.ndarray) -> dict:
        """Padded batch (pad = largest squad count of the batch) in the layout of upstream's collate_fn."""
        n = self.S_n[idx]
        pad = int(n.max())
        B = len(idx)
        ar = np.arange(pad)[None, :]
        valid = ar < n[:, None]
        rows = np.where(valid, self.off[idx][:, None] + ar, 0)
        x = np.where(valid[:, :, None], self.X57[rows], 0.0).astype(np.float32)
        pos = np.where(valid[:, :, None, None], self.pos_scaled[rows], np.nan).astype(np.float32)
        y = np.where(valid, self.TGT[rows] / S.TARGET_SCALE, 0.0).astype(np.float32)
        z = self.zone_scaled[idx]
        risk = (
            valid.copy() if self.RISK is None else (valid & self.RISK[rows])
        )  # v1: stored squads are alive by construction
        key = valid.copy() if self.KEY is None else (valid & self.KEY[rows])
        return {
            "x_continuous": torch.from_numpy(x),
            "positions": torch.from_numpy(pos),
            "bluezone_info": torch.from_numpy(z[:, :3].copy()),
            "whitezone_info": torch.from_numpy(z[:, 3:].copy()),
            "y": torch.from_numpy(y),
            "squad_alive_mask": torch.from_numpy(
                risk
            ),  # rows of the Cox loss (v2: T > t); v1: every stored squad
            "key_mask": torch.from_numpy(
                key
            ),  # attention keys (v2: causally alive); v1: every stored squad
            "padding_mask": torch.from_numpy(valid.copy()),
            "num_squads": torch.from_numpy(n),
            "time_point": torch.from_numpy(self.S_t[idx].astype(np.float32)),
            "map_idx": torch.full((B,), self.map_idx, dtype=torch.long),
        }


class BatchLoader:
    """Iterable of batches (shuffled per epoch for training); the attributes the upstream Trainer reads."""

    def __init__(self, data: SampleSet, batch_size: int, shuffle: bool, seed: int = 0) -> None:
        self.data, self.batch_size, self.shuffle, self.seed = data, batch_size, shuffle, seed
        self.epoch = 0
        self.dataset = data

    def __len__(self) -> int:
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.data)
        if self.shuffle:
            order = np.random.RandomState(self.seed * 1000003 + self.epoch).permutation(n)
            self.epoch += 1
        else:
            order = np.arange(n)
        for i in range(0, n, self.batch_size):
            yield self.data.batch(order[i : i + self.batch_size])


def _load_shards(split_dir: Path) -> dict:
    files = sorted(split_dir.glob("shard_*.npz"))
    if not files:
        raise FileNotFoundError(f"no shards in {split_dir}")
    parts, mids, base = [], [], 0
    for f in files:
        with np.load(f) as z:
            d = {k: z[k] for k in z.files}
        d["S_mi"] = d["S_mi"] + base
        base += len(d["match_ids"])
        mids.append(d["match_ids"])
        parts.append(d)
    out = {
        k: np.concatenate([p[k] for p in parts])
        for k in ("X", "POS", "TGT", "TIDX", "S_mi", "S_t", "S_cut", "S_n", "S_zone")
    }
    versions = {str(p["dump_version"]) if "dump_version" in p else "v1" for p in parts}
    if len(versions) != 1:
        raise ValueError(
            f"{split_dir}: shards of different dump versions {sorted(versions)} (mixing v1 and v2 shards)"
        )
    out["dump_version"] = versions.pop()
    if "v2" == out["dump_version"]:
        missing = [k for k in ("KEY", "RISK") if any(k not in p for p in parts)]
        if missing:
            raise ValueError(f"{split_dir}: v2 shards without the per-row flags {missing}")
        for k in ("KEY", "RISK"):
            out[k] = np.concatenate([p[k] for p in parts])
    return out, np.concatenate(mids)


def load_split(
    out_dir: str | Path, cohort: str, split: str, map_name: str | None = None
) -> SampleSet:
    arrays, mids = _load_shards(Path(out_dir) / cohort / "shards" / split)
    return SampleSet(arrays, mids, S.MAPS.index(map_name or S.cohort_map(cohort)))


def fit_standardizer(train: SampleSet) -> tuple[np.ndarray, np.ndarray]:
    """pandas-style mean / std (ddof=1) over the train rows, zero std -> 1 (upstream compute_scaler_params)."""
    R = train.n_rows
    tot = np.zeros(train.raw_X.shape[1])
    for a in range(0, R, 500_000):
        tot += train.raw_X[a : a + 500_000].astype(np.float64).sum(0)
    mean = tot / R
    ss = np.zeros_like(mean)
    for a in range(0, R, 500_000):
        ss += ((train.raw_X[a : a + 500_000].astype(np.float64) - mean) ** 2).sum(0)
    std = np.sqrt(ss / max(R - 1, 1))
    std = np.where(std == 0, 1.0, std)
    return mean, std
