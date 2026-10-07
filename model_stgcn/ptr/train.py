"""Train one PTR arm on one cohort dump (DB-free) and write predictions + metrics.

    python -m model_stgcn.ptr.train --dump-dir model_stgcn/dumps/bigcomp2027_phase_v1 \\
        --cohort erangel_tpp_squad --arm M11RZ --seed 42 \\
        --out-dir model_stgcn/checkpoints/bigcomp2027_ptr_v1 [--device mps|cpu] [--max-epochs 12]

Shared recipe for every neural arm (design doc section 4): d=128, H=4, L=2, dropout 0.1; AdamW lr 1e-3,
weight decay 1e-4, cosine schedule over ``--max-epochs`` (stepped per epoch), gradient clipping 5.0,
batches of 8 matches, loss = hazard BCE + 0.1 * pairwise rank loss (``--rank-weight``; the ``*-norank`` control arms of
``arms.NORANK_ARMS`` fix it to 0), early stopping on validation NLL
(patience 3), best epoch restored.  Seeds fix the init (per-component streams, so shared components
start identically across arms), the batch order and dropout; CPU runs are bitwise reproducible, MPS
runs are not guaranteed to be.

Outputs in ``<out-dir>/<cohort>/<arm>/seed<seed>/`` (``sub<N>_ss<s>_seed<seed>/`` with ``--train-subsample N``): ``best.pt``, ``pred_val.csv``, ``pred_test.csv``
(``metrics.PRED_COLUMNS``; ``phase`` = cut phase k = dump slot p + 1, ``rank_score`` higher = better survival)
and ``metrics.json`` (config, history with seconds per epoch, val/test metrics, the removed feature columns
of the noHab arms, the resolved combat-weight mode, elapsed seconds, git provenance).

Data selection (design doc v2.3): ``--labels v2|pipeline`` (default v2, the event-based labels) picks the label
set that defines the risk set, the hazard labels and the alive tokens for training, validation, the CSVs and every
metric; the choice is recorded in ``metrics.json`` / ``best.pt`` and a run directory holding results of the other
choice is never overwritten.  ``team_ai_frac`` is appended to every team token (required in the dump);
``match_type`` is written to the CSVs and ``--match-types`` (default ``official,competitive``; ``any`` disables)
guards the registered population.  Cohorts come from the dump directory (``python -m model_stgcn.ptr.cohorts``).

Switches beyond the spec (defaults reproduce the spec): ``--rank-reduction pair|group`` (rank loss mean over
all pairs, or over per-(match, phase) means), ``--combat-weights auto|raw|log`` (the real dump already holds
ln(1 + intensity); ``auto`` reads it off the train weights so the combat channel is log1p(intensity) either
way), ``--no-pos-feature`` (drop the 2-d team centre from the token features).

Union arm (design doc 4.6, post hoc / exploratory): ``--arm M11R-union --extra-features DIR`` appends the ``kf_*`` columns of
``DIR/<cohort>/{train,val,test}.csv.gz`` (KRAFTON's reconstructed inputs, ``model_stgcn.ptr.extra``) to the team token
features, after ``X``, the team centre and ``team_ai_frac``: z-scored on the train alive tokens, NaN -> 0, broadcast to the phase
slots by cut phase.  Every alive token of a cut slot must have a table row (a gap fails before training, with the list); the
feature set, its hash and the join counts are written to ``metrics.json`` (``extra_features``) and into the config hash.
``--extra-features`` is refused for every other arm.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model_stgcn.ptr import extra as X
from model_stgcn.ptr import metrics as M
from model_stgcn.ptr.arms import ARMS, UNION_ARMS, ArmSpec, get_arm, model_config
from model_stgcn.ptr.data import (
    DEFAULT_LABELS,
    DEFAULT_MATCH_TYPES,
    LABEL_CHOICES,
    LABEL_FIELDS,
    PlayerStandardizer,
    PTRDataset,
    Standardizer,
    alive_source,
    effective_alive,
    event_alive_summary,
    infer_combat_weights,
    load_cohort,
    match_type_counts,
    select_labels,
    subsample_dump,
    uses_event_alive,
    winner_censor_summary,
)
from model_stgcn.ptr.loss import RANK_REDUCTIONS, ptr_loss
from model_stgcn.ptr.player import (
    default_player_dump_dir,
    infer_player_combat_weights,
    load_player_cohort,
    player_summary,
)
from model_stgcn.ptr.relations import COMBAT_WEIGHT_MODES
from model_stgcn.ptr.model import PTRConfig, PTRFormer
from model_stgcn.ptr.util import (
    check_run_dir,
    code_fingerprint,
    config_hash as _hash,
    dump_fingerprint,
    git_info,
    run_dir_name,
    set_seed,
    write_json,
)

logger = logging.getLogger("model_stgcn.ptr.train")

DEFAULT_RANK_WEIGHT = 0.1  # the shared recipe: loss = hazard BCE + 0.1 * pairwise rank loss


# --------------------------------------------------------------------------- helpers
class EarlyStopper:
    """Stops after ``patience`` consecutive epochs without a strict improvement; NaN never improves."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best = float("inf")
        self.best_epoch = 0
        self.bad = 0
        self.epoch = 0

    def update(self, value: float) -> bool:
        self.epoch += 1
        if value < self.best:
            self.best, self.best_epoch, self.bad = float(value), self.epoch, 0
            return True
        self.bad += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad >= self.patience


def _device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but MPS is not available")
    return torch.device(name)


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _chunks(order, size: int):
    for start in range(0, len(order), size):
        yield [int(i) for i in order[start : start + size]]


# --------------------------------------------------------------------------- data preparation
@dataclass
class Prepared:
    arm: ArmSpec
    standardizer: Standardizer
    datasets: dict[str, PTRDataset]
    config: PTRConfig
    combat_weights: str = "raw"  # resolved mode ("raw" -> log1p is applied, "log" -> dump is already ln(1+x))
    combat_weight_max: float | None = None  # largest train combat weight (what "auto" looked at)
    player_standardizer: PlayerStandardizer | None = None  # stage 2 only
    player_combat_weights: str = "raw"
    labels: str = DEFAULT_LABELS
    match_types: tuple[str, ...] | None = DEFAULT_MATCH_TYPES
    data: dict | None = None  # what the run saw: matches / risk-set rows / match types per split
    extra_standardizer: X.ExtraStandardizer | None = None  # union arm only
    extra_info: dict | None = None  # union arm only: the ``extra_features`` block of metrics.json


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _parse_match_types(text: str) -> tuple[str, ...] | None:
    if text.strip().lower() == "any":
        return None
    types = tuple(t.strip() for t in text.split(",") if t.strip())
    if not types:
        raise argparse.ArgumentTypeError("--match-types needs at least one match type (or 'any')")
    return types


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m model_stgcn.ptr.train", description=__doc__.split("\n")[0])
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--max-epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument(
        "--rank-weight",
        type=float,
        default=None,
        help=f"weight of the pairwise rank loss (default {DEFAULT_RANK_WEIGHT}); the *-norank arms of arms.NORANK_ARMS fix it to 0 and refuse another value",
    )
    ap.add_argument(
        "--labels",
        default=DEFAULT_LABELS,
        choices=LABEL_CHOICES,
        help="label set for the risk set, the hazard labels and the alive tokens (v2 = event based, "
        "pipeline = team_phase_labels); recorded in metrics.json",
    )
    ap.add_argument(
        "--match-types",
        default=",".join(DEFAULT_MATCH_TYPES),
        type=_parse_match_types,
        help="allowed match_type values, comma separated ('any' = no check); default official,competitive",
    )
    ap.add_argument(
        "--rank-reduction",
        default="pair",
        choices=RANK_REDUCTIONS,
        help="rank loss: mean over all pairs of the batch ('pair') or mean of per-(match, phase) means",
    )
    ap.add_argument(
        "--combat-weights",
        default="auto",
        choices=("auto", *COMBAT_WEIGHT_MODES),
        help="what the dumped combat weights are: 'raw' intensity (channel = log1p(w)), already 'log' "
        "= ln(1+intensity) (channel = w), or 'auto': 'log' iff every train weight <= 30 (the real dump)",
    )
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument(
        "--threads",
        type=_positive_int,
        default=None,
        help="torch intra-op threads for this run (restored afterwards); never changes a result, only the speed",
    )
    ap.add_argument(
        "--train-subsample",
        type=_positive_int,
        default=None,
        metavar="N",
        help="train on a random subsample of N train matches (learning curve, design doc 4.4); val and test are "
        "unchanged, the standardiser is fitted on the subsample.  Results go to <cohort>/<arm>/sub<N>_ss<s>_seed<seed>/",
    )
    ap.add_argument(
        "--subsample-seed",
        type=int,
        default=None,
        help="seed of the subsample (default: --seed); subsamples of one seed are nested across N",
    )
    ap.add_argument(
        "--player-dump-dir",
        default=None,
        help="stage-2 arms (M11RZ-P, M11RZ-P-tm): directory of the player dumps <cohort>/{train,val,test}.pt "
        "(default: bigcomp2027_player_v1 next to --dump-dir)",
    )
    ap.add_argument("--player-layers", type=int, default=1, help="stage-2 arms: relation-biased player attention blocks")
    ap.add_argument(
        "--no-pos-feature",
        action="store_true",
        help="do not append the 2-d normalised team centre to the token features",
    )
    ap.add_argument(
        "--extra-features",
        default=None,
        metavar="DIR",
        help=f"union arms {UNION_ARMS} (design doc 4.6): directory of the kf tables <cohort>/{{train,val,test}}.csv.gz whose "
        "kf_* columns are appended to the team token features; required by them and refused for every other arm",
    )
    args = ap.parse_args(argv)
    fixed = ARMS[args.arm].rank_weight  # an arm that fixes its loss (the E1 controls) wins; a conflicting command line is an error
    if fixed is not None:
        if args.rank_weight is not None and float(args.rank_weight) != float(fixed):
            ap.error(f"arm {args.arm} fixes --rank-weight at {fixed}; got {args.rank_weight}")
        args.rank_weight = float(fixed)
    elif args.rank_weight is None:
        args.rank_weight = DEFAULT_RANK_WEIGHT
    if ARMS[args.arm].extra and not args.extra_features:
        ap.error(f"arm {args.arm} needs --extra-features DIR (the kf tables <DIR>/<cohort>/{{train,val,test}}.csv.gz)")
    if args.extra_features and not ARMS[args.arm].extra:
        ap.error(f"--extra-features is only defined for the union arms {UNION_ARMS}, not for {args.arm}")
    return args


def _player_dir(args: argparse.Namespace) -> Path:
    return Path(args.player_dump_dir) if args.player_dump_dir else default_player_dump_dir(args.dump_dir)


def _player_spec(args: argparse.Namespace) -> dict:
    """Only stage-2 arms carry the player fields, so the hash of every other run is what it was."""
    if not get_arm(args.arm).player:
        return {}
    return {"player_layers": args.player_layers, "player_dump": dump_fingerprint(_player_dir(args), args.cohort)}


def _extra_spec(args: argparse.Namespace) -> dict:
    """Only union arms carry the extra-feature fields (feature-set hash, size / mtime of the tables): the hash of every other
    run is what it was."""
    if not get_arm(args.arm).extra:
        return {}
    return {"extra_features": X.extra_spec(getattr(args, "extra_features", None), args.cohort)}


def _subsample_seed(args: argparse.Namespace) -> int:
    return args.seed if args.subsample_seed is None else args.subsample_seed


def _subsample_spec(args: argparse.Namespace) -> dict:
    """Only present for subsampled runs, so the hash of a full-size run does not depend on the feature."""
    if args.train_subsample is None:
        return {}
    return {"train_subsample": args.train_subsample, "subsample_seed": _subsample_seed(args)}


def config_spec(args: argparse.Namespace) -> dict:
    """Everything that can change a neural run's result (threads, out-dir and logging cannot)."""
    return {
        "kind": "neural",
        "arm": args.arm, "cohort": args.cohort, "seed": args.seed, "labels": args.labels,
        "match_types": None if args.match_types is None else list(args.match_types),
        "device": args.device, "max_epochs": args.max_epochs, "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": args.weight_decay, "grad_clip": args.grad_clip, "patience": args.patience,
        "rank_weight": args.rank_weight, "rank_reduction": args.rank_reduction,
        "combat_weights": args.combat_weights, "d_model": args.d_model, "n_heads": args.n_heads,
        "n_layers": args.n_layers, "dropout": args.dropout, "pos_feature": not args.no_pos_feature,
        "dump": dump_fingerprint(args.dump_dir, args.cohort), "code": code_fingerprint(),
        **_subsample_spec(args), **_player_spec(args), **_extra_spec(args),
    }  # fmt: skip


def config_hash(args: argparse.Namespace) -> str:
    """Hash stamped into ``metrics.json``; the runner skips a run whose stamp equals this value."""
    return _hash(config_spec(args))


def prepare(args: argparse.Namespace) -> Prepared:
    arm = get_arm(args.arm, implemented_only=True)  # stubs fail before any data is read
    splits = load_cohort(args.dump_dir, args.cohort, labels=args.labels, match_types=args.match_types)
    splits["train"], sub_info = subsample_dump(splits["train"], args.train_subsample, _subsample_seed(args))
    if sub_info is not None:
        logger.info("train subsample: %s", sub_info)
    standardizer = Standardizer.fit(splits["train"], args.labels)  # train alive tokens only (of the subsample)
    players = player_std = player_info = None
    player_mode, player_max = "raw", None
    if arm.player:
        if args.labels != "v2":
            raise ValueError(
                f"{arm.name} (stage 2) is defined for --labels v2: the player alive flag and the edges are aligned to the "
                "event-based team alive mask of the v2 labels, which the pipeline labels do not use"
            )
        player_dir = _player_dir(args)
        players, player_info = load_player_cohort(player_dir, args.cohort, splits, labels=args.labels)  # joined to the (sub)sampled matches
        player_std = PlayerStandardizer.fit(splits["train"], players["train"], args.labels)
        inferred, player_max = infer_player_combat_weights(players["train"].values())
        player_mode = inferred if args.combat_weights == "auto" else args.combat_weights
        logger.info(
            "player layer: %d features, relation channels %s, combat weights mode=%s (largest train weight %.3f), dump %s",
            len(player_info["feature_names"]), list(arm.player_relation_types), player_mode, player_max, player_dir,
        )  # fmt: skip
    extra_joined = extra_std = extra_info = None
    extra_dir = getattr(args, "extra_features", None)
    if arm.extra:
        if not extra_dir:
            raise ValueError(f"arm {arm.name} needs --extra-features DIR")
        extra_joined = X.load_extra(extra_dir, args.cohort, splits, args.labels)  # a gap / disagreement fails here, before training
        extra_std = X.ExtraStandardizer.fit(extra_joined["train"], splits["train"], args.labels)  # train alive tokens only
        extra_info = X.describe(extra_dir, args.cohort, extra_joined, extra_std)
        logger.info(
            "extra features: %d kf_ columns (sha1 %s) from %s; constant in train: %s; join %s",
            extra_info["n_features"], extra_info["names_sha1"][:12], extra_dir,
            extra_info["standardizer"]["constant_train_columns"],
            {k: (v["alive_tokens_cut_slots"], v["target_rows"], v["alignment"].get("checked")) for k, v in extra_info["join"].items()},
        )  # fmt: skip
    elif extra_dir:
        raise ValueError(f"--extra-features is only defined for the union arms {UNION_ARMS}, not for {arm.name}")
    combat_mode, combat_max = "raw", None
    if arm.relation:
        inferred, combat_max = infer_combat_weights(splits["train"])
        combat_mode = inferred if args.combat_weights == "auto" else args.combat_weights
        logger.info(
            "combat weights: mode=%s (requested %s; largest train weight %.3f)",
            combat_mode, args.combat_weights, combat_max,
        )  # fmt: skip
    datasets = {
        name: PTRDataset(
            dump,
            standardizer,
            drop_groups=arm.drop_groups,
            keep_names=arm.keep_names,
            pos_in_token=not args.no_pos_feature,
            relation_types=arm.relation_types if arm.relation else None,
            combat_weights=combat_mode,
            shuffle_seed=args.seed if arm.shuffle else None,
            labels=args.labels,
            validate=False,  # load_cohort validated
            players=None if players is None else players[name],
            player_standardizer=player_std,
            player_relation_types=arm.player_relation_types if arm.player else None,
            player_combat_weights=player_mode,
            extra=None if extra_joined is None else extra_joined[name],
            extra_standardizer=extra_std,
        )
        for name, dump in splits.items()
    }
    data = {
        "labels": args.labels,
        "train_subsample": sub_info,
        "match_types_allowed": None if args.match_types is None else list(args.match_types),
        "n_matches": {k: len(v["matches"]) for k, v in splits.items()},
        "n_target_rows": {
            k: sum(int(select_labels(m, args.labels)[1].sum()) for m in v["matches"]) for k, v in splits.items()
        },
        "match_types": {k: match_type_counts(v) for k, v in splits.items()},
        "winner_censor": {k: winner_censor_summary(v, args.labels) for k, v in splits.items()},
        "alive_mask": {
            "source": alive_source(splits["train"], args.labels),
            "per_split": {k: event_alive_summary(v, args.labels) for k, v in splits.items()},
        },
    }
    if arm.player:
        data["player"] = {
            "dump_dir": str(_player_dir(args)),
            "feature_names": player_info["feature_names"],
            "join": player_info["per_split"],  # incl. edge_endpoints_not_live: teammate / enemy_prox must be 0 (asserted)
            "ignored_edge_types": player_info["ignored_edge_types"],
            "combat_weights": player_mode,
            "combat_weight_max_train": player_max,
            "relation_types": list(arm.player_relation_types),
            "layers": args.player_layers,
            "per_split": {
                k: player_summary(
                    [players[k][str(m["match_id"])] for m in v["matches"]],
                    [effective_alive(m, args.labels, uses_event_alive(v, args.labels)) for m in v["matches"]],
                )
                for k, v in splits.items()
            },
        }
    logger.info("labels=%s  target rows per split: %s", args.labels, data["n_target_rows"])
    logger.info("winner censoring (train): %s", data["winner_censor"]["train"])
    tr = data["alive_mask"]["per_split"]["train"]
    logger.info(
        "alive tokens: %s (train: %d alive, %d ghost tokens removed from node_mask, %d tokens added to it)",
        data["alive_mask"]["source"], tr["n_alive"], tr["ghost_removed"], tr["raised"],
    )
    if "start_rule" in tr:
        logger.info(
            "event mask start rule: %s; audit vs the dump's node_mask_v2 (train): agree %s, only loader %s, only dump %s",
            tr["start_rule"], tr.get("agree"), tr.get("only_loader"), tr.get("only_dump"),
        )
    ref = datasets["train"]
    config = model_config(
        arm,
        ref.n_features,
        n_phases=ref.n_phases,
        zone_dim=ref.zone_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        player_features=ref.n_player_features,
        player_layers=args.player_layers,
    )
    for name, ds in datasets.items():
        if ds.n_phases != ref.n_phases:
            raise ValueError(f"{name} has {ds.n_phases} phase slots, train has {ref.n_phases}")
    return Prepared(
        arm, standardizer, datasets, config, combat_mode, combat_max, player_std, player_mode, args.labels,
        args.match_types, data, extra_std, extra_info,
    )  # fmt: skip


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate_nll(model: PTRFormer, ds: PTRDataset, device: torch.device, batch_size: int) -> float:
    """Pooled hazard NLL over all target rows (the early-stopping criterion)."""
    model.eval()
    total, count = 0.0, 0
    for idx in _chunks(range(len(ds)), batch_size):
        batch = _to_device(ds.get_batch(idx), device)
        tm = batch["target_mask"]
        n = int(tm.sum())
        if n == 0:
            continue
        logit = model(batch)["hazard_logit"]
        per = F.binary_cross_entropy_with_logits(logit[tm], batch["y_hazard"][tm], reduction="sum")
        total += float(per)
        count += n
    return total / count if count else float("nan")


@torch.no_grad()
def predict(
    model: PTRFormer, ds: PTRDataset, device: torch.device, batch_size: int = 8, rank_from_hazard: bool = False
) -> pd.DataFrame:
    """Predictions for every target row, in the canonical prediction-file format and order.

    ``rank_from_hazard``: the rank head was not trained (``rank_weight`` 0), so ``rank_score`` is ``-hazard logit`` instead of the
    head's output (the convention of the tabular and KRAFTON arms: the rank metrics then read the hazard head's own ranking)."""
    model.eval()
    frames = []
    for idx in _chunks(range(len(ds)), batch_size):
        batch = ds.get_batch(idx)
        out = model(_to_device(batch, device))
        logit = out["hazard_logit"].float().cpu()
        score = -logit if rank_from_hazard else out["rank_score"].float().cpu()
        b, p, i = batch["target_mask"].nonzero(as_tuple=True)
        frames.append(
            pd.DataFrame(
                {
                    "match_id": [batch["match_ids"][k] for k in b.tolist()],
                    "team_id": batch["team_ids"][b, i].numpy(),
                    "phase": p.numpy() + M.PHASE_OFFSET,  # cut phase k = slot p + 1
                    "y": batch["y_hazard"][b, p, i].numpy(),
                    "p_hazard": torch.sigmoid(logit[b, p, i]).numpy(),
                    "rank_score": score[b, p, i].numpy(),
                    "final_rank_norm": batch["final_rank_norm"][b, i].numpy(),
                    "top5": batch["top5"][b, i].numpy(),
                    "match_type": [batch["match_types"][k] for k in b.tolist()],
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=M.PRED_COLUMNS)
    return M.read_predictions(pd.concat(frames, ignore_index=True))


# --------------------------------------------------------------------------- training
@dataclass
class FitResult:
    model: PTRFormer
    history: list[dict]
    best_epoch: int
    best_val_nll: float
    stopped_early: bool
    train_sec: float


def fit(prep: Prepared, args: argparse.Namespace, device: torch.device, ckpt_path: Path | None = None) -> FitResult:
    set_seed(args.seed)
    model = PTRFormer(prep.config, init_seed=args.seed).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=max(args.max_epochs, 1))
    stopper = EarlyStopper(args.patience)
    train_ds, val_ds = prep.datasets["train"], prep.datasets["val"]
    order_rng = np.random.default_rng(args.seed)
    best_state, best_epoch, history = None, 0, []
    t_train = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        t_epoch = time.perf_counter()
        model.train()
        running, steps = 0.0, 0
        for idx in _chunks(order_rng.permutation(len(train_ds)), args.batch_size):
            batch = _to_device(train_ds.get_batch(idx), device)
            loss, _ = ptr_loss(model(batch), batch, args.rank_weight, args.rank_reduction)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            running += float(loss.detach())
            steps += 1
        lr_used = opt.param_groups[0]["lr"]
        sched.step()
        val_nll = evaluate_nll(model, val_ds, device, args.batch_size)
        improved = stopper.update(val_nll)
        if improved or best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            if ckpt_path is not None:
                _save_checkpoint(ckpt_path, model, prep, args, epoch, val_nll)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / max(steps, 1),
                "val_nll": val_nll,
                "lr": lr_used,
                "sec": time.perf_counter() - t_epoch,
            }
        )
        logger.info(
            "epoch %02d  train_loss=%.4f  val_nll=%.4f  lr=%.2e  %.1fs%s",
            epoch, history[-1]["train_loss"], val_nll, lr_used, history[-1]["sec"],
            "  *" if improved else "",
        )  # fmt: skip
        if stopper.should_stop:
            logger.info("early stop at epoch %d (best epoch %d)", epoch, best_epoch)
            break
    train_sec = time.perf_counter() - t_train
    model.load_state_dict(best_state)
    best_val = next(h["val_nll"] for h in history if h["epoch"] == best_epoch)
    return FitResult(model, history, best_epoch, best_val, len(history) < args.max_epochs, train_sec)


def _save_checkpoint(path: Path, model, prep: Prepared, args, epoch: int, val_nll: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ref = prep.datasets["train"]
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": asdict(prep.config),
            "arm": asdict(prep.arm),
            "seed": args.seed,
            "labels": prep.labels,
            "label_fields": dict(zip(("y", "target"), LABEL_FIELDS[prep.labels])),
            "epoch": epoch,
            "val_nll": val_nll,
            "standardizer": prep.standardizer.state_dict(),
            "player_standardizer": None if prep.player_standardizer is None else prep.player_standardizer.state_dict(),
            "feature_names": ref.feature_names,
            "keep_columns": ref.keep,
            "cohort": args.cohort,
            **(
                {}
                if prep.extra_standardizer is None
                else {"extra_standardizer": prep.extra_standardizer.state_dict(), "extra_feature_names": list(ref.extra_names)}
            ),
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[PTRFormer, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = PTRFormer(PTRConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


# --------------------------------------------------------------------------- run
def run(args: argparse.Namespace) -> dict:
    threads_before = torch.get_num_threads()
    if args.threads:
        torch.set_num_threads(args.threads)
    try:
        return _run(args)
    finally:
        torch.set_num_threads(threads_before)


def _run(args: argparse.Namespace) -> dict:
    t_start = time.perf_counter()
    device = _device(args.device)
    prep = prepare(args)
    out = Path(args.out_dir) / args.cohort / args.arm / run_dir_name(args.seed, args.train_subsample, _subsample_seed(args))
    check_run_dir(out, args.labels)  # never overwrite the results of the other label choice
    out.mkdir(parents=True, exist_ok=True)
    removal = prep.datasets["train"].removal
    if removal["n_removed"]:
        logger.info(
            "%s removes %d columns (groups %s), kept by exception %s: %s",
            args.arm, removal["n_removed"], removal["drop_groups"], removal["kept_by_exception"],
            removal["removed_columns"],
        )  # fmt: skip
    fit_res = fit(prep, args, device, ckpt_path=out / "best.pt")
    model = fit_res.model
    rank_head_trained = args.rank_weight > 0
    result: dict = {
        "arm": args.arm,
        "cohort": args.cohort,
        "seed": args.seed,
        "device": args.device,
        "labels": args.labels,
        "label_fields": dict(zip(("y", "target"), LABEL_FIELDS[args.labels])),
        "config_hash": config_hash(args),
        "data": prep.data,
        "arm_spec": asdict(prep.arm),
        "config": asdict(prep.config),
        "train_config": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "rank_weight": args.rank_weight,
            "rank_reduction": args.rank_reduction,
            "max_epochs": args.max_epochs,
            "pos_feature": not args.no_pos_feature,
        },
        "rank_score_source": "rank_head" if rank_head_trained else "-hazard_logit (rank_weight 0: the rank head is untrained)",
        "n_params": model.n_parameters(),
        "n_input_features": prep.datasets["train"].n_features,
        "feature_removal": prep.datasets["train"].removal,
        "relations": {
            "combat_weights": prep.combat_weights,
            "combat_weight_max_train": prep.combat_weight_max,
            "relation_types": list(prep.config.relation_types) if prep.arm.relation else [],
        },
        **({} if prep.extra_info is None else {"extra_features": prep.extra_info}),
        "n_matches": {k: len(v) for k, v in prep.datasets.items()},
        "epochs_run": len(fit_res.history),
        "best_epoch": fit_res.best_epoch,
        "best_val_nll": fit_res.best_val_nll,
        "stopped_early": fit_res.stopped_early,
        "train_sec": fit_res.train_sec,
        "history": fit_res.history,
        "git": git_info(),
    }
    for split in ("val", "test"):
        df = predict(model, prep.datasets[split], device, args.batch_size, rank_from_hazard=not rank_head_trained)
        M.write_predictions(df, out / f"pred_{split}.csv")
        result[split] = M.all_metrics(df)
    result["elapsed_sec"] = time.perf_counter() - t_start
    write_json(out / "metrics.json", result)
    logger.info(
        "%s %s seed=%d  val_nll=%.4f  test AUC=%.4f  elapsed %.1fs",
        args.cohort, args.arm, args.seed, fit_res.best_val_nll,
        result["test"]["hazard"]["auc"], result["elapsed_sec"],
    )  # fmt: skip
    return result


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
