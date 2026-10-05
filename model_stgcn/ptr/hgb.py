"""HGB reference baselines built from the same dump (identical rows to the neural arms).

Rows: every ``(match, team, phase)`` of the risk set of the chosen label set (``labels='v2'`` default, or
``'pipeline'``; the same choice as the neural arms, so the rows are identical), ordered by
``(match_id, team_id, phase)`` (design doc section 6.2).  Features are the raw, un-standardised ``X``
(trees are scale-free and ``HistGradientBoostingClassifier`` handles NaN natively) plus the team-level
``team_ai_frac`` (raw share of AI accounts, NaN = no participants, required in the dump):

``HGB-cut``    ``[X[p] | team_ai_frac]``  (the F phase-tier columns + the AI share)
``HGB-lag``    ``[X[p] | team_ai_frac | X[p-1] | X[p] - X[p-1]]``.  Only ``p - 1`` is read.  The AI share is a
               team property and is not lagged.  When there is no previous record (first cut, ``p == 0``, or
               no alive token (the neural arms' alive rule, see ``data.effective_alive``) at ``p - 1``) the lag
               and difference blocks are **NaN**
               (design doc v2.1, section 4.1), not 0.
``HGB-noHab``  ``HGB-cut`` without the ``zone`` and ``habitat`` feature groups, except the alive-count
               variables inside them (``arms.NOHAB_KEEP``, design doc v2.2); the removed column names are
               written to ``metrics.json`` (``feature_removal``)

Classifier: ``HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, l2_regularization=0.01,
random_state=seed, early_stopping=False)`` - the iteration / learning-rate / L2 settings of
``model_stgcn/feature_ablation.py:_classifier`` with early stopping off.

Predictions use the neural CSV format (``metrics.PRED_COLUMNS``).  A tree model has no rank head, so
``rank_score = -p_hazard``: ``rank_score`` is survival-like everywhere (higher = better survival, design doc
v2.2), so the rank metrics are then "ranking by hazard".

    python -m model_stgcn.ptr.hgb --dump-dir ... --cohort ... --variant HGB-lag --seed 42 --out-dir ...
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from model_stgcn.ptr import metrics as M
from model_stgcn.ptr.arms import NOHAB_GROUPS, NOHAB_KEEP
from model_stgcn.ptr.data import (
    AI_FEATURE_NAME,
    DEFAULT_LABELS,
    DEFAULT_MATCH_TYPES,
    LABEL_CHOICES,
    LABEL_FIELDS,
    alive_source,
    effective_alive,
    event_alive_summary,
    feature_removal_report,
    keep_columns,
    load_cohort,
    select_labels,
    subsample_dump,
    winner_censor_summary,
    uses_event_alive,
    validate_dump,
)
from model_stgcn.ptr.util import (
    check_run_dir,
    code_fingerprint,
    config_hash as _hash,
    dump_fingerprint,
    git_info,
    run_dir_name,
    write_json,
)

logger = logging.getLogger("model_stgcn.ptr.hgb")

HGB_VARIANTS: tuple[str, ...] = ("HGB-cut", "HGB-lag", "HGB-noHab")


@dataclass
class Rows:
    keys: pd.DataFrame  # match_id, team_id, phase
    X: np.ndarray  # float32 [n, D]
    y: np.ndarray
    final_rank_norm: np.ndarray
    top5: np.ndarray
    match_type: np.ndarray  # str per row
    feature_names: list[str]
    removal: dict  # feature_removal_report of the variant (empty lists unless HGB-noHab)


def make_classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.06,
        l2_regularization=0.01,
        random_state=seed,
        early_stopping=False,
    )


def build_rows(dump: dict, variant: str, labels: str = DEFAULT_LABELS) -> Rows:
    if variant not in HGB_VARIANTS:
        raise ValueError(f"unknown HGB variant '{variant}'; known: {HGB_VARIANTS}")
    validate_dump(dump, labels)
    meta = dump["meta"]
    event = uses_event_alive(dump, labels)  # ghost fix: event-based alive mask for v2 on an event-based dump
    names = list(meta["feature_names"])
    drop, spare = (NOHAB_GROUPS, NOHAB_KEEP) if variant == "HGB-noHab" else ((), ())
    removal = feature_removal_report(meta, drop, spare)
    keep = keep_columns(meta, drop, spare) if variant == "HGB-noHab" else None
    if keep is not None:
        names = [names[int(c)] for c in keep]
    features = list(names)
    names = features + [AI_FEATURE_NAME]
    if variant == "HGB-lag":
        names = names + [f"lag1:{n}" for n in features] + [f"diff1:{n}" for n in features]

    blocks, keys, ys, frn, top5, match_types = [], [], [], [], [], []
    for m in sorted(dump["matches"], key=lambda mm: str(mm["match_id"])):
        X = np.asarray(m["X"], dtype=np.float32)
        y_all, target_t = select_labels(m, labels)
        alive = effective_alive(m, labels, event).numpy()  # the alive tokens of the neural arms (same rule)
        target = target_t.numpy()
        p_idx, i_idx = np.nonzero(target)
        team_ids = np.asarray(m["team_ids"], dtype=np.int64)
        order = np.lexsort((p_idx, team_ids[i_idx]))  # (team_id, phase) within the match
        p_idx, i_idx = p_idx[order], i_idx[order]
        if keep is not None:
            X = X[..., keep]
        cur = X[p_idx, i_idx]
        share = np.asarray(m["team_ai_frac"], dtype=np.float32)[i_idx][:, None]  # team property, never lagged
        if variant == "HGB-lag":
            lag = np.full_like(cur, np.nan)
            prev = np.maximum(p_idx - 1, 0)
            has_prev = (p_idx >= 1) & alive[prev, i_idx]
            lag[has_prev] = X[prev[has_prev], i_idx[has_prev]]
            cur = np.concatenate([cur, share, lag, cur - lag], axis=1)  # diff is NaN wherever lag is
        else:
            cur = np.concatenate([cur, share], axis=1)
        blocks.append(cur)
        keys.append(
            pd.DataFrame(
                {
                    "match_id": str(m["match_id"]),
                    "team_id": team_ids[i_idx],
                    "phase": p_idx.astype(np.int64) + M.PHASE_OFFSET,  # cut phase k = slot p + 1
                }
            )
        )
        ys.append(y_all.numpy()[p_idx, i_idx])
        frn.append(np.asarray(m["final_rank_norm"], dtype=np.float32)[i_idx])
        top5.append(np.asarray(m["top5"], dtype=np.float32)[i_idx])
        match_types.append(np.full(len(p_idx), str(m["match_type"]), dtype=object))
    return Rows(
        keys=pd.concat(keys, ignore_index=True),
        X=np.concatenate(blocks, axis=0),
        y=np.concatenate(ys),
        final_rank_norm=np.concatenate(frn),
        top5=np.concatenate(top5),
        match_type=np.concatenate(match_types),
        feature_names=names,
        removal=removal,
    )


def _prediction_frame(rows: Rows, p: np.ndarray) -> pd.DataFrame:
    df = rows.keys.copy()
    df["y"] = rows.y
    df["p_hazard"] = p
    df["rank_score"] = -p  # survival-like like the neural rank head (-p keeps full precision in the CSV)
    df["final_rank_norm"] = rows.final_rank_norm
    df["top5"] = rows.top5
    df["match_type"] = rows.match_type
    return df


def run_config_hash(dump_dir, cohort, variant, seed, labels, match_types, subsample=None, subsample_seed=None) -> str:
    """Hash of everything that can change an HGB run's result (the classifier settings live in the code fingerprint)."""
    spec = {
        "kind": "hgb", "variant": variant, "cohort": cohort, "seed": seed, "labels": labels,
        "match_types": None if match_types is None else list(match_types),
        "dump": dump_fingerprint(dump_dir, cohort), "code": code_fingerprint(),
    }  # fmt: skip
    if subsample is not None:  # only subsampled runs carry the fields: full-size hashes do not depend on the feature
        spec.update(train_subsample=subsample, subsample_seed=seed if subsample_seed is None else subsample_seed)
    return _hash(spec)


def config_hash(args: argparse.Namespace) -> str:
    """Hash stamped into ``metrics.json``; the runner skips a run whose stamp equals this value."""
    return run_config_hash(args.dump_dir, args.cohort, args.variant, args.seed, args.labels, args.match_types,
                           args.train_subsample, args.subsample_seed)  # fmt: skip


def run(
    dump_dir,
    cohort: str,
    variant: str,
    seed: int,
    out_dir,
    labels: str = DEFAULT_LABELS,
    match_types: tuple[str, ...] | None = DEFAULT_MATCH_TYPES,
    threads: int | None = None,
    subsample: int | None = None,
    subsample_seed: int | None = None,
) -> dict:
    """``subsample`` / ``subsample_seed``: train on a random subsample of that many train matches (see
    :func:`model_stgcn.ptr.data.subsample_dump`); val and test are unchanged."""
    if threads:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=threads):
            return _run(dump_dir, cohort, variant, seed, out_dir, labels, match_types, subsample, subsample_seed)
    return _run(dump_dir, cohort, variant, seed, out_dir, labels, match_types, subsample, subsample_seed)


def _run(dump_dir, cohort, variant, seed, out_dir, labels, match_types, subsample=None, subsample_seed=None) -> dict:
    t0 = time.perf_counter()
    splits = load_cohort(dump_dir, cohort, labels=labels, match_types=match_types)
    sub_seed = seed if subsample_seed is None else subsample_seed
    splits["train"], sub_info = subsample_dump(splits["train"], subsample, sub_seed)
    if sub_info is not None:
        logger.info("train subsample: %s", sub_info)
    rows = {s: build_rows(splits[s], variant, labels) for s in splits}
    train = rows["train"]
    if len(np.unique(train.y)) < 2:
        raise ValueError(f"{cohort}: the train rows hold a single hazard class; cannot fit {variant}")
    clf = make_classifier(seed)
    t_fit = time.perf_counter()
    clf.fit(train.X, train.y)
    fit_sec = time.perf_counter() - t_fit

    logger.info("alive tokens: %s (the rule is used for the lag presence of HGB-lag)", alive_source(splits["train"], labels))
    out = Path(out_dir) / cohort / variant / run_dir_name(seed, subsample, sub_seed)
    check_run_dir(out, labels)  # never overwrite the results of the other label choice
    result: dict = {
        "variant": variant,
        "cohort": cohort,
        "seed": seed,
        "labels": labels,
        "label_fields": dict(zip(("y", "target"), LABEL_FIELDS[labels])),
        "config_hash": run_config_hash(dump_dir, cohort, variant, seed, labels, match_types, subsample, subsample_seed),
        "train_subsample": sub_info,
        "n_matches": {s: len(splits[s]["matches"]) for s in splits},
        "alive_mask": {s: event_alive_summary(splits[s], labels) for s in splits},
        "winner_censor": {s: winner_censor_summary(splits[s], labels) for s in splits},
        "match_types_allowed": None if match_types is None else list(match_types),
        "n_features": int(train.X.shape[1]),
        "n_train_rows": int(len(train.y)),
        "n_iter": int(clf.n_iter_),
        "feature_removal": train.removal,
        "classifier": clf.get_params(),
        "fit_sec": fit_sec,
        "git": git_info(),
    }
    for split in ("val", "test"):
        p = clf.predict_proba(rows[split].X)[:, 1]
        df = _prediction_frame(rows[split], p)
        M.write_predictions(df, out / f"pred_{split}.csv")
        result[split] = M.all_metrics(df)
    result["elapsed_sec"] = time.perf_counter() - t0
    write_json(out / "metrics.json", result)
    if train.removal["n_removed"]:
        logger.info("%s removed %d columns: %s; kept by exception: %s", variant, train.removal["n_removed"],
                    train.removal["removed_columns"], train.removal["kept_by_exception"])  # fmt: skip
    logger.info("%s %s seed=%d test AUC=%.4f (%.1fs)", cohort, variant, seed,
                result["test"]["hazard"]["auc"], result["elapsed_sec"])  # fmt: skip
    return result


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
    ap = argparse.ArgumentParser(prog="python -m model_stgcn.ptr.hgb", description=__doc__.split("\n")[0])
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--variant", required=True, choices=HGB_VARIANTS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--labels",
        default=DEFAULT_LABELS,
        choices=LABEL_CHOICES,
        help="label set for the rows and the labels (v2 = event based, pipeline = team_phase_labels)",
    )
    ap.add_argument(
        "--match-types",
        default=",".join(DEFAULT_MATCH_TYPES),
        type=_parse_match_types,
        help="allowed match_type values, comma separated ('any' = no check); default official,competitive",
    )
    ap.add_argument(
        "--threads",
        type=_positive_int,
        default=None,
        help="OpenMP threads for the boosting (never changes a result, only the speed)",
    )
    ap.add_argument(
        "--train-subsample",
        type=_positive_int,
        default=None,
        metavar="N",
        help="train on a random subsample of N train matches (learning curve, design doc 4.4); val and test are "
        "unchanged.  Results go to <cohort>/<variant>/sub<N>_ss<s>_seed<seed>/",
    )
    ap.add_argument(
        "--subsample-seed",
        type=int,
        default=None,
        help="seed of the subsample (default: --seed); subsamples of one seed are nested across N",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.dump_dir, args.cohort, args.variant, args.seed, args.out_dir, args.labels, args.match_types, args.threads,
        args.train_subsample, args.subsample_seed)  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
