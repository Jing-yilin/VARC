#!/usr/bin/env python3
"""Train a candidate meta-ranker on one ARC split and evaluate on another."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from prediction_meta_ranker_cv import (
    Group,
    build_groups,
    evaluate_groups,
    split_folds,
    train_ranker,
)


def make_group_args(
    *,
    data_root: Path,
    split: str,
    output_roots: str,
    vote_mode: str,
    candidate_limit: int | None,
    limit: int | None,
    progress_every: int,
) -> argparse.Namespace:
    return argparse.Namespace(
        data_root=data_root,
        split=split,
        output_roots=output_roots,
        vote_mode=vote_mode,
        candidate_limit=candidate_limit,
        limit=limit,
        progress_every=progress_every,
    )


def task_split(groups: list[Group], val_fraction: float, seed: int) -> tuple[list[Group], list[Group]]:
    by_task: dict[str, list[Group]] = {}
    for group in groups:
        by_task.setdefault(group.task_name, []).append(group)
    tasks = sorted(by_task)
    random.Random(seed).shuffle(tasks)
    val_count = max(1, int(round(len(tasks) * val_fraction))) if len(tasks) > 1 else 0
    val_tasks = set(tasks[:val_count])
    train = [group for group in groups if group.task_name not in val_tasks]
    val = [group for group in groups if group.task_name in val_tasks]
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-root", type=Path, default=Path("raw_data/ARC-AGI"))
    parser.add_argument("--eval-data-root", type=Path, default=Path("raw_data/ARC-AGI"))
    parser.add_argument("--train-split", default="training")
    parser.add_argument("--eval-split", default="evaluation")
    parser.add_argument("--train-output-roots", required=True)
    parser.add_argument("--eval-output-roots", required=True)
    parser.add_argument("--vote-mode", choices=("majority", "source_norm"), default="majority")
    parser.add_argument("--candidate-limit", type=int, default=256)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--train-val-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument(
        "--vote-weights",
        type=float,
        nargs="+",
        default=[0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5],
    )
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_group_args = make_group_args(
        data_root=args.train_data_root,
        split=args.train_split,
        output_roots=args.train_output_roots,
        vote_mode=args.vote_mode,
        candidate_limit=args.candidate_limit,
        limit=args.train_limit,
        progress_every=args.progress_every,
    )
    eval_group_args = make_group_args(
        data_root=args.eval_data_root,
        split=args.eval_split,
        output_roots=args.eval_output_roots,
        vote_mode=args.vote_mode,
        candidate_limit=args.candidate_limit,
        limit=args.eval_limit,
        progress_every=args.progress_every,
    )

    train_groups = build_groups(train_group_args)
    eval_groups = build_groups(eval_group_args)
    train_fit_groups, train_val_groups = task_split(
        train_groups,
        args.train_val_fraction,
        args.seed,
    )
    if not train_fit_groups or not train_val_groups:
        raise SystemExit("Need at least one train and validation task group.")

    model, standardizer, train_info = train_ranker(
        train_fit_groups,
        train_val_groups,
        args,
        device,
    )
    train_val_eval = evaluate_groups(model, standardizer, train_val_groups, args.vote_weights, device)
    eval_info = evaluate_groups(model, standardizer, eval_groups, args.vote_weights, device)

    summary: dict[str, Any] = {
        "device": str(device),
        "vote_mode": args.vote_mode,
        "candidate_limit": args.candidate_limit,
        "train": {
            "split": args.train_split,
            "groups": len(train_groups),
            "fit_groups": len(train_fit_groups),
            "val_groups": len(train_val_groups),
            "tasks": len({group.task_name for group in train_groups}),
            "output_roots": args.train_output_roots,
        },
        "eval": {
            "split": args.eval_split,
            "groups": len(eval_groups),
            "tasks": len({group.task_name for group in eval_groups}),
            "output_roots": args.eval_output_roots,
        },
        "train_info": train_info,
        "train_val_eval": train_val_eval,
        "eval_info": eval_info,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
