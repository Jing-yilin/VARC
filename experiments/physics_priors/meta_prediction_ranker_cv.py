#!/usr/bin/env python3
"""Cross-validate a ranker over per-view TTT metadata prediction pools.

This is a diagnostic experiment: labels come from the evaluated split, but
folds are split by task. Use it to test whether metadata features contain a
learnable signal before spending GPU time on larger supervised selector data.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from meta_prediction_ranker_train_eval import build_groups
from prediction_meta_ranker_cv import (
    evaluate_groups,
    init_metric,
    train_ranker,
)


def split_folds(groups: list[Any], folds: int, seed: int) -> list[tuple[list[Any], list[Any]]]:
    by_task: dict[str, list[Any]] = {}
    for group in groups:
        by_task.setdefault(group.task_name, []).append(group)
    tasks = sorted(by_task)
    random.Random(seed).shuffle(tasks)
    out = []
    for fold in range(folds):
        val_tasks = set(tasks[fold::folds])
        train = [group for group in groups if group.task_name not in val_tasks]
        val = [group for group in groups if group.task_name in val_tasks]
        out.append((train, val))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("raw_data/ARC-AGI"))
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--candidate-limit", type=int, default=256)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument(
        "--vote-weights",
        type=float,
        nargs="+",
        default=[0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3],
    )
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    groups = build_groups(
        args.data_root,
        args.split,
        args.prediction_root,
        args.limit,
        args.candidate_limit,
        args.progress_every,
    )
    folds = split_folds(groups, args.folds, args.seed)
    fold_summaries = []
    total_metrics: dict[str, dict[str, float]] = {}
    task_total = 0
    for fold_idx, (train_groups, val_groups) in enumerate(folds):
        print(f"fold {fold_idx + 1}/{len(folds)} train={len(train_groups)} val={len(val_groups)}", flush=True)
        model, standardizer, train_info = train_ranker(train_groups, val_groups, args, device)
        eval_info = evaluate_groups(model, standardizer, val_groups, args.vote_weights, device)
        fold_summaries.append({"fold": fold_idx, "train": train_info, "eval": eval_info})
        task_count = eval_info["tasks_evaluated"]
        task_total += task_count
        for name, metric in eval_info["metrics"].items():
            total_metrics.setdefault(name, init_metric())
            for key in ("pass_at_1", "pass_at_2", "oracle"):
                total_metrics[name][key] += metric[key] * task_count
            total_metrics[name]["examples"] += metric["examples"]

    metrics = {
        name: {
            "pass_at_1": metric["pass_at_1"] / max(task_total, 1),
            "pass_at_2": metric["pass_at_2"] / max(task_total, 1),
            "oracle": metric["oracle"] / max(task_total, 1),
            "examples": metric["examples"],
        }
        for name, metric in total_metrics.items()
    }
    best_name, best_metric = max(metrics.items(), key=lambda item: (item[1]["pass_at_2"], item[1]["pass_at_1"]))
    summary = {
        "device": str(device),
        "groups": len(groups),
        "tasks": len({group.task_name for group in groups}),
        "candidate_limit": args.candidate_limit,
        "folds": args.folds,
        "best": {"strategy": best_name, **best_metric},
        "metrics": metrics,
        "fold_summaries": fold_summaries,
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")


if __name__ == "__main__":
    main()
