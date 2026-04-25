#!/usr/bin/env python3
"""Train/evaluate a candidate ranker using per-view TTT metadata aggregates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from meta_prediction_diagnostic import aggregate_candidates, load_meta, load_predictions
from physics_reranker import (
    as_array,
    candidate_complexity,
    component_count,
    entropy,
    foreground_fraction,
    invalid_token_fraction,
    relation_features,
    serialize,
)
from prediction_meta_ranker_cv import Group, evaluate_groups, train_ranker
from rerank_varc_predictions import load_json, task_files


def candidate_features(
    demos_mean: np.ndarray,
    demos_std: np.ndarray,
    test_ex: dict[str, Any],
    candidate: dict[str, Any],
    rank_index: int,
    max_votes: float,
    total_votes: float,
    candidate_count: int,
) -> np.ndarray:
    rel = relation_features(test_ex["input"], candidate["prediction"]).astype(np.float32)
    z = (np.abs(rel - demos_mean) / demos_std).astype(np.float32)
    y = as_array(candidate["prediction"])
    x = as_array(test_ex["input"])
    votes = float(candidate.get("votes", 0.0))
    vote_share = votes / max(total_votes, 1.0)
    max_vote_share = votes / max(max_votes, 1.0)
    rank_norm = rank_index / max(candidate_count - 1, 1)
    meta = np.asarray(
        [
            math.log1p(votes),
            vote_share,
            max_vote_share,
            rank_norm,
            float(candidate.get("confidence_mean", 0.0)),
            float(candidate.get("confidence_min", 0.0)),
            float(candidate.get("margin_mean", 0.0)),
            float(candidate.get("entropy_mean", 0.0)),
            float(candidate.get("attempt_support", 0.0)) / 10.0,
            float(candidate.get("augmenter_support", 0.0)) / 8.0,
            float(candidate.get("color_support", 0.0)) / 10.0,
            float(candidate.get("scale_support", 0.0)) / 8.0,
            float(candidate.get("border_ok_fraction", 0.0)),
            math.log1p(candidate_count),
            candidate_complexity(candidate["prediction"]),
            entropy(y),
            foreground_fraction(y),
            min(component_count(y), 50) / 50.0,
            invalid_token_fraction(y),
            float(y.shape == x.shape),
            math.log((y.size + 1.0) / (x.size + 1.0)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([rel, demos_mean, demos_std, z, meta], axis=0).astype(np.float32)


def build_groups(
    data_root: Path,
    split: str,
    prediction_root: Path,
    limit: int | None,
    candidate_limit: int | None,
    progress_every: int,
) -> list[Group]:
    groups: list[Group] = []
    files = task_files(data_root, split, limit)
    for task_i, path in enumerate(files, 1):
        task_name = path.stem
        predictions = load_predictions(prediction_root, task_name)
        metadata = load_meta(prediction_root, task_name)
        if predictions is None or metadata is None:
            continue
        task = load_json(path)
        demo_rel = np.stack(
            [relation_features(ex["input"], ex["output"]) for ex in task.get("train", [])],
            axis=0,
        ).astype(np.float32)
        demos_mean = demo_rel.mean(axis=0)
        demos_std = demo_rel.std(axis=0) + 0.15
        test_examples = task.get("test", [])
        weight = 1.0 / max(len(test_examples), 1)
        for idx, test_ex in enumerate(test_examples):
            if "output" not in test_ex:
                continue
            key = str(idx)
            candidates = aggregate_candidates(predictions.get(key, []), metadata.get(key, []))
            candidates.sort(key=lambda item: item["votes"], reverse=True)
            if candidate_limit is not None:
                candidates = candidates[:candidate_limit]
            if not candidates:
                continue
            max_votes = max(float(candidate["votes"]) for candidate in candidates)
            total_votes = sum(float(candidate["votes"]) for candidate in candidates)
            features = np.stack(
                [
                    candidate_features(
                        demos_mean,
                        demos_std,
                        test_ex,
                        candidate,
                        rank,
                        max_votes,
                        total_votes,
                        len(candidates),
                    )
                    for rank, candidate in enumerate(candidates)
                ],
                axis=0,
            )
            truth = serialize(test_ex["output"])
            labels = np.asarray(
                [1.0 if serialize(candidate["prediction"]) == truth else 0.0 for candidate in candidates],
                dtype=np.float32,
            )
            entries = [
                {
                    "prediction": candidate["prediction"],
                    "votes": candidate["votes"],
                    "vote_signal": float(candidate["votes"]),
                }
                for candidate in candidates
            ]
            groups.append(Group(task_name, idx, task, test_ex, entries, features, labels, weight))
        if progress_every and task_i % progress_every == 0:
            print(f"built metadata features for {task_i}/{len(files)} tasks, groups={len(groups)}", flush=True)
    return groups


def split_fit_val(groups: list[Group], val_fraction: float, seed: int) -> tuple[list[Group], list[Group]]:
    by_task: dict[str, list[Group]] = {}
    for group in groups:
        by_task.setdefault(group.task_name, []).append(group)
    tasks = sorted(by_task)
    rng = np.random.default_rng(seed)
    rng.shuffle(tasks)
    val_count = max(1, int(round(len(tasks) * val_fraction))) if len(tasks) > 1 else 0
    val_tasks = set(tasks[:val_count])
    return (
        [group for group in groups if group.task_name not in val_tasks],
        [group for group in groups if group.task_name in val_tasks],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-root", type=Path, default=Path("raw_data/ARC-AGI"))
    parser.add_argument("--eval-data-root", type=Path, default=Path("raw_data/ARC-AGI"))
    parser.add_argument("--train-split", default="training")
    parser.add_argument("--eval-split", default="evaluation")
    parser.add_argument("--train-prediction-root", type=Path, required=True)
    parser.add_argument("--eval-prediction-root", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--candidate-limit", type=int, default=256)
    parser.add_argument("--train-val-fraction", type=float, default=0.2)
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
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_groups = build_groups(
        args.train_data_root,
        args.train_split,
        args.train_prediction_root,
        args.train_limit,
        args.candidate_limit,
        args.progress_every,
    )
    eval_groups = build_groups(
        args.eval_data_root,
        args.eval_split,
        args.eval_prediction_root,
        args.eval_limit,
        args.candidate_limit,
        args.progress_every,
    )
    fit_groups, val_groups = split_fit_val(train_groups, args.train_val_fraction, args.seed)
    if not fit_groups or not val_groups:
        raise SystemExit("Need train and validation metadata groups.")
    model, standardizer, train_info = train_ranker(fit_groups, val_groups, args, device)
    train_val_eval = evaluate_groups(model, standardizer, val_groups, args.vote_weights, device)
    eval_info = evaluate_groups(model, standardizer, eval_groups, args.vote_weights, device)
    summary = {
        "device": str(device),
        "candidate_limit": args.candidate_limit,
        "train": {
            "groups": len(train_groups),
            "tasks": len({group.task_name for group in train_groups}),
            "fit_groups": len(fit_groups),
            "val_groups": len(val_groups),
        },
        "eval": {
            "groups": len(eval_groups),
            "tasks": len({group.task_name for group in eval_groups}),
        },
        "train_info": train_info,
        "train_val_eval": train_val_eval,
        "eval_info": eval_info,
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")


if __name__ == "__main__":
    main()
