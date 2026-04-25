#!/usr/bin/env python3
"""Cross-validate a meta-ranker over official VARC prediction pools.

This is a diagnostic experiment, not a fair final benchmark: it uses labels
from the evaluated split, but keeps task-level folds disjoint. The point is to
measure whether the true candidate is learnably identifiable from candidate
statistics once it is present in VARC's prediction pool.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from physics_reranker import (
    MODEL_TOKEN_COUNT,
    as_array,
    candidate_complexity,
    component_count,
    entropy,
    foreground_fraction,
    invalid_token_fraction,
    relation_features,
    serialize,
)
from rerank_varc_predictions import (
    entries_for_index,
    load_json,
    load_prediction_sources,
    normalize_index,
    pass_metrics_for_order,
    task_files,
)


@dataclass
class Group:
    task_name: str
    test_index: int
    task: dict
    test_ex: dict
    entries: list[dict]
    features: np.ndarray
    labels: np.ndarray
    weight: float


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


class Ranker(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        if hidden <= 0:
            self.net = nn.Linear(dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def candidate_feature(
    demos_mean: np.ndarray,
    demos_std: np.ndarray,
    test_ex: dict,
    entry: dict,
    rank_index: int,
    max_votes: float,
    total_votes: float,
    candidate_count: int,
) -> np.ndarray:
    votes = float(entry.get("votes", 0.0))
    vote_signal = float(entry.get("vote_signal", votes))
    source_hits = float(entry.get("source_hits", 1.0))
    rel = relation_features(test_ex["input"], entry["prediction"]).astype(np.float32)
    z = (np.abs(rel - demos_mean) / demos_std).astype(np.float32)
    y = as_array(entry["prediction"])
    x = as_array(test_ex["input"])
    complexity = candidate_complexity(entry["prediction"])
    relation_weights = np.asarray(
        [1.2, 1.2, 0.9, 1.5, 0.7, 0.8, 1.1, 0.9, 1.0, 0.7, 0.7, 4.0],
        dtype=np.float32,
    )
    energy = float((z * relation_weights).sum() + complexity)
    rank_norm = rank_index / max(candidate_count - 1, 1)
    vote_share = votes / max(total_votes, 1.0)
    max_vote_share = votes / max(max_votes, 1.0)
    base = np.concatenate(
        [
            rel,
            demos_mean,
            demos_std,
            z,
            np.asarray(
                [
                    complexity,
                    entropy(y),
                    foreground_fraction(y),
                    min(component_count(y), 50) / 50.0,
                    invalid_token_fraction(y),
                    float(y.shape == x.shape),
                    math.log((y.size + 1.0) / (x.size + 1.0)),
                ],
                dtype=np.float32,
            ),
        ],
        axis=0,
    )
    extras = np.asarray(
        [
            math.log1p(votes),
            math.log1p(vote_signal),
            source_hits / 8.0,
            vote_share,
            max_vote_share,
            rank_norm,
            energy,
            energy - 1.75 * math.log1p(vote_signal),
            math.log1p(candidate_count),
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, extras], axis=0).astype(np.float32)


def build_groups(args: argparse.Namespace) -> list[Group]:
    groups: list[Group] = []
    files = task_files(args.data_root, args.split, args.limit)
    for task_i, path in enumerate(files, 1):
        task_name = path.stem
        sources = load_prediction_sources(args.output_roots, task_name)
        if sources is None:
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
            entries = entries_for_index(sources, normalize_index(idx), args.vote_mode)
            if args.candidate_limit is not None:
                entries = entries[: args.candidate_limit]
            if not entries:
                continue
            max_votes = max(float(entry.get("votes", 0.0)) for entry in entries)
            total_votes = sum(float(entry.get("votes", 0.0)) for entry in entries)
            features = np.stack(
                [
                    candidate_feature(demos_mean, demos_std, test_ex, entry, rank, max_votes, total_votes, len(entries))
                    for rank, entry in enumerate(entries)
                ],
                axis=0,
            )
            truth = serialize(test_ex["output"])
            labels = np.asarray(
                [1.0 if serialize(entry["prediction"]) == truth else 0.0 for entry in entries],
                dtype=np.float32,
            )
            groups.append(Group(task_name, idx, task, test_ex, entries, features, labels, weight))
        if args.progress_every and task_i % args.progress_every == 0:
            print(f"built features for {task_i}/{len(files)} tasks, groups={len(groups)}", flush=True)
    return groups


def split_folds(groups: list[Group], folds: int, seed: int) -> list[tuple[list[Group], list[Group]]]:
    by_task: dict[str, list[Group]] = {}
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


def fit_standardizer(groups: list[Group]) -> Standardizer:
    flat = np.concatenate([group.features for group in groups], axis=0)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return Standardizer(mean, std)


def group_tensors(
    groups: list[Group],
    standardizer: Standardizer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_candidates = max(group.features.shape[0] for group in groups)
    dim = groups[0].features.shape[1]
    x = np.zeros((len(groups), max_candidates, dim), dtype=np.float32)
    y = np.zeros((len(groups), max_candidates), dtype=np.float32)
    mask = np.zeros((len(groups), max_candidates), dtype=bool)
    for idx, group in enumerate(groups):
        n = group.features.shape[0]
        x[idx, :n] = standardizer.apply(group.features)
        y[idx, :n] = group.labels
        mask[idx, :n] = True
    return (
        torch.tensor(x, dtype=torch.float32, device=device),
        torch.tensor(y, dtype=torch.float32, device=device),
        torch.tensor(mask, dtype=torch.bool, device=device),
    )


def train_ranker(
    train_groups: list[Group],
    val_groups: list[Group],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Ranker, Standardizer, dict[str, Any]]:
    standardizer = fit_standardizer(train_groups)
    model = Ranker(train_groups[0].features.shape[1], args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_positive_groups = [
        group
        for group in train_groups
        if group.labels.max() > 0.5 and group.labels.min() < 0.5
    ]
    train_x, train_y, train_mask = group_tensors(train_positive_groups, standardizer, device)
    history = []
    best_state = None
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        order = torch.randperm(train_x.shape[0], device=device)
        for start in range(0, train_x.shape[0], args.batch_size):
            idx = order[start : start + args.batch_size]
            x = train_x[idx]
            y = train_y[idx]
            mask = train_mask[idx]
            batch, candidates, dim = x.shape
            scores = model(x.reshape(batch * candidates, dim)).reshape(batch, candidates)
            scores = scores.masked_fill(~mask, -1.0e9)
            pos_mask = (y > 0.5) & mask
            pos_scores = scores.masked_fill(~pos_mask, -1.0e9)
            loss = -(torch.logsumexp(pos_scores, dim=1) - torch.logsumexp(scores, dim=1)).mean()
            if args.margin > 0:
                best_pos = pos_scores.max(dim=1).values
                neg_margin = F.relu(args.margin - best_pos[:, None] + scores)
                neg_margin = neg_margin.masked_fill((~mask) | pos_mask, 0.0)
                neg_count = ((~pos_mask) & mask).float().sum().clamp_min(1.0)
                loss = loss + neg_margin.sum() / neg_count
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * batch
            seen += batch
        should_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
        if should_eval:
            metrics = evaluate_groups(model, standardizer, val_groups, args.vote_weights, device)
            best = metrics["best"]
        else:
            best = history[-1] if history else {"pass_at_1": 0.0, "pass_at_2": 0.0, "strategy": "not_evaluated"}
        score = best["pass_at_2"] + 0.1 * best["pass_at_1"]
        history.append({"epoch": epoch, "loss": loss_sum / max(seen, 1), **best})
        if should_eval and score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if args.log_every and (epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs):
            print(f"epoch {epoch}: {json.dumps(history[-1], sort_keys=True)}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, standardizer, {"train_positive_groups": len(train_positive_groups), "history": history}


def unique_order(order: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for entry in order:
        key = serialize(entry["prediction"])
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return out


def init_metric() -> dict[str, float]:
    return {"pass_at_1": 0.0, "pass_at_2": 0.0, "oracle": 0.0, "examples": 0.0}


def add_metric(metric: dict[str, float], ordered: list[dict], truth: list[list[int]], weight: float) -> None:
    p1, p2, rank = pass_metrics_for_order(ordered, truth)
    metric["pass_at_1"] += weight if p1 else 0.0
    metric["pass_at_2"] += weight if p2 else 0.0
    metric["oracle"] += weight if rank is not None else 0.0
    metric["examples"] += 1.0


def merge_metric(total: dict[str, float], delta: dict[str, float]) -> None:
    for key, value in delta.items():
        total[key] = total.get(key, 0.0) + value


def finalize(metric: dict[str, float], task_count: int) -> dict[str, float]:
    denom = max(float(task_count), 1.0)
    return {
        "pass_at_1": metric["pass_at_1"] / denom,
        "pass_at_2": metric["pass_at_2"] / denom,
        "oracle": metric["oracle"] / denom,
        "examples": metric["examples"],
    }


@torch.no_grad()
def score_group(
    model: Ranker,
    standardizer: Standardizer,
    group: Group,
    device: torch.device,
) -> np.ndarray:
    x = torch.tensor(standardizer.apply(group.features), dtype=torch.float32, device=device)
    return model(x).detach().cpu().numpy()


def evaluate_groups(
    model: Ranker,
    standardizer: Standardizer,
    groups: list[Group],
    vote_weights: list[float],
    device: torch.device,
) -> dict[str, Any]:
    totals: dict[str, dict[str, float]] = {
        "majority": init_metric(),
        "learned": init_metric(),
        "majority_then_learned": init_metric(),
    }
    for weight in vote_weights:
        totals[f"learned_vote_{weight:g}"] = init_metric()
        totals[f"majority_then_learned_vote_{weight:g}"] = init_metric()

    previews = []
    for group in groups:
        scores = score_group(model, standardizer, group, device)
        learned_entries = [
            {
                **entry,
                "learned_score": float(score),
                "vote_log": math.log1p(float(entry.get("vote_signal", entry.get("votes", 0.0)))),
            }
            for entry, score in zip(group.entries, scores)
        ]
        majority = group.entries
        learned = sorted(learned_entries, key=lambda entry: entry["learned_score"], reverse=True)
        truth = group.test_ex["output"]
        add_metric(totals["majority"], majority, truth, group.weight)
        add_metric(totals["learned"], learned, truth, group.weight)
        add_metric(totals["majority_then_learned"], unique_order(majority[:1] + learned + majority[1:]), truth, group.weight)
        for vote_weight in vote_weights:
            voted = sorted(
                learned_entries,
                key=lambda entry: (
                    entry["learned_score"] + vote_weight * entry["vote_log"],
                    float(entry.get("vote_signal", entry.get("votes", 0.0))),
                    float(entry.get("votes", 0.0)),
                ),
                reverse=True,
            )
            add_metric(totals[f"learned_vote_{vote_weight:g}"], voted, truth, group.weight)
            anchored = unique_order(majority[:1] + voted + majority[1:])
            add_metric(totals[f"majority_then_learned_vote_{vote_weight:g}"], anchored, truth, group.weight)
        if len(previews) < 12:
            previews.append(
                {
                    "task": group.task_name,
                    "test_index": group.test_index,
                    "oracle": bool(group.labels.max() > 0.5),
                    "majority_top2": [
                        {"votes": item.get("votes", 0), "matches": item["prediction"] == truth}
                        for item in majority[:2]
                    ],
                    "learned_top2": [
                        {
                            "score": round(item["learned_score"], 4),
                            "votes": item.get("votes", 0),
                            "matches": item["prediction"] == truth,
                        }
                        for item in learned[:2]
                    ],
                }
            )

    task_count = len({group.task_name for group in groups})
    metrics = {name: finalize(metric, task_count) for name, metric in totals.items()}
    best_name, best_metric = max(metrics.items(), key=lambda item: (item[1]["pass_at_2"], item[1]["pass_at_1"]))
    return {
        "tasks_evaluated": task_count,
        "best": {"strategy": best_name, **best_metric},
        "metrics": metrics,
        "preview": previews,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("raw_data/ARC-AGI"))
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-roots", required=True)
    parser.add_argument("--vote-mode", choices=("majority", "source_norm"), default="majority")
    parser.add_argument("--candidate-limit", type=int, default=128)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--vote-weights", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5])
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

    groups = build_groups(args)
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
        "vote_mode": args.vote_mode,
        "candidate_limit": args.candidate_limit,
        "folds": args.folds,
        "best": {"strategy": best_name, **best_metric},
        "metrics": metrics,
        "fold_summaries": fold_summaries,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
