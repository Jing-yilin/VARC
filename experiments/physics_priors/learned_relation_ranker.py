#!/usr/bin/env python3
"""Train a lightweight relation-energy ranker on ARC training tasks.

The goal is generalization, not fitting the evaluation labels. Training uses
only ARC training tasks and synthetic candidate pools. The learned scorer is
then applied unchanged to official VARC prediction pools.
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
    candidate_complexity,
    candidate_pool,
    component_count,
    entropy,
    foreground_fraction,
    invalid_token_fraction,
    relation_features,
    serialize,
)
from rerank_varc_predictions import (
    get_majority_vote,
    load_json,
    load_prediction_roots,
    normalize_index,
    pass_metrics_for_order,
    task_files,
)


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
                nn.Linear(hidden, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def as_array(grid: list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim > 2:
        return arr.reshape(arr.shape[0], -1)
    return arr


def feature_vector(demos: list[dict], test_input: list[list[int]], candidate: list[list[int]]) -> np.ndarray:
    rel = relation_features(test_input, candidate)
    demo_rel = np.stack([relation_features(ex["input"], ex["output"]) for ex in demos], axis=0)
    mean = demo_rel.mean(axis=0)
    std = demo_rel.std(axis=0) + 0.15
    z = np.abs(rel - mean) / std
    y = as_array(candidate)
    x = as_array(test_input)
    extras = np.asarray(
        [
            candidate_complexity(candidate),
            entropy(y),
            foreground_fraction(y),
            min(component_count(y), 50) / 50.0,
            invalid_token_fraction(y),
            float(y.shape == x.shape),
            math.log((y.size + 1.0) / (x.size + 1.0)),
        ],
        dtype=np.float64,
    )
    return np.concatenate([rel, mean, std, z, extras], axis=0).astype(np.float32)


def iter_task_examples(data_root: Path, split: str, limit: int | None) -> list[tuple[str, dict, dict]]:
    out = []
    for path in task_files(data_root, split, limit):
        task = load_json(path)
        for test_ex in task.get("test", []):
            if "output" in test_ex:
                out.append((path.stem, task, test_ex))
    return out


def build_training_data(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(args.seed)
    features = []
    labels = []
    examples = iter_task_examples(args.train_data_root, args.train_split, args.train_limit)
    for _task_name, task, test_ex in examples:
        truth_key = serialize(test_ex["output"])
        pool = candidate_pool(task, test_ex, rng)
        for candidate in pool:
            features.append(feature_vector(task["train"], test_ex["input"], candidate.grid))
            labels.append(1.0 if serialize(candidate.grid) == truth_key else 0.0)
    return np.stack(features, axis=0), np.asarray(labels, dtype=np.float32)


def build_training_groups(args: argparse.Namespace) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = random.Random(args.seed)
    groups = []
    examples = iter_task_examples(args.train_data_root, args.train_split, args.train_limit)
    for _task_name, task, test_ex in examples:
        truth_key = serialize(test_ex["output"])
        pool = candidate_pool(task, test_ex, rng)
        x = np.stack(
            [feature_vector(task["train"], test_ex["input"], candidate.grid) for candidate in pool],
            axis=0,
        )
        y = np.asarray([1.0 if serialize(candidate.grid) == truth_key else 0.0 for candidate in pool], dtype=np.float32)
        if y.sum() > 0 and y.sum() < len(y):
            groups.append((x, y))
    return groups


def train_ranker(args: argparse.Namespace) -> tuple[Ranker, Standardizer, dict[str, Any]]:
    groups = build_training_groups(args)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(groups)
    split = int(0.85 * len(groups))
    train_groups = groups[:split]
    val_groups = groups[split:]
    train_flat = np.concatenate([x for x, _y in train_groups], axis=0)
    all_labels = np.concatenate([y for _x, y in groups], axis=0)

    mean = train_flat.mean(axis=0)
    std = train_flat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    standardizer = Standardizer(mean=mean, std=std)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = Ranker(train_flat.shape[1], args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_tensors = [
        (
            torch.tensor(standardizer.apply(x), dtype=torch.float32, device=device),
            torch.tensor(y, dtype=torch.float32, device=device),
        )
        for x, y in train_groups
    ]
    val_tensors = [
        (
            torch.tensor(standardizer.apply(x), dtype=torch.float32, device=device),
            torch.tensor(y, dtype=torch.float32, device=device),
        )
        for x, y in val_groups
    ]

    best_state = None
    best_val = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(train_tensors), device=device).detach().cpu().numpy()
        loss_sum = 0.0
        seen = 0
        for idx in perm:
            xg, yg = train_tensors[int(idx)]
            logits = model(xg)
            pos_logits = logits[yg > 0.5]
            neg_logits = logits[yg <= 0.5]
            # Listwise ranking: positives should dominate the candidate partition.
            loss = -(torch.logsumexp(pos_logits, dim=0) - torch.logsumexp(logits, dim=0))
            if neg_logits.numel():
                margin = args.margin - pos_logits[:, None] + neg_logits[None, :]
                loss = loss + F.relu(margin).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            seen += 1

        with torch.no_grad():
            model.eval()
            top1 = 0.0
            top2 = 0.0
            margins = []
            for xg, yg in val_tensors:
                scores = model(xg)
                order = torch.argsort(scores, descending=True)
                positives = torch.where(yg > 0.5)[0]
                top1 += float(bool((order[:1, None] == positives[None, :]).any().item()))
                top2 += float(bool((order[:2, None] == positives[None, :]).any().item()))
                pos_score = scores[yg > 0.5].max()
                neg_score = scores[yg <= 0.5].max()
                margins.append((pos_score - neg_score).item())
            val_top1 = top1 / max(len(val_tensors), 1)
            val_top2 = top2 / max(len(val_tensors), 1)
            val_margin = float(np.mean(margins)) if margins else 0.0
            val_score = val_top2 + 0.1 * val_top1 + 0.01 * val_margin
        history.append({"epoch": epoch, "loss": loss_sum / max(seen, 1), "val_top1": val_top1, "val_top2": val_top2, "val_margin": val_margin})
        if val_score > best_val:
            best_val = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    info = {
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "train_candidates": int(sum(len(y) for _x, y in train_groups)),
        "val_candidates": int(sum(len(y) for _x, y in val_groups)),
        "positive_rate": float(all_labels.mean()),
        "feature_dim": int(train_flat.shape[1]),
        "device": str(device),
        "history": history,
    }
    return model, standardizer, info


def score_candidates(
    model: Ranker,
    standardizer: Standardizer,
    demos: list[dict],
    test_input: list[list[int]],
    entries: list[dict],
    device: torch.device,
) -> list[dict]:
    if not entries:
        return []
    x = np.stack([feature_vector(demos, test_input, entry["prediction"]) for entry in entries], axis=0)
    xt = torch.tensor(standardizer.apply(x), dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(xt).detach().cpu().numpy()
    return [{**entry, "learned_logit": float(logit), "vote_log": math.log1p(entry.get("votes", 0))} for entry, logit in zip(entries, logits)]


def init_metric() -> dict[str, float]:
    return {"pass_at_1": 0.0, "pass_at_2": 0.0, "oracle": 0.0, "examples": 0.0}


def add_metric(metric: dict[str, float], ordered: list[dict], truth: list[list[int]], weight: float) -> None:
    p1, p2, rank = pass_metrics_for_order(ordered, truth)
    metric["pass_at_1"] += weight if p1 else 0.0
    metric["pass_at_2"] += weight if p2 else 0.0
    metric["oracle"] += weight if rank is not None else 0.0
    metric["examples"] += 1.0


def finalize(metric: dict[str, float], tasks: int) -> dict[str, float]:
    denom = max(float(tasks), 1.0)
    return {
        "pass_at_1": metric["pass_at_1"] / denom,
        "pass_at_2": metric["pass_at_2"] / denom,
        "oracle": metric["oracle"] / denom,
        "examples": metric["examples"],
    }


def evaluate_official_predictions(args: argparse.Namespace, model: Ranker, standardizer: Standardizer) -> dict[str, Any]:
    device = next(model.parameters()).device
    files = task_files(args.eval_data_root, args.eval_split, args.eval_limit)
    metrics = {f"learned_vote_{w:g}": init_metric() for w in args.vote_weights}
    metrics["majority"] = init_metric()
    task_count = 0
    for path in files:
        task_name = path.stem
        task = load_json(path)
        predictions_by_index = load_prediction_roots(args.output_root, task_name)
        if predictions_by_index is None:
            continue
        task_count += 1
        test_examples = task.get("test", [])
        weight = 1.0 / max(len(test_examples), 1)
        for idx, test_ex in enumerate(test_examples):
            key = normalize_index(idx)
            if key not in predictions_by_index or "output" not in test_ex:
                continue
            entries = get_majority_vote(predictions_by_index[key])
            scored = score_candidates(model, standardizer, task["train"], test_ex["input"], entries, device)
            truth = test_ex["output"]
            add_metric(metrics["majority"], entries, truth, weight)
            for vote_weight in args.vote_weights:
                ordered = sorted(
                    scored,
                    key=lambda entry: (-(entry["learned_logit"] + vote_weight * entry["vote_log"]), -entry["votes"]),
                )
                add_metric(metrics[f"learned_vote_{vote_weight:g}"], ordered, truth, weight)

    finalized = {name: finalize(metric, task_count) for name, metric in metrics.items()}
    best_name, best_metric = max(finalized.items(), key=lambda item: (item[1]["pass_at_2"], item[1]["pass_at_1"]))
    return {
        "tasks_evaluated": task_count,
        "metrics": finalized,
        "best": {"strategy": best_name, **best_metric},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-root", type=Path, default=Path("raw_data/ARC-AGI"))
    parser.add_argument("--train-split", default="training")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-data-root", type=Path, default=Path("raw_data/ARC-AGI"))
    parser.add_argument("--eval-split", default="evaluation")
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--vote-weights", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5])
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model, standardizer, train_info = train_ranker(args)
    eval_info = evaluate_official_predictions(args, model, standardizer)
    summary = {"train": train_info, "eval": eval_info}
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
