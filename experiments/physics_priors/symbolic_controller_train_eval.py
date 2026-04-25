#!/usr/bin/env python3
"""Train a lightweight symbolic mechanism controller on ARC training tasks.

The controller is deliberately small and non-generative:

1. Generate a broad symbolic candidate bank from each task demonstration.
2. Train a groupwise ranker to select the best symbolic mechanism candidate.
3. Train a confidence head that decides whether the top symbolic candidate is
   likely correct.
4. On ARC evaluation, use VARC as the default and let the symbolic controller
   take the second pass@2 slot only when confidence is high.

No ARC evaluation labels are used for training or threshold selection. The
thresholds are selected on a held-out slice of ARC training tasks.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from learned_relation_ranker import feature_vector
from physics_reranker import relation_energy, serialize
from rerank_varc_predictions import (
    get_majority_vote,
    load_json,
    load_prediction_roots,
    normalize_index,
    pass_metrics_for_order,
    task_files,
)
from symbolic_candidate_search import SymbolicCandidate, generate_broad_symbolic_candidates


TRANSFORMS = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flipud",
    "fliplr",
    "transpose",
    "anti_transpose",
)


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


@dataclass
class CandidateGroup:
    task_name: str
    test_index: int
    task: dict
    test_ex: dict
    candidates: list[SymbolicCandidate]
    features: np.ndarray
    labels: np.ndarray
    energies: np.ndarray


class Ranker(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ConfidenceHead(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def label_features(label: str) -> np.ndarray:
    features: list[float] = []
    features.extend(
        [
            float("tile" in label),
            float("hstack" in label),
            float("vstack" in label),
            float("bbox" in label),
            float("component" in label),
            float("color" in label),
            float("largest" in label),
            float("smallest" in label),
            float("topmost" in label),
            float("bottommost" in label),
            float("leftmost" in label),
            float("rightmost" in label),
        ]
    )
    for transform in TRANSFORMS:
        features.append(float(transform in label))
    reps_h = 0.0
    reps_w = 0.0
    if "tile_" in label:
        try:
            text = label.split("tile_", 1)[1].split("_", 1)[0]
            left, right = text.split("x")
            reps_h = float(left) / 4.0
            reps_w = float(right) / 4.0
        except (IndexError, ValueError):
            pass
    features.extend([reps_h, reps_w])
    return np.asarray(features, dtype=np.float32)


def candidate_features(task: dict, test_ex: dict, candidate: SymbolicCandidate) -> np.ndarray:
    base = feature_vector(task.get("train", []), test_ex["input"], candidate.grid)
    energy = np.asarray(
        [relation_energy(task.get("train", []), test_ex["input"], candidate.grid)],
        dtype=np.float32,
    )
    return np.concatenate([base, energy, label_features(candidate.label)], axis=0)


def build_groups_for_task(path_text: str, max_candidates: int) -> list[CandidateGroup]:
    groups = []
    path = Path(path_text)
    task = load_json(path)
    for idx, test_ex in enumerate(task.get("test", [])):
        if "output" not in test_ex:
            continue
        candidates = generate_broad_symbolic_candidates(
            task, test_ex["input"], max_candidates=max_candidates
        )
        if not candidates:
            continue
        truth = serialize(test_ex["output"])
        features = np.stack([candidate_features(task, test_ex, candidate) for candidate in candidates])
        labels = np.asarray(
            [1.0 if serialize(candidate.grid) == truth else 0.0 for candidate in candidates],
            dtype=np.float32,
        )
        energies = features[
            :,
            feature_vector(task.get("train", []), test_ex["input"], candidates[0].grid).shape[0],
        ]
        groups.append(
            CandidateGroup(
                task_name=path.stem,
                test_index=idx,
                task=task,
                test_ex=test_ex,
                candidates=candidates,
                features=features.astype(np.float32),
                labels=labels,
                energies=energies.astype(np.float32),
            )
        )
    return groups


def build_groups(
    data_root: Path,
    split: str,
    limit: int | None,
    max_candidates: int,
    workers: int,
) -> list[CandidateGroup]:
    files = [str(path) for path in task_files(data_root, split, limit)]
    if workers <= 1:
        nested = [build_groups_for_task(path, max_candidates) for path in files]
    else:
        nested = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(build_groups_for_task, path, max_candidates) for path in files]
            for future in as_completed(futures):
                nested.append(future.result())
    return [group for groups in nested for group in groups]


def split_groups(groups: list[CandidateGroup], seed: int, val_fraction: float) -> tuple[list[CandidateGroup], list[CandidateGroup]]:
    rng = random.Random(seed)
    by_task: dict[str, list[CandidateGroup]] = {}
    for group in groups:
        by_task.setdefault(group.task_name, []).append(group)
    task_names = sorted(by_task)
    rng.shuffle(task_names)
    val_count = max(1, int(round(len(task_names) * val_fraction)))
    val_tasks = set(task_names[:val_count])
    train = [group for group in groups if group.task_name not in val_tasks]
    val = [group for group in groups if group.task_name in val_tasks]
    return train, val


def fit_standardizer(groups: list[CandidateGroup]) -> Standardizer:
    flat = np.concatenate([group.features for group in groups], axis=0)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return Standardizer(mean=mean, std=std)


def train_ranker(
    groups: list[CandidateGroup],
    val_groups: list[CandidateGroup],
    standardizer: Standardizer,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Ranker, dict[str, Any]]:
    dim = groups[0].features.shape[1]
    model = Ranker(dim, args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_groups = [group for group in groups if group.labels.sum() > 0 and group.labels.sum() < len(group.labels)]
    history = []
    best_state = None
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_groups)
        model.train()
        loss_sum = 0.0
        seen = 0
        for group in train_groups:
            x = torch.tensor(standardizer.apply(group.features), dtype=torch.float32, device=device)
            y = torch.tensor(group.labels, dtype=torch.float32, device=device)
            scores = model(x)
            pos = scores[y > 0.5]
            if pos.numel() == 0:
                continue
            loss = -(torch.logsumexp(pos, dim=0) - torch.logsumexp(scores, dim=0))
            neg = scores[y <= 0.5]
            if neg.numel():
                loss = loss + F.relu(args.margin - pos[:, None] + neg[None, :]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
            seen += 1
        val = evaluate_symbolic_ranker(model, standardizer, val_groups, device)
        score = val["top2"] + 0.1 * val["top1"]
        history.append({"epoch": epoch, "loss": loss_sum / max(seen, 1), **val})
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"train_positive_groups": len(train_groups), "history": history}


@torch.no_grad()
def score_group(
    model: Ranker,
    standardizer: Standardizer,
    group: CandidateGroup,
    device: torch.device,
) -> np.ndarray:
    x = torch.tensor(standardizer.apply(group.features), dtype=torch.float32, device=device)
    return model(x).detach().cpu().numpy()


def evaluate_symbolic_ranker(
    model: Ranker,
    standardizer: Standardizer,
    groups: list[CandidateGroup],
    device: torch.device,
) -> dict[str, float]:
    top1 = 0.0
    top2 = 0.0
    oracle = 0.0
    for group in groups:
        scores = score_group(model, standardizer, group, device)
        order = np.argsort(-scores)
        labels = group.labels
        top1 += float(labels[order[:1]].max() > 0.5)
        top2 += float(labels[order[:2]].max() > 0.5)
        oracle += float(labels.max() > 0.5)
    denom = max(len(groups), 1)
    return {"top1": top1 / denom, "top2": top2 / denom, "oracle": oracle / denom}


def confidence_examples(
    ranker: Ranker,
    standardizer: Standardizer,
    groups: list[CandidateGroup],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    labels = []
    for group in groups:
        scores = score_group(ranker, standardizer, group, device)
        order = np.argsort(-scores)
        top = order[0]
        second_score = scores[order[1]] if len(order) > 1 else scores[top] - 10.0
        energy_order = np.argsort(group.energies)
        energy_rank = int(np.where(energy_order == top)[0][0]) / max(len(order) - 1, 1)
        row = np.concatenate(
            [
                standardizer.apply(group.features[top : top + 1])[0],
                np.asarray(
                    [
                        scores[top],
                        scores[top] - second_score,
                        group.energies[top],
                        group.energies[energy_order[1]] - group.energies[energy_order[0]]
                        if len(order) > 1
                        else 0.0,
                        energy_rank,
                        math.log1p(len(order)),
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )
        rows.append(row)
        labels.append(float(group.labels[top] > 0.5))
    return np.stack(rows).astype(np.float32), np.asarray(labels, dtype=np.float32)


def train_confidence(
    ranker: Ranker,
    standardizer: Standardizer,
    train_groups: list[CandidateGroup],
    val_groups: list[CandidateGroup],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ConfidenceHead, dict[str, Any]]:
    x_train, y_train = confidence_examples(ranker, standardizer, train_groups, device)
    x_val, y_val = confidence_examples(ranker, standardizer, val_groups, device)
    model = ConfidenceHead(x_train.shape[1], args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight_value = (len(y_train) - y_train.sum()) / max(float(y_train.sum()), 1.0)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    xt = torch.tensor(x_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    xv = torch.tensor(x_val, dtype=torch.float32, device=device)
    best_state = None
    best_ap = -1.0
    history = []
    for epoch in range(1, args.confidence_epochs + 1):
        model.train()
        logits = model(xt)
        loss = F.binary_cross_entropy_with_logits(logits, yt, pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            probs = torch.sigmoid(model(xv)).detach().cpu().numpy()
        metrics = binary_metrics(probs, y_val)
        history.append({"epoch": epoch, "loss": float(loss.item()), **metrics})
        if metrics["average_precision"] > best_ap:
            best_ap = metrics["average_precision"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    thresholds = choose_thresholds(torch.sigmoid(model(xv)).detach().cpu().numpy(), y_val)
    return model, {
        "train_positive_rate": float(y_train.mean()),
        "val_positive_rate": float(y_val.mean()),
        "thresholds": thresholds,
        "history": history,
    }


def binary_metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    order = np.argsort(-probs)
    labels_sorted = labels[order]
    tp = np.cumsum(labels_sorted)
    precision = tp / np.arange(1, len(labels_sorted) + 1)
    recall = tp / max(labels.sum(), 1.0)
    ap = float((precision * labels_sorted).sum() / max(labels.sum(), 1.0))
    best_f1 = 0.0
    for p, r in zip(precision, recall):
        if p + r:
            best_f1 = max(best_f1, float(2 * p * r / (p + r)))
    return {"average_precision": ap, "best_f1": best_f1}


def choose_thresholds(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    out = {}
    candidates = sorted(set(float(x) for x in probs), reverse=True)
    for target_precision in (0.5, 0.67, 0.8, 0.9):
        best = None
        for threshold in candidates:
            selected = probs >= threshold
            if not selected.any():
                continue
            precision = float(labels[selected].mean())
            recall = float(labels[selected].sum() / max(labels.sum(), 1.0))
            if precision >= target_precision and (best is None or recall > best[1]):
                best = (threshold, recall, precision)
        out[f"precision>={target_precision:g}"] = best[0] if best is not None else 1.01
    return out


@torch.no_grad()
def confidence_probability(
    confidence: ConfidenceHead,
    ranker: Ranker,
    standardizer: Standardizer,
    group: CandidateGroup,
    device: torch.device,
) -> tuple[float, list[dict]]:
    scores = score_group(ranker, standardizer, group, device)
    order = np.argsort(-scores)
    top = order[0]
    x_conf, _ = confidence_examples(ranker, standardizer, [group], device)
    prob = float(torch.sigmoid(confidence(torch.tensor(x_conf, dtype=torch.float32, device=device))).item())
    ordered = [
        {
            "prediction": group.candidates[idx].grid,
            "label": group.candidates[idx].label,
            "controller_score": float(scores[idx]),
            "energy": float(group.energies[idx]),
        }
        for idx in order
    ]
    return prob, ordered


def unique_concat(*orders: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for order in orders:
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


def finalize(metric: dict[str, float], tasks: int) -> dict[str, float]:
    denom = max(float(tasks), 1.0)
    return {
        "pass_at_1": metric["pass_at_1"] / denom,
        "pass_at_2": metric["pass_at_2"] / denom,
        "oracle": metric["oracle"] / denom,
        "examples": metric["examples"],
    }


def evaluate_with_official(
    ranker: Ranker,
    confidence: ConfidenceHead,
    standardizer: Standardizer,
    groups: list[CandidateGroup],
    official_roots: str,
    thresholds: dict[str, float],
    device: torch.device,
) -> dict[str, Any]:
    totals: dict[str, dict[str, float]] = {}
    previews = []
    task_cache: dict[str, dict[str, list]] = {}
    task_names = set()
    for group in groups:
        task_names.add(group.task_name)
        if group.task_name not in task_cache:
            task_cache[group.task_name] = load_prediction_roots(official_roots, group.task_name) or {}
        official_predictions = task_cache[group.task_name]
        official = get_majority_vote(official_predictions.get(normalize_index(group.test_index), []))
        prob, symbolic = confidence_probability(confidence, ranker, standardizer, group, device)
        truth = group.test_ex["output"]
        weight = 1.0 / max(len(group.task.get("test", [])), 1)
        orders = {
            "official_majority": official,
            "controller_symbolic_first": unique_concat(symbolic, official),
            "official_then_controller_top1": unique_concat(official[:1], symbolic[:1], official, symbolic),
        }
        for name, threshold in thresholds.items():
            second = symbolic[:1] if symbolic and prob >= threshold else official[1:2]
            orders[f"official_then_controller_{name}"] = unique_concat(
                official[:1], second, official, symbolic
            )
        for name, ordered in orders.items():
            totals.setdefault(name, init_metric())
            add_metric(totals[name], ordered, truth, weight)
        if len(previews) < 12:
            previews.append(
                {
                    "task": group.task_name,
                    "test_index": group.test_index,
                    "confidence": prob,
                    "official_top2": [
                        {"votes": item["votes"], "matches": item["prediction"] == truth}
                        for item in official[:2]
                    ],
                    "symbolic_top2": [
                        {
                            "label": item["label"],
                            "score": round(item["controller_score"], 4),
                            "energy": round(item["energy"], 4),
                            "matches": item["prediction"] == truth,
                        }
                        for item in symbolic[:2]
                    ],
                }
            )
    task_count = len(task_names)
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
    parser.add_argument("--train-data-root", type=Path, default=Path("raw_data/ARC-AGI"))
    parser.add_argument("--train-split", default="training")
    parser.add_argument("--eval-data-root", type=Path, default=Path("raw_data/ARC-AGI"))
    parser.add_argument("--eval-split", default="evaluation")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--official-roots", required=True)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--confidence-epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_train_groups = build_groups(
        args.train_data_root,
        args.train_split,
        args.train_limit,
        args.max_candidates,
        args.workers,
    )
    train_groups, val_groups = split_groups(all_train_groups, args.seed, args.val_fraction)
    standardizer = fit_standardizer(train_groups)
    ranker, ranker_info = train_ranker(train_groups, val_groups, standardizer, args, device)
    ranker_val = evaluate_symbolic_ranker(ranker, standardizer, val_groups, device)
    confidence, confidence_info = train_confidence(
        ranker, standardizer, train_groups, val_groups, args, device
    )
    eval_groups = build_groups(
        args.eval_data_root,
        args.eval_split,
        args.eval_limit,
        args.max_candidates,
        args.workers,
    )
    eval_symbolic = evaluate_symbolic_ranker(ranker, standardizer, eval_groups, device)
    official_eval = evaluate_with_official(
        ranker,
        confidence,
        standardizer,
        eval_groups,
        args.official_roots,
        confidence_info["thresholds"],
        device,
    )

    summary = {
        "device": str(device),
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "eval_groups": len(eval_groups),
        "ranker_info": ranker_info,
        "ranker_val": ranker_val,
        "confidence_info": confidence_info,
        "eval_symbolic": eval_symbolic,
        "official_eval": official_eval,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
