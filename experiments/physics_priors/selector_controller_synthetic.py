#!/usr/bin/env python3
"""Train a learned selector controller on synthetic ARC-like extraction tasks.

Task family:

  input scene -> output crop

The hidden mechanism is:

  selector(scene objects) + D4 transform

Examples:

  crop the largest object, then rotate 90 degrees
  crop the leftmost object, unchanged
  crop the densest color bbox, then flip horizontally

The model does not render pixels directly. It scores candidate mechanisms from
demo consistency features, then applies the selected mechanism to the query
input. This tests the next step after the hand-written variable selector search:
can a small learned controller pick the right selector family from examples?
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from physics_reranker import (
    TRANSFORM_NAMES,
    apply_transform,
    as_array,
    as_grid,
    candidate_complexity,
    color_hist,
    component_count,
    entropy,
    foreground_fraction,
    relation_energy,
    serialize,
)
from rerank_varc_predictions import load_json, pass_metrics_for_order, task_files


SELECTOR_NAMES = (
    "max_area",
    "min_area",
    "max_bbox_area",
    "min_bbox_area",
    "max_height",
    "min_height",
    "max_width",
    "min_width",
    "topmost",
    "bottommost",
    "leftmost",
    "rightmost",
    "densest",
    "sparsest",
)


@dataclass(frozen=True)
class ColorStats:
    color: int
    area: int
    bbox_area: int
    y0: int
    y1: int
    x0: int
    x1: int
    height: int
    width: int
    density: float
    crop: np.ndarray


@dataclass(frozen=True)
class Rule:
    selector: str
    transform: str

    @property
    def label(self) -> str:
        return f"{self.selector}_{self.transform}"


@dataclass
class Group:
    task_name: str
    demos: list[dict]
    test_input: list[list[int]]
    test_output: list[list[int]]
    features: np.ndarray
    labels: np.ndarray
    rules: list[Rule]


Selector = Callable[[list[ColorStats]], ColorStats | None]


def color_stats(array: np.ndarray) -> list[ColorStats]:
    stats = []
    for color in sorted(int(v) for v in np.unique(array) if int(v) != 0):
        mask = array == color
        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        crop = array[y0:y1, x0:x1]
        area = int(mask.sum())
        bbox_area = int((y1 - y0) * (x1 - x0))
        stats.append(
            ColorStats(
                color=color,
                area=area,
                bbox_area=bbox_area,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                height=y1 - y0,
                width=x1 - x0,
                density=area / max(float(bbox_area), 1.0),
                crop=crop,
            )
        )
    return stats


def unique_extreme(
    stats: list[ColorStats],
    key: Callable[[ColorStats], tuple],
    reverse: bool = False,
) -> ColorStats | None:
    if not stats:
        return None
    ordered = sorted(stats, key=key, reverse=reverse)
    if len(ordered) > 1 and key(ordered[0]) == key(ordered[1]):
        return None
    return ordered[0]


def selector_fns() -> dict[str, Selector]:
    return {
        "max_area": lambda s: unique_extreme(s, lambda c: (c.area,), reverse=True),
        "min_area": lambda s: unique_extreme(s, lambda c: (c.area,)),
        "max_bbox_area": lambda s: unique_extreme(s, lambda c: (c.bbox_area,), reverse=True),
        "min_bbox_area": lambda s: unique_extreme(s, lambda c: (c.bbox_area,)),
        "max_height": lambda s: unique_extreme(s, lambda c: (c.height,), reverse=True),
        "min_height": lambda s: unique_extreme(s, lambda c: (c.height,)),
        "max_width": lambda s: unique_extreme(s, lambda c: (c.width,), reverse=True),
        "min_width": lambda s: unique_extreme(s, lambda c: (c.width,)),
        "topmost": lambda s: unique_extreme(s, lambda c: (c.y0,)),
        "bottommost": lambda s: unique_extreme(s, lambda c: (-c.y1,)),
        "leftmost": lambda s: unique_extreme(s, lambda c: (c.x0,)),
        "rightmost": lambda s: unique_extreme(s, lambda c: (-c.x1,)),
        "densest": lambda s: unique_extreme(s, lambda c: (round(c.density, 6),), reverse=True),
        "sparsest": lambda s: unique_extreme(s, lambda c: (round(c.density, 6),)),
    }


def all_rules() -> list[Rule]:
    return [Rule(selector, transform) for selector in SELECTOR_NAMES for transform in TRANSFORM_NAMES]


def apply_rule(grid: list[list[int]], rule: Rule) -> list[list[int]] | None:
    selected = selector_fns()[rule.selector](color_stats(as_array(grid)))
    if selected is None:
        return None
    return as_grid(apply_transform(selected.crop, rule.transform))


def array_distance(candidate: np.ndarray | None, target: np.ndarray) -> float:
    if candidate is None:
        return 4.0
    ch, cw = candidate.shape
    th, tw = target.shape
    max_h = max(ch, th, 1)
    max_w = max(cw, tw, 1)
    shape_penalty = abs(ch - th) / max_h + abs(cw - tw) / max_w
    oh = min(ch, th)
    ow = min(cw, tw)
    if oh == 0 or ow == 0:
        pixel_penalty = 1.0
    else:
        pixel_penalty = float((candidate[:oh, :ow] != target[:oh, :ow]).mean())
    area_penalty = 1.0 - (oh * ow) / max(float(max_h * max_w), 1.0)
    return shape_penalty + 0.75 * pixel_penalty + 0.5 * area_penalty


def rule_prior_features(rule: Rule) -> np.ndarray:
    return np.asarray(
        [float(rule.selector == name) for name in SELECTOR_NAMES]
        + [float(rule.transform == name) for name in TRANSFORM_NAMES],
        dtype=np.float32,
    )


def output_features(grid: list[list[int]] | None, input_grid: list[list[int]]) -> np.ndarray:
    if grid is None:
        return np.asarray([4.0, 4.0, 4.0, 4.0, 4.0, 0.0, 0.0], dtype=np.float32)
    y = as_array(grid)
    x = as_array(input_grid)
    return np.asarray(
        [
            y.shape[0] / 30.0,
            y.shape[1] / 30.0,
            math.log((y.size + 1.0) / (x.size + 1.0)),
            entropy(y),
            foreground_fraction(y),
            min(component_count(y), 30) / 30.0,
            candidate_complexity(grid),
        ],
        dtype=np.float32,
    )


def rule_features(demos: list[dict], test_input: list[list[int]], rule: Rule) -> np.ndarray:
    errors = []
    exacts = []
    shape_matches = []
    hist_l1 = []
    for demo in demos:
        pred = apply_rule(demo["input"], rule)
        pred_arr = None if pred is None else as_array(pred)
        target_arr = as_array(demo["output"])
        errors.append(array_distance(pred_arr, target_arr))
        exacts.append(float(pred is not None and np.array_equal(pred_arr, target_arr)))
        shape_matches.append(float(pred is not None and pred_arr.shape == target_arr.shape))
        if pred is None:
            hist_l1.append(2.0)
        else:
            hist_l1.append(float(np.abs(color_hist(pred_arr) - color_hist(target_arr)).sum()))

    test_pred = apply_rule(test_input, rule)
    stats = np.asarray(
        [
            float(np.mean(errors)),
            float(np.max(errors)),
            float(np.min(errors)),
            float(np.std(errors)),
            float(np.mean(exacts)),
            float(np.mean(shape_matches)),
            float(np.mean(hist_l1)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([stats, output_features(test_pred, test_input), rule_prior_features(rule)], axis=0)


def random_scene(
    rng: random.Random,
    size: int,
    colors: int,
    objects: int,
    max_obj: int,
    target_rule: Rule,
) -> np.ndarray | None:
    for _attempt in range(100):
        grid = np.zeros((size, size), dtype=np.int64)
        used_colors = rng.sample(range(1, colors), k=min(objects, colors - 1))
        occupied = np.zeros((size, size), dtype=bool)
        for color in used_colors:
            placed = False
            for _ in range(50):
                h = rng.randint(1, max_obj)
                w = rng.randint(1, max_obj)
                y = rng.randint(0, size - h)
                x = rng.randint(0, size - w)
                if occupied[y : y + h, x : x + w].any():
                    continue
                grid[y : y + h, x : x + w] = color
                occupied[y : y + h, x : x + w] = True
                placed = True
                break
            if not placed:
                break
        if len(color_stats(grid)) < 3:
            continue
        if apply_rule(as_grid(grid), target_rule) is not None:
            return grid
    return None


def synthetic_group(
    rng: random.Random,
    task_name: str,
    size: int,
    colors: int,
    objects: int,
    max_obj: int,
    demos_per_task: int,
    rules: list[Rule],
) -> Group | None:
    target_rule = rng.choice(rules)
    demos = []
    for _ in range(demos_per_task):
        scene = random_scene(rng, size, colors, objects, max_obj, target_rule)
        if scene is None:
            return None
        out = apply_rule(as_grid(scene), target_rule)
        if out is None:
            return None
        demos.append({"input": as_grid(scene), "output": out})
    query = random_scene(rng, size, colors, objects, max_obj, target_rule)
    if query is None:
        return None
    truth = apply_rule(as_grid(query), target_rule)
    if truth is None:
        return None
    features = np.stack([rule_features(demos, as_grid(query), rule) for rule in rules]).astype(np.float32)
    truth_key = serialize(truth)
    labels = []
    for rule in rules:
        demo_ok = True
        for demo in demos:
            pred = apply_rule(demo["input"], rule)
            if pred is None or serialize(pred) != serialize(demo["output"]):
                demo_ok = False
                break
        query_pred = apply_rule(as_grid(query), rule)
        labels.append(
            1.0
            if demo_ok and query_pred is not None and serialize(query_pred) == truth_key
            else 0.0
        )
    labels = np.asarray(labels, dtype=np.float32)
    if labels.sum() == 0:
        labels = np.asarray([1.0 if rule == target_rule else 0.0 for rule in rules], dtype=np.float32)
    return Group(task_name, demos, as_grid(query), truth, features, labels, rules)


def build_synthetic_groups(args: argparse.Namespace, count: int, seed: int) -> list[Group]:
    params = {
        "grid_size": args.grid_size,
        "colors": args.colors,
        "objects": args.objects,
        "max_obj": args.max_obj,
        "demos_per_task": args.demos_per_task,
    }
    if args.workers > 1 and count >= args.workers * 4:
        chunks = []
        base = count // args.workers
        rem = count % args.workers
        start = 0
        for idx in range(args.workers):
            chunk_count = base + (1 if idx < rem else 0)
            if chunk_count:
                chunks.append((params, chunk_count, seed + idx * 100003, start))
                start += chunk_count
        groups: list[Group] = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(build_synthetic_groups_chunk, *chunk) for chunk in chunks]
            for future in as_completed(futures):
                groups.extend(future.result())
        return sorted(groups, key=lambda group: group.task_name)
    return build_synthetic_groups_chunk(params, count, seed, 0)


def build_synthetic_groups_chunk(
    params: dict[str, int],
    count: int,
    seed: int,
    offset: int,
) -> list[Group]:
    rng = random.Random(seed)
    rules = all_rules()
    groups = []
    while len(groups) < count:
        group = synthetic_group(
            rng,
            f"synthetic_{offset + len(groups):06d}",
            params["grid_size"],
            params["colors"],
            params["objects"],
            params["max_obj"],
            params["demos_per_task"],
            rules,
        )
        if group is not None:
            groups.append(group)
    return groups


def arc_group_for_task(path_text: str) -> list[Group]:
    path = Path(path_text)
    task = load_json(path)
    rules = all_rules()
    groups = []
    for idx, test_ex in enumerate(task.get("test", [])):
        if "output" not in test_ex:
            continue
        features = np.stack([rule_features(task.get("train", []), test_ex["input"], rule) for rule in rules]).astype(np.float32)
        truth = serialize(test_ex["output"])
        labels = np.asarray(
            [
                1.0
                if (pred := apply_rule(test_ex["input"], rule)) is not None and serialize(pred) == truth
                else 0.0
                for rule in rules
            ],
            dtype=np.float32,
        )
        groups.append(
            Group(
                f"{path.stem}:{idx}",
                task.get("train", []),
                test_ex["input"],
                test_ex["output"],
                features,
                labels,
                rules,
            )
        )
    return groups


def build_arc_groups(data_root: Path, split: str, limit: int | None, workers: int) -> list[Group]:
    files = [str(path) for path in task_files(data_root, split, limit)]
    if workers <= 1:
        nested = [arc_group_for_task(path) for path in files]
    else:
        nested = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(arc_group_for_task, path) for path in files]
            for future in as_completed(futures):
                nested.append(future.result())
    return [group for groups in nested for group in groups]


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


class Controller(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


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
) -> tuple[torch.Tensor, torch.Tensor]:
    features = np.stack([standardizer.apply(group.features) for group in groups]).astype(np.float32)
    labels = np.stack([group.labels for group in groups]).astype(np.float32)
    return (
        torch.tensor(features, dtype=torch.float32, device=device),
        torch.tensor(labels, dtype=torch.float32, device=device),
    )


def evaluate_controller(
    model: Controller,
    standardizer: Standardizer,
    groups: list[Group],
    device: torch.device,
) -> dict[str, float]:
    if not groups:
        return {"top1": 0.0, "top2": 0.0, "oracle": 0.0}
    model.eval()
    x, y = group_tensors(groups, standardizer, device)
    n, rules, dim = x.shape
    with torch.no_grad():
        scores = model(x.reshape(n * rules, dim)).reshape(n, rules)
        top2_idx = torch.topk(scores, k=min(2, rules), dim=1).indices
        top1 = y.gather(1, top2_idx[:, :1]).amax(dim=1).gt(0.5).float().mean()
        top2 = y.gather(1, top2_idx).amax(dim=1).gt(0.5).float().mean()
        oracle = y.amax(dim=1).gt(0.5).float().mean()
    return {"top1": float(top1.item()), "top2": float(top2.item()), "oracle": float(oracle.item())}


def train_controller(
    train: list[Group],
    val: list[Group],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Controller, Standardizer, dict]:
    standardizer = fit_standardizer(train)
    model = Controller(train[0].features.shape[1], args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_x, train_y = group_tensors(train, standardizer, device)
    best_state = None
    best_score = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        order = torch.randperm(train_x.shape[0], device=device)
        for start in range(0, train_x.shape[0], args.batch_size):
            idx = order[start : start + args.batch_size]
            x = train_x[idx]
            y = train_y[idx]
            batch, rules, dim = x.shape
            scores = model(x.reshape(batch * rules, dim)).reshape(batch, rules)
            pos_mask = y > 0.5
            pos_scores = scores.masked_fill(~pos_mask, -1.0e9)
            loss = -(torch.logsumexp(pos_scores, dim=1) - torch.logsumexp(scores, dim=1)).mean()
            if args.margin > 0:
                best_pos = pos_scores.max(dim=1).values
                neg_margin = F.relu(args.margin - best_pos[:, None] + scores).masked_fill(pos_mask, 0.0)
                neg_count = (~pos_mask).float().sum().clamp_min(1.0)
                loss = loss + neg_margin.sum() / neg_count
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item()) * batch
            seen += batch
        val_metric = evaluate_controller(model, standardizer, val, device)
        score = val_metric["top2"] + 0.1 * val_metric["top1"]
        history.append({"epoch": epoch, "loss": total / max(seen, 1), **val_metric})
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if args.log_every and (epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs):
            print(f"epoch {epoch}: {json.dumps(history[-1], sort_keys=True)}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, standardizer, {"history": history}


def evaluate_arc_predictions(
    model: Controller,
    standardizer: Standardizer,
    groups: list[Group],
    device: torch.device,
) -> dict:
    metric = {"pass_at_1": 0.0, "pass_at_2": 0.0, "oracle": 0.0, "examples": 0.0}
    previews = []
    task_names = {group.task_name.split(":", 1)[0] for group in groups}
    for group in groups:
        x = torch.tensor(standardizer.apply(group.features), dtype=torch.float32, device=device)
        with torch.no_grad():
            scores = model(x).detach().cpu().numpy()
        order = np.argsort(-scores)
        ordered = []
        for idx in order:
            pred = apply_rule(group.test_input, group.rules[idx])
            if pred is None:
                continue
            ordered.append({"prediction": pred, "rule": group.rules[idx].label, "score": float(scores[idx])})
        weight = 1.0
        p1, p2, rank = pass_metrics_for_order(ordered, group.test_output)
        metric["pass_at_1"] += weight if p1 else 0.0
        metric["pass_at_2"] += weight if p2 else 0.0
        metric["oracle"] += weight if rank is not None else 0.0
        metric["examples"] += 1.0
        if len(previews) < 12:
            previews.append(
                {
                    "task": group.task_name,
                    "top2": [
                        {
                            "rule": item["rule"],
                            "score": round(item["score"], 4),
                            "matches": item["prediction"] == group.test_output,
                        }
                        for item in ordered[:2]
                    ],
                    "truth_rank": rank,
                }
            )
    denom = max(float(len(task_names)), 1.0)
    return {
        "pass_at_1": metric["pass_at_1"] / denom,
        "pass_at_2": metric["pass_at_2"] / denom,
        "oracle": metric["oracle"] / denom,
        "examples": metric["examples"],
        "preview": previews,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--colors", type=int, default=8)
    parser.add_argument("--objects", type=int, default=5)
    parser.add_argument("--max-obj", type=int, default=5)
    parser.add_argument("--demos-per-task", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    parser.add_argument("--arc-data-root", type=Path, default=Path("raw_data/ARC-AGI"))
    parser.add_argument("--arc-split", default="evaluation")
    parser.add_argument("--arc-limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train = build_synthetic_groups(args, args.train_size, args.seed)
    val = build_synthetic_groups(args, args.val_size, args.seed + 1)
    test = build_synthetic_groups(args, args.test_size, args.seed + 2)
    model, standardizer, train_info = train_controller(train, val, args, device)
    synthetic_test = evaluate_controller(model, standardizer, test, device)

    arc_groups = build_arc_groups(args.arc_data_root, args.arc_split, args.arc_limit, args.workers)
    arc_rank = evaluate_controller(model, standardizer, arc_groups, device)
    arc_pred = evaluate_arc_predictions(model, standardizer, arc_groups, device)

    summary = {
        "device": str(device),
        "rules": len(all_rules()),
        "feature_dim": int(train[0].features.shape[1]),
        "synthetic": {
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
            "test": synthetic_test,
        },
        "arc": {
            "split": args.arc_split,
            "groups": len(arc_groups),
            "rank": arc_rank,
            "pred": arc_pred,
        },
        "train_info": train_info,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
