#!/usr/bin/env python3
"""Infer variable color/component bbox selectors on ARC tasks.

Many ARC extraction tasks are not "crop color 2" globally. The selected color
changes per demonstration, but a property stays invariant, for example:

  crop the color with largest area
  crop the color whose bbox is leftmost
  crop the color with largest bbox area, then rotate

This script infers such selectors from train pairs exactly, then applies the
inferred rule to test inputs. It is a small mechanism-bank expansion focused on
selector learning, not a neural model.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from physics_reranker import TRANSFORM_NAMES, apply_transform, as_array, as_grid, relation_energy, serialize
from rerank_varc_predictions import load_json, pass_metrics_for_order, task_files


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
    demo_error: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.selector}_{self.transform}_err{self.demo_error:.3f}"


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


def matching_demo_rules(input_grid: list[list[int]], output_grid: list[list[int]]) -> list[tuple[str, int]]:
    x = as_array(input_grid)
    y = as_array(output_grid)
    matches = []
    for stat in color_stats(x):
        for transform in TRANSFORM_NAMES:
            try:
                out = apply_transform(stat.crop, transform)
            except ValueError:
                continue
            if out.shape == y.shape and np.array_equal(out, y):
                matches.append((transform, stat.color))
    return matches


def array_distance(candidate: np.ndarray, target: np.ndarray) -> float:
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


def infer_rules(task: dict, max_demo_error: float) -> list[Rule]:
    demos = task.get("train", [])
    if not demos:
        return []
    selectors = selector_fns()
    rules = []
    for transform in TRANSFORM_NAMES:
        for selector_name, selector in selectors.items():
            errors = []
            for ex in demos:
                selected = selector(color_stats(as_array(ex["input"])))
                if selected is None:
                    errors.append(float("inf"))
                    break
                candidate = apply_transform(selected.crop, transform)
                errors.append(array_distance(candidate, as_array(ex["output"])))
            if errors and max(errors) <= max_demo_error:
                rules.append(Rule(selector_name, transform, float(np.mean(errors))))
    return rules


def apply_rule(input_grid: list[list[int]], rule: Rule) -> list[list[int]] | None:
    selected = selector_fns()[rule.selector](color_stats(as_array(input_grid)))
    if selected is None:
        return None
    return as_grid(apply_transform(selected.crop, rule.transform))


def evaluate_task(path_text: str, max_demo_error: float) -> dict:
    path = Path(path_text)
    task = load_json(path)
    rules = infer_rules(task, max_demo_error)
    metric = {"pass_at_1": 0.0, "pass_at_2": 0.0, "oracle": 0.0, "examples": 0.0}
    previews = []
    tests = task.get("test", [])
    weight = 1.0 / max(len(tests), 1)
    for idx, test_ex in enumerate(tests):
        if "output" not in test_ex:
            continue
        candidates = []
        seen = set()
        for rule in rules:
            grid = apply_rule(test_ex["input"], rule)
            if grid is None:
                continue
            key = serialize(grid)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "prediction": grid,
                    "rule": rule.label,
                    "energy": relation_energy(task.get("train", []), test_ex["input"], grid),
                }
            )
        ordered = sorted(
            candidates,
            key=lambda item: (
                item["energy"] + next(
                    (rule.demo_error for rule in rules if rule.label == item["rule"]),
                    0.0,
                ),
                item["rule"],
            ),
        )
        p1, p2, rank = pass_metrics_for_order(ordered, test_ex["output"])
        metric["pass_at_1"] += weight if p1 else 0.0
        metric["pass_at_2"] += weight if p2 else 0.0
        metric["oracle"] += weight if rank is not None else 0.0
        metric["examples"] += 1.0
        if len(previews) < 5 and ordered:
            previews.append(
                {
                    "test_index": idx,
                    "rules": [item["rule"] for item in ordered[:5]],
                    "top2_matches": [item["prediction"] == test_ex["output"] for item in ordered[:2]],
                    "truth_rank": rank,
                }
            )
    return {
        "task": path.stem,
        "rule_count": len(rules),
        "rules": [rule.label for rule in rules[:20]],
        "metric": metric,
        "preview": previews,
    }


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("raw_data/ARC-AGI"))
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-demo-error", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    files = [str(path) for path in task_files(args.data_root, args.split, args.limit)]
    if args.workers <= 1:
        results = [evaluate_task(path, args.max_demo_error) for path in files]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(evaluate_task, path, args.max_demo_error) for path in files]
            for future in as_completed(futures):
                results.append(future.result())

    total = {"pass_at_1": 0.0, "pass_at_2": 0.0, "oracle": 0.0, "examples": 0.0}
    tasks_with_rules = 0
    tasks_with_oracle = 0
    previews = []
    for result in sorted(results, key=lambda item: item["task"]):
        if result["rule_count"]:
            tasks_with_rules += 1
        if result["metric"]["oracle"] > 0:
            tasks_with_oracle += 1
        merge_metric(total, result["metric"])
        if len(previews) < 20:
            for preview in result["preview"]:
                previews.append({"task": result["task"], **preview})
                if len(previews) >= 20:
                    break

    summary = {
        "data_root": str(args.data_root),
        "split": args.split,
        "max_demo_error": args.max_demo_error,
        "tasks_evaluated": len(results),
        "tasks_with_rules": tasks_with_rules,
        "tasks_with_oracle": tasks_with_oracle,
        "metric": finalize(total, len(results)),
        "preview": previews,
        "per_task": sorted(results, key=lambda item: item["task"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
