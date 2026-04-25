#!/usr/bin/env python3
"""Exact object-transport rule search on real ARC tasks.

This is the ARC-facing counterpart of object_transport_synthetic.py. It tests
whether a compact mechanism basis found in synthetic experiments appears in
real ARC data:

  selected color + spatial shift + move/copy

The search is conservative. A rule is inferred only if it exactly maps every
training input to its training output. Only then is it applied to test inputs.
No test labels are used for inference.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from physics_reranker import as_array, as_grid, relation_energy, serialize
from rerank_varc_predictions import load_json, pass_metrics_for_order, task_files


MODES = ("move", "copy")


@dataclass(frozen=True)
class Rule:
    select_color: int
    output_color: int
    fill_color: int
    dy: int
    dx: int
    mode: str

    @property
    def label(self) -> str:
        fill = f"_fill{self.fill_color}" if self.mode == "move" else ""
        return (
            f"select{self.select_color}_to{self.output_color}_"
            f"{self.mode}{fill}_dy{self.dy}_dx{self.dx}"
        )


@dataclass(frozen=True)
class ComponentRule:
    selector: str
    output_color: int
    fill_color: int
    dy: int
    dx: int
    mode: str

    @property
    def label(self) -> str:
        fill = f"_fill{self.fill_color}" if self.mode == "move" else ""
        return (
            f"component_{self.selector}_to{self.output_color}_"
            f"{self.mode}{fill}_dy{self.dy}_dx{self.dx}"
        )


AnyRule = Rule | ComponentRule


def shift_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    y_src0 = max(0, -dy)
    y_src1 = min(h, h - dy)
    x_src0 = max(0, -dx)
    x_src1 = min(w, w - dx)
    y_dst0 = max(0, dy)
    y_dst1 = min(h, h + dy)
    x_dst0 = max(0, dx)
    x_dst1 = min(w, w + dx)
    if y_src1 <= y_src0 or x_src1 <= x_src0 or y_dst1 <= y_dst0 or x_dst1 <= x_dst0:
        return out
    out[y_dst0:y_dst1, x_dst0:x_dst1] = mask[y_src0:y_src1, x_src0:x_src1]
    return out


def apply_rule(array: np.ndarray, rule: Rule) -> np.ndarray:
    out = array.copy()
    mask = array == rule.select_color
    shifted = shift_mask(mask, rule.dy, rule.dx)
    if rule.mode == "move":
        out[mask] = rule.fill_color
    out[shifted] = rule.output_color
    return out


def connected_components(array: np.ndarray) -> list[dict]:
    h, w = array.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for y in range(h):
        for x in range(w):
            color = int(array[y, x])
            if color == 0 or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            cells = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and not seen[ny, nx]
                        and int(array[ny, nx]) == color
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys = [cell[0] for cell in cells]
            xs = [cell[1] for cell in cells]
            mask = np.zeros_like(array, dtype=bool)
            for cy, cx in cells:
                mask[cy, cx] = True
            comps.append(
                {
                    "color": color,
                    "area": len(cells),
                    "y0": min(ys),
                    "x0": min(xs),
                    "y1": max(ys) + 1,
                    "x1": max(xs) + 1,
                    "mask": mask,
                }
            )
    return comps


def selector_fns(demos: list[dict]) -> list[tuple[str, Callable[[np.ndarray], dict | None]]]:
    def select_with(array: np.ndarray, key: Callable[[dict], tuple], reverse: bool = False) -> dict | None:
        comps = connected_components(array)
        if not comps:
            return None
        return sorted(comps, key=key, reverse=reverse)[0]

    selectors: list[tuple[str, Callable[[np.ndarray], dict | None]]] = [
        ("largest", lambda x: select_with(x, lambda c: (-c["area"], c["y0"], c["x0"]))),
        ("smallest", lambda x: select_with(x, lambda c: (c["area"], c["y0"], c["x0"]))),
        ("topmost", lambda x: select_with(x, lambda c: (c["y0"], c["x0"]))),
        ("bottommost", lambda x: select_with(x, lambda c: (-c["y1"], c["x0"]))),
        ("leftmost", lambda x: select_with(x, lambda c: (c["x0"], c["y0"]))),
        ("rightmost", lambda x: select_with(x, lambda c: (-c["x1"], c["y0"]))),
    ]
    for color in [c for c in colors_in_demos(demos, ("input",)) if c != 0]:
        selectors.append(
            (
                f"color{color}_largest",
                lambda x, color=color: select_with(
                    np.where(x == color, x, 0),
                    lambda c: (-c["area"], c["y0"], c["x0"]),
                ),
            )
        )
    return selectors


def apply_component_rule(
    array: np.ndarray,
    rule: ComponentRule,
    selectors: dict[str, Callable[[np.ndarray], dict | None]],
) -> np.ndarray | None:
    comp = selectors[rule.selector](array)
    if comp is None:
        return None
    out = array.copy()
    mask = comp["mask"]
    shifted = shift_mask(mask, rule.dy, rule.dx)
    if rule.mode == "move":
        out[mask] = rule.fill_color
    out[shifted] = rule.output_color
    return out


def apply_mask_transport(
    array: np.ndarray,
    mask: np.ndarray,
    output_color: int,
    fill_color: int,
    dy: int,
    dx: int,
    mode: str,
) -> np.ndarray:
    out = array.copy()
    shifted = shift_mask(mask, dy, dx)
    if mode == "move":
        out[mask] = fill_color
    out[shifted] = output_color
    return out


def apply_any_rule(
    array: np.ndarray,
    rule: AnyRule,
    selectors: dict[str, Callable[[np.ndarray], dict | None]] | None = None,
) -> np.ndarray | None:
    if isinstance(rule, ComponentRule):
        if selectors is None:
            raise ValueError("component rules require selectors")
        return apply_component_rule(array, rule, selectors)
    return apply_rule(array, rule)


def colors_in_demos(demos: list[dict], grid_names: tuple[str, ...]) -> list[int]:
    colors = {
        int(value)
        for ex in demos
        for grid_name in grid_names
        for row in ex[grid_name]
        for value in row
        if 0 <= int(value) < 10
    }
    return sorted(colors)


def infer_rules(demos: list[dict], max_shift: int) -> list[AnyRule]:
    arrays = [(as_array(ex["input"]), as_array(ex["output"])) for ex in demos]
    if any(x.shape != y.shape for x, y in arrays):
        return []

    rules: list[AnyRule] = []
    shifts = [
        (dy, dx)
        for dy in range(-max_shift, max_shift + 1)
        for dx in range(-max_shift, max_shift + 1)
        if not (dy == 0 and dx == 0)
    ]
    input_colors = [color for color in colors_in_demos(demos, ("input",)) if color != 0]
    output_colors = [color for color in colors_in_demos(demos, ("output",)) if color != 0]
    fill_colors = colors_in_demos(demos, ("input", "output"))
    for select_color in input_colors:
        if not any((x == select_color).any() for x, _ in arrays):
            continue
        for dy, dx in shifts:
            for output_color in output_colors:
                for mode in MODES:
                    fills = fill_colors if mode == "move" else [0]
                    for fill_color in fills:
                        rule = Rule(
                            select_color=select_color,
                            output_color=output_color,
                            fill_color=fill_color,
                            dy=dy,
                            dx=dx,
                            mode=mode,
                        )
                        if all(np.array_equal(apply_rule(x, rule), y) for x, y in arrays):
                            rules.append(rule)
    selector_masks: dict[str, list[np.ndarray]] = {}
    for selector_name, selector in selector_fns(demos):
        masks = []
        ok = True
        for x, _ in arrays:
            comp = selector(x)
            if comp is None:
                ok = False
                break
            masks.append(comp["mask"])
        if ok:
            selector_masks[selector_name] = masks

    for selector_name, masks in selector_masks.items():
        if not masks:
            continue
        for dy, dx in shifts:
            for output_color in output_colors:
                for mode in MODES:
                    fills = fill_colors if mode == "move" else [0]
                    for fill_color in fills:
                        rule = ComponentRule(
                            selector=selector_name,
                            output_color=output_color,
                            fill_color=fill_color,
                            dy=dy,
                            dx=dx,
                            mode=mode,
                        )
                        ok = True
                        for idx, (x, y) in enumerate(arrays):
                            candidate = apply_mask_transport(
                                x,
                                masks[idx],
                                output_color,
                                fill_color,
                                dy,
                                dx,
                                mode,
                            )
                            if not np.array_equal(candidate, y):
                                ok = False
                                break
                        if ok:
                            rules.append(rule)
    return rules


def rule_complexity(rule: AnyRule) -> float:
    if isinstance(rule, ComponentRule):
        fill = 0.25 if rule.mode == "move" and rule.fill_color != 0 else 0.0
        selector = 0.5 if not rule.selector.startswith("color") else 0.25
        return abs(rule.dy) + abs(rule.dx) + fill + selector + (
            0.5 if rule.mode == "copy" else 0.0
        )
    recolor = 0.5 if rule.select_color != rule.output_color else 0.0
    fill = 0.25 if rule.mode == "move" and rule.fill_color != 0 else 0.0
    return abs(rule.dy) + abs(rule.dx) + recolor + fill + (
        0.5 if rule.mode == "copy" else 0.0
    )


def order_candidates(
    candidates: list[tuple[AnyRule, list[list[int]]]],
    demos: list[dict],
    test_input: list[list[int]],
) -> list[dict]:
    scored = []
    for rule, grid in candidates:
        energy = relation_energy(demos, test_input, grid)
        # Prefer smaller shifts and move over copy when relation energy ties.
        complexity = rule_complexity(rule)
        scored.append(
            {
                "rule": rule.label,
                "prediction": grid,
                "energy": energy,
                "complexity": complexity,
                "score": energy + 0.05 * complexity,
            }
        )
    return sorted(scored, key=lambda item: (item["score"], item["complexity"], item["rule"]))


def evaluate_task(task_path: Path, max_shift: int, preview_limit: int) -> dict:
    task = load_json(task_path)
    demos = task.get("train", [])
    rules = infer_rules(demos, max_shift)
    metric = {
        "pass_at_1": 0.0,
        "pass_at_2": 0.0,
        "oracle": 0.0,
        "examples": 0.0,
    }
    previews = []
    if not rules:
        return {"task": task_path.stem, "rules": [], "metric": metric, "preview": previews}

    test_examples = task.get("test", [])
    example_weight = 1.0 / max(len(test_examples), 1)
    selectors = dict(selector_fns(demos))
    for idx, test_ex in enumerate(test_examples):
        if "output" not in test_ex:
            continue
        x = as_array(test_ex["input"])
        candidates = []
        seen = set()
        for rule in rules:
            y = apply_any_rule(x, rule, selectors)
            if y is None:
                continue
            grid = as_grid(y)
            key = serialize(grid)
            if key in seen:
                continue
            seen.add(key)
            candidates.append((rule, grid))
        ordered = order_candidates(candidates, demos, test_ex["input"])
        pass1, pass2, truth_rank = pass_metrics_for_order(ordered, test_ex["output"])
        metric["pass_at_1"] += example_weight if pass1 else 0.0
        metric["pass_at_2"] += example_weight if pass2 else 0.0
        metric["oracle"] += example_weight if truth_rank is not None else 0.0
        metric["examples"] += 1.0
        if len(previews) < preview_limit:
            previews.append(
                {
                    "test_index": idx,
                    "rules": [item["rule"] for item in ordered[:5]],
                    "top2_matches": [
                        item["prediction"] == test_ex["output"] for item in ordered[:2]
                    ],
                    "truth_rank": truth_rank,
                    "candidate_count": len(ordered),
                }
            )
    return {
        "task": task_path.stem,
        "rules": [rule.label for rule in rules],
        "metric": metric,
        "preview": previews,
    }


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("raw_data/ARC-AGI"))
    parser.add_argument("--split", default="training")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-shift", type=int, default=8)
    parser.add_argument("--preview", type=int, default=12)
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    total = {"pass_at_1": 0.0, "pass_at_2": 0.0, "oracle": 0.0, "examples": 0.0}
    tasks_with_rules = 0
    tasks_with_oracle = 0
    previews = []
    per_task = []
    files = task_files(args.data_root, args.split, args.limit)
    if args.workers <= 1:
        results = [
            evaluate_task(path, args.max_shift, args.preview)
            for path in files
        ]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(evaluate_task, path, args.max_shift, args.preview)
                for path in files
            ]
            for future in as_completed(futures):
                results.append(future.result())

    task_count = 0
    for result in sorted(results, key=lambda item: item["task"]):
        task_count += 1
        if result["rules"]:
            tasks_with_rules += 1
        if result["metric"]["oracle"] > 0:
            tasks_with_oracle += 1
        merge_metric(total, result["metric"])
        if len(previews) < args.preview:
            for item in result["preview"]:
                previews.append({"task": result["task"], **item})
                if len(previews) >= args.preview:
                    break
        per_task.append(
            {
                "task": result["task"],
                "rule_count": len(result["rules"]),
                "rules": result["rules"][:10],
                "metric": result["metric"],
            }
        )

    summary = {
        "data_root": str(args.data_root),
        "split": args.split,
        "max_shift": args.max_shift,
        "tasks_evaluated": task_count,
        "tasks_with_rules": tasks_with_rules,
        "tasks_with_oracle": tasks_with_oracle,
        "metric": finalize(total, task_count),
        "preview": previews,
        "per_task": per_task,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
