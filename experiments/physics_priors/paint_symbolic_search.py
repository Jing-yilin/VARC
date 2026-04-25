#!/usr/bin/env python3
"""Search conservative paint-on-canvas symbolic rules for ARC/VARC.

The previous symbolic bank mostly produced crops, transforms, tiles, and
concatenations. This file targets a different physical primitive: keep the
canvas and paint a low-entropy structure on it. A rule is only used when it
matches every demonstration exactly.
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

from physics_reranker import (
    ARC_COLOR_COUNT,
    as_array,
    as_grid,
    grid_mismatch_fraction,
    relation_energy,
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


GridFn = Callable[[np.ndarray], np.ndarray | None]


@dataclass(frozen=True)
class PaintCandidate:
    label: str
    grid: list[list[int]]
    demo_error: float = 0.0


def _colors(demos: list[dict]) -> list[int]:
    values = {
        int(value)
        for ex in demos
        for name in ("input", "output")
        for row in ex.get(name, [])
        for value in row
        if 0 <= int(value) < ARC_COLOR_COUNT
    }
    return sorted(values) if values else list(range(ARC_COLOR_COUNT))


def _nonzero_colors(demos: list[dict]) -> list[int]:
    return [color for color in _colors(demos) if color != 0]


def _aligned_step(a: tuple[int, int], b: tuple[int, int], directions: str) -> tuple[int, int] | None:
    y0, x0 = a
    y1, x1 = b
    dy = y1 - y0
    dx = x1 - x0
    if dy == 0 and dx == 0:
        return None
    if dy == 0 and directions in {"hv", "all"}:
        return 0, 1 if dx > 0 else -1
    if dx == 0 and directions in {"hv", "all"}:
        return 1 if dy > 0 else -1, 0
    if abs(dy) == abs(dx) and directions in {"diag", "all"}:
        return 1 if dy > 0 else -1, 1 if dx > 0 else -1
    return None


def draw_aligned_lines(
    array: np.ndarray,
    source_color: int,
    paint_color: int,
    directions: str,
    overwrite: bool,
) -> np.ndarray | None:
    coords = list(zip(*np.where(array == source_color)))
    if len(coords) < 2 or len(coords) > 120:
        return None
    out = array.copy()
    changed = False
    for i, a in enumerate(coords):
        for b in coords[i + 1 :]:
            step = _aligned_step(a, b, directions)
            if step is None:
                continue
            y, x = a
            sy, sx = step
            while (y, x) != b:
                if overwrite or out[y, x] == 0:
                    changed = changed or int(out[y, x]) != paint_color
                    out[y, x] = paint_color
                y += sy
                x += sx
            if overwrite or out[y, x] == 0:
                changed = changed or int(out[y, x]) != paint_color
                out[y, x] = paint_color
    return out if changed else None


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def paint_bbox(
    array: np.ndarray,
    source_color: int,
    paint_color: int,
    mode: str,
    overwrite: bool,
) -> np.ndarray | None:
    bbox = _bbox(array == source_color)
    if bbox is None:
        return None
    y0, y1, x0, x1 = bbox
    if y1 - y0 <= 0 or x1 - x0 <= 0:
        return None
    mask = np.zeros_like(array, dtype=bool)
    if mode == "full":
        mask[y0:y1, x0:x1] = True
    elif mode == "border":
        mask[y0, x0:x1] = True
        mask[y1 - 1, x0:x1] = True
        mask[y0:y1, x0] = True
        mask[y0:y1, x1 - 1] = True
    elif mode == "interior":
        if y1 - y0 <= 2 or x1 - x0 <= 2:
            return None
        mask[y0 + 1 : y1 - 1, x0 + 1 : x1 - 1] = True
    else:
        raise ValueError(mode)
    if not overwrite:
        mask &= array == 0
    if not mask.any():
        return None
    out = array.copy()
    changed = bool((out[mask] != paint_color).any())
    out[mask] = paint_color
    return out if changed else None


def fill_enclosed(array: np.ndarray, fill_color: int, background: int = 0) -> np.ndarray | None:
    h, w = array.shape
    if h == 0 or w == 0:
        return None
    outside = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []
    for y in range(h):
        for x in (0, w - 1):
            if array[y, x] == background and not outside[y, x]:
                outside[y, x] = True
                stack.append((y, x))
    for x in range(w):
        for y in (0, h - 1):
            if array[y, x] == background and not outside[y, x]:
                outside[y, x] = True
                stack.append((y, x))
    while stack:
        y, x = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and array[ny, nx] == background and not outside[ny, nx]:
                outside[ny, nx] = True
                stack.append((ny, nx))
    enclosed = (array == background) & ~outside
    if not enclosed.any():
        return None
    out = array.copy()
    out[enclosed] = fill_color
    return out


def mirror_fill(array: np.ndarray, axis: str, source: str, overwrite: bool) -> np.ndarray | None:
    h, w = array.shape
    out = array.copy()
    changed = False
    for y, x in zip(*np.where(array != 0)):
        if axis == "vertical":
            if source == "left" and x >= w // 2:
                continue
            if source == "right" and x < (w + 1) // 2:
                continue
            yy, xx = y, w - 1 - x
        elif axis == "horizontal":
            if source == "top" and y >= h // 2:
                continue
            if source == "bottom" and y < (h + 1) // 2:
                continue
            yy, xx = h - 1 - y, x
        else:
            raise ValueError(axis)
        if overwrite or out[yy, xx] == 0:
            changed = changed or int(out[yy, xx]) != int(array[y, x])
            out[yy, xx] = array[y, x]
    return out if changed else None


def gravity(array: np.ndarray, direction: str) -> np.ndarray | None:
    out = np.zeros_like(array)
    h, w = array.shape
    if direction in {"left", "right"}:
        for y in range(h):
            vals = array[y, array[y] != 0]
            if direction == "left":
                out[y, : len(vals)] = vals
            else:
                out[y, w - len(vals) :] = vals
    elif direction in {"up", "down"}:
        for x in range(w):
            vals = array[array[:, x] != 0, x]
            if direction == "up":
                out[: len(vals), x] = vals
            else:
                out[h - len(vals) :, x] = vals
    else:
        raise ValueError(direction)
    return out if not np.array_equal(out, array) else None


def paint_rule_fns(demos: list[dict]) -> list[tuple[str, GridFn]]:
    colors = _colors(demos)
    source_colors = _nonzero_colors(demos)
    out: list[tuple[str, GridFn]] = []

    for source_color in source_colors:
        for paint_color in colors:
            for directions in ("hv", "diag", "all"):
                for overwrite in (False, True):
                    name = "over" if overwrite else "bg"
                    out.append(
                        (
                            f"line_{source_color}_{paint_color}_{directions}_{name}",
                            lambda x, source_color=source_color, paint_color=paint_color, directions=directions, overwrite=overwrite: draw_aligned_lines(
                                x, source_color, paint_color, directions, overwrite
                            ),
                        )
                    )
            for mode in ("full", "border", "interior"):
                for overwrite in (False, True):
                    name = "over" if overwrite else "bg"
                    out.append(
                        (
                            f"bbox_{source_color}_{paint_color}_{mode}_{name}",
                            lambda x, source_color=source_color, paint_color=paint_color, mode=mode, overwrite=overwrite: paint_bbox(
                                x, source_color, paint_color, mode, overwrite
                            ),
                        )
                    )

    for fill_color in colors:
        out.append((f"fill_enclosed_{fill_color}", lambda x, fill_color=fill_color: fill_enclosed(x, fill_color)))

    for axis, sources in (("vertical", ("all", "left", "right")), ("horizontal", ("all", "top", "bottom"))):
        for source in sources:
            for overwrite in (False, True):
                name = "over" if overwrite else "bg"
                out.append(
                    (
                        f"mirror_{axis}_{source}_{name}",
                        lambda x, axis=axis, source=source, overwrite=overwrite: mirror_fill(
                            x, axis, source, overwrite
                        ),
                    )
                )

    for direction in ("left", "right", "up", "down"):
        out.append((f"gravity_{direction}", lambda x, direction=direction: gravity(x, direction)))
    return out


def infer_paint_rules(demos: list[dict], max_demo_error: float) -> list[tuple[str, GridFn, float]]:
    rules = []
    for label, fn in paint_rule_fns(demos):
        ok = True
        errors = []
        for ex in demos:
            out = fn(as_array(ex["input"]))
            if out is None:
                ok = False
                break
            error = grid_mismatch_fraction(ex["output"], as_grid(out))
            errors.append(error)
            if error > max_demo_error:
                ok = False
                break
        if ok:
            mean_error = float(np.mean(errors)) if errors else 0.0
            rules.append((label, fn, mean_error))
    return rules


def generate_paint_candidates(
    task: dict,
    test_input: list[list[int]],
    max_demo_error: float,
) -> list[PaintCandidate]:
    candidates = []
    for label, fn, demo_error in infer_paint_rules(task.get("train", []), max_demo_error):
        out = fn(as_array(test_input))
        if out is not None:
            candidates.append(PaintCandidate(label, as_grid(out), demo_error))
    unique: dict[str, PaintCandidate] = {}
    for candidate in candidates:
        unique.setdefault(serialize(candidate.grid), candidate)
    return list(unique.values())


def dict_entry(candidate: PaintCandidate) -> dict:
    return {
        "prediction": candidate.grid,
        "votes": 0,
        "source": "paint_symbolic",
        "label": candidate.label,
        "demo_error": candidate.demo_error,
    }


def unique_concat(*orders: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
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


def evaluate_task(
    task_path_text: str,
    output_root: str | None,
    max_demo_error: float,
    max_candidates: int,
) -> dict:
    task_path = Path(task_path_text)
    task_name = task_path.stem
    task = load_json(task_path)
    predictions_by_index = load_prediction_roots(output_root, task_name) if output_root else {}
    if predictions_by_index is None:
        predictions_by_index = {}

    metrics = {
        "varc_majority": init_metric(),
        "paint_only": init_metric(),
        "paint_first": init_metric(),
        "varc_first_paint_second": init_metric(),
        "paint_energy": init_metric(),
    }
    hit = 0.0
    count = 0
    previews = []
    test_examples = task.get("test", [])
    weight = 1.0 / max(len(test_examples), 1)
    for idx, test_ex in enumerate(test_examples):
        if "output" not in test_ex:
            continue
        truth = test_ex["output"]
        official = get_majority_vote(predictions_by_index.get(normalize_index(idx), []))
        paint = [
            dict_entry(candidate)
            for candidate in generate_paint_candidates(task, test_ex["input"], max_demo_error)
        ]
        paint_scored = sorted(
            (
                {
                    **entry,
                    "energy": relation_energy(task.get("train", []), test_ex["input"], entry["prediction"]),
                }
                for entry in paint
            ),
            key=lambda entry: (entry["demo_error"], entry["energy"]),
        )[:max_candidates]
        hit += weight if any(entry["prediction"] == truth for entry in paint) else 0.0
        count += len(paint)
        orders = {
            "varc_majority": official,
            "paint_only": paint,
            "paint_first": unique_concat(paint, official),
            "varc_first_paint_second": unique_concat(official[:1], paint_scored[:1], official, paint_scored),
            "paint_energy": unique_concat(paint_scored, official),
        }
        for name, ordered in orders.items():
            add_metric(metrics[name], ordered, truth, weight)
        if len(previews) < 3 and paint:
            previews.append(
                {
                    "task": task_name,
                    "test_index": idx,
                    "paint_top": [
                        {"label": item["label"], "matches": item["prediction"] == truth}
                        for item in paint_scored[:3]
                    ],
                    "official_top2": [
                        {"votes": item["votes"], "matches": item["prediction"] == truth}
                        for item in official[:2]
                    ],
                }
            )
    return {
        "task": task_name,
        "metrics": metrics,
        "paint_oracle": hit,
        "paint_candidates": count,
        "preview": previews,
    }


def evaluate(args: argparse.Namespace) -> dict:
    files = [str(path) for path in task_files(args.data_root, args.split, args.limit)]
    if args.workers <= 1:
        results = [
            evaluate_task(path, args.output_root, args.max_demo_error, args.max_candidates)
            for path in files
        ]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    evaluate_task,
                    path,
                    args.output_root,
                    args.max_demo_error,
                    args.max_candidates,
                )
                for path in files
            ]
            for future in as_completed(futures):
                results.append(future.result())

    totals: dict[str, dict[str, float]] = {}
    previews = []
    task_count = 0
    paint_oracle = 0.0
    paint_candidates = 0
    for result in sorted(results, key=lambda item: item["task"]):
        task_count += 1
        paint_oracle += result["paint_oracle"]
        paint_candidates += result["paint_candidates"]
        for name, metric in result["metrics"].items():
            totals.setdefault(name, init_metric())
            merge_metric(totals[name], metric)
        if len(previews) < args.preview:
            previews.extend(result["preview"][: args.preview - len(previews)])
    metrics = {name: finalize(metric, task_count) for name, metric in totals.items()}
    best_name, best_metric = max(metrics.items(), key=lambda item: (item[1]["pass_at_2"], item[1]["pass_at_1"]))
    return {
        "data_root": str(args.data_root),
        "split": args.split,
        "max_demo_error": args.max_demo_error,
        "max_candidates": args.max_candidates,
        "tasks_evaluated": task_count,
        "paint_candidates_generated": paint_candidates,
        "paint_oracle": paint_oracle / max(task_count, 1),
        "best": {"strategy": best_name, **best_metric},
        "metrics": metrics,
        "preview": previews,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("raw_data/ARC-AGI"))
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-demo-error", type=float, default=0.0)
    parser.add_argument("--max-candidates", type=int, default=128)
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    parser.add_argument("--preview", type=int, default=12)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    summary = evaluate(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
