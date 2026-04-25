#!/usr/bin/env python3
"""Generate exact symbolic ARC candidates and merge them with VARC predictions.

The generators are intentionally conservative: a rule must match every
demonstration exactly before it is applied to the test input. This makes the
experiment a fairer post-processing extension than using test labels directly.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from physics_reranker import (
    ARC_COLOR_COUNT,
    TRANSFORM_NAMES,
    TransformColorRule,
    apply_transform,
    as_array,
    as_grid,
    grid_mismatch_fraction,
    infer_color_map_from_arrays,
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
class SymbolicCandidate:
    label: str
    grid: list[list[int]]


def nonempty_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def crop_bbox(array: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    bbox = nonempty_bbox(mask)
    if bbox is None:
        return None
    y0, y1, x0, x1 = bbox
    return array[y0:y1, x0:x1]


def connected_components(array: np.ndarray, *, background: int = 0) -> list[dict]:
    if array.size == 0:
        return []
    h, w = array.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for y in range(h):
        for x in range(w):
            if seen[y, x] or int(array[y, x]) == background:
                continue
            color = int(array[y, x])
            q: deque[tuple[int, int]] = deque([(y, x)])
            seen[y, x] = True
            cells = []
            while q:
                cy, cx = q.popleft()
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
                        q.append((ny, nx))
            ys = [cell[0] for cell in cells]
            xs = [cell[1] for cell in cells]
            y0, y1, x0, x1 = min(ys), max(ys) + 1, min(xs), max(xs) + 1
            crop = array[y0:y1, x0:x1]
            comps.append(
                {
                    "color": color,
                    "area": len(cells),
                    "y0": y0,
                    "x0": x0,
                    "height": y1 - y0,
                    "width": x1 - x0,
                    "crop": crop,
                }
            )
    return comps


def extractor_fns(demos: list[dict]) -> list[tuple[str, GridFn]]:
    colors = sorted(
        {
            int(value)
            for ex in demos
            for grid_name in ("input", "output")
            for row in ex.get(grid_name, [])
            for value in row
            if 0 <= int(value) < ARC_COLOR_COUNT
        }
    )
    out: list[tuple[str, GridFn]] = [("identity", lambda x: x)]

    for bg in colors:
        out.append((f"bbox_not_{bg}", lambda x, bg=bg: crop_bbox(x, x != bg)))
    for color in colors:
        out.append((f"bbox_color_{color}", lambda x, color=color: crop_bbox(x, x == color)))

    selectors: list[tuple[str, Callable[[list[dict]], dict | None]]] = [
        ("largest_component", lambda comps: max(comps, key=lambda c: (c["area"], -c["y0"], -c["x0"])) if comps else None),
        ("smallest_component", lambda comps: min(comps, key=lambda c: (c["area"], c["y0"], c["x0"])) if comps else None),
        ("topmost_component", lambda comps: min(comps, key=lambda c: (c["y0"], c["x0"])) if comps else None),
        ("leftmost_component", lambda comps: min(comps, key=lambda c: (c["x0"], c["y0"])) if comps else None),
        ("bottommost_component", lambda comps: max(comps, key=lambda c: (c["y0"], -c["x0"])) if comps else None),
        ("rightmost_component", lambda comps: max(comps, key=lambda c: (c["x0"], -c["y0"])) if comps else None),
    ]
    for color in colors:
        selectors.append(
            (
                f"component_color_{color}",
                lambda comps, color=color: next((comp for comp in comps if comp["color"] == color), None),
            )
        )
    for name, selector in selectors:
        out.append(
            (
                name,
                lambda x, selector=selector: (
                    None
                    if selector(connected_components(x)) is None
                    else selector(connected_components(x))["crop"]
                ),
            )
        )
    return out


def infer_extractor_transform_color_rules(demos: list[dict]) -> list[tuple[str, TransformColorRule]]:
    rules: list[tuple[str, TransformColorRule]] = []
    for extractor_name, extractor in extractor_fns(demos):
        for transform in TRANSFORM_NAMES:
            mapping: dict[int, int] | None = {}
            ok = True
            for ex in demos:
                base = extractor(as_array(ex["input"]))
                if base is None:
                    ok = False
                    break
                transformed = apply_transform(base, transform)
                target = as_array(ex["output"])
                mapping = infer_color_map_from_arrays(transformed, target, mapping)
                if mapping is None:
                    ok = False
                    break
            if ok and mapping is not None:
                rules.append((extractor_name, TransformColorRule(transform, mapping)))
    return rules


def apply_extractor_rule(
    array: np.ndarray,
    extractor_name: str,
    rule: TransformColorRule,
    demos: list[dict],
) -> np.ndarray | None:
    extractors = dict(extractor_fns(demos))
    base = extractors[extractor_name](array)
    if base is None:
        return None
    transformed = apply_transform(base, rule.transform)
    return np.vectorize(lambda value: rule.color_map.get(int(value), int(value)))(transformed)


def infer_tile_rules(demos: list[dict]) -> list[tuple[int, int]]:
    rules = []
    for ex in demos:
        x = as_array(ex["input"])
        y = as_array(ex["output"])
        if x.size == 0 or y.shape[0] % x.shape[0] or y.shape[1] % x.shape[1]:
            return []
        rules.append((y.shape[0] // x.shape[0], y.shape[1] // x.shape[1]))
    unique = sorted(set(rules))
    valid = []
    for reps in unique:
        if all(np.array_equal(np.tile(as_array(ex["input"]), reps), as_array(ex["output"])) for ex in demos):
            valid.append(reps)
    return valid


def infer_concat_rules(demos: list[dict]) -> list[tuple[str, str, str]]:
    rules = []
    for axis_name, concat_fn in (("hstack", np.hstack), ("vstack", np.vstack)):
        for left in TRANSFORM_NAMES:
            for right in TRANSFORM_NAMES:
                ok = True
                for ex in demos:
                    x = as_array(ex["input"])
                    try:
                        candidate = concat_fn([apply_transform(x, left), apply_transform(x, right)])
                    except ValueError:
                        ok = False
                        break
                    if not np.array_equal(candidate, as_array(ex["output"])):
                        ok = False
                        break
                if ok:
                    rules.append((axis_name, left, right))
    return rules


def generate_symbolic_candidates(task: dict, test_input: list[list[int]]) -> list[SymbolicCandidate]:
    demos = task.get("train", [])
    x = as_array(test_input)
    candidates: list[SymbolicCandidate] = []

    for extractor_name, rule in infer_extractor_transform_color_rules(demos):
        out = apply_extractor_rule(x, extractor_name, rule, demos)
        if out is not None:
            candidates.append(
                SymbolicCandidate(
                    f"{extractor_name}+{rule.transform}+color_map",
                    as_grid(out),
                )
            )

    for reps in infer_tile_rules(demos):
        candidates.append(SymbolicCandidate(f"tile_{reps[0]}x{reps[1]}", as_grid(np.tile(x, reps))))

    for axis_name, left, right in infer_concat_rules(demos):
        concat_fn = np.hstack if axis_name == "hstack" else np.vstack
        try:
            out = concat_fn([apply_transform(x, left), apply_transform(x, right)])
        except ValueError:
            continue
        candidates.append(SymbolicCandidate(f"{axis_name}_{left}_{right}", as_grid(out)))

    unique = {}
    for candidate in candidates:
        unique.setdefault(serialize(candidate.grid), candidate)
    return list(unique.values())


def generate_broad_symbolic_candidates(task: dict, test_input: list[list[int]], *, max_candidates: int) -> list[SymbolicCandidate]:
    demos = task.get("train", [])
    x = as_array(test_input)
    candidates: list[SymbolicCandidate] = []

    for extractor_name, extractor in extractor_fns(demos):
        base = extractor(x)
        if base is None or base.size == 0:
            continue
        for transform in TRANSFORM_NAMES:
            try:
                candidates.append(
                    SymbolicCandidate(
                        f"broad_{extractor_name}_{transform}",
                        as_grid(apply_transform(base, transform)),
                    )
                )
            except ValueError:
                continue

    for reps in {(1, 2), (2, 1), (2, 2), (1, 3), (3, 1)}:
        candidates.append(SymbolicCandidate(f"broad_tile_{reps[0]}x{reps[1]}", as_grid(np.tile(x, reps))))

    for axis_name, concat_fn in (("hstack", np.hstack), ("vstack", np.vstack)):
        for left in TRANSFORM_NAMES:
            for right in TRANSFORM_NAMES:
                try:
                    out = concat_fn([apply_transform(x, left), apply_transform(x, right)])
                except ValueError:
                    continue
                candidates.append(SymbolicCandidate(f"broad_{axis_name}_{left}_{right}", as_grid(out)))

    unique: dict[str, SymbolicCandidate] = {}
    for candidate in candidates:
        unique.setdefault(serialize(candidate.grid), candidate)
    if len(unique) <= max_candidates:
        return list(unique.values())

    scored = sorted(
        (
            (relation_energy(demos, test_input, candidate.grid), candidate)
            for candidate in unique.values()
        ),
        key=lambda item: item[0],
    )
    return [candidate for _, candidate in scored[:max_candidates]]


def dict_entry(candidate: SymbolicCandidate) -> dict:
    return {"prediction": candidate.grid, "votes": 0, "source": "symbolic", "label": candidate.label}


def unique_concat(*orders: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for order in orders:
        for entry in order:
            key = serialize(entry["prediction"])
            if key not in seen:
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


def evaluate_task(task_path_text: str, output_root: str) -> dict:
    task_path = Path(task_path_text)
    task_name = task_path.stem
    task = load_json(task_path)
    predictions_by_index = load_prediction_roots(output_root, task_name)
    if predictions_by_index is None:
        predictions_by_index = {}

    test_examples = task.get("test", [])
    weight = 1.0 / max(len(test_examples), 1)
    metrics = {name: init_metric() for name in [
        "varc_majority",
        "symbolic_only",
        "symbolic_first",
        "varc_first_symbolic_second",
        "broad_symbolic_first",
        "varc_first_broad_second",
        "combined_energy",
    ]}
    symbol_count = 0
    symbol_hit = 0
    broad_count = 0
    broad_hit = 0.0

    for idx, test_ex in enumerate(test_examples):
        if "output" not in test_ex:
            continue
        key = normalize_index(idx)
        varc_order = get_majority_vote(predictions_by_index.get(key, []))
        symbolic_entries = [dict_entry(c) for c in generate_symbolic_candidates(task, test_ex["input"])]
        broad_candidates = generate_broad_symbolic_candidates(task, test_ex["input"], max_candidates=64)
        broad_entries = [dict_entry(c) for c in broad_candidates]
        broad_scored = sorted(
            (
                {
                    **entry,
                    "energy": relation_energy(task.get("train", []), test_ex["input"], entry["prediction"]),
                }
                for entry in broad_entries
            ),
            key=lambda entry: entry["energy"],
        )
        varc_energy = sorted(
            (
                {
                    **entry,
                    "energy": relation_energy(task.get("train", []), test_ex["input"], entry["prediction"]),
                }
                for entry in varc_order
            ),
            key=lambda entry: (entry["energy"] - 1.75 * np.log1p(entry["votes"]), -entry["votes"]),
        )
        symbol_count += len(symbolic_entries)
        broad_count += len(broad_entries)
        truth = test_ex["output"]
        if any(entry["prediction"] == truth for entry in symbolic_entries):
            symbol_hit += weight
        if any(entry["prediction"] == truth for entry in broad_entries):
            broad_hit += weight

        orders = {
            "varc_majority": varc_order,
            "symbolic_only": symbolic_entries,
            "symbolic_first": unique_concat(symbolic_entries, varc_order),
            "varc_first_symbolic_second": unique_concat(varc_order[:1], symbolic_entries, varc_order[1:]),
            "broad_symbolic_first": unique_concat(broad_scored, varc_order),
            "varc_first_broad_second": unique_concat(varc_order[:1], broad_scored, varc_order[1:]),
            "combined_energy": unique_concat(varc_energy, broad_scored, varc_order),
        }
        for name, ordered in orders.items():
            add_metric(metrics[name], ordered, truth, weight)

    return {
        "task": task_name,
        "missing": False,
        "metrics": metrics,
        "symbol_count": symbol_count,
        "symbol_hit": symbol_hit,
        "broad_count": broad_count,
        "broad_hit": broad_hit,
    }


def evaluate(args: argparse.Namespace) -> dict:
    files = task_files(args.data_root, args.split, args.limit)
    workers = max(args.workers, 1)
    if workers == 1:
        results = [evaluate_task(str(path), args.output_root) for path in files]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(evaluate_task, str(path), args.output_root) for path in files]
            for future in as_completed(futures):
                results.append(future.result())

    totals: dict[str, dict[str, float]] = {}
    task_count = 0
    symbol_count = 0
    symbol_hit = 0.0
    broad_count = 0
    broad_hit = 0.0
    for result in sorted(results, key=lambda item: item["task"]):
        task_count += 1
        symbol_count += result["symbol_count"]
        symbol_hit += result["symbol_hit"]
        broad_count += result["broad_count"]
        broad_hit += result["broad_hit"]
        for name, metric in result["metrics"].items():
            totals.setdefault(name, init_metric())
            merge_metric(totals[name], metric)

    metrics = {name: finalize(metric, task_count) for name, metric in totals.items()}
    best_name, best_metric = max(metrics.items(), key=lambda item: (item[1]["pass_at_2"], item[1]["pass_at_1"]))
    return {
        "data_root": str(args.data_root),
        "split": args.split,
        "output_root": args.output_root,
        "tasks_evaluated": task_count,
        "symbolic_candidates_generated": symbol_count,
        "symbolic_oracle": symbol_hit / max(task_count, 1),
        "broad_candidates_generated": broad_count,
        "broad_symbolic_oracle": broad_hit / max(task_count, 1),
        "best": {"strategy": best_name, **best_metric},
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("raw_data/ARC-AGI"))
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    summary = evaluate(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
