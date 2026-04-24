#!/usr/bin/env python3
"""Small ARC candidate-reranking lab for physics-inspired priors.

This is intentionally lightweight: it does not train VARC. It asks whether
"natural law" features such as shape conservation, color entropy, object
count, and relation consistency across demonstrations can identify the true
output among structured distractors.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ARC_COLOR_COUNT = 10
MODEL_TOKEN_COUNT = 12
TRANSFORM_NAMES = [
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flipud",
    "fliplr",
    "transpose",
    "anti_transpose",
]


@dataclass(frozen=True)
class Candidate:
    label: str
    grid: list[list[int]]
    oracle_derived: bool = False


@dataclass(frozen=True)
class TransformColorRule:
    transform: str
    color_map: dict[int, int]


def as_array(grid: list[list[int]]) -> np.ndarray:
    array = np.asarray(grid, dtype=np.int64)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        if array.size == 0:
            return np.zeros((0, 0), dtype=np.int64)
        return array.reshape(1, -1)
    if array.ndim > 2:
        return array.reshape(array.shape[0], -1)
    return array


def as_grid(array: np.ndarray) -> list[list[int]]:
    return np.asarray(array, dtype=np.int64).tolist()


def serialize(grid: list[list[int]]) -> str:
    return json.dumps(grid, separators=(",", ":"))


def color_hist(array: np.ndarray, color_count: int = MODEL_TOKEN_COUNT) -> np.ndarray:
    hist = np.bincount(array.reshape(-1), minlength=color_count).astype(np.float64)
    if hist.size > color_count:
        hist = hist[:color_count]
    total = hist.sum()
    return hist / max(total, 1.0)


def entropy(array: np.ndarray) -> float:
    hist = color_hist(array)
    probs = hist[hist > 0]
    return float(-(probs * np.log2(probs)).sum())


def foreground_fraction(array: np.ndarray) -> float:
    # ARC convention often uses 0 as background. This is a heuristic, not a rule.
    return float((array != 0).mean()) if array.size else 0.0


def invalid_token_fraction(array: np.ndarray) -> float:
    if array.size == 0:
        return 0.0
    return float(((array < 0) | (array >= ARC_COLOR_COUNT)).mean())


def component_count(array: np.ndarray) -> int:
    if array.size == 0:
        return 0
    height, width = array.shape
    seen = np.zeros((height, width), dtype=bool)
    count = 0
    for y in range(height):
        for x in range(width):
            if seen[y, x] or array[y, x] == 0:
                continue
            count += 1
            color = array[y, x]
            queue: deque[tuple[int, int]] = deque([(y, x)])
            seen[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and not seen[ny, nx]
                        and array[ny, nx] == color
                    ):
                        seen[ny, nx] = True
                        queue.append((ny, nx))
    return count


def relation_features(input_grid: list[list[int]], output_grid: list[list[int]]) -> np.ndarray:
    x = as_array(input_grid)
    y = as_array(output_grid)
    xh, xw = x.shape
    yh, yw = y.shape
    same_shape = float(x.shape == y.shape)
    same_pixels = 0.0
    changed_pixels = 1.0
    if x.shape == y.shape and x.size:
        same_pixels = float((x == y).mean())
        changed_pixels = 1.0 - same_pixels

    hist_delta = float(np.abs(color_hist(y) - color_hist(x)).sum())
    return np.asarray(
        [
            (yh - xh) / 30.0,
            (yw - xw) / 30.0,
            math.log((y.size + 1.0) / (x.size + 1.0)),
            same_shape,
            (len(np.unique(y)) - len(np.unique(x))) / MODEL_TOKEN_COUNT,
            (entropy(y) - entropy(x)) / math.log2(MODEL_TOKEN_COUNT),
            (component_count(y) - component_count(x)) / 30.0,
            foreground_fraction(y) - foreground_fraction(x),
            hist_delta,
            same_pixels,
            changed_pixels,
            invalid_token_fraction(y),
        ],
        dtype=np.float64,
    )


def candidate_complexity(grid: list[list[int]]) -> float:
    y = as_array(grid)
    palette = len(np.unique(y)) / MODEL_TOKEN_COUNT
    comps = min(component_count(y), 30) / 30.0
    ent = entropy(y) / math.log2(MODEL_TOKEN_COUNT)
    fg = foreground_fraction(y)
    invalid = invalid_token_fraction(y)
    return 0.20 * ent + 0.08 * palette + 0.08 * comps + 0.04 * fg + 8.0 * invalid


def relation_energy(
    demos: list[dict],
    test_input: list[list[int]],
    candidate: list[list[int]],
) -> float:
    demo_features = np.stack(
        [relation_features(ex["input"], ex["output"]) for ex in demos],
        axis=0,
    )
    mean = demo_features.mean(axis=0)
    std = demo_features.std(axis=0)
    current = relation_features(test_input, candidate)

    # A floor avoids letting 2-4 demos overstate certainty.
    z = np.abs(current - mean) / (std + 0.15)
    weights = np.asarray(
        [1.2, 1.2, 0.9, 1.5, 0.7, 0.8, 1.1, 0.9, 1.0, 0.7, 0.7, 4.0],
        dtype=np.float64,
    )
    return float((z * weights).sum() + candidate_complexity(candidate))


def complexity_energy(
    _demos: list[dict],
    _test_input: list[list[int]],
    candidate: list[list[int]],
) -> float:
    return candidate_complexity(candidate)


def apply_transform(array: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return array
    if name == "rot90":
        return np.rot90(array, 1)
    if name == "rot180":
        return np.rot90(array, 2)
    if name == "rot270":
        return np.rot90(array, 3)
    if name == "flipud":
        return np.flipud(array)
    if name == "fliplr":
        return np.fliplr(array)
    if name == "transpose":
        return array.T
    if name == "anti_transpose":
        return np.fliplr(np.flipud(array)).T
    raise ValueError(name)


def infer_color_map_from_arrays(
    source: np.ndarray,
    target: np.ndarray,
    mapping: dict[int, int] | None = None,
) -> dict[int, int] | None:
    if source.shape != target.shape:
        return None
    merged = {} if mapping is None else dict(mapping)
    for src, dst in zip(source.reshape(-1), target.reshape(-1)):
        src_i, dst_i = int(src), int(dst)
        if src_i in merged and merged[src_i] != dst_i:
            return None
        merged[src_i] = dst_i
    return merged


def infer_color_map(demos: list[dict]) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    for ex in demos:
        x = as_array(ex["input"])
        y = as_array(ex["output"])
        updated = infer_color_map_from_arrays(x, y, mapping)
        if updated is None:
            return None
        mapping = updated
    return mapping


def apply_color_map(grid: list[list[int]], mapping: dict[int, int]) -> list[list[int]]:
    x = as_array(grid)
    out = np.vectorize(lambda value: mapping.get(int(value), int(value)))(x)
    return as_grid(out)


def infer_exact_transforms(demos: list[dict]) -> list[str]:
    valid = []
    for name in TRANSFORM_NAMES:
        ok = True
        for ex in demos:
            transformed = apply_transform(as_array(ex["input"]), name)
            if transformed.shape != as_array(ex["output"]).shape:
                ok = False
                break
            if not np.array_equal(transformed, as_array(ex["output"])):
                ok = False
                break
        if ok:
            valid.append(name)
    return valid


def infer_transform_color_rules(demos: list[dict]) -> list[TransformColorRule]:
    if not demos:
        return []

    rules = []
    for name in TRANSFORM_NAMES:
        mapping: dict[int, int] | None = {}
        for ex in demos:
            transformed = apply_transform(as_array(ex["input"]), name)
            target = as_array(ex["output"])
            mapping = infer_color_map_from_arrays(transformed, target, mapping)
            if mapping is None:
                break
        if mapping is not None:
            rules.append(TransformColorRule(transform=name, color_map=mapping))
    return rules


def apply_transform_color_rule(grid: list[list[int]], rule: TransformColorRule) -> list[list[int]]:
    transformed = apply_transform(as_array(grid), rule.transform)
    out = np.vectorize(lambda value: rule.color_map.get(int(value), int(value)))(transformed)
    return as_grid(out)


def grid_mismatch_fraction(expected: list[list[int]], candidate: list[list[int]]) -> float:
    expected_array = as_array(expected)
    candidate_array = as_array(candidate)
    if expected_array.shape != candidate_array.shape:
        expected_size = max(expected_array.size, 1)
        candidate_size = max(candidate_array.size, 1)
        return 1.0 + abs(math.log(candidate_size / expected_size))
    if expected_array.size == 0:
        return 0.0
    return float((expected_array != candidate_array).mean())


def symbolic_rule_energy(
    demos: list[dict],
    test_input: list[list[int]],
    candidate: list[list[int]],
) -> float:
    return symbolic_rule_energy_for_rules(
        infer_transform_color_rules(demos),
        test_input,
        candidate,
    )


def symbolic_rule_energy_for_rules(
    rules: list[TransformColorRule],
    test_input: list[list[int]],
    candidate: list[list[int]],
) -> float:
    if not rules:
        return 0.0

    best_mismatch = min(
        grid_mismatch_fraction(apply_transform_color_rule(test_input, rule), candidate)
        for rule in rules
    )
    if best_mismatch == 0.0:
        return -4.0
    return 10.0 * best_mismatch


def relation_symbolic_energy(
    demos: list[dict],
    test_input: list[list[int]],
    candidate: list[list[int]],
) -> float:
    return relation_energy(demos, test_input, candidate) + symbolic_rule_energy(
        demos,
        test_input,
        candidate,
    )


def corrupt_grid(grid: list[list[int]], rng: random.Random, p: float = 0.08) -> list[list[int]]:
    y = as_array(grid).copy()
    if y.size == 0:
        return as_grid(y)
    mask = np.asarray([rng.random() < p for _ in range(y.size)], dtype=bool).reshape(y.shape)
    if mask.any():
        y[mask] = np.asarray([rng.randrange(ARC_COLOR_COUNT) for _ in range(int(mask.sum()))], dtype=np.int64)
    return as_grid(y)


def shift_grid(grid: list[list[int]]) -> list[list[int]]:
    y = as_array(grid)
    if y.size == 0:
        return as_grid(y)
    return as_grid(np.roll(y, shift=1, axis=1))


def permute_two_colors(grid: list[list[int]]) -> list[list[int]]:
    y = as_array(grid).copy()
    colors = [int(c) for c in np.unique(y) if int(c) != 0]
    if len(colors) < 2:
        colors = [int(c) for c in np.unique(y)]
    if len(colors) >= 2:
        a, b = colors[0], colors[-1]
        ya = y == a
        yb = y == b
        y[ya] = b
        y[yb] = a
    return as_grid(y)


def candidate_pool(task: dict, test_ex: dict, rng: random.Random) -> list[Candidate]:
    demos = task["train"]
    test_input = test_ex["input"]
    truth = test_ex["output"]
    x = as_array(test_input)
    truth_arr = as_array(truth)

    candidates: list[Candidate] = [
        Candidate("ground_truth", truth, oracle_derived=True),
        Candidate("blank_truth_shape", np.zeros_like(truth_arr).tolist(), oracle_derived=True),
        Candidate("noisy_truth", corrupt_grid(truth, rng), oracle_derived=True),
        Candidate("shifted_truth", shift_grid(truth), oracle_derived=True),
        Candidate("color_swapped_truth", permute_two_colors(truth), oracle_derived=True),
        Candidate("identity_input", test_input),
    ]

    for name in ["rot90", "rot180", "rot270", "flipud", "fliplr", "transpose", "anti_transpose"]:
        candidates.append(Candidate(f"input_{name}", as_grid(apply_transform(x, name))))

    for idx, ex in enumerate(demos):
        candidates.append(Candidate(f"demo_output_{idx}", ex["output"]))

    mapping = infer_color_map(demos)
    if mapping is not None:
        candidates.append(Candidate("inferred_color_map", apply_color_map(test_input, mapping)))

    for name in infer_exact_transforms(demos):
        candidates.append(Candidate(f"inferred_transform_{name}", as_grid(apply_transform(x, name))))

    unique: dict[str, Candidate] = {}
    for cand in candidates:
        key = serialize(cand.grid)
        existing = unique.get(key)
        if existing is None or (existing.oracle_derived and not cand.oracle_derived):
            unique[key] = cand
    return list(unique.values())


def iter_tasks(data_root: Path, split: str, limit: int | None) -> Iterable[tuple[str, dict]]:
    files = sorted((data_root / "data" / split).glob("*.json"))
    if limit is not None:
        files = files[:limit]
    for path in files:
        with path.open("r") as handle:
            yield path.stem, json.load(handle)


def evaluate(data_root: Path, split: str, limit: int | None, seed: int) -> dict:
    rng = random.Random(seed)
    metrics = {
        "examples": 0,
        "oracle_top1": 0,
        "oracle_top2": 0,
        "complexity_top1": 0,
        "non_oracle_pool_has_truth": 0,
        "non_oracle_top1_when_available": 0,
        "oracle_rank_sum": 0.0,
        "pool_size_sum": 0,
        "label_wins": {},
    }
    examples = []

    for task_name, task in iter_tasks(data_root, split, limit):
        for test_idx, test_ex in enumerate(task.get("test", [])):
            if "output" not in test_ex:
                continue
            pool = candidate_pool(task, test_ex, rng)
            truth_key = serialize(test_ex["output"])
            scored = sorted(
                (
                    (relation_energy(task["train"], test_ex["input"], cand.grid), cand)
                    for cand in pool
                ),
                key=lambda item: item[0],
            )
            complexity_scored = sorted(
                (
                    (complexity_energy(task["train"], test_ex["input"], cand.grid), cand)
                    for cand in pool
                ),
                key=lambda item: item[0],
            )

            labels = [cand.label for _, cand in scored]
            truth_rank = next(
                idx + 1 for idx, (_, cand) in enumerate(scored) if serialize(cand.grid) == truth_key
            )
            non_oracle_has_truth = any(
                (not cand.oracle_derived) and serialize(cand.grid) == truth_key
                for cand in pool
            )

            metrics["examples"] += 1
            metrics["oracle_top1"] += int(truth_rank == 1)
            metrics["oracle_top2"] += int(truth_rank <= 2)
            metrics["complexity_top1"] += int(serialize(complexity_scored[0][1].grid) == truth_key)
            metrics["non_oracle_pool_has_truth"] += int(non_oracle_has_truth)
            metrics["non_oracle_top1_when_available"] += int(non_oracle_has_truth and truth_rank == 1)
            metrics["oracle_rank_sum"] += truth_rank
            metrics["pool_size_sum"] += len(pool)
            metrics["label_wins"][labels[0]] = metrics["label_wins"].get(labels[0], 0) + 1

            if len(examples) < 20:
                examples.append(
                    {
                        "task": task_name,
                        "test_index": test_idx,
                        "truth_rank": truth_rank,
                        "pool_size": len(pool),
                        "top5": [
                            {"rank": idx + 1, "energy": round(score, 4), "label": cand.label}
                            for idx, (score, cand) in enumerate(scored[:5])
                        ],
                    }
                )

    n = max(metrics["examples"], 1)
    available = max(metrics["non_oracle_pool_has_truth"], 1)
    summary = {
        "examples": metrics["examples"],
        "oracle_top1_rate": metrics["oracle_top1"] / n,
        "oracle_top2_rate": metrics["oracle_top2"] / n,
        "complexity_only_top1_rate": metrics["complexity_top1"] / n,
        "mean_oracle_rank": metrics["oracle_rank_sum"] / n,
        "mean_pool_size": metrics["pool_size_sum"] / n,
        "non_oracle_pool_has_truth_rate": metrics["non_oracle_pool_has_truth"] / n,
        "non_oracle_top1_when_available_rate": metrics["non_oracle_top1_when_available"] / available,
        "label_wins": dict(sorted(metrics["label_wins"].items(), key=lambda item: item[1], reverse=True)[:12]),
        "examples_preview": examples,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("raw_data/ARC-AGI"))
    parser.add_argument("--split", default="training")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    summary = evaluate(args.data_root, args.split, args.limit, args.seed)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
