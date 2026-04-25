#!/usr/bin/env python3
"""Evaluate uncertainty-gated symbolic candidates as independent pass@2 channel."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from physics_reranker import relation_energy, serialize
from rerank_varc_predictions import (
    get_majority_vote,
    load_json,
    load_prediction_roots,
    normalize_index,
    pass_metrics_for_order,
    task_files,
)
from symbolic_candidate_search import dict_entry, generate_broad_symbolic_candidates


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
    return {
        "pass_at_1": 0.0,
        "pass_at_2": 0.0,
        "oracle": 0.0,
        "examples": 0.0,
    }


def add_metric(metric: dict[str, float], ordered: list[dict], truth: list[list[int]], weight: float) -> None:
    pass1, pass2, rank = pass_metrics_for_order(ordered, truth)
    metric["pass_at_1"] += weight if pass1 else 0.0
    metric["pass_at_2"] += weight if pass2 else 0.0
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


def symbolic_order(task: dict, test_input: list[list[int]], max_candidates: int) -> list[dict]:
    entries = [
        dict_entry(candidate)
        for candidate in generate_broad_symbolic_candidates(
            task, test_input, max_candidates=max_candidates
        )
    ]
    return sorted(
        (
            {
                **entry,
                "energy": relation_energy(task.get("train", []), test_input, entry["prediction"]),
            }
            for entry in entries
        ),
        key=lambda entry: entry["energy"],
    )


def evaluate_task(
    task_path_text: str,
    official_roots: str,
    ratio_thresholds: list[float],
    symbolic_energy_thresholds: list[float],
    max_candidates: int,
) -> dict[str, Any]:
    task_path = Path(task_path_text)
    task_name = task_path.stem
    task = load_json(task_path)
    official_predictions = load_prediction_roots(official_roots, task_name)
    if official_predictions is None:
        official_predictions = {}

    metrics: dict[str, dict[str, float]] = {}
    test_examples = task.get("test", [])
    weight = 1.0 / max(len(test_examples), 1)
    previews = []
    for idx, test_ex in enumerate(test_examples):
        if "output" not in test_ex:
            continue
        key = normalize_index(idx)
        official = get_majority_vote(official_predictions.get(key, []))
        symbolic = symbolic_order(task, test_ex["input"], max_candidates)
        truth = test_ex["output"]

        orders = {
            "official_majority": official,
            "symbolic_first": unique_concat(symbolic, official),
            "official_then_symbolic_top1": unique_concat(
                official[:1], symbolic[:1], official, symbolic
            ),
        }
        if len(official) >= 2:
            top_votes = max(official[0]["votes"], 1)
            second_votes = max(official[1]["votes"], 1)
            ratio = top_votes / second_votes
            for threshold in ratio_thresholds:
                second = symbolic[:1] if symbolic and ratio <= threshold else official[1:2]
                orders[f"uncertain_symbolic_if_ratio<={threshold:g}"] = unique_concat(
                    official[:1], second, official, symbolic
                )
        if symbolic:
            best_energy = float(symbolic[0]["energy"])
            for threshold in symbolic_energy_thresholds:
                second = symbolic[:1] if best_energy <= threshold else official[1:2]
                orders[f"symbolic_if_energy<={threshold:g}"] = unique_concat(
                    official[:1], second, official, symbolic
                )

        for name, ordered in orders.items():
            metrics.setdefault(name, init_metric())
            add_metric(metrics[name], ordered, truth, weight)

        if len(previews) < 10:
            previews.append(
                {
                    "task": task_name,
                    "official_top2": [
                        {"votes": item["votes"], "matches": item["prediction"] == truth}
                        for item in official[:2]
                    ],
                    "symbolic_top2": [
                        {
                            "energy": round(float(item["energy"]), 4),
                            "matches": item["prediction"] == truth,
                            "label": item.get("label"),
                        }
                        for item in symbolic[:2]
                    ],
                }
            )

    return {"task": task_name, "metrics": metrics, "preview": previews}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    files = task_files(args.data_root, args.split, args.limit)
    if args.workers <= 1:
        results = [
            evaluate_task(
                str(path),
                args.official_roots,
                args.ratio_thresholds,
                args.symbolic_energy_thresholds,
                args.max_candidates,
            )
            for path in files
        ]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    evaluate_task,
                    str(path),
                    args.official_roots,
                    args.ratio_thresholds,
                    args.symbolic_energy_thresholds,
                    args.max_candidates,
                )
                for path in files
            ]
            for future in as_completed(futures):
                results.append(future.result())

    totals: dict[str, dict[str, float]] = {}
    previews = []
    task_count = 0
    for result in sorted(results, key=lambda item: item["task"]):
        task_count += 1
        for name, metric in result["metrics"].items():
            totals.setdefault(name, init_metric())
            merge_metric(totals[name], metric)
        if len(previews) < args.preview:
            previews.extend(result["preview"][: args.preview - len(previews)])

    metrics = {name: finalize(metric, task_count) for name, metric in totals.items()}
    best_name, best_metric = max(
        metrics.items(),
        key=lambda item: (item[1]["pass_at_2"], item[1]["pass_at_1"]),
    )
    return {
        "data_root": str(args.data_root),
        "split": args.split,
        "tasks_evaluated": task_count,
        "best": {"strategy": best_name, **best_metric},
        "metrics": metrics,
        "preview": previews,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("raw_data/ARC-AGI"))
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--official-roots", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument(
        "--ratio-thresholds",
        type=float,
        nargs="+",
        default=[1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.5, 2.0],
    )
    parser.add_argument(
        "--symbolic-energy-thresholds",
        type=float,
        nargs="+",
        default=[2, 4, 6, 8, 10, 12, 16, 20],
    )
    parser.add_argument("--preview", type=int, default=12)
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
