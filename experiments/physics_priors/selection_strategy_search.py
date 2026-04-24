#!/usr/bin/env python3
"""Search pass@2 selection strategies over VARC candidate pools.

This is a post-processing lab: it does not retrain VARC. It asks whether the
official two-answer ARC protocol benefits from selecting a complementary second
answer instead of simply taking the majority-vote top two.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from rerank_varc_predictions import (
    get_majority_vote,
    load_json,
    load_prediction_roots,
    normalize_index,
    pass_metrics_for_order,
    score_entries,
    task_files,
)
from physics_reranker import serialize


ScoreFn = Callable[[dict, float], float]


def vote_feature(name: str) -> ScoreFn:
    if name == "log":
        return lambda entry, weight: entry["energy"] - weight * entry["vote_log"]
    if name == "linear":
        return lambda entry, weight: entry["energy"] - weight * entry["votes"]
    if name == "sqrt":
        return lambda entry, weight: entry["energy"] - weight * math.sqrt(entry["votes"])
    raise ValueError(name)


def order_by_score(scored: list[dict], weight: float, vote_mode: str) -> list[dict]:
    score_fn = vote_feature(vote_mode)
    return sorted(
        ({**entry, "score": score_fn(entry, weight)} for entry in scored),
        key=lambda item: (item["score"], -item["votes"]),
    )


def unique_concat(*orders: list[dict]) -> list[dict]:
    seen = set()
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
        "rank_sum": 0.0,
        "rank_count": 0.0,
    }


def add_metric(
    metric: dict[str, float],
    ordered: list[dict],
    ground_truth: list[list[int]],
    example_weight: float,
) -> None:
    pass1, pass2, truth_rank = pass_metrics_for_order(ordered, ground_truth)
    metric["pass_at_1"] += example_weight if pass1 else 0.0
    metric["pass_at_2"] += example_weight if pass2 else 0.0
    metric["oracle"] += example_weight if truth_rank is not None else 0.0
    metric["examples"] += 1.0
    if truth_rank is not None:
        metric["rank_sum"] += truth_rank
        metric["rank_count"] += 1.0


def merge_metric(total: dict[str, float], delta: dict[str, float]) -> None:
    for key, value in delta.items():
        total[key] = total.get(key, 0.0) + value


def finalize_metric(metric: dict[str, float], task_count: int) -> dict[str, float]:
    denom = max(float(task_count), 1.0)
    out = {
        "pass_at_1": metric["pass_at_1"] / denom,
        "pass_at_2": metric["pass_at_2"] / denom,
        "oracle": metric["oracle"] / denom,
        "examples": metric["examples"],
    }
    out["mean_truth_rank_when_present"] = (
        metric["rank_sum"] / metric["rank_count"] if metric["rank_count"] else None
    )
    return out


def strategy_orders(
    majority_order: list[dict],
    scored: list[dict],
    weights: list[float],
    vote_modes: list[str],
    topks: list[int],
    gap_thresholds: list[int],
) -> dict[str, list[dict]]:
    orders: dict[str, list[dict]] = {"majority": majority_order}
    if not majority_order:
        return orders

    top_votes = majority_order[0]["votes"]
    second_votes = majority_order[1]["votes"] if len(majority_order) > 1 else 0
    gap = top_votes - second_votes

    for vote_mode in vote_modes:
        for weight in weights:
            scored_order = order_by_score(scored, weight, vote_mode)
            name = f"hybrid/{vote_mode}/w={weight:g}"
            orders[name] = scored_order

            anchored = unique_concat([majority_order[0]], scored_order, majority_order)
            orders[f"anchored_second/{vote_mode}/w={weight:g}"] = anchored

            for topk in topks:
                top_vote_pool = majority_order[:topk]
                top_keys = {serialize(item["prediction"]) for item in top_vote_pool}
                scored_topk = [item for item in scored_order if serialize(item["prediction"]) in top_keys]
                orders[f"rerank_top{topk}/{vote_mode}/w={weight:g}"] = unique_concat(
                    scored_topk,
                    majority_order,
                )

            for threshold in gap_thresholds:
                if gap <= threshold:
                    orders[f"gap_switch<= {threshold}/{vote_mode}/w={weight:g}"] = scored_order
                else:
                    orders[f"gap_switch<= {threshold}/{vote_mode}/w={weight:g}"] = majority_order

    return orders


def evaluate_task(
    task_path_text: str,
    output_root: str,
    energy_mode: str,
    weights: list[float],
    vote_modes: list[str],
    topks: list[int],
    gap_thresholds: list[int],
) -> dict:
    task_path = Path(task_path_text)
    task_name = task_path.stem
    predictions_by_index = load_prediction_roots(output_root, task_name)
    if predictions_by_index is None:
        return {"task": task_name, "missing": True}

    task = load_json(task_path)
    demos = task.get("train", [])
    test_examples = task.get("test", [])
    example_weight = 1.0 / max(len(test_examples), 1)
    metrics: dict[str, dict[str, float]] = {}

    for idx, test_ex in enumerate(test_examples):
        key = normalize_index(idx)
        if key not in predictions_by_index or "output" not in test_ex:
            continue
        majority_order = get_majority_vote(predictions_by_index[key])
        scored = score_entries(majority_order, demos, test_ex["input"], energy_mode)
        orders = strategy_orders(
            majority_order,
            scored,
            weights,
            vote_modes,
            topks,
            gap_thresholds,
        )
        ground_truth = test_ex["output"]
        for name, ordered in orders.items():
            metrics.setdefault(name, init_metric())
            add_metric(metrics[name], ordered, ground_truth, example_weight)

    return {"task": task_name, "missing": False, "metrics": metrics}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    files = task_files(args.data_root, args.split, args.limit)
    workers = max(args.workers, 1)
    if workers == 1:
        results = [
            evaluate_task(
                str(path),
                args.output_root,
                args.energy_mode,
                args.weights,
                args.vote_modes,
                args.topks,
                args.gap_thresholds,
            )
            for path in files
        ]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    evaluate_task,
                    str(path),
                    args.output_root,
                    args.energy_mode,
                    args.weights,
                    args.vote_modes,
                    args.topks,
                    args.gap_thresholds,
                )
                for path in files
            ]
            for future in as_completed(futures):
                results.append(future.result())

    totals: dict[str, dict[str, float]] = {}
    missing = []
    task_count = 0
    for result in sorted(results, key=lambda item: item["task"]):
        if result["missing"]:
            missing.append(result["task"])
            continue
        task_count += 1
        for name, metric in result["metrics"].items():
            totals.setdefault(name, init_metric())
            merge_metric(totals[name], metric)

    finalized = {
        name: finalize_metric(metric, task_count)
        for name, metric in totals.items()
    }
    best_name, best_metric = max(
        finalized.items(),
        key=lambda item: (item[1]["pass_at_2"], item[1]["pass_at_1"]),
    )
    leaderboard = [
        {"strategy": name, **metric}
        for name, metric in sorted(
            finalized.items(),
            key=lambda item: (item[1]["pass_at_2"], item[1]["pass_at_1"]),
            reverse=True,
        )[: args.leaderboard_limit]
    ]
    return {
        "data_root": str(args.data_root),
        "split": args.split,
        "output_root": args.output_root,
        "energy_mode": args.energy_mode,
        "tasks_evaluated": task_count,
        "tasks_missing": len(missing),
        "missing_preview": missing[:20],
        "best": {"strategy": best_name, **best_metric},
        "leaderboard": leaderboard,
        "metrics": finalized,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("raw_data/ARC-AGI"))
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    parser.add_argument(
        "--energy-mode",
        choices=["relation", "relation_symbolic"],
        default="relation",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=[0, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 7.5, 10],
    )
    parser.add_argument("--vote-modes", nargs="+", default=["log", "sqrt", "linear"])
    parser.add_argument("--topks", type=int, nargs="+", default=[2, 3, 5, 10, 20])
    parser.add_argument("--gap-thresholds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 8, 13])
    parser.add_argument("--leaderboard-limit", type=int, default=30)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    summary = evaluate(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
