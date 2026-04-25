"""Evaluate pass@2 strategies that preserve independent mechanism hypotheses."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from physics_reranker import serialize
from rerank_varc_predictions import (
    get_majority_vote,
    load_json,
    load_prediction_roots,
    normalize_index,
    pass_metrics_for_order,
    score_entries,
    task_files,
)


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


def order_by_energy(entries: list[dict], demos: list[dict], test_input: list[list[int]]) -> list[dict]:
    scored = score_entries(entries, demos, test_input, energy_mode="relation")
    return sorted(
        scored,
        key=lambda item: (item["energy"], -float(item.get("vote_signal", item["votes"])), -item["votes"]),
    )


def _predictions_for_index(predictions_by_index: dict[str, list] | None, idx: str) -> list:
    if predictions_by_index is None:
        return []
    return predictions_by_index.get(idx, [])


def evaluate_task(
    task_path_text: str,
    official_roots: str,
    mechanism_roots: str,
    ratio_thresholds: list[float],
) -> dict[str, Any]:
    task_path = Path(task_path_text)
    task_name = task_path.stem
    official_predictions = load_prediction_roots(official_roots, task_name)
    mechanism_predictions = load_prediction_roots(mechanism_roots, task_name)
    if official_predictions is None and mechanism_predictions is None:
        return {"task": task_name, "missing": True}

    task = load_json(task_path)
    demos = task.get("train", [])
    test_examples = task.get("test", [])
    example_weight = 1.0 / max(len(test_examples), 1)
    metrics: dict[str, dict[str, float]] = {}
    previews = []

    for idx, test_ex in enumerate(test_examples):
        key = normalize_index(idx)
        if "output" not in test_ex:
            continue

        official_order = get_majority_vote(_predictions_for_index(official_predictions, key))
        mechanism_order = get_majority_vote(_predictions_for_index(mechanism_predictions, key))
        merged_order = get_majority_vote(
            _predictions_for_index(official_predictions, key)
            + _predictions_for_index(mechanism_predictions, key)
        )
        mechanism_energy_order = order_by_energy(mechanism_order, demos, test_ex["input"])

        orders = {
            "official_majority": official_order,
            "mechanism_majority": mechanism_order,
            "merged_majority": merged_order,
            "official_then_mechanism_top1": unique_concat(
                official_order[:1],
                mechanism_order[:1],
                official_order,
                mechanism_order,
            ),
            "official_then_mechanism_energy": unique_concat(
                official_order[:1],
                mechanism_energy_order[:1],
                official_order,
                mechanism_energy_order,
            ),
            "mechanism_then_official_top1": unique_concat(
                mechanism_order[:1],
                official_order[:1],
                mechanism_order,
                official_order,
            ),
        }
        if len(official_order) >= 2:
            top_votes = max(official_order[0]["votes"], 1)
            second_votes = max(official_order[1]["votes"], 1)
            ratio = top_votes / second_votes
            for threshold in ratio_thresholds:
                second_source = mechanism_order[:1] if ratio <= threshold else official_order[1:2]
                orders[f"uncertain_official_ttt_if_ratio<={threshold:g}"] = unique_concat(
                    official_order[:1],
                    second_source,
                    official_order,
                    mechanism_order,
                )

        ground_truth = test_ex["output"]
        for name, ordered in orders.items():
            metrics.setdefault(name, init_metric())
            add_metric(metrics[name], ordered, ground_truth, example_weight)

        if len(previews) < 20:
            previews.append(
                {
                    "task": task_name,
                    "official_top2": [
                        {"votes": item["votes"], "matches": item["prediction"] == ground_truth}
                        for item in official_order[:2]
                    ],
                    "mechanism_top2": [
                        {"votes": item["votes"], "matches": item["prediction"] == ground_truth}
                        for item in mechanism_order[:2]
                    ],
                }
            )

    return {"task": task_name, "missing": False, "metrics": metrics, "preview": previews}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    totals: dict[str, dict[str, float]] = {}
    missing = []
    previews = []
    task_count = 0
    for task_path in task_files(args.data_root, args.split, args.limit):
        result = evaluate_task(
            str(task_path),
            args.official_roots,
            args.mechanism_roots,
            args.ratio_thresholds,
        )
        if result["missing"]:
            missing.append(result["task"])
            continue
        task_count += 1
        for name, metric in result["metrics"].items():
            totals.setdefault(name, init_metric())
            merge_metric(totals[name], metric)
        if len(previews) < args.preview:
            previews.extend(result["preview"][: args.preview - len(previews)])

    finalized = {name: finalize_metric(metric, task_count) for name, metric in totals.items()}
    best_name, best_metric = max(
        finalized.items(),
        key=lambda item: (item[1]["pass_at_2"], item[1]["pass_at_1"]),
    )
    return {
        "data_root": str(args.data_root),
        "split": args.split,
        "tasks_evaluated": task_count,
        "tasks_missing": len(missing),
        "missing_preview": missing[:20],
        "best": {"strategy": best_name, **best_metric},
        "metrics": finalized,
        "preview": previews,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("raw_data/ARC-AGI"))
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--official-roots", required=True)
    parser.add_argument("--mechanism-roots", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preview", type=int, default=12)
    parser.add_argument(
        "--ratio-thresholds",
        type=float,
        nargs="+",
        default=[1.05, 1.1, 1.15, 1.25, 1.5, 2.0],
        help="Use TTT as the second answer only when official top1/top2 vote ratio is below a threshold.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    summary = evaluate(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
