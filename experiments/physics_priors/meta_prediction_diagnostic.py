#!/usr/bin/env python3
"""Evaluate simple selectors over per-view TTT prediction metadata."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from physics_reranker import serialize
from rerank_varc_predictions import get_majority_vote, load_json


def load_meta(root: Path, task_name: str) -> dict[str, list[dict[str, Any]]] | None:
    path = root / f"{task_name}_prediction_meta.json"
    if not path.exists():
        return None
    return {str(key): value for key, value in json.loads(path.read_text()).items()}


def load_predictions(root: Path, task_name: str) -> dict[str, list] | None:
    path = root / f"{task_name}_predictions.json"
    if not path.exists():
        return None
    return {str(key): value for key, value in json.loads(path.read_text()).items()}


def aggregate_candidates(predictions: list, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for idx, pred in enumerate(predictions):
        key = serialize(pred)
        record = records[idx] if idx < len(records) else {}
        item = by_key.setdefault(
            key,
            {
                "prediction": pred,
                "votes": 0,
                "conf_sum": 0.0,
                "margin_sum": 0.0,
                "entropy_sum": 0.0,
                "min_conf": 1.0,
                "attempts": set(),
                "augmenters": set(),
                "color_indices": set(),
                "scales": set(),
                "border_ok": 0,
            },
        )
        item["votes"] += 1
        item["conf_sum"] += float(record.get("confidence_mean", 0.0))
        item["margin_sum"] += float(record.get("margin_mean", 0.0))
        item["entropy_sum"] += float(record.get("entropy_mean", 0.0))
        item["min_conf"] = min(item["min_conf"], float(record.get("confidence_min", 0.0)))
        item["attempts"].add(record.get("attempt_index"))
        item["augmenters"].add(record.get("augmentation_name") or "original")
        item["color_indices"].add(record.get("color_permutation_index"))
        item["scales"].add(record.get("scale_factor"))
        item["border_ok"] += int(bool(record.get("border_found_x")) and bool(record.get("border_found_y")))

    out = []
    for item in by_key.values():
        votes = max(item["votes"], 1)
        out.append(
            {
                "prediction": item["prediction"],
                "votes": item["votes"],
                "confidence_mean": item["conf_sum"] / votes,
                "margin_mean": item["margin_sum"] / votes,
                "entropy_mean": item["entropy_sum"] / votes,
                "confidence_min": item["min_conf"],
                "attempt_support": len(item["attempts"]),
                "augmenter_support": len(item["augmenters"]),
                "color_support": len(item["color_indices"]),
                "scale_support": len(item["scales"]),
                "border_ok_fraction": item["border_ok"] / votes,
            }
        )
    return out


def init_metric() -> dict[str, float]:
    return {"pass_at_1": 0.0, "pass_at_2": 0.0, "oracle": 0.0, "examples": 0.0}


def add_metric(metric: dict[str, float], order: list[dict[str, Any]], truth: list[list[int]], weight: float) -> None:
    pass1 = bool(order and order[0]["prediction"] == truth)
    pass2 = pass1 or bool(len(order) > 1 and order[1]["prediction"] == truth)
    oracle = any(entry["prediction"] == truth for entry in order)
    metric["pass_at_1"] += weight if pass1 else 0.0
    metric["pass_at_2"] += weight if pass2 else 0.0
    metric["oracle"] += weight if oracle else 0.0
    metric["examples"] += 1.0


def finalize(metric: dict[str, float], task_count: int) -> dict[str, float]:
    denom = max(float(task_count), 1.0)
    return {
        "pass_at_1": metric["pass_at_1"] / denom,
        "pass_at_2": metric["pass_at_2"] / denom,
        "oracle": metric["oracle"] / denom,
        "examples": metric["examples"],
    }


def order_candidates(candidates: list[dict[str, Any]], strategy: str) -> list[dict[str, Any]]:
    if strategy == "majority":
        key = lambda item: (item["votes"], item["confidence_mean"], item["margin_mean"])
    elif strategy == "confidence":
        key = lambda item: (item["confidence_mean"], item["votes"], item["margin_mean"])
    elif strategy == "vote_confidence":
        key = lambda item: (math.log1p(item["votes"]) + item["confidence_mean"], item["votes"])
    elif strategy == "vote_margin":
        key = lambda item: (math.log1p(item["votes"]) + 2.0 * item["margin_mean"], item["votes"])
    elif strategy == "support":
        key = lambda item: (
            item["attempt_support"],
            item["augmenter_support"],
            item["color_support"],
            item["votes"],
        )
    elif strategy == "stability":
        key = lambda item: (
            math.log1p(item["votes"])
            + 0.5 * item["augmenter_support"]
            + 0.25 * item["color_support"]
            + item["confidence_mean"]
            + item["margin_mean"]
            - 0.2 * item["entropy_mean"],
            item["votes"],
        )
    else:
        raise ValueError(strategy)
    return sorted(candidates, key=key, reverse=True)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    task_names = [item for item in args.tasks_file.read_text().split() if item]
    strategies = ["majority", "confidence", "vote_confidence", "vote_margin", "support", "stability"]
    metrics = {strategy: init_metric() for strategy in strategies}
    previews = []
    evaluated_tasks = 0
    missing = []
    for task_name in task_names:
        predictions = load_predictions(args.prediction_root, task_name)
        metadata = load_meta(args.prediction_root, task_name)
        task_path = args.data_root / "data" / args.split / f"{task_name}.json"
        if predictions is None or metadata is None or not task_path.exists():
            missing.append(task_name)
            continue
        evaluated_tasks += 1
        task = load_json(task_path)
        test_examples = task.get("test", [])
        weight = 1.0 / max(len(test_examples), 1)
        for idx, test_ex in enumerate(test_examples):
            if "output" not in test_ex:
                continue
            key = str(idx)
            candidates = aggregate_candidates(predictions.get(key, []), metadata.get(key, []))
            truth = test_ex["output"]
            for strategy in strategies:
                ordered = order_candidates(candidates, strategy)
                add_metric(metrics[strategy], ordered, truth, weight)
            if len(previews) < args.preview:
                majority = order_candidates(candidates, "majority")
                stability = order_candidates(candidates, "stability")
                previews.append(
                    {
                        "task": task_name,
                        "index": idx,
                        "majority_top2": [
                            {
                                "votes": item["votes"],
                                "confidence": round(item["confidence_mean"], 4),
                                "matches": item["prediction"] == truth,
                            }
                            for item in majority[:2]
                        ],
                        "stability_top2": [
                            {
                                "votes": item["votes"],
                                "confidence": round(item["confidence_mean"], 4),
                                "support": item["augmenter_support"],
                                "matches": item["prediction"] == truth,
                            }
                            for item in stability[:2]
                        ],
                    }
                )
    return {
        "prediction_root": str(args.prediction_root),
        "tasks_evaluated": evaluated_tasks,
        "tasks_missing": len(missing),
        "missing_preview": missing[:20],
        "metrics": {name: finalize(metric, evaluated_tasks) for name, metric in metrics.items()},
        "preview": previews,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("raw_data/ARC-AGI"))
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--tasks-file", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--preview", type=int, default=12)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    summary = evaluate(args)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")


if __name__ == "__main__":
    main()
