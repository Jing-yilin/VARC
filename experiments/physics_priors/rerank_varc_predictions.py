#!/usr/bin/env python3
"""Rerank VARC multi-view predictions with demo-conditioned energy."""

from __future__ import annotations

import argparse
import os
import json
import math
from pathlib import Path
from typing import Any
from concurrent.futures import ProcessPoolExecutor, as_completed

from physics_reranker import (
    infer_transform_color_rules,
    relation_energy,
    serialize,
    symbolic_rule_energy_for_rules,
)


def load_json(path: Path) -> Any:
    with path.open("r") as handle:
        return json.load(handle)


def task_files(data_root: Path, split: str, limit: int | None) -> list[Path]:
    files = sorted((data_root / "data" / split).glob("*.json"))
    return files[:limit] if limit is not None else files


def normalize_index(index: str | int) -> str:
    return str(index)


def get_majority_vote(predictions: list[list[list[int]]]) -> list[dict]:
    counts: dict[str, int] = {}
    grids: dict[str, list[list[int]]] = {}
    for pred in predictions:
        key = serialize(pred)
        grids[key] = pred
        counts[key] = counts.get(key, 0) + 1
    return [
        {"prediction": grids[key], "votes": votes, "vote_signal": float(votes)}
        for key, votes in sorted(counts.items(), key=lambda item: item[1], reverse=True)
    ]


def get_source_normalized_vote(
    sources: list[dict[str, Any]],
    prediction_index: str,
) -> list[dict]:
    raw_counts: dict[str, int] = {}
    source_scores: dict[str, float] = {}
    source_hits: dict[str, int] = {}
    grids: dict[str, list[list[int]]] = {}

    for source in sources:
        predictions = source["predictions"].get(prediction_index, [])
        if not predictions:
            continue
        local_counts: dict[str, int] = {}
        local_grids: dict[str, list[list[int]]] = {}
        for pred in predictions:
            key = serialize(pred)
            local_grids[key] = pred
            local_counts[key] = local_counts.get(key, 0) + 1
        denom = max(sum(local_counts.values()), 1)
        for key, count in local_counts.items():
            grids[key] = local_grids[key]
            raw_counts[key] = raw_counts.get(key, 0) + count
            source_scores[key] = source_scores.get(key, 0.0) + count / denom
            source_hits[key] = source_hits.get(key, 0) + 1

    return [
        {
            "prediction": grids[key],
            "votes": raw_counts[key],
            "vote_signal": source_scores[key],
            "source_hits": source_hits[key],
        }
        for key in sorted(
            grids,
            key=lambda item: (
                source_scores[item],
                source_hits[item],
                raw_counts[item],
            ),
            reverse=True,
        )
    ]


def score_entries(
    entries: list[dict],
    demos: list[dict],
    test_input: list[list[int]],
    energy_mode: str,
) -> list[dict]:
    scored = []
    rules = infer_transform_color_rules(demos) if energy_mode == "relation_symbolic" else []
    for entry in entries:
        energy = relation_energy(demos, test_input, entry["prediction"])
        if energy_mode == "relation_symbolic":
            energy += symbolic_rule_energy_for_rules(rules, test_input, entry["prediction"])
        vote_signal = float(entry.get("vote_signal", entry["votes"]))
        scored.append({**entry, "energy": energy, "vote_log": math.log1p(vote_signal)})
    return scored


def order_scored_entries(scored: list[dict], vote_weight: float) -> list[dict]:
    return sorted(
        (
            {**entry, "score": entry["energy"] - vote_weight * entry["vote_log"]}
            for entry in scored
        ),
        key=lambda item: (item["score"], -float(item.get("vote_signal", item["votes"])), -item["votes"]),
    )


def pass_metrics_for_order(
    ordered: list[dict],
    ground_truth: list[list[int]],
) -> tuple[bool, bool, int | None]:
    pass1 = len(ordered) > 0 and ordered[0]["prediction"] == ground_truth
    pass2 = pass1 or (len(ordered) > 1 and ordered[1]["prediction"] == ground_truth)
    truth_rank = None
    for idx, entry in enumerate(ordered):
        if entry["prediction"] == ground_truth:
            truth_rank = idx + 1
            break
    return pass1, pass2, truth_rank


def init_metric() -> dict[str, float]:
    return {
        "pass_at_1": 0.0,
        "pass_at_2": 0.0,
        "oracle": 0.0,
        "rank_sum": 0.0,
        "rank_count": 0.0,
        "examples": 0.0,
    }


def merge_metric(total: dict[str, float], delta: dict[str, float]) -> None:
    for key, value in delta.items():
        total[key] = total.get(key, 0.0) + value


def finalize_metric(metric: dict[str, float], total_tasks: int) -> dict[str, float]:
    denom = max(float(total_tasks), 1.0)
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


def add_result(
    metric: dict[str, float],
    pass1: bool,
    pass2: bool,
    truth_rank: int | None,
    example_weight: float,
) -> None:
    metric["pass_at_1"] += example_weight if pass1 else 0.0
    metric["pass_at_2"] += example_weight if pass2 else 0.0
    metric["oracle"] += example_weight if truth_rank is not None else 0.0
    metric["examples"] += 1.0
    if truth_rank is not None:
        metric["rank_sum"] += truth_rank
        metric["rank_count"] += 1.0


def load_prediction_roots(output_roots: str, task_name: str) -> dict[str, list] | None:
    merged: dict[str, list] | None = None
    for root_text in output_roots.split(","):
        root = Path(root_text.strip())
        path = root / f"{task_name}_predictions.json"
        if not path.exists():
            continue
        data = load_json(path)
        if merged is None:
            merged = {normalize_index(k): list(v) for k, v in data.items()}
        else:
            for key, values in data.items():
                merged.setdefault(normalize_index(key), []).extend(values)
    return merged


def load_prediction_sources(output_roots: str, task_name: str) -> list[dict[str, Any]] | None:
    sources: list[dict[str, Any]] = []
    for root_text in output_roots.split(","):
        root = Path(root_text.strip())
        path = root / f"{task_name}_predictions.json"
        if not path.exists():
            continue
        data = load_json(path)
        sources.append(
            {
                "root": str(root),
                "predictions": {normalize_index(k): list(v) for k, v in data.items()},
            }
        )
    return sources or None


def entries_for_index(
    sources: list[dict[str, Any]],
    prediction_index: str,
    vote_mode: str,
) -> list[dict]:
    if vote_mode == "source_norm":
        return get_source_normalized_vote(sources, prediction_index)

    predictions: list[list[list[int]]] = []
    for source in sources:
        predictions.extend(source["predictions"].get(prediction_index, []))
    return get_majority_vote(predictions)


def evaluate_task(
    task_path_text: str,
    output_root: str,
    vote_weights: list[float],
    preview_limit: int,
    energy_mode: str,
    vote_mode: str,
) -> dict:
    task_path = Path(task_path_text)
    task_name = task_path.stem
    prediction_sources = load_prediction_sources(output_root, task_name)
    if prediction_sources is None:
        return {"task": task_name, "missing": True}

    task = load_json(task_path)
    demos = task.get("train", [])
    test_examples = task.get("test", [])
    example_weight = 1.0 / max(len(test_examples), 1)

    majority_metric = init_metric()
    energy_metric = init_metric()
    hybrid_metrics = {str(weight): init_metric() for weight in vote_weights}
    examples_preview = []

    for idx, test_ex in enumerate(test_examples):
        key = str(idx)
        if "output" not in test_ex:
            continue
        entries = entries_for_index(prediction_sources, key, vote_mode)
        if not entries:
            continue
        ground_truth = test_ex["output"]
        scored_entries = score_entries(entries, demos, test_ex["input"], energy_mode)

        majority_pass1, majority_pass2, majority_rank = pass_metrics_for_order(entries, ground_truth)
        add_result(majority_metric, majority_pass1, majority_pass2, majority_rank, example_weight)

        energy_order = order_scored_entries(scored_entries, vote_weight=0.0)
        energy_pass1, energy_pass2, energy_rank = pass_metrics_for_order(energy_order, ground_truth)
        add_result(energy_metric, energy_pass1, energy_pass2, energy_rank, example_weight)

        for weight in vote_weights:
            hybrid_order = order_scored_entries(scored_entries, vote_weight=weight)
            pass1, pass2, truth_rank = pass_metrics_for_order(hybrid_order, ground_truth)
            add_result(hybrid_metrics[str(weight)], pass1, pass2, truth_rank, example_weight)

        if len(examples_preview) < preview_limit:
            examples_preview.append(
                {
                    "task": task_name,
                    "test_index": idx,
                    "majority_top2": [
                        {
                            "votes": item["votes"],
                            "vote_signal": round(float(item.get("vote_signal", item["votes"])), 4),
                            "source_hits": item.get("source_hits"),
                            "matches": item["prediction"] == ground_truth,
                        }
                        for item in entries[:2]
                    ],
                    "energy_top2": [
                        {
                            "votes": item["votes"],
                            "vote_signal": round(float(item.get("vote_signal", item["votes"])), 4),
                            "source_hits": item.get("source_hits"),
                            "energy": round(item["energy"], 4),
                            "score": round(item["score"], 4),
                            "matches": item["prediction"] == ground_truth,
                        }
                        for item in energy_order[:2]
                    ],
                }
            )

    return {
        "task": task_name,
        "missing": False,
        "majority": majority_metric,
        "energy": energy_metric,
        "hybrid": hybrid_metrics,
        "examples_preview": examples_preview,
    }


def evaluate(args: argparse.Namespace) -> dict:
    majority_metric = init_metric()
    energy_metric = init_metric()
    hybrid_metrics = {str(weight): init_metric() for weight in args.vote_weights}
    task_count = 0
    missing_tasks = []
    examples_preview = []
    files = task_files(args.data_root, args.split, args.limit)
    workers = max(args.workers, 1)
    if workers == 1:
        results = [
            evaluate_task(
                str(path),
                args.output_root,
                args.vote_weights,
                args.preview,
                args.energy_mode,
                args.vote_mode,
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
                    args.vote_weights,
                    args.preview,
                    args.energy_mode,
                    args.vote_mode,
                )
                for path in files
            ]
            for future in as_completed(futures):
                results.append(future.result())

    for result in sorted(results, key=lambda item: item["task"]):
        if result["missing"]:
            missing_tasks.append(result["task"])
            continue
        task_count += 1
        merge_metric(majority_metric, result["majority"])
        merge_metric(energy_metric, result["energy"])
        for weight, metric in result["hybrid"].items():
            merge_metric(hybrid_metrics[weight], metric)
        if len(examples_preview) < args.preview:
            needed = args.preview - len(examples_preview)
            examples_preview.extend(result["examples_preview"][:needed])

    finalized_hybrids = {
        weight: finalize_metric(metric, task_count)
        for weight, metric in hybrid_metrics.items()
    }
    best_weight, best_metric = max(
        finalized_hybrids.items(),
        key=lambda item: (item[1]["pass_at_2"], item[1]["pass_at_1"]),
    )

    return {
        "data_root": str(args.data_root),
        "split": args.split,
        "output_root": args.output_root,
        "energy_mode": args.energy_mode,
        "vote_mode": args.vote_mode,
        "tasks_evaluated": task_count,
        "tasks_missing": len(missing_tasks),
        "missing_preview": missing_tasks[:20],
        "majority": finalize_metric(majority_metric, task_count),
        "energy_only": finalize_metric(energy_metric, task_count),
        "hybrid": finalized_hybrids,
        "best_hybrid": {"vote_weight": best_weight, **best_metric},
        "examples_preview": examples_preview,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("raw_data/ARC-AGI"))
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preview", type=int, default=12)
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    parser.add_argument(
        "--energy-mode",
        choices=["relation", "relation_symbolic"],
        default="relation",
    )
    parser.add_argument(
        "--vote-mode",
        choices=["raw", "source_norm"],
        default="raw",
        help="Use raw candidate counts or normalize vote mass inside each prediction root.",
    )
    parser.add_argument(
        "--vote-weights",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
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
