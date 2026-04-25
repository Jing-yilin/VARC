#!/usr/bin/env python3
"""Evaluate prediction roots on an explicit ARC task subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from physics_reranker import serialize
from rerank_varc_predictions import entries_for_index, load_prediction_sources


def read_tasks(path: Path) -> list[str]:
    return [item for item in path.read_text().split() if item]


def unique_concat(*orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
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


def add_metric(
    metric: dict[str, float],
    order: list[dict[str, Any]],
    ground_truth: list[list[int]],
    weight: float,
) -> tuple[bool, bool, bool]:
    pass1 = bool(order and order[0]["prediction"] == ground_truth)
    pass2 = pass1 or bool(len(order) > 1 and order[1]["prediction"] == ground_truth)
    oracle = any(entry["prediction"] == ground_truth for entry in order)
    metric["pass_at_1"] += weight if pass1 else 0.0
    metric["pass_at_2"] += weight if pass2 else 0.0
    metric["oracle"] += weight if oracle else 0.0
    metric["examples"] += 1.0
    return pass1, pass2, oracle


def finalize(metric: dict[str, float], task_count: int) -> dict[str, float]:
    denom = max(float(task_count), 1.0)
    return {
        "pass_at_1": metric["pass_at_1"] / denom,
        "pass_at_2": metric["pass_at_2"] / denom,
        "oracle": metric["oracle"] / denom,
        "examples": metric["examples"],
    }


def status_code(pass1: bool, pass2: bool, oracle: bool) -> str:
    if pass1:
        return "1"
    if pass2:
        return "2"
    if oracle:
        return "o"
    return "-"


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    tasks = read_tasks(args.tasks_file)
    root_specs = dict(item.split("=", 1) for item in args.roots)
    metrics = {name: init_metric() for name in root_specs}
    if args.official and args.mechanism:
        metrics.update(
            {
                "official_then_mechanism_top1": init_metric(),
                "merged_official_mechanism": init_metric(),
            }
        )

    rows: list[dict[str, Any]] = []
    for task_name in tasks:
        task_path = args.data_root / "data" / args.split / f"{task_name}.json"
        task = json.loads(task_path.read_text())
        sources = {
            name: load_prediction_sources(root_text, task_name)
            for name, root_text in root_specs.items()
        }
        for idx, test_example in enumerate(task.get("test", [])):
            if "output" not in test_example:
                continue
            key = str(idx)
            ground_truth = test_example["output"]
            weight = 1.0 / max(len(task.get("test", [])), 1)
            orders = {
                name: entries_for_index(source, key, "raw") if source else []
                for name, source in sources.items()
            }
            row = {"task": task_name, "index": idx, "roots": {}}
            for name, order in orders.items():
                pass1, pass2, oracle = add_metric(metrics[name], order, ground_truth, weight)
                row["roots"][name] = {
                    "status": status_code(pass1, pass2, oracle),
                    "votes": [entry["votes"] for entry in order[:2]],
                }

            if args.official and args.mechanism:
                official_order = orders.get(args.official, [])
                mechanism_order = orders.get(args.mechanism, [])
                combos = {
                    "official_then_mechanism_top1": unique_concat(
                        official_order[:1],
                        mechanism_order[:1],
                        official_order,
                        mechanism_order,
                    ),
                    "merged_official_mechanism": entries_for_index(
                        [
                            *(sources.get(args.official) or []),
                            *(sources.get(args.mechanism) or []),
                        ],
                        key,
                        "raw",
                    ),
                }
                for name, order in combos.items():
                    pass1, pass2, oracle = add_metric(metrics[name], order, ground_truth, weight)
                    row["roots"][name] = {
                        "status": status_code(pass1, pass2, oracle),
                        "votes": [entry["votes"] for entry in order[:2]],
                    }
            rows.append(row)

    return {
        "data_root": str(args.data_root),
        "split": args.split,
        "tasks_file": str(args.tasks_file),
        "tasks_evaluated": len(tasks),
        "metrics": {name: finalize(metric, len(tasks)) for name, metric in metrics.items()},
        "rows": rows[: args.preview],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("raw_data/ARC-AGI"))
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--tasks-file", type=Path, required=True)
    parser.add_argument("--roots", nargs="+", required=True, help="Name=root[,root...] entries.")
    parser.add_argument("--official", default=None, help="Root name to use as official combo source.")
    parser.add_argument("--mechanism", default=None, help="Root name to use as mechanism combo source.")
    parser.add_argument("--preview", type=int, default=20)
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
