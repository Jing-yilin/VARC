"""Run VARC test-time training for a deterministic subset of ARC tasks."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time


def _read_tasks(args: argparse.Namespace) -> list[str]:
    tasks: list[str] = []
    if args.tasks:
        tasks.extend(args.tasks)
    if args.tasks_file is not None:
        tasks.extend(
            line.strip()
            for line in args.tasks_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    seen: set[str] = set()
    unique_tasks = []
    for task in tasks:
        if task not in seen:
            unique_tasks.append(task)
            seen.add(task)
    if not unique_tasks:
        raise SystemExit("No tasks provided. Use --tasks or --tasks-file.")
    return unique_tasks


def _prediction_paths(
    output_name: str,
    task: str,
    ttt_num_each: int,
    include_metadata: bool,
) -> list[Path]:
    paths: list[Path] = []
    for idx in range(ttt_num_each):
        output_dir = Path("outputs") / f"{output_name}_attempt_{idx}"
        paths.append(output_dir / f"{task}_predictions.json")
        if include_metadata:
            paths.append(output_dir / f"{task}_prediction_meta.json")
    return paths


def _build_command(args: argparse.Namespace, task: str) -> list[str]:
    command = [
        sys.executable,
        "test_time_train_ARC.py",
        "--epochs",
        str(args.epochs),
        "--depth",
        "10",
        "--batch-size",
        str(args.batch_size),
        "--image-size",
        str(args.image_size),
        "--patch-size",
        "2",
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        "0",
        "--embed-dim",
        "512",
        "--num-heads",
        "8",
        "--num-colors",
        "12",
        "--resume-checkpoint",
        str(args.checkpoint),
        "--lr-scheduler",
        "cosine",
        "--train-split",
        f"{args.augmented_split}/{task}",
        "--data-root",
        str(args.data_root),
        "--eval-split",
        f"{args.augmented_split}/{task}",
        "--resume-skip-task-token",
        "--architecture",
        "vit",
        "--eval-save-name",
        args.output_name,
        "--num-attempts",
        str(args.num_attempts),
        "--ttt-num-each",
        str(args.ttt_num_each),
    ]
    if not args.compile:
        command.append("--no-compile")
    if args.save_prediction_metadata:
        command.append("--save-prediction-metadata")
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--tasks-file", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("raw_data/ARC-AGI"))
    parser.add_argument("--augmented-split", default="eval_color_permute_ttt_9")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("saves/offline_train_ViT/checkpoint_best.pt"),
    )
    parser.add_argument("--output-name", default="physics_ttt_arc1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-attempts", type=int, default=10)
    parser.add_argument("--ttt-num-each", type=int, default=1)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--save-prediction-metadata", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    tasks = _read_tasks(args)
    print(
        f"running TTT for {len(tasks)} tasks: epochs={args.epochs} "
        f"num_attempts={args.num_attempts} ttt_num_each={args.ttt_num_each}",
        flush=True,
    )
    global_start = time.time()
    for task_idx, task in enumerate(tasks, 1):
        expected_outputs = _prediction_paths(
            args.output_name,
            task,
            args.ttt_num_each,
            include_metadata=args.save_prediction_metadata,
        )
        if not args.no_skip_existing and all(path.exists() for path in expected_outputs):
            print(f"[{task_idx}/{len(tasks)}] skip existing {task}", flush=True)
            continue

        command = _build_command(args, task)
        print(f"[{task_idx}/{len(tasks)}] start {task}: {' '.join(command)}", flush=True)
        task_start = time.time()
        subprocess.run(command, check=True)
        print(
            f"[{task_idx}/{len(tasks)}] done {task} elapsed={time.time() - task_start:.1f}s",
            flush=True,
        )
    print(f"all done elapsed={time.time() - global_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
