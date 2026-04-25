"""Shard VARC test-time training across independent GPUs.

Each ARC task is independent during TTT, so the fastest scale-up path is one
`run_ttt_subset.py` process per GPU with disjoint task files. Run this script
inside tmux on a multi-GPU host.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import time


@dataclass(frozen=True)
class Worker:
    gpu: str
    tasks_file: Path
    log_file: Path
    process: subprocess.Popen[None]


def _read_tasks(path: Path) -> list[str]:
    tasks: list[str] = []
    seen: set[str] = set()
    for line in path.read_text().splitlines():
        task = line.strip()
        if not task or task.startswith("#") or task in seen:
            continue
        tasks.append(task)
        seen.add(task)
    if not tasks:
        raise SystemExit(f"No tasks found in {path}")
    return tasks


def _split_tasks(tasks: list[str], num_shards: int) -> list[list[str]]:
    shards = [[] for _ in range(num_shards)]
    for idx, task in enumerate(tasks):
        shards[idx % num_shards].append(task)
    return shards


def _write_shard_files(run_dir: Path, gpus: list[str], shards: list[list[str]]) -> list[Path]:
    task_files: list[Path] = []
    for worker_idx, (gpu, shard) in enumerate(zip(gpus, shards, strict=True)):
        safe_gpu = gpu.replace(",", "_").replace(":", "_")
        task_file = run_dir / f"shard_{worker_idx:02d}_gpu_{safe_gpu}.txt"
        task_file.write_text("\n".join(shard) + ("\n" if shard else ""))
        task_files.append(task_file)
    return task_files


def _worker_command(args: argparse.Namespace, task_file: Path) -> list[str]:
    command = [
        sys.executable,
        "experiments/physics_priors/run_ttt_subset.py",
        "--tasks-file",
        str(task_file),
        "--data-root",
        str(args.data_root),
        "--augmented-split",
        args.augmented_split,
        "--checkpoint",
        str(args.checkpoint),
        "--output-name",
        args.output_name,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--image-size",
        str(args.image_size),
        "--learning-rate",
        str(args.learning_rate),
        "--num-attempts",
        str(args.num_attempts),
        "--ttt-num-each",
        str(args.ttt_num_each),
    ]
    if args.compile:
        command.append("--compile")
    if args.save_prediction_metadata:
        command.append("--save-prediction-metadata")
    if args.no_skip_existing:
        command.append("--no-skip-existing")
    return command


def _launch_workers(
    args: argparse.Namespace,
    gpus: list[str],
    task_files: list[Path],
    run_dir: Path,
) -> list[Worker]:
    workers: list[Worker] = []
    for worker_idx, (gpu, task_file) in enumerate(zip(gpus, task_files, strict=True)):
        if not task_file.read_text().strip():
            print(f"[worker {worker_idx}] gpu={gpu} no tasks; skip", flush=True)
            continue
        log_file = run_dir / f"worker_{worker_idx:02d}_gpu_{gpu.replace(',', '_')}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        command = _worker_command(args, task_file)
        print(
            f"[worker {worker_idx}] gpu={gpu} tasks={task_file} log={log_file}",
            flush=True,
        )
        log_handle = log_file.open("w")
        process = subprocess.Popen(command, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
        log_handle.close()
        workers.append(Worker(gpu=gpu, tasks_file=task_file, log_file=log_file, process=process))
    return workers


def _count_outputs(output_name: str, tasks: list[str], ttt_num_each: int, metadata: bool) -> int:
    count = 0
    for task in tasks:
        for idx in range(ttt_num_each):
            output_dir = Path("outputs") / f"{output_name}_attempt_{idx}"
            if (output_dir / f"{task}_predictions.json").exists():
                count += 1
            if metadata and (output_dir / f"{task}_prediction_meta.json").exists():
                count += 1
    return count


def _expected_output_count(tasks: list[str], ttt_num_each: int, metadata: bool) -> int:
    multiplier = 2 if metadata else 1
    return len(tasks) * ttt_num_each * multiplier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-file", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", required=True)
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
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    tasks = _read_tasks(args.tasks_file)
    gpus = [gpu.strip() for gpu in args.gpus if gpu.strip()]
    if not gpus:
        raise SystemExit("No GPUs provided. Use --gpus 0 1 ...")

    run_dir = args.run_dir
    if run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(".tmp/physics_priors/parallel_ttt") / f"{stamp}_{args.output_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    shards = _split_tasks(tasks, len(gpus))
    task_files = _write_shard_files(run_dir, gpus, shards)
    expected_outputs = _expected_output_count(
        tasks,
        args.ttt_num_each,
        args.save_prediction_metadata,
    )

    print(
        f"parallel TTT: tasks={len(tasks)} gpus={','.join(gpus)} "
        f"run_dir={run_dir} expected_outputs={expected_outputs}",
        flush=True,
    )
    workers = _launch_workers(args, gpus, task_files, run_dir)
    if not workers:
        raise SystemExit("No non-empty worker shards to run.")

    start = time.time()
    while True:
        running = [worker for worker in workers if worker.process.poll() is None]
        completed_outputs = _count_outputs(
            args.output_name,
            tasks,
            args.ttt_num_each,
            args.save_prediction_metadata,
        )
        elapsed = time.time() - start
        print(
            f"status elapsed={elapsed:.0f}s running={len(running)}/{len(workers)} "
            f"outputs={completed_outputs}/{expected_outputs}",
            flush=True,
        )
        if not running:
            break
        time.sleep(max(args.poll_seconds, 5))

    failed: list[Worker] = []
    for worker in workers:
        return_code = worker.process.returncode
        if return_code != 0:
            failed.append(worker)
            print(
                f"failed gpu={worker.gpu} rc={return_code} log={worker.log_file}",
                flush=True,
            )
    if failed:
        raise SystemExit(1)
    print(f"all workers completed in {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
