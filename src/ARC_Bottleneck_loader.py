"""Task-level data loader for ARCBottleneck.

Unlike VARC's per-example loader, this yields (demo_input, demo_output,
test_input, test_output) from the SAME task. The demo pair teaches the rule;
the test pair verifies the model can apply it.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

IGNORE_INDEX = 10
PAD_INDEX = 11
NUM_COLORS = 12
MAX_GRID = 30


def pad_grid(grid: List[List[int]], size: int) -> torch.Tensor:
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    tensor = torch.full((size, size), PAD_INDEX, dtype=torch.long)
    vals = torch.tensor(grid, dtype=torch.long)
    tensor[:h, :w] = vals
    return tensor


def scale_grid(grid: List[List[int]], factor: int) -> List[List[int]]:
    arr = np.array(grid, dtype=np.int64)
    arr = np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)
    return arr.tolist()


class ARCTaskDataset(Dataset):
    """Each __getitem__ returns one (demo, test) pair from a random task.

    During training: both demo and test are sampled from the task's train set.
    During eval: demo is from train set, test is the actual test example.
    """

    def __init__(
        self,
        root: Path,
        split: str = "training",
        mode: str = "train",
        grid_size: int = 64,
        scale_factor: int = 2,
        seed: int = 42,
    ):
        self.root = Path(root)
        self.grid_size = grid_size
        self.scale_factor = scale_factor
        self.mode = mode
        self.rng = random.Random(seed)

        split_dir = self.root / "data" / split
        self.tasks: List[Dict] = []
        self.task_names: List[str] = []

        for path in sorted(split_dir.glob("*.json")):
            with path.open() as f:
                task = json.load(f)
            train_examples = task.get("train", [])
            if len(train_examples) < 2 and mode == "train":
                continue
            valid = True
            for ex in train_examples + task.get("test", []):
                h = len(ex["input"])
                w = len(ex["input"][0]) if h > 0 else 0
                if "output" in ex:
                    h = max(h, len(ex["output"]))
                    w = max(w, len(ex["output"][0]) if len(ex["output"]) > 0 else 0)
                if h * scale_factor > grid_size or w * scale_factor > grid_size:
                    valid = False
                    break
            if valid:
                self.tasks.append(task)
                self.task_names.append(path.stem)

        if not self.tasks:
            raise RuntimeError(f"No valid tasks found in {split_dir}")
        print(f"Loaded {len(self.tasks)} tasks ({mode} mode, grid_size={grid_size})")

    def __len__(self):
        if self.mode == "train":
            return len(self.tasks) * 8
        return sum(len(t.get("test", [])) for t in self.tasks)

    def _prepare(self, grid: List[List[int]]) -> torch.Tensor:
        scaled = scale_grid(grid, self.scale_factor)
        return pad_grid(scaled, self.grid_size)

    def __getitem__(self, idx: int):
        if self.mode == "train":
            task_idx = idx % len(self.tasks)
            task = self.tasks[task_idx]
            examples = task["train"]
            i, j = self.rng.sample(range(len(examples)), 2)
            demo = examples[i]
            test = examples[j]
        else:
            offset = 0
            for task_idx, task in enumerate(self.tasks):
                tests = task.get("test", [])
                if idx < offset + len(tests):
                    test = tests[idx - offset]
                    demo_idx = self.rng.randrange(len(task["train"]))
                    demo = task["train"][demo_idx]
                    break
                offset += len(tests)
            else:
                task_idx = 0
                task = self.tasks[0]
                demo = task["train"][0]
                test = task.get("test", task["train"])[0]

        return {
            "demo_input": self._prepare(demo["input"]),
            "demo_output": self._prepare(demo["output"]),
            "test_input": self._prepare(test["input"]),
            "test_output": self._prepare(test["output"]),
            "task_idx": task_idx,
            "task_name": self.task_names[task_idx],
        }


def collate_bottleneck(batch: List[Dict]) -> Dict:
    return {
        "demo_input": torch.stack([b["demo_input"] for b in batch]),
        "demo_output": torch.stack([b["demo_output"] for b in batch]),
        "test_input": torch.stack([b["test_input"] for b in batch]),
        "test_output": torch.stack([b["test_output"] for b in batch]),
        "task_idx": torch.tensor([b["task_idx"] for b in batch], dtype=torch.long),
        "task_names": [b["task_name"] for b in batch],
    }


def build_bottleneck_loaders(
    data_root: str,
    grid_size: int = 64,
    scale_factor: int = 2,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    root = Path(data_root)
    train_ds = ARCTaskDataset(root, "training", "train", grid_size, scale_factor, seed)
    eval_ds = ARCTaskDataset(root, "training", "eval", grid_size, scale_factor, seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_bottleneck,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_bottleneck,
    )
    return train_loader, eval_loader
