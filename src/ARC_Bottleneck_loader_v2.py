"""Task-level data loader v2: supports multi-demo per task.

Each __getitem__ returns N demo pairs + 1 test pair from the same task.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

IGNORE_INDEX = 10
PAD_INDEX = 11
NUM_COLORS = 12


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


class ARCMultiDemoDataset(Dataset):
    """Returns N demo pairs + 1 test pair per task."""

    def __init__(
        self,
        root: Path,
        split: str = "training",
        mode: str = "train",
        grid_size: int = 64,
        scale_factor: int = 2,
        num_demos: int = 2,
        seed: int = 42,
    ):
        self.root = Path(root)
        self.grid_size = grid_size
        self.scale_factor = scale_factor
        self.num_demos = num_demos
        self.mode = mode
        self.rng = random.Random(seed)

        split_dir = self.root / "data" / split
        self.tasks: List[Dict] = []
        self.task_names: List[str] = []

        for path in sorted(split_dir.glob("*.json")):
            with path.open() as f:
                task = json.load(f)
            train_examples = task.get("train", [])
            if len(train_examples) < 2:
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

        print(f"Loaded {len(self.tasks)} tasks ({mode} mode, num_demos={num_demos})")

    def __len__(self):
        if self.mode == "train":
            return len(self.tasks) * 8
        return sum(len(t.get("test", [])) for t in self.tasks)

    def _prepare(self, grid):
        scaled = scale_grid(grid, self.scale_factor)
        return pad_grid(scaled, self.grid_size)

    def __getitem__(self, idx: int):
        if self.mode == "train":
            task_idx = idx % len(self.tasks)
            task = self.tasks[task_idx]
            examples = task["train"]
            n = len(examples)
            # Sample num_demos + 1 indices (with replacement if needed)
            if n > self.num_demos:
                indices = self.rng.sample(range(n), self.num_demos + 1)
                demo_indices = indices[:self.num_demos]
                test_idx = indices[self.num_demos]
            else:
                # Use all but one as demos, last as test
                all_idx = list(range(n))
                self.rng.shuffle(all_idx)
                demo_indices = all_idx[:min(self.num_demos, n-1)]
                test_idx = all_idx[-1]
                # Pad demos if needed
                while len(demo_indices) < self.num_demos:
                    demo_indices.append(self.rng.choice(demo_indices))
            
            demos = [examples[i] for i in demo_indices]
            test = examples[test_idx]
        else:
            offset = 0
            for task_idx, task in enumerate(self.tasks):
                tests = task.get("test", [])
                if idx < offset + len(tests):
                    test = tests[idx - offset]
                    examples = task["train"]
                    n = len(examples)
                    demo_indices = list(range(min(self.num_demos, n)))
                    while len(demo_indices) < self.num_demos:
                        demo_indices.append(self.rng.choice(range(n)))
                    demos = [examples[i] for i in demo_indices]
                    break
                offset += len(tests)
            else:
                task_idx = 0
                task = self.tasks[0]
                demos = [task["train"][0]] * self.num_demos
                test = task.get("test", task["train"])[0]

        return {
            "demo_inputs": torch.stack([self._prepare(d["input"]) for d in demos]),
            "demo_outputs": torch.stack([self._prepare(d["output"]) for d in demos]),
            "test_input": self._prepare(test["input"]),
            "test_output": self._prepare(test["output"]),
            "task_idx": task_idx,
        }


def collate_multi_demo(batch):
    return {
        "demo_input": torch.stack([b["demo_inputs"] for b in batch]),  # (B, N, H, W)
        "demo_output": torch.stack([b["demo_outputs"] for b in batch]),
        "test_input": torch.stack([b["test_input"] for b in batch]),
        "test_output": torch.stack([b["test_output"] for b in batch]),
        "task_idx": torch.tensor([b["task_idx"] for b in batch]),
    }


def build_multi_demo_loaders(
    data_root: str,
    grid_size: int = 64,
    scale_factor: int = 2,
    batch_size: int = 32,
    num_demos: int = 2,
    num_workers: int = 0,
    seed: int = 42,
):
    root = Path(data_root)
    train_ds = ARCMultiDemoDataset(root, "training", "train", grid_size, scale_factor, num_demos, seed)
    eval_ds = ARCMultiDemoDataset(root, "training", "eval", grid_size, scale_factor, num_demos, seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_multi_demo, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_multi_demo,
    )
    return train_loader, eval_loader
