#!/usr/bin/env python3
"""Synthetic test for selective equivariance on Apple Silicon/MPS.

The task is one-demo ARC-like image-to-image translation:
  demo_in -> demo_out reveals a transformation
  query_in -> query_out must apply the same transformation

We compare:
  1. A small learned CNN conditioned on demo_in/demo_out/query_in.
  2. A zero-training selective symmetry gate over explicit transform actions.

The point is not to beat ARC. It is to test whether a structural "choose the
right natural law, then apply it" module gives a large sample-efficiency gain.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TRANSFORMS = ("identity", "rot90", "rot180", "rot270", "flipud", "fliplr")


@dataclass
class Episodes:
    demo_in: torch.Tensor
    demo_out: torch.Tensor
    query_in: torch.Tensor
    query_out: torch.Tensor
    transform_ids: torch.Tensor


def apply_np(grid: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return grid
    if name == "rot90":
        return np.rot90(grid, 1)
    if name == "rot180":
        return np.rot90(grid, 2)
    if name == "rot270":
        return np.rot90(grid, 3)
    if name == "flipud":
        return np.flipud(grid)
    if name == "fliplr":
        return np.fliplr(grid)
    raise ValueError(name)


def apply_torch(batch: torch.Tensor, name: str) -> torch.Tensor:
    if name == "identity":
        return batch
    if name == "rot90":
        return torch.rot90(batch, 1, dims=(-2, -1))
    if name == "rot180":
        return torch.rot90(batch, 2, dims=(-2, -1))
    if name == "rot270":
        return torch.rot90(batch, 3, dims=(-2, -1))
    if name == "flipud":
        return torch.flip(batch, dims=(-2,))
    if name == "fliplr":
        return torch.flip(batch, dims=(-1,))
    raise ValueError(name)


def random_grid(rng: random.Random, size: int, colors: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=np.int64)
    object_count = rng.randint(2, 5)
    for _ in range(object_count):
        color = rng.randint(1, colors - 1)
        h = rng.randint(1, max(1, size // 4))
        w = rng.randint(1, max(1, size // 4))
        y = rng.randint(0, size - h)
        x = rng.randint(0, size - w)
        grid[y : y + h, x : x + w] = color
    # Add sparse anchors to break accidental symmetries.
    for _ in range(rng.randint(1, 4)):
        grid[rng.randrange(size), rng.randrange(size)] = rng.randint(1, colors - 1)
    return grid


def make_episodes(n: int, size: int, colors: int, seed: int, device: torch.device) -> Episodes:
    rng = random.Random(seed)
    demo_in = []
    demo_out = []
    query_in = []
    query_out = []
    transform_ids = []
    for _ in range(n):
        tid = rng.randrange(len(TRANSFORMS))
        transform = TRANSFORMS[tid]
        din = random_grid(rng, size, colors)
        qin = random_grid(rng, size, colors)
        demo_in.append(din)
        demo_out.append(apply_np(din, transform).copy())
        query_in.append(qin)
        query_out.append(apply_np(qin, transform).copy())
        transform_ids.append(tid)

    def tensor(items: list[np.ndarray]) -> torch.Tensor:
        return torch.tensor(np.stack(items), dtype=torch.long, device=device)

    return Episodes(
        demo_in=tensor(demo_in),
        demo_out=tensor(demo_out),
        query_in=tensor(query_in),
        query_out=tensor(query_out),
        transform_ids=torch.tensor(transform_ids, dtype=torch.long, device=device),
    )


def random_grid_torch(n: int, size: int, colors: int, density: float, device: torch.device) -> torch.Tensor:
    values = torch.randint(1, colors, (n, size, size), device=device)
    mask = torch.rand((n, size, size), device=device) < density
    grids = torch.where(mask, values, torch.zeros((), dtype=torch.long, device=device))

    # A few anchors reduce accidental symmetries while keeping generation vectorized.
    anchor_count = 4
    batch_idx = torch.arange(n, device=device).repeat_interleave(anchor_count)
    ys = torch.randint(0, size, (n * anchor_count,), device=device)
    xs = torch.randint(0, size, (n * anchor_count,), device=device)
    colors_t = torch.randint(1, colors, (n * anchor_count,), device=device)
    grids[batch_idx, ys, xs] = colors_t
    return grids


def apply_torch_by_id(grids: torch.Tensor, transform_ids: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(grids)
    for idx, name in enumerate(TRANSFORMS):
        mask = transform_ids == idx
        if mask.any():
            out[mask] = apply_torch(grids[mask], name)
    return out


def make_episodes_torch_random(
    n: int,
    size: int,
    colors: int,
    seed: int,
    density: float,
    device: torch.device,
) -> Episodes:
    if device.type == "cuda":
        with torch.random.fork_rng(devices=[device.index or 0]):
            torch.manual_seed(seed)
            transform_ids = torch.randint(len(TRANSFORMS), (n,), device=device)
            demo_in = random_grid_torch(n, size, colors, density, device)
            query_in = random_grid_torch(n, size, colors, density, device)
    else:
        torch.manual_seed(seed)
        transform_ids = torch.randint(len(TRANSFORMS), (n,), device=device)
        demo_in = random_grid_torch(n, size, colors, density, device)
        query_in = random_grid_torch(n, size, colors, density, device)

    return Episodes(
        demo_in=demo_in,
        demo_out=apply_torch_by_id(demo_in, transform_ids),
        query_in=query_in,
        query_out=apply_torch_by_id(query_in, transform_ids),
        transform_ids=transform_ids,
    )


class DemoConditionedCNN(nn.Module):
    def __init__(self, colors: int, hidden: int = 64) -> None:
        super().__init__()
        self.colors = colors
        self.net = nn.Sequential(
            nn.Conv2d(colors * 3, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, colors, 1),
        )

    def forward(self, demo_in: torch.Tensor, demo_out: torch.Tensor, query_in: torch.Tensor) -> torch.Tensor:
        parts = [
            F.one_hot(demo_in, self.colors),
            F.one_hot(demo_out, self.colors),
            F.one_hot(query_in, self.colors),
        ]
        x = torch.cat(parts, dim=-1).float().permute(0, 3, 1, 2)
        return self.net(x)


@torch.no_grad()
def selective_symmetry_gate(episodes: Episodes) -> tuple[float, float]:
    scores = []
    transformed_queries = []
    for name in TRANSFORMS:
        transformed_demo = apply_torch(episodes.demo_in, name)
        scores.append((transformed_demo == episodes.demo_out).float().mean(dim=(-2, -1)))
        transformed_queries.append(apply_torch(episodes.query_in, name))
    score_tensor = torch.stack(scores, dim=1)
    query_tensor = torch.stack(transformed_queries, dim=1)
    selected = score_tensor.argmax(dim=1)
    batch_idx = torch.arange(episodes.query_in.shape[0], device=episodes.query_in.device)
    pred = query_tensor[batch_idx, selected]
    exact = (pred == episodes.query_out).flatten(1).all(dim=1).float().mean().item()
    transform_acc = (selected == episodes.transform_ids).float().mean().item()
    return exact, transform_acc


@torch.no_grad()
def evaluate_cnn(model: nn.Module, episodes: Episodes, batch_size: int) -> tuple[float, float]:
    model.eval()
    exact_total = 0.0
    pixel_total = 0.0
    count = 0
    for start in range(0, episodes.query_in.shape[0], batch_size):
        end = start + batch_size
        logits = model(episodes.demo_in[start:end], episodes.demo_out[start:end], episodes.query_in[start:end])
        pred = logits.argmax(dim=1)
        target = episodes.query_out[start:end]
        exact_total += (pred == target).flatten(1).all(dim=1).float().sum().item()
        pixel_total += (pred == target).float().mean(dim=(-2, -1)).sum().item()
        count += target.shape[0]
    return exact_total / count, pixel_total / count


def train_cnn(
    train: Episodes,
    test: Episodes,
    colors: int,
    batch_size: int,
    epochs: int,
    lr: float,
) -> dict:
    model = DemoConditionedCNN(colors).to(train.query_in.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(train.query_in.shape[0], device=train.query_in.device)
        loss_sum = 0.0
        seen = 0
        for start in range(0, permutation.numel(), batch_size):
            idx = permutation[start : start + batch_size]
            logits = model(train.demo_in[idx], train.demo_out[idx], train.query_in[idx])
            loss = F.cross_entropy(logits, train.query_out[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * idx.numel()
            seen += idx.numel()
        exact, pixel = evaluate_cnn(model, test, batch_size)
        history.append(
            {
                "epoch": epoch,
                "loss": loss_sum / max(seen, 1),
                "test_exact": exact,
                "test_pixel": pixel,
            }
        )
    elapsed = time.time() - start_time
    return {"history": history, "elapsed_seconds": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--colors", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--episode-generator",
        choices=("structured", "torch-random"),
        default="structured",
    )
    parser.add_argument("--density", type=float, default=0.18)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed)
    if args.episode_generator == "torch-random":
        train = make_episodes_torch_random(
            args.train_size,
            args.grid_size,
            args.colors,
            args.seed,
            args.density,
            device,
        )
        test = make_episodes_torch_random(
            args.test_size,
            args.grid_size,
            args.colors,
            args.seed + 1,
            args.density,
            device,
        )
    else:
        train = make_episodes(args.train_size, args.grid_size, args.colors, args.seed, device)
        test = make_episodes(args.test_size, args.grid_size, args.colors, args.seed + 1, device)

    gate_exact, gate_transform_acc = selective_symmetry_gate(test)
    cnn = train_cnn(train, test, args.colors, args.batch_size, args.epochs, args.lr)

    summary = {
        "device": str(device),
        "transforms": list(TRANSFORMS),
        "episode_generator": args.episode_generator,
        "density": args.density,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "selective_gate_exact": gate_exact,
        "selective_gate_transform_accuracy": gate_transform_acc,
        "cnn": cnn,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
