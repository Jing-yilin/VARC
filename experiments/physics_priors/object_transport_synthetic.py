#!/usr/bin/env python3
"""Synthetic ARC-like object transport experiment.

This extends the earlier global-shift test into a more ARC-relevant setting:

  demo_in  -> demo_out reveals which object color moves/copies and by what flow
  query_in -> query_out must apply the same object-level mechanism

The scene contains distractor objects that should remain fixed. A global image
shift is therefore wrong; the model must identify the selected object/color,
infer the spatial flow, and execute either move or copy.

We compare:

  - DemoConditionedCNN: local learned baseline.
  - GlobalShiftGate: explicit whole-image flow selection, expected to fail on
    selective object transport.
  - SelectiveTransportGate: explicit attention over mechanism space
    (selected color, shift, move/copy) followed by exact execution.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


MODES = ("move", "copy")


@dataclass
class Episodes:
    demo_in: torch.Tensor
    demo_out: torch.Tensor
    query_in: torch.Tensor
    query_out: torch.Tensor
    color_ids: torch.Tensor
    shift_ids: torch.Tensor
    mode_ids: torch.Tensor
    shifts: list[tuple[int, int]]


def shift_choices(max_shift: int, include_identity: bool = False) -> list[tuple[int, int]]:
    shifts = []
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            if not include_identity and dy == 0 and dx == 0:
                continue
            shifts.append((dy, dx))
    return shifts


def shift_grid(grid: torch.Tensor, dy: int, dx: int, fill: int = 0) -> torch.Tensor:
    out = torch.full_like(grid, fill)
    height, width = grid.shape[-2:]
    y_src0 = max(0, -dy)
    y_src1 = min(height, height - dy)
    x_src0 = max(0, -dx)
    x_src1 = min(width, width - dx)
    y_dst0 = max(0, dy)
    y_dst1 = min(height, height + dy)
    x_dst0 = max(0, dx)
    x_dst1 = min(width, width + dx)
    out[..., y_dst0:y_dst1, x_dst0:x_dst1] = grid[..., y_src0:y_src1, x_src0:x_src1]
    return out


def apply_selective_transport(
    grids: torch.Tensor,
    color: int,
    shift: tuple[int, int],
    mode: str,
) -> torch.Tensor:
    dy, dx = shift
    mask = grids == color
    shifted_mask = shift_grid(mask.long(), dy, dx).bool()
    out = grids.clone()
    if mode == "move":
        out[mask] = 0
    out[shifted_mask] = color
    return out


def draw_rect(
    grid: np.ndarray,
    color: int,
    y: int,
    x: int,
    h: int,
    w: int,
) -> None:
    grid[y : y + h, x : x + w] = color


def random_rect(rng: random.Random, height: int, width: int, max_obj: int) -> tuple[int, int, int, int]:
    h = rng.randint(1, max_obj)
    w = rng.randint(1, max_obj)
    y = rng.randint(0, height - h)
    x = rng.randint(0, width - w)
    return y, x, h, w


def source_rect_for_shift(
    rng: random.Random,
    height: int,
    width: int,
    max_obj: int,
    dy: int,
    dx: int,
) -> tuple[int, int, int, int]:
    h = rng.randint(1, max_obj)
    w = rng.randint(1, max_obj)
    y_min = max(0, -dy)
    y_max = min(height - h, height - h - dy)
    x_min = max(0, -dx)
    x_max = min(width - w, width - w - dx)
    if y_max < y_min or x_max < x_min:
        return random_rect(rng, height, width, max_obj)
    return rng.randint(y_min, y_max), rng.randint(x_min, x_max), h, w


def add_distractors(
    rng: random.Random,
    grid: np.ndarray,
    colors: int,
    selected_color: int,
    distractors: int,
    max_obj: int,
) -> None:
    height, width = grid.shape
    allowed = [color for color in range(1, colors) if color != selected_color]
    for _ in range(distractors):
        color = rng.choice(allowed)
        y, x, h, w = random_rect(rng, height, width, max_obj)
        draw_rect(grid, color, y, x, h, w)


def make_scene(
    rng: random.Random,
    height: int,
    width: int,
    colors: int,
    selected_color: int,
    shift: tuple[int, int],
    mode: str,
    distractors: int,
    max_obj: int,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros((height, width), dtype=np.int64)
    add_distractors(rng, grid, colors, selected_color, distractors, max_obj)
    y, x, h, w = source_rect_for_shift(rng, height, width, max_obj, *shift)
    draw_rect(grid, selected_color, y, x, h, w)

    tensor = torch.tensor(grid[None, ...], dtype=torch.long)
    out = apply_selective_transport(tensor, selected_color, shift, mode)[0].numpy()
    return grid, out


def make_episodes(
    n: int,
    height: int,
    width: int,
    colors: int,
    max_shift: int,
    distractors: int,
    max_obj: int,
    seed: int,
    device: torch.device,
) -> Episodes:
    rng = random.Random(seed)
    shifts = shift_choices(max_shift)
    demo_in = []
    demo_out = []
    query_in = []
    query_out = []
    color_ids = []
    shift_ids = []
    mode_ids = []
    for _ in range(n):
        selected_color = rng.randint(1, colors - 1)
        shift_id = rng.randrange(len(shifts))
        mode_id = rng.randrange(len(MODES))
        shift = shifts[shift_id]
        mode = MODES[mode_id]

        din, dout = make_scene(
            rng, height, width, colors, selected_color, shift, mode, distractors, max_obj
        )
        qin, qout = make_scene(
            rng, height, width, colors, selected_color, shift, mode, distractors, max_obj
        )
        demo_in.append(din)
        demo_out.append(dout)
        query_in.append(qin)
        query_out.append(qout)
        color_ids.append(selected_color)
        shift_ids.append(shift_id)
        mode_ids.append(mode_id)

    def tensor(items: list[np.ndarray]) -> torch.Tensor:
        return torch.tensor(np.stack(items), dtype=torch.long, device=device)

    return Episodes(
        demo_in=tensor(demo_in),
        demo_out=tensor(demo_out),
        query_in=tensor(query_in),
        query_out=tensor(query_out),
        color_ids=torch.tensor(color_ids, dtype=torch.long, device=device),
        shift_ids=torch.tensor(shift_ids, dtype=torch.long, device=device),
        mode_ids=torch.tensor(mode_ids, dtype=torch.long, device=device),
        shifts=shifts,
    )


class DemoConditionedCNN(nn.Module):
    def __init__(self, colors: int, hidden: int = 64, layers: int = 5) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = colors * 3
        for _ in range(layers):
            blocks.extend([nn.Conv2d(in_channels, hidden, 3, padding=1), nn.GELU()])
            in_channels = hidden
        blocks.append(nn.Conv2d(hidden, colors, 1))
        self.colors = colors
        self.net = nn.Sequential(*blocks)

    def forward(
        self,
        demo_in: torch.Tensor,
        demo_out: torch.Tensor,
        query_in: torch.Tensor,
    ) -> torch.Tensor:
        parts = [
            F.one_hot(demo_in, self.colors),
            F.one_hot(demo_out, self.colors),
            F.one_hot(query_in, self.colors),
        ]
        x = torch.cat(parts, dim=-1).float().permute(0, 3, 1, 2)
        return self.net(x)


@torch.no_grad()
def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    selected_color: torch.Tensor | None = None,
    true_color: torch.Tensor | None = None,
    selected_shift: torch.Tensor | None = None,
    true_shift: torch.Tensor | None = None,
    selected_mode: torch.Tensor | None = None,
    true_mode: torch.Tensor | None = None,
) -> dict[str, float]:
    exact = (pred == target).flatten(1).all(dim=1).float().mean().item()
    pixel = (pred == target).float().mean().item()
    target_fg = target != 0
    pred_fg = pred != 0
    fg_total = target_fg.float().sum().clamp_min(1.0)
    bg_total = (~target_fg).float().sum().clamp_min(1.0)
    out = {
        "exact": exact,
        "pixel": pixel,
        "fg_color_recall": (((pred == target) & target_fg).float().sum() / fg_total).item(),
        "fg_presence_recall": ((pred_fg & target_fg).float().sum() / fg_total).item(),
        "fg_false_positive_rate": ((pred_fg & ~target_fg).float().sum() / bg_total).item(),
        "target_fg_fraction": target_fg.float().mean().item(),
    }
    if selected_color is not None and true_color is not None:
        out["color_accuracy"] = (selected_color == true_color).float().mean().item()
    if selected_shift is not None and true_shift is not None:
        out["shift_accuracy"] = (selected_shift == true_shift).float().mean().item()
    if selected_mode is not None and true_mode is not None:
        out["mode_accuracy"] = (selected_mode == true_mode).float().mean().item()
    return out


@torch.no_grad()
def evaluate_cnn(model: nn.Module, episodes: Episodes, batch_size: int) -> dict[str, float]:
    model.eval()
    preds = []
    targets = []
    for start in range(0, episodes.query_in.shape[0], batch_size):
        end = start + batch_size
        logits = model(
            episodes.demo_in[start:end],
            episodes.demo_out[start:end],
            episodes.query_in[start:end],
        )
        preds.append(logits.argmax(dim=1))
        targets.append(episodes.query_out[start:end])
    return compute_metrics(torch.cat(preds, dim=0), torch.cat(targets, dim=0))


def train_cnn(
    train: Episodes,
    test: Episodes,
    ood_test: Episodes | None,
    colors: int,
    hidden: int,
    layers: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    foreground_loss_weight: float,
    log_every: int,
) -> dict:
    model = DemoConditionedCNN(colors, hidden, layers).to(train.query_in.device)
    class_weight = torch.ones(colors, device=train.query_in.device)
    class_weight[1:] = foreground_loss_weight
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    start_time = time.time()
    best_exact = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(train.query_in.shape[0], device=train.query_in.device)
        loss_sum = 0.0
        seen = 0
        for start in range(0, permutation.numel(), batch_size):
            idx = permutation[start : start + batch_size]
            logits = model(train.demo_in[idx], train.demo_out[idx], train.query_in[idx])
            loss = F.cross_entropy(logits, train.query_out[idx], weight=class_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * idx.numel()
            seen += idx.numel()
        metric = evaluate_cnn(model, test, batch_size)
        best_exact = max(best_exact, metric["exact"])
        row = {
            "epoch": epoch,
            "loss": loss_sum / max(seen, 1),
            "test_exact": metric["exact"],
            "test_pixel": metric["pixel"],
            "test_fg_color_recall": metric["fg_color_recall"],
            "test_fg_false_positive_rate": metric["fg_false_positive_rate"],
        }
        if ood_test is not None:
            ood = evaluate_cnn(model, ood_test, batch_size)
            row["ood_exact"] = ood["exact"]
            row["ood_pixel"] = ood["pixel"]
            row["ood_fg_color_recall"] = ood["fg_color_recall"]
            row["ood_fg_false_positive_rate"] = ood["fg_false_positive_rate"]
        history.append(row)
        if log_every and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
            print(f"cnn epoch {epoch}: {json.dumps(row, sort_keys=True)}", flush=True)
    return {
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "best_exact": best_exact,
        "history": history,
        "elapsed_seconds": time.time() - start_time,
    }


@torch.no_grad()
def global_shift_gate(episodes: Episodes, batch_size: int) -> dict[str, float]:
    preds = []
    selected_shifts = []
    for start in range(0, episodes.query_in.shape[0], batch_size):
        end = start + batch_size
        demo_in = episodes.demo_in[start:end]
        demo_out = episodes.demo_out[start:end]
        query_in = episodes.query_in[start:end]
        scores = []
        shifted_queries = []
        for shift in episodes.shifts:
            shifted_demo = shift_grid(demo_in, *shift)
            scores.append((shifted_demo == demo_out).float().mean(dim=(-2, -1)))
            shifted_queries.append(shift_grid(query_in, *shift))
        score_tensor = torch.stack(scores, dim=1)
        selected = score_tensor.argmax(dim=1)
        batch_idx = torch.arange(query_in.shape[0], device=query_in.device)
        query_tensor = torch.stack(shifted_queries, dim=1)
        preds.append(query_tensor[batch_idx, selected])
        selected_shifts.append(selected)
    pred = torch.cat(preds, dim=0)
    selected_shift = torch.cat(selected_shifts, dim=0)
    return compute_metrics(
        pred,
        episodes.query_out,
        selected_shift=selected_shift,
        true_shift=episodes.shift_ids,
    )


def _candidate_score(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pixel = (candidate == target).float().mean(dim=(-2, -1))
    fg = target != 0
    fg_total = fg.float().sum(dim=(-2, -1)).clamp_min(1.0)
    fg_match = ((candidate == target) & fg).float().sum(dim=(-2, -1)) / fg_total
    return pixel + fg_match


@torch.no_grad()
def selective_transport_gate(episodes: Episodes, batch_size: int) -> dict[str, float]:
    preds = []
    selected_colors = []
    selected_shifts = []
    selected_modes = []
    colors = list(range(1, int(max(episodes.demo_in.max(), episodes.demo_out.max()).item()) + 1))
    for start in range(0, episodes.query_in.shape[0], batch_size):
        end = start + batch_size
        demo_in = episodes.demo_in[start:end]
        demo_out = episodes.demo_out[start:end]
        query_in = episodes.query_in[start:end]
        batch = demo_in.shape[0]

        best_score = torch.full((batch,), -float("inf"), device=demo_in.device)
        best_color = torch.zeros((batch,), dtype=torch.long, device=demo_in.device)
        best_shift = torch.zeros((batch,), dtype=torch.long, device=demo_in.device)
        best_mode = torch.zeros((batch,), dtype=torch.long, device=demo_in.device)
        best_pred = torch.zeros_like(query_in)

        for color in colors:
            for shift_id, shift in enumerate(episodes.shifts):
                for mode_id, mode in enumerate(MODES):
                    candidate_demo = apply_selective_transport(demo_in, color, shift, mode)
                    score = _candidate_score(candidate_demo, demo_out)
                    improve = score > best_score
                    if not improve.any():
                        continue
                    candidate_query = apply_selective_transport(query_in, color, shift, mode)
                    best_score = torch.where(improve, score, best_score)
                    best_color = torch.where(
                        improve,
                        torch.full_like(best_color, color),
                        best_color,
                    )
                    best_shift = torch.where(
                        improve,
                        torch.full_like(best_shift, shift_id),
                        best_shift,
                    )
                    best_mode = torch.where(
                        improve,
                        torch.full_like(best_mode, mode_id),
                        best_mode,
                    )
                    best_pred[improve] = candidate_query[improve]

        preds.append(best_pred)
        selected_colors.append(best_color)
        selected_shifts.append(best_shift)
        selected_modes.append(best_mode)

    return compute_metrics(
        torch.cat(preds, dim=0),
        episodes.query_out,
        selected_color=torch.cat(selected_colors, dim=0),
        true_color=episodes.color_ids,
        selected_shift=torch.cat(selected_shifts, dim=0),
        true_shift=episodes.shift_ids,
        selected_mode=torch.cat(selected_modes, dim=0),
        true_mode=episodes.mode_ids,
    )


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--test-size", type=int, default=5000)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--ood-grid-size", type=int, default=0)
    parser.add_argument("--colors", type=int, default=6)
    parser.add_argument("--max-shift", type=int, default=3)
    parser.add_argument("--distractors", type=int, default=4)
    parser.add_argument("--max-obj", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gate-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--foreground-loss-weight", type=float, default=6.0)
    parser.add_argument("--cnn-hidden", type=int, default=64)
    parser.add_argument("--cnn-layers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--models", default="cnn,global_gate,selective_gate")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    train = make_episodes(
        args.train_size,
        args.grid_size,
        args.grid_size,
        args.colors,
        args.max_shift,
        args.distractors,
        args.max_obj,
        args.seed,
        device,
    )
    test = make_episodes(
        args.test_size,
        args.grid_size,
        args.grid_size,
        args.colors,
        args.max_shift,
        args.distractors,
        args.max_obj,
        args.seed + 1,
        device,
    )
    ood_test = None
    if args.ood_grid_size:
        ood_test = make_episodes(
            args.test_size,
            args.ood_grid_size,
            args.ood_grid_size,
            args.colors,
            args.max_shift,
            args.distractors,
            args.max_obj,
            args.seed + 2,
            device,
        )

    selected = {name.strip() for name in args.models.split(",") if name.strip()}
    summary: dict[str, object] = {
        "device": str(device),
        "task": "selective_object_transport",
        "train_size": args.train_size,
        "test_size": args.test_size,
        "grid_size": args.grid_size,
        "ood_grid_size": args.ood_grid_size,
        "colors": args.colors,
        "max_shift": args.max_shift,
        "shift_count": len(train.shifts),
        "mechanism_count": (args.colors - 1) * len(train.shifts) * len(MODES),
        "distractors": args.distractors,
        "max_obj": args.max_obj,
        "foreground_loss_weight": args.foreground_loss_weight,
        "models": {},
    }

    if "global_gate" in selected:
        start = time.time()
        summary["models"]["global_gate"] = {
            "test": global_shift_gate(test, args.gate_batch_size),
            "elapsed_seconds": time.time() - start,
        }
        if ood_test is not None:
            start = time.time()
            summary["models"]["global_gate"]["ood"] = global_shift_gate(
                ood_test, args.gate_batch_size
            )
            summary["models"]["global_gate"]["ood_elapsed_seconds"] = time.time() - start

    if "selective_gate" in selected:
        start = time.time()
        summary["models"]["selective_gate"] = {
            "test": selective_transport_gate(test, args.gate_batch_size),
            "elapsed_seconds": time.time() - start,
        }
        if ood_test is not None:
            start = time.time()
            summary["models"]["selective_gate"]["ood"] = selective_transport_gate(
                ood_test, args.gate_batch_size
            )
            summary["models"]["selective_gate"]["ood_elapsed_seconds"] = time.time() - start

    if "cnn" in selected and args.epochs > 0:
        summary["models"]["cnn"] = train_cnn(
            train,
            test,
            ood_test,
            args.colors,
            args.cnn_hidden,
            args.cnn_layers,
            args.epochs,
            args.batch_size,
            args.lr,
            args.weight_decay,
            args.foreground_loss_weight,
            args.log_every,
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
