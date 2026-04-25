#!/usr/bin/env python3
"""Synthetic ARC-like test for a global manifold-attention rule model.

The task is one-demo image-to-image translation:

  demo_in  -> demo_out reveals a hidden spatial shift
  query_in -> query_out must apply the same shift

This is deliberately narrower than full ARC. Its purpose is to test a model
principle that vanilla pixel losses do not isolate well:

  1. infer a low-entropy mechanism from demonstrations;
  2. apply it as a spatial flow over the query image.

We compare:

  - DemoConditionedCNN: local convolution over [demo_in, demo_out, query_in].
  - ManifoldRuleTransformer: rule slots globally attend to demo tokens; output
    coordinates attend to rule slots and to all query pixels with a relative
    spatial bias.
  - ShiftTransportAttention: differentiable attention over a spatial-flow
    basis; the selected flow is applied exactly to the query grid.
  - SelectiveShiftGate: zero-training mechanism upper bound by enumerating all
    allowed shifts.

The transformer is trained from scratch. It uses no pretrained VARC weights.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Episodes:
    demo_in: torch.Tensor
    demo_out: torch.Tensor
    query_in: torch.Tensor
    query_out: torch.Tensor
    shift_ids: torch.Tensor
    shifts: list[tuple[int, int]]


def shift_choices(max_shift: int, include_identity: bool) -> list[tuple[int, int]]:
    shifts = []
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            if not include_identity and dy == 0 and dx == 0:
                continue
            shifts.append((dy, dx))
    return shifts


def random_grid_torch(
    n: int,
    height: int,
    width: int,
    colors: int,
    density: float,
    device: torch.device,
) -> torch.Tensor:
    values = torch.randint(1, colors, (n, height, width), device=device)
    mask = torch.rand((n, height, width), device=device) < density
    grids = torch.where(mask, values, torch.zeros((), dtype=torch.long, device=device))

    # Add anchors so different shifts are usually identifiable from one demo.
    anchor_count = 6
    batch_idx = torch.arange(n, device=device).repeat_interleave(anchor_count)
    ys = torch.randint(0, height, (n * anchor_count,), device=device)
    xs = torch.randint(0, width, (n * anchor_count,), device=device)
    anchor_colors = torch.randint(1, colors, (n * anchor_count,), device=device)
    grids[batch_idx, ys, xs] = anchor_colors
    return grids


def apply_shift_batch(
    grids: torch.Tensor,
    shift_ids: torch.Tensor,
    shifts: list[tuple[int, int]],
) -> torch.Tensor:
    out = torch.zeros_like(grids)
    height, width = grids.shape[-2:]
    for idx, (dy, dx) in enumerate(shifts):
        mask = shift_ids == idx
        if not mask.any():
            continue

        y_src0 = max(0, -dy)
        y_src1 = min(height, height - dy)
        x_src0 = max(0, -dx)
        x_src1 = min(width, width - dx)
        y_dst0 = max(0, dy)
        y_dst1 = min(height, height + dy)
        x_dst0 = max(0, dx)
        x_dst1 = min(width, width + dx)
        out[mask, y_dst0:y_dst1, x_dst0:x_dst1] = grids[
            mask, y_src0:y_src1, x_src0:x_src1
        ]
    return out


def make_shift_episodes(
    n: int,
    height: int,
    width: int,
    colors: int,
    density: float,
    max_shift: int,
    include_identity: bool,
    seed: int,
    device: torch.device,
) -> Episodes:
    shifts = shift_choices(max_shift, include_identity)
    if device.type == "cuda":
        devices = [device.index or 0]
    else:
        devices = []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        shift_ids = torch.randint(len(shifts), (n,), device=device)
        demo_in = random_grid_torch(n, height, width, colors, density, device)
        query_in = random_grid_torch(n, height, width, colors, density, device)
    return Episodes(
        demo_in=demo_in,
        demo_out=apply_shift_batch(demo_in, shift_ids, shifts),
        query_in=query_in,
        query_out=apply_shift_batch(query_in, shift_ids, shifts),
        shift_ids=shift_ids,
        shifts=shifts,
    )


def coordinate_grid(
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, height, device=device)
    xs = torch.linspace(-1.0, 1.0, width, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)


class FourierCoordEmbedding(nn.Module):
    def __init__(self, dim: int, num_frequencies: int = 8) -> None:
        super().__init__()
        self.num_frequencies = num_frequencies
        in_dim = 2 + 4 * num_frequencies
        self.proj = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        feats = [coords]
        for freq in range(self.num_frequencies):
            scale = math.pi * (2**freq)
            feats.extend(
                [
                    torch.sin(scale * coords),
                    torch.cos(scale * coords),
                ]
            )
        return self.proj(torch.cat(feats, dim=-1))


class DemoConditionedCNN(nn.Module):
    def __init__(self, colors: int, hidden: int = 64, layers: int = 5) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = colors * 3
        for _ in range(layers):
            blocks.extend(
                [
                    nn.Conv2d(in_channels, hidden, 3, padding=1),
                    nn.GELU(),
                ]
            )
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


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        use_relative_bias: bool = False,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.rel_bias = (
            nn.Sequential(
                nn.Linear(6, dim),
                nn.GELU(),
                nn.Linear(dim, heads),
            )
            if use_relative_bias
            else None
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_coords: torch.Tensor | None = None,
        key_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, query_len, _ = query.shape
        key_len = key_value.shape[1]
        q = self.q_proj(query).view(batch, query_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(batch, key_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(batch, key_len, self.heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.rel_bias is not None:
            if query_coords is None or key_coords is None:
                raise ValueError("relative bias requires coordinates")
            delta = query_coords[:, None, :] - key_coords[None, :, :]
            rel = torch.cat([delta, delta.abs(), delta.square()], dim=-1)
            bias = self.rel_bias(rel).permute(2, 0, 1)
            logits = logits + bias.unsqueeze(0)

        weights = torch.softmax(logits, dim=-1)
        out = torch.matmul(weights, v).transpose(1, 2).reshape(batch, query_len, self.dim)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SlotExtractorBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.slot_norm = nn.LayerNorm(dim)
        self.demo_norm = nn.LayerNorm(dim)
        self.demo_attn = CrossAttention(dim, heads)
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = CrossAttention(dim, heads)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

    def forward(self, slots: torch.Tensor, demo_tokens: torch.Tensor) -> torch.Tensor:
        slots = slots + self.demo_attn(self.slot_norm(slots), self.demo_norm(demo_tokens))
        slots = slots + self.self_attn(self.self_norm(slots), self.self_norm(slots))
        slots = slots + self.ff(self.ff_norm(slots))
        return slots


class ManifoldDecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.out_norm = nn.LayerNorm(dim)
        self.slot_norm = nn.LayerNorm(dim)
        self.slot_attn = CrossAttention(dim, heads)
        self.src_q_norm = nn.LayerNorm(dim)
        self.src_kv_norm = nn.LayerNorm(dim)
        self.source_attn = CrossAttention(dim, heads, use_relative_bias=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

    def forward(
        self,
        out_tokens: torch.Tensor,
        slots: torch.Tensor,
        source_tokens: torch.Tensor,
        out_coords: torch.Tensor,
        source_coords: torch.Tensor,
    ) -> torch.Tensor:
        out_tokens = out_tokens + self.slot_attn(
            self.out_norm(out_tokens),
            self.slot_norm(slots),
        )
        out_tokens = out_tokens + self.source_attn(
            self.src_q_norm(out_tokens),
            self.src_kv_norm(source_tokens),
            out_coords,
            source_coords,
        )
        out_tokens = out_tokens + self.ff(self.ff_norm(out_tokens))
        return out_tokens


class ManifoldRuleTransformer(nn.Module):
    def __init__(
        self,
        colors: int,
        dim: int = 128,
        heads: int = 4,
        slots: int = 8,
        depth: int = 3,
        coord_frequencies: int = 8,
    ) -> None:
        super().__init__()
        self.colors = colors
        self.dim = dim
        self.color_emb = nn.Embedding(colors, dim)
        self.role_emb = nn.Embedding(4, dim)
        self.coord_emb = FourierCoordEmbedding(dim, coord_frequencies)
        self.slots = nn.Parameter(torch.randn(slots, dim) * 0.02)
        self.slot_blocks = nn.ModuleList([SlotExtractorBlock(dim, heads) for _ in range(depth)])
        self.decoder_blocks = nn.ModuleList([ManifoldDecoderBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, colors)

    def pixel_tokens(self, grid: torch.Tensor, role: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch, height, width = grid.shape
        coords = coordinate_grid(height, width, grid.device)
        coord_emb = self.coord_emb(coords).unsqueeze(0)
        color = self.color_emb(grid.reshape(batch, height * width))
        role_emb = self.role_emb(torch.full((height * width,), role, device=grid.device)).unsqueeze(0)
        return color + coord_emb + role_emb, coords

    def output_tokens(
        self,
        batch: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coords = coordinate_grid(height, width, device)
        coord_emb = self.coord_emb(coords).unsqueeze(0).expand(batch, -1, -1)
        role_emb = self.role_emb(torch.full((height * width,), 3, device=device)).unsqueeze(0)
        return coord_emb + role_emb, coords

    def forward(
        self,
        demo_in: torch.Tensor,
        demo_out: torch.Tensor,
        query_in: torch.Tensor,
    ) -> torch.Tensor:
        batch, height, width = query_in.shape
        demo_in_tokens, _ = self.pixel_tokens(demo_in, 0)
        demo_out_tokens, _ = self.pixel_tokens(demo_out, 1)
        source_tokens, source_coords = self.pixel_tokens(query_in, 2)
        demo_tokens = torch.cat([demo_in_tokens, demo_out_tokens], dim=1)

        slots = self.slots.unsqueeze(0).expand(batch, -1, -1)
        for block in self.slot_blocks:
            slots = block(slots, demo_tokens)

        out_tokens, out_coords = self.output_tokens(batch, height, width, query_in.device)
        for block in self.decoder_blocks:
            out_tokens = block(out_tokens, slots, source_tokens, out_coords, source_coords)

        logits = self.head(self.norm(out_tokens))
        return logits.reshape(batch, height, width, self.colors).permute(0, 3, 1, 2)


class ShiftTransportAttention(nn.Module):
    """Attention over a discrete spatial-flow manifold.

    This model is intentionally small. It encodes the prior that many ARC
    transformations are not arbitrary color classifiers; they are transports
    over a low-dimensional family of spatial maps. The learnable part calibrates
    color/background evidence and attention sharpness, while the flow action is
    exact and size-general.
    """

    def __init__(
        self,
        colors: int,
        max_shift: int,
        include_identity: bool = False,
    ) -> None:
        super().__init__()
        self.colors = colors
        self.shifts = shift_choices(max_shift, include_identity)
        self.color_log_weight = nn.Parameter(torch.zeros(colors))
        self.logit_scale = nn.Parameter(torch.tensor(4.0))

    def forward(
        self,
        demo_in: torch.Tensor,
        demo_out: torch.Tensor,
        query_in: torch.Tensor,
    ) -> torch.Tensor:
        batch, height, width = query_in.shape
        demo_out_oh = F.one_hot(demo_out, self.colors).float().permute(0, 3, 1, 2)
        query_accum = torch.zeros(
            batch,
            self.colors,
            height,
            width,
            device=query_in.device,
            dtype=torch.float32,
        )
        scores = []
        shifted_queries = []
        color_weight = self.color_log_weight.exp().view(1, self.colors, 1, 1)

        base_ids = torch.empty(batch, dtype=torch.long, device=query_in.device)
        for idx in range(len(self.shifts)):
            base_ids.fill_(idx)
            shifted_demo = apply_shift_batch(demo_in, base_ids, self.shifts)
            shifted_demo_oh = F.one_hot(shifted_demo, self.colors).float().permute(0, 3, 1, 2)
            match = (shifted_demo_oh * demo_out_oh * color_weight).sum(dim=(1, 2, 3))
            scores.append(match / math.sqrt(height * width))
            shifted_query = apply_shift_batch(query_in, base_ids, self.shifts)
            shifted_queries.append(F.one_hot(shifted_query, self.colors).float().permute(0, 3, 1, 2))

        score_tensor = torch.stack(scores, dim=1) * self.logit_scale.exp().clamp(max=100.0)
        attention = torch.softmax(score_tensor, dim=1)
        for idx, shifted_query in enumerate(shifted_queries):
            query_accum = query_accum + attention[:, idx].view(batch, 1, 1, 1) * shifted_query
        return torch.log(query_accum.clamp_min(1e-6))


@torch.no_grad()
def selective_shift_gate(episodes: Episodes) -> dict[str, float]:
    scores = []
    transformed_queries = []
    for idx in range(len(episodes.shifts)):
        shift_ids = torch.full_like(episodes.shift_ids, idx)
        transformed_demo = apply_shift_batch(episodes.demo_in, shift_ids, episodes.shifts)
        scores.append((transformed_demo == episodes.demo_out).float().mean(dim=(-2, -1)))
        transformed_queries.append(apply_shift_batch(episodes.query_in, shift_ids, episodes.shifts))
    score_tensor = torch.stack(scores, dim=1)
    query_tensor = torch.stack(transformed_queries, dim=1)
    selected = score_tensor.argmax(dim=1)
    batch_idx = torch.arange(episodes.query_in.shape[0], device=episodes.query_in.device)
    pred = query_tensor[batch_idx, selected]
    return compute_metrics(pred, episodes.query_out, selected, episodes.shift_ids)


@torch.no_grad()
def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    selected: torch.Tensor | None = None,
    truth: torch.Tensor | None = None,
) -> dict[str, float]:
    exact = (pred == target).flatten(1).all(dim=1).float().mean().item()
    pixel = (pred == target).float().mean().item()
    target_fg = target != 0
    pred_fg = pred != 0
    fg_total = target_fg.float().sum().clamp_min(1.0)
    bg_total = (~target_fg).float().sum().clamp_min(1.0)
    fg_color_recall = ((pred == target) & target_fg).float().sum() / fg_total
    fg_presence_recall = (pred_fg & target_fg).float().sum() / fg_total
    fg_false_positive_rate = (pred_fg & ~target_fg).float().sum() / bg_total
    out = {
        "exact": exact,
        "pixel": pixel,
        "fg_color_recall": fg_color_recall.item(),
        "fg_presence_recall": fg_presence_recall.item(),
        "fg_false_positive_rate": fg_false_positive_rate.item(),
        "target_fg_fraction": target_fg.float().mean().item(),
    }
    if selected is not None and truth is not None:
        out["rule_accuracy"] = (selected == truth).float().mean().item()
    return out


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    episodes: Episodes,
    batch_size: int,
) -> dict[str, float]:
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


def train_model(
    name: str,
    model: nn.Module,
    train: Episodes,
    test: Episodes,
    ood_test: Episodes | None,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    class_weight: torch.Tensor | None,
    log_every: int,
) -> dict:
    best_exact = 0.0
    start_time = time.time()
    if epochs == 0:
        eval_metric = evaluate_model(model, test, batch_size)
        row = {
            "epoch": 0,
            "loss": None,
            "test_exact": eval_metric["exact"],
            "test_pixel": eval_metric["pixel"],
            "test_fg_color_recall": eval_metric["fg_color_recall"],
            "test_fg_presence_recall": eval_metric["fg_presence_recall"],
            "test_fg_false_positive_rate": eval_metric["fg_false_positive_rate"],
        }
        if ood_test is not None:
            ood_metric = evaluate_model(model, ood_test, batch_size)
            row["ood_exact"] = ood_metric["exact"]
            row["ood_pixel"] = ood_metric["pixel"]
            row["ood_fg_color_recall"] = ood_metric["fg_color_recall"]
            row["ood_fg_presence_recall"] = ood_metric["fg_presence_recall"]
            row["ood_fg_false_positive_rate"] = ood_metric["fg_false_positive_rate"]
        print(f"{name} epoch 0: {json.dumps(row, sort_keys=True)}", flush=True)
        return {
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "best_exact": eval_metric["exact"],
            "history": [row],
            "elapsed_seconds": time.time() - start_time,
        }

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
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

        eval_metric = evaluate_model(model, test, batch_size)
        best_exact = max(best_exact, eval_metric["exact"])
        row = {
            "epoch": epoch,
            "loss": loss_sum / max(seen, 1),
            "test_exact": eval_metric["exact"],
            "test_pixel": eval_metric["pixel"],
            "test_fg_color_recall": eval_metric["fg_color_recall"],
            "test_fg_presence_recall": eval_metric["fg_presence_recall"],
            "test_fg_false_positive_rate": eval_metric["fg_false_positive_rate"],
        }
        if ood_test is not None:
            ood_metric = evaluate_model(model, ood_test, batch_size)
            row["ood_exact"] = ood_metric["exact"]
            row["ood_pixel"] = ood_metric["pixel"]
            row["ood_fg_color_recall"] = ood_metric["fg_color_recall"]
            row["ood_fg_presence_recall"] = ood_metric["fg_presence_recall"]
            row["ood_fg_false_positive_rate"] = ood_metric["fg_false_positive_rate"]
        history.append(row)
        if log_every and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
            print(f"{name} epoch {epoch}: {json.dumps(row, sort_keys=True)}", flush=True)
    return {
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "best_exact": best_exact,
        "history": history,
        "elapsed_seconds": time.time() - start_time,
    }


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
    parser.add_argument("--density", type=float, default=0.16)
    parser.add_argument("--max-shift", type=int, default=6)
    parser.add_argument("--include-identity", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--foreground-loss-weight",
        type=float,
        default=1.0,
        help="Cross-entropy weight for non-background colors.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--models", default="cnn,manifold")
    parser.add_argument("--cnn-hidden", type=int, default=64)
    parser.add_argument("--cnn-layers", type=int, default=5)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--coord-frequencies", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    train = make_shift_episodes(
        args.train_size,
        args.grid_size,
        args.grid_size,
        args.colors,
        args.density,
        args.max_shift,
        args.include_identity,
        args.seed,
        device,
    )
    test = make_shift_episodes(
        args.test_size,
        args.grid_size,
        args.grid_size,
        args.colors,
        args.density,
        args.max_shift,
        args.include_identity,
        args.seed + 1,
        device,
    )
    ood_test = None
    if args.ood_grid_size:
        ood_test = make_shift_episodes(
            args.test_size,
            args.ood_grid_size,
            args.ood_grid_size,
            args.colors,
            args.density,
            args.max_shift,
            args.include_identity,
            args.seed + 2,
            device,
        )

    selected_models = {name.strip() for name in args.models.split(",") if name.strip()}
    summary: dict[str, object] = {
        "device": str(device),
        "task": "one_demo_spatial_shift",
        "train_size": args.train_size,
        "test_size": args.test_size,
        "grid_size": args.grid_size,
        "ood_grid_size": args.ood_grid_size,
        "colors": args.colors,
        "density": args.density,
        "max_shift": args.max_shift,
        "shift_count": len(train.shifts),
        "seed": args.seed,
        "foreground_loss_weight": args.foreground_loss_weight,
        "selective_shift_gate": selective_shift_gate(test),
        "models": {},
    }
    if ood_test is not None:
        summary["selective_shift_gate_ood"] = selective_shift_gate(ood_test)

    if "cnn" in selected_models:
        class_weight = torch.ones(args.colors, device=device)
        class_weight[1:] = args.foreground_loss_weight
        cnn = DemoConditionedCNN(args.colors, args.cnn_hidden, args.cnn_layers).to(device)
        summary["models"]["cnn"] = train_model(
            "cnn",
            cnn,
            train,
            test,
            ood_test,
            args.epochs,
            args.batch_size,
            args.lr,
            args.weight_decay,
            class_weight,
            args.log_every,
        )

    if "manifold" in selected_models:
        class_weight = torch.ones(args.colors, device=device)
        class_weight[1:] = args.foreground_loss_weight
        manifold = ManifoldRuleTransformer(
            colors=args.colors,
            dim=args.dim,
            heads=args.heads,
            slots=args.slots,
            depth=args.depth,
            coord_frequencies=args.coord_frequencies,
        ).to(device)
        summary["models"]["manifold"] = train_model(
            "manifold",
            manifold,
            train,
            test,
            ood_test,
            args.epochs,
            args.batch_size,
            args.lr,
            args.weight_decay,
            class_weight,
            args.log_every,
        )

    if "transport" in selected_models:
        class_weight = torch.ones(args.colors, device=device)
        class_weight[1:] = args.foreground_loss_weight
        transport = ShiftTransportAttention(
            colors=args.colors,
            max_shift=args.max_shift,
            include_identity=args.include_identity,
        ).to(device)
        summary["models"]["transport"] = train_model(
            "transport",
            transport,
            train,
            test,
            ood_test,
            args.epochs,
            args.batch_size,
            args.lr,
            args.weight_decay,
            class_weight,
            args.log_every,
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
