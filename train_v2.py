#!/usr/bin/env python3
"""Training script v2: supports GPU pinning, deterministic mode, multi-demo.

Usage:
    CUDA_VISIBLE_DEVICES=0 python train_v2.py --rule-dim 32 --kl-weight 0.0001 --tag low_kl
    CUDA_VISIBLE_DEVICES=1 python train_v2.py --rule-dim 64 --deterministic --tag det_rd64
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from src.ARC_Bottleneck import ARCBottleneck
from src.ARC_Bottleneck_loader import build_bottleneck_loaders, PAD_INDEX


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_eta(seconds: float) -> str:
    s = int(max(seconds, 0))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


@torch.no_grad()
def evaluate(model, loader, device, deterministic=False):
    model.eval()
    total_loss = 0.0
    total_kl = 0.0
    total_exact = 0
    total_pixel_correct = 0
    total_pixels = 0
    total_examples = 0

    for batch in loader:
        demo_in = batch["demo_input"].to(device)
        demo_out = batch["demo_output"].to(device)
        test_in = batch["test_input"].to(device)
        test_out = batch["test_output"].to(device)

        logits, kl = model(demo_in, demo_out, test_in)

        valid = test_out != PAD_INDEX
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, logits.size(1))
        targets_flat = test_out.view(-1).clone()
        targets_flat[targets_flat == PAD_INDEX] = -100
        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100, reduction="sum")

        preds = logits.argmax(dim=1)
        B = preds.size(0)
        for i in range(B):
            v = valid[i]
            if v.any():
                total_exact += int(torch.equal(preds[i][v], test_out[i][v]))
                total_pixel_correct += (preds[i][v] == test_out[i][v]).sum().item()
                total_pixels += v.sum().item()
            total_examples += 1

        total_loss += loss.item()
        total_kl += kl.item() * B

    n = max(total_examples, 1)
    return {
        "loss": total_loss / max(total_pixels, 1),
        "kl": total_kl / n,
        "exact_acc": total_exact / n,
        "pixel_acc": total_pixel_correct / max(total_pixels, 1),
    }


def train(args):
    device = get_device()
    print(f"Device: {device} | Tag: {args.tag}")
    print(f"Config: rule_dim={args.rule_dim}, kl_weight={args.kl_weight}, "
          f"deterministic={args.deterministic}, embed_dim={args.embed_dim}, "
          f"encoder_depth={args.encoder_depth}, num_iterations={args.num_iterations}")

    train_loader, eval_loader = build_bottleneck_loaders(
        data_root=args.data_root,
        grid_size=args.image_size,
        scale_factor=args.scale_factor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = ARCBottleneck(
        image_size=args.image_size,
        num_colors=12,
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        num_heads=args.num_heads,
        rule_dim=args.rule_dim,
        num_iterations=args.num_iterations,
        patch_size=args.patch_size,
        kl_weight=args.kl_weight,
    ).to(device)

    # Monkey-patch for deterministic mode (skip reparameterization)
    if args.deterministic:
        original_forward = model.encoder.forward
        def det_forward(demo_input, demo_output):
            rule, kl = original_forward(demo_input, demo_output)
            # In deterministic mode, kl is computed but not used in loss
            return rule, torch.tensor(0.0, device=rule.device)
        model.encoder.forward = det_forward
        # Also disable stochasticity even in training
        model.encoder.training = False
        print("  [DETERMINISTIC] VAE sampling disabled, using mu directly")

    params = model.count_parameters()
    print(f"Parameters: {params['total']:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    save_dir = Path(args.save_dir) / args.tag
    save_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    global_start = time.time()
    log_lines = []

    for epoch in range(1, args.epochs + 1):
        if args.deterministic:
            model.encoder.eval()
            model.decoder.train()
        else:
            model.train()
            
        running_loss = 0.0
        running_kl = 0.0
        running_ce = 0.0
        n_batches = 0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader, 1):
            demo_in = batch["demo_input"].to(device)
            demo_out = batch["demo_output"].to(device)
            test_in = batch["test_input"].to(device)
            test_out = batch["test_output"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=use_amp):
                logits, kl = model(demo_in, demo_out, test_in)

                targets = test_out.clone()
                targets[targets == PAD_INDEX] = -100
                logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, logits.size(1))
                ce_loss = F.cross_entropy(logits_flat, targets.view(-1), ignore_index=-100)
                loss = ce_loss + model.kl_weight * kl

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_ce += ce_loss.item()
            running_kl += kl.item()
            n_batches += 1

        scheduler.step()

        avg_loss = running_loss / max(n_batches, 1)
        avg_ce = running_ce / max(n_batches, 1)
        avg_kl = running_kl / max(n_batches, 1)
        epoch_time = time.time() - epoch_start

        metrics = evaluate(model, eval_loader, device, args.deterministic)

        line = (f"[{args.tag}] epoch={epoch} loss={avg_loss:.4f} ce={avg_ce:.4f} "
                f"kl={avg_kl:.2f} eval_exact={metrics['exact_acc']:.4f} "
                f"eval_pixel={metrics['pixel_acc']:.4f} time={epoch_time:.1f}s")
        print(line, flush=True)
        log_lines.append(line)

        if metrics["exact_acc"] > best_acc:
            best_acc = metrics["exact_acc"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": vars(args),
                    "epoch": epoch,
                    "best_acc": best_acc,
                },
                save_dir / "best.pt",
            )
            print(f"  -> New best: {best_acc:.4f}", flush=True)

        if epoch % 25 == 0 or epoch == args.epochs:
            torch.save(
                {"model_state": model.state_dict(), "config": vars(args), "epoch": epoch},
                save_dir / f"ep{epoch}.pt",
            )

    total_time = time.time() - global_start
    summary = f"\n[{args.tag}] DONE. Best exact: {best_acc:.4f}, Time: {format_eta(total_time)}"
    print(summary, flush=True)
    log_lines.append(summary)

    with open(save_dir / "train.log", "w") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="raw_data/ARC-AGI")
    p.add_argument("--save-dir", default="saves")
    p.add_argument("--tag", default="exp", help="Experiment name for save subdirectory")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--scale-factor", type=int, default=2)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--encoder-depth", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--rule-dim", type=int, default=32)
    p.add_argument("--num-iterations", type=int, default=4)
    p.add_argument("--patch-size", type=int, default=2)
    p.add_argument("--kl-weight", type=float, default=0.001)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true", help="Skip VAE, use mu directly")
    train(p.parse_args())
