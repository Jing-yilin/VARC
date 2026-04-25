#!/usr/bin/env python3
"""Prepare alternative VARC test-time-training augmentation splits."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from utils.arclib.augmenters import IdentityAugmenter, Transpose
from utils.data_augmentation import augment_raw_data_split_per_task
from utils.preprocess import get_basic_augmenters


def copy_original_per_task(data_root: Path, split: str, output_subdir: str) -> int:
    source_dir = data_root / "data" / split
    output_root = data_root / "data" / output_subdir
    output_root.mkdir(parents=True, exist_ok=True)
    count = 0
    for source_path in sorted(source_dir.glob("*.json")):
        task_dir = output_root / source_path.stem
        task_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, task_dir / source_path.name)
        count += 1
    return count


def build_augmenters(variant: str):
    basic = get_basic_augmenters()
    if variant == "paper":
        return basic
    if variant == "identity_color_plus_paper":
        return [IdentityAugmenter()] + basic
    if variant == "transpose_color_plus_paper":
        return basic + [Transpose()]
    if variant == "identity_transpose_color_plus_paper":
        return [IdentityAugmenter()] + basic + [Transpose()]
    if variant == "geom_only":
        return basic
    raise ValueError(f"Unsupported variant: {variant}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("raw_data/ARC-AGI"))
    parser.add_argument("--split", default="evaluation", choices=["training", "evaluation"])
    parser.add_argument("--output-subdir", required=True)
    parser.add_argument(
        "--variant",
        choices=[
            "original_only",
            "geom_only",
            "paper",
            "identity_color_plus_paper",
            "transpose_color_plus_paper",
            "identity_transpose_color_plus_paper",
        ],
        required=True,
    )
    parser.add_argument("--num-permutations", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    output_root = args.data_root / "data" / args.output_subdir
    if args.clean and output_root.exists():
        shutil.rmtree(output_root)

    if args.variant == "original_only":
        count = copy_original_per_task(args.data_root, args.split, args.output_subdir)
        print(f"prepared {count} original-only task directories under {output_root}")
        return

    num_permutations = 0 if args.variant == "geom_only" else args.num_permutations
    saved = augment_raw_data_split_per_task(
        dataset_root=args.data_root,
        split=args.split,
        output_subdir=args.output_subdir,
        augmenters=build_augmenters(args.variant),
        num_permuate=num_permutations,
        seed=args.seed,
        verbose=True,
    )
    print(f"prepared {len(saved)} augmented files under {output_root}")


if __name__ == "__main__":
    main()
