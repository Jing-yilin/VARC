#!/bin/bash
# Train ARCBottleneck on cloud GPU.
# Matches the original VARC paper's training setup where possible.
#
# Recommended: 1x 4090 (24GB) or 1x A100-40G — model is only 4.6M params.
# For multi-GPU comparison with original VARC, use 8x GPUs.
#
# Usage:
#   bash script/train_bottleneck_cloud.sh          # single GPU
#   bash script/train_bottleneck_cloud.sh --multi   # 8x GPU baseline comparison

set -euo pipefail

if [[ "${1:-}" == "--multi" ]]; then
    echo "=== Training original VARC baseline (8 GPU, paper config) ==="
    torchrun --nproc_per_node=8 offline_train_ARC.py \
        --epochs 100 \
        --depth 10 \
        --batch-size 32 \
        --image-size 64 \
        --patch-size 2 \
        --learning-rate 3e-4 \
        --weight-decay 0 \
        --embed-dim 512 \
        --num-heads 8 \
        --include-rearc \
        --num-colors 12 \
        --data-root "raw_data/ARC-AGI" \
        --train-split "training" \
        --eval-split "training" \
        --save-path "saves/baseline_vit/checkpoint_final.pt" \
        --best-save-path "saves/baseline_vit/checkpoint_best.pt" \
        --lr-scheduler "cosine" \
        --architecture "vit" \
        --vis-every 25 \
        --distributed

    echo "=== Baseline done. Now training bottleneck model ==="
fi

echo "=== Training ARCBottleneck (info bottleneck, single GPU) ==="

# Experiment 1: rule_dim=32 (main hypothesis)
python -u train_bottleneck.py \
    --epochs 100 \
    --batch-size 32 \
    --image-size 64 \
    --patch-size 2 \
    --embed-dim 256 \
    --encoder-depth 4 \
    --num-heads 8 \
    --rule-dim 32 \
    --num-iterations 4 \
    --kl-weight 0.001 \
    --lr 3e-4 \
    --weight-decay 0.05 \
    --save-dir "saves/bottleneck_rd32" \
    --data-root "raw_data/ARC-AGI"

# Experiment 2: rule_dim=64 (ablation — more capacity)
python -u train_bottleneck.py \
    --epochs 100 \
    --batch-size 32 \
    --image-size 64 \
    --patch-size 2 \
    --embed-dim 256 \
    --encoder-depth 4 \
    --num-heads 8 \
    --rule-dim 64 \
    --num-iterations 4 \
    --kl-weight 0.001 \
    --lr 3e-4 \
    --weight-decay 0.05 \
    --save-dir "saves/bottleneck_rd64" \
    --data-root "raw_data/ARC-AGI"

# Experiment 3: rule_dim=16 (ablation — extreme compression)
python -u train_bottleneck.py \
    --epochs 100 \
    --batch-size 32 \
    --image-size 64 \
    --patch-size 2 \
    --embed-dim 256 \
    --encoder-depth 4 \
    --num-heads 8 \
    --rule-dim 16 \
    --num-iterations 4 \
    --kl-weight 0.001 \
    --lr 3e-4 \
    --weight-decay 0.05 \
    --save-dir "saves/bottleneck_rd16" \
    --data-root "raw_data/ARC-AGI"

echo "=== All experiments complete ==="
echo "Results saved in saves/bottleneck_rd{16,32,64}/"
