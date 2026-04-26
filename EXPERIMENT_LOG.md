# Information Bottleneck ARC — Experiment Log

**Goal**: Surpass VARC (arxiv 2511.14761) pure vision SOTA on ARC-AGI.

**VARC baseline** (18M params, ViT + task tokens):
- Offline single-view pass@1: ~35.9% on ARC eval set
- + Multi-view (510 views): improves pass@1
- + TTT (100 epochs per task): 54.5% pass@2 on ARC-1 eval
- + Ensemble (ViT+UNet): 60.4% pass@2
- ARC-2 eval: 8.3% (single), 11.1% (ensemble)

**Our approach**: Dual-path hybrid — rule vector (information bottleneck) + task tokens (memorization). Encoder extracts compressed rule from demo pairs; decoder cross-attends to both rule tokens and task tokens. Task token dropout (30%) forces learning both paths.

---

## Experiment Timeline

### V1-V7: Early exploration (prior session)
- V1: Basic bottleneck, rule_dim=32 → peaked ~1.9%
- V2: Multi-demo (3 demos), rule_dim=64, deeper encoder → peaked **4.09%** (best before this session)
- V3-V5: Change-weighted loss, RE-ARC, color augmentation — all HURT performance
- V6: Mixed ARC+RE-ARC → ARC-AGI-2 always 0%
- V7: Big model (24.3M) — too slow, killed early

### V8: Pure 5-demo (no tricks)
- **Config**: V2 arch, 5 demos, rule_dim=64, embed_dim=256, enc_depth=6, dec_depth=2, iter=6, 8.6M params
- **Result**: Best exact = **3.37%** at epoch 111. Plateaued at 2.9-3.4% range after epoch 100.
- **Conclusion**: 5 demos slightly worse than 3 demos for this architecture. Training loss went very low (0.02) but eval didn't follow — overfitting.

### V9: Hybrid (rule + task tokens) — KEY EXPERIMENT
- **Config**: V3 arch (hybrid), 5 demos, rule_dim=64, embed_dim=256, enc_depth=6, dec_depth=2, iter=6, task_dropout=0.3, 8.7M params
- **Training**: ARC training set only (no RE-ARC)
- **Result**: Best exact = **4.33%** at epoch ~118
  - Faster convergence than V8: matched V8's 3.37% by epoch 32 (V8 needed epoch 111)
  - no_task mode (rule only): 2.16% at epoch 100
  - Task tokens clearly help: 4.33% with task vs 2.16% without
- **Conclusion**: Hybrid approach validated. Task tokens provide ~2pp boost. Rule vector still contributes.

### V4: Combo (change-weight + color augment) — control
- **Config**: V2 arch, 4 demos, change_weight=3.0, color_augment, 8.6M params
- **Result**: Best exact = **2.40%** at epoch 235. Finished 300 epochs.
- **Conclusion**: Change-weighted loss and color augmentation HURT. Confirmed "back to basics" wins.

### V10: Hybrid + RE-ARC (small model)
- **Config**: V3 arch, 5 demos, rule_dim=64, embed_dim=256, 8.7M params, RE-ARC 8 samples/task
- **Training**: Mixed ARC original (3200/epoch) + RE-ARC (3200/epoch) = 6400/epoch
- **Result**: Best = **3.61%** at epoch ~70
  - Pixel accuracy 80.8% >> V9's 76.5% — RE-ARC helps pixel-level learning
  - eval_new (ARC evaluation, unseen tasks): 0.72% at epoch 70
  - no_task mode: 2.40% at epoch 70

### Multi-view eval on V9 best checkpoint
- 100 views, pass@2 on ARC training set
- Result: pass@1 = 2.64%, pass@2 = 3.85%
- Multi-view DECREASED pass@1 (from 3.37% single-view) — our translation augmentation strategy needs refinement
- pass@2 improved (3.37% → 3.85%) — diversity helps

---

## VARC vs Hybrid A/B Comparison (COMPLETED 2026-04-26) ⭐

**Instance**: 4×4090 (ssh -p 41075 root@hbd1.550w.link)

**Purpose**: Fair controlled comparison. Same data (RE-ARC 8/task, 6400 samples/epoch), same compute, same epochs (300), only architecture differs.

### VARC Baseline (varc_baseline_19M) — DONE
- **Config**: Original VARC ViT, embed=384, depth=8, heads=8, 18.9M params, batch=12
- **Training curve**:

| Epoch | Loss | Exact | Pixel | rand_task | eval_new |
|-------|------|-------|-------|-----------|----------|
| 10 | 0.5695 | 2.64% | 78.8% | 0.72% | 0.24% |
| 20 | 0.3650 | 4.81% | 80.6% | 1.44% | 0.48% |
| 30 | 0.2873 | 6.97% | 81.3% | 1.20% | 0.24% |
| 50 | 0.2252 | 9.38% | 82.3% | 1.20% | 0.48% |
| 80 | 0.1899 | 9.62% | 82.6% | 0.96% | 0.24% |
| 100 | 0.1750 | 12.74% | 84.4% | 0.96% | 0.72% |
| 150 | 0.1321 | 18.51% | 85.3% | 0.72% | 0.48% |
| 200 | 0.1042 | 21.88% | 87.7% | 1.44% | 0.48% |
| 250 | 0.0864 | 24.76% | 89.3% | 0.72% | 0.24% |
| 270 | 0.0861 | **26.68%** | 89.9% | 1.68% | 0.48% |
| 300 | 0.0795 | 25.72% | 89.8% | 1.68% | 0.48% |

- **Best exact: 26.68%** at epoch 270 (111/416 test examples). Time: 13.3h (157s/epoch)
- **eval_new: 0.48%** (2/419 unseen evaluation tasks) — essentially zero generalization
- **rand_task: 1.68%** (random task token) — near zero without correct task_id

### Hybrid V3 19M (hybrid_19M_singlegpu) — DONE
- **Config**: V3 arch (enc=6+dec=2, rule=64, iter=6), embed=384, 19.0M params, task_dropout=0.3, batch=12
- **Training curve**:

| Epoch | Loss | CE | KL | Exact | Pixel | no_task | eval_new |
|-------|------|-----|-----|-------|-------|---------|----------|
| 10 | 0.8912 | 0.8869 | 42.88 | 1.68% | 77.8% | 1.20% | 0.48% |
| 20 | 0.7330 | 0.7291 | 39.16 | 1.92% | 80.3% | 1.92% | 0.48% |
| 30 | 0.6917 | 0.6875 | 42.14 | 2.88% | 79.8% | 1.68% | 0.72% |
| 50 | 0.6318 | 0.6279 | 39.35 | 3.12% | 80.8% | 2.40% | 0.72% |
| 80 | 0.5870 | 0.5831 | 38.99 | 3.85% | 81.2% | 1.92% | 0.24% |
| 100 | 0.5588 | 0.5550 | 37.83 | 3.37% | 80.8% | 2.40% | 0.72% |
| 150 | 0.4765 | 0.4727 | 38.16 | 4.33% | 80.6% | 3.37% | 0.48% |
| 200 | 0.4125 | 0.4083 | 41.65 | 4.57% | 79.7% | 3.12% | 0.48% |
| 250 | 0.3693 | 0.3650 | 43.40 | 4.57% | 79.0% | 3.61% | 0.24% |
| 270 | 0.3654 | 0.3612 | 42.90 | 4.57% | 78.8% | 3.12% | 0.48% |
| 300 | 0.3553 | 0.3509 | 43.69 | 5.05% | 78.8% | 4.09% | 0.24% |

- **Best exact: 5.77%** at epoch 260 (24/416). Time: 15.2h (170s/epoch)
- **eval_new: 0.95%** peak at epoch 190 (4/419 unseen tasks) — slightly better than VARC
- **no_task: 4.09%** at epoch 300 — pure rule generalization

### A/B Comparison Summary

| Metric | VARC | Hybrid | Winner |
|--------|------|--------|--------|
| Best exact (training tasks) | **26.68%** | 5.77% | VARC (4.6×) |
| Pixel accuracy | **89.8%** | 78.8% | VARC |
| eval_new (unseen tasks) | 0.48% | **0.95%** | Hybrid (2×) |
| no_task / rand_task | 1.68% | **4.09%** | Hybrid (2.4×) |
| Training loss | **0.08** | 0.35 | VARC |
| Params | 18.9M | 19.0M | Equal |
| Speed | **157s/ep** | 170s/ep | VARC |

**Verdict**: VARC dominates on training task memorization (26.68% vs 5.77%). Our hybrid has a small edge on unseen task generalization (0.95% vs 0.48%) but the gap is tiny. The encoder consumes 65% of parameters (12.3M) but compresses rules to 64-dim, losing too much information for the decoder to work with.

---

## V15-V16: Entropy Hypothesis on ARC (IN PROGRESS — launched 2026-04-26)

### V15: V3 + L1 Sparsity (v15_l1_sparse) — GPU 2
- **Config**: V3 hybrid, embed=384, rule_dim=64, L1 weight=0.01, 19.0M params, batch=8
- **Progress** (epoch 33/300, 287s/epoch):

| Epoch | Loss | CE | L1_reg | Exact | Pixel | no_task | eval_new |
|-------|------|-----|--------|-------|-------|---------|----------|
| 10 | 0.9029 | 0.8927 | 0.0060 | 1.92% | 78.0% | 0.96% | 0.48% |
| 20 | 0.7551 | 0.7457 | 0.0048 | 2.64% | 80.1% | 1.68% | 0.72% |
| 30 | 0.6784 | 0.6691 | 0.0046 | 3.12% | 79.3% | 2.64% | 0.48% |

- **Best so far: 3.61%** at epoch 26. Estimated completion: ~21h remaining.

### V16: V4 Cross-Demo Attention (v16_crossdemo) — GPU 3
- **Config**: V4 arch (cross-demo encoder, 8 rich rule tokens, no 64-dim compression), embed=384, 18.9M params, L1=0.01, batch=8
- **Progress** (epoch 60/300, 160s/epoch):

| Epoch | Loss | CE | L1_reg | Exact | Pixel | no_task | eval_new |
|-------|------|-----|--------|-------|-------|---------|----------|
| 10 | 0.8714 | 0.8672 | 0.0042 | 1.44% | 77.9% | 1.44% | 0.48% |
| 20 | 0.7437 | 0.7412 | 0.0025 | 2.16% | 77.7% | 1.68% | 0.48% |
| 30 | 0.6849 | 0.6831 | 0.0018 | 3.12% | 79.4% | 1.44% | 0.72% |
| 40 | 0.6690 | 0.6676 | 0.0015 | 3.12% | 81.1% | 1.44% | 0.72% |
| 50 | 0.6280 | 0.6267 | 0.0013 | 3.12% | 80.3% | 0.96% | 0.24% |
| 60 | 0.6044 | 0.6033 | 0.0011 | 3.12% | 80.5% | 1.92% | 0.24% |

- **Best so far: 3.85%** at epoch 42/59. Estimated completion: ~11h remaining.

---

## Proxy Task Benchmarks (COMPLETED 2026-04-26)

### Multi-Task Benchmark (5 task types, 92 train / 33 novel rules)
Tasks: Visual Analogy, Pattern Completion, Sequence Extrapolation, Spatial Reasoning, Color Mapping.
Models: CNN encoder + UNet/transformer decoder, embed=128, 50 epochs, batch=64.

| Method | Params | Seen Exact | Novel Exact | Best Novel |
|--------|--------|-----------|-------------|------------|
| bottleneck | 1,506,118 | 23.4% | 0.5% | 0.6% |
| bottleneck_sparse | 1,506,118 | 17.7% | 0.5% | 0.5% |
| **taskid** | 686,598 | **52.8%** | **0.8%** | **0.9%** |
| hybrid | 1,588,294 | 51.0% | 0.5% | 0.5% |

Per-task novel accuracy (all methods ~0% except pattern_completion 1.7-2.7%).

**Conclusion**: TaskID memorization dominates on seen tasks (52.8% vs 23.4%). All methods fail on novel tasks (~0%). Short training (50ep) + small models (1.5M) insufficient for generalization.

### Focused Visual Analogy Benchmark (23 train / 7 novel transforms, IN PROGRESS)
Larger models (embed=256, 6M params), longer training (150 epochs), 500 samples/rule.
Only bottleneck method completed so far (1/5 methods):

| Method | Params | Seen Exact | Novel (no_tid) | Status |
|--------|--------|-----------|----------------|--------|
| bottleneck | 5,994,310 | 8.9% | 0.0% | ep120/150 |
| bottleneck_l1 | — | — | — | Queued |
| taskid | — | — | — | Queued |
| varc_style | — | — | — | Queued |
| hybrid | — | — | — | Queued |

Bottleneck learning very slowly: loss 1.64→1.03 over 120 epochs, seen accuracy only 8.9%, novel stuck at 0%.

---

## Entropy Hypothesis Validation (prior session)

Tested on MNIST and CIFAR-10 proxy tasks. 14 regularization configs on MNIST, 8 on CIFAR-10.

**Key validated findings**:
- Total entropy vs generalization: r = **-0.4858** (lower entropy → better generalization)
- L2 energy vs generalization: r = **-0.5407** (lower energy → better generalization)
- Best MNIST: l1_weak (new=74.5%, H=-123.5) vs baseline (new=74.2%, H=-75.9)
- Best CIFAR: l1_medium (new=57.3%, H=-253.8) vs baseline (new=55.8%, H=-86.1)
- entropy_bonus_strong: worst generalization (anti-hypothesis confirmed)
- Optimal compression: "just enough to encode the rule" — l1_weak > l1_strong

**Architecture comparison** (11 models on MNIST/CIFAR):
- v3_sparse (L1=0.01) consistently best but only +1pp over baseline
- VQ-VAE unstable, sparse composition fails on complex data
- Progressive multi-step doesn't help generalization

---

## Key Insights (updated 2026-04-26)

1. **VARC memorization is extremely efficient**: 26.68% vs our 5.77% at same param budget. Task_id lookup + deep bidirectional self-attention beats encoder-decoder by 4.6×.
2. **Our encoder is the bottleneck**: 12.3M params (65%) produce a 64-dim vector that loses too much information. VARC puts all 19M into decoding.
3. **Generalization gap is real but tiny**: eval_new 0.95% vs 0.48% — our hybrid can solve 2× more unseen tasks, but both are near zero.
4. **Entropy hypothesis validated but insufficient**: L1 sparsity gives +1pp on proxy tasks, not the +5pp needed for ARC.
5. **Architecture matters less than information flow**: VARC's task token sits IN the pixel sequence (bidirectional), ours cross-attends (unidirectional). This is the key structural disadvantage.
6. **Proxy task benchmarks show same pattern**: TaskID always dominates on seen tasks. All methods fail on novel tasks with short training.

## Architecture Analysis: Why VARC Wins

| Aspect | VARC | Our Hybrid |
|--------|------|-----------|
| Encoder params | 0 | 12.3M (65%) |
| Decoder params | 18.9M (100%) | 6.7M (35%) |
| Rule information | 384-dim task token | 64-dim compressed vector |
| Attention type | Bidirectional (task ↔ pixels) | Unidirectional (pixels → rule) |
| Decoder depth | 8 layers self-attn | 2 layers × 6 iterations |
| Position encoding | RoPE (2D-aware) | Learned nn.Parameter |

## VARC pretrained exact accuracy = 10.10% (42/416)
This is the single-view pass@1 number on training set test examples, measured with VARC's pretrained checkpoint.
Our retrained VARC baseline achieved 26.68% — better than the pretrained checkpoint due to RE-ARC augmentation.
