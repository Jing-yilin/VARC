# Physics Priors Experiments

Small local experiments for ARC/VARC ideas inspired by conservation, entropy,
symmetry selection, and energy-based reranking.

## Environment

Use `uv`:

```bash
uv sync
```

Apple Silicon is supported through PyTorch MPS. This branch also patches the
project device selection to prefer `mps` when CUDA is unavailable.

## Experiment 1: Physics-Inspired Candidate Reranking

Script:

```bash
uv run python experiments/physics_priors/physics_reranker.py \
  --json-out .tmp/physics_priors/reranker_full.json
```

This does not train a model. It creates a candidate pool for ARC training-set
test examples and ranks candidates by an energy based on relation consistency:
shape change, color histogram change, entropy change, object/component count,
foreground fraction, and input/output pixel relation.

Latest local result on ARC-AGI `training` split test examples:

```text
examples: 416
mean candidate pool size: 15.27
relation-energy oracle top-1: 63.70%
relation-energy oracle top-2: 81.97%
complexity-only top-1: 0.24%
non-oracle candidate contains truth: 6.97%
top-1 when non-oracle contains truth: 65.52%
```

Interpretation: pure low entropy/complexity is not enough. The useful signal is
low entropy under the demo-induced relation, closer to an energy function than
a standalone simplicity score.

## Experiment 2: Selective Symmetry Gate

Script:

```bash
uv run python experiments/physics_priors/synthetic_symmetry_gate.py \
  --train-size 50000 \
  --test-size 10000 \
  --epochs 30 \
  --batch-size 512 \
  --grid-size 16 \
  --colors 6 \
  --json-out .tmp/physics_priors/symmetry_big.json
```

The synthetic task gives one demonstration pair and one query input. The hidden
rule is one of:

```text
identity, rot90, rot180, rot270, flipud, fliplr
```

Latest local result on Apple M5 / MPS:

```text
selective gate exact: 100.00%
selective gate transform accuracy: 100.00%
CNN exact after 30 epochs / 50k episodes: 2.62%
CNN pixel accuracy after 30 epochs / 50k episodes: 92.15%
CNN elapsed time: 294.01s
```

Interpretation: the CNN learns strong local/background statistics but mostly
does not learn the algorithm "infer the active symmetry, then apply it" from
this amount of data. An explicit selective-equivariance module solves the task
without training.

## VARC MPS Smoke Test

Command:

```bash
uv run python offline_train_ARC.py \
  --epochs 3 \
  --depth 4 \
  --batch-size 8 \
  --image-size 64 \
  --patch-size 2 \
  --learning-rate 3e-4 \
  --weight-decay 0 \
  --embed-dim 128 \
  --num-heads 4 \
  --mlp-dim 256 \
  --num-colors 12 \
  --data-root raw_data/ARC-AGI \
  --train-split training \
  --eval-split training \
  --eval-subset test \
  --architecture vit \
  --no-compile \
  --save-path .tmp/physics_priors/medium_vit_mps_final.pt \
  --best-save-path .tmp/physics_priors/medium_vit_mps_best.pt
```

Latest result:

```text
device: mps
sequence length: 1024
parameter count: 0.73M excluding task tokens
epoch 1: train_loss=1.4292, eval_loss=1.1675, eval_acc=0.0000
epoch 2: train_loss=1.2089, eval_loss=1.1409, eval_acc=0.0000
epoch 3: train_loss=1.1315, train_acc=0.0008, eval_loss=1.0805, eval_acc=0.0000
```

Interpretation: 64x64 canvas attention training is viable on local MPS, but a
few minutes of training without RE-ARC and without TTT is only a throughput and
compatibility baseline, not an ARC-solving result.

## Experiment 3: VARC Prediction Energy Reranking

Script:

```bash
ROOT=.tmp/hf_predictions/unzipped/VARC_predictions/ARC-1_ViT
uv run python experiments/physics_priors/rerank_varc_predictions.py \
  --workers 1 \
  --data-root raw_data/ARC-AGI \
  --split evaluation \
  --output-root "$ROOT/attempt_0,$ROOT/attempt_1,$ROOT/attempt_2,$ROOT/attempt_3" \
  --json-out .tmp/physics_priors/rerank_arc1_vit.json
```

This reads real VARC multi-view prediction JSON files and compares:

- majority vote
- energy-only ranking
- hybrid ranking: `energy - vote_weight * log(1 + votes)`

Important implementation detail: real VARC prediction files can contain invalid
or malformed candidates, including leaked background/border tokens or empty
outputs. The energy function treats these as high-energy candidates instead of
crashing.

Local smoke tests completed:

```text
ARC-1 ViT, first 40 evaluation tasks, 4 TTT attempts merged
majority pass@1: 52.5%
majority pass@2: 55.0%
best hybrid in tested weights: same pass@1/pass@2 on this subset
```

The full ARC-1/ARC-2 prediction reranking run is better suited for a remote GPU
or CPU box because the current implementation is JSON/NumPy heavy and runs
mostly on CPU.

## Next Structural Direction

The strongest next step is not a larger vanilla ViT. It is a hybrid:

1. Generate or receive candidate outputs from VARC views.
2. Add a selective-symmetry/object-transport module that infers which natural
   transformation family is active for the current task.
3. Rerank candidates with demo-conditioned energy rather than majority vote
   alone.
