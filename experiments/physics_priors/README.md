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

Use `--episode-generator torch-random` for GPU-resident random-grid generation
when scaling this experiment on CUDA. The original `structured` generator is
more ARC-like but Python-loop-bound during data generation.

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

Remote CUDA scaling result on RTX 4090 with `/root/miniconda3/bin/python`:

```text
generator: torch-random, CUDA-resident
train/test: 500000 / 100000
epochs: 200
elapsed: 768.57s
selective gate exact: 100.00%
selective gate transform accuracy: 100.00%
CNN best exact: 16.99% at epoch 39
CNN best pixel accuracy: 84.79% at epoch 112
CNN final exact: 14.18%
CNN final pixel accuracy: 84.05%
```

Interpretation: even when generation and training are GPU-resident and the CNN
gets much more data and compute, the explicit "infer the active symmetry, then
apply it" mechanism remains qualitatively better. Brute force learns useful
pixel statistics but still does not reliably learn the algorithm.

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

The default energy is relation-statistical. A second mode adds an explicit
symbolic prior:

```bash
--energy-mode relation_symbolic
```

This mode enumerates exact D4 image symmetries plus color maps from the
demonstrations. If the training pairs are perfectly explained by one of those
rules, candidates matching the induced rule on the test input receive a strong
energy bonus, while near misses are penalized by pixel mismatch.

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

Remote full official-prediction results with native conda Python:

```text
Paper Table 3 final results:
ARC-1 VARC / ensemble: 54.5% / 60.4%
ARC-2 VARC / ensemble: 8.3% / 11.1%

ARC-1 ensemble, ARC-AGI evaluation, 400 tasks
majority pass@1/pass@2: 55.125% / 60.50%
best hybrid pass@1/pass@2: 55.875% / 61.00% at vote_weight=1.75
oracle: 73.75%

ARC-2 ViT, ARC-AGI-2 evaluation, 120 tasks
majority pass@1/pass@2: 8.61% / 10.28%
best hybrid pass@1/pass@2: 8.61% / 10.28% at vote_weight=5.0
oracle: 13.06%

ARC-2 ensemble, ARC-AGI-2 evaluation, 120 tasks
majority pass@1/pass@2: 9.44% / 11.11%
best hybrid pass@1/pass@2: 9.44% / 11.11% at vote_weight=5.0
oracle: 15.42%
```

`relation_symbolic` produced the same metrics on these official prediction
sets. That means exact D4-plus-color-map rules are too narrow for the current
candidate pool; the next symbolic prior should handle object transport,
cropping, tiling, and partial-copy mechanisms rather than only global image
symmetries.

Additional pass@2 selection searches:

```text
ARC-1 ensemble:
best top-2 selection strategy: hybrid/log/w=1.75
pass@1/pass@2: 55.875% / 61.00%
anchored-second, top-k reranking, and gap-switching did not exceed 61.00%.
source-aware ViT/U-Net diversity reached 60.625%, below the hybrid result.

ARC-2 ensemble:
best top-2 selection strategy: majority
pass@1/pass@2: 9.44% / 11.11%
hybrid, anchored-second, top-k reranking, gap-switching, and source-aware
diversity did not exceed majority.
```

Interpretation: the post-processing result is slightly above the paper's ARC-1
ensemble number and essentially tied with the paper's rounded ARC-2 ensemble
number. The remaining gap to the oracle is not mostly a top-2 selection problem;
it requires generating better candidates or adding broader structural solvers.

## Next Structural Direction

The strongest next step is not a larger vanilla ViT. It is a hybrid:

1. Generate or receive candidate outputs from VARC views.
2. Add a selective-symmetry/object-transport module that infers which natural
   transformation family is active for the current task.
3. Rerank candidates with demo-conditioned energy rather than majority vote
   alone.

## Experiment 8: Global Manifold-Attention Prototype

Script:

```bash
uv run python experiments/physics_priors/manifold_attention_synthetic.py \
  --train-size 50000 \
  --test-size 10000 \
  --grid-size 16 \
  --ood-grid-size 24 \
  --max-shift 6 \
  --density 0.22 \
  --foreground-loss-weight 6 \
  --epochs 50 \
  --batch-size 256 \
  --models cnn,manifold \
  --json-out .tmp/physics_priors/manifold_shift_weighted_e50.json
```

This is a synthetic one-demo spatial-flow task. A hidden `(dy, dx)` shift maps
`demo_in` to `demo_out`; the model must infer that shift and apply it to
`query_in`. It tests a minimal version of the deeper ARC idea: infer a
low-entropy mechanism, then apply it over the spatial manifold.

Models:

- `DemoConditionedCNN`: local convolution over `[demo_in, demo_out, query_in]`.
- `ManifoldRuleTransformer`: rule slots globally attend over demo pixels; each
  output coordinate attends to the inferred rule and all query pixels with a
  relative spatial bias.
- `ShiftTransportAttention`: attention over an explicit spatial-flow basis. It
  scores all allowed shifts from the demo pair and applies the selected flow
  exactly to the query grid.

Remote CUDA results on RTX 4090:

```text
Task: one-demo spatial shift
Shift basis: 168 candidate flows, max shift 6
Train/test: 50k / 10k for learned CNN and rule-slot transformer
OOD: 24x24 grids, same shift basis

DemoConditionedCNN, weighted foreground loss, 50 epochs:
test exact: 0.05%
test foreground color recall: 25.72%
OOD exact: 0.00%
OOD foreground color recall: 25.15%

ManifoldRuleTransformer, weighted foreground loss, partial run:
epoch 10 test exact: 0.00%
epoch 10 test foreground color recall: 2.35%
epoch 10 OOD foreground color recall: 1.69%

ShiftTransportAttention, zero training / epoch 0:
parameters: 7
test exact: 100.00%
OOD exact: 100.00%
elapsed: 12.80s
```

Interpretation: global attention alone is not enough if the network still has
to invent the mechanism from pixels. The strongest signal so far is to expose a
small library of natural spatial flows and let the task select among them. This
is more general than the earlier exact-symmetry gate because it frames the
solver as attention over mechanism space, not a one-off hand-coded rule. The
next ARC-facing model should expand this basis from global shifts to object
transport, crop/paste, tiling, symmetry, color maps, and line/fill operators,
then use a learned controller to select and compose these low-entropy actions.
