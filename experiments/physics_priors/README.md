# Physics Priors Experiments

Small local experiments for ARC/VARC ideas inspired by conservation, entropy,
symmetry selection, and energy-based reranking.

## Environment

Use `uv` for the original local project environment:

```bash
uv sync
```

Apple Silicon is supported through PyTorch MPS. This branch also patches the
project device selection to prefer `mps` when CUDA is unavailable.

For CUDA experiments on the AutoDL/SeetaCloud host, use the native conda Python:

```bash
/root/miniconda3/bin/python
```

This is the environment used for the later RTX 4090 runs below.

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

## Experiment 9: Selective Object-Transport Mechanism

Scripts:

```bash
uv run python experiments/physics_priors/object_transport_synthetic.py \
  --train-size 50000 \
  --test-size 10000 \
  --grid-size 16 \
  --ood-grid-size 24 \
  --max-shift 3 \
  --distractors 4 \
  --epochs 50 \
  --models cnn,global_gate,selective_gate \
  --json-out .tmp/physics_priors/object_transport_e50.json

uv run python experiments/physics_priors/object_transport_arc_search.py \
  --data-root raw_data/ARC-AGI \
  --split evaluation \
  --max-shift 8 \
  --workers 16 \
  --json-out .tmp/physics_priors/object_transport_arc1_evaluation_component.json
```

The synthetic task moves or copies only one selected object/color while leaving
distractors fixed. This is harder than global shift because the model must infer
the selected object, target flow, and move/copy mode.

Remote CUDA result:

```text
Synthetic selective object transport, 50k train / 10k test / 10k OOD
Mechanism count: selected color x 48 shifts x move/copy

DemoConditionedCNN, 50 epochs:
test exact: 0.08%
test foreground color recall: 88.22%
OOD exact: 0.04%
OOD foreground color recall: 86.86%

GlobalShiftGate:
test exact: 0.00%
OOD exact: 0.00%

SelectiveTransportGate:
test exact: 99.82%
OOD exact: 99.91%
```

Interpretation: high foreground recall is still not enough for ARC-style exact
success. The mechanism-space gate nearly solves the synthetic task because it
selects among object-level laws before rendering pixels.

Real ARC-1 exact-rule search was much narrower:

```text
ARC-1 training, 400 tasks:
tasks with exact object-transport rules: 4
oracle: 1.00%
pass@1/pass@2 after relation-energy ordering: 0.75% / 0.75%

ARC-1 evaluation, 400 tasks:
tasks with exact object-transport rules: 0
oracle: 0.00%
```

Interpretation: object transport is a valid primitive, but by itself it is far
too narrow for ARC. It should be one mechanism in a broader bank, not the core
model.

## Experiment 10: Symbolic Bank as VARC Companion

Scripts:

```bash
uv run python experiments/physics_priors/symbolic_candidate_search.py \
  --data-root raw_data/ARC-AGI \
  --split evaluation \
  --output-root "$OFFICIAL_VARC_ROOTS" \
  --workers 16 \
  --json-out .tmp/physics_priors/symbolic_plus_varc_arc1_eval.json

uv run python experiments/physics_priors/symbolic_diverse_eval.py \
  --data-root raw_data/ARC-AGI \
  --split evaluation \
  --official-roots "$OFFICIAL_VARC_ROOTS" \
  --workers 16 \
  --json-out .tmp/physics_priors/symbolic_diverse_arc1_eval.json
```

This broader symbolic bank includes extraction/crop, component selectors, D4
transforms, tiling, and concatenation.

Remote full ARC-1 results:

```text
Symbolic bank only:
training broad oracle: 7.33%
evaluation broad oracle: 2.00%

Official ARC-1 VARC ensemble:
pass@1/pass@2: 55.125% / 60.50%
oracle: 73.75%

Symbolic bank + official ensemble:
combined oracle: 74.50%
best pass@1/pass@2: 55.875% / 61.00%

Uncertainty-gated symbolic second answer:
best strategy: official_majority
pass@1/pass@2: 55.125% / 60.50%
```

Interpretation: the symbolic bank adds a small amount of candidate coverage
but is not reliable enough to directly take the second pass@2 slot. Its best
current role is as an auxiliary candidate source and as synthetic supervision
for a learned mechanism controller.

## Experiment 11: Learned Symbolic Controller

Script:

```bash
uv run python experiments/physics_priors/symbolic_controller_train_eval.py \
  --official-roots "$OFFICIAL_VARC_ROOTS" \
  --epochs 120 \
  --confidence-epochs 300 \
  --workers 16 \
  --json-out .tmp/physics_priors/symbolic_controller_arc1_full.json
```

This trains only on ARC-1 `training` tasks:

1. Generate broad symbolic candidates.
2. Train a groupwise ranker to order symbolic candidates.
3. Train a confidence head on a held-out slice of training tasks.
4. Apply the frozen controller to ARC-1 `evaluation`, using VARC as the default.

Remote result:

```text
train/val/eval examples: 330 / 86 / 419
training positive rate for symbolic top selection: 6.67%
validation positive rate: 3.49%

Ranker validation:
oracle: 9.30%
top1/top2: 3.49% / 5.81%

ARC-1 evaluation symbolic controller:
oracle: 1.91%
top1/top2: 0.48% / 0.72%

Official VARC ensemble baseline:
pass@1/pass@2: 55.125% / 60.50%

Best controller-gated result:
official_majority remains best
pass@1/pass@2: 55.125% / 60.50%
```

Interpretation: a supervised controller trained on only ARC training labels
does not generalize enough to trust the symbolic bank on evaluation. The
failure mode is data scarcity and selector ambiguity: the symbolic bank often
contains plausible crops, but choosing the correct component/color requires a
stronger learned selector trained on more mechanism-labeled synthetic data.

## Experiment 12: Variable Selector Search

Script:

```bash
uv run python experiments/physics_priors/variable_selector_search.py \
  --data-root raw_data/ARC-AGI \
  --split evaluation \
  --max-demo-error 0.75 \
  --workers 8
```

This tries to infer rules such as "crop the color with max area", "crop the
leftmost bbox", or "crop the densest bbox", optionally followed by a D4
transform.

ARC-1 evaluation result:

```text
strict max_demo_error=0.00:
tasks with rules: 1
oracle/pass@2: 0.25% / 0.25%

soft max_demo_error=0.75:
tasks with rules: 118
oracle/pass@2: 0.50% / 0.50%
```

Interpretation: hand-written selector attributes do not scale. Relaxing the
demo fit increases rule count but mostly adds wrong candidates. This supports
the same conclusion as the learned controller: the next real model needs a
learned object/selector module, not more one-off selector predicates.

## Experiment 13: Synthetic-Trained Selector Controller

Script:

```bash
/root/miniconda3/bin/python experiments/physics_priors/selector_controller_synthetic.py \
  --train-size 8000 \
  --val-size 1000 \
  --test-size 1000 \
  --epochs 25 \
  --batch-size 512 \
  --hidden 128 \
  --grid-size 16 \
  --objects 5 \
  --max-obj 5 \
  --demos-per-task 3 \
  --workers 16 \
  --arc-split evaluation \
  --json-out .tmp/physics_priors/selector_controller_synth8k_arc_eval.json
```

This is the learned version of Experiment 12. The synthetic task family is:

```text
input scene -> crop selected object -> optional D4 transform
```

The hidden rule chooses one selector, such as largest object, leftmost object,
densest object, or sparsest object, and one D4 transform. The controller scores
112 candidate mechanisms from demo-consistency features, then applies the
selected mechanism to the query input. Training is batched over task groups so
the controller step uses the GPU instead of feeding one tiny task at a time.

Remote CUDA result with native conda Python:

```text
device: cuda
rules: 112
feature dim: 36
synthetic train/val/test: 8000 / 1000 / 1000

synthetic test:
oracle: 100.00%
top1: 98.80%
top2: 99.90%

ARC-1 evaluation zero-shot:
rule-bank oracle: 0.75%
controller top1/top2: 0.00% / 0.00%
direct pass@1/pass@2: 0.00% / 0.00%
```

Interpretation: the model learns mechanism selection almost perfectly on the
clean synthetic distribution, so the controller architecture is not the current
blocker. The failure is distribution and mechanism coverage. Real ARC tasks are
not mostly single-color rectangle crops under a selector-plus-D4 law; the
available rule bank only covers about 0.75% of ARC-1 evaluation examples. To
get generalization, the next model needs a broader mechanism space and
synthetic supervision sampled from ARC-like object statistics, including
multi-object composition, line/fill operations, color remapping, masks,
counting, tiling, and partial copying.

## Experiment 14: Paint-On-Canvas Symbolic Primitives

Script:

```bash
/Users/yilin/miniforge3/bin/python experiments/physics_priors/paint_symbolic_search.py \
  --data-root raw_data/ARC-AGI \
  --split evaluation \
  --workers 8 \
  --output-root "$OFFICIAL_VARC_ROOTS" \
  --max-demo-error 0.25 \
  --max-candidates 64 \
  --json-out .tmp/physics_priors/paint_symbolic_arc1_eval_err025.json
```

This adds conservative same-canvas physical primitives that the earlier broad
symbolic bank did not cover:

- aligned line drawing between same-color points
- bounding-box fill, border, and interior paint
- enclosed-background fill
- vertical/horizontal mirror completion
- gravity-style row/column compression

Exact-rule sanity check on ARC-1 training first 80 tasks:

```text
paint candidates: 3
paint oracle/pass@1/pass@2: 3.75% / 3.75% / 3.75%
matched examples: fill_enclosed_4, gravity_down, gravity_up
```

ARC-1 evaluation results:

```text
exact max_demo_error=0.00:
paint candidates: 1
paint oracle/pass@1/pass@2: 0.25% / 0.25% / 0.25%
combined with VARC: no pass@2 gain over VARC majority

soft max_demo_error=0.25:
paint candidates: 5229
paint oracle: 0.50%
paint-only pass@1/pass@2: 0.00% / 0.25%
VARC majority pass@1/pass@2: 55.125% / 60.50%
VARC first + paint second pass@1/pass@2: 55.125% / 58.75%
```

Interpretation: adding primitives without a reliable controller is actively
harmful. The exact paint rules are too sparse, while soft demo matching creates
many plausible-looking but wrong hypotheses. This reinforces the core
direction: the next gain needs a controller trained on ARC-statistic synthetic
tasks and a mechanism bank whose oracle coverage is meaningfully above a few
percent before it is allowed to compete for VARC's pass@2 slots.

## Experiment 15: VARC Prediction Meta-Ranker Diagnostic

Script:

```bash
/root/miniconda3/bin/python experiments/physics_priors/prediction_meta_ranker_cv.py \
  --data-root raw_data/ARC-AGI \
  --split evaluation \
  --output-roots "$OFFICIAL_VARC_ROOTS" \
  --vote-mode source_norm \
  --folds 5 \
  --epochs 120 \
  --hidden 64 \
  --batch-size 128 \
  --candidate-limit 128 \
  --eval-every 5 \
  --json-out .tmp/physics_priors/meta_ranker_cv_arc1_eval_source_norm_c128.json
```

This is a diagnostic, not a fair final benchmark. It uses ARC-1 evaluation
labels, but splits by task into five folds. The question is whether the true
candidate inside VARC's prediction pool is learnably identifiable from
relation-energy, vote strength, source diversity, and candidate statistics.

Remote CUDA result with native conda Python:

```text
candidate limit: 128
groups/tasks: 419 / 400

majority vote mode:
majority pass@1/pass@2: 55.125% / 60.50%
learned-only pass@1/pass@2: 55.125% / 61.25%
majority top1 + learned second pass@1/pass@2: 55.125% / 61.50%

source-normalized vote mode:
majority pass@1/pass@2: 55.125% / 60.50%
learned-only pass@1/pass@2: 55.625% / 61.375%
learned + source-normalized vote_weight=0.25: 55.50% / 61.75%
majority top1 + learned second pass@1/pass@2: 55.125% / 61.375%
```

Interpretation: candidate reranking has a real but small learnable signal. Even
with evaluation-label cross-validation, the gain is about +1.25 pass@2, not the
+5 target. This means the 73%+ oracle gap is not easily unlocked by shallow
candidate statistics alone. A fair version would require generating TTT
prediction pools on ARC training tasks, training this meta-ranker there, and
freezing it for evaluation; but the diagnostic suggests reranking alone is
unlikely to deliver the full target without better candidate generation.

## Experiment 16: Expanded TTT Symmetry Orbit

Motivation from rereading the VARC paper images: the strongest levers are not
post-hoc symbolic primitives, but canvas priors, task-token TTT, and multi-view
voting. The paper uses the original task plus five geometric transforms
(`Rotate(90/180/270)`, `Flip(0/1)`) with ten color versions each, for 51
task-token views. This experiment extends that orbit with `IdentityAugmenter`
color permutations and `Transpose()`, giving 71 task-token views per task.

Code:

```bash
/root/miniconda3/bin/python experiments/physics_priors/prepare_ttt_variant.py \
  --data-root raw_data/ARC-AGI \
  --split evaluation \
  --output-subdir eval_idtrans_color_ttt_9 \
  --variant identity_transpose_color_plus_paper \
  --num-permutations 9 \
  --clean
```

The evaluation path now reads augmentation metadata when undoing predictions,
so extra spatial transforms such as transpose do not depend on filename
heuristics.

Remote CUDA first20 result, native conda Python, ViT checkpoint, 20 TTT epochs,
10 inference attempts:

```text
official ensemble pass@1/pass@2: 55% / 65%
paper-style 51-view TTT:        45% / 50%
71-view TTT alone:              40% / 45%
official + 51-view merged:      55% / 65%
official + 71-view merged:      55% / 70%
```

Remote CUDA first50 result:

```text
official ensemble pass@1/pass@2: 62% / 66%
paper-style 51-view TTT:        48% / 52%
71-view TTT alone:              44% / 48%
official + 71-view merged:      60% / 68%
official + 51+71 merged:        60% / 68%
```

Targeted 30-attempt rerun on eight official-oracle-but-not-top2 tasks did not
improve first50:

```text
official + 71-view gap8@30-attempt merged: 60% / 68%
```

Interpretation: the expanded orbit is useful but not sufficient. It produced a
real first20 +5 pass@2 gain when merged with official predictions, and it found
additional correct candidates on some hard tasks. But on first50 the gain
shrinks to +2, and 30 inference attempts do not rescue the missed oracle
candidates. The bottleneck is now candidate selection/ranking under task
uncertainty, not simply more symmetry views. A more promising next step is to
save per-view metadata and train a controller/ranker on TTT prediction pools
generated from ARC training tasks, so second-answer selection can learn when a
low-vote symmetry candidate should replace the official second prediction.

## Experiment 17: Per-View TTT Metadata Probe

The expanded-orbit experiments showed that correct answers can be present but
buried below top2. We added optional metadata logging to TTT inference:

```bash
--save-prediction-metadata
```

This preserves the original `*_predictions.json` format and writes a parallel
`*_prediction_meta.json` with, for every view prediction:

- source augmented task name
- augmentation family and color permutation index
- scale factor and canvas offset
- predicted crop size and border-token detection
- prediction key after undoing transform/color
- mean/min softmax confidence, mean top1-top2 margin, and entropy

Probe script:

```bash
/root/miniconda3/bin/python experiments/physics_priors/meta_prediction_diagnostic.py \
  --tasks-file .tmp/physics_priors/arc1_first20_tasks.txt \
  --prediction-root outputs/physics_ttt_arc1_meta_first20_e20_attempt_0 \
  --json-out .tmp/physics_priors/meta_prediction_first20_e20.json
```

ARC-1 first20, 71-view TTT, 20 TTT epochs, 10 inference attempts:

```text
oracle: 70%

majority:        pass@1/pass@2 = 40% / 45%
support:         pass@1/pass@2 = 40% / 45%
stability:       pass@1/pass@2 = 40% / 45%
vote+confidence: pass@1/pass@2 = 40% / 45%
vote+margin:     pass@1/pass@2 = 40% / 45%
confidence only: pass@1/pass@2 = 35% / 35%
```

Detailed inspection:

```text
070dd51e: truth is majority rank 5 but confidence rank 1
09c534e7: truth is majority rank 6, support rank 4
0a2355a6: truth is majority rank 6 and not rescued by confidence/support
```

Interpretation: metadata contains some useful local signals, but they are not
globally monotonic. Confidence can rescue a low-candidate-count task like
`070dd51e`, while strongly hurting high-candidate-count tasks because the model
is often very confident about wrong predictions. The selector therefore needs
to be conditional on candidate-pool shape and task type. A single scalar
formula over vote count, confidence, and augmentation support is not enough.

## Multi-GPU Scaling Plan

The next useful compute upgrade is not larger model memory for one TTT run; it
is task-parallel throughput. ARC tasks are independent during TTT, so a
4x4090/8x4090 node can generate full training/evaluation candidate pools and
metadata several times faster than the current single-card loop.

Runner:

```bash
tmux new-session -d -s varc_parallel_meta \
  'cd /root/autodl-tmp/VARC && \
   CUDA_DEVICE_ORDER=PCI_BUS_ID /root/miniconda3/bin/python \
     experiments/physics_priors/parallel_ttt_runner.py \
     --tasks-file .tmp/physics_priors/arc1_training_first400_tasks.txt \
     --gpus 0 1 2 3 \
     --data-root raw_data/ARC-AGI \
     --augmented-split train_idtrans_color_ttt_9 \
     --checkpoint saves/offline_train_ViT/checkpoint_best.pt \
     --output-name physics_ttt_arc1_train_meta400_e20 \
     --epochs 20 \
     --batch-size 8 \
     --num-attempts 10 \
     --ttt-num-each 1 \
     --save-prediction-metadata'
```

Recommended host shape:

- minimum useful: 4x RTX 4090/A5000/A6000-class GPUs, 24 GB+ VRAM each
- better: 8x RTX 4090 or 4x A100/H100 for one-day iteration
- 16+ vCPU, 64 GB+ RAM, and 300 GB+ fast local SSD

Use the extra GPUs to build supervised selector data: 400 ARC training tasks
with 71-view TTT metadata, then evaluate the learned selector on 400 ARC
evaluation tasks. This targets the observed bottleneck: correct candidates are
often present below top2, but simple majority/confidence rules fail to promote
them reliably.

## Experiment 18: Metadata Ranker CV and Transfer

Before shutting down the single-4090 remote machine, we preserved two small
result JSON files under `experiments/physics_priors/results/`:

- `meta_ranker_cv_eval_first20_e20.json`
- `meta_ranker_train_meta50_eval_meta20.json`

Diagnostic task-level CV on ARC-1 evaluation first20:

```bash
/root/miniconda3/bin/python experiments/physics_priors/meta_prediction_ranker_cv.py \
  --split evaluation \
  --prediction-root outputs/physics_ttt_arc1_meta_first20_e20_attempt_0 \
  --limit 20 \
  --candidate-limit 256 \
  --folds 5 \
  --epochs 120 \
  --hidden 64 \
  --batch-size 128 \
  --cpu \
  --json-out .tmp/physics_priors/meta_ranker_cv_eval_first20_e20.json
```

Result:

```text
oracle:   70%
majority: pass@1/pass@2 = 40% / 45%
learned:  pass@1/pass@2 = 40% / 50%
```

Interpretation: per-view metadata contains learnable selector signal. This is
not a fair final benchmark because labels come from the evaluated split, but
folds are task-disjoint, so it is a useful signal test. The +5 pass@2
diagnostic gain supports building a stronger selector/verifier.

Transfer test from ARC training first50 metadata to ARC evaluation first20:

```bash
/root/miniconda3/bin/python experiments/physics_priors/meta_prediction_ranker_train_eval.py \
  --train-split training \
  --eval-split evaluation \
  --train-prediction-root outputs/physics_ttt_arc1_train_meta_first50_e20_attempt_0 \
  --eval-prediction-root outputs/physics_ttt_arc1_meta_first20_e20_attempt_0 \
  --train-limit 50 \
  --eval-limit 20 \
  --candidate-limit 256 \
  --epochs 160 \
  --hidden 64 \
  --batch-size 128 \
  --cpu \
  --json-out .tmp/physics_priors/meta_ranker_train_meta50_eval_meta20.json
```

Training metadata generation completed for 50/50 ARC training tasks. The
transfer result did not beat majority:

```text
train groups: 51, train tasks: 50
eval groups:  20, eval tasks:  20

eval oracle:   70%
eval majority: pass@1/pass@2 = 40% / 45%
eval learned:  pass@1/pass@2 = 40% / 45%
```

Conclusion: the selector signal exists, but the current shallow metadata MLP
does not generalize from 50 training tasks to evaluation tasks. The next
machine should be used to generate 400-task training/evaluation metadata pools
and train a global-attention verifier over `(demos, test input, candidate
output, TTT metadata)`, rather than only increasing the number of TTT views or
using scalar metadata formulas.
