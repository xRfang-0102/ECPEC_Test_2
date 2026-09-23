# NULL-reference v2 controlled experiment

Three modes remain available: `fixed`, `adaptive_reference` (v1), and
`null_reference` (v2). Existing baseline and v1 configs/artifacts are unchanged.

## Data flow and loss

The original frozen-feature dialogue encoder and full N x N Pair classifier
produce contextual features H [B,N,D] and raw scores s [B,N,N]. NULL v2 feeds
detached H and its detached masked dialogue mean into
Linear(2D,128) -> GELU -> Linear(128,1), without Dropout. The resulting n [B,N]
is shared across all causes j for emotion row i. The output dictionary retains
`pair_logits` and adds `null_logits` and `relative_pair_logits = s - n[...,None]`.
Padding outputs are zeroed; masks exclude them from losses and metrics.

Pair BCE still uses raw s, with pos_weight=2.5. For each valid row:

- Positive term P_i = mean_pos softplus(0.2 + n_i - stopgrad(s_pos)).
- Negative term Q_i = mean_hard_neg softplus(0.2 + stopgrad(s_neg) - n_i).
- Hard negatives are the highest raw-scoring min(3, number of valid negatives)
  in that row, selected only from label==0 and valid pair positions.
- If both groups exist, row loss is (P_i+Q_i)/2. If only one exists, use that
  term alone. Negative-only rows are included. Rows without valid candidates
  are excluded. Average row losses equally across valid rows in the batch.

Total loss = raw Pair BCE + 0.2 Emotion BCE + 0.4 Cause BCE + 0.2 NULL rank loss.
The rank weight, margin and hard-negative k are configurable. NULL input and
ranking scores are detached, so ranking gradients reach only the NULL head.
The NULL head uses standard PyTorch Linear initialization inside an RNG-preserving
context. Base and NULL gradients are clipped separately with the existing
max_grad_norm; a shared norm would otherwise couple NULL learning to base updates.

Prediction is strictly `relative_pair_logits > 0`, independent of pair_threshold.
Emotion/Cause remain at 0.5. Fixed uses pair_threshold (legacy threshold remains
supported); v1 retains its adjusted-logit BCE and zero-centered prediction.

## Diagnostics and selection

Dev Raw and Relative Pair AUPRC use all valid Dev pairs globally, with the existing
average-precision definition. `pair_auprc` aliases Relative AUPRC in NULL mode.
Best checkpoint selection uses Dev Relative Pair-F1. Both AUPRCs and NULL stats
are included in checkpoint metadata and epoch history.

NULL diagnostics include logit and sigmoid-boundary mean/population std/min/max;
std across all valid utterances; std of dialogue means; mean within-dialogue std;
exact and four-decimal approximate unique counts; and Pearson correlation of NULL
with valid-row max/mean raw score. Undefined correlations are saved as JSON null.

Parameters: base 2,203,779; NULL head 65,793; total 2,269,572.

## Run and artifacts

```powershell
python .\train_feature.py --config .\config\config_feature_pos2.5_null_reference.yaml
```

This starts a new experiment from initialization using existing frozen features.
It does not load the baseline/v1 checkpoint. The config preserves seed=42,
epochs=20, batch_size=4, module_lr=1e-4, weight_decay=0.01, pair_pos_weight=2.5,
auxiliary weights 0.2/0.4, max_grad_norm=1.0 and patience=4.

```text
checkpoints/base_feature_pos2.5_null_reference_best.pt
checkpoints/training_history_feature_null_reference.json
checkpoints/run_config_feature_null_reference.yaml
logs/feature_pos2.5_null_reference/train_feature_pos2.5_null_reference.log
```

Mode-specific suffixes also protect artifacts if an old config is switched to
NULL mode without changing its filenames.

## Verification

`python test_null_reference.py` verifies real-batch shapes/finite outputs,
padding exclusion, explicit hard-negative formulas and row weighting, true
rank-only backward isolation, exact fixed eval compatibility, two in-memory
training steps with bitwise-identical base parameters, full Dev global dual
AUPRC (13,301 pairs / 838 positives), logging and checkpoint/history metadata.
Test checkpoint files are unique, temporary, and removed after verification.

The existing `test_adaptive_reference.py` also passes. Both changed production
files pass py_compile. No full training was run; these checks do not establish
a new Dev F1 result. Trajectory equality was verified on CPU for two steps, not
claimed as a guarantee of bitwise reproducibility on every CUDA configuration.
