# Centered NULL-reference v3

The new `centered_null_reference` mode retains raw Pair BCE and the v2 ranking
loss. Fixed, adaptive_reference v1 and null_reference v2 remain available.

## Boundary and loss

The original encoder and Pair classifier produce H [B,N,D] and s [B,N,N].
The new branch receives only detached H and its detached masked dialogue mean:

```text
[detach(h_i); detach(h_D)]
  -> Linear(2D,128) -> GELU -> Linear(128,1)
  -> residual_raw
delta_i = 0.5 * tanh(residual_raw_i)
b_i = null_global_center + delta_i
relative_pair_logits_ij = s_ij - b_i
prediction = relative_pair_logits_ij > 0
```

`null_global_center` is a scalar nn.Parameter in logit space, initialized to 0.0.
The residual head has no Dropout and its final weight/bias are zero-initialized.
Thus initial residuals are exactly zero. Initialization preserves the baseline
RNG stream. Residual hidden size, scale and initial center are configurable.
The output dictionary retains existing logits and adds scalar `null_global_center`,
`null_residual` [B,N], `null_logits` [B,N], and `relative_pair_logits` [B,N,N].

Ranking supervision is the unchanged v2 implementation: mean positive softplus
loss and mean top-k negative softplus loss, equally averaged when both groups
exist, then averaged over valid rows. Negative-only rows participate. Default
margin=0.2 and k=3; raw ranking scores are detached. Padding is excluded.

```text
L_residual = mean_valid_utterances(delta_i ** 2)
L_total = BCE_raw_pair(pos_weight=2.5)
        + 0.2 * BCE_emotion + 0.4 * BCE_cause
        + 0.2 * L_null_rank + 0.01 * L_residual
```

Residual regularization does not involve the center. AdamW remains unchanged
for the base and residual head (lr=1e-4, weight_decay=0.01); only the new scalar
center is exempt from weight decay to avoid an implicit anchor towards zero.
Base gradients and boundary gradients are clipped separately, as in the v2
controlled experiment, preventing the boundary gradient norm from changing
base updates. No new loss can backpropagate into the encoder or Pair scorer.

## Diagnostics and experiment duration

Every Dev epoch reports global Raw/Relative Pair AUPRC, center logit/probability,
residual and final boundary statistics, positive-row/no-positive-row boundary
statistics, residual absolute mean/std, and ranking/regularization losses.
All boundary statistics exclude padding. Positive-row grouping uses valid gold
positive pairs only and is diagnostic, not an inference-time filter.
Existing v2 variation/uniqueness/correlation diagnostics remain available.
Checkpoint metadata and history retain these statistics and both AUPRCs.

The new config has epochs=20 and patience=100. Centered mode bypasses early
stopping; all completed epoch entries are saved to history. Best checkpoint
selection remains Dev Relative Pair-F1. Emotion/Cause thresholds remain 0.5;
Pair decisions ignore pair_threshold entirely in this mode.

## Artifacts and command

```text
config/config_feature_pos2.5_centered_null_reference.yaml
checkpoints/base_feature_pos2.5_centered_null_reference_best.pt
checkpoints/training_history_feature_centered_null_reference.json
checkpoints/run_config_feature_centered_null_reference.yaml
logs/feature_pos2.5_centered_null_reference/train_feature_pos2.5_centered_null_reference.log
```

```powershell
python .\train_feature.py --config .\config\config_feature_pos2.5_centered_null_reference.yaml
```

This starts a new experiment from initialization using the existing offline
features; it does not load a prior checkpoint. Existing configs/results are
untouched. Parameters: center=1, residual head=65,793, centered module=65,794,
total/trainable=2,269,573 (one parameter more than v2).

## Verification

`python test_centered_null_reference.py` runs real-batch shape/finite/zero-init,
ranking and residual loss, padding exclusion, actual isolated backward, two
in-memory optimizer steps, global AUPRC aggregation, diagnostics and temporary
checkpoint metadata checks. It does not run a full experiment.

During implementation, a temporary pre-change source snapshot additionally
verified all three prior modes: identical state initialization/RNG, train and
eval outputs, losses, and gradients. Centered-vs-fixed two-step CPU comparisons
also retained bitwise-equal base parameters. Initial rank+regularization backward
gave nonzero center and residual-head gradients and no base/context/raw gradients.
At zero initialization, the head's first layer initially receives zero gradients
because its final layer is zero; the final layer is trainable immediately.

Both production files pass py_compile. These tests do not establish an improved
Dev F1 or guarantee bitwise trajectory equality across every CUDA configuration.
