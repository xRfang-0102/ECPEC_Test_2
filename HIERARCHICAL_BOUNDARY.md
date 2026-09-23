# Hierarchical Adaptive Boundary (V4)

`hierarchical_boundary` is a new mode. The four existing modes and their configs
remain unchanged. V4 does not modify feature extraction, the encoder, Pair MLP,
full N x N construction, embeddings, auxiliary heads, or raw Pair BCE.

## Computation

From detached contextual features H and their valid-utterance masked mean h_D:

```text
c = learnable scalar global_center, initialized to 0 in logit space
o_d = dialogue_scale * tanh(MLP_dialogue(detach(h_D)))
q_di = row_scale * tanh(MLP_row([detach(h_i); detach(h_D)]))
r_di = q_di - mean_valid_rows(q_d)
b_di = c + o_d + r_di
relative_pair_logits_dij = raw_pair_logits_dij - b_di
prediction = relative_pair_logits > 0
```

Both heads have one hidden layer of 128 units with GELU, no Dropout, and a
zero-initialized final Linear weight/bias. Default scales are 0.5. Padding rows
are excluded from pooling/centering and explicitly zeroed. Initial o and r are
exactly zero. After subtracting the mean, r may exceed row_scale in magnitude;
its mathematical bound is twice row_scale. No second tanh/clamp is applied,
because that would break the zero-mean constraint.

The model keeps `pair_logits` as raw scores and exposes `global_center`,
`dialogue_offset`, `row_residual_raw` (pre-tanh), `row_residual_bounded`,
`row_residual_zero_mean`, `dynamic_boundary`, and `relative_pair_logits`.
`null_logits` aliases the final boundary for existing ranking/diagnostic code.

## Objective and isolation

The existing v2/v3 ranking implementation is unchanged: detached raw scores,
margin=0.2, row-wise highest-scoring top-3 valid negatives, balanced positive and
negative group means, then an average over valid rows. Negative-only rows remain
included. This preserves the prior row-weighting convention rather than adding
a new dialogue-weighting scheme.

```text
L = raw Pair BCE(pos_weight=2.5) + 0.2*Emotion BCE + 0.4*Cause BCE
  + 0.2*NULL ranking loss
  + 0.005*mean_valid_dialogues(dialogue_offset**2)
  + 0.01*mean_valid_rows(row_residual_zero_mean**2)
```

Boundary inputs and ranking scores are detached. AdamW, lr=1e-4 and base/head
weight_decay=0.01 are retained. As in v3, the global scalar is exempt from weight
decay to avoid an implicit zero anchor. No explicit center anchor is used.
Boundary parameters are gradient-clipped separately from the base; head
initialization preserves the baseline RNG stream.

## Evaluation and artifacts

Emotion/Cause use 0.5. V4 ignores pair_threshold and performs no threshold sweep
or top-k final decoding. Raw and Relative AUPRC use all valid Dev pairs globally.
Epoch diagnostics include center, offsets, dialogue boundary probabilities,
pre-tanh/bounded/zero-mean residuals, final boundaries, positive/no-positive row
groups, per-dialogue residual-mean error, and absolute magnitudes for all three
boundary components. Both regularizers and ranking loss are recorded.
The same diagnostics and AUPRCs are saved in checkpoint metadata and history.

The config specifies 20 epochs and patience=100; V4 bypasses early stopping.
Best selection uses Dev Relative Pair-F1. The final training report includes
best epoch, P/R/F1, Raw AUPRC and Relative AUPRC. No full training was performed
during implementation, so no new experiment performance is claimed.

```text
config/config_feature_pos2.5_hierarchical_boundary.yaml
checkpoints/base_feature_pos2.5_hierarchical_boundary_best.pt
checkpoints/training_history_feature_hierarchical_boundary.json
checkpoints/run_config_feature_hierarchical_boundary.yaml
logs/feature_pos2.5_hierarchical_boundary/train_feature_pos2.5_hierarchical_boundary.log
```

```powershell
python .\train_feature.py --config .\config\config_feature_pos2.5_hierarchical_boundary.yaml
```

## Verification

`python test_hierarchical_boundary.py` checks real-batch shapes, finite outputs,
initialization, nonzero zero-mean behavior, padding/empty-sample exclusion,
actual extra-loss backward isolation, two in-memory baseline-vs-V4 optimizer
steps, full Dev global AUPRC aggregation (13,301 pairs / 838 positives), logging,
config and temporary checkpoint metadata. Test checkpoint files are removed.

Observed nonzero-residual maximum absolute per-dialogue mean: 1.490116119e-08.
Initial extra-loss gradient L1: center 0.04634210, dialogue head 0.67083542,
row head 0.02645458; encoder/Pair/context/raw gradients absent. At zero initialization,
head gradients initially reach the final layers; preceding layers start receiving
gradients after the final weights change.

A temporary pre-change source snapshot additionally verified exact initialization,
RNG, train/eval outputs, losses and gradients for all four prior modes. AST checks
confirmed the encoder, auxiliary heads, Pair classifier and ranking-loss function
are unchanged. CPU two-step training retained bitwise-equal base parameters;
this does not guarantee bitwise reproducibility on every CUDA setup.

Parameters: global=1, dialogue head=33,025, row head=65,793;
hierarchical module=98,819; total/trainable=2,302,598.
