# Context-Adaptive Pair Boundary

Run from the project root in PowerShell:

```powershell
python .\train_feature.py --config .\config\config_feature_pos2.5_adaptive_reference.yaml
```

This starts a new experiment with the existing frozen features; it does not load
a baseline checkpoint. `module_lr=1e-4`, `pair_pos_weight=2.5`, and all other
training/model hyperparameters are copied from the confirmed baseline config.

The model defaults to `model.pair_decision_mode: fixed` when omitted. Fixed mode
has the original state_dict, architecture, raw logits, loss and Pair prediction
comparison (`sigmoid(logits) >= threshold`). Emotion and Cause now independently
use 0.5 as requested. Pair threshold resolves from `evaluation.pair_threshold`,
then the legacy `evaluation.threshold`, then 0.66.

Adaptive mode keeps raw `pair_logits` and additionally returns:

- `adaptive_reference`: `[B,N]`, from masked mean dialogue pooling and an
  MLP of `[h_i; h_D]`: Linear(512,128), GELU, Dropout(0.1), Linear(128,1).
- `adjusted_pair_logits`: `[B,N,N]`, `pair_logits - reference.unsqueeze(-1)`.
  Invalid pairs are zeroed and excluded from both loss and metrics.

The final reference layer starts at zero. The reference is shared over all
cause candidates in each emotion row. Pair loss is masked BCEWithLogits on
adjusted logits, and prediction is strictly `adjusted_pair_logits > 0`.
Auxiliary heads never filter candidates. Full N x N construction, distance and
speaker embeddings, and the original Pair MLP are unchanged.

Dev AUPRC uses globally concatenated valid logits (raw in fixed, adjusted in
adaptive), using the existing average-precision definition. Direct logits avoid
sigmoid rounding/saturation ties; old probability-based AUPRC may differ in such
numerical edge cases. Epoch logs retain all task P/R/F1, Pair predicted/gold
counts and Dev AUPRC, and add mode plus valid-utterance reference mean/population
std/min/max for Train and Dev.

Parameters: fixed 2,203,779; adaptive 2,269,572; added reference head 65,793.

Adaptive output locations:

```text
checkpoints/adaptive_reference/base_feature_pos2.5_best_adaptive_reference.pt
checkpoints/adaptive_reference/training_history_feature_adaptive_reference.json
checkpoints/adaptive_reference/run_config_feature_adaptive_reference.yaml
logs/base_feature_pos2.5_adaptive_reference/train_feature_pos2.5_adaptive_reference.log
```

Even when switching only the mode in an old config, the training entry point
adds `_adaptive_reference` to checkpoint, log, history and saved config names.
Original configs and existing artifacts were not edited. `baseline_reference/`
contains byte-for-byte snapshots of the two original source files for regression
comparison; these copies are not alternative training entry points.

Fast checks (no full training or checkpoint writes):

```powershell
python -m py_compile train_feature.py
python -m py_compile models/feature_base_model.py
python test_feature.py
python -m dataset.feature_collate
python models/feature_base_model.py
python test_adaptive_reference.py
```

The regression test reads real Dev features, verifies the 13,301 valid / 838
positive pair totals, checks exact fixed-mode initialization/logits/loss/gradient
and Pair-metric compatibility against the original code, and verifies adaptive
shapes, zero initialization, masked loss/gradients/metrics, one in-memory optimizer
step, row alignment, context dependence, strict zero prediction, global AUPRC
over unequal batches, logging and artifact isolation. This does not establish an
adaptive Dev F1 improvement; that requires the subsequent training experiment.
