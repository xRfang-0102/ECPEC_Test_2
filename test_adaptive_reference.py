"""Fast regression checks; no training run or checkpoint writes."""
import contextlib
import importlib.util
import io
from pathlib import Path

import torch
import yaml
from sklearn.metrics import average_precision_score

from dataset.feature_dataset import FeatureECFDataset
from dataset.feature_collate import FeatureECPECCollator
from models.feature_base_model import FeatureECPECBaseModel
from train_feature import (
    ECPECMetricAccumulator, adaptive_artifact_path, compute_losses,
    evaluate, forward_model, print_result,
)

ROOT = Path(__file__).resolve().parent


def original_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'baseline_reference' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_checks():
    torch.set_num_threads(2)
    original = original_module('original_model', 'feature_base_model.py')
    original_train = original_module('original_train', 'train_feature.py')
    data = FeatureECFDataset(ROOT / 'features/ECF/dev_roberta.pt', max_dialogue_length=40)
    assert data.num_candidate_pairs == 13301
    assert data.num_positive_pairs == 838
    batch = FeatureECPECCollator()([data[i] for i in range(4)])
    valid = batch['utterance_mask']
    mask = batch['pair_mask']
    assert torch.equal(mask, valid.unsqueeze(2) & valid.unsqueeze(1))
    assert (~mask).any()

    torch.manual_seed(42)
    old = original.FeatureECPECBaseModel()
    torch.manual_seed(42)
    fixed = FeatureECPECBaseModel()
    assert old.state_dict().keys() == fixed.state_dict().keys()
    for key, value in old.state_dict().items():
        assert torch.equal(value, fixed.state_dict()[key]), key
    fixed.load_state_dict(old.state_dict(), strict=True)
    assert fixed.adaptive_reference_parameters == 0
    old.train(); fixed.train()
    torch.manual_seed(123)
    before = forward_model(old, batch)
    torch.manual_seed(123)
    after = forward_model(fixed, batch)
    for key in before:
        assert torch.equal(before[key], after[key]), key
    args = dict(batch=batch, pair_pos_weight=2.5, lambda_emotion=0.2, lambda_cause=0.4)
    old_loss = original_train.compute_losses(before, **args)
    new_loss = compute_losses(after, **args)
    for key in old_loss:
        assert torch.equal(old_loss[key], new_loss[key]), key
    old_loss['total_loss'].backward(); new_loss['total_loss'].backward()
    for (key, p), (_, q) in zip(old.named_parameters(), fixed.named_parameters()):
        assert torch.equal(p.grad, q.grad), key
    old_metrics = original_train.ECPECMetricAccumulator(0.66)
    new_metrics = ECPECMetricAccumulator(0.66)
    old_metrics.update(before, batch); new_metrics.update(after, batch)
    assert old_metrics.compute()['pair'] == new_metrics.compute()['pair']
    assert new_metrics.emotion.threshold == new_metrics.cause.threshold == 0.5
    print('PASS fixed: identical state_dict, initialization, train logits, losses, gradients and Pair metrics')

    adaptive = FeatureECPECBaseModel(pair_decision_mode='adaptive_reference')
    missing = adaptive.load_state_dict(fixed.state_dict(), strict=False)
    assert not missing.unexpected_keys
    assert all(k.startswith('reference_head.') for k in missing.missing_keys)
    adaptive.eval(); fixed.eval()
    outputs = forward_model(adaptive, batch)
    raw = outputs['pair_logits']; adjusted = outputs['adjusted_pair_logits']
    reference = outputs['adaptive_reference']
    assert raw.shape == adjusted.shape == mask.shape
    assert reference.shape == valid.shape
    assert torch.count_nonzero(reference) == 0
    assert torch.equal(raw, adjusted)
    assert torch.equal(raw, forward_model(fixed, batch)['pair_logits'])
    assert all(torch.isfinite(t).all() for t in outputs.values())
    assert adaptive.adaptive_reference_parameters == 65793
    print(f'PASS shapes: raw/adjusted/mask={tuple(raw.shape)}, reference={tuple(reference.shape)}; finite and zero initialization')

    loss = compute_losses(outputs, **args)['pair_loss']
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        adjusted[mask], batch['pair_labels'][mask], pos_weight=torch.tensor(2.5),
    )
    assert torch.equal(loss, expected)
    adjusted.retain_grad()
    loss.backward()
    assert torch.count_nonzero(adjusted.grad[~mask]) == 0
    assert adaptive.reference_head[-1].weight.grad.abs().sum() > 0
    assert adaptive.reference_head[-1].bias.grad.abs().sum() > 0
    changed = dict(outputs)
    changed['adjusted_pair_logits'] = adjusted.detach().clone()
    changed['adjusted_pair_logits'][~mask] = float('nan')
    changed_batch = dict(batch)
    changed_batch['pair_labels'] = batch['pair_labels'].clone()
    changed_batch['pair_labels'][~mask] = float('nan')
    changed_args = dict(args, batch=changed_batch)
    assert torch.equal(loss.detach(), compute_losses(changed, **changed_args)['pair_loss'])
    normal_metrics = ECPECMetricAccumulator(0.66, 'adaptive_reference')
    padded_metrics = ECPECMetricAccumulator(0.99, 'adaptive_reference')
    normal_metrics.update(outputs, batch); padded_metrics.update(changed, changed_batch)
    assert normal_metrics.compute() == padded_metrics.compute()
    print('PASS adjusted BCE, trainable reference gradients, padding excluded from loss/gradient/metrics')

    # After an update, verify the broadcast direction and the masked dialogue mean.
    optimizer = torch.optim.AdamW(adaptive.parameters(), lr=1e-4)
    optimizer.step(); optimizer.zero_grad()
    outputs = adaptive(**{k: batch[k] for k in (
        'utterance_features', 'speaker_ids', 'position_ids', 'utterance_mask', 'pair_mask'
    )}, return_features=True)
    h = outputs['context_features']
    pooled = torch.stack([h[b][valid[b]].mean(0) for b in range(h.shape[0])])
    expected_reference = adaptive.reference_head(torch.cat([
        h, pooled[:, None].expand_as(h)
    ], -1)).squeeze(-1).masked_fill(~valid, 0)
    torch.testing.assert_close(outputs['adaptive_reference'], expected_reference)
    difference = outputs['pair_logits'] - outputs['adjusted_pair_logits']
    torch.testing.assert_close(difference[mask], outputs['adaptive_reference'].unsqueeze(-1).expand_as(difference)[mask])
    assert torch.count_nonzero(outputs['adjusted_pair_logits'][~mask]) == 0
    assert all(torch.isfinite(t).all() for t in outputs.values())
    assert outputs['adaptive_reference'][valid].std() > 0
    print('PASS one optimizer step: context-dependent reference, masked mean pooling and emotion-row broadcast')

    # Deliberately pass 0.99: adaptive prediction must still use strictly > 0.
    tiny_batch = {
        'utterance_mask': torch.ones(1, 3, dtype=torch.bool),
        'pair_mask': torch.ones(1, 3, 3, dtype=torch.bool),
        'emotion_labels': torch.zeros(1, 3), 'cause_labels': torch.zeros(1, 3),
        'pair_labels': torch.zeros(1, 3, 3),
    }
    tiny_out = {
        'emotion_logits': torch.full((1, 3), 0.1),
        'cause_logits': torch.full((1, 3), 0.1),
        'pair_logits': torch.full((1, 3, 3), 100.0),
        'adjusted_pair_logits': torch.tensor([[[-1e-8, 0., 1e-8]]]).expand(1, 3, 3),
        'adaptive_reference': torch.zeros(1, 3),
    }
    metric = ECPECMetricAccumulator(0.99, 'adaptive_reference')
    metric.update(tiny_out, tiny_batch)
    assert metric.compute()['pair']['predicted_positive'] == 3
    assert metric.compute()['emotion']['predicted_positive'] == 3
    assert metric.compute()['cause']['predicted_positive'] == 3
    print('PASS strict zero boundary and independent auxiliary thresholds')

    # Unequal batches ensure AUPRC is calculated globally, including all valid pairs.
    batches = [FeatureECPECCollator()([data[0]]), FeatureECPECCollator()([data[i] for i in (1, 2, 3)])]
    for model in (fixed, adaptive):
        result = evaluate(model, batches, torch.device('cpu'), 2.5, 0.2, 0.4, 0.66)
        scores, labels, refs = [], [], []
        with torch.no_grad():
            for item in batches:
                out = forward_model(model, item)
                key = 'adjusted_pair_logits' if model is adaptive else 'pair_logits'
                scores.append(out[key][item['pair_mask']])
                labels.append(item['pair_labels'][item['pair_mask']])
                if model is adaptive:
                    refs.append(out['adaptive_reference'][item['utterance_mask']])
        expected_ap = average_precision_score(torch.cat(labels).numpy(), torch.cat(scores).numpy())
        assert result['pair_auprc'] == expected_ap
        assert result['num_pair_samples'] == sum(t.numel() for t in labels)
        if refs:
            reference_stats = result['metrics']['reference_stats']
            assert abs(reference_stats['mean'] - torch.cat(refs).mean().item()) < 1e-7
        with contextlib.redirect_stdout(io.StringIO()) as capture:
            print_result(1, 1, result, result)
        text = capture.getvalue()
        for token in ('Emotion', 'Cause', 'Pair', '#Pred=', '#Gold=', 'AUPRC', model.pair_decision_mode):
            assert token in text
        if model is adaptive:
            assert all(token in text for token in ('mean=', 'std=', 'min=', 'max='))
    print('PASS evaluate: global logit AUPRC in both modes and required epoch logging')

    config = yaml.safe_load((ROOT / 'config/config_feature_pos2.5_adaptive_reference.yaml').read_text())
    assert config['training']['module_lr'] == 1e-4
    assert config['training']['pair_pos_weight'] == 2.5
    for filename in ('baseline.pt', 'train.log', 'history.json', 'run.yaml'):
        assert adaptive_artifact_path(filename, 'fixed') == Path(filename)
        assert adaptive_artifact_path(filename, 'adaptive_reference') != Path(filename)
    assert sum(p.numel() for p in adaptive.parameters()) == 2269572
    print('Adaptive reference parameters:', adaptive.adaptive_reference_parameters)
    print('PASS Dev dataset totals: 13301 valid pairs, 838 positives; artifact isolation and hyperparameters')


if __name__ == '__main__':
    run_checks()
