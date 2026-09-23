"""Controlled NULL v2 tests. Uses real features; never launches full training."""
import contextlib
import io
import json
from uuid import uuid4
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from dataset.feature_collate import FeatureECPECCollator
from dataset.feature_dataset import FeatureECFDataset
from models.feature_base_model import FeatureECPECBaseModel
from test_adaptive_reference import original_module
from train_feature import (
    ECPECMetricAccumulator, adaptive_artifact_path, compute_losses, evaluate,
    forward_model, make_history_entry, null_ranking_loss, print_result,
    save_checkpoint, train_one_epoch, validate_config,
)

ROOT = Path(__file__).resolve().parent


def run_checks():
    torch.set_num_threads(2)
    dataset = FeatureECFDataset(ROOT / 'features/ECF/dev_roberta.pt', max_dialogue_length=40)
    collate = FeatureECPECCollator()
    batch = collate([dataset[i] for i in range(4)])
    valid, mask = batch['utterance_mask'], batch['pair_mask']
    torch.manual_seed(42)
    fixed = FeatureECPECBaseModel()
    fixed_rng = torch.get_rng_state()
    torch.manual_seed(42)
    model = FeatureECPECBaseModel(pair_decision_mode='null_reference')
    assert torch.equal(fixed_rng, torch.get_rng_state())
    assert not any(isinstance(m, torch.nn.Dropout) for m in model.null_reference_head.modules())
    for key, value in fixed.state_dict().items():
        assert torch.equal(value, model.state_dict()[key]), key
    model.eval()
    output = model(**{k: batch[k] for k in (
        'utterance_features', 'speaker_ids', 'position_ids', 'utterance_mask', 'pair_mask'
    )}, return_features=True)
    assert output['context_features'].shape == (*valid.shape, 256)
    assert output['pair_logits'].shape == output['relative_pair_logits'].shape == batch['pair_labels'].shape == mask.shape
    assert output['null_logits'].shape == valid.shape
    assert all(torch.isfinite(value).all() for value in output.values())
    assert torch.count_nonzero(output['relative_pair_logits'][~mask]) == 0
    torch.testing.assert_close(
        output['relative_pair_logits'][mask],
        (output['pair_logits'] - output['null_logits'].unsqueeze(-1))[mask],
    )
    h = output['context_features']
    hd = torch.stack([h[b][valid[b]].mean(0) for b in range(h.shape[0])])
    expected_null = model.null_reference_head(torch.cat([h.detach(), hd.detach()[:, None].expand_as(h)], -1)).squeeze(-1)
    torch.testing.assert_close(output['null_logits'][valid], expected_null[valid])
    print('PASS real batch shapes:', {k: tuple(v.shape) for k, v in output.items()})
    print('PASS finite outputs, masked pooling, row alignment, no dropout, preserved initialization RNG')

    args = dict(pair_pos_weight=2.5, lambda_emotion=0.2, lambda_cause=0.4)
    losses = compute_losses(output, batch, **args)
    baseline_loss = compute_losses({k: output[k] for k in ('pair_logits', 'emotion_logits', 'cause_logits')}, batch, **args)
    assert torch.equal(losses['pair_loss'], baseline_loss['pair_loss'])
    torch.testing.assert_close(losses['total_loss'], baseline_loss['total_loss'] + 0.2 * losses['null_rank_loss'])
    output['context_features'].retain_grad()
    output['pair_logits'].retain_grad()
    losses['null_rank_loss'].backward()
    null_grad = sum(p.grad.abs().sum().item() for p in model.null_reference_head.parameters() if p.grad is not None)
    assert null_grad > 0
    for name, parameter in model.named_parameters():
        if not name.startswith('null_reference_head.'):
            assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0, name
    assert output['context_features'].grad is None
    assert output['pair_logits'].grad is None
    print(f'PASS actual rank-only backward: NULL grad L1={null_grad:.8f}; Pair/Dialogue/Aux/context/raw gradients absent')

    # Constructed rows cover positive+negative, negative-only, positive-only,
    # fewer negatives than k, padding NaNs, and equal row/group weighting.
    scores = torch.tensor([[[2., 1., 0., -9.], [4., 3., 2., -8.], [1., 2., 3., -7.], [99., 99., 99., 99.]]], requires_grad=True)
    labels = torch.tensor([[[1., 0., 0., 0.], [0., 0., 0., 0.], [1., 1., 1., 0.], [0., 0., 0., 0.]]])
    utterance_mask = torch.tensor([[True, True, True, False]])
    pair_mask = utterance_mask[:, :, None] & utterance_mask[:, None, :]
    null = torch.tensor([[0.5, 0.7, 0.9, float('nan')]], requires_grad=True)
    rank = null_ranking_loss(scores, null, labels, pair_mask, utterance_mask, margin=0.2, hard_negative_k=1)
    expected = torch.stack([
        (F.softplus(0.2 + null[0, 0] - scores[0, 0, 0].detach()) + F.softplus(0.2 + scores[0, 0, 1].detach() - null[0, 0])) / 2,
        F.softplus(0.2 + scores[0, 1, 0].detach() - null[0, 1]),
        F.softplus(0.2 + null[0, 2] - scores[0, 2, :3].detach()).mean(),
    ]).mean()
    torch.testing.assert_close(rank, expected)
    rank.backward()
    assert scores.grad is None
    assert null.grad[0, 3] == 0
    assert null.grad[0, 1] < 0  # Negative-only row raises NULL under gradient descent.
    poison_scores = scores.detach().clone(); poison_scores[~pair_mask] = float('nan')
    poison_labels = labels.clone(); poison_labels[~pair_mask] = float('nan')
    assert torch.equal(rank.detach(), null_ranking_loss(poison_scores, null, poison_labels, pair_mask, utterance_mask, hard_negative_k=1))
    # k larger than all row candidate counts must use all valid negatives.
    assert torch.equal(null_ranking_loss(scores, null, labels, pair_mask, utterance_mask, hard_negative_k=3),
                       null_ranking_loss(scores, null, labels, pair_mask, utterance_mask, hard_negative_k=100))
    empty = null_ranking_loss(scores, null, labels, torch.zeros_like(pair_mask), utterance_mask)
    assert empty.item() == 0 and torch.isfinite(empty)
    print('PASS ranking formula, hard top-k, negative-only rows, row/group averaging and padding exclusion')

    old = original_module('pre_fixed', 'feature_base_model.py').FeatureECPECBaseModel()
    old.load_state_dict(fixed.state_dict(), strict=True)
    old.eval(); fixed.eval()
    with torch.no_grad():
        old_out, fixed_out = forward_model(old, batch), forward_model(fixed, batch)
    assert set(fixed_out) == {'emotion_logits', 'cause_logits', 'pair_logits'}
    assert not hasattr(fixed, 'null_reference_head')
    assert 'null_rank_loss' not in compute_losses(fixed_out, batch, **args)
    assert all(torch.equal(old_out[k], fixed_out[k]) for k in old_out)
    fixed_metric = ECPECMetricAccumulator(0.66)
    fixed_metric.update(fixed_out, batch)
    assert fixed_metric.compute()['pair']['predicted_positive'] == int((fixed_out['pair_logits'][mask].sigmoid() >= 0.66).sum())
    print('PASS fixed eval exact baseline compatibility and raw prediction pipeline')

    # Two actual training-entry-point steps, including dropout, clipping and AdamW.
    batches = [batch, collate([dataset[4], dataset[5]])]
    base_opt = torch.optim.AdamW(fixed.parameters(), lr=1e-4, weight_decay=0.01)
    null_opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    # Earlier gradient-isolation backward is cleared by train_one_epoch.
    def train(m, opt):
        torch.manual_seed(1234)
        return train_one_epoch(m, batches, opt, torch.device('cpu'), 2.5, 0.2, 0.4,
                               0.66, 1.0, 1, 1)
    train_base = train(fixed, base_opt)
    train_null = train(model, null_opt)
    for key, value in fixed.state_dict().items():
        assert torch.equal(value, model.state_dict()[key]), key
    assert train_base['losses']['pair_loss'] == train_null['losses']['pair_loss']
    print('PASS two in-memory optimizer steps: baseline parameters bitwise unchanged by NULL branch')

    # Full Dev global AUPRC, not a batch-wise mean; no optimizer steps here.
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate)
    result = evaluate(model, loader, torch.device('cpu'), 2.5, 0.2, 0.4, 0.99)
    assert result['num_pair_samples'] == 13301 and result['num_positive_pairs'] == 838
    raw, relative, gold = [], [], []
    reference_metrics = ECPECMetricAccumulator(0.01, 'null_reference')
    with torch.no_grad():
        for item in loader:
            out = forward_model(model, item)
            raw.append(out['pair_logits'][item['pair_mask']])
            relative.append(out['relative_pair_logits'][item['pair_mask']])
            gold.append(item['pair_labels'][item['pair_mask']])
            reference_metrics.update(out, item)
    assert result['metrics']['pair'] == reference_metrics.compute()['pair']
    assert result['raw_pair_auprc'] == average_precision_score(torch.cat(gold).numpy(), torch.cat(raw).numpy())
    assert result['relative_pair_auprc'] == average_precision_score(torch.cat(gold).numpy(), torch.cat(relative).numpy())
    assert result['pair_auprc'] == result['relative_pair_auprc']
    assert result['metrics']['pair']['predicted_positive'] == int((torch.cat(relative) > 0).sum())
    assert result['metrics']['null_stats']['variation']['unique_valid_references'] > 1
    with contextlib.redirect_stdout(io.StringIO()) as capture:
        print_result(1, 1, train_null, result)
    for token in ('Dev Raw Pair AUPRC', 'Dev Relative Pair AUPRC', 'Dev NULL Logit', 'Dev Dynamic Boundary',
                  'dialogue_mean_std', 'unique_valid_references', 'pearson_row_max', 'No manually tuned global Pair threshold'):
        assert token in capture.getvalue(), token
    print('PASS full Dev global dual AUPRC, threshold independence, boundary diagnostics and logging')

    config = yaml.safe_load((ROOT / 'config/config_feature_pos2.5_null_reference.yaml').read_text())
    validate_config(config)
    baseline_config = yaml.safe_load((ROOT / 'config/config_feature_pos2.5_thr0.66.yaml').read_text())
    for key, value in baseline_config['training'].items():
        if key != 'checkpoint_path':
            assert config['training'][key] == value
    assert model.null_reference_parameters == 65793
    assert sum(p.numel() for p in model.parameters()) == 2269572
    target = ROOT / f".null_test_{uuid4().hex}.pt"
    try:
        save_checkpoint(target, model, null_opt, 1, result['metrics']['pair']['f1'], 0.0,
                        config, result['pair_auprc'], dev_result=result)
        checkpoint = torch.load(target, weights_only=False, map_location='cpu')
        assert checkpoint['raw_pair_auprc'] == result['raw_pair_auprc']
        assert checkpoint['relative_pair_auprc'] == result['relative_pair_auprc']
        assert checkpoint['null_stats'] == result['metrics']['null_stats']
        assert checkpoint['selection_metric'] == 'Dev Relative Pair-F1'
    finally:
        target.unlink(missing_ok=True)
    json.dumps(make_history_entry(1, train_null, result), allow_nan=False)
    for filename in ('baseline.pt', 'v1_adaptive_reference.pt', 'train.log', 'history.json', 'run.yaml'):
        assert adaptive_artifact_path(filename, 'null_reference') != Path(filename)
    assert adaptive_artifact_path(config['training']['checkpoint_path'], 'null_reference') == Path(config['training']['checkpoint_path'])
    print('PASS config, artifact isolation, checkpoint/history metadata; NULL Reference parameters: 65793')


if __name__ == '__main__':
    run_checks()
