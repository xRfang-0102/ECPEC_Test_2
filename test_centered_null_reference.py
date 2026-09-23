"""Centered NULL smoke/regression checks, without a full training run."""
import argparse
import contextlib
import io
import json
from pathlib import Path
from types import ModuleType
from uuid import uuid4

import torch
import yaml
from sklearn.metrics import average_precision_score

from dataset.feature_dataset import FeatureECFDataset
from dataset.feature_collate import FeatureECPECCollator
from models.feature_base_model import FeatureECPECBaseModel
from train_feature import (CenteredNullDiagnostics, compute_losses, forward_model,
    null_ranking_loss, optimizer_parameters, train_one_epoch, evaluate, print_result,
    adaptive_artifact_path, save_checkpoint, make_history_entry, validate_config)

ROOT = Path(__file__).resolve().parent
ARGS = dict(pair_pos_weight=2.5, lambda_emotion=0.2, lambda_cause=0.4)


def check_prior_modes(path, batch):
    source = json.loads(Path(path).read_text(encoding='utf-8'))
    modules = []
    for filename in ('models/feature_base_model.py', 'train_feature.py'):
        module = ModuleType('prechange_' + Path(filename).stem)
        module.__file__ = str(ROOT / filename)
        exec(compile(source[filename], module.__file__, 'exec'), module.__dict__)
        modules.append(module)
    old_model, old_train = modules
    for mode in ('fixed', 'adaptive_reference', 'null_reference'):
        torch.manual_seed(42)
        before = old_model.FeatureECPECBaseModel(pair_decision_mode=mode)
        before_rng = torch.get_rng_state()
        torch.manual_seed(42)
        after = FeatureECPECBaseModel(pair_decision_mode=mode)
        assert torch.equal(before_rng, torch.get_rng_state())
        assert before.state_dict().keys() == after.state_dict().keys()
        assert not hasattr(after, 'null_global_center')
        for key, value in before.state_dict().items():
            assert torch.equal(value, after.state_dict()[key]), (mode, key)
        for training in (False, True):
            before.train(training); after.train(training)
            torch.manual_seed(123)
            a = forward_model(before, batch)
            torch.manual_seed(123)
            b = forward_model(after, batch)
            assert a.keys() == b.keys()
            assert all(torch.equal(a[k], b[k]) for k in a)
            la, lb = old_train.compute_losses(a, batch, **ARGS), compute_losses(b, batch, **ARGS)
            assert la.keys() == lb.keys()
            assert all(torch.equal(la[k], lb[k]) for k in la)
            before.zero_grad(); after.zero_grad()
            la['total_loss'].backward(); lb['total_loss'].backward()
            for (name, p), (_, q) in zip(before.named_parameters(), after.named_parameters()):
                assert (p.grad is None and q.grad is None) or torch.equal(p.grad, q.grad), (mode, name)
        print(f'PASS pre-change {mode}: exact state/RNG/eval+train outputs/losses/gradients')


def run_checks(prechange=None):
    torch.set_num_threads(2)
    dataset = FeatureECFDataset(ROOT / 'features/ECF/dev_roberta.pt', max_dialogue_length=40)
    collate = FeatureECPECCollator()
    batch = collate([dataset[i] for i in range(4)])
    if prechange:
        check_prior_modes(prechange, batch)
    valid, mask = batch['utterance_mask'], batch['pair_mask']
    torch.manual_seed(42)
    fixed = FeatureECPECBaseModel()
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    model = FeatureECPECBaseModel(pair_decision_mode='centered_null_reference')
    assert torch.equal(rng, torch.get_rng_state())
    assert not hasattr(fixed, 'null_global_center')
    assert not hasattr(model, 'null_reference_head')
    assert not any(isinstance(m, torch.nn.Dropout) for m in model.null_residual_head.modules())
    model.eval()
    outputs = model(**{k: batch[k] for k in ('utterance_features', 'speaker_ids', 'position_ids', 'utterance_mask', 'pair_mask')}, return_features=True)
    assert outputs['null_global_center'].shape == torch.Size([])
    assert outputs['null_residual'].shape == outputs['null_logits'].shape == valid.shape
    assert outputs['pair_logits'].shape == outputs['relative_pair_logits'].shape == mask.shape == batch['pair_labels'].shape
    assert outputs['context_features'].shape == (*valid.shape, 256)
    assert all(torch.isfinite(t).all() for t in outputs.values())
    assert torch.count_nonzero(outputs['null_residual']) == 0
    assert outputs['null_global_center'].item() == 0
    assert torch.count_nonzero(outputs['null_logits']) == 0
    print('PASS shapes:', {k: tuple(v.shape) for k, v in outputs.items()})
    print('PASS finite outputs; residual exactly zero; center=0; preserved RNG and no Dropout')

    losses = compute_losses(outputs, batch, **ARGS)
    baseline = compute_losses({k: outputs[k] for k in ('pair_logits', 'emotion_logits', 'cause_logits')}, batch, **ARGS)
    assert torch.equal(losses['pair_loss'], baseline['pair_loss'])
    torch.testing.assert_close(losses['total_loss'], baseline['total_loss'] + 0.2 * losses['null_rank_loss'] + 0.01 * losses['residual_reg_loss'])
    assert losses['residual_reg_loss'].item() == 0
    expected_rank = null_ranking_loss(outputs['pair_logits'], outputs['null_logits'], batch['pair_labels'], mask, valid)
    assert torch.equal(expected_rank, losses['null_rank_loss'])
    outputs['context_features'].retain_grad(); outputs['pair_logits'].retain_grad()
    (losses['null_rank_loss'] + 0.01 * losses['residual_reg_loss']).backward()
    center_grad = model.null_global_center.grad.abs().item()
    head_grad = sum(p.grad.abs().sum().item() for p in model.null_residual_head.parameters() if p.grad is not None)
    assert center_grad > 0 and head_grad > 0
    for name, p in model.named_parameters():
        if name != 'null_global_center' and not name.startswith('null_residual_head.'):
            assert p.grad is None or torch.count_nonzero(p.grad) == 0, name
    assert outputs['context_features'].grad is None and outputs['pair_logits'].grad is None
    print(f'PASS isolated backward: center |grad|={center_grad:.8f}; residual head grad L1={head_grad:.8f}; base/context/raw gradients absent')

    original_head = {k: v.clone() for k, v in model.null_residual_head.state_dict().items()}
    with torch.no_grad():
        model.null_residual_head[-1].weight.fill_(0.03)
        model.null_residual_head[-1].bias.fill_(0.1)
    model.zero_grad(set_to_none=True)
    outputs = forward_model(model, batch)
    assert outputs['null_residual'][valid].std() > 0
    assert outputs['null_residual'].abs().max() <= 0.5
    losses = compute_losses(outputs, batch, **ARGS)
    torch.testing.assert_close(losses['residual_reg_loss'], outputs['null_residual'][valid].square().mean())
    losses['residual_reg_loss'].backward()
    assert model.null_global_center.grad is None  # No center L2 anchor.
    assert model.null_residual_head[-1].weight.grad.abs().sum() > 0
    poisoned = {k: v.detach().clone() for k, v in outputs.items()}
    poisoned['null_residual'][~valid] = float('nan')
    poisoned['null_logits'][~valid] = float('nan')
    poisoned['pair_logits'][~mask] = float('nan')
    poisoned['relative_pair_logits'][~mask] = float('nan')
    poison_batch = dict(batch, pair_labels=batch['pair_labels'].clone())
    poison_batch['pair_labels'][~mask] = float('nan')
    poisoned_losses = compute_losses(poisoned, poison_batch, **ARGS)
    for key in ('null_rank_loss', 'residual_reg_loss'):
        assert torch.equal(losses[key].detach(), poisoned_losses[key])
    normal_stats, poisoned_stats = CenteredNullDiagnostics(), CenteredNullDiagnostics()
    normal_stats.update(outputs, batch); poisoned_stats.update(poisoned, poison_batch)
    assert normal_stats.compute() == poisoned_stats.compute()
    stats = normal_stats.compute()
    positive = ((batch['pair_labels'] == 1) & mask).any(-1) & valid
    assert stats['positive_row_boundary']['count'] == int(positive.sum())
    assert stats['no_positive_row_boundary']['count'] == int((valid & ~positive).sum())
    print('PASS bounded residual, valid-only L2, no center anchor, padding-free losses/statistics and gold-row diagnostics')

    model.null_residual_head.load_state_dict(original_head)
    base_opt = torch.optim.AdamW(optimizer_parameters(fixed), lr=1e-4, weight_decay=0.01)
    opt = torch.optim.AdamW(optimizer_parameters(model), lr=1e-4, weight_decay=0.01)
    assert opt.param_groups[1]['params'][0] is model.null_global_center
    assert opt.param_groups[1]['weight_decay'] == 0
    assert opt.param_groups[0]['weight_decay'] == 0.01
    batches = [batch, collate([dataset[4], dataset[5]])]
    def train(m, optimizer):
        torch.manual_seed(1234)
        return train_one_epoch(m, batches, optimizer, torch.device('cpu'), 2.5, 0.2, 0.4, 0.66, 1., 1, 1)
    train(fixed, base_opt)
    train_result = train(model, opt)
    for key, value in fixed.state_dict().items():
        assert torch.equal(value, model.state_dict()[key]), key
    print('PASS two in-memory optimizer steps: bitwise-equal baseline parameters; center excluded from weight decay')

    result = evaluate(model, batches, torch.device('cpu'), 2.5, 0.2, 0.4, 0.99)
    raw, relative, gold = [], [], []
    with torch.no_grad():
        for item in batches:
            out = forward_model(model, item)
            raw.append(out['pair_logits'][item['pair_mask']])
            relative.append(out['relative_pair_logits'][item['pair_mask']])
            gold.append(item['pair_labels'][item['pair_mask']])
    assert result['raw_pair_auprc'] == average_precision_score(torch.cat(gold).numpy(), torch.cat(raw).numpy())
    assert result['relative_pair_auprc'] == average_precision_score(torch.cat(gold).numpy(), torch.cat(relative).numpy())
    assert result['metrics']['pair']['predicted_positive'] == int((torch.cat(relative) > 0).sum())
    with contextlib.redirect_stdout(io.StringIO()) as capture:
        print_result(1, 20, train_result, result)
    for token in ('Global Center Logit', 'Global Center Probability', 'Residual Logit', 'Residual absolute mean',
                  'Residual std', 'Dynamic Boundary Logit', 'Dynamic Boundary Probability', 'Positive-row boundary',
                  'No-positive-row boundary', 'Residual Reg Loss', 'NULL Rank Loss', 'Dev Raw Pair AUPRC', 'Dev Relative Pair AUPRC'):
        assert token in capture.getvalue(), token
    config_path = ROOT / 'config/config_feature_pos2.5_centered_null_reference.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    validate_config(config)
    assert config['training']['epochs'] == 20 and config['training']['early_stop_patience'] == 100
    assert config['training']['module_lr'] == 1e-4 and config['training']['pair_pos_weight'] == 2.5
    assert model.centered_null_parameters == 65794
    assert sum(p.numel() for p in model.parameters()) == 2269573
    for name in ('training_history_feature.json', 'run_config_feature.yaml', 'v2_null_reference.pt'):
        assert 'centered_null_reference' in adaptive_artifact_path(name, 'centered_null_reference').stem
    target = ROOT / f'.centered_test_{uuid4().hex}.pt'
    try:
        save_checkpoint(target, model, opt, 1, result['metrics']['pair']['f1'], 0., config, result['pair_auprc'], dev_result=result)
        saved = torch.load(target, weights_only=False, map_location='cpu')
        assert saved['null_stats'] == result['metrics']['null_stats']
        assert saved['selection_metric'] == 'Dev Relative Pair-F1'
        assert saved['raw_pair_auprc'] == result['raw_pair_auprc']
    finally:
        target.unlink(missing_ok=True)
    json.dumps(make_history_entry(1, train_result, result), allow_nan=False)
    print('PASS global dual AUPRC, strict relative prediction, diagnostics, isolated metadata/paths and config')
    print('Centered NULL parameters: 65794 (center=1, residual head=65793); total=2269573')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prechange', help='Optional temporary JSON source snapshot for exact prior-mode regression')
    run_checks(parser.parse_args().prechange)
