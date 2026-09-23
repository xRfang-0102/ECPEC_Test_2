"""V4 controlled smoke tests. No full training or persistent checkpoint writes."""
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
from torch.utils.data import DataLoader
from dataset.feature_dataset import FeatureECFDataset
from dataset.feature_collate import FeatureECPECCollator
from models.feature_base_model import FeatureECPECBaseModel
from test_centered_null_reference import check_prior_modes
from train_feature import (HierarchicalDiagnostics, compute_losses, forward_model,
    null_ranking_loss, optimizer_parameters, train_one_epoch, evaluate, print_result,
    save_checkpoint, make_history_entry, adaptive_artifact_path, validate_config)

ROOT = Path(__file__).resolve().parent
ARGS = dict(pair_pos_weight=2.5, lambda_emotion=0.2, lambda_cause=0.4)


def prior_v3(path, batch):
    modules = []
    source = json.loads(Path(path).read_text(encoding='utf-8'))
    for filename in ('models/feature_base_model.py', 'train_feature.py'):
        m = ModuleType('pre_' + Path(filename).stem); m.__file__ = str(ROOT / filename)
        exec(compile(source[filename], m.__file__, 'exec'), m.__dict__)
        modules.append(m)
    torch.manual_seed(42)
    old = modules[0].FeatureECPECBaseModel(pair_decision_mode='centered_null_reference')
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    new = FeatureECPECBaseModel(pair_decision_mode='centered_null_reference')
    assert torch.equal(rng, torch.get_rng_state())
    assert old.state_dict().keys() == new.state_dict().keys()
    assert not hasattr(new, 'global_center') and not hasattr(new, 'dialogue_offset_head')
    for key, val in old.state_dict().items():
        assert torch.equal(val, new.state_dict()[key])
    for training in (False, True):
        old.train(training); new.train(training)
        torch.manual_seed(123); a = forward_model(old, batch)
        torch.manual_seed(123); b = forward_model(new, batch)
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
        la, lb = modules[1].compute_losses(a, batch, **ARGS), compute_losses(b, batch, **ARGS)
        assert la.keys() == lb.keys() and all(torch.equal(la[k], lb[k]) for k in la)
        old.zero_grad(); new.zero_grad(); la['total_loss'].backward(); lb['total_loss'].backward()
        for p, q in zip(old.parameters(), new.parameters()):
            assert (p.grad is None and q.grad is None) or torch.equal(p.grad, q.grad)
    print('PASS pre-change centered_null_reference: exact state/RNG/eval+train outputs/losses/gradients')


def run_checks(snapshot=None):
    torch.set_num_threads(2)
    dataset = FeatureECFDataset(ROOT / 'features/ECF/dev_roberta.pt', max_dialogue_length=40)
    collate = FeatureECPECCollator()
    batch = collate([dataset[i] for i in range(4)])
    if snapshot:
        check_prior_modes(snapshot, batch); prior_v3(snapshot, batch)
    valid, mask = batch['utterance_mask'], batch['pair_mask']
    torch.manual_seed(42); fixed = FeatureECPECBaseModel(); rng = torch.get_rng_state()
    torch.manual_seed(42); model = FeatureECPECBaseModel(pair_decision_mode='hierarchical_boundary')
    assert torch.equal(rng, torch.get_rng_state())
    assert not hasattr(fixed, 'global_center')
    for head in (model.dialogue_offset_head, model.row_residual_head):
        assert not any(isinstance(m, torch.nn.Dropout) for m in head.modules())
    model.eval()
    def forward():
        return model(**{k: batch[k] for k in ('utterance_features', 'speaker_ids', 'position_ids', 'utterance_mask', 'pair_mask')}, return_features=True)
    out = forward()
    assert out['context_features'].shape == (*valid.shape, 256)
    assert out['global_center'].shape == torch.Size([])
    assert out['dialogue_offset'].shape == (valid.shape[0],)
    for key in ('row_residual_raw', 'row_residual_bounded', 'row_residual_zero_mean', 'dynamic_boundary', 'null_logits'):
        assert out[key].shape == valid.shape
    assert out['pair_logits'].shape == out['relative_pair_logits'].shape == batch['pair_labels'].shape == mask.shape
    assert all(torch.isfinite(t).all() for t in out.values())
    assert torch.count_nonzero(out['dialogue_offset']) == torch.count_nonzero(out['row_residual_zero_mean']) == 0
    assert torch.equal(out['dynamic_boundary'][valid], out['global_center'].expand_as(valid)[valid])
    print('PASS shapes:', {k: tuple(v.shape) for k, v in out.items()})
    print('PASS finite and initialization: dialogue_offset=0, row_residual=0, boundary=center=0')
    losses = compute_losses(out, batch, **ARGS)
    assert 'residual_reg_loss' not in losses
    base = compute_losses({k: out[k] for k in ('pair_logits','emotion_logits','cause_logits')}, batch, **ARGS)
    torch.testing.assert_close(losses['total_loss'], base['total_loss'] + 0.2*losses['null_rank_loss'] + 0.005*losses['dialogue_reg_loss'] + 0.01*losses['row_reg_loss'])
    assert torch.equal(losses['null_rank_loss'], null_ranking_loss(out['pair_logits'],out['null_logits'],batch['pair_labels'],mask,valid))
    out['context_features'].retain_grad(); out['pair_logits'].retain_grad()
    extra = 0.2*losses['null_rank_loss'] + 0.005*losses['dialogue_reg_loss'] + 0.01*losses['row_reg_loss']
    extra.backward()
    def grad_sum(head):
        return sum(p.grad.abs().sum().item() for p in head.parameters() if p.grad is not None)
    values = (model.global_center.grad.abs().item(), grad_sum(model.dialogue_offset_head), grad_sum(model.row_residual_head))
    assert all(v > 0 for v in values)
    for name, p in model.named_parameters():
        if name != 'global_center' and not name.startswith(('dialogue_offset_head.', 'row_residual_head.')):
            assert p.grad is None or torch.count_nonzero(p.grad) == 0, name
    assert out['context_features'].grad is None and out['pair_logits'].grad is None
    print('PASS actual isolated backward: center/dialogue/row gradient L1 =', values, '; base/context/raw gradients absent')

    initial_heads = [{k:v.clone() for k,v in h.state_dict().items()} for h in (model.dialogue_offset_head, model.row_residual_head)]
    with torch.no_grad():
        model.dialogue_offset_head[-1].weight.fill_(0.03)
        model.dialogue_offset_head[-1].bias.fill_(0.1)
        model.row_residual_head[-1].weight.copy_(torch.linspace(-0.08, 0.08, 128).reshape(1,-1))
        model.row_residual_head[-1].bias.fill_(0.2)
    out = forward()
    means = torch.stack([out['row_residual_zero_mean'][b][valid[b]].mean() for b in range(valid.shape[0])])
    deviation = means.abs().max().item()
    assert deviation < 1e-6
    assert out['row_residual_zero_mean'][valid].std() > 0
    assert out['dialogue_offset'].std() > 0
    assert out['dialogue_offset'].abs().max() <= 0.5 and out['row_residual_bounded'].abs().max() <= 0.5
    # Centering is after bounding: centered residual can reach twice row_scale.
    assert out['row_residual_zero_mean'].abs().max() <= 1.0
    h = out['context_features'].detach()
    hd = torch.stack([h[b][valid[b]].mean(0) for b in range(h.shape[0])])
    expected_offset = 0.5 * model.dialogue_offset_head(hd).squeeze(-1).tanh()
    torch.testing.assert_close(out['dialogue_offset'], expected_offset)
    bounded = 0.5 * model.row_residual_head(torch.cat([h,hd[:,None].expand_as(h)],-1)).squeeze(-1).tanh()
    for b in range(valid.shape[0]):
        torch.testing.assert_close(out['row_residual_zero_mean'][b][valid[b]], bounded[b][valid[b]] - bounded[b][valid[b]].mean())
    torch.testing.assert_close(out['dynamic_boundary'][valid],(model.global_center + out['dialogue_offset'][:,None] + out['row_residual_zero_mean'])[valid])
    # Perturb encoder padding only: valid context means and boundaries must not change.
    def poison_padding(module, inputs, output):
        changed = output.clone(); changed[~valid] = 1e6; return changed
    handle = model.dialogue_encoder.register_forward_hook(poison_padding)
    try:
        changed = forward()
    finally:
        handle.remove()
    torch.testing.assert_close(changed['dialogue_offset'], out['dialogue_offset'])
    torch.testing.assert_close(changed['dynamic_boundary'][valid], out['dynamic_boundary'][valid])
    print(f'PASS nonzero zero-mean test: max absolute dialogue mean={deviation:.10g}; masked pooling excludes poisoned padding')

    losses = compute_losses(out,batch,**ARGS)
    torch.testing.assert_close(losses['dialogue_reg_loss'], out['dialogue_offset'][valid.any(-1)].square().mean())
    torch.testing.assert_close(losses['row_reg_loss'], out['row_residual_zero_mean'][valid].square().mean())
    model.zero_grad(set_to_none=True)
    (losses['dialogue_reg_loss'] + losses['row_reg_loss']).backward()
    assert model.global_center.grad is None
    poison = {k:v.detach().clone() for k,v in out.items()}
    for key in ('row_residual_raw','row_residual_bounded','row_residual_zero_mean','dynamic_boundary','null_logits'):
        poison[key][~valid] = float('nan')
    poison['pair_logits'][~mask] = float('nan')
    poisoned_losses = compute_losses(poison,batch,**ARGS)
    for key in ('null_rank_loss','dialogue_reg_loss','row_reg_loss'):
        assert torch.equal(losses[key].detach(),poisoned_losses[key])
    clean_stats, dirty_stats = HierarchicalDiagnostics(), HierarchicalDiagnostics()
    clean_stats.update(out,batch); dirty_stats.update(poison,batch)
    assert clean_stats.compute() == dirty_stats.compute()
    stats = clean_stats.compute()
    assert stats['max_absolute_dialogue_row_mean'] < 1e-6
    assert stats['positive_row_boundary']['count'] + stats['no_positive_row_boundary']['count'] == valid.sum().item()
    # Empty sample is excluded from offset regularization and diagnostics.
    empty_batch = {k:(torch.cat([v,torch.zeros_like(v[:1])]) if torch.is_tensor(v) and v.ndim else v) for k,v in batch.items()}
    empty_out = {k:(torch.cat([v.detach(),torch.zeros_like(v[:1])]) if v.ndim else v.detach()) for k,v in out.items()}
    empty_out['dialogue_offset'][-1] = float('nan')
    empty_losses = compute_losses(empty_out,empty_batch,**ARGS)
    assert torch.equal(empty_losses['dialogue_reg_loss'],losses['dialogue_reg_loss'])
    empty_stats = HierarchicalDiagnostics(); empty_stats.update(empty_out,empty_batch)
    assert empty_stats.compute() == clean_stats.compute()
    print('PASS unchanged ranking loss, valid-only regularizers/diagnostics, no center anchor and empty-sample exclusion')

    for head,state in zip((model.dialogue_offset_head,model.row_residual_head),initial_heads):
        head.load_state_dict(state)
    opt = torch.optim.AdamW(optimizer_parameters(model), lr=1e-4, weight_decay=0.01)
    fixed_opt = torch.optim.AdamW(optimizer_parameters(fixed), lr=1e-4, weight_decay=0.01)
    assert opt.param_groups[1]['params'][0] is model.global_center and opt.param_groups[1]['weight_decay'] == 0
    batches = [batch,collate([dataset[4],dataset[5]])]
    with contextlib.redirect_stderr(io.StringIO()):
        torch.manual_seed(1234)
        train_one_epoch(fixed,batches,fixed_opt,torch.device('cpu'),2.5,0.2,0.4,0.66,1.,1,1)
        torch.manual_seed(1234)
        train_result = train_one_epoch(model,batches,opt,torch.device('cpu'),2.5,0.2,0.4,0.99,1.,1,1)
    for key,val in fixed.state_dict().items():
        assert torch.equal(val,model.state_dict()[key]),key
    print('PASS two in-memory training steps: bitwise-identical base parameters')
    loader = DataLoader(dataset,batch_size=4,collate_fn=collate)
    with contextlib.redirect_stderr(io.StringIO()):
        result = evaluate(model,loader,torch.device('cpu'),2.5,0.2,0.4,0.99)
    assert result['num_pair_samples']==13301 and result['num_positive_pairs']==838
    raw,relative,gold = [],[],[]
    with torch.no_grad():
        for item in loader:
            o=forward_model(model,item); m=item['pair_mask']
            raw.append(o['pair_logits'][m]);relative.append(o['relative_pair_logits'][m]);gold.append(item['pair_labels'][m])
    assert result['raw_pair_auprc']==average_precision_score(torch.cat(gold).numpy(),torch.cat(raw).numpy())
    assert result['relative_pair_auprc']==average_precision_score(torch.cat(gold).numpy(),torch.cat(relative).numpy())
    assert result['metrics']['pair']['predicted_positive']==int((torch.cat(relative)>0).sum())
    with contextlib.redirect_stdout(io.StringIO()) as capture:
        print_result(1,20,train_result,result)
    for token in ('Global Center Logit','Global Center Probability','Dialogue Offset','Dialogue Boundary Probability',
                  'Raw Row Residual','Zero-mean Row Residual','Mean absolute dialogue-wise row residual mean',
                  'Dynamic Boundary Logit','Dynamic Boundary Probability','Positive-row boundary','No-positive-row boundary',
                  'Dialogue offset absolute mean','Row residual absolute mean','Boundary contribution','Dev Raw Pair AUPRC','Dev Relative Pair AUPRC'):
        assert token in capture.getvalue(),token
    config=yaml.safe_load((ROOT/'config/config_feature_pos2.5_hierarchical_boundary.yaml').read_text())
    validate_config(config)
    v3=yaml.safe_load((ROOT/'config/config_feature_pos2.5_centered_null_reference.yaml').read_text())
    for key,val in v3['training'].items():
        if key not in ('null_residual_reg_weight','checkpoint_path'):
            assert config['training'][key]==val
    assert config['training']['epochs']==20 and config['training']['early_stop_patience']==100
    assert model.hierarchical_boundary_parameters==98819
    assert sum(p.numel() for p in model.parameters())==2302598
    for name in ('v3_centered_null_reference.pt','training_history_feature.json','run_config_feature.yaml','train.log'):
        assert 'hierarchical_boundary' in adaptive_artifact_path(name,'hierarchical_boundary').stem
    target=ROOT/f'.hierarchical_test_{uuid4().hex}.pt'
    try:
        save_checkpoint(target,model,opt,1,result['metrics']['pair']['f1'],0.,config,result['pair_auprc'],dev_result=result)
        saved=torch.load(target,weights_only=False,map_location='cpu')
        assert saved['null_stats']==result['metrics']['null_stats']
        assert saved['selection_metric']=='Dev Relative Pair-F1'
    finally:
        target.unlink(missing_ok=True)
    json.dumps(make_history_entry(1,train_result,result),allow_nan=False)
    print('PASS full Dev dual AUPRC/global metrics, diagnostics, metadata, config and artifact isolation')
    print('Hierarchical Boundary parameters: 98819 (global=1, dialogue=33025, row=65793); total=2302598')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prechange')
    run_checks(parser.parse_args().prechange)
