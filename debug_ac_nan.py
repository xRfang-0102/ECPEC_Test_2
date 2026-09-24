import torch

from models.feature_base_model import FeatureECPECBaseModel
from dataset.feature_dataset import FeatureECFDataset
from dataset.feature_collate import FeatureECPECCollator

import train_feature as T

torch.manual_seed(42)

device = torch.device("cuda")

model = FeatureECPECBaseModel(
    use_event_retrieval=True,
    use_locality_prior=True,
).to(device)

dataset = FeatureECFDataset(
    feature_path="features/ECF/train_roberta.pt",
    max_dialogue_length=40,
    expected_hidden_size=768,
    validate=True,
)

collator = FeatureECPECCollator(hidden_size=768)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collator,
)

model.train()

for batch_index, batch in enumerate(loader):

    for key in (
        "utterance_features",
        "speaker_ids",
        "position_ids",
        "utterance_mask",
        "pair_mask",
        "emotion_labels",
        "cause_labels",
        "pair_labels",
    ):
        if key in batch and torch.is_tensor(batch[key]):
            batch[key] = batch[key].to(device)

    outputs = model(
        utterance_features=batch["utterance_features"],
        speaker_ids=batch["speaker_ids"],
        position_ids=batch["position_ids"],
        utterance_mask=batch["utterance_mask"],
        pair_mask=batch["pair_mask"],
    )

    for name, value in outputs.items():
        if torch.is_tensor(value):
            print(
                f"batch {batch_index} {name}: "
                f"finite={bool(torch.isfinite(value).all())} "
                f"min={value.min().item():.3f} max={value.max().item():.3f}"
            )

    losses = T.compute_losses(
        outputs,
        batch,
        pair_pos_weight=2.5,
        lambda_emotion=0.2,
        lambda_cause=0.4,
        pair_rank_weight=0.3,
        cond_rank_weight=0.3,
        retrieval_loss_weight=0.1,
        locality_prior_reg_weight=0.001,
    )

    for name, value in losses.items():
        print(
            f"  {name}: {value.item():.4f} "
            f"finite={bool(torch.isfinite(value).all())}"
        )

    if batch_index >= 2:
        break
