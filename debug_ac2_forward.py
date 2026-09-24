import torch
from pathlib import Path

from sweep_threshold import load_config, build_model, load_checkpoint
from dataset.feature_dataset import FeatureECFDataset
from dataset.feature_collate import FeatureECPECCollator

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG = PROJECT_ROOT / "config" / "config_feature_mean_ac2.yaml"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "base_feature_mean_ac2_best.pt"

device = torch.device("cuda")

config = load_config(CONFIG)
model = build_model(config["model"], 768).to(device)
model.eval()
load_checkpoint(CHECKPOINT, model, device)

dataset = FeatureECFDataset(
    feature_path=PROJECT_ROOT / "features" / "ECF" / "dev_roberta_mean.pt",
    max_dialogue_length=40,
    expected_hidden_size=768,
    validate=True,
)

# ---- single-sample forward (error_analysis style) ----
sample = dataset[0]
n = sample["num_utterances"]
with torch.no_grad():
    out_single = model(
        utterance_features=sample["utterance_features"][:n].unsqueeze(0).to(device),
        speaker_ids=sample["speaker_ids"][:n].unsqueeze(0).to(device),
        position_ids=torch.arange(n, device=device).unsqueeze(0),
        utterance_mask=torch.ones(1, n, dtype=torch.bool, device=device),
        pair_mask=torch.ones(1, n, n, dtype=torch.bool, device=device),
    )
print("single logits finite:", bool(torch.isfinite(out_single["pair_logits"]).all()))
print("single max sigmoid:", torch.sigmoid(out_single["pair_logits"]).max().item())
print("single #pred>0.66:", int((torch.sigmoid(out_single["pair_logits"]) >= 0.66).sum().item()))

# ---- batched forward (sweep/training style) ----
collator = FeatureECPECCollator(hidden_size=768)
loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collator)
batch = next(iter(loader))
with torch.no_grad():
    out_batch = model(
        utterance_features=batch["utterance_features"].to(device),
        speaker_ids=batch["speaker_ids"].to(device),
        position_ids=batch["position_ids"].to(device),
        utterance_mask=batch["utterance_mask"].to(device),
        pair_mask=batch["pair_mask"].to(device),
    )
print("batch logits finite:", bool(torch.isfinite(out_batch["pair_logits"]).all()))
print("batch max sigmoid:", torch.sigmoid(out_batch["pair_logits"]).max().item())
print("batch #pred>0.66:", int((torch.sigmoid(out_batch["pair_logits"]) >= 0.66).sum().item()))

# same first dialogue within the batch
b0 = batch["utterance_mask"][0].bool()
n0 = int(b0.sum().item())
pairs0 = out_batch["pair_logits"][0, :n0, :n0]
print("batch-dialogue0 max sigmoid:", torch.sigmoid(pairs0).max().item())
