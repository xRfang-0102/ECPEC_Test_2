import torch
from torch.utils.data import DataLoader

from dataset.dataset import ECFDataset
from dataset.collate import ECPECCollator
from models.utterance_encoder import UtteranceEncoder


dataset = ECFDataset(
    "data/ECF/train.json",
    max_dialogue_length=40,
)

collator = ECPECCollator(
    tokenizer_name="pretrained/roberta-base",
    max_utterance_length=128,
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    collate_fn=collator,
)

batch = next(iter(loader))


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = UtteranceEncoder(
    encoder_name="pretrained/roberta-base",
    dropout=0.1,
    freeze_encoder=False,
).to(device)


input_ids = (
    batch["input_ids"]
    .to(device)
)

attention_mask = (
    batch["attention_mask"]
    .to(device)
)

utterance_mask = (
    batch["utterance_mask"]
    .to(device)
)


with torch.no_grad():
    features = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        utterance_mask=utterance_mask,
    )


print(
    "Input IDs:",
    input_ids.shape,
)

print(
    "Utterance mask:",
    utterance_mask.shape,
)

print(
    "Output features:",
    features.shape,
)

print(
    "Hidden size:",
    model.hidden_size,
)

print(
    "Valid utterances:",
    int(
        utterance_mask
        .sum()
        .item()
    ),
)

padding_features = (
    features[
        ~utterance_mask
    ]
)

if padding_features.numel() > 0:
    print(
        "Padding feature max abs:",
        padding_features
        .abs()
        .max()
        .item()
    )