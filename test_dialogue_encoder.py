import torch
from torch.utils.data import DataLoader

from dataset.dataset import ECFDataset
from dataset.collate import ECPECCollator

from models.utterance_encoder import (
    UtteranceEncoder
)

from models.dialogue_encoder import (
    DialogueEncoder
)


dataset = ECFDataset(
    "data/ECF/train.json",
    max_dialogue_length=40,
)

collator = ECPECCollator(
    tokenizer_name=(
        "pretrained/roberta-base"
    ),
    max_utterance_length=128,
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    collate_fn=collator,
)

batch = next(
    iter(loader)
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


utterance_encoder = (
    UtteranceEncoder(
        encoder_name=(
            "pretrained/roberta-base"
        ),
        dropout=0.1,
        freeze_encoder=False,
    )
    .to(device)
)


dialogue_encoder = (
    DialogueEncoder(
        input_size=768,
        hidden_size=256,

        speaker_embedding_dim=32,
        position_embedding_dim=32,

        max_speakers=20,
        max_positions=64,

        num_layers=2,
        num_heads=4,
        ff_dim=1024,

        dropout=0.1,
    )
    .to(device)
)


input_ids = (
    batch["input_ids"]
    .to(device)
)

attention_mask = (
    batch["attention_mask"]
    .to(device)
)

speaker_ids = (
    batch["speaker_ids"]
    .to(device)
)

position_ids = (
    batch["position_ids"]
    .to(device)
)

utterance_mask = (
    batch["utterance_mask"]
    .to(device)
)


utterance_encoder.eval()
dialogue_encoder.eval()


with torch.no_grad():

    utterance_features = (
        utterance_encoder(
            input_ids=input_ids,
            attention_mask=(
                attention_mask
            ),
            utterance_mask=(
                utterance_mask
            ),
        )
    )

    context_features = (
        dialogue_encoder(
            utterance_features=(
                utterance_features
            ),
            speaker_ids=(
                speaker_ids
            ),
            position_ids=(
                position_ids
            ),
            utterance_mask=(
                utterance_mask
            ),
        )
    )


print(
    "Utterance features:",
    utterance_features.shape,
)

print(
    "Context features:",
    context_features.shape,
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
    context_features[
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