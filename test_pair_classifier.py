import torch

from models.pair_classifier import (
    PairClassifier
)


batch_size = 4
num_utterances = 9
hidden_size = 256


context_features = torch.randn(
    batch_size,
    num_utterances,
    hidden_size,
)


speaker_ids = torch.tensor([
    [0,1,0,2,0,2,3,0,0],
    [0,1,0,0,0,0,0,0,0],
    [0,1,0,1,2,2,0,1,0],
    [0,1,0,0,0,0,0,0,0],
])


position_ids = torch.arange(
    num_utterances
).unsqueeze(0).repeat(
    batch_size,
    1,
)


utterance_mask = torch.tensor([
    [1,1,1,1,1,1,1,1,0],
    [1,1,1,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1],
    [1,1,1,0,0,0,0,0,0],
], dtype=torch.bool)


pair_mask = (
    utterance_mask.unsqueeze(2)
    &
    utterance_mask.unsqueeze(1)
)


model = PairClassifier(
    hidden_size=256,
    pair_hidden_size=256,
    distance_embedding_dim=32,
    speaker_relation_embedding_dim=16,
    max_relative_distance=15,
    dropout=0.1,
)


pair_logits = model(
    context_features=context_features,
    speaker_ids=speaker_ids,
    position_ids=position_ids,
    pair_mask=pair_mask,
)


print(
    "Pair logits:",
    pair_logits.shape,
)

print(
    "Valid pairs:",
    int(
        pair_mask
        .sum()
        .item()
    ),
)


padding_logits = (
    pair_logits[
        ~pair_mask
    ]
)

if padding_logits.numel() > 0:
    print(
        "Padding pair max abs:",
        padding_logits
        .abs()
        .max()
        .item()
    )