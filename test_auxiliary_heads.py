import torch

from models.auxiliary_heads import AuxiliaryHeads


batch_size = 4
num_utterances = 9
hidden_size = 256

features = torch.randn(
    batch_size,
    num_utterances,
    hidden_size,
)

utterance_mask = torch.tensor([
    [1,1,1,1,1,1,1,1,0],
    [1,1,1,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1],
    [1,1,1,0,0,0,0,0,0],
], dtype=torch.bool)


model = AuxiliaryHeads(
    input_size=256,
    hidden_size=128,
    dropout=0.1,
)

outputs = model(
    context_features=features,
    utterance_mask=utterance_mask,
)


print(
    "Emotion logits:",
    outputs["emotion_logits"].shape,
)

print(
    "Cause logits:",
    outputs["cause_logits"].shape,
)

emotion_padding = (
    outputs["emotion_logits"][
        ~utterance_mask
    ]
)

cause_padding = (
    outputs["cause_logits"][
        ~utterance_mask
    ]
)

print(
    "Emotion padding max abs:",
    emotion_padding.abs().max().item()
)

print(
    "Cause padding max abs:",
    cause_padding.abs().max().item()
)