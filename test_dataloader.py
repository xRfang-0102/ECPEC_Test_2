from torch.utils.data import DataLoader

from dataset.dataset import ECFDataset
from dataset.collate import ECPECCollator


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

batch = next(
    iter(loader)
)


print(
    "dialogue_ids:",
    batch["dialogue_ids"],
)

print(
    "dialogue_lengths:",
    batch["dialogue_lengths"],
)

print(
    "input_ids:",
    batch["input_ids"].shape,
)

print(
    "attention_mask:",
    batch["attention_mask"].shape,
)

print(
    "speaker_ids:",
    batch["speaker_ids"].shape,
)

print(
    "position_ids:",
    batch["position_ids"].shape,
)

print(
    "utterance_mask:",
    batch["utterance_mask"].shape,
)

print(
    "emotion_labels:",
    batch["emotion_labels"].shape,
)

print(
    "cause_labels:",
    batch["cause_labels"].shape,
)

print(
    "pair_labels:",
    batch["pair_labels"].shape,
)

print(
    "pair_mask:",
    batch["pair_mask"].shape,
)


print()
print(
    "Valid utterances:",
    int(
        batch[
            "utterance_mask"
        ].sum().item()
    ),
)

print(
    "Valid pairs:",
    int(
        batch[
            "pair_mask"
        ].sum().item()
    ),
)

print(
    "Positive pairs:",
    int(
        batch[
            "pair_labels"
        ].sum().item()
    ),
)