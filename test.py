from dataset.dataset import ECFDataset


dataset = ECFDataset(
    "data/ECF/train.json",
    max_dialogue_length=None,
)

print(
    "Number of dialogues:",
    len(dataset),
)

sample = dataset[0]

print(
    "Dialogue ID:",
    sample["dialogue_id"],
)

print(
    "Num utterances:",
    sample["num_utterances"],
)

print(
    "Utterance IDs:",
    sample["utterance_ids"],
)

print(
    "Speaker IDs:",
    sample["speaker_ids"],
)

print(
    "Emotion labels:",
    sample["emotion_labels"],
)

print(
    "Cause labels:",
    sample["cause_labels"],
)

print(
    "Pairs:",
    sample["pairs"],
)

print(
    "Pair label shape:",
    sample["pair_labels"].shape,
)

print(
    "Positive pairs:",
    int(
        sample["pair_labels"]
        .sum()
        .item()
    ),
)

print(
    "Number of gold pairs:",
    len(sample["pairs"]),
)