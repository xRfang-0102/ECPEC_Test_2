import torch

for split in ["train", "dev", "test"]:
    path = f"features/ECF/{split}_roberta.pt"
    data = torch.load(path, map_location="cpu")

    print("\n", split.upper())
    print("metadata:", data["metadata"])
    print("dialogues:", len(data["dialogues"]))

    first = data["dialogues"][0]

    print(
        "first feature:",
        first["utterance_features"].shape
    )
    print(
        "speaker_ids:",
        first["speaker_ids"].shape
    )
    print(
        "emotion_labels:",
        first["emotion_labels"].shape
    )
    print(
        "cause_labels:",
        first["cause_labels"].shape
    )
    print(
        "pair_labels:",
        first["pair_labels"].shape
    )