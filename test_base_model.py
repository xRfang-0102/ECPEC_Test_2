import torch
from torch.utils.data import DataLoader

from dataset.dataset import ECFDataset
from dataset.collate import ECPECCollator
from models.base_model import ECPECBaseModel


def main():

    # --------------------------------------------------
    # 1. Device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------
    # 2. Dataset
    # --------------------------------------------------

    dataset = ECFDataset(
        "data/ECF/train.json",
        max_dialogue_length=40,
    )

    # --------------------------------------------------
    # 3. Collator
    # --------------------------------------------------

    collator = ECPECCollator(
        tokenizer_name="pretrained/roberta-base",
        max_utterance_length=128,
    )

    # --------------------------------------------------
    # 4. DataLoader
    # --------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    batch = next(iter(loader))

    # --------------------------------------------------
    # 5. Move tensors to device
    # --------------------------------------------------

    tensor_keys = [
        "input_ids",
        "attention_mask",
        "speaker_ids",
        "position_ids",
        "utterance_mask",
        "emotion_labels",
        "cause_labels",
        "pair_labels",
        "pair_mask",
    ]

    for key in tensor_keys:
        batch[key] = batch[key].to(device)

    # --------------------------------------------------
    # 6. Build BaseModel
    # --------------------------------------------------

    model = ECPECBaseModel(
        encoder_name="pretrained/roberta-base",

        roberta_dropout=0.1,
        freeze_encoder=False,

        dialogue_hidden=256,

        speaker_embedding_dim=32,
        position_embedding_dim=32,

        max_speakers=20,
        max_positions=64,

        dialogue_layers=2,
        dialogue_heads=4,
        dialogue_ffn=1024,

        auxiliary_hidden=128,

        pair_hidden=256,

        distance_embedding_dim=32,
        speaker_relation_embedding_dim=16,
        max_relative_distance=15,

        dropout=0.1,
    ).to(device)

    model.eval()

    # --------------------------------------------------
    # 7. Forward
    # --------------------------------------------------

    with torch.no_grad():

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            speaker_ids=batch["speaker_ids"],
            position_ids=batch["position_ids"],
            utterance_mask=batch["utterance_mask"],
            pair_mask=batch["pair_mask"],
            return_features=True,
        )

    # --------------------------------------------------
    # 8. Get outputs
    # --------------------------------------------------

    emotion_logits = outputs["emotion_logits"]
    cause_logits = outputs["cause_logits"]
    pair_logits = outputs["pair_logits"]

    utterance_features = outputs["utterance_features"]
    context_features = outputs["context_features"]

    utterance_mask = batch["utterance_mask"]
    pair_mask = batch["pair_mask"]

    # --------------------------------------------------
    # 9. Shape checks
    # --------------------------------------------------

    print()
    print("===== Shape Check =====")

    print(
        "Input IDs:",
        batch["input_ids"].shape,
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
        "Emotion logits:",
        emotion_logits.shape,
    )

    print(
        "Cause logits:",
        cause_logits.shape,
    )

    print(
        "Pair logits:",
        pair_logits.shape,
    )

    # --------------------------------------------------
    # 10. Valid counts
    # --------------------------------------------------

    print()
    print("===== Mask Check =====")

    print(
        "Valid utterances:",
        int(
            utterance_mask.sum().item()
        ),
    )

    print(
        "Valid pairs:",
        int(
            pair_mask.sum().item()
        ),
    )

    print(
        "Gold positive pairs:",
        int(
            batch["pair_labels"]
            .sum()
            .item()
        ),
    )

    # --------------------------------------------------
    # 11. Padding check
    # --------------------------------------------------

    emotion_padding = (
        emotion_logits[
            ~utterance_mask
        ]
    )

    cause_padding = (
        cause_logits[
            ~utterance_mask
        ]
    )

    pair_padding = (
        pair_logits[
            ~pair_mask
        ]
    )

    context_padding = (
        context_features[
            ~utterance_mask
        ]
    )

    print()
    print("===== Padding Check =====")

    if emotion_padding.numel() > 0:
        print(
            "Emotion padding max abs:",
            emotion_padding
            .abs()
            .max()
            .item(),
        )

    if cause_padding.numel() > 0:
        print(
            "Cause padding max abs:",
            cause_padding
            .abs()
            .max()
            .item(),
        )

    if pair_padding.numel() > 0:
        print(
            "Pair padding max abs:",
            pair_padding
            .abs()
            .max()
            .item(),
        )

    if context_padding.numel() > 0:
        print(
            "Context padding max abs:",
            context_padding
            .abs()
            .max()
            .item(),
        )

    # --------------------------------------------------
    # 12. NaN / Inf check
    # --------------------------------------------------

    print()
    print("===== Numerical Check =====")

    tensors_to_check = {
        "utterance_features": utterance_features,
        "context_features": context_features,
        "emotion_logits": emotion_logits,
        "cause_logits": cause_logits,
        "pair_logits": pair_logits,
    }

    all_finite = True

    for name, tensor in tensors_to_check.items():

        finite = torch.isfinite(
            tensor
        ).all().item()

        print(
            f"{name} finite:",
            finite,
        )

        if not finite:
            all_finite = False

    # --------------------------------------------------
    # 13. Assertions
    # --------------------------------------------------

    batch_size = batch["input_ids"].size(0)
    max_dialogue_length = batch["input_ids"].size(1)

    assert (
        utterance_features.shape
        == (
            batch_size,
            max_dialogue_length,
            768,
        )
    )

    assert (
        context_features.shape
        == (
            batch_size,
            max_dialogue_length,
            256,
        )
    )

    assert (
        emotion_logits.shape
        == (
            batch_size,
            max_dialogue_length,
        )
    )

    assert (
        cause_logits.shape
        == (
            batch_size,
            max_dialogue_length,
        )
    )

    assert (
        pair_logits.shape
        == (
            batch_size,
            max_dialogue_length,
            max_dialogue_length,
        )
    )

    if emotion_padding.numel() > 0:
        assert torch.allclose(
            emotion_padding,
            torch.zeros_like(
                emotion_padding
            ),
        )

    if cause_padding.numel() > 0:
        assert torch.allclose(
            cause_padding,
            torch.zeros_like(
                cause_padding
            ),
        )

    if pair_padding.numel() > 0:
        assert torch.allclose(
            pair_padding,
            torch.zeros_like(
                pair_padding
            ),
        )

    if context_padding.numel() > 0:
        assert torch.allclose(
            context_padding,
            torch.zeros_like(
                context_padding
            ),
        )

    assert all_finite

    print()
    print(
        "BaseModel forward test passed."
    )


if __name__ == "__main__":
    main()