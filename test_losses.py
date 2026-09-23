import torch
from torch.utils.data import DataLoader

from dataset.dataset import ECFDataset
from dataset.collate import ECPECCollator

from models.base_model import ECPECBaseModel

from losses import (
    ECPECLoss,
    compute_pair_pos_weight,
)


def main():

    # --------------------------------------------------
    # 1. Device
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------
    # 2. Training dataset
    # --------------------------------------------------

    train_dataset = ECFDataset(
        "data/ECF/train.json",
        max_dialogue_length=40,
    )

    print(
        "Train dialogues:",
        len(train_dataset),
    )

    # --------------------------------------------------
    # 3. Compute global pair pos_weight
    #
    # IMPORTANT:
    # calculate from TRAIN set only
    # --------------------------------------------------

    pair_pos_weight = (
        compute_pair_pos_weight(
            train_dataset
        )
    )

    print(
        "Train pair pos_weight:",
        pair_pos_weight,
    )

    # Based on our previous statistics,
    # this should be approximately 17.69.
    assert pair_pos_weight > 1.0

    # --------------------------------------------------
    # 4. Collator
    # --------------------------------------------------

    collator = ECPECCollator(
        tokenizer_name=(
            "pretrained/roberta-base"
        ),
        max_utterance_length=128,
    )

    # --------------------------------------------------
    # 5. DataLoader
    #
    # Use batch_size=2 here to keep the backward
    # sanity test lightweight.
    # --------------------------------------------------

    loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    batch = next(iter(loader))

    # --------------------------------------------------
    # 6. Move tensors to device
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
        batch[key] = (
            batch[key].to(device)
        )

    # --------------------------------------------------
    # 7. Build model
    #
    # Freeze RoBERTa here because this test only needs
    # to verify loss/backward correctness.
    # This also saves GPU memory.
    # --------------------------------------------------

    model = ECPECBaseModel(
        encoder_name=(
            "pretrained/roberta-base"
        ),

        roberta_dropout=0.1,

        freeze_encoder=True,

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

    # --------------------------------------------------
    # 8. Build loss
    # --------------------------------------------------

    criterion = ECPECLoss(
        lambda_emotion=0.2,
        lambda_cause=0.2,
        pair_pos_weight=(
            pair_pos_weight
        ),
    ).to(device)

    model.train()

    # --------------------------------------------------
    # 9. Forward
    # --------------------------------------------------

    outputs = model(
        input_ids=batch["input_ids"],

        attention_mask=(
            batch["attention_mask"]
        ),

        speaker_ids=(
            batch["speaker_ids"]
        ),

        position_ids=(
            batch["position_ids"]
        ),

        utterance_mask=(
            batch["utterance_mask"]
        ),

        pair_mask=(
            batch["pair_mask"]
        ),
    )

    # --------------------------------------------------
    # 10. Compute loss
    # --------------------------------------------------

    losses = criterion(
        outputs=outputs,

        emotion_labels=(
            batch["emotion_labels"]
        ),

        cause_labels=(
            batch["cause_labels"]
        ),

        pair_labels=(
            batch["pair_labels"]
        ),

        utterance_mask=(
            batch["utterance_mask"]
        ),

        pair_mask=(
            batch["pair_mask"]
        ),
    )

    total_loss = losses["total_loss"]
    pair_loss = losses["pair_loss"]
    emotion_loss = losses["emotion_loss"]
    cause_loss = losses["cause_loss"]

    # --------------------------------------------------
    # 11. Print basic statistics
    # --------------------------------------------------

    print()
    print("===== Batch Statistics =====")

    print(
        "Input shape:",
        batch["input_ids"].shape,
    )

    print(
        "Valid utterances:",
        int(
            batch["utterance_mask"]
            .sum()
            .item()
        ),
    )

    print(
        "Valid pairs:",
        int(
            batch["pair_mask"]
            .sum()
            .item()
        ),
    )

    print(
        "Positive pairs:",
        int(
            batch["pair_labels"][
                batch["pair_mask"]
            ]
            .sum()
            .item()
        ),
    )

    # --------------------------------------------------
    # 12. Print losses
    # --------------------------------------------------

    print()
    print("===== Loss Check =====")

    print(
        "Pair loss:",
        pair_loss.item(),
    )

    print(
        "Emotion loss:",
        emotion_loss.item(),
    )

    print(
        "Cause loss:",
        cause_loss.item(),
    )

    print(
        "Total loss:",
        total_loss.item(),
    )

    # --------------------------------------------------
    # 13. Numerical checks
    # --------------------------------------------------

    print()
    print("===== Numerical Check =====")

    loss_dict = {
        "pair_loss": pair_loss,
        "emotion_loss": emotion_loss,
        "cause_loss": cause_loss,
        "total_loss": total_loss,
    }

    for name, value in loss_dict.items():

        finite = (
            torch.isfinite(value)
            .all()
            .item()
        )

        print(
            f"{name} finite:",
            finite,
        )

        assert finite

    # Loss should be positive for this real batch.
    assert pair_loss.item() > 0
    assert emotion_loss.item() > 0
    assert cause_loss.item() > 0
    assert total_loss.item() > 0

    # --------------------------------------------------
    # 14. Verify total-loss equation
    #
    # total =
    # pair
    # + 0.2 * emotion
    # + 0.2 * cause
    # --------------------------------------------------

    expected_total = (
        pair_loss
        + 0.2 * emotion_loss
        + 0.2 * cause_loss
    )

    difference = torch.abs(
        total_loss
        - expected_total
    )

    print()
    print("===== Formula Check =====")

    print(
        "Total loss difference:",
        difference.item(),
    )

    assert torch.allclose(
        total_loss,
        expected_total,
        atol=1e-6,
    )

    # --------------------------------------------------
    # 15. Backward test
    # --------------------------------------------------

    print()
    print("===== Backward Check =====")

    model.zero_grad(
        set_to_none=True
    )

    total_loss.backward()

    # Check a representative gradient from
    # the pair classifier.
    pair_gradient = (
        model
        .pair_classifier
        .classifier[0]
        .weight
        .grad
    )

    assert pair_gradient is not None

    gradient_finite = (
        torch.isfinite(
            pair_gradient
        )
        .all()
        .item()
    )

    gradient_norm = (
        pair_gradient
        .norm()
        .item()
    )

    print(
        "Pair classifier gradient exists:",
        pair_gradient is not None,
    )

    print(
        "Pair classifier gradient finite:",
        gradient_finite,
    )

    print(
        "Pair classifier gradient norm:",
        gradient_norm,
    )

    assert gradient_finite
    assert gradient_norm > 0

    # --------------------------------------------------
    # 16. Auxiliary head gradient checks
    # --------------------------------------------------

    emotion_gradient = (
        model
        .auxiliary_heads
        .emotion_head
        .classifier[0]
        .weight
        .grad
    )

    cause_gradient = (
        model
        .auxiliary_heads
        .cause_head
        .classifier[0]
        .weight
        .grad
    )

    assert emotion_gradient is not None
    assert cause_gradient is not None

    print(
        "Emotion head gradient norm:",
        emotion_gradient
        .norm()
        .item(),
    )

    print(
        "Cause head gradient norm:",
        cause_gradient
        .norm()
        .item(),
    )

    assert (
        torch.isfinite(
            emotion_gradient
        )
        .all()
    )

    assert (
        torch.isfinite(
            cause_gradient
        )
        .all()
    )

    # --------------------------------------------------
    # 17. RoBERTa freeze check
    # --------------------------------------------------

    encoder_has_grad = False

    for parameter in (
        model
        .utterance_encoder
        .encoder
        .parameters()
    ):

        if parameter.grad is not None:
            encoder_has_grad = True
            break

    print(
        "Frozen RoBERTa has gradient:",
        encoder_has_grad,
    )

    assert not encoder_has_grad

    # --------------------------------------------------
    # 18. Final result
    # --------------------------------------------------

    print()
    print(
        "Loss and backward test passed."
    )


if __name__ == "__main__":
    main()