from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset.dataset import ECFDataset
from dataset.collate import ECPECCollator

from models.base_model import ECPECBaseModel

from losses import (
    ECPECLoss,
    compute_pair_pos_weight,
)

from trainer import (
    ECPECTrainer,
    build_optimizer,
)

from metrics import format_metrics


def main():

    # ==================================================
    # 1. Reproducibility
    # ==================================================

    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # ==================================================
    # 2. Device
    # ==================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # ==================================================
    # 3. Load full Train / Dev datasets
    # ==================================================

    train_dataset_full = ECFDataset(
        "data/ECF/train.json",
        max_dialogue_length=40,
    )

    dev_dataset_full = ECFDataset(
        "data/ECF/dev.json",
        max_dialogue_length=40,
    )

    print(
        "Full train dialogues:",
        len(train_dataset_full),
    )

    print(
        "Full dev dialogues:",
        len(dev_dataset_full),
    )

    # ==================================================
    # 4. Compute pair pos_weight
    #
    # IMPORTANT:
    # use the complete TRAIN set rather than the tiny
    # debugging subset.
    # ==================================================

    pair_pos_weight = (
        compute_pair_pos_weight(
            train_dataset_full
        )
    )

    print(
        "Train pair pos_weight:",
        pair_pos_weight,
    )

    assert pair_pos_weight > 1.0

    # ==================================================
    # 5. Create tiny subsets
    #
    # This is only a trainer sanity test.
    # Do NOT use these subsets for actual experiments.
    # ==================================================

    train_subset_size = min(
        8,
        len(train_dataset_full),
    )

    dev_subset_size = min(
        4,
        len(dev_dataset_full),
    )

    train_dataset = Subset(
        train_dataset_full,
        list(
            range(train_subset_size)
        ),
    )

    dev_dataset = Subset(
        dev_dataset_full,
        list(
            range(dev_subset_size)
        ),
    )

    print(
        "Trainer-test train dialogues:",
        len(train_dataset),
    )

    print(
        "Trainer-test dev dialogues:",
        len(dev_dataset),
    )

    # ==================================================
    # 6. Collator
    # ==================================================

    collator = ECPECCollator(
        tokenizer_name=(
            "pretrained/roberta-base"
        ),
        max_utterance_length=128,
    )

    # ==================================================
    # 7. DataLoaders
    # ==================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    print(
        "Train batches:",
        len(train_loader),
    )

    print(
        "Dev batches:",
        len(dev_loader),
    )

    assert len(train_loader) > 0
    assert len(dev_loader) > 0

    # ==================================================
    # 8. Build model
    #
    # Freeze RoBERTa ONLY for this trainer test.
    #
    # The goal is to test training logic quickly and
    # reduce GPU memory / runtime.
    #
    # Formal training should use:
    # freeze_encoder=False
    # ==================================================

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

    # ==================================================
    # 9. Build criterion
    #
    # Current Base setting:
    #
    # Pair weighted BCE
    # Emotion lambda = 0.2
    # Cause lambda   = 0.2
    #
    # If we later freeze lambda_cause=0.4,
    # change it consistently in config/train.py.
    # ==================================================

    criterion = ECPECLoss(
        lambda_emotion=0.2,
        lambda_cause=0.2,
        pair_pos_weight=(
            pair_pos_weight
        ),
    ).to(device)

    # ==================================================
    # 10. Build optimizer
    #
    # Because RoBERTa is frozen in this test,
    # only newly introduced modules are optimized.
    # ==================================================

    optimizer = build_optimizer(
        model=model,
        encoder_lr=2e-5,
        module_lr=1e-4,
        weight_decay=0.01,
    )

    print()
    print("===== Optimizer Check =====")

    print(
        "Optimizer parameter groups:",
        len(optimizer.param_groups),
    )

    for index, group in enumerate(
        optimizer.param_groups
    ):
        print(
            f"Group {index}: "
            f"lr={group['lr']} "
            f"params={len(group['params'])}"
        )

    assert len(
        optimizer.param_groups
    ) >= 1

    # ==================================================
    # 11. Test checkpoint path
    #
    # Keep it separate from the future real
    # checkpoints/best_model.pt.
    # ==================================================

    checkpoint_path = Path(
        "checkpoints/"
        "test_trainer_best_model.pt"
    )

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # ==================================================
    # 12. Build trainer
    # ==================================================

    trainer = ECPECTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,

        threshold=0.5,

        max_grad_norm=1.0,

        early_stop_patience=2,

        checkpoint_path=(
            checkpoint_path
        ),
    )

    # ==================================================
    # 13. Save representative parameter BEFORE training
    #
    # We use pair classifier weight to check that
    # optimizer.step() really changes parameters.
    # ==================================================

    parameter_before = (
        model
        .pair_classifier
        .classifier[0]
        .weight
        .detach()
        .clone()
    )

    # ==================================================
    # 14. Pre-training Dev evaluation
    # ==================================================

    print()
    print(
        "===== Pre-training Dev Check ====="
    )

    pre_dev_result = trainer.evaluate(
        dev_loader
    )

    print(
        "Dev loss:",
        pre_dev_result[
            "losses"
        ]["total_loss"],
    )

    print(
        format_metrics(
            pre_dev_result[
                "metrics"
            ]
        )
    )

    assert torch.isfinite(
        torch.tensor(
            pre_dev_result[
                "losses"
            ]["total_loss"]
        )
    )

    # ==================================================
    # 15. Train for 2 epochs
    # ==================================================

    print()
    print(
        "===== Trainer Fit Check ====="
    )

    history = trainer.fit(
        train_loader=train_loader,
        dev_loader=dev_loader,
        epochs=2,
    )

    # ==================================================
    # 16. History check
    # ==================================================

    print()
    print(
        "===== History Check ====="
    )

    print(
        "Completed epochs:",
        len(history),
    )

    assert len(history) >= 1
    assert len(history) <= 2

    for record in history:

        assert "epoch" in record
        assert "train" in record
        assert "dev" in record
        assert "is_best" in record

        train_loss = (
            record[
                "train"
            ]["losses"]["total_loss"]
        )

        dev_loss = (
            record[
                "dev"
            ]["losses"]["total_loss"]
        )

        assert torch.isfinite(
            torch.tensor(train_loss)
        )

        assert torch.isfinite(
            torch.tensor(dev_loss)
        )

    # ==================================================
    # 17. Parameter-update check
    # ==================================================

    parameter_after = (
        model
        .pair_classifier
        .classifier[0]
        .weight
        .detach()
        .clone()
    )

    parameter_difference = (
        parameter_after
        - parameter_before
    ).abs().max().item()

    print()
    print(
        "===== Parameter Update Check ====="
    )

    print(
        "Pair classifier max parameter change:",
        parameter_difference,
    )

    assert parameter_difference > 0.0

    # ==================================================
    # 18. Checkpoint check
    # ==================================================

    print()
    print(
        "===== Checkpoint Check ====="
    )

    print(
        "Checkpoint exists:",
        checkpoint_path.exists(),
    )

    assert checkpoint_path.exists()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    required_checkpoint_keys = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "best_pair_f1",
        "threshold",
        "dev_metrics",
        "dev_losses",
    }

    missing_keys = (
        required_checkpoint_keys
        - set(checkpoint.keys())
    )

    assert not missing_keys, (
        "Checkpoint missing keys: "
        f"{missing_keys}"
    )

    print(
        "Saved epoch:",
        checkpoint["epoch"],
    )

    print(
        "Saved best Pair-F1:",
        checkpoint[
            "best_pair_f1"
        ],
    )

    print(
        "Saved threshold:",
        checkpoint[
            "threshold"
        ],
    )

    assert (
        checkpoint["epoch"]
        >= 1
    )

    assert (
        checkpoint["best_pair_f1"]
        >= 0.0
    )

    # ==================================================
    # 19. Reload best checkpoint through Trainer
    # ==================================================

    print()
    print(
        "===== Reload Check ====="
    )

    loaded_checkpoint = (
        trainer.load_best_checkpoint(
            load_optimizer=False
        )
    )

    print(
        "Trainer best epoch:",
        trainer.best_epoch,
    )

    print(
        "Trainer best Pair-F1:",
        trainer.best_pair_f1,
    )

    assert (
        trainer.best_epoch
        ==
        loaded_checkpoint["epoch"]
    )

    # ==================================================
    # 20. Evaluate after loading best checkpoint
    # ==================================================

    print()
    print(
        "===== Reloaded Dev Evaluation ====="
    )

    reloaded_dev_result = (
        trainer.evaluate(
            dev_loader
        )
    )

    print(
        "Dev loss:",
        reloaded_dev_result[
            "losses"
        ]["total_loss"],
    )

    print(
        format_metrics(
            reloaded_dev_result[
                "metrics"
            ]
        )
    )

    reloaded_pair_f1 = (
        reloaded_dev_result[
            "metrics"
        ]["pair"]["f1"]
    )

    print(
        "Reloaded Dev Pair-F1:",
        reloaded_pair_f1,
    )

    # The checkpoint was selected by Dev Pair-F1.
    assert abs(
        reloaded_pair_f1
        - trainer.best_pair_f1
    ) < 1e-8

    # ==================================================
    # 21. Frozen encoder check
    # ==================================================

    print()
    print(
        "===== Encoder Freeze Check ====="
    )

    trainable_encoder_parameters = 0

    for parameter in (
        model
        .utterance_encoder
        .encoder
        .parameters()
    ):

        if parameter.requires_grad:
            trainable_encoder_parameters += (
                parameter.numel()
            )

    print(
        "Trainable RoBERTa parameters:",
        trainable_encoder_parameters,
    )

    assert (
        trainable_encoder_parameters
        == 0
    )

    # ==================================================
    # 22. Final result
    # ==================================================

    print()
    print(
        "===================================="
    )

    print(
        "Trainer test passed."
    )

    print(
        "Best epoch:",
        trainer.best_epoch,
    )

    print(
        "Best Dev Pair-F1:",
        trainer.best_pair_f1,
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    print(
        "===================================="
    )


if __name__ == "__main__":
    main()