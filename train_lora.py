# =========================================================
# v3: LoRA end-to-end ECPEC training.
#
# RoBERTa-base (LoRA on query/value) -> mean pooling ->
# dialogue Transformer + heads (+ optional A+C retrieval and
# locality prior) trained jointly on raw ECF text.
#
# Reuses train_feature's loss/metrics/EMA/eval machinery by
# monkey-patching the module-level forward_model and
# move_batch_to_device hooks to the LoRA shapes.
#
# Usage: python train_lora.py --config config/config_lora_ac2.yaml
# =========================================================

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

import train_feature as tf
from train_feature import (
    set_seed,
    train_one_epoch,
    evaluate,
    print_result,
)

from dataset.raw_ecf_dataset import RawECFDataset, collate_raw_batch
from models.lora_ecpec_model import LoRAECPECModel

PROJECT_ROOT = Path(__file__).resolve().parent


# =========================================================
# Args
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA end-to-end ECPEC training."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config.",
    )
    return parser.parse_args()


# =========================================================
# Hooks into train_feature (LoRA batch shapes)
# =========================================================

def make_forward_model_lora():
    def forward_model_lora(model, batch):
        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            speaker_ids=batch["speaker_ids"],
            position_ids=batch["position_ids"],
            utterance_mask=batch["utterance_mask"],
            pair_mask=batch["pair_mask"],
        )

    return forward_model_lora


def move_batch_to_device_lora(batch, device):
    moved = dict(batch)
    for key, value in list(moved.items()):
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
    return moved


# =========================================================
# Trainable-only EMA (avoids tracking the 125M frozen
# RoBERTa parameters)
# =========================================================

def update_ema_state_lora(model, ema_state, decay):
    if ema_state is None:
        ema_state = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
    else:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    ema_state[name].mul_(decay).add_(
                        param.data,
                        alpha=1.0 - decay,
                    )
    return ema_state


@contextlib.contextmanager
def ema_weights_context_lora(model, ema_state):
    if not ema_state:
        yield
        return

    backup = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(ema_state[name])

    try:
        yield
    finally:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(backup[name])


# =========================================================
# Null-loss option assembly (mirrors run_training)
# =========================================================

def build_null_loss_options(training_config, model):
    options = {}

    if model.pair_decision_mode in (
        "null_reference",
        "centered_null_reference",
        "hierarchical_boundary",
    ):
        options.update(
            {
                "null_rank_weight": float(
                    training_config.get("null_rank_weight", 0.2)
                ),
                "null_margin": float(
                    training_config.get("null_margin", 0.2)
                ),
                "null_hard_negative_k": int(
                    training_config.get("null_hard_negative_k", 3)
                ),
            }
        )

    options.update(
        {
            "pair_rank_weight": float(
                training_config.get("pair_rank_weight", 0.0)
            ),
            "pair_rank_margin": float(
                training_config.get("pair_rank_margin", 0.2)
            ),
            "pair_rank_hard_negative_k": int(
                training_config.get("pair_rank_hard_negative_k", 3)
            ),
        }
    )

    distance_weights = training_config.get(
        "pair_distance_weights",
        None,
    )

    if distance_weights is not None:
        options["pair_distance_weights"] = [
            float(weight) for weight in distance_weights
        ]

    options.update(
        {
            "cond_rank_weight": float(
                training_config.get("cond_rank_weight", 0.0)
            ),
            "cond_rank_margin": float(
                training_config.get("cond_rank_margin", 0.2)
            ),
            "cond_rank_hard_negative_k": int(
                training_config.get("cond_rank_hard_negative_k", 3)
            ),
            "retrieval_loss_weight": float(
                training_config.get("retrieval_loss_weight", 0.0)
            ),
            "retrieval_local_max_distance": (
                None
                if training_config.get(
                    "retrieval_local_max_distance"
                )
                is None
                else int(
                    training_config.get(
                        "retrieval_local_max_distance",
                        1,
                    )
                )
            ),
            "historical_retrieval_weight": float(
                training_config.get(
                    "historical_retrieval_weight",
                    0.0,
                )
            ),
            "locality_prior_reg_weight": float(
                training_config.get(
                    "locality_prior_reg_weight",
                    0.0,
                )
            ),
        }
    )

    return options


# =========================================================
# Checkpoint I/O
# =========================================================

def save_lora_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_pair_f1,
    pair_auprc,
    threshold,
    config,
    state_dict=None,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": (
                state_dict
                if state_dict is not None
                else model.state_dict()
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_pair_f1": best_pair_f1,
            "pair_auprc": float(pair_auprc),
            "threshold": threshold,
            "pair_decision_mode": model.pair_decision_mode,
            "emotion_threshold": 0.5,
            "cause_threshold": 0.5,
            "config": config,
            "model_type": "LoRAECPEC",
            "lora_r": model.lora_r,
            "lora_alpha": model.lora_alpha,
        },
        path,
    )


def load_lora_checkpoint(path, model, device):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    return checkpoint


# =========================================================
# Main
# =========================================================

def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config = tf.load_config(config_path)

    set_seed(int(config.get("seed", 42)))

    # cudnn deterministic mode makes RoBERTa-GEMMs crawl on
    # Blackwell; results stay seed-reproducible without it.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    eval_config = config.get("evaluation", {})
    logging_config = config.get("logging", {})

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    encoder_path = (
        PROJECT_ROOT
        / str(model_config.get("encoder", "pretrained/roberta-base"))
    )

    lora_r = int(model_config.get("lora_r", 16))
    lora_alpha = float(model_config.get("lora_alpha", 32))
    lora_dropout = float(model_config.get("lora_dropout", 0.1))
    use_bf16 = bool(training_config.get("use_bf16", True))

    model = LoRAECPECModel(
        encoder_path=encoder_path,
        model_config=model_config,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_bf16=use_bf16 and device.type == "cuda",
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("=" * 72)
    print("v3 LoRA end-to-end ECPEC training")
    print("=" * 72)
    print(f"Encoder: {encoder_path}")
    print(
        f"LoRA: r={lora_r} alpha={lora_alpha} "
        f"dropout={lora_dropout} "
        f"modules={model.num_lora_modules}"
    )
    print(
        f"Parameters: {total_params:,} total, "
        f"{trainable_params:,} trainable"
    )
    print(
        "Event memory retrieval:",
        "enabled" if model_config.get("use_event_retrieval")
        else "disabled",
    )
    print(
        "Locality prior head:",
        "enabled" if model_config.get("use_locality_prior")
        else "disabled",
    )

    # -----------------------------------------------------
    # Hooks
    # -----------------------------------------------------

    use_bf16 = bool(training_config.get("use_bf16", True))
    tf.forward_model = make_forward_model_lora()
    tf.move_batch_to_device = move_batch_to_device_lora

    # -----------------------------------------------------
    # Datasets / loaders
    # -----------------------------------------------------

    data_root = PROJECT_ROOT / data_config["data_root"]

    max_dialogue_length = int(
        data_config.get("max_dialogue_length", 40)
    )
    max_utterance_length = int(
        data_config.get("max_utterance_length", 128)
    )

    num_workers = int(data_config.get("num_workers", 0))

    def build_loader(split_file, shuffle):
        dataset = RawECFDataset(
            json_path=data_root / split_file,
            max_dialogue_length=max_dialogue_length,
            max_utterance_length=max_utterance_length,
            tokenizer_path=encoder_path,
        )
        return DataLoader(
            dataset,
            batch_size=int(
                training_config.get("batch_size", 8)
            ),
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_raw_batch,
            drop_last=False,
        )

    train_loader = build_loader(
        data_config["train_file"],
        shuffle=True,
    )
    dev_loader = build_loader(
        data_config["dev_file"],
        shuffle=False,
    )
    test_loader = build_loader(
        data_config["test_file"],
        shuffle=False,
    )

    # -----------------------------------------------------
    # Optimizer / schedule
    # -----------------------------------------------------

    trainable = [
        param
        for param in model.parameters()
        if param.requires_grad
    ]

    encoder_lr = training_config.get("encoder_lr")

    if (
        lora_r <= 0
        and encoder_lr is not None
    ):
        # Full fine-tuning: slower LR for the pretrained encoder,
        # faster LR for the task heads.
        encoder_params = [
            param
            for name, param in model.named_parameters()
            if name.startswith("encoder.")
            and param.requires_grad
        ]
        head_params = [
            param
            for name, param in model.named_parameters()
            if not name.startswith("encoder.")
            and param.requires_grad
        ]
        optimizer = AdamW(
            [
                {
                    "params": encoder_params,
                    "lr": float(encoder_lr),
                },
                {
                    "params": head_params,
                    "lr": float(
                        training_config.get("module_lr", 1e-4)
                    ),
                },
            ],
            weight_decay=float(
                training_config.get("weight_decay", 0.01)
            ),
        )
    else:
        optimizer = AdamW(
            trainable,
            lr=float(training_config.get("module_lr", 1e-4)),
            weight_decay=float(
                training_config.get("weight_decay", 0.01)
            ),
        )

    base_lrs = [
        group["lr"] for group in optimizer.param_groups
    ]

    epochs = int(training_config.get("epochs", 20))
    warmup_epochs = int(
        training_config.get("warmup_epochs", 2)
    )
    f1_smooth_window = int(
        training_config.get("f1_smooth_window", 3)
    )
    ema_decay = float(training_config.get("ema_decay", 0.9))
    ema_start_epoch = int(
        training_config.get("ema_start_epoch", 8)
    )
    early_stop_patience = int(
        training_config.get("early_stop_patience", 6)
    )

    pair_pos_weight = float(
        training_config.get("pair_pos_weight", 2.5)
    )
    lambda_emotion = float(
        training_config.get("lambda_emotion", 0.2)
    )
    lambda_cause = float(
        training_config.get("lambda_cause", 0.4)
    )
    max_grad_norm = float(
        training_config.get("max_grad_norm", 1.0)
    )
    threshold = float(
        eval_config.get("pair_threshold", 0.66)
    )
    row_normalize = bool(
        eval_config.get("row_normalize", False)
    )

    checkpoint_path = (
        PROJECT_ROOT
        / training_config.get(
            "checkpoint_path",
            "checkpoints/base_lora_best.pt",
        )
    )

    null_loss_options = build_null_loss_options(
        training_config,
        model,
    )

    print("=" * 72)
    print(
        "Training:",
        f"batch={int(training_config.get('batch_size', 8))}, "
        f"epochs={epochs}, warmup={warmup_epochs}, "
        f"bf16={use_bf16}, lr={base_lrs[0]:.0e}, "
        f"threshold={threshold}",
    )
    print("A+C objectives:", null_loss_options)
    print("=" * 72)

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------

    ema_state = None
    recent_dev_f1 = []
    best_pair_f1 = -float("inf")
    best_epoch = 0
    best_auprc = 0.0

    log_path = None
    if logging_config.get("log_file"):
        log_path = (
            PROJECT_ROOT / logging_config["log_file"]
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)

    history = []

    for epoch in range(1, epochs + 1):

        # Linear LR warmup
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            scale = max(epoch / warmup_epochs, 1e-4)
            for group, base_lr in zip(
                optimizer.param_groups,
                base_lrs,
            ):
                group["lr"] = base_lr * scale

        train_result = train_one_epoch(
            model=model,
            null_loss_options=null_loss_options,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            pair_pos_weight=pair_pos_weight,
            lambda_emotion=lambda_emotion,
            lambda_cause=lambda_cause,
            threshold=threshold,
            max_grad_norm=max_grad_norm,
            epoch=epoch,
            total_epochs=epochs,
            row_normalize=row_normalize,
        )

        # EMA update after the training epoch
        if ema_decay > 0 and epoch >= ema_start_epoch:
            ema_state = update_ema_state_lora(
                model,
                ema_state,
                ema_decay,
            )

        with ema_weights_context_lora(
            model,
            ema_state,
        ):
            dev_result = evaluate(
                model=model,
                null_loss_options=null_loss_options,
                loader=dev_loader,
                device=device,
                pair_pos_weight=pair_pos_weight,
                lambda_emotion=lambda_emotion,
                lambda_cause=lambda_cause,
                threshold=threshold,
                description="Dev",
                row_normalize=row_normalize,
            )

        print_result(
            epoch=epoch,
            total_epochs=epochs,
            train_result=train_result,
            dev_result=dev_result,
        )

        current_pair_f1 = float(
            dev_result["metrics"]["pair"]["f1"]
        )
        current_pair_auprc = float(
            dev_result["pair_auprc"]
        )

        recent_dev_f1.append(current_pair_f1)
        recent_dev_f1 = recent_dev_f1[-f1_smooth_window:]
        smoothed_pair_f1 = (
            sum(recent_dev_f1) / len(recent_dev_f1)
        )

        print(
            f"Dev Pair F1 (smoothed@{f1_smooth_window}): "
            f"{smoothed_pair_f1:.4f} "
            f"(raw {current_pair_f1:.4f}) | "
            f"AUPRC {current_pair_auprc:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(
                    train_result["losses"]["total_loss"]
                ),
                "dev_pair_f1": current_pair_f1,
                "dev_pair_auprc": current_pair_auprc,
                "smoothed_pair_f1": smoothed_pair_f1,
            }
        )

        if log_path is not None:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(history[-1]) + "\n")

        if smoothed_pair_f1 > best_pair_f1:
            best_pair_f1 = smoothed_pair_f1
            best_epoch = epoch
            best_auprc = current_pair_auprc

            state_dict = None
            if ema_state is not None:
                # EMA tracks only trainable params; merge them
                # into a FULL state dict so strict loading works.
                state_dict = model.state_dict()
                state_dict.update(ema_state)

            save_lora_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                best_pair_f1,
                current_pair_auprc,
                threshold,
                config,
                state_dict=state_dict,
            )

            print(
                "New best Dev Pair-F1 (smoothed): "
                f"{best_pair_f1:.4f} (epoch {epoch}) | "
                f"AUPRC {best_auprc:.4f}"
            )
            print(
                "Checkpoint saved to: "
                f"{checkpoint_path}"
            )

        if (
            early_stop_patience > 0
            and epoch - best_epoch >= early_stop_patience
        ):
            print(
                f"Early stop: no improvement for "
                f"{early_stop_patience} epochs "
                f"(best epoch {best_epoch})."
            )
            break

    # -----------------------------------------------------
    # Final: load best checkpoint, evaluate dev + test
    # -----------------------------------------------------

    print("=" * 72)
    print("===== Load Best Model =====")
    print(f"Best epoch: {best_epoch}")
    print(f"Best Dev Pair-F1: {best_pair_f1:.4f}")
    print(f"Best checkpoint Pair AUPRC: {best_auprc:.4f}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint missing: {checkpoint_path}"
        )

    load_lora_checkpoint(
        checkpoint_path,
        model,
        device,
    )

    model.eval()

    dev_result = evaluate(
        model=model,
        null_loss_options=null_loss_options,
        loader=dev_loader,
        device=device,
        pair_pos_weight=pair_pos_weight,
        lambda_emotion=lambda_emotion,
        lambda_cause=lambda_cause,
        threshold=threshold,
        description="Dev",
        row_normalize=row_normalize,
    )

    test_result = evaluate(
        model=model,
        null_loss_options=null_loss_options,
        loader=test_loader,
        device=device,
        pair_pos_weight=pair_pos_weight,
        lambda_emotion=lambda_emotion,
        lambda_cause=lambda_cause,
        threshold=threshold,
        description="Test",
        row_normalize=row_normalize,
    )

    print("=" * 72)
    print("Best Checkpoint Final Results")
    print("=" * 72)

    for name, result in (
        ("Dev", dev_result),
        ("Test", test_result),
    ):
        pair = result["metrics"]["pair"]
        emotion = result["metrics"]["emotion"]
        cause = result["metrics"]["cause"]

        print(
            f"{name} Emotion | P={emotion['precision']:.4f} "
            f"R={emotion['recall']:.4f} F1={emotion['f1']:.4f}"
        )
        print(
            f"{name} Cause   | P={cause['precision']:.4f} "
            f"R={cause['recall']:.4f} F1={cause['f1']:.4f}"
        )
        print(
            f"{name} Pair    | P={pair['precision']:.4f} "
            f"R={pair['recall']:.4f} F1={pair['f1']:.4f}"
        )
        print(
            f"{name} Pair AUPRC | {result['pair_auprc']:.4f}"
        )

    print("=" * 72)
    print(
        f"Trainable parameters: {trainable_params:,}"
    )
    print(f"Best epoch: {best_epoch}")
    print(
        f"Best Pair-F1 (smoothed@{f1_smooth_window}): "
        f"{best_pair_f1:.4f}"
    )
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
