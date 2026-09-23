import argparse
import json
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

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


PROJECT_ROOT = Path(__file__).resolve().parent


# =========================================================
# Command-line arguments
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ECPEC BaseModel"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help=(
            "Path to YAML config file. "
            "Example: config/config_v1.yaml"
        ),
    )

    return parser.parse_args()


# =========================================================
# Tee stdout
#
# Normal print() output:
#   console + train.log
#
# tqdm normally writes to stderr, so progress-bar refreshes
# will not pollute train.log.
# =========================================================

class TeeStdout:

    def __init__(
        self,
        terminal,
        log_file,
    ):
        self.terminal = terminal
        self.log_file = log_file

    def write(
        self,
        message,
    ):
        self.terminal.write(message)
        self.log_file.write(message)

        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        if hasattr(
            self.terminal,
            "isatty",
        ):
            return self.terminal.isatty()

        return False

    @property
    def encoding(self):
        return getattr(
            self.terminal,
            "encoding",
            "utf-8",
        )


# =========================================================
# Seed
# =========================================================

def set_seed(
    seed: int,
):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(
    worker_id,
):
    worker_seed = (
        torch.initial_seed()
        % (2 ** 32)
    )

    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =========================================================
# Config
# =========================================================

def load_config(
    config_path: Path,
):
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: "
            f"{config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(
            f"Empty config file: "
            f"{config_path}"
        )

    required_sections = [
        "data",
        "model",
        "training",
    ]

    for section in required_sections:

        if section not in config:
            raise KeyError(
                f"Missing config section: "
                f"{section}"
            )

    return config


def resolve_path(
    path_value,
):
    path = Path(
        path_value
    )

    if not path.is_absolute():
        path = (
            PROJECT_ROOT
            / path
        )

    return path.resolve()


# =========================================================
# Config validation
# =========================================================

def validate_config(
    config,
):
    data_config = config["data"]
    model_config = config["model"]
    training_config = (
        config["training"]
    )

    required_data = [
        "data_root",
        "train_file",
        "dev_file",
    ]

    required_model = [
        "encoder",
    ]

    required_training = [
        "pair_pos_weight",
    ]

    for key in required_data:

        if key not in data_config:
            raise KeyError(
                f"Missing data.{key} "
                f"in config."
            )

    for key in required_model:

        if key not in model_config:
            raise KeyError(
                f"Missing model.{key} "
                f"in config."
            )

    for key in required_training:

        if key not in training_config:
            raise KeyError(
                f"Missing training.{key} "
                f"in config. "
                f"Please explicitly set "
                f"'auto' or a numeric value."
            )


# =========================================================
# Parameters
# =========================================================

def count_parameters(
    model,
):
    total = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    return (
        total,
        trainable,
    )


def count_parameter_group(
    group,
):
    return sum(
        parameter.numel()
        for parameter
        in group["params"]
    )


# =========================================================
# DataLoader
# =========================================================

def build_dataloaders(
    train_dataset,
    dev_dataset,
    collator,
    batch_size,
    num_workers,
    device,
    seed,
):
    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    common_args = {
        "batch_size":
            batch_size,

        "num_workers":
            num_workers,

        "collate_fn":
            collator,

        "pin_memory":
            device.type == "cuda",
    }

    if num_workers > 0:

        common_args[
            "persistent_workers"
        ] = True

        common_args[
            "worker_init_fn"
        ] = seed_worker

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        generator=generator,
        **common_args,
    )

    dev_loader = DataLoader(
        dev_dataset,
        shuffle=False,
        drop_last=False,
        **common_args,
    )

    return (
        train_loader,
        dev_loader,
    )


# =========================================================
# Main training
# =========================================================

def run_training(
    config,
    config_path,
):

    # -----------------------------------------------------
    # Seed
    # -----------------------------------------------------

    seed = int(
        config.get(
            "seed",
            42,
        )
    )

    set_seed(seed)

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "=========================================="
    )
    print(
        "ECPEC BaseModel Formal Training"
    )
    print(
        "=========================================="
    )

    print(
        "Project root:",
        PROJECT_ROOT,
    )

    print(
        "Config:",
        config_path,
    )

    print(
        "Seed:",
        seed,
    )

    print(
        "Device:",
        device,
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

        gpu_memory = (
            torch.cuda
            .get_device_properties(
                0
            )
            .total_memory
            / (1024 ** 3)
        )

        print(
            "GPU memory:",
            f"{gpu_memory:.2f} GB",
        )

    # -----------------------------------------------------
    # Config sections
    # -----------------------------------------------------

    data_config = (
        config["data"]
    )

    model_config = (
        config["model"]
    )

    training_config = (
        config["training"]
    )

    evaluation_config = (
        config.get(
            "evaluation",
            {},
        )
    )

    # -----------------------------------------------------
    # Dataset paths
    # -----------------------------------------------------

    data_root = resolve_path(
        data_config[
            "data_root"
        ]
    )

    train_path = (
        data_root
        / data_config[
            "train_file"
        ]
    )

    dev_path = (
        data_root
        / data_config[
            "dev_file"
        ]
    )

    if not train_path.exists():

        raise FileNotFoundError(
            f"Train file not found: "
            f"{train_path}"
        )

    if not dev_path.exists():

        raise FileNotFoundError(
            f"Dev file not found: "
            f"{dev_path}"
        )

    max_utterance_length = int(
        data_config.get(
            "max_utterance_length",
            128,
        )
    )

    max_dialogue_length = int(
        data_config.get(
            "max_dialogue_length",
            40,
        )
    )

    num_workers = int(
        data_config.get(
            "num_workers",
            0,
        )
    )

    print()
    print(
        "===== Dataset Configuration ====="
    )

    print(
        "Train:",
        train_path,
    )

    print(
        "Dev:",
        dev_path,
    )

    print(
        "Max utterance length:",
        max_utterance_length,
    )

    print(
        "Max dialogue length:",
        max_dialogue_length,
    )

    print(
        "Num workers:",
        num_workers,
    )

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    train_dataset = ECFDataset(
        str(
            train_path
        ),
        max_dialogue_length=(
            max_dialogue_length
        ),
    )

    dev_dataset = ECFDataset(
        str(
            dev_path
        ),
        max_dialogue_length=(
            max_dialogue_length
        ),
    )

    print()
    print(
        "===== Dataset Statistics ====="
    )

    print(
        "Train dialogues:",
        len(
            train_dataset
        ),
    )

    print(
        "Dev dialogues:",
        len(
            dev_dataset
        ),
    )

    # -----------------------------------------------------
    # Pair positive weight
    #
    # Must be explicitly specified in YAML.
    # -----------------------------------------------------

    pair_pos_weight_config = (
        training_config[
            "pair_pos_weight"
        ]
    )

    if (
        isinstance(
            pair_pos_weight_config,
            str,
        )
        and
        pair_pos_weight_config.lower()
        == "auto"
    ):

        pair_pos_weight = (
            compute_pair_pos_weight(
                train_dataset
            )
        )

        pair_weight_mode = (
            "auto"
        )

    else:

        pair_pos_weight = float(
            pair_pos_weight_config
        )

        pair_weight_mode = (
            "manual"
        )

    if pair_pos_weight <= 0:

        raise ValueError(
            "pair_pos_weight "
            "must be > 0."
        )

    print(
        "Pair pos_weight mode:",
        pair_weight_mode,
    )

    print(
        "Pair pos_weight:",
        pair_pos_weight,
    )

    # -----------------------------------------------------
    # Encoder
    # -----------------------------------------------------

    encoder_path = resolve_path(
        model_config[
            "encoder"
        ]
    )

    if not encoder_path.exists():

        raise FileNotFoundError(
            "Encoder not found: "
            f"{encoder_path}"
        )

    print()
    print(
        "===== Encoder ====="
    )

    print(
        "RoBERTa:",
        encoder_path,
    )

    # -----------------------------------------------------
    # Collator
    # -----------------------------------------------------

    collator = ECPECCollator(
        tokenizer_name=str(
            encoder_path
        ),

        max_utterance_length=(
            max_utterance_length
        ),
    )

    # -----------------------------------------------------
    # DataLoader
    # -----------------------------------------------------

    batch_size = int(
        training_config.get(
            "batch_size",
            4,
        )
    )

    if batch_size <= 0:

        raise ValueError(
            "batch_size must be > 0."
        )

    (
        train_loader,
        dev_loader,
    ) = build_dataloaders(
        train_dataset=(
            train_dataset
        ),

        dev_dataset=(
            dev_dataset
        ),

        collator=collator,

        batch_size=(
            batch_size
        ),

        num_workers=(
            num_workers
        ),

        device=device,

        seed=seed,
    )

    print()
    print(
        "===== DataLoader ====="
    )

    print(
        "Batch size:",
        batch_size,
    )

    print(
        "Train batches:",
        len(
            train_loader
        ),
    )

    print(
        "Dev batches:",
        len(
            dev_loader
        ),
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    freeze_encoder = bool(
        training_config.get(
            "freeze_encoder",
            False,
        )
    )

    model = ECPECBaseModel(
        encoder_name=str(
            encoder_path
        ),

        roberta_dropout=float(
            model_config.get(
                "dropout",
                0.1,
            )
        ),

        freeze_encoder=(
            freeze_encoder
        ),

        dialogue_hidden=int(
            model_config.get(
                "dialogue_hidden",
                256,
            )
        ),

        speaker_embedding_dim=int(
            model_config.get(
                "speaker_embedding_dim",
                32,
            )
        ),

        position_embedding_dim=int(
            model_config.get(
                "position_embedding_dim",
                32,
            )
        ),

        max_speakers=int(
            model_config.get(
                "max_speakers",
                20,
            )
        ),

        max_positions=int(
            model_config.get(
                "max_positions",
                64,
            )
        ),

        dialogue_layers=int(
            model_config.get(
                "dialogue_layers",
                2,
            )
        ),

        dialogue_heads=int(
            model_config.get(
                "dialogue_heads",
                4,
            )
        ),

        dialogue_ffn=int(
            model_config.get(
                "dialogue_ffn",
                1024,
            )
        ),

        auxiliary_hidden=int(
            model_config.get(
                "auxiliary_hidden",
                128,
            )
        ),

        pair_hidden=int(
            model_config.get(
                "pair_hidden",
                256,
            )
        ),

        distance_embedding_dim=int(
            model_config.get(
                "distance_embedding_dim",
                32,
            )
        ),

        speaker_relation_embedding_dim=int(
            model_config.get(
                "speaker_relation_embedding_dim",
                16,
            )
        ),

        max_relative_distance=int(
            model_config.get(
                "max_relative_distance",
                15,
            )
        ),

        dropout=float(
            model_config.get(
                "dropout",
                0.1,
            )
        ),
    ).to(
        device
    )

    (
        total_parameters,
        trainable_parameters,
    ) = count_parameters(
        model
    )

    print()
    print(
        "===== Model ====="
    )

    print(
        "Total parameters:",
        f"{total_parameters:,}",
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}",
    )

    print(
        "RoBERTa trainable:",
        not freeze_encoder,
    )

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    lambda_emotion = float(
        training_config.get(
            "lambda_emotion",
            0.2,
        )
    )

    lambda_cause = float(
        training_config.get(
            "lambda_cause",
            0.4,
        )
    )

    criterion = ECPECLoss(
        lambda_emotion=(
            lambda_emotion
        ),

        lambda_cause=(
            lambda_cause
        ),

        pair_pos_weight=(
            pair_pos_weight
        ),
    ).to(
        device
    )

    print()
    print(
        "===== Loss ====="
    )

    print(
        "lambda_emotion:",
        lambda_emotion,
    )

    print(
        "lambda_cause:",
        lambda_cause,
    )

    print(
        "pair_pos_weight:",
        pair_pos_weight,
    )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    encoder_lr = float(
        training_config.get(
            "encoder_lr",
            2e-5,
        )
    )

    module_lr = float(
        training_config.get(
            "module_lr",
            1e-4,
        )
    )

    weight_decay = float(
        training_config.get(
            "weight_decay",
            0.01,
        )
    )

    optimizer = build_optimizer(
        model=model,

        encoder_lr=(
            encoder_lr
        ),

        module_lr=(
            module_lr
        ),

        weight_decay=(
            weight_decay
        ),
    )

    print()
    print(
        "===== Optimizer ====="
    )

    print(
        "Optimizer: AdamW"
    )

    print(
        "Encoder LR:",
        encoder_lr,
    )

    print(
        "Module LR:",
        module_lr,
    )

    print(
        "Weight decay:",
        weight_decay,
    )

    print(
        "Parameter groups:",
        len(
            optimizer.param_groups
        ),
    )

    for index, group in enumerate(
        optimizer.param_groups
    ):

        print(
            f"Group {index}: "
            f"lr={group['lr']} "
            f"parameters="
            f"{count_parameter_group(group):,}"
        )

    if not freeze_encoder:

        if len(
            optimizer.param_groups
        ) != 2:

            raise RuntimeError(
                "Expected two optimizer "
                "parameter groups when "
                "RoBERTa is trainable."
            )

    # -----------------------------------------------------
    # Training settings
    # -----------------------------------------------------

    epochs = int(
        training_config.get(
            "epochs",
            20,
        )
    )

    early_stop_patience = int(
        training_config.get(
            "early_stop_patience",
            4,
        )
    )

    max_grad_norm = float(
        training_config.get(
            "max_grad_norm",
            1.0,
        )
    )

    threshold = float(
        evaluation_config.get(
            "threshold",
            0.5,
        )
    )

    checkpoint_path = resolve_path(
        training_config.get(
            "checkpoint_path",
            "checkpoints/best_model.pt",
        )
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "===== Training Setup ====="
    )

    print(
        "Epochs:",
        epochs,
    )

    print(
        "Threshold:",
        threshold,
    )

    print(
        "Early stopping patience:",
        early_stop_patience,
    )

    print(
        "Gradient clipping:",
        max_grad_norm,
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    # -----------------------------------------------------
    # Save run config
    # -----------------------------------------------------

    run_config_path = (
        checkpoint_path.parent
        / "run_config.yaml"
    )

    with open(
        run_config_path,
        "w",
        encoding="utf-8",
    ) as f:

        yaml.safe_dump(
            config,
            f,
            allow_unicode=True,
            sort_keys=False,
        )

    # -----------------------------------------------------
    # Trainer
    # -----------------------------------------------------

    trainer = ECPECTrainer(
        model=model,

        criterion=(
            criterion
        ),

        optimizer=(
            optimizer
        ),

        device=device,

        threshold=(
            threshold
        ),

        max_grad_norm=(
            max_grad_norm
        ),

        early_stop_patience=(
            early_stop_patience
        ),

        checkpoint_path=str(
            checkpoint_path
        ),
    )

    # -----------------------------------------------------
    # Formal training
    # -----------------------------------------------------

    history = trainer.fit(
        train_loader=(
            train_loader
        ),

        dev_loader=(
            dev_loader
        ),

        epochs=(
            epochs
        ),
    )

    # -----------------------------------------------------
    # Save history
    # -----------------------------------------------------

    history_path = (
        checkpoint_path.parent
        / "training_history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # -----------------------------------------------------
    # Load best model
    # -----------------------------------------------------

    print()
    print(
        "===== Load Best Model ====="
    )

    checkpoint = (
        trainer.load_best_checkpoint(
            load_optimizer=False
        )
    )

    print(
        "Best epoch:",
        checkpoint[
            "epoch"
        ],
    )

    print(
        "Best Dev Pair-F1:",
        checkpoint[
            "best_pair_f1"
        ],
    )

    print(
        "Threshold:",
        checkpoint[
            "threshold"
        ],
    )

    # -----------------------------------------------------
    # Final best-model Dev evaluation
    # -----------------------------------------------------

    best_dev_result = (
        trainer.evaluate(
            dev_loader,
            show_progress=True,
        )
    )

    print()
    print(
        "===== Best Model Dev Result ====="
    )

    print(
        "Dev total loss:",
        f"{best_dev_result['losses']['total_loss']:.4f}",
    )

    print(
        "Dev pair loss:",
        f"{best_dev_result['losses']['pair_loss']:.4f}",
    )

    print(
        "Dev emotion loss:",
        f"{best_dev_result['losses']['emotion_loss']:.4f}",
    )

    print(
        "Dev cause loss:",
        f"{best_dev_result['losses']['cause_loss']:.4f}",
    )

    print(
        format_metrics(
            best_dev_result[
                "metrics"
            ]
        )
    )

    print()
    print(
        "=========================================="
    )

    print(
        "Formal training completed."
    )

    print(
        "Best epoch:",
        trainer.best_epoch,
    )

    print(
        "Best Dev Pair-F1:",
        f"{trainer.best_pair_f1:.4f}",
    )

    print(
        "Best checkpoint:",
        checkpoint_path,
    )

    print(
        "Training history:",
        history_path,
    )

    print(
        "Run config:",
        run_config_path,
    )

    print(
        "=========================================="
    )


# =========================================================
# Entry
# =========================================================

def main():

    args = parse_args()

    config_path = Path(
        args.config
    )

    if not config_path.is_absolute():

        config_path = (
            PROJECT_ROOT
            / config_path
        )

    config_path = (
        config_path.resolve()
    )

    config = load_config(
        config_path
    )

    validate_config(
        config
    )

    # -----------------------------------------------------
    # Logging path
    # -----------------------------------------------------

    logging_config = (
        config.get(
            "logging",
            {},
        )
    )

    default_log_dir = (
        Path("logs")
        / config_path.stem
    )

    log_dir = resolve_path(
        logging_config.get(
            "log_dir",
            str(
                default_log_dir
            ),
        )
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_filename = (
        logging_config.get(
            "log_file",
            "train.log",
        )
    )

    log_path = (
        log_dir
        / log_filename
    )

    original_stdout = (
        sys.stdout
    )

    # -----------------------------------------------------
    # Append mode:
    # existing logs will not be deleted.
    #
    # Change "a" to "w" if each run should overwrite.
    # -----------------------------------------------------

    log_handle = open(
        log_path,
        "a",
        encoding="utf-8",
        buffering=1,
    )

    sys.stdout = TeeStdout(
        terminal=(
            original_stdout
        ),
        log_file=(
            log_handle
        ),
    )

    try:

        print()
        print()
        print(
            "##########################################"
        )

        print(
            "New ECPEC training run"
        )

        print(
            "Config:",
            config_path,
        )

        print(
            "Log:",
            log_path,
        )

        print(
            "##########################################"
        )

        run_training(
            config=(
                config
            ),

            config_path=(
                config_path
            ),
        )

    except KeyboardInterrupt:

        print()
        print(
            "Training interrupted by user."
        )

        raise

    except Exception:

        error_text = (
            traceback.format_exc()
        )

        print()
        print(
            "Training failed."
        )

        print(
            error_text
        )

        raise

    finally:

        sys.stdout = (
            original_stdout
        )

        log_handle.flush()
        log_handle.close()


if __name__ == "__main__":
    main()