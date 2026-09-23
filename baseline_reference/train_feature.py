import argparse
import json
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from sklearn.metrics import average_precision_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.feature_dataset import FeatureECFDataset
from dataset.feature_collate import FeatureECPECCollator

# 根据你现在实际使用的文件名导入
try:
    from models.feature_base_model import FeatureECPECBaseModel
except ModuleNotFoundError:
    from models.feature_best_model import FeatureECPECBaseModel


PROJECT_ROOT = Path(__file__).resolve().parent


# =========================================================
# Arguments
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train ECPEC BaseModel using "
            "pre-extracted frozen RoBERTa features."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config_feature_pos2.5_thr0.66.yaml",
        help="Path to feature-training YAML config.",
    )

    return parser.parse_args()


# =========================================================
# Tee stdout
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
# Utilities
# =========================================================

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


def load_config(
    config_path,
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
        config = yaml.safe_load(
            f
        )

    if config is None:
        raise ValueError(
            f"Empty config file: "
            f"{config_path}"
        )

    for section in [
        "data",
        "model",
        "training",
    ]:
        if section not in config:
            raise KeyError(
                f"Missing config section: "
                f"{section}"
            )

    return config


def validate_config(
    config,
):
    data_config = (
        config["data"]
    )

    training_config = (
        config["training"]
    )

    required_data = [
        "feature_root",
        "train_feature_file",
        "dev_feature_file",
    ]

    for key in required_data:
        if key not in data_config:
            raise KeyError(
                f"Missing data.{key}"
            )

    if (
        "pair_pos_weight"
        not in training_config
    ):
        raise KeyError(
            "Missing "
            "training.pair_pos_weight. "
            "Explicitly set a numeric value "
            "or 'auto'."
        )


def set_seed(
    seed,
):
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed(
            seed
        )

        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(
    worker_id,
):
    worker_seed = (
        torch.initial_seed()
        % (2 ** 32)
    )

    random.seed(
        worker_seed
    )

    np.random.seed(
        worker_seed
    )


def count_parameters(
    model,
):
    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return (
        total,
        trainable,
    )


# =========================================================
# Move batch to device
# =========================================================

def move_batch_to_device(
    batch,
    device,
):
    tensor_keys = [
        "utterance_features",
        "speaker_ids",
        "position_ids",
        "utterance_mask",
        "emotion_labels",
        "cause_labels",
        "pair_labels",
        "pair_mask",
        "num_utterances",
    ]

    moved = dict(
        batch
    )

    for key in tensor_keys:

        if key in moved:
            moved[key] = (
                moved[key].to(
                    device,
                    non_blocking=True,
                )
            )

    return moved


# =========================================================
# Loss
# =========================================================

def masked_bce(
    logits,
    labels,
    mask,
    pos_weight=None,
):
    mask = mask.bool()

    if mask.sum().item() == 0:
        return (
            logits.sum()
            * 0.0
        )

    valid_logits = logits[
        mask
    ]

    valid_labels = labels[
        mask
    ].float()

    if pos_weight is None:

        return (
            F.binary_cross_entropy_with_logits(
                valid_logits,
                valid_labels,
            )
        )

    weight_tensor = torch.tensor(
        float(
            pos_weight
        ),
        dtype=valid_logits.dtype,
        device=valid_logits.device,
    )

    return (
        F.binary_cross_entropy_with_logits(
            valid_logits,
            valid_labels,
            pos_weight=weight_tensor,
        )
    )


def compute_losses(
    outputs,
    batch,
    pair_pos_weight,
    lambda_emotion,
    lambda_cause,
):
    emotion_loss = masked_bce(
        logits=outputs[
            "emotion_logits"
        ],
        labels=batch[
            "emotion_labels"
        ],
        mask=batch[
            "utterance_mask"
        ],
    )

    cause_loss = masked_bce(
        logits=outputs[
            "cause_logits"
        ],
        labels=batch[
            "cause_labels"
        ],
        mask=batch[
            "utterance_mask"
        ],
    )

    pair_loss = masked_bce(
        logits=outputs[
            "pair_logits"
        ],
        labels=batch[
            "pair_labels"
        ],
        mask=batch[
            "pair_mask"
        ],
        pos_weight=(
            pair_pos_weight
        ),
    )

    total_loss = (
        pair_loss
        + lambda_emotion
        * emotion_loss
        + lambda_cause
        * cause_loss
    )

    return {
        "total_loss":
            total_loss,

        "pair_loss":
            pair_loss,

        "emotion_loss":
            emotion_loss,

        "cause_loss":
            cause_loss,
    }


# =========================================================
# Metrics
# =========================================================

class BinaryMetricAccumulator:

    def __init__(
        self,
        threshold=0.5,
    ):
        self.threshold = float(
            threshold
        )

        self.reset()

    def reset(
        self,
    ):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    @torch.no_grad()
    def update(
        self,
        logits,
        labels,
        mask,
    ):
        mask = mask.bool()

        if (
            mask.sum().item()
            == 0
        ):
            return

        logits = logits[
            mask
        ]

        labels = labels[
            mask
        ]

        probabilities = (
            torch.sigmoid(
                logits
            )
        )

        predictions = (
            probabilities
            >= self.threshold
        )

        gold = (
            labels
            >= 0.5
        )

        self.tp += int(
            (
                predictions
                & gold
            )
            .sum()
            .item()
        )

        self.fp += int(
            (
                predictions
                & (~gold)
            )
            .sum()
            .item()
        )

        self.fn += int(
            (
                (~predictions)
                & gold
            )
            .sum()
            .item()
        )

        self.tn += int(
            (
                (~predictions)
                & (~gold)
            )
            .sum()
            .item()
        )

    def compute(
        self,
    ):
        predicted_positive = (
            self.tp
            + self.fp
        )

        gold_positive = (
            self.tp
            + self.fn
        )

        total = (
            self.tp
            + self.fp
            + self.fn
            + self.tn
        )

        precision = (
            self.tp
            / predicted_positive
            if predicted_positive > 0
            else 0.0
        )

        recall = (
            self.tp
            / gold_positive
            if gold_positive > 0
            else 0.0
        )

        f1 = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
            )
            if (
                precision
                + recall
            ) > 0
            else 0.0
        )

        accuracy = (
            (
                self.tp
                + self.tn
            )
            / total
            if total > 0
            else 0.0
        )

        return {
            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "accuracy":
                accuracy,

            "tp":
                self.tp,

            "fp":
                self.fp,

            "fn":
                self.fn,

            "tn":
                self.tn,

            "predicted_positive":
                predicted_positive,

            "gold_positive":
                gold_positive,
        }


class ECPECMetricAccumulator:

    def __init__(
        self,
        threshold=0.5,
    ):
        self.emotion = (
            BinaryMetricAccumulator(
                threshold
            )
        )

        self.cause = (
            BinaryMetricAccumulator(
                threshold
            )
        )

        self.pair = (
            BinaryMetricAccumulator(
                threshold
            )
        )

    @torch.no_grad()
    def update(
        self,
        outputs,
        batch,
    ):
        self.emotion.update(
            outputs[
                "emotion_logits"
            ],
            batch[
                "emotion_labels"
            ],
            batch[
                "utterance_mask"
            ],
        )

        self.cause.update(
            outputs[
                "cause_logits"
            ],
            batch[
                "cause_labels"
            ],
            batch[
                "utterance_mask"
            ],
        )

        self.pair.update(
            outputs[
                "pair_logits"
            ],
            batch[
                "pair_labels"
            ],
            batch[
                "pair_mask"
            ],
        )

    def compute(
        self,
    ):
        return {
            "emotion":
                self.emotion.compute(),

            "cause":
                self.cause.compute(),

            "pair":
                self.pair.compute(),
        }


# =========================================================
# Loss accumulator
# =========================================================

class LossAccumulator:

    def __init__(
        self,
    ):
        self.reset()

    def reset(
        self,
    ):
        self.total_loss = 0.0
        self.pair_loss = 0.0
        self.emotion_loss = 0.0
        self.cause_loss = 0.0
        self.num_batches = 0

    def update(
        self,
        losses,
    ):
        self.total_loss += float(
            losses[
                "total_loss"
            ]
            .detach()
            .item()
        )

        self.pair_loss += float(
            losses[
                "pair_loss"
            ]
            .detach()
            .item()
        )

        self.emotion_loss += float(
            losses[
                "emotion_loss"
            ]
            .detach()
            .item()
        )

        self.cause_loss += float(
            losses[
                "cause_loss"
            ]
            .detach()
            .item()
        )

        self.num_batches += 1

    def compute(
        self,
    ):
        denominator = max(
            self.num_batches,
            1,
        )

        return {
            "total_loss":
                self.total_loss
                / denominator,

            "pair_loss":
                self.pair_loss
                / denominator,

            "emotion_loss":
                self.emotion_loss
                / denominator,

            "cause_loss":
                self.cause_loss
                / denominator,
        }


# =========================================================
# Model forward
# =========================================================

def forward_model(
    model,
    batch,
):
    outputs = model(
        utterance_features=batch[
            "utterance_features"
        ],

        speaker_ids=batch[
            "speaker_ids"
        ],

        position_ids=batch[
            "position_ids"
        ],

        utterance_mask=batch[
            "utterance_mask"
        ],

        pair_mask=batch[
            "pair_mask"
        ],
    )

    if not isinstance(
        outputs,
        dict,
    ):
        raise TypeError(
            "FeatureECPECBaseModel must "
            "return a dictionary."
        )

    required = [
        "emotion_logits",
        "cause_logits",
        "pair_logits",
    ]

    for key in required:

        if key not in outputs:
            raise KeyError(
                f"Model output missing: "
                f"{key}"
            )

    return outputs


# =========================================================
# One training epoch
# =========================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    pair_pos_weight,
    lambda_emotion,
    lambda_cause,
    threshold,
    max_grad_norm,
    epoch,
    total_epochs,
):
    model.train()

    metric_accumulator = (
        ECPECMetricAccumulator(
            threshold=threshold
        )
    )

    loss_accumulator = (
        LossAccumulator()
    )

    progress = tqdm(
        loader,
        desc=(
            f"Epoch "
            f"{epoch:02d}/"
            f"{total_epochs:02d} "
            f"[Train]"
        ),
        dynamic_ncols=True,
    )

    for batch in progress:

        batch = (
            move_batch_to_device(
                batch,
                device,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = (
            forward_model(
                model,
                batch,
            )
        )

        losses = (
            compute_losses(
                outputs=outputs,
                batch=batch,

                pair_pos_weight=(
                    pair_pos_weight
                ),

                lambda_emotion=(
                    lambda_emotion
                ),

                lambda_cause=(
                    lambda_cause
                ),
            )
        )

        losses[
            "total_loss"
        ].backward()

        if (
            max_grad_norm
            is not None
            and
            max_grad_norm > 0
        ):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )

        optimizer.step()

        loss_accumulator.update(
            losses
        )

        metric_accumulator.update(
            outputs,
            batch,
        )

        progress.set_postfix(
            {
                "loss":
                    f"{losses['total_loss'].item():.4f}",

                "pair":
                    f"{losses['pair_loss'].item():.4f}",
            }
        )

    return {
        "losses":
            loss_accumulator.compute(),

        "metrics":
            metric_accumulator.compute(),
    }


# =========================================================
# Evaluation
#
# IMPORTANT:
# AUPRC is calculated globally over all valid Dev pairs.
# It is NOT averaged batch-by-batch.
# =========================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    pair_pos_weight,
    lambda_emotion,
    lambda_cause,
    threshold,
    description="Dev",
):
    model.eval()

    metric_accumulator = (
        ECPECMetricAccumulator(
            threshold=threshold
        )
    )

    loss_accumulator = (
        LossAccumulator()
    )

    # -----------------------------------------------------
    # AUPRC:
    # collect all valid Pair logits / labels
    # across the entire evaluation set
    # -----------------------------------------------------

    all_pair_logits = []
    all_pair_labels = []

    progress = tqdm(
        loader,
        desc=(
            f"[{description}]"
        ),
        dynamic_ncols=True,
    )

    for batch in progress:

        batch = (
            move_batch_to_device(
                batch,
                device,
            )
        )

        outputs = (
            forward_model(
                model,
                batch,
            )
        )

        losses = (
            compute_losses(
                outputs=outputs,
                batch=batch,

                pair_pos_weight=(
                    pair_pos_weight
                ),

                lambda_emotion=(
                    lambda_emotion
                ),

                lambda_cause=(
                    lambda_cause
                ),
            )
        )

        loss_accumulator.update(
            losses
        )

        metric_accumulator.update(
            outputs,
            batch,
        )

        # -------------------------------------------------
        # Collect valid Pair logits for global AUPRC
        # -------------------------------------------------

        pair_mask = (
            batch[
                "pair_mask"
            ]
            .bool()
        )

        valid_pair_logits = (
            outputs[
                "pair_logits"
            ][
                pair_mask
            ]
        )

        valid_pair_labels = (
            batch[
                "pair_labels"
            ][
                pair_mask
            ]
        )

        all_pair_logits.append(
            valid_pair_logits
            .detach()
            .float()
            .cpu()
        )

        all_pair_labels.append(
            valid_pair_labels
            .detach()
            .float()
            .cpu()
        )

        progress.set_postfix(
            {
                "loss":
                    f"{losses['total_loss'].item():.4f}"
            }
        )

    # -----------------------------------------------------
    # Global Pair AUPRC
    # -----------------------------------------------------

    if len(
        all_pair_logits
    ) == 0:
        raise RuntimeError(
            "No valid Pair logits "
            "collected during evaluation."
        )

    all_pair_logits = torch.cat(
        all_pair_logits,
        dim=0,
    )

    all_pair_labels = torch.cat(
        all_pair_labels,
        dim=0,
    )

    if (
        all_pair_logits.numel()
        != all_pair_labels.numel()
    ):
        raise RuntimeError(
            "Pair logits / labels "
            "count mismatch."
        )

    all_pair_probs = (
        torch.sigmoid(
            all_pair_logits
        )
    )

    pair_auprc = (
        average_precision_score(
            all_pair_labels.numpy(),
            all_pair_probs.numpy(),
        )
    )

    return {
        "losses":
            loss_accumulator.compute(),

        "metrics":
            metric_accumulator.compute(),

        "pair_auprc":
            float(
                pair_auprc
            ),

        "num_pair_samples":
            int(
                all_pair_labels.numel()
            ),

        "num_positive_pairs":
            int(
                all_pair_labels
                .sum()
                .item()
            ),
    }


# =========================================================
# Pretty printing
# =========================================================

def print_result(
    epoch,
    total_epochs,
    train_result,
    dev_result,
):
    train_loss = (
        train_result[
            "losses"
        ]
    )

    dev_loss = (
        dev_result[
            "losses"
        ]
    )

    train_metrics = (
        train_result[
            "metrics"
        ]
    )

    dev_metrics = (
        dev_result[
            "metrics"
        ]
    )

    print()
    print(
        "=" * 44
    )

    print(
        f"Epoch "
        f"{epoch:02d}/"
        f"{total_epochs:02d}"
    )

    print(
        "-" * 44
    )

    print(
        "Train Loss: "
        f"{train_loss['total_loss']:.4f} "
        "("
        f"Pair={train_loss['pair_loss']:.4f}, "
        f"Emotion={train_loss['emotion_loss']:.4f}, "
        f"Cause={train_loss['cause_loss']:.4f}"
        ")"
    )

    emotion = (
        train_metrics[
            "emotion"
        ]
    )

    cause = (
        train_metrics[
            "cause"
        ]
    )

    pair = (
        train_metrics[
            "pair"
        ]
    )

    print(
        "Train Emotion | "
        f"P={emotion['precision']:.4f} "
        f"R={emotion['recall']:.4f} "
        f"F1={emotion['f1']:.4f}"
    )

    print(
        "Train Cause   | "
        f"P={cause['precision']:.4f} "
        f"R={cause['recall']:.4f} "
        f"F1={cause['f1']:.4f}"
    )

    print(
        "Train Pair    | "
        f"P={pair['precision']:.4f} "
        f"R={pair['recall']:.4f} "
        f"F1={pair['f1']:.4f} "
        f"#Pred={pair['predicted_positive']} "
        f"#Gold={pair['gold_positive']}"
    )

    print()

    print(
        "Dev   Loss: "
        f"{dev_loss['total_loss']:.4f} "
        "("
        f"Pair={dev_loss['pair_loss']:.4f}, "
        f"Emotion={dev_loss['emotion_loss']:.4f}, "
        f"Cause={dev_loss['cause_loss']:.4f}"
        ")"
    )

    emotion = (
        dev_metrics[
            "emotion"
        ]
    )

    cause = (
        dev_metrics[
            "cause"
        ]
    )

    pair = (
        dev_metrics[
            "pair"
        ]
    )

    print(
        "Dev   Emotion | "
        f"P={emotion['precision']:.4f} "
        f"R={emotion['recall']:.4f} "
        f"F1={emotion['f1']:.4f}"
    )

    print(
        "Dev   Cause   | "
        f"P={cause['precision']:.4f} "
        f"R={cause['recall']:.4f} "
        f"F1={cause['f1']:.4f}"
    )

    print(
        "Dev   Pair    | "
        f"P={pair['precision']:.4f} "
        f"R={pair['recall']:.4f} "
        f"F1={pair['f1']:.4f} "
        f"#Pred={pair['predicted_positive']} "
        f"#Gold={pair['gold_positive']}"
    )

    # -----------------------------------------------------
    # New: threshold-free Pair ranking metric
    # -----------------------------------------------------

    print(
        "Dev   Pair AUPRC | "
        f"{dev_result['pair_auprc']:.4f} "
        f"(N={dev_result['num_pair_samples']}, "
        f"Pos={dev_result['num_positive_pairs']})"
    )


# =========================================================
# Checkpoint
# =========================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_pair_f1,
    threshold,
    config,
    pair_auprc,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "best_pair_f1":
                best_pair_f1,

            "pair_auprc":
                float(
                    pair_auprc
                ),

            "threshold":
                threshold,

            "config":
                config,

            "model_type":
                "FrozenFeatureECPEC",
        },
        path,
    )


def load_checkpoint(
    path,
    model,
    device,
):
    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:
        checkpoint = torch.load(
            path,
            map_location=device,
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    return checkpoint


# =========================================================
# History
# =========================================================

def make_history_entry(
    epoch,
    train_result,
    dev_result,
):
    return {
        "epoch":
            int(
                epoch
            ),

        "train":
            train_result,

        "dev":
            dev_result,
    }


def save_history(
    history,
    history_path,
):
    history_path.parent.mkdir(
        parents=True,
        exist_ok=True,
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
    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        seed
    )

    common = {
        "batch_size":
            batch_size,

        "num_workers":
            num_workers,

        "collate_fn":
            collator,

        "pin_memory":
            device.type
            == "cuda",
    }

    if num_workers > 0:

        common[
            "persistent_workers"
        ] = True

        common[
            "worker_init_fn"
        ] = seed_worker

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        generator=generator,
        **common,
    )

    dev_loader = DataLoader(
        dev_dataset,
        shuffle=False,
        drop_last=False,
        **common,
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
    data_config = (
        config[
            "data"
        ]
    )

    model_config = (
        config[
            "model"
        ]
    )

    training_config = (
        config[
            "training"
        ]
    )

    evaluation_config = (
        config.get(
            "evaluation",
            {},
        )
    )

    # -----------------------------------------------------
    # Seed / Device
    # -----------------------------------------------------

    seed = int(
        config.get(
            "seed",
            42,
        )
    )

    set_seed(
        seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "=" * 52
    )

    print(
        "ECPEC Frozen-Feature Base Training"
    )

    print(
        "=" * 52
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

    if (
        device.type
        == "cuda"
    ):
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
            / 1024
            / 1024
            / 1024
        )

        print(
            "GPU memory:",
            f"{gpu_memory:.2f} GB",
        )

    # -----------------------------------------------------
    # Feature paths
    # -----------------------------------------------------

    feature_root = (
        resolve_path(
            data_config[
                "feature_root"
            ]
        )
    )

    train_feature_path = (
        feature_root
        / data_config[
            "train_feature_file"
        ]
    ).resolve()

    dev_feature_path = (
        feature_root
        / data_config[
            "dev_feature_file"
        ]
    ).resolve()

    if not (
        train_feature_path.exists()
    ):
        raise FileNotFoundError(
            "Train feature not found: "
            f"{train_feature_path}"
        )

    if not (
        dev_feature_path.exists()
    ):
        raise FileNotFoundError(
            "Dev feature not found: "
            f"{dev_feature_path}"
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
            4,
        )
    )

    hidden_size = int(
        model_config.get(
            "input_dim",
            768,
        )
    )

    print()
    print(
        "===== Frozen Features ====="
    )

    print(
        "Train:",
        train_feature_path,
    )

    print(
        "Dev:",
        dev_feature_path,
    )

    print(
        "Feature dim:",
        hidden_size,
    )

    print(
        "Max dialogue length:",
        max_dialogue_length,
    )

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    train_dataset = (
        FeatureECFDataset(
            feature_path=(
                train_feature_path
            ),

            max_dialogue_length=(
                max_dialogue_length
            ),

            expected_hidden_size=(
                hidden_size
            ),

            validate=True,
        )
    )

    dev_dataset = (
        FeatureECFDataset(
            feature_path=(
                dev_feature_path
            ),

            max_dialogue_length=(
                max_dialogue_length
            ),

            expected_hidden_size=(
                hidden_size
            ),

            validate=True,
        )
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
        "Train utterances:",
        train_dataset.num_utterances,
    )

    print(
        "Train candidate pairs:",
        train_dataset.num_candidate_pairs,
    )

    print(
        "Train positive pairs:",
        train_dataset.num_positive_pairs,
    )

    print(
        "Dev dialogues:",
        len(
            dev_dataset
        ),
    )

    print(
        "Dev utterances:",
        dev_dataset.num_utterances,
    )

    print(
        "Dev candidate pairs:",
        dev_dataset.num_candidate_pairs,
    )

    print(
        "Dev positive pairs:",
        dev_dataset.num_positive_pairs,
    )

    random_pair_auprc_baseline = (
            dev_dataset.num_positive_pairs
            / dev_dataset.num_candidate_pairs
    )

    print(
        "Random Pair AUPRC baseline:",
        f"{random_pair_auprc_baseline:.4f}",
    )

    # -----------------------------------------------------
    # Pair pos weight
    # -----------------------------------------------------

    pos_weight_config = (
        training_config[
            "pair_pos_weight"
        ]
    )

    if (
        isinstance(
            pos_weight_config,
            str,
        )
        and
        pos_weight_config.lower()
        == "auto"
    ):
        pair_pos_weight = (
            train_dataset
            .num_negative_pairs
            / train_dataset
            .num_positive_pairs
        )

        weight_mode = (
            "auto"
        )

    else:

        pair_pos_weight = float(
            pos_weight_config
        )

        weight_mode = (
            "manual"
        )

    if pair_pos_weight <= 0:
        raise ValueError(
            "pair_pos_weight "
            "must be > 0."
        )

    print(
        "Pair pos_weight mode:",
        weight_mode,
    )

    print(
        "Pair pos_weight:",
        pair_pos_weight,
    )

    # -----------------------------------------------------
    # DataLoader
    # -----------------------------------------------------

    collator = (
        FeatureECPECCollator(
            hidden_size=(
                hidden_size
            )
        )
    )

    batch_size = int(
        training_config.get(
            "batch_size",
            4,
        )
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

        collator=(
            collator
        ),

        batch_size=(
            batch_size
        ),

        num_workers=(
            num_workers
        ),

        device=(
            device
        ),

        seed=(
            seed
        ),
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

    model = (
        FeatureECPECBaseModel(

            input_dim=(
                hidden_size
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
        )
        .to(
            device
        )
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
        "Model type:",
        "Frozen-Feature ECPEC",
    )

    print(
        "RoBERTa in training model:",
        False,
    )

    print(
        "Total parameters:",
        f"{total_parameters:,}",
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}",
    )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

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

    optimizer = AdamW(
        (
            p
            for p in model.parameters()
            if p.requires_grad
        ),
        lr=module_lr,
        weight_decay=weight_decay,
    )

    print()
    print(
        "===== Optimizer ====="
    )

    print(
        "Optimizer: AdamW"
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
        "Optimized parameters:",
        f"{trainable_parameters:,}",
    )

    # -----------------------------------------------------
    # Training parameters
    # -----------------------------------------------------

    epochs = int(
        training_config.get(
            "epochs",
            20,
        )
    )

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

    max_grad_norm = float(
        training_config.get(
            "max_grad_norm",
            1.0,
        )
    )

    patience_limit = int(
        training_config.get(
            "early_stop_patience",
            4,
        )
    )

    threshold = float(
        evaluation_config.get(
            "threshold",
            0.5,
        )
    )

    checkpoint_path = (
        resolve_path(
            training_config.get(
                "checkpoint_path",
                (
                    "checkpoints/"
                    "base_feature_best.pt"
                ),
            )
        )
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = (
        checkpoint_path.parent
        / "training_history_feature.json"
    )

    run_config_path = (
        checkpoint_path.parent
        / "run_config_feature.yaml"
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

    print()
    print(
        "===== Evaluation ====="
    )

    print(
        "Pair decision threshold:",
        threshold,
    )

    print(
        "Pair AUPRC:",
        "enabled",
    )

    print(
        "AUPRC calculation:",
        "global over all valid Dev pairs",
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
        "Early stopping patience:",
        patience_limit,
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
    # Train
    # -----------------------------------------------------

    best_pair_f1 = -1.0
    best_epoch = 0
    patience = 0

    history = []

    print()
    print(
        "=" * 44
    )

    print(
        "ECPEC Frozen-Feature Training"
    )

    print(
        "=" * 44
    )

    print(
        "Selection metric: "
        "Dev Pair-F1"
    )

    print(
        "Pair threshold:",
        threshold,
    )

    print(
        "Diagnostic metric:",
        "Dev Pair AUPRC"
    )

    print(
        "Early stopping patience:",
        patience_limit,
    )

    print()

    for epoch in range(
        1,
        epochs + 1,
    ):

        train_result = (
            train_one_epoch(
                model=model,

                loader=(
                    train_loader
                ),

                optimizer=(
                    optimizer
                ),

                device=(
                    device
                ),

                pair_pos_weight=(
                    pair_pos_weight
                ),

                lambda_emotion=(
                    lambda_emotion
                ),

                lambda_cause=(
                    lambda_cause
                ),

                threshold=(
                    threshold
                ),

                max_grad_norm=(
                    max_grad_norm
                ),

                epoch=(
                    epoch
                ),

                total_epochs=(
                    epochs
                ),
            )
        )

        dev_result = (
            evaluate(
                model=model,

                loader=(
                    dev_loader
                ),

                device=(
                    device
                ),

                pair_pos_weight=(
                    pair_pos_weight
                ),

                lambda_emotion=(
                    lambda_emotion
                ),

                lambda_cause=(
                    lambda_cause
                ),

                threshold=(
                    threshold
                ),

                description=(
                    "Dev"
                ),
            )
        )

        print_result(
            epoch=(
                epoch
            ),

            total_epochs=(
                epochs
            ),

            train_result=(
                train_result
            ),

            dev_result=(
                dev_result
            ),
        )

        history.append(
            make_history_entry(
                epoch=(
                    epoch
                ),

                train_result=(
                    train_result
                ),

                dev_result=(
                    dev_result
                ),
            )
        )

        # Save history after every epoch
        save_history(
            history,
            history_path,
        )

        current_pair_f1 = float(
            dev_result[
                "metrics"
            ][
                "pair"
            ][
                "f1"
            ]
        )

        current_pair_auprc = float(
            dev_result[
                "pair_auprc"
            ]
        )

        if (
            current_pair_f1
            > best_pair_f1
        ):
            best_pair_f1 = (
                current_pair_f1
            )

            best_epoch = (
                epoch
            )

            patience = 0

            save_checkpoint(
                path=(
                    checkpoint_path
                ),

                model=(
                    model
                ),

                optimizer=(
                    optimizer
                ),

                epoch=(
                    epoch
                ),

                best_pair_f1=(
                    best_pair_f1
                ),

                threshold=(
                    threshold
                ),

                config=(
                    config
                ),

                pair_auprc=(
                    current_pair_auprc
                ),
            )

            print()
            print(
                "New best Dev Pair-F1: "
                f"{best_pair_f1:.4f}"
            )

            print(
                "Corresponding Pair AUPRC: "
                f"{current_pair_auprc:.4f}"
            )

            print(
                "Checkpoint saved to:"
            )

            print(
                checkpoint_path
            )

        else:

            patience += 1

            print()
            print(
                "No improvement. "
                f"Patience: "
                f"{patience}/"
                f"{patience_limit}"
            )

        print(
            "=" * 44
        )

        if (
            patience
            >= patience_limit
        ):
            print()
            print(
                "Early stopping triggered."
            )

            break

    # -----------------------------------------------------
    # Load best checkpoint
    # -----------------------------------------------------

    print()
    print(
        "===== Load Best Model ====="
    )

    checkpoint = (
        load_checkpoint(
            path=(
                checkpoint_path
            ),

            model=(
                model
            ),

            device=(
                device
            ),
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
        f"{checkpoint['best_pair_f1']:.4f}",
    )

    if (
        "pair_auprc"
        in checkpoint
    ):
        print(
            "Best checkpoint Pair AUPRC:",
            f"{checkpoint['pair_auprc']:.4f}",
        )

    # -----------------------------------------------------
    # Final Dev evaluation
    # -----------------------------------------------------

    best_dev_result = (
        evaluate(
            model=model,

            loader=(
                dev_loader
            ),

            device=(
                device
            ),

            pair_pos_weight=(
                pair_pos_weight
            ),

            lambda_emotion=(
                lambda_emotion
            ),

            lambda_cause=(
                lambda_cause
            ),

            threshold=(
                threshold
            ),

            description=(
                "Best Dev"
            ),
        )
    )

    dev_pair = (
        best_dev_result[
            "metrics"
        ][
            "pair"
        ]
    )

    dev_emotion = (
        best_dev_result[
            "metrics"
        ][
            "emotion"
        ]
    )

    dev_cause = (
        best_dev_result[
            "metrics"
        ][
            "cause"
        ]
    )

    print()
    print(
        "=" * 52
    )

    print(
        "Best Frozen-Feature Dev Result"
    )

    print(
        "=" * 52
    )

    print(
        "Emotion | "
        f"P={dev_emotion['precision']:.4f} "
        f"R={dev_emotion['recall']:.4f} "
        f"F1={dev_emotion['f1']:.4f}"
    )

    print(
        "Cause   | "
        f"P={dev_cause['precision']:.4f} "
        f"R={dev_cause['recall']:.4f} "
        f"F1={dev_cause['f1']:.4f}"
    )

    print(
        "Pair    | "
        f"P={dev_pair['precision']:.4f} "
        f"R={dev_pair['recall']:.4f} "
        f"F1={dev_pair['f1']:.4f} "
        f"#Pred={dev_pair['predicted_positive']} "
        f"#Gold={dev_pair['gold_positive']}"
    )

    print(
        "Pair AUPRC | "
        f"{best_dev_result['pair_auprc']:.4f}"
    )

    print()
    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}",
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best Pair-F1:",
        f"{best_pair_f1:.4f}",
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    print(
        "History:",
        history_path,
    )

    print(
        "Run config:",
        run_config_path,
    )

    print(
        "=" * 52
    )


# =========================================================
# Main
# =========================================================

def main():

    args = parse_args()

    config_path = resolve_path(
        args.config
    )

    config = load_config(
        config_path
    )

    validate_config(
        config
    )

    # -----------------------------------------------------
    # Logging
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
            "train_feature.log",
        )
    )

    log_path = (
        log_dir
        / log_filename
    )

    original_stdout = (
        sys.stdout
    )

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
        print(
            "#" * 52
        )

        print(
            "New Frozen-Feature ECPEC training run"
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
            "#" * 52
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

        print()
        print(
            "Training failed."
        )

        print(
            traceback.format_exc()
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