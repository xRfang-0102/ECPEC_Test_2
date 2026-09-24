import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.feature_dataset import FeatureECFDataset
from dataset.feature_collate import FeatureECPECCollator
from models.feature_base_model import FeatureECPECBaseModel


PROJECT_ROOT = Path(__file__).resolve().parent


# =========================================================
# Arguments
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep Pair decision threshold on the Dev set "
            "for Frozen-Feature ECPEC."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config_feature_pos2.5.yaml",
        help="Feature training config file.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint path. "
            "If omitted, use training.checkpoint_path "
            "from the config."
        ),
    )

    parser.add_argument(
        "--start",
        type=float,
        default=0.40,
        help="Start of Pair threshold sweep.",
    )

    parser.add_argument(
        "--end",
        type=float,
        default=0.85,
        help="End of Pair threshold sweep.",
    )

    parser.add_argument(
        "--step",
        type=float,
        default=0.01,
        help="Threshold step.",
    )

    parser.add_argument(
        "--aux-threshold",
        type=float,
        default=0.50,
        help=(
            "Fixed threshold for Emotion/Cause metrics. "
            "Does not affect Pair threshold sweep."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Evaluation batch size. "
            "If omitted, use training.batch_size."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory. "
            "Default: logs/threshold_sweep."
        ),
    )

    parser.add_argument(
        "--row-normalize",
        action="store_true",
        help=(
            "Subtract the per-emotion-row mean from the Pair "
            "logits before thresholding. Removes the global "
            "logit-drift component of the decision."
        ),
    )

    return parser.parse_args()


# =========================================================
# Utilities
# =========================================================

def resolve_path(path_value):
    path = Path(path_value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def load_config(config_path):
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(
            f"Empty config: {config_path}"
        )

    for section in [
        "data",
        "model",
        "training",
    ]:
        if section not in config:
            raise KeyError(
                f"Missing config section: {section}"
            )

    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# =========================================================
# Metrics
# =========================================================

def compute_binary_metrics(
    logits,
    labels,
    threshold,
):
    """
    logits:
        1-D CPU tensor

    labels:
        1-D CPU tensor containing 0/1
    """

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities
        >= threshold
    )

    gold = (
        labels >= 0.5
    )

    tp = int(
        (
            predictions
            & gold
        )
        .sum()
        .item()
    )

    fp = int(
        (
            predictions
            & (~gold)
        )
        .sum()
        .item()
    )

    fn = int(
        (
            (~predictions)
            & gold
        )
        .sum()
        .item()
    )

    tn = int(
        (
            (~predictions)
            & (~gold)
        )
        .sum()
        .item()
    )

    predicted_positive = (
        tp + fp
    )

    gold_positive = (
        tp + fn
    )

    total = (
        tp
        + fp
        + fn
        + tn
    )

    precision = (
        tp / predicted_positive
        if predicted_positive > 0
        else 0.0
    )

    recall = (
        tp / gold_positive
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
            tp + tn
        )
        / total
        if total > 0
        else 0.0
    )

    return {
        "threshold":
            float(threshold),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "accuracy":
            float(accuracy),

        "tp":
            tp,

        "fp":
            fp,

        "fn":
            fn,

        "tn":
            tn,

        "predicted_positive":
            predicted_positive,

        "gold_positive":
            gold_positive,
    }


# =========================================================
# Device
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

    moved = dict(batch)

    for key in tensor_keys:
        if key in moved:
            moved[key] = (
                moved[key]
                .to(
                    device,
                    non_blocking=True,
                )
            )

    return moved


# =========================================================
# Model
# =========================================================

def build_model(
    model_config,
    input_dim,
):
    model = FeatureECPECBaseModel(
        input_dim=input_dim,

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

        pair_decision_mode=str(
            model_config.get(
                "pair_decision_mode",
                "fixed",
            )
        ),

        pair_node_gating=bool(
            model_config.get(
                "pair_node_gating",
                False,
            )
        ),

        use_event_retrieval=bool(
            model_config.get(
                "use_event_retrieval",
                False,
            )
        ),

        retrieval_key_dim=int(
            model_config.get(
                "retrieval_key_dim",
                64,
            )
        ),

        use_speaker_thread=bool(
            model_config.get(
                "use_speaker_thread",
                True,
            )
        ),

        use_historical_retrieval=bool(
            model_config.get(
                "use_historical_retrieval",
                True,
            )
        ),

        use_locality_prior=bool(
            model_config.get(
                "use_locality_prior",
                False,
            )
        ),

        null_hidden=int(
            model_config.get(
                "null_hidden",
                128,
            )
        ),

        null_residual_hidden=int(
            model_config.get(
                "null_residual_hidden",
                128,
            )
        ),

        null_residual_scale=float(
            model_config.get(
                "null_residual_scale",
                0.5,
            )
        ),

        null_center_init=float(
            model_config.get(
                "null_center_init",
                0.0,
            )
        ),

        hierarchical_center_init=float(
            model_config.get(
                "hierarchical_center_init",
                0.0,
            )
        ),

        hierarchical_dialogue_hidden=int(
            model_config.get(
                "hierarchical_dialogue_hidden",
                128,
            )
        ),

        hierarchical_dialogue_scale=float(
            model_config.get(
                "hierarchical_dialogue_scale",
                0.5,
            )
        ),

        hierarchical_row_hidden=int(
            model_config.get(
                "hierarchical_row_hidden",
                128,
            )
        ),

        hierarchical_row_scale=float(
            model_config.get(
                "hierarchical_row_scale",
                0.5,
            )
        ),
    )

    return model


def load_checkpoint(
    checkpoint_path,
    model,
    device,
):
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    if (
        "model_state_dict"
        not in checkpoint
    ):
        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    return checkpoint


# =========================================================
# Collect Dev logits
# =========================================================

@torch.no_grad()
def collect_dev_outputs(
    model,
    dev_loader,
    device,
    row_normalize=False,
):
    model.eval()

    all_emotion_logits = []
    all_emotion_labels = []

    all_cause_logits = []
    all_cause_labels = []

    all_pair_logits = []
    all_pair_labels = []

    progress = tqdm(
        dev_loader,
        desc="Collecting Dev logits",
        dynamic_ncols=True,
    )

    for batch in progress:

        batch = move_batch_to_device(
            batch,
            device,
        )

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

        utterance_mask = (
            batch[
                "utterance_mask"
            ].bool()
        )

        pair_mask = (
            batch[
                "pair_mask"
            ].bool()
        )

        # ---------------------------------------------
        # Emotion
        # ---------------------------------------------

        all_emotion_logits.append(
            outputs[
                "emotion_logits"
            ][
                utterance_mask
            ]
            .detach()
            .float()
            .cpu()
        )

        all_emotion_labels.append(
            batch[
                "emotion_labels"
            ][
                utterance_mask
            ]
            .detach()
            .float()
            .cpu()
        )

        # ---------------------------------------------
        # Cause
        # ---------------------------------------------

        all_cause_logits.append(
            outputs[
                "cause_logits"
            ][
                utterance_mask
            ]
            .detach()
            .float()
            .cpu()
        )

        all_cause_labels.append(
            batch[
                "cause_labels"
            ][
                utterance_mask
            ]
            .detach()
            .float()
            .cpu()
        )

        # ---------------------------------------------
        # Pair
        # ---------------------------------------------

        pair_logits = outputs[
            "pair_logits"
        ]

        if row_normalize:
            counts = (
                pair_mask
                .sum(dim=-1, keepdim=True)
                .clamp_min(1)
            )

            row_mean = (
                pair_logits
                .sum(dim=-1, keepdim=True)
                / counts
            )

            pair_logits = (
                pair_logits
                - row_mean
            ).masked_fill(
                ~pair_mask,
                0.0,
            )

        all_pair_logits.append(
            pair_logits[
                pair_mask
            ]
            .detach()
            .float()
            .cpu()
        )

        all_pair_labels.append(
            batch[
                "pair_labels"
            ][
                pair_mask
            ]
            .detach()
            .float()
            .cpu()
        )

    emotion_logits = torch.cat(
        all_emotion_logits,
        dim=0,
    )

    emotion_labels = torch.cat(
        all_emotion_labels,
        dim=0,
    )

    cause_logits = torch.cat(
        all_cause_logits,
        dim=0,
    )

    cause_labels = torch.cat(
        all_cause_labels,
        dim=0,
    )

    pair_logits = torch.cat(
        all_pair_logits,
        dim=0,
    )

    pair_labels = torch.cat(
        all_pair_labels,
        dim=0,
    )

    return {
        "emotion_logits":
            emotion_logits,

        "emotion_labels":
            emotion_labels,

        "cause_logits":
            cause_logits,

        "cause_labels":
            cause_labels,

        "pair_logits":
            pair_logits,

        "pair_labels":
            pair_labels,
    }


# =========================================================
# Threshold generation
# =========================================================

def build_thresholds(
    start,
    end,
    step,
):
    if step <= 0:
        raise ValueError(
            "step must be > 0."
        )

    if end < start:
        raise ValueError(
            "end must be >= start."
        )

    thresholds = []

    value = float(start)

    while (
        value
        <= end + 1e-12
    ):
        thresholds.append(
            round(
                value,
                6,
            )
        )

        value += step

    return thresholds


# =========================================================
# Save results
# =========================================================

def save_csv(
    rows,
    path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "threshold",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "tp",
        "fp",
        "fn",
        "tn",
        "predicted_positive",
        "gold_positive",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# =========================================================
# Main
# =========================================================

def main():
    args = parse_args()

    # -----------------------------------------------------
    # Config
    # -----------------------------------------------------

    config_path = resolve_path(
        args.config
    )

    config = load_config(
        config_path
    )

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

    seed = int(
        config.get(
            "seed",
            42,
        )
    )

    set_seed(
        seed
    )

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # -----------------------------------------------------
    # Checkpoint
    # -----------------------------------------------------

    if args.checkpoint is not None:

        checkpoint_path = resolve_path(
            args.checkpoint
        )

    else:

        if (
            "checkpoint_path"
            not in training_config
        ):
            raise KeyError(
                "training.checkpoint_path "
                "not found in config."
            )

        checkpoint_path = resolve_path(
            training_config[
                "checkpoint_path"
            ]
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    # -----------------------------------------------------
    # Feature path
    # -----------------------------------------------------

    feature_root = resolve_path(
        data_config[
            "feature_root"
        ]
    )

    dev_feature_path = (
        feature_root
        / data_config[
            "dev_feature_file"
        ]
    ).resolve()

    if not dev_feature_path.exists():
        raise FileNotFoundError(
            f"Dev feature file not found: "
            f"{dev_feature_path}"
        )

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    input_dim = int(
        model_config.get(
            "input_dim",
            768,
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
            4,
        )
    )

    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(
            training_config.get(
                "batch_size",
                4,
            )
        )
    )

    dev_dataset = FeatureECFDataset(
        feature_path=(
            dev_feature_path
        ),

        max_dialogue_length=(
            max_dialogue_length
        ),

        expected_hidden_size=(
            input_dim
        ),

        validate=True,
    )

    collator = FeatureECPECCollator(
        hidden_size=input_dim
    )

    dev_loader = DataLoader(
        dev_dataset,

        batch_size=(
            batch_size
        ),

        shuffle=False,

        num_workers=(
            num_workers
        ),

        collate_fn=(
            collator
        ),

        pin_memory=(
            device.type == "cuda"
        ),

        persistent_workers=(
            num_workers > 0
        ),
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = build_model(
        model_config=(
            model_config
        ),

        input_dim=(
            input_dim
        ),
    ).to(
        device
    )

    checkpoint = load_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),

        model=(
            model
        ),

        device=(
            device
        ),
    )

    total_parameters, trainable_parameters = (
        count_parameters(
            model
        )
    )

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------

    print()
    print(
        "=" * 64
    )

    print(
        "ECPEC Dev Pair Threshold Sweep"
    )

    print(
        "=" * 64
    )

    print(
        "Config:",
        config_path,
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    print(
        "Device:",
        device,
    )

    print(
        "Dev feature:",
        dev_feature_path,
    )

    print(
        "Dev dialogues:",
        len(
            dev_dataset
        ),
    )

    print(
        "Dev candidate pairs:",
        dev_dataset.num_candidate_pairs,
    )

    print(
        "Dev positive pairs:",
        dev_dataset.num_positive_pairs,
    )

    print(
        "Total parameters:",
        f"{total_parameters:,}",
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}",
    )

    if "epoch" in checkpoint:
        print(
            "Checkpoint epoch:",
            checkpoint[
                "epoch"
            ],
        )

    if "best_pair_f1" in checkpoint:
        print(
            "Checkpoint stored best F1:",
            checkpoint[
                "best_pair_f1"
            ],
        )

    print(
        "Auxiliary threshold:",
        args.aux_threshold,
    )

    print(
        "Pair sweep:",
        f"{args.start:.2f} "
        f"to {args.end:.2f} "
        f"step {args.step:.3f}",
    )

    # -----------------------------------------------------
    # Collect logits once
    # -----------------------------------------------------

    print(
        "Pair row normalization:",
        args.row_normalize,
    )

    outputs = collect_dev_outputs(
        model=(
            model
        ),

        dev_loader=(
            dev_loader
        ),

        device=(
            device
        ),

        row_normalize=(
            args.row_normalize
        ),
    )

    print()
    print(
        "Collected:"
    )

    print(
        "Emotion logits:",
        outputs[
            "emotion_logits"
        ].shape,
    )

    print(
        "Cause logits:",
        outputs[
            "cause_logits"
        ].shape,
    )

    print(
        "Pair logits:",
        outputs[
            "pair_logits"
        ].shape,
    )

    # -----------------------------------------------------
    # Auxiliary metrics
    # -----------------------------------------------------

    emotion_result = (
        compute_binary_metrics(
            logits=outputs[
                "emotion_logits"
            ],

            labels=outputs[
                "emotion_labels"
            ],

            threshold=(
                args.aux_threshold
            ),
        )
    )

    cause_result = (
        compute_binary_metrics(
            logits=outputs[
                "cause_logits"
            ],

            labels=outputs[
                "cause_labels"
            ],

            threshold=(
                args.aux_threshold
            ),
        )
    )

    print()
    print(
        "===== Auxiliary Metrics "
        f"@ {args.aux_threshold:.2f} ====="
    )

    print(
        "Emotion | "
        f"P={emotion_result['precision']:.4f} "
        f"R={emotion_result['recall']:.4f} "
        f"F1={emotion_result['f1']:.4f}"
    )

    print(
        "Cause   | "
        f"P={cause_result['precision']:.4f} "
        f"R={cause_result['recall']:.4f} "
        f"F1={cause_result['f1']:.4f}"
    )

    # -----------------------------------------------------
    # Sweep Pair threshold
    # -----------------------------------------------------

    thresholds = build_thresholds(
        start=(
            args.start
        ),

        end=(
            args.end
        ),

        step=(
            args.step
        ),
    )

    rows = []

    print()
    print(
        "=" * 85
    )

    print(
        f"{'Thr':>6} "
        f"{'P':>9} "
        f"{'R':>9} "
        f"{'F1':>9} "
        f"{'#Pred':>9} "
        f"{'#Gold':>9} "
        f"{'TP':>7} "
        f"{'FP':>7} "
        f"{'FN':>7}"
    )

    print(
        "-" * 85
    )

    for threshold in thresholds:

        result = compute_binary_metrics(
            logits=outputs[
                "pair_logits"
            ],

            labels=outputs[
                "pair_labels"
            ],

            threshold=(
                threshold
            ),
        )

        rows.append(
            result
        )

        print(
            f"{threshold:6.2f} "
            f"{result['precision']:9.4f} "
            f"{result['recall']:9.4f} "
            f"{result['f1']:9.4f} "
            f"{result['predicted_positive']:9d} "
            f"{result['gold_positive']:9d} "
            f"{result['tp']:7d} "
            f"{result['fp']:7d} "
            f"{result['fn']:7d}"
        )

    print(
        "=" * 85
    )

    # -----------------------------------------------------
    # Select best
    #
    # Primary criterion:
    #   F1
    #
    # Tie breaking:
    #   Precision
    #   Recall
    # -----------------------------------------------------

    best_result = max(
        rows,
        key=lambda x: (
            x["f1"],
            x["precision"],
            x["recall"],
        ),
    )

    print()
    print(
        "=" * 64
    )

    print(
        "BEST DEV PAIR THRESHOLD"
    )

    print(
        "=" * 64
    )

    print(
        "Threshold:",
        f"{best_result['threshold']:.4f}",
    )

    print(
        "Precision:",
        f"{best_result['precision']:.4f}",
    )

    print(
        "Recall:",
        f"{best_result['recall']:.4f}",
    )

    print(
        "F1:",
        f"{best_result['f1']:.4f}",
    )

    print(
        "#Pred:",
        best_result[
            "predicted_positive"
        ],
    )

    print(
        "#Gold:",
        best_result[
            "gold_positive"
        ],
    )

    print(
        "TP:",
        best_result[
            "tp"
        ],
    )

    print(
        "FP:",
        best_result[
            "fp"
        ],
    )

    print(
        "FN:",
        best_result[
            "fn"
        ],
    )

    # -----------------------------------------------------
    # Output directory
    # -----------------------------------------------------

    if args.output_dir is not None:

        output_dir = resolve_path(
            args.output_dir
        )

    else:

        output_dir = resolve_path(
            "logs/threshold_sweep"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_stem = (
        checkpoint_path.stem
        + ("_row_normalized" if args.row_normalize else "")
    )

    csv_path = (
        output_dir
        / (
            f"{checkpoint_stem}"
            "_threshold_sweep.csv"
        )
    )

    json_path = (
        output_dir
        / (
            f"{checkpoint_stem}"
            "_threshold_best.json"
        )
    )

    save_csv(
        rows=rows,
        path=csv_path,
    )

    result_payload = {
        "config":
            str(
                config_path
            ),

        "checkpoint":
            str(
                checkpoint_path
            ),

        "checkpoint_epoch":
            checkpoint.get(
                "epoch"
            ),

        "checkpoint_stored_best_f1":
            checkpoint.get(
                "best_pair_f1"
            ),

        "dev_feature":
            str(
                dev_feature_path
            ),

        "num_dev_dialogues":
            len(
                dev_dataset
            ),

        "num_dev_candidate_pairs":
            dev_dataset.num_candidate_pairs,

        "num_dev_positive_pairs":
            dev_dataset.num_positive_pairs,

        "aux_threshold":
            float(
                args.aux_threshold
            ),

        "emotion_metrics":
            emotion_result,

        "cause_metrics":
            cause_result,

        "sweep_start":
            float(
                args.start
            ),

        "sweep_end":
            float(
                args.end
            ),

        "sweep_step":
            float(
                args.step
            ),

        "best_pair":
            best_result,
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result_payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "Sweep CSV saved to:"
    )

    print(
        csv_path
    )

    print()
    print(
        "Best threshold JSON saved to:"
    )

    print(
        json_path
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Select threshold on Dev only. "
        "After choosing it, keep it fixed "
        "for the final Test evaluation."
    )

    print(
        "=" * 64
    )


if __name__ == "__main__":
    main()