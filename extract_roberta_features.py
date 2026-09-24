import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from dataset.dataset import ECFDataset


PROJECT_ROOT = Path(__file__).resolve().parent


# =========================================================
# Arguments
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract fixed RoBERTa utterance features "
            "for ECPEC training."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config_v1.yaml",
        help="Path to YAML config file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="features/ECF",
        help="Directory used to save extracted features.",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=[
            "train",
            "dev",
            "test",
        ],
        choices=[
            "train",
            "dev",
            "test",
        ],
        help="Dataset splits to extract.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="RoBERTa extraction batch size.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing feature files.",
    )

    parser.add_argument(
        "--pooling",
        type=str,
        default="cls",
        choices=["cls", "mean"],
        help=(
            "RoBERTa pooling strategy: cls token (original) "
            "or attention-masked mean pooling."
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
            f"Config file not found: {config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(
            f"Empty config file: {config_path}"
        )

    if "data" not in config:
        raise KeyError(
            "Missing 'data' section in config."
        )

    if "model" not in config:
        raise KeyError(
            "Missing 'model' section in config."
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


def to_cpu_tensor(
    value,
    dtype=None,
):
    if torch.is_tensor(value):
        tensor = value.detach().cpu().clone()
    else:
        tensor = torch.tensor(value)

    if dtype is not None:
        tensor = tensor.to(dtype)

    return tensor


# =========================================================
# Dataset file mapping
# =========================================================

def get_split_path(
    data_config,
    split,
):
    data_root = resolve_path(
        data_config["data_root"]
    )

    key_map = {
        "train": "train_file",
        "dev": "dev_file",
        "test": "test_file",
    }

    key = key_map[split]

    if key not in data_config:
        raise KeyError(
            f"Missing data.{key} in config."
        )

    path = (
        data_root
        / data_config[key]
    )

    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"{split} file not found: {path}"
        )

    return path


# =========================================================
# Collect utterances
# =========================================================

def collect_dataset_information(
    dataset,
):
    all_texts = []
    dialogue_records = []

    cursor = 0

    for index in range(
        len(dataset)
    ):
        item = dataset[index]

        utterances = list(
            item["utterances"]
        )

        num_utterances = int(
            item["num_utterances"]
        )

        if len(utterances) != num_utterances:
            raise ValueError(
                "Utterance count mismatch "
                f"for dialogue index {index}: "
                f"{len(utterances)} vs "
                f"{num_utterances}"
            )

        start = cursor
        end = (
            start
            + num_utterances
        )

        all_texts.extend(
            utterances
        )

        record = {
            "dialogue_id":
                item["dialogue_id"],

            "utterance_ids":
                list(
                    item["utterance_ids"]
                ),

            "speaker_ids":
                to_cpu_tensor(
                    item["speaker_ids"],
                    dtype=torch.long,
                ),

            "emotion_labels":
                to_cpu_tensor(
                    item["emotion_labels"],
                    dtype=torch.float32,
                ),

            "cause_labels":
                to_cpu_tensor(
                    item["cause_labels"],
                    dtype=torch.float32,
                ),

            "pair_labels":
                to_cpu_tensor(
                    item["pair_labels"],
                    dtype=torch.float32,
                ),

            "pairs":
                item.get(
                    "pairs",
                    [],
                ),

            "num_utterances":
                num_utterances,

            "_feature_start":
                start,

            "_feature_end":
                end,
        }

        dialogue_records.append(
            record
        )

        cursor = end

    return (
        all_texts,
        dialogue_records,
    )


# =========================================================
# RoBERTa extraction
# =========================================================

def encode_utterances(
    texts,
    tokenizer,
    encoder,
    device,
    batch_size,
    max_length,
    pooling="cls",
):
    if len(texts) == 0:
        raise ValueError(
            "No utterances found."
        )

    if pooling not in ("cls", "mean"):
        raise ValueError(
            f"Unknown pooling: {pooling}"
        )

    hidden_size = int(
        encoder.config.hidden_size
    )

    all_features = torch.empty(
        (
            len(texts),
            hidden_size,
        ),
        dtype=torch.float32,
        device="cpu",
    )

    encoder.eval()

    print()
    print(
        "RoBERTa training mode:",
        encoder.training,
    )

    print(
        "Gradient enabled:",
        torch.is_grad_enabled(),
        "(outside no_grad context)",
    )

    total_batches = (
        len(texts)
        + batch_size
        - 1
    ) // batch_size

    progress = tqdm(
        range(
            0,
            len(texts),
            batch_size,
        ),
        total=total_batches,
        desc="Extracting RoBERTa features",
        dynamic_ncols=True,
    )

    with torch.no_grad():

        for start in progress:

            end = min(
                start + batch_size,
                len(texts),
            )

            batch_texts = (
                texts[start:end]
            )

            encoded = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            model_inputs = {}

            for key, value in encoded.items():

                if key in {
                    "input_ids",
                    "attention_mask",
                    "token_type_ids",
                }:
                    model_inputs[key] = (
                        value.to(
                            device,
                            non_blocking=True,
                        )
                    )

            outputs = encoder(
                **model_inputs
            )

            # Pooling:
            #   cls  = last_hidden_state[:, 0, :]   (original)
            #   mean = attention-masked mean pooling over tokens
            #          (richer content signal for long-range matching)
            if pooling == "mean":
                token_mask = (
                    model_inputs["attention_mask"]
                    .unsqueeze(-1)
                    .to(outputs.last_hidden_state.dtype)
                )

                batch_features = (
                    outputs.last_hidden_state
                    * token_mask
                ).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1.0)
            else:
                batch_features = (
                    outputs
                    .last_hidden_state[
                        :,
                        0,
                        :
                    ]
                )

            batch_features = (
                batch_features
                .detach()
                .float()
                .cpu()
            )

            if batch_features.shape != (
                end - start,
                hidden_size,
            ):
                raise RuntimeError(
                    "Unexpected feature shape: "
                    f"{batch_features.shape}"
                )

            all_features[
                start:end
            ] = batch_features

            progress.set_postfix(
                {
                    "utterances":
                        f"{end}/{len(texts)}"
                }
            )

    if not torch.isfinite(
        all_features
    ).all():

        raise RuntimeError(
            "Non-finite values detected "
            "in extracted features."
        )

    return all_features


# =========================================================
# Reconstruct dialogue-level features
# =========================================================

def build_feature_payload(
    split,
    dataset_path,
    dialogue_records,
    all_features,
    encoder_path,
    max_length,
):
    dialogues = []

    for record in dialogue_records:

        start = record.pop(
            "_feature_start"
        )

        end = record.pop(
            "_feature_end"
        )

        features = (
            all_features[
                start:end
            ]
            .clone()
            .contiguous()
        )

        if features.shape[0] != (
            record[
                "num_utterances"
            ]
        ):
            raise RuntimeError(
                "Feature / utterance count "
                "mismatch."
            )

        dialogue = dict(
            record
        )

        dialogue[
            "utterance_features"
        ] = features

        dialogues.append(
            dialogue
        )

    payload = {

        "metadata": {

            "format_version":
                1,

            "dataset":
                "ECF",

            "split":
                split,

            "dataset_path":
                str(
                    dataset_path
                ),

            "encoder":
                str(
                    encoder_path
                ),

            "encoder_type":
                "RoBERTa",

            "hidden_size":
                int(
                    all_features.shape[
                        1
                    ]
                ),

            "max_utterance_length":
                int(
                    max_length
                ),

            "feature_type":
                (
                    "last_hidden_state"
                    "[:,0,:]"
                ),

            "frozen":
                True,

            "encoder_eval_mode":
                True,

            "dtype":
                "float32",

            "num_dialogues":
                len(
                    dialogues
                ),

            "num_utterances":
                int(
                    all_features.shape[
                        0
                    ]
                ),
        },

        "dialogues":
            dialogues,
    }

    return payload


# =========================================================
# Validation
# =========================================================

def validate_payload(
    payload,
):
    metadata = payload[
        "metadata"
    ]

    dialogues = payload[
        "dialogues"
    ]

    total_utterances = 0

    for index, dialogue in enumerate(
        dialogues
    ):
        features = dialogue[
            "utterance_features"
        ]

        num_utterances = int(
            dialogue[
                "num_utterances"
            ]
        )

        if features.ndim != 2:
            raise RuntimeError(
                f"Dialogue {index}: "
                "features must be 2D."
            )

        if (
            features.shape[0]
            != num_utterances
        ):
            raise RuntimeError(
                f"Dialogue {index}: "
                "feature count mismatch."
            )

        if (
            features.shape[1]
            != metadata[
                "hidden_size"
            ]
        ):
            raise RuntimeError(
                f"Dialogue {index}: "
                "hidden size mismatch."
            )

        if (
            len(
                dialogue[
                    "utterance_ids"
                ]
            )
            != num_utterances
        ):
            raise RuntimeError(
                f"Dialogue {index}: "
                "utterance ID count mismatch."
            )

        if (
            dialogue[
                "speaker_ids"
            ].shape[0]
            != num_utterances
        ):
            raise RuntimeError(
                f"Dialogue {index}: "
                "speaker count mismatch."
            )

        if (
            dialogue[
                "emotion_labels"
            ].shape[0]
            != num_utterances
        ):
            raise RuntimeError(
                f"Dialogue {index}: "
                "emotion label count mismatch."
            )

        if (
            dialogue[
                "cause_labels"
            ].shape[0]
            != num_utterances
        ):
            raise RuntimeError(
                f"Dialogue {index}: "
                "cause label count mismatch."
            )

        expected_pair_shape = (
            num_utterances,
            num_utterances,
        )

        if tuple(
            dialogue[
                "pair_labels"
            ].shape
        ) != expected_pair_shape:
            raise RuntimeError(
                f"Dialogue {index}: "
                "pair label shape mismatch."
            )

        if not torch.isfinite(
            features
        ).all():
            raise RuntimeError(
                f"Dialogue {index}: "
                "non-finite feature detected."
            )

        total_utterances += (
            num_utterances
        )

    if total_utterances != (
        metadata[
            "num_utterances"
        ]
    ):
        raise RuntimeError(
            "Total utterance count mismatch."
        )


# =========================================================
# Extract one split
# =========================================================

def extract_split(
    split,
    dataset_path,
    output_path,
    tokenizer,
    encoder,
    encoder_path,
    device,
    batch_size,
    max_length,
    max_dialogue_length,
    overwrite,
    pooling="cls",
):
    print()
    print(
        "=========================================="
    )

    print(
        f"Extracting split: {split}"
    )

    print(
        "=========================================="
    )

    print(
        "Dataset:",
        dataset_path,
    )

    print(
        "Output:",
        output_path,
    )

    if output_path.exists():

        if not overwrite:
            raise FileExistsError(
                f"Feature file already exists: "
                f"{output_path}\n"
                f"Use --overwrite to replace it."
            )

        print(
            "Existing output will be overwritten."
        )

    dataset = ECFDataset(
        str(
            dataset_path
        ),
        max_dialogue_length=(
            max_dialogue_length
        ),
    )

    print(
        "Dialogues:",
        len(
            dataset
        ),
    )

    (
        all_texts,
        dialogue_records,
    ) = collect_dataset_information(
        dataset
    )

    print(
        "Utterances:",
        len(
            all_texts
        ),
    )

    all_features = encode_utterances(
        texts=all_texts,
        tokenizer=tokenizer,
        encoder=encoder,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        pooling=pooling,
    )

    print(
        "Feature matrix:",
        tuple(
            all_features.shape
        ),
    )

    payload = build_feature_payload(
        split=split,
        dataset_path=dataset_path,
        dialogue_records=dialogue_records,
        all_features=all_features,
        encoder_path=encoder_path,
        max_length=max_length,
    )

    validate_payload(
        payload
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        payload,
        output_path,
    )

    file_size_mb = (
        output_path.stat().st_size
        / 1024
        / 1024
    )

    print()
    print(
        f"{split} extraction completed."
    )

    print(
        "Saved:",
        output_path,
    )

    print(
        "Feature shape:",
        (
            payload[
                "metadata"
            ][
                "num_utterances"
            ],
            payload[
                "metadata"
            ][
                "hidden_size"
            ],
        ),
    )

    print(
        "File size:",
        f"{file_size_mb:.2f} MB",
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

    data_config = config[
        "data"
    ]

    model_config = config[
        "model"
    ]

    seed = int(
        config.get(
            "seed",
            42,
        )
    )

    set_seed(
        seed
    )

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size must be > 0."
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

    encoder_path = resolve_path(
        model_config[
            "encoder"
        ]
    )

    if not encoder_path.exists():
        raise FileNotFoundError(
            f"RoBERTa path not found: "
            f"{encoder_path}"
        )

    output_dir = resolve_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        "ECPEC Frozen RoBERTa Feature Extraction"
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
        "Encoder:",
        encoder_path,
    )

    print(
        "Output directory:",
        output_dir,
    )

    print(
        "Device:",
        device,
    )

    print(
        "Seed:",
        seed,
    )

    print(
        "Extraction batch size:",
        args.batch_size,
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
        "Splits:",
        args.splits,
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

        total_memory = (
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
            f"{total_memory:.2f} GB",
        )

    # -----------------------------------------------------
    # Load tokenizer
    # -----------------------------------------------------

    print()
    print(
        "Loading tokenizer..."
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            str(
                encoder_path
            ),
            use_fast=False,
            local_files_only=True,
        )
    )

    # -----------------------------------------------------
    # Load RoBERTa
    # -----------------------------------------------------

    print(
        "Loading RoBERTa..."
    )

    encoder = (
        AutoModel
        .from_pretrained(
            str(
                encoder_path
            ),
            local_files_only=True,
            add_pooling_layer=False,
        )
    )

    encoder = encoder.to(
        device
    )

    # Critical:
    # Frozen deterministic feature extractor
    encoder.eval()

    for parameter in (
        encoder.parameters()
    ):
        parameter.requires_grad = False

    trainable_encoder_params = sum(
        parameter.numel()
        for parameter
        in encoder.parameters()
        if parameter.requires_grad
    )

    total_encoder_params = sum(
        parameter.numel()
        for parameter
        in encoder.parameters()
    )

    print()
    print(
        "===== RoBERTa Status ====="
    )

    print(
        "Training mode:",
        encoder.training,
    )

    print(
        "Total parameters:",
        f"{total_encoder_params:,}",
    )

    print(
        "Trainable parameters:",
        f"{trainable_encoder_params:,}",
    )

    print(
        "Hidden size:",
        encoder.config.hidden_size,
    )

    if encoder.training:
        raise RuntimeError(
            "RoBERTa must be in eval mode "
            "during feature extraction."
        )

    if trainable_encoder_params != 0:
        raise RuntimeError(
            "RoBERTa must have zero "
            "trainable parameters."
        )

    # -----------------------------------------------------
    # Extract requested splits
    # -----------------------------------------------------

    output_name_map = {
        "train":
            (
                "train_roberta.pt"
                if args.pooling == "cls"
                else "train_roberta_mean.pt"
            ),

        "dev":
            (
                "dev_roberta.pt"
                if args.pooling == "cls"
                else "dev_roberta_mean.pt"
            ),

        "test":
            (
                "test_roberta.pt"
                if args.pooling == "cls"
                else "test_roberta_mean.pt"
            ),
    }

    for split in args.splits:

        dataset_path = (
            get_split_path(
                data_config,
                split,
            )
        )

        output_path = (
            output_dir
            / output_name_map[
                split
            ]
        )

        extract_split(
            split=split,

            dataset_path=(
                dataset_path
            ),

            output_path=(
                output_path
            ),

            tokenizer=(
                tokenizer
            ),

            encoder=(
                encoder
            ),

            encoder_path=(
                encoder_path
            ),

            device=(
                device
            ),

            batch_size=(
                args.batch_size
            ),

            max_length=(
                max_utterance_length
            ),

            max_dialogue_length=(
                max_dialogue_length
            ),

            pooling=(
                args.pooling
            ),

            overwrite=(
                args.overwrite
            ),
        )

    print()
    print(
        "=========================================="
    )

    print(
        "All requested feature extraction "
        "completed."
    )

    print(
        "Output directory:",
        output_dir,
    )

    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()