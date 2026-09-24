# =========================================================
# Shared loader for analysis scripts: supports BOTH the
# frozen-feature checkpoints (model_type FrozenFeatureECPEC)
# and the v3 LoRA checkpoints (model_type LoRAECPEC).
#
# Usage:
#     model, dataset, forward = load_model_and_dataset(
#         checkpoint_path, config, split="dev",
#     )
#     outputs = forward(model, sample)   # sample from dataset
# =========================================================

import torch

from dataset.feature_dataset import FeatureECFDataset
from dataset.raw_ecf_dataset import RawECFDataset
from models.lora_ecpec_model import LoRAECPECModel
from sweep_threshold import build_model


def load_model_and_dataset(
    checkpoint_path,
    config,
    split="dev",
    device=None,
    use_bf16=False,
):
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model_config = config["model"]
    data_config = config.get("data", {})

    model_type = checkpoint.get(
        "model_type",
        "FrozenFeatureECPEC",
    )

    if model_type == "LoRAECPEC":
        from pathlib import Path

        project_root = Path(__file__).resolve().parent

        encoder_path = (
            project_root
            / str(model_config.get(
                "encoder",
                "pretrained/roberta-base",
            ))
        )

        model = LoRAECPECModel(
            encoder_path=encoder_path,
            model_config=model_config,
            lora_r=int(model_config.get("lora_r", 16)),
            lora_alpha=float(model_config.get("lora_alpha", 32)),
            lora_dropout=float(model_config.get("lora_dropout", 0.1)),
            use_bf16=use_bf16,
        )

        data_root = data_config.get("data_root", "data/ECF")
        split_key = f"{split}_file"
        dataset = RawECFDataset(
            json_path=(
                project_root
                / data_root
                / data_config[split_key]
            ),
            max_dialogue_length=int(
                data_config.get("max_dialogue_length", 40)
            ),
            max_utterance_length=int(
                data_config.get("max_utterance_length", 128)
            ),
            tokenizer_path=encoder_path,
        )

        def forward_lora(model, sample):
            n = int(sample["num_utterances"])
            return model(
                input_ids=(
                    sample["input_ids"][:n].unsqueeze(0).to(device)
                ),
                attention_mask=(
                    sample["attention_mask"][:n].unsqueeze(0).to(device)
                ),
                speaker_ids=(
                    sample["speaker_ids"][:n].unsqueeze(0).to(device)
                ),
                position_ids=torch.arange(
                    n, device=device
                ).unsqueeze(0),
                utterance_mask=torch.ones(
                    1, n, dtype=torch.bool, device=device
                ),
                pair_mask=torch.ones(
                    1, n, n, dtype=torch.bool, device=device
                ),
            )

        forward = forward_lora

    else:
        model = build_model(model_config, 768)

        feature_root = data_config.get(
            "feature_root",
            "features/ECF",
        )
        dev_file = data_config.get(
            f"{split}_feature_file",
            data_config.get("dev_feature_file"),
        )
        if dev_file is None:
            raise KeyError(
                f"No feature file for split {split}"
            )

        from pathlib import Path

        project_root = Path(__file__).resolve().parent
        dataset = FeatureECFDataset(
            feature_path=project_root / feature_root / dev_file,
            max_dialogue_length=40,
            expected_hidden_size=768,
            validate=True,
        )

        def forward_feature(model, sample):
            n = int(sample["num_utterances"])
            return model(
                utterance_features=(
                    sample["utterance_features"][:n]
                    .unsqueeze(0)
                    .to(device)
                ),
                speaker_ids=(
                    sample["speaker_ids"][:n]
                    .unsqueeze(0)
                    .to(device)
                ),
                position_ids=torch.arange(
                    n, device=device
                ).unsqueeze(0),
                utterance_mask=torch.ones(
                    1, n, dtype=torch.bool, device=device
                ),
                pair_mask=torch.ones(
                    1, n, n, dtype=torch.bool, device=device
                ),
            )

        forward = forward_feature

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model.to(device)
    model.eval()

    return model, dataset, forward
