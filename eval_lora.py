# =========================================================
# Evaluate a v3 LoRA checkpoint on dev and test.
#
# Usage: python eval_lora.py [checkpoint] [config]
# =========================================================

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import train_feature as tf
from train_feature import set_seed, evaluate
from train_lora import (
    make_forward_model_lora,
    move_batch_to_device_lora,
    build_null_loss_options,
    load_lora_checkpoint,
)
from dataset.raw_ecf_dataset import RawECFDataset, collate_raw_batch
from models.lora_ecpec_model import LoRAECPECModel

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    checkpoint_path = Path(sys.argv[1])
    config_path = Path(sys.argv[2])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    config = tf.load_config(config_path)
    set_seed(int(config.get("seed", 42)))
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    eval_config = config.get("evaluation", {})

    encoder_path = (
        PROJECT_ROOT
        / str(model_config.get("encoder", "pretrained/roberta-base"))
    )

    model = LoRAECPECModel(
        encoder_path=encoder_path,
        model_config=model_config,
        lora_r=int(model_config.get("lora_r", 16)),
        lora_alpha=float(model_config.get("lora_alpha", 32)),
        lora_dropout=float(model_config.get("lora_dropout", 0.1)),
        use_bf16=bool(training_config.get("use_bf16", True)),
    ).to(device)

    checkpoint = load_lora_checkpoint(
        checkpoint_path,
        model,
        device,
    )
    model.eval()

    print(f"Loaded {checkpoint_path.name}")
    print(
        f"  epoch {checkpoint.get('epoch')}, "
        f"best_pair_f1 {checkpoint.get('best_pair_f1')}, "
        f"pair_auprc {checkpoint.get('pair_auprc')}"
    )

    tf.forward_model = make_forward_model_lora()
    tf.move_batch_to_device = move_batch_to_device_lora

    null_loss_options = build_null_loss_options(
        training_config,
        model,
    )

    threshold = float(eval_config.get("pair_threshold", 0.66))
    row_normalize = bool(eval_config.get("row_normalize", False))

    pair_pos_weight = float(
        training_config.get("pair_pos_weight", 2.5)
    )
    lambda_emotion = float(
        training_config.get("lambda_emotion", 0.2)
    )
    lambda_cause = float(
        training_config.get("lambda_cause", 0.4)
    )
    batch_size = int(training_config.get("batch_size", 16))

    data_root = PROJECT_ROOT / data_config["data_root"]

    print("=" * 72)
    print(f"Evaluating @ threshold {threshold}")
    print("=" * 72)

    for split in ("dev", "test"):
        dataset = RawECFDataset(
            json_path=data_root / data_config[f"{split}_file"],
            max_dialogue_length=int(
                data_config.get("max_dialogue_length", 40)
            ),
            max_utterance_length=int(
                data_config.get("max_utterance_length", 128)
            ),
            tokenizer_path=encoder_path,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_raw_batch,
        )

        result = evaluate(
            model=model,
            null_loss_options=null_loss_options,
            loader=loader,
            device=device,
            pair_pos_weight=pair_pos_weight,
            lambda_emotion=lambda_emotion,
            lambda_cause=lambda_cause,
            threshold=threshold,
            description=split.capitalize(),
            row_normalize=row_normalize,
        )

        pair = result["metrics"]["pair"]
        emotion = result["metrics"]["emotion"]
        cause = result["metrics"]["cause"]

        print(f"[{split.capitalize()}]")
        print(
            f"  Emotion | P={emotion['precision']:.4f} "
            f"R={emotion['recall']:.4f} F1={emotion['f1']:.4f}"
        )
        print(
            f"  Cause   | P={cause['precision']:.4f} "
            f"R={cause['recall']:.4f} F1={cause['f1']:.4f}"
        )
        print(
            f"  Pair    | P={pair['precision']:.4f} "
            f"R={pair['recall']:.4f} F1={pair['f1']:.4f} "
            f"(#Pred={pair['predicted_positive']}, "
            f"#Gold={pair['gold_positive']})"
        )
        print(f"  Pair AUPRC | {result['pair_auprc']:.4f}")

    print("=" * 72)


if __name__ == "__main__":
    main()
