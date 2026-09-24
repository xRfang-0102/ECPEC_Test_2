# =========================================================
# Demo: predict emotion-cause pairs for one dialogue with
# the best model (v3a LoRA-ECPEC).
#
# Usage:
#   python predict_pairs.py --checkpoint checkpoints/base_lora_ctrl_best.pt \
#       --config config/config_lora_ctrl.yaml --split dev --dialogue-id 0
# =========================================================

import argparse
from pathlib import Path

import torch

from train_feature import load_config
from analysis_common import load_model_and_dataset

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict emotion-cause pairs for one dialogue."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/base_lora_ctrl_best.pt",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_lora_ctrl.yaml",
    )
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--dialogue-id", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.66)
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    config = load_config(config_path)

    model, dataset, forward = load_model_and_dataset(
        checkpoint_path,
        config,
        split=args.split,
        device=device,
        use_bf16=False,
    )

    sample = dataset[args.dialogue_id]
    n = int(sample["num_utterances"])

    with torch.no_grad():
        out = forward(model, sample)

    pair_logits = out["pair_logits"][0].cpu()
    labels = sample["pair_labels"][:n, :n]

    if hasattr(dataset, "base"):
        utterances = list(dataset.base[args.dialogue_id]["utterances"])
    else:
        utterances = [f"u{i}" for i in range(n)]

    speaker_ids = sample["speaker_ids"][:n].tolist()
    speakers = [f"spk{s}" for s in speaker_ids]

    print("=" * 72)
    print(
        f"Dialogue {args.dialogue_id} ({args.split}), "
        f"{n} utterances"
    )
    print("=" * 72)
    for i in range(n):
        print(f"[{i}] ({speakers[i]}) {utterances[i][:90]}")

    print()
    print("Gold pairs (emotion <- cause):")
    gold = (labels == 1).nonzero(as_tuple=False)
    for i, j in gold:
        print(
            f"  emotion u{i} <- cause u{j} "
            f"(dist={int(i) - int(j)}, "
            f"same_spk={speakers[int(i)] == speakers[int(j)]})"
        )
    if gold.numel() == 0:
        print("  (none)")

    probs = torch.sigmoid(pair_logits)
    preds = (probs >= args.threshold).nonzero(as_tuple=False)

    print()
    print(f"Predicted pairs (p >= {args.threshold}):")
    for i, j in preds:
        print(
            f"  emotion u{i} <- cause u{j} "
            f"(p={probs[int(i), int(j)].item():.3f}, "
            f"dist={int(i) - int(j)}, "
            f"same_spk={speakers[int(i)] == speakers[int(j)]})"
        )
    if preds.numel() == 0:
        print("  (none)")

    print()
    tp = sum(
        1 for i, j in preds.tolist()
        if labels[int(i), int(j)].item() == 1
    )
    print(
        f"Summary: {len(preds)} predictions, "
        f"{tp} match gold ({len(gold)} gold pairs)"
    )


if __name__ == "__main__":
    main()
