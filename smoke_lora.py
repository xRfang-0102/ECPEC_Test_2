# =========================================================
# Smoke test for the v3 LoRA pipeline:
#   1. build LoRAECPECModel (ctrl and A+C)
#   2. one forward + backward on a mini-batch
#   3. report LoRA module count, NaN check, GPU memory
# =========================================================

import sys
from pathlib import Path

import torch

import train_feature as tf
from train_feature import set_seed

from dataset.raw_ecf_dataset import RawECFDataset, collate_raw_batch
from models.lora_ecpec_model import LoRAECPECModel
from train_lora import (
    make_forward_model_lora,
    move_batch_to_device_lora,
    build_null_loss_options,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    set_seed(42)

    dataset = RawECFDataset(
        json_path=PROJECT_ROOT / "data/ECF/dev.json",
        max_dialogue_length=40,
        max_utterance_length=128,
        tokenizer_path=PROJECT_ROOT / "pretrained/roberta-base",
    )

    batch = collate_raw_batch([dataset[0], dataset[1]])
    batch = move_batch_to_device_lora(batch, device)

    for name, model_config in (
        ("ctrl", {}),
        (
            "ac2",
            {
                "use_event_retrieval": True,
                "use_speaker_thread": True,
                "use_historical_retrieval": True,
                "use_locality_prior": True,
            },
        ),
    ):
        print("=" * 72)
        print(f"Smoke test: LoRAECPECModel ({name})")
        print("=" * 72)

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

        model = LoRAECPECModel(
            encoder_path=PROJECT_ROOT / "pretrained/roberta-base",
            model_config=model_config,
            use_bf16=True,
        ).to(device)
        model.train()

        trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(f"LoRA modules wrapped: {model.num_lora_modules}")
        print(f"Trainable params: {trainable:,}")

        forward = make_forward_model_lora()
        tf.forward_model = forward
        tf.move_batch_to_device = move_batch_to_device_lora

        outputs = forward(model, batch)

        print("output keys:", sorted(outputs.keys()))

        losses = tf.compute_losses(
            outputs=outputs,
            batch=batch,
            pair_pos_weight=2.5,
            lambda_emotion=0.2,
            lambda_cause=0.4,
            pair_rank_weight=0.3,
            pair_rank_margin=0.2,
            pair_rank_hard_negative_k=3,
            cond_rank_weight=0.3 if name == "ac2" else 0.0,
            cond_rank_margin=0.2,
            cond_rank_hard_negative_k=3,
            retrieval_loss_weight=0.1 if name == "ac2" else 0.0,
            retrieval_local_max_distance=1,
            historical_retrieval_weight=0.1 if name == "ac2" else 0.0,
            locality_prior_reg_weight=(
                0.001 if name == "ac2" else 0.0
            ),
        )

        total = losses["total_loss"]
        print(f"total_loss: {total.item():.6f}")
        print(f"pair_loss: {losses['pair_loss'].item():.6f}")

        for key, value in losses.items():
            if torch.is_tensor(value) and not torch.isfinite(
                value
            ).all():
                print(f"  !! non-finite loss: {key} = {value.item()}")

        total.backward()

        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += float(param.grad.norm().item()) ** 2
        grad_norm = grad_norm ** 0.5
        print(f"grad norm: {grad_norm:.4f}")

        finite_grads = all(
            param.grad is None
            or torch.isfinite(param.grad).all()
            for param in model.parameters()
        )
        print("grads finite:", finite_grads)

        print(
            "peak GPU memory: "
            f"{torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GiB"
        )
        print(
            "GPU memory reserved: "
            f"{torch.cuda.memory_reserved(device) / 1024**3:.2f} GiB"
        )

        del model, outputs, losses, total
        torch.cuda.empty_cache()

    print("=" * 72)
    print("Smoke test done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
