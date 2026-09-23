from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from metrics import (
    ECPECMetricAccumulator,
)


class ECPECTrainer:

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        threshold=0.5,
        max_grad_norm=1.0,
        early_stop_patience=4,
        checkpoint_path="checkpoints/best_model.pt",
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.threshold = float(threshold)
        self.max_grad_norm = max_grad_norm

        self.early_stop_patience = int(
            early_stop_patience
        )

        self.checkpoint_path = Path(
            checkpoint_path
        )

        self.checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.best_pair_f1 = -1.0
        self.best_epoch = -1
        self.bad_epochs = 0

    # ==================================================
    # Move batch
    # ==================================================

    def _move_batch_to_device(
        self,
        batch,
    ):
        tensor_keys = [
            "input_ids",
            "attention_mask",
            "speaker_ids",
            "position_ids",
            "utterance_mask",
            "emotion_labels",
            "cause_labels",
            "pair_labels",
            "pair_mask",
        ]

        for key in tensor_keys:

            batch[key] = (
                batch[key].to(
                    self.device,
                    non_blocking=True,
                )
            )

        return batch

    # ==================================================
    # Forward
    # ==================================================

    def _forward(
        self,
        batch,
    ):
        return self.model(
            input_ids=(
                batch["input_ids"]
            ),

            attention_mask=(
                batch["attention_mask"]
            ),

            speaker_ids=(
                batch["speaker_ids"]
            ),

            position_ids=(
                batch["position_ids"]
            ),

            utterance_mask=(
                batch["utterance_mask"]
            ),

            pair_mask=(
                batch["pair_mask"]
            ),
        )

    # ==================================================
    # Loss
    # ==================================================

    def _compute_loss(
        self,
        outputs,
        batch,
    ):
        return self.criterion(
            outputs=outputs,

            emotion_labels=(
                batch["emotion_labels"]
            ),

            cause_labels=(
                batch["cause_labels"]
            ),

            pair_labels=(
                batch["pair_labels"]
            ),

            utterance_mask=(
                batch["utterance_mask"]
            ),

            pair_mask=(
                batch["pair_mask"]
            ),
        )

    # ==================================================
    # Train one epoch
    # ==================================================

    def train_epoch(
        self,
        train_loader,
        epoch,
        total_epochs,
    ):
        self.model.train()

        metric_accumulator = (
            ECPECMetricAccumulator(
                threshold=self.threshold
            )
        )

        total_loss_sum = 0.0
        pair_loss_sum = 0.0
        emotion_loss_sum = 0.0
        cause_loss_sum = 0.0

        num_batches = 0

        progress_bar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=(
                f"Epoch {epoch:02d}/"
                f"{total_epochs:02d} [Train]"
            ),
            dynamic_ncols=True,
            leave=True,
        )

        for batch_index, batch in enumerate(
            progress_bar,
            start=1,
        ):
            batch = (
                self._move_batch_to_device(
                    batch
                )
            )

            # ------------------------------------------
            # Zero grad
            # ------------------------------------------

            self.optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------
            # Forward
            # ------------------------------------------

            outputs = self._forward(
                batch
            )

            # ------------------------------------------
            # Loss
            # ------------------------------------------

            losses = self._compute_loss(
                outputs,
                batch,
            )

            total_loss = (
                losses["total_loss"]
            )

            if not torch.isfinite(
                total_loss
            ).all():

                raise RuntimeError(
                    "Non-finite training loss "
                    f"at batch {batch_index}: "
                    f"{total_loss.item()}"
                )

            # ------------------------------------------
            # Backward
            # ------------------------------------------

            total_loss.backward()

            # ------------------------------------------
            # Gradient clipping
            # ------------------------------------------

            if (
                self.max_grad_norm
                is not None
                and
                self.max_grad_norm > 0
            ):

                grad_norm = clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )

                if not torch.isfinite(
                    grad_norm
                ):
                    raise RuntimeError(
                        "Non-finite gradient norm "
                        f"at batch {batch_index}."
                    )

            # ------------------------------------------
            # Optimizer
            # ------------------------------------------

            self.optimizer.step()

            # ------------------------------------------
            # Metrics
            # ------------------------------------------

            metric_accumulator.update(
                outputs={
                    "emotion_logits":
                        outputs[
                            "emotion_logits"
                        ].detach(),

                    "cause_logits":
                        outputs[
                            "cause_logits"
                        ].detach(),

                    "pair_logits":
                        outputs[
                            "pair_logits"
                        ].detach(),
                },

                emotion_labels=(
                    batch[
                        "emotion_labels"
                    ]
                ),

                cause_labels=(
                    batch[
                        "cause_labels"
                    ]
                ),

                pair_labels=(
                    batch[
                        "pair_labels"
                    ]
                ),

                utterance_mask=(
                    batch[
                        "utterance_mask"
                    ]
                ),

                pair_mask=(
                    batch[
                        "pair_mask"
                    ]
                ),
            )

            # ------------------------------------------
            # Accumulate loss
            # ------------------------------------------

            total_loss_sum += (
                losses[
                    "total_loss"
                ].item()
            )

            pair_loss_sum += (
                losses[
                    "pair_loss"
                ].item()
            )

            emotion_loss_sum += (
                losses[
                    "emotion_loss"
                ].item()
            )

            cause_loss_sum += (
                losses[
                    "cause_loss"
                ].item()
            )

            num_batches += 1

            # ------------------------------------------
            # Running loss on progress bar
            # ------------------------------------------

            running_total = (
                total_loss_sum
                / num_batches
            )

            running_pair = (
                pair_loss_sum
                / num_batches
            )

            progress_bar.set_postfix(
                loss=f"{running_total:.4f}",
                pair=f"{running_pair:.4f}",
            )

        if num_batches == 0:
            raise RuntimeError(
                "Training DataLoader "
                "contains zero batches."
            )

        metrics = (
            metric_accumulator.compute()
        )

        epoch_losses = {
            "total_loss":
                total_loss_sum
                / num_batches,

            "pair_loss":
                pair_loss_sum
                / num_batches,

            "emotion_loss":
                emotion_loss_sum
                / num_batches,

            "cause_loss":
                cause_loss_sum
                / num_batches,
        }

        return {
            "losses": epoch_losses,
            "metrics": metrics,
        }

    # ==================================================
    # Evaluation
    # ==================================================

    @torch.no_grad()
    def evaluate(
        self,
        data_loader,
        epoch=None,
        total_epochs=None,
        show_progress=True,
    ):
        self.model.eval()

        metric_accumulator = (
            ECPECMetricAccumulator(
                threshold=self.threshold
            )
        )

        total_loss_sum = 0.0
        pair_loss_sum = 0.0
        emotion_loss_sum = 0.0
        cause_loss_sum = 0.0

        num_batches = 0

        if (
            epoch is not None
            and total_epochs is not None
        ):
            description = (
                f"Epoch {epoch:02d}/"
                f"{total_epochs:02d} [Dev]"
            )
        else:
            description = "Evaluation"

        iterator = tqdm(
            data_loader,
            total=len(data_loader),
            desc=description,
            dynamic_ncols=True,
            leave=True,
            disable=not show_progress,
        )

        for batch in iterator:

            batch = (
                self._move_batch_to_device(
                    batch
                )
            )

            outputs = self._forward(
                batch
            )

            losses = self._compute_loss(
                outputs,
                batch,
            )

            if not torch.isfinite(
                losses["total_loss"]
            ).all():

                raise RuntimeError(
                    "Non-finite evaluation loss."
                )

            metric_accumulator.update(
                outputs=outputs,

                emotion_labels=(
                    batch[
                        "emotion_labels"
                    ]
                ),

                cause_labels=(
                    batch[
                        "cause_labels"
                    ]
                ),

                pair_labels=(
                    batch[
                        "pair_labels"
                    ]
                ),

                utterance_mask=(
                    batch[
                        "utterance_mask"
                    ]
                ),

                pair_mask=(
                    batch[
                        "pair_mask"
                    ]
                ),
            )

            total_loss_sum += (
                losses[
                    "total_loss"
                ].item()
            )

            pair_loss_sum += (
                losses[
                    "pair_loss"
                ].item()
            )

            emotion_loss_sum += (
                losses[
                    "emotion_loss"
                ].item()
            )

            cause_loss_sum += (
                losses[
                    "cause_loss"
                ].item()
            )

            num_batches += 1

            iterator.set_postfix(
                loss=(
                    f"{total_loss_sum / num_batches:.4f}"
                )
            )

        if num_batches == 0:
            raise RuntimeError(
                "Evaluation DataLoader "
                "contains zero batches."
            )

        metrics = (
            metric_accumulator.compute()
        )

        epoch_losses = {
            "total_loss":
                total_loss_sum
                / num_batches,

            "pair_loss":
                pair_loss_sum
                / num_batches,

            "emotion_loss":
                emotion_loss_sum
                / num_batches,

            "cause_loss":
                cause_loss_sum
                / num_batches,
        }

        return {
            "losses": epoch_losses,
            "metrics": metrics,
        }

    # ==================================================
    # Pretty epoch report
    # ==================================================

    @staticmethod
    def _print_epoch_result(
        name,
        result,
    ):
        losses = result["losses"]
        metrics = result["metrics"]

        emotion = metrics["emotion"]
        cause = metrics["cause"]
        pair = metrics["pair"]

        print(
            f"{name} Loss: "
            f"{losses['total_loss']:.4f} "
            f"(Pair={losses['pair_loss']:.4f}, "
            f"Emotion={losses['emotion_loss']:.4f}, "
            f"Cause={losses['cause_loss']:.4f})"
        )

        print(
            f"{name} Emotion | "
            f"P={emotion['precision']:.4f} "
            f"R={emotion['recall']:.4f} "
            f"F1={emotion['f1']:.4f}"
        )

        print(
            f"{name} Cause   | "
            f"P={cause['precision']:.4f} "
            f"R={cause['recall']:.4f} "
            f"F1={cause['f1']:.4f}"
        )

        print(
            f"{name} Pair    | "
            f"P={pair['precision']:.4f} "
            f"R={pair['recall']:.4f} "
            f"F1={pair['f1']:.4f} "
            f"#Pred={pair['predicted_positive']} "
            f"#Gold={pair['gold_positive']}"
        )

    # ==================================================
    # Checkpoint
    # ==================================================

    def save_checkpoint(
        self,
        epoch,
        dev_result,
    ):
        checkpoint = {
            "epoch":
                int(epoch),

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "best_pair_f1":
                float(
                    self.best_pair_f1
                ),

            "threshold":
                float(
                    self.threshold
                ),

            "dev_metrics":
                dev_result["metrics"],

            "dev_losses":
                dev_result["losses"],
        }

        torch.save(
            checkpoint,
            self.checkpoint_path,
        )

    def load_best_checkpoint(
        self,
        load_optimizer=False,
    ):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "Checkpoint not found: "
                f"{self.checkpoint_path}"
            )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        if load_optimizer:

            self.optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )

        self.best_pair_f1 = float(
            checkpoint.get(
                "best_pair_f1",
                -1.0,
            )
        )

        self.best_epoch = int(
            checkpoint.get(
                "epoch",
                -1,
            )
        )

        self.threshold = float(
            checkpoint.get(
                "threshold",
                self.threshold,
            )
        )

        return checkpoint

    # ==================================================
    # Fit
    # ==================================================

    def fit(
        self,
        train_loader,
        dev_loader,
        epochs,
    ):
        epochs = int(epochs)

        history = []

        print()
        print(
            "===================================="
        )
        print(
            "ECPEC BaseModel Training"
        )
        print(
            "===================================="
        )

        print(
            f"Selection metric: Dev Pair-F1"
        )

        print(
            f"Threshold: {self.threshold}"
        )

        print(
            f"Early stopping patience: "
            f"{self.early_stop_patience}"
        )

        print(
            f"Gradient clipping: "
            f"{self.max_grad_norm}"
        )

        print()

        # ==================================================
        # Epoch loop
        # ==================================================

        for epoch in range(
            1,
            epochs + 1,
        ):

            # ------------------------------------------
            # Train
            # ------------------------------------------

            train_result = (
                self.train_epoch(
                    train_loader=train_loader,
                    epoch=epoch,
                    total_epochs=epochs,
                )
            )

            # ------------------------------------------
            # Dev
            # ------------------------------------------

            dev_result = (
                self.evaluate(
                    data_loader=dev_loader,
                    epoch=epoch,
                    total_epochs=epochs,
                    show_progress=True,
                )
            )

            dev_pair_f1 = float(
                dev_result[
                    "metrics"
                ]["pair"]["f1"]
            )

            # ------------------------------------------
            # Epoch summary
            # ------------------------------------------

            print()
            print(
                "===================================="
            )

            print(
                f"Epoch {epoch:02d}/{epochs:02d}"
            )

            print(
                "------------------------------------"
            )

            self._print_epoch_result(
                "Train",
                train_result,
            )

            print()

            self._print_epoch_result(
                "Dev  ",
                dev_result,
            )

            # ------------------------------------------
            # Best checkpoint
            # ------------------------------------------

            improved = (
                dev_pair_f1
                > self.best_pair_f1
            )

            if improved:

                self.best_pair_f1 = (
                    dev_pair_f1
                )

                self.best_epoch = (
                    epoch
                )

                self.bad_epochs = 0

                self.save_checkpoint(
                    epoch=epoch,
                    dev_result=dev_result,
                )

                print()
                print(
                    f"New best Dev Pair-F1: "
                    f"{dev_pair_f1:.4f}"
                )

                print(
                    "Checkpoint saved to:"
                )

                print(
                    self.checkpoint_path
                )

            else:

                self.bad_epochs += 1

                print()
                print(
                    "No improvement. "
                    f"Patience: "
                    f"{self.bad_epochs}/"
                    f"{self.early_stop_patience}"
                )

            print(
                "===================================="
            )

            history.append(
                {
                    "epoch":
                        int(epoch),

                    "train":
                        train_result,

                    "dev":
                        dev_result,

                    "is_best":
                        bool(improved),
                }
            )

            # ------------------------------------------
            # Early stopping
            # ------------------------------------------

            if (
                self.bad_epochs
                >=
                self.early_stop_patience
            ):

                print()
                print(
                    "Early stopping triggered."
                )

                break

        print()
        print(
            "===================================="
        )

        print(
            "Training finished."
        )

        print(
            "Best epoch:",
            self.best_epoch,
        )

        print(
            "Best Dev Pair-F1:",
            f"{self.best_pair_f1:.4f}",
        )

        print(
            "Best checkpoint:",
            self.checkpoint_path,
        )

        print(
            "===================================="
        )

        return history


# ======================================================
# Optimizer
# ======================================================

def build_optimizer(
    model,
    encoder_lr=2e-5,
    module_lr=1e-4,
    weight_decay=0.01,
):

    encoder_parameters = []
    module_parameters = []

    for name, parameter in (
        model.named_parameters()
    ):

        if not parameter.requires_grad:
            continue

        if name.startswith(
            "utterance_encoder.encoder."
        ):

            encoder_parameters.append(
                parameter
            )

        else:

            module_parameters.append(
                parameter
            )

    parameter_groups = []

    if encoder_parameters:

        parameter_groups.append(
            {
                "params":
                    encoder_parameters,

                "lr":
                    float(
                        encoder_lr
                    ),

                "weight_decay":
                    float(
                        weight_decay
                    ),
            }
        )

    if module_parameters:

        parameter_groups.append(
            {
                "params":
                    module_parameters,

                "lr":
                    float(
                        module_lr
                    ),

                "weight_decay":
                    float(
                        weight_decay
                    ),
            }
        )

    if not parameter_groups:

        raise RuntimeError(
            "No trainable parameters found."
        )

    return torch.optim.AdamW(
        parameter_groups
    )