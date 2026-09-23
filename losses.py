import torch
import torch.nn as nn
import torch.nn.functional as F


class ECPECLoss(nn.Module):
    """
    Loss function for the ECPEC BaseModel.

    Total loss:
        L = L_pair
            + lambda_emotion * L_emotion
            + lambda_cause * L_cause

    Inputs
    ------
    outputs:
        {
            "emotion_logits": [B, N],
            "cause_logits":   [B, N],
            "pair_logits":    [B, N, N]
        }

    emotion_labels:
        [B, N]

    cause_labels:
        [B, N]

    pair_labels:
        [B, N, N]

    utterance_mask:
        [B, N]

    pair_mask:
        [B, N, N]
    """

    def __init__(
        self,
        lambda_emotion: float = 0.2,
        lambda_cause: float = 0.2,
        pair_pos_weight: float = 1.0,
    ):
        super().__init__()

        if lambda_emotion < 0:
            raise ValueError(
                "lambda_emotion must be >= 0."
            )

        if lambda_cause < 0:
            raise ValueError(
                "lambda_cause must be >= 0."
            )

        if pair_pos_weight <= 0:
            raise ValueError(
                "pair_pos_weight must be > 0."
            )

        self.lambda_emotion = (
            lambda_emotion
        )

        self.lambda_cause = (
            lambda_cause
        )

        # Register as buffer so that it automatically
        # moves to CPU / CUDA together with the model.
        self.register_buffer(
            "pair_pos_weight",
            torch.tensor(
                float(pair_pos_weight),
                dtype=torch.float32,
            ),
        )

    def forward(
        self,
        outputs,
        emotion_labels: torch.Tensor,
        cause_labels: torch.Tensor,
        pair_labels: torch.Tensor,
        utterance_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ):

        self._validate_inputs(
            outputs=outputs,
            emotion_labels=emotion_labels,
            cause_labels=cause_labels,
            pair_labels=pair_labels,
            utterance_mask=utterance_mask,
            pair_mask=pair_mask,
        )

        emotion_logits = (
            outputs["emotion_logits"]
        )

        cause_logits = (
            outputs["cause_logits"]
        )

        pair_logits = (
            outputs["pair_logits"]
        )

        # --------------------------------------------------
        # 1. Emotion loss
        # --------------------------------------------------

        emotion_loss = (
            self._masked_bce_loss(
                logits=emotion_logits,
                labels=emotion_labels,
                mask=utterance_mask,
            )
        )

        # --------------------------------------------------
        # 2. Cause loss
        # --------------------------------------------------

        cause_loss = (
            self._masked_bce_loss(
                logits=cause_logits,
                labels=cause_labels,
                mask=utterance_mask,
            )
        )

        # --------------------------------------------------
        # 3. Pair loss
        #
        # Weighted BCE:
        #
        # positive examples receive pair_pos_weight.
        #
        # IMPORTANT:
        # pair_pos_weight should be calculated from the
        # TRAINING SET only, not separately per batch.
        # --------------------------------------------------

        pair_loss = (
            self._masked_pair_bce_loss(
                logits=pair_logits,
                labels=pair_labels,
                mask=pair_mask,
            )
        )

        # --------------------------------------------------
        # 4. Total loss
        # --------------------------------------------------

        total_loss = (
            pair_loss
            + self.lambda_emotion
            * emotion_loss
            + self.lambda_cause
            * cause_loss
        )

        return {
            "loss": total_loss,
            "total_loss": total_loss,
            "pair_loss": pair_loss,
            "emotion_loss": emotion_loss,
            "cause_loss": cause_loss,
        }

    @staticmethod
    def _masked_bce_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        valid_mask = mask.bool()

        valid_logits = logits[
            valid_mask
        ]

        valid_labels = labels[
            valid_mask
        ].float()

        if valid_logits.numel() == 0:
            return logits.sum() * 0.0

        loss = F.binary_cross_entropy_with_logits(
            valid_logits,
            valid_labels,
            reduction="mean",
        )

        return loss

    def _masked_pair_bce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        valid_mask = mask.bool()

        valid_logits = logits[
            valid_mask
        ]

        valid_labels = labels[
            valid_mask
        ].float()

        if valid_logits.numel() == 0:
            return logits.sum() * 0.0

        loss = F.binary_cross_entropy_with_logits(
            valid_logits,
            valid_labels,
            pos_weight=self.pair_pos_weight,
            reduction="mean",
        )

        return loss

    @staticmethod
    def _validate_inputs(
        outputs,
        emotion_labels: torch.Tensor,
        cause_labels: torch.Tensor,
        pair_labels: torch.Tensor,
        utterance_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> None:

        required_outputs = {
            "emotion_logits",
            "cause_logits",
            "pair_logits",
        }

        missing = (
            required_outputs
            - set(outputs.keys())
        )

        if missing:
            raise KeyError(
                "Missing model outputs: "
                f"{sorted(missing)}"
            )

        emotion_logits = (
            outputs["emotion_logits"]
        )

        cause_logits = (
            outputs["cause_logits"]
        )

        pair_logits = (
            outputs["pair_logits"]
        )

        if (
            emotion_logits.shape
            != emotion_labels.shape
        ):
            raise ValueError(
                "emotion_logits and "
                "emotion_labels shape mismatch: "
                f"{tuple(emotion_logits.shape)} vs "
                f"{tuple(emotion_labels.shape)}."
            )

        if (
            cause_logits.shape
            != cause_labels.shape
        ):
            raise ValueError(
                "cause_logits and "
                "cause_labels shape mismatch: "
                f"{tuple(cause_logits.shape)} vs "
                f"{tuple(cause_labels.shape)}."
            )

        if (
            pair_logits.shape
            != pair_labels.shape
        ):
            raise ValueError(
                "pair_logits and "
                "pair_labels shape mismatch: "
                f"{tuple(pair_logits.shape)} vs "
                f"{tuple(pair_labels.shape)}."
            )

        if (
            utterance_mask.shape
            != emotion_labels.shape
        ):
            raise ValueError(
                "utterance_mask shape mismatch: "
                f"{tuple(utterance_mask.shape)} vs "
                f"{tuple(emotion_labels.shape)}."
            )

        if (
            pair_mask.shape
            != pair_labels.shape
        ):
            raise ValueError(
                "pair_mask shape mismatch: "
                f"{tuple(pair_mask.shape)} vs "
                f"{tuple(pair_labels.shape)}."
            )


def compute_pair_pos_weight(
    dataset,
) -> float:
    """
    Compute global pair positive weight from a TRAINING dataset.

    pos_weight = negative_pairs / positive_pairs

    This function should be called once on the training set.

    Do NOT calculate a separate weight for dev/test.
    Do NOT calculate it independently for every batch.
    """

    positive = 0
    negative = 0

    for index in range(len(dataset)):

        sample = dataset[index]

        pair_labels = (
            sample["pair_labels"]
        )

        if pair_labels.dim() != 2:
            raise ValueError(
                "Each pair_labels tensor must "
                "have shape [N, N]."
            )

        labels = pair_labels.float()

        positive += int(
            (labels == 1)
            .sum()
            .item()
        )

        negative += int(
            (labels == 0)
            .sum()
            .item()
        )

    if positive == 0:
        raise ValueError(
            "Training dataset contains "
            "zero positive pairs."
        )

    pos_weight = (
        negative
        / positive
    )

    return float(pos_weight)


def build_loss(
    lambda_emotion: float = 0.2,
    lambda_cause: float = 0.2,
    pair_pos_weight: float = 1.0,
) -> ECPECLoss:

    return ECPECLoss(
        lambda_emotion=lambda_emotion,
        lambda_cause=lambda_cause,
        pair_pos_weight=pair_pos_weight,
    )