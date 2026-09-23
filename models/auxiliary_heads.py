import torch
import torch.nn as nn


class BinaryAuxiliaryHead(nn.Module):
    """
    Generic binary classification head for utterance-level prediction.

    Input
    -----
    features:
        [B, N, H]

    utterance_mask:
        [B, N]
        True  = valid utterance
        False = padding

    Output
    ------
    logits:
        [B, N]
    """

    def __init__(
        self,
        input_size: int = 256,
        hidden_size: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(
                input_size,
                hidden_size,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_size,
                1,
            ),
        )

    def forward(
        self,
        features: torch.Tensor,
        utterance_mask: torch.Tensor = None,
    ) -> torch.Tensor:

        self._validate_inputs(
            features=features,
            utterance_mask=utterance_mask,
        )

        # [B, N, H]
        #     ↓
        # [B, N, 1]
        logits = self.classifier(
            features
        )

        # [B, N, 1]
        #     ↓
        # [B, N]
        logits = logits.squeeze(-1)

        if utterance_mask is not None:
            logits = logits.masked_fill(
                ~utterance_mask.bool(),
                0.0,
            )

        return logits

    def _validate_inputs(
        self,
        features: torch.Tensor,
        utterance_mask: torch.Tensor = None,
    ) -> None:

        if features.dim() != 3:
            raise ValueError(
                "features must have shape "
                "[B, N, H], "
                f"got {tuple(features.shape)}."
            )

        if (
            features.size(-1)
            != self.input_size
        ):
            raise ValueError(
                "Unexpected feature size: "
                f"expected {self.input_size}, "
                f"got {features.size(-1)}."
            )

        if utterance_mask is not None:

            if utterance_mask.dim() != 2:
                raise ValueError(
                    "utterance_mask must have "
                    "shape [B, N]."
                )

            expected_shape = (
                features.size(0),
                features.size(1),
            )

            if (
                tuple(utterance_mask.shape)
                != expected_shape
            ):
                raise ValueError(
                    "utterance_mask shape mismatch: "
                    f"expected {expected_shape}, "
                    f"got "
                    f"{tuple(utterance_mask.shape)}."
                )


class AuxiliaryHeads(nn.Module):
    """
    Emotion and Cause auxiliary prediction heads.

    Input
    -----
    context_features:
        [B, N, H]

    utterance_mask:
        [B, N]

    Output
    ------
    emotion_logits:
        [B, N]

    cause_logits:
        [B, N]

    Notes
    -----
    These heads provide auxiliary supervision only.

    They do NOT filter candidate emotion-cause pairs.
    """

    def __init__(
        self,
        input_size: int = 256,
        hidden_size: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.emotion_head = (
            BinaryAuxiliaryHead(
                input_size=input_size,
                hidden_size=hidden_size,
                dropout=dropout,
            )
        )

        self.cause_head = (
            BinaryAuxiliaryHead(
                input_size=input_size,
                hidden_size=hidden_size,
                dropout=dropout,
            )
        )

    def forward(
        self,
        context_features: torch.Tensor,
        utterance_mask: torch.Tensor = None,
    ):
        emotion_logits = self.emotion_head(
            features=context_features,
            utterance_mask=utterance_mask,
        )

        cause_logits = self.cause_head(
            features=context_features,
            utterance_mask=utterance_mask,
        )

        return {
            "emotion_logits": emotion_logits,
            "cause_logits": cause_logits,
        }


def build_auxiliary_heads(
    input_size: int = 256,
    hidden_size: int = 128,
    dropout: float = 0.1,
) -> AuxiliaryHeads:

    return AuxiliaryHeads(
        input_size=input_size,
        hidden_size=hidden_size,
        dropout=dropout,
    )