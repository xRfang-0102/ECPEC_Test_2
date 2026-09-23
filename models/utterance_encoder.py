from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel


class UtteranceEncoder(nn.Module):
    """
    Encode each utterance independently with a pretrained
    Transformer language model.

    Input
    -----
    input_ids:
        [B, N, L]

    attention_mask:
        [B, N, L]

    utterance_mask:
        [B, N]

    Output
    ------
    utterance_features:
        [B, N, H]

    where:
        B = batch size
        N = maximum dialogue length
        L = maximum token length
        H = pretrained encoder hidden size

    For roberta-base:
        H = 768
    """

    def __init__(
        self,
        encoder_name: str = "roberta-base",
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        self.encoder_name = encoder_name
        self.freeze_encoder = freeze_encoder

        self.encoder = AutoModel.from_pretrained(
            encoder_name,
            local_files_only=True,
            add_pooling_layer=False,
        )

        self.hidden_size = (
            self.encoder.config.hidden_size
        )

        self.dropout = nn.Dropout(
            dropout
        )

        if freeze_encoder:
            self._freeze_encoder()

    def _freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

        self.freeze_encoder = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        utterance_mask: Optional[
            torch.Tensor
        ] = None,
    ) -> torch.Tensor:

        self._validate_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            utterance_mask=utterance_mask,
        )

        batch_size, num_utterances, seq_length = (
            input_ids.shape
        )

        # --------------------------------------------------
        # 1. Flatten dialogue dimension
        #
        # [B, N, L]
        #     ->
        # [B*N, L]
        # --------------------------------------------------

        flat_input_ids = input_ids.reshape(
            batch_size * num_utterances,
            seq_length,
        )

        flat_attention_mask = (
            attention_mask.reshape(
                batch_size * num_utterances,
                seq_length,
            )
        )

        # --------------------------------------------------
        # 2. Identify real utterances
        #
        # Padding utterances should not be sent through
        # RoBERTa.
        # --------------------------------------------------

        if utterance_mask is None:
            flat_valid_mask = (
                flat_attention_mask.sum(dim=1) > 0
            )
        else:
            flat_valid_mask = (
                utterance_mask.reshape(-1).bool()
            )

        valid_indices = torch.nonzero(
            flat_valid_mask,
            as_tuple=False,
        ).squeeze(-1)

        # --------------------------------------------------
        # 3. Allocate complete output
        # --------------------------------------------------

        flat_features = torch.zeros(
            (
                batch_size * num_utterances,
                self.hidden_size,
            ),
            dtype=self.encoder.dtype,
            device=input_ids.device,
        )

        # --------------------------------------------------
        # 4. Encode only real utterances
        # --------------------------------------------------

        if valid_indices.numel() > 0:

            valid_input_ids = (
                flat_input_ids[
                    valid_indices
                ]
            )

            valid_attention_mask = (
                flat_attention_mask[
                    valid_indices
                ]
            )

            outputs = self.encoder(
                input_ids=valid_input_ids,
                attention_mask=(
                    valid_attention_mask
                ),
                return_dict=True,
            )

            # RoBERTa:
            # position 0 corresponds to <s>
            valid_features = (
                outputs.last_hidden_state[
                    :,
                    0,
                    :
                ]
            )

            valid_features = self.dropout(
                valid_features
            )

            flat_features[
                valid_indices
            ] = valid_features

        # --------------------------------------------------
        # 5. Restore dialogue structure
        #
        # [B*N, H]
        #     ->
        # [B, N, H]
        # --------------------------------------------------

        utterance_features = (
            flat_features.reshape(
                batch_size,
                num_utterances,
                self.hidden_size,
            )
        )

        # Ensure padded utterances remain zero
        if utterance_mask is not None:
            utterance_features = (
                utterance_features
                * utterance_mask
                .unsqueeze(-1)
                .to(
                    utterance_features.dtype
                )
            )

        return utterance_features

    @staticmethod
    def _validate_inputs(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        utterance_mask: Optional[
            torch.Tensor
        ],
    ) -> None:

        if input_ids.dim() != 3:
            raise ValueError(
                "input_ids must have shape "
                "[B, N, L], "
                f"but got {tuple(input_ids.shape)}."
            )

        if attention_mask.dim() != 3:
            raise ValueError(
                "attention_mask must have shape "
                "[B, N, L], "
                f"but got "
                f"{tuple(attention_mask.shape)}."
            )

        if (
            input_ids.shape
            != attention_mask.shape
        ):
            raise ValueError(
                "input_ids and attention_mask "
                "must have the same shape."
            )

        if utterance_mask is not None:

            if utterance_mask.dim() != 2:
                raise ValueError(
                    "utterance_mask must have "
                    "shape [B, N], "
                    f"but got "
                    f"{tuple(utterance_mask.shape)}."
                )

            expected_shape = (
                input_ids.size(0),
                input_ids.size(1),
            )

            if (
                tuple(utterance_mask.shape)
                != expected_shape
            ):
                raise ValueError(
                    "utterance_mask shape mismatch. "
                    f"Expected {expected_shape}, "
                    f"got "
                    f"{tuple(utterance_mask.shape)}."
                )


def build_utterance_encoder(
    encoder_name: str = "roberta-base",
    dropout: float = 0.1,
    freeze_encoder: bool = False,
) -> UtteranceEncoder:

    return UtteranceEncoder(
        encoder_name=encoder_name,
        dropout=dropout,
        freeze_encoder=freeze_encoder,
    )