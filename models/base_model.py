import torch
import torch.nn as nn

from models.utterance_encoder import UtteranceEncoder
from models.dialogue_encoder import DialogueEncoder
from models.auxiliary_heads import AuxiliaryHeads
from models.pair_classifier import PairClassifier


class ECPECBaseModel(nn.Module):
    """
    Complete baseline model for text-based ECPEC.

    Pipeline
    --------
    input_ids / attention_mask
            ↓
    UtteranceEncoder
            ↓
    [B, N, 768]
            ↓
    DialogueEncoder
            ↓
    [B, N, 256]
       ┌───────────────┐
       ↓               ↓
    AuxiliaryHeads   PairClassifier
       ↓               ↓
    emotion_logits   pair_logits
    cause_logits

    Outputs
    -------
    emotion_logits:
        [B, N]

    cause_logits:
        [B, N]

    pair_logits:
        [B, N, N]
    """

    def __init__(
        self,
        encoder_name: str = "pretrained/roberta-base",
        roberta_dropout: float = 0.1,
        freeze_encoder: bool = False,

        dialogue_hidden: int = 256,
        speaker_embedding_dim: int = 32,
        position_embedding_dim: int = 32,
        max_speakers: int = 20,
        max_positions: int = 64,
        dialogue_layers: int = 2,
        dialogue_heads: int = 4,
        dialogue_ffn: int = 1024,

        auxiliary_hidden: int = 128,

        pair_hidden: int = 256,
        distance_embedding_dim: int = 32,
        speaker_relation_embedding_dim: int = 16,
        max_relative_distance: int = 15,

        dropout: float = 0.1,
    ):
        super().__init__()

        # --------------------------------------------------
        # 1. Utterance encoder
        # --------------------------------------------------

        self.utterance_encoder = UtteranceEncoder(
            encoder_name=encoder_name,
            dropout=roberta_dropout,
            freeze_encoder=freeze_encoder,
        )

        utterance_hidden = (
            self.utterance_encoder.hidden_size
        )

        # --------------------------------------------------
        # 2. Dialogue encoder
        # --------------------------------------------------

        self.dialogue_encoder = DialogueEncoder(
            input_size=utterance_hidden,
            hidden_size=dialogue_hidden,
            speaker_embedding_dim=(
                speaker_embedding_dim
            ),
            position_embedding_dim=(
                position_embedding_dim
            ),
            max_speakers=max_speakers,
            max_positions=max_positions,
            num_layers=dialogue_layers,
            num_heads=dialogue_heads,
            ff_dim=dialogue_ffn,
            dropout=dropout,
        )

        # --------------------------------------------------
        # 3. Emotion / Cause auxiliary heads
        # --------------------------------------------------

        self.auxiliary_heads = AuxiliaryHeads(
            input_size=dialogue_hidden,
            hidden_size=auxiliary_hidden,
            dropout=dropout,
        )

        # --------------------------------------------------
        # 4. Pair classifier
        # --------------------------------------------------

        self.pair_classifier = PairClassifier(
            hidden_size=dialogue_hidden,
            pair_hidden_size=pair_hidden,
            distance_embedding_dim=(
                distance_embedding_dim
            ),
            speaker_relation_embedding_dim=(
                speaker_relation_embedding_dim
            ),
            max_relative_distance=(
                max_relative_distance
            ),
            dropout=dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        speaker_ids: torch.Tensor,
        position_ids: torch.Tensor,
        utterance_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        return_features: bool = False,
    ):

        self._validate_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            speaker_ids=speaker_ids,
            position_ids=position_ids,
            utterance_mask=utterance_mask,
            pair_mask=pair_mask,
        )

        # --------------------------------------------------
        # 1. Utterance-level encoding
        #
        # [B, N, L]
        #     ↓
        # [B, N, 768]
        # --------------------------------------------------

        utterance_features = (
            self.utterance_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                utterance_mask=utterance_mask,
            )
        )

        # --------------------------------------------------
        # 2. Dialogue-level context modeling
        #
        # [B, N, 768]
        #     ↓
        # [B, N, 256]
        # --------------------------------------------------

        context_features = (
            self.dialogue_encoder(
                utterance_features=(
                    utterance_features
                ),
                speaker_ids=speaker_ids,
                position_ids=position_ids,
                utterance_mask=(
                    utterance_mask
                ),
            )
        )

        # --------------------------------------------------
        # 3. Emotion / Cause auxiliary prediction
        # --------------------------------------------------

        auxiliary_outputs = (
            self.auxiliary_heads(
                context_features=(
                    context_features
                ),
                utterance_mask=(
                    utterance_mask
                ),
            )
        )

        emotion_logits = (
            auxiliary_outputs[
                "emotion_logits"
            ]
        )

        cause_logits = (
            auxiliary_outputs[
                "cause_logits"
            ]
        )

        # --------------------------------------------------
        # 4. Pair prediction
        #
        # [B, N, 256]
        #     ↓
        # [B, N, N]
        # --------------------------------------------------

        pair_logits = (
            self.pair_classifier(
                context_features=(
                    context_features
                ),
                speaker_ids=speaker_ids,
                position_ids=position_ids,
                pair_mask=pair_mask,
            )
        )

        outputs = {
            "emotion_logits": (
                emotion_logits
            ),

            "cause_logits": (
                cause_logits
            ),

            "pair_logits": (
                pair_logits
            ),
        }

        # Useful later for visualization /
        # error analysis / innovation modules.
        if return_features:
            outputs.update(
                {
                    "utterance_features": (
                        utterance_features
                    ),

                    "context_features": (
                        context_features
                    ),
                }
            )

        return outputs

    @staticmethod
    def _validate_inputs(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        speaker_ids: torch.Tensor,
        position_ids: torch.Tensor,
        utterance_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> None:

        if input_ids.dim() != 3:
            raise ValueError(
                "input_ids must have shape "
                "[B, N, L]."
            )

        if (
            input_ids.shape
            != attention_mask.shape
        ):
            raise ValueError(
                "input_ids and attention_mask "
                "must have identical shapes."
            )

        batch_size = input_ids.size(0)
        num_utterances = input_ids.size(1)

        utterance_shape = (
            batch_size,
            num_utterances,
        )

        pair_shape = (
            batch_size,
            num_utterances,
            num_utterances,
        )

        if (
            tuple(speaker_ids.shape)
            != utterance_shape
        ):
            raise ValueError(
                "speaker_ids shape mismatch: "
                f"expected {utterance_shape}, "
                f"got "
                f"{tuple(speaker_ids.shape)}."
            )

        if (
            tuple(position_ids.shape)
            != utterance_shape
        ):
            raise ValueError(
                "position_ids shape mismatch: "
                f"expected {utterance_shape}, "
                f"got "
                f"{tuple(position_ids.shape)}."
            )

        if (
            tuple(utterance_mask.shape)
            != utterance_shape
        ):
            raise ValueError(
                "utterance_mask shape mismatch: "
                f"expected {utterance_shape}, "
                f"got "
                f"{tuple(utterance_mask.shape)}."
            )

        if (
            tuple(pair_mask.shape)
            != pair_shape
        ):
            raise ValueError(
                "pair_mask shape mismatch: "
                f"expected {pair_shape}, "
                f"got "
                f"{tuple(pair_mask.shape)}."
            )


def build_base_model(
    encoder_name: str = "pretrained/roberta-base",

    roberta_dropout: float = 0.1,
    freeze_encoder: bool = False,

    dialogue_hidden: int = 256,
    speaker_embedding_dim: int = 32,
    position_embedding_dim: int = 32,

    max_speakers: int = 20,
    max_positions: int = 64,

    dialogue_layers: int = 2,
    dialogue_heads: int = 4,
    dialogue_ffn: int = 1024,

    auxiliary_hidden: int = 128,

    pair_hidden: int = 256,
    distance_embedding_dim: int = 32,
    speaker_relation_embedding_dim: int = 16,
    max_relative_distance: int = 15,

    dropout: float = 0.1,
) -> ECPECBaseModel:

    return ECPECBaseModel(
        encoder_name=encoder_name,

        roberta_dropout=roberta_dropout,
        freeze_encoder=freeze_encoder,

        dialogue_hidden=dialogue_hidden,

        speaker_embedding_dim=(
            speaker_embedding_dim
        ),

        position_embedding_dim=(
            position_embedding_dim
        ),

        max_speakers=max_speakers,
        max_positions=max_positions,

        dialogue_layers=dialogue_layers,
        dialogue_heads=dialogue_heads,
        dialogue_ffn=dialogue_ffn,

        auxiliary_hidden=(
            auxiliary_hidden
        ),

        pair_hidden=pair_hidden,

        distance_embedding_dim=(
            distance_embedding_dim
        ),

        speaker_relation_embedding_dim=(
            speaker_relation_embedding_dim
        ),

        max_relative_distance=(
            max_relative_distance
        ),

        dropout=dropout,
    )