import torch
import torch.nn as nn


class PairClassifier(nn.Module):
    """
    Pair-level classifier for ECPEC.

    Pair convention
    ---------------
    pair (i, j):

        i = emotion-side utterance
        j = cause-side utterance

    Inputs
    ------
    context_features:
        [B, N, H]

    speaker_ids:
        [B, N]

    position_ids:
        [B, N]

    pair_mask:
        [B, N, N]
        True  = valid pair
        False = padding pair

    Output
    ------
    pair_logits:
        [B, N, N]
    """

    def __init__(
        self,
        hidden_size: int = 256,
        pair_hidden_size: int = 256,
        distance_embedding_dim: int = 32,
        speaker_relation_embedding_dim: int = 16,
        max_relative_distance: int = 15,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.max_relative_distance = (
            max_relative_distance
        )

        # Relative distance:
        #
        # distance = emotion_position - cause_position
        #
        # Range:
        # [-max_distance, +max_distance]
        #
        # Number of embeddings:
        # 2 * max_distance + 1
        self.distance_embedding = nn.Embedding(
            num_embeddings=(
                2 * max_relative_distance + 1
            ),
            embedding_dim=(
                distance_embedding_dim
            ),
        )

        # Speaker relation:
        #
        # 0 = different speaker
        # 1 = same speaker
        self.speaker_relation_embedding = (
            nn.Embedding(
                num_embeddings=2,
                embedding_dim=(
                    speaker_relation_embedding_dim
                ),
            )
        )

        # Pair representation:
        #
        # h_i
        # h_j
        # h_i * h_j
        # |h_i - h_j|
        #
        # => 4 * hidden_size
        #
        # + distance embedding
        # + speaker relation embedding
        pair_input_size = (
            4 * hidden_size
            + distance_embedding_dim
            + speaker_relation_embedding_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                pair_input_size,
                pair_hidden_size,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                pair_hidden_size,
                1,
            ),
        )

    def forward(
        self,
        context_features: torch.Tensor,
        speaker_ids: torch.Tensor,
        position_ids: torch.Tensor,
        pair_mask: torch.Tensor = None,
    ) -> torch.Tensor:

        self._validate_inputs(
            context_features=context_features,
            speaker_ids=speaker_ids,
            position_ids=position_ids,
            pair_mask=pair_mask,
        )

        batch_size = context_features.size(0)
        num_utterances = context_features.size(1)

        # --------------------------------------------------
        # 1. Construct emotion-side and cause-side features
        #
        # emotion_features:
        # [B, N, 1, H]
        #
        # cause_features:
        # [B, 1, N, H]
        # --------------------------------------------------

        emotion_features = (
            context_features.unsqueeze(2)
        )

        cause_features = (
            context_features.unsqueeze(1)
        )

        # Expand:
        # [B, N, N, H]
        emotion_features = (
            emotion_features.expand(
                -1,
                -1,
                num_utterances,
                -1,
            )
        )

        cause_features = (
            cause_features.expand(
                -1,
                num_utterances,
                -1,
                -1,
            )
        )

        # --------------------------------------------------
        # 2. Basic pair interaction
        # --------------------------------------------------

        element_product = (
            emotion_features
            * cause_features
        )

        absolute_difference = torch.abs(
            emotion_features
            - cause_features
        )

        # --------------------------------------------------
        # 3. Relative distance embedding
        #
        # distance = emotion_pos - cause_pos
        #
        # [B, N, N]
        # --------------------------------------------------

        emotion_positions = (
            position_ids.unsqueeze(2)
        )

        cause_positions = (
            position_ids.unsqueeze(1)
        )

        relative_distance = (
            emotion_positions
            - cause_positions
        )

        relative_distance = (
            relative_distance.clamp(
                min=-self.max_relative_distance,
                max=self.max_relative_distance,
            )
        )

        # Shift:
        #
        # [-15, +15]
        #     ->
        # [0, 30]
        distance_indices = (
            relative_distance
            + self.max_relative_distance
        )

        distance_features = (
            self.distance_embedding(
                distance_indices
            )
        )

        # --------------------------------------------------
        # 4. Speaker relation embedding
        #
        # same speaker      -> 1
        # different speaker -> 0
        #
        # [B, N, N]
        # --------------------------------------------------

        emotion_speakers = (
            speaker_ids.unsqueeze(2)
        )

        cause_speakers = (
            speaker_ids.unsqueeze(1)
        )

        same_speaker = (
            emotion_speakers
            == cause_speakers
        ).long()

        speaker_relation_features = (
            self.speaker_relation_embedding(
                same_speaker
            )
        )

        # --------------------------------------------------
        # 5. Concatenate pair representation
        #
        # [B, N, N, pair_dim]
        # --------------------------------------------------

        pair_features = torch.cat(
            [
                emotion_features,
                cause_features,
                element_product,
                absolute_difference,
                distance_features,
                speaker_relation_features,
            ],
            dim=-1,
        )

        # --------------------------------------------------
        # 6. Pair classification
        #
        # [B, N, N, pair_dim]
        #     ->
        # [B, N, N, 1]
        #     ->
        # [B, N, N]
        # --------------------------------------------------

        pair_logits = self.classifier(
            pair_features
        ).squeeze(-1)

        # --------------------------------------------------
        # 7. Mask padding pairs
        # --------------------------------------------------

        if pair_mask is not None:
            pair_logits = (
                pair_logits.masked_fill(
                    ~pair_mask.bool(),
                    0.0,
                )
            )

        return pair_logits

    def _validate_inputs(
        self,
        context_features: torch.Tensor,
        speaker_ids: torch.Tensor,
        position_ids: torch.Tensor,
        pair_mask: torch.Tensor = None,
    ) -> None:

        if context_features.dim() != 3:
            raise ValueError(
                "context_features must have "
                "shape [B, N, H], "
                f"got "
                f"{tuple(context_features.shape)}."
            )

        if speaker_ids.dim() != 2:
            raise ValueError(
                "speaker_ids must have "
                "shape [B, N], "
                f"got "
                f"{tuple(speaker_ids.shape)}."
            )

        if position_ids.dim() != 2:
            raise ValueError(
                "position_ids must have "
                "shape [B, N], "
                f"got "
                f"{tuple(position_ids.shape)}."
            )

        batch_size = (
            context_features.size(0)
        )

        num_utterances = (
            context_features.size(1)
        )

        if (
            context_features.size(-1)
            != self.hidden_size
        ):
            raise ValueError(
                "Unexpected hidden size: "
                f"expected {self.hidden_size}, "
                f"got "
                f"{context_features.size(-1)}."
            )

        expected_shape = (
            batch_size,
            num_utterances,
        )

        if (
            tuple(speaker_ids.shape)
            != expected_shape
        ):
            raise ValueError(
                "speaker_ids shape mismatch: "
                f"expected {expected_shape}, "
                f"got "
                f"{tuple(speaker_ids.shape)}."
            )

        if (
            tuple(position_ids.shape)
            != expected_shape
        ):
            raise ValueError(
                "position_ids shape mismatch: "
                f"expected {expected_shape}, "
                f"got "
                f"{tuple(position_ids.shape)}."
            )

        if pair_mask is not None:

            expected_pair_shape = (
                batch_size,
                num_utterances,
                num_utterances,
            )

            if (
                tuple(pair_mask.shape)
                != expected_pair_shape
            ):
                raise ValueError(
                    "pair_mask shape mismatch: "
                    f"expected "
                    f"{expected_pair_shape}, "
                    f"got "
                    f"{tuple(pair_mask.shape)}."
                )


def build_pair_classifier(
    hidden_size: int = 256,
    pair_hidden_size: int = 256,
    distance_embedding_dim: int = 32,
    speaker_relation_embedding_dim: int = 16,
    max_relative_distance: int = 15,
    dropout: float = 0.1,
) -> PairClassifier:

    return PairClassifier(
        hidden_size=hidden_size,
        pair_hidden_size=(
            pair_hidden_size
        ),
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