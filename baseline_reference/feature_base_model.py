import torch
import torch.nn as nn


# =========================================================
# Dialogue Encoder
# =========================================================

class FeatureDialogueEncoder(nn.Module):

    def __init__(
        self,
        input_dim=768,
        hidden_dim=256,
        speaker_embedding_dim=32,
        position_embedding_dim=32,
        max_speakers=20,
        max_positions=64,
        num_layers=2,
        num_heads=4,
        ffn_dim=1024,
        dropout=0.1,
    ):
        super().__init__()

        self.input_dim = int(
            input_dim
        )

        self.hidden_dim = int(
            hidden_dim
        )

        self.max_speakers = int(
            max_speakers
        )

        self.max_positions = int(
            max_positions
        )

        # -------------------------------------------------
        # Frozen RoBERTa feature projection
        # 768 -> 256
        # -------------------------------------------------

        self.feature_projection = nn.Linear(
            self.input_dim,
            self.hidden_dim,
        )

        # -------------------------------------------------
        # Speaker embedding
        # -------------------------------------------------

        self.speaker_embedding = nn.Embedding(
            self.max_speakers,
            speaker_embedding_dim,
        )

        # -------------------------------------------------
        # Position embedding
        # -------------------------------------------------

        self.position_embedding = nn.Embedding(
            self.max_positions,
            position_embedding_dim,
        )

        # -------------------------------------------------
        # Feature fusion
        #
        # 256
        # + speaker_dim
        # + position_dim
        # -> 256
        # -------------------------------------------------

        fusion_dim = (
            self.hidden_dim
            + speaker_embedding_dim
            + position_embedding_dim
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                fusion_dim,
                self.hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
        )

        # -------------------------------------------------
        # Dialogue Transformer
        # -------------------------------------------------

        transformer_layer = (
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer=transformer_layer,
                num_layers=num_layers,
            )
        )

        self.output_norm = nn.LayerNorm(
            self.hidden_dim
        )

    # =====================================================
    # Forward
    # =====================================================

    def forward(
        self,
        utterance_features,
        speaker_ids,
        position_ids,
        utterance_mask,
    ):
        """
        utterance_features:
            [B, N, 768]

        speaker_ids:
            [B, N]

        position_ids:
            [B, N]

        utterance_mask:
            [B, N]
            True = valid utterance
        """

        if utterance_features.ndim != 3:
            raise ValueError(
                "utterance_features must "
                "have shape [B, N, H]."
            )

        batch_size, num_utterances, feature_dim = (
            utterance_features.shape
        )

        if feature_dim != self.input_dim:
            raise ValueError(
                "Unexpected input feature dim: "
                f"{feature_dim}. "
                f"Expected {self.input_dim}."
            )

        if tuple(
            speaker_ids.shape
        ) != (
            batch_size,
            num_utterances,
        ):
            raise ValueError(
                "speaker_ids shape mismatch."
            )

        if tuple(
            position_ids.shape
        ) != (
            batch_size,
            num_utterances,
        ):
            raise ValueError(
                "position_ids shape mismatch."
            )

        if tuple(
            utterance_mask.shape
        ) != (
            batch_size,
            num_utterances,
        ):
            raise ValueError(
                "utterance_mask shape mismatch."
            )

        # -------------------------------------------------
        # Safety checks
        # -------------------------------------------------

        if (
            speaker_ids[
                utterance_mask
            ].numel()
            > 0
        ):
            max_speaker_id = int(
                speaker_ids[
                    utterance_mask
                ].max().item()
            )

            if (
                max_speaker_id
                >= self.max_speakers
            ):
                raise ValueError(
                    "Speaker ID exceeds "
                    "max_speakers: "
                    f"{max_speaker_id} >= "
                    f"{self.max_speakers}"
                )

        if (
            position_ids[
                utterance_mask
            ].numel()
            > 0
        ):
            max_position_id = int(
                position_ids[
                    utterance_mask
                ].max().item()
            )

            if (
                max_position_id
                >= self.max_positions
            ):
                raise ValueError(
                    "Position ID exceeds "
                    "max_positions: "
                    f"{max_position_id} >= "
                    f"{self.max_positions}"
                )

        # -------------------------------------------------
        # Projection
        # -------------------------------------------------

        projected_features = (
            self.feature_projection(
                utterance_features
            )
        )

        # -------------------------------------------------
        # Embeddings
        # -------------------------------------------------

        speaker_features = (
            self.speaker_embedding(
                speaker_ids
            )
        )

        position_features = (
            self.position_embedding(
                position_ids
            )
        )

        # -------------------------------------------------
        # Fusion
        # -------------------------------------------------

        fused_features = torch.cat(
            [
                projected_features,
                speaker_features,
                position_features,
            ],
            dim=-1,
        )

        fused_features = (
            self.fusion(
                fused_features
            )
        )

        # -------------------------------------------------
        # Zero padded input positions
        # -------------------------------------------------

        fused_features = (
            fused_features
            * utterance_mask
            .unsqueeze(-1)
            .to(
                fused_features.dtype
            )
        )

        # -------------------------------------------------
        # Transformer padding mask
        #
        # True means ignore in PyTorch Transformer
        # -------------------------------------------------

        padding_mask = (
            ~utterance_mask.bool()
        )

        context_features = (
            self.transformer(
                fused_features,
                src_key_padding_mask=(
                    padding_mask
                ),
            )
        )

        context_features = (
            self.output_norm(
                context_features
            )
        )

        # -------------------------------------------------
        # Ensure padded positions remain exactly zero
        # -------------------------------------------------

        context_features = (
            context_features
            * utterance_mask
            .unsqueeze(-1)
            .to(
                context_features.dtype
            )
        )

        return context_features


# =========================================================
# Auxiliary Binary Head
# =========================================================

class FeatureBinaryAuxiliaryHead(nn.Module):

    def __init__(
        self,
        input_dim=256,
        hidden_dim=128,
        dropout=0.1,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        features,
        utterance_mask,
    ):
        """
        features:
            [B, N, H]

        output:
            [B, N]
        """

        logits = (
            self.classifier(
                features
            )
            .squeeze(-1)
        )

        logits = logits.masked_fill(
            ~utterance_mask.bool(),
            0.0,
        )

        return logits


# =========================================================
# Full N x N Pair Classifier
# =========================================================

class FeaturePairClassifier(nn.Module):

    def __init__(
        self,
        hidden_dim=256,
        pair_hidden=256,
        distance_embedding_dim=32,
        speaker_relation_embedding_dim=16,
        max_relative_distance=15,
        dropout=0.1,
    ):
        super().__init__()

        self.hidden_dim = int(
            hidden_dim
        )

        self.max_relative_distance = int(
            max_relative_distance
        )

        self.distance_embedding_dim = int(
            distance_embedding_dim
        )

        self.speaker_relation_embedding_dim = int(
            speaker_relation_embedding_dim
        )

        # -------------------------------------------------
        # Relative distance
        #
        # range:
        # [-max_distance, +max_distance]
        #
        # number of embeddings:
        # 2 * max_distance + 1
        # -------------------------------------------------

        self.distance_embedding = nn.Embedding(
            (
                2
                * self.max_relative_distance
                + 1
            ),
            self.distance_embedding_dim,
        )

        # -------------------------------------------------
        # Same / different speaker
        #
        # 0 = different speaker
        # 1 = same speaker
        # -------------------------------------------------

        self.speaker_relation_embedding = (
            nn.Embedding(
                2,
                self.speaker_relation_embedding_dim,
            )
        )

        # -------------------------------------------------
        # Pair representation
        #
        # h_e
        # h_c
        # h_e * h_c
        # |h_e - h_c|
        #
        # => 4 * hidden_dim
        #
        # + distance embedding
        # + speaker relation embedding
        # -------------------------------------------------

        pair_input_dim = (
            4
            * self.hidden_dim
            + self.distance_embedding_dim
            + self.speaker_relation_embedding_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                pair_input_dim,
                pair_hidden,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                pair_hidden,
                1,
            ),
        )

    # =====================================================
    # Forward
    # =====================================================

    def forward(
        self,
        context_features,
        speaker_ids,
        utterance_mask,
        pair_mask=None,
    ):
        """
        Pair convention:

        i = emotion candidate
        j = cause candidate

        context_features:
            [B, N, H]

        speaker_ids:
            [B, N]

        utterance_mask:
            [B, N]

        pair_mask:
            [B, N, N]

        output:
            [B, N, N]
        """

        batch_size, num_utterances, hidden_dim = (
            context_features.shape
        )

        if hidden_dim != self.hidden_dim:
            raise ValueError(
                "Unexpected context hidden dim: "
                f"{hidden_dim}. "
                f"Expected {self.hidden_dim}."
            )

        # -------------------------------------------------
        # Emotion-side representation
        #
        # [B,N,1,H] -> [B,N,N,H]
        # -------------------------------------------------

        emotion_features = (
            context_features
            .unsqueeze(2)
            .expand(
                -1,
                -1,
                num_utterances,
                -1,
            )
        )

        # -------------------------------------------------
        # Cause-side representation
        #
        # [B,1,N,H] -> [B,N,N,H]
        # -------------------------------------------------

        cause_features = (
            context_features
            .unsqueeze(1)
            .expand(
                -1,
                num_utterances,
                -1,
                -1,
            )
        )

        # -------------------------------------------------
        # Interaction
        # -------------------------------------------------

        product_features = (
            emotion_features
            * cause_features
        )

        difference_features = torch.abs(
            emotion_features
            - cause_features
        )

        # -------------------------------------------------
        # Relative position
        #
        # distance = emotion_position - cause_position
        #
        # i - j
        # -------------------------------------------------

        positions = torch.arange(
            num_utterances,
            device=context_features.device,
            dtype=torch.long,
        )

        emotion_positions = (
            positions.view(
                num_utterances,
                1,
            )
        )

        cause_positions = (
            positions.view(
                1,
                num_utterances,
            )
        )

        relative_distance = (
            emotion_positions
            - cause_positions
        )

        relative_distance = (
            relative_distance.clamp(
                min=(
                    -self.max_relative_distance
                ),
                max=(
                    self.max_relative_distance
                ),
            )
        )

        distance_indices = (
            relative_distance
            + self.max_relative_distance
        )

        distance_features = (
            self.distance_embedding(
                distance_indices
            )
        )

        # [N,N,D]
        # ->
        # [B,N,N,D]

        distance_features = (
            distance_features
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
                -1,
                -1,
            )
        )

        # -------------------------------------------------
        # Speaker relation
        # -------------------------------------------------

        emotion_speaker = (
            speaker_ids
            .unsqueeze(2)
        )

        cause_speaker = (
            speaker_ids
            .unsqueeze(1)
        )

        same_speaker = (
            emotion_speaker
            == cause_speaker
        ).long()

        speaker_relation_features = (
            self.speaker_relation_embedding(
                same_speaker
            )
        )

        # -------------------------------------------------
        # Pair representation
        # -------------------------------------------------

        pair_features = torch.cat(
            [
                emotion_features,
                cause_features,
                product_features,
                difference_features,
                distance_features,
                speaker_relation_features,
            ],
            dim=-1,
        )

        # -------------------------------------------------
        # Pair logits
        # -------------------------------------------------

        pair_logits = (
            self.classifier(
                pair_features
            )
            .squeeze(-1)
        )

        # -------------------------------------------------
        # Pair mask
        # -------------------------------------------------

        if pair_mask is None:

            pair_mask = (
                utterance_mask
                .unsqueeze(2)
                &
                utterance_mask
                .unsqueeze(1)
            )

        pair_logits = (
            pair_logits.masked_fill(
                ~pair_mask.bool(),
                0.0,
            )
        )

        return pair_logits


# =========================================================
# Complete Frozen-Feature ECPEC Base Model
# =========================================================

class FeatureECPECBaseModel(nn.Module):

    def __init__(
        self,
        input_dim=768,

        dialogue_hidden=256,

        speaker_embedding_dim=32,
        position_embedding_dim=32,

        max_speakers=20,
        max_positions=64,

        dialogue_layers=2,
        dialogue_heads=4,
        dialogue_ffn=1024,

        auxiliary_hidden=128,

        pair_hidden=256,

        distance_embedding_dim=32,
        speaker_relation_embedding_dim=16,
        max_relative_distance=15,

        dropout=0.1,
    ):
        super().__init__()

        self.input_dim = int(
            input_dim
        )

        self.dialogue_hidden = int(
            dialogue_hidden
        )

        # -------------------------------------------------
        # Dialogue Encoder
        # -------------------------------------------------

        self.dialogue_encoder = (
            FeatureDialogueEncoder(
                input_dim=(
                    input_dim
                ),

                hidden_dim=(
                    dialogue_hidden
                ),

                speaker_embedding_dim=(
                    speaker_embedding_dim
                ),

                position_embedding_dim=(
                    position_embedding_dim
                ),

                max_speakers=(
                    max_speakers
                ),

                max_positions=(
                    max_positions
                ),

                num_layers=(
                    dialogue_layers
                ),

                num_heads=(
                    dialogue_heads
                ),

                ffn_dim=(
                    dialogue_ffn
                ),

                dropout=(
                    dropout
                ),
            )
        )

        # -------------------------------------------------
        # Emotion Head
        # -------------------------------------------------

        self.emotion_head = (
            FeatureBinaryAuxiliaryHead(
                input_dim=(
                    dialogue_hidden
                ),

                hidden_dim=(
                    auxiliary_hidden
                ),

                dropout=(
                    dropout
                ),
            )
        )

        # -------------------------------------------------
        # Cause Head
        # -------------------------------------------------

        self.cause_head = (
            FeatureBinaryAuxiliaryHead(
                input_dim=(
                    dialogue_hidden
                ),

                hidden_dim=(
                    auxiliary_hidden
                ),

                dropout=(
                    dropout
                ),
            )
        )

        # -------------------------------------------------
        # Pair Classifier
        # -------------------------------------------------

        self.pair_classifier = (
            FeaturePairClassifier(
                hidden_dim=(
                    dialogue_hidden
                ),

                pair_hidden=(
                    pair_hidden
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

                dropout=(
                    dropout
                ),
            )
        )

    # =====================================================
    # Forward
    # =====================================================

    def forward(
        self,
        utterance_features,
        speaker_ids,
        position_ids,
        utterance_mask,
        pair_mask=None,
        return_features=False,
    ):
        """
        utterance_features:
            [B, N, 768]

        speaker_ids:
            [B, N]

        position_ids:
            [B, N]

        utterance_mask:
            [B, N]

        pair_mask:
            [B, N, N]
        """

        # -------------------------------------------------
        # Input validation
        # -------------------------------------------------

        if utterance_features.ndim != 3:
            raise ValueError(
                "utterance_features must "
                "have shape [B, N, H]."
            )

        if (
            utterance_features.shape[-1]
            != self.input_dim
        ):
            raise ValueError(
                "Unexpected utterance "
                "feature dimension: "
                f"{utterance_features.shape[-1]} "
                f"!= {self.input_dim}"
            )

        if not torch.isfinite(
            utterance_features
        ).all():
            raise ValueError(
                "Non-finite values detected "
                "in utterance_features."
            )

        # -------------------------------------------------
        # Dialogue Encoding
        # -------------------------------------------------

        context_features = (
            self.dialogue_encoder(
                utterance_features=(
                    utterance_features
                ),

                speaker_ids=(
                    speaker_ids
                ),

                position_ids=(
                    position_ids
                ),

                utterance_mask=(
                    utterance_mask
                ),
            )
        )

        # -------------------------------------------------
        # Emotion
        # -------------------------------------------------

        emotion_logits = (
            self.emotion_head(
                features=(
                    context_features
                ),

                utterance_mask=(
                    utterance_mask
                ),
            )
        )

        # -------------------------------------------------
        # Cause
        # -------------------------------------------------

        cause_logits = (
            self.cause_head(
                features=(
                    context_features
                ),

                utterance_mask=(
                    utterance_mask
                ),
            )
        )

        # -------------------------------------------------
        # Pair
        # -------------------------------------------------

        pair_logits = (
            self.pair_classifier(
                context_features=(
                    context_features
                ),

                speaker_ids=(
                    speaker_ids
                ),

                utterance_mask=(
                    utterance_mask
                ),

                pair_mask=(
                    pair_mask
                ),
            )
        )

        output = {
            "emotion_logits":
                emotion_logits,

            "cause_logits":
                cause_logits,

            "pair_logits":
                pair_logits,
        }

        if return_features:

            output[
                "context_features"
            ] = context_features

        return output


# =========================================================
# Parameter Counter
# =========================================================

def count_parameters(
    model,
):
    total = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    return (
        total,
        trainable,
    )


# =========================================================
# Standalone Forward Test
# =========================================================

if __name__ == "__main__":

    batch_size = 4
    num_utterances = 9
    input_dim = 768

    # -----------------------------------------------------
    # Fake fixed RoBERTa features
    # -------------------------------------------------

    utterance_features = torch.randn(
        batch_size,
        num_utterances,
        input_dim,
    )

    # Same dialogue lengths as our verified first batch
    dialogue_lengths = [
        8,
        3,
        9,
        3,
    ]

    utterance_mask = torch.zeros(
        batch_size,
        num_utterances,
        dtype=torch.bool,
    )

    for index, length in enumerate(
        dialogue_lengths
    ):
        utterance_mask[
            index,
            :length
        ] = True

    # Zero padded features
    utterance_features[
        ~utterance_mask
    ] = 0.0

    # -----------------------------------------------------
    # Speaker IDs
    # -------------------------------------------------

    speaker_ids = torch.zeros(
        batch_size,
        num_utterances,
        dtype=torch.long,
    )

    for b, length in enumerate(
        dialogue_lengths
    ):
        speaker_ids[
            b,
            :length
        ] = torch.arange(
            length
        ) % 2

    # -----------------------------------------------------
    # Position IDs
    # -------------------------------------------------

    position_ids = torch.zeros(
        batch_size,
        num_utterances,
        dtype=torch.long,
    )

    for b, length in enumerate(
        dialogue_lengths
    ):
        position_ids[
            b,
            :length
        ] = torch.arange(
            length
        )

    # -----------------------------------------------------
    # Pair Mask
    # -------------------------------------------------

    pair_mask = (
        utterance_mask.unsqueeze(
            2
        )
        &
        utterance_mask.unsqueeze(
            1
        )
    )

    # -----------------------------------------------------
    # Model
    # -------------------------------------------------

    model = FeatureECPECBaseModel(
        input_dim=768,

        dialogue_hidden=256,

        speaker_embedding_dim=32,
        position_embedding_dim=32,

        max_speakers=20,
        max_positions=64,

        dialogue_layers=2,
        dialogue_heads=4,
        dialogue_ffn=1024,

        auxiliary_hidden=128,

        pair_hidden=256,

        distance_embedding_dim=32,
        speaker_relation_embedding_dim=16,

        max_relative_distance=15,

        dropout=0.1,
    )

    model.eval()

    # -----------------------------------------------------
    # Forward
    # -------------------------------------------------

    with torch.no_grad():

        outputs = model(
            utterance_features=(
                utterance_features
            ),

            speaker_ids=(
                speaker_ids
            ),

            position_ids=(
                position_ids
            ),

            utterance_mask=(
                utterance_mask
            ),

            pair_mask=(
                pair_mask
            ),

            return_features=True,
        )

    # -----------------------------------------------------
    # Parameter statistics
    # -------------------------------------------------

    (
        total_parameters,
        trainable_parameters,
    ) = count_parameters(
        model
    )

    print()
    print(
        "=" * 60
    )

    print(
        "Frozen-Feature ECPEC Model Test"
    )

    print(
        "=" * 60
    )

    print(
        "Input features:",
        utterance_features.shape,
    )

    print(
        "Context features:",
        outputs[
            "context_features"
        ].shape,
    )

    print(
        "Emotion logits:",
        outputs[
            "emotion_logits"
        ].shape,
    )

    print(
        "Cause logits:",
        outputs[
            "cause_logits"
        ].shape,
    )

    print(
        "Pair logits:",
        outputs[
            "pair_logits"
        ].shape,
    )

    print()

    print(
        "Valid utterances:",
        int(
            utterance_mask
            .sum()
            .item()
        ),
    )

    print(
        "Valid pairs:",
        int(
            pair_mask
            .sum()
            .item()
        ),
    )

    print()

    print(
        "Total parameters:",
        f"{total_parameters:,}",
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}",
    )

    # -----------------------------------------------------
    # Padding validation
    # -------------------------------------------------

    emotion_padding_sum = float(
        outputs[
            "emotion_logits"
        ][
            ~utterance_mask
        ]
        .abs()
        .sum()
        .item()
    )

    cause_padding_sum = float(
        outputs[
            "cause_logits"
        ][
            ~utterance_mask
        ]
        .abs()
        .sum()
        .item()
    )

    pair_padding_sum = float(
        outputs[
            "pair_logits"
        ][
            ~pair_mask
        ]
        .abs()
        .sum()
        .item()
    )

    context_padding_sum = float(
        outputs[
            "context_features"
        ][
            ~utterance_mask
        ]
        .abs()
        .sum()
        .item()
    )

    print()

    print(
        "Context padding abs sum:",
        context_padding_sum,
    )

    print(
        "Emotion padding abs sum:",
        emotion_padding_sum,
    )

    print(
        "Cause padding abs sum:",
        cause_padding_sum,
    )

    print(
        "Pair padding abs sum:",
        pair_padding_sum,
    )

    # -----------------------------------------------------
    # Finite validation
    # -------------------------------------------------

    print()

    print(
        "Context finite:",
        bool(
            torch.isfinite(
                outputs[
                    "context_features"
                ]
            ).all()
        ),
    )

    print(
        "Emotion finite:",
        bool(
            torch.isfinite(
                outputs[
                    "emotion_logits"
                ]
            ).all()
        ),
    )

    print(
        "Cause finite:",
        bool(
            torch.isfinite(
                outputs[
                    "cause_logits"
                ]
            ).all()
        ),
    )

    print(
        "Pair finite:",
        bool(
            torch.isfinite(
                outputs[
                    "pair_logits"
                ]
            ).all()
        ),
    )

    print()
    print(
        "=" * 60
    )