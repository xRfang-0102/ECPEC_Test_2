import torch
import torch.nn as nn


class DialogueEncoder(nn.Module):
    """
    Dialogue-level context encoder.

    Inputs
    ------
    utterance_features:
        [B, N, H_in]

    speaker_ids:
        [B, N]

    position_ids:
        [B, N]

    utterance_mask:
        [B, N]
        True  = real utterance
        False = padding

    Output
    ------
    context_features:
        [B, N, H]

    Default configuration
    ---------------------
    input_size              = 768
    hidden_size             = 256
    speaker_embedding_dim   = 32
    position_embedding_dim  = 32
    num_layers              = 2
    num_heads               = 4
    ff_dim                  = 1024
    dropout                 = 0.1
    """

    def __init__(
        self,
        input_size: int = 768,
        hidden_size: int = 256,
        speaker_embedding_dim: int = 32,
        position_embedding_dim: int = 32,
        max_speakers: int = 20,
        max_positions: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible "
                f"by num_heads ({num_heads})."
            )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.speaker_embedding_dim = speaker_embedding_dim
        self.position_embedding_dim = position_embedding_dim
        self.max_speakers = max_speakers
        self.max_positions = max_positions

        # --------------------------------------------------
        # 1. Project RoBERTa utterance features
        #
        # [B, N, 768] -> [B, N, 256]
        # --------------------------------------------------

        self.utterance_projection = nn.Linear(
            input_size,
            hidden_size,
        )

        # --------------------------------------------------
        # 2. Dialogue structural embeddings
        # --------------------------------------------------

        self.speaker_embedding = nn.Embedding(
            num_embeddings=max_speakers,
            embedding_dim=speaker_embedding_dim,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=max_positions,
            embedding_dim=position_embedding_dim,
        )

        # --------------------------------------------------
        # 3. Fuse semantic + speaker + position information
        #
        # 256 + 32 + 32 = 320
        # 320 -> 256
        # --------------------------------------------------

        fused_size = (
            hidden_size
            + speaker_embedding_dim
            + position_embedding_dim
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                fused_size,
                hidden_size,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --------------------------------------------------
        # 4. Dialogue Transformer
        # --------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.output_norm = nn.LayerNorm(
            hidden_size
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        utterance_features: torch.Tensor,
        speaker_ids: torch.Tensor,
        position_ids: torch.Tensor,
        utterance_mask: torch.Tensor,
    ) -> torch.Tensor:

        self._validate_inputs(
            utterance_features=utterance_features,
            speaker_ids=speaker_ids,
            position_ids=position_ids,
            utterance_mask=utterance_mask,
        )

        # --------------------------------------------------
        # 1. RoBERTa feature projection
        #
        # [B, N, H_in]
        #     ->
        # [B, N, H]
        # --------------------------------------------------

        semantic_features = (
            self.utterance_projection(
                utterance_features
            )
        )

        semantic_features = (
            self.dropout(
                semantic_features
            )
        )

        # --------------------------------------------------
        # 2. Speaker embeddings
        #
        # [B, N]
        #     ->
        # [B, N, speaker_dim]
        # --------------------------------------------------

        safe_speaker_ids = (
            speaker_ids.clamp(
                min=0,
                max=self.max_speakers - 1,
            )
        )

        speaker_features = (
            self.speaker_embedding(
                safe_speaker_ids
            )
        )

        # --------------------------------------------------
        # 3. Position embeddings
        #
        # [B, N]
        #     ->
        # [B, N, position_dim]
        # --------------------------------------------------

        safe_position_ids = (
            position_ids.clamp(
                min=0,
                max=self.max_positions - 1,
            )
        )

        position_features = (
            self.position_embedding(
                safe_position_ids
            )
        )

        # --------------------------------------------------
        # 4. Feature fusion
        #
        # [semantic ; speaker ; position]
        # --------------------------------------------------

        fused_features = torch.cat(
            [
                semantic_features,
                speaker_features,
                position_features,
            ],
            dim=-1,
        )

        fused_features = self.fusion(
            fused_features
        )

        # Explicitly zero padded utterances before Transformer
        fused_features = (
            fused_features
            * utterance_mask
            .unsqueeze(-1)
            .to(fused_features.dtype)
        )

        # --------------------------------------------------
        # 5. Transformer padding mask
        #
        # Our mask:
        # True  = valid utterance
        # False = padding
        #
        # PyTorch Transformer:
        # True  = padding position
        # False = valid position
        # --------------------------------------------------

        key_padding_mask = (
            ~utterance_mask.bool()
        )

        context_features = self.transformer(
            src=fused_features,
            src_key_padding_mask=(
                key_padding_mask
            ),
        )

        context_features = (
            self.output_norm(
                context_features
            )
        )

        # --------------------------------------------------
        # 6. Zero padded outputs again
        # --------------------------------------------------

        context_features = (
            context_features
            * utterance_mask
            .unsqueeze(-1)
            .to(context_features.dtype)
        )

        return context_features

    def _validate_inputs(
        self,
        utterance_features: torch.Tensor,
        speaker_ids: torch.Tensor,
        position_ids: torch.Tensor,
        utterance_mask: torch.Tensor,
    ) -> None:

        if utterance_features.dim() != 3:
            raise ValueError(
                "utterance_features must have shape "
                "[B, N, H], "
                f"got {tuple(utterance_features.shape)}."
            )

        if speaker_ids.dim() != 2:
            raise ValueError(
                "speaker_ids must have shape "
                "[B, N], "
                f"got {tuple(speaker_ids.shape)}."
            )

        if position_ids.dim() != 2:
            raise ValueError(
                "position_ids must have shape "
                "[B, N], "
                f"got {tuple(position_ids.shape)}."
            )

        if utterance_mask.dim() != 2:
            raise ValueError(
                "utterance_mask must have shape "
                "[B, N], "
                f"got {tuple(utterance_mask.shape)}."
            )

        batch_size = (
            utterance_features.size(0)
        )

        num_utterances = (
            utterance_features.size(1)
        )

        expected_shape = (
            batch_size,
            num_utterances,
        )

        if tuple(
            speaker_ids.shape
        ) != expected_shape:
            raise ValueError(
                "speaker_ids shape mismatch: "
                f"expected {expected_shape}, "
                f"got {tuple(speaker_ids.shape)}."
            )

        if tuple(
            position_ids.shape
        ) != expected_shape:
            raise ValueError(
                "position_ids shape mismatch: "
                f"expected {expected_shape}, "
                f"got {tuple(position_ids.shape)}."
            )

        if tuple(
            utterance_mask.shape
        ) != expected_shape:
            raise ValueError(
                "utterance_mask shape mismatch: "
                f"expected {expected_shape}, "
                f"got {tuple(utterance_mask.shape)}."
            )

        if (
            utterance_features.size(-1)
            != self.input_size
        ):
            raise ValueError(
                "Unexpected utterance feature size: "
                f"expected {self.input_size}, "
                f"got "
                f"{utterance_features.size(-1)}."
            )

        if speaker_ids.numel() > 0:
            max_speaker_id = int(
                speaker_ids.max().item()
            )

            if max_speaker_id >= self.max_speakers:
                raise ValueError(
                    f"speaker_id={max_speaker_id} exceeds "
                    f"max_speakers={self.max_speakers}."
                )

        if position_ids.numel() > 0:
            max_position_id = int(
                position_ids.max().item()
            )

            if max_position_id >= self.max_positions:
                raise ValueError(
                    f"position_id={max_position_id} exceeds "
                    f"max_positions={self.max_positions}."
                )


def build_dialogue_encoder(
    input_size: int = 768,
    hidden_size: int = 256,
    speaker_embedding_dim: int = 32,
    position_embedding_dim: int = 32,
    max_speakers: int = 20,
    max_positions: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    ff_dim: int = 1024,
    dropout: float = 0.1,
) -> DialogueEncoder:

    return DialogueEncoder(
        input_size=input_size,
        hidden_size=hidden_size,
        speaker_embedding_dim=(
            speaker_embedding_dim
        ),
        position_embedding_dim=(
            position_embedding_dim
        ),
        max_speakers=max_speakers,
        max_positions=max_positions,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
    )