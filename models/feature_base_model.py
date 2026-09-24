import torch
import torch.nn as nn


# =========================================================
# Event Memory Retriever (cross-speaker causal retrieval)
#
# Motivation
# ----------
# In conversations, an emotion utterance usually REFERS BACK to an
# event established earlier in the thread, frequently in another
# speaker's turn (on ECF, 64% of long-range causes are cross-speaker).
# We give every utterance a per-speaker "thread" state (GRU over that
# speaker's own turns) and let each utterance QUERY a differentiable
# memory of all utterances. The retrieved evidence vector is then used
# to score pairs, so the pair decision is EXPLAINED by the retrieved
# event (supervised directly by pair labels).
# =========================================================

class EventMemoryRetriever(nn.Module):

    def __init__(
        self,
        hidden_dim=256,
        key_dim=64,
        dropout=0.1,
        use_speaker_thread=True,
        use_historical_head=True,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.key_dim = int(key_dim)
        self.use_speaker_thread = bool(use_speaker_thread)
        self.use_historical_head = bool(use_historical_head)

        # Local head: attends anywhere (self-cause / adjacent causes).
        self.query_projection = nn.Linear(
            self.hidden_dim,
            self.key_dim,
        )

        self.key_projection = nn.Linear(
            self.hidden_dim,
            self.key_dim,
        )

        self.value_projection = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
        )

        # Historical head: forced to attend ONLY utterances that are
        # >= 2 turns back AND from a DIFFERENT speaker. This structural
        # constraint prevents the retrieval from collapsing onto the
        # locality shortcut and forces cross-speaker event reference.
        if self.use_historical_head:
            self.hist_query_projection = nn.Linear(
                self.hidden_dim,
                self.key_dim,
            )

            self.hist_key_projection = nn.Linear(
                self.hidden_dim,
                self.key_dim,
            )

            self.hist_value_projection = nn.Linear(
                self.hidden_dim,
                self.hidden_dim,
            )

        # Per-speaker thread memory over that speaker's own turns.
        if self.use_speaker_thread:
            self.thread_gru = nn.GRU(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                batch_first=True,
            )

            self.thread_gate = nn.Sequential(
                nn.Linear(
                    2 * self.hidden_dim,
                    self.hidden_dim,
                ),
                nn.GELU(),
            )

    # =====================================================
    # Forward
    # =====================================================

    def forward(
        self,
        context_features,
        speaker_ids,
        utterance_mask,
    ):
        """
        context_features: [B, N, H]
        speaker_ids:      [B, N]
        utterance_mask:   [B, N]  True = valid

        Returns
        -------
        attn / retrieved / explain:       local head outputs
        hist_attn / hist_retrieved / hist_explain: historical head
        """
        batch, num_utterances, hidden = context_features.shape

        valid = utterance_mask.bool()

        # ---------------------------------------------
        # Speaker thread states (per-speaker GRU)
        # ---------------------------------------------

        thread_states = torch.zeros_like(
            context_features
        )

        if self.use_speaker_thread:

            for b in range(batch):

                for speaker_id in torch.unique(
                    speaker_ids[b][valid[b]]
                ):

                    positions = (
                        valid[b]
                        & (
                            speaker_ids[b]
                            == speaker_id
                        )
                    )

                    turns = (
                        context_features[b][positions]
                        .unsqueeze(0)
                    )  # [1, T, H]

                    if turns.shape[1] == 0:
                        continue

                    states, _ = self.thread_gru(
                        turns
                    )  # [1, T, H]

                    thread_states[b][positions] = states[0]

            gate_input = torch.cat(
                [
                    context_features,
                    thread_states,
                ],
                dim=-1,
            )

            memory_features = self.thread_gate(
                gate_input
            )

        else:

            memory_features = context_features

        # ---------------------------------------------
        # Retrieval attention over the dialogue memory
        # ---------------------------------------------

        queries = self.query_projection(
            memory_features
        )  # [B, N, D]

        keys = self.key_projection(
            memory_features
        )  # [B, N, D]

        values = self.value_projection(
            memory_features
        )  # [B, N, H]

        attention_mask = (
            valid.unsqueeze(2)
            & valid.unsqueeze(1)
        )

        scores = torch.bmm(
            queries,
            keys.transpose(1, 2),
        ) / (self.key_dim ** 0.5)

        scores = scores.masked_fill(
            ~attention_mask,
            float("-inf"),
        )

        attn = torch.softmax(
            scores,
            dim=-1,
        )  # [B, N, N]

        attn = attn.masked_fill(
            ~attention_mask,
            0.0,
        )

        retrieved = torch.bmm(
            attn,
            values,
        )  # [B, N, H]

        explain = torch.bmm(
            retrieved,
            context_features.transpose(1, 2),
        ) / (self.hidden_dim ** 0.5)

        explain = explain.masked_fill(
            ~attention_mask,
            0.0,
        )

        output = {
            "attn": attn,
            "retrieved": retrieved,
            "explain": explain,
        }

        # ---------------------------------------------
        # Historical head: distance >= 2 AND cross-speaker
        # ---------------------------------------------

        if self.use_historical_head:

            positions = torch.arange(
                num_utterances,
                device=context_features.device,
            )

            dist = (
                positions.unsqueeze(1)
                - positions.unsqueeze(0)
            )  # i - j

            cross_speaker = (
                speaker_ids.unsqueeze(2)
                != speaker_ids.unsqueeze(1)
            )

            hist_mask = (
                attention_mask
                & (dist >= 2)
                & cross_speaker
            )

            hist_queries = self.hist_query_projection(
                memory_features
            )

            hist_keys = self.hist_key_projection(
                memory_features
            )

            hist_values = self.hist_value_projection(
                memory_features
            )

            hist_scores = torch.bmm(
                hist_queries,
                hist_keys.transpose(1, 2),
            ) / (self.key_dim ** 0.5)

            hist_scores = hist_scores.masked_fill(
                ~hist_mask,
                float("-inf"),
            )

            hist_attn = torch.softmax(
                hist_scores,
                dim=-1,
            )

            # Rows without any admissible historical key
            # yield all -inf -> NaN; overwrite with 0.
            hist_attn = hist_attn.masked_fill(
                ~hist_mask,
                0.0,
            )

            hist_retrieved = torch.bmm(
                hist_attn,
                hist_values,
            )  # [B, N, H]

            hist_explain = torch.bmm(
                hist_retrieved,
                context_features.transpose(1, 2),
            ) / (self.hidden_dim ** 0.5)

            hist_explain = hist_explain.masked_fill(
                ~hist_mask,
                0.0,
            )

            output.update(
                {
                    "hist_attn": hist_attn,
                    "hist_retrieved": hist_retrieved,
                    "hist_explain": hist_explain,
                }
            )

        return output


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
        node_gating=False,
        use_explain=False,
    ):
        super().__init__()

        self.hidden_dim = int(
            hidden_dim
        )

        self.node_gating = bool(
            node_gating
        )

        self.use_explain = bool(
            use_explain
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
        # + optional node gating
        #   (emotion-head prob of i, cause-head prob of j)
        # -------------------------------------------------

        pair_input_dim = (
            4
            * self.hidden_dim
            + self.distance_embedding_dim
            + self.speaker_relation_embedding_dim
            + (2 if self.node_gating else 0)
            + (2 if self.use_explain else 0)
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
        emotion_probs=None,
        cause_probs=None,
        explain_scores=None,
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

        pair_parts = [
            emotion_features,
            cause_features,
            product_features,
            difference_features,
            distance_features,
            speaker_relation_features,
        ]

        if self.node_gating:
            if emotion_probs is None or cause_probs is None:
                raise ValueError(
                    "node_gating=True requires "
                    "emotion_probs and cause_probs."
                )

            # [B, N] -> [B, N, N]
            emotion_probs_features = (
                emotion_probs
                .unsqueeze(2)
                .expand(
                    -1,
                    -1,
                    num_utterances,
                )
                .unsqueeze(-1)
            )

            cause_probs_features = (
                cause_probs
                .unsqueeze(1)
                .expand(
                    -1,
                    num_utterances,
                    -1,
                )
                .unsqueeze(-1)
            )

            pair_parts.append(
                emotion_probs_features
            )

            pair_parts.append(
                cause_probs_features
            )

        if self.use_explain:
            if explain_scores is None:
                raise ValueError(
                    "use_explain=True requires "
                    "explain_scores [B, N, N, 2]."
                )

            pair_parts.append(
                explain_scores
            )

        pair_features = torch.cat(
            pair_parts,
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
# Locality Prior Head (causal intervention on the shortcut)
#
# The "locality shortcut": 50% of ECF pairs are self-cause, so plain
# BCE training lets the model collapse onto distance/speaker cues and
# never learn long-range content evidence. We give the shortcut an
# EXPLICIT structural-only head (distance + speaker relation, zero
# content input) and train the content classifier with conditional
# ranking (see train_feature.conditional_ranking_loss).
# =========================================================

class LocalityPriorHead(nn.Module):

    def __init__(
        self,
        max_relative_distance=15,
        distance_dim=16,
        speaker_dim=8,
        hidden=64,
    ):
        super().__init__()

        self.max_relative_distance = int(
            max_relative_distance
        )

        self.distance_embedding = nn.Embedding(
            2 * self.max_relative_distance + 1,
            distance_dim,
        )

        self.speaker_relation_embedding = nn.Embedding(
            2,
            speaker_dim,
        )

        self.mlp = nn.Sequential(
            nn.Linear(
                distance_dim + speaker_dim,
                hidden,
            ),
            nn.GELU(),
            nn.Linear(
                hidden,
                1,
            ),
        )

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        speaker_ids,
        num_utterances,
    ):
        batch = speaker_ids.shape[0]
        device = speaker_ids.device

        positions = torch.arange(
            num_utterances,
            device=device,
        )

        relative_distance = (
            positions.view(num_utterances, 1)
            - positions.view(1, num_utterances)
        )

        relative_distance = relative_distance.clamp(
            min=-self.max_relative_distance,
            max=self.max_relative_distance,
        )

        distance_indices = (
            relative_distance
            + self.max_relative_distance
        )

        distance_features = (
            self.distance_embedding(
                distance_indices
            )
            .unsqueeze(0)
            .expand(batch, -1, -1, -1)
        )

        same_speaker = (
            speaker_ids.unsqueeze(2)
            == speaker_ids.unsqueeze(1)
        ).long()

        speaker_features = (
            self.speaker_relation_embedding(
                same_speaker
            )
        )

        features = torch.cat(
            [
                distance_features,
                speaker_features,
            ],
            dim=-1,
        )

        bias = (
            self.mlp(features)
            .squeeze(-1)
        )  # [B, N, N]

        return bias


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
        pair_decision_mode="fixed",
        pair_node_gating=False,
        use_event_retrieval=False,
        retrieval_key_dim=64,
        use_speaker_thread=True,
        use_historical_retrieval=True,
        use_locality_prior=False,
        null_hidden=128,
        null_residual_hidden=128,
        null_residual_scale=0.5,
        null_center_init=0.0,
        hierarchical_center_init=0.0,
        hierarchical_dialogue_hidden=128,
        hierarchical_dialogue_scale=0.5,
        hierarchical_row_hidden=128,
        hierarchical_row_scale=0.5,
    ):
        super().__init__()

        if pair_decision_mode not in ("fixed", "adaptive_reference", "null_reference", "centered_null_reference", "hierarchical_boundary"):
            raise ValueError(f"Unknown pair_decision_mode: {pair_decision_mode}")
        self.pair_decision_mode = pair_decision_mode

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

                node_gating=(
                    pair_node_gating
                ),

                use_explain=(
                    use_event_retrieval
                ),
            )
        )

        # -------------------------------------------------
        # A: cross-speaker event memory retrieval
        # -------------------------------------------------

        self.use_event_retrieval = bool(
            use_event_retrieval
        )

        if self.use_event_retrieval:
            self.event_retriever = EventMemoryRetriever(
                hidden_dim=dialogue_hidden,
                key_dim=int(retrieval_key_dim),
                dropout=dropout,
                use_speaker_thread=use_speaker_thread,
                use_historical_head=use_historical_retrieval,
            )

        # -------------------------------------------------
        # C: locality prior (structural shortcut head)
        # -------------------------------------------------

        self.use_locality_prior = bool(
            use_locality_prior
        )

        if self.use_locality_prior:
            self.locality_prior_head = LocalityPriorHead(
                max_relative_distance=max_relative_distance,
            )

        # Created only in adaptive mode: fixed state_dict and RNG are unchanged.
        if self.pair_decision_mode == "adaptive_reference":
            self.reference_head = nn.Sequential(
                nn.Linear(2 * self.dialogue_hidden, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )
            nn.init.zeros_(self.reference_head[-1].weight)
            nn.init.zeros_(self.reference_head[-1].bias)

        if self.pair_decision_mode == "null_reference":
            # Do not consume the baseline CPU initialization/dropout RNG stream.
            with torch.random.fork_rng(devices=[]):
                self.null_reference_head = nn.Sequential(
                    nn.Linear(2 * self.dialogue_hidden, int(null_hidden)),
                    nn.GELU(),
                    nn.Linear(int(null_hidden), 1),
                )

        if self.pair_decision_mode == "centered_null_reference":
            if null_residual_hidden < 1 or null_residual_scale <= 0:
                raise ValueError("Residual hidden size and scale must be positive")
            self.null_residual_scale = float(null_residual_scale)
            self.null_global_center = nn.Parameter(torch.tensor(float(null_center_init)))
            with torch.random.fork_rng(devices=[]):
                self.null_residual_head = nn.Sequential(
                    nn.Linear(2 * self.dialogue_hidden, int(null_residual_hidden)),
                    nn.GELU(),
                    nn.Linear(int(null_residual_hidden), 1),
                )
                nn.init.zeros_(self.null_residual_head[-1].weight)
                nn.init.zeros_(self.null_residual_head[-1].bias)

        if self.pair_decision_mode == "hierarchical_boundary":
            if min(hierarchical_dialogue_hidden, hierarchical_row_hidden) < 1:
                raise ValueError("Hierarchical hidden sizes must be positive")
            if min(hierarchical_dialogue_scale, hierarchical_row_scale) <= 0:
                raise ValueError("Hierarchical scales must be positive")
            self.hierarchical_dialogue_scale = float(hierarchical_dialogue_scale)
            self.hierarchical_row_scale = float(hierarchical_row_scale)
            self.global_center = nn.Parameter(torch.tensor(float(hierarchical_center_init)))
            with torch.random.fork_rng(devices=[]):
                self.dialogue_offset_head = nn.Sequential(
                    nn.Linear(self.dialogue_hidden, int(hierarchical_dialogue_hidden)),
                    nn.GELU(), nn.Linear(int(hierarchical_dialogue_hidden), 1),
                )
                self.row_residual_head = nn.Sequential(
                    nn.Linear(2 * self.dialogue_hidden, int(hierarchical_row_hidden)),
                    nn.GELU(), nn.Linear(int(hierarchical_row_hidden), 1),
                )
                for head in (self.dialogue_offset_head, self.row_residual_head):
                    nn.init.zeros_(head[-1].weight)
                    nn.init.zeros_(head[-1].bias)

    @property
    def hierarchical_boundary_parameters(self):
        if self.pair_decision_mode != "hierarchical_boundary":
            return 0
        return 1 + sum(p.numel() for head in (self.dialogue_offset_head, self.row_residual_head)
                       for p in head.parameters())

    @property
    def centered_null_parameters(self):
        if self.pair_decision_mode != "centered_null_reference":
            return 0
        return 1 + sum(p.numel() for p in self.null_residual_head.parameters())

    @property
    def adaptive_reference_parameters(self):
        if self.pair_decision_mode != "adaptive_reference":
            return 0
        return sum(p.numel() for p in self.reference_head.parameters())

    @property
    def null_reference_parameters(self):
        if self.pair_decision_mode != "null_reference":
            return 0
        return sum(p.numel() for p in self.null_reference_head.parameters())

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
        # A: event memory retrieval -> explain scores
        # -------------------------------------------------

        explain_scores = None
        retrieval_attn = None
        hist_retrieval_attn = None

        if self.use_event_retrieval:

            retrieval = self.event_retriever(
                context_features=context_features,
                speaker_ids=speaker_ids,
                utterance_mask=utterance_mask,
            )

            retrieval_attn = retrieval["attn"]

            explain_parts = [
                retrieval["explain"]
            ]

            if "hist_explain" in retrieval:
                explain_parts.append(
                    retrieval["hist_explain"]
                )
                hist_retrieval_attn = retrieval["hist_attn"]

            explain_scores = torch.stack(
                explain_parts,
                dim=-1,
            )  # [B, N, N, C]

        # -------------------------------------------------
        # Pair
        # -------------------------------------------------

        if self.pair_classifier.node_gating:

            emotion_probs = (
                torch.sigmoid(
                    emotion_logits
                )
                .detach()
            )

            cause_probs = (
                torch.sigmoid(
                    cause_logits
                )
                .detach()
            )

        else:

            emotion_probs = None
            cause_probs = None

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

                emotion_probs=(
                    emotion_probs
                ),

                cause_probs=(
                    cause_probs
                ),

                explain_scores=(
                    explain_scores
                ),
            )
        )

        # -------------------------------------------------
        # C: locality prior decomposition
        #
        # pair_logits = content_logits + locality_bias
        # -------------------------------------------------

        content_logits = pair_logits
        locality_bias = None

        if self.use_locality_prior:

            effective_pair_mask = (
                pair_mask
                if pair_mask is not None
                else (
                    utterance_mask.unsqueeze(2)
                    & utterance_mask.unsqueeze(1)
                )
            )

            locality_bias = (
                self.locality_prior_head(
                    speaker_ids=speaker_ids,
                    num_utterances=(
                        context_features.shape[1]
                    ),
                )
                .masked_fill(
                    ~effective_pair_mask.bool(),
                    0.0,
                )
            )

            pair_logits = (
                content_logits
                + locality_bias
            )

        output = {
            "emotion_logits":
                emotion_logits,

            "cause_logits":
                cause_logits,

            "pair_logits":
                pair_logits,
        }

        if self.use_event_retrieval:
            output["content_pair_logits"] = content_logits
            output["retrieval_attn"] = retrieval_attn

            if hist_retrieval_attn is not None:
                output["hist_retrieval_attn"] = hist_retrieval_attn

        if self.use_locality_prior:
            output["content_pair_logits"] = content_logits
            output["locality_bias"] = locality_bias

        if self.pair_decision_mode == "adaptive_reference":
            valid = utterance_mask.bool()
            weights = valid.unsqueeze(-1).to(context_features.dtype)
            dialogue_features = (context_features * weights).sum(dim=1)
            dialogue_features = dialogue_features / weights.sum(dim=1).clamp_min(1)
            reference_input = torch.cat([
                context_features,
                dialogue_features.unsqueeze(1).expand_as(context_features),
            ], dim=-1)
            reference = self.reference_head(reference_input).squeeze(-1)
            reference = reference.masked_fill(~valid, 0.0)
            valid_pairs = valid.unsqueeze(2) & valid.unsqueeze(1)
            if pair_mask is not None:
                valid_pairs = valid_pairs & pair_mask.bool()
            adjusted = (pair_logits - reference.unsqueeze(-1)).masked_fill(
                ~valid_pairs, 0.0,
            )
            output["adaptive_reference"] = reference
            output["adjusted_pair_logits"] = adjusted

        if self.pair_decision_mode in ("null_reference", "centered_null_reference", "hierarchical_boundary"):
            valid = utterance_mask.bool()
            detached_context = context_features.detach()
            weights = valid.unsqueeze(-1).to(detached_context.dtype)
            dialogue_context = (detached_context * weights).sum(dim=1)
            dialogue_context = dialogue_context / weights.sum(dim=1).clamp_min(1)
            null_input = torch.cat([
                detached_context,
                dialogue_context.detach().unsqueeze(1).expand_as(detached_context),
            ], dim=-1)
            if self.pair_decision_mode == "hierarchical_boundary":
                dialogue_raw = self.dialogue_offset_head(dialogue_context.detach()).squeeze(-1)
                dialogue_offset = self.hierarchical_dialogue_scale * torch.tanh(dialogue_raw)
                dialogue_offset = dialogue_offset.masked_fill(~valid.any(-1), 0.0)
                row_raw = self.row_residual_head(null_input).squeeze(-1)
                row_bounded = self.hierarchical_row_scale * torch.tanh(row_raw)
                row_bounded = row_bounded.masked_fill(~valid, 0.0)
                row_mean = row_bounded.sum(-1, keepdim=True) / valid.sum(-1, keepdim=True).clamp_min(1)
                row_zero_mean = (row_bounded - row_mean).masked_fill(~valid, 0.0)
                null_logits = self.global_center + dialogue_offset.unsqueeze(-1) + row_zero_mean
                output["global_center"] = self.global_center
                output["dialogue_offset"] = dialogue_offset
                output["row_residual_raw"] = row_raw.masked_fill(~valid, 0.0)
                output["row_residual_bounded"] = row_bounded
                output["row_residual_zero_mean"] = row_zero_mean
                output["dynamic_boundary"] = null_logits.masked_fill(~valid, 0.0)
            elif self.pair_decision_mode == "centered_null_reference":
                residual_raw = self.null_residual_head(null_input).squeeze(-1)
                residual = self.null_residual_scale * torch.tanh(residual_raw)
                residual = residual.masked_fill(~valid, 0.0)
                null_logits = self.null_global_center + residual
                output["null_global_center"] = self.null_global_center
                output["null_residual"] = residual
            else:
                null_logits = self.null_reference_head(null_input).squeeze(-1)
            null_logits = null_logits.masked_fill(~valid, 0.0)
            valid_pairs = valid.unsqueeze(2) & valid.unsqueeze(1)
            if pair_mask is not None:
                valid_pairs = valid_pairs & pair_mask.bool()
            output["null_logits"] = null_logits
            output["relative_pair_logits"] = (
                pair_logits - null_logits.unsqueeze(-1)
            ).masked_fill(~valid_pairs, 0.0)

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
