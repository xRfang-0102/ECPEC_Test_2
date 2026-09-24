# =========================================================
# LoRAECPECModel: end-to-end ECPEC model for v3.
#
#   RoBERTa-base (LoRA fine-tuned, mean pooling)
#       -> dialogue Transformer + emotion/cause heads
#       -> pair classifier (+ A+C retrieval / locality prior)
#
# The head part is the existing FeatureECPECBaseModel, so the
# output dict is identical and all train_feature losses,
# metrics and analysis scripts work unchanged.
# =========================================================

import torch
from torch import nn
from transformers import AutoModel

from models.lora import LoRALinear, apply_lora_to_roberta
from sweep_threshold import build_model


class LoRAECPECModel(nn.Module):

    def __init__(
        self,
        encoder_path="pretrained/roberta-base",
        model_config=None,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        use_bf16=True,
    ):
        super().__init__()

        if model_config is None:
            model_config = {}

        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.use_bf16 = bool(use_bf16)

        # -------------------------------------------------
        # Encoder: frozen RoBERTa + LoRA query/value deltas
        # -------------------------------------------------

        self.encoder = AutoModel.from_pretrained(
            str(encoder_path)
        )

        self.num_lora_modules = 0

        if self.lora_r > 0:
            # -------------------------------------------------
            # LoRA path: wrap query/value, freeze the base,
            # re-enable only the low-rank deltas.
            # -------------------------------------------------

            wrapped = apply_lora_to_roberta(
                self.encoder,
                r=self.lora_r,
                alpha=self.lora_alpha,
                dropout=lora_dropout,
            )

            self.num_lora_modules = wrapped

            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

            for module in self.encoder.modules():
                if isinstance(module, LoRALinear):
                    module.lora_A.requires_grad_(True)
                    module.lora_B.requires_grad_(True)

        # else: full fine-tuning (lora_r <= 0), everything
        # stays trainable.

        # Recompute encoder activations instead of storing them:
        # elementwise ops (GELU/LayerNorm/dropout) would otherwise
        # save every intermediate even for frozen layers.
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        # -------------------------------------------------
        # Head: dialogue transformer + pair classifier + A+C
        # -------------------------------------------------

        self.head = build_model(
            model_config,
            768,
        )

        # Delegate attributes that train_feature helpers use.
        self.pair_decision_mode = self.head.pair_decision_mode

    def encode_dialogue(
        self,
        input_ids,
        attention_mask,
    ):
        """
        input_ids:      [B, N, L]
        attention_mask: [B, N, L]

        Returns mean-pooled utterance embeddings [B, N, H].
        """

        batch_size, num_utterances, max_length = input_ids.shape

        flat_ids = input_ids.reshape(
            batch_size * num_utterances,
            max_length,
        )

        flat_mask = attention_mask.reshape(
            batch_size * num_utterances,
            max_length,
        )

        encoder_output = None

        # bf16 only around the encoder: the dialogue head
        # (GRU retrieval, softmax attention, pair heads) runs
        # in fp32 to avoid dtype-mixing issues.
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(
                self.use_bf16
                and flat_ids.is_cuda
            ),
        ):
            encoder_output = self.encoder(
                input_ids=flat_ids,
                attention_mask=flat_mask,
            )

        hidden = encoder_output.last_hidden_state  # [B*N, L, H]

        mask = flat_mask.unsqueeze(-1).to(hidden.dtype)

        pooled = (
            (hidden * mask).sum(dim=1)
            / mask.sum(dim=1).clamp_min(1.0)
        )  # [B*N, H]

        # Feed the fp32 head with fp32 features.
        return pooled.reshape(
            batch_size,
            num_utterances,
            -1,
        ).float()

    def forward(
        self,
        input_ids,
        attention_mask,
        speaker_ids,
        position_ids,
        utterance_mask,
        pair_mask,
    ):
        utterance_features = self.encode_dialogue(
            input_ids,
            attention_mask,
        )

        return self.head(
            utterance_features=utterance_features,
            speaker_ids=speaker_ids,
            position_ids=position_ids,
            utterance_mask=utterance_mask,
            pair_mask=pair_mask,
        )
