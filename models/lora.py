# =========================================================
# Minimal hand-rolled LoRA for RoBERTa (no peft dependency).
#
# LoRALinear wraps a frozen nn.Linear and adds a trainable
# low-rank delta:  y = W x + b + scaling * (B A) x
#
# apply_lora_to_roberta walks every RobertaSelfAttention and
# wraps its query/value projections (standard RoBERTa LoRA
# configuration; the key projection is left untouched).
# =========================================================

import math

import torch
from torch import nn


class LoRALinear(nn.Module):

    def __init__(
        self,
        base_linear,
        r=16,
        alpha=32,
        dropout=0.1,
    ):
        super().__init__()

        if not isinstance(base_linear, nn.Linear):
            raise TypeError(
                "LoRALinear wraps nn.Linear, got "
                f"{type(base_linear).__name__}"
            )

        self.base = base_linear
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r

        out_features, in_features = (
            self.base.weight.shape
        )

        self.lora_A = nn.Parameter(
            torch.empty(
                self.r,
                in_features,
                device=self.base.weight.device,
                dtype=self.base.weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                out_features,
                self.r,
                device=self.base.weight.device,
                dtype=self.base.weight.dtype,
            )
        )

        nn.init.kaiming_uniform_(
            self.lora_A,
            a=math.sqrt(5),
        )

        self.dropout = (
            nn.Dropout(dropout)
            if dropout and dropout > 0
            else nn.Identity()
        )

        # The wrapped base layer is frozen forever.
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, x):
        base_out = self.base(x)
        delta = (
            self.dropout(x)
            @ self.lora_A.transpose(0, 1)
            @ self.lora_B.transpose(0, 1)
        )
        return base_out + self.scaling * delta


def apply_lora_to_roberta(
    model,
    r=16,
    alpha=32,
    dropout=0.1,
    targets=("query", "value"),
):
    """
    Wrap the query/value Linear projections of every
    self-attention module in the encoder.

    Returns the number of wrapped projections.
    """

    wrapped = 0

    for module in model.modules():

        for attribute_name in targets:

            attribute = getattr(module, attribute_name, None)

            if isinstance(attribute, nn.Linear):
                setattr(
                    module,
                    attribute_name,
                    LoRALinear(
                        attribute,
                        r=r,
                        alpha=alpha,
                        dropout=dropout,
                    ),
                )
                wrapped += 1

    if wrapped == 0:
        raise RuntimeError(
            "apply_lora_to_roberta found no query/value "
            "projections — is this a RoBERTa-style encoder?"
        )

    return wrapped


def lora_state_dict_prefixes(model):
    """Parameter-name prefixes that hold LoRA deltas."""
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and name.split(".")[-1] in ("lora_A", "lora_B")
    ]
