import argparse
import contextlib
import json
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from sklearn.metrics import average_precision_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.feature_dataset import FeatureECFDataset
from dataset.feature_collate import FeatureECPECCollator

# 根据你现在实际使用的文件名导入
try:
    from models.feature_base_model import FeatureECPECBaseModel
except ModuleNotFoundError:
    from models.feature_best_model import FeatureECPECBaseModel


PROJECT_ROOT = Path(__file__).resolve().parent


# =========================================================
# Arguments
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train ECPEC BaseModel using "
            "pre-extracted frozen RoBERTa features."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config_feature_pos2.5_thr0.66.yaml",
        help="Path to feature-training YAML config.",
    )

    return parser.parse_args()


# =========================================================
# Tee stdout
# =========================================================

class TeeStdout:

    def __init__(
        self,
        terminal,
        log_file,
    ):
        self.terminal = terminal
        self.log_file = log_file

    def write(
        self,
        message,
    ):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        if hasattr(
            self.terminal,
            "isatty",
        ):
            return self.terminal.isatty()

        return False

    @property
    def encoding(self):
        return getattr(
            self.terminal,
            "encoding",
            "utf-8",
        )


# =========================================================
# Utilities
# =========================================================

def resolve_path(
    path_value,
):
    path = Path(
        path_value
    )

    if not path.is_absolute():
        path = (
            PROJECT_ROOT
            / path
        )

    return path.resolve()


def load_config(
    config_path,
):
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: "
            f"{config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(
            f
        )

    if config is None:
        raise ValueError(
            f"Empty config file: "
            f"{config_path}"
        )

    for section in [
        "data",
        "model",
        "training",
    ]:
        if section not in config:
            raise KeyError(
                f"Missing config section: "
                f"{section}"
            )

    return config


def validate_config(
    config,
):
    mode = config.get("model", {}).get("pair_decision_mode", "fixed")
    if mode not in ("fixed", "adaptive_reference", "null_reference", "centered_null_reference", "hierarchical_boundary"):
        raise ValueError(f"Unknown pair_decision_mode: {mode}")

    evaluation = config.get("evaluation", {})
    for key in ("emotion_threshold", "cause_threshold"):
        if float(evaluation.get(key, 0.5)) != 0.5:
            raise ValueError(f"{key} must remain 0.5 for this controlled experiment")
    if mode in ("null_reference", "centered_null_reference", "hierarchical_boundary"):
        training = config["training"]
        if int(training.get("null_hard_negative_k", 3)) < 1:
            raise ValueError("null_hard_negative_k must be >= 1")
        if float(training.get("null_margin", 0.2)) < 0 or float(training.get("null_rank_weight", 0.2)) < 0:
            raise ValueError("NULL margin and rank weight must be nonnegative")

    if mode == "centered_null_reference":
        if float(config["training"].get("null_residual_reg_weight", 0.01)) < 0:
            raise ValueError("null_residual_reg_weight must be nonnegative")

    if mode == "hierarchical_boundary":
        for key, default in (("hierarchical_dialogue_reg_weight", 0.005), ("hierarchical_row_reg_weight", 0.01)):
            if float(config["training"].get(key, default)) < 0:
                raise ValueError(f"{key} must be nonnegative")

    data_config = (
        config["data"]
    )

    training_config = (
        config["training"]
    )

    required_data = [
        "feature_root",
        "train_feature_file",
        "dev_feature_file",
    ]

    for key in required_data:
        if key not in data_config:
            raise KeyError(
                f"Missing data.{key}"
            )

    if (
        "pair_pos_weight"
        not in training_config
    ):
        raise KeyError(
            "Missing "
            "training.pair_pos_weight. "
            "Explicitly set a numeric value "
            "or 'auto'."
        )


def set_seed(
    seed,
):
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed(
            seed
        )

        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(
    worker_id,
):
    worker_seed = (
        torch.initial_seed()
        % (2 ** 32)
    )

    random.seed(
        worker_seed
    )

    np.random.seed(
        worker_seed
    )


def count_parameters(
    model,
):
    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return (
        total,
        trainable,
    )


# =========================================================
# Move batch to device
# =========================================================

def move_batch_to_device(
    batch,
    device,
):
    tensor_keys = [
        "utterance_features",
        "speaker_ids",
        "position_ids",
        "utterance_mask",
        "emotion_labels",
        "cause_labels",
        "pair_labels",
        "pair_mask",
        "num_utterances",
    ]

    moved = dict(
        batch
    )

    for key in tensor_keys:

        if key in moved:
            moved[key] = (
                moved[key].to(
                    device,
                    non_blocking=True,
                )
            )

    return moved


# =========================================================
# Loss
# =========================================================

def masked_bce(
    logits,
    labels,
    mask,
    pos_weight=None,
):
    mask = mask.bool()

    if mask.sum().item() == 0:
        return (
            logits.sum()
            * 0.0
        )

    valid_logits = logits[
        mask
    ]

    valid_labels = labels[
        mask
    ].float()

    if pos_weight is None:

        return (
            F.binary_cross_entropy_with_logits(
                valid_logits,
                valid_labels,
            )
        )

    weight_tensor = torch.tensor(
        float(
            pos_weight
        ),
        dtype=valid_logits.dtype,
        device=valid_logits.device,
    )

    return (
        F.binary_cross_entropy_with_logits(
            valid_logits,
            valid_labels,
            pos_weight=weight_tensor,
        )
    )


def decision_pair_logits(outputs):
    """Decision/ranking scores; NULL mode deliberately keeps raw BCE separately."""
    return outputs.get("relative_pair_logits", outputs.get("adjusted_pair_logits", outputs["pair_logits"]))


def row_normalize_pair_logits(pair_logits, pair_mask):
    """
    Subtract the per-emotion-row mean (over valid pairs) before thresholding.

    Motivation
    ----------
    BCE + pos_weight training lets the global mean / scale of the pair
    logits drift from epoch to epoch. With a fixed global threshold the
    effective operating point then slides through the dense mass of
    borderline pairs, flipping hundreds of predictions per epoch on a
    small Dev set (visible as P/R/F1 swings).

    Row-centering makes the decision depend only on the WITHIN-row
    ranking (which the AUPRC curves show to be stable) and removes the
    global drift component completely.
    """
    valid = pair_mask.bool()
    counts = valid.sum(dim=-1, keepdim=True).clamp_min(1)
    row_mean = pair_logits.sum(dim=-1, keepdim=True) / counts
    return (pair_logits - row_mean).masked_fill(~valid, 0.0)


def adaptive_artifact_path(path, mode):
    """Protect baseline artifacts even if only the model mode is changed."""
    path = Path(path)
    if mode == "hierarchical_boundary" and "hierarchical_boundary" not in path.stem:
        return path.with_name(path.stem + "_hierarchical_boundary" + path.suffix)
    if mode == "centered_null_reference" and "centered_null_reference" not in path.stem:
        return path.with_name(path.stem + "_centered_null_reference" + path.suffix)
    if mode == "adaptive_reference" and not path.stem.endswith("_adaptive_reference"):
        return path.with_name(path.stem + "_adaptive_reference" + path.suffix)
    if mode == "null_reference" and "null_reference" not in path.stem:
        return path.with_name(path.stem + "_null_reference" + path.suffix)
    return path


def null_ranking_loss(raw_logits, null_logits, labels, pair_mask, utterance_mask,
                      margin=0.2, hard_negative_k=3):
    """Balance positive/negative groups within a row, then average valid rows."""
    if hard_negative_k < 1:
        raise ValueError("null_hard_negative_k must be >= 1")
    if margin < 0:
        raise ValueError("null_margin must be >= 0")
    scores = raw_logits.detach()
    valid = utterance_mask.bool()
    mask = pair_mask.bool() & valid.unsqueeze(2) & valid.unsqueeze(1)
    rows = []
    for b, i in (valid & mask.any(dim=-1)).nonzero(as_tuple=False):
        row_scores = scores[b, i][mask[b, i]]
        row_labels = labels[b, i][mask[b, i]]
        positive = row_scores[row_labels == 1]
        negative = row_scores[row_labels == 0]
        terms = []
        if positive.numel():
            terms.append(F.softplus(margin + null_logits[b, i] - positive).mean())
        if negative.numel():
            hard = negative.topk(min(hard_negative_k, negative.numel())).values
            terms.append(F.softplus(margin + hard - null_logits[b, i]).mean())
        if terms:
            rows.append(torch.stack(terms).mean())
    if not rows:
        # Empty masked selection gives a differentiable zero without reading padding.
        return null_logits[valid & mask.any(dim=-1)].sum() * 0.0
    return torch.stack(rows).mean()


def pair_ranking_loss(pair_logits, labels, pair_mask, utterance_mask,
                      margin=0.2, hard_negative_k=3):
    """
    Scorer-side margin ranking: pull every positive pair above the top-k
    hardest negatives of its emotion-row.

    Unlike null_ranking_loss this trains the pair scorer itself (no detach).
    It sharpens the score distribution around the decision boundary, which
    reduces the mass of borderline pairs and therefore the epoch-to-epoch
    P/R/F1 swings caused by small score-drift on a small Dev set.
    """
    if hard_negative_k < 1:
        raise ValueError("pair_rank_hard_negative_k must be >= 1")
    if margin < 0:
        raise ValueError("pair_rank_margin must be >= 0")
    valid = utterance_mask.bool()
    mask = pair_mask.bool() & valid.unsqueeze(2) & valid.unsqueeze(1)
    rows = []
    for b, i in (valid & mask.any(dim=-1)).nonzero(as_tuple=False):
        row_scores = pair_logits[b, i][mask[b, i]]
        row_labels = labels[b, i][mask[b, i]]
        positive = row_scores[row_labels == 1]
        negative = row_scores[row_labels == 0]
        if positive.numel() and negative.numel():
            hard = negative.topk(
                min(hard_negative_k, negative.numel())
            ).values
            rows.append(
                F.softplus(
                    margin
                    + hard.unsqueeze(-1)
                    - positive.unsqueeze(0)
                ).mean()
            )
    if not rows:
        # Differentiable zero without reading padding positions.
        return pair_logits[valid & mask.any(dim=-1)].sum() * 0.0
    return torch.stack(rows).mean()


def optimizer_parameters(model):
    if model.pair_decision_mode == "hierarchical_boundary":
        return [
            {"params": [p for name, p in model.named_parameters()
                        if p.requires_grad and name != "global_center"]},
            {"params": [model.global_center], "weight_decay": 0.0},
        ]
    if model.pair_decision_mode == "centered_null_reference":
        return [
            {"params": [p for name, p in model.named_parameters()
                        if p.requires_grad and name != "null_global_center"]},
            {"params": [model.null_global_center], "weight_decay": 0.0},
        ]
    return (p for p in model.parameters() if p.requires_grad)


def clip_model_gradients(model, max_grad_norm):
    if model.pair_decision_mode == "hierarchical_boundary":
        boundary = [model.global_center, *model.dialogue_offset_head.parameters(),
                    *model.row_residual_head.parameters()]
        boundary_ids = {id(p) for p in boundary}
        base = [p for p in model.parameters() if id(p) not in boundary_ids]
        torch.nn.utils.clip_grad_norm_(base, max_grad_norm)
        torch.nn.utils.clip_grad_norm_(boundary, max_grad_norm)
    elif model.pair_decision_mode == "centered_null_reference":
        boundary = [model.null_global_center, *model.null_residual_head.parameters()]
        boundary_ids = {id(p) for p in boundary}
        base = [p for p in model.parameters() if id(p) not in boundary_ids]
        torch.nn.utils.clip_grad_norm_(base, max_grad_norm)
        torch.nn.utils.clip_grad_norm_(boundary, max_grad_norm)
    elif model.pair_decision_mode == "null_reference":
        # Shared clipping would couple NULL gradients back into scorer updates.
        base = [p for name, p in model.named_parameters()
                if not name.startswith("null_reference_head.")]
        torch.nn.utils.clip_grad_norm_(base, max_grad_norm)
        torch.nn.utils.clip_grad_norm_(model.null_reference_head.parameters(), max_grad_norm)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)


def tensor_stats(values):
    if not values.numel():
        return {key: None for key in ("mean", "std", "min", "max")}
    return dict(mean=values.mean().item(), std=values.std(unbiased=False).item(),
                min=values.min().item(), max=values.max().item())


def pearson_or_none(x, y):
    if x.numel() < 2:
        return None
    x, y = x.double() - x.double().mean(), y.double() - y.double().mean()
    denominator = x.norm() * y.norm()
    return (x.dot(y) / denominator).item() if denominator > 0 else None


class NullDiagnostics:
    """Epoch-wide statistics over valid utterances; no new dependencies."""
    def __init__(self):
        self.values, self.dialogue_means, self.within_stds = [], [], []
        self.row_null, self.row_max, self.row_mean = [], [], []

    @torch.no_grad()
    def update(self, outputs, batch):
        null = outputs["null_logits"].detach().float().cpu()
        raw = outputs["pair_logits"].detach().float().cpu()
        valid = batch["utterance_mask"].bool().cpu()
        mask = batch["pair_mask"].bool().cpu() & valid.unsqueeze(2) & valid.unsqueeze(1)
        self.values.append(null[valid])
        for b in range(null.shape[0]):
            values = null[b][valid[b]]
            if values.numel():
                self.dialogue_means.append(values.mean())
                self.within_stds.append(values.std(unbiased=False))
            for i in (valid[b] & mask[b].any(-1)).nonzero().flatten():
                scores = raw[b, i][mask[b, i]]
                self.row_null.append(null[b, i])
                self.row_max.append(scores.max())
                self.row_mean.append(scores.mean())

    def compute(self):
        values = torch.cat(self.values) if self.values else torch.empty(0)
        row_null = torch.stack(self.row_null) if self.row_null else torch.empty(0)
        maximum = torch.stack(self.row_max) if self.row_max else torch.empty(0)
        average = torch.stack(self.row_mean) if self.row_mean else torch.empty(0)
        return {
            "logit": tensor_stats(values),
            "probability": tensor_stats(values.sigmoid()),
            "variation": {
                "utterance_std": tensor_stats(values)["std"],
                "dialogue_mean_std": (torch.stack(self.dialogue_means).std(unbiased=False).item()
                                      if self.dialogue_means else None),
                "mean_within_dialogue_std": (torch.stack(self.within_stds).mean().item()
                                             if self.within_stds else None),
                "unique_valid_references": values.unique().numel(),
                "unique_rounded_4dp": (values * 10000).round().unique().numel(),
                "pearson_row_max": pearson_or_none(row_null, maximum),
                "pearson_row_mean": pearson_or_none(row_null, average),
            },
        }


class CenteredNullDiagnostics(NullDiagnostics):
    """Keep v2 diagnostics intact; extend only centered-mode statistics."""
    def __init__(self):
        super().__init__()
        self.residuals, self.positive_rows, self.no_positive_rows = [], [], []
        self.center = None

    @torch.no_grad()
    def update(self, outputs, batch):
        super().update(outputs, batch)
        valid = batch["utterance_mask"].bool()
        mask = batch["pair_mask"].bool() & valid.unsqueeze(2) & valid.unsqueeze(1)
        positive = ((batch["pair_labels"] == 1) & mask).any(-1) & valid
        null = outputs["null_logits"].detach()
        self.positive_rows.append(null[positive].float().cpu())
        self.no_positive_rows.append(null[valid & ~positive].float().cpu())
        self.residuals.append(outputs["null_residual"][valid].detach().float().cpu())
        self.center = outputs["null_global_center"].detach().float().cpu().clone()

    def compute(self):
        stats = super().compute()
        residual = torch.cat(self.residuals) if self.residuals else torch.empty(0)
        positive = torch.cat(self.positive_rows) if self.positive_rows else torch.empty(0)
        negative = torch.cat(self.no_positive_rows) if self.no_positive_rows else torch.empty(0)
        stats.update({
            "global_center_logit": self.center.item() if self.center is not None else None,
            "global_center_probability": self.center.sigmoid().item() if self.center is not None else None,
            "residual": tensor_stats(residual),
            "residual_absolute_mean": residual.abs().mean().item() if residual.numel() else None,
            "positive_row_boundary": {**tensor_stats(positive), "count": positive.numel()},
            "no_positive_row_boundary": {**tensor_stats(negative), "count": negative.numel()},
        })
        return stats


class HierarchicalDiagnostics(CenteredNullDiagnostics):
    """Reuse valid-row boundary/group diagnostics without changing older modes."""
    def __init__(self):
        super().__init__()
        self.offsets, self.dialogue_probabilities = [], []
        self.raw_rows, self.bounded_rows, self.row_means = [], [], []

    @torch.no_grad()
    def update(self, outputs, batch):
        super().update({**outputs,
                        "null_global_center": outputs["global_center"],
                        "null_residual": outputs["row_residual_zero_mean"]}, batch)
        valid = batch["utterance_mask"].bool()
        sample_valid = valid.any(-1)
        offset = outputs["dialogue_offset"][sample_valid].detach().float().cpu()
        self.offsets.append(offset)
        self.dialogue_probabilities.append((self.center + offset).sigmoid())
        self.raw_rows.append(outputs["row_residual_raw"][valid].detach().float().cpu())
        self.bounded_rows.append(outputs["row_residual_bounded"][valid].detach().float().cpu())
        row = outputs["row_residual_zero_mean"].detach().masked_fill(~valid, 0.0)
        means = row.sum(-1) / valid.sum(-1).clamp_min(1)
        self.row_means.append(means[sample_valid].float().cpu())

    def compute(self):
        stats = super().compute()
        offsets = torch.cat(self.offsets) if self.offsets else torch.empty(0)
        raw = torch.cat(self.raw_rows) if self.raw_rows else torch.empty(0)
        bounded = torch.cat(self.bounded_rows) if self.bounded_rows else torch.empty(0)
        means = torch.cat(self.row_means) if self.row_means else torch.empty(0)
        probs = torch.cat(self.dialogue_probabilities) if self.dialogue_probabilities else torch.empty(0)
        absolute_offset = offsets.abs().mean().item() if offsets.numel() else None
        stats.update({
            "dialogue_offset": tensor_stats(offsets),
            "dialogue_boundary_probability": tensor_stats(probs),
            "row_residual_raw": tensor_stats(raw),
            "row_residual_bounded": tensor_stats(bounded),
            "row_residual_zero_mean": stats["residual"],
            "mean_absolute_dialogue_row_mean": means.abs().mean().item() if means.numel() else None,
            "max_absolute_dialogue_row_mean": means.abs().max().item() if means.numel() else None,
            "dialogue_offset_absolute_mean": absolute_offset,
            "row_residual_absolute_mean": stats["residual_absolute_mean"],
            "boundary_contribution": {
                "global": abs(stats["global_center_logit"]) if self.center is not None else None,
                "dialogue": absolute_offset,
                "row": stats["residual_absolute_mean"],
            },
        })
        return stats


def compute_losses(
    outputs,
    batch,
    pair_pos_weight,
    lambda_emotion,
    lambda_cause,
    null_rank_weight=0.2,
    null_margin=0.2,
    null_hard_negative_k=3,
    null_residual_reg_weight=0.01,
    hierarchical_dialogue_reg_weight=0.005,
    hierarchical_row_reg_weight=0.01,
    pair_rank_weight=0.0,
    pair_rank_margin=0.2,
    pair_rank_hard_negative_k=3,
):
    emotion_loss = masked_bce(
        logits=outputs[
            "emotion_logits"
        ],
        labels=batch[
            "emotion_labels"
        ],
        mask=batch[
            "utterance_mask"
        ],
    )

    cause_loss = masked_bce(
        logits=outputs[
            "cause_logits"
        ],
        labels=batch[
            "cause_labels"
        ],
        mask=batch[
            "utterance_mask"
        ],
    )

    pair_loss = masked_bce(
        logits=(outputs["pair_logits"] if "null_logits" in outputs
                else decision_pair_logits(outputs)),
        labels=batch[
            "pair_labels"
        ],
        mask=batch[
            "pair_mask"
        ],
        pos_weight=(
            pair_pos_weight
        ),
    )

    total_loss = (
        pair_loss
        + lambda_emotion
        * emotion_loss
        + lambda_cause
        * cause_loss
    )

    result = {
        "total_loss":
            total_loss,

        "pair_loss":
            pair_loss,

        "emotion_loss":
            emotion_loss,

        "cause_loss":
            cause_loss,
    }
    if pair_rank_weight > 0:
        rank_loss = pair_ranking_loss(
            outputs["pair_logits"],
            batch["pair_labels"],
            batch["pair_mask"],
            batch["utterance_mask"],
            margin=pair_rank_margin,
            hard_negative_k=pair_rank_hard_negative_k,
        )
        result["pair_rank_loss"] = rank_loss
        result["total_loss"] = total_loss + pair_rank_weight * rank_loss
    if "null_logits" in outputs:
        rank_loss = null_ranking_loss(
            outputs["pair_logits"], outputs["null_logits"],
            batch["pair_labels"], batch["pair_mask"], batch["utterance_mask"],
            margin=null_margin, hard_negative_k=null_hard_negative_k,
        )
        result["null_rank_loss"] = rank_loss
        result["total_loss"] = total_loss + null_rank_weight * rank_loss
    if "null_residual" in outputs:
        residual = outputs["null_residual"][batch["utterance_mask"].bool()]
        residual_loss = residual.square().mean() if residual.numel() else residual.sum() * 0.0
        result["residual_reg_loss"] = residual_loss
        result["total_loss"] = result["total_loss"] + null_residual_reg_weight * residual_loss
    if "dialogue_offset" in outputs:
        valid = batch["utterance_mask"].bool()
        dialogue = outputs["dialogue_offset"][valid.any(-1)]
        row = outputs["row_residual_zero_mean"][valid]
        dialogue_loss = dialogue.square().mean() if dialogue.numel() else dialogue.sum() * 0.0
        row_loss = row.square().mean() if row.numel() else row.sum() * 0.0
        result["dialogue_reg_loss"] = dialogue_loss
        result["row_reg_loss"] = row_loss
        result["total_loss"] = (result["total_loss"]
                                + hierarchical_dialogue_reg_weight * dialogue_loss
                                + hierarchical_row_reg_weight * row_loss)
    return result



# =========================================================
# Metrics
# =========================================================

class BinaryMetricAccumulator:

    def __init__(
        self,
        threshold=0.5,
        zero_centered=False,
    ):
        self.zero_centered = zero_centered
        self.threshold = float(
            threshold
        )

        self.reset()

    def reset(
        self,
    ):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    @torch.no_grad()
    def update(
        self,
        logits,
        labels,
        mask,
    ):
        mask = mask.bool()

        if (
            mask.sum().item()
            == 0
        ):
            return

        logits = logits[
            mask
        ]

        labels = labels[
            mask
        ]

        probabilities = (
            torch.sigmoid(
                logits
            )
        )

        predictions = (
            logits > 0 if self.zero_centered
            else probabilities >= self.threshold
        )

        gold = (
            labels
            >= 0.5
        )

        self.tp += int(
            (
                predictions
                & gold
            )
            .sum()
            .item()
        )

        self.fp += int(
            (
                predictions
                & (~gold)
            )
            .sum()
            .item()
        )

        self.fn += int(
            (
                (~predictions)
                & gold
            )
            .sum()
            .item()
        )

        self.tn += int(
            (
                (~predictions)
                & (~gold)
            )
            .sum()
            .item()
        )

    def compute(
        self,
    ):
        predicted_positive = (
            self.tp
            + self.fp
        )

        gold_positive = (
            self.tp
            + self.fn
        )

        total = (
            self.tp
            + self.fp
            + self.fn
            + self.tn
        )

        precision = (
            self.tp
            / predicted_positive
            if predicted_positive > 0
            else 0.0
        )

        recall = (
            self.tp
            / gold_positive
            if gold_positive > 0
            else 0.0
        )

        f1 = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
            )
            if (
                precision
                + recall
            ) > 0
            else 0.0
        )

        accuracy = (
            (
                self.tp
                + self.tn
            )
            / total
            if total > 0
            else 0.0
        )

        return {
            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "accuracy":
                accuracy,

            "tp":
                self.tp,

            "fp":
                self.fp,

            "fn":
                self.fn,

            "tn":
                self.tn,

            "predicted_positive":
                predicted_positive,

            "gold_positive":
                gold_positive,
        }


class ECPECMetricAccumulator:

    def __init__(self, threshold=0.66, pair_decision_mode="fixed", row_normalize=False):
        self.pair_decision_mode = pair_decision_mode
        self.row_normalize = bool(row_normalize)
        self.emotion = BinaryMetricAccumulator(0.5)
        self.cause = BinaryMetricAccumulator(0.5)
        self.pair = BinaryMetricAccumulator(
            threshold, zero_centered=pair_decision_mode in ("adaptive_reference", "null_reference", "centered_null_reference", "hierarchical_boundary"),
        )
        self.references = []
        self.null_diagnostics = (HierarchicalDiagnostics() if pair_decision_mode == "hierarchical_boundary" else
                                 CenteredNullDiagnostics() if pair_decision_mode == "centered_null_reference" else
                                 NullDiagnostics() if pair_decision_mode == "null_reference" else None)

    @torch.no_grad()
    def update(
        self,
        outputs,
        batch,
    ):
        self.emotion.update(
            outputs[
                "emotion_logits"
            ],
            batch[
                "emotion_labels"
            ],
            batch[
                "utterance_mask"
            ],
        )

        self.cause.update(
            outputs[
                "cause_logits"
            ],
            batch[
                "cause_labels"
            ],
            batch[
                "utterance_mask"
            ],
        )

        if self.pair_decision_mode == "adaptive_reference":
            self.references.append(
                outputs["adaptive_reference"][batch["utterance_mask"].bool()]
                .detach().float().cpu()
            )
        if self.null_diagnostics is not None:
            self.null_diagnostics.update(outputs, batch)
        pair_logits = decision_pair_logits(outputs)
        if self.row_normalize:
            pair_logits = row_normalize_pair_logits(
                pair_logits,
                batch["pair_mask"],
            )
        self.pair.update(
            pair_logits,
            batch[
                "pair_labels"
            ],
            batch[
                "pair_mask"
            ],
        )

    def compute(
        self,
    ):
        reference_stats = None
        if self.references:
            values = torch.cat(self.references)
            if values.numel():
                reference_stats = {
                    "mean": values.mean().item(),
                    "std": values.std(unbiased=False).item(),
                    "min": values.min().item(),
                    "max": values.max().item(),
                }
        return {
            "pair_decision_mode": self.pair_decision_mode,
            "reference_stats": reference_stats,
            **({"null_stats": self.null_diagnostics.compute()} if self.null_diagnostics is not None else {}),
            "emotion":
                self.emotion.compute(),

            "cause":
                self.cause.compute(),

            "pair":
                self.pair.compute(),
        }


# =========================================================
# Loss accumulator
# =========================================================

class LossAccumulator:

    def __init__(
        self,
    ):
        self.reset()

    def reset(
        self,
    ):
        self.total_loss = 0.0
        self.pair_loss = 0.0
        self.emotion_loss = 0.0
        self.cause_loss = 0.0
        self.num_batches = 0
        self.null_rank_loss = None
        self.pair_rank_loss = None
        self.residual_reg_loss = None
        self.hierarchical_losses = {}

    def update(
        self,
        losses,
    ):
        self.total_loss += float(
            losses[
                "total_loss"
            ]
            .detach()
            .item()
        )

        self.pair_loss += float(
            losses[
                "pair_loss"
            ]
            .detach()
            .item()
        )

        self.emotion_loss += float(
            losses[
                "emotion_loss"
            ]
            .detach()
            .item()
        )

        self.cause_loss += float(
            losses[
                "cause_loss"
            ]
            .detach()
            .item()
        )

        if "null_rank_loss" in losses:
            self.null_rank_loss = (self.null_rank_loss or 0.0) + losses["null_rank_loss"].detach().item()
        if "pair_rank_loss" in losses:
            self.pair_rank_loss = (self.pair_rank_loss or 0.0) + losses["pair_rank_loss"].detach().item()
        if "residual_reg_loss" in losses:
            self.residual_reg_loss = (self.residual_reg_loss or 0.0) + losses["residual_reg_loss"].detach().item()
        for key in ("dialogue_reg_loss", "row_reg_loss"):
            if key in losses:
                self.hierarchical_losses[key] = self.hierarchical_losses.get(key, 0.0) + losses[key].detach().item()
        self.num_batches += 1

    def compute(
        self,
    ):
        denominator = max(
            self.num_batches,
            1,
        )

        return {
            **{key: value / denominator for key, value in self.hierarchical_losses.items()},
            **({"residual_reg_loss": self.residual_reg_loss / denominator} if self.residual_reg_loss is not None else {}),
            **({"null_rank_loss": self.null_rank_loss / denominator} if self.null_rank_loss is not None else {}),
            **({"pair_rank_loss": self.pair_rank_loss / denominator} if self.pair_rank_loss is not None else {}),
            "total_loss":
                self.total_loss
                / denominator,

            "pair_loss":
                self.pair_loss
                / denominator,

            "emotion_loss":
                self.emotion_loss
                / denominator,

            "cause_loss":
                self.cause_loss
                / denominator,
        }


# =========================================================
# Model forward
# =========================================================

def forward_model(
    model,
    batch,
):
    outputs = model(
        utterance_features=batch[
            "utterance_features"
        ],

        speaker_ids=batch[
            "speaker_ids"
        ],

        position_ids=batch[
            "position_ids"
        ],

        utterance_mask=batch[
            "utterance_mask"
        ],

        pair_mask=batch[
            "pair_mask"
        ],
    )

    if not isinstance(
        outputs,
        dict,
    ):
        raise TypeError(
            "FeatureECPECBaseModel must "
            "return a dictionary."
        )

    required = [
        "emotion_logits",
        "cause_logits",
        "pair_logits",
    ]

    for key in required:

        if key not in outputs:
            raise KeyError(
                f"Model output missing: "
                f"{key}"
            )

    return outputs


# =========================================================
# One training epoch
# =========================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    pair_pos_weight,
    lambda_emotion,
    lambda_cause,
    threshold,
    max_grad_norm,
    epoch,
    total_epochs,
    null_loss_options=None,
    row_normalize=False,
):
    model.train()

    metric_accumulator = (
        ECPECMetricAccumulator(
            threshold=threshold,
            pair_decision_mode=model.pair_decision_mode,
            row_normalize=row_normalize,
        )
    )

    loss_accumulator = (
        LossAccumulator()
    )

    progress = tqdm(
        loader,
        desc=(
            f"Epoch "
            f"{epoch:02d}/"
            f"{total_epochs:02d} "
            f"[Train]"
        ),
        dynamic_ncols=True,
    )

    for batch in progress:

        batch = (
            move_batch_to_device(
                batch,
                device,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = (
            forward_model(
                model,
                batch,
            )
        )

        losses = (
            compute_losses(
                outputs=outputs,
                batch=batch,
                **(null_loss_options or {}),

                pair_pos_weight=(
                    pair_pos_weight
                ),

                lambda_emotion=(
                    lambda_emotion
                ),

                lambda_cause=(
                    lambda_cause
                ),
            )
        )

        losses[
            "total_loss"
        ].backward()

        if (
            max_grad_norm
            is not None
            and
            max_grad_norm > 0
        ):
            clip_model_gradients(model, max_grad_norm)

        optimizer.step()

        loss_accumulator.update(
            losses
        )

        metric_accumulator.update(
            outputs,
            batch,
        )

        progress.set_postfix(
            {
                "loss":
                    f"{losses['total_loss'].item():.4f}",

                "pair":
                    f"{losses['pair_loss'].item():.4f}",
            }
        )

    return {
        "losses":
            loss_accumulator.compute(),

        "metrics":
            metric_accumulator.compute(),
    }


# =========================================================
# Evaluation
#
# IMPORTANT:
# AUPRC is calculated globally over all valid Dev pairs.
# It is NOT averaged batch-by-batch.
# =========================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    pair_pos_weight,
    lambda_emotion,
    lambda_cause,
    threshold,
    description="Dev",
    null_loss_options=None,
    row_normalize=False,
):
    model.eval()

    metric_accumulator = (
        ECPECMetricAccumulator(
            threshold=threshold,
            pair_decision_mode=model.pair_decision_mode,
            row_normalize=row_normalize,
        )
    )

    loss_accumulator = (
        LossAccumulator()
    )

    # -----------------------------------------------------
    # AUPRC:
    # collect all valid Pair logits / labels
    # across the entire evaluation set
    # -----------------------------------------------------

    all_pair_logits = []
    all_raw_pair_logits = []
    all_pair_labels = []

    progress = tqdm(
        loader,
        desc=(
            f"[{description}]"
        ),
        dynamic_ncols=True,
    )

    for batch in progress:

        batch = (
            move_batch_to_device(
                batch,
                device,
            )
        )

        outputs = (
            forward_model(
                model,
                batch,
            )
        )

        losses = (
            compute_losses(
                outputs=outputs,
                batch=batch,
                **(null_loss_options or {}),

                pair_pos_weight=(
                    pair_pos_weight
                ),

                lambda_emotion=(
                    lambda_emotion
                ),

                lambda_cause=(
                    lambda_cause
                ),
            )
        )

        loss_accumulator.update(
            losses
        )

        metric_accumulator.update(
            outputs,
            batch,
        )

        # -------------------------------------------------
        # Collect valid Pair logits for global AUPRC
        # -------------------------------------------------

        pair_mask = (
            batch[
                "pair_mask"
            ]
            .bool()
        )

        pair_logits = decision_pair_logits(outputs)

        if row_normalize:
            pair_logits = row_normalize_pair_logits(
                pair_logits,
                batch["pair_mask"],
            )

        valid_pair_logits = (
            pair_logits[pair_mask]
        )

        valid_pair_labels = (
            batch[
                "pair_labels"
            ][
                pair_mask
            ]
        )

        all_pair_logits.append(
            valid_pair_logits
            .detach()
            .float()
            .cpu()
        )

        if model.pair_decision_mode in ("null_reference", "centered_null_reference", "hierarchical_boundary"):
            all_raw_pair_logits.append(outputs["pair_logits"][pair_mask].detach().float().cpu())

        all_pair_labels.append(
            valid_pair_labels
            .detach()
            .float()
            .cpu()
        )

        progress.set_postfix(
            {
                "loss":
                    f"{losses['total_loss'].item():.4f}"
            }
        )

    # -----------------------------------------------------
    # Global Pair AUPRC
    # -----------------------------------------------------

    if len(
        all_pair_logits
    ) == 0:
        raise RuntimeError(
            "No valid Pair logits "
            "collected during evaluation."
        )

    all_pair_logits = torch.cat(
        all_pair_logits,
        dim=0,
    )

    all_pair_labels = torch.cat(
        all_pair_labels,
        dim=0,
    )

    if (
        all_pair_logits.numel()
        != all_pair_labels.numel()
    ):
        raise RuntimeError(
            "Pair logits / labels "
            "count mismatch."
        )

    pair_auprc = (
        average_precision_score(
            all_pair_labels.numpy(),
            all_pair_logits.numpy(),
        )
    )

    null_auprc = {}
    if model.pair_decision_mode in ("null_reference", "centered_null_reference", "hierarchical_boundary"):
        null_auprc = {
            "raw_pair_auprc": float(average_precision_score(
                all_pair_labels.numpy(), torch.cat(all_raw_pair_logits).numpy(),
            )),
            "relative_pair_auprc": float(pair_auprc),
        }

    return {
        **null_auprc,
        "losses":
            loss_accumulator.compute(),

        "metrics":
            metric_accumulator.compute(),

        "pair_auprc":
            float(
                pair_auprc
            ),

        "num_pair_samples":
            int(
                all_pair_labels.numel()
            ),

        "num_positive_pairs":
            int(
                all_pair_labels
                .sum()
                .item()
            ),
    }


# =========================================================
# Pretty printing
# =========================================================

def print_result(
    epoch,
    total_epochs,
    train_result,
    dev_result,
):
    print("Pair decision mode:", dev_result["metrics"]["pair_decision_mode"])
    for split, result in (("Train", train_result), ("Dev", dev_result)):
        stats = result["metrics"]["reference_stats"]
        if stats is not None:
            print(f"{split} Reference | " + " | ".join(
                f"{key}={value:.6f}" for key, value in stats.items()
            ))
    if "raw_pair_auprc" in dev_result:
        print("Pair decision: raw_pair_logit > dynamic NULL reference")
        print("No manually tuned global Pair threshold")
        print(f"Dev Raw Pair AUPRC      | {dev_result['raw_pair_auprc']:.4f}")
        print(f"Dev Relative Pair AUPRC | {dev_result['relative_pair_auprc']:.4f}")
        stats = dev_result["metrics"]["null_stats"]
        for title, key in (("NULL Logit", "logit"), ("Dynamic Boundary", "probability")):
            print(f"Dev {title}: " + " | ".join(f"{k}={v:.6f}" for k, v in stats[key].items()))
        print("Dev NULL Variation:", stats["variation"])
        print("NULL Rank Loss | Train={:.6f} Dev={:.6f}".format(
            train_result["losses"]["null_rank_loss"], dev_result["losses"]["null_rank_loss"],
        ))
    if dev_result["metrics"]["pair_decision_mode"] == "centered_null_reference":
        stats = dev_result["metrics"]["null_stats"]
        print("Global Center Logit:", stats["global_center_logit"])
        print("Global Center Probability:", stats["global_center_probability"])
        for title, key in (("Residual Logit", "residual"),
                           ("Dynamic Boundary Logit", "logit"),
                           ("Dynamic Boundary Probability", "probability"),
                           ("Positive-row boundary", "positive_row_boundary"),
                           ("No-positive-row boundary", "no_positive_row_boundary")):
            print(f"Dev {title}: " + " | ".join(
                f"{k}={v:.6f}" if v is not None else f"{k}=N/A" for k, v in stats[key].items()
            ))
        print("Residual absolute mean:", stats["residual_absolute_mean"])
        print("Residual std:", stats["residual"]["std"])
        print("Residual Reg Loss | Train={:.6f} Dev={:.6f}".format(
            train_result["losses"]["residual_reg_loss"], dev_result["losses"]["residual_reg_loss"],
        ))
    if dev_result["metrics"]["pair_decision_mode"] == "hierarchical_boundary":
        stats = dev_result["metrics"]["null_stats"]
        print("Global Center Logit:", stats["global_center_logit"])
        print("Global Center Probability:", stats["global_center_probability"])
        for title, key in (("Dialogue Offset", "dialogue_offset"),
                           ("Dialogue Boundary Probability", "dialogue_boundary_probability"),
                           ("Raw Row Residual", "row_residual_raw"),
                           ("Bounded Row Residual", "row_residual_bounded"),
                           ("Zero-mean Row Residual", "row_residual_zero_mean"),
                           ("Dynamic Boundary Logit", "logit"),
                           ("Dynamic Boundary Probability", "probability"),
                           ("Positive-row boundary", "positive_row_boundary"),
                           ("No-positive-row boundary", "no_positive_row_boundary")):
            print(f"Dev {title}: " + " | ".join(
                f"{k}={v:.6f}" if v is not None else f"{k}=N/A" for k, v in stats[key].items()
            ))
        print("Mean absolute dialogue-wise row residual mean:", stats["mean_absolute_dialogue_row_mean"])
        print("Dialogue offset absolute mean:", stats["dialogue_offset_absolute_mean"])
        print("Dialogue offset std:", stats["dialogue_offset"]["std"])
        print("Row residual absolute mean:", stats["row_residual_absolute_mean"])
        print("Row residual std:", stats["row_residual_zero_mean"]["std"])
        print("Boundary contribution:", stats["boundary_contribution"])
        for key in ("dialogue_reg_loss", "row_reg_loss"):
            print(f"{key} | Train={train_result['losses'][key]:.6f} Dev={dev_result['losses'][key]:.6f}")
    train_loss = (
        train_result[
            "losses"
        ]
    )

    dev_loss = (
        dev_result[
            "losses"
        ]
    )

    train_metrics = (
        train_result[
            "metrics"
        ]
    )

    dev_metrics = (
        dev_result[
            "metrics"
        ]
    )

    print()
    print(
        "=" * 44
    )

    print(
        f"Epoch "
        f"{epoch:02d}/"
        f"{total_epochs:02d}"
    )

    print(
        "-" * 44
    )

    print(
        "Train Loss: "
        f"{train_loss['total_loss']:.4f} "
        "("
        f"Pair={train_loss['pair_loss']:.4f}, "
        f"Emotion={train_loss['emotion_loss']:.4f}, "
        f"Cause={train_loss['cause_loss']:.4f}"
        + (
            f", PairRank={train_loss['pair_rank_loss']:.4f}"
            if "pair_rank_loss" in train_loss
            else ""
        )
        + ")"
    )

    emotion = (
        train_metrics[
            "emotion"
        ]
    )

    cause = (
        train_metrics[
            "cause"
        ]
    )

    pair = (
        train_metrics[
            "pair"
        ]
    )

    print(
        "Train Emotion | "
        f"P={emotion['precision']:.4f} "
        f"R={emotion['recall']:.4f} "
        f"F1={emotion['f1']:.4f}"
    )

    print(
        "Train Cause   | "
        f"P={cause['precision']:.4f} "
        f"R={cause['recall']:.4f} "
        f"F1={cause['f1']:.4f}"
    )

    print(
        "Train Pair    | "
        f"P={pair['precision']:.4f} "
        f"R={pair['recall']:.4f} "
        f"F1={pair['f1']:.4f} "
        f"#Pred={pair['predicted_positive']} "
        f"#Gold={pair['gold_positive']}"
    )

    print()

    print(
        "Dev   Loss: "
        f"{dev_loss['total_loss']:.4f} "
        "("
        f"Pair={dev_loss['pair_loss']:.4f}, "
        f"Emotion={dev_loss['emotion_loss']:.4f}, "
        f"Cause={dev_loss['cause_loss']:.4f}"
        + (
            f", PairRank={dev_loss['pair_rank_loss']:.4f}"
            if "pair_rank_loss" in dev_loss
            else ""
        )
        + ")"
    )

    emotion = (
        dev_metrics[
            "emotion"
        ]
    )

    cause = (
        dev_metrics[
            "cause"
        ]
    )

    pair = (
        dev_metrics[
            "pair"
        ]
    )

    print(
        "Dev   Emotion | "
        f"P={emotion['precision']:.4f} "
        f"R={emotion['recall']:.4f} "
        f"F1={emotion['f1']:.4f}"
    )

    print(
        "Dev   Cause   | "
        f"P={cause['precision']:.4f} "
        f"R={cause['recall']:.4f} "
        f"F1={cause['f1']:.4f}"
    )

    print(
        "Dev   Pair    | "
        f"P={pair['precision']:.4f} "
        f"R={pair['recall']:.4f} "
        f"F1={pair['f1']:.4f} "
        f"#Pred={pair['predicted_positive']} "
        f"#Gold={pair['gold_positive']}"
    )

    # -----------------------------------------------------
    # New: threshold-free Pair ranking metric
    # -----------------------------------------------------

    print(
        "Dev   Pair AUPRC | "
        f"{dev_result['pair_auprc']:.4f} "
        f"(N={dev_result['num_pair_samples']}, "
        f"Pos={dev_result['num_positive_pairs']})"
    )


# =========================================================
# Checkpoint
# =========================================================

def update_ema_state(model, ema_state, decay):
    """
    Exponentially average model parameters across epochs.

    ema_state: dict name -> detached parameter tensor, or None.
    Returns the (possibly newly created) ema_state.
    """
    if ema_state is None:
        return {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }

    with torch.no_grad():
        for name, param in model.named_parameters():
            ema_state[name].mul_(
                decay
            ).add_(
                param.detach(),
                alpha=1.0 - decay,
            )

    return ema_state


def ema_state_dict(model, ema_state):
    """Model state_dict with parameters replaced by EMA parameters."""
    state = model.state_dict()

    for name in ema_state:
        state[name] = ema_state[name]

    return state


@contextlib.contextmanager
def ema_weights_context(model, ema_state):
    """Temporarily load EMA parameters into the model for evaluation."""
    if not ema_state:
        yield
        return

    backup = {
        name: param.data.clone()
        for name, param in model.named_parameters()
    }

    for name, param in model.named_parameters():
        param.data.copy_(ema_state[name])

    try:
        yield
    finally:
        for name, param in model.named_parameters():
            param.data.copy_(backup[name])


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_pair_f1,
    threshold,
    config,
    pair_auprc,
    dev_result=None,
    state_dict=None,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            **({
                "raw_pair_auprc": dev_result["raw_pair_auprc"],
                "relative_pair_auprc": dev_result["relative_pair_auprc"],
                "null_stats": dev_result["metrics"]["null_stats"],
                "selection_metric": "Dev Relative Pair-F1",
            } if model.pair_decision_mode in ("null_reference", "centered_null_reference", "hierarchical_boundary") and dev_result is not None else {}),
            "epoch":
                epoch,

            "model_state_dict":
                (
                    state_dict
                    if state_dict is not None
                    else model.state_dict()
                ),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "best_pair_f1":
                best_pair_f1,

            "pair_auprc":
                float(
                    pair_auprc
                ),

            "threshold":
                threshold,
            "pair_decision_mode": model.pair_decision_mode,
            "emotion_threshold": 0.5,
            "cause_threshold": 0.5,

            "config":
                config,

            "model_type":
                "FrozenFeatureECPEC",
        },
        path,
    )


def load_checkpoint(
    path,
    model,
    device,
):
    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:
        checkpoint = torch.load(
            path,
            map_location=device,
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    return checkpoint


# =========================================================
# History
# =========================================================

def make_history_entry(
    epoch,
    train_result,
    dev_result,
):
    return {
        "epoch":
            int(
                epoch
            ),

        "train":
            train_result,

        "dev":
            dev_result,
    }


def save_history(
    history,
    history_path,
):
    history_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            history,
            f,
            ensure_ascii=False,
            indent=2,
        )


# =========================================================
# DataLoader
# =========================================================

def build_dataloaders(
    train_dataset,
    dev_dataset,
    collator,
    batch_size,
    num_workers,
    device,
    seed,
):
    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        seed
    )

    common = {
        "batch_size":
            batch_size,

        "num_workers":
            num_workers,

        "collate_fn":
            collator,

        "pin_memory":
            device.type
            == "cuda",
    }

    if num_workers > 0:

        common[
            "persistent_workers"
        ] = True

        common[
            "worker_init_fn"
        ] = seed_worker

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        generator=generator,
        **common,
    )

    dev_loader = DataLoader(
        dev_dataset,
        shuffle=False,
        drop_last=False,
        **common,
    )

    return (
        train_loader,
        dev_loader,
    )


# =========================================================
# Main training
# =========================================================

def run_training(
    config,
    config_path,
):
    data_config = (
        config[
            "data"
        ]
    )

    model_config = (
        config[
            "model"
        ]
    )

    training_config = (
        config[
            "training"
        ]
    )

    evaluation_config = (
        config.get(
            "evaluation",
            {},
        )
    )

    # -----------------------------------------------------
    # Seed / Device
    # -----------------------------------------------------

    seed = int(
        config.get(
            "seed",
            42,
        )
    )

    set_seed(
        seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "=" * 52
    )

    print(
        "ECPEC Frozen-Feature Base Training"
    )

    print(
        "=" * 52
    )

    print(
        "Project root:",
        PROJECT_ROOT,
    )

    print(
        "Config:",
        config_path,
    )

    print(
        "Seed:",
        seed,
    )

    print(
        "Device:",
        device,
    )

    if (
        device.type
        == "cuda"
    ):
        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

        gpu_memory = (
            torch.cuda
            .get_device_properties(
                0
            )
            .total_memory
            / 1024
            / 1024
            / 1024
        )

        print(
            "GPU memory:",
            f"{gpu_memory:.2f} GB",
        )

    # -----------------------------------------------------
    # Feature paths
    # -----------------------------------------------------

    feature_root = (
        resolve_path(
            data_config[
                "feature_root"
            ]
        )
    )

    train_feature_path = (
        feature_root
        / data_config[
            "train_feature_file"
        ]
    ).resolve()

    dev_feature_path = (
        feature_root
        / data_config[
            "dev_feature_file"
        ]
    ).resolve()

    if not (
        train_feature_path.exists()
    ):
        raise FileNotFoundError(
            "Train feature not found: "
            f"{train_feature_path}"
        )

    if not (
        dev_feature_path.exists()
    ):
        raise FileNotFoundError(
            "Dev feature not found: "
            f"{dev_feature_path}"
        )

    max_dialogue_length = int(
        data_config.get(
            "max_dialogue_length",
            40,
        )
    )

    num_workers = int(
        data_config.get(
            "num_workers",
            4,
        )
    )

    hidden_size = int(
        model_config.get(
            "input_dim",
            768,
        )
    )

    print()
    print(
        "===== Frozen Features ====="
    )

    print(
        "Train:",
        train_feature_path,
    )

    print(
        "Dev:",
        dev_feature_path,
    )

    print(
        "Feature dim:",
        hidden_size,
    )

    print(
        "Max dialogue length:",
        max_dialogue_length,
    )

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    train_dataset = (
        FeatureECFDataset(
            feature_path=(
                train_feature_path
            ),

            max_dialogue_length=(
                max_dialogue_length
            ),

            expected_hidden_size=(
                hidden_size
            ),

            validate=True,
        )
    )

    dev_dataset = (
        FeatureECFDataset(
            feature_path=(
                dev_feature_path
            ),

            max_dialogue_length=(
                max_dialogue_length
            ),

            expected_hidden_size=(
                hidden_size
            ),

            validate=True,
        )
    )

    print()
    print(
        "===== Dataset Statistics ====="
    )

    print(
        "Train dialogues:",
        len(
            train_dataset
        ),
    )

    print(
        "Train utterances:",
        train_dataset.num_utterances,
    )

    print(
        "Train candidate pairs:",
        train_dataset.num_candidate_pairs,
    )

    print(
        "Train positive pairs:",
        train_dataset.num_positive_pairs,
    )

    print(
        "Dev dialogues:",
        len(
            dev_dataset
        ),
    )

    print(
        "Dev utterances:",
        dev_dataset.num_utterances,
    )

    print(
        "Dev candidate pairs:",
        dev_dataset.num_candidate_pairs,
    )

    print(
        "Dev positive pairs:",
        dev_dataset.num_positive_pairs,
    )

    random_pair_auprc_baseline = (
            dev_dataset.num_positive_pairs
            / dev_dataset.num_candidate_pairs
    )

    print(
        "Random Pair AUPRC baseline:",
        f"{random_pair_auprc_baseline:.4f}",
    )

    # -----------------------------------------------------
    # Pair pos weight
    # -----------------------------------------------------

    pos_weight_config = (
        training_config[
            "pair_pos_weight"
        ]
    )

    if (
        isinstance(
            pos_weight_config,
            str,
        )
        and
        pos_weight_config.lower()
        == "auto"
    ):
        pair_pos_weight = (
            train_dataset
            .num_negative_pairs
            / train_dataset
            .num_positive_pairs
        )

        weight_mode = (
            "auto"
        )

    else:

        pair_pos_weight = float(
            pos_weight_config
        )

        weight_mode = (
            "manual"
        )

    if pair_pos_weight <= 0:
        raise ValueError(
            "pair_pos_weight "
            "must be > 0."
        )

    print(
        "Pair pos_weight mode:",
        weight_mode,
    )

    print(
        "Pair pos_weight:",
        pair_pos_weight,
    )

    # -----------------------------------------------------
    # DataLoader
    # -----------------------------------------------------

    collator = (
        FeatureECPECCollator(
            hidden_size=(
                hidden_size
            )
        )
    )

    batch_size = int(
        training_config.get(
            "batch_size",
            4,
        )
    )

    (
        train_loader,
        dev_loader,
    ) = build_dataloaders(
        train_dataset=(
            train_dataset
        ),

        dev_dataset=(
            dev_dataset
        ),

        collator=(
            collator
        ),

        batch_size=(
            batch_size
        ),

        num_workers=(
            num_workers
        ),

        device=(
            device
        ),

        seed=(
            seed
        ),
    )

    print()
    print(
        "===== DataLoader ====="
    )

    print(
        "Batch size:",
        batch_size,
    )

    print(
        "Train batches:",
        len(
            train_loader
        ),
    )

    print(
        "Dev batches:",
        len(
            dev_loader
        ),
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = (
        FeatureECPECBaseModel(
            pair_decision_mode=model_config.get("pair_decision_mode", "fixed"),
            null_hidden=int(model_config.get("null_hidden", 128)),
            null_residual_hidden=int(model_config.get("null_residual_hidden", 128)),
            null_residual_scale=float(model_config.get("null_residual_scale", 0.5)),
            null_center_init=float(model_config.get("null_center_init", 0.0)),
            hierarchical_center_init=float(model_config.get("hierarchical_center_init", 0.0)),
            hierarchical_dialogue_hidden=int(model_config.get("hierarchical_dialogue_hidden", 128)),
            hierarchical_dialogue_scale=float(model_config.get("hierarchical_dialogue_scale", 0.5)),
            hierarchical_row_hidden=int(model_config.get("hierarchical_row_hidden", 128)),
            hierarchical_row_scale=float(model_config.get("hierarchical_row_scale", 0.5)),

            input_dim=(
                hidden_size
            ),

            dialogue_hidden=int(
                model_config.get(
                    "dialogue_hidden",
                    256,
                )
            ),

            speaker_embedding_dim=int(
                model_config.get(
                    "speaker_embedding_dim",
                    32,
                )
            ),

            position_embedding_dim=int(
                model_config.get(
                    "position_embedding_dim",
                    32,
                )
            ),

            max_speakers=int(
                model_config.get(
                    "max_speakers",
                    20,
                )
            ),

            max_positions=int(
                model_config.get(
                    "max_positions",
                    64,
                )
            ),

            dialogue_layers=int(
                model_config.get(
                    "dialogue_layers",
                    2,
                )
            ),

            dialogue_heads=int(
                model_config.get(
                    "dialogue_heads",
                    4,
                )
            ),

            dialogue_ffn=int(
                model_config.get(
                    "dialogue_ffn",
                    1024,
                )
            ),

            auxiliary_hidden=int(
                model_config.get(
                    "auxiliary_hidden",
                    128,
                )
            ),

            pair_hidden=int(
                model_config.get(
                    "pair_hidden",
                    256,
                )
            ),

            distance_embedding_dim=int(
                model_config.get(
                    "distance_embedding_dim",
                    32,
                )
            ),

            speaker_relation_embedding_dim=int(
                model_config.get(
                    "speaker_relation_embedding_dim",
                    16,
                )
            ),

            max_relative_distance=int(
                model_config.get(
                    "max_relative_distance",
                    15,
                )
            ),

            dropout=float(
                model_config.get(
                    "dropout",
                    0.1,
                )
            ),
        )
        .to(
            device
        )
    )

    (
        total_parameters,
        trainable_parameters,
    ) = count_parameters(
        model
    )

    print()
    print(
        "===== Model ====="
    )

    print(
        "Model type:",
        "Frozen-Feature ECPEC",
    )

    print(
        "RoBERTa in training model:",
        False,
    )

    print(
        "Total parameters:",
        f"{total_parameters:,}",
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}",
    )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    print("Pair decision mode:", model.pair_decision_mode)
    print("Emotion threshold: 0.5 | Cause threshold: 0.5")
    print("Adaptive reference parameters:", model.adaptive_reference_parameters)
    print("NULL Reference parameters:", model.null_reference_parameters)
    if model.pair_decision_mode == "centered_null_reference":
        print("Centered NULL parameters:", model.centered_null_parameters)
        print("Global center parameters: 1")
        print("Residual head parameters:", model.centered_null_parameters - 1)
        print("Global Center Logit:", model.null_global_center.detach().item())
        print("Global Center Probability:", model.null_global_center.detach().sigmoid().item())
    if model.pair_decision_mode == "hierarchical_boundary":
        print("Hierarchical Boundary parameters:", model.hierarchical_boundary_parameters)
        print("Global center params: 1")
        print("Dialogue offset head params:", sum(p.numel() for p in model.dialogue_offset_head.parameters()))
        print("Row residual head params:", sum(p.numel() for p in model.row_residual_head.parameters()))
    null_loss_options = {
        "null_rank_weight": float(training_config.get("null_rank_weight", 0.2)),
        "null_margin": float(training_config.get("null_margin", 0.2)),
        "null_hard_negative_k": int(training_config.get("null_hard_negative_k", 3)),
    } if model.pair_decision_mode in ("null_reference", "centered_null_reference", "hierarchical_boundary") else {}

    # Scorer-side margin ranking (sharpens the borderline mass;
    # applies to every decision mode).
    null_loss_options.update({
        "pair_rank_weight": float(training_config.get("pair_rank_weight", 0.0)),
        "pair_rank_margin": float(training_config.get("pair_rank_margin", 0.2)),
        "pair_rank_hard_negative_k": int(training_config.get("pair_rank_hard_negative_k", 3)),
    })

    if model.pair_decision_mode == "hierarchical_boundary":
        null_loss_options.update({
            "hierarchical_dialogue_reg_weight": float(training_config.get("hierarchical_dialogue_reg_weight", 0.005)),
            "hierarchical_row_reg_weight": float(training_config.get("hierarchical_row_reg_weight", 0.01)),
        })
    if model.pair_decision_mode == "centered_null_reference":
        null_loss_options["null_residual_reg_weight"] = float(training_config.get("null_residual_reg_weight", 0.01))
    if model.pair_decision_mode in ("null_reference", "centered_null_reference", "hierarchical_boundary"):
        print("Pair decision: raw_pair_logit > dynamic NULL reference")
        print("No manually tuned global Pair threshold")
        print("NULL ranking options:", null_loss_options)
        print("Checkpoint selection: Dev Relative Pair-F1")
    if model.pair_decision_mode == "adaptive_reference":
        print("Pair decision rule: adjusted_pair_logits > 0 (probability > 0.5)")

    if null_loss_options.get("pair_rank_weight", 0.0) > 0:
        print(
            "Pair ranking options:",
            {
                key: value
                for key, value
                in null_loss_options.items()
                if key.startswith("pair_rank")
            },
        )

    module_lr = float(
        training_config.get(
            "module_lr",
            1e-4,
        )
    )

    weight_decay = float(
        training_config.get(
            "weight_decay",
            0.01,
        )
    )

    optimizer = AdamW(
        optimizer_parameters(model),
        lr=module_lr,
        weight_decay=weight_decay,
    )

    # -----------------------------------------------------
    # Linear LR warmup over the first warmup_epochs epochs.
    #
    # The early-epoch score-distribution overshoot (e.g. the Cause
    # recall collapse around epoch 2-3) is a major contributor to the
    # Dev P/R/F1 swings; warmup damps those large initial steps.
    # -----------------------------------------------------

    warmup_epochs = int(
        training_config.get(
            "warmup_epochs",
            0,
        )
    )

    base_lrs = [
        float(group["lr"])
        for group in optimizer.param_groups
    ]

    print()
    print(
        "===== Optimizer ====="
    )

    print(
        "Optimizer: AdamW"
    )

    print(
        "Module LR:",
        module_lr,
    )

    print(
        "Warmup epochs:",
        warmup_epochs,
    )

    print(
        "Weight decay:",
        weight_decay,
    )

    print(
        "Optimized parameters:",
        f"{trainable_parameters:,}",
    )

    # -----------------------------------------------------
    # Training parameters
    # -----------------------------------------------------

    epochs = int(
        training_config.get(
            "epochs",
            20,
        )
    )

    lambda_emotion = float(
        training_config.get(
            "lambda_emotion",
            0.2,
        )
    )

    lambda_cause = float(
        training_config.get(
            "lambda_cause",
            0.4,
        )
    )

    max_grad_norm = float(
        training_config.get(
            "max_grad_norm",
            1.0,
        )
    )

    patience_limit = int(
        training_config.get(
            "early_stop_patience",
            4,
        )
    )

    # Sliding window over the last N raw Dev Pair-F1 values used for
    # checkpoint selection. A single-epoch F1 on a ~110-dialogue Dev
    # set is noisy; selecting on its smoothed version avoids picking
    # (and early-stopping on) noise spikes.
    f1_smooth_window = max(
        1,
        int(
            training_config.get(
                "f1_smooth_window",
                1,
            )
        ),
    )

    threshold = (
        0.0 if model.pair_decision_mode in ("null_reference", "centered_null_reference", "hierarchical_boundary") else
        0.5 if model.pair_decision_mode == "adaptive_reference" else float(
            evaluation_config.get("pair_threshold", evaluation_config.get("threshold", 0.66))
        )
    )

    row_normalize = bool(
        evaluation_config.get(
            "row_normalize",
            False,
        )
    )

    checkpoint_path = (
        resolve_path(
            training_config.get(
                "checkpoint_path",
                (
                    "checkpoints/"
                    "base_feature_best.pt"
                ),
            )
        )
    )

    checkpoint_path = adaptive_artifact_path(checkpoint_path, model.pair_decision_mode)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = (
        checkpoint_path.parent
        / "training_history_feature.json"
    )

    run_config_path = (
        checkpoint_path.parent
        / "run_config_feature.yaml"
    )

    history_path = adaptive_artifact_path(history_path, model.pair_decision_mode)
    run_config_path = adaptive_artifact_path(run_config_path, model.pair_decision_mode)

    with open(
        run_config_path,
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            config,
            f,
            allow_unicode=True,
            sort_keys=False,
        )

    print()
    print(
        "===== Loss ====="
    )

    print(
        "lambda_emotion:",
        lambda_emotion,
    )

    print(
        "lambda_cause:",
        lambda_cause,
    )

    print(
        "pair_pos_weight:",
        pair_pos_weight,
    )

    print()
    print(
        "===== Evaluation ====="
    )

    print(
        "Pair relative-logit comparison point:" if model.pair_decision_mode in ("null_reference", "centered_null_reference", "hierarchical_boundary") else "Pair decision threshold:",
        threshold,
    )

    print(
        "Pair row normalization:",
        row_normalize,
    )

    print(
        "Pair AUPRC:",
        "enabled",
    )

    print(
        "AUPRC calculation:",
        "global over all valid Dev pairs",
    )

    print()
    print(
        "===== Training Setup ====="
    )

    print(
        "Epochs:",
        epochs,
    )

    print(
        "Early stopping patience:",
        patience_limit,
    )

    print(
        "Gradient clipping:",
        max_grad_norm,
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    best_pair_f1 = -1.0
    best_epoch = 0
    patience = 0

    recent_dev_f1 = []
    history = []

    # EMA/SWA-style parameter averaging: Dev evaluation and the saved
    # best checkpoint use the averaged weights, which damps the
    # epoch-to-epoch weight churn that shows up as Dev P/R/F1 jitter.
    ema_decay = float(
        training_config.get(
            "ema_decay",
            0.0,
        )
    )

    # Start averaging only after the warm-up/ramp-up phase, so that
    # early bad epochs do not contaminate the averaged weights.
    ema_start_epoch = int(
        training_config.get(
            "ema_start_epoch",
            1,
        )
    )

    ema_state = None

    print()
    print(
        "=" * 44
    )

    print(
        "ECPEC Frozen-Feature Training"
    )

    print(
        "=" * 44
    )

    print(
        "Selection metric: "
        "Dev Pair-F1"
    )

    print(
        "Pair relative-logit comparison point:" if model.pair_decision_mode in ("null_reference", "centered_null_reference", "hierarchical_boundary") else "Pair threshold:",
        threshold,
    )

    print(
        "Diagnostic metric:",
        "Dev Pair AUPRC"
    )

    print(
        "F1 smoothing window:",
        f1_smooth_window,
    )

    print(
        "EMA decay:",
        ema_decay,
    )

    print(
        "EMA start epoch:",
        ema_start_epoch,
    )

    print(
        "Early stopping patience:",
        patience_limit,
    )

    print()

    for epoch in range(
        1,
        epochs + 1,
    ):

        # -------------------------------------------------
        # Linear LR warmup over the first warmup_epochs
        # -------------------------------------------------

        if warmup_epochs > 0 and epoch <= warmup_epochs:
            scale = max(
                epoch / warmup_epochs,
                1e-4,
            )

            for group, base_lr in zip(
                optimizer.param_groups,
                base_lrs,
            ):
                group["lr"] = (
                    base_lr
                    * scale
                )

        train_result = (
            train_one_epoch(
                model=model,
                null_loss_options=null_loss_options,

                loader=(
                    train_loader
                ),

                optimizer=(
                    optimizer
                ),

                device=(
                    device
                ),

                pair_pos_weight=(
                    pair_pos_weight
                ),

                lambda_emotion=(
                    lambda_emotion
                ),

                lambda_cause=(
                    lambda_cause
                ),

                threshold=(
                    threshold
                ),

                max_grad_norm=(
                    max_grad_norm
                ),

                epoch=(
                    epoch
                ),

                total_epochs=(
                    epochs
                ),

                row_normalize=(
                    row_normalize
                ),
            )
        )

        # ---------------------------------------------
        # Update EMA weights after the training epoch;
        # Dev evaluation then uses the averaged weights.
        # ---------------------------------------------

        if (
            ema_decay > 0
            and epoch >= ema_start_epoch
        ):
            ema_state = update_ema_state(
                model,
                ema_state,
                ema_decay,
            )

        with ema_weights_context(
            model,
            ema_state if ema_decay > 0 else None,
        ):
            dev_result = (
                evaluate(
                    model=model,
                    null_loss_options=null_loss_options,

                    loader=(
                        dev_loader
                    ),

                    device=(
                        device
                    ),

                    pair_pos_weight=(
                        pair_pos_weight
                    ),

                    lambda_emotion=(
                        lambda_emotion
                    ),

                    lambda_cause=(
                        lambda_cause
                    ),

                    threshold=(
                        threshold
                    ),

                    description=(
                        "Dev"
                    ),

                    row_normalize=(
                        row_normalize
                    ),
                )
            )

        print_result(
            epoch=(
                epoch
            ),

            total_epochs=(
                epochs
            ),

            train_result=(
                train_result
            ),

            dev_result=(
                dev_result
            ),
        )

        history.append(
            make_history_entry(
                epoch=(
                    epoch
                ),

                train_result=(
                    train_result
                ),

                dev_result=(
                    dev_result
                ),
            )
        )

        # Save history after every epoch
        save_history(
            history,
            history_path,
        )

        current_pair_f1 = float(
            dev_result[
                "metrics"
            ][
                "pair"
            ][
                "f1"
            ]
        )

        current_pair_auprc = float(
            dev_result[
                "pair_auprc"
            ]
        )

        # ---------------------------------------------
        # Smoothed selection score: checkpoint / early
        # stopping decisions are made on the mean of the
        # last f1_smooth_window raw Dev Pair-F1 values.
        # ---------------------------------------------

        recent_dev_f1.append(
            current_pair_f1
        )

        recent_dev_f1 = (
            recent_dev_f1[
                -f1_smooth_window:
            ]
        )

        smoothed_pair_f1 = (
            sum(recent_dev_f1)
            / len(recent_dev_f1)
        )

        if (
            smoothed_pair_f1
            > best_pair_f1
        ):
            best_pair_f1 = (
                smoothed_pair_f1
            )

            best_epoch = (
                epoch
            )

            patience = 0

            save_checkpoint(
                path=(
                    checkpoint_path
                ),

                model=(
                    model
                ),

                optimizer=(
                    optimizer
                ),

                epoch=(
                    epoch
                ),

                best_pair_f1=(
                    best_pair_f1
                ),

                threshold=(
                    threshold
                ),

                config=(
                    config
                ),

                pair_auprc=(
                    current_pair_auprc
                ),
                dev_result=dev_result,

                state_dict=(
                    ema_state_dict(
                        model,
                        ema_state,
                    )
                    if (
                        ema_decay > 0
                        and ema_state is not None
                    )
                    else None
                ),
            )

            print()
            print(
                "New best Dev Pair-F1 "
                f"(smoothed@{f1_smooth_window}): "
                f"{best_pair_f1:.4f} "
                f"(raw {current_pair_f1:.4f})"
            )

            print(
                "Corresponding Pair AUPRC: "
                f"{current_pair_auprc:.4f}"
            )

            print(
                "Checkpoint saved to:"
            )

            print(
                checkpoint_path
            )

        else:

            patience += 1

            print()
            print(
                "No improvement. "
                f"Patience: "
                f"{patience}/"
                f"{patience_limit}"
            )

        print(
            "=" * 44
        )

        if (
            model.pair_decision_mode not in ("centered_null_reference", "hierarchical_boundary")
            and patience >= patience_limit
        ):
            print()
            print(
                "Early stopping triggered."
            )

            break

    # -----------------------------------------------------
    # Load best checkpoint
    # -----------------------------------------------------

    print()
    print(
        "===== Load Best Model ====="
    )

    checkpoint = (
        load_checkpoint(
            path=(
                checkpoint_path
            ),

            model=(
                model
            ),

            device=(
                device
            ),
        )
    )

    print(
        "Best epoch:",
        checkpoint[
            "epoch"
        ],
    )

    print(
        "Best Dev Pair-F1:",
        f"{checkpoint['best_pair_f1']:.4f}",
    )

    if (
        "pair_auprc"
        in checkpoint
    ):
        print(
            "Best checkpoint Pair AUPRC:",
            f"{checkpoint['pair_auprc']:.4f}",
        )

    # -----------------------------------------------------
    # Final Dev evaluation
    # -----------------------------------------------------

    best_dev_result = (
        evaluate(
            null_loss_options=null_loss_options,
            model=model,

            loader=(
                dev_loader
            ),

            device=(
                device
            ),

            pair_pos_weight=(
                pair_pos_weight
            ),

            lambda_emotion=(
                lambda_emotion
            ),

            lambda_cause=(
                lambda_cause
            ),

            threshold=(
                threshold
            ),

            description=(
                "Best Dev"
            ),
        )
    )

    dev_pair = (
        best_dev_result[
            "metrics"
        ][
            "pair"
        ]
    )

    dev_emotion = (
        best_dev_result[
            "metrics"
        ][
            "emotion"
        ]
    )

    dev_cause = (
        best_dev_result[
            "metrics"
        ][
            "cause"
        ]
    )

    print()
    print(
        "=" * 52
    )

    print(
        "Best Frozen-Feature Dev Result"
    )

    print(
        "=" * 52
    )

    print(
        "Emotion | "
        f"P={dev_emotion['precision']:.4f} "
        f"R={dev_emotion['recall']:.4f} "
        f"F1={dev_emotion['f1']:.4f}"
    )

    print(
        "Cause   | "
        f"P={dev_cause['precision']:.4f} "
        f"R={dev_cause['recall']:.4f} "
        f"F1={dev_cause['f1']:.4f}"
    )

    print(
        "Pair    | "
        f"P={dev_pair['precision']:.4f} "
        f"R={dev_pair['recall']:.4f} "
        f"F1={dev_pair['f1']:.4f} "
        f"#Pred={dev_pair['predicted_positive']} "
        f"#Gold={dev_pair['gold_positive']}"
    )

    if model.pair_decision_mode == "hierarchical_boundary":
        print(f"Best Raw AUPRC | {best_dev_result['raw_pair_auprc']:.4f}")
        print(f"Best Relative AUPRC | {best_dev_result['relative_pair_auprc']:.4f}")

    print(
        "Pair AUPRC | "
        f"{best_dev_result['pair_auprc']:.4f}"
    )

    print()
    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}",
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        f"Best Pair-F1 (smoothed@{f1_smooth_window}):",
        f"{best_pair_f1:.4f}",
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    print(
        "History:",
        history_path,
    )

    print(
        "Run config:",
        run_config_path,
    )

    print(
        "=" * 52
    )


# =========================================================
# Main
# =========================================================

def main():

    args = parse_args()

    config_path = resolve_path(
        args.config
    )

    config = load_config(
        config_path
    )

    validate_config(
        config
    )

    # -----------------------------------------------------
    # Logging
    # -----------------------------------------------------

    logging_config = (
        config.get(
            "logging",
            {},
        )
    )

    default_log_dir = (
        Path("logs")
        / config_path.stem
    )

    log_dir = resolve_path(
        logging_config.get(
            "log_dir",
            str(
                default_log_dir
            ),
        )
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_filename = (
        logging_config.get(
            "log_file",
            "train_feature.log",
        )
    )

    log_path = (
        log_dir
        / log_filename
    )

    log_path = adaptive_artifact_path(
        log_path, config.get("model", {}).get("pair_decision_mode", "fixed"),
    )

    original_stdout = (
        sys.stdout
    )

    log_handle = open(
        log_path,
        "a",
        encoding="utf-8",
        buffering=1,
    )

    sys.stdout = TeeStdout(
        terminal=(
            original_stdout
        ),

        log_file=(
            log_handle
        ),
    )

    try:

        print()
        print(
            "#" * 52
        )

        print(
            "New Frozen-Feature ECPEC training run"
        )

        print(
            "Config:",
            config_path,
        )

        print(
            "Log:",
            log_path,
        )

        print(
            "#" * 52
        )

        run_training(
            config=(
                config
            ),

            config_path=(
                config_path
            ),
        )

    except KeyboardInterrupt:

        print()
        print(
            "Training interrupted by user."
        )

        raise

    except Exception:

        print()
        print(
            "Training failed."
        )

        print(
            traceback.format_exc()
        )

        raise

    finally:

        sys.stdout = (
            original_stdout
        )

        log_handle.flush()
        log_handle.close()


if __name__ == "__main__":
    main()
