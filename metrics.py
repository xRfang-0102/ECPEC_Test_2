import torch


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """
    Safe division.

    Returns 0.0 when denominator is zero.
    """
    if denominator == 0:
        return 0.0

    return numerator / denominator


def compute_binary_metrics_from_counts(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
):
    """
    Compute binary classification metrics from
    accumulated TP / FP / FN / TN.
    """

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    f1 = safe_divide(
        2.0 * precision * recall,
        precision + recall,
    )

    accuracy = safe_divide(
        tp + tn,
        tp + fp + fn + tn,
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,

        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),

        "predicted_positive": int(
            tp + fp
        ),

        "gold_positive": int(
            tp + fn
        ),
    }


def compute_binary_counts(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
):
    """
    Compute TP / FP / FN / TN for one binary task.

    Parameters
    ----------
    logits:
        Arbitrary shape.

    labels:
        Same shape as logits.
        Binary 0 / 1 labels.

    mask:
        Same shape as logits.
        True  = valid position
        False = padding / ignored position

    threshold:
        Probability threshold after sigmoid.
    """

    if logits.shape != labels.shape:
        raise ValueError(
            "logits and labels shape mismatch: "
            f"{tuple(logits.shape)} vs "
            f"{tuple(labels.shape)}."
        )

    if logits.shape != mask.shape:
        raise ValueError(
            "logits and mask shape mismatch: "
            f"{tuple(logits.shape)} vs "
            f"{tuple(mask.shape)}."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    valid_mask = mask.bool()

    valid_logits = logits[
        valid_mask
    ]

    valid_labels = labels[
        valid_mask
    ].bool()

    if valid_logits.numel() == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }

    probabilities = torch.sigmoid(
        valid_logits
    )

    predictions = (
        probabilities >= threshold
    )

    tp = (
        predictions
        & valid_labels
    ).sum().item()

    fp = (
        predictions
        & ~valid_labels
    ).sum().item()

    fn = (
        ~predictions
        & valid_labels
    ).sum().item()

    tn = (
        ~predictions
        & ~valid_labels
    ).sum().item()

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def compute_binary_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
):
    """
    Compute binary classification metrics directly
    from logits / labels / mask.
    """

    counts = compute_binary_counts(
        logits=logits,
        labels=labels,
        mask=mask,
        threshold=threshold,
    )

    return compute_binary_metrics_from_counts(
        tp=counts["tp"],
        fp=counts["fp"],
        fn=counts["fn"],
        tn=counts["tn"],
    )


def compute_ecpec_metrics(
    outputs,
    emotion_labels: torch.Tensor,
    cause_labels: torch.Tensor,
    pair_labels: torch.Tensor,
    utterance_mask: torch.Tensor,
    pair_mask: torch.Tensor,
    threshold: float = 0.5,
):
    """
    Compute Emotion / Cause / Pair metrics
    for a single batch or full dataset tensor.

    outputs:
        {
            "emotion_logits": [B, N],
            "cause_logits":   [B, N],
            "pair_logits":    [B, N, N]
        }
    """

    required_keys = {
        "emotion_logits",
        "cause_logits",
        "pair_logits",
    }

    missing_keys = (
        required_keys
        - set(outputs.keys())
    )

    if missing_keys:
        raise KeyError(
            "Missing outputs: "
            f"{sorted(missing_keys)}"
        )

    emotion_metrics = (
        compute_binary_metrics(
            logits=outputs[
                "emotion_logits"
            ],
            labels=emotion_labels,
            mask=utterance_mask,
            threshold=threshold,
        )
    )

    cause_metrics = (
        compute_binary_metrics(
            logits=outputs[
                "cause_logits"
            ],
            labels=cause_labels,
            mask=utterance_mask,
            threshold=threshold,
        )
    )

    pair_metrics = (
        compute_binary_metrics(
            logits=outputs[
                "pair_logits"
            ],
            labels=pair_labels,
            mask=pair_mask,
            threshold=threshold,
        )
    )

    return {
        "emotion": emotion_metrics,
        "cause": cause_metrics,
        "pair": pair_metrics,
    }


class BinaryMetricAccumulator:
    """
    Accumulate TP / FP / FN / TN over an entire epoch.

    Important:
    We accumulate counts first and compute F1 only
    after all batches are processed.

    Do NOT average batch-level F1 scores.
    """

    def __init__(
        self,
        threshold: float = 0.5,
    ):

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        self.threshold = threshold

        self.reset()

    def reset(self):

        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ):

        counts = compute_binary_counts(
            logits=logits,
            labels=labels,
            mask=mask,
            threshold=self.threshold,
        )

        self.tp += counts["tp"]
        self.fp += counts["fp"]
        self.fn += counts["fn"]
        self.tn += counts["tn"]

    def compute(self):

        return (
            compute_binary_metrics_from_counts(
                tp=self.tp,
                fp=self.fp,
                fn=self.fn,
                tn=self.tn,
            )
        )


class ECPECMetricAccumulator:
    """
    Epoch-level metric accumulator for ECPEC.

    Tracks:
        Emotion classification
        Cause classification
        Pair extraction

    Pair F1 is the primary metric.
    """

    def __init__(
        self,
        threshold: float = 0.5,
    ):

        self.threshold = threshold

        self.emotion = (
            BinaryMetricAccumulator(
                threshold=threshold
            )
        )

        self.cause = (
            BinaryMetricAccumulator(
                threshold=threshold
            )
        )

        self.pair = (
            BinaryMetricAccumulator(
                threshold=threshold
            )
        )

    def reset(self):

        self.emotion.reset()
        self.cause.reset()
        self.pair.reset()

    @torch.no_grad()
    def update(
        self,
        outputs,
        emotion_labels: torch.Tensor,
        cause_labels: torch.Tensor,
        pair_labels: torch.Tensor,
        utterance_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ):

        self.emotion.update(
            logits=outputs[
                "emotion_logits"
            ],
            labels=emotion_labels,
            mask=utterance_mask,
        )

        self.cause.update(
            logits=outputs[
                "cause_logits"
            ],
            labels=cause_labels,
            mask=utterance_mask,
        )

        self.pair.update(
            logits=outputs[
                "pair_logits"
            ],
            labels=pair_labels,
            mask=pair_mask,
        )

    def compute(self):

        return {
            "emotion": (
                self.emotion.compute()
            ),

            "cause": (
                self.cause.compute()
            ),

            "pair": (
                self.pair.compute()
            ),
        }


def flatten_metrics(
    metrics,
):
    """
    Convert nested metrics:

    {
        "emotion": {"precision": ...},
        "cause": {...},
        "pair": {...}
    }

    into:

    {
        "emotion_precision": ...,
        "emotion_recall": ...,
        ...
    }

    Useful for logging and checkpoint selection.
    """

    flattened = {}

    for task_name, task_metrics in (
        metrics.items()
    ):

        for metric_name, value in (
            task_metrics.items()
        ):

            flattened[
                f"{task_name}_{metric_name}"
            ] = value

    return flattened


def format_metrics(
    metrics,
    digits: int = 4,
):
    """
    Return a compact readable summary string.
    """

    emotion = metrics["emotion"]
    cause = metrics["cause"]
    pair = metrics["pair"]

    return (
        f"Emotion "
        f"P={emotion['precision']:.{digits}f} "
        f"R={emotion['recall']:.{digits}f} "
        f"F1={emotion['f1']:.{digits}f} | "
        f"Cause "
        f"P={cause['precision']:.{digits}f} "
        f"R={cause['recall']:.{digits}f} "
        f"F1={cause['f1']:.{digits}f} | "
        f"Pair "
        f"P={pair['precision']:.{digits}f} "
        f"R={pair['recall']:.{digits}f} "
        f"F1={pair['f1']:.{digits}f} "
        f"#Pred={pair['predicted_positive']} "
        f"#Gold={pair['gold_positive']}"
    )