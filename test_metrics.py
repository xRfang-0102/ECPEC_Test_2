import math
import torch

from metrics import (
    compute_binary_metrics,
    compute_ecpec_metrics,
    ECPECMetricAccumulator,
    flatten_metrics,
    format_metrics,
)


def assert_close(
    value,
    expected,
    tol=1e-6,
):
    assert math.isclose(
        value,
        expected,
        rel_tol=tol,
        abs_tol=tol,
    ), (
        f"Expected {expected}, "
        f"got {value}"
    )


def main():

    # ==================================================
    # 1. Utterance mask
    #
    # Dialogue 1: 3 valid utterances
    # Dialogue 2: 2 valid utterances
    #
    # Total valid utterances = 5
    # ==================================================

    utterance_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )

    # ==================================================
    # 2. Pair mask
    #
    # Dialogue 1: 3^2 = 9
    # Dialogue 2: 2^2 = 4
    #
    # Total valid pairs = 13
    # ==================================================

    pair_mask = (
        utterance_mask.unsqueeze(2)
        & utterance_mask.unsqueeze(1)
    )

    print(
        "Valid utterances:",
        int(utterance_mask.sum().item()),
    )

    print(
        "Valid pairs:",
        int(pair_mask.sum().item()),
    )

    assert (
        utterance_mask.sum().item()
        == 5
    )

    assert (
        pair_mask.sum().item()
        == 13
    )

    # ==================================================
    # 3. Emotion task
    #
    # Valid predictions at threshold=0.5:
    #
    # Pred : [1, 0, 0, 1, 1]
    # Gold : [1, 0, 1, 0, 1]
    #
    # TP = 2
    # FP = 1
    # FN = 1
    # TN = 1
    #
    # Precision = 2/3
    # Recall    = 2/3
    # F1        = 2/3
    # ==================================================

    emotion_logits = torch.tensor(
        [
            [2.0, -2.0, -1.0, 100.0],
            [1.0, 2.0, 100.0, 100.0],
        ]
    )

    emotion_labels = torch.tensor(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 0],
        ],
        dtype=torch.float32,
    )

    # ==================================================
    # 4. Cause task
    #
    # Valid predictions:
    #
    # Pred : [0, 1, 1, 1, 0]
    # Gold : [0, 1, 0, 1, 0]
    #
    # TP = 2
    # FP = 1
    # FN = 0
    # TN = 2
    #
    # Precision = 2/3
    # Recall    = 1.0
    # F1        = 0.8
    # ==================================================

    cause_logits = torch.tensor(
        [
            [-2.0, 2.0, 2.0, 100.0],
            [2.0, -2.0, 100.0, 100.0],
        ]
    )

    cause_labels = torch.tensor(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=torch.float32,
    )

    # ==================================================
    # 5. Pair task
    #
    # There are 13 valid pairs.
    #
    # Expected:
    #
    # TP = 3
    # FP = 2
    # FN = 1
    # TN = 7
    #
    # Precision = 3 / 5 = 0.6
    # Recall    = 3 / 4 = 0.75
    # F1        = 2/3
    #
    # #Pred = 5
    # #Gold = 4
    # ==================================================

    pair_labels = torch.zeros(
        2,
        4,
        4,
        dtype=torch.float32,
    )

    pair_logits = torch.full(
        (2, 4, 4),
        -2.0,
        dtype=torch.float32,
    )

    # --------------------------------------------------
    # Dialogue 1 valid 3x3 pairs
    #
    # Flattened gold:
    # [1,1,0,
    #  0,1,0,
    #  0,0,0]
    #
    # Flattened prediction:
    # [1,0,1,
    #  0,1,0,
    #  0,0,0]
    # --------------------------------------------------

    pair_labels[0, :3, :3] = torch.tensor(
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=torch.float32,
    )

    pair_logits[0, :3, :3] = torch.tensor(
        [
            [2.0, -2.0, 2.0],
            [-2.0, 2.0, -2.0],
            [-2.0, -2.0, -2.0],
        ]
    )

    # --------------------------------------------------
    # Dialogue 2 valid 2x2 pairs
    #
    # Gold:
    # [1,0,
    #  0,0]
    #
    # Prediction:
    # [1,1,
    #  0,0]
    # --------------------------------------------------

    pair_labels[1, :2, :2] = torch.tensor(
        [
            [1, 0],
            [0, 0],
        ],
        dtype=torch.float32,
    )

    pair_logits[1, :2, :2] = torch.tensor(
        [
            [2.0, 2.0],
            [-2.0, -2.0],
        ]
    )

    # Deliberately assign huge positive logits to
    # padding positions.
    #
    # Correct masking must completely ignore them.
    pair_logits[
        ~pair_mask
    ] = 100.0

    # ==================================================
    # 6. Test individual binary metric
    # ==================================================

    print()
    print("===== Emotion Metric Check =====")

    emotion_metrics = (
        compute_binary_metrics(
            logits=emotion_logits,
            labels=emotion_labels,
            mask=utterance_mask,
            threshold=0.5,
        )
    )

    print(emotion_metrics)

    assert emotion_metrics["tp"] == 2
    assert emotion_metrics["fp"] == 1
    assert emotion_metrics["fn"] == 1
    assert emotion_metrics["tn"] == 1

    assert (
        emotion_metrics[
            "predicted_positive"
        ]
        == 3
    )

    assert (
        emotion_metrics[
            "gold_positive"
        ]
        == 3
    )

    assert_close(
        emotion_metrics["precision"],
        2 / 3,
    )

    assert_close(
        emotion_metrics["recall"],
        2 / 3,
    )

    assert_close(
        emotion_metrics["f1"],
        2 / 3,
    )

    # ==================================================
    # 7. Test full ECPEC metrics
    # ==================================================

    outputs = {
        "emotion_logits": emotion_logits,
        "cause_logits": cause_logits,
        "pair_logits": pair_logits,
    }

    metrics = compute_ecpec_metrics(
        outputs=outputs,
        emotion_labels=emotion_labels,
        cause_labels=cause_labels,
        pair_labels=pair_labels,
        utterance_mask=utterance_mask,
        pair_mask=pair_mask,
        threshold=0.5,
    )

    # ==================================================
    # 8. Emotion assertions
    # ==================================================

    emotion = metrics["emotion"]

    assert emotion["tp"] == 2
    assert emotion["fp"] == 1
    assert emotion["fn"] == 1
    assert emotion["tn"] == 1

    assert_close(
        emotion["precision"],
        2 / 3,
    )

    assert_close(
        emotion["recall"],
        2 / 3,
    )

    assert_close(
        emotion["f1"],
        2 / 3,
    )

    # ==================================================
    # 9. Cause assertions
    # ==================================================

    cause = metrics["cause"]

    assert cause["tp"] == 2
    assert cause["fp"] == 1
    assert cause["fn"] == 0
    assert cause["tn"] == 2

    assert_close(
        cause["precision"],
        2 / 3,
    )

    assert_close(
        cause["recall"],
        1.0,
    )

    assert_close(
        cause["f1"],
        0.8,
    )

    # ==================================================
    # 10. Pair assertions
    # ==================================================

    pair = metrics["pair"]

    assert pair["tp"] == 3
    assert pair["fp"] == 2
    assert pair["fn"] == 1
    assert pair["tn"] == 7

    assert (
        pair["predicted_positive"]
        == 5
    )

    assert (
        pair["gold_positive"]
        == 4
    )

    assert_close(
        pair["precision"],
        0.6,
    )

    assert_close(
        pair["recall"],
        0.75,
    )

    assert_close(
        pair["f1"],
        2 / 3,
    )

    print()
    print("===== Full Metric Check =====")

    print(
        format_metrics(metrics)
    )

    # ==================================================
    # 11. Test epoch accumulator
    #
    # Feed dialogue 1 and dialogue 2 separately.
    # Final results MUST equal the metrics calculated
    # from the whole batch at once.
    # ==================================================

    accumulator = (
        ECPECMetricAccumulator(
            threshold=0.5
        )
    )

    # Dialogue 1
    accumulator.update(
        outputs={
            "emotion_logits":
                emotion_logits[0:1],

            "cause_logits":
                cause_logits[0:1],

            "pair_logits":
                pair_logits[0:1],
        },

        emotion_labels=(
            emotion_labels[0:1]
        ),

        cause_labels=(
            cause_labels[0:1]
        ),

        pair_labels=(
            pair_labels[0:1]
        ),

        utterance_mask=(
            utterance_mask[0:1]
        ),

        pair_mask=(
            pair_mask[0:1]
        ),
    )

    # Dialogue 2
    accumulator.update(
        outputs={
            "emotion_logits":
                emotion_logits[1:2],

            "cause_logits":
                cause_logits[1:2],

            "pair_logits":
                pair_logits[1:2],
        },

        emotion_labels=(
            emotion_labels[1:2]
        ),

        cause_labels=(
            cause_labels[1:2]
        ),

        pair_labels=(
            pair_labels[1:2]
        ),

        utterance_mask=(
            utterance_mask[1:2]
        ),

        pair_mask=(
            pair_mask[1:2]
        ),
    )

    accumulated_metrics = (
        accumulator.compute()
    )

    print()
    print("===== Accumulator Check =====")

    print(
        format_metrics(
            accumulated_metrics
        )
    )

    # --------------------------------------------------
    # Accumulated counts must equal full-batch counts
    # --------------------------------------------------

    for task in [
        "emotion",
        "cause",
        "pair",
    ]:

        for key in [
            "tp",
            "fp",
            "fn",
            "tn",
            "predicted_positive",
            "gold_positive",
        ]:

            assert (
                accumulated_metrics[
                    task
                ][key]
                ==
                metrics[
                    task
                ][key]
            )

        for key in [
            "precision",
            "recall",
            "f1",
            "accuracy",
        ]:

            assert_close(
                accumulated_metrics[
                    task
                ][key],
                metrics[
                    task
                ][key],
            )

    # ==================================================
    # 12. Flatten test
    # ==================================================

    flattened = flatten_metrics(
        metrics
    )

    print()
    print("===== Flatten Check =====")

    print(
        "pair_precision:",
        flattened[
            "pair_precision"
        ],
    )

    print(
        "pair_recall:",
        flattened[
            "pair_recall"
        ],
    )

    print(
        "pair_f1:",
        flattened[
            "pair_f1"
        ],
    )

    assert_close(
        flattened["pair_precision"],
        0.6,
    )

    assert_close(
        flattened["pair_recall"],
        0.75,
    )

    assert_close(
        flattened["pair_f1"],
        2 / 3,
    )

    # ==================================================
    # 13. Mask safety check
    #
    # Padding logits were deliberately set to +100.
    # If masking is wrong, predicted positives would
    # become much larger than 5.
    # ==================================================

    print()
    print("===== Mask Safety Check =====")

    print(
        "Pair predicted positives:",
        pair["predicted_positive"],
    )

    assert (
        pair["predicted_positive"]
        == 5
    )

    print(
        "Padding positions correctly ignored."
    )

    # ==================================================
    # 14. Final
    # ==================================================

    print()
    print(
        "Metrics test passed."
    )


if __name__ == "__main__":
    main()