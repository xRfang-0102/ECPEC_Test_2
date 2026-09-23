from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from dataset.feature_dataset import FeatureECFDataset
except ImportError:
    from feature_dataset import FeatureECFDataset


class FeatureECPECCollator:
    """
    Collator for pre-extracted frozen RoBERTa features.

    Input:
        A list of dialogue dictionaries produced by FeatureECFDataset.

    Output:
        utterance_features : [B, N, H]
        speaker_ids        : [B, N]
        position_ids       : [B, N]
        utterance_mask     : [B, N]

        emotion_labels     : [B, N]
        cause_labels       : [B, N]

        pair_labels        : [B, N, N]
        pair_mask          : [B, N, N]

        num_utterances     : [B]

        dialogue_ids       : list
        utterance_ids      : list
        pairs              : list
    """

    def __init__(
        self,
        hidden_size=768,
    ):
        self.hidden_size = int(
            hidden_size
        )

        if self.hidden_size <= 0:
            raise ValueError(
                "hidden_size must be > 0."
            )

    # =====================================================
    # Main collate
    # =====================================================

    def __call__(
        self,
        batch,
    ):
        if batch is None:
            raise ValueError(
                "Batch cannot be None."
            )

        if len(batch) == 0:
            raise ValueError(
                "Batch cannot be empty."
            )

        batch_size = len(
            batch
        )

        # -------------------------------------------------
        # Dialogue lengths
        # -------------------------------------------------

        dialogue_lengths = []

        for index, item in enumerate(
            batch
        ):
            if "num_utterances" not in item:
                raise KeyError(
                    f"Batch item {index} "
                    "missing num_utterances."
                )

            n = int(
                item[
                    "num_utterances"
                ]
            )

            if n <= 0:
                raise ValueError(
                    f"Batch item {index} "
                    "contains no utterances."
                )

            dialogue_lengths.append(
                n
            )

        max_num_utterances = max(
            dialogue_lengths
        )

        # -------------------------------------------------
        # Allocate tensors
        # -------------------------------------------------

        utterance_features = torch.zeros(
            (
                batch_size,
                max_num_utterances,
                self.hidden_size,
            ),
            dtype=torch.float32,
        )

        speaker_ids = torch.zeros(
            (
                batch_size,
                max_num_utterances,
            ),
            dtype=torch.long,
        )

        position_ids = torch.zeros(
            (
                batch_size,
                max_num_utterances,
            ),
            dtype=torch.long,
        )

        utterance_mask = torch.zeros(
            (
                batch_size,
                max_num_utterances,
            ),
            dtype=torch.bool,
        )

        emotion_labels = torch.zeros(
            (
                batch_size,
                max_num_utterances,
            ),
            dtype=torch.float32,
        )

        cause_labels = torch.zeros(
            (
                batch_size,
                max_num_utterances,
            ),
            dtype=torch.float32,
        )

        pair_labels = torch.zeros(
            (
                batch_size,
                max_num_utterances,
                max_num_utterances,
            ),
            dtype=torch.float32,
        )

        # -------------------------------------------------
        # Metadata
        # -------------------------------------------------

        dialogue_ids = []
        utterance_ids = []
        pairs = []

        # -------------------------------------------------
        # Fill batch
        # -------------------------------------------------

        for batch_index, item in enumerate(
            batch
        ):
            n = int(
                item[
                    "num_utterances"
                ]
            )

            # ---------------------------------------------
            # Validate required keys
            # ---------------------------------------------

            required_keys = [
                "dialogue_id",
                "utterance_ids",
                "utterance_features",
                "speaker_ids",
                "emotion_labels",
                "cause_labels",
                "pair_labels",
            ]

            for key in required_keys:
                if key not in item:
                    raise KeyError(
                        f"Batch item "
                        f"{batch_index} "
                        f"missing key: {key}"
                    )

            # ---------------------------------------------
            # Features
            # ---------------------------------------------

            features = item[
                "utterance_features"
            ]

            if not torch.is_tensor(
                features
            ):
                features = torch.tensor(
                    features,
                    dtype=torch.float32,
                )

            features = features.to(
                dtype=torch.float32
            )

            if features.ndim != 2:
                raise ValueError(
                    f"Batch item "
                    f"{batch_index}: "
                    "utterance_features "
                    "must have shape [N, H]."
                )

            if features.shape[0] != n:
                raise ValueError(
                    f"Batch item "
                    f"{batch_index}: "
                    "feature count mismatch."
                )

            if (
                features.shape[1]
                != self.hidden_size
            ):
                raise ValueError(
                    f"Batch item "
                    f"{batch_index}: "
                    "hidden size mismatch: "
                    f"{features.shape[1]} "
                    f"!= {self.hidden_size}"
                )

            if not torch.isfinite(
                features
            ).all():
                raise ValueError(
                    f"Batch item "
                    f"{batch_index}: "
                    "non-finite feature "
                    "values detected."
                )

            # ---------------------------------------------
            # Speaker IDs
            # ---------------------------------------------

            item_speaker_ids = item[
                "speaker_ids"
            ]

            if not torch.is_tensor(
                item_speaker_ids
            ):
                item_speaker_ids = (
                    torch.tensor(
                        item_speaker_ids,
                        dtype=torch.long,
                    )
                )

            item_speaker_ids = (
                item_speaker_ids.to(
                    dtype=torch.long
                )
            )

            if tuple(
                item_speaker_ids.shape
            ) != (n,):
                raise ValueError(
                    f"Batch item "
                    f"{batch_index}: "
                    "speaker_ids shape "
                    "mismatch."
                )

            # ---------------------------------------------
            # Emotion labels
            # ---------------------------------------------

            item_emotion_labels = item[
                "emotion_labels"
            ]

            if not torch.is_tensor(
                item_emotion_labels
            ):
                item_emotion_labels = (
                    torch.tensor(
                        item_emotion_labels,
                        dtype=torch.float32,
                    )
                )

            item_emotion_labels = (
                item_emotion_labels.to(
                    dtype=torch.float32
                )
            )

            if tuple(
                item_emotion_labels.shape
            ) != (n,):
                raise ValueError(
                    f"Batch item "
                    f"{batch_index}: "
                    "emotion_labels "
                    "shape mismatch."
                )

            # ---------------------------------------------
            # Cause labels
            # ---------------------------------------------

            item_cause_labels = item[
                "cause_labels"
            ]

            if not torch.is_tensor(
                item_cause_labels
            ):
                item_cause_labels = (
                    torch.tensor(
                        item_cause_labels,
                        dtype=torch.float32,
                    )
                )

            item_cause_labels = (
                item_cause_labels.to(
                    dtype=torch.float32
                )
            )

            if tuple(
                item_cause_labels.shape
            ) != (n,):
                raise ValueError(
                    f"Batch item "
                    f"{batch_index}: "
                    "cause_labels "
                    "shape mismatch."
                )

            # ---------------------------------------------
            # Pair labels
            # ---------------------------------------------

            item_pair_labels = item[
                "pair_labels"
            ]

            if not torch.is_tensor(
                item_pair_labels
            ):
                item_pair_labels = (
                    torch.tensor(
                        item_pair_labels,
                        dtype=torch.float32,
                    )
                )

            item_pair_labels = (
                item_pair_labels.to(
                    dtype=torch.float32
                )
            )

            if tuple(
                item_pair_labels.shape
            ) != (n, n):
                raise ValueError(
                    f"Batch item "
                    f"{batch_index}: "
                    "pair_labels "
                    f"must have shape "
                    f"[{n}, {n}]."
                )

            # ---------------------------------------------
            # Copy to padded tensors
            # ---------------------------------------------

            utterance_features[
                batch_index,
                :n,
                :
            ] = features

            speaker_ids[
                batch_index,
                :n
            ] = item_speaker_ids

            position_ids[
                batch_index,
                :n
            ] = torch.arange(
                n,
                dtype=torch.long,
            )

            utterance_mask[
                batch_index,
                :n
            ] = True

            emotion_labels[
                batch_index,
                :n
            ] = item_emotion_labels

            cause_labels[
                batch_index,
                :n
            ] = item_cause_labels

            pair_labels[
                batch_index,
                :n,
                :n
            ] = item_pair_labels

            # ---------------------------------------------
            # Metadata
            # ---------------------------------------------

            dialogue_ids.append(
                item[
                    "dialogue_id"
                ]
            )

            utterance_ids.append(
                list(
                    item[
                        "utterance_ids"
                    ]
                )
            )

            pairs.append(
                item.get(
                    "pairs",
                    [],
                )
            )

        # -------------------------------------------------
        # Pair mask
        #
        # Valid pair iff both emotion candidate and
        # cause candidate are valid utterances.
        #
        # [B,N,1] & [B,1,N] -> [B,N,N]
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

        # -------------------------------------------------
        # Number of utterances
        # -------------------------------------------------

        num_utterances = torch.tensor(
            dialogue_lengths,
            dtype=torch.long,
        )

        # -------------------------------------------------
        # Final validation
        # -------------------------------------------------

        self._validate_output(
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
            emotion_labels=(
                emotion_labels
            ),
            cause_labels=(
                cause_labels
            ),
            pair_labels=(
                pair_labels
            ),
            pair_mask=(
                pair_mask
            ),
            dialogue_lengths=(
                dialogue_lengths
            ),
        )

        return {
            "utterance_features":
                utterance_features,

            "speaker_ids":
                speaker_ids,

            "position_ids":
                position_ids,

            "utterance_mask":
                utterance_mask,

            "emotion_labels":
                emotion_labels,

            "cause_labels":
                cause_labels,

            "pair_labels":
                pair_labels,

            "pair_mask":
                pair_mask,

            "num_utterances":
                num_utterances,

            "dialogue_ids":
                dialogue_ids,

            "utterance_ids":
                utterance_ids,

            "pairs":
                pairs,
        }

    # =====================================================
    # Output validation
    # =====================================================

    def _validate_output(
        self,
        utterance_features,
        speaker_ids,
        position_ids,
        utterance_mask,
        emotion_labels,
        cause_labels,
        pair_labels,
        pair_mask,
        dialogue_lengths,
    ):
        batch_size = len(
            dialogue_lengths
        )

        max_n = max(
            dialogue_lengths
        )

        # -------------------------------------------------
        # Shapes
        # -------------------------------------------------

        if tuple(
            utterance_features.shape
        ) != (
            batch_size,
            max_n,
            self.hidden_size,
        ):
            raise RuntimeError(
                "Invalid "
                "utterance_features shape."
            )

        expected_2d_shape = (
            batch_size,
            max_n,
        )

        for name, tensor in [
            (
                "speaker_ids",
                speaker_ids,
            ),
            (
                "position_ids",
                position_ids,
            ),
            (
                "utterance_mask",
                utterance_mask,
            ),
            (
                "emotion_labels",
                emotion_labels,
            ),
            (
                "cause_labels",
                cause_labels,
            ),
        ]:
            if tuple(
                tensor.shape
            ) != expected_2d_shape:
                raise RuntimeError(
                    f"Invalid {name} shape: "
                    f"{tensor.shape}"
                )

        expected_pair_shape = (
            batch_size,
            max_n,
            max_n,
        )

        if tuple(
            pair_labels.shape
        ) != expected_pair_shape:
            raise RuntimeError(
                "Invalid pair_labels shape."
            )

        if tuple(
            pair_mask.shape
        ) != expected_pair_shape:
            raise RuntimeError(
                "Invalid pair_mask shape."
            )

        # -------------------------------------------------
        # Mask counts
        # -------------------------------------------------

        expected_valid_utterances = sum(
            dialogue_lengths
        )

        actual_valid_utterances = int(
            utterance_mask
            .sum()
            .item()
        )

        if (
            actual_valid_utterances
            != expected_valid_utterances
        ):
            raise RuntimeError(
                "utterance_mask count "
                "mismatch."
            )

        expected_valid_pairs = sum(
            n * n
            for n in dialogue_lengths
        )

        actual_valid_pairs = int(
            pair_mask
            .sum()
            .item()
        )

        if (
            actual_valid_pairs
            != expected_valid_pairs
        ):
            raise RuntimeError(
                "pair_mask count mismatch."
            )

        # -------------------------------------------------
        # Padding features must be zero
        # -------------------------------------------------

        invalid_utterances = (
            ~utterance_mask
        )

        if invalid_utterances.any():

            padded_features = (
                utterance_features[
                    invalid_utterances
                ]
            )

            if not torch.all(
                padded_features == 0
            ):
                raise RuntimeError(
                    "Non-zero padded "
                    "utterance features."
                )

        # -------------------------------------------------
        # Padded pair labels must be zero
        # -------------------------------------------------

        invalid_pairs = (
            ~pair_mask
        )

        if invalid_pairs.any():

            padded_pair_labels = (
                pair_labels[
                    invalid_pairs
                ]
            )

            if not torch.all(
                padded_pair_labels == 0
            ):
                raise RuntimeError(
                    "Non-zero padded "
                    "pair labels."
                )

        # -------------------------------------------------
        # Finite values
        # -------------------------------------------------

        if not torch.isfinite(
            utterance_features
        ).all():
            raise RuntimeError(
                "Non-finite values in "
                "utterance_features."
            )

        if not torch.isfinite(
            emotion_labels
        ).all():
            raise RuntimeError(
                "Non-finite values in "
                "emotion_labels."
            )

        if not torch.isfinite(
            cause_labels
        ).all():
            raise RuntimeError(
                "Non-finite values in "
                "cause_labels."
            )

        if not torch.isfinite(
            pair_labels
        ).all():
            raise RuntimeError(
                "Non-finite values in "
                "pair_labels."
            )


# =========================================================
# Standalone test
# =========================================================

if __name__ == "__main__":

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    feature_path = (
        project_root
        / "features"
        / "ECF"
        / "train_roberta.pt"
    )

    dataset = FeatureECFDataset(
        feature_path=feature_path,
        max_dialogue_length=40,
        expected_hidden_size=768,
        validate=True,
    )

    collator = FeatureECPECCollator(
        hidden_size=768
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    batch = next(
        iter(
            loader
        )
    )

    print()
    print(
        "=" * 60
    )

    print(
        "Feature Collator Test"
    )

    print(
        "=" * 60
    )

    print(
        "Dialogue IDs:",
        batch[
            "dialogue_ids"
        ],
    )

    print(
        "Dialogue lengths:",
        batch[
            "num_utterances"
        ].tolist(),
    )

    print()

    print(
        "utterance_features:",
        batch[
            "utterance_features"
        ].shape,
    )

    print(
        "speaker_ids:",
        batch[
            "speaker_ids"
        ].shape,
    )

    print(
        "position_ids:",
        batch[
            "position_ids"
        ].shape,
    )

    print(
        "utterance_mask:",
        batch[
            "utterance_mask"
        ].shape,
    )

    print(
        "emotion_labels:",
        batch[
            "emotion_labels"
        ].shape,
    )

    print(
        "cause_labels:",
        batch[
            "cause_labels"
        ].shape,
    )

    print(
        "pair_labels:",
        batch[
            "pair_labels"
        ].shape,
    )

    print(
        "pair_mask:",
        batch[
            "pair_mask"
        ].shape,
    )

    print()

    valid_utterances = int(
        batch[
            "utterance_mask"
        ]
        .sum()
        .item()
    )

    valid_pairs = int(
        batch[
            "pair_mask"
        ]
        .sum()
        .item()
    )

    positive_pairs = int(
        (
            batch[
                "pair_labels"
            ]
            *
            batch[
                "pair_mask"
            ].float()
        )
        .sum()
        .item()
    )

    print(
        "Valid utterances:",
        valid_utterances,
    )

    print(
        "Valid pairs:",
        valid_pairs,
    )

    print(
        "Positive pairs:",
        positive_pairs,
    )

    # -----------------------------------------------------
    # Padding checks
    # -----------------------------------------------------

    invalid_utterances = (
        ~batch[
            "utterance_mask"
        ]
    )

    if invalid_utterances.any():

        padded_features = (
            batch[
                "utterance_features"
            ][
                invalid_utterances
            ]
        )

        padding_feature_sum = float(
            padded_features
            .abs()
            .sum()
            .item()
        )

    else:
        padding_feature_sum = 0.0

    invalid_pairs = (
        ~batch[
            "pair_mask"
        ]
    )

    if invalid_pairs.any():

        padding_pair_sum = float(
            batch[
                "pair_labels"
            ][
                invalid_pairs
            ]
            .abs()
            .sum()
            .item()
        )

    else:
        padding_pair_sum = 0.0

    print(
        "Padding feature abs sum:",
        padding_feature_sum,
    )

    print(
        "Padding pair-label abs sum:",
        padding_pair_sum,
    )

    # -----------------------------------------------------
    # Expected first batch check
    # -----------------------------------------------------

    expected_lengths = [
        8,
        3,
        9,
        3,
    ]

    if (
        batch[
            "num_utterances"
        ].tolist()
        == expected_lengths
    ):

        print()
        print(
            "Expected first batch "
            "dialogue lengths: PASS"
        )

    else:

        print()
        print(
            "Expected first batch "
            "dialogue lengths: "
            "DIFFERENT"
        )

        print(
            "Expected:",
            expected_lengths,
        )

    if valid_utterances == 23:
        print(
            "Valid utterance count: PASS"
        )
    else:
        print(
            "Valid utterance count: "
            "DIFFERENT"
        )

    if valid_pairs == 163:
        print(
            "Valid pair count: PASS"
        )
    else:
        print(
            "Valid pair count: "
            "DIFFERENT"
        )

    if positive_pairs == 20:
        print(
            "Positive pair count: PASS"
        )
    else:
        print(
            "Positive pair count: "
            "DIFFERENT"
        )

    print()
    print(
        "=" * 60
    )