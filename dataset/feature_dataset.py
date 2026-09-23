from pathlib import Path

import torch
from torch.utils.data import Dataset


class FeatureECFDataset(Dataset):
    """
    ECF dataset based on pre-extracted frozen RoBERTa features.

    Expected .pt structure:

    {
        "metadata": {
            "format_version": 1,
            "dataset": "ECF",
            "split": "train/dev/test",
            "hidden_size": 768,
            ...
        },

        "dialogues": [
            {
                "dialogue_id": ...,
                "utterance_ids": [...],
                "utterance_features": Tensor[N, 768],
                "speaker_ids": Tensor[N],
                "emotion_labels": Tensor[N],
                "cause_labels": Tensor[N],
                "pair_labels": Tensor[N, N],
                "pairs": [...],
                "num_utterances": N,
            },
            ...
        ]
    }
    """

    def __init__(
        self,
        feature_path,
        max_dialogue_length=None,
        expected_hidden_size=768,
        validate=True,
    ):
        super().__init__()

        self.feature_path = Path(
            feature_path
        ).resolve()

        self.max_dialogue_length = (
            max_dialogue_length
        )

        self.expected_hidden_size = (
            expected_hidden_size
        )

        # -------------------------------------------------
        # Check file
        # -------------------------------------------------

        if not self.feature_path.exists():
            raise FileNotFoundError(
                f"Feature file not found: "
                f"{self.feature_path}"
            )

        # -------------------------------------------------
        # Load .pt
        #
        # Explicit weights_only=False because this file is
        # a structured dataset payload, not only model weights.
        # -------------------------------------------------

        try:
            payload = torch.load(
                self.feature_path,
                map_location="cpu",
                weights_only=False,
            )

        except TypeError:
            # Compatibility with older PyTorch versions
            payload = torch.load(
                self.feature_path,
                map_location="cpu",
            )

        if not isinstance(
            payload,
            dict,
        ):
            raise TypeError(
                "Feature file must contain "
                "a dictionary payload."
            )

        if "metadata" not in payload:
            raise KeyError(
                "Missing 'metadata' in "
                "feature file."
            )

        if "dialogues" not in payload:
            raise KeyError(
                "Missing 'dialogues' in "
                "feature file."
            )

        self.metadata = (
            payload["metadata"]
        )

        raw_dialogues = (
            payload["dialogues"]
        )

        if not isinstance(
            raw_dialogues,
            list,
        ):
            raise TypeError(
                "'dialogues' must be a list."
            )

        # -------------------------------------------------
        # Metadata
        # -------------------------------------------------

        self.dataset_name = (
            self.metadata.get(
                "dataset",
                "unknown",
            )
        )

        self.split = (
            self.metadata.get(
                "split",
                "unknown",
            )
        )

        self.hidden_size = int(
            self.metadata.get(
                "hidden_size",
                expected_hidden_size,
            )
        )

        if (
            expected_hidden_size
            is not None
            and
            self.hidden_size
            != expected_hidden_size
        ):
            raise ValueError(
                "Unexpected feature hidden size: "
                f"{self.hidden_size}. "
                f"Expected: "
                f"{expected_hidden_size}"
            )

        # -------------------------------------------------
        # Prepare dialogues
        # -------------------------------------------------

        self.dialogues = []

        for index, dialogue in enumerate(
            raw_dialogues
        ):
            processed = (
                self._prepare_dialogue(
                    dialogue=dialogue,
                    index=index,
                )
            )

            self.dialogues.append(
                processed
            )

        # Release original payload reference
        del payload
        del raw_dialogues

        # -------------------------------------------------
        # Full validation
        # -------------------------------------------------

        if validate:
            self._validate_dataset()

    # =====================================================
    # Dialogue preparation
    # =====================================================

    def _prepare_dialogue(
        self,
        dialogue,
        index,
    ):
        if not isinstance(
            dialogue,
            dict,
        ):
            raise TypeError(
                f"Dialogue {index} "
                f"must be a dictionary."
            )

        required_keys = [
            "dialogue_id",
            "utterance_ids",
            "utterance_features",
            "speaker_ids",
            "emotion_labels",
            "cause_labels",
            "pair_labels",
            "num_utterances",
        ]

        for key in required_keys:
            if key not in dialogue:
                raise KeyError(
                    f"Dialogue {index} "
                    f"missing key: {key}"
                )

        # -------------------------------------------------
        # Basic fields
        # -------------------------------------------------

        dialogue_id = (
            dialogue["dialogue_id"]
        )

        utterance_ids = list(
            dialogue["utterance_ids"]
        )

        num_utterances = int(
            dialogue["num_utterances"]
        )

        # -------------------------------------------------
        # Tensors
        # -------------------------------------------------

        utterance_features = (
            self._as_tensor(
                dialogue[
                    "utterance_features"
                ],
                dtype=torch.float32,
            )
        )

        speaker_ids = (
            self._as_tensor(
                dialogue[
                    "speaker_ids"
                ],
                dtype=torch.long,
            )
        )

        emotion_labels = (
            self._as_tensor(
                dialogue[
                    "emotion_labels"
                ],
                dtype=torch.float32,
            )
        )

        cause_labels = (
            self._as_tensor(
                dialogue[
                    "cause_labels"
                ],
                dtype=torch.float32,
            )
        )

        pair_labels = (
            self._as_tensor(
                dialogue[
                    "pair_labels"
                ],
                dtype=torch.float32,
            )
        )

        pairs = dialogue.get(
            "pairs",
            [],
        )

        # -------------------------------------------------
        # Optional dialogue truncation
        # -------------------------------------------------

        if (
            self.max_dialogue_length
            is not None
        ):
            max_length = int(
                self.max_dialogue_length
            )

            if max_length <= 0:
                raise ValueError(
                    "max_dialogue_length "
                    "must be > 0."
                )

            if num_utterances > max_length:

                num_utterances = (
                    max_length
                )

                utterance_ids = (
                    utterance_ids[
                        :num_utterances
                    ]
                )

                utterance_features = (
                    utterance_features[
                        :num_utterances
                    ]
                )

                speaker_ids = (
                    speaker_ids[
                        :num_utterances
                    ]
                )

                emotion_labels = (
                    emotion_labels[
                        :num_utterances
                    ]
                )

                cause_labels = (
                    cause_labels[
                        :num_utterances
                    ]
                )

                pair_labels = (
                    pair_labels[
                        :num_utterances,
                        :num_utterances,
                    ]
                )

                pairs = (
                    self._filter_pairs(
                        pairs,
                        num_utterances,
                    )
                )

        # -------------------------------------------------
        # Contiguous CPU tensors
        # -------------------------------------------------

        utterance_features = (
            utterance_features
            .contiguous()
            .cpu()
        )

        speaker_ids = (
            speaker_ids
            .contiguous()
            .cpu()
        )

        emotion_labels = (
            emotion_labels
            .contiguous()
            .cpu()
        )

        cause_labels = (
            cause_labels
            .contiguous()
            .cpu()
        )

        pair_labels = (
            pair_labels
            .contiguous()
            .cpu()
        )

        return {
            "dialogue_id":
                dialogue_id,

            "utterance_ids":
                utterance_ids,

            "utterance_features":
                utterance_features,

            "speaker_ids":
                speaker_ids,

            "emotion_labels":
                emotion_labels,

            "cause_labels":
                cause_labels,

            "pair_labels":
                pair_labels,

            "pairs":
                pairs,

            "num_utterances":
                num_utterances,
        }

    # =====================================================
    # Tensor helper
    # =====================================================

    @staticmethod
    def _as_tensor(
        value,
        dtype,
    ):
        if torch.is_tensor(
            value
        ):
            return (
                value
                .detach()
                .to(dtype=dtype)
            )

        return torch.tensor(
            value,
            dtype=dtype,
        )

    # =====================================================
    # Pair filtering for optional truncation
    # =====================================================

    @staticmethod
    def _filter_pairs(
        pairs,
        num_utterances,
    ):
        """
        pairs are not required during training because pair_labels
        is the authoritative pair target.

        This function only keeps compatible zero-based pairs when
        possible. Unknown pair formats are preserved safely.
        """

        if pairs is None:
            return []

        filtered = []

        for pair in pairs:

            if (
                isinstance(
                    pair,
                    (list, tuple),
                )
                and
                len(pair) >= 2
            ):
                try:
                    emotion_index = int(
                        pair[0]
                    )

                    cause_index = int(
                        pair[1]
                    )

                    if (
                        0
                        <= emotion_index
                        < num_utterances
                        and
                        0
                        <= cause_index
                        < num_utterances
                    ):
                        filtered.append(
                            pair
                        )

                except (
                    TypeError,
                    ValueError,
                ):
                    filtered.append(
                        pair
                    )

            else:
                filtered.append(
                    pair
                )

        return filtered

    # =====================================================
    # Validation
    # =====================================================

    def _validate_dataset(
        self,
    ):
        if len(
            self.dialogues
        ) == 0:
            raise ValueError(
                "Feature dataset is empty."
            )

        total_utterances = 0
        total_pairs = 0
        total_positive_pairs = 0

        seen_dialogue_ids = set()

        for index, item in enumerate(
            self.dialogues
        ):
            n = int(
                item[
                    "num_utterances"
                ]
            )

            if n <= 0:
                raise ValueError(
                    f"Dialogue {index} "
                    "contains no utterances."
                )

            dialogue_id = (
                item[
                    "dialogue_id"
                ]
            )

            # Do not enforce uniqueness too aggressively
            # across unusual datasets, but ECF should be unique.
            if dialogue_id in (
                seen_dialogue_ids
            ):
                raise ValueError(
                    "Duplicate dialogue_id: "
                    f"{dialogue_id}"
                )

            seen_dialogue_ids.add(
                dialogue_id
            )

            # ---------------------------------------------
            # Features
            # ---------------------------------------------

            features = (
                item[
                    "utterance_features"
                ]
            )

            if features.ndim != 2:
                raise ValueError(
                    f"Dialogue {index}: "
                    "utterance_features must "
                    "have shape [N, H]."
                )

            if features.shape[0] != n:
                raise ValueError(
                    f"Dialogue {index}: "
                    "feature/utterance count "
                    "mismatch."
                )

            if (
                features.shape[1]
                != self.hidden_size
            ):
                raise ValueError(
                    f"Dialogue {index}: "
                    "feature hidden size "
                    f"{features.shape[1]} "
                    "does not match "
                    f"{self.hidden_size}."
                )

            if not torch.isfinite(
                features
            ).all():
                raise ValueError(
                    f"Dialogue {index}: "
                    "non-finite feature "
                    "values detected."
                )

            # ---------------------------------------------
            # Utterance IDs
            # ---------------------------------------------

            if len(
                item[
                    "utterance_ids"
                ]
            ) != n:
                raise ValueError(
                    f"Dialogue {index}: "
                    "utterance_ids length "
                    "mismatch."
                )

            # ---------------------------------------------
            # Speaker
            # ---------------------------------------------

            if tuple(
                item[
                    "speaker_ids"
                ].shape
            ) != (n,):
                raise ValueError(
                    f"Dialogue {index}: "
                    "speaker_ids must "
                    f"have shape [{n}]."
                )

            # ---------------------------------------------
            # Emotion
            # ---------------------------------------------

            if tuple(
                item[
                    "emotion_labels"
                ].shape
            ) != (n,):
                raise ValueError(
                    f"Dialogue {index}: "
                    "emotion_labels must "
                    f"have shape [{n}]."
                )

            # ---------------------------------------------
            # Cause
            # ---------------------------------------------

            if tuple(
                item[
                    "cause_labels"
                ].shape
            ) != (n,):
                raise ValueError(
                    f"Dialogue {index}: "
                    "cause_labels must "
                    f"have shape [{n}]."
                )

            # ---------------------------------------------
            # Pair
            # ---------------------------------------------

            if tuple(
                item[
                    "pair_labels"
                ].shape
            ) != (n, n):
                raise ValueError(
                    f"Dialogue {index}: "
                    "pair_labels must "
                    f"have shape "
                    f"[{n}, {n}]."
                )

            # ---------------------------------------------
            # Label range
            # ---------------------------------------------

            emotion_labels = (
                item[
                    "emotion_labels"
                ]
            )

            cause_labels = (
                item[
                    "cause_labels"
                ]
            )

            pair_labels = (
                item[
                    "pair_labels"
                ]
            )

            if (
                (
                    emotion_labels < 0
                ).any()
                or
                (
                    emotion_labels > 1
                ).any()
            ):
                raise ValueError(
                    f"Dialogue {index}: "
                    "invalid emotion labels."
                )

            if (
                (
                    cause_labels < 0
                ).any()
                or
                (
                    cause_labels > 1
                ).any()
            ):
                raise ValueError(
                    f"Dialogue {index}: "
                    "invalid cause labels."
                )

            if (
                (
                    pair_labels < 0
                ).any()
                or
                (
                    pair_labels > 1
                ).any()
            ):
                raise ValueError(
                    f"Dialogue {index}: "
                    "invalid pair labels."
                )

            total_utterances += n

            total_pairs += (
                n * n
            )

            total_positive_pairs += int(
                pair_labels
                .sum()
                .item()
            )

        # -------------------------------------------------
        # Statistics
        # -------------------------------------------------

        self.num_dialogues = len(
            self.dialogues
        )

        self.num_utterances = int(
            total_utterances
        )

        self.num_candidate_pairs = int(
            total_pairs
        )

        self.num_positive_pairs = int(
            total_positive_pairs
        )

        self.num_negative_pairs = int(
            self.num_candidate_pairs
            - self.num_positive_pairs
        )

    # =====================================================
    # PyTorch Dataset interface
    # =====================================================

    def __len__(
        self,
    ):
        return len(
            self.dialogues
        )

    def __getitem__(
        self,
        index,
    ):
        return self.dialogues[
            index
        ]

    # =====================================================
    # Convenience
    # =====================================================

    def get_metadata(
        self,
    ):
        return dict(
            self.metadata
        )

    def summary(
        self,
    ):
        positive_ratio = (
            self.num_positive_pairs
            / self.num_candidate_pairs
            if self.num_candidate_pairs > 0
            else 0.0
        )

        negative_positive_ratio = (
            self.num_negative_pairs
            / self.num_positive_pairs
            if self.num_positive_pairs > 0
            else float("inf")
        )

        return {
            "feature_path":
                str(
                    self.feature_path
                ),

            "dataset":
                self.dataset_name,

            "split":
                self.split,

            "hidden_size":
                self.hidden_size,

            "num_dialogues":
                self.num_dialogues,

            "num_utterances":
                self.num_utterances,

            "num_candidate_pairs":
                self.num_candidate_pairs,

            "num_positive_pairs":
                self.num_positive_pairs,

            "num_negative_pairs":
                self.num_negative_pairs,

            "positive_pair_ratio":
                positive_ratio,

            "negative_positive_ratio":
                negative_positive_ratio,
        }


# =========================================================
# Standalone check
# =========================================================

if __name__ == "__main__":

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    feature_root = (
        project_root
        / "features"
        / "ECF"
    )

    for split in [
        "train",
        "dev",
        "test",
    ]:

        path = (
            feature_root
            / f"{split}_roberta.pt"
        )

        dataset = (
            FeatureECFDataset(
                feature_path=path,
                max_dialogue_length=40,
                expected_hidden_size=768,
                validate=True,
            )
        )

        print()
        print(
            "=" * 50
        )

        print(
            split.upper()
        )

        print(
            "=" * 50
        )

        summary = (
            dataset.summary()
        )

        for key, value in (
            summary.items()
        ):
            print(
                f"{key}: {value}"
            )

        first = dataset[0]

        print()
        print(
            "First dialogue:"
        )

        print(
            "dialogue_id:",
            first[
                "dialogue_id"
            ],
        )

        print(
            "num_utterances:",
            first[
                "num_utterances"
            ],
        )

        print(
            "utterance_features:",
            first[
                "utterance_features"
            ].shape,
        )

        print(
            "speaker_ids:",
            first[
                "speaker_ids"
            ].shape,
        )

        print(
            "emotion_labels:",
            first[
                "emotion_labels"
            ].shape,
        )

        print(
            "cause_labels:",
            first[
                "cause_labels"
            ].shape,
        )

        print(
            "pair_labels:",
            first[
                "pair_labels"
            ].shape,
        )