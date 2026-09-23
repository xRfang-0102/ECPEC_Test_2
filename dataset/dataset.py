import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class ECFDataset(Dataset):
    """
    Dataset loader for the ECF dataset.

    Expected JSON structure
    -----------------------
    [
        {
            "conversation_ID": 1,
            "conversation": [
                {
                    "utterance_ID": 1,
                    "speaker": "...",
                    "emotion": "...",
                    "text": "..."
                },
                ...
            ],
            "emotion-cause_pairs": [
                ["3_surprise", "1_some cause text"],
                ...
            ]
        },
        ...
    ]

    Internal pair convention
    ------------------------
    pair = (emotion_index, cause_index)

    All utterance indices returned by this class are zero-based.

    Returned sample
    ---------------
    {
        "dialogue_id": str,
        "utterance_ids": LongTensor [N],
        "utterances": List[str],
        "speaker_ids": LongTensor [N],
        "emotion_labels": FloatTensor [N],
        "cause_labels": FloatTensor [N],
        "pair_labels": FloatTensor [N, N],
        "pairs": List[(emotion_idx, cause_idx)],
        "num_utterances": int
    }
    """

    def __init__(
        self,
        json_path: str,
        max_dialogue_length: Optional[int] = None,
    ):
        super().__init__()

        self.json_path = Path(json_path)
        self.max_dialogue_length = max_dialogue_length

        if not self.json_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.json_path}"
            )

        if (
            self.max_dialogue_length is not None
            and self.max_dialogue_length <= 0
        ):
            raise ValueError(
                "max_dialogue_length must be positive or None."
            )

        raw_data = self._load_json()

        self.samples = []

        for index, dialogue in enumerate(raw_data):
            sample = self._parse_dialogue(
                dialogue=dialogue,
                fallback_id=index,
            )

            if sample is not None:
                self.samples.append(sample)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid conversations found in {self.json_path}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        return self.samples[index]

    def _load_json(self) -> List[Dict]:
        """
        ECF currently uses a top-level JSON list.

        We intentionally do not use:
            list(raw_data.values())

        because that can silently mix metadata or unrelated fields
        into the dialogue list.
        """

        with open(
            self.json_path,
            "r",
            encoding="utf-8",
        ) as file:
            raw_data = json.load(file)

        if not isinstance(raw_data, list):
            if isinstance(raw_data, dict):
                raise ValueError(
                    "Current ECF loader expects the JSON root "
                    "to be a list of conversations, but received "
                    f"a dictionary with keys: {list(raw_data.keys())}"
                )

            raise TypeError(
                "Current ECF loader expects the JSON root "
                f"to be a list, but got {type(raw_data)}."
            )

        for index, item in enumerate(raw_data):
            if not isinstance(item, dict):
                raise TypeError(
                    f"Conversation {index} must be a dictionary, "
                    f"but got {type(item)}."
                )

        return raw_data

    def _parse_dialogue(
        self,
        dialogue: Dict,
        fallback_id: int,
    ) -> Optional[Dict]:

        dialogue_id = dialogue.get(
            "conversation_ID",
            fallback_id,
        )

        conversation = dialogue.get(
            "conversation"
        )

        if conversation is None:
            raise KeyError(
                f"'conversation' field not found "
                f"in dialogue {dialogue_id}."
            )

        if not isinstance(conversation, list):
            raise TypeError(
                f"'conversation' must be a list "
                f"in dialogue {dialogue_id}."
            )

        if len(conversation) == 0:
            return None

        utterance_ids: List[int] = []
        utterances: List[str] = []
        speakers: List[str] = []

        for position, utterance in enumerate(conversation):

            if not isinstance(utterance, dict):
                raise TypeError(
                    f"Utterance {position} in dialogue "
                    f"{dialogue_id} must be a dictionary."
                )

            if "utterance_ID" not in utterance:
                raise KeyError(
                    f"'utterance_ID' missing from utterance "
                    f"{position} in dialogue {dialogue_id}."
                )

            if "text" not in utterance:
                raise KeyError(
                    f"'text' missing from utterance "
                    f"{position} in dialogue {dialogue_id}."
                )

            try:
                utterance_id = int(
                    utterance["utterance_ID"]
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid utterance_ID "
                    f"{utterance.get('utterance_ID')} "
                    f"in dialogue {dialogue_id}."
                ) from error

            text = str(
                utterance["text"]
            ).strip()

            speaker = str(
                utterance.get(
                    "speaker",
                    "unknown",
                )
            ).strip()

            if speaker == "":
                speaker = "unknown"

            utterance_ids.append(
                utterance_id
            )

            utterances.append(
                text
            )

            speakers.append(
                speaker
            )

        self._validate_utterance_ids(
            utterance_ids=utterance_ids,
            dialogue_id=dialogue_id,
        )

        id_to_index = {
            utterance_id: index
            for index, utterance_id
            in enumerate(utterance_ids)
        }

        speaker_ids = self._encode_speakers(
            speakers
        )

        raw_pairs = dialogue.get(
            "emotion-cause_pairs",
            [],
        )

        if raw_pairs is None:
            raw_pairs = []

        if not isinstance(raw_pairs, list):
            raise TypeError(
                f"'emotion-cause_pairs' must be a list "
                f"in dialogue {dialogue_id}."
            )

        pairs: List[Tuple[int, int]] = []

        for pair in raw_pairs:

            emotion_id, cause_id = (
                self._parse_ecf_pair(
                    pair=pair,
                    dialogue_id=dialogue_id,
                )
            )

            if emotion_id not in id_to_index:
                raise ValueError(
                    f"Emotion utterance_ID {emotion_id} "
                    f"does not exist in dialogue {dialogue_id}."
                )

            if cause_id not in id_to_index:
                raise ValueError(
                    f"Cause utterance_ID {cause_id} "
                    f"does not exist in dialogue {dialogue_id}."
                )

            emotion_index = id_to_index[
                emotion_id
            ]

            cause_index = id_to_index[
                cause_id
            ]

            pairs.append(
                (
                    emotion_index,
                    cause_index,
                )
            )

        # Remove duplicate gold pairs while preserving order
        pairs = list(
            dict.fromkeys(pairs)
        )

        (
            utterance_ids,
            utterances,
            speaker_ids,
            pairs,
        ) = self._truncate_dialogue(
            utterance_ids=utterance_ids,
            utterances=utterances,
            speaker_ids=speaker_ids,
            pairs=pairs,
        )

        num_utterances = len(
            utterances
        )

        emotion_labels = torch.zeros(
            num_utterances,
            dtype=torch.float32,
        )

        cause_labels = torch.zeros(
            num_utterances,
            dtype=torch.float32,
        )

        pair_labels = torch.zeros(
            (
                num_utterances,
                num_utterances,
            ),
            dtype=torch.float32,
        )

        for emotion_index, cause_index in pairs:

            emotion_labels[
                emotion_index
            ] = 1.0

            cause_labels[
                cause_index
            ] = 1.0

            pair_labels[
                emotion_index,
                cause_index,
            ] = 1.0

        self._validate_labels(
            pairs=pairs,
            emotion_labels=emotion_labels,
            cause_labels=cause_labels,
            pair_labels=pair_labels,
            dialogue_id=dialogue_id,
        )

        return {
            "dialogue_id": str(
                dialogue_id
            ),

            "utterance_ids": torch.tensor(
                utterance_ids,
                dtype=torch.long,
            ),

            "utterances": utterances,

            "speaker_ids": torch.tensor(
                speaker_ids,
                dtype=torch.long,
            ),

            "emotion_labels": (
                emotion_labels
            ),

            "cause_labels": (
                cause_labels
            ),

            "pair_labels": (
                pair_labels
            ),

            "pairs": pairs,

            "num_utterances": (
                num_utterances
            ),
        }

    @staticmethod
    def _validate_utterance_ids(
        utterance_ids: List[int],
        dialogue_id,
    ) -> None:

        if len(utterance_ids) != len(
            set(utterance_ids)
        ):
            raise ValueError(
                f"Duplicate utterance_ID found "
                f"in dialogue {dialogue_id}: "
                f"{utterance_ids}"
            )

    @staticmethod
    def _parse_ecf_pair(
        pair,
        dialogue_id,
    ) -> Tuple[int, int]:
        """
        Example ECF pair:

            [
                "3_surprise",
                "1_I realize I am totally naked ."
            ]

        Extracts:
            emotion utterance_ID = 3
            cause utterance_ID   = 1
        """

        if not isinstance(
            pair,
            (list, tuple),
        ):
            raise TypeError(
                f"Invalid pair in dialogue "
                f"{dialogue_id}: {pair}"
            )

        if len(pair) < 2:
            raise ValueError(
                f"Pair must contain at least "
                f"two elements in dialogue "
                f"{dialogue_id}: {pair}"
            )

        emotion_value = str(
            pair[0]
        ).strip()

        cause_value = str(
            pair[1]
        ).strip()

        try:
            emotion_id_text = (
                emotion_value.split(
                    "_",
                    1,
                )[0]
            )

            cause_id_text = (
                cause_value.split(
                    "_",
                    1,
                )[0]
            )

            emotion_id = int(
                emotion_id_text
            )

            cause_id = int(
                cause_id_text
            )

        except (ValueError, IndexError) as error:
            raise ValueError(
                f"Cannot parse emotion-cause pair "
                f"in dialogue {dialogue_id}: {pair}"
            ) from error

        return (
            emotion_id,
            cause_id,
        )

    @staticmethod
    def _encode_speakers(
        speakers: List[str],
    ) -> List[int]:
        """
        Dialogue-local speaker encoding.

        Example:

            Ross    -> 0
            Rachel  -> 1
            Ross    -> 0
            Monica  -> 2

        Returns:

            [0, 1, 0, 2]
        """

        speaker_map = {}
        speaker_ids = []

        for speaker in speakers:

            if speaker not in speaker_map:
                speaker_map[
                    speaker
                ] = len(
                    speaker_map
                )

            speaker_ids.append(
                speaker_map[
                    speaker
                ]
            )

        return speaker_ids

    def _truncate_dialogue(
        self,
        utterance_ids: List[int],
        utterances: List[str],
        speaker_ids: List[int],
        pairs: List[Tuple[int, int]],
    ):
        """
        Truncates the dialogue only when
        max_dialogue_length is explicitly set.

        Gold pairs outside the retained region
        are removed.
        """

        if self.max_dialogue_length is None:
            return (
                utterance_ids,
                utterances,
                speaker_ids,
                pairs,
            )

        max_len = self.max_dialogue_length

        if len(utterances) <= max_len:
            return (
                utterance_ids,
                utterances,
                speaker_ids,
                pairs,
            )

        utterance_ids = (
            utterance_ids[:max_len]
        )

        utterances = (
            utterances[:max_len]
        )

        speaker_ids = (
            speaker_ids[:max_len]
        )

        pairs = [
            (
                emotion_index,
                cause_index,
            )
            for (
                emotion_index,
                cause_index,
            )
            in pairs
            if (
                emotion_index < max_len
                and cause_index < max_len
            )
        ]

        return (
            utterance_ids,
            utterances,
            speaker_ids,
            pairs,
        )

    @staticmethod
    def _validate_labels(
        pairs: List[Tuple[int, int]],
        emotion_labels: torch.Tensor,
        cause_labels: torch.Tensor,
        pair_labels: torch.Tensor,
        dialogue_id,
    ) -> None:
        """
        Internal sanity checks.

        These checks make silent label bugs
        fail immediately during dataset loading.
        """

        expected_positive_pairs = len(
            pairs
        )

        actual_positive_pairs = int(
            pair_labels.sum().item()
        )

        if (
            expected_positive_pairs
            != actual_positive_pairs
        ):
            raise RuntimeError(
                f"Pair label mismatch in dialogue "
                f"{dialogue_id}: "
                f"{expected_positive_pairs} gold pairs "
                f"but {actual_positive_pairs} positive "
                f"entries in pair_labels."
            )

        emotion_indices = {
            emotion_index
            for emotion_index, _
            in pairs
        }

        cause_indices = {
            cause_index
            for _, cause_index
            in pairs
        }

        actual_emotion_count = int(
            emotion_labels.sum().item()
        )

        actual_cause_count = int(
            cause_labels.sum().item()
        )

        if (
            len(emotion_indices)
            != actual_emotion_count
        ):
            raise RuntimeError(
                f"Emotion label mismatch in "
                f"dialogue {dialogue_id}."
            )

        if (
            len(cause_indices)
            != actual_cause_count
        ):
            raise RuntimeError(
                f"Cause label mismatch in "
                f"dialogue {dialogue_id}."
            )


def build_ecf_dataset(
    json_path: str,
    max_dialogue_length: Optional[int] = None,
) -> ECFDataset:

    return ECFDataset(
        json_path=json_path,
        max_dialogue_length=(
            max_dialogue_length
        ),
    )