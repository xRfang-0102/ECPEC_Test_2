from typing import Dict, List

import torch
from transformers import AutoTokenizer


class ECPECCollator:
    """
    Collate function for dialogue-level ECPEC batches.

    Input:
        List of samples returned by ECFDataset

    Output:
        input_ids:
            [B, N_max, L]

        attention_mask:
            [B, N_max, L]

        speaker_ids:
            [B, N_max]

        position_ids:
            [B, N_max]

        utterance_mask:
            [B, N_max]

        emotion_labels:
            [B, N_max]

        cause_labels:
            [B, N_max]

        pair_labels:
            [B, N_max, N_max]

        pair_mask:
            [B, N_max, N_max]

    where:
        B     = batch size
        N_max = maximum number of utterances in the batch
        L     = max_utterance_length
    """

    def __init__(
        self,
        tokenizer_name: str = "roberta-base",
        max_utterance_length: int = 128,
    ):
        self.tokenizer_name = tokenizer_name
        self.max_utterance_length = max_utterance_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,
            local_files_only=True,
        )

        if self.tokenizer.pad_token_id is None:
            raise ValueError(
                f"Tokenizer {tokenizer_name} has no pad_token_id."
            )

        self.pad_token_id = (
            self.tokenizer.pad_token_id
        )

    def __call__(
        self,
        batch: List[Dict],
    ) -> Dict:

        if len(batch) == 0:
            raise ValueError(
                "Cannot collate an empty batch."
            )

        batch_size = len(batch)

        dialogue_lengths = [
            sample["num_utterances"]
            for sample in batch
        ]

        max_dialogue_length = max(
            dialogue_lengths
        )

        if max_dialogue_length <= 0:
            raise ValueError(
                "Dialogue length must be positive."
            )

        self._validate_batch(
            batch=batch,
        )

        # --------------------------------------------------
        # 1. Flatten real utterances
        # --------------------------------------------------

        flat_utterances = []

        flat_positions = []

        for batch_idx, sample in enumerate(batch):

            for utterance_idx, text in enumerate(
                sample["utterances"]
            ):
                flat_utterances.append(
                    text
                )

                flat_positions.append(
                    (
                        batch_idx,
                        utterance_idx,
                    )
                )

        # --------------------------------------------------
        # 2. Tokenize all real utterances together
        # --------------------------------------------------

        encoded = self.tokenizer(
            flat_utterances,
            padding="max_length",
            truncation=True,
            max_length=self.max_utterance_length,
            return_tensors="pt",
        )

        flat_input_ids = encoded[
            "input_ids"
        ]

        flat_attention_mask = encoded[
            "attention_mask"
        ]

        token_length = (
            flat_input_ids.size(1)
        )

        # --------------------------------------------------
        # 3. Allocate dialogue-level tensors
        # --------------------------------------------------

        input_ids = torch.full(
            (
                batch_size,
                max_dialogue_length,
                token_length,
            ),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )

        attention_mask = torch.zeros(
            (
                batch_size,
                max_dialogue_length,
                token_length,
            ),
            dtype=torch.long,
        )

        speaker_ids = torch.zeros(
            (
                batch_size,
                max_dialogue_length,
            ),
            dtype=torch.long,
        )

        position_ids = torch.zeros(
            (
                batch_size,
                max_dialogue_length,
            ),
            dtype=torch.long,
        )

        utterance_mask = torch.zeros(
            (
                batch_size,
                max_dialogue_length,
            ),
            dtype=torch.bool,
        )

        emotion_labels = torch.zeros(
            (
                batch_size,
                max_dialogue_length,
            ),
            dtype=torch.float32,
        )

        cause_labels = torch.zeros(
            (
                batch_size,
                max_dialogue_length,
            ),
            dtype=torch.float32,
        )

        pair_labels = torch.zeros(
            (
                batch_size,
                max_dialogue_length,
                max_dialogue_length,
            ),
            dtype=torch.float32,
        )

        pair_mask = torch.zeros(
            (
                batch_size,
                max_dialogue_length,
                max_dialogue_length,
            ),
            dtype=torch.bool,
        )

        # --------------------------------------------------
        # 4. Restore flattened tokens to dialogue layout
        # --------------------------------------------------

        for flat_idx, (
            batch_idx,
            utterance_idx,
        ) in enumerate(flat_positions):

            input_ids[
                batch_idx,
                utterance_idx,
            ] = flat_input_ids[
                flat_idx
            ]

            attention_mask[
                batch_idx,
                utterance_idx,
            ] = flat_attention_mask[
                flat_idx
            ]

        # --------------------------------------------------
        # 5. Fill labels / speaker / masks
        # --------------------------------------------------

        for batch_idx, sample in enumerate(batch):

            num_utterances = (
                sample["num_utterances"]
            )

            speaker_ids[
                batch_idx,
                :num_utterances,
            ] = sample[
                "speaker_ids"
            ]

            position_ids[
                batch_idx,
                :num_utterances,
            ] = torch.arange(
                num_utterances,
                dtype=torch.long,
            )

            utterance_mask[
                batch_idx,
                :num_utterances,
            ] = True

            emotion_labels[
                batch_idx,
                :num_utterances,
            ] = sample[
                "emotion_labels"
            ]

            cause_labels[
                batch_idx,
                :num_utterances,
            ] = sample[
                "cause_labels"
            ]

            pair_labels[
                batch_idx,
                :num_utterances,
                :num_utterances,
            ] = sample[
                "pair_labels"
            ]

            pair_mask[
                batch_idx,
                :num_utterances,
                :num_utterances,
            ] = True

        # --------------------------------------------------
        # 6. Final consistency checks
        # --------------------------------------------------

        self._validate_collated_batch(
            batch=batch,
            dialogue_lengths=dialogue_lengths,
            utterance_mask=utterance_mask,
            pair_mask=pair_mask,
            pair_labels=pair_labels,
        )

        return {
            "dialogue_ids": [
                sample["dialogue_id"]
                for sample in batch
            ],

            "utterance_ids": [
                sample["utterance_ids"]
                for sample in batch
            ],

            "dialogue_lengths": torch.tensor(
                dialogue_lengths,
                dtype=torch.long,
            ),

            "input_ids": input_ids,

            "attention_mask": (
                attention_mask
            ),

            "speaker_ids": speaker_ids,

            "position_ids": position_ids,

            "utterance_mask": (
                utterance_mask
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

            "pair_mask": pair_mask,
        }

    @staticmethod
    def _validate_batch(
        batch: List[Dict],
    ) -> None:

        required_keys = {
            "dialogue_id",
            "utterance_ids",
            "utterances",
            "speaker_ids",
            "emotion_labels",
            "cause_labels",
            "pair_labels",
            "num_utterances",
        }

        for index, sample in enumerate(batch):

            missing_keys = (
                required_keys
                - set(sample.keys())
            )

            if missing_keys:
                raise KeyError(
                    f"Sample {index} is missing "
                    f"keys: {missing_keys}"
                )

            num_utterances = (
                sample["num_utterances"]
            )

            if len(
                sample["utterances"]
            ) != num_utterances:
                raise ValueError(
                    f"Sample {index}: "
                    f"utterance count mismatch."
                )

            if (
                sample["speaker_ids"].numel()
                != num_utterances
            ):
                raise ValueError(
                    f"Sample {index}: "
                    f"speaker_ids length mismatch."
                )

            if (
                sample["emotion_labels"].numel()
                != num_utterances
            ):
                raise ValueError(
                    f"Sample {index}: "
                    f"emotion_labels length mismatch."
                )

            if (
                sample["cause_labels"].numel()
                != num_utterances
            ):
                raise ValueError(
                    f"Sample {index}: "
                    f"cause_labels length mismatch."
                )

            expected_pair_shape = (
                num_utterances,
                num_utterances,
            )

            if (
                tuple(
                    sample[
                        "pair_labels"
                    ].shape
                )
                != expected_pair_shape
            ):
                raise ValueError(
                    f"Sample {index}: "
                    f"pair_labels shape is "
                    f"{tuple(sample['pair_labels'].shape)}, "
                    f"expected "
                    f"{expected_pair_shape}."
                )

    @staticmethod
    def _validate_collated_batch(
        batch,
        dialogue_lengths,
        utterance_mask,
        pair_mask,
        pair_labels,
    ) -> None:

        # Number of real utterances
        expected_utterances = sum(
            dialogue_lengths
        )

        actual_utterances = int(
            utterance_mask.sum().item()
        )

        if (
            expected_utterances
            != actual_utterances
        ):
            raise RuntimeError(
                "utterance_mask validation failed: "
                f"expected {expected_utterances}, "
                f"got {actual_utterances}."
            )

        # Every valid dialogue contributes N x N
        # candidate pairs.
        expected_pair_count = sum(
            length * length
            for length in dialogue_lengths
        )

        actual_pair_count = int(
            pair_mask.sum().item()
        )

        if (
            expected_pair_count
            != actual_pair_count
        ):
            raise RuntimeError(
                "pair_mask validation failed: "
                f"expected {expected_pair_count}, "
                f"got {actual_pair_count}."
            )

        # Positive labels must equal the sum of
        # gold pairs from individual samples.
        expected_positive_pairs = sum(
            len(sample["pairs"])
            for sample in batch
        )

        actual_positive_pairs = int(
            pair_labels.sum().item()
        )

        if (
            expected_positive_pairs
            != actual_positive_pairs
        ):
            raise RuntimeError(
                "pair_labels validation failed: "
                f"expected "
                f"{expected_positive_pairs} "
                f"positive pairs, got "
                f"{actual_positive_pairs}."
            )


def build_collator(
    tokenizer_name: str = "roberta-base",
    max_utterance_length: int = 128,
) -> ECPECCollator:

    return ECPECCollator(
        tokenizer_name=tokenizer_name,
        max_utterance_length=(
            max_utterance_length
        ),
    )