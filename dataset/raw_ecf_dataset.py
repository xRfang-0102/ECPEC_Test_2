# =========================================================
# Raw-dialogue dataset for LoRA end-to-end training.
#
# Wraps ECFDataset (raw text, speakers, labels) and
# pre-tokenizes every utterance with the RoBERTa tokenizer
# ONCE at construction time, then pads every dialogue to
# max_dialogue_length x max_utterance_length.
#
# num_workers=0 friendly (no shared state beyond the
# pre-tokenized tensors).
# =========================================================

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from dataset.dataset import ECFDataset


class RawECFDataset(Dataset):

    def __init__(
        self,
        json_path,
        max_dialogue_length=40,
        max_utterance_length=128,
        tokenizer_path="pretrained/roberta-base",
        tokenizer=None,
    ):
        super().__init__()

        self.base = ECFDataset(
            json_path=str(json_path),
            max_dialogue_length=max_dialogue_length,
        )

        self.max_dialogue_length = int(max_dialogue_length)
        self.max_utterance_length = int(max_utterance_length)

        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else AutoTokenizer.from_pretrained(tokenizer_path)
        )

        self.pad_token_id = int(
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        # -------------------------------------------------
        # One-time tokenization of every utterance.
        # -------------------------------------------------

        self.records = []

        for index in tqdm(
            range(len(self.base)),
            desc="tokenizing",
            dynamic_ncols=True,
        ):

            item = self.base[index]

            n = int(item["num_utterances"])
            utterances = list(item["utterances"])

            input_ids = []
            attention_masks = []

            for utterance in utterances:
                encoding = self.tokenizer(
                    utterance,
                    truncation=True,
                    max_length=self.max_utterance_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids.append(encoding["input_ids"][0])
                attention_masks.append(encoding["attention_mask"][0])

            input_ids = torch.stack(input_ids)              # [n, L]
            attention_masks = torch.stack(attention_masks)  # [n, L]

            record_max_len = int(
                attention_masks.sum(dim=1).max().item()
            )

            self.records.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_masks,
                    "speaker_ids": item["speaker_ids"],
                    "emotion_labels": item["emotion_labels"],
                    "cause_labels": item["cause_labels"],
                    "pair_labels": item["pair_labels"],
                    "num_utterances": n,
                    "max_len": record_max_len,
                }
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        n = int(record["num_utterances"])
        N = self.max_dialogue_length

        return {
            "input_ids": torch.nn.functional.pad(
                record["input_ids"],
                (0, 0, 0, N - n),
                value=self.pad_token_id,
            ),
            "attention_mask": torch.nn.functional.pad(
                record["attention_mask"],
                (0, 0, 0, N - n),
                value=0,
            ),
            "speaker_ids": torch.nn.functional.pad(
                record["speaker_ids"],
                (0, N - n),
                value=0,
            ),
            "emotion_labels": torch.nn.functional.pad(
                record["emotion_labels"],
                (0, N - n),
                value=0.0,
            ),
            "cause_labels": torch.nn.functional.pad(
                record["cause_labels"],
                (0, N - n),
                value=0.0,
            ),
            "pair_labels": torch.nn.functional.pad(
                record["pair_labels"],
                (0, N - n, 0, N - n),
                value=0.0,
            ),
            "num_utterances": torch.tensor(n, dtype=torch.long),
            "max_len": torch.tensor(
                int(record["max_len"]),
                dtype=torch.long,
            ),
        }


def collate_raw_batch(batch):
    """
    Stack a list of RawECFDataset items into a training batch
    and build the masks the head model expects.

    Dynamic padding: sequences are trimmed/padded to the
    longest utterance in THIS batch (not the global 128),
    which cuts the encoder cost several-fold.
    """

    max_len = max(2, int(max(b["max_len"].item() for b in batch)))

    input_ids = torch.stack(
        [b["input_ids"][:, :max_len] for b in batch]
    )
    attention_mask = torch.stack(
        [b["attention_mask"][:, :max_len] for b in batch]
    )
    speaker_ids = torch.stack([b["speaker_ids"] for b in batch])
    emotion_labels = torch.stack([b["emotion_labels"] for b in batch])
    cause_labels = torch.stack([b["cause_labels"] for b in batch])
    pair_labels = torch.stack([b["pair_labels"] for b in batch])
    num_utterances = torch.stack([b["num_utterances"] for b in batch])

    batch_size, max_utterances = speaker_ids.shape
    device = speaker_ids.device

    position_ids = (
        torch.arange(max_utterances, device=device)
        .unsqueeze(0)
        .expand(batch_size, max_utterances)
    )

    utterance_mask = (
        torch.arange(max_utterances, device=device)
        .unsqueeze(0)
        < num_utterances.unsqueeze(1)
    )

    pair_mask = (
        utterance_mask.unsqueeze(2)
        & utterance_mask.unsqueeze(1)
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "speaker_ids": speaker_ids,
        "position_ids": position_ids,
        "utterance_mask": utterance_mask,
        "emotion_labels": emotion_labels,
        "cause_labels": cause_labels,
        "pair_labels": pair_labels,
        "pair_mask": pair_mask,
        "num_utterances": num_utterances,
    }
