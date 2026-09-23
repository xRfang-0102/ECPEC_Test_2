import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from dataset.dataset import ECFDataset


def safe_percent(value, total):
    if total == 0:
        return 0.0
    return value / total * 100.0


def analyze_dataset(dataset, split_name):
    num_dialogues = len(dataset)

    total_utterances = 0
    total_pairs = 0
    total_candidate_pairs = 0

    total_emotion_utterances = 0
    total_cause_utterances = 0

    dialogue_lengths = []
    pair_distances = []

    cause_before = 0
    self_cause = 0
    cause_after = 0

    same_speaker_pairs = 0
    cross_speaker_pairs = 0

    emotion_nodes_total = 0
    cause_nodes_total = 0

    emotion_single_cause = 0
    emotion_multi_cause = 0

    cause_single_emotion = 0
    cause_multi_emotion = 0

    dialogues_with_pairs = 0
    dialogues_without_pairs = 0

    max_dialogue_id = None
    max_dialogue_length = 0

    max_pair_distance = 0
    max_pair_distance_dialogue = None
    max_pair_distance_pair = None

    for sample in dataset:
        dialogue_id = sample["dialogue_id"]
        num_utterances = sample["num_utterances"]

        speaker_ids = sample["speaker_ids"].tolist()
        emotion_labels = sample["emotion_labels"]
        cause_labels = sample["cause_labels"]
        pairs = sample["pairs"]

        total_utterances += num_utterances
        total_pairs += len(pairs)

        # Full N x N pair candidate matrix
        total_candidate_pairs += (
            num_utterances * num_utterances
        )

        total_emotion_utterances += int(
            emotion_labels.sum().item()
        )

        total_cause_utterances += int(
            cause_labels.sum().item()
        )

        dialogue_lengths.append(
            num_utterances
        )

        if num_utterances > max_dialogue_length:
            max_dialogue_length = num_utterances
            max_dialogue_id = dialogue_id

        if len(pairs) > 0:
            dialogues_with_pairs += 1
        else:
            dialogues_without_pairs += 1

        emotion_to_causes = defaultdict(set)
        cause_to_emotions = defaultdict(set)

        for emotion_idx, cause_idx in pairs:
            emotion_to_causes[
                emotion_idx
            ].add(
                cause_idx
            )

            cause_to_emotions[
                cause_idx
            ].add(
                emotion_idx
            )

            # Positive means cause is before emotion
            signed_distance = (
                emotion_idx - cause_idx
            )

            abs_distance = abs(
                signed_distance
            )

            pair_distances.append(
                abs_distance
            )

            if abs_distance > max_pair_distance:
                max_pair_distance = abs_distance
                max_pair_distance_dialogue = (
                    dialogue_id
                )

                max_pair_distance_pair = (
                    emotion_idx,
                    cause_idx,
                )

            if cause_idx < emotion_idx:
                cause_before += 1

            elif cause_idx == emotion_idx:
                self_cause += 1

            else:
                cause_after += 1

            if (
                speaker_ids[emotion_idx]
                == speaker_ids[cause_idx]
            ):
                same_speaker_pairs += 1
            else:
                cross_speaker_pairs += 1

        emotion_nodes_total += len(
            emotion_to_causes
        )

        cause_nodes_total += len(
            cause_to_emotions
        )

        for causes in emotion_to_causes.values():
            if len(causes) == 1:
                emotion_single_cause += 1
            elif len(causes) > 1:
                emotion_multi_cause += 1

        for emotions in cause_to_emotions.values():
            if len(emotions) == 1:
                cause_single_emotion += 1
            elif len(emotions) > 1:
                cause_multi_emotion += 1

    avg_dialogue_length = (
        mean(dialogue_lengths)
        if dialogue_lengths
        else 0.0
    )

    median_dialogue_length = (
        median(dialogue_lengths)
        if dialogue_lengths
        else 0.0
    )

    min_dialogue_length = (
        min(dialogue_lengths)
        if dialogue_lengths
        else 0
    )

    avg_pair_distance = (
        mean(pair_distances)
        if pair_distances
        else 0.0
    )

    median_pair_distance = (
        median(pair_distances)
        if pair_distances
        else 0.0
    )

    negative_pairs = (
        total_candidate_pairs - total_pairs
    )

    positive_ratio = (
        total_pairs / total_candidate_pairs
        if total_candidate_pairs > 0
        else 0.0
    )

    negative_ratio = (
        negative_pairs / total_candidate_pairs
        if total_candidate_pairs > 0
        else 0.0
    )

    suggested_pos_weight = (
        negative_pairs / total_pairs
        if total_pairs > 0
        else 1.0
    )

    print()
    print("=" * 68)
    print(
        f"Dataset Split: {split_name}"
    )
    print("=" * 68)

    print()
    print("[Basic Statistics]")

    print(
        f"Dialogues               : "
        f"{num_dialogues}"
    )

    print(
        f"Utterances              : "
        f"{total_utterances}"
    )

    print(
        f"Gold pairs              : "
        f"{total_pairs}"
    )

    print(
        f"Emotion utterances      : "
        f"{total_emotion_utterances}"
    )

    print(
        f"Cause utterances        : "
        f"{total_cause_utterances}"
    )

    print(
        f"Dialogues with pairs    : "
        f"{dialogues_with_pairs}"
    )

    print(
        f"Dialogues without pairs : "
        f"{dialogues_without_pairs}"
    )

    print()
    print("[Dialogue Length]")

    print(
        f"Mean                    : "
        f"{avg_dialogue_length:.2f}"
    )

    print(
        f"Median                  : "
        f"{median_dialogue_length:.2f}"
    )

    print(
        f"Minimum                 : "
        f"{min_dialogue_length}"
    )

    print(
        f"Maximum                 : "
        f"{max_dialogue_length}"
    )

    print(
        f"Longest dialogue ID     : "
        f"{max_dialogue_id}"
    )

    print()
    print("[Pair Direction]")

    print(
        f"Cause before emotion    : "
        f"{cause_before} "
        f"("
        f"{safe_percent(cause_before, total_pairs):.2f}%"
        f")"
    )

    print(
        f"Self-cause              : "
        f"{self_cause} "
        f"("
        f"{safe_percent(self_cause, total_pairs):.2f}%"
        f")"
    )

    print(
        f"Cause after emotion     : "
        f"{cause_after} "
        f"("
        f"{safe_percent(cause_after, total_pairs):.2f}%"
        f")"
    )

    print()
    print("[Pair Distance]")

    print(
        f"Mean absolute distance  : "
        f"{avg_pair_distance:.2f}"
    )

    print(
        f"Median absolute distance: "
        f"{median_pair_distance:.2f}"
    )

    print(
        f"Maximum distance        : "
        f"{max_pair_distance}"
    )

    print(
        f"Max-distance dialogue   : "
        f"{max_pair_distance_dialogue}"
    )

    print(
        f"Max-distance pair       : "
        f"{max_pair_distance_pair}"
    )

    print()
    print("[Speaker Relation]")

    print(
        f"Same-speaker pairs      : "
        f"{same_speaker_pairs} "
        f"("
        f"{safe_percent(same_speaker_pairs, total_pairs):.2f}%"
        f")"
    )

    print(
        f"Cross-speaker pairs     : "
        f"{cross_speaker_pairs} "
        f"("
        f"{safe_percent(cross_speaker_pairs, total_pairs):.2f}%"
        f")"
    )

    print()
    print("[Pair Multiplicity]")

    print(
        f"Emotion nodes           : "
        f"{emotion_nodes_total}"
    )

    print(
        f"Emotion -> 1 cause      : "
        f"{emotion_single_cause} "
        f"("
        f"{safe_percent(emotion_single_cause, emotion_nodes_total):.2f}%"
        f")"
    )

    print(
        f"Emotion -> >1 causes    : "
        f"{emotion_multi_cause} "
        f"("
        f"{safe_percent(emotion_multi_cause, emotion_nodes_total):.2f}%"
        f")"
    )

    print(
        f"Cause nodes             : "
        f"{cause_nodes_total}"
    )

    print(
        f"Cause -> 1 emotion      : "
        f"{cause_single_emotion} "
        f"("
        f"{safe_percent(cause_single_emotion, cause_nodes_total):.2f}%"
        f")"
    )

    print(
        f"Cause -> >1 emotions    : "
        f"{cause_multi_emotion} "
        f"("
        f"{safe_percent(cause_multi_emotion, cause_nodes_total):.2f}%"
        f")"
    )

    print()
    print("[Pair Class Balance]")

    print(
        f"Candidate pairs         : "
        f"{total_candidate_pairs}"
    )

    print(
        f"Positive pairs          : "
        f"{total_pairs}"
    )

    print(
        f"Negative pairs          : "
        f"{negative_pairs}"
    )

    print(
        f"Positive ratio          : "
        f"{positive_ratio * 100:.2f}%"
    )

    print(
        f"Negative ratio          : "
        f"{negative_ratio * 100:.2f}%"
    )

    print(
        f"Suggested pos_weight    : "
        f"{suggested_pos_weight:.2f}"
    )

    print("=" * 68)

    return {
        "split": split_name,

        "dialogues": num_dialogues,
        "utterances": total_utterances,

        "gold_pairs": total_pairs,
        "candidate_pairs": total_candidate_pairs,
        "positive_pairs": total_pairs,
        "negative_pairs": negative_pairs,

        "positive_ratio": positive_ratio,
        "negative_ratio": negative_ratio,

        "suggested_pos_weight": (
            suggested_pos_weight
        ),

        "emotion_utterances": (
            total_emotion_utterances
        ),

        "cause_utterances": (
            total_cause_utterances
        ),

        "max_dialogue_length": (
            max_dialogue_length
        ),

        "max_pair_distance": (
            max_pair_distance
        ),

        "cause_before_ratio": (
            safe_percent(
                cause_before,
                total_pairs,
            )
        ),

        "self_cause_ratio": (
            safe_percent(
                self_cause,
                total_pairs,
            )
        ),

        "cause_after_ratio": (
            safe_percent(
                cause_after,
                total_pairs,
            )
        ),

        "same_speaker_ratio": (
            safe_percent(
                same_speaker_pairs,
                total_pairs,
            )
        ),

        "cross_speaker_ratio": (
            safe_percent(
                cross_speaker_pairs,
                total_pairs,
            )
        ),

        "multi_cause_ratio": (
            safe_percent(
                emotion_multi_cause,
                emotion_nodes_total,
            )
        ),

        "multi_emotion_ratio": (
            safe_percent(
                cause_multi_emotion,
                cause_nodes_total,
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze ECF dataset statistics."
        )
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="data/ECF",
        help=(
            "Directory containing "
            "train.json, dev.json and test.json."
        ),
    )

    args = parser.parse_args()

    data_root = Path(
        args.data_root
    )

    split_files = {
        "TRAIN": (
            data_root / "train.json"
        ),

        "DEV": (
            data_root / "dev.json"
        ),

        "TEST": (
            data_root / "test.json"
        ),
    }

    for split_name, json_path in split_files.items():

        if not json_path.exists():
            print(
                f"[Warning] Missing split: "
                f"{json_path}"
            )
            continue

        dataset = ECFDataset(
            json_path=str(json_path),
            max_dialogue_length=None,
        )

        analyze_dataset(
            dataset=dataset,
            split_name=split_name,
        )


if __name__ == "__main__":
    main()