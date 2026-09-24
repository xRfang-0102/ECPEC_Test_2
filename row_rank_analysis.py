import sys
from pathlib import Path

import torch

from models.feature_base_model import FeatureECPECBaseModel
from dataset.feature_dataset import FeatureECFDataset

from sweep_threshold import load_config, build_model

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    PROJECT_ROOT / "checkpoints" / "base_feature_pos2.5_stable_best.pt"
)
CONFIG_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else None
FEATURES_OVERRIDE = Path(sys.argv[3]) if len(sys.argv) > 3 else None
THRESHOLD = 0.66


def resolve_dev_features():
    if CONFIG_PATH is not None:
        config = load_config(CONFIG_PATH)
        data_config = config.get("data", {})
        feature_root = Path(data_config.get("feature_root", "features/ECF"))
        dev_file = data_config.get("dev_feature_file")
        if dev_file:
            return PROJECT_ROOT / feature_root / dev_file
    if FEATURES_OVERRIDE is not None:
        return FEATURES_OVERRIDE
    return PROJECT_ROOT / "features" / "ECF" / "dev_roberta.pt"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if CONFIG_PATH is not None:
        config = load_config(CONFIG_PATH)
        model = build_model(config["model"], 768).to(device)
    else:
        model = FeatureECPECBaseModel().to(device)
    model.eval()
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    dataset = FeatureECFDataset(
        feature_path=resolve_dev_features(),
        max_dialogue_length=40,
        expected_hidden_size=768,
        validate=True,
    )

    # rank histogram per distance bucket for gold positives
    from collections import defaultdict

    buckets = defaultdict(list)  # key -> list of (rank, found)

    with torch.no_grad():
        for sample in dataset:
            n = int(sample["num_utterances"])
            feats = sample["utterance_features"][:n].unsqueeze(0).to(device)
            spk = sample["speaker_ids"][:n].unsqueeze(0).to(device)
            pos = torch.arange(n, device=device).unsqueeze(0)
            mask = torch.ones(1, n, dtype=torch.bool, device=device)
            pmask = torch.ones(1, n, n, dtype=torch.bool, device=device)

            out = model(
                utterance_features=feats,
                speaker_ids=spk,
                position_ids=pos,
                utterance_mask=mask,
                pair_mask=pmask,
            )
            logits = out["pair_logits"][0]
            labels = sample["pair_labels"][:n, :n].to(device)
            probs = torch.sigmoid(logits)

            for i in range(n):
                row = logits[i]
                order = row.argsort(descending=True)
                rank_of = {
                    int(j): int((order == j).nonzero(as_tuple=False)[0]) + 1
                    for j in range(n)
                }
                for j in range(n):
                    if labels[i, j].item() != 1:
                        continue
                    d = i - j
                    if d >= 3:
                        key = "d>=3"
                    elif d == 2:
                        key = "d==2"
                    elif d == 1:
                        key = "d==1"
                    else:
                        key = "d<=0"
                    found = bool((probs[i, j] >= THRESHOLD).item())
                    buckets[key].append((rank_of[j], found))

    print(f"checkpoint: {CHECKPOINT}")
    print(f"{'bucket':<8} {'pos':>4} {'recall':>7} {'med_rank':>9} {'%rank1':>7} {'%rank<=2':>9} {'%rank<=3':>9}")
    for key in ["d<=0", "d==1", "d==2", "d>=3"]:
        rows = buckets[key]
        if not rows:
            continue
        ranks = [r for r, _ in rows]
        found = sum(1 for _, f in rows if f)
        ranks_sorted = sorted(ranks)
        med = ranks_sorted[len(ranks_sorted) // 2]
        p1 = sum(1 for r in ranks if r == 1) / len(ranks)
        p2 = sum(1 for r in ranks if r <= 2) / len(ranks)
        p3 = sum(1 for r in ranks if r <= 3) / len(ranks)
        print(
            f"{key:<8} {len(rows):>4} {found/len(rows):>7.1%} "
            f"{med:>9} {p1:>7.0%} {p2:>9.0%} {p3:>9.0%}"
        )

    # among MISSED far positives, how many are within top-3 of their row?
    missed_far = [r for key in ("d==2", "d>=3") for r, f in buckets[key] if not f]
    if missed_far:
        within3 = sum(1 for r in missed_far if r <= 3) / len(missed_far)
        print(
            f"\nmissed far positives (d>=2): {len(missed_far)}, "
            f"within row top-3: {within3:.0%}"
        )


if __name__ == "__main__":
    main()
