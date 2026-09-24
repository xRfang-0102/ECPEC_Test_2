import math
import sys
from pathlib import Path

import torch

from models.feature_base_model import FeatureECPECBaseModel
from dataset.feature_dataset import FeatureECFDataset

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    PROJECT_ROOT / "checkpoints" / "base_feature_pos2.5_stable_best.pt"
)


def f1_at(preds, labels):
    tp = int((preds & (labels == 1)).sum().item())
    fp = int((preds & (labels == 0)).sum().item())
    fn = int(((~preds) & (labels == 1)).sum().item())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1, tp, fp, fn


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureECPECBaseModel().to(device)
    model.eval()
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    dataset = FeatureECFDataset(
        feature_path=PROJECT_ROOT / "features" / "ECF" / "dev_roberta.pt",
        max_dialogue_length=40,
        expected_hidden_size=768,
        validate=True,
    )

    logits_list, labels_list, dist_list = [], [], []

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
            logits = out["pair_logits"][0].cpu()
            labels = sample["pair_labels"][:n, :n].cpu()

            positions = torch.arange(n)
            dist = positions.unsqueeze(1) - positions.unsqueeze(0)

            logits_list.append(logits.reshape(-1))
            labels_list.append(labels.reshape(-1))
            dist_list.append(dist.reshape(-1))

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    dist = torch.cat(dist_list)

    row_ids = torch.cat(
        [
            torch.arange(
                sample["num_utterances"]
            )
            .unsqueeze(1)
            .expand(
                sample["num_utterances"],
                sample["num_utterances"],
            )
            .reshape(-1)
            for sample in dataset
        ]
    )

    print("row-adaptive decision scan: logit >= thr AND logit >= row_max - delta")
    print(f"{'thr':>5} {'delta':>6} {'P':>6} {'R':>6} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4} | d2 R  d>=3 R")

    for thr in [0.50, 0.66, 0.75]:
        thr_logit = math.log(thr / (1 - thr))
        for delta in [0.3, 0.5, 0.8, 1.2, 1.6, 2.0, 2.5]:
            kept = torch.zeros_like(logits, dtype=torch.bool)

            # row max per (dialogue,row) — approximate with row_ids since
            # row ids are unique per dialogue here (single concatenation
            # across dialogues; ids repeat, so include dialogue via offset).
            # Simpler: recompute per dialogue.
            offset = 0
            for sample in dataset:
                n = sample["num_utterances"]
                block_logits = logits[offset:offset + n * n].reshape(n, n)
                row_max = block_logits.max(dim=1).values  # [n]
                row_thr = row_max.unsqueeze(1).expand(n, n).reshape(-1) - delta
                block_kept = (
                    (block_logits.reshape(-1) >= thr_logit)
                    & (block_logits.reshape(-1) >= row_thr)
                )
                kept[offset:offset + n * n] = block_kept
                offset += n * n

            p, r, f1, tp, fp, fn = f1_at(kept, labels)

            # recall per distance bucket
            d2_mask = dist == 2
            d3_mask = dist >= 3
            d2_r = (
                (kept[d2_mask] & (labels[d2_mask] == 1)).sum().item()
                / max((labels[d2_mask] == 1).sum().item(), 1)
            )
            d3_r = (
                (kept[d3_mask] & (labels[d3_mask] == 1)).sum().item()
                / max((labels[d3_mask] == 1).sum().item(), 1)
            )

            print(
                f"{thr:>5} {delta:>6} {p:>6.3f} {r:>6.3f} {f1:>6.3f} "
                f"{tp:>4} {fp:>4} {fn:>4} | {d2_r:>4.0%} {d3_r:>6.0%}"
            )


if __name__ == "__main__":
    main()
