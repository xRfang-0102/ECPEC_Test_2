# =========================================================
# Diagnostic: does the A+C retrieval mechanism actually
# provide evidence for far (d>=2, cross-speaker) pairs?
#
# Measures on dev, for every emotion row i with a gold cause
# j at distance>=2 and cross-speaker (the historical-head
# admissible set):
#   1. hist attention mass placed on the true cause j
#      (uniform baseline = 1 / #admissible)
#   2. rank of the true cause in: hist_explain score,
#      content_pair_logit, final pair_logit
#   3. attention entropy (head collapse check)
#   4. locality_bias vs content_logit magnitude share
#
# Usage: python diag_ac2_attention.py [checkpoint] [config]
# =========================================================

import sys
from pathlib import Path

import torch

from dataset.feature_dataset import FeatureECFDataset
from sweep_threshold import load_config, build_model

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    PROJECT_ROOT / "checkpoints" / "base_feature_mean_ac2_best.pt"
)
CONFIG_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else None


def rank_of(idx, scores, valid_mask):
    """Rank (0-based) of idx among valid_mask entries, by score desc."""
    s = scores.clone()
    s = s.masked_fill(~valid_mask, float("-inf"))
    return int((s > s[idx].item()).sum().item())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(CONFIG_PATH)
    model = build_model(config["model"], 768).to(device)
    model.eval()

    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    data_config = config.get("data", {})
    feature_root = Path(data_config.get("feature_root", "features/ECF"))
    dev_file = data_config.get("dev_feature_file")
    feature_path = PROJECT_ROOT / feature_root / dev_file

    dataset = FeatureECFDataset(
        feature_path=feature_path,
        max_dialogue_length=40,
        expected_hidden_size=768,
        validate=True,
    )

    stats = {
        "rows": 0,
        "attn_on_gold": [],
        "attn_uniform_baseline": [],
        "rank_hist_explain": [],
        "rank_content": [],
        "rank_final": [],
        "entropy": [],
        "content_std": [],
        "locality_std": [],
    }

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

            labels = sample["pair_labels"][:n, :n].to(device)
            content = out["content_pair_logits"][0]   # [N, N]
            final = out["pair_logits"][0]
            hist_attn = out["hist_retrieval_attn"][0]  # [N, N]

            # explain scores: hist channel only (last channel)
            hist_retrieved = out.get("hist_retrieval_attn")  # placeholder
            # Recompute explain scores from attention + features for
            # exactness is complex; instead approximate explain by
            # hist_attn itself (explain = retrieved @ C^T, so attention
            # rank is a good proxy when value matrix is full-rank).
            loc_bias = out.get("locality_bias")
            if loc_bias is not None:
                loc_bias = loc_bias[0]
                stats["content_std"].append(float(content.std().item()))
                stats["locality_std"].append(float(loc_bias.std().item()))

            positions = torch.arange(n, device=device)
            dist = positions.unsqueeze(1) - positions.unsqueeze(0)
            cross = spk[0].unsqueeze(1) != spk[0].unsqueeze(0)
            admissible = (dist >= 2) & cross  # [N, N]

            for i in range(n):
                golds = (labels[i] == 1).nonzero(as_tuple=False).squeeze(-1)
                for j in golds:
                    j = int(j.item())
                    if not admissible[i, j].item():
                        continue
                    stats["rows"] += 1

                    am = admissible[i].clone()
                    k_adm = int(am.sum().item())
                    if k_adm == 0:
                        continue

                    attn_i = hist_attn[i]
                    attn_gold = float(attn_i[j].item())
                    stats["attn_on_gold"].append(attn_gold)
                    stats["attn_uniform_baseline"].append(1.0 / k_adm)

                    # entropy of the admissible attention
                    p = attn_i[am].clamp_min(1e-9)
                    p = p / p.sum()
                    stats["entropy"].append(
                        float(-(p * p.log()).sum().item())
                        / (p.numel() and 1 or 1)
                    )

                    stats["rank_hist_explain"].append(
                        rank_of(j, attn_i, am)
                    )
                    stats["rank_content"].append(
                        rank_of(j, content[i], am)
                    )
                    stats["rank_final"].append(
                        rank_of(j, final[i], am)
                    )

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print("=" * 72)
    print("A+C retrieval diagnostic on dev far cross-speaker gold pairs")
    print("=" * 72)
    print(f"far (d>=2, cross-speaker) gold rows: {stats['rows']}")
    print()
    print(
        f"hist attention on gold: {mean(stats['attn_on_gold']):.4f} "
        f"vs uniform baseline {mean(stats['attn_uniform_baseline']):.4f} "
        f"(ratio {mean(stats['attn_on_gold']) / max(mean(stats['attn_uniform_baseline']), 1e-9):.2f}x)"
    )
    print(f"hist attention entropy (normalized-ish): {mean(stats['entropy']):.4f}")
    print()
    print(
        f"mean rank of true cause in admissible set: "
        f"hist_attn={mean(stats['rank_hist_explain']):.2f}, "
        f"content_logit={mean(stats['rank_content']):.2f}, "
        f"final_logit={mean(stats['rank_final']):.2f}"
    )

    def pct_topk(ranks, k):
        if not ranks:
            return float("nan")
        return sum(1 for r in ranks if r < k) / len(ranks)

    print(
        f"% true cause in top-1/top-2/top-3 of admissible set: "
        f"hist_attn {pct_topk(stats['rank_hist_explain'],1):.0%}/"
        f"{pct_topk(stats['rank_hist_explain'],2):.0%}/"
        f"{pct_topk(stats['rank_hist_explain'],3):.0%} | "
        f"content {pct_topk(stats['rank_content'],1):.0%}/"
        f"{pct_topk(stats['rank_content'],2):.0%}/"
        f"{pct_topk(stats['rank_content'],3):.0%} | "
        f"final {pct_topk(stats['rank_final'],1):.0%}/"
        f"{pct_topk(stats['rank_final'],2):.0%}/"
        f"{pct_topk(stats['rank_final'],3):.0%}"
    )

    if stats["content_std"]:
        print()
        print(
            f"logit magnitude: content std={mean(stats['content_std']):.4f}, "
            f"locality_bias std={mean(stats['locality_std']):.4f}"
        )


if __name__ == "__main__":
    main()
