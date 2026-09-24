# =========================================================
# Rank-decoding scan: is the far-pair failure a DECISION
# problem rather than a representation problem?
#
# Collects per-dialogue pair logits (final + content), the
# historical retrieval attention (AC2), labels, distance and
# cross-speaker masks on dev, then evaluates decision rules:
#
#   A  threshold 0.66                     (baseline)
#   B  per-row top-k pure ranking (full row)
#   C  threshold OR per-row top-k over far-admissible set
#   D  rank fusion: content top-k OR hist-attn top-k (AC2)
#   E  row z-score normalization + threshold
#   F  per-bucket thresholds (dev-swept) + row top-k union
#
# Usage: python scan_rank_decoding.py [checkpoint] [config]
# =========================================================

import sys
from pathlib import Path

import torch

from dataset.feature_dataset import FeatureECFDataset
from sweep_threshold import load_config, build_model

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    PROJECT_ROOT / "checkpoints" / "base_feature_mean_ctrl_best.pt"
)
CONFIG_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else None
THRESHOLD = 0.66


def metrics(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


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
    dataset = FeatureECFDataset(
        feature_path=PROJECT_ROOT / feature_root / dev_file,
        max_dialogue_length=40,
        expected_hidden_size=768,
        validate=True,
    )

    flat = {
        "final": [], "content": [], "hist": [], "labels": [],
        "dist": [], "cross": [], "row": [], "did": [],
    }

    with torch.no_grad():
        for did, sample in enumerate(dataset):
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

            final = out["pair_logits"][0].cpu()
            content = out.get("content_pair_logits")
            content = content[0].cpu() if content is not None else final
            hist = out.get("hist_retrieval_attn")
            hist = hist[0].cpu() if hist is not None else None

            labels = sample["pair_labels"][:n, :n].cpu()

            positions = torch.arange(n)
            dist = positions.unsqueeze(1) - positions.unsqueeze(0)
            cross = (
                sample["speaker_ids"][:n].unsqueeze(1)
                != sample["speaker_ids"][:n].unsqueeze(0)
            )

            flat["final"].append(final.reshape(-1))
            flat["content"].append(content.reshape(-1))
            if hist is not None:
                flat["hist"].append(hist.reshape(-1))
            flat["labels"].append(labels.reshape(-1))
            flat["dist"].append(dist.reshape(-1))
            flat["cross"].append(cross.reshape(-1))
            flat["row"].append(
                torch.arange(n).unsqueeze(1).expand(n, n).reshape(-1)
            )
            flat["did"].append(torch.full((n * n,), did, dtype=torch.long))

    final = torch.cat(flat["final"])
    content = torch.cat(flat["content"])
    labels = torch.cat(flat["labels"])
    dist = torch.cat(flat["dist"])
    cross = torch.cat(flat["cross"])
    rows = torch.cat(flat["row"])
    dids = torch.cat(flat["did"])
    has_hist = bool(flat["hist"])
    hist = torch.cat(flat["hist"]) if has_hist else torch.zeros_like(final)

    far_admissible = (dist >= 2) & cross  # matches historical head mask

    def f1_at(preds):
        tp = int((preds & (labels == 1)).sum().item())
        fp = int((preds & (labels == 0)).sum().item())
        fn = int(((~preds) & (labels == 1)).sum().item())
        p, r, f1 = metrics(tp, fp, fn)
        return p, r, f1, tp, fp, fn

    def far_recall(preds):
        mask = far_admissible & (labels == 1)
        tp = int((preds & mask).sum().item())
        n = int(mask.sum().item())
        return tp / n if n else 0.0

    def bucket_recall(preds, lo, hi):
        d = dist
        if lo == "n":  # d <= -1
            mask = (d <= -1) & (labels == 1)
        else:
            mask = (d >= lo) & (d <= hi) & (labels == 1)
        tp = int((preds & mask).sum().item())
        n = int(mask.sum().item())
        return tp / n if n else 0.0

    def row_topk_union(preds, scores, k, admissible_only):
        """Union of preds with per-row top-k of scores."""
        out = preds.clone()
        for did in torch.unique(dids):
            dmask = dids == did
            for r in torch.unique(rows[dmask]):
                idx = (dmask & (rows == r)).nonzero(
                    as_tuple=False
                ).squeeze(-1)
                cand = idx
                if admissible_only:
                    cand = idx[far_admissible[idx]]
                if cand.numel() == 0:
                    continue
                s = scores[cand]
                topk = s.topk(min(k, cand.numel())).indices
                out[cand[topk]] = True
        return out

    print("=" * 72)
    print(f"Rank-decoding scan: {CHECKPOINT.name}")
    print("=" * 72)

    base = torch.sigmoid(final) >= THRESHOLD
    p, r, f, tp, fp, fn = f1_at(base)
    print(f"[A] threshold {THRESHOLD}: P={p:.3f} R={r:.3f} F1={f:.3f} "
          f"(TP {tp}, FP {fp}, FN {fn}) | far recall={far_recall(base):.1%}")

    print("\n[B] per-row top-k pure ranking (full row, no threshold):")
    for k in [1, 2, 3]:
        preds = torch.zeros_like(base)
        preds = row_topk_union(preds, final, k, admissible_only=False)
        p, r, f, tp, fp, fn = f1_at(preds)
        print(f"    k={k}: P={p:.3f} R={r:.3f} F1={f:.3f} "
              f"(TP {tp}, FP {fp}, FN {fn}) | far recall={far_recall(preds):.1%}")

    print("\n[C] threshold OR per-row top-k over far-admissible set:")
    for k in [1, 2]:
        preds = row_topk_union(base.clone(), final, k, admissible_only=True)
        p, r, f, tp, fp, fn = f1_at(preds)
        print(f"    k={k}: P={p:.3f} R={r:.3f} F1={f:.3f} "
              f"(TP {tp}, FP {fp}, FN {fn}) | far recall={far_recall(preds):.1%}")

    if has_hist:
        print("\n[D] rank fusion (AC2): threshold OR content top-k OR "
              "hist-attn top-1 over far-admissible set:")
        for k in [1, 2]:
            preds = row_topk_union(
                base.clone(), content, k, admissible_only=True
            )
            preds = row_topk_union(
                preds, hist, 1, admissible_only=True
            )
            p, r, f, tp, fp, fn = f1_at(preds)
            print(f"    content top-{k} + hist top-1: P={p:.3f} "
                  f"R={r:.3f} F1={f:.3f} (TP {tp}, FP {fp}, FN {fn}) | "
                  f"far recall={far_recall(preds):.1%}")

    print("\n[E] row z-score normalization + threshold:")
    for thr in [0.5, 0.66, 1.0]:
        zscores = torch.zeros_like(final)
        for did in torch.unique(dids):
            dmask = dids == did
            for r in torch.unique(rows[dmask]):
                idx = (dmask & (rows == r)).nonzero(
                    as_tuple=False
                ).squeeze(-1)
                s = final[idx]
                mu, sd = s.mean(), s.std(unbiased=False)
                if sd > 0:
                    zscores[idx] = (s - mu) / sd
        preds = zscores >= thr
        p, r, f, tp, fp, fn = f1_at(preds)
        print(f"    z>={thr}: P={p:.3f} R={r:.3f} F1={f:.3f} "
              f"(TP {tp}, FP {fp}, FN {fn}) | far recall={far_recall(preds):.1%}")

    print("\n[F] per-bucket thresholds (0.35/0.60/0.30/0.30/0.31) "
          "+ far top-1 union:")
    buckets = torch.where(
        dist == 0, 0,
        torch.where(dist == 1, 1,
                    torch.where(dist == 2, 2,
                                torch.where(dist >= 3, 3, 4))),
    )
    thr_map = {0: 0.35, 1: 0.60, 2: 0.30, 3: 0.30, 4: 0.31}
    thr_t = torch.zeros_like(final)
    for b, t in thr_map.items():
        thr_t[buckets == b] = torch.logit(torch.tensor(t))
    preds = final >= thr_t
    p, r, f, tp, fp, fn = f1_at(preds)
    print(f"    bucket-thr only: P={p:.3f} R={r:.3f} F1={f:.3f} "
          f"(TP {tp}, FP {fp}, FN {fn}) | far recall={far_recall(preds):.1%}")
    preds2 = row_topk_union(preds.clone(), final, 1, admissible_only=True)
    p, r, f, tp, fp, fn = f1_at(preds2)
    print(f"    bucket-thr + far top-1: P={p:.3f} R={r:.3f} F1={f:.3f} "
          f"(TP {tp}, FP {fp}, FN {fn}) | far recall={far_recall(preds2):.1%}")

    print("\n[G] d>=2 recall by rule (target: 161 positives):")
    for name, preds in [
        ("threshold 0.66", base),
        ("bucket-thr", final >= thr_t),
    ]:
        print(f"    {name}: d==2 {bucket_recall(preds, 2, 2):.1%} | "
              f"d>=3 {bucket_recall(preds, 3, 999):.1%} | "
              f"d<=-1 {bucket_recall(preds, 'n', 'n'):.1%}")


if __name__ == "__main__":
    main()
