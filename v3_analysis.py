# =========================================================
# v3 comprehensive verdict script.
#
# Works on BOTH frozen-feature and LoRA checkpoints.
# Per dialogue: one forward, then reports:
#
#   1. overall + per-distance error buckets @0.66
#   2. per-bucket threshold calibration (dev-swept ceiling)
#   3. per-row top-k decoding (full row / far-admissible)
#   4. d>=2 recall by rule
#   5. A+C retrieval attention diagnostic (if present)
#   6. locality_bias magnitude (if present)
#
# Usage: python v3_analysis.py [checkpoint] [config]
# =========================================================

import math
import sys
from pathlib import Path

import torch

from analysis_common import load_model_and_dataset

PROJECT_ROOT = Path(__file__).resolve().parent

CHECKPOINT = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    PROJECT_ROOT / "checkpoints" / "base_lora_ac2_best.pt"
)
CONFIG_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else None
SPLIT = sys.argv[3] if len(sys.argv) > 3 else "dev"
THRESHOLD = 0.66


def metrics(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def main():
    from train_feature import load_config

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    config = load_config(CONFIG_PATH)

    model, dataset, forward = load_model_and_dataset(
        CHECKPOINT,
        config,
        split=SPLIT,
        device=device,
        use_bf16=False,
    )

    flat = {
        "final": [], "content": [], "hist": [], "labels": [],
        "dist": [], "cross": [], "row": [], "did": [],
        "emo": [], "cause": [], "loc_bias": [],
    }

    attn_stats = {
        "rows": 0,
        "attn_on_gold": [],
        "uniform": [],
        "rank_attn": [],
        "rank_content": [],
    }

    with torch.no_grad():
        for did, sample in enumerate(dataset):
            n = int(sample["num_utterances"])

            out = forward(model, sample)

            final = out["pair_logits"][0].cpu()
            content = out.get("content_pair_logits")
            content = content[0].cpu() if content is not None else final
            hist = out.get("hist_retrieval_attn")
            hist = hist[0].cpu() if hist is not None else None
            emo = out["emotion_logits"][0].cpu()
            cause = out["cause_logits"][0].cpu()
            loc = out.get("locality_bias")
            if loc is not None:
                flat["loc_bias"].append(loc[0].cpu().reshape(-1))

            labels = sample["pair_labels"][:n, :n].cpu()
            speakers = sample["speaker_ids"][:n]

            positions = torch.arange(n)
            dist = positions.unsqueeze(1) - positions.unsqueeze(0)
            cross = (
                speakers.unsqueeze(1) != speakers.unsqueeze(0)
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
            flat["emo"].append(emo.unsqueeze(1).expand(n, n).reshape(-1))
            flat["cause"].append(cause.unsqueeze(0).expand(n, n).reshape(-1))

            # Retrieval attention on far cross-speaker gold pairs
            if hist is not None:
                admissible = (dist >= 2) & cross
                for i in range(n):
                    golds = (labels[i] == 1).nonzero(
                        as_tuple=False
                    ).squeeze(-1)
                    for j in golds:
                        j = int(j.item())
                        if not admissible[i, j].item():
                            continue
                        am = admissible[i].clone()
                        k_adm = int(am.sum().item())
                        if k_adm == 0:
                            continue
                        attn_stats["rows"] += 1
                        attn_i = hist[i]
                        attn_stats["attn_on_gold"].append(
                            float(attn_i[j].item())
                        )
                        attn_stats["uniform"].append(1.0 / k_adm)

                        def rank_of(idx, scores, valid_mask):
                            s = scores.clone()
                            s = s.masked_fill(
                                ~valid_mask, float("-inf")
                            )
                            return int(
                                (s > s[idx].item()).sum().item()
                            )

                        attn_stats["rank_attn"].append(
                            rank_of(j, attn_i, am)
                        )
                        attn_stats["rank_content"].append(
                            rank_of(j, content[i], am)
                        )

    final = torch.cat(flat["final"])
    content = torch.cat(flat["content"])
    labels = torch.cat(flat["labels"])
    dist = torch.cat(flat["dist"])
    cross = torch.cat(flat["cross"])
    rows = torch.cat(flat["row"])
    dids = torch.cat(flat["did"])
    emo = torch.cat(flat["emo"])
    cause = torch.cat(flat["cause"])
    has_hist = bool(flat["hist"])
    hist = torch.cat(flat["hist"]) if has_hist else torch.zeros_like(final)
    has_loc = bool(flat["loc_bias"])
    loc = torch.cat(flat["loc_bias"]) if has_loc else None

    far_admissible = (dist >= 2) & cross

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
        if lo == "n":
            mask = (dist <= -1) & (labels == 1)
        else:
            mask = (dist >= lo) & (dist <= hi) & (labels == 1)
        tp = int((preds & mask).sum().item())
        n = int(mask.sum().item())
        return tp / n if n else 0.0

    def row_topk_union(preds, scores, k, admissible_only):
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

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    def pct_topk(ranks, k):
        if not ranks:
            return float("nan")
        return sum(1 for r in ranks if r < k) / len(ranks)

    print("=" * 72)
    print(f"V3 verdict: {CHECKPOINT.name} [{SPLIT}]")
    print("=" * 72)

    base = torch.sigmoid(final) >= THRESHOLD
    p, r, f, tp, fp, fn = f1_at(base)
    print(f"[1] threshold {THRESHOLD}: P={p:.3f} R={r:.3f} F1={f:.3f} "
          f"(TP {tp}, FP {fp}, FN {fn})")

    print("\n[1b] Per-distance buckets:")
    for name, lo, hi in (
        ("self(0)", 0, 0),
        ("d==1", 1, 1),
        ("d==2", 2, 2),
        ("d>=3", 3, 999),
        ("d<=-1", "n", "n"),
    ):
        if lo == "n":
            mask = dist <= -1
        else:
            mask = (dist >= lo) & (dist <= hi)
        tp_b = int((base & mask & (labels == 1)).sum().item())
        fp_b = int((base & mask & (labels == 0)).sum().item())
        fn_b = int(((~base) & mask & (labels == 1)).sum().item())
        p_b, r_b, f_b = metrics(tp_b, fp_b, fn_b)
        print(f"    {name:>8}: P={p_b:.3f} R={r_b:.3f} F1={f_b:.3f} "
              f"(TP {tp_b}, FP {fp_b}, FN {fn_b})")

    print("\n[2] Per-bucket threshold calibration (dev-swept ceiling):")
    buckets = torch.where(
        dist == 0, 0,
        torch.where(dist == 1, 1,
                    torch.where(dist == 2, 2,
                                torch.where(dist >= 3, 3, 4))),
    )
    best_thr = {}
    for b, name in ((0, "self(0)"), (1, "d==1"), (2, "d==2"),
                    (3, "d>=3"), (4, "d<=-1")):
        mask_b = buckets == b
        n_pos = int((labels[mask_b] == 1).sum().item())
        best = (0.0, None)
        for thr in [x / 100 for x in range(20, 90)]:
            preds = torch.sigmoid(final) >= thr
            tp_b = int((preds[mask_b] & (labels[mask_b] == 1)).sum().item())
            fp_b = int((preds[mask_b] & (labels[mask_b] == 0)).sum().item())
            fn_b = int(((~preds[mask_b]) & (labels[mask_b] == 1)).sum().item())
            _, _, f1_b = metrics(tp_b, fp_b, fn_b)
            if f1_b > best[0]:
                best = (f1_b, thr)
        best_thr[b] = best[1]
        print(f"    {name:>8}: best thr={best[1]:.2f} F1={best[0]:.3f} "
              f"(pos={n_pos})")

    thr_tensor = torch.zeros_like(final)
    for b, thr in best_thr.items():
        if thr is not None:
            thr_tensor[buckets == b] = math.log(thr / (1 - thr))
    preds_bucket = final >= thr_tensor
    p, r, f, tp, fp, fn = f1_at(preds_bucket)
    print(f"    [combined] P={p:.3f} R={r:.3f} F1={f:.3f} "
          f"(TP {tp}, FP {fp}, FN {fn}) | far recall="
          f"{far_recall(preds_bucket):.1%}")

    print("\n[3] Per-row top-k decoding:")
    print("    full row, no threshold:")
    for k in [1, 2]:
        preds = row_topk_union(
            torch.zeros_like(base), final, k, False
        )
        p, r, f, tp, fp, fn = f1_at(preds)
        print(f"        k={k}: P={p:.3f} R={r:.3f} F1={f:.3f} "
              f"| far recall={far_recall(preds):.1%}")
    print("    threshold OR far-admissible top-k:")
    for k in [1, 2]:
        preds = row_topk_union(base.clone(), final, k, True)
        p, r, f, tp, fp, fn = f1_at(preds)
        print(f"        k={k}: P={p:.3f} R={r:.3f} F1={f:.3f} "
              f"| far recall={far_recall(preds):.1%}")

    print("\n[4] d>=2 recall by rule:")
    for name, preds in (
        (f"threshold {THRESHOLD}", base),
        ("bucket-thr", preds_bucket),
    ):
        print(f"    {name}: d==2 {bucket_recall(preds, 2, 2):.1%} | "
              f"d>=3 {bucket_recall(preds, 3, 999):.1%} | "
              f"d<=-1 {bucket_recall(preds, 'n', 'n'):.1%}")

    if has_hist:
        print("\n[5] A+C historical retrieval diagnostic:")
        print(
            f"    far cross gold rows: {attn_stats['rows']}"
        )
        print(
            f"    attn on gold: {mean(attn_stats['attn_on_gold']):.4f} "
            f"vs uniform {mean(attn_stats['uniform']):.4f} "
            f"(ratio {mean(attn_stats['attn_on_gold']) / max(mean(attn_stats['uniform']), 1e-9):.2f}x)"
        )
        print(
            f"    mean rank in admissible set: "
            f"attn={mean(attn_stats['rank_attn']):.2f}, "
            f"content={mean(attn_stats['rank_content']):.2f}"
        )
        print(
            f"    top-1/top-2/top-3 of admissible set: "
            f"attn {pct_topk(attn_stats['rank_attn'], 1):.0%}/"
            f"{pct_topk(attn_stats['rank_attn'], 2):.0%}/"
            f"{pct_topk(attn_stats['rank_attn'], 3):.0%} | "
            f"content {pct_topk(attn_stats['rank_content'], 1):.0%}/"
            f"{pct_topk(attn_stats['rank_content'], 2):.0%}/"
            f"{pct_topk(attn_stats['rank_content'], 3):.0%}"
        )

    if has_loc:
        print("\n[6] Locality prior:")
        print(f"    content std: {content.std(unbiased=False).item():.4f}")
        print(f"    locality_bias std: {loc.std(unbiased=False).item():.4f}")
        print(f"    locality_bias max abs: {loc.abs().max().item():.4f}")

    print("=" * 72)


if __name__ == "__main__":
    main()
