# =========================================================
# Dev error analysis for the stable base model.
#
# Buckets Dev pair errors by:
#   - relative distance (emotion_idx - cause_idx)
#   - speaker relation (same / cross)
#   - row multiplicity (1 cause vs multiple causes)
#   - node-head confidence (would coupling node heads help?)
#
# Usage: python error_analysis.py [checkpoint_path] [config_path] [features_override]
# =========================================================

import math
import sys
from pathlib import Path

import torch

from models.feature_base_model import FeatureECPECBaseModel
from dataset.feature_dataset import FeatureECFDataset

from sweep_threshold import load_config, build_model


PROJECT_ROOT = Path(__file__).resolve().parent

CHECKPOINT = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else (
        PROJECT_ROOT
        / "checkpoints"
        / "base_feature_pos2.5_stable_best.pt"
    )
)

CONFIG_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else None

FEATURES_OVERRIDE = Path(sys.argv[3]) if len(sys.argv) > 3 else None

DEV_FEATURES = PROJECT_ROOT / "features" / "ECF" / "dev_roberta.pt"


def resolve_dev_features():
    """Dev feature file for this checkpoint: config data section wins,
    then an explicit CLI override, then the legacy [CLS] file."""
    if CONFIG_PATH is not None:
        config = load_config(CONFIG_PATH)
        data_config = config.get("data", {})
        feature_root = Path(
            data_config.get("feature_root", "features/ECF")
        )
        dev_file = data_config.get("dev_feature_file")
        if dev_file:
            return PROJECT_ROOT / feature_root / dev_file
    if FEATURES_OVERRIDE is not None:
        return FEATURES_OVERRIDE
    return DEV_FEATURES

THRESHOLD = 0.66


def metrics_from_counts(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if CONFIG_PATH is not None:
        config = load_config(CONFIG_PATH)
        model = build_model(config["model"], 768).to(device)
    else:
        model = FeatureECPECBaseModel().to(device)

    model.eval()

    dev_features = resolve_dev_features()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    dataset = FeatureECFDataset(
        feature_path=dev_features,
        max_dialogue_length=40,
        expected_hidden_size=768,
        validate=True,
    )

    # ---------------------------------------------
    # Accumulators
    # ---------------------------------------------

    total = {"tp": 0, "fp": 0, "fn": 0}

    dist_buckets = {
        "self(0)": {"tp": 0, "fp": 0, "fn": 0},
        "1": {"tp": 0, "fp": 0, "fn": 0},
        "2": {"tp": 0, "fp": 0, "fn": 0},
        ">=3": {"tp": 0, "fp": 0, "fn": 0},
        "<=-1": {"tp": 0, "fp": 0, "fn": 0},
    }

    speaker_buckets = {
        "same": {"tp": 0, "fp": 0, "fn": 0},
        "cross": {"tp": 0, "fp": 0, "fn": 0},
    }

    multi_buckets = {
        "single_cause_row": {"tp": 0, "fp": 0, "fn": 0},
        "multi_cause_row": {"tp": 0, "fp": 0, "fn": 0},
    }

    node_coupling = {
        "missed_pos_total": 0,
        "missed_pos_both_heads_ok": 0,
        "fp_total": 0,
        "fp_cause_head_low": 0,
        "fp_emotion_head_low": 0,
    }

    top1_row = {"rows": 0, "hits": 0}

    # ---------------------------------------------
    # Iterate dialogues one at a time (B=1)
    # ---------------------------------------------

    with torch.no_grad():

        for sample in dataset:

            n = int(sample["num_utterances"])

            features = (
                sample["utterance_features"][:n]
                .unsqueeze(0)
                .to(device)
            )

            speaker = (
                sample["speaker_ids"][:n]
                .unsqueeze(0)
                .to(device)
            )

            position = (
                torch.arange(
                    n,
                    device=device,
                )
                .unsqueeze(0)
            )

            mask = torch.ones(
                1,
                n,
                dtype=torch.bool,
                device=device,
            )

            pair_mask = torch.ones(
                1,
                n,
                n,
                dtype=torch.bool,
                device=device,
            )

            outputs = model(
                utterance_features=features,
                speaker_ids=speaker,
                position_ids=position,
                utterance_mask=mask,
                pair_mask=pair_mask,
            )

            logits = outputs["pair_logits"][0]        # [N, N]
            emotion_logits = outputs["emotion_logits"][0]
            cause_logits = outputs["cause_logits"][0]

            labels = sample["pair_labels"][:n, :n].to(device)
            speakers = sample["speaker_ids"][:n]

            probs = torch.sigmoid(logits)
            predictions = probs >= THRESHOLD

            # -----------------------------------------
            # Per-pair buckets
            # -----------------------------------------

            positions = torch.arange(n, device=device)
            distance = (
                positions.unsqueeze(1)
                - positions.unsqueeze(0)
            )  # i - j

            same_speaker = (
                speakers.unsqueeze(1)
                == speakers.unsqueeze(0)
            ).to(device)

            row_cause_count = labels.sum(dim=1)  # [N]

            for i in range(n):
                for j in range(n):
                    gold = bool(labels[i, j].item())
                    pred = bool(predictions[i, j].item())

                    d = int(distance[i, j].item())

                    if d == 0:
                        key = "self(0)"
                    elif d == 1:
                        key = "1"
                    elif d == 2:
                        key = "2"
                    elif d >= 3:
                        key = ">=3"
                    else:
                        key = "<=-1"

                    if gold and pred:
                        total["tp"] += 1
                        dist_buckets[key]["tp"] += 1
                    elif gold:
                        total["fn"] += 1
                        dist_buckets[key]["fn"] += 1
                    elif pred:
                        total["fp"] += 1
                        dist_buckets[key]["fp"] += 1

                    if gold or pred:
                        sp_key = "same" if same_speaker[i, j] else "cross"
                        if gold and pred:
                            speaker_buckets[sp_key]["tp"] += 1
                        elif gold:
                            speaker_buckets[sp_key]["fn"] += 1
                        else:
                            speaker_buckets[sp_key]["fp"] += 1

                        m_key = (
                            "multi_cause_row"
                            if row_cause_count[i].item() > 1
                            else "single_cause_row"
                        )

                        if gold and pred:
                            multi_buckets[m_key]["tp"] += 1
                        elif gold:
                            multi_buckets[m_key]["fn"] += 1
                        else:
                            multi_buckets[m_key]["fp"] += 1

                    # Node-head coupling diagnostics
                    if gold and not pred:
                        node_coupling["missed_pos_total"] += 1
                        emotion_ok = (
                            torch.sigmoid(emotion_logits[i]).item()
                            >= 0.5
                        )
                        cause_ok = (
                            torch.sigmoid(cause_logits[j]).item()
                            >= 0.5
                        )
                        if emotion_ok and cause_ok:
                            node_coupling[
                                "missed_pos_both_heads_ok"
                            ] += 1

                    if pred and not gold:
                        node_coupling["fp_total"] += 1
                        if (
                            torch.sigmoid(cause_logits[j]).item()
                            < 0.5
                        ):
                            node_coupling["fp_cause_head_low"] += 1
                        if (
                            torch.sigmoid(emotion_logits[i]).item()
                            < 0.5
                        ):
                            node_coupling["fp_emotion_head_low"] += 1

            # Top-1 per row recall ceiling
            for i in range(n):
                if labels[i].sum().item() == 0:
                    continue
                top1_row["rows"] += 1
                if labels[i, logits[i].argmax().item()].item() == 1:
                    top1_row["hits"] += 1

    # ---------------------------------------------
    # Report
    # ---------------------------------------------

    def fmt(bucket):
        p, r, f1 = metrics_from_counts(
            bucket["tp"],
            bucket["fp"],
            bucket["fn"],
        )
        return f"P={p:.3f} R={r:.3f} F1={f1:.3f} (TP {bucket['tp']}, FP {bucket['fp']}, FN {bucket['fn']})"

    print("=" * 72)
    print("Dev error analysis @ threshold", THRESHOLD)
    print("=" * 72)

    p, r, f1 = metrics_from_counts(total["tp"], total["fp"], total["fn"])
    print(f"\n[Overall] {fmt(total)}")

    print("\n[By relative distance (i - j)]")
    for key, bucket in dist_buckets.items():
        print(f"  {key:>8}: {fmt(bucket)}")

    print("\n[By speaker relation]")
    for key, bucket in speaker_buckets.items():
        print(f"  {key:>6}: {fmt(bucket)}")

    print("\n[By row multiplicity]")
    for key, bucket in multi_buckets.items():
        print(f"  {key:>18}: {fmt(bucket)}")

    print("\n[Node-head coupling potential]")
    missed = node_coupling["missed_pos_total"]
    both_ok = node_coupling["missed_pos_both_heads_ok"]
    fp_total = node_coupling["fp_total"]
    print(
        f"  missed positives: {missed}; of these, BOTH emotion&cause heads "
        f"predict positive: {both_ok} ({both_ok / max(missed,1):.1%})"
    )
    print(
        f"  false positives: {fp_total}; with cause head negative: "
        f"{node_coupling['fp_cause_head_low']} "
        f"({node_coupling['fp_cause_head_low'] / max(fp_total,1):.1%}); "
        f"with emotion head negative: {node_coupling['fp_emotion_head_low']} "
        f"({node_coupling['fp_emotion_head_low'] / max(fp_total,1):.1%})"
    )

    print("\n[Top-1 per-row ceiling]")
    print(
        f"  rows with >=1 gold cause: {top1_row['rows']}; "
        f"top-1 score hits a true cause: {top1_row['hits']} "
        f"({top1_row['hits'] / max(top1_row['rows'],1):.1%})"
    )

    # =====================================================
    # Decision-side quick experiments (no retraining):
    #
    # 1. per-distance-bucket threshold calibration
    # 2. node-head gating (pair score + alpha * node scores)
    # 3. per-row top-k selection
    # =====================================================

    print()
    print("=" * 72)
    print("Decision-side experiments (same checkpoint, no retraining)")
    print("=" * 72)

    # Re-collect everything into flat tensors for grid search.
    flat = {"logits": [], "labels": [], "dist": [], "emo_logit": [], "cause_logit": [], "row": [], "did": []}

    with torch.no_grad():
        for sample in dataset:
            n = int(sample["num_utterances"])
            features = sample["utterance_features"][:n].unsqueeze(0).to(device)
            speaker = sample["speaker_ids"][:n].unsqueeze(0).to(device)
            position = torch.arange(n, device=device).unsqueeze(0)
            mask = torch.ones(1, n, dtype=torch.bool, device=device)
            pair_mask = torch.ones(1, n, n, dtype=torch.bool, device=device)

            outputs = model(
                utterance_features=features,
                speaker_ids=speaker,
                position_ids=position,
                utterance_mask=mask,
                pair_mask=pair_mask,
            )

            logits = outputs["pair_logits"][0].cpu()
            emo = outputs["emotion_logits"][0].cpu()
            cause = outputs["cause_logits"][0].cpu()
            labels = sample["pair_labels"][:n, :n].cpu()

            positions = torch.arange(n)
            dist = positions.unsqueeze(1) - positions.unsqueeze(0)

            flat["logits"].append(logits.reshape(-1))
            flat["labels"].append(labels.reshape(-1))
            flat["dist"].append(dist.reshape(-1))
            flat["emo_logit"].append(emo.unsqueeze(1).expand(n, n).reshape(-1))
            flat["cause_logit"].append(cause.unsqueeze(0).expand(n, n).reshape(-1))
            flat["row"].append(
                torch.arange(n).unsqueeze(1).expand(n, n).reshape(-1)
            )
            flat["did"].append(
                torch.full((n * n,), len(flat["did"]), dtype=torch.long)
            )

    logits = torch.cat(flat["logits"])
    labels = torch.cat(flat["labels"])
    dist = torch.cat(flat["dist"])
    emo = torch.cat(flat["emo_logit"])
    cause = torch.cat(flat["cause_logit"])
    rows = torch.cat(flat["row"])
    dids = torch.cat(flat["did"])

    def f1_at(preds):
        tp = int(((preds) & (labels == 1)).sum().item())
        fp = int((preds & (labels == 0)).sum().item())
        fn = int(((~preds) & (labels == 1)).sum().item())
        p, r, f1 = metrics_from_counts(tp, fp, fn)
        return p, r, f1, tp, fp, fn

    # ---- Experiment 1: per-distance-bucket thresholds ----
    print("\n[Exp1] Per-distance-bucket threshold calibration")
    dist_bucket = torch.where(
        dist == 0, 0,
        torch.where(dist == 1, 1, torch.where(dist == 2, 2, torch.where(dist >= 3, 3, 4))),
    )
    best_thr = {}
    for b, name in [(0, "self(0)"), (1, "1"), (2, "2"), (3, ">=3"), (4, "<=-1")]:
        mask_b = dist_bucket == b
        n_pos = int((labels[mask_b] == 1).sum().item())
        best = (0.0, None)
        for thr in [x / 100 for x in range(30, 86)]:
            preds = torch.sigmoid(logits) >= thr
            tp = int((preds[mask_b] & (labels[mask_b] == 1)).sum().item())
            fp = int((preds[mask_b] & (labels[mask_b] == 0)).sum().item())
            fn = int(((~preds[mask_b]) & (labels[mask_b] == 1)).sum().item())
            _, _, f1 = metrics_from_counts(tp, fp, fn)
            if f1 > best[0]:
                best = (f1, thr)
        best_thr[b] = best[1]
        print(f"  {name:>8}: best thr={best[1]:.2f} F1={best[0]:.3f} (pos={n_pos})")

    # Combined prediction with per-bucket thresholds
    thr_tensor = torch.zeros_like(logits)
    for b, thr in best_thr.items():
        if thr is not None:
            thr_tensor[dist_bucket == b] = math.log(thr / (1 - thr))
    preds_bucket = logits >= thr_tensor
    p, r, f1, tp, fp, fn = f1_at(preds_bucket)
    print(f"  [combined] P={p:.3f} R={r:.3f} F1={f1:.3f} (TP {tp}, FP {fp}, FN {fn})")

    # ---- Experiment 2: node-head gating ----
    print("\n[Exp2] Node-head gating: score = pair_logit + alpha*(emotion_logit + cause_logit)")
    best = (0.0, None)
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]:
        adjusted = logits + alpha * (emo + cause)
        preds = torch.sigmoid(adjusted) >= THRESHOLD
        p, r, f1, tp, fp, fn = f1_at(preds)
        if f1 > best[0]:
            best = (f1, alpha)
        print(f"  alpha={alpha:.2f}: P={p:.3f} R={r:.3f} F1={f1:.3f}")
    print(f"  [best] alpha={best[1]} F1={best[0]:.3f}")

    # ---- Experiment 3: per-row top-k ----
    print("\n[Exp3] Per-row top-k selection (threshold %.2f)" % THRESHOLD)
    probs = torch.sigmoid(logits)
    for k in [1, 2, 3, 4]:
        kept = torch.zeros_like(logits, dtype=torch.bool)
        # iterate (dialogue, row) pairs
        for did in torch.unique(dids):
            did_mask = dids == did
            for r_ in torch.unique(rows[did_mask]):
                idx = (did_mask & (rows == r_)).nonzero(as_tuple=False).squeeze(-1)
                row_probs = probs[idx]
                topk = row_probs.topk(min(k, idx.numel())).indices
                kept[idx[topk]] = True
        preds = kept & (probs >= THRESHOLD)
        p, r, f1, tp, fp, fn = f1_at(preds)
        print(f"  k={k}: P={p:.3f} R={r:.3f} F1={f1:.3f} (TP {tp}, FP {fp}, FN {fn})")


if __name__ == "__main__":
    main()
