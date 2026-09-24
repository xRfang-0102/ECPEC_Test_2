import json

for split in ["train", "dev"]:
    data = json.load(
        open(f"data/ECF/{split}.json", encoding="utf-8")
    )

    buckets = {
        "d>=3": {"tot": 0, "near": 0, "cross": 0},
        "d==2": {"tot": 0, "near": 0, "cross": 0},
    }

    for x in data:
        conv = x["conversation"]
        pairs = [
            (int(e.split("_")[0]) - 1, int(c.split("_")[0]) - 1)
            for e, c in x["emotion-cause_pairs"]
        ]
        for e, c in pairs:
            d = e - c
            if d >= 3 or d == 2:
                key = "d>=3" if d >= 3 else "d==2"
                buckets[key]["tot"] += 1
                row_has_near = any(
                    -1 <= (e - c2) <= 1
                    for e2, c2 in pairs
                    if e2 == e and (e2, c2) != (e, c)
                )
                if row_has_near:
                    buckets[key]["near"] += 1
                if conv[e]["speaker"] != conv[c]["speaker"]:
                    buckets[key]["cross"] += 1

    print(f"=== {split} ===")
    for k, v in buckets.items():
        tot = max(v["tot"], 1)
        print(
            f"  {k}: total={v['tot']} "
            f"row-also-has-near-pair={v['near']} ({v['near']/tot:.0%}) "
            f"cross-speaker={v['cross']} ({v['cross']/tot:.0%})"
        )
