import argparse
from collections import Counter
from datasets import load_dataset, Dataset

# ComplexTempQA question types
ALLOWED_TYPES = {"1c", "2c", "3c"}

LOG_EVERY = 50_000
MAX_SCAN = 55_000_000

def norm(x):
    return str(x).strip().lower() if x is not None else None

def main(num_samples, output_file):
    ds = load_dataset("DataScienceUIBK/ComplexTempQA", split="train", streaming=True)

    subset = []
    seen = kept = 0
    type_counts = Counter()

    for ex in ds:
        seen += 1

        qtype = norm(ex.get("type"))
        type_counts[qtype] += 1

        if qtype in ALLOWED_TYPES:
            subset.append(ex)
            kept += 1
            if kept >= num_samples:
                break

        if seen % LOG_EVERY == 0:
            print(f"[progress] scanned={seen:,} kept={kept:,} top_types={type_counts.most_common(6)}")

        if seen >= MAX_SCAN:
            break

    if not subset:
        raise ValueError(f"No matches found. Observed types: {type_counts.most_common(20)}")

    Dataset.from_list(subset).to_json(output_file)
    print(f"Saved {kept} rows to {output_file} (scanned {seen:,})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num_samples", type=int, default=300)
    p.add_argument("--output_file", type=str, default="complex_tempqa_subset.jsonl")
    args = p.parse_args()
    main(args.num_samples, args.output_file)