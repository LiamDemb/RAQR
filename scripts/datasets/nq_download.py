#!/usr/bin/env python3
# Usage: python nq_download.py 50  -> saves nq_train_50.jsonl

import os, sys
from itertools import islice
from datasets import load_dataset, Dataset

def main():
    if len(sys.argv) != 2:
        print("Usage: python sample_streamed.py <num_examples>")
        sys.exit(1)
    n = int(sys.argv[1])
    if n < 1:
        raise ValueError("num_examples must be >= 1")

    # Optional: hide progress bars so it doesn't look stuck
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BAR"] = "1"

    ds_stream = load_dataset(
        "google-research-datasets/natural_questions",
        "default",
        split="train",         # <- full split only in streaming mode
        streaming=True,
    )

    # Take first N without downloading full shards
    buf = list(islice(ds_stream, n))
    sampled = Dataset.from_list(buf)

    out = f"nq_train_{n}.jsonl"
    sampled.to_json(out, lines=True)
    print(f"Saved {len(sampled)} examples to {out}")

if __name__ == "__main__":
    main()
