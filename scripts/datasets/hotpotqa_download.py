"""
Download HotPotQA records from Hugging Face.

HotPotQA is a multi-hop QA dataset requiring reasoning over multiple Wikipedia
documents. See: https://huggingface.co/datasets/hotpotqa/hotpot_qa

Subsets:
  - distractor: 10 gold paragraphs + 8 distractors per question (~98k rows)
  - fullwiki: full Wikipedia retrieval setting (~105k rows)

Splits: train, validation
"""

import argparse
from pathlib import Path

from datasets import load_dataset

# HotPotQA question types: bridge (multi-hop), comparison
ALLOWED_TYPES = {"bridge", "comparison"}


def main(
    subset: str,
    split: str,
    num_samples: int | None,
    output_file: str,
    types: str | None,
    level: str | None,
) -> None:
    ds = load_dataset("hotpotqa/hotpot_qa", subset, split=split)

    if types is None:
        allowed = ALLOWED_TYPES
    elif types.strip() == "":
        allowed = None  # keep all types
    else:
        allowed = set(t.strip().lower() for t in types.split(",") if t.strip())

    # Filter by question type
    if allowed:
        ds = ds.filter(lambda x: (x.get("type") or "").lower() in allowed)

    # Filter by level (difficulty)
    if level is not None and level != "all":
        ds = ds.filter(lambda x: (x.get("level") or "").lower() == level)

    n_total = len(ds)

    if num_samples is not None and num_samples > 0:
        # Sample without replacement, seed for reproducibility
        ds = ds.shuffle(seed=42).select(range(min(num_samples, n_total)))

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds.to_json(output_file)
    print(f"Saved {len(ds)} rows to {output_file} (subset={subset}, split={split})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Download HotPotQA from Hugging Face (hotpotqa/hotpot_qa)"
    )
    p.add_argument(
        "--subset",
        type=str,
        default="distractor",
        choices=["distractor", "fullwiki"],
        help="Dataset subset: distractor (gold+distractors) or fullwiki",
    )
    p.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation"],
        help="Split to download",
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Max samples to save (default: all). Uses fixed seed for reproducibility.",
    )
    p.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSONL path (default: data/raw/hotpotqa_{subset}_{split}.jsonl)",
    )
    p.add_argument(
        "--types",
        type=str,
        default=None,
        help="Comma-separated question types to keep (default: bridge,comparison). "
        "Use empty string to keep all.",
    )
    p.add_argument(
        "--level",
        type=str,
        default="all",
        choices=["all", "easy", "medium", "hard"],
        help="Level to download (default: all).",
    )
    args = p.parse_args()

    output_file = args.output_file or f"data/raw/hotpotqa_{args.subset}_{args.split}_{args.level}.jsonl"

    main(
        subset=args.subset,
        split=args.split,
        num_samples=args.num_samples,
        output_file=output_file,
        types=args.types,
        level=args.level,
    )
