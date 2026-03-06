"""
Download 2WikiMultiHopQA records from Hugging Face.

2WikiMultiHopQA is a multi-hop QA dataset with reasoning paths over Wikipedia,
similar to HotPotQA. The framolfese variant uses a HotPotQA-compatible schema
(supporting_facts.title, supporting_facts.sent_id) so existing loaders work.

See: https://huggingface.co/datasets/framolfese/2WikiMultihopQA

Splits: train, validation, test
Types: bridge_comparison, comparison, etc.
"""

import argparse
from pathlib import Path

from datasets import load_dataset


def main(
    split: str,
    num_samples: int | None,
    output_file: str,
    types: str | None,
) -> None:
    ds = load_dataset("framolfese/2WikiMultihopQA", split=split)

    # Filter by question type (e.g. bridge_comparison, comparison)
    if types is not None and types.strip():
        allowed = set(t.strip().lower() for t in types.split(",") if t.strip())
        if allowed:
            ds = ds.filter(lambda x: (x.get("type") or "").lower() in allowed)

    n_total = len(ds)

    if num_samples is not None and num_samples > 0:
        ds = ds.shuffle(seed=42).select(range(min(num_samples, n_total)))

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds.to_json(output_file)
    print(f"Saved {len(ds)} rows to {output_file} (split={split})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Download 2WikiMultiHopQA from Hugging Face (framolfese/2WikiMultihopQA)"
    )
    p.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
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
        help="Output JSONL path (default: data/raw/2wikimultihop_{num_samples}.jsonl)",
    )
    p.add_argument(
        "--types",
        type=str,
        default=None,
        help="Comma-separated question types to keep (e.g. bridge_comparison,comparison). "
        "Default: all types.",
    )
    args = p.parse_args()

    output_file = args.output_file or f"data/raw/2wikimultihop_{args.num_samples}.jsonl"

    main(
        split=args.split,
        num_samples=args.num_samples,
        output_file=output_file,
        types=args.types,
    )
