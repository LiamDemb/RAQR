"""
Download 2WikiMultiHopQA records from Hugging Face.

2WikiMultiHopQA is a multi-hop QA dataset with reasoning paths over Wikipedia.

See: https://huggingface.co/datasets/framolfese/2WikiMultihopQA

Splits: train, validation, test
Types: bridge_comparison, comparison, etc.

Wikipedia validation: HTTP HEAD on each supporting title’s /wiki/ URL (no redirects);
exact URL must return 200. See scripts/datasets/_common.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import Dataset, load_dataset
from tqdm import tqdm

_DATASETS_DIR = Path(__file__).resolve().parent
if str(_DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATASETS_DIR))

import _common  # noqa: E402


def titles_exist(row: dict, *, wiki_language: str) -> bool:
    return _common.titles_all_exist_head(
        _common.twowiki_supporting_titles(row),
        language=wiki_language,
    )


def main(
    split: str,
    num_samples: int | None,
    output_file: str,
    types: str | None,
    wiki_language: str,
) -> None:
    ds = load_dataset("framolfese/2WikiMultihopQA", split=split)

    # Filter by question type
    if types is not None and types.strip():
        allowed = set(t.strip().lower() for t in types.split(",") if t.strip())
        if allowed:
            ds = ds.filter(lambda x: (x.get("type") or "").lower() in allowed)

    ds = ds.shuffle(seed=42)

    if num_samples is not None and num_samples > 0:
        valid_rows = []

        with tqdm(
            total=num_samples,
            desc="Valid samples",
            unit="sample",
            mininterval=0.2,
        ) as pbar:
            rows_scanned = 0
            for row in ds:
                rows_scanned += 1
                pbar.set_postfix(rows_scanned=rows_scanned, refresh=True)
                if titles_exist(row, wiki_language=wiki_language):
                    valid_rows.append(row)
                    pbar.update(1)

                if len(valid_rows) >= num_samples:
                    break

        ds = Dataset.from_list(valid_rows)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds.to_json(output_file)
    print(f"Saved {len(ds)} rows to {output_file} (split={split})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Download 2WikiMultihopQA from Hugging Face (framolfese/2WikiMultihopQA)"
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
    p.add_argument(
        "--wiki-language",
        type=str,
        default="en",
        help="Wikipedia language for HEAD checks (default: en).",
    )
    args = p.parse_args()

    output_file = args.output_file or f"data/raw/2wikimultihop_{args.num_samples}.jsonl"

    main(
        split=args.split,
        num_samples=args.num_samples,
        output_file=output_file,
        types=args.types,
        wiki_language=args.wiki_language,
    )
