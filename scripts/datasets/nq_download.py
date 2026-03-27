#!/usr/bin/env python3
"""
Download a sample of Natural Questions (Hugging Face streaming).

Wikipedia validation: HTTP HEAD on the document title’s /wiki/ URL (no redirects);
exact URL must return 200. See scripts/datasets/_common.py.

Annotation check: at least one entry in annotations.short_answers must have a non-empty
text list with at least one non-whitespace string.

Usage:
  python nq_download.py 50
  python nq_download.py 50 -o data/raw/nq_50.jsonl
  caffeinate -i poetry run python scripts/datasets/nq_download.py 2500 -o data/raw/nq_2500.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from datasets import Dataset, load_dataset
from tqdm import tqdm

_DATASETS_DIR = Path(__file__).resolve().parent
if str(_DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATASETS_DIR))

import _common  # noqa: E402


def _has_non_empty_short_answer(row: dict) -> bool:
    """True if some short_answer has a non-empty text list with real content."""
    ann = row.get("annotations")
    if not isinstance(ann, dict):
        return False
    short_answers = ann.get("short_answers")
    if not isinstance(short_answers, list):
        return False
    for sa in short_answers:
        if not isinstance(sa, dict):
            continue
        texts = sa.get("text")
        if not isinstance(texts, list) or not texts:
            continue
        if any(isinstance(t, str) and t.strip() for t in texts):
            return True
    return False


def _row_valid(row: dict, *, wiki_language: str) -> bool:
    if not _has_non_empty_short_answer(row):
        return False
    title = _common.nq_document_title(row)
    if not title:
        return False
    return _common.wikipedia_find_page_head(title, language=wiki_language)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Natural Questions samples (streaming train split). "
        "Keeps rows with a non-empty short answer in annotations and whose document "
        "title passes Wikipedia HEAD checks."
    )
    parser.add_argument(
        "num_samples",
        type=int,
        help="Target number of examples to save (after Wikipedia filtering).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSONL path (default: nq_train_{num_samples}.jsonl)",
    )
    parser.add_argument(
        "--wiki-language",
        default="en",
        help="Wikipedia language for HEAD checks (default: en).",
    )
    args = parser.parse_args()

    n = args.num_samples
    if n < 1:
        raise ValueError("num_samples must be >= 1")

    out = args.output or f"nq_train_{n}.jsonl"

    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BAR", "1")

    ds_stream = load_dataset(
        "google-research-datasets/natural_questions",
        "default",
        split="train",
        streaming=True,
    )

    buf: list[dict] = []
    with tqdm(
        total=n,
        desc="Valid samples",
        unit="sample",
        mininterval=0.2,
    ) as pbar:
        rows_scanned = 0
        for row in ds_stream:
            rows_scanned += 1
            pbar.set_postfix(rows_scanned=rows_scanned, refresh=True)
            if _row_valid(row, wiki_language=args.wiki_language):
                buf.append(row)
                pbar.update(1)
            if len(buf) >= n:
                break

    sampled = Dataset.from_list(buf)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sampled.to_json(out, lines=True)
    print(f"Saved {len(sampled)} examples to {out}")


if __name__ == "__main__":
    main()
