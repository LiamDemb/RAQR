"""
Download the full ComplexTempQA dataset (~100M rows) with progress reporting.

Usage:
    python scripts/datasets/complex_tempqa_full_download.py
    python scripts/datasets/complex_tempqa_full_download.py --output_file data/raw/complex_tempqa_full.jsonl
    python scripts/datasets/complex_tempqa_full_download.py --resume  # resume from existing file
"""

import argparse
import json
import subprocess
from pathlib import Path

from datasets import load_dataset

# Total rows in ComplexTempQA (from dataset card)
TOTAL_ROWS = 100_228_457

# Print a milestone message every N rows (in addition to tqdm)
MILESTONE_EVERY = 1_000_000

# Flush to disk every N rows to reduce data loss if killed
FLUSH_EVERY = 100_000


def count_existing_rows(path: Path) -> int:
    """Count lines in existing JSONL file (uses wc -l for speed on large files)."""
    if not path.exists():
        return 0
    try:
        result = subprocess.run(
            ["wc", "-l", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.split()[0])
    except (subprocess.CalledProcessError, IndexError, ValueError):
        # Fallback: count line by line
        count = 0
        with open(path) as f:
            for _ in f:
                count += 1
        return count


def main(output_file: str, resume: bool) -> None:
    output_path = Path(output_file)
    existing_count = count_existing_rows(output_path) if resume else 0

    if resume and existing_count > 0:
        print(f"Resuming: found {existing_count:,} existing rows, will append from there")
        file_mode = "a"
        ds = load_dataset("DataScienceUIBK/ComplexTempQA", split="train", streaming=True)
        ds = ds.skip(existing_count)
    else:
        file_mode = "w"
        ds = load_dataset("DataScienceUIBK/ComplexTempQA", split="train", streaming=True)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    # Wrap iterator with tqdm for progress bar (items/sec, ETA when total known)
    iterator = ds
    if tqdm is not None:
        iterator = tqdm(
            ds,
            total=TOTAL_ROWS,
            initial=existing_count,
            unit=" rows",
            unit_scale=True,
            desc="Downloading ComplexTempQA",
            ncols=100,
        )

    written = 0
    with open(output_file, file_mode) as f:
        for ex in iterator:
            # Serialize to JSON (handle any non-standard types)
            line = json.dumps(ex, default=str, ensure_ascii=False) + "\n"
            f.write(line)
            written += 1

            if written % FLUSH_EVERY == 0:
                f.flush()

            # Optional: print milestone flags for rough progress
            total_so_far = existing_count + written
            if total_so_far % MILESTONE_EVERY == 0 and tqdm is not None:
                tqdm.write(f"[milestone] {total_so_far:,} rows written -> {output_file}")

    total = existing_count + written
    print(f"\nDone. Saved {total:,} rows to {output_file} ({written:,} new)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download full ComplexTempQA dataset (~100M rows)")
    p.add_argument(
        "--output_file",
        type=str,
        default="complex_tempqa_full.jsonl",
        help="Output JSONL file path",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing file (skip already-downloaded rows, append new ones)",
    )
    args = p.parse_args()
    main(args.output_file, args.resume)
