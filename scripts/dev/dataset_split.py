#!/usr/bin/env python3
"""
Copy a contiguous range of records from a JSONL file.

Numbering is **1-based and inclusive** (human-style, not Python indexing):
record **1** is the **first** line; ``start=1`` includes that line. ``end=1000``
includes the **1000th** line. The range ``1 1000`` is exactly 1000 records.

Each physical line is one record.

Usage (from repo root):

  poetry run python scripts/dev/dataset_split.py data/raw/all.jsonl data/raw/part1.jsonl 1 1000
  poetry run python scripts/dev/dataset_split.py in.jsonl out.jsonl 1001 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Write records START through END (1-based, both inclusive): "
            "START=1 is the first record, END=N includes the Nth record."
        )
    )
    p.add_argument("input", type=Path, help="Source .jsonl path")
    p.add_argument("output", type=Path, help="Destination .jsonl path")
    p.add_argument(
        "start",
        type=int,
        metavar="START",
        help="First record to keep (1 = first line; inclusive)",
    )
    p.add_argument(
        "end",
        type=int,
        metavar="END",
        help="Last record to keep (inclusive; e.g. 1000 = 1000th line)",
    )
    args = p.parse_args()

    if args.start < 1:
        print("error: START must be >= 1", file=sys.stderr)
        return 1
    if args.end < args.start:
        print("error: END must be >= START", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.input.open("r", encoding="utf-8") as fin, args.output.open(
        "w", encoding="utf-8"
    ) as fout:
        for i, line in enumerate(fin, 1):
            if i < args.start:
                continue
            if i > args.end:
                break
            fout.write(line)
            written += 1

    expected = args.end - args.start + 1
    if written < expected:
        print(
            f"warning: file has only {written} lines in range "
            f"[{args.start}, {args.end}] (expected up to {expected})",
            file=sys.stderr,
        )
    print(f"wrote {written} lines -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
