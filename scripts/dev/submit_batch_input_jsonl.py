#!/usr/bin/env python3
"""Upload an existing batch JSONL and create an OpenAI Batch job (no strategy recording).

Use when you already have files like ``data/processed/batch_input_strategy_001.jsonl``
from a prior ``submit_batches`` run and only need to re-enqueue (e.g. after enqueue
failure). Does not re-run dense/graph or entity extraction.

After the batch completes, put the printed ``batch_id`` into
``batch_state_strategy.json`` for the corresponding shard (or add a shard entry),
then run ``make collect-answer-batch``.

Usage:
  poetry run python scripts/dev/submit_batch_input_jsonl.py data/processed/batch_input_strategy_001.jsonl
  poetry run python scripts/dev/submit_batch_input_jsonl.py --completion-window 24h path/to/shard.jsonl
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Submit an existing batch input JSONL to OpenAI Batch API.",
    )
    parser.add_argument(
        "jsonl_path",
        type=Path,
        help="Path to JSONL (e.g. data/processed/batch_input_strategy_001.jsonl).",
    )
    parser.add_argument(
        "--completion-window",
        default="24h",
        help="Batch completion window (default: 24h).",
    )
    parser.add_argument(
        "--endpoint",
        default="/v1/chat/completions",
        help="Batch API endpoint (default: /v1/chat/completions, matches strategy batch).",
    )
    args = parser.parse_args()

    path = args.jsonl_path.resolve()
    if not path.is_file():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 1

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1

    client = OpenAI(api_key=api_key)

    print(f"Uploading {path} (purpose=batch)...")
    with path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint=args.endpoint,
        completion_window=args.completion_window,
        metadata={
            "description": "submit_batch_input_jsonl (resubmit shard)",
            "input_file": path.name,
        },
    )

    print(f"batch_id: {batch.id}")
    print(f"input_file_id: {uploaded.id}")
    print(
        "Next: when completed, set this batch_id on the matching shard in "
        "batch_state_strategy.json, then run make collect-answer-batch."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
