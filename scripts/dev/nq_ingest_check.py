#!/usr/bin/env python3
"""
List Natural Questions JSONL rows that would be skipped by ``load_nq`` / ingest.

Uses the same rules as ``raqr.data.loaders.load_nq`` (see ``nq_row_fail_reasons``).

Usage (from repo root, with PYTHONPATH=src or ``poetry run``):

  poetry run python scripts/dev/nq_ingest_check.py data/raw/nq_2500.jsonl
  poetry run python scripts/dev/nq_ingest_check.py data/raw/nq_50.jsonl --max-print 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from raqr.data.loaders import nq_row_fail_reasons, nq_row_question_text


def main() -> int:
    p = argparse.ArgumentParser(
        description="Print NQ records that fail loaders.load_nq validation."
    )
    p.add_argument(
        "path",
        type=Path,
        help="Path to NQ .jsonl",
    )
    p.add_argument(
        "--max-print",
        type=int,
        default=0,
        metavar="N",
        help="Max number of failed rows to print in full (0 = no limit).",
    )
    args = p.parse_args()

    path = args.path
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    total = 0
    failed = 0
    printed = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            reasons = nq_row_fail_reasons(row)
            if not reasons:
                continue
            failed += 1
            if args.max_print and printed >= args.max_print:
                continue
            qtext = nq_row_question_text(row) or "(could not read question)"
            print(f"--- line {line_no} ---")
            for r in reasons:
                print(f"  - {r}")
            print(f"  question: {qtext[:500]!s}{'…' if len(qtext) > 500 else ''}")
            print()
            printed += 1

    ok = total - failed
    print(
        f"Summary: {total} rows, {ok} would ingest, {failed} would skip",
        file=sys.stderr,
    )
    if args.max_print and failed > args.max_print:
        print(
            f"(Only printed first {args.max_print} failures; "
            f"{failed - args.max_print} more skipped rows not shown.)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
