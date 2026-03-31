#!/usr/bin/env python3
"""
Reset LLM IE flags for chunk IDs so they are included in the next IE batch submit.

The input file can be:

1. **Raw terminal / log paste** — any text; every line is scanned for the pattern
   ``Failed to parse onepass batch output for <chunk_id>:`` (same as
   ``raqr.data.llm_onepass``). Chunk IDs are extracted automatically.

2. **Plain list (optional)** — a line that is only a 64-char hex string (SHA256-style
   ``chunk_id``) is also treated as an ID, so old one-id-per-line files still work.

For each matching row in ``corpus.jsonl``, sets ``metadata.ie_extracted`` to false
and clears ``metadata.entities`` and ``metadata.relations``.

Usage (from repo root):

  poetry run python scripts/dev/reset_failed_ie_chunks.py build_log.txt
  poetry run python scripts/dev/reset_failed_ie_chunks.py \\
    --corpus data/processed/corpus.jsonl build_log.txt
  poetry run python scripts/dev/reset_failed_ie_chunks.py --dry-run build_log.txt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

# Same log format as llm_onepass.parse_onepass_batch_output_line JSONDecodeError warning.
_WARN_PARSE_PATTERN = re.compile(
    r"Failed to parse onepass batch output for\s+([^:\s]+)\s*:",
    re.IGNORECASE,
)
# Standalone chunk_id line (sha256 hex from sha256_text in pipeline).
_PLAIN_HEX_ID = re.compile(r"^[a-fA-F0-9]{64}$")


def _extract_chunk_ids_from_file(path: Path) -> set[str]:
    """Collect chunk IDs from log lines and optional plain hex lines."""
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            for m in _WARN_PARSE_PATTERN.finditer(raw):
                cid = (m.group(1) or "").strip()
                if cid:
                    ids.add(cid)
            stripped = raw.strip()
            if _PLAIN_HEX_ID.match(stripped):
                ids.add(stripped)
    return ids


def main() -> int:
    load_dotenv()
    default_corpus = Path(os.getenv("OUTPUT_DIR", "data/processed")) / "corpus.jsonl"

    p = argparse.ArgumentParser(
        description=(
            "Reset ie_extracted for failed IE chunk IDs "
            "(parse log warnings or plain id list)."
        )
    )
    p.add_argument(
        "input_file",
        type=Path,
        help=(
            "Log paste or text file: extracts chunk IDs from "
            "'Failed to parse onepass batch output for …' lines; "
            "also accepts plain 64-char hex ids per line."
        ),
    )
    p.add_argument(
        "--corpus",
        type=Path,
        default=default_corpus,
        help=f"Path to corpus.jsonl (default: {default_corpus}).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only; do not write corpus.",
    )
    args = p.parse_args()

    if not args.input_file.is_file():
        print(f"ERROR: input file not found: {args.input_file}", file=sys.stderr)
        return 1
    if not args.corpus.is_file():
        print(f"ERROR: corpus not found: {args.corpus}", file=sys.stderr)
        return 1

    target_ids = _extract_chunk_ids_from_file(args.input_file)
    if not target_ids:
        print(
            "ERROR: no chunk IDs found. Expected lines containing "
            "'Failed to parse onepass batch output for <id>:' "
            "or a plain 64-char hex id per line.",
            file=sys.stderr,
        )
        return 1

    print(f"Extracted {len(target_ids)} unique chunk_id(s) from {args.input_file}")

    missing: set[str] = set(target_ids)
    updated = 0

    if args.dry_run:
        with args.corpus.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cid = obj.get("chunk_id") or ""
                if cid not in target_ids:
                    continue
                missing.discard(cid)
                updated += 1
                print(f"would reset: {cid}")
        print(f"Dry run: would reset {updated} chunk(s); {len(missing)} id(s) not in corpus.")
        if missing:
            print(f"Not in corpus ({len(missing)}):", file=sys.stderr)
            for cid in sorted(missing)[:30]:
                print(f"  {cid}", file=sys.stderr)
            if len(missing) > 30:
                print(f"  ... and {len(missing) - 30} more", file=sys.stderr)
        return 0

    corpus_path = args.corpus.resolve()
    dir_path = corpus_path.parent
    fd, tmp_name = tempfile.mkstemp(
        suffix=".jsonl",
        prefix=".corpus_reset_ie_",
        dir=dir_path,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        with args.corpus.open("r", encoding="utf-8") as fin, tmp_path.open(
            "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                obj = json.loads(line_stripped)
                cid = obj.get("chunk_id") or ""
                if cid in target_ids:
                    missing.discard(cid)
                    meta = obj.setdefault("metadata", {})
                    meta["ie_extracted"] = False
                    meta["entities"] = []
                    meta["relations"] = []
                    updated += 1
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        tmp_path.replace(corpus_path)
    except Exception:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        raise

    print(f"Updated {updated} chunk(s) in {corpus_path}")
    if missing:
        print(f"WARNING: {len(missing)} id(s) from log were not in corpus:", file=sys.stderr)
        for cid in sorted(missing)[:30]:
            print(f"  {cid}", file=sys.stderr)
        if len(missing) > 30:
            print(f"  ... and {len(missing) - 30} more", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
