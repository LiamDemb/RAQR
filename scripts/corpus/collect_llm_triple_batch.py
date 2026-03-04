"""Collect OpenAI Batch API results for LLM triple extraction and merge into corpus.

Usage:
    poetry run python scripts/corpus/collect_llm_triple_batch.py --batch-id batch_xxx --corpus data/processed/corpus.jsonl --output data/processed/corpus_llm.jsonl
    poetry run python scripts/corpus/collect_llm_triple_batch.py --state data/processed/batch_state.json --corpus data/processed/corpus.jsonl

Downloads batch output, parses tool-call responses, normalizes triples with alias_map,
and writes corpus_llm.jsonl with metadata.relations populated. Run after the batch
has completed (check status first if unsure).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from raqr.data.llm_relations import _post_process_raw_triples, parse_batch_output_line

load_dotenv()

logger = logging.getLogger(__name__)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect Batch API results and merge LLM triples into corpus.",
    )
    parser.add_argument(
        "--batch-id",
        help="OpenAI batch ID (or use --state).",
    )
    parser.add_argument(
        "--state",
        help="Path to batch_state.json from submit script.",
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Path to corpus.jsonl (same as used for submit).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output corpus path (default: <output-dir>/corpus_llm.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "data/processed"),
        help="Directory for alias_map.json and default output path.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    batch_id = args.batch_id
    if not batch_id and args.state:
        state_path = Path(args.state)
        if not state_path.is_file():
            logger.error("State file not found: %s", state_path)
            return 1
        with state_path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        batch_id = state.get("batch_id")
        if not batch_id:
            logger.error("batch_id not found in state file.")
            return 1
    if not batch_id:
        logger.error("Provide --batch-id or --state.")
        return 1

    corpus_path = Path(args.corpus)
    if not corpus_path.is_file():
        logger.error("Corpus file not found: %s", corpus_path)
        return 1

    output_dir = Path(args.output_dir)
    output_path = Path(args.output) if args.output else output_dir / "corpus_llm.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    alias_map: dict = {}
    alias_path = output_dir / "alias_map.json"
    if alias_path.is_file():
        with alias_path.open("r", encoding="utf-8") as f:
            alias_map = json.load(f)
        logger.info("Loaded alias_map from %s", alias_path)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY is not set.")
        return 1

    client = OpenAI(api_key=api_key)

    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        logger.info("Batch status: %s (not yet completed)", batch.status)
        if batch.request_counts:
            logger.info(
                "  completed=%d failed=%d total=%d",
                getattr(batch.request_counts, "completed", 0),
                getattr(batch.request_counts, "failed", 0),
                getattr(batch.request_counts, "total", 0),
            )
        print("Status:", batch.status)
        return 0

    raw_by_id: dict[str, list] = {}
    completed = 0
    failed = 0

    output_file_id = getattr(batch, "output_file_id", None) or getattr(batch, "output_file", None)
    if not output_file_id:
        logger.error("Batch has no output file.")
        return 1

    logger.info("Downloading batch output...")
    content = client.files.content(output_file_id)
    text = content.read()
    if isinstance(text, bytes):
        text = text.decode("utf-8")

    for line in text.strip().split("\n"):
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("error"):
            failed += 1
            custom_id = obj.get("custom_id", "")
            raw_by_id[custom_id] = []
            continue
        completed += 1
        cid, raw_triples = parse_batch_output_line(obj)
        raw_by_id[cid] = raw_triples

    error_file_id = getattr(batch, "error_file_id", None) or getattr(batch, "error_file", None)
    if error_file_id:
        try:
            err_content = client.files.content(error_file_id)
            err_text = err_content.read()
            if isinstance(err_text, bytes):
                err_text = err_text.decode("utf-8")
            for line in err_text.strip().split("\n"):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                failed += 1
                custom_id = obj.get("custom_id", "")
                raw_by_id.setdefault(custom_id, [])
        except Exception as e:
            logger.warning("Could not fetch error file: %s", e)

    total_requests = completed + failed
    empty_count = 0
    total_triples = 0

    logger.info("Merging triples into corpus...")
    with output_path.open("w", encoding="utf-8") as out:
        for chunk in _iter_jsonl(corpus_path):
            chunk_id = chunk.get("chunk_id") or ""
            raw_triples = raw_by_id.get(chunk_id, [])
            text = chunk.get("text") or ""

            triples = _post_process_raw_triples(raw_triples, text, alias_map, chunk_id)
            if not triples:
                empty_count += 1
            total_triples += len(triples)

            chunk = dict(chunk)
            chunk.setdefault("metadata", {})["relations"] = triples
            out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info("Wrote %s", output_path)
    logger.info(
        "Summary: requests completed=%d failed=%d; chunks with 0 triples=%d; total triples=%d",
        completed,
        failed,
        empty_count,
        total_triples,
    )
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
