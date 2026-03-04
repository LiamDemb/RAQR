"""Submit Stage 2 (Validation) OpenAI Batch API job for LLM triple extraction.

Usage:
    poetry run python scripts/corpus/submit_llm_triple_batch_stage2.py \\
        --corpus data/processed/corpus.jsonl \\
        --candidates data/processed/llm_candidates_stage1.jsonl

Reads llm_candidates_stage1.jsonl and corpus.jsonl, builds validation prompts,
uploads to OpenAI, creates a batch job. Skips chunks with 0 candidates.
Saves batch_state_stage2.json.

Requires OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from raqr.data.canonical_clean import normalize_text_for_extraction
from raqr.data.llm_relations import build_batch_line, build_chat_completion_request
from raqr.prompts import get_triple_validation_prompt

load_dotenv()

logger = logging.getLogger(__name__)

BATCH_LIMIT_REQUESTS = 50_000
BATCH_LIMIT_BYTES = 200 * 1024 * 1024  # 200 MB


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Submit Stage 2 (Validation) Batch API job for LLM triple extraction.",
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Path to corpus.jsonl.",
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to llm_candidates_stage1.jsonl from Stage 1 collect.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "data/processed"),
        help="Directory for batch_input_stage2.jsonl, batch_state_stage2.json.",
    )
    parser.add_argument(
        "--completion-window",
        default="24h",
        help="Batch completion window (default: 24h).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    corpus_path = Path(args.corpus)
    if not corpus_path.is_file():
        logger.error("Corpus file not found: %s", corpus_path)
        return 1

    candidates_path = Path(args.candidates)
    if not candidates_path.is_file():
        logger.error("Candidates file not found: %s", candidates_path)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_by_id = {chunk.get("chunk_id") or f"chunk_{i}": chunk for i, chunk in enumerate(_iter_jsonl(corpus_path))}
    candidates_list = list(_iter_jsonl(candidates_path))

    prompt_template = get_triple_validation_prompt()

    batch_input_path = output_dir / "batch_input_stage2.jsonl"
    manifest_path = output_dir / "batch_manifest_stage2.jsonl"

    count = 0
    total_bytes = 0
    skipped_empty = 0
    with batch_input_path.open("w", encoding="utf-8") as batch_out, manifest_path.open(
        "w", encoding="utf-8"
    ) as manifest_out:
        for row in candidates_list:
            chunk_id = row.get("chunk_id") or ""
            raw_triples = row.get("raw_triples") or []
            if not raw_triples:
                skipped_empty += 1
                continue

            chunk = corpus_by_id.get(chunk_id, {})
            text_raw = chunk.get("text") or ""
            title = chunk.get("title") or ""
            text_for_extraction = normalize_text_for_extraction(text_raw)

            candidates_json = json.dumps(raw_triples, ensure_ascii=False)
            prompt = prompt_template.format(
                title=title or "N/A",
                text=text_for_extraction,
                candidates_from_stage_1=candidates_json,
            )

            body = build_chat_completion_request(prompt)
            batch_line = build_batch_line(custom_id=chunk_id, body=body)

            line_json = json.dumps(batch_line, ensure_ascii=False) + "\n"
            batch_out.write(line_json)
            total_bytes += len(line_json.encode("utf-8"))

            manifest_out.write(
                json.dumps(
                    {"custom_id": chunk_id, "chunk_id": chunk_id, "candidate_count": len(raw_triples)},
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1

    if skipped_empty:
        logger.info("Skipped %d chunks with 0 candidates", skipped_empty)
    if count == 0:
        logger.error("No chunks with candidates to process.")
        return 1

    if count > BATCH_LIMIT_REQUESTS:
        logger.error(
            "Batch exceeds %d requests (%d). Sharding not yet implemented.",
            BATCH_LIMIT_REQUESTS,
            count,
        )
        return 1
    if total_bytes > BATCH_LIMIT_BYTES:
        logger.error(
            "Batch input file exceeds 200MB (%.1f MB). Sharding not yet implemented.",
            total_bytes / (1024 * 1024),
        )
        return 1

    logger.info("Created batch_input_stage2.jsonl with %d requests (%.1f KB)", count, total_bytes / 1024)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY is not set.")
        return 1

    client = OpenAI(api_key=api_key)

    logger.info("Uploading batch input file...")
    with batch_input_path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    logger.info("Creating Stage 2 batch job (completion_window=%s)...", args.completion_window)
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window=args.completion_window,
        metadata={"description": "LLM triple extraction Stage 2 (Validation)", "corpus": str(corpus_path)},
    )

    sample_body = build_chat_completion_request("")
    state = {
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "model": sample_body.get("model", "gpt-4o-mini"),
        "created_at": batch.created_at,
        "status": batch.status,
        "corpus_path": str(corpus_path),
        "candidates_path": str(candidates_path),
        "batch_input_path": str(batch_input_path),
        "manifest_path": str(manifest_path),
        "request_count": count,
        "stage": 2,
    }
    state_path = output_dir / "batch_state_stage2.json"
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    logger.info("Stage 2 batch created: %s", batch.id)
    logger.info("State saved to %s", state_path)
    print(batch.id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
