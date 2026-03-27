import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from raqr.generation.batch_orchestrator import (
    STATE_FILENAME,
    OUTPUT_FILENAME,
    submit_batches,
    collect_batches,
)

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_POLL_SECONDS = 600
DEFAULT_TIMEOUT_SECONDS = 48 * 3600


def _wait_for_batch(
    client: OpenAI,
    batch_id: str,
    poll_seconds: int,
    timeout_seconds: int,
) -> bool:
    start = time.monotonic()
    while True:
        batch = client.batches.retrieve(batch_id)
        status = getattr(batch, "status", "unknown")
        if status == "completed":
            logger.info("Batch %s completed.", batch_id)
            return True
        if status in ("failed", "cancelled", "expired"):
            logger.error("Batch %s ended with status: %s", batch_id, status)
            return False
        elapsed = time.monotonic() - start
        if elapsed >= timeout_seconds:
            logger.error("Batch %s timed out.", batch_id)
            return False
        next_poll = min(poll_seconds, max(1, int(timeout_seconds - elapsed)))
        logger.info(
            "Batch %s status=%s (elapsed %.1f min). Polling in %d s...",
            batch_id,
            status,
            elapsed / 60,
            next_poll,
        )
        time.sleep(next_poll)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Orchestrate strategy generation batch (submit -> wait -> collect).",
    )
    parser.add_argument(
        "--benchmark",
        default=os.getenv("BENCHMARK_PATH", "data/processed/benchmark.jsonl"),
        help="Path to benchmark JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "data/processed"),
        help="Output directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max questions (Train+Dev only) for quick runs.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not wait for batches; exit after submit.",
    )
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / STATE_FILENAME
    output_path = output_dir / OUTPUT_FILENAME

    benchmark_path = Path(args.benchmark)
    if not benchmark_path.is_file():
        logger.error("Benchmark not found: %s", benchmark_path)
        return 1

    # Clean stale batch state from prior completed runs so submit can re-evaluate
    # (Do NOT delete oracle_raw_scores.jsonl — submit_batches reads it for caching)
    if state_path.is_file():
        state_path.unlink()
        logger.info("Removed stale %s", STATE_FILENAME)

    # 1. Submit Batch
    logger.info("Submitting strategy generation batch...")
    exit_code = submit_batches(
        benchmark_path=benchmark_path,
        output_dir=output_dir,
        limit=args.limit,
        completion_window="24h",
    )
    if exit_code != 0:
        logger.error("Submit failed.")
        return exit_code
        
    if not state_path.is_file():
        logger.error("State file not created after submit.")
        return 1

    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    shards = state.get("shards") or []
    if not shards:
        logger.info("No shards to process (all questions already answered). Output is up to date.")
        return 0

    # 2. Wait for Batch
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not args.no_wait:
        all_ok = True
        for shard in shards:
            batch_id = shard.get("batch_id")
            if not batch_id:
                continue
            b = client.batches.retrieve(batch_id)
            if b.status != "completed":
                if not _wait_for_batch(
                    client, batch_id, args.poll_seconds, args.timeout_seconds
                ):
                    all_ok = False
        if not all_ok:
            logger.error("One or more batches failed or timed out.")
            return 1

    # 3. Collect Batch — always merge when we have shards (collect_batches loads existing
    # oracle_raw_scores.jsonl and merges new batch answers; see batch_orchestrator.collect_batches).
    if args.no_wait:
        logger.info(
            "Skipping collect (--no-wait). When batches complete, run: "
            "make collect-answer-batch"
        )
    else:
        logger.info("Collecting batch results into %s (merges with existing file if any)...", output_path)
        exit_code = collect_batches(state_path=state_path, output_dir=output_dir)
        if exit_code != 0:
            logger.error("Collect failed.")
            return exit_code

    logger.info("Strategy batch complete. Output: %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
