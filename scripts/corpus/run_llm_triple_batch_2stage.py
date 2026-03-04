"""Orchestrate two-stage LLM triple extraction (Discovery -> Validation) with wait.

Runs the full pipeline: Stage 1 submit -> wait -> collect -> Stage 2 submit -> wait -> collect -> build graph.
Waits by default (poll every 10 min, timeout 48h). Resume-safe: skips steps already done.

Usage:
    poetry run python scripts/corpus/run_llm_triple_batch_2stage.py --corpus data/processed/corpus.jsonl
    poetry run python scripts/corpus/run_llm_triple_batch_2stage.py --corpus data/processed/corpus.jsonl --no-wait

Requires OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_POLL_SECONDS = 600  # 10 minutes
DEFAULT_TIMEOUT_SECONDS = 48 * 3600  # 48 hours


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _wait_for_batch(
    client: OpenAI,
    batch_id: str,
    poll_seconds: int = DEFAULT_POLL_SECONDS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> bool:
    """Poll until batch completed or timeout. Returns True if completed, False if timeout/failed."""
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
            logger.error("Batch %s timed out after %.1f hours.", batch_id, elapsed / 3600)
            return False
        next_poll = min(poll_seconds, max(1, int(timeout_seconds - elapsed)))
        logger.info(
            "Batch %s status=%s (elapsed %.1f min). Polling again in %d s...",
            batch_id,
            status,
            elapsed / 60,
            next_poll,
        )
        time.sleep(next_poll)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Orchestrate two-stage LLM triple extraction with wait.",
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Path to corpus.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "data/processed"),
        help="Directory for batch state files; Stage 2 collect writes a temp corpus_llm.jsonl (atomically replaced into corpus.jsonl).",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not wait for batches; exit after submit and print status.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=DEFAULT_POLL_SECONDS,
        help=f"Poll interval in seconds (default: {DEFAULT_POLL_SECONDS}).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Max wait per batch in seconds (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    corpus_path = Path(args.corpus)
    if not corpus_path.is_file():
        logger.error("Corpus file not found: %s", corpus_path)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parent.parent.parent
    script_dir = Path(__file__).resolve().parent

    def run_script(name: str, *cmd_args: str) -> int:
        script = script_dir / name
        cmd = [sys.executable, str(script)] + list(cmd_args)
        result = subprocess.run(cmd, cwd=str(project_root))
        return result.returncode

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY is not set.")
        return 1
    client = OpenAI(api_key=api_key)

    # ----- Stage 1 -----
    state1_path = output_dir / "batch_state_stage1.json"
    if not state1_path.is_file():
        logger.info("Submitting Stage 1 (Discovery) batch...")
        if run_script("submit_llm_triple_batch_stage1.py", "--corpus", str(corpus_path), "--output-dir", str(output_dir)) != 0:
            return 1
        if not state1_path.is_file():
            logger.error("Stage 1 state file not created.")
            return 1

    with state1_path.open("r", encoding="utf-8") as f:
        state1 = json.load(f)
    batch1_id = state1.get("batch_id")
    if not batch1_id:
        logger.error("Stage 1 state has no batch_id.")
        return 1

    batch1 = client.batches.retrieve(batch1_id)
    if batch1.status != "completed":
        if args.no_wait:
            logger.info("Stage 1 not completed (status=%s). Exiting (use without --no-wait to block).", batch1.status)
            return 0
        if not _wait_for_batch(client, batch1_id, args.poll_seconds, args.timeout_seconds):
            return 1

    candidates_path = output_dir / "llm_candidates_stage1.jsonl"
    if not candidates_path.is_file() or candidates_path.stat().st_size == 0:
        logger.info("Collecting Stage 1 results...")
        if run_script(
            "collect_llm_triple_batch_stage1.py",
            "--state", str(state1_path),
            "--corpus", str(corpus_path),
            "--output-dir", str(output_dir),
        ) != 0:
            return 1

    # ----- Stage 2 -----
    state2_path = output_dir / "batch_state_stage2.json"
    if not state2_path.is_file():
        logger.info("Submitting Stage 2 (Validation) batch...")
        if run_script(
            "submit_llm_triple_batch_stage2.py",
            "--corpus", str(corpus_path),
            "--candidates", str(candidates_path),
            "--output-dir", str(output_dir),
        ) != 0:
            return 1
        if not state2_path.is_file():
            logger.error("Stage 2 state file not created.")
            return 1

    with state2_path.open("r", encoding="utf-8") as f:
        state2 = json.load(f)
    batch2_id = state2.get("batch_id")
    if not batch2_id:
        logger.error("Stage 2 state has no batch_id.")
        return 1

    batch2 = client.batches.retrieve(batch2_id)
    if batch2.status != "completed":
        if args.no_wait:
            logger.info("Stage 2 not completed (status=%s). Exiting (use without --no-wait to block).", batch2.status)
            return 0
        if not _wait_for_batch(client, batch2_id, args.poll_seconds, args.timeout_seconds):
            return 1

    corpus_llm_path = output_dir / "corpus_llm.jsonl"
    logger.info("Collecting Stage 2 results...")
    if run_script(
        "collect_llm_triple_batch_stage2.py",
        "--state", str(state2_path),
        "--corpus", str(corpus_path),
        "--output", str(corpus_llm_path),
        "--output-dir", str(output_dir),
    ) != 0:
        return 1

    graph_path = output_dir / "graph.pkl"
    logger.info("Building graph from %s -> %s", corpus_llm_path, graph_path)
    if run_script(
        "build_graph_from_corpus.py",
        "--corpus", str(corpus_llm_path),
        "--graph-out", str(graph_path),
    ) != 0:
        return 1

    # Atomically replace corpus.jsonl with enriched content to avoid redundant storage.
    # os.replace() is atomic on POSIX and Windows: corpus.jsonl is unchanged until the
    # operation completes; on failure, the temp corpus_llm file remains.
    if corpus_llm_path.resolve() != corpus_path.resolve():
        logger.info("Replacing %s with enriched corpus (atomic)...", corpus_path)
        try:
            os.replace(corpus_llm_path, corpus_path)
        except OSError as e:
            logger.error("Failed to replace corpus.jsonl: %s", e)
            return 1
    else:
        # Same path: collect wrote directly to corpus; nothing to replace.
        pass

    logger.info("Two-stage LLM extraction complete. corpus.jsonl and graph.pkl updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
