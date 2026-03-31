import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from raqr.generation.batch_orchestrator import submit_batches

load_dotenv()

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Submit OpenAI Batch for strategy generation (Dense + Graph per question).",
    )
    parser.add_argument(
        "--benchmark",
        default=os.getenv("BENCHMARK_PATH", "data/processed/benchmark.jsonl"),
        help="Path to benchmark JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "data/processed"),
        help="Directory for batch files and state.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max benchmark questions to include (optional).",
    )
    parser.add_argument(
        "--completion-window",
        default="24h",
        help="Batch completion window.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    benchmark_path = Path(args.benchmark)
    if not benchmark_path.is_file():
        logger.error("Benchmark not found: %s", benchmark_path)
        return 1

    return submit_batches(
        benchmark_path=benchmark_path,
        output_dir=Path(args.output_dir),
        limit=args.limit,
        completion_window=args.completion_window,
    )

if __name__ == "__main__":
    sys.exit(main())
