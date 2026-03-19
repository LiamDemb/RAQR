import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from raqr.generation.batch_orchestrator import collect_batches, OUTPUT_FILENAME

load_dotenv()

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect strategy batch results into oracle_raw_scores.jsonl.",
    )
    parser.add_argument(
        "--state",
        default=os.getenv("STATE_PATH", "data/processed/batch_state_strategy.json"),
        help="Path to batch_state_strategy.json from submit script.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "data/processed"),
        help="Output directory.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Output path (default: <output-dir>/{OUTPUT_FILENAME}).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    state_path = Path(args.state)
    output_dir = Path(args.output_dir)
    output_path = Path(args.output) if args.output else None

    return collect_batches(state_path, output_dir, output_path)


if __name__ == "__main__":
    sys.exit(main())
