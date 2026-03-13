from __future__ import annotations

import argparse
import json
import logging
import random
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from dotenv import load_dotenv
from raqr.data.loaders import load_2wiki, load_nq
from raqr.data.schemas import BenchmarkItem, sha256_text

logger = logging.getLogger(__name__)


def normalize_question(text: str) -> str:
    return " ".join(text.lower().strip().split())


def stratified_split(
    items_by_source: Dict[str, List[BenchmarkItem]],
    seed: int,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
) -> List[BenchmarkItem]:
    rng = random.Random(seed)
    split_items: List[BenchmarkItem] = []
    for source, items in items_by_source.items():
        ordered = sorted(items, key=lambda item: item.question_id)
        rng.shuffle(ordered)
        total = len(ordered)
        train_count = int(total * train_ratio)
        dev_count = int(total * dev_ratio)
        test_count = total - train_count - dev_count
        split_map = (
            ["train"] * train_count + ["dev"] * dev_count + ["test"] * test_count
        )
        for item, split in zip(ordered, split_map):
            split_items.append(
                BenchmarkItem(
                    question_id=item.question_id,
                    question=item.question,
                    gold_answers=item.gold_answers,
                    dataset_source=item.dataset_source,
                    split=split,
                    dataset_version=item.dataset_version,
                )
            )
        logger.info(
            "Split %s: total=%d train=%d dev=%d test=%d",
            source,
            total,
            train_count,
            dev_count,
            test_count,
        )
    return split_items


def write_jsonl(path: Path, items: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def validate_outputs(benchmark: List[BenchmarkItem]) -> None:
    if not benchmark:
        raise ValueError("Benchmark is empty.")

    sources = {item.dataset_source for item in benchmark}
    if not sources:
        raise ValueError("Benchmark has no dataset sources.")

    question_ids = set()
    question_texts = {}
    for item in benchmark:
        if not item.question_id or not item.question:
            raise ValueError("Benchmark contains empty question or question_id.")
        if not item.gold_answers:
            raise ValueError("Benchmark contains missing gold_answers.")
        if item.question_id in question_ids:
            raise ValueError("Duplicate question_id found in benchmark.")
        question_ids.add(item.question_id)

        normalized = normalize_question(item.question)
        normalized_hash = sha256_text(normalized)
        if normalized_hash in question_texts:
            if question_texts[normalized_hash] != item.split:
                raise ValueError("Normalized question leakage across splits.")
        else:
            question_texts[normalized_hash] = item.split


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Phase 1 ingestion pipeline.")
    parser.add_argument(
        "--nq",
        default=os.getenv("NQ_PATH"),
        help="Path to NQ JSON/JSONL.",
    )
    parser.add_argument(
        "--2wiki",
        default=os.getenv("2WIKI_PATH"),
        help="Path to 2WikiMultiHopQA JSON/JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "data/processed"),
        help="Output directory for processed artifacts.",
    )
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")))
    parser.add_argument(
        "--train-ratio", type=float, default=float(os.getenv("TRAIN_RATIO", "0.8"))
    )
    parser.add_argument(
        "--dev-ratio", type=float, default=float(os.getenv("DEV_RATIO", "0.1"))
    )
    parser.add_argument(
        "--test-ratio", type=float, default=float(os.getenv("TEST_RATIO", "0.1"))
    )
    parser.add_argument("--nq-version", default=os.getenv("NQ_VERSION"))
    parser.add_argument("--2wiki-version", default=os.getenv("2WIKI_VERSION"))
    args = parser.parse_args()

    dataset_paths = [
        args.nq,
        args.2wiki,
    ]
    if not any(p and str(p).strip() for p in dataset_paths):
        raise ValueError(
            "Provide at least one dataset path via CLI or env. "
            "Example: --2wiki data/raw/2wikimultihop_50.jsonl"
        )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if abs(args.train_ratio + args.dev_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")

    benchmark_by_source: Dict[str, List[BenchmarkItem]] = defaultdict(list)

    if args.nq and str(args.nq).strip():
        for item in load_nq(args.nq, dataset_version=args.nq_version):
            benchmark_by_source[item.dataset_source].append(item)
        logger.info("Loaded NQ from %s", args.nq)

    if args.2wiki and str(args.2wiki).strip():
        for item in load_2wiki(args.2wiki, dataset_version=args.2wiki_version):
            benchmark_by_source[item.dataset_source].append(item)
        logger.info("Loaded 2WikiMultiHopQA from %s", args.2wiki)

    sources_loaded = sorted(benchmark_by_source.keys())
    logger.info("Datasets included: %s", ", ".join(sources_loaded))

    benchmark = stratified_split(
        benchmark_by_source,
        seed=args.seed,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
    )
    validate_outputs(benchmark)

    output_dir = Path(args.output_dir)
    benchmark_path = output_dir / "benchmark.jsonl"
    write_jsonl(benchmark_path, (item.to_json() for item in benchmark))

    logger.info("Benchmark size: %d", len(benchmark))
    logger.info("Wrote %s", benchmark_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
