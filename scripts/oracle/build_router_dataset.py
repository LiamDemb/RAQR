#!/usr/bin/env python3
"""Build router dataset from oracle_raw_scores.jsonl.

Reads oracle_raw_scores, adds embeddings, probe signals, Q-feat, F1/EM scores,
and gold labels, then writes split-specific labeled_{split}.jsonl files.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _undersample_split_to_50_50(
    rows: list[dict],
    split_name: str,
    rng: random.Random,
) -> list[dict]:
    """Undersample majority class so Dense and Graph each appear equally (50/50)."""
    if not rows:
        return rows
    labels = [r.get("gold_label", "Dense") for r in rows]
    from collections import Counter

    counts = Counter(labels)
    if len(counts) != 2:
        logger.warning(
            "Cannot undersample %s: need both Dense and Graph (got %s).",
            split_name,
            dict(counts),
        )
        return rows

    min_label, min_count = counts.most_common()[-1]
    grouped: dict[str, list[dict]] = {"Dense": [], "Graph": []}
    for r in rows:
        grouped[r.get("gold_label", "Dense")].append(r)

    balanced: list[dict] = []
    for label, rs in grouped.items():
        if label == min_label:
            balanced.extend(rs)
        else:
            balanced.extend(rng.sample(rs, min_count))

    rng.shuffle(balanced)
    logger.info(
        "Undersampled %s: %d → %d rows (50/50 Dense=%d Graph=%d).",
        split_name,
        len(rows),
        len(balanced),
        sum(1 for r in balanced if r.get("gold_label") == "Dense"),
        sum(1 for r in balanced if r.get("gold_label") == "Graph"),
    )
    return balanced


def _to_json_safe(obj):
    """Convert numpy types and NaN to JSON-serializable Python types."""
    import math
    import numpy as np

    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, float) or isinstance(obj, (np.floating, np.float32, np.float64)):
        x = float(obj)
        if math.isnan(x):
            return None
        return x
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if hasattr(obj, "tolist"):
        return _to_json_safe(obj.tolist())
    return obj

DEFAULT_OUTPUT_DIR = "data/training"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_PROBE_MODEL = "all-MiniLM-L6-v2"
DEFAULT_DELTA = 0.05
DEFAULT_PROBE_TOP_K = 30


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build router dataset from oracle_raw_scores.jsonl.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help=f"Input JSONL path (default: <output-dir>/../processed/oracle_raw_scores.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for labeled_{{split}}.jsonl (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--probe-top-k",
        type=int,
        default=DEFAULT_PROBE_TOP_K,
        help=f"Probe top-k (default: {DEFAULT_PROBE_TOP_K}).",
    )
    parser.add_argument(
        "--probe-model",
        default=DEFAULT_PROBE_MODEL,
        help=f"Probe embedding model (default: {DEFAULT_PROBE_MODEL}).",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help=f"Question embedding model (default: {DEFAULT_EMBED_MODEL}).",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=DEFAULT_DELTA,
        help=f"Oracle margin for Graph to win (default: {DEFAULT_DELTA}).",
    )
    parser.add_argument(
        "--undersample",
        action="store_true",
        help="Undersample train, dev, and test each to 50/50 Dense vs Graph.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default input: data/processed/oracle_raw_scores.jsonl
    if args.input:
        input_path = Path(args.input)
    else:
        processed = Path(os.getenv("OUTPUT_DIR", "data/processed"))
        input_path = processed / "oracle_raw_scores.jsonl"

    if not input_path.is_file():
        logger.error("Input file not found: %s", input_path)
        return 1

    # Load oracle_raw_scores
    all_items = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_items.append(json.loads(line))

    if not all_items:
        logger.warning("No items in input. Exiting.")
        return 0

    logger.info("Loaded %d items from %s", len(all_items), input_path)

    # ── Incremental: load existing labeled files to skip already-computed questions ──
    existing_rows_by_split: dict[str, list[dict]] = {}
    completed_qids: set[str] = set()
    for split_name in ("train", "dev", "test"):
        labeled_path = output_dir / f"labeled_{split_name}.jsonl"
        if not labeled_path.is_file():
            continue
        rows = []
        with labeled_path.open("r", encoding="utf-8") as ef:
            for line in ef:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = row.get("question_id", "")
                if qid:
                    completed_qids.add(qid)
                rows.append(row)
        existing_rows_by_split[split_name] = rows

    # Filter to novel items only
    novel_items = [
        item for item in all_items
        if item.get("question_id", "") not in completed_qids
    ]
    skipped = len(all_items) - len(novel_items)
    if skipped:
        logger.info(
            "Skipping %d already-computed questions (%d novel to process).",
            skipped,
            len(novel_items),
        )

    if not novel_items:
        logger.info("All questions already computed. Nothing to do.")
        return 0

    # Initialize components (load once)
    processed_dir = input_path.parent
    index_path = str(processed_dir / "vector_index.faiss")
    meta_path = str(processed_dir / "vector_meta.parquet")

    if not Path(index_path).is_file() or not Path(meta_path).is_file():
        logger.error(
            "FAISS index not found. Expected %s and %s. Run build-corpus first.",
            index_path,
            meta_path,
        )
        return 1

    from raqr.embedder import SentenceTransformersEmbedder
    from raqr.features import compute_qfeat, get_qfeat_nlp
    from raqr.probe.runner import DenseProbeRunner
    from raqr.training import build_router_dataset_rows

    embedder = SentenceTransformersEmbedder(model_name=args.embed_model)
    probe = DenseProbeRunner(
        index_path=index_path,
        meta_path=meta_path,
        model_name=args.probe_model,
        top_k=args.probe_top_k,
    )
    nlp = get_qfeat_nlp()
    qfeat_fn = lambda q: compute_qfeat(q, nlp=nlp)

    # Build rows only for novel items (embeddings/probe scores are expensive)
    all_new_rows: list[dict] = list(
        build_router_dataset_rows(
            novel_items,
            embedder=embedder,
            probe=probe,
            compute_qfeat=qfeat_fn,
            delta=args.delta,
        )
    )

    # ── Collect ALL rows (feature cache from existing files + newly computed) ──
    # Existing rows are used ONLY as a feature cache; their old split
    # assignments are discarded. We re-split the entire pool by gold_label.
    all_existing_rows: list[dict] = [
        row
        for rows in existing_rows_by_split.values()
        for row in rows
    ]
    all_rows = all_existing_rows + all_new_rows

    if not all_rows:
        logger.info("No rows to write.")
        print(f"Output: {output_dir}")
        return 0

    # ── Full 3-way stratified split of ALL rows by gold_label ──
    train_ratio = float(os.getenv("TRAIN_RATIO", "0.8"))
    dev_ratio   = float(os.getenv("DEV_RATIO",   "0.1"))
    test_ratio  = float(os.getenv("TEST_RATIO",  "0.1"))
    seed        = int(os.getenv("SEED", "42"))

    try:
        from collections import Counter
        from sklearn.model_selection import train_test_split as sk_split

        labels = [r.get("gold_label", "Dense") for r in all_rows]
        label_counts = Counter(labels)

        # First carve out test set
        test_frac = test_ratio / (train_ratio + dev_ratio + test_ratio)
        can_stratify = all(c >= 2 for c in label_counts.values()) and len(all_rows) > 2

        traindev_rows, test_rows = sk_split(
            all_rows,
            test_size=test_frac,
            random_state=seed,
            stratify=labels if can_stratify else None,
        )

        # Then carve dev from the remaining train+dev pool
        traindev_labels = [r.get("gold_label", "Dense") for r in traindev_rows]
        td_counts = Counter(traindev_labels)
        can_stratify_td = all(c >= 2 for c in td_counts.values()) and len(traindev_rows) > 2
        dev_frac_of_td = dev_ratio / (train_ratio + dev_ratio)

        train_rows, dev_rows = sk_split(
            traindev_rows,
            test_size=dev_frac_of_td,
            random_state=seed,
            stratify=traindev_labels if can_stratify_td else None,
        )

        rng = random.Random(seed)
        if args.undersample:
            train_rows = _undersample_split_to_50_50(train_rows, "train", rng)
            dev_rows = _undersample_split_to_50_50(dev_rows, "dev", rng)
            test_rows = _undersample_split_to_50_50(test_rows, "test", rng)

        for r in train_rows: r["split"] = "train"
        for r in dev_rows:   r["split"] = "dev"
        for r in test_rows:  r["split"] = "test"

        final_splits: dict[str, list[dict]] = {
            "train": list(train_rows),
            "dev":   list(dev_rows),
            "test":  list(test_rows),
        }

        logger.info(
            "Final stratified split%s: %d train / %d dev / %d test (total %d).",
            " (by gold_label)" if can_stratify else " (random – too few samples per class)",
            len(train_rows), len(dev_rows), len(test_rows), len(all_rows),
        )

    except ImportError:
        logger.warning("sklearn not found; assigning all rows to 'train'.")
        for r in all_rows:
            r["split"] = "train"
        final_splits = {"train": all_rows}

    # ── Write final split files ──
    for split, rows in sorted(final_splits.items()):
        out_path = output_dir / f"labeled_{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(_to_json_safe(row), ensure_ascii=False) + "\n")
        logger.info("Wrote %d rows to %s", len(rows), out_path)

    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
