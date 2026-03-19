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
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


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
    items = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    if not items:
        logger.warning("No items in input. Exiting.")
        return 0

    logger.info("Loaded %d items from %s", len(items), input_path)

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

    # Build rows and group by split
    split_to_rows: dict[str, list[dict]] = {}
    for row in build_router_dataset_rows(
        items,
        embedder=embedder,
        probe=probe,
        compute_qfeat=qfeat_fn,
        delta=args.delta,
    ):
        split = row.get("split", "train")
        split_to_rows.setdefault(split, []).append(row)

    # Write split-specific files
    for split, rows in split_to_rows.items():
        out_path = output_dir / f"labeled_{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(_to_json_safe(row), ensure_ascii=False) + "\n")
        logger.info("Wrote %d rows to %s", len(rows), out_path)

    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
