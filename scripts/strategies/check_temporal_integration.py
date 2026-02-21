"""Run a real end-to-end integration check for TemporalStrategy.

This script validates that TemporalStrategy can:
1) load real artifacts (including vector_meta.parquet with year_min/year_max),
2) extract years from a query and filter by year metadata,
3) call the real generator with filtered contexts, and
4) return a structurally valid StrategyResult.

The default query contains a year so that the Temporal path is exercised.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from raqr.embedder import SentenceTransformersEmbedder
from raqr.generator import SimpleLLMGenerator
from raqr.index_store import FaissIndexStore
from raqr.loaders import JsonCorpusLoader, VectorMetaWithYears
from raqr.strategies.temporal import TemporalStrategy


def _check(condition: bool, name: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _build_strategy(
    output_dir: str,
    model_name: str,
    top_k: int,
    candidate_multiplier: int,
) -> TemporalStrategy:
    corpus_path = f"{output_dir}/corpus.jsonl"
    index_path = f"{output_dir}/vector_index.faiss"
    meta_path = f"{output_dir}/vector_meta.parquet"

    return TemporalStrategy(
        index_store=FaissIndexStore(index_path=index_path),
        meta=VectorMetaWithYears(parquet_path=meta_path),
        embedder=SentenceTransformersEmbedder(model_name=model_name),
        generator=SimpleLLMGenerator(
            model_id=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_prompt=(
                "Answer the question based only on the provided context. "
                "If the context does not contain the answer, say so."
            ),
        ),
        corpus=JsonCorpusLoader(jsonl_path=corpus_path),
        top_k=top_k,
        candidate_multiplier=candidate_multiplier,
    )


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="TemporalStrategy integration check.")
    parser.add_argument(
        "--query",
        default="What happened in 2017?",
        help="Question must contain a year so Temporal filtering runs (e.g. 'What happened in 2017?').",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "data/processed_rebel"),
        help="Directory containing corpus.jsonl, vector_index.faiss, vector_meta.parquet.",
    )
    parser.add_argument(
        "--model-name",
        default=os.getenv("MODEL_NAME", "all-MiniLM-L6-v2"),
        help="Embedding model used to query FAISS.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.getenv("TEMPORAL_TOP_K", "10")),
        help="Top-k retrieval for TemporalStrategy.",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=int(os.getenv("TEMPORAL_CANDIDATE_MULTIPLIER", "5")),
        help="Multiplier for candidate pool size (N = multiplier * top_k).",
    )
    parser.add_argument(
        "--show-contexts",
        type=int,
        default=3,
        help="How many retrieved contexts to print.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    artifact_paths = [
        Path(output_dir) / "corpus.jsonl",
        Path(output_dir) / "vector_index.faiss",
        Path(output_dir) / "vector_meta.parquet",
    ]

    print("Temporal integration check")
    print(f"- output_dir: {output_dir}")
    print(f"- model_name: {args.model_name}")
    print(f"- top_k: {args.top_k}")
    print(f"- candidate_multiplier: {args.candidate_multiplier}")
    print(f"- query: {args.query}")
    print("")

    all_ok = True
    for p in artifact_paths:
        all_ok &= _check(p.exists(), f"artifact exists: {p.as_posix()}")

    if not os.getenv("OPENAI_API_KEY"):
        all_ok &= _check(False, "OPENAI_API_KEY is set")
        print("\nCannot run generation without OPENAI_API_KEY.")
        return 1
    all_ok &= _check(True, "OPENAI_API_KEY is set")

    if not all_ok:
        print("\nIntegration check aborted due to missing prerequisites.")
        return 1

    strategy = _build_strategy(
        output_dir=output_dir,
        model_name=args.model_name,
        top_k=args.top_k,
        candidate_multiplier=args.candidate_multiplier,
    )

    result = strategy.retrieve_and_generate(args.query)

    print("")
    all_ok &= _check(result.status in {"OK", "NO_CONTEXT", "ERROR"}, "valid status")
    all_ok &= _check("total" in result.latency_ms, "latency includes total")
    all_ok &= _check("retrieval" in result.latency_ms, "latency includes retrieval")
    if result.status == "OK":
        all_ok &= _check(len(result.context_scores) > 0, "OK has non-empty context_scores")
        all_ok &= _check(bool(result.answer.strip()), "OK has non-empty answer")
    elif result.status == "NO_CONTEXT":
        all_ok &= _check(len(result.context_scores) == 0, "NO_CONTEXT has empty context_scores")
    else:
        all_ok &= _check(bool(result.error), "ERROR has non-empty error message")

    print("\nResult")
    print(f"- status: {result.status}")
    print(f"- error: {result.error}")
    print(f"- latency_ms: {result.latency_ms}")
    print(f"- answer: {result.answer}")

    if result.context_scores:
        show_n = max(0, args.show_contexts)
        print(f"\nTop {min(show_n, len(result.context_scores))} contexts")
        for idx, (ctx, score) in enumerate(result.context_scores[:show_n], start=1):
            print(f"\n[{idx}] score={score:.4f}")
            print(ctx[:800])

    print("\nSummary")
    if all_ok:
        print("Temporal integration check PASSED.")
        return 0

    print("Temporal integration check FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
