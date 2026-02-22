"""Run a real end-to-end integration check for GraphStrategy."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from raqr.entity_alias_resolver import EntityAliasResolver
from raqr.generator import SimpleLLMGenerator
from raqr.graph_store import NetworkXGraphStore
from raqr.loaders import JsonCorpusLoader
from raqr.strategies.graph import GraphStrategy, SpacyQueryEntityExtractor


def _check(condition: bool, name: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _build_strategy(output_dir: str, top_k: int) -> GraphStrategy:
    corpus_path = f"{output_dir}/corpus.jsonl"
    graph_path = f"{output_dir}/graph.pkl"
    lexicon_path = f"{output_dir}/entity_lexicon.parquet"

    alias_resolver = EntityAliasResolver.from_lexicon(lexicon_path=lexicon_path)
    return GraphStrategy(
        graph_store=NetworkXGraphStore(graph_path=graph_path),
        corpus=JsonCorpusLoader(jsonl_path=corpus_path),
        generator=SimpleLLMGenerator(
            model_id=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_prompt=(
                "Answer the question based only on the provided context. "
                "If the context does not contain the answer, say so."
            ),
        ),
        entity_extractor=SpacyQueryEntityExtractor(alias_resolver=alias_resolver),
        top_k=top_k,
        max_hops=1,
    )


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="GraphStrategy integration check.")
    parser.add_argument(
        "--query",
        default="How is Barack Obama related to the United States?",
        help="Question to run through GraphStrategy.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "data/processed_rebel"),
        help="Directory containing corpus.jsonl, graph.pkl, entity_lexicon.parquet.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.getenv("GRAPH_TOP_K", "10")),
        help="Top-k retrieval for GraphStrategy.",
    )
    parser.add_argument(
        "--show-contexts",
        type=int,
        default=3,
        help="How many retrieved contexts to print.",
    )
    args = parser.parse_args()

    artifact_paths = [
        Path(args.output_dir) / "corpus.jsonl",
        Path(args.output_dir) / "graph.pkl",
        Path(args.output_dir) / "entity_lexicon.parquet",
    ]

    print("Graph integration check")
    print(f"- output_dir: {args.output_dir}")
    print(f"- top_k: {args.top_k}")
    print(f"- query: {args.query}")
    print("")

    all_ok = True
    for path in artifact_paths:
        all_ok &= _check(path.exists(), f"artifact exists: {path.as_posix()}")

    if not os.getenv("OPENAI_API_KEY"):
        all_ok &= _check(False, "OPENAI_API_KEY is set")
        print("\nCannot run generation without OPENAI_API_KEY.")
        return 1
    all_ok &= _check(True, "OPENAI_API_KEY is set")

    if not all_ok:
        print("\nIntegration check aborted due to missing prerequisites.")
        return 1

    strategy = _build_strategy(output_dir=args.output_dir, top_k=args.top_k)
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
        print("Graph integration check PASSED.")
        return 0
    print("Graph integration check FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

