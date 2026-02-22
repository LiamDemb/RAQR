"""Run a single Graph strategy query with real corpus and LLM."""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from raqr.entity_alias_resolver import EntityAliasResolver
from raqr.generator import SimpleLLMGenerator
from raqr.graph_store import NetworkXGraphStore
from raqr.loaders import JsonCorpusLoader
from raqr.strategies.graph import GraphStrategy, SpacyQueryEntityExtractor


def main() -> int:
    load_dotenv()
    output_dir = os.getenv("OUTPUT_DIR", "data/processed_rebel")
    corpus_path = f"{output_dir}/corpus.jsonl"
    graph_path = f"{output_dir}/graph.pkl"
    lexicon_path = f"{output_dir}/entity_lexicon.parquet"

    parser = argparse.ArgumentParser(description="Run one query using GraphStrategy.")
    parser.add_argument("query", help="The question to ask the GraphStrategy pipeline.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.getenv("GRAPH_TOP_K", "10")),
        help="Maximum contexts to pass to generation.",
    )
    parser.add_argument(
        "--show-contexts",
        type=int,
        default=5,
        help="How many retrieved contexts to print.",
    )
    args = parser.parse_args()

    alias_resolver = EntityAliasResolver.from_lexicon(lexicon_path=lexicon_path)
    strategy = GraphStrategy(
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
        top_k=args.top_k,
        max_hops=1,
    )

    result = strategy.retrieve_and_generate(args.query)
    print("Status:", result.status)
    if result.status == "ERROR":
        print("Error:", result.error)
    print("Answer:", result.answer)
    print("Latency (ms):", result.latency_ms)

    if result.context_scores:
        show_n = max(0, args.show_contexts)
        print(f"\nTop {min(show_n, len(result.context_scores))} contexts:")
        for idx, (ctx, score) in enumerate(result.context_scores[:show_n], start=1):
            print(f"\n[{idx}] score={score:.4f}")
            print(ctx[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

