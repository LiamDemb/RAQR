"""Run a single Graph strategy query with real corpus and LLM."""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from raqr.entity_alias_resolver import EntityAliasResolver
from raqr.generator import SimpleLLMGenerator
from raqr.graph_store import NetworkXGraphStore
from raqr.loaders import JsonCorpusLoader
from raqr.prompts import get_generator_prompt
from raqr.strategies.graph import GraphStrategy, SpacyQueryEntityExtractor


def main() -> int:
    load_dotenv()
    output_dir = os.getenv("OUTPUT_DIR", "data/processed")
    corpus_path = f"{output_dir}/corpus.jsonl"
    graph_path = f"{output_dir}/graph.pkl"
    lexicon_path = f"{output_dir}/entity_lexicon.parquet"
    alias_map_path = f"{output_dir}/alias_map.json"

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
    parser.add_argument(
        "--max-hops",
        type=int,
        default=int(os.getenv("GRAPH_MAX_HOPS", "1")),
        help="Maximum graph traversal depth for relation expansion.",
    )
    args = parser.parse_args()

    if not os.path.exists(alias_map_path):
        raise FileNotFoundError(
            f"Required artifact missing: {alias_map_path}. Rebuild corpus with Phase 1 pipeline."
        )

    alias_resolver = EntityAliasResolver.from_artifacts(output_dir=output_dir)
    entity_df_by_norm = EntityAliasResolver.load_df_map_from_lexicon(lexicon_path=lexicon_path)
    strategy = GraphStrategy(
        graph_store=NetworkXGraphStore(graph_path=graph_path),
        corpus=JsonCorpusLoader(jsonl_path=corpus_path),
        generator=SimpleLLMGenerator(
            model_id=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_prompt=get_generator_prompt(),
        ),
        entity_extractor=SpacyQueryEntityExtractor(alias_resolver=alias_resolver),
        top_k=args.top_k,
        max_hops=args.max_hops,
        entity_df_by_norm=entity_df_by_norm,
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

