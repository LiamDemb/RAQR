"""Debug GraphStrategy reasoning trace for a single query."""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from raqr.entity_alias_resolver import EntityAliasResolver
from raqr.generator import SimpleLLMGenerator
from raqr.graph_store import NetworkXGraphStore
from raqr.loaders import JsonCorpusLoader
from raqr.strategies.graph import GraphStrategy, SpacyQueryEntityExtractor


def _print_list(title: str, values: list[str]) -> None:
    print(f"\n{title}")
    if not values:
        print("- (none)")
        return
    for value in values:
        print(f"- {value}")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Debug GraphStrategy reasoning trace.")
    parser.add_argument("query", help="Query to inspect.")
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "data/processed"),
        help="Directory containing corpus.jsonl, graph.pkl, entity_lexicon.parquet, alias_map.json.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.getenv("GRAPH_TOP_K", "10")),
        help="Maximum contexts to pass to generation.",
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        default=int(os.getenv("GRAPH_MAX_HOPS", "1")),
        help="Maximum graph traversal depth for relation expansion.",
    )
    parser.add_argument(
        "--show-contexts",
        type=int,
        default=5,
        help="How many retrieved contexts to print.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    corpus_path = f"{output_dir}/corpus.jsonl"
    graph_path = f"{output_dir}/graph.pkl"
    lexicon_path = f"{output_dir}/entity_lexicon.parquet"
    alias_map_path = f"{output_dir}/alias_map.json"

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
            base_prompt=(
                "Answer the question based only on the provided context. "
                "If the context does not contain the answer, say so."
            ),
        ),
        entity_extractor=SpacyQueryEntityExtractor(alias_resolver=alias_resolver),
        top_k=args.top_k,
        max_hops=args.max_hops,
        entity_df_by_norm=entity_df_by_norm,
    )

    result = strategy.retrieve_and_generate(args.query, debug=True)
    trace = result.debug_info or {}

    print("Graph RAG Debug Trace")
    print(f"- query: {args.query}")
    print(f"- status: {result.status}")
    print(f"- latency_ms: {result.latency_ms}")

    _print_list("Extracted entities", trace.get("extracted_entities", []))
    _print_list("Matched start nodes", trace.get("start_nodes", []))
    _print_list("Unmatched entities", trace.get("unmatched_entities", []))
    _print_list("Expanded entity nodes", trace.get("expanded_nodes", []))

    print("\nTraversed relation edges")
    rel_edges = trace.get("rel_edges", [])
    if not rel_edges:
        print("- (none)")
    else:
        for edge in rel_edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            label = edge.get("label", "")
            print(f"- {source} -> {target} [{label}]")

    print("\nRanked chunk trace")
    chunk_trace = trace.get("chunk_trace", [])
    if not chunk_trace:
        print("- (none)")
    else:
        for chunk in chunk_trace:
            chunk_id = chunk.get("chunk_id", "")
            score = float(chunk.get("score", 0.0))
            supporting = chunk.get("supporting_entities", [])
            support_text = ", ".join(supporting) if supporting else "(none)"
            print(f"- {chunk_id} score={score:.4f} support={support_text}")

    print("\nAnswer")
    print(result.answer or "(empty)")

    if result.context_scores:
        show_n = max(0, args.show_contexts)
        print(f"\nTop {min(show_n, len(result.context_scores))} contexts:")
        for idx, (ctx, score) in enumerate(result.context_scores[:show_n], start=1):
            print(f"\n[{idx}] score={score:.4f}")
            print(ctx[:800])

    if result.status == "ERROR":
        print("\nError")
        print(result.error)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
