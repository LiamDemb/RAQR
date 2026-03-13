"""Debug GraphStrategy reasoning trace for a single query."""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

from dotenv import load_dotenv

from raqr.embedder import SentenceTransformersEmbedder
from raqr.entity_alias_resolver import EntityAliasResolver
from raqr.entity_index_store import EntityIndexStore
from raqr.generator import SimpleLLMGenerator
from raqr.graph_store import NetworkXGraphStore
from raqr.loaders import JsonCorpusLoader
from raqr.prompts import get_generator_prompt
from raqr.strategies.graph import GraphStrategy
from raqr.scoring_config import DEFAULT_SCORING_CONFIG


def _print_list(title: str, values: List[str]) -> None:
    print(f"\n{title}")
    if not values:
        print("- (none)")
        return
    for value in values:
        print(f"- {value}")


def _print_kv_list(title: str, rows: List[Dict[str, Any]]) -> None:
    print(f"\n{title}")
    if not rows:
        print("- (none)")
        return
    for row in rows:
        pretty = ", ".join(f"{k}={v}" for k, v in row.items())
        print(f"- {pretty}")


def _format_path(path: Dict[str, Any]) -> str:
    hops = path.get("hops", []) or []
    if not hops:
        return str(path.get("start_node", ""))

    parts: List[str] = []
    first = hops[0].get("source", "")
    parts.append(first)

    for hop in hops:
        relation = hop.get("relation", "")
        if hop.get("is_reverse", False):
            relation = f"inv:{relation}"
        target = hop.get("target", "")
        parts.append(f"-[{relation}]-> {target}")

    return " ".join(parts)


def _print_candidate_paths(paths: List[Dict[str, Any]], limit: int) -> None:
    print("\nCandidate paths")
    if not paths:
        print("- (none)")
        return

    for idx, path in enumerate(paths[:limit], start=1):
        print(f"- [{idx}] {_format_path(path)}")

    remaining = len(paths) - min(len(paths), limit)
    if remaining > 0:
        print(f"- ... ({remaining} more)")


def _print_bundle_trace(bundles: List[Dict[str, Any]], limit: int) -> None:
    print("\nBundle trace")
    if not bundles:
        print("- (none)")
        return

    for idx, bundle in enumerate(bundles[:limit], start=1):
        score = float(bundle.get("score", 0.0))
        score_breakdown = bundle.get("score_breakdown", {}) or {}
        supporting_chunk_ids = bundle.get("supporting_chunk_ids", []) or []
        path = bundle.get("path", {}) or {}
        grounded_hops = bundle.get("grounded_hops", []) or []

        print(f"\n[{idx}] score={score:.4f}")
        print(f"path: {_format_path(path)}")

        if score_breakdown:
            breakdown_text = ", ".join(
                (f"{k}=[{','.join(f'{x:.4f}' for x in v)}]" if isinstance(v, list) else f"{k}={float(v):.4f}")
                for k, v in score_breakdown.items()
            )
            print(f"score_breakdown: {breakdown_text}")

        if supporting_chunk_ids:
            print(f"supporting_chunk_ids: {', '.join(supporting_chunk_ids)}")
        else:
            print("supporting_chunk_ids: (none)")

        if grounded_hops:
            print("grounded_hops:")
            for hop in grounded_hops:
                relation = hop.get("relation", "")
                if hop.get("is_reverse", False):
                    relation = f"inv:{relation}"
                print(
                    "  - "
                    f"{hop.get('source', '')} -[{relation}]-> {hop.get('target', '')} "
                    f"| chunk={hop.get('chunk_id', '')} "
                    f"| support={float(hop.get('support_score', 0.0)):.4f}"
                )
        else:
            print("grounded_hops: (none)")

    remaining = len(bundles) - min(len(bundles), limit)
    if remaining > 0:
        print(f"\n... ({remaining} more bundles)")


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Debug GraphStrategy reasoning trace.")
    parser.add_argument("query", help="Query to inspect.")
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "data/processed"),
        help="Directory containing corpus.jsonl, graph.pkl, alias_map.json, and optional entity index files.",
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
        default=int(os.getenv("GRAPH_MAX_HOPS", "2")),
        help="Maximum graph traversal depth for candidate path enumeration.",
    )
    parser.add_argument(
        "--show-contexts",
        type=int,
        default=5,
        help="How many retrieved contexts to print.",
    )
    parser.add_argument(
        "--show-paths",
        type=int,
        default=20,
        help="How many candidate paths to print.",
    )
    parser.add_argument(
        "--show-bundles",
        type=int,
        default=10,
        help="How many ranked bundles to print.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    corpus_path = f"{output_dir}/corpus.jsonl"
    graph_path = f"{output_dir}/graph.pkl"
    alias_map_path = f"{output_dir}/alias_map.json"

    if not os.path.exists(alias_map_path):
        raise FileNotFoundError(
            f"Required artifact missing: {alias_map_path}. Rebuild corpus / graph artifacts."
        )

    alias_resolver = EntityAliasResolver.from_artifacts(output_dir=output_dir)

    entity_index_store = None
    entity_index_path = f"{output_dir}/entity_index.faiss"
    entity_index_meta_path = f"{output_dir}/entity_index_meta.parquet"
    if os.path.exists(entity_index_path) and os.path.exists(entity_index_meta_path):
        entity_index_store = EntityIndexStore(
            entity_index_path,
            entity_index_meta_path,
        )

    strategy = GraphStrategy(
        graph_store=NetworkXGraphStore(graph_path=graph_path),
        corpus=JsonCorpusLoader(jsonl_path=corpus_path),
        generator=SimpleLLMGenerator(
            model_id=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_prompt=get_generator_prompt(),
        ),
        alias_resolver=alias_resolver,
        entity_index_store=entity_index_store,
        embedder=SentenceTransformersEmbedder(model_name="all-MiniLM-L6-v2"),
        top_k=args.top_k,
        max_hops=args.max_hops,
        scoring_config=DEFAULT_SCORING_CONFIG,
    )

    result = strategy.retrieve_and_generate(args.query, debug=True)
    trace = result.debug_info or {}

    print("Graph RAG Debug Trace")
    print(f"- query: {args.query}")
    print(f"- status: {result.status}")
    print(f"- latency_ms: {result.latency_ms}")

    _print_list("Extracted entities", trace.get("extracted_entities", []) or [])
    _print_list("Matched start nodes", trace.get("start_nodes", []) or [])
    _print_list("Unmatched entities", trace.get("unmatched_entities", []) or [])
    _print_kv_list("Entity index matches", trace.get("entity_index_matches", []) or [])

    grounded_bundle_count = trace.get("grounded_bundle_count")
    if grounded_bundle_count is not None:
        print(f"\nGrounded bundle count\n- {grounded_bundle_count}")

    _print_candidate_paths(trace.get("candidate_paths", []) or [], args.show_paths)
    _print_bundle_trace(trace.get("bundle_trace", []) or [], args.show_bundles)

    print("\nAnswer")
    print(result.answer or "(empty)")

    if result.context_scores:
        show_n = max(0, args.show_contexts)
        print(f"\nTop {min(show_n, len(result.context_scores))} contexts:")
        for idx, (ctx, score) in enumerate(result.context_scores[:show_n], start=1):
            print(f"\n[{idx}] score={score:.4f}")
            print(ctx[:1200])

    if result.status == "ERROR":
        print("\nError")
        print(result.error or "(unknown)")
        error_stage = trace.get("error_stage")
        if error_stage:
            print(f"error_stage: {error_stage}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
